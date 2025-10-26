"""
RVC Model Trainer
Handles training of custom voice conversion models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class RVCTrainer:
    """Trainer for RVC models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'auto',
        checkpoint_dir: str = "models/checkpoints"
    ):
        """Initialize trainer
        
        Args:
            model: RVC model to train
            device: Training device ('cuda', 'cpu', or 'auto')
            checkpoint_dir: Directory to save checkpoints
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Default training config
        self.config = {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'epochs': 500,
            'gradient_clip': 1.0,
            'weight_decay': 1e-5,
            'warmup_steps': 1000,
            'save_interval': 10,
            'validation_split': 0.1,
            'early_stopping_patience': 50,
            'reduce_lr_patience': 20,
            'min_lr': 1e-6
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
        
    def configure_training(self, config: Dict[str, Any]):
        """Update training configuration
        
        Args:
            config: Training configuration dictionary
        """
        self.config.update(config)
        logger.info(f"Training configuration updated: {config}")
        
    def prepare_optimizer(self) -> torch.optim.Optimizer:
        """Prepare optimizer for training
        
        Returns:
            Configured optimizer
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
        
    def prepare_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Prepare learning rate scheduler
        
        Args:
            optimizer: Optimizer instance
            num_training_steps: Total number of training steps
            
        Returns:
            Configured scheduler
        """
        # Cosine annealing with warmup
        def lr_lambda(current_step: int):
            warmup_steps = self.config['warmup_steps']
            
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
                
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return scheduler
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        progress_callback: Optional[Callable] = None
    ) -> float:
        """Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            scheduler: Optional learning rate scheduler
            progress_callback: Optional callback for progress updates
            
        Returns:
            Average epoch loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Add regularization
            l2_reg = 0.0
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss = loss + self.config['weight_decay'] * l2_reg
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )
            
            # Optimizer step
            optimizer.step()
            
            # Scheduler step
            if scheduler:
                scheduler.step()
                
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Progress callback
            if progress_callback:
                progress_callback(
                    epoch=self.current_epoch,
                    batch=batch_idx,
                    total_batches=num_batches,
                    loss=loss.item()
                )
                
        avg_loss = epoch_loss / num_batches
        return avg_loss
        
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Validate model
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        avg_loss = val_loss / num_batches
        return avg_loss
        
    def train(
        self,
        train_dataset,
        val_dataset=None,
        resume_from: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Full training loop
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            resume_from: Optional checkpoint path to resume from
            progress_callback: Optional callback for progress updates
            
        Returns:
            Training results dictionary
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
        # Prepare optimizer and scheduler
        optimizer = self.prepare_optimizer()
        
        num_training_steps = len(train_loader) * self.config['epochs']
        scheduler = self.prepare_scheduler(optimizer, num_training_steps)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from, optimizer, scheduler)
            
        # Early stopping
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Training loop
        logger.info(f"Starting training for {self.config['epochs']} epochs")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                scheduler,
                progress_callback
            )
            
            self.training_history['loss'].append(train_loss)
            self.training_history['learning_rate'].append(
                optimizer.param_groups[0]['lr']
            )
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        optimizer,
                        scheduler,
                        is_best=True
                    )
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info("Early stopping triggered")
                    break
                    
            else:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
                
            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(optimizer, scheduler)
                
        # Save final model
        self.save_checkpoint(optimizer, scheduler, is_final=True)
        
        # Plot training history
        self.plot_training_history()
        
        # Return results
        results = {
            'final_epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'model_path': str(self.checkpoint_dir / 'best_model.pth')
        }
        
        logger.info("Training completed")
        return results
        
    def save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save training checkpoint
        
        Args:
            optimizer: Optimizer state to save
            scheduler: Optional scheduler state to save
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
        # Determine filename
        if is_best:
            filename = 'best_model.pth'
        elif is_final:
            filename = 'final_model.pth'
        else:
            filename = f'checkpoint_epoch_{self.current_epoch}.pth'
            
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
        
    def load_checkpoint(
        self,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        """Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'loss': [],
            'val_loss': [],
            'learning_rate': []
        })
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        
    def plot_training_history(self):
        """Plot and save training history"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot losses
            epochs = range(len(self.training_history['loss']))
            ax1.plot(epochs, self.training_history['loss'], label='Train Loss')
            
            if self.training_history['val_loss']:
                ax1.plot(epochs, self.training_history['val_loss'], label='Val Loss')
                
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot learning rate
            ax2.plot(epochs, self.training_history['learning_rate'])
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(self.checkpoint_dir / 'training_history.png')
            plt.close()
            
            logger.info("Training history plot saved")
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")