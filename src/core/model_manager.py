"""
Model Manager Module
Handles model loading, saving, and management
"""

import os
import json
import shutil
import hashlib
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import yaml
import torch

logger = logging.getLogger(__name__)


class ModelInfo:
    """Model metadata container"""
    
    def __init__(
        self,
        name: str,
        path: str,
        model_type: str = "rvc",
        version: str = "1.0",
        description: str = "",
        author: str = "",
        created_date: str = "",
        tags: List[str] = None,
        config: Dict[str, Any] = None
    ):
        self.name = name
        self.path = path
        self.model_type = model_type
        self.version = version
        self.description = description
        self.author = author
        self.created_date = created_date or datetime.now().isoformat()
        self.tags = tags or []
        self.config = config or {}
        self.file_hash = self._calculate_hash()
        
    def _calculate_hash(self) -> str:
        """Calculate file hash for integrity checking"""
        if not os.path.exists(self.path):
            return ""
            
        sha256_hash = hashlib.sha256()
        with open(self.path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'path': self.path,
            'model_type': self.model_type,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'created_date': self.created_date,
            'tags': self.tags,
            'config': self.config,
            'file_hash': self.file_hash
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(**data)


class ModelManager:
    """Manages RVC models and their metadata"""
    
    # Default model repository URLs (example)
    DEFAULT_REPOS = {
        'official': 'https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main',
        'community': 'https://huggingface.co/spaces/RVC-Models/Community'
    }
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model manager
        
        Args:
            models_dir: Directory to store models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.models: Dict[str, ModelInfo] = {}
        
        # Create subdirectories
        self.pretrained_dir = self.models_dir / "pretrained"
        self.custom_dir = self.models_dir / "custom"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        for dir_path in [self.pretrained_dir, self.custom_dir, self.checkpoints_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Load existing metadata
        self.load_metadata()
        
        # Scan for new models
        self.scan_models_directory()
        
    def load_metadata(self) -> bool:
        """Load models metadata from file
        
        Returns:
            Success status
        """
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
                for name, info_dict in metadata.items():
                    self.models[name] = ModelInfo.from_dict(info_dict)
                    
                logger.info(f"Loaded metadata for {len(self.models)} models")
                return True
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            
        return False
        
    def save_metadata(self) -> bool:
        """Save models metadata to file
        
        Returns:
            Success status
        """
        try:
            metadata = {
                name: info.to_dict()
                for name, info in self.models.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info("Saved models metadata")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
            
    def scan_models_directory(self) -> int:
        """Scan models directory for new models
        
        Returns:
            Number of new models found
        """
        new_models = 0
        
        # Scan all subdirectories
        for subdir in [self.pretrained_dir, self.custom_dir, self.checkpoints_dir]:
            if not subdir.exists():
                continue
                
            # Look for model files
            for model_file in subdir.glob("*.pth"):
                model_name = model_file.stem
                
                # Skip if already registered
                if model_name in self.models:
                    # Verify file integrity
                    info = self.models[model_name]
                    if info.path == str(model_file) and info.file_hash:
                        current_hash = info._calculate_hash()
                        if current_hash != info.file_hash:
                            logger.warning(f"Model file changed: {model_name}")
                    continue
                    
                # Register new model
                self.register_model(
                    name=model_name,
                    path=str(model_file),
                    model_type="rvc",
                    auto_save=False
                )
                new_models += 1
                
        if new_models > 0:
            self.save_metadata()
            logger.info(f"Found {new_models} new models")
            
        return new_models
        
    def register_model(
        self,
        name: str,
        path: str,
        model_type: str = "rvc",
        version: str = "1.0",
        description: str = "",
        author: str = "",
        tags: List[str] = None,
        config: Dict[str, Any] = None,
        auto_save: bool = True
    ) -> bool:
        """Register a new model
        
        Args:
            name: Model name
            path: Path to model file
            model_type: Type of model
            version: Model version
            description: Model description
            author: Model author
            tags: Model tags
            config: Model configuration
            auto_save: Whether to save metadata immediately
            
        Returns:
            Success status
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
                
            model_info = ModelInfo(
                name=name,
                path=path,
                model_type=model_type,
                version=version,
                description=description,
                author=author,
                tags=tags,
                config=config
            )
            
            self.models[name] = model_info
            
            if auto_save:
                self.save_metadata()
                
            logger.info(f"Registered model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return False
            
    def unregister_model(self, name: str, delete_file: bool = False) -> bool:
        """Unregister a model
        
        Args:
            name: Model name
            delete_file: Whether to delete the model file
            
        Returns:
            Success status
        """
        try:
            if name not in self.models:
                logger.warning(f"Model not found: {name}")
                return False
                
            model_info = self.models[name]
            
            # Delete file if requested
            if delete_file and os.path.exists(model_info.path):
                os.remove(model_info.path)
                logger.info(f"Deleted model file: {model_info.path}")
                
            # Remove from registry
            del self.models[name]
            self.save_metadata()
            
            logger.info(f"Unregistered model: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering model: {e}")
            return False
            
    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get model information
        
        Args:
            name: Model name
            
        Returns:
            Model info or None
        """
        return self.models.get(name)
        
    def list_models(
        self,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """List available models
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags
            
        Returns:
            List of model info objects
        """
        models = list(self.models.values())
        
        # Filter by type
        if model_type:
            models = [m for m in models if m.model_type == model_type]
            
        # Filter by tags
        if tags:
            models = [
                m for m in models
                if any(tag in m.tags for tag in tags)
            ]
            
        return models
        
    def import_model(
        self,
        source_path: str,
        name: Optional[str] = None,
        model_type: str = "rvc",
        copy: bool = True,
        **kwargs
    ) -> bool:
        """Import a model from external source
        
        Args:
            source_path: Path to source model file
            name: Model name (uses filename if None)
            model_type: Type of model
            copy: Whether to copy file to models directory
            **kwargs: Additional model metadata
            
        Returns:
            Success status
        """
        try:
            source_path = Path(source_path)
            
            if not source_path.exists():
                logger.error(f"Source file not found: {source_path}")
                return False
                
            name = name or source_path.stem
            
            if copy:
                # Copy to appropriate directory
                if model_type == "pretrained":
                    dest_dir = self.pretrained_dir
                else:
                    dest_dir = self.custom_dir
                    
                dest_path = dest_dir / source_path.name
                
                # Check if file already exists
                if dest_path.exists():
                    logger.warning(f"Model file already exists: {dest_path}")
                    response = input("Overwrite? (y/n): ")
                    if response.lower() != 'y':
                        return False
                        
                shutil.copy2(source_path, dest_path)
                model_path = str(dest_path)
                
            else:
                model_path = str(source_path)
                
            # Register model
            return self.register_model(
                name=name,
                path=model_path,
                model_type=model_type,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            return False
            
    def export_model(
        self,
        name: str,
        dest_path: str,
        include_metadata: bool = True
    ) -> bool:
        """Export a model to external location
        
        Args:
            name: Model name
            dest_path: Destination path
            include_metadata: Whether to include metadata file
            
        Returns:
            Success status
        """
        try:
            if name not in self.models:
                logger.error(f"Model not found: {name}")
                return False
                
            model_info = self.models[name]
            dest_path = Path(dest_path)
            
            # Create destination directory if needed
            if dest_path.is_dir():
                dest_file = dest_path / Path(model_info.path).name
            else:
                dest_file = dest_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Copy model file
            shutil.copy2(model_info.path, dest_file)
            
            # Export metadata if requested
            if include_metadata:
                metadata_file = dest_file.with_suffix('.json')
                with open(metadata_file, 'w') as f:
                    json.dump(model_info.to_dict(), f, indent=2)
                    
            logger.info(f"Exported model to: {dest_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            return False
            
    def download_model(
        self,
        url: str,
        name: Optional[str] = None,
        model_type: str = "pretrained",
        **kwargs
    ) -> bool:
        """Download a model from URL
        
        Args:
            url: Model URL
            name: Model name
            model_type: Type of model
            **kwargs: Additional model metadata
            
        Returns:
            Success status
        """
        try:
            # Determine destination
            if model_type == "pretrained":
                dest_dir = self.pretrained_dir
            else:
                dest_dir = self.custom_dir
                
            # Extract filename from URL
            filename = url.split('/')[-1]
            if not filename.endswith('.pth'):
                filename += '.pth'
                
            name = name or Path(filename).stem
            dest_path = dest_dir / filename
            
            # Download file
            logger.info(f"Downloading model from: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"Download progress: {progress:.1f}%", end='\r')
                            
            print()  # New line after progress
            
            # Register model
            return self.register_model(
                name=name,
                path=str(dest_path),
                model_type=model_type,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
            
    def validate_model(self, name: str) -> bool:
        """Validate a model file
        
        Args:
            name: Model name
            
        Returns:
            Validation status
        """
        try:
            if name not in self.models:
                logger.error(f"Model not found: {name}")
                return False
                
            model_info = self.models[name]
            
            # Check file exists
            if not os.path.exists(model_info.path):
                logger.error(f"Model file not found: {model_info.path}")
                return False
                
            # Try loading the model
            checkpoint = torch.load(model_info.path, map_location='cpu')
            
            # Check for required keys
            required_keys = ['model_state_dict', 'config']
            for key in required_keys:
                if key not in checkpoint:
                    logger.warning(f"Missing key in model: {key}")
                    
            logger.info(f"Model validation passed: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False