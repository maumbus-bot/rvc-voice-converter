"""
RVC Engine - Core voice conversion implementation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import soundfile as sf
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RVCModel(nn.Module):
    """RVC Neural Network Model"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(config.get('input_dim', 768), 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(256, config.get('latent_dim', 128)),
            nn.Tanh()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(config.get('latent_dim', 128), 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, config.get('output_dim', 768))
        )
        
        # Pitch predictor
        self.pitch_predictor = nn.Sequential(
            nn.Linear(config.get('latent_dim', 128), 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x: torch.Tensor, pitch_shift: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode
        encoded = self.encoder(x)
        latent = self.bottleneck(encoded)
        
        # Apply pitch shift
        if pitch_shift != 0:
            pitch_factor = torch.tensor([pitch_shift]).to(x.device)
            latent = latent + pitch_factor.unsqueeze(0).expand_as(latent) * 0.1
        
        # Decode
        decoded = self.decoder(latent)
        pitch = self.pitch_predictor(latent)
        
        return decoded, pitch


class RVCEngine:
    """Main RVC processing engine"""
    
    def __init__(self, device: str = 'auto'):
        """Initialize RVC engine
        
        Args:
            device: 'cuda', 'cpu', or 'auto' for automatic detection
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"RVC Engine initialized with device: {self.device}")
        
        self.models = {}
        self.current_model = None
        self.sample_rate = 16000
        self.hop_length = 512
        self.win_length = 2048
        self.n_fft = 2048
        
    def load_model(self, model_path: str, model_name: str = None) -> bool:
        """Load an RVC model from file
        
        Args:
            model_path: Path to model file (.pth or .onnx)
            model_name: Optional name for the model
            
        Returns:
            Success status
        """
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
                
            model_name = model_name or path.stem
            
            if path.suffix == '.pth':
                # Load PyTorch model
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extract config and create model
                config = checkpoint.get('config', {
                    'input_dim': 768,
                    'output_dim': 768,
                    'latent_dim': 128
                })
                
                model = RVCModel(config)
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.to(self.device)
                model.eval()
                
                self.models[model_name] = {
                    'model': model,
                    'config': config,
                    'path': model_path
                }
                self.current_model = model_name
                
                logger.info(f"Successfully loaded model: {model_name}")
                return True
                
            elif path.suffix == '.onnx':
                # TODO: Implement ONNX model loading
                logger.warning("ONNX model loading not yet implemented")
                return False
                
            else:
                logger.error(f"Unsupported model format: {path.suffix}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
            
    def save_model(self, model_name: str, save_path: str) -> bool:
        """Save a model to file
        
        Args:
            model_name: Name of the model to save
            save_path: Path to save the model
            
        Returns:
            Success status
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model not found: {model_name}")
                return False
                
            model_data = self.models[model_name]
            torch.save({
                'model_state_dict': model_data['model'].state_dict(),
                'config': model_data['config']
            }, save_path)
            
            logger.info(f"Model saved to: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
            
    def convert_voice(
        self,
        audio_path: str,
        output_path: str,
        pitch_shift: float = 0.0,
        formant_shift: float = 0.0,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        rms_mix_rate: float = 0.25,
        protect_voiceless: float = 0.33
    ) -> bool:
        """Convert voice using current model
        
        Args:
            audio_path: Input audio file path
            output_path: Output audio file path
            pitch_shift: Pitch shift in semitones (-12 to 12)
            formant_shift: Formant shift ratio (0.8 to 1.2)
            index_rate: Feature retrieval index rate (0 to 1)
            filter_radius: Median filter radius for pitch smoothing
            rms_mix_rate: RMS mix rate for volume envelope
            protect_voiceless: Protection for voiceless consonants (0 to 0.5)
            
        Returns:
            Success status
        """
        try:
            if not self.current_model:
                logger.error("No model loaded")
                return False
                
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = self._extract_features(audio)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Apply model
            model = self.models[self.current_model]['model']
            with torch.no_grad():
                converted_features, pitch = model(features_tensor, pitch_shift)
                
            # Convert back to audio
            converted_audio = self._features_to_audio(
                converted_features.squeeze(0).cpu().numpy(),
                formant_shift,
                filter_radius,
                rms_mix_rate,
                protect_voiceless
            )
            
            # Apply post-processing
            converted_audio = self._post_process_audio(converted_audio, audio, rms_mix_rate)
            
            # Save output
            sf.write(output_path, converted_audio, self.sample_rate)
            
            logger.info(f"Voice conversion completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during voice conversion: {e}")
            return False
            
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features for RVC processing"""
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        magnitude = np.abs(D)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            S=magnitude**2,
            sr=self.sample_rate,
            n_mels=128
        )
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            S=librosa.power_to_db(mel_spec),
            n_mfcc=13
        )
        
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude)
        
        # Combine features
        features = np.vstack([
            mfccs,
            spectral_centroid,
            spectral_rolloff,
            spectral_contrast
        ])
        
        # Flatten and pad/trim to fixed size
        features_flat = features.flatten()
        target_size = 768  # Match model input dimension
        
        if len(features_flat) > target_size:
            features_flat = features_flat[:target_size]
        else:
            features_flat = np.pad(features_flat, (0, target_size - len(features_flat)))
            
        return features_flat
        
    def _features_to_audio(
        self,
        features: np.ndarray,
        formant_shift: float,
        filter_radius: int,
        rms_mix_rate: float,
        protect_voiceless: float
    ) -> np.ndarray:
        """Convert features back to audio signal"""
        # This is a simplified version - real RVC uses more sophisticated methods
        # Generate phase information
        phase = np.random.uniform(-np.pi, np.pi, (self.n_fft // 2 + 1, len(features) // 128))
        
        # Reshape features to spectrogram-like shape
        magnitude = np.abs(features.reshape(-1, 128).T)
        
        # Interpolate to full frequency resolution
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, magnitude.shape[0]), magnitude, axis=0, kind='cubic')
        magnitude_full = f(np.linspace(0, 1, self.n_fft // 2 + 1))
        
        # Apply formant shift
        if formant_shift != 0:
            shift_bins = int(formant_shift * magnitude_full.shape[0] * 0.1)
            magnitude_full = np.roll(magnitude_full, shift_bins, axis=0)
            
        # Reconstruct complex spectrogram
        D_reconstructed = magnitude_full * np.exp(1j * phase)
        
        # Apply median filter for smoothing
        if filter_radius > 0:
            from scipy.signal import medfilt
            magnitude_full = medfilt(magnitude_full, kernel_size=(filter_radius * 2 + 1, 1))
            
        # Inverse STFT
        audio_reconstructed = librosa.istft(
            D_reconstructed,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        return audio_reconstructed
        
    def _post_process_audio(
        self,
        converted_audio: np.ndarray,
        original_audio: np.ndarray,
        rms_mix_rate: float
    ) -> np.ndarray:
        """Apply post-processing to converted audio"""
        # Match length
        target_length = len(original_audio)
        if len(converted_audio) > target_length:
            converted_audio = converted_audio[:target_length]
        else:
            converted_audio = np.pad(converted_audio, (0, target_length - len(converted_audio)))
            
        # Apply RMS mixing
        rms_original = np.sqrt(np.mean(original_audio**2))
        rms_converted = np.sqrt(np.mean(converted_audio**2))
        
        if rms_converted > 0:
            # Adjust converted audio RMS to match original
            rms_target = rms_original * (1 - rms_mix_rate) + rms_converted * rms_mix_rate
            converted_audio = converted_audio * (rms_target / rms_converted)
            
        # Apply fade in/out to avoid clicks
        fade_length = int(0.01 * self.sample_rate)  # 10ms fade
        converted_audio[:fade_length] *= np.linspace(0, 1, fade_length)
        converted_audio[-fade_length:] *= np.linspace(1, 0, fade_length)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(converted_audio))
        if max_val > 0.95:
            converted_audio = converted_audio * (0.95 / max_val)
            
        return converted_audio
        
    def real_time_convert(
        self,
        input_callback,
        output_callback,
        pitch_shift: float = 0.0,
        **kwargs
    ):
        """Real-time voice conversion
        
        Args:
            input_callback: Function to get input audio chunks
            output_callback: Function to send converted audio chunks
            pitch_shift: Real-time pitch shift
            **kwargs: Additional conversion parameters
        """
        # TODO: Implement real-time processing with streaming
        pass