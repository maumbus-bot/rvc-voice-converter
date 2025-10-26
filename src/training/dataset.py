"""
Voice Dataset Module
Handles training data preparation and loading
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """Dataset for voice training"""
    
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 16000,
        segment_length: float = 2.0,
        augment: bool = False,
        cache: bool = True
    ):
        """Initialize voice dataset
        
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate
            segment_length: Length of audio segments in seconds
            augment: Whether to apply data augmentation
            cache: Whether to cache processed data
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.augment = augment
        self.cache = cache
        
        # Find all audio files
        self.audio_files = self._find_audio_files()
        
        if not self.audio_files:
            raise ValueError(f"No audio files found in {data_dir}")
            
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        # Cache for processed segments
        self.segments_cache = [] if cache else None
        
        # Process all files
        self._process_audio_files()
        
    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in data directory
        
        Returns:
            List of audio file paths
        """
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(self.data_dir.glob(f"**/*{ext}"))
            
        return sorted(audio_files)
        
    def _process_audio_files(self):
        """Process all audio files and create segments"""
        all_segments = []
        
        logger.info("Processing audio files...")
        
        for audio_file in tqdm(self.audio_files, desc="Processing"):
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                
                # Skip if too short
                if len(audio) < self.segment_samples:
                    logger.warning(f"Skipping short file: {audio_file}")
                    continue
                    
                # Split into segments
                segments = self._split_into_segments(audio)
                
                if self.cache:
                    # Extract features for each segment
                    for segment in segments:
                        features = self._extract_features(segment)
                        all_segments.append((features, features))  # Using same as input/target for now
                        
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                continue
                
        if self.cache:
            self.segments_cache = all_segments
            
        self.num_segments = len(all_segments) if self.cache else self._count_segments()
        logger.info(f"Total segments: {self.num_segments}")
        
    def _count_segments(self) -> int:
        """Count total number of segments without caching
        
        Returns:
            Total segment count
        """
        count = 0
        for audio_file in self.audio_files:
            try:
                audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                count += len(audio) // self.segment_samples
            except:
                continue
        return count
        
    def _split_into_segments(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into fixed-length segments
        
        Args:
            audio: Audio samples
            
        Returns:
            List of audio segments
        """
        segments = []
        
        for i in range(0, len(audio) - self.segment_samples + 1, self.segment_samples // 2):
            segment = audio[i:i + self.segment_samples]
            
            # Apply augmentation if enabled
            if self.augment:
                segment = self._augment_segment(segment)
                
            segments.append(segment)
            
        return segments
        
    def _augment_segment(self, segment: np.ndarray) -> np.ndarray:
        """Apply data augmentation to segment
        
        Args:
            segment: Audio segment
            
        Returns:
            Augmented segment
        """
        # Random pitch shift
        if random.random() < 0.3:
            pitch_shift = random.uniform(-2, 2)
            segment = librosa.effects.pitch_shift(segment, sr=self.sample_rate, n_steps=pitch_shift)
            
        # Random time stretch
        if random.random() < 0.3:
            stretch_rate = random.uniform(0.9, 1.1)
            segment = librosa.effects.time_stretch(segment, rate=stretch_rate)
            # Pad or trim to original length
            if len(segment) > self.segment_samples:
                segment = segment[:self.segment_samples]
            else:
                segment = np.pad(segment, (0, self.segment_samples - len(segment)))
                
        # Random noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.005, len(segment))
            segment = segment + noise
            
        # Random volume
        if random.random() < 0.3:
            volume = random.uniform(0.8, 1.2)
            segment = segment * volume
            
        return segment
        
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio segment
        
        Args:
            audio: Audio segment
            
        Returns:
            Feature vector
        """
        # Compute spectral features
        stft = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        magnitude = np.abs(stft)
        
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(S=magnitude**2, sr=self.sample_rate, n_mels=128)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude)
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Combine features
        features = np.vstack([
            mfccs,
            spectral_centroid,
            spectral_rolloff,
            spectral_contrast,
            zero_crossing_rate
        ])
        
        # Flatten and standardize size
        features_flat = features.flatten()
        target_size = 768  # Match model input
        
        if len(features_flat) > target_size:
            features_flat = features_flat[:target_size]
        else:
            features_flat = np.pad(features_flat, (0, target_size - len(features_flat)))
            
        return features_flat
        
    def __len__(self) -> int:
        """Get dataset length
        
        Returns:
            Number of samples
        """
        return self.num_segments
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (input_features, target_features)
        """
        if self.cache:
            # Return cached item
            features_in, features_out = self.segments_cache[idx]
            
        else:
            # Load and process on the fly
            # This is simplified - in practice you'd want more efficient indexing
            segment_idx = 0
            for audio_file in self.audio_files:
                try:
                    audio, _ = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                    segments = self._split_into_segments(audio)
                    
                    if segment_idx + len(segments) > idx:
                        local_idx = idx - segment_idx
                        segment = segments[local_idx]
                        features_in = self._extract_features(segment)
                        features_out = features_in  # Using same for now
                        break
                        
                    segment_idx += len(segments)
                    
                except Exception as e:
                    logger.error(f"Error loading {audio_file}: {e}")
                    continue
                    
        return torch.FloatTensor(features_in), torch.FloatTensor(features_out)