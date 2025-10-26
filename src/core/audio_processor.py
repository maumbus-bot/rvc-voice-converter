"""
Audio Processing Module
Handles audio I/O, preprocessing, and format conversions
"""

import numpy as np
import librosa
import soundfile as sf
import pyaudio
import wave
import threading
import queue
from typing import Optional, Tuple, Callable
from pathlib import Path
import logging
from pydub import AudioSegment
import io

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing and I/O handler"""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma']
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """Initialize audio processor
        
        Args:
            sample_rate: Target sample rate for processing
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024
        
        # PyAudio instance for real-time I/O
        self.pyaudio = None
        self.stream = None
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.recorded_frames = []
        
        # Playback state
        self.is_playing = False
        self.playback_thread = None
        
    def load_audio(self, file_path: str, normalize: bool = True) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to target format
        
        Args:
            file_path: Path to audio file
            normalize: Whether to normalize audio amplitude
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported audio format: {path.suffix}")
                
            # Load audio using librosa (handles multiple formats)
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=(self.channels == 1))
            
            # Normalize if requested
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95
                    
            logger.info(f"Loaded audio: {file_path} (shape: {audio.shape}, sr: {sr})")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
            
    def save_audio(
        self,
        audio_data: np.ndarray,
        file_path: str,
        sample_rate: Optional[int] = None,
        format: Optional[str] = None
    ) -> bool:
        """Save audio data to file
        
        Args:
            audio_data: Audio samples as numpy array
            file_path: Output file path
            sample_rate: Sample rate (uses default if None)
            format: Output format (inferred from extension if None)
            
        Returns:
            Success status
        """
        try:
            path = Path(file_path)
            sr = sample_rate or self.sample_rate
            
            # Determine format from extension
            if format is None:
                format = path.suffix.lower().lstrip('.')
                
            # Ensure audio is in correct range
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            if format == 'mp3':
                # Use pydub for MP3 encoding
                audio_segment = AudioSegment(
                    audio_data.tobytes(),
                    frame_rate=sr,
                    sample_width=audio_data.dtype.itemsize,
                    channels=1 if audio_data.ndim == 1 else audio_data.shape[1]
                )
                audio_segment.export(file_path, format='mp3', bitrate='192k')
                
            else:
                # Use soundfile for other formats
                sf.write(file_path, audio_data, sr)
                
            logger.info(f"Saved audio to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return False
            
    def resample(
        self,
        audio_data: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to different sample rate
        
        Args:
            audio_data: Input audio samples
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio_data
            
        return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        
    def change_pitch(
        self,
        audio_data: np.ndarray,
        semitones: float,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """Change pitch of audio
        
        Args:
            audio_data: Input audio samples
            semitones: Pitch shift in semitones
            sample_rate: Sample rate of audio
            
        Returns:
            Pitch-shifted audio
        """
        sr = sample_rate or self.sample_rate
        return librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=semitones)
        
    def time_stretch(
        self,
        audio_data: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """Time stretch audio without changing pitch
        
        Args:
            audio_data: Input audio samples
            rate: Stretch rate (>1 for slower, <1 for faster)
            
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio_data, rate=rate)
        
    def apply_effects(
        self,
        audio_data: np.ndarray,
        reverb: float = 0.0,
        echo: float = 0.0,
        noise_gate: float = -40.0
    ) -> np.ndarray:
        """Apply audio effects
        
        Args:
            audio_data: Input audio samples
            reverb: Reverb amount (0-1)
            echo: Echo amount (0-1)
            noise_gate: Noise gate threshold in dB
            
        Returns:
            Processed audio
        """
        processed = audio_data.copy()
        
        # Apply noise gate
        if noise_gate > -60:
            threshold = 10 ** (noise_gate / 20)
            gate_mask = np.abs(processed) > threshold
            processed = processed * gate_mask
            
        # Apply reverb (simple convolution reverb)
        if reverb > 0:
            ir_length = int(0.1 * self.sample_rate)  # 100ms impulse response
            impulse_response = np.random.randn(ir_length) * reverb * 0.1
            impulse_response[0] = 1.0
            impulse_response = impulse_response * np.exp(-np.linspace(0, 10, ir_length))
            processed = np.convolve(processed, impulse_response, mode='same')
            
        # Apply echo
        if echo > 0:
            delay_samples = int(0.3 * self.sample_rate)  # 300ms delay
            echo_signal = np.zeros_like(processed)
            echo_signal[delay_samples:] = processed[:-delay_samples] * echo * 0.5
            processed = processed + echo_signal
            
        # Normalize to prevent clipping
        max_val = np.max(np.abs(processed))
        if max_val > 0.95:
            processed = processed * (0.95 / max_val)
            
        return processed
        
    def split_audio(
        self,
        audio_data: np.ndarray,
        segment_length: float,
        overlap: float = 0.1
    ) -> list:
        """Split audio into segments
        
        Args:
            audio_data: Input audio samples
            segment_length: Length of each segment in seconds
            overlap: Overlap between segments in seconds
            
        Returns:
            List of audio segments
        """
        segment_samples = int(segment_length * self.sample_rate)
        overlap_samples = int(overlap * self.sample_rate)
        step_samples = segment_samples - overlap_samples
        
        segments = []
        for i in range(0, len(audio_data) - segment_samples + 1, step_samples):
            segments.append(audio_data[i:i + segment_samples])
            
        return segments
        
    def merge_audio(
        self,
        segments: list,
        overlap: float = 0.1,
        crossfade: bool = True
    ) -> np.ndarray:
        """Merge audio segments
        
        Args:
            segments: List of audio segments
            overlap: Overlap between segments in seconds
            crossfade: Whether to apply crossfade at segment boundaries
            
        Returns:
            Merged audio
        """
        if not segments:
            return np.array([])
            
        if len(segments) == 1:
            return segments[0]
            
        overlap_samples = int(overlap * self.sample_rate)
        
        # Calculate total length
        total_length = sum(len(seg) for seg in segments) - overlap_samples * (len(segments) - 1)
        merged = np.zeros(total_length)
        
        current_pos = 0
        for i, segment in enumerate(segments):
            segment_length = len(segment)
            
            if i == 0:
                # First segment
                merged[:segment_length] = segment
                current_pos = segment_length - overlap_samples
                
            elif i == len(segments) - 1:
                # Last segment
                if crossfade and overlap_samples > 0:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, overlap_samples)
                    fade_out = np.linspace(1, 0, overlap_samples)
                    
                    merged[current_pos:current_pos + overlap_samples] *= fade_out
                    merged[current_pos:current_pos + overlap_samples] += segment[:overlap_samples] * fade_in
                    merged[current_pos + overlap_samples:current_pos + segment_length] = segment[overlap_samples:]
                else:
                    merged[current_pos:current_pos + segment_length] = segment
                    
            else:
                # Middle segments
                if crossfade and overlap_samples > 0:
                    # Apply crossfade
                    fade_in = np.linspace(0, 1, overlap_samples)
                    fade_out = np.linspace(1, 0, overlap_samples)
                    
                    merged[current_pos:current_pos + overlap_samples] *= fade_out
                    merged[current_pos:current_pos + overlap_samples] += segment[:overlap_samples] * fade_in
                    merged[current_pos + overlap_samples:current_pos + segment_length - overlap_samples] = \
                        segment[overlap_samples:-overlap_samples]
                else:
                    merged[current_pos:current_pos + segment_length - overlap_samples] = segment[:-overlap_samples]
                    
                current_pos += segment_length - overlap_samples
                
        return merged
        
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """Start audio recording
        
        Args:
            callback: Optional callback function for real-time processing
            
        Returns:
            Success status
        """
        try:
            if self.is_recording:
                logger.warning("Already recording")
                return False
                
            self.pyaudio = pyaudio.PyAudio()
            self.recorded_frames = []
            self.is_recording = True
            
            def record_thread():
                stream = self.pyaudio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                
                while self.is_recording:
                    try:
                        data = stream.read(self.chunk_size, exception_on_overflow=False)
                        audio_chunk = np.frombuffer(data, dtype=np.float32)
                        self.recorded_frames.append(audio_chunk)
                        
                        if callback:
                            callback(audio_chunk)
                            
                    except Exception as e:
                        logger.error(f"Recording error: {e}")
                        break
                        
                stream.stop_stream()
                stream.close()
                
            self.recording_thread = threading.Thread(target=record_thread)
            self.recording_thread.start()
            
            logger.info("Recording started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False
            
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop audio recording
        
        Returns:
            Recorded audio data or None
        """
        if not self.is_recording:
            return None
            
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
            
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
            
        if self.recorded_frames:
            audio_data = np.concatenate(self.recorded_frames)
            logger.info(f"Recording stopped. Duration: {len(audio_data)/self.sample_rate:.2f} seconds")
            return audio_data
            
        return None
        
    def play_audio(
        self,
        audio_data: np.ndarray,
        callback: Optional[Callable] = None
    ) -> bool:
        """Play audio
        
        Args:
            audio_data: Audio samples to play
            callback: Optional callback for playback events
            
        Returns:
            Success status
        """
        try:
            if self.is_playing:
                logger.warning("Already playing audio")
                return False
                
            self.is_playing = True
            
            def playback_thread():
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )
                
                # Play in chunks
                for i in range(0, len(audio_data), self.chunk_size):
                    if not self.is_playing:
                        break
                        
                    chunk = audio_data[i:i + self.chunk_size]
                    if len(chunk) < self.chunk_size:
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                        
                    stream.write(chunk.astype(np.float32).tobytes())
                    
                    if callback:
                        callback(i / len(audio_data))
                        
                stream.stop_stream()
                stream.close()
                p.terminate()
                self.is_playing = False
                
            self.playback_thread = threading.Thread(target=playback_thread)
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
            
    def stop_playback(self):
        """Stop audio playback"""
        self.is_playing = False
        
        if self.playback_thread:
            self.playback_thread.join()