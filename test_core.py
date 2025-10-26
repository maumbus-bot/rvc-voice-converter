#!/usr/bin/env python3
"""
Test script for RVC Voice Converter core functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.core import RVCEngine, AudioProcessor, ModelManager
        print("✅ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core modules: {e}")
        return False
    
    try:
        from src.training import RVCTrainer, VoiceDataset
        print("✅ Training modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import training modules: {e}")
        return False
    
    return True

def test_audio_processor():
    """Test audio processor functionality"""
    print("\nTesting AudioProcessor...")
    
    from src.core.audio_processor import AudioProcessor
    
    processor = AudioProcessor()
    
    # Test with synthetic audio
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz (A4 note)
    
    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Test pitch shifting
    pitched = processor.change_pitch(audio, 2)  # Up 2 semitones
    assert len(pitched) == len(audio), "Pitch shift changed audio length"
    print("✅ Pitch shifting works")
    
    # Test time stretching
    stretched = processor.time_stretch(audio, 1.5)  # Slower
    assert len(stretched) > len(audio), "Time stretch didn't increase length"
    print("✅ Time stretching works")
    
    # Test effects
    effected = processor.apply_effects(audio, reverb=0.3, echo=0.2)
    assert len(effected) == len(audio), "Effects changed audio length"
    print("✅ Audio effects work")
    
    return True

def test_model_manager():
    """Test model manager functionality"""
    print("\nTesting ModelManager...")
    
    from src.core.model_manager import ModelManager
    
    manager = ModelManager("test_models")
    
    # Test listing models
    models = manager.list_models()
    print(f"✅ Found {len(models)} models")
    
    # Test model registration
    success = manager.register_model(
        name="test_model",
        path="test_model.pth",
        model_type="test",
        description="Test model",
        auto_save=False  # Don't save to avoid creating files
    )
    
    if success:
        print("✅ Model registration works")
    
    # Clean up test directory
    import shutil
    if Path("test_models").exists():
        shutil.rmtree("test_models")
    
    return True

def test_rvc_engine():
    """Test RVC engine initialization"""
    print("\nTesting RVCEngine...")
    
    from src.core.rvc_engine import RVCEngine
    
    # Test initialization
    engine = RVCEngine(device='cpu')  # Use CPU for testing
    print("✅ RVC Engine initialized")
    
    # Test model structure
    from src.core.rvc_engine import RVCModel
    import torch
    
    config = {
        'input_dim': 768,
        'output_dim': 768,
        'latent_dim': 128
    }
    
    model = RVCModel(config)
    
    # Test forward pass with dummy data
    batch_size = 4
    input_tensor = torch.randn(batch_size, config['input_dim'])
    
    output, pitch = model(input_tensor)
    
    assert output.shape == (batch_size, config['output_dim']), "Output shape mismatch"
    assert pitch.shape == (batch_size, 1), "Pitch shape mismatch"
    
    print("✅ Model forward pass works")
    
    return True

def test_gui_imports():
    """Test if GUI modules can be imported"""
    print("\nTesting GUI imports...")
    
    try:
        # This might fail if PyQt6 is not installed
        from src.gui import MainWindow, ConversionTab, TrainingTab, ModelsTab
        print("✅ GUI modules imported successfully")
        return True
    except ImportError as e:
        print(f"⚠️  GUI modules not available (PyQt6 might not be installed): {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("RVC Voice Converter - Core Functionality Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_audio_processor,
        test_model_manager,
        test_rvc_engine,
        test_gui_imports
    ]
    
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed. Please check the requirements.")
    print("=" * 50)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())