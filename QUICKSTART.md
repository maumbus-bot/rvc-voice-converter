# RVC Voice Converter - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended for training)
- NVIDIA GPU with CUDA support (optional but recommended)

### Installation

#### 1. Clone or Download
```bash
git clone https://github.com/yourusername/rvc-voice-converter.git
cd rvc-voice-converter
```

#### 2. Install Dependencies

**Windows:**
```bash
run_dev.bat
```

**macOS/Linux:**
```bash
chmod +x run_dev.sh
./run_dev.sh
```

**Manual Installation:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## 🎯 Basic Usage

### Voice Conversion
1. **Launch the application**
2. **Load a model** from the Models tab or use the default model
3. **Select input audio** (Browse or drag-and-drop)
4. **Adjust parameters:**
   - Pitch Shift: Change voice pitch (-12 to +12 semitones)
   - Formant Shift: Adjust voice timbre
   - Index Rate: Control feature similarity (0-100%)
5. **Click "Convert Voice"**
6. **Save or play** the converted audio

### Training Custom Models
1. **Prepare your dataset:**
   - Collect 10-30 minutes of clean audio
   - Save as WAV files (16kHz or higher)
   - Place in a single folder
2. **Go to Training tab**
3. **Select dataset folder**
4. **Configure training:**
   - Model Name: Your custom name
   - Epochs: 500-1000 (more = better quality)
   - Batch Size: 8-16 (depending on GPU memory)
5. **Start Training**
6. **Monitor progress** in the training log

### Model Management
- **Import Models:** File → Import Model or Models tab
- **Download Models:** Models tab → Download from Hub
- **Export Models:** Select model → Export

## 🎮 Keyboard Shortcuts
- `Ctrl+N`: New project
- `Ctrl+O`: Open project
- `Ctrl+S`: Save project
- `Ctrl+Q`: Quit application
- `Ctrl+,`: Settings

## 🛠️ Building Executables

### Windows
```bash
cd scripts
python build_windows.py
```
Output: `dist/RVC-Voice-Converter.exe`

### macOS
```bash
cd scripts
python build_macos.py
```
Output: `dist/RVC Voice Converter.app`

## 📝 Tips & Tricks

### Best Practices for Voice Conversion
1. **Use high-quality input audio** (clear, no background noise)
2. **Match the pitch range** of your target voice
3. **Start with default settings** then fine-tune
4. **Use Index Rate 75%** for balanced conversion
5. **Enable "Protect Voiceless Consonants"** for clearer speech

### Training Tips
1. **Dataset Quality > Quantity** - Clean audio is crucial
2. **Consistent recording conditions** - Same mic, room, distance
3. **Remove silence and noise** before training
4. **Start with 500 epochs** and increase if needed
5. **Monitor validation loss** to avoid overfitting

### Performance Optimization
- **Enable GPU Acceleration** in settings
- **Reduce batch size** if running out of memory
- **Close other applications** during training
- **Use SSD storage** for datasets and models

## 🔧 Troubleshooting

### Common Issues

**"No model loaded" error:**
- Go to Models tab and click "Load" on any model
- Or import your own model file

**Audio playback not working:**
- Check audio device settings
- Try different output format (WAV/MP3)
- Restart the application

**Training fails to start:**
- Verify dataset contains valid audio files
- Check available disk space
- Reduce batch size if GPU memory error

**Conversion sounds robotic:**
- Reduce Index Rate to 50-60%
- Adjust Formant Shift slightly
- Try different filter radius (2-5)

## 📚 Advanced Features

### Real-Time Mode
1. Click "Real-Time Mode" button
2. Select input device (microphone)
3. Adjust latency settings if needed
4. Speak into microphone for live conversion

### Batch Processing
1. Tools → Batch Processing
2. Select multiple input files
3. Configure conversion settings
4. Process all files at once

### API Integration
The core engine can be used programmatically:

```python
from src.core import RVCEngine, AudioProcessor

# Initialize engine
engine = RVCEngine(device='cuda')  # or 'cpu'

# Load model
engine.load_model('path/to/model.pth')

# Convert voice
engine.convert_voice(
    'input.wav',
    'output.wav',
    pitch_shift=0,
    formant_shift=0
)
```

## 🆘 Support

- **Documentation:** [GitHub Wiki](https://github.com/yourusername/rvc-voice-converter/wiki)
- **Issues:** [GitHub Issues](https://github.com/yourusername/rvc-voice-converter/issues)
- **Discord:** [Join our community](https://discord.gg/rvc-voice)

## ⚠️ Legal Notice

This software is for educational and research purposes. Always respect copyright laws and obtain proper permissions when using voice conversion technology. Do not use this software to impersonate others without consent.