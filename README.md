# RVC Voice Converter

A powerful AI-based voice conversion application using Retrieval-based Voice Conversion (RVC) technology with a user-friendly GUI interface.

## Features

- 🎤 **Real-time Voice Conversion**: Convert voices in real-time or from audio files
- 🎯 **Multiple Voice Models**: Support for loading and managing multiple RVC models
- 🎓 **Training Capability**: Train your own voice models with custom datasets
- 🖥️ **Cross-Platform GUI**: Beautiful and intuitive interface built with PyQt6
- 📦 **Standalone Executables**: Pre-built binaries for Windows and macOS
- 🎵 **Multiple Audio Formats**: Support for WAV, MP3, FLAC, and more
- ⚡ **GPU Acceleration**: CUDA support for faster processing
- 🔧 **Customizable Settings**: Fine-tune conversion parameters

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rvc-voice-converter.git
cd rvc-voice-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

### Pre-built Executables

Download the latest release for your platform:
- **Windows**: `RVC-Voice-Converter-Win64.exe`
- **macOS**: `RVC-Voice-Converter-macOS.app`

## Usage

### Basic Voice Conversion

1. Launch the application
2. Load a pre-trained RVC model from the "Models" tab
3. Select or record input audio
4. Adjust conversion parameters (pitch, formant, etc.)
5. Click "Convert" to process the audio
6. Save or play the converted audio

### Training Custom Models

1. Go to the "Training" tab
2. Prepare your dataset (WAV files, 16kHz or higher)
3. Configure training parameters:
   - Epochs: 500-1000 (recommended)
   - Batch size: 8-16 (depending on GPU memory)
   - Learning rate: 0.0001
4. Click "Start Training"
5. Monitor progress in the training console

### Model Management

- **Import Models**: File → Import Model (.pth, .onnx formats)
- **Export Models**: File → Export Model
- **Model Library**: Access community models from the built-in library

## System Requirements

### Minimum
- OS: Windows 10/11, macOS 11+
- CPU: Intel i5 or AMD Ryzen 5
- RAM: 8GB
- Storage: 2GB available space

### Recommended
- OS: Windows 11, macOS 13+
- CPU: Intel i7/i9 or AMD Ryzen 7/9
- RAM: 16GB+
- GPU: NVIDIA GTX 1060+ with CUDA support
- Storage: 10GB+ for models and datasets

## Architecture

The application is built with a modular architecture:

```
rvc-voice-converter/
├── src/
│   ├── core/          # RVC engine and audio processing
│   ├── gui/           # PyQt6 interface components
│   ├── training/      # Model training modules
│   └── utils/         # Utility functions
├── models/            # Pre-trained and custom models
├── data/              # Training datasets and temp files
└── configs/           # Configuration files
```

## Development

### Building from Source

#### Windows
```bash
python scripts/build_windows.py
```

#### macOS
```bash
python scripts/build_macos.py
```

### Running Tests
```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RVC technology based on [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- PyQt6 for the GUI framework
- PyTorch for deep learning capabilities

## Support

- Documentation: [docs.rvc-voice-converter.com](https://docs.rvc-voice-converter.com)
- Issues: [GitHub Issues](https://github.com/yourusername/rvc-voice-converter/issues)
- Discord: [Join our community](https://discord.gg/rvc-voice)

## Disclaimer

This software is for educational and research purposes only. Please respect copyright laws and obtain proper permissions when using voice conversion technology.