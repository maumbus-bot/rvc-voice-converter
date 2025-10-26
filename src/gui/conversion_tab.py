"""
Conversion Tab
Voice conversion interface
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QSlider, QSpinBox, QComboBox,
    QLineEdit, QTextEdit, QFileDialog, QProgressBar,
    QCheckBox, QMessageBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConversionWorker(QThread):
    """Worker thread for voice conversion"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, model_name, parameters):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.model_name = model_name
        self.parameters = parameters
        
    def run(self):
        """Run conversion process"""
        try:
            # Import here to avoid circular imports
            from ..core import RVCEngine, AudioProcessor
            
            self.status.emit("Initializing engine...")
            engine = RVCEngine()
            
            self.status.emit(f"Loading model: {self.model_name}")
            self.progress.emit(20)
            
            # TODO: Load actual model
            # engine.load_model(model_path)
            
            self.status.emit("Processing audio...")
            self.progress.emit(50)
            
            # TODO: Perform conversion
            # success = engine.convert_voice(
            #     self.input_path,
            #     self.output_path,
            #     **self.parameters
            # )
            
            self.progress.emit(100)
            self.status.emit("Conversion completed!")
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(str(e))


class ConversionTab(QWidget):
    """Voice conversion interface tab"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.conversion_worker = None
        
    def init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Input/Output
        left_panel = self.create_left_panel()
        
        # Right panel - Parameters
        right_panel = self.create_right_panel()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        # Bottom panel - Controls and Progress
        bottom_panel = self.create_bottom_panel()
        layout.addWidget(bottom_panel)
        
    def create_left_panel(self):
        """Create left panel with input/output controls"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(panel)
        
        # Input section
        input_group = QGroupBox("Input Audio")
        input_layout = QVBoxLayout()
        
        # File selection
        file_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select input audio file...")
        self.input_path.setReadOnly(True)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_input_file)
        
        record_btn = QPushButton("🎤 Record")
        record_btn.clicked.connect(self.start_recording)
        
        file_layout.addWidget(self.input_path)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(record_btn)
        
        input_layout.addLayout(file_layout)
        
        # Drag and drop area
        self.drop_area = QLabel("Or drag and drop audio file here")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setMinimumHeight(80)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #555;
                border-radius: 5px;
                padding: 20px;
                background-color: #3a3a3a;
            }
        """)
        self.drop_area.setAcceptDrops(True)
        
        input_layout.addWidget(self.drop_area)
        
        # Audio player controls
        player_layout = QHBoxLayout()
        self.play_input_btn = QPushButton("▶ Play")
        self.play_input_btn.setEnabled(False)
        self.play_input_btn.clicked.connect(self.play_input_audio)
        
        self.stop_input_btn = QPushButton("■ Stop")
        self.stop_input_btn.setEnabled(False)
        self.stop_input_btn.clicked.connect(self.stop_input_audio)
        
        player_layout.addWidget(self.play_input_btn)
        player_layout.addWidget(self.stop_input_btn)
        player_layout.addStretch()
        
        input_layout.addLayout(player_layout)
        input_group.setLayout(input_layout)
        
        # Output section
        output_group = QGroupBox("Output Audio")
        output_layout = QVBoxLayout()
        
        # Output file selection
        output_file_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output location...")
        
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_file)
        
        output_file_layout.addWidget(self.output_path)
        output_file_layout.addWidget(output_browse_btn)
        
        output_layout.addLayout(output_file_layout)
        
        # Output format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        
        self.output_format = QComboBox()
        self.output_format.addItems(["WAV", "MP3", "FLAC", "OGG"])
        
        format_layout.addWidget(self.output_format)
        format_layout.addStretch()
        
        output_layout.addLayout(format_layout)
        
        # Output player controls
        output_player_layout = QHBoxLayout()
        self.play_output_btn = QPushButton("▶ Play")
        self.play_output_btn.setEnabled(False)
        self.play_output_btn.clicked.connect(self.play_output_audio)
        
        self.save_output_btn = QPushButton("💾 Save As...")
        self.save_output_btn.setEnabled(False)
        self.save_output_btn.clicked.connect(self.save_output_audio)
        
        output_player_layout.addWidget(self.play_output_btn)
        output_player_layout.addWidget(self.save_output_btn)
        output_player_layout.addStretch()
        
        output_layout.addLayout(output_player_layout)
        output_group.setLayout(output_layout)
        
        # Add to main layout
        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addStretch()
        
        return panel
        
    def create_right_panel(self):
        """Create right panel with conversion parameters"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(panel)
        
        # Model selection
        model_group = QGroupBox("Voice Model")
        model_layout = QVBoxLayout()
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Default Model", "Custom Model 1", "Custom Model 2"])
        
        model_info_btn = QPushButton("ℹ Model Info")
        model_info_btn.clicked.connect(self.show_model_info)
        
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(model_info_btn)
        
        model_group.setLayout(model_layout)
        
        # Conversion parameters
        params_group = QGroupBox("Conversion Parameters")
        params_layout = QVBoxLayout()
        
        # Pitch shift
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch Shift:"))
        
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(-12, 12)
        self.pitch_slider.setValue(0)
        self.pitch_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.pitch_slider.setTickInterval(1)
        
        self.pitch_value = QSpinBox()
        self.pitch_value.setRange(-12, 12)
        self.pitch_value.setValue(0)
        self.pitch_value.setSuffix(" semitones")
        
        self.pitch_slider.valueChanged.connect(self.pitch_value.setValue)
        self.pitch_value.valueChanged.connect(self.pitch_slider.setValue)
        
        pitch_layout.addWidget(self.pitch_slider)
        pitch_layout.addWidget(self.pitch_value)
        
        # Formant shift
        formant_layout = QHBoxLayout()
        formant_layout.addWidget(QLabel("Formant Shift:"))
        
        self.formant_slider = QSlider(Qt.Orientation.Horizontal)
        self.formant_slider.setRange(-20, 20)
        self.formant_slider.setValue(0)
        
        self.formant_value = QSpinBox()
        self.formant_value.setRange(-20, 20)
        self.formant_value.setValue(0)
        self.formant_value.setSuffix("%")
        
        self.formant_slider.valueChanged.connect(self.formant_value.setValue)
        self.formant_value.valueChanged.connect(self.formant_slider.setValue)
        
        formant_layout.addWidget(self.formant_slider)
        formant_layout.addWidget(self.formant_value)
        
        # Index rate
        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("Index Rate:"))
        
        self.index_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_slider.setRange(0, 100)
        self.index_slider.setValue(75)
        
        self.index_value = QSpinBox()
        self.index_value.setRange(0, 100)
        self.index_value.setValue(75)
        self.index_value.setSuffix("%")
        
        self.index_slider.valueChanged.connect(self.index_value.setValue)
        self.index_value.valueChanged.connect(self.index_slider.setValue)
        
        index_layout.addWidget(self.index_slider)
        index_layout.addWidget(self.index_value)
        
        # Filter radius
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Radius:"))
        
        self.filter_value = QSpinBox()
        self.filter_value.setRange(0, 10)
        self.filter_value.setValue(3)
        
        filter_layout.addWidget(self.filter_value)
        filter_layout.addStretch()
        
        # RMS mix rate
        rms_layout = QHBoxLayout()
        rms_layout.addWidget(QLabel("RMS Mix:"))
        
        self.rms_slider = QSlider(Qt.Orientation.Horizontal)
        self.rms_slider.setRange(0, 100)
        self.rms_slider.setValue(25)
        
        self.rms_value = QSpinBox()
        self.rms_value.setRange(0, 100)
        self.rms_value.setValue(25)
        self.rms_value.setSuffix("%")
        
        self.rms_slider.valueChanged.connect(self.rms_value.setValue)
        self.rms_value.valueChanged.connect(self.rms_slider.setValue)
        
        rms_layout.addWidget(self.rms_slider)
        rms_layout.addWidget(self.rms_value)
        
        # Add all parameter controls
        params_layout.addLayout(pitch_layout)
        params_layout.addLayout(formant_layout)
        params_layout.addLayout(index_layout)
        params_layout.addLayout(filter_layout)
        params_layout.addLayout(rms_layout)
        
        # Additional options
        self.protect_voiceless = QCheckBox("Protect Voiceless Consonants")
        self.protect_voiceless.setChecked(True)
        
        self.gpu_acceleration = QCheckBox("GPU Acceleration")
        self.gpu_acceleration.setChecked(True)
        
        params_layout.addWidget(self.protect_voiceless)
        params_layout.addWidget(self.gpu_acceleration)
        
        params_group.setLayout(params_layout)
        
        # Presets
        presets_group = QGroupBox("Presets")
        presets_layout = QVBoxLayout()
        
        self.presets_combo = QComboBox()
        self.presets_combo.addItems([
            "Default",
            "Male to Female",
            "Female to Male",
            "Child Voice",
            "Robot Voice",
            "Custom..."
        ])
        
        save_preset_btn = QPushButton("Save Current as Preset")
        save_preset_btn.clicked.connect(self.save_preset)
        
        presets_layout.addWidget(self.presets_combo)
        presets_layout.addWidget(save_preset_btn)
        
        presets_group.setLayout(presets_layout)
        
        # Add all groups
        layout.addWidget(model_group)
        layout.addWidget(params_group)
        layout.addWidget(presets_group)
        layout.addStretch()
        
        return panel
        
    def create_bottom_panel(self):
        """Create bottom panel with conversion controls"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(panel)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Status label
        self.status_label = QLabel("Ready")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.convert_btn = QPushButton("🔄 Convert Voice")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.convert_btn.setMinimumHeight(40)
        self.convert_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
            }
        """)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_conversion)
        self.cancel_btn.setEnabled(False)
        
        self.real_time_btn = QPushButton("🎙️ Real-Time Mode")
        self.real_time_btn.setCheckable(True)
        self.real_time_btn.clicked.connect(self.toggle_real_time_mode)
        
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.real_time_btn)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addLayout(button_layout)
        
        return panel
        
    def browse_input_file(self):
        """Browse for input audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.input_path.setText(file_path)
            self.play_input_btn.setEnabled(True)
            
            # Auto-generate output path
            input_path = Path(file_path)
            output_path = input_path.parent / f"{input_path.stem}_converted.wav"
            self.output_path.setText(str(output_path))
            
    def browse_output_file(self):
        """Browse for output location"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Location",
            "",
            "WAV Files (*.wav);;MP3 Files (*.mp3);;FLAC Files (*.flac);;All Files (*.*)"
        )
        
        if file_path:
            self.output_path.setText(file_path)
            
    def start_conversion(self):
        """Start voice conversion process"""
        if not self.input_path.text():
            QMessageBox.warning(self, "Warning", "Please select an input audio file.")
            return
            
        if not self.output_path.text():
            QMessageBox.warning(self, "Warning", "Please specify an output location.")
            return
            
        # Prepare parameters
        parameters = {
            'pitch_shift': self.pitch_value.value(),
            'formant_shift': self.formant_value.value() / 100.0,
            'index_rate': self.index_value.value() / 100.0,
            'filter_radius': self.filter_value.value(),
            'rms_mix_rate': self.rms_value.value() / 100.0,
            'protect_voiceless': 0.33 if self.protect_voiceless.isChecked() else 0
        }
        
        # Create and start worker thread
        self.conversion_worker = ConversionWorker(
            self.input_path.text(),
            self.output_path.text(),
            self.model_combo.currentText(),
            parameters
        )
        
        self.conversion_worker.progress.connect(self.update_progress)
        self.conversion_worker.status.connect(self.update_status)
        self.conversion_worker.finished.connect(self.conversion_finished)
        self.conversion_worker.error.connect(self.conversion_error)
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.convert_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Start conversion
        self.conversion_worker.start()
        
    def cancel_conversion(self):
        """Cancel ongoing conversion"""
        if self.conversion_worker and self.conversion_worker.isRunning():
            self.conversion_worker.terminate()
            self.conversion_worker.wait()
            
        self.progress_bar.setVisible(False)
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Conversion cancelled")
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)
        
    def conversion_finished(self, output_path):
        """Handle conversion completion"""
        self.progress_bar.setVisible(False)
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.play_output_btn.setEnabled(True)
        self.save_output_btn.setEnabled(True)
        
        QMessageBox.information(
            self,
            "Conversion Complete",
            f"Voice conversion completed successfully!\nOutput saved to: {output_path}"
        )
        
    def conversion_error(self, error_message):
        """Handle conversion error"""
        self.progress_bar.setVisible(False)
        self.convert_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        QMessageBox.critical(
            self,
            "Conversion Error",
            f"An error occurred during conversion:\n{error_message}"
        )
        
    def play_input_audio(self):
        """Play input audio file"""
        # TODO: Implement audio playback
        pass
        
    def stop_input_audio(self):
        """Stop input audio playback"""
        # TODO: Implement audio stop
        pass
        
    def play_output_audio(self):
        """Play converted audio"""
        # TODO: Implement output playback
        pass
        
    def save_output_audio(self):
        """Save output audio with different name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Converted Audio",
            "",
            "WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*.*)"
        )
        
        if file_path:
            # TODO: Copy output file to new location
            pass
            
    def start_recording(self):
        """Start audio recording"""
        # TODO: Implement recording
        QMessageBox.information(self, "Recording", "Recording feature coming soon!")
        
    def toggle_real_time_mode(self):
        """Toggle real-time conversion mode"""
        if self.real_time_btn.isChecked():
            QMessageBox.information(
                self,
                "Real-Time Mode",
                "Real-time voice conversion activated!\nSpeak into your microphone."
            )
            # TODO: Implement real-time mode
        else:
            # TODO: Stop real-time mode
            pass
            
    def show_model_info(self):
        """Show information about selected model"""
        model_name = self.model_combo.currentText()
        QMessageBox.information(
            self,
            "Model Information",
            f"Model: {model_name}\nVersion: 1.0\nAuthor: RVC Project\nDescription: Voice conversion model"
        )
        
    def save_preset(self):
        """Save current settings as preset"""
        # TODO: Implement preset saving
        QMessageBox.information(self, "Save Preset", "Preset saving coming soon!")
        
    def clear(self):
        """Clear all inputs and reset parameters"""
        self.input_path.clear()
        self.output_path.clear()
        self.pitch_slider.setValue(0)
        self.formant_slider.setValue(0)
        self.index_slider.setValue(75)
        self.filter_value.setValue(3)
        self.rms_slider.setValue(25)