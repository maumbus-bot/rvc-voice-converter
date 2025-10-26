"""
Main Window
Central GUI window with all components
"""

import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QMenu, QStatusBar, QMessageBox,
    QFileDialog, QProgressBar, QLabel, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QAction, QIcon, QPixmap, QPalette, QColor
import logging

# Import tabs
from .conversion_tab import ConversionTab
from .training_tab import TrainingTab
from .models_tab import ModelsTab
from .settings_dialog import SettingsDialog

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Window properties
        self.setWindowTitle("RVC Voice Converter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QTabWidget::pane {
                background-color: #2b2b2b;
                border: 1px solid #555;
            }
            QTabWidget::tab-bar {
                alignment: left;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #ffffff;
                padding: 10px 20px;
                margin: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-size: 14px;
            }
            QTabBar::tab:selected {
                background-color: #4a90e2;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar::item:selected {
                background-color: #4a90e2;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background-color: #4a90e2;
            }
            QStatusBar {
                background-color: #2b2b2b;
                color: #ffffff;
                border-top: 1px solid #555;
            }
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2968a3;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #999;
            }
            QLineEdit, QTextEdit, QListWidget, QTreeWidget {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
                margin-right: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #3a3a3a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #4a90e2;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #357abd;
            }
            QProgressBar {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 3px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox, QRadioButton {
                color: #ffffff;
            }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555;
                border-radius: 3px;
                background-color: #3a3a3a;
            }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #4a90e2;
                border-color: #4a90e2;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # Initialize components
        self.init_ui()
        self.init_menu()
        self.init_statusbar()
        
    def init_ui(self):
        """Initialize UI components"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # Create tabs
        self.conversion_tab = ConversionTab()
        self.models_tab = ModelsTab()
        self.training_tab = TrainingTab()
        
        # Add tabs
        self.tab_widget.addTab(self.conversion_tab, "🎤 Voice Conversion")
        self.tab_widget.addTab(self.models_tab, "📦 Model Manager")
        self.tab_widget.addTab(self.training_tab, "🎓 Training")
        
        # Add tab widget to main layout
        main_layout.addWidget(self.tab_widget)
        
    def init_menu(self):
        """Initialize menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # New project
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        # Open project
        open_action = QAction("Open Project", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        # Save project
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Import model
        import_model_action = QAction("Import Model...", self)
        import_model_action.triggered.connect(self.import_model)
        file_menu.addAction(import_model_action)
        
        # Export model
        export_model_action = QAction("Export Model...", self)
        export_model_action.triggered.connect(self.export_model)
        file_menu.addAction(export_model_action)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        # Settings
        settings_action = QAction("Settings...", self)
        settings_action.setShortcut("Ctrl+,")
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Toggle dark theme
        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        # Audio recorder
        recorder_action = QAction("Audio Recorder", self)
        recorder_action.triggered.connect(self.show_recorder)
        tools_menu.addAction(recorder_action)
        
        # Batch processing
        batch_action = QAction("Batch Processing", self)
        batch_action.triggered.connect(self.show_batch_processor)
        tools_menu.addAction(batch_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # Documentation
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        # About
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def init_statusbar(self):
        """Initialize status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Status message
        self.status_message = QLabel("Ready")
        self.statusbar.addWidget(self.status_message)
        
        # GPU status
        self.gpu_status = QLabel("GPU: Checking...")
        self.statusbar.addPermanentWidget(self.gpu_status)
        
        # Check GPU status
        self.check_gpu_status()
        
    def check_gpu_status(self):
        """Check and update GPU status"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.gpu_status.setText(f"GPU: {device_name}")
                self.gpu_status.setStyleSheet("color: #4a90e2;")
            else:
                self.gpu_status.setText("GPU: Not Available (CPU Mode)")
                self.gpu_status.setStyleSheet("color: #ff9800;")
        except:
            self.gpu_status.setText("GPU: Error")
            self.gpu_status.setStyleSheet("color: #f44336;")
            
    def new_project(self):
        """Create new project"""
        # Clear all tabs
        reply = QMessageBox.question(
            self,
            "New Project",
            "Create a new project? This will clear current work.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.conversion_tab.clear()
            self.training_tab.clear()
            self.status_message.setText("New project created")
            
    def open_project(self):
        """Open project file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "RVC Project Files (*.rvcproj);;All Files (*.*)"
        )
        
        if file_path:
            # TODO: Load project
            self.status_message.setText(f"Project loaded: {Path(file_path).name}")
            
    def save_project(self):
        """Save current project"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "RVC Project Files (*.rvcproj);;All Files (*.*)"
        )
        
        if file_path:
            # TODO: Save project
            self.status_message.setText(f"Project saved: {Path(file_path).name}")
            
    def import_model(self):
        """Import model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Model",
            "",
            "Model Files (*.pth *.onnx);;All Files (*.*)"
        )
        
        if file_path:
            self.models_tab.import_model(file_path)
            self.status_message.setText(f"Model imported: {Path(file_path).name}")
            
    def export_model(self):
        """Export current model"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            "",
            "PyTorch Model (*.pth);;ONNX Model (*.onnx);;All Files (*.*)"
        )
        
        if file_path:
            # TODO: Export model
            self.status_message.setText(f"Model exported: {Path(file_path).name}")
            
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        dialog.exec()
        
    def toggle_theme(self):
        """Toggle between light and dark theme"""
        # TODO: Implement theme toggle
        pass
        
    def show_recorder(self):
        """Show audio recorder window"""
        # TODO: Implement audio recorder
        QMessageBox.information(self, "Audio Recorder", "Audio recorder coming soon!")
        
    def show_batch_processor(self):
        """Show batch processing window"""
        # TODO: Implement batch processor
        QMessageBox.information(self, "Batch Processing", "Batch processing coming soon!")
        
    def show_documentation(self):
        """Show documentation"""
        import webbrowser
        webbrowser.open("https://github.com/yourusername/rvc-voice-converter/wiki")
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About RVC Voice Converter",
            """<h2>RVC Voice Converter</h2>
            <p>Version 1.0.0</p>
            <p>A powerful AI-based voice conversion application using 
            Retrieval-based Voice Conversion technology.</p>
            <p>© 2024 RVC Voice Converter</p>
            <p>Licensed under MIT License</p>"""
        )
        
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(
            self,
            "Exit Application",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()