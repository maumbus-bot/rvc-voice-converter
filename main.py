#!/usr/bin/env python3
"""
RVC Voice Converter
Main application entry point
"""

import sys
import os
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPalette, QColor, QFont, QIcon

from src.gui.main_window import MainWindow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RVCApplication(QApplication):
    """Main application class"""
    
    def __init__(self, argv):
        super().__init__(argv)
        
        # Set application info
        self.setApplicationName("RVC Voice Converter")
        self.setOrganizationName("RVC Project")
        self.setApplicationDisplayName("RVC Voice Converter")
        
        # Set application style
        self.setStyle("Fusion")
        
        # Set dark palette
        self.set_dark_palette()
        
        # Set default font
        font = QFont("Segoe UI", 10)
        self.setFont(font)
        
    def set_dark_palette(self):
        """Set dark color palette for application"""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        
        # Base colors (for text editors, etc)
        palette.setColor(QPalette.ColorRole.Base, QColor(58, 58, 58))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(66, 66, 66))
        
        # Text colors
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        
        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(58, 58, 58))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        
        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(74, 144, 226))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        
        # Link colors
        palette.setColor(QPalette.ColorRole.Link, QColor(74, 144, 226))
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(80, 80, 120))
        
        # Disabled colors
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(127, 127, 127))
        
        self.setPalette(palette)


def create_splash_screen():
    """Create and show splash screen"""
    splash = QSplashScreen()
    
    # Create a simple splash pixmap (in production, use actual image)
    pixmap = QPixmap(600, 400)
    pixmap.fill(QColor(43, 43, 43))
    
    splash.setPixmap(pixmap)
    splash.showMessage(
        "Loading RVC Voice Converter...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white
    )
    
    return splash


def main():
    """Main application entry point"""
    # Create application
    app = RVCApplication(sys.argv)
    
    # Show splash screen
    splash = create_splash_screen()
    splash.show()
    app.processEvents()
    
    # Create main window
    logger.info("Starting RVC Voice Converter...")
    main_window = MainWindow()
    
    # Close splash and show main window
    QTimer.singleShot(1500, lambda: [
        splash.close(),
        main_window.show()
    ])
    
    # Run application
    logger.info("Application started successfully")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()