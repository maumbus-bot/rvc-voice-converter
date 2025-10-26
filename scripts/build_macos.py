#!/usr/bin/env python3
"""
Build script for macOS application
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess

def build_macos():
    """Build macOS application bundle"""
    
    print("Building RVC Voice Converter for macOS...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Clean previous builds
    dist_path = project_root / "dist"
    build_path = project_root / "build"
    
    if dist_path.exists():
        shutil.rmtree(dist_path)
    if build_path.exists():
        shutil.rmtree(build_path)
    
    # Check if py2app is installed
    try:
        import py2app
    except ImportError:
        print("py2app not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "py2app"])
        import py2app
    
    # Create setup.py for py2app
    setup_content = """
from setuptools import setup

APP = ['main.py']
DATA_FILES = [
    ('src', ['src']),
    ('models', ['models']),
    ('configs', ['configs']),
    ('assets', ['assets']),
]

OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'assets/icons/app_icon.icns',
    'plist': {
        'CFBundleName': 'RVC Voice Converter',
        'CFBundleDisplayName': 'RVC Voice Converter',
        'CFBundleGetInfoString': 'RVC Voice Converter 1.0.0',
        'CFBundleIdentifier': 'com.rvcproject.voiceconverter',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHumanReadableCopyright': '© 2024 RVC Project',
        'NSHighResolutionCapable': True,
    },
    'packages': [
        'torch',
        'torchaudio',
        'torchvision',
        'librosa',
        'numpy',
        'scipy',
        'sklearn',
        'PyQt6',
        'soundfile',
        'resampy',
    ],
    'includes': [
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
"""
    
    # Write setup.py
    with open("setup_mac.py", "w") as f:
        f.write(setup_content)
    
    # Run py2app
    try:
        subprocess.run([sys.executable, "setup_mac.py", "py2app"], check=True)
        print("macOS app bundle created successfully!")
        
        # Create DMG installer
        create_dmg = input("Create DMG installer? (y/n): ")
        if create_dmg.lower() == 'y':
            create_dmg_installer()
            
    except subprocess.CalledProcessError as e:
        print(f"Error building app: {e}")
        
        # Try alternative with PyInstaller
        print("Trying with PyInstaller...")
        build_with_pyinstaller()
        
def build_with_pyinstaller():
    """Alternative build using PyInstaller for macOS"""
    
    import PyInstaller.__main__
    
    args = [
        'main.py',
        '--name=RVC Voice Converter',
        '--windowed',
        '--onefile',
        '--osx-bundle-identifier=com.rvcproject.voiceconverter',
        '--icon=assets/icons/app_icon.icns',
        '--add-data=src:src',
        '--add-data=models:models',
        '--add-data=configs:configs',
        '--add-data=assets:assets',
        '--add-data=README.md:.',
        '--hidden-import=torch',
        '--hidden-import=torchaudio',
        '--hidden-import=torchvision',
        '--hidden-import=librosa',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=PyQt6',
        '--hidden-import=numpy',
        '--collect-all=torch',
        '--collect-all=torchaudio',
        '--collect-all=librosa',
        '--collect-all=PyQt6',
        '--noconfirm',
        '--clean',
    ]
    
    PyInstaller.__main__.run(args)
    print("macOS executable created with PyInstaller!")
    
def create_dmg_installer():
    """Create DMG installer for macOS"""
    
    print("Creating DMG installer...")
    
    # Check if create-dmg is installed
    if not shutil.which('create-dmg'):
        print("create-dmg not found. Installing via Homebrew...")
        subprocess.run(["brew", "install", "create-dmg"])
    
    # Create DMG
    dmg_script = """
    create-dmg \\
        --volname "RVC Voice Converter" \\
        --volicon "assets/icons/app_icon.icns" \\
        --window-pos 200 120 \\
        --window-size 600 400 \\
        --icon-size 100 \\
        --icon "RVC Voice Converter.app" 150 185 \\
        --hide-extension "RVC Voice Converter.app" \\
        --app-drop-link 450 185 \\
        "RVC-Voice-Converter-macOS.dmg" \\
        "dist/"
    """
    
    try:
        subprocess.run(dmg_script, shell=True, check=True)
        print("DMG installer created successfully!")
    except subprocess.CalledProcessError:
        # Alternative DMG creation
        print("Creating simple DMG...")
        subprocess.run([
            "hdiutil", "create",
            "-volname", "RVC Voice Converter",
            "-srcfolder", "dist/RVC Voice Converter.app",
            "-ov",
            "-format", "UDZO",
            "RVC-Voice-Converter-macOS.dmg"
        ])
        print("DMG created!")

def code_sign_app():
    """Code sign the macOS app (requires Apple Developer certificate)"""
    
    print("Code signing requires Apple Developer certificate.")
    sign = input("Do you have a Developer ID certificate? (y/n): ")
    
    if sign.lower() == 'y':
        identity = input("Enter your Developer ID: ")
        
        # Sign the app
        subprocess.run([
            "codesign",
            "--deep",
            "--force",
            "--verify",
            "--verbose",
            "--sign", identity,
            "dist/RVC Voice Converter.app"
        ])
        
        print("App signed successfully!")
        
        # Notarize the app (requires Apple Developer account)
        notarize = input("Notarize the app? (y/n): ")
        if notarize.lower() == 'y':
            username = input("Apple ID username: ")
            password = input("App-specific password: ")
            
            # Create ZIP for notarization
            subprocess.run([
                "ditto", "-c", "-k", "--keepParent",
                "dist/RVC Voice Converter.app",
                "RVC-Voice-Converter.zip"
            ])
            
            # Submit for notarization
            subprocess.run([
                "xcrun", "altool",
                "--notarize-app",
                "--primary-bundle-id", "com.rvcproject.voiceconverter",
                "--username", username,
                "--password", password,
                "--file", "RVC-Voice-Converter.zip"
            ])
            
            print("App submitted for notarization!")

if __name__ == "__main__":
    build_macos()
    
    # Optional code signing
    code_sign_app()