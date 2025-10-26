#!/usr/bin/env python3
"""
Build script for Windows executable
"""

import os
import sys
import shutil
from pathlib import Path
import PyInstaller.__main__

def build_windows():
    """Build Windows executable using PyInstaller"""
    
    print("Building RVC Voice Converter for Windows...")
    
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
        
    # PyInstaller arguments
    args = [
        'main.py',
        '--name=RVC-Voice-Converter',
        '--onefile',
        '--windowed',
        '--icon=assets/icons/app_icon.ico',
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
    
    # Add UPX if available (for compression)
    if shutil.which('upx'):
        args.append('--upx-dir=upx')
        
    # Run PyInstaller
    PyInstaller.__main__.run(args)
    
    # Create installer (optional - requires NSIS)
    create_installer = input("Create Windows installer? (y/n): ")
    if create_installer.lower() == 'y':
        create_nsis_installer()
        
    print("Build complete! Executable is in dist/ folder")
    
def create_nsis_installer():
    """Create NSIS installer for Windows"""
    
    nsis_script = """
    !define APP_NAME "RVC Voice Converter"
    !define APP_VERSION "1.0.0"
    !define APP_PUBLISHER "RVC Project"
    !define APP_URL "https://github.com/yourusername/rvc-voice-converter"
    !define APP_EXECUTABLE "RVC-Voice-Converter.exe"
    
    Name "${APP_NAME}"
    OutFile "RVC-Voice-Converter-Setup.exe"
    InstallDir "$PROGRAMFILES64\\${APP_NAME}"
    InstallDirRegKey HKLM "Software\\${APP_NAME}" "Install_Dir"
    RequestExecutionLevel admin
    
    ; Pages
    Page directory
    Page instfiles
    UninstPage uninstConfirm
    UninstPage instfiles
    
    Section "MainSection" SEC01
        SetOutPath "$INSTDIR"
        File "dist\\${APP_EXECUTABLE}"
        
        ; Create shortcuts
        CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
        CreateShortcut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXECUTABLE}"
        CreateShortcut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXECUTABLE}"
        
        ; Write uninstaller
        WriteUninstaller "$INSTDIR\\uninstall.exe"
        
        ; Registry information
        WriteRegStr HKLM "Software\\${APP_NAME}" "Install_Dir" "$INSTDIR"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayName" "${APP_NAME}"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "UninstallString" "$INSTDIR\\uninstall.exe"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
        WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
    SectionEnd
    
    Section "Uninstall"
        Delete "$INSTDIR\\${APP_EXECUTABLE}"
        Delete "$INSTDIR\\uninstall.exe"
        Delete "$DESKTOP\\${APP_NAME}.lnk"
        Delete "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk"
        RMDir "$SMPROGRAMS\\${APP_NAME}"
        RMDir "$INSTDIR"
        
        DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"
        DeleteRegKey HKLM "Software\\${APP_NAME}"
    SectionEnd
    """
    
    # Save NSIS script
    with open("installer.nsi", "w") as f:
        f.write(nsis_script)
        
    # Compile with NSIS if available
    if shutil.which('makensis'):
        os.system('makensis installer.nsi')
        print("Windows installer created!")
    else:
        print("NSIS not found. Install NSIS to create installer.")
        print("Installer script saved as installer.nsi")

if __name__ == "__main__":
    build_windows()