@echo off
REM Development run script for RVC Voice Converter (Windows)

echo RVC Voice Converter - Development Mode
echo ======================================

REM Check Python version
python --version

REM Create virtual environment if not exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
if not exist "models\pretrained" mkdir models\pretrained
if not exist "models\custom" mkdir models\custom
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "data" mkdir data
if not exist "output" mkdir output
if not exist "logs" mkdir logs

REM Run the application
echo Starting RVC Voice Converter...
python main.py

REM Deactivate virtual environment on exit
deactivate
pause