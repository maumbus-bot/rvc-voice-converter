#!/bin/bash

# Development run script for RVC Voice Converter

echo "RVC Voice Converter - Development Mode"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p models/pretrained
mkdir -p models/custom
mkdir -p models/checkpoints
mkdir -p data
mkdir -p output
mkdir -p logs

# Run the application
echo "Starting RVC Voice Converter..."
python main.py

# Deactivate virtual environment on exit
deactivate