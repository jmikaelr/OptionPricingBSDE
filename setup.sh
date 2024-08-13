#!/bin/bash

# Check if Python 3 is installed
if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before running this script."
    exit 1
fi

# Check if pip3 is installed
if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Please install pip3 before running this script."
    exit 1
fi

# Create a Python virtual environment
echo "Creating a Python virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install necessary Python packages
echo "Installing required Python packages..."
pip install numpy pandas matplotlib scipy pyyaml torch

echo "Setup is complete. Your Python environment is ready."
echo "To activate your environment, use the command: source venv/bin/activate"

