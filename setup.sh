#!/bin/bash

if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3 before running this script."
    exit 1
fi

if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Please install pip3 before running this script."
    exit 1
fi

echo "Creating a Python virtual environment..."
python3 -m venv venv

echo "Activating the virtual environment..."
source venv/bin/activate

pip3 install numpy pandas matplotlib scipy pyyaml

echo "Setup is complete. Your Python environment is ready."
echo "To activate your environment, use the command: source venv/bin/activate"

