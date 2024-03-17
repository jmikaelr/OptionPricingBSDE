#!/bin/bash

python3 -m venv optbsde || { echo "Failed to create virtual environment"; exit 1; }

source optbsde/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

pip install --upgrade pip || { echo "Failed to upgrade pip"; exit 1; }

pip install pandas scipy numpy matplotlib || { echo "Failed to install packages"; exit 1; }

echo "Setup completed successfully."

