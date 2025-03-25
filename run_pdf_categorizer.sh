#!/bin/bash

# Define virtual environment name
VENV_DIR="venv"

# Check if virtual environment exists, if not, create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies if not installed
pip install --upgrade pip > /dev/null
REQUIRED_PACKAGES=("requests" "xml_cleaner" "epub-conversion" "pymupdf" "scikit-learn" "nltk" "pdfminer.six" "ebooklib" "epub-conversion")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "$pkg" > /dev/null; then
        echo "Installing missing package: $pkg"
        pip install "$pkg"
    fi
done

# Run the Python script with optional directory argument
python3 pdf_categorizer.py "$1"

# Deactivate virtual environment
deactivate

echo "Script execution complete!"
