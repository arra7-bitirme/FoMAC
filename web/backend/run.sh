#!/bin/bash

# FOMAC Backend Startup Script

echo "Starting FOMAC Backend Server..."
echo "Server will run on http://0.0.0.0:8000"
echo "CORS enabled for http://localhost:3000"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    if python3 -m venv venv; then
        echo "Virtual environment created."
    else
        echo "Failed to create virtual environment. ensurepip may be missing. Will continue using system Python." >&2
    fi
fi

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate

    # Install dependencies if needed
    if [ ! -f ".deps_installed" ]; then
        echo "Installing dependencies in venv..."
        pip install -r requirements.txt
        touch .deps_installed
    fi
else
    echo "Virtual environment not available; using system Python environment."
    if [ ! -f ".deps_installed_system" ]; then
        echo "Installing dependencies system-wide (may require permissions)..."
        python3 -m pip install --user -r requirements.txt || echo "System install failed or skipped. Ensure required packages are available." >&2
        touch .deps_installed_system
    fi
fi

# Run the server (prefer venv python if active)
if [ -n "$VIRTUAL_ENV" ]; then
    python main.py
else
    python3 main.py
fi
