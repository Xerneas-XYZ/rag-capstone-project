#!/usr/bin/env python
"""
Run script for FastAPI backend
Usage: python run_api.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

# Run uvicorn
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "app.main:app",
    "--reload",
    "--port", "8000",
    "--host", "0.0.0.0"
])
