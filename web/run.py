#!/usr/bin/env python
"""
PHISE Web Application Launch Script

This script launches the Streamlit web interface for PHISE analysis.

Usage:
    python run.py              # Launch main hub
    python run.py --page 01_data_representations  # Launch specific page
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    
    # Get the directory of this script
    web_dir = Path(__file__).parent.absolute()
    main_app = web_dir / "main.py"
    
    # Streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(main_app),
        "--logger.level=info",
        "--client.showErrorDetails=true",
    ]
    
    # Add additional arguments if provided
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"🚀 Launching PHISE Web Application...")
    print(f"   App directory: {web_dir}")
    print(f"   Main app: {main_app}")
    print(f"\n   The application will open in your default browser.")
    print(f"   Local URL: http://localhost:8501")
    print()
    
    # Run Streamlit
    subprocess.run(cmd, cwd=str(web_dir))

if __name__ == "__main__":
    main()
