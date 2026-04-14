#!/usr/bin/env python
"""
Run SSDiff GUI application.

Usage:
    python run.py
"""

import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ssdiff_gui.main import main  # noqa: E402

if __name__ == "__main__":
    main()
