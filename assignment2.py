#!/usr/bin/env python3
"""EC24313 Artificial Intelligence and Machine Learning - Assignment 2.

Voice-Based Gender Classification Using MFCC and Machine Learning.
Author: Naman Gupta (Roll No: BTECH/10887/24).

Usage:
    python assignment2.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assignment2.pipeline import main

if __name__ == "__main__":
    main()

