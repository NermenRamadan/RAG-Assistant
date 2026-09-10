"""Puts the api/ folder on sys.path so `import main` works in tests,
the same way it works when you run `uvicorn main:app` from inside api/.
"""

import os
import sys

API_DIR = os.path.join(os.path.dirname(__file__), "..", "api")
sys.path.insert(0, os.path.abspath(API_DIR))
