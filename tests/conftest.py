"""conftest.py — pytest configuration for the tests/ directory.

Adds src/ to sys.path so that `import scientific_rag` works
without needing to install the package first (useful in CI).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

