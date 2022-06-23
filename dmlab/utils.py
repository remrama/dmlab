"""Helper functions for the helper functions."""

from pathlib import Path

def ensure_path_is_pathlib(filepath):
    if not isinstance(filepath, Path):
        assert isinstance(filepath, str)
        filepath = Path(filepath)
    return filepath
