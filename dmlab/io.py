"""IO helper functions."""

import json
from pathlib import Path

def ensure_path_is_pathlib(filepath):
    if not isinstance(filepath, Path):
        assert isinstance(filepath, str)
        filepath = Path(filepath)
    return filepath

def load_json(filepath: str) -> dict:
    """Loads json file as a dictionary"""
    with open(filepath, "r", encoding="utf-8") as fp:
        return json.load(fp)

def export_json(obj: dict, filepath: str):
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=4, sort_keys=False, ensure_ascii=True)

# def make_pathdir_if_not_exists(filepath: str):
#     directory = os.path.dirname(filepath)
#     os.makedirs(directory, exist_ok=True)
