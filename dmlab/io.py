"""Input/Output helper functions."""

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

def export_json(obj: dict, filepath: str, mode: str="x", **kwargs):
    dump_kwargs = dict(indent=4, sort_keys=False, ensure_ascii=True)
    dump_kwargs.update(**kwargs)
    with open(filepath, mode, encoding="utf-8") as fp:
        json.dump(obj, fp, **dump_kwargs)

def export_dataframe(
        df,
        filepath,
        sep="\t",
        na_rep="n/a",
        index=False,
        **kwargs
    ):
    """Wrapper set with preferred defaults.
    Tab separation and "n/a" are BIDS-motivated.
    """
    filepath = ensure_path_is_pathlib(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, sep=sep, na_rep=na_rep, index=index, **kwargs)
