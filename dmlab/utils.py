"""Helper functions for the helper functions."""

import os

def make_pathdir_if_not_exists(filepath):
    directory = os.path.dirname(filepath)
    # if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)
