import numpy as np
import os


def get_project_path():
    cur_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return cur_dir
