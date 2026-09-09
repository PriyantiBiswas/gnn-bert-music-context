"""Shared helpers: seeding, config loading, small metrics utilities."""
import random
import json
import os
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.yaml") -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    with open(full, "r") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")