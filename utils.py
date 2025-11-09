import os, csv, json, random, numpy as np
from pathlib import Path
from datetime import datetime

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def now_iso():
    return datetime.utcnow().isoformat() + 'Z'

def write_metadata_row(csv_path: str, row: dict, fieldnames=None):
    ensure_dir(os.path.dirname(csv_path) or '.')
    write_header = not Path(csv_path).exists()
    if fieldnames is None:
        fieldnames = list(row.keys())
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def save_json(path: str, obj):
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
