
import os, joblib, numpy as np

def save_label_encoder(le, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(le, path)

def load_label_encoder(path):
    return joblib.load(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
