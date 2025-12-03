# src/train.py
"""
Train pipeline (uses src/preprocess.normalize_landmarks + augmentation).
Saves:
 - experiments/best_model.h5
 - experiments/label_encoder.joblib
"""

import os, glob, argparse
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.models.mlp import build_mlp
from src.utils import save_label_encoder, ensure_dir
from src import preprocess
import tensorflow as tf

def load_csvs(path_pattern):
    files = glob.glob(path_pattern)
    X_list, y_list = [], []
    for f in files:
        df = pd.read_csv(f, header=None)
        if df.shape[1] < 64:
            continue
        X_list.append(df.iloc[:, :63].values)
        y_list += df.iloc[:, 63].astype(str).tolist()
    if not X_list:
        return None, None
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y

def generate_synthetic(num_classes=3, samples_per_class=300):
    labels = ['hello','yes','no'][:num_classes]
    X = []
    y = []
    for idx,lab in enumerate(labels):
        for i in range(samples_per_class):
            v = np.random.normal(loc=0.5 + idx*0.05, scale=0.12, size=(63,))
            X.append(v)
            y.append(lab)
    return np.vstack(X), np.array(y)

def apply_normalize(X):
    Xn = np.vstack([preprocess.normalize_landmarks(x) for x in X])
    return Xn

def augment_batch(X, y, times=1):
    """Return augmented X_aug, y_aug (times times more)."""
    X_aug, y_aug = [], []
    for _ in range(times):
        for xi, yi in zip(X, y):
            aug = preprocess.augment_landmarks(xi, jitter_std=0.02, scale_range=(0.96,1.04), rotate_deg=6)
            aug = preprocess.normalize_landmarks(aug)
            X_aug.append(aug)
            y_aug.append(yi)
    if X_aug:
        return np.vstack(X_aug), np.array(y_aug)
    else:
        return np.empty((0,63)), np.array([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/landmarks")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", default="experiments")
    parser.add_argument("--augment-times", type=int, default=1, help="How many augmentation copies per sample for training")
    args = parser.parse_args()

    X, y = load_csvs(os.path.join(args.data_dir, "label_*.csv"))
    if X is None:
        print("No landmark CSVs found — creating synthetic dataset for demo.")
        X, y = generate_synthetic(num_classes=3, samples_per_class=400)
    print("Raw dataset:", X.shape, "labels:", np.unique(y))

    # normalize all
    Xn = apply_normalize(X)
    print("Normalized dataset shape:", Xn.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    ensure_dir(args.save_dir)

    # split
    X_train, X_val, y_train, y_val = train_test_split(Xn, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

    # augmentation on training set
    if args.augment_times > 0:
        X_aug, y_aug = augment_batch(X_train, y_train, times=args.augment_times)
        if X_aug.size:
            X_train = np.vstack([X_train, X_aug])
            y_train = np.concatenate([y_train, y_aug])
            print("After augmentation, train size:", X_train.shape)

    model = build_mlp(input_dim=Xn.shape[1], num_classes=len(le.classes_))
    # callbacks
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=cb)

    # save model & encoder
    save_path = os.path.join(args.save_dir, "best_model.h5")
    model.save(save_path)
    save_label_encoder(le, os.path.join(args.save_dir, "label_encoder.joblib"))
    print("Saved model to", save_path)
    print("Saved label encoder to", os.path.join(args.save_dir, "label_encoder.joblib"))

if __name__ == '__main__':
    main()
