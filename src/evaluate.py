# src/evaluate.py
"""
Evaluate saved model on CSV dataset (normalized).
Prints classification report & confusion matrix.
"""

import glob, numpy as np, pandas as pd, joblib, argparse
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src import preprocess

def load_data(path_pattern):
    files = glob.glob(path_pattern)
    X, y = [], []
    for f in files:
        df = pd.read_csv(f, header=None)
        X.append(df.iloc[:, :63].values)
        y += df.iloc[:, 63].astype(str).tolist()
    if not X:
        return None, None
    X = np.vstack(X)
    return X, np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="experiments/best_model.h5")
    parser.add_argument("--labelenc", default="experiments/label_encoder.joblib")
    parser.add_argument("--data", default="data/landmarks/label_*.csv")
    args = parser.parse_args()

    X, y = load_data(args.data)
    if X is None:
        print("No data found.")
        return
    # normalize
    Xn = np.vstack([preprocess.normalize_landmarks(x) for x in X])
    le = joblib.load(args.labelenc)
    model = load_model(args.model)
    preds = model.predict(Xn)
    pred_idxs = preds.argmax(axis=1)
    pred_labels = le.classes_[pred_idxs]
    print("Classification report:")
    print(classification_report(y, pred_labels))
    print("Confusion matrix:")
    print(confusion_matrix(y, pred_labels, labels=le.classes_))

if __name__ == "__main__":
    main()
