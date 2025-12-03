# src/preprocess.py
"""
Normalization and augmentation helpers for hand landmarks.

- normalize_landmarks(vec): translate to wrist origin and scale by max distance.
- augment_landmarks(vec): small random jitter + slight scale/rotation (for training).
"""

import numpy as np

def normalize_landmarks(vec):
    """
    vec: 1D array length 63 (21*(x,y,z))
    returns normalized 1D array length 63 (z preserved but x/y translated/scaled)
    """
    lm = np.array(vec, dtype=np.float32).reshape(21, 3)
    # Use wrist (landmark 0) as origin (x,y)
    wrist = lm[0, :2].copy()
    lm[:, :2] = lm[:, :2] - wrist
    # scale using max distance from origin among x,y coords
    dists = np.linalg.norm(lm[:, :2], axis=1)
    maxd = dists.max()
    if maxd <= 0:
        maxd = 1.0
    lm[:, :2] = lm[:, :2] / maxd
    # Optionally scale z relative to same factor (helps invariance)
    lm[:, 2] = lm[:, 2] / (maxd + 1e-6)
    return lm.flatten()

def random_rotate_xy(lm21, max_angle_deg=10):
    """
    Rotate the x,y coordinates around origin by a small random angle.
    lm21: (21,3) array
    """
    theta = np.deg2rad(np.random.uniform(-max_angle_deg, max_angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = lm21[:, :2].dot(R.T)
    lm21[:, :2] = xy
    return lm21

def augment_landmarks(vec, jitter_std=0.02, scale_range=(0.95, 1.05), rotate_deg=8, prob_flip=0.0):
    """
    Apply augmentation to a single landmarks vector (63,)
    - jitter: gaussian noise added to x,y,z
    - scale: small uniform scaling
    - rotate: small rotation in xy plane
    - prob_flip: probability to flip horizontally (if your labels are symmetric)
    """
    lm = np.array(vec, dtype=np.float32).reshape(21, 3)
    # random scale
    s = np.random.uniform(scale_range[0], scale_range[1])
    lm[:, :2] *= s
    # random rotate
    lm = random_rotate_xy(lm, max_angle_deg=rotate_deg)
    # jitter
    lm += np.random.normal(0, jitter_std, lm.shape).astype(np.float32)
    # optional horizontal flip (be careful: labels must be symmetric)
    if prob_flip > 0 and np.random.rand() < prob_flip:
        lm[:, 0] = -lm[:, 0]
    return lm.flatten()
