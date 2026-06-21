#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# export_data.py  —  DATA PLUMBING ONLY.  This is NOT the test.
#
# The actual experiment (test 011) is written entirely in C. But the C code
# needs the Fashion-MNIST pixels on disk in a form it can read with fread().
# So this tiny script does one job, once: load the dataset (already cached by
# the earlier Python tests), shrink each photo 28x28 -> 14x14 exactly the way
# tests 009/010 did, and dump it as flat binary files.
#
# Layout written into ./data (run from the test_011/ folder):
#   train_X.bin : float32, 60000 * 196   (row-major, values in [0,1])
#   train_y.bin : uint8,   60000         (class 0..9)
#   test_X.bin  : float32, 10000 * 196
#   test_y.bin  : uint8,   10000
#
# Same shuffle seed (0) as tests 009/010, so the train/test split is identical
# and the head-to-head is apples-to-apples with the earlier LLM numbers.
#
# Run:  python3 tools/export_data.py
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
from sklearn.datasets import fetch_openml

print("loading Fashion-MNIST (cached from earlier tests)...")
X, y = fetch_openml('Fashion-MNIST', version=1, as_frame=False,
                    parser='liac-arff', return_X_y=True)

# keep FULL 28x28 resolution, stored as uint8 (0-255) so the file stays small
# enough for git; the C loader normalizes to [0,1].
X = X.astype(np.uint8)          # N x 784, values 0..255
y = y.astype(np.uint8)

# identical shuffle + split to tests 009/010
idx = np.random.default_rng(0).permutation(len(X))
X, y = X[idx], y[idx]
Xtr, ytr = X[:60000], y[:60000]
Xte, yte = X[60000:70000], y[60000:70000]

os.makedirs('data', exist_ok=True)
Xtr.tofile('data/train_X.bin'); ytr.tofile('data/train_y.bin')
Xte.tofile('data/test_X.bin');  yte.tofile('data/test_y.bin')
print(f"wrote data/  train={Xtr.shape} test={Xte.shape}  (28x28, uint8)")
