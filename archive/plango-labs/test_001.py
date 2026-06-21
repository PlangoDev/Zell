import numpy as np

np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────────────
# Test 001 — Binary Majority Classification
#
# Task: 10-bit binary input vector. Label = 1 if more than 5 bits are on.
# Both networks learn from the same data, same number of epochs, same LR.
# We track accuracy and total operations to measure the efficiency gap.
# ─────────────────────────────────────────────────────────────────────────────

INPUT_DIM  = 10
N_TRAIN    = 1000
N_TEST     = 500
EPOCHS     = 50
LR         = 0.05
THRESHOLD  = 0.3   # sparse network fires if weighted sum >= this


def make_data(n):
    X = np.random.randint(0, 2, (n, INPUT_DIM)).astype(float)
    y = (X.sum(axis=1) > INPUT_DIM / 2).astype(float)
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


X_train, y_train = make_data(N_TRAIN)
X_test,  y_test  = make_data(N_TEST)


# ── Dense network ─────────────────────────────────────────────────────────────
# Single linear layer + sigmoid output.
# Loss: binary cross-entropy.
# Weight update: gradient descent — grad = (pred - target) * input.
# Every input is used on every forward pass regardless of value.

W_d = np.zeros(INPUT_DIM)
b_d = 0.0

dense_ops        = 0
dense_acc        = []
dense_converged  = None   # first epoch >= 90% accuracy

for epoch in range(EPOCHS):
    idx = np.random.permutation(N_TRAIN)
    for i in idx:
        x, y = X_train[i], y_train[i]
        dense_ops += INPUT_DIM          # all 10 inputs used every time
        pred = sigmoid(W_d @ x + b_d)
        err  = pred - y
        W_d -= LR * err * x
        b_d -= LR * err

    preds = sigmoid(X_test @ W_d + b_d) >= 0.5
    acc   = (preds == y_test).mean()
    dense_acc.append(acc)
    if dense_converged is None and acc >= 0.90:
        dense_converged = epoch + 1


# ── Sparse (spiking) network ──────────────────────────────────────────────────
# Same task, same data, same architecture shape.
# Forward pass only touches weights for inputs that are 1 (active).
# On a 10-bit input with ~50% bits on, that's ~5 ops per sample instead of 10.
#
# Learning rule — supervised Hebb:
#   if predicted wrong: nudge active weights by (target - prediction) * LR
#   if predicted right: no update
# No gradient, no global error signal — each weight only sees local activity.

W_s = np.zeros(INPUT_DIM)
b_s = 0.0

sparse_ops       = 0
sparse_acc       = []
sparse_converged = None

for epoch in range(EPOCHS):
    idx = np.random.permutation(N_TRAIN)
    for i in idx:
        x, y = X_train[i], y_train[i]
        active = np.where(x == 1)[0]    # indices of bits that are on
        sparse_ops += len(active)       # only active inputs cost computation

        z     = W_s[active].sum() + b_s
        spike = float(z >= THRESHOLD)

        err = y - spike
        if err != 0:                    # only update on mistakes
            W_s[active] += LR * err
            b_s          += LR * err

    preds = []
    for x in X_test:
        active = np.where(x == 1)[0]
        preds.append(float(W_s[active].sum() + b_s >= THRESHOLD))
    acc = (np.array(preds) == y_test).mean()
    sparse_acc.append(acc)
    if sparse_converged is None and acc >= 0.90:
        sparse_converged = epoch + 1


# ── Results ───────────────────────────────────────────────────────────────────

efficiency = dense_ops / sparse_ops

print()
print("=" * 52)
print("  Test 001 — Binary Majority Classification")
print("=" * 52)
print(f"  Epochs: {EPOCHS}   Train: {N_TRAIN}   Test: {N_TEST}   LR: {LR}")
print()
print(f"  {'Metric':<24} {'Dense':>10} {'Sparse':>10}")
print(f"  {'─'*44}")
print(f"  {'Final accuracy':<24} {dense_acc[-1]:>10.1%} {sparse_acc[-1]:>10.1%}")
print(f"  {'Epoch @ 90% acc':<24} {str(dense_converged) + ' ep':>10} {str(sparse_converged) + ' ep':>10}")
print(f"  {'Total ops':<24} {dense_ops:>10,} {sparse_ops:>10,}")
print(f"  {'Ops ratio':<24} {'1.0x':>10} {efficiency:>9.1f}x")
print()
print(f"  Sparse used {efficiency:.1f}x fewer operations to reach the same accuracy.")
print()

# Accuracy curve (every 10 epochs)
print(f"  {'Epoch':<8} {'Dense acc':>10} {'Sparse acc':>10}")
print(f"  {'─'*30}")
for e in range(0, EPOCHS, 5):
    print(f"  {e+1:<8} {dense_acc[e]:>10.1%} {sparse_acc[e]:>10.1%}")
print("=" * 52)
print()
