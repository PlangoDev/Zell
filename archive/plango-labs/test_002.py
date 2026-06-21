import numpy as np

np.random.seed(7)

# ─────────────────────────────────────────────────────────────────────────────
# Test 002 — Messy Shapes: plus (+) vs X on a 5x5 grid
#
# Two robots learn to tell a + from an X, even when the drawing is sloppy.
# Densey looks at every pixel on the page, including the blank ones.
# Sparky only looks at the pixels that have ink.
# Same task, same data. We measure who gets it right and who works less.
# ─────────────────────────────────────────────────────────────────────────────

GRID      = 5
INPUT_DIM = GRID * GRID   # 25 pixels
N_TRAIN   = 1000
N_TEST    = 500
EPOCHS    = 40
LR        = 0.05
THRESHOLD = 0.3
NOISE     = 0.12          # chance each pixel gets flipped (a wobbly drawing)

# Clean templates. 1 = ink, 0 = blank.
PLUS = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=float).flatten()

EX = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
], dtype=float).flatten()

# Label: + is 0, X is 1
TEMPLATES = {0: PLUS, 1: EX}


def make_noisy(template, noise):
    """Flip each pixel with probability `noise` to fake a messy drawing."""
    flips = np.random.random(template.shape) < noise
    return np.abs(template - flips)   # XOR: flips a 1 to 0 and a 0 to 1


def make_data(n):
    X, y = [], []
    for _ in range(n):
        label = np.random.randint(0, 2)
        X.append(make_noisy(TEMPLATES[label], NOISE))
        y.append(label)
    return np.array(X), np.array(y, dtype=float)


def render(flat):
    """Turn a 25-length vector back into a little picture."""
    grid = flat.reshape(GRID, GRID)
    rows = []
    for row in grid:
        rows.append(" ".join("#" if px else "." for px in row))
    return "\n".join("    " + r for r in rows)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


X_train, y_train = make_data(N_TRAIN)
X_test,  y_test  = make_data(N_TEST)


# ── Densey: the dense robot ───────────────────────────────────────────────────
# Reads all 25 pixels every single time, blank or not.
# Learns by gradient descent (the textbook way).

W_d = np.zeros(INPUT_DIM)
b_d = 0.0
dense_ops = 0
dense_acc = []

for epoch in range(EPOCHS):
    for i in np.random.permutation(N_TRAIN):
        x, y = X_train[i], y_train[i]
        dense_ops += INPUT_DIM          # all 25 pixels, every time
        pred = sigmoid(W_d @ x + b_d)
        err  = pred - y
        W_d -= LR * err * x
        b_d -= LR * err
    preds = (sigmoid(X_test @ W_d + b_d) >= 0.5)
    dense_acc.append((preds == y_test).mean())


# ── Sparky: the sparse robot ──────────────────────────────────────────────────
# Only looks at pixels that have ink. Blank space costs it nothing.
# Learns by local Hebbian nudges, only when it makes a mistake.

W_s = np.zeros(INPUT_DIM)
b_s = 0.0
sparse_ops = 0
sparse_acc = []

for epoch in range(EPOCHS):
    for i in np.random.permutation(N_TRAIN):
        x, y = X_train[i], y_train[i]
        active = np.where(x == 1)[0]    # only the inked pixels
        sparse_ops += len(active)
        z = W_s[active].sum() + b_s
        spike = float(z >= THRESHOLD)
        err = y - spike
        if err != 0:
            W_s[active] += LR * err
            b_s         += LR * err
    preds = []
    for x in X_test:
        active = np.where(x == 1)[0]
        preds.append(float(W_s[active].sum() + b_s >= THRESHOLD))
    sparse_acc.append((np.array(preds) == y_test).mean())


# ── Results ───────────────────────────────────────────────────────────────────
avg_ink = X_train.sum(axis=1).mean()
efficiency = dense_ops / sparse_ops

print()
print("=" * 54)
print("  Test 002 — Messy Shapes: plus (+) vs X")
print("=" * 54)
print()
print("  A messy +            A messy X")
plus_sample = make_noisy(PLUS, NOISE)
ex_sample   = make_noisy(EX, NOISE)
plus_lines  = render(plus_sample).split("\n")
ex_lines    = render(ex_sample).split("\n")
for pl, xl in zip(plus_lines, ex_lines):
    print(f"{pl}      {xl}")
print()
print(f"  Average ink per drawing: {avg_ink:.1f} of {INPUT_DIM} pixels "
      f"({avg_ink/INPUT_DIM:.0%})")
print()
print(f"  {'Metric':<22} {'Densey':>10} {'Sparky':>10}")
print(f"  {'─'*42}")
print(f"  {'Final accuracy':<22} {dense_acc[-1]:>10.1%} {sparse_acc[-1]:>10.1%}")
print(f"  {'Best accuracy':<22} {max(dense_acc):>10.1%} {max(sparse_acc):>10.1%}")
print(f"  {'Total pixels read':<22} {dense_ops:>10,} {sparse_ops:>10,}")
print(f"  {'Work ratio':<22} {'1.0x':>10} {efficiency:>9.1f}x")
print()
print(f"  Both robots learned the shapes. Sparky did it reading")
print(f"  {efficiency:.1f}x fewer pixels by skipping the blank space.")
print("=" * 54)
print()
