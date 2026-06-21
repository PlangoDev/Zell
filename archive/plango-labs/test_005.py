import numpy as np
from sklearn.datasets import load_digits

# ─────────────────────────────────────────────────────────────────────────────
# Test 005 — Real Handwriting, and How Far Sparsity Can Go
#
# The big showdown, scaled way up from test 003. Everything we have learned,
# stacked into one robot, fighting on real data.
#
#   The task: recognize real handwritten digits, 0 through 9. Not toy shapes
#   we drew, but 1,797 little 8x8 images written by actual humans.
#
#   Densey  = the way today's AI works:
#             a deep network, EVERY neuron fires every time, learns by
#             backpropagation (the biologically impossible one).
#
#   Sparky  = everything the brain taught us, all at once:
#             - a deep network (two hidden layers)
#             - only the top few neurons fire, the rest stay silent (k-WTA)
#             - learns by Direct Feedback Alignment (random feedback wires,
#               no weight transport), now reaching BOTH hidden layers
#             - the calm-down rule from test 004
#             - skips blank pixels for free
#
# Then we crank Sparky's sparsity from loose to brutal and find the cliff:
# how few neurons can fire before it stops being able to read, and how much
# energy we save on the way down.
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 1
D      = 64        # 8x8 pixels in
H1     = 128       # first hidden layer
H2     = 64        # second hidden layer
C      = 10        # ten digits out
EPOCHS = 35


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def relu(a):
    return np.maximum(0, a)


def kwta(v, k):
    """Keep only the k strongest neurons. Silence the rest (set to 0)."""
    if k >= len(v):
        return v, (v > 0)
    keep = np.argpartition(v, -k)[-k:]      # indices of the k biggest
    out = np.zeros_like(v)
    out[keep] = v[keep]
    return out, (out > 0)                    # values, and who is awake


# ── Load real handwritten digits ──────────────────────────────────────────────
X, y = load_digits(return_X_y=True)
X = X / 16.0                                  # pixels 0..16 -> 0..1

shuf = np.random.default_rng(0).permutation(len(X))
X, y = X[shuf], y[shuf]
NTR = 1400
Xtr, ytr = X[:NTR], y[:NTR]
Xte, yte = X[NTR:], y[NTR:]

avg_ink = (Xtr > 0).sum(axis=1).mean()       # how many pixels actually have ink


# ── Densey: deep, dense, backprop (the LLM stack) ─────────────────────────────
def train_dense(lr0=0.1, decay=0.02):
    rng = np.random.default_rng(SEED)
    W1 = rng.standard_normal((H1, D))  * np.sqrt(2 / D)
    W2 = rng.standard_normal((H2, H1)) * np.sqrt(2 / H1)
    W3 = rng.standard_normal((C,  H2)) * np.sqrt(2 / H2)
    b1, b2, b3 = np.zeros(H1), np.zeros(H2), np.zeros(C)

    for epoch in range(EPOCHS):
        lr = lr0 / (1 + decay * epoch)
        for i in rng.permutation(NTR):
            x = Xtr[i]
            a1 = W1 @ x + b1;  h1 = relu(a1)
            a2 = W2 @ h1 + b2; h2 = relu(a2)
            out = softmax(W3 @ h2 + b3)

            e = out.copy(); e[ytr[i]] -= 1            # output error
            d2 = (W3.T @ e) * (a2 > 0)                # backprop chains down...
            d1 = (W2.T @ d2) * (a1 > 0)               # ...through real weights

            W3 -= lr * np.outer(e, h2);  b3 -= lr * e
            W2 -= lr * np.outer(d2, h1); b2 -= lr * d2
            W1 -= lr * np.outer(d1, x);  b1 -= lr * d1

    return (W1, b1, W2, b2, W3, b3)


def eval_dense(p):
    W1, b1, W2, b2, W3, b3 = p
    H1a = relu(Xte @ W1.T + b1)
    H2a = relu(H1a @ W2.T + b2)
    pred = np.argmax(H2a @ W3.T + b3, axis=1)
    return (pred == yte).mean()


# ── Sparky: deep, sparse k-WTA, Direct Feedback Alignment, calmed ─────────────
def train_sparse(kfrac, lr0=0.05, decay=0.03):
    rng = np.random.default_rng(SEED)
    W1 = rng.standard_normal((H1, D))  * np.sqrt(2 / D)
    W2 = rng.standard_normal((H2, H1)) * np.sqrt(2 / H1)
    W3 = rng.standard_normal((C,  H2)) * np.sqrt(2 / H2)
    b1, b2, b3 = np.zeros(H1), np.zeros(H2), np.zeros(C)

    # fixed random feedback wires, one set per hidden layer (never trained)
    B1 = rng.standard_normal((H1, C)) / np.sqrt(C)
    B2 = rng.standard_normal((H2, C)) / np.sqrt(C)

    k1 = max(1, int(H1 * kfrac))
    k2 = max(1, int(H2 * kfrac))

    for epoch in range(EPOCHS):
        lr = lr0 / (1 + decay * epoch)
        for i in rng.permutation(NTR):
            x = Xtr[i]
            a1 = W1 @ x + b1;  h1, g1 = kwta(relu(a1), k1)   # only k1 awake
            a2 = W2 @ h1 + b2; h2, g2 = kwta(relu(a2), k2)   # only k2 awake
            out = softmax(W3 @ h2 + b3)

            e = out.copy(); e[ytr[i]] -= 1            # output error
            dh2 = (B2 @ e) * g2                        # error beamed straight to
            dh1 = (B1 @ e) * g1                        # each layer via random wires

            W3 -= lr * np.outer(e, h2);   b3 -= lr * e
            W2 -= lr * np.outer(dh2, h1); b2 -= lr * dh2
            W1 -= lr * np.outer(dh1, x);  b1 -= lr * dh1

    return (W1, b1, W2, b2, W3, b3), (k1, k2)


def eval_sparse(p, k1, k2):
    W1, b1, W2, b2, W3, b3 = p
    correct = 0
    for i in range(len(Xte)):
        x = Xte[i]
        h1, _ = kwta(relu(W1 @ x + b1), k1)
        h2, _ = kwta(relu(W2 @ h1 + b2), k2)
        if np.argmax(W3 @ h2 + b3) == yte[i]:
            correct += 1
    return correct / len(Xte)


def dense_ops():
    return H1 * D + H2 * H1 + C * H2          # every input, every neuron


def sparse_ops(k1, k2):
    # layer 1: must score all H1 neurons, but only over inked pixels
    # layer 2: only the k1 awake neurons from layer 1 feed forward
    # layer 3: only the k2 awake neurons from layer 2 feed forward
    return H1 * avg_ink + H2 * k1 + C * k2


# ── Run it ────────────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("  Test 005 — Real Handwriting, and How Far Sparsity Can Go")
print("=" * 64)
print(f"  Data: {len(X)} real handwritten digits (8x8), 10 classes")
print(f"  Train {NTR} / Test {len(Xte)}.  Avg ink per image: "
      f"{avg_ink:.0f} of {D} pixels ({avg_ink/D:.0%})")
print(f"  Network: {D} -> {H1} -> {H2} -> {C}, two hidden layers")
print()

dense_p = train_dense()
dense_acc = eval_dense(dense_p)
d_ops = dense_ops()

print(f"  Densey  (dense + backprop, the LLM way)")
print(f"     accuracy {dense_acc:.1%}   ops/image {d_ops:,}")
print()
print("  Sparky  (deep + k-WTA sparsity + direct feedback alignment + calm):")
print(f"  {'awake neurons':>14} {'accuracy':>9} {'ops/image':>11} {'vs Densey':>11}")
print(f"  {'─'*48}")

sweep = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03]
results = []
for kf in sweep:
    p, (k1, k2) = train_sparse(kf)
    acc = eval_sparse(p, k1, k2)
    ops = sparse_ops(k1, k2)
    ratio = d_ops / ops
    results.append((kf, k1, k2, acc, ops, ratio))
    tag = "  (all awake)" if kf == 1.0 else ""
    print(f"  {f'{kf:.0%} ({k1}+{k2})':>14} {acc:>9.1%} {ops:>11,.0f} "
          f"{ratio:>10.1f}x{tag}")

print()

# pick the best efficiency point that still reads well (>= 90%)
good = [r for r in results if r[3] >= 0.90]
if good:
    best = max(good, key=lambda r: r[5])
    print(f"  Sweet spot: {best[0]:.0%} of neurons awake still reads digits at "
          f"{best[3]:.1%},")
    print(f"  while doing {best[5]:.1f}x less work than the dense LLM-style net.")
print()

print("  Where does the efficiency go as you scale sparsity deeper?")
print("  Each sparse hidden layer multiplies its input-sparsity by its")
print("  output-sparsity, so the savings COMPOUND as nets get deeper.")
print("  Idealized projection, one shared sparsity f across L hidden layers:")
print(f"    {'sparsity f':>11} {'savings per interior layer (1/f^2)':>36}")
for f in [0.5, 0.25, 0.10, 0.05, 0.02]:
    print(f"    {f:>11.0%} {1/f**2:>33.0f}x")
print()
print("  The brain runs near f = 1-5%. Stack that across many layers, add")
print("  event-driven spikes and low-precision signals, and the gap grows")
print("  toward the ~100,000,000x we started this whole project chasing.")
print("=" * 64)
print()
