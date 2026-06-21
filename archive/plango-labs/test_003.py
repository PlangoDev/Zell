import numpy as np

np.random.seed(3)

# ─────────────────────────────────────────────────────────────────────────────
# Test 003 — Four Shapes and a Shouting Match
#
# Bigger: a 7x7 grid (49 pixels) instead of 5x5.
# Harder: four shapes now (+ , X , square, triangle), not two.
# Messier: enough noise that nobody gets a perfect score.
#
# New idea this test: WINNER-TAKE-ALL.
# With more than two answers, each robot grows one output neuron per shape.
# All of them shout a number. The loudest one wins and becomes the guess.
# That is how a brain picks one thought out of many at once.
#
# The real question: when the puzzle is genuinely hard, does Sparky's
# cheaper way of learning still keep up with Densey's textbook way?
# ─────────────────────────────────────────────────────────────────────────────

GRID      = 7
INPUT_DIM = GRID * GRID   # 49 pixels
N_CLASSES = 4
N_TRAIN   = 2000
N_TEST    = 800
EPOCHS    = 60
LR        = 0.05
NOISE     = 0.35          # cranked way up so the shapes blur together

PLUS = [
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [1,1,1,1,1,1,1],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
]
EX = [
    [1,0,0,0,0,0,1],
    [0,1,0,0,0,1,0],
    [0,0,1,0,1,0,0],
    [0,0,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,0,0,1,0],
    [1,0,0,0,0,0,1],
]
SQUARE = [
    [1,1,1,1,1,1,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
]
TRIANGLE = [
    [0,0,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],
    [1,0,0,0,0,0,1],
    [1,1,1,1,1,1,1],
]

NAMES     = ["plus", "X", "square", "triangle"]
TEMPLATES = [np.array(t, dtype=float).flatten() for t in (PLUS, EX, SQUARE, TRIANGLE)]


def make_noisy(template, noise):
    flips = np.random.random(template.shape) < noise
    return np.abs(template - flips)


def make_data(n):
    X, y = [], []
    for _ in range(n):
        label = np.random.randint(0, N_CLASSES)
        X.append(make_noisy(TEMPLATES[label], NOISE))
        y.append(label)
    return np.array(X), np.array(y)


def render(flat):
    grid = flat.reshape(GRID, GRID)
    return "\n".join("    " + " ".join("#" if px else "." for px in row) for row in grid)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


X_train, y_train = make_data(N_TRAIN)
X_test,  y_test  = make_data(N_TEST)


# ── Densey: dense softmax classifier ──────────────────────────────────────────
# Four output neurons, each reads all 49 pixels. Softmax turns the four
# scores into probabilities, and gradient descent nudges every weight.

W_d = np.zeros((N_CLASSES, INPUT_DIM))
b_d = np.zeros(N_CLASSES)
dense_ops = 0
dense_acc = []

for epoch in range(EPOCHS):
    for i in np.random.permutation(N_TRAIN):
        x, y = X_train[i], y_train[i]
        dense_ops += N_CLASSES * INPUT_DIM       # every neuron reads every pixel
        probs = softmax(W_d @ x + b_d)
        probs[y] -= 1.0                          # gradient of cross-entropy
        W_d -= LR * np.outer(probs, x)
        b_d -= LR * probs
    preds = np.argmax(X_test @ W_d.T + b_d, axis=1)
    dense_acc.append((preds == y_test).mean())


# ── Sparky: sparse competitive classifier ─────────────────────────────────────
# Four output neurons, but each only reads the inked pixels. They shout their
# scores, loudest wins (winner-take-all). Learning is the perceptron rule:
# on a wrong guess, reward the right neuron's active weights and punish the
# wrong winner's. Each weight only sees whether its own pixel was lit. Local.

W_s = np.zeros((N_CLASSES, INPUT_DIM))
b_s = np.zeros(N_CLASSES)
sparse_ops = 0
sparse_acc = []

for epoch in range(EPOCHS):
    for i in np.random.permutation(N_TRAIN):
        x, y = X_train[i], y_train[i]
        active = np.where(x == 1)[0]
        sparse_ops += N_CLASSES * len(active)    # only inked pixels counted
        scores = W_s[:, active].sum(axis=1) + b_s
        pred = int(np.argmax(scores))
        if pred != y:                            # learn only from mistakes
            W_s[y,    active] += LR
            b_s[y]            += LR
            W_s[pred, active] -= LR
            b_s[pred]         -= LR
    preds = []
    for x in X_test:
        active = np.where(x == 1)[0]
        preds.append(int(np.argmax(W_s[:, active].sum(axis=1) + b_s)))
    sparse_acc.append((np.array(preds) == y_test).mean())


# ── Results ───────────────────────────────────────────────────────────────────
avg_ink = X_train.sum(axis=1).mean()
efficiency = dense_ops / sparse_ops

print()
print("=" * 56)
print("  Test 003 — Four Shapes and a Shouting Match")
print("=" * 56)
print()
for c in range(N_CLASSES):
    print(f"  a messy {NAMES[c]}")
    print(render(make_noisy(TEMPLATES[c], NOISE)))
    print()

print(f"  Grid: {GRID}x{GRID} = {INPUT_DIM} pixels   Shapes: {N_CLASSES}   Noise: {NOISE:.0%}")
print(f"  Average ink per drawing: {avg_ink:.1f} of {INPUT_DIM} pixels "
      f"({avg_ink/INPUT_DIM:.0%})")
print()
print(f"  {'Metric':<22} {'Densey':>10} {'Sparky':>10}")
print(f"  {'─'*42}")
print(f"  {'Final accuracy':<22} {dense_acc[-1]:>10.1%} {sparse_acc[-1]:>10.1%}")
print(f"  {'Best accuracy':<22} {max(dense_acc):>10.1%} {max(sparse_acc):>10.1%}")
print(f"  {'Pixels read':<22} {dense_ops:>10,} {sparse_ops:>10,}")
print(f"  {'Work ratio':<22} {'1.0x':>10} {efficiency:>9.1f}x")
print()
gap = (dense_acc[-1] - sparse_acc[-1]) * 100
print(f"  Accuracy gap: {gap:+.1f} points (Densey minus Sparky)")
print()
print(f"  {'Epoch':<8} {'Densey':>10} {'Sparky':>10}")
print(f"  {'─'*30}")
for e in range(0, EPOCHS, 10):
    print(f"  {e+1:<8} {dense_acc[e]:>10.1%} {sparse_acc[e]:>10.1%}")
print("=" * 56)
print()
