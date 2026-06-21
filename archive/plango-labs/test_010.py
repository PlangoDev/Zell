import numpy as np
from sklearn.datasets import fetch_openml

# ─────────────────────────────────────────────────────────────────────────────
# Test 010 — Beating the LLM, the brain's own way
#
# Every earlier test said the accuracy gap was ONLY the learning rule: random
# feedback cannot train deep features as well as backprop. So we stop trying.
# We do what the cerebellum and dentate gyrus actually do:
#
#   FIXED RANDOM FEATURES. A huge bank of feature detectors, each wired to a
#   small patch of the input with random weights, never trained. No deep
#   learning to get wrong. Only a shallow readout learns, by a local rule.
#
#   A COMMITTEE. The brain is a crowd of cheap circuits voting. Because each
#   of ours is far cheaper than the dense LLM, we can run several and still
#   come in cheaper, then let them vote.
#
# We show the whole accuracy-vs-work curve as experts are added, and find the
# point where the brain beats the LLM on BOTH accuracy and energy.
# ─────────────────────────────────────────────────────────────────────────────

SEED      = 11
C         = 10
PATCH     = 5          # each detector sees a 5x5 patch (a receptive field)
HEXP      = 4000       # detectors per expert
EXPERTS   = 6
RD_EPOCHS = 20
LLM_BATCH = 100        # the LLM at full strength (small batch -> its real 88.3%)
RD_BATCH  = 256
ONEHOT    = np.eye(C)


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ── Data: full Fashion-MNIST, 14x14, 60k train / 10k test ─────────────────────
print("  loading the full laundry pile (cached)...")
Xall, yall = fetch_openml('Fashion-MNIST', version=1, as_frame=False,
                          parser='liac-arff', return_X_y=True)
Xall = Xall.astype(np.float32).reshape(-1, 28, 28)
Xall = Xall.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196) / 255.0
yall = yall.astype(int)
idx = np.random.default_rng(0).permutation(len(Xall))
Xall, yall = Xall[idx], yall[idx]
Xtr, ytr = Xall[:60000], yall[:60000]
Xte, yte = Xall[60000:70000], yall[60000:70000]
D = 196
INK = float((Xtr > 0.1).mean())


# ── The LLM contestant: dense backprop MLP, at full strength ──────────────────
def train_llm(width=256, epochs=20, lr0=0.06):
    rng = np.random.default_rng(SEED)
    W1 = rng.standard_normal((width, D)) * np.sqrt(2 / D)
    W2 = rng.standard_normal((width, width)) * np.sqrt(2 / width)
    W3 = rng.standard_normal((C, width)) * np.sqrt(2 / width)
    b1, b2, b3 = np.zeros(width), np.zeros(width), np.zeros(C)
    for ep in range(epochs):
        lr = lr0 / (1 + 0.02 * ep)
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), LLM_BATCH):
            i = order[s:s + LLM_BATCH]
            x, yb = Xtr[i], ONEHOT[ytr[i]]
            h1 = np.maximum(0, x @ W1.T + b1)
            h2 = np.maximum(0, h1 @ W2.T + b2)
            p = softmax_rows(h2 @ W3.T + b3)
            e = (p - yb) / len(i)
            d2 = (e @ W3) * (h2 > 0)
            d1 = (d2 @ W2) * (h1 > 0)
            W3 -= lr * e.T @ h2; b3 -= lr * e.sum(0)
            W2 -= lr * d2.T @ h1; b2 -= lr * d2.sum(0)
            W1 -= lr * d1.T @ x;  b1 -= lr * d1.sum(0)
    h1 = np.maximum(0, Xte @ W1.T + b1)
    h2 = np.maximum(0, h1 @ W2.T + b2)
    acc = (np.argmax(h2 @ W3.T + b3, axis=1) == yte).mean()
    return acc, D * width + width * width + width * C


# ── A brain expert: fixed random patch detectors + a learned linear readout ───
# Each detector has random weights over one small random patch of the image
# (a receptive field) and is NEVER trained. Only the linear readout learns,
# by a local delta rule. No backprop, no deep credit assignment, anywhere.
def make_expert(seed):
    rng = np.random.default_rng(seed)
    W = np.zeros((HEXP, D), dtype=np.float32)
    for h in range(HEXP):
        r, c = rng.integers(0, 14 - PATCH + 1), rng.integers(0, 14 - PATCH + 1)
        blk = np.zeros((14, 14), dtype=np.float32)
        blk[r:r + PATCH, c:c + PATCH] = rng.standard_normal((PATCH, PATCH))
        W[h] = blk.ravel()
    return W


def feats(X, W, mu=None, sd=None):
    F = np.maximum(0, X @ W.T)                          # local patch matches, ReLU
    if mu is None:
        mu, sd = F.mean(0), F.std(0) + 1e-6
    return ((F - mu) / sd).astype(np.float32), mu, sd


def train_readout(F, y, rng):
    Wr = np.zeros((C, F.shape[1]), dtype=np.float32)
    br = np.zeros(C, dtype=np.float32)
    for ep in range(RD_EPOCHS):
        lr = 0.3 / (1 + 0.05 * ep)
        order = rng.permutation(len(F))
        for s in range(0, len(F), RD_BATCH):
            i = order[s:s + RD_BATCH]
            p = softmax_rows(F[i] @ Wr.T + br)
            e = (p - ONEHOT[y[i]]) / len(i)
            Wr -= lr * e.T @ F[i]; br -= lr * e.sum(0)
    return Wr, br


# ── Run the showdown ──────────────────────────────────────────────────────────
print()
print("=" * 68)
print("  Test 010 — Beating the LLM, the brain's own way")
print("=" * 68)
print(f"  Full Fashion-MNIST: {len(Xtr):,} train / {len(Xte):,} test, 10 classes.")
print()

llm_acc, llm_ops = train_llm()
print(f"  THE BAR TO BEAT — LLM (dense backprop): {llm_acc:.2%}   ({llm_ops:,} ops/photo)")
print()

per_expert_ops = HEXP * PATCH * PATCH * INK + HEXP * C
print(f"  THE BRAIN'S COMMITTEE (each expert = {HEXP} patch detectors")
print(f"  + one locally-trained readout, ~{per_expert_ops:,.0f} ops each):")
print()
print(f"  {'experts':>8} {'accuracy':>9} {'ops/photo':>10} {'beats LLM?':>20}")
print(f"  {'─'*52}")

vote = np.zeros((len(Xte), C))
acc_win = None
for ex in range(EXPERTS):
    W = make_expert(100 + ex)
    Ftr, mu, sd = feats(Xtr, W)
    Fte, _, _ = feats(Xte, W, mu, sd)
    Wr, br = train_readout(Ftr, ytr, np.random.default_rng(ex))
    vote += softmax_rows(Fte @ Wr.T + br)
    n = ex + 1
    acc = (np.argmax(vote, axis=1) == yte).mean()
    ops = n * per_expert_ops
    tag = "BEATS LLM" if acc > llm_acc else ("cheaper" if ops < llm_ops else "")
    print(f"  {n:>8} {acc:>9.2%} {ops:>10,.0f} {tag:>20}")
    if acc > llm_acc and acc_win is None:
        acc_win = (n, acc, ops)

print(f"  {'─'*52}")
best = (np.argmax(vote, axis=1) == yte).mean()
if acc_win:
    n, acc, ops = acc_win
    cost = ops / llm_ops
    print(f"  BRAIN WINS ACCURACY: {best:.2%} vs LLM {llm_acc:.2%}  "
          f"(+{(best - llm_acc) * 100:.1f} points), zero backprop.")
    print(f"  It crosses the LLM at {n} experts, and spends {cost:.1f}x the compute")
    print(f"  to do it -- the committee trades the brain's efficiency for accuracy.")
    print(f"  (The other end of the dial, test 009, was 10x CHEAPER at ~85%.)")
else:
    print(f"  Best committee accuracy {best:.2%} vs LLM {llm_acc:.2%}.")
print("=" * 68)
print()
