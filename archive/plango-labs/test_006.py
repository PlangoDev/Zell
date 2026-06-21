import numpy as np
from sklearn.datasets import load_digits

# ─────────────────────────────────────────────────────────────────────────────
# Test 006 — Going Deep, and Wiring Like a Brain
#
# Every test so far wired every neuron to every other neuron. Real brains
# never do that. Each of your neurons listens to only a tiny handful of the
# billions around it. That is SPARSE CONNECTIVITY, and it is the biggest
# efficiency trick we have not used yet.
#
# This test stacks our networks DEEP (where today's AI spends most of its
# energy) and gives the brain robot two kinds of sparsity at once:
#   - sparse firing      : only a few neurons awake per image (k-WTA)
#   - sparse wiring       : each neuron connected to only a few others
# both learned with random feedback (no backprop), and calmed.
#
# The question: when we go vertical, does the brain way pull AHEAD of the
# LLM way by a lot, or does the cheapness finally cost too much accuracy?
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 2
D      = 64
C      = 10
WIDTH  = 128
BATCH  = 50
EPOCHS = 60
FIRE   = 0.20      # fraction of neurons awake per image (k-WTA)
WIRE   = 0.30      # fraction of possible connections that exist


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kwta_rows(r, k):
    """Keep the k strongest neurons in each row, silence the rest."""
    if k >= r.shape[1]:
        return r, (r > 0).astype(float)
    kth = np.partition(r, -k, axis=1)[:, -k][:, None]   # kth largest per row
    gate = (r >= kth).astype(float)
    return r * gate, gate


def dnorm(h):
    """Divisive normalization: each layer rescales itself so the signal
    neither explodes nor fades as the network gets deep. A real brain does
    this constantly (homeostatic gain control)."""
    return h / (np.sqrt((h * h).mean(axis=1, keepdims=True)) + 1e-6)


# ── Data: real handwritten digits ─────────────────────────────────────────────
X, y = load_digits(return_X_y=True)
X = X / 16.0
shuf = np.random.default_rng(0).permutation(len(X))
X, y = X[shuf], y[shuf]
NTR = 1400
Xtr, ytr = X[:NTR], y[:NTR]
Xte, yte = X[NTR:], y[NTR:]
ONEHOT = np.eye(C)
avg_ink = (Xtr > 0).sum(axis=1).mean()


# ── One flexible deep network trainer ─────────────────────────────────────────
# method "bp"  = backpropagation (chains the error back through real weights)
# method "dfa" = direct feedback alignment (random fixed wires to each layer)
# fire < 1 turns on k-WTA sparse firing. wire < 1 turns on sparse connectivity.

def build(L, wire, rng):
    sizes = [D] + [WIDTH] * L + [C]
    W, b, mask = [], [], []
    for i in range(len(sizes) - 1):
        nin, nout = sizes[i], sizes[i + 1]
        is_hidden_in = i < len(sizes) - 2          # don't sparsify the last layer
        m = (rng.random((nout, nin)) < wire).astype(float) if (wire < 1 and is_hidden_in) \
            else np.ones((nout, nin))
        fan = max(1.0, m[0].sum())                 # effective inputs per neuron
        W.append(rng.standard_normal((nout, nin)) * np.sqrt(2 / fan) * m)
        b.append(np.zeros(nout))
        mask.append(m)
    # fixed random feedback wires, one per hidden layer, output error -> layer
    Bfb = [rng.standard_normal((WIDTH, C)) / np.sqrt(C) for _ in range(L)]
    return W, b, mask, Bfb


def train(L, method, fire=1.0, wire=1.0, lr0=0.08, decay=0.03, ntr=NTR):
    rng = np.random.default_rng(SEED)
    W, b, mask, Bfb = build(L, wire, rng)
    k = max(1, int(WIDTH * fire))

    for epoch in range(EPOCHS):
        lr = lr0 / (1 + decay * epoch)
        order = rng.permutation(ntr)
        for s in range(0, ntr, BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xtr[idx], ONEHOT[ytr[idx]]
            n = len(idx)

            # forward
            acts, gates, pres = [xb], [], []
            a = xb
            for l in range(L):
                z = a @ W[l].T + b[l]
                pres.append(z)
                r = np.maximum(0, z)
                if fire < 1:
                    h, g = kwta_rows(r, k)
                    h = dnorm(h)                       # homeostatic gain control
                else:
                    h, g = r, (z > 0).astype(float)
                gates.append(g)
                acts.append(h)
                a = h
            logits = np.clip(a @ W[L].T + b[L], -30, 30)
            probs = softmax_rows(logits)

            # error at the output
            e = (probs - yb) / n
            gW = [None] * (L + 1)
            gW[L] = e.T @ acts[L]
            gb = [None] * (L + 1)
            gb[L] = e.sum(0)

            # send the error back to the hidden layers
            if method == "bp":
                delta = e
                for l in range(L - 1, -1, -1):
                    delta = (delta @ W[l + 1]) * gates[l]
                    gW[l] = delta.T @ acts[l]
                    gb[l] = delta.sum(0)
            else:  # dfa: broadcast the same output error to every layer
                for l in range(L):
                    dh = (e @ Bfb[l].T) * gates[l]
                    gW[l] = dh.T @ acts[l]
                    gb[l] = dh.sum(0)

            for l in range(L + 1):
                W[l] -= lr * gW[l] * mask[l]
                b[l] -= lr * gb[l]

    return (W, b, mask, k, fire)


def evaluate(params, split="test"):
    W, b, _, k, fire = params
    Xe, ye = (Xte, yte) if split == "test" else (Xtr, ytr)
    a = Xe
    for l in range(len(W) - 1):
        r = np.maximum(0, a @ W[l].T + b[l])
        a = dnorm(kwta_rows(r, k)[0]) if fire < 1 else r
    pred = np.argmax(a @ W[-1].T + b[-1], axis=1)
    return (pred == ye).mean()


def ops(L, fire, wire):
    f = fire if fire < 1 else 1.0
    w = wire if wire < 1 else 1.0
    if fire >= 1 and wire >= 1:                    # dense everything
        total = D * WIDTH + (L - 1) * WIDTH * WIDTH + WIDTH * C
    else:
        first  = w * avg_ink * WIDTH               # first layer: awake pixels, sparse wires
        middle = (L - 1) * w * f * WIDTH * WIDTH    # interior: awake inputs, sparse wires
        last   = f * WIDTH * C                      # output reads only awake hidden units
        total  = first + middle + last
    return total


# ── Experiment 1: go vertical ─────────────────────────────────────────────────
print()
print("=" * 70)
print("  Test 006 — Going Deep, and Wiring Like a Brain")
print("=" * 70)
print(f"  Real digits, {NTR} train / {len(Xte)} test.  Width {WIDTH}.")
print(f"  Brain robot: {FIRE:.0%} neurons awake, {WIRE:.0%} of wires present, "
      f"random-feedback learning.")
print()
print("  EXPAND VERTICAL — same task, more hidden layers:")
print(f"  {'depth':>6} {'LLM acc':>8} {'Brain acc':>10} {'LLM ops':>9} "
      f"{'Brain ops':>10} {'Brain saves':>12}")
print(f"  {'─'*60}")

for L in [1, 2, 3, 4, 6, 8]:
    llm = train(L, "bp")
    brain = train(L, "dfa", fire=FIRE, wire=WIRE)
    llm_acc = evaluate(llm)
    brain_acc = evaluate(brain)
    llm_ops = ops(L, 1.0, 1.0)
    brain_ops = ops(L, FIRE, WIRE)
    print(f"  {L:>6} {llm_acc:>8.1%} {brain_acc:>10.1%} {llm_ops:>9,.0f} "
          f"{brain_ops:>10,.0f} {llm_ops/brain_ops:>11.1f}x")

print()

# ── Experiment 2: where does the saving come from (depth 4) ───────────────────
print("  WHERE THE WIN COMES FROM — building up the brain robot at depth 4:")
print(f"  {'configuration':<40} {'test acc':>9} {'vs LLM ops':>11}")
print(f"  {'─'*62}")

L = 4
base_ops = ops(L, 1.0, 1.0)
configs = [
    ("LLM: dense firing, dense wiring, backprop", "bp", 1.0, 1.0),
    ("+ sparse firing (k-WTA)",                   "bp", FIRE, 1.0),
    ("+ sparse wiring",                            "bp", FIRE, WIRE),
    ("+ random-feedback learning (full brain)",    "dfa", FIRE, WIRE),
]
for name, method, fire, wire in configs:
    p = train(L, method, fire=fire, wire=wire)
    acc = evaluate(p)
    o = ops(L, fire, wire)
    print(f"  {name:<40} {acc:>9.1%} {base_ops/o:>10.1f}x")

print()

# overfitting check at depth 4
llm4 = train(4, "bp")
brain4 = train(4, "dfa", fire=FIRE, wire=WIRE)
print("  OVERFITTING CHECK (gap between memorizing train and real test):")
print(f"    LLM   : train {evaluate(llm4,'train'):.1%}  test {evaluate(llm4):.1%}  "
      f"gap {evaluate(llm4,'train')-evaluate(llm4):.1%}")
print(f"    Brain : train {evaluate(brain4,'train'):.1%}  test {evaluate(brain4):.1%}  "
      f"gap {evaluate(brain4,'train')-evaluate(brain4):.1%}")
print()

# ── Experiment 3: the low-data regime brains evolved for ──────────────────────
print("  LOW-DATA REGIME — learning from few examples (depth 2):")
print("  A human reads a 7 after seeing a handful, not sixty thousand.")
print(f"  {'train images':>13} {'LLM test':>9} {'Brain test':>11} {'winner':>9}")
print(f"  {'─'*46}")
for n in [50, 100, 200, 400, 800, 1400]:
    llm = train(2, "bp", ntr=n)
    brain = train(2, "dfa", fire=FIRE, wire=WIRE, ntr=n)
    la, ba = evaluate(llm), evaluate(brain)
    win = "Brain" if ba > la else ("LLM" if la > ba else "tie")
    print(f"  {n:>13} {la:>9.1%} {ba:>11.1%} {win:>9}")
print("=" * 70)
print()
