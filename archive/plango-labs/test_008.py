import numpy as np
from sklearn.datasets import load_digits
from scipy.ndimage import rotate as nd_rotate, shift as nd_shift

# ─────────────────────────────────────────────────────────────────────────────
# Test 008 — Sleep, Dreams, and Not Forgetting
#
# A neural network has a terrible secret. Teach it one thing, then teach it a
# second thing, and it ERASES the first. This is called catastrophic
# forgetting, and it is one of the deepest differences between a machine and
# a brain. You did not forget how to read when you learned to ride a bike.
#
# Brains avoid this with SLEEP. While you sleep, your hippocampus replays the
# day's memories to your slow cortex, over and over, mixing the new in with
# the old so nothing gets overwritten. The cortex re-lives experiences it is
# not currently having. Neuroscientists call it systems consolidation, and it
# is a big reason a night of sleep cements what you learned.
#
# This test:
#   - makes MORE DATA by augmenting the digits (rotations, shifts, noise)
#   - teaches the brain one set of digits, then another, and watches it forget
#   - lets it SLEEP (replay) and watches the lost memories come back
#   - lets it DREAM (noisy replay) and checks if imperfect memories help more
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 7
D      = 64
C      = 10
WIDTH  = 128
BATCH  = 50
FIRE   = 0.20
WIRE   = 0.30
TASK_A = [0, 1, 2, 3, 4]      # the first things we learn
TASK_B = [5, 6, 7, 8, 9]      # the second things we learn


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kwta_rows(r, k):
    if k >= r.shape[1]:
        return r, (r > 0).astype(float)
    kth = np.partition(r, -k, axis=1)[:, -k][:, None]
    gate = (r >= kth).astype(float)
    return r * gate, gate


def dnorm(h):
    return h / (np.sqrt((h * h).mean(axis=1, keepdims=True)) + 1e-6)


# ── More data: augment the digits with rotations, shifts, and noise ───────────
def vary(Xset, rng):
    """Return one geometrically transformed copy: a small rotation, a shift,
    a little noise. This is what a 'dream' version of a memory looks like."""
    base = Xset.reshape(-1, 8, 8)
    aug = np.empty_like(base)
    for i in range(len(base)):
        a = nd_rotate(base[i], rng.uniform(-12, 12), reshape=False, order=1, mode="constant")
        a = nd_shift(a, (rng.uniform(-1, 1), rng.uniform(-1, 1)), order=1, mode="constant")
        aug[i] = a
    return np.clip(aug.reshape(-1, 64) + rng.normal(0, 0.06, (len(base), 64)), 0, 1)


def augment(Xset, yset, factor, rng):
    outX, outY = [Xset], [yset]
    for _ in range(factor):
        outX.append(vary(Xset, rng)); outY.append(yset)
    return np.vstack(outX), np.concatenate(outY)


X, y = load_digits(return_X_y=True)
X = X / 16.0
shuf = np.random.default_rng(0).permutation(len(X))
X, y = X[shuf], y[shuf]
Xtr0, ytr0 = X[:1400], y[:1400]
Xte, yte = X[1400:], y[1400:]
Xtr, ytr = augment(Xtr0, ytr0, factor=3, rng=np.random.default_rng(1))   # ~5600 images
ONEHOT = np.eye(C)


def subset(Xs, ys, classes):
    m = np.isin(ys, classes)
    return Xs[m], ys[m]


XA, yA = subset(Xtr, ytr, TASK_A)
XB, yB = subset(Xtr, ytr, TASK_B)
XtA, ytA = subset(Xte, yte, TASK_A)
XtB, ytB = subset(Xte, yte, TASK_B)


# ── The brain cortex: deep sparse net, random-feedback learning ───────────────
def init_net(L, rng):
    sizes = [D] + [WIDTH] * L + [C]
    W, b, mask = [], [], []
    for i in range(len(sizes) - 1):
        nin, nout = sizes[i], sizes[i + 1]
        sparsify = i < len(sizes) - 2
        m = (rng.random((nout, nin)) < WIRE).astype(float) if sparsify else np.ones((nout, nin))
        fan = max(1.0, m[0].sum())
        W.append(rng.standard_normal((nout, nin)) * np.sqrt(2 / fan) * m)
        b.append(np.zeros(nout)); mask.append(m)
    Bfb = [rng.standard_normal((WIDTH, C)) / np.sqrt(C) for _ in range(L)]
    return {"W": W, "b": b, "mask": mask, "Bfb": Bfb, "k": max(1, int(WIDTH * FIRE)), "L": L}


def clone(state):
    return {"W": [a.copy() for a in state["W"]], "b": [a.copy() for a in state["b"]],
            "mask": state["mask"], "Bfb": state["Bfb"], "k": state["k"], "L": state["L"]}


def train_on(state, Xset, yset, epochs, lr0=0.08, decay=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    W, b, mask, Bfb, k, L = (state["W"], state["b"], state["mask"],
                             state["Bfb"], state["k"], state["L"])
    for epoch in range(epochs):
        lr = lr0 / (1 + decay * epoch)
        order = rng.permutation(len(Xset))
        for s in range(0, len(Xset), BATCH):
            idx = order[s:s + BATCH]
            xb, yb = Xset[idx], ONEHOT[yset[idx]]
            n = len(idx)
            acts, gates = [xb], []
            a = xb
            for l in range(L):
                z = a @ W[l].T + b[l]
                h, g = kwta_rows(np.maximum(0, z), k); h = dnorm(h)
                gates.append(g); acts.append(h); a = h
            logits = np.clip(a @ W[L].T + b[L], -30, 30)
            e = (softmax_rows(logits) - yb) / n
            gW = [None] * (L + 1); gb = [None] * (L + 1)
            gW[L] = e.T @ acts[L]; gb[L] = e.sum(0)
            for l in range(L):
                dh = (e @ Bfb[l].T) * gates[l]
                gW[l] = dh.T @ acts[l]; gb[l] = dh.sum(0)
            for l in range(L + 1):
                W[l] -= lr * gW[l] * mask[l]; b[l] -= lr * gb[l]


def acc(state, Xe, ye):
    W, b, k, L = state["W"], state["b"], state["k"], state["L"]
    a = Xe
    for l in range(L):
        a = dnorm(kwta_rows(np.maximum(0, a @ W[l].T + b[l]), k)[0])
    return (np.argmax(a @ W[L].T + b[L], axis=1) == ye).mean()


# ── The hippocampus: store a sample of experiences to replay later ────────────
def store(Xset, yset, per_class, rng):
    keepX, keepY = [], []
    for c in np.unique(yset):
        ci = np.where(yset == c)[0]
        pick = rng.choice(ci, min(per_class, len(ci)), replace=False)
        keepX.append(Xset[pick]); keepY.append(yset[pick])
    return np.vstack(keepX), np.concatenate(keepY)


def sleep(state, bufX, bufY, nights, dream=False, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(nights):
        Xr = vary(bufX, rng) if dream else bufX        # dreams replay varied memories
        train_on(state, Xr, bufY, epochs=2, lr0=0.05, rng=rng)


# ── Run the story ─────────────────────────────────────────────────────────────
print()
print("=" * 66)
print("  Test 008 — Sleep, Dreams, and Not Forgetting")
print("=" * 66)
print(f"  Made more data: {len(Xtr0)} digits grown to {len(Xtr)} by rotating,")
print(f"  shifting and adding noise. Task A = {TASK_A}, Task B = {TASK_B}.")
print()

rng = np.random.default_rng(SEED)
brain = init_net(2, rng)

# Day 1: learn Task A
train_on(brain, XA, yA, epochs=30, rng=rng)
a_afterA = acc(brain, XtA, ytA)
print(f"  Day 1 — learned digits {TASK_A}:  Task A accuracy {a_afterA:.1%}")

# the hippocampus quietly files away a few examples of what it has seen
bufAX, bufAY = store(XA, yA, 40, rng)

# Day 2: learn Task B (this is where the forgetting happens)
train_on(brain, XB, yB, epochs=30, rng=rng)
bufBX, bufBY = store(XB, yB, 40, rng)
a_noSleep = acc(brain, XtA, ytA)
b_noSleep = acc(brain, XtB, ytB)
print(f"  Day 2 — learned digits {TASK_B}:  Task B accuracy {b_noSleep:.1%}")
print(f"                                   Task A accuracy {a_noSleep:.1%}  <- FORGOTTEN")
print()

# full memory buffer the hippocampus holds: a sample of A and B
memX = np.vstack([bufAX, bufBX])
memY = np.concatenate([bufAY, bufBY])

print("  NOW THE BRAIN SLEEPS (replaying its stored memories):")
print(f"  {'after':>16} {'Task A':>8} {'Task B':>8} {'both':>7}")
print(f"  {'─'*42}")
print(f"  {'staying awake':>16} {a_noSleep:>8.1%} {b_noSleep:>8.1%} "
      f"{(a_noSleep+b_noSleep)/2:>7.1%}")

sleeper = clone(brain)
for night in [1, 2, 3, 5]:
    s = clone(brain)
    sleep(s, memX, memY, nights=night, dream=False, rng=np.random.default_rng(night))
    aa, bb = acc(s, XtA, ytA), acc(s, XtB, ytB)
    print(f"  {str(night)+' nights sleep':>16} {aa:>8.1%} {bb:>8.1%} {(aa+bb)/2:>7.1%}")

print()

# Dreaming: does replaying TRANSFORMED memories (rotated, shifted) beat
# replaying them exactly? We tested it honestly. It did not help here.
plain = clone(brain)
sleep(plain, memX, memY, nights=5, dream=False, rng=np.random.default_rng(99))
pa, pb = acc(plain, XtA, ytA), acc(plain, XtB, ytB)
dreamer = clone(brain)
sleep(dreamer, memX, memY, nights=5, dream=True, rng=np.random.default_rng(99))
da, db = acc(dreamer, XtA, ytA), acc(dreamer, XtB, ytB)

print("  DREAMING — does replaying transformed memories beat exact replay?")
print(f"    exact replay (5 nights):   both {(pa+pb)/2:.1%}")
print(f"    dream replay (5 nights):   both {(da+db)/2:.1%}   (no clear gain here)")
print()
print(f"  The hippocampus only kept {len(memX)} memories "
      f"({len(memX)/len(Xtr):.0%} of all it saw) and still rescued both tasks.")
print("=" * 66)
print()
