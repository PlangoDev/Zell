import numpy as np
from sklearn.datasets import load_digits

# ─────────────────────────────────────────────────────────────────────────────
# Test 007 — Giving the Brain a Memory
#
# Tests 001-006 built one learning system: a slow net that grinds out general
# patterns. It is efficient but it underfits, because random-feedback learning
# cannot teach a deep stack as sharply as backprop. So it loses on accuracy.
#
# Real brains do not rely on one system. They have TWO, and they cooperate:
#
#   NEOCORTEX  - slow, learns general rules over a lifetime of examples.
#                This is the deep sparse net we already have.
#   HIPPOCAMPUS - fast, stores specific experiences and recalls them instantly.
#                See a thing once, remember it. Decide later by asking
#                "what does this remind me of?"
#
# This is Complementary Learning Systems theory (McClelland et al. 1995).
# The hippocampus stores memories in SPARSE codes on purpose, so they do not
# smear together. That is the exact sparse firing we built in test 005-006.
# The efficiency trick and the memory trick are the same trick.
#
# We bolt a hippocampus onto the cortex and ask: does remembering buy back
# the accuracy that the slow learner gave up? And in the low-data world brains
# actually live in, can memory finally beat the LLM outright?
# ─────────────────────────────────────────────────────────────────────────────

SEED   = 5
D      = 64
C      = 10
WIDTH  = 128
BATCH  = 50
EPOCHS = 60
FIRE   = 0.20
WIRE   = 0.30


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


# ── Data ──────────────────────────────────────────────────────────────────────
X, y = load_digits(return_X_y=True)
X = X / 16.0
shuf = np.random.default_rng(0).permutation(len(X))
X, y = X[shuf], y[shuf]
NTR = 1400
Xtr, ytr = X[:NTR], y[:NTR]
Xte, yte = X[NTR:], y[NTR:]
ONEHOT = np.eye(C)


# ── The cortex: deep sparse net, random-feedback learning (from test 006) ─────
def build(L, wire, rng):
    sizes = [D] + [WIDTH] * L + [C]
    W, b, mask = [], [], []
    for i in range(len(sizes) - 1):
        nin, nout = sizes[i], sizes[i + 1]
        sparsify = wire < 1 and i < len(sizes) - 2
        m = (rng.random((nout, nin)) < wire).astype(float) if sparsify else np.ones((nout, nin))
        fan = max(1.0, m[0].sum())
        W.append(rng.standard_normal((nout, nin)) * np.sqrt(2 / fan) * m)
        b.append(np.zeros(nout))
        mask.append(m)
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
            acts, gates = [xb], []
            a = xb
            for l in range(L):
                z = a @ W[l].T + b[l]
                r = np.maximum(0, z)
                if fire < 1:
                    h, g = kwta_rows(r, k); h = dnorm(h)
                else:
                    h, g = r, (z > 0).astype(float)
                gates.append(g); acts.append(h); a = h
            logits = np.clip(a @ W[L].T + b[L], -30, 30)
            probs = softmax_rows(logits)
            e = (probs - yb) / n
            gW = [None] * (L + 1); gb = [None] * (L + 1)
            gW[L] = e.T @ acts[L]; gb[L] = e.sum(0)
            if method == "bp":
                delta = e
                for l in range(L - 1, -1, -1):
                    delta = (delta @ W[l + 1]) * gates[l]
                    gW[l] = delta.T @ acts[l]; gb[l] = delta.sum(0)
            else:
                for l in range(L):
                    dh = (e @ Bfb[l].T) * gates[l]
                    gW[l] = dh.T @ acts[l]; gb[l] = dh.sum(0)
            for l in range(L + 1):
                W[l] -= lr * gW[l] * mask[l]; b[l] -= lr * gb[l]
    return (W, b, mask, k, fire)


def forward(params, Xe):
    """Return both the output logits and the top hidden representation."""
    W, b, _, k, fire = params
    a = Xe
    rep = a
    for l in range(len(W) - 1):
        r = np.maximum(0, a @ W[l].T + b[l])
        a = dnorm(kwta_rows(r, k)[0]) if fire < 1 else r
        rep = a
    logits = a @ W[-1].T + b[-1]
    return logits, rep


def cortex_acc(params, Xe, ye):
    logits, _ = forward(params, Xe)
    return (np.argmax(logits, axis=1) == ye).mean()


# ── The hippocampus: episodic memory with its own input pathway ───────────────
# The dentate gyrus expands each input into a big, very sparse code so two
# memories barely overlap (pattern separation). This is a FIXED random
# projection, not learned, so it does not depend on the cortex being good.
# Then we store (sparse code -> label) and decide by recalling the closest
# memories and letting them vote. This is the brain's fast, one-shot system.

DG_DIM      = 800     # the dentate gyrus blows the input up to many neurons
DG_SPARSITY = 0.05    # then keeps only the strongest 5% awake
_dg_rng = np.random.default_rng(99)
P_DG = _dg_rng.standard_normal((DG_DIM, D)) / np.sqrt(D)   # fixed random wiring


def dg_code(Xe):
    h = np.maximum(0, Xe @ P_DG.T)                  # expand
    k = max(1, int(DG_DIM * DG_SPARSITY))
    c, _ = kwta_rows(h, k)                           # sparsify (separate)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


def remember(Xmem, ymem):
    return dg_code(Xmem), ymem


def recall_probs(memory, Xe, knn=9):
    keys, labels = memory
    q = dg_code(Xe)
    sim = q @ keys.T                                 # overlap with every memory
    kk = min(knn, keys.shape[0])
    top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
    votes = np.zeros((len(Xe), C))
    for i in range(len(Xe)):
        for j in top[i]:
            votes[i, labels[j]] += max(sim[i, j], 0.0)
    return votes / (votes.sum(axis=1, keepdims=True) + 1e-8)


def hpc_acc(memory, Xe, ye, knn=9):
    return (np.argmax(recall_probs(memory, Xe, knn), axis=1) == ye).mean()


def combined_acc(params, memory, Xe, ye, alpha=0.5, knn=9):
    cortex = softmax_rows(forward(params, Xe)[0])
    final = alpha * cortex + (1 - alpha) * recall_probs(memory, Xe, knn)
    return (np.argmax(final, axis=1) == ye).mean()


print()
print("=" * 68)
print("  Test 007 — Giving the Brain a Memory")
print("=" * 68)
print(f"  Real digits, {NTR} train / {len(Xte)} test.")
print()

# ── Experiment 0: one-shot learning — the thing a trained net CANNOT do ────────
# Nobody trains on the digits 8 or 9. Then we show the brain a SINGLE 8 and a
# single 9 and ask it to recognise brand-new 8s and 9s. A standard trained
# network has no output for a class it never saw and cannot add one from a
# lone example. The hippocampus does it instantly, with zero training.
NOVEL = [8, 9]
nv_tr = np.isin(ytr, NOVEL)
nv_te = np.isin(yte, NOVEL)
Xpool, ypool = Xtr[nv_tr], ytr[nv_tr]          # source of example(s) to show
Xnov, ynov = Xte[nv_te], yte[nv_te]            # brand-new ones to identify

print("  ONE-SHOT LEARNING — recognise digits 8 vs 9 never trained on:")
print(f"    A standard trained net: cannot. It has no 8 or 9 to output (50% guess).")
print(f"  {'examples shown':>17} {'brain memory':>13}")
print(f"  {'─'*32}")
for shots in [1, 2, 3, 5, 10]:
    trials = []
    for t in range(25):                            # average over many draws
        rs = np.random.default_rng(1000 * shots + t)
        pick = []
        for c in NOVEL:
            ci = np.where(ypool == c)[0]
            pick += list(rs.choice(ci, shots, replace=False))
        m = remember(Xpool[pick], ypool[pick])
        trials.append(hpc_acc(m, Xnov, ynov, knn=min(3, len(pick))))
    label = "1 each (one-shot)" if shots == 1 else f"{shots} each"
    print(f"  {label:>17} {np.mean(trials):>11.1%} ±{np.std(trials)*100:.0f}")
print()

# ── Experiment 1: does memory buy back the accuracy? (full data, depth 3) ─────
print(f"  MAIN TASK — all ten digits, depth 3:")
print()

L = 3
llm = train(L, "bp")
cortex = train(L, "dfa", fire=FIRE, wire=WIRE)
mem = remember(Xtr, ytr)

llm_a = cortex_acc(llm, Xte, yte)
cortex_a = cortex_acc(cortex, Xte, yte)
memonly_a = hpc_acc(mem, Xte, yte)
combo_a = combined_acc(cortex, mem, Xte, yte)

print("  CAN MEMORY CLOSE THE ACCURACY GAP?")
print(f"    LLM (dense, backprop)                  {llm_a:.1%}")
print(f"    Brain cortex alone (slow learner)      {cortex_a:.1%}")
print(f"    Brain hippocampus alone (recall only)  {memonly_a:.1%}")
print(f"    Brain cortex + hippocampus together    {combo_a:.1%}")
print(f"    gap to LLM: cortex {cortex_a:.1%} -> together {combo_a:.1%} "
      f"vs LLM {llm_a:.1%}")
print()

# ── Experiment 2: the low-data flip (depth 2) ─────────────────────────────────
print("  THE LOW-DATA WORLD BRAINS LIVE IN (depth 2):")
print("  Memory means learning from few examples. Does the brain win now?")
print(f"  {'images':>7} {'LLM':>7} {'cortex':>8} {'+memory':>8} {'winner':>8}")
print(f"  {'─'*42}")
flips = 0
for n in [50, 100, 200, 400, 800, 1400]:
    llm_n = train(2, "bp", ntr=n)
    cortex_n = train(2, "dfa", fire=FIRE, wire=WIRE, ntr=n)
    mem_n = remember(Xtr[:n], ytr[:n])
    la = cortex_acc(llm_n, Xte, yte)
    ca = cortex_acc(cortex_n, Xte, yte)
    ma = combined_acc(cortex_n, mem_n, Xte, yte)
    win = "Brain" if ma > la else ("LLM" if la > ma else "tie")
    if ma > la:
        flips += 1
    print(f"  {n:>7} {la:>7.1%} {ca:>8.1%} {ma:>8.1%} {win:>8}")
print()
print(f"  Brain+memory beat the LLM in {flips} of 6 data sizes.")
print()

# ── Experiment 3: why sparse codes make memory work ───────────────────────────
# Pattern separation: sparse memories overlap less, so they interfere less.
print("  WHY SPARSE CODES MAKE MEMORY WORK (pattern separation):")
print("  Average overlap between two stored memories, lower = more distinct -")


def mean_overlap(R):
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    s = R @ R.T
    n = R.shape[0]
    return (s.sum() - np.trace(s)) / (n * (n - 1))


raw_codes = Xtr[:500]                                   # store the raw pixels
dg_codes = dg_code(Xtr[:500])                           # store the sparse codes
print(f"    raw pixel memories      : {mean_overlap(raw_codes):.3f} overlap  (smear together)")
print(f"    dentate sparse memories : {mean_overlap(dg_codes):.3f} overlap  (stay distinct)")
print("=" * 68)
print()
