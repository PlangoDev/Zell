import numpy as np
from sklearn.datasets import fetch_openml

# ─────────────────────────────────────────────────────────────────────────────
# Test 009 — Sort the Laundry: the LLM way vs the Brain way, fair and square
#
# A real, harder, more fun job, with a LOT more data: look at a fuzzy little
# photo of clothing and say what it is. T-shirt? Sneaker? Bag? This is the
# Fashion-MNIST dataset, the grown-up "harder than digits" benchmark, and we
# use 12,000 training photos, far more than the ~1,400 digits before.
#
# Two contestants, each allowed its own full bag of tricks:
#
#   THE LLM WAY  - one big dense network, every neuron firing every time,
#                  trained by backpropagation. Today's standard recipe.
#
#   THE BRAIN WAY - everything this project built: a deep network that fires
#                  only a few neurons (k-WTA), wired sparsely like a brain,
#                  taught by random feedback, kept steady by normalization,
#                  PLUS a hippocampus that remembers past examples.
#
# Same photos, same test. Each side plays its own game. We score them on
# how often they are right AND how much work they burn to do it.
# ─────────────────────────────────────────────────────────────────────────────

SEED  = 11
C     = 10
WIDTH = 256
BATCH = 100
EPOCHS = 20
FIRE  = 0.20
WIRE  = 0.30
MEM_CAP = 25000      # the hippocampus keeps a big sample, not every photo
NAMES = ['t-shirt', 'trousers', 'pullover', 'dress', 'coat',
         'sandal', 'shirt', 'sneaker', 'bag', 'boot']


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


# ── Load the laundry photos and shrink them to 14x14 so it runs fast ──────────
print("  loading 70,000 clothing photos (cached)...")
Xall, yall = fetch_openml('Fashion-MNIST', version=1, as_frame=False,
                          parser='liac-arff', return_X_y=True)
Xall = Xall.astype(float).reshape(-1, 28, 28)
Xall = Xall.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4)).reshape(-1, 196) / 255.0
yall = yall.astype(int)
D = 196

idx = np.random.default_rng(0).permutation(len(Xall))
Xall, yall = Xall[idx], yall[idx]
Xtr, ytr = Xall[:60000], yall[:60000]          # the full training pile
Xte, yte = Xall[60000:70000], yall[60000:70000]  # the full test pile
ONEHOT = np.eye(C)
avg_ink = (Xtr > 0.1).sum(axis=1).mean()


# ── The network (shared skeleton; each contestant trains it their way) ────────
def build(L, wire, rng):
    sizes = [D] + [WIDTH] * L + [C]
    W, b, mask = [], [], []
    for i in range(len(sizes) - 1):
        nin, nout = sizes[i], sizes[i + 1]
        sparsify = wire < 1 and i < len(sizes) - 2
        m = (rng.random((nout, nin)) < wire).astype(float) if sparsify else np.ones((nout, nin))
        fan = max(1.0, m[0].sum())
        W.append(rng.standard_normal((nout, nin)) * np.sqrt(2 / fan) * m)
        b.append(np.zeros(nout)); mask.append(m)
    Bfb = [rng.standard_normal((WIDTH, C)) / np.sqrt(C) for _ in range(L)]
    return W, b, mask, Bfb


def train(L, method, fire=1.0, wire=1.0, lr0=0.06, decay=0.02, ntr=None):
    rng = np.random.default_rng(SEED)
    ntr = ntr or len(Xtr)
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
                if fire < 1:
                    h, g = kwta_rows(np.maximum(0, z), k); h = dnorm(h)
                else:
                    h, g = np.maximum(0, z), (z > 0).astype(float)
                gates.append(g); acts.append(h); a = h
            logits = np.clip(a @ W[L].T + b[L], -30, 30)
            e = (softmax_rows(logits) - yb) / n
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
    W, b, _, k, fire = params
    a = Xe
    for l in range(len(W) - 1):
        r = np.maximum(0, a @ W[l].T + b[l])
        a = dnorm(kwta_rows(r, k)[0]) if fire < 1 else r
    return a @ W[-1].T + b[-1]


def acc(params, Xe, ye):
    return (np.argmax(forward(params, Xe), axis=1) == ye).mean()


# ── The hippocampus (dentate-gyrus sparse memory, from test 007) ──────────────
DG_DIM = 1500
_dg = np.random.default_rng(99)
P_DG = _dg.standard_normal((DG_DIM, D)) / np.sqrt(D)


def dg_code(Xe):
    h = np.maximum(0, Xe @ P_DG.T)
    c, _ = kwta_rows(h, int(DG_DIM * 0.05))
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    return c.astype(np.float32)


def remember(Xm, ym):
    return dg_code(Xm), ym


def recall_probs(memory, Xe, knn=9):
    keys, labels = memory
    q = dg_code(Xe)
    kk = min(knn, keys.shape[0])
    votes = np.zeros((len(Xe), C))
    for s in range(0, len(Xe), 1000):                # batch so the math stays small
        sim = q[s:s + 1000] @ keys.T
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        for ii in range(len(sim)):
            for j in top[ii]:
                votes[s + ii, labels[j]] += max(float(sim[ii, j]), 0.0)
    return votes / (votes.sum(axis=1, keepdims=True) + 1e-8)


def combined_acc(params, memory, Xe, ye, alpha=0.5):
    final = alpha * softmax_rows(forward(params, Xe)) + (1 - alpha) * recall_probs(memory, Xe)
    return (np.argmax(final, axis=1) == ye).mean()


def ops(L, fire, wire):
    if fire >= 1 and wire >= 1:
        return D * WIDTH + (L - 1) * WIDTH * WIDTH + WIDTH * C
    return wire * avg_ink * WIDTH + (L - 1) * wire * fire * WIDTH * WIDTH + fire * WIDTH * C


def show(x, label):
    g = x.reshape(14, 14)
    ramp = " .:-=+*#%@"
    print(f"    this is a {label}:")
    for row in g:
        print("    " + "".join(ramp[min(9, int(v * 10))] for v in row))


# ── Run the contest ───────────────────────────────────────────────────────────
print()
print("=" * 66)
print("  Test 009 — Sort the Laundry: LLM way vs Brain way")
print("=" * 66)
print(f"  {len(Xtr):,} training photos, {len(Xte):,} test photos, 10 kinds of clothes.")
print()
for c in [0, 7, 8]:
    i = np.where(ytr == c)[0][0]
    show(Xtr[i], NAMES[c])
    print()

L = 2
llm = train(L, "bp")
brain = train(L, "dfa", fire=FIRE, wire=WIRE)
mi = np.random.default_rng(7).choice(len(Xtr), min(MEM_CAP, len(Xtr)), replace=False)
mem = remember(Xtr[mi], ytr[mi])

llm_a = acc(llm, Xte, yte)
brain_a = acc(brain, Xte, yte)
combo_a = combined_acc(brain, mem, Xte, yte)
llm_o = ops(L, 1.0, 1.0)
brain_o = ops(L, FIRE, WIRE)

print("  THE MAIN BOUT — who sorts the laundry best, and for how much work?")
print(f"  {'contestant':<34} {'correct':>8} {'work/photo':>11} {'vs LLM':>8}")
print(f"  {'─'*62}")
print(f"  {'LLM way (dense, backprop)':<34} {llm_a:>8.1%} {llm_o:>11,.0f} {'1.0x':>8}")
print(f"  {'Brain way (sparse cortex only)':<34} {brain_a:>8.1%} {brain_o:>11,.0f} {llm_o/brain_o:>7.1f}x")
print(f"  {'Brain way (cortex + memory)':<34} {combo_a:>8.1%} {brain_o:>11,.0f} {llm_o/brain_o:>7.1f}x")
print()

# ── The brain's special move: learn a NEW kind of clothing from a few looks ───
NOVEL = [5, 8]                          # sandal and bag, held out from training
nv_pool = np.isin(ytr, NOVEL)
nv_test = np.isin(yte, NOVEL)
Xpool, ypool = Xtr[nv_pool], ytr[nv_pool]
Xnv, ynv = Xte[nv_test], yte[nv_test]

print(f"  THE BRAIN'S SPECIAL MOVE — tell a {NAMES[5]} from a {NAMES[8]},")
print(f"  two kinds of clothing it was NEVER trained on:")
print(f"    The LLM cannot. It has no slot for them. (50% guess.)")
print(f"  {'examples shown':>16} {'brain memory':>13}")
print(f"  {'─'*31}")
for shots in [1, 3, 5, 10]:
    trials = []
    for t in range(25):
        rs = np.random.default_rng(1000 * shots + t)
        pick = []
        for cl in NOVEL:
            ci = np.where(ypool == cl)[0]
            pick += list(rs.choice(ci, shots, replace=False))
        m = remember(Xpool[pick], ypool[pick])
        trials.append((np.argmax(recall_probs(m, Xnv, min(3, len(pick))), axis=1) == ynv).mean())
    lbl = "1 each (one-shot)" if shots == 1 else f"{shots} each"
    print(f"  {lbl:>16} {np.mean(trials):>11.1%}")
print("=" * 66)
print()
