import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Test 004 — The Wall, and the Trick That Climbs It
#
# This is the big one. Three things in one test:
#
#   1. THE WALL. A task that a flat (one-layer) network can NEVER solve,
#      no matter how long it trains. This proves depth is not optional.
#
#   2. THE TEXTBOOK CLIMB. A deep network with a hidden layer, trained by
#      backpropagation. It solves the task. But backprop needs a trick a
#      real brain cannot do: sending error signals backward through the
#      exact same wires used going forward (the "weight transport problem").
#
#   3. THE BRAIN-PLAUSIBLE CLIMB. The same deep network, but the error is
#      sent backward through FIXED RANDOM wires instead. This is Feedback
#      Alignment (Lillicrap et al. 2016). It should not work. It does. The
#      forward weights quietly rotate to line up with the random feedback.
#
#   Plus: we fix Sparky's wobble from test 003 with a calm-down rule
#   (a learning rate that shrinks over time) and show it settle.
# ─────────────────────────────────────────────────────────────────────────────

SEED      = 11
HIDDEN    = 10
N_TRAIN   = 800
N_TEST    = 500
EPOCHS    = 150
SPREAD    = 0.20   # how fuzzy the input clouds are


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ── The XOR task ──────────────────────────────────────────────────────────────
# Two switches, each roughly OFF (0) or ON (1) with a little wobble.
# The lamp turns on (label 1) only when EXACTLY ONE switch is on.
#
#   (off, off) -> 0      (off, on) -> 1
#   (on,  on ) -> 0      (on,  off) -> 1
#
# The catch that makes this impossible for a flat network:
# the two "lamp off" corners sit on one diagonal, the two "lamp on"
# corners on the other. Both classes have the SAME center point (0.5, 0.5).
# No straight line can split two clouds that share a center. A flat network
# is stuck at a coin flip forever.

CORNERS = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def make_data(n, rng):
    X, y = [], []
    for _ in range(n):
        (cx, cy), label = CORNERS[rng.integers(0, 4)]
        X.append([cx + rng.normal(0, SPREAD), cy + rng.normal(0, SPREAD)])
        y.append(label)
    return np.array(X), np.array(y, dtype=float)


# ── A flat network (no hidden layer) ──────────────────────────────────────────
# One layer of weights straight from the two inputs to the answer.
# This is exactly the kind of network from tests 001 to 003.

def train_flat(Xtr, ytr, Xte, yte, rng):
    w = rng.standard_normal(2) * 0.5
    b = 0.0
    lr = 0.3
    curve = []
    for _ in range(EPOCHS):
        for i in rng.permutation(len(Xtr)):
            x, y = Xtr[i], ytr[i]
            out = sigmoid(w @ x + b)
            err = out - y
            w -= lr * err * x
            b -= lr * err
        pred = sigmoid(Xte @ w + b) >= 0.5
        curve.append((pred == yte).mean())
    return curve


# ── A deep network (one hidden layer) ─────────────────────────────────────────
# inputs -> HIDDEN neurons (ReLU) -> one answer neuron (sigmoid)
#
# The ONLY difference between the textbook brain and the plausible brain is
# a single line: how the hidden layer hears about the output's mistake.
#
#   backprop          : dh = W2 * error   (reuses the forward weights — the
#                                           biologically impossible trick)
#   feedback_alignment: dh = B  * error   (uses fixed RANDOM weights instead)
#
# Everything else is identical. Watch that one line decide everything.

def train_deep(method, Xtr, ytr, Xte, yte, rng, lr0=0.3, decay=0.0):
    W1 = rng.standard_normal((HIDDEN, 2)) * 0.5   # input  -> hidden
    b1 = np.zeros(HIDDEN)
    W2 = rng.standard_normal(HIDDEN) * 0.02       # hidden -> output (start near zero)
    b2 = 0.0
    B  = rng.standard_normal(HIDDEN)              # fixed random feedback wires

    curve, align, active = [], [], []

    # fine-grained trace of how fast the forward weights rotate to line up
    # with the random feedback, sampled within the very first epoch
    def cos_WB():
        return (W2 @ B) / (np.linalg.norm(W2) * np.linalg.norm(B) + 1e-9)

    milestones = [0, 20, 50, 100, 200, 400, 800]
    early = [(0, cos_WB())]   # record alignment before any training
    mi, seen = 1, 0

    for epoch in range(EPOCHS):
        lr = lr0 / (1 + decay * epoch)            # the calm-down rule
        for i in rng.permutation(len(Xtr)):
            x, y = Xtr[i], ytr[i]

            # forward pass
            h_pre = W1 @ x + b1
            h     = np.maximum(0, h_pre)          # ReLU: silent neurons stay 0
            out   = sigmoid(W2 @ h + b2)

            # how wrong was the answer
            err = out - y

            # output layer learns the normal way
            gW2, gb2 = err * h, err

            # hidden layer's error signal — THE ONE LINE THAT DIFFERS
            if method == "backprop":
                dh = W2 * err                     # weight transport
            else:
                dh = B * err                      # random feedback

            dh_pre = dh * (h_pre > 0)             # back through the ReLU gate
            gW1, gb1 = np.outer(dh_pre, x), dh_pre

            # apply updates
            W2 -= lr * gW2; b2 -= lr * gb2
            W1 -= lr * gW1; b1 -= lr * gb1

            seen += 1
            if mi < len(milestones) and seen == milestones[mi]:
                early.append((seen, cos_WB()))
                mi += 1

        # evaluate on held-out data
        H_te = np.maximum(0, Xte @ W1.T + b1)
        pred = sigmoid(H_te @ W2 + b2) >= 0.5
        curve.append((pred == yte).mean())
        active.append((H_te > 0).mean())          # fraction of hidden neurons firing

        align.append(cos_WB())   # alignment at the end of each epoch

    return curve, align, active, early


# ── Run everything ────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
X_train, y_train = make_data(N_TRAIN, rng)
X_test,  y_test  = make_data(N_TEST, rng)

flat = train_flat(X_train, y_train, X_test, y_test, np.random.default_rng(SEED))
bp, bp_align, bp_active, _ = train_deep("backprop", X_train, y_train, X_test, y_test,
                                        np.random.default_rng(SEED), lr0=0.3, decay=0.0)
fa_wobble, fw_align, _, _ = train_deep("feedback_alignment", X_train, y_train, X_test, y_test,
                                       np.random.default_rng(SEED), lr0=0.3, decay=0.0)
fa_calm, fc_align, fc_active, fc_early = train_deep("feedback_alignment", X_train, y_train, X_test, y_test,
                                                    np.random.default_rng(SEED), lr0=0.3, decay=0.05)


def last10(c):  # average of final 10 epochs = "settled" score
    return np.mean(c[-10:])


def wobble(c):  # how much the final 30 epochs bounce around
    return np.std(c[-30:]) * 100


print()
print("=" * 60)
print("  Test 004 — The Wall, and the Trick That Climbs It")
print("=" * 60)
print()
print("  Task: noisy XOR. Lamp turns on only if exactly one switch is on.")
print("  Both answer-classes share the same center, so no straight")
print("  line can ever separate them.")
print()
print(f"  {'Network':<34} {'Settled':>9} {'Best':>7}")
print(f"  {'─'*52}")
print(f"  {'Flat, one layer (the wall)':<34} {last10(flat):>8.1%} {max(flat):>7.1%}")
print(f"  {'Deep + backprop (textbook)':<34} {last10(bp):>8.1%} {max(bp):>7.1%}")
print(f"  {'Deep + feedback align (no calm)':<34} {last10(fa_wobble):>8.1%} {max(fa_wobble):>7.1%}")
print(f"  {'Deep + feedback align (calmed)':<34} {last10(fa_calm):>8.1%} {max(fa_calm):>7.1%}")
print()
print("  Wobble in the final stretch (lower = steadier):")
print(f"    feedback align, no calm-down : {wobble(fa_wobble):.2f}")
print(f"    feedback align, with calm-down: {wobble(fa_calm):.2f}")
print()
print("  Forward weights rotating to line up with the RANDOM feedback,")
print("  sampled within the first epoch (cos angle climbs from ~0):")
print(f"    {'samples seen':>13} {'cos angle':>10}")
for s, c in fc_early:
    print(f"    {s:>13} {c:>10.2f}")
print(f"    {'final epoch':>13} {fc_align[-1]:>10.2f}  (holds steady)")
print()
print(f"  Hidden neurons firing at the end: {fc_active[-1]:.0%} "
      f"(the rest stay silent and cost nothing)")
print()
print("  Learning curves (accuracy each checkpoint):")
print(f"    {'epoch':>6} {'flat':>7} {'backprop':>9} {'FA calm':>8}")
for e in [0, 4, 9, 24, 49, 99, EPOCHS - 1]:
    print(f"    {e+1:>6} {flat[e]:>7.0%} {bp[e]:>9.0%} {fa_calm[e]:>8.0%}")
print("=" * 60)
print()
