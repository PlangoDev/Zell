"""Hierarchical class->word readout, trained by the local delta rule (no backprop).

P(word) = P(class) * P(word | class). The vocabulary is split into n_classes
frequency-ordered classes; a sparse code scores n_classes class logits plus the
words of ONE class, so the cost is K*(n_classes + V/n_classes) instead of K*V.

ClassHierarchy is built once from the corpus frequency and SHARED across all
layers (it depends only on the vocabulary). Each layer owns its own Readout
(Wc, Ww, bc, bw). EMA is applied only to the small tensors (Wc, bc, bw); Ww is
too large to shadow on a T4, so it uses the raw master.
"""
import math

import torch

from . import kernels


class ClassHierarchy:
    """Frequency-ordered balanced word classes + unigram bias initializers.
    Built once on the setup block; identical across ranks (same freq input)."""

    def __init__(self, vocab, freq, n_classes, device):
        V, C = vocab, n_classes
        S = math.ceil(V / C)
        order = torch.argsort(freq, descending=True)
        word_order = torch.cat([order, torch.full((C * S - V,), V, dtype=torch.long)])
        slots = torch.arange(C * S)
        w2class = torch.zeros(V + 1, dtype=torch.long)
        w2within = torch.zeros(V + 1, dtype=torch.long)
        w2class[word_order] = slots // S
        w2within[word_order] = slots % S
        self.V, self.C, self.S = V, C, S
        self.class_members = word_order.view(C, S).to(device)        # [C, S]
        self.w2class = w2class.to(device)
        self.w2within = w2within.to(device)
        # unigram bias init: readout starts at the unigram baseline, not uniform
        bw = torch.full((V + 1,), -30.0, dtype=torch.float32)
        bw[:V] = torch.log(freq + 0.5)
        cf = torch.zeros(C, dtype=torch.float32)
        cf.index_add_(0, w2class[:V], freq)
        self.bw_init = bw.to(device)
        self.bc_init = torch.log(cf + 0.5).to(device)


class Readout:
    """One layer's class+word heads. Delta-rule updates; optional EMA for eval."""

    def __init__(self, n_gran, hier, cfg, device):
        self.cfg = cfg
        self.hier = hier
        self.device = device
        V, C = hier.V, hier.C
        self.Wc = torch.zeros(n_gran, C, dtype=torch.float32, device=device)
        self.Ww = torch.zeros(n_gran, V + 1, dtype=torch.float32, device=device)
        self.bc = hier.bc_init.clone()
        self.bw = hier.bw_init.clone()
        self._step = 0
        # EMA shadows for the cheap tensors only (Ww would double a ~6GB payload)
        self.use_ema = cfg.use_ema
        if self.use_ema:
            self.Wc_ema = self.Wc.clone()
            self.bc_ema = self.bc.clone()
            self.bw_ema = self.bw.clone()

    # ---- forward over the K active granules ----
    def heads(self, idx, val, cmem, eval_mode=False):
        Wc = self.Wc_ema if (eval_mode and self.use_ema) else self.Wc
        bc = self.bc_ema if (eval_mode and self.use_ema) else self.bc
        bw = self.bw_ema if (eval_mode and self.use_ema) else self.bw
        vk = val.unsqueeze(-1)                                  # [B,K,1]
        lc = bc + (vk * Wc[idx]).sum(1)                         # [B,C]
        Ww_g = self.Ww[idx.unsqueeze(-1), cmem.unsqueeze(1)]   # [B,K,S]
        lw = bw[cmem] + (vk * Ww_g).sum(1)                     # [B,S]
        return lc, lw

    @torch.no_grad()
    def logprob_true(self, idx, val, y, eval_mode=False):
        """log P(class)+log P(word|class) for the TRUE word y, per example [B]."""
        ce, within = self.hier.w2class[y], self.hier.w2within[y]
        lc, lw = self.heads(idx, val, self.hier.class_members[ce], eval_mode=eval_mode)
        ar = torch.arange(y.size(0), device=y.device)
        return lc.log_softmax(1)[ar, ce] + lw.log_softmax(1)[ar, within]

    # ---- local delta-rule update against the next-token target ----
    @torch.no_grad()
    def train_step(self, idx, val, y, st):
        """One delta-rule step on this layer's heads AND the true-word logprob,
        from a single heads() forward (the [B,K,S] gather is the expensive op, so
        we never recompute it for the mixture). idx [B,K], val [B,K], y [B], st =
        lr/batch. Returns logprob_true [B] (pre-update snapshot) for the mixture."""
        hier = self.hier
        K, S = self.cfg.k_active, hier.S
        B = y.size(0)
        ar = torch.arange(B, device=y.device)
        ce, within = hier.w2class[y], hier.w2within[y]
        cmem = hier.class_members[ce]                          # [B,S]
        lc, lw = self.heads(idx, val, cmem)
        # logprob of the true word (for the mixture), from these same logits
        logp = lc.log_softmax(1)[ar, ce] + lw.log_softmax(1)[ar, within]   # [B]
        # delta-rule error signals
        errc = lc.softmax(1); errc[ar, ce] -= 1.0             # [B,C]
        errw = lw.softmax(1); errw[ar, within] -= 1.0         # [B,S]
        vk = val.unsqueeze(-1)                                 # [B,K,1]
        # class head: one index_add over all (example, granule) pairs
        self.Wc.index_add_(0, idx.reshape(-1),
                           ((-st) * vk * errc.unsqueeze(1)).reshape(-1, errc.size(1)))
        # word head: flat 1D scatter_add (fast path, replaces 2D index_put)
        rows = idx.unsqueeze(-1).expand(B, K, S).reshape(-1)
        cols = cmem.unsqueeze(1).expand(B, K, S).reshape(-1)
        vals = ((-st) * vk * errw.unsqueeze(1)).reshape(-1)
        kernels.scatter_add_2d(self.Ww, rows, cols, vals)
        # biases
        self.bc.add_(errc.sum(0), alpha=-st)
        self.bw.index_put_((cmem.reshape(-1),), ((-st) * errw).reshape(-1), accumulate=True)
        return logp

    @torch.no_grad()
    def post_step(self):
        """Call once per training step (after all chunks): weight decay + EMA."""
        cfg = self.cfg
        self._step += 1
        if cfg.wd > 0 and self._step % cfg.decay_every == 0:
            keep = (1.0 - cfg.lr * cfg.wd) ** cfg.decay_every
            self.Ww.mul_(keep); self.Wc.mul_(keep)
        if self.use_ema and self._step % cfg.ema_every == 0:
            a = cfg.ema_alpha
            self.Wc_ema.mul_(a).add_(self.Wc, alpha=1 - a)
            self.bc_ema.mul_(a).add_(self.bc, alpha=1 - a)
            self.bw_ema.mul_(a).add_(self.bw, alpha=1 - a)

    # ---- tensors that must be averaged across ranks (local-SGD) ----
    def sync_tensors(self):
        """Return (big_fp32_tensors, small_fp32_tensors). Ww is big (bf16 transport);
        Wc/bc/bw are small/precision-critical (fp32 transport)."""
        return [self.Ww], [self.Wc, self.bc, self.bw]
