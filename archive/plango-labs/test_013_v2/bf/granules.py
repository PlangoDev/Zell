"""GranuleField — one cerebellar layer's fixed expansion + sparse coder.

Operates on a generic dense input vector [B, in_dim] (so the same field serves
layer 0, whose input is the bound context, and deeper layers, whose input is the
previous layer's densified relay). Owns:

  gidx   [G, fan_in]  int   fixed sparse wiring (never learned, never synced)
  gwt    [G, fan_in]  fp32  competitively-learned granule weights (k-means)
  gbias  [G]          act   per-granule mean activation, subtracted (homeostasis)
  proj_t [G, in_dim]  act   dense transpose of the wiring for the fused matmul
  neg_gbias [G]       act   -gbias, folded into the fused linear

Learning here is competitive (online k-means under WTA) -- NOT backprop. The
wiring is fixed; only the weights tune so the expansion tiles the data manifold.
"""
import math

import torch

from . import kernels


class GranuleField:
    def __init__(self, in_dim, cfg, device, act_dtype, seed):
        self.cfg = cfg
        self.in_dim = in_dim
        self.device = device
        self.act_dtype = act_dtype
        G, F_ = cfg.n_gran, cfg.fan_in

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.gidx = torch.randint(0, in_dim, (G, F_), generator=g).to(device)
        gwt = torch.randn(G, F_, generator=g)
        gwt *= math.sqrt(F_) / gwt.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.gwt = gwt.to(device)                              # fp32 master
        self.gbias = torch.zeros(G, device=device).to(act_dtype)
        self.proj_t = None
        self.neg_gbias = None

    # ---- dense projection (built from the fixed wiring + learned weights) ----
    def build_proj(self):
        """Scatter the fan_in weights into a dense [in_dim, G] matrix, store its
        transpose [G, in_dim] for F.linear, and cache -gbias. accumulate so
        duplicate (input, granule) pairs sum, matching the sparse gather."""
        cfg = self.cfg
        proj = torch.zeros(self.in_dim, cfg.n_gran, dtype=self.act_dtype, device=self.device)
        cols = torch.arange(cfg.n_gran, device=self.device).repeat_interleave(cfg.fan_in)
        proj.index_put_((self.gidx.reshape(-1), cols),
                        self.gwt.to(self.act_dtype).reshape(-1), accumulate=True)
        self.proj_t = proj.t().contiguous()                   # [G, in_dim]
        self.neg_gbias = (-self.gbias).contiguous()

    # ---- sparse encode (fused matmul + two-level approx topk) ----
    def act(self, x):
        return kernels.fused_act(x, self.proj_t, self.neg_gbias)   # [B, G]

    def encode(self, x):
        a = self.act(x)
        return kernels.approx_topk(a, self.cfg.topk_groups, self.cfg.topk_per_group,
                                   self.cfg.k_active)

    # ---- helpers for competitive learning on raw input vectors ----
    def _xg(self, x):
        """x [B, in_dim] -> the fan_in-sampled inputs per granule [B, G, fan_in]."""
        return x[:, self.gidx]

    def _competitive_update(self, x, eta):
        """One competitive (WTA k-means) pass over input batch x [B, in_dim]."""
        cfg = self.cfg
        xg = self._xg(x.float())                               # [B, G, fan_in]
        a = (xg * self.gwt).sum(-1)                            # [B, G]
        _, win = a.topk(cfg.k_active, dim=1)                   # [B, K]
        B = x.size(0)
        sel = xg.gather(1, win.unsqueeze(-1).expand(B, cfg.k_active, cfg.fan_in))
        gids = win.reshape(-1)
        # count-normalized move: a granule that wins m times moves ONCE toward the
        # mean of its m activating inputs (summing m raw deltas overshoots by ~m).
        sums = torch.zeros_like(self.gwt)
        sums.index_add_(0, gids, sel.reshape(-1, cfg.fan_in))
        counts = torch.zeros(cfg.n_gran, device=self.gwt.device)
        counts.index_add_(0, gids, torch.ones_like(gids, dtype=counts.dtype))
        touched = counts > 0
        mean_sel = sums[touched] / counts[touched].unsqueeze(1)
        self.gwt[touched] += eta * (mean_sel - self.gwt[touched])
        self.gwt *= math.sqrt(cfg.fan_in) / self.gwt.norm(dim=1, keepdim=True).clamp_min(1e-6)

    def learn_granules(self, sample_inputs):
        """Initial competitive learning over a pool of input vectors
        sample_inputs [N, in_dim] (already on device). Multiple annealed passes."""
        cfg = self.cfg
        N = sample_inputs.size(0)
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7)
        for p in range(cfg.feat_passes):
            eta = cfg.feat_eta * (1 - p / cfg.feat_passes)
            perm = torch.randperm(N, generator=gen)[:min(cfg.feat_sample, N)]
            for i in range(0, perm.numel(), 256):
                idx = perm[i:i + 256].to(sample_inputs.device)
                self._competitive_update(sample_inputs[idx], eta)

    def refine_online(self, x):
        """Lightweight competitive pass on a live training batch x, then rebuild
        the projection. Chunked at 256 so the [chunk, G, fan_in] transient stays
        bounded (a single 4k-row pass would allocate tens of GB). Lets granules
        keep adapting across all tokens instead of freezing after setup."""
        n = min(self.cfg.refine_sample, x.size(0))
        for i in range(0, n, 256):
            self._competitive_update(x[i:min(i + 256, n)], self.cfg.refine_eta)
        self.build_proj()

    def calibrate_bias(self, sample_inputs):
        """Set gbias to the mean granule activation over a sample, so no granule
        hogs the code and dead granules revive (intrinsic plasticity)."""
        n = sample_inputs.size(0)
        self.gbias = torch.zeros(self.cfg.n_gran, device=self.device).to(self.act_dtype)
        acc = torch.zeros(self.cfg.n_gran, device=self.device)
        cnt = 0
        for i in range(0, n, 512):
            xb = sample_inputs[i:min(i + 512, n)]
            acc += self.act(xb).float().sum(0)
            cnt += xb.size(0)
        self.gbias = (acc / max(cnt, 1)).to(self.act_dtype)
        self.neg_gbias = (-self.gbias).contiguous()
