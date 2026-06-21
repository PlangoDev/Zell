"""Token codes + hyperdimensional context binding.

v13 concatenated the CTX token codes into a [B, ctx*code_dim] vector, so each
fixed-fan-in granule saw a shrinking fraction of the input as ctx grew (the
CTX=4 regression in test_012). Hyperdimensional binding fixes this: every token
code is *bound* to its position by an element-wise sign flip (a cheap, invertible
vector-symbolic operation), then all positions are summed under several temporal
decay scales. The result is a single [B, code_dim] vector REGARDLESS of ctx, so:

  * granule fan_in covers a constant fraction of the input no matter how long
    the context is -> ctx can grow to 32-64 without diluting features;
  * the granule matmul shrinks from [B, ctx*code_dim] to [B, code_dim];
  * recent tokens (fast decay) stay sharp while distant tokens (slow decay)
    contribute gist -- a temporal hierarchy, like cortical time constants.

Neuroscience: entorhinal/hippocampal circuits bind 'what' (identity) to 'where/
when' (position/phase) by exactly this kind of conjunctive code.
"""
import torch


class ContextCoder:
    """Owns the fixed random token codes and position-binding codes. All tensors
    are seed-deterministic so every rank builds an identical coder with no
    communication."""

    def __init__(self, vocab, cfg, device, act_dtype):
        self.cfg = cfg
        self.V = vocab
        self.device = device
        self.act_dtype = act_dtype
        D = cfg.code_dim

        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        # fixed random token codes (dentate-gyrus style random projection)
        self.codes = torch.randn(vocab, D, generator=g).to(device).to(act_dtype)

        if cfg.hd_binding:
            gp = torch.Generator(device="cpu").manual_seed(cfg.seed + 42)
            # per-position +/-1 sign code: binding is an element-wise multiply,
            # which is its own inverse and preserves the code's magnitude.
            signs = torch.randint(0, 2, (cfg.ctx, D), generator=gp).float() * 2 - 1
            self.pos_codes = signs.to(device).to(act_dtype)            # [ctx, D]
            # precompute the [n_scales, ctx] decay weight matrix (most-recent
            # token is position ctx-1, so weight by distance-from-the-end).
            dist = torch.arange(cfg.ctx - 1, -1, -1, dtype=torch.float32)  # [ctx]
            rows = []
            for rate in cfg.decay_rates:
                rows.append(torch.exp(-dist * rate))
            self.decay = torch.stack(rows, 0).to(device).to(act_dtype)  # [S, ctx]
            self.out_dim = D
        else:
            self.pos_codes = None
            self.decay = None
            self.out_dim = cfg.ctx * D

    def bind(self, X):
        """X [B, ctx] token ids -> bound context vector.

        hd_binding: returns [B, code_dim] (sum over positions of code*sign, under
        each decay scale, scales averaged). Otherwise: the v13 concat [B, ctx*D].
        """
        cfg = self.cfg
        if not cfg.hd_binding:
            return self.codes[X].reshape(X.size(0), self.out_dim)

        tok = self.codes[X]                                   # [B, ctx, D]
        L = X.size(1)
        bound = tok * self.pos_codes[:L]                      # bind token x position
        # weighted sum across positions for each decay scale, then average scales.
        # decay[:, :L] is [S, L]; einsum over the position axis.
        out = torch.einsum("sl,bld->bd", self.decay[:, :L], bound)
        return out / self.decay.size(0)                       # [B, D]
