"""CerebellarLayer — one locally-trained layer of the deep stack.

A layer = a GranuleField (sparse expansion + coder) + a Readout (class->word
head). It has its OWN next-token objective and is trained by the local delta
rule; no gradient ever flows between layers. To compose into depth, each layer
emits a RELAY vector: a fixed random projection of its sparse code, concatenated
with the (bound) context, which becomes the next layer's input. The relay is a
Johnson-Lindenstrauss projection (information-preserving, fixed, K*relay_dim
MACs), so deeper layers build abstractions on the sparse codes below them while
a context skip keeps surface signal available at every depth.
"""
import math

import torch

from .granules import GranuleField
from .readout import Readout


class CerebellarLayer:
    def __init__(self, depth, in_dim, ctx_dim, hier, cfg, device, act_dtype):
        self.depth = depth
        self.cfg = cfg
        self.device = device
        self.act_dtype = act_dtype
        self.in_dim = in_dim
        self.ctx_dim = ctx_dim
        # distinct seed per layer so wirings/codes differ (decorrelated experts)
        seed = cfg.seed + 1000 * (depth + 1)
        self.field = GranuleField(in_dim, cfg, device, act_dtype, seed)
        self.readout = Readout(cfg.n_gran, hier, cfg, device)
        # fixed random relay projection: sparse code -> relay_dim dense vector
        g = torch.Generator(device="cpu").manual_seed(seed + 99)
        R = torch.randn(cfg.n_gran, cfg.relay_dim, generator=g)
        R *= 1.0 / math.sqrt(cfg.relay_dim)
        self.relay_R = R.to(device).to(act_dtype)
        self._relay_scale = math.sqrt(cfg.relay_dim)

    # ---- encode this layer's input into a sparse code ----
    def encode(self, x):
        return self.field.encode(x)

    # ---- the relay handed to the next layer ----
    def relay(self, idx, val):
        """[B, relay_dim], an L2-normalized random projection of the sparse code.
        Normalizing to a fixed scale keeps the next layer's activation magnitudes
        (and thus its kWTA margins / delta-rule step sizes) stable with depth."""
        vk = val.to(self.act_dtype).unsqueeze(-1)              # [B,K,1]
        r = (vk * self.relay_R[idx]).sum(1)                    # [B, relay_dim]
        r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-6) * self._relay_scale
        return r

    # ---- build the next layer's input: [relay ; bound context] ----
    def next_input(self, idx, val, bound_ctx):
        return torch.cat([self.relay(idx, val), bound_ctx], dim=1)

    # ---- training-time setup helpers (delegate to the field) ----
    def build_proj(self):
        self.field.build_proj()

    def learn_granules(self, sample_inputs):
        self.field.learn_granules(sample_inputs)

    def calibrate_bias(self, sample_inputs):
        self.field.calibrate_bias(sample_inputs)

    def refine_online(self, x):
        self.field.refine_online(x)
