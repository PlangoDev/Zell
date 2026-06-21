"""DeepBrain — a stack of locally-trained cerebellar layers + a learned mixture.

Architecture (no backprop anywhere):

  X (ctx token ids)
    -> ContextCoder.bind  ->  bound [B, code_dim]
    -> layer 0  (granules -> kWTA -> readout_0 predicts next token)
         relay_0 = proj(sparse_0);  x_1 = [relay_0 ; bound]
    -> layer 1  (predicts next token from x_1)        ... and so on
    -> layer L-1

Each layer has its OWN next-token objective (deep supervision) and is trained by
the delta rule on its own readout. The final distribution is a probabilistic
MIXTURE of the layers:  P(y) = sum_l w_l P_l(y),  with w = softmax(mixture logits)
learned by a local EM-style rule (responsibility minus prior). Early layers
capture surface n-gram structure; deeper layers, fed the sparse codes below them,
capture longer-range structure.
"""
import math

import torch

from .codes import ContextCoder
from .readout import ClassHierarchy
from .layer import CerebellarLayer


class DeepBrain:
    def __init__(self, vocab, cfg, device):
        self.cfg = cfg
        self.V = vocab
        self.device = device
        # fp16 tensor-core math is CUDA-only; CPU falls back to fp32.
        self.act_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        self.coder = ContextCoder(vocab, cfg, device, self.act_dtype)
        self.ctx_dim = self.coder.out_dim
        self.hier = None                       # built in setup() once freq is known
        self.layers = []                       # built in setup() (needs hierarchy)
        # mixture logits over layers (uniform prior); learned by local rule.
        self.mix = torch.zeros(cfg.n_layers, dtype=torch.float32, device=device)

    # ───────────────────────── setup (shared, deterministic) ────────────────
    def _sample_contexts(self, block, m):
        """Gather m random [ctx] windows from a 1D token block -> X [m, ctx]."""
        n = block.numel() - self.cfg.ctx - 1
        gen = torch.Generator(device="cpu").manual_seed(self.cfg.seed + 5)
        pos = torch.randint(0, n, (min(m, n),), generator=gen).to(block.device)
        off = torch.arange(self.cfg.ctx, device=block.device)
        return block[pos[:, None] + off].long()

    def setup(self, setup_block):
        """Build the class hierarchy and every layer (competitive features, bias,
        projection) from the global head of the corpus. Layers are built in order
        because layer L's input depends on layer L-1's relay."""
        cfg = self.cfg
        # word frequency -> class hierarchy + unigram biases
        freq = torch.bincount(setup_block.cpu(), minlength=self.V + 1)[:self.V].float()
        self.hier = ClassHierarchy(self.V, freq, cfg.n_classes, self.device)

        # build the layers (input widths: layer0 = ctx_dim; deeper = relay+ctx)
        self.layers = []
        for d in range(cfg.n_layers):
            in_dim = self.ctx_dim if d == 0 else (cfg.relay_dim + self.ctx_dim)
            self.layers.append(CerebellarLayer(d, in_dim, self.ctx_dim, self.hier,
                                               cfg, self.device, self.act_dtype))

        # a pool of sample inputs to learn features against, flowed up the stack
        X = self._sample_contexts(setup_block, cfg.feat_sample)
        bound = self.coder.bind(X)                              # [M, code_dim]
        x = bound
        for d, layer in enumerate(self.layers):
            if cfg.learn_feat:
                layer.learn_granules(x)
            layer.build_proj()
            layer.calibrate_bias(x)
            if d < cfg.n_layers - 1:
                idx, val = layer.encode(x)
                x = layer.next_input(idx, val, bound)

    # ───────────────────────── forward through the stack ────────────────────
    def _encode_stack(self, X):
        """Return the per-layer sparse codes [(idx,val), ...] and the bound ctx."""
        bound = self.coder.bind(X)
        codes = []
        x = bound
        for d, layer in enumerate(self.layers):
            idx, val = layer.encode(x)
            codes.append((idx, val))
            if d < self.cfg.n_layers - 1:
                x = layer.next_input(idx, val, bound)
        return codes, bound

    def _mixture_logprob(self, codes, y, eval_mode=False):
        """log P(y) under the layer mixture, and the per-layer logprobs [B,L]."""
        per = []
        for layer, (idx, val) in zip(self.layers, codes):
            per.append(layer.readout.logprob_true(idx, val, y, eval_mode=eval_mode))
        lp = torch.stack(per, dim=1)                            # [B, L]
        logw = torch.log_softmax(self.mix, dim=0).unsqueeze(0)  # [1, L]
        mixed = torch.logsumexp(lp + logw, dim=1)              # [B]
        return mixed, lp

    # ───────────────────────────── training step ───────────────────────────
    @torch.no_grad()
    def step(self, X, y):
        cfg = self.cfg
        B = X.size(0)
        st = cfg.lr / B
        chunk = max(1, min(cfg.step_chunk, B))
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            Xc, yc = X[lo:hi], y[lo:hi]
            codes, _ = self._encode_stack(Xc)
            # 1) each layer trains its OWN readout against the next token (local
            #    deep supervision) and returns its true-word logprob in one pass
            per = [layer.readout.train_step(idx, val, yc, st)
                   for layer, (idx, val) in zip(self.layers, codes)]
            # 2) the mixture learns which layers to trust (EM responsibilities):
            #    grad of log P(y)=logsumexp(logw+logp_l) wrt mix logits is r - w
            lp = torch.stack(per, dim=1)                       # [bc, L]
            logw = torch.log_softmax(self.mix, dim=0)         # [L]
            resp = torch.softmax(lp + logw.unsqueeze(0), dim=1)  # [bc, L]
            w = torch.softmax(self.mix, dim=0)                # [L]
            self.mix += cfg.mix_lr * (resp.mean(0) - w)
        for layer in self.layers:
            layer.readout.post_step()

    def train_on_block(self, sampler, n_steps):
        for _ in range(n_steps):
            X, y = sampler.next_xy()
            self.step(X, y)
        return n_steps * self.cfg.batch

    # ───────────────────────── online refinement ───────────────────────────
    @torch.no_grad()
    def refine(self, sampler):
        """Lightweight competitive refinement of every layer's granules on a live
        batch, flowed up the stack so deeper layers refine on current relays."""
        X, _ = sampler.next_xy()
        bound = self.coder.bind(X)
        x = bound
        for d, layer in enumerate(self.layers):
            layer.refine_online(x)
            if d < self.cfg.n_layers - 1:
                idx, val = layer.encode(x)
                x = layer.next_input(idx, val, bound)

    # ───────────────────────────── evaluation ──────────────────────────────
    @torch.no_grad()
    def eval_ppl(self, stream, eval_tokens):
        cfg = self.cfg
        ids = stream[:eval_tokens]
        off = torch.arange(cfg.ctx, device=ids.device)
        nll = torch.zeros((), device=ids.device)
        ntok = 0
        last = ids.numel() - cfg.ctx - 1
        for i in range(0, last, cfg.batch):
            p = torch.arange(i, min(i + cfg.batch, last), device=ids.device)
            if p.numel() == 0:
                break
            X = ids[p[:, None] + off].long()
            y = ids[p + cfg.ctx].long()
            codes, _ = self._encode_stack(X)
            mixed, _ = self._mixture_logprob(codes, y, eval_mode=True)
            nll += -mixed.sum()
            ntok += X.size(0)
        return math.exp(nll.item() / max(ntok, 1)), ntok

    @torch.no_grad()
    def eval_per_layer_ppl(self, stream, eval_tokens):
        """Per-layer perplexity (for the scoreboard insight) + mixture weights."""
        cfg = self.cfg
        ids = stream[:eval_tokens]
        off = torch.arange(cfg.ctx, device=ids.device)
        nll = torch.zeros(cfg.n_layers, device=ids.device)
        ntok = 0
        last = ids.numel() - cfg.ctx - 1
        for i in range(0, last, cfg.batch):
            p = torch.arange(i, min(i + cfg.batch, last), device=ids.device)
            if p.numel() == 0:
                break
            X = ids[p[:, None] + off].long()
            y = ids[p + cfg.ctx].long()
            codes, _ = self._encode_stack(X)
            for li, (layer, (idx, val)) in enumerate(zip(self.layers, codes)):
                nll[li] += -layer.readout.logprob_true(idx, val, y, eval_mode=True).sum()
            ntok += X.size(0)
        ppl = [math.exp(nll[li].item() / max(ntok, 1)) for li in range(cfg.n_layers)]
        w = torch.softmax(self.mix, dim=0).tolist()
        return ppl, w

    # ───────────────────────────── accounting ──────────────────────────────
    def macs_per_token(self):
        """Honest sparse inference cost summed across layers: per layer, the
        granule expansion (n_gran*fan_in) + readout (K*(C + S)); plus the relay
        (K*relay_dim) for every layer that feeds a next one; plus the context
        bind (ctx*code_dim). Mixture combination is n_layers (negligible)."""
        c = self.cfg
        S = self.hier.S
        per_layer = c.n_gran * c.fan_in + c.k_active * (c.n_classes + S)
        relays = (c.n_layers - 1) * c.k_active * c.relay_dim
        bind = c.ctx * c.code_dim
        return c.n_layers * per_layer + relays + bind + c.n_layers
