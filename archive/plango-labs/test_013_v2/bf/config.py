"""Configuration for BrainFormer v13-v2.

Every knob lives here. The Cfg dataclass is serialized to a plain dict before
crossing the torch.multiprocessing.spawn boundary (no tensors, no CUDA state),
then rebuilt in each worker. `scale_for_smoke` shrinks everything to a CPU-OK
size that still exercises the full multi-layer path.
"""
import argparse
from dataclasses import dataclass, asdict, field, fields
from typing import List


# ────────────────────────────── environment ────────────────────────────────
# These must be set BEFORE torch / CUDA are touched, so callers import this
# module first. We do not import torch here (keeps the parent CUDA-clean before
# spawn, and lets config be reasoned about on a box without torch).
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")   # T4x2 NCCL hangs on P2P without this


@dataclass
class Cfg:
    # ── opponent / corpus ───────────────────────────────────────────────────
    model_id: str = "EleutherAI/pythia-410m"
    corpus: str = "wikimedia/wikipedia"
    corpus_config: str = "20231101.en"
    text_key: str = "text"

    # ── context + token codes ───────────────────────────────────────────────
    ctx: int = 32                 # hyperdimensional binding makes long ctx cheap
    code_dim: int = 128           # per-token random code width (also the bound width)
    hd_binding: bool = True       # bind token x position then sum (no concatenation)
    # multi-scale temporal decay: recent tokens sharp, distant tokens gist.
    decay_rates: List[float] = field(default_factory=lambda: [0.20, 0.05, 0.0125])

    # ── the deep stack ──────────────────────────────────────────────────────
    # MEMORY: the word head Ww[n_gran, V+1] fp32 is the dominant payload, summed
    # across layers. 4 x 12288 x ~50305 x 4B = 9.9GB, matching v13's single-layer
    # footprint and fitting one T4. Raising n_layers or n_gran scales this
    # linearly -- drop one if VRAM OOMs (both are CLI knobs).
    n_layers: int = 4             # stacked cerebellar layers (each locally trained)
    n_gran: int = 12288           # granules PER LAYER (keep n_gran % topk_groups == 0)
    fan_in: int = 48              # mossy fibers per granule
    k_active: int = 64            # kWTA winners per layer
    n_classes: int = 256          # hierarchical readout class count
    # layer L>0 reads a fixed-width densification of layer L-1's sparse code.
    relay_dim: int = 128          # width of the inter-layer relay vector

    # ── two-level approximate topk (kernels.approx_topk) ────────────────────
    topk_groups: int = 128        # split n_gran into this many groups (12288/128=96)
    topk_per_group: int = 16      # take this many candidates per group (2048 cands)

    # ── local learning (delta rule) ─────────────────────────────────────────
    lr: float = 0.3
    wd: float = 1e-5
    decay_every: int = 200
    mix_lr: float = 0.05          # learning rate for the layer-mixture weights
    # EMA on the cheap readout tensors (class head + biases) only.
    ema_alpha: float = 0.999
    ema_every: int = 100
    use_ema: bool = True

    # ── batching ────────────────────────────────────────────────────────────
    batch: int = 8192
    step_chunk: int = 2048        # bound the [chunk,K,S] word-head transient
    block: int = 1_000_000        # tokens per training block
    train_tokens: int = 500_000_000

    # ── competitive granule feature learning ────────────────────────────────
    learn_feat: bool = True
    feat_passes: int = 2
    feat_sample: int = 40_000
    feat_eta: float = 0.05
    # continuous online refinement during training (granules keep adapting).
    refine_every: int = 50        # blocks between online refinements (0 = off)
    refine_sample: int = 4000
    refine_eta: float = 0.01

    # ── evaluation ──────────────────────────────────────────────────────────
    eval_tokens: int = 250_000
    eval_window: int = 512        # LLM strided-perplexity window
    eval_stride: int = 256

    # ── data / multi-gpu ────────────────────────────────────────────────────
    data_dir: str = ""            # resolved at runtime
    sync_every: int = 25          # local-SGD averaging interval, in blocks
    master_port: int = 0          # filled by the parent (a free port) before spawn
    seed: int = 13

    # ── derived (filled at runtime, not a real knob) ────────────────────────
    n_scales: int = 0             # len(decay_rates); set in __post_init__

    def __post_init__(self):
        self.n_scales = len(self.decay_rates)

    def scale_for_smoke(self):
        """Tiny everything, CPU-OK, but keep >1 layer so the stack path runs."""
        self.n_layers = 2
        self.n_gran, self.fan_in, self.k_active, self.n_classes = 1024, 16, 16, 8
        self.relay_dim = 32
        self.ctx, self.code_dim = 8, 32
        self.decay_rates = [0.2, 0.05]
        self.n_scales = 2
        self.topk_groups, self.topk_per_group = 16, 8
        self.batch, self.block = 256, 50_000
        self.step_chunk = 256
        self.train_tokens, self.eval_tokens = 300_000, 20_000
        self.feat_sample = 8_000
        self.refine_every = 5
        self.refine_sample = 1_000
        if self.model_id == "EleutherAI/pythia-410m":
            self.model_id = "sshleifer/tiny-gpt2"
        return self

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        valid = {f.name for f in fields(Cfg)}
        return Cfg(**{k: v for k, v in d.items() if k in valid})


# ──────────────────────────────── CLI ──────────────────────────────────────
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="BrainFormer v13-v2 (deep cerebellar LM)")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-OK run of the whole path")
    ap.add_argument("--build-data", action="store_true", help="build the memmap then exit")
    ap.add_argument("--single", action="store_true", help="force single GPU")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--train-tokens", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-gran", type=int, default=None, help="granules per layer; lower if VRAM OOMs")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--sync-every", type=int, default=None)
    # speed knobs: raising n-classes shrinks S=V/n_classes -> far less word-head
    # memory traffic (the dominant cost); lowering k-active scales readout linearly.
    ap.add_argument("--n-classes", type=int, default=None, help="readout classes; raise (e.g. 1024/2048) for speed")
    ap.add_argument("--k-active", type=int, default=None, help="kWTA winners per layer; lower for speed")
    ap.add_argument("--step-chunk", type=int, default=None, help="readout chunk; raise to cut launch overhead if VRAM allows")
    ap.add_argument("--no-feat", action="store_true", help="disable competitive granule learning")
    ap.add_argument("--no-refine", action="store_true", help="disable online granule refinement")
    ap.add_argument("--no-ema", action="store_true", help="disable readout EMA")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args(argv)


def cfg_from_args(args):
    """Build a Cfg from parsed CLI args. --smoke scaling runs first; explicit
    overrides win over it (so you can e.g. --smoke --n-gran 2048)."""
    cfg = Cfg()
    if args.model:
        cfg.model_id = args.model
    if args.smoke:
        cfg.scale_for_smoke()
    if args.train_tokens is not None:
        cfg.train_tokens = args.train_tokens
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    if args.n_gran is not None:
        cfg.n_gran = args.n_gran
    if args.ctx is not None:
        cfg.ctx = args.ctx
    if args.batch is not None:
        cfg.batch = args.batch
    if args.block is not None:
        cfg.block = args.block
    if args.sync_every is not None:
        cfg.sync_every = args.sync_every
    if args.n_classes is not None:
        cfg.n_classes = args.n_classes
    if args.k_active is not None:
        cfg.k_active = args.k_active
    if args.step_chunk is not None:
        cfg.step_chunk = args.step_chunk
    if args.no_feat:
        cfg.learn_feat = False
    if args.no_refine:
        cfg.refine_every = 0
    if args.no_ema:
        cfg.use_ema = False
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.__post_init__()           # refresh derived fields after overrides
    return cfg
