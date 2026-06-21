#!/usr/bin/env python3
"""
TEST 013 - the Brain vs a REAL pretrained LLM, fed a lot more data (PyTorch/HF/GPU).

Opponent = a real pretrained transformer (default pythia-410m), scored on WikiText-103
test perplexity with a strided sliding window (the standard, fair method, which gives the
LLM its full context advantage). Brain = cerebellar coder, no backprop: fixed random token
codes -> COMPETITIVELY-LEARNED granule features -> kWTA -> hierarchical (class->word)
delta-rule readout, with the output bias initialized to the unigram distribution. Trained
by streaming English Wikipedia (same domain as the test set). Score = perplexity + MACs/token.

This version is built for a Kaggle "GPU T4 x2" box:
  - data is pre-tokenized ONCE into a uint16 memmap .bin and reused (nanoGPT style); the
    train loop samples random windows through pinned host memory with non_blocking H2D copies,
    so there is zero tokenization in the hot loop.
  - the granule expansion is a DENSE fp16 tensor-core matmul built from the fixed sparse
    wiring (mathematically identical to the gather-multiply-sum). The reported MACs/token
    stays the SPARSE cost; the dense matmul is only a faster way to compute the same model.
  - both GPUs are used for real via torch.multiprocessing.spawn + NCCL local-SGD: one full
    Brain replica per GPU, each on its own shard of the memmap, with the readout tensors
    averaged by all_reduce every K blocks. fp16 is used for the all_reduce transport only.
  - readout masters (Wc, Ww, bw, bc) stay fp32 so the many tiny delta-rule scatter-adds do
    not drift; the heavy activation matmul is fp16 (accumulates in fp32 on the tensor cores).

Run on Kaggle (the program is launched as a real subprocess, not in the kernel):
    !python showdown.py --build-data          # tokenize once into the memmap cache
    !python showdown.py                        # auto dual-GPU when 2 are present
    !python showdown.py --single               # force single GPU
    !python showdown.py --smoke                # tiny, CPU-OK, exercises the whole path

STATUS: validate with `--smoke` first. Honest numbers; losses are sharpened targets.
"""
import argparse
import json
import math
import os
import re
import socket
import subprocess
import time
import warnings
from dataclasses import dataclass, asdict, fields

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")   # use all CPU cores to tokenize
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")            # T4x2 NCCL hangs on P2P without this

import numpy as np

# torch/transformers/datasets are imported lazily where used so this file at least parses
# and can be reasoned about on a box without them. On Kaggle they are present.
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta


# ─────────────────────────────── config ────────────────────────────────────
@dataclass
class Cfg:
    model_id: str = "EleutherAI/pythia-410m"
    corpus: str = "wikimedia/wikipedia"
    corpus_config: str = "20231101.en"
    text_key: str = "text"
    ctx: int = 6
    code_dim: int = 96
    n_gran: int = 49152           # Ww[n_gran, V+1] fp32 dominates VRAM (~9.9GB at 49152); fills the T4
    fan_in: int = 48
    k_active: int = 64
    n_classes: int = 256
    lr: float = 0.3
    wd: float = 1e-5
    decay_every: int = 200
    batch: int = 8192             # large batch to fill VRAM; fp16 activation transient is small
    step_chunk: int = 1024        # process step()/_heads in batch chunks so the [chunk,K,S] word-head
                                  # transient stays bounded (the B*K*S scatter was the real OOM/perf risk)
    block: int = 1_000_000        # tokens per training block -> steps_per_block = block // batch
    train_tokens: int = 500_000_000
    eval_tokens: int = 250_000
    eval_window: int = 512        # LLM strided-perplexity window
    eval_stride: int = 256        # ... and stride (context each target token gets)
    # competitive granule feature learning (the test-12 quality lever; cost-neutral)
    learn_feat: bool = True
    feat_passes: int = 2
    feat_sample: int = 40_000
    feat_eta: float = 0.05
    # data / multi-gpu
    data_dir: str = ""            # resolved at runtime (default /kaggle/tmp else /tmp)
    sync_every: int = 25          # local-SGD averaging interval, in blocks
    master_port: int = 0          # filled in by the parent (a free port) before spawn
    seed: int = 13
    # NOTE: no import-time `device` field. Touching torch.cuda.is_available() at import
    # initializes a CUDA context in the parent and poisons spawned children. Device is
    # resolved only inside the worker / single-GPU path.

    def scale_for_smoke(self):
        self.n_gran, self.fan_in, self.k_active, self.n_classes = 1024, 16, 16, 8
        self.batch, self.block = 256, 50_000
        self.step_chunk = 256
        self.train_tokens, self.eval_tokens = 300_000, 20_000
        self.feat_sample = 8_000
        if self.model_id == "EleutherAI/pythia-410m":
            self.model_id = "sshleifer/tiny-gpt2"
        return self

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        valid = {f.name for f in fields(Cfg)}
        return Cfg(**{k: v for k, v in d.items() if k in valid})


# ─────────────────────────── small utilities ───────────────────────────────
def pick_free_port():
    """Bind a socket to port 0, let the OS assign a free port, return it. Avoids the
    Kaggle-rerun TIME_WAIT hang that a fixed MASTER_PORT runs into."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def print_topology():
    """Run `nvidia-smi topo -m` once and print it, to confirm the PHB/SYS interconnect.
    Swallow any error (no nvidia-smi off Kaggle)."""
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            print("  GPU topology (nvidia-smi topo -m):")
            for line in out.stdout.strip().splitlines():
                print("    " + line)
    except Exception:
        pass


def resolve_data_dir(cfg):
    if cfg.data_dir:
        return cfg.data_dir
    # prefer /kaggle/working: it is persistent across notebook restarts (~20GB quota), so the
    # prebuilt .bin survives a rerun. /tmp is wiped between sessions and would force a rebuild.
    for cand in ("/kaggle/working", "/kaggle/tmp", "/tmp"):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def data_paths(cfg, vocab):
    """Cache-key path for the memmap. The model_id (tokenizer identity) is hashed into the
    name: omitting it would silently reuse a wrong-vocab bin and produce garbage perplexity."""
    d = resolve_data_dir(cfg)
    key = f"wiki_{_sanitize(cfg.model_id)}_{vocab}_{cfg.train_tokens}"
    return os.path.join(d, key + ".bin"), os.path.join(d, key + ".meta.json")


# ───────────────────────────── data builder ────────────────────────────────
def load_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def build_memmap(cfg, tok):
    """Stream + batched-tokenize the corpus ONCE into a uint16 (or uint32) .bin of exactly
    cfg.train_tokens tokens, cache it with a sidecar .meta.json, and reuse if present.
    Returns the meta dict."""
    vocab = int(tok.vocab_size)
    bin_path, meta_path = data_paths(cfg, vocab)

    # uint16 holds ids < 65536 (pythia 50304, tiny-gpt2 50257). Fall back to uint32 otherwise.
    if vocab < 65536:
        np_dtype, dtype_name = np.uint16, "uint16"
    else:
        np_dtype, dtype_name = np.uint32, "uint32"

    budget = int(cfg.train_tokens)

    # reuse if the sidecar matches model/vocab/dtype and has enough tokens
    meta = load_meta(meta_path)
    if (meta and os.path.exists(bin_path)
            and meta.get("model_id") == cfg.model_id
            and meta.get("vocab") == vocab
            and meta.get("dtype") == dtype_name
            and meta.get("token_count", 0) >= 1
            and meta.get("budget") == budget):
        print(f"  data: reusing cache {bin_path}  ({meta['token_count']:,} tokens)")
        return meta

    from datasets import load_dataset
    print(f"  data: building memmap {bin_path}  (budget {budget:,} tokens, {dtype_name})")
    print("  data: this step streams the corpus and needs Kaggle Internet ON.", flush=True)
    t0 = time.time()
    arr = np.memmap(bin_path, dtype=np_dtype, mode="w+", shape=(budget,))
    cur = 0
    docs = []

    def flush():
        nonlocal cur
        if not docs:
            return
        # batched call -> the Rust tokenizer uses all CPU cores. Never pass num_proc to a
        # streaming map; it deadlocks. Here we tokenize plain lists ourselves.
        enc = tok(docs, add_special_tokens=False)["input_ids"]
        docs.clear()
        flat = []
        for e in enc:
            flat.extend(e)
        if not flat:
            return
        m = min(len(flat), budget - cur)
        if m > 0:
            arr[cur:cur + m] = np.asarray(flat[:m], dtype=np_dtype)
            cur += m

    ds = load_dataset(cfg.corpus, cfg.corpus_config, split="train", streaming=True)
    for row in ds:
        t = row.get(cfg.text_key)
        if t:
            docs.append(t)
        if len(docs) >= 2000:
            flush()
        if cur >= budget:
            break
    if cur < budget:
        flush()

    arr.flush()
    del arr
    real = int(cur)
    dt = time.time() - t0
    rate = real / dt if dt > 0 else 0.0
    print(f"  data: built {real:,} tokens in {dt:.0f}s  ({rate:,.0f} tok/s build)")

    meta = {
        "model_id": cfg.model_id,
        "vocab": vocab,
        "dtype": dtype_name,
        "token_count": real,        # the real cursor; the preallocated tail is id 0, do not train on it
        "budget": budget,
        "bin_path": bin_path,
        "corpus": cfg.corpus,
        "corpus_config": cfg.corpus_config,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta


def ensure_memmap(cfg, tok):
    """Build the memmap if missing, return the meta. Must complete in a single process
    BEFORE any NCCL spawn (tokenization must finish first)."""
    return build_memmap(cfg, tok)


# ───────────────────────── memmap window sampler ────────────────────────────
class WindowSampler:
    """Per call, sample `batch` random windows from this rank's shard of the memmap, copy
    them through a preallocated pinned host buffer, and non_blocking-copy to the device.
    Zero tokenization in the loop."""

    def __init__(self, path, dtype_name, length, ctx, batch, device, rank, world, pin=True, seed=0):
        np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[dtype_name]
        # read-only memmap is spawn-safe: each rank opens its own handle.
        self.data = np.memmap(path, dtype=np_dtype, mode="r")[:length]
        self.ctx = ctx
        self.batch = batch
        self.device = device
        self.span = ctx + 1
        # shard by rank: disjoint contiguous ranges -> true world x data throughput
        lo = rank * length // world
        hi = (rank + 1) * length // world - self.span
        if hi <= lo:                              # tiny corpus / smoke: fall back to whole range
            lo, hi = 0, max(1, length - self.span)
        self.lo, self.hi = lo, hi
        # per-rank RNG so sampling is reproducible and decorrelated across ranks (the disjoint
        # shards already guarantee no overlap; this makes the seeding intent real, not just a label).
        self.rng = np.random.default_rng(seed)
        # allocate the pinned host buffer ONCE. Pinning a fresh tensor per step is ~2x slower.
        # int32 holds every token id (< 65536) and halves both PCIe H2D bytes and the CPU copy
        # vs int64; codes[X] indexing accepts int32 (cast to long only inside the kernels that need it).
        use_pin = pin and (device != "cpu")
        self.host = torch.empty(batch, self.span, dtype=torch.int32, pin_memory=use_pin)
        self._arange = np.arange(self.span, dtype=np.int64)

    def next_xy(self):
        ix = self.rng.integers(self.lo, self.hi, size=self.batch)
        # vectorized gather: data[ix[:,None] + arange(span)] -> [batch, ctx+1]
        win = self.data[ix[:, None] + self._arange[None, :]].astype(np.int32)
        self.host.copy_(torch.from_numpy(win))
        gpu = self.host.to(self.device, non_blocking=True)
        return gpu[:, :self.ctx], gpu[:, self.ctx]


def start_producer(tokenizer, cfg, doc_batch=2000, prefetch=6):
    """Streaming fallback used only when the memmap dir is unwritable. Background thread
    tokenizes batched (multi-core Rust) and fills a queue with CPU token blocks."""
    import threading
    import queue
    from datasets import load_dataset
    q = queue.Queue(maxsize=prefetch)

    def producer():
        ds = load_dataset(cfg.corpus, cfg.corpus_config, split="train", streaming=True)
        buf, docs = [], []

        def flush():
            if docs:
                for e in tokenizer(docs, add_special_tokens=False)["input_ids"]:
                    buf.extend(e)
                docs.clear()

        for row in ds:
            t = row.get(cfg.text_key)
            if t:
                docs.append(t)
            if len(docs) >= doc_batch:
                flush()
            while len(buf) >= cfg.block:
                q.put(torch.tensor(buf[:cfg.block], dtype=torch.long))
                del buf[:cfg.block]
        flush()
        while len(buf) >= cfg.block:
            q.put(torch.tensor(buf[:cfg.block], dtype=torch.long))
            del buf[:cfg.block]
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    return q


# ──────────────── LLM eval — fair strided perplexity (KEPT) ─────────────────
@torch.no_grad()
def eval_llm_ppl(model, stream, window, stride, device):
    """Standard sliding-window perplexity: each target token is scored with up to
    (window-stride) tokens of real context. This is the LLM's fair (strong) number."""
    model.eval()
    ids = stream.long().to(device)
    n = ids.numel()
    # accumulate on device, .item() once after the loop (avoid a host sync per window)
    nll = torch.zeros((), device=device)
    ntok = torch.zeros((), dtype=torch.long, device=device)
    prev_end = 0
    t0 = time.time()
    for begin in range(0, n - 1, stride):
        end = min(begin + window, n)
        inp = ids[begin:end].unsqueeze(0)
        tgt = inp.clone()
        new = end - prev_end                       # only the unseen tokens count
        tgt[:, :-new] = -100
        logits = model(inp).logits
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                               tgt[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        nll += loss
        ntok += (tgt[:, 1:] != -100).sum()
        prev_end = end
        if end == n:
            break
    nt = int(ntok.item())
    return math.exp(nll.item() / max(nt, 1)), nt, time.time() - t0


def load_wikitext_test(tokenizer, max_tokens, device):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    ids = []
    for row in ds:
        if row["text"]:
            ids.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids[:max_tokens], dtype=torch.int32, device=device)


# ─────────────────────────────── the Brain ─────────────────────────────────
class Brain:
    """No-backprop cerebellar coder. Heavy activation in fp16 (tensor cores, fp32 accumulate);
    readout masters Wc/Ww/bw/bc in fp32 so the many tiny delta-rule scatter-adds do not drift.

    dtypes:
      codes [V, code_dim]      fp16   fixed random token codes
      gidx  [n_gran, fan_in]   int64  fixed sparse wiring (never communicated)
      gwt   [n_gran, fan_in]   fp32   competitively learned features (stable k-means)
      gbias [n_gran]           fp16   per-granule mean, subtracted after the matmul
      proj  [IN, n_gran]       fp16   DENSE projection built from (gidx, gwt) -> the fast path
      Wc    [n_gran, C]        fp32   class head master
      Ww    [n_gran, V+1]      fp32   word head master (dominant VRAM payload)
      bw    [V+1]              fp32   log-unigram init
      bc    [C]                fp32   log class-freq init
    """

    def __init__(self, vocab, cfg, device):
        self.cfg, self.V, self.device = cfg, vocab, device
        dev = device
        # fp16 tensor-core math is CUDA-only; on CPU (smoke fallback) fp16 @ fp16 raises
        # "addmm_impl_cpu_ not implemented for 'Half'", so keep the activation path in fp32 there.
        self.act_dtype = torch.float16 if str(dev).startswith("cuda") else torch.float32
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)   # seed-deterministic: all ranks match
        self.IN = cfg.ctx * cfg.code_dim
        self.codes = torch.randn(vocab, cfg.code_dim, generator=g).to(dev).to(self.act_dtype)
        self.gidx = torch.randint(0, self.IN, (cfg.n_gran, cfg.fan_in), generator=g).to(dev)
        gwt = torch.randn(cfg.n_gran, cfg.fan_in, generator=g)
        gwt *= math.sqrt(cfg.fan_in) / gwt.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.gwt = gwt.to(dev)                                    # fp32
        self.gbias = torch.zeros(cfg.n_gran, device=dev).to(self.act_dtype)
        self.proj = None                                         # built by build_proj after learn_granules
        self.Wc = self.bc = self.Ww = self.bw = None
        self.class_members = self.w2class = self.w2within = self.S = None
        self._step = 0

    # ---- DENSE projection built from the fixed sparse wiring ----
    def build_proj(self):
        """Scatter the fan_in weights into a dense [IN, n_gran] fp16 matrix. accumulate=True
        so duplicate (input, granule) pairs sum, matching the sparse gather semantics. Call
        once after learn_granules + renorm; rebuild after any gwt mutation, never per step."""
        cfg, dev = self.cfg, self.device
        proj = torch.zeros(self.IN, cfg.n_gran, dtype=self.act_dtype, device=dev)
        cols = torch.arange(cfg.n_gran, device=dev).repeat_interleave(cfg.fan_in)
        proj.index_put_((self.gidx.reshape(-1), cols),
                        self.gwt.to(self.act_dtype).reshape(-1), accumulate=True)
        self.proj = proj

    def _xg(self, X):                                # X[B,ctx] -> sampled inputs [B,G,fan_in]
        x = self.codes[X].reshape(X.size(0), self.IN).float()
        return x[:, self.gidx]                       # only used by learn_granules (small B)

    def _act(self, X):
        """DENSE forward, mathematically identical to the old gather-multiply-sum.
        codes are fp16; the matmul runs on tensor cores and accumulates in fp32."""
        x = self.codes[X].reshape(X.size(0), self.IN)        # fp16 [B, IN]
        return (x @ self.proj) - self.gbias                   # fp16 [B, n_gran]

    def encode(self, X):
        a = self._act(X)                                      # fp16
        K = self.cfg.k_active
        vals, idx = a.topk(K + 1, dim=1)
        val = (vals[:, :K] - vals[:, K:K + 1]).clamp(min=0)   # kWTA relu margin
        return idx[:, :K], val.float()                        # fp32 so the delta rule stays fp32

    # ---- competitive (Hebbian/k-means) granule feature learning, no backprop ----
    def learn_granules(self, block):
        cfg = self.cfg
        n = block.numel() - cfg.ctx - 1
        off = torch.arange(cfg.ctx, device=block.device)
        gen = torch.Generator(device=block.device).manual_seed(cfg.seed + 7)
        for p in range(cfg.feat_passes):
            eta = cfg.feat_eta * (1 - p / cfg.feat_passes)
            pos = torch.randint(0, n, (min(cfg.feat_sample, n),), generator=gen, device=block.device)
            for i in range(0, pos.numel(), 256):
                X = block[pos[i:i + 256][:, None] + off].long()
                xg = self._xg(X)                                   # [B,G,fan_in]
                a = (xg * self.gwt).sum(-1)
                _, win = a.topk(cfg.k_active, dim=1)                # [B,K] winners
                B = X.size(0)
                sel = xg.gather(1, win.unsqueeze(-1).expand(B, cfg.k_active, cfg.fan_in))
                gids = win.reshape(-1)
                # Count-normalized competitive move. If a granule wins m times in this chunk,
                # summing m raw deltas (each off the SAME pre-update gwt[g]) overshoots the
                # cluster mean by ~m. Instead accumulate sum(sel) and counts, then move each
                # touched granule once toward ITS mean: gwt[g] += eta*(mean_sel[g] - gwt[g]).
                sums = torch.zeros_like(self.gwt)
                sums.index_add_(0, gids, sel.reshape(-1, cfg.fan_in))
                counts = torch.zeros(cfg.n_gran, device=self.gwt.device)
                counts.index_add_(0, gids, torch.ones_like(gids, dtype=counts.dtype))
                touched = counts > 0
                mean_sel = sums[touched] / counts[touched].unsqueeze(1)
                self.gwt[touched] += eta * (mean_sel - self.gwt[touched])
            self.gwt *= math.sqrt(cfg.fan_in) / self.gwt.norm(dim=1, keepdim=True).clamp_min(1e-6)

    # ---- class hierarchy + UNIGRAM bias init from a frequency sample ----
    def build_classes(self, freq):
        V, C, dev = self.V, self.cfg.n_classes, self.device
        S = math.ceil(V / C)
        order = torch.argsort(freq, descending=True)
        word_order = torch.cat([order, torch.full((C * S - V,), V, dtype=torch.long)])
        slots = torch.arange(C * S)
        w2class = torch.zeros(V + 1, dtype=torch.long)
        w2within = torch.zeros(V + 1, dtype=torch.long)
        w2class[word_order] = slots // S
        w2within[word_order] = slots % S
        self.class_members = word_order.view(C, S).to(dev)
        self.w2class, self.w2within, self.S = w2class.to(dev), w2within.to(dev), S
        # readout masters: explicit fp32 (stable delta-rule accumulation)
        self.Wc = torch.zeros(self.cfg.n_gran, C, dtype=torch.float32, device=dev)
        self.Ww = torch.zeros(self.cfg.n_gran, V + 1, dtype=torch.float32, device=dev)
        # bias = log unigram (so the readout starts at the unigram baseline, not uniform)
        bw = torch.full((V + 1,), -30.0, dtype=torch.float32)
        bw[:V] = torch.log(freq + 0.5)
        cf = torch.zeros(C, dtype=torch.float32)
        cf.index_add_(0, w2class[:V], freq)
        self.bw = bw.to(dev)
        self.bc = torch.log(cf + 0.5).to(dev)

    def calibrate_bias(self, block):
        n = min(8000, block.numel() - self.cfg.ctx - 1)
        off = torch.arange(self.cfg.ctx, device=block.device)
        acc = torch.zeros(self.cfg.n_gran, device=block.device)
        cnt = 0
        # zero gbias first so _act inside the loop does not subtract a stale bias; accumulate
        # in fp32, then store gbias in the activation dtype.
        self.gbias = torch.zeros(self.cfg.n_gran, device=block.device).to(self.act_dtype)
        for i in range(0, n, 512):                                       # chunk over batch
            p = torch.arange(i, min(i + 512, n), device=block.device)
            acc += self._act(block[p[:, None] + off].long()).float().sum(0)
            cnt += p.numel()
        self.gbias = (acc / max(cnt, 1)).to(self.act_dtype)

    def _heads(self, idx, val, cmem):                # vectorized over the K active granules
        vk = val.unsqueeze(-1)                                  # [B,K,1]
        lc = self.bc + (vk * self.Wc[idx]).sum(1)              # [B,C]
        Ww_g = self.Ww[idx.unsqueeze(-1), cmem.unsqueeze(1)]   # [B,K,S]
        lw = self.bw[cmem] + (vk * Ww_g).sum(1)                # [B,S]
        return lc, lw

    @torch.no_grad()
    def logprob_true(self, X, y):
        idx, val = self.encode(X)
        ce, within = self.w2class[y], self.w2within[y]
        lc, lw = self._heads(idx, val, self.class_members[ce])
        ar = torch.arange(X.size(0), device=X.device)
        return lc.log_softmax(1)[ar, ce] + lw.log_softmax(1)[ar, within]

    def step(self, X, y):
        cfg = self.cfg
        B, K, S = X.size(0), self.cfg.k_active, self.S
        st = cfg.lr / B                                        # normalize by the FULL batch
        # Process the batch in chunks so the word-head [chunk,K,S] expand/value transient stays
        # bounded. Chunking is exactly equivalent: every update below accumulates into the master
        # and st = lr/B is fixed, so the result is identical to processing all B rows at once.
        chunk = max(1, min(getattr(cfg, "step_chunk", B), B))
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            Xc, yc = X[lo:hi], y[lo:hi]
            idx, val = self.encode(Xc)
            bc_n = Xc.size(0)
            ce, within = self.w2class[yc], self.w2within[yc]
            cmem = self.class_members[ce]                      # [bc,S]
            lc, lw = self._heads(idx, val, cmem)
            ar = torch.arange(bc_n, device=X.device)
            errc = lc.softmax(1); errc[ar, ce] -= 1.0          # [bc,C]
            errw = lw.softmax(1); errw[ar, within] -= 1.0      # [bc,S]
            vk = val.unsqueeze(-1)                             # [bc,K,1]
            # class head - one scatter over all (example,granule) pairs in this chunk
            self.Wc.index_add_(0, idx.reshape(-1),
                               ((-st) * vk * errc.unsqueeze(1)).reshape(-1, errc.size(1)))
            # word head - one scatter. accumulate=True is required: removing it overwrites
            # shared-granule updates (research probe four).
            rows = idx.unsqueeze(-1).expand(bc_n, K, S).reshape(-1)
            cols = cmem.unsqueeze(1).expand(bc_n, K, S).reshape(-1)
            self.Ww.index_put_((rows, cols),
                               ((-st) * vk * errw.unsqueeze(1)).reshape(-1), accumulate=True)
            self.bc.add_(errc.sum(0), alpha=-st)
            self.bw.index_put_((cmem.reshape(-1),), ((-st) * errw).reshape(-1), accumulate=True)
        self._step += 1
        if cfg.wd > 0 and self._step % cfg.decay_every == 0:
            keep = (1.0 - cfg.lr * cfg.wd) ** cfg.decay_every
            self.Ww.mul_(keep); self.Wc.mul_(keep)

    def train_on_block(self, sampler, n_steps):
        """Pull n_steps batches from the WindowSampler and apply the delta rule. No .item()
        in the loop (no host sync). Returns tokens consumed this block."""
        for _ in range(n_steps):
            X, y = sampler.next_xy()
            self.step(X, y)
        return n_steps * self.cfg.batch

    @torch.no_grad()
    def eval_ppl(self, stream, eval_tokens):
        cfg = self.cfg
        ids = stream[:eval_tokens]
        off = torch.arange(cfg.ctx, device=ids.device)
        nll = torch.zeros((), device=ids.device)               # accumulate on device, .item() once
        ntok = 0
        for i in range(0, ids.numel() - cfg.ctx - 1, cfg.batch):
            p = torch.arange(i, min(i + cfg.batch, ids.numel() - cfg.ctx - 1), device=ids.device)
            if p.numel() == 0:
                break
            X = ids[p[:, None] + off].long()
            y = ids[p + cfg.ctx].long()
            nll += -self.logprob_true(X, y).sum()
            ntok += X.size(0)
        return math.exp(nll.item() / max(ntok, 1)), ntok

    def macs_per_token(self):
        # HEADLINE metric: the SPARSE inference cost. The dense matmul is only a faster way
        # to compute the same sparse model at train time; it does not change this number.
        c = self.cfg
        return c.n_gran * c.fan_in + c.k_active * (c.n_classes + self.S)

    # ---- multi-GPU readout sync (local-SGD all_reduce average) ----
    def average_readout(self, world_size):
        """all_reduce-average ONLY the readout (Ww, Wc, bw, bc). codes/gidx/gwt/gbias/proj/
        classes are fixed and seed-identical, so they are never synced.

        Ww is the dominant payload and PCIe-bound, so it is transported in bf16 (half the bytes;
        no NVLink on T4x2). bf16 has fp32's 8-bit exponent, so the tiny near-zero delta-rule
        updates for rare words are not flushed to zero the way fp16 would flush them. It is
        reduced IN CHUNKS through a small reusable scratch buffer so the transient stays bounded
        regardless of n_gran (a full fp16/bf16 copy of Ww would OOM once n_gran is raised).

        Wc, bw, bc are tiny but precision-critical: bw holds the log-unigram bias (~14-17 for
        frequent tokens) where fp16's ~0.016 step would quantize away ~400x the per-step bias
        delta and undo the fp32-master design. They are sent in fp32 (a few hundred KB, trivial
        PCIe cost) so the master semantics are preserved exactly."""
        if world_size <= 1:
            return
        # Ww (the big payload) -> bf16, chunked in place via a reusable scratch buffer.
        n_rows = self.Ww.size(0)
        rows_per_chunk = max(1, 2048)
        scratch = torch.empty(min(rows_per_chunk, n_rows), self.Ww.size(1),
                              dtype=torch.bfloat16, device=self.device)
        for r in range(0, n_rows, rows_per_chunk):
            r2 = min(r + rows_per_chunk, n_rows)
            buf = scratch[:r2 - r]
            buf.copy_(self.Ww[r:r2])                # fp32 -> bf16
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(world_size)
            self.Ww[r:r2].copy_(buf)                # bf16 -> fp32 master
        del scratch
        # Wc, bw, bc -> coalesce into one flat fp32 buffer, one all_reduce (precision preserved).
        parts = [self.Wc, self.bw, self.bc]
        flat = torch.cat([p.reshape(-1).float() for p in parts])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        o = 0
        for p in parts:
            nele = p.numel()
            p.copy_(flat[o:o + nele].reshape(p.shape))
            o += nele


# ───────────────────────── per-GPU worker (replica) ────────────────────────
def worker(rank, world_size, cfg_dict, data_meta):
    """One full Brain replica per GPU. NCCL local-SGD when world_size>1; straight-line
    single-GPU / CPU(smoke) when world_size==1. rank 0 does the LLM eval + scoreboard."""
    cfg = Cfg.from_dict(cfg_dict)
    torch.manual_seed(cfg.seed)

    # --- device + (optional) NCCL setup ---
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        # hard assignment (not setdefault): main() picks a free port to dodge the Kaggle-rerun
        # TIME_WAIT hang; a stale inherited MASTER_PORT must NOT win over it.
        os.environ["MASTER_PORT"] = str(cfg.master_port)
        # set_device BEFORE init_process_group and before any device tensor. Skipping this
        # is the #1 bug that silently collapses every rank onto cuda:0.
        torch.cuda.set_device(rank)
        # generous timeout for the setup all_reduces; the long rank-0-only eval is NOT under any
        # collective (the group is torn down before it), so this only covers training-time syncs.
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                timeout=timedelta(minutes=30))
        dev = f"cuda:{rank}"
    else:
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dev.startswith("cuda"):
            torch.cuda.set_device(0)

    if dev.startswith("cuda"):
        print(f"  [rank {rank}] device cuda:{torch.cuda.current_device()} "
              f"({torch.cuda.get_device_name(torch.cuda.current_device())})", flush=True)
    else:
        print(f"  [rank {rank}] device cpu", flush=True)

    # --- tokenizer (each worker loads its own; CPU only) + vocab ---
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.model_max_length = 10 ** 9
    # Load the token memmap first and size the vocab to cover the TRUE id range.
    # tok.vocab_size undercounts some tokenizers (GPT-NeoX emits ids past vocab_size),
    # so trust the data: V must exceed every token id, or codes[token] gathers out of
    # bounds (a device-side assert). This reuses the existing cache, no rebuild.
    np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[data_meta["dtype"]]
    length = int(data_meta["token_count"])
    full = np.memmap(data_meta["bin_path"], dtype=np_dtype, mode="r")[:length]
    V = max(int(data_meta["vocab"]), len(tok), int(full.max()) + 1)

    # --- the brain (seed-deterministic, identical across ranks) ---
    brain = Brain(V, cfg, dev)

    # --- shared setup before the parallel loop, so all replicas start identical.
    # The global head of the corpus (rank-independent) builds classes / bias / features,
    # so every replica is identical without communication. ---
    setup_n = min(cfg.block, length - 1)
    setup_block = torch.from_numpy(np.asarray(full[:setup_n], dtype=np.int64)).to(dev)

    freq = torch.bincount(setup_block.cpu(), minlength=V + 1)[:V].float()
    brain.build_classes(freq)
    brain.build_proj()                       # dense fp16 projection from the granule wiring;
    brain.calibrate_bias(setup_block)        # _act needs proj, so build it first
    if cfg.learn_feat:
        if rank == 0:
            print("  brain: learning granule features (competitive, no backprop) ...", flush=True)
        brain.learn_granules(setup_block)
        # belt-and-suspenders: average gwt once so no replica can drift even if RNG differs.
        if world_size > 1:
            dist.all_reduce(brain.gwt, op=dist.ReduceOp.SUM)
            brain.gwt.div_(world_size)
        brain.build_proj()
        brain.calibrate_bias(setup_block)
        # gbias is recomputed locally from a CUDA reduction (not bitwise-deterministic across
        # ranks); average it once so every replica is a strictly identical encoder (matching the
        # gwt sync). Tiny gbias drift can otherwise flip borderline kWTA winners between ranks.
        if world_size > 1:
            gb = brain.gbias.float()
            dist.all_reduce(gb, op=dist.ReduceOp.SUM)
            gb.div_(world_size)
            brain.gbias = gb.to(brain.act_dtype)
    del setup_block, full

    # --- sharded training loop. Fixed step count per rank so every rank calls the same
    #     number of collectives (mismatched collective counts deadlock NCCL). ---
    batch = cfg.batch
    steps_per_block = max(1, cfg.block // batch)
    total_steps = max(1, cfg.train_tokens // (batch * max(1, world_size)))
    n_blocks = max(1, total_steps // steps_per_block)

    sampler = WindowSampler(data_meta["bin_path"], data_meta["dtype"], length,
                            cfg.ctx, batch, dev, rank, world_size, pin=(dev != "cpu"),
                            seed=cfg.seed + 2 + rank)

    if rank == 0:
        print(f"  brain: {n_blocks} blocks x {steps_per_block} steps x batch {batch} "
              f"per rank ({world_size} rank(s))", flush=True)
    t0 = time.time()
    consumed = 0
    for blk in range(n_blocks):
        consumed += brain.train_on_block(sampler, steps_per_block)
        if world_size > 1 and (blk + 1) % cfg.sync_every == 0:
            brain.average_readout(world_size)
        if rank == 0 and (blk + 1) % max(1, n_blocks // 20) == 0:
            torch.cuda.synchronize() if dev.startswith("cuda") else None
            tot = consumed * world_size
            print(f"    trained ~{tot:,} tokens  ({time.time()-t0:.0f}s)", flush=True)
    if world_size > 1:
        brain.average_readout(world_size)          # final mandatory average
        # Tear the process group DOWN before the long rank-0-only eval. Otherwise rank 1 would
        # sit on a collective (barrier) under the NCCL timeout while rank 0 does a cold
        # pythia-410m + WikiText-103 download and full strided eval; a slow cold run could blow
        # the timeout and abort a finished training run. After this point rank 0 runs eval as a
        # pure local section with no surviving NCCL dependency.
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()
        if rank != 0:
            return

    # --- eval + scoreboard on rank 0 only (others already returned; parent stayed CUDA-clean) ---
    if rank == 0:
        # free brain transients before loading the LLM; readout masters stay resident.
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id)
        model = base.to(dev).half() if dev.startswith("cuda") else base
        test_stream = load_wikitext_test(tok, cfg.eval_tokens, dev)
        win = min(cfg.eval_window,
                  int(getattr(model.config, "max_position_embeddings", cfg.eval_window)))
        llm_ppl, llm_ntok, llm_t = eval_llm_ppl(model, test_stream, win, cfg.eval_stride, dev)
        llm_macs = sum(p.numel() for p in model.parameters())
        print(f"\n  LLM (pretrained, strided window {win}): perplexity {llm_ppl:.2f}  "
              f"{llm_macs:,} MACs/token  ({llm_t:.1f}s)", flush=True)
        del model, base
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

        br_ppl, _ = brain.eval_ppl(test_stream, cfg.eval_tokens)
        br_macs = brain.macs_per_token()
        total_consumed = consumed * world_size
        print(f"  Brain (no-bp): perplexity {br_ppl:.2f}  {br_macs:,} MACs/token  "
              f"(trained {total_consumed:,} tok in {time.time()-t0:.0f}s)", flush=True)
        run_scoreboard(llm_ppl, llm_macs, br_ppl, br_macs, total_consumed)


def run_scoreboard(llm_ppl, llm_macs, br_ppl, br_macs, consumed):
    print("\n" + "=" * 70)
    print("  SCOREBOARD   (WikiText-103 test, same tokenizer/stream)")
    print("=" * 70)
    print(f"    {'system':<28}{'perplexity':>12}{'MACs/token':>16}{'backprop':>10}")
    print(f"    {'LLM (pretrained)':<28}{llm_ppl:>12.2f}{llm_macs:>16,}{'yes':>10}")
    print(f"    {'Brain ('+str(consumed//1_000_000)+'M tok, no-bp)':<28}"
          f"{br_ppl:>12.2f}{br_macs:>16,}{'NO':>10}")
    print(f"    {'-'*64}")
    print(f"    brain runs at {br_macs / llm_macs:.5f}x the LLM's compute per token.")
    print("=" * 70)


# ─────────────────────────────── main ──────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--train-tokens", type=int, default=None)
    ap.add_argument("--no-feat", action="store_true", help="disable competitive granule learning")
    ap.add_argument("--build-data", action="store_true", help="build the uint16 memmap then exit")
    ap.add_argument("--data-dir", default=None, help="where the .bin lives (default /kaggle/working else /kaggle/tmp else /tmp)")
    ap.add_argument("--single", action="store_true", help="force single GPU even if 2 are present")
    ap.add_argument("--sync-every", type=int, default=None, help="local-SGD averaging interval in blocks")
    ap.add_argument("--block", type=int, default=None, help="tokens per training block")
    ap.add_argument("--n-gran", type=int, default=None, help="granule count; lower it (e.g. 32768) if VRAM OOMs (default 49152)")
    ap.add_argument("--batch", type=int, default=None, help="training batch size; lower it if VRAM OOMs")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Cfg()
    if args.model:
        cfg.model_id = args.model
    if args.smoke:
        cfg.scale_for_smoke()
    if args.train_tokens:
        cfg.train_tokens = args.train_tokens
    if args.no_feat:
        cfg.learn_feat = False
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.sync_every is not None:
        cfg.sync_every = args.sync_every
    if args.block is not None:
        cfg.block = args.block
    if args.n_gran is not None:        # explicit override wins, even over --smoke scaling
        cfg.n_gran = args.n_gran
    if args.batch is not None:
        cfg.batch = args.batch
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.data_dir = resolve_data_dir(cfg)

    warnings.filterwarnings("ignore")
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # device_count() reads the driver and does NOT create a CUDA context, so it is safe in
    # the parent before spawn. is_available()/.to(cuda)/allocations would poison children.
    ngpu = torch.cuda.device_count()
    use_mp = (not args.single) and (not args.smoke) and ngpu >= 2

    print("=" * 70)
    print(f"  TEST 013: Brain vs a real LLM   (gpus={ngpu}, model={cfg.model_id}, "
          f"mode={'dual-GPU' if use_mp else ('smoke' if args.smoke else 'single')})")
    print("=" * 70)
    print_topology()

    # tokenizer in the parent (CPU only) for the data build + vocab.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.model_max_length = 10 ** 9

    # --build-data: build the cache and exit.
    if args.build_data:
        meta = build_memmap(cfg, tok)
        print(f"  done: {meta['token_count']:,} tokens at {meta['bin_path']}")
        return

    # smoke: straight-line worker with world_size=1 (exercises the worker body + sampler).
    if args.smoke:
        meta = ensure_memmap(cfg, tok)
        worker(0, 1, cfg.to_dict(), meta)
        return

    if use_mp:
        # Tokenization must FINISH before any NCCL spawn, in this single parent process.
        meta = ensure_memmap(cfg, tok)
        cfg.master_port = pick_free_port()         # avoid TIME_WAIT hang on Kaggle reruns
        world_size = ngpu
        # spawn: separate processes, no GIL, real NCCL all_reduce. Parent did NO CUDA work.
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker, args=(world_size, cfg.to_dict(), meta),
                 nprocs=world_size, join=True)
    else:
        # single GPU (or --single): run the worker body directly, no spawn, no NCCL.
        meta = ensure_memmap(cfg, tok)
        worker(0, 1, cfg.to_dict(), meta)


if __name__ == "__main__":
    main()
