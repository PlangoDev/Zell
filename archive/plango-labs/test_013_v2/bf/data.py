"""Data pipeline: one-time tokenization into a memmap, then a window sampler that
streams random context windows to the GPU with zero tokenization in the hot loop.

The builder logic is carried over from v13 (it works well). The sampler adds an
async double-buffer: while the GPU trains on batch N, the next batch's H2D copy
runs on a dedicated CUDA stream, hiding PCIe latency on the T4 (no NVLink).
"""
import json
import os
import re
import time

import numpy as np


# ─────────────────────────── path resolution ───────────────────────────────
def resolve_data_dir(cfg):
    if cfg.data_dir:
        return cfg.data_dir
    # /kaggle/working persists across notebook restarts (~20GB), so the prebuilt
    # .bin survives a rerun; /tmp is wiped between sessions.
    for cand in ("/kaggle/working", "/kaggle/tmp", "/tmp"):
        if os.path.isdir(cand):
            return cand
    return os.getcwd()


def _sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def data_paths(cfg, vocab):
    """Cache-key path. The model_id (tokenizer identity) and vocab are hashed in:
    reusing a wrong-vocab bin would silently produce garbage perplexity."""
    d = resolve_data_dir(cfg)
    key = f"wiki_{_sanitize(cfg.model_id)}_{vocab}_{cfg.train_tokens}"
    return os.path.join(d, key + ".bin"), os.path.join(d, key + ".meta.json")


def load_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


# ───────────────────────────── memmap builder ──────────────────────────────
def build_memmap(cfg, tok):
    """Stream + batched-tokenize the corpus ONCE into a uint16/uint32 .bin of
    exactly cfg.train_tokens tokens, cache with a sidecar .meta.json, reuse if
    present. Returns the meta dict. Must finish before any NCCL spawn."""
    vocab = int(tok.vocab_size)
    bin_path, meta_path = data_paths(cfg, vocab)

    if vocab < 65536:
        np_dtype, dtype_name = np.uint16, "uint16"
    else:
        np_dtype, dtype_name = np.uint32, "uint32"

    budget = int(cfg.train_tokens)
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
        # batched call -> the Rust tokenizer uses all cores. Never pass num_proc
        # to a streaming map; it deadlocks. We tokenize plain lists ourselves.
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
        "token_count": real,   # the real cursor; the preallocated tail is id 0
        "budget": budget,
        "bin_path": bin_path,
        "corpus": cfg.corpus,
        "corpus_config": cfg.corpus_config,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return meta


def ensure_memmap(cfg, tok):
    return build_memmap(cfg, tok)


# ───────────────────────── async window sampler ────────────────────────────
class WindowSampler:
    """Per call, sample `batch` random context windows from this rank's shard of
    the memmap and deliver them on-device. A second pinned buffer + a dedicated
    CUDA copy-stream let the NEXT batch's H2D copy overlap the current step's
    compute (double buffering). Zero tokenization in the loop.
    """

    def __init__(self, path, dtype_name, length, ctx, batch, device, rank, world,
                 pin=True, seed=0):
        import torch
        self.torch = torch
        np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[dtype_name]
        self.data = np.memmap(path, dtype=np_dtype, mode="r")[:length]
        self.ctx = ctx
        self.batch = batch
        self.device = device
        self.span = ctx + 1
        # shard by rank: disjoint contiguous ranges -> world x data throughput
        lo = rank * length // world
        hi = (rank + 1) * length // world - self.span
        if hi <= lo:                       # tiny corpus / smoke: whole range
            lo, hi = 0, max(1, length - self.span)
        self.lo, self.hi = lo, hi
        self.rng = np.random.default_rng(seed)
        self._arange = np.arange(self.span, dtype=np.int64)

        # two pinned host buffers for double buffering. int32 holds every id
        # (< 65536) and halves PCIe bytes vs int64; codes[X] accepts int32.
        use_pin = pin and (device != "cpu")
        self.host = [torch.empty(batch, self.span, dtype=torch.int32, pin_memory=use_pin)
                     for _ in range(2)]
        self._cuda = str(device).startswith("cuda")
        self._stream = torch.cuda.Stream(device=device) if self._cuda else None
        self._slot = 0
        self._prefetched = None            # (gpu_tensor, slot) waiting to be consumed

    def _gather_into(self, host_buf):
        ix = self.rng.integers(self.lo, self.hi, size=self.batch)
        win = self.data[ix[:, None] + self._arange[None, :]].astype(np.int32)
        host_buf.copy_(self.torch.from_numpy(win))

    def _launch(self):
        """Fill a host buffer and kick off its async H2D copy on the copy-stream."""
        slot = self._slot
        host_buf = self.host[slot]
        self._gather_into(host_buf)
        if self._cuda:
            with self.torch.cuda.stream(self._stream):
                gpu = host_buf.to(self.device, non_blocking=True)
        else:
            gpu = host_buf.to(self.device)
        self._slot ^= 1
        self._prefetched = (gpu, slot)

    def next_xy(self):
        if self._prefetched is None:
            self._launch()                 # first call: prime the pipeline
        gpu, _slot = self._prefetched
        if self._cuda:
            # make the current step's stream wait on the copy that produced `gpu`
            self.torch.cuda.current_stream().wait_stream(self._stream)
        self._launch()                     # start the NEXT batch's copy now
        return gpu[:, :self.ctx], gpu[:, self.ctx]
