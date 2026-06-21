"""Multi-GPU plumbing: NCCL init, a free-port picker, and local-SGD averaging of
every learned tensor across ranks.

Only the LEARNED tensors are synced (each layer's Ww/Wc/bc/bw and the mixture).
The fixed front-end (token codes, position codes, granule wiring, and the
seed-identical learned granule weights, which are averaged once at setup) is never
communicated. Ww is the dominant payload and PCIe-bound on T4x2 (no NVLink), so
it is transported in bf16 chunks; bf16 keeps fp32's 8-bit exponent so tiny
delta-rule updates for rare words are not flushed to zero. The small,
precision-critical tensors go in fp32.
"""
import socket
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist


def pick_free_port():
    """OS-assigned free port. Avoids the Kaggle-rerun TIME_WAIT hang a fixed
    MASTER_PORT runs into."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def print_topology():
    try:
        out = subprocess.run(["nvidia-smi", "topo", "-m"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip():
            print("  GPU topology (nvidia-smi topo -m):")
            for line in out.stdout.strip().splitlines():
                print("    " + line)
    except Exception:
        pass


def init_worker(rank, world_size, cfg):
    """Set the device and (if multi-GPU) the NCCL process group. Returns the
    device string. set_device MUST precede init_process_group and any device
    tensor, or every rank silently collapses onto cuda:0."""
    import os
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(cfg.master_port)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                timeout=timedelta(minutes=30))
        return f"cuda:{rank}"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dev.startswith("cuda"):
        torch.cuda.set_device(0)
    return dev


def _avg_big_bf16(t, world_size, device, rows_per_chunk=2048):
    """all_reduce-average a big fp32 tensor via bf16 chunks through a reusable
    scratch buffer (bounded transient regardless of size)."""
    n_rows = t.size(0)
    scratch = torch.empty(min(rows_per_chunk, n_rows), t.size(1),
                          dtype=torch.bfloat16, device=device)
    for r in range(0, n_rows, rows_per_chunk):
        r2 = min(r + rows_per_chunk, n_rows)
        buf = scratch[:r2 - r]
        buf.copy_(t[r:r2])
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf.div_(world_size)
        t[r:r2].copy_(buf)
    del scratch


def _avg_small_fp32(tensors, world_size):
    """Coalesce small precision-critical tensors into one flat fp32 all_reduce."""
    flat = torch.cat([p.reshape(-1).float() for p in tensors])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    o = 0
    for p in tensors:
        n = p.numel()
        p.copy_(flat[o:o + n].reshape(p.shape))
        o += n


def average_brain(brain, world_size):
    """Average every learned tensor across ranks: per-layer Ww (bf16 chunked) and
    Wc/bc/bw (fp32 coalesced), plus the mixture logits."""
    if world_size <= 1:
        return
    smalls = []
    for layer in brain.layers:
        bigs, sm = layer.readout.sync_tensors()
        for t in bigs:
            _avg_big_bf16(t, world_size, brain.device)
        smalls.extend(sm)
    smalls.append(brain.mix)
    _avg_small_fp32(smalls, world_size)


def average_granules(brain, world_size):
    """Average the competitively-learned granule weights + biases ONCE after setup
    so every replica is a bitwise-identical encoder (CUDA reductions are not
    deterministic across ranks; tiny drift can flip borderline kWTA winners)."""
    if world_size <= 1:
        return
    for layer in brain.layers:
        f = layer.field
        dist.all_reduce(f.gwt, op=dist.ReduceOp.SUM); f.gwt.div_(world_size)
        gb = f.gbias.float()
        dist.all_reduce(gb, op=dist.ReduceOp.SUM); gb.div_(world_size)
        f.gbias = gb.to(f.act_dtype)
        layer.build_proj()
