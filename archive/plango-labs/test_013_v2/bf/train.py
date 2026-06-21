"""Per-rank worker: deterministic shared setup, sharded training loop with local-
SGD sync + online refinement, then a rank-0-only LLM eval and scoreboard.

One full DeepBrain replica per GPU. Every rank issues the same number of
collectives (mismatched counts deadlock NCCL): sync happens on a fixed block
cadence and once more at the end, and the process group is torn down BEFORE the
long rank-0-only eval so other ranks never sit under the NCCL timeout.
"""
import time

import numpy as np
import torch
import torch.distributed as dist

from .config import Cfg
from .data import WindowSampler
from .brain import DeepBrain
from . import distributed as D
from . import evaluation as E


def worker(rank, world_size, cfg_dict, data_meta):
    cfg = Cfg.from_dict(cfg_dict)
    torch.manual_seed(cfg.seed)
    dev = D.init_worker(rank, world_size, cfg)

    if dev.startswith("cuda"):
        print(f"  [rank {rank}] device cuda:{torch.cuda.current_device()} "
              f"({torch.cuda.get_device_name(torch.cuda.current_device())})", flush=True)
    else:
        print(f"  [rank {rank}] device cpu", flush=True)

    # tokenizer + true vocab (trust the data: some tokenizers emit ids past
    # vocab_size, which would gather out of bounds in codes[token]).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.model_max_length = 10 ** 9
    np_dtype = {"uint16": np.uint16, "uint32": np.uint32}[data_meta["dtype"]]
    length = int(data_meta["token_count"])
    full = np.memmap(data_meta["bin_path"], dtype=np_dtype, mode="r")[:length]
    V = max(int(data_meta["vocab"]), len(tok), int(full.max()) + 1)

    # ── the deep brain (seed-deterministic, identical across ranks) ──
    brain = DeepBrain(V, cfg, dev)

    # ── shared setup on the global head of the corpus (rank-independent) ──
    setup_n = min(cfg.block, length - 1)
    setup_block = torch.from_numpy(np.asarray(full[:setup_n], dtype=np.int64)).to(dev)
    if rank == 0 and cfg.learn_feat:
        print("  brain: learning granule features per layer (competitive, no backprop) ...",
              flush=True)
    brain.setup(setup_block)
    # average the competitively-learned granules once so every replica is an
    # identical encoder (belt-and-suspenders against cross-rank CUDA nondeterminism)
    D.average_granules(brain, world_size)
    del setup_block, full

    # ── sharded training loop ──
    batch = cfg.batch
    steps_per_block = max(1, cfg.block // batch)
    total_steps = max(1, cfg.train_tokens // (batch * max(1, world_size)))
    n_blocks = max(1, total_steps // steps_per_block)

    sampler = WindowSampler(data_meta["bin_path"], data_meta["dtype"], length,
                            cfg.ctx, batch, dev, rank, world_size,
                            pin=(dev != "cpu"), seed=cfg.seed + 2 + rank)

    if rank == 0:
        print(f"  brain: {cfg.n_layers} layers x {n_blocks} blocks x {steps_per_block} "
              f"steps x batch {batch} per rank ({world_size} rank(s))", flush=True)
    t0 = time.time()
    consumed = 0
    for blk in range(n_blocks):
        consumed += brain.train_on_block(sampler, steps_per_block)
        if cfg.learn_feat and cfg.refine_every and (blk + 1) % cfg.refine_every == 0:
            brain.refine(sampler)
        if world_size > 1 and (blk + 1) % cfg.sync_every == 0:
            D.average_brain(brain, world_size)
        if rank == 0 and (blk + 1) % max(1, n_blocks // 20) == 0:
            if dev.startswith("cuda"):
                torch.cuda.synchronize()
            tot = consumed * world_size
            el = time.time() - t0
            print(f"    trained ~{tot:,} tokens  ({el:.0f}s, {tot/max(el,1e-6):,.0f} tok/s)",
                  flush=True)

    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    train_s = time.time() - t0                     # pure training wall-clock (pre-eval)

    if world_size > 1:
        D.average_brain(brain, world_size)         # final mandatory average
        dist.barrier(device_ids=[rank])
        dist.destroy_process_group()
        if rank != 0:
            return

    # ── eval + scoreboard on rank 0 only ──
    if rank == 0:
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id)
        model = base.to(dev).half() if dev.startswith("cuda") else base
        test_stream = E.load_wikitext_test(tok, cfg.eval_tokens, dev)
        win = min(cfg.eval_window,
                  int(getattr(model.config, "max_position_embeddings", cfg.eval_window)))
        llm_ppl, _, llm_t = E.eval_llm_ppl(model, test_stream, win, cfg.eval_stride, dev)
        llm_macs = sum(p.numel() for p in model.parameters())
        print(f"\n  LLM (pretrained, strided window {win}): perplexity {llm_ppl:.2f}  "
              f"{llm_macs:,} MACs/token  ({llm_t:.1f}s)", flush=True)
        del model, base
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

        br_ppl, _ = brain.eval_ppl(test_stream, cfg.eval_tokens)
        per_layer, mix = brain.eval_per_layer_ppl(test_stream, cfg.eval_tokens)
        br_macs = brain.macs_per_token()
        total_consumed = consumed * world_size
        print(f"  Brain (no-bp, {cfg.n_layers}L): perplexity {br_ppl:.2f}  "
              f"{br_macs:,} MACs/token  (trained {total_consumed:,} tok in "
              f"{train_s:.0f}s, {total_consumed/max(train_s,1e-6):,.0f} tok/s)", flush=True)
        E.run_scoreboard(llm_ppl, llm_macs, br_ppl, br_macs, total_consumed,
                         per_layer=per_layer, mix=mix)
