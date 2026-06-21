#!/usr/bin/env python3
"""BrainFormer v13-v2 — entrypoint.

A DEEP, no-backprop cerebellar language model vs a real pretrained LLM, scored on
WikiText-103 perplexity. The brain is a stack of locally-trained cerebellar
layers with hyperdimensional long-context binding and continuous granule
refinement. See README.md for the architecture.

Kaggle ("GPU T4 x2"):
    !python run.py --build-data     # tokenize once into the memmap cache
    !python run.py                  # auto dual-GPU when 2 are present
    !python run.py --single         # force single GPU
    !python run.py --smoke          # tiny, CPU-OK, exercises the whole path
"""
import warnings

# config sets the CUDA/NCCL/tokenizer env vars at import, before torch is touched.
from bf.config import parse_args, cfg_from_args
from bf.data import resolve_data_dir, build_memmap, ensure_memmap
from bf.train import worker
from bf import distributed as D


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    cfg.data_dir = resolve_data_dir(cfg)

    warnings.filterwarnings("ignore")
    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    import torch
    import torch.multiprocessing as mp

    # device_count() reads the driver WITHOUT creating a CUDA context, so it is
    # safe in the parent before spawn. is_available()/.to(cuda) would poison
    # spawned children.
    ngpu = torch.cuda.device_count()
    use_mp = (not args.single) and (not args.smoke) and ngpu >= 2

    print("=" * 70)
    mode = "dual-GPU" if use_mp else ("smoke" if args.smoke else "single")
    print(f"  TEST 013 v2: DEEP Brain vs a real LLM   (gpus={ngpu}, "
          f"model={cfg.model_id}, layers={cfg.n_layers}, ctx={cfg.ctx}, mode={mode})")
    print("=" * 70)
    D.print_topology()

    # tokenizer in the parent (CPU only) for the data build + vocab.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    tok.model_max_length = 10 ** 9

    if args.build_data:
        meta = build_memmap(cfg, tok)
        print(f"  done: {meta['token_count']:,} tokens at {meta['bin_path']}")
        return

    if args.smoke:
        meta = ensure_memmap(cfg, tok)
        worker(0, 1, cfg.to_dict(), meta)
        return

    if use_mp:
        meta = ensure_memmap(cfg, tok)            # MUST finish before any spawn
        cfg.master_port = D.pick_free_port()
        world_size = ngpu
        mp.set_start_method("spawn", force=True)
        mp.spawn(worker, args=(world_size, cfg.to_dict(), meta),
                 nprocs=world_size, join=True)
    else:
        meta = ensure_memmap(cfg, tok)
        worker(0, 1, cfg.to_dict(), meta)


if __name__ == "__main__":
    main()
