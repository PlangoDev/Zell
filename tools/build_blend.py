#!/usr/bin/env python3
"""Zell: blended multi-source dataset builder (Qwen2.5 tokenizer port).

Streams several high-quality Hugging Face sources, plus local synthetic JSONL
(OpenRouter-generated chat + tool-call data), interleaves them by deterministic
per-source TOKEN QUOTAS into ONE memmap, serializes chat/instruction sources to
the native Qwen2.5 ChatML template, holds out a chat-eval split (no leakage),
and writes a meta.json the trainer reads.

This is one MIXED corpus: general text + code + math + chat + synthetic
tool-call data, all interleaved. The core is continued-pretrained on the whole
mix, so coherence, code, chatting, and tool-call formatting are all present from
the first step.

Run on Kaggle in a dedicated **Internet-ON** notebook (separate from training):
    python build_blend.py --pool-tokens 50000000 --out-dir /kaggle/working \
        --synth-glob "/kaggle/input/zell-synth/*.jsonl"

Then train internet-OFF against the emitted meta.json.

Port notes (vs the BrainFormer Pythia build):
- Tokenizer is Qwen2.5 (~151k vocab) -> token ids exceed uint16, so the memmap
  is uint32. The old `assert vocab < 65536` is gone.
- Chat is serialized with the tokenizer's native ChatML template (im_start/
  im_end), not hand-written <|user|> markers, so SFT uses tokens the core knows.
- Doc separator is the Qwen <|endoftext|> id (not Pythia id 0).
- Held-out hashing is on raw first-user text, so it is tokenizer-independent and
  the same conversations stay held out across the re-tokenization.
- NEVER pass num_proc to a streaming map (deadlocks). Batched tokenization gives
  multi-core throughput on its own.
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hfcache")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# kind: "text" (plain field) | "chat_messages" ([{role,content}]) | "chat_dolly"
#       | "synth_jsonl" (local files, each line {"messages":[...]}).
# weight = fraction of the pool token budget.
SOURCES = [
    {"name": "fineweb-edu", "hf": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
     "split": "train", "field": "text", "kind": "text", "weight": 0.38},
    {"name": "wikipedia", "hf": "wikimedia/wikipedia", "config": "20231101.en",
     "split": "train", "field": "text", "kind": "text", "weight": 0.12},
    {"name": "cosmopedia", "hf": "HuggingFaceTB/cosmopedia", "config": "web_samples_v2",
     "split": "train", "field": "text", "kind": "text", "weight": 0.12},
    {"name": "open-web-math", "hf": "open-web-math/open-web-math", "config": None,
     "split": "train", "field": "text", "kind": "text", "weight": 0.10},
    {"name": "github-code", "hf": "codeparrot/github-code-clean", "config": "Python-all",
     "split": "train", "field": "code", "kind": "text", "weight": 0.10},
    {"name": "ultrachat", "hf": "HuggingFaceH4/ultrachat_200k", "config": None,
     "split": "train_sft", "field": "messages", "kind": "chat_messages", "weight": 0.06},
    {"name": "smoltalk", "hf": "HuggingFaceTB/smoltalk", "config": "all",
     "split": "train", "field": "messages", "kind": "chat_messages", "weight": 0.03},
    {"name": "dolly", "hf": "databricks/databricks-dolly-15k", "config": None,
     "split": "train", "field": None, "kind": "chat_dolly", "weight": 0.005},
    # local OpenRouter-generated chat + tool-call corpus (see synth/generate.py).
    # files matched by --synth-glob; if none found the source is dropped and its
    # weight is renormalized away.
    {"name": "synth", "hf": None, "config": None, "split": None,
     "field": "messages", "kind": "synth_jsonl", "weight": 0.085},
]

EVAL_HASH_MOD = 500      # hold out ~1/500 of ultrachat for chat eval (no leakage)
EVAL_MAX_CONV = 2000     # cap held-out conversations


def conv_key(text):
    return int(hashlib.md5(text.encode("utf-8", "ignore")).hexdigest(), 16)


def _msgs_to_chatml(tok, msgs):
    """Render a [{role,content}] conversation to a Qwen ChatML string.

    Uses the tokenizer's native template (tokenize=False) so the special tokens
    are exactly the ones the core was pretrained on. Drops empty turns first.
    """
    clean = []
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role in ("system", "user", "assistant", "tool") and content:
            clean.append({"role": role, "content": content})
    if not clean:
        return ""
    return tok.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)


def serialize(tok, src, row):
    """Return (train_text_or_None, eval_text_or_None). eval_text is set only for
    held-out ultrachat conversations (and excluded from train)."""
    k = src["kind"]
    if k == "text":
        t = row.get(src["field"])
        return (t if t else None), None
    if k == "chat_dolly":
        instr = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        resp = (row.get("response") or "").strip()
        if not instr or not resp:
            return None, None
        user = instr + ("\n\n" + ctx if ctx else "")
        return _msgs_to_chatml(tok, [{"role": "user", "content": user},
                                     {"role": "assistant", "content": resp}]) or None, None
    if k in ("chat_messages", "synth_jsonl"):
        msgs = row.get("messages") or []
        if not msgs:
            return None, None
        if src["name"] == "ultrachat":
            first_user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
            if conv_key(first_user) % EVAL_HASH_MOD == 0:
                return None, _msgs_to_chatml(tok, msgs)   # eval, exclude from train
        return _msgs_to_chatml(tok, msgs) or None, None
    return None, None


def _synth_iter(paths):
    """Yield {"messages": [...]} rows from local JSONL files."""
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("messages"):
                    yield obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-tokens", type=int, default=4_500_000_000)
    ap.add_argument("--train-tokens", type=int, default=6_000_000_000,
                    help="training budget the trainer re-samples the pool toward")
    ap.add_argument("--eval-tokens", type=int, default=1_500_000)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--synth-glob", default="",
                    help="glob for local synthetic JSONL (OpenRouter chat+tools)")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.model_max_length = 10 ** 9
    vocab = int(len(tok))
    # uint16 cannot hold Qwen ids; uint32 covers any realistic vocab.
    dtype = np.uint32
    sep_id = tok.convert_tokens_to_ids("<|endoftext|>")
    if sep_id is None or sep_id < 0:
        sep_id = tok.eos_token_id
    print(f"  tokenizer {args.model}: vocab {vocab:,}, doc-sep id {sep_id}", flush=True)

    train_path = os.path.join(args.out_dir, "tokens.bin")
    eval_path = os.path.join(args.out_dir, "chat_eval.bin")
    meta_path = os.path.join(args.out_dir, "meta.json")

    from datasets import load_dataset

    # ---- resolve synthetic source (drop + renormalize if absent) ----
    synth_paths = sorted(glob.glob(args.synth_glob)) if args.synth_glob else []
    sources = [dict(s) for s in SOURCES]
    if not synth_paths:
        dropped = next((s for s in sources if s["kind"] == "synth_jsonl"), None)
        if dropped:
            print(f"  synth: no files matched {args.synth_glob!r}; dropping synth source", flush=True)
            sources = [s for s in sources if s["kind"] != "synth_jsonl"]
    else:
        print(f"  synth: {len(synth_paths)} file(s) matched {args.synth_glob!r}", flush=True)
    wsum = sum(s["weight"] for s in sources)
    for s in sources:
        s["weight"] /= wsum

    # ---- per-source quotas + streaming iterators ----
    quota = {s["name"]: int(round(s["weight"] * args.pool_tokens)) for s in sources}
    written = {s["name"]: 0 for s in sources}
    iters = {}
    for s in sources:
        if s["kind"] == "synth_jsonl":
            iters[s["name"]] = _synth_iter(synth_paths)
        else:
            ds = load_dataset(s["hf"], s["config"], split=s["split"], streaming=True)
            ds = ds.shuffle(seed=args.seed, buffer_size=10_000)
            iters[s["name"]] = iter(ds)
    active = [s["name"] for s in sources]
    src_by_name = {s["name"]: s for s in sources}

    train_arr = np.memmap(train_path, dtype=dtype, mode="w+", shape=(args.pool_tokens,))
    cur = 0
    eval_ids = []
    eval_convs = 0

    print(f"  blend: pool {args.pool_tokens:,} tokens across {len(sources)} sources", flush=True)
    t0 = time.time()
    docbuf = []
    BATCH = 1000

    def flush_train():
        nonlocal cur
        if not docbuf:
            return
        encs = tok(docbuf, add_special_tokens=False)["input_ids"]
        docbuf.clear()
        flat = []
        for e in encs:
            flat.extend(e)
            flat.append(sep_id)
        if flat:
            m = min(len(flat), args.pool_tokens - cur)
            if m > 0:
                train_arr[cur:cur + m] = np.asarray(flat[:m], dtype=dtype)
                cur += m

    rr = 0
    while active and cur < args.pool_tokens:
        name = active[rr % len(active)]
        rr += 1
        s = src_by_name[name]
        if written[name] >= quota[name]:
            active.remove(name)
            continue
        try:
            row = next(iters[name])
        except StopIteration:
            active.remove(name)
            continue
        train_text, eval_text = serialize(tok, s, row)
        if eval_text is not None and eval_convs < EVAL_MAX_CONV:
            ids = tok(eval_text, add_special_tokens=False)["input_ids"]
            eval_ids.extend(ids); eval_ids.append(sep_id)
            eval_convs += 1
            continue
        if not train_text:
            continue
        docbuf.append(train_text)
        approx = max(1, len(train_text) // 4)   # cheap len proxy; exactness from flush clamp
        written[name] += approx
        if len(docbuf) >= BATCH:
            flush_train()
            if (cur // 50_000_000) != ((cur - 1) // 50_000_000):
                dt = time.time() - t0
                print(f"    {cur:,}/{args.pool_tokens:,} tokens  ({dt:.0f}s, {cur/max(dt,1):,.0f} tok/s)", flush=True)
    flush_train()
    train_arr.flush()
    del train_arr
    real_pool = cur
    print(f"  blend: built {real_pool:,} pool tokens in {time.time()-t0:.0f}s", flush=True)

    eval_ids = eval_ids[: args.eval_tokens]
    np.asarray(eval_ids, dtype=dtype).tofile(eval_path)
    print(f"  blend: {len(eval_ids):,} held-out chat-eval tokens ({eval_convs} convs) -> {eval_path}", flush=True)

    meta = {
        "model_id": args.model,
        "vocab": vocab,
        "dtype": "uint32",
        "train_path": train_path,
        "train_tokens": int(real_pool),
        "test_path": eval_path,
        "test_tokens": int(len(eval_ids)),
        "eval_window": args.window,
        "eval_stride": args.stride,
        "doc_separator_id": int(sep_id),
        "build_seed": args.seed,
        "pool_tokens": int(real_pool),
        "train_budget": int(args.train_tokens),
        "sources": [{"name": s["name"], "hf": s["hf"], "weight": round(s["weight"], 5),
                     "tokens": written[s["name"]]} for s in sources],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  blend: wrote {meta_path}")
    print(json.dumps({k: meta[k] for k in ("train_tokens", "test_tokens", "vocab", "dtype")}, indent=2))


if __name__ == "__main__":
    main()
