"""Opponent evaluation + scoreboard.

The LLM number is the standard strided sliding-window perplexity (the LLM's fair,
strong number: each target sees up to window-stride tokens of real context). The
brain is scored on the same WikiText-103 test stream and tokenizer.
"""
import math
import time

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_llm_ppl(model, stream, window, stride, device):
    model.eval()
    ids = stream.long().to(device)
    n = ids.numel()
    nll = torch.zeros((), device=device)
    ntok = torch.zeros((), dtype=torch.long, device=device)
    prev_end = 0
    t0 = time.time()
    for begin in range(0, n - 1, stride):
        end = min(begin + window, n)
        inp = ids[begin:end].unsqueeze(0)
        tgt = inp.clone()
        new = end - prev_end
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


def run_scoreboard(llm_ppl, llm_macs, br_ppl, br_macs, consumed, per_layer=None, mix=None):
    print("\n" + "=" * 70)
    print("  SCOREBOARD   (WikiText-103 test, same tokenizer/stream)")
    print("=" * 70)
    print(f"    {'system':<28}{'perplexity':>12}{'MACs/token':>16}{'backprop':>10}")
    print(f"    {'LLM (pretrained)':<28}{llm_ppl:>12.2f}{llm_macs:>16,}{'yes':>10}")
    print(f"    {'Brain ('+str(consumed//1_000_000)+'M tok, no-bp)':<28}"
          f"{br_ppl:>12.2f}{br_macs:>16,}{'NO':>10}")
    if per_layer is not None:
        print(f"    {'-'*64}")
        for i, p in enumerate(per_layer):
            wtxt = f"   w={mix[i]:.3f}" if mix is not None else ""
            print(f"      layer {i} ppl {p:>10.2f}{wtxt}")
    print(f"    {'-'*64}")
    print(f"    brain runs at {br_macs / llm_macs:.5f}x the LLM's compute per token.")
    print("=" * 70)
