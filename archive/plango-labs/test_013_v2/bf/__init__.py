"""BrainFormer v13-v2 — a DEEP, no-backprop cerebellar language model.

A stack of locally-trained cerebellar layers (each: fixed random expansion ->
competitively-learned granule features -> kWTA sparse code -> hierarchical
delta-rule readout), with hyperdimensional context binding for long context and
a learned layer-mixture on top. No backpropagation anywhere; every layer has its
own local next-token objective, so credit assignment never crosses a layer.

Modules:
  config       Cfg dataclass + smoke scaling + CLI parsing
  data         memmap builder, WindowSampler (async double-buffer prefetch)
  codes        fixed token codes + hyperdimensional context binding
  kernels      fused activation, two-level approximate topk, flat scatter update
  granules     one cerebellar layer's front-end (wiring, k-means, kWTA, proj)
  readout      hierarchical class->word delta-rule head + EMA
  layer        CerebellarLayer = granules + readout + local objective
  brain        DeepBrain = stack of layers + learned mixture
  distributed  NCCL init + local-SGD averaging
  evaluation   LLM strided perplexity, WikiText loader, scoreboard
  train        per-rank worker (setup, loop, sync, eval)
"""

__version__ = "2.0.0"
