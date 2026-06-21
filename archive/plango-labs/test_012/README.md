# Test 012 — LLM vs the Brain, on words (C++)

The head-to-head moves from pictures to language. Same contest, new task:

- The LLM way: a neural language model (Bengio 2003, ancestor of today's LLMs).
  Learned word embeddings, a hidden layer, softmax over the vocabulary, trained by
  backpropagation (AdamW, dropout, early stopping).
- The Brain way: a cerebellar n-gram coder. Fixed random word codes, a
  sparsely-wired bank of granule cells whose features are tuned by competitive
  learning, homeostatic kWTA (only a few fire), and one linear readout trained by
  the local delta rule. No backpropagation anywhere.

Task: see the previous 3 words, predict the next. Scored on perplexity (lower is
better), top-1/5/10 accuracy, multiply-adds per prediction (the energy proxy), and
wall-clock time. Full write-up: [`../doc/test_012.md`](../doc/test_012.md).

## Build and run (one command, cross-platform, no external libraries)

```sh
cd test_012 && make run
```

Needs a C++17 compiler and `make`. The corpus (`data/corpus.txt`, Tiny
Shakespeare) is committed, so there is no data step and nothing to download.
Tokenizing, vocab building, training, scoring, and generation happen in one
binary. The same sources build on macOS, Linux, and Windows (mingw or MSVC).

## Standing

Final validated result (see the doc for the full run-by-run history, including the
refuted ideas: distributional codes, hippocampal caches, a bigger LLM, longer
context):

| system | test perplexity | bits/word | MACs / prediction | backprop |
|---|---|---|---|---|
| n-gram interpolated (reference) | 564 | - | - | no |
| LLM (regularized, best-tuned) | 462.3 | 8.773 | 542,720 | yes |
| Brain (flat readout) | 408.5 | 8.581 | 131,072 (0.24x) | no |
| Brain (hierarchical + weight decay) | 415.1 | 8.546 | 35,840 (0.066x) | no |

A no-backprop, brain-style model beats the regularized neural LM on perplexity at
about 1/15th of the inference compute (and about 1/3 of the training compute at
about 5x the throughput). The Brain's one losing axis is parameter/memory
footprint (8.5M vs 0.74M params), but it reads only 24 of 2048 rows per word, so
inference stays cheap. The biggest quality lever was competitive granule feature
learning; the biggest cost lever was the hierarchical class-then-word readout.

## Source map

| file | what it is |
|---|---|
| `src/config.h` | every knob (CTX, vocab, layer and granule sizes) |
| `src/common.h` | RNG, timing, softmax/argmax, `RESTRICT` (header-only) |
| `src/corpus.{h,cpp}` | reads and tokenizes the text, builds the vocabulary |
| `src/threads.{h,cpp}` | portable `parallel_for` thread pool (from test_011) |
| `src/llm.{h,cpp}` | the neural LM: batched AdamW backprop, dropout, early stop |
| `src/brain.{h,cpp}` | the cerebellar coder: random codes, learned granules, delta-rule readout |
| `src/analytics.{h,cpp}` | n-gram baselines, top-k, head-to-head, granule health |
| `src/main.cpp` | runs the head-to-head, prints the scoreboard and diagnostics |

To use your own text, replace `data/corpus.txt` and rebuild.
