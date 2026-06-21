# Test 011 — LLM vs the Brain (in C)

A head-to-head, written from scratch in multithreaded C, between:

- **The LLM way** — a dense, fully-connected 2-layer neural net trained by
  **backpropagation** (the standard recipe behind every mainstream AI).
- **The Brain way** — a **Cerebellar Sparse Coder**: a committee of circuits that
  use fixed local receptive fields, whitening, **competitive (k-means) feature
  learning**, sparse coding, and a single linear readout trained by a local rule.
  **No backpropagation anywhere.**

Same data (Fashion-MNIST, "sort the laundry"), same test set. Scored on accuracy
**and** on multiply-adds per image (the energy proxy) **and** on wall-clock time.

## The result

On the full 60,000-image dataset, the brain **beats the LLM on both axes at once**:

| | accuracy | work / image | backprop? |
|---|---|---|---|
| LLM (dense MLP) | ~88.1% | 118,272 | yes |
| **Brain (1 expert)** | **~88.5%** | **112,000** | **no** |
| Brain (3-expert committee) | ~90.0% | 336,000 | no |

A single brain expert is **both more accurate and cheaper**, with no backprop.
The committee pushes accuracy to ~90%. On the images where the two models
disagree, the brain wins the majority. Full per-class, confusion-matrix, and
committee-diversity diagnostics print at the end of every run.

## Build & run

**macOS / Linux** (needs a C compiler + `make`):

```sh
make data     # one-time: writes data/*.bin (needs python3 + scikit-learn)
make          # builds ./showdown  (-O3 -march=native -ffast-math, multithreaded)
make run      # or just ./showdown
```

**Windows** — two easy options:

- *MSYS2 / mingw-w64* (recommended): install it, then the same `make` commands
  work as-is (`pacman -S mingw-w64-x86_64-gcc make`).
- *MSVC* (`cl`): compile the sources directly, e.g.
  ```
  cl /O2 /fp:fast src\*.c /Fe:showdown.exe
  ```
  The threading layer (`src/threads.c`) auto-selects the Win32 API on Windows,
  so no code changes are needed.

The dataset binaries (`data/*.bin`) are committed, so on any machine you can skip
`make data` and just build + run. To regenerate them, `make data` needs Python
with `scikit-learn`.

## How it's wired (source map)

| file | what it is |
|---|---|
| `src/common.h` | fast RNG, timing, softmax/argmax (header-only) |
| `src/dataset.{h,c}` | loads the binary Fashion-MNIST |
| `src/threads.{h,c}` | portable `parallel_for` (pthreads / Win32) |
| `src/llm.{h,c}` | the dense backprop MLP (the bar to beat) |
| `src/brain.{h,c}` | the Cerebellar Sparse Coder (the challenger) |
| `src/analytics.{h,c}` | per-class, confusion, agreement, committee diagnostics |
| `src/main.c` | runs the head-to-head, prints everything live |
| `tools/export_data.py` | one-time data bridge (not part of the test) |

## Why the brain wins without backprop

Every earlier experiment showed the brain's only weakness was its *learning rule*
(local feedback can't train deep features as sharply as backprop). So this design
**sidesteps deep learning entirely**, the way the cerebellum does: a big bank of
fixed/locally-learned feature detectors, with learning only at a shallow readout
that a local rule masters. Cheap per circuit, so we can run a committee and still
come in cheaper than one dense net.
