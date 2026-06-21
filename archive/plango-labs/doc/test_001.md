# Test 001 — Binary Majority Classification

**Date:** 2026-06-20
**Status:** Complete

---

## Objective

First head-to-head between a dense (LLM-style) and sparse (brain-style) network on an identical task. Baseline to measure efficiency gap as architecture complexity grows.

---

## Task

- Input: 10-bit binary vector
- Label: 1 if `sum(x) > 5`, else 0
- Linear separability: yes (trivially)
- Dataset: 1,000 train / 500 test, balanced by construction

---

## Networks

**Dense**
- Single linear layer, sigmoid output
- Update: gradient descent on binary cross-entropy
- `pred = sigmoid(W · x + b)`, `ΔW = -(pred - target) * x * lr`
- Every input used on every pass regardless of value

**Sparse**
- Same shape, threshold-based output
- Update: supervised Hebb — only fires on misclassification, only touches active inputs
- `spike = (W[active].sum() + b >= threshold)`
- Ops scale with number of active bits, not total input dimension

---

## Config

| Param | Value |
|---|---|
| Input dim | 10 |
| Epochs | 50 |
| LR | 0.05 |
| Threshold (sparse) | 0.3 |
| Seed | 0 |

---

## Results

| Metric | Dense | Sparse |
|---|---|---|
| Final accuracy | 100.0% | 100.0% |
| Epoch @ 90% | 1 | 2 |
| Total ops | 500,000 | 254,250 |
| Ops ratio | 1.0x | **2.0x fewer** |

---

## Accuracy per epoch

| Epoch | Dense | Sparse |
|---|---|---|
| 1 | 98.8% | 84.2% |
| 6 | 100.0% | 100.0% |
| 11–50 | 100.0% | 100.0% |

---

## Notes

- 2x ops reduction matches theory: ~50% of bits are on per sample, so sparse processes ~5 inputs per forward pass vs 10.
- Dense converges one epoch faster (epoch 1 vs 2). Cost: double the compute.
- Both plateau at 100% — task is too easy to distinguish quality of learning.
- 2x is the floor for this input distribution. Real gains come from pushing input sparsity lower and stacking layers.

---

## Next

- Test 002: harder task (non-linearly separable), add hidden layer to both, measure whether sparse learning degrades
- Increase input dim to 100+ to amplify the ops gap
- Push input sparsity to 5–10% to approach brain territory
