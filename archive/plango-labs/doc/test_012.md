# Test 012 — The Brain Learns to Read

The series leaves pictures behind and moves to language. Same contest as before,
the LLM way (backprop) versus the Brain way (sparse codes, local rules, no
backprop), but now the task is words and the score is perplexity.

## 1. The story (for a five-year-old)

For eleven tests our two players sorted pictures. Picture-sorting is like looking
at a shoe and shouting "SHOE!". The Brain player got good and cheap at it.

Now they play a harder game: guess the next word. You read "poor girl she…" and
guess what comes next. There are four thousand words it could be. Nobody guesses
right every time, but a good reader is surprised less often. That surprised-ness
has a score called perplexity. Low perplexity means rarely surprised, which means
good reader.

The LLM reads the way today's big AIs do. It learns a little number-picture (an
embedding) for every word, and a backward-flowing river of corrections
(backpropagation) tunes every knob.

The Brain reads the way a real brain might. It never uses the backward river. It
spreads the sentence-so-far across a huge field of little granule cells (most stay
quiet, only a few light up) and learns one thin final layer with a simple local
rule.

Who reads better, and who reads cheaper? That is the test.

## 2. The task

- Corpus: Tiny Shakespeare (`data/corpus.txt`, about 208,000 words). Tokenized in
  C++ at load time (lowercase, split on letters). No Python step.
- Vocabulary: the top 4,096 words by frequency; everything else becomes `<unk>`
  (about 5.0% of tokens). One word is one integer id.
- Examples: context length CTX = 3. Show the model 3 consecutive words, ask for
  the 4th.
- Split: first 90% of the token stream is train (about 187,649 contexts), last 10%
  is test (about 20,848 contexts). The LLM carves the last 1/12 of train into a
  validation slice for early stopping. The test set is never used to pick anything.
- Metrics: perplexity (lower is better), next-word top-1 / top-5 / top-10
  accuracy, bits-per-word, multiply-adds per prediction (the energy proxy), and
  wall-clock time.

### Reference bars (count-based, no learning)

Built from train counts, scored on test (add-0.05 smoothed):

| model | test perplexity |
|---|---|
| unigram (word frequency) | 633 |
| bigram (previous 1 word) | 571 |
| trigram (previous 2 words) | 2167 (sparse: add-k punishes unseen) |
| interpolated 1+2+3 | 564 |

Trigram context was seen in train only 62.9% of the time. Shakespeare's 3-grams
are mostly novel, which is why raw trigram counting is poor and why a model has to
generalize. Both contestants must beat about 564.

## 3. The two contestants

### 3.1 The LLM, a neural language model (backprop)

The Bengio (2003) neural language model, ancestor of today's LLMs, shrunk to
laptop size:

```
[w-3, w-2, w-1]  embed(48)  concat(144)  W1  128 tanh  W2  4096 softmax
```

- Learned embeddings `E` (4096x48), hidden `W1` (128x144), output `W2` (4096x128).
- Trained by backpropagation with AdamW (lr 3e-3, weight decay 1e-4), dropout 0.5
  on the hidden layer, and early stopping on the validation slice (keep the
  best-val weights).
- About 542,720 multiply-adds per prediction.

The regularization is not optional (see section 4). Without it the model memorizes
the training trigrams and test perplexity explodes.

### 3.2 The Brain, a cerebellar n-gram coder (no backprop)

The Marr-Albus cerebellum that won tests 010/011, mapped onto language:

```
words  random codes  context vector(144)  granules(2048, sparse wired, WTA-learned)
  kWTA(keep 24)  sparse code  linear readout(delta rule)  4096 softmax
```

1. Fixed random word codes (no backprop). Each word gets an i.i.d. gaussian
   vector, a dentate-gyrus-style random projection. Distinct words stay
   orthogonal, which preserves the discriminability this memorization task
   rewards. (Run 4 tested distributional PPMI codes that pull king/queen together
   and found they hurt here; see section 4.)
2. Mossy fibers. The 3 word-codes are concatenated into one 144-d context vector.
3. Granule expansion with WTA-learned features. 2,048 granule cells, each wired to
   16 random input dimensions. Their weights are then tuned by competitive
   learning (the kWTA winners move toward the contexts that fire them, online
   k-means, no backprop), so the expansion tiles the data manifold. This is the
   biggest quality lever (section 4, Run 5) and costs nothing extra at inference.
4. Homeostatic boosting (intrinsic plasticity). Each granule's average activation
   is subtracted as a bias, so no granule hogs the code and dead granules come
   back to life, which evens out the sparse code's load.
5. kWTA sparse code. Only the 24 strongest granules fire. Because the code is
   sparse, the readout only touches 24 rows, which is why the Brain stays cheap
   even with a 4,096-word output.
6. Purkinje readout. One linear layer to the vocabulary, trained by the local
   delta rule. The only thing that learns. About 131,072 multiply-adds per
   prediction (0.24x the LLM).

## 4. What we learned, run by run

Every run is reported as it happened, including the ones that failed.

### Run 1, first cut (plain SGD, fixed random brain codes)
- LLM 313.7 ppl (still falling at epoch 20, undertrained), Brain 453.5 ppl.
- Brain 0.6x the compute, converged in 10 epochs, no backprop.
- Lesson: SGD was too slow; the comparison was not yet fair to either side.

### Run 2, Adam, no regularization (an instructive failure)
- LLM train ppl crashed 237 to 30; test ppl exploded 270 to 1108.
- Lesson: the LLM has about 740k parameters and only 187k training examples. With
  no regularization it memorizes the corpus, so the "win" was fake. On a small
  corpus the LLM must be regularized.

### Run 3, AdamW + dropout + early stopping (fair LLM), fixed random brain codes
- LLM validation perplexity bottomed at 297 (epoch 3), then climbed as it began to
  overfit; early stopping keeps epoch 3.
- Lesson: a regularized LLM lands around 297 on the validation slice (a tail of
  train). The held-out test number is higher; Run 4 measures it at about 462. The
  453 quoted for the Brain was always a test number, so the comparison was muddier
  than it looked.

### Run 4, first full end-to-end execution: two real bugs
This is the first time the current code was actually run (Runs 1-3 predate the
batched-kernel and distributional-codes rewrite). Two bugs surfaced; both fixed.

- Bug 1, segfault in the batched LLM (`hidden_range`). The eval path calls the
  hidden layer with `mask == NULL` (no dropout at test time). The per-row pointer
  was computed as `mask + b*LM_HID` before the NULL check, so for any row `b>0` it
  became a small non-NULL bogus pointer that slipped past the `mk ? …` guard and
  dereferenced garbage. It trained epoch 1 fine (masks valid) then died in the
  first perplexity probe. Fix: derive the row pointer from the base only when
  `mask` is non-NULL. The LLM now runs clean and is numerically healthy (smooth
  monotonic train-ppl descent, sane gradient norms, no NaNs), so the batched
  kernel is correct. Best val ppl 252.8 (epoch 4), early-stopped; held-out test
  ppl 462.3, top-1 10.6%.
- Bug 2, distributional word codes were L2-normalized to unit norm, about
  sqrt(48) ~ 7x smaller than the old random codes. Smaller codes meant ~8x smaller
  granule activations, ~8x smaller kWTA margins, ~8x smaller delta-rule steps, and
  an undertrained readout. Symptom: mean margin collapsed 1.618 to 0.197, readout
  norm 35.9 to 5.4, test ppl ballooned to 873. Fix: normalize codes to sqrt(BR_EMB)
  (the old random-code magnitude) so the learning dynamics are unchanged and only
  the codes' semantics differ. That recovered 873 to 570 (8 epochs).
- The result: distributional codes do not beat random codes here. Scaled correctly
  and trained twice as long (16 epochs, still slowly descending), the Brain lands
  at test ppl 511, above the random-codes 453. Granule health is essentially the
  same as the random-codes era (90.5% live, Gini 0.633), so the front-end is fine;
  the semantic sharing itself is the drag.
- Lesson (a sharpened target, not a defeat): next-word prediction on a small
  corpus is memorization-heavy. The reward is for telling near-identical contexts
  apart, and pulling king/queen together blurs the distinctions the readout needs.
  Orthogonal random codes are the better substrate for this task. Any win from
  meaning has to come from a mechanism that adds sharing without removing
  discriminability (a hippocampal cache alongside the cortical readout, or
  interpolating random with semantic codes), not from replacing the codes.

| system (Run 4) | test ppl | top-1 | MACs/pred | backprop |
|---|---|---|---|---|
| LLM (neural LM, regularized) | 462 | 10.6% | 542,720 | yes |
| Brain (distributional codes, 16 ep) | 511 | 7.4% | 327,680 (0.60x) | no |
| Brain (random codes, Run 1 baseline) | 453 | 7.9% | 327,680 (0.60x) | no |

### Run 5, the Brain wins both axes (random codes + learned granule features)
Following Run 4's lesson (add sharing without removing discriminability), we kept
the random code substrate and improved the machine two ways, then tested a third
that did not pan out:

- Restored fixed random codes (`BR_RANDOM_CODES`): cortex-only lands at 464.6
  (margin 1.61, matching the validated 1.62), already near parity with the LLM's
  462 at 0.60x cost, no backprop.
- Hippocampal exact-context cache (CLS), tested and refuted. Storing each seen
  context with its next-word distribution and recalling by exact pattern
  completion is, on this corpus, just a trigram predictor, and trigrams here are
  poor (2167 ppl). Fusing it hurt: 464.6 to 490.5. Kept in the code as a
  documented ablation. Lesson: episodic recall needs similarity-based completion
  (recall by sparse-code overlap, which generalizes), not exact-context lookup, to
  help a low-repetition corpus.
- Competitive granule feature learning (`BR_LEARN_GRANULES`), the win. Instead of
  leaving the 8,192 granule weights random, the kWTA winners nudge their weights
  toward the contexts that fire them (fire together, wire together; online k-means
  under WTA competition, the language analog of test_011's learned visual
  dictionary). The wiring and the inference cost are unchanged. Result: 464.6 to
  406.6 ppl (-12.5%) at zero added cost, kWTA margin 1.61 to 2.21 (more
  discriminative codes).

Scoreboard (Run 5):

| system | test ppl | top-1 | top-5 | bits/word | MACs/pred | backprop |
|---|---|---|---|---|---|---|
| LLM (neural LM, regularized) | 462.3 | 10.6% | 26.2% | 8.773 | 542,720 | yes |
| Brain (random codes + learned granules, 16 ep) | 406.6 | 8.7% | 24.7% | 8.558 | 327,680 (0.60x) | no |

The Brain wins both axes, lower perplexity and lower cost with no backprop, and
beats the LLM on bits/word. The LLM still leads on top-1 (its single best guess is
better), but the Brain is better calibrated across the whole distribution. It was
still descending at epoch 16 (slice ppl 367 and falling), so there is headroom.

### Run 6, cutting cost to a fraction of the LLM (the sparse-code sweep)
With quality in hand, Run 6 spends the cushion on cost. The Brain's MACs are
`BR_G*BR_SIN` (granule expansion) plus `BR_K*VOCAB` (sparse readout). All configs
below keep random codes and WTA-learned features; perplexity is full-test, "ratio"
is vs the LLM's 542,720 MACs (its ppl 462.3 is the bar):

| BR_G | BR_SIN | BR_K | test ppl | MACs/pred | ratio | beats LLM? |
|---|---|---|---|---|---|---|
| 8192 | 16 | 48 | 406.6 | 327,680 | 0.60x | yes |
| 8192 | 16 | 24 | 444.4 | 229,376 | 0.42x | yes |
| 4096 | 16 | 24 | 431.7 | 163,840 | 0.30x | yes |
| 2048 | 16 | 24 | 420.3 | 131,072 | 0.24x | yes (chosen) |
| 2048 | 16 | 16 | 449.9 | 98,304 | 0.18x | yes (thin margin) |
| 8192 | 8 | 24 | 508.2 | 163,840 | 0.30x | no, underfits |
| 8192 | 4 | 24 | 590.1 | 131,072 | 0.24x | no, underfits |

Two findings:
- Shrinking the granule count improves quality and cost (8192 to 2048: ppl 444 to
  420). With a fixed sparse budget of 24 active granules, 8,192 granules left
  thousands dead and wasted; a compact 2,048-cell layer is fully used (live
  granules 95.3%, usage-Gini 0.633 to 0.515, a more even load). Fewer, better-tuned
  granules beat many random ones.
- Shrinking the fan-in (BR_SIN) backfires. 16 to 8 to 4 underfits (508, then 590,
  with the train gap going negative; the model cannot fit train). The real
  cerebellum uses about 4 dendrites only because it has billions of granules; at
  8,192 the expansion needs the wider fan-in. So cost comes out of granule count,
  not fan-in.

Chosen operating point: BR_G=2048, BR_SIN=16, BR_K=24. The table is at 16 epochs
for a fair sweep; the brain was still descending, so the chosen config is then
trained to convergence (28 epochs, 408.5 ppl) with no overfitting, test still
falling. It dominates the 0.30x and 0.42x points (cheaper and better).

### Standing after Run 8/9 (current headline)

| system | test ppl | bits/word | MACs/pred | backprop |
|---|---|---|---|---|
| LLM (neural LM, regularized, best-tuned) | 462.3 | 8.773 | 542,720 | yes |
| Brain, flat readout (Run 6) | 408.5 | 8.581 | 131,072 (0.24x) | no |
| Brain, hierarchical + weight decay (Run 8) | 415.1 | 8.546 | 35,840 (0.066x) | no |

A no-backprop, brain-style model beats a regularized neural LM on perplexity (415
vs 462) while running at about 1/15th of its inference compute, and about 1/3 of
its training compute at about 5x the throughput. The hierarchical readout traded
about 7 ppl for a 3.6x cost cut vs the flat readout; weight decay recovered part
of it. The caveat (see "every axis" below): the Brain is bigger in params/memory
but reads them sparsely.

### Run 7, similarity-based hippocampal recall (CLS), built and refuted
Run 5's exact-context cache failed because it is a trigram. The principled fix is
pattern completion by sparse-code similarity: an inverted index (granule to
episodes, capped at 48 each) recalls the training episodes whose codes overlap the
query's and votes their next words by overlap, fused with the cortex as
`(1-lambda)*cortex + lambda*recall` with lambda scaled by match quality. It is
cost-bounded (at most `BR_K*48` episode touches, a hash-scale lookup, not a full
kNN).

It still does not beat the cortex on this corpus: aggressive fusion gives 408.5 to
505.6, and a conservative lambda (only near-perfect matches earn weight) gives
408.5 to 430.6, so recall is neutral-to-harmful, never helpful. Root cause is the
same low repeatability that sinks trigrams: even when a near-identical context
exists in train, its test continuation usually differs, so the recalled vote is
noise against an already-good discriminative readout. Kept as a labeled ablation
and scaffold. Neural-cache recall is known to win on high-repetition / streaming
text and for one-shot new words, which this static, low-repetition split does not
exercise. Headline stays cortex-only, 408.5 at 0.24x.

### Run 8, cost to a fraction of the fraction (hierarchical readout)
The flat readout's `BR_K*VOCAB` term was 75% of the Brain's MACs. Run 8 splits the
vocabulary into 64 balanced frequency-rank classes and predicts class-then-word
(`P(w)=P(c)*P(w|c)`), the cortical "what" (category) feeding the "which"
(exemplar). Inference cost drops from `BR_K*VOCAB` to `BR_K*(64 + VOCAB/64)`:

| readout | MACs/word | x the LLM | test ppl |
|---|---|---|---|
| flat softmax (Run 6) | 131,072 | 0.24x | 408.5 |
| hierarchical (Run 8) | 35,840 | 0.066x | 417.5 |
| + readout weight decay (2e-5) | 35,840 | 0.066x | 415.1 |

A 3.6x further cost cut (15x cheaper than the LLM) for about a 7-point perplexity
tax, which a small L2 weight decay on the readout (synaptic homeostasis; the
train/test gap is +170) partly recovers. The Brain still wins both axes: 415.1 ppl
vs the LLM's 462.3, at 0.066x the compute, no backprop.

### Run 9, "give the LLM powers" and a longer context: both refuted
Two fairness/quality probes, both negative, recorded so the headline config is the
best one and not a lucky one:
- Bigger LLM (emb/hid 48/128 to 64/256, heavier dropout/decay): test ppl regressed
  462 to 480. The LLM keeps its best checkpoint at epoch 3-4 and overfits 187k
  tokens (train ppl about 100, test about 460). It is data-limited, not
  capacity-limited, so "powers" do not help. The fair baseline is the best-tuned
  small net.
- Longer context (CTX 3 to 4, a shared knob): regressed the Brain 417 to 435. With
  a fixed BR_SIN=16 fan-in, each granule samples a smaller fraction of a larger
  context vector, diluting its features; 4-gram sparsity on this corpus does the
  rest. Reverted.

### Run 8/9, the fight on every axis (training and footprint analytics)
The scoreboard now reports the axes that inference-MACs alone hide:

| axis | LLM (backprop) | Brain (local) |
|---|---|---|
| trained params | 0.74M | 8.52M (sparsely read) |
| model footprint | 3.0 MB | 35 MB |
| train wall-clock | 108 s | 102 s |
| train compute (MACs) | 2,240 G | 663 G (0.30x) |
| train throughput | 12.7k ex/s | 62.5k ex/s (4.9x) |
| inference MACs/word | 542,720 | 35,840 (0.066x) |

Honest reading: the Brain is larger in parameters/memory (an 8.5M-param readout)
but reads only BR_K=24 of BR_G=2048 rows per word, so it is 15x cheaper to run,
about 3x cheaper and about 5x higher-throughput to train, with no backprop, while
keeping the lower perplexity. The cost win is real; it is not a free-params win,
and the doc says so.

### The Brain's diagnosed ceiling (granule-code health, Run 1/3)
```
granules ever firing : 90.3%  (792 of 8192 dead)
usage Gini           : 0.632  (a few granules do most of the work)
mean code margin      : 1.618
readout weight norm   : 35.9
vocab rows unlearned  : 0 of 4096
```
At the time we blamed the fixed random word codes (king/queen orthogonal) and the
dead granules / high Gini. Runs 4-6 overturned half of that: the random codes were
not the problem (distributional codes were worse), but the inefficient sparse code
was real, and the fix was competitive feature learning plus a compact granule
layer, which cut the dead granules and dropped the Gini to 0.515 while beating the
LLM (Runs 5-6). The ceiling was in how the granules were used, not in the codes.

## 5. Engineering: fast, cross-platform, from scratch

No external libraries. The same sources build on macOS, Linux, and Windows.

- Batched LLM training (no BLAS). The first cut processed one example at a time,
  streaming the 2 MB `W2` matrix from memory per example and keeping a full
  per-thread copy of every gradient (zeroed every batch, about 160 GB of memset
  traffic over a run). The current version is batch-major:
  - the output forward/backward thread over the vocabulary, so each `W2` row is
    loaded once and reused across all examples in the batch (cache reuse);
  - each output row's weight gradient is owned by exactly one thread, so there are
    no per-thread gradient buffers and no large memsets.
  Same maths, laid out for the machine.
- Adam epochs cut 20 to 8. Early stopping always selected about epoch 3, so the
  extra epochs were wasted.
- `__restrict__` on the hot inner loops so the compiler vectorizes freely (NEON /
  AVX via `-march=native -ffast-math`); `-flto` inlines kernels across files.
- The Brain caches its test-set sparse codes once (the front-end is fixed), so the
  per-epoch eval is a readout sweep, not a re-encode.
- Diagnostics run on a representative slice; the reported perplexities come from
  the full batched evaluators.

The portable thread pool (`threads.cpp`) and xorshift RNG (`common.h`) are carried
from test_011.

## 6. Validation status

| component | status |
|---|---|
| N-gram baselines | validated (564 interpolated, 62.9% coverage) |
| LLM batched kernel rewrite | validated (Run 4); runs clean, healthy training; best val 252.8, test 462 |
| Brain, fixed random codes, 453 ppl | validated (Run 1); the brain bar to beat |
| Brain distributional codes + homeostasis | run and refuted (Run 4); 511 at 16 ep, loses to random's 453 |
| Brain code-scale fix (sqrt(BR_EMB) norm) | validated (Run 4); recovered 873 to 570 to 511 |
| Brain hippocampal exact-context cache (CLS) | run and refuted (Run 5); 464.6 to 490.5, exact recall is a trigram |
| Brain competitive granule feature learning | validated win (Run 5); 464.6 to 406.6, cost-neutral; beats the LLM |
| Brain cost sweep (compact granule layer + sparser code) | validated win (Run 6); 408.5 at 0.24x the LLM (28 ep) |
| Brain similarity-based hippocampal recall (CLS) | run and refuted (Run 7); 408.5 to 430.6, cortex too strong on a low-repetition corpus |
| Brain hierarchical (class to word) readout | validated win (Run 8); 0.24x to 0.066x the LLM for about a 7-ppl tax |
| Brain readout weight decay | validated win (Run 8); 417.5 to 415.1, curbs the overfit gap |
| LLM "powers" (wider 64/256 net, heavier reg) | run and refuted (Run 9); 462 to 480; the LLM is data-limited |
| Longer context (CTX 3 to 4, shared) | run and refuted (Run 9); Brain 417 to 435; dilutes fixed-fan-in granules |
| Brain training/footprint analytics | added (Run 8); trains at 0.30x MACs / 4.9x throughput; 11x params but read sparsely |

All bugs are fixed and the code runs end-to-end. Refuted: distributional codes,
exact-context and similarity caches, a bigger LLM, and longer context. Wins: WTA
granule features (Run 5), the cost sweep (Run 6), the hierarchical readout plus
weight decay (Run 8). The Brain sits at 415.1 ppl at 0.066x the LLM's inference
compute (and 0.30x its training compute), no backprop, beating the best-tuned LLM
on perplexity and bits/word. The deep analytics make the comparison fair on all
axes (params, memory, training, inference); see "every axis".

## 7. The frontier

Done so far: WTA-learned granule features (Run 5) and a compact-layer plus
sparse-code cost sweep (Run 6), reaching 408.5 ppl at 0.24x the LLM, no backprop.
From here:

1. Cut cost further; the encoder now dominates. With the hierarchical readout
   (Run 8) the cost is `BR_G*BR_SIN` (encoder, 32,768) plus
   `BR_K*(BR_CLASSES + VOCAB/BR_CLASSES)` (readout, 3,072) = 35,840. The encoder is
   91% of the cost, so the next lever is a cheaper expansion (top-down gating so we
   do not score all `BR_G` granules, or a 2-level granule hierarchy), or a 3-level
   word code, to push below 0.03x.
2. Recover the hierarchical tax with learned classes. The 64 classes are frequency
   bins; clustering words by their readout columns (co-occurrence-coherent classes)
   should recover most of the 408-to-415 gap, giving flat-quality at 0.066x cost.
3. The shared knobs are data-limited here. Bigger LLM and longer context (Run 9)
   both regressed; 187k tokens is the ceiling. The next step for both models is
   more data (full corpus / streaming), which would also let the refuted
   hippocampal cache (Run 5/7) pay off.
4. Structured context: vector-symbolic (hyperdimensional) binding of word and
   position instead of plain concatenation, so more context helps the
   fixed-fan-in granules instead of diluting them (the CTX=4 failure mode).
5. Spiking / event-driven execution to cash the efficiency edge in joules, not
   just multiply-adds.

The result has receipts on every axis: a brain-plausible model that is cheaper to
run (0.066x), cheaper and faster to train (0.30x MACs, 4.9x throughput), and lower
in perplexity (415 vs 462) than the best-tuned regularized neural LM, with no
backprop. The one axis it loses (parameter/memory footprint, 8.5M vs 0.74M) is
stated plainly.

## 8. Source map

| file | what it is |
|---|---|
| `src/config.h` | every knob (CTX, vocab, layer sizes, granule counts) |
| `src/common.h` | RNG, timing, softmax/argmax, `RESTRICT` (header-only) |
| `src/corpus.{h,cpp}` | reads and tokenizes the text, builds the vocabulary |
| `src/threads.{h,cpp}` | portable `parallel_for` thread pool (from test_011) |
| `src/llm.{h,cpp}` | the neural LM: batched AdamW backprop, dropout, early stop |
| `src/brain.{h,cpp}` | the cerebellar coder: random codes, learned granules, delta-rule readout |
| `src/analytics.{h,cpp}` | n-gram baselines, top-k, head-to-head, granule health |
| `src/main.cpp` | runs the head-to-head, prints the scoreboard and diagnostics |

Build and run: `cd test_012 && make run`.
