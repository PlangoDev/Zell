/* ─────────────────────────────────────────────────────────────────────────────
 * config.h — every knob for Test 12 in one place (the way test_011 kept its
 * shape constants together). Change a number here, rebuild, and both contestants
 * pick it up.
 *
 * TEST 12 — the jump from pictures to WORDS. Same contest as before:
 *   LLM way   : a neural language model trained by BACKPROPAGATION.
 *   Brain way : a cerebellar n-gram coder — fixed sparse expansion + a local
 *               readout rule. NO backprop anywhere.
 * Task: given the previous CTX words, predict the next one.
 * Scored on PERPLEXITY (the language metric), next-word top-1 accuracy,
 * multiply-adds per prediction (the energy proxy), and wall-clock time.
 * ───────────────────────────────────────────────────────────────────────────*/
#ifndef CONFIG_H
#define CONFIG_H

/* ── shared task shape ────────────────────────────────────────────────────── */
#define CTX     3        /* context length: predict word N from the 3 before it.
                          * (CTX=4 was tried and REGRESSED the brain: with fixed
                          * BR_SIN fan-in, each granule samples a smaller fraction of
                          * a larger context, diluting its features. See doc §4.)   */
#define VOCAB   4096     /* keep the top (VOCAB-1) words; everything else = <unk> */

/* ── the LLM contestant (Bengio-style neural language model) ──────────────── */
/* This is the STRONGEST LLM config found. Giving it "powers" was tried and failed:
 * a wider 64/256 net (2x the params/compute) plus heavier dropout/decay REGRESSED
 * test perplexity (462 -> 480) — it overfits on 187k tokens, so it is data-limited,
 * not capacity-limited. The fair, honest baseline is therefore the best-tuned small
 * net, not a bigger-but-worse one. See doc §4. */
#define LM_EMB  48       /* learned embedding dimension per word                 */
#define LM_IN   (CTX * LM_EMB)   /* concatenated context vector = 144            */
#define LM_HID  128      /* one hidden (tanh) layer                              */

/* ── the Brain contestant (cerebellar granule coder) ──────────────────────── */
#define BR_EMB  48                 /* FIXED random word code dimension           */
#define BR_IN   (CTX * BR_EMB)     /* concatenated context vector = 144          */
#define BR_G    2048               /* granule cells (the big sparse expansion)   */
#define BR_SIN  16                 /* mossy fibers per granule. Fan-in is quality-
                                    * critical here (4 and 8 both underfit at this
                                    * granule count), so the expansion is shrunk via
                                    * BR_G instead, which keeps each granule expressive. */
#define BR_K    24                 /* active granules after kWTA (sparse code)    */

/* Two-level (hierarchical) output. The flat readout costs BR_K*VOCAB MACs/word and
 * dominates the brain's compute. Splitting the vocabulary into BR_CLASSES balanced
 * word-classes and predicting class-then-word turns that into BR_K*(BR_CLASSES +
 * VOCAB/BR_CLASSES) — minimized near sqrt(VOCAB). This is the cortical "what stream"
 * (category) feeding the "which" (exemplar), and it cuts the readout cost ~30x. */
#define BR_CLASSES 64

/* Readout weight decay (L2). The delta-rule readout memorizes the train stream
 * (train ppl << test ppl); a small decay pulls unused weight back toward zero each
 * update — synaptic homeostasis — and lowers TEST perplexity by curbing that
 * overfitting. Applied to the active granule rows (use-dependent forgetting). */
#define BR_WD 2e-5f

/* Word-code substrate. Run 4 (doc/test_012.md §4) showed FIXED RANDOM codes
 * (orthogonal, like a dentate-gyrus random projection) BEAT distributional codes
 * on this memorization-heavy task: next-word prediction rewards telling near-
 * identical contexts apart, and semantic sharing blurs exactly that. So random is
 * the default substrate; set to 0 to A/B the distributional (PPMI) codes. The
 * "meaning"/sharing now comes from the hippocampal cache (CLS), which ADDS recall
 * without REMOVING discriminability. */
#define BR_RANDOM_CODES 1

/* Granule feature learning (competitive / Hebbian). Instead of leaving the granule
 * weights random, let the kWTA winners nudge their weights toward the contexts that
 * fire them ("fire together, wire together") — online k-means with WTA competition,
 * the way cortical/cerebellar maps self-organize. The wiring (which inputs each
 * granule samples) and the inference cost are UNCHANGED; only the weights tune, so
 * the expansion tiles the real data manifold instead of random noise. Set to 0 for
 * the pure-random-wiring baseline. */
#define BR_LEARN_GRANULES 1
#define BR_GFEAT_PASSES   3        /* sweeps of competitive learning over the sample */
#define BR_GFEAT_SAMPLE   8000     /* contexts sampled per sweep                      */
#define BR_GFEAT_ETA      0.05f    /* learning rate (annealed across sweeps)          */

#endif /* CONFIG_H */
