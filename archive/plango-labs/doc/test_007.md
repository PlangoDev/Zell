# Test 007 — Giving the Brain a Memory

*The one where the brain learns a brand-new thing from a single glance, which the LLM simply cannot do, and where the sparsity we built for cheapness turns out to be the secret ingredient for remembering.*

This is the most advanced test in the series, and it changes the strategy. For six tests we tried to win on accuracy by making the brain *learn* better. It kept underfitting, because random-feedback learning cannot teach a deep stack as sharply as backprop. So this time we stop fighting that, and we steal a completely different idea from real brains: do not just learn. **Remember.**

---

## Two memories are better than one

Here is a fact about your own head that took neuroscientists decades to pin down. You do not have one learning system. You have two, and they do opposite jobs on purpose.

- Your **neocortex** is the slow one. It sees thousands of examples and grinds out general rules over months and years. It is careful and it generalizes. It is the deep net we have been building this whole time.
- Your **hippocampus** is the fast one. It snapshots a single experience and stores it instantly. See a friend's new haircut once, and you know it forever. No thousands of repetitions required.

This is called **Complementary Learning Systems** theory, written down in 1995 by McClelland and friends. The slow cortex learns what is generally true. The fast hippocampus remembers what specifically happened. You think by combining them: "in general, faces look like this, and specifically, I have seen this exact face before."

Our brain robot has only ever had the slow cortex. This test bolts on a hippocampus.

---

## How the hippocampus stores a memory

You cannot just dump raw experiences into a memory and hope to find them later. They blur together. The digit 8 and the digit 9 look fairly similar as raw pixels, so their memories would overlap and confuse each other.

Real brains solve this with a beautiful trick in a region called the **dentate gyrus**. Before storing anything, it takes the input and blows it up into a much larger, much sparser code. Sixty-four pixels become hundreds of neurons, of which only the strongest 5% are allowed to fire. This is called **pattern separation**, and the point is to push memories apart so they stop stepping on each other.

We built exactly that. Each image gets expanded to 800 neurons, then all but the top 5% are silenced, then it is stored with its label. To recognize something new, the brain finds the stored memories whose sparse codes overlap the most and lets them vote.

We measured how well it separates. We took 500 memories and checked how much an average pair overlaps. Lower means more distinct, less confusable.

| How memories are stored | Average overlap |
|---|---|
| Raw pixels | 0.690 (smear together) |
| Dentate sparse codes | **0.451** (stay distinct) |

The sparse codes are far more separated. And here is the payoff that ties this whole project together: **the sparse firing we invented back in test 005 to save energy is the very same thing that makes memory work.** Sparseness is not two tricks. It is one trick that buys efficiency and memory at the same time. That is almost certainly not a coincidence in real brains either.

*(One thing we learned the hard way: our first attempt built the memory out of the cortex's own features, and it barely helped, because a weak cortex makes weak memories. The hippocampus needs its own clean input pathway, which is exactly why the real dentate gyrus has one. The biology was right.)*

---

## The headline: learning a new thing from one glance

Now the result that a normal trained network cannot match no matter how hard it tries.

We never let anyone train on the digits 8 or 9. The cortex has never seen them. Then, at decision time, we show the brain a single 8 and a single 9 and ask it to recognize brand-new 8s and 9s it has never met.

A standard trained net is helpless here. Its output slots are for the digits it was trained on. It has no slot for an 8, and you cannot conjure one from a single example without retraining the whole thing on lots of 8s. Faced with an 8, it can only shrug and guess, which on a two-way choice means 50%.

The hippocampus just does it. Watch.

| Examples shown of each new digit | Brain recognizes new ones |
|---|---|
| **1 each (a single glance)** | **78.3%** |
| 2 each | 80.9% |
| 3 each | 86.5% |
| 5 each | 90.8% |
| 10 each | 94.0% |

From **one** example of each, with zero training and zero gradient updates, the brain jumps to 78%, far above the 50% coin flip. Show it ten and it is at 94%. The standard net is stuck at the starting line the whole time, because this is not a thing its design can do.

This is what your hippocampus does every single day. It is why a child can be told "that animal is a quokka" once and recognize quokkas for life. Big trained networks famously cannot do this, which is why a whole research field exists just to teach them clumsy versions of it. Our brain robot got it almost for free, the moment it had a memory.

This is the brain genuinely beating the LLM, not on speed, but on a kind of intelligence the LLM does not possess.

---

## Memory also patches the accuracy hole

Back on the normal job, recognizing all ten digits, memory helped there too. Remember, the brain's cortex underfits and trails the LLM. So we let it lean on its hippocampus.

| System | Accuracy |
|---|---|
| LLM (dense, backprop) | 99.0% |
| Brain cortex alone (the slow learner) | 91.7% |
| Brain hippocampus alone (pure recall) | 97.0% |
| **Brain cortex + hippocampus together** | **97.5%** |

The cortex alone was stuck at 91.7%. Add the memory and the brain jumps to 97.5%, clawing back most of the gap to the LLM's 99%. Neither half is as good as the two together. The slow learner and the fast rememberer cover for each other, which is the entire point of having both.

We did not beat the LLM on this clean, easy task. Backprop on plain digits is close to perfect and very hard to top. But we turned a 7-point deficit into a 1.5-point one by remembering instead of only learning, and that is real progress on the exact weakness test 006 exposed.

---

## The low-data world, revisited honestly

In test 006 the LLM beat the brain at every dataset size, even tiny ones, which stung because few-shot learning is supposed to be the brain's home turf. With a memory now, did we flip it?

| Training images | LLM | Brain cortex | Brain + memory | Winner |
|---|---|---|---|---|
| 50 | 80.6% | 60.5% | 69.8% | LLM |
| 100 | 90.7% | 72.3% | 86.6% | LLM |
| 200 | 92.9% | 81.1% | 91.2% | LLM |
| 400 | 96.7% | 87.2% | 94.7% | LLM |
| 800 | 98.0% | 90.7% | 96.5% | LLM |
| 1400 | 98.2% | 93.5% | 97.7% | LLM |

Memory helped a lot. Look how far the brain climbed once it could remember, from 81% up to 91% at two hundred images, from 93% up to 98% at the full set. But the LLM still edged it out everywhere. We are not going to pretend otherwise. On this clean, friendly dataset, backprop's grip is strong even when examples are scarce.

The honest read: memory closed most of the gap but not all of it, and the clean one-shot victory above is where memory's true edge shows, on genuinely new things the trained net was never built to handle.

---

## What we learned this test

1. There are two ways to get smarter, learning and remembering, and brains use both. When the slow learner underfits, a fast memory can carry the accuracy it dropped.
2. A memory lets the brain learn a brand-new category from a single example, instantly, with no retraining. A standard trained network cannot do this at all. This is a real capability gap, in the brain's favor.
3. Memory needs sparse, separated codes or experiences blur together. The dentate-gyrus trick, expand then sparsify, cut memory overlap from 0.69 to 0.45.
4. The sparse firing we built for energy savings is the same mechanism that makes memory reliable. Efficiency and memory are one idea.
5. The cortex and hippocampus together beat either one alone, 91.7% and 97.0% separately becoming 97.5% combined.

---

## Being honest about what this costs and is not

- **Memory is not free to search.** Recalling means comparing the query against stored memories. With thousands of memories that is a real pile of work, and a naive scan would eat the efficiency win from the earlier tests. Brains dodge this with content-addressable recall that jumps to the relevant memories instead of scanning all of them, and building that cheap version is its own job we have not done yet. For the one-shot and low-data cases the memory is tiny and the cost is trivial, which is the regime where it shines.
- **We still did not beat the LLM on the clean full task.** 97.5% against 99.0%. Memory narrowed the gap, it did not erase it. Backprop on easy data remains the bar.
- **The cortex still underfits.** Memory is a patch over that wound, not a cure. The deep credit-assignment problem from test 006 is still the real disease, and still unsolved.
- The one-shot numbers wobble (the ±12 on a single example), because one example is genuinely a thin thing to learn from. The averages over many tries are solid, but expect variance from any single glance.

---

## Next test ideas

- **Make recall cheap.** Build the brain's trick of jumping straight to relevant memories instead of scanning them all, so memory keeps the efficiency win instead of spending it.
- **Continual learning.** Teach the brain digits one at a time and show it does not forget the old ones, the catastrophic-forgetting problem that wrecks standard nets and that a hippocampus is built to prevent. Another place the brain should win outright.
- **Still go after the cure.** Better deep credit assignment remains the main quest. Memory bought us accuracy without fixing the learner. Fixing the learner would compound with everything here.
- **Sleep and replay.** Real brains replay hippocampal memories back to the cortex during sleep to slowly teach the slow system. Simulating that consolidation could let the fast memory actually train the cortex over time, the missing link between the two systems.

---

*Run it yourself: `python3 test_007.py`*
