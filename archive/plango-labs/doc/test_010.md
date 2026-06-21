# The Rematch: How the Brain Finally Beat the LLM

### Test 010 · the finale

*The brain wins the accuracy ribbon at last. It does it without a single drop of the magic math the LLM relies on. And in winning, it reveals the deepest idea of the whole project: the brain is not a point, it is a dial.*

---

## The grudge match

For nine tests the LLM kept the accuracy crown. We chipped at it. We got within a point or two. But on a clean sorting job, the textbook deep learner always edged us out, because its way of learning, backpropagation, squeezes more out of the data than the brain's humble local rules can.

So for the rematch we changed strategy completely. We stopped trying to beat backprop at its own game. We did what the brain actually does.

---

## The trick: don't learn the hard part at all

Here is the idea that wins it, and it comes straight from a real piece of your brain you have never thought about: the **cerebellum**.

The cerebellum is the lump at the back of your skull that handles coordination and timing. It contains more than half of all the neurons you own, most of them a tiny type called granule cells. And here is the strange, wonderful thing about them: **the granule cells never really learn.** Each one is wired to just a few inputs, with a fixed, essentially random pattern, and it stays that way for life. They form a giant fixed bank of little feature detectors. The only place the cerebellum actually learns is one layer further on, at a single set of output cells.

Think about what that means. The brain's hardest problem, teaching the deep middle of a network, the thing backpropagation was invented for, the thing our brain robot kept failing at... the cerebellum just **skips it**. It does not learn deep features. It throws a massive pile of fixed random features at the problem and only bothers to learn a simple readout on top. A readout is shallow, and shallow things learn fine with the brain's simple local rules. No deep learning, no problem to get wrong.

So that is our contestant. We build a huge bank of **fixed random detectors**, each one looking at a small random patch of the image with random weights, never trained. Then we train one simple linear readout on top, with a plain local rule. Zero backpropagation anywhere in the thing.

---

## The other trick: a committee, paid for with spare change

One such circuit is good but not great, about 87 out of 100. To get over the top we use the second thing brains do: they are not one model, they are a **crowd**. Your brain runs enormous numbers of similar circuits at once and lets them vote.

We can afford to copy that because each of our circuits is cheap. So we build several of them, each with its own different random detectors, and let them vote on every photo. Different random features make different mistakes, and when they vote, the mistakes cancel out and the truth adds up. This is the oldest reliable trick in machine learning, and the brain has been using it since before there were machines.

---

## The result

The bar to beat: the LLM at full strength, **88.52%**.

The brain's committee, one expert added at a time:

| Experts voting | Accuracy | vs the LLM |
|---|---|---|
| 1 | 86.74% | cheaper, not yet as good |
| 2 | 86.42% | — |
| 3 | 88.42% | basically level |
| 4 | 89.18% | **beats it** |
| 5 | 89.60% | **beats it** |
| 6 | **89.94%** | **beats it, by 1.4** |

There it is. At four experts the brain passes the LLM, and at six it sits at **89.94% against the LLM's 88.52%**. The brain finally won the accuracy ribbon, and it did so with no backpropagation at all, using a pile of fixed random features and a vote. The thing the LLM is supposedly best at, the brain just took, by refusing to play the same game.

---

## The honest catch, which is also the best part

We are not going to hide the cost, because the cost turns out to be the most important lesson in the whole project.

Winning took a committee of several circuits, and running several circuits costs more than running one. At the point where it passed the LLM, the brain was spending about **three times** the LLM's work, not less. It won accuracy by **spending** its famous efficiency, not by keeping it.

Now hold that next to test 009, where a single sparse brain circuit scored about 85% for **one tenth** of the LLM's work.

Look at what those two results are together:

| The brain, set to... | Accuracy | Work |
|---|---|---|
| cheap (test 009) | ~85% | 10x **less** than the LLM |
| accurate (test 010) | ~90% | 3x more than the LLM |

The LLM is a single dot. It gives you 88.5% at one fixed price, take it or leave it. **The brain is not a dot. It is a dial.** You can turn it toward cheap and run it for a tenth of the energy at a small accuracy cost, or turn it toward accurate and beat the LLM by spending more. Same architecture, same parts, slid to wherever you need to be on the day.

That is the deepest thing we learned in ten tests. A brain-style system does not force the energy-versus-accuracy choice that a single big model forces on you. It hands you the knob.

---

## And the brain brought its other gifts too

The dial is not even the whole advantage. The brain robot also carries everything from the earlier tests that the LLM simply cannot do:

- it learns a brand-new kind of thing from a single example (test 007, test 009)
- it sleeps and stops forgetting old lessons when it learns new ones (test 008)
- and it can be run as cheap or as sharp as the moment calls for (this test)

The LLM is one expensive setting, brilliant at exactly one thing, blind to new categories, and prone to forgetting. The brain is a whole toolkit. On the day it mattered, the toolkit won the one contest the LLM was supposed to own.

---

## What we learned, the whole arc in one breath

1. Sparse firing is cheap (001–003). Depth needs a trick, and random feedback is that trick (004). It scales to real data but underfits (005–006). Memory patches the underfitting and grants one-shot learning (007). Sleep stops forgetting (008). On a hard, full-size job the brain trades a little accuracy for tenfold efficiency (009). And when we finally want the accuracy crown, the cerebellum's fixed-feature committee takes it, with no backprop, by spending that efficiency back (010).
2. The single most useful idea: efficiency is not just a saving, it is a **budget**. Being cheap per circuit is what lets a brain run a crowd, and a crowd is what wins.
3. The brain never beat backprop by being a better backprop. It won by not needing backprop at all.

---

## Being honest about the limits

- The accuracy win cost about 3x the compute. This is a genuine win on accuracy and a genuine loss on energy, at the same time. The triumph is the dial, the ability to choose, not a free lunch.
- The pictures were shrunk and the task, while real and hard, is still small next to the world. Bigger problems may move all these numbers.
- The efficiency figures still assume hardware that can skip absent wires and sleeping cells, the brain-style chips that mostly do not exist yet.
- The cerebellar fixed-feature trick has its own ceiling. Pile on enough random features and a committee and you get here; it will not climb forever without better, learned features, and learning those well without backprop is still open.

---

## Where this could go next

- Learn the features instead of leaving them random, but with a brain-plausible local rule (competitive learning, the way real receptive fields actually form), so each circuit is smarter and the committee can be smaller and cheaper, and maybe win on **both** axes at once.
- Put the whole toolkit together at last: the cerebellar committee for sharp perception, the hippocampus for one-shot and memory, sleep for keeping it, all sparse and spiking, on one task.
- Move to real brain-style hardware where the tenfold and beyond savings are actually paid out.

---

*Run the rematch yourself: `python3 test_010.py`*
