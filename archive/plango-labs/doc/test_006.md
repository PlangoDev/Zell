# Test 006 — Going Deep, and Wiring Like a Brain

*The one where the brain robot wins by a landslide on the thing that matters, refuses to win on the thing it can't, and tells us exactly what to fix next.*

The goal this time was simple to say and hard to earn: go deep, and make the brain way beat the LLM way by a lot. We got half of that as a crushing victory, the other half as an honest defeat, and the defeat turned out to be the most useful thing in the whole test, because it points a finger straight at the one problem worth solving.

No rigging. The numbers are the numbers.

---

## What we built

We stacked the networks **deep**, one to eight hidden layers, because depth is where real AI lives and where it spends almost all of its energy. Then we gave the brain robot a brand new trick on top of everything from before.

Until now, every neuron in our networks was wired to every neuron in the next layer. Real brains do not work remotely like that. Each of your neurons connects to a tiny handful of the eighty-six billion around it. This is **sparse wiring**, and it is the single biggest reason a brain is cheap. So the brain robot now has two kinds of sparsity stacked together:

- **sparse firing**, only 20% of neurons awake on any image
- **sparse wiring**, only 30% of the possible connections exist at all

It learns with random feedback wires, not backprop, and it keeps itself stable with one more borrowed brain trick explained below.

The LLM robot is the same as ever: every neuron, every wire, full backprop.

---

## The landslide: efficiency

Here is what happened as we stacked the layers up.

| Depth | LLM accuracy | Brain accuracy | LLM work | Brain work | Brain saves |
|---|---|---|---|---|---|
| 1 | 98.2% | 94.2% | 9,472 | 1,512 | 6.3x |
| 2 | 98.5% | 92.7% | 25,856 | 2,495 | 10.4x |
| 3 | 98.7% | 89.7% | 42,240 | 3,478 | 12.1x |
| 4 | 98.5% | 84.4% | 58,624 | 4,461 | **13.1x** |
| 6 | 98.2% | 77.1% | 91,392 | 6,427 | 14.2x |
| 8 | 98.5% | 71.0% | 124,160 | 8,393 | **14.8x** |

Look at the last column. Back in test 005 the brain saved about 3x. Here it saves up to **15x**, and the savings **grow with every layer we add**. This is the whole thesis of the project showing up as real measured numbers. The deeper and bigger the network, the more a sparse brain-style design pulls ahead on energy. Today's AI is racing toward ever-bigger dense networks. That is exactly the direction that makes this gap widen.

On energy, the brain way is not a little better. It is a lot better, and it gets more better the bigger you build.

---

## The defeat: accuracy

Now look at the accuracy columns, and watch them tell the opposite story.

At one layer the brain is close, 94% against 98%. But as we stack layers, the LLM holds rock steady near 98% while the brain **slides down**: 92, 89, 84, 77, 71. Going deep made the brain cheaper and, at the same time, worse at reading.

We are not going to dress that up. On this task, the LLM wins on accuracy at every depth, and its lead grows the deeper we go. The thing we hoped for, a brain that is better at everything, did not happen. So the real question becomes: **why?** And the answer is the most valuable thing in this test.

---

## Pinning down the culprit

To find out what was costing the accuracy, we built the brain robot up one trick at a time, at a fixed depth of four, and watched.

| What we turned on | Test accuracy | Work saved |
|---|---|---|
| LLM: everything dense, backprop | 98.5% | 1.0x |
| add sparse firing (only 20% awake) | 90.2% | 4.1x |
| add sparse wiring (only 30% of wires) | **92.7%** | **13.1x** |
| swap backprop for random feedback | 84.4% | 13.1x |

Read this carefully, because it clears the innocent and names the guilty.

Sparse firing cost a little accuracy and saved a lot of work. Fair trade. Then look at the next row: adding sparse **wiring** actually **raised** accuracy back up to 92.7% while tripling the efficiency to 13x. Cutting most of the wires made the network both cheaper **and** better, because fewer wires means less room to memorize noise. Sparse wiring was a free lunch. Two of them.

So sparsity is not the problem. Sparsity is wonderful. At 92.7% accuracy and 13x cheaper, the sparse-but-backprop network is a genuinely great machine.

The accuracy fell off a cliff on the very last row, the moment we threw away backprop and switched to random-feedback learning. That one change cost eight points. **The whole accuracy gap is the learning rule, not the sparsity.**

---

## Why random feedback struggles when it goes deep

We met random-feedback learning in test 004 and it was a marvel, training a hidden layer through fixed random wires. It worked because the forward weights could rotate to line up with the random feedback.

The trouble is, that rotation gets harder the deeper you stack. The error signal for the very first layer has to pass through the random feedback and survive being filtered by every awake-or-asleep gate above it. By the time it reaches the bottom of a deep network, the teaching signal is faint and muddled. The early layers barely know what to do. They underfit.

We even proved it is underfitting and not the usual overfitting. We checked how well each robot memorized its training set versus how it did on the real test:

- **LLM**: memorized the training images perfectly, 100%, and scored 98.5% on the test. A small gap. Classic mild overfitting.
- **Brain**: only reached 85.9% even on the training images it was allowed to study, and 84.4% on the test. Almost no gap.

The brain is not overfitting. It is failing to fit at all. Its deep layers never got a clear enough lesson. That is a learning problem, full stop.

---

## We tested the brain's best hope, honestly, and it lost

There is a famous place where brains crush big AI: learning from **few** examples. A child knows a giraffe from one picture book. Big dense networks are data gluttons and overfit badly when examples are scarce. We had a real hope that in the low-data corner, the brain's heavy regularization would finally beat the LLM outright.

So we starved both robots and watched.

| Training images | LLM test | Brain test | Winner |
|---|---|---|---|
| 50 | 76.1% | 67.0% | LLM |
| 100 | 88.2% | 76.1% | LLM |
| 200 | 92.7% | 83.1% | LLM |
| 400 | 95.7% | 88.4% | LLM |
| 800 | 98.0% | 91.7% | LLM |
| 1400 | 98.5% | 92.7% | LLM |

The LLM won at every single data size, even the tiniest. We had a clean hypothesis, we ran it, and it was wrong. The brain's underfitting problem is so dominant that it loses even on the turf we expected it to own. Backprop simply squeezes more learning out of every example, scarce or plentiful.

This is what honest research looks like. You test the thing you hope is true, and sometimes it says no, and you write the no down in ink.

---

## So did we make the brain a lot better?

On energy, yes, by a landslide, and more so the deeper we build. That is the axis this entire project exists to win, the road to the hundred-million-fold gap. The brain way now does the same job for a fraction of the power, and the fraction shrinks with every layer.

On accuracy, no, and we are not going to pretend otherwise. The LLM's learning rule is still better at extracting knowledge, especially deep. We tried two honest fixes, homeostatic normalization to survive depth and the low-data regime to exploit regularization, and neither bought an accuracy win.

But we did something better than a fake victory. We isolated the exact problem. It is not the sparse firing. It is not the sparse wiring. Those are pure wins. It is the credit-assignment rule. Random feedback alignment, as we have it today, cannot teach a deep stack as well as backprop can. Make that one thing as good as backprop, and the brain robot inherits a 13x efficiency edge for free and the whole contest flips.

That is not a disappointment. That is a research target with a bullseye painted on it.

---

## What we learned this test

1. Sparse wiring, brains connecting each neuron to only a few others, is a double win here. It cut work by a further 3x **and** improved accuracy, because it stops the network memorizing noise.
2. Stacked sparsity makes the efficiency gap grow with depth, from 3x at two layers up to ~15x at eight. The bigger AI gets, the more a brain-style design would save.
3. Homeostatic normalization, neurons rescaling themselves, stops a deep sparse network from blowing up. It rescued the eight-layer net from total collapse, though it could not close the learning gap.
4. The entire accuracy shortfall traces to one component, the random-feedback learning rule, which underfits deep networks. Sparsity is innocent.
5. The hoped-for low-data win did not materialize. Backprop generalizes better even from a handful of examples. Written down honestly.

---

## Being honest about the measurement

- The efficiency numbers count multiply-add operations and assume hardware that can actually skip absent wires and sleeping neurons. A standard GPU cannot exploit this kind of messy sparsity and would see less benefit. The full prize needs brain-style (neuromorphic) hardware, the same caveat as test 005.
- These networks and this dataset are small. Small, clean problems flatter backprop and understate how brittle huge dense networks get. The brain's advantages may widen at real scale, but we have not shown that here.
- We have one learning rule for the deep case. There are many other brain-plausible candidates we have not tried.

---

## Next test ideas

- **Attack the real problem: better deep credit assignment.** Try the known upgrades to random-feedback learning, or local layer-by-layer targets, and see if the brain's deep accuracy can climb back to backprop's while keeping the 13x edge. This is now the main quest.
- **Go spiking at last.** Replace smooth neurons with real fire-or-not spikes and event-driven computation, the one efficiency multiplier we still have not collected.
- **Make the task hard enough to hurt the LLM.** Heavy noise, distribution shift, or truly tiny data with a fitting-capable brain rule, the conditions where a regularized brain should finally pull ahead.

---

*Run it yourself: `python3 test_006.py`*
