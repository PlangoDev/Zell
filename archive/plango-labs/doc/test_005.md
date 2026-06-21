# Test 005 — Real Handwriting, and How Far Sparsity Can Go

*The one where we stop using toys, put every trick together, and find the exact edge of the cliff.*

This is the big one. For four tests we built ideas one at a time on shapes we drew ourselves. Now we throw all of it at real data at once and see if it holds up. It mostly does, it breaks in a way that teaches us something, and it points straight at the number this whole project is chasing.

---

## Real handwriting this time

No more plus signs and X's we made up. This test uses **1,797 real handwritten digits**, zero through nine, each a tiny 8 by 8 picture written by an actual human hand. Some people write a sloppy 4. Some close the loop on a 9, some do not. The robots have to learn what a 4 *is*, across all the messy ways people draw it.

This is a real jump in difficulty. Ten answers instead of four. Real human variation instead of computer smudges. 64 pixels per image. If the brain tricks were only working because our toy shapes were easy, this is where they fall apart.

---

## The two contestants, fully loaded

**Densey** is built the way every AI you have heard of is built. A deep network, two hidden layers. Every neuron computes on every image. It learns by backpropagation, the textbook method with the biologically impossible backward wiring. This is the modern-AI stack, shrunk to fit on a laptop.

**Sparky** is everything the last four tests taught us, stacked into one robot:

- It is **deep**, two hidden layers, because test 004 proved depth is mandatory.
- Only a **few neurons fire** on each image. The rest stay completely silent. This is new this test and explained below.
- It learns by **random feedback wires**, not backprop, the trick from test 004, now reaching both hidden layers at once.
- It uses the **calm-down rule** so it settles instead of wobbling.
- It **skips blank pixels** for free, the oldest trick from test 001.

Two new ideas show up here, and both are worth meeting properly.

---

## New idea one: only the loudest neurons get to speak

Until now we used ReLU, which just silences neurons that come out negative. That gets you maybe half the neurons quiet. The brain is far stricter. At any instant only a tiny slice of neurons fire, because they actively shush each other. A neuron that fires sends "quiet down" signals to its neighbors. The loudest few win and the rest are forced silent. Brain scientists call this lateral inhibition.

We copied it with a rule called **k-Winner-Take-All**. After each layer computes, we keep only the *k* strongest neurons awake and shove every other one to zero. If we say "12% awake," then out of 128 neurons in a layer, only the 15 loudest get to pass their signal on. The other 113 are switched off for this image. A different image wakes a different 15.

Why this matters for energy: a silent neuron sends nothing to the next layer, so the next layer has almost nothing to read. Silence is free.

---

## New idea two: a single shout heard by every layer

Backprop passes the error backward one layer at a time, like a message whispered down a line of people. Test 004 showed that line needs impossible wiring.

This test uses the bolder cousin of the random-feedback trick, called **Direct Feedback Alignment**. Instead of whispering the error backward step by step, the output's mistake is **broadcast directly to every hidden layer at once**, each through its own set of fixed random wires. One global shout: "the answer was wrong, by this much, in this direction." Every layer hears it at the same time and adjusts.

This is much more like how a real brain might work. When you get a reward or a shock, your brain releases a chemical signal that washes over many regions at once, a single broadcast rather than a careful backward relay. That this even *could* train a network was the open question we left at the end of test 004. Now we find out.

---

## What happened

First the baseline. Densey, the full dense backprop network, read the real digits at **98.0%**. Strong. That is our bar.

Now Sparky, swept from "almost everyone awake" down to "almost nobody awake":

| Neurons awake | Accuracy | Work per image | vs Densey |
|---|---|---|---|
| 100% | 96.7% | 13,019 | 1.3x cheaper |
| 50% | 97.0% | 8,603 | 2.0x cheaper |
| 25% | 95.7% | 6,395 | 2.7x cheaper |
| **12%** | **90.2%** | **5,217** | **3.3x cheaper** |
| 6% | 78.1% | 4,665 | 3.6x cheaper |
| 3% | 11.6% | 4,389 | 3.9x cheaper |

Read that table slowly, top to bottom. It tells a story in three acts.

---

## Act one: the random-feedback trick survived growing up

Look at the very first row. With everyone awake, Sparky scored **96.7%**, barely a point under Densey's 98%.

That number is the answer to the cliffhanger from test 004. Back then, random-feedback learning worked, but only on a tiny two-switch toy with one hidden layer, and we openly worried it would crumble on something bigger. Here it is running across **two hidden layers, ten classes, and real human handwriting**, and it landed within about one point of real backpropagation.

The biologically impossible method is not pulling away. The brain-plausible one kept up, on real data. That is the most important single result in this whole series.

---

## Act two: turning off most of the brain was nearly free

Now walk down the rows. Half the neurons asleep: 97%, even a hair *better*, at twice the efficiency. A quarter awake: still 95.7%, almost no loss, at 2.7x cheaper. Even at just **12% awake**, with 113 of every 128 neurons switched off, it still read digits at **90%**, doing a third of the work.

This is the headline the whole project rests on. You can put the vast majority of the network to sleep on any given input and barely lose anything. The brain is not wasteful by accident. Sparsity is close to free, right up until it is not.

---

## Act three: the cliff

Then look at the bottom two rows, and watch it fall off a cliff.

At 6% awake, accuracy drops hard to 78%. At 3% awake, it **collapses to 11.6%**, which is basically pure guessing. Ten digits, one in ten odds, a coin with ten sides. The robot stopped being able to read at all.

Why so sudden? Look at what 3% actually means. The first hidden layer is down to **3 awake neurons**, and the second hidden layer is down to a **single** neuron. Ask yourself: can one neuron tell ten digits apart? One neuron can basically say "a little" or "a lot." That is not enough different messages to stand for ten different numbers. The information has nowhere to fit. The bottleneck strangles it.

This is a real and deep lesson. Sparsity is not free all the way down. You need at least enough awake neurons to physically carry the answer. Ten digits need more than one wire. The art is finding the floor and sitting just above it, which here was around 12 to 25% awake. Go below the floor and the lights do not dim, they switch off.

---

## The honest catch: why the win was "only" 3x

We chased a hundred-million-fold gap, and the best clean point here saved about 3x. What gives?

The answer is sitting in the data. These digit images are **51% ink**. Half of every picture is the actual number. Sparky's oldest trick is skipping blank pixels, but there is barely any blank to skip. The first layer has to look at roughly half the image no matter how sparse we make the hidden neurons, because you cannot recognize a digit while ignoring the digit. That first layer becomes the floor on savings.

This is the exact same lesson as test 003, where noise filled in the blanks and shrank Sparky's lead, now showing up in a new disguise. The efficiency of a brain-style system depends on how sparse its input actually is. A page that is half ink hands you a much smaller prize than a quiet world where almost nothing is happening at any moment, which is the world real brains evolved to sip energy in.

---

## So where does the 100-million actually come from?

Not from any one trick. From stacking sparse layers deep, where the savings multiply.

Here is the idea. In one sparse layer, if only a fraction *f* of neurons are awake, then the inputs to the next layer are sparse *and* the outputs are sparse, so that layer's work shrinks by roughly *f* twice over, which is *f squared*. Each interior layer you add multiplies the savings again.

| If this fraction is awake | One interior layer gets this much cheaper |
|---|---|
| 50% | 4x |
| 25% | 16x |
| 10% | 100x |
| 5% | 400x |
| 2% | 2,500x |

A real brain runs near 1 to 5% awake. Put that across many layers, then add the two things we have not even built yet, all-or-nothing spikes instead of smooth numbers, and cheap low-precision signals instead of full-fat math, and the gaps stack on top of each other. That is the road to the hundred-million. We measured one small paving stone of it here, honestly, and the projection shows where the rest of the road runs.

---

## What the five tests proved, end to end

- **001** A sparse brain matches a dense one at lower cost. The cost gap is real.
- **002** The emptier the input, the bigger the win.
- **003** Pile on competing answers and the cheap learner nearly ties at its best, but wobbles, and noise eats the savings.
- **004** A flat brain hits a hard wall. Depth breaks it. And you can train the depth with random feedback wires instead of the biologically impossible backprop, because the network rotates itself into alignment.
- **005** All of it together, on real handwriting, holds up. Random feedback survives depth and real data at a one-point cost. Turning off most of the brain is nearly free down to a floor, below which it falls off a cliff. The path to the huge efficiency gap is sparse layers stacked deep.

---

## Being honest about what this is not

- Our networks are small and our savings are real but modest. We have *measured* roughly 3x and *projected* the rest. Projection is not proof. The deep, extreme-sparsity regime is exactly where these methods are known to get harder, and we have not climbed there yet.
- We are still using smooth numbers, not real spikes. The single biggest remaining efficiency source, event-driven all-or-nothing spiking, is still ahead of us. That is the next mountain.
- Direct feedback alignment held up at two hidden layers. Nobody should assume it holds at twenty. Each new layer is a fresh test.
- The digit set is small and clean compared to the real world. Easy data flatters every method.

None of that dims the result. We took five separate brain ideas, sparsity, emptiness, competition, depth, and plausible learning, and showed they survive being bolted together and pointed at real handwriting. That is a foundation you can build the spiking, deeper, harder versions on top of.

---

## Next test ideas

- **Go spiking.** Replace the smooth neurons with ones that either fire or do not, and only compute when a spike actually arrives. This is the missing efficiency multiplier and the truest piece of the brain.
- **Go deeper.** Stack four or six hidden layers and find where direct feedback alignment finally strains, because that edge is where the real research lives.
- **Find genuinely sparse data.** Move to a task where the input really is mostly nothing at any moment, the way the world looks to a real sense, and watch the savings climb past anything we have seen.

---

*Run it yourself: `python3 test_005.py`*
