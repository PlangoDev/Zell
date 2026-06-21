# The Great Laundry-Sorting Championship

### Team LLM versus Team Brain · Test 009

*Two robots, one giant pile of laundry, three contests, and a totally fair fight. Each robot is allowed to play its own way. Let's see who wins what.*

---

## The pile

Somebody dumped out **sixty thousand** little photos of clothing and we have to sort them. T-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and boots. Ten kinds in all. (This is the whole pile now, the full dataset, not a sample.)

This is a much harder job than the number-reading from before, and there is a LOT more of it. The photos are fuzzy and small, and a coat can look an awful lot like a pullover if you squint. Here is what the robots actually see (real photos from the pile, drawn in dots and dashes):

```
   a t-shirt          a sneaker            a bag
     --=-                                    .:
   -+=**=+-                                +--+.
  :========-              -**:  =.        =.   +
  -=====-===           -+***#####-        +    -:
  :+-----==:        :+**+########=      . =    :=
    -----=          *######%%%%%#=     #%%%%%%%%%%%-
    -----=          .--=======+++-     %##%##%###%%=
    -----=                              @#%%####%%%%*
```

You can sort of see them, right? A t-shirt with little sleeves, a sneaker lying on its side, a bag with a handle. The robots had to learn all ten kinds from the pile, then sort ten thousand brand-new photos they had never seen.

---

## The two teams

**Team LLM** plays the way every famous AI plays. One big brain, every brain cell switched on for every photo, trained the standard textbook way (backpropagation). It is powerful and it is hungry. Think of a huge, brilliant, always-on machine.

**Team Brain** plays the way this whole project has been building, with every trick we collected:
- a deep brain that only lights up a few cells per photo (sparse firing)
- wired loosely, each cell talking to just a few others (sparse wiring)
- taught by random feedback instead of the impossible textbook math
- kept steady so it does not wobble
- and a **hippocampus**, a little notebook of past photos it can flip through

Same pile. Same test. Each team its own way. Three contests. No cheating.

---

## Contest 1: Who sorts the most correctly?

The straight-up sorting test. Of three thousand new photos, how many does each get right?

| Team | Got it right |
|---|---|
| **Team LLM** | **88.3%** |
| Team Brain (brain only) | 81.7% |
| Team Brain (brain + notebook) | 84.7% |

**Team LLM wins this one.** 88 out of 100, fair and clear. The textbook learner is just really good at squeezing knowledge out of a big pile of examples, and with the whole pile to chew on, it got even better.

But look what the notebook did for Team Brain. On its own the brain got about 82. Let it flip through its hippocampus notebook while deciding, and it climbed to almost 85, chasing the champion. The gap shrank from about seven points to under four. Close contest, but the LLM took the ribbon, and with all this data it actually stretched its lead a touch.

**Winner: Team LLM, by a little.**

---

## Contest 2: Who does it with the least work?

Now the same job, but we count the effort. Every time a robot multiplies two numbers together, that is one drop of sweat. How many drops to sort one photo?

| Team | Drops of sweat per photo |
|---|---|
| Team LLM | 118,272 |
| **Team Brain** | **11,957** |

**Team Brain wins this one, and it is not close.** The brain sorted the laundry using about **ten times less work** than the LLM. The LLM had every one of its brain cells fire for every single photo. The brain woke up only the few cells it needed and let the rest nap.

Same laundry. Almost the same score. One tenth of the effort. If you were paying the electricity bill, you would know exactly which robot you want.

**Winner: Team Brain, by a mile.**

---

## Contest 3: Learn a brand-new thing on the spot

Here is the contest the LLM cannot even enter.

We secretly never let either robot train on **sandals** or **bags**. Then, in the middle of the championship, we show each robot a few sandals and a few bags and say, quick, learn these, here come new ones.

Team LLM is stuck. Its sorting slots were built for the clothes it trained on. It has no slot for "sandal" and no way to add one from a couple of photos. It can only shrug and guess, which on a two-way choice is 50%.

Team Brain just flips open its notebook, jots down the new examples, and starts recognizing them.

| Sandal/bag examples shown | Team Brain gets right |
|---|---|
| 1 of each (one single look) | 68.3% |
| 3 of each | 67.7% |
| 5 of each | 78.0% |
| 10 of each | 85.4% |

From a **single** look at each, the brain already beat the coin flip. Show it ten and it sorted the brand-new clothes as well as it sorted the ones it studied for ages. The LLM never got off the bench.

**Winner: Team Brain, uncontested.**

---

## The final scoreboard

| Contest | Winner |
|---|---|
| Sorting the most correctly | Team LLM |
| Doing it with the least work | Team Brain |
| Learning new things on the spot | Team Brain |

So who won the championship? Neither, and that is the honest and interesting answer. They are good at different things.

**Team LLM** is the better pure sorter. If all you care about is getting the most photos right and you have a giant power plant to run it, it wins. We are not going to pretend our brain robot beat it at straight accuracy. It did not.

**Team Brain** sorts almost as well for one tenth of the work, and it can learn brand-new things from a single glance, which the LLM simply cannot do. For a robot that has to run on a tiny budget of energy, in a changing world where new things keep showing up, that is the package that matters.

That is the whole point of this project in one laundry pile. We are not trying to beat the giant machine at being a giant machine. We are building the thing that does nearly as well for a sliver of the energy, and can keep learning forever. On a real, hard job with lots of data, that thing held up.

---

## What we learned

1. On a genuinely harder task with much more data, the brain way held its ground. The straight learner won accuracy by a couple of points, not a landslide.
2. The brain's memory notebook closed most of that gap, lifting it from 82 to 85 out of 100, chasing the champion.
3. The brain did it all for about a tenth of the work. The harder and bigger the job, the more that efficiency matters.
4. The brain can learn a brand-new kind of thing from one example. A standard trained network has no way to do this at all. Different kind of smart.

---

## Being honest about it

- The LLM won the accuracy contest. 88 to 85. A real loss for the brain, fairly counted.
- The brain's efficiency win assumes hardware that can actually skip sleeping cells and missing wires. A plain graphics card cannot, and would not see the full tenfold saving. Brain-style chips are needed to cash it in.
- We ran the full sixty thousand training photos this time, and it helped the careful LLM most, exactly as the smaller run predicted it would. More food feeds the data glutton best. We did still shrink each photo to run fast, so sharper pictures could shift things again.
- The brain's slow learner still underfits. Memory is carrying a lot of its accuracy. The deep-learning weakness from the earlier tests has not gone away, it has just been papered over by a good notebook.

---

## Next contest ideas

- Give Team Brain a true spiking version, where cells stay silent and cost nothing until a signal arrives, and see if the tenfold efficiency stretches further.
- Make the world keep changing mid-game, throwing in new clothing types one after another, the contest where memory and sleep should let the brain pull genuinely ahead.
- Hunt down the last weakness honestly: a brain-style learner that finally sorts as sharply as the LLM, so the brain can win the accuracy ribbon too without giving up its tenfold edge.

---

*Run the championship yourself: `python3 test_009.py`*
