# Test 003 — Four Shapes and a Shouting Match

*The one where the lazy robot almost wins, then trips over its own feet.*

---

## What we changed

Last time the puzzle was too easy. Both robots got a perfect score, so we could not tell who was actually smarter, only who worked less. This time we made it bigger, harder, and a lot messier.

- The page grew from 5 boxes wide to **7 boxes wide**. More room, more blank space.
- We added more shapes. Now there are **four** to tell apart: a plus, an X, a square, and a triangle.
- We smudged the drawings really badly. Before, a few boxes got messed up. Now about **one in three boxes is wrong**.

Here is how rough the drawings got. This is supposed to be a square:

```
    . # # # . . #
    # . # . # . #
    # . . . . # .
    # . # . # . .
    # # . # # . #
    . . . . . . #
    # # . . . . .
```

You can sort of see the square hiding in there if you already know it is a square. The robots did not get that hint. They had to dig it out of the mess.

---

## The new trick: a shouting match

Last time there was one question with a yes or no answer, so one output was enough. Now there are four possible answers, so each robot grows **four output neurons**, one for each shape.

When a drawing comes in, all four neurons look at it and shout a number for how much it looks like their shape. The plus-neuron shouts how plus-y it is, the square-neuron shouts how square-y it is, and so on. Whoever shouts loudest wins, and that becomes the robot's guess.

This is called **winner-take-all**, and your brain does it constantly. Right now, while you read this, lots of little groups of brain cells are quietly voting on what each word means, and the loudest vote wins so fast you never notice the election happened.

---

## What happened

This time nobody got a perfect score. The mess was too strong. Good. Now we can finally see a difference.

| | Densey | Sparky |
|---|---|---|
| Best score it ever reached | 83.5% | 82.1% |
| Score at the very end | 81.5% | 72.1% |
| Pixels it had to read | 23,520,000 | 10,628,880 |
| How hard it worked | normal | 2.2x less |

Read those two score rows carefully, because they tell two different stories.

**Story one: at its best, Sparky almost tied.** 82.1% versus 83.5%. Barely more than one point apart. The cheap, brain-style robot got nearly as good as the textbook robot, while doing less than half the work. That is a big deal. It means looking at less does not have to mean knowing less.

**Story two: Sparky is a wobbler.** Densey learned smoothly and then sat still, scoring around 81 or 82 every single time we checked. Sparky bounced all over the place: 78, then 77, then 78, then 75, like a kid who can ride a bike but keeps wobbling and never quite stops wobbling. At the very last check it happened to wobble low, down to 72.

Why does Sparky wobble? Because of how it learns. Sparky only fixes itself when it makes a mistake, and when it does, it shoves its weights hard in a new direction. Shove, shove, shove. It never takes small careful steps, so it never fully settles down. Densey takes tiny smooth steps, so it glides to a stop.

---

## The surprise nobody ordered

Look at the work numbers. Last test Sparky worked 2.5x less. This test only 2.2x less. Sparky's lead got *smaller*. Why, on a bigger emptier page, did the lazy robot save less?

The noise did it.

Sparky's whole superpower is skipping blank boxes. But smudging the drawing fills blank boxes with random ink. The messier we made things, the fewer truly empty boxes were left to skip. We took away the very thing Sparky was good at ignoring.

This is a real lesson for the whole project, and it is easy to miss. A brain saves the most energy in a clean, quiet, mostly-empty world, because there is lots of nothing to ignore. Fill the world with noise and the savings shrink. So part of building a brain-style AI is not only the robot. It is also feeding it clean, sparse information in the first place.

---

## Three things we learned this test

1. A brain-style robot can almost match a normal one even on a hard, four-way puzzle. At its best it was within about one point.
2. The price of cheap learning is wobble. Sparky learns only from its mistakes with big rough shoves, so it never fully settles, and on a bad day it dips. Densey is slower-thinking but steady.
3. Noise eats the savings. Sparky's whole edge is skipping empty space, so a messy world shrinks its lead. Clean, sparse input is part of the design, not just the robot.

---

## What is still wrong

Sparky's wobble is the next mountain to climb. It can clearly reach a good score. It just cannot hold onto it. A real brain has tricks to calm itself down and stop wobbling, like a thermostat that keeps a room from getting too hot or too cold. Sparky has none of those yet, so it overshoots forever.

---

## Next test ideas

- Give Sparky a calm-down rule (smaller shoves over time, or a brake) and see if its wobble settles closer to its best score.
- Stack a hidden middle layer on both robots. This is the big one. Real thinking needs layers, and teaching layers without the usual textbook math is the hardest and most important problem in the whole project.
- Try clean, naturally sparse pictures (like little stick figures on a big empty page) and watch Sparky's savings shoot back up past 3x.

---

*Run it yourself: `python3 test_003.py`*
