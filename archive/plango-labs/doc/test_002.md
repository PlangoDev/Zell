# Test 002 — Two Robots Learn Their Shapes

*A story about a lazy robot who was actually the smart one.*

---

## Meet the robots

We built two little robots and gave them one job: look at a drawing and say if it is a **plus sign (+)** or an **X**.

The drawings are tiny, just 5 boxes across and 5 boxes down. Some boxes have ink in them. Some are empty.

This is what a plus and an X look like when they are drawn neatly:

```
   a plus              an X
   . . # . .          # . . . #
   . . # . .          . # . # .
   # # # # #          . . # . .
   . . # . .          . # . # .
   . . # . .          # . . . #
```

The `#` is ink. The `.` is empty space.

Here is the tricky part. Both shapes have ink right in the very middle box. So the robots cannot just check the middle. They have to look at the arms to know which is which. It is a real little puzzle.

---

## The messy drawings

We did not give the robots neat shapes. That would be too easy. We gave them **messy** ones, like a shape drawn by a kid who wiggles the crayon. The computer takes a neat shape and randomly smudges a few boxes. An inked box might go blank. A blank box might get a smudge of ink.

So the robots had to learn the *idea* of a plus and the *idea* of an X, not just memorize one picture. Here are two messy ones they had to figure out:

```
   a messy +           a messy X
   . # # . .          # . . . .
   # . # . .          . # . # .
   # # # # #          . . # . .
   . . . . .          . # . # #
   . . # # #          # # . # #
```

Still a plus and still an X if you squint. The robots had to squint too.

---

## The two robots are different inside

**Densey** is the eager one. Densey looks at *every single box*, all 25 of them, every single time. Even the empty ones. Densey stares at the blank boxes just as hard as the inked ones. "Gotta check everything!" says Densey.

**Sparky** is the lazy one. Or that is what everyone thought. Sparky only looks at the boxes that have ink. If a box is empty, Sparky does not even glance at it. "Why would I look at nothing?" says Sparky.

And here is the secret: most of the drawing IS nothing. Out of 25 boxes, only about 10 have ink. The other 15 are empty space. Densey checks all 15 empty boxes anyway. Sparky skips them.

---

## We taught them 40 times over

We showed each robot a thousand messy drawings, then showed them all again, forty times through. Every time a robot guessed wrong, it adjusted itself a tiny bit to do better next time. That adjusting is what "learning" means.

Then we gave them 500 brand new drawings they had never seen, to check if they actually learned or just memorized.

---

## What happened

Both robots got **every single one right**. 100 out of 100. They both really learned their shapes.

But look at how much work each one did:

| | Densey | Sparky |
|---|---|---|
| Got the shapes right? | Yes, 100% | Yes, 100% |
| How many pixels it looked at | 1,000,000 | 397,400 |
| How hard it worked | normal | **2.5x less** |

Same answer. Same score. But Sparky got there by looking at **2 and a half times fewer pixels.**

The lazy robot was not lazy at all. It was smart. It figured out that empty space has no information in it, so looking at it is a waste. Sparky spent its effort only where something was actually happening.

---

## Why this matters (the grown-up part)

This is the whole idea behind the project in one tiny picture.

Your brain is Sparky. When you look at something, your brain does not fire up every cell you own. It wakes up the few cells that matter and lets the rest nap. That is why your brain runs on about as much power as a dim light bulb while a giant computer brain needs a whole power plant.

Today's big AI models are Densey. They check everything, every time, even the parts that do not matter for the question.

In test 001 the savings were 2x. Here it grew to 2.5x, just because shapes have more empty space than random dots do. The lesson is clear: **the more nothing there is to skip, the more the brain-style robot wins.** Real life is mostly nothing happening. That is where the giant savings are hiding.

---

## What we learned this time

- A brain-style robot can learn a real shape puzzle, not just a counting trick.
- It got the same perfect score as the normal robot.
- It did it with much less work, and the messier and emptier the picture, the bigger its lead.
- The savings come from one simple rule: **do not spend energy looking at nothing.**

---

## What is still too easy

Both robots scored a perfect 100%, which means the puzzle was still a little too simple to tell a *great* robot from a *good* one. Next time we make it harder, so the scores split apart and we can see which way of learning is actually stronger, not just cheaper.

---

## Next test ideas

- Add a third shape so it is not just a yes/no guess (maybe a circle, or a smiley).
- Make the drawings way messier so 100% becomes impossible and we see who copes better.
- Grow the grid bigger, because a bigger emptier page should hand Sparky an even bigger win.
- Give both robots a hidden middle layer and see if Sparky can still learn without the usual textbook math.

---

*Run it yourself: `python3 test_002.py`*
