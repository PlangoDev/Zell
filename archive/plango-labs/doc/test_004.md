# Test 004 — The Wall, and the Trick That Climbs It

*The one where a robot learns to think in layers, using backward wires that were wired up completely at random, and it works anyway.*

This is the most important test so far. It is longer than the others because there is a lot worth understanding here, and every piece of it is a real idea that grown-up scientists argue about. Take it slow. It is worth it.

---

## The problem we kept dodging

In the first three tests, both our robots had a flat little brain. Pixels went in one side, an answer came out the other, with one layer of connections in between. That was enough for counting bits and telling a plus from an X.

But a flat brain has a ceiling, and it is a low one. There are puzzles a flat brain can **never** solve, not in a million years of practice. To get past them you need a brain with **layers**, where the first layer makes up little in-between ideas and a later layer uses those ideas to answer.

This test does three things at once:

1. Build a puzzle the flat brain literally cannot solve, and watch it fail.
2. Add a hidden layer and watch the wall crumble.
3. Solve the deepest puzzle of all, the one this whole project is really about: **how does a layered brain learn without the magic math that real brains cannot do?**

And while we are here, we fix Sparky's wobble from test 003.

---

## The puzzle: two switches and a stubborn lamp

Picture two light switches and one lamp. The lamp is wired up funny. It only turns on when **exactly one** switch is up. Both down, lamp off. Both up, lamp off. One up and one down, lamp on.

| Switch 1 | Switch 2 | Lamp |
|---|---|---|
| down | down | off |
| down | up | **on** |
| up | down | **on** |
| up | up | off |

This is called XOR, and it has been breaking simple brains since the 1960s. A famous book in 1969 pointed out that the flat networks of the day could not solve it, and that one observation more or less froze the whole field of AI research for over a decade. It is a tiny puzzle with a giant history.

We made it a little harder and more lifelike by letting each switch be *roughly* up or down, with a wobble, so the robot sees fuzzy clouds of switch positions instead of perfect ons and offs. It has to learn the rule, not memorize four cases.

---

## Why the flat brain hits a wall

Here is the heart of it, and it is beautiful once you see it.

Look at where the "lamp on" cases sit versus the "lamp off" cases. On is top-left and bottom-right. Off is bottom-left and top-right. The two on-cases are diagonal from each other. So are the two off-cases.

Now find the center of all the on-cases. It is smack in the middle, at half-up, half-up. Find the center of all the off-cases. It is **the exact same spot**, the middle.

The two groups have the same center. They are like two X shapes overlapping at the bullseye. A flat brain can only draw one straight line and say "on this side, lamp on; that side, lamp off." But no straight line on Earth can separate two groups that pile up on the same center point. Tilt the line any way you like. You always cut both groups in half.

So the flat brain is doomed to guess. And it does:

| Network | Settled score | Best it ever got |
|---|---|---|
| Flat, one layer | **53.4%** | 75.6% |

53 percent. A coin flip is 50. The flat brain trained for 150 rounds and ended up barely better than flipping a coin. The wall is real. You cannot out-practice it. You need a different shape of brain.

---

## Knocking the wall down with a hidden layer

So we add a middle layer. Ten hidden neurons sit between the switches and the lamp. The switches talk to the hidden neurons, the hidden neurons talk to the lamp.

Why does this help? Because the hidden neurons can invent **helper questions** that the raw switches could not ask. One hidden neuron can learn to mean "switch 1 up and switch 2 down." Another can learn "switch 1 down and switch 2 up." Those two helper questions ARE the lamp-on cases. The output neuron just has to notice when either helper lights up.

The flat brain could only weigh the switches directly. The deep brain can build new ideas out of the switches first, then reason about those. That is what layers buy you. That is what depth means.

And it works:

| Network | Settled score |
|---|---|
| Flat, one layer | 53.4% |
| **Deep, with a hidden layer** | **96.3%** |

From a coin flip to almost perfect, just by adding one layer of in-between ideas. The wall is gone.

---

## The real question: how does the hidden layer learn?

Here is where it gets deep, and here is where the whole project lives.

The output neuron is easy to teach. It sees the right answer, it sees what it guessed, and it nudges its weights to close the gap. Simple.

But how do you teach a **hidden** neuron? A hidden neuron never sees the right answer. It is buried in the middle. It only talks to other neurons. When the lamp is wrong, whose fault was it? How much should hidden neuron number 3 change? Nobody handed it a report card.

This is the single hardest problem in all of brain-style learning. It has a name: **credit assignment**. When the final answer is wrong, how do you fairly spread the blame backward to all the hidden helpers who contributed?

The textbook answer is **backpropagation**. It is the engine inside basically every AI you have heard of. The idea: to know how much hidden neuron 3 is to blame, look at how strongly neuron 3 was wired to the output. If neuron 3 was shouting loudly into the final answer, it shares more of the blame.

That works great on a computer. But look closely at what it demands, because there is a catch that matters enormously for this project.

---

## The catch backprop hides (and why brains can't do it)

To send blame back to hidden neuron 3, backprop needs to know the **exact strength of the forward wire** from neuron 3 to the output, and it sends the error signal backward along a path that mirrors that exact strength.

Think about what that asks of real biology. A real synapse is a one-way street. Neuron 3 fires forward into the output neuron through a connection of some strength. For backprop to work, there would have to be a second wire running *backward*, from the output all the way back to neuron 3, carrying a signal scaled by the **exact same strength** as the forward wire. And every time the forward wire changed strength, the backward wire would have to instantly change to match it perfectly.

Brains do not have that. There is no known way for a synapse to broadcast its precise strength to a separate return wire and keep the two in lockstep. Scientists call this the **weight transport problem**, and it is the biggest reason most people think the brain cannot be running backpropagation. The textbook answer to credit assignment is, as far as we can tell, biologically impossible.

So if brains do not use backprop, what could they possibly use? For decades this looked like a dead end. Then came a result that genuinely shocked people.

---

## The shock: random backward wires work anyway

In 2016 a group of researchers tried something that sounds like a joke. Instead of sending the error backward along wires that match the forward strengths, they used **fixed, completely random backward wires**. Wired up once, at the start, by rolling dice, and never touched again. No weight transport. No matching anything. Just random junk carrying the error signal back to the hidden layer.

It should not work. The blame being handed to each hidden neuron is, at first, basically nonsense. Random noise.

It works.

This is called **Feedback Alignment**, and it is one of those results that makes you sit up. We ran it. Same deep brain, same XOR puzzle, but the hidden layer learns its blame through random fixed wires instead of proper backprop:

| Network | Settled score |
|---|---|
| Deep + backprop (the textbook, biologically impossible) | 96.3% |
| Deep + random feedback (biologically plausible) | **96.7%** |

The brain-plausible method did not just kind of work. It **matched** backprop. Tied it. Beat it by a hair on this run.

---

## Why on earth does that work?

Here is the magic, and we measured it happening.

The random backward wires never change. So if anything is going to make them useful, it has to be the **forward** wires bending themselves to fit. And that is exactly what happens. The network slowly rotates its forward weights until the direction they point lines up with the direction the random backward wires are pointing. Once the two roughly agree, the random feedback stops being noise and starts being real, useful teaching.

We tracked how aligned the two were, from the very first training example. A score near +1 means "pointing the same way, useful." Near 0 means "sideways, useless." Negative means "pointing the opposite way, actively harmful."

| Examples seen | Alignment |
|---|---|
| 0 (before any training) | **-0.47** |
| 20 | -0.01 |
| 50 | 0.46 |
| 100 | 0.80 |
| 200 | 0.87 |
| 800 (one pass through the data) | **0.92** |

Read that top to bottom. At the very start the random wires were pointing the **wrong way**, worse than useless, actively dragging the hidden layer in the wrong direction. After just 20 examples the forward weights had swung around to sideways. By 100 examples they were strongly aligned. By one full pass they were at 0.92, nearly perfect agreement.

The random wires sat still the whole time. The brain turned itself to face them. It manufactured its own teaching signal out of noise. Nobody told the forward weights to do this. It falls out of the learning on its own.

That is the step in the right direction this whole project was looking for. A layered brain that learns its hidden layers **without** the one trick real brains cannot perform.

---

## Fixing Sparky's wobble while we are here

Remember test 003, where the brain-style robot kept bouncing around and never settled? We promised a calm-down rule. Here it is, and it is simple: make the learning steps **shrink** as training goes on. Big bold steps early when there is a lot to learn, tiny careful steps later when you are just polishing. Like sanding wood with rough paper first and fine paper last.

We measured the wobble as how much the score jitters in the final stretch. Lower is steadier.

| | Wobble |
|---|---|
| Random feedback, no calm-down | 0.98 |
| Random feedback, with calm-down | **0.20** |

Five times steadier. The robot reaches its good score and actually stays there now instead of twitching around it. Test 003's loose end, tied off.

---

## And the sparsity thread is still alive

One quiet detail. Even in this deeper brain, only **58%** of the hidden neurons were firing on a typical input. The rest sat silent at zero, thanks to the ReLU gate, and silent neurons cost nothing. Depth did not kill the efficiency story from the earlier tests. The two ideas, sparse activation and layered learning, stack together fine.

---

## What we actually learned, in plain words

1. A flat brain has a hard ceiling. Some puzzles, like the stubborn lamp, are impossible for it no matter how long it trains, because no straight line can split two groups that share a center.
2. Adding a hidden layer smashes that ceiling, because hidden neurons invent helper ideas that the raw inputs could not express. This is why real intelligence needs depth.
3. Teaching a hidden layer means solving credit assignment: spreading the blame backward fairly. The textbook tool, backprop, needs a trick (perfectly matched backward wires) that real brains almost certainly cannot do. This is the weight transport problem.
4. You can throw that trick away. Random fixed backward wires work just as well, because the forward weights rotate to align with them. We watched the alignment climb from -0.47 to +0.92 in a single pass. This is Feedback Alignment, and it is a real door into brain-plausible deep learning.
5. Shrinking the learning steps over time cures the wobble.

---

## Being honest about the limits

This is a real result, but it is a small one, and the project does not get to pretend otherwise.

- XOR with ten hidden neurons is a toy. Feedback Alignment is known to get **shakier as networks get deeper and wider**. On big modern-scale networks it falls behind backprop, and patching that gap is an active, unsolved research area. We have not shown it scales. We have shown it is real.
- We are still using smooth numbers and gradients under the hood, not the all-or-nothing spikes a real neuron sends. The spiking version is harder and comes later.
- We have one hidden layer. Brains have many, stacked deep. Each extra layer makes credit assignment harder, and that is exactly where these plausible methods start to strain.

None of that erases the win. We took the single biggest objection to brain-style deep learning, the weight transport problem, and showed a way around it that actually trains a working layered network. That is a real step, with real ground still ahead.

---

## Next test ideas

- Stack a second hidden layer and see how far the random-feedback trick stretches before it strains. This is where the real difficulty lives.
- Swap the smooth neurons for spiking ones and try to keep the layered learning working. This is the bridge back to the sparse, event-driven efficiency from tests 001 to 003.
- Push the puzzle from two switches up to a real little image again, now that the brain has layers, and see if depth plus sparsity together beat anything we have built so far.

---

*Run it yourself: `python3 test_004.py`*
