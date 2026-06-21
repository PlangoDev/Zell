# Sleep, Dreams, and Not Forgetting

### A Research Paper for Five-Year-Olds

*Plango Labs · Test 008 · written so a kid could follow it and still true enough for a grown-up*

---

## Abstract

*(This is the part of a paper where scientists tell you the whole story in a tiny box, so you know if you want to read the rest. Here is ours.)*

Robots have an embarrassing problem. When you teach a robot a new thing, it forgets the old thing. Not a little. All the way. We taught our brain-robot the digits 0 to 4, and it was great at them, 99 out of 100. Then we taught it 5 to 9. It got good at those, and its score on 0 to 4 dropped to **zero**. Gone. Like it had never seen them.

Then we let the robot do something people do every night. We let it **sleep**. While it slept it replayed its old memories to itself, quietly, a few times. When it woke up, it remembered the old digits again. The forgetting was undone. And it only needed to keep a tiny scrapbook of memories, about 7 out of every 100 it had seen, to pull this off.

This is a real thing brains do, and now our robot does it too.

---

## 1. Introduction: the robot that forgets

Imagine you learned to read. Then next week you learned to ride a bike. And the moment you could ride the bike, you completely forgot how to read. Every letter, gone.

That would be horrifying. And that is exactly what happens to normal robot brains. Scientists gave it a scary name: **catastrophic forgetting**. "Catastrophic" is a big word that means "as bad as it possibly gets."

Why does it happen? A robot brain learns by nudging little knobs inside itself until it gets answers right. When you teach it something new, it nudges the same knobs again to fit the new thing, and the settings that knew the old thing get shoved out of the way. The robot is not being lazy. It just has no reason to protect the old knobs, so it stomps right over them.

You do not do this. You learned to read years ago and you still can. So your brain must have a trick the robot is missing. This paper is about stealing that trick.

---

## 2. Background: what your brain does at night

Here is the trick, and it is wonderful. It happens while you are asleep and have no idea it is going on.

You have two memory helpers. One is fast and one is slow.

The **fast helper** is called the hippocampus. It is like a notebook. When something happens, it scribbles it down right away. Quick, but small. It cannot hold a whole lifetime.

The **slow helper** is called the cortex. It is like a big, careful library. It learns things properly and keeps them forever, but it learns slowly, by seeing things again and again.

So at night, the fast notebook reads its scribbles out loud to the slow library, over and over. The library gets to hear today's memories many times without you having to live the day again. Bit by bit, the new memories get filed into the library next to the old ones, instead of on top of them. Nothing gets stomped.

Scientists call this **replay**, and the whole filing process is called **consolidation**. It is a big reason grown-ups say "sleep on it" before a test. Your hippocampus literally spends the night teaching your cortex.

We wanted to know: if we give our robot a notebook and let it replay at night, will it stop forgetting?

---

## 3. Methods: what we actually built

*(Methods is the part where scientists explain exactly what they did, so anyone could do it too. No magic, no hiding.)*

**More pictures.** We started with 1,400 little pictures of handwritten numbers. To make the job harder and more lifelike, we made copies and wobbled each one: tilted it a bit, slid it sideways, sprinkled some fuzz. That turned 1,400 pictures into **5,600**. Now the robot had to learn the *idea* of each number, not memorize one picture.

**Two lessons.** We split the numbers into two lessons. Lesson A was the digits 0, 1, 2, 3, 4. Lesson B was 5, 6, 7, 8, 9. We taught them one after the other, like school days, to see if Lesson B would clobber Lesson A.

**A notebook (the hippocampus).** We gave the robot a small scrapbook. Every so often it tucked away a few example pictures of what it had seen. Not all of them. Just 40 pictures per digit, a little sample.

**Sleep.** After the lessons, we let the robot "sleep." Sleeping just meant: take the scrapbook out and practice on those saved pictures a few times, quietly, mixing the old digits and the new ones together. That is replay.

**Dreams.** We also tried a fancier sleep, where instead of replaying the saved pictures exactly, the robot replayed *wobbled, transformed* versions of them, the way real dreams twist things around. We wanted to know if dreaming helps.

---

## 4. Results: what happened

*(Results is the part with the actual numbers. We are not allowed to fib in here. Whatever happened, happened.)*

### 4.1 The forgetting was real, and total

| Day | What we taught | Score on the new lesson | Score on the OLD lesson |
|---|---|---|---|
| Day 1 | digits 0–4 | 99.5% | — |
| Day 2 | digits 5–9 | 93.4% | **0.0%** |

After Day 2 the robot was wonderful at the new digits and could not get a **single** old one right. Zero out of a hundred. The new lesson had completely stomped the old one. This is the catastrophe, and it showed up exactly as the textbooks promised.

### 4.2 Then it slept, and the memories came back

| What the robot did | Score on Lesson A | Score on Lesson B | Both together |
|---|---|---|---|
| stayed awake | 0.0% | 93.4% | 46.7% |
| slept 1 night | 70.6% | 84.7% | 77.6% |
| slept 2 nights | 75.2% | 83.1% | 79.1% |
| slept 3 nights | 77.1% | 79.8% | 78.4% |
| slept 5 nights | **78.0%** | **85.8%** | **81.9%** |

Look at the first night. The old lesson went from **0% back up to 70%** after a single night of replaying memories. By five nights it held both lessons at once, scoring in the high 70s and 80s on each. The robot that knew only half its numbers now knew all of them again.

And here is the part that made us grin: the scrapbook only held **400 pictures**, which is about **7%** of everything the robot had ever seen. A tiny notebook, replayed at night, was enough to rescue a whole forgotten lesson. Brains are not big because they remember everything. They are clever about what little they keep.

### 4.3 Dreaming did not help (and we are telling you anyway)

We hoped the fancy dreaming sleep would beat plain replay. It did not.

| Kind of sleep | Both lessons together |
|---|---|
| plain replay (exact memories) | 81.0% |
| dream replay (wobbled memories) | 68.2% |

Replaying wobbled, dreamlike versions actually did **worse**. Our guess for why: with only a few saved pictures, twisting them too hard dragged them away from what real digits look like, so the robot drifted off. Real brains almost certainly dream in a smarter way than our clumsy wobble. We tried it, it flopped, and a good paper writes down the flops too, not just the wins.

---

## 5. Discussion: what it all means

The big finding is simple. A robot brain forgets old things the instant it learns new ones, and **sleep fixes it**. By keeping a small notebook of memories and replaying them quietly at night, our robot held on to old lessons while still learning new ones. That is the same trick your own head pulls every night.

This matters for the whole project. We have been building a brain-style robot that is wildly cheaper to run than a normal AI. But cheap is no good if it forgets everything it learns. Now it does not have to. Memory plus sleep gives it something normal AIs are famously bad at: learning new things without throwing away the old.

There is also a lovely tidiness here. The same sparse, separated memories that made the robot efficient (test 005), and that let it learn from one example (test 007), are the memories it now replays to avoid forgetting. One idea keeps paying off in new ways. That is usually a sign you are near something true.

We should be honest about the rough edges:

- The old lesson came back to about 78%, not all the way to its first-day 99%. A small notebook replayed a few times rescues *most* of a memory, not every last drop.
- Dreaming, in our simple version, hurt rather than helped. The real brain's version is surely cleverer, and figuring out the clever version is a job for another day.
- The robot's slow library still does not learn as sharply as the best normal robots. Sleep helped it keep what it had. It did not make it a better learner. That older problem is still waiting for us.

---

## 6. Conclusion

We gave a robot a notebook and a bedtime. It stopped forgetting. From a tiny scrapbook of old memories, replayed over a few nights, it brought a completely erased lesson back from zero to nearly fourscore out of a hundred, all while keeping the new lesson too. The fanciest part, dreaming, did not work yet, and we said so. The plain, humble part, just sleeping on it, worked beautifully.

Brains sleep for a reason. Now our robot does too.

---

## 7. References (the grown-up ideas hiding in here)

- The two-helper idea, fast notebook and slow library, is really called **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995).
- The night-time read-aloud is **hippocampal replay** and **systems consolidation**, seen in real sleeping brains.
- The robot forgetting everything is **catastrophic forgetting** (McCloskey & Cohen, 1989).
- Fixing it by replaying saved examples is known to engineers as **experience replay**, and the dreaming version is **generative replay** (Shin et al., 2017).

---

## 8. Future Work (what we want to try next)

- Teach the robot to dream *properly*, with a smarter way of imagining new versions of its memories instead of just wobbling them.
- Let the night-time replay slowly make the slow library genuinely smarter over many nights, not just stop it forgetting.
- Make remembering cheap, so the notebook does not cost more than it saves.
- Keep chasing the oldest problem of all: a brain-style learner that learns as sharply as the big machines, but for a sliver of the energy.

---

*Run the experiment yourself: `python3 test_008.py`*
