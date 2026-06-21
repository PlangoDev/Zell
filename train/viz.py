#!/usr/bin/env python3
"""Colorful live training dashboard for the Zell trainer (24-bit ANSI, no deps).

Built for high visual feedback: a neon gradient progress bar, a live loss
sparkline, tok/s, ETA, and a GPU-memory bar, redrawn every step, plus periodic
scrolling snapshots. Renders in a Kaggle notebook cell (ANSI colors + carriage
return). Set NO_COLOR=1 to fall back to plain text.
"""
import os
from collections import deque

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
NO_COLOR = bool(os.environ.get("NO_COLOR"))

# neon palette (r, g, b)
GREEN = (57, 255, 20)
CYAN = (0, 245, 255)
PINK = (255, 32, 200)
YELLOW = (255, 226, 0)
ORANGE = (255, 140, 0)
PURPLE = (170, 90, 255)
RED = (255, 60, 60)
GREY = (110, 110, 125)
WHITE = (235, 235, 245)

SPARK = "▁▂▃▄▅▆▇█"


def fg(c):
    return "" if NO_COLOR else f"\x1b[38;2;{c[0]};{c[1]};{c[2]}m"


def paint(text, c, bold=False):
    if NO_COLOR:
        return text
    return f"{BOLD if bold else ''}{fg(c)}{text}{RESET}"


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _grad(t):
    """green -> cyan -> yellow -> pink across t in [0,1]."""
    stops = [GREEN, CYAN, YELLOW, PINK]
    seg = t * (len(stops) - 1)
    i = min(int(seg), len(stops) - 2)
    return _lerp(stops[i], stops[i + 1], seg - i)


def gradient_bar(frac, width=34):
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    if NO_COLOR:
        return "[" + "#" * filled + "-" * (width - filled) + "]"
    cells = []
    for i in range(width):
        if i < filled:
            cells.append(fg(_grad(i / max(width - 1, 1))) + "█")
        else:
            cells.append(fg(GREY) + "░")
    return "".join(cells) + RESET


def sparkline(vals, width=32):
    if not vals:
        return ""
    v = list(vals)[-width:]
    lo, hi = min(v), max(v)
    rng = (hi - lo) or 1.0
    if NO_COLOR:
        return "".join(SPARK[int((x - lo) / rng * (len(SPARK) - 1))] for x in v)
    out = []
    for x in v:
        t = (x - lo) / rng
        idx = int(t * (len(SPARK) - 1))
        # low loss = green (good), high = pink (hot)
        out.append(fg(_lerp(GREEN, PINK, t)) + SPARK[idx])
    return "".join(out) + RESET


def mem_bar(used, total, width=10):
    if not total:
        return ""
    frac = max(0.0, min(1.0, used / total))
    filled = int(round(frac * width))
    col = GREEN if frac < 0.7 else (YELLOW if frac < 0.9 else RED)
    if NO_COLOR:
        return f"[{'#' * filled}{'-' * (width - filled)}]"
    return fg(col) + "▰" * filled + fg(GREY) + "▱" * (width - filled) + RESET


def hms(sec):
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class LiveDashboard:
    """Holds run state and renders the live line + snapshots."""

    def __init__(self, total_steps, tokens_per_step, title="ZELL",
                 subtitle="", meta_line=""):
        self.total = max(1, total_steps)
        self.tps = tokens_per_step
        self.title, self.subtitle, self.meta_line = title, subtitle, meta_line
        self.losses = deque(maxlen=60)
        self.best = None
        self.prev = None

    def banner(self):
        w = 64
        top = "╔" + "═" * w + "╗"
        bot = "╚" + "═" * w + "╝"
        print(paint(top, PURPLE, bold=True), flush=True)
        print(paint("║ ", PURPLE) + paint(self.title + " · continued-pretrain", CYAN, bold=True)
              + " " * (w - len(self.title) - 22) + paint("║", PURPLE), flush=True)
        if self.subtitle:
            print(paint("║ ", PURPLE) + paint(self.subtitle, WHITE)
                  + " " * (w - len(self.subtitle) - 1) + paint("║", PURPLE), flush=True)
        if self.meta_line:
            print(paint("║ ", PURPLE) + paint(self.meta_line, GREEN)
                  + " " * (w - len(self.meta_line) - 1) + paint("║", PURPLE), flush=True)
        print(paint(bot, PURPLE, bold=True), flush=True)

    def add_loss(self, loss):
        self.losses.append(loss)
        self.best = loss if self.best is None else min(self.best, loss)

    def _trend(self, loss):
        if self.prev is None:
            arrow, col = "·", GREY
        elif loss < self.prev - 1e-4:
            arrow, col = "▼", GREEN
        elif loss > self.prev + 1e-4:
            arrow, col = "▲", RED
        else:
            arrow, col = "=", YELLOW
        return paint(arrow, col)

    def tick(self, step, loss, tok_s, gpu_used=0.0, gpu_total=0.0):
        frac = step / self.total
        bar = gradient_bar(frac)
        pct = paint(f"{frac * 100:5.1f}%", _grad(frac), bold=True)
        st = paint(f"{step:>4}", CYAN) + paint("/", GREY) + f"{self.total}"
        if loss is None:
            loss_s = paint("  ··· ", GREY)
        else:
            loss_s = paint(f"{loss:6.3f}", ORANGE, bold=True) + " " + self._trend(loss)
        tps = paint(f"{tok_s:>6,.0f}", GREEN) + paint(" tok/s", GREY)
        elapsed_frac = step / self.total
        eta = (1 - elapsed_frac) / elapsed_frac * self._elapsed if elapsed_frac > 0 else 0
        eta_s = paint(hms(eta), PINK)
        spark = sparkline(self.losses)
        mem = ""
        if gpu_total:
            mem = " " + paint("mem ", GREY) + mem_bar(gpu_used, gpu_total) + \
                  paint(f" {gpu_used:4.1f}/{gpu_total:.0f}G", GREY)
        line = (f"\r {paint('ZELL', PINK, bold=True)} {bar} {pct} "
                f"{paint('│', GREY)} {st} {paint('│', GREY)} "
                f"{paint('loss', GREY)} {loss_s} {paint('│', GREY)} {tps} "
                f"{paint('│', GREY)} {paint('ETA', GREY)} {eta_s}{mem} "
                f"{paint('│', GREY)} {spark}  ")
        print(line, end="", flush=True)

    def snapshot(self, step, label="checkpoint"):
        best = paint(f"{self.best:.3f}", GREEN, bold=True) if self.best is not None else "?"
        cur = paint(f"{self.losses[-1]:.3f}", ORANGE) if self.losses else "?"
        bar = "═" * 3
        msg = (f"\n {paint(bar, PURPLE)} {paint(label.upper(), PURPLE, bold=True)} "
               f"step {paint(str(step), CYAN, bold=True)}/{self.total} "
               f"{paint('│', GREY)} loss {cur} {paint('│', GREY)} best {best} "
               f"{paint('│', GREY)} {sparkline(self.losses, 44)}")
        print(msg, flush=True)

    # set by the callback each tick so ETA can use real elapsed seconds
    _elapsed = 1.0

    def set_elapsed(self, sec):
        self._elapsed = max(sec, 1e-6)

    def done(self):
        best = paint(f"{self.best:.3f}", GREEN, bold=True) if self.best is not None else "?"
        print("\n" + paint("  ▰▰▰ training complete ", GREEN, bold=True)
              + paint(f"· best loss {best}", WHITE) + "\n", flush=True)
