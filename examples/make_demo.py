#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a small demo audio file so the CLI can be tried without a library.

The repository ships no audio (large binaries, unclear licensing), so this
script synthesises a short chord-and-pulse clip instead. It is deliberately
simple: the point is to have *something* to analyse, not to sound good.

Usage:
    python examples/make_demo.py                # writes examples/demo.wav
    python examples/make_demo.py --bpm 140 --mode major --out fast.wav
    python moodio.py analyze --path examples/demo.wav
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SR = 22050


def midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def tone(freq: float, dur: float, amp: float, harmonics: int = 6) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        out += (amp / h) * np.sin(2 * np.pi * freq * h * t)
    env = np.minimum(1.0, t / 0.02) * np.minimum(1.0, np.maximum(dur - t, 0) / 0.05)
    return out * np.clip(env, 0, 1)


def click(amp: float, rng: np.random.Generator) -> np.ndarray:
    n = int(0.035 * SR)
    return amp * rng.normal(0, 1, n) * np.exp(-np.linspace(0, 9, n))


def synth(bpm: float, mode: str, root: int, amp_db: float, dur: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    buf = np.zeros(n)

    third = 4 if mode == "major" else 3
    beat = 60.0 / bpm
    bar = beat * 4
    progression = [0, 5, 3, 7]

    t, idx = 0.0, 0
    while t < dur - 0.05:
        chord_root = root + progression[idx % len(progression)]
        notes = [chord_root, chord_root + third, chord_root + 7]
        seg_dur = min(bar, dur - t)
        acc = np.zeros(int(seg_dur * SR))
        for m in notes:
            partial = tone(midi_to_hz(m), seg_dur, amp=1.0)
            if len(partial) < len(acc):
                partial = np.pad(partial, (0, len(acc) - len(partial)))
            acc += partial[: len(acc)]
        start = int(t * SR)
        end = min(start + len(acc), n)
        buf[start:end] += acc[: end - start]
        t += seg_dur
        idx += 1

    t = 0.0
    while t < dur - 0.05:
        c = click(0.9 if (t % bar) < 1e-6 else 0.55, rng)
        start = int(t * SR)
        end = min(start + len(c), n)
        buf[start:end] += c[: end - start]
        t += beat

    peak = float(np.abs(buf).max())
    if peak > 0:
        buf = buf / peak
    target = 10.0 ** (amp_db / 20.0)
    rms = float(np.sqrt(np.mean(buf ** 2))) + 1e-12
    buf = buf * (target / rms)
    peak = float(np.abs(buf).max())
    if peak > 0.98:
        buf *= 0.98 / peak
    return buf.astype(np.float32)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Synthesise a demo clip for moodio")
    ap.add_argument("--bpm", type=float, default=100.0)
    ap.add_argument("--mode", choices=["major", "minor"], default="minor")
    ap.add_argument("--root", type=int, default=57, help="root MIDI note (57 = A)")
    ap.add_argument("--amp-db", type=float, default=-18.0)
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(here, "demo.wav"))
    args = ap.parse_args()

    try:
        import soundfile as sf
    except ImportError:
        print("soundfile is required: pip install soundfile", file=sys.stderr)
        return 1

    y = synth(args.bpm, args.mode, args.root, args.amp_db, args.duration, args.seed)
    sf.write(args.out, y, SR)
    print(f"wrote {args.out}  ({args.duration:.0f}s, {args.bpm:.0f} BPM, {args.mode})")
    print(f"try: python python/moodio.py analyze --path {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
