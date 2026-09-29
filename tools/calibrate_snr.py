"""Measure where a real pulse sits against where nothing sits, in SNR.

There is no accept/reject gate any more, but the separation this prints is still
worth watching: if a real pulse stops scoring far above noise, something in the
chain has broken. Do not turn the numbers back into a threshold -- they come
from signals written in this repository, not from a reference measurement.

    venv/Scripts/python.exe tools/calibrate_snr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ppg  # noqa: E402

BASE = np.array([160.0, 120.0, 110.0])
# Through a fingertip the light is transmitted, so red is modulated most.
TRANSMISSION = np.array([1.0, 0.45, 0.2])
# Reflected off skin, haemoglobin absorption puts the most modulation in green
# (Wang et al., 2017).
REFLECTION = np.array([0.33, 0.77, 0.53])


def synth(source, bpm=72.0, fps=30.0, seconds=13.0, strength=0.01, noise=0.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    count = int(seconds * fps)
    times = np.arange(count) / fps
    phase = 2 * np.pi * (bpm / 60.0) * times
    # A real pulse is not a sine: the systolic peak and the dicrotic notch put
    # real energy at the harmonics.
    wave = np.sin(phase) + 0.45 * np.sin(2 * phase + 0.6) + 0.2 * np.sin(3 * phase + 1.1)

    direction = TRANSMISSION if source == "finger" else REFLECTION
    rgb = BASE * (1.0 + strength * np.outer(wave, direction))
    rgb = rgb + drift * np.sin(2 * np.pi * 0.15 * times)[:, None]
    rgb = rgb + rng.normal(0, noise, rgb.shape)
    return times, rgb[:, None, :]


def sweep(source, label, seeds=20, **kwargs):
    values = []
    for seed in range(seeds):
        times, regions = synth(source, seed=seed, **kwargs)
        result = ppg.estimate(times, regions, source=source)
        values.append(-np.inf if result is None else result.snr_db)
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        print(f"  {label:28s} no estimate at all in {seeds} runs")
        return values
    print(
        f"  {label:28s} min {min(finite):7.2f}  median {np.median(finite):7.2f}  "
        f"max {max(finite):7.2f}   ({len(finite)}/{seeds} produced a number)"
    )
    return values


def main():
    for source in ("finger", "face"):
        print(f"\n=== {source} ===")
        print(" signal present:")
        present = []
        for bpm in (50, 72, 96, 130):
            present += sweep(source, f"pulse {bpm} bpm, clean", bpm=bpm, seeds=5)
        present += sweep(source, "pulse 72 bpm, noisy", bpm=72, noise=0.4, seeds=10)
        present += sweep(source, "pulse 72 bpm, drifting", bpm=72, drift=4.0, seeds=10)

        print(" nothing present:")
        absent = []
        absent += sweep(source, "noise only", strength=0.0, noise=3.0)
        absent += sweep(source, "drift only", strength=0.0, drift=25.0, noise=0.3)
        absent += sweep(source, "flat", strength=0.0)

        lo = [v for v in present if np.isfinite(v)]
        hi = [v for v in absent if np.isfinite(v)]
        floor = min(lo) if lo else float("nan")
        ceiling = max(hi) if hi else float("-inf")
        print(f" weakest real pulse : {floor:7.2f} dB")
        print(f" strongest non-pulse: {ceiling:7.2f} dB")
        if lo and floor > ceiling:
            print(f" -> a gate anywhere in ({ceiling:.2f}, {floor:.2f}) separates them")
        else:
            print(" -> NO GAP: the two overlap, so no threshold can separate them")


if __name__ == "__main__":
    main()
