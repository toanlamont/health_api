"""Take a saved capture apart and say why it did or did not yield a pulse.

    venv/Scripts/python.exe tools/inspect_capture.py                # newest
    venv/Scripts/python.exe tools/inspect_capture.py captures/*.json

A rejected capture reports one SNR number, which cannot distinguish between a
pulse buried under drift, a pulse erased by the camera's temporal noise
reduction, and no pulse at all. These are different problems with different
fixes, so each is checked separately here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ppg  # noqa: E402

CHANNELS = ("R", "G", "B")
PULSE_BAND = (0.75, 3.0)
# Well above any physiology, so whatever is here is the camera's own noise.
NOISE_BAND = (6.0, 13.0)


def band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    return float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else 0.0


def describe(path: Path):
    blob = json.loads(path.read_text(encoding="utf-8"))
    times = np.asarray(blob["times"], dtype=float)
    regions = int(blob["region_count"])
    values = np.asarray(blob["values"], dtype=float).reshape(len(times), regions, 3)
    source = blob["source"]

    print(f"\n{'=' * 72}\n{path.name}   source={source}  frames={len(times)}  regions={regions}")
    outcome = blob.get("outcome") or {}
    if outcome:
        print("  as measured: " + "  ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in outcome.items()))

    # --- frame timing ---
    gaps = np.diff(times)
    fps = 1.0 / np.median(gaps)
    print(f"\n  timing      {fps:5.1f} fps median, span {times[-1] - times[0]:.1f}s, "
          f"jitter {np.std(gaps) * 1000:.1f}ms, worst gap {np.max(gaps) * 1000:.0f}ms")

    uniform, fps = ppg.resample_uniform(times, values.reshape(len(times), -1))
    uniform = uniform.reshape(len(uniform), regions, 3)

    for r in range(regions):
        if regions > 1:
            print(f"\n  --- region {r} ---")
        for c, name in enumerate(CHANNELS):
            x = uniform[:, r, c]
            dc = float(np.mean(x))
            if dc <= 0:
                continue

            # Drift: how much of the swing is the slow trend rather than anything
            # periodic. Auto-exposure settling looks like a large number here.
            trend = np.polyval(np.polyfit(np.arange(len(x)), x, 2), np.arange(len(x)))
            drift_pp = float(trend.max() - trend.min())

            detrended = x - trend
            freqs, psd = sps.welch(detrended, fs=fps, nperseg=min(256, len(x)))
            pulse = band_power(freqs, psd, *PULSE_BAND)
            noise = band_power(freqs, psd, *NOISE_BAND)

            # A sensor's own noise is white, so its power density should be flat
            # from the pulse band up through the noise band. Temporal denoising
            # in the camera pipeline rolls the high end off, and it takes any
            # pulse with it.
            width_ratio = (PULSE_BAND[1] - PULSE_BAND[0]) / (NOISE_BAND[1] - NOISE_BAND[0])
            flatness = (noise / width_ratio / pulse) if pulse > 0 else float("nan")

            band = sps.butter(4, [PULSE_BAND[0], PULSE_BAND[1]], btype="band", fs=fps, output="sos")
            filtered = sps.sosfiltfilt(band, detrended)
            modulation = float(np.percentile(filtered, 99) - np.percentile(filtered, 1)) / dc

            steps = np.unique(np.abs(np.diff(np.unique(x))))
            steps = steps[steps > 1e-12]
            resolution = float(steps.min()) if steps.size else float("nan")

            print(f"  {name}  dc {dc:6.1f}   drift {drift_pp:6.2f} ({drift_pp / dc * 100:5.2f}% of dc)"
                  f"   pulse-band modulation {modulation * 100:5.3f}%")
            print(f"        noise/pulse spectral flatness {flatness:6.2f}"
                  f"   smallest step {resolution:.4f} ({resolution / dc * 100:.4f}% of dc)")

    # --- the verdict ---
    print()
    best_flat, best_mod = [], []
    for r in range(regions):
        for c in range(3):
            x = uniform[:, r, c]
            dc = float(np.mean(x))
            if dc <= 0:
                continue
            trend = np.polyval(np.polyfit(np.arange(len(x)), x, 2), np.arange(len(x)))
            detrended = x - trend
            freqs, psd = sps.welch(detrended, fs=fps, nperseg=min(256, len(x)))
            p = band_power(freqs, psd, *PULSE_BAND)
            n = band_power(freqs, psd, *NOISE_BAND)
            wr = (PULSE_BAND[1] - PULSE_BAND[0]) / (NOISE_BAND[1] - NOISE_BAND[0])
            if p > 0:
                best_flat.append(n / wr / p)
            band = sps.butter(4, list(PULSE_BAND), btype="band", fs=fps, output="sos")
            best_mod.append(abs(np.percentile(sps.sosfiltfilt(band, detrended), 99)
                                - np.percentile(sps.sosfiltfilt(band, detrended), 1)) / dc)

    result = ppg.estimate(times, values, source=source)
    print(f"  re-analysed: " + ("nothing usable" if result is None else
          f"{result.bpm:.1f} bpm, {result.snr_db:.2f} dB via {result.method}, "
          f"(snr {result.snr_db:.2f} dB)"))

    if best_flat:
        flat = float(np.nanmin(best_flat))
        print(f"  quietest channel has {flat:.2f}x more noise power per Hz above 6 Hz "
              f"than in the pulse band")
        if flat < 0.25:
            print("  -> the high frequencies are far quieter than white noise would be.")
            print("     Something in the camera pipeline is averaging frames together,")
            print("     which removes a 1 Hz pulse along with the noise.")
        elif flat > 4:
            print("  -> there is much more power in the pulse band than above it, but no")
            print("     clean peak: that is movement or lighting change, not a pulse.")
        else:
            print("  -> noise looks white, so the pipeline is not smoothing. If no pulse")
            print("     shows up, it is not reaching the sensor in the first place.")
    if best_mod:
        print(f"  strongest pulse-band swing: {max(best_mod) * 100:.3f}% of dc "
              f"(a fingertip usually gives 0.5-2%, a face 0.1-0.5%)")


def main():
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        folder = Path(__file__).resolve().parent.parent / "captures"
        found = sorted(folder.glob("*.json"))
        if not found:
            print(f"no captures in {folder} -- take a measurement first")
            return
        paths = found[-2:]
    for path in paths:
        describe(path)


if __name__ == "__main__":
    main()
