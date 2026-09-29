"""Score the analysis against heart rates read off a watch.

    venv/Scripts/python.exe tools/compare_captures.py

Reference readings go in REFERENCE, keyed by the capture's timestamp. Captures
without one are still listed, they just cannot be scored.

Everything here runs through app.ppg rather than rebuilding the pipeline. An
earlier version of this file resampled to a fixed 30 fps and kept its own copy
of the candidate list; both had drifted from the code, so it reported a mean
error of 2.6 bpm for a choice that actually gave 9.0. A tool that scores
something other than what ships is worse than no tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ppg  # noqa: E402

CAPTURES = Path(__file__).resolve().parent.parent / "captures"

# Heart rate from a watch at the moment of capture, by capture timestamp.
REFERENCE: dict[str, float] = {
    "20260928-181145-581": 95.0,
    "20260928-181426-882": 93.0,
    "20260928-181617-820": 92.5,
    # A second phone, delivering 12-15 fps rather than 30. CHROM cannot run
    # below about 13 fps, so some of these have three readings, not six.
    "20260928-182458-717": 88.0,
    "20260928-183313-960": 88.0,
    "20260928-183411-578": 95.0,
    # A third phone, 20.6 fps. The live figure on screen read 6 of 6 during the
    # capture and the finished capture scores 4 of 6: agreement held from 10 to
    # 24 seconds and fell away in the last few, while the rate itself settled at
    # 14 seconds and never moved again.
    "20260928-184035-062": 93.0,
    # Confirmed correct by the watch despite only two of six readings agreeing:
    # the first capture where a low agreement count went with a right answer.
    # Taken as 95 because the watch shows whole numbers and the app said 95.3,
    # so the error here is a floor, not a measurement.
    "20260928-184433-251": 95.0,
    "20260928-185108-060": 91.5,
}

# References taken from "the app was right" rather than read off the watch
# independently. Their error is bounded by how the number was obtained, so they
# flatter the mean and are scored separately.
SELF_CONFIRMED = {"20260928-184433-251", "20260928-185108-060"}


def main():
    scored: list[tuple[str, float, float, int, int]] = []

    print(f"{'capture':22s} {'source':>6s} {'fps':>5s} {'watch':>6s} "
          f"{'app':>7s} {'error':>7s}  agreement")
    for path in sorted(CAPTURES.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        stamp = path.stem.rsplit("-", 1)[0]
        source = blob.get("source", "face")
        times = np.asarray(blob["times"], dtype=float)
        count = int(blob["region_count"])
        values = np.asarray(blob["values"], dtype=float).reshape(len(times), count, 3)

        result = ppg.estimate(times, values, source=source)
        if result is None:
            print(f"{stamp:22s} {source:>6s}   nothing usable")
            continue

        truth = REFERENCE.get(stamp)
        error = "" if truth is None else f"{result.bpm - truth:+7.1f}"
        print(f"{stamp:22s} {source:>6s} {result.fps:5.1f} "
              f"{'-' if truth is None else f'{truth:6.1f}'} {result.bpm:7.1f} "
              f"{error:>7s}  {result.agreement}/{result.candidates}")
        if truth is not None:
            scored.append(
                (stamp, truth, result.bpm, result.agreement, result.candidates)
            )

    if not scored:
        print("\nNo capture has a reference reading yet -- add one to REFERENCE.")
        return

    errors = np.array([abs(bpm - truth) for _, truth, bpm, _, _ in scored])
    independent = np.array([
        abs(bpm - truth) for stamp, truth, bpm, _, _ in scored
        if stamp not in SELF_CONFIRMED
    ])
    print(f"\nOver {len(scored)} captures with a reference: "
          f"mean absolute error {errors.mean():.1f} bpm, worst {errors.max():.1f}")
    print(f"  of those, {len(independent)} had the watch read out rather than "
          f"confirmed against the app: "
          f"mean {independent.mean():.1f} bpm, worst {independent.max():.1f}")

    # Whether the agreement count is worth showing depends on this staying true
    # as captures accumulate. If the columns stop separating, it is not a
    # quality indicator any more and the app should stop presenting it as one.
    print("\nerror against how many readings agreed:")
    for _, truth, bpm, agreed, total in sorted(scored, key=lambda s: -s[3] / s[4]):
        print(f"  {agreed}/{total}  ({agreed / total:4.0%})   "
              f"error {abs(bpm - truth):5.1f} bpm")


if __name__ == "__main__":
    main()
