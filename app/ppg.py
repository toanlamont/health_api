"""Heart-rate estimation from per-frame colour means.

The client sends the average R, G, B of one or more skin regions per frame,
plus the capture time of each frame. Averaging on the device is not a loss:
every rPPG method starts by spatially averaging skin pixels, and sending the
means instead of video avoids the lossy compression that would erase a pulse
worth a fraction of a percent of brightness.

None of the estimation is done here. The fingertip path is neurokit2's Elgendi
pipeline, the standard for contact PPG; the face path is rPPG-Toolbox's
reference POS and CHROM together with the toolbox's own periodogram rate and
SNR. This module only reshapes the payload into what those libraries expect,
resamples onto the uniform grid they assume, and decides whether the result is
strong enough to report.

References:
  Wang et al. (2017), Algorithmic principles of remote PPG, IEEE TBME 64(7).
  de Haan & Jeanne (2013), Robust pulse rate from chrominance-based rPPG.
  Elgendi et al. (2013), Systolic peak detection in acceleration photoplethysmograms.
  Liu et al. (2023), rPPG-Toolbox, NeurIPS Datasets and Benchmarks.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal

# rPPG-Toolbox ships as a repository rather than a package, so it is vendored
# next to the app and put on the path here.
_VENDOR = Path(__file__).resolve().parent.parent / "vendor_rppg"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# NumPy 2 removed the `np.mat` alias that the toolbox still calls. Restoring it
# is a compatibility shim, not a change to the algorithms.
if not hasattr(np, "mat"):
    np.mat = np.asmatrix  # type: ignore[attr-defined]

import neurokit2 as nk  # noqa: E402
from evaluation.post_process import _calculate_fft_hr, _calculate_SNR  # noqa: E402
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN  # noqa: E402
from unsupervised_methods.methods.POS_WANG import POS_WANG  # noqa: E402

# The band the vendored implementations were published with: POS_WANG filters
# 0.75-3 Hz internally and CHROM 0.7-2.5 Hz. Asking for a wider range would
# report rates the filters have already removed.
LOW_HZ = 0.75
HIGH_HZ = 3.0
MIN_BPM = LOW_HZ * 60.0   # 45
MAX_BPM = HIGH_HZ * 60.0  # 180

# There is deliberately no accept/reject threshold here.
#
# An earlier version refused to report a rate below a fixed SNR, with the
# threshold calibrated against synthetic signals written in this same
# repository -- which meant the gate was only ever tested against my own idea
# of what a pulse looks like. It passed that test and then rejected every real
# capture, including ones whose rate may well have been right.
#
# Neither rPPG-Toolbox nor neurokit2 has such a gate: the toolbox is a
# benchmark that always emits a rate and scores it against ground truth. The
# SNR is reported alongside the rate so it can be judged against a reference
# measurement, which is the only thing that can actually settle whether a
# capture is trustworthy.

# Slowest capture each path can be trusted on. The face methods only need the
# 3 Hz top of the band to clear Nyquist; neurokit2's Elgendi cleaner filters up
# to 8 Hz and warns then aliases below 16 Hz. A phone delivers around 30 either
# way, so this only ever fires on a capture that went wrong -- and refusing
# beats reporting a rate read off aliased data.
MIN_FPS = {"finger": 17.0, "face": 8.0}


# Two readings of the same capture count as agreeing when they are this close.
# A 30-second capture resolves frequency to about 2 bpm, so this is a couple of
# bins of slack. It decides what the agreement count reports, nothing else --
# no capture is refused on the strength of it.
AGREEMENT_BPM = 5.0

log = logging.getLogger("heartbeat.ppg")


@dataclass
class Estimate:
    bpm: float
    snr_db: float
    confidence: float
    method: str
    region: int
    fps: float
    amplitude: float
    # How many of the independent readings landed on this rate, out of how many
    # were taken.
    #
    # Across eight captures scored against a watch this is one-sided evidence,
    # no more. Every reading agreeing has so far meant an error under 4 bpm.
    # Anything less means nothing on its own: the worst capture in the set
    # (11.2 bpm out) had one reading in six agreeing, and so did one of the
    # best (0.3 bpm out, two in six). An earlier version of this comment
    # claimed the two separated cleanly; the counterexample arrived within the
    # hour. Report the count, do not read reliability into a low one, and
    # re-check it with tools/compare_captures.py as captures accumulate.
    agreement: int = 0
    candidates: int = 0
    # neurokit2's template-match SQI, on the fingertip path only. None on a face.
    quality: float | None = None


def resample_uniform(
    times: np.ndarray, values: np.ndarray, fps: float | None = None
) -> tuple[np.ndarray, float]:
    """Interpolate irregularly delivered frames onto an even grid.

    Camera frames do not arrive on a fixed cadence, and every method below
    assumes they do.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]

    order = np.argsort(times)
    times, values = times[order], values[order]

    span = float(times[-1] - times[0])
    if span <= 0:
        raise ValueError("timestamps span no time")
    if fps is None:
        fps = (len(times) - 1) / span

    count = int(np.floor(span * fps))
    if count < 2:
        raise ValueError("not enough samples after resampling")

    grid = times[0] + np.arange(count) / fps
    out = np.empty((count, values.shape[1]), dtype=float)
    for channel in range(values.shape[1]):
        out[:, channel] = np.interp(grid, times, values[:, channel])
    return out, float(fps)


def _rate_and_snr(bvp: np.ndarray, fps: float) -> tuple[float, float] | None:
    """Pulse rate and SNR, both from rPPG-Toolbox's own post-processing."""
    if bvp.size < 16 or not np.all(np.isfinite(bvp)) or float(np.std(bvp)) <= 1e-12:
        return None
    bpm = float(_calculate_fft_hr(bvp, fs=fps, low_pass=LOW_HZ, high_pass=HIGH_HZ))
    if not MIN_BPM <= bpm <= MAX_BPM:
        return None
    snr = float(_calculate_SNR(bvp, bpm, fs=fps, low_pass=LOW_HZ, high_pass=HIGH_HZ))
    if not np.isfinite(snr):
        return None
    return bpm, snr


def _face_candidates(rgb: np.ndarray, fps: float):
    """POS and CHROM, exactly as vendored.

    Both take video and average each frame themselves, so the colour means are
    handed over as 1x1 frames: the mean of a single-pixel frame is that pixel.
    """
    frames = np.ascontiguousarray(rgb, dtype=float).reshape(len(rgb), 1, 1, 3)
    for name, method in (("pos", POS_WANG), ("chrom", CHROME_DEHAAN)):
        try:
            bvp = np.asarray(method(frames, fps), dtype=float).ravel()
        except (ValueError, FloatingPointError, ZeroDivisionError) as error:
            # Not silently: a method dropping out halves the evidence behind
            # the answer, and the agreement count then means something
            # different. CHROM in particular cannot run below about 13 fps --
            # it filters inside a window of ceil(1.6 * fps) samples and
            # filtfilt needs more than 21 of them.
            log.warning("%s could not run at %.1f fps: %s", name, fps, error)
            continue
        yield name, bvp, None

    # The raw green channel was tried here as a third voice. Across three
    # captures scored against a watch it lowered the mean error from 2.6 to
    # 2.1 bpm -- but that is half a beat on three captures, while the cost is
    # concrete: green does not cancel a brightness flicker the way POS and
    # CHROM do, so mains or screen flicker starts being read as a pulse. Weak
    # evidence, known cost, left out.


def _finger_candidates(rgb: np.ndarray, fps: float):
    """neurokit2's Elgendi pipeline on the raw channels.

    Through a fingertip the light is transmitted, not reflected, so the channel
    means are a contact PPG and the chrominance methods above do not apply --
    they exist to cancel exactly the common-mode intensity change that a
    fingertip signal consists of.
    """
    for name, index in (("red", 0), ("green", 1)):
        # More blood means more absorption, so the camera sees the trace
        # inverted relative to the clinical PPG that Elgendi's detector expects.
        series = -np.asarray(rgb[:, index], dtype=float)
        if float(np.std(series)) <= 1e-12:
            continue
        try:
            cleaned = np.asarray(
                nk.ppg_clean(series, sampling_rate=fps, method="elgendi"), dtype=float
            )
            peaks = nk.ppg_findpeaks(
                cleaned, sampling_rate=fps, method="elgendi"
            )["PPG_Peaks"]
        except Exception:
            continue
        if not np.all(np.isfinite(cleaned)):
            continue

        quality = None
        if len(peaks) >= 4:
            try:
                sqi = nk.ppg_quality(
                    cleaned, peaks=peaks, sampling_rate=fps, method="templatematch"
                )
                quality = float(np.nanmean(np.asarray(sqi, dtype=float)))
            except Exception:
                quality = None
            if quality is not None and not np.isfinite(quality):
                quality = None
        yield name, cleaned, quality


def _confidence(snr_db: float, quality: float | None) -> float:
    """A 0..1 version of the SNR, for the progress bar and nothing else.

    This is presentation, not evidence. The number to judge a capture by is
    snr_db, next to a reference heart rate.
    """
    score = float(np.clip((snr_db + 10.0) / 20.0, 0.0, 1.0))
    if quality is not None:
        score *= float(np.clip(quality, 0.0, 1.0))
    return score


def estimate(
    times, regions, source: str = "finger", min_seconds: float = 4.0
) -> Estimate | None:
    """Best heart-rate estimate across every region and extraction method.

    `regions` is (frames, regions, 3). Returns None when the capture is too
    short or nothing in it can be turned into a pulse signal at all.
    """
    times = np.asarray(times, dtype=float)
    regions = np.asarray(regions, dtype=float)
    if regions.ndim == 2:
        regions = regions[:, None, :]
    if regions.ndim != 3 or regions.shape[2] != 3:
        raise ValueError("regions must be shaped (frames, regions, 3)")
    if len(times) != regions.shape[0]:
        raise ValueError("times and regions disagree on frame count")
    if len(times) < 2:
        return None
    if float(times[-1] - times[0]) < min_seconds:
        return None

    flat, fps = resample_uniform(times, regions.reshape(len(times), -1))
    if fps < MIN_FPS.get(source, MIN_FPS["face"]):
        return None
    resampled = flat.reshape(flat.shape[0], -1, 3)

    candidates = _finger_candidates if source == "finger" else _face_candidates

    # Every region read by every method, as independent readings of one rate.
    readings: list[tuple[str, int, float, float, float, float | None]] = []
    for index in range(resampled.shape[1]):
        rgb = resampled[:, index, :]
        if not np.all(np.isfinite(rgb)):
            continue
        for name, bvp, quality in candidates(rgb, fps):
            measured = _rate_and_snr(bvp, fps)
            if measured is None:
                continue
            bpm, snr = measured
            readings.append(
                (name, index, bpm, snr, float(bvp.max() - bvp.min()), quality)
            )

    if not readings:
        return None

    # The largest group of readings that agree with each other, rather than the
    # median of all of them. Scored against a watch on six captures this had a
    # mean error of 2.6 bpm and a worst case of 4.6; the median of everything
    # came to 5.7 with a worst case of 19, and taking the best SNR came to 14.9
    # with a worst case of 43. The median of a scattered set lands between the
    # two middle readings, which on one capture meant reporting 69 bpm when no
    # method had said anything near it. tools/compare_captures.py rescores all
    # of these whenever a capture with a reference reading is added.
    rates = np.array([r[2] for r in readings])
    middle = float(np.median(rates))

    groups = [
        [r for r in readings if abs(r[2] - centre) <= AGREEMENT_BPM]
        for centre in rates
    ]
    largest = max(len(g) for g in groups)
    # Ties go to the group nearest the middle of all the readings, never to the
    # one with the best SNR: when nothing agrees, every group is a single
    # reading and that tie-break is just "best SNR" again, which on one capture
    # reached for 45 bpm against a watch reading 88.
    agreeing = min(
        (g for g in groups if len(g) == largest),
        key=lambda g: abs(float(np.median([r[2] for r in g])) - middle),
    )
    bpm = float(np.median([r[2] for r in agreeing]))
    # Describe the capture by its strongest agreeing reading, so method and
    # region point at something that actually supports the answer.
    lead = max(agreeing, key=lambda r: r[3]) if agreeing else max(readings, key=lambda r: r[3])
    snr = float(np.median([r[3] for r in agreeing])) if agreeing else lead[3]
    quality = lead[5]

    return Estimate(
        bpm=bpm,
        snr_db=snr,
        confidence=_confidence(snr, quality),
        method=lead[0],
        region=lead[1],
        fps=fps,
        amplitude=lead[4],
        agreement=len(agreeing),
        candidates=len(readings),
        quality=quality,
    )
