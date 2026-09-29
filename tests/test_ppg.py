import numpy as np
import pytest

from app import ppg

BASE = np.array([160.0, 120.0, 110.0])

# The two modes see light that has travelled different paths, and the pulse
# leaves a different fingerprint on the colour channels in each. Testing the
# face methods against a fingertip signal is meaningless: POS and CHROM exist
# to cancel exactly the common-mode brightness change that a fingertip signal
# is made of, so they correctly return nothing for it.
#
# Through a fingertip the light is transmitted and red is modulated most.
TRANSMISSION = np.array([1.0, 0.45, 0.2])
# Reflected off a face, haemoglobin absorption puts the most modulation in
# green (Wang et al., 2017).
REFLECTION = np.array([0.33, 0.77, 0.53])


def pulse_wave(phase):
    """A real pulse is not a sine: the systolic peak is followed by a dicrotic
    notch, which puts strong energy at the harmonics."""
    return np.sin(phase) + 0.45 * np.sin(2 * phase + 0.6) + 0.2 * np.sin(3 * phase + 1.1)


def synth(
    source="finger",
    bpm=72.0,
    fps=30.0,
    seconds=13.0,
    strength=0.01,
    noise=0.0,
    drift=0.0,
    jitter=0.0,
    regions=1,
    seed=7,
    direction=None,
):
    """Colour means for a capture, shaped (frames, regions, 3)."""
    rng = np.random.default_rng(seed)
    count = int(seconds * fps)
    times = np.arange(count) / fps
    if jitter:
        times = times + rng.uniform(-jitter / 2, jitter / 2, count)
        times = np.maximum.accumulate(np.clip(times, 0, None))

    if direction is None:
        direction = TRANSMISSION if source == "finger" else REFLECTION
    wave = pulse_wave(2 * np.pi * (bpm / 60.0) * times)
    baseline = drift * np.sin(2 * np.pi * 0.15 * times)

    one = BASE * (1.0 + strength * np.outer(wave, direction)) + baseline[:, None]
    out = np.repeat(one[:, None, :], regions, axis=1)
    return times, out + rng.normal(0, noise, out.shape)


# --- the rate itself ---

def test_recovers_a_clean_pulse():
    times, regions = synth(bpm=72)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.bpm == pytest.approx(72, abs=3)


@pytest.mark.parametrize("source", ["finger", "face"])
@pytest.mark.parametrize("bpm", [50, 66, 90, 128])
def test_covers_the_plausible_range(source, bpm):
    times, regions = synth(source=source, bpm=bpm)
    result = ppg.estimate(times, regions, source=source)
    assert result is not None
    assert result.bpm == pytest.approx(bpm, abs=4)


@pytest.mark.parametrize("fps", [20.0, 24.0, 30.0, 60.0])
def test_rate_does_not_depend_on_frame_rate(fps):
    # Assuming a fixed frame rate was the single biggest error in the on-device
    # version: the reported rate scaled with however fast the camera ran.
    times, regions = synth(bpm=78, fps=fps)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.bpm == pytest.approx(78, abs=4)


def test_refuses_a_capture_too_slow_for_the_finger_pipeline():
    # Below 16 Hz the Elgendi cleaner aliases. A rate off aliased data would
    # look perfectly plausible, so nothing is returned at all.
    times, regions = synth(bpm=78, fps=12.0)
    assert ppg.estimate(times, regions, source="finger") is None


def test_tolerates_uneven_frame_delivery():
    times, regions = synth(bpm=84, fps=25, jitter=0.02)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.bpm == pytest.approx(84, abs=5)


def test_survives_drift_and_noise():
    times, regions = synth(bpm=66, noise=0.4, drift=6)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.bpm == pytest.approx(66, abs=5)


def test_does_not_report_a_harmonic_of_a_fast_pulse():
    times, regions = synth(bpm=132)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.bpm == pytest.approx(132, abs=5)


# --- how a capture with nothing in it reports itself ---
#
# A rate always comes back now, so these check the only thing that can be
# checked without a reference measurement: that an empty capture scores far
# worse than a real pulse. They deliberately compare against a pulse measured
# in the same test rather than against a fixed number -- a fixed threshold
# tuned on this file is a test of my own imagination, which is exactly how the
# previous version came to pass every test here and reject every real capture.

def snr_of(source, **kwargs):
    times, regions = synth(source=source, **kwargs)
    result = ppg.estimate(times, regions, source=source)
    return None if result is None else result.snr_db


@pytest.mark.parametrize("source", ["finger", "face"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_noise_scores_far_below_a_real_pulse(source, seed):
    pulse = snr_of(source, bpm=72)
    noise = snr_of(source, strength=0.0, noise=3.0, seed=seed)
    assert pulse is not None
    assert noise is None or noise < pulse - 6.0


@pytest.mark.parametrize("source", ["finger", "face"])
def test_a_drifting_baseline_scores_far_below_a_real_pulse(source):
    pulse = snr_of(source, bpm=72)
    drift = snr_of(source, strength=0.0, noise=0.3, drift=25)
    assert pulse is not None
    assert drift is None or drift < pulse - 6.0


@pytest.mark.parametrize("source", ["finger", "face"])
def test_a_flat_signal_yields_nothing_usable(source):
    times, regions = synth(source=source, strength=0.0)
    result = ppg.estimate(times, regions, source=source)
    assert result is None or result.snr_db < 0.0


def test_a_capture_that_is_too_short_returns_nothing():
    times, regions = synth(seconds=2.0)
    assert ppg.estimate(times, regions, source="finger") is None


def test_face_mode_scores_a_plain_brightness_flicker_poorly():
    # Screen or mains flicker moves all three channels together. POS and CHROM
    # are built to cancel that, so a rate read off it should look much weaker
    # than one read off a real pulse.
    pulse = snr_of("face", bpm=100)
    times, regions = synth(
        source="face", bpm=100, direction=np.array([1.0, 1.0, 1.0]), strength=0.02
    )
    flicker = ppg.estimate(times, regions, source="face")
    assert pulse is not None
    assert flicker is None or flicker.snr_db < pulse - 6.0


# --- what comes back with the rate ---

def test_regions_that_agree_are_reported_as_agreeing():
    times, regions = synth(source="face", bpm=75, regions=4, noise=0.2)
    result = ppg.estimate(times, regions, source="face")
    assert result is not None
    assert result.bpm == pytest.approx(75, abs=5)
    assert result.agreement == result.candidates


def test_regions_that_disagree_are_reported_as_disagreeing():
    # Bury the pulse in every region but one. An earlier version claimed to
    # find the surviving region by picking the best SNR; on real captures that
    # selection chose 102 and 51 bpm on two captures a minute apart, so the
    # promise was never kept. Saying the readings do not agree is worth more
    # than silently trusting one of them.
    times, regions = synth(source="face", bpm=75, regions=4, noise=0.2)
    rng = np.random.default_rng(3)
    for r in range(1, 4):
        regions[:, r, :] += rng.normal(0, 8, (regions.shape[0], 3))

    result = ppg.estimate(times, regions, source="face")
    assert result is not None
    assert result.agreement < result.candidates


def test_agreement_separates_a_clean_capture_from_a_confused_one():
    times, clean = synth(source="face", bpm=75, regions=4, noise=0.2)
    times_bad, confused = synth(source="face", bpm=75, regions=4, noise=0.2, seed=5)
    rng = np.random.default_rng(3)
    for r in range(1, 4):
        confused[:, r, :] += rng.normal(0, 8, (confused.shape[0], 3))

    good = ppg.estimate(times, clean, source="face")
    poor = ppg.estimate(times_bad, confused, source="face")
    assert good is not None and poor is not None
    assert good.agreement > poor.agreement


def test_confidence_tracks_signal_quality():
    times, clean = synth(bpm=70, noise=0.05)
    times_noisy, noisy = synth(bpm=70, noise=2.0, seed=11)

    good = ppg.estimate(times, clean, source="finger")
    poor = ppg.estimate(times_noisy, noisy, source="finger")

    assert good is not None
    assert good.confidence > 0.5
    if poor is not None:
        assert poor.confidence < good.confidence


def test_reports_which_method_and_frame_rate_it_used():
    times, regions = synth(bpm=72, fps=24)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.method in {"red", "green"}
    assert result.fps == pytest.approx(24, abs=1)


def test_face_mode_uses_a_chrominance_method():
    times, regions = synth(source="face", bpm=72)
    result = ppg.estimate(times, regions, source="face")
    assert result is not None
    assert result.method in {"pos", "chrom"}


def test_finger_mode_reports_a_pulse_shape_quality():
    times, regions = synth(bpm=72)
    result = ppg.estimate(times, regions, source="finger")
    assert result is not None
    assert result.quality is not None
    assert 0.0 <= result.quality <= 1.0
