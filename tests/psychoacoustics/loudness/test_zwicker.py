#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Zwicker loudness (ISO 532-1:2017) conformance tests.

Expected values and tolerances come from the results workbooks of the
freely downloadable ISO 532-1:2017 electronic attachment (Annex B),
extracted to ``tests/data/iso532_1/iso532_1_annexB_expected.json``.
Synthetic test signals are regenerated per their normative Annex B
descriptions (pure tones at stated levels; 100 ms of silence before and
after, as noted in the workbooks). The recorded technical signals (B.5)
ship in ``tests/data/iso532_1/`` (see its README for provenance and ISO
attribution); ``ISO532_1_TESTDATA`` overrides the location.
"""

import json
import os
import pathlib

import numpy as np
import pytest

from phonometry import psychoacoustics

FS = 48000
# Single ISO 532-1 data root, honouring the ISO532_1_TESTDATA override so the
# presence gate, the expected-values JSON and the Annex B.5 recordings all
# resolve from the same directory. Defaults to the in-repo fixtures.
DATA = pathlib.Path(os.environ.get(
    "ISO532_1_TESTDATA", str(pathlib.Path(__file__).parents[2] / "data" / "iso532_1")
))
# The ISO 532-1 Annex B validation data may be absent (its README documents a
# removal policy); the tests that need it then skip rather than crash on import.
_ISO_DATA_PRESENT = (DATA / "iso532_1_annexB_expected.json").is_file()
EXPECTED = (
    json.loads((DATA / "iso532_1_annexB_expected.json").read_text())
    if _ISO_DATA_PRESENT
    else {}
)
requires_iso_data = pytest.mark.skipif(
    not _ISO_DATA_PRESENT,
    reason="ISO 532-1 Annex B data absent (see tests/data/iso532_1/README.md)",
)


def test_iso_data_present_in_repo() -> None:
    """The in-repo Annex B fixtures must be found by the default DATA path.

    This assertion (not a skip) guards the strongest oracle of the
    psychoacoustics suite: a broken relative path after a test-tree
    reorganization once made all 21 Annex B validation tests silently skip
    (they resolve their data through the ``requires_iso_data`` guard). If
    this test fails, fix ``DATA`` above -- do not delete the test. The
    ``ISO532_1_TESTDATA`` override still relocates the data explicitly.
    """
    if "ISO532_1_TESTDATA" in os.environ:
        pytest.skip("external ISO532_1_TESTDATA override in use")
    assert _ISO_DATA_PRESENT, (
        f"ISO 532-1 Annex B fixtures not found at {DATA}; the 21 Annex B "
        "validation tests would silently skip"
    )


def _tone(freq: float, level_db: float, seconds: float = 1.0, pad_ms: float = 100.0) -> np.ndarray:
    """Pure tone at an SPL (dB re 20 uPa) with leading/trailing silence."""
    t = np.arange(int(FS * seconds)) / FS
    amp = np.sqrt(2.0) * 2e-5 * 10 ** (level_db / 20)
    x = amp * np.sin(2 * np.pi * freq * t)
    pad = np.zeros(int(FS * pad_ms / 1000))
    return np.concatenate([pad, x, pad])


def _levels_from_file() -> np.ndarray:
    levels = []
    for line in (DATA / "iso532_1_test_signal_1_levels.txt").read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            levels.append(float(line.split(":")[1]))
    return np.array(levels)


# ---------------------------------------------------------------------------
# Annex B.2 - stationary loudness from one-third-octave levels
# ---------------------------------------------------------------------------

@requires_iso_data
def test_annex_b2_stationary_from_levels() -> None:
    exp = EXPECTED["Test signal 1.txt"]
    res = psychoacoustics.loudness_zwicker_from_spectrum(
        _levels_from_file(), field="free"
    )
    assert exp["Nmin"] <= res.loudness <= exp["Nmax"]
    assert res.loudness == pytest.approx(exp["N"], rel=0.001)


@requires_iso_data
def test_specific_loudness_shape_and_integral() -> None:
    res = psychoacoustics.loudness_zwicker_from_spectrum(_levels_from_file())
    assert res.specific.shape == (240,)
    # N equals the integral of N'(z) dz (dz = 0.1 Bark) up to the slope
    # method's fractional band-edge handling
    assert float(np.sum(res.specific) * 0.1) == pytest.approx(res.loudness, rel=0.02)


# ---------------------------------------------------------------------------
# Annex B.3 - stationary loudness from synthetic signals (regenerated per
# the normative descriptions: pure tones at the stated levels)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("case", "freq", "level"),
    [("Test signal 2", 250.0, 80.0), ("Test signal 3", 1000.0, 60.0),
     ("Test signal 4", 4000.0, 40.0)],
)
@requires_iso_data
def test_annex_b3_stationary_tones(case: str, freq: float, level: float) -> None:
    exp = EXPECTED[case]
    x = _tone(freq, level, seconds=2.0, pad_ms=0.0)
    res = psychoacoustics.loudness_zwicker(x, FS, stationary=True)
    assert exp["Nmin"] <= res.loudness <= exp["Nmax"], (
        f"{case}: N={res.loudness:.4f} outside [{exp['Nmin']}, {exp['Nmax']}]"
    )


def _pink_noise(seconds: float, level_db: float, seed: int = 5321) -> np.ndarray:
    """Deterministic pink noise at an overall SPL (dB re 20 uPa)."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    spec = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    spec[1:] /= np.sqrt(freqs[1:])  # 1/f power shape
    spec[0] = 0.0
    x = np.fft.irfft(spec, n=n)
    return np.asarray(x / np.sqrt(np.mean(x**2)) * 2e-5 * 10 ** (level_db / 20))


@requires_iso_data
def test_annex_b3_pink_noise_bounds() -> None:
    """Annex B.3 Test signal 5 (pink noise, 60 dB overall, stationary).

    The workbook tabulates N = 10.4978 with the tolerance band
    [Nmin, Nmax] = [9.97291, 11.02269]. The ISO WAV is not regenerable
    bit-exactly (stochastic), so a deterministic pink-noise realization at
    the stated overall level is checked against the workbook bounds only.
    """
    exp = EXPECTED["Test signal 5"]
    res = psychoacoustics.loudness_zwicker(
        _pink_noise(5.0, 60.0), FS, stationary=True
    )
    assert exp["Nmin"] <= res.loudness <= exp["Nmax"], (
        f"pink noise: N={res.loudness:.4f} outside [{exp['Nmin']}, {exp['Nmax']}]"
    )


def test_stationary_time_skip() -> None:
    """Annex B.1 TimeSkip: the stationary calculation "shall start from
    0,2 s" when validating against the official Annex B WAVs, so their
    leading silence and the filterbank onset do not dilute the mean square.
    A tone with 150 ms of leading silence (like the shipped WAVs) reads low
    without the skip and recovers the steady-tone value with it; invalid
    skips raise."""
    lead = np.zeros(int(FS * 0.15))
    x = np.concatenate([lead, _tone(1000.0, 60.0, seconds=1.0, pad_ms=0.0)])
    n_all = psychoacoustics.loudness_zwicker(x, FS, stationary=True).loudness
    n_skip = psychoacoustics.loudness_zwicker(
        x, FS, stationary=True, time_skip=0.2
    ).loudness
    n_steady = psychoacoustics.loudness_zwicker(
        _tone(1000.0, 60.0, seconds=1.0, pad_ms=0.0), FS, stationary=True
    ).loudness
    assert n_skip > n_all  # leading silence no longer dilutes the mean square
    # ~1 % residual vs the from-onset tone: that one still contains the
    # broadband onset click the skip removes.
    assert n_skip == pytest.approx(n_steady, rel=0.02)
    with pytest.raises(ValueError, match="time_skip"):
        psychoacoustics.loudness_zwicker(
            x, FS, stationary=True, time_skip=-0.1
        )
    with pytest.raises(ValueError, match="time_skip"):
        psychoacoustics.loudness_zwicker(x, FS, stationary=True, time_skip=2.0)


def test_1khz_60db_anchor() -> None:
    """Definitional anchor: a 1 kHz tone at 40 phon is 1 sone; 60 dB -> 4 sone
    (each 10 phon doubles loudness)."""
    res = psychoacoustics.loudness_zwicker(
        _tone(1000.0, 60.0, seconds=2.0, pad_ms=0.0), FS, stationary=True
    )
    # Measured +0.97 % vs the 4.0 sone definitional value (the +0.5-0.8 %
    # Zwicker stationary bias); 0.02 keeps ~2x headroom (was 0.05).
    assert res.loudness == pytest.approx(4.0, rel=0.02)
    assert res.loudness_level == pytest.approx(60.0, abs=0.5)


# ---------------------------------------------------------------------------
# Annex B.4 - time-varying loudness (synthetic, regenerated)
# ---------------------------------------------------------------------------

def _read_wav_fullscale(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit WAV as first-channel samples in [-1, 1) with its rate.

    WAV/RIFF is little-endian, so the sample dtype is fixed to ``<i2`` rather
    than the host-native ``int16``; a multi-channel file keeps only channel 0.
    """
    import wave

    with wave.open(path) as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels)[:, 0]
    return raw.astype(np.float64) / 32768.0, int(fs)


def _read_wav_pa(path: pathlib.Path, peak_rms_db: float) -> tuple[np.ndarray, int]:
    """Read a 16-bit ISO test WAV and calibrate its peak short-time RMS to
    the level stated in Annex B (the WAVs carry no absolute scale)."""
    x, fs = _read_wav_fullscale(str(path))
    win = max(1, int(fs * 0.002))
    sq = np.convolve(x**2, np.ones(win) / win, mode="same")
    peak_rms = float(np.sqrt(sq.max()))
    target = 2e-5 * 10 ** (peak_rms_db / 20)
    return x * (target / peak_rms), fs


def _level_ramp_tone(freq: float) -> np.ndarray:
    """Annex B.4 Test signals 6-8: a tone whose SPL ramps 30 dB -> 80 dB.

    Parameters measured from the official electronic-attachment WAVs
    (Annex B.1 calibration, full-scale sine = 100 dB SPL): 0.1 s of leading
    silence, a 10 s tone whose level ramps linearly in dB at 5 dB/s from
    30 dB to 80 dB, then 0.5 s of trailing silence (10.6 s total). A short
    raised-cosine fade-out (5 ms) replaces the official WAVs' cut, whose
    end-of-file transient otherwise inflates Nmax (the loudness maximum
    falls exactly at the ramp top).
    """
    n_act = 10 * FS
    tau = np.arange(n_act) / FS
    amp = np.sqrt(2.0) * 2e-5 * 10 ** ((30.0 + 5.0 * tau) / 20.0)
    act = amp * np.sin(2 * np.pi * freq * tau)
    n_fade = int(0.005 * FS)
    act[-n_fade:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / n_fade))
    return np.concatenate(
        [np.zeros(int(0.1 * FS)), act, np.zeros(int(0.5 * FS))]
    )


@pytest.mark.parametrize(
    ("case", "freq"),
    [("Test signal 6", 250.0), ("Test signal 7", 1000.0),
     ("Test signal 8", 4000.0)],
)
@requires_iso_data
def test_annex_b4_level_ramp_tones(case: str, freq: float) -> None:
    """Annex B.4 Test signals 6-8 (tone level ramps 30 -> 80 dB, 5 dB/s).

    Tolerances calibrated against the official attachment WAVs: the module
    reproduces the workbook Nmax on the official WAVs to < 0.01 % for all
    three signals; this regenerated synthesis lands within +0.3 % of the
    workbook Nmax (end-of-ramp handling) and within -0.8 % on N5, so the
    pins are 0.6 % and 2 % with about 2x headroom.
    """
    exp = EXPECTED[case]
    res = psychoacoustics.loudness_zwicker(
        _level_ramp_tone(freq), FS, stationary=False
    )
    assert res.loudness == pytest.approx(exp["Nmax"], rel=6e-3), (
        f"{case}: Nmax={res.loudness:.4f} vs workbook {exp['Nmax']}"
    )
    assert res.n5 == pytest.approx(exp["N5"], rel=0.02), (
        f"{case}: N5={res.n5:.4f} vs workbook {exp['N5']}"
    )


@requires_iso_data
def test_annex_b4_pink_noise_ramp() -> None:
    """Annex B.4 Test signal 9 (pink-noise level ramp, 5 dB/s over 10 s).

    The workbook titles the ramp "0 dB - 50 dB"; the WAV's own Annex B.1
    calibration puts the measured overall SPL at 25.85 + 5 t dB (the title
    tracks the generator gain, not the calibrated SPL). The official WAV
    reproduces the workbook Nmax/N5 to < 0.01 % / -0.5 % through the
    module; a regenerated pink realization scatters by up to ~10 % around
    the workbook values (Nmax is dominated by the last few hundred
    milliseconds of one noise realization), so this deterministic-seed
    synthesis is pinned at the realization-honest 10 %.
    """
    exp = EXPECTED["Test signal 9"]
    n_act = 10 * FS
    tau = np.arange(n_act) / FS
    rng = np.random.default_rng(1)
    spec = np.fft.rfft(rng.standard_normal(n_act))
    freqs = np.fft.rfftfreq(n_act, d=1.0 / FS)
    spec[1:] /= np.sqrt(freqs[1:])
    spec[0] = 0.0
    x = np.fft.irfft(spec, n=n_act)
    x /= np.sqrt(np.mean(x**2))
    act = x * 2e-5 * 10 ** ((25.85 + 5.0 * tau) / 20.0)
    n_fade = int(0.005 * FS)
    act[-n_fade:] *= 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / n_fade))
    sig = np.concatenate([np.zeros(int(0.1 * FS)), act, np.zeros(int(0.5 * FS))])
    res = psychoacoustics.loudness_zwicker(sig, FS, stationary=False)
    assert res.loudness == pytest.approx(exp["Nmax"], rel=0.10)
    assert res.n5 == pytest.approx(exp["N5"], rel=0.10)


@pytest.mark.parametrize(
    ("case", "num", "level"),
    [("Test signal 10", 10, 70.0), ("Test signal 11", 11, 70.0),
     ("Test signal 12", 12, 70.0), ("Test signal 13", 13, 80.0)],
)
@requires_iso_data
def test_annex_b4_tone_pulses(case: str, num: int, level: float) -> None:
    """The normative 1 kHz tone-pulse WAVs (Annex B.4): the prose only
    states duration and peak rms level; the ramp shapes exist only in the
    signals themselves, so the small originals are used directly. The
    normative acceptance criterion is the per-sample tolerance band around
    the published N(t) trace (workbook columns Nmin/Nmax), which is what
    this test enforces sample by sample, plus the Nmax header value.
    (The workbook header N5 values are not reproducible from their own
    published traces with the Annex A percentile formula and are therefore
    not asserted.)"""
    exp = EXPECTED[case]
    x, fs = _read_wav_pa(DATA / f"iso532_1_test_signal_{num}.wav", level)
    res = psychoacoustics.loudness_zwicker(x, fs, stationary=False)
    assert res.n5 is not None
    assert res.loudness_vs_time is not None
    # Nmax reproduces the workbook header to < 0.01 %; 1e-3 locks that in
    # (was 0.05, ~700x looser than the achieved accuracy).
    assert res.loudness == pytest.approx(exp["Nmax"], rel=1e-3), (
        f"{case}: Nmax={res.loudness:.4f} vs expected {exp['Nmax']}"
    )
    ref = np.load(DATA / "iso532_1_annexB4_traces.npz")[f"signal_{num}"]
    n = min(ref.shape[0], res.loudness_vs_time.shape[0])
    ours, lo, hi = res.loudness_vs_time[:n], ref[:n, 2], ref[:n, 3]
    inside = np.mean((ours >= lo) & (ours <= hi))
    assert inside >= 0.99, f"{case}: only {inside:.1%} of N(t) inside the tolerance band"


@requires_iso_data
def test_n5_n10_use_full_rate_series(monkeypatch) -> None:
    """N5/N10 must come from the full-rate 2000 Hz weighted-loudness series,
    not the 4x-decimated 500 Hz output trace. Decimation keeps only one of
    four phases, spreading N5 by up to ~3 % across the phases (Annex B
    TS 10, 0.7513..0.7752); the full-rate percentile (0.7628) is
    phase-unambiguous and sits inside that envelope. The 500 Hz
    ``loudness_vs_time`` output is unchanged (public contract)."""
    import sys

    from phonometry.psychoacoustics.loudness.zwicker import _SR_LEVEL, _SR_LOUDNESS

    # The package re-exports the *function* ``loudness_zwicker``, which shadows
    # the module attribute of the same name; go through sys.modules instead.
    L = sys.modules["phonometry.psychoacoustics.loudness.zwicker"]

    seen: dict[int, np.ndarray] = {}
    orig = L._percentile

    def spy(values: np.ndarray, pct: int) -> float:
        seen[pct] = np.asarray(values, dtype=float).copy()
        return orig(values, pct)

    monkeypatch.setattr(L, "_percentile", spy)
    x, fs = _read_wav_pa(DATA / "iso532_1_test_signal_10.wav", 70.0)
    res = psychoacoustics.loudness_zwicker(x, fs, stationary=False)
    assert res.n5 is not None
    assert res.loudness_vs_time is not None

    full = seen[5]
    dec = _SR_LEVEL // _SR_LOUDNESS
    # (a) computed on the un-decimated series (~dec x longer than the trace).
    assert full.size >= (dec - 0.5) * res.loudness_vs_time.size
    # (b) the reported N5 is the phase-unambiguous full-rate percentile, and
    #     lies strictly inside the envelope of the four decimation phases.
    phase_n5 = [orig(full[p::dec], 5) for p in range(dec)]
    assert min(phase_n5) < res.n5 < max(phase_n5)
    assert res.n5 == pytest.approx(orig(full, 5))


def test_time_varying_outputs() -> None:
    x = _tone(1000.0, 70.0, seconds=0.5)
    res = psychoacoustics.loudness_zwicker(x, FS)
    assert res.time is not None
    assert res.loudness_vs_time is not None
    assert res.time.shape == res.loudness_vs_time.shape
    assert res.n5 is not None
    assert res.n10 is not None
    assert res.n5 >= res.n10


# ---------------------------------------------------------------------------
# Full validation against the recorded ISO signals (Annex B.5), shipped in
# tests/data/iso532_1 (see its README) or the ISO532_1_TESTDATA override.
# ---------------------------------------------------------------------------


@requires_iso_data
@pytest.mark.parametrize("num", list(range(14, 26)))
def test_annex_b5_technical_signals(num: int) -> None:
    import glob

    matches = glob.glob(str(DATA / "Annex B.5" / f"Test signal {num} *.wav"))
    if not matches:
        pytest.skip(f"signal {num} not found")
    # Annex B.1: "0 dB (relative to full scale) shall correspond to a sound
    # pressure level of 100 dB" — a full-scale sine is 100 dB SPL (2 Pa RMS),
    # so one full-scale unit is 2*sqrt(2) Pa peak.
    fullscale, fs = _read_wav_fullscale(matches[0])
    x = fullscale * (2.0 * np.sqrt(2.0))
    exp = EXPECTED[f"Test signal {num}"]
    # Each B.5 signal is validated in the sound field its ISO results
    # workbook was computed in; signal 15 (vehicle interior) is diffuse.
    res = psychoacoustics.loudness_zwicker(
        np.asarray(x, dtype=np.float64), int(fs), field=exp["field"]
    )
    # Nmax reproduces the ISO results workbook to < 0.01 % for all twelve
    # signals; 1e-3 locks that in. N5 is a percentile of the loudness-vs-time
    # trace, phase-sensitive on impulsive signals (e.g. the machine gun),
    # so it is held to the clause 6.1 +-5 % tolerance with a small margin.
    assert res.loudness == pytest.approx(exp["Nmax"], rel=1e-3)
    assert res.n5 == pytest.approx(exp["N5"], rel=0.06)


# ---------------------------------------------------------------------------
# Validation and API behavior
# ---------------------------------------------------------------------------

def test_invalid_inputs() -> None:
    ten_bands = np.zeros(10)
    with pytest.raises(ValueError, match="28"):
        psychoacoustics.loudness_zwicker_from_spectrum(ten_bands)
    levels = np.full(28, 60.0)
    with pytest.raises(ValueError, match="field"):
        psychoacoustics.loudness_zwicker_from_spectrum(
            levels, field="reverberant"
        )
    x = np.ones(1000)
    with pytest.raises(ValueError, match="fs"):
        psychoacoustics.loudness_zwicker(x, 0)


def test_non_finite_inputs_rejected() -> None:
    levels = np.full(28, 60.0)
    levels[5] = np.nan
    with pytest.raises(ValueError, match="finite"):
        psychoacoustics.loudness_zwicker_from_spectrum(levels)
    x = np.ones(1000)
    x[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        psychoacoustics.loudness_zwicker(x, FS)


def test_pathological_resampling_ratio_rejected() -> None:
    """gcd(48000, 44101) = 1 would demand a 48000/44101 polyphase filter;
    reject instead of hanging."""
    x = np.ones(1000)
    with pytest.raises(ValueError, match="resampl"):
        psychoacoustics.loudness_zwicker(x, 44101)


def test_diffuse_field_differs() -> None:
    levels = np.full(28, 70.0)
    free = psychoacoustics.loudness_zwicker_from_spectrum(levels, field="free")
    diffuse = psychoacoustics.loudness_zwicker_from_spectrum(
        levels, field="diffuse"
    )
    assert isinstance(free, psychoacoustics.ZwickerLoudness)
    assert free.loudness != diffuse.loudness


def test_minimal_length_validation() -> None:
    """Signals shorter than one 500 Hz output sample raise cleanly instead
    of crashing on an empty percentile buffer."""
    too_short = np.ones(48)
    with pytest.raises(ValueError, match="too short"):
        psychoacoustics.loudness_zwicker(too_short, FS)
    res = psychoacoustics.loudness_zwicker(
        np.ones(96 * 4), FS
    )  # exactly a few output samples
    assert res.loudness >= 0.0


def test_specific_pattern_matches_reported_max() -> None:
    """The returned pattern is taken at the same decimated instant as the
    reported Nmax. For a steady tone the temporal weighting converges, so
    the pattern integral must match Nmax there (for transients the
    instantaneous pattern legitimately exceeds the weighted maximum)."""
    res = psychoacoustics.loudness_zwicker(
        _tone(1000.0, 70.0, seconds=2.0, pad_ms=0.0), FS
    )
    assert float(np.sum(res.specific) * 0.1) == pytest.approx(res.loudness, rel=0.03)
