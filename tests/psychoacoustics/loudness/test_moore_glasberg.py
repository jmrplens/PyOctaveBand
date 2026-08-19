#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
ISO 532-2:2017 (Moore-Glasberg) stationary loudness conformance tests.

The primary oracle is Annex B of the standard, which tabulates the calculated
loudness (sone) and loudness level (phon) for a set of reference signals:
sinusoidal tones (B.1), bands of white and pink noise (B.2), multiple-tone
complexes (B.3) and a tone in noise (B.4).  These informative values are
reproduced here to within a tolerance well inside the standard's own expanded
uncertainty (2.8 phon, clause 9).  The definitional anchor of the sone
(clause 3.17) - a 1 kHz tone at 40 dB SPL, binaural, free field = 1.000 sone /
40 phon - is asserted end-to-end; it follows from the tabulated calibration
constant C = 0.0617 sone/Cam (Formula 7) without any tuning.

The remaining checks are internal consistency properties the model implies:
monotonicity in level, the +10 phon ~ x2 sone rule, free/diffuse/eardrum
differences, silence and input validation.
"""

import numpy as np
import pytest

from phonometry import psychoacoustics

FS = 48000


def _white_band(f_lo: float, f_hi: float, spectrum_density_level: float) -> list:
    """White-noise band as components spaced 10 Hz (clause 5.3).

    Level of each component = spectrum density level + 10 dB (10 Hz spacing).
    """
    start = int(np.ceil(f_lo / 10.0) * 10)
    return [
        (float(f), spectrum_density_level + 10.0)
        for f in range(start, int(f_hi) + 1, 10)
    ]


def _pink_band(f_lo: float, f_hi: float, density_at_1k: float) -> list:
    """Pink-noise band as components spaced 10 Hz (clause 5.3).

    Spectrum level falls 3 dB/octave; the value at 1 kHz is ``density_at_1k``.
    """
    start = int(np.ceil(f_lo / 10.0) * 10)
    comps = []
    for f in range(start, int(f_hi) + 1, 10):
        level = density_at_1k - 3.0 * np.log2(f / 1000.0)
        comps.append((float(f), level + 10.0))
    return comps


# --------------------------------------------------------------------------
# Definitional anchor (clause 3.17 / Annex B.1.1)
# --------------------------------------------------------------------------


def test_anchor_1khz_40db_is_one_sone() -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 40.0)]
    )
    # C is the standard's tabulated constant; the anchor emerges near-exactly.
    assert result.loudness == pytest.approx(1.0, abs=0.01)
    assert result.loudness_level == pytest.approx(40.0, abs=0.3)


def test_anchor_single_ear_is_two_thirds_of_binaural() -> None:
    binaural = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 40.0)]
    ).loudness
    monaural = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 40.0)], presentation="monaural"
    ).loudness
    # Diotic binaural loudness is 1.5x the single-ear loudness (clause 8.1).
    assert binaural / monaural == pytest.approx(1.5, abs=0.02)


def test_calibration_constant_is_the_tabulated_c() -> None:
    # ISO 532-2 Formula (7) tabulates C = 0.0617 sone/Cam; the implementation
    # must use the standard's value verbatim (shared oracle in
    # tests/reference_data/).
    from reference_data import ISO532_2_C

    from phonometry.psychoacoustics.loudness.moore_glasberg import _C_SONE

    assert _C_SONE == ISO532_2_C


# --------------------------------------------------------------------------
# Annex B.1 - sinusoidal tones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "sone", "phon"),
    [
        (10.0, 0.03, 10.0),
        (20.0, 0.14, 20.0),
        (30.0, 0.43, 30.0),
        (40.0, 1.0, 40.0),
        (50.0, 2.1, 50.0),
        (60.0, 4.1, 60.0),
        (70.0, 8.1, 70.0),
        (80.0, 15.8, 80.0),
    ],
)
def test_b1_1_tone_1khz_free_binaural(level: float, sone: float, phon: float) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, level)]
    )
    assert result.loudness == pytest.approx(sone, rel=0.05, abs=0.01)
    assert result.loudness_level == pytest.approx(phon, abs=1.0)


@pytest.mark.parametrize(
    ("level", "sone", "phon"),
    [(20.0, 0.35, 28.0), (40.0, 1.8, 48.0), (60.0, 7.0, 68.0), (80.0, 27.2, 87.5)],
)
def test_b1_2_tone_3khz_free_binaural(level: float, sone: float, phon: float) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(3000.0, level)]
    )
    assert result.loudness == pytest.approx(sone, rel=0.05)
    # Annex B.1.2 phon column (reproduces to < 0.1 phon).
    assert result.loudness_level == pytest.approx(phon, abs=0.15)


@pytest.mark.parametrize(
    ("level", "sone", "phon"),
    [(20.0, 0.07, 14.7), (40.0, 0.54, 32.7), (60.0, 2.31, 51.5), (80.0, 8.82, 71.4)],
)
def test_b1_3_tone_1khz_earphone_monaural(
    level: float, sone: float, phon: float
) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, level)], field="eardrum", presentation="monaural"
    )
    assert result.loudness == pytest.approx(sone, rel=0.05)
    # Annex B.1.3 phon column (reproduces to < 0.1 phon).
    assert result.loudness_level == pytest.approx(phon, abs=0.15)


def test_b1_4_tone_100hz_50db_free_binaural() -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(100.0, 50.0)]
    )
    assert result.loudness == pytest.approx(0.351, rel=0.05)


# --------------------------------------------------------------------------
# Annex B.2 - filtered noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("f_lo", "f_hi", "sone"),
    [(950.0, 1050.0, 4.21), (500.0, 1500.0, 14.17)],
)
def test_b2_1_white_noise_density_40db(f_lo: float, f_hi: float, sone: float) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        _white_band(f_lo, f_hi, 40.0)
    )
    assert result.loudness == pytest.approx(sone, rel=0.06)


def test_b2_2_white_noise_bw1000_density_30db() -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        _white_band(500.0, 1500.0, 30.0)
    )
    assert result.loudness == pytest.approx(7.97, rel=0.06)


@pytest.mark.parametrize(
    ("density", "sone", "phon"),
    [(0.0, 3.64, 58.1), (20.0, 15.85, 80.0), (40.0, 48.59, 95.2)],
)
def test_b2_3_pink_noise(density: float, sone: float, phon: float) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        _pink_band(50.0, 15000.0, density)
    )
    assert result.loudness == pytest.approx(sone, rel=0.05)
    # Annex B.2.3 phon column (reproduces to < 0.1 phon).
    assert result.loudness_level == pytest.approx(phon, abs=0.15)


@pytest.mark.parametrize(
    ("level", "sone", "phon"),
    [
        (0.0, 0.077, 15.4),
        (10.0, 0.69, 35.5),
        (20.0, 2.54, 52.8),
        (30.0, 6.25, 66.2),
        (40.0, 12.6, 76.7),
        (50.0, 23.1, 85.2),
    ],
)
def test_b2_4_broadband_third_octave_binaural(
    level: float, sone: float, phon: float
) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_third_octave(
        [level] * 29
    )
    assert result.loudness == pytest.approx(sone, rel=0.05)
    # Annex B.2.4 phon column (reproduces to < 0.1 phon).
    assert result.loudness_level == pytest.approx(phon, abs=0.15)


@pytest.mark.parametrize(
    ("level", "sone", "phon"),
    [
        (10.0, 0.08, 16.0),
        (20.0, 0.72, 35.9),
        (30.0, 2.41, 52.0),
        (40.0, 5.55, 64.4),
        (50.0, 10.7, 74.3),
    ],
)
def test_b2_5_broadband_third_octave_monaural(
    level: float, sone: float, phon: float
) -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_third_octave(
        [level] * 29, field="eardrum", presentation="monaural"
    )
    assert result.loudness == pytest.approx(sone, rel=0.05)
    # Annex B.2.5 phon column (reproduces to < 0.1 phon).
    assert result.loudness_level == pytest.approx(phon, abs=0.15)


# --------------------------------------------------------------------------
# Annex B.3 - multiple tones, and B.4 - tone with noise
# --------------------------------------------------------------------------


def test_b3_1_three_close_tones() -> None:
    comps = [(1500.0, 60.0), (1600.0, 60.0), (1700.0, 60.0)]
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(comps)
    assert result.loudness == pytest.approx(6.31, rel=0.04)


def test_b3_2_three_spread_tones_louder_than_close() -> None:
    close = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1500.0, 60.0), (1600.0, 60.0), (1700.0, 60.0)]
    ).loudness
    spread = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0), (1600.0, 60.0), (2400.0, 60.0)]
    ).loudness
    assert spread == pytest.approx(12.49, rel=0.04)
    # Wider spectral spacing is louder (less mutual masking), Annex B.3 note.
    assert spread > close


def test_b3_3_harmonic_complex() -> None:
    comps = [(float(f), 30.0) for f in range(100, 1001, 100)]
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(comps)
    assert result.loudness == pytest.approx(2.00, rel=0.04)


def test_b4_tone_with_noise_less_overlap_is_louder() -> None:
    overlap = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0), *_white_band(950.0, 1050.0, 40.0)]
    ).loudness
    apart = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0), *_white_band(1450.0, 1550.0, 40.0)]
    ).loudness
    assert overlap == pytest.approx(5.09, rel=0.06)
    assert apart == pytest.approx(7.17, rel=0.06)
    # Reduced spectral overlap of tone and noise is louder (Annex B.4.2 note).
    assert apart > overlap


# --------------------------------------------------------------------------
# Internal cross-checks
# --------------------------------------------------------------------------


def test_monotonic_in_level() -> None:
    values = [
        psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, lvl)]
        ).loudness
        for lvl in (20.0, 40.0, 60.0, 80.0)
    ]
    assert values[0] < values[1] < values[2] < values[3]


def test_ten_phon_doubles_loudness() -> None:
    # A +10 phon step doubles the loudness in sone (clause 3.17 note 4).
    for phon in (50.0, 60.0, 70.0):
        low = psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, phon)]
        ).loudness
        high = psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, phon + 10.0)]
        ).loudness
        assert high / low == pytest.approx(2.0, rel=0.1)


def test_free_and_diffuse_differ() -> None:
    free = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(3000.0, 60.0)], field="free"
    )
    diffuse = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(3000.0, 60.0)], field="diffuse"
    )
    assert free.loudness != pytest.approx(diffuse.loudness, rel=1e-3)


def test_silence_is_zero() -> None:
    empty = psychoacoustics.loudness_moore_glasberg_from_spectrum([])
    assert empty.loudness == 0.0
    assert empty.loudness_level == 0.0
    assert np.all(empty.specific == 0.0)
    quiet = psychoacoustics.loudness_moore_glasberg_from_third_octave(
        [-80.0] * 29
    )
    assert quiet.loudness < 1e-3


def test_ballpark_agreement_with_zwicker_broadband() -> None:
    # Different model, so only order-of-magnitude agreement is expected.

    levels_28 = [60.0] * 28  # ISO 532-1 uses 28 bands (25 Hz..12.5 kHz)
    zwicker = psychoacoustics.loudness_zwicker_from_spectrum(
        levels_28
    ).loudness
    moore = psychoacoustics.loudness_moore_glasberg_from_third_octave(
        [60.0] * 29
    ).loudness
    assert 0.4 < moore / zwicker < 2.5


# --------------------------------------------------------------------------
# Structure, signal path and validation
# --------------------------------------------------------------------------


def test_result_structure() -> None:
    result = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0)]
    )
    assert isinstance(result, psychoacoustics.MooreGlasbergLoudness)
    assert result.specific.shape == (372,)
    assert result.erb_number.shape == (372,)
    assert result.centre_frequencies.shape == (372,)
    assert result.erb_number[0] == pytest.approx(1.8)
    assert result.erb_number[-1] == pytest.approx(38.9)
    assert result.field == "free"
    assert result.presentation == "binaural"
    # Specific loudness peaks near the 1 kHz filter (~15.6 Cam).
    peak_cam = result.erb_number[int(np.argmax(result.specific))]
    assert 14.0 < peak_cam < 17.0


def _tone_signal(level_db: float, freq: float = 1000.0, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(FS * seconds)) / FS
    amp = np.sqrt(2.0) * 2e-5 * 10 ** (level_db / 20.0)
    return amp * np.sin(2.0 * np.pi * freq * t)


def test_signal_path_monotonic_and_calibrated() -> None:
    low = psychoacoustics.loudness_moore_glasberg(_tone_signal(40.0), FS)
    high = psychoacoustics.loudness_moore_glasberg(_tone_signal(60.0), FS)
    assert isinstance(low, psychoacoustics.MooreGlasbergLoudness)
    assert high.loudness > low.loudness
    # The signal path builds a narrowband spectrum, so a pure tone reduces to a
    # single component and reproduces the definitional anchor.
    assert 0.5 < low.loudness < 8.0


def test_signal_path_anchor_is_one_sone() -> None:
    # A calibrated 1 kHz tone at 40 dB SPL is the definitional anchor of the
    # sone (clause 3.17): the signal path must reproduce 1.000 sone / 40 phon,
    # the same value as the exact spectrum path for a single 1 kHz line.
    result = psychoacoustics.loudness_moore_glasberg(_tone_signal(40.0), FS)
    assert result.loudness == pytest.approx(1.0, abs=0.02)
    assert result.loudness_level == pytest.approx(40.0, abs=0.5)


def test_signal_path_matches_spectrum_for_multitone() -> None:
    # The narrowband signal path must agree with the exact sinusoidal-component
    # (clause 5.2/5.4) spectrum path for a multi-tone signal.
    signal = (
        _tone_signal(50.0, freq=400.0)
        + _tone_signal(50.0, freq=1000.0)
        + _tone_signal(50.0, freq=2500.0)
    )
    from_signal = psychoacoustics.loudness_moore_glasberg(signal, FS).loudness
    from_spectrum = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(400.0, 50.0), (1000.0, 50.0), (2500.0, 50.0)]
    ).loudness
    assert from_signal == pytest.approx(from_spectrum, rel=0.02)


def test_diotic_alias_matches_binaural() -> None:
    a = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0)], presentation="binaural"
    )
    b = psychoacoustics.loudness_moore_glasberg_from_spectrum(
        [(1000.0, 60.0)], presentation="diotic"
    )
    assert a.loudness == pytest.approx(b.loudness)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="field must be one of"):
        psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, 40.0)], field="near"
        )
    with pytest.raises(ValueError, match="presentation must be one of"):
        psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, 40.0)], presentation="mono"
        )
    with pytest.raises(
        ValueError, match="component frequencies must be positive"
    ):
        psychoacoustics.loudness_moore_glasberg_from_spectrum([(-10.0, 40.0)])
    with pytest.raises(
        ValueError, match="components must contain only finite values"
    ):
        psychoacoustics.loudness_moore_glasberg_from_spectrum(
            [(1000.0, np.inf)]
        )
    with pytest.raises(ValueError, match="band_levels must contain exactly"):
        psychoacoustics.loudness_moore_glasberg_from_third_octave(
            [60.0] * 10
        )  # wrong length
    empty = np.array([])
    with pytest.raises(ValueError, match="Input signal 'x' cannot be empty"):
        psychoacoustics.loudness_moore_glasberg(empty, FS)
    short = np.ones(100)
    with pytest.raises(
        ValueError, match="'fs' must be a positive sampling rate"
    ):
        psychoacoustics.loudness_moore_glasberg(short, -1.0)
