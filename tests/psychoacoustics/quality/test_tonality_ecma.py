#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
ECMA-418-2:2025 (Sottek Hearing Model) tonality conformance tests.

The primary oracle is the standard's own calibration (Clause 6.2.8): a 1 kHz
tone at 40 dB SPL yields 1 tu_HMS. That is the only tonality reference value
the standard tabulates. The remaining checks are the behaviours the standard
states qualitatively (pure tone tonality far exceeds broadband noise, the
tonal frequency tracks the tone, silence -> 0, band restriction of the
time-dependent maximum) plus result-structure guards.
"""

import numpy as np
import pytest

from phonometry import EcmaTonality, tonality_ecma

FS = 48000


def _tone(freq: float, level_db: float, seconds: float = 1.5) -> np.ndarray:
    """Pure tone at an SPL (dB re 20 uPa), pressure in pascals."""
    t = np.arange(int(FS * seconds)) / FS
    amp = np.sqrt(2.0) * 2e-5 * 10 ** (level_db / 20.0)
    return amp * np.sin(2.0 * np.pi * freq * t)


def _noise(level_db: float, seconds: float = 1.5) -> np.ndarray:
    """White Gaussian noise at an overall SPL, pressure in pascals."""
    rng = np.random.default_rng(2025)
    x = rng.standard_normal(int(FS * seconds))
    x *= 2e-5 * 10 ** (level_db / 20.0) / x.std()
    return x


@pytest.fixture(scope="module")
def ref_1k_40() -> EcmaTonality:
    """The calibration signal result, computed once for the module."""
    return tonality_ecma(_tone(1000.0, 40.0), FS)


@pytest.fixture(scope="module")
def broadband_noise() -> EcmaTonality:
    """A broadband-noise result, computed once for the module.

    0.8 s: a qualitative reference (noise stays far below a tone), so it only
    needs to clear the transient-discard window, not the calibration length.
    """
    return tonality_ecma(_noise(60.0, seconds=0.8), FS)


# --------------------------------------------------------------------------
# Primary reference value (Clause 6.2.8 calibration)
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_calibration_1khz_40db_is_one_tu(ref_1k_40: EcmaTonality) -> None:
    # c_T is defined so the 1 kHz / 40 dB tone gives 1 tu_HMS; the standard
    # allows c_T to vary within 0.25 %, and implementation differences add a
    # little more, so a 3 % window is comfortably tight.
    assert ref_1k_40.tonality == pytest.approx(1.0, abs=0.03)


def test_calibration_constant_is_the_tabulated_c_t() -> None:
    # Clause 6.2.8 tabulates c_T = 2.8758615; the implementation must use the
    # standard's value verbatim (shared oracle in tests/reference_data.py).
    from reference_data import ECMA418_2_TONALITY_C_T

    from phonometry.psychoacoustics.quality.tonality_ecma import _C_T

    assert _C_T == ECMA418_2_TONALITY_C_T


def test_decision_thresholds_are_the_standard_constants() -> None:
    # The standard's verbatim decision criteria (its only numeric anchors
    # beyond the calibration points): prominence 0.4 tu_HMS (Clause 6.3),
    # prominent roughness 0.2 asper (Clause 7.2) and the 0.01 sone_HMS
    # audibility criterion on the total basis loudness (Clause 5.1.9).
    import reference_data as ref

    from phonometry.psychoacoustics.loudness.ecma import (
        AUDIBILITY_THRESHOLD_SONE_HMS,
    )
    from phonometry.psychoacoustics.quality.roughness_ecma import (
        PROMINENT_ROUGHNESS_ASPER,
    )
    from phonometry.psychoacoustics.quality.tonality_ecma import (
        PROMINENT_TONALITY_TU_HMS,
    )

    assert PROMINENT_TONALITY_TU_HMS == ref.ECMA418_2_PROMINENT_TONALITY_TU
    assert PROMINENT_ROUGHNESS_ASPER == ref.ECMA418_2_PROMINENT_ROUGHNESS_ASPER
    assert (
        AUDIBILITY_THRESHOLD_SONE_HMS == ref.ECMA418_2_AUDIBILITY_THRESHOLD_SONE
    )


# --------------------------------------------------------------------------
# Standard's qualitative behaviours
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_pure_tone_far_exceeds_noise(
    ref_1k_40: EcmaTonality, broadband_noise: EcmaTonality
) -> None:
    # Broadband noise -> near 0 (the sigmoid + 0.02 tu gate suppress it),
    # a pure tone -> near the top of the scale.
    assert broadband_noise.tonality < 0.2
    assert ref_1k_40.tonality > 5.0 * broadband_noise.tonality
    assert ref_1k_40.tonality > 0.8


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_tonal_frequency_tracks_tone(ref_1k_40: EcmaTonality) -> None:
    peak_band = int(np.argmax(ref_1k_40.specific_tonality))
    # The peaking band centres on the tone and its estimated tonal frequency
    # matches (Formulae 39/55).
    assert ref_1k_40.centre_frequencies[peak_band] == pytest.approx(1000.0, rel=0.15)
    assert ref_1k_40.tonal_frequencies[peak_band] == pytest.approx(1000.0, rel=0.05)


def test_tonal_frequency_tracks_a_second_tone() -> None:
    # Frequency tracking is spectral, not duration-driven: 0.7 s suffices.
    result = tonality_ecma(_tone(2000.0, 50.0, seconds=0.7), FS)
    peak_band = int(np.argmax(result.specific_tonality))
    assert result.tonal_frequencies[peak_band] == pytest.approx(2000.0, rel=0.05)


def test_short_signal_averages_over_all_blocks() -> None:
    """A signal too short for the normal transient discard (< ~0.3 s) averages
    over all available blocks, matching ``loudness_ecma``'s fall-back.

    Previously such signals collapsed onto the final block alone; the fall-back
    keeps a short tonal signal clearly tonal and well above short broadband
    noise (the normal-length calibration anchor is unaffected: those signals
    have far more blocks than the 56-block transient window).
    """
    short_tone = tonality_ecma(_tone(1000.0, 40.0, seconds=0.2), FS)
    assert np.isfinite(short_tone.tonality)
    assert short_tone.tonality > 0.3
    short_noise = tonality_ecma(_noise(60.0, seconds=0.2), FS)
    assert short_tone.tonality > short_noise.tonality


def test_silence_is_zero() -> None:
    # Pure-property check: zero in, zero out at any length past the
    # transient-discard window, so 0.6 s is enough.
    result = tonality_ecma(np.zeros(int(FS * 0.6)), FS)
    assert result.tonality == 0.0
    assert np.all(result.specific_tonality == 0.0)


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_specific_tonality_peaks_near_tone(ref_1k_40: EcmaTonality) -> None:
    peak_band = int(np.argmax(ref_1k_40.specific_tonality))
    assert ref_1k_40.centre_frequencies[peak_band] == pytest.approx(1000.0, rel=0.15)


def test_user_band_excluding_tone_lowers_tonality() -> None:
    # Restricting the time-dependent maximum search (Formulae 56-61) to bands
    # above the tone removes the tonal event, driving T towards 0. A 0.7 s
    # segment is enough: the property is spectral, not duration-driven.
    restricted = tonality_ecma(_tone(1000.0, 40.0, seconds=0.7), FS, f_low=3000.0)
    assert restricted.tonality < 0.2


def test_loudness_and_tonality_share_the_tonal_split(monkeypatch) -> None:
    """Loudness and tonality must report the same underlying N'_tonal(l, z).

    Clause 8.1.1 builds the loudness on the Clause 6.2 outputs, so the two
    metrics have to agree on the shared intermediate, not just at their
    endpoints. Both modules call the same ``_tonal_noise_split``; this test
    spies on it through both call sites and asserts the recorded
    N'_tonal(l, z) arrays are identical.
    """
    import sys

    from phonometry import loudness_ecma

    L = sys.modules["phonometry.psychoacoustics.loudness.ecma"]
    T = sys.modules["phonometry.psychoacoustics.quality.tonality_ecma"]
    assert T._tonal_noise_split is L._tonal_noise_split

    recorded = []
    orig = L._tonal_noise_split

    def spy(x, field):
        out = orig(x, field)
        recorded.append(out[0].copy())  # N'_tonal(l, z)
        return out

    monkeypatch.setattr(L, "_tonal_noise_split", spy)
    monkeypatch.setattr(T, "_tonal_noise_split", spy)
    sig = _tone(1000.0, 40.0, seconds=0.7)
    loudness_ecma(sig, FS)
    tonality_ecma(sig, FS)
    assert len(recorded) == 2
    assert np.array_equal(recorded[0], recorded[1])


def test_band_range_rejects_out_of_range_edges() -> None:
    # Formulae 56/57 preconditions: 16 Hz < f_L, f_H < 20 kHz, f_L < f_H.
    x = _tone(1000.0, 40.0, seconds=0.5)
    with pytest.raises(ValueError, match="16 Hz"):
        tonality_ecma(x, FS, f_low=10.0)
    with pytest.raises(ValueError, match="20 kHz"):
        tonality_ecma(x, FS, f_high=25000.0)
    with pytest.raises(ValueError, match="below 'f_high'"):
        tonality_ecma(x, FS, f_low=2000.0, f_high=500.0)


def test_band_range_uses_edge_midpoints() -> None:
    # Formulae 56-57 select bands by the edge midpoints to the neighbours, not
    # by the centre frequency.  For an f_low that lies between band z's centre
    # F(z) and its upper boundary (F(z)+F(z+0.5))/2, band z must still be the
    # lower edge z_L (a centre-frequency threshold would wrongly exclude it).
    from phonometry.psychoacoustics.loudness.ecma import _F_CENTRE
    from phonometry.psychoacoustics.quality.tonality_ecma import _band_range

    z = 20
    upper_mid = 0.5 * (_F_CENTRE[z] + _F_CENTRE[z + 1])
    f_low = 0.5 * (_F_CENTRE[z] + upper_mid)  # inside band z, above its centre
    assert f_low > _F_CENTRE[z]
    z_lo, _ = _band_range(f_low, None)
    assert z_lo == z
    # Symmetric check for the upper edge via Formula 57.
    lower_mid = 0.5 * (_F_CENTRE[z] + _F_CENTRE[z - 1])
    f_high = 0.5 * (_F_CENTRE[z] + lower_mid)  # inside band z, below its centre
    assert f_high < _F_CENTRE[z]
    _, z_hi = _band_range(None, f_high)
    assert z_hi == z


# --------------------------------------------------------------------------
# Field handling
# --------------------------------------------------------------------------


def test_free_and_diffuse_fields_differ() -> None:
    # Property check (ear-filter difference), not a calibration: 0.7 s is
    # enough for a stable value in both fields.
    x = _tone(1000.0, 60.0, seconds=0.7)
    free = tonality_ecma(x, FS, field="free").tonality
    diffuse = tonality_ecma(x, FS, field="diffuse").tonality
    assert free > 0.5
    assert diffuse > 0.5
    assert free != diffuse


# --------------------------------------------------------------------------
# Result structure and validation
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_result_structure(ref_1k_40: EcmaTonality) -> None:
    assert ref_1k_40.specific_tonality.shape == (53,)
    assert ref_1k_40.tonal_frequencies.shape == (53,)
    assert ref_1k_40.bark.shape == (53,)
    assert ref_1k_40.bark[0] == pytest.approx(0.5)
    assert ref_1k_40.bark[-1] == pytest.approx(26.5)
    assert ref_1k_40.centre_frequencies.shape == (53,)
    assert ref_1k_40.time.shape == ref_1k_40.tonality_vs_time.shape
    assert ref_1k_40.time.shape == ref_1k_40.tonal_frequency_vs_time.shape
    assert ref_1k_40.field == "free"
    # Time-dependent tonality is sampled at 187.5 Hz (Clause 6.2.8/6.2.6).
    dt = np.diff(ref_1k_40.time)
    assert np.allclose(dt, 1.0 / 187.5)


def test_invalid_field() -> None:
    with pytest.raises(ValueError):
        tonality_ecma(_tone(1000.0, 40.0, seconds=0.5), FS, field="reverberant")


def test_invalid_fs() -> None:
    with pytest.raises(ValueError):
        tonality_ecma(_tone(1000.0, 40.0, seconds=0.5), 0.0)


def test_empty_signal() -> None:
    with pytest.raises(ValueError):
        tonality_ecma(np.array([]), FS)


@pytest.mark.xdist_group("ecma-tonality-ref")
def test_plot_smoke(ref_1k_40: EcmaTonality) -> None:
    import matplotlib

    matplotlib.use("Agg")
    axes = ref_1k_40.plot()
    assert axes.shape == (2,)
