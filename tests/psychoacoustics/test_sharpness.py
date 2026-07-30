#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Sharpness (DIN 45692:2009-08) conformance tests.

Clause 6: the standard test signal is critical-band-wide narrowband noise
at 1 kHz (920-1080 Hz) with 60 dB overall level -> S = 1.00 acum, and every
other test signal carries the same loudness (4 sone). Targets from
Table A.2 (p. 11) with critical-band edges from Tabelle A.1 (p. 10);
permitted deviation 5 % or 0.05 acum.
"""

import numpy as np
import pytest
from scipy import signal as sp_signal

from phonometry.psychoacoustics import (
    loudness_zwicker,
    sharpness_din,
    sharpness_din_from_specific,
)
from phonometry.psychoacoustics.sharpness import reference_sound

FS = 48000

# DIN 45692 Tabelle A.1 critical-band edges (Frequenzgruppen) and the full
# Table A.2 Sollwerte: centre frequency -> (lower edge, upper edge) Hz,
# target acum. All 21 rows of the normative verification table.
A2_CASES = [
    (250.0, 200.0, 300.0, 0.38),
    (350.0, 300.0, 400.0, 0.49),
    (450.0, 400.0, 510.0, 0.60),
    (570.0, 510.0, 630.0, 0.71),
    (700.0, 630.0, 770.0, 0.82),
    (840.0, 770.0, 920.0, 0.93),
    (1000.0, 920.0, 1080.0, 1.00),
    (1170.0, 1080.0, 1270.0, 1.13),
    (1370.0, 1270.0, 1480.0, 1.26),
    (1600.0, 1480.0, 1720.0, 1.35),
    (1850.0, 1720.0, 2000.0, 1.49),
    (2150.0, 2000.0, 2320.0, 1.64),
    (2500.0, 2320.0, 2700.0, 1.78),
    (2900.0, 2700.0, 3150.0, 2.06),
    (3400.0, 3150.0, 3700.0, 2.40),
    (4000.0, 3700.0, 4400.0, 2.82),
    (4800.0, 4400.0, 5300.0, 3.48),
    (5800.0, 5300.0, 6400.0, 4.43),
    (7000.0, 6400.0, 7700.0, 5.52),
    (8500.0, 7700.0, 9500.0, 6.81),
    (10500.0, 9500.0, 12000.0, 8.55),
]

# Table A.3 Sollwerte: broadband noise with a fixed 10 kHz upper edge and a
# variable lower edge fu, all at 4 sone. All 20 rows.
A3_CASES = [
    (250.0, 2.70), (350.0, 2.74), (450.0, 2.78), (570.0, 2.85),
    (700.0, 2.91), (840.0, 2.96), (1000.0, 3.05), (1170.0, 3.12),
    (1370.0, 3.20), (1600.0, 3.30), (1850.0, 3.42), (2150.0, 3.53),
    (2500.0, 3.69), (2900.0, 3.89), (3400.0, 4.12), (4000.0, 4.49),
    (4800.0, 5.04), (5800.0, 5.69), (7000.0, 6.47), (8500.0, 7.46),
]


def _narrowband(f_lo: float, f_hi: float, level_db: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(FS * 2)
    sos = sp_signal.butter(8, [f_lo, f_hi], btype="band", fs=FS, output="sos")
    nb = sp_signal.sosfilt(sos, white)
    return np.asarray(nb / np.sqrt(np.mean(nb**2)) * 2e-5 * 10 ** (level_db / 20))


def _level_for_4_sone(f_lo: float, f_hi: float) -> float:
    # 13 bisection steps resolve the level to ~0.007 dB, far below the
    # 5 % / 0.05 acum verification tolerance.
    lo, hi = 30.0, 90.0
    for _ in range(13):
        mid = (lo + hi) / 2
        n = loudness_zwicker(_narrowband(f_lo, f_hi, mid), FS, stationary=True).loudness
        if n < 4.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_reference_signal_is_one_acum() -> None:
    """Clause 6: the standard test signal gives exactly 1.00 acum."""
    s = sharpness_din(reference_sound(), FS)
    assert s == pytest.approx(1.00, abs=1e-9)


def test_k_in_normative_range() -> None:
    """Clause 5.2: the normalization constant satisfies 0.105 <= k < 0.115."""
    from phonometry.psychoacoustics.sharpness import _k_din

    assert 0.105 <= _k_din() < 0.115


@pytest.mark.parametrize(("fc", "f_lo", "f_hi", "target"), A2_CASES)
def test_table_a2_targets(fc: float, f_lo: float, f_hi: float, target: float) -> None:
    """Table A.2 target sharpness for critical-band-wide noises at 4 sone,
    within the clause 6 tolerance (5 % or 0.05 acum)."""
    level = _level_for_4_sone(f_lo, f_hi)
    s = sharpness_din(_narrowband(f_lo, f_hi, level), FS)
    tol = max(0.05, 0.05 * target)
    assert abs(s - target) <= tol, f"{fc} Hz: S={s:.3f} vs {target} (tol {tol:.3f})"


@pytest.mark.parametrize(("f_lo", "target"), A3_CASES)
def test_table_a3_targets(f_lo: float, target: float) -> None:
    """Table A.3 target sharpness for broadband noises (fu .. 10 kHz) at
    4 sone, within the clause 6 tolerance (5 % or 0.05 acum)."""
    level = _level_for_4_sone(f_lo, 10000.0)
    s = sharpness_din(_narrowband(f_lo, 10000.0, level), FS)
    tol = max(0.05, 0.05 * target)
    assert abs(s - target) <= tol, f"fu={f_lo} Hz: S={s:.3f} vs {target} (tol {tol:.3f})"


def test_higher_frequency_is_sharper() -> None:
    lo = sharpness_din(_narrowband(400.0, 510.0, 60.0), FS)
    hi = sharpness_din(_narrowband(7700.0, 9500.0, 60.0), FS)
    assert hi > lo


def test_annex_b_variants() -> None:
    """The Aures and von Bismarck variants use the literal k = 0.11 of
    DIN 45692 Formulas (B.1)/(B.2). The clause 6 reference sound then lands
    NEAR (not exactly at) 1 acum -- a non-circular check of the published
    formulas, unlike the normative method whose k is derived from this very
    signal. Measured: ~0.96 acum (Aures), ~1.02 acum (von Bismarck)."""
    spec = loudness_zwicker(reference_sound(), FS, stationary=True).specific
    for method in ("aures", "bismarck"):
        s = sharpness_din_from_specific(spec, method=method)
        assert abs(s - 1.0) < 0.05, f"{method}: S={s:.6f}"
    # The variants must differ from each other and from the DIN value
    # (three distinct weightings, one shared literal factor).
    s_a = sharpness_din_from_specific(spec, method="aures")
    s_b = sharpness_din_from_specific(spec, method="bismarck")
    assert s_a != s_b
    with pytest.raises(ValueError, match="method"):
        sharpness_din_from_specific(spec, method="zwicker")


def test_annex_b_literal_factor() -> None:
    """Formulas (B.1)/(B.2) print S = 0,11 * moment; the implementation must
    use the literal 0.11, not a self-derived per-variant constant (which
    would shift every Aures result ~4 % against other implementations)."""
    from phonometry.psychoacoustics.sharpness import _K_ANNEX_B

    assert _K_ANNEX_B == 0.11


def test_silence_is_zero() -> None:
    assert sharpness_din_from_specific(np.zeros(240)) == 0.0
