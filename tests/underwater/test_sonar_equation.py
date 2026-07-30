#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the sonar equation (passive and active).

Oracles: a hand-worked textbook term balance (Urick via Etter, Table 10.2) —
pure arithmetic, independent of the implementation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from phonometry.underwater.sonar_equation import (
    SonarEquationResult,
    active_sonar_equation,
    passive_sonar_equation,
)


def test_passive_signal_excess_and_fom() -> None:
    # SE = SL - TL - (NL - DI) - DT ; hand values.
    res = passive_sonar_equation(140.0, 80.0, 60.0, directivity_index=10.0,
                                 detection_threshold=5.0)
    assert isinstance(res, SonarEquationResult)
    assert res.mode == "passive"
    # SE = 140 - 80 - (60 - 10) - 5 = 5
    assert res.signal_excess[0] == pytest.approx(5.0)
    # SNR = SE + DT = 10
    assert res.snr[0] == pytest.approx(10.0)
    # FOM = SL - (NL - DI) - DT = 140 - 50 - 5 = 85
    assert res.figure_of_merit == pytest.approx(85.0)


def test_passive_detection_at_fom() -> None:
    # At TL = FOM the signal excess is exactly zero (detection limit).
    res = passive_sonar_equation(140.0, [85.0], 60.0, directivity_index=10.0,
                                 detection_threshold=5.0)
    assert res.signal_excess[0] == pytest.approx(0.0, abs=1e-9)


def test_active_noise_limited() -> None:
    # SE = SL - 2 TL + TS - (NL - DI) - DT
    res = active_sonar_equation(220.0, 70.0, 15.0, 60.0, directivity_index=20.0,
                                detection_threshold=10.0)
    assert res.mode == "active"
    # SE = 220 - 140 + 15 - (60 - 20) - 10 = 45
    assert res.signal_excess[0] == pytest.approx(45.0)
    # FOM = (SL + TS - (NL - DI) - DT)/2 = (220 + 15 - 40 - 10)/2 = 92.5
    assert res.figure_of_merit == pytest.approx(92.5)


def test_active_reverberation_limited_ignores_di() -> None:
    # With RL given, masking is RL (DI does not apply to reverberation).
    res = active_sonar_equation(220.0, 70.0, 15.0, 60.0, directivity_index=20.0,
                                detection_threshold=10.0, reverberation_level=55.0)
    assert res.reverberation_limited is True
    # SE = 220 - 140 + 15 - 55 - 10 = 30
    assert res.signal_excess[0] == pytest.approx(30.0)
    assert res.figure_of_merit == pytest.approx((220.0 + 15.0 - 55.0 - 10.0) / 2.0)


def test_signal_excess_decreases_with_transmission_loss() -> None:
    tl = np.linspace(50.0, 120.0, 8)
    res = passive_sonar_equation(150.0, tl, 55.0)
    assert np.all(np.diff(res.signal_excess) < 0.0)
    # Passive SE decreases 1 dB per dB of one-way TL.
    np.testing.assert_allclose(np.diff(res.signal_excess), -np.diff(tl))


def test_active_two_way_loss() -> None:
    # Active SE loses 2 dB per dB of one-way TL.
    res = active_sonar_equation(200.0, [60.0, 61.0], 10.0, 50.0)
    assert res.signal_excess[1] - res.signal_excess[0] == pytest.approx(-2.0)


def test_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        passive_sonar_equation(float("nan"), 80.0, 60.0)


def test_plot_smoke() -> None:
    res = passive_sonar_equation(150.0, np.linspace(40.0, 110.0, 40), 55.0,
                                 detection_threshold=8.0)
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# Ainslie (2010) worked-example figure-of-merit oracles
# ---------------------------------------------------------------------------

# Ainslie, Principles of Sonar Performance Modelling (Springer, 2010),
# publishes complete term tables for his sonar-equation worked examples. The
# term balances below re-combine those printed values through the library's
# equations; every expected number is the book's, so these anchor the sign
# conventions of the SL/NL/DI/DT/TS combination against an independent source.


def test_ainslie_passive_narrowband_fom() -> None:
    # Table 3.1 (p. 76), passive narrowband example of Sec. 3.2.3.8:
    # SL = 133.9 dB re uPa2 m2, NLf = 59.7 dB re uPa2/Hz, AG = 11.5 dB,
    # DT = 13.8 dB, analysis bandwidth 0.25 Hz (the printed "BW = -6.0 dB"
    # row) -> FOM = 78.0 dB re m2. The bandwidth term folds into the masking
    # noise as NL = NLf + 10 lg(0.25 Hz) = 59.7 - 6.0 dB. Five printed
    # terms, each rounded to 0.1 dB, give a 0.15 dB accumulation allowance.
    res = passive_sonar_equation(
        133.9, 0.0, 59.7 - 6.0,
        directivity_index=11.5, detection_threshold=13.8,
    )
    assert res.figure_of_merit == pytest.approx(78.0, abs=0.15)


def test_ainslie_passive_broadband_fom() -> None:
    # Table 3.2 (p. 90), passive broadband example of Sec. 3.2.4.8 (spectral
    # density form, no bandwidth term): SLf = 100.9 dB re uPa2 m2/Hz,
    # NLf = 53.2 dB re uPa2/Hz, AGm = 12.8 dB, DT = -18.6 dB ->
    # FOM = 79.0 dB re m2 (four printed 0.1 dB terms -> 0.15 dB allowance).
    res = passive_sonar_equation(
        100.9, 0.0, 53.2,
        directivity_index=12.8, detection_threshold=-18.6,
    )
    assert res.figure_of_merit == pytest.approx(79.0, abs=0.15)


def test_ainslie_active_orca_noise_limited_fom() -> None:
    # Sec. 11.4.6 (orca vs salmon, Tables 11.6-11.7 pp. 620-624): SL(RMS) =
    # 198.2 dB re uPa2 m2, TS(salmon, 0.8 m) = -29.0 dB re m2, wind noise
    # NL = 75.0 dB re uPa2, AG = 16.5 dB, DT = 8.7 dB -> noise-limited
    # FOM_NL = (SL + TS - (NL - AG) - DT)/2 = 51.0 dB re m2.
    res = active_sonar_equation(
        198.2, 0.0, -29.0, 75.0,
        directivity_index=16.5, detection_threshold=8.7,
    )
    assert res.figure_of_merit == pytest.approx(51.0, abs=0.05)


def test_ainslie_active_orca_hearing_threshold_fom() -> None:
    # Same example: against the orca's hearing threshold at 50 kHz,
    # HT = 51.2 dB re uPa2 (audiogram Eq. 11.159), the book's
    # FOM_HT = (SL + TS - HT)/2 = 59.0 dB re m2; the threshold acts as the
    # masking level with no array gain and no detection threshold.
    res = active_sonar_equation(198.2, 0.0, -29.0, 51.2)
    assert res.figure_of_merit == pytest.approx(59.0, abs=0.05)
