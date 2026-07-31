#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Metadiffuser panels against the published designs (Sci. Rep. 7:5389, 2017).

The oracles are the printed numbers of the paper and its supplementary
material, computed here through an independent implementation of the same
transfer-matrix model: the quadratic-residue metadiffuser of Table 1
(critical coupling of its first slit, and the headline claim that its
spatially dependent reflection matches the target QRD at the evaluation
frequency), the primitive-root metadiffuser of Table 2 (the sharp
single-slit absorption peak and the specular notch), the ternary-sequence
states of Table 3 (perfect absorber and phase inverter; the well pitch is
read from Fig. 6, eight wells over 80 cm) and the broadband panel of
Table 4 (soft bounds only: several of its optimised necks are wider than
their slits, outside the fitted domain of the Dubos end correction, so the
low-frequency features of the paper are not recoverable from the text
alone).
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry.materials.metadiffuser import (
    MetadiffuserWell,
    metadiffuser_diffusion_spectrum,
    metadiffuser_polar_response,
    metadiffuser_reflection,
)
from phonometry.materials.slow_sound_absorber import (
    HelmholtzResonator,
    SlowSoundAbsorberWarning,
)

MM = 1e-3


def _well(
    h: float, ln: float, lc: float, wn: float, wc: float, m: int = 1
) -> MetadiffuserWell:
    resonator = HelmholtzResonator(
        neck_length=max(ln, 0.05) * MM, neck_side=wn * MM,
        cavity_length=lc * MM, cavity_side=wc * MM,
    )
    return MetadiffuserWell(h * MM, (resonator,) * m)


# Table 1: QR-metadiffuser, N = 5 slits, M = 2, L = 2 cm, d = 7 cm.
QR_WELLS = [
    _well(14.7, 13.0, 16.4, 6.2, 9.0, m=2),
    _well(30.9, 9.1, 4.3, 3.5, 9.0, m=2),
    _well(30.9, 9.1, 4.3, 3.5, 9.0, m=2),
    _well(15.7, 13.3, 17.0, 6.3, 9.0, m=2),
    _well(20.3, 18.0, 20.7, 3.2, 9.0, m=2),
]
QR_SEQUENCE = (1.0, 4.0, 4.0, 1.0, 0.0)

# Table 2: PR-metadiffuser, N = 6 slits, M = 1, L = 3.5 cm, d = 7 cm.
PR_WELLS = [
    _well(0.5, 26.1, 23.4, 19.4, 34.0),
    _well(14.4, 16.7, 26.0, 14.7, 34.0),
    _well(1.1, 5.7, 26.2, 7.3, 34.0),
    _well(22.0, 14.3, 19.1, 13.9, 34.0),
    _well(14.6, 18.6, 24.7, 14.7, 34.0),
    _well(22.4, 14.9, 19.9, 18.3, 34.0),
]

# Table 3 states (ternary sequence, L = 3 cm, d = 10 cm from Fig. 6).
INVERTER = _well(8.5, 1.8, 88.7, 8.4, 29.0)
ABSORBER = _well(10.0, 69.4, 10.2, 2.4, 29.0)

# Table 4: broadband panel, N = 11 slits, M = 1, L = 3 cm, d = 12 cm.
BROADBAND_ROWS = [
    (5.7, 16.3, 97.1, 6.7, 29.0), (4.9, 7.3, 106.8, 6.5, 29.0),
    (7.7, 37.1, 74.2, 10.0, 29.0), (82.9, 0.0, 36.0, 29.0, 29.0),
    (48.4, 35.3, 35.3, 29.0, 29.0), (74.9, 22.1, 22.1, 29.0, 29.0),
    (20.0, 14.7, 84.3, 14.0, 29.0), (6.6, 0.1, 112.2, 9.5, 29.0),
    (76.2, 0.0, 42.7, 29.0, 29.0), (29.5, 0.1, 89.4, 27.6, 29.0),
    (7.6, 4.8, 106.5, 6.2, 29.0),
]


def test_qr_first_slit_reaches_critical_coupling() -> None:
    # Paper: "at f = 2270 Hz the reflection coefficient vanishes at the
    # n = 1 slit" of the QR-metadiffuser.
    f = np.arange(2000.0, 2601.0, 5.0)
    result = metadiffuser_reflection(f, QR_WELLS, depth=0.02, period=0.07)
    alpha = result.well_absorption[0]
    peak = int(np.argmax(alpha))
    assert alpha[peak] > 0.95
    assert f[peak] == pytest.approx(2270.0, rel=0.035)


def test_qr_reflection_matches_target_qrd_at_evaluation_frequency() -> None:
    # The headline claim: the metadiffuser's spatially dependent reflection
    # reproduces the QRD designed for 500 Hz when evaluated at 2000 Hz
    # ("perfect agreement", Fig. 3(a)). The QRD wells are s_n lambda0 / 2N.
    c0 = 343.0
    lam0 = c0 / 500.0
    depths = np.array([s * lam0 / (2 * 5) for s in QR_SEQUENCE])
    k = 2.0 * np.pi * 2000.0 / c0
    target = np.exp(-2j * k * depths)
    result = metadiffuser_reflection(
        np.array([2000.0]), QR_WELLS, depth=0.02, period=0.07
    )
    mismatch = np.degrees(
        np.abs(np.angle(result.reflection[:, 0] * np.conj(target)))
    )
    assert float(mismatch.max()) < 10.0


def test_qr_panel_diffuses_like_the_supplementary_says() -> None:
    # Supplementary Table 1: nominal normalized diffusion ~0.54 at the
    # 2 kHz evaluation. The polar reduction differs in discretisation from
    # the paper's, so the bound is soft.
    spectrum = metadiffuser_diffusion_spectrum(
        np.array([2000.0]), QR_WELLS, depth=0.02, period=0.07
    )
    assert 0.4 < float(spectrum.normalized[0]) < 0.8


def test_pa_state_is_a_perfect_absorber_at_500_hz() -> None:
    # Table 3 zero state: critical coupling at the 500 Hz design point.
    f = np.arange(420.0, 601.0, 2.0)
    result = metadiffuser_reflection(
        f, [ABSORBER, ABSORBER], depth=0.03, period=0.10
    )
    alpha = result.well_absorption[0]
    peak = int(np.argmax(alpha))
    assert alpha[peak] > 0.99
    assert f[peak] == pytest.approx(500.0, rel=0.02)


def test_inverter_state_reflects_out_of_phase() -> None:
    # Table 3 [-1] state: nearly full-magnitude reflection well beyond
    # quadrature at the design frequency (the paper itself reports the
    # inverting slits as imperfect due to the thermo-viscous losses).
    result = metadiffuser_reflection(
        np.array([500.0]), [INVERTER, INVERTER], depth=0.03, period=0.10
    )
    r = result.reflection[0, 0]
    assert abs(r) > 0.9
    assert abs(np.degrees(np.angle(r))) > 110.0


def test_pr_metadiffuser_sharp_absorption_peak() -> None:
    # Fig. 5(d): one slit of the PR-metadiffuser shows a sharp absorption
    # peak at 1510 Hz (quasi-perfect, not critically coupled).
    f = np.arange(1300.0, 1701.0, 5.0)
    result = metadiffuser_reflection(f, PR_WELLS, depth=0.035, period=0.07)
    per_slit_peak = result.well_absorption.max(axis=1)
    best = int(np.argmax(per_slit_peak))
    assert per_slit_peak[best] > 0.8
    f_best = f[int(np.argmax(result.well_absorption[best]))]
    assert f_best == pytest.approx(1510.0, rel=0.02)


def test_pr_metadiffuser_specular_notch() -> None:
    # The PRD-like scattered field presents a notch at the specular
    # direction (Fig. 4(g), evaluated with 6 repetitions at 1 kHz).
    polar = metadiffuser_polar_response(
        1000.0, PR_WELLS, depth=0.035, period=0.07, periods=6
    )
    angles = np.asarray(polar.angles)
    specular = float(polar.levels[np.abs(angles) < 3.0].mean())
    assert specular < -15.0


def test_ternary_sequence_suppresses_the_specular_beam() -> None:
    # Fig. 6: the [1, -1, -1, 0, -1, 1, 1, 0] sequence balances in-phase
    # and inverted wells, so the specular direction is no longer the peak.
    wells = [None, INVERTER, INVERTER, ABSORBER,
             INVERTER, None, None, ABSORBER]
    polar = metadiffuser_polar_response(
        500.0, wells, depth=0.03, period=0.10, periods=6,
        angles=np.arange(-90.0, 91.0, 1.0),
    )
    angles = np.asarray(polar.angles)
    specular = float(polar.levels[angles == 0.0][0])
    off_peak = float(polar.levels[np.abs(angles) > 2.0].max())
    assert specular < off_peak - 2.0


def test_broadband_panel_soft_bounds() -> None:
    # Table 4 with the supplementary's nominal diffusion at 1 kHz (0.65).
    # Several optimised necks are wider than their slits (outside the
    # Dubos-fit domain), so only the mid-band value is pinned, softly.
    wells = [_well(*row) for row in BROADBAND_ROWS]
    with pytest.warns(SlowSoundAbsorberWarning):
        spectrum = metadiffuser_diffusion_spectrum(
            np.array([1000.0]), wells, depth=0.03, period=0.12
        )
    assert 0.4 < float(spectrum.normalized[0]) < 0.8


def test_face_average_and_flat_strips() -> None:
    # A None well is a rigid strip with R = 1 exactly, and the
    # face-averaged absorption is the mean of the per-well coefficients.
    f = np.array([500.0, 1000.0])
    result = metadiffuser_reflection(
        f, [ABSORBER, None], depth=0.03, period=0.10
    )
    assert np.allclose(result.reflection[1], 1.0)
    assert np.allclose(
        result.absorption, result.well_absorption.mean(axis=0)
    )
    assert result.depth == pytest.approx(0.03)
    assert result.period == pytest.approx(0.10)


def test_panel_validation() -> None:
    f = np.array([500.0])
    with pytest.raises(ValueError, match="at least two wells"):
        metadiffuser_reflection(f, [ABSORBER], depth=0.03, period=0.10)
    not_a_well = [ABSORBER, 0.01]
    with pytest.raises(TypeError, match="MetadiffuserWell"):
        metadiffuser_reflection(f, not_a_well, depth=0.03, period=0.10)
    too_tall = [ABSORBER, _well(110.0, 5.0, 20.0, 4.0, 20.0)]
    with pytest.raises(ValueError, match="smaller than the period"):
        metadiffuser_reflection(f, too_tall, depth=0.03, period=0.10)
    empty_band = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        metadiffuser_diffusion_spectrum(
            empty_band, [ABSORBER, None], depth=0.03, period=0.10
        )
    with pytest.raises(ValueError, match="at least one resonator"):
        MetadiffuserWell(0.01, ())


def test_square_resonator_geometry_and_validation() -> None:
    # The square-duct variant runs end to end, and the geometry switch
    # validates its inputs.
    f = np.array([500.0, 1000.0])
    result = metadiffuser_reflection(
        f, [ABSORBER, None], depth=0.03, period=0.10,
        resonator_geometry="square",
    )
    assert result.reflection.shape == (2, 2)
    with pytest.raises(ValueError, match="geometry"):
        metadiffuser_reflection(
            f, [ABSORBER, None], depth=0.03, period=0.10,
            resonator_geometry="round",
        )


def test_result_plots_render_and_validate() -> None:
    # Smoke pass through both renderers, in both languages, plus the
    # retained-geometry guard of the drawing.
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from phonometry.materials.metadiffuser import MetadiffuserResult

    f = np.arange(400.0, 601.0, 50.0)
    result = metadiffuser_reflection(
        f, [ABSORBER, INVERTER], depth=0.03, period=0.10
    )
    for language in ("en", "es"):
        ax = result.plot(language=language)
        assert ax.get_lines()
        plt.close(ax.figure)
        ax = result.plot_geometry(language=language)
        assert ax.patches
        plt.close(ax.figure)
    bare = MetadiffuserResult(
        frequency=f,
        reflection=result.reflection,
        absorption=result.absorption,
        well_absorption=result.well_absorption,
    )
    with pytest.raises(ValueError, match="retain"):
        bare.plot_geometry()
    plt.close("all")
