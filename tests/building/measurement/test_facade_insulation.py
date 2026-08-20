#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for ISO 16283-3:2016 field façade sound insulation and its ISO 717-1
weighted rating.

Validation strategy: the standard's own formulas and closed-form
identities, plus reuse of the already-verified ISO 717-1 engine. The field
quantities are defined by unnumbered formulas inline in the Clause 3 terms.

- ``D2m = L1,2m - L2`` (Clause 3.14); ``Dls,2m,nT = D2m + 10 lg(T/T0)``
  reduces to ``D2m`` at ``T = T0 = 0,5 s`` (Clause 3.15); ``Dls,2m,n =
  D2m - 10 lg(A/A0)`` with ``A = 0,16 V/T``, ``A0 = 10 m²`` reduces to
  ``D2m`` when ``A = A0`` (Clause 3.16, 3.17).
- ``R'45° = L1,s - L2 + 10 lg(S/A) - 1,5`` (Clause 3.12, loudspeaker) and
  ``R'tr,s = ... - 3`` (Clause 3.13, road traffic); with ``S = A`` the
  ``10 lg`` term vanishes so only the incidence correction remains.
- Positions are energy-averaged (Clause 9.5.1, Formula 7), reusing the
  16283-1 helper.
- The single-number rating goes through the ISO 717-1 airborne
  ``weighted_rating`` unchanged (Annex F), reproducing a known ``Rw``.
"""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import building

# ISO 717-1 Annex C Table C.1 measured curve; rated Rw = 30 (-2; -3).
_ANNEX_C_R = [
    20.4,
    16.3,
    17.7,
    22.6,
    22.4,
    22.7,
    24.8,
    26.6,
    28.0,
    30.5,
    31.8,
    32.5,
    33.4,
    33.0,
    31.0,
    25.5,
]
_CORE_FREQS = [
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
]


def _flat(n: int, value: float) -> np.ndarray:
    return np.full(n, value, dtype=float)


# --------------------------------------------------------------------------
# Global loudspeaker method: D2m family (Clauses 3.14-3.17)
# --------------------------------------------------------------------------
def test_d2m_is_level_difference() -> None:
    """D2m = L1,2m - L2 per band (Clause 3.14)."""
    l1 = np.array([80.0, 78.0, 76.0])
    l2 = np.array([40.0, 41.0, 39.0])
    res = building.facade_insulation(l1, l2, _flat(3, 0.5))
    assert isinstance(res, building.FacadeInsulationResult)
    np.testing.assert_allclose(res.d_2m, l1 - l2)


def test_dnt_reduces_to_d_at_reference_time() -> None:
    """Dls,2m,nT = D2m when T = T0 = 0,5 s (Clause 3.15)."""
    l1 = _flat(3, 75.0)
    l2 = _flat(3, 35.0)
    res = building.facade_insulation(l1, l2, _flat(3, 0.5))
    np.testing.assert_allclose(res.d_2m_nt, res.d_2m)


def test_dnt_standardization_term() -> None:
    """Dls,2m,nT = D2m + 10 lg(T/T0) for T != T0."""
    l1 = _flat(2, 70.0)
    l2 = _flat(2, 30.0)
    t = np.array([1.0, 0.25])
    res = building.facade_insulation(l1, l2, t)
    expected = (l1 - l2) + 10.0 * np.log10(t / 0.5)
    np.testing.assert_allclose(res.d_2m_nt, expected)


def test_d2m_n_reduces_to_d_when_absorption_equals_reference() -> None:
    """Dls,2m,n = D2m when A = 0,16 V/T = A0 = 10 m² (Clause 3.16, 3.17)."""
    # 0,16 * 62,5 / 1,0 = 10,0 = A0.
    res = building.facade_insulation(
        _flat(3, 72.0), _flat(3, 32.0), _flat(3, 1.0), volume=62.5
    )
    assert res.d_2m_n is not None
    np.testing.assert_allclose(res.d_2m_n, res.d_2m)


def test_d2m_n_none_without_volume() -> None:
    res = building.facade_insulation(_flat(3, 72.0), _flat(3, 32.0), _flat(3, 0.5))
    assert res.d_2m_n is None
    assert res.r_prime is None


# --------------------------------------------------------------------------
# Element method: apparent sound reduction index (Clauses 3.12, 3.13)
# --------------------------------------------------------------------------
def test_r45_loudspeaker_correction() -> None:
    """R'45° = L1,s - L2 + 10 lg(S/A) - 1,5; with S = A only -1,5 remains."""
    n = 3
    # A = 0,16 * 62,5 / 1,0 = 10; pick S = 10 so 10 lg(S/A) = 0.
    surf = _flat(n, 60.0)
    l2 = _flat(n, 20.0)
    res = building.facade_insulation(
        _flat(n, 55.0),
        l2,
        _flat(n, 1.0),
        area=10.0,
        volume=62.5,
        surface_level=surf,
    )
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, surf - l2 - 1.5)


def test_rtrs_road_traffic_correction() -> None:
    """R'tr,s uses a -3 dB correction (Clause 3.13)."""
    n = 3
    surf = _flat(n, 60.0)
    l2 = _flat(n, 20.0)
    res = building.facade_insulation(
        _flat(n, 55.0),
        l2,
        _flat(n, 1.0),
        area=10.0,
        volume=62.5,
        surface_level=surf,
        method="road_traffic",
    )
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, surf - l2 - 3.0)


def test_r45_full_formula_with_absorption() -> None:
    """R'45° with a non-trivial 10 lg(S/A) term (Clause 3.12, 3.17)."""
    surf = np.array([65.0, 63.0])
    l2 = np.array([25.0, 24.0])
    t = np.array([0.8, 0.8])
    area, volume = 12.0, 50.0
    a = 0.16 * volume / t
    res = building.facade_insulation(
        np.array([50.0, 50.0]),
        l2,
        t,
        area=area,
        volume=volume,
        surface_level=surf,
    )
    expected = surf - l2 + 10.0 * np.log10(area / a) - 1.5
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, expected)


def test_r_prime_needs_surface_area_and_volume() -> None:
    # surface_level but no area/volume -> no R'.
    res = building.facade_insulation(
        _flat(3, 70.0),
        _flat(3, 30.0),
        _flat(3, 0.5),
        surface_level=_flat(3, 72.0),
    )
    assert res.r_prime is None


# --------------------------------------------------------------------------
# Energy averaging of microphone positions (Clause 9.5.1, Formula 7)
# --------------------------------------------------------------------------
def test_positions_are_energy_averaged() -> None:
    """2-D (positions x bands) inputs are energy-averaged (Clause 9.5.1, Formula 7)."""
    l1 = np.array([[80.0, 70.0], [86.0, 70.0]])  # two positions, two bands
    l2 = np.array([[40.0, 30.0], [40.0, 30.0]])
    res = building.facade_insulation(l1, l2, np.array([0.5, 0.5]))
    l1_avg = 10.0 * np.log10(np.mean(10.0 ** (l1 / 10.0), axis=0))
    np.testing.assert_allclose(res.d_2m, l1_avg - l2[0])


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_band_count_mismatch_raises() -> None:
    outdoor, indoor_two_bands, rt = _flat(3, 70.0), _flat(2, 30.0), _flat(3, 0.5)
    with pytest.raises(ValueError):
        building.facade_insulation(outdoor, indoor_two_bands, rt)


def test_nonpositive_reverberation_raises() -> None:
    outdoor, indoor = _flat(3, 70.0), _flat(3, 30.0)
    rt_with_zero = np.array([0.5, 0.0, 0.5])
    with pytest.raises(ValueError):
        building.facade_insulation(outdoor, indoor, rt_with_zero)


def test_nonpositive_area_volume_raises() -> None:
    outdoor, indoor, rt = _flat(3, 70.0), _flat(3, 30.0), _flat(3, 0.5)
    surface = _flat(3, 72.0)
    with pytest.raises(ValueError):
        building.facade_insulation(
            outdoor,
            indoor,
            rt,
            area=-1.0,
            volume=50.0,
            surface_level=surface,
        )


def test_surface_and_area_without_volume_raises() -> None:
    # surface_level + area but no volume: R' would silently be None, so raise
    # a clear error naming 'volume' as the missing input.
    outdoor, indoor, rt = _flat(3, 70.0), _flat(3, 30.0), _flat(3, 0.5)
    surface = _flat(3, 72.0)
    with pytest.raises(ValueError, match="volume"):
        building.facade_insulation(
            outdoor, indoor, rt, area=10.0, surface_level=surface
        )


def test_surface_area_and_volume_returns_r_prime() -> None:
    # The complete set of R' inputs still yields a value.
    res = building.facade_insulation(
        _flat(3, 70.0),
        _flat(3, 30.0),
        _flat(3, 0.5),
        area=10.0,
        volume=62.5,
        surface_level=_flat(3, 72.0),
    )
    assert res.r_prime is not None


def test_invalid_method_raises() -> None:
    outdoor, indoor, rt = _flat(3, 70.0), _flat(3, 30.0), _flat(3, 0.5)
    with pytest.raises(ValueError):
        building.facade_insulation(outdoor, indoor, rt, method="airplane")


def test_invalid_method_on_result_raises() -> None:
    # Direct construction must not bypass the method validation: an unknown
    # method would otherwise be rendered as the loudspeaker quantity R'45.
    d_2m = np.array([30.0])
    d_2m_nt = np.array([31.0])
    with pytest.raises(ValueError, match="'method' must be"):
        building.FacadeInsulationResult(
            d_2m=d_2m,
            d_2m_nt=d_2m_nt,
            d_2m_n=None,
            r_prime=None,
            method="typo",
        )


def test_nonfinite_raises() -> None:
    outdoor_with_nan = np.array([70.0, np.nan, 70.0])
    indoor, rt = _flat(3, 30.0), _flat(3, 0.5)
    with pytest.raises(ValueError):
        building.facade_insulation(outdoor_with_nan, indoor, rt)


def test_frequencies_length_mismatch_raises() -> None:
    # 'frequencies' shorter than the band count must fail clearly here rather
    # than deferring a confusing matplotlib shape error to plot().
    outdoor, indoor, rt = _flat(3, 70.0), _flat(3, 30.0), _flat(3, 0.5)
    with pytest.raises(ValueError, match="frequencies"):
        building.facade_insulation(outdoor, indoor, rt, frequencies=[125.0, 250.0])


# --------------------------------------------------------------------------
# Extended frequency range (Clause 5) still computes per band
# --------------------------------------------------------------------------
def test_extended_bands_supported() -> None:
    """50-5000 Hz (21 bands) may be supplied; all quantities per band."""
    n = 21
    res = building.facade_insulation(_flat(n, 70.0), _flat(n, 30.0), _flat(n, 0.5))
    assert res.d_2m.shape == (n,)
    np.testing.assert_allclose(res.d_2m, _flat(n, 40.0))


# --------------------------------------------------------------------------
# Single-number rating via ISO 717-1 airborne engine (Annex F)
# --------------------------------------------------------------------------
def test_rating_path_reproduces_known_rw() -> None:
    """R'45° fed to weighted_rating reproduces ISO 717-1 Annex C Rw=30."""
    ref = np.asarray(_ANNEX_C_R)
    # Build L1,s so that R'45° == _ANNEX_C_R with S = A (term 0), L2 = 0:
    # R' = L1,s - 0 - 1,5 => L1,s = R' + 1,5.
    surf = ref + 1.5
    res = building.facade_insulation(
        _flat(16, 50.0),
        _flat(16, 0.0),
        _flat(16, 1.0),
        area=10.0,
        volume=62.5,
        surface_level=surf,
    )
    assert res.r_prime is not None
    np.testing.assert_allclose(res.r_prime, ref, atol=1e-9)
    rating = building.weighted_rating(res.r_prime)
    assert rating.rating == 30
    assert rating.c == -2
    assert rating.ctr == -3


# --------------------------------------------------------------------------
# Result dataclass + plotting
# --------------------------------------------------------------------------
def test_result_is_frozen() -> None:
    res = building.facade_insulation(_flat(3, 70.0), _flat(3, 30.0), _flat(3, 0.5))
    with pytest.raises(AttributeError):
        res.d_2m = np.zeros(3)  # type: ignore[misc]


def test_plot_returns_axes_with_dnt_curve() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = building.facade_insulation(
        np.asarray(_ANNEX_C_R) + 40.0,
        _flat(16, 40.0),
        _flat(16, 0.5),
        volume=62.5,
        frequencies=_CORE_FREQS,
    )
    ax = res.plot()
    assert not isinstance(ax, np.ndarray)
    # The standardized level difference is drawn as the first line.
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res.d_2m_nt)
    plt.close("all")


def test_plot_forwards_kwargs() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = building.facade_insulation(_flat(16, 70.0), _flat(16, 30.0), _flat(16, 0.5))
    ax = res.plot(linewidth=2)
    assert ax.lines[0].get_linewidth() == 2.0
    plt.close("all")
