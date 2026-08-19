#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for orthotropic (ribbed / corrugated) panel sound insulation.

Oracles, all printed:

* **Vigran, Building Acoustics (2008)**, worked example following Eq. (3.115),
  printed p. 96 (pdf p. 118): a 1 mm steel plate 1 m x 1 m (``E = 2,1e11`` Pa,
  ``m'' = 7,8`` kg/m2, ``nu = 0,3``) has ``f_1,1 = 4,9`` Hz and
  ``f_2,2 = 19,7`` Hz flat, and ``f_1,1 = 25,5`` Hz, ``f_2,2 = 102`` Hz once
  corrugated with a sinusoid of total height 20 mm (``H = 10`` mm) and
  wavelength 100 mm. Four published values, and reproducing them requires the
  developed-length surface density Vigran warns about in the same paragraph.
* **Bies, Hansen & Howard 5e** Figure 7.9(b) and Eqs. (7.59)/(7.60), printed
  pp. 381 and 384: the design-chart constants -54, -13,2, -17 and -23 dB
  (stated for ``rho c = 414``). Bies Eqs. (7.30)/(7.31)/(7.38) are an
  independent transcription of Vigran Eqs. (6.108)/(6.109)/(6.111).
* **Hopkins, Sound Insulation (2007)** Table A2, printed p. 608 (pdf p. 635):
  the ``h.fc`` product of 25 building-material rows, assuming
  ``c0 = 343`` m/s: a multi-row digit oracle for the isotropic coincidence
  frequency the orthotropic pair generalises.
* **Closed-form identities**: Vigran's own statement (printed p. 95) that
  Eq. (3.113) collapses to the isotropic Eq. (3.109); the elementary integral
  the diffuse-field average of Eq. (6.111) reduces to in the mass-law region;
  the reduction of Eq. (6.108) to the isotropic Bies Eq. (7.29) as
  ``fc2 -> fc1``; and the developed length of a sinusoid.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import reference_data
from numpy.typing import ArrayLike
from scipy.integrate import quad

from phonometry import building, vibration

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: ISO 717-1 one-third-octave band centres, 100 Hz to 3150 Hz.
BANDS = np.array(
    [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
     1000, 1250, 1600, 2000, 2500, 3150], dtype=float
)

#: Vigran's worked example (printed p. 96).
VIGRAN_STEEL: dict[str, float] = {"youngs_modulus": 2.1e11, "poisson_ratio": 0.3}
VIGRAN_THICKNESS = 1.0e-3
VIGRAN_MASS = 7.8
VIGRAN_AMPLITUDE = 0.010
VIGRAN_WAVELENGTH = 0.100

#: Vigran Figure 6.27 (printed p. 254): a corrugated panel of 7,5 kg/m2 with
#: the coincidence range 400 Hz to 4000 Hz, the case the book integrates.
FIG627_FC1 = 400.0
FIG627_FC2 = 4000.0
FIG627_MASS = 7.5

#: Hopkins Table A2 (printed p. 608): the material names, paired positionally
#: with the ``(cL, h.fc)`` rows of ``reference_data.HOPKINS_TABLE_A2_H_FC``,
#: which is the single source of truth shared with the conformance report.
HOPKINS_TABLE_A2_NAMES: tuple[str, ...] = (
    "aircrete/AAC blocks (solid)",
    "aluminium",
    "bricks (solid)",
    "calcium-silicate blocks (solid)",
    "chipboard",
    "clinker concrete blocks, 1030 kg/m3",
    "clinker concrete blocks, 1720 kg/m3",
    "clinker concrete slabs",
    "concrete, cast in situ",
    "dense aggregate blocks (solid)",
    "expanded clay blocks (solid)",
    "glass",
    "lightweight aggregate blocks (solid)",
    "medium density fibreboard",
    "mortar",
    "oriented strand board",
    "perspex, plexiglass",
    "plaster, gypsum based",
    "plasterboard, natural gypsum",
    "plasterboard, flue gas plus natural gypsum",
    "plasterboard, gypsum with glass fibre",
    "plywood (birch)",
    "sand-cement screed",
    "steel",
    "timber (soft wood)",
)

#: Air density that makes ``rho0 c0`` the 414 Pa.s/m Bies quotes for his
#: design-chart constants.
BIES_AIR_DENSITY = 414.0 / 343.0


def _vigran_flat_stiffness() -> float:
    """``B'`` of Vigran's 1 mm flat steel plate."""
    return vibration.plate_bending_stiffness(
        VIGRAN_STEEL["youngs_modulus"], VIGRAN_THICKNESS,
        VIGRAN_STEEL["poisson_ratio"],
    )


def _vigran_corrugated() -> tuple[float, tuple[float, float, float]]:
    """``(m'', (Bx, Bz, Bxz))`` of Vigran's corrugated version."""
    mass = VIGRAN_MASS * building.corrugated_plate_mass_factor(
        VIGRAN_AMPLITUDE, VIGRAN_WAVELENGTH
    )
    stiffness = building.corrugated_plate_stiffness(
        VIGRAN_THICKNESS, VIGRAN_AMPLITUDE, VIGRAN_WAVELENGTH,
        youngs_modulus=VIGRAN_STEEL["youngs_modulus"],
        poisson_ratio=VIGRAN_STEEL["poisson_ratio"],
    )
    return mass, stiffness


# ---------------------------------------------------------------------------
# Eigenfrequencies and equivalent stiffnesses (Vigran 3.7.3.3).
# ---------------------------------------------------------------------------
def test_flat_plate_eigenfrequencies_match_vigran_example() -> None:
    """Vigran printed p. 96: ``f_1,1 = 4,9`` Hz and ``f_2,2 = 19,7`` Hz."""
    b = _vigran_flat_stiffness()
    kwargs: dict[str, float] = {
        "length_x": 1.0, "length_z": 1.0, "mass_per_area": VIGRAN_MASS,
        "bending_stiffness_x": b, "bending_stiffness_z": b,
        "bending_stiffness_xz": b,
    }
    assert building.orthotropic_plate_resonance(
        1, 1, **kwargs
    ) == pytest.approx(4.9, abs=0.05)
    assert building.orthotropic_plate_resonance(
        2, 2, **kwargs
    ) == pytest.approx(19.7, abs=0.05)


def test_corrugated_plate_eigenfrequencies_match_vigran_example() -> None:
    """Vigran printed p. 96: ``f_1,1 = 25,5`` Hz and ``f_2,2 = 102`` Hz.

    Reproducing the two published values needs the surface density to carry the
    developed length of the profile, which is exactly the caveat Vigran states
    in the same paragraph ("the mass per unit area will increase when making
    the corrugations"); with the flat 7,8 kg/m2 the model would return 26,7 Hz
    and 106,7 Hz instead.
    """
    mass, (b_x, b_z, b_xz) = _vigran_corrugated()
    kwargs: dict[str, float] = {
        "length_x": 1.0, "length_z": 1.0, "mass_per_area": mass,
        "bending_stiffness_x": b_x, "bending_stiffness_z": b_z,
        "bending_stiffness_xz": b_xz,
    }
    assert building.orthotropic_plate_resonance(
        1, 1, **kwargs
    ) == pytest.approx(25.5, abs=0.05)
    assert building.orthotropic_plate_resonance(
        2, 2, **kwargs
    ) == pytest.approx(102.0, abs=0.1)


def test_corrugated_stiffness_is_stiff_along_and_soft_across() -> None:
    """Vigran Eq. (3.115): ``Bz >> B'_flat > Bx``, the point of corrugating."""
    _mass, (b_x, b_z, _b_xz) = _vigran_corrugated()
    flat = _vigran_flat_stiffness()
    assert b_x < flat
    assert b_z > 100.0 * flat


def test_corrugated_stiffness_returns_to_the_flat_plate_when_flattened() -> None:
    """As ``H -> 0`` Eq. (3.115) gives back the flat plate.

    ``Bx -> E h^3/(12(1-nu^2))`` and ``Bxz -> E h^3/(12(1+nu))``, the pure
    twisting term: Timoshenko's corrugated ``Bxz`` carries no ``Bx nu_z``
    contribution, unlike the general Eq. (3.114).
    """
    b_x, b_z, b_xz = building.corrugated_plate_stiffness(
        VIGRAN_THICKNESS, 1.0e-9, VIGRAN_WAVELENGTH,
        youngs_modulus=VIGRAN_STEEL["youngs_modulus"],
        poisson_ratio=VIGRAN_STEEL["poisson_ratio"],
    )
    flat = _vigran_flat_stiffness()
    nu = VIGRAN_STEEL["poisson_ratio"]
    assert b_x == pytest.approx(flat, rel=1e-9)
    assert b_xz == pytest.approx(flat * (1.0 - nu), rel=1e-9)
    assert b_z == pytest.approx(0.0, abs=1e-6)


def test_corrugation_mass_factor_is_the_developed_length() -> None:
    """The factor is the arc length of one period divided by the period.

    Independent oracle: the same integral evaluated numerically.
    """
    q = 2.0 * math.pi * VIGRAN_AMPLITUDE / VIGRAN_WAVELENGTH
    numeric = quad(
        lambda t: math.sqrt(1.0 + q**2 * math.cos(t) ** 2), 0.0, 2.0 * math.pi
    )[0] / (2.0 * math.pi)
    assert building.corrugated_plate_mass_factor(
        VIGRAN_AMPLITUDE, VIGRAN_WAVELENGTH
    ) == pytest.approx(numeric, rel=1e-10)
    assert building.corrugated_plate_mass_factor(1e-12, 0.1) == pytest.approx(
        1.0, abs=1e-12
    )


def test_orthotropic_resonance_collapses_to_the_isotropic_formula() -> None:
    """Vigran printed p. 95: Eq. (3.113) simplifies to Eq. (3.109).

    In the isotropic case Eq. (3.114) gives ``Bxz = B' nu + 2 G h^3/12 = B'``,
    and Eq. (3.113) must then equal
    ``f_in = (pi/2) sqrt(B'/m'') [(i/a)^2 + (n/b)^2]``.
    """
    b, m2, a, c = 19.23, 7.8, 1.4, 0.9
    for i, n in ((1, 1), (2, 3), (4, 2)):
        expected = math.pi / 2.0 * math.sqrt(b / m2) * ((i / a) ** 2 + (n / c) ** 2)
        assert building.orthotropic_plate_resonance(
            i, n, length_x=a, length_z=c, mass_per_area=m2,
            bending_stiffness_x=b, bending_stiffness_z=b, bending_stiffness_xz=b,
        ) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Coincidence frequencies (Hopkins Table A2, Vigran Eq. 6.107).
# ---------------------------------------------------------------------------
def test_hopkins_table_a2_h_fc_products() -> None:
    """Hopkins Table A2 (printed p. 608): ``h.fc`` for 25 material rows.

    ``h fc = c0^2 sqrt(12) / (2 pi cL)`` is independent of density and Poisson
    ratio, so each tabulated quasi-longitudinal speed pins one product. The
    library route is ``plate_bending_stiffness`` (with
    ``E = rho cL^2 (1 - nu^2)``) into ``coincidence_frequency``.
    """
    rho, nu, h = 2500.0, 0.24, 0.01
    rows = reference_data.HOPKINS_TABLE_A2_H_FC
    assert len(rows) == len(HOPKINS_TABLE_A2_NAMES)
    for name, (c_l, product) in zip(HOPKINS_TABLE_A2_NAMES, rows, strict=True):
        b = vibration.plate_bending_stiffness(
            rho * c_l**2 * (1.0 - nu**2), h, nu
        )
        fc = vibration.coincidence_frequency(rho * h, b)
        assert h * fc == pytest.approx(product, abs=0.06), name


def test_orthotropic_critical_frequencies_bound_the_isotropic_one() -> None:
    """Vigran Eq. (6.107) with equal stiffnesses is the isotropic ``fc``."""
    b, m2 = 19.23, 7.8
    fc = vibration.coincidence_frequency(m2, b)
    assert building.orthotropic_critical_frequencies(
        m2, b, b
    ) == pytest.approx((fc, fc))
    # A hundredfold stiffer direction moves its coincidence frequency down by
    # a decade, and the pair comes back sorted whichever way it is given.
    low, high = building.orthotropic_critical_frequencies(m2, 100.0 * b, b)
    assert low == pytest.approx(fc / 10.0)
    assert high == pytest.approx(fc)
    assert building.orthotropic_critical_frequencies(m2, b, 100.0 * b) == (
        low,
        high,
    )


def test_corrugating_vigran_plate_moves_the_coincidence_range() -> None:
    """The corrugated version of Vigran's plate spans 1,16 kHz to 13,1 kHz.

    Both bounds follow from the published Eq. (3.115) stiffnesses and the
    developed-length mass; the flat sheet's single ``fc`` is 11,9 kHz, so
    corrugating drags the lower bound down by more than a decade, which is the
    mechanism Vigran Sect. 6.5.3 describes.
    """
    mass, (b_x, b_z, _b_xz) = _vigran_corrugated()
    fc1, fc2 = building.orthotropic_critical_frequencies(mass, b_x, b_z)
    flat_fc = vibration.coincidence_frequency(
        VIGRAN_MASS, _vigran_flat_stiffness()
    )
    assert flat_fc == pytest.approx(11925.0, rel=1e-3)
    assert fc1 == pytest.approx(1164.6, rel=1e-3)
    assert fc2 == pytest.approx(13064.0, rel=1e-3)
    assert fc1 < flat_fc < fc2


# ---------------------------------------------------------------------------
# Heckl's closed-form approximation (Bies 7.2.4.5 / Vigran Eq. 6.112).
# ---------------------------------------------------------------------------
def _fig627(
    freq: ArrayLike,
    *,
    method: str = "integral",
    loss_factor: float = 0.01,
    area: float | None = None,
    limiting_angle: float = 78.0,
    air_density: float = 1.205,
) -> building.SoundReductionResult:
    """The Figure 6.27 panel: 7,5 kg/m2 with the range 400 Hz to 4000 Hz."""
    return building.orthotropic_transmission_loss(
        freq, FIG627_MASS,
        critical_frequency_lower=FIG627_FC1, critical_frequency_upper=FIG627_FC2,
        method=method, loss_factor=loss_factor, area=area,
        limiting_angle=limiting_angle, air_density=air_density,
    )


def _heckl(freq: np.ndarray, air_density: float = 1.205) -> np.ndarray:
    res = _fig627(freq, method="heckl", air_density=air_density)
    return np.asarray(res.transmission_loss, dtype=np.float64)


def test_heckl_coincidence_branch_matches_bies_equation_759() -> None:
    """Bies Eq. (7.59): the printed constant is -13,2 dB for ``rho c = 414``.

    ``TL = 20 lg f + 10 lg m'' - 10 lg fc1 - 20 lg(ln(4 f / fc1)) - 13.2``, so
    subtracting the four leading terms from the computed TL must leave the
    printed constant; its exact value is ``10 lg(2 pi^2 / rho c) = -13,216``.
    """
    f = np.array([400.0, 800.0, 1600.0, 2000.0])
    constant = (
        _heckl(f, BIES_AIR_DENSITY)
        - 20.0 * np.log10(f)
        - 10.0 * math.log10(FIG627_MASS)
        + 10.0 * math.log10(FIG627_FC1)
        + 20.0 * np.log10(np.log(4.0 * f / FIG627_FC1))
    )
    assert np.allclose(constant, -13.2, atol=0.02)
    assert float(constant[0]) == pytest.approx(
        10.0 * math.log10(2.0 * math.pi**2 / 414.0), abs=1e-6
    )


def test_heckl_recovery_branch_matches_bies_equation_760() -> None:
    """Bies Eq. (7.60): the printed constant is -23 dB for ``rho c = 414``.

    ``TL = 20 lg f + 10 lg m'' - 5 lg fc1 - 5 lg fc2 - 23`` above ``2 fc2``;
    its exact value is ``10 lg(2 / rho c) = -23,15``.
    """
    f = np.array([8000.0, 12500.0, 20000.0])
    constant = (
        _heckl(f, BIES_AIR_DENSITY)
        - 20.0 * np.log10(f)
        - 10.0 * math.log10(FIG627_MASS)
        + 5.0 * math.log10(FIG627_FC1)
        + 5.0 * math.log10(FIG627_FC2)
    )
    assert np.allclose(constant, -23.0, atol=0.2)
    assert float(constant[0]) == pytest.approx(
        10.0 * math.log10(2.0 / 414.0), abs=1e-6
    )


def test_heckl_design_chart_points_a_and_d() -> None:
    """Bies Figure 7.9(b): point A ``20 lg(fc1 m'') - 54`` and point D.

    Point A sits at ``fc1/2`` on the field-incidence mass law; point D at
    ``2 fc2`` on Eq. (7.60), printed as
    ``TL = 10 lg m'' + 15 lg fc2 - 5 lg fc1 - 17``. Both printed constants are
    rounded, so 0,15 dB is the honest tolerance.
    """
    fc1 = FIG627_FC1
    fc2 = FIG627_FC2
    tl = _heckl(np.array([0.5 * fc1, 2.0 * fc2]), BIES_AIR_DENSITY)
    point_a = 20.0 * math.log10(fc1 * FIG627_MASS) - 54.0
    point_d = (
        10.0 * math.log10(FIG627_MASS) + 15.0 * math.log10(fc2)
        - 5.0 * math.log10(fc1) - 17.0
    )
    assert float(tl[0]) == pytest.approx(point_a, abs=0.15)
    assert float(tl[1]) == pytest.approx(point_d, abs=0.15)


def test_heckl_construction_is_continuous_at_its_four_knots() -> None:
    """The two straight-line bridges of Figure 7.9(b) must join up."""
    knots = np.array([200.0, 400.0, 2000.0, 8000.0])
    below = _heckl(knots * (1.0 - 1e-9))
    above = _heckl(knots * (1.0 + 1e-9))
    assert np.allclose(below, above, atol=1e-6)


def test_heckl_needs_a_wide_coincidence_range() -> None:
    with pytest.raises(ValueError, match="four times"):
        building.orthotropic_transmission_loss(
            BANDS, FIG627_MASS, critical_frequency_lower=400.0,
            critical_frequency_upper=1000.0, method="heckl",
        )


# ---------------------------------------------------------------------------
# The diffuse-field integral (Vigran Eq. 6.111 = Bies Eq. 7.38).
# ---------------------------------------------------------------------------
def test_orthotropic_integral_reduces_to_the_exact_mass_law_integral() -> None:
    """Well below ``fc1`` the impedance is ``j w m''`` and Eq. (6.111) is exact.

    The azimuth then drops out and, writing ``q = w m''/(2 Z0)`` and
    ``x = sin^2(phi)``, the double integral collapses to the elementary
    ``tau = ln((1 + q^2)/(1 + q^2 (1 - u))) / q^2`` with ``u = sin^2(theta_L)``.
    """
    m2, f, angle = FIG627_MASS, 50.0, 78.0
    res = building.orthotropic_transmission_loss(
        [f], m2, critical_frequency_lower=4.0e5,
        critical_frequency_upper=4.0e6, limiting_angle=angle,
    )
    z0 = 1.205 * 343.0
    q = 2.0 * math.pi * f * m2 / (2.0 * z0)
    u = math.sin(math.radians(angle)) ** 2
    tau = math.log((1.0 + q**2) / (1.0 + q**2 * (1.0 - u))) / q**2
    assert float(res.transmission_loss[0]) == pytest.approx(
        -10.0 * math.log10(tau), abs=1e-6
    )


def test_orthotropic_integral_reduces_to_the_isotropic_integral() -> None:
    """With ``fc2 -> fc1`` Eq. (6.108) becomes the isotropic Bies Eq. (7.29).

    Independent oracle: the single integral of Bies Eq. (7.37) with
    ``Z = j w m'' [1 - (f/fc)^2 (1 + j eta) sin^4(phi)]``, evaluated here.
    """
    m2, fc, eta, angle = FIG627_MASS, 400.0, 0.05, 78.0
    z0 = 1.205 * 343.0
    u = math.sin(math.radians(angle)) ** 2
    for f in (250.0, 500.0, 1000.0, 4000.0):
        def integrand(x: float, f: float = f) -> float:
            z_w = 1j * 2.0 * math.pi * f * m2 * (
                1.0 - (f / fc) ** 2 * (1.0 + 1j * eta) * x * x
            )
            return 1.0 / abs(1.0 + z_w * math.sqrt(1.0 - x) / (2.0 * z0)) ** 2

        expected = -10.0 * math.log10(quad(integrand, 0.0, u, limit=200)[0])
        res = building.orthotropic_transmission_loss(
            [f], m2, critical_frequency_lower=fc,
            critical_frequency_upper=fc * (1.0 + 1e-12),
            loss_factor=eta, limiting_angle=angle,
        )
        assert float(res.transmission_loss[0]) == pytest.approx(expected, abs=1e-4)


def test_undamped_integral_meets_heckls_recovery_branch_above_2fc2() -> None:
    """Above ``2 fc2`` the two published routes must agree as ``eta -> 0``.

    Bies Eq. (7.60) is derived from Eq. (7.38) for ``eta = 0``, so the
    numerically averaged integral has to converge on it once the damping is
    removed. This is a cross-check of the two transcriptions against each
    other: each decade of ``eta`` closes the gap until it settles about 0,3 dB
    below the closed form, which is the accuracy of the approximation itself.
    """
    f = 5.0 * FIG627_FC2
    heckl = float(_fig627([f], method="heckl").transmission_loss[0])
    damped, less, undamped = (
        abs(float(_fig627([f], loss_factor=eta).transmission_loss[0]) - heckl)
        for eta in (1e-2, 1e-3, 1e-4)
    )
    assert damped > less > undamped
    assert undamped < 0.3
    for eta in (1e-5, 1e-6):
        gap = abs(float(_fig627([f], loss_factor=eta).transmission_loss[0]) - heckl)
        assert gap < 0.3


def test_orthotropic_integral_flattens_below_the_flat_plate() -> None:
    """Vigran Sect. 6.5.3: ``R`` collapses across the coincidence range.

    Vigran's own corrugated plate against a flat sheet of the same material and
    thickness: within 2 dB below ``fc1``, then more than 8 dB worse across the
    coincidence range, which is the trade Sect. 6.5.3 describes.
    """
    bands = np.array([125.0, 250.0, 500.0, 2000.0, 4000.0])
    mass, (b_x, b_z, _b_xz) = _vigran_corrugated()
    fc1, fc2 = building.orthotropic_critical_frequencies(mass, b_x, b_z)
    flat = building.single_panel_transmission_loss(
        bands, VIGRAN_MASS,
        critical_frequency=vibration.coincidence_frequency(
            VIGRAN_MASS, _vigran_flat_stiffness()
        ),
        loss_factor=0.011,
    ).transmission_loss
    corrugated = building.orthotropic_transmission_loss(
        bands, mass, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, loss_factor=0.011,
    ).transmission_loss
    assert np.all(np.abs(corrugated[:3] - flat[:3]) < 2.0)
    assert np.all(flat[3:] - corrugated[3:] > 8.0)


def test_orthotropic_damping_lifts_the_coincidence_range_only() -> None:
    """More damping only helps where the resonant transmission dominates."""
    bands = np.array([100.0, 1000.0])
    light = _fig627(bands, loss_factor=0.01)
    heavy = _fig627(bands, loss_factor=0.1)
    delta = heavy.transmission_loss - light.transmission_loss
    assert float(delta[0]) == pytest.approx(0.0, abs=0.05)
    assert float(delta[1]) > 3.0


def test_orthotropic_area_limits_the_incidence_angle() -> None:
    """Bies Eq. (7.36): ``cos^2(theta_L) = min(lambda/(2 pi sqrt(A)), 0.9)``.

    A 10 m2 specimen at 1 kHz gives ``cos^2(theta_L) = 0,0173``, i.e.
    ``theta_L = 82,4 deg``; the clamp at 0,9 bites only when the wavelength is
    large compared with the panel. Both routes must agree with the equivalent
    fixed angle.
    """
    area, f = 10.0, 1000.0
    cos2 = (343.0 / f) / (2.0 * math.pi * math.sqrt(area))
    sized = _fig627([f], area=area)
    fixed = _fig627(
        [f], limiting_angle=math.degrees(math.asin(math.sqrt(1.0 - cos2)))
    )
    assert float(sized.transmission_loss[0]) == pytest.approx(
        float(fixed.transmission_loss[0]), abs=1e-6
    )
    clamped = _fig627([50.0], area=0.05)
    capped = _fig627(
        [50.0], limiting_angle=math.degrees(math.asin(math.sqrt(0.1)))
    )
    assert float(clamped.transmission_loss[0]) == pytest.approx(
        float(capped.transmission_loss[0]), abs=1e-6
    )


def test_orthotropic_result_carries_the_coincidence_range() -> None:
    res = _fig627(BANDS)
    assert isinstance(res, building.SoundReductionResult)
    assert res.model == "orthotropic-integral"
    assert res.critical_frequency == FIG627_FC1
    assert res.critical_frequency_upper == FIG627_FC2
    assert np.all(res.transmission_coefficient <= 1.0)
    assert res.rating().rating > 0
    heckl = _fig627(BANDS, method="heckl")
    assert heckl.model == "orthotropic-heckl"
    assert heckl.critical_frequency_upper == FIG627_FC2


def test_orthotropic_plot_shades_the_coincidence_range() -> None:
    """``.plot()`` marks both bounds and shades the band between them.

    The isotropic result labels its single dip ``f_c``; the orthotropic one
    has to label the pair ``f_c1`` and ``f_c2`` and shade what lies between,
    in either language.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _legend(axes: Axes) -> list[str]:
        legend = axes.get_legend()
        assert legend is not None
        return [t.get_text() for t in legend.get_texts()]

    res = _fig627(BANDS)
    for language, opener in (
        ("en", "coincidence range"),
        ("es", "rango de coincidencia"),
    ):
        ax = res.plot(language=language)
        labels = _legend(ax)
        assert any(opener in label for label in labels), labels
        assert any("f_{c1}" in label for label in labels), labels
        assert any("f_{c2}" in label for label in labels), labels
        assert ax.patches, "the coincidence range is not shaded"
        plt.close("all")

    # The isotropic result keeps its single unsubscripted f_c and no shading.
    flat = building.single_panel_transmission_loss(
        BANDS, FIG627_MASS, critical_frequency=FIG627_FC1
    )
    labels = _legend(flat.plot())
    assert any(label.startswith("$f_\\mathrm{c}$") for label in labels), labels
    plt.close("all")


def test_orthotropic_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        building.orthotropic_transmission_loss(
            BANDS, FIG627_MASS, critical_frequency_lower=4000.0,
            critical_frequency_upper=400.0,
        )
    with pytest.raises(ValueError):
        _fig627(BANDS, method="heckle")
    with pytest.raises(ValueError, match="must be positive"):
        building.orthotropic_transmission_loss(
            BANDS, -1.0, critical_frequency_lower=FIG627_FC1,
            critical_frequency_upper=FIG627_FC2,
        )
    with pytest.raises(ValueError, match="limiting_angle"):
        _fig627(BANDS, limiting_angle=95.0)
    with pytest.raises(ValueError, match="area"):
        _fig627(BANDS, area=0.0)


@pytest.mark.parametrize("method", ["integral", "heckl"])
def test_orthotropic_validates_every_argument_on_both_routes(method: str) -> None:
    """Both routes reject the same out-of-range arguments.

    ``method="heckl"`` uses neither *loss_factor*, *area* nor *limiting_angle*,
    but silently accepting a negative area or a 170 degree limiting angle on
    one route while raising on the other is a trap for the caller, so all three
    are validated before the method branch.
    """
    with pytest.raises(ValueError, match="area"):
        _fig627(BANDS, method=method, area=-5.0)
    with pytest.raises(ValueError, match="limiting_angle"):
        _fig627(BANDS, method=method, limiting_angle=170.0)
    with pytest.raises(ValueError, match="loss_factor"):
        _fig627(BANDS, method=method, loss_factor=-0.01)


def test_orthotropic_helpers_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        building.corrugated_plate_mass_factor(-0.01, 0.1)
    with pytest.raises(ValueError, match="poisson_ratio"):
        building.corrugated_plate_stiffness(
            1e-3, 0.01, 0.1, youngs_modulus=2.1e11, poisson_ratio=1.5
        )
    with pytest.raises(ValueError, match="must be integers"):
        building.orthotropic_plate_resonance(
            0, 1, length_x=1.0, length_z=1.0, mass_per_area=7.8,
            bending_stiffness_x=1.0, bending_stiffness_z=1.0,
            bending_stiffness_xz=1.0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        building.orthotropic_critical_frequencies(7.8, 0.0, 1.0)
