#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the wall-tie bridge across a masonry cavity wall.

Oracles, with the source reference in each test docstring:

* **Hopkins (2007) "Sound Insulation", Table A4** (printed p. 610) for the
  dynamic stiffness of four wall ties, corroborated by **Hopkins, Wilson &
  Craik (1999)**, *Applied Acoustics* **58**, 51-68, Table 1 (printed p. 64)
  for the three 50 mm rows.
* **Hopkins Fig. 4.35** (printed p. 468) for the mass-spring-mass resonance of
  a fully specified masonry cavity wall with and without ties: the caption
  prints ``fmsm = 26 Hz`` for the untied wall and ``fmsm = 50 Hz`` once
  2,5 ties/m2 of ``k = 2 x 10**6 N/m`` are added, and cross-checks the same
  plate with ``fc = 295 Hz`` and the first cross-cavity mode at 2287 Hz.
* **Hopkins Eqs. 4.87, 4.88 and 4.89** and **Eq. 2.190** for the closed forms.

The per-band transmission-loss penalty of a bridged cavity wall has **no
printed numeric oracle**: Hopkins Fig. 4.35, Craik & Wilson (1995) Figs. 10-18
and Hopkins Fig. 5.30 are all figures. The coupling-loss-factor tests below are
therefore closed-form identities and limits, not regressions on measured data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from phonometry import building
from phonometry.fluids import Fluid

# ---------------------------------------------------------------------------
# Printed oracles
# ---------------------------------------------------------------------------

#: Hopkins (2007) Table A4, dynamic stiffness of wall ties: (cavity width in
#: mm, sX mm in MN/m).
HOPKINS_TABLE_A4 = {
    "butterfly": (50, 1.7),
    "double_triangle": (50, 16.1),
    "vertical_twist": (50, 94.0),
    "vertical_twist_100mm": (100, 43.4),
}

#: Hopkins Fig. 4.35 (printed p. 468), SEA model of a masonry cavity wall.
#: Leaves 4 m x 2,5 m, h = 0,1 m, rho_s = 140 kg/m2, cL = 2200 m/s, nu = 0,2;
#: empty cavity Lz = 0,075 m; wall ties 2,5 per m2 with k = s_75mm = 2e6 N/m.
FIG_4_35_SURFACE_DENSITY = 140.0
FIG_4_35_CAVITY_DEPTH = 0.075
FIG_4_35_TIES_PER_AREA = 2.5
FIG_4_35_TIE_STIFFNESS = 2.0e6
FIG_4_35_FMSM_UNTIED = 26.0
FIG_4_35_FMSM_TIED = 50.0

#: Hopkins uses rho0 c0**2 = 1,42e5 Pa for the cavity air stiffness, quoted as
#: such throughout Chapter 4; expressed as the density that goes with
#: c0 = 343 m/s so it can be handed to the library's air-stiffness term.
HOPKINS_AIR_DENSITY = 1.42e5 / 343.0**2

#: The same, as the `Fluid` the models take.
HOPKINS_AIR = Fluid(
    temperature_c=20.0,
    static_pressure_pa=101_325.0,
    composition={},
    model="Hopkins, rho0 c0^2 = 1,42e5 Pa at c0 = 343 m/s",
    validity="",
    properties={"speed_of_sound": 343.0, "density": HOPKINS_AIR_DENSITY},
)


# ---------------------------------------------------------------------------
# Table A4
# ---------------------------------------------------------------------------


def test_wall_tie_table() -> None:
    """Hopkins (2007) Table A4, digit for digit, converted to SI base units."""
    assert set(building.WALL_TIE_STIFFNESS) == set(HOPKINS_TABLE_A4)
    for name, (cavity_mm, stiffness_mn) in HOPKINS_TABLE_A4.items():
        cavity, stiffness = building.wall_tie_stiffness(name)
        assert cavity == pytest.approx(cavity_mm / 1000.0)
        assert stiffness == pytest.approx(stiffness_mn * 1e6)


def test_wall_tie_stiffness_ordering() -> None:
    """Table A4 spans nearly two decades at the same 50 mm cavity.

    Butterfly 1,7 MN/m, double-triangle 16,1 MN/m and vertical-twist 94,0 MN/m
    are the same three values that Hopkins, Wilson & Craik (1999) Table 1
    prints for a 50 mm spacing, and they order butterfly < double-triangle <
    vertical-twist by a factor of about 55 end to end.
    """
    butterfly = building.wall_tie_stiffness("butterfly")[1]
    triangle = building.wall_tie_stiffness("double_triangle")[1]
    twist = building.wall_tie_stiffness("vertical_twist")[1]
    assert butterfly < triangle < twist
    assert twist / butterfly == pytest.approx(94.0 / 1.7)


def test_unknown_tie_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown wall tie"):
        building.wall_tie_stiffness("staple")


def test_stiffness_per_area_from_a_named_tie() -> None:
    """Eq. 4.89's N k / S term: 2,5 vertical-twist ties per m2 at 100 mm."""
    per_area = building.wall_tie_stiffness_per_area(2.5, "vertical_twist_100mm")
    assert per_area == pytest.approx(2.5 * 43.4e6)


def test_stiffness_per_area_accepts_an_explicit_stiffness() -> None:
    """Hopkins Fig. 4.35 uses s_75mm = 2e6 N/m, which is not in Table A4."""
    assert building.wall_tie_stiffness_per_area(
        FIG_4_35_TIES_PER_AREA, FIG_4_35_TIE_STIFFNESS
    ) == pytest.approx(5.0e6)


# ---------------------------------------------------------------------------
# Mass-spring-mass resonance with ties (Hopkins Eq. 4.89)
# ---------------------------------------------------------------------------


def test_fig_4_35_untied_resonance() -> None:
    """Hopkins Fig. 4.35, scenario A (no ties): the caption prints fmsm = 26 Hz.

    Two 140 kg/m2 leaves across an empty 75 mm cavity, air stiffness
    rho0 c0**2 / d = 1,42e5 / 0,075 = 1,893e6 N/m3 over the equivalent surface
    density 140 x 140 / 280 = 70 kg/m2.
    """
    f0 = building.mass_spring_mass_resonance(
        FIG_4_35_SURFACE_DENSITY,
        FIG_4_35_SURFACE_DENSITY,
        FIG_4_35_CAVITY_DEPTH,
        fluid=HOPKINS_AIR,
    )
    assert f0 == pytest.approx(26.17, abs=0.02)
    assert round(f0) == FIG_4_35_FMSM_UNTIED


def test_fig_4_35_tied_resonance() -> None:
    """Hopkins Fig. 4.35, scenario B (2,5 ties/m2, k = 2e6 N/m): fmsm = 50 Hz.

    The tie array adds N k / S = 5,0e6 N/m3 in parallel with the 1,893e6 N/m3
    air spring, so the resonance nearly doubles.
    """
    ties = building.wall_tie_stiffness_per_area(
        FIG_4_35_TIES_PER_AREA, FIG_4_35_TIE_STIFFNESS
    )
    f0 = building.mass_spring_mass_resonance(
        FIG_4_35_SURFACE_DENSITY,
        FIG_4_35_SURFACE_DENSITY,
        FIG_4_35_CAVITY_DEPTH,
        tie_stiffness_per_area=ties,
        fluid=HOPKINS_AIR,
    )
    assert f0 == pytest.approx(49.94, abs=0.02)
    assert round(f0) == FIG_4_35_FMSM_TIED


def test_fig_4_35_resonance_pair_with_the_library_default_air() -> None:
    """The same pair with the library's default air (1,205 kg/m3, 343 m/s).

    The default air stiffness differs from Hopkins' rounded 1,42e5 Pa by 0,2 %,
    which does not move either printed integer.
    """
    kwargs = {
        "mass1": FIG_4_35_SURFACE_DENSITY,
        "mass2": FIG_4_35_SURFACE_DENSITY,
        "gap": FIG_4_35_CAVITY_DEPTH,
    }
    ties = building.wall_tie_stiffness_per_area(
        FIG_4_35_TIES_PER_AREA, FIG_4_35_TIE_STIFFNESS
    )
    assert round(building.mass_spring_mass_resonance(**kwargs)) == FIG_4_35_FMSM_UNTIED
    assert (
        round(
            building.mass_spring_mass_resonance(**kwargs, tie_stiffness_per_area=ties)
        )
        == FIG_4_35_FMSM_TIED
    )


def test_resonance_is_unchanged_by_a_zero_tie_term() -> None:
    """The default tie_stiffness_per_area = 0 reproduces the air-only value bit for bit."""
    plain = building.mass_spring_mass_resonance(140.0, 170.0, 0.1)
    tied = building.mass_spring_mass_resonance(
        140.0, 170.0, 0.1, tie_stiffness_per_area=0.0
    )
    assert tied == plain


def test_resonance_grows_as_the_square_root_of_the_tie_stiffness() -> None:
    """Eq. 4.89 is a square root, so quadrupling N k / S doubles the excess.

    Taking the air stiffness away by comparing two tie terms that dominate it.
    """
    kwargs = {"mass1": 140.0, "mass2": 140.0, "gap": 0.075}
    low = building.mass_spring_mass_resonance(**kwargs, tie_stiffness_per_area=1e9)
    high = building.mass_spring_mass_resonance(**kwargs, tie_stiffness_per_area=4e9)
    assert high / low == pytest.approx(2.0, rel=1e-3)


def test_negative_tie_stiffness_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"'tie_stiffness_per_area' must be non-negative"
    ):
        building.mass_spring_mass_resonance(
            140.0, 140.0, 0.075, tie_stiffness_per_area=-1.0
        )


# ---------------------------------------------------------------------------
# The bridge in the double-wall prediction
# ---------------------------------------------------------------------------


def test_ties_push_the_double_wall_resonance_up() -> None:
    """double_wall_transmission_loss carries the Eq. 4.89 resonance through."""
    freqs = np.array([50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0])
    ties = building.wall_tie_stiffness_per_area(
        FIG_4_35_TIES_PER_AREA, FIG_4_35_TIE_STIFFNESS
    )
    plain = building.double_wall_transmission_loss(freqs, 140.0, 140.0, 0.075)
    tied = building.double_wall_transmission_loss(
        freqs, 140.0, 140.0, 0.075, tie_stiffness_per_area=ties
    )
    assert plain.resonance_frequency is not None
    assert tied.resonance_frequency is not None
    assert tied.resonance_frequency > plain.resonance_frequency
    assert round(tied.resonance_frequency) == FIG_4_35_FMSM_TIED


def test_ties_extend_the_combined_mass_branch() -> None:
    """Below fmsm the leaves move as one plate (Hopkins, after Eq. 4.89).

    Raising fmsm from 26 Hz to 50 Hz therefore moves the 40 Hz band from the
    cavity-boosted branch down onto the combined-mass law, costing insulation.
    """
    freqs = np.array([40.0])
    ties = building.wall_tie_stiffness_per_area(
        FIG_4_35_TIES_PER_AREA, FIG_4_35_TIE_STIFFNESS
    )
    plain = building.double_wall_transmission_loss(freqs, 140.0, 140.0, 0.075)
    tied = building.double_wall_transmission_loss(
        freqs, 140.0, 140.0, 0.075, tie_stiffness_per_area=ties
    )
    assert float(tied.transmission_loss[0]) < float(plain.transmission_loss[0])


def test_double_wall_without_ties_is_bit_identical() -> None:
    """The new keyword must not perturb the existing prediction at all.

    Swept over a realistic construction space, the default path has to return
    exactly the same floating-point values as before the tie term existed,
    which is what ``tie_stiffness_per_area = 0`` adding 0,0 to the stiffness
    guarantees.
    """
    freqs = np.logspace(np.log10(50.0), np.log10(5000.0), 31)
    for m1 in (8.0, 25.0, 140.0):
        for m2 in (8.0, 60.0, 170.0):
            for gap in (0.025, 0.075, 0.2):
                plain = building.double_wall_transmission_loss(freqs, m1, m2, gap)
                zero = building.double_wall_transmission_loss(
                    freqs, m1, m2, gap, tie_stiffness_per_area=0.0
                )
                assert np.array_equal(plain.transmission_loss, zero.transmission_loss)
                assert plain.resonance_frequency == zero.resonance_frequency


# ---------------------------------------------------------------------------
# Coupling loss factor (Hopkins Eqs. 4.87 and 4.88)
# ---------------------------------------------------------------------------


def test_rigid_connection_matches_the_closed_form() -> None:
    """Eq. 4.87 with Yc = 0: eta = n Yj / (omega rho_s1 (Yi + Yj)**2).

    The plate area cancels because N = n S and mi = rho_s1 S.
    """
    freqs = np.array([100.0, 500.0, 2000.0])
    res = building.wall_tie_coupling_loss_factor(
        freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5
    )
    expected = (
        2.5
        / (2.0 * np.pi * freqs * 150.0)
        * res.mobility2
        / (res.mobility1 + res.mobility2) ** 2
    )
    np.testing.assert_allclose(res.coupling_loss_factor, expected, rtol=1e-12)
    np.testing.assert_allclose(res.connector_mobility, 0.0)
    assert res.tie_stiffness is None


def test_plate_mobility_is_the_infinite_thin_plate_value() -> None:
    """Eq. 2.190: Z = 8 sqrt(B' m''), so Y = 1 / (8 sqrt(B' m''))."""
    res = building.wall_tie_coupling_loss_factor(
        [100.0], 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5
    )
    assert res.mobility1 == pytest.approx(1.0 / (8.0 * np.sqrt(1.0e5 * 150.0)))
    assert res.mobility2 == pytest.approx(1.0 / (8.0 * np.sqrt(1.2e5 * 170.0)))


def test_rigid_coupling_falls_as_one_over_frequency() -> None:
    """With Yc = 0 the only frequency dependence in Eq. 4.87 is the 1/omega."""
    freqs = np.array([100.0, 200.0, 400.0])
    res = building.wall_tie_coupling_loss_factor(
        freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5
    )
    ratios = res.coupling_loss_factor[:-1] / res.coupling_loss_factor[1:]
    np.testing.assert_allclose(ratios, 2.0, rtol=1e-12)


def test_a_stiff_tie_approaches_the_rigid_limit_at_low_frequency() -> None:
    """|Yc| = omega/k vanishes as k grows, recovering the Yc = 0 result."""
    res = building.wall_tie_coupling_loss_factor(
        [50.0], 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie=1e12
    )
    assert float(res.coupling_loss_factor[0]) == pytest.approx(
        float(res.rigid_coupling_loss_factor[0]), rel=1e-6
    )


def test_a_resilient_tie_rolls_off_as_one_over_frequency_cubed() -> None:
    """Once |Yc| >> Yi + Yj, Eq. 4.87 goes as 1/omega x 1/omega**2.

    The butterfly tie at 1,7 MN/m is soft enough to be in that regime well
    inside the building acoustics range for masonry leaves.
    """
    freqs = np.array([1000.0, 2000.0, 4000.0])
    res = building.wall_tie_coupling_loss_factor(
        freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie="butterfly"
    )
    ratios = res.coupling_loss_factor[:-1] / res.coupling_loss_factor[1:]
    np.testing.assert_allclose(ratios, 8.0, rtol=0.02)


def test_softer_ties_couple_less_than_stiffer_ones() -> None:
    """Table A4 ordered by stiffness must order the coupling the same way."""
    freqs = np.array([500.0])
    values = [
        float(
            building.wall_tie_coupling_loss_factor(
                freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie=name
            ).coupling_loss_factor[0]
        )
        for name in ("butterfly", "double_triangle", "vertical_twist")
    ]
    assert values == sorted(values)


def test_resilient_coupling_never_exceeds_the_rigid_ceiling() -> None:
    """Adding a finite spring in series can only reduce the power flow."""
    freqs = np.logspace(1.7, 3.7, 40)
    for name in building.WALL_TIE_STIFFNESS:
        res = building.wall_tie_coupling_loss_factor(
            freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie=name
        )
        assert bool(np.all(res.coupling_loss_factor <= res.rigid_coupling_loss_factor))


def test_coupling_is_proportional_to_the_tie_density() -> None:
    """Eq. 4.87 is linear in N: doubling the ties doubles eta_ij."""
    freqs = np.array([250.0])
    single = building.wall_tie_coupling_loss_factor(
        freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie="butterfly"
    )
    double = building.wall_tie_coupling_loss_factor(
        freqs, 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=5.0, tie="butterfly"
    )
    assert float(double.coupling_loss_factor[0]) == pytest.approx(
        2.0 * float(single.coupling_loss_factor[0])
    )


def test_coupling_rejects_a_non_positive_tie_density() -> None:
    with pytest.raises(ValueError, match=r"'ties_per_area' must be positive"):
        building.wall_tie_coupling_loss_factor(
            [100.0], 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=0.0
        )


def test_coupling_rejects_an_unknown_tie() -> None:
    with pytest.raises(ValueError, match="Unknown wall tie"):
        building.wall_tie_coupling_loss_factor(
            [100.0], 150.0, 170.0, 1.0e5, 1.2e5, ties_per_area=2.5, tie="staple"
        )


def test_a_connector_mobility_of_another_length_is_refused() -> None:
    """``connector_mobility`` is the spectrum of this result nothing draws.

    The figure plots ``eta_ij`` and the rigid ceiling against ``frequencies``
    and never opens ``connector_mobility``, so a tie mobility one band short
    redraws the same picture pixel for pixel and only shows up in whatever
    reads the field, paired with the wrong frequencies. The invariant
    therefore has to be held at construction.
    """
    result = building.wall_tie_coupling_loss_factor(
        np.logspace(np.log10(50.0), np.log10(5000.0), 16),
        150.0,
        170.0,
        1.0e5,
        1.2e5,
        ties_per_area=2.5,
        tie="butterfly",
    )
    shorter = result.connector_mobility[:-1]
    with pytest.raises(ValueError, match=r"'connector_mobility' \(15\).*per frequency"):
        dataclasses.replace(result, connector_mobility=shorter)
