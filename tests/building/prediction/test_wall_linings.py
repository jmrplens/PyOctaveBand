#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for wall linings and additional layers (ISO 12354-1:2017 Annex D).

A lining is a second leaf hung off a base wall, and the annex treats it as one
subject from end to end: Formulae (D.1)/(D.2) place its mass-spring-mass
resonance ``fo``, Table D.1 and Formulae (D.3), (D.4) and (D.7) read a weighted
improvement off that resonance, Formulae (D.5)/(D.6) correct it for anchors and
for the glued fraction of the area, and Formula (D.8) transfers a laboratory
``DeltaRlab`` to the element actually built. Everything below hangs on ``fo``,
which is why it travels together.

Two conventions in that chain are worth more than the formulae themselves and
are pinned here band by band. Clause D.2.2 rounds ``fo`` to the centre frequency
of the one-third-octave band *in which it falls*, so the ISO 266 band edges
``10^((n +- 0,5)/10)`` decide the row, not proximity to a rounded nominal label:
the two disagree by up to 1,4 % in frequency and by up to 8,8 dB in the answer.
And the rounding happens *before* Table D.1 is read, so the
``74,4 - 20 lg(fo) - Rw/2`` branch is evaluated at the nominal centre, not at
the raw resonance.

Sources and page references used as oracles below (printed page / pdf page):

* **H** = Hopkins, *Sound Insulation* (Butterworth-Heinemann, 2007);
  printed page + 27 = pdf page. Fig. 4.48 (printed p. 486) supplies the four
  measured lining resonances on a 100 mm aircrete block wall.
* **ISO 12354-1:2017** Annex D, printed pp. 37-41 / pdf pp. 43-47.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phonometry import building
from phonometry.building.prediction.linings import (
    _THIRD_OCTAVE_CENTRES,
    _THIRD_OCTAVE_FIRST_INDEX,
)


# ===========================================================================
# Wall linings and additional layers (ISO 12354-1 Annex D)
# ===========================================================================
def test_formula_d1_reproduces_the_printed_hopkins_lining_resonance() -> None:
    """H Fig. 4.48(c) caption, printed p. 486.

    "100 mm aircrete block wall (rhos = 51 kg/m2) ... Lining: 9,5 mm
    plasterboard (6,3 kg/m2) laminated with 32 mm expanded polystyrene
    (0,4 kg/m2, s = 65 MN/m3) attached with gypsum based adhesive dabs ...
    Mass-resilient material-mass: fmsm = 542 Hz."
    This is ISO 12354-1:2017 Formula (D.1) with the resilient layer bonded
    directly to the wall.
    """
    f0 = building.lining_resonance_frequency(51.0, 6.3, dynamic_stiffness=65e6)
    assert f0 == pytest.approx(542.0, rel=0.01)


@pytest.mark.parametrize(
    ("base", "lining", "gap", "printed"),
    [
        # H Fig. 4.48 captions, printed p. 486, all on the same 100 mm aircrete
        # block wall (rhos = 51 kg/m2):
        # (a) 12,5 mm plasterboard (10,8 kg/m2), 37 mm cavity with rock fibre
        #     quilt -> "Mass-air-mass: fmsm = 104 Hz";
        # (b) 9,5 mm plasterboard + 32 mm XPS (6,3 + 1,1 kg/m2), 37 mm empty
        #     cavity -> "fmsm = 123 Hz";
        # (c) 9,5 mm plasterboard + 32 mm EPS (6,3 + 0,4 kg/m2), 10 mm empty
        #     cavity -> "Mass-air-mass: fmsm = 247 Hz".
        (51.0, 10.8, 0.037, 104.0),
        (51.0, 7.4, 0.037, 123.0),
        (51.0, 6.7, 0.010, 247.0),
    ],
)
def test_hopkins_mass_air_mass_lining_resonances(
    base: float, lining: float, gap: float, printed: float
) -> None:
    """The adiabatic cavity stiffness ``rho0 c0^2/d`` of H Eq. (4.72)."""
    assert building.mass_spring_mass_resonance(base, lining, gap) == pytest.approx(
        printed, rel=0.01
    )


def test_formula_d2_is_formula_d1_with_the_filled_cavity_stiffness() -> None:
    """ISO 12354-1:2017 Formula (D.2), printed p. 38.

    A stud cavity filled with a porous layer of ``r >= 5 kPa s/m2`` is given the
    stiffness ``0,111/d`` MN/m3, i.e. a near-isothermal value: it sits about
    10 % above the isothermal ``p0/d = 0,1013/d`` and about 22 % below the
    adiabatic ``rho0 c0^2/d`` that H uses for an empty cavity, so a filled
    cavity resonates below an empty one of the same depth. Formula (D.2) has no
    printed worked example; only this bracketing and its identity with
    Formula (D.1) can be pinned.
    """
    base, lining, depth = 51.0, 10.8, 0.037
    filled = building.lining_resonance_frequency(base, lining, cavity_depth=depth)
    assert filled == pytest.approx(
        building.lining_resonance_frequency(
            base, lining, dynamic_stiffness=0.111e6 / depth
        ),
        rel=1e-12,
    )
    isothermal = building.lining_resonance_frequency(
        base, lining, dynamic_stiffness=101325.0 / depth
    )
    adiabatic = building.mass_spring_mass_resonance(base, lining, depth)
    assert isothermal < filled < adiabatic


def test_table_d1_low_frequency_branch() -> None:
    """ISO 12354-1:2017 Table D.1, printed p. 39.

    "30 <= f0 <= 160 -> DeltaRw = 74,4 - 20 lg(f0) - Rw/2", with NOTE 1 "For
    resonance frequencies below 200 Hz, the minimum value of DeltaRw is 0 dB".
    """
    assert building.weighted_lining_improvement(50.0, 50.0) == pytest.approx(
        74.4 - 20.0 * np.log10(50.0) - 25.0, abs=1e-9
    )
    assert building.weighted_lining_improvement(100.0, 40.0) == pytest.approx(
        74.4 - 40.0 - 20.0, abs=1e-9
    )
    # NOTE 1's floor. Inside the stated validity box (30 Hz to 160 Hz,
    # 20 dB <= Rw <= 60 dB) the branch never reaches 0 dB: its minimum is
    # 74,4 - 20 lg(160) - 30 = 0,32 dB, so the floor only binds for an Rw
    # above the range Clause D.2.2 states.
    assert building.weighted_lining_improvement(160.0, 60.0) == pytest.approx(
        74.4 - 20.0 * np.log10(160.0) - 30.0, abs=1e-9
    )
    with pytest.warns(UserWarning, match="outside the 20 dB"):
        assert building.weighted_lining_improvement(160.0, 62.0) == 0.0


@pytest.mark.parametrize(
    ("resonance", "printed"),
    [
        (200.0, -1.0),
        (250.0, -3.0),
        (315.0, -5.0),
        (400.0, -7.0),
        (500.0, -9.0),
        (630.0, -10.0),
        (1000.0, -10.0),
        (1250.0, -10.0),
        (2000.0, -5.0),
        (5000.0, -5.0),
    ],
)
def test_table_d1_fixed_rows(resonance: float, printed: float) -> None:
    """ISO 12354-1:2017 Table D.1, the rows above 160 Hz (printed p. 39).

    Above 200 Hz the improvement is negative and independent of ``Rw``.
    """
    assert building.weighted_lining_improvement(resonance, 45.0) == printed
    assert building.weighted_lining_improvement(resonance, 30.0) == printed


@pytest.mark.parametrize(
    ("lower_band", "upper_band", "lower_value", "upper_value"),
    [
        # ISO 266 band edges are 10^((n +- 0,5)/10) about the exact midband
        # frequencies 10^(n/10), NOT the geometric mean of the rounded nominal
        # labels. The two disagree by up to 1,4 % and straddle real inputs.
        (250.0, 315.0, -3.0, -5.0),
        (315.0, 400.0, -5.0, -7.0),
        (400.0, 500.0, -7.0, -9.0),
        (500.0, 630.0, -9.0, -10.0),
    ],
)
def test_table_d1_reads_the_band_containing_fo_not_the_nearest_label(
    lower_band: float, upper_band: float, lower_value: float, upper_value: float
) -> None:
    """ISO 12354-1:2017 Clause D.2.2, printed p. 38: ``fo`` is "rounded to the
    centre frequency of the one-third-octave band in which fo falls".

    Band *containment*, not proximity to a label. The edge between two
    one-third-octave bands ``n`` and ``n + 1`` is ``10^((n + 0,5)/10)``, which
    for 250|315 is 281,838 Hz while the geometric mean of the labels is
    280,624 Hz: a resonance at 281 Hz belongs to the 250 Hz band and reads
    -3 dB, and a nearest-label rounding would answer -5 dB.
    """
    index = int(np.round(10.0 * np.log10(lower_band)))
    edge = 10.0 ** ((index + 0.5) / 10.0)
    assert building.weighted_lining_improvement(edge * 0.999, 45.0) == lower_value
    assert building.weighted_lining_improvement(edge * 1.001, 45.0) == upper_value
    # Every nominal centre must read its own row.
    assert building.weighted_lining_improvement(lower_band, 45.0) == lower_value
    assert building.weighted_lining_improvement(upper_band, 45.0) == upper_value


def test_table_d1_low_branch_evaluates_at_the_rounded_nominal_not_raw_fo() -> None:
    """Clause D.2.2 rounds ``fo`` *before* Table D.1 is read, and the
    ``74,4 - 20 lg(fo) - Rw/2`` row is part of Table D.1.

    So the formula is evaluated at the band's nominal centre, not at the raw
    resonance frequency: two linings whose resonances fall in the same band
    must get the same rating. Using raw ``fo`` instead is what lets a
    band-edge error through unnoticed, since the answer then varies smoothly
    and no boundary test can see it.
    """
    # 100 Hz band: 89,1 Hz to 112,2 Hz (ISO 266 edges).
    expected = 74.4 - 20.0 * np.log10(100.0) - 22.5
    for f0 in (89.2, 95.0, 100.0, 108.0, 112.0):
        assert building.weighted_lining_improvement(f0, 45.0) == pytest.approx(
            expected, abs=1e-9
        )
    # Raw fo would spread these over 2 dB.
    raw = [74.4 - 20.0 * np.log10(f) - 22.5 for f in (89.2, 112.0)]
    assert raw[0] - raw[1] == pytest.approx(1.98, abs=0.02)


def test_table_d1_band_edges_beat_the_nominal_geometric_mean() -> None:
    """The two rounding conventions disagree on real inputs, by up to 8,8 dB.

    The worst case is the 160|200 boundary, where Table D.1 changes from the
    ``74,4 - 20 lg(fo) - Rw/2`` branch to the fixed -1 dB row. The ISO 266
    edge is ``10^(2,25) = 177,83 Hz``; the geometric mean of the nominal
    labels 160 and 200 is 178,89 Hz. A resonance at 178,5 Hz belongs to the
    200 Hz band.
    """
    assert 10.0**2.25 < 178.5 < np.sqrt(160.0 * 200.0)
    assert building.weighted_lining_improvement(178.5, 45.0) == -1.0
    assert building.weighted_lining_improvement(177.0, 45.0) == pytest.approx(
        74.4 - 20.0 * np.log10(160.0) - 22.5, abs=1e-9
    )
    swing = (74.4 - 20.0 * np.log10(160.0) - 22.5) - (-1.0)
    assert swing == pytest.approx(8.8, abs=0.05)


def test_table_d1_63_to_80_boundary() -> None:
    """The same distinction inside the low-frequency branch, worth 2,1 dB.

    The 63 Hz band ends at ``10^1,85 = 70,79 Hz``; the geometric mean of the
    labels 63 and 80 is 70,99 Hz. A lining resonance of 70,81 Hz, which the
    guide's worked stud lining produces, therefore reads off the 80 Hz row.
    """
    assert 10.0**1.85 < 70.8115 < np.sqrt(63.0 * 80.0)
    assert building.weighted_lining_improvement(70.8115, 45.0) == pytest.approx(
        74.4 - 20.0 * np.log10(80.0) - 22.5, abs=1e-9
    )
    difference = 20.0 * np.log10(80.0 / 63.0)
    assert difference == pytest.approx(2.08, abs=0.01)


def test_table_d1_1600_hz_overlap_takes_the_conservative_value() -> None:
    """ISO 12354-1:2017 Table D.1 prints 1 600 Hz in two rows with different
    values, "630 to 1 600 -> -10" and "1 600 <= f0 <= 5 000 -> -5"; see
    docs/ERRATA.md. The library takes the more conservative -10 dB."""
    assert building.weighted_lining_improvement(1600.0, 45.0) == -10.0
    assert building.weighted_lining_improvement(2000.0, 45.0) == -5.0


@pytest.mark.parametrize("resonance", [20.0, 6000.0])
def test_table_d1_rejects_untabulated_resonances(resonance: float) -> None:
    """Table D.1 covers 30 Hz to 5 000 Hz only."""
    with pytest.raises(ValueError, match="Table D.1"):
        building.weighted_lining_improvement(resonance, 45.0)


@pytest.mark.parametrize("rating", [-50.0, 0.0, 19.9, 60.1, 80.0])
def test_table_d1_warns_outside_the_stated_rw_box(rating: float) -> None:
    """ISO 12354-1:2017 Clause D.2.2, printed p. 38, states Table D.1 "for
    basic structural elements with a weighted sound reduction index in the
    range of 20 dB <= Rw <= 60 dB".

    The low branch is linear in ``Rw`` and returns nonsense outside it: an
    ``Rw`` of -50 dB yields +65,4 dB of improvement from a lining. The value
    is still returned, because the formula is what it is, but the caller is
    told the standard does not cover it.
    """
    with pytest.warns(UserWarning, match="outside the 20 dB"):
        building.weighted_lining_improvement(50.0, rating)


@pytest.mark.parametrize("rating", [20.0, 45.0, 60.0])
def test_table_d1_is_silent_inside_the_stated_rw_box(rating: float) -> None:
    """No warning at either endpoint, both of which Clause D.2.2 includes."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        building.weighted_lining_improvement(50.0, rating)
        building.weighted_lining_improvement(500.0, rating)


@pytest.mark.parametrize("resonance", [1.0, 29.0, 5001.0, 20000.0])
def test_lining_formulae_warn_outside_the_annex_d_range(resonance: float) -> None:
    """Formulae (D.3), (D.4) and (D.7) are monotonic in ``lg(fo)`` and
    unbounded below the range Annex D puts lining resonances on.

    At ``fo = 1 Hz`` the mineral-wool fit returns +82,5 dB, which is more
    improvement than any lining has ever given. Table D.1 is the only place
    the annex bounds ``fo``, at 30 Hz to 5 000 Hz, so that is the envelope
    used.
    """
    with pytest.warns(UserWarning, match="outside the 30 Hz"):
        building.lining_improvement(resonance)
    with pytest.warns(UserWarning, match="outside the 30 Hz"):
        building.lining_improvement(resonance, system="studs")


@pytest.mark.parametrize("resonance", [30.0, 100.0, 5000.0])
def test_lining_formulae_are_silent_inside_the_annex_d_range(
    resonance: float,
) -> None:
    """Both endpoints included, so plotting the whole range stays quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for system in ("mineral_wool", "foam", "studs"):
            building.lining_improvement(resonance, system=system)  # type: ignore[arg-type]


def test_stacked_d5_and_d6_corrections_can_fall_below_the_reference_floor() -> None:
    """Annex D places the ``>= -4 dB`` floor inside Formulae (D.3)/(D.4) and
    does not restate it after (D.5) or (D.6), so the library does not either.

    This pins the literal reading, and the size of the excursion, so that
    changing it later is a deliberate act rather than a silent one.
    """
    reference = building.lining_improvement(5000.0)
    assert reference.ratings == (-4.0, -4.0, -4.0)
    corrected = building.lining_improvement(5000.0, anchors=True, glued_area=100.0)
    assert corrected.delta_rw == pytest.approx(0.66 * -4.0 - 1.2 - 5.0 + 2.0, abs=1e-9)
    assert corrected.delta_rw < -4.0
    assert corrected.delta_rw == pytest.approx(-6.84, abs=0.01)


def test_exterior_lining_formulae_d3_and_d4() -> None:
    """ISO 12354-1:2017 Formulae (D.3)/(D.4), printed p. 40.

    Mineral wool: -36 lg(fo) + 82,5 / -42 lg(fo) + 92,0 / -39 lg(fo) + 87,7,
    each ">= -4"; foams: -33 lg(fo) + 76,0 / -33 lg(fo) + 74,0 /
    -36 lg(fo) + 77,0, each ">= -3".
    """
    wool = building.lining_improvement(100.0)
    assert wool.ratings == pytest.approx((82.5 - 72.0, 92.0 - 84.0, 87.7 - 78.0))
    foam = building.lining_improvement(100.0, system="foam")
    assert foam.ratings == pytest.approx((76.0 - 66.0, 74.0 - 66.0, 77.0 - 72.0))
    # The printed floors bind at a high resonance frequency.
    assert building.lining_improvement(1000.0).ratings == pytest.approx(
        (-4.0, -4.0, -4.0)
    )
    assert building.lining_improvement(1000.0, system="foam").ratings == pytest.approx(
        (-3.0, -3.0, -3.0)
    )


def test_anchor_and_glued_area_corrections_d5_and_d6() -> None:
    """ISO 12354-1:2017 Formulae (D.5)/(D.6), printed p. 40.

    (D.5) ``0,66 DeltaRw,ref - 1,2``, ``0,62 DeltaRA,ref - 1,3``,
    ``0,54 DeltaRA,tr,ref - 1,6``; (D.6) ``DeltaR - 0,05 %So + 2,0``, which is
    the identity at the 40 % reference glued area.
    """
    reference = building.lining_improvement(100.0)
    anchored = building.lining_improvement(100.0, anchors=True)
    assert anchored.delta_rw == pytest.approx(0.66 * reference.delta_rw - 1.2)
    assert anchored.delta_ra == pytest.approx(0.62 * reference.delta_ra - 1.3)
    assert anchored.delta_ratr == pytest.approx(0.54 * reference.delta_ratr - 1.6)
    at_reference = building.lining_improvement(100.0, glued_area=40.0)
    assert at_reference.ratings == pytest.approx(reference.ratings)
    # More glued area couples the layer to the wall and costs improvement.
    assert (
        building.lining_improvement(100.0, glued_area=80.0).delta_rw
        < reference.delta_rw
    )


def test_stud_system_formula_d7() -> None:
    """ISO 12354-1:2017 Formula (D.7), printed p. 41: ``-20 lg(fo) + 48``,
    ``-22 lg(fo) + 51``, ``-24 lg(fo) + 54``, each ">= -4"."""
    studs = building.lining_improvement(100.0, system="studs")
    assert studs.ratings == pytest.approx((48.0 - 40.0, 51.0 - 44.0, 54.0 - 48.0))
    assert building.lining_improvement(5000.0, system="studs").delta_rw == -4.0


@pytest.mark.parametrize(
    ("system", "floor"),
    [("mineral_wool", -4.0), ("foam", -3.0), ("studs", -4.0)],
)
def test_every_annex_d_rating_has_its_own_printed_floor(
    system: str, floor: float
) -> None:
    """Formulae (D.3), (D.4) and (D.7) print the floor on **all three**
    ratings, not only on ``DeltaRw``.

    Each is ">= -4 dB" for mineral wool and for studs, ">= -3 dB" for the
    foams. Testing only ``delta_rw`` leaves the other two floors unpinned, and
    every fit is steep enough (20 to 42 dB per decade) that 5 000 Hz is well
    into the floored regime for all three.
    """
    result = building.lining_improvement(5000.0, system=system)  # type: ignore[arg-type]
    assert result.ratings == (floor, floor, floor)
    # Unfloored, all three would be far below it.
    unfloored = building.lining_improvement(100.0, system=system)  # type: ignore[arg-type]
    assert all(value > floor + 3.0 for value in unfloored.ratings)


def test_third_octave_centres_are_the_iso_266_nominal_series() -> None:
    """Every entry of the rounding table is the ISO 266 nominal label of its
    band, i.e. within about 1 % of the exact midband frequency ``10^(n/10)``.

    Without this the entries below 31,5 Hz and above 5 000 Hz are unreachable
    through Table D.1 and could hold any value at all.
    """
    for offset, nominal in enumerate(_THIRD_OCTAVE_CENTRES):
        exact = 10.0 ** ((offset + _THIRD_OCTAVE_FIRST_INDEX) / 10.0)
        assert abs(nominal / exact - 1.0) < 0.012, (offset, nominal, exact)
    assert _THIRD_OCTAVE_CENTRES[0] == 12.5
    assert _THIRD_OCTAVE_CENTRES[-1] == 10000.0
    assert len(set(_THIRD_OCTAVE_CENTRES)) == len(_THIRD_OCTAVE_CENTRES)


def test_in_situ_transfer_formula_d8() -> None:
    """ISO 12354-1:2017 Formula (D.8), printed p. 41.

    ``DeltaRsitu = DeltaRlab + a X`` with ``a = 1,35 lg(fo) - 3,5 <= 0`` and
    ``X = Rw,situ - 53`` clamped to ``[-10, +7]``. The cap on ``a`` bites above
    ``fo = 10^(3,5/1,35) = 392 Hz``, where the transfer becomes the identity.
    """
    a = 1.35 * np.log10(100.0) - 3.5
    assert building.lining_improvement_in_situ(10.0, 100.0, 60.0) == pytest.approx(
        10.0 + a * 7.0
    )
    assert building.lining_improvement_in_situ(10.0, 100.0, 53.0) == pytest.approx(10.0)
    # X clamps at -10 and +7.
    assert building.lining_improvement_in_situ(10.0, 100.0, 20.0) == pytest.approx(
        10.0 + a * -10.0
    )
    assert building.lining_improvement_in_situ(10.0, 100.0, 70.0) == pytest.approx(
        10.0 + a * 7.0
    )
    # a is capped at 0 above about 392 Hz.
    assert building.lining_improvement_in_situ(10.0, 500.0, 30.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"system": "studs", "glued_area": 40.0}, "glued exterior systems"),
        ({"glued_area": 100.1}, "glued_area"),
        ({"glued_area": 0.0}, "glued_area"),
    ],
)
def test_glued_area_guards(kwargs: dict[str, object], match: str) -> None:
    """Formula (D.6) applies to glued exterior systems over part of the area.

    A stud system is corrected by Formula (D.5) instead, and the glued fraction
    is a percentage of the element area, so it cannot exceed 100 nor be zero:
    the formula divides by it. None of the three was covered.
    """
    with pytest.raises(ValueError, match=match):
        building.lining_improvement(100.0, **kwargs)  # type: ignore[arg-type]
