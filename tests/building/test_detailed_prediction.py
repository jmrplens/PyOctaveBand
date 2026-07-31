#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for the EN/ISO 12354-1/-2:2017 detailed per-band prediction model.

The oracle is the pair of worked examples that ISO 12354-1:2017 Annex L and
ISO 12354-2:2017 Annex G print for the **same** heavy homogeneous building
(two dwellings one above the other): about twenty per-band tables exposing
every intermediate quantity, from the radiation factors and the in-situ total
loss factor through ``Rsitu``/``Ln,situ``, the equivalent absorption lengths
and ``Dv,ij,situ`` to each transmission path and the totals. The second worked
example of the same annexes (a wood frame lightweight building, Tables L.11 to
L.16 and G.11 to G.13) exercises the Type B branch, and ISO 12354-2 Table B.2
gives an independent octave-band plausibility check on the impact chain.

**Two inputs are taken corrected, not as printed.** The Table L.3 / G.3 input
block prints one perimeter sum ``Σ lk αk`` per element *type*, but the columns
need one per element, and it prints the external walls' internal loss factor as
0,013 where the element specification (and Table B.3) gives 0,012 5. This
fixture therefore

* builds every ``Σ lk αk`` from Formula (C.4), ``αk = Σj √(fc,j/fref)
  10^(−Kij/10)``, using the *unrounded* Annex E junction indices. That
  derivation reproduces the two printed values that are self-consistent
  (external wall 1: 2,375 m; internal wall 2: 1,839 m against the printed
  1,840 m) and supplies the three that are not, and
* uses ``ηint = 0,012 5``.

Both discrepancies are recorded in ``docs/ERRATA.md``. The printed tables are
rounded but were computed unrounded, so the per-band comparisons use a 0,1 dB
tolerance as the standard's own note (Table L.1) invites.

The building itself is built once, in :mod:`iso12354_building`, which the
conformance report and the documentation figure import too, so none of the
three can drift from the transcription in :mod:`reference_data`.
"""

from __future__ import annotations

import iso12354_building
import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    airborne_flanking_path,
    bare_floor_impact_level,
    bending_radiation_factor,
    calculated_sound_reduction_index,
    detailed_airborne_prediction,
    detailed_impact_prediction,
    direct_impact_level,
    direct_reduction_index,
    flanking_element,
    flanking_impact_level_from_flanking_level,
    flanking_impact_level_from_normalized_difference,
    flanking_reduction_index_from_flanking_level,
    flanking_reduction_index_from_normalized_difference,
    floating_floor_improvement,
    forced_radiation_factor,
    in_situ_element,
    in_situ_equivalent_absorption_length,
    in_situ_impact_level,
    in_situ_reduction_index,
    in_situ_total_loss_factor,
    in_situ_velocity_level_difference,
    laboratory_total_loss_factor,
    perimeter_absorption_coefficient,
    predicted_airborne_insulation,
    reciprocity_impact_level,
    resonant_sound_reduction_index,
    structural_reverberation_time,
    weighted_impact_rating,
    weighted_rating,
)

BANDS = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)
# The building's own geometry and material data live in the shared fixture
# module, so the tests, the conformance report and the documentation figure
# can never be built from three different transcriptions.
FC_INT = iso12354_building.FC_INT
M_INT = iso12354_building.M_INT
SEPARATING_AREA = iso12354_building.SEPARATING_AREA
TOL = 0.1


@pytest.fixture(scope="module")
def situ() -> dict:
    """Every element of the worked example evaluated in situ, per band."""
    return {k: in_situ_element(e, BANDS) for k, e in iso12354_building.elements().items()}


@pytest.fixture(scope="module")
def delta_l() -> np.ndarray:
    """The floating floor's improvement, 30 lg(f/fo) above fo = 52,8 Hz."""
    return floating_floor_improvement(
        BANDS, resonance_frequency=iso12354_building.floating_floor_resonance()
    )


# --------------------------------------------------------------------------
# Annex E junctions and the Formula (C.4) perimeter sums
# --------------------------------------------------------------------------
def test_junction_indices_match_tables_l5_to_l9() -> None:
    """Tables L.5 to L.9 / G.5 to G.9 (junction vibration reduction indices)."""
    computed = iso12354_building.junction_indices()
    for name, printed in ref.ISO12354_ANNEX_L_KIJ.items():
        assert computed[name] == pytest.approx(printed, abs=0.05), name


def test_perimeter_sums_reproduce_the_two_consistent_printed_values() -> None:
    """Table L.3 / G.3 input block, via Formula (C.4).

    Only external wall 1 (2,375 m) and internal wall 2 (1,840 m) are printed
    consistently with their columns; see docs/ERRATA.md.
    """
    sums = iso12354_building.perimeter_sums()
    printed = ref.ISO12354_ANNEX_L3_PRINTED_PERIMETER
    assert sums["ext1"] == pytest.approx(printed["ext1"], abs=0.005)
    assert sums["int2"] == pytest.approx(printed["int"], abs=0.005)
    # The printed floor value does not reproduce its column; Formula (C.4)
    # gives 2,659 m instead of the printed 2,364 m.
    assert sums["floor"] == pytest.approx(2.659, abs=0.005)
    assert sums["floor"] != pytest.approx(printed["floor"], abs=0.05)


def test_floating_floor_resonance_and_improvement(delta_l: np.ndarray) -> None:
    """Part 2 Formulae (C.1)/(C.2) and the Table L.4 / G.4 ΔL column."""
    f0 = iso12354_building.floating_floor_resonance()
    assert f0 == pytest.approx(ref.ISO12354_ANNEX_L_FLOATING_F0, abs=0.05)
    assert delta_l == pytest.approx(ref.ISO12354_ANNEX_L4_DELTA, abs=TOL)


# --------------------------------------------------------------------------
# Table L.2 / G.2 - radiation factors
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", sorted(ref.ISO12354_ANNEX_L2_SIGMA))
def test_table_l2_free_radiation_factor(situ: dict, label: str) -> None:
    """Table L.2 / G.2, radiation factor for free bending waves σ.

    The internal walls' printed values at 100 Hz sit 0,2 % below the exact
    recomputation because the printed critical frequency (128,4 Hz) is itself
    rounded. On a σ of about 1,9 that is 0,004, so an absolute tolerance of
    5·10⁻³ covers it and stays tight everywhere else.
    """
    got = situ[label].radiation_factor
    assert got == pytest.approx(ref.ISO12354_ANNEX_L2_SIGMA[label], abs=5e-3)


@pytest.mark.parametrize("label", sorted(ref.ISO12354_ANNEX_L2_SIGMA_F))
def test_table_l2_forced_radiation_factor(situ: dict, label: str) -> None:
    """Table L.2 / G.2, radiation factor for forced waves σf (Formula B.3)."""
    got = situ[label].forced_radiation_factor
    assert got == pytest.approx(ref.ISO12354_ANNEX_L2_SIGMA_F[label], abs=5e-4)


def test_forced_radiation_factor_matches_table_b1_openings() -> None:
    """ISO 12354-1 Table B.1: 10 lg σf of the 2 m² and 10 m² test openings."""
    bands = BANDS[:-1]  # Table B.1 stops at 4 kHz
    small = forced_radiation_factor(bands, length1=1.5, length2=1.25)
    large = forced_radiation_factor(bands, length1=3.75, length2=2.65)
    printed_small = (-6.5, -4.8, -3.5, -2.6, -1.8, -1.1, -0.5, 0.0, 0.5, 0.9,
                     1.3, 1.7, 2.0, 2.3, 2.6, 2.9, 3.0, 3.0, 3.0, 3.0)
    printed_large = (-2.1, -1.4, -0.7, -0.2, 0.3, 0.8, 1.1, 1.5, 1.8, 2.2, 2.5,
                     2.7, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    assert 10.0 * np.log10(small) == pytest.approx(printed_small, abs=0.05)
    assert 10.0 * np.log10(large) == pytest.approx(printed_large, abs=0.05)


# --------------------------------------------------------------------------
# Table L.3 / G.3 - in-situ loss factors, Rsitu and Ln,situ
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", sorted(ref.ISO12354_ANNEX_L3_ETA))
def test_table_l3_total_loss_factor(situ: dict, label: str) -> None:
    """Table L.3 / G.3, in-situ total loss factor ηtot,situ (Formula C.1)."""
    got = situ[label].total_loss_factor
    assert got == pytest.approx(ref.ISO12354_ANNEX_L3_ETA[label], abs=5e-5)


@pytest.mark.parametrize("label", sorted(ref.ISO12354_ANNEX_L3_R_SITU))
def test_table_l3_in_situ_reduction_index(situ: dict, label: str) -> None:
    """Table L.3 / G.3, in-situ sound reduction index (Formulae B.2/B.10)."""
    got = situ[label].sound_reduction_index
    assert got == pytest.approx(ref.ISO12354_ANNEX_L3_R_SITU[label], abs=TOL)


def test_table_g3_in_situ_impact_level(situ: dict) -> None:
    """Table G.3, in-situ impact level of the bare slab (Part 2, Formula B.2)."""
    assert situ["floor"].impact_level == pytest.approx(
        ref.ISO12354_ANNEX_G3_LN_SITU, abs=TOL
    )


def test_structural_reverberation_time_is_the_loss_factor_inverse(situ: dict) -> None:
    """``Ts = 2,2/(f ηtot)`` closes the Formula (C.1) pair exactly."""
    floor = situ["floor"]
    assert structural_reverberation_time(
        BANDS, floor.total_loss_factor
    ) == pytest.approx(floor.reverberation_time, rel=1e-12)


# --------------------------------------------------------------------------
# Table L.4 / G.4 - absorption lengths and junction level differences
# --------------------------------------------------------------------------
@pytest.mark.parametrize("label", sorted(ref.ISO12354_ANNEX_L4_ABSORPTION))
def test_table_l4_absorption_lengths(situ: dict, label: str) -> None:
    """Table L.4 / G.4, in-situ equivalent absorption length (Formula 11)."""
    got = situ[label].absorption_length
    assert got == pytest.approx(ref.ISO12354_ANNEX_L4_ABSORPTION[label], abs=TOL)


def test_table_l4_velocity_level_differences(situ: dict) -> None:
    """Table L.4 / G.4, ``Dv,ij,situ`` of the two printed paths (Formula 10).

    The second printed block is labelled "2d" but carries the numbers of path
    4d (internal wall 2); see docs/ERRATA.md.
    """
    kij = iso12354_building.junction_indices()
    lengths = iso12354_building.COUPLING_LENGTH
    d1 = in_situ_velocity_level_difference(
        kij["floor-ext"], coupling_length=lengths["ext1"],
        absorption_length_i=situ["floor"].absorption_length,
        absorption_length_j=situ["ext1"].absorption_length,
    )
    fourth = in_situ_velocity_level_difference(
        kij["floor-int"], coupling_length=lengths["int2"],
        absorption_length_i=situ["floor"].absorption_length,
        absorption_length_j=situ["int2"].absorption_length,
    )
    assert d1 == pytest.approx(ref.ISO12354_ANNEX_L4_DV["D1"], abs=TOL)
    assert fourth == pytest.approx(ref.ISO12354_ANNEX_L4_DV["4d"], abs=TOL)


def test_velocity_level_difference_is_floored_at_zero() -> None:
    """Formula (10) states ``Dv,ij,situ ≥ 0 dB``."""
    got = in_situ_velocity_level_difference(
        -30.0, coupling_length=4.0, absorption_length_i=10.0,
        absorption_length_j=10.0,
    )
    assert got == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Table L.1 - the airborne chain
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def airborne(situ: dict, delta_l: np.ndarray):
    """The Annex L airborne prediction, assembled from the fixture."""
    return detailed_airborne_prediction(
        BANDS,
        direct_index=direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta_l
        ),
        flanking_paths=iso12354_building.airborne_paths(situ, delta_l),
    )


def test_table_l1_every_transmission_path(airborne) -> None:
    """Table L.1, all thirteen direct and flanking sound reduction indices."""
    for path in airborne.paths:
        assert path.values == pytest.approx(
            ref.ISO12354_ANNEX_L1_PATHS[path.label], abs=TOL
        ), path.label


def test_table_l1_apparent_index_and_rating(airborne) -> None:
    """Table L.1, the total ``R'`` per band and its ISO 717-1 rating.

    Table L.1 prints ``Rw = 57,8`` and the statement line ``R'w = 57,9``: both
    are the *continuously* shifted reference curve, one truncated and one
    rounded, not the 1 dB-step rating ISO 717-1 defines (see docs/ERRATA.md).
    The ISO 717-1 rating of the same spectrum is 57 dB.
    """
    assert airborne.r_prime == pytest.approx(ref.ISO12354_ANNEX_L1_R_PRIME, abs=TOL)
    assert airborne.rating is not None
    assert airborne.rating.rating == ref.ISO12354_ANNEX_L1_R_PRIME_W


def test_path_fractions_sum_to_one_and_name_a_dominant_path(airborne) -> None:
    """Every band's path shares partition the transmitted energy."""
    assert airborne.fractions.sum(axis=0) == pytest.approx(1.0)
    assert len(airborne.dominant) == BANDS.size
    # At 50 Hz the direct path dominates; by 2 kHz a flanking path does.
    assert airborne.dominant[0] == "Dd"
    assert airborne.dominant[-1] != "Dd"


# --------------------------------------------------------------------------
# Tables G.1 / G.4 - the impact chain
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def impact(situ: dict, delta_l: np.ndarray):
    """The Annex G impact prediction, assembled from the same fixture."""
    return detailed_impact_prediction(
        BANDS,
        direct_level=direct_impact_level(situ["floor"].impact_level, delta_l=delta_l),
        flanking_paths=iso12354_building.impact_paths(situ, delta_l),
    )


def test_table_g4_direct_and_flanking_impact_levels(impact) -> None:
    """Table G.4, ``Ln,Dd`` (Formula 11) and ``Ln,Df`` (Formula 12)."""
    by_label = {p.label: p.values for p in impact.paths}
    assert by_label["Dd"] == pytest.approx(ref.ISO12354_ANNEX_G4_LN_DD, abs=TOL)
    assert by_label["Df1"] == pytest.approx(ref.ISO12354_ANNEX_G4_LN_DF, abs=TOL)


def test_table_g1_direct_path_over_the_whole_range(impact) -> None:
    """Table G.1, the direct column ``Ln,Dd`` over all 21 bands.

    The Table G.1 / G.4 disagreement recorded in docs/ERRATA.md affects only
    the four *flanking* columns: Table G.1 and Table G.4 print the same direct
    column, 57,3 / 55,9 / 53,8 dB at 50 Hz to 80 Hz included, so this one is
    asserted over the full range.
    """
    by_label = {p.label: p.values for p in impact.paths}
    assert by_label["Dd"] == pytest.approx(ref.ISO12354_ANNEX_G1_PATHS["Dd"], abs=TOL)


@pytest.mark.parametrize("label", ("Df1", "Df2", "Df3", "Df4"))
def test_table_g1_flanking_paths_from_100_hz(impact, label: str) -> None:
    """Table G.1, per-element flanking impact levels, 100 Hz to 5 kHz.

    The 50 Hz, 63 Hz and 80 Hz cells of the four flanking columns of Table G.1
    disagree with Table G.4, which prints the same path Df1 for external wall 1
    from the same inputs; the recomputation reproduces Table G.4 (see
    docs/ERRATA.md), so the comparison here starts at 100 Hz. The 0,15 dB
    tolerance covers the rounding of the printed intermediate columns the
    standard chained together.
    """
    by_label = {p.label: p.values for p in impact.paths}
    assert by_label[label][3:] == pytest.approx(
        ref.ISO12354_ANNEX_G1_PATHS[label][3:], abs=0.15
    )


def test_table_g1_total_and_rating(impact) -> None:
    """Table G.1, the total ``L'n`` per band and ``L'n,w (CI)`` per ISO 717-2.

    The 50 Hz to 80 Hz flanking cells of Table G.1 feed its own total, so the
    total inherits the disagreement of docs/ERRATA.md there: the recomputation
    gives 58,7 / 57,2 / 56,1 dB against the printed 58,6 / 57,0 / 55,9 dB. The
    three bands are therefore asserted at 0,25 dB, which is tight enough to
    catch any real drift and honest about where the printed total comes from,
    and the rest of the range at the usual 0,1 dB.
    """
    assert impact.l_prime_n[3:] == pytest.approx(
        ref.ISO12354_ANNEX_G1_L_PRIME_N[3:], abs=TOL
    )
    assert impact.l_prime_n[:3] == pytest.approx(
        ref.ISO12354_ANNEX_G1_L_PRIME_N[:3], abs=0.25
    )
    assert impact.rating is not None
    assert impact.rating.rating == ref.ISO12354_ANNEX_G1_L_PRIME_N_W
    assert impact.rating.ci == ref.ISO12354_ANNEX_G1_CI


# --------------------------------------------------------------------------
# Tables L.10 / G.10 - the simplified model on the same building
# --------------------------------------------------------------------------
def test_simplified_model_agrees_with_the_detailed_model(airborne) -> None:
    """Tables L.10 and L.1: the two models rate the same building alike.

    The simplified single-number model of Clause 4.4 gives ``R'w = 57,0 dB``
    for the same building; the detailed per-band model of Clause 4.2 gives 57
    (the standard printing 57,8/57,9 from a continuous reference shift). The
    two agree to well within the model's own standard deviation.

    The printed per-path values round the coupling term ``10 lg(Ss/(lo lf))``
    to 7,0 dB and 6,0 dB, which the exact form gives as 6,9897 and 6,0206, so
    the per-path tolerance is 0,1 dB.
    """
    rw = ref.ISO12354_ANNEX_L10_RW
    # The simplified model reads the junction indices as Table L.10 prints
    # them, rounded to 0,1 dB, not the unrounded Annex E values the detailed
    # model needs for the perimeter sums.
    kij = ref.ISO12354_ANNEX_L_KIJ
    lengths = iso12354_building.COUPLING_LENGTH
    paths = []
    for tag, name in iso12354_building.enumerate_flanking():
        key = iso12354_building.JUNCTION_KIND[name]
        lij = lengths[name]
        k_cross = kij["floor-ext"] if key == "ext" else kij["floor-int"]
        k_through = kij["ext-ext"] if key == "ext" else kij["int-int"]
        paths.extend(flanking_element(
            label=tag, r_flanking=rw[key], r_separating=rw["floor"],
            k_ff=k_through, k_fd=k_cross, k_df=k_cross,
            separating_area=SEPARATING_AREA, coupling_length=lij,
            delta_r_df=ref.ISO12354_ANNEX_L10_DELTA_RW,
        ))
    simplified = predicted_airborne_insulation(
        r_direct=rw["floor"],
        delta_r_direct=ref.ISO12354_ANNEX_L10_DELTA_RW,
        flanking_paths=paths,
    )
    printed = ref.ISO12354_ANNEX_L10_PATH_RW
    for path in simplified.paths:
        if path.kind == "Dd":
            key = "Dd"
        else:
            tag = path.label.split("-")[0]
            key = {"Ff": tag + tag, "Df": "D" + tag, "Fd": tag + "d"}[path.kind]
        assert path.r_w == pytest.approx(printed[key], abs=TOL), path.label
    assert simplified.r_prime_w == pytest.approx(
        ref.ISO12354_ANNEX_L10_R_PRIME_W, abs=TOL
    )
    assert abs(simplified.r_prime_w - airborne.rating.rating) < 1.5


# --------------------------------------------------------------------------
# Second worked example: the wood frame lightweight building (Type B)
# --------------------------------------------------------------------------
def test_table_l12_resonant_correction_of_the_wall_leaf() -> None:
    """Table L.12, ``R*`` from ``R`` by the Annex B.2 8 dB estimate.

    The gypsum leaf's critical frequency falls between the 2 kHz and 2,5 kHz
    bands, above which the printed correction drops to zero.
    """
    got = resonant_sound_reduction_index(
        ref.ISO12354_TABLE_L12_R_WALL, BANDS, critical_frequency=2200.0
    )
    assert got == pytest.approx(ref.ISO12354_TABLE_L12_R_STAR_WALL, abs=TOL)


def test_table_l11_lightweight_flanking_paths_and_total() -> None:
    """Table L.11, the Ff and Df paths (Formula 17) and the total ``R'``."""
    ff = flanking_reduction_index_from_normalized_difference(
        index_i=ref.ISO12354_TABLE_L12_R_STAR_WALL,
        index_j=ref.ISO12354_TABLE_L12_R_STAR_WALL,
        normalized_velocity_level_difference=ref.ISO12354_TABLE_L12_DV_FF,
        separating_area=ref.ISO12354_LIGHTWEIGHT_AREA,
        coupling_length=ref.ISO12354_LIGHTWEIGHT_COUPLING,
    )
    df = flanking_reduction_index_from_normalized_difference(
        index_i=ref.ISO12354_TABLE_L13_R_STAR_BARE,
        index_j=ref.ISO12354_TABLE_L12_R_STAR_WALL,
        normalized_velocity_level_difference=ref.ISO12354_TABLE_L13_DV_DF,
        separating_area=ref.ISO12354_LIGHTWEIGHT_AREA,
        coupling_length=ref.ISO12354_LIGHTWEIGHT_COUPLING,
        delta_r_i=ref.ISO12354_TABLE_L13_DELTA_R,
    )
    assert ff == pytest.approx(ref.ISO12354_TABLE_L11_R_FF, abs=TOL)
    assert df == pytest.approx(ref.ISO12354_TABLE_L11_R_DF, abs=TOL)

    from phonometry.building.detailed_prediction import BandPath

    result = detailed_airborne_prediction(
        BANDS,
        direct_index=ref.ISO12354_TABLE_L12_RD_FLOOR,
        flanking_paths=[BandPath("Ff", "Ff", ff), BandPath("Df", "Df", df)],
    )
    assert result.r_prime == pytest.approx(ref.ISO12354_TABLE_L11_R_PRIME, abs=TOL)
    assert result.rating is not None
    assert result.rating.rating == ref.ISO12354_TABLE_L11_RATINGS["R_prime"]


def test_table_l11_path_ratings() -> None:
    """Table L.11, the ISO 717-1 rating of each lightweight path."""
    index = [int(np.flatnonzero(np.isclose(BANDS, f))[0])
             for f in (100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                       1000, 1250, 1600, 2000, 2500, 3150)]
    printed = ref.ISO12354_TABLE_L11_RATINGS
    for key, values in (("Rd", ref.ISO12354_TABLE_L12_RD_FLOOR),
                        ("RFf", ref.ISO12354_TABLE_L11_R_FF),
                        ("RDf", ref.ISO12354_TABLE_L11_R_DF)):
        spectrum = np.asarray(values, dtype=np.float64)[index]
        assert weighted_rating(spectrum).rating == printed[key], key


def test_table_l15_flanking_index_from_measured_level_difference() -> None:
    """Table L.15, ``R13`` from the measured ``Dn,f,13`` (Formula 16)."""
    got = flanking_reduction_index_from_flanking_level(
        ref.ISO12354_TABLE_L15_DNF,
        separating_area=ref.ISO12354_TABLE_L14_SEPARATING_AREA,
        coupling_length=ref.ISO12354_TABLE_L14_COUPLING,
        laboratory_coupling_length=ref.ISO12354_TABLE_L14_LAB_COUPLING,
    )
    assert got == pytest.approx(ref.ISO12354_TABLE_L15_R13, abs=TOL)


def test_flanking_impact_level_from_a_measured_flanking_level() -> None:
    """Part 2, Formula (13): ``Ln,ij = Ln,f,ij,situ − 10 lg(Si llab/(Si,lab lij))``.

    ISO 12354-2 prints no worked example for this route (Annex G characterises
    every flanking path through the elements), so the oracle is the printed
    formula evaluated by hand: with Si = 20 m², Si,lab = 10 m², lij = 4 m and
    llab = 4,5 m the geometry term is 10 lg(20 · 4,5/(10 · 4)) = 10 lg 2,25 =
    3,521 8 dB, and it must vanish when the field geometry equals the
    laboratory one.
    """
    ln_f = np.array([60.0, 58.0, 56.0])
    assert flanking_impact_level_from_flanking_level(
        ln_f, area=20.0, laboratory_area=10.0,
        coupling_length=4.0, laboratory_coupling_length=4.5,
    ) == pytest.approx([56.4782, 54.4782, 52.4782], abs=5e-4)
    assert flanking_impact_level_from_flanking_level(
        ln_f, area=10.0, laboratory_area=10.0,
        coupling_length=2.5, laboratory_coupling_length=2.5,
    ) == pytest.approx(ln_f)


def test_impact_prediction_without_a_direct_path_sums_the_flanking_paths() -> None:
    """Part 2, Formula (2): rooms next to each other carry no direct path.

    ``L'n = 10 lg Σ 10^(Ln,ij/10)`` over the flanking paths alone, so two equal
    flanking paths add exactly 10 lg 2 = 3,010 3 dB and each carries half the
    energy. The result must contain no direct path at all, not a placeholder.
    """
    from phonometry.building.detailed_prediction import BandPath

    values = np.full(BANDS.size, 50.0)
    result = detailed_impact_prediction(
        BANDS,
        flanking_paths=[BandPath("Ff1", "Ff", values), BandPath("Ff2", "Ff", values)],
    )
    assert [p.label for p in result.paths] == ["Ff1", "Ff2"]
    assert result.l_prime_n == pytest.approx(50.0 + 10.0 * np.log10(2.0))
    assert result.fractions == pytest.approx(0.5)
    with pytest.raises(ValueError, match="direct_level"):
        detailed_impact_prediction(BANDS)


def test_table_l16_predicted_flanking_index_matches_the_measurement() -> None:
    """Table L.16, ``R13`` predicted from the elements (Formula 17).

    The prediction reproduces the printed column and lands within about 1 dB
    of the measured ``R13`` of Table L.15 in the weighted rating, which is the
    point of the standard's comparison.
    """
    got = flanking_reduction_index_from_normalized_difference(
        index_i=ref.ISO12354_TABLE_L16_R_SITU,
        index_j=ref.ISO12354_TABLE_L16_R_SITU,
        normalized_velocity_level_difference=ref.ISO12354_TABLE_L16_DV,
        separating_area=ref.ISO12354_TABLE_L14_SEPARATING_AREA,
        coupling_length=ref.ISO12354_TABLE_L14_COUPLING,
    )
    assert got == pytest.approx(ref.ISO12354_TABLE_L16_R13_PRED, abs=TOL)


def test_table_g11_lightweight_impact_paths_and_total() -> None:
    """Tables G.11 to G.13, the lightweight impact chain (Part 2, (11)/(14))."""
    direct = direct_impact_level(
        ref.ISO12354_TABLE_G12_LN_BARE,
        delta_l=ref.ISO12354_TABLE_G12_DELTA_LI,
        delta_l_ceiling=ref.ISO12354_TABLE_G12_DELTA_LDI,
    )
    flanking = flanking_impact_level_from_normalized_difference(
        floor_level=ref.ISO12354_TABLE_G12_LN_BARE,
        index_i=ref.ISO12354_TABLE_G13_R_BARE,
        index_j=ref.ISO12354_TABLE_G13_R_WALL,
        normalized_velocity_level_difference=ref.ISO12354_TABLE_G13_DV,
        area_i=ref.ISO12354_LIGHTWEIGHT_AREA,
        coupling_length=ref.ISO12354_LIGHTWEIGHT_COUPLING,
        delta_l=ref.ISO12354_TABLE_G12_DELTA_LI,
    )
    assert direct == pytest.approx(ref.ISO12354_TABLE_G11_LN_DD, abs=TOL)
    assert flanking == pytest.approx(ref.ISO12354_TABLE_G11_LN_DF, abs=TOL)

    from phonometry.building.detailed_prediction import BandPath

    result = detailed_impact_prediction(
        BANDS, direct_level=direct,
        flanking_paths=[BandPath("Df", "Df", flanking)],
    )
    assert result.l_prime_n == pytest.approx(ref.ISO12354_TABLE_G11_LN_TOTAL, abs=TOL)
    assert result.rating is not None
    assert result.rating.rating == ref.ISO12354_TABLE_G11_RATINGS["total"]


def test_table_g11_path_ratings() -> None:
    """Table G.11, the ISO 717-2 rating of each lightweight impact path."""
    index = [int(np.flatnonzero(np.isclose(BANDS, f))[0])
             for f in (100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                       1000, 1250, 1600, 2000, 2500, 3150)]
    for key, values in (("LnDd", ref.ISO12354_TABLE_G11_LN_DD),
                        ("LnDf", ref.ISO12354_TABLE_G11_LN_DF)):
        spectrum = np.asarray(values, dtype=np.float64)[index]
        rating = weighted_impact_rating(spectrum).rating
        assert rating == ref.ISO12354_TABLE_G11_RATINGS[key], key


# --------------------------------------------------------------------------
# ISO 12354-2 Table B.2 and the reciprocity relation
# --------------------------------------------------------------------------
def test_table_b2_octave_impact_levels_of_monolithic_floors() -> None:
    """ISO 12354-2 Table B.2 (plausibility oracle).

    The four printed monolithic floors are rebuilt from the Table B.1 material
    properties with the laboratory loss factor of Formula (C.3) on the 10 m²
    test opening (2,65 m x 3,75 m, the geometry the companion Table B.2 of
    ISO 12354-1 names). Following the table's own note the levels are computed
    at one-third-octave spacing and combined into octave bands, energetically:
    an octave band collects the energy of its three one-third-octave bands.

    The standard prints neither the radiation factor nor the structural
    reverberation time it used, and it applies an unpublished empirical
    reduction of the tapping-machine force at high frequency (Annex B.1), so
    this is a plausibility oracle: the 125 Hz to 2 kHz octaves that carry the
    ISO 717-2 rating agree to 2 dB and the rating itself to 1 dB, while the
    63 Hz and 4 kHz ends are left out (the closed form has no force reduction
    at 4 kHz, and 63 Hz falls below the critical frequency of the heaviest
    floor).
    """
    third = np.asarray(ref.ISO12354_ANNEX_L_BANDS, dtype=np.float64)
    for material, mass, printed, rating in ref.ISO12354_2_TABLE_B2:
        rho, c_l, eta_int = ref.ISO12354_2_TABLE_B1[material]
        thickness = mass / rho
        fc = 340.0**2 / (1.8 * c_l * thickness)
        sigma = bending_radiation_factor(
            third, critical_frequency=fc, length1=3.75, length2=2.65
        )
        eta = laboratory_total_loss_factor(
            third, mass_per_area=mass, internal_loss_factor=eta_int
        )
        ln = bare_floor_impact_level(
            third, mass_per_area=mass,
            structural_reverberation_time=structural_reverberation_time(third, eta),
            radiation_factor=sigma,
        )
        octave = 10.0 * np.log10(
            (10.0 ** (ln.reshape(-1, 3) / 10.0)).sum(axis=1)
        )
        assert octave[1:6] == pytest.approx(printed[1:6], abs=2.0), material
        assert abs(
            weighted_impact_rating(octave[1:6], "octave").rating - rating
        ) <= 1


def test_reciprocity_relation_closes_on_itself() -> None:
    """Part 2 Formulae (B.3)/(B.4): ``R + Ln`` depends only on frequency."""
    r = np.linspace(30.0, 70.0, BANDS.size)
    ln = reciprocity_impact_level(r, BANDS)
    assert r + ln == pytest.approx(38.0 + 30.0 * np.log10(BANDS))
    ln_oct = reciprocity_impact_level(r, BANDS, bands="octave")
    assert ln_oct - ln == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Laboratory-to-in-situ transfer (Formulae 9 and Part 2 (5))
# --------------------------------------------------------------------------
def test_in_situ_transfer_is_symmetric_between_index_and_impact_level() -> None:
    """Formula (9) and Part 2 Formula (5) move ``R`` and ``Ln`` oppositely."""
    ts_lab = laboratory_total_loss_factor(BANDS, mass_per_area=484.0)
    ts_situ = 2.2 / (BANDS * 0.02)
    lab = 2.2 / (BANDS * ts_lab)
    r = in_situ_reduction_index(50.0, ts_situ, lab)
    ln = in_situ_impact_level(60.0, ts_situ, lab)
    assert (r - 50.0) == pytest.approx(-(ln - 60.0))


def test_in_situ_transfer_direction_matches_the_printed_signs() -> None:
    """Formula (9) and Part 2 Formula (5) fix the *direction*, not just the pair.

    ``Rsitu = R − 10 lg(Ts,situ/Ts,lab)`` and ``Ln,situ = Ln + 10 lg(...)``, so
    an element damped harder in the building than in the test frame
    (``Ts,situ < Ts,lab``) rings shorter, radiates less, and therefore *gains*
    sound reduction index and *loses* impact level. The symmetry test above
    would still pass with both signs flipped; halving and doubling the
    structural reverberation time pins each sign on its own, against the exact
    ``10 lg 2 = 3,010 3 dB`` the formulae must produce.
    """
    lab = np.full(BANDS.size, 0.20)
    half, double = np.full(BANDS.size, 0.10), np.full(BANDS.size, 0.40)
    three = 10.0 * np.log10(2.0)
    assert in_situ_reduction_index(50.0, half, lab) == pytest.approx(50.0 + three)
    assert in_situ_impact_level(60.0, half, lab) == pytest.approx(60.0 - three)
    assert in_situ_reduction_index(50.0, double, lab) == pytest.approx(50.0 - three)
    assert in_situ_impact_level(60.0, double, lab) == pytest.approx(60.0 + three)


def test_in_situ_absorption_length_matches_formula_11(situ: dict) -> None:
    """Formula (11) evaluated directly matches the element's own column."""
    floor = situ["floor"]
    assert in_situ_equivalent_absorption_length(
        BANDS, area=20.0, situ_reverberation_time=floor.reverberation_time
    ) == pytest.approx(floor.absorption_length, rel=1e-12)


def test_in_situ_total_loss_factor_without_perimeter_is_internal_plus_radiation() -> None:
    """With no perimeter losses Formula (C.1) keeps only its first two terms."""
    sigma = np.ones(BANDS.size)
    eta = in_situ_total_loss_factor(
        BANDS, internal_loss_factor=0.01, mass_per_area=400.0, area=20.0,
        critical_frequency=100.0, radiation_factor=sigma, perimeter_absorption=0.0,
    )
    expected = 0.01 + 2.0 * 1.29 * 340.0 / (2.0 * np.pi * BANDS * 400.0)
    assert eta == pytest.approx(expected)


# --------------------------------------------------------------------------
# Validation and options
# --------------------------------------------------------------------------
def test_resonant_only_raises_the_index_below_the_critical_frequency() -> None:
    """Annex B.1/B.3: dropping the forced term can only increase ``R``."""
    sigma = bending_radiation_factor(
        BANDS, critical_frequency=FC_INT, length1=4.0, length2=2.75
    )
    sigma_f = forced_radiation_factor(BANDS, length1=4.0, length2=2.75)
    kwargs = {
        "mass_per_area": M_INT, "critical_frequency": FC_INT,
        "total_loss_factor": 0.03, "radiation_factor": sigma,
        "forced_radiation_factor": sigma_f,
    }
    full = calculated_sound_reduction_index(BANDS, **kwargs)
    resonant = calculated_sound_reduction_index(BANDS, resonant_only=True, **kwargs)
    # The band straddling fc uses the f ~ fc branch, which has no forced term.
    below = BANDS < FC_INT / 2.0 ** (1.0 / 6.0)
    assert np.all(resonant[below] > full[below])
    assert resonant[~below] == pytest.approx(full[~below])


def test_octave_bands_select_the_critical_band_and_rating_range(situ: dict) -> None:
    """``bands="octave"`` widens the ``f ≈ fc`` window and rates 125-2000 Hz."""
    octaves = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    element = iso12354_building.elements()["floor"]
    got = in_situ_element(element, octaves, bands="octave")
    assert got.sound_reduction_index.size == octaves.size
    result = detailed_airborne_prediction(
        octaves, direct_index=got.sound_reduction_index, bands="octave"
    )
    assert result.rating is not None


def test_rating_is_none_when_the_bands_do_not_cover_iso717() -> None:
    """A partial spectrum yields per-band results but no single number."""
    result = detailed_airborne_prediction(
        BANDS[:5], direct_index=np.full(5, 50.0)
    )
    assert result.rating is None
    assert result.r_prime == pytest.approx(np.full(5, 50.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"critical_frequency": -1.0, "length1": 4.0, "length2": 2.75},
        {"critical_frequency": 100.0, "length1": 0.0, "length2": 2.75},
        {"critical_frequency": 100.0, "length1": 4.0, "length2": float("nan")},
    ],
)
def test_radiation_factor_rejects_invalid_geometry(kwargs: dict) -> None:
    """Non-positive or non-finite geometry is refused."""
    with pytest.raises(ValueError):
        bending_radiation_factor(BANDS, **kwargs)


def test_perimeter_absorption_rejects_mismatched_lengths() -> None:
    """Formula (C.4) needs one ``Kij`` per connected element."""
    with pytest.raises(ValueError, match="same length"):
        perimeter_absorption_coefficient([100.0, 200.0], [6.0])


def test_flanking_path_rejects_an_unknown_kind(situ: dict) -> None:
    """Only the three Annex-defined flanking paths exist."""
    with pytest.raises(ValueError, match="kind"):
        airborne_flanking_path(
            label="x", kind="Dd", element_i=situ["floor"], element_j=situ["ext1"],
            vibration_reduction_index=6.0, coupling_length=4.0,
            separating_area=20.0,
        )


def test_band_count_mismatch_is_reported(situ: dict) -> None:
    """A path built on a different band set cannot be combined."""
    too_short = np.zeros(5)
    with pytest.raises(ValueError, match="one value per band"):
        detailed_airborne_prediction(BANDS, direct_index=too_short)


# --------------------------------------------------------------------------
# .plot() coverage
# --------------------------------------------------------------------------
@pytest.mark.parametrize("language", ("en", "es"))
def test_detailed_airborne_plot_draws_bars_and_the_total(airborne, language) -> None:
    """The airborne renderer stacks one bar per path and overlays ``R'``."""
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = airborne.plot(language=language)
    assert ax.patches, "no stacked path bars drawn"
    twin = [other for other in ax.get_figure().axes if other is not ax]
    assert twin and twin[0].lines, "the apparent index is not overlaid"
    assert ax.get_legend() is not None
    plt.close(ax.get_figure())


@pytest.mark.parametrize("language", ("en", "es"))
def test_detailed_impact_plot_draws_bars_and_the_total(impact, language) -> None:
    """The impact renderer is the same figure with ``L'n`` overlaid."""
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = impact.plot(language=language)
    assert ax.patches
    twin = [other for other in ax.get_figure().axes if other is not ax]
    assert twin and twin[0].lines
    plt.close(ax.get_figure())


@pytest.mark.parametrize("language", ("en", "es"))
def test_in_situ_element_plot_draws_both_spectra(situ: dict, language) -> None:
    """The element renderer draws ``Rsitu`` and ``Ln,situ``."""
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = situ["floor"].plot(language=language)
    assert len(ax.lines) == 2
    assert ax.get_xscale() == "log"
    plt.close(ax.get_figure())


def test_plot_rejects_an_unknown_language(airborne) -> None:
    """Only the languages the renderers translate are accepted."""
    pytest.importorskip("matplotlib.pyplot")
    with pytest.raises(ValueError):
        airborne.plot(language="fr")


def test_plot_pools_the_paths_beyond_the_named_set(airborne) -> None:
    """With thirteen paths the renderer names six and pools the rest."""
    plt = pytest.importorskip("matplotlib.pyplot")
    ax = airborne.plot()
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "other paths" in labels
    # Every band's dominant path is named rather than pooled.
    assert set(airborne.dominant) <= set(labels)
    plt.close(ax.get_figure())
