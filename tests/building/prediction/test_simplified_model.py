#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for EN 12354-1/-2:2000 building performance prediction.

The primary oracles are the worked examples in the standards' annexes:
EN 12354-1 Annex H.3 (airborne, R'w = 52 dB) and EN 12354-2 Annex E.3
(impact, L'n,w = 45 dB).
"""

from __future__ import annotations

import math
import warnings

import pytest
import reference_data as ref

from phonometry import building

# --------------------------------------------------------------------------
# Annex E junction Kij — validated against the Annex H input table
# --------------------------------------------------------------------------


def test_kij_rigid_cross_matches_annex_h_floor() -> None:
    # Floor junction: m's/m'f = 460/287 = 1.61; Annex H gives KFf = 12.4,
    # KFd = KDf = 8.9.
    ratio = 460.0 / 287.0
    assert building.junction_vibration_reduction(
        "rigid_cross", "through", ratio
    ) == pytest.approx(12.4, abs=0.05)
    assert building.junction_vibration_reduction(
        "rigid_cross", "corner", ratio
    ) == pytest.approx(8.9, abs=0.05)


def test_kij_rigid_cross_matches_annex_h_ceiling() -> None:
    # Ceiling: ratio 2.00; KFf = 14.4, KFd = KDf = 9.2.
    assert building.junction_vibration_reduction(
        "rigid_cross", "through", 2.0
    ) == pytest.approx(14.4, abs=0.05)
    assert building.junction_vibration_reduction(
        "rigid_cross", "corner", 2.0
    ) == pytest.approx(9.2, abs=0.05)


def test_kij_rigid_t_matches_annex_h_facade() -> None:
    # Facade T-junction: ratio 2.63; KFf = 12.6, KFd = KDf = 6.7.
    assert building.junction_vibration_reduction(
        "rigid_t", "through", 2.63
    ) == pytest.approx(12.6, abs=0.05)
    assert building.junction_vibration_reduction(
        "rigid_t", "corner", 2.63
    ) == pytest.approx(6.7, abs=0.05)


def test_kij_flexible_t_matches_annex_h_internal_wall() -> None:
    # Internal wall (flexible interlayer, E.5): m's/m'f = 460/67 at 500 Hz.
    # KFf = 33.5, KFd = KDf = 15.7.
    ratio = 460.0 / 67.0
    assert building.junction_vibration_reduction(
        "flexible_t", "through", ratio
    ) == pytest.approx(33.5, abs=0.1)
    assert building.junction_vibration_reduction(
        "flexible_t", "corner", ratio
    ) == pytest.approx(15.7, abs=0.1)


def test_kij_flexible_delta1_zero_below_f1() -> None:
    # Below f1 the interlayer term Delta1 vanishes; flexible_t through then
    # equals rigid_t through.
    ratio = 3.0
    below = building.junction_vibration_reduction(
        "flexible_t", "through", ratio, frequency=100.0, f1=125.0
    )
    rigid = building.junction_vibration_reduction("rigid_t", "through", ratio)
    assert below == pytest.approx(rigid, abs=1e-9)


def test_kij_lightweight_facade_minimum() -> None:
    # Through path has a 5 dB floor; at ratio 1 (M = 0) it clamps to 5.
    assert building.junction_vibration_reduction(
        "lightweight_facade", "through", 0.1
    ) == pytest.approx(5.0)
    # corner: 10 + 10|M|, symmetric in M.
    assert building.junction_vibration_reduction(
        "lightweight_facade", "corner", 10.0
    ) == pytest.approx(20.0)
    assert building.junction_vibration_reduction(
        "lightweight_facade", "corner", 0.1
    ) == pytest.approx(20.0)


def test_kij_flexible_t_double_leaf_clamped() -> None:
    # (E.5) K24 = 3.7 + 14.1 M + 5.7 M^2, clamped to -4..0 dB (the print's
    # "0 <= K24 <= -4 dB" read with the bounds in ascending order).
    # M = lg(0.1) = -1: 3.7 - 14.1 + 5.7 = -4.7 -> clamps to -4.
    assert building.junction_vibration_reduction(
        "flexible_t", "double_leaf", 0.1
    ) == pytest.approx(-4.0)
    # M = 0: 3.7 -> clamps to 0 (upper bound).
    assert building.junction_vibration_reduction(
        "flexible_t", "double_leaf", 1.0
    ) == pytest.approx(0.0)
    # M = lg(0.316) = -0.5: 3.7 - 7.05 + 1.425 = -1.925 (inside the clamp).
    m = math.log10(0.316227766)
    assert building.junction_vibration_reduction(
        "flexible_t", "double_leaf", 0.316227766
    ) == pytest.approx(3.7 + 14.1 * m + 5.7 * m * m)


def test_kij_lightweight_double_homogeneous_e7() -> None:
    # (E.7): K13 = 10 + 20 M - 3.3 lg(f/fk), min 10; K12 = 10 + 10|M|
    # + 3.3 lg(f/fk); K24 = 3.0 + 14.1 M + 5.7 M^2 in the per-path
    # convention M = lg(m'perp,i/m'i) (leaf over homogeneous element for the
    # 2-4 path), valid only for mass_ratio < 1/3 (m2/m1 > 3); fk = 500 Hz.
    m = math.log10(5.0)
    assert building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "through", 5.0
    ) == pytest.approx(10.0 + 20.0 * m)  # f = fk -> lg term vanishes
    assert building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "through", 5.0, frequency=1000.0
    ) == pytest.approx(10.0 + 20.0 * m - 3.3 * math.log10(2.0))
    # The 10 dB floor binds for small ratios.
    assert building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "through", 1.0
    ) == pytest.approx(10.0)
    assert building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "corner", 5.0, frequency=1000.0
    ) == pytest.approx(10.0 + 10.0 * m + 3.3 * math.log10(2.0))
    m24 = math.log10(0.2)  # mass_ratio = m1/m2 = 1/5 (m2/m1 = 5 > 3)
    k24 = building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "double_leaf", 0.2
    )
    assert k24 == pytest.approx(3.0 + 14.1 * m24 + 5.7 * m24 * m24)
    # The per-path form equals the 2000 print's figure-axis recast
    # 3,0 - 14,1 lg(m2/m1) + 5,7 lg(m2/m1)^2 (same relation, see ERRATA).
    assert k24 == pytest.approx(3.0 - 14.1 * m + 5.7 * m * m)
    # The Figure E.9 curve anchors: about -2,4 dB at m2/m1 = 3 and
    # -5,4 dB at m2/m1 = 10.
    assert building.junction_vibration_reduction(
        "lightweight_double_homogeneous", "double_leaf", 0.1
    ) == pytest.approx(-5.4)
    with pytest.raises(ValueError, match="mass_ratio = m'⊥,i/m'i < 1/3"):
        building.junction_vibration_reduction(
            "lightweight_double_homogeneous", "double_leaf", 0.5
        )
    with pytest.raises(ValueError, match=r"mass_ratio = m'⊥,i/m'i < 1/3"):
        building.junction_vibration_reduction(
            "lightweight_double_homogeneous", "double_leaf", 5.0
        )


def test_kij_lightweight_double_coupled_e8() -> None:
    # (E.8): K13 as E.7; K12 = 10 + 10|M| - 3.3 lg(f/fk); no K24 branch.
    m = math.log10(4.0)
    assert building.junction_vibration_reduction(
        "lightweight_double_coupled", "through", 4.0, frequency=2000.0
    ) == pytest.approx(10.0 + 20.0 * m - 3.3 * math.log10(4.0))
    assert building.junction_vibration_reduction(
        "lightweight_double_coupled", "corner", 4.0, frequency=2000.0
    ) == pytest.approx(10.0 + 10.0 * m - 3.3 * math.log10(4.0))
    with pytest.raises(ValueError, match="no 'double_leaf'"):
        building.junction_vibration_reduction(
            "lightweight_double_coupled", "double_leaf", 4.0
        )


def test_kij_corner_and_thickness_change_e9() -> None:
    # (E.9) A corner: K12 = 15|M| - 3, minimum -2 dB; B change: K12 = 5 M^2 - 5.
    assert building.junction_vibration_reduction(
        "corner", "corner", 10.0
    ) == pytest.approx(12.0)
    assert building.junction_vibration_reduction(
        "corner", "corner", 0.1
    ) == pytest.approx(12.0)
    # |M| small: 15*0 - 3 = -3 clamps to the -2 dB minimum.
    assert building.junction_vibration_reduction(
        "corner", "corner", 1.0
    ) == pytest.approx(-2.0)
    m = math.log10(3.0)
    assert building.junction_vibration_reduction(
        "thickness_change", "through", 3.0
    ) == pytest.approx(5.0 * m * m - 5.0)
    # M = 0 (no change) gives the -5 dB minimum of the parabola.
    assert building.junction_vibration_reduction(
        "thickness_change", "through", 1.0
    ) == pytest.approx(-5.0)
    with pytest.raises(ValueError, match=r"A 'corner' junction has the single path"):
        building.junction_vibration_reduction("corner", "through", 10.0)
    with pytest.raises(ValueError, match="single in-line path"):
        building.junction_vibration_reduction("thickness_change", "corner", 3.0)


def test_kij_double_leaf_path_rejected_for_rigid_junctions() -> None:
    with pytest.raises(ValueError, match="no 'double_leaf'"):
        building.junction_vibration_reduction("rigid_cross", "double_leaf", 2.0)


def test_kij_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="'mass_ratio' must be positive"):
        building.junction_vibration_reduction("rigid_cross", "through", 0.0)
    with pytest.raises(ValueError, match="'mass_ratio' must be positive"):
        building.junction_vibration_reduction("rigid_cross", "through", -1.0)
    with pytest.raises(ValueError, match="'junction_type' must be one of"):
        building.junction_vibration_reduction("unknown", "through", 1.0)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="'path' must be 'through', 'corner' or 'double_leaf'"
    ):
        building.junction_vibration_reduction("rigid_cross", "diagonal", 1.0)  # type: ignore[arg-type]


def test_kij_min_formula_29() -> None:
    # Kij,min = 10 lg[ lf * l0 * (1/Si + 1/Sj) ], l0 = 1 m. Hand check.
    lf, si, sj = 4.5, 11.5, 19.6
    expected = 10.0 * math.log10(4.5 * (1.0 / 11.5 + 1.0 / 19.6))
    assert building.junction_min_vibration_reduction(lf, si, sj) == pytest.approx(
        expected
    )
    with pytest.raises(
        ValueError,
        match="'coupling_length', 's_i' and 's_j' must be positive",
    ):
        building.junction_min_vibration_reduction(0.0, 11.5, 19.6)


# --------------------------------------------------------------------------
# Lining composition (Formulas 30/31)
# --------------------------------------------------------------------------


def test_combine_linings() -> None:
    # Two layers: larger + half the smaller. Annex H second example uses
    # 14 + 0.5*14 = 21 for a floating floor on both sides.
    assert building.combine_linings(14.0, 14.0) == pytest.approx(21.0)
    assert building.combine_linings(10.0, 4.0) == pytest.approx(12.0)
    # single lining
    assert building.combine_linings(8.0, 0.0) == pytest.approx(8.0)


# --------------------------------------------------------------------------
# Flanking path construction (Formula 28a)
# --------------------------------------------------------------------------


def test_flanking_path_reproduces_annex_h_floor_ff() -> None:
    # Floor Ff: (49+49)/2 + 12.4 + 10 lg(11.5/4.5) = 65.5 dB.
    path = building.flanking_path(
        label="floor-Ff",
        kind="Ff",
        r_source=49.0,
        r_receive=49.0,
        k_ij=12.4,
        separating_area=11.5,
        coupling_length=4.5,
    )
    assert path.r_ij_w == pytest.approx(65.5, abs=0.05)


def test_flanking_element_triplet() -> None:
    ff, df, fd = building.flanking_element(
        label="floor",
        r_flanking=49.0,
        r_separating=57.0,
        k_ff=12.4,
        k_fd=8.9,
        k_df=8.9,
        separating_area=11.5,
        coupling_length=4.5,
    )
    assert ff.r_ij_w == pytest.approx(65.5, abs=0.05)
    # Fd and Df: (49+57)/2 + 8.9 + 4.08 = 66.0
    assert fd.r_ij_w == pytest.approx(66.0, abs=0.05)
    assert df.r_ij_w == pytest.approx(66.0, abs=0.05)
    assert (ff.kind, df.kind, fd.kind) == ("Ff", "Df", "Fd")


def test_flanking_element_applies_kij_min_from_geometry() -> None:
    # Clause 4.4.2 (Eq. (23) in the BS:2000 print): with the flanking area, each path is
    # clamped to its own Kij,min. Small light elements (lf = 4 m, SF = 1.5 m2)
    # give KFf,min = 10 lg(4*(2/1.5)) = 7.27 dB, above a raw 5 dB Kij.
    lf, sf, ss = 4.0, 1.5, 11.5
    ff, df, fd = building.flanking_element(
        label="light",
        r_flanking=30.0,
        r_separating=57.0,
        k_ff=5.0,
        k_fd=5.0,
        k_df=5.0,
        separating_area=ss,
        coupling_length=lf,
        flanking_area=sf,
    )
    min_ff = building.junction_min_vibration_reduction(lf, sf, sf)
    min_cross = building.junction_min_vibration_reduction(lf, sf, ss)
    assert min_ff == pytest.approx(7.27, abs=0.01)
    base_ff = 30.0 + 10.0 * math.log10(ss / lf)
    base_cross = (30.0 + 57.0) / 2.0 + 10.0 * math.log10(ss / lf)
    assert ff.r_ij_w == pytest.approx(base_ff + min_ff)
    assert df.r_ij_w == pytest.approx(base_cross + max(min_cross, 5.0))
    assert fd.r_ij_w == pytest.approx(base_cross + max(min_cross, 5.0))
    # Without the flanking area the raw Kij is used unchanged (documented).
    ff_raw, _, _ = building.flanking_element(
        label="light",
        r_flanking=30.0,
        r_separating=57.0,
        k_ff=5.0,
        k_fd=5.0,
        k_df=5.0,
        separating_area=ss,
        coupling_length=lf,
    )
    assert ff_raw.r_ij_w == pytest.approx(base_ff + 5.0)


def test_flanking_element_kij_min_is_noop_for_annex_h_geometry() -> None:
    # The Annex H.3 floor junction floors are far below the tabulated Kij
    # (Kij,min ~ -2 dB there), so passing the areas leaves the oracle intact.
    ff, df, fd = building.flanking_element(
        label="floor",
        r_flanking=49.0,
        r_separating=57.0,
        k_ff=12.4,
        k_fd=8.9,
        k_df=8.9,
        separating_area=11.5,
        coupling_length=4.5,
        flanking_area=13.5,
    )
    assert ff.r_ij_w == pytest.approx(65.5, abs=0.05)
    assert fd.r_ij_w == pytest.approx(66.0, abs=0.05)
    assert df.r_ij_w == pytest.approx(66.0, abs=0.05)


def test_flanking_path_invalid() -> None:
    with pytest.raises(ValueError, match="'kind' must be 'Ff', 'Df' or 'Fd'"):
        building.flanking_path(
            label="x",
            kind="XX",
            r_source=40.0,
            r_receive=40.0,  # type: ignore[arg-type]
            k_ij=5.0,
            separating_area=11.5,
            coupling_length=4.5,
        )
    with pytest.raises(
        ValueError,
        match="'separating_area' and 'coupling_length' must be positive",
    ):
        building.flanking_path(
            label="x",
            kind="Ff",
            r_source=40.0,
            r_receive=40.0,
            k_ij=5.0,
            separating_area=-1.0,
            coupling_length=4.5,
        )


def test_flanking_path_kij_min_clamps() -> None:
    # Clause 4.4.2 floor: k_ij below kij_min is raised, so Rij,w rises with it;
    # a k_ij already above the floor (and kij_min=None) is left untouched.
    kwargs = {
        "label": "floor-Ff",
        "kind": "Ff",
        "r_source": 49.0,
        "r_receive": 49.0,
        "separating_area": 11.5,
        "coupling_length": 4.5,
    }
    unclamped = building.flanking_path(k_ij=2.0, **kwargs)  # type: ignore[arg-type]
    clamped = building.flanking_path(k_ij=2.0, kij_min=12.4, **kwargs)  # type: ignore[arg-type]
    assert clamped.r_ij_w == pytest.approx(unclamped.r_ij_w + 10.4, abs=0.05)
    # Floor at or below k_ij is a no-op (matches the raw Annex-H Ff path).
    above = building.flanking_path(k_ij=12.4, kij_min=8.0, **kwargs)  # type: ignore[arg-type]
    assert above.r_ij_w == pytest.approx(65.5, abs=0.05)


# --------------------------------------------------------------------------
# Airborne prediction — Formula (26), Annex H oracle
# --------------------------------------------------------------------------


def _annex_h_paths() -> list:
    """The 12 flanking paths of EN 12354-1 Annex H.3 (built from raw inputs)."""
    ss = 11.5
    paths = []
    # element: (label, Rw, KFf, KFd=KDf, lf)
    elements = [
        ("floor", 49.0, 12.4, 8.9, 4.5),
        ("ceiling", 46.0, 14.4, 9.2, 4.5),
        ("facade", 42.0, 12.6, 6.7, 2.55),
        ("intwall", 33.0, 33.5, 15.7, 2.55),
    ]
    for label, rw, kff, kfd, lf in elements:
        ff, df, fd = building.flanking_element(
            label=label,
            r_flanking=rw,
            r_separating=57.0,
            k_ff=kff,
            k_fd=kfd,
            k_df=kfd,
            separating_area=ss,
            coupling_length=lf,
        )
        paths += [ff, df, fd]
    return paths


def test_airborne_annex_h_example() -> None:
    result = building.predicted_airborne_insulation(
        r_direct=57.0, flanking_paths=_annex_h_paths()
    )
    # Standard result: R'w = 52.2 -> rounds to 52 dB.
    assert result.r_prime_w == pytest.approx(52.2, abs=0.1)
    assert round(result.r_prime_w) == 52
    assert result.r_direct_w == pytest.approx(57.0)
    # 1 direct + 12 flanking paths, energy fractions sum to 1.
    assert len(result.paths) == 13
    assert sum(c.fraction for c in result.paths) == pytest.approx(1.0)


def test_airborne_annex_h_all_twelve_path_values() -> None:
    """Every printed H.3 path Rij,w reproduces to the table's 0,1 dB."""
    by_label = {
        p.label: p.r_w
        for p in building.predicted_airborne_insulation(
            r_direct=57.0, flanking_paths=_annex_h_paths()
        ).paths
    }
    for element, (r_ff, r_cross) in ref.EN12354_1_ANNEX_H3_PATH_RW.items():
        assert by_label[f"{element}-Ff"] == pytest.approx(r_ff, abs=0.05)
        assert by_label[f"{element}-Fd"] == pytest.approx(r_cross, abs=0.05)
        assert by_label[f"{element}-Df"] == pytest.approx(r_cross, abs=0.05)


def test_airborne_annex_h_dnt_closures() -> None:
    """Formula (5b) closes both H.3 examples to DnT,w = 54 dB."""
    v, ss = ref.EN12354_1_ANNEX_H3_VOLUME, ref.EN12354_1_ANNEX_H3_SEPARATING_AREA
    first = building.standardized_level_difference(52.2, v, ss)
    assert round(first) == ref.EN12354_1_ANNEX_H3_DNT_W
    # The printed 53,8 dB uses the standard's own V/(3 S) rounding; the exact
    # 0,32 V/Ss form sits 0,18 dB below it.
    assert first == pytest.approx(ref.EN12354_1_ANNEX_H3_DNT_W_PRINTED, abs=0.2)
    second = building.standardized_level_difference(52.7, v, ss)
    assert round(second) == ref.EN12354_1_ANNEX_H3_DNT_W_SECOND


def test_airborne_second_example_floating_floor() -> None:
    # Add a floating floor (both sides) on the floor element: ΔRw = 14 dB.
    # Ff (two-side) = combine(14, 14) = 21; Fd, Df (one side) = 14.
    ss = 11.5
    paths = []
    elements = [
        (
            "floor",
            49.0,
            12.4,
            8.9,
            4.5,
            building.combine_linings(14.0, 14.0),
            14.0,
        ),
        ("ceiling", 46.0, 14.4, 9.2, 4.5, 0.0, 0.0),
        ("facade", 42.0, 12.6, 6.7, 2.55, 0.0, 0.0),
        ("intwall", 33.0, 33.5, 15.7, 2.55, 0.0, 0.0),
    ]
    for label, rw, kff, kfd, lf, dr_ff, dr_other in elements:
        ff, df, fd = building.flanking_element(
            label=label,
            r_flanking=rw,
            r_separating=57.0,
            k_ff=kff,
            k_fd=kfd,
            k_df=kfd,
            separating_area=ss,
            coupling_length=lf,
            delta_r_ff=dr_ff,
            delta_r_fd=dr_other,
            delta_r_df=dr_other,
        )
        paths += [ff, df, fd]
    result = building.predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)
    # Standard result: R'w = 52.7 -> 53 dB.
    assert result.r_prime_w == pytest.approx(52.7, abs=0.1)
    assert round(result.r_prime_w) == 53


def test_airborne_no_flanking_equals_direct() -> None:
    result = building.predicted_airborne_insulation(r_direct=55.0)
    assert result.r_prime_w == pytest.approx(55.0)
    assert result.dominant.kind == "Dd"
    assert result.paths[0].fraction == pytest.approx(1.0)


def test_airborne_direct_lining() -> None:
    result = building.predicted_airborne_insulation(r_direct=52.0, delta_r_direct=5.0)
    assert result.r_direct_w == pytest.approx(57.0)
    assert result.r_prime_w == pytest.approx(57.0)


def test_airborne_adding_flanking_strictly_lowers() -> None:
    base = building.predicted_airborne_insulation(r_direct=57.0).r_prime_w
    one = building.predicted_airborne_insulation(
        r_direct=57.0,
        flanking_paths=[
            building.flanking_path(
                label="f",
                kind="Ff",
                r_source=49.0,
                r_receive=49.0,
                k_ij=12.4,
                separating_area=11.5,
                coupling_length=4.5,
            )
        ],
    ).r_prime_w
    two = building.predicted_airborne_insulation(
        r_direct=57.0,
        flanking_paths=[
            building.flanking_path(
                label="f",
                kind="Ff",
                r_source=49.0,
                r_receive=49.0,
                k_ij=12.4,
                separating_area=11.5,
                coupling_length=4.5,
            ),
            building.flanking_path(
                label="g",
                kind="Df",
                r_source=57.0,
                r_receive=49.0,
                k_ij=8.9,
                separating_area=11.5,
                coupling_length=4.5,
            ),
        ],
    ).r_prime_w
    assert one < base
    assert two < one


def test_airborne_energy_composition_two_equal_paths() -> None:
    # Two identical paths each at R: R' = R - 10 lg 2 = R - 3.0103.
    r = 50.0
    p = building.flanking_path(
        label="p",
        kind="Ff",
        r_source=r,
        r_receive=r,
        k_ij=0.0,
        separating_area=1.0,
        coupling_length=1.0,
    )
    # Ff at r_source=r_receive=r, k=0, coupling term 10 lg(1/1)=0 -> r_ij = r.
    assert p.r_ij_w == pytest.approx(r)
    # Direct at r, plus one identical flanking path -> two equal paths.
    result = building.predicted_airborne_insulation(r_direct=r, flanking_paths=[p])
    assert result.r_prime_w == pytest.approx(r - 10.0 * math.log10(2.0))
    assert result.paths[0].fraction == pytest.approx(0.5)


def test_airborne_dominant_path_is_weakest() -> None:
    # A single very weak flanking path (low R) dominates the energy.
    weak = building.flanking_path(
        label="weak",
        kind="Ff",
        r_source=30.0,
        r_receive=30.0,
        k_ij=0.0,
        separating_area=1.0,
        coupling_length=1.0,
    )
    result = building.predicted_airborne_insulation(
        r_direct=60.0, flanking_paths=[weak]
    )
    assert result.dominant.label == "weak"


# --------------------------------------------------------------------------
# Impact prediction — Formula (21), Annex E.3 oracle
# --------------------------------------------------------------------------


def test_equivalent_impact_level_annex_e3() -> None:
    # Concrete floor m' = 322 kg/m² -> Ln,w,eq = 164 - 35 lg(322) = 76.2 dB.
    assert building.equivalent_impact_level(322.0) == pytest.approx(76.2, abs=0.1)
    with pytest.raises(ValueError, match="'mass_per_area' must be positive"):
        building.equivalent_impact_level(0.0)


def test_impact_flanking_correction_annex_e3() -> None:
    # sep 322 -> row 300; flanking mean 145 -> col 150; Table 1 -> K = 2.
    assert building.impact_flanking_correction(322.0, 145.0) == 2


def test_impact_flanking_correction_table_cells() -> None:
    # Spot-check tabulated corners of Table 1.
    assert building.impact_flanking_correction(100.0, 100.0) == 1
    assert building.impact_flanking_correction(100.0, 500.0) == 0
    assert building.impact_flanking_correction(900.0, 100.0) == 6
    assert building.impact_flanking_correction(900.0, 500.0) == 2
    assert building.impact_flanking_correction(500.0, 100.0) == 4
    with pytest.raises(
        ValueError,
        match="'separating_mass' and 'flanking_mass' must be positive",
    ):
        building.impact_flanking_correction(-1.0, 100.0)


def test_impact_prediction_annex_e3() -> None:
    # L'n,w = Ln,w,eq - ΔLw + K = 76 - 33 + 2 = 45 dB.
    ln_eq = building.equivalent_impact_level(322.0)
    k = building.impact_flanking_correction(322.0, 145.0)
    result = building.predicted_impact_insulation(
        ln_w_eq=round(ln_eq), delta_l_w=33.0, k_correction=k
    )
    assert result.l_prime_n_w == pytest.approx(45.0)
    assert result.k_correction == 2


def test_impact_prediction_from_raw_equivalent() -> None:
    # Using the unrounded equivalent level: 76.2 - 33 + 2 = 45.2 -> 45 dB.
    result = building.predicted_impact_insulation(
        ln_w_eq=building.equivalent_impact_level(322.0),
        delta_l_w=33.0,
        k_correction=2.0,
    )
    assert round(result.l_prime_n_w) == 45


def test_standardized_impact_level_annex_e3() -> None:
    # Exact Formula (3): L'nT,w = L'n,w - 10 lg(0.032 V) = 45 - 10 lg(1.6)
    # = 42.96 dB. Annex E.3's own "10 lg(V/30)" rounding gives 42.78; both
    # round to 43 dB.
    exact = 45.0 - 10.0 * math.log10(0.032 * 50.0)
    assert building.standardized_impact_level(45.0, 50.0) == pytest.approx(exact)
    assert building.standardized_impact_level(45.0, 50.0) == pytest.approx(
        42.96, abs=0.01
    )
    assert round(building.standardized_impact_level(45.0, 50.0)) == 43
    with pytest.raises(ValueError, match="'volume' must be positive"):
        building.standardized_impact_level(45.0, 0.0)


def test_standardized_impact_level_exact_vs_e3_rounding() -> None:
    # The exact 0.032 V factor sits 10 lg(31.25/30) = 0.177 dB above the E.3
    # V/30 chain for every volume.
    for v in (20.0, 50.0, 120.0):
        exact = building.standardized_impact_level(60.0, v)
        e3 = 60.0 - 10.0 * math.log10(v / 30.0)
        assert exact - e3 == pytest.approx(10.0 * math.log10(31.25 / 30.0))


def test_standardized_level_difference_annex_h3_closure() -> None:
    # Formula (5b): DnT,w = R'w + 10 lg(0.32 V/Ss). H.3 prints
    # 52.2 + 10 lg[50/(3 x 11.5)] = 53.8 (V/(3 S) rounding of 1/0.32 = 3.125);
    # the exact form gives 53.6; both round to 54 dB.
    dnt = building.standardized_level_difference(52.2, 50.0, 11.5)
    assert dnt == pytest.approx(52.2 + 10.0 * math.log10(0.32 * 50.0 / 11.5))
    assert dnt == pytest.approx(53.63, abs=0.01)
    assert round(dnt) == 54
    # Second H.3 example (floating floor): 52.7 + 1.6 = 54.3 -> 54 dB.
    assert round(building.standardized_level_difference(52.7, 50.0, 11.5)) == 54
    with pytest.raises(ValueError, match="'volume' must be positive"):
        building.standardized_level_difference(52.2, 0.0, 11.5)
    with pytest.raises(ValueError, match="'separating_area' must be positive"):
        building.standardized_level_difference(52.2, 50.0, 0.0)


def test_equivalent_impact_level_warns_outside_envelope() -> None:
    # Annex B covers 100-600 kg/m2; outside it the closed form extrapolates.
    with pytest.warns(
        UserWarning, match=r"mass_per_area = .* lies outside the .* envelope"
    ):
        building.equivalent_impact_level(50.0)
    with pytest.warns(
        UserWarning, match=r"mass_per_area = .* lies outside the .* envelope"
    ):
        building.equivalent_impact_level(800.0)
    # Inside the envelope no warning is emitted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        building.equivalent_impact_level(322.0)


def test_impact_covering_improves_level() -> None:
    # A better covering (larger ΔLw) lowers L'n,w.
    a = building.predicted_impact_insulation(ln_w_eq=76.0, delta_l_w=20.0).l_prime_n_w
    b = building.predicted_impact_insulation(ln_w_eq=76.0, delta_l_w=30.0).l_prime_n_w
    assert b < a


def test_impact_non_finite_rejected() -> None:
    with pytest.raises(ValueError, match="'ln_w_eq' must be a finite number"):
        building.predicted_impact_insulation(ln_w_eq=float("nan"))


# --------------------------------------------------------------------------
# Direct construction of the exported result types
# --------------------------------------------------------------------------
#
# The builders above refuse a non-finite input, but both results and the path
# contribution are exported and can be built by hand, and the EN 12354 fiche
# rounds every rating it boxes for display: ``math.floor`` answered a NaN with
# "cannot convert float NaN to integer" from inside the renderer, naming
# neither the field nor the result it came from.


@pytest.mark.parametrize("field", ["r_w", "fraction"])
def test_path_contribution_non_finite_refused(field: str) -> None:
    """A hand-built path contribution names the field that is not finite."""
    values = {"r_w": 55.0, "fraction": 1.0}
    values[field] = float("nan")
    with pytest.raises(ValueError, match=rf"'{field}' must be a finite number"):
        building.PathContribution(label="Dd", kind="Dd", **values)


@pytest.mark.parametrize("field", ["r_prime_w", "r_direct_w"])
def test_airborne_result_non_finite_refused(field: str) -> None:
    """A hand-built airborne prediction names the rating that is not finite."""
    path = building.PathContribution(label="Dd", kind="Dd", r_w=55.0, fraction=1.0)
    values = {"r_prime_w": 52.0, "r_direct_w": 55.0}
    values[field] = float("nan")
    with pytest.raises(ValueError, match=rf"'{field}' must be a finite number"):
        building.AirbornePredictionResult(paths=(path,), dominant=path, **values)


@pytest.mark.parametrize(
    "field", ["l_prime_n_w", "ln_w_eq", "delta_l_w", "k_correction"]
)
def test_impact_result_non_finite_refused(field: str) -> None:
    """A hand-built impact prediction names the term that is not finite."""
    values = {
        "l_prime_n_w": 58.0,
        "ln_w_eq": 76.0,
        "delta_l_w": 20.0,
        "k_correction": 2.0,
    }
    values[field] = float("nan")
    with pytest.raises(ValueError, match=rf"'{field}' must be a finite number"):
        building.ImpactPredictionResult(**values)


# --------------------------------------------------------------------------
# Annex E junction Kij — independent literature oracles
# --------------------------------------------------------------------------


def test_kij_rigid_t_vigran_aerated_concrete_example() -> None:
    # Vigran, Building Acoustics (Taylor & Francis, 2008), Sec. 9.3.2.3,
    # p. 353: aerated-concrete test building, 125 mm walls and a 150 mm floor
    # at 600 kg/m3 (m' = 75 and 90 kg/m2). For the wall-floor-wall path Ff
    # across the rigid T-junction, "using the expression for a T-junction in
    # EN 12354-1 we arrive at the value 6.9 dB for Kij". The book prints one
    # decimal; the exact Annex E value is 6.85 dB, so 0.05 dB covers the
    # rounding.
    assert building.junction_vibration_reduction(
        "rigid_t", "through", 90.0 / 75.0
    ) == pytest.approx(6.9, abs=0.05)


def test_kij_rigid_t_equal_masses_alba() -> None:
    # Alba, Escuder, Ramis, del Rey and Segovia, J. Vib. Acoust. 134 (2012)
    # 021009, Table 1: "The standard gives a fixed value of 5.7 for the
    # vibration [reduction] index in a rigid T-junction when the surface
    # densities of the elements forming the junction are equal"
    # (K12354 = 5.7 dB). Exact value, no rounding involved.
    assert building.junction_vibration_reduction(
        "rigid_t", "corner", 1.0
    ) == pytest.approx(5.7, abs=1e-9)


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        # Alba et al. (2012) Table 1, T-junction with elastic interlayer and
        # equal surface densities: K12 = 10 lg f - 10 lg f1 + 5.7 dB for
        # f > f1, evaluated by hand from the paper's expression with the
        # f1 = 125 Hz the paper uses for its theoretical curve, over the
        # paper's fitted range 200-1250 Hz. Hand values to 0.01 dB.
        (200.0, 7.74),
        (500.0, 11.72),
        (1250.0, 15.70),
    ],
)
def test_kij_flexible_t_alba_frequency_law(frequency: float, expected: float) -> None:
    assert building.junction_vibration_reduction(
        "flexible_t", "corner", 1.0, frequency=frequency, f1=125.0
    ) == pytest.approx(expected, abs=0.005)


# Schiavi & Astolfi, Applied Acoustics 71 (2010) 523-530, Table 3: measured
# average vibration reduction index (one-third-octave bands, in-situ per
# EN ISO 10848-1) for junctions between beam-and-block floors and brickwork
# walls in Southern-European buildings. CB = concrete beam-brick wall,
# BB = ribbed slab with brick blocks-brick wall.
_SCHIAVI_TABLE3_FREQS = (
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
    4000.0,
    5000.0,
)
_SCHIAVI_TABLE3_CB = (
    18.0,
    17.5,
    17.0,
    16.5,
    16.0,
    15.5,
    15.0,
    14.5,
    14.0,
    14.0,
    14.0,
    14.0,
    15.0,
    16.0,
    17.0,
    18.0,
    19.0,
    20.0,
)
_SCHIAVI_TABLE3_BB = (
    17.0,
    16.5,
    16.0,
    15.5,
    15.0,
    14.5,
    14.0,
    13.5,
    13.0,
    12.5,
    12.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
)
# Table 3 "Single number (average)" row.
_SCHIAVI_SINGLE_CB = 16.0
_SCHIAVI_SINGLE_BB = 14.0
# Table 1, mwall/mfloor of the 13 BB junctions (junction numbers 3, 5, 7, 11,
# 13, 14, 15, 16, 17, 18, 19, 21, 29); all are cross junctions.
_SCHIAVI_BB_RATIOS = (0.4, 0.4, 0.2, 0.3, 0.3, 0.3, 0.4, 0.3, 0.3, 0.2, 0.3, 0.4, 0.2)
# Table 2, "essential" mass ratios mEwall/mEfloor of the 19 CB junctions
# (junction numbers 1, 2, 4, 6, 8, 9, 10, 12, 20, 22, 23, 24, 25, 26, 27,
# 28, 30, 31, 32).
_SCHIAVI_CB_RATIOS = (
    0.1,
    0.1,
    0.1,
    0.2,
    0.1,
    0.2,
    0.2,
    0.2,
    0.2,
    0.1,
    0.2,
    0.1,
    0.1,
    0.1,
    0.2,
    0.2,
    0.1,
    0.1,
    0.2,
)


def test_schiavi_table3_transcription_matches_single_numbers() -> None:
    # The paper's single-number rows are "obtained from the frequency
    # averaging from 100 Hz to 5 kHz" and printed as integers: pin the
    # transcribed per-band columns to them (0.5 dB integer rounding).
    n_bands = len(_SCHIAVI_TABLE3_FREQS)
    assert len(_SCHIAVI_TABLE3_CB) == len(_SCHIAVI_TABLE3_BB) == n_bands
    assert sum(_SCHIAVI_TABLE3_CB) / n_bands == pytest.approx(
        _SCHIAVI_SINGLE_CB, abs=0.5
    )
    assert sum(_SCHIAVI_TABLE3_BB) / n_bands == pytest.approx(
        _SCHIAVI_SINGLE_BB, abs=0.5
    )


@pytest.mark.parametrize(
    ("ratios", "measured"),
    [
        (_SCHIAVI_BB_RATIOS, _SCHIAVI_SINGLE_BB),
        (_SCHIAVI_CB_RATIOS, _SCHIAVI_SINGLE_CB),
    ],
    ids=["bb-junctions", "cb-junctions"],
)
def test_schiavi_measured_kij_anchors_annex_e_prediction(
    ratios: tuple[float, ...], measured: float
) -> None:
    # Schiavi & Astolfi compare their measured Kij with the EN 12354-1
    # Annex E rigid-cross corner formula (their Eq. (5), chosen "due to the
    # prevalence of cross junctions in the full set of data") and report that
    # the differences "are normally distributed, with an average value of
    # 3.4 dB" (Sec. 5.1); the essential-mass CB comparison of Sec. 4.2 gives
    # "about 4 dB". This is a tolerance anchor, not a digit oracle: the
    # library prediction evaluated at the junctions' mean mass ratio must sit
    # within 4.5 dB (the Sec. 4.2 "about 4 dB" figure plus the paper's
    # ~0.5 dB standard deviation of the mean) of the measured single-number
    # Kij. It fixes the
    # sign convention of M and the Annex E constants against real data.
    mean_ratio = sum(ratios) / len(ratios)
    predicted = building.junction_vibration_reduction(
        "rigid_cross", "corner", mean_ratio
    )
    assert abs(measured - predicted) <= 4.5
    # And the corner branch must be direction-symmetric (the paper notes the
    # choice of the mass ratio for M is arbitrary for corner transmission).
    assert building.junction_vibration_reduction(
        "rigid_cross", "corner", 1.0 / mean_ratio
    ) == pytest.approx(predicted, abs=1e-9)
