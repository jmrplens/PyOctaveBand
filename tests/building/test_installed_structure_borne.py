#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for EN 12354-5:2009 installed structure-borne sound from equipment.

Anchored on the standard's closed-form chain: the coupling term (Formula 19b)
and its force- and velocity-source limits (Formulae 19c/19d), the installed
power level L_Ws,inst = L_Ws,c − D_C (Formula 18b), the per-path normalised SPL
(Formula 18a) and the energetic path sum (Formula 17).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phonometry import (
    InstalledSourceResult,
    coupling_term,
    coupling_term_force_source,
    coupling_term_velocity_source,
    installed_source_prediction,
    installed_structure_borne_power_level,
    structure_borne_pressure_level_path,
    total_structure_borne_pressure_level,
)

S0 = 10.0


# ---------------------------------------------------------------------------
# Coupling term (Formula 19b) and its limiting forms (19c/19d)
# ---------------------------------------------------------------------------
def test_coupling_term_hand_value() -> None:
    ys, yi = 1e-4 + 0j, 1e-5 + 0j
    expected = 10.0 * math.log10(abs(ys + yi) ** 2 / (abs(ys) * yi.real))
    assert coupling_term(ys, yi) == pytest.approx(expected)


def test_coupling_reduces_to_force_source_limit() -> None:
    """|Y_s| >> |Y_i|: Formula 19b -> 19c (10 lg(|Y_s|/Re Y_i))."""
    ys, yi = 1e-3 + 0j, 1e-7 + 0j
    assert coupling_term(ys, yi) == pytest.approx(
        float(coupling_term_force_source(ys, yi)), abs=1e-2
    )


def test_coupling_reduces_to_velocity_source_limit() -> None:
    """|Y_s| << |Y_i|: Formula 19b -> 19d (−10 lg(|Y_s| Re Z_i))."""
    ys, yi = 1e-8 + 0j, 1e-4 + 0j
    zi = 1.0 / yi
    assert coupling_term(ys, yi) == pytest.approx(
        float(coupling_term_velocity_source(ys, zi)), abs=1e-2
    )


def test_coupling_elastic_support_reduces_transmission() -> None:
    # Adding a soft elastic support (large transfer mobility) raises D_C.
    ys, yi = 1e-4 + 0j, 1e-5 + 0j
    rigid = float(coupling_term(ys, yi))
    isolated = float(coupling_term(ys, yi, transfer_mobility=1e-3 + 0j))
    assert isolated > rigid


# ---------------------------------------------------------------------------
# Installed power (Formula 18b)
# ---------------------------------------------------------------------------
def test_installed_power_level() -> None:
    lw = installed_structure_borne_power_level(80.0, 10.828)
    assert float(lw) == pytest.approx(69.172)


def test_installed_power_per_band() -> None:
    lwc = np.array([75.0, 78.0, 80.0])
    dc = np.array([8.0, 9.0, 10.0])
    assert np.allclose(
        installed_structure_borne_power_level(lwc, dc), lwc - dc
    )


# ---------------------------------------------------------------------------
# Path SPL (Formula 18a) and total (Formula 17)
# ---------------------------------------------------------------------------
def test_path_level_hand_value() -> None:
    lp = structure_borne_pressure_level_path(69.0, 5.0, 50.0, 12.0)
    expected = 69.0 - 5.0 - 50.0 - 10.0 * math.log10(12.0 / S0) - 10.0 * math.log10(S0 / 4.0)
    assert float(lp) == pytest.approx(expected)


def test_path_level_rejects_bad_area() -> None:
    with pytest.raises(ValueError, match="element_area"):
        structure_borne_pressure_level_path(69.0, 5.0, 50.0, 0.0)


def test_total_is_energetic_sum() -> None:
    paths = np.array([[40.0, 42.0], [44.0, 41.0], [38.0, 39.0]])
    expected = 10.0 * np.log10(np.sum(10.0 ** (0.1 * paths), axis=0))
    assert np.allclose(total_structure_borne_pressure_level(paths), expected)


# ---------------------------------------------------------------------------
# Full prediction / result
# ---------------------------------------------------------------------------
def test_prediction_bundle() -> None:
    bands = np.array([250.0, 500.0, 1000.0])
    lwc = np.array([78.0, 80.0, 77.0])
    dc = np.array([9.0, 10.0, 11.0])
    paths = [
        {"adjustment_term": 5.0,
         "flanking_reduction_index": np.array([50.0, 52.0, 55.0]),
         "element_area": 12.0},
        {"adjustment_term": 6.0,
         "flanking_reduction_index": np.array([52.0, 54.0, 57.0]),
         "element_area": 8.0},
    ]
    res = installed_source_prediction(lwc, dc, paths, frequencies=bands)
    assert isinstance(res, InstalledSourceResult)
    assert res.path_levels.shape == (2, 3)
    # total is the energetic combination of the two paths
    assert np.allclose(
        res.total_level, total_structure_borne_pressure_level(res.path_levels)
    )
    # installed power is L_Ws,c − D_C
    assert np.allclose(res.installed_power_level, lwc - dc)


def test_prediction_requires_paths() -> None:
    with pytest.raises(ValueError, match="paths"):
        installed_source_prediction(80.0, 10.0, [])


def test_prediction_missing_path_key() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        installed_source_prediction(
            80.0, 10.0, [{"adjustment_term": 5.0, "element_area": 12.0}]
        )


def test_prediction_band_count_mismatch() -> None:
    with pytest.raises(ValueError, match="flanking_reduction_index"):
        installed_source_prediction(
            np.array([80.0, 82.0, 78.0]), np.array([9.0, 10.0, 11.0]),
            [{"adjustment_term": 5.0,
              "flanking_reduction_index": np.array([50.0, 52.0]),  # 2 vs 3 bands
              "element_area": 12.0}],
        )


def test_prediction_scalar_source_broadcasts_over_path_bands() -> None:
    # A single-number L_Ws,c / D_C with per-band path data broadcasts to the
    # path band count; installed_power_level then matches the band axis
    # (regression: the fiche table indexed it per band and crashed).
    res = installed_source_prediction(
        80.0, 10.0,
        [{"adjustment_term": 0.0,
          "flanking_reduction_index": np.array([40.0, 50.0, 60.0]),
          "element_area": 10.0}],
    )
    assert res.installed_power_level.shape == (3,)
    np.testing.assert_allclose(res.installed_power_level, 70.0)
    assert res.path_levels.shape == (1, 3)
    # L_n,s,ij = 70 - 0 - Rij - 10 lg(10/10) - 10 lg(10/4)
    expected = 70.0 - np.array([40.0, 50.0, 60.0]) - 10.0 * np.log10(2.5)
    np.testing.assert_allclose(res.path_levels[0], expected)


def test_prediction_scalar_source_band_mismatch_across_paths() -> None:
    with pytest.raises(ValueError, match="adjustment_term"):
        installed_source_prediction(
            80.0, 10.0,
            [{"adjustment_term": np.zeros(2),
              "flanking_reduction_index": np.zeros(3),
              "element_area": 10.0}],
        )


def test_overall_level_numeric() -> None:
    res = installed_source_prediction(
        np.array([80.0, 82.0]), np.array([10.0, 10.0]),
        [{"adjustment_term": 0.0, "flanking_reduction_index": np.array([0.0, 0.0]),
          "element_area": 10.0}],
    )
    expected = 10.0 * math.log10(np.sum(10.0 ** (0.1 * res.total_level)))
    assert res.overall_level == pytest.approx(expected)


def test_total_level_1d_paths_only() -> None:
    # a 1-D (paths-only) array reduces to a scalar-like total
    total = total_structure_borne_pressure_level(np.array([40.0, 44.0, 38.0]))
    expected = 10.0 * math.log10(np.sum(10.0 ** (0.1 * np.array([40.0, 44.0, 38.0]))))
    assert float(total) == pytest.approx(expected)


def test_coupling_elastic_support_hand_value() -> None:
    ys, yi, yk = 1e-4 + 0j, 1e-5 + 0j, 5e-4 + 0j
    expected = 10.0 * math.log10(abs(ys + yi + yk) ** 2 / (abs(ys) * yi.real))
    assert coupling_term(ys, yi, transfer_mobility=yk) == pytest.approx(expected)


def test_plot_returns_axes() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    bands = np.array([250.0, 500.0, 1000.0])
    paths = [
        {"adjustment_term": 5.0,
         "flanking_reduction_index": np.array([50.0, 52.0, 55.0]),
         "element_area": 12.0},
    ]
    res = installed_source_prediction(
        np.array([78.0, 80.0, 77.0]), np.array([9.0, 10.0, 11.0]),
        paths, frequencies=bands,
    )
    assert res.plot() is not None


# ---------------------------------------------------------------------------
# EN 12354-5:2009 Annex I worked examples (the standard's own oracles)
# ---------------------------------------------------------------------------

def _aw_total(levels: np.ndarray) -> float:
    """A-weighted energetic total over the 63-2000 Hz octaves."""
    a = np.array([-26.2, -16.1, -8.6, -3.2, 0.0, 1.2])
    return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * (levels + a)))))


def test_annex_i2_whirlpool_floor_component() -> None:
    """Tables I.6a/I.7: mobility correction and path 11 within +/-0,15 dB."""
    import reference_data as ref

    from phonometry import installed_power_from_reception_plate

    tol = ref.EN12354_5_ANNEX_I_TOL
    inst = installed_power_from_reception_plate(
        ref.EN12354_5_I6A_LWSN_FLOOR, ref.EN12354_5_I6A_Y_FLOOR
    )
    # Y_floor = 1.25e-6 vs Y_rec = 5e-6 -> a flat 10 lg(1/4) = -6,02 dB
    # correction (the example prints -6,0).
    np.testing.assert_allclose(
        inst,
        np.asarray(ref.EN12354_5_I6A_LWSN_FLOOR) + 10.0 * np.log10(0.25),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        inst, ref.EN12354_5_I6A_LWSN_INST_FLOOR, atol=tol
    )
    # Path 11 per Formula (18a); the example's -4 dB corresponds to
    # S_i = S0 = 10 m2 (10 lg(10/4) = 3,98 dB).
    lns_11 = structure_borne_pressure_level_path(
        inst, ref.EN12354_5_I6A_DSA_FLOOR, ref.EN12354_5_I6A_R11, 10.0
    )
    np.testing.assert_allclose(lns_11, ref.EN12354_5_I6A_LNS_11, atol=tol)


def test_annex_i3_cistern_source_conversion() -> None:
    """Table I.8: measured plate power -> installed and characteristic levels."""
    import reference_data as ref

    from phonometry import installed_power_from_reception_plate

    tol = ref.EN12354_5_ANNEX_I_TOL
    for measured, y_i, installed, lwsc in (
        (ref.EN12354_5_I8_WALL_LWS, ref.EN12354_5_I8_Y_WALL,
         ref.EN12354_5_I8_WALL_INSTALLED, ref.EN12354_5_I8_WALL_LWSC),
        (ref.EN12354_5_I8_FLOOR_LWS, ref.EN12354_5_I8_Y_FLOOR,
         ref.EN12354_5_I8_FLOOR_INSTALLED, ref.EN12354_5_I8_FLOOR_LWSC),
    ):
        inst = installed_power_from_reception_plate(
            measured, y_i, plate_mobility=ref.EN12354_5_I8_PLATE_MOBILITY
        )
        np.testing.assert_allclose(inst, installed, atol=tol)
        char = installed_power_from_reception_plate(
            measured, ref.EN12354_5_I8_Y_SOURCE,
            plate_mobility=ref.EN12354_5_I8_PLATE_MOBILITY,
        )
        np.testing.assert_allclose(char, lwsc, atol=tol)


def test_annex_i3_cistern_coupling_terms() -> None:
    """Table I.9: D_C from the force-source limit (Formula 19c)."""
    import reference_data as ref

    dc_wall = float(coupling_term_force_source(
        ref.EN12354_5_I8_Y_SOURCE + 0j, ref.EN12354_5_I8_Y_WALL + 0j
    ))
    dc_floor = float(coupling_term_force_source(
        ref.EN12354_5_I8_Y_SOURCE + 0j, ref.EN12354_5_I8_Y_FLOOR + 0j
    ))
    assert dc_wall == pytest.approx(ref.EN12354_5_I9_DC_WALL, abs=0.05)
    assert dc_floor == pytest.approx(ref.EN12354_5_I9_DC_FLOOR, abs=0.05)


def test_annex_i3_cistern_full_chain_table_i9() -> None:
    """Table I.9 end-to-end: all four paths, the Formula (17) total and the
    29 dB(A) receiving-room value, within the +/-0,15 dB table rounding."""
    import reference_data as ref

    tol = ref.EN12354_5_ANNEX_I_TOL
    lws_inst_wall = installed_structure_borne_power_level(
        ref.EN12354_5_I8_WALL_LWSC, ref.EN12354_5_I9_DC_WALL
    )
    lws_inst_floor = installed_structure_borne_power_level(
        ref.EN12354_5_I8_FLOOR_LWSC, ref.EN12354_5_I9_DC_FLOOR
    )
    paths = {
        "wall>floor": structure_borne_pressure_level_path(
            lws_inst_wall, ref.EN12354_5_I9_DSA_WALL,
            ref.EN12354_5_I9_R_WALL_FLOOR, ref.EN12354_5_I9_S_WALL,
        ),
        "wall>wall": structure_borne_pressure_level_path(
            lws_inst_wall, ref.EN12354_5_I9_DSA_WALL,
            ref.EN12354_5_I9_R_WALL_WALL, ref.EN12354_5_I9_S_WALL,
        ),
        "floor>floor": structure_borne_pressure_level_path(
            lws_inst_floor, ref.EN12354_5_I9_DSA_FLOOR,
            ref.EN12354_5_I9_R_FLOOR_FLOOR, ref.EN12354_5_I9_S_FLOOR,
        ),
        "floor>wall": structure_borne_pressure_level_path(
            lws_inst_floor, ref.EN12354_5_I9_DSA_FLOOR,
            ref.EN12354_5_I9_R_FLOOR_WALL, ref.EN12354_5_I9_S_FLOOR,
        ),
    }
    expected = {
        "wall>floor": ref.EN12354_5_I9_LNS_WALL_FLOOR,
        "wall>wall": ref.EN12354_5_I9_LNS_WALL_WALL,
        "floor>floor": ref.EN12354_5_I9_LNS_FLOOR_FLOOR,
        "floor>wall": ref.EN12354_5_I9_LNS_FLOOR_WALL,
    }
    for name, lns in paths.items():
        np.testing.assert_allclose(lns, expected[name], atol=tol, err_msg=name)
    total = total_structure_borne_pressure_level(
        np.vstack(list(paths.values()))
    )
    np.testing.assert_allclose(total, ref.EN12354_5_I9_LNS_TOTAL, atol=tol)
    assert round(_aw_total(total)) == ref.EN12354_5_I9_LNS_TOTAL_A


def test_coupling_terms_validate_mobilities() -> None:
    """Re{Y_i} <= 0 or Y_s = 0 raise instead of yielding NaN/inf silently."""
    with pytest.raises(ValueError, match="positive real part"):
        coupling_term(1e-4 + 0j, -1e-5 + 0j)
    with pytest.raises(ValueError, match="positive real part"):
        coupling_term(1e-4 + 0j, 1e-5j)  # purely imaginary receiver
    with pytest.raises(ValueError, match="finite and non-zero"):
        coupling_term(0.0, 1e-5 + 0j)
    with pytest.raises(ValueError, match="positive real part"):
        coupling_term_force_source(1e-4 + 0j, 0.0 + 0j)
    with pytest.raises(ValueError, match="finite and non-zero"):
        coupling_term_velocity_source(0.0, 1e4 + 0j)
    with pytest.raises(ValueError, match="receiver_mobility"):
        from phonometry import installed_power_from_reception_plate

        installed_power_from_reception_plate([60.0], 0.0)
