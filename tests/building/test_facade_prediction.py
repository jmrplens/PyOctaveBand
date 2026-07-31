#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Tests for the EN 12354-3/-4:2000 façade prediction module
(:mod:`phonometry.building.facade_prediction`).

The primary oracles are the two worked examples of the standard: Annex F of
Part 3 (façade airborne insulation) and Annex G of Part 4 (sound radiated to
the outside). Both are transcribed clean-room into ``reference_data`` and
cross-checked here.

The 2000 worked examples carry internal rounding/consistency defects at the
higher octave bands (documented alongside the data): Part 3's printed R' row
disagrees with its own per-element partial indices at 1 k / 2 k, and Part 4's
R' rows above 500 Hz disagree with its Table G.2 inputs. The tests therefore
anchor on the digit-exact low bands, the per-element partial indices, the
single-number ratings and the whole Annex E propagation chain, and separately
assert that the implementation stays self-consistent with the standard's own
per-element values at the noisy bands.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import reference_data as ref

from phonometry import (
    FacadeElement,
    FacadePredictionResult,
    RadiatedPowerResult,
    facade_shape_level_difference,
    facade_sound_reduction,
    outdoor_attenuation,
    outdoor_level,
    radiated_sound_power,
)

# ---------------------------------------------------------------------------
# EN 12354-3 Annex F - façade airborne insulation
# ---------------------------------------------------------------------------


def _annex_f_result() -> FacadePredictionResult:
    elements = [
        FacadeElement(name=name, area=area, r=r)
        for name, area, r in ref.EN12354_3_ANNEX_F_ELEMENTS
    ]
    elements.append(FacadeElement(name="inlet", dn_e=ref.EN12354_3_ANNEX_F_INLET_DNE))
    return facade_sound_reduction(
        elements,
        area=ref.EN12354_3_ANNEX_F_AREA,
        volume=ref.EN12354_3_ANNEX_F_VOLUME,
        frequencies=ref.EN12354_3_ANNEX_F_BANDS,
        bands="octave",
    )


def test_annex_f_low_bands_exact() -> None:
    """R' at 125/250/500 Hz reproduces the standard's printed values exactly."""
    res = _annex_f_result()
    assert np.allclose(res.r_prime[:3], ref.EN12354_3_ANNEX_F_RPRIME_LOW, atol=0.05)


def test_annex_f_element_partial_indices() -> None:
    """Per-element Rp = -10 lg τe matches the printed Table F.1.3 partial indices."""
    res = _annex_f_result()
    # From Table F.1.3 (rounded to 0.1 dB as printed).
    assert np.allclose(res.element_r["wall"], [43.7, 48.7, 54.7, 60.7, 66.7], atol=0.05)
    assert np.allclose(res.element_r["window2"], [27.0, 26.0, 34.0, 40.0, 41.0], atol=0.05)
    assert np.allclose(res.element_r["inlet"], [28.5, 23.5, 25.5, 38.5, 44.5], atol=0.05)


def test_annex_f_single_numbers_exact() -> None:
    """R'tr,s,w, Ctr and D2m,nT,w match the standard's single-number ratings."""
    res = _annex_f_result()
    assert res.r_tr_s_w == ref.EN12354_3_ANNEX_F_RTRS_W
    assert res.c_tr == ref.EN12354_3_ANNEX_F_CTR
    assert res.d_2m_nt_w == ref.EN12354_3_ANNEX_F_D2MNT_W


def test_annex_f_derived_indices() -> None:
    """R45 = R' + 1 (Formula 11) and Rtr,s = R' (Formula 12)."""
    res = _annex_f_result()
    assert np.allclose(res.r_45, res.r_prime + 1.0)
    assert np.allclose(res.r_tr_s, res.r_prime)


def test_annex_f_high_bands_self_consistent() -> None:
    """At 1 k / 2 k the code reproduces the standard's OWN per-element values.

    The standard's printed R' row (35,4 / 37,5) is inconsistent with its printed
    partial indices; summing those indices energetically yields 35,8 / 38,0,
    which is what a faithful implementation must produce.
    """
    res = _annex_f_result()
    rp = res.element_r
    for i, band in ((3, "1 kHz"), (4, "2 kHz")):
        tau = sum(10.0 ** (-rp[name][i] / 10.0) for name in rp)
        expected = -10.0 * np.log10(tau)
        assert res.r_prime[i] == pytest.approx(expected, abs=0.05), band


def test_d2m_nt_formula() -> None:
    """D2m,nT = R' + ΔLfs + 10 lg(V/(6 T0 S)) with T0 = 0,5 s (Formula 13)."""
    res = _annex_f_result()
    term = 10.0 * np.log10(
        ref.EN12354_3_ANNEX_F_VOLUME
        / (6.0 * 0.5 * ref.EN12354_3_ANNEX_F_AREA)
    )
    assert np.allclose(res.d_2m_nt, res.r_prime + term)


def test_delta_l_fs_shifts_d2m_nt() -> None:
    """A non-zero façade-shape term ΔLfs adds directly to D2m,nT (Annex C)."""
    elements = [FacadeElement(name="w", area=10.0, r=[40.0] * 5)]
    base = facade_sound_reduction(elements, area=10.0, volume=50.0)
    shaped = facade_sound_reduction(
        elements, area=10.0, volume=50.0, delta_l_fs=-2.0
    )
    assert np.allclose(shaped.d_2m_nt, base.d_2m_nt - 2.0)


# ---------------------------------------------------------------------------
# EN 12354-4 Annex G - sound radiated to the outside
# ---------------------------------------------------------------------------


def _annex_g_side1_with_door() -> RadiatedPowerResult:
    return radiated_sound_power(
        [
            FacadeElement(
                name="wall",
                area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA - ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_CONCRETE_R,
            ),
            FacadeElement(
                name="door", area=ref.EN12354_4_ANNEX_G_DOOR_AREA,
                r=ref.EN12354_4_ANNEX_G_DOOR_R,
            ),
        ],
        lp_in=ref.EN12354_4_ANNEX_G_LP_IN,
        area=ref.EN12354_4_ANNEX_G_SEGMENT_AREA,
        c_d=ref.EN12354_4_ANNEX_G_CD,
        r_prime_cap=ref.EN12354_4_ANNEX_G_RPRIME_CAP,
        octave_bands=[int(f) for f in ref.EN12354_4_ANNEX_G_BANDS],
    )


def test_annex_g_side1_rprime_low_bands() -> None:
    """Combined wall+door R' at 63/125/250 Hz matches Table G.3 exactly."""
    res = _annex_g_side1_with_door()
    assert np.allclose(
        res.r_prime[:3], ref.EN12354_4_ANNEX_G_SIDE1_RPRIME_LOW, atol=0.05
    )


def test_annex_g_side1_lw_low_bands() -> None:
    """LW = Lp,in + Cd - R' + 10 lg(S/S0) at 63/125 Hz matches Table G.3."""
    res = _annex_g_side1_with_door()
    assert np.allclose(res.l_w[:2], ref.EN12354_4_ANNEX_G_SIDE1_LW_LOW, atol=0.1)


def test_annex_g_lw_relation() -> None:
    """LW obeys Formula (2) against the resulting R' at every band."""
    res = _annex_g_side1_with_door()
    s = ref.EN12354_4_ANNEX_G_SEGMENT_AREA
    expected = (
        np.asarray(ref.EN12354_4_ANNEX_G_LP_IN)
        + ref.EN12354_4_ANNEX_G_CD
        - res.r_prime
        + 10.0 * np.log10(s / 1.0)
    )
    assert np.allclose(res.l_w, expected)


def test_annex_g_rprime_cap() -> None:
    """R' is limited to 40 dB (Table G.3 footnote) for a high-performance wall."""
    res = radiated_sound_power(
        [FacadeElement(name="wall", area=200.0, r=ref.EN12354_4_ANNEX_G_CONCRETE_R)],
        lp_in=ref.EN12354_4_ANNEX_G_LP_IN,
        area=200.0,
        c_d=ref.EN12354_4_ANNEX_G_CD,
        r_prime_cap=40.0,
        octave_bands=[int(f) for f in ref.EN12354_4_ANNEX_G_BANDS],
    )
    # A single element's R' equals its own R, then the cap clips it.
    assert res.r_prime.max() <= 40.0 + 1e-9
    # Below the cap, R' == input R exactly (bands 63-500 Hz).
    assert np.allclose(res.r_prime[:4], ref.EN12354_4_ANNEX_G_CONCRETE_R[:4])


def test_rprime_cap_off_by_default() -> None:
    """The 40 dB cap is an Annex G example footnote, not Formula (2)/(3):
    by default R' is the bare energy sum, uncapped."""
    res = radiated_sound_power(
        [FacadeElement(name="wall", area=100.0, r=[60.0])],
        lp_in=90.0, area=100.0,
    )
    assert res.r_prime[0] == pytest.approx(60.0)
    # LW then follows the bare Formula (2): 90 - 6 - 60 + 20 = 44 dB.
    assert res.l_w[0] == pytest.approx(44.0)


@pytest.mark.parametrize("shape,height,alpha,expected", ref.EN12354_3_ANNEX_C_DLFS)
def test_facade_shape_level_difference_table(
    shape: str, height: float, alpha: float, expected: float
) -> None:
    """ΔLfs lookup reproduces the Annex C Figure C.2 cells."""
    assert facade_shape_level_difference(
        shape, line_of_sight=height, absorption=alpha
    ) == pytest.approx(expected)


def test_facade_shape_level_difference_interpolates_alpha() -> None:
    # Balcony 6, 1,5-2,5 m: -1 / 1 / 3 over αw 0,3 / 0,6 / 0,9.
    mid = facade_shape_level_difference(
        "balcony_6", line_of_sight=2.0, absorption=0.45
    )
    assert mid == pytest.approx(0.0)  # halfway between -1 and 1
    # αw outside the tabulated range clamps to the edge column.
    assert facade_shape_level_difference(
        "balcony_6", line_of_sight=2.0, absorption=1.0
    ) == pytest.approx(3.0)
    assert facade_shape_level_difference(
        "balcony_6", line_of_sight=2.0, absorption=0.1
    ) == pytest.approx(-1.0)


def test_facade_shape_level_difference_does_not_apply() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        facade_shape_level_difference("gallery_2", line_of_sight=2.0)
    with pytest.raises(ValueError, match="does not apply"):
        facade_shape_level_difference("gallery_5", line_of_sight=1.0)
    with pytest.raises(ValueError, match="Unknown facade shape"):
        facade_shape_level_difference("igloo")
    with pytest.raises(ValueError, match="absorption"):
        facade_shape_level_difference("balcony_6", absorption=1.5)
    with pytest.raises(ValueError, match="line_of_sight"):
        facade_shape_level_difference("balcony_6", line_of_sight=-1.0)


def test_opening_insertion_loss() -> None:
    """A bare opening (insertion_loss = 0) transmits its full area fraction.

    This exercises the module's practical single-pool opening model (an opening
    is an element with R = insertion loss), which is a deliberate extension of
    the standard's separate segment-of-openings Formula (4) -- so the oracle
    here is the transmission physics (1 % open area -> R' ~ 20 dB), not a
    printed Annex G value.
    """
    seg = radiated_sound_power(
        [
            FacadeElement(name="wall", area=99.0, r=[60.0]),
            FacadeElement(name="hole", area=1.0, insertion_loss=0.0),
        ],
        lp_in=90.0, area=100.0, r_prime_cap=None,
    )
    # 1 % open area with R=0 dominates: τ ~ 0.01 -> R' ~ 20 dB.
    assert seg.r_prime[0] == pytest.approx(20.0, abs=0.2)


def test_a_weighted_single_number() -> None:
    """Passing known octave centres yields an A-weighted LW; unknown -> None."""
    with_bands = _annex_g_side1_with_door()
    assert with_bands.l_w_dba is not None
    without = radiated_sound_power(
        [FacadeElement(name="w", area=200.0, r=ref.EN12354_4_ANNEX_G_CONCRETE_R)],
        lp_in=ref.EN12354_4_ANNEX_G_LP_IN, area=200.0,
    )
    assert without.l_w_dba is None


# ---------------------------------------------------------------------------
# EN 12354-4 Annex E - simplified outdoor propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width,height,distance,expected", ref.EN12354_4_ANNEX_G_ATTENUATION
)
def test_annex_e_attenuation(
    width: float, height: float, distance: float, expected: float
) -> None:
    """A'tot of a finite radiating side reproduces Table G.9 exactly."""
    assert outdoor_attenuation(width, height, distance) == pytest.approx(
        expected, abs=0.05
    )


def test_annex_e_exterior_level() -> None:
    """Lp = LW - A'tot for all four Table G.9 reception cells."""
    lp1 = outdoor_level(
        ref.EN12354_4_ANNEX_G_SIDE1_LWA, outdoor_attenuation(60.0, 10.0, 5.0)
    )
    assert lp1 == pytest.approx(ref.EN12354_4_ANNEX_G_LP_SIDE1_D5, abs=0.05)
    lp4 = outdoor_level(
        ref.EN12354_4_ANNEX_G_SIDE4_LWA, outdoor_attenuation(100.0, 10.0, 25.0)
    )
    assert lp4 == pytest.approx(ref.EN12354_4_ANNEX_G_LP_SIDE4_D25, abs=0.05)
    # Remaining Table G.9 cells: side 1 at 25 m and side 4 at 5 m.
    lp1_far = outdoor_level(
        ref.EN12354_4_ANNEX_G_SIDE1_LWA, outdoor_attenuation(60.0, 10.0, 25.0)
    )
    assert lp1_far == pytest.approx(ref.EN12354_4_ANNEX_G_LP_SIDE1_D25, abs=0.05)
    lp4_near = outdoor_level(
        ref.EN12354_4_ANNEX_G_SIDE4_LWA, outdoor_attenuation(100.0, 10.0, 5.0)
    )
    assert lp4_near == pytest.approx(ref.EN12354_4_ANNEX_G_LP_SIDE4_D5, abs=0.05)


def test_outdoor_level_energy_sum() -> None:
    """Two equal sides energetically sum to +3 dB over one side."""
    single = outdoor_level(60.0, 0.0)
    pair = outdoor_level([60.0, 60.0], [0.0, 0.0])
    assert pair == pytest.approx(single + 3.0103, abs=1e-3)


def test_outdoor_level_scalar_broadcast() -> None:
    """A scalar attenuation broadcasts against an array of side powers."""
    broadcast = outdoor_level([60.0, 60.0], 3.0)
    explicit = outdoor_level([60.0, 60.0], [3.0, 3.0])
    assert broadcast == pytest.approx(explicit)


def test_outdoor_level_incompatible_shapes_raise() -> None:
    with pytest.raises(ValueError, match="broadcast-compatible"):
        outdoor_level([60.0, 60.0, 60.0], [0.0, 0.0])


def test_far_field_point_source() -> None:
    """Beyond the largest side dimension the point-source form (E.2b) applies."""
    # d=200 > max(10,10): A'tot = -10 lg(S0/(π d²)) = 10 lg(π d²).
    a = outdoor_attenuation(10.0, 10.0, 200.0)
    assert a == pytest.approx(10.0 * np.log10(np.pi * 200.0**2), abs=1e-6)


# ---------------------------------------------------------------------------
# Input validation & broadcasting
# ---------------------------------------------------------------------------


def test_element_requires_exactly_one_quantity() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FacadeElement(name="x", area=1.0).tau(10.0, 1)
    with pytest.raises(ValueError, match="exactly one"):
        FacadeElement(name="x", area=1.0, r=[40.0], dn_e=[30.0]).tau(10.0, 1)


def test_area_element_requires_positive_area() -> None:
    with pytest.raises(ValueError, match="area"):
        FacadeElement(name="x", r=[40.0]).tau(10.0, 1)


def test_inconsistent_band_counts_raise() -> None:
    with pytest.raises(ValueError, match="band count"):
        facade_sound_reduction(
            [
                FacadeElement(name="a", area=5.0, r=[40.0, 41.0, 42.0]),
                FacadeElement(name="b", area=5.0, r=[30.0, 31.0]),
            ],
            area=10.0, volume=50.0,
        )


def test_scalar_broadcasts_to_band_count() -> None:
    """A scalar element R broadcasts to the common band count."""
    res = facade_sound_reduction(
        [
            FacadeElement(name="a", area=5.0, r=40.0),
            FacadeElement(name="b", area=5.0, r=[30.0, 31.0, 32.0]),
        ],
        area=10.0, volume=50.0,
    )
    assert res.r_prime.size == 3


def test_non_finite_input_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        FacadeElement(name="x", area=1.0, r=[np.nan]).tau(10.0, 1)


def test_duplicate_element_names_raise() -> None:
    """Elements are keyed by name in the result, so names must be unique."""
    with pytest.raises(ValueError, match="unique"):
        facade_sound_reduction(
            [
                FacadeElement(name="glass", area=5.0, r=[40.0]),
                FacadeElement(name="glass", area=5.0, r=[30.0]),
            ],
            area=10.0, volume=50.0,
        )


def test_no_element_silently_dropped_when_names_unique() -> None:
    """Every element contributes to R' even if two share the same R values."""
    two = facade_sound_reduction(
        [
            FacadeElement(name="a", area=5.0, r=[30.0]),
            FacadeElement(name="b", area=5.0, r=[30.0]),
        ],
        area=10.0, volume=50.0,
    )
    one = facade_sound_reduction(
        [FacadeElement(name="a", area=5.0, r=[30.0])], area=10.0, volume=50.0
    )
    # Two 5 m² halves at R=30 over S=10 => R'=30; one 5 m² over S=10 => R'=33.
    assert two.r_prime[0] == pytest.approx(30.0, abs=1e-9)
    assert one.r_prime[0] == pytest.approx(33.0103, abs=1e-3)


def test_frequencies_length_must_match_bands() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        facade_sound_reduction(
            [FacadeElement(name="a", area=5.0, r=[40.0, 41.0, 42.0])],
            area=10.0, volume=50.0, frequencies=[125.0, 250.0],
        )


def test_octave_bands_length_must_match() -> None:
    with pytest.raises(ValueError, match="octave_bands"):
        radiated_sound_power(
            [FacadeElement(name="w", area=10.0, r=[40.0, 41.0])],
            lp_in=[70.0, 71.0], area=10.0, octave_bands=[125],
        )


def test_tau_rejects_non_positive_total_area() -> None:
    with pytest.raises(ValueError, match="total_area"):
        FacadeElement(name="x", area=1.0, r=[40.0]).tau(0.0, 1)


def test_positive_area_and_volume_required() -> None:
    with pytest.raises(ValueError, match="volume"):
        facade_sound_reduction(
            [FacadeElement(name="a", area=5.0, r=[40.0])], area=10.0, volume=0.0
        )
    with pytest.raises(ValueError, match="area"):
        facade_sound_reduction(
            [FacadeElement(name="a", area=5.0, r=[40.0])], area=0.0, volume=50.0
        )


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------


def test_facade_prediction_plot() -> None:
    ax = _annex_f_result().plot()
    assert ax.get_ylabel() != ""
    assert len(ax.lines) >= 2


def test_radiated_power_plot() -> None:
    ax = _annex_g_side1_with_door().plot()
    assert ax.get_ylabel() != ""
    assert len(ax.patches) == len(ref.EN12354_4_ANNEX_G_BANDS)


# ---------------------------------------------------------------------------
# Element compositing — independent literature oracles
# ---------------------------------------------------------------------------

# The area-weighted transmission-factor sum (Formula 15) is unit-free in the
# area ratio, so Long's foot-pound examples apply as printed. The receiving
# volume is not part of R', so a nominal volume is passed.


def test_composite_r_long_window_in_wall() -> None:
    # Long, Architectural Acoustics 2e (Academic Press, 2014), Ch. 10 p. 385:
    # a 12 ft2 window of TL 25 dB in a 160 ft2 wall of TL 45 dB gives
    # R = 10 lg[160 / (12*0.0034 + 148*0.00003)] = 35.5 dB. Long rounds the
    # transmission factors to two significant figures (0.0034 for 10^-2.5 =
    # 0.00316), which moves the printed result by up to 0.3 dB; the exact
    # compositing gives 35.74 dB.
    res = facade_sound_reduction(
        [
            FacadeElement("wall", area=148.0, r=45.0),
            FacadeElement("window", area=12.0, r=25.0),
        ],
        area=160.0,
        volume=50.0,
    )
    assert res.r_prime[0] == pytest.approx(35.5, abs=0.3)


def test_composite_r_long_slot_under_door() -> None:
    # Long (2014), Ch. 10 p. 385: a 20 ft2 door of TL 30 dB with a 0.5 in
    # slot (0.125 ft2, TL 0 dB) beneath it composites to
    # R = 10 lg[20.125 / (20*0.001 + 0.125*1)] = 21.4 dB (printed to 0.1 dB).
    res = facade_sound_reduction(
        [
            FacadeElement("door", area=20.0, r=30.0),
            FacadeElement("slot", area=0.125, r=0.0),
        ],
        area=20.125,
        volume=50.0,
    )
    assert res.r_prime[0] == pytest.approx(21.4, abs=0.05)


def test_composite_r_manual_es_facade() -> None:
    # Aviles Lopez & Perera Martin, Manual de acustica ambiental y
    # arquitectonica (Paraninfo), Ejemplo 7.5 (p. 410): 8 m2 facade with a
    # 2 m2 window; blind part RA = 40 dBA, window RA = 26 dBA ->
    # Rg = 10 lg[8 / (6*10^-4.0 + 2*10^-2.6)] = 31.5 dBA (printed to 0.1 dB).
    res = facade_sound_reduction(
        [
            FacadeElement("blind part", area=6.0, r=40.0),
            FacadeElement("window", area=2.0, r=26.0),
        ],
        area=8.0,
        volume=50.0,
    )
    assert res.r_prime[0] == pytest.approx(31.5, abs=0.05)
