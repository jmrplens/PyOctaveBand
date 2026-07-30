#  Copyright (c) 2026. Jose M. Requena-Plens
"""Tests for the regulatory auditory weighting of marine mammals.

Oracles:

* **NMFS (2018) revision v2.0, Appendix D (printed p. 130)** -- the published
  worked example: "a 1 kHz narrowband sound would result in the following WFAs:
  LF cetaceans: -0.06 dB; MF cetaceans: -29.11 dB; HF cetaceans: -37.55 dB;
  Phocid pinnipeds: -5.90 dB; Otariid pinnipeds: -4.87 dB".
* **NMFS (2018) Table 3 / Table ES3 (printed pp. 18 and 4)**, **NMFS (2024)
  v3.0 Table 5 / Table ES3 / Table A.E-2 (printed pp. 25, 4 and 43)** and
  **Southall et al. (2019) Tables 5, 6 and 7** (Table 7 in the errata-corrected
  form of *Aquatic Mammals* 45(5), printed p. 570) -- parameter and threshold
  tables, checked through the identities the documents state themselves:
  ``C = −max W(f)``, ``Tw = K + C`` and ``injury = TTS + 20 dB`` (non-impulsive)
  or ``+15 dB`` SEL / ``+6 dB`` peak (impulsive, Southall printed pp. 155-156).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from phonometry.underwater.marine_mammal_weighting import (
    WEIGHTING_GUIDANCE,
    AuditoryWeightingResult,
    auditory_weighting,
    exposure_criteria,
    hearing_groups,
    weighted_exposure,
    weighting_parameters,
)

#: NMFS (2018) Appendix D worked example, printed p. 130.
_APPENDIX_D_WFA_1KHZ = {
    "LF": -0.06, "MF": -29.11, "HF": -37.55, "PW": -5.90, "OW": -4.87,
}

#: Generalised hearing ranges as printed in Table ES1 of each NMFS document
#: (2018 v2.0 printed p. 2; 2024 v3.0 printed p. 2). Southall et al. tabulate
#: none, so those rows carry ``None`` and are checked as such.
_PRINTED_HEARING_RANGES = {
    "nmfs-2018": {
        "LF": (7.0, 35e3), "MF": (150.0, 160e3), "HF": (275.0, 160e3),
        "PW": (50.0, 86e3), "OW": (60.0, 39e3),
    },
    "nmfs-2024": {
        "LF": (7.0, 36e3), "HF": (150.0, 160e3), "VHF": (200.0, 165e3),
        "PW": (40.0, 90e3), "OW": (60.0, 68e3), "PA": (42.0, 52e3),
        "OA": (90.0, 40e3),
    },
}

#: Weighted TTS onset thresholds as printed: NMFS 2018 Table 3, NMFS 2024
#: Table 5 and Southall Table 6.
_PRINTED_TTS = {
    "nmfs-2018": {"LF": 179, "MF": 178, "HF": 153, "PW": 181, "OW": 199},
    "nmfs-2024": {"LF": 177, "HF": 181, "VHF": 161, "PW": 175, "OW": 179,
                  "PA": 134, "OA": 157},
    "southall-2019": {"LF": 179, "HF": 178, "VHF": 153, "SI": 186, "PCW": 181,
                      "OCW": 199, "PCA": 134, "OCA": 157},
}


# ---------------------------------------------------------------------------
# The published worked example
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("group", "printed"), sorted(_APPENDIX_D_WFA_1KHZ.items()))
def test_appendix_d_weighting_at_1_khz(group: str, printed: float) -> None:
    """NMFS (2018) Appendix D prints W(1 kHz) for the five groups, to 2 decimals."""
    got = auditory_weighting(1000.0, group, guidance="nmfs-2018").weighting[0]
    assert got == pytest.approx(printed, abs=0.01)


def test_appendix_d_hand_evaluation_of_the_lf_row() -> None:
    """LF at 1 kHz term by term: a = 1, b = 2, f1 = 0.2, f2 = 19, C = 0.13.

    (f/f1)^2a = 25; [1+(f/f1)^2]^a = 26; [1+(f/f2)^2]^b = 1.00554783;
    10·lg(25/26.1442435) = −0.19434; W = 0.13 − 0.19434 = −0.0643 dB.
    """
    ratio = 25.0 / (26.0 * (1.0 + (1.0 / 19.0) ** 2) ** 2)
    expected = 0.13 + 10.0 * np.log10(ratio)
    assert expected == pytest.approx(-0.0644, abs=1e-4)
    assert auditory_weighting(1000.0, "LF", guidance="nmfs-2018").weighting[0] == pytest.approx(
        expected, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Identities the documents state about their own tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guidance", WEIGHTING_GUIDANCE)
def test_c_is_the_negated_peak_of_the_weighting_function(guidance: str) -> None:
    """"C ... determined by ... setting the peak amplitude of the function to zero"."""
    freqs = np.logspace(-3.0, 3.0, 200_001) * 1000.0  # 1 Hz to 1 MHz
    for group in hearing_groups(guidance):
        res = auditory_weighting(freqs, group, guidance=guidance)
        assert float(np.max(res.weighting)) == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("guidance", WEIGHTING_GUIDANCE)
def test_weighted_tts_onset_is_k_plus_c(guidance: str) -> None:
    """"mathematically equivalent to K + C" -- the printed footnote of every table.

    Checked twice over: against the printed thresholds transcribed here, and
    against the ones the criteria table itself serves, so the two copies of the
    same published numbers cannot drift apart.
    """
    for group, printed in _PRINTED_TTS[guidance].items():
        params = weighting_parameters(group, guidance=guidance)
        assert round(params.k_db + params.c_db) == printed
        assert exposure_criteria(group, guidance=guidance).tts_sel == float(printed)


@pytest.mark.parametrize("guidance", WEIGHTING_GUIDANCE)
def test_exposure_function_minimum_is_the_weighted_tts_onset(guidance: str) -> None:
    """E(f) = K + C − W(f) bottoms out at Tw where W peaks."""
    freqs = np.logspace(0.0, 6.0, 100_001)
    for group in hearing_groups(guidance):
        res = auditory_weighting(freqs, group, guidance=guidance)
        assert float(np.min(res.exposure_function)) == pytest.approx(
            res.weighted_tts_onset, abs=0.01
        )


@pytest.mark.parametrize("guidance", WEIGHTING_GUIDANCE)
def test_non_impulsive_injury_is_tts_plus_20_db(guidance: str) -> None:
    """The published non-impulsive injury SEL is 20 dB above the TTS SEL."""
    for group in hearing_groups(guidance):
        crit = exposure_criteria(group, guidance=guidance)
        assert crit.tts_sel is not None
        assert crit.injury_sel is not None
        assert crit.injury_sel - crit.tts_sel == pytest.approx(20.0, abs=1e-9)


def test_southall_impulsive_sel_offset_from_the_non_impulsive_table() -> None:
    """"181 dB re 1 µPa²s for non-impulsive TTS onset −11 dB" (printed p. 155)."""
    cont = exposure_criteria("PCW", guidance="southall-2019")
    imp = exposure_criteria("PCW", guidance="southall-2019", impulsive=True)
    assert cont.tts_sel is not None and imp.tts_sel is not None
    assert imp.tts_sel - cont.tts_sel == pytest.approx(-11.0, abs=1e-9)


@pytest.mark.parametrize("guidance", ["southall-2019", "nmfs-2024"])
def test_impulsive_injury_offsets_are_15_db_sel_and_6_db_peak(guidance: str) -> None:
    """Southall printed p. 155: PTS SEL = TTS SEL + 15 dB, PTS peak = TTS peak + 6 dB.

    NMFS (2024) Table A.E-2 carries the Navy Phase 4 impulsive TTS values that
    Table ES3 does not repeat, and satisfies the same two offsets on every row,
    so both versions that publish an impulsive TTS onset are checked here.
    NMFS 2018 publishes none, which
    ``test_nmfs_2018_publishes_no_impulsive_tts`` pins instead.
    """
    for group in hearing_groups(guidance):
        row = exposure_criteria(group, guidance=guidance, impulsive=True)
        assert row.tts_sel is not None and row.injury_sel is not None
        assert row.injury_sel - row.tts_sel == pytest.approx(15.0, abs=1e-9)
        assert row.tts_peak_spl is not None and row.injury_peak_spl is not None
        assert row.injury_peak_spl - row.tts_peak_spl == pytest.approx(6.0, abs=1e-9)


#: Groups whose impulsive peak-SPL TTS onset the article derives from the
#: audiogram with the +159 dB rule. Printed p. 155 restricts the rule to the
#: groups "in water": "For other species groups in water (LF, SI, PCW, and
#: OCW), 159 dB was added to the value of the hearing threshold at f0". HF and
#: VHF are excluded because their peak thresholds "were directly based on
#: empirical data", and LF because Southall et al. publish no audiogram fit
#: for it. PCA and OCA are in air and are covered by the separate
#: extrapolation test below, not by this one.
_PLUS_159_GROUPS = ("SI", "PCW", "OCW")

#: The two in-air carnivore groups, for which the article uses a nominal 15 dB
#: SEL-to-peak offset instead. Applying the +159 dB rule to them is an
#: extrapolation this project makes, not the article's own method.
_IN_AIR_GROUPS = ("PCA", "OCA")


@pytest.mark.parametrize("group", _PLUS_159_GROUPS)
def test_southall_impulsive_peak_spl_is_threshold_at_f0_plus_159_db(group: str) -> None:
    """Printed p. 155: "159 dB was added to the value of the hearing threshold at f0".

    The article works the rule through for PCW: "Peak SPL TTS onset was
    estimated as 212 dB re 1 µPa (53 dB at f0 + 159 dB)". This test covers only
    the in-water groups the sentence names, which is where the rule is
    validated against published values.
    """
    from phonometry.underwater.marine_mammal_audiograms import (
        BEST_HEARING_FREQUENCY_KHZ,
        group_audiogram,
    )

    f0_hz = BEST_HEARING_FREQUENCY_KHZ[group][0] * 1000.0
    threshold = float(group_audiogram(f0_hz, group).threshold[0])
    row = exposure_criteria(group, guidance="southall-2019", impulsive=True)
    assert row.tts_peak_spl is not None
    # The published values are whole decibels; f0 itself is tabulated to two
    # significant figures, so half a decibel of slack on the derived value.
    assert threshold + 159.0 == pytest.approx(row.tts_peak_spl, abs=0.7)


def test_the_plus_159_rule_rejects_the_printed_in_air_peak_values() -> None:
    """Extrapolating the +159 dB rule to the two in-air groups.

    The article restricts the rule to the in-water groups, so this is an
    extrapolation rather than the article's own method (its in-air rule is a
    nominal 15 dB SEL-to-peak offset, which is what reproduces the *printed*
    138 and 161). Extrapolated, it lands 0.6 dB and 0.4 dB from the
    errata-corrected 155 and 170 and 16 dB and 9 dB from the printed values,
    so it corroborates the errata without depending on it. Note that 154.4
    rounds to 154, not to the corrected 155; see docs/ERRATA.md.
    """
    from phonometry.underwater.marine_mammal_audiograms import (
        BEST_HEARING_FREQUENCY_KHZ,
        group_audiogram,
    )

    for group, printed, corrected in (("PCA", 138.0, 155.0), ("OCA", 161.0, 170.0)):
        f0_hz = BEST_HEARING_FREQUENCY_KHZ[group][0] * 1000.0
        derived = float(group_audiogram(f0_hz, group).threshold[0]) + 159.0
        assert abs(derived - printed) > 8.0
        assert abs(derived - corrected) < 0.7


def test_southall_table_7_errata_values_are_implemented() -> None:
    """The errata (45(5), printed p. 570) corrects four peak SPL values."""
    pca = exposure_criteria("PCA", guidance="southall-2019", impulsive=True)
    oca = exposure_criteria("OCA", guidance="southall-2019", impulsive=True)
    assert (pca.tts_peak_spl, pca.injury_peak_spl) == (155.0, 161.0)  # printed 138 / 144
    assert (oca.tts_peak_spl, oca.injury_peak_spl) == (170.0, 176.0)  # printed 161 / 167


def test_nmfs_2024_otariid_c_uses_the_corrected_1_36() -> None:
    """NMFS's own footnote says the printed 1.37 should be 1.36; C = −max W = 1.3643."""
    params = weighting_parameters("OW", guidance="nmfs-2024")
    assert params.c_db == 1.36
    assert params.c_db_as_printed == 1.37
    freqs = np.logspace(0.0, 6.0, 400_001)
    shape = auditory_weighting(freqs, "OW", guidance="nmfs-2024").weighting - params.c_db
    assert -float(np.max(shape)) == pytest.approx(1.3643, abs=5e-4)


@pytest.mark.parametrize("guidance", sorted(_PRINTED_HEARING_RANGES))
def test_generalised_hearing_ranges_match_table_es1(guidance: str) -> None:
    """Table ES1 of each NMFS document, group by group (24 published numbers)."""
    printed = _PRINTED_HEARING_RANGES[guidance]
    assert set(printed) == set(hearing_groups(guidance))
    for group, expected in printed.items():
        assert weighting_parameters(group, guidance=guidance).hearing_range_hz == expected


def test_southall_rows_carry_no_hearing_range() -> None:
    """The article tabulates none, so the field is ``None`` rather than borrowed."""
    for group in hearing_groups("southall-2019"):
        assert weighting_parameters(group, guidance="southall-2019").hearing_range_hz is None


def test_nmfs_2018_and_southall_agree_on_the_five_shared_groups() -> None:
    """Southall Table 5 and NMFS 2018 Table 3 are numerically identical there."""
    for nmfs, southall in (("LF", "LF"), ("MF", "HF"), ("HF", "VHF"),
                           ("PW", "PCW"), ("OW", "OCW")):
        a = weighting_parameters(nmfs, guidance="nmfs-2018")
        b = weighting_parameters(southall, guidance="southall-2019")
        assert (a.a, a.b, a.f1_khz, a.f2_khz, a.c_db, a.k_db) == (
            b.a, b.b, b.f1_khz, b.f2_khz, b.c_db, b.k_db
        )


def test_asymptotic_slopes_are_20a_and_minus_20b_per_decade() -> None:
    """W falls at 20a dB/decade below f1 and 20b dB/decade above f2 (NMFS 2018 §2.2.3)."""
    params = weighting_parameters("VHF", guidance="nmfs-2024")
    low = auditory_weighting([params.f1_khz * 1e3 / 1000.0, params.f1_khz * 1e3 / 100.0],
                             "VHF", guidance="nmfs-2024").weighting
    high = auditory_weighting([params.f2_khz * 1e3 * 100.0, params.f2_khz * 1e3 * 1000.0],
                              "VHF", guidance="nmfs-2024").weighting
    assert float(low[1] - low[0]) == pytest.approx(20.0 * params.a, abs=0.05)
    assert float(high[1] - high[0]) == pytest.approx(-20.0 * params.b, abs=0.05)


# ---------------------------------------------------------------------------
# Weighted exposure
# ---------------------------------------------------------------------------


def test_weighted_exposure_reduces_the_unweighted_sel() -> None:
    """Weighting can only remove energy: W(f) ≤ 0 everywhere."""
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    band_sel = np.full(freqs.size, 170.0)
    res = weighted_exposure(freqs, band_sel, "VHF", guidance="nmfs-2024")
    assert res.weighted_sel < res.unweighted_sel
    assert np.all(res.weighting <= 0.0)


def test_weighted_exposure_accumulates_10_lg_n() -> None:
    """ISO 18406 Formula 9 for identical events: SEL_cum = SEL + 10·lg(N)."""
    freqs = np.array([500.0, 1000.0, 2000.0])
    band_sel = np.array([170.0, 172.0, 168.0])
    one = weighted_exposure(freqs, band_sel, "LF", n_events=1)
    many = weighted_exposure(freqs, band_sel, "LF", n_events=1000)
    assert many.cumulative_sel - one.cumulative_sel == pytest.approx(30.0, abs=1e-9)


def test_weighted_exposure_flags_exceedance_of_the_published_criterion() -> None:
    """A single band at the criterion frequency reproduces the criterion exactly."""
    params = weighting_parameters("LF", guidance="nmfs-2024")
    crit = exposure_criteria("LF", guidance="nmfs-2024", impulsive=True)
    assert crit.injury_sel is not None
    # Place one band at the peak of W (where the weighting is 0 dB) so the
    # weighted SEL equals the band level exactly.
    peak_hz = 1000.0 * _peak_frequency_khz(params.a, params.b, params.f1_khz, params.f2_khz)
    below = weighted_exposure([peak_hz], [crit.injury_sel - 1.0], "LF", impulsive=True)
    above = weighted_exposure([peak_hz], [crit.injury_sel + 1.0], "LF", impulsive=True)
    assert below.exceeds_injury is False
    assert above.exceeds_injury is True
    assert above.sel_margin == pytest.approx(1.0, abs=0.01)


def _peak_frequency_khz(a: float, b: float, f1: float, f2: float) -> float:
    """Frequency of the maximum of the band-pass shape, by dense search."""
    f = np.logspace(-3.0, 3.0, 200_001)
    shape = (f / f1) ** (2 * a) / ((1 + (f / f1) ** 2) ** a * (1 + (f / f2) ** 2) ** b)
    return float(f[int(np.argmax(shape))])


def test_peak_spl_criterion_is_compared_unweighted() -> None:
    """The dual metric compares the peak SPL flat, not weighted."""
    crit = exposure_criteria("VHF", guidance="nmfs-2024", impulsive=True)
    assert crit.injury_peak_spl == 202.0
    res = weighted_exposure([1000.0], [100.0], "VHF", impulsive=True, peak_spl=210.0)
    assert res.peak_margin == pytest.approx(8.0, abs=1e-9)
    assert res.exceeds_injury is True
    assert res.sel_margin is not None and res.sel_margin < 0.0


def test_an_exposure_exactly_at_the_criterion_counts_as_exceeding_it() -> None:
    """``exceeds_*`` tests ``margin >= 0``: the onset threshold itself is included."""
    crit = exposure_criteria("VHF", guidance="nmfs-2024", impulsive=True)
    assert crit.injury_peak_spl is not None
    res = weighted_exposure([1000.0], [100.0], "VHF", impulsive=True,
                            peak_spl=crit.injury_peak_spl)
    assert res.peak_margin == 0.0
    assert res.exceeds_injury is True


def test_peak_spl_can_trip_tts_alone_and_the_margin_says_so() -> None:
    """A peak between the TTS and injury peak criteria trips only ``exceeds_tts``.

    NMFS 2024 gives VHF cetaceans 196 dB re 1 µPa (TTS) and 202 dB (AUD INJ),
    so a 199 dB peak sits between them while the weighted SEL stays far below
    both SEL criteria.
    """
    res = weighted_exposure([1000.0], [100.0], "VHF", impulsive=True, peak_spl=199.0)
    assert res.tts_peak_margin == pytest.approx(3.0, abs=1e-9)
    assert res.peak_margin == pytest.approx(-3.0, abs=1e-9)
    assert res.tts_margin is not None and res.tts_margin < 0.0
    assert res.exceeds_tts is True
    assert res.exceeds_injury is False


def test_non_impulsive_criteria_have_no_peak_metric() -> None:
    res = weighted_exposure([1000.0], [150.0], "LF", impulsive=False, peak_spl=250.0)
    assert res.criteria.injury_peak_spl is None
    assert res.peak_margin is None


def test_in_air_groups_report_the_20_micropascal_reference() -> None:
    crit = exposure_criteria("PA", guidance="nmfs-2024")
    assert crit.sel_reference == "dB re (20 µPa)²·s"
    assert crit.peak_reference == "dB re 20 µPa"


def test_nmfs_2018_publishes_no_impulsive_tts() -> None:
    crit = exposure_criteria("LF", guidance="nmfs-2018", impulsive=True)
    assert crit.tts_sel is None
    assert crit.injury_sel == 183.0
    assert crit.injury_label == "PTS"


def test_nmfs_2024_uses_the_aud_inj_label() -> None:
    assert exposure_criteria("LF", guidance="nmfs-2024").injury_label == "AUD INJ"


# ---------------------------------------------------------------------------
# Validation and plotting
# ---------------------------------------------------------------------------


def test_default_guidance_is_the_current_one() -> None:
    res = auditory_weighting(1000.0, "LF")
    assert res.guidance == "nmfs-2024"
    assert isinstance(res, AuditoryWeightingResult)
    assert res.parameters.b == 5.0


def test_group_codes_are_not_portable_between_versions() -> None:
    """NMFS 2018 has MF, NMFS 2024 does not; NMFS 2024 has VHF, NMFS 2018 does not."""
    assert "MF" in hearing_groups("nmfs-2018")
    with pytest.raises(ValueError, match="not portable"):
        auditory_weighting(1000.0, "MF", guidance="nmfs-2024")
    with pytest.raises(ValueError, match="not portable"):
        auditory_weighting(1000.0, "VHF", guidance="nmfs-2018")


def test_unknown_guidance_raises() -> None:
    with pytest.raises(ValueError, match="guidance"):
        auditory_weighting(1000.0, "LF", guidance="nmfs-2007")


@pytest.mark.parametrize("frequency", [[], [0.0], [np.nan]])
def test_invalid_frequencies_raise(frequency: list[float]) -> None:
    with pytest.raises(ValueError, match="frequency_hz"):
        auditory_weighting(frequency, "LF")


def test_mismatched_band_spectrum_raises() -> None:
    with pytest.raises(ValueError, match="band_sel"):
        weighted_exposure([100.0, 200.0], [170.0], "LF")


def test_non_finite_band_spectrum_raises() -> None:
    with pytest.raises(ValueError, match="band_sel"):
        weighted_exposure([100.0, 200.0], [170.0, np.nan], "LF")
    with pytest.raises(ValueError, match="band_sel"):
        weighted_exposure([100.0, 200.0], [170.0, np.inf], "LF")


def test_minus_infinity_is_an_empty_band_not_an_error() -> None:
    """A band with no energy contributes nothing and must not be rejected."""
    both = weighted_exposure([100.0, 200.0], [170.0, -np.inf], "LF")
    one = weighted_exposure([100.0], [170.0], "LF")
    assert both.weighted_sel == pytest.approx(one.weighted_sel, abs=1e-12)
    assert both.unweighted_sel == pytest.approx(170.0, abs=1e-12)


def test_the_result_does_not_alias_the_caller_arrays() -> None:
    """The dataclass is frozen, so its arrays must not be the caller's own."""
    freqs = np.array([100.0, 200.0])
    levels = np.array([170.0, 171.0])
    res = weighted_exposure(freqs, levels, "LF")
    assert res.frequencies is not freqs
    assert res.band_sel is not levels
    levels[0] = 0.0
    freqs[0] = 999.0
    assert res.band_sel[0] == 170.0
    assert res.frequencies[0] == 100.0


@pytest.mark.parametrize("n_events", [0, 1.5, -3])
def test_invalid_event_count_raises(n_events: float) -> None:
    with pytest.raises(ValueError, match="n_events"):
        weighted_exposure([1000.0], [170.0], "LF", n_events=n_events)  # type: ignore[arg-type]


def test_non_finite_peak_spl_raises() -> None:
    not_a_number = float("nan")
    with pytest.raises(ValueError, match="peak_spl"):
        weighted_exposure([1000.0], [170.0], "LF", peak_spl=not_a_number)


def test_plot_returns_axes() -> None:
    freqs = np.logspace(1.0, 5.5, 200)
    assert auditory_weighting(freqs, "VHF").plot() is not None
    assert auditory_weighting(freqs, "LF").plot(language="es") is not None
    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])
    res = weighted_exposure(bands, np.full(bands.size, 180.0), "LF", peak_spl=210.0)
    assert res.plot() is not None
    assert res.plot(language="es") is not None
    plt.close("all")
