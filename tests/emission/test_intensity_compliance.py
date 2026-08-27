#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 61043:1993 sound-intensity instrument class verification.

Oracles (independent of the implementation):

* **Table 2** of EN 61043:1994 (standard page 14), transcribed digit for digit
  in ``tests/reference_data/`` as :data:`reference_data.IEC61043_TABLE2` and
  cross-checked against the same table reproduced in Fahy, *Sound Intensity*
  2nd ed., Table 6.1 (printed page 136). Every band, every device kind and
  both classes are asserted, plus the structural identity that ties the three
  column groups together: the instrument minimum is the probe and processor
  residual intensities added as powers.
* **Note 1** of that table: for a microphone separation ``x`` in millimetres,
  add ``10 lg(x/25)`` dB to every figure. Checked at several separations, with
  the exact closed-form values (a doubling is exactly ``10 lg 2``, and 250 mm
  is exactly +10 dB).
* **Clause 8**: a class 1 instrument requires a class 1 probe *and* a class 1
  processor; every other combination of class 1/2 components is class 2.
* **Clauses 6.1 and 12.4** for the frequency range a class attests: 45 Hz to
  7,1 kHz in one-third octaves, or 45 Hz to 5,6 kHz in octaves for an
  octave-band processor, while a probe's index is determined at one-third
  octave intervals over the whole 50 Hz to 6,3 kHz range.
* Fahy section 6.8 (printed page 135) for the phase-mismatch conversion:
  ``delta_pI0 = 20 dB`` corresponds to a mismatch of one hundredth of ``kd``,
  about 0,26 degrees at 1 kHz over a 25 mm separation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import reference_data as ref

from phonometry import emission

#: Table 2 column index of each device kind's (class 1, class 2) pair.
_COLUMNS = {"probe": (1, 2), "processor": (3, 4), "instrument": (5, 6)}

_BANDS = [row[0] for row in ref.IEC61043_TABLE2]


# ---------------------------------------------------------------------------
# Table 2, digit for digit
# ---------------------------------------------------------------------------


def _table_column(device: str, cls: int) -> list[float]:
    """The Table 2 column of a device kind and class, straight from the oracle."""
    col = _COLUMNS[device][cls - 1]
    return [row[col] for row in ref.IEC61043_TABLE2]


@pytest.mark.parametrize("device", ["probe", "processor", "instrument"])
def test_table2_transcription_digit_for_digit(device: str) -> None:
    """Every band of every Table 2 column group reproduces the standard."""
    col1, col2 = _COLUMNS[device]
    freqs, class1, class2 = emission.residual_index_limits(device)

    assert list(freqs) == pytest.approx(_BANDS)
    for i, row in enumerate(ref.IEC61043_TABLE2):
        assert class1[i] == pytest.approx(row[col1]), f"{device} class 1 @ {row[0]} Hz"
        assert class2[i] == pytest.approx(row[col2]), f"{device} class 2 @ {row[0]} Hz"


def test_table2_class1_never_looser_than_class2() -> None:
    """Class 1 demands at least as much residual index as class 2 everywhere."""
    for device in _COLUMNS:
        _, class1, class2 = emission.residual_index_limits(device)
        assert np.all(class1 >= class2)


@pytest.mark.parametrize("cls", [1, 2])
def test_table2_instrument_column_is_the_probe_plus_processor_residual(
    cls: int,
) -> None:
    """The instrument minima are the probe and processor residuals added.

    An independent structural check of the transcription. The residual
    intensities of the probe and of the processor come from unrelated phase
    errors, so they add as powers, and since ``delta_pI0`` is a level
    difference from the same pressure that is::

        10**(-d_instrument/10) = 10**(-d_probe/10) + 10**(-d_processor/10)

    Every one of the 22 rows of Table 2 satisfies this to within the rounding
    of the printed figures, in both classes, including the lone half-decibel
    entry (400 Hz, instrument class 2, 14,5 dB, whose exact value is 14,54).
    A single mistyped digit anywhere in the table would break the identity in
    that row.
    """
    probe = _table_column("probe", cls)
    processor = _table_column("processor", cls)
    instrument = _table_column("instrument", cls)
    for i, row in enumerate(ref.IEC61043_TABLE2):
        combined = -10.0 * math.log10(
            10.0 ** (-probe[i] / 10.0) + 10.0 ** (-processor[i] / 10.0)
        )
        assert combined == pytest.approx(instrument[i], abs=0.25), (
            f"class {cls} @ {row[0]} Hz"
        )


def test_table2_accepts_exact_base_ten_band_centres() -> None:
    """The exact IEC 61260 centres behind the nominal labels select the same rows."""
    exact = [1000.0 * 10.0 ** (n / 10.0) for n in (-13, -10, 0, 8)]  # 50/100/1k/6.3k
    freqs, class1, _ = emission.residual_index_limits("instrument", frequencies=exact)
    assert list(freqs) == [50.0, 100.0, 1000.0, 6300.0]
    assert list(class1) == pytest.approx([12.0, 15.0, 19.0, 19.0])


def test_frequency_outside_table_is_rejected() -> None:
    """A frequency that is not a tabulated band raises, it is not snapped."""
    with pytest.raises(ValueError, match="Table 2 band"):
        emission.residual_index_limits("instrument", frequencies=[700.0])
    with pytest.raises(ValueError, match="Table 2 band"):
        emission.residual_index_limits("instrument", frequencies=[8000.0])


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="probe"):
        emission.residual_index_limits("analyser")


# ---------------------------------------------------------------------------
# Table 2 Note 1: the +10 lg(x/25) separation rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spacing_mm", "offset_db"),
    [
        (25.0, 0.0),
        (50.0, 10.0 * math.log10(2.0)),
        (12.5, -10.0 * math.log10(2.0)),
        (250.0, 10.0),
        (2.5, -10.0),
        (6.0, 10.0 * math.log10(6.0 / 25.0)),
    ],
)
def test_note1_separation_rule(spacing_mm: float, offset_db: float) -> None:
    """Every tabulated figure shifts by exactly 10 lg(x/25) dB."""
    for device in _COLUMNS:
        _, base1, base2 = emission.residual_index_limits(device)
        _, shifted1, shifted2 = emission.residual_index_limits(
            device, spacing=spacing_mm / 1000.0
        )
        assert shifted1 == pytest.approx(base1 + offset_db)
        assert shifted2 == pytest.approx(base2 + offset_db)


def test_note1_rule_matches_the_table_values_at_50_mm() -> None:
    """A 50 mm spacer earns 3,0103 dB over every printed 25 mm figure."""
    _, class1, _ = emission.residual_index_limits("instrument", spacing=0.050)
    expected = [row[5] + 10.0 * math.log10(2.0) for row in ref.IEC61043_TABLE2]
    assert list(class1) == pytest.approx(expected)


def test_non_positive_spacing_is_rejected() -> None:
    with pytest.raises(ValueError, match="spacing"):
        emission.residual_index_limits("probe", spacing=0.0)


# ---------------------------------------------------------------------------
# Clause 8: instrument class from separate components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("probe", "processor", "expected"),
    [(1, 1, 1), (1, 2, 2), (2, 1, 2), (2, 2, 2)],
)
def test_instrument_class_from_components(
    probe: int, processor: int, expected: int
) -> None:
    """IEC 61043 clause 8: only 1 + 1 makes a class 1 instrument."""
    assert emission.instrument_class_from_components(probe, processor) == expected


def test_instrument_class_rejects_class_2x() -> None:
    with pytest.raises(ValueError, match="2X"):
        emission.instrument_class_from_components(1, 3)


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def test_instrument_exactly_on_the_class1_limit_passes() -> None:
    """A band exactly on the minimum meets the class (the limit is inclusive)."""
    measured = _table_column("instrument", 1)
    verdict = emission.verify_intensity_class(measured, _BANDS)
    assert verdict["overall_class"] == 1
    assert verdict["range_limited"] is False
    assert all(
        band["margin_class1_db"] == pytest.approx(0.0) for band in verdict["bands"]
    )


def test_instrument_just_below_the_class1_limit_falls_to_class2() -> None:
    """One band 0,1 dB short of class 1 drops the whole verdict to class 2."""
    measured = _table_column("instrument", 1)
    measured[7] -= 0.1  # 250 Hz
    verdict = emission.verify_intensity_class(measured, _BANDS)
    assert verdict["overall_class"] == 2
    classes = [band["class"] for band in verdict["bands"]]
    assert classes[7] == 2
    assert set(classes) == {1, 2}


def test_instrument_below_both_limits_meets_no_class() -> None:
    """A band under the class 2 minimum leaves the chain unclassified."""
    measured = _table_column("instrument", 1)
    measured[0] = _table_column("instrument", 2)[0] - 0.5  # 50 Hz under class 2
    verdict = emission.verify_intensity_class(measured, _BANDS)
    assert verdict["overall_class"] is None
    assert verdict["bands"][0]["class"] is None


def test_probe_and_processor_are_judged_against_their_own_columns() -> None:
    """The same measured spectrum classifies differently per device kind."""
    measured = _table_column("processor", 1)
    assert (
        emission.verify_intensity_class(measured, _BANDS, device="processor")[
            "overall_class"
        ]
        == 1
    )
    # The processor class 1 column is 6-7 dB above the probe class 1 column, so
    # the same spectrum clears the probe requirement with room to spare.
    probe = emission.verify_intensity_class(measured, _BANDS, device="probe")
    assert probe["overall_class"] == 1
    assert min(b["margin_class1_db"] for b in probe["bands"]) > 5.0


def test_spacing_shifts_the_verdict() -> None:
    """A chain that fails at 25 mm can pass with a wider spacer, and back."""
    measured = [row[5] - 2.0 for row in ref.IEC61043_TABLE2]  # 2 dB short
    assert emission.verify_intensity_class(measured, _BANDS)["overall_class"] == 2
    # A 12,5 mm spacer relaxes every requirement by 10 lg(0,5) = -3,01 dB.
    relaxed = emission.verify_intensity_class(measured, _BANDS, spacing=0.0125)
    assert relaxed["overall_class"] == 1
    assert relaxed["spacing_offset_db"] == pytest.approx(-10.0 * math.log10(2.0))


_OCTAVES = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


def test_octave_band_subset_attests_a_full_range_class_2() -> None:
    """Clause 6.1 offers the 7 octave bands 63 Hz to 4 kHz to class 2."""
    _, _, class2 = emission.residual_index_limits("processor", frequencies=_OCTAVES)
    verdict = emission.verify_intensity_class(
        list(class2), _OCTAVES, device="processor"
    )
    assert verdict["overall_class"] == 2
    assert verdict["range_limited"] is False
    # The instrument built on that processor inherits the same alternative.
    _, _, inst2 = emission.residual_index_limits("instrument", frequencies=_OCTAVES)
    instrument = emission.verify_intensity_class(list(inst2), _OCTAVES)
    assert instrument["overall_class"] == 2
    assert instrument["range_limited"] is False


def test_octave_band_class_1_verdict_is_still_range_limited() -> None:
    """Clause 6.1 gives the one-third-octave range to class 1.

    The octave range is the *class 2* alternative, so clearing the class 1
    minima at seven octave centres attests class 1 over those seven bands and
    not over the 45 Hz to 7,1 kHz one-third-octave range the class requires.
    """
    _, class1, _ = emission.residual_index_limits("processor", frequencies=_OCTAVES)
    verdict = emission.verify_intensity_class(
        list(class1), _OCTAVES, device="processor"
    )
    assert verdict["overall_class"] == 1
    assert verdict["range_limited"] is True
    # The same chain measured across all 22 one-third-octave bands does attest
    # the full class 1 range.
    _, full1, _ = emission.residual_index_limits("processor")
    full = emission.verify_intensity_class(list(full1), _BANDS, device="processor")
    assert full["overall_class"] == 1
    assert full["range_limited"] is False


def test_octave_band_subset_is_range_limited_for_a_probe() -> None:
    """Clause 12.4 determines a probe's index in one-third octaves, 50 Hz-6,3 kHz.

    The octave-band alternative of clause 6.1 is a *processor* option, so a
    probe verified at seven octave centres only attests those seven bands.
    """
    _, _, class2 = emission.residual_index_limits("probe", frequencies=_OCTAVES)
    verdict = emission.verify_intensity_class(list(class2), _OCTAVES, device="probe")
    assert verdict["range_limited"] is True
    assert verdict["overall_class"] == 2
    # All 22 one-third-octave bands do attest the probe's full range.
    _, full, _ = emission.residual_index_limits("probe")
    assert (
        emission.verify_intensity_class(list(full), _BANDS, device="probe")[
            "range_limited"
        ]
        is False
    )


def test_partial_band_set_is_range_limited() -> None:
    """A verdict over a handful of bands attests only those bands."""
    bands = [500.0, 1000.0, 2000.0]
    _, class1, _ = emission.residual_index_limits("instrument", frequencies=bands)
    verdict = emission.verify_intensity_class(list(class1), bands)
    assert verdict["range_limited"] is True


def test_mismatched_lengths_and_repeats_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"verify_intensity_class: 'residual_index' .* must each carry "
            r"one value per band"
        ),
    ):
        emission.verify_intensity_class([20.0, 20.0], [1000.0])
    with pytest.raises(ValueError, match="repeats"):
        emission.verify_intensity_class([20.0, 20.0], [1000.0, 1000.0])
    with pytest.raises(ValueError, match="At least one"):
        emission.verify_intensity_class([], [])


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


def test_result_carries_the_masks_and_the_binding_margin() -> None:
    """The result mirrors verify_intensity_class and exposes the margins."""
    measured = [value + 1.5 for value in _table_column("instrument", 1)]
    measured[4] -= 1.0  # 125 Hz becomes the binding band (+0,5 dB)
    result = emission.intensity_class_compliance(measured, _BANDS)

    assert result.overall_class == 1
    assert result.binding_margin() == pytest.approx(0.5)
    assert result.binding_margin(1) == pytest.approx(0.5)
    assert result.failing_bands() == []
    assert list(result.limit_class1) == pytest.approx(_table_column("instrument", 1))
    assert list(result.limit_class2) == pytest.approx(_table_column("instrument", 2))
    assert result.reference_class() == 1
    assert result.device == "instrument"
    assert result.spacing == pytest.approx(0.025)


def test_result_reference_class_and_failing_bands_when_non_compliant() -> None:
    """A chain meeting no class reports class 2 as reference and lists failures."""
    measured = [value - 1.0 for value in _table_column("instrument", 2)]
    result = emission.intensity_class_compliance(measured, _BANDS)

    assert result.overall_class is None
    assert result.reference_class() == 2
    assert result.failing_bands() == pytest.approx(_BANDS)
    assert result.binding_margin() == pytest.approx(-1.0)


def test_result_rejects_an_unknown_class() -> None:
    result = emission.intensity_class_compliance(
        _table_column("probe", 1), _BANDS, device="probe"
    )
    with pytest.raises(ValueError, match="must be 1 or 2"):
        result.binding_margin(0)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        result.failing_bands(3)


# ---------------------------------------------------------------------------
# delta_pI0 <-> phase mismatch (Fahy 6.8 / eqn (7.16))
# ---------------------------------------------------------------------------


def test_fahy_worked_check_026_degrees() -> None:
    """20 dB at 1 kHz over 25 mm is a mismatch of about 0,26 degrees."""
    phi = emission.phase_mismatch_from_residual_index(
        ref.IEC61043_PHASE_INDEX_DB,
        ref.IEC61043_PHASE_FREQUENCY_HZ,
        ref.IEC61043_PHASE_SPACING_M,
    )
    assert float(phi) == pytest.approx(ref.IEC61043_PHASE_MISMATCH_DEG, abs=0.005)


def test_20_db_is_one_hundredth_of_kd() -> None:
    """Fahy 6.8: delta_pI0 = 20 dB is a mismatch of kd / 100, at any frequency."""
    freqs = np.array([50.0, 250.0, 1000.0, 6300.0])
    spacing = 0.012
    kd_deg = 360.0 * freqs * spacing / 343.0
    phi = emission.phase_mismatch_from_residual_index(20.0, freqs, spacing)
    assert phi == pytest.approx(kd_deg / 100.0)


def test_phase_conversion_round_trip() -> None:
    """The two conversions invert each other exactly over a wide range."""
    freqs = np.array([50.0, 125.0, 500.0, 2000.0, 6300.0])
    index = np.array([12.0, 16.0, 19.0, 19.0, 19.0])
    phi = emission.phase_mismatch_from_residual_index(index, freqs, 0.050)
    back = emission.residual_index_from_phase_mismatch(phi, freqs, 0.050)
    assert back == pytest.approx(index)


def test_index_rises_10_db_per_decade_for_a_fixed_mismatch() -> None:
    """kd grows with f while a constant mismatch does not: +10 dB per decade."""
    index_100 = float(emission.residual_index_from_phase_mismatch(0.5, 100.0, 0.025))
    index_1000 = float(emission.residual_index_from_phase_mismatch(0.5, 1000.0, 0.025))
    assert index_1000 - index_100 == pytest.approx(10.0)


def test_index_follows_note1_when_the_spacer_changes() -> None:
    """Doubling the spacer at a fixed mismatch adds exactly 10 lg 2 dB."""
    narrow = float(emission.residual_index_from_phase_mismatch(0.3, 500.0, 0.0125))
    wide = float(emission.residual_index_from_phase_mismatch(0.3, 500.0, 0.025))
    assert wide - narrow == pytest.approx(10.0 * math.log10(2.0))


def test_phase_conversion_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="phase_mismatch"):
        emission.residual_index_from_phase_mismatch(0.0, 1000.0, 0.025)
    with pytest.raises(ValueError, match="spacing"):
        emission.phase_mismatch_from_residual_index(20.0, 1000.0, -0.01)
    with pytest.raises(ValueError, match="'c'"):
        emission.phase_mismatch_from_residual_index(20.0, 1000.0, 0.025, c=0.0)
    with pytest.raises(ValueError, match="frequency"):
        emission.phase_mismatch_from_residual_index(20.0, 0.0, 0.025)


def test_result_phase_mismatch_matches_the_standalone_conversion() -> None:
    """The result's convenience conversion is the public function, band by band."""
    measured = _table_column("instrument", 1)
    result = emission.intensity_class_compliance(measured, _BANDS, spacing=0.012)
    expected = emission.phase_mismatch_from_residual_index(
        np.asarray(measured), np.asarray(_BANDS), 0.012
    )
    assert result.phase_mismatch() == pytest.approx(expected)


# --------------------------------------------------------------------------
# Per-band entries that do not agree
# --------------------------------------------------------------------------
def test_an_instrument_verdict_refuses_per_band_entries_that_disagree() -> None:
    """The fiche prints one row per band under the instrument's overall class."""
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0),
        frequencies,
        device="instrument",
        spacing=0.025,
    )
    with pytest.raises(ValueError, match="'bands'"):
        dataclasses.replace(result, bands=result.bands[:-1])


def test_a_verdict_whose_device_is_not_a_table_2_column_is_refused() -> None:
    """Two readers dispatch on ``device``, and neither survives an unknown tag.

    :func:`verify_intensity_class` checks the tag at the call, but a verdict
    rewritten by hand reaches the plot's title lookup once the masks are drawn
    and dies there with a bare ``KeyError`` naming the string alone, while the
    fiche's basis strip labels the same result as a complete instrument.
    """
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0),
        frequencies,
        device="instrument",
        spacing=0.025,
    )
    with pytest.raises(ValueError, match="'device' must be"):
        dataclasses.replace(result, device="microphone")


@pytest.mark.parametrize(
    "field_name", ["frequency", "residual_index", "limit_class1", "limit_class2"]
)
def test_a_non_finite_band_of_the_verdict_is_refused(field_name: str) -> None:
    """A NaN band prints as ``nan`` under a boxed verdict that still complies.

    Every comparison the class rests on is ``>=`` against a mask, and a NaN
    loses each of them silently: the band is marked failing, or the figure's
    axis autoscaling stops inside matplotlib naming no field at all.
    """
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0),
        frequencies,
        device="instrument",
        spacing=0.025,
    )
    values = np.asarray(getattr(result, field_name), dtype=float).copy()
    values[1] = float("nan")
    with pytest.raises(ValueError, match=f"'{field_name}' must be finite"):
        dataclasses.replace(result, **{field_name: values})


def test_a_non_finite_per_band_verdict_value_is_refused() -> None:
    """The certificate's table and its boxed verdict read different keys.

    The per-band rows print ``residual_index_db`` and the margin taken from
    it, while the box reads ``margin_class<n>_db``; a NaN in one of them gave
    an accredited verification certificate reading ``nan`` and ``+nan`` in the
    results table while still declaring ``Class 1 - COMPLIES``. Every one of
    these numbers descends from a spectrum the verifier validated finite.
    """
    import copy
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0),
        frequencies,
        device="instrument",
        spacing=0.025,
    )
    bands = tuple(copy.deepcopy(band) for band in result.bands)
    bands[0]["residual_index_db"] = float("nan")
    with pytest.raises(ValueError, match=r"'bands' must carry finite per-band"):
        dataclasses.replace(result, bands=bands)


def test_a_verdict_refuses_a_class_its_own_bands_do_not_derive() -> None:
    """The box and the table are one statement, and the box is not measured.

    The certificate prints the per-band classes in its table and the class of
    the whole instrument in its box, so a summary that does not restate the
    rows is the sheet contradicting itself. Built by hand it did: an
    instrument with one band meeting no class at all, whose honest summary is
    therefore ``None``, was accepted claiming class 1 and printed ``Class 1 -
    COMPLIES (binding margin -6.81 dB)``, a negative binding margin beside the
    word COMPLIES.

    The producer cannot emit the pair: it derives the summary from the band
    classes in one line. Only a result assembled by hand can hold it, which is
    why the guard is at construction.
    """
    import dataclasses

    frequencies = np.array([100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0])
    measured = np.full(frequencies.size, 18.0)
    measured[4] = 9.0  # this band meets neither class
    result = emission.intensity_class_compliance(
        measured, frequencies, device="instrument", spacing=0.012
    )
    assert result.overall_class is None
    with pytest.raises(
        ValueError, match=r"'overall_class' must be the class the bands derive"
    ):
        dataclasses.replace(result, overall_class=1)


def test_a_verdict_refuses_a_class_that_is_no_designation() -> None:
    """A class is a label the readers splice into a key, not a number.

    ``1.0`` reads as class 1 to every comparison and builds
    ``margin_class1.0_db``, a key no band carries, so it reached the boxed
    statement as a bare ``KeyError`` naming neither the field nor the result.
    """
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0), frequencies, device="instrument", spacing=0.025
    )
    with pytest.raises(
        ValueError, match=r"'overall_class' must be a class of \[1, 2\] or None"
    ):
        dataclasses.replace(result, overall_class=1.0)


def test_a_narrow_numpy_nan_per_band_value_is_refused() -> None:
    """A NaN margin held as ``np.float32`` is a NaN the boxed verdict misses.

    ``np.float64`` subclasses ``float`` and ``np.float32`` does not, so a
    single-precision NaN margin used to reach :meth:`binding_margin`, where
    :func:`min` compares it away against whichever value it meets first and
    returns a finite binding margin for a band that has none.
    """
    import copy
    import dataclasses

    frequencies = np.array([250.0, 500.0, 1000.0, 2000.0])
    result = emission.intensity_class_compliance(
        np.full(frequencies.size, 20.0),
        frequencies,
        device="instrument",
        spacing=0.025,
    )
    bands = tuple(copy.deepcopy(band) for band in result.bands)
    bands[1]["margin_class1_db"] = np.float32("nan")
    with pytest.raises(ValueError, match=r"'bands' must carry finite per-band"):
        dataclasses.replace(result, bands=bands)
