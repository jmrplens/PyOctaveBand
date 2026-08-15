#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power by intensity determination fiches (reportlab, ISO 9614-2 / -3).

Renders a
:class:`~phonometry.emission.sound_power_intensity.SoundPowerIntensityResult`
(sound power determined from the normal sound intensity scanned over a surface
enclosing the source, ISO 9614-2:1996 engineering grade 2 or survey grade 3) to
a one-page PDF laid out like a sound-power test sheet:

* a title and the standard-basis line naming the intensity-scanning method and
  its measurement grade;
* an optional metadata header grid (client, noise source, test environment,
  instrumentation, climate, date), rendered only for the fields supplied on
  the :class:`ReportMetadata`;
* a full-width per-band table (nominal octave/one-third-octave frequency and
  the intensity-derived band sound-power level ``LW``); with ``verbose=True``
  the table adds the field indicators ``FpI`` (surface pressure-intensity) and
  ``F+/-`` (negative partial power) and the per-band achieved grade;
* the sound-power spectrum ``LW(f)`` drawn by the result's own ``plot(ax=...)``
  (band axis with nominal labels; net-negative bands hatched as unusable);
* a boxed single-number result, the A-weighted sound power level ``LWA``
  (dB re 1 pW) with the total ``LW``, the measurement surface area ``S`` and
  the determination grade alongside;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (a sound-power emission passes at or below the limit);
* a measurement-basis strip stating the partial-power model, the field
  indicators and the Annex B qualification criteria, and a footer
  identity/disclaimer block.

The precision sibling,
:class:`~phonometry.emission.sound_power_intensity.PrecisionIntensityResult`
(ISO 9614-3:2002, the same scan qualified to accuracy grade 1), prints the same
sheet with the content **that** standard asks a report to state (clause 10):

* the *normalized* sound power level ``LW0`` beside ``LW`` in the per-band
  table, since clause 10 f) 2) asks for the level referred to the reference
  atmosphere of 23 degrees Celsius and 101 325 Pa (Eq. 10), and the per-band
  expanded uncertainty ``U``, twice the standard deviation of reproducibility
  of Table 1 (clause 4.3, clause 10 f) 4));
* the four Annex B field indicators ``FT``, ``Fp|In|``, ``FpIn`` and ``FS``,
  tabulated per band with ``verbose=True`` (clause 10 f) 1)) and summarised in
  the basis strip, with the pressure-residual intensity index of the probe
  (clause 10 d) 5)) and the dynamic capability it yields;
* the five Annex C acceptance criteria and the per-band grade cell they
  decide, and the statement clause 10 f) 2) makes mandatory: the bands whose
  criteria are not satisfied are omitted from the A-weighted determination and
  named in the basis strip, alongside the bands with net power ``P <= 0`` the
  method is not applicable to (clause 9.2).

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the ``LW(f)`` spectrum, the boxed ``LWA`` and the
flow assembly) is shared with the ISO 3744 pressure fiche in
:mod:`._sound_power_fiche`; this module only holds the intensity-method
specifics (the basis lines, the indicator/grade columns and the field-indicator
strips). reportlab, matplotlib and svglib are soft dependencies imported
lazily; each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._sound_power_fiche import (
    band_labels,
    d1,
    energy_sum,
    fraction_caption,
    power_statement,
    power_value_table,
    power_verdict,
    range_str,
    render_sound_power_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..emission.sound_power_intensity import (
        PrecisionCriteria,
        PrecisionFieldIndicators,
        PrecisionIntensityResult,
        SoundPowerIntensityResult,
    )

#: Bias-error factor K of ISO 9614-2:1996 Table 1, in dB, by measurement grade.
_K = {"engineering": 10.0, "survey": 7.0}
#: The achieved-grade phrase for the basis line and the boxed result term.
_GRADE_PHRASE = {
    "engineering": "engineering grade (accuracy grade 2)",
    "survey": "survey grade (accuracy grade 3)",
}
#: Per-band achieved-grade cell (the accuracy grade number, or an em dash).
_GRADE_CELL = {"engineering": "2", "survey": "3", "none": "—"}

#: The em dash printed wherever a quantity was not supplied or is not finite.
_ABSENT = "—"
#: Bias-error factor K of ISO 9614-3:2002 (definition 3.11), in dB. Part 3
#: recognises the single precision grade, so K is fixed rather than tabulated.
_K_PRECISION = 10.0
#: The grade phrase of the precision method (ISO 9614-3:2002 has one grade).
_PRECISION_GRADE_PHRASE = "precision grade (accuracy grade 1)"
#: Coverage factor of the expanded uncertainty: ISO 9614-3:2002 clause 4.3
#: states it as twice the Table 1 standard deviation of reproducibility, for a
#: coverage probability of 95 % as defined in the GUM.
_COVERAGE_FACTOR = 2.0
#: The Table 1 standard deviation of reproducibility of the A-weighted sound
#: power level, in dB (footnote b: a relatively flat 50 Hz to 6,3 kHz spectrum).
_SIGMA_A_WEIGHTED = 1.0
#: Band count above which the precision table pairs its rows into two column
#: groups. The one-third-octave range this method works in does not fit the
#: page as one column of rows beside a header grid, a spectrum, a boxed result
#: and the Annex B/C strips, and an accredited sheet prints a wide band set in
#: two groups rather than on a second page.
_SPLIT_ABOVE = 8
#: Size of the embedded spectrum on the precision sheet, in inches. Shallower
#: than the pressure fiches': this sheet carries a one-third-octave band set,
#: and the rows it costs have to come from somewhere for the determination to
#: stay on one page.
_PRECISION_FIGSIZE = (9.2, 2.5)


def _basis(result: Any, language: str = "en") -> str:
    """The standard-basis line naming the scanning method and its grade."""
    grade = _GRADE_PHRASE.get(getattr(result, "grade", "engineering"),
                              _GRADE_PHRASE["engineering"])
    return t(
        "Determination of the sound power level from the normal sound intensity "
        "scanned over a surface enclosing the source (ISO 9614-2:1996, "
        "{grade}).",
        language,
    ).format(grade=t(grade, language))


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band table (nominal frequency, LW, indicators).

    The default table is the recommended ``f | LW`` form; ``verbose`` adds the
    field indicators ``FpI`` and ``F+/-`` and the per-band achieved grade. The
    styling and octave grouping are shared with the pressure fiche via
    :func:`._sound_power_fiche.power_value_table`. Called only after the
    renderer has imported reportlab.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    n = lw.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if not verbose:
        header = [t("f [Hz]", language), "L<sub>W</sub> [dB]"]
        widths = [87.0, 87.0]
        rows_data = [[labels[i], d1(lw[i], language)] for i in range(n)]
        return power_value_table(header, rows_data, widths, fraction)

    fpi = _band_array(result.surface_pressure_intensity_index, n)
    fpm = _band_array(result.negative_partial_power_index, n)
    grade = result.achieved_grade
    header = [
        t("f [Hz]", language),
        "L<sub>W</sub> [dB]",
        "F<sub>pI</sub> [dB]",
        "F<sub>+/&#8722;</sub> [dB]",
        t("Grade", language),
    ]
    widths = [30.0, 29.0, 29.0, 29.0, 57.0]
    rows_data = [
        [
            labels[i],
            d1(lw[i], language),
            d1(fpi[i], language),
            d1(fpm[i], language),
            _grade_cell(grade, i),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _band_array(values: np.ndarray | None, n: int) -> np.ndarray:
    """A per-band float array, or all-``NaN`` when the quantity is absent."""
    if values is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _grade_cell(grade: np.ndarray | None, i: int) -> str:
    """The per-band achieved-grade cell (grade number, or an em dash)."""
    if grade is None:
        return _ABSENT
    return _GRADE_CELL.get(str(grade[i]), _ABSENT)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed sound-power result and its extended terms.

    Delegates the ``LWA``/``LW`` box to the shared
    :func:`._sound_power_fiche.power_statement` and appends the intensity-method
    extended terms: the measurement surface area ``S`` and the determination
    grade (the intensity result carries no expanded uncertainty).
    """
    statement, extended = power_statement(result, language)
    surface = float(result.surface_area)
    grade_phrase = _GRADE_PHRASE.get(getattr(result, "grade", "engineering"),
                                     _GRADE_PHRASE["engineering"])
    extended.append(
        t("Measurement surface S = {value} m<super>2</super>", language).format(
            value=format_number(surface, language, decimals=2)
        )
    )
    extended.append(
        t("Determination grade: {grade}", language).format(
            grade=t(grade_phrase, language)
        )
    )
    return statement, extended


def _indicator_strip(result: Any, language: str = "en") -> str:
    """The partial-power / field-indicator line for the basis strip."""
    grade = getattr(result, "grade", "engineering")
    return t(
        "L<sub>W</sub> = 10 lg(P/P<sub>0</sub>), with the total power P the sum "
        "of the segment partial powers P<sub>i</sub> = I<sub>n,i</sub> "
        "S<sub>i</sub>, P<sub>0</sub> = 1 pW (ISO 9614-2:1996 Eq. 6/12/13). "
        "Surface pressure-intensity indicator F<sub>pI</sub> = {fpi} dB, "
        "negative-partial-power indicator F<sub>+/&#8722;</sub> = {fpm} dB, "
        "dynamic capability L<sub>d</sub> = {ld} dB (bias factor K = {k} dB).",
        language,
    ).format(
        fpi=_indicator_range(result.surface_pressure_intensity_index, language),
        fpm=_indicator_range(result.negative_partial_power_index, language),
        ld=_indicator_range(result.dynamic_capability_index, language),
        k=int(_K.get(grade, 10.0)),
    )


def _criteria_strip(result: Any, language: str = "en") -> str:
    """The Annex B qualification / A-weighting line for the basis strip."""
    text = t(
        "A band reaches engineering grade when L<sub>d</sub> &gt; F<sub>pI</sub>"
        ", F<sub>+/&#8722;</sub> &#8804; 3 dB and the two sweeps repeat within "
        "the ISO 9614-2:1996 Table 2 limit s per segment; survey grade drops "
        "the F<sub>+/&#8722;</sub> criterion (Annex B). The A-weighted L"
        "<sub>WA</sub> sums the determinable bands only: a band with net "
        "power P &#8804; 0 is not determinable (clause 9.2), and a band "
        "failing criterion 1 and/or 2 is omitted (clause 10.6 b). Levels "
        "are referenced to the reference sound power 1 pW.",
        language,
    )
    omitted = getattr(result, "a_weighting_omitted_bands", None)
    if omitted is not None and bool(np.any(omitted)):
        mask = np.asarray(omitted, dtype=bool)
        frequencies = getattr(result, "frequencies", None)
        labels, _fraction = band_labels(frequencies, mask.size)
        bands = ", ".join(str(labels[i]) for i in range(mask.size) if mask[i])
        if frequencies is not None:
            bands += " Hz"
        text += " " + t(
            "Bands omitted from L<sub>WA</sub> per clause 10.6 b: {bands}.",
            language,
        ).format(bands=bands)
    return text


def _indicator_range(values: np.ndarray | None, language: str = "en") -> str:
    """A field-indicator range, or an em dash when the indicator is absent."""
    if values is None:
        return _ABSENT
    return range_str(np.asarray(values, dtype=np.float64), language)


def render_intensity_power_report(
    result: SoundPowerIntensityResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a sound-power-by-intensity determination fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.emission.sound_power_intensity.SoundPowerIntensityResult`
        carrying the per-band ``sound_power_level``, the field indicators and
        the A-weighted total.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the declared A-weighted sound-power limit.
    :param verbose: When ``True`` the per-band table adds the field indicators
        ``FpI`` and ``F+/-`` and the per-band achieved grade.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    statement, extended = _statement(result, language)
    return render_sound_power_fiche(
        result,
        path,
        title=t("Sound power determination", language),
        basis=_basis(result, language),
        caption=fraction_caption(result, language),
        value_table=_value_table(result, verbose, language),
        statement=statement,
        extended=extended,
        basis_strips=[
            _indicator_strip(result, language),
            _criteria_strip(result, language),
        ],
        metadata=metadata,
        language=language,
    )


# ===========================================================================
# ISO 9614-3:2002 - the precision fiche. The same scan and the same sheet, but
# the content clause 10 asks a report of *this* part to state: the normalized
# level, the per-band uncertainty, the four Annex B indicators and the five
# Annex C criteria, with the mandatory statement naming the bands their
# rejection keeps out of the A-weighted determination.
# ===========================================================================


def _precision_basis(language: str = "en") -> str:
    """The standard-basis line of the precision scanning method."""
    return t(
        "Determination of the sound power level from the normal sound intensity "
        "scanned over a surface enclosing the source, qualified by the Annex C "
        "procedure (ISO 9614-3:2002, {grade}).",
        language,
    ).format(grade=t(_PRECISION_GRADE_PHRASE, language))


def _band_uncertainty(result: Any) -> np.ndarray:
    """Per-band expanded uncertainty ``U = 2 sigma_R0`` (Table 1), in dB.

    ISO 9614-3:2002 clause 4.3 states the expanded uncertainty for a coverage
    probability of 95 % as twice the standard deviation of reproducibility of
    Table 1, which is tabulated for the one-third-octave bands from 50 Hz to
    6,3 kHz. A band outside that set, or a determination whose bands are not
    one-third octaves, has no tabulated value and reports ``NaN`` (an em dash
    on the sheet) rather than an uncertainty the standard does not give.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    n = lw.size
    uncertainty = np.full(n, np.nan, dtype=np.float64)
    frequencies = getattr(result, "frequencies", None)
    if frequencies is None:
        return uncertainty
    labels, fraction = band_labels(frequencies, n)
    if fraction != 3:
        return uncertainty
    from ..emission.sound_power_intensity import _sigma_r0_9614_3

    for i, label in enumerate(labels):
        try:
            sigma = _sigma_r0_9614_3(round(float(label)))
        except ValueError:  # a band outside the Table 1 range
            continue
        uncertainty[i] = _COVERAGE_FACTOR * sigma
    return uncertainty


def _paired(
    header: list[str], rows_data: list[list[str]]
) -> tuple[list[str], list[list[str]], list[float]]:
    """Fold a tall band table into two column groups printed side by side.

    The lower half of the band set heads the left group and the upper half the
    right one, separated by a narrow blank column; an odd band count leaves
    the last right-hand row empty. Returns the paired ``(header, rows,
    widths)`` for :func:`._sound_power_fiche.power_value_table`.
    """
    half = (len(rows_data) + 1) // 2
    blank = [""] * len(header)
    paired_rows = [
        [*rows_data[i], "", *(rows_data[i + half] if i + half < len(rows_data) else blank)]
        for i in range(half)
    ]
    widths = [20.0, 22.0, 22.0, 18.0, 10.0, 20.0, 22.0, 22.0, 18.0]
    return [*header, "", *header], paired_rows, widths


def _precision_grade_cell(criteria: PrecisionCriteria | None, i: int) -> str:
    """The per-band grade cell: accuracy grade 1 when qualified, else em dash."""
    if criteria is None or criteria.qualified is None:
        return _ABSENT
    return "1" if bool(criteria.qualified[i]) else _ABSENT


def _precision_value_table(
    result: Any,
    indicators: PrecisionFieldIndicators | None,
    criteria: PrecisionCriteria | None,
    verbose: bool,
    language: str = "en",
) -> Any:
    """Build the per-band table of the precision fiche.

    The default columns are the ones clause 10 f) asks for: the band level
    ``LW``, the normalized level ``LW0`` the standard reports (Eq. 10, clause
    10 f) 2)) and the expanded uncertainty ``U`` (clause 10 f) 4)). A band set
    wider than :data:`_SPLIT_ABOVE` is printed in two column groups side by
    side, as accredited sheets print the one-third-octave range this method
    works in, which is what keeps the whole determination on one page.
    ``verbose`` adds the tabulation of the four Annex B field indicators clause
    10 f) 1) requires and the grade cell the Annex C criteria decide; those
    nine columns leave no room to pair, so a wide verbose set prints as one
    tall table and can run onto a second page. Called only after the renderer
    has imported reportlab.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    lw0 = np.asarray(result.sound_power_level_normalized, dtype=np.float64)
    n = lw.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)
    uncertainty = _band_uncertainty(result)

    if not verbose:
        header = [
            t("f [Hz]", language),
            "L<sub>W</sub> [dB]",
            "L<sub>W0</sub> [dB]",
            "U [dB]",
        ]
        rows_data = [
            [
                labels[i],
                d1(lw[i], language),
                d1(lw0[i], language),
                d1(uncertainty[i], language),
            ]
            for i in range(n)
        ]
        if n > _SPLIT_ABOVE:
            return power_value_table(*_paired(header, rows_data), 0)
        return power_value_table(header, rows_data, [43.5] * 4, fraction)

    ft = _band_array(None if indicators is None else indicators.ft, n)
    unsigned = _band_array(None if indicators is None else indicators.f_pi_unsigned, n)
    signed = _band_array(None if indicators is None else indicators.f_pi_signed, n)
    fs = _band_array(None if indicators is None else indicators.fs, n)
    header = [
        t("f [Hz]", language),
        "L<sub>W</sub> [dB]",
        "L<sub>W0</sub> [dB]",
        "U [dB]",
        "F<sub>T</sub>",
        "F<sub>p|In|</sub> [dB]",
        "F<sub>pIn</sub> [dB]",
        "F<sub>S</sub>",
        t("Grade", language),
    ]
    widths = [19.0, 21.0, 21.0, 17.0, 16.0, 22.0, 21.0, 16.0, 21.0]
    rows_data = [
        [
            labels[i],
            d1(lw[i], language),
            d1(lw0[i], language),
            d1(uncertainty[i], language),
            d1(ft[i], language),
            d1(unsigned[i], language),
            d1(signed[i], language),
            d1(fs[i], language),
            _precision_grade_cell(criteria, i),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _precision_caption(result: Any, language: str = "en") -> str:
    """The band-set caption, extended with the frequency range determined.

    ISO 9614-3:2002 clause 4.3 covers the one-third-octave bands from 50 Hz to
    6,3 kHz and requires a determination made over a more restricted range to
    state that range. Naming the first and last band in the caption states it
    where the reader meets the table, and costs the sheet no line of its own.
    """
    caption = fraction_caption(result, language)
    frequencies = getattr(result, "frequencies", None)
    n = np.asarray(result.sound_power_level, dtype=np.float64).size
    if frequencies is None or n < 2:
        return caption
    labels, _fraction = band_labels(frequencies, n)
    return t("{caption}, {lo} Hz to {hi} Hz", language).format(
        caption=caption, lo=labels[0], hi=labels[-1]
    )


def _precision_omitted(result: Any, criteria: PrecisionCriteria | None) -> np.ndarray:
    """The bands the A-weighted determination leaves out, per band.

    Two rules put a band here, and the sheet has to distinguish them: the
    method is not applicable to a band whose net power is non-positive (clause
    9.2), and clause 10 f) 2) omits the contribution of a band whose Annex C
    criteria are not satisfied. Without the criteria only the first rule can
    be applied, and the strip says so.
    """
    omitted = np.asarray(result.not_applicable_band, dtype=bool).copy()
    if criteria is not None and criteria.qualified is not None:
        omitted |= ~np.asarray(criteria.qualified, dtype=bool)
    return omitted


def _precision_level_a(result: Any, omitted: np.ndarray) -> float:
    """The A-weighted level the fiche boxes: the qualified bands only.

    The result object sums every applicable band, because it is computed
    before the Annex C criteria are evaluated. Clause 10 f) 2) requires the
    bands the criteria reject to be dropped from the A-weighted determination,
    so the fiche recombines the levels over the bands that survive both rules.
    Recomputing is skipped when the criteria reject nothing, which keeps the
    printed number identical to ``result.sound_power_level_a``.
    """
    not_applicable = np.asarray(result.not_applicable_band, dtype=bool)
    if np.array_equal(omitted, not_applicable):
        return float(result.sound_power_level_a)
    from ..emission.sound_power_intensity import _precision_a_weighted_total

    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    return _precision_a_weighted_total(
        lw, omitted, getattr(result, "frequencies", None), lw.size
    )


def _precision_statement(
    result: Any, level_a: float, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed precision result and its extended terms.

    Boxes the A-weighted level of the qualified bands and states beside it the
    quantities ISO 9614-3:2002 asks the report to carry: the band-summed total
    ``LW`` with its normalized counterpart ``LW0`` (Eq. 10) on the same line,
    the measurement surface area, the expanded uncertainty of clause 4.3 and
    the single grade the method recognises.
    """
    statement, _base = power_statement(result, language, level_a=level_a)
    extended: list[str] = []
    total = energy_sum(result.sound_power_level)
    total_normalized = energy_sum(result.sound_power_level_normalized)
    if math.isfinite(total):
        extended.append(
            t(
                "Total L<sub>W</sub> = {value} dB, normalized L<sub>W0</sub> = "
                "{normalized} dB re {ref}",
                language,
            ).format(
                value=d1(total, language),
                normalized=d1(total_normalized, language),
                ref="1 pW",
            )
        )
    extended.append(
        t("Measurement surface S = {value} m<super>2</super>", language).format(
            value=format_number(float(result.surface_area), language, decimals=2)
        )
    )
    if math.isfinite(level_a):
        extended.append(
            t("Expanded uncertainty U = {value} dB", language).format(
                value=d1(_COVERAGE_FACTOR * _SIGMA_A_WEIGHTED, language)
            )
        )
    extended.append(
        t("Determination grade: {grade}", language).format(
            grade=t(_PRECISION_GRADE_PHRASE, language)
        )
    )
    return statement, extended


def _normalization_offset(result: Any) -> float:
    """The applied meteorological normalization ``LW - LW0``, in dB (Eq. 10).

    The term depends only on the test atmosphere, so it is the same in every
    band; the first determinable band carries it for the whole sheet.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    lw0 = np.asarray(result.sound_power_level_normalized, dtype=np.float64)
    offset = lw - lw0
    finite = offset[np.isfinite(offset)]
    return float(finite[0]) if finite.size else float("nan")


def _dynamic_capability(residual_index: Any) -> np.ndarray | None:
    """``Ld = delta_pI0 - K`` with the fixed precision ``K``, or ``None``."""
    if residual_index is None:
        return None
    return np.atleast_1d(np.asarray(residual_index, dtype=np.float64)) - _K_PRECISION


def _precision_model_strip(
    result: Any,
    indicators: PrecisionFieldIndicators | None,
    residual_index: Any,
    language: str = "en",
) -> str:
    """The partial-power / normalization / field-indicator line of the strip."""
    offset = _normalization_offset(result)
    residual = (
        None
        if residual_index is None
        else np.atleast_1d(np.asarray(residual_index, dtype=np.float64))
    )
    return t(
        "L<sub>W</sub> = 10 lg(P/P<sub>0</sub>), P the sum of the "
        "partial-surface powers P<sub>i</sub> = I<sub>n,i</sub> S<sub>i</sub>, "
        "P<sub>0</sub> = 1 pW (ISO 9614-3:2002 Eq. 5/8/9); L<sub>W0</sub> = "
        "L<sub>W</sub> &#8722; 15 lg[(B/101 325)(296,15/(273,15 + &#952;))] "
        "refers it to 23 &#176;C, 101 325 Pa (Eq. 10), applied normalization "
        "{offset} dB. Annex B indicators F<sub>T</sub> = {ft}, "
        "F<sub>p|In|</sub> = {unsigned} dB, F<sub>pIn</sub> = {signed} dB, "
        "F<sub>S</sub> = {fs}; pressure-residual intensity index "
        "&#948;<sub>pI0</sub> = {residual} dB (IEC 61043), dynamic capability "
        "L<sub>d</sub> = {ld} dB.",
        language,
    ).format(
        offset=(
            _ABSENT
            if not math.isfinite(offset)
            else format_number(offset, language, decimals=2)
        ),
        ft=_indicator_range(None if indicators is None else indicators.ft, language),
        unsigned=_indicator_range(
            None if indicators is None else indicators.f_pi_unsigned, language
        ),
        signed=_indicator_range(
            None if indicators is None else indicators.f_pi_signed, language
        ),
        fs=_indicator_range(None if indicators is None else indicators.fs, language),
        residual=_indicator_range(residual, language),
        ld=_indicator_range(_dynamic_capability(residual_index), language),
    )


def _failed_criteria(criteria: PrecisionCriteria | None, i: int) -> list[int]:
    """The numbers of the Annex C criteria a band does not satisfy."""
    if criteria is None:
        return []
    evaluated = (
        criteria.criterion_1,
        criteria.criterion_2,
        criteria.criterion_3,
        criteria.criterion_4,
        criteria.criterion_5,
    )
    return [
        number
        for number, values in enumerate(evaluated, start=1)
        if values is not None and not bool(values[i])
    ]


def _omission_reason(
    result: Any, criteria: PrecisionCriteria | None, i: int, language: str = "en"
) -> str:
    """Why a band is out of the A-weighted determination: clause 9.2, or which criteria."""
    if bool(np.asarray(result.not_applicable_band, dtype=bool)[i]):
        return t("clause 9.2", language)
    failed = _failed_criteria(criteria, i)
    numbers = ", ".join(str(number) for number in failed)
    if len(failed) == 1:
        return t("criterion {numbers}", language).format(numbers=numbers)
    return t("criteria {numbers}", language).format(numbers=numbers)


def _omitted_band_list(
    result: Any,
    criteria: PrecisionCriteria | None,
    omitted: np.ndarray,
    language: str = "en",
) -> str:
    """The omitted bands, each with the rule that omitted it."""
    labels, _fraction = band_labels(getattr(result, "frequencies", None), omitted.size)
    unit = " Hz" if getattr(result, "frequencies", None) is not None else ""
    return ", ".join(
        f"{labels[i]}{unit} ({_omission_reason(result, criteria, i, language)})"
        for i in range(omitted.size)
        if bool(omitted[i])
    )


def _precision_criteria_strip(
    result: Any,
    criteria: PrecisionCriteria | None,
    omitted: np.ndarray,
    language: str = "en",
) -> str:
    """The Annex C qualification / uncertainty line, with the mandatory statement."""
    text = t(
        "Annex C qualifies a band when the two scans repeat within "
        "|L<sub>In</sub>(1) &#8722; L<sub>In</sub>(2)| &#8804; s/2, s from "
        "Table 1 (criterion 1), L<sub>d</sub> &#8805; F<sub>pIn</sub> with "
        "K = 10 dB (2), F<sub>pIn</sub> &#8722; F<sub>p|In|</sub> &#8804; 3 dB "
        "(3) and F<sub>S</sub> &#8804; 2 (4), or a doubled scan-line density "
        "gives 0,83 &#8804; F<sub>S</sub>(1)/F<sub>S</sub>(2) &#8804; 1,2 (5), "
        "which qualifies the band even where F<sub>S</sub>(2) &#8805; 2. The "
        "A-weighted L<sub>WA</sub> sums the qualified bands only: a band whose "
        "criteria are not satisfied is omitted (clause 10 f) 2)), as is a band "
        "with net power P &#8804; 0, to which the method is not applicable "
        "(clause 9.2). U is twice the Table 1 standard deviation of "
        "reproducibility (clause 4.3, coverage probability 95 %). Levels are "
        "referenced to the reference sound power 1 pW.",
        language,
    )
    if criteria is None or criteria.qualified is None:
        return text + " " + t(
            "The Annex C qualification was not supplied with this "
            "determination, so no band is omitted on its criteria.",
            language,
        )
    if not bool(np.any(omitted)):
        return text + " " + t(
            "Every band satisfies the Annex C criteria; none is omitted from "
            "L<sub>WA</sub>.",
            language,
        )
    return text + " " + t(
        "Bands omitted from L<sub>WA</sub>: {bands}.", language
    ).format(bands=_omitted_band_list(result, criteria, omitted, language))


def render_precision_intensity_report(
    result: PrecisionIntensityResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    indicators: PrecisionFieldIndicators | None = None,
    criteria: PrecisionCriteria | None = None,
    residual_index: Any = None,
) -> str:
    """Render an ISO 9614-3 precision sound-power fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.emission.sound_power_intensity.PrecisionIntensityResult`
        carrying the per-band ``sound_power_level``, its normalized
        counterpart and the A-weighted total.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the declared A-weighted sound-power limit.
    :param verbose: When ``True`` the per-band table adds the Annex B field
        indicators and the per-band grade cell.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :param indicators: Optional :class:`PrecisionFieldIndicators` (clause
        10 f) 1)); tabulated with ``verbose`` and summarised in the strip.
    :param criteria: Optional :class:`PrecisionCriteria`; its ``qualified``
        mask drives the per-band grade cell and the clause 10 f) 2) omission.
    :param residual_index: Optional pressure-residual intensity index
        ``delta_pI0`` of the probe and analyser (clause 10 d) 5)), scalar or
        per band; it yields the dynamic capability ``Ld`` of criterion 2.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    omitted = _precision_omitted(result, criteria)
    level_a = _precision_level_a(result, omitted)
    statement, extended = _precision_statement(result, level_a, language)
    verdict = (
        None
        if metadata is None or metadata.requirement is None
        else power_verdict(result, metadata.requirement, language, level_a=level_a)
    )
    return render_sound_power_fiche(
        result,
        path,
        title=t("Sound power determination", language),
        basis=_precision_basis(language),
        caption=_precision_caption(result, language),
        value_table=_precision_value_table(
            result, indicators, criteria, verbose, language
        ),
        statement=statement,
        extended=extended,
        basis_strips=[
            _precision_model_strip(result, indicators, residual_index, language),
            _precision_criteria_strip(result, criteria, omitted, language),
        ],
        metadata=metadata,
        language=language,
        verdict=verdict,
        figsize=_PRECISION_FIGSIZE,
    )
