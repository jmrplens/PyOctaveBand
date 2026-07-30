#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power by intensity determination fiche (reportlab, ISO 9614-2).

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

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the ``LW(f)`` spectrum, the boxed ``LWA`` and the
flow assembly) is shared with the ISO 3744 pressure fiche in
:mod:`._sound_power_fiche`; this module only holds the intensity-method
specifics (the basis line, the ``FpI``/``F+/-``/grade columns and the
field-indicator strip). reportlab, matplotlib and svglib are soft dependencies
imported lazily; each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
from ._sound_power_fiche import (
    band_labels,
    d1,
    fraction_caption,
    power_statement,
    power_value_table,
    range_str,
    render_sound_power_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..emission.sound_power_intensity import SoundPowerIntensityResult

#: Bias-error factor K of ISO 9614-2:1996 Table 1, in dB, by measurement grade.
_K = {"engineering": 10.0, "survey": 7.0}
#: The achieved-grade phrase for the basis line and the boxed result term.
_GRADE_PHRASE = {
    "engineering": "engineering grade (accuracy grade 2)",
    "survey": "survey grade (accuracy grade 3)",
}
#: Per-band achieved-grade cell (the accuracy grade number, or an em dash).
_GRADE_CELL = {"engineering": "2", "survey": "3", "none": "—"}


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
        return "—"
    return _GRADE_CELL.get(str(grade[i]), "—")


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed sound-power result and its extended terms.

    Delegates the ``LWA``/``LW`` box to the shared
    :func:`._sound_power_fiche.power_statement` and appends the intensity-method
    extended terms: the measurement surface area ``S`` and the determination
    grade (the intensity result carries no expanded uncertainty).
    """
    from ._i18n import format_number

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
        return "—"
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
