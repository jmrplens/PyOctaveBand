#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound power determination fiche (reportlab renderer, ISO 3744 / ISO 3745).

Renders a :class:`~phonometry.emission.sound_power.SoundPowerResult`
(enveloping-surface pressure method, ISO 3744:2010 engineering grade 2 or
ISO 3746:2010 survey grade 3) or a
:class:`~phonometry.emission.sound_power_anechoic.PrecisionSoundPowerResult`
(precision method in an anechoic or hemi-anechoic room, ISO 3745:2012 grade 1)
to a one-page PDF laid out like a sound-power test sheet:

* a title and the standard-basis line naming the applied method and accuracy
  grade;
* an optional metadata header grid (client, noise source, test environment,
  instrumentation, climate, date), rendered only for the fields supplied on
  the :class:`ReportMetadata`;
* a full-width per-band table (nominal octave/one-third-octave frequency, the
  surface sound-pressure level ``Lp`` and the band sound-power level ``LW``),
  grouped by octave for a one-third-octave set; with ``verbose=True`` the table
  adds the energy-averaged level ``Lp'`` and the applied corrections (the
  background ``K1`` and environmental ``K2`` for the ISO 3744/3746 surface
  method);
* the sound-power spectrum ``LW(f)`` drawn by the result's own ``plot(ax=...)``
  (band axis with nominal labels), so the chart is native to the library;
* a boxed single-number result, the A-weighted sound power level ``LWA``
  (dB re 1 pW) with the total ``LW``, the expanded uncertainty ``U`` and the
  measurement surface area ``S`` alongside;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (a sound-power emission passes at or below the limit);
* a short measurement-basis strip stating the correction model and the applied
  ``K1``/``K2`` (surface method) or ``C1``/``C2``/``C3`` (precision method),
  and a footer identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the ``LW(f)`` spectrum, the boxed ``LWA`` and the
flow assembly) is shared with the ISO 9614 intensity fiche in
:mod:`._sound_power_fiche`; this module only holds the pressure-method
specifics (the basis line, the ``Lp``/``K1``/``K2`` table columns and the
correction strip). reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]``
extra, matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
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
    from ..emission.sound_power import SoundPowerResult
    from ..emission.sound_power_anechoic import PrecisionSoundPowerResult


def _is_precision(result: Any) -> bool:
    """Return ``True`` for an ISO 3745 precision result, ``False`` for ISO 3744.

    The precision result carries the meteorological corrections ``c1``/``c2``/
    ``c3`` (and no environmental ``K2``); the surface-pressure result carries
    the per-band ``environmental_correction`` (``K2``) and a ``grade``. The
    duck-typed check keeps this module free of a module-level import of the
    domain classes (the render layer imports them only under TYPE_CHECKING).
    """
    return hasattr(result, "c1") and hasattr(result, "c2") and hasattr(result, "c3")


def _method(result: Any, language: str = "en") -> tuple[str, str]:
    """Return ``(standard, grade_phrase)`` for the applied determination method.

    ISO 3744:2010 is the engineering method (accuracy grade 2), ISO 3746:2010
    the survey method (grade 3) and ISO 3745:2012 the precision method
    (grade 1). The surface result distinguishes engineering from survey by its
    ``grade`` attribute.
    """
    if _is_precision(result):
        return "ISO 3745:2012", t("precision method, accuracy grade 1", language)
    if getattr(result, "grade", "engineering") == "survey":
        return "ISO 3746:2010", t("survey method, accuracy grade 3", language)
    return "ISO 3744:2010", t("engineering method, accuracy grade 2", language)


def _basis(result: Any, language: str = "en") -> str:
    """The standard-basis line naming the method, the surface and the grade."""
    standard, grade_phrase = _method(result, language)
    if _is_precision(result):
        return t(
            "Determination of the sound power level from sound pressure over an "
            "enveloping measurement surface in an anechoic or hemi-anechoic "
            "room ({standard}, {grade}).",
            language,
        ).format(standard=standard, grade=grade_phrase)
    return t(
        "Determination of the sound power level from sound pressure over an "
        "enveloping measurement surface ({standard}, {grade}).",
        language,
    ).format(standard=standard, grade=grade_phrase)


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band table (nominal frequency, Lp, LW).

    ``verbose`` adds the energy-averaged level ``Lp'`` and, for the ISO 3744/
    3746 surface method, the background ``K1`` and environmental ``K2``
    corrections. The styling and octave grouping are shared with the intensity
    fiche via :func:`._sound_power_fiche.power_value_table`; this only assembles
    the pressure-method columns. Called only after the renderer has imported
    reportlab.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    lp = np.asarray(result.surface_pressure_level, dtype=np.float64)
    lp_mean = np.asarray(result.mean_pressure_level, dtype=np.float64)
    n = lw.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    precision = _is_precision(result)
    if verbose and not precision:
        k1 = np.asarray(result.background_correction, dtype=np.float64)
        k2 = np.asarray(result.environmental_correction, dtype=np.float64)
        header = [
            t("f [Hz]", language),
            "L'<sub>p</sub> [dB]",
            "K<sub>1</sub> [dB]",
            "K<sub>2</sub> [dB]",
            "L<sub>p</sub> [dB]",
            "L<sub>W</sub> [dB]",
        ]
        widths = [30.0, 29.0, 29.0, 29.0, 29.0, 28.0]
        rows_data = [
            [labels[i], d1(lp_mean[i], language), d1(k1[i], language),
             d1(k2[i], language), d1(lp[i], language), d1(lw[i], language)]
            for i in range(n)
        ]
    elif verbose and precision:
        header = [
            t("f [Hz]", language),
            "L'<sub>p</sub> [dB]",
            "L<sub>p</sub> [dB]",
            "L<sub>W</sub> [dB]",
        ]
        widths = [45.0, 43.0, 43.0, 43.0]
        rows_data = [
            [labels[i], d1(lp_mean[i], language), d1(lp[i], language),
             d1(lw[i], language)]
            for i in range(n)
        ]
    else:
        header = [
            t("f [Hz]", language),
            "L<sub>p</sub> [dB]",
            "L<sub>W</sub> [dB]",
        ]
        widths = [58.0, 58.0, 58.0]
        rows_data = [
            [labels[i], d1(lp[i], language), d1(lw[i], language)]
            for i in range(n)
        ]

    return power_value_table(header, rows_data, widths, fraction)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed sound-power result and its extended terms.

    Delegates the ``LWA``/``LW`` box to the shared
    :func:`._sound_power_fiche.power_statement` and appends the pressure-method
    extended terms: the expanded uncertainty ``U`` and the measurement surface
    area ``S``.
    """
    statement, extended = power_statement(result, language)
    uncertainty = float(result.uncertainty)
    surface = float(result.surface_area)
    extended.append(
        t("Expanded uncertainty U = {value} dB", language).format(
            value=d1(uncertainty, language)
        )
    )
    extended.append(
        t("Measurement surface S = {value} m<super>2</super>", language).format(
            value=format_number(surface, language, decimals=2)
        )
    )
    return statement, extended


def _corrections_strip(result: Any, language: str = "en") -> str:
    """The applied-corrections line for the basis strip (K1/K2 or C1/C2/C3)."""
    if _is_precision(result):
        return t(
            "L<sub>W</sub> = L<sub>p</sub> + 10 lg(S/S<sub>0</sub>) + C1 + C2 + "
            "C3, S<sub>0</sub> = 1 m<super>2</super> (ISO 3745:2012 Eq. 14/15). "
            "Applied meteorological corrections C1 = {c1} dB, C2 = {c2} dB, "
            "C3 = {c3} dB.",
            language,
        ).format(
            c1=format_number(float(result.c1), language, decimals=2),
            c2=format_number(float(result.c2), language, decimals=2),
            c3=range_str(np.atleast_1d(np.asarray(result.c3, dtype=np.float64)), language),
        )
    return t(
        "Surface level L<sub>p</sub> is the energy average over the microphone "
        "positions corrected for background noise (K1, ISO 3744:2010 Eq. 16) "
        "and the test environment (K2, Eq. A.2); L<sub>W</sub> = L<sub>p</sub> "
        "+ 10 lg(S/S<sub>0</sub>), S<sub>0</sub> = 1 m<super>2</super> "
        "(Eq. 18). Applied corrections K1 = {k1} dB, K2 = {k2} dB.",
        language,
    ).format(
        k1=range_str(np.asarray(result.background_correction, dtype=np.float64), language),
        k2=range_str(np.asarray(result.environmental_correction, dtype=np.float64), language),
    )


def _a_weighting_strip(result: Any, language: str = "en") -> str:
    """The A-weighting citation line, keyed to the method that combined LWA.

    The surface-pressure result combines the band levels with the ISO 3744:2010
    Annex E corrections (Eq. E.1); the precision result combines them with the
    ISO 3745:2012 Annex C corrections (Eq. C.1). Citing the method that actually
    produced ``sound_power_level_a`` keeps the fiche truthful for both.
    """
    if _is_precision(result):
        return t(
            "The A-weighted sound power level L<sub>WA</sub> combines the band "
            "levels with the A-weighting band corrections of ISO 3745:2012 "
            "Annex C (Eq. C.1). Levels are referenced to the reference sound "
            "power 1 pW.",
            language,
        )
    return t(
        "The A-weighted sound power level L<sub>WA</sub> combines the band "
        "levels with the A-weighting band corrections of ISO 3744:2010 Annex E "
        "(Tables E.1/E.2, Eq. E.1). Levels are referenced to the reference "
        "sound power 1 pW.",
        language,
    )


def render_sound_power_report(
    result: SoundPowerResult | PrecisionSoundPowerResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a sound-power determination fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.emission.sound_power.SoundPowerResult`
        (ISO 3744/3746 enveloping-surface pressure method) or a
        :class:`~phonometry.emission.sound_power_anechoic.PrecisionSoundPowerResult`
        (ISO 3745 precision method) carrying the per-band ``sound_power_level``,
        ``surface_pressure_level`` and A-weighted total.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the declared A-weighted sound-power limit.
    :param verbose: When ``True`` the per-band table adds the energy-averaged
        level ``Lp'`` and the applied corrections (``K1``/``K2`` for the
        surface method).
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
            _corrections_strip(result, language),
            _a_weighting_strip(result, language),
        ],
        metadata=metadata,
        language=language,
    )
