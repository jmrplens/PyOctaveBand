#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Reverberation-room sound power determination fiche (reportlab, ISO 3741).

Renders a
:class:`~phonometry.emission.sound_power_reverberation.ReverberationSoundPowerResult`
(sound power determined in a qualified hard-walled reverberation test room,
ISO 3741:2010 precision method, accuracy grade 1) to a one-page PDF laid out
like a sound-power test sheet:

* a title and the standard-basis line naming the reverberation-room method
  (the direct method using the room equivalent absorption area, Eq. 20, or the
  comparison method using a reference sound source, Eq. 21) and the precision
  accuracy grade;
* an optional metadata header grid (client, noise source, test environment,
  instrumentation, climate, date), rendered only for the fields supplied on
  the :class:`ReportMetadata`;
* a full-width per-band table (nominal octave/one-third-octave frequency, the
  mean room sound-pressure level ``Lp`` and the band sound-power level ``LW``);
  with ``verbose=True`` the direct-method table adds the background correction
  ``K1``, the equivalent absorption area ``A`` and the Waterhouse boundary
  correction ``Cw``, and the comparison-method table adds ``K1``;
* the sound-power spectrum ``LW(f)`` drawn by the result's own ``plot(ax=...)``
  (band axis with nominal labels);
* a boxed single-number result, the A-weighted sound power level ``LWA``
  (dB re 1 pW) with the total ``LW`` and the determination method alongside;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (a sound-power emission passes at or below the limit);
* a measurement-basis strip stating the correction model (Eq. 20 or Eq. 21),
  the applied meteorological corrections and the speed of sound, plus the
  Annex F A-weighting citation, and a footer identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the ``LW(f)`` spectrum, the boxed ``LWA`` and the
flow assembly) is shared with the ISO 3744 pressure fiche and the ISO 9614
intensity fiche in :mod:`._sound_power_fiche`; this module only holds the
reverberation-room specifics (the basis line, the ``Lp``/``K1``/``A``/``Cw``
table columns and the correction strip). reportlab, matplotlib and svglib are
soft dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._sound_power_fiche import (
    FicheCopy,
    band_labels,
    d1,
    fraction_caption,
    power_statement,
    power_value_table,
    render_sound_power_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..emission.sound_power_reverberation import ReverberationSoundPowerResult

#: Band-frequency column heading of the per-band table (translated by ``t``).
_COL_FREQUENCY = "f [Hz]"
#: Mean room sound-pressure level column heading (mini-HTML for reportlab).
_COL_LP = "L<sub>p</sub> [dB]"
#: Band sound-power level column heading (mini-HTML for reportlab).
_COL_LW = "L<sub>W</sub> [dB]"


def _is_comparison(result: Any) -> bool:
    """Return ``True`` for a comparison-method result, ``False`` for direct."""
    return getattr(result, "method", "direct") == "comparison"


def _basis(result: Any, language: str = "en") -> str:
    """The standard-basis line naming the reverberation-room method and grade."""
    grade = t("precision method, accuracy grade 1", language)
    if _is_comparison(result):
        return t(
            "Determination of the sound power level from the mean sound pressure "
            "level in a qualified reverberation test room, comparison method "
            "using a reference sound source (ISO 3741:2010, {grade}).",
            language,
        ).format(grade=grade)
    return t(
        "Determination of the sound power level from the mean sound pressure "
        "level in a qualified reverberation test room, direct method using the "
        "room equivalent absorption area (ISO 3741:2010, {grade}).",
        language,
    ).format(grade=grade)


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band table (nominal frequency, Lp, LW).

    The default table is the ``f | Lp | LW`` form; ``verbose`` adds the
    background correction ``K1`` and, for the direct method, the equivalent
    absorption area ``A`` and the Waterhouse boundary correction ``Cw`` (both
    undefined, and so omitted, for the comparison method). The styling and
    octave grouping are shared with the pressure/intensity fiches via
    :func:`._sound_power_fiche.power_value_table`. Called only after the
    renderer has imported reportlab.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    lp = np.asarray(result.mean_pressure_level, dtype=np.float64)
    n = lw.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if not verbose:
        header = [t(_COL_FREQUENCY, language), _COL_LP, _COL_LW]
        widths = [58.0, 58.0, 58.0]
        rows_data = [
            [labels[i], d1(lp[i], language), d1(lw[i], language)] for i in range(n)
        ]
        return power_value_table(header, rows_data, widths, fraction)

    k1 = np.asarray(result.background_correction, dtype=np.float64)
    if _is_comparison(result):
        header = [
            t(_COL_FREQUENCY, language),
            _COL_LP,
            "K<sub>1</sub> [dB]",
            _COL_LW,
        ]
        widths = [45.0, 43.0, 43.0, 43.0]
        rows_data = [
            [labels[i], d1(lp[i], language), d1(k1[i], language), d1(lw[i], language)]
            for i in range(n)
        ]
        return power_value_table(header, rows_data, widths, fraction)

    area = np.asarray(result.absorption_area, dtype=np.float64)
    waterhouse = np.asarray(result.waterhouse_correction, dtype=np.float64)
    header = [
        t(_COL_FREQUENCY, language),
        _COL_LP,
        "K<sub>1</sub> [dB]",
        "A [m<super>2</super>]",
        "C<sub>w</sub> [dB]",
        _COL_LW,
    ]
    widths = [30.0, 29.0, 29.0, 29.0, 29.0, 28.0]
    rows_data = [
        [
            labels[i],
            d1(lp[i], language),
            d1(k1[i], language),
            d1(area[i], language),
            d1(waterhouse[i], language),
            d1(lw[i], language),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed sound-power result and its extended terms.

    Delegates the ``LWA``/``LW`` box to the shared
    :func:`._sound_power_fiche.power_statement` and appends the
    reverberation-method extended term: the determination method (direct or
    comparison). The reverberation result carries no expanded uncertainty.
    """
    statement, extended = power_statement(result, language)
    if _is_comparison(result):
        method_term = t(
            "Determination method: comparison method (reference sound source)",
            language,
        )
    else:
        method_term = t(
            "Determination method: direct method (room absorption)", language
        )
    extended.append(method_term)
    return statement, extended


def _corrections_strip(result: Any, language: str = "en") -> str:
    """The applied-corrections line for the basis strip (Eq. 20 or Eq. 21)."""
    c2 = format_number(float(result.c2), language, decimals=2)
    c = format_number(float(result.speed_of_sound), language, decimals=1)
    if _is_comparison(result):
        return t(
            "The mean room levels L<sub>p</sub>(ST) and L<sub>p</sub>(RSS) are "
            "energy averages over the microphone positions corrected for "
            "background noise (K1, ISO 3741:2010 Eq. 14); L<sub>W</sub> = "
            "L<sub>W</sub>(RSS) + (L<sub>p</sub>(ST) &#8722; L<sub>p</sub>(RSS) "
            "+ C2) (Eq. 21), so the room absorption, Waterhouse and C1 terms "
            "cancel against the reference sound source. Applied correction "
            "C2 = {c2} dB, speed of sound c = {c} m/s.",
            language,
        ).format(c2=c2, c=c)
    c1 = format_number(float(result.c1), language, decimals=2)
    return t(
        "The mean room level L<sub>p</sub> is the energy average over the "
        "microphone positions corrected for background noise (K1, "
        "ISO 3741:2010 Eq. 14); L<sub>W</sub> = L<sub>p</sub> + 10 lg(A/"
        "A<sub>0</sub>) + 4,34 (A/S) + 10 lg(1 + S c /(8 V f)) + C1 + C2 "
        "&#8722; 6, A<sub>0</sub> = 1 m<super>2</super> (Eq. 20), with the "
        "Sabine absorption area A = (55,26/c)(V/T<sub>60</sub>) and c = {c} "
        "m/s. Applied meteorological corrections C1 = {c1} dB, C2 = {c2} dB.",
        language,
    ).format(c=c, c1=c1, c2=c2)


def _a_weighting_strip(language: str = "en") -> str:
    """The A-weighting citation line (ISO 3741 Annex F)."""
    return t(
        "The A-weighted sound power level L<sub>WA</sub> combines the band "
        "levels with the A-weighting band corrections of ISO 3741:2010 Annex F "
        "(Eq. F.2, reusing the ISO 3744:2010 Annex E corrections). Levels are "
        "referenced to the reference sound power 1 pW.",
        language,
    )


def render_reverberation_power_report(
    result: ReverberationSoundPowerResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a reverberation-room sound-power determination fiche to ``path``.

    :param result: A
        :class:`~phonometry.emission.sound_power_reverberation.ReverberationSoundPowerResult`
        carrying the per-band ``sound_power_level``, the mean room
        ``mean_pressure_level`` and the A-weighted total.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the declared A-weighted sound-power limit.
    :param verbose: When ``True`` the per-band table adds the background
        correction ``K1`` and, for the direct method, the equivalent absorption
        area ``A`` and the Waterhouse boundary correction ``Cw``.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    statement, extended = _statement(result, language)
    return render_sound_power_fiche(
        result,
        path,
        copy=FicheCopy(
            title=t("Sound power determination", language),
            basis=_basis(result, language),
            caption=fraction_caption(result, language),
            statement=statement,
            extended=extended,
            basis_strips=[
                _corrections_strip(result, language),
                _a_weighting_strip(language),
            ],
        ),
        value_table=_value_table(result, verbose, language),
        metadata=metadata,
        language=language,
    )
