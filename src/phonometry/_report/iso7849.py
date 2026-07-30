#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Sound-power-from-vibration determination fiche (reportlab, ISO/TS 7849).

Renders a
:class:`~phonometry.emission.vibration_sound_power.VibrationSoundPowerResult`
(the airborne sound power a machine radiates through the vibration of its outer
surface, ISO/TS 7849-1/-2:2009, from the surface-averaged vibratory velocity
level, the radiating area ``S`` and the radiation factor ``epsilon``) to a
one-page PDF laid out like a sound-power test sheet:

* a title and the standard-basis line naming the vibration method (the
  ISO/TS 7849-1 survey method with a fixed radiation factor ``epsilon = 1``, or
  the ISO/TS 7849-2 engineering method with a determined radiation factor);
* an optional metadata header grid (client, machine/source, test environment,
  instrumentation, climate, date), rendered only for the fields supplied on
  the :class:`ReportMetadata`;
* a full-width per-band table (nominal octave/one-third-octave frequency, the
  surface vibratory velocity level ``Lv`` and the radiated band sound-power
  level ``LW``); with ``verbose=True`` the table adds the radiation factor
  ``epsilon`` column;
* the sound-power spectrum ``LW(f)`` drawn by the result's own ``plot(ax=...)``
  (band axis with nominal labels);
* a boxed single-number result, the A-weighted sound power level ``LWA``
  (dB re 1 pW) with the total ``LW``, the radiating area ``S`` and the applied
  method alongside;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (a sound-power emission passes at or below the limit);
* a measurement-basis strip stating the sound-power relation and its fixed
  impedance term and the radiation-factor / A-weighting model, and a footer
  identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the ``LW(f)`` spectrum, the boxed ``LWA`` and the
flow assembly) is shared with the ISO 3744 pressure fiche, the ISO 3741
reverberation-room fiche and the ISO 9614 intensity fiche in
:mod:`._sound_power_fiche`; this module only holds the vibration-method
specifics (the basis line, the ``Lv``/``epsilon`` table columns and the
relation strip). reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._sound_power_fiche import (
    band_labels,
    d1,
    fraction_caption,
    power_statement,
    power_value_table,
    render_sound_power_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..emission.vibration_sound_power import VibrationSoundPowerResult


def _is_survey(result: Any) -> bool:
    """Return ``True`` for the survey method (every band uses ``epsilon = 1``)."""
    eps = np.asarray(result.radiation_factor, dtype=np.float64)
    return bool(np.all(np.abs(eps - 1.0) < 1e-9))


def _basis(result: Any, language: str = "en") -> str:
    """The standard-basis line naming the vibration method and its part."""
    if _is_survey(result):
        return t(
            "Determination of the airborne sound power level from the surface "
            "vibratory velocity with a fixed radiation factor, survey method "
            "(ISO/TS 7849-1:2009).",
            language,
        )
    return t(
        "Determination of the airborne sound power level from the surface "
        "vibratory velocity with a determined radiation factor, engineering "
        "method (ISO/TS 7849-2:2009).",
        language,
    )


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band table (nominal frequency, Lv, LW).

    The default table is the ``f | Lv | LW`` form; ``verbose`` inserts the
    radiation factor ``epsilon`` column. The styling and octave grouping are
    shared with the pressure/intensity/reverberation fiches via
    :func:`._sound_power_fiche.power_value_table`. Called only after the
    renderer has imported reportlab.
    """
    lw = np.asarray(result.sound_power_level, dtype=np.float64)
    lv = np.asarray(result.velocity_level, dtype=np.float64)
    n = lw.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if not verbose:
        header = [t("f [Hz]", language), "L<sub>v</sub> [dB]", "L<sub>W</sub> [dB]"]
        widths = [58.0, 58.0, 58.0]
        rows_data = [
            [labels[i], d1(lv[i], language), d1(lw[i], language)] for i in range(n)
        ]
        return power_value_table(header, rows_data, widths, fraction)

    eps = np.asarray(result.radiation_factor, dtype=np.float64)
    header = [
        t("f [Hz]", language),
        "L<sub>v</sub> [dB]",
        "&#949;",
        "L<sub>W</sub> [dB]",
    ]
    widths = [45.0, 43.0, 43.0, 43.0]
    rows_data = [
        [
            labels[i],
            d1(lv[i], language),
            _eps_cell(eps[i], language),
            d1(lw[i], language),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _eps_cell(value: float, language: str = "en") -> str:
    """The radiation-factor cell (three decimals), or an em dash when not finite."""
    if not np.isfinite(value):
        return "—"
    return format_number(float(value), language, decimals=3)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed sound-power result and its extended terms.

    Delegates the ``LWA``/``LW`` box to the shared
    :func:`._sound_power_fiche.power_statement` and appends the
    vibration-method extended terms: the radiating area ``S`` and the applied
    method (survey with a fixed radiation factor, or the engineering method
    with a determined radiation factor). The vibration result carries no
    expanded uncertainty.
    """
    statement, extended = power_statement(result, language)
    extended.append(
        t("Radiating area S = {value} m<super>2</super>", language).format(
            value=format_number(float(result.area), language, decimals=2)
        )
    )
    if _is_survey(result):
        method_term = t(
            "Method: survey, fixed radiation factor &#949; = 1 (upper limit)",
            language,
        )
    else:
        method_term = t(
            "Method: engineering, determined radiation factor &#949;", language
        )
    extended.append(method_term)
    return statement, extended


def _relation_strip(language: str = "en") -> str:
    """The sound-power-relation line for the basis strip (Eq. 12/15)."""
    imp = format_number(
        10.0 * math.log10(411.0 / 400.0), language, decimals=2
    )
    return t(
        "L<sub>W</sub> = L<sub>v</sub> + 10 lg(S/S<sub>0</sub>) + 10 lg(&#949;) "
        "+ 10 lg(Z<sub>c,n</sub>/Z<sub>c,0</sub>), S<sub>0</sub> = 1 m"
        "<super>2</super> (ISO/TS 7849-1:2009 Eq. 12, -2:2009 Eq. 15), with the "
        "surface velocity level L<sub>v</sub> = 20 lg(v/v<sub>0</sub>), "
        "v<sub>0</sub> = 5&#183;10<super>&#8722;8</super> m/s (Eq. 3) and the "
        "fixed impedance term 10 lg(Z<sub>c,n</sub>/Z<sub>c,0</sub>) = "
        "10 lg(411/400) = {imp} dB.",
        language,
    ).format(imp=imp)


def _factor_strip(result: Any, language: str = "en") -> str:
    """The radiation-factor / A-weighting line for the basis strip."""
    if _is_survey(result):
        factor = t(
            "The survey method fixes the radiation factor at &#949; = 1, so "
            "L<sub>W</sub> is the upper limit of the radiated power (ISO/TS "
            "7849-1:2009); the true level is lower below the coincidence "
            "frequency of plate-like parts.",
            language,
        )
    else:
        factor = t(
            "The engineering method uses a band-wise radiation factor &#949; "
            "= P/(Z<sub>c,n</sub> v<super>2</super> S) determined from an "
            "independent power measurement (ISO/TS 7849-2:2009 Eq. 8, "
            "ISO 9614).",
            language,
        )
    if getattr(result, "frequencies", None) is None:
        # A broadband level cannot be A-weighted, so make no L_WA claim.
        return t(
            "{factor} Levels are referenced to the reference sound power 1 pW.",
            language,
        ).format(factor=factor)
    return t(
        "{factor} The A-weighted sound power level L<sub>WA</sub> combines the "
        "band levels with the A-weighting band corrections of ISO 3744:2010 "
        "Annex E. Levels are referenced to the reference sound power 1 pW.",
        language,
    ).format(factor=factor)


def render_vibration_power_report(
    result: VibrationSoundPowerResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a sound-power-from-vibration determination fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.emission.vibration_sound_power.VibrationSoundPowerResult`
        carrying the per-band ``sound_power_level``, the surface
        ``velocity_level``, the ``radiation_factor`` and the ``area``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no header). A supplied
        ``requirement`` is read as the declared A-weighted sound-power limit.
    :param verbose: When ``True`` the per-band table adds the radiation factor
        ``epsilon`` column.
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
            _relation_strip(language),
            _factor_strip(result, language),
        ],
        metadata=metadata,
        language=language,
    )
