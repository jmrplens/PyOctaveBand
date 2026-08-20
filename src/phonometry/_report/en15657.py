#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Structure-borne sound power characterization fiche (reportlab, EN 15657).

Renders a
:class:`~phonometry.building.measurement.structure_borne_power.StructureBornePowerResult`
(the structure-borne sound power a piece of building service equipment injects
into a reception plate of known mass per area ``m`` and area ``S`` from the
spatial-average plate velocity level ``Lv`` and the plate loss factor ``eta``,
EN 15657:2018 reception-plate method, Formula 14) to a one-page test sheet laid
out like a sound-power determination:

* a title and the standard-basis line naming the EN 15657 reception-plate
  method (Formula 14);
* an optional metadata header (client, source equipment, test environment,
  instrumentation, climate, date);
* a full-width per-band table (nominal octave/one-third-octave frequency, the
  spatial mean plate velocity level ``Lv`` and the injected structure-borne
  sound power level ``L_Ws``); with ``verbose=True`` the table adds the plate
  loss factor ``eta`` column;
* the ``L_Ws(f)`` spectrum drawn by the result's own ``plot(ax=...)``;
* a boxed single-number result, the band-summed total ``L_Ws`` (dB re 1 pW),
  with the plate mass per area ``m``, the plate area ``S`` and a note that the
  level is plate-specific (EN 15657 Formula 14 is not a source descriptor);
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (lower is better);
* a basis strip stating Formula 14 and its fixed reference term, and a note
  that the plate-injected level must be converted to the plate-independent
  source quantities (Formulae 15/17) before it feeds EN 12354-5.

The quantity-independent skeleton lives in :mod:`._layout`; the sound-power
body (the per-band table, the spectrum, the boxed result and the flow assembly)
is shared with the airborne sound-power fiches in :mod:`._sound_power_fiche`,
which this module reuses because the injected structure-borne power is a
power spectrum re 1 pW of the same shape. reportlab, matplotlib and svglib are
soft dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._sound_power_fiche import (
    POWER_REFERENCE,
    FicheCopy,
    band_labels,
    d1,
    level_limit_verdict,
    power_value_table,
    render_sound_power_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.measurement.structure_borne_power import StructureBornePowerResult


def _basis(language: str = "en") -> str:
    """The standard-basis line naming the EN 15657 reception-plate method."""
    return t(
        "Determination of the structure-borne sound power injected into the "
        "reception plate by building service equipment (EN 15657:2018 "
        "reception-plate method, Formula 14).",
        language,
    )


def _caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    freqs = getattr(result, "frequencies", None)
    n = np.asarray(result.power_level, dtype=np.float64).size
    if freqs is None:
        return t("Structure-borne sound power levels per band", language)
    _, fraction = band_labels(freqs, n)
    if fraction == 1:
        return t("Octave-band structure-borne sound power levels", language)
    return t("One-third-octave-band structure-borne sound power levels", language)


def _eta_cell(value: float, language: str = "en") -> str:
    """The loss-factor cell (four decimals), or an em dash when not finite."""
    if not np.isfinite(value):
        return "—"
    return format_number(float(value), language, decimals=4)


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band table (nominal frequency, Lv, L_Ws).

    The default table is the ``f | Lv | L_Ws`` form; ``verbose`` inserts the
    plate loss factor ``eta`` column. The styling and octave grouping are shared
    with the airborne sound-power fiches via
    :func:`._sound_power_fiche.power_value_table`. Called only after the renderer
    has imported reportlab.
    """
    lws = np.asarray(result.power_level, dtype=np.float64)
    lv = np.asarray(result.velocity_level, dtype=np.float64)
    n = lws.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if not verbose:
        header = [t("f [Hz]", language), "L<sub>v</sub> [dB]", "L<sub>Ws</sub> [dB]"]
        widths = [58.0, 58.0, 58.0]
        rows_data = [
            [labels[i], d1(lv[i], language), d1(lws[i], language)] for i in range(n)
        ]
        return power_value_table(header, rows_data, widths, fraction)

    eta = np.asarray(result.loss_factor, dtype=np.float64)
    header = [
        t("f [Hz]", language),
        "L<sub>v</sub> [dB]",
        "&#951;",
        "L<sub>Ws</sub> [dB]",
    ]
    widths = [45.0, 43.0, 43.0, 43.0]
    rows_data = [
        [
            labels[i],
            d1(lv[i], language),
            _eta_cell(eta[i], language),
            d1(lws[i], language),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed total ``L_Ws`` and its extended plate terms."""
    total = float(result.total_level)
    statement = t(
        "Structure-borne sound power level L<sub>Ws</sub> = <b>{value} dB</b> re {ref}",
        language,
    ).format(value=d1(total, language), ref=POWER_REFERENCE)
    extended = [
        t("Plate mass per area m = {value} kg/m<super>2</super>", language).format(
            value=format_number(float(result.mass_per_area), language, decimals=1)
        ),
        t("Reception-plate area S = {value} m<super>2</super>", language).format(
            value=format_number(float(result.area), language, decimals=2)
        ),
        t(
            "Plate-specific level (Formula 14); convert to the source quantity "
            "before EN 12354-5.",
            language,
        ),
    ]
    return statement, extended


def _relation_strip(language: str = "en") -> str:
    """The Formula 14 relation line for the basis strip."""
    return t(
        "L<sub>Ws</sub> = 10 lg(2&#960; f &#951; m S / (f<sub>0</sub> "
        "m<sub>0</sub> S<sub>0</sub>)) + L<sub>v</sub> - 60 dB re 1 pW "
        "(EN 15657:2018 Formula 14), with the references f<sub>0</sub> = 1 Hz, "
        "m<sub>0</sub> = 1 kg, S<sub>0</sub> = 1 m<super>2</super>, the plate "
        "loss factor &#951; = 2,2/(f T<sub>s</sub>) (Formula 13) and the "
        "spatial mean velocity level L<sub>v</sub> = 10 lg((1/N) &#931; "
        "10<super>L<sub>v,i</sub>/10</super>) re v<sub>0</sub> = "
        "10<super>&#8722;9</super> m/s (Formula 12).",
        language,
    )


def _conversion_strip(language: str = "en") -> str:
    """The plate-specific note directing the EN 12354-5 conversion chain."""
    return t(
        "Formula 14 gives the power injected into this particular reception "
        "plate, not a plate-independent source descriptor: convert it to the "
        "equivalent blocked force level (Formula 15) and the characteristic "
        "reception-plate power level L<sub>Ws,n</sub> (Formula 17), then apply "
        "the EN 12354-5:2009 Annex I mobility correction, before predicting an "
        "installed level.",
        language,
    )


def render_structure_borne_power_report(
    result: StructureBornePowerResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an EN 15657 structure-borne sound power fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.building.measurement.structure_borne_power.StructureBornePowerResult`
        carrying the per-band injected ``power_level`` ``L_Ws``, the plate
        ``velocity_level`` ``Lv``, the ``loss_factor`` ``eta``, the plate
        ``mass_per_area`` ``m`` and ``area`` ``S``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + basis, no header). A supplied ``requirement`` is
        read as a declared upper limit on the total ``L_Ws`` (lower is better).
    :param verbose: When ``True`` the per-band table adds the plate loss factor
        ``eta`` column.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    statement, extended = _statement(result, language)
    verdict = None
    if metadata is not None and metadata.requirement is not None:
        verdict = level_limit_verdict(
            float(result.total_level),
            metadata.requirement,
            "L<sub>Ws</sub>",
            language,
        )
    return render_sound_power_fiche(
        result,
        path,
        copy=FicheCopy(
            title=t("Structure-borne sound power determination", language),
            basis=_basis(language),
            caption=_caption(result, language),
            statement=statement,
            extended=extended,
            basis_strips=[_relation_strip(language), _conversion_strip(language)],
        ),
        value_table=_value_table(result, verbose, language),
        metadata=metadata,
        language=language,
        verdict=verdict,
    )
