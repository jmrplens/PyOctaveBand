#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Installed structure-borne sound prediction fiche (reportlab, EN 12354-5).

Renders an
:class:`~phonometry.building.prediction.installed_structure_borne.InstalledSourceResult`
(the normalised structure-borne sound pressure level ``L_n,s`` predicted in a
receiving room from building service equipment, EN 12354-5:2009) to a one-page
**prediction** report. Unlike the measurement fiches, the reported result is an
*estimate* computed from the characteristic structure-borne source power and the
building transmission paths (Formulae 17/18); the sheet states this explicitly
and is clearly labelled a prediction, never a measurement.

The sheet carries:

* a title and a prediction-basis line naming EN 12354-5:2009;
* an optional metadata header (client, source equipment, receiving room,
  instrumentation, climate, date);
* a full-width per-band table (nominal octave/one-third-octave frequency, the
  installed structure-borne power level ``L_Ws,inst``, the normalised path SPL
  ``L_n,s,ij`` for each transmission path and the combined total ``L_n,s``);
* the per-path and total ``L_n,s(f)`` spectra drawn by the result's own
  ``plot(ax=...)``;
* a boxed single-number result, the band-summed total normalised SPL
  ``L_n,s`` (dB), with the installed power total and the path count;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (lower is better);
* a basis strip stating Formulae 18a/17 and a note that the sheet is a
  prediction from the source characterization and the element data, not a
  measurement.

The quantity-independent skeleton lives in :mod:`._layout`; the full-width
per-band table, the spectrum, the boxed result and the flow assembly are shared
with the sound-power fiches in :mod:`._sound_power_fiche` (the installed level
is a per-band level spectrum of the same shape), so this module only holds the
EN 12354-5 specifics (the prediction-basis line, the multi-path table columns,
the boxed total and the relation strip). reportlab, matplotlib and svglib are
soft dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
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
    from ..building.prediction.installed_structure_borne import InstalledSourceResult


def _basis(language: str = "en") -> str:
    """The prediction-basis line naming EN 12354-5:2009."""
    return t(
        "Predicted normalised structure-borne sound pressure level "
        "L<sub>n,s</sub> in the receiving room from building service equipment, "
        "estimated in accordance with EN 12354-5:2009 (installed power "
        "Formula 18b, per-path Formula 18a, energetic path sum Formula 17). "
        "This is a prediction from the characteristic source power and the "
        "transmission paths, not a measurement.",
        language,
    )


def _caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    freqs = getattr(result, "frequencies", None)
    total = np.atleast_1d(np.asarray(result.total_level, dtype=np.float64))
    if freqs is None:
        return t("Per-path and total normalised structure-borne SPL", language)
    _, fraction = band_labels(freqs, total.size)
    if fraction == 1:
        return t("Octave-band normalised structure-borne SPL by path", language)
    return t("One-third-octave-band normalised structure-borne SPL by path", language)


#: Maximum number of per-path columns the verbose table prints before it would
#: overflow the A4 content width; above it, only the total is shown.
_MAX_PATH_COLUMNS = 5

#: The 174 mm content width the per-band table spans.
_CONTENT_WIDTH_MM = 174.0


def _column_widths(n_columns: int) -> list[float]:
    """Even column widths spanning the content width for ``n_columns`` columns."""
    return [_CONTENT_WIDTH_MM / n_columns] * n_columns


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the full-width per-band multi-path table.

    Columns are ``f | L_Ws,inst | L_n,s,1 ... L_n,s,P | L_n,s`` (the installed
    power, one column per transmission path, and the combined total). With
    ``verbose=False`` the individual path columns are dropped and only the
    installed power and the combined total are shown, keeping the sheet compact
    when there are many paths. The styling and octave grouping are shared with
    the sound-power fiches via :func:`._sound_power_fiche.power_value_table`.
    Called only after the renderer has imported reportlab.
    """
    lws_inst = np.atleast_1d(np.asarray(result.installed_power_level, dtype=np.float64))
    total = np.atleast_1d(np.asarray(result.total_level, dtype=np.float64))
    paths = np.atleast_2d(np.asarray(result.path_levels, dtype=np.float64))
    n = total.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    header: list[str] = [t("f [Hz]", language), "L<sub>Ws,inst</sub> [dB]"]
    show_paths = verbose and paths.shape[0] <= _MAX_PATH_COLUMNS
    if show_paths:
        header.extend(f"L<sub>n,s,{k + 1}</sub> [dB]" for k in range(paths.shape[0]))
    header.append("L<sub>n,s</sub> [dB]")

    rows_data: list[list[str]] = []
    for i in range(n):
        row = [labels[i], d1(lws_inst[i], language)]
        if show_paths:
            row.extend(d1(paths[k, i], language) for k in range(paths.shape[0]))
        row.append(d1(total[i], language))
        rows_data.append(row)

    widths = _column_widths(len(header))
    return power_value_table(header, rows_data, widths, fraction)


def _statement(result: Any, language: str = "en") -> tuple[str, list[str]]:
    """The boxed total ``L_n,s`` and its extended terms."""
    overall = float(result.overall_level)
    statement = t(
        "Predicted normalised SPL L<sub>n,s</sub> = <b>{value} dB</b>", language
    ).format(value=d1(overall, language))
    lws_inst = np.atleast_1d(np.asarray(result.installed_power_level, dtype=np.float64))
    finite = lws_inst[np.isfinite(lws_inst)]
    inst_total = (
        float(10.0 * np.log10(np.sum(10.0 ** (0.1 * finite))))
        if finite.size
        else float("nan")
    )
    n_paths = np.atleast_2d(np.asarray(result.path_levels)).shape[0]
    extended = [
        t(
            "Installed power total L<sub>Ws,inst</sub> = {value} dB re {ref}",
            language,
        ).format(value=d1(inst_total, language), ref=POWER_REFERENCE),
        t("Transmission paths: {n}", language).format(n=n_paths),
        t("Prediction (EN 12354-5:2009), not a measurement.", language),
    ]
    return statement, extended


def _relation_strip(language: str = "en") -> str:
    """The Formula 18a/17 relation line for the basis strip."""
    return t(
        "L<sub>n,s,ij</sub> = L<sub>Ws,inst,i</sub> - D<sub>sa,i</sub> - "
        "R<sub>ij,ref</sub> - 10 lg(S<sub>i</sub>/S<sub>0</sub>) - "
        "10 lg(A<sub>0</sub>/4), S<sub>0</sub> = A<sub>0</sub> = 10 m"
        "<super>2</super> (EN 12354-5:2009 Formula 18a), with the installed "
        "power L<sub>Ws,inst,i</sub> = L<sub>Ws,c</sub> - D<sub>C,i</sub> "
        "(Formula 18b) and the energetic path sum L<sub>n,s</sub> = 10 lg("
        "&#931;<sub>j</sub> 10<super>L<sub>n,s,ij</sub>/10</super>) "
        "(Formula 17).",
        language,
    )


def _prediction_strip(language: str = "en") -> str:
    """The prediction disclaimer for the basis strip."""
    return t(
        "Predicted (estimated) result computed from the characteristic "
        "structure-borne source power and the building transmission paths by "
        "the EN 12354-5:2009 model; it is not a measurement. The accuracy "
        "depends on the input source and element data.",
        language,
    )


def render_installed_structure_borne_report(
    result: InstalledSourceResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an EN 12354-5 installed structure-borne prediction fiche.

    :param result: An
        :class:`~phonometry.building.prediction.installed_structure_borne.InstalledSourceResult`
        carrying the per-path ``path_levels`` ``L_n,s,ij``, the combined
        ``total_level`` ``L_n,s`` and the ``installed_power_level``
        ``L_Ws,inst``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + basis, no header). A supplied ``requirement`` is
        read as a declared upper limit on the overall ``L_n,s`` (lower is
        better).
    :param verbose: When ``True`` the per-band table adds one column per
        transmission path (up to five); otherwise only the installed power and
        the combined total are shown.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    statement, extended = _statement(result, language)
    verdict = None
    if metadata is not None and metadata.requirement is not None:
        verdict = level_limit_verdict(
            float(result.overall_level),
            metadata.requirement,
            "L<sub>n,s</sub>",
            language,
        )
    return render_sound_power_fiche(
        result,
        path,
        copy=FicheCopy(
            title=t("Predicted installed structure-borne sound", language),
            basis=_basis(language),
            caption=_caption(result, language),
            statement=statement,
            extended=extended,
            basis_strips=[_relation_strip(language), _prediction_strip(language)],
        ),
        value_table=_value_table(result, verbose, language),
        metadata=metadata,
        language=language,
        verdict=verdict,
        disclaimer="Predicted result: the values relate only to the modelled "
        "configuration, not to a tested specimen.",
    )
