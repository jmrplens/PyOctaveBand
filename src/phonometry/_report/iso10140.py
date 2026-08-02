#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 10140 laboratory sound-insulation test report (reportlab renderer).

Renders a
:class:`~phonometry.building.measurement.lab_insulation.LabAirborneInsulationResult`
(laboratory airborne, ISO 10140-2:2010) or
:class:`~phonometry.building.measurement.lab_insulation.LabImpactInsulationResult`
(laboratory impact, ISO 10140-3:2010) to the one-page laboratory test report
each standard's Clause 9 prescribes, laid out like the accredited laboratory
reports rated per ISO 717. The shared two-panel skeleton (title and basis
line, optional metadata header, per-band table beside the
measured-versus-shifted-reference curve, boxed single-number rating, method
statement, optional verdict, footer) lives in :mod:`._insulation_fiche`; this
module only holds the ISO 10140 specifics: the fixed labels and the verbose
table that annexes the per-band equivalent sound absorption area
``A = 0,16 V / T`` (ISO 10140-4:2010, Formula (5)).

reportlab, matplotlib and svglib are soft dependencies imported lazily by the
shared renderer (reportlab and svglib ship in the ``phonometry[report]``
extra, matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
from ._insulation_fiche import Column, render_insulation_fiche
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.measurement.insulation import (
        ImpactRatingResult,
        WeightedRatingResult,
    )
    from ..building.measurement.lab_insulation import (
        LabAirborneInsulationResult,
        LabImpactInsulationResult,
    )

#: Per-quantity fixed labels: title, basis line, the quantity symbol used in
#: the table header, the rating symbol of the boxed result, the plot y-axis
#: label (mathtext) and the laboratory-method statement.
_SPECS: dict[str, dict[str, str]] = {
    "r": {
        "title": "Laboratory airborne sound insulation of a building element",
        "basis": (
            "Sound reduction index R measured in accordance with "
            "ISO 10140-2:2010 in a laboratory suppressing flanking "
            "transmission. Rating per ISO 717-1:2020."
        ),
        "symbol": "R",
        "rating_symbol": "R<sub>w</sub>",
        "ylabel": "$R$ [dB]",
        "statement": (
            "Evaluation based on laboratory measurement results obtained by a "
            "precision method (flanking transmission suppressed)."
        ),
    },
    "l_n": {
        "title": "Laboratory impact sound insulation of floors",
        "basis": (
            "Normalized impact sound pressure level L<sub>n</sub> measured in "
            "accordance with ISO 10140-3:2010 using the tapping machine. "
            "Rating per ISO 717-2:2020."
        ),
        "symbol": "L<sub>n</sub>",
        "rating_symbol": "L<sub>n,w</sub>",
        "ylabel": "$L_{n}$ [dB]",
        "statement": (
            "Evaluation based on laboratory measurement results obtained by a "
            "precision method with the standard tapping machine."
        ),
    },
}


def render_iso10140_report(
    result: LabAirborneInsulationResult | LabImpactInsulationResult,
    rating: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 10140 laboratory sound-insulation test report to a PDF.

    :param result: The laboratory result
        (:class:`~phonometry.building.measurement.lab_insulation.LabAirborneInsulationResult`
        or
        :class:`~phonometry.building.measurement.lab_insulation.LabImpactInsulationResult`)
        carrying the per-band quantity and the equivalent absorption area.
    :param rating: The ISO 717 rating of the reported quantity (the result's
        own ``rating``); its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param quantity: ``"r"`` (airborne) or ``"l_n"`` (impact); validated by
        the caller.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table annexes the per-band
        equivalent absorption area beside the reported quantity.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is
        not installed.
    """
    spec = _SPECS[quantity]
    is_impact = quantity == "l_n"

    def build_columns(
        value_header: str, curve: np.ndarray, verbose: bool, language: str
    ) -> tuple[Sequence[Column], str, Any]:
        """The laboratory table: ``f | value`` or, verbose, ``f | A | value``."""
        from reportlab.lib.units import mm

        if verbose:
            absorption = np.asarray(result.absorption, dtype=np.float64)
            columns: list[Column] = [
                (t("A [m<super>2</super>]", language), absorption, 1),
                (value_header, curve, 1),
            ]
            caption = t("Per-band quantity and absorption area", language)
            col_widths = [12 * mm] + [22 * mm for _ in columns]
            return columns, caption, col_widths
        # ISO 717-1:2020 Clause 5.3 / ISO 717-2:2020 Clause 4.4 require
        # stating whether the rating came from one-third-octave or octave
        # bands; the caption declares the set.
        caption_key = (
            "Octave-band {vh} [dB]"
            if np.asarray(rating.band_centers).size == 5
            else "One-third-octave {vh} [dB]"
        )
        caption = t(caption_key, language).format(vh=spec["symbol"])
        return [(value_header, curve, 1)], caption, None

    return render_insulation_fiche(
        result, rating, path,
        spec=spec,
        is_impact=is_impact,
        curve_attr=quantity,
        build_columns=build_columns,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )
