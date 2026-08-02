#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 16283 field sound-insulation test report (reportlab renderer).

Renders a :class:`~phonometry.building.measurement.insulation.AirborneInsulationResult`
(field airborne, ISO 16283-1:2014),
:class:`~phonometry.building.measurement.insulation.ImpactInsulationResult` (field
impact, ISO 16283-2:2020) or
:class:`~phonometry.building.measurement.insulation.FacadeInsulationResult` (field
facade, ISO 16283-3:2016) to the one-page field test report of each
standard's Clause 14, laid out like the recommended results form (ISO
16283-1 Annex B / ISO 16283-2 Annex C) and the accredited field reports
built on it:

* a title and the standard-basis line naming the field standard, the
  reported quantity and the ISO 717 rating part;
* an optional metadata header block (client, construction description,
  room volumes, partition area, climatic conditions ...), rendered only
  for the fields supplied on the :class:`ReportMetadata`;
* a two-panel body with the one-third-octave table on the left (the
  quantity to one decimal place, Clause 12) and the field curve versus the
  shifted ISO 717 reference on the right, drawn by the rating's own
  ``plot(ax=...)`` so the curve is native to the library;
* a boxed single-number field rating ``DnT,w (C; Ctr)`` / ``R'w (C; Ctr)``
  (ISO 717-1) or ``L'nT,w (CI)`` / ``L'n,w (CI)`` (ISO 717-2);
* the mandatory statement that the evaluation is based on field
  measurement results obtained by an engineering method (Clause 14 and the
  Annex forms print it verbatim);
* an optional verdict row when a requirement is supplied (airborne passes
  at or above it, impact at or below it);
* a footer identity/disclaimer block.

With ``verbose=True`` the table shows the measurement chain instead: the
energy-average source/receiving (or impact) levels and the receiving-room
reverberation time beside the reported quantity, the per-band content the
accredited field reports annex. The shared two-panel skeleton lives in
:mod:`._insulation_fiche` (used by both this field fiche and the laboratory
ISO 10140 fiche); this module only holds the ISO 16283 specifics. reportlab,
matplotlib and svglib are soft dependencies imported lazily by the shared
renderer (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._i18n import t
from ._insulation_fiche import (
    Column,
    iso717_columns_builder,
    render_insulation_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.measurement.insulation import (
        AirborneInsulationResult,
        FacadeInsulationResult,
        ImpactInsulationResult,
        ImpactRatingResult,
        WeightedRatingResult,
    )

#: The field engineering-method statement the airborne (ISO 16283-1) and facade
#: (ISO 16283-3) reports print verbatim.
_FIELD_STATEMENT = (
    "Evaluation based on field measurement using results obtained "
    "by an engineering method."
)

#: Per-quantity fixed labels: title, basis line, the quantity symbol used in
#: the table header, the rating symbol of the boxed result, the plot y-axis
#: label (mathtext) and the field-method statement each standard's results
#: form prints verbatim.
_SPECS: dict[str, dict[str, str]] = {
    "dnt": {
        "title": "Field airborne sound insulation between rooms",
        "basis": (
            "Standardized level difference D<sub>nT</sub> measured in "
            "accordance with ISO 16283-1:2014 (field measurement). "
            "Rating per ISO 717-1:2020."
        ),
        "symbol": "D<sub>nT</sub>",
        "rating_symbol": "D<sub>nT,w</sub>",
        "ylabel": "$D_{nT}$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
    "r_prime": {
        "title": "Field airborne sound insulation between rooms",
        "basis": (
            "Apparent sound reduction index R&#8242; measured in "
            "accordance with ISO 16283-1:2014 (field measurement). "
            "Rating per ISO 717-1:2020."
        ),
        "symbol": "R&#8242;",
        "rating_symbol": "R&#8242;<sub>w</sub>",
        "ylabel": "$R'$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
    "l_n_t": {
        "title": "Field impact sound insulation of floors",
        "basis": (
            "Standardized impact sound pressure level L&#8242;<sub>nT</sub> "
            "measured in accordance with ISO 16283-2:2020 using the tapping "
            "machine. Rating per ISO 717-2:2020."
        ),
        "symbol": "L&#8242;<sub>nT</sub>",
        "rating_symbol": "L&#8242;<sub>nT,w</sub>",
        "ylabel": "$L'_{nT}$ [dB]",
        "statement": (
            "Evaluation based on field measurement results obtained by an "
            "engineering method."
        ),
    },
    "l_n": {
        "title": "Field impact sound insulation of floors",
        "basis": (
            "Normalized impact sound pressure level L&#8242;<sub>n</sub> "
            "measured in accordance with ISO 16283-2:2020 using the tapping "
            "machine. Rating per ISO 717-2:2020."
        ),
        "symbol": "L&#8242;<sub>n</sub>",
        "rating_symbol": "L&#8242;<sub>n,w</sub>",
        "ylabel": "$L'_{n}$ [dB]",
        "statement": (
            "Evaluation based on field measurement results obtained by an "
            "engineering method."
        ),
    },
}

#: The reported quantities carried by an impact field result (ISO 717-2).
_IMPACT_QUANTITIES = ("l_n_t", "l_n")


def render_iso16283_report(
    result: AirborneInsulationResult | ImpactInsulationResult,
    rating: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 16283 field sound-insulation test report to a PDF.

    :param result: The field result
        (:class:`~phonometry.building.measurement.insulation.AirborneInsulationResult`
        or :class:`~phonometry.building.measurement.insulation.ImpactInsulationResult`)
        carrying the per-band quantities and, when built by the measurement
        functions, the per-band chain (levels and reverberation times).
    :param rating: The ISO 717 rating of the reported 16-band quantity,
        already evaluated by the caller
        (:meth:`~phonometry.building.measurement.insulation.AirborneInsulationResult.report`
        validates and computes it); its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param quantity: ``"dnt"``, ``"r_prime"``, ``"l_n_t"`` or ``"l_n"``
        (validated by the caller).
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the per-band
        measurement chain (levels, reverberation time, quantity) instead of
        the two-column recommended form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is
        not installed.
    """
    spec = _SPECS[quantity]
    is_impact = quantity in _IMPACT_QUANTITIES

    def build_columns(
        value_header: str, curve: np.ndarray, verbose: bool, language: str
    ) -> tuple[Sequence[Column], str, Any]:
        """The field table: ``f | value`` or, verbose, the measurement chain."""
        from reportlab.lib.units import mm

        if not verbose:
            caption = t("One-third-octave {vh} [dB]", language).format(
                vh=spec["symbol"]
            )
            return [(value_header, curve, 1)], caption, None

        if is_impact:
            impact_result = cast("ImpactInsulationResult", result)
            chain: list[Column] = [
                ("L<sub>i</sub> [dB]",
                 np.asarray(impact_result.li, dtype=np.float64), 1),
                ("T [s]", np.asarray(impact_result.t2, dtype=np.float64), 2),
            ]
        else:
            airborne_result = cast("AirborneInsulationResult", result)
            chain = [
                ("L<sub>1</sub> [dB]",
                 np.asarray(airborne_result.l1, dtype=np.float64), 1),
                ("L<sub>2</sub> [dB]",
                 np.asarray(airborne_result.l2, dtype=np.float64), 1),
                ("T [s]", np.asarray(airborne_result.t2, dtype=np.float64), 2),
            ]
        columns = chain + [(value_header, curve, 1)]
        # Compact widths for the chain; the reported quantity (last column)
        # carries the longest header, so give it room to render on one line.
        col_widths = [10 * mm] + [
            (10 if decimals == 2 else 12) * mm for _, _, decimals in columns
        ]
        col_widths[-1] = 15 * mm
        caption = t("Per-band measurement chain", language)
        return columns, caption, col_widths

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


#: Per-quantity fixed labels of the ISO 16283-3 field facade report: title,
#: standard-basis line, the quantity symbol used in the table header, the rating
#: symbol of the boxed result and the plot y-axis label (mathtext). The facade
#: quantities reference the outdoor level 2 m in front of the facade (``D2m``),
#: so the notation is kept distinct from the ISO 16283-1 between-rooms ``DnT``.
_FACADE_TITLE = "Field facade sound insulation"
_FACADE_SPECS: dict[str, dict[str, str]] = {
    "d_2m_nt": {
        "title": _FACADE_TITLE,
        "basis": (
            "Standardized facade level difference D<sub>2m,nT</sub> measured "
            "in accordance with ISO 16283-3:2016 (field measurement). Rating "
            "per ISO 717-1:2020."
        ),
        "symbol": "D<sub>2m,nT</sub>",
        "rating_symbol": "D<sub>2m,nT,w</sub>",
        "ylabel": "$D_{2m,nT}$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
    "d_2m_n": {
        "title": _FACADE_TITLE,
        "basis": (
            "Normalized facade level difference D<sub>2m,n</sub> measured in "
            "accordance with ISO 16283-3:2016 (field measurement). Rating per "
            "ISO 717-1:2020."
        ),
        "symbol": "D<sub>2m,n</sub>",
        "rating_symbol": "D<sub>2m,n,w</sub>",
        "ylabel": "$D_{2m,n}$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
    "r_prime": {
        "title": _FACADE_TITLE,
        "basis": (
            "Apparent sound reduction index R&#8242;<sub>45</sub> measured in "
            "accordance with ISO 16283-3:2016 (field measurement, loudspeaker "
            "method). Rating per ISO 717-1:2020."
        ),
        "symbol": "R&#8242;<sub>45</sub>",
        "rating_symbol": "R&#8242;<sub>45,w</sub>",
        "ylabel": "$R'_{45}$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
    # The apparent index of the road-traffic method (ISO 16283-3:2016 Clause
    # 3.13) is a different quantity, R'tr,s, with its own -3 dB all-angle
    # correction; the fiche labels it by the result's ``method`` so a
    # road-traffic measurement is never mislabelled R'45.
    "r_prime_tr": {
        "title": _FACADE_TITLE,
        "basis": (
            "Apparent sound reduction index R&#8242;<sub>tr,s</sub> measured "
            "in accordance with ISO 16283-3:2016 (field measurement, road "
            "traffic method). Rating per ISO 717-1:2020."
        ),
        "symbol": "R&#8242;<sub>tr,s</sub>",
        "rating_symbol": "R&#8242;<sub>tr,s,w</sub>",
        "ylabel": "$R'_{\\mathrm{tr,s}}$ [dB]",
        "statement": _FIELD_STATEMENT,
    },
}


def render_iso16283_facade_report(
    result: FacadeInsulationResult,
    rating: WeightedRatingResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 16283-3 field facade sound-insulation report to a PDF.

    :param result: The field facade result
        (:class:`~phonometry.building.measurement.insulation.FacadeInsulationResult`)
        carrying the per-band facade quantities.
    :param rating: The ISO 717-1 rating of the reported facade quantity,
        already evaluated by the caller; its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param quantity: ``"d_2m_nt"`` (the standardized facade level difference),
        ``"d_2m_n"`` (the normalized facade level difference) or ``"r_prime"``
        (the apparent sound reduction index, labelled ``R'45`` or ``R'tr,s``
        following the result's ``method``); validated by the caller.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    spec_key = quantity
    if quantity == "r_prime" and getattr(result, "method", "loudspeaker") == (
        "road_traffic"
    ):
        spec_key = "r_prime_tr"
    spec = _FACADE_SPECS[spec_key]
    return render_insulation_fiche(
        result, rating, path,
        spec=spec,
        is_impact=False,
        curve_attr=quantity,
        build_columns=iso717_columns_builder(rating, False, spec["symbol"]),
        metadata=metadata,
        verbose=verbose,
        language=language,
    )
