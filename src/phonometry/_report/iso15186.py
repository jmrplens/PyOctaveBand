#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 15186-1 intensity sound-insulation test reports (reportlab renderer).

Renders the two ISO 15186-1:2000 sound-intensity insulation quantities of
:mod:`phonometry.building.measurement.intensity_insulation` to the one-page laboratory
test report of the standard's Clause 8, laid out like the accredited
laboratory reports rated per ISO 717-1:

* :class:`~phonometry.building.measurement.intensity_insulation.IntensityReductionResult`
  to the intensity sound reduction index ``RI`` (Clause 3.8, Formula (7)); its
  single-number rating ``RI,w`` is the ordinary ISO 717-1 airborne rating
  evaluated on the intensity spectrum. The verbose table annexes the
  Kc-modified index ``RI,M`` (Clause 3.10, Formula (9)) beside ``RI``.
* :class:`~phonometry.building.measurement.intensity_insulation.IntensityElementNormalizedResult`
  to the element-normalized level difference ``DI,n,e`` for small building
  elements (Clause 3.9, Formula (8)), rated ``DI,n,e,w`` by the same ISO 717-1
  airborne machinery; its verbose table shows the ISO 717 evaluation per band.

Clause 8 support: the method statement appends the test-object and
measurement-surface areas ``S`` / ``Sm`` (Clause 8 g) when the result carries
them, and the optional per-band surface pressure-intensity indicator ``FpI``
and pressure-residual intensity index ``δpI0`` (Clause 8 i) are annexed as
table columns when the caller supplies them.

Both quantities are ISO 717-1-rated insulation curves, so the shared two-panel
skeleton (title and basis line, optional metadata header, per-band table
beside the measured-versus-shifted-reference curve, boxed single-number
rating, method statement, optional verdict, footer) of :mod:`._insulation_fiche`
is reused unchanged. This module only holds the ISO 15186 specifics: the fixed
labels and the intensity-method statements.

reportlab, matplotlib and svglib are soft dependencies imported lazily by the
shared renderer (reportlab and svglib ship in the ``phonometry[report]``
extra, matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
from ._insulation_fiche import (
    Column,
    iso717_columns_builder,
    render_insulation_fiche,
)
from ._layout import fmt_num

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..building.measurement.insulation import WeightedRatingResult
    from ..building.measurement.intensity_insulation import (
        IntensityElementNormalizedResult,
        IntensityReductionResult,
    )
    from .metadata import ReportMetadata

#: Fixed labels for the intensity sound reduction index fiche: title, basis
#: line, the quantity symbol used in the table header, the rating symbol of the
#: boxed result, the plot y-axis label (mathtext) and the method statement.
_SPEC: dict[str, str] = {
    "title": "Laboratory sound insulation by sound intensity",
    "basis": (
        "Intensity sound reduction index R<sub>I</sub> measured in "
        "accordance with ISO 15186-1:2000 (transmitted sound power measured "
        "directly with an intensity probe). Rating per ISO 717-1:2020."
    ),
    "symbol": "R<sub>I</sub>",
    "rating_symbol": "R<sub>I,w</sub>",
    "ylabel": "$R_I$ [dB]",
    "statement": (
        "Evaluation based on laboratory measurement results obtained by the "
        "sound-intensity method (transmitted sound power measured directly "
        "over the measurement surface)."
    ),
}

#: The content width of the left-hand table cell, in mm (the compact cell of
#: the shared two-panel insulation skeleton).
_LEFT_CELL_MM = 56.0

#: The band count identifying an octave-band ISO 717-1 rating:
#: ``band_centers`` carries 5 centres for octave bands versus 16 for
#: one-third octaves (ISO 717-1:2020 Clause 5.3).
_N_OCTAVE_BANDS = 5


def _qualification_columns(
    fpi: Sequence[float] | np.ndarray | None,
    residual_index: Sequence[float] | np.ndarray | None,
    n_bands: int,
) -> list[Column]:
    """The optional Clause 8 i) columns: ``FpI`` and ``δpI0`` per band.

    Validates the supplied arrays against the reported band count and returns
    them as table columns (symbols with units are never translated).

    :raises ValueError: If an array does not match the band count or is not
        finite.
    """
    columns: list[Column] = []
    for name, header, values in (
        ("fpi", "F<sub>pI</sub> [dB]", fpi),
        ("residual_index", "&#948;<sub>pI0</sub> [dB]", residual_index),
    ):
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (n_bands,):
            msg = (
                f"'{name}' must carry one value per reported band "
                f"({n_bands}); got shape {arr.shape}."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(arr)):
            msg = f"'{name}' must contain only finite values."
            raise ValueError(msg)
        columns.append((header, arr, 1))
    return columns


def _even_widths(n_columns: int) -> Any:
    """Even column widths spanning the compact left cell, in points."""
    from reportlab.lib.units import mm

    return [(_LEFT_CELL_MM / n_columns) * mm] * n_columns


def _areas_sentence(
    area: float | None, measurement_area: float | None, language: str
) -> str:
    """The Clause 8 g) sentence stating ``S`` and ``Sm``, or ``""``.

    Pre-translated, so the caller can append it to an already-translated
    statement (a re-translation of the composed text is a no-op).
    """
    if area is not None and measurement_area is not None:
        key = (
            "S = {s} m<super>2</super>; S<sub>m</sub> = {sm} "
            "m<super>2</super> (ISO 15186-1:2000 Clause 8)."
        )
    elif area is not None:
        key = "S = {s} m<super>2</super> (ISO 15186-1:2000 Clause 8)."
    elif measurement_area is not None:
        key = "S<sub>m</sub> = {sm} m<super>2</super> (ISO 15186-1:2000 Clause 8)."
    else:
        return ""
    return " " + t(key, language).format(
        s=fmt_num(float(area), language) if area is not None else "",
        sm=(
            fmt_num(float(measurement_area), language)
            if measurement_area is not None
            else ""
        ),
    )


def render_iso15186_report(
    result: IntensityReductionResult,
    rating: WeightedRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    fpi: Sequence[float] | np.ndarray | None = None,
    residual_index: Sequence[float] | np.ndarray | None = None,
) -> str:
    """Render an ISO 15186-1 intensity sound-insulation test report to a PDF.

    :param result: The intensity result
        (:class:`~phonometry.building.measurement.intensity_insulation.IntensityReductionResult`)
        carrying the per-band intensity sound reduction index ``RI`` and,
        optionally, the Kc-modified index ``RI,M``.
    :param rating: The ISO 717-1 rating of the reported ``RI`` (the result's
        own ``rating``); its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True`` and the modified index ``RI,M`` is present,
        the left table annexes ``RI,M`` beside the reported ``RI``.
    :param language: ``"en"`` (default) or ``"es"``.
    :param fpi: Optional per-band ``FpI`` (Clause 8 i), annexed as a column.
    :param residual_index: Optional per-band ``δpI0`` (Clause 8 i), annexed
        as a column.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``fpi`` / ``residual_index`` do not match the band
        count.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is
        not installed.
    """
    modified = result.r_i_modified
    extra = _qualification_columns(
        fpi, residual_index, int(np.asarray(result.r_i).size)
    )

    def build_columns(
        value_header: str, curve: np.ndarray, verbose: bool, language: str
    ) -> tuple[Sequence[Column], str, Any]:
        """The table: ``f | RI`` plus the optional ``RI,M`` / ``FpI`` / ``δpI0``."""
        from reportlab.lib.units import mm

        # ISO 717-1:2020 Clause 5.3 requires stating whether the rating came
        # from one-third-octave or octave bands; both the plain and the
        # verbose caption declare the set (the verbose one also names the
        # RI,M column).
        is_octave = np.asarray(rating.band_centers).size == _N_OCTAVE_BANDS
        if verbose and modified is not None:
            columns: list[Column] = [
                (value_header, curve, 1),
                (
                    t("R<sub>I,M</sub> [dB]", language),
                    np.asarray(modified, dtype=np.float64),
                    1,
                ),
                *extra,
            ]
            caption_key = (
                "Octave-band {vh} and Kc-modified index"
                if is_octave
                else "One-third-octave {vh} and Kc-modified index"
            )
            caption = t(caption_key, language).format(vh=_SPEC["symbol"])
            if extra:
                col_widths = _even_widths(len(columns) + 1)
            else:
                col_widths = [12 * mm] + [22 * mm for _ in columns]
            return columns, caption, col_widths
        caption_key = (
            "Octave-band {vh} [dB]" if is_octave else "One-third-octave {vh} [dB]"
        )
        caption = t(caption_key, language).format(vh=_SPEC["symbol"])
        if extra:
            columns = [(value_header, curve, 1), *extra]
            return columns, caption, _even_widths(len(columns) + 1)
        return [(value_header, curve, 1)], caption, None

    spec = dict(_SPEC)
    spec["statement"] = t(_SPEC["statement"], language) + _areas_sentence(
        getattr(result, "area", None),
        getattr(result, "measurement_area", None),
        language,
    )
    return render_insulation_fiche(
        result,
        rating,
        path,
        spec=spec,
        is_impact=False,
        curve_attr="r_i",
        build_columns=build_columns,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


#: Fixed labels for the element-normalized level difference fiche. The quantity
#: ``DI,n,e`` (Clause 3.9, Formula (8)) is an ISO 717-1-rated level difference
#: for small building elements, so it feeds the shared skeleton exactly like
#: the reduction-index fiche above, differing only in these labels and its
#: intensity-method statement.
_DINE_SPEC: dict[str, str] = {
    "title": "Element sound insulation by sound intensity",
    "basis": (
        "Element-normalized level difference D<sub>I,n,e</sub> measured in "
        "accordance with ISO 15186-1:2000 (transmitted sound power measured "
        "directly with an intensity probe, normalized to the reference "
        "absorption area A<sub>0</sub> = 10 m<sup>2</sup>). Rating per "
        "ISO 717-1:2020."
    ),
    "symbol": "D<sub>I,n,e</sub>",
    "rating_symbol": "D<sub>I,n,e,w</sub>",
    "ylabel": "$D_{I,n,e}$ [dB]",
    "statement": (
        "Evaluation based on laboratory measurement results obtained by the "
        "sound-intensity method (transmitted sound power measured directly "
        "over the measurement surface enclosing the element)."
    ),
}


def _element_extras_sentence(
    result: IntensityElementNormalizedResult, language: str
) -> str:
    """The Clause 8 g) sentence for the element fiche: ``Sm`` and ``N``."""
    sentence = _areas_sentence(
        None, getattr(result, "measurement_area", None), language
    )
    n = int(getattr(result, "n", 1))
    if n > 1:
        note = t(
            "DI,n,e is the per-unit value of N = {n} element units measured together.",
            language,
        ).format(n=n)
        sentence = f"{sentence} {note}" if sentence else f" {note}"
    return sentence


def render_iso15186_element_report(
    result: IntensityElementNormalizedResult,
    rating: WeightedRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    fpi: Sequence[float] | np.ndarray | None = None,
    residual_index: Sequence[float] | np.ndarray | None = None,
) -> str:
    """Render an ISO 15186-1 element-normalized insulation report to a PDF.

    :param result: The element result
        (:class:`~phonometry.building.measurement.intensity_insulation.IntensityElementNormalizedResult`)
        carrying the per-band element-normalized level difference ``DI,n,e``.
    :param rating: The ISO 717-1 rating of the reported ``DI,n,e`` (the
        result's own ``rating``); its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band (the ``DI,n,e`` value, the shifted reference and the
        unfavourable deviation) instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :param fpi: Optional per-band ``FpI`` (Clause 8 i), annexed as a column
        of the two-column form.
    :param residual_index: Optional per-band ``δpI0`` (Clause 8 i), annexed
        as a column of the two-column form.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``fpi`` / ``residual_index`` do not match the band
        count.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    # ISO 717-1:2020 Clause 5.3 requires the table caption to state the band
    # set; the rating carries 5 centres for octave bands, 16 for one-third
    # octaves.
    band_set = (
        "Octave-band"
        if np.asarray(rating.band_centers).size == _N_OCTAVE_BANDS
        else "One-third-octave"
    )
    extra = _qualification_columns(
        fpi, residual_index, int(np.asarray(result.d_i_n_e).size)
    )
    iso717_build = iso717_columns_builder(
        rating, False, _DINE_SPEC["symbol"], band_set=band_set
    )

    def build_columns(
        value_header: str, curve: np.ndarray, verbose: bool, language: str
    ) -> tuple[Sequence[Column], str, Any]:
        """The ISO 717 table; the ``FpI`` / ``δpI0`` columns replace verbose."""
        if not extra:
            return iso717_build(value_header, curve, verbose, language)
        # The compact left cell cannot hold the verbose ISO 717 evaluation
        # plus the qualification columns, so the Clause 8 i) content takes
        # precedence whenever it is supplied.
        columns: list[Column] = [(value_header, curve, 1), *extra]
        caption = t(f"{band_set} {{vh}} [dB]", language).format(vh=_DINE_SPEC["symbol"])
        return columns, caption, _even_widths(len(columns) + 1)

    spec = dict(_DINE_SPEC)
    spec["statement"] = t(_DINE_SPEC["statement"], language) + _element_extras_sentence(
        result, language
    )
    return render_insulation_fiche(
        result,
        rating,
        path,
        spec=spec,
        is_impact=False,
        curve_attr="d_i_n_e",
        build_columns=build_columns,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )
