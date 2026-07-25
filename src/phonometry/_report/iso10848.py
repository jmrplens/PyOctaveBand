#  Copyright (c) 2026. Jose M. Requena-Plens
"""ISO 10848 laboratory flanking-transmission fiches (reportlab renderer).

Renders the three results of the laboratory flanking-transmission measurement
(:mod:`phonometry.building.flanking_transmission`, ISO 10848-1/-2:2006 and
ISO 10848-4:2010) to one-page PDF fiches:

* :class:`~phonometry.building.flanking_transmission.VibrationReductionResult`
  to a **junction characterization** report of the vibration reduction index
  ``Kij`` (ISO 10848-1:2006): the per-band ``Kij`` beside the ``Kij(f)`` curve,
  the single-number arithmetic-mean ``Kij`` over the Annex A band range and the
  bands bracketed for poor modal overlap (ISO 10848-4:2010 Clause 9), excluded
  from that mean. This report has its own compact layout (a dB junction index
  with a mean, not a shifted-reference insulation curve).
* :class:`~phonometry.building.flanking_transmission.FlankingLevelDifferenceResult`
  to a **measurement** report of the normalized flanking level difference
  ``Dn,f`` (airborne, ISO 10848-1:2006 Formula (4)), with the ISO 717-1 single
  number ``Dn,f,w (C; Ctr)``.
* :class:`~phonometry.building.flanking_transmission.FlankingImpactLevelResult`
  to a **measurement** report of the normalized flanking impact level ``Ln,f``
  (tapping machine, ISO 10848-1:2006 Formula (5)), with the ISO 717-2 single
  number ``Ln,f,w (CI)``.

The ``Dn,f`` / ``Ln,f`` basis line names the ISO 10848 part governing the
tested construction, selected by the ``part`` argument: Part 2 (lightweight
elements, default), Part 3 (lightweight elements with a junction of
substantial influence) or Part 4 (heavy, Type A elements).

The two overall flanking descriptors are shaped like a sound-insulation
quantity (a per-band curve rated by ISO 717 against a shifted reference), so
they reuse the shared two-panel insulation skeleton
(:func:`._insulation_fiche.render_insulation_fiche`); with ``verbose=True`` their
left table shows the ISO 717 evaluation per band (the reported quantity, the
shifted reference and the unfavourable deviation). The ``Kij`` fiche keeps its
own body here.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._insulation_fiche import iso717_columns_builder, render_insulation_fiche
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
)
from .iso717 import _metadata_pairs
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.flanking_transmission import (
        FlankingImpactLevelResult,
        FlankingLevelDifferenceResult,
        VibrationReductionResult,
    )
    from ..building.insulation import ImpactRatingResult, WeightedRatingResult

#: Designation of each applicable ISO 10848 part. The overall flanking
#: descriptors ``Dn,f`` / ``Ln,f`` are defined in the Part 1 frame document
#: (Formulas (4)/(5)); which part governs the measurement depends on the
#: construction under test: Part 2 (lightweight elements, junction of light
#: components), Part 3 (lightweight elements when the junction has a
#: substantial influence) or Part 4 (heavy, Type A elements).
_PART_DESIGNATIONS: dict[int, str] = {
    2: "ISO 10848-2:2006",
    3: "ISO 10848-3:2006",
    4: "ISO 10848-4:2010",
}


def _part_designation(part: int) -> str:
    """The designation of ISO 10848 part ``part``, or raise."""
    try:
        return _PART_DESIGNATIONS[int(part)]
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "'part' must be 2, 3 or 4 (the ISO 10848 part governing the "
            f"tested construction); got {part!r}."
        ) from exc


def _dnf_spec(standard: str, language: str) -> dict[str, str]:
    """Fixed labels of the ``Dn,f`` fiche for the governing ``standard``.

    The natural-language strings are pre-translated here (with the standard
    designation filled in), so the shared skeleton's re-translation of the
    composed text is a no-op.
    """
    return {
        "title": "Flanking sound insulation between rooms",
        "basis": t(
            "Normalized flanking level difference D<sub>n,f</sub> measured "
            "in accordance with {standard} (airborne excitation of the "
            "source-room element). Rating per ISO 717-1:2020.",
            language,
        ).format(standard=standard),
        "symbol": "D<sub>n,f</sub>",
        "rating_symbol": "D<sub>n,f,w</sub>",
        "ylabel": "$D_{n,f}$ [dB]",
        "statement": t(
            "Evaluation based on laboratory measurement of the normalized "
            "flanking level difference ({standard}, airborne excitation).",
            language,
        ).format(standard=standard),
    }


def _lnf_spec(standard: str, language: str) -> dict[str, str]:
    """Fixed labels of the ``Ln,f`` fiche for the governing ``standard``."""
    return {
        "title": "Flanking impact sound insulation of floors",
        "basis": t(
            "Normalized flanking impact sound pressure level L<sub>n,f</sub> "
            "measured in accordance with {standard} using the tapping "
            "machine on the source-room floor. Rating per ISO 717-2:2020.",
            language,
        ).format(standard=standard),
        "symbol": "L<sub>n,f</sub>",
        "rating_symbol": "L<sub>n,f,w</sub>",
        "ylabel": "$L_{n,f}$ [dB]",
        "statement": t(
            "Evaluation based on laboratory measurement of the normalized "
            "flanking impact level ({standard}, tapping machine on the "
            "source-room floor).",
            language,
        ).format(standard=standard),
    }


def _require_rating(
    rating: WeightedRatingResult | ImpactRatingResult | None,
) -> WeightedRatingResult | ImpactRatingResult:
    """Return the ISO 717 rating, or raise when it is absent.

    The overall flanking descriptors carry a single-number rating only for the
    ISO 717 band sets (16 one-third-octave or 5 octave bands); the ``report``
    methods guard this, so this is a defensive narrowing for a directly called
    renderer.
    """
    if rating is None:
        raise ValueError(
            "The flanking-descriptor report needs its ISO 717 single-number "
            "rating (16 one-third-octave or 5 octave bands)."
        )
    return rating


def render_flanking_level_difference_report(
    result: FlankingLevelDifferenceResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    part: int = 2,
) -> str:
    """Render a normalized flanking level difference ``Dn,f`` fiche (ISO 10848).

    :param result: The
        :class:`~phonometry.building.flanking_transmission.FlankingLevelDifferenceResult`;
        its ``rating`` (ISO 717-1) carries the per-band data the fiche draws.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band (the ``Dn,f`` value, the shifted reference and the unfavourable
        deviation) instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :param part: The ISO 10848 part governing the tested construction: ``2``
        (default), ``3`` or ``4``; it selects the basis-line designation.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``part`` is not 2, 3 or 4.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    rating = _require_rating(result.rating)
    spec = _dnf_spec(_part_designation(part), language)
    return render_insulation_fiche(
        result,
        rating,
        path,
        spec=spec,
        is_impact=False,
        curve_attr="d_n_f",
        build_columns=iso717_columns_builder(rating, False, spec["symbol"]),
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def render_flanking_impact_level_report(
    result: FlankingImpactLevelResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    part: int = 2,
) -> str:
    """Render a normalized flanking impact level ``Ln,f`` fiche (ISO 10848).

    :param result: The
        :class:`~phonometry.building.flanking_transmission.FlankingImpactLevelResult`;
        its ``rating`` (ISO 717-2) carries the per-band data the fiche draws.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band (the ``Ln,f`` value, the shifted reference and the unfavourable
        deviation) instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :param part: The ISO 10848 part governing the tested construction: ``2``
        (default), ``3`` or ``4``; it selects the basis-line designation.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``part`` is not 2, 3 or 4.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    rating = _require_rating(result.rating)
    spec = _lnf_spec(_part_designation(part), language)
    return render_insulation_fiche(
        result,
        rating,
        path,
        spec=spec,
        is_impact=True,
        curve_attr="l_n_f",
        build_columns=iso717_columns_builder(rating, True, spec["symbol"]),
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


#: Fixed labels of the vibration-reduction-index (``Kij``) junction fiche.
_KIJ_TITLE = "Junction vibration reduction index"
_KIJ_BASIS = (
    "Vibration reduction index K<sub>ij</sub> measured in accordance with "
    "ISO 10848-1:2006. Junction characterization for the EN 12354 "
    "flanking-transmission prediction."
)
_KIJ_STATEMENT = (
    "The single-number K<sub>ij</sub> is the arithmetic mean over the Annex A "
    "band range (200 Hz to 1250 Hz for one-third-octave bands, 125 Hz to "
    "1000 Hz for octave bands). Bracketed bands, where the modal overlap "
    "factor is below 0,25 (ISO 10848-4:2010 Clause 9), are excluded from the "
    "mean. The reported index relates only to the tested junction."
)


def _kij_band_type(frequencies: np.ndarray) -> str:
    """``"octave"`` when consecutive centres step by ~2, else third-octave."""
    if frequencies.size < 2:
        return "third-octave"
    ratio = float(np.median(frequencies[1:] / frequencies[:-1]))
    return "octave" if ratio > 1.6 else "third-octave"


def _kij_mean_membership(
    frequencies: np.ndarray, bracketed: np.ndarray | None
) -> tuple[np.ndarray, float, float]:
    """Per-band single-number membership plus the Annex A band range.

    A band enters the single-number mean when it lies inside the Annex A range
    (200 Hz to 1250 Hz for one-third-octave bands, 125 Hz to 1000 Hz for
    octave bands) and is not bracketed for poor modal overlap. Returns the
    boolean membership per band together with the applicable ``(low, high)``.
    """
    if _kij_band_type(frequencies) == "octave":
        low, high = 125.0, 1000.0
    else:
        low, high = 200.0, 1250.0
    in_range = (frequencies >= low) & (frequencies <= high)
    in_mean = in_range if bracketed is None else (in_range & ~bracketed)
    return in_mean, low, high


def _kij_value_table(
    frequencies: np.ndarray,
    k_ij: np.ndarray,
    bracketed: np.ndarray | None,
    in_mean: np.ndarray,
    verbose: bool,
    language: str,
) -> Any:
    """Build the per-band ``Kij`` table.

    Bands bracketed for poor modal overlap are printed with their value in
    brackets. The default table is ``f | Kij``; ``verbose`` adds a column
    stating whether the band enters the single-number mean (inside the Annex A
    range and not bracketed). Called only after the renderer has imported
    reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()
    header: list[Any] = [
        Paragraph(t("f [Hz]", language), head_style),
        Paragraph("K<sub>ij</sub> [dB]", head_style),
    ]
    widths = [24 * mm, 30 * mm]
    if verbose:
        header.append(Paragraph(t("In mean", language), head_style))
        widths = [16 * mm, 22 * mm, 18 * mm]

    rows: list[list[Any]] = [header]
    for k, fk in enumerate(frequencies):
        value = format_number(float(k_ij[k]), language, decimals=1)
        is_bracketed = bracketed is not None and bool(bracketed[k])
        if is_bracketed:
            value = f"[{value}]"
        row: list[Any] = [f"{round(float(fk))}", value]
        if verbose:
            row.append(t("yes", language) if bool(in_mean[k]) else t("no", language))
        rows.append(row)

    return band_table(rows, widths, len(frequencies))


def _kij_result_box(
    result: VibrationReductionResult,
    bracketed: np.ndarray | None,
    in_mean: np.ndarray,
    band_range: tuple[float, float],
    styles: Any,
    accent: Any,
    language: str,
) -> Any:
    """Build the boxed single-number ``Kij`` result with its extended terms."""
    low, high = band_range
    averaged = int(np.count_nonzero(in_mean))

    extended: list[str] = [
        t("Annex A band range: {low} Hz to {high} Hz", language).format(
            low=f"{int(low)}", high=f"{int(high)}"
        ),
        t("Bands averaged: {n}", language).format(n=averaged),
    ]
    if bracketed is not None:
        extended.append(
            t("Bracketed (M &lt; 0,25): {n}", language).format(
                n=int(np.count_nonzero(bracketed))
            )
        )

    if result.single_number is None:
        # Distinguish why the mean is empty: no measured band lies inside the
        # Annex A range, or in-range bands exist but every one is bracketed
        # for poor modal overlap (M < 0,25, ISO 10848-4:2010 Clause 9).
        frequencies = np.asarray(result.frequencies, dtype=np.float64)
        any_in_range = bool(np.any((frequencies >= low) & (frequencies <= high)))
        key = (
            "No single-number K<sub>ij</sub> (all in-range bands bracketed, "
            "M &lt; 0,25)"
            if any_in_range
            else "No single-number K<sub>ij</sub> (no bands in the Annex A range)"
        )
        return result_box(t(key, language), styles, accent, extended)
    statement = t("Single-number K<sub>ij</sub> = <b>{value} dB</b>", language).format(
        value=format_number(float(result.single_number), language, decimals=1)
    )
    return result_box(statement, styles, accent, extended)


def render_vibration_reduction_report(
    result: VibrationReductionResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a vibration reduction index ``Kij`` junction fiche (ISO 10848-1).

    :param result: The
        :class:`~phonometry.building.flanking_transmission.VibrationReductionResult`;
        it must carry the band centre frequencies (``result.frequencies``) that
        label the table and drive the single-number band range.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, result, statement and disclaimer).
    :param verbose: When ``True``, the per-band table adds a column stating
        whether each band enters the single-number mean.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If the result carries no band centre frequencies.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    if result.frequencies is None:
        raise ValueError(
            "The Kij report needs the band centre frequencies; build the result "
            "with vibration_reduction_index(..., frequency=...)."
        )
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    frequencies = np.asarray(result.frequencies, dtype=np.float64)
    k_ij = np.asarray(result.k_ij, dtype=np.float64)
    bracketed = (
        None
        if result.bracketed is None
        else np.asarray(result.bracketed, dtype=np.bool_)
    )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title_text = t(_KIJ_TITLE, language)
    flow: list[Any] = [
        Paragraph(title_text, title_style),
        Paragraph(t(_KIJ_BASIS, language), basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        identity, conditions = _metadata_pairs(metadata, language)
        header_pairs = identity + conditions
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    in_mean, low, high = _kij_mean_membership(frequencies, bracketed)
    caption = t("Vibration reduction index per band", language)
    left_cell = [
        Paragraph(caption, caption_style),
        _kij_value_table(frequencies, k_ij, bracketed, in_mean, verbose, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, figsize=(6.4, 5.8), language=language
    )
    flow.append(
        two_panel_body(left_cell, plot_drawing, left_width_mm=58.0, plot_width_mm=116.0)
    )
    flow.append(Spacer(1, 8))

    flow.append(
        _kij_result_box(
            result, bracketed, in_mean, (low, high), styles, accent, language
        )
    )
    statement_style = ParagraphStyle(
        "kij_statement",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=4,
    )
    flow.append(Paragraph(t(_KIJ_STATEMENT, language), statement_style))
    if bracketed is not None and bool(np.any(bracketed)):
        note_style = ParagraphStyle(
            "kij_note",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor(_MUTED_HEX),
            spaceBefore=2,
        )
        flow.append(
            Paragraph(
                t(
                    "Values in brackets are bracketed bands, excluded from the "
                    "single number.",
                    language,
                ),
                note_style,
            )
        )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title_text)
