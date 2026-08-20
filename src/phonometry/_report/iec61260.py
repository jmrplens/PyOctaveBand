#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 61260-1 filter-class-compliance fiche (reportlab renderer).

Renders a
:class:`~phonometry.filters.compliance.FilterComplianceResult` to a one-page
PDF laid out like an accredited electroacoustic type-test report:

* a title and the standard-basis line (measurement standard + the IEC 61260
  edition whose Table 1 the bank was checked against);
* an optional metadata header block (client, instrument, laboratory ...),
  rendered only for the fields supplied on the :class:`ReportMetadata`, plus a
  short caption noting the bandwidth designator and sampling rate;
* a two-panel body with a compact per-band classification table on the left
  (mid-band frequency, achieved class and the binding margin to the overall
  class) and the mask-overlay plot on the right, drawn by the result's own
  ``plot(ax=...)`` so the curve is native to the library;
* a boxed class-compliance result (``Class n - COMPLIES`` with the binding
  margin, or a non-compliance statement);
* an optional verdict row when a required class is supplied; and
* a footer identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the IEC 61260 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import decimal_comma, format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    document_styles,
    footer_flow,
    grid_table,
    metrics_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)

if TYPE_CHECKING:
    from ..filters.compliance import FilterComplianceResult
    from .metadata import ReportMetadata


def _binding_margin(result: FilterComplianceResult, cls: int) -> float:
    """Smallest per-band margin to class ``cls`` (the binding margin)."""
    key = f"margin_class{cls}_db"
    return min(float(band[key]) for band in result.bands)


def _basis(metadata: ReportMetadata | None, edition: str, language: str = "en") -> str:
    """The standard-basis line for the fiche."""
    table = (
        t("IEC 61260-1:2014, Table 1", language)
        if edition == "2014"
        else t("IEC 61260:1995 / ANSI S1.11-2004, Table 1", language)
    )
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        return t("{standard} type test. Conformance to {table}.", language).format(
            standard=html.escape(measurement_standard), table=table
        )
    return t("Conformance to {table}.", language).format(table=table)


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    Only fields that are set are returned. A filter bank is an instrument, so
    the room/climate fields of the insulation fiche do not apply; the specimen
    field labels the tested filter or analyser.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Filter / instrument", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test facility", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _fraction_label(fraction: int, language: str = "en") -> str:
    """Human bandwidth-designator label (``1/1-octave``, ``1/3-octave``, ...)."""
    if fraction == 1:
        return t("1/1-octave", language)
    return t("1/{fraction}-octave", language).format(fraction=fraction)


def _band_label(exact_freq: float, fraction: int) -> str:
    """The nominal band label of an exact mid-band frequency (e.g. ``125 Hz``).

    Both editions identify the filters of a bank by their nominal mid-band
    frequencies (IEC 61260-1:2014 5.5; IEC 61260:1995 4.2), not by the exact
    base-ten values behind them, so the per-band table is labelled with the
    nominal frequency (125 Hz, not the exact 125.89... Hz).
    """
    from ..filters.frequencies import _nominal_freq_for_band

    nominal = _nominal_freq_for_band(exact_freq, float(fraction))
    if nominal >= 1000.0:
        return f"{nominal / 1000.0:g} kHz"
    return f"{nominal:g} Hz"


def _metric_rows(
    result: FilterComplianceResult, language: str = "en"
) -> list[tuple[str, str]]:
    """Per-band rows: nominal mid-band frequency vs achieved class and margin."""
    cls = result.reference_class()
    key = f"margin_class{cls}_db"
    rows: list[tuple[str, str]] = []
    for band in result.bands:
        band_class = band["class"]
        class_text = (
            t("none", language)
            if band_class is None
            else t("Class {n}", language).format(n=band_class)
        )
        margin = float(band[key])
        freq = decimal_comma(
            _band_label(float(band["freq"]), result.fraction), language
        )
        margin_text = decimal_comma(f"{margin:+.2f}", language)
        rows.append((freq, f"{class_text} ({margin_text} dB)"))
    return rows


def _statement(result: FilterComplianceResult, language: str = "en") -> str:
    """The boxed class-compliance statement with the binding margin."""
    if result.overall_class is not None:
        margin = _binding_margin(result, result.overall_class)
        return t(
            "Class <b>{cls}</b> - COMPLIES &nbsp; (margin {margin} dB)", language
        ).format(
            cls=result.overall_class,
            margin=decimal_comma(f"{margin:+.2f}", language),
        )
    classes = "/".join(str(c) for c in result.available_classes())
    margin = _binding_margin(result, result.reference_class())
    return t(
        "<b>Does not comply</b> with class {classes} &nbsp; (closest margin {margin} dB)",
        language,
    ).format(
        classes=classes,
        margin=decimal_comma(f"{margin:+.2f}", language),
    )


def _verdict(
    result: FilterComplianceResult, required_class: int, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a required class.

    A bank meets a required class ``N`` when it achieves a class at least as
    strict, i.e. an overall class index of ``N`` or lower (class 0 is the
    strictest).
    """
    passed = result.overall_class is not None and result.overall_class <= required_class
    achieved = (
        t("none", language)
        if result.overall_class is None
        else str(result.overall_class)
    )
    text = t("Achieved class {achieved}, required class {required}", language).format(
        achieved=achieved, required=required_class
    )
    return text, passed


def render_iec61260_report(
    result: FilterComplianceResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an IEC 61260-1 filter-class-compliance fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.filters.compliance.FilterComplianceResult`
        carrying the per-band verdicts and the data to redraw the binding
        band's relative attenuation.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        prediction fiche (body + result + disclaimer, no test metadata). A
        supplied ``required_class`` drives the verdict row.
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        filter-compliance fiche has a single body layout, so it has no effect.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``metadata.required_class`` does not exist in the
        edition the result was verified against (class 0 exists only in the
        1995 edition).
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the fiche has one two-panel body layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Filter class compliance", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(_basis(metadata, result.edition, language), basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        header_pairs = _metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))

    # Measurement-basis strip: the bandwidth designator and sampling rate, plus
    # the reference-attenuation convention the margins are read against.
    basis_strip_style = ParagraphStyle(
        "fiche_iec61260_basis",
        parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )
    basis_strip_key = (
        "{fraction} bank, sampling rate f<sub>s</sub> = {fs} Hz; relative attenuation referenced to the mid-band level (IEC 61260-1 Formula 8)."
        if result.edition == "2014"
        else "{fraction} bank, sampling rate f<sub>s</sub> = {fs} Hz; relative attenuation referenced to the mid-band level (IEC 61260:1995, 3.13 Note)."
    )
    flow.append(
        fiche_paragraph(
            t(basis_strip_key, language).format(
                fraction=_fraction_label(result.fraction, language),
                fs=format_number(result.fs, language, decimals=0),
            ),
            basis_strip_style,
        )
    )
    flow.append(Spacer(1, 8))

    # Two-panel body: the per-band classification table on the left, the
    # mask-overlay plot (self-scaling dB axis) on the right.
    left_cell = [
        fiche_paragraph(t("Per-band classification", language), caption_style),
        metrics_table(_metric_rows(result, language)),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    if getattr(result, "range_limited", False):
        # The multirate verifier cannot exercise the stop-band mask beyond a
        # band's processing Nyquist (no decimated signal energy exists there),
        # so the stated class attests the verified range and says so.
        flow.append(
            fiche_paragraph(
                t(
                    "Stop-band limits verified up to each band's processing Nyquist frequency; the multirate anti-aliasing leaves no signal energy beyond it, but the Table 1 limits there are not demonstrated, so the stated class attests the verified frequency range.",
                    language,
                ),
                basis_strip_style,
            )
        )
    if metadata is not None and metadata.required_class is not None:
        if metadata.required_class not in result.available_classes():
            raise ValueError(
                f"required_class={metadata.required_class} does not exist in "
                f"the {result.edition} edition this result was verified "
                f"against (available classes: {result.available_classes()}); "
                "class 0 requires edition='1995'."
            )
        text, passed = _verdict(result, metadata.required_class, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
