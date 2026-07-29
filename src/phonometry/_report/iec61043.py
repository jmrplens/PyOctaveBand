#  Copyright (c) 2026. Jose M. Requena-Plens
"""IEC 61043 residual-index class-verification fiche (reportlab renderer).

Renders an
:class:`~phonometry.metrology.intensity_compliance.IntensityInstrumentComplianceResult`
to a one-page PDF laid out like an accredited electroacoustic verification
certificate. The section order is the one published accredited acoustic
calibration certificates use: identification of the item calibrated, customer,
specifications, procedure basis, results, conformity statement, signature
block. Two conventions are borrowed from real documents. The results table
follows the reduced column set such certificates use for a *one-sided*
(minimum) requirement, the requirement, the measured value and the signed
deviation, and tabulates *both* Table 2 minima beside the measured index, as
the published intensity-analyser software does when it draws a measured
residual index against the class 1 and class 2 tolerance curves together. The
measurement-basis strip closes with the scope limiter such certificates carry
when a check covers only part of a standard: the residual index is one
IEC 61043 requirement among others, so a verdict on it is not a verdict on the
whole standard.

The page holds:

* a title and the standard-basis line (measurement standard + IEC 61043
  Table 2, the table the chain was judged against);
* an optional metadata header block (client, instrument, laboratory ...),
  rendered only for the fields supplied on the :class:`ReportMetadata`;
* a measurement-basis strip naming the device kind, the microphone separation
  and the Table 2 Note 1 term applied to the printed 25 mm figures, plus the
  definition of the index;
* a two-panel body with the per-band table on the left (nominal frequency, the
  class 1 and class 2 minima, the measured index, the margin to the achieved
  class and the class that band meets) and the mask-overlay plot on the right,
  drawn by the result's own ``plot(ax=...)``;
* a boxed class statement (``Class n - COMPLIES`` with the binding margin, or a
  non-conformity statement) with the equivalent phase-mismatch range alongside;
* an optional verdict row when a required class is supplied; and
* a footer identity/disclaimer block.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the IEC 61043 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import decimal_comma, format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    fiche_paragraph,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..metrology.intensity_compliance import (
        IntensityInstrumentComplianceResult,
    )

#: Fiche prose for the ``"instrument"`` column of Table 2, and the fallback
#: for an unrecognised device kind.
_COMPLETE_INSTRUMENT = "complete instrument"

#: Table 2 column group each device kind is judged against, as fiche prose.
_DEVICE_PHRASE = {
    "probe": "probe",
    "processor": "processor",
    "instrument": _COMPLETE_INSTRUMENT,
}


def _device_phrase(device: str, language: str) -> str:
    """The localised Table 2 column-group name of a device kind."""
    return t(_DEVICE_PHRASE.get(device, _COMPLETE_INSTRUMENT), language)


def _basis(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line for the fiche."""
    table = t("IEC 61043:1993, Table 2", language)
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

    An intensity chain is an instrument, so the room/climate fields of the
    insulation fiches do not apply; the specimen field labels the probe,
    processor or complete instrument that was verified.
    """
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Probe / instrument", language), metadata.specimen),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Test facility", language), metadata.test_room),
        (t("Date of test", language), metadata.test_date),
    ]
    return [
        (label, html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _band_label(frequency: float) -> str:
    """The nominal Table 2 band label of a centre frequency (``6.3k``)."""
    if frequency >= 1000.0:
        return f"{frequency / 1000.0:g}k"
    return f"{frequency:g}"


def _class_cell(band_class: int | None) -> str:
    """The achieved-class cell of a band row (the class number, or an en dash)."""
    if band_class is None:
        return "&#8211;"
    return str(band_class)


def _band_rows(
    result: IntensityInstrumentComplianceResult, language: str = "en"
) -> list[list[Any]]:
    """Header plus per-band rows of the results table.

    Both Table 2 minima are tabulated beside the measured value, the convention
    the published intensity-analyser software uses when it draws a measured
    residual index against the class 1 and class 2 tolerance curves together.
    The remaining columns are the reduced one-sided form of an accredited
    calibration certificate (requirement, measured value, signed deviation),
    with the achieved class in place of the measurement uncertainty, which is a
    property of the caller's residual-intensity test and not of this
    classification. The margin is taken against the achieved class, the one the
    boxed result states.
    """
    header_style = band_table_header_style()
    cls = result.reference_class()
    key = f"limit_class{cls}_db"
    header = [
        fiche_paragraph(t("f [Hz]", language), header_style),
        fiche_paragraph(t("Cl. 1 [dB]", language), header_style),
        fiche_paragraph(t("Cl. 2 [dB]", language), header_style),
        fiche_paragraph(t("Meas. [dB]", language), header_style),
        fiche_paragraph(t("Margin [dB]", language), header_style),
        fiche_paragraph(t("Class", language), header_style),
    ]
    rows: list[list[Any]] = [header]
    for band in result.bands:
        margin = float(band["residual_index_db"]) - float(band[key])
        rows.append(
            [
                decimal_comma(_band_label(float(band["freq"])), language),
                format_number(float(band["limit_class1_db"]), language, decimals=1),
                format_number(float(band["limit_class2_db"]), language, decimals=1),
                format_number(float(band["residual_index_db"]), language, decimals=1),
                decimal_comma(f"{margin:+.1f}", language),
                _class_cell(band["class"]),
            ]
        )
    return rows


def _statement(
    result: IntensityInstrumentComplianceResult, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed class statement and the extended terms beside it."""
    device = _device_phrase(result.device, language)
    if result.overall_class is not None:
        margin = result.binding_margin()
        statement = t(
            "Class <b>{cls}</b> - COMPLIES &nbsp; (binding margin {margin} dB)",
            language,
        ).format(
            cls=result.overall_class,
            margin=decimal_comma(f"{margin:+.2f}", language),
        )
    else:
        margin = result.binding_margin(2)
        statement = t(
            "<b>Does not comply</b> with class 1 or class 2 &nbsp; "
            "(closest margin {margin} dB)",
            language,
        ).format(margin=decimal_comma(f"{margin:+.2f}", language))

    phase = np.asarray(result.phase_mismatch(), dtype=np.float64)
    extended = [
        t("Verified as: {device}", language).format(device=device),
        t(
            "Microphone separation d = {spacing} mm ({offset} dB on Table 2)",
            language,
        ).format(
            spacing=decimal_comma(f"{result.spacing * 1000.0:g}", language),
            offset=decimal_comma(f"{result.spacing_offset_db:+.1f}", language),
        ),
        t(
            "Equivalent phase mismatch {lo} to {hi} degrees",
            language,
        ).format(
            lo=format_number(float(phase.min()), language, decimals=3),
            hi=format_number(float(phase.max()), language, decimals=3),
        ),
    ]
    return statement, extended


def _basis_strip(
    result: IntensityInstrumentComplianceResult, language: str = "en"
) -> str:
    """The measurement-basis line: the index definition, mask and scope.

    The closing sentence is the scope limiter accredited certificates carry
    when a periodic check covers only part of a standard: the residual index is
    one requirement of IEC 61043 among many, so a verdict on it is not a
    verdict on the whole standard.
    """
    device = _device_phrase(result.device, language)
    return t(
        "Pressure-residual intensity index &#948;<sub>pI0</sub> = L<sub>p</sub> "
        "&#8722; L<sub>I,res</sub> with identical pink-noise signals on both "
        "channels, for an air density of 1.2048 kg/m<super>3</super> "
        "(IEC 61043:1993, 3.11). Judged against the Table 2 minima for a "
        "{device}, rescaled to the microphone separation in use by the Note 1 "
        "term +10 lg(x/25). It is one IEC 61043 requirement among others; "
        "full conformance to the standard does not follow from it alone.",
        language,
    ).format(device=device)


def _verdict(
    result: IntensityInstrumentComplianceResult,
    required_class: int,
    language: str = "en",
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a required class.

    A chain meets a required class ``N`` when it achieves a class at least as
    strict, i.e. an overall class index of ``N`` or lower (class 1 is the
    stricter of the two).
    """
    passed = (
        result.overall_class is not None and result.overall_class <= required_class
    )
    achieved = (
        t("none", language)
        if result.overall_class is None
        else str(result.overall_class)
    )
    text = t("Achieved class {achieved}, required class {required}", language).format(
        achieved=achieved, required=required_class
    )
    return text, passed


def render_iec61043_report(
    result: IntensityInstrumentComplianceResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an IEC 61043 class-verification fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.metrology.intensity_compliance.IntensityInstrumentComplianceResult`
        carrying the per-band verdicts and the masks they were judged against.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + disclaimer, no test metadata). A supplied
        ``required_class`` drives the verdict row.
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        intensity-compliance fiche has a single body layout, so it has no
        effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``metadata.required_class`` is neither 1 nor 2
        (IEC 61043 defines only those two classes for the residual index).
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the fiche has one two-panel body layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, _caption_style = document_styles(accent)
    title = t("Sound intensity instrument class verification", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(_basis(metadata, language), basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        header_pairs = _metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))

    strip_style = measurement_basis_style()
    flow.append(fiche_paragraph(_basis_strip(result, language), strip_style))
    flow.append(Spacer(1, 8))

    rows = _band_rows(result, language)
    widths = [13 * mm, 13 * mm, 13 * mm, 14 * mm, 15 * mm, 12 * mm]
    left_cell = band_table(rows, widths, len(rows) - 1)
    # A tall, narrow figure: the 22-row band table sets the body height, and a
    # squat plot beside it would leave a band of white across the page.
    plot_drawing = render_figure_drawing(
        result.plot, 92 * mm, y_top=None, figsize=(5.2, 7.2), language=language
    )
    flow.append(
        two_panel_body(
            left_cell, plot_drawing, left_width_mm=80.0, plot_width_mm=94.0
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if result.range_limited:
        # With no class reached there is no "stated class" for the note to
        # qualify, so it talks about the verification itself instead. Both
        # completed sentences are keys of the translation table.
        tail = (
            "this verification covers"
            if result.overall_class is None
            else "the stated class attests"
        )
        strip = (
            "The verified bands cover neither the 22 one-third-octave bands "
            "from 50 Hz to 6.3 kHz nor the 7 octave bands from 63 Hz to 4 kHz "
            f"of IEC 61043:1993 clause 6.1, so {tail} the bands listed above "
            "and not the full frequency range of the standard."
        )
        flow.append(fiche_paragraph(t(strip, language), strip_style))
    if metadata is not None and metadata.required_class is not None:
        if metadata.required_class not in (1, 2):
            raise ValueError(
                f"required_class={metadata.required_class} is not an IEC 61043 "
                "residual-index class; the standard defines classes 1 and 2."
            )
        text, passed = _verdict(result, metadata.required_class, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
