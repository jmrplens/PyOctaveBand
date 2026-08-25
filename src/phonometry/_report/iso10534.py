#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 10534-2 impedance-tube sound-absorption and impedance fiche.

Renders an
:class:`~phonometry.materials.absorbers.impedance_tube.ImpedanceTubeResult` to a one-page
PDF laid out like an accredited normal-incidence impedance-tube test report
(BS EN ISO 10534-2:2001, two-microphone transfer-function method):

* a title and the standard-basis line;
* an optional metadata header block (client, specimen, tube diameter ``d``,
  microphone spacing ``s``, the measured frequency range, mounting, climate
  ...), built from the supplied :class:`ReportMetadata`;
* a two-panel body with the per-frequency table on the left (frequency, the
  normal-incidence absorption coefficient ``alpha`` and the real/imaginary
  parts of the normalised surface impedance ``z = Z / (rho c0)``, plus the
  reflection-factor magnitude ``|r|`` when ``verbose``) and the ``alpha(f)``
  curve on the right, drawn by the result's own ``plot(ax=...)`` so the curve
  is native to the library;
* a boxed characterisation headline (ISO 10534-2 has no single-number rating);
* a footer identity/disclaimer block.

ISO 10534-2 is a characterisation, so this fiche carries no pass/fail verdict
and no weighted rating; the random-incidence weighted coefficient ``alpha_w``
is an ISO 11654 quantity measured in a reverberation room (ISO 354), rendered by
:mod:`.iso11654` / :mod:`.iso354`, and is not comparable to the normal-incidence
coefficient reported here.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 10534-2 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import require_equal_shapes
from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    fmt_meta,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Table

    from ..materials.absorbers.impedance_tube import ImpedanceTubeResult

#: Printable label for each accepted ``ReportMetadata.tube_shape`` value.
_SHAPE_LABELS = {
    "circular": "Circular",
    "rectangular": "Rectangular",
    "square": "Square",
}


def _a2(value: float, language: str = "en") -> str:
    """Two decimals (``alpha``, ``|r|`` and the normalised impedance parts)."""
    return format_number(value, language, decimals=2)


def _metadata_pairs(
    result: ImpedanceTubeResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the impedance-tube header grid.

    The measured frequency range is taken from the result (the frequency vector
    that carries every reported quantity); the descriptive and geometric fields
    (client, specimen, tube diameter ``d``, microphone spacing ``s``, mounting,
    climate ...) come from the :class:`ReportMetadata` when one is supplied. The
    tube diameter and microphone spacing are stored in metres and printed in
    millimetres. Only fields that are set are returned, so empty rows never
    appear.
    """
    md = metadata if metadata is not None else ReportMetadata()
    client = md.client
    manufacturer = md.manufacturer
    specimen = md.specimen
    tube_diameter = md.tube_diameter
    mic_spacing = md.mic_spacing
    tube_shape = md.tube_shape
    mounting = md.mounting
    test_room = md.test_room
    test_date = md.test_date
    temperature = md.temperature
    pressure = md.pressure

    freqs = np.asarray(result.frequency, dtype=np.float64)
    freq_range = None
    if freqs.size:
        freq_range = t("{lo} to {hi}", language).format(
            lo=round(float(freqs.min())), hi=round(float(freqs.max()))
        )

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), client),
        (t("Manufacturer", language), manufacturer),
        (t("Description", language), specimen),
        (
            t("Tube diameter d [mm]", language),
            fmt_meta(tube_diameter * 1e3, language)
            if tube_diameter is not None
            else None,
        ),
        (
            t("Tube cross-section", language),
            t(_SHAPE_LABELS[tube_shape], language) if tube_shape is not None else None,
        ),
        (
            t("Microphone spacing s [mm]", language),
            fmt_meta(mic_spacing * 1e3, language) if mic_spacing is not None else None,
        ),
        (t("Frequency range [Hz]", language), freq_range),
        (t("Mounting", language), mounting),
        (t("Test facility", language), test_room),
        (t("Date of test", language), test_date),
        (
            t("Temperature [&#176;C]", language),
            fmt_meta(temperature, language) if temperature is not None else None,
        ),
        (
            t("Ambient pressure [kPa]", language),
            fmt_meta(pressure, language) if pressure is not None else None,
        ),
    ]
    # The frequency range is a formatted (label-safe) string; the remaining
    # values are user-supplied free text, so escape XML specials so a stray '&'
    # or '<' cannot break reportlab's Paragraph parser. Labels carry markup.
    return [
        (
            label,
            value
            if label == t("Frequency range [Hz]", language)
            else html.escape(str(value)),
        )
        for label, value in specs
        if value is not None
    ]


def _value_table(
    result: ImpedanceTubeResult, verbose: bool, language: str = "en"
) -> Table:
    """Build the per-frequency value table (~102 mm wide).

    The default table lists ``f | alpha | Re z | Im z`` (the normal-incidence
    absorption coefficient and the real/imaginary parts of the normalised
    surface impedance ``z = Z / (rho c0)``); ``verbose`` inserts the
    reflection-factor magnitude ``|r|`` column.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    head_style = band_table_header_style()
    freqs = np.asarray(result.frequency, dtype=np.float64)
    alpha = np.asarray(result.absorption, dtype=np.float64)
    z = np.asarray(result.normalized_impedance, dtype=np.complex128)
    r_mag = np.abs(np.asarray(result.reflection, dtype=np.complex128))

    if verbose:
        header = [
            fiche_paragraph(t("f [Hz]", language), head_style),
            fiche_paragraph("&#945;", head_style),
            fiche_paragraph("|r|", head_style),
            fiche_paragraph("Re z", head_style),
            fiche_paragraph("Im z", head_style),
        ]
        rows: list[list[Any]] = [header]
        for fk, ak, rk, zk in zip(freqs, alpha, r_mag, z, strict=True):
            rows.append(
                [
                    f"{round(fk)}",
                    _a2(ak, language),
                    _a2(rk, language),
                    _a2(zk.real, language),
                    _a2(zk.imag, language),
                ]
            )
        col_widths = [16 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm]
    else:
        header = [
            fiche_paragraph(t("f [Hz]", language), head_style),
            fiche_paragraph("&#945;", head_style),
            fiche_paragraph("Re z", head_style),
            fiche_paragraph("Im z", head_style),
        ]
        rows = [header]
        for fk, ak, zk in zip(freqs, alpha, z, strict=True):
            rows.append(
                [
                    f"{round(fk)}",
                    _a2(ak, language),
                    _a2(zk.real, language),
                    _a2(zk.imag, language),
                ]
            )
        col_widths = [21 * mm, 23 * mm, 23 * mm, 23 * mm]
    return band_table(rows, col_widths, len(freqs), band_centres=freqs)


def _statement(result: ImpedanceTubeResult, language: str = "en") -> str:
    """The boxed characterisation headline (ISO 10534-2 has no single number)."""
    freqs = np.asarray(result.frequency, dtype=np.float64)
    lo = round(float(freqs.min())) if freqs.size else 0
    hi = round(float(freqs.max())) if freqs.size else 0
    return t(
        "Normal-incidence sound absorption coefficient <b>&#945;</b>, "
        "{lo} Hz to {hi} Hz",
        language,
    ).format(lo=lo, hi=hi)


def _basis_line(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line, naming the measurement standard when supplied."""
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        return t(
            "{standard} normal-incidence measurement of sound absorption and "
            "surface impedance in an impedance tube by the two-microphone "
            "transfer-function method per ISO 10534-2:2001.",
            language,
        ).format(standard=html.escape(measurement_standard))
    return t(
        "Normal-incidence measurement of sound absorption and surface "
        "impedance in an impedance tube by the two-microphone "
        "transfer-function method per ISO 10534-2:2001.",
        language,
    )


def _caption(verbose: bool, language: str = "en") -> str:
    """The left-panel caption, listing the columns present for this fiche."""
    if verbose:
        return t(
            "Normal-incidence &#945;, |r| and normalised surface impedance z",
            language,
        )
    return t("Normal-incidence &#945; and normalised surface impedance z", language)


def _body(
    result: ImpedanceTubeResult,
    verbose: bool,
    caption_style: ParagraphStyle,
    language: str = "en",
) -> Table:
    """The two-panel body: the per-frequency table beside the ``alpha(f)`` curve.

    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph

    left_cell = [
        fiche_paragraph(_caption(verbose, language), caption_style),
        _value_table(result, verbose, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 82 * mm, y_top=None, language=language
    )
    return two_panel_body(
        left_cell, plot_drawing, left_width_mm=90.0, plot_width_mm=84.0
    )


def render_iso10534_report(
    result: ImpedanceTubeResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 10534-2 impedance-tube test-report fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.materials.absorbers.impedance_tube.ImpedanceTubeResult`
        carrying the ``frequency`` vector, the normal-incidence ``absorption``,
        the complex ``reflection`` factor and the ``normalized_impedance``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        with only the result's own frequency range and no descriptive header
        fields. The ``requirement`` field is ignored (ISO 10534-2 has no
        verdict).
    :param verbose: When ``True``, the value table inserts the reflection-factor
        magnitude ``|r|`` column beside the absorption and the normalised
        impedance real/imaginary parts.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    # Guard a manually constructed result: ImpedanceTubeResult is a plain
    # frozen dataclass, so every per-frequency array the table and the figure
    # consume has to be checked here (a short one would otherwise cut the
    # fiche table).
    freqs = np.asarray(result.frequency, dtype=np.float64)
    alpha = np.asarray(result.absorption, dtype=np.float64)
    reflection = np.asarray(result.reflection, dtype=np.complex128)
    z = np.asarray(result.normalized_impedance, dtype=np.complex128)
    require_equal_shapes(
        "ImpedanceTubeResult.report",
        {
            "frequency": freqs.shape,
            "absorption": alpha.shape,
            "reflection": reflection.shape,
            "normalized_impedance": z.shape,
        },
        "frequency",
    )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Impedance-tube sound absorption and impedance", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(_basis_line(metadata, language), basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(_body(result, verbose, caption_style, language))
    flow.append(Spacer(1, 8))

    # ISO 10534-2 is a characterisation: a headline label, no single-number
    # rating and no pass/fail verdict.
    flow.append(result_box(_statement(result, language), styles, accent))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
