#  Copyright (c) 2026. Jose Manuel Requena Plens
"""IEC 60268-5 loudspeaker-characteristics fiche (reportlab renderer).

Renders a
:class:`~phonometry.electroacoustics.loudspeaker.LoudspeakerCharacteristics` to
a one-page PDF laid out like a loudspeaker rated-characteristics data sheet,
following IEC 60268-5:2003+A1:2007:

* a title and the standard-basis line (IEC 60268-5, graphs to IEC 60263:1982);
* an optional metadata header (client, manufacturer, model, room conditions);
* a rated-characteristics table beside the on-axis frequency response, the
  response drawn with a shaded tolerance band and the effective-frequency-range
  markers, to the IEC 60263 25 dB-per-decade scale proportion (clause 2);
* a full-width row of the impedance modulus, the total-harmonic-distortion and
  the polar-directivity panels for the data supplied, the polar diagram to the
  IEC 60263 25 dB reference-circle convention (clause 3);
* a boxed characteristic-sensitivity / effective-range result;
* an optional sensitivity verdict when a requirement is supplied; and
* a footer identity/disclaimer block.

The quantity-independent skeleton (metadata grid, table, result box, verdict
styling, footer, document build) lives in :mod:`._layout`; this module holds the
IEC 60268-5 specifics (labels, the rated-characteristics table and the
IEC 60263 characteristic graphs). reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
import os
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MATPLOTLIB_HINT,
    _REPORTLAB_HINT,
    _SVGLIB_HINT,
    build_document,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    metrics_table,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..electroacoustics.loudspeaker import LoudspeakerCharacteristics

#: IEC 60263 clause 2 scale proportion: one frequency decade equals this many
#: decibels on the ordinate (the "square" 25 dB-per-decade grid).
_DB_PER_DECADE = 25.0

_OHM = "&#937;"  # reportlab entity for the ohm sign


def _basis(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line: IEC 60268-5 with the IEC 60263 graph convention."""
    standard = metadata.measurement_standard if metadata is not None else None
    if standard:
        return t(
            "{standard} measurement of loudspeaker rated characteristics per "
            "IEC 60268-5:2003+A1:2007; frequency and polar graphs to "
            "IEC 60263:1982.",
            language,
        ).format(standard=html.escape(standard))
    return t(
        "Loudspeaker rated characteristics per IEC 60268-5:2003+A1:2007; "
        "frequency and polar graphs to IEC 60263:1982.",
        language,
    )


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Ordered (label, value) header pairs for the supplied loudspeaker fields."""

    def num(value: float | None) -> str | None:
        return fmt_num(value, language) if value is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Manufacturer", language), metadata.manufacturer),
        (t("Description", language), metadata.specimen),
        (t("Test room", language), metadata.test_room),
        (t("Mounting", language), metadata.mounting),
        (t("Date of test", language), metadata.test_date),
        (t("Temperature [&#176;C]", language), num(metadata.temperature)),
        (t("Relative humidity [%]", language), num(metadata.relative_humidity)),
        (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
    ]
    return [
        (label, html.escape(str(value))) for label, value in specs if value is not None
    ]


def _freq(value: float, language: str = "en") -> str:
    """A frequency rounded to the nearest hertz, locale-formatted."""
    return format_number(round(float(value)), language, decimals=0)


def _range_text(lo: float, hi: float, language: str = "en") -> str:
    """A frequency range ``lo - hi Hz`` with an en dash."""
    return f"{_freq(lo, language)}&#8211;{_freq(hi, language)} Hz"


def _rated_rows(
    result: LoudspeakerCharacteristics, language: str = "en"
) -> list[tuple[str, str]]:
    """The rated-characteristics table rows (only the quantities that exist)."""
    rows: list[tuple[str, str]] = [
        (
            t("Rated impedance", language),
            f"{fmt_num(result.rated_impedance, language)} {_OHM}",
        ),
        (
            t("Characteristic sensitivity", language),
            f"{format_number(result.sensitivity_level_db, language, decimals=1)} dB",
        ),
        (
            t("Effective frequency range", language),
            _range_text(*result.effective_range, language=language),
        ),
    ]
    if result.rated_frequency_range is not None:
        rows.append(
            (
                t("Rated frequency range", language),
                _range_text(*result.rated_frequency_range, language=language),
            )
        )
    if result.resonance_frequency is not None:
        rows.append(
            (
                t("Resonance frequency", language),
                f"{_freq(result.resonance_frequency, language)} Hz",
            )
        )
    if result.rated_noise_power is not None:
        rows.append(
            (
                t("Rated noise power", language),
                f"{fmt_num(result.rated_noise_power, language)} W",
            )
        )
    if result.rated_sinusoidal_power is not None:
        rows.append(
            (
                t("Rated sinusoidal power", language),
                f"{fmt_num(result.rated_sinusoidal_power, language)} W",
            )
        )
    min_z = result.minimum_impedance
    if min_z is not None:
        rows.append(
            (
                t("Minimum impedance", language),
                f"{fmt_num(min_z, language)} {_OHM}",
            )
        )
    if result.directivity_index_db is not None:
        label = t("Directivity index", language)
        if result.polar_frequency is not None:
            label = t("Directivity index at {freq} Hz", language).format(
                freq=_freq(result.polar_frequency, language)
            )
        rows.append(
            (
                label,
                f"{format_number(result.directivity_index_db, language, decimals=1)} dB",
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# IEC 60263 characteristic graphs
# --------------------------------------------------------------------------- #
def _drawing_from_figure(fig: Any, target_width: float, language: str = "en") -> Any:
    """Convert a matplotlib figure to a scaled, vector reportlab ``Drawing``.

    Mirrors :func:`phonometry._report._layout.render_figure_drawing` (SVG with
    text rasterised to vector paths, converted by svglib) but keeps the caller's
    own multi-panel figure and axes untouched, so the IEC 60263 scale
    proportions set on each axes survive.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_MATPLOTLIB_HINT) from exc
    try:
        from svglib.svglib import svg2rlg
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_SVGLIB_HINT) from exc
    from .._i18n import localize_axes

    for ax in fig.axes:
        localize_axes(ax, language)
    svg_fd, svg_path = tempfile.mkstemp(suffix=".svg")
    os.close(svg_fd)
    try:
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            fig.savefig(svg_path, format="svg", bbox_inches="tight")
        drawing = svg2rlg(svg_path)
    finally:
        if os.path.exists(svg_path):
            os.remove(svg_path)
    if drawing is None or not drawing.width:
        raise ValueError("Could not convert the report plot to vector graphics.")
    scale = target_width / drawing.width
    drawing.scale(scale, scale)
    drawing.width = drawing.width * scale
    drawing.height = drawing.height * scale
    drawing.hAlign = "CENTER"
    return drawing


def _response_drawing(
    result: LoudspeakerCharacteristics, target_width: float, language: str = "en"
) -> Any:
    """On-axis response with the tolerance band and effective-range markers.

    Drawn by the shared :func:`phonometry._plot.electroacoustics` panel renderer
    (the single source of truth also used by ``.plot()``), then stretched to the
    IEC 60263 clause 2 proportion: one frequency decade equals 25 dB on the
    ordinate (:data:`_DB_PER_DECADE`).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from .._plot.electroacoustics import _draw_loudspeaker_response

    f = result.frequencies
    fig = Figure(figsize=(5.6, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    _draw_loudspeaker_response(result, ax, language=language)
    decades = np.log10(float(np.max(f)) / float(np.min(f)))
    from .._plot.electroacoustics import _RESPONSE_SPAN_LSP

    ax.set_box_aspect(_RESPONSE_SPAN_LSP / (_DB_PER_DECADE * decades))
    fig.tight_layout()
    return _drawing_from_figure(fig, target_width, language)


def _secondary_drawing(
    result: LoudspeakerCharacteristics, target_width: float, language: str = "en"
) -> Any | None:
    """Impedance, THD and polar-directivity panels for the data supplied.

    The panels are drawn by the shared
    :func:`phonometry._plot.electroacoustics` renderers, so the fiche and the
    ``.plot()`` data sheet never diverge.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from .._plot.electroacoustics import (
        _draw_datasheet_polar,
        _draw_impedance,
        _draw_loudspeaker_thd,
    )

    has_imp = result.impedance_frequencies is not None
    has_thd = result.thd_frequencies is not None
    has_polar = result.polar_angles_deg is not None
    panels = int(has_imp) + int(has_thd) + int(has_polar)
    if panels == 0:
        return None

    fig = Figure(figsize=(5.6 * panels / 2.4 + 0.8, 2.6))
    FigureCanvasAgg(fig)
    idx = 1
    if has_imp:
        ax = fig.add_subplot(1, panels, idx)
        _draw_impedance(result, ax, language=language)
        _thin_freq_ticklabels(ax)
        idx += 1
    if has_thd:
        ax = fig.add_subplot(1, panels, idx)
        _draw_loudspeaker_thd(result, ax, language=language)
        _thin_freq_ticklabels(ax)
        idx += 1
    if has_polar:
        _draw_datasheet_polar(
            result,
            fig.add_subplot(1, panels, idx, projection="polar"),
            language=language,
        )
        idx += 1
    fig.tight_layout()
    return _drawing_from_figure(fig, target_width, language)


def _thin_freq_ticklabels(ax: Any, keep_every: int = 2) -> None:
    """Blank alternate labelled frequency ticks on a narrow report panel.

    The shared ``.plot()`` panel drawers label every nominal octave centre
    (``format_frequency_axis``), which reads well on a full-size ``.plot()``
    panel but crowds the compressed multi-panel row of the PDF fiche. This keeps
    every ``keep_every``-th label (and the tick positions untouched) so the
    narrow panels stay legible without re-implementing any drawing.
    """
    from matplotlib.ticker import FixedFormatter

    formatter = ax.xaxis.get_major_formatter()
    labels = list(getattr(formatter, "seq", []))
    if len(labels) <= 6:
        return
    ax.xaxis.set_major_formatter(
        FixedFormatter(
            [lab if i % keep_every == 0 else "" for i, lab in enumerate(labels)]
        )
    )


def _statement(result: LoudspeakerCharacteristics, language: str = "en") -> str:
    """The boxed headline: characteristic sensitivity and effective range."""
    return t(
        "Characteristic sensitivity <b>{level} dB</b> (1 W / 1 m); "
        "effective frequency range <b>{range}</b>",
        language,
    ).format(
        level=format_number(result.sensitivity_level_db, language, decimals=1),
        range=_range_text(*result.effective_range, language=language),
    )


def _verdict(
    result: LoudspeakerCharacteristics, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Sensitivity verdict: the characteristic sensitivity level must reach it."""
    level = result.sensitivity_level_db
    passed = level >= requirement
    text = t(
        "Characteristic sensitivity {level} dB, required &#8805; {req} dB",
        language,
    ).format(
        level=format_number(level, language, decimals=1),
        req=fmt_num(requirement, language),
    )
    return text, passed


def render_iec60268_5_report(
    result: LoudspeakerCharacteristics,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    language: str = "en",
) -> str:
    """Render an IEC 60268-5 loudspeaker-characteristics fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.electroacoustics.loudspeaker.LoudspeakerCharacteristics`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity and, through ``requirement``, a characteristic sensitivity
        level the verdict row compares against.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figures, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Loudspeaker characteristics", language)

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(_basis(metadata, language), basis_style),
    ]

    if metadata is not None and not metadata.is_empty():
        pairs = _metadata_pairs(metadata, language)
        if pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(t("Rated characteristics", language), caption_style),
        metrics_table(_rated_rows(result, language), col_widths=[33 * mm, 23 * mm]),
    ]
    response = _response_drawing(result, 116 * mm, language)
    flow.append(two_panel_body(left_cell, response))

    secondary = _secondary_drawing(result, 174 * mm, language)
    if secondary is not None:
        flow.append(Spacer(1, 6))
        flow.append(secondary)
    flow.append(Spacer(1, 8))

    flow.append(result_box(_statement(result, language), styles, accent))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
