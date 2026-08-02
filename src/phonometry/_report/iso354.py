#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 354 reverberation-room sound-absorption fiche (reportlab renderer).

Renders a
:class:`~phonometry.materials.absorbers.sound_absorption.SoundAbsorptionMeasurement` to a
one-page PDF laid out like an accredited reverberation-room absorption test
report (ISO 354:2003):

* a title and the standard-basis line;
* an optional metadata header block (client, specimen, area ``S``, room volume
  ``V``, speed of sound ``c``, mounting, climate ...), built from the supplied
  :class:`ReportMetadata` fields together with the physical conditions carried
  on the result;
* a two-panel body with the one-third-octave ``alpha_s`` table on the left and
  the ``alpha_s`` curve on the right, drawn by the result's own ``plot(ax=...)``
  so the curve is native to the library;
* a boxed characterisation headline (ISO 354 has no single-number rating);
* a footer identity/disclaimer block.

With ``verbose=True`` the two-panel body is replaced by a full-width detail
table that also lists the reverberation times ``T1``/``T2`` and the equivalent
sound absorption areas ``A1``/``A2``, with the ``alpha_s`` curve stacked below.

ISO 354 is a characterisation, so this fiche carries no pass/fail verdict and no
weighted rating; the weighted coefficient ``alpha_w`` is an ISO 11654 quantity
rendered by :mod:`.iso11654`.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 354 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import numpy as np

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
    from ..materials.absorbers.sound_absorption import SoundAbsorptionMeasurement


def _a2(value: float, language: str = "en") -> str:
    """Two decimals (``alpha_s``), locale-aware separator."""
    return format_number(value, language, decimals=2)


def _a1(value: float, language: str = "en") -> str:
    """One decimal (absorption areas ``A``, ISO 354 Clause 8.3), locale-aware."""
    return format_number(value, language, decimals=1)


def _metadata_pairs(
    result: SoundAbsorptionMeasurement,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the absorption header grid.

    The physical test conditions (specimen area ``S``, room volume ``V``, speed
    of sound ``c``, temperature and humidity) are taken from the result, which
    is authoritative because it drove the Sabine inversion; the descriptive
    fields (client, specimen, mounting, room, date, pressure) come from the
    :class:`ReportMetadata` when one is supplied. Only fields that are set are
    returned, so empty rows never appear.
    """
    client = metadata.client if metadata is not None else None
    manufacturer = metadata.manufacturer if metadata is not None else None
    specimen = metadata.specimen if metadata is not None else None
    mounting = metadata.mounting if metadata is not None else None
    test_room = metadata.test_room if metadata is not None else None
    test_date = metadata.test_date if metadata is not None else None
    pressure = metadata.pressure if metadata is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), client),
        (t("Manufacturer", language), manufacturer),
        (t("Description", language), specimen),
        (t("Sample area S [m<super>2</super>]", language),
         fmt_meta(result.area, language)),
        (t("Room volume V [m<super>3</super>]", language),
         fmt_meta(result.volume, language)),
        (t("Speed of sound c [m/s]", language),
         fmt_meta(result.speed_of_sound, language)),
        (t("Mounting", language), mounting),
        (t("Test room", language), test_room),
        (t("Date of test", language), test_date),
        (t("Temperature [&#176;C]", language),
         fmt_meta(result.temperature, language)),
        (t("Relative humidity [%]", language),
         fmt_meta(result.humidity, language)
         if result.humidity is not None else None),
        (t("Ambient pressure [kPa]", language),
         fmt_meta(pressure, language) if pressure is not None else None),
    ]
    # Values are user-supplied free text; escape XML specials so a '&' or '<'
    # cannot break reportlab's Paragraph parser. Labels carry intentional markup.
    return [
        (label, html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _alpha_table(
    freqs: np.ndarray, alpha_s: np.ndarray, language: str = "en"
) -> Any:
    """Build the compact two-column ``f | alpha_s`` table (accredited default)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()
    header = [
        Paragraph(t("Frequency f [Hz]", language), head_style),
        Paragraph("&#945;<sub>s</sub>", head_style),
    ]
    rows: list[list[Any]] = [header]
    for fk, a_s in zip(freqs, alpha_s):
        rows.append([f"{round(fk)}", _a2(a_s, language)])
    return band_table(rows, [28 * mm, 28 * mm], len(freqs))


def _detail_table(
    result: SoundAbsorptionMeasurement, language: str = "en"
) -> Any:
    """Build the verbose table ``f | T1 | T2 | A1 | A2 | alpha_s`` (~102 mm wide)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()
    header = [
        Paragraph(t("f [Hz]", language), head_style),
        Paragraph("T<sub>1</sub> [s]", head_style),
        Paragraph("T<sub>2</sub> [s]", head_style),
        Paragraph("A<sub>1</sub> [m<super>2</super>]", head_style),
        Paragraph("A<sub>2</sub> [m<super>2</super>]", head_style),
        Paragraph("&#945;<sub>s</sub>", head_style),
    ]
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    t1 = np.asarray(result.t_empty, dtype=np.float64)
    t2 = np.asarray(result.t_specimen, dtype=np.float64)
    a1 = np.asarray(result.absorption_area_empty, dtype=np.float64)
    a2 = np.asarray(result.absorption_area_with_specimen, dtype=np.float64)
    alpha_s = np.asarray(result.alpha_s, dtype=np.float64)
    rows: list[list[Any]] = [header]
    for fk, t1k, t2k, a1k, a2k, ask in zip(freqs, t1, t2, a1, a2, alpha_s):
        rows.append(
            [
                f"{round(fk)}",
                _a2(t1k, language),
                _a2(t2k, language),
                _a1(a1k, language),
                _a1(a2k, language),
                _a2(ask, language),
            ]
        )
    col_widths = [15 * mm, 17 * mm, 17 * mm, 18 * mm, 18 * mm, 17 * mm]
    return band_table(rows, col_widths, len(freqs))


def _statement(result: SoundAbsorptionMeasurement, language: str = "en") -> str:
    """The boxed characterisation headline (ISO 354 has no single number)."""
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    lo = round(float(freqs.min())) if freqs.size else 0
    hi = round(float(freqs.max())) if freqs.size else 0
    return t(
        "Sound absorption coefficient <b>&#945;<sub>s</sub></b>, "
        "{lo} Hz to {hi} Hz",
        language,
    ).format(lo=lo, hi=hi)


def render_iso354_report(
    result: SoundAbsorptionMeasurement,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 354 sound-absorption test-report fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.absorbers.sound_absorption.SoundAbsorptionMeasurement`
        carrying the one-third-octave ``frequencies`` and ``alpha_s`` (and, for
        ``verbose``, ``t_empty``/``t_specimen`` and the absorption areas).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        with the result's own physical conditions and no descriptive header
        fields. The ``requirement`` field is ignored (ISO 354 has no verdict).
    :param verbose: When ``True``, the body is a full-width detail table
        (``f | T1 | T2 | A1 | A2 | alpha_s``) with the ``alpha_s`` curve stacked
        below; otherwise the two-panel ``f | alpha_s`` table beside the curve.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    alpha_s = np.asarray(result.alpha_s, dtype=np.float64)
    if freqs.shape != alpha_s.shape:
        raise ValueError(
            "render_iso354_report() needs 'frequencies' and 'alpha_s' of "
            "equal length."
        )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Sound absorption measurement", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} reverberation-room measurement of sound absorption "
            "per ISO 354:2003.",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Reverberation-room measurement of sound absorption per "
            "ISO 354:2003.",
            language,
        )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    from reportlab.lib.units import mm

    if verbose:
        # A wider detail table with the T1/T2/A1/A2 columns beside a narrower
        # plot, keeping the fiche to one page.
        caption = t(
            "One-third-octave &#945;<sub>s</sub> with reverberation times "
            "and absorption areas",
            language,
        )
        left_cell = [Paragraph(caption, caption_style), _detail_table(
            result, language
        )]
        plot_drawing = render_figure_drawing(
            result.plot, 70 * mm, y_top=None, language=language
        )
        flow.append(
            two_panel_body(
                left_cell, plot_drawing, left_width_mm=102.0, plot_width_mm=72.0
            )
        )
    else:
        caption = t("One-third-octave &#945;<sub>s</sub>", language)
        left_cell = [Paragraph(caption, caption_style), _alpha_table(
            freqs, alpha_s, language
        )]
        plot_drawing = render_figure_drawing(
            result.plot, 116 * mm, y_top=None, language=language
        )
        flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    # ISO 354 is a characterisation: a headline label, no single-number rating
    # and no pass/fail verdict.
    flow.append(result_box(_statement(result, language), styles, accent))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
