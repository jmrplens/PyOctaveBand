#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 16251-1 floor-covering impact-sound-improvement fiche.

Renders a
:class:`~phonometry.building.measurement.floor_covering_improvement.FloorCoveringImprovementResult`
to a one-page PDF laid out like an accredited impact-improvement test report
(BS EN ISO 16251-1:2014, the small-mock-up laboratory method for the reduction
of transmitted impact sound by soft, locally-reacting floor coverings):

* a title and the standard-basis line naming ISO 16251-1 and the ISO 717-2
  rating of the weighted improvement;
* an optional metadata header block (client, floor-covering description,
  mounting, mass per unit area, the measured frequency range, climate ...),
  built from the supplied :class:`ReportMetadata`;
* a two-panel body with the per-band table on the left (frequency and the
  reduction of impact sound pressure level ``delta-L`` to one decimal place,
  bands at the 1,3 dB limit of measurement prefixed ``>``) and the
  ``delta-L(f)`` improvement curve on the right, drawn by the result's own
  ``plot(ax=...)`` so the curve is native to the library;
* a boxed single-number weighted improvement ``delta-Lw (CI,delta)`` (ISO 717-2,
  the ISO 16251-1 Clause 8 e) statement of results), or, when the spectrum does
  not contain the 16 rating bands, a characterisation headline;
* an optional verdict row when a requirement is supplied (a higher weighted
  improvement is better, so the result passes at or above the requirement);
* a footer identity/disclaimer block.

With ``verbose=True`` the table gains the reference-floor-with-covering column
``Ln,r = Ln,r,0 - delta-L`` (ISO 717-2:2020 Formula (1)), the derivation basis
of ``delta-Lw``, when the spectrum is exactly the 16 rating bands.

The reference floor behind ``delta-Lw`` is the standardised heavyweight floor
of ISO 717-2:2020 Table 4, fixed by the standard, so it is not a user field. The
quantity-independent skeleton lives in :mod:`._layout`; this module only holds
the ISO 16251-1 specifics. reportlab, matplotlib and svglib are soft
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
    _MUTED_HEX,
    _REPORTLAB_HINT,
    band_table,
    band_table_header_style,
    build_document,
    document_styles,
    fmt_meta,
    fmt_num,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.measurement.floor_covering_improvement import (
        FloorCoveringImprovementResult,
    )


def _metadata_pairs(
    result: FloorCoveringImprovementResult,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the header grid.

    The measured frequency range is taken from the result; the descriptive
    fields (client, floor-covering description, mounting, mass per unit area,
    climate ...) come from the :class:`ReportMetadata` when one is supplied.
    Only fields that are set are returned, so empty rows never appear.
    """
    def _md(name: str) -> Any:
        return getattr(metadata, name) if metadata is not None else None

    mass_per_area = _md("mass_per_area")
    temperature = _md("temperature")
    pressure = _md("pressure")

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    freq_range = None
    if freqs.size:
        freq_range = t("{lo} to {hi}", language).format(
            lo=round(float(freqs.min())), hi=round(float(freqs.max()))
        )

    range_label = t("Frequency range [Hz]", language)
    specs: list[tuple[str, str | None]] = [
        (t("Client", language), _md("client")),
        (t("Manufacturer", language), _md("manufacturer")),
        (t("Floor covering", language), _md("specimen")),
        (t("Mounting", language), _md("mounting")),
        (t("Mass per unit area [kg/m<super>2</super>]", language),
         fmt_meta(mass_per_area, language) if mass_per_area is not None else None),
        (range_label, freq_range),
        (t("Test facility", language), _md("test_room")),
        (t("Date of test", language), _md("test_date")),
        (t("Temperature [&#176;C]", language),
         fmt_meta(temperature, language) if temperature is not None else None),
        (t("Ambient pressure [kPa]", language),
         fmt_meta(pressure, language) if pressure is not None else None),
    ]
    # The frequency range is a formatted (label-safe) string; the remaining
    # values are user-supplied free text, so escape XML specials so a stray '&'
    # or '<' cannot break reportlab's Paragraph parser. Labels carry markup.
    return [
        (label, value if label == range_label else html.escape(str(value)))
        for label, value in specs
        if value is not None
    ]


def _reference_floor_with_covering(
    result: FloorCoveringImprovementResult,
) -> np.ndarray | None:
    """Return ``Ln,r = Ln,r,0 - delta-L`` on the 16 rating bands, or ``None``.

    The heavyweight reference floor ``Ln,r,0`` (ISO 717-2:2020 Table 4) is
    defined only on the 16 one-third-octave bands 100 Hz to 3150 Hz, so the
    verbose derivation column is available only when the spectrum is exactly
    those bands. Uses a lazy domain import (the report layer keeps its building
    imports deferred).
    """
    from ..building.measurement.floor_covering_improvement import _RATING_FREQS
    from ..building.measurement.insulation import _IMPACT_REFERENCE_FLOOR

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    rating = np.asarray(_RATING_FREQS, dtype=np.float64)
    # The derivation subtracts the Table 4 reference floor band-by-band, so it
    # needs the exact 16-band ISO 717-2 rating set (100 Hz to 3150 Hz). Match on
    # a tight absolute tolerance (no two nominal bands are within 1 Hz) so a
    # non-standard band set is rejected (the verbose column is then omitted); a
    # relative tolerance would widen to tens of Hz at 3150 Hz and misclassify.
    if freqs.shape != rating.shape or not np.allclose(
        freqs, rating, rtol=0.0, atol=0.5
    ):
        return None
    reference = np.asarray(_IMPACT_REFERENCE_FLOOR, dtype=np.float64)
    return reference - np.asarray(result.improvement, dtype=np.float64)


def _value_table(
    result: FloorCoveringImprovementResult,
    ln_r: np.ndarray | None,
    language: str = "en",
) -> Any:
    """Build the per-band ``f | delta-L`` table.

    A band at the 1,3 dB limit of measurement is reported as ``> delta-L``
    (ISO 16251-1 Formula (2)), so its value carries a ``>`` prefix. When
    ``ln_r`` is given the reference-floor-with-covering ``Ln,r = Ln,r,0 -
    delta-L`` (ISO 717-2:2020 Formula (1)) is added as a derivation column.
    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    delta_l = np.asarray(result.improvement, dtype=np.float64)
    limited = np.asarray(result.limited, dtype=bool)

    header = [
        Paragraph(t("Frequency f [Hz]", language), head_style),
        Paragraph(t("&#916;L [dB]", language), head_style),
    ]
    if ln_r is not None:
        header.append(Paragraph(t("L<sub>n,r</sub> [dB]", language), head_style))
        widths = [21 * mm, 20 * mm, 20 * mm]
    else:
        widths = [28 * mm, 28 * mm]

    rows: list[list[Any]] = [header]
    for k, fk in enumerate(freqs):
        value = format_number(float(delta_l[k]), language, decimals=1)
        # band_table receives plain strings drawn literally (not Paragraph
        # markup), so the limit marker is a literal ">", not the XML entity.
        row: list[Any] = [
            f"{round(fk)}", f"> {value}" if limited[k] else value
        ]
        if ln_r is not None:
            row.append(format_number(float(ln_r[k]), language, decimals=1))
        rows.append(row)
    return band_table(rows, widths, len(freqs))


def _basis_line(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line, naming the measurement standard when supplied."""
    standard = metadata.measurement_standard if metadata is not None else None
    if standard:
        return t(
            "{standard} laboratory measurement of the reduction of transmitted "
            "impact sound by a floor covering &#916;L on a small mock-up per "
            "ISO 16251-1:2014. Weighted improvement &#916;L<sub>w</sub> rated "
            "per ISO 717-2:2020.",
            language,
        ).format(standard=html.escape(standard))
    return t(
        "Laboratory measurement of the reduction of transmitted impact sound by "
        "a floor covering &#916;L on a small mock-up per ISO 16251-1:2014. "
        "Weighted improvement &#916;L<sub>w</sub> rated per ISO 717-2:2020.",
        language,
    )


def _is_background_limited(result: FloorCoveringImprovementResult) -> bool:
    """``True`` when any band sits at the 1,3 dB limit of measurement."""
    return bool(np.any(np.asarray(result.limited, dtype=bool)))


def _box_statement(
    result: FloorCoveringImprovementResult, language: str = "en"
) -> str:
    """The boxed single-number statement, or a characterisation headline.

    With the 16 rating bands present the box carries the ISO 16251-1 Clause 8 e)
    statement of results ``delta-Lw (CI,delta)``; otherwise (a spectrum missing
    the rating bands) it names the reported quantity and its frequency range.
    A result with background-limited bands (Formula (2), the 1,3 dB limit of
    measurement) is a lower bound, printed with a ``>=`` qualifier; a manually
    built result without ``ci_delta`` prints the bare ``delta-Lw`` rather than
    fabricating a ``(+0)`` adaptation term.
    """
    if result.delta_lw is not None:
        prefix = "&#8805; " if _is_background_limited(result) else ""
        if result.ci_delta is None:
            return f"&#916;L<sub>w</sub> = <b>{prefix}{result.delta_lw} dB</b>"
        return (
            f"&#916;L<sub>w</sub> (C<sub>I,&#916;</sub>) = "
            f"<b>{prefix}{result.delta_lw} ({result.ci_delta:+d}) dB</b>"
        )
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    lo = round(float(freqs.min())) if freqs.size else 0
    hi = round(float(freqs.max())) if freqs.size else 0
    return t(
        "Reduction of impact sound pressure level <b>&#916;L</b>, "
        "{lo} Hz to {hi} Hz",
        language,
    ).format(lo=lo, hi=hi)


def _body(
    result: FloorCoveringImprovementResult,
    verbose: bool,
    caption_style: Any,
    language: str = "en",
) -> Any:
    """The two-panel body: the per-band table beside the ``delta-L(f)`` curve.

    Called only after the renderer has imported reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    ln_r = _reference_floor_with_covering(result) if verbose else None
    caption = t(
        "One-third-octave &#916;L with the reference floor L<sub>n,r</sub>",
        language,
    ) if ln_r is not None else t("One-third-octave improvement &#916;L [dB]", language)
    left_cell: list[Any] = [
        Paragraph(caption, caption_style),
        _value_table(result, ln_r, language),
    ]

    def _plot(ax: Any = None, language: str = language) -> Any:
        return result.plot(ax=ax, language=language)

    left_width, plot_width = (66.0, 108.0) if ln_r is not None else (56.0, 118.0)
    plot_drawing = render_figure_drawing(
        _plot, plot_width * mm, y_top=None, language=language
    )
    return two_panel_body(
        left_cell, plot_drawing, left_width_mm=left_width, plot_width_mm=plot_width
    )


def _verdict_row(
    result: FloorCoveringImprovementResult,
    requirement: float,
    styles: Any,
    language: str,
) -> list[Any]:
    """The requirement verdict: a higher weighted improvement passes."""
    value = float(result.delta_lw) if result.delta_lw is not None else 0.0
    passed = value >= requirement
    text = t(
        "&#916;L<sub>w</sub> = {value} dB, required &#8805; {req} dB", language
    ).format(value=result.delta_lw, req=fmt_num(requirement, language))
    return verdict_flow(text, passed, styles, language)


def render_iso16251_report(
    result: FloorCoveringImprovementResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 16251-1 floor-covering impact-improvement fiche to a PDF.

    :param result: A
        :class:`~phonometry.building.measurement.floor_covering_improvement.FloorCoveringImprovementResult`
        carrying the ``frequencies``, the improvement ``improvement``
        (``delta-L``), the ``limited`` mask and the weighted ``delta_lw`` /
        ``ci_delta``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` renders the body
        with only the result's own frequency range and no descriptive header.
    :param verbose: When ``True``, the per-band table gains the
        reference-floor-with-covering column ``Ln,r = Ln,r,0 - delta-L``
        (ISO 717-2:2020 Formula (1)), the derivation basis of ``delta-Lw``,
        when the spectrum is exactly the 16 rating bands 100-3150 Hz.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    delta_l = np.asarray(result.improvement, dtype=np.float64)
    if freqs.shape != delta_l.shape:
        raise ValueError(
            "render_iso16251_report() needs 'frequencies' and 'improvement' of "
            "equal length."
        )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Floor-covering impact sound improvement", language)

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(_basis_line(metadata, language), basis_style),
    ]

    header_pairs = _metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(_body(result, verbose, caption_style, language))
    flow.append(Spacer(1, 8))

    flow.append(result_box(_box_statement(result, language), styles, accent))
    statement_style = ParagraphStyle(
        "iso16251_statement", parent=styles["Normal"], fontSize=8.5,
        textColor=colors.HexColor(_MUTED_HEX), spaceBefore=4,
    )
    flow.append(
        Paragraph(
            t(
                "Weighted reduction of impact sound pressure level "
                "&#916;L<sub>w</sub> rated on the 100 Hz to 3150 Hz "
                "one-third-octave bands using the ISO 717-2:2020 heavyweight "
                "reference floor (Table 4); the statement of results carries the "
                "spectrum adaptation term C<sub>I,&#916;</sub> "
                "(ISO 717-2:2020 Formula (A.4)).",
                language,
            ),
            statement_style,
        )
    )
    if _is_background_limited(result):
        flow.append(
            Paragraph(
                t(
                    "One or more bands are at the 1,3 dB limit of measurement "
                    "(ISO 16251-1:2014 Formula (2), reported as &gt; &#916;L); "
                    "the reported &#916;L<sub>w</sub> is a lower bound.",
                    language,
                ),
                statement_style,
            )
        )
    if (
        metadata is not None
        and metadata.requirement is not None
        and result.delta_lw is not None
    ):
        flow.extend(_verdict_row(result, metadata.requirement, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
