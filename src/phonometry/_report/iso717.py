#  Copyright (c) 2026. Jose M. Requena-Plens
"""ISO 717 sound-insulation rating fiche (reportlab renderer).

Renders a :class:`~phonometry.building.insulation.WeightedRatingResult`
(airborne, ISO 717-1) or
:class:`~phonometry.building.insulation.ImpactRatingResult` (impact,
ISO 717-2) to a one-page PDF laid out like an accredited-laboratory test
report (modelled on ISO 10140-2 / ISO 16283 lab reports rated per ISO 717):

* a title and the standard-basis line (measurement standard + ISO 717 rating);
* an optional metadata header block (client, specimen, room conditions ...),
  rendered only for the fields supplied on the :class:`ReportMetadata`;
* a two-panel body with the one-third-octave table on the left and the
  measured-versus-shifted-reference curve on the right, drawn by the result's
  own ``plot(ax=...)`` so the curve is native to the library;
* a boxed single-number result ``Rw (C; Ctr) = X (a; b) dB`` (airborne) or
  ``Ln,w (CI) = X (y) dB`` (impact);
* an optional verdict row when a requirement is supplied;
* a footer identity/disclaimer block.

With ``verbose=True`` the table uses the ISO 717 Annex C columns instead
(frequency, measured value, shifted reference, unfavourable deviation).

The quantity-independent skeleton (metadata grid, figure embedding, result box,
verdict styling, footer, document build) lives in :mod:`._layout`; this module
only holds the ISO 717 specifics (labels, the one-third-octave table and the
verdict sign rule). reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import html
import math
import re
from typing import TYPE_CHECKING, Any, cast

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
    from ..building.insulation import ImpactRatingResult, WeightedRatingResult

#: Threshold below which an unfavourable deviation is shown as an em dash.
_DEVIATION_EPS = 0.05

#: Fixed 0-based y-axis tops (dB) for the embedded plot, expanded to the next
#: 10 dB if the data exceeds them, so fiches stay visually comparable.
_Y_TOP_AIRBORNE = 60.0
_Y_TOP_IMPACT = 80.0


#: Plain-text form of a valid ISO 717 single-number-quantity symbol: a leading
#: capital (optionally primed, e.g. ``R'``) followed by the subscripted rest
#: (letters, digits, commas), e.g. ``Rw``, ``R'w``, ``Dn,w``, ``DnT,w``,
#: ``Ln,w``, ``L'nT,w``.
_SYMBOL_RE = re.compile(r"^([A-Z]'?)([A-Za-z0-9,]+)$")


def _symbol_markup(symbol: str) -> str:
    """Reportlab markup of a plain-text quantity symbol (``DnT,w`` -> ``D<sub>nT,w</sub>``).

    :raises ValueError: If ``symbol`` is not a leading (optionally primed)
        capital followed by its subscript, the shape of every ISO 717-1
        Tables 1-2 / ISO 717-2 Table 1 single-number quantity.
    """
    match = _SYMBOL_RE.match(symbol)
    if match is None:
        raise ValueError(
            f"Invalid ISO 717 quantity symbol {symbol!r}; expected a leading "
            "capital letter (optionally primed) followed by its subscript, "
            "e.g. 'Rw', 'R'w', 'DnT,w', 'L'nT,w'."
        )
    stem, subscript = match.groups()
    return f"{stem}<sub>{subscript}</sub>"


def _band_symbol_markup(symbol: str) -> str:
    """Markup of the per-band quantity behind a weighted symbol.

    Strips the trailing weighted-``w`` marker (and its comma) so the value
    table is headed by the band quantity the curve actually holds
    (``DnT,w`` -> ``D<sub>nT</sub>``, ``Rw`` -> ``R``).
    """
    band = symbol
    if band.endswith(",w"):
        band = band[:-2]
    elif band.endswith("w"):
        band = band[:-1]
    if _SYMBOL_RE.match(band):
        return _symbol_markup(band)
    return band


def _labels(
    result: WeightedRatingResult | ImpactRatingResult,
    language: str = "en",
    symbol: str | None = None,
) -> tuple[str, str, str, str]:
    """Return ``(title, rating_part, statement, value_header)`` for the quantity.

    ``symbol`` selects the reported single-number quantity (``Rw``, ``R'w``,
    ``Dn,w``, ``DnT,w`` per ISO 717-1 Tables 1-2; ``Ln,w``, ``L'n,w``,
    ``L'nT,w`` per ISO 717-2 Table 1); ``None`` keeps the laboratory defaults
    ``Rw`` / ``Ln,w``. The adaptation terms print a sign only when negative,
    the style of the standard's own examples (e.g. ``41 (0; -5) dB``).
    """
    if result.quantity == "impact":
        impact = cast("ImpactRatingResult", result)
        title = t("Impact sound insulation rating", language)
        rating_part = "ISO 717-2"
        sym = _symbol_markup(symbol if symbol is not None else "Ln,w")
        statement = (
            f"{sym} (C<sub>I</sub>) = "
            f"<b>{impact.rating} ({impact.ci:d}) dB</b>"
        )
        value_header = _band_symbol_markup(symbol) if symbol else "L<sub>n</sub>"
    else:
        airborne = cast("WeightedRatingResult", result)
        title = t("Airborne sound insulation rating", language)
        rating_part = "ISO 717-1"
        sym = _symbol_markup(symbol if symbol is not None else "Rw")
        statement = (
            f"{sym} (C; C<sub>tr</sub>) = "
            f"<b>{airborne.rating} ({airborne.c:d}; {airborne.ctr:d}) dB</b>"
        )
        value_header = _band_symbol_markup(symbol) if symbol else "R"
    return title, rating_part, statement, value_header


def _metadata_pairs(
    metadata: ReportMetadata,
    language: str = "en",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build the two ordered (label, value) groups of the header grid.

    Only fields that are set are returned, so empty rows never appear.
    """

    def group(specs: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
        # Values are user-supplied free text; escape XML specials so a '&' or
        # '<' cannot break reportlab's Paragraph parser. Labels are fixed and
        # carry intentional markup (e.g. <super>), so they are left as-is.
        return [
            (label, html.escape(str(value)))
            for label, value in specs
            if value is not None
        ]

    identity = group(
        [
            (t("Client", language), metadata.client),
            (t("Mounted by", language), metadata.mounted_by),
            (
                t("Sample area S [m<super>2</super>]", language),
                fmt_meta(metadata.area, language)
                if metadata.area is not None else None,
            ),
            (t("Manufacturer", language), metadata.manufacturer),
            (t("Description", language), metadata.specimen),
            (t("Test room", language), metadata.test_room),
            (t("Date of test", language), metadata.test_date),
        ]
    )

    def num(value: float | None) -> str | None:
        # Round-trip formatting: the header grid reprints client-supplied
        # values and must not silently reduce them (1.23 m^2 stays "1.23").
        return fmt_meta(value, language) if value is not None else None

    # Per-room temperature/humidity when supplied; otherwise a single value.
    per_room_t = (
        metadata.source_temperature is not None
        or metadata.receiving_temperature is not None
    )
    per_room_rh = (
        metadata.source_relative_humidity is not None
        or metadata.receiving_relative_humidity is not None
    )
    conditions = group(
        [
            (t("Source room volume [m<super>3</super>]", language), num(metadata.source_volume)),
            (t("Source room temp. [&#176;C]", language), num(metadata.source_temperature)),
            (
                t("Source room humidity [%]", language),
                num(metadata.source_relative_humidity),
            ),
            (
                t("Receiving room volume [m<super>3</super>]", language),
                num(metadata.receiving_volume),
            ),
            (
                t("Receiving room temp. [&#176;C]", language),
                num(metadata.receiving_temperature),
            ),
            (
                t("Receiving room humidity [%]", language),
                num(metadata.receiving_relative_humidity),
            ),
            (
                t("Temperature [&#176;C]", language),
                None if per_room_t else num(metadata.temperature),
            ),
            (
                t("Relative humidity [%]", language),
                None if per_room_rh else num(metadata.relative_humidity),
            ),
            (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
            (
                t("Mass per unit area [kg/m<super>2</super>]", language),
                num(metadata.mass_per_area),
            ),
            (t("Mounting", language), metadata.mounting),
        ]
    )
    return identity, conditions


def _value_table(
    centers: np.ndarray,
    measured: np.ndarray,
    shifted: np.ndarray,
    deviations: np.ndarray,
    value_header: str,
    verbose: bool,
    language: str = "en",
) -> Any:
    """Build the left-hand one-third-octave table (accredited or Annex C).

    Called only after :func:`render_iso717_report` has imported reportlab.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    head_style = band_table_header_style()

    if verbose:
        header = [
            Paragraph(t("f [Hz]", language), head_style),
            Paragraph(
                t("Measured {vh} [dB]", language).format(vh=value_header),
                head_style,
            ),
            Paragraph(t("Shifted ref. [dB]", language), head_style),
            Paragraph(t("Unfav. dev. [dB]", language), head_style),
        ]
        col_widths = [15 * mm, 19 * mm, 18 * mm, 18 * mm]
    else:
        header = [
            Paragraph(t("Frequency f [Hz]", language), head_style),
            Paragraph(
                t("{vh} [dB]", language).format(vh=value_header), head_style
            ),
        ]
        col_widths = [28 * mm, 28 * mm]

    def d1(value: float) -> str:
        # One decimal, locale-aware separator (period for English, matching
        # fmt_num; comma for Spanish).
        return format_number(value, language, decimals=1)

    rows: list[list[Any]] = [header]
    for fk, m, r_, d in zip(centers, measured, shifted, deviations):
        if verbose:
            rows.append(
                [
                    f"{round(fk)}",
                    d1(m),
                    format_number(r_, language, decimals=0),
                    d1(d) if d > _DEVIATION_EPS else "—",
                ]
            )
        else:
            rows.append([f"{round(fk)}", d1(m)])

    n_data = len(centers)
    extra_styles: list[Any] = []
    if verbose:
        from reportlab.lib import colors

        rows.append(
            ["", "", t("sum", language), d1(float(deviations.sum()))]
        )
        extra_styles = [
            ("LINEABOVE", (0, -1), (-1, -1), 0.6, colors.HexColor(_ACCENT_HEX)),
            ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, -1), (-1, -1), 7.5),
        ]

    return band_table(rows, col_widths, n_data, extra_styles)


def _verdict(
    result: WeightedRatingResult | ImpactRatingResult,
    requirement: float,
    language: str = "en",
    symbol: str | None = None,
) -> tuple[str, bool]:
    """Return the verdict text and a PASS flag for a supplied requirement.

    Airborne ratings pass when the rating meets or exceeds the requirement;
    impact ratings pass when the rating is at or below it (lower is better).
    The verdict names the reported quantity (``symbol``, defaulting to the
    laboratory ``Rw`` / ``Ln,w``).
    """
    rating = float(result.rating)
    req_text = fmt_num(requirement, language)
    if result.quantity == "impact":
        passed = rating <= requirement
        sym = _symbol_markup(symbol if symbol is not None else "Ln,w")
        text = t("{sym} = {rating} dB, required &#8804; {req} dB", language).format(
            sym=sym, rating=result.rating, req=req_text
        )
    else:
        passed = rating >= requirement
        sym = _symbol_markup(symbol if symbol is not None else "Rw")
        text = t("{sym} = {rating} dB, required &#8805; {req} dB", language).format(
            sym=sym, rating=result.rating, req=req_text
        )
    return text, passed


def _extended_terms(
    result: WeightedRatingResult | ImpactRatingResult,
) -> list[str]:
    """Return the extended spectrum-adaptation terms the result carries, if any.

    The core rating results do not hold the enlarged-range terms, so this is
    normally empty; it only lists terms that are genuinely present on the
    object (never fabricated).
    """
    terms: list[str] = []
    airborne_specs = [
        ("c_50_3150", "C<sub>50-3150</sub>"),
        ("c_50_5000", "C<sub>50-5000</sub>"),
        ("c_100_5000", "C<sub>100-5000</sub>"),
        ("ctr_50_3150", "C<sub>tr,50-3150</sub>"),
        ("ctr_50_5000", "C<sub>tr,50-5000</sub>"),
        ("ctr_100_5000", "C<sub>tr,100-5000</sub>"),
    ]
    impact_specs = [("ci_50_2500", "C<sub>I,50-2500</sub>")]
    specs = impact_specs if result.quantity == "impact" else airborne_specs
    for attr, label in specs:
        value = getattr(result, attr, None)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if not math.isfinite(number):  # pragma: no cover - defensive
            continue
        # Sign only when negative, the style of the standard's own examples
        # (format_number also normalises a signed zero away).
        terms.append(f"{label} = {format_number(number, decimals=0)} dB")
    return terms


def render_iso717_report(
    result: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    symbol: str | None = None,
) -> str:
    """Render an ISO 717 accredited-laboratory rating fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.building.insulation.WeightedRatingResult`
        (airborne, ISO 717-1) or
        :class:`~phonometry.building.insulation.ImpactRatingResult` (impact,
        ISO 717-2) carrying the per-band ``band_centers``, ``measured`` and
        ``shifted_reference`` curves.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        prediction fiche (body + result + disclaimer, no test metadata).
    :param verbose: When ``True``, the left table uses the ISO 717 Annex C
        columns (f, measured, shifted reference, unfavourable deviation);
        otherwise the accredited two-column ``f | value`` table.
    :param symbol: The reported single-number quantity, as plain text (e.g.
        ``"R'w"``, ``"Dn,w"``, ``"DnT,w"`` per ISO 717-1 Tables 1-2;
        ``"L'n,w"``, ``"L'nT,w"`` per ISO 717-2 Table 1). ``None`` keeps the
        laboratory defaults ``Rw`` / ``Ln,w``. The symbol heads the boxed
        result, the value-table column and the verdict row, so a field
        measurement is not mislabelled as the laboratory quantity.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If ``symbol`` is not a valid quantity-symbol shape.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
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

    centers = result.band_centers
    measured = result.measured
    shifted = result.shifted_reference
    if centers is None or measured is None or shifted is None:
        raise ValueError(
            "render_iso717_report() needs 'band_centers', 'measured' and "
            "'shifted_reference' on the result."
        )
    centers = np.asarray(centers, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    shifted = np.asarray(shifted, dtype=np.float64)
    if not centers.shape == measured.shape == shifted.shape:
        raise ValueError(
            "render_iso717_report() needs 'band_centers', 'measured' and "
            "'shifted_reference' of equal length."
        )

    # Unfavourable deviation: reference above measurement (airborne) or
    # measurement above the reference (impact, the opposite sign).
    if result.quantity == "impact":
        deviations = np.maximum(measured - shifted, 0.0)
    else:
        deviations = np.maximum(shifted - measured, 0.0)

    title, rating_part, statement, value_header = _labels(result, language, symbol)

    styles, title_style, basis_style, caption_style = document_styles(accent)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t("{standard} laboratory measurement of sound insulation. Rating per {part}:2020.", language).format(
            standard=html.escape(measurement_standard), part=rating_part
        )
    else:
        basis = t("Sound insulation rating per {part}:2020.", language).format(part=rating_part)

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    # Metadata header block (only the supplied fields).
    if metadata is not None and not metadata.is_empty():
        identity, conditions = _metadata_pairs(metadata, language)
        header_pairs = identity + conditions
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    # Two-panel body: the one-third-octave table on the left (~70 mm), the
    # vector plot on the right filling the rest of the content width.
    value_table = _value_table(
        centers, measured, shifted, deviations, value_header, verbose, language
    )
    # ISO 717-1:2020 Clause 5.3 / ISO 717-2:2020 Clause 4.4 require stating
    # whether the rating came from one-third-octave or octave bands; the
    # caption declares the actual set.
    caption_key = (
        "Octave-band {vh} [dB]"
        if centers.size == 5
        else "One-third-octave {vh} [dB]"
    )
    left_cell = [
        Paragraph(
            t(caption_key, language).format(vh=value_header),
            caption_style,
        ),
        value_table,
    ]
    y_top = _Y_TOP_IMPACT if result.quantity == "impact" else _Y_TOP_AIRBORNE
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=y_top, expand_step=10.0, language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    # Boxed single-number result, optional verdict row, footer.
    flow.append(result_box(statement, styles, accent, _extended_terms(result)))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language, symbol)
        flow.extend(verdict_flow(text, passed, styles, language))
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
