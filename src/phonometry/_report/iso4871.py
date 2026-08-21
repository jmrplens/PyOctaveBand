#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 4871:1996 noise-emission-declaration fiche (reportlab renderer).

Renders a
:class:`~phonometry.emission.declaration.NoiseEmissionDeclaration` to a one-page
PDF laid out like a machinery noise-emission declaration, reproducing the
declaration format of ISO 4871:1996 Annex B:

* a title and the standard-basis line (ISO 4871:1996 plus the cited basic
  emission standard);
* the machine identification and operating conditions (clause 5 a / 5 c),
  printed as the declaration-table header row exactly as ISO 4871 Annex B lays
  it out;
* the declared dual- or single-number table across the operating-mode columns
  following the Annex B layouts: the measured A-weighted sound power level
  ``L_WA`` and its uncertainty ``K_WA`` for the dual-number form (Annex B.2),
  or the derived declared value ``L_WAd = L_WA + K_WA`` (the resulting upper
  value, clause 3.15) for the single-number form (Annex B.1), plus the
  emission sound pressure level ``L_pA`` at a work station when it is declared;
* the noise-test-code and basic-standards footnote (clause 5 b) and the ISO 4871
  upper-boundary note;
* a verification verdict table when a verification measurement is supplied
  (clause 6.2: verified when ``L_1 <= L_WAd`` for the single-number form, or
  ``L_1 <= L_WA + K_WA`` -- the separately rounded declared values -- for the
  dual-number form); and
* a footer identity/disclaimer block.

Like :mod:`.broadcast` and :mod:`.annex16_epnl` this uses a stacked table layout
rather than the narrow two-panel one: an ISO 4871 declaration is a table
document, not a spectrum. The quantity-independent skeleton lives in
:mod:`._layout`; this module only holds the ISO 4871 specifics. reportlab is a
soft dependency imported lazily (it ships in the ``phonometry[report]`` extra)
and guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    compliance_table,
    document_styles,
    fmt_num,
    footer_flow,
)

if TYPE_CHECKING:
    from ..emission.declaration import NoiseEmissionDeclaration
    from .metadata import ReportMetadata

#: En dash printed wherever the declaration carries no value for a cell.
_NOT_DECLARED = "&#8211;"

#: Widest declaration the mode columns can print. The 100 mm left over after
#: the 74 mm label column divides into 16.7 mm at six modes, which still holds
#: a three-digit level with its padding; at seven the column is narrower than
#: the number and the table starts stacking digits. Mirrors the cap
#: :data:`~phonometry._report.en12354_5._MAX_PATH_COLUMNS` puts on path
#: columns for the same reason.
_MAX_MODE_COLUMNS = 6


def _basis(
    declaration: NoiseEmissionDeclaration,
    metadata: ReportMetadata | None,
    language: str = "en",
) -> str:
    """The standard-basis line: ISO 4871:1996 plus the basic emission standard."""
    standards = list(declaration.basic_standards)
    if not standards and metadata is not None and metadata.measurement_standard:
        standards = [metadata.measurement_standard]
    if standards:
        return t(
            "Declaration and verification of noise emission values per "
            "ISO 4871:1996; values determined per {standard}.",
            language,
        ).format(standard=html.escape(", ".join(standards)))
    return t(
        "Declaration and verification of noise emission values per ISO 4871:1996.",
        language,
    )


def _footnote(declaration: NoiseEmissionDeclaration, language: str = "en") -> str:
    """The clause-5 b determination footnote (noise test code + basic standards)."""
    standards = ", ".join(html.escape(s) for s in declaration.basic_standards)
    code = (
        html.escape(declaration.noise_test_code)
        if declaration.noise_test_code
        else None
    )
    if code and standards:
        return t(
            "Values determined according to noise test code {code}, using the "
            "basic standard(s) {standards}.",
            language,
        ).format(code=code, standards=standards)
    if standards:
        return t(
            "Values determined using the basic standard(s) {standards}.",
            language,
        ).format(standards=standards)
    if code:
        return t(
            "Values determined according to noise test code {code}.",
            language,
        ).format(code=code)
    return t("Values determined in accordance with ISO 4871:1996.", language)


def _fmt_level(value: float, language: str = "en") -> str:
    """A declared level, rounded to the nearest decibel (ISO 4871 3.15/3.16)."""
    # Reuse the model's halves-up rounding so the printed value matches the
    # declared quantity exactly (Python's round() is round-half-to-even).
    from ..emission.declaration import _round_db

    return format_number(_round_db(value), language, decimals=0)


def _dual_rows(
    declaration: NoiseEmissionDeclaration, language: str = "en"
) -> list[tuple[str, list[str]]]:
    """Dual-number value rows: (label, per-mode cells) in ISO 4871 Annex B order."""
    modes = list(declaration.modes)
    rows: list[tuple[str, list[str]]] = [
        (
            t(
                "Measured A-weighted sound power level, "
                "L<sub>WA</sub> (ref. 1 pW), in decibels",
                language,
            ),
            [_fmt_level(m.sound_power_level, language) for m in modes],
        ),
        (
            t("Uncertainty, K<sub>WA</sub>, in decibels", language),
            [_fmt_level(m.sound_power_uncertainty, language) for m in modes],
        ),
    ]
    if any(m.emission_pressure_level is not None for m in modes):
        rows.append(
            (
                t(
                    "Measured A-weighted emission sound pressure level, "
                    "L<sub>pA</sub> (ref. 20 &#181;Pa) at the work station, "
                    "in decibels",
                    language,
                ),
                [_emission_cell(m.emission_pressure_level, language) for m in modes],
            )
        )
        rows.append(
            (
                t("Uncertainty, K<sub>pA</sub>, in decibels", language),
                [
                    _emission_cell(m.emission_pressure_uncertainty, language)
                    for m in modes
                ],
            )
        )
    # The ISO 4871 Annex B.2 dual-number layout states only L and K, each
    # rounded to the nearest decibel (clause 3.16); no derived L_WAd row (the
    # NOTE below the table already explains that the sum is an upper boundary).
    return rows


def _single_rows(
    declaration: NoiseEmissionDeclaration, language: str = "en"
) -> list[tuple[str, list[str]]]:
    """Single-number value rows: the declared L_WAd (and L_pAd) per mode (3.15)."""
    modes = list(declaration.modes)
    rows: list[tuple[str, list[str]]] = [
        (
            t(
                "A-weighted sound power level, "
                "L<sub>WAd</sub> (ref. 1 pW), in decibels",
                language,
            ),
            [
                format_number(m.declared_sound_power_level, language, decimals=0)
                for m in modes
            ],
        )
    ]
    if any(m.emission_pressure_level is not None for m in modes):
        rows.append(
            (
                t(
                    "A-weighted emission sound pressure level, "
                    "L<sub>pAd</sub> (ref. 20 &#181;Pa) at the work station, "
                    "in decibels",
                    language,
                ),
                [
                    _emission_declared_cell(
                        m.declared_emission_pressure_level, language
                    )
                    for m in modes
                ],
            )
        )
    return rows


def _emission_cell(value: float | None, language: str = "en") -> str:
    """A per-mode emission-pressure cell, or an en dash when not declared."""
    if value is None:
        return _NOT_DECLARED
    return _fmt_level(value, language)


def _emission_declared_cell(value: int | None, language: str = "en") -> str:
    """A per-mode declared emission-pressure cell, or an en dash when absent."""
    if value is None:
        return _NOT_DECLARED
    return format_number(value, language, decimals=0)


def _declaration_table(
    declaration: NoiseEmissionDeclaration, language: str = "en"
) -> Any:
    """Reproduce the ISO 4871 Annex B declaration table (identification + values)."""
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph

    styles = getSampleStyleSheet()
    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)

    ident_label_style = ParagraphStyle(
        "iso4871_ident_label",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor(_MUTED_HEX),
    )
    ident_value_style = ParagraphStyle(
        "iso4871_ident_value",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    )
    banner_style = ParagraphStyle(
        "iso4871_banner",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        alignment=1,
        textColor=colors.white,
    )
    col_header_style = ParagraphStyle(
        "iso4871_col_header",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=1,
    )
    label_style = ParagraphStyle(
        "iso4871_row_label",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
    )
    value_style = ParagraphStyle(
        "iso4871_row_value",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11,
        alignment=1,
    )

    modes = list(declaration.modes)
    n_modes = len(modes)
    value_rows = (
        _dual_rows(declaration, language)
        if declaration.form == "dual-number"
        else _single_rows(declaration, language)
    )
    banner_text = (
        t("DECLARED DUAL-NUMBER NOISE EMISSION VALUES", language)
        if declaration.form == "dual-number"
        else t("DECLARED SINGLE-NUMBER NOISE EMISSION VALUES", language)
    )

    # Identification header (clause 5 a / 5 c), spanning the whole table.
    identity_bits: list[str] = []
    if declaration.machine:
        identity_bits.append(html.escape(declaration.machine))
    if declaration.operating_conditions:
        identity_bits.append(html.escape(declaration.operating_conditions))
    identity_text = ", ".join(identity_bits) if identity_bits else _NOT_DECLARED

    span_cols = 1 + n_modes
    data: list[list[Any]] = [
        [
            fiche_paragraph(
                t(
                    "Machine model number, operating conditions, and other "
                    "identifying information:",
                    language,
                ),
                ident_label_style,
            )
        ]
        + [""] * n_modes,
        [fiche_paragraph(identity_text, ident_value_style)] + [""] * n_modes,
        [
            fiche_paragraph(
                f"<b>{banner_text}</b><br/>"
                f"{t('in accordance with ISO 4871:1996', language)}",
                banner_style,
            )
        ]
        + [""] * n_modes,
        [fiche_paragraph("", col_header_style)]
        + [
            fiche_paragraph(f"<b>{html.escape(m.mode)}</b>", col_header_style)
            for m in modes
        ],
    ]
    first_value_row = len(data)
    for label, cells in value_rows:
        data.append(
            [fiche_paragraph(label, label_style)]
            + [fiche_paragraph(cell, value_style) for cell in cells]
        )
    footnote_row = len(data)
    data.append(
        [
            fiche_paragraph(
                f"{_footnote(declaration, language)}<br/><br/>"
                f"{_note(declaration, language)}",
                ident_label_style,
            )
        ]
        + [""] * n_modes,
    )

    label_w = 74.0
    if n_modes > _MAX_MODE_COLUMNS:
        msg = (
            f"A declaration of {n_modes} operating modes does not fit the "
            f"sheet: the {174.0 - label_w:g} mm of mode columns divide into "
            f"{(174.0 - label_w) / n_modes:.1f} mm each, narrower than the "
            "levels they hold. Declare the modes in groups of "
            f"{_MAX_MODE_COLUMNS} or fewer."
        )
        raise ValueError(msg)
    mode_w = (174.0 - label_w) / n_modes
    table = Table(
        data,
        colWidths=[label_w * mm] + [mode_w * mm] * n_modes,
    )
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1.0, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        # Identification block (rows 0-1) spans the full width.
        ("SPAN", (0, 0), (span_cols - 1, 0)),
        ("SPAN", (0, 1), (span_cols - 1, 1)),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, accent),
        # Accent banner row (row 2) spans the full width.
        ("SPAN", (0, 2), (span_cols - 1, 2)),
        ("BACKGROUND", (0, 2), (-1, 2), accent),
        # Column-header row (row 3): light fill, vertical rules between modes.
        ("BACKGROUND", (0, 3), (-1, 3), light),
        ("LINEBELOW", (0, 3), (-1, 3), 0.5, accent),
        # Footnote row spans the full width.
        ("SPAN", (0, footnote_row), (span_cols - 1, footnote_row)),
        ("LINEABOVE", (0, footnote_row), (-1, footnote_row), 0.5, accent),
        ("TOPPADDING", (0, footnote_row), (-1, footnote_row), 5),
    ]
    # Zebra striping on the value rows and a light rule under each.
    style.extend(
        ("BACKGROUND", (0, i), (-1, i), light)
        for i in range(first_value_row, footnote_row)
        if (i - first_value_row) % 2 == 1
    )
    # A vertical rule separating the label column from the mode columns.
    style.append(("LINEAFTER", (0, 3), (0, footnote_row - 1), 0.4, accent))
    table.setStyle(TableStyle(style))
    return table


def _note(declaration: NoiseEmissionDeclaration, language: str = "en") -> str:
    """The ISO 4871 Annex B upper-boundary note (dual- or single-number wording)."""
    if declaration.form == "dual-number":
        return t(
            "NOTE The sum of a measured noise emission value and its associated "
            "uncertainty represents an upper boundary of the range of values "
            "which is likely to occur in measurements.",
            language,
        )
    return t(
        "NOTE Declared single-number noise emission values are the sum of "
        "measured values and the associated uncertainty, and represent upper "
        "boundaries of the range of values which is likely to occur in "
        "measurements.",
        language,
    )


def _verification_rows(
    declaration: NoiseEmissionDeclaration, language: str = "en"
) -> list[tuple[str, str, str, str]]:
    """Verification rows (clause 6.2): L_1 against the declared form, per mode.

    The single-number form verifies against the combined ``L_WAd``; the
    dual-number form against the sum of the separately rounded declared values
    ``L_WA + K_WA`` (clauses 3.16 and 6.2).
    """
    dual = declaration.form == "dual-number"
    rows: list[tuple[str, str, str, str]] = []
    for mode in declaration.modes:
        if mode.verification_level is None:
            continue
        if dual:
            limit = mode.dual_number_verification_limit
            verdict = mode.verified_dual
            criterion = t(
                "&#8804; L<sub>WA</sub> + K<sub>WA</sub> = {ld} dB", language
            ).format(ld=format_number(limit, language, decimals=0))
        else:
            limit = mode.declared_sound_power_level
            verdict = mode.verified
            criterion = t("&#8804; L<sub>WAd</sub> = {ld} dB", language).format(
                ld=format_number(limit, language, decimals=0)
            )
        rows.append(
            (
                t("A-weighted sound power level, {mode}", language).format(
                    mode=html.escape(mode.mode)
                ),
                t("L<sub>1</sub> = {l1} dB", language).format(
                    l1=fmt_num(float(mode.verification_level), language)
                ),
                criterion,
                "pass" if verdict else "fail",
            )
        )
    return rows


def render_iso4871_report(
    declaration: NoiseEmissionDeclaration,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 4871:1996 noise-emission-declaration fiche to a PDF at ``path``.

    :param declaration: A
        :class:`~phonometry.emission.declaration.NoiseEmissionDeclaration`
        carrying the per-operating-mode declared values and the accompanying
        clause-5 information.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the footer
        identity and, through ``measurement_standard``, the basic emission
        standard shown in the basis line when the declaration cites none.
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        declaration fiche has a single table layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab is not installed.
    """
    del verbose  # uniform signature; the fiche has one stacked table layout
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    _styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Noise emission declaration", language)

    flow: list[Any] = [
        fiche_paragraph(title, title_style),
        fiche_paragraph(_basis(declaration, metadata, language), basis_style),
        Spacer(1, 8),
        _declaration_table(declaration, language),
    ]

    verification_rows = _verification_rows(declaration, language)
    if verification_rows:
        flow.append(Spacer(1, 8))
        flow.append(
            fiche_paragraph(
                t("Verification (ISO 4871 clause 6.2)", language), caption_style
            )
        )
        flow.append(
            compliance_table(
                verification_rows,
                col_widths=[74 * mm, 34 * mm, 42 * mm, 24 * mm],
                language=language,
            )
        )

    flow.extend(footer_flow(metadata, language))
    return build_document(path, flow, title)
