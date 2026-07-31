#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Room acoustic parameters fiche (reportlab renderer, ISO 3382-1/-2).

Renders a :class:`~phonometry.room.room_acoustics.RoomAcousticsResult` to a
one-page PDF laid out like a room-acoustics measurement report (a performance
space per ISO 3382-1:2009 or an ordinary room per ISO 3382-2:2008, both
evaluated by the integrated impulse-response method of ISO 3382-1:2009, 5.3.3):

* a title and the standard-basis line (measurement standard + the ISO 3382
  parts and the integrated impulse-response method);
* an optional metadata header block (room, volume, source/receiver positions,
  climate, laboratory ...), rendered only for the fields supplied on the
  :class:`ReportMetadata`;
* a full-width per-band parameter table (frequency rows against the
  reverberation times T20/T30/EDT and the energy parameters C50/C80/D50/Ts)
  above the result's own per-band decay-time plot (drawn by ``plot(ax=...)`` so
  the chart is native to the library);
* a boxed single-number result, the mid-frequency reverberation time T_mid
  (the mean of the 500 Hz and 1000 Hz band T30, the customary room
  descriptor; an octave analysis averages the two octaves, a one-third-octave
  analysis averages the 500 Hz and 1 kHz one-third-octave bands and is
  labelled as such) with the mid-frequency EDT alongside;
* an optional verdict row when a target mid-frequency reverberation time is
  supplied (ISO 3382-1/-2 are characterisation standards with no intrinsic
  pass/fail, so the row appears only when a requirement is given);
* a short measurement-basis strip and a footer identity/disclaimer block.

Unlike the two-panel band fiches (:mod:`.iso717`, :mod:`.iso11654`) this uses a
stacked layout: the per-band table carries eight columns (frequency and seven
parameters), which need the full content width, and the decay-time plot is a
landscape bar chart. The quantity-independent skeleton lives in :mod:`._layout`;
this module only holds the ISO 3382 specifics. reportlab, matplotlib and svglib
are soft dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _LIGHT_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    escaped_pairs,
    fmt_meta,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..room.room_acoustics import RoomAcousticsResult

#: Band centres whose T30 mean is the mid-frequency reverberation-time
#: descriptor T_mid quoted for rooms: the 500 Hz and 1000 Hz bands at the
#: analysed fraction (octaves, or the two one-third-octave bands, labelled
#: accordingly by the statement).
_TMID_BANDS = (500.0, 1000.0)

#: Tolerance (Hz, relative) for matching a band centre to a nominal frequency.
_BAND_MATCH_REL = 0.06


def _cell(value: float, decimals: int, language: str, *, scale: float = 1.0) -> str:
    """Format one table value, or an em dash when it is not finite.

    A non-finite parameter (evaluation range unreachable, or reaching below the
    trusted part of the decay curve) is shown as ``"—"``, the accredited
    empty-cell symbol, rather than a spurious number.
    """
    v = float(value) * scale
    if not math.isfinite(v):
        return "—"
    return format_number(v, language, decimals=decimals)


def _fraction_label(frequency: np.ndarray | None, language: str) -> str:
    """Return the caption describing the analysis band set.

    Only a ``frequency is None`` result is a broadband analysis; a result that
    carries a single band is a single octave or one-third-octave band (from a
    narrow ``limits`` selection), so it is labelled as a single band rather than
    broadband. Two or more bands are classified octave vs one-third-octave by
    their centre ratios.
    """
    if frequency is None:
        return t("Broadband analysis", language)
    if frequency.size < 2:
        return t("Single-band parameters", language)
    from ..metrology.frequencies import _infer_band_fraction

    if _infer_band_fraction(frequency) == 1:
        return t("Octave-band parameters", language)
    return t("One-third-octave-band parameters", language)


def _band_label(exact_freq: float, fraction: int) -> str:
    """The nominal band label of an exact mid-band frequency (e.g. ``125``).

    ISO 3382-1/-2 report parameters by their nominal octave/one-third-octave
    mid-band frequencies (IEC 61260), not the exact base-ten centre, so the
    table is labelled with the nominal frequency (125, not 125.89...).
    """
    from ..metrology.frequencies import _nominal_freq_for_band

    nominal = _nominal_freq_for_band(exact_freq, float(fraction))
    return f"{nominal:g}"


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the room-acoustics header grid.

    Only fields that are set are returned, so empty rows never appear. A room
    is a single enclosure, so the single ``room_volume`` and the source/receiver
    position counts are used rather than the source/receiving transmission pair.
    """

    def num(value: float | None) -> str | None:
        # Round-trip formatting: the header grid reprints client-supplied
        # values and must not silently reduce them (a 2830 m^3 hall volume).
        return fmt_meta(value, language) if value is not None else None

    def count(value: int | None) -> str | None:
        return str(int(value)) if value is not None else None

    specs: list[tuple[str, str | None]] = [
        (t("Client", language), metadata.client),
        (t("Room", language), metadata.test_room),
        (t("Description", language), metadata.specimen),
        (t("Room volume V [m<super>3</super>]", language), num(metadata.room_volume)),
        (t("Floor area S [m<super>2</super>]", language), num(metadata.area)),
        (t("Source positions", language), count(metadata.source_positions)),
        (t("Microphone positions", language), count(metadata.receiver_positions)),
        (t("Instrumentation", language), metadata.instrumentation),
        (t("Temperature [&#176;C]", language), num(metadata.temperature)),
        (t("Relative humidity [%]", language), num(metadata.relative_humidity)),
        (t("Ambient pressure [kPa]", language), num(metadata.pressure)),
        (t("Date of test", language), metadata.test_date),
    ]
    return escaped_pairs(specs)


def _parameter_table(
    result: RoomAcousticsResult, language: str = "en"
) -> Any:
    """Build the full-width per-band parameter table.

    Rows are the analysis bands (or a single ``Broadband`` row); the columns are
    the reverberation times T20/T30/EDT (s), the clarity indices C50/C80 (dB),
    the definition D50 and the centre time Ts (ms), per ISO 3382-1:2009 Annex A.
    A non-finite value is rendered as an em dash. Called only after
    :func:`render_iso3382_report` has imported reportlab.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Table, TableStyle

    from ._layout import fiche_paragraph as Paragraph

    accent = colors.HexColor(_ACCENT_HEX)
    light = colors.HexColor(_LIGHT_HEX)
    thin = colors.HexColor("#c9d4e0")
    styles = getSampleStyleSheet()
    head_style = ParagraphStyle(
        "iso3382_thead", parent=styles["Normal"], fontSize=7.4,
        textColor=colors.white, alignment=1, leading=8.6,
    )

    freq = result.frequency
    t20 = np.asarray(result.t20, dtype=np.float64)
    t30 = np.asarray(result.t30, dtype=np.float64)
    edt = np.asarray(result.edt, dtype=np.float64)
    c50 = np.asarray(result.c50, dtype=np.float64)
    c80 = np.asarray(result.c80, dtype=np.float64)
    d50 = np.asarray(result.d50, dtype=np.float64)
    ts = np.asarray(result.ts, dtype=np.float64)
    n = t30.size

    from ..metrology.frequencies import _infer_band_fraction

    fraction = _infer_band_fraction(freq) if freq is not None else 1
    if freq is None:
        labels = [t("Broadband", language)] * n
    else:
        labels = [
            _band_label(f, fraction) for f in np.asarray(freq, dtype=np.float64)
        ]

    # A one-third-octave range carries up to 18 rows; tighten the padding and
    # cell font so the stacked table plus the landscape plot still fit one A4
    # page (an octave range has at most 6 rows and uses the roomier spacing).
    compact = n > 8
    pad = 1.0 if compact else 2.6
    body_font = 6.8 if compact else 8.0

    header = [
        Paragraph(t("f [Hz]", language), head_style),
        Paragraph("T<sub>20</sub> [s]", head_style),
        Paragraph("T<sub>30</sub> [s]", head_style),
        Paragraph("EDT [s]", head_style),
        Paragraph("C<sub>50</sub> [dB]", head_style),
        Paragraph("C<sub>80</sub> [dB]", head_style),
        Paragraph("D<sub>50</sub>", head_style),
        Paragraph("T<sub>s</sub> [ms]", head_style),
    ]
    rows: list[list[Any]] = [header]
    for i in range(n):
        rows.append(
            [
                labels[i],
                _cell(t20[i], 2, language),
                _cell(t30[i], 2, language),
                _cell(edt[i], 2, language),
                _cell(c50[i], 1, language),
                _cell(c80[i], 1, language),
                _cell(d50[i], 2, language),
                _cell(ts[i], 0, language, scale=1000.0),
            ]
        )

    col_widths = [
        24 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm, 22 * mm, 20 * mm, 26 * mm,
    ]
    style_cmds: list[Any] = [
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("FONTSIZE", (0, 1), (-1, -1), body_font),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, light]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, accent),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("BOX", (0, 0), (-1, -1), 0.5, accent),
    ]
    # A one-third-octave set groups by octave (a thin rule after every triplet),
    # exactly as accredited room-acoustics tables print it; an octave set or a
    # broadband single row has no one-third-octave triplets to group, so the
    # grouping rule is gated on the band structure actually being one-third
    # octave rather than merely on the row count.
    if freq is not None and fraction == 3:
        for triplet_end in range(3, n, 3):
            style_cmds.append(
                ("LINEBELOW", (0, triplet_end), (-1, triplet_end), 0.4, thin)
            )
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    return table


def _band_value(
    frequency: np.ndarray | None, values: np.ndarray, target: float
) -> float:
    """Return the parameter value at the band closest to ``target`` Hz.

    Used to read the 500 Hz and 1000 Hz octave values for the mid-frequency
    descriptor; returns NaN when there is no band within tolerance.
    """
    if frequency is None:
        return float("nan")
    freq = np.asarray(frequency, dtype=np.float64)
    idx = int(np.argmin(np.abs(freq - target)))
    if abs(freq[idx] - target) > _BAND_MATCH_REL * target:
        return float("nan")
    return float(values[idx])


def _band_fraction(frequency: np.ndarray | None) -> int:
    """The band fraction of a result: octave (1) or one-third octave (3).

    Uses the shared :func:`.._infer_band_fraction` classifier so the boxed
    descriptor is labelled with exactly the band structure the table and plot
    use; a broadband result defaults to octave (the label is unused then).
    """
    if frequency is None:
        return 1
    from ..metrology.frequencies import _infer_band_fraction

    return _infer_band_fraction(np.asarray(frequency, dtype=np.float64))


def _reverberation_descriptor(
    result: RoomAcousticsResult, values: np.ndarray
) -> tuple[float, bool, float]:
    """Return ``(value, is_mid, band)`` for the boxed reverberation descriptor.

    When the result carries frequency bands and both the 500 Hz and 1000 Hz
    band centres are present and finite, the descriptor is the mid-frequency
    mean of those two bands (T_mid / EDT_mid, the customary ISO 3382 room
    descriptor; for a one-third-octave analysis these are the 500 Hz and
    1 kHz one-third-octave bands and the fiche labels them as such) and
    ``is_mid`` is True (``band`` is NaN). For a broadband result
    (``frequency is None``) or a band range that does not span both mid
    bands, the descriptor is the first finite value with ``is_mid`` False and
    ``band`` its exact band centre (NaN for broadband), so the fiche never
    makes a false mid-frequency claim about a value that was not averaged
    over those bands and always names the band the fallback value came from.
    """
    freq = result.frequency
    arr = np.asarray(values, dtype=np.float64)
    if freq is not None:
        low = _band_value(freq, arr, _TMID_BANDS[0])
        high = _band_value(freq, arr, _TMID_BANDS[1])
        if math.isfinite(low) and math.isfinite(high):
            return 0.5 * (low + high), True, float("nan")
    finite = np.flatnonzero(np.isfinite(arr))
    if not finite.size:
        return float("nan"), False, float("nan")
    idx = int(finite[0])
    band = (
        float(np.asarray(freq, dtype=np.float64)[idx])
        if freq is not None
        else float("nan")
    )
    return float(arr[idx]), False, band


def _statement(result: RoomAcousticsResult, language: str = "en") -> tuple[str, list[str]]:
    """The boxed reverberation-time descriptor and its extended EDT term.

    A band result spanning the 500 Hz and 1000 Hz bands boxes the
    mid-frequency T_mid, labelled "500-1000 Hz" for an octave analysis; a
    thirds analysis boxes a plain T_mid and adds an extended note naming the
    500 Hz and 1 kHz one-third-octave bands, since only those two bands are
    averaged (not the full octaves) and the long label does not fit the
    box. A broadband result boxes the plain T30; a band range without
    both mid bands boxes the first finite T30 band, named explicitly. The
    EDT term picks its "EDT_mid" vs "EDT" form from EDT's own band coverage,
    which can differ from T30's when a mid band's EDT is not evaluable.
    """
    fraction = _band_fraction(result.frequency)
    t_value, t_is_mid, t_band = _reverberation_descriptor(
        result, np.asarray(result.t30, dtype=np.float64)
    )
    edt_value, edt_is_mid, edt_band = _reverberation_descriptor(
        result, np.asarray(result.edt, dtype=np.float64)
    )
    if t_is_mid and fraction == 1:
        statement = t(
            "T<sub>mid</sub> (500-1000 Hz) = <b>{value} s</b>", language
        ).format(value=_cell(t_value, 2, language))
    elif t_is_mid:
        # One-third-octave data: only the 500 Hz and 1 kHz one-third-octave
        # bands are averaged, so the octave "500-1000 Hz" label would be
        # false; the extended note names the bands (the long label does not
        # fit the boxed statement).
        statement = t("T<sub>mid</sub> = <b>{value} s</b>", language).format(
            value=_cell(t_value, 2, language)
        )
    elif math.isfinite(t_band):
        statement = t(
            "T<sub>30</sub> ({band} Hz) = <b>{value} s</b>", language
        ).format(
            band=_band_label(t_band, fraction),
            value=_cell(t_value, 2, language),
        )
    else:
        statement = t("T<sub>30</sub> = <b>{value} s</b>", language).format(
            value=_cell(t_value, 2, language)
        )
    extended: list[str] = []
    if (t_is_mid or edt_is_mid) and fraction == 3:
        extended.append(
            t(
                "Mid-frequency mean over the 500 Hz and 1 kHz "
                "one-third-octave bands.",
                language,
            )
        )
    if math.isfinite(edt_value):
        if edt_is_mid:
            extended.append(
                t("EDT<sub>mid</sub> = {value} s", language).format(
                    value=format_number(edt_value, language, decimals=2)
                )
            )
        elif math.isfinite(edt_band):
            extended.append(
                t("EDT ({band} Hz) = {value} s", language).format(
                    band=_band_label(edt_band, fraction),
                    value=format_number(edt_value, language, decimals=2),
                )
            )
        else:
            extended.append(
                t("EDT = {value} s", language).format(
                    value=format_number(edt_value, language, decimals=2)
                )
            )
    return statement, extended


def _verdict(
    result: RoomAcousticsResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag for a supplied target reverberation time.

    The requirement is read as the maximum acceptable reverberation time (the
    common form of a room-acoustics target, e.g. a classroom or open-plan upper
    limit): the room passes when its measured descriptor is at or below it. The
    verdict names T_mid only when the descriptor is the mid-frequency mean; a
    broadband result is checked against its plain T30, with no "500-1000 Hz"
    claim. The comparison uses both values rounded exactly as the fiche
    displays them (two decimals, halves away from zero), so the printed
    numbers can never contradict the verdict at the tolerance boundary.
    """
    t_value, is_mid, _ = _reverberation_descriptor(
        result, np.asarray(result.t30, dtype=np.float64)
    )
    passed = math.isfinite(t_value) and display_round(t_value, 2) <= display_round(
        requirement, 2
    )
    key = (
        "T<sub>mid</sub> = {value} s, required &#8804; {req} s"
        if is_mid
        else "T<sub>30</sub> = {value} s, required &#8804; {req} s"
    )
    text = t(key, language).format(
        value=_cell(t_value, 2, language),
        req=format_number(requirement, language, decimals=2),
    )
    return text, passed


def render_iso3382_report(
    result: RoomAcousticsResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a room-acoustics parameters fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.room.room_acoustics.RoomAcousticsResult` carrying
        the per-band reverberation times and energy parameters.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        bare characterisation fiche (body + result + disclaimer, no header). A
        supplied ``requirement`` is read as the maximum mid-frequency T.
    :param verbose: Accepted for signature parity with the other fiches; the
        room table already shows every computed parameter, so it has no effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # every parameter is always shown; kept for signature parity
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Room acoustic parameters", language)

    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )
    if measurement_standard:
        basis = t(
            "{standard} measurement of room acoustic parameters by the "
            "integrated impulse-response method (ISO 3382-1:2009 / "
            "ISO 3382-2:2008).",
            language,
        ).format(standard=html.escape(measurement_standard))
    else:
        basis = t(
            "Room acoustic parameters by the integrated impulse-response "
            "method (ISO 3382-1:2009 / ISO 3382-2:2008).",
            language,
        )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    # A one-third-octave range makes the stacked table tall; shrink the
    # inter-element gaps and the landscape plot so the table + plot still fit
    # one A4 page (an octave range keeps the roomier spacing and a taller plot).
    n_bands = np.asarray(result.t30, dtype=np.float64).size
    compact = n_bands > 8
    gap = 3 if compact else 8
    fig_height = 2.0 if compact else 3.9

    if metadata is not None and not metadata.is_empty():
        header_pairs = _metadata_pairs(metadata, language)
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, gap))

    # Full-width per-band parameter table, then the landscape decay-time plot
    # drawn by the result's own single-panel plot(ax=...).
    flow.append(
        Paragraph(_fraction_label(result.frequency, language), caption_style)
    )
    flow.append(_parameter_table(result, language))
    flow.append(Spacer(1, gap))
    plot_drawing = render_figure_drawing(
        result.plot, 174 * mm, y_top=None, figsize=(9.2, fig_height),
        language=language,
    )
    flow.append(plot_drawing)
    flow.append(Spacer(1, gap))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "Decay curves by Schroeder backward integration with noise "
                "truncation and tail compensation (ISO 3382-1:2009, 5.3.3); "
                "T20 over -5 to -25 dB, T30 over -5 to -35 dB, EDT over 0 to "
                "-10 dB, each extrapolated to 60 dB (ISO 3382-2:2008, Clause 6; "
                "ISO 3382-1:2009, A.2.2).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.append(
        Paragraph(
            t(
                "C50/C80, D50 and Ts follow ISO 3382-1:2009 Equations (A.10), "
                "(A.11) and (A.13). The just-noticeable differences (Table A.1) "
                "are 5 % for EDT/T, 1 dB for C80, 0.05 for D50 and 10 ms for Ts.",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
