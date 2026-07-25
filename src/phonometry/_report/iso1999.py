#  Copyright (c) 2026. Jose M. Requena-Plens
"""ISO 1999:2013 noise-induced hearing-loss prediction fiches (reportlab renderer).

Renders the two occupational-hearing-loss result types of
:mod:`phonometry.hearing.noise_induced_hearing_loss` to one-page PDFs laid out
like a statistical hearing-damage prediction sheet. Both quantities are
population estimates over the six ISO 1999 audiometric frequencies (500 Hz to
6000 Hz), not clinical measurements, so the fiches read as predictions:

* :class:`~phonometry.hearing.noise_induced_hearing_loss.NiptsResult` renders
  the noise-induced permanent threshold shift (NIPTS, clause 6.3): a
  per-audiometric-frequency table of the median ``N50`` (Formula 2/3) and the
  NIPTS at the chosen population fractile (Formula 4/5) beside the result's own
  spectrum plot, the boxed representative shift averaged over the 2/3/4 kHz
  hearing-handicap set, and the exposure conditions (``L_EX,8h``, exposure
  years, fractile); ``verbose=True`` adds the upper/lower spread columns
  (``du``/``dl``, Formulae 6/7);
* :class:`~phonometry.hearing.noise_induced_hearing_loss.HtlanResult` renders
  the hearing threshold level associated with age and noise (HTLAN, clause 6.1):
  a per-audiometric-frequency table of the age component ``H`` (HTLA, database
  A = ISO 7029), the noise component ``N`` (NIPTS) and the combined threshold
  ``H' = H + N - H*N/120`` (Formula 1) beside the plot, the boxed representative
  threshold averaged over the 2/3/4 kHz set, and the listener/exposure
  conditions; ``verbose=True`` adds the compression term ``H*N/120``.

A metadata ``requirement`` (a maximum acceptable representative value in dB)
adds a PASS/FAIL verdict; without it the fiche prints no verdict, since neither
quantity carries a normative limit of its own. The quantity-independent
skeleton lives in :mod:`._layout`; this module holds the ISO 1999 specifics.
reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import decimal_comma, format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    analysis_cell_styles,
    build_document,
    display_round,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    render_figure_drawing,
    result_box,
    stacked_table,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..hearing.noise_induced_hearing_loss import HtlanResult, NiptsResult

#: The 2/3/4 kHz hearing-handicap audiometric set, in hertz. The mean threshold
#: shift over these three frequencies is the descriptor most occupational
#: schemes use as a single hearing-damage index.
_HANDICAP_SET: tuple[float, float, float] = (2000.0, 3000.0, 4000.0)

#: The compression-term denominator of the HTLAN Formula (1).
_HTLAN_DENOM = 120.0

#: Footer scope statement of the ISO 1999 fiches: the results describe a
#: population under the stated exposure, not a tested specimen or any single
#: person. English key, translated via :func:`t` at render time.
_POPULATION_DISCLAIMER = (
    "The results are a statistical prediction for the stated population and "
    "exposure conditions; they do not describe any individual person."
)


def _fmt_db(value: float, language: str = "en") -> str:
    """A threshold shift or level rounded to one decimal place."""
    return format_number(display_round(float(value)), language, decimals=1)


def _fmt_freq(value: float, language: str = "en") -> str:
    """An audiometric frequency in hertz, as an integer (500 ... 6000)."""
    return format_number(round(float(value)), language, decimals=0)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _handicap_indices(frequencies: np.ndarray) -> list[int]:
    """Indices of the 2/3/4 kHz hearing-handicap frequencies present in the set."""
    return [
        i
        for i, f in enumerate(frequencies)
        if any(abs(float(f) - h) < 1e-6 for h in _HANDICAP_SET)
    ]


def _representative(
    frequencies: np.ndarray, values: np.ndarray
) -> tuple[bool, float, float]:
    """Return the representative shift/level and whether it is the 2/3/4 kHz mean.

    When all three hearing-handicap frequencies are present the representative
    value is their arithmetic mean and the flag is ``True``; otherwise it falls
    back to the peak value across the available frequencies (flag ``False``),
    reported together with its frequency.
    """
    vals = np.asarray(values, dtype=np.float64)
    idx = _handicap_indices(frequencies)
    if len(idx) == 3:
        return True, float(vals[idx].mean()), 0.0
    peak = int(np.argmax(vals))
    return False, float(vals[peak]), float(frequencies[peak])


def _iso_q_percent(fractile: float) -> float:
    """The ISO 1999 percentage ``Q`` for a library ``fractile``.

    ISO 1999:2013 states its distributions for a percentage ``Q`` of the
    population whose values are *worse* (larger) than ``N_Q``/``H_Q``
    (Formulae (4)/(5), 6.3.2): ``Q = 10 %`` is the most-susceptible tenth.
    The library's ``fractile`` is the complementary fraction with *smaller*
    values, so ``Q = 100 (1 - fractile)``.
    """
    return round(100.0 * (1.0 - float(fractile)), 6)


def _fractile_phrase(fractile: float, language: str = "en") -> str:
    """A short gloss of the population fractile, in ISO 1999's own ``Q``.

    ``Q`` is printed with the meaning ISO 1999:2013 gives it in 6.3.2
    (Formulae (4)/(5)): the percentage of the population predicted to have
    values worse (larger) than the stated ones, so a request for the library
    fractile 0.9 prints ``Q = 10 %``.
    """
    q = decimal_comma(f"{_iso_q_percent(fractile):g}", language)
    if abs(fractile - 0.5) < 1e-9:
        return t("Population fractile Q = {q} % (median)", language).format(q=q)
    return t(
        "Population fractile Q = {q} % (fraction with worse hearing)", language
    ).format(q=q)


# --------------------------------------------------------------------------- #
# NIPTS fiche.
# --------------------------------------------------------------------------- #
def _nipts_metadata_pairs(
    metadata: ReportMetadata | None, language: str = "en"
) -> list[tuple[str, str]]:
    """The (label, value) header-grid pairs of a NIPTS/HTLAN fiche."""
    if metadata is None:
        return []
    specs: list[tuple[str, str | None]] = [
        (t("Company", language), _esc(metadata.client)),
        (t("Worker(s) / group", language), _esc(metadata.specimen)),
        (t("Workplace", language), _esc(metadata.test_room)),
        (t("Date of assessment", language), _esc(metadata.test_date)),
    ]
    return [(label, value) for label, value in specs if value]


def _nipts_table(
    result: NiptsResult, verbose: bool = False, language: str = "en"
) -> Any:
    """The per-audiometric-frequency NIPTS table (median and fractile value)."""
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso1999n")

    headers = [
        t("Frequency [Hz]", language),
        "N<sub>50</sub> [dB]",
        t("NIPTS [dB]", language),
    ]
    widths = [24.0, 21.0, 21.0]
    if verbose:
        headers += ["d<sub>u</sub> [dB]", "d<sub>l</sub> [dB]"]
        widths = [22.0, 18.0, 18.0, 19.0, 19.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i, freq in enumerate(result.frequencies):
        row = [
            Paragraph(_fmt_freq(float(freq), language), label_style),
            Paragraph(_fmt_db(float(result.median[i]), language), value_style),
            Paragraph(_fmt_db(float(result.value[i]), language), value_style),
        ]
        if verbose:
            row += [
                Paragraph(_fmt_db(float(result.spread_upper[i]), language), value_style),
                Paragraph(_fmt_db(float(result.spread_lower[i]), language), value_style),
            ]
        data.append(row)

    return stacked_table(data, [w * mm for w in widths])


def _nipts_statement(result: NiptsResult, language: str = "en") -> tuple[str, list[str]]:
    """The boxed representative NIPTS statement and the exposure-condition terms."""
    is_handicap, value, freq = _representative(result.frequencies, result.value)
    if is_handicap:
        statement = t(
            "Predicted NIPTS averaged over 2/3/4 kHz = <b>{value} dB</b>",
            language,
        ).format(value=_fmt_db(value, language))
    else:
        statement = t(
            "Predicted peak NIPTS = <b>{value} dB</b> at {freq} Hz",
            language,
        ).format(value=_fmt_db(value, language), freq=_fmt_freq(freq, language))
    extended = [
        t("Noise exposure L<sub>EX,8h</sub> = {lex} dB", language).format(
            lex=decimal_comma(f"{result.l_ex:g}", language)
        ),
        t("Exposure duration = {years} years", language).format(
            years=decimal_comma(f"{result.years:g}", language)
        ),
        _fractile_phrase(result.fractile, language),
    ]
    return statement, extended


def _nipts_verdict(
    result: NiptsResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: the representative NIPTS at or below a maximum.

    The requirement is read as the maximum acceptable representative NIPTS (a
    lower shift is better); the comparison uses the displayed one-decimal value,
    so the printed number can never contradict the verdict.
    """
    _, value, _ = _representative(result.frequencies, result.value)
    passed = display_round(value) <= requirement + 1e-9
    text = t(
        "representative NIPTS = {value} dB, maximum {req} dB",
        language,
    ).format(value=_fmt_db(value, language), req=fmt_num(requirement, language))
    return text, passed


def render_nipts_report(
    result: NiptsResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a NIPTS prediction fiche to a PDF at ``path`` (ISO 1999:2013, 6.3).

    :param result: A
        :class:`~phonometry.hearing.noise_induced_hearing_loss.NiptsResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``client`` is the company, ``specimen`` the worker(s)/group,
        ``test_room`` the workplace) and, via ``requirement``, a maximum
        acceptable representative NIPTS that adds a PASS/FAIL verdict.
    :param verbose: When True, the table adds the upper/lower spread columns.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab, matplotlib or svglib is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Spacer
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Noise-induced hearing loss prediction", language)
    basis = t(
        "Statistical prediction of the noise-induced permanent threshold shift "
        "of a noise-exposed population per ISO 1999:2013 (clause 6.3).",
        language,
    )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _nipts_metadata_pairs(metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(t("Threshold shift by frequency", language), caption_style),
        _nipts_table(result, verbose, language),
    ]
    left_width = 96.0 if verbose else 66.0
    plot_drawing = render_figure_drawing(
        result.plot, (174.0 - left_width) * mm, y_top=None,
        figsize=(5.4, 4.6), language=language,
    )
    flow.append(
        two_panel_body(
            left_cell, plot_drawing,
            left_width_mm=left_width, plot_width_mm=174.0 - left_width,
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = _nipts_statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _nipts_verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    is_handicap, _, _ = _representative(result.frequencies, result.value)
    flow.extend(
        _prediction_notes(
            language,
            l_ex=float(result.l_ex),
            years=float(result.years),
            fractile=float(result.fractile),
            handicap_mean=is_handicap,
        )
    )
    flow.extend(footer_flow(metadata, language, disclaimer=_POPULATION_DISCLAIMER))

    return build_document(path, flow, title)


# --------------------------------------------------------------------------- #
# HTLAN fiche.
# --------------------------------------------------------------------------- #
def _htlan_table(
    result: HtlanResult, verbose: bool = False, language: str = "en"
) -> Any:
    """The per-audiometric-frequency HTLAN table (age, noise and combined)."""
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph

    header_style, label_style, value_style = analysis_cell_styles("iso1999h")

    headers = [
        t("Frequency [Hz]", language),
        "H [dB]",
        "N [dB]",
        "H&#8242; [dB]",
    ]
    widths = [22.0, 17.0, 17.0, 18.0]
    if verbose:
        headers.insert(3, "H&#183;N/120 [dB]")
        widths = [20.0, 16.0, 16.0, 20.0, 20.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i, freq in enumerate(result.frequencies):
        h = float(result.htla[i])
        n = float(result.nipts[i])
        row = [
            Paragraph(_fmt_freq(float(freq), language), label_style),
            Paragraph(_fmt_db(h, language), value_style),
            Paragraph(_fmt_db(n, language), value_style),
        ]
        if verbose:
            row.append(Paragraph(_fmt_db(h * n / _HTLAN_DENOM, language), value_style))
        row.append(Paragraph(_fmt_db(float(result.threshold[i]), language), value_style))
        data.append(row)

    return stacked_table(data, [w * mm for w in widths])


def _htlan_statement(result: HtlanResult, language: str = "en") -> tuple[str, list[str]]:
    """The boxed representative HTLAN statement and the listener/exposure terms."""
    is_handicap, value, freq = _representative(result.frequencies, result.threshold)
    if is_handicap:
        statement = t(
            "Predicted hearing threshold level (age and noise) averaged over "
            "2/3/4 kHz = <b>{value} dB HL</b>",
            language,
        ).format(value=_fmt_db(value, language))
    else:
        statement = t(
            "Predicted peak hearing threshold level (age and noise) = "
            "<b>{value} dB HL</b> at {freq} Hz",
            language,
        ).format(value=_fmt_db(value, language), freq=_fmt_freq(freq, language))
    sex = t(result.sex, language)
    extended = [
        t("Listener: {sex}, age {age} years", language).format(
            sex=sex, age=decimal_comma(f"{result.age:g}", language)
        ),
        t("Noise exposure L<sub>EX,8h</sub> = {lex} dB over {years} years", language).format(
            lex=decimal_comma(f"{result.l_ex:g}", language),
            years=decimal_comma(f"{result.years:g}", language),
        ),
        _fractile_phrase(result.fractile, language),
    ]
    return statement, extended


def _htlan_verdict(
    result: HtlanResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag: the representative HTLAN at or below a maximum."""
    _, value, _ = _representative(result.frequencies, result.threshold)
    passed = display_round(value) <= requirement + 1e-9
    text = t(
        "representative HTLAN = {value} dB HL, maximum {req} dB HL",
        language,
    ).format(value=_fmt_db(value, language), req=fmt_num(requirement, language))
    return text, passed


def render_htlan_report(
    result: HtlanResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an HTLAN prediction fiche to a PDF at ``path`` (ISO 1999:2013, 6.1).

    :param result: A
        :class:`~phonometry.hearing.noise_induced_hearing_loss.HtlanResult`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity and, via ``requirement``, a maximum acceptable representative
        HTLAN that adds a PASS/FAIL verdict.
    :param verbose: When True, the table adds the compression term ``H*N/120``.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab, matplotlib or svglib is not installed.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Paragraph, Spacer
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Hearing threshold level prediction (age and noise)", language)
    basis = t(
        "Statistical prediction of the hearing threshold level associated with "
        "age and noise per ISO 1999:2013 (clause 6.1).",
        language,
    )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _nipts_metadata_pairs(metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(t("Threshold level by frequency", language), caption_style),
        _htlan_table(result, verbose, language),
    ]
    left_width = 92.0 if verbose else 74.0
    plot_drawing = render_figure_drawing(
        result.plot, (174.0 - left_width) * mm, y_top=None,
        figsize=(5.4, 4.6), language=language,
    )
    flow.append(
        two_panel_body(
            left_cell, plot_drawing,
            left_width_mm=left_width, plot_width_mm=174.0 - left_width,
        )
    )
    flow.append(Spacer(1, 8))

    statement, extended = _htlan_statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))
    if metadata is not None and metadata.requirement is not None:
        text, passed = _htlan_verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    is_handicap, _, _ = _representative(result.frequencies, result.threshold)
    flow.extend(
        _prediction_notes(
            language,
            l_ex=float(result.l_ex),
            years=float(result.years),
            fractile=float(result.fractile),
            handicap_mean=is_handicap,
            with_age_component=True,
        )
    )
    flow.extend(footer_flow(metadata, language, disclaimer=_POPULATION_DISCLAIMER))

    return build_document(path, flow, title)


def _outside_validated_domain(l_ex: float, years: float, fractile: float) -> bool:
    """Whether the exposure conditions leave the standard's validated domain.

    The bounds live with the calculation
    (:mod:`~phonometry.hearing.noise_induced_hearing_loss`, which also warns at
    call time); the fiche re-evaluates them so the caveat always accompanies an
    extrapolated print-out.
    """
    from ..hearing.noise_induced_hearing_loss import (
        VALIDATED_FRACTILES,
        VALIDATED_L_EX_MAX,
        VALIDATED_YEARS,
    )

    return (
        l_ex > VALIDATED_L_EX_MAX
        or not VALIDATED_YEARS[0] <= years <= VALIDATED_YEARS[1]
        or not VALIDATED_FRACTILES[0] <= fractile <= VALIDATED_FRACTILES[1]
    )


def _prediction_notes(
    language: str = "en",
    *,
    l_ex: float,
    years: float,
    fractile: float,
    handicap_mean: bool,
    with_age_component: bool = False,
) -> list[Any]:
    """The shared statistical-prediction notes of the ISO 1999 fiches.

    Always states the population-statistics character and ISO 1999's own
    reading of the percentage ``Q`` (6.3.2, Formulae (4)/(5): the fraction with
    worse hearing). ``handicap_mean`` adds the Scope NOTE 1 caveat that the
    2/3/4 kHz combination is the user's choice, ``with_age_component`` adds the
    source of the age-related threshold ``H`` (ISO 7029:2017), and exposure
    conditions outside the standard's validated domain add an explicit
    extrapolation caveat.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph

    from ._layout import _MUTED_HEX

    note_style = ParagraphStyle(
        "iso1999_notes", parent=getSampleStyleSheet()["Normal"],
        fontSize=7.5, leading=10, textColor=colors.HexColor(_MUTED_HEX),
        spaceBefore=6,
    )
    notes: list[Any] = [
        Paragraph(
            t(
                "These values are a statistical prediction for a noise-exposed "
                "population (ISO 1999:2013), not a clinical diagnosis or a "
                "measured audiogram of any individual.",
                language,
            ),
            note_style,
        ),
        Paragraph(
            t(
                "Q is the percentage of the noise-exposed population predicted "
                "to show a larger (worse) value than the one stated, as "
                "ISO 1999:2013 defines it (6.3.2, Formulae (4) and (5)); "
                "Q = 10 % is the most-susceptible tenth.",
                language,
            ),
            note_style,
        ),
    ]
    if with_age_component:
        notes.append(
            Paragraph(
                t(
                    "The age component H (database A) is evaluated from "
                    "ISO 7029:2017, the edition ISO 1999:2013 references "
                    "undated (6.2.2); its values differ from the illustrative "
                    "Table A.3 selection, which derives from an earlier "
                    "ISO 7029 edition.",
                    language,
                ),
                note_style,
            )
        )
    if handicap_mean:
        notes.append(
            Paragraph(
                t(
                    "ISO 1999:2013 does not specify frequencies or frequency "
                    "combinations for evaluating hearing disability (Scope, "
                    "NOTE 1); the 2/3/4 kHz average shown is a commonly used "
                    "descriptor whose choice is left to the user.",
                    language,
                ),
                note_style,
            )
        )
    if _outside_validated_domain(l_ex, years, fractile):
        notes.append(
            Paragraph(
                t(
                    "The stated conditions lie outside the validated domain of "
                    "ISO 1999:2013 (exposure durations of 1 year to 40 years, "
                    "fractiles Q of 5 % to 95 %, exposure levels up to the "
                    "100 dB covered by Annex D); these values are an "
                    "extrapolation.",
                    language,
                ),
                note_style,
            )
        )
    return notes
