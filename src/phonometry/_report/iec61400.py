#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Wind-turbine tonal audibility assessment fiche (reportlab renderer).

Renders a
:class:`~phonometry.environmental.wind_turbine_noise.WindTurbineTonalityResult`
to a one-page PDF laid out like a wind-turbine tonality-assessment report of an
environmental-noise laboratory. IEC 61400-11:2012+A1:2018 decides whether a
wind turbine emits an audible tone: from the narrow-band spectrum it forms the
critical band about the candidate tone (Formula 30), the masking-noise level
``L_pn`` (Formula 31), the tonality ``ΔL_tn = L_pt - L_pn`` (Formula 32), the
audibility criterion ``L_a`` (Formula 34) and the tonal audibility
``ΔL_a = ΔL_tn - L_a`` (Formula 33); a tone is audible when ``ΔL_a`` exceeds
0 dB.

* a title and the standard-basis line (measurement standard + the IEC 61400-11
  tonality method);
* an optional metadata header grid, rendered only for the fields supplied on
  the :class:`ReportMetadata`: the source/situation (``specimen``), the client,
  the measurement position (``test_room``), the instrumentation and the date;
* a two-panel body with the critical-band / masking analysis in a compact
  metrics table on the left (tone frequency, critical bandwidth, tone level
  ``L_pt``, masking-noise level ``L_pn``, tonality ``ΔL_tn``, audibility
  criterion ``L_a`` and tonal audibility ``ΔL_a``) beside the result's own
  narrowband-spectrum plot with the critical band, masking level and tone
  marked (the result's :meth:`plot`);
* a boxed single-number result: the decisive tonal audibility ``ΔL_a`` and the
  tone frequency, with the audibility decision the result carries;
* an optional PASS/FAIL verdict row when a maximum acceptable tonal audibility
  is supplied via the metadata ``requirement`` (read as the maximum acceptable
  ``ΔL_a``, a lower audibility being better); and
* a footer identity/disclaimer block.

This is a non-band fiche: the plot self-scales (``y_top=None``) rather than
using the fixed dB axis of the insulation fiches. The quantity-independent
skeleton lives in :mod:`._layout`; this module only holds the IEC 61400-11
tonality specifics. reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    _VERDICT_BAD_HEX,
    _VERDICT_OK_HEX,
    build_document,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    measurement_basis_style,
    metrics_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..environmental.wind_turbine_noise import WindTurbineTonalityResult


def _fmt(value: float, language: str, decimals: int = 1) -> str:
    """A quantity rounded to ``decimals`` decimals, localised separator."""
    return format_number(float(value), language, decimals=decimals)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _audible_as_displayed(result: WindTurbineTonalityResult) -> bool:
    """Whether the fiche's decision reads the tone as audible.

    The box, verdict and decision text all commit to the tonal audibility
    rounded exactly as displayed, so the audibility decision is taken on that
    same rounded value rather than on the result's raw ``is_audible`` flag;
    otherwise a raw ``ΔL_a`` of, say, 0.03 dB (audible) would print "0.0 dB
    > 0", contradicting its own number at the 0 dB boundary. A tone must still
    have been identified (subclause 9.5.4) for any audibility decision to apply.
    """
    return (
        result.has_identified_tone and display_round(result.tonal_audibility) > 0.0
    )


def _metadata_pairs(
    metadata: ReportMetadata | None, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the tonality header grid.

    A wind-turbine tonality assessment identifies the source/situation whose
    noise was analysed (``specimen``), the client, the measurement position
    (``test_room``) and the instrumentation; only supplied fields are returned.
    """
    if metadata is None:
        return []
    specs: list[tuple[str, str | None]] = [
        (t("Source / situation", language), _esc(metadata.specimen)),
        (t("Client", language), _esc(metadata.client)),
        (t("Measurement position", language), _esc(metadata.test_room)),
        (t("Instrumentation", language), _esc(metadata.instrumentation)),
        (t("Date of test", language), _esc(metadata.test_date)),
    ]
    return [(label, value) for label, value in specs if value]


def _metric_rows(
    result: WindTurbineTonalityResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The critical-band / masking analysis shown in the left-hand table.

    Lists the tone frequency and the chain of Formulae 30-34: the critical
    bandwidth, the tone level ``L_pt``, the masking-noise level ``L_pn``, the
    tonality ``ΔL_tn``, the audibility criterion ``L_a`` and the tonal
    audibility ``ΔL_a``.
    """
    return [
        (
            t("Tone frequency f [Hz]", language),
            _fmt(result.tone_frequency, language),
        ),
        (
            t("Critical bandwidth [Hz]", language),
            _fmt(result.critical_bandwidth, language),
        ),
        ("L<sub>pt</sub> [dB]", _fmt(result.tone_level, language)),
        ("L<sub>pn</sub> [dB]", _fmt(result.masking_level, language)),
        ("&#916;L<sub>tn</sub> [dB]", _fmt(result.tonality, language)),
        ("L<sub>a</sub> [dB]", _fmt(result.audibility_criterion, language)),
        (
            "&#916;L<sub>a</sub> [dB]",
            _fmt(display_round(result.tonal_audibility), language),
        ),
    ]


def _basis_line(measurement_standard: str | None, language: str = "en") -> str:
    """The standard-basis line, with the measurement standard when supplied."""
    if measurement_standard:
        return t(
            "{standard} tonal audibility of a wind turbine per "
            "IEC 61400-11:2012+A1:2018 (subclauses 9.5.2 to 9.5.8); a tone is "
            "audible when the tonal audibility &#916;L<sub>a</sub> exceeds 0 dB.",
            language,
        ).format(standard=html.escape(measurement_standard))
    return t(
        "Tonal audibility of a wind turbine per IEC 61400-11:2012+A1:2018 "
        "(subclauses 9.5.2 to 9.5.8); a tone is audible when the tonal audibility "
        "&#916;L<sub>a</sub> exceeds 0 dB.",
        language,
    )


def _statement(
    result: WindTurbineTonalityResult, language: str = "en"
) -> tuple[str, list[str]]:
    """The boxed decisive tonal audibility ``ΔL_a`` and the tone frequency.

    The tonal audibility is rounded exactly as displayed so the box can never
    contradict its own decision at the 0 dB boundary. The extended terms name
    the tonality and audibility criterion, and the audibility decision the
    result carries.
    """
    delta = display_round(result.tonal_audibility)
    statement = t(
        "Tonal audibility &#916;L<sub>a</sub> = <b>{dl} dB</b> &nbsp; "
        "tone f = <b>{f} Hz</b>",
        language,
    ).format(dl=_fmt(delta, language), f=_fmt(result.tone_frequency, language))
    extended = [
        t(
            "Tonality &#916;L<sub>tn</sub> = {tn} dB "
            "(L<sub>pt</sub> &#8722; L<sub>pn</sub>)",
            language,
        ).format(tn=_fmt(result.tonality, language)),
        t("Audibility criterion L<sub>a</sub> = {la} dB", language).format(
            la=_fmt(result.audibility_criterion, language)
        ),
        _decision_phrase(result, language),
    ]
    return statement, extended


def _decision_phrase(
    result: WindTurbineTonalityResult, language: str = "en"
) -> str:
    """A short coloured phrase carrying the audibility decision.

    When no tone was identified in the critical band the result carries no
    standard tonal audibility, so the phrase says so; otherwise it states
    whether the tone is audible.
    """
    if not result.has_identified_tone:
        return (
            f"<font color='{_MUTED_HEX}'>"
            f"{t('Decision: no tone identified', language)}</font>"
        )
    if _audible_as_displayed(result):
        return (
            f"<font color='{_VERDICT_BAD_HEX}'>&#9679; "
            f"{t('Decision: tone audible', language)}</font>"
        )
    return (
        f"<font color='{_VERDICT_OK_HEX}'>&#9679; "
        f"{t('Decision: tone not audible', language)}</font>"
    )


def _verdict(
    result: WindTurbineTonalityResult, requirement: float, language: str = "en"
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a supplied maximum tonal audibility.

    The ``requirement`` is read as the maximum acceptable tonal audibility
    ``ΔL_a`` in dB (a lower audibility is better, as a more audible tone is more
    intrusive); the assessment passes when the tonal audibility, rounded as
    displayed, is at or below it.
    """
    delta = display_round(result.tonal_audibility)
    passed = delta <= requirement + 1e-9
    text = t(
        "&#916;L<sub>a</sub> = {dl} dB, required &#8804; {req} dB", language
    ).format(dl=_fmt(delta, language), req=_fmt(requirement, language))
    return text, passed


def _decision_note(
    result: WindTurbineTonalityResult, language: str = "en"
) -> str:
    """The measurement-basis note stating the audibility decision."""
    delta = _fmt(display_round(result.tonal_audibility), language)
    if not result.has_identified_tone:
        return t(
            "No tone was identified in the critical band, so the spectrum "
            "carries no tonal audibility and is excluded from the standardised "
            "averaging over the spectra of a wind-speed bin (IEC 61400-11:2012 "
            "subclause 9.5.1).",
            language,
        )
    if _audible_as_displayed(result):
        return t(
            "The tone is audible (decisive &#916;L<sub>a</sub> = {dl} dB "
            "&gt; 0): the tonality rises above the audibility criterion "
            "(IEC 61400-11:2012+A1:2018, Formulae 33-34).",
            language,
        ).format(dl=delta)
    return t(
        "The tone is not audible (decisive &#916;L<sub>a</sub> = {dl} dB "
        "&#8804; 0): the tonality stays at or below the audibility criterion "
        "(IEC 61400-11:2012+A1:2018, Formulae 33-34).",
        language,
    ).format(dl=delta)


def render_wind_turbine_tonality_report(
    result: WindTurbineTonalityResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a wind-turbine tonal audibility assessment fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.environmental.wind_turbine_noise.WindTurbineTonalityResult`
        carrying the tone/masking levels, the critical band and the tonal
        audibility.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata` supplying the header
        identity (``specimen`` the source/situation, ``client``, ``test_room``
        the measurement position, ``instrumentation`` and ``test_date``) and the
        footer identity. A supplied ``requirement`` is read as the maximum
        acceptable tonal audibility ``ΔL_a`` in dB.
    :param verbose: Accepted for signature parity with the other fiches; the
        metrics table already shows the full Formula 30-34 chain, so it has no
        effect.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # the metrics table already shows the full chain; kept for parity
    try:
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Wind-turbine tonal audibility assessment", language)
    measurement_standard = (
        metadata.measurement_standard if metadata is not None else None
    )

    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(_basis_line(measurement_standard, language), basis_style),
    ]

    header_pairs = _metadata_pairs(metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(t("Critical-band analysis", language), caption_style),
        metrics_table(
            _metric_rows(result, language), col_widths=[38 * mm, 18 * mm]
        ),
    ]
    # Non-band plot (the narrowband spectrum with the critical band, masking
    # level and tone): self-scaling axis, drawn by the result's own ``plot``.
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, figsize=(5.8, 4.4), language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    statement, extended = _statement(result, language)
    flow.append(result_box(statement, styles, accent, extended))

    if metadata is not None and metadata.requirement is not None:
        text, passed = _verdict(result, metadata.requirement, language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(Paragraph(_decision_note(result, language), basis_strip_style))
    flow.append(
        Paragraph(
            t(
                "The tonal audibility &#916;L<sub>a</sub> = &#916;L<sub>tn</sub> "
                "&#8722; L<sub>a</sub> is the amount by which the tonality rises "
                "above the audibility criterion (IEC 61400-11:2012+A1:2018, "
                "Formulae 33-34); the tonality &#916;L<sub>tn</sub> = "
                "L<sub>pt</sub> &#8722; L<sub>pn</sub> compares the tone level "
                "with the masking-noise level within the critical band about the "
                "tone (Formulae 30-32).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
