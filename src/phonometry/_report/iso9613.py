#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 9613-2 outdoor sound propagation prediction fiches (reportlab renderer).

Renders the two outdoor-propagation result types to one-page **prediction**
reports (clearly labelled predictions, never measurement certificates):

* :func:`render_outdoor_attenuation_report` for an
  :class:`~phonometry.environmental.outdoor_propagation.OutdoorAttenuation`:
  the ISO 9613-2:1996 octave-band attenuation breakdown (geometrical divergence
  ``Adiv``, atmospheric absorption ``Aatm``, ground effect ``Agr`` and
  screening ``Abar``) and, from the source sound power the result was composed
  with, the downwind octave-band level ``LfT(DW)`` (Eq. (3)) and the boxed
  A-weighted downwind level ``LAT(DW)`` at the receiver. A stacked layout: the
  wide per-band term table above the attenuation-breakdown plot.

* :func:`render_barrier_insertion_loss_report` for a
  :class:`~phonometry.environmental.ground_barriers.BarrierInsertionLoss`: the
  per-band barrier insertion loss ``IL`` and its mean over the octave bands. A
  two-panel layout: the per-band table beside the insertion-loss spectrum.
  Its diffraction model is the wave-theoretic rigid-screen solution (``method
  = "exact"``) or the Kurze-Anderson closed form (``method = "kurze_anderson"``),
  a wave-acoustics complement to the tabulated ISO 9613-2 screening term, and
  the basis line cites the actual model, not ISO 9613-2's ``Dz`` formula.

Both fiches state the meteorological/ground assumptions of the prediction. The
quantity-independent skeleton lives in :mod:`._layout`; this module holds only
the ISO 9613-2 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _REPORTLAB_HINT,
    analysis_cell_styles,
    build_document,
    display_round,
    document_styles,
    footer_flow,
    grid_table,
    measurement_basis_style,
    render_figure_drawing,
    result_box,
    stacked_table,
    two_panel_body,
    verdict_flow,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..environmental.ground_barriers import BarrierInsertionLoss
    from ..environmental.outdoor_propagation import (
        OutdoorAttenuation,
        SourceEmission,
    )

    NDArrayF = NDArray[np.float64]
else:  # pragma: no cover - runtime alias only
    NDArrayF = Any

#: Reference distance ``d0`` in the divergence term (ISO 9613-2:1996, Eq. (7)).
_D0 = 1.0

#: The frequency-column header (English key), shared by both per-band tables.
_FREQ_HEADER = "f [Hz]"


class _AttenuationLevels(NamedTuple):
    """The per-band source power ``Lw``, downwind level ``LfT`` and A-weighting.

    Bundles the display quantities the attenuation fiche derives from a
    :class:`~phonometry.environmental.outdoor_propagation.SourceEmission`, so
    the table and the boxed A-weighted level share one computation.
    """

    sound_power_level: NDArrayF
    receiver_level: NDArrayF
    ck: NDArrayF


def _fmt(value: float, language: str, decimals: int = 1) -> str:
    """A quantity rounded to ``decimals`` decimals, localised separator."""
    return format_number(float(value), language, decimals=decimals)


def _esc(value: str | None) -> str | None:
    """HTML-escape an optional free-text metadata value."""
    return html.escape(value) if value else None


def _distance_from_divergence(a_div: float) -> float:
    """Recover the propagation distance ``d`` from ``Adiv`` (Eq. (7)).

    ``Adiv = 20 lg(d/d0) + 11`` inverts exactly to ``d = d0 10^((Adiv - 11)/20)``
    (``d0 = 1 m``), so the fiche can state the source-receiver distance the
    attenuation was computed for without the result carrying the geometry.
    """
    return float(_D0 * 10.0 ** ((a_div - 11.0) / 20.0))


def _a_weighting_octave(frequencies: NDArrayF) -> NDArrayF:
    """A-weighting band corrections ``Ck`` for the nominal octave bands.

    Reuses the single ISO 3744:2010 Annex E / IEC 61672-1 A-weighting table of
    the emission domain (a lazy sibling import keeps the report leaf free of
    module-level parent imports).
    """
    from ..emission.sound_power import _a_weighting_corrections

    return np.asarray(_a_weighting_corrections(frequencies), dtype=np.float64)


def _a_weighted_level(receiver_level: NDArrayF, ck: NDArrayF) -> float:
    """Energy-summed A-weighted downwind level ``LAT(DW)`` from the band levels.

    ``LAT = 10 lg( sum_k 10^((LfT_k + Ck)/10) )``, the standard A-weighted
    total of a set of octave-band sound pressure levels.
    """
    band_a = receiver_level + ck
    return float(10.0 * np.log10(np.sum(10.0 ** (band_a / 10.0))))


# --------------------------------------------------------------------------- #
# ISO 9613-2 attenuation prediction fiche
# --------------------------------------------------------------------------- #
def _attenuation_metadata_pairs(
    result: OutdoorAttenuation,
    metadata: ReportMetadata | None,
    language: str,
) -> list[tuple[str, str]]:
    """Ordered (label, value) pairs of the attenuation-fiche header grid.

    The propagation distance ``d`` is always shown, recovered from ``Adiv``;
    the source/situation, client, receiver position and meteorological
    conditions are shown only when supplied on the metadata.
    """
    distance = _distance_from_divergence(float(np.asarray(result.a_div)[0]))
    specs: list[tuple[str, str | None]] = []
    if metadata is not None:
        specs += [
            (t("Source / situation", language), _esc(metadata.specimen)),
            (t("Client", language), _esc(metadata.client)),
            (t("Receiver position", language), _esc(metadata.test_room)),
        ]
    specs.append((t("Distance d [m]", language), _fmt(distance, language, 0)))
    if metadata is not None:
        if metadata.temperature is not None:
            specs.append(
                (t("Air temperature [&#176;C]", language),
                 _fmt(metadata.temperature, language))
            )
        if metadata.relative_humidity is not None:
            specs.append(
                (t("Relative humidity [%]", language),
                 _fmt(metadata.relative_humidity, language, 0))
            )
        if metadata.pressure is not None:
            specs.append(
                (t("Pressure [kPa]", language), _fmt(metadata.pressure, language))
            )
        specs.append((t("Date of prediction", language), _esc(metadata.test_date)))
    return [(label, value) for label, value in specs if value]


def _attenuation_table(
    result: OutdoorAttenuation,
    levels: _AttenuationLevels | None,
    verbose: bool,
    language: str,
) -> Any:
    """The full-width per-band attenuation table (ISO 9613-2:1996, clause 7).

    One row per octave band: the divergence, atmospheric, ground and barrier
    terms and the total attenuation ``A``. When ``levels`` is supplied (a source
    emission was given) the source power level ``Lw`` and the downwind level
    ``LfT(DW)`` are added, and ``verbose`` appends the A-weighted band level
    ``L_A``, whose energy sum is the boxed ``LAT(DW)``.
    """
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, _, value_style = analysis_cell_styles("iso9613atten")
    freqs = np.asarray(result.frequencies, dtype=np.float64)

    term_headers = [
        "A<sub>div</sub>", "A<sub>atm</sub>", "A<sub>gr</sub>",
        "A<sub>bar</sub>", "A [dB]",
    ]
    if levels is None:
        headers = [t(_FREQ_HEADER, language), *term_headers]
        widths = [26.0, 29.0, 29.0, 29.0, 29.0, 30.0]
    else:
        headers = [t(_FREQ_HEADER, language), "L<sub>w</sub> [dB]", *term_headers,
                   "L<sub>fT</sub> [dB]"]
        widths = [22.0, 20.0, 20.0, 20.0, 20.0, 20.0, 22.0, 24.0]
        if verbose:
            headers.append("L<sub>A</sub> [dB(A)]")
            widths = [20.0, 18.0, 17.0, 17.0, 17.0, 17.0, 18.0, 20.0, 22.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i in range(freqs.size):
        row = [Paragraph(_fmt(freqs[i], language, 0), value_style)]
        if levels is not None:
            row.append(Paragraph(_fmt(levels.sound_power_level[i], language),
                                 value_style))
        row += [
            Paragraph(_fmt(result.a_div[i], language), value_style),
            Paragraph(_fmt(result.a_atm[i], language), value_style),
            Paragraph(_fmt(result.a_gr[i], language), value_style),
            Paragraph(_fmt(result.a_bar[i], language), value_style),
            Paragraph(_fmt(result.a_total[i], language), value_style),
        ]
        if levels is not None:
            row.append(Paragraph(_fmt(levels.receiver_level[i], language),
                                 value_style))
            if verbose:
                row.append(Paragraph(
                    _fmt(levels.receiver_level[i] + levels.ck[i], language),
                    value_style,
                ))
        data.append(row)
    return stacked_table(data, [w * mm for w in widths])


def _attenuation_statement(la_dw: float, language: str) -> str:
    """The boxed A-weighted downwind level ``LAT(DW)`` at the receiver."""
    return t(
        "A-weighted downwind level at the receiver "
        "L<sub>AT</sub>(DW) = <b>{la} dB</b>", language
    ).format(la=_fmt(la_dw, language))


def _breakdown_statement(result: OutdoorAttenuation, language: str) -> str:
    """The boxed octave-band range of the total attenuation ``A``.

    Reported when no source emission is supplied, so the fiche still closes on
    the headline result of the breakdown (the span of the total attenuation
    over the analysis bands) rather than a receiver level it cannot compute.
    """
    a_total = np.asarray(result.a_total, dtype=np.float64)
    return t(
        "Total attenuation A = <b>{lo} to {hi} dB</b> over the octave bands",
        language,
    ).format(
        lo=_fmt(float(np.min(a_total)), language),
        hi=_fmt(float(np.max(a_total)), language),
    )


def _attenuation_verdict(
    la_dw: float, requirement: float, language: str
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a declared maximum level.

    The ``requirement`` is read as the maximum acceptable A-weighted downwind
    level in dB; a lower level is better, so the prediction passes when the
    level, rounded exactly as displayed, is at or below the limit.
    """
    displayed = display_round(la_dw)
    passed = displayed <= requirement + 1e-9
    text = t(
        "L<sub>AT</sub>(DW) = {la} dB, required &#8804; {req} dB", language
    ).format(la=_fmt(la_dw, language), req=_fmt(requirement, language))
    return text, passed


def _resolve_levels(
    result: OutdoorAttenuation, emission: SourceEmission
) -> _AttenuationLevels:
    """Compose the per-band levels the fiche boxes from a source emission.

    Calls the shared
    :func:`~phonometry.environmental.outdoor_propagation._compose_receiver_level`
    so the report reuses the domain's Eq. (3) composition rather than
    reimplementing it.
    """
    from ..environmental.outdoor_propagation import _compose_receiver_level

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    lw = np.atleast_1d(np.asarray(emission.sound_power_level, dtype=np.float64))
    if lw.shape != freqs.shape:
        raise ValueError(
            "source_emission.sound_power_level must have one value per frequency "
            f"band (got {lw.size}, expected {freqs.size})."
        )
    receiver = _compose_receiver_level(
        lw, emission.directivity_index, emission.d_omega,
        np.asarray(result.a_total, dtype=np.float64), emission.cmet,
    )
    return _AttenuationLevels(
        sound_power_level=lw,
        receiver_level=receiver,
        ck=_a_weighting_octave(freqs),
    )


def render_outdoor_attenuation_report(
    result: OutdoorAttenuation,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
    source_emission: SourceEmission | None = None,
) -> str:
    """Render an ISO 9613-2 outdoor-propagation prediction fiche to ``path``.

    :param result: An
        :class:`~phonometry.environmental.outdoor_propagation.OutdoorAttenuation`
        (the per-band attenuation breakdown).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; a ``requirement`` is the
        maximum acceptable A-weighted downwind level (a lower level is better,
        used only when a ``source_emission`` is given).
    :param verbose: When True and a ``source_emission`` is supplied, the per-band
        table adds the A-weighted band level.
    :param language: ``"en"`` (default) or ``"es"``.
    :param source_emission: Optional
        :class:`~phonometry.environmental.outdoor_propagation.SourceEmission`;
        when supplied the fiche boxes the A-weighted downwind level at the
        receiver, otherwise it boxes the range of the total attenuation.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If a supplied ``source_emission`` sound power does not
        match the number of frequency bands.
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

    levels = (
        _resolve_levels(result, source_emission)
        if source_emission is not None
        else None
    )

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Predicted outdoor sound propagation", language)
    basis = t(
        "Octave-band attenuation and the downwind sound pressure level at the "
        "receiver predicted in accordance with ISO 9613-2:1996 (general method "
        "of calculation), for conditions favourable to propagation (downwind, "
        "or the equivalent moderate temperature inversion; clause 5). This is a "
        "prediction, not a measurement.",
        language,
    )
    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _attenuation_metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    flow.append(Paragraph(t("Attenuation breakdown", language), caption_style))
    flow.append(_attenuation_table(result, levels, verbose, language))
    flow.append(Spacer(1, 8))
    flow.append(
        render_figure_drawing(
            result.plot, 174 * mm, y_top=None, figsize=(9.2, 3.1),
            language=language,
        )
    )
    flow.append(Spacer(1, 8))

    if levels is not None:
        la_dw = _a_weighted_level(levels.receiver_level, levels.ck)
        flow.append(
            result_box(_attenuation_statement(la_dw, language), styles, accent)
        )
        if metadata is not None and metadata.requirement is not None:
            text, passed = _attenuation_verdict(
                la_dw, float(metadata.requirement), language
            )
            flow.extend(verdict_flow(text, passed, styles, language))
    else:
        flow.append(
            result_box(_breakdown_statement(result, language), styles, accent)
        )

    basis_strip_style = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "The attenuation A = A<sub>div</sub> + A<sub>atm</sub> + "
                "A<sub>gr</sub> + A<sub>bar</sub> (Eq. (4)); a negative "
                "A<sub>gr</sub> is a net gain from ground reflection.",
                language,
            ),
            basis_strip_style,
        )
    )
    if levels is not None:
        flow.append(
            Paragraph(
                t(
                    "The downwind level L<sub>fT</sub>(DW) = L<sub>w</sub> + "
                    "D<sub>c</sub> &#8722; A (Eq. (3)); L<sub>AT</sub>(DW) is its "
                    "A-weighted total.",
                    language,
                ),
                basis_strip_style,
            )
        )
    flow.append(
        Paragraph(
            t(
                "A<sub>misc</sub> (foliage, industrial sites, housing; Annex A) "
                "and reflections from vertical surfaces (clause 7.5) are not "
                "included. The method is accurate to within about 1 dB to 3 dB "
                "for broadband noise up to 1000 m (Table 5).",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)


# --------------------------------------------------------------------------- #
# Barrier insertion-loss prediction fiche
# --------------------------------------------------------------------------- #
#: The diffraction-model phrase for the basis line, keyed by the result method.
_BARRIER_MODEL: dict[str, str] = {
    "exact": (
        "the wave-theoretic rigid-screen diffraction model (the flat-wedge "
        "MacDonald / Hadden-Pierce solution)"
    ),
    "kurze_anderson": (
        "the Kurze-Anderson closed form of the Fresnel number (Kurze & "
        "Anderson, 1971)"
    ),
}


def _barrier_metadata_pairs(
    result: BarrierInsertionLoss,
    metadata: ReportMetadata | None,
    language: str,
) -> list[tuple[str, str]]:
    """Ordered (label, value) pairs of the barrier-fiche header grid."""
    ground = (
        t("with coherent ground", language)
        if result.ground
        else t("free field (no ground)", language)
    )
    specs: list[tuple[str, str | None]] = []
    if metadata is not None:
        specs += [
            (t("Source / situation", language), _esc(metadata.specimen)),
            (t("Client", language), _esc(metadata.client)),
            (t("Receiver position", language), _esc(metadata.test_room)),
        ]
    specs.append((t("Ground model", language), ground))
    if metadata is not None:
        specs.append((t("Date of prediction", language), _esc(metadata.test_date)))
    return [(label, value) for label, value in specs if value]


def _barrier_table(
    result: BarrierInsertionLoss, verbose: bool, language: str
) -> Any:
    """The per-band barrier insertion-loss table (``IL`` and, verbose, ``N``)."""
    from reportlab.lib.units import mm

    from ._layout import fiche_paragraph as Paragraph

    header_style, _, value_style = analysis_cell_styles("iso9613bar")
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    il = np.asarray(result.insertion_loss, dtype=np.float64)
    n = np.asarray(result.fresnel_number, dtype=np.float64)

    headers = [t(_FREQ_HEADER, language), "IL [dB]"]
    widths = [22.0, 24.0]
    if verbose:
        headers.insert(1, "N")
        widths = [18.0, 16.0, 20.0]

    data: list[list[Any]] = [[Paragraph(h, header_style) for h in headers]]
    for i in range(freqs.size):
        row = [
            Paragraph(_fmt(freqs[i], language, 0), value_style),
            Paragraph(_fmt(il[i], language), value_style),
        ]
        if verbose:
            row.insert(1, Paragraph(_fmt(n[i], language, 2), value_style))
        data.append(row)
    return stacked_table(data, [w * mm for w in widths])


def _barrier_statement(mean_il: float, language: str) -> str:
    """The boxed mean insertion loss over the octave bands."""
    return t(
        "Mean insertion loss (63 Hz to 8 kHz) IL = <b>{il} dB</b>", language
    ).format(il=_fmt(mean_il, language))


def _barrier_verdict(
    mean_il: float, requirement: float, language: str
) -> tuple[str, bool]:
    """Verdict text and PASS flag against a declared minimum insertion loss.

    The ``requirement`` is the minimum required mean insertion loss in dB; a
    higher insertion loss is better, so the prediction passes when the mean,
    rounded exactly as displayed, reaches or exceeds it.
    """
    displayed = display_round(mean_il)
    passed = displayed >= requirement - 1e-9
    text = t(
        "IL = {il} dB, required &#8805; {req} dB", language
    ).format(il=_fmt(mean_il, language), req=_fmt(requirement, language))
    return text, passed


def render_barrier_insertion_loss_report(
    result: BarrierInsertionLoss,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a barrier insertion-loss prediction fiche to ``path``.

    :param result: A
        :class:`~phonometry.environmental.ground_barriers.BarrierInsertionLoss`.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; a ``requirement`` is the
        minimum required mean insertion loss (a higher insertion loss is better).
    :param verbose: When True, the per-band table adds the Fresnel number ``N``.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
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

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title = t("Predicted barrier insertion loss", language)
    model = t(_BARRIER_MODEL.get(result.method, result.method), language)
    basis = t(
        "Barrier insertion loss IL predicted with {model}, a wave-acoustics "
        "complement to the tabulated ISO 9613-2:1996 screening term. This is a "
        "prediction, not a measurement.",
        language,
    ).format(model=model)
    flow: list[Any] = [
        Paragraph(title, title_style),
        Paragraph(basis, basis_style),
    ]

    header_pairs = _barrier_metadata_pairs(result, metadata, language)
    if header_pairs:
        flow.append(Spacer(1, 3))
        flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    left_cell = [
        Paragraph(t("Insertion loss per band", language), caption_style),
        _barrier_table(result, verbose, language),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, figsize=(6.4, 5.4), language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    mean_il = float(np.mean(np.asarray(result.insertion_loss, dtype=np.float64)))
    flow.append(result_box(_barrier_statement(mean_il, language), styles, accent))

    if metadata is not None and metadata.requirement is not None:
        text, passed = _barrier_verdict(mean_il, float(metadata.requirement), language)
        flow.extend(verdict_flow(text, passed, styles, language))

    basis_strip_style = measurement_basis_style()
    flow.append(
        Paragraph(
            t(
                "The insertion loss IL = 20 lg|p<sub>free</sub> / "
                "p<sub>barrier</sub>| is the level reduction the screen adds "
                "over the direct path; the Fresnel number N is positive for a "
                "receiver in the geometric shadow. The boxed value is the "
                "arithmetic mean over the octave bands; the A-weighted screening "
                "of a given source spectrum follows by summing the band levels.",
                language,
            ),
            basis_strip_style,
        )
    )
    flow.extend(footer_flow(metadata, language))

    return build_document(path, flow, title)
