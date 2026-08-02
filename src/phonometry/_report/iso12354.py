#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ISO 12354-1/-2/-3 predicted building sound-insulation fiche (reportlab).

Renders a
:class:`~phonometry.building.prediction.simplified_model.AirbornePredictionResult`
(predicted apparent sound reduction index ``R'`` between rooms, EN/ISO 12354-1),
an
:class:`~phonometry.building.prediction.simplified_model.ImpactPredictionResult`
(predicted apparent normalized impact sound pressure level ``L'n``, EN/ISO
12354-2) or a
:class:`~phonometry.building.prediction.facade.FacadePredictionResult`
(predicted standardized level difference of a facade ``D2m,nT``, EN/ISO
12354-3) to a one-page **prediction** report. Unlike the measurement fiches
(:mod:`.iso10140`, :mod:`.iso16283`, :mod:`.iso15186`), the reported result is
an *estimate* of the in-situ performance computed from the laboratory
performance of the building elements plus the flanking transmission across the
junctions (the simplified single-number model, EN 12354-1 Clause 4.4 airborne /
EN 12354-2 Clause 4.3 impact) or, for the facade, from the energy summation of
the envelope elements' transmission factors and the room geometry (EN 12354-3
Formula 13); the sheet states this explicitly and is clearly labelled a
prediction, never a measurement.

The simplified model works directly on the elements' single-number ratings and
returns the apparent weighted rating (``R'w`` / ``L'n,w``) without an
intermediate one-third-octave spectrum, so this fiche has no per-band
measured-versus-shifted-reference curve to draw and therefore does not reuse the
per-band :func:`._insulation_fiche.render_insulation_fiche` skeleton. It reuses
the same quantity-independent building blocks that skeleton is itself built on
(:mod:`._layout`: the title/basis styles, the metadata header grid, a compact
left-hand metrics table beside the result's own vector plot in a two-panel body,
the boxed single number, the optional requirement verdict and the footer), the
layout the EPNL prediction fiche (:mod:`.annex16_epnl`) also uses.

reportlab, matplotlib and svglib are soft dependencies imported lazily
(reportlab and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._i18n import format_number, t
from ._layout import (
    _ACCENT_HEX,
    _MUTED_HEX,
    _REPORTLAB_HINT,
    build_document,
    display_round,
    document_styles,
    fmt_num,
    footer_flow,
    grid_table,
    metrics_table,
    render_figure_drawing,
    result_box,
    two_panel_body,
    verdict_flow,
)
from .iso717 import _metadata_pairs
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..building.prediction.detailed_model import (
        DetailedAirborneResult,
        DetailedImpactResult,
    )
    from ..building.prediction.facade import FacadePredictionResult
    from ..building.prediction.simplified_model import (
        AirbornePredictionResult,
        ImpactPredictionResult,
    )

#: Model standard deviation of the simplified single-number prediction, stated
#: for reference in the method statement (EN 12354-1:2000 Clause 5, EN 12354-2
#: Clause 6): about 2 dB.
_MODEL_SD_DB = 2

#: Method statement of the simplified single-number fiches, below the boxed
#: rating. ``{sd}`` takes :data:`_MODEL_SD_DB`.
_SIMPLIFIED_STATEMENT = (
    "Predicted (estimated) result computed from the building "
    "elements' performance by the EN/ISO 12354 simplified "
    "single-number model; it is not a measurement. The reported "
    "standard deviation of the model is about {sd} dB."
)

#: Model standard deviation bounds of the *detailed* per-band model for
#: buildings with homogeneous elements (ISO 12354-1:2017 Clause 5).
_DETAILED_MODEL_SD_RANGE = (1.5, 2.5)


def _detailed_model_sd(language: str) -> str:
    """The detailed-model standard-deviation range, localised.

    Every other number on a fiche goes through :func:`format_number`, so the
    range is built here rather than written out, and the English sheet prints
    "1.5 dB to 2.5 dB" where the Spanish one prints "1,5 dB a 2,5 dB".
    """
    low, high = _DETAILED_MODEL_SD_RANGE
    return (
        f"{format_number(low, language, decimals=1)} dB "
        f"{t('to', language)} {format_number(high, language, decimals=1)} dB"
    )


def _prediction_verdict(
    rating: float,
    requirement: float,
    rating_symbol: str,
    *,
    is_impact: bool,
    language: str,
) -> tuple[str, bool]:
    """Verdict text and PASS flag for a predicted rating against a requirement.

    Airborne insulation passes at or above the requirement; impact level passes
    at or below it (a lower impact level is better). The comparison is on the
    displayed integer rating so a printed number can never contradict the
    verdict at the boundary.
    """
    value = display_round(rating, 0)
    req_text = fmt_num(requirement, language)
    if is_impact:
        passed = value <= requirement
        template = t("{sym} = {rating} dB, required &#8804; {req} dB", language)
    else:
        passed = value >= requirement
        template = t("{sym} = {rating} dB, required &#8805; {req} dB", language)
    text = template.format(
        sym=rating_symbol,
        rating=format_number(value, language, decimals=0),
        req=req_text,
    )
    return text, passed


def _render_prediction_fiche(
    *,
    result: Any,
    path: str,
    title_key: str,
    basis_key: str,
    rating_value: float,
    rating_symbol: str,
    left_caption_key: str,
    metric_rows: list[tuple[str, str]],
    is_impact: bool,
    metadata: ReportMetadata | None,
    language: str,
    basis_values: dict[str, str] | None = None,
    statement: tuple[str, dict[str, Any]] = (_SIMPLIFIED_STATEMENT, {}),
) -> str:
    """Render a predicted-insulation fiche to a PDF at ``path``.

    Shared body of every prediction fiche: title and prediction-basis line, an
    optional metadata header, a two-panel body (the left-hand model-term metrics
    table beside the result's own plot), the boxed single-number rating, the
    prediction statement, an optional requirement verdict and the footer.

    ``basis_values`` fills ``{}`` placeholders in the (already translated) basis
    line, so a fiche can state its computed apparent-index values in prose; pass
    ``None`` when the basis line is a plain sentence. ``statement`` is the
    ``(text, placeholder values)`` pair of the method statement below the boxed
    rating, defaulting to the simplified model's; the detailed fiches pass
    their own.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import Spacer

        from ._layout import fiche_paragraph as Paragraph
    except ImportError as exc:
        raise ImportError(_REPORTLAB_HINT) from exc
    accent = colors.HexColor(_ACCENT_HEX)

    styles, title_style, basis_style, caption_style = document_styles(accent)
    title_text = t(title_key, language)
    basis_text = t(basis_key, language)
    if basis_values is not None:
        basis_text = basis_text.format(**basis_values)
    flow: list[Any] = [
        Paragraph(title_text, title_style),
        Paragraph(basis_text, basis_style),
    ]

    # Metadata header grid (the same room/element grid the insulation fiches
    # share; only the supplied fields render).
    if metadata is not None and not metadata.is_empty():
        identity, conditions = _metadata_pairs(metadata, language)
        header_pairs = identity + conditions
        if header_pairs:
            flow.append(Spacer(1, 3))
            flow.append(grid_table(header_pairs))
    flow.append(Spacer(1, 8))

    # Two-panel body: the model-term metrics table on the left, the result's own
    # vector plot (self-scaling bar chart) on the right.
    left_cell = [
        Paragraph(t(left_caption_key, language), caption_style),
        metrics_table(metric_rows, col_widths=[34 * mm, 22 * mm]),
    ]
    plot_drawing = render_figure_drawing(
        result.plot, 116 * mm, y_top=None, figsize=(6.4, 5.8), language=language
    )
    flow.append(two_panel_body(left_cell, plot_drawing))
    flow.append(Spacer(1, 8))

    # Boxed single-number predicted rating (an integer dB, as ISO 717 rates).
    rating_text = format_number(display_round(rating_value, 0), language, decimals=0)
    flow.append(
        result_box(f"{rating_symbol} = <b>{rating_text} dB</b>", styles, accent)
    )

    statement_style = ParagraphStyle(
        "prediction_statement", parent=styles["Normal"], fontSize=8.5,
        textColor=colors.HexColor(_MUTED_HEX), spaceBefore=4,
    )
    statement_key, statement_values = statement
    values = statement_values or {"sd": _MODEL_SD_DB}
    flow.append(
        Paragraph(t(statement_key, language).format(**values), statement_style)
    )

    if metadata is not None and metadata.requirement is not None:
        text, passed = _prediction_verdict(
            rating_value, float(metadata.requirement), rating_symbol,
            is_impact=is_impact, language=language,
        )
        flow.extend(verdict_flow(text, passed, styles, language))

    flow.extend(footer_flow(
        metadata,
        language,
        disclaimer=(
            "Predicted result: the values relate only to the modelled "
            "configuration, not to a tested specimen."
        ),
    ))

    return build_document(path, flow, title_text)


def render_iso12354_airborne_report(
    result: AirbornePredictionResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a predicted apparent airborne insulation fiche (EN/ISO 12354-1).

    :param result: The
        :class:`~phonometry.building.prediction.simplified_model.AirbornePredictionResult`
        carrying the predicted apparent weighted index ``R'w``, the direct-path
        index ``RDd,w`` and the per-path contributions.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table lists every transmission path
        (direct and flanking) with its share of the transmitted energy; when
        ``False`` it lists only the direct path and each flanking path's
        weighted index.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    metric_rows: list[tuple[str, str]] = []
    for contribution in result.paths:
        value = fmt_num(contribution.r_w, language)
        if verbose:
            share = format_number(
                100.0 * contribution.fraction, language, decimals=1
            )
            value = f"{value} &#183; {share}%"
        metric_rows.append((contribution.label, value))

    caption_key = (
        "Path R<sub>ij,w</sub> [dB] and energy share"
        if verbose
        else "Transmission paths R<sub>ij,w</sub> [dB]"
    )
    return _render_prediction_fiche(
        result=result,
        path=path,
        title_key="Predicted airborne sound insulation between rooms",
        basis_key=(
            "Predicted apparent sound reduction index R&#8242; "
            "(direct and flanking transmission) estimated in accordance with "
            "EN 12354-1:2000 (simplified single-number model, Clause 4.4). "
            "This is a prediction from element data, not a measurement. "
            "R&#8242;<sub>w</sub> per ISO 717-1."
        ),
        rating_value=result.r_prime_w,
        rating_symbol="R&#8242;<sub>w</sub>",
        left_caption_key=caption_key,
        metric_rows=metric_rows,
        is_impact=False,
        metadata=metadata,
        language=language,
    )


def render_iso12354_impact_report(
    result: ImpactPredictionResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a predicted apparent impact insulation fiche (EN/ISO 12354-2).

    :param result: The
        :class:`~phonometry.building.prediction.simplified_model.ImpactPredictionResult`
        carrying the predicted apparent weighted level ``L'n,w`` and the
        Formula (21) terms (the bare-floor equivalent level ``Ln,w,eq``, the
        covering improvement ``ΔLw`` and the flanking correction ``K``).
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: Accepted for a uniform ``.report()`` signature; the impact
        fiche has a single body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # uniform signature; the impact fiche has one body layout
    metric_rows: list[tuple[str, str]] = [
        (
            t("Bare-floor level L<sub>n,w,eq</sub>", language),
            fmt_num(result.ln_w_eq, language),
        ),
        (
            t("Covering improvement &#916;L<sub>w</sub>", language),
            fmt_num(result.delta_l_w, language),
        ),
        (
            t("Flanking correction K", language),
            fmt_num(result.k_correction, language),
        ),
    ]
    return _render_prediction_fiche(
        result=result,
        path=path,
        title_key="Predicted impact sound insulation of floors",
        basis_key=(
            "Predicted apparent normalized impact sound pressure level "
            "L&#8242;<sub>n</sub> (bare floor, covering and flanking) estimated "
            "in accordance with EN 12354-2:2000 (simplified single-number "
            "model, Clause 4.3). This is a prediction from element data, not a "
            "measurement. L&#8242;<sub>n,w</sub> per ISO 717-2."
        ),
        rating_value=result.l_prime_n_w,
        rating_symbol="L&#8242;<sub>n,w</sub>",
        left_caption_key="Impact level prediction - Formula (21) terms [dB]",
        metric_rows=metric_rows,
        is_impact=True,
        metadata=metadata,
        language=language,
    )


_DETAILED_STATEMENT = (
    "Predicted (estimated) result computed band by band from the building "
    "elements' performance by the EN/ISO 12354 detailed model; it is not a "
    "measurement. For buildings with homogeneous elements the prediction of "
    "the single-number rating carries no bias error and a standard deviation "
    "of {sd}."
)


def _detailed_path_rows(
    result: Any, verbose: bool, language: str
) -> list[tuple[str, str]]:
    """Per-path rows of a detailed-model fiche: energy share, largest first.

    Each path's figure is the *mean of its per-band shares* over the spectrum
    supplied; the per-band shares already partition the energy of each band, so
    an energy-weighted share across bands cannot be recovered from them alone.
    The mean still ranks the paths in the order a consultant would treat them.
    In verbose mode each row also names the band in which that path peaks.
    """
    import numpy as np

    fractions = np.atleast_2d(np.asarray(result.fractions, dtype=np.float64))
    totals = fractions.mean(axis=1)
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    rows: list[tuple[str, str]] = []
    for k in np.argsort(-totals):
        value = format_number(100.0 * float(totals[k]), language, decimals=1)
        text = f"{value}%"
        if verbose:
            peak = freqs[int(np.argmax(fractions[k]))]
            text = f"{text} &#183; {fmt_num(float(peak), language)} Hz"
        rows.append((result.paths[int(k)].label, text))
    return rows


def _detailed_rating(result: Any) -> Any:
    """Return the result's ISO 717 rating, refusing an unrated spectrum."""
    if result.rating is None:
        raise ValueError(
            "A detailed prediction report needs the ISO 717 single-number "
            "rating: build the result on bands covering 100 Hz to 3150 Hz "
            "(one-third octave) or 125 Hz to 2000 Hz (octave)."
        )
    return result.rating


def render_iso12354_detailed_airborne_report(
    result: DetailedAirborneResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a detailed per-band airborne prediction fiche (EN/ISO 12354-1).

    :param result: The
        :class:`~phonometry.building.prediction.detailed_model.DetailedAirborneResult`
        carrying the per-band ``R'``, the per-band path contributions and the
        ISO 717-1 rating.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True`` each path row also names the band in which
        that path contributes most.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If the result carries no ISO 717-1 rating.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    rating = _detailed_rating(result)
    return _render_prediction_fiche(
        result=result,
        path=path,
        title_key="Predicted airborne sound insulation between rooms",
        basis_key=(
            "Predicted apparent sound reduction index R&#8242; per frequency "
            "band (direct and flanking transmission, in-situ element and "
            "junction data) estimated in accordance with ISO 12354-1:2017 "
            "(detailed model, Clause 4.2). This is a prediction from element "
            "data, not a measurement. R&#8242;<sub>w</sub> per ISO 717-1."
        ),
        rating_value=float(rating.rating),
        rating_symbol="R&#8242;<sub>w</sub>",
        left_caption_key="Transmission paths - share of energy",
        metric_rows=_detailed_path_rows(result, verbose, language),
        is_impact=False,
        metadata=metadata,
        language=language,
        statement=(_DETAILED_STATEMENT, {"sd": _detailed_model_sd(language)}),
    )


def render_iso12354_detailed_impact_report(
    result: DetailedImpactResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a detailed per-band impact prediction fiche (EN/ISO 12354-2).

    :param result: The
        :class:`~phonometry.building.prediction.detailed_model.DetailedImpactResult`
        carrying the per-band ``L'n``, the per-band path contributions and the
        ISO 717-2 rating.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True`` each path row also names the band in which
        that path contributes most.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If the result carries no ISO 717-2 rating.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    rating = _detailed_rating(result)
    return _render_prediction_fiche(
        result=result,
        path=path,
        title_key="Predicted impact sound insulation of floors",
        basis_key=(
            "Predicted apparent normalized impact sound pressure level "
            "L&#8242;<sub>n</sub> per frequency band (direct and flanking "
            "transmission, in-situ element and junction data) estimated in "
            "accordance with ISO 12354-2:2017 (detailed model, Clause 4.2). "
            "This is a prediction from element data, not a measurement. "
            "L&#8242;<sub>n,w</sub> per ISO 717-2."
        ),
        rating_value=float(rating.rating),
        rating_symbol="L&#8242;<sub>n,w</sub>",
        left_caption_key="Transmission paths - share of energy",
        metric_rows=_detailed_path_rows(result, verbose, language),
        is_impact=True,
        metadata=metadata,
        language=language,
        statement=(_DETAILED_STATEMENT, {"sd": _detailed_model_sd(language)}),
    )


def _facade_energy_shares(result: FacadePredictionResult) -> dict[str, float]:
    """Each element's share (%) of the facade's transmitted acoustic energy.

    The transmission factor of an element is ``&#964; = 10^(-Rp/10)`` from its
    partial index ``Rp`` (already area-weighted); summed over the bands and
    normalised to the whole envelope, the shares add up to 100 % and single out
    the limiting element (an air inlet or a light window rather than the wall).
    """
    import numpy as np

    taus = {
        name: 10.0 ** (-partial / 10.0)
        for name, partial in result.element_r.items()
    }
    total = float(np.sum([t.sum() for t in taus.values()]))
    return {name: 100.0 * float(t.sum()) / total for name, t in taus.items()}


def render_iso12354_facade_report(
    result: FacadePredictionResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a predicted facade sound insulation fiche (EN/ISO 12354-3).

    :param result: The
        :class:`~phonometry.building.prediction.facade.FacadePredictionResult`
        carrying the predicted standardized level difference ``D2m,nT``, its
        single number ``D2m,nT,w`` (with ``R'tr,s,w`` and ``Ctr``) and the
        per-element partial indices ``Rp``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table annexes each façade element's
        share of the transmitted sound energy; when ``False`` it lists only
        each element's weighted partial index ``Rp,w``.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ValueError: If the result lacks the ISO 717-1 single-number ratings
        (it must be built on the 5 octave or 16 one-third-octave bands).
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    if result.d_2m_nt_w is None or result.r_tr_s_w is None or result.c_tr is None:
        raise ValueError(
            "A facade prediction report needs the ISO 717-1 single-number "
            "ratings: build the result on the 5 octave or 16 one-third-octave "
            "bands (pass matching per-band element data and 'bands' to "
            "facade_sound_reduction)."
        )
    from ..building.measurement.insulation import weighted_rating

    shares = _facade_energy_shares(result) if verbose else {}
    metric_rows: list[tuple[str, str]] = []
    for name, partial in result.element_r.items():
        value = fmt_num(weighted_rating(partial).rating, language)
        if verbose:
            share = format_number(shares[name], language, decimals=1)
            value = f"{value} &#183; {share}%"
        metric_rows.append((name, value))

    caption_key = (
        "Facade element R<sub>p,w</sub> [dB] and energy share"
        if verbose
        else "Facade elements R<sub>p,w</sub> [dB]"
    )
    return _render_prediction_fiche(
        result=result,
        path=path,
        title_key="Predicted facade sound insulation",
        basis_key=(
            "Predicted standardized level difference of a facade "
            "D<sub>2m,nT</sub> (the envelope elements' apparent sound reduction "
            "index R&#8242; combined energetically with the room geometry) "
            "estimated in accordance with EN 12354-3:2000 (simplified "
            "model, Formula 13). This is a prediction from element data, not a "
            "measurement. D<sub>2m,nT,w</sub> (with apparent index "
            "R&#8242;<sub>tr,s,w</sub> = {rtrs} dB and C<sub>tr</sub> = {ctr} "
            "dB) per ISO 717-1."
        ),
        rating_value=float(result.d_2m_nt_w),
        rating_symbol="D<sub>2m,nT,w</sub>",
        left_caption_key=caption_key,
        metric_rows=metric_rows,
        is_impact=False,
        metadata=metadata,
        language=language,
        basis_values={
            "rtrs": fmt_num(float(result.r_tr_s_w), language),
            "ctr": fmt_num(float(result.c_tr), language),
        },
    )
