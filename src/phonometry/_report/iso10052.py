#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 10052 survey-method field sound-insulation fiches (reportlab renderer).

Renders the survey (control) method results of
:mod:`phonometry.building.measurement.survey_insulation` (ISO 10052:2021, identical in the
harmonized EN ISO 10052:2004+A1:2010) to one-page field test reports:

* :class:`~phonometry.building.measurement.survey_insulation.SurveyAirborneResult` to a
  field airborne report of the standardized level difference ``DnT`` (Clause
  3.4) with the ISO 717-1 single number ``DnT,w (C; Ctr)``, or the apparent
  sound reduction index ``R'`` (Clause 3.6) with ``R'w (C; Ctr)``;
* :class:`~phonometry.building.measurement.survey_insulation.SurveyImpactResult` to a field
  impact report of the standardized impact level ``L'nT`` (Clause 3.8) with the
  ISO 717-2 single number ``L'nT,w (CI)``;
* :class:`~phonometry.building.measurement.survey_insulation.SurveyFacadeResult` to a field
  facade report of the standardized facade level difference ``D2m,nT`` (Clause
  3.14) with ``D2m,nT,w (C; Ctr)``.

Every survey quantity is a per-band curve rated by ISO 717 against a shifted
reference, exactly the shape of the shared two-panel insulation skeleton
(:func:`._insulation_fiche.render_insulation_fiche`), so it is reused unchanged;
the survey method works in octave bands (125 Hz to 2000 Hz, Clause 6.4), so the
table caption names the octave-band set and, with ``verbose=True``, the left
table shows the ISO 717 evaluation per band (the reported quantity, the shifted
reference and the unfavourable deviation).

reportlab, matplotlib and svglib are soft dependencies imported lazily by the
shared renderer (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._insulation_fiche import iso717_columns_builder, render_insulation_fiche

if TYPE_CHECKING:
    from ..building.measurement.insulation import (
        ImpactRatingResult,
        WeightedRatingResult,
    )
    from ..building.measurement.survey_insulation import (
        SurveyAirborneResult,
        SurveyFacadeResult,
        SurveyImpactResult,
    )
    from .metadata import ReportMetadata

#: Band count of the octave-band survey set (5 values, 125 Hz to 2000 Hz, vs
#: 16 one-third-octave values, 100 Hz to 3150 Hz); it picks the caption and the
#: standard-basis wording ISO 717-1:2020 Clause 5.3 / ISO 717-2:2020 Clause 4.4
#: require stating.
_N_OCTAVE_BANDS = 5

#: The survey-method field statement each report prints verbatim; it names the
#: control (survey) method so it is never conflated with the ISO 16283
#: engineering method.
_SURVEY_STATEMENT = (
    "Evaluation based on the ISO 10052 survey (control) method (octave bands, "
    "a single hand-held sound level meter swept through the room)."
)

#: Per-quantity fixed labels consumed by the shared insulation skeleton: title,
#: standard-basis line (a ``{bands}`` template naming the band set the curve
#: actually carries), the quantity symbol used in the table header, the rating
#: symbol of the boxed result and the plot y-axis label (mathtext).
_DNT_SPEC: dict[str, str] = {
    "title": "Field airborne sound insulation between rooms",
    "basis": (
        "Standardized level difference D<sub>nT</sub> measured in accordance "
        "with ISO 10052:2021 (survey method, {bands}). Rating per "
        "ISO 717-1:2020."
    ),
    "symbol": "D<sub>nT</sub>",
    "rating_symbol": "D<sub>nT,w</sub>",
    "ylabel": "$D_{nT}$ [dB]",
    "statement": _SURVEY_STATEMENT,
}
_R_PRIME_SPEC: dict[str, str] = {
    "title": "Field airborne sound insulation between rooms",
    "basis": (
        "Apparent sound reduction index R&#8242; measured in accordance with "
        "ISO 10052:2021 (survey method, {bands}). Rating per "
        "ISO 717-1:2020."
    ),
    "symbol": "R&#8242;",
    "rating_symbol": "R&#8242;<sub>w</sub>",
    "ylabel": "$R'$ [dB]",
    "statement": _SURVEY_STATEMENT,
}
_L_NT_SPEC: dict[str, str] = {
    "title": "Field impact sound insulation of floors",
    "basis": (
        "Standardized impact sound pressure level L&#8242;<sub>nT</sub> "
        "measured in accordance with ISO 10052:2021 (survey method, {bands}) "
        "using the tapping machine. Rating per ISO 717-2:2020."
    ),
    "symbol": "L&#8242;<sub>nT</sub>",
    "rating_symbol": "L&#8242;<sub>nT,w</sub>",
    "ylabel": "$L'_{nT}$ [dB]",
    "statement": _SURVEY_STATEMENT,
}
_D_2M_NT_SPEC: dict[str, str] = {
    "title": "Field facade sound insulation",
    "basis": (
        "Standardized facade level difference D<sub>2m,nT</sub> measured in "
        "accordance with ISO 10052:2021 (survey method, {bands}). Rating "
        "per ISO 717-1:2020."
    ),
    "symbol": "D<sub>2m,nT</sub>",
    "rating_symbol": "D<sub>2m,nT,w</sub>",
    "ylabel": "$D_{2m,nT}$ [dB]",
    "statement": _SURVEY_STATEMENT,
}

#: The reportable airborne quantities of a survey airborne result, mapping the
#: ``quantity`` argument to its curve attribute, rating attribute and spec.
_AIRBORNE_QUANTITIES: dict[str, tuple[str, str, dict[str, str]]] = {
    "dnt": ("d_nt", "rating", _DNT_SPEC),
    "r_prime": ("r_prime", "r_prime_rating", _R_PRIME_SPEC),
}


def _render_survey(
    result: Any,
    rating: WeightedRatingResult | ImpactRatingResult,
    path: str,
    *,
    spec: dict[str, str],
    is_impact: bool,
    curve_attr: str,
    metadata: ReportMetadata | None,
    verbose: bool,
    language: str,
) -> str:
    """Drive the shared insulation skeleton with the survey band table.

    The survey method may be run in octave bands (5 values, 125 Hz to 2000 Hz)
    or one-third-octave bands (16 values, 100 Hz to 3150 Hz); the table
    caption *and* the standard-basis line follow the band set the reported
    curve actually carries (ISO 717-1:2020 Clause 5.3 / ISO 717-2:2020
    Clause 4.4 require stating it).
    """
    from ._i18n import t

    curve = np.atleast_1d(np.asarray(getattr(result, curve_attr), dtype=np.float64))
    is_octave = curve.size <= _N_OCTAVE_BANDS
    band_set = "Octave-band" if is_octave else "One-third-octave"
    bands_wording = t(
        "octave bands" if is_octave else "one-third-octave bands", language
    )
    # Pre-translate the {bands} basis template so the skeleton's
    # re-translation of the composed line is a no-op.
    spec = dict(spec)
    spec["basis"] = t(spec["basis"], language).format(bands=bands_wording)
    return render_insulation_fiche(
        result,
        rating,
        path,
        spec=spec,
        is_impact=is_impact,
        curve_attr=curve_attr,
        build_columns=iso717_columns_builder(
            rating, is_impact, spec["symbol"], band_set=band_set
        ),
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def render_survey_airborne_report(
    result: SurveyAirborneResult,
    rating: WeightedRatingResult,
    path: str,
    *,
    quantity: str,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a survey-method airborne fiche (ISO 10052) to a PDF.

    :param result: The
        :class:`~phonometry.building.measurement.survey_insulation.SurveyAirborneResult`.
    :param rating: The ISO 717-1 rating of the reported quantity (the result's
        ``rating`` for ``DnT`` or ``r_prime_rating`` for ``R'``); its ``plot``
        draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param quantity: ``"dnt"`` (the standardized level difference) or
        ``"r_prime"`` (the apparent sound reduction index).
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    curve_attr, _, spec = _AIRBORNE_QUANTITIES[quantity]
    return _render_survey(
        result,
        rating,
        path,
        spec=spec,
        is_impact=False,
        curve_attr=curve_attr,
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def render_survey_impact_report(
    result: SurveyImpactResult,
    rating: ImpactRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a survey-method impact fiche ``L'nT`` (ISO 10052) to a PDF.

    :param result: The
        :class:`~phonometry.building.measurement.survey_insulation.SurveyImpactResult`.
    :param rating: The ISO 717-2 rating of ``L'nT`` (the result's ``rating``);
        its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    return _render_survey(
        result,
        rating,
        path,
        spec=_L_NT_SPEC,
        is_impact=True,
        curve_attr="l_nt",
        metadata=metadata,
        verbose=verbose,
        language=language,
    )


def render_survey_facade_report(
    result: SurveyFacadeResult,
    rating: WeightedRatingResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a survey-method facade fiche ``D2m,nT`` (ISO 10052) to a PDF.

    :param result: The
        :class:`~phonometry.building.measurement.survey_insulation.SurveyFacadeResult`.
    :param rating: The ISO 717-1 rating of ``D2m,nT`` (the result's ``rating``);
        its ``plot`` draws the fiche curve.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        lightweight fiche (body, rating, statement and disclaimer).
    :param verbose: When ``True``, the left table shows the ISO 717 evaluation
        per band instead of the two-column form.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    return _render_survey(
        result,
        rating,
        path,
        spec=_D_2M_NT_SPEC,
        is_impact=False,
        curve_attr="d_2m_nt",
        metadata=metadata,
        verbose=verbose,
        language=language,
    )
