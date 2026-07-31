#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 9053-1:2018 static-method airflow-resistance fiche (reportlab renderer).

Renders a
:class:`~phonometry.materials.airflow_resistance.StaticAirflowResult` to a
one-page PDF laid out like an accredited airflow-resistance test report
(ISO 9053-1:2018, static/direct airflow method):

* a title and the standard-basis line;
* an optional metadata header block (client, manufacturer, specimen, the
  specimen thickness ``d``, test facility, date, climate ...), built from the
  supplied :class:`ReportMetadata`;
* a two-panel body with a compact metrics table on the left (the evaluation
  velocity, the fitted pressure difference, the airflow resistance ``R``, the
  specific airflow resistance ``R_s``, the airflow resistivity ``sigma`` when a
  thickness is available, and the through-origin fit coefficients ``a`` and
  ``b``) beside the fitted ``dp(u)`` curve on the right, drawn by the result's
  own ``plot(ax=...)`` so the curve is native to the library;
* a boxed specific airflow resistance ``R_s`` with the airflow resistance ``R``
  and the airflow resistivity ``sigma`` alongside;
* a footer identity/disclaimer block.

ISO 9053-1 is a material characterisation, so this fiche carries no pass/fail
verdict. The clause 7.5 stepwise procedure fits the pressure difference against
the linear airflow velocity with a through-origin second-order regression
``dp = a*u + b*u**2`` and evaluates the resistances at the reference velocity
``u = 0.5 mm/s``; the linear coefficient ``a`` is the zero-velocity specific
airflow resistance. All resistance quantities are printed to the nearest whole
Pa*s unit and the evaluation velocity to one decimal place (mm/s).

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the ISO 9053-1 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._i18n import format_number, t
from ._layout import fmt_meta
from ._material_fiche import (
    MaterialFicheContent,
    material_metadata_pairs,
    render_material_fiche,
    standard_basis_line,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..materials.airflow_resistance import StaticAirflowResult


def _pas(value: float, language: str = "en") -> str:
    """A resistance quantity to the nearest whole Pa*s unit."""
    return format_number(value, language, decimals=0)


def _mm_s(value_m_s: float, language: str = "en") -> str:
    """A linear airflow velocity (stored in m/s) printed in mm/s to 0,1 mm/s."""
    return format_number(value_m_s * 1e3, language, decimals=1, trim=True)


def _pa(value: float, language: str = "en") -> str:
    """A pressure difference to 0,1 Pa."""
    return format_number(value, language, decimals=1, trim=True)


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the airflow-resistance header grid.

    Only fields that are set are returned. The applicable fields are the generic
    identity fields plus the specimen thickness ``d`` in the flow direction
    (``thickness``, stored in metres and printed in millimetres) and the
    environmental conditions.
    """
    middle: list[tuple[str, str | None]] = [
        (t("Thickness d [mm]", language),
         fmt_meta(metadata.thickness * 1e3, language)
         if metadata.thickness is not None else None),
    ]
    return material_metadata_pairs(metadata, language, middle)


def _metric_rows(
    result: StaticAirflowResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The scalar results shown in the left-hand metrics table.

    The evaluation velocity is shown in mm/s, the fitted pressure difference in
    Pa, and the resistance quantities to the nearest whole Pa*s unit. The
    airflow resistivity ``sigma`` is shown only when the result carries a
    thickness. The through-origin fit coefficients ``a`` (the zero-velocity
    specific airflow resistance) and ``b`` close the table.
    """
    rows: list[tuple[str, str]] = [
        (t("Evaluation velocity u [mm/s]", language),
         _mm_s(result.evaluation_velocity, language)),
        (t("Fitted pressure difference &#916;p [Pa]", language),
         _pa(result.pressure_drop, language)),
        (t("Airflow resistance R [Pa&#183;s/m<super>3</super>]", language),
         _pas(result.resistance, language)),
        (t("Specific airflow resistance R<sub>s</sub> [Pa&#183;s/m]", language),
         _pas(result.specific_resistance, language)),
    ]
    if result.resistivity is not None:
        rows.append(
            (t("Airflow resistivity &#963; [Pa&#183;s/m<super>2</super>]", language),
             _pas(result.resistivity, language))
        )
    rows.append(
        (t("Zero-velocity resistance a [Pa&#183;s/m]", language),
         _pas(result.linear_coefficient, language))
    )
    rows.append(
        (t("Quadratic coefficient b [Pa&#183;s<super>2</super>/m<super>2</super>]",
           language),
         _pas(result.quadratic_coefficient, language))
    )
    return rows


def _statement(result: StaticAirflowResult, language: str = "en") -> str:
    """The boxed specific airflow resistance ``R_s`` (clause 7.5)."""
    return t(
        "Specific airflow resistance R<sub>s</sub> = "
        "<b>{value} Pa&#183;s/m</b>", language
    ).format(value=_pas(result.specific_resistance, language))


def _extended_terms(
    result: StaticAirflowResult, language: str = "en"
) -> list[str]:
    """The airflow resistance ``R``, resistivity ``sigma`` and evaluation velocity."""
    terms = [
        t("Airflow resistance R = {value} Pa&#183;s/m<super>3</super>", language).format(
            value=_pas(result.resistance, language)
        ),
    ]
    if result.resistivity is not None:
        terms.append(
            t("Airflow resistivity &#963; = {value} Pa&#183;s/m<super>2</super>",
              language).format(value=_pas(result.resistivity, language))
        )
    terms.append(
        t("Evaluated at u = {value} mm/s", language).format(
            value=_mm_s(result.evaluation_velocity, language)
        )
    )
    return terms


def _basis_line(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line, naming the measurement standard when supplied."""
    return standard_basis_line(
        "{standard} static-method determination of the airflow resistance "
        "of a porous material by a steady unidirectional (DC) flow per "
        "ISO 9053-1:2018.",
        "Static-method determination of the airflow resistance of a porous "
        "material by a steady unidirectional (DC) flow per ISO 9053-1:2018.",
        metadata,
        language,
    )


def render_static_airflow_report(
    result: StaticAirflowResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 9053-1 static airflow-resistance fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.airflow_resistance.StaticAirflowResult`
        carrying the airflow resistance ``R``, the specific airflow resistance
        ``R_s``, the airflow resistivity ``sigma`` (when a thickness was
        supplied), the through-origin fit coefficients and the evaluation
        point.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        body-and-disclaimer fiche. The ``requirement`` field is ignored
        (ISO 9053-1 is a characterisation, so there is no verdict).
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        airflow-resistance fiche has a single body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The fiche
        always embeds the fitted ``dp(u)`` curve, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the airflow-resistance fiche has one layout
    pairs = (
        _metadata_pairs(metadata, language)
        if metadata is not None and not metadata.is_empty()
        else []
    )
    content = MaterialFicheContent(
        title=t("Airflow resistance of porous materials", language),
        basis_line=_basis_line(metadata, language),
        caption=t("Static airflow-resistance results", language),
        metadata_pairs=pairs,
        metric_rows=_metric_rows(result, language),
        statement=_statement(result, language),
        extended=_extended_terms(result, language),
    )
    return render_material_fiche(result.plot, path, content, metadata, language=language)
