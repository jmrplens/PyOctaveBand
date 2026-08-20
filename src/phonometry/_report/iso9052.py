#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN 29052-1 / ISO 9052-1 dynamic-stiffness fiche (reportlab renderer).

Renders a
:class:`~phonometry.materials.resilient.dynamic_stiffness.DynamicStiffnessResult` to a
one-page PDF laid out like an accredited dynamic-stiffness test report
(EN 29052-1:1992, identical to ISO 9052-1:1989):

* a title and the standard-basis line;
* an optional metadata header block (client, specimen, the total mass per unit
  area ``m't`` used during the test, the loaded specimen thickness ``d``, test
  facility, date, climate ...), built from the supplied :class:`ReportMetadata`
  and covering the Clause 9 b)/d) reporting items;
* a two-panel body with a compact metrics table on the left (the resonant
  frequency ``fr``, the apparent dynamic stiffness ``s't`` of Formula 4, the
  enclosed-gas term ``s'a`` of Formula 7 when it applies, the installed dynamic
  stiffness ``s'`` of Clause 8.2 and the supported-floor natural frequency
  ``f0`` of Formula 2) beside the ``f0(s')`` design curve on the right, drawn by
  the result's own ``plot(ax=...)`` so the curve is native to the library;
* a boxed apparent dynamic stiffness ``s't`` with the installed ``s'`` and the
  resonant frequency ``fr`` alongside;
* a footer identity/disclaimer block.

EN 29052-1 is a characterisation, so this fiche carries no pass/fail verdict.
Clause 9 requires every dynamic stiffness per unit area to be stated in
meganewtons per cubic metre to the nearest meganewton per cubic metre, so the
stiffness values are rounded to the nearest MN/m3; the frequencies are shown to
0,1 Hz.

The quantity-independent skeleton lives in :mod:`._layout`; this module only
holds the EN 29052-1 specifics. reportlab, matplotlib and svglib are soft
dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._i18n import format_number, t
from ._layout import fmt_meta, fmt_num
from ._material_fiche import (
    MaterialFicheContent,
    material_metadata_pairs,
    render_material_fiche,
    standard_basis_line,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..materials.resilient.dynamic_stiffness import DynamicStiffnessResult


def _mn(value: float, language: str = "en") -> str:
    """Dynamic stiffness in MN/m3 to the nearest MN/m3 (Clause 9)."""
    return format_number(value / 1e6, language, decimals=0)


def _hz(value: float, language: str = "en") -> str:
    """A frequency to 0,1 Hz."""
    return format_number(value, language, decimals=1)


def _metadata_pairs(
    metadata: ReportMetadata, language: str = "en"
) -> list[tuple[str, str]]:
    """Build the ordered (label, value) pairs of the dynamic-stiffness header grid.

    Only fields that are set are returned. The applicable fields are the generic
    identity fields plus the total mass per unit area ``m't`` used during the
    test (``mass_per_area``, kg/m2), the loaded specimen thickness ``d``
    (``thickness``, stored in metres and printed in millimetres, Clause 9 b))
    and the environmental conditions (Clause 9 d)). The frequency-range/room
    fields of the other fiches do not apply here.
    """
    middle: list[tuple[str, str | None]] = [
        (
            t(
                "Total mass per area m&#8242;<sub>t</sub> [kg/m<super>2</super>]",
                language,
            ),
            fmt_meta(metadata.mass_per_area, language)
            if metadata.mass_per_area is not None
            else None,
        ),
        (
            t("Thickness under load d [mm]", language),
            fmt_meta(metadata.thickness * 1e3, language)
            if metadata.thickness is not None
            else None,
        ),
    ]
    return material_metadata_pairs(metadata, language, middle)


def _metric_rows(
    result: DynamicStiffnessResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The scalar results shown in the left-hand metrics table.

    The resonant frequency and the natural frequency are printed to 0,1 Hz; the
    dynamic stiffnesses to the nearest MN/m3 (Clause 9). The enclosed-gas term
    ``s'a`` is shown only when it is non-zero (an air-permeable material in the
    intermediate/low airflow-resistivity regime); the installed ``s'`` shows an
    em dash and the natural frequency is omitted when the method cannot resolve
    ``s'`` (a non-finite result, Clause 8.2 c)).
    """
    resolved = bool(np.isfinite(result.dynamic_stiffness))
    rows: list[tuple[str, str]] = [
        (
            t("Resonant frequency f<sub>r</sub> [Hz]", language),
            _hz(result.resonant_frequency, language),
        ),
        (
            t(
                "Apparent dynamic stiffness s&#8242;<sub>t</sub> [MN/m<super>3</super>]",
                language,
            ),
            _mn(result.apparent_stiffness, language),
        ),
    ]
    if result.gas_stiffness > 0.0:
        rows.append(
            (
                t(
                    "Enclosed-gas stiffness s&#8242;<sub>a</sub> [MN/m<super>3</super>]",
                    language,
                ),
                _mn(result.gas_stiffness, language),
            )
        )
    rows.append(
        (
            t("Dynamic stiffness s&#8242; [MN/m<super>3</super>]", language),
            _mn(result.dynamic_stiffness, language) if resolved else "&#8212;",
        )
    )
    rows.append(
        (
            t("Supported-floor mass m&#8242; [kg/m<super>2</super>]", language),
            fmt_num(result.floor_mass_per_area, language),
        )
    )
    if resolved and np.isfinite(result.natural_frequency):
        rows.append(
            (
                t("Natural frequency f<sub>0</sub> [Hz]", language),
                _hz(result.natural_frequency, language),
            )
        )
    return rows


def _statement(result: DynamicStiffnessResult, language: str = "en") -> str:
    """The boxed apparent dynamic stiffness ``s't`` (Formula 4)."""
    return t(
        "Apparent dynamic stiffness s&#8242;<sub>t</sub> = "
        "<b>{value} MN/m<super>3</super></b>",
        language,
    ).format(value=_mn(result.apparent_stiffness, language))


def _extended_terms(result: DynamicStiffnessResult, language: str = "en") -> list[str]:
    """The installed ``s'`` and the test resonance ``fr`` shown beside the box."""
    resolved = bool(np.isfinite(result.dynamic_stiffness))
    s_installed = (
        f"{_mn(result.dynamic_stiffness, language)} MN/m<super>3</super>"
        if resolved
        else "&#8212;"
    )
    terms = [
        t("Dynamic stiffness s&#8242; = {value}", language).format(value=s_installed),
        t("Resonant frequency f<sub>r</sub> = {value} Hz", language).format(
            value=_hz(result.resonant_frequency, language)
        ),
    ]
    if resolved and np.isfinite(result.natural_frequency):
        terms.append(
            t("Natural frequency f<sub>0</sub> = {value} Hz", language).format(
                value=_hz(result.natural_frequency, language)
            )
        )
    return terms


def _basis_line(metadata: ReportMetadata | None, language: str = "en") -> str:
    """The standard-basis line, naming the measurement standard when supplied."""
    return standard_basis_line(
        "{standard} resonance measurement of the apparent dynamic stiffness "
        "per unit area of a resilient layer used under a floating floor per "
        "EN 29052-1:1992 (ISO 9052-1:1989).",
        "Resonance measurement of the apparent dynamic stiffness per unit area "
        "of a resilient layer used under a floating floor per EN 29052-1:1992 "
        "(ISO 9052-1:1989).",
        metadata,
        language,
    )


def render_dynamic_stiffness_report(
    result: DynamicStiffnessResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an EN 29052-1 dynamic-stiffness fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.materials.resilient.dynamic_stiffness.DynamicStiffnessResult`
        carrying the apparent, enclosed-gas and installed dynamic stiffnesses,
        the test resonance ``fr`` and the supported-floor natural frequency
        ``f0``.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        body-and-disclaimer fiche. The ``requirement`` field is ignored
        (EN 29052-1 is a characterisation, so there is no verdict).
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        dynamic-stiffness fiche has a single body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The fiche
        always embeds the ``f0(s')`` design curve, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the dynamic-stiffness fiche has one layout
    pairs = (
        _metadata_pairs(metadata, language)
        if metadata is not None and not metadata.is_empty()
        else []
    )
    content = MaterialFicheContent(
        title=t("Dynamic stiffness of resilient materials", language),
        basis_line=_basis_line(metadata, language),
        caption=t("Dynamic-stiffness results", language),
        metadata_pairs=pairs,
        metric_rows=_metric_rows(result, language),
        statement=_statement(result, language),
        extended=_extended_terms(result, language),
    )
    return render_material_fiche(
        result.plot, path, content, metadata, language=language
    )
