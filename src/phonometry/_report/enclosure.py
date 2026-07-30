#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Machine-enclosure insertion-loss performance fiche (reportlab renderer).

Renders a :class:`~phonometry.noise_control.enclosures.EnclosureResult` (the
net insertion loss of a close or free-standing machine enclosure, Bies, Hansen
& Howard, *Engineering Noise Control* 5th ed., section 7.4.2) to a one-page
enclosure-performance sheet:

* a title and the method-basis line naming the Bies insertion-loss model;
* an optional metadata header (client, enclosed machine, test environment,
  instrumentation, climate, date);
* a two-panel body with the per-band table on the left (nominal frequency, the
  supplied panel transmission loss ``R``, the interior build-up correction
  ``C`` and the net insertion loss ``IL = R - C``; with ``verbose=True`` the
  interior room constant ``R_i`` is added) and the panel ``R``, correction
  ``C`` and insertion-loss ``IL`` curves drawn by the result's own
  ``plot(ax=...)`` on the right;
* a boxed single-number result, the mean insertion loss over the analysis
  bands, with the external and internal surface areas alongside;
* an optional verdict row when a declared minimum is supplied via the metadata
  ``requirement`` (more insertion loss is better);
* a method-basis strip stating ``IL = R - C`` with
  ``C = 10 lg(0.3 + S_E / R_i)`` and the interior room constant.

The quantity-independent skeleton and the two-panel assembly live in
:mod:`._layout` and :mod:`._noise_control_fiche`; this module only holds the
enclosure specifics. reportlab, matplotlib and svglib are soft dependencies
imported lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import format_number, t
from ._noise_control_fiche import (
    band_labels,
    d1,
    mean_finite,
    performance_verdict,
    power_value_table,
    render_noise_control_fiche,
)
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..noise_control.enclosures import EnclosureResult


def _basis(language: str = "en") -> str:
    """The prediction-basis line naming the Bies enclosure insertion-loss model."""
    return t(
        "Predicted insertion loss of a machine enclosure, estimated from the "
        "panel transmission loss and the interior build-up correction (Bies, "
        "Hansen &amp; Howard, Engineering Noise Control 5th ed., section 7.4.2, "
        "Eqs. (7.103) and (7.111)). This is a prediction from the declared "
        "data, not a measurement of the enclosed machine.",
        language,
    )


def _prediction_statement(language: str = "en") -> str:
    """The prediction statement printed under the boxed insertion loss."""
    return t(
        "Predicted (estimated) result computed from the declared enclosure "
        "geometry, its panel transmission loss and the interior absorption; it "
        "is not a measurement. The realised insertion loss also depends on "
        "leaks, structural flanking and machine-to-enclosure contact, which "
        "this model does not represent.",
        language,
    )


def _caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    freqs = getattr(result, "frequencies", None)
    n = np.asarray(result.insertion_loss, dtype=np.float64).size
    if freqs is None:
        return t("Insertion loss per band", language)
    _, fraction = band_labels(freqs, n)
    if fraction == 1:
        return t("Octave-band insertion loss", language)
    return t("One-third-octave-band insertion loss", language)


def _value_table(result: Any, verbose: bool, language: str = "en") -> Any:
    """Build the per-band table (nominal frequency, R, C, IL; verbose adds R_i).

    Called only after the renderer has imported reportlab.
    """
    r = np.asarray(result.panel_transmission_loss, dtype=np.float64)
    c = np.asarray(result.correction, dtype=np.float64)
    il = np.asarray(result.insertion_loss, dtype=np.float64)
    n = il.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if not verbose:
        header = [
            t("f [Hz]", language),
            "R [dB]",
            "C [dB]",
            "IL [dB]",
        ]
        widths = [16.0, 16.0, 16.0, 16.0]
        rows_data = [
            [labels[i], d1(r[i], language), d1(c[i], language), d1(il[i], language)]
            for i in range(n)
        ]
        return power_value_table(header, rows_data, widths, fraction)

    r_i = np.asarray(result.room_constant, dtype=np.float64)
    header = [
        t("f [Hz]", language),
        "R [dB]",
        "C [dB]",
        "R<sub>i</sub> [m<super>2</super>]",
        "IL [dB]",
    ]
    widths = [12.0, 13.0, 13.0, 13.0, 13.0]
    rows_data = [
        [
            labels[i],
            d1(r[i], language),
            d1(c[i], language),
            d1(r_i[i], language),
            d1(il[i], language),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _statement(result: Any, language: str = "en") -> tuple[float, str, list[str]]:
    """The boxed mean insertion loss and its extended enclosure terms."""
    mean_il = mean_finite(result.insertion_loss)
    mean_r = mean_finite(result.panel_transmission_loss)
    statement = t(
        "Mean insertion loss IL = <b>{value} dB</b>", language
    ).format(value=d1(mean_il, language))
    extended = [
        t("Mean panel transmission loss R = {value} dB", language).format(
            value=d1(mean_r, language)
        ),
        t(
            "External surface area S<sub>E</sub> = {value} m<super>2</super>",
            language,
        ).format(value=format_number(float(result.external_area), language, decimals=2)),
        t(
            "Internal surface area S<sub>i</sub> = {value} m<super>2</super>",
            language,
        ).format(value=format_number(float(result.internal_area), language, decimals=2)),
    ]
    return mean_il, statement, extended


def _basis_strip(language: str = "en") -> str:
    """The IL = R - C relation line for the method-basis strip."""
    return t(
        "IL = R - C with the interior build-up correction "
        "C = 10 lg(0.3 + S<sub>E</sub> / R<sub>i</sub>) and the interior room "
        "constant R<sub>i</sub> = S<sub>i</sub> &#945;<sub>i</sub> / "
        "(1 - &#945;<sub>i</sub>) (Eqs. (7.103) and (7.111)). The panel "
        "transmission loss R is supplied by the caller (measured or predicted); "
        "this evaluation combines it with the interior build-up correction only.",
        language,
    )


def render_enclosure_report(
    result: EnclosureResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a machine-enclosure insertion-loss fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.noise_control.enclosures.EnclosureResult` carrying
        the per-band panel transmission loss ``R``, the interior build-up
        correction ``C``, the net insertion loss ``IL = R - C`` and the external
        and internal surface areas.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + basis, no header). A supplied ``requirement`` is
        read as a declared minimum mean insertion loss (more is better).
    :param verbose: When ``True`` the per-band table adds the interior room
        constant ``R_i`` column.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    mean_il, statement, extended = _statement(result, language)
    verdict = None
    if metadata is not None and metadata.requirement is not None:
        verdict = performance_verdict(
            mean_il, metadata.requirement, "IL",
            higher_is_better=True, language=language,
        )
    return render_noise_control_fiche(
        result,
        path,
        title=t("Machine enclosure insertion loss", language),
        basis=_basis(language),
        caption=_caption(result, language),
        value_table=_value_table(result, verbose, language),
        statement=statement,
        extended=extended,
        basis_strips=[_basis_strip(language)],
        metadata=metadata,
        language=language,
        prediction_statement=_prediction_statement(language),
        verdict=verdict,
    )
