#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Reactive-silencer transmission-loss performance fiche (reportlab renderer).

Renders a
:class:`~phonometry.noise_control.silencers.ReactiveSilencerResult` (the
transmission and, when end impedances are given, insertion loss of a reactive
silencer computed by the plane-wave four-pole method, Munjal, *Acoustics of
Ducts and Mufflers* 2nd ed.; Bies, Hansen & Howard, *Engineering Noise Control*
5th ed., sections 8.8-8.9) to a one-page silencer-performance sheet:

* a title and the method-basis line naming the four-pole (transfer-matrix)
  method and the device kind;
* an optional metadata header (client, device, test environment,
  instrumentation, climate, date);
* a two-panel body with the per-band table on the left (nominal frequency, the
  transmission loss ``TL`` and, when computed, the insertion loss ``IL``) and
  the ``TL`` (and ``IL``) curves drawn by the result's own ``plot(ax=...)`` on
  the right;
* a boxed single-number result, the mean transmission loss over the analysis
  bands, with the peak transmission loss and the device kind alongside;
* an optional verdict row when a declared minimum is supplied via the metadata
  ``requirement`` (more transmission loss is better);
* a method-basis strip stating the four-pole transmission-loss relation.

The quantity-independent skeleton and the two-panel assembly live in
:mod:`._layout` and :mod:`._noise_control_fiche`; this module only holds the
silencer specifics. reportlab, matplotlib and svglib are soft dependencies
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
    from ..noise_control.silencers import ReactiveSilencerResult


def _basis(language: str = "en") -> str:
    """The prediction-basis line naming the four-pole transmission-loss method."""
    return t(
        "Predicted transmission loss of a reactive silencer, computed by the "
        "plane-wave four-pole (transfer-matrix) method (Munjal, Acoustics of "
        "Ducts and Mufflers 2nd ed., Eq. (3.27); Bies, Hansen &amp; Howard, "
        "Engineering Noise Control 5th ed., sections 8.8 to 8.9). This is a "
        "prediction from design data, not a measurement.",
        language,
    )


def _prediction_statement(language: str = "en") -> str:
    """The prediction statement printed under the boxed transmission loss."""
    return t(
        "Predicted (estimated) result computed from the declared silencer "
        "geometry by the plane-wave model without flow; it is not a "
        "measurement. Above the plane-wave cut-on frequency of the duct, and "
        "with mean flow or dissipative lining present, the realised "
        "performance departs from this model.",
        language,
    )


def _caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    freqs = getattr(result, "frequencies", None)
    n = np.asarray(result.transmission_loss, dtype=np.float64).size
    if freqs is None:
        return t("Transmission loss per band", language)
    _, fraction = band_labels(freqs, n)
    if fraction == 1:
        return t("Octave-band transmission loss", language)
    return t("One-third-octave-band transmission loss", language)


def _value_table(result: Any, language: str = "en") -> Any:
    """Build the per-band table (nominal frequency, TL, and IL when computed).

    Called only after the renderer has imported reportlab.
    """
    tl = np.asarray(result.transmission_loss, dtype=np.float64)
    n = tl.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)

    if result.insertion_loss is None:
        header = [t("f [Hz]", language), "TL [dB]"]
        widths = [32.0, 32.0]
        rows_data = [[labels[i], d1(tl[i], language)] for i in range(n)]
        return power_value_table(header, rows_data, widths, fraction)

    il = np.asarray(result.insertion_loss, dtype=np.float64)
    header = [t("f [Hz]", language), "TL [dB]", "IL [dB]"]
    widths = [22.0, 21.0, 21.0]
    rows_data = [
        [labels[i], d1(tl[i], language), d1(il[i], language)] for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _resonance_term(result: Any, language: str = "en") -> str | None:
    """The extended term naming the resonance frequencies, or ``None``."""
    resonances = getattr(result, "resonances", None)
    if resonances is None:
        return None
    res = np.atleast_1d(np.asarray(resonances, dtype=np.float64))
    res = res[np.isfinite(res)]
    if res.size == 0:
        return None
    listed = ", ".join(format_number(float(fr), language, decimals=0) for fr in res)
    return t("Resonance frequencies: {values} Hz", language).format(values=listed)


def _statement(result: Any, language: str = "en") -> tuple[float, str, list[str]]:
    """The boxed mean transmission loss and its extended silencer terms."""
    tl = np.asarray(result.transmission_loss, dtype=np.float64)
    mean_tl = mean_finite(tl)
    finite = tl[np.isfinite(tl)]
    peak_tl = float(np.max(finite)) if finite.size else float("nan")
    statement = t("Mean transmission loss TL = <b>{value} dB</b>", language).format(
        value=d1(mean_tl, language)
    )
    extended = [
        t("Peak transmission loss = {value} dB", language).format(
            value=d1(peak_tl, language)
        ),
        t("Device: {kind}", language).format(kind=t(str(result.kind), language)),
    ]
    if result.insertion_loss is not None:
        mean_il = mean_finite(result.insertion_loss)
        extended.append(
            t("Mean insertion loss IL = {value} dB", language).format(
                value=d1(mean_il, language)
            )
        )
    resonance = _resonance_term(result, language)
    if resonance is not None:
        extended.append(resonance)
    return mean_tl, statement, extended


def _basis_strip(language: str = "en") -> str:
    """The four-pole transmission-loss relation line for the method-basis strip."""
    return t(
        "TL = 10 lg[(Z<sub>n</sub>/Z<sub>1</sub>) (1/4) |T<sub>11</sub> + "
        "T<sub>12</sub>/Z<sub>n</sub> + Z<sub>1</sub> T<sub>21</sub> + "
        "(Z<sub>1</sub>/Z<sub>n</sub>) T<sub>22</sub>|<super>2</super>] from the "
        "compound four-pole matrix T, with the characteristic impedances "
        "Z<sub>1</sub> = &#961;c/S<sub>in</sub> and Z<sub>n</sub> = "
        "&#961;c/S<sub>out</sub> (Munjal Eq. (3.27), no flow). TL is the "
        "intrinsic attenuation for an anechoic termination; the insertion loss "
        "adds the source and radiation impedance mismatch.",
        language,
    )


def render_reactive_silencer_report(
    result: ReactiveSilencerResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render a reactive-silencer transmission-loss fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.noise_control.silencers.ReactiveSilencerResult`
        carrying the per-band transmission loss ``TL`` and, when end impedances
        were supplied, the insertion loss ``IL``, the device ``kind`` and any
        resonance frequencies.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + basis, no header). A supplied ``requirement`` is
        read as a declared minimum mean transmission loss (more is better).
    :param verbose: Accepted for signature symmetry with the other fiches; the
        silencer table already shows the insertion loss when it was computed.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    del verbose  # the silencer table has no extra verbose column
    mean_tl, statement, extended = _statement(result, language)
    verdict = None
    if metadata is not None and metadata.requirement is not None:
        verdict = performance_verdict(
            mean_tl,
            metadata.requirement,
            "TL",
            higher_is_better=True,
            language=language,
        )
    return render_noise_control_fiche(
        result,
        path,
        title=t("Reactive silencer transmission loss", language),
        basis=_basis(language),
        caption=_caption(result, language),
        value_table=_value_table(result, language),
        statement=statement,
        extended=extended,
        basis_strips=[_basis_strip(language)],
        metadata=metadata,
        language=language,
        prediction_statement=_prediction_statement(language),
        verdict=verdict,
    )
