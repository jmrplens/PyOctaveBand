#  Copyright (c) 2026. Jose Manuel Requena Plens
"""HVAC duct-noise-spectrum performance fiche (reportlab renderer).

Renders a :class:`~phonometry.noise_control.hvac.HvacSpectrumResult` (a
per-band HVAC duct quantity: the flow-generated (regenerated) sound power level
of a straight duct or bend, or the passive attenuation of an end reflection,
bend or plenum, Bies, Hansen & Howard, *Engineering Noise Control* 5th ed.,
Chapter 8) to a one-page HVAC-noise sheet:

* a title and the method-basis line naming the reported quantity and the Bies
  chapter;
* an optional metadata header (client, duct element, test environment,
  instrumentation, climate, date);
* a two-panel body with the per-band table on the left (nominal frequency and
  the reported quantity; with ``verbose=True`` a regenerated-noise spectrum
  adds the A-weighting correction and the A-weighted band level) and the
  spectrum drawn by the result's own ``plot(ax=...)`` on the right;
* a boxed single-number result: for a regenerated-noise spectrum the A-weighted
  sound power level ``L_WA`` (dB(A) re 1 pW) with the overall unweighted total;
  for an attenuation spectrum the mean attenuation with its band range;
* an optional verdict row when a declared limit is supplied via the metadata
  ``requirement`` (lower regenerated noise is better; more attenuation is
  better);
* a method-basis strip stating the reported quantity's relation.

The quantity-independent skeleton and the two-panel assembly live in
:mod:`._layout` and :mod:`._noise_control_fiche`; this module only holds the
HVAC specifics. reportlab, matplotlib and svglib are soft dependencies imported
lazily (reportlab and svglib ship in the ``phonometry[report]`` extra,
matplotlib in ``phonometry[plot]``); each is guarded with an actionable
:class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._i18n import t
from ._noise_control_fiche import (
    band_labels,
    d1,
    mean_finite,
    performance_verdict,
    power_value_table,
    render_noise_control_fiche,
)

if TYPE_CHECKING:
    from ..noise_control.hvac import HvacSpectrumResult
    from .metadata import ReportMetadata

#: Reference sound power for a regenerated-noise sound power level, 1 pW.
_POWER_REFERENCE = "1 pW"

#: Band fraction returned by ``band_labels`` for a one-third-octave band set
#: (1 = octave, 3 = one-third-octave, 0 = no octave grouping).
_THIRD_OCTAVE_FRACTION = 3


def _is_power(result: Any) -> bool:
    """Return ``True`` for a regenerated sound power spectrum, else attenuation."""
    return getattr(result, "quantity", "") == "sound_power_level"


def _a_weighting(frequencies: np.ndarray) -> np.ndarray | None:
    """A-weighting band corrections at nominal band centres, or ``None``.

    Reuses the library's ISO 3744:2010 Annex E A-weighting table (the same
    corrections the sound-power determination applies), so the fiche's
    A-weighted level is consistent with the rest of the library. Returns
    ``None`` when the spectrum is not on nominal octave/one-third-octave centres
    (the table is undefined there), and the fiche then falls back to the
    unweighted total. Imported lazily to keep this render leaf free of a
    module-level domain import.
    """
    try:
        from ..emission._shared import _a_weighting_corrections

        return np.asarray(
            _a_weighting_corrections(np.asarray(frequencies, dtype=np.float64)),
            dtype=np.float64,
        )
    except (ValueError, ImportError):
        return None


def _overall_level(levels: np.ndarray) -> float:
    """Energy-summed overall level, ``10 lg(sum 10^(L/10))``."""
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.sum(10.0 ** (finite / 10.0))))


def _a_weighted_total(
    levels: np.ndarray, corrections: np.ndarray | None
) -> float | None:
    """A-weighted energy-summed total, or ``None`` when no corrections apply."""
    if corrections is None:
        return None
    weighted = levels + corrections
    return _overall_level(weighted)


def _basis(result: Any, language: str = "en") -> str:
    """The prediction-basis line naming the quantity and the Bies chapter."""
    if _is_power(result):
        return t(
            "Predicted flow-generated (regenerated) octave-band sound power "
            "level of a duct element (Bies, Hansen &amp; Howard, Engineering "
            "Noise Control 5th ed., Chapter 8, section 8.15; VDI 2081-1). This "
            "is a prediction from design data, not a measurement.",
            language,
        )
    return t(
        "Predicted octave-band attenuation of a duct element (Bies, Hansen "
        "&amp; Howard, Engineering Noise Control 5th ed., Chapter 8). This is "
        "a prediction from design data, not a measurement.",
        language,
    )


def _prediction_statement(result: Any, language: str = "en") -> str:
    """The prediction statement printed under the boxed figure."""
    if _is_power(result):
        return t(
            "Predicted (estimated) result computed from the declared duct "
            "geometry and flow conditions by an empirical spectrum model; it "
            "is not a measurement. The realised level also depends on the "
            "upstream flow condition and on fittings near the element.",
            language,
        )
    return t(
        "Predicted (estimated) result computed from the declared element "
        "geometry by a design model; it is not a measurement. Path "
        "attenuations of a real installation also depend on the duct "
        "construction and on breakout along the run.",
        language,
    )


def _element_label(result: Any, language: str = "en") -> str:
    """The element label, with its descriptive words translated.

    The result carries a short English label built from the element kind and
    its configuration and dimensions, e.g. ``"Elbow (square, lined, W = 300
    mm)"``. The kind and each configuration word are translated; a
    comma-separated item holding a quantity (``W = 300 mm``) is a symbol,
    number and unit, so it is left exactly as the domain wrote it.
    """
    label = str(getattr(result, "label", ""))
    if not label or language == "en":
        return label
    head, sep, tail = label.partition(" (")
    if not sep:
        return t(head, language)
    parts = [
        item if "=" in item else t(item, language)
        for item in tail.rstrip(")").split(", ")
    ]
    return f"{t(head, language)} ({', '.join(parts)})"


def _caption(result: Any, language: str = "en") -> str:
    """The caption declaring the analysis band set above the table."""
    n = np.asarray(result.values, dtype=np.float64).size
    _, fraction = band_labels(getattr(result, "frequencies", None), n)
    power = _is_power(result)
    if fraction == 1:
        return (
            t("Octave-band regenerated sound power levels", language)
            if power
            else t("Octave-band attenuation", language)
        )
    if fraction == _THIRD_OCTAVE_FRACTION:
        return (
            t("One-third-octave-band regenerated sound power levels", language)
            if power
            else t("One-third-octave-band attenuation", language)
        )
    return (
        t("Regenerated sound power levels per band", language)
        if power
        else t("Attenuation per band", language)
    )


def _value_table(
    result: Any, verbose: bool, corrections: np.ndarray | None, language: str = "en"
) -> Any:
    """Build the per-band table (nominal frequency and the reported quantity).

    A verbose regenerated-noise table adds the A-weighting correction and the
    A-weighted band level. Called only after the renderer has imported reportlab.
    """
    values = np.asarray(result.values, dtype=np.float64)
    n = values.size
    labels, fraction = band_labels(getattr(result, "frequencies", None), n)
    power = _is_power(result)
    quantity_header = "L<sub>W</sub> [dB]" if power else "D [dB]"

    if not (verbose and power and corrections is not None):
        header = [t("f [Hz]", language), quantity_header]
        widths = [32.0, 32.0]
        rows_data = [[labels[i], d1(values[i], language)] for i in range(n)]
        return power_value_table(header, rows_data, widths, fraction)

    weighted = values + corrections
    header = [
        t("f [Hz]", language),
        "L<sub>W</sub> [dB]",
        "C<sub>A</sub> [dB]",
        "L<sub>WA</sub> [dB]",
    ]
    widths = [16.0, 16.0, 16.0, 16.0]
    rows_data = [
        [
            labels[i],
            d1(values[i], language),
            d1(corrections[i], language),
            d1(weighted[i], language),
        ]
        for i in range(n)
    ]
    return power_value_table(header, rows_data, widths, fraction)


def _statement(
    result: Any, corrections: np.ndarray | None, language: str = "en"
) -> tuple[float, bool, str, list[str]]:
    """The boxed headline figure, its pass direction and the extended terms.

    Returns ``(value, higher_is_better, statement, extended)``. A
    regenerated-noise spectrum boxes the A-weighted sound power level (lower is
    better) with the overall unweighted total alongside; an attenuation spectrum
    boxes the mean attenuation (more is better) with its band range.
    """
    values = np.asarray(result.values, dtype=np.float64)
    label = _element_label(result, language)
    if _is_power(result):
        overall = _overall_level(values)
        lwa = _a_weighted_total(values, corrections)
        if lwa is not None:
            statement = t(
                "A-weighted sound power level L<sub>WA</sub> = "
                "<b>{value} dB(A)</b> re {ref}",
                language,
            ).format(value=d1(lwa, language), ref=_POWER_REFERENCE)
            extended = [
                t(
                    "Overall sound power level L<sub>W</sub> = {value} dB re {ref}",
                    language,
                ).format(value=d1(overall, language), ref=_POWER_REFERENCE),
                label,
            ]
            return lwa, False, statement, extended
        statement = t(
            "Overall sound power level L<sub>W</sub> = <b>{value} dB</b> re {ref}",
            language,
        ).format(value=d1(overall, language), ref=_POWER_REFERENCE)
        return overall, False, statement, [label]

    mean_att = mean_finite(values)
    finite = values[np.isfinite(values)]
    lo = float(np.min(finite)) if finite.size else float("nan")
    hi = float(np.max(finite)) if finite.size else float("nan")
    statement = t("Mean attenuation D = <b>{value} dB</b>", language).format(
        value=d1(mean_att, language)
    )
    extended = [
        t("Band attenuation range {lo} to {hi} dB", language).format(
            lo=d1(lo, language), hi=d1(hi, language)
        ),
        label,
    ]
    return mean_att, True, statement, extended


def _basis_strip(
    result: Any, corrections: np.ndarray | None, language: str = "en"
) -> list[str]:
    """The method-basis strip(s) for the reported quantity."""
    if _is_power(result):
        strips = [
            t(
                "L<sub>W</sub> is the flow-generated octave-band sound power "
                "level radiated by the element (Chapter 8, section 8.15; "
                "VDI 2081-1), referenced to the reference sound power 1 pW.",
                language,
            )
        ]
        if corrections is not None:
            strips.append(
                t(
                    "The A-weighted level L<sub>WA</sub> combines the band levels "
                    "with the ISO 3744:2010 Annex E A-weighting corrections "
                    "C<sub>A</sub> (Tables E.1/E.2).",
                    language,
                )
            )
        return strips
    return [
        t(
            "The octave-band attenuation is the passive insertion loss the "
            "element adds along the duct path (Chapter 8). Path attenuations "
            "add band by band; regenerated noise is added separately.",
            language,
        )
    ]


def render_hvac_report(
    result: HvacSpectrumResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an HVAC duct-noise-spectrum fiche to a PDF at ``path``.

    :param result: An
        :class:`~phonometry.noise_control.hvac.HvacSpectrumResult` carrying the
        per-band quantity (a regenerated sound power level or a passive
        attenuation), its band frequencies and a short element label.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a bare
        fiche (body + result + basis, no header). A supplied ``requirement`` is
        read as a declared maximum A-weighted level for a regenerated-noise
        spectrum (lower is better) or a declared minimum mean attenuation for an
        attenuation spectrum (more is better).
    :param verbose: When ``True`` a regenerated-noise table adds the A-weighting
        correction and the A-weighted band level columns.
    :param language: Fiche language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab (or, for the figure, matplotlib) is not
        installed.
    """
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    corrections = _a_weighting(freqs) if _is_power(result) else None
    value, higher_is_better, statement, extended = _statement(
        result, corrections, language
    )
    verdict = None
    if metadata is not None and metadata.requirement is not None:
        a_weighted = _is_power(result) and corrections is not None
        if a_weighted:
            symbol = "L<sub>WA</sub>"
        elif _is_power(result):
            symbol = "L<sub>W</sub>"
        else:
            symbol = "D"
        unit = "dB(A)" if a_weighted else "dB"
        verdict = performance_verdict(
            value,
            metadata.requirement,
            symbol,
            higher_is_better=higher_is_better,
            unit=unit,
            language=language,
        )
    return render_noise_control_fiche(
        result,
        path,
        title=t("HVAC duct noise spectrum", language),
        basis=_basis(result, language),
        caption=_caption(result, language),
        value_table=_value_table(result, verbose, corrections, language),
        statement=statement,
        extended=extended,
        basis_strips=_basis_strip(result, corrections, language),
        metadata=metadata,
        language=language,
        prediction_statement=_prediction_statement(result, language),
        verdict=verdict,
    )
