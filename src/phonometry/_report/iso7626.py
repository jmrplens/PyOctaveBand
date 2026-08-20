#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 7626 mechanical-mobility fiche (reportlab renderer).

Renders a
:class:`~phonometry.vibration.structural.mechanical_mobility.MobilityResult` to a one-page
PDF laid out like a mechanical-mobility measurement report (ISO 7626-1:2011 for
the frequency-response-function definitions, ISO 7626-2:2015 for the
measurement):

* a title and the standard-basis line, naming whether the mobility is a
  driving-point FRF (response and force co-located) or a transfer FRF;
* an optional metadata header grid (client, manufacturer, the tested structure,
  test facility, instrumentation, date, temperature);
* a two-panel body with a compact table of the FRF's characteristic points on
  the left (the FRF type, the frequency range, the peak frequency, the peak
  mobility magnitude and the phase there) beside the mobility magnitude spectrum
  ``|Y(f)|`` drawn by the result's own ``plot(ax=...)``;
* a boxed representative value, the peak mobility ``|Y|`` and the frequency it
  occurs at (for a driving-point FRF this is a resonance, where ``|Y| = 1/c``
  measures the damping); and
* a footer identity/disclaimer block.

Mechanical mobility is a continuous frequency-response function over a fine
frequency axis, not an octave-band quantity, so the fiche presents it honestly
as a spectrum plot plus a small table of characteristic points; it carries no
per-band table and no pass/fail verdict (a mobility measurement is a
characterisation). The shared FRF skeleton lives in :mod:`._frf_fiche`; this
module only holds the mobility specifics. reportlab, matplotlib and svglib are
soft dependencies imported lazily (reportlab and svglib ship in the
``phonometry[report]`` extra, matplotlib in ``phonometry[plot]``); each is
guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._frf_fiche import frequency_range, frf_metadata_pairs, render_frf_fiche
from ._i18n import format_number, t
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..vibration.structural.mechanical_mobility import MobilityResult

#: The mobility unit, m/(N.s), as reportlab markup (a middle dot, not a hyphen).
_MOBILITY_UNIT = "m/(N&#183;s)"


def _kind(result: MobilityResult, language: str = "en") -> str:
    """The FRF-type phrase: driving-point (i = j) or transfer."""
    key = "driving-point" if result.driving_point else "transfer"
    return t(key, language)


def _basis(result: MobilityResult, language: str = "en") -> str:
    """The standard-basis line naming the FRF type and the ISO 7626 parts."""
    return t(
        "Measurement of the {kind} mechanical mobility Y = v/F over frequency "
        "(ISO 7626-1:2011 frequency-response function; measurement per "
        "ISO 7626-2:2015).",
        language,
    ).format(kind=_kind(result, language))


def _peak(result: MobilityResult) -> tuple[int, float, float]:
    """Return the peak index, peak magnitude and peak frequency of ``|Y|``."""
    magnitude = np.asarray(result.magnitude, dtype=np.float64)
    index = int(np.argmax(magnitude))
    return index, float(magnitude[index]), float(result.frequencies[index])


def _sig(value: float, language: str = "en") -> str:
    """A magnitude to three significant figures (mobility spans many decades)."""
    from ._i18n import decimal_comma

    return decimal_comma(f"{value:.3g}", language)


def _deg(value_rad: float, language: str = "en") -> str:
    """A phase angle in degrees, to the nearest degree."""
    return format_number(np.degrees(value_rad), language, decimals=0)


def _metric_rows(result: MobilityResult, language: str = "en") -> list[tuple[str, str]]:
    """The characteristic points shown in the left-hand table.

    The FRF type, the measured frequency range, the peak (resonance for a
    driving-point FRF) frequency, the peak mobility magnitude and the phase of
    the mobility there (near zero at a driving-point resonance, where the
    mobility is real).
    """
    index, peak_mag, peak_freq = _peak(result)
    phase = float(np.asarray(result.phase, dtype=np.float64)[index])
    return [
        (t("FRF type", language), _kind(result, language)),
        (
            t("Frequency range f [Hz]", language),
            frequency_range(np.asarray(result.frequencies, dtype=np.float64), language),
        ),
        (
            t("Peak frequency f [Hz]", language),
            format_number(peak_freq, language, decimals=1),
        ),
        (
            t("Peak mobility |Y| [{unit}]", language).format(unit=_MOBILITY_UNIT),
            _sig(peak_mag, language),
        ),
        (t("Phase at peak [&#176;]", language), _deg(phase, language)),
    ]


def _statement(result: MobilityResult, language: str = "en") -> str:
    """The boxed representative value: the peak mobility and its frequency."""
    _, peak_mag, peak_freq = _peak(result)
    return t("Peak mobility |Y| = <b>{value} {unit}</b> at {freq} Hz", language).format(
        value=_sig(peak_mag, language),
        unit=_MOBILITY_UNIT,
        freq=format_number(peak_freq, language, decimals=1),
    )


def _extended_terms(result: MobilityResult, language: str = "en") -> list[str]:
    """The FRF type, the frequency range and the peak phase shown beside the box."""
    index, _, _ = _peak(result)
    phase = float(np.asarray(result.phase, dtype=np.float64)[index])
    return [
        t("FRF type: {kind}", language).format(kind=_kind(result, language)),
        t("Frequency range: {range} Hz", language).format(
            range=frequency_range(
                np.asarray(result.frequencies, dtype=np.float64), language
            )
        ),
        t("Phase at peak: {value}&#176;", language).format(value=_deg(phase, language)),
    ]


def render_mobility_report(
    result: MobilityResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 7626 mechanical-mobility fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.vibration.structural.mechanical_mobility.MobilityResult`
        carrying the complex mobility ``Y(f)`` and whether it is a driving-point
        FRF.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        body-and-disclaimer fiche. The ``requirement`` field is ignored (a
        mobility measurement is a characterisation, so there is no verdict).
    :param verbose: Accepted for a uniform ``.report()`` signature; the mobility
        fiche has a single body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The fiche
        always embeds the mobility spectrum, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the mobility fiche has one body layout
    header_pairs = (
        frf_metadata_pairs(metadata, [], language)
        if metadata is not None and not metadata.is_empty()
        else []
    )
    return render_frf_fiche(
        result,
        path,
        title=t("Mechanical mobility measurement", language),
        basis=_basis(result, language),
        caption=t("Mobility FRF characteristics", language),
        header_pairs=header_pairs,
        metric_rows=_metric_rows(result, language),
        statement=_statement(result, language),
        extended=_extended_terms(result, language),
        metadata=metadata,
        language=language,
    )
