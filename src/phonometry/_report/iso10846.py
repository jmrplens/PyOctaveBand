#  Copyright (c) 2026. Jose Manuel Requena Plens
"""ISO 10846 dynamic-transfer-stiffness fiche (reportlab renderer).

Renders a
:class:`~phonometry.vibration.transfer_stiffness.TransferStiffnessResult` to a
one-page PDF laid out like a dynamic-transfer-stiffness characterisation report
for a resilient element (a vibration isolator, mount, bellows or hose) per
ISO 10846-1:2008 (the transfer-stiffness definition, 3.7), determined by either
the direct method (ISO 10846-2:2008) or the indirect blocking-mass method
(ISO 10846-3:2002):

* a title and the standard-basis line naming the determination method;
* an optional metadata header grid (client, manufacturer, the tested element,
  test facility, instrumentation, date, temperature);
* a two-panel body with a compact table of the FRF's characteristic points on
  the left (the method, the blocking mass for the indirect method, the frequency
  range, the low-frequency stiffness plateau ``|k2,1|``, its level ``L_k`` and
  the low-frequency loss factor ``eta``) beside the transfer-stiffness level
  spectrum ``L_k(f)`` drawn by the result's own ``plot(ax=...)``;
* a boxed representative value, the low-frequency dynamic-transfer-stiffness
  level ``L_k`` (the plateau that characterises the element below its internal
  resonances), with the stiffness magnitude and the method alongside; and
* a footer identity/disclaimer block.

Dynamic transfer stiffness is a continuous frequency-response function over a
fine frequency axis, not an octave-band quantity, so the fiche presents it
honestly as a spectrum plot plus a small table of characteristic points; it
carries no per-band table and no pass/fail verdict (a transfer-stiffness
determination is a characterisation). The shared FRF skeleton lives in
:mod:`._frf_fiche`; this module only holds the transfer-stiffness specifics.
reportlab, matplotlib and svglib are soft dependencies imported lazily (reportlab
and svglib ship in the ``phonometry[report]`` extra, matplotlib in
``phonometry[plot]``); each is guarded with an actionable :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._frf_fiche import frequency_range, frf_metadata_pairs, render_frf_fiche
from ._i18n import format_number, t
from .metadata import ReportMetadata

if TYPE_CHECKING:
    from ..vibration.transfer_stiffness import TransferStiffnessResult


def _is_indirect(result: TransferStiffnessResult) -> bool:
    """Return ``True`` for the indirect (blocking-mass) method (ISO 10846-3)."""
    return result.blocking_mass is not None


def _method(result: TransferStiffnessResult, language: str = "en") -> str:
    """The determination-method phrase naming the ISO 10846 part."""
    if _is_indirect(result):
        return t("indirect blocking-mass method (ISO 10846-3:2002)", language)
    return t("direct method (ISO 10846-2:2008)", language)


def _basis(result: TransferStiffnessResult, language: str = "en") -> str:
    """The standard-basis line naming the determination method."""
    return t(
        "Determination of the dynamic transfer stiffness k<sub>2,1</sub> of a "
        "resilient element by the {method} (ISO 10846-1:2008, 3.7).",
        language,
    ).format(method=_method(result, language))


def _low_frequency_values(
    result: TransferStiffnessResult,
) -> tuple[float, float, float, float]:
    """Return the lowest frequency and the ``|k2,1|``, ``L_k`` and ``eta`` there.

    The low-frequency point characterises the element below its internal
    resonances: the transfer stiffness there is the plateau reported as the
    headline value, and ISO 10846-1 (3.8) defines the loss factor only in the
    low-frequency range where inertial forces in the element are negligible.
    """
    freq = np.asarray(result.frequencies, dtype=np.float64)
    index = int(np.argmin(freq))
    magnitude = float(np.asarray(result.magnitude, dtype=np.float64)[index])
    level = float(np.asarray(result.level, dtype=np.float64)[index])
    # Reuse the result's own loss-factor property (eta = Im/Re, ISO 10846-1 3.8)
    # rather than recomputing it, so the fiche shares the single definition and
    # its validation (a purely imaginary stiffness is rejected there).
    eta = float(np.asarray(result.loss_factor, dtype=np.float64)[index])
    return float(freq[index]), magnitude, level, eta


def _mn(value: float, language: str = "en") -> str:
    """A stiffness magnitude in MN/m, to two decimals."""
    return format_number(value / 1e6, language, decimals=2)


def _metric_rows(
    result: TransferStiffnessResult, language: str = "en"
) -> list[tuple[str, str]]:
    """The characteristic points shown in the left-hand table.

    The determination method, the blocking mass (indirect method only), the
    measured frequency range, and the low-frequency stiffness plateau: its
    magnitude ``|k2,1|`` (MN/m), its level ``L_k`` (dB re 1 N/m) and the loss
    factor ``eta`` there.
    """
    _, magnitude, level, eta = _low_frequency_values(result)
    rows: list[tuple[str, str]] = [
        (t("Method", language), _method(result, language)),
    ]
    blocking_mass = result.blocking_mass
    if blocking_mass is not None:
        rows.append(
            (t("Blocking mass m<sub>2</sub> [kg]", language),
             format_number(float(blocking_mass), language, decimals=1))
        )
    rows.extend(
        [
            (t("Frequency range f [Hz]", language),
             frequency_range(np.asarray(result.frequencies, dtype=np.float64),
                             language)),
            (t("Low-frequency |k<sub>2,1</sub>| [MN/m]", language),
             _mn(magnitude, language)),
            (t("Low-frequency L<sub>k</sub> [dB re 1 N/m]", language),
             format_number(level, language, decimals=1)),
            (t("Loss factor &#951; (low frequency)", language),
             format_number(eta, language, decimals=3)),
        ]
    )
    return rows


def _statement(result: TransferStiffnessResult, language: str = "en") -> str:
    """The boxed representative value: the low-frequency transfer-stiffness level."""
    _, _, level, _ = _low_frequency_values(result)
    return t(
        "Low-frequency dynamic transfer stiffness level "
        "L<sub>k</sub> = <b>{value} dB re 1 N/m</b>", language
    ).format(value=format_number(level, language, decimals=1))


def _extended_terms(
    result: TransferStiffnessResult, language: str = "en"
) -> list[str]:
    """The stiffness magnitude, the method and the loss factor shown beside the box."""
    freq, magnitude, _, eta = _low_frequency_values(result)
    return [
        t("Low-frequency stiffness |k<sub>2,1</sub>| = {value} MN/m at {freq} Hz",
          language).format(
              value=_mn(magnitude, language),
              freq=format_number(freq, language, decimals=1, trim=True),
        ),
        t("Method: {method}", language).format(method=_method(result, language)),
        t("Loss factor &#951; = {value} (low frequency)", language).format(
            value=format_number(eta, language, decimals=3)
        ),
    ]


def render_transfer_stiffness_report(
    result: TransferStiffnessResult,
    path: str,
    *,
    metadata: ReportMetadata | None = None,
    verbose: bool = False,
    language: str = "en",
) -> str:
    """Render an ISO 10846 dynamic-transfer-stiffness fiche to a PDF at ``path``.

    :param result: A
        :class:`~phonometry.vibration.transfer_stiffness.TransferStiffnessResult`
        carrying the complex ``k2,1(f)`` and, for the indirect method, the
        blocking mass used.
    :param path: Destination path of the PDF file.
    :param metadata: Optional :class:`ReportMetadata`; ``None`` produces a
        body-and-disclaimer fiche. The ``requirement`` field is ignored (a
        transfer-stiffness determination is a characterisation, so there is no
        verdict).
    :param verbose: Accepted for a uniform ``.report()`` signature; the
        transfer-stiffness fiche has a single body layout, so it has no effect.
    :param language: ``"en"`` (default) or ``"es"``.
    :return: The written ``path`` as a :class:`str`.
    :raises ImportError: If reportlab or matplotlib is not installed. The fiche
        always embeds the ``L_k(f)`` spectrum, so both are required
        (``pip install "phonometry[report,plot]"``).
    """
    del verbose  # uniform signature; the transfer-stiffness fiche has one layout
    header_pairs = (
        frf_metadata_pairs(metadata, [], language)
        if metadata is not None and not metadata.is_empty()
        else []
    )
    return render_frf_fiche(
        result,
        path,
        title=t("Dynamic transfer stiffness of a resilient element", language),
        basis=_basis(result, language),
        caption=t("Transfer-stiffness characteristics", language),
        header_pairs=header_pairs,
        metric_rows=_metric_rows(result, language),
        statement=_statement(result, language),
        extended=_extended_terms(result, language),
        metadata=metadata,
        language=language,
    )
