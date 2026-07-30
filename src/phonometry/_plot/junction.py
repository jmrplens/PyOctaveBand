#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderer for the rigid plate-junction transmission result.

Lazy-imported from :meth:`JunctionTransmissionResult.plot`; domain classes are
referenced only under ``TYPE_CHECKING`` so this rendering leaf never imports
domain code at module level (see ``tests/test_package_architecture.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import _C_MUTED, _C_PRIMARY, _C_SECONDARY, _new_axes

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..vibration.junction_transmission import JunctionTransmissionResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Incidence angle [deg]": "Ángulo de incidencia [grados]",
    r"Transmission coefficient $\tau$": r"Coeficiente de transmisión $\tau$",
    "Bending-wave transmission, {junction}-junction "
    "($\\chi$ = {chi}, $\\psi$ = {psi})":
        "Transmisión de onda de flexión, unión {junction} "
        "($\\chi$ = {chi}, $\\psi$ = {psi})",
    r"corner $\tau_{12}(\theta)$": r"esquina $\tau_{12}(\theta)$",
    r"straight $\tau_{13}(\theta)$": r"recta $\tau_{13}(\theta)$",
    r"corner average $\bar\tau_{12}$ = {value}":
        r"media esquina $\bar\tau_{12}$ = {value}",
    r"straight average $\bar\tau_{13}$ = {value}":
        r"media recta $\bar\tau_{13}$ = {value}",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_junction_transmission(
    result: JunctionTransmissionResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Plot ``tau(theta)`` versus incidence angle (Hopkins Eqs 5.12/5.13).

    Draws the corner coefficient and, where it exists, the straight-section
    coefficient, with their diffuse-field angular averages as horizontal
    reference lines.

    :param result: A
        :class:`~phonometry.vibration.junction_transmission.JunctionTransmissionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the corner-curve ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    angles = np.asarray(result.angles_deg, dtype=np.float64)
    corner = np.asarray(result.corner, dtype=np.float64)

    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(angles, corner, label=_t(r"corner $\tau_{12}(\theta)$", language), **kwargs)
    ax.axhline(
        result.corner_average, color=_C_PRIMARY, ls="--", lw=1.0,
        label=_t(r"corner average $\bar\tau_{12}$ = {value}", language).replace(
            "{value}", format_number(result.corner_average, language, decimals=4)
        ),
    )
    if result.straight is not None and result.straight_average is not None:
        straight = np.asarray(result.straight, dtype=np.float64)
        ax.plot(
            angles, straight, color=_C_SECONDARY,
            label=_t(r"straight $\tau_{13}(\theta)$", language),
        )
        ax.axhline(
            result.straight_average, color=_C_SECONDARY, ls=":", lw=1.0,
            label=_t(r"straight average $\bar\tau_{13}$ = {value}", language).replace(
                "{value}", format_number(result.straight_average, language, decimals=4)
            ),
        )

    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(_t("Incidence angle [deg]", language))
    ax.set_ylabel(_t(r"Transmission coefficient $\tau$", language))
    ax.set_title(
        _t(
            "Bending-wave transmission, {junction}-junction "
            "($\\chi$ = {chi}, $\\psi$ = {psi})",
            language,
        ).format(
            junction=result.junction,
            chi=format_number(result.chi, language, decimals=3),
            psi=format_number(result.psi, language, decimals=3),
        )
    )
    ax.grid(True, color=_C_MUTED, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax
