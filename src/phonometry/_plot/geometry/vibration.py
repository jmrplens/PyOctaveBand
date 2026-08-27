#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the vibration domain: plates and their junctions.

Two drawings of structure-borne sound, both to scale in the thickness that
governs the physics. The junction drawing shows the L, T or X arrangement of
plates whose bending-wave transmission is computed across it; the plate drawing
shows the baffled rectangular plate, with its aspect ratio and boundary
condition, whose radiation efficiency is computed from it.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..._internal.validation import require_positive
from ..common import _new_axes
from ._draft import (
    _check_language,
    _dim,
    _finish_geometry_axes,
    _incidence_arrow,
    _material_rect,
    _metres,
    _mm,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...vibration.structural.junction_transmission import (
        JunctionTransmissionResult,
    )
    from ...vibration.structural.radiation_efficiency import (
        RadiationEfficiencyResult,
    )

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Plate junction ({junction})": "Unión de placas ({junction})",
    "Baffled plate ({boundary})": "Placa en pantalla ({boundary})",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Plate junction (L / T / X).
# ---------------------------------------------------------------------------
def plot_junction_geometry(
    junction: str,
    thickness1: float,
    thickness2: float,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw a plate junction cross-section to scale.

    Plate 1 runs horizontally; the perpendicular plate(s) of thickness 2
    form the L, T or X. The incident bending wave arrives on plate 1 and
    the junction type follows
    :func:`~phonometry.vibration.structural.junction_transmission`.

    :param junction: ``"L"``, ``"T1"``, ``"T2"`` or ``"X"``.
    :param thickness1: Plate 1 thickness, in metres.
    :param thickness2: Plate 2 thickness, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the plate-1 rectangle.
    :return: The axes.
    :raises ValueError: for an unknown junction type, or a thickness that is
        not positive and finite.
    """
    _check_language(language)
    if junction not in ("L", "T1", "T2", "X"):
        msg = "'junction' must be 'L', 'T1', 'T2' or 'X'."
        raise ValueError(msg)
    # require_positive rather than a bare ``<= 0.0``: NaN compares False
    # against every bound, so a NaN thickness used to pass and draw plates
    # collapsed to nothing under a dimension label reading "nan mm". Each
    # thickness is named separately because the caller passed two.
    h1 = require_positive(thickness1, "thickness1")
    h2 = require_positive(thickness2, "thickness2")
    if ax is None:
        ax = _new_axes()
    arm = 5.0 * max(h1, h2)
    # Plate 1 is continuous for T1 and X (its "plates 1 and 3" are the same
    # panel); it stops against the perpendicular plate for L and T2.
    x_left = -arm
    x_right = arm if junction in ("T1", "X") else 0.5 * h2
    _material_rect(ax, x_left, -0.5 * h1, x_right - x_left, h1, "plate", **kwargs)
    # The perpendicular plate is continuous (both branches) for T2 and X.
    down = junction in ("T2", "X")
    _material_rect(ax, -0.5 * h2, 0.5 * h1, h2, arm, "plate")
    if down:
        _material_rect(ax, -0.5 * h2, -0.5 * h1 - arm, h2, arm, "plate")
    _incidence_arrow(ax, -arm - 0.5 * arm, 0.0, 0.35 * arm, language)
    _dim(
        ax,
        (-0.85 * arm, -0.5 * h1),
        (-0.85 * arm, 0.5 * h1),
        _mm(h1, language),
        offset=-0.15 * arm,
        tight=True,
    )
    y_h2 = 0.5 * h1 + 1.12 * arm
    _dim(ax, (-0.5 * h2, y_h2), (0.5 * h2, y_h2), _mm(h2, language), tight=True)
    _finish_geometry_axes(
        ax, _t("Plate junction ({junction})", language).format(junction=junction)
    )
    return ax


def plot_junction_result_geometry(
    result: JunctionTransmissionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Junction drawing for a result that retained its thicknesses."""
    if result.thickness1 is None or result.thickness2 is None:
        msg = (
            "This result does not retain the plate thicknesses; call "
            "plot_junction_geometry(junction, thickness1, thickness2)."
        )
        raise ValueError(msg)
    return plot_junction_geometry(
        result.junction,
        result.thickness1,
        result.thickness2,
        ax=ax,
        language=language,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Baffled rectangular plate (radiation efficiency).
# ---------------------------------------------------------------------------
def plot_plate_geometry(
    length_x: float,
    length_y: float,
    ax: Axes | None = None,
    *,
    boundary: str = "simply_supported",
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the baffled rectangular plate of the radiation model, to scale.

    Plate a x b inside its baffle frame, boundary condition in the title.

    :param length_x: Plate length ``a``, in metres.
    :param length_y: Plate width ``b``, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param boundary: Boundary-condition label of the model.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the plate rectangle.
    :return: The axes.
    :raises ValueError: for a side that is not positive and finite.
    """
    _check_language(language)
    # Same reason as the junction thicknesses: a NaN side slipped past the
    # bare ``<= 0.0`` and drew the baffle frame with no plate in it, under a
    # dimension label reading "nan m".
    a = require_positive(length_x, "length_x")
    b = require_positive(length_y, "length_y")
    if ax is None:
        ax = _new_axes()
    frame = 0.18 * max(a, b)
    _material_rect(ax, -frame, -frame, a + 2.0 * frame, b + 2.0 * frame, "rigid")
    _material_rect(ax, 0.0, 0.0, a, b, "plate", **kwargs)
    _dim(ax, (0.0, b + 0.5 * frame), (a, b + 0.5 * frame), _metres(a, language))
    _dim(ax, (a + 0.5 * frame, 0.0), (a + 0.5 * frame, b), _metres(b, language))
    _finish_geometry_axes(
        ax, _t("Baffled plate ({boundary})", language).format(boundary=boundary)
    )
    return ax


def plot_radiation_result_geometry(
    result: RadiationEfficiencyResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Plate drawing for a radiation-efficiency result (always retained)."""
    return plot_plate_geometry(
        result.length_x,
        result.length_y,
        ax=ax,
        boundary=result.boundary,
        language=language,
        **kwargs,
    )
