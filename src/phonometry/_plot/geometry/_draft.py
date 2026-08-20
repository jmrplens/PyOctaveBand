#  Copyright (c) 2026. Jose Manuel Requena Plens
"""The drafting primitives every geometry drawing is assembled from.

A drawing in this package is a technical drawing: rectangles of material,
double-headed dimension lines with their measured length, an incidence arrow,
a microphone, a loudspeaker, and axes stripped to the drawing itself with the
aspect ratio locked so that lengths on paper are lengths in metres. Those marks
are the same whether the subject is an impedance tube or a noise barrier, so
they live here once, together with the axis labels, the legend placement and
the material fill styles they draw with, and the handful of fixed strings the
primitives themselves render.

Localisation goes through ``_check_language`` plus ``_mm``/``_metres``, which
format a length in the reader's language; every module in the package calls
them, and each keeps its own table for the strings only its drawings render.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    theme_fill_alpha,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Axis labels shared by the plan-view and 3-D renderers.
_AXIS_X = "x [m]"
_AXIS_Y = "y [m]"
_AXIS_Z = "z [m]"
#: Shared legend placement of the geometry drawings.
_LEGEND_LOC: Final = "upper right"
#: Tolerance below which a span, offset or label shift counts as zero: a
#: degenerate dimension is skipped before its unit normal would divide by
#: zero, and only an offset that actually moves a line or label changes it.
_EPS: Final = 1e-12
#: Upper edge (degrees, inclusive) of the half-open band (-90, 90] of label
#: rotations that read left-to-right; past it a label would render
#: upside-down, so its rotation is folded back by a half turn.
_MAX_READABLE_ROTATION: Final = 90.0


#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Incident sound": "Sonido incidente",
    "Loudspeaker": "Altavoz",
    "mm": "mm",
    "m": "m",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


def _check_language(language: str) -> None:
    """Reject unknown languages with the shared package error."""
    from ..._i18n import check_language

    check_language(language)


def _mm(value: float, language: str) -> str:
    """A length in millimetres, localised, trimmed (0.05 -> ``"50 mm"``)."""
    from ..._i18n import format_number

    return (
        format_number(value * 1e3, language, decimals=1, trim=True)
        + " "
        + _t("mm", language)
    )


def _metres(value: float, language: str) -> str:
    """A length in metres, localised, trimmed (1.5 -> ``"1.5 m"``)."""
    from ..._i18n import format_number

    return (
        format_number(value, language, decimals=2, trim=True) + " " + _t("m", language)
    )


# ---------------------------------------------------------------------------
# Drafting primitives (matplotlib patches, imported lazily).
# ---------------------------------------------------------------------------
def _dim(
    ax: Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    label: str,
    *,
    offset: float = 0.0,
    fontsize: float = 8.0,
    tight: bool = False,
    label_side: float = 1.0,
    label_upright: bool = False,
) -> None:
    """A drafting dimension: double-headed arrow between two points + label.

    ``offset`` displaces the dimension line perpendicular to ``p1 -> p2``
    (positive = to the left of the direction of travel), with dashed
    extension lines back to the measured points. ``tight`` switches to bar
    ends for spans too short for two arrowheads.

    The label rides beside the dimension line, by default on the side the
    normal points to. A feature narrower than its own label leaves no room
    there, because that is the strip the extension lines cross: ``label_side
    = -1.0`` moves the label to the far side of the dimension line, past the
    end of those lines, and ``label_upright`` sets it horizontally rather
    than along the dimension, anchored by its near edge so that a label
    longer than a short span no longer overruns it.
    """
    if not label:
        return
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    direction = b - a
    length = float(np.hypot(*direction))
    if length < _EPS:
        return
    normal = np.array([-direction[1], direction[0]]) / length
    ao = a + normal * offset
    bo = b + normal * offset
    if abs(offset) > _EPS:
        _dim_extension_lines(ax, ((a, ao), (b, bo)))
    style = "|-|, widthA=0.4, widthB=0.4" if tight else "<->"
    ax.annotate(
        "",
        xy=tuple(bo),
        xytext=tuple(ao),
        arrowprops={"arrowstyle": style, "color": _C_EDGE, "linewidth": 0.9},
        zorder=5,
    )
    mid = (ao + bo) / 2.0
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle > _MAX_READABLE_ROTATION or angle <= -_MAX_READABLE_ROTATION:
        angle += 180.0
    shift = normal * (9.0 * label_side)
    ha, va, rotation = _dim_label_anchor(shift, angle, label_upright)
    ax.annotate(
        label,
        xy=(mid[0], mid[1]),
        xytext=tuple(shift),
        textcoords="offset points",
        fontsize=fontsize,
        ha=ha,
        va=va,
        rotation=rotation,
        rotation_mode="anchor",
        zorder=6,
    )


def _dim_extension_lines(
    ax: Axes, spans: tuple[tuple[np.ndarray, np.ndarray], ...]
) -> None:
    """The dashed witness lines from each measured point to its offset twin."""
    for point, moved in spans:
        ax.plot(
            [point[0], moved[0]],
            [point[1], moved[1]],
            linestyle=":",
            linewidth=0.7,
            color=_C_EDGE,
            zorder=4,
        )


def _dim_label_anchor(
    shift: np.ndarray, angle: float, upright: bool
) -> tuple[str, str, float]:
    """Alignment and rotation of a dimension label.

    Along the dimension by default; an upright label instead anchors by the
    edge nearest the line, on whichever sides the offset actually moves it.
    """
    if not upright:
        return "center", "center", angle
    ha, va = "center", "center"
    if abs(shift[0]) > _EPS:
        ha = "left" if shift[0] > 0.0 else "right"
    if abs(shift[1]) > _EPS:
        va = "bottom" if shift[1] > 0.0 else "top"
    return ha, va, 0.0


#: Fill styles per material kind: (facecolor, hatch, alpha).
_MATERIAL_STYLE: dict[str, tuple[str, str | None, float]] = {
    "air": ("none", None, 1.0),
    "porous": (_C_SECONDARY_LIGHT, "...", 0.9),
    "plate": (_C_MUTED, None, 0.85),
    "membrane": (_C_SECONDARY, None, 0.9),
    "rigid": (_C_MUTED, "//", 0.5),
    "cavity": (_C_PRIMARY_LIGHT, None, 0.5),
}


def _material_rect(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    kind: str,
    **kwargs: Any,
) -> Any:
    """A material cross-section rectangle in the house style; returns it."""
    from matplotlib.patches import Rectangle

    face, hatch, alpha = _MATERIAL_STYLE[kind]
    kwargs.setdefault("facecolor", face)
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 0.9)
    # The opacity reads as material density (a rigid backing is lighter than a
    # plate) and a caller may lighten it further to let an overlay through, but
    # in both cases it is a floor away from the page, not below it: the
    # thinnest layers would otherwise wash out on the page they sit on. The
    # floor follows the drawn face, which the caller may also have overridden.
    face = kwargs["facecolor"]
    alpha = float(kwargs.pop("alpha", alpha))
    if face != "none":
        alpha = max(alpha, theme_fill_alpha(face, ax))
    kwargs["alpha"] = alpha
    patch = Rectangle((x, y), width, height, hatch=hatch, **kwargs)
    ax.add_patch(patch)
    return patch


def _incidence_arrow(
    ax: Axes,
    x: float,
    y: float,
    length: float,
    language: str,
    *,
    downward: bool = False,
) -> None:
    """Incident-sound arrow pointing in +x (or -y), with its label beside."""
    tip = (x, y - length) if downward else (x + length, y)
    ax.annotate(
        "",
        xy=tip,
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": _C_PRIMARY, "linewidth": 1.6},
        zorder=5,
    )
    label = _t("Incident sound", language)
    if downward:
        ax.text(
            x,
            y + 0.08 * length,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
            color=_C_PRIMARY,
        )
    else:
        ax.text(
            x + 0.5 * length,
            y + 0.06 * length,
            label,
            fontsize=8,
            ha="center",
            va="bottom",
            color=_C_PRIMARY,
        )


def _microphone(ax: Axes, x: float, y: float, size: float, label: str) -> None:
    """A flush wall microphone: stem + head circle + label above."""
    from matplotlib.patches import Circle

    ax.plot([x, x], [y, y + 0.55 * size], color=_C_EDGE, linewidth=1.4)
    ax.add_patch(
        Circle(
            (x, y + 0.75 * size),
            0.22 * size,
            facecolor=_C_PRIMARY,
            edgecolor=_C_EDGE,
            linewidth=0.8,
            zorder=5,
        )
    )
    ax.text(x, y + 1.15 * size, label, fontsize=8, ha="center", va="bottom")


def _loudspeaker(
    ax: Axes, x: float, y_centre: float, size: float, language: str
) -> None:
    """A loudspeaker driver: magnet box + cone opening toward +x."""
    from matplotlib.patches import Polygon, Rectangle

    ax.add_patch(
        Rectangle(
            (x - 0.6 * size, y_centre - 0.25 * size),
            0.35 * size,
            0.5 * size,
            facecolor=_C_MUTED,
            edgecolor=_C_EDGE,
            linewidth=0.9,
        )
    )
    ax.add_patch(
        Polygon(
            [
                (x - 0.25 * size, y_centre - 0.18 * size),
                (x - 0.25 * size, y_centre + 0.18 * size),
                (x, y_centre + 0.48 * size),
                (x, y_centre - 0.48 * size),
            ],
            closed=True,
            facecolor=_C_SECONDARY_LIGHT,
            edgecolor=_C_EDGE,
            linewidth=0.9,
        )
    )
    ax.text(
        x - 0.42 * size,
        y_centre - 0.62 * size,
        _t("Loudspeaker", language),
        fontsize=8,
        ha="center",
        va="top",
    )


def _finish_geometry_axes(ax: Axes, title: str) -> None:
    """Equal aspect, no spines/ticks, padded autoscale, bold title."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.margins(0.12)
    ax.set_axis_off()
    ax.set_title(title)
