#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the room domain: image sources and the open plan.

Both drawings are plan views of a room with the measurement laid over it. The
image-source plan shows the real source, the receiver and the mirrored sources
of each reflection order in the source plane; the open-plan drawing lays the
ISO 3382-3 microphone line along the workstations and marks the distraction and
privacy distances read off the spatial decay.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import require_positive_array
from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _new_axes,
)
from ._draft import (
    _AXIS_X,
    _AXIS_Y,
    _LEGEND_LOC,
    _check_language,
    _dim,
    _finish_geometry_axes,
    _material_rect,
    _metres,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ...room.image_source import ImageSourceResult
    from ...room.open_plan import OpenPlanResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Image-source room plan (z of the source plane)": "Planta de fuentes imagen (plano z de la fuente)",
    "Source": "Fuente",
    "Receiver": "Receptor",
    "order {n}": "orden {n}",
    _AXIS_X: _AXIS_X,
    _AXIS_Y: _AXIS_Y,
    "m": "m",
    "Open-plan measurement line": "Línea de medida en oficina diáfana",
    "Workstations": "Puestos de trabajo",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Image-source room plan.
# ---------------------------------------------------------------------------
#: Colour cycle for image orders 1, 2, 3, ... (order 0 is the source).
_ORDER_COLOURS = (_C_SECONDARY, _C_TERTIARY, _C_QUATERNARY, _C_MUTED)


def plot_image_source_geometry(
    result: ImageSourceResult,
    ax: Axes | None = None,
    *,
    max_order: int | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the image-source lattice in plan view (x-y), to scale.

    The real room sits at the origin with the source and receiver marked;
    every image source up to ``max_order`` is projected onto the plan and
    coloured by reflection order, with the mirror-room grid dotted behind.

    :param result: An :class:`~phonometry.room.image_source.ImageSourceResult`
        (it retains the room, the positions and the image lattice).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param max_order: Highest reflection order drawn; ``None`` draws
        ``min(result.max_order, 3)`` to keep the plan readable. ``0`` draws
        the room with its source and receiver and no image at all, which is
        the whole of a direct-sound result.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the room-outline rectangle.
    :return: The axes.
    :raises ValueError: for a negative ``max_order``.
    """
    from matplotlib.patches import Rectangle

    _check_language(language)
    from ..._i18n import localize_axes

    # Zero is a cap, not an absence: :func:`~phonometry.room.image_source_rir`
    # accepts ``max_order=0`` and returns the direct sound alone, so the
    # default cap ``min(result.max_order, 3)`` is legitimately 0 for such a
    # result. Refusing it here blamed 'max_order' for a value the caller had
    # never passed; only a negative cap, which no result can carry, is
    # refused, and then the parameter named is one the caller did pass.
    order_cap = min(int(result.max_order), 3) if max_order is None else int(max_order)
    if order_cap < 0:
        msg = "'max_order' must be >= 0."
        raise ValueError(msg)
    if ax is None:
        ax = _new_axes()
    lx, ly = float(result.dimensions[0]), float(result.dimensions[1])
    orders = np.asarray(result.orders)
    keep = orders <= order_cap
    pos = np.asarray(result.image_positions)[keep]
    ords = orders[keep]
    x_min = float(min(pos[:, 0].min(), 0.0)) - 0.2 * lx
    x_max = float(max(pos[:, 0].max(), lx)) + 0.2 * lx
    y_min = float(min(pos[:, 1].min(), 0.0)) - 0.2 * ly
    y_max = float(max(pos[:, 1].max(), ly)) + 0.2 * ly
    # Mirror-room grid.
    for x in np.arange(np.floor(x_min / lx) * lx, x_max + lx, lx):
        ax.plot(
            [x, x],
            [y_min, y_max],
            linestyle=":",
            linewidth=0.6,
            color=_C_MUTED,
            zorder=1,
        )
    for y in np.arange(np.floor(y_min / ly) * ly, y_max + ly, ly):
        ax.plot(
            [x_min, x_max],
            [y, y],
            linestyle=":",
            linewidth=0.6,
            color=_C_MUTED,
            zorder=1,
        )
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 2.0)
    ax.add_patch(Rectangle((0.0, 0.0), lx, ly, zorder=3, **kwargs))
    for order in range(order_cap, 0, -1):
        sel = ords == order
        if not np.any(sel):
            continue
        colour = _ORDER_COLOURS[(order - 1) % len(_ORDER_COLOURS)]
        ax.plot(
            pos[sel, 0],
            pos[sel, 1],
            "o",
            markersize=4,
            color=colour,
            linestyle="none",
            zorder=4,
            label=_t("order {n}", language).format(n=order),
        )
    ax.plot(
        [result.source[0]],
        [result.source[1]],
        marker="*",
        markersize=13,
        color=_C_REFERENCE,
        linestyle="none",
        zorder=5,
        label=_t("Source", language),
    )
    ax.plot(
        [result.receiver[0]],
        [result.receiver[1]],
        marker="^",
        markersize=8,
        color=_C_PRIMARY,
        linestyle="none",
        zorder=5,
        label=_t("Receiver", language),
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(_t(_AXIS_X, language))
    ax.set_ylabel(_t(_AXIS_Y, language))
    ax.set_title(
        _t("Image-source room plan (z of the source plane)", language),
    )
    ax.legend(loc=_LEGEND_LOC, fontsize=8)
    localize_axes(ax, language)
    return ax


# ---------------------------------------------------------------------------
# Open-plan measurement line (plan view).
# ---------------------------------------------------------------------------
def plot_open_plan_geometry(
    positions: ArrayLike,
    ax: Axes | None = None,
    *,
    rd: float | None = None,
    rp: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the open-plan measurement line to scale.

    Source at the origin, the microphone line across the workstations, and
    the distraction and privacy distances marked on the axis when given.

    :param positions: Microphone distances from the source, in metres (1-D).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param rd: Distraction distance, in metres, or ``None``.
    :param rp: Privacy distance, in metres, or ``None``.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the microphone scatter.
    :return: The axes.
    :raises ValueError: for fewer than two positions, or a position that is
        not a finite, strictly positive distance.
    """
    _check_language(language)
    # Through the shared array guard rather than a bare ``pos <= 0.0``: every
    # comparison against a NaN is False, so a NaN distance passed the old
    # guard and came out as a span dimension reading 'nan m' with the
    # workstation blocks silently missing. ISO 3382-3 positions are measured
    # distances from the source, and the one producer,
    # :func:`~phonometry.room.open_plan_metrics`, pins them finite already.
    pos = np.sort(require_positive_array(np.ravel(positions), "positions"))
    if pos.size < 2:  # noqa: PLR2004
        msg = "'positions' needs at least two positive distances."
        raise ValueError(msg)
    if ax is None:
        ax = _new_axes()
    span = float(pos.max())
    desk = 0.16 * span / max(pos.size - 1, 1)
    # Workstation blocks midway between consecutive microphones.
    from itertools import pairwise

    for left, right in pairwise(pos):
        centre = 0.5 * (left + right)
        _material_rect(
            ax,
            centre - desk,
            -0.5 * desk,
            2.0 * desk,
            desk,
            "plate",
            linewidth=0.6,
            alpha=0.5,
        )
    ax.text(
        0.12 * span,
        3.0 * desk,
        _t("Workstations", language),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    ax.plot(
        [0.0],
        [0.0],
        marker="*",
        markersize=13,
        color=_C_REFERENCE,
        linestyle="none",
        zorder=6,
    )
    ax.text(0.0, -1.4 * desk, _t("Source", language), fontsize=8, ha="center", va="top")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("s", 18)
    ax.scatter(pos, np.zeros_like(pos), zorder=5, **kwargs)
    ax.plot(
        [0.0, span], [0.0, 0.0], color=_C_MUTED, linewidth=0.8, linestyle=":", zorder=2
    )
    from ..._i18n import format_number

    for value, key in ((rd, "r$_D$"), (rp, "r$_P$")):
        if value is not None and np.isfinite(value):
            ax.plot(
                [value, value],
                [-2.2 * desk, 2.2 * desk],
                color=_C_SECONDARY,
                linewidth=1.2,
                linestyle="--",
            )
            ax.text(
                value,
                2.4 * desk,
                key
                + " = "
                + format_number(value, language, decimals=1, trim=True)
                + " "
                + _t("m", language),
                fontsize=8,
                ha="center",
                va="bottom",
            )
    _dim(ax, (0.0, -2.6 * desk), (span, -2.6 * desk), _metres(span, language))
    _finish_geometry_axes(ax, _t("Open-plan measurement line", language))
    return ax


def plot_open_plan_result_geometry(
    result: OpenPlanResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Measurement-line drawing for a result that retained its positions."""
    if result.positions_m is None:
        msg = (
            "This result does not retain its microphone positions; call "
            "plot_open_plan_geometry(positions)."
        )
        raise ValueError(msg)
    return plot_open_plan_geometry(
        result.positions_m,
        ax=ax,
        rd=result.rd,
        rp=result.rp,
        language=language,
        **kwargs,
    )
