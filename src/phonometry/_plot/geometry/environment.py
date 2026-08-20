#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawing of the environment domain: the barrier section.

The vertical section through source, screen and receiver over the ground that
the barrier insertion-loss models are stated on: the direct ray, the ray
diffracted over the top edge, and the path difference between them, which is
the single number the attenuation is a function of.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _new_axes,
)
from ._draft import (
    _LEGEND_LOC,
    _check_language,
    _dim,
    _finish_geometry_axes,
    _material_rect,
    _metres,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...environment.propagation.ground_barriers import BarrierInsertionLoss

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Barrier section": "Sección de la barrera",
    "Source": "Fuente",
    "Receiver": "Receptor",
    "Ground": "Suelo",
    "Direct path": "Camino directo",
    "Diffracted path": "Camino difractado",
    "Path difference {delta} m": "Diferencia de camino {delta} m",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Barrier section (source, screen, receiver over ground).
# ---------------------------------------------------------------------------
def plot_barrier_geometry(
    ax: Axes | None = None,
    *,
    source_height: float,
    barrier_distance: float,
    barrier_height: float,
    receiver_distance: float,
    receiver_height: float,
    thickness: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the source-barrier-receiver section to scale.

    Ground line, thin (or thick) screen, the direct path cut by the screen
    and the diffracted path over the top edge(s), with the path-length
    difference annotated. Distances follow
    :func:`~phonometry.environment.barrier_insertion_loss`:
    ``receiver_distance`` is horizontal from the source.

    :param ax: Existing axes, or ``None`` to create a figure.
    :param source_height: Source height above ground, in metres.
    :param barrier_distance: Source-to-barrier horizontal distance, in metres.
    :param barrier_height: Barrier height, in metres.
    :param receiver_distance: Source-to-receiver horizontal distance, in
        metres (> ``barrier_distance``).
    :param receiver_height: Receiver height above ground, in metres.
    :param thickness: Barrier top width, in metres; ``None`` draws a thin
        screen.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the barrier rectangle.
    :return: The axes.
    """
    _check_language(language)
    if min(source_height, barrier_height, receiver_height) < 0.0:
        msg = "Heights must be non-negative."
        raise ValueError(msg)
    if barrier_distance <= 0.0 or receiver_distance <= barrier_distance:
        msg = (
            "'barrier_distance' must be positive and 'receiver_distance' "
            "greater than it."
        )
        raise ValueError(msg)
    if thickness is not None and thickness <= 0.0:
        msg = "'thickness' must be positive when given."
        raise ValueError(msg)
    if ax is None:
        ax = _new_axes()
    e = 0.0 if thickness is None else float(thickness)
    drawn_e = e if e > 0.0 else 0.012 * receiver_distance
    top = barrier_height
    src = (0.0, source_height)
    rcv = (receiver_distance, receiver_height)
    near = (barrier_distance, top)
    far = (barrier_distance + e, top)
    # Ground.
    _material_rect(
        ax,
        -0.08 * receiver_distance,
        -0.04 * receiver_distance,
        1.2 * receiver_distance,
        0.04 * receiver_distance,
        "rigid",
    )
    ax.text(
        1.06 * receiver_distance,
        -0.02 * receiver_distance,
        _t("Ground", language),
        fontsize=8,
        ha="left",
        va="center",
    )
    _material_rect(ax, barrier_distance, 0.0, drawn_e, top, "plate", **kwargs)
    # Paths.
    ax.plot(
        [src[0], rcv[0]],
        [src[1], rcv[1]],
        linestyle="--",
        linewidth=1.1,
        color=_C_MUTED,
        label=_t("Direct path", language),
        zorder=4,
    )
    diff_x = [src[0], near[0]]
    diff_y = [src[1], near[1]]
    if e > 0.0:
        diff_x.append(far[0])
        diff_y.append(far[1])
    diff_x.append(rcv[0])
    diff_y.append(rcv[1])
    ax.plot(
        diff_x,
        diff_y,
        linewidth=1.6,
        color=_C_PRIMARY,
        label=_t("Diffracted path", language),
        zorder=5,
    )
    ax.plot(
        [src[0]],
        [src[1]],
        marker="*",
        markersize=13,
        color=_C_REFERENCE,
        linestyle="none",
        zorder=6,
        label=_t("Source", language),
    )
    ax.plot(
        [rcv[0]],
        [rcv[1]],
        marker="^",
        markersize=8,
        color=_C_PRIMARY,
        linestyle="none",
        zorder=6,
        label=_t("Receiver", language),
    )
    # Path difference over the top (delta = A + B (+ e) - d).
    a_len = float(np.hypot(near[0] - src[0], near[1] - src[1]))
    b_len = float(np.hypot(rcv[0] - far[0], rcv[1] - far[1]))
    d_len = float(np.hypot(rcv[0] - src[0], rcv[1] - src[1]))
    delta = a_len + e + b_len - d_len
    from ..._i18n import format_number

    ax.text(
        barrier_distance,
        top + 0.06 * receiver_distance,
        _t("Path difference {delta} m", language).format(
            delta=format_number(delta, language, decimals=2, trim=True)
        ),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    y_dim = -0.10 * receiver_distance
    _dim(
        ax, (0.0, y_dim), (barrier_distance, y_dim), _metres(barrier_distance, language)
    )
    _dim(
        ax,
        (0.0, y_dim - 0.06 * receiver_distance),
        (receiver_distance, y_dim - 0.06 * receiver_distance),
        _metres(receiver_distance, language),
    )
    _dim(
        ax,
        (barrier_distance - 0.04 * receiver_distance, 0.0),
        (barrier_distance - 0.04 * receiver_distance, top),
        _metres(top, language),
    )
    _finish_geometry_axes(ax, _t("Barrier section", language))
    ax.legend(loc=_LEGEND_LOC, fontsize=8)
    return ax


def plot_barrier_result_geometry(
    result: BarrierInsertionLoss,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Barrier section for a result that retained its geometry."""
    if (
        result.source_height is None
        or result.barrier_distance is None
        or result.barrier_height is None
        or result.receiver_distance is None
        or result.receiver_height is None
    ):
        msg = (
            "This result does not retain its geometry; call "
            "plot_barrier_geometry(...) with the original arguments."
        )
        raise ValueError(msg)
    return plot_barrier_geometry(
        ax=ax,
        source_height=result.source_height,
        barrier_distance=result.barrier_distance,
        barrier_height=result.barrier_height,
        receiver_distance=result.receiver_distance,
        receiver_height=result.receiver_height,
        thickness=result.thickness,
        language=language,
        **kwargs,
    )
