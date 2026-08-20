#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawing of the simulation domain: the FDTD grid before it runs.

The 2-D finite-difference domain as configured: the obstacle map, the sponge
layers on the sides that absorb, the impedance and rigid edges, the source and
any probes. Drawn without stepping the simulation, so a set-up can be checked
before the hours of computation that follow it.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_REFERENCE,
    _C_SECONDARY,
    _new_axes,
)
from ._draft import (
    _AXIS_X,
    _AXIS_Y,
    _LEGEND_LOC,
    _check_language,
    _material_rect,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...simulation.fdtd import FDTD2D

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    _AXIS_X: _AXIS_X,
    _AXIS_Y: _AXIS_Y,
    "Source": "Fuente",
    "FDTD domain": "Dominio FDTD",
    "Sponge layer": "Capa esponja",
    "Impedance edge": "Borde de impedancia",
    "Rigid edge": "Borde rígido",
    "Probe": "Sonda",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# FDTD domain (drawn without running the simulation).
# ---------------------------------------------------------------------------
def _fdtd_sponge_bands(
    ax: Axes, sim: FDTD2D, lx: float, ly: float, handles: dict[str, Any]
) -> None:
    """Shade the sponge layers on their configured sides."""
    depth = sim.sponge_width * sim.dx
    if depth <= 0.0:
        return
    rects = {
        "left": (0.0, 0.0, depth, ly),
        "right": (lx - depth, 0.0, depth, ly),
        "top": (0.0, 0.0, lx, depth),
        "bottom": (0.0, ly - depth, lx, depth),
    }
    for side in sim.sponge_sides:
        x0, y0, width, height = rects[side]
        handles["sponge"] = _material_rect(
            ax,
            x0,
            y0,
            width,
            height,
            "cavity",
            edgecolor="none",
        )


def _fdtd_edges(
    ax: Axes, sim: FDTD2D, lx: float, ly: float, handles: dict[str, Any]
) -> None:
    """Mark impedance edges and rigid edges (sponge sides stay open)."""
    segments = {
        "left": ((0.0, 0.0), (0.0, ly)),
        "right": ((lx, 0.0), (lx, ly)),
        "top": ((0.0, 0.0), (lx, 0.0)),
        "bottom": ((0.0, ly), (lx, ly)),
    }
    for side, seg in segments.items():
        if side in sim.edge_impedance:
            colour, key = _C_SECONDARY, "impedance"
        elif side in sim.sponge_sides:
            continue
        else:
            colour, key = _C_EDGE, "rigid"
        (line,) = ax.plot(
            [seg[0][0], seg[1][0]],
            [seg[0][1], seg[1][1]],
            color=colour,
            linewidth=2.6,
        )
        handles[key] = line


def plot_fdtd_domain(
    sim: FDTD2D,
    ax: Axes | None = None,
    *,
    probes: ArrayLike | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw an FDTD2D domain before running it: what will be simulated.

    Domain extent in metres (same orientation as the snapshot renderer),
    obstacles in grey, sponge layers shaded on their sides, impedance edges
    and rigid edges marked, sources starred and optional probe positions
    dotted.

    :param sim: A :class:`~phonometry.simulation.FDTD2D` instance (fresh or
        already stepped; only its static configuration is drawn).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param probes: Optional probe positions to preview, shape ``(N, 2)`` as
        ``(x, y)`` in metres (the ``fdtd_simulation`` convention).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the domain ``imshow`` of the obstacle layer.
    :return: The axes.
    """
    _check_language(language)
    from ..._i18n import localize_axes

    pts: NDArray[np.float64] | None = None
    if probes is not None:
        pts = np.asarray(probes, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("'probes' must have shape (N, 2).")
    if ax is None:
        ax = _new_axes()
    ny, nx = sim.p.shape
    lx, ly = nx * sim.dx, ny * sim.dx
    extent = (0.0, lx, ly, 0.0)
    handles: dict[str, Any] = {}
    mask = getattr(sim, "_obstacle", None)
    if mask is not None:
        overlay = np.ma.masked_where(~mask, np.ones_like(mask, dtype=float))
        kwargs.setdefault("cmap", "gray")
        kwargs.setdefault("vmin", 0.0)
        kwargs.setdefault("vmax", 2.0)
        ax.imshow(
            overlay, extent=extent, origin="upper", interpolation="nearest", **kwargs
        )
    _fdtd_sponge_bands(ax, sim, lx, ly, handles)
    _fdtd_edges(ax, sim, lx, ly, handles)
    for src in getattr(sim, "_sources", ()):  # star per source
        (marker,) = ax.plot(
            [(src.ix + 0.5) * sim.dx],
            [(src.iy + 0.5) * sim.dx],
            marker="*",
            markersize=11,
            color=_C_REFERENCE,
            linestyle="none",
        )
        handles["source"] = marker
    if pts is not None:
        (dots,) = ax.plot(
            pts[:, 0],
            pts[:, 1],
            marker="o",
            markersize=5,
            color=_C_MUTED,
            markeredgecolor="black",
            linestyle="none",
        )
        handles["probe"] = dots
    ax.set_xlim(0.0, lx)
    ax.set_ylim(ly, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel(_t(_AXIS_X, language))
    ax.set_ylabel(_t(_AXIS_Y, language))
    ax.set_title(_t("FDTD domain", language))
    label_keys = {
        "sponge": "Sponge layer",
        "impedance": "Impedance edge",
        "rigid": "Rigid edge",
        "source": "Source",
        "probe": "Probe",
    }
    if handles:
        ax.legend(
            handles.values(),
            [_t(label_keys[k], language) for k in handles],
            loc=_LEGEND_LOC,
            fontsize=8,
        )
    localize_axes(ax, language)
    return ax
