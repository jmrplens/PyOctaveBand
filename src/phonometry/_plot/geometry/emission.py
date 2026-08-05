#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the emission domain: where the microphones go.

Sound power and sound intensity are measured by placing microphones around a
source, and both drawings here are that placement. The 3-D scatter shows a
measurement surface of ISO 3744/3745/3746 with its microphone array on the
hemisphere or box; the p-p probe section shows the two pressure microphones
face to face across their spacer, the separation that sets the usable
frequency range.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_SECONDARY_LIGHT,
    _import_pyplot,
    _new_axes,
    theme_fill_alpha,
)
from ._draft import (
    _AXIS_X,
    _AXIS_Y,
    _AXIS_Z,
    _check_language,
    _dim,
    _finish_geometry_axes,
    _material_rect,
    _mm,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...emission.intensity import IntensityResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    _AXIS_X: _AXIS_X,
    _AXIS_Y: _AXIS_Y,
    _AXIS_Z: _AXIS_Z,
    "Microphone positions": "Posiciones de micrófono",
    "Reflecting plane": "Plano reflectante",
    "p-p intensity probe": "Sonda de intensidad p-p",
    "Spacer": "Espaciador",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Microphone position arrays (ISO 3744/3745/3746), 3-D.
# ---------------------------------------------------------------------------
def plot_microphone_positions(
    positions: ArrayLike,
    ax: Any | None = None,
    *,
    radius: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Any:
    """Draw a microphone position array on its measurement surface, in 3-D.

    Numbered microphone points with a wireframe of the hemisphere (or full
    sphere when positions dip below the reflecting plane) of the given
    ``radius``; pairs with
    :func:`~phonometry.emission.measurement_positions` and
    :func:`~phonometry.emission.precision_positions`, whose ``(N, 3)``
    arrays it accepts directly.

    :param positions: Cartesian microphone positions, shape ``(N, 3)``, in
        metres.
    :param ax: Existing 3-D axes (``projection="3d"``), or ``None`` to
        create a figure.
    :param radius: Surface radius for the wireframe, in metres; ``None``
        uses the largest position norm.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the microphone ``scatter``.
    :return: The 3-D axes.
    """
    _check_language(language)
    pts = np.asarray(positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError("'positions' must have shape (N, 3) with N >= 1.")
    if radius is not None and radius <= 0.0:
        raise ValueError("'radius' must be positive when given.")
    r = float(radius) if radius is not None else float(
        np.linalg.norm(pts, axis=1).max()
    )
    if ax is None:
        plt = _import_pyplot()
        _fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    full_sphere = bool(np.any(pts[:, 2] < -1e-9))
    theta_max = np.pi if full_sphere else 0.5 * np.pi
    theta = np.linspace(0.0, theta_max, 13)
    phi = np.linspace(0.0, 2.0 * np.pi, 25)
    t_grid, p_grid = np.meshgrid(theta, phi)
    ax.plot_wireframe(
        r * np.sin(t_grid) * np.cos(p_grid),
        r * np.sin(t_grid) * np.sin(p_grid),
        r * np.cos(t_grid),
        color=_C_MUTED, linewidth=0.4, alpha=0.5,
    )
    if not full_sphere:
        # Reflecting plane disc at z = 0.
        disc_r = np.linspace(0.0, 1.15 * r, 2)
        d_grid, a_grid = np.meshgrid(disc_r, phi)
        # Translucent so the wireframe behind it keeps reading, with the
        # opacity derived from the page and the surface left unshaded so the
        # drawn colour is the one that was measured.
        ax.plot_surface(
            d_grid * np.cos(a_grid), d_grid * np.sin(a_grid),
            np.zeros_like(d_grid), color=_C_SECONDARY_LIGHT, linewidth=0.0,
            alpha=theme_fill_alpha(_C_SECONDARY_LIGHT, ax), shade=False,
        )
        ax.text(
            1.1 * r, 0.0, 0.0, _t("Reflecting plane", language), fontsize=8,
        )
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("s", 30)
    kwargs.setdefault("depthshade", False)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kwargs)
    for index, (x, y, z) in enumerate(pts, start=1):
        ax.text(x, y, z, f" {index}", fontsize=7)
    ax.set_xlabel(_t(_AXIS_X, language))
    ax.set_ylabel(_t(_AXIS_Y, language))
    ax.set_zlabel(_t(_AXIS_Z, language))
    ax.set_title(_t("Microphone positions", language), fontweight="bold")
    ax.set_box_aspect((1.0, 1.0, 0.55 if not full_sphere else 1.0))
    return ax


# ---------------------------------------------------------------------------
# p-p intensity probe.
# ---------------------------------------------------------------------------
def plot_pp_probe_geometry(
    spacing: float = 0.012,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the face-to-face p-p intensity probe to scale.

    Two phase-matched microphones separated by the solid spacer, with the
    intensity axis through both; default is the classic 12 mm spacer.

    :param spacing: Microphone separation ``dr``, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the spacer rectangle.
    :return: The axes.
    """
    from matplotlib.patches import Circle, Rectangle

    _check_language(language)
    if spacing <= 0.0:
        raise ValueError("'spacing' must be positive.")
    if ax is None:
        ax = _new_axes()
    dr = float(spacing)
    d_mic = 0.55 * dr                       # half-inch capsule vs 12 mm
    body = 2.2 * dr
    # Spacer between the two face-to-face capsules.
    _material_rect(ax, -0.5 * dr + 0.5 * d_mic, -0.18 * dr,
                   dr - d_mic, 0.36 * dr, "cavity", **kwargs)
    ax.text(0.0, 0.28 * dr, _t("Spacer", language), fontsize=8,
            ha="center", va="bottom")
    for sign in (-1.0, 1.0):
        centre = sign * 0.5 * dr
        ax.add_patch(Circle((centre, 0.0), 0.5 * d_mic,
                            facecolor=_C_PRIMARY, edgecolor=_C_EDGE,
                            linewidth=0.8, zorder=5))
        ax.add_patch(Rectangle(
            (centre + sign * 0.5 * d_mic, -0.30 * dr)
            if sign > 0 else (centre - body - 0.5 * d_mic, -0.30 * dr),
            body, 0.60 * dr, facecolor=_C_MUTED, edgecolor=_C_EDGE,
            linewidth=0.8, alpha=0.6,
        ))
    x_arrow = 0.5 * dr + d_mic + body
    # Annotation arrows do not autoscale: a silent sentinel keeps the tip
    # inside the axes limits.
    ax.plot([x_arrow + 1.5 * dr], [0.0], linestyle="none")
    ax.annotate(
        "", xy=(x_arrow + 1.3 * dr, 0.0), xytext=(x_arrow + 0.2 * dr, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": _C_PRIMARY,
                    "linewidth": 1.6},
    )
    ax.text(x_arrow + 0.75 * dr, 0.1 * dr, "I$_r$", fontsize=9,
            ha="center", va="bottom", color=_C_PRIMARY)
    _dim(ax, (-0.5 * dr, -0.5 * dr), (0.5 * dr, -0.5 * dr),
         _mm(dr, language), tight=True)
    _finish_geometry_axes(ax, _t("p-p intensity probe", language))
    return ax


def plot_intensity_result_geometry(
    result: IntensityResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Probe drawing for a result that retained its spacer."""
    if result.spacing is None:
        raise ValueError(
            "This result does not retain its microphone spacing; call "
            "plot_pp_probe_geometry(spacing)."
        )
    return plot_pp_probe_geometry(
        result.spacing, ax=ax, language=language, **kwargs
    )
