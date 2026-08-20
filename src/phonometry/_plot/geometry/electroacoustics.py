#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the electroacoustics domain: sources and the loop.

The baffled piston is drawn with its far-field directivity lobe on the same
axes as the radiator, so the beam width can be read against the radius that
produced it. The sound-reinforcement drawing is the plan of the four distances
talker, microphone, loudspeaker and listener that the feedback stability margin
is computed from, with the signal path and the feedback path marked.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np
from numpy.typing import ArrayLike

from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    _C_TERTIARY,
    _new_axes,
)
from ._draft import (
    _LEGEND_LOC,
    _check_language,
    _dim,
    _finish_geometry_axes,
    _material_rect,
    _metres,
    _mm,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...electroacoustics.piston import RadiatingPistonResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Baffled piston": "Pistón en pantalla infinita",
    "Baffle": "Pantalla",
    "Normalised directivity": "Directividad normalizada",
    "Sound-reinforcement feedback loop": "Lazo de realimentación del refuerzo sonoro",
    "Talker (T)": "Hablante (T)",
    "Microphone (M)": "Micrófono (M)",
    "Loudspeaker (H)": "Altavoz (H)",
    "Listener (L)": "Oyente (L)",
    "Feedback path": "Camino de realimentación",
    "Signal path": "Camino de la señal",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Baffled piston with its directivity lobe.
# ---------------------------------------------------------------------------
@overload
def plot_piston_geometry(
    radius: float,
    ax: Axes | None = ...,
    *,
    angles: ArrayLike,
    directivity: ArrayLike,
    lobe_label: str | None = ...,
    language: str = ...,
    **kwargs: Any,
) -> Axes: ...


@overload
def plot_piston_geometry(
    radius: float,
    ax: Axes | None = ...,
    *,
    lobe_label: str | None = ...,
    language: str = ...,
    **kwargs: Any,
) -> Axes: ...


def plot_piston_geometry(
    radius: float,
    ax: Axes | None = None,
    *,
    angles: ArrayLike | None = None,
    directivity: ArrayLike | None = None,
    lobe_label: str | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw a baffled piston to scale, optionally with a directivity lobe.

    The rigid baffle is the vertical wall, the piston the plate of radius
    ``a`` set into it; when ``angles``/``directivity`` are given the
    normalised far-field lobe is overlaid on the radiation side.

    :param radius: Piston radius ``a``, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param angles: Far-field angles, in radians (0 on axis), matching
        ``directivity``.
    :param directivity: Linear directivity values in ``[0, 1]``.
    :param lobe_label: Optional legend label for the lobe (e.g. the ``ka``).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the piston rectangle.
    :return: The axes.
    """
    _check_language(language)
    if radius <= 0.0:
        raise ValueError("'radius' must be positive.")
    if (angles is None) != (directivity is None):
        raise ValueError("Give 'angles' and 'directivity' together.")
    if ax is None:
        ax = _new_axes()
    a = float(radius)
    baffle_h = 3.0 * a
    wall = 0.35 * a
    for y0 in (a, -a - baffle_h):
        _material_rect(ax, -wall, y0, wall, baffle_h, "rigid")
    ax.text(
        -0.5 * wall,
        a + baffle_h * 1.03,
        _t("Baffle", language),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    kwargs.setdefault("facecolor", _C_SECONDARY_LIGHT)
    kwargs.setdefault("edgecolor", _C_EDGE)
    from matplotlib.patches import Rectangle

    piston = Rectangle((-wall, -a), 0.6 * wall, 2.0 * a, **kwargs)
    ax.add_patch(piston)
    ax.plot([0.0, 3.2 * a], [0.0, 0.0], linestyle=":", linewidth=0.8, color=_C_MUTED)
    if angles is not None and directivity is not None:
        ang = np.asarray(angles, dtype=np.float64)
        d_lin = np.abs(np.asarray(directivity, dtype=np.float64))
        if ang.shape != d_lin.shape:
            raise ValueError("'angles' and 'directivity' must match.")
        peak = float(d_lin.max())
        if peak > 0.0:
            scale = 2.8 * a / peak
            ax.plot(
                d_lin * scale * np.cos(ang),
                d_lin * scale * np.sin(ang),
                linewidth=1.6,
                color=_C_PRIMARY,
                label=lobe_label or _t("Normalised directivity", language),
            )
            ax.legend(loc=_LEGEND_LOC, fontsize=8)
    _dim(ax, (-1.4 * wall, -a), (-1.4 * wall, a), "2a = " + _mm(2.0 * a, language))
    _finish_geometry_axes(ax, _t("Baffled piston", language))
    return ax


def plot_piston_result_geometry(
    result: RadiatingPistonResult,
    ax: Axes | None = None,
    *,
    frequency_index: int = -1,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Baffled-piston drawing for a radiating-piston result.

    The result always retains ``radius``; when it also carries a computed
    directivity, the lobe of the selected frequency is overlaid with its
    ``ka`` in the legend.
    """
    angles = None
    lobe = None
    label = None
    if result.angles is not None and result.directivity is not None:
        directivity = np.asarray(result.directivity, dtype=np.float64)
        ka = np.atleast_1d(np.asarray(result.ka, dtype=np.float64))
        n_rows = int(directivity.shape[0])
        if not -n_rows <= frequency_index < n_rows or ka.size < n_rows:
            raise ValueError(
                f"'frequency_index' must index the {n_rows} computed frequencies."
            )
        row = directivity[frequency_index]
        angles = np.asarray(result.angles, dtype=np.float64)
        lobe = row
        from ..._i18n import format_number

        label = "ka = " + format_number(
            float(ka[frequency_index]), language, decimals=1, trim=True
        )
    return plot_piston_geometry(
        result.radius,
        ax=ax,
        angles=angles,
        directivity=lobe,
        lobe_label=label,
        language=language,
        **kwargs,
    )


def plot_sound_reinforcement_geometry(
    talker_distance: float,
    microphone_distance: float,
    listener_distance: float,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the four points of a reinforcement feedback loop.

    A schematic in the layout of Long's Figure 18.15, with each path annotated
    with its own length: the talker ``T`` close in front of the microphone
    ``M``, the flown loudspeaker ``H`` above and downstage of it, and the
    average listener ``L`` out in the audience. The signal path
    ``T -> M`` and ``H -> L`` is solid, the feedback path ``H -> M`` dashed.
    The two loudspeaker paths are the direct-field levels ``L_H-M`` and
    ``L_H-L`` that drive
    :func:`phonometry.electroacoustics.feedback_stability`; the drawing is
    deliberately *not* to scale, because a talker 0.3 m from the microphone
    and a listener 20 m from the loudspeaker cannot share one usable scale.

    :param talker_distance: Talker-to-microphone distance, m.
    :param microphone_distance: Loudspeaker-to-microphone distance, m.
    :param listener_distance: Loudspeaker-to-listener distance, m.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the feedback-path ``Axes.plot``.
    :return: The axes.
    :raises ValueError: If any distance is not positive and finite.
    """
    from matplotlib.patches import Polygon

    _check_language(language)
    lengths = (
        float(talker_distance),
        float(microphone_distance),
        float(listener_distance),
    )
    if not all(np.isfinite(v) and v > 0.0 for v in lengths):
        raise ValueError("The three distances must be positive and finite.")
    d_tm, d_hm, d_hl = lengths
    if ax is None:
        ax = _new_axes()

    # Schematic coordinates (arbitrary units); only the annotations carry the
    # real lengths.
    t_xy, m_xy, h_xy, l_xy = (1.0, 0.0), (2.2, 0.0), (3.6, 2.4), (9.0, 0.0)
    ax.plot([0.2, 9.8], [0.0, 0.0], color=_C_MUTED, linewidth=1.0, zorder=1)

    ax.plot(
        [t_xy[0]],
        [t_xy[1]],
        marker="*",
        markersize=14,
        color=_C_REFERENCE,
        linestyle="none",
        zorder=6,
    )
    ax.text(
        t_xy[0], -0.28, _t("Talker (T)", language), fontsize=8, ha="center", va="top"
    )
    ax.plot([m_xy[0], m_xy[0]], [0.0, 0.5], color=_C_EDGE, linewidth=1.4, zorder=5)
    ax.plot(
        [m_xy[0]],
        [0.62],
        marker="o",
        markersize=9,
        color=_C_PRIMARY,
        markeredgecolor=_C_EDGE,
        linestyle="none",
        zorder=6,
    )
    ax.text(
        m_xy[0] + 0.15,
        0.82,
        _t("Microphone (M)", language),
        fontsize=8,
        ha="left",
        va="bottom",
    )
    ax.add_patch(
        Polygon(
            [
                (h_xy[0] - 0.28, h_xy[1] - 0.22),
                (h_xy[0] - 0.28, h_xy[1] + 0.22),
                (h_xy[0] + 0.22, h_xy[1] + 0.5),
                (h_xy[0] + 0.22, h_xy[1] - 0.5),
            ],
            closed=True,
            facecolor=_C_SECONDARY_LIGHT,
            edgecolor=_C_EDGE,
            linewidth=0.9,
            zorder=5,
        )
    )
    ax.text(
        h_xy[0],
        h_xy[1] + 0.62,
        _t("Loudspeaker (H)", language),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    ax.plot(
        [l_xy[0]],
        [l_xy[1]],
        marker="o",
        markersize=9,
        color=_C_TERTIARY,
        markeredgecolor=_C_EDGE,
        linestyle="none",
        zorder=6,
    )
    ax.text(
        l_xy[0], -0.28, _t("Listener (L)", language), fontsize=8, ha="center", va="top"
    )

    ax.plot(
        [t_xy[0], m_xy[0]],
        [0.12, 0.5],
        color=_C_PRIMARY,
        linewidth=1.6,
        zorder=4,
        label=_t("Signal path", language),
    )
    ax.plot(
        [h_xy[0], l_xy[0]],
        [h_xy[1], l_xy[1]],
        color=_C_PRIMARY,
        linewidth=1.6,
        zorder=4,
    )
    kwargs.setdefault("color", _C_SECONDARY)
    kwargs.setdefault("linewidth", 1.4)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("label", _t("Feedback path", language))
    ax.plot([h_xy[0], m_xy[0]], [h_xy[1], 0.62], zorder=4, **kwargs)

    for (x0, y0), (x1, y1), value, dy in (
        (t_xy, (m_xy[0], 0.5), d_tm, 0.16),
        ((h_xy[0], h_xy[1]), (m_xy[0], 0.62), d_hm, 0.12),
        (h_xy, l_xy, d_hl, 0.12),
    ):
        ax.text(
            0.5 * (x0 + x1),
            0.5 * (y0 + y1) + dy,
            _metres(value, language),
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_ylim(-0.9, 3.6)
    ax.legend(loc=_LEGEND_LOC, fontsize=8)
    _finish_geometry_axes(ax, _t("Sound-reinforcement feedback loop", language))
    return ax
