#  Copyright (c) 2026. Jose M. Requena-Plens
"""To-scale geometry drawings of measurement set-ups and treatment designs.

Every renderer draws the *physical* device the way a lab manual would: a
dimensioned cross-section in metres with ``ax.set_aspect("equal")``, so a
100 mm tube really is twice as tall as a 50 mm one. They complement the
spectral ``plot()`` renderers: the geometry is what you build, the spectrum is
what you measure.

Lazy-imported from the ``plot()``/``plot_geometry()`` methods of the domain
objects and from the public ``plot_*_geometry`` functions; domain classes are
referenced only under ``TYPE_CHECKING`` so this rendering leaf never imports
domain code at module level (see ``tests/test_package_architecture.py``).
Layer dataclasses are dispatched by class name at runtime for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    _new_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..materials.diffuser_design import DiffuserPolarResponse
    from ..materials.impedance_tube import ImpedanceTubeResult, TransferMatrix
    from ..materials.porous_absorber import Layer, LayeredAbsorberResult
    from ..materials.slow_sound_absorber import (
        HelmholtzResonator,
        SlitResonatorAbsorberResult,
    )

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Air": "Aire",
    "Porous": "Poroso",
    "Perforated plate": "Placa perforada",
    "Microperforated plate": "Placa microperforada",
    "Membrane": "Membrana",
    "Rigid backing": "Respaldo rígido",
    "Incident sound": "Sonido incidente",
    "Layered absorber cross-section": "Sección del absorbente multicapa",
    "Helmholtz resonator cross-section": "Sección del resonador de Helmholtz",
    "Slit metamaterial absorber cross-section (one period)":
        "Sección del absorbente metamaterial de rendija (un periodo)",
    "QRD well profile": "Perfil de pozos del difusor QRD",
    "Impedance tube (ISO 10534-2), to scale":
        "Tubo de impedancia (ISO 10534-2), a escala",
    "Transmission tube (ASTM E2611), to scale":
        "Tubo de transmisión (ASTM E2611), a escala",
    "Loudspeaker": "Altavoz",
    "Sample": "Muestra",
    "Termination": "Terminación",
    "Cross-section": "Sección transversal",
    "Plane-wave range {fl} to {fu} Hz": "Rango de onda plana {fl} a {fu} Hz",
    "depth sequence {seq}": "secuencia de profundidades {seq}",
    "Slit": "Rendija",
    "mm": "mm",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


def _mm(value: float, language: str) -> str:
    """A length in millimetres, localised, trimmed (0.05 -> ``"50 mm"``)."""
    from .._i18n import format_number

    return (
        format_number(value * 1e3, language, decimals=1, trim=True)
        + " " + _t("mm", language)
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
) -> None:
    """A drafting dimension: double-headed arrow between two points + label.

    ``offset`` displaces the dimension line perpendicular to ``p1 -> p2``
    (positive = to the left of the direction of travel), with dashed
    extension lines back to the measured points. ``tight`` switches to bar
    ends for spans too short for two arrowheads.
    """
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    direction = b - a
    length = float(np.hypot(*direction))
    if length == 0.0:
        return
    normal = np.array([-direction[1], direction[0]]) / length
    ao = a + normal * offset
    bo = b + normal * offset
    if offset != 0.0:
        for point, moved in ((a, ao), (b, bo)):
            ax.plot(
                [point[0], moved[0]], [point[1], moved[1]],
                linestyle=":", linewidth=0.7, color=_C_EDGE, zorder=4,
            )
    style = "|-|, widthA=0.4, widthB=0.4" if tight else "<->"
    ax.annotate(
        "", xy=tuple(bo), xytext=tuple(ao),
        arrowprops={"arrowstyle": style, "color": _C_EDGE, "linewidth": 0.9},
        zorder=5,
    )
    if not label:
        return
    mid = (ao + bo) / 2.0
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    if angle > 90.0 or angle <= -90.0:
        angle += 180.0
    ax.annotate(
        label, xy=(mid[0], mid[1]), xytext=tuple(normal * 9.0),
        textcoords="offset points", fontsize=fontsize, ha="center",
        va="center", rotation=angle, rotation_mode="anchor", zorder=6,
    )


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
    kwargs.setdefault("alpha", alpha)
    patch = Rectangle((x, y), width, height, hatch=hatch, **kwargs)
    ax.add_patch(patch)
    return patch


def _incidence_arrow(
    ax: Axes, x: float, y: float, length: float, language: str,
    *, downward: bool = False,
) -> None:
    """Incident-sound arrow pointing in +x (or -y), with its label beside."""
    tip = (x, y - length) if downward else (x + length, y)
    ax.annotate(
        "", xy=tip, xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": _C_PRIMARY,
                    "linewidth": 1.6},
        zorder=5,
    )
    if downward:
        ax.text(
            x, y + 0.08 * length, _t("Incident sound", language),
            fontsize=8, ha="center", va="bottom", color=_C_PRIMARY,
        )
    else:
        ax.text(
            x + 0.5 * length, y + 0.06 * length,
            _t("Incident sound", language),
            fontsize=8, ha="center", va="bottom", color=_C_PRIMARY,
        )


def _microphone(ax: Axes, x: float, y: float, size: float, label: str) -> None:
    """A flush wall microphone: stem + head circle + label above."""
    from matplotlib.patches import Circle

    ax.plot([x, x], [y, y + 0.55 * size], color=_C_EDGE, linewidth=1.4)
    ax.add_patch(
        Circle((x, y + 0.75 * size), 0.22 * size, facecolor=_C_PRIMARY,
               edgecolor=_C_EDGE, linewidth=0.8, zorder=5)
    )
    ax.text(x, y + 1.15 * size, label, fontsize=8, ha="center", va="bottom")


def _loudspeaker(
    ax: Axes, x: float, y_centre: float, size: float, language: str
) -> None:
    """A loudspeaker driver: magnet box + cone opening toward +x."""
    from matplotlib.patches import Polygon, Rectangle

    ax.add_patch(
        Rectangle((x - 0.6 * size, y_centre - 0.25 * size), 0.35 * size,
                  0.5 * size, facecolor=_C_MUTED, edgecolor=_C_EDGE,
                  linewidth=0.9)
    )
    ax.add_patch(
        Polygon(
            [(x - 0.25 * size, y_centre - 0.18 * size),
             (x - 0.25 * size, y_centre + 0.18 * size),
             (x, y_centre + 0.48 * size),
             (x, y_centre - 0.48 * size)],
            closed=True, facecolor=_C_SECONDARY_LIGHT, edgecolor=_C_EDGE,
            linewidth=0.9,
        )
    )
    ax.text(
        x - 0.42 * size, y_centre - 0.62 * size, _t("Loudspeaker", language),
        fontsize=8, ha="center", va="top",
    )


def _finish_geometry_axes(ax: Axes, title: str) -> None:
    """Equal aspect, no spines/ticks, padded autoscale, bold title."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale()
    ax.margins(0.12)
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold")


def _tube_frequency_note(
    ax: Axes,
    x: float,
    y: float,
    f_range: tuple[float, float],
    language: str,
) -> None:
    """The plane-wave working range printed under a tube drawing."""
    from .._i18n import format_number

    fl, fu = f_range
    ax.text(
        x, y,
        _t("Plane-wave range {fl} to {fu} Hz", language).format(
            fl=format_number(fl, language, decimals=0, trim=True),
            fu=format_number(fu, language, decimals=0, trim=True),
        ),
        fontsize=8, ha="center", va="top",
    )


def _tube_bore(
    ax: Axes,
    x_left: float,
    x_right: float,
    bore: float,
    language: str,
    *,
    shape: str | None,
    diameter_known: bool,
    **kwargs: Any,
) -> Any:
    """Tube walls + bore + cross-section emblem; returns the primary patch.

    The bore spans ``y in [0, bore]``; walls are drawn just outside it. The
    cross-section emblem (circle or square, to the same scale) sits left of
    the tube with the inner dimension dimensioned when it is known.
    """
    from matplotlib.patches import Circle, Rectangle

    wall = max(0.06 * bore, 0.004)
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 1.6)
    primary = Rectangle(
        (x_left, 0.0), x_right - x_left, bore, **kwargs
    )
    ax.add_patch(primary)
    for y_wall in (-wall, bore):
        ax.add_patch(
            Rectangle((x_left, y_wall), x_right - x_left, wall,
                      facecolor=_C_MUTED, edgecolor=_C_EDGE, linewidth=0.6,
                      alpha=0.5)
        )
    # Cross-section emblem, same scale, centred on the bore axis.
    cx = x_left - 2.4 * bore
    cy = 0.5 * bore
    if shape == "circular":
        ax.add_patch(
            Circle((cx, cy), 0.5 * bore, facecolor="none",
                   edgecolor=_C_EDGE, linewidth=1.2)
        )
    else:
        ax.add_patch(
            Rectangle((cx - 0.5 * bore, cy - 0.5 * bore), bore, bore,
                      facecolor="none", edgecolor=_C_EDGE, linewidth=1.2)
        )
    ax.text(
        cx, cy - 0.72 * bore, _t("Cross-section", language), fontsize=8,
        ha="center", va="top",
    )
    if diameter_known:
        _dim(ax, (cx - 0.5 * bore, cy), (cx + 0.5 * bore, cy),
             _mm(bore, language))
    return primary


# ---------------------------------------------------------------------------
# Layered absorber stack.
# ---------------------------------------------------------------------------
#: Nominal drawn thickness of a zero-thickness membrane, as a fraction of the
#: total stack depth (a membrane has surface density but no depth of its own).
_MEMBRANE_DRAW_FRACTION = 0.012


def _layer_kind_and_thickness(layer: Any, total: float) -> tuple[str, float, str]:
    """Map a layer dataclass to (style kind, drawn thickness, label key).

    Dispatch is by class name so this rendering leaf never imports the domain
    dataclasses at runtime.
    """
    name = type(layer).__name__
    if name == "AirLayer":
        return "air", float(layer.thickness), "Air"
    if name == "PorousLayer":
        return "porous", float(layer.thickness), "Porous"
    if name == "PerforatedPlateLayer":
        return "plate", float(layer.thickness), "Perforated plate"
    if name == "MicroperforatedPlateLayer":
        return "plate", float(layer.thickness), "Microperforated plate"
    if name == "MembraneLayer":
        return "membrane", _MEMBRANE_DRAW_FRACTION * total, "Membrane"
    raise TypeError(f"Unsupported layer type: {name!r}.")


def _stack_total_depth(layers: Sequence[Any]) -> float:
    """Physical depth of the stack (membranes contribute zero)."""
    return float(sum(getattr(layer, "thickness", 0.0) for layer in layers))


def _draw_plate_holes(
    ax: Axes, layer: Any, x: float, thickness: float, height: float
) -> None:
    """Carve the hole pattern of a (micro)perforated plate, to scale.

    Holes of diameter ``2 r`` at the square-lattice pitch that reproduces the
    plate's ``open_area`` (``b = 2 r sqrt(pi / (4 sigma))``).
    """
    radius = float(layer.hole_radius)
    sigma = float(layer.open_area)
    if radius <= 0.0 or sigma <= 0.0:
        return
    pitch = 2.0 * radius * float(np.sqrt(np.pi / (4.0 * sigma)))
    n = int(height // pitch)
    if n < 1:
        return
    y0 = 0.5 * (height - (n - 1) * pitch)
    for i in range(n):
        _material_rect(
            ax, x, y0 + i * pitch - radius, thickness, 2.0 * radius, "cavity",
            edgecolor="none",
        )


def plot_absorber_stack(
    layers: Sequence[Layer] | Layer,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw a layered-absorber cross-section to scale, rigid backing at right.

    Sound arrives from the left; each layer is drawn with its material fill
    and its thickness dimensioned below the stack. A membrane (no physical
    depth) is drawn as a thin sheet.

    :param layers: The layer sequence of
        :func:`~phonometry.materials.layered_absorber`, front layer first, or
        a single layer.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the front-layer rectangle.
    :return: The axes.
    """
    stack = list(layers) if isinstance(layers, (list, tuple)) else [layers]
    if not stack:
        raise ValueError("'layers' must contain at least one layer.")
    if ax is None:
        ax = _new_axes()
    total = _stack_total_depth(stack)
    if total <= 0.0:
        total = 0.05
    height = 0.9 * total
    thin = 0.14 * total
    x = 0.0
    for index, layer in enumerate(stack):
        kind, drawn, label_key = _layer_kind_and_thickness(layer, total)
        extra = dict(kwargs) if index == 0 else {}
        _material_rect(ax, x, 0.0, drawn, height, kind, **extra)
        if kind == "plate":
            _draw_plate_holes(ax, layer, x, drawn, height)
        label = _t(label_key, language)
        physical = float(getattr(layer, "thickness", 0.0))
        if physical > 0.0 and drawn >= thin:
            _dim(
                ax, (x, 0.0), (x + drawn, 0.0), _mm(physical, language),
                offset=-(0.10 * height + 0.14 * height * (index % 2)),
            )
        elif physical > 0.0:
            # Too thin for a dimension line: fold it into the label.
            label = f"{label}, {_mm(physical, language)}"
        centre = x + 0.5 * drawn
        y_label = height * (1.03 + 0.08 * (index % 2))
        ax.text(centre, y_label, label, fontsize=8, ha="center", va="bottom")
        if drawn < thin:
            ax.plot(
                [centre, centre], [height, y_label],
                linestyle=":", linewidth=0.7, color=_C_EDGE, zorder=4,
            )
        x += drawn
    backing = 0.10 * total
    _material_rect(ax, x, -0.05 * height, backing, 1.1 * height, "rigid")
    ax.text(
        x + 0.5 * backing, height * 1.11, _t("Rigid backing", language),
        fontsize=8, ha="center", va="bottom",
    )
    _incidence_arrow(ax, -0.55 * total, 0.5 * height, 0.35 * total, language)
    _finish_geometry_axes(ax, _t("Layered absorber cross-section", language))
    return ax


# ---------------------------------------------------------------------------
# Helmholtz resonator and the slit metamaterial absorber.
# ---------------------------------------------------------------------------
def _draw_resonator(
    ax: Axes,
    x_mouth: float,
    y_mouth: float,
    resonator: Any,
    *,
    wall: float,
) -> tuple[float, float]:
    """Draw one square-section resonator hanging below its neck mouth.

    The neck opens upward at ``(x_mouth, y_mouth)`` (its centre); the cavity
    sits below the neck. Returns the total (width, depth) drawn, wall
    included, for autoscaling by the caller.
    """
    w_n = float(resonator.neck_side)
    l_n = float(resonator.neck_length)
    w_c = float(resonator.cavity_side)
    l_c = float(resonator.cavity_length)
    # Solid body around the air volume.
    body_w = w_c + 2.0 * wall
    body_d = l_n + l_c + wall
    _material_rect(
        ax, x_mouth - 0.5 * body_w, y_mouth - body_d, body_w, body_d, "rigid",
        linewidth=0.7,
    )
    # Air volume: neck then cavity.
    _material_rect(
        ax, x_mouth - 0.5 * w_n, y_mouth - l_n, w_n, l_n, "cavity",
        edgecolor=_C_EDGE,
    )
    _material_rect(
        ax, x_mouth - 0.5 * w_c, y_mouth - l_n - l_c, w_c, l_c, "cavity",
        edgecolor=_C_EDGE,
    )
    return body_w, body_d


def plot_helmholtz_resonator_geometry(
    resonator: HelmholtzResonator,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw a square-section Helmholtz resonator cross-section, to scale.

    Neck opening upward into free air, cavity below, with the four defining
    dimensions (neck side and length, cavity side and length) dimensioned.

    :param resonator: A
        :class:`~phonometry.materials.slow_sound_absorber.HelmholtzResonator`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the cavity rectangle.
    :return: The axes.
    """
    if ax is None:
        ax = _new_axes()
    w_n = float(resonator.neck_side)
    l_n = float(resonator.neck_length)
    w_c = float(resonator.cavity_side)
    l_c = float(resonator.cavity_length)
    wall = 0.06 * max(w_c, l_n + l_c)
    _draw_resonator(ax, 0.0, 0.0, resonator, wall=wall)
    if kwargs:
        _material_rect(
            ax, -0.5 * w_c, -l_n - l_c, w_c, l_c, "cavity", **kwargs
        )
    off = 0.12 * max(w_c, l_n + l_c)
    _dim(ax, (-0.5 * w_n, 0.0), (0.5 * w_n, 0.0), _mm(w_n, language),
         offset=2.0 * off)
    _dim(ax, (0.5 * w_c + wall, 0.0), (0.5 * w_c + wall, -l_n),
         _mm(l_n, language), offset=-2.0 * off)
    _dim(ax, (0.5 * w_c + wall, -l_n), (0.5 * w_c + wall, -l_n - l_c),
         _mm(l_c, language), offset=-off)
    _dim(ax, (-0.5 * w_c, -l_n - l_c - wall), (0.5 * w_c, -l_n - l_c - wall),
         _mm(w_c, language), offset=-off)
    _finish_geometry_axes(
        ax, _t("Helmholtz resonator cross-section", language)
    )
    return ax


def plot_slit_absorber_geometry(
    resonators: Sequence[HelmholtzResonator] | HelmholtzResonator,
    ax: Axes | None = None,
    *,
    slit_height: float,
    lattice_step: float,
    period: float,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw one period of the slit metamaterial absorber, to scale.

    Side cut of the panel: the slit (height ``h``) runs from the mouth at the
    left into the panel; ``N`` Helmholtz resonators load it from below at the
    lattice step ``a`` (total depth ``L = N a``); the panel repeats vertically
    with ``period`` ``d``; rigid back wall at the right.

    :param resonators: The resonator chain of
        :func:`~phonometry.materials.slit_helmholtz_absorber` (one per
        lattice step, or a single resonator reused for all steps).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param slit_height: Slit height ``h``, in metres.
    :param lattice_step: Lattice step ``a``, in metres.
    :param period: Panel period ``d``, in metres.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the slit rectangle.
    :return: The axes.
    """
    if slit_height <= 0.0 or lattice_step <= 0.0 or period <= 0.0:
        raise ValueError(
            "'slit_height', 'lattice_step' and 'period' must be positive."
        )
    chain = (
        list(resonators)
        if isinstance(resonators, (list, tuple))
        else [resonators]
    )
    if not chain:
        raise ValueError("'resonators' must contain at least one resonator.")
    if ax is None:
        ax = _new_axes()
    n = len(chain)
    depth = n * lattice_step
    h = slit_height
    d = period
    wall = 0.05 * h
    # Panel slab (one period tall): slit at the top of the cell.
    y_slit = d - h
    _material_rect(ax, 0.0, 0.0, depth, d, "rigid", linewidth=0.7, alpha=0.35)
    _material_rect(ax, 0.0, y_slit, depth, h, "cavity", **kwargs)
    # The slit can be a fraction of a millimetre: label it from above with a
    # dotted leader instead of squeezing text inside it.
    x_label = 0.3 * depth
    ax.text(
        x_label, d + 0.10 * d, _t("Slit", language), fontsize=8,
        ha="center", va="bottom",
    )
    ax.plot(
        [x_label, x_label], [y_slit + h, d + 0.10 * d],
        linestyle=":", linewidth=0.7, color=_C_EDGE, zorder=4,
    )
    for index, resonator in enumerate(chain):
        x_mouth = (index + 0.5) * lattice_step
        _draw_resonator(ax, x_mouth, y_slit, resonator, wall=wall)
    # Rigid back wall.
    back = 0.08 * depth
    _material_rect(ax, depth, -0.05 * d, back, 1.1 * d, "rigid")
    _incidence_arrow(
        ax, -0.75 * depth, y_slit + 0.5 * h, 0.45 * depth, language
    )
    off = 0.08 * d
    _dim(ax, (0.0, d), (depth, d), _mm(depth, language), offset=2.0 * off)
    _dim(ax, (0.72 * depth, y_slit), (0.72 * depth, y_slit + h),
         _mm(h, language), tight=True)
    _dim(ax, (depth + back, 0.0), (depth + back, d), _mm(d, language),
         offset=-off)
    if n > 1:
        _dim(ax, (0.0, 0.0), (lattice_step, 0.0), _mm(lattice_step, language),
             offset=-off)
    _finish_geometry_axes(
        ax,
        _t("Slit metamaterial absorber cross-section (one period)", language),
    )
    return ax


# ---------------------------------------------------------------------------
# QRD well profile.
# ---------------------------------------------------------------------------
def plot_qrd_geometry(
    depths: ArrayLike,
    well_width: float,
    ax: Axes | None = None,
    *,
    periods: int = 1,
    fin_width: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the well profile of a quadratic-residue diffuser, to scale.

    Wells open upward; the profile repeats ``periods`` times with thin fins
    between wells. Pairs with
    :func:`~phonometry.materials.qrd_well_depths`, which supplies the depth
    sequence.

    :param depths: Well depths ``d_n``, in metres (one period).
    :param well_width: Well width ``w``, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param periods: Number of repeated periods (>= 1).
    :param fin_width: Fin thickness between wells, in metres; ``None`` draws
        ``w / 12``.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the base-slab rectangle.
    :return: The axes.
    """
    d = np.asarray(depths, dtype=np.float64)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("'depths' must be a non-empty 1-D sequence.")
    if np.any(d < 0.0):
        raise ValueError("'depths' must be non-negative.")
    if well_width <= 0.0:
        raise ValueError("'well_width' must be positive.")
    if periods < 1:
        raise ValueError("'periods' must be >= 1.")
    fin = well_width / 12.0 if fin_width is None else float(fin_width)
    if fin < 0.0:
        raise ValueError("'fin_width' must be non-negative.")
    if ax is None:
        ax = _new_axes()
    n = int(d.size)
    d_max = float(d.max()) if float(d.max()) > 0.0 else well_width
    base = 0.15 * d_max
    pitch = well_width + fin
    total_width = periods * n * pitch + fin
    # Base slab behind the deepest well.
    _material_rect(
        ax, 0.0, -d_max - base, total_width, base, "rigid", **kwargs
    )
    # Fins and well bottoms: the solid between the carved wells.
    x = 0.0
    for _period in range(periods):
        for depth in d:
            _material_rect(ax, x, -d_max, fin, d_max, "plate",
                           linewidth=0.6)
            x += fin
            if d_max - depth > 0.0:
                _material_rect(
                    ax, x, -d_max, well_width, d_max - float(depth), "plate",
                    linewidth=0.6,
                )
            x += well_width
    _material_rect(ax, x, -d_max, fin, d_max, "plate", linewidth=0.6)
    _incidence_arrow(
        ax, 0.5 * total_width, 1.1 * d_max, 0.55 * d_max, language,
        downward=True,
    )
    off = 0.12 * d_max
    _dim(ax, (fin, 0.0), (fin + well_width, 0.0), _mm(well_width, language),
         offset=1.5 * off)
    _dim(ax, (0.0, 0.0), (0.0, -d_max), _mm(d_max, language),
         offset=-3.0 * off)
    from .._i18n import format_number

    seq = ", ".join(
        format_number(float(v * 1e3), language, decimals=1, trim=True)
        for v in d
    )
    ax.text(
        0.5 * total_width, -d_max - 1.6 * base,
        _t("depth sequence {seq}", language).format(seq=seq) + " "
        + _t("mm", language),
        fontsize=8, ha="center", va="top", color=_C_EDGE,
    )
    _finish_geometry_axes(ax, _t("QRD well profile", language))
    return ax


# ---------------------------------------------------------------------------
# Impedance tube (ISO 10534-2) and transmission tube (ASTM E2611).
# ---------------------------------------------------------------------------
#: ISO 10534-2 4.3: non-plane source modes die out within about three tube
#: diameters, so the drawn source section keeps that margin before the first
#: microphone.
_SOURCE_MARGIN_DIAMETERS = 3.0

#: Drawn sample thickness when the caller does not supply one, in metres.
_NOMINAL_SAMPLE_THICKNESS = 0.05


def _nominal_bore(diameter: float | None, fallback: float) -> float:
    """The drawn bore: the real diameter, or a nominal stand-in."""
    return float(diameter) if diameter is not None else fallback


def plot_impedance_tube_geometry(
    ax: Axes | None = None,
    *,
    spacing: float,
    x1: float,
    diameter: float | None = None,
    shape: str | None = "circular",
    sample_thickness: float | None = None,
    speed_of_sound: float = 343.2,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the ISO 10534-2 two-microphone tube to scale.

    Side view: loudspeaker at the left (three tube diameters before the
    farther microphone, the Clause 4.3 margin), the two flush microphones at
    ``x1`` and ``x1 - s`` from the sample face, the sample against its rigid
    backing at the right, the cross-section emblem, and the plane-wave
    working range of :func:`~phonometry.materials.plane_wave_frequency_range`.

    :param ax: Existing axes, or ``None`` to create a figure.
    :param spacing: Microphone spacing ``s``, in metres.
    :param x1: Distance from the sample face to the farther microphone, in
        metres.
    :param diameter: Inner diameter (circular) or lateral dimension
        (rectangular/square), in metres; ``None`` draws a nominal bore and
        omits the bore dimension and the cut-on bound.
    :param shape: ``"circular"``, ``"rectangular"``, ``"square"`` or ``None``.
    :param sample_thickness: Drawn sample thickness, in metres; ``None``
        draws a 50 mm nominal sample.
    :param speed_of_sound: Speed of sound for the working range, in m/s.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the tube-bore rectangle.
    :return: The axes.
    """
    from ..materials.impedance_tube import plane_wave_frequency_range

    if spacing <= 0.0 or x1 <= spacing:
        raise ValueError("'spacing' must be positive and 'x1' > 'spacing'.")
    if ax is None:
        ax = _new_axes()
    bore = _nominal_bore(diameter, 1.5 * spacing)
    thickness = (
        _NOMINAL_SAMPLE_THICKNESS
        if sample_thickness is None else float(sample_thickness)
    )
    lead_in = _SOURCE_MARGIN_DIAMETERS * bore
    x_left = -(x1 + lead_in)
    x_right = thickness + 0.35 * bore
    _tube_bore(
        ax, x_left, x_right, bore, language,
        shape="circular" if shape is None else
        ("circular" if shape == "circular" else "rectangular"),
        diameter_known=diameter is not None, **kwargs,
    )
    # Sample against the rigid backing plug: front face at x = 0.
    _material_rect(ax, 0.0, 0.0, thickness, bore, "porous")
    _material_rect(ax, thickness, 0.0, x_right - thickness, bore, "rigid")
    ax.text(0.5 * thickness, -0.16 * bore, _t("Sample", language),
            fontsize=8, ha="center", va="top")
    _loudspeaker(ax, x_left + 0.1 * bore, 0.5 * bore, bore, language)
    wall = max(0.06 * bore, 0.004)
    _microphone(ax, -x1, bore + wall, 0.5 * bore, "1")
    _microphone(ax, -(x1 - spacing), bore + wall, 0.5 * bore, "2")
    off = 0.3 * bore
    y_dim = -0.5 * bore
    _dim(ax, (-x1, y_dim), (-(x1 - spacing), y_dim),
         "s = " + _mm(spacing, language),
         tight=spacing < 0.12 * (x_right - x_left))
    _dim(ax, (-x1, y_dim - 2.0 * off), (0.0, y_dim - 2.0 * off),
         "x1 = " + _mm(x1, language))
    f_range = plane_wave_frequency_range(
        spacing, speed_of_sound, diameter=diameter,
        shape=shape if shape is not None else "circular",
    )
    _tube_frequency_note(
        ax, 0.5 * (x_left + x_right), y_dim - 3.6 * off, f_range, language
    )
    _finish_geometry_axes(
        ax, _t("Impedance tube (ISO 10534-2), to scale", language)
    )
    return ax


def plot_transmission_tube_geometry(
    ax: Axes | None = None,
    *,
    l1: float,
    s1: float,
    l2: float,
    s2: float,
    thickness: float,
    diameter: float | None = None,
    shape: str | None = "circular",
    speed_of_sound: float = 343.2,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the ASTM E2611 four-microphone transmission tube to scale.

    Side view: loudspeaker, the two upstream microphones at ``l1`` and
    ``l1 + s1`` from the front face of the specimen, the specimen spanning
    its thickness, the two downstream microphones at ``l2`` and ``l2 + s2``
    (measured from the front face, the module's locked convention), and the
    changeable termination of the two-load method, with the ASTM working
    range of
    :func:`~phonometry.materials.plane_wave_frequency_range_astm`.

    :param ax: Existing axes, or ``None`` to create a figure.
    :param l1: Front face to the nearer upstream microphone, in metres.
    :param s1: Upstream microphone spacing, in metres.
    :param l2: Front face to the nearer downstream microphone, in metres.
    :param s2: Downstream microphone spacing, in metres.
    :param thickness: Specimen thickness, in metres; must be smaller than
        ``l2`` (the downstream microphones sit past the back face).
    :param diameter: Inner diameter (circular) or largest section dimension
        (rectangular/square), in metres; ``None`` draws a nominal bore and
        omits the bore dimension and the cut-on bound.
    :param shape: ``"circular"``, ``"rectangular"``, ``"square"`` or ``None``.
    :param speed_of_sound: Speed of sound for the working range, in m/s.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the tube-bore rectangle.
    :return: The axes.
    """
    from ..materials.impedance_tube import plane_wave_frequency_range_astm

    if min(l1, s1, l2, s2, thickness) <= 0.0:
        raise ValueError(
            "'l1', 's1', 'l2', 's2' and 'thickness' must be positive."
        )
    if l2 <= thickness:
        raise ValueError(
            "'l2' is measured from the front face and must exceed "
            "'thickness'."
        )
    if ax is None:
        ax = _new_axes()
    bore = _nominal_bore(diameter, 1.5 * max(s1, s2))
    lead_in = _SOURCE_MARGIN_DIAMETERS * bore
    x_left = -(l1 + s1 + lead_in)
    x_term = l2 + s2 + 0.8 * bore
    x_right = x_term + 0.6 * bore
    _tube_bore(
        ax, x_left, x_right, bore, language,
        shape="circular" if shape is None else
        ("circular" if shape == "circular" else "rectangular"),
        diameter_known=diameter is not None, **kwargs,
    )
    _material_rect(ax, 0.0, 0.0, thickness, bore, "porous")
    ax.text(0.5 * thickness, bore + 0.22 * bore, _t("Sample", language),
            fontsize=8, ha="center", va="bottom")
    _loudspeaker(ax, x_left + 0.1 * bore, 0.5 * bore, bore, language)
    # Changeable termination of the two-load method.
    _material_rect(ax, x_term, -0.1 * bore, x_right - x_term, 1.2 * bore,
                   "plate")
    ax.text(0.5 * (x_term + x_right), -0.28 * bore,
            _t("Termination", language), fontsize=8, ha="center", va="top")
    wall = max(0.06 * bore, 0.004)
    positions = (
        (-(l1 + s1), "1"), (-l1, "2"), (l2, "3"), (l2 + s2, "4"),
    )
    for x_mic, label in positions:
        _microphone(ax, x_mic, bore + wall, 0.5 * bore, label)
    y_dim = -0.62 * bore
    off = 0.3 * bore
    width = x_right - x_left
    _dim(ax, (-(l1 + s1), y_dim), (-l1, y_dim),
         "s1 = " + _mm(s1, language), tight=s1 < 0.12 * width)
    _dim(ax, (l2, y_dim), (l2 + s2, y_dim), "s2 = " + _mm(s2, language),
         tight=s2 < 0.12 * width)
    _dim(ax, (-l1, y_dim - 2.0 * off), (0.0, y_dim - 2.0 * off),
         "l1 = " + _mm(l1, language))
    _dim(ax, (0.0, y_dim - 4.0 * off), (l2, y_dim - 4.0 * off),
         "l2 = " + _mm(l2, language))
    f_range = plane_wave_frequency_range_astm(
        max(s1, s2), speed_of_sound, diameter=diameter,
        shape=shape if shape is not None else "circular",
    )
    _tube_frequency_note(
        ax, 0.5 * (x_left + x_right), y_dim - 5.6 * off, f_range, language
    )
    _finish_geometry_axes(
        ax, _t("Transmission tube (ASTM E2611), to scale", language)
    )
    return ax


# ---------------------------------------------------------------------------
# Renderers bound to result objects (geometry retained on the result).
# ---------------------------------------------------------------------------
def plot_layered_absorber_geometry(
    result: LayeredAbsorberResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Stack drawing for a result that retained its ``layers``."""
    if result.layers is None:
        raise ValueError(
            "This result does not retain its layers; call "
            "plot_absorber_stack(layers) with the original layer sequence."
        )
    return plot_absorber_stack(
        result.layers, ax=ax, language=language, **kwargs
    )


def plot_slit_absorber_result_geometry(
    result: SlitResonatorAbsorberResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Slit-panel drawing for a result that retained its geometry."""
    if (
        result.resonators is None
        or result.slit_height is None
        or result.lattice_step is None
        or result.period is None
    ):
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_slit_absorber_geometry(...) with the original arguments."
        )
    return plot_slit_absorber_geometry(
        result.resonators, ax=ax, slit_height=result.slit_height,
        lattice_step=result.lattice_step, period=result.period,
        language=language, **kwargs,
    )


def plot_diffuser_geometry(
    result: DiffuserPolarResponse,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """QRD well profile for a polar response that retained its geometry."""
    if result.depths is None or result.well_width is None:
        raise ValueError(
            "This response does not retain its well geometry; call "
            "plot_qrd_geometry(depths, well_width) instead."
        )
    return plot_qrd_geometry(
        result.depths, result.well_width, ax=ax,
        periods=result.periods if result.periods is not None else 1,
        language=language, **kwargs,
    )


def plot_impedance_tube_result_geometry(
    result: ImpedanceTubeResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Tube drawing for an ISO 10534-2 result that retained its geometry."""
    if result.spacing is None or result.x1 is None:
        raise ValueError(
            "This result does not retain its tube geometry; call "
            "plot_impedance_tube_geometry(...) with the original arguments."
        )
    return plot_impedance_tube_geometry(
        ax=ax, spacing=result.spacing, x1=result.x1,
        diameter=result.diameter, shape=result.shape, language=language,
        **kwargs,
    )


def plot_transfer_matrix_geometry(
    result: TransferMatrix,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Tube drawing for an ASTM E2611 matrix that retained its context."""
    if (
        result.l1 is None or result.s1 is None or result.l2 is None
        or result.s2 is None or result.thickness is None
    ):
        raise ValueError(
            "This matrix does not retain its tube geometry; call "
            "plot_transmission_tube_geometry(...) with the original "
            "arguments."
        )
    return plot_transmission_tube_geometry(
        ax=ax, l1=result.l1, s1=result.s1, l2=result.l2, s2=result.s2,
        thickness=result.thickness, diameter=result.diameter,
        shape=result.shape, language=language, **kwargs,
    )
