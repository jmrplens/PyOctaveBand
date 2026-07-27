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
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    _C_TERTIARY,
    _import_pyplot,
    _new_axes,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..building.aperture_transmission import ApertureTransmissionResult
    from ..electroacoustics.piston import RadiatingPistonResult
    from ..environmental.ground_barriers import BarrierInsertionLoss
    from ..materials.diffuser_design import DiffuserPolarResponse
    from ..materials.impedance_tube import ImpedanceTubeResult, TransferMatrix
    from ..materials.porous_absorber import Layer, LayeredAbsorberResult
    from ..materials.slow_sound_absorber import (
        HelmholtzResonator,
        SlitResonatorAbsorberResult,
    )
    from ..noise_control.silencers import ReactiveSilencerResult
    from ..room.image_source import ImageSourceResult
    from ..simulation.fdtd import FDTD2D

#: The ``ReactiveSilencerResult.kind`` strings, shared with the dispatcher.
_KIND_EXPANSION = "expansion chamber"
_KIND_EXTENDED = "extended-tube chamber"
_KIND_HELMHOLTZ = "Helmholtz resonator"
_KIND_QUARTER = "quarter-wave resonator"

#: Axis labels shared by the plan-view and 3-D renderers.
_AXIS_X = "x [m]"
_AXIS_Y = "y [m]"
_AXIS_Z = "z [m]"
#: Shared legend placement of the geometry drawings.
_LEGEND_LOC: Final = "upper right"
#: Shared validation message for length arguments.
_LENGTH_POSITIVE = "'length' must be positive."


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
    "m": "m",
    "Reactive silencer cross-section": "Sección del silenciador reactivo",
    _KIND_EXPANSION: "cámara de expansión",
    _KIND_HELMHOLTZ: "resonador de Helmholtz",
    _KIND_QUARTER: "resonador de cuarto de onda",
    _KIND_EXTENDED: "cámara con tubos extendidos",
    "Image-source room plan (z of the source plane)":
        "Planta de fuentes imagen (plano z de la fuente)",
    "Source": "Fuente",
    "Receiver": "Receptor",
    "order {n}": "orden {n}",
    _AXIS_X: _AXIS_X,
    _AXIS_Y: _AXIS_Y,
    _AXIS_Z: _AXIS_Z,
    "Barrier section": "Sección de la barrera",
    "Ground": "Suelo",
    "Direct path": "Camino directo",
    "Diffracted path": "Camino difractado",
    "Path difference {delta} m": "Diferencia de camino {delta} m",
    "Microphone positions": "Posiciones de micrófono",
    "Reflecting plane": "Plano reflectante",
    "Wall aperture cross-section": "Sección de la abertura en el muro",
    "Wall": "Muro",
    "Baffled piston": "Pistón en pantalla infinita",
    "Baffle": "Pantalla",
    "Normalised directivity": "Directividad normalizada",
    "Plenum chamber section": "Sección de la cámara plenum",
    "Inlet": "Entrada",
    "Outlet": "Salida",
    "FDTD domain": "Dominio FDTD",
    "Sponge layer": "Capa esponja",
    "Impedance edge": "Borde de impedancia",
    "Rigid edge": "Borde rígido",
    "Probe": "Sonda",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


def _check_language(language: str) -> None:
    """Reject unknown languages with the shared package error."""
    from .._i18n import check_language

    check_language(language)


def _mm(value: float, language: str) -> str:
    """A length in millimetres, localised, trimmed (0.05 -> ``"50 mm"``)."""
    from .._i18n import format_number

    return (
        format_number(value * 1e3, language, decimals=1, trim=True)
        + " " + _t("mm", language)
    )


def _metres(value: float, language: str) -> str:
    """A length in metres, localised, trimmed (1.5 -> ``"1.5 m"``)."""
    from .._i18n import format_number

    return (
        format_number(value, language, decimals=2, trim=True)
        + " " + _t("m", language)
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
    if not label:
        return
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    direction = b - a
    length = float(np.hypot(*direction))
    if length < 1e-12:
        return
    normal = np.array([-direction[1], direction[0]]) / length
    ao = a + normal * offset
    bo = b + normal * offset
    if abs(offset) > 1e-12:
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
    label = _t("Incident sound", language)
    if downward:
        ax.text(
            x, y + 0.08 * length, label,
            fontsize=8, ha="center", va="bottom", color=_C_PRIMARY,
        )
    else:
        ax.text(
            x + 0.5 * length, y + 0.06 * length, label,
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
    _check_language(language)
    stack = list(layers) if isinstance(layers, Sequence) else [layers]
    if not stack:
        raise ValueError("'layers' must contain at least one layer.")
    total = _stack_total_depth(stack)
    if total <= 0.0:
        total = 0.05
    # Dispatching every layer up front also validates the types before any
    # figure is created.
    kinds = [_layer_kind_and_thickness(layer, total) for layer in stack]
    if ax is None:
        ax = _new_axes()
    height = 0.9 * total
    thin = 0.14 * total
    x = 0.0
    for index, (layer, dispatched) in enumerate(zip(stack, kinds)):
        kind, drawn, label_key = dispatched
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
    _check_language(language)
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
    _check_language(language)
    if slit_height <= 0.0 or lattice_step <= 0.0 or period <= 0.0:
        raise ValueError(
            "'slit_height', 'lattice_step' and 'period' must be positive."
        )
    chain = (
        list(resonators)
        if isinstance(resonators, Sequence)
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
    _check_language(language)
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

    _check_language(language)
    if spacing <= 0.0 or x1 <= spacing:
        raise ValueError("'spacing' must be positive and 'x1' > 'spacing'.")
    if diameter is not None and diameter <= 0.0:
        raise ValueError("'diameter' must be positive when given.")
    if sample_thickness is not None and sample_thickness <= 0.0:
        raise ValueError("'sample_thickness' must be positive when given.")
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
    emblem = "circular" if shape in (None, "circular") else "rectangular"
    _tube_bore(
        ax, x_left, x_right, bore, language, shape=emblem,
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

    _check_language(language)
    if min(l1, s1, l2, s2, thickness) <= 0.0:
        raise ValueError(
            "'l1', 's1', 'l2', 's2' and 'thickness' must be positive."
        )
    if diameter is not None and diameter <= 0.0:
        raise ValueError("'diameter' must be positive when given.")
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
    emblem = "circular" if shape in (None, "circular") else "rectangular"
    _tube_bore(
        ax, x_left, x_right, bore, language, shape=emblem,
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


# ---------------------------------------------------------------------------
# Reactive silencers (four-pole elements).
# ---------------------------------------------------------------------------
#: Recognised silencer kinds (the ``ReactiveSilencerResult.kind`` strings).
_SILENCER_KINDS = (
    _KIND_EXPANSION,
    _KIND_EXTENDED,
    _KIND_HELMHOLTZ,
    _KIND_QUARTER,
)


def _duct_diameter(area: float) -> float:
    """Equivalent circular diameter of a duct cross-section."""
    if area <= 0.0:
        raise ValueError("Cross-section areas must be positive.")
    return float(2.0 * np.sqrt(area / np.pi))


def _draw_duct(ax: Axes, x0: float, x1: float, d: float, **kwargs: Any) -> Any:
    """A straight duct run: bore centred on y = 0, walls just outside."""
    from matplotlib.patches import Rectangle

    wall = max(0.05 * d, 0.002)
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 1.2)
    bore = Rectangle((x0, -0.5 * d), x1 - x0, d, **kwargs)
    ax.add_patch(bore)
    for y in (-0.5 * d - wall, 0.5 * d):
        _material_rect(ax, x0, y, x1 - x0, wall, "plate", linewidth=0.5)
    return bore


def _draw_chamber(
    ax: Axes,
    length: float,
    chamber_area: float,
    pipe_area: float,
    language: str,
    *,
    inlet_extension: float = 0.0,
    outlet_extension: float = 0.0,
    **kwargs: Any,
) -> None:
    """Expansion chamber (optionally with extended inlet/outlet tubes)."""
    d_p = _duct_diameter(pipe_area)
    d_c = _duct_diameter(chamber_area)
    if chamber_area <= pipe_area:
        raise ValueError("'chamber_area' must exceed 'pipe_area'.")
    if inlet_extension + outlet_extension > length:
        raise ValueError(
            "'inlet_extension' + 'outlet_extension' must not exceed 'length'."
        )
    stub = max(0.5 * length, 1.5 * d_c)
    # Chamber shell.
    from matplotlib.patches import Rectangle

    wall = max(0.03 * d_c, 0.003)
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 1.4)
    ax.add_patch(Rectangle((0.0, -0.5 * d_c), length, d_c, **kwargs))
    for y in (-0.5 * d_c - wall, 0.5 * d_c):
        _material_rect(ax, 0.0, y, length, wall, "plate", linewidth=0.5)
    # End plates (leave the pipe openings free).
    for x in (0.0, length):
        for y0, y1 in ((0.5 * d_p, 0.5 * d_c), (-0.5 * d_c, -0.5 * d_p)):
            ax.plot([x, x], [y0, y1], color=_C_EDGE, linewidth=1.4)
    _draw_duct(ax, -stub, 0.0, d_p)
    _draw_duct(ax, length, length + stub, d_p)
    # Extended tubes protrude into the chamber as thin-walled pipes.
    for ext, x0, x1 in (
        (inlet_extension, 0.0, inlet_extension),
        (outlet_extension, length - outlet_extension, length),
    ):
        if ext > 0.0:
            for y in (-0.5 * d_p, 0.5 * d_p):
                ax.plot([x0, x1], [y, y], color=_C_EDGE, linewidth=1.2)
    off = 0.18 * d_c
    _dim(ax, (0.0, -0.5 * d_c - 2.0 * off), (length, -0.5 * d_c - 2.0 * off),
         "L = " + _mm(length, language))
    _dim(ax, (-stub, -0.5 * d_p), (-stub, 0.5 * d_p),
         _mm(d_p, language), offset=2.0 * off, tight=True)
    x_dim = inlet_extension + 0.5 * (
        length - inlet_extension - outlet_extension
    )
    _dim(ax, (x_dim, -0.5 * d_c), (x_dim, 0.5 * d_c),
         _mm(d_c, language), offset=0.0)
    if inlet_extension > 0.0:
        _dim(ax, (0.0, 0.5 * d_p), (inlet_extension, 0.5 * d_p),
             _mm(inlet_extension, language), offset=-2.0 * off, tight=True)
    if outlet_extension > 0.0:
        _dim(ax, (length - outlet_extension, 0.5 * d_p), (length, 0.5 * d_p),
             _mm(outlet_extension, language), offset=-2.0 * off, tight=True)


def _draw_hr_cavity(
    ax: Axes,
    y0: float,
    branch_len: float,
    cavity_side: float,
    cavity_volume: float,
    language: str,
) -> None:
    """Cavity drawn as the equivalent cube (V^(1/3) on each side)."""
    from .._i18n import format_number

    _material_rect(
        ax, -0.5 * cavity_side, y0 + branch_len, cavity_side,
        cavity_side, "cavity",
    )
    ax.text(
        0.0, y0 + branch_len + 0.5 * cavity_side,
        "V = {volume} L".format(
            volume=format_number(
                cavity_volume * 1e3, language, decimals=1, trim=True
            )
        ),
        fontsize=8, ha="center", va="center",
    )


def _branch_dimensions(
    kind: str,
    *,
    neck_area: float | None,
    neck_length: float | None,
    cavity_volume: float | None,
    length: float | None,
    branch_area: float | None,
) -> tuple[float, float, float]:
    """Resolve (branch diameter, branch length, cavity side) per kind."""
    if kind == _KIND_HELMHOLTZ:
        if neck_area is None or neck_length is None or cavity_volume is None:
            raise ValueError(
                "A Helmholtz resonator drawing needs 'neck_area', "
                "'neck_length' and 'cavity_volume'."
            )
        if cavity_volume <= 0.0 or neck_length <= 0.0:
            raise ValueError(
                "'cavity_volume' and 'neck_length' must be positive."
            )
        return (
            _duct_diameter(neck_area), neck_length,
            float(cavity_volume ** (1.0 / 3.0)),
        )
    if length is None or branch_area is None:
        raise ValueError(
            "A quarter-wave drawing needs 'length' and 'branch_area'."
        )
    if length <= 0.0:
        raise ValueError(_LENGTH_POSITIVE)
    return _duct_diameter(branch_area), length, 0.0


def _draw_branch_silencer(
    ax: Axes,
    kind: str,
    duct_area: float,
    language: str,
    *,
    neck_area: float | None = None,
    neck_length: float | None = None,
    cavity_volume: float | None = None,
    length: float | None = None,
    branch_area: float | None = None,
    **kwargs: Any,
) -> None:
    """Side-branch silencer: Helmholtz resonator or quarter-wave tube.

    Parameters are already validated by :func:`_validate_branch_geometry`.
    """
    d_d = _duct_diameter(duct_area)
    d_b, branch_len, cavity_side = _branch_dimensions(
        kind, neck_area=neck_area, neck_length=neck_length,
        cavity_volume=cavity_volume, length=length, branch_area=branch_area,
    )
    run = max(4.0 * d_d, 2.0 * d_b + 2.0 * d_d, 2.0 * cavity_side)
    _draw_duct(ax, -0.5 * run, 0.5 * run, d_d, **kwargs)
    # Branch mouth opens through the upper duct wall at x = 0.
    from matplotlib.patches import Rectangle

    y0 = 0.5 * d_d
    ax.add_patch(Rectangle((-0.5 * d_b, y0), d_b, branch_len,
                           facecolor="none", edgecolor=_C_EDGE,
                           linewidth=1.2))
    off = 0.25 * d_d
    if kind == _KIND_HELMHOLTZ and cavity_volume is not None:
        _draw_hr_cavity(
            ax, y0, branch_len, cavity_side, cavity_volume, language
        )
        _dim(ax, (0.5 * d_b, y0), (0.5 * d_b, y0 + branch_len),
             _mm(branch_len, language), offset=-2.0 * off, tight=True)
    else:
        # Closed end of the quarter-wave tube.
        ax.plot([-0.5 * d_b, 0.5 * d_b],
                [y0 + branch_len, y0 + branch_len],
                color=_C_EDGE, linewidth=2.2)
        _dim(ax, (0.5 * d_b, y0), (0.5 * d_b, y0 + branch_len),
             _mm(branch_len, language), offset=-2.0 * off)
    _dim(ax, (-0.5 * d_b, y0), (0.5 * d_b, y0), _mm(d_b, language),
         offset=-1.5 * off, tight=True)
    _dim(ax, (-0.5 * run, -0.5 * d_d), (-0.5 * run, 0.5 * d_d),
         _mm(d_d, language), offset=2.0 * off, tight=True)


def _validate_chamber_geometry(
    length: float | None,
    chamber_area: float | None,
    pipe_area: float | None,
    inlet_extension: float,
    outlet_extension: float,
) -> None:
    """Chamber parameter validation, before any figure exists."""
    if length is None or chamber_area is None or pipe_area is None:
        raise ValueError(
            "A chamber drawing needs 'length', 'chamber_area' and "
            "'pipe_area'."
        )
    if length <= 0.0:
        raise ValueError(_LENGTH_POSITIVE)
    _duct_diameter(pipe_area)
    _duct_diameter(chamber_area)
    if chamber_area <= pipe_area:
        raise ValueError("'chamber_area' must exceed 'pipe_area'.")
    if inlet_extension + outlet_extension > length:
        raise ValueError(
            "'inlet_extension' + 'outlet_extension' must not exceed "
            "'length'."
        )


def _validate_branch_geometry(
    kind: str,
    duct_area: float | None,
    neck_area: float | None,
    neck_length: float | None,
    cavity_volume: float | None,
    length: float | None,
    branch_area: float | None,
) -> None:
    """Side-branch parameter validation, before any figure exists."""
    if duct_area is None:
        raise ValueError("A side-branch drawing needs 'duct_area'.")
    _duct_diameter(duct_area)
    if kind == _KIND_HELMHOLTZ:
        if neck_area is None or neck_length is None or cavity_volume is None:
            raise ValueError(
                "A Helmholtz resonator drawing needs 'neck_area', "
                "'neck_length' and 'cavity_volume'."
            )
        if neck_length <= 0.0 or cavity_volume <= 0.0:
            raise ValueError(
                "'cavity_volume' and 'neck_length' must be positive."
            )
        _duct_diameter(neck_area)
        return
    if length is None or branch_area is None:
        raise ValueError(
            "A quarter-wave drawing needs 'length' and 'branch_area'."
        )
    if length <= 0.0:
        raise ValueError(_LENGTH_POSITIVE)
    _duct_diameter(branch_area)


def plot_silencer_geometry(
    kind: str,
    ax: Axes | None = None,
    *,
    length: float | None = None,
    chamber_area: float | None = None,
    pipe_area: float | None = None,
    inlet_extension: float = 0.0,
    outlet_extension: float = 0.0,
    duct_area: float | None = None,
    neck_area: float | None = None,
    neck_length: float | None = None,
    cavity_volume: float | None = None,
    branch_area: float | None = None,
    language: str = "en",
) -> Axes:
    """Draw a reactive silencer cross-section to scale.

    Side cut through the duct axis with equivalent circular diameters
    (``d = 2 sqrt(S / pi)``) for every cross-section area, matching the
    parameters of the four :mod:`~phonometry.noise_control` silencer
    constructors. A Helmholtz cavity is drawn as the cube of equal volume
    with its volume annotated.

    :param kind: One of ``"expansion chamber"``, ``"extended-tube chamber"``,
        ``"Helmholtz resonator"``, ``"quarter-wave resonator"`` (the
        ``ReactiveSilencerResult.kind`` strings).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param length: Chamber length or quarter-wave tube length, in metres.
    :param chamber_area: Chamber cross-section, in m2 (chambers).
    :param pipe_area: Inlet/outlet pipe cross-section, in m2 (chambers).
    :param inlet_extension: Inlet tube extension into the chamber, in metres.
    :param outlet_extension: Outlet tube extension, in metres.
    :param duct_area: Main duct cross-section, in m2 (side branches).
    :param neck_area: Neck cross-section, in m2 (Helmholtz).
    :param neck_length: Neck length, in metres (Helmholtz).
    :param cavity_volume: Cavity volume, in m3 (Helmholtz).
    :param branch_area: Branch tube cross-section, in m2 (quarter-wave).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :return: The axes.
    """
    _check_language(language)
    if kind not in _SILENCER_KINDS:
        raise ValueError(
            f"Unknown silencer kind {kind!r}; expected one of "
            f"{_SILENCER_KINDS}."
        )
    chamber = kind in (_KIND_EXPANSION, _KIND_EXTENDED)
    if chamber:
        _validate_chamber_geometry(
            length, chamber_area, pipe_area, inlet_extension,
            outlet_extension,
        )
    else:
        _validate_branch_geometry(
            kind, duct_area, neck_area, neck_length, cavity_volume,
            length, branch_area,
        )
    if ax is None:
        ax = _new_axes()
    if chamber:
        _draw_chamber(
            ax, length or 0.0, chamber_area or 0.0, pipe_area or 0.0,
            language,
            inlet_extension=inlet_extension,
            outlet_extension=outlet_extension,
        )
    else:
        _draw_branch_silencer(
            ax, kind, duct_area or 0.0, language,
            neck_area=neck_area, neck_length=neck_length,
            cavity_volume=cavity_volume, length=length,
            branch_area=branch_area,
        )
    _finish_geometry_axes(
        ax,
        _t("Reactive silencer cross-section", language)
        + f" ({_t(kind, language)})",
    )
    return ax


def plot_silencer_result_geometry(
    result: ReactiveSilencerResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
) -> Axes:
    """Silencer drawing for a result that retained its ``geometry``."""
    if result.geometry is None:
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_silencer_geometry(kind, ...) with the original arguments."
        )
    return plot_silencer_geometry(
        result.kind, ax=ax, language=language, **dict(result.geometry),
    )


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
        ``min(result.max_order, 3)`` to keep the plan readable.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the room-outline rectangle.
    :return: The axes.
    """
    from matplotlib.patches import Rectangle

    _check_language(language)
    from .._i18n import localize_axes

    order_cap = (
        min(int(result.max_order), 3) if max_order is None else int(max_order)
    )
    if order_cap < 1:
        raise ValueError("'max_order' must be >= 1.")
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
        ax.plot([x, x], [y_min, y_max], linestyle=":", linewidth=0.6,
                color=_C_MUTED, zorder=1)
    for y in np.arange(np.floor(y_min / ly) * ly, y_max + ly, ly):
        ax.plot([x_min, x_max], [y, y], linestyle=":", linewidth=0.6,
                color=_C_MUTED, zorder=1)
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
            pos[sel, 0], pos[sel, 1], "o", markersize=4, color=colour,
            linestyle="none", zorder=4,
            label=_t("order {n}", language).format(n=order),
        )
    ax.plot(
        [result.source[0]], [result.source[1]], marker="*", markersize=13,
        color=_C_REFERENCE, linestyle="none", zorder=5,
        label=_t("Source", language),
    )
    ax.plot(
        [result.receiver[0]], [result.receiver[1]], marker="^",
        markersize=8, color=_C_PRIMARY, linestyle="none", zorder=5,
        label=_t("Receiver", language),
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(_t(_AXIS_X, language))
    ax.set_ylabel(_t(_AXIS_Y, language))
    ax.set_title(
        _t("Image-source room plan (z of the source plane)", language),
        fontweight="bold",
    )
    ax.legend(loc=_LEGEND_LOC, fontsize=8)
    localize_axes(ax, language)
    return ax


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
    :func:`~phonometry.environmental.barrier_insertion_loss`:
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
        raise ValueError("Heights must be non-negative.")
    if barrier_distance <= 0.0 or receiver_distance <= barrier_distance:
        raise ValueError(
            "'barrier_distance' must be positive and 'receiver_distance' "
            "greater than it."
        )
    if thickness is not None and thickness <= 0.0:
        raise ValueError("'thickness' must be positive when given.")
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
        ax, -0.08 * receiver_distance, -0.04 * receiver_distance,
        1.2 * receiver_distance, 0.04 * receiver_distance, "rigid",
    )
    ax.text(
        1.06 * receiver_distance, -0.02 * receiver_distance,
        _t("Ground", language), fontsize=8, ha="left", va="center",
    )
    _material_rect(
        ax, barrier_distance, 0.0, drawn_e, top, "plate", **kwargs
    )
    # Paths.
    ax.plot(
        [src[0], rcv[0]], [src[1], rcv[1]], linestyle="--", linewidth=1.1,
        color=_C_MUTED, label=_t("Direct path", language), zorder=4,
    )
    diff_x = [src[0], near[0]]
    diff_y = [src[1], near[1]]
    if e > 0.0:
        diff_x.append(far[0])
        diff_y.append(far[1])
    diff_x.append(rcv[0])
    diff_y.append(rcv[1])
    ax.plot(
        diff_x, diff_y, linewidth=1.6, color=_C_PRIMARY,
        label=_t("Diffracted path", language), zorder=5,
    )
    ax.plot([src[0]], [src[1]], marker="*", markersize=13,
            color=_C_REFERENCE, linestyle="none", zorder=6,
            label=_t("Source", language))
    ax.plot([rcv[0]], [rcv[1]], marker="^", markersize=8, color=_C_PRIMARY,
            linestyle="none", zorder=6, label=_t("Receiver", language))
    # Path difference over the top (delta = A + B (+ e) - d).
    a_len = float(np.hypot(near[0] - src[0], near[1] - src[1]))
    b_len = float(np.hypot(rcv[0] - far[0], rcv[1] - far[1]))
    d_len = float(np.hypot(rcv[0] - src[0], rcv[1] - src[1]))
    delta = a_len + e + b_len - d_len
    from .._i18n import format_number

    ax.text(
        barrier_distance, top + 0.06 * receiver_distance,
        _t("Path difference {delta} m", language).format(
            delta=format_number(delta, language, decimals=2, trim=True)
        ),
        fontsize=8, ha="center", va="bottom",
    )
    y_dim = -0.10 * receiver_distance
    _dim(ax, (0.0, y_dim), (barrier_distance, y_dim),
         _metres(barrier_distance, language))
    _dim(ax, (0.0, y_dim - 0.06 * receiver_distance),
         (receiver_distance, y_dim - 0.06 * receiver_distance),
         _metres(receiver_distance, language))
    _dim(ax, (barrier_distance - 0.04 * receiver_distance, 0.0),
         (barrier_distance - 0.04 * receiver_distance, top),
         _metres(top, language))
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
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_barrier_geometry(...) with the original arguments."
        )
    return plot_barrier_geometry(
        ax=ax, source_height=result.source_height,
        barrier_distance=result.barrier_distance,
        barrier_height=result.barrier_height,
        receiver_distance=result.receiver_distance,
        receiver_height=result.receiver_height,
        thickness=result.thickness, language=language, **kwargs,
    )


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
        ax.plot_surface(
            d_grid * np.cos(a_grid), d_grid * np.sin(a_grid),
            np.zeros_like(d_grid), color=_C_SECONDARY_LIGHT, alpha=0.15,
            linewidth=0.0,
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
# Wall aperture (slit or circular hole) cross-section.
# ---------------------------------------------------------------------------
def plot_aperture_geometry(
    depth: float,
    ax: Axes | None = None,
    *,
    width: float | None = None,
    radius: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the section through a wall aperture to scale.

    Wall of thickness ``depth`` with a slit of the given ``width`` (or the
    diametral section of a circular hole of the given ``radius``), incident
    sound on the left and the transmitted wavefronts sketched on the right.
    Give exactly one of ``width``/``radius``, matching
    :func:`~phonometry.building.slit_transmission_coefficient` /
    :func:`~phonometry.building.circular_aperture_transmission_coefficient`.

    :param depth: Wall thickness ``d``, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param width: Slit width ``w``, in metres.
    :param radius: Circular-hole radius, in metres.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the wall rectangles.
    :return: The axes.
    """
    from matplotlib.patches import Arc

    _check_language(language)
    if depth <= 0.0:
        raise ValueError("'depth' must be positive.")
    if (width is None) == (radius is None):
        raise ValueError("Give exactly one of 'width' or 'radius'.")
    opening = float(width) if width is not None else 2.0 * float(radius or 0.0)
    if opening <= 0.0:
        raise ValueError("The aperture size must be positive.")
    if ax is None:
        ax = _new_axes()
    wall_h = max(4.0 * opening, 1.1 * depth)
    for y0 in (0.5 * opening, -0.5 * opening - wall_h):
        _material_rect(ax, 0.0, y0, depth, wall_h, "rigid", **kwargs)
    ax.text(
        0.5 * depth, 0.5 * opening + wall_h * 1.03, _t("Wall", language),
        fontsize=8, ha="center", va="bottom",
    )
    reach = max(2.5 * opening, 0.45 * depth)
    _incidence_arrow(ax, -2.1 * reach, 0.0, 1.2 * reach, language)
    for k in (0.45, 0.72, 1.0):
        ax.add_patch(Arc(
            (depth, 0.0), 2.0 * k * reach, 2.0 * k * reach,
            theta1=-80.0, theta2=80.0, color=_C_PRIMARY, linewidth=1.0,
        ))
    off = max(opening, 0.12 * depth)
    _dim(ax, (0.0, -0.5 * opening), (0.0, 0.5 * opening),
         _mm(opening, language), offset=1.5 * off, tight=True)
    _dim(ax, (0.0, 0.5 * opening + 0.55 * wall_h),
         (depth, 0.5 * opening + 0.55 * wall_h),
         _mm(depth, language), tight=depth < 3.0 * opening)
    _finish_geometry_axes(
        ax, _t("Wall aperture cross-section", language)
    )
    return ax


def plot_aperture_result_geometry(
    result: ApertureTransmissionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Aperture section for a result that retained its geometry."""
    if result.depth is None or (
        result.width is None and result.radius is None
    ):
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_aperture_geometry(depth, ...) with the original arguments."
        )
    return plot_aperture_geometry(
        result.depth, ax=ax, width=result.width, radius=result.radius,
        language=language, **kwargs,
    )


# ---------------------------------------------------------------------------
# Baffled piston with its directivity lobe.
# ---------------------------------------------------------------------------
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
        -0.5 * wall, a + baffle_h * 1.03, _t("Baffle", language),
        fontsize=8, ha="center", va="bottom",
    )
    kwargs.setdefault("facecolor", _C_SECONDARY_LIGHT)
    kwargs.setdefault("edgecolor", _C_EDGE)
    from matplotlib.patches import Rectangle

    piston = Rectangle((-wall, -a), 0.6 * wall, 2.0 * a, **kwargs)
    ax.add_patch(piston)
    ax.plot([0.0, 3.2 * a], [0.0, 0.0], linestyle=":", linewidth=0.8,
            color=_C_MUTED)
    if angles is not None and directivity is not None:
        ang = np.asarray(angles, dtype=np.float64)
        d_lin = np.abs(np.asarray(directivity, dtype=np.float64))
        if ang.shape != d_lin.shape:
            raise ValueError("'angles' and 'directivity' must match.")
        peak = float(d_lin.max())
        if peak > 0.0:
            scale = 2.8 * a / peak
            ax.plot(
                d_lin * scale * np.cos(ang), d_lin * scale * np.sin(ang),
                linewidth=1.6, color=_C_PRIMARY,
                label=lobe_label or _t("Normalised directivity", language),
            )
            ax.legend(loc=_LEGEND_LOC, fontsize=8)
    _dim(ax, (-1.4 * wall, -a), (-1.4 * wall, a),
         "2a = " + _mm(2.0 * a, language))
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
                f"'frequency_index' must index the {n_rows} computed "
                "frequencies."
            )
        row = directivity[frequency_index]
        angles = np.asarray(result.angles, dtype=np.float64)
        lobe = row
        from .._i18n import format_number

        label = "ka = " + format_number(
            float(ka[frequency_index]), language, decimals=1, trim=True
        )
    return plot_piston_geometry(
        result.radius, ax=ax, angles=angles, directivity=lobe,
        lobe_label=label, language=language, **kwargs,
    )


# ---------------------------------------------------------------------------
# Plenum chamber section.
# ---------------------------------------------------------------------------
def plot_plenum_geometry(
    exit_area: float,
    line_of_sight: float,
    wall_area: float,
    ax: Axes | None = None,
    *,
    angle: float = 0.0,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the plenum-chamber section honouring the acoustic geometry.

    The two truly geometric parameters of
    :func:`~phonometry.noise_control.plenum_attenuation` are drawn exactly:
    the inlet-to-outlet line of sight ``r`` and its ``angle`` off the inlet
    axis fix the box; the exit area sets the drawn outlet mouth (square-duct
    side ``sqrt(S_out)``) and the wall area is annotated.

    :param exit_area: Outlet area ``S_out``, in m2.
    :param line_of_sight: Inlet-to-outlet distance ``r``, in metres.
    :param wall_area: Total internal wall area ``S_w``, in m2 (annotation).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param angle: Angle between the inlet axis and the line of sight, in
        radians (0 <= angle < pi/2).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the wall-segment ``plot`` calls
        (line properties such as ``linewidth`` or ``color``).
    :return: The axes.
    """
    from matplotlib.patches import Rectangle

    _check_language(language)
    if exit_area <= 0.0 or line_of_sight <= 0.0 or wall_area <= 0.0:
        raise ValueError(
            "'exit_area', 'line_of_sight' and 'wall_area' must be positive."
        )
    if not 0.0 <= angle < 0.5 * np.pi:
        raise ValueError("'angle' must be in [0, pi/2).")
    if ax is None:
        ax = _new_axes()
    r = float(line_of_sight)
    mouth = float(np.sqrt(exit_area))
    duct = max(mouth, 0.18 * r)
    margin = max(0.5 * duct, 0.12 * r)
    width = r * float(np.cos(angle))
    rise = r * float(np.sin(angle))
    box_w = width
    box_h = rise + 2.0 * margin
    y_in = margin
    y_out = margin + rise
    lw = kwargs.pop("linewidth", 1.6)
    colour = kwargs.pop("color", _C_EDGE)
    # Walls drawn as segments so the inlet and outlet mouths stay open.
    walls = [
        ((0.0, box_w), (0.0, 0.0)),
        ((0.0, box_w), (box_h, box_h)),
        ((0.0, 0.0), (0.0, y_in - 0.5 * duct)),
        ((0.0, 0.0), (y_in + 0.5 * duct, box_h)),
        ((box_w, box_w), (0.0, y_out - 0.5 * mouth)),
        ((box_w, box_w), (y_out + 0.5 * mouth, box_h)),
    ]
    for (x_pair, y_pair) in walls:
        ax.plot(x_pair, y_pair, color=colour, linewidth=lw, **kwargs)
    stub = 0.35 * r
    wall_t = max(0.05 * duct, 0.002)
    ax.add_patch(Rectangle((-stub, y_in - 0.5 * duct), stub, duct,
                           facecolor="none", edgecolor=_C_EDGE,
                           linewidth=1.2))
    for y_wall in (y_in - 0.5 * duct - wall_t, y_in + 0.5 * duct):
        _material_rect(ax, -stub, y_wall, stub, wall_t, "plate",
                       linewidth=0.5)
    ax.text(-0.5 * stub, y_in + 0.85 * duct, _t("Inlet", language),
            fontsize=8, ha="center", va="bottom")
    # Outlet mouth on the right wall.
    ax.plot([box_w, box_w], [y_out - 0.5 * mouth, y_out + 0.5 * mouth],
            color=_C_PRIMARY, linewidth=3.0)
    ax.text(box_w * 1.02, y_out + 0.7 * mouth, _t("Outlet", language),
            fontsize=8, ha="left", va="bottom")
    # Line of sight, labelled along its own slope.
    ax.plot([0.0, box_w], [y_in, y_out], linestyle="--", linewidth=1.2,
            color=_C_SECONDARY)
    slope = float(np.degrees(angle))
    ax.text(
        0.5 * box_w, 0.5 * (y_in + y_out) + 0.03 * r,
        "r = " + _metres(r, language), fontsize=8, ha="center",
        va="bottom", rotation=slope, rotation_mode="anchor",
        color=_C_SECONDARY,
    )
    from .._i18n import format_number

    ax.text(
        0.5 * box_w, -0.08 * r,
        "S_w = " + format_number(wall_area, language, decimals=1, trim=True)
        + " m$^2$, S_out = "
        + format_number(exit_area, language, decimals=2, trim=True)
        + " m$^2$",
        fontsize=8, ha="center", va="top",
    )
    _finish_geometry_axes(ax, _t("Plenum chamber section", language))
    return ax


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
            ax, x0, y0, width, height, "cavity", edgecolor="none",
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
            [seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]],
            color=colour, linewidth=2.6,
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
    from .._i18n import localize_axes

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
        ax.imshow(overlay, extent=extent, origin="upper",
                  interpolation="nearest", **kwargs)
    _fdtd_sponge_bands(ax, sim, lx, ly, handles)
    _fdtd_edges(ax, sim, lx, ly, handles)
    for src in getattr(sim, "_sources", ()):  # star per source
        (marker,) = ax.plot(
            [(src.ix + 0.5) * sim.dx], [(src.iy + 0.5) * sim.dx],
            marker="*", markersize=11, color=_C_REFERENCE, linestyle="none",
        )
        handles["source"] = marker
    if pts is not None:
        (dots,) = ax.plot(
            pts[:, 0], pts[:, 1], marker="o", markersize=5, color=_C_MUTED,
            markeredgecolor="black", linestyle="none",
        )
        handles["probe"] = dots
    ax.set_xlim(0.0, lx)
    ax.set_ylim(ly, 0.0)
    ax.set_aspect("equal")
    ax.set_xlabel(_t(_AXIS_X, language))
    ax.set_ylabel(_t(_AXIS_Y, language))
    ax.set_title(_t("FDTD domain", language), fontweight="bold")
    label_keys = {
        "sponge": "Sponge layer", "impedance": "Impedance edge",
        "rigid": "Rigid edge", "source": "Source", "probe": "Probe",
    }
    if handles:
        ax.legend(
            handles.values(),
            [_t(label_keys[k], language) for k in handles],
            loc=_LEGEND_LOC, fontsize=8,
        )
    localize_axes(ax, language)
    return ax
