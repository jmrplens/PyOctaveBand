#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the materials domain: specimens, panels and rigs.

One subject seen from both ends: the treatment you build and the apparatus you
measure it in. The layered absorber stack, the Helmholtz and slit metamaterial
resonators, the QRD well profile and the metadiffuser panel are designs; the
impedance tube (ISO 10534-2), the transmission tube (ASTM E2611), the
free-field diffusion goniometer, the dynamic stiffness rig and the in-situ
road set-up are the rigs that put a specimen of one of them in front of a
microphone.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level, and layer dataclasses are
dispatched by class name at runtime for the same reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _new_axes,
)
from ._draft import (
    _check_language,
    _dim,
    _finish_geometry_axes,
    _incidence_arrow,
    _loudspeaker,
    _material_rect,
    _metres,
    _microphone,
    _mm,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...materials.absorbers.impedance_tube import (
        ImpedanceTubeResult,
        TransferMatrix,
    )
    from ...materials.absorbers.porous import Layer, LayeredAbsorberResult
    from ...materials.absorbers.slow_sound import (
        HelmholtzResonator,
        SlitResonatorAbsorberResult,
    )
    from ...materials.diffusers.design import DiffuserPolarResponse
    from ...materials.diffusers.metadiffuser import MetadiffuserResult
    from ...materials.surfaces.road_absorption import InsituAbsorptionResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Metadiffuser cross-section (one period)":
        "Sección del metadifusor (un periodo)",
    "Air": "Aire",
    "Porous": "Poroso",
    "Poroelastic": "Poroelástico",
    "Perforated plate": "Placa perforada",
    "Microperforated plate": "Placa microperforada",
    "Membrane": "Membrana",
    "Rigid backing": "Respaldo rígido",
    "Layered absorber cross-section": "Sección del absorbente multicapa",
    "Helmholtz resonator cross-section": "Sección del resonador de Helmholtz",
    "Slit metamaterial absorber cross-section (one period)":
        "Sección del absorbente metamaterial de rendija (un periodo)",
    "QRD well profile": "Perfil de pozos del difusor QRD",
    "Impedance tube (ISO 10534-2), to scale":
        "Tubo de impedancia (ISO 10534-2), a escala",
    "Transmission tube (ASTM E2611), to scale":
        "Tubo de transmisión (ASTM E2611), a escala",
    "Termination": "Terminación",
    "Cross-section": "Sección transversal",
    "Plane-wave range {fl} to {fu} Hz": "Rango de onda plana {fl} a {fu} Hz",
    "depth sequence {seq}": "secuencia de profundidades {seq}",
    "Slit": "Rendija",
    "mm": "mm",
    "In-situ absorption set-up": "Montaje de absorción in situ",
    "Road surface": "Superficie de la calzada",
    "Sampled area": "Zona muestreada",
    "Dynamic stiffness rig": "Banco de rigidez dinámica",
    "Load plate {mass} kg": "Placa de carga de {mass} kg",
    "Specimen": "Probeta",
    "Exciter": "Excitador",
    "Free-field diffusion goniometer (plan)":
        "Goniómetro de difusión en campo libre (planta)",
    "Sample": "Muestra",
    "Source": "Fuente",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


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
    if name == "PoroelasticLayer":
        return "porous", float(layer.thickness), "Poroelastic"
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
        :class:`~phonometry.materials.absorbers.slow_sound.HelmholtzResonator`.
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
    from ..._i18n import format_number

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


def _tube_frequency_note(
    ax: Axes,
    x: float,
    y: float,
    f_range: tuple[float, float],
    language: str,
) -> None:
    """The plane-wave working range printed under a tube drawing."""
    from ..._i18n import format_number

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
    from ...materials.absorbers.impedance_tube import plane_wave_frequency_range

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
    from ...materials.absorbers.impedance_tube import plane_wave_frequency_range_astm

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


def _draw_metadiffuser_well(ax: Axes, well: Any, x_slit: float,
                            depth: float, kwargs: dict[str, Any]) -> None:
    """One slit with its sideways resonator shelves, panel coordinates."""
    h = float(well.slit_height)
    _material_rect(ax, x_slit, 0.0, h, depth, "cavity", **kwargs)
    chain = list(well.resonators)
    step = depth / len(chain)
    for m, resonator in enumerate(chain):
        w_n = float(resonator.neck_side)
        l_n = float(resonator.neck_length)
        w_c = float(resonator.cavity_side)
        l_c = float(resonator.cavity_length)
        y_m = depth - (m + 0.5) * step
        x_neck = x_slit + h
        _material_rect(
            ax, x_neck, y_m - 0.5 * w_n, l_n, w_n, "cavity",
            edgecolor=_C_EDGE, linewidth=0.5,
        )
        _material_rect(
            ax, x_neck + l_n, y_m - 0.5 * w_c, l_c, w_c, "cavity",
            edgecolor=_C_EDGE, linewidth=0.5,
        )


def plot_metadiffuser_panel_geometry(
    wells: Sequence[Any],
    ax: Axes | None = None,
    *,
    depth: float,
    period: float,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw one period of a metadiffuser panel, to scale.

    Side cut of the slotted panel (Sci. Rep. 7:5389 Fig. 1(b)): the face
    runs along the top with the sound arriving from above, the wells
    repeat horizontally at the pitch ``d``, and each well is a slit of
    height ``h_n`` descending the panel depth, loaded by its resonators
    at the lattice step ``a = L / M`` shelved sideways into the septum;
    ``None`` wells are flat rigid strips; rigid back wall underneath.

    :param wells: The well sequence of
        :func:`~phonometry.materials.diffusers.metadiffuser.metadiffuser_reflection`
        (:class:`~phonometry.materials.diffusers.metadiffuser.MetadiffuserWell` or
        ``None`` per well).
    :param ax: Existing axes, or ``None`` to create a figure.
    :param depth: Panel depth ``L``, in metres.
    :param period: Well pitch ``d``, in metres.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the slit rectangles.
    :return: The axes.
    """
    _check_language(language)
    if depth <= 0.0 or period <= 0.0:
        raise ValueError("'depth' and 'period' must be positive.")
    cells = list(wells)
    if len(cells) < 2:
        raise ValueError("'wells' must contain at least two wells.")
    for well in cells:
        if well is not None and well.slit_height >= period:
            raise ValueError(
                "every slit height must be smaller than the period."
            )
    if ax is None:
        ax = _new_axes()
    n_wells = len(cells)
    d = period
    total = n_wells * d
    # Face along x with the sound arriving from above; the thin panel depth
    # runs downward (Fig. 1(b) of the paper). Each cell carries its slit at
    # the left, with the resonator necks and cavities branching sideways
    # into the solid septum between slits.
    _material_rect(
        ax, 0.0, 0.0, total, depth, "plate", linewidth=0.7, alpha=0.45,
    )
    back = 0.4 * depth
    _material_rect(ax, -0.01 * total, -back, 1.02 * total, back, "rigid")
    kwargs.setdefault("linewidth", 0.5)
    for index, well in enumerate(cells):
        if well is not None:
            _draw_metadiffuser_well(ax, well, index * d + 0.12 * d, depth,
                                    kwargs)
    for index, well in enumerate(cells):
        if well is None:
            continue
        x_mark = index * d + 0.12 * d + 0.5 * float(well.slit_height)
        ax.text(
            x_mark, depth - 0.012 * total, str(index + 1),
            fontsize=6, ha="center", va="top", color=_C_EDGE,
        )
    _incidence_arrow(
        ax, 1.5 * d, depth + 0.5 * total * 0.16, 0.4 * total * 0.16,
        language, downward=True,
    )
    off = 0.045 * total
    _dim(ax, (0.0, -back), (total, -back), _mm(total, language),
         offset=-off)
    _dim(ax, (total, 0.0), (total, depth), _mm(depth, language),
         offset=-1.3 * off, tight=True)
    _dim(ax, ((n_wells - 1) * d, depth), (total, depth),
         _mm(d, language), offset=0.6 * off)
    first = next(
        (well for well in cells if well is not None), None,
    )
    if first is not None:
        index = cells.index(first)
        x_slit = index * d + 0.12 * d
        h = float(first.slit_height)
        _dim(ax, (x_slit, depth), (x_slit + h, depth), _mm(h, language),
             offset=0.6 * off, tight=True)
    _finish_geometry_axes(
        ax, _t("Metadiffuser cross-section (one period)", language)
    )
    return ax


def plot_metadiffuser_geometry(
    result: MetadiffuserResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Panel drawing for a metadiffuser result that retained its geometry."""
    if result.wells is None or result.depth is None or result.period is None:
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_metadiffuser_panel_geometry(...) with the original "
            "arguments."
        )
    return plot_metadiffuser_panel_geometry(
        result.wells, ax=ax, depth=result.depth, period=result.period,
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
# In-situ absorption set-up (source over the road, microphone below).
# ---------------------------------------------------------------------------
def plot_insitu_geometry(
    ax: Axes | None = None,
    *,
    source_height: float = 1.25,
    mic_height: float = 0.25,
    sampled_radius: float | None = 1.34,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the in-situ surface-absorption set-up to scale.

    Loudspeaker on its mast above the surface, microphone below it, direct
    and surface-reflected paths, and the sampled-area radius on the ground.
    Defaults are the standard heights (1,25 m source, 0,25 m microphone)
    with the sampled radius of the standard 5 ms window.

    :param ax: Existing axes, or ``None`` to create a figure.
    :param source_height: Source height above the surface, in metres.
    :param mic_height: Microphone height, in metres (< ``source_height``).
    :param sampled_radius: Sampled-area radius drawn on the ground, in
        metres; ``None`` omits it.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the surface rectangle.
    :return: The axes.
    """
    _check_language(language)
    if mic_height <= 0.0 or source_height <= mic_height:
        raise ValueError(
            "'source_height' must exceed 'mic_height' and both be positive."
        )
    if sampled_radius is not None and sampled_radius <= 0.0:
        raise ValueError("'sampled_radius' must be positive when given.")
    if ax is None:
        ax = _new_axes()
    half = max(source_height, sampled_radius or 0.0) * 1.2
    _material_rect(
        ax, -half, -0.06 * source_height, 2.0 * half, 0.06 * source_height,
        "rigid", **kwargs,
    )
    ax.text(
        half * 1.02, -0.03 * source_height, _t("Road surface", language),
        fontsize=8, ha="left", va="center",
    )
    # Mast, source and microphone on the same vertical.
    ax.plot([0.0, 0.0], [0.0, source_height], color=_C_MUTED,
            linewidth=1.0, linestyle=":")
    _loudspeaker(
        ax, 0.0, source_height, 0.28 * source_height, language
    )
    _microphone(ax, 0.0, mic_height, 0.22 * source_height, "")
    # Direct and reflected paths to the microphone.
    ax.plot([0.0, 0.0], [source_height, mic_height], color=_C_PRIMARY,
            linewidth=1.6)
    ax.plot([0.0, 0.0], [mic_height, 0.0], color=_C_SECONDARY,
            linewidth=1.4, linestyle="--")
    if sampled_radius is not None:
        ax.plot(
            [-sampled_radius, sampled_radius],
            [0.012 * source_height] * 2,
            color=_C_PRIMARY, linewidth=3.0, solid_capstyle="butt",
        )
        _dim(ax, (0.0, -0.12 * source_height),
             (sampled_radius, -0.12 * source_height),
             _metres(sampled_radius, language))
        ax.text(
            -0.5 * sampled_radius, -0.14 * source_height,
            _t("Sampled area", language), fontsize=8, ha="center", va="top",
        )
    off = 0.12 * source_height
    _dim(ax, (-2.0 * off, 0.0), (-2.0 * off, mic_height),
         _metres(mic_height, language), tight=True)
    _dim(ax, (-4.0 * off, 0.0), (-4.0 * off, source_height),
         _metres(source_height, language))
    _finish_geometry_axes(ax, _t("In-situ absorption set-up", language))
    return ax


def plot_insitu_result_geometry(
    result: InsituAbsorptionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Set-up drawing for a result that retained its geometry."""
    if result.source_height is None or result.mic_height is None:
        raise ValueError(
            "This result does not retain its set-up heights; call "
            "plot_insitu_geometry(source_height=..., mic_height=...)."
        )
    return plot_insitu_geometry(
        ax=ax, source_height=result.source_height,
        mic_height=result.mic_height, language=language, **kwargs,
    )


# ---------------------------------------------------------------------------
# Dynamic stiffness rig (resonance method).
# ---------------------------------------------------------------------------
def plot_dynamic_stiffness_rig(
    ax: Axes | None = None,
    *,
    specimen_side: float = 0.2,
    specimen_thickness: float = 0.02,
    load_mass: float = 8.0,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the dynamic-stiffness resonance rig to scale.

    Resilient specimen on the rigid base, the standard square load plate on
    top (its mass annotated), the exciter above and an accelerometer on the
    plate; defaults are the standard 200 mm square specimen under the 8 kg
    plate.

    :param ax: Existing axes, or ``None`` to create a figure.
    :param specimen_side: Specimen side length, in metres.
    :param specimen_thickness: Specimen thickness, in metres.
    :param load_mass: Load-plate mass, in kilograms (annotation).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the specimen rectangle.
    :return: The axes.
    """
    _check_language(language)
    if specimen_side <= 0.0 or specimen_thickness <= 0.0 or load_mass <= 0.0:
        raise ValueError(
            "'specimen_side', 'specimen_thickness' and 'load_mass' must be "
            "positive."
        )
    if ax is None:
        ax = _new_axes()
    side = float(specimen_side)
    t = float(specimen_thickness)
    plate_t = 0.125 * side
    from ..._i18n import format_number

    # Rigid base, specimen, load plate.
    _material_rect(ax, -0.75 * side, -0.15 * side, 1.5 * side, 0.15 * side,
                   "rigid")
    _material_rect(ax, -0.5 * side, 0.0, side, t, "porous", **kwargs)
    ax.text(0.55 * side, 0.5 * t, _t("Specimen", language), fontsize=8,
            ha="left", va="center")
    _material_rect(ax, -0.5 * side, t, side, plate_t, "plate")
    ax.text(
        0.0, t + 0.5 * plate_t,
        _t("Load plate {mass} kg", language).format(
            mass=format_number(load_mass, language, decimals=0, trim=True)
        ),
        fontsize=8, ha="center", va="center",
    )
    # Exciter above, accelerometer on the plate.
    _material_rect(ax, -0.1 * side, t + plate_t + 0.25 * side, 0.2 * side,
                   0.25 * side, "plate")
    ax.annotate(
        "", xy=(0.0, t + plate_t), xytext=(0.0, t + plate_t + 0.25 * side),
        arrowprops={"arrowstyle": "-|>", "color": _C_REFERENCE,
                    "linewidth": 1.6},
    )
    ax.text(0.14 * side, t + plate_t + 0.32 * side, _t("Exciter", language),
            fontsize=8, ha="left", va="center")
    _microphone(ax, 0.35 * side, t + plate_t, 0.18 * side, "")
    _dim(ax, (-0.5 * side, -0.2 * side), (0.5 * side, -0.2 * side),
         _mm(side, language))
    _dim(ax, (-0.62 * side, 0.0), (-0.62 * side, t), _mm(t, language),
         tight=True)
    _finish_geometry_axes(ax, _t("Dynamic stiffness rig", language))
    return ax


# ---------------------------------------------------------------------------
# Free-field diffusion goniometer (plan view).
# ---------------------------------------------------------------------------
def plot_goniometer_geometry(
    ax: Axes | None = None,
    *,
    source_distance: float = 10.0,
    receiver_radius: float = 5.0,
    angular_step: float = 5.0,
    sample_width: float = 0.6,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the free-field diffusion goniometer in plan, to scale.

    Receiver semicircle at its radius with one microphone per angular step,
    the source on the normal at its distance and the sample at the centre;
    defaults are the standard 10 m source, 5 m receiver arc and 5-degree
    resolution (37 microphones).

    :param ax: Existing axes, or ``None`` to create a figure.
    :param source_distance: Source distance from the sample, in metres.
    :param receiver_radius: Receiver-arc radius, in metres.
    :param angular_step: Angular spacing of the receivers, in degrees.
    :param sample_width: Drawn sample width, in metres.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the receiver scatter.
    :return: The axes.
    """
    _check_language(language)
    if source_distance <= 0.0 or receiver_radius <= 0.0:
        raise ValueError(
            "'source_distance' and 'receiver_radius' must be positive."
        )
    if not 0.0 < angular_step <= 90.0:
        raise ValueError("'angular_step' must be in (0, 90] degrees.")
    if sample_width <= 0.0:
        raise ValueError("'sample_width' must be positive.")
    if ax is None:
        ax = _new_axes()
    angles = np.radians(np.arange(-90.0, 90.0 + 0.5 * angular_step,
                                  angular_step))
    xs = receiver_radius * np.sin(angles)
    ys = receiver_radius * np.cos(angles)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("s", 14)
    ax.scatter(xs, ys, zorder=5, **kwargs)
    ax.plot(xs, ys, color=_C_MUTED, linewidth=0.6, linestyle=":", zorder=2)
    # Sample slab at the origin on the baseline.
    _material_rect(
        ax, -0.5 * sample_width, -0.03 * receiver_radius, sample_width,
        0.03 * receiver_radius, "porous",
    )
    ax.text(-0.6 * sample_width, -0.015 * receiver_radius,
            _t("Sample", language), fontsize=8, ha="right", va="center")
    ax.plot([0.0], [source_distance], marker="*", markersize=13,
            color=_C_REFERENCE, linestyle="none", zorder=6)
    ax.text(0.0, 1.03 * source_distance, _t("Source", language), fontsize=8,
            ha="center", va="bottom")
    ax.plot([0.0, 0.0], [0.0, source_distance], color=_C_MUTED,
            linewidth=0.8, linestyle="--", zorder=2)
    _dim(ax, (0.0, 0.0), (receiver_radius, 0.0),
         _metres(receiver_radius, language), offset=-0.06 * receiver_radius)
    ax.text(
        0.02 * source_distance, 0.5 * (receiver_radius + source_distance),
        _metres(source_distance, language), fontsize=8, ha="left",
        va="center", rotation=90,
    )
    _finish_geometry_axes(
        ax, _t("Free-field diffusion goniometer (plan)", language)
    )
    return ax
