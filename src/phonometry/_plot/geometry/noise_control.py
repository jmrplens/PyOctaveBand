#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the noise-control domain: silencers and plenums.

Two families of duct element, drawn from the same arguments their attenuation
models take. The reactive silencer renderer covers the four ``kind`` strings of
:class:`~phonometry.noise_control.silencers.ReactiveSilencerResult` (expansion
chamber, extended-tube chamber, Helmholtz resonator and quarter-wave stub),
validating that the geometry it is handed can actually be built; the plenum
renderer draws the inlet, the outlet and the line of sight between them.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..common import (
    _C_EDGE,
    _C_PRIMARY,
    _C_SECONDARY,
    _new_axes,
)
from ._draft import (
    _check_language,
    _dim,
    _finish_geometry_axes,
    _material_rect,
    _metres,
    _mm,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ...noise_control.silencers import ReactiveSilencerResult

#: The ``ReactiveSilencerResult.kind`` strings, shared with the dispatcher.
_KIND_EXPANSION = "expansion chamber"
_KIND_EXTENDED = "extended-tube chamber"
_KIND_HELMHOLTZ = "Helmholtz resonator"
_KIND_QUARTER = "quarter-wave resonator"

#: Shared validation message for length arguments.
_LENGTH_POSITIVE = "'length' must be positive."


#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Reactive silencer cross-section": "Sección del silenciador reactivo",
    _KIND_EXPANSION: "cámara de expansión",
    _KIND_HELMHOLTZ: "resonador de Helmholtz",
    _KIND_QUARTER: "resonador de cuarto de onda",
    _KIND_EXTENDED: "cámara con tubos extendidos",
    "Plenum chamber section": "Sección de la cámara plenum",
    "Inlet": "Entrada",
    "Outlet": "Salida",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


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
    from ..._i18n import format_number

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
    # A Helmholtz neck is a slot a couple of centimetres long and a few
    # millimetres wide: both of its dimensions are shorter than the text that
    # labels them. Their labels therefore go on the far side of the dimension
    # line, past the end of the extension lines, and the neck length is
    # lettered upright so that it fits between the duct wall and the cavity.
    neck = kind == _KIND_HELMHOLTZ
    if neck and cavity_volume is not None:
        _draw_hr_cavity(
            ax, y0, branch_len, cavity_side, cavity_volume, language
        )
        _dim(ax, (0.5 * d_b, y0), (0.5 * d_b, y0 + branch_len),
             _mm(branch_len, language), offset=-2.0 * off, tight=True,
             label_side=-1.0, label_upright=True)
    else:
        # Closed end of the quarter-wave tube.
        ax.plot([-0.5 * d_b, 0.5 * d_b],
                [y0 + branch_len, y0 + branch_len],
                color=_C_EDGE, linewidth=2.2)
        _dim(ax, (0.5 * d_b, y0), (0.5 * d_b, y0 + branch_len),
             _mm(branch_len, language), offset=-2.0 * off)
    _dim(ax, (-0.5 * d_b, y0), (0.5 * d_b, y0), _mm(d_b, language),
         offset=-1.5 * off, tight=True, label_side=-1.0 if neck else 1.0)
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
    from ..._i18n import format_number

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
