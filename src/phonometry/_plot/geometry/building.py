#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the building domain: what the sound gets through.

Three sections through a building element, each drawn from the arguments of
the transmission model it belongs to: the slit or circular aperture through a
wall of finite thickness, the composite facade elevation with every element
drawn to its real share of the area, and the mass-spring-mass double wall whose
leaf thicknesses follow from their surface densities.

Domain classes are referenced only under ``TYPE_CHECKING`` so this rendering
leaf never imports domain code at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..._internal.validation import require_positive
from ..common import (
    _C_PRIMARY,
    _new_axes,
)
from ._draft import (
    _check_language,
    _chip,
    _dim,
    _finish_geometry_axes,
    _incidence_arrow,
    _material_rect,
    _metres,
    _mm,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ...building.prediction.aperture_transmission import (
        ApertureTransmissionResult,
    )
    from ...building.prediction.facade import FacadeElement, FacadePredictionResult
    from ...building.prediction.panel_transmission import SoundReductionResult

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    "Wall aperture cross-section": "Sección de la abertura en el muro",
    "Wall": "Muro",
    "Composite facade elevation (areas to scale)": "Alzado de fachada compuesta (áreas a escala)",
    "Double wall cross-section": "Sección de la doble hoja",
}


def _t(text: str, language: str = "en") -> str:
    """Translate a fixed UI string to Spanish, else return it unchanged."""
    return _STRINGS.get(text, text) if language == "es" else text


# ---------------------------------------------------------------------------
# Wall aperture (slit or circular hole) cross-section.
# ---------------------------------------------------------------------------
@overload
def plot_aperture_geometry(
    depth: float,
    ax: Axes | None = ...,
    *,
    width: float,
    language: str = ...,
    **kwargs: Any,
) -> Axes: ...


@overload
def plot_aperture_geometry(
    depth: float,
    ax: Axes | None = ...,
    *,
    radius: float,
    language: str = ...,
    **kwargs: Any,
) -> Axes: ...


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
    :raises ValueError: for a ``depth``, ``width`` or ``radius`` that is not
        finite and positive, or for anything other than exactly one of
        ``width``/``radius``.
    """
    from matplotlib.patches import Arc

    _check_language(language)
    # Through the shared guard, which rejects NaN and the infinities the bare
    # ``<= 0.0`` waved through: a non-finite depth or opening poisons every
    # coordinate and leaves a titled, blank section.
    require_positive(depth, "depth")
    if width is not None and radius is None:
        opening = require_positive(width, "width")
    elif radius is not None and width is None:
        opening = 2.0 * require_positive(radius, "radius")
    else:
        msg = "Give exactly one of 'width' or 'radius'."
        raise ValueError(msg)
    if ax is None:
        ax = _new_axes()
    wall_h = max(4.0 * opening, 1.1 * depth)
    for y0 in (0.5 * opening, -0.5 * opening - wall_h):
        _material_rect(ax, 0.0, y0, depth, wall_h, "rigid", **kwargs)
    ax.text(
        0.5 * depth,
        0.5 * opening + wall_h * 1.03,
        _t("Wall", language),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    reach = max(2.5 * opening, 0.45 * depth)
    _incidence_arrow(ax, -2.1 * reach, 0.0, 1.2 * reach, language)
    for k in (0.45, 0.72, 1.0):
        ax.add_patch(
            Arc(
                (depth, 0.0),
                2.0 * k * reach,
                2.0 * k * reach,
                theta1=-80.0,
                theta2=80.0,
                color=_C_PRIMARY,
                linewidth=1.0,
            )
        )
    off = max(opening, 0.12 * depth)
    _dim(
        ax,
        (0.0, -0.5 * opening),
        (0.0, 0.5 * opening),
        _mm(opening, language),
        offset=1.5 * off,
        tight=True,
    )
    _dim(
        ax,
        (0.0, 0.5 * opening + 0.55 * wall_h),
        (depth, 0.5 * opening + 0.55 * wall_h),
        _mm(depth, language),
        tight=depth < 3.0 * opening,
    )
    _finish_geometry_axes(ax, _t("Wall aperture cross-section", language))
    return ax


def plot_aperture_result_geometry(
    result: ApertureTransmissionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Aperture section for a result that retained its geometry."""
    # One branch per shape, which is what the result carries: a 'slit' keeps
    # its width and a 'circular' aperture its radius, never both. Passing both
    # through and letting one be None would not match either call form.
    if result.depth is not None:
        if result.width is not None:
            return plot_aperture_geometry(
                result.depth,
                ax=ax,
                width=result.width,
                language=language,
                **kwargs,
            )
        if result.radius is not None:
            return plot_aperture_geometry(
                result.depth,
                ax=ax,
                radius=result.radius,
                language=language,
                **kwargs,
            )
    msg = (
        "This result does not retain its geometry; call "
        "plot_aperture_geometry(depth, ...) with the original arguments."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Composite facade elevation.
# ---------------------------------------------------------------------------
def plot_facade_elements(
    elements: Sequence[FacadeElement],
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw a composite facade elevation with element areas to scale.

    Each element of :func:`~phonometry.building.facade_sound_reduction` is a
    tile whose drawn area equals its real area (small-area elements without
    an area, such as airbricks rated by ``dn_e``, get a nominal 0,1 m2 tile).

    :param elements: The
        :class:`~phonometry.building.FacadeElement` sequence.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the first element rectangle.
    :return: The axes.
    :raises ValueError: for an empty sequence, or an element that supplies an
        ``area`` which is not positive and finite. An element with no ``area``
        at all is the nominal tile above, not a refusal.
    """
    _check_language(language)
    tiles = list(elements)
    if not tiles:
        msg = "'elements' must contain at least one element."
        raise ValueError(msg)
    areas: list[float] = []
    for element in tiles:
        area = getattr(element, "area", None)
        if area is None:
            areas.append(0.1)  # nominal tile for dn_e-rated elements
        else:
            # Named per element, and finite: a NaN area passed the old
            # ``<= 0.0`` into the tile widths and rendered a titled, empty
            # elevation. :class:`~phonometry.building.FacadeElement` refuses
            # one at construction with this same message, so a caller who
            # duck-types the sequence gets the same refusal here.
            areas.append(require_positive(area, "area"))
    if ax is None:
        ax = _new_axes()
    total = sum(areas)
    height = float(np.sqrt(total / 2.0))  # a 2:1 facade footprint
    from ..._i18n import format_number

    x = 0.0
    fills = ("rigid", "cavity", "plate", "porous")
    for index, (element, area) in enumerate(zip(tiles, areas, strict=True)):
        width = area / height
        _material_rect(
            ax,
            x,
            0.0,
            width,
            height,
            fills[index % len(fills)],
            **(dict(kwargs) if index == 0 else {}),
        )
        label = getattr(element, "name", "") or f"#{index + 1}"
        ax.text(
            x + 0.5 * width,
            0.5 * height,
            label,
            fontsize=8,
            ha="center",
            va="center",
            rotation=90 if width < 0.35 * height else 0,
            zorder=6,
            # Every element is named across the middle of its own tile, and
            # the hatched fills draw straight through the word. The chip goes
            # on all of them, not only the hatched ones, so the row of names
            # reads as one row.
            bbox=_chip(ax, 0.2),
        )
        ax.text(
            x + 0.5 * width,
            -0.04 * height,
            format_number(area, language, decimals=1, trim=True) + " m$^2$",
            fontsize=8,
            ha="center",
            va="top",
        )
        x += width
    _dim(ax, (0.0, 1.06 * height), (x, 1.06 * height), _metres(x, language))
    _dim(ax, (-0.03 * x, 0.0), (-0.03 * x, height), _metres(height, language))
    _finish_geometry_axes(
        ax, _t("Composite facade elevation (areas to scale)", language)
    )
    return ax


def plot_facade_result_geometry(
    result: FacadePredictionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Facade elevation for a prediction that retained its ``elements``."""
    if result.elements is None:
        msg = (
            "This result does not retain its elements; call "
            "plot_facade_elements(elements) with the original sequence."
        )
        raise ValueError(msg)
    return plot_facade_elements(result.elements, ax=ax, language=language, **kwargs)


# ---------------------------------------------------------------------------
# Double wall (mass-spring-mass) cross-section.
# ---------------------------------------------------------------------------
#: Drawn leaf thickness per unit surface density (gypsum board density
#: ~700 kg/m3 gives 12,5 mm for the classic 8,8 kg/m2 board).
_LEAF_DENSITY = 700.0


def plot_double_wall_geometry(
    mass1: float,
    mass2: float,
    gap: float,
    ax: Axes | None = None,
    *,
    resonance_frequency: float | None = None,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Draw the mass-spring-mass double wall to scale.

    Two leaves separated by the ``gap``; leaf thicknesses are drawn from the
    surface densities at a nominal board density, and the mass-spring-mass
    resonance is annotated when given.

    :param mass1: Surface density of the first leaf, in kg/m2.
    :param mass2: Surface density of the second leaf, in kg/m2.
    :param gap: Cavity depth, in metres.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param resonance_frequency: Optional ``f0`` to annotate, in Hz.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the leaf rectangles.
    :return: The axes.
    :raises ValueError: naming the first of the three that is not finite and
        positive, or ``resonance_frequency`` when one is passed that is not.
    """
    _check_language(language)
    require_positive(mass1, "mass1")
    require_positive(mass2, "mass2")
    require_positive(gap, "gap")
    # The annotation between the leaves is the one number the drawing does
    # not derive from the geometry, and it went through unchecked: a NaN
    # lettered 'f$_0$ = nan Hz' across an otherwise complete section. The one
    # producer, :func:`~phonometry.building.double_wall_transmission_loss`,
    # computes f0 from three surface densities it has already pinned
    # positive, so nothing the library emits is refused here.
    if resonance_frequency is not None:
        require_positive(resonance_frequency, "resonance_frequency")
    if ax is None:
        ax = _new_axes()
    t1 = mass1 / _LEAF_DENSITY
    t2 = mass2 / _LEAF_DENSITY
    height = 2.6 * (t1 + gap + t2)
    from ..._i18n import format_number

    _material_rect(ax, 0.0, 0.0, t1, height, "plate", **kwargs)
    _material_rect(ax, t1 + gap, 0.0, t2, height, "plate")
    for x, mass, thickness in ((0.0, mass1, t1), (t1 + gap, mass2, t2)):
        ax.text(
            x + 0.5 * thickness,
            0.72 * height,
            format_number(mass, language, decimals=1, trim=True) + " kg/m2",
            fontsize=8,
            ha="center",
            va="center",
            rotation=90,
        )
    _dim(ax, (t1, 0.0), (t1 + gap, 0.0), _mm(gap, language), offset=-0.06 * height)
    _dim(
        ax, (0.0, 0.0), (t1, 0.0), _mm(t1, language), offset=-0.16 * height, tight=True
    )
    _incidence_arrow(ax, -0.5 * height * 0.5, 0.5 * height, 0.2 * height, language)
    if resonance_frequency is not None:
        ax.text(
            t1 + 0.5 * gap,
            0.5 * height,
            "f$_0$ = "
            + format_number(resonance_frequency, language, decimals=0, trim=True)
            + " Hz",
            fontsize=8,
            ha="center",
            va="center",
            rotation=90,
        )
    _finish_geometry_axes(ax, _t("Double wall cross-section", language))
    return ax


def plot_double_wall_result_geometry(
    result: SoundReductionResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Double-wall drawing for a result that retained its geometry."""
    if result.mass1 is None or result.mass2 is None or result.gap is None:
        msg = (
            "This result does not retain its double-wall geometry; call "
            "plot_double_wall_geometry(mass1, mass2, gap)."
        )
        raise ValueError(msg)
    return plot_double_wall_geometry(
        result.mass1,
        result.mass2,
        result.gap,
        ax=ax,
        resonance_frequency=result.resonance_frequency,
        language=language,
        **kwargs,
    )
