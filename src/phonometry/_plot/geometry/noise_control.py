#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Geometry drawings of the noise-control domain: silencers and plenums.

Two families of duct element, drawn from the same arguments their attenuation
models take. The reactive silencer renderer covers the four ``kind`` strings of
:class:`~phonometry.noise_control.silencers.ReactiveSilencerResult` (expansion
chamber, extended-tube chamber, Helmholtz resonator and quarter-wave stub),
validating that the geometry it is handed can actually be built; the plenum
renderer draws the inlet, the outlet and the line of sight between them.

A third renderer draws what a hand-built
:class:`~phonometry.noise_control.silencers.SilencerChain` declares, which is
less than a named device does and is drawn as such: the ducts to scale, and
the shunts as marked branch points, because a shunt element is handed an
impedance and no shape at all.

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
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from ...noise_control.silencers import (
        ReactiveSilencerResult,
        SilencerChain,
        SilencerChainElement,
    )

#: The ``ReactiveSilencerResult.kind`` strings, shared with the dispatcher.
_KIND_EXPANSION = "expansion chamber"
_KIND_EXTENDED = "extended-tube chamber"
_KIND_HELMHOLTZ = "Helmholtz resonator"
_KIND_QUARTER = "quarter-wave resonator"
_KIND_CHAIN = "element chain"

#: Shared validation message for length arguments.
_LENGTH_POSITIVE = "'length' must be positive."

#: The note that keeps a chain drawing honest, rendered under it whenever it
#: carries a branch point (see :func:`plot_silencer_chain_geometry`).
_CHAIN_NOTE = (
    "Ducts drawn to scale from their declared length and area. A side branch\n"
    "declares only an impedance, so it is marked where it joins, not drawn."
)


#: Title every reactive-silencer drawing carries, whichever renderer draws it.
_CROSS_SECTION = "Reactive silencer cross-section"

#: Spanish translations of the fixed strings rendered here, keyed by their
#: verbatim English text. ``_t`` returns the English key unchanged for any
#: language other than ``"es"``.
_STRINGS: dict[str, str] = {
    _CROSS_SECTION: "Sección del silenciador reactivo",
    _KIND_EXPANSION: "cámara de expansión",
    _KIND_HELMHOLTZ: "resonador de Helmholtz",
    _KIND_QUARTER: "resonador de cuarto de onda",
    _KIND_EXTENDED: "cámara con tubos extendidos",
    _KIND_CHAIN: "cadena de elementos",
    "Plenum chamber section": "Sección de la cámara plenum",
    "Inlet": "Entrada",
    "Outlet": "Salida",
    "Side branch": "Ramal lateral",
    "min |Z| at {frequency} Hz": "|Z| mínima en {frequency} Hz",
    _CHAIN_NOTE: (
        "Conductos dibujados a escala a partir de su longitud y área "
        "declaradas.\nUn ramal lateral solo declara una impedancia: se marca "
        "dónde se conecta, no se dibuja."
    ),
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


def _duct_walls(ax: Axes, x0: float, x1: float, d: float, wall: float) -> None:
    """The two wall plates of a duct run, lying just outside its bore."""
    for y in (-0.5 * d - wall, 0.5 * d):
        _material_rect(ax, x0, y, x1 - x0, wall, "plate", linewidth=0.5)


def _draw_duct(ax: Axes, x0: float, x1: float, d: float, **kwargs: Any) -> Any:
    """A straight duct run: bore centred on y = 0, walls just outside."""
    from matplotlib.patches import Rectangle

    wall = max(0.05 * d, 0.002)
    kwargs.setdefault("facecolor", "none")
    kwargs.setdefault("edgecolor", _C_EDGE)
    kwargs.setdefault("linewidth", 1.2)
    bore = Rectangle((x0, -0.5 * d), x1 - x0, d, **kwargs)
    ax.add_patch(bore)
    _duct_walls(ax, x0, x1, d, wall)
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
    _dim(
        ax,
        (0.0, -0.5 * d_c - 2.0 * off),
        (length, -0.5 * d_c - 2.0 * off),
        "L = " + _mm(length, language),
    )
    _dim(
        ax,
        (-stub, -0.5 * d_p),
        (-stub, 0.5 * d_p),
        _mm(d_p, language),
        offset=2.0 * off,
        tight=True,
    )
    x_dim = inlet_extension + 0.5 * (length - inlet_extension - outlet_extension)
    _dim(ax, (x_dim, -0.5 * d_c), (x_dim, 0.5 * d_c), _mm(d_c, language), offset=0.0)
    if inlet_extension > 0.0:
        _dim(
            ax,
            (0.0, 0.5 * d_p),
            (inlet_extension, 0.5 * d_p),
            _mm(inlet_extension, language),
            offset=-2.0 * off,
            tight=True,
        )
    if outlet_extension > 0.0:
        _dim(
            ax,
            (length - outlet_extension, 0.5 * d_p),
            (length, 0.5 * d_p),
            _mm(outlet_extension, language),
            offset=-2.0 * off,
            tight=True,
        )


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
        ax,
        -0.5 * cavity_side,
        y0 + branch_len,
        cavity_side,
        cavity_side,
        "cavity",
    )
    ax.text(
        0.0,
        y0 + branch_len + 0.5 * cavity_side,
        f"V = {format_number(cavity_volume * 1e3, language, decimals=1, trim=True)} L",
        fontsize=8,
        ha="center",
        va="center",
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
            raise ValueError("'cavity_volume' and 'neck_length' must be positive.")
        return (
            _duct_diameter(neck_area),
            neck_length,
            float(cavity_volume ** (1.0 / 3.0)),
        )
    if length is None or branch_area is None:
        raise ValueError("A quarter-wave drawing needs 'length' and 'branch_area'.")
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
        kind,
        neck_area=neck_area,
        neck_length=neck_length,
        cavity_volume=cavity_volume,
        length=length,
        branch_area=branch_area,
    )
    run = max(4.0 * d_d, 2.0 * d_b + 2.0 * d_d, 2.0 * cavity_side)
    _draw_duct(ax, -0.5 * run, 0.5 * run, d_d, **kwargs)
    # Branch mouth opens through the upper duct wall at x = 0.
    from matplotlib.patches import Rectangle

    y0 = 0.5 * d_d
    ax.add_patch(
        Rectangle(
            (-0.5 * d_b, y0),
            d_b,
            branch_len,
            facecolor="none",
            edgecolor=_C_EDGE,
            linewidth=1.2,
        )
    )
    off = 0.25 * d_d
    # A Helmholtz neck is a slot a couple of centimetres long and a few
    # millimetres wide: both of its dimensions are shorter than the text that
    # labels them. Their labels therefore go on the far side of the dimension
    # line, past the end of the extension lines, and the neck length is
    # lettered upright so that it fits between the duct wall and the cavity.
    neck = kind == _KIND_HELMHOLTZ
    if neck and cavity_volume is not None:
        _draw_hr_cavity(ax, y0, branch_len, cavity_side, cavity_volume, language)
        _dim(
            ax,
            (0.5 * d_b, y0),
            (0.5 * d_b, y0 + branch_len),
            _mm(branch_len, language),
            offset=-2.0 * off,
            tight=True,
            label_side=-1.0,
            label_upright=True,
        )
    else:
        # Closed end of the quarter-wave tube.
        ax.plot(
            [-0.5 * d_b, 0.5 * d_b],
            [y0 + branch_len, y0 + branch_len],
            color=_C_EDGE,
            linewidth=2.2,
        )
        _dim(
            ax,
            (0.5 * d_b, y0),
            (0.5 * d_b, y0 + branch_len),
            _mm(branch_len, language),
            offset=-2.0 * off,
        )
    _dim(
        ax,
        (-0.5 * d_b, y0),
        (0.5 * d_b, y0),
        _mm(d_b, language),
        offset=-1.5 * off,
        tight=True,
        label_side=-1.0 if neck else 1.0,
    )
    _dim(
        ax,
        (-0.5 * run, -0.5 * d_d),
        (-0.5 * run, 0.5 * d_d),
        _mm(d_d, language),
        offset=2.0 * off,
        tight=True,
    )


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
            "A chamber drawing needs 'length', 'chamber_area' and 'pipe_area'."
        )
    if length <= 0.0:
        raise ValueError(_LENGTH_POSITIVE)
    _duct_diameter(pipe_area)
    _duct_diameter(chamber_area)
    if chamber_area <= pipe_area:
        raise ValueError("'chamber_area' must exceed 'pipe_area'.")
    if inlet_extension + outlet_extension > length:
        raise ValueError(
            "'inlet_extension' + 'outlet_extension' must not exceed 'length'."
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
            raise ValueError("'cavity_volume' and 'neck_length' must be positive.")
        _duct_diameter(neck_area)
        return
    if length is None or branch_area is None:
        raise ValueError("A quarter-wave drawing needs 'length' and 'branch_area'.")
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
            f"Unknown silencer kind {kind!r}; expected one of {_SILENCER_KINDS}."
        )
    chamber = kind in (_KIND_EXPANSION, _KIND_EXTENDED)
    if chamber:
        _validate_chamber_geometry(
            length,
            chamber_area,
            pipe_area,
            inlet_extension,
            outlet_extension,
        )
    else:
        _validate_branch_geometry(
            kind,
            duct_area,
            neck_area,
            neck_length,
            cavity_volume,
            length,
            branch_area,
        )
    if ax is None:
        ax = _new_axes()
    if chamber:
        _draw_chamber(
            ax,
            length or 0.0,
            chamber_area or 0.0,
            pipe_area or 0.0,
            language,
            inlet_extension=inlet_extension,
            outlet_extension=outlet_extension,
        )
    else:
        _draw_branch_silencer(
            ax,
            kind,
            duct_area or 0.0,
            language,
            neck_area=neck_area,
            neck_length=neck_length,
            cavity_volume=cavity_volume,
            length=length,
            branch_area=branch_area,
        )
    _finish_geometry_axes(
        ax,
        _t(_CROSS_SECTION, language) + f" ({_t(kind, language)})",
    )
    return ax


def plot_silencer_result_geometry(
    result: ReactiveSilencerResult,
    ax: Axes | None = None,
    *,
    language: str = "en",
) -> Axes:
    """Silencer drawing for a result that retained a geometry of either kind.

    A named device retains the arguments of its constructor in ``geometry``; a
    result assembled element by element retains the ``chain`` that built it,
    which carries the geometry of each element separately.
    """
    if result.chain is not None:
        return plot_silencer_chain_geometry(result.chain, ax=ax, language=language)
    if result.geometry is None:
        raise ValueError(
            "This result does not retain its geometry; call "
            "plot_silencer_geometry(kind, ...) with the original arguments."
        )
    return plot_silencer_geometry(
        result.kind,
        ax=ax,
        language=language,
        **dict(result.geometry),
    )


# ---------------------------------------------------------------------------
# Hand-built element chains.
# ---------------------------------------------------------------------------
#: A drawn duct run: where it starts, where it ends, what its area is.
_Segment = tuple[float, float, float]
#: A drawn branch point: its station, its label and its shorting frequency.
_Branch = tuple[float, str | None, float | None]


def _chain_layout(
    elements: Sequence[SilencerChainElement],
) -> tuple[list[_Segment], list[_Branch], float]:
    """Resolve the chain into drawn duct runs, branch points and a length.

    Stations accumulate along the run in the order the elements were added.
    A duct of zero length is dropped from both lists: its four-pole matrix is
    the identity, so it is nothing acoustically and has no extent to draw.
    """
    segments: list[_Segment] = []
    branches: list[_Branch] = []
    x = 0.0
    for element in elements:
        if element.area is None:
            branches.append((x, element.label, element.shorting_frequency))
            continue
        length = element.length or 0.0
        if length > 0.0:
            segments.append((x, x + length, element.area))
            x += length
    return segments, branches, x


def _equal_area_runs(segments: Sequence[_Segment]) -> list[_Segment]:
    """Merge consecutive duct runs of equal area into one drawn run."""
    runs: list[_Segment] = []
    for x0, x1, area in segments:
        if runs and runs[-1][2] == area:
            runs[-1] = (runs[-1][0], x1, area)
        else:
            runs.append((x0, x1, area))
    return runs


def _diameter_at(segments: Sequence[_Segment], x: float) -> float:
    """The drawn diameter of the run a branch joins at station ``x``.

    A branch landing on the joint between two runs is drawn on the wider of
    the two, which is the shell its mouth would open through.
    """
    touching = [_duct_diameter(area) for x0, x1, area in segments if x0 <= x <= x1]
    return max(touching) if touching else _duct_diameter(segments[0][2])


def _draw_chain_ducts(ax: Axes, segments: Sequence[_Segment], wall: float) -> None:
    """The bores, the wall plates and the annular faces of the area steps.

    The two ends of the whole run are left open: a chain declares its elements
    and not what it is connected between, so nothing is drawn to cap it.
    """
    diameters = [_duct_diameter(area) for _, _, area in segments]
    for (x0, x1, _area), d in zip(segments, diameters, strict=True):
        _duct_walls(ax, x0, x1, d, wall)
        for y in (-0.5 * d, 0.5 * d):
            ax.plot([x0, x1], [y, y], color=_C_EDGE, linewidth=1.2)
    for index in range(1, len(segments)):
        d_before, d_after = diameters[index - 1], diameters[index]
        if d_before == d_after:
            continue
        x = segments[index][0]
        near, far = sorted((d_before, d_after))
        for sign in (-1.0, 1.0):
            ax.plot(
                [x, x],
                [sign * 0.5 * near, sign * (0.5 * far + wall)],
                color=_C_EDGE,
                linewidth=1.4,
            )


def _branch_label(branch: _Branch, index: int, total: int, language: str) -> str:
    """The callout text of a branch point: what it is, and where it bites."""
    from ..._i18n import format_number

    _station, label, frequency = branch
    text = label or _t("Side branch", language)
    if not label and total > 1:
        text = f"{text} {index + 1}"
    if frequency is not None:
        text += "\n" + _t("min |Z| at {frequency} Hz", language).format(
            frequency=format_number(frequency, language, decimals=1, trim=True)
        )
    return text


def _draw_branch_points(
    ax: Axes,
    segments: Sequence[_Segment],
    branches: Sequence[_Branch],
    wall: float,
    d_max: float,
    language: str,
) -> None:
    """Mark each shunt where it joins the run, with a leader to its callout.

    A leader and a node dot are annotation, not features: they carry no
    dimension line and no measurement, which is the whole of what may honestly
    be drawn for an element that declares an impedance and no shape.
    """
    run = 0.30 * d_max
    for index, branch in enumerate(branches):
        x = branch[0]
        y0 = 0.5 * _diameter_at(segments, x) + wall
        rise = (0.55 if index % 2 == 0 else 1.00) * d_max
        ax.plot(
            [x, x + run, x + run + 0.10 * d_max],
            [y0, y0 + rise, y0 + rise],
            color=_C_SECONDARY,
            linewidth=0.9,
            zorder=4,
        )
        ax.plot(
            [x],
            [y0],
            marker="o",
            markersize=5.0,
            markerfacecolor=_C_SECONDARY,
            markeredgecolor=_C_EDGE,
            markeredgewidth=0.6,
            zorder=6,
        )
        ax.text(
            x + run + 0.14 * d_max,
            y0 + rise,
            _branch_label(branch, index, len(branches), language),
            fontsize=8,
            ha="left",
            va="center",
        )


def _dimension_chain(
    ax: Axes,
    segments: Sequence[_Segment],
    total: float,
    d_max: float,
    wall: float,
    language: str,
) -> None:
    """Dimension every run's length, each distinct bore, and the whole.

    The lengths are lettered on one row below the widest section, each with
    witness lines back to its own run, and the overall length on a second row
    under them. A bore is dimensioned on the first run that has it, so a chain
    returning to a diameter it already showed does not letter the same number
    twice.
    """
    off = 0.18 * d_max
    y_lengths = -0.5 * d_max - 2.0 * off
    for x0, x1, area in segments:
        y_wall = -0.5 * _duct_diameter(area) - wall
        _dim(
            ax,
            (x0, y_wall),
            (x1, y_wall),
            _mm(x1 - x0, language),
            offset=y_lengths - y_wall,
            tight=(x1 - x0) < 0.25 * total,
        )
    # A single segment is its own overall run, so the second line would repeat
    # the first; the overall run is dimensioned only when there is more than one.
    if len(segments) > 1:
        _dim(
            ax,
            (0.0, y_lengths),
            (total, y_lengths),
            "L = " + _mm(total, language),
            offset=-1.6 * off,
        )
    drawn: set[float] = set()
    for x0, x1, area in _equal_area_runs(segments):
        if area in drawn:
            continue
        drawn.add(area)
        d = _duct_diameter(area)
        x_mid = 0.5 * (x0 + x1)
        _dim(
            ax,
            (x_mid, -0.5 * d),
            (x_mid, 0.5 * d),
            _mm(d, language),
            offset=0.0,
        )


def _note_under(ax: Axes, width: float, note: str) -> None:
    """Letter a two-line note under everything drawn so far.

    Text does not enlarge the data limits, and the drawing is autoscaled from
    them, so the band the note will occupy is claimed first. Both the drop and
    the band are fractions of what has been drawn, which keeps the note in
    proportion at any scale the chain happens to have.
    """
    bounds = ax.dataLim
    span = float(bounds.height)
    y_note = float(bounds.y0) - 0.06 * span
    ax.update_datalim(((0.0, y_note - 0.16 * span), (width, y_note)))
    ax.text(
        0.5 * width,
        y_note,
        note,
        fontsize=8,
        ha="center",
        va="top",
    )


def plot_silencer_chain_geometry(
    chain: SilencerChain,
    ax: Axes | None = None,
    *,
    language: str = "en",
) -> Axes:
    """Draw a hand-built element chain: ducts to scale, branch points marked.

    The companion of :func:`plot_silencer_geometry` for a layout that is not
    one of the named devices. A
    :class:`~phonometry.noise_control.silencers.SilencerChain` records what
    each of its elements was given, and this draws exactly that and no more:
    every duct at its declared length and equivalent circular diameter
    (``d = 2 sqrt(S / pi)``), the annular face of every area step between
    them, and each length, bore and the overall run dimensioned.

    A shunt element is handed an acoustic impedance, which fixes no length, no
    area and no volume. It is therefore not drawn as a stub of any size: it is
    marked with a node and a leader at the station where it joins the run,
    lettered with its label and with the frequency at which its impedance is
    least, and the drawing says so underneath. The two ends of the run are
    left open for the same reason, a chain being a list of elements and not a
    statement about the pipes it sits between.

    :param chain: The chain to draw.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :return: The axes.
    :raises ValueError: If the chain holds no duct of positive length, and so
        declares no geometry and no scale to draw it at.
    """
    _check_language(language)
    segments, branches, total = _chain_layout(chain.elements)
    if not segments:
        raise ValueError(
            "This chain has no duct of positive length, so it declares no "
            "geometry to draw: a shunt element holds an impedance and no "
            "shape. Add the duct runs its branches sit on."
        )
    if ax is None:
        ax = _new_axes()
    d_max = max(_duct_diameter(area) for _, _, area in segments)
    wall = max(0.04 * d_max, 0.003)
    _draw_chain_ducts(ax, segments, wall)
    _draw_branch_points(ax, segments, branches, wall, d_max, language)
    _dimension_chain(ax, segments, total, d_max, wall, language)
    if branches:
        _note_under(ax, total, _t(_CHAIN_NOTE, language))
    _finish_geometry_axes(
        ax,
        _t(_CROSS_SECTION, language) + f" ({_t(_KIND_CHAIN, language)})",
    )
    return ax


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
    for x_pair, y_pair in walls:
        ax.plot(x_pair, y_pair, color=colour, linewidth=lw, **kwargs)
    stub = 0.35 * r
    wall_t = max(0.05 * duct, 0.002)
    ax.add_patch(
        Rectangle(
            (-stub, y_in - 0.5 * duct),
            stub,
            duct,
            facecolor="none",
            edgecolor=_C_EDGE,
            linewidth=1.2,
        )
    )
    for y_wall in (y_in - 0.5 * duct - wall_t, y_in + 0.5 * duct):
        _material_rect(ax, -stub, y_wall, stub, wall_t, "plate", linewidth=0.5)
    ax.text(
        -0.5 * stub,
        y_in + 0.85 * duct,
        _t("Inlet", language),
        fontsize=8,
        ha="center",
        va="bottom",
    )
    # Outlet mouth on the right wall.
    ax.plot(
        [box_w, box_w],
        [y_out - 0.5 * mouth, y_out + 0.5 * mouth],
        color=_C_PRIMARY,
        linewidth=3.0,
    )
    ax.text(
        box_w * 1.02,
        y_out + 0.7 * mouth,
        _t("Outlet", language),
        fontsize=8,
        ha="left",
        va="bottom",
    )
    # Line of sight, labelled along its own slope.
    ax.plot(
        [0.0, box_w], [y_in, y_out], linestyle="--", linewidth=1.2, color=_C_SECONDARY
    )
    slope = float(np.degrees(angle))
    ax.text(
        0.5 * box_w,
        0.5 * (y_in + y_out) + 0.03 * r,
        "r = " + _metres(r, language),
        fontsize=8,
        ha="center",
        va="bottom",
        rotation=slope,
        rotation_mode="anchor",
        color=_C_SECONDARY,
    )
    from ..._i18n import format_number

    ax.text(
        0.5 * box_w,
        -0.08 * r,
        "S_w = "
        + format_number(wall_area, language, decimals=1, trim=True)
        + " m$^2$, S_out = "
        + format_number(exit_area, language, decimals=2, trim=True)
        + " m$^2$",
        fontsize=8,
        ha="center",
        va="top",
    )
    _finish_geometry_axes(ax, _t("Plenum chamber section", language))
    return ax
