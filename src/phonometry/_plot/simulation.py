#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the simulation domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import _C_MUTED, _C_REFERENCE, _field_cmap, _new_axes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..simulation.elastic_fdtd import ElasticFDTDResult
    from ..simulation.fdtd import FDTDResult

_TIME_LABEL = "Time [ms]"
_PRESSURE_LABEL = "Pressure [Pa]"
_VELOCITY_LABEL = "Particle velocity [m/s]"

#: Spanish translations of the fixed strings rendered by the simulation
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Time [ms]": "Tiempo [ms]",
    "Pressure [Pa]": "Presión [Pa]",
    "FDTD pressure field at $t$ = {t_txt} ms":
        "Campo de presión FDTD en $t$ = {t_txt} ms",
    "FDTD probe pressure": "Presión en las sondas FDTD",
    "probe": "sonda",
    "Particle velocity [m/s]": "Velocidad de partícula [m/s]",
    "Signal [Pa, m/s]": "Señal [Pa, m/s]",
    "Elastic FDTD probe signals": "Señales en las sondas del FDTD elástico",
    "Elastic FDTD pressure field at $t$ = {t_txt} ms":
        "Campo de presión del FDTD elástico en $t$ = {t_txt} ms",
    "Elastic FDTD $v_x$ field at $t$ = {t_txt} ms":
        "Campo de $v_x$ del FDTD elástico en $t$ = {t_txt} ms",
    "Elastic FDTD $v_y$ field at $t$ = {t_txt} ms":
        "Campo de $v_y$ del FDTD elástico en $t$ = {t_txt} ms",
}

#: Colorbar label of each elastic snapshot field.
_ELASTIC_FIELD_LABELS = {
    "p": _PRESSURE_LABEL,
    "vx": _VELOCITY_LABEL,
    "vy": _VELOCITY_LABEL,
}

#: Snapshot title of each elastic snapshot field. The velocity component is
#: set as mathematics (``$v_x$``, never ``$v_{x}$``): ``_t`` runs ``format``
#: over the whole template, so a brace group other than ``{t_txt}`` would
#: raise :class:`KeyError` at draw time.
_ELASTIC_FIELD_TITLES = {
    "p": "Elastic FDTD pressure field at $t$ = {t_txt} ms",
    "vx": "Elastic FDTD $v_x$ field at $t$ = {t_txt} ms",
    "vy": "Elastic FDTD $v_y$ field at $t$ = {t_txt} ms",
}

#: Legend symbol of each recorded probe field, for the elastic probe legend
#: that prefixes the probe label with the field. A symbol identifier is never
#: translated, so this table is language-independent.
_FIELD_SYMBOLS = {
    "p": "$p$",
    "vx": "$v_x$",
    "vy": "$v_y$",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def _render_probe_lines(
    ax: Axes, t_ms: NDArray[np.float64],
    traces: Sequence[NDArray[np.float64]], labels: Sequence[str],
    *, xlabel: str, ylabel: str, title: str, language: str,
    **kwargs: Any,
) -> Axes:
    """Draw labelled probe time histories with the shared styling."""
    from .._i18n import localize_axes

    for trace, label in zip(traces, labels, strict=True):
        trace_kwargs = dict(kwargs)
        trace_kwargs.setdefault("label", label)
        ax.plot(t_ms, trace, **trace_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if labels:
        ax.legend(loc="upper right", fontsize="small")
    localize_axes(ax, language)
    return ax


def _probe_labels(
    positions: NDArray[np.float64], language: str,
) -> list[str]:
    """One ``probe (x, y) m`` label per probe position."""
    from .._i18n import format_number

    label = _t("probe", language)
    out = []
    for k in range(positions.shape[0]):
        x, y = positions[k]
        px = format_number(x, language, decimals=2)
        py = format_number(y, language, decimals=2)
        out.append(f"{label} ({px}, {py}) m")
    return out


def _render_snapshot(
    ax: Axes, field: NDArray[np.float64],
    *, size: tuple[float, float], shape: tuple[int, int],
    obstacle_mask: NDArray[np.bool_] | None,
    source_positions: Sequence[tuple[float, float]],
    probe_positions: NDArray[np.float64],
    colorbar_label: str, title: str, language: str,
    **kwargs: Any,
) -> Axes:
    """Render one field raster with the geometry overlaid (shared core)."""
    from .._i18n import localize_axes

    lx, ly = size
    extent = (0.0, lx, ly, 0.0)
    vmax = float(np.abs(field).max()) or 1.0
    img = ax.imshow(
        field,
        **{
            "cmap": _field_cmap(ax),
            "vmin": -vmax,
            "vmax": vmax,
            "origin": "upper",
            "extent": extent,
            "interpolation": "nearest",
            **kwargs,
        },
    )
    if obstacle_mask is not None:
        overlay = np.ma.masked_where(~obstacle_mask, np.ones(shape))
        ax.imshow(overlay, cmap="gray", vmin=0.0, vmax=2.0, origin="upper",
                  extent=extent, interpolation="nearest")
    for x, y in source_positions:
        ax.plot(x, y, marker="*", markersize=10, color=_C_REFERENCE,
                linestyle="none")
    for k in range(probe_positions.shape[0]):
        x, y = probe_positions[k]
        ax.plot(x, y, marker="o", markersize=5, color=_C_MUTED,
                markeredgecolor="black", linestyle="none")
    ax.figure.colorbar(img, ax=ax, label=colorbar_label)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title(title)
    localize_axes(ax, language)
    return ax


def _snapshot_time_text(
    snapshot_times: NDArray[np.float64], frame: int, language: str,
) -> str:
    """The localised snapshot time in milliseconds."""
    from .._i18n import format_number

    return format_number(float(snapshot_times[frame]) * 1000.0, language,
                         decimals=2)


def plot_fdtd_probes(
    result: FDTDResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Pressure time history at each probe of an FDTD run.

    :param result: A :class:`~phonometry.simulation.fdtd.FDTDResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``Axes.plot``.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    t_ms = np.asarray(result.times, dtype=np.float64) * 1000.0
    return _render_probe_lines(
        ax, t_ms,
        [result.pressures[k] for k in range(result.pressures.shape[0])],
        _probe_labels(result.probe_positions, language),
        xlabel=_t(_TIME_LABEL, language),
        ylabel=_t(_PRESSURE_LABEL, language),
        title=_t("FDTD probe pressure", language),
        language=language, **kwargs,
    )


def plot_fdtd_snapshot(
    result: FDTDResult, ax: Axes | None = None, *, frame: int = -1,
    language: str = "en", **kwargs: Any
) -> Axes:
    """One recorded pressure-field snapshot with the geometry overlaid.

    The field is rendered as a single raster image (``imshow``, symmetric
    diverging scale); rigid obstacle cells are drawn in grey and the source
    and probe cells are marked.  The default colormap follows the axes
    background so zero pressure blends into the page -- ``RdBu_r`` on a
    light background, the black-centred ``phonometry_field_dark`` on a dark
    one -- and ``cmap`` overrides it.

    :param result: A :class:`~phonometry.simulation.fdtd.FDTDResult` with
        recorded snapshots.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param frame: Snapshot index (default: the last recorded frame).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    :raises ValueError: If the result holds no snapshots.
    """
    if result.snapshots is None or result.snapshot_times is None:
        raise ValueError("the result holds no snapshots; rerun the "
                         "simulation with snapshot_every set")
    ax = ax if ax is not None else _new_axes()
    t_txt = _snapshot_time_text(result.snapshot_times, frame, language)
    return _render_snapshot(
        ax, np.asarray(result.snapshots[frame], dtype=np.float64),
        size=result.size, shape=result.shape,
        obstacle_mask=result.obstacle_mask,
        source_positions=[((s.ix + 0.5) * result.dx, (s.iy + 0.5) * result.dx)
                          for s in result.sources],
        probe_positions=result.probe_positions,
        colorbar_label=_t(_PRESSURE_LABEL, language),
        title=_t("FDTD pressure field at $t$ = {t_txt} ms", language,
                 t_txt=t_txt),
        language=language, **kwargs,
    )


def _elastic_source_position(
    source: Any, dx: float,
) -> tuple[float, float]:
    """Physical position of an elastic source marker."""
    from ..simulation.elastic_fdtd import ForceSource

    if isinstance(source, ForceSource):
        if source.direction == "x":
            return (source.ix + 1.0) * dx, (source.iy + 0.5) * dx
        return (source.ix + 0.5) * dx, (source.iy + 1.0) * dx
    return (source.ix + 0.5) * dx, (source.iy + 0.5) * dx


def plot_elastic_probes(
    result: ElasticFDTDResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Recorded time history of every probe/field pair of an elastic run.

    One line per probe and recorded field; the field name prefixes the
    probe label when more than one field was recorded.

    :param result: An
        :class:`~phonometry.simulation.elastic_fdtd.ElasticFDTDResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``Axes.plot``.
    :return: The axes.
    """
    ax = ax if ax is not None else _new_axes()
    t_ms = np.asarray(result.times, dtype=np.float64) * 1000.0
    base = _probe_labels(result.probe_positions, language)
    many = len(result.probe_fields) > 1
    traces = []
    labels = []
    for k in range(result.signals.shape[0]):
        for j, field in enumerate(result.probe_fields):
            traces.append(result.signals[k, j])
            symbol = _FIELD_SYMBOLS.get(field, field)
            labels.append(f"{symbol}, {base[k]}" if many else base[k])
    if set(result.probe_fields) <= {"vx", "vy"}:
        ylabel = _VELOCITY_LABEL
    elif result.probe_fields == ("p",):
        ylabel = _PRESSURE_LABEL
    else:
        ylabel = "Signal [Pa, m/s]"
    return _render_probe_lines(
        ax, t_ms, traces, labels,
        xlabel=_t(_TIME_LABEL, language),
        ylabel=_t(ylabel, language),
        title=_t("Elastic FDTD probe signals", language),
        language=language, **kwargs,
    )


def plot_elastic_snapshot(
    result: ElasticFDTDResult, ax: Axes | None = None, *, frame: int = -1,
    language: str = "en", **kwargs: Any
) -> Axes:
    """One recorded elastic field snapshot with the geometry overlaid.

    Renders the recorded ``snapshot_field`` (synthetic pressure or one
    velocity component, at cell centres) with the same raster styling as
    the acoustic snapshot renderer, including its background-aware default
    colormap (``RdBu_r`` on a light axes background, the black-centred
    ``phonometry_field_dark`` on a dark one; ``cmap`` overrides it).

    :param result: An
        :class:`~phonometry.simulation.elastic_fdtd.ElasticFDTDResult`
        with recorded snapshots.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param frame: Snapshot index (default: the last recorded frame).
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``imshow``.
    :return: The axes.
    :raises ValueError: If the result holds no snapshots.
    """
    if result.snapshots is None or result.snapshot_times is None:
        raise ValueError("the result holds no snapshots; rerun the "
                         "simulation with snapshot_every set")
    ax = ax if ax is not None else _new_axes()
    t_txt = _snapshot_time_text(result.snapshot_times, frame, language)
    return _render_snapshot(
        ax, np.asarray(result.snapshots[frame], dtype=np.float64),
        size=result.size, shape=result.shape,
        obstacle_mask=result.obstacle_mask,
        source_positions=[_elastic_source_position(s, result.dx)
                          for s in result.sources],
        probe_positions=result.probe_positions,
        colorbar_label=_t(_ELASTIC_FIELD_LABELS[result.snapshot_field],
                          language),
        title=_t(_ELASTIC_FIELD_TITLES[result.snapshot_field], language,
                 t_txt=t_txt),
        language=language, **kwargs,
    )
