#  Copyright (c) 2026. Jose M. Requena-Plens
"""Plot renderers for the noise_control domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_SECONDARY_LIGHT,
    _C_TERTIARY,
    _new_axes,
    format_frequency_axis,
    theme_fill,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..noise_control.duct_modes import DuctModeResult
    from ..noise_control.duct_path import DuctPathResult
    from ..noise_control.enclosures import EnclosureResult
    from ..noise_control.hvac import HvacSpectrumResult
    from ..noise_control.room_to_room import RoomToRoomResult
    from ..noise_control.silencers import ReactiveSilencerResult

_FREQ_LABEL = "Frequency [Hz]"
_LEVEL_LABEL = "Level [dB]"

#: Spanish translations of the fixed labels/titles/legends rendered by the
#: noise-control ``.plot()`` renderers, keyed by their verbatim English
#: text. ``_t`` returns the English key unchanged for any language other
#: than ``"es"``, so the English output is byte-for-byte identical to the
#: pre-i18n renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Band": "Banda",
    "Transmission loss": "Pérdida por transmisión",
    "Insertion loss": "Pérdida por inserción",
    "Resonance": "Resonancia",
    "Loss [dB]": "Pérdida [dB]",
    "Reactive silencer": "Silenciador reactivo",
    "Sound power level [dB re 1 pW]": "Nivel de potencia acústica [dB re 1 pW]",
    "Attenuation [dB]": "Atenuación [dB]",
    "Panel R": "R del panel",
    "Interior correction C": "Corrección interior C",
    "Insertion loss (R - C)": "Pérdida por inserción (R - C)",
    _LEVEL_LABEL: "Nivel [dB]",
    "Machine enclosure insertion loss": "Pérdida por inserción de encapsulado de máquina",
    "Sound power level [dB]": "Nivel de potencia acústica [dB]",
    "Received level": "Nivel recibido",
    "Source": "Fuente",
    "Duct-borne noise path": "Trayecto de ruido por conductos",
    "Mode order (p, q)": "Orden del modo (p, q)",
    "Cut-on frequency [Hz]": "Frecuencia de corte [Hz]",
    "No flow": "Sin flujo",
    "Plane waves only": "Solo ondas planas",
    "Duct higher-order-mode cut-on": "Corte de modos superiores del conducto",
    "Source room": "Recinto emisor",
    "Receiving room": "Recinto receptor",
    "Noise reduction": "Reducción de ruido",
    "Loss and noise reduction [dB]": "Pérdida y reducción de ruido [dB]",
    "Room-to-room transmission": "Transmisión entre recintos",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_reactive_silencer(
    result: ReactiveSilencerResult, ax: Axes | None = None,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Transmission (and insertion) loss of a reactive silencer over frequency.

    :param result: A
        :class:`~phonometry.noise_control.silencers.ReactiveSilencerResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the transmission-loss ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequencies, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Transmission loss", language))
    kwargs.setdefault("lw", 1.8)
    ax.plot(f, np.asarray(result.transmission_loss), **kwargs)
    if result.insertion_loss is not None:
        ax.plot(f, np.asarray(result.insertion_loss), color=_C_SECONDARY,
                lw=1.4, ls="--", label=_t("Insertion loss", language))
    if result.resonances is not None:
        labeled = False
        for fr in np.atleast_1d(np.asarray(result.resonances)):
            if f.min() <= fr <= f.max():
                ax.axvline(fr, color=_C_TERTIARY, ls=":", lw=1.0,
                           label=_t("Resonance", language) if not labeled else None)
                labeled = True
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t("Loss [dB]", language))
    ax.set_title(f"{_t('Reactive silencer', language)}: {result.kind}")
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_hvac_spectrum(
    result: HvacSpectrumResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Per-frequency HVAC attenuation or regenerated sound power level.

    :param result: A :class:`~phonometry.noise_control.hvac.HvacSpectrumResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequencies, dtype=np.float64)
    is_power = result.quantity == "sound_power_level"
    kwargs.setdefault("color", _C_SECONDARY if is_power else _C_PRIMARY)
    kwargs.setdefault("label", result.label)
    kwargs.setdefault("lw", 1.8)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 3)
    ax.plot(f, np.asarray(result.values), **kwargs)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(
        _t("Sound power level [dB re 1 pW]", language) if is_power
        else _t("Attenuation [dB]", language)
    )
    ax.set_title(result.label)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


#: Colours cycled over the intermediate spectra of a duct-path cascade. They are
#: mid-tone on purpose: a sequential colormap puts its ends at near-black and
#: near-yellow, which one theme or the other loses against its background, and
#: these lines are supporting detail behind the received level and the criterion
#: curve, so they have to stay legible in both.
_CASCADE_COLOURS: tuple[str, ...] = (
    _C_SECONDARY, _C_TERTIARY, _C_QUATERNARY, _C_MUTED, "#17becf",
    _C_SECONDARY_LIGHT,
)

#: Line styles the cascade steps through each time the colour cycle wraps, so a
#: long path stays readable past six elements.
_CASCADE_STYLES: tuple[str, ...] = ("-", "--", ":")

#: Longest element name carried into the cascade legend before it is elided; the
#: full description belongs in the table, not in a corner of the chart.
_LEGEND_CHARS = 30


def _cascade_series(result: DuctPathResult) -> list[tuple[str, np.ndarray]]:
    """The labelled spectra of a duct path, from the source to the receiver."""
    if result.contributions:
        return [(name, np.asarray(values)) for name, values in result.contributions]
    series: list[tuple[str, np.ndarray]] = [
        (result.source_label, np.asarray(result.source_level))
    ]
    series += [(stage.label, np.asarray(stage.level)) for stage in result.stages]
    return series


def plot_duct_path(
    result: DuctPathResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Cascade of a duct-borne noise path against the room-criterion curve.

    One faint line per element shows where the octave-band spectrum stands
    after that element, from the source at the top to the received level, which
    is drawn thick; the design criterion curve, when the path declares one, is
    overlaid as a dashed reference.

    :param result: A
        :class:`~phonometry.noise_control.duct_path.DuctPathResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the received-level ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequencies, dtype=np.float64)
    series = _cascade_series(result)
    for position, (name, values) in enumerate(series):
        colour = _CASCADE_COLOURS[position % len(_CASCADE_COLOURS)]
        style = _CASCADE_STYLES[
            (position // len(_CASCADE_COLOURS)) % len(_CASCADE_STYLES)
        ]
        label = name if len(name) <= _LEGEND_CHARS else name[:_LEGEND_CHARS - 1] + "\u2026"
        ax.semilogx(f, values, ls=style, color=colour, lw=1.1, alpha=0.85,
                    label=label if len(series) <= 10 else None)
    curve = result.criterion_curve
    if curve is not None:
        ax.semilogx(f, curve, ls="--", color=_C_REFERENCE, lw=1.5,
                    label=f"{result.criterion} {result.target:g}")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Received level", language))
    kwargs.setdefault("lw", 2.4)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 4)
    ax.semilogx(f, np.asarray(result.received_level), **kwargs)
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    ax.set_ylabel(_t(_LEVEL_LABEL, language))
    ax.set_title(f"{_t('Duct-borne noise path', language)}: {result.label}")
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax)
    ax.legend(
        loc="upper right", fontsize="xx-small",
        ncol=2 if len(series) > 4 else 1, framealpha=0.85,
    )
    localize_axes(ax, language)
    return ax


def plot_room_to_room(
    result: RoomToRoomResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Source-room level, received level and criterion curve of a partition.

    The two reverberant spectra bracket the design criterion curve on the left
    axis; the transmission loss of the partition and the noise reduction it
    actually delivers share a twin axis on the right, which is where the point
    of Equation (4.101) shows: the two are not the same number.

    :param result: A
        :class:`~phonometry.noise_control.room_to_room.RoomToRoomResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the received-level ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    f = np.asarray(result.frequencies, dtype=np.float64)
    ax.semilogx(f, np.asarray(result.source_level), color=_C_SECONDARY, lw=1.6,
                ls="--", marker="s", ms=4, label=_t("Source room", language))
    curve = result.criterion_curve
    if curve is not None:
        ax.semilogx(f, curve, ls=":", color=_C_REFERENCE, lw=1.5,
                    label=f"{result.criterion} {result.target:g}")
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Receiving room", language))
    kwargs.setdefault("lw", 2.4)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 4)
    ax.semilogx(f, np.asarray(result.received_level), **kwargs)
    ax.set_ylabel(_t(_LEVEL_LABEL, language))
    ax.set_title(f"{_t('Room-to-room transmission', language)}: {result.label}")
    ax.grid(True, which="both", alpha=0.3)

    twin = ax.twinx()
    twin.plot(f, np.asarray(result.transmission_loss), color=_C_MUTED, lw=1.2,
              ls="--", label=_t("Transmission loss", language))
    twin.plot(f, np.asarray(result.noise_reduction), color=_C_TERTIARY,
              lw=1.5, ls="-.", marker="^", ms=3,
              label=_t("Noise reduction", language))
    twin.set_ylabel(_t("Loss and noise reduction [dB]", language),
                    color=_C_TERTIARY)
    twin.tick_params(axis="y", labelcolor=_C_TERTIARY)
    twin.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra_handles, labels + extra_labels, loc="best",
              fontsize="small", framealpha=0.85, ncol=2)
    # The twin axis resets the shared log-frequency formatting, so the ticks are
    # set last and on both axes.
    ax.set_xlabel(_t(_FREQ_LABEL, language))
    format_frequency_axis(ax)
    format_frequency_axis(twin)
    localize_axes(ax, language)
    localize_axes(twin, language)
    return ax


def plot_duct_modes(
    result: DuctModeResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Cut-on frequency of each higher-order duct mode, with and without flow.

    The still-air cut-on of every mode is drawn beside the value the mean flow
    shifts it to, and the band below the first cut-on -- where the duct carries
    plane waves only, and where the plane-wave methods of the library are valid
    -- is shaded.

    :param result: A
        :class:`~phonometry.noise_control.duct_modes.DuctModeResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the with-flow ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    x = np.arange(len(result.modes), dtype=np.float64)
    ax.axhspan(0.0, result.plane_wave_limit, color=theme_fill(_C_PRIMARY, ax),
               zorder=0, label=_t("Plane waves only", language))
    ax.plot(x, np.asarray(result.cut_on_no_flow), ls="--", color=_C_MUTED,
            marker="s", ms=4, lw=1.3, label=_t("No flow", language))
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", f"M = {result.mach:.3f}")
    kwargs.setdefault("lw", 1.8)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 5)
    ax.plot(x, np.asarray(result.cut_on), **kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels([f"({p}, {q})" for p, q in result.modes])
    ax.set_xlabel(_t("Mode order (p, q)", language))
    ax.set_ylabel(_t("Cut-on frequency [Hz]", language))
    ax.set_title(f"{_t('Duct higher-order-mode cut-on', language)}: {result.label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax


def plot_enclosure(
    result: EnclosureResult, ax: Axes | None = None, language: str = "en",
    **kwargs: Any
) -> Axes:
    """Panel R, interior correction C and net insertion loss of an enclosure.

    :param result: An
        :class:`~phonometry.noise_control.enclosures.EnclosureResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the insertion-loss ``Axes.plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    n = np.asarray(result.insertion_loss).size
    if result.frequencies is not None:
        x = np.asarray(result.frequencies, dtype=np.float64)
        continuous = True
    else:
        x = np.arange(n, dtype=np.float64)
        continuous = False
    ax.plot(x, np.asarray(result.panel_transmission_loss), color=_C_REFERENCE,
            lw=1.3, ls="--", marker="s", ms=3, label=_t("Panel R", language))
    ax.plot(x, np.asarray(result.correction), color=_C_TERTIARY, lw=1.3,
            ls=":", marker="^", ms=3, label=_t("Interior correction C", language))
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("label", _t("Insertion loss (R - C)", language))
    kwargs.setdefault("lw", 1.9)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("ms", 3)
    ax.plot(x, np.asarray(result.insertion_loss), **kwargs)
    ax.set_ylabel(_t(_LEVEL_LABEL, language))
    ax.set_title(_t("Machine enclosure insertion loss", language))
    ax.grid(True, which="both", alpha=0.3)
    if continuous:
        ax.set_xlabel(_t(_FREQ_LABEL, language))
        format_frequency_axis(ax)
    else:
        ax.set_xlabel(_t("Band", language))
        ax.set_xticks(x)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax
