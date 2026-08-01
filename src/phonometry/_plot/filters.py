#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the filters domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..filters.compliance import FilterComplianceResult
    from ..filters.equalizer import EQResponseResult

from .common import (
    _C_MUTED,
    _C_PRIMARY,
    _C_REFERENCE,
    _C_TERTIARY,
    _LEGEND_UPPER_RIGHT,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
    theme_fill,
)

#: Spanish translations of the fixed strings rendered by the filters
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Magnitude [dB]": "Magnitud [dB]",
    "Phase [deg]": "Fase [grados]",
    "Class {cls} pass corridor": "Corredor de aceptación clase {cls}",
    r"Measured $\Delta A$": r"$\Delta A$ medida",
    "Out of tolerance": "Fuera de tolerancia",
    r"Normalised frequency $f\,/\,f_m$": r"Frecuencia normalizada $f\,/\,f_m$",
    "Relative attenuation [dB]": "Atenuación relativa [dB]",
    "IEC 61260-1 class {cls} mask — $f_m$ = {fm} Hz":
        "Máscara clase {cls} IEC 61260-1 — $f_m$ = {fm} Hz",
    "lowpass": "paso bajo",
    "highpass": "paso alto",
    "magnitude": "magnitud",
    "Cascade": "Cascada",
    "Parametric EQ response (Audio EQ Cookbook)":
        "Respuesta del EQ paramétrico (Audio EQ Cookbook)",
    "peaking": "campana",
    "lowshelf": "shelf de graves",
    "highshelf": "shelf de agudos",
    "bandpass": "paso banda",
    "bandpass_skirt": "paso banda (faldón)",
    "notch": "muesca",
    "allpass": "paso todo",
}


def _t(text: str, language: str = "en", **fmt: Any) -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    s = _STRINGS.get(text, text) if language == "es" else text
    return s.format(**fmt) if fmt else s


def _worst_band_index(result: FilterComplianceResult) -> int:
    """Index of the band with the smallest margin to the reference class."""
    key = f"margin_class{result.reference_class()}_db"
    margins = [float(band[key]) for band in result.bands]
    return int(np.argmin(margins))


def plot_filter_class(
    result: FilterComplianceResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any
) -> Axes:
    """Measured relative attenuation of the binding band over its class corridor.

    Selects the worst-margin band (the one whose margin to the achieved class
    is smallest, or, when the bank meets no class, the band that misses the
    loosest class by the most) and draws its measured relative attenuation
    ``ΔA`` against the normalized frequency ``f / f_m`` on a logarithmic axis.
    The acceptance corridor of the reference class is shaded green (between the
    lower and upper limits of Table 1) and any part of the measured curve that
    leaves the corridor is marked red, following the MATLAB
    ``octaveFilter.visualize`` convention.

    :param result: A
        :class:`~phonometry.filters.compliance.FilterComplianceResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the measured-curve ``plot`` call.
    :return: The axes.
    """
    from scipy import signal

    from .._i18n import format_number, localize_axes
    from ..filters.compliance import class_limits

    ax = ax if ax is not None else _new_axes()
    cls = result.reference_class()
    idx = _worst_band_index(result)
    fm = float(result.band_frequencies[idx])
    fsd = result.fs / float(result.factors[idx])
    sos = np.asarray(result.sos[idx], dtype=np.float64)

    # Recompute the relative attenuation exactly as verify_filter_class does:
    # -20 log10|H| minus the attenuation at the exact mid-band frequency.
    eps = np.finfo(float).eps
    w, h = signal.sosfreqz(sos, worN=result.num_points, fs=fsd)
    attenuation = -20.0 * np.log10(np.abs(h) + eps)
    _, h_ref = signal.sosfreqz(sos, worN=np.array([fm]), fs=fsd)
    a_ref = float(-20.0 * np.log10(np.abs(h_ref[0]) + eps))
    delta_a = attenuation - a_ref
    omega = w / fm

    keep = omega > 0.0
    omega, delta_a = omega[keep], delta_a[keep]
    order = np.argsort(omega)
    omega, delta_a = omega[order], delta_a[order]

    lower, upper = class_limits(result.fraction, cls, omega, edition=result.edition)

    # Symmetric log window centred on the mid-band (f / f_m = 1).
    omega_max = float(omega[-1])
    lo_x, hi_x = 1.0 / omega_max, omega_max
    win = (omega >= lo_x) & (omega <= hi_x)
    if not np.any(win):
        # Degenerate band (mid-band at or above the decimated Nyquist), so the
        # symmetric window is empty; fail clearly instead of a cryptic reduction.
        raise ValueError(
            "Cannot plot the filter class corridor: the mid-band frequency is at "
            "or above the analysis Nyquist, so the f/f_m window is empty."
        )
    finite_upper = np.isfinite(upper)

    # Scale the axis to the mask, not to the measured curve: a steep bank
    # reaches hundreds of dB of attenuation deep in the stop-band, which would
    # squash the corridor to a sliver. The measured curve is allowed to leave
    # the top; what matters is that it stays inside the green corridor.
    corridor_top = float(np.max(lower[win]))
    y_top = max(20.0, float(np.ceil((corridor_top + 8.0) / 10.0) * 10.0))
    y_bot = min(-2.0, float(np.floor(np.min(delta_a[win]) - 1.0)))

    # Green acceptance corridor; the upper limit is +inf in the stop-band
    # (unbounded attenuation allowed), so it is clipped to the axis top there.
    upper_fill = np.where(finite_upper, upper, y_top)
    ax.fill_between(
        omega[win], lower[win], upper_fill[win], color=theme_fill(_C_TERTIARY, ax),
        lw=0.0, label=_t("Class {cls} pass corridor", language, cls=cls),
    )
    ax.plot(omega[win], lower[win], color=_C_TERTIARY, lw=1.0, ls="--")
    fin = win & finite_upper
    ax.plot(omega[fin], upper[fin], color=_C_TERTIARY, lw=1.0, ls="--")

    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.6)
    kwargs.setdefault("label", _t(r"Measured $\Delta A$", language))
    ax.plot(omega[win], delta_a[win], **kwargs)

    violated = (delta_a < lower - 1e-9) | (finite_upper & (delta_a > upper + 1e-9))
    viol_win = violated & win
    if np.any(viol_win):
        ax.plot(
            omega[viol_win], delta_a[viol_win], ls="", marker="o", ms=3.5,
            color=_C_REFERENCE, label=_t("Out of tolerance", language),
        )

    ax.axvline(1.0, color=_C_MUTED, ls=":", lw=1.0)
    _normalized_frequency_axis(ax, lo_x, hi_x)
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlabel(_t(r"Normalised frequency $f\,/\,f_m$", language))
    ax.set_ylabel(_t("Relative attenuation [dB]", language))
    ax.set_title(
        _t("IEC 61260-1 class {cls} mask — $f_m$ = {fm} Hz", language, cls=cls,
           fm=format_number(fm, language, decimals=0))
    )
    ax.legend(loc="upper center", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


#: Fiche/plot prose for each IEC 61043 Table 2 column group.
def _normalized_frequency_axis(ax: Axes, lo: float, hi: float) -> None:
    """Label a logarithmic ``f / f_m`` axis with plain decimal ratios."""
    import matplotlib.ticker as mticker

    ax.set_xscale("log")
    ticks = [t for t in (0.25, 0.5, 0.7, 1.0, 1.4, 2.0, 4.0) if lo <= t <= hi]
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(
        mticker.FixedFormatter([f"{t:g}" for t in ticks])
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
def plot_parametric_eq(
    result: EQResponseResult, ax: Axes | None = None, *,
    language: str = "en", show_sections: bool = True, **kwargs: Any
) -> Axes | np.ndarray:
    """Magnitude and phase response of a parametric-EQ cascade.

    With ``ax`` given, only the magnitude panel is drawn on it.

    :param result: An
        :class:`~phonometry.filters.equalizer.EQResponseResult`.
    :param ax: Existing axes for the magnitude panel, or ``None`` for a
        fresh two-panel (magnitude + phase) figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param show_sections: Also draw each section's magnitude (light lines)
        when the cascade has more than one section.
    :param kwargs: Forwarded to the cascade magnitude line.
    :return: The magnitude axes (``ax`` given) or the array of two axes.
    """
    from .._i18n import decimal_comma, localize_axes

    freqs = np.asarray(result.frequencies, dtype=np.float64)
    fmin, fmax = float(freqs[0]), float(freqs[-1])
    color = kwargs.pop("color", _C_PRIMARY)

    def _magnitude(axm: Axes) -> None:
        if show_sections and result.section_magnitude_db.shape[0] > 1:
            for idx, section in enumerate(result.sections):
                label = decimal_comma(
                    f"{_t(section.filter_type, language)} "
                    f"{section.f0:g} Hz",
                    language,
                )
                axm.semilogx(
                    freqs, result.section_magnitude_db[idx],
                    color=_C_MUTED, lw=0.9, alpha=0.7,
                    label=label if idx < 8 else None,
                )
        kwargs.setdefault("lw", 1.8)
        kwargs.setdefault("label", _t("Cascade", language))
        axm.semilogx(freqs, result.magnitude_db, color=color, **kwargs)
        axm.axhline(0.0, color=_C_REFERENCE, linestyle=":", lw=0.8, alpha=0.5)
        axm.set_ylabel(_t("Magnitude [dB]", language))
        axm.grid(True, which="both", alpha=0.3)
        axm.legend(loc=_LEGEND_UPPER_RIGHT, fontsize="small")

    if ax is not None:
        _magnitude(ax)
        ax.set_xlabel(_t("Frequency [Hz]", language))
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 6.4))
    _magnitude(axes[0])
    axes[0].set_title(
        _t("Parametric EQ response (Audio EQ Cookbook)", language)
    )
    axes[1].semilogx(freqs, np.degrees(result.phase_rad), color=color, lw=1.4)
    axes[1].set_ylabel(_t("Phase [deg]", language))
    axes[1].set_xlabel(_t("Frequency [Hz]", language))
    axes[1].grid(True, which="both", alpha=0.3)
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes
