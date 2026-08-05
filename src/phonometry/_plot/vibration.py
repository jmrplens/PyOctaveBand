#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Plot renderers for the vibration domain (lazy imports from result .plot())."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .common import (
    _C_EDGE,
    _C_MUTED,
    _C_PRIMARY,
    _C_PRIMARY_LIGHT,
    _C_QUATERNARY,
    _C_REFERENCE,
    _C_SECONDARY,
    _C_TERTIARY,
    _band_axis,
    _new_axes,
    _new_axes_column,
    format_frequency_axis,
)

#: Bar colours for the per-operation partial exposures, cycled in order. Blue,
#: green, purple, light blue and grey deliberately avoid the orange EAV and red
#: ELV threshold-line colours, so a bar is never mistaken for a limit line.
_OP_COLORS = (_C_PRIMARY, _C_TERTIARY, _C_QUATERNARY, _C_PRIMARY_LIGHT, _C_MUTED)
#: The combined A(8) bar keeps a distinct neutral-dark colour: it is the
#: root-sum-of-squares of the operations, not one of them.
_A8_COLOR = _C_EDGE

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..vibration.human.exposure import (
        DailyVibrationExposure,
        WeightedSpectrum,
        WeightingResponse,
    )
    from ..vibration.human.multiple_shock import MultipleShockResult
    from ..vibration.machinery.diagnostics import FaultFrequencyResult
    from ..vibration.structural.experimental_sea import PowerInjectionResult
    from ..vibration.structural.junction_transmission import (
        JunctionTransmissionResult,
    )
    from ..vibration.structural.mechanical_mobility import (
        MobilityResult,
        RigidMassCalibrationResult,
    )
    from ..vibration.structural.radiation_efficiency import RadiationEfficiencyResult
    from ..vibration.structural.transfer_stiffness import TransferStiffnessResult

#: Spanish translations of the fixed strings rendered by the vibration
#: ``.plot()`` renderers, keyed by their verbatim English text. ``_t``
#: returns the English key unchanged for any language other than ``"es"``,
#: so the English output is byte-for-byte identical to the pre-i18n
#: renderers.
_STRINGS: dict[str, str] = {
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Weighting factor [dB]": "Factor de ponderación [dB]",
    "Unweighted $a_i$": "Sin ponderar $a_i$",
    "r.m.s. acceleration [m/s$^2$]": "Aceleración eficaz [m/s$^2$]",
    "Vibration exposure A(8) [m/s$^2$]": "Exposición a vibración A(8) [m/s$^2$]",
    "driving-point mobility": "movilidad en punto de excitación",
    "transfer mobility": "movilidad de transferencia",
    "Mobility $|Y|$ [m/(N·s)]": "Movilidad $|Y|$ [m/(N·s)]",
    "Transfer stiffness level $L_k$ [dB re 1 N/m]": "Nivel de rigidez de transferencia $L_k$ [dB re 1 N/m]",
    r"Radiation efficiency $\sigma$": r"Eficiencia de radiación $\sigma$",
    "Stress variable $R$": "Variable de tensión $R$",
    "Probability of lumbar injury [%]": "Probabilidad de lesión lumbar [%]",
    "ISO 7626-1 mechanical mobility": "ISO 7626-1 movilidad mecánica",
    "ISO 7626-2 rigid-mass calibration check": "ISO 7626-2 verificación de calibración con masa rígida",
    "Accelerance $|A|$ [1/kg]": "Acelerancia $|A|$ [1/kg]",
    "Deviation [%]": "Desviación [%]",
    r"expected $|A| = 1/m$": r"esperado $|A| = 1/m$",
    r"expected $|Y| = 1/(2\pi f m)$": r"esperado $|Y| = 1/(2\pi f m)$",
    "measured (within tolerance)": "medido (dentro de tolerancia)",
    "measured (out of tolerance)": "medido (fuera de tolerancia)",
    r"$\pm${p} % tolerance": r"tolerancia $\pm${p} %",
    "ISO 10846 dynamic transfer stiffness": "ISO 10846 rigidez dinámica de transferencia",
    "Plate radiation efficiency (Leppington / Maidanik)": "Eficiencia de radiación de placa (Leppington / Maidanik)",
    "Frequency weighting {name} (ISO 8041-1)": "Ponderación en frecuencia {name} (ISO 8041-1)",
    "Weighted $W_i a_i$ ({name})": "Ponderada $W_i a_i$ ({name})",
    "{designation} weighted acceleration spectrum  ($a_w$ = {aw} m/s$^2$)": "{designation} espectro de aceleración ponderada  ($a_w$ = {aw} m/s$^2$)",
    "Directive 2002/44/EC daily {kind} exposure  (A(8) = {a8} m/s$^2$, {zone})": "Directiva 2002/44/CE exposición diaria {kind}  (A(8) = {a8} m/s$^2$, {zone})",
    "below action": "por debajo de la acción",
    "action": "acción",
    "limit": "límite",
    "peak at {v} Hz": "máximo en {v} Hz",
    "ISO 2631-5 injury probability — {sex}": "ISO 2631-5 probabilidad de lesión — {sex}",
    "male": "hombre",
    "female": "mujer",
    r"$R$ = {r},  $\Pi$ = {p} %": r"$R$ = {r},  $\Pi$ = {p} %",
    "envelope spectrum": "espectro de envolvente",
    "Envelope amplitude": "Amplitud de la envolvente",
    "Predicted fault line": "Línea de fallo prevista",
    "Predicted fault lines: {src}, shaft {fs} Hz": "Líneas de fallo previstas: {src}, eje {fs} Hz",
    "rolling-contact bearing": "rodamiento de contacto rodante",
    "gear pair": "engranaje",
    "induction motor": "motor de inducción",
    "bladed rotor": "rotor con álabes",
    "shaft": "eje",
    "bearing": "rodamiento",
    "gear": "engranaje",
    "motor": "motor",
    "blade": "álabe",
    "Loss factor": "Factor de pérdidas",
    "Power-injection SEA loss factors ({method})": "Factores de pérdidas SEA por inyección de potencia ({method})",
    "single-drive": "excitación única",
    "two-drive": "doble excitación",
    # Plate-junction transmission (moved here with its renderer: _plot
    # holds one module per domain, and a junction is a vibration result).
    "Incidence angle [deg]": "Ángulo de incidencia [grados]",
    r"Transmission coefficient $\tau$": r"Coeficiente de transmisión $\tau$",
    "Bending-wave transmission, {junction}-junction "
    "($\\chi$ = {chi}, $\\psi$ = {psi})":
        "Transmisión de onda de flexión, unión {junction} "
        "($\\chi$ = {chi}, $\\psi$ = {psi})",
    r"corner $\tau_{12}(\theta)$": r"esquina $\tau_{12}(\theta)$",
    r"straight $\tau_{13}(\theta)$": r"recta $\tau_{13}(\theta)$",
    r"corner average $\bar\tau_{12}$ = {value}":
        r"media esquina $\bar\tau_{12}$ = {value}",
    r"straight average $\bar\tau_{13}$ = {value}":
        r"media recta $\bar\tau_{13}$ = {value}",
}


def _t(text: str, language: str = "en") -> str:
    """Localise a fixed string; English is returned verbatim (byte-identical)."""
    return _STRINGS.get(text, text) if language == "es" else text


def plot_vibration_weighting(
    result: WeightingResponse, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Frequency-weighting factor (dB) versus frequency (ISO 8041-1).

    :param result: A
        :class:`~phonometry.vibration.human.exposure.WeightingResponse` exposing
        ``name``, ``frequencies`` and ``magnitude_db``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the weighting curve ``plot`` call.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    mag_db = np.asarray(result.magnitude_db, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.semilogx(freqs, mag_db, **kwargs)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Weighting factor [dB]", language))
    ax.set_title(_t("Frequency weighting {name} (ISO 8041-1)", language).format(name=result.name))
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    localize_axes(ax, language)
    return ax


def plot_weighted_spectrum(
    result: WeightedSpectrum, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Unweighted vs weighted one-third-octave acceleration spectrum.

    Draws the measured band accelerations and, overlaid, the weighted band
    contributions ``W_i*a_i``; the overall ``a_w`` is annotated in the title.

    :param result: A
        :class:`~phonometry.vibration.human.exposure.WeightedSpectrum` exposing
        ``frequencies``, ``band_accelerations``, ``weighted``, ``overall`` and
        ``weighting_name``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the weighted (primary) bars.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    raw = np.asarray(result.band_accelerations, dtype=np.float64)
    weighted = np.asarray(result.weighted, dtype=np.float64)
    positions = _band_axis(ax, freqs, language=language)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    width = 0.4
    # The weighted bars are the primary artist; forward user kwargs there.
    kwargs.setdefault("color", _C_PRIMARY)
    ax.bar(
        positions - width / 2, raw, width, color=_C_MUTED,
        label=_t("Unweighted $a_i$", language),
    )
    ax.bar(
        positions + width / 2,
        weighted,
        width,
        label=_t("Weighted $W_i a_i$ ({name})", language).format(name=result.weighting_name),
        **kwargs,
    )
    ax.set_ylabel(_t("r.m.s. acceleration [m/s$^2$]", language))
    # Wh is the hand-arm weighting of ISO 5349-1; the others (Wk, Wd, Wm...)
    # are the whole-body weightings of ISO 2631.
    designation = "ISO 5349-1" if str(result.weighting_name) == "Wh" else "ISO 2631"
    ax.set_title(_t("{designation} weighted acceleration spectrum  ($a_w$ = {aw} m/s$^2$)", language).format(
        designation=designation,
        aw=format_number(float(result.overall), language, decimals=3),
    ))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    # localize_axes leaves the categorical band axis (a FuncFormatter) alone.
    localize_axes(ax, language)
    return ax


def plot_daily_exposure(
    result: DailyVibrationExposure, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Partial daily exposures against the EAV / ELV (Directive 2002/44/EC).

    Draws one bar per operation (its partial exposure ``A_i(8)``), a combined
    ``A(8)`` bar, and the exposure action and limit value as horizontal lines.

    :param result: A
        :class:`~phonometry.vibration.human.exposure.DailyVibrationExposure` exposing
        ``labels``, ``partials``, ``a8`` and ``assessment``.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to each exposure :meth:`~matplotlib.axes.Axes.bar`.
    :return: The axes.
    """
    from .._i18n import decimal_comma, format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    partials = np.asarray(result.partials, dtype=np.float64)
    n_ops = partials.size
    values = [*partials.tolist(), float(result.a8)]
    # A small gap separates the combined A(8) bar from the operation bars.
    positions = [*range(n_ops), n_ops + 0.5]

    # One bar per operation, each with its own colour and legend label, so the
    # operation identity lives in the legend rather than in crowded, rotated
    # x-axis category labels. The combined A(8) bar takes the distinct A(8)
    # colour. A caller-supplied colour/width would collide with the per-bar
    # colouring, so those keys are consumed here.
    kwargs.pop("color", None)
    width = kwargs.pop("width", 0.7)
    edgecolor = kwargs.pop("edgecolor", _C_EDGE)
    for i in range(n_ops):
        ax.bar(
            positions[i], partials[i], width=width,
            color=_OP_COLORS[i % len(_OP_COLORS)], edgecolor=edgecolor,
            linewidth=0.6, zorder=3, label=str(result.labels[i]), **kwargs,
        )
    ax.bar(
        positions[-1], values[-1], width=width, color=_A8_COLOR,
        edgecolor=edgecolor, linewidth=0.6, zorder=3, label="A(8)", **kwargs,
    )
    # The legend carries the bar identity, so the crowded category ticks go.
    ax.set_xticks([])
    ax.set_ylabel(_t("Vibration exposure A(8) [m/s$^2$]", language))

    assessment = result.assessment
    eav = float(assessment.action_value)
    elv = float(assessment.limit_value)
    ax.axhline(eav, color=_C_SECONDARY, ls="--", lw=1.4,
               label="EAV = " + decimal_comma(f"{eav:g}", language))
    ax.axhline(elv, color=_C_REFERENCE, ls="--", lw=1.4,
               label="ELV = " + decimal_comma(f"{elv:g}", language))
    top = max(elv, float(np.max(values))) * 1.28
    ax.set_ylim(0.0, top)
    kind = str(assessment.kind).upper()
    zone = _t(str(assessment.zone), language)
    ax.set_title(_t("Directive 2002/44/EC daily {kind} exposure  (A(8) = {a8} m/s$^2$, {zone})", language).format(
        kind=kind, a8=format_number(float(result.a8), language, decimals=2),
        zone=zone,
    ))
    handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(handles, leg_labels, loc="upper center", ncol=min(3, len(handles)),
              fontsize="small", frameon=True, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    localize_axes(ax, language)
    return ax


def plot_mobility(
    result: MobilityResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Mobility magnitude ``|Y(f)|`` on log-log axes (ISO 7626-1).

    :param result: A :class:`~phonometry.vibration.structural.mechanical_mobility.MobilityResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the magnitude ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    mag = np.asarray(result.magnitude, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    label = (_t("driving-point mobility", language) if result.driving_point
             else _t("transfer mobility", language))
    ax.loglog(freq, mag, label=label, **kwargs)
    # Mark the mobility peak (a resonance for a driving-point FRF).
    peak = int(np.argmax(mag))
    ax.plot(freq[peak], mag[peak], "o", color=_C_REFERENCE, zorder=5,
            label=_t("peak at {v} Hz", language).format(
                v=format_number(freq[peak], language, decimals=1)))
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Mobility $|Y|$ [m/(N·s)]", language))
    ax.set_title(_t("ISO 7626-1 mechanical mobility", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_rigid_mass_calibration(
    result: RigidMassCalibrationResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes | np.ndarray:
    """Rigid-mass operational calibration check (ISO 7626-2, 7.5.2).

    The single-concept calibration-check figure: the measured driving-point
    FRF magnitude against the known rigid-mass line with its +/- tolerance
    band (upper panel), and the relative deviation against the same tolerance
    band (lower panel, where the few-percent band is actually readable). A
    point outside the band is a failed frequency and is drawn in the
    out-of-tolerance colour; the title carries the overall verdict.

    With ``ax`` supplied, only the deviation diagnostic is drawn on it and that
    axes is returned; otherwise a fresh two-panel column is created and the
    two-axes array is returned.

    :param result: A
        :class:`~phonometry.vibration.structural.mechanical_mobility.RigidMassCalibrationResult`.
    :param ax: Existing axes for the deviation panel, or ``None`` for a fresh
        two-panel figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the primary artist: the measured-magnitude
        curve for the full two-panel figure, or the deviation curve when ``ax``
        is supplied.
    :return: The deviation axes (``ax`` given) or the two-axes array.
    """
    from .._i18n import format_number, localize_axes

    freq = np.asarray(result.frequencies, dtype=np.float64)
    measured = np.asarray(result.measured, dtype=np.float64)
    expected = np.asarray(result.expected, dtype=np.float64)
    deviation = np.asarray(result.deviation, dtype=np.float64)
    within = np.asarray(result.within_tolerance, dtype=bool)
    tol = float(result.tolerance)
    tol_pct = 100.0 * tol
    fmin, fmax = float(freq.min()), float(freq.max())
    p = format_number(tol_pct, language, decimals=1, trim=True)
    band_label = _t(r"$\pm${p} % tolerance", language).format(p=p)
    exp_label = (_t(r"expected $|A| = 1/m$", language)
                 if result.quantity == "accelerance"
                 else _t(r"expected $|Y| = 1/(2\pi f m)$", language))
    mag_ylabel = (_t("Accelerance $|A|$ [1/kg]", language)
                  if result.quantity == "accelerance"
                  else _t("Mobility $|Y|$ [m/(N·s)]", language))
    within_label = _t("measured (within tolerance)", language)
    outside_label = _t("measured (out of tolerance)", language)

    def _deviation_panel(axd: Axes, **line_kwargs: Any) -> None:
        axd.axhspan(-tol_pct, tol_pct, color=_C_REFERENCE, alpha=0.15,
                    label=band_label)
        axd.axhline(0.0, color=_C_MUTED, ls=":", lw=0.9)
        line_kwargs.setdefault("color", _C_PRIMARY)
        axd.semilogx(freq, 100.0 * deviation, "-", lw=1.4, zorder=2,
                     **line_kwargs)
        axd.plot(freq[within], 100.0 * deviation[within], "o", color=_C_PRIMARY,
                 zorder=3, label=within_label)
        if not np.all(within):
            axd.plot(freq[~within], 100.0 * deviation[~within], "o",
                     color=_C_SECONDARY, zorder=4, label=outside_label)
        axd.set_xlabel(_t("Frequency [Hz]", language))
        axd.set_ylabel(_t("Deviation [%]", language))
        axd.grid(True, which="both", alpha=0.3)
        axd.legend(loc="best", fontsize="small")

    if ax is not None:
        _deviation_panel(ax, **kwargs)
        format_frequency_axis(ax, fmin, fmax)
        localize_axes(ax, language)
        return ax

    axes = _new_axes_column(2, sharex=True, figsize=(8.0, 6.6))
    axm, axd = axes[0], axes[1]
    kwargs.setdefault("color", _C_PRIMARY)
    axm.fill_between(freq, expected * (1.0 - tol), expected * (1.0 + tol),
                     color=_C_REFERENCE, alpha=0.15, label=band_label)
    axm.loglog(freq, expected, ls="--", color=_C_REFERENCE, lw=1.4,
               label=exp_label)
    axm.loglog(freq, measured, "o-", lw=1.4, label=within_label, **kwargs)
    if not np.all(within):
        axm.plot(freq[~within], measured[~within], "o", color=_C_SECONDARY,
                 zorder=4, label=outside_label)
    axm.set_ylabel(mag_ylabel)
    axm.grid(True, which="both", alpha=0.3)
    axm.legend(loc="best", fontsize="small")
    if result.passed:
        verdict = "CORRECTO" if language == "es" else "PASS"
    else:
        verdict = "INCORRECTO" if language == "es" else "FAIL"
    axm.set_title(
        _t("ISO 7626-2 rigid-mass calibration check", language)
        + f" ({verdict})"
    )
    _deviation_panel(axd)
    for axf in axes:
        format_frequency_axis(axf, fmin, fmax)
        localize_axes(axf, language)
    return axes


def plot_transfer_stiffness(
    result: TransferStiffnessResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Dynamic transfer stiffness level ``L_k(f)`` on a log-frequency axis.

    :param result: A :class:`~phonometry.vibration.structural.transfer_stiffness.TransferStiffnessResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the level ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    level = np.asarray(result.level, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    ax.semilogx(freq, level, label=r"$L_k = 20\,\log_{10}(|k_{2,1}|/k_0)$", **kwargs)
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Transfer stiffness level $L_k$ [dB re 1 N/m]", language))
    ax.set_title(_t("ISO 10846 dynamic transfer stiffness", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_radiation_efficiency(
    result: RadiationEfficiencyResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Radiation efficiency ``sigma(f)`` on log-log axes (Hopkins 2.9.4).

    :param result: A
        :class:`~phonometry.vibration.structural.radiation_efficiency.RadiationEfficiencyResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``sigma`` curve ``plot``.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    sigma = np.asarray(result.radiation_efficiency, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 3)
    ax.loglog(freq, sigma, label=r"$\sigma(f)$", **kwargs)
    ax.axhline(1.0, color=_C_MUTED, ls=":", lw=0.9, label=r"$\sigma = 1$")
    ax.axvline(
        result.critical_frequency,
        color=_C_REFERENCE,
        ls="--",
        lw=1.0,
        label=f"$f_c$ = {result.critical_frequency:.0f} Hz",
    )
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t(r"Radiation efficiency $\sigma$", language))
    ax.set_title(_t("Plate radiation efficiency (Leppington / Maidanik)", language))
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_multiple_shock(
    result: MultipleShockResult, ax: Axes | None = None, *, language: str = "en",
    **kwargs: Any,
) -> Axes:
    """Injury-probability curve ``P(R)`` with this assessment's ``R`` marked.

    :param result: A :class:`~phonometry.vibration.human.multiple_shock.MultipleShockResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``R`` marker ``scatter``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes
    from ..vibration.human.multiple_shock import injury_probability

    ax = ax if ax is not None else _new_axes()
    r10, r50, r90 = result.risk_thresholds
    r_max = max(result.risk, r90) * 1.3
    grid = np.linspace(0.0, r_max, 240)
    prob = np.asarray(injury_probability(grid, sex=result.sex), dtype=np.float64)
    ax.plot(grid, 100.0 * prob, color=_C_PRIMARY,
            label=r"$\Pi(R) = 1 - e^{-(R/\alpha)^{\beta}}$")
    for level, r_val in zip((10, 50, 90), (r10, r50, r90)):
        ax.axhline(level, color=_C_MUTED, ls=":", lw=0.8)
        ax.plot([r_val, r_val], [0.0, level], color=_C_MUTED, ls=":", lw=0.8)

    kwargs.setdefault("color", _C_REFERENCE)
    kwargs.setdefault("zorder", 4)
    kwargs.setdefault("s", 90)
    ax.scatter([result.risk], [100.0 * result.probability],
               label=_t(r"$R$ = {r},  $\Pi$ = {p} %", language).format(
                   r=format_number(result.risk, language, decimals=2),
                   p=format_number(100.0 * result.probability, language, decimals=0)),
               **kwargs)
    ax.set_xlabel(_t("Stress variable $R$", language))
    ax.set_ylabel(_t("Probability of lumbar injury [%]", language))
    ax.set_title(
        _t("ISO 2631-5 injury probability — {sex}", language).format(
            sex=_t(str(result.sex), language)
        )
    )
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 100.0)
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(True, alpha=0.3)
    localize_axes(ax, language)
    return ax


#: Marker colour of each fault-line family, so shaft harmonics never read as
#: bearing evidence on the overlay.
_FAMILY_COLORS: dict[str, str] = {
    "shaft": _C_MUTED,
    "bearing": _C_REFERENCE,
    "gear": _C_TERTIARY,
    "motor": _C_QUATERNARY,
    "blade": _C_SECONDARY,
}


def _draw_measured_spectrum(
    ax: Axes,
    frequencies: Any,
    amplitude: Any,
    f_max: float,
    language: str,
    kwargs: dict[str, Any],
) -> float:
    """Draw the measured curve under the overlay; return its peak amplitude.

    Without a spectrum only the axis label is set and the reference height is
    1, so the predicted lines fill the axes on their own.
    """
    if frequencies is None or amplitude is None:
        ax.set_ylabel(_t("Predicted fault line", language))
        return 1.0
    keep = frequencies <= f_max
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("lw", 1.0)
    ax.plot(frequencies[keep], amplitude[keep],
            label=_t("envelope spectrum", language), **kwargs)
    ax.set_ylabel(_t("Envelope amplitude", language))
    return float(np.max(amplitude[keep])) if np.any(keep) else 1.0


#: Gap between a fault line and its own rotated label, in points.
_LABEL_PAD_PT = 2.0
#: Horizontal room one rotated ``x-small`` label needs, in points. The name is
#: turned through 90 degrees, so what it occupies across the axis is its text
#: height, not its length.
_LABEL_WIDTH_PT = 8.5
#: Fallback axis width, in points, when the axes geometry is not available.
_LABEL_FALLBACK_WIDTH_PT = 400.0


def _axis_width_points(ax: Axes) -> float:
    """Width of the axes box in points, for laying labels out across it."""
    try:
        width_px = float(ax.get_window_extent().width)
        dpi = float(ax.get_figure().dpi)  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - defensive
        return _LABEL_FALLBACK_WIDTH_PT
    width_pt = width_px * 72.0 / dpi
    return width_pt if width_pt > 0.0 else _LABEL_FALLBACK_WIDTH_PT


def _label_offsets(
    frequencies: list[float], f_max: float, width_pt: float
) -> dict[int, float]:
    """Horizontal label offsets in points, keyed by index into *frequencies*.

    The names are drawn rotated, so each occupies a narrow vertical strip and
    two of them collide when their lines sit less than :data:`_LABEL_WIDTH_PT`
    apart across the axis. Walking the lines in frequency order and pushing
    each label just far enough right to clear the previous one separates a
    crowded group with the least displacement that fits: an isolated line keeps
    its label exactly where it was, and a run of evenly spaced sidebands drifts
    by the difference between the label width and the spacing, no more.
    """
    scale = width_pt / f_max if f_max > 0.0 else 0.0
    offsets: dict[int, float] = {}
    cursor = -np.inf
    for i in sorted(range(len(frequencies)), key=lambda i: frequencies[i]):
        anchor = frequencies[i] * scale + _LABEL_PAD_PT
        placed = max(anchor, cursor)
        offsets[i] = placed - frequencies[i] * scale
        cursor = placed + _LABEL_WIDTH_PT
    return offsets


def _draw_fault_lines(
    ax: Axes,
    result: FaultFrequencyResult,
    f_max: float,
    top: float,
    language: str,
    *,
    annotate: bool,
) -> None:
    """Draw one dashed line per predicted fault frequency, coloured by family.

    Each family contributes a single legend entry, so shaft harmonics never
    read as bearing evidence.
    """
    visible = [line for line in result.lines if line.frequency <= f_max]
    offsets = _label_offsets(
        [line.frequency for line in visible], f_max, _axis_width_points(ax)
    )
    labelled: set[str] = set()
    for i, line in enumerate(visible):
        colour = _FAMILY_COLORS.get(line.family, _C_EDGE)
        label = None if line.family in labelled else _t(line.family, language)
        labelled.add(line.family)
        ax.axvline(line.frequency, color=colour, ls="--", lw=1.0, alpha=0.85,
                   label=label, zorder=2)
        if annotate:
            ax.annotate(
                line.name, xy=(line.frequency, 1.02 * top),
                xytext=(offsets[i], 0), textcoords="offset points",
                rotation=90, fontsize="x-small", color=colour,
                ha="left", va="bottom",
            )


def plot_fault_frequencies(
    result: FaultFrequencyResult, ax: Axes | None = None, *,
    language: str = "en", spectrum: Any | None = None,
    max_frequency: float | None = None, annotate: bool = True,
    **kwargs: Any,
) -> Axes:
    """Predicted machine fault lines over a measured envelope spectrum.

    The working diagnostic view: the envelope spectrum of a band-passed
    vibration record with the kinematic lines of the bearing, gear, motor or
    impeller drawn on top and named. A peak is only evidence when it lands on
    a named line, which is what the overlay makes readable.

    :param result: A
        :class:`~phonometry.vibration.machinery.diagnostics.FaultFrequencyResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param spectrum: Measured spectrum to draw underneath: an
        :class:`~phonometry.signals.envelope.EnvelopeSpectrumResult`, or any
        object exposing ``frequencies`` and ``amplitude``. Without it the
        predicted lines are drawn alone.
    :param max_frequency: Upper limit of the frequency axis, in hertz
        (Default: the spectrum's own upper limit, or 1,15 x the highest line).
    :param annotate: Label each line with its name (Default: ``True``).
    :param kwargs: Forwarded to the spectrum curve.
    :return: The axes.
    :raises ValueError: If the result carries no lines.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    freqs = np.asarray(result.frequencies, dtype=np.float64)
    if freqs.size == 0:
        raise ValueError("the result carries no fault lines to plot.")

    f_max = max_frequency
    spectrum_f = spectrum_a = None
    if spectrum is not None:
        spectrum_f = np.asarray(spectrum.frequencies, dtype=np.float64)
        spectrum_a = np.asarray(spectrum.amplitude, dtype=np.float64)
        if f_max is None:
            f_max = float(spectrum_f.max())
    if f_max is None:
        f_max = 1.15 * float(freqs.max())

    top = _draw_measured_spectrum(
        ax, spectrum_f, spectrum_a, f_max, language, kwargs
    )
    ax.set_ylim(0.0, 1.24 * top)
    _draw_fault_lines(ax, result, f_max, top, language, annotate=annotate)
    ax.set_xlim(0.0, f_max)
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_title(
        _t("Predicted fault lines: {src}, shaft {fs} Hz", language).format(
            src=_t(result.source, language),
            fs=format_number(result.shaft_rate, language, decimals=2, trim=True),
        )
    )
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_power_injection(
    result: PowerInjectionResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Loss-factor budget of a two-subsystem SEA model (Norton Ch. 6).

    The measured coupling loss factors against the internal loss factors on
    one logarithmic axis: SEA is only trustworthy where the coupling stays
    below the internal damping, so the ordering of the four curves is the
    diagnosis.

    :param result: A
        :class:`~phonometry.vibration.structural.experimental_sea.PowerInjectionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the ``eta_12`` curve.
    :return: The axes.
    """
    from .._i18n import localize_axes

    ax = ax if ax is not None else _new_axes()
    freq = np.asarray(result.frequencies, dtype=np.float64)
    kwargs.setdefault("color", _C_PRIMARY)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 4)
    ax.loglog(freq, result.coupling_loss_factor12, label=r"$\eta_{12}$",
              **kwargs)
    ax.loglog(freq, result.coupling_loss_factor21, color=_C_SECONDARY,
              marker="s", markersize=4, label=r"$\eta_{21}$")
    ax.loglog(freq, result.internal_loss_factor1, color=_C_TERTIARY, ls="--",
              lw=1.1, label=r"$\eta_{1}$")
    ax.loglog(freq, result.internal_loss_factor2, color=_C_QUATERNARY, ls=":",
              lw=1.3, label=r"$\eta_{2}$")
    format_frequency_axis(ax, float(freq.min()), float(freq.max()))
    ax.set_xlabel(_t("Frequency [Hz]", language))
    ax.set_ylabel(_t("Loss factor", language))
    ax.set_title(
        _t("Power-injection SEA loss factors ({method})", language).format(
            method=_t(result.method, language)
        )
    )
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    localize_axes(ax, language)
    return ax


def plot_junction_transmission(
    result: JunctionTransmissionResult, ax: Axes | None = None, *,
    language: str = "en", **kwargs: Any,
) -> Axes:
    """Plot ``tau(theta)`` versus incidence angle (Hopkins Eqs 5.12/5.13).

    Draws the corner coefficient and, where it exists, the straight-section
    coefficient, with their diffuse-field angular averages as horizontal
    reference lines.

    :param result: A
        :class:`~phonometry.vibration.structural.junction_transmission.JunctionTransmissionResult`.
    :param ax: Existing axes, or ``None`` to create a figure.
    :param language: Label language, ``"en"`` (default) or ``"es"``.
    :param kwargs: Forwarded to the corner-curve ``plot``.
    :return: The axes.
    """
    from .._i18n import format_number, localize_axes

    ax = ax if ax is not None else _new_axes()
    angles = np.asarray(result.angles_deg, dtype=np.float64)
    corner = np.asarray(result.corner, dtype=np.float64)

    kwargs.setdefault("color", _C_PRIMARY)
    ax.plot(angles, corner, label=_t(r"corner $\tau_{12}(\theta)$", language), **kwargs)
    ax.axhline(
        result.corner_average, color=_C_PRIMARY, ls="--", lw=1.0,
        label=_t(r"corner average $\bar\tau_{12}$ = {value}", language).replace(
            "{value}", format_number(result.corner_average, language, decimals=4)
        ),
    )
    if result.straight is not None and result.straight_average is not None:
        straight = np.asarray(result.straight, dtype=np.float64)
        ax.plot(
            angles, straight, color=_C_SECONDARY,
            label=_t(r"straight $\tau_{13}(\theta)$", language),
        )
        ax.axhline(
            result.straight_average, color=_C_SECONDARY, ls=":", lw=1.0,
            label=_t(r"straight average $\bar\tau_{13}$ = {value}", language).replace(
                "{value}", format_number(result.straight_average, language, decimals=4)
            ),
        )

    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(_t("Incidence angle [deg]", language))
    ax.set_ylabel(_t(r"Transmission coefficient $\tau$", language))
    ax.set_title(
        _t(
            "Bending-wave transmission, {junction}-junction "
            "($\\chi$ = {chi}, $\\psi$ = {psi})",
            language,
        ).format(
            junction=result.junction,
            chi=format_number(result.chi, language, decimals=3),
            psi=format_number(result.psi, language, decimals=3),
        )
    )
    ax.grid(True, color=_C_MUTED, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    localize_axes(ax, language)
    return ax
