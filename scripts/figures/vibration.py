#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the vibration guides: exposure, mobility and structure-borne power.

Vibration as a measured quantity and as a transmission path: the human
exposure weightings and daily exposure, mechanical mobility and dynamic
stiffness, structure-borne sound power, the SEA coupling loss factors, and the
bearing-fault envelope that reads a machine's condition. Everything here is
embedded by a page under ``vibration/``.
"""

import math
from typing import Literal


def _sci_math(value: float, digits: int = 2) -> str:
    """A scientific-notation reading composed as mathtext.

    ``f"{4.4e-3:.2e}"`` writes "4.40e-03", which a figure has no business
    showing; this returns ``$4.40\\times10^{-3}$``, where mathtext sets the
    proper multiplication sign and a typographic minus in the exponent.
    """
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10.0**exponent
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from phonometry._plot.common import format_frequency_axis, theme_fill

from .i18n import _LANG
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    _band_index_axis,
    save_figure,
)


def generate_junction_transmission(output_dir: str) -> None:
    """Hopkins 5.2.1.3 bending-wave transmission at a rigid X-junction."""
    print("Generating junction_transmission...")
    from phonometry import junction_transmission

    # X-junction between a 100 mm and a 200 mm concrete plate (cL = 3200 m/s,
    # rho = 2400 kg/m^3 -> rho_s = 240 and 480 kg/m^2).
    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
    assert res.straight is not None and res.straight_average is not None
    angles = res.angles_deg

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(angles, res.corner, color=COLOR_PRIMARY, linewidth=2.0,
            label=r"corner $\tau_{12}(\theta)$")
    ax.plot(angles, res.straight, color=COLOR_SECONDARY, linewidth=2.0,
            label=r"straight $\tau_{13}(\theta)$")
    ax.axhline(res.corner_average, color=COLOR_PRIMARY, linestyle="--",
               linewidth=1.3, label="corner average")
    ax.axhline(res.straight_average, color=COLOR_SECONDARY, linestyle=":",
               linewidth=1.3, label="straight average")

    # The cut-off: beyond arcsin(chi) the receiving plate has no propagating
    # bending wave to accept, so the corner path shuts completely.
    theta_co = math.degrees(math.asin(min(1.0, res.chi)))
    ax.axvspan(theta_co, 90.0, color=theme_fill(COLOR_SECONDARY, ax),
               zorder=0)
    ax.axvline(theta_co, color=COLOR_FG, linestyle="-.", linewidth=1.4,
               label=r"cut-off $\theta_\mathrm{co} = \arcsin\chi$")
    ax.annotate(rf"$\theta_\mathrm{{co}} = {theta_co:.0f}°$: "
                r"$\tau_{12} = 0$ beyond it",
                xy=(theta_co, 0.5 * float(np.max(res.corner))),
                xytext=(theta_co + 6.0, 0.78 * float(np.max(res.corner))),
                fontsize=9.5, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})

    ax.set_xlabel("Incidence angle [degrees]")
    ax.set_ylabel(r"Transmission coefficient $\tau$")
    ax.set_title(
        "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)",
        pad=12,
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "X-junction: 100 mm / 200 mm concrete",
        rf"$\chi$ = {res.chi:.3f},  $\psi$ = {res.psi:.3f}",
        rf"$\theta_\mathrm{{co}} = \arcsin\chi$ = {theta_co:.1f}°",
        f"corner avg = {res.corner_average:.4f}",
        f"straight avg = {res.straight_average:.4f}",
    ]
    ax.text(0.015, 0.60, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "junction_transmission.svg")
    plt.close()


def generate_mechanical_mobility(output_dir: str) -> None:
    """ISO 7626-1 receptance/mobility/accelerance of a SDOF resonator."""
    print("Generating mechanical_mobility...")
    from phonometry import (
        convert_frf,
        resonance_frequency,
        sdof_receptance,
    )

    m, k, c = 2.0, 8000.0, 5.0
    f0 = resonance_frequency(m, k)
    freq = np.logspace(np.log10(f0 / 20.0), np.log10(f0 * 20.0), 600)
    w0 = 2.0 * np.pi * f0
    h = sdof_receptance(freq, m, k, c)
    y = convert_frf(h, freq, "receptance", "mobility")
    a = convert_frf(h, freq, "receptance", "accelerance")
    # Normalise each FRF to O(1) near resonance so all three share one axis.
    curves = [
        (np.abs(h) * k, COLOR_PRIMARY, r"Receptance $|H|$ ($\times k$)"),
        (np.abs(y) * k / w0, COLOR_SECONDARY,
         r"Mobility $|Y|$ ($\times k/\omega_0$)"),
        (np.abs(a) * k / w0**2, COLOR_TERTIARY,
         r"Accelerance $|A|$ ($\times k/\omega_0^2$)"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for mag, color, label in curves:
        ax.loglog(freq, mag, color=color, linewidth=2.0, label=label)
    ax.axvline(f0, color=COLOR_GRID, linestyle="--", linewidth=1.2,
               label="resonance $f_0$")

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized FRF magnitude")
    ax.set_title("ISO 7626-1 Mechanical Mobility FRFs", pad=12)
    ax.set_xlim(freq[0], freq[-1])
    format_frequency_axis(ax, float(freq[0]), float(freq[-1]))
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        "SDOF: $m$ = 2 kg, $k$ = 8000 N/m, $c$ = 5 N·s/m",
        r"$H = 1/(k - \omega^2 m + \mathrm{j}\,\omega c)$",
        r"$Y = \mathrm{j}\,\omega H$,   $A = -\omega^2 H$  (Table 1)",
        rf"$f_0$ = {f0:.1f} Hz,  $|Y(f_0)| = 1/c$",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "mechanical_mobility.svg")
    plt.close()


def generate_transfer_stiffness(output_dir: str) -> None:
    """ISO 10846 dynamic transfer stiffness: true vs indirect-method recovery."""
    print("Generating transfer_stiffness...")
    from phonometry import (
        base_transmissibility,
        transfer_stiffness_indirect,
        transfer_stiffness_level,
    )

    # Kelvin-Voigt isolator k + jwc, loaded by a blocking mass m2.
    k, c, m2 = 1.0e6, 120.0, 8.0
    f0 = np.sqrt(k / m2) / (2.0 * np.pi)
    freq = np.logspace(np.log10(f0 / 5.0), np.log10(f0 * 40.0), 600)
    w = 2.0 * np.pi * freq

    k_true = k + 1j * w * c                                # exact transfer stiffness
    t = base_transmissibility(freq, m2, k, c)              # mass-loaded transmissibility
    k_indirect = transfer_stiffness_indirect(freq, t, m2)  # ISO 10846-3 Eq. (1)

    # Where the standard's own criterion starts holding: |T| <= 0.1
    # (Inequality 2), not the 3 f0 rule of thumb.
    magnitude_t = np.abs(np.asarray(t, dtype=np.complex128))
    valid = np.flatnonzero(magnitude_t <= 0.1)
    f_valid = float(freq[valid[0]]) if valid.size else float(freq[-1])
    level_error = (transfer_stiffness_level(k_indirect)
                   - transfer_stiffness_level(k_true))
    eta = np.imag(k_true) / np.real(k_true)          # Kelvin-Voigt: w c / k

    fig, (ax, mid, low) = plt.subplots(
        3, 1, figsize=(10, 10.2), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0, 0.85]})
    ax.semilogx(freq, transfer_stiffness_level(k_true), color=COLOR_PRIMARY,
                linewidth=2.2,
                label=r"true $L_k$ of $k_{2,1}=k+\mathrm{j}\,\omega c$")
    ax.semilogx(freq, transfer_stiffness_level(k_indirect), color=COLOR_SECONDARY,
                linewidth=2.0, linestyle="--",
                label=r"indirect method $-(2\pi f)^2 m_2 T$")
    ax.axvline(f0, color=COLOR_GRID, linestyle=":", linewidth=1.2,
               label="resonance $f_0$")
    for panel in (ax, mid, low):
        panel.axvspan(freq[0], f_valid, color=theme_fill(COLOR_SECONDARY, panel),
                      zorder=0)
        panel.axvline(f_valid, color=COLOR_FG, linestyle="-.", linewidth=1.3)
        panel.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
        panel.set_axisbelow(True)

    ax.set_ylabel(r"$L_k$ [dB re 1 N/m]")
    ax.set_title("ISO 10846 Dynamic Transfer Stiffness",
                 pad=12)
    ax.set_xlim(freq[0], freq[-1])
    ax.legend(loc="upper left", fontsize=9)
    info = [
        "Kelvin-Voigt: $k$ = 1 MN/m, $c$ = 120 N·s/m",
        rf"blocking mass $m_2$ = 8 kg,  $f_0$ = {f0:.1f} Hz",
        rf"$|T|$ falls to 0.1 at {f_valid:.0f} Hz",
        "shaded: Inequality (2) not met → no result",
    ]
    ax.text(0.985, 0.05, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    mid.loglog(freq, magnitude_t, color=COLOR_PRIMARY, linewidth=2.0,
               label=r"transmissibility $|T|$")
    mid.axhline(0.1, color=COLOR_FG, linestyle="--", linewidth=1.4,
                label=r"TRANSMISSIBILITY_LIMIT = 0.1  ($\Delta L_{1,2}$ = 20 dB)")
    mid.set_ylabel("$|T|$")
    mid.legend(loc="lower left", fontsize=9)
    twin = mid.twinx()
    # The primary axes must draw over the twin, or the shaded +-1 dB band
    # hides the |T| curve where the two cross.
    mid.set_zorder(twin.get_zorder() + 1)
    mid.patch.set_visible(False)
    twin.semilogx(freq, level_error, color=COLOR_TERTIARY, linewidth=1.8)
    twin.axhspan(-1.0, 1.0, color=theme_fill(COLOR_TERTIARY, twin), zorder=0)
    twin.set_ylim(-6.0, 6.0)
    twin.set_ylabel(r"$L_{k,\mathrm{ind}} - L_{k,\mathrm{true}}$ [dB]",
                    color=COLOR_TERTIARY)
    twin.tick_params(axis="y", colors=COLOR_TERTIARY)
    twin.text(0.985, 0.06, "the ±1 dB the criterion buys", transform=twin.transAxes,
              va="bottom", ha="right", fontsize=9.5, color=COLOR_TERTIARY)

    low.semilogx(freq, eta, color=COLOR_SECONDARY, linewidth=2.0,
                 label=r"loss factor $\eta = \mathrm{Im}(k_{2,1})/"
                       r"\mathrm{Re}(k_{2,1})$")
    low.set_ylabel(r"$\eta$")
    low.set_xlabel(LABEL_FREQ_HZ)
    format_frequency_axis(low, float(freq[0]), float(freq[-1]))
    low.legend(loc="upper left", fontsize=9)
    low.text(0.985, 0.90, "Kelvin-Voigt makes $\\eta$ rise with frequency; real\n"
             "elastomers are far flatter, so this is a model, not a material",
             transform=low.transAxes, va="top", ha="right", fontsize=9.5,
             color=COLOR_FG)
    fig.align_ylabels()
    plt.tight_layout()
    save_figure(output_dir, "transfer_stiffness.svg")
    plt.close()


def generate_rigid_mass_calibration(output_dir: str) -> None:
    """ISO 7626-2 (7.5.2) operational rigid-mass calibration check."""
    print("Generating rigid_mass_calibration...")
    from phonometry import rigid_mass_calibration_check

    # A 10 kg calibration block: the accelerance must be a flat |A| = 1/m over
    # frequency. The measured chain has a mild ripple and drifts above the
    # +/-5 % band towards a few kHz (a transducer/attachment-compliance error,
    # exactly what the check is meant to catch).
    m = 10.0
    freq = np.logspace(np.log10(20.0), np.log10(5000.0), 400)
    expected = 1.0 / m
    ripple = 0.015 * np.sin(2.0 * np.pi * np.log10(freq))
    drift = 0.05 * (freq / 2500.0) ** 2
    measured = expected * (1.0 + ripple + drift)
    res = rigid_mass_calibration_check(measured, freq, mass=m)
    within = res.within_tolerance
    tol = res.tolerance

    _fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7.0),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )
    # Upper panel: measured accelerance against the rigid-mass line + band.
    ax_top.fill_between(freq, res.expected * (1.0 - tol), res.expected * (1.0 + tol),
                        color=COLOR_SECONDARY, alpha=0.15,
                        label=r"$\pm$5 % tolerance band")
    ax_top.semilogx(freq, res.expected, color=COLOR_SECONDARY, linestyle="--",
                    linewidth=1.6, label=r"expected $|A| = 1/m$")
    ax_top.semilogx(freq, res.measured, color=COLOR_PRIMARY, linewidth=2.0,
                    label="within tolerance")
    ax_top.semilogx(freq[~within], res.measured[~within], linestyle="none",
                    marker="o", markersize=4, color=COLOR_SECONDARY,
                    label="out of tolerance")
    ax_top.set_ylabel("Accelerance $|A|$ [1/kg]")
    ax_top.set_title("ISO 7626-2 Rigid-Mass Calibration Check",
                     pad=12)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper left", fontsize=9)

    # Lower panel: the relative deviation against the same +/-5 % band, where
    # the few-percent tolerance is actually readable.
    ax_bot.axhspan(-100.0 * tol, 100.0 * tol, color=COLOR_SECONDARY, alpha=0.15)
    ax_bot.axhline(0.0, color=COLOR_GRID, linestyle=":", linewidth=1.0)
    ax_bot.semilogx(freq, 100.0 * res.deviation, color=COLOR_PRIMARY,
                    linewidth=2.0)
    ax_bot.semilogx(freq[~within], 100.0 * res.deviation[~within],
                    linestyle="none", marker="o", markersize=4,
                    color=COLOR_SECONDARY)
    ax_bot.set_xlabel(LABEL_FREQ_HZ)
    ax_bot.set_ylabel("Deviation [%]")
    ax_bot.set_xlim(freq[0], freq[-1])
    ax_bot.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)

    format_frequency_axis(ax_top, float(freq[0]), float(freq[-1]))
    format_frequency_axis(ax_bot, float(freq[0]), float(freq[-1]))

    info = [
        "calibration block $m$ = 10 kg",
        r"$|A| = 1/m$ = 0.100 1/kg  (7.5.2)",
        "criterion: agree within ±5 %",
        "high-f drift → attachment error",
    ]
    ax_top.text(0.985, 0.05, "\n".join(info), transform=ax_top.transAxes,
                va="bottom", ha="right", fontsize=10, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "rigid_mass_calibration.svg")
    plt.close()


def generate_junction_plate_geometry(output_dir: str) -> None:
    """A T-junction of heavyweight plates to scale.

    A 140 mm concrete floor ending against a continuous 200 mm wall (the
    T-junction whose perpendicular plates are the identical pair), the
    incident bending wave marked. One concept: the junction the
    transmission coefficients describe.
    """
    print("Generating junction_plate_geometry...")
    from phonometry import plot_junction_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_junction_geometry("T2", 0.14, 0.2, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "junction_plate_geometry.svg")
    plt.close()


def generate_vibration_weighting(output_dir: str) -> None:
    """ISO 8041-1: the whole-body vertical weighting Wk over its band."""
    print("Generating vibration_weighting.png...")
    from phonometry import frequency_weighting

    # A user evaluates the principal ISO 2631-1 weighting Wk on a fine
    # frequency grid across the whole-body band (0,4-100 Hz). The result is the
    # ISO 8041-1 cascade H(f): a gentle +0,5 dB peak near 6 Hz, a band-limiting
    # roll-off below 0,4 Hz and above ~16 Hz.
    freqs = np.geomspace(0.4, 100.0, 240)
    result = frequency_weighting("Wk", freqs)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.semilogx(result.frequencies, result.magnitude_db, color=COLOR_PRIMARY,
                linewidth=1.9, zorder=3)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_title("Whole-body vertical weighting Wk (ISO 8041-1)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Weighting factor [dB]")
    ax.set_xlim(0.4, 100.0)
    ax.set_ylim(-40.0, 5.0)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    # Explicit string labels install a FixedFormatter so the Spanish pass can
    # apply the decimal comma (a log-axis ScalarFormatter would not be caught).
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "vibration_weighting.png")
    plt.close()


def generate_weighted_acceleration(output_dir: str) -> None:
    """ISO 2631-1: measured seat spectrum weighted to a_w (Eq. (9))."""
    print("Generating weighted_acceleration.png...")
    from phonometry import weighted_acceleration

    # A measured vertical seat-pan acceleration spectrum (r.m.s. per one-third
    # octave, m/s^2) from a vehicle seat: energy concentrated in the 2-8 Hz
    # whole-body range. Weighting it with Wk gives the health-relevant a_w.
    freqs = np.array([1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
                      10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 63.0, 80.0])
    accel = np.array([0.18, 0.24, 0.33, 0.46, 0.52, 0.55, 0.48, 0.39, 0.31,
                      0.26, 0.21, 0.17, 0.13, 0.10, 0.078, 0.060, 0.045,
                      0.028, 0.020])
    result = weighted_acceleration(accel, freqs, "Wk")

    positions = np.arange(freqs.size, dtype=float)
    width = 0.4
    _fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.bar(positions - width / 2, result.band_accelerations, width,
           color="#9e9e9e", edgecolor=COLOR_FG, linewidth=0.5,
           label="Unweighted $a_i$", zorder=2)
    ax.bar(positions + width / 2, result.weighted, width, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.5, label="Weighted $W_i\\,a_i$ (Wk)",
           zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Weighted seat acceleration (ISO 2631-1)  $a_w$ = {result.overall:.3f} "
        "m/s²", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("r.m.s. acceleration [m/s²]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "weighted_acceleration.png")
    plt.close()


def generate_daily_vibration_exposure(output_dir: str) -> None:
    """ISO 5349 + Directive 2002/44/EC: A(8) vs the EAV/ELV thresholds."""
    print("Generating daily_vibration_exposure.png...")
    from phonometry import daily_vibration_exposure

    # A forestry worker's day across three chain-saw tasks (the ISO 5349-2
    # Annex E.3 worked example): each task's a_hv and duration give a partial
    # exposure A_i(8); they combine to A(8) = 3,6 m/s^2, assessed against the
    # hand-arm action (2,5) and limit (5,0) values of Directive 2002/44/EC.
    result = daily_vibration_exposure(
        [4.6, 6.0, 3.6],
        [2 * 3600.0, 1 * 3600.0, 2 * 3600.0],
        kind="hav",
        labels=["brush-saw", "felling", "stripping"],
    )

    labels = [*result.labels, "$A(8)$"]
    values = [*result.partials.tolist(), result.a8]
    positions = np.arange(len(values), dtype=float)
    colors = ["#9e9e9e"] * result.partials.size + [COLOR_PRIMARY]
    _fig, ax = plt.subplots(figsize=(9.5, 6.3))
    ax.bar(positions, values, width=0.62, color=colors, edgecolor=COLOR_FG,
           linewidth=0.6, zorder=3)
    eav = result.assessment.action_value
    elv = result.assessment.limit_value
    ax.axhline(eav, color=COLOR_TERTIARY, linestyle="--", linewidth=1.6,
               label=f"EAV = {eav:g} m/s²", zorder=2)
    ax.axhline(elv, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6,
               label=f"ELV = {elv:g} m/s²", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Daily exposure $A(8)$ [m/s²]")
    ax.set_ylim(0.0, elv * 1.2)
    ax.set_title(
        f"Hand-arm daily exposure (ISO 5349 / 2002-44-EC)  $A(8)$ = "
        f"{result.a8:.2f} m/s²", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "daily_vibration_exposure.png")
    plt.close()


def generate_multiple_shock(output_dir: str) -> None:
    """ISO 2631-5: seat-to-spine transmissibility and the injury probability."""
    print("Generating multiple_shock.png...")
    from phonometry import (
        compression_dose,
        dose_from_peaks,
        injury_probability,
        injury_risk,
        seat_to_spine_transfer,
    )
    from phonometry.vibration.human.multiple_shock import (
        MZ_MALE,
        RISK_THRESHOLDS_MALE,
    )

    _fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: seat-to-spine transmissibility |H(f)| (Formula 1). ---
    freq = np.logspace(np.log10(0.5), np.log10(80.0), 400)
    ax_h.plot(freq, np.abs(seat_to_spine_transfer(freq)), color=COLOR_PRIMARY,
              label=r"$|H(f)|$")
    ax_h.axhline(1.0, color=COLOR_GRID, linestyle="--", alpha=0.7)
    ax_h.set_xscale("log")
    ax_h.set_xlabel("Frequency [Hz]")
    ax_h.set_ylabel("Transmissibility  seat $\\rightarrow$ spine")
    ax_h.set_title("Seat-to-spine transfer function", pad=10)
    ax_h.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_h.set_axisbelow(True)
    format_frequency_axis(ax_h, float(freq[0]), float(freq[-1]))
    ax_h.legend(loc="upper right")

    # --- Right: injury probability Pi(R) with the Annex C male example. ---
    grid = np.linspace(0.0, 3.0, 300)
    sexes: tuple[tuple[Literal["male", "female"], str], ...] = (
        ("male", COLOR_PRIMARY),
        ("female", COLOR_SECONDARY),
    )
    for sex, colour in sexes:
        prob = 100.0 * injury_probability(grid, sex=sex)
        ax_r.plot(grid, prob, color=colour, label=f"{sex}")
    # The worked example: five 40 m/s2 peaks, 82 kg male -> R = 1.22.
    sd = compression_dose(dose_from_peaks([40.0] * 5), mz=MZ_MALE)
    r_male = injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="male")
    for level, r_val in zip((10, 50, 90), RISK_THRESHOLDS_MALE):
        ax_r.axhline(level, color="#7f7f7f", linestyle=":", lw=0.8)
        ax_r.plot([r_val, r_val], [0.0, level], color="#7f7f7f", linestyle=":", lw=0.8)
    ax_r.scatter([r_male], [100.0 * injury_probability(r_male, sex="male")],
                 color=COLOR_TERTIARY, marker="*", s=160, zorder=4,
                 label=f"Example  $R$ = {r_male:.2f}")
    ax_r.set_xlabel("Stress variable $R$")
    ax_r.set_ylabel("Probability of lumbar injury [%]")
    ax_r.set_title("Injury probability (Annex C)", pad=10)
    ax_r.set_xlim(left=0.0)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower right")

    plt.tight_layout()
    save_figure(output_dir, "multiple_shock.png")
    plt.close()


def generate_vibration_weighting_family(output_dir: str) -> None:
    """ISO 8041-1: all nine human-vibration weightings on one frequency axis."""
    print("Generating vibration_weighting_family.png...")
    from phonometry import vibration

    # One evaluation per weighting over the band the *family* spans: Wf is a
    # sub-hertz weighting and Wh is still within 40 dB of its peak at 1 kHz,
    # so the axis has to run from 0,05 Hz to 1,5 kHz for the nine curves to be
    # comparable at all. Grouped by the part of the family each belongs to.
    freqs = np.geomspace(0.05, 1500.0, 900)
    curves = (
        ("Wk", "Wk — seat surface, vertical (ISO 2631-1)",
         COLOR_PRIMARY, "-", 2.4),
        ("Wd", "Wd — seat surface, horizontal", COLOR_PRIMARY, "--", 1.6),
        ("Wc", "Wc — backrest, x", COLOR_PRIMARY, (0, (1, 1, 3, 1)), 1.6),
        ("We", "We — rotational (per rad)", "#9467bd", "-.", 1.6),
        ("Wj", "Wj — recumbent, under the head", "#9467bd", ":", 1.8),
        ("Wm", "Wm — building occupants, all axes (ISO 2631-2)",
         "#ff7f0e", "-", 1.8),
        ("Wb", "Wb — rail ride comfort, vertical (ISO 2631-4)",
         "#ff7f0e", "--", 1.6),
        ("Wf", "Wf — motion sickness, vertical", COLOR_TERTIARY, "-", 2.0),
        ("Wh", "Wh — hand-transmitted, all three axes (ISO 5349-1)",
         COLOR_SECONDARY, "-", 2.4),
    )

    _fig, ax = plt.subplots(figsize=(11, 6.8))
    for name, label, colour, style, width in curves:
        factors = np.asarray(vibration.weighting_factors(name, freqs))
        db = 20.0 * np.log10(np.maximum(factors, 1e-9))
        ax.semilogx(freqs, db, color=colour, linestyle=style, linewidth=width,
                    label=label, zorder=3)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.4, zorder=1)

    # The band each part of the family is tabulated over, drawn above the
    # curves: ISO 2631-1 Table 3 (0,5-80 Hz, band-limited 0,4/100 Hz),
    # its Wf column (0,1-0,5 Hz, band-limited 0,08/0,63 Hz) and ISO 5349-1
    # Table A.2 (6,3-1250 Hz, band-limited 6,31/1258,9 Hz).
    bands = (
        (0.1, 0.5, 11.0, COLOR_TERTIARY, "Wf: 0.1–0.5 Hz"),
        (0.5, 80.0, 7.5, COLOR_PRIMARY, "whole body: 0.5–80 Hz"),
        (6.3, 1250.0, 4.0, COLOR_SECONDARY, "hand-arm Wh: 6.3–1250 Hz"),
    )
    for low, high, level, colour, label in bands:
        ax.plot([low, high], [level, level], color=colour, linewidth=3.0,
                solid_capstyle="butt", zorder=4)
        for edge in (low, high):
            ax.plot([edge, edge], [level - 1.1, level + 1.1], color=colour,
                    linewidth=1.4, zorder=4)
        ax.text(np.sqrt(low * high), level + 1.6, label, fontsize=8.5,
                color=colour, ha="center", va="bottom")

    ax.annotate("Wf peaks at 0.17 Hz:\nmotion sickness is sub-hertz",
                xy=(0.150, -1.6), xytext=(0.052, 2.6), fontsize=9,
                color=COLOR_TERTIARY, ha="left",
                arrowprops={"arrowstyle": "->", "lw": 0.9,
                            "color": COLOR_TERTIARY})
    ax.annotate("Wd peaks 2.5 octaves below Wk:\nthe body is more compliant\n"
                "horizontally at low frequency",
                xy=(1.09, 0.1), xytext=(2.1, -46.0), fontsize=9,
                color=COLOR_PRIMARY, ha="left",
                arrowprops={"arrowstyle": "->", "lw": 0.9,
                            "color": COLOR_PRIMARY})

    ax.set_title("The nine human-vibration weightings (ISO 8041-1 Table 3)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Weighting factor [dB]")
    ax.set_xlim(0.05, 1500.0)
    ax.set_ylim(-70.0, 16.0)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1250])
    ax.set_xticklabels(["0.05", "0.1", "0.5", "1", "5", "10", "50", "100",
                        "500", "1250"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8.5, ncol=2, framealpha=0.9)
    plt.tight_layout()
    save_figure(output_dir, "vibration_weighting_family.png")
    plt.close()


def _shock_ride_record(fs: float, duration: float) -> np.ndarray:
    """A synthetic seated off-road record: 4.5 Hz ride plus five impacts.

    The record the dose-measure figure and the guide's snippet both run on.
    Its crest factor is above 9 and both ISO 2631-1 clause 6.3.3 ratios are
    exceeded, which is the case the basic r.m.s. method cannot describe.
    """
    time = np.arange(int(duration * fs)) / fs
    rng = np.random.default_rng(3)
    signal = (0.35 * np.sin(2.0 * np.pi * 4.5 * time)
              + 0.10 * rng.standard_normal(time.size))
    for start, amplitude in ((2.6, 9.0), (6.1, 14.0), (9.4, 6.0),
                             (13.8, 18.0), (17.2, 11.0)):
        mask = time >= start
        signal[mask] += amplitude * np.exp(-28.0 * (time[mask] - start)) \
            * np.sin(2.0 * np.pi * 8.0 * (time[mask] - start))
    return signal


def generate_shock_dose_measures(output_dir: str) -> None:
    """ISO 2631-1: one shock-laden record read by r.m.s., MTVV and VDV."""
    print("Generating shock_dose_measures.png...")
    from phonometry import vibration

    fs, duration = 200.0, 20.0
    time = np.arange(int(duration * fs)) / fs
    raw = _shock_ride_record(fs, duration)
    weighted = np.asarray(vibration.apply_weighting(raw, fs, "Wk"))
    a_w = float(np.sqrt(np.mean(weighted ** 2)))
    running = np.asarray(vibration.running_rms(weighted, fs,
                                               integration_time=1.0))
    mtvv = float(vibration.mtvv(weighted, fs))
    vdv = float(vibration.vibration_dose_value(weighted, fs))
    # The running fourth-power accumulation whose end point is the VDV, drawn
    # against a_w t^(1/4), the value the basic method would predict for it.
    fourth = np.cumsum(weighted ** 4) / fs
    running_vdv = fourth ** 0.25
    basic_vdv = a_w * time ** 0.25

    _fig, (ax_t, ax_r, ax_v) = plt.subplots(
        3, 1, sharex=True, figsize=(11, 9.2),
        gridspec_kw={"height_ratios": (1.15, 1.0, 1.0)})

    ax_t.plot(time, raw, color=COLOR_MUTED, linewidth=0.7,
              label="$a_z(t)$, unweighted", zorder=2)
    ax_t.plot(time, weighted, color=COLOR_PRIMARY, linewidth=1.0,
              label="$a_w(t)$, Wk-weighted", zorder=3)
    ax_t.set_ylabel("acceleration [m/s²]")
    ax_t.set_title(
        "(a)  A seated off-road record: 4.5 Hz ride plus five impacts",
        fontsize=10, loc="left", pad=6)
    ax_t.legend(loc="upper right", fontsize=9)

    ax_r.plot(time, running, color=COLOR_PRIMARY, linewidth=1.4,
              label="running r.m.s., 1 s (Eq. (3))", zorder=3)
    ax_r.axhline(a_w, color=COLOR_MUTED, linestyle="--", linewidth=1.6,
                 label=f"$a_w$ = {a_w:.2f} m/s² (Eq. (1))", zorder=2)
    peak_index = int(np.argmax(running))
    ax_r.plot([time[peak_index]], [mtvv], marker="o", markersize=7,
              color=COLOR_SECONDARY, zorder=4,
              label=f"MTVV = {mtvv:.2f} m/s² (Eq. (4))")
    ax_r.annotate(rf"$\mathrm{{MTVV}}/a_w$ = {mtvv / a_w:.2f}   (> 1.5)",
                  xy=(time[peak_index], mtvv),
                  xytext=(time[peak_index] - 6.5, mtvv * 0.78), fontsize=9,
                  color=COLOR_SECONDARY,
                  arrowprops={"arrowstyle": "->", "lw": 0.9,
                              "color": COLOR_SECONDARY})
    ax_r.set_ylabel("r.m.s. [m/s²]")
    ax_r.set_title("(b)  The 1 s running r.m.s., whose maximum is the MTVV",
                   fontsize=10, loc="left", pad=6)
    ax_r.legend(loc="upper left", fontsize=9)

    ax_v.plot(time, running_vdv, color=COLOR_PRIMARY, linewidth=1.6,
              label=r"$\left(\int_0^t a_w^4\,dt\right)^{1/4}$", zorder=3)
    ax_v.plot(time, basic_vdv, color=COLOR_MUTED, linestyle="--",
              linewidth=1.6, label=r"$a_w\,t^{1/4}$ (the basic method)",
              zorder=2)
    ax_v.plot([duration], [vdv], marker="o", markersize=7,
              color=COLOR_SECONDARY, zorder=4,
              label=f"VDV = {vdv:.2f} m/s$^{{1.75}}$ (Eq. (5))")
    ax_v.annotate(
        rf"$\mathrm{{VDV}}/(a_w T^{{1/4}})$ = {vdv / (a_w * duration ** 0.25):.2f}"
        "   (> 1.75)",
        xy=(duration, vdv), xytext=(7.0, vdv * 0.55), fontsize=9,
        color=COLOR_SECONDARY,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_SECONDARY})
    ax_v.set_ylabel("dose [m/s$^{1.75}$]")
    ax_v.set_xlabel("Time [s]")
    ax_v.set_title("(c)  The fourth-power accumulation, whose end point is "
                   "the VDV", fontsize=10, loc="left",
                   pad=6)
    ax_v.legend(loc="upper left", fontsize=9)

    for axis in (ax_t, ax_r, ax_v):
        axis.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        axis.set_axisbelow(True)
    ax_v.set_xlim(0.0, duration)
    plt.tight_layout()
    save_figure(output_dir, "shock_dose_measures.png")
    plt.close()


def generate_hav_vwf_lifetime(output_dir: str) -> None:
    """ISO 5349-1 Annex C: years to a 10 % prevalence of white finger."""
    print("Generating hav_vwf_lifetime.png...")
    from phonometry import vibration

    # Formula (C.1) over the exposure range a workplace assessment produces,
    # with the four tabulated points it interpolates between (Table C.1) and
    # the Directive's two thresholds read off the same curve.
    a8 = np.geomspace(1.0, 40.0, 400)
    years = np.array([vibration.hav_vwf_lifetime_years(float(v)) for v in a8])
    table_a8 = np.array([26.0, 14.0, 7.0, 3.7])
    table_years = np.array([1.0, 2.0, 4.0, 8.0])

    _fig, ax = plt.subplots(figsize=(10, 6.4))
    # Outside 1-8 years the curve is an extrapolation of Table C.1, not an
    # interpolation of it: shade what the table does not underwrite.
    wash = theme_fill(COLOR_MUTED, ax)
    ax.axhspan(0.3, 1.0, color=wash, zorder=0)
    ax.axhspan(8.0, 60.0, color=wash, zorder=0)
    ax.text(1.15, 22.0, "extrapolation beyond Table C.1", fontsize=9,
            color=COLOR_FG, alpha=0.75, va="center")
    ax.text(1.15, 0.42, "extrapolation beyond Table C.1", fontsize=9,
            color=COLOR_FG, alpha=0.75, va="center")
    ax.loglog(a8, years, color=COLOR_PRIMARY, linewidth=2.0, zorder=3,
              label=r"$D_y = 31.8\,A(8)^{-1.06}$ (Eq. (C.1))")
    ax.loglog(table_a8, table_years, linestyle="none", marker="o",
              markersize=8, color=COLOR_SECONDARY, zorder=4,
              label="Table C.1: 26 / 14 / 7 / 3.7 m/s²")
    for value, label, colour in ((2.5, "EAV 2.5", COLOR_TERTIARY),
                                 (5.0, "ELV 5.0", COLOR_SECONDARY)):
        ax.axvline(value, color=colour, linestyle="--", linewidth=1.5,
                   zorder=2)
        ax.text(value * 1.04, 0.36,
                f"{label} m/s²\n{vibration.hav_vwf_lifetime_years(value):.1f}"
                " years", fontsize=9, color=colour, va="bottom")

    ax.set_title("Group-mean years to a 10 % prevalence of vibration white "
                 "finger (ISO 5349-1 Annex C)", pad=12)
    ax.set_xlabel("Daily exposure $A(8)$ [m/s²]")
    ax.set_ylabel("Exposure duration $D_y$ [years]")
    ax.set_xlim(1.0, 40.0)
    ax.set_ylim(0.3, 60.0)
    ax.set_xticks([1, 2, 2.5, 5, 10, 20, 40])
    ax.set_xticklabels(["1", "2", "2.5", "5", "10", "20", "40"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 20, 50])
    ax.set_yticklabels(["0.5", "1", "2", "4", "8", "20", "50"])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "hav_vwf_lifetime.png")
    plt.close()


def _positive_peak_indices(response: np.ndarray) -> np.ndarray:
    """Index of the maximum of every positive run between zero crossings."""
    positive = response > 0.0
    edges = np.flatnonzero(np.diff(positive.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    stops = np.r_[edges + 1, response.size]
    return np.array([int(a + np.argmax(response[a:b]))
                     for a, b in zip(starts, stops) if positive[a]])


def generate_spinal_response_peaks(output_dir: str) -> None:
    """ISO 2631-5 clause 5: the spinal response in time and its counted peaks."""
    print("Generating spinal_response_peaks.png...")
    from phonometry import vibration

    # A synthetic off-road seat record: a 3.6 Hz ride oscillation, four
    # impacts of clearly different severity, and a 0.4 s free fall before the
    # largest one (the case clause 5.1.3 NOTE 2 sets the 0,01 Hz high pass
    # for). 256 samples per second is the rate clause 5.1.2 asks for.
    fs, duration = 256.0, 8.0
    time = np.arange(int(duration * fs)) / fs
    rng = np.random.default_rng(11)
    seat = (1.5 * np.sin(2.0 * np.pi * 3.6 * time)
            + 0.45 * rng.standard_normal(time.size))
    for start, amplitude in ((1.15, 26.0), (3.05, 8.0), (5.60, 55.0),
                             (6.95, 15.0)):
        mask = time >= start
        seat[mask] += amplitude * np.exp(-20.0 * (time[mask] - start)) \
            * np.sin(2.0 * np.pi * 5.5 * (time[mask] - start))
    free_fall = (time >= 5.18) & (time < 5.58)
    seat[free_fall] = -9.81

    response = np.asarray(vibration.spinal_response(seat, fs))
    peaks = np.asarray(vibration.response_peaks(response))
    indices = _positive_peak_indices(response)
    shares = 100.0 * peaks ** 6 / float(np.sum(peaks ** 6))
    dose = float(vibration.dose_from_peaks(peaks))

    _fig, (ax_t, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(11, 7.6))

    ax_t.plot(time, seat, color=COLOR_MUTED, linewidth=0.9,
              label="$a_z(t)$, conditioned seat acceleration", zorder=2)
    ax_t.plot(time, response, color=COLOR_PRIMARY, linewidth=1.5,
              label="$A_z(t)$, spinal response (Formula 2)", zorder=3)
    ax_t.axhline(-9.81, color=COLOR_SECONDARY, linestyle=":", linewidth=1.2,
                 zorder=1)
    ax_t.annotate("0.4 s of free fall at $-1\\,g$:\nthe 0.01 Hz high pass of "
                  "5.1.3 keeps it",
                  xy=(5.30, -9.81), xytext=(2.25, 13.0), fontsize=9,
                  color=COLOR_SECONDARY,
                  arrowprops={"arrowstyle": "->", "lw": 0.9,
                              "color": COLOR_SECONDARY})
    ax_t.set_ylabel("acceleration [m/s²]")
    ax_t.set_title("(a)  The seat-to-spine filter turns an impact into a "
                   "ringing response", fontsize=10,
                   loc="left", pad=6)
    ax_t.legend(loc="upper left", fontsize=9)

    ax_p.plot(time, response, color=COLOR_PRIMARY, linewidth=1.2, zorder=2)
    ax_p.axhline(0.0, color=COLOR_FG, linewidth=0.9, alpha=0.6, zorder=1)
    ax_p.plot(time[indices], peaks, linestyle="none", marker="o",
              markersize=5, color=COLOR_SECONDARY, zorder=4,
              label=f"{peaks.size} counted positive peaks $A_{{z,i}}$")
    largest = int(np.argmax(peaks))
    ax_p.axhline(peaks[largest] / 3.0, color=COLOR_TERTIARY, linestyle="--",
                 linewidth=1.4, zorder=3,
                 label="a third of the largest peak: below this, "
                       "no contribution")
    for rank in np.argsort(peaks)[::-1][:3]:
        ax_p.annotate(f"{peaks[rank]:.1f}  ({shares[rank]:.1f} %)",
                      xy=(time[indices[rank]], peaks[rank]),
                      xytext=(time[indices[rank]] + 0.18, peaks[rank] + 1.6),
                      fontsize=9, color=COLOR_FG)
    ax_p.set_ylabel("$A_z$ [m/s²]")
    ax_p.set_xlabel("Time [s]")
    ax_p.set_title(
        f"(b)  Each peak's share of $\\sum A_{{z,i}}^6$ — dose $D_z$ = "
        f"{dose:.1f} m/s²", fontsize=10, loc="left",
        pad=6)
    ax_p.legend(loc="upper left", fontsize=9)

    for axis in (ax_t, ax_p):
        axis.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        axis.set_axisbelow(True)
    ax_p.set_xlim(0.0, duration)
    plt.tight_layout()
    save_figure(output_dir, "spinal_response_peaks.png")
    plt.close()


def generate_junction_kij_thickness(output_dir: str) -> None:
    """Wave-approach Kij versus the plate thickness ratio (Hopkins Eq. 5.116)."""
    print("Generating junction_kij_thickness...")
    from phonometry import junction_transmission, wave_vibration_reduction_index

    # Concrete plates (cL = 3200 m/s, rho = 2400 kg/m3): plate 1 fixed at
    # 100 mm, plate 2 swept from 50 mm to 400 mm.
    h1, cl, rho = 0.1, 3200.0, 2400.0
    ratios = np.linspace(0.5, 4.0, 36)
    curves: dict[str, list[float]] = {
        "X corner": [], "X straight": [], "T-junction (1) corner": [],
        "L corner": [],
    }
    for ratio in ratios:
        h2 = h1 * float(ratio)
        res_x = junction_transmission("X", h1, cl, rho * h1, h2, cl, rho * h2)
        assert res_x.straight_average is not None
        curves["X corner"].append(res_x.corner_reduction_index)
        curves["X straight"].append(float(wave_vibration_reduction_index(
            res_x.straight_average, res_x.critical_frequency2)))
        res_t = junction_transmission("T1", h1, cl, rho * h1, h2, cl, rho * h2)
        curves["T-junction (1) corner"].append(res_t.corner_reduction_index)
        res_l = junction_transmission("L", h1, cl, rho * h1, h2, cl, rho * h2)
        curves["L corner"].append(res_l.corner_reduction_index)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    styles = [("-", COLOR_PRIMARY), ("--", COLOR_PRIMARY),
              ("-", COLOR_SECONDARY), ("-", COLOR_TERTIARY)]
    for (label, values), (ls, color) in zip(curves.items(), styles):
        ax.plot(ratios, values, ls, color=color, linewidth=2.0, label=label)
    # The identical-plate X-junction: Kij = 10 log10 12 + 5 log10(fc2/1000).
    res_eq = junction_transmission("X", h1, cl, rho * h1, h1, cl, rho * h1)
    ax.scatter([1.0], [res_eq.corner_reduction_index], color=COLOR_FG, s=70,
               zorder=6, label=r"identical plates ($\tau = 1/12$)")

    ax.set_xticks(np.arange(0.5, 4.01, 0.5))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.5, 4.01, 0.5)])
    ax.set_xlabel("Thickness ratio $h_2/h_1$")
    ax.set_ylabel("Vibration reduction index $K_{ij}$ [dB]")
    ax.set_title("Wave-Approach Junction $K_{ij}$ (Hopkins Eq. 5.116)",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        r"$K_{ij} = 10\,\log_{10}(1/\bar{\tau}) + 5\,\log_{10}(f_{c2}/1000)$",
        "concrete, plate 1 fixed at 100 mm",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "junction_kij_thickness.svg")
    plt.close()


def generate_bearing_fault_envelope(output_dir: str) -> None:
    """Predicted bearing fault lines over a measured envelope spectrum."""
    print("Generating bearing_fault_envelope...")
    from phonometry import bearing_fault_frequencies, envelope_spectrum, noise_signal

    # Norton problem 8.5 geometry: fifteen rollers, 34 mm pitch diameter,
    # 6 mm rollers, 12.96 deg contact angle, 2000 r/min.
    faults = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                       contact_angle_deg=12.96)
    bpfo, fs_shaft = faults["BPFO"], faults.shaft_rate

    # A spalled outer race: one impact per BPFO period ringing a 3 kHz housing
    # resonance, load-modulated once per revolution, buried in broadband noise.
    fs, seconds = 20000.0, 2.0
    t = np.arange(int(fs * seconds)) / fs
    impacts = np.zeros_like(t)
    for k in range(int(seconds * bpfo)):
        idx = round(k / bpfo * fs)
        if idx < impacts.size:
            impacts[idx] = 1.0 + 0.35 * np.cos(2.0 * np.pi * fs_shaft * idx / fs)
    tau = np.arange(int(0.004 * fs)) / fs
    ring = np.exp(-tau / 6.0e-4) * np.sin(2.0 * np.pi * 3000.0 * tau)
    x = np.convolve(impacts, ring)[: t.size] * 0.6
    x += 0.35 * np.sin(2.0 * np.pi * fs_shaft * t)          # residual unbalance
    x += noise_signal(fs, seconds, color="white", rms=0.25, seed=17)

    res = envelope_spectrum(x, fs, band=(2000.0, 4000.0))
    keep = res.frequencies <= 4.6 * bpfo
    freq, amp = res.frequencies[keep], res.amplitude[keep]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(freq, amp, color=COLOR_PRIMARY, linewidth=1.1,
            label="envelope spectrum of the 2-4 kHz band")
    top = float(amp.max())
    for order in range(1, 5):
        line = order * bpfo
        ax.axvline(line, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2,
                   alpha=0.85, zorder=2,
                   label="predicted BPFO and harmonics" if order == 1 else None)
        ax.annotate(f"{order}×BPFO" if order > 1 else "BPFO",
                    xy=(line, 1.02 * top), xytext=(3, 0),
                    textcoords="offset points", rotation=90, fontsize=8.5,
                    color=COLOR_SECONDARY, ha="left", va="bottom")
    for name, colour in (("BPFI", COLOR_TERTIARY), ("BSF", "#9467bd")):
        ax.axvline(faults[name], color=colour, linestyle=":", linewidth=1.3,
                   alpha=0.9, zorder=2, label=f"predicted {name}")
    ax.axvline(fs_shaft, color=COLOR_FG, linestyle="-.", linewidth=1.0,
               alpha=0.6, zorder=2, label="shaft rate")

    ax.set_xlim(0.0, 4.6 * bpfo)
    ax.set_ylim(0.0, 1.55 * top)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Envelope amplitude")
    ax.set_title("Bearing Fault Lines on a Measured Envelope Spectrum",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        r"15 rollers, $D$ = 34 mm, $d$ = 6 mm, $\varphi$ = 12.96°, 2000 r/min",
        f"BPFO = {bpfo:.0f} Hz, BPFI = {faults['BPFI']:.0f} Hz",
        "the envelope lines fall on BPFO, not on BPFI: outer-race spall",
    ]
    ax.text(0.985, 0.47, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "bearing_fault_envelope.svg")
    plt.close()


def generate_experimental_sea_clf(output_dir: str) -> None:
    """Measured and predicted coupling loss factors of a plate junction."""
    print("Generating experimental_sea_clf...")
    from phonometry import (
        coupling_loss_factor,
        cylindrical_shell_modal_density,
        flat_plate_modal_density,
        plate_bending_stiffness,
        point_connection_coupling_loss_factor,
        power_injection_clf,
        right_angle_transmission_coefficient,
    )
    from phonometry.vibration.structural.point_mobility import plate_bending_wave_speed

    rho, nu, young = 2700.0, 0.33, 7.1e10
    c_l = math.sqrt(young / (rho * (1.0 - nu**2)))
    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])

    # Two aluminium plates at right angles: 3 mm x 2.5 m x 1.2 m coupled to
    # 5.5 mm x 2.0 m x 1.2 m along the 1.2 m edge (Norton problem 6.13).
    h1, h2, area1, length = 0.003, 0.0055, 2.5 * 1.2, 1.2
    tau = right_angle_transmission_coefficient(
        h1, h2, density1=rho, density2=rho, wave_speed1=c_l, wave_speed2=c_l)
    c_b = plate_bending_wave_speed(bands, plate_bending_stiffness(young, h1, nu),
                                   rho * h1)
    welded = np.array([float(coupling_loss_factor(tau, 2.0 * c, length, f, area1))
                       for c, f in zip(c_b, bands, strict=True)])
    bolted = point_connection_coupling_loss_factor(
        bands, 12, thickness1=h1, thickness2=h2, surface_density1=rho * h1,
        surface_density2=rho * h2, wave_speed1=c_l, wave_speed2=c_l,
        plate_area1=area1)

    # The satellite platform and cylinder of Norton problem 6.10, inverted from
    # its measured energies in the 500 Hz octave.
    t_p, t_c, radius, cyl_len = 0.005, 0.003, 0.75, 2.0
    area_c = 2.0 * math.pi * radius * cyl_len
    area_p = 3.5 * 3.0 - math.pi * radius**2
    sea = power_injection_clf(
        500.0,
        rho * t_p * area_p * 0.0272**2,
        rho * t_c * area_c * 0.0132**2,
        4.4e-3, 2.4e-3,
        flat_plate_modal_density(area_p, t_p, c_l),
        float(cylindrical_shell_modal_density(500.0, area_c, t_c, radius, c_l)[0]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    ax = axes[0]
    x = _band_index_axis(ax, bands, fontsize=9)
    ax.semilogy(x, welded, "-o", color=COLOR_PRIMARY, linewidth=2.0,
                markersize=5, label=r"welded line junction $\eta_{12}$")
    ax.semilogy(x, bolted, "-s", color=COLOR_SECONDARY, linewidth=2.0,
                markersize=5, label=r"12 bolts, point connections $\eta_{12}$")
    ax.axhline(1.0e-2, color=COLOR_FG, linestyle="--", linewidth=1.2,
               alpha=0.7, label=r"internal loss factor $\eta_1$")
    ax.set_ylabel("Coupling loss factor")
    ax.set_title("Predicted: Weld Against Bolts", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    labels = [r"$\eta_1$", r"$\eta_2$", r"$\eta_{12}$", r"$\eta_{21}$"]
    values = [4.4e-3, 2.4e-3, float(sea.coupling_loss_factor12[0]),
              float(sea.coupling_loss_factor21[0])]
    # COLOR_MUTED, not COLOR_GRID: this is a de-emphasised *bar*, and the
    # grid colour is tuned to disappear into the page it is drawn on.
    colours = [COLOR_FG, COLOR_MUTED, COLOR_PRIMARY, COLOR_SECONDARY]
    ax.bar(labels, values, color=colours, edgecolor=COLOR_FG, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 1.0e-2)
    ax.set_ylabel("Loss factor")
    ax.set_title("Measured: Power Injection, 500 Hz Octave",
                 pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", which="both")
    ax.set_axisbelow(True)
    for label, value in zip(labels, values, strict=True):
        ax.annotate(_sci_math(value), xy=(label, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=COLOR_FG)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        "platform driven, cylinder driven only through the joints",
        f"input power = {float(sea.input_power[0]):.2f} W",
        r"coupling stays well below the damping: valid SEA",
    ]
    ax.text(0.98, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    fig.suptitle("Coupling Loss Factors: Prediction and Power Injection",
                 fontsize=13)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(output_dir, "experimental_sea_clf.svg")
    plt.close()


def generate_mobility_result_lines(output_dir: str) -> None:
    """ISO 7626 driving-point mobility with its stiffness and mass lines."""
    print("Generating mobility_result_lines...")
    from phonometry import sdof_mobility_result

    m, k, c = 2.0, 8000.0, 5.0
    f = np.logspace(np.log10(0.5), np.log10(200.0), 400)
    res = sdof_mobility_result(f, mass=m, stiffness=k, damping=c)
    w = 2.0 * np.pi * f
    f0 = float(res.frequencies[int(np.argmax(res.magnitude))])

    fig, (ax, low) = plt.subplots(
        2, 1, figsize=(10, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0]})
    ax.loglog(f, res.magnitude, color=COLOR_PRIMARY, linewidth=2.2,
              label="driving-point $|Y(f)|$")
    ax.loglog(f, w / k, ":", color=COLOR_SECONDARY, linewidth=1.6,
              label=r"stiffness line $\omega/k$")
    ax.loglog(f, 1.0 / (w * m), ":", color=COLOR_TERTIARY, linewidth=1.6,
              label=r"mass line $1/(\omega m)$")
    ax.axhline(1.0 / c, color=COLOR_GRID, linestyle="--", linewidth=1.2)
    ax.scatter([f0], [1.0 / c], color=COLOR_FG, s=60, zorder=6,
               label="peak $|Y| = 1/c$ (damping)")

    ax.set_xlim(float(f[0]), float(f[-1]))
    ax.set_ylim(2e-4, 0.6)
    ax.set_ylabel("Mobility $|Y|$ [m/(N·s)]")
    ax.set_title("Reading a Driving-Point Mobility (ISO 7626-1)",
                 pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        r"below $f_0$: stiffness-controlled, $|Y| \sim \omega/k$",
        r"above $f_0$: mass-controlled, $|Y| \sim 1/(\omega m)$",
        rf"$f_0$ = {f0:.1f} Hz,  $1/c$ = {1.0 / c:.2f} m/(N·s)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    # The phase is the other half of the reading, and the one the +-90 deg
    # bound of ISO 7626-2 A.4 turns into a validity check on the setup.
    phase = np.degrees(np.asarray(res.phase, dtype=np.float64))
    low.axhspan(-90.0, 90.0, color=theme_fill(COLOR_PRIMARY, low), zorder=0)
    low.semilogx(f, phase, color=COLOR_PRIMARY, linewidth=2.2,
                 label="phase of $Y(f)$")
    low.axhline(0.0, color=COLOR_GRID, linestyle="--", linewidth=1.2)
    low.scatter([f0], [0.0], color=COLOR_FG, s=55, zorder=6,
                label="0° at the resonance: $Y$ is real, $|Y| = 1/c$")
    low.set_ylim(-115.0, 115.0)
    low.set_yticks([-90, -45, 0, 45, 90])
    low.set_ylabel("Phase [degrees]")
    low.set_xlabel(LABEL_FREQ_HZ)
    format_frequency_axis(low, float(f[0]), float(f[-1]))
    low.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    low.set_axisbelow(True)
    low.legend(loc="lower left", fontsize=9)
    low.text(0.985, 0.93, "a driving-point FRF never leaves ±90°\n"
             "(ISO 7626-2, A.4)", transform=low.transAxes, va="top",
             ha="right", fontsize=9.5, color=COLOR_FG)
    fig.align_ylabels()
    plt.tight_layout()
    save_figure(output_dir, "mobility_result_lines.svg")


class _LineSpectrum:
    """A synthesised narrow-band spectrum for a ``FaultFrequencyResult`` overlay.

    ``FaultFrequencyResult.plot(spectrum=…)`` accepts anything exposing
    ``frequencies`` and ``amplitude``; the fault families of Norton Section 8.4
    are recognised by the *pattern* of their lines, so the panels below place
    Gaussian lines at the frequencies the library itself predicts and let the
    result draw its own prediction on top.
    """

    def __init__(self, freq: np.ndarray, floor: float = 0.004) -> None:
        self.frequencies = freq
        rng = np.random.default_rng(11)
        self.amplitude = floor * (0.6 + rng.random(freq.size))

    def add(self, centre: float, height: float, width: float = 2.2) -> None:
        self.amplitude = self.amplitude + height * np.exp(
            -0.5 * ((self.frequencies - centre) / width) ** 2)


def _relabel_spectrum(ax: Axes, title: str) -> None:
    """Say what the drawn curve is: these panels are ordinary spectra.

    ``FaultFrequencyResult.plot`` labels its curve "envelope spectrum",
    which is true of the bearing route and false of the gear, motor and fan
    families drawn here, so the label, the y axis and the title are set to
    what the panel actually shows.
    """
    curve = ax.get_lines()[0]
    curve.set_label("vibration spectrum")
    ax.set_ylabel("Spectrum amplitude")
    ax.set_title(title, pad=10, fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5)


def generate_machine_fault_families(output_dir: str) -> None:
    """The three fault families of Norton 8.4 as patterns, not as numbers."""
    print("Generating machine_fault_families...")
    from phonometry import (
        blade_pass_frequencies,
        gear_mesh_frequencies,
        induction_motor_frequencies,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))

    # --- (a) and (b): the same gear pair with two different faults ----------
    gear = gear_mesh_frequencies(1500.0, 28, harmonics=3, sidebands=2)
    shaft = gear.shaft_rate                                    # 25 Hz
    gmf = gear["GMF"]                                          # 700 Hz
    cases = (
        ("Localised fault: one chipped tooth", (1.0, 0.40, 0.16),
         (0.13, 0.09), axes[0][0]),
        ("Distributed wear: every tooth", (1.0, 0.78, 0.62), (0.46, 0.33),
         axes[0][1]),
    )
    for title, harm, side, ax in cases:
        freq = np.linspace(0.0, 2400.0, 4800)
        spec = _LineSpectrum(freq)
        for order in range(1, 4):                              # shaft orders
            spec.add(order * shaft, 0.05 / order)
        for k, height in enumerate(harm, start=1):
            spec.add(k * gmf, height)
            for order, ratio in enumerate(side, start=1):
                spec.add(k * gmf - order * shaft, height * ratio)
                spec.add(k * gmf + order * shaft, height * ratio)
        gear.within(1.0, 2400.0).plot(spectrum=spec, ax=ax)
        _relabel_spectrum(ax, title)
        ax.set_ylim(0.0, 1.25)
        ax.annotate(
            "sidebands at $\\pm f_s$ = 25 Hz:\n"
            + ("low and flat" if "Localised" in title
               else "tall groups, and the\nhigher harmonics lift"),
            xy=(gmf + 2.0 * shaft, 0.34 if "Localised" in title else 0.62),
            xytext=(910.0, 0.80), fontsize=9, color=COLOR_FG,
            arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})

    # --- (c): the motor family, three decades of amplitude ------------------
    ax = axes[1][0]
    motor = induction_motor_frequencies(3600.0, 6, 60, slip=0.0)
    freq = np.linspace(0.0, 3900.0, 7800)
    spec = _LineSpectrum(freq, floor=2.0e-5)
    for name, height in (("1x", 1.0), ("2x", 0.30), ("2fe", 0.06)):
        spec.add(motor[name], height, width=3.0)
    spec.add(motor["fsh"], 1.6e-3, width=3.0)                  # rotor slot
    for sign in (-1.0, 1.0):                                   # ± shaft rate
        spec.add(motor["fsh"] + sign * motor["1x"], 5.5e-4, width=3.0)
    motor.plot(spectrum=spec, ax=ax)
    _relabel_spectrum(ax, "Induction motor: 1x, 2x, 2fe and the rotor-slot "
                      "family")
    ax.set_yscale("log")
    ax.set_ylim(1.0e-5, 3.0)
    ax.annotate("rotor-slot harmonic\nwith $\\pm f_s$ sidebands:\n"
                "the spacing is the diagnosis",
                xy=(motor["fsh"], 1.6e-3), xytext=(1700.0, 0.02),
                fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})

    # --- (d): the ducted fan, blade rate against its lobe patterns ----------
    ax = axes[1][1]
    fan = blade_pass_frequencies(3500.0, 6, harmonics=1, n_vanes=4)
    freq = np.linspace(0.0, 420.0, 3200)
    spec = _LineSpectrum(freq, floor=0.006)
    spec.add(fan.shaft_rate, 0.10, width=0.8)
    spec.add(fan["BPF"], 1.0, width=0.8)
    spec.add(fan["lobe n=1 m=2"], 0.62, width=0.8)
    spec.add(fan["lobe n=1 m=10"], 0.07, width=0.8)
    fan.plot(spectrum=spec, ax=ax)
    _relabel_spectrum(ax, "Ducted fan: the blade rate and its rotating lobe "
                      "patterns")
    ax.set_ylim(0.0, 1.3)
    ax.annotate("$m_L$ = 10 turns at 35 Hz,\nbelow the shaft: weak",
                xy=(fan["lobe n=1 m=10"], 0.12), xytext=(64.0, 0.72),
                fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})
    ax.annotate("$m_L$ = 2 turns at 175 Hz,\n3x the shaft speed: strong",
                xy=(fan["lobe n=1 m=2"], 0.68), xytext=(196.0, 0.96),
                fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})

    fig.suptitle("Fault Families Are Recognised by Their Pattern",
                 fontsize=13)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_figure(output_dir, "machine_fault_families.svg")
    plt.close()


def generate_envelope_chain_steps(output_dir: str) -> None:
    """The three steps of the envelope route, on one bearing record."""
    print("Generating envelope_chain_steps...")
    from scipy import signal as sp_signal

    from phonometry import bearing_fault_frequencies, envelope_spectrum, noise_signal

    faults = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                       contact_angle_deg=12.96)
    bpfo, fs_shaft = faults["BPFO"], faults.shaft_rate

    # The record of the opening figure: impacts at BPFO ringing a 3 kHz
    # housing resonance, load-modulated once per revolution, under noise.
    fs, seconds = 20000.0, 2.0
    t = np.arange(int(fs * seconds)) / fs
    impacts = np.zeros_like(t)
    for k in range(int(seconds * bpfo)):
        idx = round(k / bpfo * fs)
        if idx < impacts.size:
            impacts[idx] = 1.0 + 0.35 * np.cos(2.0 * np.pi * fs_shaft * idx / fs)
    tau = np.arange(int(0.004 * fs)) / fs
    ring = np.exp(-tau / 6.0e-4) * np.sin(2.0 * np.pi * 3000.0 * tau)
    x = np.convolve(impacts, ring)[: t.size] * 0.6
    x += 0.35 * np.sin(2.0 * np.pi * fs_shaft * t)
    x += noise_signal(fs, seconds, color="white", rms=0.25, seed=17)

    band = (2000.0, 4000.0)
    res = envelope_spectrum(x, fs, band=band)
    sos = sp_signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
    narrow = np.asarray(sp_signal.sosfiltfilt(sos, x), dtype=np.float64)

    window = slice(0, int(0.05 * fs))                 # the first 50 ms
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 10.4))

    axes[0].plot(t[window] * 1e3, x[window], color=COLOR_MUTED, linewidth=0.9)
    axes[0].set_ylabel("raw record")
    axes[0].set_title("1. As recorded: noise and unbalance, no visible impacts",
                      fontsize=11, pad=8)

    axes[1].plot(t[window] * 1e3, narrow[window], color=COLOR_PRIMARY,
                 linewidth=0.9)
    axes[1].set_ylabel("2-4 kHz band")
    axes[1].set_title("2. Band-passed on the 3 kHz housing resonance: "
                      "the impact train", fontsize=11,
                      pad=8)
    top = float(np.max(np.abs(narrow[window])))
    axes[1].set_ylim(-1.35 * top, 1.35 * top)
    for k in range(11):                       # one impact per BPFO period
        axes[1].axvline(1e3 * k / bpfo, color=COLOR_MUTED, linewidth=0.8,
                        alpha=0.55, zorder=0)
    t0 = 1e3 / bpfo * 3.0
    axes[1].annotate("", xy=(t0, 1.06 * top), xytext=(t0 + 1e3 / bpfo,
                     1.06 * top),
                     arrowprops={"arrowstyle": "<->", "color": COLOR_SECONDARY})
    axes[1].text(t0 + 0.5e3 / bpfo, 1.14 * top,
                 rf"$1/\mathrm{{BPFO}}$ = {1e3 / bpfo:.2f} ms", ha="center",
                 fontsize=9, color=COLOR_SECONDARY)

    axes[2].plot(res.times[window] * 1e3, res.envelope[window],
                 color=COLOR_SECONDARY, linewidth=1.0)
    axes[2].set_ylabel("envelope")
    axes[2].set_xlabel("Time [ms]")
    axes[2].set_title("3. Hilbert envelope: one pulse per impact",
                      fontsize=11, pad=8)

    keep = res.frequencies <= 4.6 * bpfo
    axes[3].plot(res.frequencies[keep], res.amplitude[keep],
                 color=COLOR_PRIMARY, linewidth=1.1)
    peak = float(np.max(res.amplitude[keep]))
    for order in range(1, 5):
        axes[3].axvline(order * bpfo, color=COLOR_SECONDARY, linestyle="--",
                        linewidth=1.2, alpha=0.85,
                        label="predicted BPFO and harmonics"
                        if order == 1 else None)
    for name, colour in (("BPFI", COLOR_TERTIARY), ("BSF", "#9467bd")):
        axes[3].axvline(faults[name], color=colour, linestyle=":",
                        linewidth=1.3, alpha=0.9, label=f"predicted {name}")
    axes[3].set_xlim(0.0, 4.6 * bpfo)
    axes[3].set_ylim(0.0, 1.35 * peak)
    axes[3].set_xlabel(LABEL_FREQ_HZ)
    axes[3].set_ylabel("envelope spectrum")
    axes[3].set_title("4. Its spectrum: the period has become a line at BPFO",
                      fontsize=11, pad=8)
    axes[3].legend(loc="upper right", fontsize=8.5)

    for ax in axes:
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("What Each Step of the Envelope Route Does to the Signal",
                 fontsize=13)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    save_figure(output_dir, "envelope_chain_steps.svg")
    plt.close()


def generate_infinite_mobilities(output_dir: str) -> None:
    """The point mobilities a real structure has, beside the SDOF reference."""
    print("Generating infinite_mobilities...")
    from phonometry import sdof_mobility_result, vibration

    freq = np.logspace(np.log10(0.5), np.log10(2000.0), 600)

    # A 140 mm concrete slab (E = 30 GPa, nu = 0.2, rho = 2400 kg/m3), a
    # 100 x 200 mm steel beam, and a 2 cm2 steel strut in longitudinal motion.
    b_plate = vibration.plate_bending_stiffness(3.0e10, 0.14, 0.2)
    plate = vibration.infinite_plate_point_mobility(freq, b_plate, 336.0)
    b_beam = 2.1e11 * 0.1 * 0.2**3 / 12.0
    beam = vibration.infinite_beam_point_mobility(freq, b_beam, 7800.0 * 0.02)
    rod = vibration.longitudinal_rod_mobility(7800.0, 5100.0, 2.0e-4)
    sdof = sdof_mobility_result(freq, mass=2.0, stiffness=8000.0, damping=5.0)

    fig, (ax, low) = plt.subplots(
        2, 1, figsize=(10, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0]})
    curves = (
        (sdof.magnitude, np.degrees(np.asarray(sdof.phase)), COLOR_MUTED,
         "SDOF resonator (section 2)", "--"),
        (plate.magnitude, np.degrees(np.asarray(plate.phase)), COLOR_PRIMARY,
         "infinite plate, 140 mm concrete", "-"),
        (beam.magnitude, np.degrees(np.asarray(beam.phase)), COLOR_SECONDARY,
         "infinite beam, 100 × 200 mm steel", "-"),
        (np.full_like(freq, rod), np.zeros_like(freq), COLOR_TERTIARY,
         "steel strut, longitudinal", ":"),
    )
    for mag, phase, colour, label, style in curves:
        ax.loglog(freq, mag, style, color=colour, linewidth=2.0, label=label)
        low.semilogx(freq, phase, style, color=colour, linewidth=2.0)

    ax.set_ylim(5.0e-7, 0.6)
    ax.set_ylabel("Mobility $|Y|$ [m/(N·s)]")
    ax.set_title("Point Mobilities of Infinite Structures (Cremer Table 5.1)",
                 pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    info = [
        r"plate:  $Y = 1/(8\sqrt{B^{\prime} m^{\prime\prime}})$, real and flat",
        r"beam:   $Y = (1-\mathrm{j})/(4 m^{\prime} c_B)$, $\propto f^{-1/2}$",
        r"rod:    $Y = 1/(\rho c_L S)$, real and flat",
        f"plate $|Y|$ = {_sci_math(float(plate.magnitude[0]))} m/(N·s)",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    low.axhline(0.0, color=COLOR_GRID, linestyle="--", linewidth=1.1)
    low.axhline(-45.0, color=COLOR_GRID, linestyle=":", linewidth=1.1)
    low.set_ylim(-115.0, 115.0)
    low.set_yticks([-90, -45, 0, 45, 90])
    low.set_ylabel("Phase [degrees]")
    low.set_xlabel(LABEL_FREQ_HZ)
    low.set_xlim(float(freq[0]), float(freq[-1]))
    format_frequency_axis(low, float(freq[0]), float(freq[-1]))
    low.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    low.set_axisbelow(True)
    fig.align_ylabels()
    plt.tight_layout()
    save_figure(output_dir, "infinite_mobilities.svg")
    plt.close()


def generate_mobility_random_error(output_dir: str) -> None:
    """ISO 7626-2 Annex A: how many averages the 5 % criterion costs."""
    print("Generating mobility_random_error...")
    from phonometry import random_error_percent

    averages = np.unique(np.round(
        np.logspace(np.log10(2.0), np.log10(1000.0), 260)))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    palette = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_MUTED,
               COLOR_FG)
    for coherence, colour in zip((0.5, 0.7, 0.8, 0.9, 0.95), palette,
                                 strict=True):
        error = np.array([float(random_error_percent(coherence, int(n)))
                          for n in np.round(averages)], dtype=np.float64)
        ax.loglog(averages, error, color=colour, linewidth=2.0,
                  label=rf"$\gamma^2 = {coherence}$")
        needed = (1.0 - coherence) / (2.0 * coherence * 0.05**2)
        ax.scatter([needed], [5.0], color=colour, s=36, zorder=6)
        ax.annotate(f"{needed:.0f}", xy=(needed, 5.0), xytext=(0, -14),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=colour)
    ax.axhline(5.0, color=COLOR_FG, linestyle="--", linewidth=1.4,
               label="the Annex A criterion, 5 %")
    ax.scatter([75.0], [float(random_error_percent(0.8, 75))], marker="*",
               s=190, color=COLOR_FG, zorder=7,
               label=r"the standard's own example: $\gamma^2 = 0.8$, "
                     "$n$ = 75 → 4.08 %")

    ax.set_xlabel("Number of averaged spectra $n$")
    ax.set_ylabel(r"Normalized random error $\varepsilon$ [%]")
    ax.set_title("How Many Averages the 5 % Criterion Costs (ISO 7626-2, "
                 "Annex A)", pad=12)
    ax.set_xlim(2.0, 1000.0)
    ax.set_ylim(1.0, 60.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.05, "the marked $n$ is what each coherence needs to reach "
            "5 %: 11 averages at 0.95, 200 at 0.5.\nFixing the measurement is "
            "cheaper than averaging through it.", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9.5, color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "mobility_random_error.svg")
    plt.close()
