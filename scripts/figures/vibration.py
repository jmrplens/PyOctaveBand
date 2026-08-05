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

import matplotlib.pyplot as plt
import numpy as np

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

    ax.set_xlabel("Incidence angle [degrees]")
    ax.set_ylabel(r"Transmission coefficient $\tau$")
    ax.set_title(
        "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)",
        fontweight="bold", pad=12,
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "X-junction: 100 mm / 200 mm concrete",
        f"chi = {res.chi:.3f},  psi = {res.psi:.3f}",
        f"corner avg = {res.corner_average:.4f}",
        f"straight avg = {res.straight_average:.4f}",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
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
        (np.abs(h) * k, COLOR_PRIMARY, "Receptance $|H|$ (× k)"),
        (np.abs(y) * k / w0, COLOR_SECONDARY, r"Mobility $|Y|$ (× k/$\omega_0$)"),
        (np.abs(a) * k / w0**2, COLOR_TERTIARY, r"Accelerance $|A|$ (× k/$\omega_0^2$)"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for mag, color, label in curves:
        ax.loglog(freq, mag, color=color, linewidth=2.0, label=label)
    ax.axvline(f0, color=COLOR_GRID, linestyle="--", linewidth=1.2,
               label="resonance $f_0$")

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized FRF magnitude")
    ax.set_title("ISO 7626-1 Mechanical Mobility FRFs", fontweight="bold", pad=12)
    ax.set_xlim(freq[0], freq[-1])
    format_frequency_axis(ax, float(freq[0]), float(freq[-1]))
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        "SDOF: m = 2 kg, k = 8000 N/m, c = 5 N.s/m",
        "H = 1/(k - w^2 m + j w c)",
        "Y = j w H,   A = -w^2 H  (Table 1)",
        f"f0 = {f0:.1f} Hz,  |Y(f0)| = 1/c",
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

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(freq, transfer_stiffness_level(k_true), color=COLOR_PRIMARY,
                linewidth=2.2, label=r"true $L_k$ of $k_{2,1}=k+j\omega c$")
    ax.semilogx(freq, transfer_stiffness_level(k_indirect), color=COLOR_SECONDARY,
                linewidth=2.0, linestyle="--",
                label=r"indirect method $-(2\pi f)^2 m_2 T$")
    ax.axvline(f0, color=COLOR_GRID, linestyle=":", linewidth=1.2,
               label="resonance $f_0$")
    ax.axvspan(freq[0], 3.0 * f0, color=theme_fill(COLOR_FG, ax), zorder=0)

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Transfer stiffness level $L_k$ [dB re 1 N/m]")
    ax.set_title("ISO 10846 Dynamic Transfer Stiffness", fontweight="bold", pad=12)
    ax.set_xlim(freq[0], freq[-1])
    format_frequency_axis(ax, float(freq[0]), float(freq[-1]))
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "Kelvin-Voigt: k = 1 MN/m, c = 120 N.s/m",
        f"blocking mass m2 = 8 kg,  f0 = {f0:.1f} Hz",
        "indirect valid for T << 1  (f >> f0)",
        "shaded: T not small -> method invalid",
    ]
    ax.text(0.985, 0.05, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
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
                     fontweight="bold", pad=12)
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
        "calibration block m = 10 kg",
        "|A| = 1/m = 0.100 1/kg  (7.5.2)",
        "criterion: agree within +/- 5 %",
        "high-f drift -> attachment error",
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
                 fontweight="bold", pad=12)
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
        "m/s$^2$", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("r.m.s. acceleration [m/s$^2$]")
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

    labels = [*result.labels, "A(8)"]
    values = [*result.partials.tolist(), result.a8]
    positions = np.arange(len(values), dtype=float)
    colors = ["#9e9e9e"] * result.partials.size + [COLOR_PRIMARY]
    _fig, ax = plt.subplots(figsize=(9.5, 6.3))
    ax.bar(positions, values, width=0.62, color=colors, edgecolor=COLOR_FG,
           linewidth=0.6, zorder=3)
    eav = result.assessment.action_value
    elv = result.assessment.limit_value
    ax.axhline(eav, color=COLOR_TERTIARY, linestyle="--", linewidth=1.6,
               label=f"EAV = {eav:g} m/s$^2$", zorder=2)
    ax.axhline(elv, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6,
               label=f"ELV = {elv:g} m/s$^2$", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Daily exposure A(8) [m/s$^2$]")
    ax.set_ylim(0.0, elv * 1.2)
    ax.set_title(
        f"Hand-arm daily exposure (ISO 5349 / 2002-44-EC)  A(8) = "
        f"{result.a8:.2f} m/s$^2$", fontweight="bold", pad=12)
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
    ax_h.set_title("Seat-to-spine transfer function", fontweight="bold", pad=10)
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
    ax_r.set_title("Injury probability (Annex C)", fontweight="bold", pad=10)
    ax_r.set_xlim(left=0.0)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower right")

    plt.tight_layout()
    save_figure(output_dir, "multiple_shock.png")
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
               zorder=6, label="identical plates (τ = 1/12)")

    ax.set_xticks(np.arange(0.5, 4.01, 0.5))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.5, 4.01, 0.5)])
    ax.set_xlabel("Thickness ratio h2/h1")
    ax.set_ylabel("Vibration reduction index $K_{ij}$ [dB]")
    ax.set_title("Wave-Approach Junction $K_{ij}$ (Hopkins Eq. 5.116)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "Kij = 10 log10(1/τ̄) + 5 log10(fc2/1000)",
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
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        "15 rollers, D = 34 mm, d = 6 mm, φ = 12.96°, 2000 r/min",
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
    ax.set_title("Predicted: Weld Against Bolts", fontweight="bold", pad=10)
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
                 fontweight="bold", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", which="both")
    ax.set_axisbelow(True)
    for label, value in zip(labels, values, strict=True):
        ax.annotate(f"{value:.2e}", xy=(label, value), xytext=(0, 4),
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
                 fontweight="bold", fontsize=13)
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

    _fig, ax = plt.subplots(figsize=(10, 6.2))
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
    ax.set_ylim(1e-5, 1.0)
    format_frequency_axis(ax, float(f[0]), float(f[-1]))
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Mobility $|Y|$ [m/(N·s)]")
    ax.set_title("Reading a Driving-Point Mobility (ISO 7626-1)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        "below f0: stiffness-controlled, |Y| ~ ω/k",
        "above f0: mass-controlled, |Y| ~ 1/(ωm)",
        f"f0 = {f0:.1f} Hz,  1/c = {1.0 / c:.2f} m/(N·s)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "mobility_result_lines.svg")
