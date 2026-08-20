#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the materials guides: absorbers, diffusers and impedance.

How a surface answers a sound field: porous and resonant absorbers, the
impedance and transmission tubes that measure them, the transfer-matrix and
Biot models behind them, and the diffusers (Schroeder wells and metadiffuser
alike) that scatter rather than absorb. Everything here is embedded by a page
under ``materials/``.
"""

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import (
    format_frequency_axis,
    theme_fill,
    theme_fill_alpha,
    theme_line,
)

from .i18n import _LANG, _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
)

if TYPE_CHECKING:
    from phonometry.materials.absorbers.layered import Layer


def generate_dynamic_stiffness(output_dir: str) -> None:
    """EN 29052-1 floating-floor natural frequency f0(s') for typical floors."""
    print("Generating dynamic_stiffness...")
    from phonometry import materials

    s_mn = np.logspace(np.log10(2.0), np.log10(100.0), 300)  # MN/m3
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Two typical floating-floor masses per unit area (light vs heavy screed).
    for m, color, label in (
        (40.0, COLOR_SECONDARY, r"$m^{\prime}$ = 40 kg/m²"),
        (120.0, COLOR_PRIMARY, r"$m^{\prime}$ = 120 kg/m²"),
    ):
        f0 = np.asarray(materials.natural_frequency(s_mn * 1e6, m), dtype=float)
        ax.plot(s_mn, f0, color=color, linewidth=2.2, label=label)

    # A worked design point: s' = 10 MN/m3 on the 120 kg/m2 floor.
    s0, m0 = 10.0, 120.0
    f00 = float(materials.natural_frequency(s0 * 1e6, m0))
    ax.scatter(
        [s0],
        [f00],
        color=COLOR_TERTIARY,
        s=90,
        zorder=6,
        label=f"design point ({s0:g} MN/m³, {f00:.0f} Hz)",
    )
    ax.plot([s0, s0], [0, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)
    ax.plot([s_mn[0], s0], [f00, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dynamic stiffness per unit area $s^{\prime}$ [MN/m³]")
    ax.set_ylabel(r"Natural frequency $f_0$ [Hz]")
    ax.set_title("EN 29052-1 Floating-Floor Resonance", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)

    info = [
        r"$f_0 = (1/2\pi)\sqrt{s^{\prime}/m^{\prime}}$  (Formula 2)",
        r"$s^{\prime} = s^{\prime}_\mathrm{t} + s^{\prime}_\mathrm{a}$  (clause 8.2)",
        (
            r"$s^{\prime}_\mathrm{t} = 4\pi^2 m^{\prime}_\mathrm{t} f_\mathrm{r}^2$"
            r"  (Formula 4)"
        ),
        (
            r"$s^{\prime}_\mathrm{a} = p_0/(d\,\varepsilon) \approx 111/d$"
            r" MN/m³  (NOTE)"
        ),
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "dynamic_stiffness.svg")
    plt.close()


def generate_floating_floor_transmissibility(output_dir: str) -> None:
    """The rig's peak, and what the installed floor does either side of f0."""
    print("Generating floating_floor_transmissibility...")
    from phonometry import building, materials

    # Left: the response the operator reads off the rig, and what over-driving
    # does to it. Right: the ISO 12354-2 improvement law the installed floor
    # obeys, against the ideal mass-spring transmissibility that explains the
    # amplification the law truncates to zero below f0.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5.8))

    def sdof(freqs: np.ndarray, f_n: float, eta: float) -> np.ndarray:
        r = freqs / f_n
        return np.abs(1.0 / (1.0 - r**2 + 1j * eta * r))

    rig = np.linspace(8.0, 70.0, 600)
    ax_l.plot(
        rig,
        20.0 * np.log10(sdof(rig, 25.0, 0.14)),
        color=COLOR_PRIMARY,
        linewidth=2.0,
        zorder=3,
        label=r"small excitation: $f_\mathrm{r}$ = 25.0 Hz",
    )
    ax_l.plot(
        rig,
        20.0 * np.log10(sdof(rig, 22.0, 0.22)),
        color=COLOR_SECONDARY,
        linewidth=1.8,
        linestyle="--",
        zorder=3,
        label="over-driven: peak lower and at 22 Hz",
    )
    ax_l.axvline(25.0, color=COLOR_MUTED, linestyle=":", linewidth=1.2, zorder=1)
    ax_l.annotate(
        "this peak is the whole measurement\n"
        r"(Formula 4), extrapolated to $F \to 0$",
        (28.0, 6.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax_l.set_xlim(rig[0], rig[-1])
    ax_l.set_xlabel(LABEL_FREQ_HZ)
    ax_l.set_ylabel("Load-plate response [dB re static]")
    ax_l.set_title(r"On the rig: reading $f_\mathrm{r}$", pad=12)
    ax_l.legend(loc="upper right", fontsize=9)
    ax_l.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_l.set_axisbelow(True)

    # The installed floor. The worked determination of the guide gives
    # s' = 10.49 MN/m3 on a 120 kg/m2 screed; halving s' moves f0 by sqrt(2).
    bands = np.array(
        [
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
        ],
        float,
    )
    f0_hard = float(materials.natural_frequency(10.49e6, 120.0))
    f0_soft = float(materials.natural_frequency(5.0e6, 120.0))
    fine = np.logspace(np.log10(30.0), np.log10(3150.0), 600)
    ax_r.axvspan(
        fine[0],
        np.sqrt(2.0) * f0_hard,
        color=theme_fill(COLOR_SECONDARY, ax_r),
        zorder=0,
    )
    ax_r.semilogx(
        fine,
        -20.0 * np.log10(sdof(fine, f0_hard, 0.15)),
        color=COLOR_MUTED,
        linewidth=1.5,
        linestyle=":",
        zorder=2,
        label="ideal mass-spring isolation",
    )
    for f0, color, marker, label in (
        (
            f0_hard,
            COLOR_PRIMARY,
            "o",
            rf"$s^{{\prime}}$ = 10.5 MN/m³, $f_0$ = {f0_hard:.0f} Hz",
        ),
        (
            f0_soft,
            COLOR_TERTIARY,
            "s",
            rf"$s^{{\prime}}$ = 5.0 MN/m³, $f_0$ = {f0_soft:.0f} Hz",
        ),
    ):
        law = building.floating_floor_improvement_spectrum(
            bands, resonance_frequency=f0
        )
        ax_r.semilogx(
            bands,
            np.asarray(law.improvement),
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.9,
            zorder=3,
            label=label,
        )
    ax_r.axhline(0.0, color=COLOR_FG, linewidth=1.0, zorder=1)
    ax_r.annotate(
        f"amplification below\n$\\sqrt{{2}}\\,f_0$ = {np.sqrt(2) * f0_hard:.0f} Hz",
        (34.0, -14.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax_r.annotate(
        r"$30\,\mathrm{lg}(f/f_0)$: 30.8 dB" "\nagainst 35.6 dB at 500 Hz",
        (560.0, 12.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax_r.set_ylim(-22.0, 60.0)
    ax_r.set_xlabel(LABEL_FREQ_HZ)
    ax_r.set_ylabel("Improvement of impact insulation [dB]")
    ax_r.set_title("Installed: only well above $f_0$", pad=12)
    format_frequency_axis(ax_r, 30.0, 3150.0)
    ax_r.legend(loc="upper left", fontsize=9)
    ax_r.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_r.set_axisbelow(True)
    fig.align_ylabels((ax_l, ax_r))
    plt.tight_layout()
    save_figure(output_dir, "floating_floor_transmissibility.svg")
    plt.close()


def generate_enclosed_gas_stiffness(output_dir: str) -> None:
    """Below about 20 mm the pore air, not the frame, is the spring."""
    print("Generating enclosed_gas_stiffness...")
    from phonometry import materials

    # The clause 8.2 sum against the loaded thickness d, with the frame term
    # held at the worked determination's 4.94 MN/m3 and the enclosed-gas term
    # from Formula (7). The right-hand axis turns the installed s' into the
    # natural frequency of a 120 kg/m2 screed, which is what the reader wants.
    d_mm = np.logspace(np.log10(5.0), np.log10(60.0), 300)
    s_frame = np.full_like(d_mm, 4.935)
    s_gas = np.array(
        [
            float(materials.enclosed_gas_stiffness(thickness=d * 1e-3, porosity=0.9))
            / 1e6
            for d in d_mm
        ]
    )
    s_total = s_frame + s_gas

    _fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.loglog(
        d_mm,
        s_frame,
        color=COLOR_PRIMARY,
        linewidth=1.9,
        linestyle="--",
        zorder=3,
        label=r"$s^{\prime}_\mathrm{t}$, frame (Formula 4)",
    )
    ax.loglog(
        d_mm,
        s_gas,
        color=COLOR_TERTIARY,
        linewidth=1.9,
        linestyle="-.",
        zorder=3,
        label=r"$s^{\prime}_\mathrm{a}$, enclosed gas"
        r" (Formula 7, $\varepsilon = 0.9$)",
    )
    ax.loglog(
        d_mm,
        s_total,
        color=COLOR_SECONDARY,
        linewidth=2.4,
        zorder=4,
        label=r"installed $s^{\prime} = s^{\prime}_\mathrm{t}"
        r" + s^{\prime}_\mathrm{a}$ (clause 8.2)",
    )
    ax.scatter([20.0], [10.49], color=COLOR_FG, s=70, zorder=6)
    ax.annotate(
        "the worked determination:\n$d$ = 20 mm, 4.94 + 5.56 = 10.49",
        (21.0, 10.49),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="center",
    )
    ax.set_xlabel("Loaded thickness $d$ [mm]")
    ax.set_ylabel("Dynamic stiffness per unit area [MN/m³]")
    ax.set_title("The gas spring takes over as the layer gets thinner", pad=12)
    ax.set_xlim(d_mm[0], d_mm[-1])
    ax.set_ylim(1.0, 40.0)
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticks([5, 10, 20, 40, 60])
    ax.set_yticks([1, 2, 5, 10, 20, 40])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Right-hand axis: the same s' read as the natural frequency of a
    # 120 kg/m2 floating screed, which is the number the design turns on.
    ax_f = ax.twinx()
    ax_f.set_yscale("log")
    ax_f.set_ylim(
        float(materials.natural_frequency(1.0e6, 120.0)),
        float(materials.natural_frequency(40.0e6, 120.0)),
    )
    ax_f.set_ylabel("$f_0$ of a 120 kg/m² screed [Hz]")
    ax_f.yaxis.set_major_formatter(ScalarFormatter())
    ax_f.yaxis.set_minor_formatter(NullFormatter())
    ax_f.set_yticks([15, 20, 30, 40, 60, 90])

    ax.text(
        0.015,
        0.05,
        r"clause 8.2:  $r \geq 100$ kPa·s/m²"
        r"  →  $s^{\prime} = s^{\prime}_\mathrm{t}$"
        "\n"
        r"$10 \leq r < 100$"
        r"  →  $s^{\prime} = s^{\prime}_\mathrm{t} + s^{\prime}_\mathrm{a}$"
        "\n"
        r"$r < 10$  →  $s^{\prime} = s^{\prime}_\mathrm{t}$ only if"
        r" $s^{\prime}_\mathrm{a}$ is negligible",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "enclosed_gas_stiffness.svg")
    plt.close()


def generate_absorption_uncertainty(output_dir: str) -> None:
    """ISO 12999-2 absorption-coefficient uncertainty: alpha_s with a +/-U ribbon."""
    print("Generating absorption_uncertainty...")
    from phonometry import materials

    # The standard's worked Example (Table 4): a measured sound absorption
    # coefficient alpha_s per one-third-octave band and its reproducibility
    # expanded uncertainty at k = 2.
    freqs = np.array(
        [
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
        dtype=float,
    )
    alpha_s = np.array(
        [
            0.33,
            0.35,
            0.39,
            0.38,
            0.37,
            0.36,
            0.36,
            0.36,
            0.43,
            0.49,
            0.58,
            0.63,
            0.68,
            0.71,
            0.73,
            0.75,
            0.77,
            0.79,
            0.81,
            0.81,
        ]
    )
    res = materials.sound_absorption_coefficient_uncertainty(
        alpha_s, freqs, confidence=0.95
    )
    u = res.expanded_uncertainty

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(
        x,
        alpha_s - u,
        alpha_s + u,
        color=COLOR_TERTIARY,
        alpha=0.22,
        zorder=0,
        label=r"$\pm U$ ($k$ = 2), reproducibility",
    )
    ax.plot(
        x,
        alpha_s,
        "-",
        color=COLOR_PRIMARY,
        linewidth=2.4,
        marker="o",
        markersize=6,
        zorder=5,
        label=r"$\alpha_\mathrm{s}$ (ISO 354)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound absorption coefficient")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("ISO 12999-2 Sound Absorption Coefficient Uncertainty", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        r"$\sigma_\mathrm{R} = m\,\alpha_\mathrm{s} + n$  (Table 1)",
        r"$U = k\,u$,  $k$ = 2  (95 %)",
    ]
    ax.text(
        0.985,
        0.03,
        "\n".join(info),
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    plt.tight_layout()
    save_figure(output_dir, "absorption_uncertainty.png")
    plt.close()


def generate_absorption_rating(output_dir: str) -> None:
    """ISO 11654 alpha_w: practical curve, shifted reference, deviations (Annex A.2)."""
    print("Generating absorption_rating.png...")
    from phonometry import materials

    # ISO 11654:1997 Annex A.2 worked example -> alpha_w = 0.60(M).
    alpha_p = [0.35, 1.00, 0.65, 0.60, 0.55]
    result = materials.weighted_absorption(alpha_p)
    freqs = np.asarray(result.band_centers, dtype=float)
    measured = np.asarray(result.measured, dtype=float)
    shifted = np.asarray(result.shifted_reference, dtype=float)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(
        freqs,
        measured,
        shifted,
        where=(measured < shifted).tolist(),
        interpolate=True,
        color=COLOR_SECONDARY,
        alpha=0.25,
        zorder=1,
        label="Unfavourable deviations",
    )
    ax.semilogx(
        freqs,
        shifted,
        marker="s",
        color=COLOR_FG,
        linewidth=1.6,
        linestyle="--",
        markersize=5,
        zorder=3,
        label="Shifted reference curve (ISO 11654)",
    )
    ax.semilogx(
        freqs,
        measured,
        marker="o",
        color=COLOR_PRIMARY,
        linewidth=1.8,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=4,
        label=r"Practical absorption $\alpha_\mathrm{p}$",
    )

    # alpha_w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.alpha_w, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(
        f"$\\alpha_\\mathrm{{w}}$ = {result.rating_label}",
        xy=(500, result.alpha_w),
        xytext=(600, result.alpha_w - 0.16),
        fontsize=12,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    # Placed low-left, clear of the practical curve that peaks top-centre.
    for dy, text in (
        (0.30, f"Reference curve shifted by {result.shift:.2f}"),
        (
            0.23,
            (
                f"Sum of unfavourable deviations = {result.unfavourable_sum:.2f}"
                f"  (limit 0.10)"
            ),
        ),
        (
            0.16,
            (
                f"Absorption class {result.absorption_class}  "
                f"(shape indicator: {result.shape_indicator or 'none'})"
            ),
        ),
    ):
        ax.text(
            0.03,
            dy,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color=COLOR_FG,
        )

    ax.set_title(
        "ISO 11654 Weighted Sound Absorption Coefficient (Annex A.2 example)", pad=12
    )
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound absorption coefficient")
    ax.set_xscale("log")
    ax.set_xlim(220, 4600)
    ax.set_ylim(0.0, 1.08)
    from matplotlib.ticker import NullFormatter

    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(["250", "500", "1k", "2k", "4k"], fontsize=9)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "absorption_rating.png")
    plt.close()


def generate_airflow_resistance(output_dir: str) -> None:
    """ISO 9053-1 static method: dp vs u, through-origin quadratic fit, R_s at 0.5 mm/s."""
    print("Generating airflow_resistance.png...")
    from phonometry import materials

    # A porous specimen (area 100 mm dia, 50 mm thick) measured stepwise. The
    # pressure drop is slightly super-linear in velocity; the through-origin
    # quadratic fit dp = a*u + b*u**2 recovers R_s = a at the reference 0.5 mm/s.
    area = float(np.pi) * (0.05**2)  # 100 mm diameter cell
    r_s_true, curvature = 1.6e4, 4.0e5  # Pa*s/m, Pa*s2/m2
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3  # m/s
    dp = r_s_true * u + curvature * u**2
    result = materials.static_airflow_resistance(u, dp, area=area, thickness=0.05)

    u_fit = np.linspace(0.0, 13e-3, 200)
    dp_fit = result.linear_coefficient * u_fit + result.quadratic_coefficient * u_fit**2

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(
        u_fit * 1e3,
        dp_fit,
        color=COLOR_PRIMARY,
        linewidth=1.8,
        zorder=2,
        label=r"Through-origin quadratic fit  $\Delta p = a\,u + b\,u^2$",
    )
    ax.plot(
        u * 1e3,
        dp,
        "o",
        color=COLOR_SECONDARY,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=4,
        label="Measured pressure drop",
    )

    u_ref = result.evaluation_velocity
    ax.axvline(u_ref * 1e3, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(
        u_ref * 1e3,
        result.pressure_drop,
        "D",
        color=COLOR_TERTIARY,
        markersize=9,
        zorder=6,
    )
    ax.annotate(
        "evaluation at 0.5 mm/s",
        xy=(u_ref * 1e3, result.pressure_drop),
        xytext=(2.0, result.pressure_drop + 40),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    for dy, text in (
        (
            0.97,
            (
                f"Specific airflow resistance $R_\\mathrm{{s}}$ = "
                f"{result.specific_resistance:.0f} Pa·s/m"
            ),
        ),
        (0.90, f"Airflow resistivity $\\sigma$ = {result.resistivity:.0f} Pa·s/m²"),
        (
            0.83,
            (
                f"Linear term $a$ = {result.linear_coefficient:.0f} Pa·s/m"
                f"  (= $R_\\mathrm{{s}}$ at $u \\to 0$)"
            ),
        ),
    ):
        ax.text(
            0.03,
            dy,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color=COLOR_FG,
        )

    ax.set_title("ISO 9053-1 Static-Method Airflow Resistance", pad=12)
    ax.set_xlabel("Linear airflow velocity $u$ [mm/s]")
    ax.set_ylabel(r"Pressure drop $\Delta p$ [Pa]")
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "airflow_resistance.png")
    plt.close()


def generate_impedance_tube(output_dir: str) -> None:
    """ISO 10534-1 standing-wave-ratio method: alpha and |r| vs level difference."""
    print("Generating impedance_tube.png...")
    from phonometry import materials

    # Level difference between pressure maximum and minimum (Eq. 15): a large dL
    # means a strong reflection (little absorption); dL -> 0 is a perfect absorber.
    level_diff = np.linspace(0.5, 40.0, 300)
    swr = np.array([materials.standing_wave_ratio_from_level(dl) for dl in level_diff])
    alpha = np.array([materials.standing_wave_absorption(s) for s in swr])
    r_mag = np.array([materials.standing_wave_reflection_magnitude(s) for s in swr])

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(
        level_diff,
        alpha,
        color=COLOR_PRIMARY,
        linewidth=2.0,
        zorder=3,
        label=r"Absorption coefficient $\alpha = 1 - |r|^2$",
    )
    ax.set_xlabel(
        r"Standing-wave level difference $L_{\mathrm{max}} - L_{\mathrm{min}}$ [dB]"
    )
    ax.set_ylabel(r"Sound absorption coefficient $\alpha$")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 40.0)

    ax_r = ax.twinx()
    ax_r.plot(
        level_diff,
        r_mag,
        color=COLOR_SECONDARY,
        linewidth=1.8,
        linestyle="--",
        zorder=2,
        label="Reflection factor magnitude $|r|$",
    )
    ax_r.set_ylabel("Reflection factor magnitude $|r|$")
    ax_r.set_ylim(0.0, 1.02)

    # Mark the didactic anchor: dL = 9.54 dB -> s = 3 -> |r| = 0.5 -> alpha = 0.75.
    dl_anchor = 20.0 * float(np.log10(3.0))
    ax.plot(dl_anchor, 0.75, "D", color=COLOR_TERTIARY, markersize=9, zorder=6)
    # Text sits in the lens that opens between the diverging alpha and |r| curves.
    ax.annotate(
        r"$s = 3$ → $|r| = 0.5$ → $\alpha = 0.75$",
        xy=(dl_anchor, 0.75),
        xytext=(15.0, 0.44),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "lw": 1.0},
    )

    ax.set_title("ISO 10534-1 Standing-Wave-Ratio Method", pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    save_figure(output_dir, "impedance_tube.png")
    plt.close()


def generate_porous_absorber_designs(output_dir: str) -> None:
    """Multilayer absorber packages predicted by the transfer-matrix method.

    Normal-incidence absorption of four 50-mm-deep constructions built from
    the same porous model (Miki) and the resonant sheet layers: a plain
    porous layer, a Maa microperforated panel over a cavity, a perforated
    panel over porous + air, and a limp membrane over a porous-filled
    cavity. One concept: the same 2x2 layer matrices predict broadband and
    resonant absorbers alike.
    """
    print("Generating porous_absorber_designs...")
    import warnings as _warnings

    from phonometry import materials

    f = np.logspace(np.log10(50.0), np.log10(5000.0), 500)
    with _warnings.catch_warnings():
        # The 50 Hz decade end sits below the published Miki fit range on
        # purpose (the figure shows the bass behaviour of the resonators).
        _warnings.simplefilter("ignore", materials.PorousAbsorberWarning)
        med = materials.miki(f, 20000.0)
        med_light = materials.miki(f, 10000.0)
        cases: list[tuple[str, list[Layer], str, str]] = [
            (
                r"Porous layer 50 mm ($\sigma$ = 20 kPa·s/m²)",
                [materials.PorousLayer(0.05, med)],
                COLOR_PRIMARY,
                "-",
            ),
            (
                "Microperforated panel + 48 mm cavity",
                [
                    materials.MicroperforatedPlateLayer(0.5e-3, 0.15e-3, 0.008),
                    materials.AirLayer(0.048),
                ],
                COLOR_SECONDARY,
                "-",
            ),
            (
                "Perforated panel 6 mm + porous 25 mm + air",
                [
                    materials.PerforatedPlateLayer(0.006, 0.0025, 0.05),
                    materials.PorousLayer(0.025, med),
                    materials.AirLayer(0.019),
                ],
                COLOR_TERTIARY,
                "-",
            ),
            (
                "Membrane 2 kg/m² + air + porous 38 mm",
                [
                    materials.MembraneLayer(2.0),
                    materials.AirLayer(0.01),
                    materials.PorousLayer(0.038, med_light),
                ],
                "#9467bd",
                "-",
            ),
        ]
        _fig, ax = plt.subplots(figsize=(10, 6.2))
        for label, layers, color, ls in cases:
            res = materials.layered_absorber(f, layers)
            ax.semilogx(f, res.absorption, ls, color=color, linewidth=2.2, label=label)
    # Closed-form resonance anchors of the two classical designs.
    f_helm = materials.helmholtz_resonance_frequency(
        cavity_depth=0.044,
        plate_thickness=0.006,
        hole_radius=0.0025,
        open_area=0.05,
    )
    f_mem = materials.membrane_resonance_frequency(
        surface_density=2.0, cavity_depth=0.048
    )
    # Each guide label hangs from its own ceiling instead of standing on a
    # common floor: the Spanish label is a third longer than the English one,
    # so a shared baseline pushes the longer one up into whatever crosses its
    # dotted line. The ceiling and the offset from the line are the free
    # stretch each label has -- the Helmholtz one stops under the crossing of
    # the porous and perforated curves, the membrane one stands clear of the
    # steep flank of its own resonance and under the legend.
    for f0, color, label, offset, top in (
        (f_helm, COLOR_TERTIARY, "Helmholtz closed form", 1.04, 0.65),
        (f_mem, "#9467bd", "Membrane closed form", 1.09, 0.82),
    ):
        # The guides stop above the note row: the only stretch of the bottom
        # margin wide enough for the note is the one the Helmholtz guide ran
        # through. Above that row each guide still crosses every curve it is
        # there to mark.
        ax.axvline(f0, ymin=0.085, color=color, linestyle=":", linewidth=1.1, alpha=0.8)
        ax.text(
            f0 * offset,
            top,
            label,
            rotation=90,
            va="top",
            ha="left",
            fontsize=8.5,
            color=color,
        )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Sound absorption coefficient $\alpha$")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(50.0, 5000.0)
    ax.set_title("Multilayer Absorber Prediction (Transfer-Matrix Method)", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax.legend(loc="upper left", fontsize=9)
    # The note stops short of the right frame: further right the bottom margin
    # belongs to the microperforated dip and its spike, and to the tail of the
    # perforated curve. Anchored here it clears the perforated tail above and
    # the membrane tail below at both label lengths.
    ax.text(
        0.865,
        0.036,
        "Normal incidence, rigid backing, 50 mm total depth",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.5,
        color=COLOR_FG,
    )
    plt.tight_layout()
    save_figure(output_dir, "porous_absorber_designs.svg")
    plt.close()


def generate_limp_frame_effective_density(output_dir: str) -> None:
    """Rigid-frame against limp-frame effective density (Allard & Atalla 11.3.4).

    One concept: a light frame carries inertia. The rigid-frame equivalent
    fluid lets the imaginary part of the effective density run away as
    sigma/(j w) below the decoupling frequency, because a motionless frame
    forbids rigid-body motion of the sample; the limp correction of
    Eqs. (11.53)-(11.55) converges instead on the apparent total density
    rho_t = rho1 + phi rho0. Above the decoupling frequency the two models
    coincide, which is the plot's own check on the correction.
    """
    print("Generating limp_frame_effective_density...")
    from phonometry import materials

    # Allard & Atalla Table 11.2 (printed p. 254): the soft fibrous material
    # behind their Figure 11.2.
    porosity, resistivity, frame_density = 0.98, 25.0e3, 30.0
    f = np.linspace(1.0, 2000.0, 800)
    rigid = materials.johnson_champoux_allard(
        f,
        resistivity,
        porosity=porosity,
        tortuosity=1.02,
        viscous_length=90e-6,
        thermal_length=180e-6,
    )
    limp = materials.limp_frame(rigid, frame_density, porosity=porosity)
    rho0 = rigid.air_density
    total = frame_density + porosity * rho0
    f_d = materials.decoupling_frequency(
        resistivity, porosity=porosity, frame_density=frame_density
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(
        f,
        np.real(rigid.effective_density) / rho0,
        "--",
        color=COLOR_PRIMARY,
        linewidth=1.8,
        label="rigid frame, real part",
    )
    ax.plot(
        f,
        np.imag(rigid.effective_density) / rho0,
        ":",
        color=COLOR_PRIMARY,
        linewidth=1.8,
        label="rigid frame, imaginary part",
    )
    ax.plot(
        f,
        np.real(limp.effective_density) / rho0,
        "-",
        color=COLOR_SECONDARY,
        linewidth=2.4,
        label="limp frame, real part",
    )
    ax.plot(
        f,
        np.imag(limp.effective_density) / rho0,
        "-",
        color=COLOR_TERTIARY,
        linewidth=2.4,
        label="limp frame, imaginary part",
    )
    ax.axhline(total / rho0, color=COLOR_FG, linestyle="-", linewidth=1.0, alpha=0.45)
    ax.axvline(f_d, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.annotate(
        f"apparent total density $\\rho_\\mathrm{{t}}/\\rho_0$ = {total / rho0:.1f}",
        xy=(1960.0, total / rho0),
        xytext=(1960.0, total / rho0 - 1.4),
        ha="right",
        va="top",
        fontsize=9,
        color=COLOR_FG,
    )
    ax.annotate(
        f"decoupling frequency {f_d:.0f} Hz",
        xy=(f_d, -18.0),
        xytext=(f_d + 60.0, -18.0),
        ha="left",
        va="center",
        fontsize=9,
        color=COLOR_FG,
    )

    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised effective density $\rho_\mathrm{e}/\rho_0$")
    ax.set_title("A Limp Frame Carries Its Own Inertia", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.015,
        0.03,
        "Soft fibrous layer: porosity 0.98, "
        "flow resistivity 25 kPa·s/m², frame density 30 kg/m³",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=COLOR_FG,
    )
    plt.tight_layout()
    save_figure(output_dir, "limp_frame_effective_density.svg")
    plt.close()


def generate_biot_frame_resonance(output_dir: str) -> None:
    """The frame resonance an equivalent fluid cannot produce (A&A 6.6.3).

    One concept: the surface impedance of a glass-wool layer glued to a rigid
    wall, computed with the full Biot theory and with the same material treated
    as a rigid-frame equivalent fluid. The two curves lie on top of each other
    everywhere except around the quarter-wavelength resonance of the
    frame-borne wave, where the Biot prediction develops the dip-and-peak in
    the real part and the sharp maximum in the imaginary part that Allard &
    Atalla measure and plot in their Figure 6.10. The closed form Eq. (6.110)
    marks where that resonance is expected.
    """
    print("Generating biot_frame_resonance...")
    from phonometry import materials

    # Allard & Atalla Table 6.1 (printed p. 124): the glass wool "Domisol
    # Coffrage", with the characteristic lengths of their printed p. 123.
    porosity, tortuosity, resistivity = 0.94, 1.06, 40.0e3
    frame_density, shear_modulus = 130.0, 2.2e6 * (1.0 + 0.1j)
    thickness = 0.10
    f = np.linspace(200.0, 1500.0, 1300)
    medium = materials.johnson_champoux_allard(
        f,
        resistivity,
        porosity=porosity,
        tortuosity=tortuosity,
        viscous_length=0.56e-4,
        thermal_length=1.1e-4,
    )
    rigid = materials.layered_absorber(f, [materials.PorousLayer(thickness, medium)])
    biot = materials.layered_absorber(
        f,
        [
            materials.PoroelasticLayer(
                thickness, medium, porosity, tortuosity, frame_density, shear_modulus
            )
        ],
    )
    f_r = materials.frame_quarter_wave_resonance(
        thickness,
        shear_modulus=shear_modulus,
        poisson_ratio=0.0,
        frame_density=frame_density,
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(
        f,
        np.real(rigid.normalized_impedance),
        "--",
        color=COLOR_PRIMARY,
        linewidth=1.8,
        label="rigid frame, real part",
    )
    ax.plot(
        f,
        np.imag(rigid.normalized_impedance),
        ":",
        color=COLOR_PRIMARY,
        linewidth=1.8,
        label="rigid frame, imaginary part",
    )
    ax.plot(
        f,
        np.real(biot.normalized_impedance),
        "-",
        color=COLOR_SECONDARY,
        linewidth=2.4,
        label="Biot poroelastic, real part",
    )
    ax.plot(
        f,
        np.imag(biot.normalized_impedance),
        "-",
        color=COLOR_TERTIARY,
        linewidth=2.4,
        label="Biot poroelastic, imaginary part",
    )
    ax.axvline(f_r, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    ax.annotate(
        f"frame quarter-wave resonance {f_r:.0f} Hz",
        xy=(f_r, 2.7),
        xytext=(f_r + 25.0, 2.7),
        ha="left",
        va="center",
        fontsize=9,
        color=COLOR_FG,
    )

    ax.set_xlim(200.0, 1500.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised surface impedance $Z_\mathrm{s}/\rho_0 c_0$")
    ax.set_title("Only an Elastic Frame Resonates", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    # The empty band between the real and imaginary parts, to the right of the
    # resonance marker: the bottom-left corner is crossed by both imaginary
    # curves and by the marker itself, in either language.
    ax.text(
        0.30,
        0.57,
        "Glass wool, 100 mm glued to a rigid wall: porosity 0.94,\n"
        "flow resistivity 40 kPa s/m², frame density 130 kg/m³,\n"
        "shear modulus 2.2 MPa",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=8.5,
        color=COLOR_FG,
    )
    plt.tight_layout()
    save_figure(output_dir, "biot_frame_resonance.svg")
    plt.close()


def generate_slow_sound_absorber(output_dir: str) -> None:
    """Perfect absorption by critical coupling in a slow-sound slit panel.

    A single Helmholtz resonator loads a thin closed slit; the critical
    coupling design tunes the cavity length and slit height so the intrinsic
    visco-thermal losses exactly balance the leakage, giving alpha = 1 at
    300 Hz. Detuning the slit height (more or less loss) breaks the balance
    and lowers the peak. One concept: perfect absorption is a loss-versus-
    leakage balance in a deep-subwavelength (L = lambda/38) panel.
    """
    print("Generating slow_sound_absorber.svg...")
    from phonometry import materials

    lattice_step = 3.0e-2
    period = 5.0e-2
    f0 = 300.0
    base = materials.HelmholtzResonator(
        neck_length=1.0e-3,
        neck_side=3.0e-3,
        cavity_length=30.0e-3,
        cavity_side=27.0e-3,
    )
    design = materials.critical_coupling_design(
        f0,
        base,
        lattice_step=lattice_step,
        period=period,
    )
    h0 = design.slit_height
    f = np.linspace(150.0, 500.0, 700)
    cases = [
        (h0, COLOR_SECONDARY, "-", "Critically coupled (perfect)"),
        (0.6 * h0, COLOR_PRIMARY, "--", "Narrow slit (over-damped)"),
        (1.7 * h0, COLOR_TERTIARY, "--", "Wide slit (under-damped)"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for height, color, ls, label in cases:
        res = materials.slit_helmholtz_absorber(
            f,
            design.resonator,
            slit_height=height,
            lattice_step=lattice_step,
            period=period,
        )
        ax.plot(f, res.absorption, ls, color=color, linewidth=2.2, label=label)
    ax.axvline(f0, color=COLOR_FG, linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(
        f0 * 1.01,
        0.05,
        "design 300 Hz",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=COLOR_FG,
    )
    panel_depth = lattice_step  # slit depth L = N a with N = 1
    ratio = (343.0 / f0) / panel_depth
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"Sound absorption coefficient $\alpha$")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(150.0, 500.0)
    ax.set_title("Perfect Absorption by Critical Coupling (Slow-Sound Panel)", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(
        0.985,
        0.03,
        f"Normal incidence, rigid backing, panel depth $L = \\lambda/{ratio:.0f}$",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.5,
        color=COLOR_FG,
    )
    plt.tight_layout()
    save_figure(output_dir, "slow_sound_absorber.svg")
    plt.close()


def generate_absorber_stack_geometry(output_dir: str) -> None:
    """To-scale cross-section of a three-layer absorber design.

    The microperforated plate + air gap + porous layer stack drawn the way a
    lab manual would: each layer with its material fill and dimensioned
    thickness, the rigid backing at the right and the sound arriving from the
    left. One concept: the geometry behind the multilayer absorption curve.
    """
    print("Generating absorber_stack_geometry...")
    from phonometry import materials

    frequency = np.linspace(200.0, 4000.0, 100)
    medium = materials.miki(frequency, 20000.0)
    layers: list[Any] = [
        materials.MicroperforatedPlateLayer(0.001, 0.0002, 0.01),
        materials.AirLayer(0.03),
        materials.PorousLayer(0.05, medium),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_absorber_stack(layers, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "absorber_stack_geometry.svg")
    plt.close()


def generate_slit_absorber_geometry(output_dir: str) -> None:
    """To-scale cross-section of the critically-coupled slit panel.

    One period of the slow-sound metamaterial absorber of the companion
    absorption figure: the 300 Hz critical-coupling design (single Helmholtz
    resonator loading a thin slit, panel depth lambda/38). One concept: what
    the deep-subwavelength panel actually looks like.
    """
    print("Generating slit_absorber_geometry...")
    from phonometry import materials

    base = materials.HelmholtzResonator(
        neck_length=1.0e-3,
        neck_side=3.0e-3,
        cavity_length=30.0e-3,
        cavity_side=27.0e-3,
    )
    design = materials.critical_coupling_design(
        300.0,
        base,
        lattice_step=3.0e-2,
        period=5.0e-2,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_slit_absorber_geometry(
        [design.resonator],
        ax=ax,
        slit_height=design.slit_height,
        lattice_step=3.0e-2,
        period=5.0e-2,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "slit_absorber_geometry.svg")
    plt.close()


def generate_qrd_geometry(output_dir: str) -> None:
    """To-scale well profile of the N = 7 QRD of the polar-response figure.

    Two periods of the commercial N = 7 quadratic-residue diffuser (42 wells
    across 3,6 m, so an 85,7 mm pitch split into an 80,7 mm well plus a 5 mm
    fin; deepest well 0,2 m, design frequency 490 Hz) drawn as the physical
    profile the Fraunhofer prediction models. One concept: the
    quadratic-residue depth sequence as a real, buildable surface.
    """
    print("Generating qrd_geometry...")
    from phonometry import materials

    depths = materials.qrd_well_depths(7, 490.0, speed_of_sound=343.0)
    pitch = 3.6 / 42
    fin = 0.005
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_qrd_geometry(
        depths,
        pitch - fin,
        ax=ax,
        periods=2,
        fin_width=fin,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "qrd_geometry.svg")
    plt.close()


#: Table-1 metadiffuser rows (slit h, neck l_n, cavity l_c, neck w_n,
#: cavity w_c, in mm), shared by the figures and the animation mesh.
_METADIFFUSER_T1_ROWS = (
    (14.7, 13.0, 16.4, 6.2, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (15.7, 13.3, 17.0, 6.3, 9.0),
    (20.3, 18.0, 20.7, 3.2, 9.0),
)


def _qr_metadiffuser_wells() -> tuple[Any, float, float]:
    """The published 2 cm quadratic-residue metadiffuser (wells, L, d)."""
    from phonometry import materials

    rows = _METADIFFUSER_T1_ROWS
    wells = [
        materials.MetadiffuserWell(
            h * 1e-3,
            (materials.HelmholtzResonator(ln * 1e-3, wn * 1e-3, lc * 1e-3, wc * 1e-3),)
            * 2,
        )
        for h, ln, lc, wn, wc in rows
    ]
    return wells, 0.02, 0.07


def generate_metadiffuser_polar(output_dir: str) -> None:
    """A 2 cm metadiffuser scatters like the 27 cm QRD it mimics.

    Far-field polar responses at 2 kHz of the five-slit metadiffuser panel
    and of the quadratic-residue diffuser (design frequency 500 Hz, wells up
    to 27.4 cm deep) whose reflection-phase profile it reproduces, both with
    six repetitions. One concept: the deep-subwavelength panel replaces a
    13.7 times thicker classical diffuser.
    """
    print("Generating metadiffuser_polar...")
    from phonometry import materials

    wells, depth, period = _qr_metadiffuser_wells()
    sequence = np.roll(materials.quadratic_residue_sequence(5), -1)
    qrd_depths = sequence * (343.0 / 500.0) / (2 * 5)
    meta = materials.metadiffuser_polar_response(
        2000.0,
        wells,
        depth=depth,
        period=period,
        periods=6,
    )
    qrd = materials.predict_diffuser_polar_response(
        period,
        2000.0,
        depths=qrd_depths,
        periods=6,
        include_obliquity=False,
    )
    _fig, ax = plt.subplots(
        figsize=(10, 6.2),
        subplot_kw={"projection": "polar"},
    )
    meta.plot(
        ax=ax,
        color=COLOR_SECONDARY,
        marker="",
        linewidth=2.2,
        label="Metadiffuser, panel 2 cm",
        language=_LANG,
    )
    qrd.plot(
        ax=ax,
        color=COLOR_PRIMARY,
        marker="",
        linewidth=1.6,
        linestyle="--",
        label="QRD, wells up to 27.4 cm",
        language=_LANG,
    )
    # The default theta formatter writes its negative angles with an ASCII
    # hyphen; restate the same grid with the typographic minus.
    polar: Any = ax
    polar.set_thetagrids(
        np.arange(-90, 91, 30), [f"{_fmt_minus(a, '.0f')}°" for a in range(-90, 91, 30)]
    )
    ax.set_title(
        "The 2 cm metadiffuser scatters like the 27 cm QRD (2 kHz)",
        pad=18,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_polar.svg")
    plt.close()


def generate_metadiffuser_geometry(output_dir: str) -> None:
    """To-scale cross-section of the published 2 cm metadiffuser panel.

    One period of the five-slit quadratic-residue metadiffuser: numbered
    slits open at the face, each loaded by two Helmholtz resonators
    shelved sideways into the septum, over a rigid backing. One concept:
    the whole 35 cm x 2 cm panel that replaces a 27 cm deep diffuser.
    """
    print("Generating metadiffuser_geometry...")
    from phonometry.materials import plot_metadiffuser_panel_geometry

    wells, depth, period = _qr_metadiffuser_wells()
    _fig, ax = plt.subplots(figsize=(10, 3.4))
    plot_metadiffuser_panel_geometry(
        wells,
        ax=ax,
        depth=depth,
        period=period,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_geometry.svg")
    plt.close()


def generate_metadiffuser_absorption(output_dir: str) -> None:
    """What a metadiffuser costs in absorption: per-well and face-averaged."""
    print("Generating metadiffuser_absorption...")
    from phonometry import materials

    # The published five-slit panel over the band its resonators live in. Two
    # of the five slits pass through critical coupling above the 2 kHz design
    # frequency, so the panel is a near-lossless phase grating where it is
    # tuned and a quarter-absorbing surface 300 Hz higher up.
    wells, depth, period = _qr_metadiffuser_wells()
    freqs = np.arange(1800.0, 2601.0, 5.0)
    panel = materials.metadiffuser_reflection(freqs, wells, depth=depth, period=period)
    per_well = np.asarray(panel.well_absorption)
    face = np.asarray(panel.absorption)
    i_design = int(np.argmin(np.abs(freqs - 2000.0)))

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    palette = (COLOR_PRIMARY, "#9467bd", "#8c564b", COLOR_TERTIARY, "#e8a838")
    for index, alpha in enumerate(per_well):
        ax.plot(
            freqs,
            alpha,
            color=palette[index % len(palette)],
            linewidth=1.5,
            zorder=3,
            label=f"slit {index + 1}",
        )
    ax.plot(
        freqs,
        face,
        color=COLOR_SECONDARY,
        linewidth=2.6,
        zorder=4,
        label="face average (what a room model consumes)",
    )
    ax.axvline(2000.0, color=COLOR_FG, linestyle=":", linewidth=1.3, zorder=2)
    ax.annotate(
        f"design frequency:\nface average {face[i_design]:.3f}",
        (2010.0, 0.45),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Absorption coefficient")
    ax.set_title("The absorption a metadiffuser pays for its phases", pad=12)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_absorption.svg")
    plt.close()


def generate_metadiffuser_phase_match(output_dir: str) -> None:
    """The whole trick, and its limit: matched phases, only at one frequency."""
    print("Generating metadiffuser_phase_match...")
    from phonometry import materials

    # Left: the five reflection phases at the 2 kHz design frequency against
    # the targets of the 27.4 cm deep N = 5 QRD, with |R_n| beside each point.
    # Right: the same five phases swept across the band, where the rigid
    # well's linear phase and the resonator-loaded slit's dispersive one part
    # company either side of the crossing.
    wells, depth, period = _qr_metadiffuser_wells()
    sequence = np.roll(materials.quadratic_residue_sequence(5), -1)
    qrd_depths = np.asarray(sequence) * (343.0 / 500.0) / (2 * 5)
    index = np.arange(1, 6)

    at_design = materials.metadiffuser_reflection(
        np.array([2000.0]), wells, depth=depth, period=period
    )
    r_design = np.asarray(at_design.reflection)[:, 0]
    k_design = 2.0 * np.pi * 2000.0 / 343.0
    target_design = np.degrees(np.angle(np.exp(-2j * k_design * qrd_depths)))

    freqs = np.linspace(1600.0, 2600.0, 201)
    swept = materials.metadiffuser_reflection(freqs, wells, depth=depth, period=period)
    phases = np.degrees(np.angle(np.asarray(swept.reflection)))
    k = 2.0 * np.pi * freqs / 343.0
    targets = np.degrees(np.angle(np.exp(-2j * np.outer(qrd_depths, k))))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 5.6))
    ax_l.plot(
        index,
        target_design,
        "o",
        markersize=11,
        color=COLOR_PRIMARY,
        markerfacecolor="none",
        markeredgewidth=2.0,
        zorder=3,
        label="QRD target, wells to 27.4 cm",
    )
    ax_l.plot(
        index,
        np.degrees(np.angle(r_design)),
        "s",
        markersize=7,
        color=COLOR_SECONDARY,
        zorder=4,
        label="Metadiffuser, panel 2 cm",
    )
    for i, r in zip(index, r_design):
        ax_l.annotate(
            f"$|R|$ = {abs(r):.2f}",
            (i, np.degrees(np.angle(r)) - 12.0),
            fontsize=8.5,
            color=COLOR_FG,
            ha="center",
            va="top",
        )
    ax_l.set_xticks(index)
    ax_l.set_xlabel("Slit index $n$")
    ax_l.set_ylabel("Reflection phase [deg]")
    # Wider than the data so the |R| annotations centred on the first and the
    # last slit, which are about 0.31 of a slit wide either side of the marker,
    # stay inside the spines rather than being cut by them.
    ax_l.set_xlim(0.55, 5.45)
    ax_l.set_ylim(-115.0, 115.0)
    ax_l.set_title("At the 2 kHz design frequency", pad=12)
    # Above the phases rather than under them: at the lower right the box met
    # the |R| annotation of slit 3, whose comma in Spanish reached its border.
    ax_l.legend(loc="upper right", fontsize=9)
    ax_l.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_l.set_axisbelow(True)

    # Plotted as the wrapped phase *error* per slit rather than as ten
    # curves: the rigid well's phase is linear in frequency and the loaded
    # slit's is not, so what matters is how fast the difference between them
    # grows away from the one frequency where it was driven to zero.
    error = np.degrees(np.angle(np.exp(1j * np.radians(phases - targets))))
    ax_r.axhspan(-10.0, 10.0, color=theme_fill(COLOR_TERTIARY, ax_r), zorder=0)
    palette = (COLOR_PRIMARY, "#9467bd", "#8c564b", COLOR_TERTIARY, "#e8a838")
    for slit in range(error.shape[0]):
        ax_r.plot(
            freqs,
            error[slit],
            color=palette[slit % len(palette)],
            linewidth=1.5,
            zorder=3,
            label=f"slit {slit + 1}",
        )
    ax_r.axvline(2000.0, color=COLOR_FG, linestyle=":", linewidth=1.3, zorder=1)
    # Set on two lines inside the band it names: the only stretch of the panel
    # no curve reaches is the left end of the band itself, before the three
    # slits that converge on zero drop into it around 1900 Hz. One line was
    # 330 Hz wide in English and 400 Hz in Spanish, wider than any clear
    # corridor here, so it was crossed by the curves and by the 2 kHz guide.
    ax_r.annotate(
        "within 10 deg\nof the target",
        (1665.0, -1.25),
        fontsize=8.0,
        color=COLOR_FG,
        ha="left",
        va="center",
    )
    ax_r.set_xlim(freqs[0], freqs[-1])
    ax_r.set_ylim(-100.0, 100.0)
    ax_r.set_xlabel(LABEL_FREQ_HZ)
    ax_r.set_ylabel("Phase error against the QRD target [deg]")
    ax_r.set_title("Across the band: exact only where it was tuned", pad=12)
    # Below the axes rather than in them: five curves with poles in the band
    # leave no clear rectangle inside, and at the lower left the box sat on
    # the slit-3 curve, which crossed its own sample in the key.
    ax_r.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        fontsize=9,
        ncol=5,
        columnspacing=1.4,
        handlelength=1.6,
    )
    ax_r.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_r.set_axisbelow(True)
    fig.align_ylabels((ax_l, ax_r))
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_phase_match.svg")
    plt.close()


def generate_metadiffuser_spectrum(output_dir: str) -> None:
    """Band by band: where the 2 cm panel and the 27 cm grating agree."""
    print("Generating metadiffuser_spectrum...")
    from phonometry import materials

    # Both panels are six periods of five 7 cm wells, a 2.1 m array, so the
    # comparison is of the wells alone. Below c / L = 980 Hz the period is
    # shorter than a wavelength, no grating lobe exists and neither panel can
    # do better than the flat reference: the normalised coefficient of both
    # collapses there for a reason that has nothing to do with the wells.
    wells, depth, period = _qr_metadiffuser_wells()
    sequence = np.roll(materials.quadratic_residue_sequence(5), -1)
    qrd_depths = np.asarray(sequence) * (343.0 / 500.0) / (2 * 5)
    freqs = np.array(
        [500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], dtype=float
    )
    meta = materials.metadiffuser_diffusion_spectrum(
        freqs, wells, depth=depth, period=period, periods=6
    )
    qrd = materials.predicted_diffusion_spectrum(
        period, freqs, depths=qrd_depths, periods=6, include_obliquity=False
    )
    f_lobe = 343.0 / (5 * period)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.axvspan(freqs[0] * 0.92, f_lobe, color=theme_fill(COLOR_MUTED, ax), zorder=0)
    ax.semilogx(
        freqs,
        np.asarray(qrd.normalized),
        color=COLOR_PRIMARY,
        marker="o",
        markersize=5,
        linewidth=1.8,
        linestyle="--",
        zorder=3,
        label="QRD, wells up to 27.4 cm",
    )
    ax.semilogx(
        freqs,
        np.asarray(meta.normalized),
        color=COLOR_SECONDARY,
        marker="s",
        markersize=5,
        linewidth=1.8,
        zorder=4,
        label="Metadiffuser, panel 2 cm",
    )
    ax.axvline(2000.0, color=COLOR_FG, linestyle=":", linewidth=1.3, zorder=2)
    ax.annotate(
        "tuned here:\n0.32 against 0.32",
        (2080.0, 0.36),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax.annotate(
        f"below $c/L$ = {f_lobe:.0f} Hz no grating\n"
        f"lobe exists: neither panel can\nbeat the flat reference",
        (freqs[0] * 0.97, 0.36),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax.set_ylim(0.0, 0.45)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalised diffusion coefficient")
    ax.set_title("The honest bandwidth of the trick", pad=12)
    format_frequency_axis(ax, freqs[0], freqs[-1])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_spectrum.svg")
    plt.close()


def generate_impedance_tube_geometry(output_dir: str) -> None:
    """To-scale side view of a 100 mm ISO 10534-2 impedance tube.

    Loudspeaker, the two flush microphones (s = 50 mm, x1 = 150 mm), the
    sample against its rigid backing, the circular cross-section and the
    plane-wave working range those dimensions buy. One concept: the tube
    geometry fixes the usable frequency band.
    """
    print("Generating impedance_tube_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_impedance_tube_geometry(
        ax=ax,
        spacing=0.05,
        x1=0.15,
        diameter=0.10,
        shape="circular",
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "impedance_tube_geometry.svg")
    plt.close()


def generate_transmission_tube_geometry(output_dir: str) -> None:
    """To-scale side view of a 100 mm ASTM E2611 transmission tube.

    Four flush microphones around the specimen (l1 = 100 mm, s1 = 50 mm,
    l2 = 200 mm, s2 = 50 mm), the changeable termination of the two-load
    method and the ASTM working range. One concept: where the four
    microphones of the transfer-matrix method actually sit.
    """
    print("Generating transmission_tube_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_transmission_tube_geometry(
        ax=ax,
        l1=0.10,
        s1=0.05,
        l2=0.20,
        s2=0.05,
        thickness=0.05,
        diameter=0.10,
        shape="circular",
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "transmission_tube_geometry.svg")
    plt.close()


def generate_helmholtz_resonator_geometry(output_dir: str) -> None:
    """To-scale cross-section of the 300 Hz slow-sound Helmholtz resonator.

    The square-section resonator of the critical-coupling design (neck 1 x 3
    mm, cavity 30 x 27 mm) with its four defining dimensions. One concept:
    the resonator geometry the slit-panel figures build on.
    """
    print("Generating helmholtz_resonator_geometry...")
    from phonometry import materials

    resonator = materials.HelmholtzResonator(
        neck_length=1.0e-3,
        neck_side=3.0e-3,
        cavity_length=30.0e-3,
        cavity_side=27.0e-3,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    resonator.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "helmholtz_resonator_geometry.svg")
    plt.close()


def generate_insitu_setup_geometry(output_dir: str) -> None:
    """The in-situ road absorption set-up to scale.

    Loudspeaker at 1,25 m, microphone at 0,25 m on the same vertical, and
    the 1,34 m sampled-area radius of the standard 5 ms window on the
    surface. One concept: what the subtraction technique actually samples.
    """
    print("Generating insitu_setup_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    materials.plot_insitu_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "insitu_setup_geometry.svg")
    plt.close()


def generate_dynamic_stiffness_rig_geometry(output_dir: str) -> None:
    """The dynamic-stiffness resonance rig to scale.

    The standard 200 mm square resilient specimen under the 8 kg load
    plate, exciter above and accelerometer on the plate. One concept: the
    little mass-spring oscillator behind s' and the floating-floor f0.
    """
    print("Generating dynamic_stiffness_rig_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    materials.plot_dynamic_stiffness_rig(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "dynamic_stiffness_rig_geometry.svg")
    plt.close()


def generate_diffusion_goniometer_geometry(output_dir: str) -> None:
    """The free-field diffusion goniometer in plan, to scale.

    The 37-microphone semicircle at 5 m, the source at 10 m on the normal
    and the sample at the centre. One concept: where the polar responses
    of the diffusion coefficient are measured.
    """
    print("Generating diffusion_goniometer_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(9.0, 6.6))
    materials.plot_goniometer_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "diffusion_goniometer_geometry.svg")
    plt.close()


def generate_scattering_coefficient(output_dir: str) -> None:
    """ISO 17497-1 Eq. (5): the alpha pair on top, the s(f) it produces below."""
    print("Generating scattering_coefficient.png...")
    from phonometry import materials

    # A realistic reverberation-room measurement reduced to two absorption
    # spectra over the 13 one-third-octave bands 250-4000 Hz: the random-
    # incidence absorption alpha_s (static turntable) and the specular
    # absorption alpha_spec (rotating turntable). A diffuser scatters more with
    # frequency, so alpha_spec climbs above alpha_s and s(f) = (alpha_spec -
    # alpha_s)/(1 - alpha_s) rises smoothly from near 0 towards 0.8. The upper
    # panel is Eq. (5) drawn as it is computed: the filled gap is the
    # numerator, and the fixed alpha_s sets the 1 - alpha_s denominator.
    freqs = np.array(
        [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000],
        dtype=float,
    )
    alpha_s = np.full_like(freqs, 0.10)
    alpha_spec = 0.11 + 0.75 * (np.log10(freqs / 250.0) / np.log10(4000.0 / 250.0))
    result = materials.scattering_coefficient_spectrum(freqs, alpha_spec, alpha_s)

    fig, (ax_a, ax) = plt.subplots(
        2,
        1,
        figsize=(10, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    ax_a.fill_between(
        freqs,
        alpha_s,
        alpha_spec,
        color=COLOR_TERTIARY,
        alpha=theme_fill_alpha(COLOR_TERTIARY, ax_a),
        zorder=1,
        label=r"$\alpha_{\mathrm{spec}} - \alpha_\mathrm{s}$"
        r"  (numerator of Eq. (5))",
    )
    ax_a.semilogx(
        freqs,
        alpha_spec,
        color=COLOR_SECONDARY,
        linewidth=1.9,
        marker="s",
        markersize=5,
        zorder=3,
        label=r"$\alpha_{\mathrm{spec}}$, rotating turntable (T3, T4)",
    )
    ax_a.semilogx(
        freqs,
        alpha_s,
        color=COLOR_PRIMARY,
        linewidth=1.9,
        marker="o",
        markersize=5,
        zorder=3,
        label=r"$\alpha_\mathrm{s}$, static turntable (T1, T2)",
    )
    ax_a.set_ylabel("Absorption coefficient")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_title("Random-incidence scattering coefficient (ISO 17497-1)", pad=12)
    ax_a.legend(loc="upper left", fontsize=9)
    ax_a.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_a.set_axisbelow(True)

    ax.semilogx(
        result.frequencies,
        result.scattering,
        color=COLOR_FG,
        linewidth=1.9,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Scattering coefficient $s$")
    ax.set_xlim(freqs.min() * 0.9, freqs.max() * 1.1)
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([250, 500, 1000, 2000, 4000])
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.text(
        0.985,
        0.06,
        r"$s = (\alpha_{\mathrm{spec}} - \alpha_\mathrm{s})"
        r"/(1 - \alpha_\mathrm{s})$   Eq. (5)",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=11,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    fig.align_ylabels((ax_a, ax))
    plt.tight_layout()
    save_figure(output_dir, "scattering_coefficient.png")
    plt.close()


def _goniometer_path_times(
    half_width: float,
    receiver_angle: float,
    source_distance: float = 10.0,
    receiver_distance: float = 5.0,
    speed_of_sound: float = 343.0,
) -> tuple[float, float, float]:
    """Direct, shortest and longest reflected arrival times of a goniometer.

    The window of ISO 17497-2 Clause 7.4.3 is sized from the shortest and the
    longest sample-to-receiver path, and both follow from the geometry the
    standard fixes: a source on the reference normal at ``source_distance``,
    a receiver on an arc of ``receiver_distance``, and a flat sample of
    half-width ``half_width`` at the origin. Returned in seconds.
    """
    theta = np.radians(receiver_angle)
    sx, sy = 0.0, source_distance
    rx, ry = receiver_distance * np.sin(theta), receiver_distance * np.cos(theta)
    x = np.linspace(-half_width, half_width, 2001)
    paths = np.hypot(x - sx, sy) + np.hypot(x - rx, ry)
    direct = float(np.hypot(rx - sx, ry - sy))
    return (
        direct / speed_of_sound,
        float(paths.min()) / speed_of_sound,
        float(paths.max()) / speed_of_sound,
    )


def generate_diffusion_measurement_chain(output_dir: str) -> None:
    """ISO 17497-2 Clause 7.4: h1, h2, their difference, h4 and the window."""
    print("Generating diffusion_measurement_chain...")
    from phonometry import materials

    # The page's own published geometry read on the standard rig: the N = 7
    # QRD, 6 periods, 3.6 m wide, source at 10 m on the reference normal and
    # the receiver on the 5 m arc at 60 deg, well outside the specular zone.
    # Every arrival time below is that geometry divided by c (Clause 7.4.3
    # sizes the window from the shortest and the longest path), so the figure
    # is the standard's own data-reduction chain and not a sketch of it.
    c, fs, n = 343.0, 96000.0, 8192
    t = np.arange(n) / fs
    t_direct, t_first, t_last = _goniometer_path_times(1.8, 60.0, speed_of_sound=c)
    t_room = t_last + 6.0e-3  # a wall reflection, in h1 and in h2

    def _speaker(delay: float) -> np.ndarray:
        """The loudspeaker-microphone response h3: a short 3 kHz burst."""
        env = np.exp(-(((t - delay) * 3000.0 / 2.2) ** 2))
        return env * np.cos(2.0 * np.pi * 3000.0 * (t - delay))

    def _spike(delay: float, amp: float) -> np.ndarray:
        out = np.zeros(n)
        out[min(round(delay * fs), n - 1)] = amp
        return out

    # The sample reflection is one arrival per well of the illuminated array,
    # spread between the shortest and the longest path and delayed further by
    # the round trip down each well of the quadratic-residue sequence, so the
    # whole return still lands inside [t_first, t_last].
    depths = materials.qrd_well_depths(7, 490.0, speed_of_sound=c)
    seq = np.tile(depths, 6)
    t_well = 2.0 * float(depths.max()) / c
    delays = np.linspace(t_first, t_last - t_well, seq.size) + 2.0 * seq / c
    ideal_sample = sum(_spike(d, 0.16 / (1.0 + 6.0 * s)) for d, s in zip(delays, seq))
    ideal_room = _spike(t_room, 0.30)
    ideal_direct = _spike(t_direct, 1.0)

    t_h3 = 2.0e-3
    h3 = _speaker(t_h3)

    def _through(ideal: np.ndarray) -> np.ndarray:
        return np.fft.irfft(np.fft.rfft(ideal) * np.fft.rfft(h3), n)

    h1 = _through(ideal_direct + ideal_sample + ideal_room)
    h2 = _through(ideal_direct + ideal_room)
    diff = h1 - h2
    # Formula (1): deconvolve the subtracted response by h3. Regularised, as
    # any implementation must be. Dividing h3 out removes its own delay too,
    # so every arrival of h4 sits t_h3 earlier than the same arrival of h1 --
    # the Clause 7.4.3 note, and the reason the window cannot be placed by
    # reading panel (a).
    spec3 = np.fft.rfft(h3)
    h4 = np.fft.irfft(
        np.fft.rfft(diff)
        * np.conj(spec3)
        / (np.abs(spec3) ** 2 + 1e-6 * np.max(np.abs(spec3) ** 2)),
        n,
    )
    guard = 0.4e-3
    lo, hi = t_first - guard, t_last + guard
    window = ((t >= lo) & (t <= hi)).astype(float)

    ms = t * 1e3
    # (a)-(c) share the measured scale; (d)-(e) share the deconvolved one,
    # which is a different quantity in different units.
    s_meas, s_dec = float(np.abs(h1).max()), float(np.abs(h4).max())
    panels = (
        (h1, "(a) $h_1$: sample present", s_meas),
        (h2, "(b) $h_2$: sample removed", s_meas),
        (diff, "(c) $h_1 - h_2$: the room is gone", s_meas),
        (h4, "(d) $h_4$: deconvolved by $h_3$, Formula (1)", s_dec),
        (h4 * window, "(e) windowed, Clause 7.4.3", s_dec),
    )
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 9.6))
    for ax, (y, label, scale) in zip(axes, panels):
        ax.plot(ms, y, color=COLOR_PRIMARY, linewidth=1.1, zorder=3)
        ax.set_ylabel(
            label, fontsize=9, rotation=0, ha="right", va="center", labelpad=8
        )
        ax.set_ylim(-1.15 * scale, 1.15 * scale)
        ax.set_yticks([])
        ax.grid(axis="x", color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    for ax in (axes[0], axes[1]):
        ax.annotate(
            "direct",
            ((t_direct + t_h3) * 1e3 - 1.4, 0.45 * s_meas),
            fontsize=8,
            color=COLOR_FG,
            ha="right",
            va="bottom",
        )
        ax.annotate(
            "room",
            ((t_room + t_h3) * 1e3 + 1.4, 0.34 * s_meas),
            fontsize=8,
            color=COLOR_FG,
            ha="left",
            va="bottom",
        )
    axes[0].annotate(
        "sample",
        ((0.5 * (t_first + t_last) + t_h3) * 1e3, 0.34 * s_meas),
        fontsize=8,
        color=COLOR_FG,
        ha="center",
        va="bottom",
    )
    axes[3].annotate(
        f"$h_4$ arrives {t_h3 * 1e3:.0f} ms earlier than $h_1$:\n"
        f"dividing by $h_3$ removes its delay too",
        (0.985, 0.10),
        xycoords="axes fraction",
        fontsize=8.5,
        color=COLOR_FG,
        ha="right",
        va="bottom",
    )
    win = axes[-1]
    win.axvspan(
        lo * 1e3,
        hi * 1e3,
        color=COLOR_TERTIARY,
        alpha=theme_fill_alpha(COLOR_TERTIARY, win),
        zorder=0,
    )
    for edge, text, ha in (
        (lo, f"shortest path\n{t_first * 1e3:.1f} ms", "right"),
        (hi, f"longest path\n{t_last * 1e3:.1f} ms", "left"),
    ):
        win.axvline(
            edge * 1e3, color=COLOR_SECONDARY, linewidth=1.4, linestyle="--", zorder=4
        )
        offset = -0.3 if ha == "right" else 0.3
        win.annotate(
            text,
            (edge * 1e3 + offset, -1.05 * s_dec),
            fontsize=8,
            color=COLOR_FG,
            ha=ha,
            va="bottom",
        )
    win.set_xlabel("Time [ms]")
    win.set_xlim(20.0, (t_room + t_h3 + 4.0e-3) * 1e3)
    axes[0].set_title(
        "From three impulse responses to one level (ISO 17497-2, Clause 7.4)",
        pad=12,
    )
    axes[1].text(
        0.30,
        0.10,
        f"source 10 m, arc 5 m, receiver 60 deg\n"
        f"window {(hi - lo) * 1e3:.1f} ms  →  analysis from about "
        f"{1.0 / (hi - lo):.0f} Hz\nS/N $\\geq$ 40 dB inside it on the flat reference",
        transform=axes[1].transAxes,
        va="bottom",
        ha="center",
        fontsize=9,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    fig.align_ylabels(axes)
    plt.tight_layout()
    save_figure(output_dir, "diffusion_measurement_chain.svg")
    plt.close()


def generate_qrd_working_band(output_dir: str) -> None:
    """The band a Schroeder design actually works over: f0, N f0 and c/(2w)."""
    print("Generating qrd_working_band...")
    from phonometry import materials

    # The N = 7, f0 = 500 Hz, 10 cm well, five-period design of the guide,
    # evaluated on a fine linear grid instead of band centres so that the two
    # design rules stated in prose are resolved: the panel collapses onto the
    # flat reference at multiples of N f0 (3500 Hz), where every well reflects
    # in phase again, and the single-plane-wave picture inside a well fails
    # above f_max = c / (2 w) = 1715 Hz.
    c, n_seq, f0, width, periods = 343.0, 7, 500.0, 0.10, 5
    freqs = np.linspace(200.0, 6000.0, 601)
    depths = materials.qrd_well_depths(n_seq, f0, speed_of_sound=c)
    qrd = materials.predicted_diffusion_spectrum(
        width,
        freqs,
        depths=depths,
        periods=periods,
        speed_of_sound=c,
        normalize=False,
    )
    flat = materials.predicted_diffusion_spectrum(
        width,
        freqs,
        depths=np.zeros_like(depths),
        periods=periods,
        speed_of_sound=c,
        normalize=False,
    )
    f_max = c / (2.0 * width)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.axvspan(f_max, freqs[-1], color=theme_fill(COLOR_MUTED, ax), zorder=0)
    ax.axvline(f0, color=COLOR_MUTED, linestyle=":", linewidth=1.3, zorder=1)
    ax.axvline(f_max, color=COLOR_MUTED, linestyle="-", linewidth=1.3, zorder=1)
    ax.axvline(
        n_seq * f0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6, zorder=2
    )
    ax.plot(
        freqs,
        np.asarray(qrd.diffusion),
        color=COLOR_PRIMARY,
        linewidth=1.7,
        zorder=4,
        label="$N$ = 7 QRD, $f_0$ = 500 Hz, 5 periods",
    )
    ax.plot(
        freqs,
        np.asarray(flat.diffusion),
        color=COLOR_MUTED,
        linewidth=1.4,
        linestyle="--",
        zorder=3,
        label="Flat panel, same footprint",
    )
    ax.annotate(
        "$N f_0$ = 3500 Hz: every well\nback in phase, flat again",
        (n_seq * f0, 0.05),
        xytext=(n_seq * f0 - 500.0, 0.30),
        fontsize=9,
        color=COLOR_FG,
        ha="right",
        arrowprops={"arrowstyle": "->", "color": COLOR_SECONDARY},
    )
    ax.annotate(
        f"$f_{{\\mathrm{{max}}}} = c/(2w)$ = {f_max:.0f} Hz:\n"
        f"the well stops being\na single-mode waveguide",
        (f_max + 130.0, 0.40),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax.annotate(
        "$f_0$ = 500 Hz",
        (f0 + 90.0, 0.50),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax.set_xlim(freqs[0], freqs[-1])
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Predicted diffusion coefficient $d$")
    ax.set_title("The working band of a Schroeder design", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "qrd_working_band.svg")
    plt.close()


def generate_diffuser_modulation(output_dir: str) -> None:
    """Periodicity concentrates, modulation spreads: same wells, same width."""
    print("Generating diffuser_modulation...")
    from phonometry import materials

    # Two arrangements of the same 42 wells over the same 4.2 m footprint:
    # six repeats of one N = 7 quadratic-residue period, and the same period
    # alternated with its inverse (depth d_max - d_n), which is the cheapest
    # modulation Cox & D'Antonio recommend. Nothing else differs, so every
    # difference below is periodicity.
    c, f0, width = 343.0, 500.0, 0.10
    base = materials.qrd_well_depths(7, f0, speed_of_sound=c)
    inverse = base.max() - base
    periodic = np.tile(base, 6)
    modulated = np.concatenate([base if k % 2 == 0 else inverse for k in range(6)])
    angles = np.arange(-90.0, 90.5, 5.0)
    freqs = np.array(
        [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000], dtype=float
    )

    fig = plt.figure(figsize=(11.5, 5.8))
    polar: Any = fig.add_subplot(1, 2, 1, projection="polar")
    theta = np.radians(angles)
    for depths, color, style, label in (
        (periodic, COLOR_PRIMARY, "-", "Periodic, 6 × $N$ = 7"),
        (modulated, COLOR_SECONDARY, "--", "Modulated, period + inverse"),
    ):
        response = materials.predict_diffuser_polar_response(
            width,
            1000.0,
            depths=depths,
            periods=1,
            angles=angles,
            speed_of_sound=c,
        )
        polar.plot(
            theta,
            np.asarray(response.levels),
            color=color,
            linestyle=style,
            linewidth=1.8,
            zorder=3,
            label=f"{label}  ($d$ = {response.coefficient:.2f})",
        )
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_thetamin(-90)
    polar.set_thetamax(90)
    # The default theta formatter writes its negative angles with an ASCII
    # hyphen; restate the same grid with the typographic minus.
    polar.set_thetagrids(
        np.arange(-90, 91, 30), [f"{_fmt_minus(a, '.0f')}°" for a in range(-90, 91, 30)]
    )
    polar.set_title("Reflected polar response at 1 kHz", pad=18)
    polar.legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), fontsize=9)

    ax = fig.add_subplot(1, 2, 2)
    for depths, color, marker, label in (
        (periodic, COLOR_PRIMARY, "o", "Periodic, 6 × $N$ = 7"),
        (modulated, COLOR_SECONDARY, "s", "Modulated, period + inverse"),
    ):
        spectrum = materials.predicted_diffusion_spectrum(
            width,
            freqs,
            depths=depths,
            periods=1,
            angles=angles,
            speed_of_sound=c,
        )
        ax.semilogx(
            freqs,
            np.asarray(spectrum.normalized),
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            zorder=3,
            label=label,
        )
    ax.axvline(f0, color=COLOR_MUTED, linestyle=":", linewidth=1.2, zorder=1)
    ax.annotate(
        "$f_0$: the one band the\nperiodic array wins",
        (f0 * 1.06, 0.34),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
    )
    ax.set_ylim(0.0, 0.45)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalised diffusion coefficient")
    ax.set_title("Band by band, same 4.2 m panel", pad=12)
    format_frequency_axis(ax, freqs[0], freqs[-1])
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "diffuser_modulation.svg")
    plt.close()


def generate_diffusion_polar(output_dir: str) -> None:
    """ISO 17497-2: polar reflected response and its diffusion coefficient d."""
    print("Generating diffusion_polar.png...")
    from phonometry import materials

    # Reflected sound-pressure levels L_i(theta) on the standard 37-point
    # semicircle (-90 to 90 deg, 5 deg spacing) of a published diffuser
    # geometry: the N = 7 QRD, 6 periods, 3.6 m wide, 0.2 m deep array of
    # Cox & D'Antonio 3e Appendix B section 7 (the commercial N = 7 QRD of
    # Hargreaves et al. 2000, Table I), predicted with the library's own
    # Fraunhofer far-field model at 1000 Hz, normal incidence, and rounded to
    # the 1e-3 dB precision of the committed tests/reference_data/ arc. Six
    # periods of a periodic QRD concentrate the reflected energy into grating
    # lobes, so the ISO 17497-2 Formula (5) coefficient d is modest.
    angles = np.arange(-90.0, 90.5, 5.0)
    depths = materials.qrd_well_depths(7, 490.0, speed_of_sound=343.0)  # deepest 0.2 m
    predicted = materials.predict_diffuser_polar_response(
        3.6 / 42,
        1000.0,
        depths=depths,
        periods=6,
        speed_of_sound=343.0,
    )
    result = materials.directional_diffusion(angles, np.round(predicted.levels, 3))

    _fig, ax = plt.subplots(figsize=(8.0, 7.5), subplot_kw={"projection": "polar"})
    # The theta-* setters live on PolarAxes, not the base Axes type.
    polar: Any = ax
    theta = np.radians(result.angles)
    polar.plot(
        theta,
        result.levels,
        color=COLOR_PRIMARY,
        linewidth=1.9,
        marker="o",
        markersize=4,
        zorder=3,
    )
    # Translucent so the polar grid keeps reading through the lobe.
    polar.fill(
        theta,
        result.levels,
        color=COLOR_PRIMARY,
        zorder=1,
        alpha=theme_fill_alpha(COLOR_PRIMARY, ax),
    )
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_thetamin(-90)
    polar.set_thetamax(90)
    # The default theta formatter writes its negative angles with an ASCII
    # hyphen; restate the same grid with the typographic minus.
    polar.set_thetagrids(
        np.arange(-90, 91, 30), [f"{_fmt_minus(a, '.0f')}°" for a in range(-90, 91, 30)]
    )
    polar.set_title(
        f"Directional diffusion  $d$ = {result.coefficient:.2f}  (ISO 17497-2)",
        pad=20,
    )
    plt.tight_layout()
    save_figure(output_dir, "diffusion_polar.png")
    plt.close()


def generate_diffuser_prediction(output_dir: str) -> None:
    """Predicted diffusion d(f): an N = 7 QRD design versus a flat panel."""
    print("Generating diffuser_prediction.png...")
    from phonometry import materials

    # An N = 7 quadratic residue diffuser, design frequency 500 Hz, 10 cm wells,
    # five periods (Cox & D'Antonio far-field Fraunhofer model). The predicted
    # diffusion is compared band by band with the flat reference panel of the
    # same footprint: the QRD spreads the reflected energy far more evenly, so
    # its diffusion coefficient sits well above the near-specular flat panel.
    freqs = np.array(
        [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
        float,
    )
    depths = materials.qrd_well_depths(7, 500.0)
    qrd = materials.predicted_diffusion_spectrum(0.10, freqs, depths=depths, periods=5)
    flat = materials.predicted_diffusion_spectrum(
        0.10, freqs, depths=np.zeros_like(depths), periods=5, normalize=False
    )

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(
        freqs,
        qrd.diffusion,
        color=COLOR_PRIMARY,
        linewidth=1.9,
        marker="o",
        markersize=4,
        label="$N$ = 7 QRD design",
    )
    ax.semilogx(
        freqs,
        flat.diffusion,
        color=COLOR_SECONDARY,
        linewidth=1.9,
        marker="s",
        markersize=4,
        linestyle="--",
        label="Flat panel",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Predicted diffusion coefficient $d$")
    format_frequency_axis(ax, 250.0, 5000.0)
    ax.set_title(
        "Predicted diffusion from design (Cox & D'Antonio Fraunhofer model)",
    )
    ax.legend(loc="best")
    plt.tight_layout()
    save_figure(output_dir, "diffuser_prediction.png")
    plt.close()


def generate_insitu_absorption(output_dir: str) -> None:
    """ISO 13472-1: in-situ one-third-octave absorption spectrum alpha(f)."""
    print("Generating insitu_absorption.png...")
    from phonometry import materials

    # A synthetic-but-realistic in-situ measurement. The incident impulse hi is
    # a unit spike; the road reflection is hr = Kr * r0 * roll(hi, shift) with
    # Kr the geometrical-spreading factor (2/3 for ds=1.25 m, dm=0.25 m), a
    # mildly frequency-dependent r0 realised by a gentle low-pass (a porous
    # surface reflects less as frequency rises), and the reflected-path delay
    # shift = round(2 dm / c * fs). The library forms the narrow-band
    # alpha = 1 - (1/Kr^2)|Hr/Hi|^2 and reduces it to one-third-octave bands.
    fs, n = 48000.0, 8192
    kr = materials.geometric_spreading_factor()  # (ds - dm)/(ds + dm) = 2/3
    hi = np.zeros(n)
    hi[0] = 1.0
    r0 = 0.85
    taps = scipy_signal.firwin(41, 1200.0, fs=fs)
    taps = taps / taps.sum()
    shift = round(2.0 * 0.25 / 340.0 * fs)  # reflected-path delay 2 dm / c
    hr = kr * r0 * np.roll(scipy_signal.lfilter(taps, 1.0, hi), shift)
    result = materials.insitu_absorption_spectrum(hi, hr, fs)

    freqs = result.frequencies
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(
        positions,
        np.nan_to_num(result.absorption),
        width=0.7,
        color=COLOR_PRIMARY,
        edgecolor=COLOR_FG,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title("In-situ road-surface absorption (ISO 13472-1)", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Absorption coefficient $\alpha$")
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.04,
        0.94,
        r"$K_\mathrm{r} = 2/3$"
        "\n"
        r"$\alpha = 1 - (1/K_\mathrm{r}^2)\,"
        r"|H_\mathrm{r}/H_\mathrm{i}|^2$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color=COLOR_FG,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": COLOR_PANEL,
            "edgecolor": COLOR_GRID,
        },
    )
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "insitu_absorption.png")
    plt.close()


def generate_adrienne_window(output_dir: str) -> None:
    """ISO 13472-1: the two arrivals, the Adrienne gate and what it costs."""
    print("Generating adrienne_window...")
    from phonometry import materials

    # The mandatory geometry: source at ds = 1.25 m, microphone at
    # dm = 0.25 m on the same vertical, so the direct path is ds - dm = 1.0 m
    # and the reflected one ds + dm = 1.5 m. The reflection therefore trails
    # the direct sound by 2 dm / c = 1.47 ms, which is why the two overlap in
    # any real impulse response and why the method is called subtraction.
    c, fs, n = 340.0, 48000.0, 4096
    t = np.arange(n) / fs
    t_direct, t_reflected = 1.0 / c, 1.5 / c
    t_parasite = 15.8e-3  # mast, vehicle, operator: outside the gate

    # A band-limited click as the loudspeaker response, its FIR group delay
    # removed so that every arrival lands at its own geometric time.
    band = scipy_signal.firwin(129, [200.0, 6000.0], fs=fs, pass_zero=False)
    lag = (band.size - 1) // 2

    def arrival(
        delay: float, amp: float, extra: np.ndarray | None = None
    ) -> np.ndarray:
        spike = np.zeros(n)
        spike[round(delay * fs) - lag] = amp
        out = scipy_signal.lfilter(band, 1.0, spike)
        if extra is not None:
            out = scipy_signal.lfilter(extra, 1.0, out)
            out = np.roll(out, -(extra.size - 1) // 2)
        return np.asarray(out)

    # A porous road reflects less as frequency rises, so the reflection is the
    # incident pulse through a gentle low-pass as well as the band filter.
    porous = scipy_signal.firwin(21, 2500.0, fs=fs)
    porous = porous / porous.sum()
    kr = float(materials.geometric_spreading_factor())
    h_free = arrival(t_direct, 1.0)
    reflection = arrival(t_reflected, kr * 0.85, porous)
    h_road = h_free + reflection + arrival(t_parasite, 0.18)
    h_reflected = h_road - h_free  # the subtraction technique
    scale = float(np.abs(h_road).max())
    h_free, h_road, h_reflected = (h_free / scale, h_road / scale, h_reflected / scale)

    window = np.asarray(materials.adrienne_window(fs))
    lead = 0.5e-3

    def place(w: np.ndarray, start: float) -> np.ndarray:
        out = np.zeros(n)
        i = round(start * fs)
        out[i : i + w.size] = w[: max(0, n - i)]
        return out

    gate = place(window, t_reflected - lead)
    gate_incident = place(window, t_direct - lead)
    duration = window.size / fs

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.6))
    ms = t * 1e3
    top, mid, bot = axes

    top.plot(
        ms,
        h_free,
        color=COLOR_PRIMARY,
        linewidth=1.3,
        zorder=3,
        label=r"$h_\mathrm{i}$: free field, the rig clear of every surface",
    )
    top.plot(
        ms,
        gate_incident,
        color=COLOR_TERTIARY,
        linewidth=1.5,
        linestyle="--",
        zorder=2,
        label="the same window, on the direct sound",
    )
    top.set_ylabel("Free field")
    top.legend(loc="upper right", fontsize=9)

    mid.plot(
        ms,
        h_road,
        color=COLOR_MUTED,
        linewidth=1.2,
        zorder=2,
        label="measured over the road: the two arrivals overlap",
    )
    mid.plot(
        ms,
        h_reflected,
        color=COLOR_SECONDARY,
        linewidth=1.5,
        zorder=4,
        label=r"$h_\mathrm{r}$ = road - free field, the surface alone",
    )
    mid.plot(
        ms,
        gate,
        color=COLOR_TERTIARY,
        linewidth=1.6,
        zorder=3,
        label=f"Adrienne window: 0.5 + 5 + 5 ms = {duration * 1e3:.1f} ms",
    )
    mid.annotate(
        "",
        xy=(t_direct * 1e3, -0.60),
        xytext=(t_reflected * 1e3, -0.60),
        arrowprops={"arrowstyle": "<->", "color": COLOR_FG},
    )
    mid.annotate(
        f"$2 d_\\mathrm{{m}}/c$ = {(t_reflected - t_direct) * 1e3:.2f} ms",
        (0.5 * (t_direct + t_reflected) * 1e3, -0.66),
        fontsize=9,
        color=COLOR_FG,
        ha="center",
        va="top",
    )
    mid.annotate(
        "parasitic arrival, just outside\nthe gate: lengthen the window\n"
        "and it comes in with the road",
        (t_parasite * 1e3 - 0.5, -0.95),
        fontsize=9,
        color=COLOR_FG,
        ha="right",
        va="bottom",
    )
    mid.set_ylabel("Over the road")
    mid.set_xlabel("Time [ms]")
    mid.legend(loc="upper right", fontsize=9)

    for ax in (top, mid):
        ax.set_xlim(0.0, 18.0)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        return freqs, 20.0 * np.log10(np.abs(np.fft.rfft(x)) + 1e-12)

    f_i, mag_i = spectrum(h_free * gate_incident)
    _, mag_r = spectrum(h_reflected * gate)
    peak = float(mag_i.max())
    mag_i, mag_r = mag_i - peak, mag_r - peak
    keep = (f_i >= 100.0) & (f_i <= 5000.0)
    bot.semilogx(
        f_i[keep],
        mag_i[keep],
        color=COLOR_PRIMARY,
        linewidth=1.6,
        zorder=3,
        label=r"$|H_\mathrm{i}|$, the windowed free-field reference",
    )
    bot.semilogx(
        f_i[keep],
        mag_r[keep],
        color=COLOR_SECONDARY,
        linewidth=1.6,
        zorder=3,
        label=r"$|H_\mathrm{r}|$, the windowed surface reflection",
    )
    bot.axvspan(100.0, 250.0, color=theme_fill(COLOR_MUTED, bot), zorder=0)
    bot.axvline(1.0 / duration, color=COLOR_FG, linestyle=":", linewidth=1.3, zorder=2)
    bot.annotate(
        f"$1/T$ = {1.0 / duration:.0f} Hz:\nthe window's own\nlow-frequency limit",
        (1.0 / duration + 6.0, -26.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="bottom",
    )
    bot.annotate(
        "below the reported 250 Hz",
        (104.0, -43.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="bottom",
    )
    bot.set_ylim(-46.0, 4.0)
    bot.set_ylabel("Level [dB re max]")
    bot.set_xlabel(LABEL_FREQ_HZ)
    format_frequency_axis(bot, 100.0, 5000.0)
    bot.legend(loc="lower right", fontsize=9)
    bot.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    bot.set_axisbelow(True)

    top.set_title("The one free parameter of ISO 13472-1", pad=12)
    fig.align_ylabels(axes)
    plt.tight_layout()
    save_figure(output_dir, "adrienne_window.svg")
    plt.close()


def generate_insitu_method_windows(output_dir: str) -> None:
    """Subtraction or spot: the two reported bands, and what sets the ceiling."""
    print("Generating insitu_method_windows...")
    from phonometry import materials

    c0 = 343.0
    _fig, (ax_b, ax_d) = plt.subplots(
        2,
        1,
        figsize=(10, 7.2),
        gridspec_kw={"height_ratios": [1.0, 1.5]},
    )

    # Top: the two reported one-third-octave ranges on a common log axis, with
    # the interval where Part 2's introduction expects the two to agree.
    bars = (
        ("ISO 13472-1, subtraction", 250.0, 4000.0, COLOR_PRIMARY, 1.0),
        ("ISO 13472-2, spot tube", 250.0, 1600.0, COLOR_SECONDARY, 0.0),
    )
    for label, lo, hi, color, y in bars:
        ax_b.barh(
            y,
            hi - lo,
            left=lo,
            height=0.52,
            color=color,
            edgecolor=COLOR_FG,
            linewidth=0.8,
            zorder=3,
        )
        ax_b.text(
            np.sqrt(lo * hi),
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            zorder=4,
        )
    ax_b.axvspan(315.0, 1600.0, color=theme_fill(COLOR_TERTIARY, ax_b), zorder=0)
    ax_b.annotate(
        "315-1600 Hz: the interval in which\nPart 2 expects the two to agree",
        (330.0, 1.55),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax_b.set_xscale("log")
    ax_b.set_xlim(200.0, 5000.0)
    ax_b.set_ylim(-0.6, 1.8)
    ax_b.set_yticks([])
    ax_b.set_title("Two in-situ methods, two reported bands", pad=12)
    format_frequency_axis(ax_b, 200.0, 5000.0)
    ax_b.grid(axis="x", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_b.set_axisbelow(True)

    # Bottom: why the spot tube stops where it does. The plane-wave ceiling
    # f_u = 0.58 c0 / d falls as the bore grows, and the bore has to be big
    # enough to seal on a textured pavement.
    diameters = np.linspace(0.055, 0.155, 300)
    ceiling = np.array(
        [float(materials.spot_tube_upper_frequency(d, c0)) for d in diameters]
    )
    ax_d.plot(
        diameters * 1e3,
        ceiling,
        color=COLOR_SECONDARY,
        linewidth=2.2,
        zorder=4,
        label=r"$f_\mathrm{u} = 0.58\,c_0/d$ (Clause 5.4)",
    )
    ax_d.axhline(1800.0, color=COLOR_FG, linestyle="--", linewidth=1.3, zorder=3)
    ax_d.axhspan(ceiling.min(), 1800.0, color=theme_fill(COLOR_MUTED, ax_d), zorder=0)
    d_max = 0.58 * c0 / 1800.0
    ax_d.axvline(
        d_max * 1e3, color=COLOR_TERTIARY, linestyle=":", linewidth=1.6, zorder=3
    )
    ax_d.scatter(
        [100.0],
        [float(materials.spot_tube_upper_frequency(0.1, c0))],
        color=COLOR_FG,
        s=70,
        zorder=6,
    )
    ax_d.annotate(
        "the 100 mm bore of the worked\nexample: 1989 Hz",
        (102.0, 1989.4),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="bottom",
    )
    ax_d.annotate(
        f"1800 Hz, the top edge of the 1600 Hz band:\n"
        f"a bore above {d_max * 1e3:.0f} mm no longer covers it",
        (56.0, 1740.0),
        fontsize=9,
        color=COLOR_FG,
        ha="left",
        va="top",
    )
    ax_d.set_xlim(55.0, 155.0)
    ax_d.set_ylim(1100.0, 3600.0)
    ax_d.set_xlabel("Tube diameter $d$ [mm]")
    ax_d.set_ylabel("Plane-wave ceiling [Hz]")
    ax_d.set_title("The bore that seals is the bore that caps the band", pad=12)
    ax_d.legend(loc="upper right", fontsize=9)
    ax_d.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_d.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "insitu_method_windows.svg")
    plt.close()


def generate_sound_absorption_measurement(output_dir: str) -> None:
    """ISO 354: reverberation-room alpha_s spectrum from the two decay times."""
    print("Generating sound_absorption_measurement...")
    from phonometry import materials

    # The materials guide's ISO 354 example: one-third-octave reverberation
    # times of a 200 m^3 room, empty (T1) and with a 10.8 m^2 porous absorber
    # sample installed (T2), inverted through Sabine to the alpha_s spectrum.
    freqs = np.array(
        [
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
        float,
    )
    t_empty = np.array(
        [
            9.0,
            9.0,
            8.8,
            8.6,
            8.4,
            8.2,
            8.0,
            7.8,
            7.5,
            7.2,
            6.9,
            6.6,
            6.2,
            5.8,
            5.4,
            5.0,
            4.6,
            4.2,
        ]
    )
    t_specimen = np.array(
        [
            8.4,
            8.2,
            7.7,
            7.2,
            6.5,
            5.7,
            4.9,
            4.2,
            3.6,
            3.15,
            2.85,
            2.65,
            2.55,
            2.5,
            2.55,
            2.6,
            2.7,
            2.85,
        ]
    )
    m = materials.measure_sound_absorption(
        freqs, t_empty, t_specimen, volume=200.0, area=10.8, temperature=20.0
    )
    m.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "sound_absorption_measurement.svg")
    plt.close()


def generate_impedance_tube_result(output_dir: str) -> None:
    """ISO 10534-2: alpha(f) and |r| of a porous sample in a 100 mm tube."""
    print("Generating impedance_tube_result...")
    from phonometry import materials

    # A 50 mm porous absorber (Miki model, sigma = 20 kPa s/m^2) measured in a
    # 100 mm tube with 100 mm microphone spacing (working band roughly 170 Hz
    # to 1.5 kHz): the layer model provides the true reflection factor, from
    # which the measured transfer function H12 is synthesised and reduced back
    # through the Clause 7 chain.
    f = np.linspace(200.0, 1500.0, 260)
    med = materials.miki(f, 20000.0)
    layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
    spacing, x1, c0 = 0.10, 0.20, 343.2
    k0 = materials.tube_wavenumber(f, c0)
    x2 = x1 - spacing
    r_true = layer.reflection
    h12 = (np.exp(1j * k0 * x2) + r_true * np.exp(-1j * k0 * x2)) / (
        np.exp(1j * k0 * x1) + r_true * np.exp(-1j * k0 * x1)
    )
    result = materials.two_microphone_impedance(
        h12,
        frequency=f,
        spacing=spacing,
        x1=x1,
        speed_of_sound=c0,
        characteristic_impedance=407.0,
        diameter=0.10,
    )
    result.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "impedance_tube_result.svg")
    plt.close()


def generate_transfer_matrix_tl(output_dir: str) -> None:
    """ASTM E2611: TL and hard-backed absorption from the four-pole matrix."""
    print("Generating transfer_matrix_tl...")
    from phonometry import materials

    # The chain matrix of a 50 mm porous layer (Miki, sigma = 20 kPa s/m^2)
    # exposed by the layer solver, read back through the ASTM E2611 machinery:
    # the four-pole entries give the normal-incidence transmission loss and the
    # hard-backed absorption of the same specimen.
    f = np.linspace(200.0, 1600.0, 300)
    med = materials.miki(f, 20000.0)
    layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
    chain = layer.transfer_matrix
    tm = materials.TransferMatrix(
        t11=chain[0, 0], t12=chain[0, 1], t21=chain[1, 0], t22=chain[1, 1]
    )
    tm.plot(f, 407.0, language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "transfer_matrix_tl.svg")
    plt.close()


def generate_porous_medium_model(output_dir: str) -> None:
    """Miki equivalent fluid: normalised Zc and k of a porous material."""
    print("Generating porous_medium_model...")
    from phonometry import materials

    # The porous-absorbers guide's model example: a rockwool-class material of
    # flow resistivity 20 kPa s/m^2 evaluated with the Miki (1990) regression.
    f = np.geomspace(100.0, 5000.0, 260)
    med = materials.miki(f, 20000.0)
    med.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "porous_medium_model.svg")
    plt.close()


def generate_mpp_absorption_peak(output_dir: str) -> None:
    """Maa (1998) Fig. 5 microperforated panel: resonant absorption peak."""
    print("Generating mpp_absorption_peak...")
    from phonometry import materials

    # Maa's own design: d = t = 0.2 mm, holes every 2.5 mm, 6 cm cavity. The
    # viscous losses in the submillimetre holes absorb without any porous
    # material; the peak reaches alpha = 0.96 near 677 Hz.
    eps = (np.pi / 4.0) * (0.2 / 2.5) ** 2
    f = np.linspace(100.0, 4000.0, 1200)
    res = materials.layered_absorber(
        f,
        [
            materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, eps),
            materials.AirLayer(0.06),
        ],
    )
    ax = res.plot(language=_LANG)
    format_frequency_axis(ax, 100.0, 4000.0)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "mpp_absorption_peak.svg")
    plt.close()


def generate_diffuse_field_absorption(output_dir: str) -> None:
    """Paris integral: random-incidence vs normal-incidence absorption."""
    print("Generating diffuse_field_absorption...")
    from phonometry import materials

    # A 50 mm porous layer (Miki, sigma = 20 kPa s/m^2): the diffuse-field
    # coefficient exceeds the normal-incidence one at low frequency because
    # the oblique waves travel a longer path inside the layer.
    f = np.geomspace(125.0, 4000.0, 200)
    layers: list[Any] = [materials.PorousLayer(0.05, materials.miki(f, 20000.0))]
    normal = materials.layered_absorber(f, layers)
    diffuse = materials.diffuse_field_absorption(f, layers)
    ax = diffuse.plot(language=_LANG)
    ax.plot(
        f,
        normal.absorption,
        ls="--",
        color=COLOR_SECONDARY,
        label=r"Normal incidence $\alpha(0°)$",
    )
    ax.legend(loc="best", fontsize="small")
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "diffuse_field_absorption.svg")
    plt.close()


def generate_sound_absorption_inversion(output_dir: str) -> None:
    """ISO 354 inversion: the two decay times and the two Sabine areas behind alpha_s."""
    print("Generating sound_absorption_inversion...")
    from phonometry import materials

    # The absorption-measurement guide's own example: a 10.8 m^2 specimen in a
    # 200 m^3 room at 20 degC. Everything drawn is a field of the result.
    freqs = np.array(
        [
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
        ],
        float,
    )
    t_empty = np.array(
        [
            9.0,
            9.0,
            8.8,
            8.6,
            8.4,
            8.2,
            8.0,
            7.8,
            7.5,
            7.2,
            6.9,
            6.6,
            6.2,
            5.8,
            5.4,
            5.0,
            4.6,
            4.2,
        ]
    )
    t_specimen = np.array(
        [
            8.4,
            8.2,
            7.7,
            7.2,
            6.5,
            5.7,
            4.9,
            4.2,
            3.6,
            3.15,
            2.85,
            2.65,
            2.55,
            2.5,
            2.55,
            2.6,
            2.7,
            2.85,
        ]
    )
    area = 10.8
    meas = materials.measure_sound_absorption(
        freqs, t_empty, t_specimen, volume=200.0, area=area, temperature=20.0
    )

    band = np.arange(freqs.size)
    _fig, (ax_t, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(10, 7.6))

    ax_t.plot(
        band,
        meas.t_empty,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.0,
        markersize=5,
        label=r"$T_1$  empty room",
    )
    ax_t.plot(
        band,
        meas.t_specimen,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.0,
        markersize=5,
        label=r"$T_2$  specimen installed",
    )
    ax_t.fill_between(
        band,
        meas.t_specimen,
        meas.t_empty,
        color=theme_fill(COLOR_SECONDARY, ax_t),
        zorder=0,
    )
    ax_t.set_ylabel("Reverberation time [s]")
    ax_t.set_title("ISO 354: What the Sabine Inversion Consumes", pad=12)
    ax_t.set_ylim(bottom=0.0)
    ax_t.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_t.set_axisbelow(True)
    ax_t.legend(loc="lower left", fontsize=10)
    ax_t.annotate(
        "the two decays nearly coincide:\n0.6 s out of 9 s",
        xy=(0.0, 8.7),
        xytext=(1.2, 5.4),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )

    a_1 = meas.absorption_area_empty
    a_2 = meas.absorption_area_with_specimen
    ax_a.plot(
        band,
        a_1,
        "o-",
        color=COLOR_PRIMARY,
        linewidth=2.0,
        markersize=5,
        label=r"$A_1$  empty room",
    )
    ax_a.plot(
        band,
        a_2,
        "s-",
        color=COLOR_SECONDARY,
        linewidth=2.0,
        markersize=5,
        label=r"$A_2$  with specimen",
    )
    ax_a.fill_between(
        band,
        a_1,
        a_2,
        color=theme_fill(COLOR_TERTIARY, ax_a),
        zorder=0,
        label=r"$A_2 - A_1$  the specimen",
    )
    ax_a.set_ylabel("Equivalent absorption area [m²]")
    ax_a.set_xlabel(LABEL_FREQ_HZ)
    ax_a.set_ylim(bottom=0.0)
    ax_a.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", fontsize=10)

    # One band read explicitly, so the right-hand axis has a worked example:
    # the HEIGHT of the filled band, not the level of a curve, is alpha_s.
    peak = int(np.argmax(meas.alpha_s))
    ax_a.annotate(
        "",
        xy=(band[peak], a_2[peak]),
        xytext=(band[peak], a_1[peak]),
        arrowprops={"arrowstyle": "<->", "lw": 1.4, "color": COLOR_FG},
    )
    ax_a.annotate(
        f"{a_2[peak] - a_1[peak]:.1f} m² / 10.8 m² = {meas.alpha_s[peak]:.2f}",
        xy=(band[peak], 0.5 * (a_1[peak] + a_2[peak])),
        xytext=(band[peak] - 6.4, 10.6),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )

    # Right-hand axis: the same filled difference read as alpha_s = (A2-A1)/S.
    ax_alpha = ax_a.twinx()
    ax_alpha.set_ylim(ax_a.get_ylim()[0] / area, ax_a.get_ylim()[1] / area)
    ax_alpha.set_ylabel(
        r"Band height as $\alpha_\mathrm{s} = (A_2 - A_1)/S$", color=COLOR_TERTIARY
    )
    ax_alpha.tick_params(axis="y", colors=COLOR_TERTIARY)

    ax_a.set_xticks(band)
    ax_a.set_xticklabels(
        [f"{f:g}" if f < 1000 else f"{f / 1000:g}k" for f in freqs], rotation=45
    )
    plt.tight_layout()
    save_figure(output_dir, "sound_absorption_inversion.svg")
    plt.close()


def generate_effective_kappa(output_dir: str) -> None:
    """ISO 9053-2 Annex A: kappa' over the 1-4 Hz piston band for three cavities."""
    print("Generating effective_kappa...")
    from phonometry import materials

    # The Annex A.3 cavity is a closed 100 mm x 100 mm cylinder:
    # V = 7.854e-4 m^3, S = 0.0471 m^2, so S/V = 60 1/m exactly. The two other
    # curves scale that ratio by a factor of two either way at the same volume.
    volume = 7.854e-4
    kappa_adiabatic = 1.4008
    freq = np.linspace(1.0, 4.0, 160)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for s_over_v, color in (
        (30.0, COLOR_TERTIARY),
        (60.0, COLOR_PRIMARY),
        (120.0, COLOR_SECONDARY),
    ):
        kappa = np.array(
            [
                materials.effective_kappa(
                    cavity_surface=s_over_v * volume,
                    cavity_volume=volume,
                    frequency=float(f),
                )
                for f in freq
            ]
        )
        ax.plot(
            freq, kappa, color=color, linewidth=2.2, label=f"$S/V$ = {s_over_v:g} 1/m"
        )
    ax.axhline(
        kappa_adiabatic, color=COLOR_FG, linestyle="--", linewidth=1.4, alpha=0.7
    )
    ax.text(
        3.92,
        kappa_adiabatic - 0.0035,
        r"adiabatic $\kappa$ = 1.4008",
        fontsize=10,
        color=COLOR_FG,
        ha="right",
        va="top",
    )

    kappa_a3 = materials.effective_kappa(
        cavity_surface=0.0471, cavity_volume=volume, frequency=2.0
    )
    ax.scatter(
        [2.0],
        [kappa_a3],
        color=COLOR_FG,
        s=80,
        zorder=6,
        label=f"Annex A.3 example (2 Hz, {kappa_a3:.3f})",
    )

    ax.set_xlabel("Piston frequency $f$ [Hz]  (ISO 9053-2 Clause 6.2: 1 Hz to 4 Hz)")
    ax.set_ylabel(r"Effective ratio of specific heats $\kappa^{\prime}$")
    ax.set_title("ISO 9053-2 Annex A Heat-Conduction Correction", pad=12)
    ax.set_xlim(1.0, 4.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10)

    # Right axis: R is proportional to kappa' in Formula (2), so the shortfall
    # against the adiabatic value is the bias skipping the correction causes.
    # 1 - kappa'/kappa is linear in kappa', so it maps onto a twin axis exactly.
    ax_err = ax.twinx()
    lo, hi = ax.get_ylim()
    ax_err.set_ylim(
        100.0 * (1.0 - lo / kappa_adiabatic), 100.0 * (1.0 - hi / kappa_adiabatic)
    )
    ax_err.set_ylabel("Bias in $R$ if the adiabatic value is used [%]")
    plt.tight_layout()
    save_figure(output_dir, "effective_kappa.svg")
    plt.close()


def generate_flow_resistivity_window(output_dir: str) -> None:
    """The sigma*d design window: five 50 mm layers, and alpha against sigma d / rho0 c0."""
    print("Generating flow_resistivity_window...")
    from phonometry import materials

    thickness = 0.05
    rho_c = 1.205 * 343.0
    _fig, (ax_f, ax_w) = plt.subplots(1, 2, figsize=(12.4, 5.8))

    # Left: five resistivities spanning the window and both sides of it.
    freq = np.geomspace(100.0, 5000.0, 240)
    # The two out-of-window extremes are drawn broken, the three inside solid.
    family = (
        (2e3, COLOR_MUTED, (0, (6, 3)), "too transparent"),
        (8e3, COLOR_TERTIARY, "-", "window"),
        (20e3, COLOR_PRIMARY, "-", "window"),
        (33e3, COLOR_SECONDARY, "-", "window"),
        (100e3, COLOR_FG, (0, (1, 2)), "too reflecting"),
    )
    for sigma, color, style, tag in family:
        layer = materials.layered_absorber(
            freq, [materials.PorousLayer(thickness, materials.miki(freq, sigma))]
        )
        ax_f.semilogx(
            freq,
            layer.absorption,
            color=color,
            linewidth=2.0,
            linestyle=style,
            label=f"{sigma / 1e3:g} kPa·s/m²  ({tag})",
        )
    format_frequency_axis(ax_f, 100.0, 5000.0)
    ax_f.set_xlabel(LABEL_FREQ_HZ)
    ax_f.set_ylabel(r"Normal-incidence absorption $\alpha$")
    ax_f.set_title("50 mm hard-backed layer", pad=10)
    ax_f.set_ylim(0.0, 1.0)
    ax_f.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_f.set_axisbelow(True)
    ax_f.legend(loc="upper left", fontsize=9)

    # Right: the same family against the dimensionless total flow resistance.
    sigmas = np.geomspace(1e3, 3e5, 90)
    ratio = sigmas * thickness / rho_c
    for band, color, marker in (
        (500.0, COLOR_SECONDARY, "o"),
        (1000.0, COLOR_PRIMARY, "s"),
    ):
        alpha = np.array(
            [
                float(
                    materials.layered_absorber(
                        np.array([band]),
                        [
                            materials.PorousLayer(
                                thickness, materials.miki(np.array([band]), s)
                            )
                        ],
                    ).absorption[0]
                )
                for s in sigmas
            ]
        )
        ax_w.semilogx(
            ratio,
            alpha,
            color=color,
            linewidth=2.0,
            marker=marker,
            markevery=12,
            markersize=5,
            label=rf"$\alpha(0°)$ at {band:.0f} Hz",
        )
    alpha_dif = np.array(
        [
            float(
                materials.diffuse_field_absorption(
                    np.array([1000.0]),
                    [
                        materials.PorousLayer(
                            thickness, materials.miki(np.array([1000.0]), s)
                        )
                    ],
                ).absorption[0]
            )
            for s in sigmas
        ]
    )
    ax_w.semilogx(
        ratio,
        alpha_dif,
        color=COLOR_TERTIARY,
        linewidth=2.0,
        linestyle="--",
        label=r"$\alpha_\mathrm{dif}$ at 1 kHz",
    )
    ax_w.axvspan(1.0, 4.0, color=theme_fill(COLOR_PRIMARY, ax_w), linewidth=0, zorder=0)
    ax_w.text(
        2.0,
        0.06,
        r"$\rho_0 c_0 \leq \sigma d \leq 4\rho_0 c_0$",
        fontsize=10,
        ha="center",
        color=COLOR_FG,
    )
    ax_w.set_xlabel(r"$\sigma\,d\,/\,(\rho_0 c_0)$")
    ax_w.set_ylabel(r"Absorption coefficient")
    ax_w.set_title("The design window", pad=10)
    ax_w.set_ylim(0.0, 1.0)
    ax_w.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.set_axisbelow(True)
    ax_w.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "flow_resistivity_window.svg")
    plt.close()


def generate_tube_working_ranges(output_dir: str) -> None:
    """Which impedance-tube geometry covers which one-third-octave bands."""
    print("Generating tube_working_ranges...")
    from phonometry import materials

    c0 = 343.2
    # Four geometries: the two spacings of a 100 mm bore, a 29 mm small tube,
    # and the same 100 mm bore read under the ASTM E2611 constants.
    rows = (
        ("100 mm bore, $s$ = 100 mm", 0.100, 0.100, False, COLOR_PRIMARY),
        ("100 mm bore, $s$ = 50 mm", 0.050, 0.100, False, COLOR_SECONDARY),
        ("29 mm bore, $s$ = 20 mm", 0.020, 0.029, False, COLOR_TERTIARY),
        ("100 mm bore, $s$ = 100 mm (ASTM E2611)", 0.100, 0.100, True, COLOR_MUTED),
    )

    _fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, (label, spacing, diameter, astm, color) in enumerate(rows):
        band = (
            materials.plane_wave_frequency_range_astm
            if astm
            else materials.plane_wave_frequency_range
        )
        f_l, f_u = band(spacing, c0, diameter=diameter)
        y = len(rows) - 1 - i
        ax.plot(
            [f_l, f_u],
            [y, y],
            color=color,
            linewidth=13,
            solid_capstyle="butt",
            alpha=0.85,
        )
        ax.text(
            f_l * 0.94,
            y,
            f"{f_l:.0f}",
            ha="right",
            va="center",
            fontsize=9,
            color=COLOR_FG,
        )
        ax.text(
            f_u * 1.06,
            y,
            f"{f_u:.0f} Hz",
            ha="left",
            va="center",
            fontsize=9,
            color=COLOR_FG,
        )
        # Which constraint binds the top end: cut-on or the spacing singularity.
        cut_on = (0.586 if astm else 0.58) * c0 / diameter
        binding = (
            r"cut-on $0.58\,c/d$"
            if abs(f_u - cut_on) < 1.0
            else (r"spacing $0.40\,c/s$" if astm else r"spacing $0.45\,c/s$")
        )
        ax.text(
            np.sqrt(f_l * f_u),
            y + 0.30,
            f"{label}   ·   top end: {binding}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLOR_FG,
        )

    # The splice band the large and the small tube share.
    big = materials.plane_wave_frequency_range(0.100, c0, diameter=0.100)
    small = materials.plane_wave_frequency_range(0.020, c0, diameter=0.029)
    ax.axvspan(small[0], big[1], color=theme_fill(COLOR_TERTIARY, ax), zorder=0)
    ax.text(
        np.sqrt(small[0] * big[1]),
        -0.75,
        "splice band\n(the two tubes must agree)",
        ha="center",
        va="center",
        fontsize=9.5,
        color=COLOR_FG,
    )

    ax.set_xscale("log")
    ax.set_xlim(25.0, 11000.0)
    ax.set_ylim(-1.2, len(rows) - 0.35)
    ax.set_yticks([])
    format_frequency_axis(ax, 25.0, 11000.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_title("Plane-Wave Working Range of an Impedance Tube", pad=12)
    ax.grid(which="both", axis="x", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "tube_working_ranges.svg")
    plt.close()


def generate_standing_wave_envelope(output_dir: str) -> None:
    """ISO 10534-1: the interior level the probe carriage traverses."""
    print("Generating standing_wave_envelope...")
    from phonometry import materials

    c0, freq, diameter = 343.2, 500.0, 0.10
    k = 2.0 * np.pi * freq / c0
    x = np.linspace(0.0, 1.0, 2400)

    def level(magnitude: Any, phi: float) -> Any:
        """Envelope in decibels, which is what the analyser shows."""
        env2 = 1.0 + magnitude**2 + 2.0 * magnitude * np.cos(2 * k * x - phi)
        return 10.0 * np.log10(np.maximum(env2, 1e-6))

    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for magnitude, phi, color, label in (
        (1.0, -np.pi, COLOR_FG, r"rigid wall  $|r| = 1$  ($\Delta L \to \infty$)"),
        (
            0.5,
            np.radians(-54.1),
            COLOR_PRIMARY,
            r"the worked sample  $|r| = 0.5$  ($\Delta L$ = 9.54 dB)",
        ),
        (
            0.1,
            0.0,
            COLOR_TERTIARY,
            r"near-anechoic  $|r| = 0.1$  ($\Delta L$ = 1.74 dB)",
        ),
    ):
        ax.plot(x, level(magnitude, phi), color=color, linewidth=2.0, label=label)

    # The same |r| = 0.5 sample with the Eq. (A.18) tube attenuation: the
    # reflected wave has travelled 2x further, so the far notches fill in.
    atten = float(materials.tube_attenuation_constant(freq, c0, diameter))
    r_eff = 0.5 * np.exp(-2.0 * atten * x)
    lossy = 10.0 * np.log10(
        1.0 + r_eff**2 + 2.0 * r_eff * np.cos(2 * k * x + np.radians(54.1))
    )
    ax.plot(
        x,
        lossy,
        color=COLOR_SECONDARY,
        linewidth=1.8,
        linestyle=":",
        label=rf"the same, with tube loss ($k_0^{{\prime\prime}}$ = {atten:.3f} Np/m)",
    )

    # How far the far minima have filled in, in decibels.
    minima = [float(lossy[np.argmin(np.abs(x - xm))]) for xm in (0.12, 0.463, 0.806)]
    ax.annotate(
        f"minima at 12, 46 and 81 cm: {_fmt_minus(minima[0], '.2f')}, "
        f"{_fmt_minus(minima[1], '.2f')}, {_fmt_minus(minima[2], '.2f')} dB\n"
        "(read the nearest one, and extrapolate to $x$ = 0)",
        xy=(0.806, minima[2]),
        xytext=(0.27, -14.6),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )

    # x_min,1 of the worked sample, and the quarter-wavelength ruler.
    x_min1 = 0.12
    ax.axvline(x_min1, color=COLOR_GRID, linestyle="--", linewidth=1.2)
    ax.annotate(
        r"$x_{\min,1}$ = 12 cm",
        xy=(x_min1, -6.0),
        xytext=(0.16, -11.0),
        fontsize=10,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    quarter = 0.25 * c0 / freq
    ax.annotate(
        "",
        xy=(0.0, 7.2),
        xytext=(quarter, 7.2),
        arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": COLOR_FG},
    )
    ax.text(
        quarter / 2,
        7.6,
        r"$\lambda/4$ = 17.2 cm",
        ha="center",
        fontsize=9.5,
        color=COLOR_FG,
    )
    ax.text(
        0.005, -19.4, "specimen face  $x$ = 0", ha="left", fontsize=9.5, color=COLOR_FG
    )

    ax.set_xlabel("Distance from the specimen face $x$ [m]  (towards the source)")
    ax.set_ylabel(r"Envelope level $20\log_{10}(|p(x)|/A)$ [dB]")
    ax.set_title("What the Probe Carriage Traverses (500 Hz, 100 mm tube)", pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-20.0, 15.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "standing_wave_envelope.svg")
    plt.close()


def generate_porous_model_comparison(output_dir: str) -> None:
    """Delany-Bazley, Miki and JCA: agreement inside the fit window, and outside."""
    print("Generating porous_model_comparison...")
    from phonometry import materials

    sigma = 20000.0
    freq = np.geomspace(20.0, 20000.0, 400)
    media = (
        ("Delany-Bazley", materials.delany_bazley(freq, sigma), COLOR_SECONDARY, "-"),
        ("Miki", materials.miki(freq, sigma), COLOR_PRIMARY, "--"),
        (
            "JCA",
            materials.johnson_champoux_allard(
                freq,
                sigma,
                porosity=0.98,
                tortuosity=1.0,
                viscous_length=8.7e-5,
                thermal_length=8.7e-5,
            ),
            COLOR_TERTIARY,
            ":",
        ),
    )

    _fig, (ax_z, ax_s) = plt.subplots(1, 2, figsize=(12.4, 5.8))
    for label, medium, color, style in media:
        z = medium.normalized_impedance
        ax_z.loglog(
            freq,
            z.real,
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=f"{label}, $\\mathrm{{Re}}$",
        )
        # The imaginary part is the companion of its own real part, so it is
        # drawn in a quieter shade of the same colour rather than at a lower
        # opacity: half opacity of the red and the blue composites to within a
        # couple of levels of the dark page and only the green survived.
        ax_z.loglog(
            freq,
            -z.imag,
            color=theme_line(color, ax_z, quiet=0.55),
            linestyle=style,
            linewidth=1.2,
            label=f"{label}, $-\\mathrm{{Im}}$",
        )
    # The Delany-Bazley fit window in its own variable, X = rho0 f / sigma.
    x_lo, x_hi = 0.01 * sigma / 1.205, 1.0 * sigma / 1.205
    ax_z.axvspan(x_lo, x_hi, color=theme_fill(COLOR_PRIMARY, ax_z), zorder=0)
    ax_z.text(
        np.sqrt(x_lo * x_hi),
        0.6,
        "$0.01 < X < 1$",
        ha="center",
        fontsize=10,
        color=COLOR_FG,
    )
    format_frequency_axis(ax_z, 20.0, 20000.0)
    ax_z.set_xlabel(LABEL_FREQ_HZ)
    ax_z.set_ylabel(r"$Z_\mathrm{c}/(\rho_0 c_0)$")
    ax_z.set_title(r"Characteristic impedance, $\sigma$ = 20 kPa·s/m²", pad=10)
    ax_z.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_z.set_axisbelow(True)
    ax_z.legend(loc="upper right", fontsize=8.5, ncol=2)

    # Right: the input impedance of a 50 mm hard-backed layer, where the
    # extrapolation failure the guide warns about actually shows.
    for label, medium, color, style in media:
        zs = materials.layered_absorber(
            freq, [materials.PorousLayer(0.05, medium)]
        ).normalized_impedance
        ax_s.semilogx(
            freq, zs.real, color=color, linestyle=style, linewidth=2.0, label=label
        )
    ax_s.axhline(0.0, color=COLOR_FG, linewidth=1.2)
    ax_s.axvspan(20.0, 74.6, color=theme_fill(COLOR_SECONDARY, ax_s), zorder=0)
    ax_s.annotate(
        "Delany-Bazley returns a NEGATIVE resistance\n"
        "below 74.6 Hz: a passive layer generating energy",
        xy=(40.0, -3.0),
        xytext=(150.0, -5.4),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    format_frequency_axis(ax_s, 20.0, 20000.0)
    ax_s.set_xlabel(LABEL_FREQ_HZ)
    ax_s.set_ylabel(
        r"$\mathrm{Re}(Z_\mathrm{s})/(\rho_0 c_0)$,"
        r"  50 mm hard-backed layer"
    )
    ax_s.set_title("Where the extrapolation fails", pad=10)
    ax_s.set_ylim(-8.0, 8.0)
    ax_s.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_s.set_axisbelow(True)
    ax_s.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "porous_model_comparison.svg")
    plt.close()


def generate_biot_waves(output_dir: str) -> None:
    """The three Biot wavenumbers, and the ratios that name two of them."""
    print("Generating biot_waves...")
    from phonometry import materials

    # Allard & Atalla Table 6.1 glass wool, the same input set as the
    # frame-resonance figure on the same page.
    freq = np.linspace(50.0, 1500.0, 1451)
    shear = 2.2e6 * (1 + 0.1j)
    medium = materials.johnson_champoux_allard(
        freq,
        40e3,
        porosity=0.94,
        tortuosity=1.06,
        viscous_length=0.56e-4,
        thermal_length=1.1e-4,
    )
    waves = materials.biot_waves(
        medium, porosity=0.94, tortuosity=1.06, frame_density=130.0, shear_modulus=shear
    )

    fig, (ax_k, ax_mu) = plt.subplots(
        2, 1, figsize=(10, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    waves.plot(ax=ax_k, language=_LANG)

    # Where the solver's root ordering swaps: the airborne branch changes
    # eigenvalue, which is bookkeeping and not physics.
    swap = freq[np.flatnonzero(np.diff(waves.airborne_is_second.astype(int)))]
    if swap.size:
        ax_k.axvline(float(swap[0]), color=COLOR_FG, linestyle=":", linewidth=1.4)
        ax_k.text(
            float(swap[0]) * 1.04,
            ax_k.get_ylim()[1] * 0.94,
            f"root swap, {swap[0]:.0f} Hz",
            fontsize=9.5,
            color=COLOR_FG,
            va="top",
        )
    ax_k.set_xlabel("")

    mu_a = np.abs(waves.airborne_velocity_ratio)
    mu_b = np.abs(waves.frame_borne_velocity_ratio)
    ax_mu.semilogy(
        freq,
        mu_a,
        color=COLOR_PRIMARY,
        linewidth=2.0,
        label=r"airborne  $|\mu_\mathrm{a}|$",
    )
    ax_mu.semilogy(
        freq,
        mu_b,
        color=COLOR_SECONDARY,
        linewidth=2.0,
        label=r"frame-borne  $|\mu_\mathrm{b}|$",
    )
    ax_mu.axhline(1.0, color=COLOR_FG, linestyle="--", linewidth=1.0)
    ax_mu.text(
        1460.0,
        1.12,
        "fluid and frame move together",
        fontsize=9,
        color=COLOR_FG,
        ha="right",
    )
    ax_mu.annotate(
        f"$|\\mu_\\mathrm{{a}}| \\geq$ {mu_a.min():.0f} everywhere: "
        "the fluid moves "
        "and the frame barely does",
        xy=(700.0, float(np.interp(700.0, freq, mu_a))),
        xytext=(300.0, 6.0),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    ax_mu.set_xscale("log")
    format_frequency_axis(ax_mu, 50.0, 1500.0)
    ax_mu.set_xlabel(LABEL_FREQ_HZ)
    ax_mu.set_ylabel(r"$|\mu|$ = fluid / frame displacement")
    ax_mu.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_mu.set_axisbelow(True)
    ax_mu.legend(loc="lower left", fontsize=9.5)
    fig.tight_layout()
    save_figure(output_dir, "biot_waves.svg")
    plt.close()


def generate_oblique_absorption(output_dir: str) -> None:
    """alpha(theta) of a hard-backed layer, bulk against locally reacting."""
    print("Generating oblique_absorption...")
    from phonometry import materials

    theta = np.radians(np.linspace(0.0, 88.0, 90))
    _fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(12.4, 5.8))

    for band, color in (
        (250.0, COLOR_TERTIARY),
        (500.0, COLOR_SECONDARY),
        (1000.0, COLOR_PRIMARY),
        (2000.0, COLOR_FG),
    ):
        one = np.array([band])
        layers = (materials.PorousLayer(0.05, materials.miki(one, 20000.0)),)
        bulk = np.array(
            [
                float(
                    materials.layered_absorber(one, layers, angle=float(t)).absorption[
                        0
                    ]
                )
                for t in theta
            ]
        )
        z_normal = complex(
            materials.layered_absorber(one, layers).normalized_impedance[0]
        )
        # Locally reacting: the same surface impedance at every angle.
        refl = (z_normal * np.cos(theta) - 1.0) / (z_normal * np.cos(theta) + 1.0)
        local = 1.0 - np.abs(refl) ** 2
        ax_t.plot(
            np.degrees(theta),
            bulk,
            color=color,
            linewidth=2.0,
            label=f"{band:.0f} Hz, bulk",
        )
        ax_t.plot(
            np.degrees(theta),
            local,
            color=color,
            linewidth=1.4,
            linestyle="--",
            alpha=0.75,
        )
    ax_t.axvline(78.0, color=COLOR_GRID, linestyle=":", linewidth=1.6)
    ax_t.text(
        77.0,
        0.06,
        "78° truncation",
        rotation=90,
        fontsize=9.5,
        color=COLOR_FG,
        ha="right",
    )
    ax_t.set_xlabel(r"Incidence angle $\theta$ [°]")
    ax_t.set_ylabel(r"$\alpha(\theta)$")
    ax_t.set_title(
        "The integrand: solid bulk-reacting, dashed locally reacting", pad=10
    )
    ax_t.set_xlim(0.0, 90.0)
    ax_t.set_ylim(0.0, 1.0)
    ax_t.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_t.set_axisbelow(True)
    ax_t.legend(loc="lower left", fontsize=9)

    freq = np.geomspace(125.0, 4000.0, 90)
    layers_f = (materials.PorousLayer(0.05, materials.miki(freq, 20000.0)),)
    diffuse = materials.diffuse_field_absorption(freq, layers_f).absorption
    normal = materials.layered_absorber(freq, layers_f)
    statistical = np.asarray(
        materials.statistical_absorption(normal.normalized_impedance)
    )
    ax_f.semilogx(
        freq,
        diffuse,
        color=COLOR_PRIMARY,
        linewidth=2.2,
        label=r"$\alpha_\mathrm{dif}$  bulk (Paris integral)",
    )
    ax_f.semilogx(
        freq,
        statistical,
        color=COLOR_SECONDARY,
        linewidth=2.0,
        linestyle="--",
        label=r"$\alpha_\mathrm{st}$  locally reacting (closed form)",
    )
    ax_f.semilogx(
        freq,
        normal.absorption,
        color=COLOR_MUTED,
        linewidth=1.6,
        linestyle=":",
        label=r"$\alpha(0°)$  what the tube reads",
    )
    ax_f.axhline(0.951, color=COLOR_FG, linestyle=":", linewidth=1.2)
    ax_f.text(
        3900.0,
        0.975,
        "0.951 ceiling of the closed form",
        fontsize=9,
        color=COLOR_FG,
        ha="right",
    )
    format_frequency_axis(ax_f, 125.0, 4000.0)
    ax_f.set_xlabel(LABEL_FREQ_HZ)
    ax_f.set_ylabel("Absorption coefficient")
    ax_f.set_title("The two averages, and the tube", pad=10)
    ax_f.set_ylim(0.0, 1.05)
    ax_f.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_f.set_axisbelow(True)
    ax_f.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "oblique_absorption.svg")
    plt.close()


def generate_sheet_transfer_impedance(output_dir: str) -> None:
    """Maa's MPP: where its reactance meets the cavity, and what r is there."""
    print("Generating sheet_transfer_impedance...")
    from phonometry import materials

    # Maa (1998) Fig. 5: 0.2 mm holes on a 2.5 mm pitch in a 0.2 mm plate,
    # over a 60 mm cavity -- the geometry the guide already evaluates.
    open_area = (np.pi / 4.0) * (0.2 / 2.5) ** 2
    cavity, c0, rho_c = 0.06, 343.0, 1.205 * 343.0
    freq = np.geomspace(100.0, 2500.0, 900)

    z = (
        materials.microperforated_plate_impedance(
            freq, thickness=0.2e-3, hole_radius=0.1e-3, open_area=open_area
        )
        / rho_c
    )
    # Resonance is where the panel reactance cancels the cavity's, which Maa
    # writes as omega_0 m = cot(omega_0 D / c0).
    cavity_curve = 1.0 / np.tan(2.0 * np.pi * freq * cavity / c0)
    stack = materials.layered_absorber(
        freq,
        [
            materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, open_area),
            materials.AirLayer(cavity),
        ],
    )
    peak = int(np.argmax(stack.absorption))

    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax_a = ax.twinx()
    ax_a.plot(
        freq, stack.absorption, color=COLOR_TERTIARY, linewidth=1.6, alpha=0.8, zorder=1
    )
    ax_a.set_ylabel(r"Absorption $\alpha$ of the stack", color=COLOR_TERTIARY)
    ax_a.tick_params(axis="y", colors=COLOR_TERTIARY)
    ax_a.set_ylim(0.0, 1.05)

    ax.set_zorder(ax_a.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(
        freq,
        z.real,
        color=COLOR_SECONDARY,
        linewidth=2.2,
        label=r"panel resistance $r$",
    )
    ax.plot(
        freq, z.imag, color=COLOR_PRIMARY, linewidth=2.2, label=r"panel reactance $x$"
    )
    ax.plot(
        freq,
        cavity_curve,
        color=COLOR_FG,
        linewidth=1.8,
        linestyle="--",
        label=r"cavity $\cot(\omega D/c_0)$,  $D$ = 60 mm",
    )
    ax.axhline(0.0, color=COLOR_GRID, linewidth=1.0)
    ax.scatter([freq[peak]], [z.real[peak]], color=COLOR_SECONDARY, s=80, zorder=6)
    ax.scatter([freq[peak]], [z.imag[peak]], color=COLOR_PRIMARY, s=60, zorder=6)
    ax.annotate(
        f"the reactances meet at {freq[peak]:.0f} Hz\n"
        f"there $r$ = {z.real[peak]:.2f}, so "
        f"$4r/(1+r)^2$ = {4 * z.real[peak] / (1 + z.real[peak]) ** 2:.2f}",
        xy=(freq[peak], z.imag[peak]),
        xytext=(freq[peak] * 1.5, 3.2),
        fontsize=10,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    ax.axvline(freq[peak], color=COLOR_GRID, linestyle=":", linewidth=1.4)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised transfer impedance $z/\rho_0 c_0$")
    ax.set_title("Where a Resonant Sheet Resonates, and How Well It Absorbs", pad=12)
    ax.set_ylim(-2.0, 5.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9.5)
    format_frequency_axis(ax, 100.0, 2500.0)
    plt.tight_layout()
    save_figure(output_dir, "sheet_transfer_impedance.svg")
    plt.close()


def _slit_design(target: float = 300.0) -> Any:
    """The guide's 300 Hz critical-coupling design, solved once."""
    from phonometry import materials

    base = materials.HelmholtzResonator(
        neck_length=1.0e-3,
        neck_side=3.0e-3,
        cavity_length=30.0e-3,
        cavity_side=27.0e-3,
    )
    return materials.critical_coupling_design(
        target, base, lattice_step=3.0e-2, period=5.0e-2
    )


def generate_critical_coupling_impedance(output_dir: str) -> None:
    """Critical coupling seen as an impedance: the sweep and the locus."""
    print("Generating critical_coupling_impedance...")
    from phonometry import materials

    design = _slit_design()
    h0 = design.slit_height
    _fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(12.4, 5.8))

    # Left: both matching conditions against the slit height, at 300 Hz.
    heights = np.linspace(0.5 * h0, 2.5 * h0, 70)
    one = np.array([300.0])
    z = np.array(
        [
            complex(
                materials.slit_helmholtz_absorber(
                    one,
                    design.resonator,
                    slit_height=float(h),
                    lattice_step=3.0e-2,
                    period=5.0e-2,
                ).normalized_impedance[0]
            )
            for h in heights
        ]
    )
    alpha = 1.0 - np.abs((z - 1.0) / (z + 1.0)) ** 2
    ax_h.plot(
        heights * 1e3,
        z.real,
        color=COLOR_SECONDARY,
        linewidth=2.2,
        label=r"$\mathrm{Re}(z)$",
    )
    ax_h.plot(
        heights * 1e3,
        z.imag,
        color=COLOR_PRIMARY,
        linewidth=2.2,
        label=r"$\mathrm{Im}(z)$",
    )
    ax_h.axhline(1.0, color=COLOR_FG, linestyle="--", linewidth=1.0)
    ax_h.axhline(0.0, color=COLOR_FG, linestyle=":", linewidth=1.0)
    ax_h.axvline(h0 * 1e3, color=COLOR_GRID, linestyle=":", linewidth=1.6)
    ax_h.scatter([h0 * 1e3, h0 * 1e3], [1.0, 0.0], color=COLOR_FG, s=55, zorder=6)
    ax_h.annotate(
        f"solved $h$ = {h0 * 1e3:.3f} mm:\n"
        f"$\\mathrm{{Re}}(z) = 1$ and $\\mathrm{{Im}}(z) = 0$ together",
        xy=(h0 * 1e3, 1.0),
        xytext=(1.35 * h0 * 1e3, 3.0),
        fontsize=9.5,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    ax_h.set_xlabel("Slit height $h$ [mm]")
    ax_h.set_ylabel(r"Normalised surface impedance $z$ at 300 Hz")
    ax_h.set_title("What the design solver solves", pad=10)
    ax_h.set_ylim(-4.0, 6.0)
    ax_h.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_h.set_axisbelow(True)
    ax_h.legend(loc="upper right", fontsize=9.5)

    ax_al = ax_h.twinx()
    ax_al.plot(heights * 1e3, alpha, color=COLOR_TERTIARY, linewidth=1.8, alpha=0.85)
    ax_al.set_ylabel(r"$\alpha$ at 300 Hz", color=COLOR_TERTIARY)
    ax_al.tick_params(axis="y", colors=COLOR_TERTIARY)
    ax_al.set_ylim(0.0, 1.05)

    # Right: the locus of z over frequency, for the three slit heights.
    freq = np.linspace(150.0, 500.0, 260)
    for factor, color, label in (
        (0.6, COLOR_SECONDARY, r"$0.6\,h$  over-damped"),
        (1.0, COLOR_PRIMARY, "$h$  critically coupled"),
        (1.7, COLOR_TERTIARY, r"$1.7\,h$  under-damped"),
    ):
        loc = materials.slit_helmholtz_absorber(
            freq,
            design.resonator,
            slit_height=factor * h0,
            lattice_step=3.0e-2,
            period=5.0e-2,
        ).normalized_impedance
        ax_l.plot(loc.real, loc.imag, color=color, linewidth=2.0, label=label)
        at300 = complex(loc[int(np.argmin(np.abs(freq - 300.0)))])
        ax_l.scatter([at300.real], [at300.imag], color=color, s=70, zorder=6)
        ax_l.text(
            at300.real + 0.08,
            at300.imag,
            "300 Hz",
            fontsize=8.5,
            color=color,
            va="center",
        )
    ax_l.scatter([1.0], [0.0], color=COLOR_FG, s=90, zorder=6, marker="x")
    ax_l.text(1.12, 0.18, r"matched  $1 + 0\mathrm{j}$", fontsize=10, color=COLOR_FG)
    ax_l.axhline(0.0, color=COLOR_GRID, linewidth=1.0)
    ax_l.axvline(1.0, color=COLOR_GRID, linewidth=1.0)
    ax_l.set_xlabel(r"Re$(z)$")
    ax_l.set_ylabel(r"Im$(z)$")
    ax_l.set_title("The locus over 150-500 Hz", pad=10)
    ax_l.set_xlim(0.0, 4.0)
    ax_l.set_ylim(-3.0, 3.0)
    ax_l.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "critical_coupling_impedance.svg")
    plt.close()


def generate_slow_sound_dispersion(output_dir: str) -> None:
    """The phase speed inside the loaded slit, and the depth it buys."""
    print("Generating slow_sound_dispersion...")
    from phonometry import materials

    design = _slit_design()
    depth, c0 = 3.0e-2, 343.0
    freq = np.geomspace(50.0, 3000.0, 500)
    res = materials.slit_helmholtz_absorber(
        freq,
        design.resonator,
        slit_height=design.slit_height,
        lattice_step=3.0e-2,
        period=5.0e-2,
    )
    ratio = 2.0 * np.pi * freq / res.effective_wavenumber.real / c0

    # The resonator's own resonance: where its shunt reactance changes sign.
    z_hr = materials.helmholtz_resonator_impedance(
        freq, design.resonator, slit_height=design.slit_height, lattice_step=3.0e-2
    )
    crossing = freq[np.flatnonzero(np.diff(np.sign(z_hr.imag)))]
    f_res = float(crossing[0]) if crossing.size else float("nan")

    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.semilogx(freq, ratio, color=COLOR_PRIMARY, linewidth=2.4, label="loaded slit")
    ax.axhline(1.0, color=COLOR_FG, linestyle="--", linewidth=1.4, label="empty slit")
    ax.axvspan(f_res, freq[-1], color=theme_fill(COLOR_MUTED, ax), zorder=0)
    ax.text(
        np.sqrt(f_res * freq[-1]),
        0.9,
        f"branch closed above the\nresonator resonance ({f_res:.0f} Hz)",
        ha="center",
        fontsize=9.5,
        color=COLOR_FG,
    )
    lowest = int(np.argmin(ratio))
    ax.scatter([freq[lowest]], [ratio[lowest]], color=COLOR_SECONDARY, s=70, zorder=6)
    ax.annotate(
        f"{ratio[lowest]:.2f} $c_0$ at {freq[lowest]:.0f} Hz",
        xy=(freq[lowest], ratio[lowest]),
        xytext=(90.0, 0.42),
        fontsize=10,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    design_ratio = float(np.interp(300.0, freq, ratio))
    ax.scatter([300.0], [design_ratio], color=COLOR_FG, s=70, zorder=6)
    ax.annotate(
        f"design point: {design_ratio:.2f} $c_0$ = {design_ratio * c0:.0f} m/s,\n"
        f"so the 30 mm depth is a quarter wave at "
        f"{design_ratio * c0 / (4 * depth):.0f} Hz",
        xy=(300.0, design_ratio),
        xytext=(330.0, 0.62),
        fontsize=10,
        color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG},
    )
    ax.text(
        60.0,
        1.06,
        "an empty 30 mm slit is a quarter wave at 2858 Hz",
        fontsize=9.5,
        color=COLOR_FG,
        ha="left",
    )
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Phase speed in the slit  $c_\mathrm{eff}/c_0$")
    ax.set_title("Slow Sound: the Phase Speed Inside a Loaded Slit", pad=12)
    ax.set_ylim(0.0, 1.25)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9.5)
    format_frequency_axis(ax, 50.0, 3000.0)
    plt.tight_layout()
    save_figure(output_dir, "slow_sound_dispersion.svg")
    plt.close()


def generate_graded_slit_absorber(output_dir: str) -> None:
    """One cell against a four-resonator chain, graded and uniform."""
    print("Generating graded_slit_absorber...")
    from phonometry import materials

    freq = np.linspace(120.0, 700.0, 700)
    single = _slit_design(320.0)
    graded = [_slit_design(t).resonator for t in (250.0, 320.0, 410.0, 520.0)]
    uniform = [single.resonator] * 4

    _fig, ax = plt.subplots(figsize=(10.5, 6.2))
    curves = (
        (
            "one cell, $L$ = 30 mm",
            single.resonator,
            single.slit_height,
            COLOR_PRIMARY,
            "-",
        ),
        ("four graded, $L$ = 120 mm", graded, 1.20e-3, COLOR_SECONDARY, "-"),
        ("four identical, $L$ = 120 mm", uniform, 1.20e-3, COLOR_TERTIARY, "--"),
    )
    step = float(freq[1] - freq[0])
    for i, (label, resonators, height, color, style) in enumerate(curves):
        alpha = materials.slit_helmholtz_absorber(
            freq, resonators, slit_height=height, lattice_step=3.0e-2, period=5.0e-2
        ).absorption
        ax.plot(freq, alpha, color=color, linewidth=2.2, linestyle=style, label=label)
        # The band above 0.8 is not one interval for a chain, so shade where
        # it actually is and report the total, not the span.
        good = alpha >= 0.8
        base_y = 0.055 + 0.045 * i
        ax.fill_between(
            freq, base_y, base_y + 0.030, where=good, color=color, linewidth=0
        )
        ax.text(
            freq[-1] - 6.0,
            base_y + 0.015,
            f"{float(np.sum(good)) * step:.0f} Hz above 0.8",
            fontsize=9.5,
            color=color,
            va="center",
            ha="right",
        )
    ax.axhline(0.8, color=COLOR_GRID, linestyle=":", linewidth=1.4)
    ax.text(640.0, 0.815, r"$\alpha$ = 0.8", fontsize=9.5, color=COLOR_FG, ha="right")
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Absorption coefficient")
    ax.set_title("What a Chain of Resonators Buys, and What It Costs", pad=12)
    ax.set_xlim(120.0, 700.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    save_figure(output_dir, "graded_slit_absorber.svg")
    plt.close()
