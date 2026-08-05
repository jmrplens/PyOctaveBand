#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the materials guides: absorbers, diffusers and impedance.

How a surface answers a sound field: porous and resonant absorbers, the
impedance and transmission tubes that measure them, the transfer-matrix and
Biot models behind them, and the diffusers (Schroeder wells and metadiffuser
alike) that scatter rather than absorb. Everything here is embedded by a page
under ``materials/``.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from phonometry._plot.common import format_frequency_axis, theme_fill_alpha

from .i18n import _LANG
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    save_figure,
)


def generate_dynamic_stiffness(output_dir: str) -> None:
    """EN 29052-1 floating-floor natural frequency f0(s') for typical floors."""
    print("Generating dynamic_stiffness...")
    from phonometry import natural_frequency

    s_mn = np.logspace(np.log10(2.0), np.log10(100.0), 300)   # MN/m3
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Two typical floating-floor masses per unit area (light vs heavy screed).
    for m, color, label in ((40.0, COLOR_SECONDARY, "m' = 40 kg/m^2"),
                             (120.0, COLOR_PRIMARY, "m' = 120 kg/m^2")):
        f0 = np.asarray(natural_frequency(s_mn * 1e6, m), dtype=float)
        ax.plot(s_mn, f0, color=color, linewidth=2.2, label=label)

    # A worked design point: s' = 10 MN/m3 on the 120 kg/m2 floor.
    s0, m0 = 10.0, 120.0
    f00 = float(natural_frequency(s0 * 1e6, m0))
    ax.scatter([s0], [f00], color=COLOR_TERTIARY, s=90, zorder=6,
               label=f"design point ({s0:g} MN/m^3, {f00:.0f} Hz)")
    ax.plot([s0, s0], [0, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)
    ax.plot([s_mn[0], s0], [f00, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dynamic stiffness per unit area $s'$ [MN/m³]")
    ax.set_ylabel(r"Natural frequency $f_0$ [Hz]")
    ax.set_title("EN 29052-1 Floating-Floor Resonance", fontweight="bold", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)

    info = [
        "f0 = (1/2pi) sqrt(s'/m')  (Formula 2)",
        "s'  = s't + s'a  (clause 8.2)",
        "s't = 4 pi^2 m't fr^2  (Formula 4)",
        "s'a = p0/(d eps) ~ 111/d MN/m^3  (NOTE)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "dynamic_stiffness.svg")
    plt.close()


def generate_absorption_uncertainty(output_dir: str) -> None:
    """ISO 12999-2 absorption-coefficient uncertainty: alpha_s with a +/-U ribbon."""
    print("Generating absorption_uncertainty...")
    from phonometry import sound_absorption_coefficient_uncertainty

    # The standard's worked Example (Table 4): a measured sound absorption
    # coefficient alpha_s per one-third-octave band and its reproducibility
    # expanded uncertainty at k = 2.
    freqs = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                      630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
                     dtype=float)
    alpha_s = np.array([0.33, 0.35, 0.39, 0.38, 0.37, 0.36, 0.36, 0.36, 0.43,
                        0.49, 0.58, 0.63, 0.68, 0.71, 0.73, 0.75, 0.77, 0.79,
                        0.81, 0.81])
    res = sound_absorption_coefficient_uncertainty(alpha_s, freqs, confidence=0.95)
    u = res.expanded_uncertainty

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(x, alpha_s - u, alpha_s + u, color=COLOR_TERTIARY, alpha=0.22,
                    zorder=0, label="+/-U (k = 2), reproducibility")
    ax.plot(x, alpha_s, "-", color=COLOR_PRIMARY, linewidth=2.4, marker="o",
            markersize=6, zorder=5, label="alpha_s (ISO 354)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound absorption coefficient")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("ISO 12999-2 Sound Absorption Coefficient Uncertainty",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "sigma_R = m alpha_s + n  (Table 1)",
        "U = k u,  k = 2  (95 %)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "absorption_uncertainty.png")
    plt.close()


def generate_absorption_rating(output_dir: str) -> None:
    """ISO 11654 alpha_w: practical curve, shifted reference, deviations (Annex A.2)."""
    print("Generating absorption_rating.png...")
    from phonometry import weighted_absorption

    # ISO 11654:1997 Annex A.2 worked example -> alpha_w = 0.60(M).
    alpha_p = [0.35, 1.00, 0.65, 0.60, 0.55]
    result = weighted_absorption(alpha_p)
    freqs = np.asarray(result.band_centers, dtype=float)
    measured = np.asarray(result.measured, dtype=float)
    shifted = np.asarray(result.shifted_reference, dtype=float)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(freqs, measured, shifted, where=(measured < shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=5, zorder=3,
                label="Shifted reference curve (ISO 11654)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=6, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Practical absorption alpha_p")

    # alpha_w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.alpha_w, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(f"alpha_w = {result.rating_label}", xy=(500, result.alpha_w),
                xytext=(600, result.alpha_w - 0.16), fontsize=12, fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    # Placed low-left, clear of the practical curve that peaks top-centre.
    for dy, text in (
        (0.30, f"Reference curve shifted by {result.shift:.2f}"),
        (0.23, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.2f}"
               f"  (limit 0.10)")),
        (0.16, (f"Absorption class {result.absorption_class}  "
               f"(shape indicator: {result.shape_indicator or 'none'})")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 11654 Weighted Sound Absorption Coefficient (Annex A.2 example)",
                 fontweight="bold", pad=12)
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
    from phonometry import static_airflow_resistance

    # A porous specimen (area 100 mm dia, 50 mm thick) measured stepwise. The
    # pressure drop is slightly super-linear in velocity; the through-origin
    # quadratic fit dp = a*u + b*u**2 recovers R_s = a at the reference 0.5 mm/s.
    area = float(np.pi) * (0.05 ** 2)  # 100 mm diameter cell
    r_s_true, curvature = 1.6e4, 4.0e5  # Pa*s/m, Pa*s2/m2
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3  # m/s
    dp = r_s_true * u + curvature * u**2
    result = static_airflow_resistance(u, dp, area=area, thickness=0.05)

    u_fit = np.linspace(0.0, 13e-3, 200)
    dp_fit = result.linear_coefficient * u_fit + result.quadratic_coefficient * u_fit**2

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(u_fit * 1e3, dp_fit, color=COLOR_PRIMARY, linewidth=1.8, zorder=2,
            label="Through-origin quadratic fit  dp = a u + b u^2")
    ax.plot(u * 1e3, dp, "o", color=COLOR_SECONDARY, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6, zorder=4,
            label="Measured pressure drop")

    u_ref = result.evaluation_velocity
    ax.axvline(u_ref * 1e3, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(u_ref * 1e3, result.pressure_drop, "D", color=COLOR_TERTIARY,
            markersize=9, zorder=6)
    ax.annotate("evaluation at 0.5 mm/s", xy=(u_ref * 1e3, result.pressure_drop),
                xytext=(2.0, result.pressure_drop + 40), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, (f"Specific airflow resistance R_s = {result.specific_resistance:.0f}"
               f" Pa s/m")),
        (0.90, f"Airflow resistivity sigma = {result.resistivity:.0f} Pa s/m^2"),
        (0.83, (f"Linear term a = {result.linear_coefficient:.0f} Pa s/m"
               f"  (= R_s at u -> 0)")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 9053-1 Static-Method Airflow Resistance", fontweight="bold",
                 pad=12)
    ax.set_xlabel("Linear airflow velocity u [mm/s]")
    ax.set_ylabel("Pressure drop dp [Pa]")
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "airflow_resistance.png")
    plt.close()


def generate_impedance_tube(output_dir: str) -> None:
    """ISO 10534-1 standing-wave-ratio method: alpha and |r| vs level difference."""
    print("Generating impedance_tube.png...")
    from phonometry import (
        standing_wave_absorption,
        standing_wave_ratio_from_level,
        standing_wave_reflection_magnitude,
    )

    # Level difference between pressure maximum and minimum (Eq. 15): a large dL
    # means a strong reflection (little absorption); dL -> 0 is a perfect absorber.
    level_diff = np.linspace(0.5, 40.0, 300)
    swr = np.array([standing_wave_ratio_from_level(dl) for dl in level_diff])
    alpha = np.array([standing_wave_absorption(s) for s in swr])
    r_mag = np.array([standing_wave_reflection_magnitude(s) for s in swr])

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(level_diff, alpha, color=COLOR_PRIMARY, linewidth=2.0, zorder=3,
            label="Absorption coefficient alpha = 1 - |r|^2")
    ax.set_xlabel("Standing-wave level difference L_max - L_min [dB]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 40.0)

    ax_r = ax.twinx()
    ax_r.plot(level_diff, r_mag, color=COLOR_SECONDARY, linewidth=1.8,
              linestyle="--", zorder=2, label="Reflection factor magnitude |r|")
    ax_r.set_ylabel("Reflection factor magnitude |r|")
    ax_r.set_ylim(0.0, 1.02)

    # Mark the didactic anchor: dL = 9.54 dB -> s = 3 -> |r| = 0.5 -> alpha = 0.75.
    dl_anchor = 20.0 * float(np.log10(3.0))
    ax.plot(dl_anchor, 0.75, "D", color=COLOR_TERTIARY, markersize=9, zorder=6)
    # Text sits in the lens that opens between the diverging alpha and |r| curves.
    ax.annotate("s = 3 -> |r| = 0.5 -> alpha = 0.75",
                xy=(dl_anchor, 0.75), xytext=(15.0, 0.44),
                fontsize=10, arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax.set_title("ISO 10534-1 Standing-Wave-Ratio Method", fontweight="bold", pad=12)
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

    from phonometry import (
        AirLayer,
        MembraneLayer,
        MicroperforatedPlateLayer,
        PerforatedPlateLayer,
        PorousAbsorberWarning,
        PorousLayer,
        helmholtz_resonance_frequency,
        layered_absorber,
        membrane_resonance_frequency,
        miki,
    )
    from phonometry.materials.absorbers.porous import Layer

    f = np.logspace(np.log10(50.0), np.log10(5000.0), 500)
    with _warnings.catch_warnings():
        # The 50 Hz decade end sits below the published Miki fit range on
        # purpose (the figure shows the bass behaviour of the resonators).
        _warnings.simplefilter("ignore", PorousAbsorberWarning)
        med = miki(f, 20000.0)
        med_light = miki(f, 10000.0)
        cases: list[tuple[str, list[Layer], str, str]] = [
            ("Porous layer 50 mm (sigma = 20 kPa s/m2)",
             [PorousLayer(0.05, med)], COLOR_PRIMARY, "-"),
            ("Microperforated panel + 48 mm cavity",
             [MicroperforatedPlateLayer(0.5e-3, 0.15e-3, 0.008),
              AirLayer(0.048)], COLOR_SECONDARY, "-"),
            ("Perforated panel 6 mm + porous 25 mm + air",
             [PerforatedPlateLayer(0.006, 0.0025, 0.05),
              PorousLayer(0.025, med), AirLayer(0.019)], COLOR_TERTIARY, "-"),
            ("Membrane 2 kg/m2 + air + porous 38 mm",
             [MembraneLayer(2.0), AirLayer(0.01),
              PorousLayer(0.038, med_light)], "#9467bd", "-"),
        ]
        _fig, ax = plt.subplots(figsize=(10, 6.2))
        for label, layers, color, ls in cases:
            res = layered_absorber(f, layers)
            ax.semilogx(f, res.absorption, ls, color=color, linewidth=2.2,
                        label=label)
    # Closed-form resonance anchors of the two classical designs.
    f_helm = helmholtz_resonance_frequency(
        cavity_depth=0.044, plate_thickness=0.006, hole_radius=0.0025,
        open_area=0.05,
    )
    f_mem = membrane_resonance_frequency(surface_density=2.0, cavity_depth=0.048)
    for f0, color, label in (
        (f_helm, COLOR_TERTIARY, "Helmholtz closed form"),
        (f_mem, "#9467bd", "Membrane closed form"),
    ):
        ax.axvline(f0, color=color, linestyle=":", linewidth=1.1, alpha=0.8)
        ax.text(f0 * 1.04, 0.44, label, rotation=90, va="bottom",
                ha="left", fontsize=8.5, color=color)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(50.0, 5000.0)
    ax.set_title("Multilayer Absorber Prediction (Transfer-Matrix Method)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.03, "Normal incidence, rigid backing, 50 mm total depth",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            color=COLOR_FG)
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
    from phonometry import (
        decoupling_frequency,
        johnson_champoux_allard,
        limp_frame,
    )

    # Allard & Atalla Table 11.2 (printed p. 254): the soft fibrous material
    # behind their Figure 11.2.
    porosity, resistivity, frame_density = 0.98, 25.0e3, 30.0
    f = np.linspace(1.0, 2000.0, 800)
    rigid = johnson_champoux_allard(
        f, resistivity, porosity=porosity, tortuosity=1.02,
        viscous_length=90e-6, thermal_length=180e-6,
    )
    limp = limp_frame(rigid, frame_density, porosity=porosity)
    rho0 = rigid.air_density
    total = frame_density + porosity * rho0
    f_d = decoupling_frequency(
        resistivity, porosity=porosity, frame_density=frame_density
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(f, np.real(rigid.effective_density) / rho0, "--",
            color=COLOR_PRIMARY, linewidth=1.8, label="rigid frame, real part")
    ax.plot(f, np.imag(rigid.effective_density) / rho0, ":",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, imaginary part")
    ax.plot(f, np.real(limp.effective_density) / rho0, "-",
            color=COLOR_SECONDARY, linewidth=2.4, label="limp frame, real part")
    ax.plot(f, np.imag(limp.effective_density) / rho0, "-",
            color=COLOR_TERTIARY, linewidth=2.4,
            label="limp frame, imaginary part")
    ax.axhline(total / rho0, color=COLOR_FG, linestyle="-", linewidth=1.0,
               alpha=0.45)
    ax.axvline(f_d, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    ax.annotate(
        f"apparent total density rho_t/rho0 = {total / rho0:.1f}",
        xy=(1960.0, total / rho0), xytext=(1960.0, total / rho0 - 1.4),
        ha="right", va="top", fontsize=9, color=COLOR_FG,
    )
    ax.annotate(f"decoupling frequency {f_d:.0f} Hz", xy=(f_d, -18.0),
                xytext=(f_d + 60.0, -18.0), ha="left", va="center",
                fontsize=9, color=COLOR_FG)

    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised effective density $\rho_e/\rho_0$")
    ax.set_title("A Limp Frame Carries Its Own Inertia", fontweight="bold",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.015, 0.03,
            "Soft fibrous layer: porosity 0.98, "
            "flow resistivity 25 kPa s/m², frame density 30 kg/m³",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
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
    from phonometry import (
        PoroelasticLayer,
        PorousLayer,
        frame_quarter_wave_resonance,
        johnson_champoux_allard,
        layered_absorber,
    )

    # Allard & Atalla Table 6.1 (printed p. 124): the glass wool "Domisol
    # Coffrage", with the characteristic lengths of their printed p. 123.
    porosity, tortuosity, resistivity = 0.94, 1.06, 40.0e3
    frame_density, shear_modulus = 130.0, 2.2e6 * (1.0 + 0.1j)
    thickness = 0.10
    f = np.linspace(200.0, 1500.0, 1300)
    medium = johnson_champoux_allard(
        f, resistivity, porosity=porosity, tortuosity=tortuosity,
        viscous_length=0.56e-4, thermal_length=1.1e-4,
    )
    rigid = layered_absorber(f, [PorousLayer(thickness, medium)])
    biot = layered_absorber(
        f,
        [PoroelasticLayer(thickness, medium, porosity, tortuosity,
                          frame_density, shear_modulus)],
    )
    f_r = frame_quarter_wave_resonance(
        thickness, shear_modulus=shear_modulus, poisson_ratio=0.0,
        frame_density=frame_density,
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(f, np.real(rigid.normalized_impedance), "--",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, real part")
    ax.plot(f, np.imag(rigid.normalized_impedance), ":",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, imaginary part")
    ax.plot(f, np.real(biot.normalized_impedance), "-",
            color=COLOR_SECONDARY, linewidth=2.4,
            label="Biot poroelastic, real part")
    ax.plot(f, np.imag(biot.normalized_impedance), "-",
            color=COLOR_TERTIARY, linewidth=2.4,
            label="Biot poroelastic, imaginary part")
    ax.axvline(f_r, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    ax.annotate(
        f"frame quarter-wave resonance {f_r:.0f} Hz",
        xy=(f_r, 2.7), xytext=(f_r + 25.0, 2.7),
        ha="left", va="center", fontsize=9, color=COLOR_FG,
    )

    ax.set_xlim(200.0, 1500.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised surface impedance $Z_s/\rho_0 c_0$")
    ax.set_title("Only an Elastic Frame Resonates", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.015, 0.03,
            "Glass wool, 100 mm glued to a rigid wall: porosity 0.94,\n"
            "flow resistivity 40 kPa s/m², frame density 130 kg/m³,\n"
            "shear modulus 2.2 MPa",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
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
    from phonometry import (
        HelmholtzResonator,
        critical_coupling_design,
        slit_helmholtz_absorber,
    )

    lattice_step = 3.0e-2
    period = 5.0e-2
    f0 = 300.0
    base = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    design = critical_coupling_design(
        f0, base, lattice_step=lattice_step, period=period,
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
        res = slit_helmholtz_absorber(
            f, design.resonator, slit_height=height,
            lattice_step=lattice_step, period=period,
        )
        ax.plot(f, res.absorption, ls, color=color, linewidth=2.2, label=label)
    ax.axvline(f0, color=COLOR_FG, linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(f0 * 1.01, 0.05, "design 300 Hz", rotation=90, va="bottom",
            ha="left", fontsize=8.5, color=COLOR_FG)
    panel_depth = lattice_step  # slit depth L = N a with N = 1
    ratio = (343.0 / f0) / panel_depth
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(150.0, 500.0)
    ax.set_title("Perfect Absorption by Critical Coupling (Slow-Sound Panel)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.03,
            f"Normal incidence, rigid backing, panel depth L = lambda/{ratio:.0f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            color=COLOR_FG)
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
    from phonometry import (
        HelmholtzResonator,
        critical_coupling_design,
        materials,
    )

    base = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    design = critical_coupling_design(
        300.0, base, lattice_step=3.0e-2, period=5.0e-2,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_slit_absorber_geometry(
        [design.resonator], ax=ax, slit_height=design.slit_height,
        lattice_step=3.0e-2, period=5.0e-2, language=_LANG,
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
        depths, pitch - fin, ax=ax, periods=2, fin_width=fin, language=_LANG,
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
    from phonometry import HelmholtzResonator, MetadiffuserWell

    rows = _METADIFFUSER_T1_ROWS
    wells = [
        MetadiffuserWell(
            h * 1e-3,
            (HelmholtzResonator(ln * 1e-3, wn * 1e-3, lc * 1e-3, wc * 1e-3),)
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
    from phonometry import (
        metadiffuser_polar_response,
        predict_diffuser_polar_response,
        quadratic_residue_sequence,
    )

    wells, depth, period = _qr_metadiffuser_wells()
    sequence = np.roll(quadratic_residue_sequence(5), -1)
    qrd_depths = sequence * (343.0 / 500.0) / (2 * 5)
    meta = metadiffuser_polar_response(
        2000.0, wells, depth=depth, period=period, periods=6,
    )
    qrd = predict_diffuser_polar_response(
        period, 2000.0, depths=qrd_depths, periods=6,
        include_obliquity=False,
    )
    _fig, ax = plt.subplots(
        figsize=(10, 6.2), subplot_kw={"projection": "polar"},
    )
    meta.plot(
        ax=ax, color=COLOR_SECONDARY, marker="", linewidth=2.2,
        label="Metadiffuser, panel 2 cm", language=_LANG,
    )
    qrd.plot(
        ax=ax, color=COLOR_PRIMARY, marker="", linewidth=1.6,
        linestyle="--", label="QRD, wells up to 27.4 cm", language=_LANG,
    )
    ax.set_title(
        "The 2 cm metadiffuser scatters like the 27 cm QRD (2 kHz)",
        pad=18, fontweight="bold",
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
        wells, ax=ax, depth=depth, period=period, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_geometry.svg")
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
        ax=ax, spacing=0.05, x1=0.15, diameter=0.10, shape="circular",
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
        ax=ax, l1=0.10, s1=0.05, l2=0.20, s2=0.05, thickness=0.05,
        diameter=0.10, shape="circular", language=_LANG,
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
    from phonometry import HelmholtzResonator

    resonator = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
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
    from phonometry import plot_insitu_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_insitu_geometry(ax=ax, language=_LANG)
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
    from phonometry import plot_dynamic_stiffness_rig

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_dynamic_stiffness_rig(ax=ax, language=_LANG)
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
    from phonometry import plot_goniometer_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.6))
    plot_goniometer_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "diffusion_goniometer_geometry.svg")
    plt.close()


def generate_scattering_coefficient(output_dir: str) -> None:
    """ISO 17497-1: scattering coefficient s(f) from a per-band measurement."""
    print("Generating scattering_coefficient.png...")
    from phonometry import scattering_coefficient_spectrum

    # A realistic reverberation-room measurement reduced to two absorption
    # spectra over the 13 one-third-octave bands 250-4000 Hz: the random-
    # incidence absorption alpha_s (stationary sample) and the specular
    # absorption alpha_spec (rotating turntable). A diffuser scatters more with
    # frequency, so alpha_spec climbs above alpha_s and s(f) = (alpha_spec -
    # alpha_s)/(1 - alpha_s) rises smoothly from near 0 towards 0.8.
    freqs = np.array(
        [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000],
        dtype=float,
    )
    alpha_s = np.full_like(freqs, 0.10)
    alpha_spec = 0.11 + 0.75 * (np.log10(freqs / 250.0) / np.log10(4000.0 / 250.0))
    result = scattering_coefficient_spectrum(freqs, alpha_spec, alpha_s)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.semilogx(result.frequencies, result.scattering, color=COLOR_PRIMARY,
                linewidth=1.9, marker="o", markersize=6, markerfacecolor="white",
                markeredgewidth=1.4, zorder=3)
    ax.set_title("Random-incidence scattering coefficient (ISO 17497-1)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Scattering coefficient s")
    ax.set_xlim(freqs.min() * 0.9, freqs.max() * 1.1)
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([250, 500, 1000, 2000, 4000])
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "scattering_coefficient.png")
    plt.close()


def generate_diffusion_polar(output_dir: str) -> None:
    """ISO 17497-2: polar reflected response and its diffusion coefficient d."""
    print("Generating diffusion_polar.png...")
    from phonometry import (
        directional_diffusion,
        predict_diffuser_polar_response,
        qrd_well_depths,
    )

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
    depths = qrd_well_depths(7, 490.0, speed_of_sound=343.0)  # deepest 0.2 m
    predicted = predict_diffuser_polar_response(
        3.6 / 42, 1000.0, depths=depths, periods=6, speed_of_sound=343.0,
    )
    result = directional_diffusion(angles, np.round(predicted.levels, 3))

    _fig, ax = plt.subplots(figsize=(8.0, 7.5),
                           subplot_kw={"projection": "polar"})
    # The theta-* setters live on PolarAxes, not the base Axes type.
    polar: Any = ax
    theta = np.radians(result.angles)
    polar.plot(theta, result.levels, color=COLOR_PRIMARY, linewidth=1.9,
               marker="o", markersize=4, zorder=3)
    # Translucent so the polar grid keeps reading through the lobe.
    polar.fill(theta, result.levels, color=COLOR_PRIMARY, zorder=1,
               alpha=theme_fill_alpha(COLOR_PRIMARY, ax))
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_thetamin(-90)
    polar.set_thetamax(90)
    polar.set_title(
        f"Directional diffusion  d = {result.coefficient:.2f}  (ISO 17497-2)",
        fontweight="bold", pad=20,
    )
    plt.tight_layout()
    save_figure(output_dir, "diffusion_polar.png")
    plt.close()


def generate_diffuser_prediction(output_dir: str) -> None:
    """Predicted diffusion d(f): an N = 7 QRD design versus a flat panel."""
    print("Generating diffuser_prediction.png...")
    from phonometry import (
        predicted_diffusion_spectrum,
        qrd_well_depths,
    )

    # An N = 7 quadratic residue diffuser, design frequency 500 Hz, 10 cm wells,
    # five periods (Cox & D'Antonio far-field Fraunhofer model). The predicted
    # diffusion is compared band by band with the flat reference panel of the
    # same footprint: the QRD spreads the reflected energy far more evenly, so
    # its diffusion coefficient sits well above the near-specular flat panel.
    freqs = np.array([250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                      2000, 2500, 3150, 4000, 5000], float)
    depths = qrd_well_depths(7, 500.0)
    qrd = predicted_diffusion_spectrum(0.10, freqs, depths=depths, periods=5)
    flat = predicted_diffusion_spectrum(
        0.10, freqs, depths=np.zeros_like(depths), periods=5, normalize=False
    )

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, qrd.diffusion, color=COLOR_PRIMARY, linewidth=1.9,
                marker="o", markersize=4, label="N = 7 QRD design")
    ax.semilogx(freqs, flat.diffusion, color=COLOR_SECONDARY, linewidth=1.9,
                marker="s", markersize=4, linestyle="--", label="Flat panel")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Predicted diffusion coefficient d")
    format_frequency_axis(ax, 250.0, 5000.0)
    ax.set_title(
        "Predicted diffusion from design (Cox & D'Antonio Fraunhofer model)",
        fontweight="bold",
    )
    ax.legend(loc="best")
    plt.tight_layout()
    save_figure(output_dir, "diffuser_prediction.png")
    plt.close()


def generate_insitu_absorption(output_dir: str) -> None:
    """ISO 13472-1: in-situ one-third-octave absorption spectrum alpha(f)."""
    print("Generating insitu_absorption.png...")
    from phonometry import geometric_spreading_factor, insitu_absorption_spectrum

    # A synthetic-but-realistic in-situ measurement. The incident impulse hi is
    # a unit spike; the road reflection is hr = Kr * r0 * roll(hi, shift) with
    # Kr the geometrical-spreading factor (2/3 for ds=1.25 m, dm=0.25 m), a
    # mildly frequency-dependent r0 realised by a gentle low-pass (a porous
    # surface reflects less as frequency rises), and the reflected-path delay
    # shift = round(2 dm / c * fs). The library forms the narrow-band
    # alpha = 1 - (1/Kr^2)|Hr/Hi|^2 and reduces it to one-third-octave bands.
    fs, n = 48000.0, 8192
    kr = geometric_spreading_factor()  # (ds - dm)/(ds + dm) = 2/3
    hi = np.zeros(n)
    hi[0] = 1.0
    r0 = 0.85
    taps = scipy_signal.firwin(41, 1200.0, fs=fs)
    taps = taps / taps.sum()
    shift = round(2.0 * 0.25 / 340.0 * fs)  # reflected-path delay 2 dm / c
    hr = kr * r0 * np.roll(scipy_signal.lfilter(taps, 1.0, hi), shift)
    result = insitu_absorption_spectrum(hi, hr, fs)

    freqs = result.frequencies
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, np.nan_to_num(result.absorption), width=0.7,
           color=COLOR_PRIMARY, edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title("In-situ road-surface absorption (ISO 13472-1)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Absorption coefficient alpha")
    ax.set_ylim(0.0, 1.0)
    ax.text(0.04, 0.94, "Kr = 2/3\nalpha = 1 - (1/Kr^2)|Hr/Hi|^2",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "insitu_absorption.png")
    plt.close()


def generate_sound_absorption_measurement(output_dir: str) -> None:
    """ISO 354: reverberation-room alpha_s spectrum from the two decay times."""
    print("Generating sound_absorption_measurement...")
    from phonometry import materials

    # The materials guide's ISO 354 example: one-third-octave reverberation
    # times of a 200 m^3 room, empty (T1) and with a 10.8 m^2 porous absorber
    # sample installed (T2), inverted through Sabine to the alpha_s spectrum.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], float)
    t_empty = np.array([9.0, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8, 7.5, 7.2,
                        6.9, 6.6, 6.2, 5.8, 5.4, 5.0, 4.6, 4.2])
    t_specimen = np.array([8.4, 8.2, 7.7, 7.2, 6.5, 5.7, 4.9, 4.2, 3.6, 3.15,
                           2.85, 2.65, 2.55, 2.5, 2.55, 2.6, 2.7, 2.85])
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
    h12 = (np.exp(1j * k0 * x2) + r_true * np.exp(-1j * k0 * x2)) / \
          (np.exp(1j * k0 * x1) + r_true * np.exp(-1j * k0 * x1))
    result = materials.two_microphone_impedance(
        h12, frequency=f, spacing=spacing, x1=x1, speed_of_sound=c0,
        characteristic_impedance=407.0, diameter=0.10,
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
        f, [materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, eps),
            materials.AirLayer(0.06)],
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
    ax.plot(f, normal.absorption, ls="--", color=COLOR_SECONDARY,
            label=r"Normal incidence $\alpha(0°)$")
    ax.legend(loc="best", fontsize="small")
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "diffuse_field_absorption.svg")
    plt.close()
