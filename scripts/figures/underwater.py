#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the underwater guides: transmission loss, sources and criteria.

The ocean channel and what is heard in it: sound-speed profiles, the Weston
propagation regimes and the sonar equation, seabed reflection, ambient and
shipping noise, and the marine-mammal weighting the impact criteria apply.
Everything here is embedded by a page under ``underwater/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import format_frequency_axis

from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    save_figure,
)


def generate_ship_source_level(output_dir: str) -> None:
    """Ship equivalent monopole source level and the ΔL surface correction."""
    print("Generating ship_source_level...")
    from phonometry import monopole_source_level

    # One-third-octave centres 20 Hz-20 kHz and a plausible broadband ship RNL
    # that rolls off with frequency; draught 6 m -> source depth 4.2 m.
    freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                      400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                      4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000],
                     dtype=float)
    rnl = 175.0 - 12.0 * np.log10(freqs / 20.0)
    res = monopole_source_level(rnl, freqs, draught=6.0)

    _fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.semilogx(freqs, res.source_level, "o-", color=COLOR_PRIMARY, linewidth=2.0,
                markersize=4, label="Source level Ls")
    ax.semilogx(freqs, res.radiated_noise_level, "s--", color=COLOR_SECONDARY,
                linewidth=1.6, markersize=3, alpha=0.8, label="Radiated noise level")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB re 1 µPa·m]")
    ax.set_title("Ship Equivalent Monopole Source Level (ISO 17208-2)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.semilogx(freqs, res.surface_correction, ":", color=COLOR_TERTIARY,
                  linewidth=2.0, label="Surface correction ΔL")
    twin.set_ylabel("Surface correction ΔL [dB]")
    # After twinx() re-initialises the shared x-axis with the default log
    # locator, so the octave-band labelling is not reset to 10^n ticks.
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="lower left", fontsize=9)

    info = [
        "Ls = LRN + ΔL",
        "ΔL = -10 log10[(2u^4+14u^2)/(14+2u^2+u^4)]",
        "u = k d_s,  d_s = 0.7 D = 4.2 m",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "ship_source_level.svg")
    plt.close()


def generate_pile_driving(output_dir: str) -> None:
    """Pile-driving strike waveform, single-strike SEL and cumulative-SEL growth."""
    print("Generating pile_driving...")
    from phonometry import cumulative_sel_identical, pile_strike_metrics

    fs = 48000
    dur = 0.3
    t = np.arange(int(dur * fs)) / fs
    # An impulsive strike: a short rise then an exponentially decaying ring.
    envelope = np.where(t < 0.01, t / 0.01, np.exp(-(t - 0.01) / 0.04))
    pressure = 8000.0 * envelope * np.sin(2.0 * np.pi * 180.0 * t)
    res = pile_strike_metrics(pressure, fs)

    # Cumulative SEL growth over a driving sequence of identical strikes.
    strikes = np.arange(1, 2001)
    sel_cum = np.array([cumulative_sel_identical(res.single_strike_sel, int(n))
                        for n in strikes])

    _fig, (ax_w, ax_c) = plt.subplots(
        2, 1, figsize=(10, 7.2),
        gridspec_kw={"height_ratios": [1.4, 1.0]})
    ax_w.plot(t * 1e3, pressure, color=COLOR_PRIMARY, linewidth=0.8)
    peak_idx = int(np.argmax(np.abs(pressure)))
    ax_w.plot([t[peak_idx] * 1e3], [pressure[peak_idx]], "o", color=COLOR_SECONDARY,
              markersize=8, label=f"Peak = {res.peak_spl:.0f} dB re 1 µPa")
    ax_w.set_xlabel("Time [ms]")
    ax_w.set_ylabel("Pressure [Pa]")
    ax_w.set_title("Percussive Pile-Driving Strike (ISO 18406)",
                   fontweight="bold", pad=12)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.set_axisbelow(True)
    ax_w.legend(loc="upper right", fontsize=9)

    ax_c.semilogx(strikes, sel_cum, color=COLOR_TERTIARY, linewidth=2.2)
    ax_c.set_xlabel("Number of strikes N")
    ax_c.set_ylabel("Cumulative SEL [dB re 1 µPa²·s]")
    ax_c.set_title(
        f"SEL_ss = {res.single_strike_sel:.0f} dB;  "
        f"SEL_cum = SEL_ss + 10 log10(N)", fontsize=10)
    ax_c.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_c.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "pile_driving.svg")
    plt.close()


def generate_underwater_transmission_loss(output_dir: str) -> None:
    """Underwater TL vs range: geometrical spreading + volume absorption."""
    print("Generating underwater_transmission_loss...")
    from phonometry import transmission_loss

    ranges = np.linspace(10.0, 20_000.0, 400)
    res = transmission_loss(
        ranges, 10_000.0, law="practical", transition_range=1000.0,
        temperature=10.0, salinity=35.0, depth=100.0, model="francois-garrison",
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.range_m, res.tl, color=COLOR_PRIMARY, linewidth=2.0,
            label="Total transmission loss")
    ax.plot(res.range_m, res.spreading, color="#8c8c8c", linestyle="--", linewidth=1.4,
            label="Geometrical spreading")
    ax.plot(res.range_m, res.absorption, color=COLOR_SECONDARY, linestyle=":", linewidth=1.6,
            label="Volume absorption")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Underwater Transmission Loss (Francois–Garrison)",
                 fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.02, 0.05,
            f"f = 10 kHz, α = {res.absorption_coefficient:.2f} dB/km\n"
            "practical spreading (R₀ = 1000 m)",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "underwater_transmission_loss.svg")
    plt.close()


def generate_weston_regimes(output_dir: str) -> None:
    """Weston's four shallow-water propagation regimes and their boundaries."""
    print("Generating weston_regimes...")
    from phonometry import weston_propagation_loss

    # A 50 m shallow-water site over medium sand at 250 Hz, the frequency and
    # sediment pair Ainslie uses to illustrate the transition (Figure 9.7).
    ranges = np.logspace(1.0, 5.3, 500)
    res = weston_propagation_loss(ranges, 250.0, 50.0, seabed="sand",
                                  source_depth=10.0, receiver_depth=25.0)
    bounds = res.boundaries
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    # Each law is drawn from a third of a decade before it takes over: an
    # asymptotic form extrapolated far outside its own regime is meaningless
    # (the single-mode formula would sit below free field at 10 m).
    for curve, onset, label, color, style in (
        (res.spherical, 0.0, "Spherical, 20 log10 r", "#8c8c8c", ":"),
        (res.cylindrical, bounds.spherical_to_cylindrical,
         "Cylindrical, 10 log10 r", COLOR_SECONDARY, "--"),
        (res.mode_stripping, bounds.cylindrical_to_mode_stripping,
         "Mode stripping, 15 log10 r", COLOR_TERTIARY, "-."),
        (res.single_mode, bounds.mode_stripping_to_single_mode,
         "Single mode", "#9467bd", (0, (3, 1, 1, 1))),
    ):
        shown = np.where(res.range_m >= onset / 3.0, curve, np.nan)
        ax.plot(res.range_m, shown, linestyle=style, linewidth=1.3, color=color, label=label)
    ax.plot(res.range_m, res.propagation_loss, color=COLOR_PRIMARY, linewidth=2.6,
            label="Composite propagation loss")
    for boundary, name in (
        (bounds.spherical_to_cylindrical, "H/2ψc"),
        (bounds.cylindrical_to_mode_stripping, "r_CS"),
        (bounds.mode_stripping_to_single_mode, "r_MS"),
    ):
        ax.axvline(boundary, color=COLOR_SECONDARY, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.annotate(name, xy=(boundary, 22.0), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=COLOR_SECONDARY)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Propagation loss [dB re 1 m²]")
    ax.set_title("Weston Shallow-Water Propagation Regimes (Ainslie §9.1.1.2)",
                 fontweight="bold", pad=12)
    ax.set_ylim(130.0, 18.0)
    ax.set_xlim(float(ranges[0]), float(ranges[-1]))
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05,
            f"f = 250 Hz, H = 50 m, medium sand\n"
            f"ψc = {np.degrees(bounds.critical_angle):.1f}°, "
            f"η = {bounds.reflection_loss_gradient:.2f} Np/rad, "
            f"{bounds.mode_count:.0f} modes",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "weston_regimes.svg")
    plt.close()


def generate_marine_mammal_weighting(output_dir: str) -> None:
    """NMFS 2024 auditory weighting functions for the five in-water groups."""
    print("Generating marine_mammal_weighting...")
    from phonometry import auditory_weighting, exposure_criteria

    freqs = np.logspace(1.0, 5.4, 700)
    groups = (
        ("LF", "Low-frequency cetaceans", COLOR_PRIMARY, "-"),
        ("HF", "High-frequency cetaceans", COLOR_SECONDARY, "--"),
        ("VHF", "Very high-frequency cetaceans", COLOR_TERTIARY, "-."),
        ("PW", "Phocid pinnipeds (water)", "#9467bd", ":"),
        ("OW", "Otariid pinnipeds (water)", "#8c564b", (0, (3, 1, 1, 1))),
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    for group, label, color, style in groups:
        res = auditory_weighting(freqs, group, guidance="nmfs-2024")
        crit = exposure_criteria(group, guidance="nmfs-2024", impulsive=True)
        ax.semilogx(res.frequencies, res.weighting, color=color, linestyle=style,
                    linewidth=2.0,
                    label=f"{label} (AUD INJ {crit.injury_sel:.0f} dB)")
    ax.axhline(0.0, color="#8c8c8c", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Weighting amplitude W(f) [dB]")
    ax.set_title("Marine-Mammal Auditory Weighting (NMFS 2024, v3.0)",
                 fontweight="bold", pad=12)
    ax.set_ylim(-75.0, 5.0)
    ax.set_xlim(10.0, 250e3)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9, ncol=2, framealpha=0.92)
    plt.tight_layout()
    save_figure(output_dir, "marine_mammal_weighting.svg")
    plt.close()


def generate_underwater_sound_speed(output_dir: str) -> None:
    """Sea-water sound-speed profile (UNESCO): mixed layer, thermocline, deep channel."""
    print("Generating underwater_sound_speed...")
    from phonometry import sound_speed_profile

    depths = np.linspace(0.0, 3000.0, 121)
    # A warm mixed layer (18 °C to 80 m), a thermocline down to 4 °C at 1000 m,
    # then an isothermal deep layer; the pressure term then lifts c with depth.
    temps = 4.0 + 14.0 / (1.0 + (np.maximum(depths - 80.0, 0.0) / 250.0) ** 2)
    prof = sound_speed_profile(depths, temps, 35.0, model="unesco")
    axis_depth = depths[int(np.argmin(prof.sound_speed))]
    _fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(prof.sound_speed, prof.depth, color=COLOR_PRIMARY, linewidth=2.0,
            label="UNESCO sound speed")
    ax.axhline(axis_depth, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
               label="Sound-channel axis")
    ax.set_xlabel("Sound speed [m/s]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Sea-Water Sound-Speed Profile (UNESCO)",
                 fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "underwater_sound_speed.svg")
    plt.close()


def generate_sonar_equation(output_dir: str) -> None:
    """Passive sonar equation: signal excess vs transmission loss."""
    print("Generating sonar_equation...")
    from phonometry import passive_sonar_equation

    tl = np.linspace(40.0, 120.0, 400)
    res = passive_sonar_equation(140.0, tl, 60.0, directivity_index=15.0,
                                 detection_threshold=8.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.transmission_loss, res.signal_excess, color=COLOR_PRIMARY, linewidth=2.0,
            label="Signal excess")
    ax.axhline(0.0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
               label="Detection limit (SE = 0)")
    ax.axvline(res.figure_of_merit, color="#8c8c8c", linestyle=":", linewidth=1.6,
               label="Figure of merit")
    ax.set_xlabel("Transmission loss [dB]")
    ax.set_ylabel("Signal excess [dB]")
    ax.set_title("Passive Sonar Equation", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05,
            f"SL = 140, NL = 60, DI = 15, DT = 8 dB\n"
            f"figure of merit = {res.figure_of_merit:.1f} dB",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "sonar_equation.svg")
    plt.close()


def generate_seabed_reflection(output_dir: str) -> None:
    """Seabed reflection loss vs grazing angle, marking the critical angle."""
    print("Generating seabed_reflection...")
    from phonometry import bottom_reflection_loss

    phi = np.linspace(0.0, 90.0, 361)
    res = bottom_reflection_loss(phi, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.grazing_angle, res.reflection_loss, color=COLOR_PRIMARY, linewidth=2.0,
            label="Bottom loss (sand)")
    if res.critical_angle is not None:
        ax.axvline(res.critical_angle, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
                   label=f"Critical angle ({res.critical_angle:.1f}°)")
    ax.set_xlabel("Grazing angle [°]")
    ax.set_ylabel("Bottom loss [dB]")
    ax.set_title("Seabed Reflection Loss (Rayleigh)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "seabed_reflection.svg")
    plt.close()


def generate_seabed_reflection_coefficient(output_dir: str) -> None:
    """Seabed reflection-coefficient magnitude |R| vs grazing angle."""
    print("Generating seabed_reflection_coefficient...")
    from phonometry import seabed_reflection

    phi = np.linspace(0.0, 90.0, 361)
    res = seabed_reflection(phi, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.grazing_angle, res.magnitude, color=COLOR_PRIMARY, linewidth=2.0,
            label="Reflection coefficient magnitude |R| (sand)")
    if res.critical_angle is not None:
        ax.axvline(res.critical_angle, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
                   label=f"Critical angle ({res.critical_angle:.1f}°)")
    ax.set_xlabel("Grazing angle [°]")
    ax.set_ylabel("Reflection coefficient magnitude |R|")
    ax.set_title("Seabed Reflection Coefficient (Rayleigh)", fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.02, 0.95,
            "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "seabed_reflection_coefficient.svg")
    plt.close()


def generate_ocean_ambient_noise(output_dir: str) -> None:
    """Wenz ambient-noise curves: wind + thermal energy sum vs frequency."""
    print("Generating ocean_ambient_noise...")
    from phonometry.underwater import ocean_ambient_noise

    freqs = np.logspace(2, 5.5, 300)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # Label the wind/thermal components only once to avoid repeated legend rows.
    for i, (u, color) in enumerate(((5.0, COLOR_SECONDARY), (20.0, COLOR_PRIMARY))):
        res = ocean_ambient_noise(freqs, wind_speed_knots=u)
        _plot_ambient_curve(res, u, color, label_components=(i == 0))
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectrum level [dB re 1 µPa²/Hz]")
    ax.set_title("Ocean Ambient Noise (Wenz)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ocean_ambient_noise.svg")
    plt.close()


def _plot_ambient_curve(res: object, wind_speed: float, color: str,
                        label_components: bool = False) -> None:
    ax = plt.gca()
    ax.plot(res.frequency, res.spectrum_level, color=color, linewidth=2.0,  # type: ignore[attr-defined]
            label=f"Total ({wind_speed:.0f} kn)")
    ax.plot(res.frequency, res.wind, color=color, linewidth=1.0, linestyle="--", alpha=0.6,  # type: ignore[attr-defined]
            label="Wind" if label_components else None)
    ax.plot(res.frequency, res.thermal, color="#8c8c8c", linewidth=1.0, linestyle=":", alpha=0.8,  # type: ignore[attr-defined]
            label="Thermal" if label_components else None)


def generate_ship_traffic_noise(output_dir: str) -> None:
    """JOMOPANS-ECHO ship source-level spectra for three vessel classes."""
    print("Generating ship_traffic_noise...")
    from phonometry import ship_source_spectrum

    _fig, ax = plt.subplots(figsize=(10, 6))
    cases = (
        ("containership", 18.0, 300.0, COLOR_PRIMARY),
        ("cruise", 17.1, 250.0, COLOR_SECONDARY),
        ("tug", 3.7, 30.0, "#8c8c8c"),
    )
    for vessel_class, speed, length, color in cases:
        s = ship_source_spectrum(speed, length, vessel_class=vessel_class)
        ax.plot(s.frequency, s.source_psd, color=color, linewidth=2.0,
                label=f"{vessel_class} ({speed:.0f} kn, {length:.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Source spectral density [dB re 1 µPa²/Hz at 1 m]")
    ax.set_title("Ship Traffic Source Level (JOMOPANS-ECHO)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    _sf = np.asarray(s.frequency, dtype=float)
    format_frequency_axis(ax, float(_sf.min()), float(_sf.max()))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ship_traffic_noise.svg")
    plt.close()


def generate_numerical_propagation(output_dir: str) -> None:
    """Three numerical solvers: ray paths, the PE field and modes-vs-PE loss."""
    print("Generating numerical_propagation...")
    from phonometry import normal_modes, parabolic_equation, ray_trace

    # A Munk deep-water sound-speed profile for the ray / PE panels.
    zprof = np.linspace(0.0, 5000.0, 60)
    eta = 2.0 * (zprof - 1300.0) / 1300.0
    cprof = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Ray trace through the Munk profile (vectorised over rays).
    rt = ray_trace(zprof, cprof, source_depth=1000.0,
                   launch_angles_deg=np.linspace(-12.0, 12.0, 13),
                   max_range=100_000.0, n_steps=6000)
    for i in range(rt.ranges.shape[0]):
        axes[0].plot(rt.ranges[i] / 1000.0, rt.depths[i], color=COLOR_PRIMARY,
                     linewidth=0.6, alpha=0.7)
    axes[0].plot([0.0], [1000.0], "o", color=COLOR_SECONDARY, label="Source")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Range [km]")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_title("Ray trace (Munk profile)", fontweight="bold")
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) Parabolic-equation transmission-loss field for the same environment;
    # imshow renders a single raster image (the figure is a raster WebP).
    pe_field = parabolic_equation(50.0, zprof, cprof, source_depth=1000.0,
                                  max_range=100_000.0, range_step=25.0,
                                  n_depth_points=1024)
    tl = pe_field.transmission_loss
    vmax = float(np.percentile(tl[np.isfinite(tl)], 95))
    tl = np.where(np.isfinite(tl), tl, vmax)  # clip the infinite zero-range column
    img = axes[1].imshow(tl, cmap="viridis_r", vmin=vmax - 50.0, vmax=vmax,
                         aspect="auto", origin="upper", interpolation="bilinear",
                         extent=(0.0, 100.0, float(zprof[-1]), 0.0))
    fig.colorbar(img, ax=axes[1], label="Transmission loss [dB]")
    axes[1].set_xlabel("Range [km]")
    axes[1].set_ylabel("Depth [m]")
    axes[1].set_title("Parabolic equation (50 Hz)", fontweight="bold")

    # (c) Transmission loss vs range: modes and PE agree for a shallow gradient.
    r = np.linspace(100.0, 20_000.0, 400)
    nm = normal_modes(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                      receiver_depth=120.0, ranges_m=r, n_depth_points=800)
    pe = parabolic_equation(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                            max_range=20_000.0, range_step=20.0, n_depth_points=512)
    zi = int(np.argmin(np.abs(pe.depths - 120.0)))
    axes[2].plot(nm.ranges / 1000.0, nm.transmission_loss, color=COLOR_PRIMARY,
                 linewidth=1.0, label="Normal modes")
    axes[2].plot(pe.ranges / 1000.0, pe.transmission_loss[zi], color=COLOR_SECONDARY,
                 linewidth=0.8, alpha=0.7, label="Parabolic equation")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Range [km]")
    axes[2].set_ylabel("Transmission loss [dB]")
    axes[2].set_title("Modes vs PE (50 Hz, z = 120 m)", fontweight="bold")
    axes[2].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "numerical_propagation.png")
    plt.close()
