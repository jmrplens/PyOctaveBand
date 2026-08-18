#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the underwater guides: propagation loss, sources and criteria.

The ocean channel and what is heard in it: sound-speed profiles, the Weston
propagation regimes and the sonar equation, seabed reflection, ambient and
shipping noise, and the marine-mammal weighting the impact criteria apply.
Everything here is embedded by a page under ``underwater/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import format_frequency_axis, theme_fill, theme_line

from .i18n import _LANG, _fmt_minus
from .theme import (
    COLOR_FG,
    COLOR_GRID,
    COLOR_MUTED,
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
                markersize=4, label=r"Source level $L_\mathrm{s}$")
    ax.semilogx(freqs, res.radiated_noise_level, "s--", color=COLOR_SECONDARY,
                linewidth=1.6, markersize=3, alpha=0.8, label="Radiated noise level")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB re 1 µPa·m]")
    ax.set_title("Ship Equivalent Monopole Source Level (ISO 17208-2)",
                 pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.semilogx(freqs, res.surface_correction, ":", color=COLOR_TERTIARY,
                  linewidth=2.0, label=r"Surface correction $\Delta L$")
    twin.set_ylabel(r"Surface correction $\Delta L$ [dB]")
    # After twinx() re-initialises the shared x-axis with the default log
    # locator, so the octave-band labelling is not reset to 10^n ticks.
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="lower left", fontsize=9)

    info = [
        r"$L_\mathrm{s} = L_\mathrm{RN} + \Delta L$",
        r"$\Delta L = -10\,\log_{10}[(2u^4 + 14u^2)/(14 + 2u^2 + u^4)]$",
        r"$u = k\,d_\mathrm{s}$,  $d_\mathrm{s} = 0.7\,D$ = 4.2 m",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
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
                   pad=12)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.set_axisbelow(True)
    ax_w.legend(loc="upper right", fontsize=9)

    ax_c.semilogx(strikes, sel_cum, color=COLOR_TERTIARY, linewidth=2.2)
    ax_c.set_xlabel("Number of strikes $N$")
    ax_c.set_ylabel("Cumulative SEL [dB re 1 µPa²·s]")
    ax_c.set_title(
        f"$\\mathrm{{SEL}}_{{\\mathrm{{ss}}}}$ = {res.single_strike_sel:.0f} dB;  "
        r"$\mathrm{SEL}_{\mathrm{cum}} = \mathrm{SEL}_{\mathrm{ss}}"
        r" + 10\,\log_{10}(N)$", fontsize=10)
    ax_c.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_c.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "pile_driving.svg")
    plt.close()


def generate_underwater_propagation_loss(output_dir: str) -> None:
    """Underwater PL vs range: geometrical spreading + volume absorption."""
    print("Generating underwater_propagation_loss...")
    from phonometry import propagation_loss

    ranges = np.linspace(10.0, 20_000.0, 400)
    res = propagation_loss(
        ranges, 10_000.0, law="practical", transition_range=1000.0,
        temperature=10.0, salinity=35.0, depth=100.0, model="francois-garrison",
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.range_m, res.pl, color=COLOR_PRIMARY, linewidth=2.0,
            label="Total propagation loss")
    ax.plot(res.range_m, res.spreading, color="#8c8c8c", linestyle="--", linewidth=1.4,
            label="Geometrical spreading")
    ax.plot(res.range_m, res.absorption, color=COLOR_SECONDARY, linestyle=":", linewidth=1.6,
            label="Volume absorption")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Propagation loss [dB]")
    ax.set_title("Underwater Propagation Loss (Francois–Garrison)",
                 pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.02, 0.05,
            f"$f$ = 10 kHz, $\\alpha$ = {res.absorption_coefficient:.2f} dB/km\n"
            "practical spreading ($R_0$ = 1000 m)",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "underwater_propagation_loss.svg")
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
        (res.spherical, 0.0, r"Spherical, $20\,\log_{10} r$", "#8c8c8c", ":"),
        (res.cylindrical, bounds.spherical_to_cylindrical,
         r"Cylindrical, $10\,\log_{10} r$", COLOR_SECONDARY, "--"),
        (res.mode_stripping, bounds.cylindrical_to_mode_stripping,
         r"Mode stripping, $15\,\log_{10} r$", COLOR_TERTIARY, "-."),
        (res.single_mode, bounds.mode_stripping_to_single_mode,
         "Single mode", "#9467bd", (0, (3, 1, 1, 1))),
    ):
        shown = np.where(res.range_m >= onset / 3.0, curve, np.nan)
        ax.plot(res.range_m, shown, linestyle=style, linewidth=1.3, color=color, label=label)
    ax.plot(res.range_m, res.propagation_loss, color=COLOR_PRIMARY, linewidth=2.6,
            label="Composite propagation loss")
    for boundary, name in (
        (bounds.spherical_to_cylindrical, r"$H/(2\psi_\mathrm{c})$"),
        (bounds.cylindrical_to_mode_stripping, r"$r_{\mathrm{CS}}$"),
        (bounds.mode_stripping_to_single_mode, r"$r_{\mathrm{MS}}$"),
    ):
        # The regime boundary is scaffolding for the curves, so it is drawn
        # back a step -- in shade, since a step back in opacity on the dark
        # page is a step into it.
        ax.axvline(boundary, color=theme_line(COLOR_SECONDARY, ax, quiet=0.6),
                   linestyle="--", linewidth=0.9)
        ax.annotate(name, xy=(boundary, 22.0), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=COLOR_SECONDARY)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Propagation loss [dB re 1 m²]")
    ax.set_title("Weston Shallow-Water Propagation Regimes (Ainslie §9.1.1.2)",
                 pad=12)
    ax.set_ylim(130.0, 18.0)
    ax.set_xlim(float(ranges[0]), float(ranges[-1]))
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05,
            "$f$ = 250 Hz, $H$ = 50 m, medium sand\n"
            f"$\\psi_\\mathrm{{c}}$ = {np.degrees(bounds.critical_angle):.1f}°, "
            f"$\\eta$ = {bounds.reflection_loss_gradient:.2f} Np/rad, "
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
    ax.set_ylabel("Weighting amplitude $W(f)$ [dB]")
    ax.set_title("Marine-Mammal Auditory Weighting (NMFS 2024, v3.0)",
                 pad=12)
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
                 pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "underwater_sound_speed.svg")
    plt.close()


def generate_sonar_equation(output_dir: str) -> None:
    """Passive sonar equation: signal excess vs propagation loss."""
    print("Generating sonar_equation...")
    from phonometry import passive_sonar_equation

    pl = np.linspace(40.0, 120.0, 400)
    res = passive_sonar_equation(140.0, pl, 60.0, directivity_index=15.0,
                                 detection_threshold=8.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.propagation_loss, res.signal_excess, color=COLOR_PRIMARY, linewidth=2.0,
            label="Signal excess")
    ax.axhline(0.0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
               label="Detection limit (SE = 0)")
    ax.axvline(res.figure_of_merit, color="#8c8c8c", linestyle=":", linewidth=1.6,
               label="Figure of merit")
    ax.set_xlabel("Propagation loss [dB]")
    ax.set_ylabel("Signal excess [dB]")
    ax.set_title("Passive Sonar Equation", pad=12)
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
    ax.set_title("Seabed Reflection Loss (Rayleigh)", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            "Water $\\rho$ = 1000, $c$ = 1500\nSand $\\rho$ = 1900, $c$ = 1650",
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
            label="Reflection coefficient magnitude $|R|$ (sand)")
    if res.critical_angle is not None:
        ax.axvline(res.critical_angle, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
                   label=f"Critical angle ({res.critical_angle:.1f}°)")
    ax.set_xlabel("Grazing angle [°]")
    ax.set_ylabel("Reflection coefficient magnitude $|R|$")
    ax.set_title("Seabed Reflection Coefficient (Rayleigh)", pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.02, 0.95,
            "Water $\\rho$ = 1000, $c$ = 1500\nSand $\\rho$ = 1900, $c$ = 1650",
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
    ax.set_title("Ocean Ambient Noise (Wenz)", pad=12)
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
    # The wind component is a part of the total drawn above it: same colour,
    # one shade back. Opacity would take the red component off the dark page.
    ax.plot(res.frequency, res.wind, color=theme_line(color, ax, quiet=0.6),  # type: ignore[attr-defined]
            linewidth=1.0, linestyle="--",
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
    ax.set_title("Ship Traffic Source Level (JOMOPANS-ECHO)", pad=12)
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
    axes[0].set_title("Ray trace (Munk profile)")
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) Parabolic-equation propagation-loss field for the same environment;
    # imshow renders a single raster image (the figure is a raster WebP).
    pe_field = parabolic_equation(50.0, zprof, cprof, source_depth=1000.0,
                                  max_range=100_000.0, range_step=25.0,
                                  n_depth_points=1024)
    pl = pe_field.propagation_loss
    vmax = float(np.percentile(pl[np.isfinite(pl)], 95))
    pl = np.where(np.isfinite(pl), pl, vmax)  # clip the infinite zero-range column
    img = axes[1].imshow(pl, cmap="viridis_r", vmin=vmax - 50.0, vmax=vmax,
                         aspect="auto", origin="upper", interpolation="bilinear",
                         extent=(0.0, 100.0, float(zprof[-1]), 0.0))
    fig.colorbar(img, ax=axes[1], label="Propagation loss [dB]")
    axes[1].set_xlabel("Range [km]")
    axes[1].set_ylabel("Depth [m]")
    axes[1].set_title("Parabolic equation (50 Hz)")

    # (c) Propagation loss vs range: modes and PE agree for a shallow gradient.
    r = np.linspace(100.0, 20_000.0, 400)
    nm = normal_modes(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                      receiver_depth=120.0, ranges_m=r, n_depth_points=800)
    pe = parabolic_equation(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                            max_range=20_000.0, range_step=20.0, n_depth_points=512)
    zi = int(np.argmin(np.abs(pe.depths - 120.0)))
    axes[2].plot(nm.ranges / 1000.0, nm.propagation_loss, color=COLOR_PRIMARY,
                 linewidth=1.0, label="Normal modes")
    axes[2].plot(pe.ranges / 1000.0, pe.propagation_loss[zi], color=COLOR_SECONDARY,
                 linewidth=0.8, alpha=0.7, label="Parabolic equation")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Range [km]")
    axes[2].set_ylabel("Propagation loss [dB]")
    axes[2].set_title("Modes vs PE (50 Hz, $z$ = 120 m)")
    axes[2].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "numerical_propagation.png")
    plt.close()


def generate_seawater_absorption(output_dir: str) -> None:
    """Volume absorption of the three models, and how far the two simpler ones
    depart from the Francois-Garrison reference."""
    print("Generating seawater_absorption...")
    from phonometry import seawater_absorption

    freqs = np.logspace(1.0, 6.0, 500)
    kw = {"temperature": 10.0, "salinity": 35.0, "depth": 100.0}
    alpha = {m: np.asarray(seawater_absorption(freqs, model=m, **kw), dtype=float)
             for m in ("francois-garrison", "ainslie-mccolm", "thorp")}

    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    styles = (("francois-garrison", COLOR_PRIMARY, "-", 2.2),
              ("ainslie-mccolm", COLOR_SECONDARY, "--", 1.8),
              ("thorp", COLOR_TERTIARY, ":", 2.0))
    for name, color, ls, lw in styles:
        ax_a.loglog(freqs, alpha[name], ls, color=color, linewidth=lw, label=name)
    ax_a.set_xlabel("Frequency [Hz]")
    ax_a.set_ylabel(r"Absorption coefficient $\alpha$ [dB/km]")
    ax_a.set_title("Volume Absorption (10 °C, 35 ppt, 100 m)",
                   pad=12)
    ax_a.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="upper left", fontsize=9)
    # The three relaxation regions of the Francois-Garrison expression. The
    # curve climbs six decades across the panel, so a label offset by a fixed
    # factor of its own ordinate leaves the axes near the top: at 500 kHz the
    # absorption is already within a factor of seven of the ceiling, and "pure
    # water" was drawn astride the top spine, which struck through its
    # descenders. Each label is offset by the same factor and then held one
    # tenth of a decade inside the axis, so the three keep a common look and
    # none of them lands on the frame.
    top = float(np.max(alpha["francois-garrison"])) * 10.0
    for f_mark, label in ((300.0, "boric acid"), (30e3, r"$\mathrm{MgSO_4}$"),
                          (500e3, "pure water")):
        a_mark = float(np.interp(f_mark, freqs, alpha["francois-garrison"]))
        ax_a.annotate(label, xy=(f_mark, a_mark),
                      xytext=(f_mark * 0.32, min(a_mark * 7.0, top / 1.26)),
                      fontsize=9,
                      color=COLOR_FG,
                      arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                                  "linewidth": 1.0})

    ref = alpha["francois-garrison"]
    ax_r.axhspan(-10.0, 10.0, color=theme_fill(COLOR_PRIMARY, ax_r), zorder=0)
    ax_r.text(14.0, 10.5, "±10 % of Francois-Garrison", fontsize=9,
              color=COLOR_FG)
    for name, color, ls, lw in styles[1:]:
        ax_r.semilogx(freqs, 100.0 * (alpha[name] / ref - 1.0), ls, color=color,
                      linewidth=lw, label=name)
    ax_r.axhline(0.0, color=COLOR_PRIMARY, linewidth=1.4)
    ax_r.set_xlabel("Frequency [Hz]")
    ax_r.set_ylabel("Departure from Francois-Garrison [%]")
    ax_r.set_title("Where Each Simplification Is Honest",
                   pad=12)
    ax_r.set_ylim(-70.0, 90.0)
    ax_r.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower left", fontsize=9)
    for ax in (ax_a, ax_r):
        format_frequency_axis(ax, 10.0, 1e6)
    plt.tight_layout()
    save_figure(output_dir, "seawater_absorption.svg")
    plt.close(fig)


def generate_sound_speed_models(output_dir: str) -> None:
    """The four sound-speed equations on one profile, and their spread."""
    print("Generating sound_speed_models...")
    from phonometry import sea_water_sound_speed, sound_speed_profile

    depths = np.linspace(0.0, 5000.0, 251)
    temps = 4.0 + 14.0 / (1.0 + (np.maximum(depths - 80.0, 0.0) / 250.0) ** 2)
    models = ("unesco", "del_grosso", "mackenzie", "medwin")
    colors = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#8c8c8c")
    profiles = {m: np.asarray(sound_speed_profile(depths, temps, 35.0, model=m)
                              .sound_speed, dtype=float) for m in models}

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(13.0, 6.2))
    for m, color in zip(models, colors, strict=True):
        ax_c.plot(profiles[m], depths, color=color, linewidth=1.8, label=m)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Sound speed $c$ [m/s]")
    ax_c.set_ylabel("Depth [m]")
    ax_c.set_title("Four Equations, One Profile", pad=12)
    ax_c.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_c.set_axisbelow(True)
    ax_c.legend(loc="lower left", fontsize=9)

    ref = profiles["unesco"]
    for m, color in zip(models[1:], colors[1:], strict=True):
        ax_d.plot(profiles[m] - ref, depths, color=color, linewidth=1.8, label=m)
    ax_d.axvline(0.0, color=COLOR_PRIMARY, linewidth=1.4)
    # Validity envelopes: shade the depths at which a model is out of its domain.
    ax_d.axhspan(1000.0, 5000.0, color=theme_fill(COLOR_MUTED, ax_d), zorder=0)
    ax_d.text(-2.9, 1120.0, "Medwin: beyond ~1000 m", fontsize=9, color=COLOR_FG)
    ax_d.invert_yaxis()
    ax_d.set_xlim(-3.0, 3.0)
    ax_d.set_xlabel("Difference from UNESCO / Chen-Millero [m/s]")
    ax_d.set_ylabel("Depth [m]")
    spread = float(np.max([np.max(np.abs(profiles[m] - ref)) for m in models[1:]]))
    ax_d.set_title(f"Spread on This Profile: up to {spread:.1f} m/s",
                   pad=12)
    ax_d.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_d.set_axisbelow(True)
    ax_d.legend(loc="lower left", fontsize=9)

    check = float(sea_water_sound_speed(25.0, 35.0, 1000.0, model="mackenzie"))
    ax_c.text(0.03, 0.05,
              f"Mackenzie check point: {check:.3f} m/s\n"
              "at 25 °C, 35 ppt, 1000 m (not on this profile)",
              transform=ax_c.transAxes, fontsize=8.5, color=COLOR_FG,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                    "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "sound_speed_models.svg")
    plt.close(fig)


def generate_detection_range(output_dir: str) -> None:
    """One crossing in closed form, several against a modal loss curve."""
    print("Generating detection_range...")
    from phonometry import (
        detection_range,
        detection_range_from_curve,
        normal_modes,
    )

    fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    res = detection_range(82.7, 50e3)
    res.plot(ax=ax_c)
    ax_c.set_title("Closed Form: One Crossing (FOM = 82.7 dB, 50 kHz)",
                   pad=12)

    # A shallow waveguide: the modal loss oscillates, so a figure of merit can
    # be crossed several times and "the" detection range needs a convention.
    ranges = np.linspace(200.0, 6000.0, 1200)
    modes = normal_modes(30.0, [0.0, 100.0], [1500.0, 1500.0],
                         source_depth=25.0, receiver_depth=60.0,
                         ranges_m=ranges, n_depth_points=800)
    pl = np.asarray(modes.propagation_loss, dtype=float)
    fom = 60.0
    ax_m.plot(ranges / 1000.0, pl, color=COLOR_PRIMARY, linewidth=1.2,
              label="Normal-mode PL (30 Hz, 100 m waveguide, 3 modes)")
    ax_m.axhline(fom, color=COLOR_SECONDARY, linewidth=1.8, linestyle="--",
                 label=f"Figure of merit = {fom:.0f} dB")
    crossings = np.where(np.diff(np.sign(pl - fom)) != 0)[0]
    for i, idx in enumerate(crossings):
        ax_m.plot([ranges[idx] / 1000.0], [pl[idx]], "o", color=COLOR_TERTIARY,
                  markersize=6, zorder=5,
                  label="Crossings" if i == 0 else None)
    first = detection_range_from_curve(fom, ranges, pl, crossing="first")
    last = float(ranges[crossings[-1]])
    ax_m.annotate(f"first crossing: {first / 1000.0:.2f} km\n"
                  "(what detection_range_from_curve returns)",
                  xy=(first / 1000.0, fom), xytext=(0.06, 0.20),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                              "linewidth": 1.0})
    ax_m.annotate(f"still detectable at {last / 1000.0:.2f} km",
                  xy=(last / 1000.0, fom), xytext=(0.52, 0.86),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                              "linewidth": 1.0})
    ax_m.invert_yaxis()
    ax_m.set_xlabel("Range [km]")
    ax_m.set_ylabel("Propagation loss [dB]")
    ax_m.set_title(f"Real Waveguide: {len(crossings)} Crossings",
                   pad=12)
    ax_m.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_m.set_axisbelow(True)
    ax_m.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "detection_range.svg")
    plt.close(fig)


def generate_normal_modes(output_dir: str) -> None:
    """Mode shapes, the mode count against frequency, and the modal loss."""
    print("Generating normal_modes...")
    from phonometry import normal_modes

    depth_m, freq = 200.0, 50.0
    ranges = np.linspace(100.0, 20_000.0, 500)
    res = normal_modes(freq, [0.0, depth_m], [1500.0, 1500.0],
                       source_depth=50.0, receiver_depth=100.0,
                       ranges_m=ranges, n_depth_points=800)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))

    ax = axes[0]
    colors = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#8c8c8c")
    k = 2.0 * np.pi * freq / 1500.0
    for m in range(4):
        exact = float(np.sqrt(k**2 - ((m + 1) * np.pi / depth_m) ** 2))
        ax.plot(res.mode_functions[m], res.mode_depths, color=colors[m],
                linewidth=1.8,
                label=f"$m$ = {m + 1},  $k_r$ = {res.wavenumbers[m]:.5f} "
                      f"(exact {exact:.5f})")
    ax.axhline(50.0, color=COLOR_FG, linewidth=1.2, linestyle="--")
    ax.text(0.02, 0.885, "source depth 50 m", transform=ax.transAxes,
            fontsize=9, color=COLOR_FG)
    ax.annotate("mode 4 has a null at the source depth,\nso the source does not excite it",
                xy=(0.0, 50.0), xytext=(0.06, 0.72),
                textcoords="axes fraction", fontsize=8.5, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                            "linewidth": 1.0})
    ax.invert_yaxis()
    ax.set_xlabel(r"Mode function $\Psi_m(z)$")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Mode $m$ Has $m - 1$ Interior Nulls", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    freqs = np.arange(10.0, 201.0, 1.0)
    counts = np.array([normal_modes(f, [0.0, depth_m], [1500.0, 1500.0],
                                    source_depth=50.0, receiver_depth=100.0,
                                    n_depth_points=1200).wavenumbers.size
                       for f in freqs], dtype=float)
    ax.step(freqs, counts, where="post", color=COLOR_PRIMARY, linewidth=1.8,
            label="Propagating modes returned")
    ax.plot(freqs, 2.0 * freqs * depth_m / 1500.0, color=COLOR_SECONDARY,
            linewidth=1.6, linestyle="--", label=r"$M = kD/\pi = 2fD/c$")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Number of propagating modes $M$")
    ax.set_title("One Mode Cuts On at a Time", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[2]
    res.plot(ax=ax)
    ax.set_title("Modal Propagation Loss ($z$ = 100 m)",
                 pad=12)
    plt.tight_layout()
    save_figure(output_dir, "normal_modes.svg")
    plt.close(fig)


def generate_sonar_budget(output_dir: str) -> None:
    """The worked budget as a picture: every term moves the crossing."""
    print("Generating sonar_budget...")
    from phonometry import detection_range, passive_sonar_equation, propagation_loss

    ranges = np.linspace(50.0, 30_000.0, 800)
    cases = (
        ("10 kHz, spherical only", 10e3, "spherical", None, COLOR_PRIMARY, "-"),
        ("10 kHz, practical $R_0$ = 1 km", 10e3, "practical", 1000.0,
         COLOR_TERTIARY, "-"),
        ("20 kHz, practical $R_0$ = 1 km", 20e3, "practical", 1000.0,
         COLOR_SECONDARY, "--"),
    )
    fom_full = float(passive_sonar_equation(
        source_level=140.0, propagation_loss=np.array([50.0]),
        noise_level=60.0, directivity_index=15.0,
        detection_threshold=8.0).figure_of_merit)
    fom_trim = float(passive_sonar_equation(
        source_level=140.0, propagation_loss=np.array([50.0]),
        noise_level=60.0, directivity_index=7.5,
        detection_threshold=8.0).figure_of_merit)

    _fig, ax = plt.subplots(figsize=(11.0, 6.4))
    for label, freq, law, r0, color, ls in cases:
        loss = propagation_loss(ranges, freq, law=law, transition_range=r0,
                                temperature=10.0, salinity=35.0, depth=100.0)
        ax.plot(ranges / 1000.0, loss.pl, ls, color=color, linewidth=1.9,
                label=label)
    for fom, style, lab, dy in (
            (fom_full, "-", f"FOM = {fom_full:.1f} dB (DI 15)", -15),
            (fom_trim, ":", f"FOM = {fom_trim:.1f} dB (DI 7.5)", 8)):
        ax.axhline(fom, color=COLOR_FG, linewidth=1.4, linestyle=style,
                   alpha=0.8, label=lab)
        for k, (label, freq, law, r0, color, _ls) in enumerate(cases):
            r50 = float(detection_range(fom, freq, law=law,
                                        transition_range=r0, temperature=10.0,
                                        salinity=35.0, depth=100.0)
                        .detection_range) / 1000.0
            ax.plot([r50, r50], [fom, 124.0], color=color, linewidth=0.9,
                    linestyle=":", alpha=0.8)
            ax.plot([r50], [fom], "o", color=color, markersize=5, zorder=5)
            step = 13 if dy > 0 else -13
            ax.annotate(f"{r50:.1f} km", xy=(r50, fom),
                        xytext=(0, dy + step * k),
                        textcoords="offset points", fontsize=8.5,
                        color=color, ha="center")
    ax.set_ylim(40.0, 126.0)
    ax.invert_yaxis()
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Propagation loss [dB]")
    ax.set_title("A Passive Sonar Budget, End to End", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    info = [
        "SL = 140 dB re 1 µPa²/Hz at 10 kHz",
        "NL = 60 dB,  DI = 15 dB,  DT = 8 dB",
        (r"$\mathrm{FOM} = \mathrm{SL} - (\mathrm{NL} - \mathrm{DI})"
         r" - \mathrm{DT}$"),
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "sonar_budget.svg")
    plt.close()


def generate_ray_turning_point(output_dir: str) -> None:
    """Snell's turning condition as a picture: a fan of rays, each turning."""
    print("Generating ray_turning_point...")
    from phonometry import ray_trace

    z_top, z_bot, c_top, grad = 0.0, 2000.0, 1490.0, 0.017
    depths = [z_top, z_bot]
    speeds = [c_top, c_top + grad * z_bot]
    angles = [2.0, 4.0, 6.0, 8.0, 10.0]
    z_source = 100.0
    c_source = c_top + grad * z_source

    fig, (ax_c, ax_r) = plt.subplots(
        1, 2, figsize=(13.0, 6.0), gridspec_kw={"width_ratios": [1.0, 3.4]})

    zz = np.linspace(z_top, z_bot, 200)
    ax_c.plot(c_top + grad * zz, zz, color=COLOR_PRIMARY, linewidth=2.0)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("$c(z)$ [m/s]")
    ax_c.set_ylabel("Depth [m]")
    ax_c.set_title("Linear Gradient", pad=12)
    ax_c.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_c.set_axisbelow(True)

    colors = (COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY, "#8c8c8c",
              "#7b5ea7")
    for angle, color in zip(angles, colors, strict=True):
        rays = ray_trace(depths, speeds, source_depth=z_source,
                         launch_angles_deg=[angle], max_range=32e3,
                         n_steps=20000)
        ax_r.plot(rays.ranges[0] / 1000.0, rays.depths[0], color=color,
                  linewidth=1.6, label=f"{angle:.0f}°")
        z_turn = (c_source / np.cos(np.radians(angle)) - c_top) / grad
        deepest = float(np.max(rays.depths[0]))
        first_turn = int(np.argmax(rays.depths[0] >= deepest - 0.5))
        r_turn = float(rays.ranges[0][first_turn]) / 1000.0
        ax_r.plot([r_turn], [z_turn], "x", color=color, markersize=9,
                  markeredgewidth=2.0, zorder=5)
        if abs(angle - 6.0) < 1e-9:
            # The exact circular arc of a linear gradient, over the traced ray.
            radius = c_source / (grad * np.cos(np.radians(angle)))
            theta0 = np.radians(angle)
            t = np.linspace(-theta0, theta0, 200)
            z_arc = z_source + radius * (np.cos(t) - np.cos(theta0))
            r_arc = radius * (np.sin(t) + np.sin(theta0))
            ax_r.plot(r_arc / 1000.0, z_arc, "--", color=COLOR_FG,
                      linewidth=1.2, alpha=0.85,
                      label=r"exact circular arc, $R = c_0/(g\,\cos\theta_0)$")
            ax_r.annotate(
                f"analytic $z_\\mathrm{{t}}$ = {z_turn:.1f} m\n"
                f"traced      = {deepest:.1f} m",
                xy=(r_turn, z_turn), xytext=(26, -62),
                textcoords="offset points", fontsize=8.5, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID},
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                            "linewidth": 1.0})
    ax_r.plot([0.0], [z_source], "o", color=COLOR_FG, markersize=6)
    ax_r.text(0.4, z_source - 46.0, "source, 100 m", fontsize=9, color=COLOR_FG)
    ax_r.set_ylim(1760.0, -70.0)
    ax_r.set_xlabel("Range [km]")
    ax_r.set_ylabel("Depth [m]")
    ax_r.set_title(
        r"Every Ray Turns Where $c(z_\mathrm{t}) = c(z_\mathrm{s})/\cos\theta_0$",
        pad=12)
    ax_r.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower center", fontsize=8.5, ncols=3)
    plt.tight_layout()
    save_figure(output_dir, "ray_turning_point.svg")
    plt.close(fig)


def generate_gaussian_beam_caustic(output_dir: str) -> None:
    """A finite level on the caustic and a graded one in the shadow behind it."""
    print("Generating gaussian_beam_caustic...")
    from phonometry import gaussian_beams, ray_trace

    # Jensen's n^2-linear profile, Eq. (3.77): c(z) = c0/sqrt(1 + 2.4 z/c0), the
    # one profile whose modes are Airy functions in closed form, and the one the
    # book draws its caustic and its shadow zone on (Figs. 3.13-3.17). The
    # source sits 7.5 m off the bottom, so the up-going fan turns inside the
    # column and its envelope is a caustic; the 201-point grid puts no node at
    # 992.5 m, which would be a gradient kink under the source and a spurious
    # horizontal jet in the field.
    c0 = 1550.0
    z_prof = np.linspace(0.0, 1000.0, 201)
    c_prof = c0 / np.sqrt(1.0 + 2.4 * z_prof / c0)
    freq, z_src, r_max = 600.0, 992.5, 2500.0

    beams = gaussian_beams(freq, z_prof, c_prof, source_depth=z_src,
                           max_range=r_max, range_step=12.5,
                           max_angle_deg=45.0, n_depth_points=400)
    pl = np.asarray(beams.propagation_loss, dtype=float)
    ranges = np.asarray(beams.ranges, dtype=float)
    depths = np.asarray(beams.depths, dtype=float)

    fig, ax = plt.subplots(figsize=(11.6, 6.4))

    # contourf, not imshow: a loss field drawn as a raster inside an SVG is a
    # base64 bitmap, which the figure policy keeps out of the vector corpus, and
    # the bands are what the reader actually measures the picture with. Six 8 dB
    # bands from 50 dB, with both ends open: below 50 dB is the near field the
    # sum does not resolve (inside three beam widths of the source the level
    # runs down to 0.7 dB, which is arithmetic and not physics), and above 98 dB
    # is the shadow, which has no bottom -- the wedge no beam reaches at all is
    # 18.4% of these cells and is exactly infinite, so it is folded into the
    # quiet end rather than left as a hole.
    #
    # The band colours are sampled by hand, evenly along the whole ramp,
    # because the spread matplotlib picks for ``extend="both"`` crowds its
    # quiet end: it left the 90 to 98 dB band and the open band above it
    # #481f70 and #440154, a CIEDE2000 distance of 6.8, which is inside the
    # shadow zone and is exactly where the reader is trying to tell one band
    # from the next. Sampled evenly the closest pair is 12.1.
    #
    # Taking the ramp whole then needs no theme switch, which for a sequential
    # map is worth checking rather than assuming: its quiet end is dE00 = 26.4
    # from the dark theme's black axes and 81.4 from the white ones, and its
    # loud end (#fde725) is 29.8 from white, all clear of the 10 the contrast
    # gate asks for. Stopping short of the dark end, which is what a family of
    # curves has to do (see theme.series_colors), would have cost 23 of those
    # points on white to buy 5 on black.
    edges = np.arange(50.0, 99.0, 8.0)
    bands = plt.get_cmap("viridis_r")(np.linspace(0.0, 1.0, edges.size + 1))
    field = np.where(np.isfinite(pl), pl, edges[-1] + 1.0)
    filled = ax.contourf(ranges, depths, field, levels=edges,
                         colors=[tuple(rgba) for rgba in bands], extend="both")

    # The rays the beams are hung on, from the same profile and the same source:
    # every second one of a 15-ray up-going fan, so the fold that makes the
    # caustic is visible as geometry over the level it produces.
    rays = ray_trace(z_prof, c_prof, source_depth=z_src,
                     launch_angles_deg=np.linspace(-45.0, -3.0, 15),
                     max_range=r_max, n_steps=4000)
    # Grey rather than COLOR_FG: these lines cross the whole ramp, and each end
    # of it takes a different ink (black vanishes in the shadow band, white in
    # the bright core). COLOR_MUTED is the one grey that reads on both.
    for i in range(0, rays.ranges.shape[0], 2):
        ax.plot(rays.ranges[i], rays.depths[i], color=COLOR_MUTED,
                linewidth=0.7, zorder=3)

    ax.plot([0.0], [z_src], "o", color=COLOR_FG, markersize=7, zorder=5)
    # "source at 992.5 m" rather than "source, 992.5 m": the clip fingerprint
    # selects an exact-table key by the pieces left when its numbers are taken
    # out, and "source, " is written inside the title of anim_power_two_rooms
    # ("One source, two rooms, one sound power"), so the comma form attaches
    # this figure's translation to that clip and reports it stale.
    ax.annotate("source at 992.5 m", xy=(0.0, z_src), xytext=(16, 20),
                textcoords="offset points", fontsize=9, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})

    # The two things the reader is here for, pinned to where they are: the
    # caustic apex (the shallowest point the 70 dB contour reaches, 162 m at
    # 1588 m) and the graded shadow above it. Both labels are boxed on
    # COLOR_PANEL because they sit over the field, where neither ink reads on
    # its own, and both are placed inside the axes so the saved canvas is the
    # picture rather than the picture plus a margin of captions.
    ax.annotate(
        "caustic: the ray fan folds on itself here,\n"
        "and the beam sum answers 59 dB, not infinity",
        xy=(1600.0, 190.0), xytext=(0.975, 0.74), textcoords="axes fraction",
        fontsize=9, color=COLOR_FG, ha="right", va="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
              "edgecolor": COLOR_GRID},
        arrowprops={"arrowstyle": "->", "color": COLOR_MUTED, "linewidth": 1.2})
    ax.text(0.025, 0.955,
            "shadow zone: no ray reaches it, and the field\n"
            "grades into it rather than stopping at an edge\n"
            "(+88 dB over the 100 m above the caustic)",
            transform=ax.transAxes, fontsize=9, color=COLOR_FG,
            ha="left", va="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.text(0.985, 0.06,
            f"{int(freq)} Hz, $n^2$-linear profile (Jensen Eq. 3.77)\n"
            f"{beams.launch_angles.size} beams over ±45°, "
            f"$W_0$ = {beams.initial_beam_width:.1f} m",
            transform=ax.transAxes, fontsize=8.5, color=COLOR_FG,
            ha="right", va="bottom",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    fig.colorbar(filled, ax=ax, label="Propagation loss [dB]")
    ax.set_xlim(0.0, r_max)
    ax.set_ylim(1000.0, 0.0)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Where Rays Give Infinity and Nothing, Beams Give a Level",
                 pad=12)
    ax.grid(False)
    plt.tight_layout()
    save_figure(output_dir, "gaussian_beam_caustic.svg")
    plt.close(fig)


def generate_pe_paraxial_error(output_dir: str) -> None:
    """Where the small-angle PE disagrees with the modal reference, and why."""
    print("Generating pe_paraxial_error...")
    from phonometry import normal_modes, parabolic_equation

    freq, depth_m, z_rx = 50.0, 100.0, 60.0
    ranges = np.linspace(50.0, 6000.0, 700)
    modes = normal_modes(freq, [0.0, depth_m], [1500.0, 1500.0],
                         source_depth=30.0, receiver_depth=z_rx,
                         ranges_m=ranges, n_depth_points=800)
    field = parabolic_equation(freq, [0.0, depth_m], [1500.0, 1500.0],
                               source_depth=30.0, max_range=6000.0,
                               range_step=5.0, n_depth_points=1024)
    iz = int(np.argmin(np.abs(field.depths - z_rx)))
    nm = np.asarray(modes.propagation_loss, dtype=float)
    pe = np.interp(ranges, field.ranges, field.propagation_loss[iz])

    def _range_average(pl: "np.ndarray", width: int = 61) -> "np.ndarray":
        """Incoherent (energy-domain) running mean over range."""
        kernel = np.ones(width) / width
        return -10.0 * np.log10(np.convolve(10.0 ** (-pl / 10.0), kernel,
                                            mode="same"))

    sm_nm, sm_pe = _range_average(nm), _range_average(pe)
    inner = (ranges > 300.0) & (ranges < 5700.0)
    bias = float(np.mean(sm_pe[inner] - sm_nm[inner]))

    fig, (ax_pl, ax_ang) = plt.subplots(
        1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [2.0, 1.0]})

    # The unsmoothed traces are the interference pattern the range average is
    # taken over: they belong behind their own smoothed curve, which is a
    # matter of shade and width, not of opacity. At a third of the opacity
    # both of these landed within half a level of the dark page.
    ax_pl.plot(ranges / 1000.0, nm,
               color=theme_line(COLOR_PRIMARY, ax_pl, quiet=0.35), linewidth=0.7)
    ax_pl.plot(ranges / 1000.0, pe,
               color=theme_line(COLOR_SECONDARY, ax_pl, quiet=0.35), linewidth=0.7)
    ax_pl.plot(ranges[inner] / 1000.0, sm_nm[inner], color=COLOR_PRIMARY,
               linewidth=2.2, label="Normal modes, range-averaged (reference)")
    ax_pl.plot(ranges[inner] / 1000.0, sm_pe[inner], color=COLOR_SECONDARY,
               linewidth=2.2, label="Parabolic equation, range-averaged")
    ax_pl.set_ylim(30.0, 92.0)
    ax_pl.invert_yaxis()
    ax_pl.set_xlabel("Range [km]")
    ax_pl.set_ylabel("Propagation loss [dB]")
    ax_pl.set_title("50 Hz in 100 m of Water, Receiver at 60 m",
                    pad=12)
    ax_pl.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_pl.set_axisbelow(True)
    ax_pl.legend(loc="lower right", fontsize=9)
    ax_pl.text(0.03, 0.06,
               f"the PE loses {bias:.1f} dB of level at every range,\n"
               "and the offset does not shrink: an ideal\n"
               "waveguide strips nothing away",
               transform=ax_pl.transAxes, fontsize=9, color=COLOR_FG,
               bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                     "edgecolor": COLOR_GRID})

    k = 2.0 * np.pi * freq / 1500.0
    grazing = np.degrees(np.arccos(np.clip(modes.wavenumbers / k, -1.0, 1.0)))
    inside = grazing <= 20.0
    index = np.arange(1, grazing.size + 1)
    ax_ang.bar(index[inside], grazing[inside], color=COLOR_PRIMARY,
               label="within the paraxial band")
    ax_ang.bar(index[~inside], grazing[~inside], color=COLOR_SECONDARY,
               label="steeper than 20°")
    ax_ang.axhline(20.0, color=COLOR_FG, linewidth=1.4, linestyle="--")
    ax_ang.axhspan(0.0, 20.0, color=theme_fill(COLOR_PRIMARY, ax_ang), zorder=0)
    ax_ang.set_xlabel("Mode index $m$")
    ax_ang.set_ylabel(r"Modal grazing angle $\arccos(k_{rm}/k)$ [°]")
    ax_ang.set_title(f"{int(np.sum(~inside))} of {grazing.size} Modes Are "
                     "Outside It", pad=12)
    ax_ang.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_ang.set_axisbelow(True)
    ax_ang.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "pe_paraxial_error.svg")
    plt.close(fig)


def generate_marine_mammal_audiograms(output_dir: str) -> None:
    """The group hearing curves, and the orca curve with its branch trap."""
    print("Generating marine_mammal_audiograms...")
    from phonometry import AUDIOGRAM_GROUPS, group_audiogram, orca_audiogram

    freqs = np.logspace(2.0, np.log10(200e3), 700)
    fig, (ax_g, ax_o) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    palette = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#8c8c8c",
               "#7b5ea7", "#c07a2c", "#2c8c8c")
    for group, color in zip(AUDIOGRAM_GROUPS, palette, strict=True):
        res = group_audiogram(freqs, group)
        style = "--" if res.in_air else "-"
        ax_g.semilogx(res.frequencies, res.threshold, style, color=color,
                      linewidth=1.8,
                      label=f"{group}{' (in air)' if res.in_air else ''}")
        ax_g.plot([res.best_frequency], [res.best_threshold], "o", color=color,
                  markersize=5, zorder=5)
    ax_g.set_ylim(-20.0, 170.0)
    ax_g.set_xlabel("Frequency [Hz]")
    ax_g.set_ylabel("Threshold [dB re 1 µPa; in-air groups re 20 µPa]")
    ax_g.set_title("Southall et al. (2019) Group Audiograms",
                   pad=12)
    ax_g.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_g.set_axisbelow(True)
    ax_g.legend(loc="upper right", fontsize=8.5, ncols=2)
    ax_g.text(0.02, 0.06, "no published fit for LF cetaceans",
              transform=ax_g.transAxes, fontsize=9, color=COLOR_FG,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                    "edgecolor": COLOR_GRID})
    format_frequency_axis(ax_g, 100.0, 200e3)

    orca_f = np.logspace(np.log10(500.0), np.log10(80e3), 600)
    orca = orca_audiogram(orca_f)
    ax_o.semilogx(orca.frequencies, orca.threshold, color=COLOR_PRIMARY,
                  linewidth=2.0, label="orca_audiogram (three branches)")
    for edge in (11.3e3, 46.2e3):
        ax_o.axvline(edge, color=COLOR_MUTED, linewidth=1.0, linestyle=":")
    ax_o.plot([orca.best_frequency], [orca.best_threshold], "o",
              color=COLOR_SECONDARY, markersize=7, zorder=5,
              label=f"minimum {orca.best_threshold:.1f} dB at "
                    f"{orca.best_frequency / 1000.0:.1f} kHz")
    at_50k = float(orca_audiogram(50e3).threshold[0])
    ax_o.plot([50e3], [at_50k], "s", color=COLOR_TERTIARY, markersize=7,
              zorder=5, label=f"third branch at 50 kHz: {at_50k:.1f} dB")
    second_branch = 242.9 * 50.0 ** (-0.7578) + 0.5643 * 50.0**1.076
    ax_o.plot([50e3], [second_branch], "o", markerfacecolor="none",
              markeredgecolor=COLOR_SECONDARY, markersize=9,
              markeredgewidth=1.8, zorder=5,
              label=f"second branch there: {second_branch:.1f} dB")
    ax_o.annotate("the third branch starts at 46.2 kHz;\n"
                  f"reading the second one at 50 kHz loses "
                  f"{at_50k - second_branch:.1f} dB",
                  xy=(50e3, second_branch), xytext=(0.06, 0.24),
                  textcoords="axes fraction", fontsize=8.5, color=COLOR_FG,
                  bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                        "edgecolor": COLOR_GRID},
                  arrowprops={"arrowstyle": "->", "color": COLOR_MUTED,
                              "linewidth": 1.0})
    ax_o.set_xlabel("Frequency [Hz]")
    ax_o.set_ylabel("Threshold [dB re 1 µPa]")
    ax_o.set_title("Killer Whale (Ainslie 2010, Eq. 11.159)",
                   pad=12)
    ax_o.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_o.set_axisbelow(True)
    ax_o.legend(loc="upper center", fontsize=8.5)
    format_frequency_axis(ax_o, 500.0, 80e3)
    plt.tight_layout()
    save_figure(output_dir, "marine_mammal_audiograms.svg")
    plt.close(fig)


def _piling_strike(fs: int = 48_000) -> "np.ndarray":
    """The 200 Hz decaying strike the exposure guide works through."""
    t = np.arange(int(0.2 * fs)) / fs
    return 50.0 * np.exp(-t / 0.06) * np.sin(2.0 * np.pi * 200.0 * t)


def generate_marine_mammal_assessment(output_dir: str) -> None:
    """The band spectrum the chain starts from, and the same campaign judged
    for a low-frequency cetacean and for a porpoise."""
    print("Generating marine_mammal_assessment...")
    from phonometry import (
        peak_sound_pressure_level,
        strike_sel_spectrum,
        weighted_exposure,
    )

    fs = 48_000
    strike = _piling_strike(fs)
    spectrum = strike_sel_spectrum(strike, fs, fraction=3)
    peak = float(peak_sound_pressure_level(strike))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))
    spectrum.plot(ax=axes[0], language=_LANG)
    axes[0].set_title("Step 1: Single-Strike SEL by Band",
                      pad=12)

    for ax, group in zip(axes[1:], ("LF", "VHF"), strict=True):
        res = weighted_exposure(spectrum.frequencies, spectrum.band_sel, group,
                                guidance="nmfs-2024", impulsive=True,
                                n_events=3000, peak_spl=peak)
        res.plot(ax=ax, language=_LANG)
        ax.set_title(f"{group}: cumulative {res.cumulative_sel:.1f} dB, "
                     f"margin {_fmt_minus(res.sel_margin, '+.1f')} dB",
                     pad=12)
    plt.tight_layout()
    save_figure(output_dir, "marine_mammal_assessment.svg")
    plt.close(fig)


def generate_marine_mammal_exposure_functions(output_dir: str) -> None:
    """The exposure function regulators apply, what the 2024 revision changed,
    and the criteria of every in-water group side by side."""
    print("Generating marine_mammal_exposure_functions...")
    from phonometry import auditory_weighting, exposure_criteria

    groups = ("LF", "HF", "VHF", "PW", "OW")
    palette = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#8c8c8c",
               "#7b5ea7")
    freqs = np.logspace(1.0, np.log10(400e3), 800)

    fig = plt.figure(figsize=(13.5, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.32,
                          wspace=0.24)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]

    ax = axes[0]
    for group, color in zip(groups, palette, strict=True):
        res = auditory_weighting(freqs, group, guidance="nmfs-2024")
        exposure = np.asarray(res.exposure_function, dtype=float)
        ax.semilogx(res.frequencies, exposure, color=color, linewidth=1.8,
                    label=group)
        imin = int(np.argmin(exposure))
        ax.plot([res.frequencies[imin]], [exposure[imin]], "o", color=color,
                markersize=5, zorder=5)
    lf = auditory_weighting(freqs, "LF", guidance="nmfs-2024")
    p = lf.parameters
    ax.annotate("each minimum is that group's weighted TTS onset "
                "$T_\\mathrm{w} = K + C$\nLF: below $f_1$ the filter falls at "
                f"$20a$ = {20 * p.a:.0f} dB/decade, above $f_2$ at "
                f"$20b$ = {20 * p.b:.0f} dB/decade",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8.5,
                color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Exposure function $E(f) = K + C - W(f)$ [dB re 1 µPa²·s]")
    ax.set_title("What a Band Level Is Compared Against",
                 pad=12)
    ax.set_ylim(135.0, 265.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=9, ncols=5)
    format_frequency_axis(ax, 10.0, 400e3)

    ax = axes[1]
    for guidance, color, ls in (("nmfs-2024", COLOR_PRIMARY, "-"),
                                ("nmfs-2018", COLOR_SECONDARY, "--")):
        res = auditory_weighting(freqs, "LF", guidance=guidance)
        ax.semilogx(res.frequencies, res.weighting, ls, color=color,
                    linewidth=1.9,
                    label=f"{guidance}  ($b$ = {res.parameters.b:g})")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Weighting $W(f)$ [dB]")
    ax.set_title("What $b$ = 5 Changed for LF Cetaceans",
                 pad=12)
    ax.set_ylim(-60.0, 5.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    format_frequency_axis(ax, 10.0, 400e3)

    ax = axes[2]
    idx = np.arange(len(groups), dtype=float)
    width = 0.2
    series = (
        ("TTS SEL (weighted)", "tts_sel", COLOR_PRIMARY, -1.5),
        ("AUD INJ SEL (weighted)", "injury_sel", COLOR_TERTIARY, -0.5),
        ("TTS peak (flat)", "tts_peak_spl", "#8c8c8c", 0.5),
        ("AUD INJ peak (flat)", "injury_peak_spl", COLOR_SECONDARY, 1.5),
    )
    for label, attr, color, offset in series:
        values = [float(getattr(exposure_criteria(g, guidance="nmfs-2024",
                                                  impulsive=True), attr))
                  for g in groups]
        ax.bar(idx + offset * width, values, width, color=color, label=label)
    ax.set_xticks(idx)
    ax.set_xticklabels(groups)
    ax.set_ylim(120.0, 245.0)
    ax.set_xlabel("Hearing group")
    ax.set_ylabel("Onset criterion [dB re 1 µPa²·s / dB re 1 µPa]")
    ax.set_title("Impulsive Onset Criteria (NMFS 2024)",
                 pad=12)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncols=4)
    save_figure(output_dir, "marine_mammal_exposure_functions.svg")
    plt.close(fig)


def generate_piling_campaign_accumulation(output_dir: str) -> None:
    """How many strikes a group can take before each criterion is crossed."""
    print("Generating piling_campaign_accumulation...")
    from phonometry import strike_sel_spectrum, weighted_exposure

    fs = 48_000
    spectrum = strike_sel_spectrum(_piling_strike(fs), fs, fraction=3)
    counts = np.unique(np.round(np.logspace(0.0, 4.0, 90)).astype(int))
    groups = ("LF", "HF", "VHF", "PW", "OW")
    palette = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#8c8c8c",
               "#7b5ea7")

    _fig, ax = plt.subplots(figsize=(11.5, 6.4))
    for group, color in zip(groups, palette, strict=True):
        curve = np.array([
            weighted_exposure(spectrum.frequencies, spectrum.band_sel, group,
                              guidance="nmfs-2024", impulsive=True,
                              n_events=int(n)).cumulative_sel
            for n in counts], dtype=float)
        ax.semilogx(counts, curve, color=color, linewidth=1.9, label=group)
        crit = weighted_exposure(spectrum.frequencies, spectrum.band_sel, group,
                                 guidance="nmfs-2024", impulsive=True,
                                 n_events=1).criteria
        levels = [(value, style) for value, style in
                  ((crit.tts_sel, ":"), (crit.injury_sel, "--"))
                  if value is not None]
        for level, ls in levels:
            ax.axhline(level, color=color, linewidth=0.9, linestyle=ls,
                       alpha=0.7)
            if curve[0] < level <= curve[-1]:
                n_cross = float(np.interp(level, curve, counts.astype(float)))
                ax.plot([n_cross], [level], "o", color=color, markersize=5,
                        zorder=5)
                ax.annotate(f"{n_cross:.0f}", xy=(n_cross, level),
                            xytext=(0, 6), textcoords="offset points",
                            fontsize=8.5, color=color, ha="center")
    vhf = weighted_exposure(spectrum.frequencies, spectrum.band_sel, "VHF",
                            guidance="nmfs-2024", impulsive=True,
                            n_events=10_000)
    headroom = float(vhf.criteria.tts_sel or 0.0) - float(vhf.cumulative_sel)
    ax.text(0.03, 0.90,
            "the same campaign, judged five ways: a 200 Hz hammer reaches only\n"
            "the LF onset, while the porpoise group stays "
            f"{headroom:.0f} dB "
            "below its own\nTTS criterion even after 10 000 strikes",
            transform=ax.transAxes, fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.set_xlabel("Number of strikes $N$")
    ax.set_ylabel("Weighted cumulative SEL [dB re 1 µPa²·s]")
    ax.set_title("Accumulation Against the Criteria (dotted TTS, dashed AUD INJ)",
                 pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncols=5)
    plt.tight_layout()
    save_figure(output_dir, "piling_campaign_accumulation.svg")
    plt.close()
