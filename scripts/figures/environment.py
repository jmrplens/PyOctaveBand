#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the environmental-noise guides: outdoor propagation and exposure.

Sound on its way from a source to a neighbour: atmospheric absorption, ground
effect, refraction and barriers, the CNOSSOS-EU road and rail emission laws,
and the exposure indices and uncertainties the assessment ends in. Everything
here is embedded by a page under ``environment/``.
"""

from typing import Any

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
    save_figure,
)


def generate_lden_profile(output_dir: str) -> None:
    """A 24 h urban level profile with the Lden period weightings."""
    print("Generating lden_profile.png...")
    from phonometry import lden

    hours = np.arange(24)
    # Typical urban road profile (synthetic hourly LAeq, dB)
    laeq_h = np.array([48, 46, 45, 45, 46, 50, 56, 64, 66, 65, 63, 63,
                       64, 63, 63, 64, 65, 66, 65, 64, 63, 62, 61, 50],
                      dtype=float)

    def _period_leq(idx: "np.ndarray") -> float:
        return float(10 * np.log10(np.mean(10 ** (0.1 * laeq_h[idx]))))

    ld = _period_leq(np.arange(7, 19))    # day 07-19
    le = _period_leq(np.arange(19, 23))   # evening 19-23
    ln_ = _period_leq(np.r_[np.arange(23, 24), np.arange(0, 7)])  # night 23-07
    l_den = lden(ld, le, ln_)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(7, 19, color=theme_fill(COLOR_TERTIARY, ax))
    ax.axvspan(19, 23, color=theme_fill("#e8a838", ax))
    ax.axvspan(23, 24, color=theme_fill(COLOR_PRIMARY, ax))
    ax.axvspan(0, 7, color=theme_fill(COLOR_PRIMARY, ax))
    ax.step(np.r_[hours, 24], np.r_[laeq_h, laeq_h[-1]], where="post",
            color=COLOR_FG, linewidth=1.6, label="Hourly LAeq")
    ax.hlines(ld, 7, 19, color=COLOR_TERTIARY, linestyle="--", linewidth=2,
              label="Lday (+0 dB)")
    ax.hlines(le + 5, 19, 23, color="#e8a838", linestyle="--", linewidth=2,
              label="Levening + 5 dB")
    ax.hlines(ln_ + 10, 23, 24, color=COLOR_PRIMARY, linestyle="--", linewidth=2)
    ax.hlines(ln_ + 10, 0, 7, color=COLOR_PRIMARY, linestyle="--", linewidth=2,
              label="Lnight + 10 dB")
    ax.hlines(l_den, 0, 24, color=COLOR_SECONDARY, linewidth=2.4,
              label=f"Lden = {l_den:.1f} dB")
    ax.set_title("Day-Evening-Night Level Lden (ISO 1996-1)",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, 24)
    ax.set_ylim(42, 80)
    ax.set_xticks([0, 4, 7, 12, 16, 19, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Level [dB]")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    save_figure(output_dir, "lden_profile.png")
    plt.close()


def generate_wind_turbine_tonality(output_dir: str) -> None:
    """IEC 61400-11 wind-turbine tonal audibility: narrowband spectrum + masking."""
    print("Generating wind_turbine_tonality...")
    from phonometry import wind_turbine_tonality
    from phonometry.environment.sources.wind_turbine import _critical_band_edges

    # A narrowband spectrum: a shaped broadband floor with a blade-passing-style
    # tone near 200 Hz, at 2 Hz resolution.
    df = 2.0
    freqs = np.arange(50.0, 400.0 + df, df)
    floor = 42.0 - 6.0 * np.log10(freqs / 100.0)
    tone_bin = int(np.argmin(np.abs(freqs - 200.0)))
    levels = floor.copy()
    levels[tone_bin] += 22.0
    res = wind_turbine_tonality(levels, freqs, tone_frequency=200.0)

    _fig, ax = plt.subplots(figsize=(10, 6))
    band_lo, band_hi = _critical_band_edges(res.tone_frequency)
    ax.axvspan(band_lo, band_hi, color=COLOR_TERTIARY, alpha=0.15,
               label="Critical band")
    ax.plot(freqs, levels, color=COLOR_PRIMARY, linewidth=1.0,
            label="Narrowband spectrum")
    ax.axhline(res.masking_level, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"Masking level = {res.masking_level:.1f} dB")
    ax.plot([res.tone_frequency], [res.tone_level], "o", color=COLOR_SECONDARY,
            markersize=9, label=f"Tone = {res.tone_level:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.set_title("Wind-Turbine Tonal Audibility (IEC 61400-11)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"Tonal audibility ΔLₐ = {res.tonal_audibility:.1f} dB\n"
            f"{'audible' if res.is_audible else 'not audible'}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "wind_turbine_tonality.svg")
    plt.close()


def generate_air_absorption_alpha(output_dir: str) -> None:
    """ISO 9613-1 pure-tone atmospheric attenuation coefficient alpha(f)."""
    print("Generating air_absorption_alpha.png...")
    from phonometry import air_attenuation

    freqs = np.logspace(np.log10(50.0), np.log10(10000.0), 400)
    # Four representative (temperature, relative humidity) conditions spanning
    # the relaxation behaviour: the reference, a dry warm day, a cold humid day
    # and a hot humid day. alpha is returned in dB/m; plot in dB/km (Table 1).
    conditions = [
        (20.0, 50.0, COLOR_PRIMARY),
        (20.0, 10.0, COLOR_SECONDARY),
        (0.0, 70.0, COLOR_TERTIARY),
        (30.0, 80.0, "#ff7f0e"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for temp, rh, color in conditions:
        alpha_km = air_attenuation(freqs, temp, rh) * 1000.0
        ax.loglog(freqs, alpha_km, color=color, linewidth=2.0,
                  label=f"{temp:g} °C, {rh:g} % RH")
    ax.set_title("ISO 9613-1 Atmospheric Absorption α(f)", fontweight="bold",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Attenuation coefficient α [dB/km]")
    ax.set_xlim(50.0, 10000.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    save_figure(output_dir, "air_absorption_alpha.png")
    plt.close()


def generate_atmospheric_attenuation(output_dir: str) -> None:
    """ISO 9613-1 atmospheric attenuation via the AtmosphericAttenuation.plot()."""
    print("Generating atmospheric_attenuation.png...")
    from phonometry import atmospheric_attenuation

    freqs = np.logspace(np.log10(50.0), np.log10(10000.0), 400)
    _, ax = plt.subplots(figsize=(10, 6.2))
    # The result's own .plot() draws alpha in dB/km on a 1k/2k-labelled log
    # frequency axis for the reference 20 degC / 50 % RH atmosphere (ISO 9613-1).
    atmospheric_attenuation(freqs, temperature=20.0, relative_humidity=50.0).plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_attenuation.png")
    plt.close()


def generate_outdoor_attenuation_breakdown(output_dir: str) -> None:
    """ISO 9613-2 per-term octave-band attenuation breakdown, with a barrier."""
    print("Generating outdoor_attenuation_breakdown.png...")
    from phonometry import Barrier, outdoor_propagation_attenuation

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # A point source 200 m away over porous ground (G = 1), screened by a 4 m
    # barrier midway (source and receiver 1,5 m high; the diffraction geometry
    # gives dss = dsr ~ 100 m over the raised edge).
    barrier = Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    att = outdoor_propagation_attenuation(
        200.0, 1.5, 1.5, bands, ground_source=1.0, ground_middle=1.0,
        ground_receiver=1.0, barrier=barrier, temperature=15.0, relative_humidity=70.0,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    # Separate positive and negative cumulative baselines so a negative term
    # (Agr is a net gain at 63 Hz here) stacks below zero instead of being
    # drawn on top of the previous bars; the signed heights sum to a_total.
    pos_bottom = np.zeros(len(bands))
    neg_bottom = np.zeros(len(bands))
    for term, color, label in [
        (att.a_div, COLOR_PRIMARY, "Adiv — divergence"),
        (att.a_atm, COLOR_TERTIARY, "Aatm — atmospheric"),
        (att.a_gr, "#9467bd", "Agr — ground"),
        (att.a_bar, "#ff7f0e", "Abar — barrier"),
    ]:
        bottom = np.where(term >= 0.0, pos_bottom, neg_bottom)
        ax.bar(x, term, bottom=bottom, color=color, edgecolor=COLOR_FG,
               linewidth=0.6, label=label, zorder=3)
        pos_bottom += np.maximum(term, 0.0)
        neg_bottom += np.minimum(term, 0.0)
    ax.plot(x, att.a_total, marker="D", color=COLOR_SECONDARY, linewidth=2.0,
            markersize=6, markerfacecolor="white", markeredgewidth=1.4,
            zorder=5, label="A — total")

    ax.set_title("ISO 9613-2 Attenuation Breakdown (with a 4 m barrier)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Attenuation A [dB]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "outdoor_attenuation_breakdown.png")
    plt.close()


def generate_cnossos_road_emission(output_dir: str) -> None:
    """CNOSSOS-EU road source-line power of an urban arterial traffic mix."""
    print("Generating cnossos_road_emission.png...")
    from phonometry import (
        JunctionType,
        RoadSurface,
        RoadTraffic,
        RoadVehicleCategory,
        road_source_power,
    )

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # A two-lane urban arterial on a thin-layer surface: 1 200 light vehicles,
    # 90 medium heavy and 45 heavy vehicles per hour at 50 km/h, plus 60
    # motorcycles, 60 m before a signalised crossing, on a 3 % upgrade, at the
    # 12 degC yearly mean of a temperate city.
    traffic = [
        RoadTraffic(RoadVehicleCategory.LIGHT, 1200.0, 50.0),
        RoadTraffic(RoadVehicleCategory.MEDIUM_HEAVY, 90.0, 50.0),
        RoadTraffic(RoadVehicleCategory.HEAVY, 45.0, 50.0),
        RoadTraffic(RoadVehicleCategory.MOTORCYCLES, 60.0, 50.0),
    ]
    result = road_source_power(
        traffic, surface=RoadSurface.THIN_LAYER_A, temperature=12.0, gradient=3.0,
        junction_distance=60.0, junction_type=JunctionType.CROSSING,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.bar(x, result.total_line_power, color=COLOR_MUTED, edgecolor=COLOR_FG,
           linewidth=0.6, label="Total source line", zorder=2)
    labels = {
        "1": "Light vehicles (1)",
        "2": "Medium heavy vehicles (2)",
        "3": "Heavy vehicles (3)",
        "4b": "Motorcycles (4b)",
    }
    colors = [COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY, "#9467bd"]
    markers = ["o", "s", "D", "^"]
    for row, category, color, marker in zip(
        result.line_power, result.categories, colors, markers, strict=True
    ):
        ax.plot(x, row, color=color, marker=marker, linewidth=1.8, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3, zorder=5,
                label=labels[category.value])

    ax.set_title(
        "CNOSSOS-EU Road Source Line Power (urban arterial, 50 km/h)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Line power L'W,eq,line [dB re 1 pW/m]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_ylim(45.0, 90.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_emission.png")
    plt.close()


def generate_cnossos_road_speed_law(output_dir: str) -> None:
    """Rolling and propulsion noise against speed, and where they cross over."""
    print("Generating cnossos_road_speed_law.png...")
    from phonometry import (
        CNOSSOS_A_WEIGHTING,
        road_propulsion_noise,
        road_rolling_noise,
        road_vehicle_sound_power,
    )

    weights = np.asarray(CNOSSOS_A_WEIGHTING)

    def a_weighted(bands: np.ndarray) -> float:
        return float(10.0 * np.log10(np.sum(10.0 ** ((bands + weights) / 10.0))))

    speeds = np.linspace(20.0, 130.0, 221)
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    for category, color, name in [
        ("1", COLOR_PRIMARY, "Light vehicles (1)"),
        ("3", COLOR_SECONDARY, "Heavy vehicles (3)"),
    ]:
        rolling = np.array([a_weighted(road_rolling_noise(category, v)) for v in speeds])
        propulsion = np.array(
            [a_weighted(road_propulsion_noise(category, v)) for v in speeds]
        )
        total = np.array([a_weighted(road_vehicle_sound_power(category, v)) for v in speeds])
        ax.plot(speeds, total, color=color, linewidth=2.4, zorder=5, label=f"{name} — total")
        ax.plot(speeds, rolling, color=color, linewidth=1.3, linestyle="--", zorder=4,
                label=f"{name} — rolling")
        ax.plot(speeds, propulsion, color=color, linewidth=1.3, linestyle=":", zorder=4,
                label=f"{name} — propulsion")
        crossing = int(np.argmin(np.abs(rolling - propulsion)))
        ax.plot(speeds[crossing], rolling[crossing], marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6,
                zorder=6)
        ax.annotate(
            f"crossover {speeds[crossing]:.0f} km/h",
            xy=(speeds[crossing], rolling[crossing]),
            xytext=(6, -16), textcoords="offset points", fontsize=9, color=color,
        )

    ax.set_title(
        "CNOSSOS-EU Single-Vehicle Sound Power against Speed (reference conditions)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Speed v [km/h]")
    ax.set_ylabel("A-weighted sound power LW,A [dB(A) re 1 pW]")
    ax.set_xlim(20.0, 130.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_speed_law.png")
    plt.close()


def generate_ground_effect_spherical(output_dir: str) -> None:
    """Spherical-wave ground effect (Weyl-Van der Pol) for four ground types."""
    print("Generating ground_effect_spherical.png...")
    import warnings as _warnings

    from phonometry import ground_effect

    freqs = np.geomspace(50.0, 4000.0, 400)
    # Effective flow resistivities (kPa s/m2 -> Pa s/m2) after Attenborough Ch. 2
    # / Salomons Sec. 3.1: fresh snow, forest floor, grassland and a near-hard
    # surface (asphalt / compacted soil).
    grounds = [
        ("Fresh snow (10 kPa·s·m⁻²)", 10e3, COLOR_TERTIARY),
        ("Forest floor (50 kPa·s·m⁻²)", 50e3, "#9467bd"),
        ("Grassland (200 kPa·s·m⁻²)", 200e3, COLOR_PRIMARY),
        ("Asphalt (20 000 kPa·s·m⁻²)", 20000e3, COLOR_SECONDARY),
    ]
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for label, sigma, color in grounds:
            res = ground_effect(freqs, 1.0, 1.5, 50.0, flow_resistivity=sigma)
            ax.plot(freqs, res.excess_attenuation, color=color, linewidth=1.8,
                    label=label, zorder=3)
    ax.axhline(6.0, color=COLOR_FG, linestyle=":", linewidth=1.0, alpha=0.7,
               label="Hard-ground limit (+6 dB)")
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("Spherical-Wave Ground Effect (Weyl-Van der Pol)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level re free field [dB]")
    ax.set_xlim(50.0, 4000.0)
    ax.set_ylim(-20.0, 8.0)
    format_frequency_axis(ax, 50.0, 4000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ground_effect_spherical.png")
    plt.close()


def generate_atmospheric_refraction(output_dir: str) -> None:
    """Atmospheric refraction: curved rays and the GFPE shadow-zone field."""
    print("Generating atmospheric_refraction.png...")
    import warnings as _warnings

    from phonometry import (
        atmospheric_parabolic_equation,
        atmospheric_ray_paths,
        log_linear_sound_speed_profile,
        shadow_zone_distance,
    )

    # Upward-refracting surface layer (b = -1 m/s) over grassland: rays curve
    # up and leave an acoustic shadow near the ground (Salomons Sec. 4.4/4.6).
    c0, b = 340.0, -1.0
    prof = log_linear_sound_speed_profile(b, ground_speed=c0, max_height=60.0)
    zs = 2.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)

    # (a) Ray fan from a near-ground source.
    rays = atmospheric_ray_paths(prof, source_height=zs,
                                 launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                                 max_range=600.0, n_steps=3000)
    for i in range(rays.heights.shape[0]):
        axes[0].plot(rays.ranges[i], rays.heights[i], color=COLOR_PRIMARY,
                     lw=0.8, alpha=0.7, zorder=2)
    axes[0].plot([0.0], [zs], "o", color=COLOR_SECONDARY, ms=7, zorder=4,
                 label="Source")
    # The linear-gradient shadow distance (grazing ray at hr = zs) as a guide.
    grad = (prof.speed_at(10.0) - c0) / 10.0
    x_sh = shadow_zone_distance(float(grad), zs, zs, ground_speed=c0)
    axes[0].axvline(x_sh, color=COLOR_SECONDARY, ls="--", lw=1.2, zorder=3,
                    label="Shadow-zone boundary")
    axes[0].set_ylabel("Height [m]")
    axes[0].set_ylim(0.0, 40.0)
    axes[0].set_title("Sound rays (upward refraction)", fontweight="bold", pad=8)
    axes[0].grid(which="both", color=COLOR_GRID, ls="--", alpha=0.5, zorder=0)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) GFPE relative-level field over the same atmosphere and ground.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        pe = atmospheric_parabolic_equation(400.0, prof, source_height=zs,
                                            flow_resistivity=200e3,
                                            max_range=600.0, max_height=40.0)
    dl = np.where(np.isfinite(pe.relative_level), pe.relative_level, np.nan)
    img = axes[1].imshow(
        dl, cmap="RdBu_r", vmin=-30.0, vmax=6.0, aspect="auto", origin="lower",
        interpolation="bilinear",
        extent=(float(pe.ranges[0]), float(pe.ranges[-1]),
                float(pe.heights[0]), float(pe.heights[-1])),
        zorder=1,
    )
    axes[1].axvline(x_sh, color=COLOR_FG, ls="--", lw=1.2, alpha=0.8, zorder=2)
    axes[1].plot([0.0], [zs], "o", color="k", ms=6, zorder=3)
    axes[1].set_ylabel("Height [m]")
    axes[1].set_xlabel("Range [m]")
    axes[1].set_ylim(0.0, 40.0)
    axes[1].set_title("GFPE relative sound level", fontweight="bold", pad=8)
    fig.colorbar(img, ax=axes[1], label="Level re free field [dB]", pad=0.01)

    fig.suptitle("Atmospheric Refraction: Ray Bending and the Acoustic Shadow",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_refraction.png")
    plt.close()


def generate_barrier_geometry(output_dir: str) -> None:
    """To-scale section of the barrier of the insertion-loss figure.

    The 4 m screen 50 m from a 1 m source with the receiver 1,5 m high at
    100 m: direct and diffracted paths and the path-length difference that
    drives the Fresnel number. One concept: the geometry the insertion-loss
    methods share.
    """
    print("Generating barrier_geometry...")
    from phonometry import plot_barrier_geometry

    _fig, ax = plt.subplots(figsize=(10, 5.4))
    plot_barrier_geometry(
        ax=ax, source_height=1.0, barrier_distance=50.0,
        barrier_height=4.0, receiver_distance=100.0, receiver_height=1.5,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "barrier_geometry.svg")
    plt.close()


def generate_impulse_prominence(output_dir: str) -> None:
    """NT ACOU 112: predicted prominence and the LAeq adjustment."""
    print("Generating impulse_prominence.png...")
    from phonometry import (
        impulse_adjustment,
        impulse_prominence,
        predicted_prominence,
    )
    from phonometry.environment.assessment.impulsive_sound import (
        ADJUSTMENT_THRESHOLD,
    )

    _fig, (ax_p, ax_k) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: P vs onset rate for three level differences (Formula 1). ---
    orate = np.logspace(1, 4, 200)  # 10 to 10000 dB/s
    # Distinct hues (not COLOR_GRID, which is near-invisible on a light ground).
    for ld, colour in ((5.0, COLOR_TERTIARY), (15.0, COLOR_PRIMARY),
                       (30.0, COLOR_SECONDARY)):
        ax_p.plot(orate, predicted_prominence(orate, np.full_like(orate, ld)),
                  color=colour, label=f"LD = {ld:g} dB")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Onset rate [dB/s]")
    ax_p.set_ylabel("Predicted prominence $P$")
    ax_p.set_title(r"$P = 3\,\log_{10}(\mathrm{OR}) + 2\,\log_{10}(\mathrm{LD})$",
                   fontweight="bold", pad=10)
    ax_p.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_p.set_axisbelow(True)
    ax_p.legend(loc="upper left")

    # --- Right: the adjustment KI(P) with example impulses. ---
    result = impulse_prominence([1200.0, 300.0, 60.0], [32.0, 18.0, 11.0])
    grid = np.linspace(0.0, 16.0, 200)
    ax_k.plot(grid, impulse_adjustment(grid), color=COLOR_PRIMARY,
              label=r"$K_I = 1.8\,(P-5)$")
    ax_k.axvline(ADJUSTMENT_THRESHOLD, color="#7f7f7f", linestyle=":",
                 label=f"threshold $P = {ADJUSTMENT_THRESHOLD:g}$")
    ax_k.scatter(result.per_impulse, impulse_adjustment(result.per_impulse),
                 color="#aec7e8", zorder=3, label="Impulses")
    ax_k.scatter([result.prominence], [result.adjustment], color=COLOR_SECONDARY,
                 marker="*", s=140, zorder=4,
                 label=f"Governing  $K_I$ = {result.adjustment:.1f} dB")
    ax_k.set_xlabel("Predicted prominence $P$")
    ax_k.set_ylabel("Adjustment $K_I$ [dB]")
    ax_k.set_title("Adjustment to $L_{Aeq}$", fontweight="bold", pad=10)
    ax_k.set_ylim(bottom=0.0)
    ax_k.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_k.set_axisbelow(True)
    ax_k.legend(loc="upper left")

    plt.tight_layout()
    save_figure(output_dir, "impulse_prominence.png")
    plt.close()


def generate_tonal_audibility(output_dir: str) -> None:
    """ISO 1996-2: tonal adjustment Kt(ΔLta) with the Annex C.5 examples."""
    print("Generating tonal_audibility...")
    from phonometry import assess_tonal_audibility, tonal_adjustment

    # The four ISO 1996-2:2007 Annex C.5 worked examples: (Lpt, Lpn, fc).
    examples = [(46.7, 37.3, 4000.0), (54.1, 45.2, 430.0),
                (53.6, 45.5, 755.0), (54.6, 45.5, 308.0)]
    assessed = [assess_tonal_audibility(lpt, lpn, fc) for lpt, lpn, fc in examples]
    # A synthetic mid-range tone to exercise the sloped branch.
    mid = assess_tonal_audibility(50.0, 44.0, 500.0)

    grid = np.linspace(0.0, 15.0, 300)
    curve = np.array([tonal_adjustment(d) for d in grid])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(grid, curve, "-", color=COLOR_PRIMARY, linewidth=2.4, zorder=5,
            label=r"$K_t(\Delta L_{ta})$ (Formulae C.4-C.6)")
    for x in (4.0, 10.0):
        ax.axvline(x, color=COLOR_GRID, linestyle=":", alpha=0.8, zorder=1)
    ax.scatter([a.audibility for a in assessed], [a.adjustment for a in assessed],
               color=COLOR_SECONDARY, marker="o", s=70, zorder=6,
               label="Annex C.5 examples")
    ax.scatter([mid.audibility], [mid.adjustment], color=COLOR_TERTIARY,
               marker="*", s=150, zorder=7, label="mid-range tone")

    ax.set_xlabel(r"Tonal audibility $\Delta L_{ta}$ [dB]")
    ax.set_ylabel("Tonal adjustment $K_t$ [dB]")
    ax.set_ylim(-0.3, 6.6)
    ax.set_title("ISO 1996-2 Tonal Adjustment", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    info = [
        "Kt = 0            (dLta < 4)",
        "Kt = dLta - 4  (4 <= dLta <= 10)",
        "Kt = 6            (dLta > 10)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "tonal_audibility.png")
    plt.close()


def generate_impulsive_sound_onsets(output_dir: str) -> None:
    """ISO/PAS 1996-3 LpAF history with the detected impulse onsets."""
    print("Generating impulsive_sound_onsets...")
    import warnings as _warnings

    from phonometry import environment

    # Three hammer strikes over a 55 dB(A) background, 6 s at 48 kHz: the
    # objective chain samples LpAF, detects the onsets and rates the source.
    fs = 48000
    rng = np.random.default_rng(7)
    t = np.arange(int(6.0 * fs)) / fs
    background = rng.standard_normal(t.size)
    background *= 2e-5 * 10 ** (55 / 20) / np.sqrt(np.mean(background**2))
    signal = background.copy()
    for onset_time in (1.0, 2.6, 4.2):
        decay = np.exp(-(t - onset_time) / 0.08) * (t >= onset_time)
        strike = decay * rng.standard_normal(t.size)
        window = (t >= onset_time) & (t < onset_time + 0.1)
        strike *= 2e-5 * 10 ** (95 / 20) / np.sqrt(np.mean(strike[window] ** 2))
        signal += strike
    with _warnings.catch_warnings():
        # The synthetic interval is shorter than the assessment period.
        _warnings.simplefilter("ignore")
        res = environment.impulsive_sound_adjustment(signal, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "impulsive_sound_onsets.svg")
    plt.close()
def generate_atmospheric_sound_speed_profiles(output_dir: str) -> None:
    """Log-linear effective sound-speed profiles: downward vs upward refraction."""
    print("Generating atmospheric_sound_speed_profiles...")
    from phonometry import log_linear_sound_speed_profile

    down = log_linear_sound_speed_profile(+1.0, ground_speed=340.0, max_height=60.0)
    up = log_linear_sound_speed_profile(-1.0, ground_speed=340.0, max_height=60.0)
    _fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.plot(down.sound_speeds, down.heights, color=COLOR_PRIMARY, linewidth=2.0,
            label="Downward refraction (b = +1 m/s)", zorder=3)
    ax.plot(up.sound_speeds, up.heights, color=COLOR_SECONDARY, linewidth=2.0,
            label="Upward refraction (b = -1 m/s)", zorder=3)
    ax.axvline(340.0, color=COLOR_FG, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Effective sound speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_ylim(0.0, 60.0)
    ax.set_title("Effective Sound-Speed Profiles (Salomons Eq. 4.5)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_sound_speed_profiles.svg")
    plt.close()


def generate_atmospheric_ray_fan(output_dir: str) -> None:
    """Downward-refraction ray fan: every ray returns to the ground and bounces."""
    print("Generating atmospheric_ray_fan...")
    from phonometry import atmospheric_ray_paths, log_linear_sound_speed_profile

    profile = log_linear_sound_speed_profile(+1.0, ground_speed=340.0,
                                             max_height=60.0)
    zs = 2.0
    rays = atmospheric_ray_paths(profile, source_height=zs,
                                 launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                                 max_range=600.0, n_steps=600)
    _fig, ax = plt.subplots(figsize=(11.0, 5.6))
    for i in range(rays.heights.shape[0]):
        ax.plot(rays.ranges[i], rays.heights[i], color=COLOR_PRIMARY, lw=0.8,
                alpha=0.7, zorder=2)
    ax.plot([0.0], [zs], "o", color=COLOR_SECONDARY, ms=7, zorder=4,
            label="Source")
    ax.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.7)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Height [m]")
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(0.0, 40.0)
    ax.set_title("Sound Rays under Downward Refraction (b = +1 m/s)",
                 fontweight="bold", pad=12)
    ax.text(0.985, 0.94, "shallow rays are bent back to the ground\nand bounce on down-range",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            color=COLOR_FG)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_ray_fan.svg")
    plt.close()


def generate_atmospheric_pe_range(output_dir: str) -> None:
    """GFPE relative level vs range at 2 m for three refraction conditions."""
    print("Generating atmospheric_pe_range...")
    import warnings as _warnings

    from phonometry import (
        atmospheric_parabolic_equation,
        linear_sound_speed_profile,
        log_linear_sound_speed_profile,
        shadow_zone_distance,
    )

    c0, zs, zr = 340.0, 2.0, 2.0
    cases = [
        (log_linear_sound_speed_profile(+1.0, ground_speed=c0, max_height=60.0),
         COLOR_PRIMARY, "-", "Downward (b = +1 m/s)"),
        (linear_sound_speed_profile(0.0, ground_speed=c0, max_height=60.0),
         COLOR_FG, "--", "Homogeneous (b = 0)"),
        (log_linear_sound_speed_profile(-1.0, ground_speed=c0, max_height=60.0),
         COLOR_SECONDARY, "-", "Upward (b = -1 m/s)"),
    ]
    _fig, ax = plt.subplots(figsize=(11.0, 6.2))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for profile, color, ls, label in cases:
            pe = atmospheric_parabolic_equation(400.0, profile, source_height=zs,
                                                flow_resistivity=200e3,
                                                max_range=600.0, max_height=40.0)
            ax.plot(pe.ranges, pe.level_at_height(zr), color=color, ls=ls,
                    lw=1.6, label=label, zorder=3)
    # The closed-form shadow boundary of the equivalent linear upward gradient
    # (the 10 m mean gradient of the log profile), as in the page-top figure.
    up_prof = cases[2][0]
    grad = (up_prof.speed_at(10.0) - c0) / 10.0
    x_sh = shadow_zone_distance(float(grad), zs, zr, ground_speed=c0)
    ax.axvline(x_sh, color=COLOR_SECONDARY, ls=":", lw=1.2, zorder=2,
               label="Shadow-zone boundary")
    ax.axhline(0.0, color=COLOR_FG, lw=0.8, alpha=0.6)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Level re free field [dB]")
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(-40.0, 10.0)
    ax.set_title("GFPE Relative Level at the Receiver Height (400 Hz, 2 m)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_pe_range.svg")
    plt.close()


def generate_barrier_insertion_loss_methods(output_dir: str) -> None:
    """Wave-theoretic barrier insertion loss: Kurze-Anderson, exact, exact + ground."""
    print("Generating barrier_insertion_loss_methods...")
    import warnings as _warnings

    from phonometry import barrier_insertion_loss

    # A 4 m barrier 50 m from a 1 m source, receiver 1.5 m high at 100 m
    # (the geometry of the ground-barriers guide snippets).
    freqs = np.geomspace(50.0, 5000.0, 240)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        il_ka = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="kurze_anderson")
        il_ex = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="exact")
        il_gr = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="exact",
                                       ground_flow_resistivity=2e5)
    _fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.plot(freqs, il_ka.insertion_loss, color=COLOR_TERTIARY, lw=1.8,
            ls="--", label="Kurze-Anderson (thin screen)", zorder=3)
    ax.plot(freqs, il_ex.insertion_loss, color=COLOR_PRIMARY, lw=1.8,
            label="Exact rigid half-plane", zorder=3)
    ax.plot(freqs, il_gr.insertion_loss, color=COLOR_SECONDARY, lw=1.4,
            alpha=0.9, label="Exact + coherent ground (four paths)", zorder=2)
    ax.axhline(5.0, color=COLOR_FG, ls=":", lw=1.0, alpha=0.7,
               label="Kurze-Anderson grazing limit (5 dB)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Insertion loss [dB]")
    ax.set_xlim(50.0, 5000.0)
    format_frequency_axis(ax, 50.0, 5000.0)
    ax.set_title("Wave-Theoretic Barrier Insertion Loss",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "barrier_insertion_loss_methods.svg")


def generate_rd1367_activity_assessment(output_dir: str) -> None:
    """RD 1367/2007 activity assessment against the Annex III Table B1 limits.

    The worked case of Aviles Lopez & Perera Martin, Manual de acustica
    ambiental y arquitectonica, Ejemplos 3.1 to 3.3: an activity on residential
    land whose day period splits into 2 h shut down, 6 h with a noisy machine
    (LKeq,Ti = 59 dB) and 4 h with the remaining sources (54 dB), giving
    LKeq,d = 57 dB and, over 303 operating days, LK,d = 56 dB. Against the
    55 dB limit of area type a the phase (+5 dB) and daily (+3 dB) criteria are
    met but the annual one is not.
    """
    print("Generating rd1367_activity_assessment.png...")
    from phonometry import NoisePhase, activity_limits, assess_activity

    day = [
        NoisePhase(2.0, 0.0, label="closed"),
        NoisePhase(6.0, 50.0, kt=6.0, kf=3.0),
        NoisePhase(4.0, 48.0, kt=3.0, kf=3.0),
    ]
    evening = [NoisePhase(2.0, 48.0, kt=3.0, kf=3.0), NoisePhase(2.0, 0.0)]
    verdict = assess_activity(
        {"day": day, "evening": evening}, activity_limits("a"), operating_days=303
    )

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the three Article 25.1 b indices per
    # period with their own allowances marked on each group.
    verdict.plot(ax=ax, language=_LANG)
    ax.set_ylim(0, 72)
    save_figure(output_dir, "rd1367_activity_assessment.png")
    plt.close()


def _cnossos_rail_scene() -> tuple[Any, Any]:
    """A four-axle disc-braked coach on a normally maintained ballasted track.

    The vehicle is the 920 mm wheel of Table G-3b at the 50 kN wheel load of
    Table G-2, the track a concrete mono-block sleeper on a medium-stiffness
    rail pad with one joint per 100 m, which is the default Annex II prescribes
    for jointed track.
    """
    from phonometry import (
        BrakeType,
        ContactFilter,
        RailRoughnessClass,
        RailwayTrack,
        RollingStock,
        TrackTransferClass,
        TractionVehicle,
        WheelDiameter,
        aerodynamic_sound_power,
        contact_filter,
        impact_roughness_single,
        rail_roughness,
        track_transfer,
        traction_sound_power,
        wheel_roughness,
        wheel_transfer,
    )

    stock = RollingStock(
        axles=4,
        wheel_roughness=wheel_roughness(BrakeType.NON_TREAD),
        contact_filter=contact_filter(ContactFilter.LOAD_50_DIAMETER_920),
        wheel_transfer=wheel_transfer(WheelDiameter.MM_920),
        traction=traction_sound_power(TractionVehicle.ELECTRIC_MULTIPLE_UNIT),
        aerodynamic=aerodynamic_sound_power(),
    )
    track = RailwayTrack(
        rail_roughness=rail_roughness(RailRoughnessClass.NORMAL),
        track_transfer=track_transfer(TrackTransferClass.MONOBLOCK_MEDIUM),
        impact_roughness=impact_roughness_single(),
    )
    return stock, track


def generate_cnossos_rail_emission(output_dir: str) -> None:
    """CNOSSOS-EU railway source-line power at the two equivalent heights."""
    print("Generating cnossos_rail_emission.png...")
    from phonometry import RailwayVehicle, railway_source_power

    stock, track = _cnossos_rail_scene()
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # 96 coaches per hour at 160 km/h: eight eight-car trains an hour on a
    # conventional main line, seen broadside and slightly from above.
    result = railway_source_power(
        RailwayVehicle(stock, flow_rate=96.0, speed=160.0), track, phi=90.0, psi=10.0,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.bar(x, result.total_line_power, color=COLOR_MUTED, edgecolor=COLOR_FG,
           linewidth=0.6, label="Both source heights", zorder=2)
    for row, color, marker, label in zip(
        result.line_power, (COLOR_PRIMARY, COLOR_SECONDARY), ("o", "s"),
        ("Source A, 0,5 m (rolling, impact, traction)",
         "Source B, 4,0 m (traction, aerodynamic)"), strict=True,
    ):
        ax.plot(x, row, color=color, marker=marker, linewidth=1.8, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3, zorder=5, label=label)

    ax.set_title(
        "CNOSSOS-EU Railway Source Line Power (96 coaches/h at 160 km/h)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Line power L'W,eq,line [dB re 1 pW/m]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_ylim(30.0, 95.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_emission.png")
    plt.close()


def generate_cnossos_rail_roughness_shift(output_dir: str) -> None:
    """The roughness spectrum sliding along the frequency axis with speed."""
    print("Generating cnossos_rail_roughness_shift.png...")
    from phonometry import (
        RAILWAY_THIRD_OCTAVE_BANDS,
        BrakeType,
        ContactFilter,
        RailRoughnessClass,
        contact_filter,
        rail_roughness,
        roughness_to_frequency,
        total_effective_roughness,
        wheel_roughness,
    )

    freqs = np.asarray(RAILWAY_THIRD_OCTAVE_BANDS)
    rail = rail_roughness(RailRoughnessClass.NORMAL)
    wheel = wheel_roughness(BrakeType.NON_TREAD)
    filt = contact_filter(ContactFilter.LOAD_50_DIAMETER_920)

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    for speed, color, marker in (
        (60.0, COLOR_PRIMARY, "o"), (160.0, COLOR_TERTIARY, "s"),
        (300.0, COLOR_SECONDARY, "D"),
    ):
        total = total_effective_roughness(
            roughness_to_frequency(rail[1], rail[0], speed),
            roughness_to_frequency(wheel[1], wheel[0], speed),
            roughness_to_frequency(filt[1], filt[0], speed),
        )
        ax.semilogx(freqs, total, color=color, marker=marker, linewidth=1.8,
                    markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                    label=f"v = {speed:g} km/h", zorder=4)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title(
        "CNOSSOS-EU Total Effective Roughness against Speed "
        "(f = v/λ, λ in the wavelength domain)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("1/3-octave band centre frequency [Hz]")
    ax.set_ylabel("Total effective roughness LR,TOT [dB re 1 μm]")
    ax.set_xlim(50.0, 10000.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_roughness_shift.png")
    plt.close()
