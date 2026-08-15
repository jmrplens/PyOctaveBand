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
            color=COLOR_FG, linewidth=1.6, label="Hourly $L_\\mathrm{Aeq}$")
    ax.hlines(ld, 7, 19, color=COLOR_TERTIARY, linestyle="--", linewidth=2,
              label=r"$L_{\mathrm{day}}$ (+0 dB)")
    ax.hlines(le + 5, 19, 23, color="#e8a838", linestyle="--", linewidth=2,
              label=r"$L_{\mathrm{evening}}$ + 5 dB")
    ax.hlines(ln_ + 10, 23, 24, color=COLOR_PRIMARY, linestyle="--", linewidth=2)
    ax.hlines(ln_ + 10, 0, 7, color=COLOR_PRIMARY, linestyle="--", linewidth=2,
              label=r"$L_{\mathrm{night}}$ + 10 dB")
    ax.hlines(l_den, 0, 24, color=COLOR_SECONDARY, linewidth=2.4,
              label=rf"$L_{{\mathrm{{den}}}}$ = {l_den:.1f} dB")
    ax.set_title(r"Day-Evening-Night Level $L_{\mathrm{den}}$ (ISO 1996-1)",
                 pad=12)
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
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"Tonal audibility $\\Delta L_\\mathrm{{a}}$ = "
            f"{_fmt_minus(res.tonal_audibility, '.1f')} dB\n"
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
    ax.set_title(r"ISO 9613-1 Atmospheric Absorption $\alpha(f)$",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Attenuation coefficient $\alpha$ [dB/km]")
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
        (att.a_div, COLOR_PRIMARY, r"$A_{\mathrm{div}}$ — divergence"),
        (att.a_atm, COLOR_TERTIARY, r"$A_{\mathrm{atm}}$ — atmospheric"),
        (att.a_gr, "#9467bd", r"$A_{\mathrm{gr}}$ — ground"),
        (att.a_bar, "#ff7f0e", r"$A_{\mathrm{bar}}$ — barrier"),
    ]:
        bottom = np.where(term >= 0.0, pos_bottom, neg_bottom)
        ax.bar(x, term, bottom=bottom, color=color, edgecolor=COLOR_FG,
               linewidth=0.6, label=label, zorder=3)
        pos_bottom += np.maximum(term, 0.0)
        neg_bottom += np.minimum(term, 0.0)
    ax.plot(x, att.a_total, marker="D", color=COLOR_SECONDARY, linewidth=2.0,
            markersize=6, markerfacecolor="white", markeredgewidth=1.4,
            zorder=5, label="$A$ — total")

    ax.set_title("ISO 9613-2 Attenuation Breakdown (with a 4 m barrier)",
                 pad=12)
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Attenuation $A$ [dB]")
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
        pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel(r"Line power $L^{\prime}_{W,\mathrm{eq,line}}$ [dB re 1 pW/m]")
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
        pad=12,
    )
    ax.set_xlabel("Speed $v$ [km/h]")
    ax.set_ylabel(r"A-weighted sound power $L_{W\!,A}$ [dB(A) re 1 pW]")
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
                 pad=12)
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
    axes[0].set_title("Sound rays (upward refraction)", pad=8)
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
    axes[1].set_title("GFPE relative sound level", pad=8)
    fig.colorbar(img, ax=axes[1], label="Level re free field [dB]", pad=0.01)

    fig.suptitle("Atmospheric Refraction: Ray Bending and the Acoustic Shadow",
                 fontsize=13)
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
                  color=colour, label=rf"$\mathrm{{LD}}$ = {ld:g} dB")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Onset rate [dB/s]")
    ax_p.set_ylabel("Predicted prominence $P$")
    ax_p.set_title(r"$P = 3\,\log_{10}(\mathrm{OR}) + 2\,\log_{10}(\mathrm{LD})$",
                   pad=10)
    ax_p.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_p.set_axisbelow(True)
    ax_p.legend(loc="upper left")

    # --- Right: the adjustment KI(P) with example impulses. ---
    result = impulse_prominence([1200.0, 300.0, 60.0], [32.0, 18.0, 11.0])
    grid = np.linspace(0.0, 16.0, 200)
    ax_k.plot(grid, impulse_adjustment(grid), color=COLOR_PRIMARY,
              label=r"$K_\mathrm{I} = 1.8\,(P-5)$")
    ax_k.axvline(ADJUSTMENT_THRESHOLD, color="#7f7f7f", linestyle=":",
                 label=f"threshold $P = {ADJUSTMENT_THRESHOLD:g}$")
    ax_k.scatter(result.per_impulse, impulse_adjustment(result.per_impulse),
                 color="#aec7e8", zorder=3, label="Impulses")
    ax_k.scatter([result.prominence], [result.adjustment], color=COLOR_SECONDARY,
                 marker="*", s=140, zorder=4,
                 label=f"Governing  $K_\\mathrm{{I}}$ = {result.adjustment:.1f} dB")
    ax_k.set_xlabel("Predicted prominence $P$")
    ax_k.set_ylabel("Adjustment $K_\\mathrm{I}$ [dB]")
    ax_k.set_title("Adjustment to $L_\\mathrm{Aeq}$", pad=10)
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
            label=r"$K_\mathrm{t}(\Delta L_\mathrm{ta})$ (Formulae C.4-C.6)")
    for x in (4.0, 10.0):
        ax.axvline(x, color=COLOR_GRID, linestyle=":", alpha=0.8, zorder=1)
    ax.scatter([a.audibility for a in assessed], [a.adjustment for a in assessed],
               color=COLOR_SECONDARY, marker="o", s=70, zorder=6,
               label="Annex C.5 examples")
    ax.scatter([mid.audibility], [mid.adjustment], color=COLOR_TERTIARY,
               marker="*", s=150, zorder=7, label="mid-range tone")

    ax.set_xlabel(r"Tonal audibility $\Delta L_\mathrm{ta}$ [dB]")
    ax.set_ylabel("Tonal adjustment $K_\\mathrm{t}$ [dB]")
    ax.set_ylim(-0.3, 6.6)
    ax.set_title("ISO 1996-2 Tonal Adjustment", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    info = [
        r"$K_\mathrm{t} = 0$  ($\Delta L_\mathrm{ta} < 4$)",
        r"$K_\mathrm{t} = \Delta L_\mathrm{ta} - 4$  ($4 \leq \Delta L_\mathrm{ta} \leq 10$)",
        r"$K_\mathrm{t} = 6$  ($\Delta L_\mathrm{ta} > 10$)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
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
            label="Downward refraction ($b$ = +1 m/s)", zorder=3)
    ax.plot(up.sound_speeds, up.heights, color=COLOR_SECONDARY, linewidth=2.0,
            label="Upward refraction ($b$ = −1 m/s)", zorder=3)
    ax.axvline(340.0, color=COLOR_FG, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Effective sound speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_ylim(0.0, 60.0)
    ax.set_title("Effective Sound-Speed Profiles (Salomons Eq. 4.5)",
                 pad=12)
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
    ax.set_title("Sound Rays under Downward Refraction ($b$ = +1 m/s)",
                 pad=12)
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
         COLOR_PRIMARY, "-", "Downward ($b$ = +1 m/s)"),
        (linear_sound_speed_profile(0.0, ground_speed=c0, max_height=60.0),
         COLOR_FG, "--", "Homogeneous ($b$ = 0)"),
        (log_linear_sound_speed_profile(-1.0, ground_speed=c0, max_height=60.0),
         COLOR_SECONDARY, "-", "Upward ($b$ = −1 m/s)"),
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
                 pad=12)
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
                 pad=12)
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
        pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel(r"Line power $L^{\prime}_{W,\mathrm{eq,line}}$ [dB re 1 pW/m]")
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
                    label=f"$v$ = {speed:g} km/h", zorder=4)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title(
        "CNOSSOS-EU Total Effective Roughness against Speed "
        r"($f = v/\lambda$, $\lambda$ in the wavelength domain)",
        pad=12,
    )
    ax.set_xlabel("1/3-octave band centre frequency [Hz]")
    ax.set_ylabel(r"Total effective roughness $L_{R,\mathrm{TOT}}$ [dB re 1 μm]")
    ax.set_xlim(50.0, 10000.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_roughness_shift.png")
    plt.close()


def generate_outdoor_level_cascade(output_dir: str) -> None:
    """Where the source power goes, band by band (ISO 9613-2 book-keeping)."""
    print("Generating outdoor_level_cascade.svg...")
    from phonometry import environment

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    barrier = environment.Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    att = environment.outdoor_propagation_attenuation(
        200.0, 1.5, 1.5, bands, 1.0, 1.0, 1.0, barrier=barrier,
        temperature=15.0, relative_humidity=70.0,
    )
    lw = np.full(bands.size, 95.0)

    x = np.arange(bands.size)
    level = lw.copy()
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.plot(x, lw, "s--", color=COLOR_MUTED, linewidth=1.4, markersize=6,
            label="$L_W$ = 95 dB (source power)", zorder=4)
    terms = (
        (att.a_div, r"$-A_{\mathrm{div}}$ (divergence)", COLOR_PRIMARY),
        (att.a_atm, r"$-A_{\mathrm{atm}}$ (air)", COLOR_TERTIARY),
        (att.a_gr, r"$-A_{\mathrm{gr}}$ (ground)", "#9467bd"),
        (att.a_bar, r"$-A_{\mathrm{bar}}$ (barrier)", COLOR_SECONDARY),
    )
    for term, label, color in terms:
        ax.bar(x, -np.asarray(term), bottom=level, color=color, alpha=0.85,
               width=0.62, label=label, zorder=2)
        level = level - np.asarray(term)
    ax.plot(x, level, "D-", color=COLOR_FG, linewidth=2.0, markersize=6,
            label=r"$L_{fT}(\mathrm{{DW}})$ at the receiver", zorder=5)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title("Where 95 dB of Source Power Goes (ISO 9613-2, 200 m)",
                 pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "outdoor_level_cascade.svg")
    plt.close()


def generate_iso9613_screening_anatomy(output_dir: str) -> None:
    """The ISO 9613-2 screening term: the caps, Kmet and the spent ground effect."""
    print("Generating iso9613_screening_anatomy.svg...")
    from phonometry import environment

    heights = np.linspace(1.6, 12.0, 90)
    single, double = [], []
    for h in heights:
        leg = float(np.hypot(100.0, h - 1.5))
        single.append(float(environment.barrier_attenuation(
            environment.Barrier(source_to_edge=leg, edge_to_receiver=leg),
            200.0, [500.0])[0]))
        double.append(float(environment.barrier_attenuation(
            environment.Barrier(source_to_edge=leg, edge_to_receiver=leg,
                                edge_separation=2.0), 200.0, [500.0])[0]))

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    barrier = environment.Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    dz = np.asarray(environment.barrier_attenuation(barrier, 200.0, bands))
    att = environment.outdoor_propagation_attenuation(
        200.0, 1.5, 1.5, bands, 1.0, 1.0, 1.0, barrier=barrier,
        temperature=15.0, relative_humidity=70.0,
    )

    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    left.plot(heights, single, color=COLOR_PRIMARY, linewidth=2.0,
              label="Single edge", zorder=4)
    left.plot(heights, double, color=COLOR_TERTIARY, linewidth=2.0,
              label="Double edge, $e$ = 2 m", zorder=4)
    left.axhline(20.0, color=COLOR_SECONDARY, linestyle=":", linewidth=1.6,
                 label="20 dB cap (single)", zorder=3)
    left.axhline(25.0, color="#9467bd", linestyle=":", linewidth=1.6,
                 label="25 dB cap (double)", zorder=3)
    left.set_title("$D_z$ at 500 Hz against barrier height ($d$ = 200 m)",
                   pad=10)
    left.set_xlabel("Barrier height [m]")
    left.set_ylabel("Diffraction insertion loss $D_z$ [dB]")
    left.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="lower right", fontsize=9)

    x = np.arange(bands.size)
    right.plot(x, dz, "o-", color=COLOR_PRIMARY, linewidth=2.0, markersize=6,
               label="$D_z$ (Eq. (14))", zorder=4)
    right.plot(x, att.a_bar, "D-", color=COLOR_SECONDARY, linewidth=2.0,
               markersize=6,
               label=r"$A_{\mathrm{bar}} = \max(D_z - A_{\mathrm{gr}}, 0)$ (Eq. (12))", zorder=4)
    right.fill_between(x, att.a_bar, dz, color=COLOR_TERTIARY, alpha=0.20,
                       label=r"$A_{\mathrm{gr}}$, spent on the screened path", zorder=2)
    right.axhline(20.0, color=COLOR_MUTED, linestyle=":", linewidth=1.2, zorder=1)
    right.set_title("The ground effect is spent, not kept",
                    pad=10)
    right.set_xticks(x)
    right.set_xticklabels([f"{b:g}" for b in bands])
    right.set_xlabel(LABEL_FREQ_HZ)
    right.set_ylabel("Attenuation [dB]")
    right.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    right.set_axisbelow(True)
    right.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "iso9613_screening_anatomy.svg")
    plt.close()


def generate_ground_reflection_coefficient(output_dir: str) -> None:
    """The ground wave: Q departing from Rp as the geometry grazes."""
    print("Generating ground_reflection_coefficient.svg...")
    import warnings as _warnings

    from phonometry import ground_effect

    heights = np.geomspace(3.0, 0.02, 120)
    c0, freq, dist = 343.0, 500.0, 50.0
    k = 2.0 * np.pi * freq / c0
    rp_mag, q_mag, dl_q, dl_p = [], [], [], []
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for h in heights:
            res = ground_effect([freq], float(h), float(h), dist,
                                flow_resistivity=2e5, model="miki")
            rp = complex(res.plane_reflection_coefficient[0])
            q = complex(res.reflection_coefficient[0])
            ratio = float(res.r_direct) / float(res.r_reflected)
            phase = np.exp(1j * k * (float(res.r_reflected) - float(res.r_direct)))
            rp_mag.append(abs(rp))
            q_mag.append(abs(q))
            dl_q.append(20.0 * np.log10(abs(1.0 + q * ratio * phase)))
            dl_p.append(20.0 * np.log10(abs(1.0 + rp * ratio * phase)))

    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    left.semilogx(heights, rp_mag, color=COLOR_PRIMARY, linewidth=2.0,
                  label="$|R_\\mathrm{p}|$ — plane wave", zorder=4)
    left.semilogx(heights, q_mag, color=COLOR_SECONDARY, linewidth=2.0,
                  label="$|Q|$ — spherical wave", zorder=4)
    left.invert_xaxis()
    left.set_title("The coefficients part as the path grazes",
                   pad=10)
    left.set_xlabel("Source = receiver height [m]  (grazing to the right)")
    left.set_ylabel("Magnitude")
    left.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="upper left", fontsize=9)

    right.semilogx(heights, dl_q, color=COLOR_SECONDARY, linewidth=2.0,
                   label="with $Q$ (spherical wave)", zorder=4)
    right.semilogx(heights, dl_p, color=COLOR_PRIMARY, linewidth=2.0,
                   linestyle="--", label="with $R_\\mathrm{p}$ alone (plane wave)",
                   zorder=4)
    right.invert_xaxis()
    right.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    right.annotate(
        f"{_fmt_minus(dl_q[-1], '.1f')} dB against {_fmt_minus(dl_p[-1], '.1f')} dB",
        xy=(float(heights[-1]), dl_q[-1]), xytext=(0.30, 0.18),
        textcoords="axes fraction", fontsize=10, color=COLOR_FG,
        arrowprops={"arrowstyle": "->", "color": COLOR_MUTED},
    )
    right.set_title("What the ground wave keeps alive (500 Hz, 50 m)",
                    pad=10)
    right.set_xlabel("Source = receiver height [m]  (grazing to the right)")
    right.set_ylabel("Level re free field [dB]")
    right.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    right.set_axisbelow(True)
    right.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ground_reflection_coefficient.svg")
    plt.close()


def generate_barrier_thickness_gain(output_dir: str) -> None:
    """What a thick barrier actually buys, in two models."""
    print("Generating barrier_thickness_gain.svg...")
    import math
    import warnings as _warnings

    from phonometry import barrier_insertion_loss, environment

    widths = np.linspace(0.0, 30.0, 46)
    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for freq, color in ((250.0, COLOR_PRIMARY), (500.0, COLOR_TERTIARY),
                            (1000.0, COLOR_SECONDARY)):
            thin = float(barrier_insertion_loss(
                [freq], 1.0, 50.0, 4.0, 100.0, 1.5,
                method="exact").insertion_loss[0])
            gain = []
            for e in widths:
                if e <= 0.0:
                    gain.append(0.0)
                    continue
                gain.append(float(barrier_insertion_loss(
                    [freq], 1.0, 50.0, 4.0, 100.0, 1.5, method="exact",
                    thickness=float(e)).insertion_loss[0]) - thin)
            left.plot(widths, gain, color=color, linewidth=2.0,
                      label=f"{freq:g} Hz", zorder=4)
    left.set_title("Wave-theoretic model: the path length alone",
                   pad=10)
    left.set_xlabel("Top width $e$ [m]")
    left.set_ylabel("Gain over the thin screen [dB]")
    left.set_ylim(-0.1, 5.0)
    left.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="upper left", fontsize=9)

    dss = math.hypot(50.0, 3.0)
    for freq, color in ((250.0, COLOR_PRIMARY), (500.0, COLOR_TERTIARY),
                        (1000.0, COLOR_SECONDARY)):
        thin = float(environment.barrier_attenuation(
            environment.Barrier(source_to_edge=dss,
                                edge_to_receiver=math.hypot(50.0, 2.5)),
            100.0, [freq])[0])
        gain = []
        for e in widths:
            if e <= 0.0:
                gain.append(0.0)
                continue
            dsr = math.hypot(50.0 - float(e), 2.5)
            gain.append(float(environment.barrier_attenuation(
                environment.Barrier(source_to_edge=dss, edge_to_receiver=dsr,
                                    edge_separation=float(e)),
                100.0, [freq])[0]) - thin)
        right.plot(widths, gain, color=color, linewidth=2.0,
                   label=f"{freq:g} Hz", zorder=4)
    right.axhline(10.0 * np.log10(3.0), color=COLOR_MUTED, linestyle=":",
                  linewidth=1.4,
                  label=r"$10\,\mathrm{lg}\,3 = 4.77$ dB (the $C_3$ ceiling)",
                  zorder=3)
    right.set_title("ISO 9613-2: the path length plus the $C_3$ factor",
                    pad=10)
    right.set_xlabel("Edge separation $e$ [m]")
    right.set_ylabel("Gain over the single edge [dB]")
    right.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    right.set_axisbelow(True)
    right.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "barrier_thickness_gain.svg")
    plt.close()


def generate_shadow_zone_map(output_dir: str) -> None:
    """Is my range long enough for refraction to matter?"""
    print("Generating shadow_zone_map.svg...")
    from phonometry import shadow_zone_distance

    gradients = np.geomspace(0.02, 0.4, 160)
    _fig, ax = plt.subplots(figsize=(11, 6.2))
    pairs = ((0.5, 1.5, COLOR_PRIMARY), (2.0, 2.0, COLOR_TERTIARY),
             (2.0, 10.0, COLOR_SECONDARY))
    for hs, hr, color in pairs:
        ax.loglog(gradients,
                  [shadow_zone_distance(-float(g), hs, hr, ground_speed=340.0)
                   for g in gradients], color=color, linewidth=2.0,
                  label=f"$h_\\mathrm{{s}}/h_\\mathrm{{r}}$ = {hs:g} / {hr:g} m", zorder=4)
    ax.axvline(0.1, color=COLOR_MUTED, linestyle="--", linewidth=1.3, zorder=2)
    ax.plot([0.1], [shadow_zone_distance(-0.1, 2.0, 2.0, ground_speed=340.0)],
            "o", color=COLOR_FG, markersize=8, zorder=5)
    ax.annotate("representative −0.1 s⁻¹: 233 m",
                xy=(0.1, 233.2), xytext=(0.022, 90.0), fontsize=10,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})
    grad_log = 1.0 * np.log(101.0) / 10.0
    ax.plot([grad_log],
            [shadow_zone_distance(-grad_log, 2.0, 2.0, ground_speed=340.0)],
            "s", color=COLOR_SECONDARY, markersize=8, zorder=5)
    ax.annotate("the page's $b$ = −1 m/s case: 109 m",
                xy=(grad_log, 108.6), xytext=(0.12, 480.0), fontsize=10,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})
    ax.set_title("Where the Acoustic Shadow Starts ($c_0$ = 340 m/s)",
                 pad=12)
    ax.set_xlabel("Upward sound-speed gradient $|g|$ [1/s]")
    ax.set_ylabel(r"Shadow-zone distance $x_{\mathrm{shadow}}$ [m]")
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda g: 340.0 / np.maximum(g, 1e-9),
                          lambda r: 340.0 / np.maximum(r, 1e-9)))
    secondary.set_xlabel("Radius of curvature $R_\\mathrm{c} = c_0/|g|$ [m]")
    plt.tight_layout()
    save_figure(output_dir, "shadow_zone_map.svg")
    plt.close()


def generate_refraction_homogeneous_check(output_dir: str) -> None:
    """The GFPE against the coherent two-ray field with the gradient switched off."""
    print("Generating refraction_homogeneous_check.svg...")
    import warnings as _warnings

    from phonometry import atmospheric_parabolic_equation, linear_sound_speed_profile

    c0, freq, zs = 343.0, 500.0, 2.0
    flat = linear_sound_speed_profile(1e-12, ground_speed=c0, max_height=200.0)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        pe = atmospheric_parabolic_equation(freq, flat, source_height=zs,
                                            impedance=1e6 + 0j, max_range=520.0,
                                            max_height=150.0)
    ranges = np.asarray(pe.ranges)
    solver = np.asarray(pe.level_at_height(zs))
    k = 2.0 * np.pi * freq / c0
    r1 = np.hypot(ranges, 0.0)
    r2 = np.hypot(ranges, 2.0 * zs)
    two_ray = 20.0 * np.log10(np.abs(1.0 + (r1 / r2) * np.exp(1j * k * (r2 - r1))))

    _fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]})
    top.plot(ranges, solver, color=COLOR_PRIMARY, linewidth=1.8,
             label="GFPE, zero gradient", zorder=4)
    top.plot(ranges, two_ray, color=COLOR_SECONDARY, linewidth=1.6,
             linestyle="--", label="Coherent two-ray closed form", zorder=4)
    top.axhline(6.0, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                label="+6 dB (coherent sum)", zorder=2)
    top.set_ylim(-35.0, 10.0)
    top.set_ylabel("Level re free field [dB]")
    top.set_title("The Homogeneous Limit: 500 Hz, Rigid Ground, "
                  "$h_\\mathrm{s} = h_\\mathrm{r}$ = 2 m", pad=12)
    top.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    top.set_axisbelow(True)
    top.legend(loc="lower right", fontsize=9)

    bottom.plot(ranges, solver - two_ray, color=COLOR_TERTIARY, linewidth=1.6,
                zorder=4)
    bottom.axhline(0.6, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                   zorder=2)
    bottom.axhline(-0.6, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                   zorder=2)
    bottom.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=2)
    bottom.set_ylim(-6.0, 18.0)
    bottom.set_xlabel("Range [m]")
    bottom.set_ylabel("Residual [dB]")
    bottom.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    bottom.set_axisbelow(True)
    bottom.text(0.98, 0.90, "dotted lines: ±0.6 dB", transform=bottom.transAxes,
                ha="right", va="top", fontsize=9, color=COLOR_MUTED)
    plt.tight_layout()
    save_figure(output_dir, "refraction_homogeneous_check.svg")
    plt.close()


def generate_cnossos_road_corrections(output_dir: str) -> None:
    """Temperature, studded tyres and junctions, as changes in the line power."""
    print("Generating cnossos_road_corrections.svg...")
    from phonometry import (
        JunctionType,
        RoadTraffic,
        RoadVehicleCategory,
        road_source_power,
    )

    def a_line(**kwargs: Any) -> float:
        return float(road_source_power(**kwargs).a_weighted_line_power)

    light = [RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0, 50.0)]
    heavy = [RoadTraffic(RoadVehicleCategory.HEAVY, 1000.0, 50.0)]

    _fig, (t_ax, s_ax, j_ax) = plt.subplots(1, 3, figsize=(14.5, 4.8))
    temps = np.linspace(-10.0, 40.0, 60)
    for flow, name, color in ((light, "Light (1)", COLOR_PRIMARY),
                              (heavy, "Heavy (3)", COLOR_SECONDARY)):
        ref = a_line(traffic=flow)
        t_ax.plot(temps, [a_line(traffic=flow, temperature=float(t)) - ref
                          for t in temps], color=color, linewidth=2.0,
                  label=name, zorder=4)
    t_ax.axvline(20.0, color=COLOR_MUTED, linestyle=":", linewidth=1.3, zorder=2)
    t_ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    t_ax.set_title("Air temperature (2.2.10)", pad=10)
    t_ax.set_xlabel(r"Air temperature $\tau$ [°C]")
    t_ax.set_ylabel("Change in line power [dB(A)]")
    t_ax.legend(loc="upper right", fontsize=9)

    speeds = np.linspace(20.0, 130.0, 56)
    for share, color in ((0.2, COLOR_PRIMARY), (0.5, COLOR_TERTIARY)):
        s_ax.plot(speeds, [
            a_line(traffic=[RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0,
                                        float(v), studded_fraction=share)],
                   studded_months=4.0)
            - a_line(traffic=[RoadTraffic(RoadVehicleCategory.LIGHT, 1000.0,
                                          float(v))])
            for v in speeds], color=color, linewidth=2.0,
            label=rf"$Q_{{\mathrm{{stud}}}}$ = {share:g}, $T_\mathrm{{s}}$ = 4 months",
            zorder=4)
    for edge in (50.0, 90.0):
        s_ax.axvline(edge, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                     zorder=2)
    s_ax.set_title("Studded tyres (2.2.6-2.2.9)", pad=10)
    s_ax.set_xlabel("Speed $v$ [km/h]")
    s_ax.set_ylabel("Change in line power [dB(A)]")
    s_ax.legend(loc="upper right", fontsize=8)

    xs = np.linspace(0.0, 120.0, 61)
    ref = a_line(traffic=light)
    for junction, name, color in (
        (JunctionType.CROSSING, "Crossing with lights", COLOR_PRIMARY),
        (JunctionType.ROUNDABOUT, "Roundabout", COLOR_SECONDARY),
    ):
        j_ax.plot(xs, [a_line(traffic=light, junction_distance=float(x),
                              junction_type=junction) - ref for x in xs],
                  color=color, linewidth=2.0, label=name, zorder=4)
    j_ax.axvline(100.0, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                 zorder=2)
    j_ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    j_ax.set_title("Junctions (2.2.17, 2.2.18)", pad=10)
    j_ax.set_xlabel("Distance to the junction $|x|$ [m]")
    j_ax.set_ylabel("Change in line power [dB(A)]")
    j_ax.legend(loc="lower right", fontsize=9)

    for ax in (t_ax, s_ax, j_ax):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_corrections.svg")
    plt.close()


def generate_cnossos_road_gradient(output_dir: str) -> None:
    """The asymmetric road-gradient correction to propulsion noise."""
    print("Generating cnossos_road_gradient.svg...")
    from phonometry import CNOSSOS_A_WEIGHTING, road_propulsion_noise

    weights = np.asarray(CNOSSOS_A_WEIGHTING)

    def a_weighted(bands: Any) -> float:
        return float(10.0 * np.log10(
            np.sum(10.0 ** ((np.asarray(bands) + weights) / 10.0))))

    slopes = np.linspace(-14.0, 14.0, 141)
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    colors = {"1": COLOR_PRIMARY, "2": COLOR_TERTIARY, "3": COLOR_SECONDARY}
    for category, color in colors.items():
        for speed, style in ((50.0, "--"), (80.0, "-")):
            flat = a_weighted(road_propulsion_noise(category, speed))
            ax.plot(slopes,
                    [a_weighted(road_propulsion_noise(category, speed,
                                                      gradient=float(s))) - flat
                     for s in slopes], style, color=color, linewidth=1.9,
                    label=f"Category {category}, {speed:g} km/h", zorder=4)
    for edge in (-12.0, 12.0):
        ax.axvline(edge, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                   zorder=2)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title("CNOSSOS-EU Road-Gradient Correction (2.2.13-2.2.16)",
                 pad=12)
    ax.set_xlabel("Road gradient $s$ [%]  (negative = downhill)")
    ax.set_ylabel("Propulsion-noise correction [dB(A)]")
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=9, ncol=3)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_gradient.svg")
    plt.close()


def generate_cnossos_road_surfaces(output_dir: str) -> None:
    """Table F-4: the octave-band surface coefficient of five pavements."""
    print("Generating cnossos_road_surfaces.svg...")
    from phonometry import RoadSurface, road_surface_coefficients

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    rows = (
        (RoadSurface.REFERENCE, COLOR_FG, ":"),
        (RoadSurface.TWO_LAYER_ZOAB_FINE, COLOR_PRIMARY, "-"),
        (RoadSurface.ONE_LAYER_ZOAB, COLOR_TERTIARY, "-"),
        (RoadSurface.THIN_LAYER_A, "#9467bd", "-"),
        (RoadSurface.SMA_NL8, "#e8a838", "-"),
        (RoadSurface.HARD_ELEMENTS_NOT_HERRINGBONE, COLOR_SECONDARY, "-"),
    )
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    for surface, color, style in rows:
        row = road_surface_coefficients(surface)
        span = ("all speeds" if row.speed_range is None
                else "{:g}-{:g} km/h".format(*row.speed_range))
        ax.semilogx(bands, np.asarray(row.alpha["1"]), style, color=color,
                    marker="o", markersize=5, linewidth=1.9,
                    markerfacecolor="white", markeredgewidth=1.2,
                    label=f"{surface.value} ({span})", zorder=4)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title("CNOSSOS-EU Road Surfaces (Table F-4, light vehicles)",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Surface coefficient $\alpha$ [dB]")
    ax.set_xlim(63.0, 8000.0)
    format_frequency_axis(ax, 63.0, 8000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_surfaces.svg")
    plt.close()


def generate_cnossos_rail_components(output_dir: str) -> None:
    """Which source line carries what, and when source B wakes up."""
    print("Generating cnossos_rail_components.svg...")
    from phonometry import RailwayVehicle, railway_source_power

    stock, track = _cnossos_rail_scene()

    def total(bands: Any) -> float:
        return float(10.0 * np.log10(np.sum(10.0 ** (np.asarray(bands) / 10.0))))

    _fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4))
    result = railway_source_power(
        RailwayVehicle(stock, flow_rate=96.0, speed=160.0), track,
        phi=90.0, psi=10.0)
    freqs = np.asarray(result.third_octave_frequencies)
    palette = {"rolling": COLOR_PRIMARY, "traction": COLOR_TERTIARY,
               "aerodynamic": COLOR_SECONDARY, "bridge": "#9467bd"}
    for name, spectrum in result.components.items():
        for row, height, style in ((0, "0,5 m", "-"), (1, "4,0 m", "--")):
            values = np.asarray(spectrum)[row]
            if not np.any(np.isfinite(values)):
                continue
            left.semilogx(freqs, values, style, color=palette.get(name, COLOR_MUTED),
                          linewidth=1.9, label=f"{name} @ {height}", zorder=4)
    left.set_title("Components at 160 km/h, by source height",
                   pad=10)
    left.set_xlabel("1/3-octave band centre frequency [Hz]")
    left.set_ylabel("Sound power [dB re 1 pW]")
    left.set_xlim(50.0, 10000.0)
    format_frequency_axis(left, 50.0, 10000.0)
    left.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="lower center", fontsize=9)

    speeds = np.linspace(60.0, 350.0, 40)
    for row, label, color in ((0, "Source A (0,5 m)", COLOR_PRIMARY),
                              (1, "Source B (4,0 m)", COLOR_SECONDARY)):
        right.plot(speeds, [
            total(railway_source_power(
                RailwayVehicle(stock, flow_rate=96.0, speed=float(v)), track,
                phi=90.0, psi=10.0).line_power[row]) for v in speeds],
            color=color, linewidth=2.0, label=label, zorder=4)
    right.axvline(200.0, color=COLOR_MUTED, linestyle=":", linewidth=1.4,
                  label="Aerodynamic threshold (2.3.13)", zorder=2)
    right.set_title("Each source line against speed", pad=10)
    right.set_xlabel("Speed $v$ [km/h]")
    right.set_ylabel("Total line power [dB re 1 pW per metre]")
    right.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    right.set_axisbelow(True)
    right.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_components.svg")
    plt.close()


def generate_cnossos_rail_directivity(output_dir: str) -> None:
    """Two editions of the vertical correction, and the horizontal dipole."""
    print("Generating cnossos_rail_directivity.svg...")
    from phonometry import DirectivityEdition, vertical_directivity

    psi = np.linspace(-89.0, 89.0, 179)
    fig, (left, right) = plt.subplots(
        1, 2, subplot_kw={"projection": "polar"}, figsize=(12.5, 5.8))
    bands = ((7, "250 Hz", COLOR_PRIMARY), (19, "4 kHz", COLOR_SECONDARY))
    for band, name, color in bands:
        for edition, style in ((DirectivityEdition.CURRENT, "-"),
                               (DirectivityEdition.ORIGINAL_2015, "--")):
            values = [float(vertical_directivity(float(a), edition=edition)[band])
                      for a in psi]
            left.plot(np.radians(psi), values, style, color=color, linewidth=1.9,
                      label=f"{name}, {edition.value}")
    left.set_title("Vertical correction of source A (2.3.16)",
                   pad=18)
    left.set_thetamin(-90.0)
    left.set_thetamax(90.0)
    left.set_theta_zero_location("E")
    left.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    left.legend(loc="lower left", fontsize=8, bbox_to_anchor=(-0.15, -0.12))

    phi = np.radians(np.linspace(0.0, 360.0, 361))
    right.plot(phi, 10.0 * np.log10(0.01 + 0.99 * np.sin(phi) ** 2),
               color=COLOR_TERTIARY, linewidth=2.0)
    right.set_title("Horizontal dipole (2.3.15)", pad=18)
    right.set_theta_zero_location("E")
    right.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    right.annotate("−20 dB along the track", xy=(0.0, -20.0),
                   xytext=(0.35, 0.02), textcoords="figure fraction",
                   fontsize=9, color=COLOR_FG,
                   arrowprops={"arrowstyle": "->", "color": COLOR_MUTED})
    fig.suptitle("CNOSSOS-EU Railway Source Directivity",
                 fontsize=13)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_directivity.svg")
    plt.close()


def generate_wind_turbine_apparent_power(output_dir: str) -> None:
    """LWA as a function of wind speed, with a voided and an asterisked bin."""
    print("Generating wind_turbine_apparent_power.svg...")
    from phonometry import apparent_sound_power_level, slant_distance

    r1 = slant_distance(hub_height=80.0, rotor_diameter=100.0)
    bins = np.arange(4.0, 12.5, 0.5)
    shape = np.array([-4.0, -1.5, 0.0, -1.0, -3.0])   # 250 Hz to 1 kHz
    lwa = np.array([
        float(apparent_sound_power_level(
            44.0 + 12.0 / (1.0 + np.exp(-(v - 7.0))) + shape, r1))
        for v in bins])
    voided, flagged = 0, 1        # the two lowest bins, by their margin

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.plot(bins, lwa, "-", color=COLOR_PRIMARY, linewidth=2.2, zorder=3)
    keep = np.ones(bins.size, dtype=bool)
    keep[[voided, flagged]] = False
    ax.plot(bins[keep], lwa[keep], "o", color=COLOR_PRIMARY, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6,
            label="Valid bin (margin $>$ 6 dB)", zorder=5)
    ax.plot([bins[flagged]], [lwa[flagged]], "o", color="#e8a838", markersize=9,
            label="3-6 dB margin: reported with an asterisk", zorder=5)
    ax.plot([bins[voided]], [lwa[voided]], "x", color=COLOR_SECONDARY,
            markersize=12, markeredgewidth=2.4,
            label=r"Margin $\leq$ 3 dB: bin voided", zorder=5)
    ax.set_title("Apparent Sound Power against Wind Speed (IEC 61400-11)",
                 pad=12)
    ax.set_xlabel("Hub-height wind speed [m/s]  (0,5 m/s bins)")
    ax.set_ylabel(r"$L_{W\!A}$ [dB(A) re 1 pW]")
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    # Formula (29): the same series against the 10 m reference-roughness speed.
    z0ref, hub = 0.05, 80.0
    factor = np.log(10.0 / z0ref) / np.log(hub / z0ref)
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda v: v * factor, lambda v: v / factor))
    secondary.set_xlabel(
        r"Formula (29) wind speed at 10 m, $z_{0,\mathrm{ref}}$ = 0,05 m [m/s]")
    plt.tight_layout()
    save_figure(output_dir, "wind_turbine_apparent_power.svg")
    plt.close()


def generate_wind_turbine_audibility_criterion(output_dir: str) -> None:
    """The critical bandwidth and the audibility criterion against frequency."""
    print("Generating wind_turbine_audibility_criterion.svg...")
    from phonometry import critical_bandwidth

    fc = np.geomspace(20.0, 10000.0, 400)
    zwicker = 25.0 + 75.0 * (1.0 + 1.4 * (fc / 1000.0) ** 2) ** 0.69
    la = -2.0 - np.log10(1.0 + (fc / 502.0) ** 2.5)

    _fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    left.loglog(fc, zwicker, color=COLOR_PRIMARY, linewidth=2.0,
                label="IEC 61400-11 critical band (Zwicker)", zorder=4)
    left.loglog(fc, [critical_bandwidth(float(f)) for f in fc], "--",
                color=COLOR_TERTIARY, linewidth=1.8,
                label="ISO 1996-2 Table C.1", zorder=4)
    left.plot([20.0, 70.0], [100.0, 100.0], color=COLOR_SECONDARY, linewidth=5.0,
              solid_capstyle="butt", label="9.5.3: fixed 20-120 Hz band", zorder=5)
    left.set_title("Two critical bandwidths on one page",
                   pad=10)
    left.set_xlabel("Tone frequency $f_\\mathrm{c}$ [Hz]")
    left.set_ylabel("Critical bandwidth [Hz]")
    left.set_xlim(20.0, 10000.0)
    format_frequency_axis(left, 20.0, 10000.0)
    left.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    left.set_axisbelow(True)
    left.legend(loc="upper left", fontsize=8.5)

    right.semilogx(fc, -la, color=COLOR_PRIMARY, linewidth=2.0,
                   label=r"Tonality needed to be audible ($\Delta L_\mathrm{a} > 0$)",
                   zorder=4)
    right.semilogx(fc, -la - 3.0, "--", color=COLOR_SECONDARY, linewidth=1.8,
                   label=r"Tonality needed to be reportable "
                         r"($\Delta L_\mathrm{a} \geq -3$ dB)",
                   zorder=4)
    right.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    right.set_title("The audibility criterion, read as required tonality",
                    pad=10)
    right.set_xlabel("Tone frequency [Hz]")
    right.set_ylabel(r"Required tonality $\Delta L_\mathrm{tn}$ [dB]")
    right.set_xlim(20.0, 10000.0)
    format_frequency_axis(right, 20.0, 10000.0)
    right.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5,
               zorder=0)
    right.set_axisbelow(True)
    right.legend(loc="upper left", fontsize=8.5)
    plt.tight_layout()
    save_figure(output_dir, "wind_turbine_audibility_criterion.svg")
    plt.close()


def generate_rd1367_tonal_correction(output_dir: str) -> None:
    """RD 1367/2007: the Kt test applied to the guide's own spectrum."""
    print("Generating rd1367_tonal_correction.svg...")
    from phonometry import environment

    freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000]
    levels = [58.0, 60.0, 59.0, 61.0, 72.0, 62.0, 60.0, 58.0, 56.0, 54.0, 52.0]
    result = environment.tonal_correction(levels, freqs)
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    result.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "rd1367_tonal_correction.svg")
    plt.close()


def generate_rd1367_kf_ki(output_dir: str) -> None:
    """The two step functions that decide Kf and Ki."""
    print("Generating rd1367_kf_ki.svg...")
    from phonometry import environment

    diffs = np.linspace(0.0, 20.0, 401)
    kf = [environment.low_frequency_correction(lceq=63.0 + float(d), laeq=63.0)
          for d in diffs]
    ki = [environment.impulsive_correction(laieq=63.0 + float(d), laeq=63.0)
          for d in diffs]

    _fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.step(diffs, kf, where="post", color=COLOR_PRIMARY, linewidth=2.4,
            label="$K_\\mathrm{f}$, from $L_{Ceq,Ti} - L_{Aeq,Ti}$", zorder=4)
    ax.step(diffs, ki, where="post", color=COLOR_SECONDARY, linewidth=2.0,
            linestyle="--", label="$K_\\mathrm{i}$, from $L_{AIeq,Ti} - L_{Aeq,Ti}$",
            zorder=4)
    for edge in (10.0, 15.0):
        ax.axvline(edge, color=COLOR_MUTED, linestyle=":", linewidth=1.3,
                   zorder=2)
    ax.plot([13.0], [3.0], "o", color=COLOR_PRIMARY, markersize=9, zorder=5,
            label="worked example: $L_f$ = 13 dB → $K_\\mathrm{f}$ = 3 dB")
    ax.plot([5.0], [0.0], "s", color=COLOR_SECONDARY, markersize=9, zorder=5,
            label="worked example: $L_i$ = 5 dB → $K_\\mathrm{i}$ = 0 dB")
    ax.set_title("RD 1367/2007: the Low-Frequency and Impulsive Corrections",
                 pad=12)
    ax.set_xlabel("Level difference [dB]")
    ax.set_ylabel("Correction [dB]")
    ax.set_ylim(-0.6, 7.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "rd1367_kf_ki.svg")
    plt.close()


def generate_rd1367_vs_iso_tonal(output_dir: str) -> None:
    """The RD grades a spectrum the ISO survey method does not flag."""
    print("Generating rd1367_vs_iso_tonal.svg...")
    from phonometry import environment

    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000],
                     dtype=float)
    deep = np.array([58.0, 60.0, 59.0, 61.0, 72.0, 62.0, 60.0, 58.0, 56.0,
                     54.0, 52.0])
    shallow = deep.copy()
    shallow[4] = 69.0

    _fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, spectrum, title in (
        (axes[0], deep, "$L_\\mathrm{t}$ = 10.5 dB: both methods agree"),
        (axes[1], shallow, "$L_\\mathrm{t}$ = 7.5 dB: the verdicts split"),
    ):
        rd = environment.tonal_correction(spectrum, freqs)
        flags = np.asarray(environment.tonal_seeking_survey(spectrum, freqs))
        colors = [COLOR_SECONDARY if i == 4 else COLOR_PRIMARY
                  for i in range(spectrum.size)]
        ax.bar(np.arange(spectrum.size), spectrum, color=colors, alpha=0.85,
               width=0.66, zorder=3)
        neighbours = np.full(spectrum.size, np.nan)
        neighbours[1:-1] = (spectrum[:-2] + spectrum[2:]) / 2.0
        ax.plot(np.arange(spectrum.size), neighbours, "_", color=COLOR_FG,
                markersize=22, markeredgewidth=2.4,
                label="Arithmetic mean of the two neighbours", zorder=5)
        ax.set_title(title, pad=10)
        ax.set_xticks(np.arange(spectrum.size))
        ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45)
        ax.set_xlabel("1/3-octave band centre frequency [Hz]")
        ax.set_ylim(40.0, 80.0)
        ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.text(
            0.03, 0.97,
            f"RD 1367 tonal_correction: $K_\\mathrm{{t}}$ = {rd.correction:.0f} dB\n"
            f"ISO 1996-2 survey flag at 250 Hz: {bool(flags[4])}\n"
            f"$L_\\mathrm{{t}}$ at 250 Hz = {_fmt_minus(rd.differences[4], '.1f')} dB",
            transform=ax.transAxes, va="top", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round", "facecolor": COLOR_PANEL, "alpha": 0.9},
        )
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Band sound pressure level [dB]")
    plt.tight_layout()
    save_figure(output_dir, "rd1367_vs_iso_tonal.svg")
    plt.close()
