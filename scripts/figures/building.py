#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Figures for the sound-insulation guides: what a finished element achieves.

Airborne and impact sound insulation as it is measured, verified and rated:
the laboratory quantities of ISO 10140 and ISO 15186, the field quantities of
ISO 16283 and ISO 10052, the laboratory flanking descriptors of ISO 10848, the
building envelope of ISO 16283-3 and EN 12354-3/-4, and the single-number
ratings of ISO 717 and CTE DB-HR that collapse every one of them to the figure
a regulation quotes. Everything here is embedded by a page under
``buildings/insulation/``.
"""

import matplotlib.pyplot as plt
import numpy as np

from phonometry._plot.common import theme_fill

from .i18n import _LANG
from .theme import (
    _THIRD_OCTAVE_16,
    COLOR_FG,
    COLOR_GRID,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    LABEL_FREQ_HZ,
    _band_index_axis,
    save_figure,
)


def generate_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 weighted rating: measured R', shifted reference, deviations."""
    print("Generating insulation_rating.png...")
    from phonometry.building.measurement.insulation import (
        _INDEX_500_THIRD,
        _REF_THIRD_OCTAVE,
        weighted_rating,
    )

    # ISO 717-1 Annex C worked example (Table C.1), 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])

    result = weighted_rating(measured)
    reference = np.asarray(_REF_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Rw

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(freqs, measured, shifted, where=(measured < shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-1)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R' (third octave)")

    # Rw is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(f"Rw = {result.rating} dB", xy=(500, result.rating),
                xytext=(560, result.rating - 9), fontsize=12, fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, (f"Rw (C ; Ctr) = {result.rating} "
               f"({result.c:+d} ; {result.ctr:+d}) dB")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-1 Weighted Sound Reduction Index (Annex C example)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 44)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_rating.png")
    plt.close()


def generate_impact_rating(output_dir: str) -> None:
    """ISO 717-2 weighted impact rating: measured Ln, shifted reference, CI."""
    print("Generating impact_rating.png...")
    from phonometry.building.measurement.insulation import (
        _INDEX_500_THIRD,
        _REF_IMPACT_THIRD_OCTAVE,
        weighted_impact_rating,
    )

    # ISO 717-2 Annex C worked example (Table C.1): laboratory bare massive
    # floor, one-third octave 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                         73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])

    result = weighted_impact_rating(measured)
    reference = np.asarray(_REF_IMPACT_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_IMPACT_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Ln,w

    _, ax = plt.subplots(figsize=(10, 6.5))
    # For impact sound an unfavourable deviation occurs where the MEASURED
    # curve lies ABOVE the reference (opposite sign to ISO 717-1 airborne).
    ax.fill_between(freqs, shifted, measured, where=(measured > shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations (measured above reference)")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-2)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured Ln (third octave)")

    # Ln,w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    # The annotation sits in the clear gap between the rising measured curve
    # and the flat low-frequency reference plateau; the string is identical
    # in both languages, so the placement holds for every variant.
    ax.annotate(f"Ln,w = {result.rating} dB", xy=(500, result.rating),
                xytext=(135, result.rating - 4.2), fontsize=12,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, f"Ln,w = {result.rating} dB ; CI = {result.ci:+d} dB"),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-2 Weighted Normalized Impact Sound Level "
                 "(Annex C example)", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized impact sound pressure level Ln [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(55, 86)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "impact_rating.png")
    plt.close()


def generate_facade_prediction(output_dir: str) -> None:
    """EN 12354-3 façade airborne insulation prediction (Annex F worked example)."""
    print("Generating facade_prediction.png...")
    from phonometry import FacadeElement, facade_sound_reduction

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Annex F elements: double wall, two windows (area, R) + a small air inlet (Dn,e).
    elements = [
        FacadeElement(name="wall", area=6.0, r=[41, 46, 52, 58, 64]),
        FacadeElement(name="window", area=4.5, r=[23, 22, 30, 36, 37]),
        FacadeElement(name="skylight", area=0.5, r=[24, 27, 30, 33, 30]),
        FacadeElement(name="air inlet", dn_e=[28, 23, 25, 38, 44]),
    ]
    result = facade_sound_reduction(
        elements, area=11.3, volume=50.0, frequencies=bands, bands="octave"
    )

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Per-element partial indices Rp: thin, faded; they set the transmission floor.
    el_colors = [COLOR_PRIMARY, COLOR_SECONDARY, "#9467bd", "#ff7f0e"]
    for (name, rp), colour in zip(result.element_r.items(), el_colors):
        ax.plot(x, rp, "--", color=colour, linewidth=1.1, alpha=0.65,
                marker=".", markersize=6, label=f"Rp — {name}")
    # Façade apparent reduction R' and standardized level difference D2m,nT.
    ax.plot(x, result.r_prime, "-", color=COLOR_FG, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="R′ (façade)")
    ax.plot(x, result.d_2m_nt, "-", color=COLOR_TERTIARY, linewidth=2.2, marker="s",
            markersize=6, zorder=5, label="D2m,nT")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Reduction index / level difference [dB]")
    ax.set_title("EN 12354-3 Façade Sound Insulation (Annex F example)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    info = [
        f"R′tr,s,w = {result.r_tr_s_w} dB   (Ctr = {result.c_tr})",
        f"D2m,nT,w = {result.d_2m_nt_w} dB",
        "air inlet limits the low bands",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_prediction.png")
    plt.close()


def generate_intensity_insulation(output_dir: str) -> None:
    """ISO 15186-1 intensity SRI and the Kc-modified index RI,M = RI + Kc."""
    print("Generating intensity_insulation...")
    from phonometry import adaptation_term_kc, intensity_sound_reduction

    # 16 one-third-octave bands (100-3150 Hz); reuse the ISO 717-1 Annex C
    # airborne shape as the intensity SRI target (RI,w = 30 dB, a light wall).
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    ri = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                   28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    lp1, sm, s = 85.0, 12.0, 10.0
    # Levels that make Formula (7) land on RI, then modify with Kc (Annex B).
    l_in = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ri
    kc = adaptation_term_kc(freqs)
    result = intensity_sound_reduction(
        np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
    )
    assert result.r_i_modified is not None
    assert result.rating is not None and result.rating_modified is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the Kc adaptation lift between RI and RI,M (largest at low bands).
    ax.fill_between(x, result.r_i, result.r_i_modified, color=COLOR_TERTIARY,
                    alpha=0.18, zorder=0, label="Kc adaptation")
    ax.plot(x, result.r_i, "-", color=COLOR_PRIMARY, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="RI (intensity)")
    ax.plot(x, result.r_i_modified, "--", color=COLOR_TERTIARY, linewidth=2.2,
            marker="s", markersize=6, zorder=5, label="RI,M = RI + Kc")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in freqs], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound reduction index [dB]")
    ax.set_title("ISO 15186-1 Intensity Sound Reduction Index (RI and RI,M)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    # Data-only info box (language-neutral); the Kc lift is explained by the
    # shaded "Kc adaptation" legend entry, which the ES translator handles.
    info = [
        f"RI,w = {result.rating.rating} dB",
        f"RI,M,w = {result.rating_modified.rating} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_insulation.png")
    plt.close()


def generate_survey_insulation(output_dir: str) -> None:
    """ISO 10052 survey method: the reverberation-index correction D -> DnT."""
    print("Generating survey_insulation...")
    from phonometry import reverberation_index, survey_airborne_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # A masonry partition: raw level difference D and the measured receiving-
    # room reverberation time T per octave band.
    l1 = np.array([88.0, 90.0, 92.0, 92.0, 90.0])
    l2 = np.array([55.0, 51.0, 47.0, 41.0, 35.0])
    t = np.array([0.7, 0.6, 0.5, 0.45, 0.4])
    k = reverberation_index(t)
    res = survey_airborne_insulation(l1, l2, k, volume=50.0)
    assert res.rating is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the reverberation-index correction k between D and DnT.
    ax.fill_between(x, res.d, res.d_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="k = 10 log10(T/T0)")
    ax.plot(x, res.d, "--", color=COLOR_PRIMARY, linewidth=1.8, marker="o",
            markersize=6, zorder=5, label="D (level difference)")
    ax.plot(x, res.d_nt, "-", color=COLOR_FG, linewidth=2.6, marker="s",
            markersize=6, zorder=5, label="DnT (standardized)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 10052 Survey Method: Reverberation-Index Correction",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w = {res.rating.rating} dB  (C = {res.rating.c})",
        "octave bands, T0 = 0.5 s",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_insulation.png")
    plt.close()


def generate_heavy_impact_sources(output_dir: str) -> None:
    """Rubber-ball and bang-machine LFE spectra with their printed tolerances."""
    print("Generating heavy_impact_sources...")
    from phonometry import (
        a_weighted_maximum_impact_level,
        heavy_impact_source_limits,
        heavy_impact_source_specification,
    )

    _fig, (ax_src, ax_rate) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    x = np.arange(5)
    for source, colour in (
        ("rubber_ball", COLOR_PRIMARY), ("bang_machine", COLOR_SECONDARY)
    ):
        spec = heavy_impact_source_specification(source)
        _f, lower, upper = heavy_impact_source_limits(source)
        label = source.replace("_", " ")
        ax_src.fill_between(x, lower, upper, color=colour, alpha=0.30, zorder=1,
                            label=f"{label} tolerance")
        ax_src.plot(x, spec.force_exposure_level, "-o", color=colour, linewidth=2.4,
                    markersize=7, zorder=4, label=f"{label} nominal")
    ax_src.set_xticks(x)
    ax_src.set_xticklabels(["31.5", "63", "125", "250", "500"])
    ax_src.set_xlabel(LABEL_FREQ_HZ)
    ax_src.set_ylabel("Impact force exposure level LFE [dB re 1 N]")
    ax_src.set_title("Standard heavy impact sources\n(ISO 16283-2 Table A.1, "
                     "JIS A 1418-2 Tables A.1/A.2)", fontweight="bold", pad=10)
    ax_src.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_src.set_axisbelow(True)
    ax_src.legend(loc="upper right", fontsize=9)

    # ISO 717-2:2020 Table D.4: a field measurement in octave bands.
    levels = [65.3, 64.5, 58.0, 55.8]
    res = a_weighted_maximum_impact_level(levels)
    xr = np.arange(4)
    ax_rate.bar(xr - 0.19, res.levels, width=0.36, color=COLOR_PRIMARY,
                label="Li,Fmax (measured)", zorder=3)
    ax_rate.bar(xr + 0.19, res.corrected, width=0.36, color=COLOR_TERTIARY,
                label="Li,Fmax + A (Table D.3)", zorder=3)
    ax_rate.axhline(res.rating, color=COLOR_SECONDARY, linewidth=2.0, zorder=4,
                    label=f"LiA,Fmax = {res.rating} dB")
    ax_rate.set_xticks(xr)
    ax_rate.set_xticklabels(["63", "125", "250", "500"])
    ax_rate.set_xlabel(LABEL_FREQ_HZ)
    ax_rate.set_ylabel("Maximum impact sound pressure level [dB]")
    ax_rate.set_title("A-weighted heavy-impact rating\n(ISO 717-2 Annex D, "
                      "Table D.4 worked example)", fontweight="bold", pad=10)
    ax_rate.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_rate.set_axisbelow(True)
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.text(
        0.02, 0.03,
        f"unrounded sum = {res.unrounded:.6f} dB",
        transform=ax_rate.transAxes, va="bottom", ha="left", fontsize=10,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
              "edgecolor": COLOR_GRID},
    )

    plt.tight_layout()
    save_figure(output_dir, "heavy_impact_sources.png")
    plt.close()


def generate_ceiling_plenum_flanking(output_dir: str) -> None:
    """Ceiling/plenum flanking path Rcl and the CAC of an accredited report."""
    print("Generating ceiling_plenum_flanking...")
    from phonometry import (
        ceiling_attenuation_class,
        plenum_flanking_reduction_index,
    )

    _fig, (ax_path, ax_cac) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Vigran Figs. 9.11-9.13 geometry: LS = LR = 4,75 m, plenum h = 0,43 m,
    # 9,5 mm plasterboard ceiling, reflecting plenum sidewalls.
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    ceiling = np.array([17.0, 21.0, 25.0, 29.0, 32.0, 30.0, 38.0])
    x = np.arange(freqs.size)
    for depth, colour, style in (
        (0.43, COLOR_PRIMARY, "-"), (0.86, COLOR_TERTIARY, "--")
    ):
        res = plenum_flanking_reduction_index(
            ceiling, ceiling, ceiling_length=4.75, plenum_height=depth,
            frequency=freqs,
        )
        ax_path.plot(x, res.reduction_index, style, color=colour, linewidth=2.4,
                     marker="o", markersize=6, zorder=4,
                     label=f"Rcl, plenum h = {depth:g} m")
    ax_path.plot(x, 2.0 * ceiling, ":", color=COLOR_SECONDARY, linewidth=2.0,
                 marker="s", markersize=5, zorder=3, label="RS + RR (two ceilings)")
    ax_path.set_xticks(x)
    ax_path.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax_path.set_xlabel(LABEL_FREQ_HZ)
    ax_path.set_ylabel("Sound reduction index [dB]")
    ax_path.set_title("Suspended-ceiling plenum path\n(one-dimensional model, "
                      "LR = 4.75 m, reflecting sidewalls)",
                      fontweight="bold", pad=10)
    ax_path.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_path.set_axisbelow(True)
    ax_path.legend(loc="upper left", fontsize=9)

    # ALA 16-091-4 (2016), tested to ASTM E1414/E1414M-11a: CAC 34.
    dnc = np.array([14.4, 18.6, 21.7, 24.1, 23.4, 30.3, 33.7, 35.2,
                    41.6, 44.2, 42.1, 36.8, 35.7, 36.0, 36.9, 37.9])
    cac = ceiling_attenuation_class(dnc)
    xc = np.arange(dnc.size)
    ax_cac.fill_between(xc, cac.measured, cac.shifted_reference,
                        where=(cac.measured < cac.shifted_reference).tolist(),
                        color=COLOR_SECONDARY, alpha=0.25, interpolate=True,
                        zorder=1, label="deficiencies")
    ax_cac.plot(xc, cac.measured, "-o", color=COLOR_PRIMARY, linewidth=2.4,
                markersize=6, zorder=4, label="Dn,c (measured)")
    ax_cac.plot(xc, cac.shifted_reference, "--s", color=COLOR_TERTIARY,
                linewidth=2.0, markersize=5, zorder=3,
                label="ASTM E413 contour, fitted")
    ax_cac.set_xticks(xc[::2])
    ax_cac.set_xticklabels(["125", "200", "315", "500", "800", "1.25k",
                            "2k", "3.15k"], rotation=45, fontsize=8)
    ax_cac.set_xlabel(LABEL_FREQ_HZ)
    ax_cac.set_ylabel("Normalized ceiling attenuation Dn,c [dB]")
    ax_cac.set_title(f"Ceiling attenuation class\n(ASTM E1414/E413, "
                     f"CAC = {cac.rating} dB)", fontweight="bold", pad=10)
    ax_cac.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_cac.set_axisbelow(True)
    ax_cac.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "ceiling_plenum_flanking.png")
    plt.close()


def generate_flanking_transmission(output_dir: str) -> None:
    """ISO 10848 vibration reduction index Kij per band with the mean K̄ij."""
    print("Generating flanking_transmission...")
    from phonometry import vibration_reduction_index

    freqs = [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
             800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
    # A rigid T-junction of two heavy walls: measured direction-averaged velocity
    # level difference rising gently with frequency (typical laboratory data).
    dv = np.array([4.5, 4.8, 5.2, 5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7, 9.2, 9.8,
                   10.3, 10.9, 11.4, 11.9, 12.3, 12.7])
    res = vibration_reduction_index(
        dv, junction_length=4.0, area_i=12.0, area_j=10.0,
        frequency=freqs,
        structural_reverberation_time_i=0.35,
        structural_reverberation_time_j=0.40,
    )
    assert res.single_number is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(x, res.k_ij, "-", color=COLOR_PRIMARY, linewidth=2.4, marker="o",
            markersize=6, zorder=5, label="Kij (ISO 10848)")
    ax.axhline(res.single_number, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=4, label="mean Kij (200-1250 Hz)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Vibration reduction index Kij [dB]")
    ax.set_title("ISO 10848 Junction Vibration Reduction Index",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "rigid T-junction, two heavy walls",
        "lij = 4 m, Si = 12 m^2, Sj = 10 m^2",
        "Formula (13), one-third octave",
        f"mean Kij = {res.single_number:.1f} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_transmission.png")
    plt.close()


def generate_insulation_uncertainty_demo(output_dir: str) -> None:
    """ISO 12999-1 per-band + single-number measurement uncertainty (situation B)."""
    print("Generating insulation_uncertainty_demo.png...")
    from phonometry.building.measurement.insulation import weighted_rating
    from phonometry.building.measurement.uncertainty import (
        band_uncertainty,
        insulation_coverage_factor,
        insulation_expanded_uncertainty,
        single_number_uncertainty,
    )

    # Reuse the ISO 717-1 Annex C measured R' curve (100 Hz .. 3150 Hz); its
    # weighted rating is R'w = 30 dB.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    rating = weighted_rating(measured).rating

    # Per-band standard uncertainty u (ISO 12999-1 Table 2, situation B); match
    # each measured band to its tabulated value, then expand at k = 1.96 (95 %).
    band = band_uncertainty("airborne", "B")
    band_f, band_u = band.to_arrays()
    idx = [int(np.argmin(np.abs(band_f - f))) for f in freqs]
    u_band = band_u[idx]
    k = insulation_coverage_factor(0.95)
    exp_band = np.array(
        [insulation_expanded_uncertainty(float(v), 0.95) for v in u_band]
    )

    # Single-number expanded uncertainty for the rating.
    u_single = single_number_uncertainty("r_w", "B")
    exp_single = insulation_expanded_uncertainty(u_single, 0.95)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Nested bands: the inner one is placed twice as far from the page as the
    # outer, so the two read as distinct steps of the same hue.
    ax.fill_between(freqs, measured - exp_band, measured + exp_band,
                    color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
                    label="Expanded uncertainty ±U (95 %)")
    ax.fill_between(freqs, measured - u_band, measured + u_band,
                    color=theme_fill(COLOR_PRIMARY, ax, delta_e=26.0), zorder=0.1,
                    label="Standard uncertainty ±u")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.9,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R'")

    # Single-number R'w with its expanded uncertainty, read at 500 Hz.
    ax.errorbar(500, rating, yerr=exp_single, fmt="D", color=COLOR_SECONDARY,
                markersize=9, capsize=6, elinewidth=1.8, zorder=6,
                label="R'w ± U (single number)")
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.35, zorder=0)

    # Word-free box (the situation and band meanings are in the title/legend, so
    # translation reduces to the automatic decimal-comma substitution).
    box = [
        f"R'w = {rating} ± {exp_single:.1f} dB",
        f"U = k·u ,  k = {k:g} (95 %)",
    ]
    ax.text(0.03, 0.97, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_title("ISO 12999-1 Measurement Uncertainty (situation B, airborne)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 42)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_uncertainty_demo.png")
    plt.close()


def generate_facade_elevation_geometry(output_dir: str) -> None:
    """Composite facade elevation with the element areas to scale.

    A masonry wall of 6 m2 with a 1,5 m2 window and its 0,3 m2 roller
    shutter box, the classic composite of the facade prediction. One
    concept: the areas the composite sound reduction index weighs.
    """
    print("Generating facade_elevation_geometry...")
    from phonometry import FacadeElement, plot_facade_elements

    elements = [
        FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
        FacadeElement("Window", area=1.5, r=[30.0] * 5),
        FacadeElement("Roller shutter box", area=0.3, r=[22.0] * 5),
    ]
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_facade_elements(elements, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "facade_elevation_geometry.svg")
    plt.close()


def generate_extended_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 Annex B enlarged-range rating of the Annex C Table C.2 wall."""
    print("Generating extended_insulation_rating...")
    from phonometry import weighted_rating_extended

    # ISO 717-1 Annex C: the 16-band R spectrum (Table C.1) enlarged to
    # 50 Hz - 5 kHz with the Table C.2 values.
    r_core = [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
              28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
    freqs = [50, 63, 80, *(_THIRD_OCTAVE_16), 4000, 5000]
    ext = weighted_rating_extended([18.7, 19.2, 20.0, *r_core, 26.8, 29.2],
                                   freqs)
    assert ext.measured is not None and ext.core.shifted_reference is not None
    assert ext.core.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, freqs)
    core = slice(3, 19)                     # the 16 core bands 100-3150 Hz
    enlarged = theme_fill(COLOR_FG, ax)
    ax.axvspan(-0.5, 2.5, color=enlarged, zorder=0,
               label="enlarged range (Annex B)")
    ax.axvspan(18.5, 20.5, color=enlarged, zorder=0)
    ax.plot(x, ext.measured, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="measured R")
    ax.plot(x[core], ext.core.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5,
            label="shifted reference (100-3150 Hz)")
    unfav = ext.core.measured < ext.core.shifted_reference
    ax.fill_between(x[core], ext.core.measured, ext.core.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_title("ISO 717-1 Enlarged-Range Rating (Annex B)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Rw(C;Ctr) = {ext.rating:g}({ext.c:g};{ext.ctr:g})",
        f"C50-5000 = {ext.c_50_5000:g},  Ctr,50-5000 = {ext.ctr_50_5000:g}",
        "rating on the core bands, terms on the full range",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "extended_insulation_rating.svg")
    plt.close()


def generate_field_airborne_insulation(output_dir: str) -> None:
    """ISO 16283-1 field airborne chain: D and DnT with the T correction."""
    print("Generating field_airborne_insulation...")
    from phonometry import airborne_insulation, weighted_rating

    # A separating wall between dwellings: source/receiving levels and the
    # measured receiving-room T per one-third-octave band.
    l1 = np.array([92.3, 93.1, 94.0, 94.4, 94.8, 95.0, 95.2, 95.4,
                   95.3, 95.1, 94.8, 94.4, 93.9, 93.3, 92.5, 91.6])
    d = np.array([38.2, 40.1, 42.6, 45.2, 47.8, 50.1, 52.3, 54.0,
                  55.6, 57.1, 58.2, 59.0, 59.6, 60.1, 60.3, 59.8])
    t2 = np.array([0.62, 0.58, 0.55, 0.53, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42])
    field = airborne_insulation(l1, l1 - d, t2, area=12.5, volume=30.4)
    assert field.r_prime is not None
    w = weighted_rating(field.dnt)
    w_rp = weighted_rating(field.r_prime)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.fill_between(x, field.d, field.dnt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="10 log10(T/T0)")
    ax.plot(x, field.d, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D (level difference)")
    ax.plot(x, field.dnt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=5, label="DnT (standardized)")

    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 16283-1 Field Airborne Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        f"R'w = {w_rp.rating} dB   (S = 12.5 m², V = 30.4 m³)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "field_airborne_insulation.svg")
    plt.close()


def generate_facade_field_insulation(output_dir: str) -> None:
    """ISO 16283-3 façade quantities D2m, D2m,nT, D2m,n and R'45 per band."""
    print("Generating facade_field_insulation...")
    from phonometry import facade_insulation, weighted_rating

    # A dwelling façade under the 45-degree loudspeaker method: the outdoor
    # level 2 m in front, the receiving-room level and T per band.
    l1_2m = np.array([76.0, 77.0, 78.0, 78.5, 79.0, 79.0, 79.0, 79.0,
                      78.5, 78.0, 77.5, 77.0, 76.5, 76.0, 75.0, 74.0])
    d2m = np.array([24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5,
                    36.0, 37.0, 38.0, 38.5, 39.0, 39.0, 38.5, 38.0])
    t2 = np.array([0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42])
    fac = facade_insulation(l1_2m, l1_2m - d2m, t2, volume=32.0, area=10.8,
                            surface_level=l1_2m + 3.0, method="loudspeaker",
                            frequencies=np.asarray(_THIRD_OCTAVE_16, float))
    assert fac.d_2m_n is not None and fac.r_prime is not None
    w = weighted_rating(fac.d_2m_nt)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, fac.d_2m_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=6, label="D2m,nT (standardized)")
    ax.plot(x, fac.d_2m, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D2m = L1,2m - L2")
    ax.plot(x, fac.d_2m_n, ":", color=COLOR_SECONDARY, linewidth=1.8,
            marker=".", zorder=5, label="D2m,n (normalized)")
    ax.plot(x, fac.r_prime, "-.", color=COLOR_TERTIARY, linewidth=1.8,
            marker="^", markersize=5, zorder=5, label="R'45° (element)")

    ax.set_ylabel("Level difference / reduction index [dB]")
    ax.set_title("ISO 16283-3 Field Facade Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Dls,2m,nT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        "45° loudspeaker method (-1.5 dB on R')",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_field_insulation.svg")
    plt.close()


def generate_survey_impact_insulation(output_dir: str) -> None:
    """ISO 10052 survey impact method: Li -> L'nT with the k correction."""
    print("Generating survey_impact_insulation...")
    from phonometry import reverberation_index, survey_impact_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Tapping machine on the floor above: octave-band receiving-room levels
    # and the measured receiving-room reverberation time.
    li = np.array([66.0, 64.0, 62.0, 60.0, 55.0])
    k = reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])
    res = survey_impact_insulation(li, k, volume=50.0)
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands, fontsize=10)
    ax.fill_between(x, res.l_i, res.l_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="-k = -10 log10(T/T0)")
    ax.plot(x, res.l_i, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=6, zorder=5, label="Li (impact level)")
    ax.plot(x, res.l_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=6,
            zorder=5, label="L'nT (standardized)")

    ax.set_ylabel("Impact sound pressure level [dB]")
    ax.set_title("ISO 10052 Survey Method: Impact Sound",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    info = [
        f"L'nT,w(CI) = {res.rating.rating}({res.rating.ci}) dB",
        "note the minus sign: a live room lowers L'nT",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_impact_insulation.svg")
    plt.close()


def generate_lab_insulation_result(output_dir: str) -> None:
    """ISO 10140 laboratory quantities: the airborne R and the impact Ln."""
    print("Generating lab_insulation_result...")
    from phonometry import lab_airborne_insulation, lab_impact_insulation

    # ISO 717-1 Annex C wall measured in an ISO 10140 suite (S = 10 m2,
    # V = 50 m3, T = 0.8 s) and the ISO 717-2 Annex C floor tapping levels.
    r = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                  28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    l1 = np.full(16, 90.0)
    t2 = np.full(16, 0.8)
    lab = lab_airborne_insulation(l1, l1 - r, t2, area=10.0, volume=50.0)
    li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                   73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
    imp = lab_impact_insulation(li, t2, volume=50.0)
    assert lab.rating is not None and imp.rating is not None

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax in (ax1, ax2):
        _band_index_axis(ax, _THIRD_OCTAVE_16, fontsize=7)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
    x = np.arange(16)

    ax1.plot(x, lab.r, "-o", color=COLOR_PRIMARY, linewidth=2.2,
             markersize=5, zorder=5, label="measured R")
    assert lab.rating.shifted_reference is not None
    ax1.plot(x, lab.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax1.set_ylabel("Sound reduction index R [dB]")
    ax1.set_title(f"Airborne: Rw(C;Ctr) = {lab.rating.rating}"
                  f"({lab.rating.c};{lab.rating.ctr}) dB", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(x, imp.l_n, "-o", color=COLOR_SECONDARY, linewidth=2.2,
             markersize=5, zorder=5, label="normalized Ln")
    assert imp.rating.shifted_reference is not None
    ax2.plot(x, imp.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax2.set_ylabel("Impact sound pressure level Ln [dB]")
    ax2.set_title(f"Impact: Ln,w(CI) = {imp.rating.rating}"
                  f"({imp.rating.ci}) dB", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("ISO 10140 Laboratory Insulation (flanking suppressed)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "lab_insulation_result.svg")
    plt.close()


def generate_intensity_element_insulation(output_dir: str) -> None:
    """ISO 15186-1 element-normalized level difference of a small element."""
    print("Generating intensity_element_insulation...")
    from phonometry import intensity_element_normalized_difference

    # A trickle ventilator in a masonry wall: source-room SPL 85 dB and the
    # normal intensity level over the Sm = 12 m2 measurement surface.
    l_in = np.array([57.9, 62.0, 60.6, 55.7, 55.9, 55.6, 53.5, 51.7,
                     50.3, 47.8, 46.5, 45.8, 44.9, 45.3, 47.3, 52.8])
    res = intensity_element_normalized_difference(
        np.full(16, 85.0), l_in, measurement_area=12.0, n=1
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_i_n_e, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="DI,n,e (element)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Element normalized level difference [dB]")
    ax.set_title("ISO 15186-1 Small-Element Insulation by Intensity",
                 fontweight="bold", pad=12)
    ax.margins(y=0.1)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"DI,n,e,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "DI,n,e = Lp1 - 6 - [LIn + 10 log10(Sm/A0)] + 10 log10 N",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_element_insulation.svg")
    plt.close()


def generate_flanking_level_difference(output_dir: str) -> None:
    """ISO 10848 overall flanking descriptor Dn,f with its Dn,f,w rating."""
    print("Generating flanking_level_difference...")
    from phonometry import normalized_flanking_level_difference

    # A lightweight junction measured in the laboratory: source-room level,
    # receiving-room level over the flanking path and the absorption area.
    l1 = np.full(16, 80.0)
    dnf_target = np.array([48, 49, 50, 51, 52, 54, 55, 57,
                           58, 59, 60, 61, 62, 63, 64, 65], dtype=float)
    res = normalized_flanking_level_difference(
        l1, l1 - dnf_target, absorption_area=np.full(16, 10.0)
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_n_f, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="Dn,f (flanking)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Normalized flanking level difference [dB]")
    ax.set_title("ISO 10848 Airborne Flanking Transmission",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"Dn,f,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "Dn,f = L1 - L2 - 10 log10(A/A0)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_level_difference.svg")
    plt.close()


def generate_radiated_power_outdoor(output_dir: str) -> None:
    """EN 12354-4 Annex G radiated sound power of a wall with a door."""
    print("Generating radiated_power_outdoor...")
    from phonometry import FacadeElement, radiated_sound_power

    # EN 12354-4 Annex G, side 1: a 10 x 20 m concrete wall segment with a
    # 6 x 4 m industrial door, inside level Lp,in and Cd = -5 dB.
    bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    seg = radiated_sound_power(
        [FacadeElement("wall", area=176.0, r=[32, 36, 36, 33, 39, 49, 57, 63]),
         FacadeElement("door", area=24.0, r=[21, 23, 28, 30, 30, 30, 30, 30])],
        lp_in=[70, 74, 76, 72, 70, 67, 62, 57], area=200.0, c_d=-5.0,
        r_prime_cap=40.0, octave_bands=bands)
    assert seg.l_w_dba is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x, seg.l_w, color=COLOR_PRIMARY, alpha=0.85, zorder=5,
           label="radiated $L_W$ per octave")
    ax.axhline(seg.l_w_dba, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=6,
               label=f"$L_{{WA}}$ = {seg.l_w_dba:.1f} dB(A)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Radiated sound power level [dB re 1 pW]")
    ax.set_ylim(0.0, float(np.max(seg.l_w)) * 1.35)
    ax.set_title("EN 12354-4 Radiated Sound Power (Annex G)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LW = Lp,in + Cd - R' + 10 log10(S/S0)",
        "wall 176 m² + industrial door 24 m², Cd = -5 dB",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG, zorder=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "radiated_power_outdoor.svg")
    plt.close()


def generate_dbhr_global_index(output_dir: str) -> None:
    """CTE DB-HR global index R'A of a measured wall (Manual Ejemplo 7.2).

    The published eighteen-band apparent sound reduction index R' of a field
    test, weighted with the normalised pink-noise spectrum of DB-HR Annex A
    Table A.5. The energy sum of the per-band transmitted level L_Ar,i - R'_i
    sets the global index, R'A = 51,4 dBA.
    """
    print("Generating dbhr_global_index.png...")
    from phonometry import ra

    r_prime = [
        36.2, 41.5, 36.9, 40.4, 44.7, 42.4, 45.7, 46.1, 47.1,
        52.3, 54.3, 57.5, 57.8, 57.3, 59.0, 62.8, 64.7, 65.3,
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the band insulation as bars and the
    # weighted per-band transmitted level as a line on a 1k/2k-labelled axis.
    ra(r_prime).plot(ax=ax, language=_LANG)
    save_figure(output_dir, "dbhr_global_index.png")
    plt.close()
