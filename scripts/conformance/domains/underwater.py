#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Underwater acoustics: quantities, propagation and the numerical solvers.

The ISO 18405 terminology quantities and the radiated-noise measurement
standards ISO 17208-1/-2 and ISO 18406, then the propagation that carries
them: sound speed in sea water (UNESCO/Chen-Millero, Del Grosso, Mackenzie,
Medwin), absorption (Thorp, Francois-Garrison, Ainslie-McColm), spreading laws
and the Weston flux-theory regimes, ambient noise (Wenz, Mellen), the
JOMOPANS-ECHO ship source level and the NMFS/Southall marine-mammal auditory
weighting.

The numerical solvers close it: normal modes, ray tracing and the parabolic
equation, from Jensen et al. Their oracles are analytic - the ideal waveguide,
an image-source sum, a linear-gradient ray family, free field - because a
solver checked against another solver is only checked against a shared
mistake.
"""

from __future__ import annotations

import math

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_UNDERWATER = "Underwater acoustics (ISO 18405/17208/18406)"


@register(
    _UNDERWATER,
    "ISO 18405:2017 / ISO 18406 Formula 7",
    "Sound pressure level of a synthetic tone, dB re 1 µPa",
)
def _chk_uw_spl() -> Outcome:
    fs = 48000
    t = np.arange(fs) / fs
    amp = 2.0  # Pa
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    expected = 20.0 * math.log10((amp / math.sqrt(2.0)) / 1e-6)
    return numeric(expected, ph.underwater.sound_pressure_level(x), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 18405:2017 / ISO 18406 Formulae 3-4",
    "Sound exposure level of a 2 s tone, dB re 1 µPa²·s",
)
def _chk_uw_sel() -> Outcome:
    fs = 48000
    t = np.arange(2 * fs) / fs
    amp = 1.0
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    spl = 20.0 * math.log10((amp / math.sqrt(2.0)) / 1e-6)
    expected = spl + 10.0 * math.log10(2.0)
    return numeric(expected, ph.underwater.sound_exposure_level(x, fs), 1e-3, places=4)


@register(
    _UNDERWATER,
    "ISO 18406:2017 (6.4.2.1.3)",
    "Peak sound pressure level of a known waveform, dB re 1 µPa",
)
def _chk_uw_peak() -> Outcome:
    fs = 48000
    t = np.arange(fs) / fs
    amp = 3.0
    x = amp * np.sin(2.0 * np.pi * 500.0 * t)
    expected = 20.0 * math.log10(amp / 1e-6)
    return numeric(expected, ph.underwater.peak_sound_pressure_level(x), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 17208-1:2016",
    "Radiated noise level from RMS pressure and distance, dB re 1 µPa·m",
)
def _chk_uw_rnl() -> Outcome:
    expected = 20.0 * math.log10(2.0) + 40.0  # p = 2 µPa, r = 100 m
    return numeric(
        expected,
        ph.underwater.radiated_noise_level(2e-6, 100.0),
        1e-4,
        places=4,
    )


@register(
    _UNDERWATER,
    "ISO 17208-2:2019 (Formula 3)",
    "Lloyd's-mirror surface correction ΔL at a known k·d_s",
)
def _chk_uw_delta_l() -> Outcome:
    draught, c, f = 10.0, 1500.0, 200.0
    ds = 0.7 * draught
    u = 2.0 * math.pi * f / c * ds
    expected = -10.0 * math.log10((2 * u**4 + 14 * u**2) / (14 + 2 * u**2 + u**4))
    res = ph.underwater.monopole_source_level(120.0, f, draught, c=c)
    return numeric(expected, float(res.surface_correction[0]), 1e-4, places=4)


@register(
    _UNDERWATER,
    "ISO 18406:2017 (Formulae 8-9)",
    "Cumulative SEL of N identical strikes = SEL_ss + 10·lg(N)",
)
def _chk_uw_cumulative_sel() -> Outcome:
    return numeric(
        180.0 + 10.0 * math.log10(50),
        ph.underwater.cumulative_sel_identical(180.0, 50),
        1e-6,
        places=4,
    )


# ===========================================================================
# Underwater sound propagation (propagation loss, closed-form)
# ===========================================================================
_UW_PROP = "Underwater sound propagation (propagation loss)"


@register(
    _UW_PROP,
    "Mackenzie (1981) nine-term equation",
    "Speed of sound at 25 °C, 35 ‰, 1000 m (canonical check value), m/s",
)
def _chk_uwp_mackenzie() -> Outcome:
    return numeric(
        1550.744,
        ph.underwater.sea_water_sound_speed(25.0, 35.0, 1000.0, model="mackenzie"),
        1e-2,
        unit="m/s",
        places=3,
    )


@register(
    _UW_PROP,
    "UNESCO/Chen-Millero vs Mackenzie",
    "Sound-speed agreement at 10 °C, 35 ‰, 1000 m (cross-model), m/s",
)
def _chk_uwp_unesco() -> Outcome:
    expected = ph.underwater.sea_water_sound_speed(
        10.0, 35.0, 1000.0, model="mackenzie"
    )
    got = ph.underwater.sea_water_sound_speed(10.0, 35.0, 1000.0, model="unesco")
    return numeric(expected, got, 1.0, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Del Grosso (1974) vs Mackenzie",
    "Sound-speed agreement at 10 °C, 35 ‰, 1000 m (cross-model), m/s",
)
def _chk_uwp_del_grosso() -> Outcome:
    expected = ph.underwater.sea_water_sound_speed(
        10.0, 35.0, 1000.0, model="mackenzie"
    )
    got = ph.underwater.sea_water_sound_speed(10.0, 35.0, 1000.0, model="del_grosso")
    return numeric(expected, got, 1.0, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Spherical spreading 20·lg(R)",
    "Geometrical spreading loss at R = 1000 m, dB",
)
def _chk_uwp_spreading() -> Outcome:
    return numeric(
        20.0 * math.log10(1000.0),
        float(ph.underwater.spreading_loss([1000.0], law="spherical")[0]),
        1e-9,
        unit="dB",
        places=4,
    )


@register(
    _UW_PROP,
    "Thorp (1967) absorption",
    "Volume absorption α at 10 kHz (cold deep water), dB/km",
)
def _chk_uwp_thorp() -> Outcome:
    f = 10.0  # kHz
    expected = 1.0936 * (0.1 * f**2 / (1 + f**2) + 40 * f**2 / (4100 + f**2))
    got = float(ph.underwater.seawater_absorption(10_000.0, model="thorp")[0])
    return numeric(expected, got, 1e-6, unit="dB/km", places=4)


@register(
    _UW_PROP,
    "Ainslie-McColm (1998) vs Francois-Garrison (1982)",
    "Absorption agreement at 10 kHz, 10 °C, 35 ‰, 0 m, pH 8, dB/km",
)
def _chk_uwp_absorption_agreement() -> Outcome:
    kw = {"temperature": 10.0, "salinity": 35.0, "depth": 0.0, "ph": 8.0}
    fg = float(
        ph.underwater.seawater_absorption(10_000.0, model="francois-garrison", **kw)[0]
    )
    am = float(
        ph.underwater.seawater_absorption(10_000.0, model="ainslie-mccolm", **kw)[0]
    )
    return numeric(fg, am, 0.1 * fg, unit="dB/km", places=4)


@register(
    _UW_PROP,
    "Francois-Garrison (1982) Part II Table IV",
    "Absorption α at 100 kHz, 10 °C, 35 ‰, 0 m, pH 8 (printed value), dB/km",
)
def _chk_uwp_fg_printed_table() -> Outcome:
    # Oracle: the printed absorption table of the source paper (J. Acoust.
    # Soc. Am. 72(6), 1982); tolerance is half a unit of the last printed
    # digit, i.e. the print's own rounding.
    kw = {"temperature": 10.0, "salinity": 35.0, "depth": 0.0, "ph": 8.0}
    got = float(
        ph.underwater.seawater_absorption(100_000.0, model="francois-garrison", **kw)[0]
    )
    return numeric(33.6, got, 0.05, unit="dB/km", places=3)


@register(
    _UW_PROP,
    "Del Grosso refit (Wong-Zhu 1995 Table IV)",
    "c(t90 = 20 °C, S = 35, P = 500 bar) vs the printed check table, m/s",
)
def _chk_uwp_del_grosso_printed_check() -> Outcome:
    # Oracle: the printed ITS-90 check table of the refit the module
    # implements (J. Acoust. Soc. Am. 97(3), 1995); the table lists pressure
    # in bars, Del Grosso's polynomial takes kg/cm² (1 bar = 1.019716 kg/cm²).
    from phonometry.underwater.propagation.sound_speed import _del_grosso

    got = float(_del_grosso(20.0, 35.0, 500.0 * 1.019716))
    return numeric(1603.679, got, 1e-3, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Wales-Heitmeyer (2002) ensemble spectrum",
    "Merchant-ship source PSD at 100 Hz (printed equation), dB re 1 µPa²/Hz",
)
def _chk_uwp_wales_heitmeyer() -> Outcome:
    # Oracle: the mean-spectrum closed form printed in J. Acoust. Soc. Am.
    # 111(3), 2002, hand-evaluated at 100 Hz.
    s = ph.underwater.ship_source_spectrum(
        model="wales-heitmeyer", frequency_hz=[100.0]
    )
    return numeric(158.4504, float(s.source_psd[0]), 1e-3, unit="dB", places=3)


@register(
    _UW_PROP,
    "Passive sonar equation (Urick/Etter)",
    "Figure of merit SL − (NL − DI) − DT, dB",
)
def _chk_uwp_sonar() -> Outcome:
    res = ph.underwater.passive_sonar_equation(
        140.0, 80.0, 60.0, directivity_index=10.0, detection_threshold=5.0
    )
    return numeric(85.0, res.figure_of_merit, 1e-9, unit="dB", places=4)


@register(
    _UW_PROP,
    "Seabed reflection (Rayleigh, normal incidence)",
    "Bottom loss at 90° grazing, sand ρ=1900 c=1650 over water, dB",
)
def _chk_uwp_seabed() -> Outcome:
    # Normal-incidence oracle: R = (Z2 − Z1)/(Z2 + Z1), BL = −20·lg|R|.
    z1, z2 = 1000.0 * 1500.0, 1900.0 * 1650.0
    expected = -20.0 * math.log10(abs((z2 - z1) / (z2 + z1)))
    res = ph.underwater.bottom_reflection_loss(
        90.0, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0
    )
    return numeric(expected, float(res.reflection_loss[0]), 1e-6, unit="dB", places=4)


@register(
    _UW_PROP,
    "Wenz wind noise (rule of fives)",
    "Wind spectrum level at 1 kHz, 5 kn (canonical anchor), dB re 1 µPa²/Hz",
)
def _chk_uwp_wind_noise() -> Outcome:
    # Wenz/Knudsen "25 dB (5 x 5)" is re 0.0002 dyn/cm2 = 20 uPa; re 1 uPa
    # (ISO 18405) the anchor is 25 + 20*lg(20) = 51.0206 dB (matches the
    # published Wenz chart: ~50 dB at 1 kHz for 4-6 kn).
    got = float(ph.underwater.wind_noise_spectrum(1000.0, 5.0)[0])
    return numeric(51.0206, got, 1e-4, unit="dB", places=4)


@register(
    _UW_PROP,
    "Mellen thermal noise",
    "Thermal spectrum level at 50 kHz, 16.85 °C (physical), dB re 1 µPa²/Hz",
)
def _chk_uwp_thermal_noise() -> Outcome:
    f, t, rho, c = 5.0e4, 16.85, 1025.0, 1500.0
    p2 = 4.0 * math.pi * 1.380649e-23 * (t + 273.15) * rho * f**2 / c
    expected = 10.0 * math.log10(p2 / (1e-6) ** 2)
    got = float(
        ph.underwater.thermal_noise_spectrum(
            f, temperature=t, density=rho, sound_speed=c
        )[0]
    )
    return numeric(expected, got, 1e-6, unit="dB", places=4)


@register(
    _UW_PROP,
    "JOMOPANS-ECHO ship source level",
    "Bulker V=13.5 kn L=211 m band level at 1 kHz (File S1 oracle), dB re 1 µPa m",
)
def _chk_uwp_ship_traffic() -> Outcome:
    # Oracle: authors' Excel reference calculator (File S1), decidecade band.
    s = ph.underwater.ship_source_spectrum(
        13.5, 211.0, vessel_class="bulker", model="jomopans-echo"
    )
    idx = int(min(range(len(s.frequency)), key=lambda i: abs(s.frequency[i] - 1000.0)))
    return numeric(161.394, float(s.band_level[idx]), 1e-2, unit="dB", places=3)


# ===========================================================================
# Underwater numerical propagation (Jensen et al., modes / rays / PE)
# ===========================================================================
@register(
    _UW_PROP,
    "UNESCO sound speed (EOS-80 canonical value)",
    "SVEL(S = 40, T68 = 40 °C, P = 1000 bar) vs Fofonoff & Millard 1983, m/s",
)
def _chk_uwp_unesco_canonical() -> Outcome:
    # Published canonical check of the UNESCO algorithm; the module implements
    # the Wong-Zhu ITS-90 refit, so T90 = T68/1.00024 and the tolerance covers
    # the published refit residual.
    from phonometry.underwater.propagation.sound_speed import _unesco

    got = float(_unesco(40.0 / 1.00024, 40.0, 1000.0))
    return numeric(1731.995, got, 0.02, unit="m/s", places=3)


@register(
    _UW_PROP,
    "Medwin (1975) sound speed (Ainslie Eqs. 1.2-1.4)",
    "∂c/∂T at 10 °C, neglecting the bracketed terms, m/s per °C",
)
def _chk_uwp_medwin_derivative() -> Outcome:
    # Oracle: Ainslie prints "∂c/∂T ≈ 4.6 − 0.110·T", i.e. 3.5 m/s per °C at
    # 10 °C (printed p. 20). The cubic term sits inside the brackets the
    # published derivative excludes, so it is removed before comparing.
    from phonometry.underwater.propagation.sound_speed import sea_water_sound_speed

    h = 1e-5
    grad = (
        sea_water_sound_speed(10.0 + h, 35.0, 0.0, model="medwin")
        - sea_water_sound_speed(10.0 - h, 35.0, 0.0, model="medwin")
    ) / (2.0 * h)
    return numeric(3.5, grad - 3.0 * 2.9e-4 * 100.0, 1e-3, unit="m/s per °C", places=4)


_UW_WESTON = "Underwater propagation regimes (Weston flux theory)"


@register(
    _UW_WESTON,
    "Ainslie (2010) Table 9.1, medium sand",
    "Reflection loss gradient η from Equation (9.51), Np/rad",
)
def _chk_uww_eta_sand() -> Outcome:
    # Oracle: the printed value 0.28 Np/rad of Table 9.1 (printed p. 454).
    return numeric(
        0.28,
        float(ph.underwater.reflection_loss_gradient("sand")),
        5e-3,
        unit="Np/rad",
        places=4,
    )


@register(
    _UW_WESTON,
    "Ainslie (2010) Table 9.1, mud",
    "Reflection loss gradient η from Equation (9.53) at 1 Hz, Np/rad",
)
def _chk_uww_eta_mud() -> Outcome:
    # Oracle: Table 9.1 prints η_mud = 0.021·f̂ Np/rad.
    got = float(ph.underwater.reflection_loss_gradient("mud", frequency_hz=1.0))
    return numeric(0.021, got, 5e-4, unit="Np/rad", places=5)


@register(
    _UW_WESTON,
    "Weston cylindrical spreading vs normal modes",
    "Range-averaged PL in an ideal 100 m waveguide at 100 Hz, 20-30 km, dB",
)
def _chk_uww_flux_vs_modes() -> Outcome:
    # Independent cross-check: the range average of the coherent modal field is
    # the incoherent modal sum, whose many-mode limit is exactly F = π/(r·H) --
    # Equation (9.42) with ψc = π/2. Averaged over receiver depth to remove the
    # sin² sampling bias of a single depth.
    import numpy as _np

    ranges = _np.linspace(20_000.0, 30_000.0, 1001)
    energies = [
        _np.mean(
            10.0
            ** (
                -ph.underwater.normal_modes(
                    100.0,
                    [0.0, 100.0],
                    [1500.0, 1500.0],
                    source_depth=41.0,
                    receiver_depth=float(zr),
                    ranges_m=ranges,
                ).propagation_loss
                / 10.0
            )
        )
        for zr in _np.linspace(10.0, 90.0, 9)
    ]
    numeric_pl = -10.0 * math.log10(float(_np.mean(energies)))
    flux = ph.underwater.weston_propagation_loss(
        ranges,
        100.0,
        100.0,
        critical_angle=90.0,
        reflection_loss_gradient_value=0.0,
    )
    expected = -10.0 * math.log10(
        float(_np.mean(10.0 ** (-flux.propagation_loss / 10.0)))
    )
    return numeric(expected, numeric_pl, 1.0, unit="dB", places=3)


_UW_FAUNA = "Marine-mammal auditory weighting (NMFS / Southall)"


@register(
    _UW_FAUNA,
    "NMFS (2018) Appendix D worked example",
    "Weighting factor adjustment W(1 kHz) for high-frequency cetaceans, dB",
)
def _chk_uwf_appendix_d() -> Outcome:
    # Oracle: the published worked example (printed p. 130) lists W(1 kHz) for
    # the five hearing groups; the HF value is -37.55 dB.
    got = float(
        ph.underwater.auditory_weighting(1000.0, "HF", guidance="nmfs-2018").weighting[
            0
        ]
    )
    return numeric(-37.55, got, 0.01, unit="dB", places=3)


@register(
    _UW_FAUNA,
    "NMFS (2024) v3.0 Table 5, otariid C",
    "C recomputed as the peak of W(f) for the OW row (printed 1.37, corrected 1.36), dB",
)
def _chk_uwf_otariid_c() -> Outcome:
    # NMFS's own footnote states the printed 1.37 should read 1.36; recomputing
    # C = -max W(f) from the same row's a/b/f1/f2 gives 1.3643 dB.
    import numpy as _np

    params = ph.underwater.weighting_parameters("OW", guidance="nmfs-2024")
    freqs = _np.logspace(0.0, 6.0, 400_001)
    shape = (
        ph.underwater.auditory_weighting(freqs, "OW", guidance="nmfs-2024").weighting
        - params.c_db
    )
    return numeric(1.3643, -float(_np.max(shape)), 5e-4, unit="dB", places=4)


@register(
    _UW_FAUNA,
    "Ainslie (2010) Equation (11.159), orca audiogram",
    "Hearing threshold at 50 kHz (third branch), dB re 1 µPa",
)
def _chk_uwf_orca() -> Outcome:
    # Oracle: "The threshold ... at the pulse center frequency (50 kHz) is
    # 51.2 dB re µPa²" (printed p. 619); the second branch would give 50.5 dB.
    return numeric(
        51.2,
        float(ph.underwater.orca_audiogram(50e3).threshold[0]),
        0.05,
        unit="dB",
        places=3,
    )


@register(
    _UW_FAUNA,
    "Ainslie (2010) §11.4.6, orca versus salmon",
    "Noise-limited figure of merit (SL + TS − NL + AG − DT)/2, dB re m²",
)
def _chk_uwf_orca_fom() -> Outcome:
    # Oracle: Table 11.7 (printed p. 624) prints FOM_NL = 51.0 dB re m².
    res = ph.underwater.active_sonar_equation(
        198.2,
        [0.0],
        -29.0,
        75.0,
        directivity_index=16.5,
        detection_threshold=8.7,
    )
    return numeric(51.0, res.figure_of_merit, 1e-6, unit="dB", places=3)


_UW_NUM = "Underwater numerical propagation (modes / rays / PE)"


@register(
    _UW_NUM,
    "Normal modes vs ideal waveguide",
    "Fundamental horizontal wavenumber kr1 at 20 Hz, 100 m (analytic), rad/m",
)
def _chk_uwn_modes() -> Outcome:
    d, c, f = 100.0, 1500.0, 20.0
    k = 2.0 * math.pi * f / c
    kr1 = math.sqrt(k**2 - (math.pi / d) ** 2)
    res = ph.underwater.normal_modes(
        f,
        [0.0, d],
        [c, c],
        source_depth=36.0,
        receiver_depth=46.0,
        bottom="pressure-release",
        n_depth_points=800,
    )
    return numeric(kr1, float(res.wavenumbers[0]), 1e-4, unit="rad/m", places=6)


@register(
    _UW_NUM,
    "Normal modes vs image-source oracle",
    "Absolute PL at 1 km in the ideal waveguide (converged image sum), dB",
)
def _chk_uwn_modes_absolute() -> Outcome:
    # Independent absolute anchor (does not share the Eq. 5.14 prefactor with
    # the implementation): converged image-source sum for D = 100 m, f = 20 Hz,
    # zs = 36 m, zr = 46 m gives PL(1 km) = 48.238 dB.
    res = ph.underwater.normal_modes(
        20.0,
        [0.0, 100.0],
        [1500.0, 1500.0],
        source_depth=36.0,
        receiver_depth=46.0,
        ranges_m=[1000.0],
        n_depth_points=3000,
    )
    return numeric(48.238, float(res.propagation_loss[0]), 0.02, unit="dB", places=3)


@register(
    _UW_NUM,
    "Ray tracing vs linear gradient",
    "Turning depth of a 10° ray, c = 1500 + 0.05z (circular arc), m",
)
def _chk_uwn_rays() -> Outcome:
    c0, g = 1500.0, 0.05
    xi = math.cos(math.radians(10.0)) / c0
    z_turn = (1.0 / xi - c0) / g
    res = ph.underwater.ray_trace(
        [0.0, 2000.0],
        [c0, c0 + g * 2000.0],
        source_depth=0.0,
        launch_angles_deg=[10.0],
        max_range=10_500.0,
        n_steps=20_000,
    )
    return numeric(z_turn, float(res.depths[0].max()), 1.0, unit="m", places=2)


@register(
    _UW_NUM,
    "Ray travel time vs iso-gradient closed form",
    "Travel time of a 10° ray at 10 km, c = 1500 + 0.05z (Medwin & Clay Eq. 3.3.20), s",
)
def _chk_uwn_ray_time() -> Outcome:
    # Oracle: Medwin & Clay, Fundamentals of Acoustical Oceanography (1998),
    # Eq. (3.3.20) p. 88, which in this library's horizontal-angle convention
    # reads t = (1/g) ln[(c2/c1)(1 + sin th1)/(1 + sin th2)], with the ray angle
    # at range r from their companion Eq. (3.3.21), sin th2 = sin th1 - xi g r.
    c0, g, th = 1500.0, 0.05, math.radians(10.0)
    xi = math.cos(th) / c0
    sin_th2 = math.sin(th) - xi * g * 10_000.0
    c2 = math.sqrt(1.0 - sin_th2**2) / xi
    t = math.log((c2 / c0) * (1.0 + math.sin(th)) / (1.0 + sin_th2)) / g
    res = ph.underwater.ray_trace(
        [0.0, 2000.0],
        [c0, c0 + g * 2000.0],
        source_depth=0.0,
        launch_angles_deg=[10.0],
        max_range=10_000.0,
        n_steps=20_001,
    )
    return numeric(t, float(res.travel_times[0, -1]), 1e-6, unit="s", places=6)


@register(
    _UW_NUM,
    "Parabolic equation vs free field",
    "PE propagation loss at 2 km, homogeneous medium (spherical spreading), dB",
)
def _chk_uwn_pe() -> Outcome:
    res = ph.underwater.parabolic_equation(
        50.0,
        [0.0, 20_000.0],
        [1500.0, 1500.0],
        source_depth=10_000.0,
        max_range=3000.0,
        range_step=2.0,
        n_depth_points=8192,
    )
    zi = int(min(range(res.depths.size), key=lambda i: abs(res.depths[i] - 10_000.0)))
    ri = int(min(range(res.ranges.size), key=lambda i: abs(res.ranges[i] - 2000.0)))
    return numeric(
        20.0 * math.log10(2000.0),
        float(res.propagation_loss[zi][ri]),
        0.1,
        unit="dB",
        places=3,
    )
