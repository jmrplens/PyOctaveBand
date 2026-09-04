#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for material characterization: how a surface answers a sound field.

What a specimen absorbs, scatters or yields under load: the reverberation-room
absorption of ISO 354 and the impedance-tube absorption of ISO 10534-2 with
the ISO 11654 rating over them, the scattering and diffusion coefficients of
ISO 17497-1/-2, the dynamic stiffness of a resilient layer (EN 29052-1) and
the airflow resistance of a porous one (ISO 9053-1).
"""

from __future__ import annotations

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _absorption_example() -> tuple[object, ReportMetadata, str]:
    """Absorption fiche: an ISO 11654 weighted sound absorption rating."""
    # The fifteen one-third-octave alpha_s (200 Hz to 5000 Hz) an accredited
    # ISO 354 certificate reports, whose octave means are the practical
    # coefficients (0.35, 1.00, 0.65, 0.60, 0.55) giving alpha_w = 0.60 with an
    # M shape indicator (ISO 11654 Annex A.2 shape); a broadband porous absorber.
    alpha_s = (
        0.30,
        0.35,
        0.40,  # 250 Hz octave -> alpha_p 0.35
        1.00,
        1.00,
        1.00,  # 500 Hz octave -> alpha_p 1.00
        0.62,
        0.66,
        0.67,  # 1000 Hz octave -> alpha_p 0.65
        0.58,
        0.60,
        0.62,  # 2000 Hz octave -> alpha_p 0.60
        0.53,
        0.55,
        0.57,  # 4000 Hz octave -> alpha_p 0.55
    )
    result = ph.materials.weighted_absorption_from_third_octave(alpha_s)
    metadata = ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        client="Example client",
        manufacturer="Example acoustics",
        area=10.8,
        mounting="Type A (mounted directly against a rigid wall)",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 354",
        test_date="2026-07-20",
        temperature=21.4,
        relative_humidity=54.0,
        pressure=101.0,
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-11654",
        requirement=0.55,
    )
    return result, metadata, "iso11654_absorption_example.pdf"


def _sound_absorption_example() -> tuple[object, ReportMetadata, str]:
    """ISO 354 fiche: a reverberation-room sound-absorption measurement.

    A documented clean-room example, derived in closed form from ISO 354:2003
    Eq. (5)/(7)/(8)/(9) with no air-attenuation correction (m = 0, the
    zero-attenuation reference condition). Room volume V = 200 m3 (the ISO 354
    reference volume) and specimen area S = 10.8 m2 (clause 6.2.1.1 range); at
    20 degC the speed of sound is c = 331 + 0.6*20 = 343 m/s (Eq. (6)), so the
    Sabine constant of the inversion is 55.3*V/c = 55.3*200/343 = 32.24490 m2 s.
    With the empty-room T1 and with-specimen T2 tables below, the equivalent
    sound absorption areas A = 55.3*V/(c*T) and the coefficient
    alpha_s = (A2 - A1)/S follow. Two worked bands:

    * 500 Hz: A1 = 32.24490/7.80 = 4.13396 m2, A2 = 32.24490/4.20 = 7.67736 m2,
      alpha_s = (7.67736 - 4.13396)/10.8 = 0.328 -> 0.33.
    * 1000 Hz: A1 = 32.24490/6.90 = 4.67317 m2, A2 = 32.24490/2.85 = 11.31400 m2,
      alpha_s = (11.31400 - 4.67317)/10.8 = 0.615 -> 0.61.

    The resulting alpha_s rises from 0.02 at 100 Hz to a 0.69 plateau near
    1600 Hz and falls back to 0.34 at 5000 Hz, a broadband porous absorber.
    """
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
        dtype=float,
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
    result = ph.materials.measure_sound_absorption(
        freqs,
        t_empty,
        t_specimen,
        volume=200.0,
        area=10.8,
        temperature=20.0,
        humidity=54.0,
    )
    metadata = ReportMetadata(
        specimen="50 mm porous absorber over a 100 mm air gap",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Type A (mounted directly against a rigid wall)",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 354",
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-354",
    )
    return result, metadata, "iso354_absorption_example.pdf"


def _impedance_tube_example() -> tuple[object, ReportMetadata, str]:
    """ISO 10534-2 fiche: a two-microphone impedance-tube measurement.

    A documented clean-room example with a known closed-form absorption curve.
    The specimen is modelled as a locally-reacting resistive screen of
    normalised flow resistance theta = 1 backed by a rigidly-terminated air
    cavity of depth L, whose normalised surface impedance is the textbook
    z(f) = theta - j*cot(k0*L) (a resistive layer in series with the cavity
    reactance; Cox & D'Antonio, Acoustic Absorbers and Diffusers). From it the
    reflection factor r = (z - 1)/(z + 1) (ISO 10534-2 Eq. (19) inverted) and
    the absorption alpha = 1 - |r|^2 (Eq. (18)) follow exactly. The transfer
    function H12 that a tube would measure is synthesised from r via the
    Annex D field model (Eq. (D.7)) and fed back through
    ``two_microphone_impedance`` (Eq. (17)), so the fiche exercises the real
    reduction and its printed alpha matches the closed form.

    A 100 mm circular tube with s = 50 mm microphone spacing and the far mic
    at x1 = 100 mm works from f_l = c0/(20 s) ~ 343 Hz to the cut-on
    f_u = 0.58 c0/d ~ 1991 Hz at 20 degC (c0 = 343.29 m/s). The cavity depth
    L = c0/(4*1000 Hz) = 85.8 mm places the quarter-wave resonance at 1000 Hz,
    where the matched screen (theta = 1) gives z = 1, r = 0 and alpha = 1.00.
    Two further worked bands: at 500 Hz k0*L = pi/4, so cot = 1, z = 1 - j and
    alpha = 1 - |(-j)/(2 - j)|^2 = 1 - 1/5 = 0.80; at 1600 Hz the reactance is
    mass-like and alpha falls back to 0.68.
    """
    speed_of_sound_iso10534 = ph.materials.speed_of_sound_iso10534
    air_density_iso10534 = ph.materials.air_density_iso10534

    temperature_c = 20.0
    pressure_kpa = 101.0
    c0 = float(speed_of_sound_iso10534(temperature_c=temperature_c))
    rho = float(
        air_density_iso10534(
            temperature_c=temperature_c, atmospheric_pressure_kpa=pressure_kpa
        )
    )
    rc = ph.fluids.characteristic_impedance(rho, c0)

    diameter, spacing, x1 = 0.100, 0.050, 0.100
    theta, cavity = 1.0, c0 / (4.0 * 1000.0)
    freqs = np.array([400, 500, 630, 800, 1000, 1250, 1600], dtype=float)

    k0 = 2.0 * np.pi * freqs / c0
    z = theta - 1j / np.tan(k0 * cavity)
    r = (z - 1.0) / (z + 1.0)
    # Synthesise H12 from the known r (ISO 10534-2 Annex D, Eq. (D.7)).
    kk = np.asarray(ph.materials.tube_wavenumber(freqs, c0))
    x2 = x1 - spacing
    h12 = (np.exp(1j * kk * x2) + r * np.exp(-1j * kk * x2)) / (
        np.exp(1j * kk * x1) + r * np.exp(-1j * kk * x1)
    )
    result = ph.materials.two_microphone_impedance(
        h12,
        frequency=freqs,
        spacing=spacing,
        x1=x1,
        speed_of_sound=c0,
        characteristic_impedance=rc,
        diameter=diameter,
        shape="circular",
    )
    metadata = ReportMetadata(
        specimen="Resistive facing over an 86 mm rigidly-backed air cavity",
        client="Example client",
        manufacturer="Example acoustics",
        tube_diameter=diameter,
        mic_spacing=spacing,
        mounting="Deliberate 86 mm backing air cavity, rigid termination",
        test_room="Impedance tube B&K 4206 (example)",
        measurement_standard="ISO 10534-2",
        temperature=20.0,
        pressure=pressure_kpa,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-10534-2",
    )
    return result, metadata, "iso10534_impedance_tube_example.pdf"


#: One-third-octave centre frequencies of ISO 17497 Table 1 / Clause 5, in Hz
#: (100 Hz to 5000 Hz, full scale).
_SCATTER_FREQS = np.array(
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
    dtype=float,
)


def _scattering_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-1 fiche: a random-incidence scattering-coefficient measurement.

    A documented clean-room example (ISO 17497-1 has no numeric worked example,
    so an end-to-end synthetic case is built from Eqs. (1)/(4)/(5)/(6), as the
    standard itself directs). A reverberation room of volume V = 200 m3 holds a
    circular test sample of area S = 10 m2 on a turntable; at 20 degC the speed
    of sound is c = 343.2 m/s (Eq. (2)) and the air attenuation is neglected
    (m = 0, the zero-attenuation reference). The four reverberation times of
    Table 2 are chosen with a perfectly symmetrical base plate (T1 = T3, so the
    base-plate scattering of Eq. (6) is exactly 0 and passes the Table 1 limits)
    and a rotating turntable whose apparent (specular) absorption grows with
    frequency as the surface relief scatters more energy out of the specular
    direction. The random-incidence absorption alpha_s (Eq. (1)) stays below the
    0.50 ceiling of Clause 6.3.4. Two worked bands, from
    s = (alpha_spec - alpha_s) / (1 - alpha_s) (Eq. (5)):

    * 500 Hz: alpha_s = 0.053, alpha_spec = 0.131, s = 0.082 -> 0.08.
    * 4000 Hz: alpha_s = 0.112, alpha_spec = 0.515, s = 0.454 -> 0.45.

    The scattering coefficient rises from 0.01 at 100 Hz to 0.55 at 5000 Hz, a
    broadband diffusing surface.
    """
    volume, area, c = 200.0, 10.0, 343.2
    t1 = np.array(
        [
            8.0,
            7.9,
            7.8,
            7.6,
            7.4,
            7.2,
            7.0,
            6.7,
            6.4,
            6.0,
            5.6,
            5.2,
            4.8,
            4.4,
            4.0,
            3.6,
            3.2,
            2.9,
        ]
    )
    t3 = t1.copy()  # symmetrical base plate: T1 = T3
    t2 = t1 * 0.90  # sample, static turntable
    t4 = t2 * (1.0 - np.linspace(0.02, 0.28, _SCATTER_FREQS.size))
    alpha_s = ph.materials.random_incidence_absorption(
        volume, area, c1=c, t1=t1, c2=c, t2=t2
    )
    alpha_spec = ph.materials.specular_absorption_coefficient(
        volume, area, c3=c, t3=t3, c4=c, t4=t4
    )
    result = ph.materials.scattering_coefficient_spectrum(
        _SCATTER_FREQS, alpha_spec, alpha_s
    )
    metadata = ReportMetadata(
        specimen="1:1 quadratic-residue diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        area=area,
        room_volume=volume,
        mounting="Circular sample on the rotating turntable, centre displaced d/8",
        test_room="Reverberation room (example)",
        measurement_standard="ISO 17497-1",
        temperature=20.0,
        relative_humidity=54.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-1",
    )
    return result, metadata, "iso17497_scattering_example.pdf"


#: The 2-D single-plane source positions of ISO 17497-2 Clause 6.2.2 (0 deg and
#: +/-30 deg, +/-60 deg about the reference normal); paired with the Clause 8.4
#: source weights (0 deg -> 1, the four others -> 3).
_DIFFUSION_SOURCES = np.array([0.0, 30.0, -30.0, 60.0, -60.0])


def _diffuser_polar_energy(
    angles: np.ndarray, width: float, peak: float, specular: float = 0.0
) -> np.ndarray:
    """A synthetic reflected-level polar response (a specular lobe over a floor).

    The band energy is a diffuse floor of unity plus a specular lobe of linear
    amplitude ``peak`` and Gaussian half-width ``width`` (degrees) centred on the
    ``specular`` reflection angle; the level is ``10 lg(energy) + 60`` dB.
    """
    energy = 1.0 + peak * np.exp(-(((angles - specular) / width) ** 2))
    return 10.0 * np.log10(energy) + 60.0


def _diffusion_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-2 fiche: a random-incidence diffusion-coefficient spectrum d(f).

    A documented clean-room example (ISO 17497-2 has no numeric worked example
    or reference polar dataset, so the polar responses are synthesised and the
    coefficient computed from Formula (5), as the standard directs). A
    single-plane goniometer sweeps 19 equal-area receivers from -90 to 90 deg
    (10 deg spacing) about the reference normal for each of the five 2-D source
    positions of Clause 6.2.2 (0 deg and +/-30 deg, +/-60 deg), whose specular
    reflection falls at the mirror angle. As frequency rises the diffuser spreads
    the reflected energy ever more evenly (the specular lobe broadens and
    flattens), so the directional coefficient d_theta (Formula (5)) of each
    source climbs with frequency. The per-band **random-incidence** coefficient
    d (Clause 8.4) is the weighted average of the five directional coefficients
    over the source positions (0 deg -> 1, the four others -> 3), computed band
    by band, and the normalised d_n (Formula (7), against a rigid flat reference
    of the same footprint) is likewise averaged over the sources. Both climb
    with frequency: d from 0.23 at 100 Hz to 0.86 at 5000 Hz. Two worked bands:
    at 500 Hz d = 0.51 (d_n = 0.35); at 4000 Hz d = 0.81 (d_n = 0.68).
    """
    angles = np.arange(-90.0, 90.5, 10.0)
    n = _SCATTER_FREQS.size
    widths = np.linspace(15.0, 70.0, n)
    peaks = np.linspace(30.0, 3.0, n)
    weights = np.array(ph.materials.TWO_DIMENSIONAL_SOURCE_WEIGHTS, dtype=float)
    d = np.empty(n)
    d_n = np.empty(n)
    for k in range(n):
        d_theta = []
        d_theta_n = []
        for source in _DIFFUSION_SOURCES:
            specular = -source  # specular reflection about the reference normal
            d_s = ph.materials.directional_diffusion_coefficient(
                _diffuser_polar_energy(angles, widths[k], peaks[k], specular)
            )
            d_ref = ph.materials.directional_diffusion_coefficient(
                _diffuser_polar_energy(angles, 0.5 * widths[k], 60.0, specular)
            )
            d_theta.append(d_s)
            d_theta_n.append(
                float(ph.materials.normalized_diffusion_coefficient(d_s, d_ref))
            )
        # Clause 8.4: average the directional coefficients over the source
        # positions, band by band, to get the random-incidence coefficient.
        d[k] = ph.materials.random_incidence_diffusion(d_theta, weights=weights)
        d_n[k] = ph.materials.random_incidence_diffusion(d_theta_n, weights=weights)
    result = ph.materials.diffusion_spectrum(_SCATTER_FREQS, d, normalized=d_n)
    metadata = ReportMetadata(
        specimen="1:1 single-plane Schroeder diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Single-plane diffuser, plane of maximum diffusion",
        test_room="Anechoic goniometer (example), source at 10 m, arc at 5 m",
        measurement_standard="ISO 17497-2",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-2",
    )
    return result, metadata, "iso17497_diffusion_example.pdf"


def _diffusion_polar_example() -> tuple[object, ReportMetadata, str]:
    """ISO 17497-2 fiche: the single-source polar response of one band.

    The corrected 1000 Hz polar response behind the ``_diffusion_example``
    spectrum (Clause 8.5): 19 equal-area receivers from -90 to 90 deg, whose
    autocorrelation diffusion coefficient d = 0.67 (Formula (5)) for the
    normal-incidence source position.
    """
    angles = np.arange(-90.0, 90.5, 10.0)
    widths = np.linspace(15.0, 70.0, _SCATTER_FREQS.size)
    peaks = np.linspace(30.0, 3.0, _SCATTER_FREQS.size)
    band = int(np.argmin(np.abs(_SCATTER_FREQS - 1000.0)))
    levels = _diffuser_polar_energy(angles, widths[band], peaks[band])
    result = ph.materials.directional_diffusion(angles, levels)
    metadata = ReportMetadata(
        specimen="1:1 single-plane Schroeder diffuser (N = 7)",
        client="Example client",
        manufacturer="Example acoustics",
        mounting="Single-plane diffuser, normal-incidence source (0 deg)",
        test_room="Anechoic goniometer (example), source at 10 m, arc at 5 m",
        measurement_standard="ISO 17497-2",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-17497-2P",
    )
    return result, metadata, "iso17497_diffusion_polar_example.pdf"


def _dynamic_stiffness_example() -> tuple[object, ReportMetadata, str]:
    """EN 29052-1 fiche: the dynamic stiffness of a resilient floating-floor layer.

    A 20 mm mineral-wool resilient layer under the standard 8 kg load plate on
    the 0.04 m2 specimen (total mass per unit area m't = 8 kg / 0.04 m2 =
    200 kg/m2, EN 29052-1:1992 Clauses 5 and 6), whose fundamental resonance is
    measured at fr = 45.0 Hz. Formula 4 gives the apparent dynamic stiffness
    s't = 4*pi^2 * m't * fr^2 = 15.99 MN/m3 -> 16 MN/m3 (Clause 9 rounds to the
    nearest MN/m3). At the intermediate lateral airflow resistivity of
    r = 50 kPa.s/m2 the enclosed-gas term applies (Clause 8.2 b): s'a = 111/d =
    5.56 MN/m3 -> 6 MN/m3 (Clause 8.2 NOTE, d = 20 mm), so the installed
    stiffness is s' = s't + s'a = 21.54 MN/m3 -> 22 MN/m3 (Formula 6). Installed
    under a 110 kg/m2 floating screed the natural frequency is
    f0 = (1/2pi) sqrt(s'/m') = 70.4 Hz (Formula 2).
    """
    result = ph.materials.floating_floor_resonance(
        resonant_frequency=45.0,
        total_mass_per_area=200.0,
        floor_mass_per_area=110.0,
        airflow_resistivity=50.0,
        thickness=0.020,
        porosity=0.9,
    )
    metadata = ReportMetadata(
        specimen="20 mm mineral-wool resilient layer",
        client="Example client",
        manufacturer="Example insulation works",
        mass_per_area=200.0,
        thickness=0.020,
        test_room="Dynamic-stiffness rig (example), 8 kg load plate",
        measurement_standard="EN 29052-1",
        temperature=21.0,
        relative_humidity=50.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-29052-1",
    )
    return result, metadata, "en29052_dynamic_stiffness_example.pdf"


def _airflow_resistance_example() -> tuple[object, ReportMetadata, str]:
    """ISO 9053-1 fiche: the static airflow resistance of a porous specimen.

    A 50 mm porous absorber measured in a 100 mm diameter cell (cross-section
    A = pi*0.05^2 = 7.854e-3 m2, ISO 9053-1:2018 clause 7). The linear airflow
    velocity is stepped up to 12 mm/s (below the 15 mm/s clause-7.5 limit) and
    the measured pressure difference fitted through the origin with a
    second-order regression dp = a*u + b*u^2 (clause 7.5), here a = 16000 Pa*s/m
    and b = 400000 Pa*s^2/m^2. Read at the reference velocity u = 0.5 mm/s this
    gives R_s = a + b*u = 16200 Pa*s/m, an airflow resistance
    R = R_s/A = 2.06e6 Pa*s/m^3 and, for the 50 mm thickness, an airflow
    resistivity sigma = R_s/d = 324000 Pa*s/m^2.
    """
    area = np.pi * 0.05**2
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3
    dp = 1.6e4 * u + 4.0e5 * u**2
    result = ph.materials.static_airflow_resistance(u, dp, area=area, thickness=0.05)
    metadata = ReportMetadata(
        specimen="50 mm porous absorber (open-cell)",
        client="Example client",
        manufacturer="Example insulation works",
        thickness=0.050,
        test_room="Static airflow rig (example), 100 mm cell",
        measurement_standard="ISO 9053-1",
        temperature=23.0,
        relative_humidity=50.0,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9053-1",
    )
    return result, metadata, "iso9053_airflow_resistance_example.pdf"
