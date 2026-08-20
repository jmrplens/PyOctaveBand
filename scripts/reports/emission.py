#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Fiches for what a source emits: sound power, declared and certificated.

The emission of a machine, by every route the standards give to it: the
perceived noise level an aircraft is certificated on (ICAO Annex 16), the
dual-number declaration of ISO 4871, and the sound power determined from
pressure over a measurement surface (ISO 3744), in a reverberation room
(ISO 3741), by an intensity scan (ISO 9614-2, and its precision sibling
ISO 9614-3), from surface vibration (ISO/TS 7849-2) and in an anechoic room
(ISO 3745).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import phonometry as ph
from phonometry import ReportMetadata


def _epnl_example() -> tuple[object, ReportMetadata, str]:
    """EPNL fiche: an ICAO Annex 16 aircraft-noise-certification result.

    A deterministic synthetic half-second flyover: a broadband spectral shape
    peaking near 400 Hz swept by a Gaussian temporal gain (the aircraft
    approaching and receding) with a 2500 Hz fan tone that adds the tone
    correction, giving a valid ``EPNLResult`` with a peak PNLTM near mid-record
    and a clear 10 dB-down window. The requirement is a plausible certification
    EPNL limit the example passes.
    """
    k, dt = 24, 0.5
    idx = np.arange(k)
    shape = 15.0 * np.exp(
        -((np.log10(ph.aircraft.NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5
    )
    gain = 24.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 3.5**2)) - 3.0
    spectra = (46.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 10.0 * np.exp(-((idx - 12.0) ** 2) / (2 * 4.0**2))  # 2500 Hz tone
    result = ph.aircraft.effective_perceived_noise_level(spectra, dt)
    metadata = ReportMetadata(
        specimen="Example twin-turbofan transport (synthetic flyover)",
        manufacturer="Example Aircraft Company",
        client="Example Aircraft Company",
        test_room="Flyover reference point",
        measurement_standard="ICAO Annex 16 Vol I Amendment 14 Chapter 4",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-EPNL",
        requirement=101.0,
    )
    return result, metadata, "icao_epnl_example.pdf"


def _iso4871_declaration_example() -> tuple[object, ReportMetadata, str]:
    """ISO 4871 fiche: a dual-number machinery noise-emission declaration.

    Reproduces the ISO 4871:1996 Annex B.2 example (Type 990, Model 11-TC): two
    operating modes with a measured A-weighted sound power level and an
    uncertainty of 2 dB, stated separately per the dual-number layout, plus
    emission sound pressure levels at the work station. A verification
    measurement is added per mode against the dual-number limit L_WA + K_WA
    (clause 6.2): mode 1 passes (89 <= 90) and mode 2 fails (98 > 97),
    exercising the verdict both ways.
    """
    mode1 = ph.emission.OperatingModeDeclaration(
        mode="Operating mode 1",
        sound_power_level=88.0,
        sound_power_uncertainty=2.0,
        emission_pressure_level=78.0,
        emission_pressure_uncertainty=2.0,
        verification_level=89.0,
    )
    mode2 = ph.emission.OperatingModeDeclaration(
        mode="Operating mode 2",
        sound_power_level=95.0,
        sound_power_uncertainty=2.0,
        emission_pressure_level=86.0,
        emission_pressure_uncertainty=2.0,
        verification_level=98.0,
    )
    result = ph.emission.NoiseEmissionDeclaration(
        modes=(mode1, mode2),
        machine="Type 990, Model 11-TC",
        operating_conditions="50 Hz, 230 V, rated load",
        noise_test_code="ISO 3746 test code (example)",
        basic_standards=("ISO 3744", "ISO 11202"),
        form="dual-number",
    )
    metadata = ReportMetadata(
        measurement_standard="ISO 3744",
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-4871",
        notes="Declaration reproduces the ISO 4871:1996 Annex B example machine.",
    )
    return result, metadata, "iso4871_declaration_example.pdf"


def _sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power fiche: an ISO 3744 engineering-grade determination.

    A floor-standing machine on one reflecting plane, measured on a hemisphere
    of radius r = 4 m (surface area S = 2*pi*r^2 = 100.53 m^2, ISO 3744:2010
    clause 7.2.3) at the ten key microphone positions (clause 8.1.1). The
    energy-averaged surface pressure levels (Eq. 12) per octave band (63 Hz to
    8 kHz) are 72, 76, 80, 82, 81, 78, 73 and 66 dB, each with the background
    noise a uniform 10 dB below, so the background correction is a uniform
    K1 = -10*lg(1 - 10^(-1,0)) = 0.46 dB (Eq. 16), and an equivalent absorption
    area A = 1500 m^2 gives K2 = 10*lg(1 + 4*S/A) = 1.03 dB (Eq. A.2), inside
    the 4 dB engineering validity limit. The surface level (Eq. 17) is then
    LW = Lp + 10*lg(S/S0) with 10*lg(S/S0) = 20.02 dB (Eq. 18), giving the band
    levels 90.5, 94.5, 98.5, 100.5, 99.5, 96.5, 91.5 and 84.5 dB and, with the
    Annex E octave A-weighting corrections (Table E.2, Eq. E.1), an A-weighted
    sound power level LWA = 103.7 dB(A) re 1 pW. The declared limit of
    105 dB(A) is met, so the verdict passes.
    """
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    surface_pressure = np.array([72.0, 76, 80, 82, 81, 78, 73, 66])
    # A uniform, well-behaved field: ten identical position spectra, so the
    # energy average (Eq. 12) equals the documented surface-pressure spectrum.
    positions = np.tile(surface_pressure, (10, 1))
    result = ph.emission.sound_power_pressure(
        positions,
        "hemisphere",
        radius=4.0,
        reflecting_planes=1,
        background_levels=np.tile(surface_pressure - 10.0, (10, 1)),
        frequencies=freqs,
        room=ph.emission.RoomEnvironment(absorption_area=1500.0),
        grade="engineering",
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Hemi-anechoic room over a reflecting floor",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-20",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3744",
        requirement=105.0,
        notes="Enveloping-surface pressure method over a hemisphere "
        "(ISO 3744:2010, engineering grade 2).",
    )
    return result, metadata, "iso3744_sound_power_example.pdf"


def _intensity_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power-by-intensity fiche: an ISO 9614-2 engineering-grade scan.

    A machine enclosed by a hypothetical box divided into six equal segments of
    area Si = 0.5 m2 (measurement surface S = 3.0 m2). The probe is swept twice
    over each segment; the field is uniform, so every segment reports the same
    signed normal intensity per octave band (125 Hz to 4 kHz),
    In = [0.6, 1.0, 1.5, 1.4, 0.9, 0.5] x 1e-4 W/m2. The partial powers
    Pi = <In,i>*Si sum to the band power P = In*S (Eq. 12/6), so the band
    sound-power level is LW = 10*lg(In*S/P0), P0 = 1 pW (Eq. 13): 82.5, 84.8,
    86.5, 86.2, 84.3 and 81.8 dB. With the octave A-weighting corrections
    (-16.1, -8.6, -3.2, 0.0, 1.2, 1.0 dB) this gives LWA = 90.9 dB(A) re 1 pW.
    The two sweeps are identical (perfect repeatability), the surface SPL is a
    uniform 80 dB and the instrument pressure-residual index is delta_pI0 =
    15 dB, so the dynamic capability Ld = 15 - 10 = 5 dB clears FpI in every
    band and all six bands qualify at engineering grade. The declared limit of
    93 dB(A) is met, so the verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    intensity = np.array([0.6e-4, 1.0e-4, 1.5e-4, 1.4e-4, 0.9e-4, 0.5e-4])
    areas = np.full(6, 0.5)
    # A uniform field: the six segments share one intensity spectrum, and the
    # two sweeps coincide, so P = In*S and the repeatability is exact.
    scan = np.tile(intensity, (6, 1))
    result = ph.emission.sound_power_intensity(
        scan,
        areas,
        normal_intensity_2=scan.copy(),
        pressure_levels=np.full((6, 6), 80.0),
        pressure_residual_index=15.0,
        frequencies=freqs,
        band_type="octave",
        grade="engineering",
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Machine hall with steady background noise",
        instrumentation="Class 1 p-p intensity probe (IEC 61043), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-21",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9614",
        requirement=93.0,
        notes="Intensity-scanning method over a box surface "
        "(ISO 9614-2:1996, engineering grade 2).",
    )
    return result, metadata, "iso9614_sound_power_intensity_example.pdf"


def _precision_intensity_example() -> tuple[
    object, ReportMetadata, str, dict[str, Any]
]:
    """Precision sound-power-by-intensity fiche: an ISO 9614-3 scan.

    The same machine as the ISO 9614-2 example, measured the way part 3 asks:
    a box measurement surface 0,25 m off a 1,0 x 0,6 x 0,8 m floor-standing
    unit, so the five partial surfaces are the 1,5 x 1,1 m top (1,65 m2), the
    two 1,5 x 1,05 m long sides (1,575 m2 each) and the two 1,1 x 1,05 m ends
    (1,155 m2 each), a measurement surface S = 7,11 m2. Each partial surface is
    scanned in four segments, so the Annex B indicators are formed over N = 20
    segments of the whole surface.

    The band sound power levels are built from a closed form rather than from
    a worked example (part 3 gives none): the surface-averaged normal intensity
    of each band is the one that yields
    LW = 80 + 9*exp(-(lb(f/800))^2/3) dB, a machine peaking at 89,0 dB in the
    800 Hz band over the one-third-octave bands 100 Hz to 3150 Hz. The five
    partial surfaces share that spectrum scaled by [1,35 1,10 0,75 0,95 0,85]
    (renormalized to an area-weighted mean of 1, so the surface integral is
    unchanged) and the four segments of each by [1,15 0,95 0,90 1,00]; the
    field is therefore non-uniform, as FS = 0,24 records. The far long side
    sees a net inflow in the 100 Hz band (a neighbouring line radiating into
    the surface), which is what separates the signed indicator FpIn = 12,5 dB
    there from the unsigned Fp|In| = 11,0 dB and lifts FS to 1,1.

    The segment pressure levels sit a band-dependent 11,0 dB (100 Hz) down to
    1,5 dB (2 kHz and above) above the segment intensity levels, so the
    pressure-intensity indicators take those values directly. With the probe's
    pressure-residual intensity index delta_pI0 = 18 dB the dynamic capability
    is Ld = 18 - 10 = 8 dB, which clears FpIn in every band but the 100 Hz one:
    that band fails criterion 2, is not qualified, and is the band the fiche
    names as omitted from the A-weighted determination (clause 10 f) 2)). The
    two scans differ by 0,2 dB, inside the s/2 of Table 1 everywhere (criterion
    1); the temporal variability over ten averaging windows is FT = 0,04, well
    under the 0,6 of C.1.2; and halving the scan-line density (segments merged
    in pairs) leaves the ratio FS(1)/FS(2) within the 0,83 to 1,2 band of
    criterion 5, which constrains that ratio and not FS itself.

    At 28 degC and 94,0 kPa the meteorological normalization of Eq. 10 is
    -0,60 dB, so every normalized level LW0 sits 0,6 dB above its LW. The
    determination gives LWA = 96,7 dB(A) re 1 pW against a declared limit of
    98 dB(A), so the verdict passes.
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
        ],
        dtype=float,
    )
    # Pressure-intensity margin per band: a reverberant machine hall, worst at
    # the low bands where the probe's dynamic capability is also tightest.
    margin = np.array(
        [
            11.0,
            7.0,
            6.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            2.5,
            2.2,
            2.0,
            1.8,
            1.6,
            1.5,
            1.5,
            1.5,
        ]
    )
    areas = np.array([1.65, 1.575, 1.575, 1.155, 1.155])
    surface = float(np.sum(areas))
    target = 80.0 + 9.0 * np.exp(-(np.log2(freqs / 800.0) ** 2) / 3.0)
    mean_intensity = 10.0 ** (target / 10.0) * 1e-12 / surface
    face = np.array([1.35, 1.10, 0.75, 0.95, 0.85])
    face = face / (np.sum(face * areas) / surface)  # area-weighted mean of 1
    segment = np.array([1.15, 0.95, 0.90, 1.00])
    segment = segment / segment.mean()
    partial = mean_intensity[None, :] * face[:, None]
    partial[2, 0] *= -1.0  # net inflow on the far long side at 100 Hz
    segments = (partial[:, None, :] * segment[None, :, None]).reshape(-1, freqs.size)
    pressure_levels = 10.0 * np.log10(np.abs(segments) / 1e-12) + margin[None, :]
    windows = mean_intensity[None, :] * (
        1.0 + 0.05 * np.cos(2.0 * np.pi * np.arange(10)[:, None] / 10.0)
    )

    result = ph.emission.sound_power_intensity_precision(
        partial,
        areas,
        frequencies=freqs,
        temperature=28.0,
        barometric_pressure=94_000.0,
    )
    indicators = ph.emission.precision_field_indicators(
        segments, pressure_levels, time_window_intensity=windows
    )
    # The same scan at half the line density: neighbouring segments merged.
    coarse = segments.reshape(-1, 2, freqs.size).mean(axis=1)
    coarse_levels = 10.0 * np.log10(
        np.mean(10.0 ** (pressure_levels.reshape(-1, 2, freqs.size) / 10.0), axis=1)
    )
    initial = ph.emission.precision_field_indicators(coarse, coarse_levels)
    scan_level = 10.0 * np.log10(np.abs(np.mean(segments, axis=0)) / 1e-12)
    criteria = ph.emission.precision_qualification(
        indicators,
        scan_intensity_level_1=scan_level,
        scan_intensity_level_2=scan_level + 0.2,
        pressure_residual_index=18.0,
        field_nonuniformity_1=initial.fs,
        field_nonuniformity_2=indicators.fs,
        frequencies=freqs,
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Machine hall with steady background noise",
        instrumentation="Class 1 p-p intensity probe (IEC 61043), 12 mm spacer, s/n 0042",
        temperature=28.0,
        relative_humidity=40.0,
        pressure=94.0,
        test_date="2026-08-14",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-9614-3",
        requirement=98.0,
        notes="Box surface at 0,25 m; five partial surfaces, four segments "
        "each, scanned twice at 0,15 m/s.",
    )
    return (
        result,
        metadata,
        "iso9614_3_precision_intensity_example.pdf",
        {
            "indicators": indicators,
            "criteria": criteria,
            "residual_index": 18.0,
        },
    )


def _reverberation_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Reverberation-room fiche: an ISO 3741 precision-grade determination.

    A steady, broadband source measured by the direct method in a qualified
    hard-walled reverberation room of volume V = 200 m3 and total surface
    S = 240 m2 (ISO 3741:2010 clause 9.1.4). The octave-band mean room
    sound-pressure levels Lp (125 Hz to 8 kHz octave bands) peak near 500 Hz. At
    the test temperature theta = 20 degC the speed of sound is
    c = 20.05*sqrt(293) = 343.2 m/s and, with a uniform reverberation time
    T60 = 2.0 s, the Sabine equivalent absorption area
    A = (55.26/c)*(V/T60) = 16.10 m2 is constant across bands, so
    10*lg(A/A0) = 12.07 dB and 4.34*(A/S) = 0.29 dB are fixed; the Waterhouse
    boundary correction 10*lg(1 + S*c/(8*V*f)) falls from 1.50 dB at 125 Hz to
    0.03 dB at 8 kHz. At 101.325 kPa the meteorological corrections are
    C1 = 5*lg(293.15/314) = -0.15 dB and C2 = 15*lg(293.15/296) = -0.06 dB.
    Eq. (20) then gives the band levels, e.g. LW = 90.0 dB at 250 Hz, 91.6 dB
    at 500 Hz and 90.4 dB at 1 kHz, a total LW = 96.7 dB and, with the Annex F
    (Annex E) A-weighting corrections, an A-weighted sound power level
    LWA = 94.3 dB(A) re 1 pW. The declared limit of 96 dB(A) is met, so the
    verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    # Documented mean corrected room level Lp(ST) per band (Eq. 16), in dB.
    lp = np.array([80.0, 83.0, 85.0, 84.0, 80.0, 75.0, 68.0])
    result = ph.emission.sound_power_reverberation(
        lp,
        2.0,
        volume=200.0,
        surface_area=240.0,
        frequencies=freqs,
        temperature=20.0,
        static_pressure=101.325,
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Hydraulic power pack (floor-standing)",
        test_room="Qualified reverberation room, V = 200 m3, T60 = 2.0 s",
        instrumentation="Class 1 sound level meter (IEC 61672-1), s/n 0042",
        temperature=20.0,
        relative_humidity=50.0,
        pressure=101.325,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3741",
        requirement=96.0,
        notes="Reverberation-room direct method (ISO 3741:2010, precision grade 1).",
    )
    return result, metadata, "iso3741_reverberation_power_example.pdf"


def _vibration_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Sound-power-from-vibration fiche: an ISO/TS 7849-2 engineering example.

    A documented clean-room example (ISO/TS 7849 gives no numeric worked case
    beyond the calibration example, so the band spectrum is built from the
    closed-form Eq. 12/15, as the standard directs). A machine casing of
    radiating area S = 1.6 m2 (so 10*lg(S/S0) = 2.04 dB) is surveyed over six
    octave bands (125 Hz to 4 kHz); the surface-averaged vibratory velocity
    level (Eq. 3) is Lv = [78, 82, 85, 83, 79, 74] dB re 5e-8 m/s and the
    band-wise radiation factor determined from an independent power measurement
    (Eq. 8, ISO 9614) is epsilon = [0.20, 0.45, 0.75, 0.95, 1.00, 1.00]. The
    radiated band sound-power level is
    LW = Lv + 10*lg(S/S0) + 10*lg(epsilon) + 10*lg(411/400) with the fixed
    impedance term 10*lg(411/400) = 0.12 dB, giving LW = [73.2, 80.7, 85.9,
    84.9, 81.2, 76.2] dB re 1 pW, a total LW = 90.0 dB and, with the octave
    A-weighting corrections (-16.1, -8.6, -3.2, 0.0, 1.2, 1.0 dB), an A-weighted
    sound power level LWA = 88.7 dB(A) re 1 pW. The declared limit of 90 dB(A)
    is met, so the verdict passes.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    lv = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
    eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])
    result = ph.emission.sound_power_from_vibration(
        lv, area=1.6, radiation_factor=eps, frequencies=freqs
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Gearbox casing (steel panel)",
        test_room="Machine hall (source vibration survey)",
        instrumentation="Piezoelectric accelerometer (ISO 16063-21 calibration), s/n 0042",
        temperature=21.0,
        relative_humidity=45.0,
        pressure=101.1,
        test_date="2026-07-22",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-7849",
        requirement=90.0,
        notes="Sound power from surface vibration, engineering method "
        "(ISO/TS 7849-2:2009).",
    )
    return result, metadata, "iso7849_vibration_power_example.pdf"


def _precision_sound_power_example() -> tuple[object, ReportMetadata, str]:
    """Precision sound-power fiche: ISO 3745:2012 in an anechoic room.

    The guide's own worked example: a mid-frequency-peaked machine measured
    over the forty standardized hemisphere positions of Annex E at a radius of
    1 m (surface S = 2*pi*r^2 = 6,283 m2), with a base spectrum peaked near
    1 kHz and a 1 dB per-position spread from a seeded generator, so the fiche
    and the guide print the same numbers. The determination gives an
    A-weighted sound power level LWA = 89,3 dB(A) re 1 pW. The expanded
    uncertainty is the Clause 10.5 example: the method's own sigma_omc =
    2,0 dB at k = 2, over the ISO 3745 Table 1 reproducibility standard
    deviation of the band.
    """
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
    rng = np.random.default_rng(7)
    levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))
    result = ph.emission.sound_power_anechoic(
        levels,
        "hemisphere",
        radius=1.0,
        frequencies=freqs,
        sigma_omc=2.0,
    )
    metadata = ReportMetadata(
        client="Example manufacturing plant",
        specimen="Mid-frequency-peaked machine (guide example)",
        test_room="Qualified anechoic room, 40-position hemisphere array",
        measurement_standard="ISO 3745",
        test_date="2026-08-05",
        laboratory="Phonometry reference example",
        operator="phonometry",
        report_id="EXAMPLE-3745",
        notes=(
            "Precision grade: Annex E hemisphere array at r = 1 m, "
            "meteorological corrections at the 23 C / 101,325 kPa reference, "
            "expanded uncertainty from the Clause 10.5 example."
        ),
    )
    return result, metadata, "iso3745_precision_power_example.pdf"
