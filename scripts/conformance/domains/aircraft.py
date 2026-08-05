#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Aircraft and rotorcraft noise, from certification to airport contours.

The certification side: the ICAO Annex 16 effective perceived noise level
chain (noy conversion, tone correction, duration correction) and the IEC 61265
requirements on the equipment that measures it, with the SAE ARP 866B/5534
atmospheric absorption underneath.

The airport-contour side: the ECAC Doc 29 single-event chain - noise-power
-distance interpolation, the noise fraction of a finite segment, impedance and
directivity adjustments, event assembly - checked against the reference
workbook that ships with the document, and the ECAC Doc 32 rotorcraft chain
with the NORAH2 guidance values for ground plane, flow resistivity and
diffraction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import phonometry as ph

from ..registry import _ROOT, Outcome, numeric, register

_AIRCRAFT = "Aircraft noise (ICAO Annex 16 / IEC 61265)"


@register(
    _AIRCRAFT,
    "ECAC Doc 29 noise fraction (half path)",
    "Finite-segment correction ΔF for a perpendicular foot at the segment start, dB",
)
def _chk_ecac_noise_fraction() -> Outcome:
    # A half-infinite segment (Sp at the start) receives half the energy: −3.01 dB.
    got = float(ph.noise_fraction(0.0, 10_000.0, 100.0))
    return numeric(-10.0 * math.log10(2.0), got, 1e-3, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 single-event chain",
    "SEL of a long level flyover vs the infinite-path limit LE∞ + ΔI − Λ, dB",
)
def _chk_ecac_event_level() -> Outcome:
    # A long straight level segment through the CPA reduces to the infinite-path
    # baseline plus the geometry corrections (ΔF → 0, ΔV = 0 at Vref).
    npd_p = [8000.0, 12000.0]
    npd_d = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0]
    sel = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0], [102.0, 96.0, 90.0, 84.0, 78.0, 72.0, 66.0]]
    lmax = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0], [98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
    vref = 160.0 * 0.514444
    import numpy as _np

    xs = _np.linspace(-40000.0, 40000.0, 801)
    path = _np.column_stack([xs, _np.zeros_like(xs), _np.full_like(xs, 300.0),
                             _np.full_like(xs, 10000.0), _np.full_like(xs, vref)])
    got = ph.event_level(path, [0.0, 300.0, 0.0], npd_p, npd_d, sel, lmax, metric="exposure").level
    dp = math.hypot(300.0, 300.0)
    beta = math.degrees(math.acos(300.0 / dp))
    expected = (float(ph.npd_level(npd_p, npd_d, sel, 10000.0, dp)[0])
                + ph.impedance_adjustment()
                + ph.engine_installation_correction(beta, "wing")
                - ph.lateral_attenuation(beta, 300.0))
    return numeric(expected, float(got), 1e-2, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 impedance adjustment (standard atmosphere)",
    "Acoustic-impedance adjustment of NPD data at 15 °C / 101.325 kPa (Eq. 4-6/4-7), dB",
)
def _chk_ecac_impedance() -> Outcome:
    # ECAC Doc 29 Vol 2 §4.2.1 states the standard-atmosphere adjustment is
    # +0.074 dB (ρc = 416.86, reference impedance 409.81 N·s/m³).
    return numeric(0.074, float(ph.impedance_adjustment()), 5e-4, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 reference workbook (segment Λ)",
    "Lateral attenuation of a climbing segment vs the ECAC Vol 3 Part 1 workbook, dB",
)
def _chk_ecac_workbook_segment() -> Outcome:
    # ECAC Doc 29 5th ed. Vol 3 Part 1 reference workbook, sheet
    # B-2_Segment_Results, case JETFAC receptor R02 segment 1 (climbing): the
    # workbook lists β = 4.2226°, lateral displacement 81363.28 ft (24799.6 m)
    # and Λ = 6.3769 dB. Here we anchor the Λ(β, ℓ) formula to those reference
    # values (the β/φ geometry itself is validated in tests/test_airport_noise).
    beta_ref, lateral_m = 4.2225708673, 81363.2829 * 0.3048
    got = float(ph.lateral_attenuation(beta_ref, lateral_m))
    return numeric(6.3768594165, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 start-of-roll directivity (jet)",
    "ΔSOR behind a takeoff ground-roll segment vs the Vol 3 Part 1 workbook, dB",
)
def _chk_ecac_start_of_roll() -> Outcome:
    # ECAC Doc 29 5th ed. Vol 3 Part 1 workbook, case JETFDC: at ψ = 112.8895°,
    # dSOR = 217.09 m the turbofan directivity (Eq. 4-24a) is +0.3196 dB.
    got = float(ph.start_of_roll_directivity(112.889545, 217.0934, "jet"))
    return numeric(0.31961, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 start-of-roll directivity (turboprop)",
    "ΔSOR behind a takeoff ground-roll segment (turboprop, Eq. 4-24b), dB",
)
def _chk_ecac_start_of_roll_prop() -> Outcome:
    # Same workbook, case PROPDC: at ψ = 128.1824°, dSOR = 254.44 m the turboprop
    # directivity (Eq. 4-24b) is +1.0943 dB.
    got = float(ph.start_of_roll_directivity(128.182381, 254.4361, "turboprop"))
    return numeric(1.09434, got, 1e-2, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 workbook event assembly (JETFDS/R03, behind SOR)",
    "Energy sum of the reference per-segment SELs vs the B-1 event total, dB",
)
def _chk_doc29_event_assembly() -> Outcome:
    # Doc 29 5th ed. Vol 3 Part 1 workbook, departure case JETFDS receptor R03
    # (centreline behind the start of roll): the Eq. 4-11 energy sum of the 29
    # reference segment SELs must reproduce the B-1 total 74.73 dB. No oracle
    # exists for per-event LAmax, non-zero bank angles or the Annex 16
    # bandsharing adjustment (no ETM worked example); registered gaps.
    import sys as _sys

    tests_dir = str(_ROOT / "tests" / "aircraft")
    if tests_dir not in _sys.path:
        _sys.path.insert(0, tests_dir)
    from doc29_workbook_data import B1, SEGMENTS

    rows = SEGMENTS[("JETFDS", "R03")]
    total = 10.0 * math.log10(sum(10.0 ** (r[-1] / 10.0) for r in rows))
    return numeric(B1[("JETFDS", "R03")], total, 1e-2, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "SAE ARP 5534 band-attenuation continuity",
    "SAE-Method δ_B at the 150 dB branch split (Eq. 7 vs Eq. 8), dB",
)
def _chk_arp5534_continuity() -> Outcome:
    # The two SAE-Method branches meet at δ_t = 150 dB (Eqs. 7-8).
    a, b, c, d, e = 0.867942, 0.111761, 0.95824, 0.008191, 1.6
    eq7 = a * 150.0 * (1.0 + b * (c - d * 150.0)) ** e
    eq8 = 9.2 + 0.765 * 150.0
    return numeric(eq8, eq7, 0.01, unit="dB", places=3)


@register(
    _AIRCRAFT,
    "EASA ANP database round-trip",
    "Interpolated NPD level at a tabulated node vs the published ANP value, dB",
)
def _chk_anp_round_trip() -> Outcome:
    # Independent oracle: the EASA ANP database's own published NPD values
    # (curated subset shipped under aircraft/data/anp). At a tabulated
    # (power, distance) node the loader/interpolation must recover the
    # published value exactly. Boeing 747-100 (JT9DBD), departure SEL,
    # 28000 lb corrected net thrust, 2000 ft slant distance = 98.8 dB.
    curves = ph.load_anp_database().npd_curves("747100", "departure", "SEL")
    got = float(curves.level(28000.0, 2000.0 * 0.3048)[0])
    return numeric(98.8, got, 1e-9, unit="dB", places=4)


@register(
    _AIRCRAFT,
    "ECAC Doc 29 NPD interpolation",
    "Log-linear NPD level at the log-midpoint distance (Eq. 4-4), dB",
)
def _chk_ecac_npd() -> Outcome:
    # Log-midpoint distance -> arithmetic mean of the bracketing node levels.
    p, d = [1000.0, 2000.0], [200.0, 400.0, 800.0, 1600.0]
    lv = [[100.0, 94.0, 88.0, 82.0], [110.0, 104.0, 98.0, 92.0]]
    got = float(ph.npd_level(p, d, lv, 1000.0, math.sqrt(200.0 * 400.0))[0])
    return numeric(97.0, got, 1e-9, unit="dB", places=4)


_ROTORCRAFT = "Rotorcraft noise (ECAC Doc 32 / NORAH2)"


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 atmospheric attenuation (Table 4)",
    "ΔLa over a 1 km excess path at 1 kHz vs the NORAH2 guidance Table 4, dB",
)
def _chk_doc32_atmospheric() -> Outcome:
    # NORAH2 guidance Table 4: 6.3 dB/km at 1 kHz (ICAO reference conditions).
    got = -float(ph.atmospheric_adjustment([1000.0], 1000.0 + 60.0)[0])
    return numeric(6.3, got, 0.2, unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 spherical spreading",
    "ΔLs at ten times the 60 m hemisphere reference distance (Eq. 24), dB",
)
def _chk_doc32_spreading() -> Outcome:
    return numeric(-20.0, float(ph.spherical_spreading_adjustment(600.0)), 1e-9,
                   unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 ground effect (rigid limit)",
    "ΔLg over a rigid surface at grazing incidence tends to +6 dB (Eq. 29), dB",
)
def _chk_doc32_ground() -> Outcome:
    # Over a near-rigid surface (Zs -> inf), coherent reflection at a small path
    # difference reinforces the direct ray towards the +6.02 dB pressure-doubling
    # limit. A low frequency and near-grazing geometry approaches it.
    got = float(ph.ground_effect_adjustment([20.0], 50.0, 1.2, 400.0, flow_resistivity="H")[0])
    return numeric(6.0, got, 1.0, unit="dB", places=2)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 propagation chain (NORAH2 prototype)",
    "LA of a single-hemisphere emission vs the NORAH2 prototype single-event"
    " history (R22 approach, 223.66 m slant), dB(A)",
)
def _chk_doc32_chain() -> Outcome:
    # NORAH2 prototype ARP Case 4 (R22_H1_APP_STD2_NE, mic at the origin,
    # hr = 0.2 m, sigma = 1e6 Pa·s/m2), row t = 831.26 s: the nearest-hemisphere
    # source spectrum propagated with dLs + dLa + dLg and A-weighted reproduces
    # the tabulated LA = 55.87 dB(A). Spectrum: R22_Approach_53kts_12deg at
    # (phi, theta) = (-88.70, 108.43) deg (EASA.2020.FC.06, (c) EASA).
    bands = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0,
                      80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
                      500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
                      2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0])
    spec = np.array([35.8, 52.2, 70.0, 63.6, 48.4, 59.2, 53.1, 63.3, 61.2,
                     74.6, 68.7, 61.8, 65.7, 59.7, 59.7, 63.6, 57.9, 57.7,
                     58.6, 61.0, 61.9, 64.7, 65.7, 65.6, 64.1, 61.6, 57.9,
                     55.8, 56.4, 53.4, 52.9])
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217

    def _ra(x: np.ndarray) -> np.ndarray:
        return (f4**2 * x**4) / ((x**2 + f1**2)
                                 * np.sqrt((x**2 + f2**2) * (x**2 + f3**2))
                                 * (x**2 + f4**2))

    level = (spec + float(ph.spherical_spreading_adjustment(223.66))
             + ph.atmospheric_adjustment(bands, 223.66)
             + ph.ground_effect_adjustment(bands, 5.0, 0.2, 223.607,
                                           flow_resistivity=1.0e6)
             + 20.0 * np.log10(_ra(bands) / _ra(np.array(1000.0))))
    got = float(10.0 * np.log10(np.sum(10.0 ** (level / 10.0))))
    return numeric(55.87, got, 0.1, unit="dB(A)", places=3)


def _uniform_hemisphere(level: float, bands: list[float] | None = None) -> Any:
    """A synthetic hemisphere with one uniform level on the standard 10° grid."""
    freqs = np.asarray(bands if bands is not None else [50.0], dtype=np.float64)
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    return ph.RotorcraftHemisphere(
        freqs, az, po, np.full((az.size, po.size, freqs.size), level))


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 flight-condition interpolation (NORAH2 Eq. 8)",
    "Distance-scaled triangle blend of three uniform hemispheres, hand-checked, dB",
)
def _chk_doc32_flight_condition() -> Outcome:
    # Conditions (V, gamma) = (50, 0), (70, 0), (60, 10); query (60, 2.5).
    # Normalised (Eq. 3-6, Ffc = 2, spans 20 kt / 10 deg): points (2.5, 0),
    # (3.5, 0), (3.0, 2), query (3.0, 0.5) -> deltas sqrt(0.5), sqrt(0.5), 1.5
    # (Eq. 7) -> weights 0.404629, 0.404629, 0.190743 (Eq. 8). Uniform levels
    # 100 / 90 / 95 dB blend to 10*lg(sum w*10^(L/10)) = 97.0367 dB by hand.
    hems = [_uniform_hemisphere(100.0), _uniform_hemisphere(90.0),
            _uniform_hemisphere(95.0)]
    got = float(ph.interpolated_source_level(
        hems, [50.0, 70.0, 60.0], [0.0, 0.0, 10.0], 60.0, 2.5, 0.0, 90.0)[0])
    return numeric(97.0367, got, 1e-3, unit="dB", places=4)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 flight-path kinematics (Eq. 17)",
    "Airspeed of a straight climbing track, 40 m/s ground speed at a 5° path angle, m/s",
)
def _chk_doc32_kinematics() -> Outcome:
    # Constant-velocity track: Vg = 40 m/s at heading 30 deg, path angle 5 deg
    # -> VA = 40/cos(5 deg) = 40.152786 m/s (Eq. 16/17), by hand.
    t = np.arange(0.0, 10.5, 0.5)
    vg, gamma = 40.0, np.radians(5.0)
    heading = np.radians(30.0)
    pos = np.column_stack([vg * np.sin(heading) * t, vg * np.cos(heading) * t,
                           100.0 + vg * np.tan(gamma) * t])
    kin = ph.flight_path_kinematics(t, pos)
    return numeric(40.152786, float(kin.airspeed[10]), 1e-4, unit="m/s", places=5)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 retarded time (Eq. 22)",
    "Recorded-time delay at 100 m slant distance, r/c with c = 346.1 m/s, s",
)
def _chk_doc32_retarded_time() -> Outcome:
    # Level flyover directly over the receiver: at the closest-approach step the
    # slant distance is 101.2 - (0 + 1.2) = 100.0 m and t_r - t_e = 100/346.1
    # = 0.288934 s, by hand.
    t = np.arange(0.0, 20.5, 0.5)
    pos = np.column_stack([np.zeros_like(t), 50.0 * (t - 10.0),
                           np.full_like(t, 101.2)])
    res = ph.rotorcraft_event_level(
        [_uniform_hemisphere(100.0)], [50.0], [0.0], t, pos, (0.0, 0.0),
        receiver_height=1.2, flow_resistivity="H")
    k = int(np.argmin(res.distance))
    return numeric(0.288934, float(res.times[k] - res.emission_times[k]), 1e-5,
                   unit="s", places=6)


@register(
    _ROTORCRAFT,
    "ECAC Doc 32 single event (Eq. 27)",
    "SEL − LASmax of a constant-speed level flyover, 10·lg(π·d/V) closed form, dB",
)
def _chk_doc32_event_sel() -> Outcome:
    # Single 31.5 Hz band over class-H ground with a 0.1 m receiver: dLg stays
    # within 0.03 dB of the +6 dB pressure doubling along the whole path, so the
    # exposure integral is the Lorentzian closed form SEL - LASmax =
    # 10·lg(pi*d/V) = 7.982 dB for d = 100 m and V = 50 m/s (truncation and
    # absorption stay below 0.1 dB with the track spanning +-6 km).
    t = np.arange(0.0, 240.1, 0.5)
    pos = np.column_stack([np.zeros_like(t), 50.0 * (t - 120.0),
                           np.full_like(t, 100.1)])
    res = ph.rotorcraft_event_level(
        [_uniform_hemisphere(100.0, [31.5])], [50.0], [0.0], t, pos, (0.0, 0.0),
        receiver_height=0.1, flow_resistivity="H")
    return numeric(7.982, res.sel - res.la_max, 0.1, unit="dB", places=3)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance mean ground plane (Eq. 36-40)",
    "Intercept of the plane fitted to a symmetric 20 m roofline, hand-checked, m",
)
def _chk_doc32_mean_plane() -> Outcome:
    # Roof (0,0)-(100,20)-(200,0): by symmetry the continuous least-squares
    # line is horizontal at the mean height, area/span = 2000/200 = 10 m.
    res = ph.mean_ground_plane([0.0, 100.0, 200.0], [0.0, 20.0, 0.0])
    return numeric(10.0, res.intercept, 1e-6, unit="m", places=4)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance mean flow resistivity (Eq. 41)",
    "Log-average of equal 1e4 and 1e6 Pa·s/m2 halves, hand-checked, Pa·s/m2",
)
def _chk_doc32_mean_sigma() -> Outcome:
    # Equal lengths: sigma_bar = 10^((log 1e4 + log 1e6)/2) = 1e5 by hand.
    got = ph.mean_flow_resistivity([120.0, 120.0], [1.0e4, 1.0e6])
    return numeric(1.0e5, got, 1e-3, unit="Pa·s/m²", places=1)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance diffraction at grazing (Eq. 42)",
    "Pure diffraction with the edge on the line of sight, 10·lg 3, dB",
)
def _chk_doc32_diffraction_grazing() -> Outcome:
    got = float(ph.diffraction_attenuation([1000.0], 0.0, edge_height=10.0)[0])
    return numeric(4.7712, got, 1e-4, unit="dB", places=4)


@register(
    _ROTORCRAFT,
    "NORAH2 guidance screening path difference (§A.4.5)",
    "Rubber-band delta over a 40 m hill, hand-checked geometry, m",
)
def _chk_doc32_screening_delta() -> Outcome:
    # Source (0, 20), receiver (400, 1.2), single edge at (200, 40):
    # delta = SO + OR - SR = sqrt(200^2+20^2) + sqrt(200^2+38.8^2)
    #         - sqrt(400^2+18.8^2) = 4.28480 m by hand.
    res = ph.terrain_screening_adjustment(
        [500.0], (0.0, 20.0), (400.0, 1.2),
        [0.0, 190.0, 200.0, 210.0, 400.0], [0.0, 0.0, 40.0, 0.0, 0.0])
    expected = (np.hypot(200.0, 20.0) + np.hypot(200.0, 38.8)
                - np.hypot(400.0, 18.8))
    return numeric(float(expected), res.path_difference, 1e-9, unit="m", places=5)


@register(
    _AIRCRAFT,
    "SAE ARP 5534 pure-tone coefficient (ISO 9613-1)",
    "Mid-band α at 1 kHz, 25 °C, 70 % RH, 101.325 kPa, dB/m",
)
def _chk_arp5534_coefficient() -> Outcome:
    # ARP 5534 §3.1: the pure-tone coefficient is the ISO 9613-1 one.
    expected = float(ph.air_attenuation(1000.0, 25.0, 70.0, 101.325, exact_midband=True))
    res = ph.sae_band_attenuation([1000.0], 1000.0, temperature=25.0, relative_humidity=70.0)
    return numeric(expected, float(res.coefficient[0]), 1e-9, unit="dB/m", places=6)

# ICAO Doc 9501 ETM Vol. I (2018) Table 3-7 turbofan spectrum (bands 1-2 blank).
_ETM_SPL_37 = [
    -999.0, -999.0, 70.0, 62.0, 70.0, 80.0, 82.0, 83.0, 76.0, 80.0,
    80.0, 79.0, 78.0, 80.0, 78.0, 76.0, 79.0, 85.0, 79.0, 78.0, 71.0, 60.0, 54.0, 45.0,
]
# ICAO Doc 9501 ETM Vol. I (2018) Table 4-4 integrated-method EPNL example.
_ETM_PNLTR_44 = [
    84.62, 85.84, 85.37, 88.57, 88.82, 88.03, 88.76, 87.06, 86.92, 90.39, 89.89,
    91.00, 90.08, 89.71, 89.61, 90.21, 91.14, 92.10, 93.68, 94.89, 95.87, 97.06,
    97.40, 96.23, 94.73, 92.30, 88.75, 86.96, 85.41, 83.88, 83.01,
]
_ETM_DTR_44 = [
    0.3950, 0.3950, 0.3951, 0.3951, 0.3952, 0.3953, 0.3954, 0.3956, 0.3957, 0.3960,
    0.3963, 0.3967, 0.3973, 0.3981, 0.3992, 0.4009, 0.4033, 0.4066, 0.4108, 0.4153,
    0.4196, 0.4231, 0.4256, 0.4273, 0.4285, 0.4294, 0.4299, 0.4304, 0.4307, 0.4309,
    0.4311,
]


@register(
    _AIRCRAFT,
    "ICAO Annex 16 Vol. I App. 2 Table A2-3",
    "Perceived noisiness at SPL(b), 1 kHz band, in noys",
)
def _chk_ac_noy() -> Outcome:
    spl = np.full(24, -999.0)
    spl[13] = 40.0  # 1000 Hz, SPL(b) -> n = 1
    return numeric(1.0, float(ph.perceived_noisiness(spl)[13]), 1e-6, places=4)


@register(
    _AIRCRAFT,
    "ICAO Doc 9501 ETM Vol. I Table 3-7",
    "Tone correction of the turbofan example, dB",
)
def _chk_ac_tone() -> Outcome:
    return numeric(2.0, ph.tone_correction(_ETM_SPL_37), 1e-6, places=4)


@register(
    _AIRCRAFT,
    "ICAO Doc 9501 ETM Vol. I Table 4-4",
    "Integrated-method reference EPNL, EPNdB",
)
def _chk_ac_epnl() -> Outcome:
    epnl, _, _, _ = ph.epnl_from_pnlt(np.array(_ETM_PNLTR_44), np.array(_ETM_DTR_44))
    return numeric(92.61892, epnl, 1e-2, unit="EPNdB", places=3)


@register(
    _AIRCRAFT,
    "IEC 61265:1995 Table 1",
    "Directional-response tolerance at 4 kHz / 90°, dB",
)
def _chk_ac_iec61265() -> Outcome:
    from phonometry.aircraft.measurement_system import (
        _iec61265_directional_limit,
    )

    return numeric(2.0, _iec61265_directional_limit(4000.0, 90.0), 1e-9, unit="dB", places=1)
