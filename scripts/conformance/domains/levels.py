#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Domain 2 - Levels & dosimetry.

Sound level meter and dosimeter arithmetic: the IEC 61672-1 time and frequency
weightings applied to signals, the IEC 61252 exposure quantities, and the
ISO 1996-1/-2 rating levels and adjustments.

Two bodies of work sit with them because they are the same arithmetic applied
by law and by a room: the Spanish noise regulation RD 1367/2007 and building
code CTE DB-HR, whose oracles are the printed limit tables and the worked
examples of Aviles Lopez & Perera Martin; and reverberation-time prediction
(Sabine, Eyring, Millington, Fitzroy, Arau-Puchades), which no source carries
as a machine-readable worked example, so those checks anchor on hand-computed
closed-form values and on the identities the models must satisfy against each
other.
"""

from __future__ import annotations

import math

import numpy as np
import reference_data as ref

import phonometry as ph

from ..registry import Outcome, numeric, register

_FS = 48000


def _tone(freq: float, seconds: float = 1.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(_FS * seconds)) / _FS
    return amp * np.sin(2.0 * np.pi * freq * t)


@register(
    "Levels & dosimetry",
    "IEC 61672-1:2013 (Leq)",
    "Leq of a 1 Pa 1 kHz sine",
)
def _chk_leq_sine() -> Outcome:
    # 20*log10((1/sqrt2) / 20e-6) = 90.97 dB.
    computed = float(ph.signals.leq(_tone(1000.0)))
    return numeric(90.97, computed, 0.05, unit="dB", places=3)


@register(
    "Levels & dosimetry",
    "IEC 61252:1993 (LEX,8h)",
    "8 h exposure to 90 dB(A) noise",
)
def _chk_lex_8h() -> Outcome:
    rms = 2e-5 * 10 ** (90.0 / 20.0)
    x = math.sqrt(2) * rms * _tone(1000.0, seconds=2.0)
    computed = float(ph.signals.lex_8h(x, _FS, duration_hours=8.0))
    return numeric(90.0, computed, 0.05, unit="dB", places=3)


@register(
    "Levels & dosimetry",
    "ISO 1996-1:2016 3.6.4",
    "Lden, constant 60 dB in day/evening/night",
)
def _chk_lden() -> Outcome:
    offset = 10.0 * math.log10((12 + 4 * 10**0.5 + 8 * 10) / 24)
    computed = float(ph.environment.lden(60.0, 60.0, 60.0))
    return numeric(60.0 + offset, computed, 1e-6, unit="dB", places=4)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2007 Annex C.5 Example 1",
    "Tonal audibility ΔLta (Formula C.3), 4 kHz tone",
)
def _chk_iso1996_2_tonal_audibility() -> Outcome:
    lpt, lpn, fc, delta_expected, _kt = ref.ISO1996_2_TONAL_EXAMPLES[0]
    computed = ph.environment.tonal_audibility(lpt, lpn, fc)
    return numeric(delta_expected, computed, 0.05, unit="dB", places=2)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2007 Annex C.5 Example 1",
    "Tonal adjustment Kt (Formulae C.4-C.6)",
)
def _chk_iso1996_2_tonal_adjustment() -> Outcome:
    lpt, lpn, fc, _delta, kt_expected = ref.ISO1996_2_TONAL_EXAMPLES[0]
    computed = ph.environment.tonal_adjustment(
        ph.environment.tonal_audibility(lpt, lpn, fc)
    )
    return numeric(kt_expected, computed, 1e-9, unit="dB", places=2)


@register(
    "Levels & dosimetry",
    "ISO 1996-2:2017 Annex G.2",
    "Combined measurement uncertainty u = √(Σ(cj·uj)²)",
)
def _chk_iso1996_2_uncertainty() -> Outcome:
    computed = ph.environment.combined_standard_uncertainty(
        ref.ISO1996_2_G2_CONTRIBUTIONS
    )
    return numeric(ref.ISO1996_2_G2_COMBINED, computed, 0.01, unit="dB", places=2)


# ---------------------------------------------------------------------------
# Spanish noise regulation (RD 1367/2007) and building code (CTE DB-HR).
# Oracles: the printed limit tables and procedures of the two legal texts, and
# the worked examples of Aviles Lopez & Perera Martin, "Manual de acustica
# ambiental y arquitectonica" (Paraninfo), Ejemplos 3.1-3.3 and 7.2 / 7.4.
# ---------------------------------------------------------------------------
#: Manual Ejemplo 3.1: the day period of an activity on residential land,
#: split into 2 h shut down, 6 h with the noisy machine (LAeq 50 dB, Kt 6,
#: Kf 3) and 4 h with the remaining sources (LAeq 48 dB, Kt 3, Kf 3).
_RD1367_DAY_PHASES = [
    (2.0, 0.0, 0.0, 0.0),
    (6.0, 50.0, 6.0, 3.0),
    (4.0, 48.0, 3.0, 3.0),
]
#: Manual Ejemplo 7.2: the measured apparent sound reduction index R' of a
#: field test, one-third-octave bands 100 Hz to 5 kHz.
_DBHR_R_PRIME = [
    36.2,
    41.5,
    36.9,
    40.4,
    44.7,
    42.4,
    45.7,
    46.1,
    47.1,
    52.3,
    54.3,
    57.5,
    57.8,
    57.3,
    59.0,
    62.8,
    64.7,
    65.3,
]


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex IV A.3.4.2 b",
    "Corrected period level LKeq,d (Manual Ejemplo 3.1: 3 noise phases, 12 h)",
)
def _chk_rd1367_period_level() -> Outcome:
    phases = [
        ph.environment.NoisePhase(hours, laeq, kt=kt, kf=kf)
        for hours, laeq, kt, kf in _RD1367_DAY_PHASES
    ]
    level = ph.environment.evaluation_period_level(phases, hours=12.0)
    return numeric(
        57.0,
        float(ph.environment.round_reported_level(level)),
        1e-9,
        unit="dB",
    )


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex I A.2 d",
    "Long-term level LK,d (Manual Ejemplo 3.2: 303 operating days of 365)",
)
def _chk_rd1367_long_term_level() -> Outcome:
    level = ph.environment.long_term_corrected_level([57.0, 0.0], weights=[303.0, 62.0])
    return numeric(
        56.0,
        float(ph.environment.round_reported_level(level)),
        1e-9,
        unit="dB",
    )


@register(
    "Levels & dosimetry",
    "RD 1367/2007 Annex III Table B1, Article 25",
    "Activity verdict (Manual Ejemplo 3.3: area type a, LK,d 56 dB over 55 dB)",
)
def _chk_rd1367_activity_verdict() -> Outcome:
    day = [
        ph.environment.NoisePhase(hours, laeq, kt=kt, kf=kf)
        for hours, laeq, kt, kf in _RD1367_DAY_PHASES
    ]
    evening = [
        ph.environment.NoisePhase(2.0, 48.0, kt=3.0, kf=3.0),
        ph.environment.NoisePhase(2.0, 0.0),
    ]
    verdict = ph.environment.assess_activity(
        {"day": day, "evening": evening},
        ph.environment.activity_limits("a"),
        operating_days=303,
    )
    # The book: the phase and daily criteria pass, the annual one does not, so
    # a new activity does not comply while one already in operation does.
    computed = (
        verdict.periods[0].phase_pass
        and verdict.periods[0].daily_pass
        and not verdict.periods[0].long_term_pass
        and not verdict.complies
    )
    return Outcome(
        expected="phase and daily pass, annual fails, activity not compliant",
        computed=(
            "phase and daily pass, annual fails, activity not compliant"
            if computed
            else "verdict differs"
        ),
        delta="-",
        passed=bool(computed),
    )


@register(
    "Room & building acoustics",
    "CTE DB-HR Annex A, Formula (A.5)",
    "Global index R'A for pink noise (Manual Ejemplo 7.2)",
)
def _chk_dbhr_ra() -> Outcome:
    computed = ph.building.ra(_DBHR_R_PRIME).intermediate
    return numeric(51.4, float(computed), 0.05, unit="dBA", places=2)


@register(
    "Room & building acoustics",
    "CTE DB-HR Annex A, Formula (A.6)",
    "Global index D2m,nT,Atr for road traffic (Manual Ejercicio 7.1)",
)
def _chk_dbhr_d2m_nt_atr() -> Outcome:
    values = [
        28.5,
        28.5,
        18.9,
        23.7,
        30.7,
        31.3,
        37.8,
        35.2,
        34.7,
        38.5,
        37.7,
        43.1,
        42.3,
        44.2,
        41.9,
        37.5,
        39.4,
        41.5,
    ]
    computed = ph.building.d2m_nt_atr(values).intermediate
    return numeric(32.8, float(computed), 0.05, unit="dBA", places=2)


@register(
    "Room & building acoustics",
    "Manual de acustica ambiental y arquitectonica, Ejemplo 7.1",
    "Reported R'A of the field-test wall (printed 51 dBA = R'w 52 + C -1)",
)
def _chk_dbhr_route_agreement() -> Outcome:
    # The book prints the pink-noise figure as 51 dBA, so 51 is the oracle;
    # comparing the direct route against the library's own ISO 717-1 engine
    # would be a self-consistency check, not an external validation. The unit
    # test pins R'w = 52 and C = -1 against the same printed example.
    computed = float(ph.building.ra(_DBHR_R_PRIME).reported)
    return numeric(51.0, computed, 1e-9, unit="dBA", places=1)


@register(
    "Room & building acoustics",
    "Manual de acustica ambiental y arquitectonica, Ejemplo 7.1",
    "Reported R'A,tr of the same wall (printed 47 dBA = R'w 52 + Ctr -5)",
)
def _chk_dbhr_ra_tr() -> Outcome:
    computed = float(ph.building.ra_tr(_DBHR_R_PRIME).reported)
    return numeric(47.0, computed, 1e-9, unit="dBA", places=1)


@register(
    "Room & building acoustics",
    "CTE Catalogo de Elementos Constructivos",
    "Window size correction of RA (Manual Ejemplo 7.4: 4 m2 window, -2 dB)",
)
def _chk_dbhr_window_correction() -> Outcome:
    computed = 26.0 + ph.building.window_size_correction(4.0)
    return numeric(24.0, float(computed), 1e-9, unit="dBA", places=1)


# ---------------------------------------------------------------------------
# Reverberation-time prediction (Sabine / Eyring / Millington / Fitzroy /
# Arau-Puchades). No source carries a machine-readable worked example, so the
# checks anchor on hand-computed closed-form values and the model identities.
# ---------------------------------------------------------------------------
# Shoebox 8x5x3 m: V = 120 m3, S = 158 m2. Values hand-derived with the default
# c0 = 343 m/s (Sabine constant k = 24 ln10 / c0 = 0.161113...).
_RT_DIMS = (8.0, 5.0, 3.0)
_RT_VOLUME = 120.0
_RT_SURFACES = [
    (40.0, 0.2),
    (40.0, 0.2),
    (24.0, 0.2),
    (24.0, 0.2),
    (15.0, 0.2),
    (15.0, 0.2),
]


@register(
    "Room acoustics",
    "Sabine (W. C. Sabine, 1922)",
    "Reverberation time T = k·V/A  (V=120 m³, S=158 m², α=0.2)",
)
def _chk_sabine_rt() -> Outcome:
    computed = float(ph.room.sabine_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    return numeric(0.6118246547, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Table 8.1",
    "Room modes of a 7 x 5 x 3 m room: the six printed frequencies, Hz",
)
def _chk_long_room_modes() -> Outcome:
    # Long's Chapter 2 quotes c0 = 344 m/s at 20 degC; the printed table is
    # consistent with 344.7 m/s, so the tolerance is one printed digit.
    printed = np.array([24.6, 34.5, 42.4, 49.2, 57.4, 60.1])
    res = ph.room.room_modes((7.0, 5.0, 3.0), max_frequency=61.0, speed_of_sound=344.0)
    computed = np.asarray(res.frequencies, dtype=float)
    worst = int(np.argmax(np.abs(computed - printed)))
    return numeric(printed[worst], computed[worst], 0.13, unit="Hz", places=2)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (8.46)",
    "Modal density of a 7 x 5 x 3 m room at 1 kHz = 34 modes/Hz",
)
def _chk_long_modal_density() -> Outcome:
    density = float(
        np.asarray(
            ph.room.room_modal_density(1000.0, (7.0, 5.0, 3.0), speed_of_sound=344.0)
        )[()]
    )
    return numeric(34.0, density, 0.5, unit="modes/Hz", places=2)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (17.51)",
    "Restaurant self-noise, 20 talkers over 20 metric sabins = 76 dB",
)
def _chk_long_restaurant_self_noise() -> Outcome:
    level = float(np.asarray(ph.room.crowd_noise_level(20, 20.0))[()])
    return numeric(76.0, level, 0.05, unit="dB", places=3)


@register(
    "Room acoustics",
    "Long, Architectural Acoustics 2e, Eq. (17.54)",
    "Privacy bound A_tab < 3.16 rt^2 (Q = 2, L_SN = -9 dB)",
)
def _chk_long_privacy_bound() -> Outcome:
    a_tab = float(np.asarray(ph.room.absorption_per_table(1.0, -9.0))[()])
    return numeric(3.16, a_tab, 0.005, unit="m^2", places=4)


@register(
    "Room acoustics",
    "Everest, Master Handbook of Acoustics 4th ed, Fig. 7-22",
    "Sabine RT, worked Example 1 @ 1 kHz (untreated 23.3×16×10 ft room, SI)",
)
def _chk_sabine_everest() -> Outcome:
    surfaces = [
        (ref.EVEREST_EX1_FLOOR_AREA, ref.EVEREST_EX1_FLOOR_ALPHA[3]),
        (ref.EVEREST_EX1_SHELL_AREA, ref.EVEREST_EX1_SHELL_ALPHA[3]),
    ]
    computed = float(
        ph.room.sabine_reverberation_time(ref.EVEREST_EX1_VOLUME, surfaces)
    )
    return numeric(ref.EVEREST_EX1_RT[3], computed, 0.02, unit="s", places=3)


@register(
    "Room acoustics",
    "Eyring (Norris-Eyring, 1930)",
    "Reverberation time T = k·V/(-S·ln(1-ᾱ))  (α=0.2)",
)
def _chk_eyring_rt() -> Outcome:
    computed = float(ph.room.eyring_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    return numeric(0.5483686633, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Arau-Puchades (Acustica 65, 1988, Formula 18)",
    "T (α=0.5/0.1/0.1 per wall pair, dims 8×5×3 m)",
)
def _chk_arau_rt() -> Outcome:
    computed = float(
        ph.room.arau_puchades_reverberation_time(_RT_DIMS, (0.5, 0.1, 0.1))
    )
    return numeric(0.8121469281, computed, 1e-6, unit="s", places=6)


@register(
    "Room acoustics",
    "Model identity (uniform absorption)",
    "Arau-Puchades ≡ Eyring when ᾱ is uniform",
)
def _chk_arau_eyring_identity() -> Outcome:
    eyring = float(ph.room.eyring_reverberation_time(_RT_VOLUME, _RT_SURFACES))
    arau = float(ph.room.arau_puchades_reverberation_time(_RT_DIMS, (0.2, 0.2, 0.2)))
    return numeric(
        eyring,
        arau,
        1e-9,
        unit="s",
        places=6,
        expected_label=f"{eyring:.6f} s (= Eyring)",
    )


@register(
    "Room acoustics",
    "Vorlander Auralization 2e, Eq. (11.38)-(11.39)",
    "Image-source direct-sound amplitude 1/(4πr) and delay r/c (r = 4 m)",
)
def _chk_image_source_direct() -> Outcome:
    res = ph.room.image_source_rir(
        (8.0, 5.0, 3.0), (2.0, 2.5, 1.5), (6.0, 2.5, 1.5), 0.2, fs=48000, max_order=2
    )
    amp = float(np.atleast_1d(res.amplitudes)[0])
    return numeric(1.0 / (4.0 * math.pi * 4.0), amp, 1e-9, places=7)


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (9.23)",
    "Audible shoebox image count up to order 10 (= 1560)",
)
def _chk_image_source_count() -> Outcome:
    computed = float(ph.room.audible_image_count(10))
    return numeric(1560.0, computed, 0.0, places=0)


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (4.6)",
    "Temporal reflection density dN/dt = 4πc³t²/V (t = 0.1 s, V = 120 m³)",
)
def _chk_reflection_density() -> Outcome:
    computed = float(ph.room.reflection_density(0.1, 120.0))
    expected = 4.0 * math.pi * 343.0**3 * 0.1**2 / 120.0
    return numeric(expected, computed, 1e-6, unit="1/s", places=2)


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.44)",
    "Room constant R = Sᾱ/(1-ᾱ)  (S = 100 m², ᾱ = 0.2 → 25 m²)",
)
def _chk_room_constant() -> Outcome:
    return numeric(
        25.0, float(ph.room.room_constant(100.0, 0.2)), 1e-9, unit="m²", places=6
    )


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.43)",
    "Critical distance rc: direct field = reverberant field (R = 25, Q = 1)",
)
def _chk_critical_distance_crossover() -> Outcome:
    rc = float(ph.room.critical_distance(25.0))
    direct = 1.0 / (4.0 * math.pi * rc**2)
    reverberant = 4.0 / 25.0
    return numeric(
        reverberant,
        direct,
        1e-9,
        places=6,
        expected_label=f"{reverberant:.6f} (= reverberant term)",
    )


@register(
    "Room acoustics",
    "Kuttruff Room Acoustics 6e, Eq. (3.44)",
    "Schroeder frequency f_s = 2000√(T/V)  (V = 200 m³, T = 1 s)",
)
def _chk_schroeder_frequency() -> Outcome:
    computed = float(ph.room.schroeder_frequency(1.0, 200.0))
    return numeric(2000.0 * math.sqrt(1.0 / 200.0), computed, 1e-6, unit="Hz", places=3)


@register(
    "Room acoustics",
    "Bies Engineering Noise Control 5e, Eq. (6.43)",
    "Steady-state SPL Lp = Lw + 10lg(Q/4πr² + 4/R)  (Lw=90, r=1, R=25, Q=1)",
)
def _chk_steady_state_spl() -> Outcome:
    computed = float(ph.room.steady_state_spl(90.0, 1.0, 25.0))
    expected = 90.0 + 10.0 * math.log10(1.0 / (4.0 * math.pi) + 4.0 / 25.0)
    return numeric(expected, computed, 1e-6, unit="dB", places=4)
