#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Industrial noise control (Bies 5e; silencers, HVAC, enclosures).

The engineering calculations of a noise-control design: dissipative and
reactive silencers and their insertion loss, duct-borne sound in an HVAC path
(breakout, elbows, end reflection, ASHRAE regenerated noise), machine
enclosures and their required transmission loss, and the room correction that
turns a sound power into a level at a receiver.

Every check is a closed form of Bies 5th edition, Long or the ASHRAE
Applications handbook, evaluated on a plant room sized like a real one.
"""

from __future__ import annotations

import math

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, register

_NOISE_CONTROL = "Industrial noise control"


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.111)",
    "Expansion-chamber peak TL = 10 lg[1 + (1/4)(m - 1/m)^2], m = 4 at kL = pi/2",
)
def _chk_expansion_chamber_peak() -> Outcome:
    c, length, s_duct = 343.0, 0.3, 0.01
    f = np.array([c / (4.0 * length)])  # kL = pi/2
    res = ph.expansion_chamber(f, length, 4.0 * s_duct, s_duct, speed_of_sound=c)
    expected = 10.0 * math.log10(1.0 + 0.25 * (4.0 - 0.25) ** 2)
    return numeric(expected, float(res.transmission_loss[0]), 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.111)",
    "Expansion-chamber trough TL = 0 at kL = pi (chamber transparent)",
)
def _chk_expansion_chamber_trough() -> Outcome:
    c, length, s_duct = 343.0, 0.3, 0.01
    f = np.array([c / (2.0 * length)])  # kL = pi
    res = ph.expansion_chamber(f, length, 4.0 * s_duct, s_duct, speed_of_sound=c)
    return numeric(0.0, float(res.transmission_loss[0]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.44) / Example 8.1",
    "Quarter-wave tube tuning f = c/(4 l_e), l_e = 1.516 m -> 56.6 Hz",
)
def _chk_quarter_wave_tuning() -> Outcome:
    area = math.pi * 0.05**2 / 4.0
    res = ph.quarter_wave_resonator([100.0], area, 1.516, area, speed_of_sound=343.24)
    assert res.resonances is not None
    return numeric(56.6, float(res.resonances[0]), 0.1, unit="Hz", places=2)


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.46)",
    "Helmholtz resonance f0 = (c/2pi) sqrt(S/(l_e V))  (S=1e-4, l_e=0.02, V=1e-3)",
)
def _chk_helmholtz_resonance() -> Outcome:
    c = 343.0
    res = ph.helmholtz_resonator([100.0], 0.01, 1e-4, 0.02, 1e-3, speed_of_sound=c)
    assert res.resonances is not None
    expected = c / (2.0 * math.pi) * math.sqrt(1e-4 / (0.02 * 1e-3))
    return numeric(expected, float(res.resonances[0]), 1e-6, unit="Hz", places=3)


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.73)",
    "Side-branch TL = 20 lg abs(1 + rho c/(2 Sd Zb)) (QWT branch, closed form)",
)
def _chk_side_branch_closed_form() -> Outcome:
    from phonometry.noise_control import silencers as _sl

    f = np.array([120.0])
    zb = _sl.quarter_wave_impedance(f, 0.5, 0.002)
    t = _sl.shunt_matrix(zb)
    tl = float(_sl.transmission_loss(t, inlet_area=0.01, outlet_area=0.01)[0])
    closed = 20.0 * math.log10(abs(1.0 + 1.206 * 343.0 / (2.0 * 0.01 * zb[0])))
    return numeric(closed, tl, 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eqs. (8.141)/(8.148) (four-pole insertion loss)",
    "Insertion loss = transmission loss for the anechoic reference Zs=Zr=rho c/S",
)
def _chk_insertion_loss_equals_tl() -> Outcome:
    from phonometry.noise_control import silencers as _sl

    c, rho, s = 343.0, 1.206, 0.01
    z = rho * c / s
    f = np.array([232.0])  # a frequency with a substantial, positive TL
    t = _sl.expansion_chamber(f, 0.3, 0.04, s).transfer_matrix
    tl = float(_sl.transmission_loss(t, inlet_area=s, outlet_area=s)[0])
    il = float(_sl.insertion_loss(t, source_impedance=z, radiation_impedance=z)[0])
    return numeric(tl, il, 1e-9, unit="dB",
                   expected_label=f"{tl:.4f} dB (= TL)")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eq. (8.275) (Wells' plenum method)",
    "Plenum TL = -10 lg[S_out(cos0/pi r^2 + (1-a)/(Sw a))] (S_out=.1,r=1,Sw=20,a=.2)",
)
def _chk_plenum_wells() -> Outcome:
    tl = float(ph.plenum_attenuation(0.1, 1.0, 20.0, 0.2))
    direct = 1.0 / (math.pi * 1.0**2)
    reverb = (1.0 - 0.2) / (20.0 * 0.2)
    expected = -10.0 * math.log10(0.1 * (direct + reverb))
    return numeric(expected, tl, 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Bies 5e Table 8.14 (ASHRAE end reflection, flush)",
    "Duct end reflection D = 200 mm at 125 Hz = 10 dB (table node)",
)
def _chk_end_reflection_table() -> Outcome:
    res = ph.end_reflection_loss([125.0], 0.200, termination="flush")
    return numeric(10.0, float(res.values[0]), 1e-6, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 13.1 with Table 13.5 (ASHRAE 1987 fan model)",
    "Forward-curved fan at Q_REF, P_REF, peak efficiency -> K_F + C_BFI at 500 Hz",
)
def _chk_fan_sound_power_reference_point() -> Outcome:
    res = ph.fan_sound_power(
        0.472e-3, 249.0, fan_type="forward_curved", relative_efficiency=100.0
    )
    # Table 13.5 forward-curved 500 Hz entry (36 dB) plus the Table 13.7
    # blade frequency increment of the 500 Hz octave (2 dB).
    return numeric(38.0, float(res.values[3]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 14.12 with Table 14.2 (Reynolds lined rectangular duct)",
    "18 x 12 in duct, 6 ft, 1 in lining at 1 kHz -> 1.77 (10/3)^0.695 6 dB",
)
def _chk_lined_rectangular_duct() -> Outcome:
    res = ph.lined_rectangular_duct_attenuation(
        None, 18.0 * 0.0254, 12.0 * 0.0254, 6.0 * 0.3048, 0.0254
    )
    expected = 1.7700 * (10.0 / 3.0) ** 0.695 * 6.0
    return numeric(expected, float(res.values[4]), 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Table 14.4 (ASHRAE 1995 lined flexible duct)",
    "8 in diameter, 9 ft long -> 6/8/16/25/28/28/18 dB (table node)",
)
def _chk_flexible_duct_table() -> Outcome:
    res = ph.flexible_duct_insertion_loss(None, 8.0 * 0.0254, 9.0 * 0.3048)
    printed = np.array([6, 8, 16, 25, 28, 28, 18], dtype=float)
    worst = float(np.max(np.abs(res.values - printed)))
    return numeric(0.0, worst, 1e-9, unit="dB",
                   expected_label="0 dB (max |diff| over the 7 bands)")


@register(
    _NOISE_CONTROL,
    "Long 2e Eq. 14.17 (branch power division)",
    "25 per cent split with area-matched branches -> -10 lg 0.25 = 6.02 dB",
)
def _chk_split_loss() -> Outcome:
    loss = ph.split_loss(0.6, [0.15, 0.15, 0.15, 0.15], branch=0)
    return numeric(-10.0 * math.log10(0.25), loss, 1e-9, unit="dB")


@register(
    _NOISE_CONTROL,
    "Long 2e Table 14.9 (worked duct-borne sheet, supply path)",
    "Fan to room, 8 octave bands -> 52/42/30/18/9/-2/-2/-1 dB at the receiver",
)
def _chk_duct_path_table_14_9() -> Outcome:
    from phonometry.noise_control.duct_path import DuctElement, duct_path
    from phonometry.noise_control.hvac import OCTAVE_BANDS

    res = duct_path(
        OCTAVE_BANDS,
        [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0],
        [
            DuctElement("Elbow", [0, 1, 2, 3, 3, 3, 3, 3],
                        [41, 39, 36, 29, 20, 6, 0, 0]),
            DuctElement("Silencer", [7, 12, 16, 28, 35, 35, 28, 17],
                        [49, 43, 44, 42, 42, 45, 35, 24]),
            DuctElement("Lined duct 36x24", [2, 2, 3, 7, 15, 12, 11, 9]),
            DuctElement("Split 25%", 6.0),
            DuctElement("Lined duct 18x12", [3, 3, 5, 11, 25, 22, 16, 13]),
            DuctElement("Flexible duct", [14, 14, 16, 15, 17, 22, 16, 13]),
            DuctElement("Diffuser", 0.0, [33, 32, 29, 23, 15, 4, 0, 0]),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
    )
    printed = np.array([52, 42, 30, 18, 9, -2, -2, -1], dtype=float)
    worst = float(np.max(np.abs(np.round(res.received_level) - printed)))
    # The printed sheet carries whole decibels and its own rounding is not
    # always self-consistent, so 1 dB is the published resolution.
    return numeric(0.0, worst, 1.0, unit="dB",
                   expected_label="0 dB +/-1 (max |diff| over the 8 bands)")


@register(
    _NOISE_CONTROL,
    "Long 2e Eqs. 13.27-13.33 (Reynolds diffuser self-noise)",
    "24 x 24 in rectangular diffuser, 312 cfm, 0.05 in pd -> the 33/32/29/23/15 dB "
    "row of Table 14.9",
)
def _chk_diffuser_sound_power() -> Outcome:
    res = ph.diffuser_sound_power(
        None, (24.0 * 0.0254) ** 2, 312.0 * 0.0004719474432, 0.05 * 249.0
    )
    printed = np.array([33.0, 32.0, 29.0, 23.0, 15.0])
    worst = float(np.max(np.abs(res.values[:5] - printed)))
    return numeric(0.0, worst, 1.0, unit="dB",
                   expected_label="0 dB +/-1 (max |diff| over the five bands)")


@register(
    _NOISE_CONTROL,
    "ASHRAE 2019 Applications Ch. 49 Table 9",
    "Max neck velocity of a supply outlet for design RC(30) -> 2.2 m/s",
)
def _chk_air_terminal_velocity() -> Outcome:
    return numeric(2.2, ph.air_terminal_velocity_limit(30), 1e-9, unit="m/s")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eqs. 7.6/7.8/7.9 (problem 7.1 answer)",
    "254 mm duct, steam, 200 m/s: (1,0) cut-on 812 Hz and k_x = -8.23 1/m",
)
def _chk_duct_cut_on_with_flow() -> Outcome:
    res = ph.circular_duct_cut_on(
        0.254, flow_velocity=200.0, speed_of_sound=405.0, count=1
    )
    worst = max(
        abs(float(res.cut_on[0]) - 812.0),
        abs(float(res.axial_wavenumber[0]) + 8.23) * 100.0,
    )
    return numeric(0.0, worst, 1.0,
                   expected_label="0 +/-1 (Hz, and 1/m x100)")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eq. 7.10 (problem 7.2 answer)",
    "0.65 x 0.4 m duct, 15 m/s: first three cut-on 264 / 428 / 503 Hz",
)
def _chk_rectangular_duct_cut_on() -> Outcome:
    res = ph.rectangular_duct_cut_on(0.65, 0.4, flow_velocity=15.0, count=3)
    printed = np.array([264.0, 428.0, 503.0])
    worst = float(np.max(np.abs(np.round(res.cut_on) - printed)))
    return numeric(0.0, worst, 1e-9, unit="Hz",
                   expected_label="0 Hz (max |diff| over the 3 modes)")


@register(
    _NOISE_CONTROL,
    "Bies 5e Eqs. (7.103), (7.111) (enclosure, fully absorbing limit)",
    "Enclosure correction C -> 10 lg 0.3 = -5.23 dB as alpha_i -> 1",
)
def _chk_enclosure_floor() -> Outcome:
    res = ph.enclosure_insertion_loss([40.0], 6.0, 5.0, 0.999999)
    return numeric(10.0 * math.log10(0.3), float(res.correction[0]), 1e-3,
                   unit="dB")


# --- Room-to-room chain (Norton & Karczub 2e, Chapter 4 problems) ---------

#: Receiving room of Norton problem 4.21: 8 x 9 x 3 m, printed absorption
#: coefficients of the walls, floor and ceiling over 125 Hz to 4 kHz.
_N421_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
_N421_SURFACES = (
    (2.0 * (8.0 * 3.0) + 2.0 * (9.0 * 3.0),
     np.array([0.04, 0.04, 0.09, 0.15, 0.17, 0.23])),
    (72.0, np.array([0.02, 0.06, 0.14, 0.37, 0.60, 0.66])),
    (72.0, np.array([0.30, 0.20, 0.15, 0.05, 0.05, 0.05])),
)


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eq. (4.101) (problem 4.21 answer)",
    "Double brick wall into an 8 x 9 x 3 m room -> NR 37.5/40.8/49.0/62.8/65.3/65.9 dB",
)
def _chk_room_to_room_noise_reduction() -> Outcome:
    res = ph.room_to_room_transmission(
        _N421_BANDS,
        [37.0, 41.0, 48.0, 60.0, 61.0, 61.0],
        8.0 * 3.0,
        ph.equivalent_absorption_area(_N421_SURFACES),
        source=ph.SourceRoom(level=90.0),
    )
    printed = np.array([37.5, 40.8, 49.0, 62.8, 65.3, 65.9])
    worst = float(np.max(np.abs(np.asarray(res.noise_reduction) - printed)))
    return numeric(0.0, worst, 0.05, unit="dB",
                   expected_label="0 dB +/-0.05 (max |diff| over the 6 bands)")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e 4.6/4.9 (problem 4.18 answer)",
    "Blower in a plant room to the operator room -> "
    "72.3/60.4/41.4/41.0/33.8/30.7 dB",
)
def _chk_room_to_room_chain() -> Outcome:
    ceiling = np.array([0.07, 0.20, 0.40, 0.52, 0.60, 0.67])
    walls = np.array([0.03, 0.03, 0.03, 0.04, 0.05, 0.07])
    plant = (
        (80.0, np.array([0.01, 0.01, 0.015, 0.02, 0.02, 0.02])),
        (80.0, ceiling),
        (108.0, walls),
    )
    operator = (
        (25.0, np.array([0.08, 0.24, 0.57, 0.69, 0.71, 0.73])),
        (25.0, ceiling),
        (60.0, walls),
    )
    res = ph.room_to_room_transmission(
        _N421_BANDS,
        [39.0, 42.0, 50.0, 58.0, 63.0, 67.0],
        15.0,
        ph.equivalent_absorption_area(operator),
        source=ph.SourceRoom(
            power_level=[105.0, 103.0, 98.0, 108.0, 107.0, 109.0],
            room_constant=ph.room_constant(268.0, ph.mean_absorption(plant)),
            directivity=4.0,
            model="constant_volume",
        ),
    )
    printed = np.array([72.3, 60.4, 41.4, 41.0, 33.8, 30.7])
    worst = float(np.max(np.abs(np.asarray(res.received_level) - printed)))
    return numeric(0.0, worst, 0.1, unit="dB",
                   expected_label="0 dB +/-0.1 (max |diff| over the 6 bands)")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Eq. (4.115) (problem 4.16 answer)",
    "Lined compressor enclosure against NC-45 -> required TL "
    "14.4/25.2/28.9/34.4/35.2/34.7/34.7/31.6 dB",
)
def _chk_enclosure_required_transmission_loss() -> Outcome:
    external = 2.0 * (2.5 * 2.5) + 2.0 * (3.5 * 2.5) + 2.5 * 3.5
    machine = 2.0 * (1.5 * 1.5) + 2.0 * (2.5 * 1.5) + 1.5 * 2.5
    bare_floor = 2.5 * 3.5 - 1.5 * 2.5
    wool = np.array([0.10, 0.20, 0.45, 0.65, 0.75, 0.80, 0.80, 0.80])
    concrete = np.array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.03, 0.03])
    lp1 = np.array([72.0, 79.0, 81.0, 84.0, 83.0, 81.0, 80.0, 75.0])
    lp2 = np.array([67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 41.0])
    res = ph.enclosure_required_transmission_loss(
        lp1 - lp2,
        external,
        external + bare_floor + machine,
        ph.mean_absorption(
            ((external, wool), (bare_floor + machine, concrete))
        ),
        frequencies=[63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0],
        model="norton",
    )
    printed = np.array([14.4, 25.2, 28.9, 34.4, 35.2, 34.7, 34.7, 31.6])
    worst = float(np.max(np.abs(res.panel_transmission_loss - printed)))
    return numeric(0.0, worst, 0.15, unit="dB",
                   expected_label="0 dB +/-0.15 (max |diff| over the 8 bands)")


@register(
    _NOISE_CONTROL,
    "Norton & Karczub 2e Table 4.5 (constant-volume source power)",
    "Source in the intersection of two flat surfaces (Q = 4) -> +10 lg 4 = 6.02 dB",
)
def _chk_constant_volume_source_model() -> Outcome:
    base = float(ph.steady_state_spl(100.0, None, 40.0, directivity=4.0))
    raised = float(
        ph.steady_state_spl(
            100.0, None, 40.0, directivity=4.0, source_model="constant_volume"
        )
    )
    return numeric(10.0 * math.log10(4.0), raised - base, 1e-9, unit="dB")
