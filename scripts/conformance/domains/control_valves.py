#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Control valve aerodynamic noise (IEC 60534-8-3), against its own Annex A.

Annex A of IEC 60534-8-3:2010 is an unusually complete oracle: it prints six
operating points of one valve, one per flow regime with regime V used twice,
and it prints *every intermediate* of each of them beside the number of the
equation that produced it. A seventh example, on a different valve, prints the
transmission loss of its pipe in all 33 one-third-octave bands.

Oracle: Annex A, Table A.1 (printed folios 32 to 39, PDF pages 34 to 41) and
Table A.2 (printed folios 41 to 45, PDF pages 43 to 47) of
BS EN 60534-8-3:2011, which endorses IEC 60534-8-3:2010 without modification.

Two of the annex's printed values do not reproduce themselves, and the rows
below use the values that make it self-consistent:

* The piping geometry factor is printed ``F_p = 0,98``. Every one of the six
  printed vena contracta pressures needs ``0,984``: solving Equation (2) for
  ``(F_LP/F_P)^2`` from each printed pair gives 0,647 83 to five digits in all
  six columns, and that is ``F_p = 0,984``.
* The equivalent orifice diameter is printed ``d_o = 0,010`` m where
  Equation (8c) with the annex's own ``N_O = 6`` and ``A = 0,00137`` m² gives
  0,102 m. The ``F_d = 0,30`` printed on the next row is the ratio of the
  printed ``d_H = 0,030`` m to the larger value.

Both are in ``docs/ERRATA.md``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, record, register

_IEC60534 = "Control valve noise (IEC 60534-8-3)"

#: A.2's given data, shared by examples 1 to 6. The pressure recovery is
#: ``F_LP/F_p`` with the 0,984 the annex computed with rather than the 0,98 it
#: printed.
_COMMON: dict[str, Any] = {
    "inlet_pressure": 1.0e6,
    "inlet_density": 5.3,
    "inlet_temperature": 450.0,
    "specific_heat_ratio": 1.22,
    "molecular_mass": 19.8,
    "pressure_recovery": 0.792 / 0.984,
    "wall_thickness": 0.008,
    "pipe_density": 8000.0,
    "efficiency_correction": -3.8,
    "strouhal_number": 0.2,
}

#: The cage of A.2: six passages, each 0,00137 m² with a 0,181 m perimeter.
_PASSAGE_AREA = 0.00137
_WETTED_PERIMETER = 0.181
_PASSAGES = 6

#: Table A.1's six columns: mass flow (kg/s), outlet pressure (Pa), required
#: flow coefficient, valve outlet diameter (m) and internal pipe diameter (m).
_EXAMPLES = {
    1: (2.22, 7.2e5, 90.0, 0.1, 0.2031),
    2: (2.29, 6.9e5, 90.0, 0.1, 0.2031),
    3: (2.59, 4.8e5, 90.0, 0.1, 0.2031),
    4: (1.18, 4.2e5, 40.0, 0.2031, 0.2031),
    5: (1.19, 5.0e4, 40.0, 0.2031, 0.2031),
    6: (0.89, 5.0e4, 30.0, 0.1, 0.15),
}

#: The regime each column lands in, printed in the "Regime definition" row.
_PRINTED_REGIMES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}

#: Equation (11), the sound power of each column, in W.
_PRINTED_SOUND_POWER = {1: 22.3, 2: 30.4, 3: 141.3, 4: 86.1, 5: 291.9, 6: 218.3}

#: Equation (18), the internal level at the pipe wall of each column, in dB.
_PRINTED_INTERNAL_LEVEL = {
    1: 155.3,
    2: 156.5,
    3: 161.7,
    4: 158.8,
    5: 157.0,
    6: 158.4,
}

#: Equation (2), the vena contracta pressure of each column, in Pa. It is
#: negative in the last two, where the equation is being read far past the
#: point at which it stops describing a pressure.
_PRINTED_VENA_CONTRACTA = {
    1: 567787.0,
    2: 521478.0,
    3: 197319.0,
    4: 104702.0,
    5: -466437.0,
    6: -466437.0,
}

#: Table A.2's pipe: DN 200 with 8 mm walls, and the fluid leaving the valve
#: at 11,1 kg/m3 and 408 m/s.
_EXAMPLE_7_PIPE: dict[str, Any] = {
    "internal_diameter": 0.200,
    "wall_thickness": 0.008,
    "valve_outlet_diameter": 0.200,
    "downstream_density": 11.1,
    "downstream_sound_speed": 408.0,
    "pipe_density": 8000.0,
}

#: Equation (25), the A-weighted level 1 m from the pipe wall, in dB. The
#: sixth column is the only one whose outlet Mach number passes the 0,3 of
#: NOTE 1 to Equation (15), so it is the only one the annex works Clause 7
#: for, and the only one that needs the expander to close.
_PRINTED_EXTERNAL_LEVEL = {1: 92.0, 2: 93.0, 3: 98.0, 4: 94.0, 5: 97.0}
_PRINTED_EXTERNAL_LEVEL_WITH_EXPANDER = 94.0

#: The expander chain of Clause 7, as the sixth column prints it: the pipe
#: velocity of Equation (34) in m/s, the expander inlet velocity of (35) in
#: m/s, the Mach number of (39), the stream power of (36) in W, the
#: acoustical efficiency of (38), the sound power of (40) in W, the peak
#: frequency of (37) in Hz and the internal level of (41) in dB.
_PRINTED_EXPANDER = {
    "U_p": 190.0,
    "U_R": 460.0,
    "M_R": 0.96,
    "W_mR": 47854.0,
    "eta_R": 8.8e-4,
    "W_aR": 42.0,
    "f_pR": 920.0,
    "L_piR": 151.0,
}

#: Equations (21), (22) and (23) of example 7, in Hz.
_PRINTED_PIPE_FREQUENCIES = {"f_r": 7958.0, "f_o": 2365.0, "f_g": 1622.0}

#: Equation (20a) of example 7, all 33 bands, in dB: the annex prints the
#: first 24 on one folio and the remaining nine on the next. The column turns
#: at band 24, where the loss is least, which is the 7 958 Hz ring frequency
#: of this pipe showing through.
_PRINTED_TRANSMISSION_LOSS = (
    -94.1, -92.0, -90.0, -88.1, -86.1, -84.1, -82.2, -80.2,
    -78.1, -76.2, -74.3, -72.2, -70.4, -68.5, -66.5, -64.5,
    -62.6, -60.7, -58.7, -56.9, -55.1, -53.0, -51.2, -49.4,
    -51.1, -52.8, -54.4, -56.1, -57.9, -60.0, -62.2, -64.6,
    -66.8,
)  # fmt: skip


def _style_modifier() -> float:
    """Equations (8a) to (8c) on the cage A.2 describes."""
    return ph.noise_control.valve_style_modifier(
        _PASSAGE_AREA, _WETTED_PERIMETER, _PASSAGES
    )


def _example(
    index: int, *, expander: bool = False
) -> ph.noise_control.AerodynamicValveNoise:
    """One column of Table A.1, through the whole of Clause 5.

    With ``expander`` it also runs Clause 7 on the flow leaving the valve
    outlet, which the annex does for the sixth column and no other.
    """
    flow, outlet, coefficient, diameter, bore = _EXAMPLES[index]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ph.noise_control.ValveNoiseWarning)
        return ph.noise_control.valve_aerodynamic_noise(
            **_COMMON,
            mass_flow=flow,
            outlet_pressure=outlet,
            flow_coefficient=coefficient,
            valve_outlet_diameter=diameter,
            internal_diameter=bore,
            style_modifier=_style_modifier(),
            expander=ph.noise_control.Expander() if expander else None,
        )


@register(_IEC60534, "IEC 60534-8-3:2010", "Regime, examples 1 to 6 (Table A.1)")
def _chk_regimes() -> Outcome:
    """Every column lands in the regime the annex prints for it.

    The six examples were chosen to walk the five regimes of Clause 5.2 in
    order, which is what makes them an oracle for the boundaries of
    Equations (3) to (7) and not only for the levels.
    """
    computed = {f"example {i}": float(_example(i).regime) for i in _EXAMPLES}
    expected = {f"example {i}": float(v) for i, v in _PRINTED_REGIMES.items()}
    return record(expected, computed, label="I, II, III, IV, V, V")


@register(
    _IEC60534, "IEC 60534-8-3:2010", "Valve style modifier F_d (Eqs. (8a) to (8c))"
)
def _chk_style_modifier() -> Outcome:
    """The cage of A.2 gives the printed 0,30.

    Only with the 0,102 m of Equation (8c); the annex prints 0,010 m one row
    above, which would make it 3,0.
    """
    return numeric(0.30, _style_modifier(), 5e-3, places=3)


@register(_IEC60534, "IEC 60534-8-3:2010", "Jet diameter D_j, example 1 (Eq. (9))")
def _chk_jet_diameter() -> Outcome:
    """Equation (9) on the first column."""
    return numeric(0.012, _example(1).jet_diameter, 5e-4, unit="m", places=4)


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Vena contracta pressure, examples 1 to 6 (Eq. (2))",
)
def _chk_vena_contracta() -> Outcome:
    """Equation (2) across the six columns, including the two negative ones."""
    worst = max(
        (
            abs(_example(i).vena_contracta_pressure - _PRINTED_VENA_CONTRACTA[i]),
            i,
        )
        for i in _EXAMPLES
    )
    index = worst[1]
    return numeric(
        _PRINTED_VENA_CONTRACTA[index],
        _example(index).vena_contracta_pressure,
        2.0,
        unit="Pa",
        places=0,
        expected_label=f"example {index}, the worst of the six",
    )


@register(
    _IEC60534, "IEC 60534-8-3:2010", "Sound power W_a, examples 1 to 6 (Eq. (11))"
)
def _chk_sound_power() -> Outcome:
    """Equation (11) across the six columns, one per regime.

    The acoustical efficiency changes form in every regime, so agreeing on
    all six is agreeing on the whole of Table 3.
    """
    computed = {f"example {i}": round(_example(i).sound_power, 1) for i in _EXAMPLES}
    expected = {f"example {i}": v for i, v in _PRINTED_SOUND_POWER.items()}
    return record(expected, computed, unit="W")


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Internal level at the pipe wall, examples 1 to 6 (Eq. (18))",
)
def _chk_internal_level() -> Outcome:
    """Equation (18) across the six columns.

    This is the end of the valve half of the method: everything after it is
    the pipe. Example 6 exercises the 0,3 limit NOTE 2 puts on the pipe Mach
    number, which its printed 0,4 would otherwise breach.
    """
    computed = {f"example {i}": round(_example(i).internal_level, 1) for i in _EXAMPLES}
    expected = {f"example {i}": v for i, v in _PRINTED_INTERNAL_LEVEL.items()}
    return record(expected, computed, unit="dB")


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Ring and coincidence frequencies, example 7 (Eqs. (21) to (23))",
)
def _chk_pipe_frequencies() -> Outcome:
    """The three frequencies that shape the pipe transmission loss."""
    pipe = ph.noise_control.coincidence_frequencies(0.200, 0.008, 408.0)
    computed = {
        "f_r": round(pipe.ring),
        "f_o": round(pipe.internal_coincidence),
        "f_g": round(pipe.external_coincidence),
    }
    expected = {
        "f_r": 7958.0,
        "f_o": 2366.0,  # the annex prints 2365, from its rounded c_2 of 408
        "f_g": 1622.0,
    }
    return record(expected, computed, unit="Hz")


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Pipe transmission loss, example 7, 33 bands (Eq. (20a))",
)
def _chk_transmission_loss() -> Outcome:
    """Equation (20a) across all 33 bands the annex prints.

    The loss depends on the pipe and on the fluid leaving the valve, not on
    the trim, so this row runs example 7's pipe without example 7's valve.
    """
    bands = np.asarray(_example(1).frequency, dtype=np.float64)
    loss = ph.noise_control.pipe_transmission_loss(bands, **_EXAMPLE_7_PIPE)
    deltas = [
        abs(float(loss[i]) - expected)
        for i, expected in enumerate(_PRINTED_TRANSMISSION_LOSS)
    ]
    index = int(np.argmax(deltas))
    return numeric(
        _PRINTED_TRANSMISSION_LOSS[index],
        float(loss[index]),
        0.1,
        unit="dB",
        places=2,
        expected_label=f"band {index + 1} at {bands[index]:g} Hz, the worst of 33",
    )


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "A-weighted level 1 m from the pipe wall, examples 1 to 5 (Eq. (25))",
)
def _chk_external_level() -> Outcome:
    """The end of the chain, on the five columns that end there.

    Everything the method does is behind this number: the regime, the
    efficiency, the internal level, the spectrum of Equation (19), the
    transmission loss of Equation (20a) and the A weighting of Table 7. The
    annex prints it to the decibel.
    """
    computed = {
        f"example {i}": round(_example(i).external_level)
        for i in _PRINTED_EXTERNAL_LEVEL
    }
    expected = {f"example {i}": v for i, v in _PRINTED_EXTERNAL_LEVEL.items()}
    return record(expected, computed, unit="dB")


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Expander chain of Clause 7, example 6 (Eqs. (34) to (41))",
)
def _chk_expander() -> Outcome:
    """Every printed intermediate of the one column the annex works it for.

    Example 6 runs its valve outlet at Mach 0,89, far past the 0,3 that
    NOTE 1 to Equation (15) holds Clause 5 to, so the flow leaving the outlet
    is a second source and the annex computes it.
    """
    found = _example(6, expander=True).expander
    if found is None:  # pragma: no cover - the call above asks for one
        msg = "the expander was not computed"
        raise RuntimeError(msg)
    computed = {
        "U_p": round(found.pipe_velocity),
        "U_R": round(found.inlet_velocity),
        "M_R": round(found.mach, 2),
        "W_mR": round(found.stream_power),
        "eta_R": round(found.acoustical_efficiency, 5),
        "W_aR": round(found.sound_power, 1),
        "f_pR": round(found.peak_frequency),
        "L_piR": round(found.internal_level),
    }
    expected = dict(_PRINTED_EXPANDER)
    expected["eta_R"] = round(expected["eta_R"], 5)
    return record(expected, computed)


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "A-weighted level with the expander, example 6 (Eqs. (43) and (25))",
)
def _chk_external_with_expander() -> Outcome:
    """The sixth column, closed.

    Clause 5 alone gives 93 dB(A) for this valve; Equation (43) adds the
    outlet flow to the trim before Equation (24) takes the sum through the
    pipe wall, and the answer becomes the 94 the annex prints.
    """
    return numeric(
        _PRINTED_EXTERNAL_LEVEL_WITH_EXPANDER,
        round(_example(6, expander=True).external_level),
        0.0,
        unit="dB",
        places=0,
        expected_label="94 dB(A), where the trim alone gives 93",
    )


#: Annex A.3's example 7: a multipath, multistage cage of 432 passages in a
#: DN 200 line, with the fluid a vapour at 70 bar. Table 4's "Globe,
#: multihole drilled plug or cage, to open" gives the efficiency correction;
#: the annex takes the Strouhal number at the bottom of the 0,1 to 0,3 range
#: 5.4.2 prints for free jets rather than from the table.
_EXAMPLE_7 = {
    "mass_flow": 23.1,
    "inlet_pressure": 7.0e6,
    "outlet_pressure": 1.4e6,
    "inlet_density": 55.3,
    "inlet_temperature": 290.0,
    "specific_heat_ratio": 1.31,
    "molecular_mass": 19.0,
    "flow_coefficient": 81.5,
    "last_stage_area": 6.44e-3,
    "passages": 432,
    "hydraulic_diameter": 0.0025,
    "last_stage_recovery": 0.98,
    "diameter": 0.200,
    "wall_thickness": 0.008,
    "pipe_density": 8000.0,
    "efficiency_correction": -4.8,
    "strouhal_number": 0.1,
}

#: Table A.2's printed intermediates for it.
_PRINTED_EXAMPLE_7 = {
    "C_n": 315.0,
    "x": 0.334,
    "p_vc": 1371038.0,
    "F_d": 0.028,
    "W_a": 10.3,
    "L_pi": 156.9,
    "f_p": 14381.0,
    "L_pAe": 89.0,
}


def _example_seven() -> ph.noise_control.AerodynamicValveNoise:
    """Example 7, through Clause 6's substitution and then Clause 5."""
    case = _EXAMPLE_7
    conditions = ph.noise_control.multistage_trim_conditions(
        inlet_pressure=case["inlet_pressure"],
        outlet_pressure=case["outlet_pressure"],
        inlet_density=case["inlet_density"],
        flow_coefficient=case["flow_coefficient"],
        last_stage_coefficient=ph.noise_control.last_stage_flow_coefficient(
            case["last_stage_area"]
        ),
    )
    area = case["last_stage_area"] / case["passages"]
    modifier = ph.noise_control.valve_style_modifier(
        area, 4.0 * area / case["hydraulic_diameter"], int(case["passages"])
    )
    return ph.noise_control.valve_aerodynamic_noise(
        mass_flow=case["mass_flow"],
        inlet_pressure=conditions.stagnation_pressure,
        outlet_pressure=case["outlet_pressure"],
        inlet_density=conditions.stagnation_density,
        inlet_temperature=case["inlet_temperature"],
        specific_heat_ratio=case["specific_heat_ratio"],
        molecular_mass=case["molecular_mass"],
        flow_coefficient=conditions.flow_coefficient,
        style_modifier=modifier,
        pressure_recovery=case["last_stage_recovery"],
        valve_outlet_diameter=case["diameter"],
        internal_diameter=case["diameter"],
        wall_thickness=case["wall_thickness"],
        pipe_density=case["pipe_density"],
        efficiency_correction=case["efficiency_correction"],
        strouhal_number=case["strouhal_number"],
    )


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Multistage trim substitution, example 7 (Eqs. (27) to (29))",
)
def _chk_multistage_conditions() -> Outcome:
    """What Clause 6 hands Clause 5 in place of the valve inlet.

    A multistage trim drops most of its pressure before the stage that makes
    the noise, so 6.3 runs Clause 5 on that stage. NOTE 3 to Equation (28)
    picks the branch: with a valve pressure ratio of five it tries (28a)
    first, and here the answer is 1,5 times the outlet pressure, which is
    below the 2 that would send it to (28b).
    """
    conditions = ph.noise_control.multistage_trim_conditions(
        inlet_pressure=_EXAMPLE_7["inlet_pressure"],
        outlet_pressure=_EXAMPLE_7["outlet_pressure"],
        inlet_density=_EXAMPLE_7["inlet_density"],
        flow_coefficient=_EXAMPLE_7["flow_coefficient"],
        last_stage_coefficient=ph.noise_control.last_stage_flow_coefficient(
            _EXAMPLE_7["last_stage_area"]
        ),
    )
    return numeric(
        _PRINTED_EXAMPLE_7["C_n"],
        round(conditions.flow_coefficient),
        0.5,
        places=0,
        expected_label="C_n = 315 from Equation (27), then (28a) for p_n",
    )


@register(
    _IEC60534,
    "IEC 60534-8-3:2010",
    "Multipath multistage trim, example 7 (Table A.2)",
)
def _chk_example_seven() -> Outcome:
    """Every printed intermediate of the seventh example.

    It is the only one on a low-noise trim, and the only one whose valve is
    not the six-opening cage of A.2: 432 passages of 15 mm² each, a hydraulic
    diameter of 2,5 mm, and a jet 2,2 mm across.
    """
    found = _example_seven()
    computed = {
        "x": round(found.pressure_ratio, 3),
        "p_vc": round(found.vena_contracta_pressure),
        "W_a": round(found.sound_power, 1),
        "L_pi": round(found.internal_level, 1),
        "f_p": round(found.peak_frequency),
        "L_pAe": round(found.external_level),
    }
    expected = {
        name: value for name, value in _PRINTED_EXAMPLE_7.items() if name in computed
    }
    return record(expected, computed)
