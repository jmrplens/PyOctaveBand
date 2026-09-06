#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Control valve hydrodynamic noise (IEC 60534-8-4), against its own Annex A.

Annex A of IEC 60534-8-4:2005 prints three operating points of one DN 100
globe valve on water and, beside each equation number, every intermediate it
produced: the pressure ratios, the jet, the two acoustical efficiencies, the
sound power, the internal level, both peak frequencies, the transmission
losses and the level 1 m outside, plus the whole frequency route evaluated at
8 kHz. The first column is turbulent, the second cavitates, and the third
repeats the second with the characteristic pressure ratio shifted by 0,1 to
show what that uncertainty costs: 14 dB.

Oracle: Annex A, A.1 and Table A.1 (printed folios 21 to 25, PDF pages 23 to
27) of BS EN 60534-8-4:2005, which endorses IEC 60534-8-4:2005 without
modification.

Four printed defects sit inside this oracle, all recorded in
``docs/ERRATA.md``:

* Equation (12) is printed one way in Clause 5.1 (``0,02 F_L^2 C``) and
  another in Table A.1 (``0,036 F_L^2 C F_d^0,75``). Only the second
  reproduces the annex's own ``N_Str = 0,399``, and it is the one used here.
* Row (22a) prints ``TL(8 000 Hz) = 51,76 dB`` without its minus sign.
* Row (17) prints ``TL_cav`` values its own printed intermediates do not
  give, by 0,06 to 0,08 dB, which is why that row alone carries a tolerance
  of 0,1 dB.
* Column 1's ``f_p,turb`` is printed 494,5 Hz where the unrounded chain gives
  494,64 Hz; columns 2 and 3 reproduce to the last printed digit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import phonometry as ph

from ..registry import Outcome, numeric, record, register

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phonometry.noise_control import HydrodynamicValveNoise

_IEC60534_8_4 = "Control valve noise (IEC 60534-8-4)"

#: A.1's given data, shared by the three columns, in SI units.
_COMMON: dict[str, Any] = {
    "inlet_pressure": 1.0e6,
    "vapour_pressure": 2.32e3,
    "liquid_density": 997.0,
    "liquid_sound_speed": 1400.0,
    "flow_coefficient": 90.0,
    "style_modifier": 0.42,
    "pressure_recovery": 0.92,
    "power_ratio": 0.25,
    "valve_diameter": 0.1,
    "seat_diameter": 0.1,
    "internal_diameter": 0.1071,
    "wall_thickness": 0.0036,
    "pipe_density": 7800.0,
}

#: Equation (3a) for this valve; the annex prints 0,2543.
_INCIPIENT = ph.noise_control.incipient_cavitation_ratio(90.0, 0.42, 0.92)

#: The per-column data. The third column is the second with the annex's own
#: "Calculation with x_Fz = x_Fz + 0,1".
_EXAMPLES: dict[int, dict[str, Any]] = {
    1: {"mass_flow": 30.0, "outlet_pressure": 8.0e5, "shift": 0.0},
    2: {"mass_flow": 40.0, "outlet_pressure": 6.5e5, "shift": 0.0},
    3: {"mass_flow": 40.0, "outlet_pressure": 6.5e5, "shift": 0.1},
}

#: The band Table A.1 evaluates the frequency route at, in Hz.
_BAND_HZ = 8000.0


def _example(index: int) -> HydrodynamicValveNoise:
    """Column ``index`` of Table A.1, one to three."""
    case = dict(_EXAMPLES[index])
    shift = case.pop("shift")
    return ph.noise_control.valve_hydrodynamic_noise(
        **_COMMON, **case, incipient_ratio=_INCIPIENT + shift
    )


def _cavitating(value: float | None, name: str) -> float:
    """One of the cavitating fields of a column Annex A prints as cavitating.

    The dataclass leaves them ``None`` in the turbulent regime, and only the
    first column is turbulent; this says so once instead of at every use.
    """
    if value is None:  # pragma: no cover - columns 2 and 3 cavitate
        msg = f"{name} is None on a column Annex A prints as cavitating."
        raise RuntimeError(msg)
    return value


def _at_band(index: int, attribute: str) -> float:
    """One band array of column ``index``, read at 8 kHz."""
    found = _example(index)
    band = int(np.argmin(np.abs(found.frequency - _BAND_HZ)))
    return float(getattr(found, attribute)[band])


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Flow conditions, examples 1 to 3 (5.1)",
)
def _chk_regimes() -> Outcome:
    """Each column lands in the regime the annex prints for it.

    The test is on the differential against ``x_Fzp1 (p_1 - p_v)``, printed
    as 2,38 x 10^5 Pa for the first two columns and 3,32 x 10^5 Pa for the
    third, which is what moves column 3 off column 2's answer.
    """
    computed = {}
    for i in _EXAMPLES:
        found = _example(i)
        threshold = found.corrected_ratio * (1.0e6 - 2.32e3)
        computed[f"x_Fzp1 (p1-pv) {i}"] = round(threshold / 1.0e5, 2)
        computed[f"cavitating {i}"] = float(found.regime == "cavitating")
    expected = {
        "x_Fzp1 (p1-pv) 1": 2.38,
        "cavitating 1": 0.0,
        "x_Fzp1 (p1-pv) 2": 2.38,
        "cavitating 2": 1.0,
        "x_Fzp1 (p1-pv) 3": 3.32,
        "cavitating 3": 1.0,
    }
    return record(
        expected,
        computed,
        label=(
            "Delta p = 2,0 / 3,5 / 3,5 x 1e5 Pa against 2,38 / 2,38 / 3,32 "
            "x 1e5 Pa -> turbulent, cavitating, cavitating"
        ),
    )


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Characteristic pressure ratio x_Fz and x_Fzp1 (Eqs. (3a), (3c))",
)
def _chk_incipient_ratios() -> Outcome:
    """The threshold, estimated and then corrected to the inlet pressure."""
    computed = {
        "x_Fz": round(_INCIPIENT, 4),
        "x_Fzp1": round(_example(1).corrected_ratio, 4),
        "x_Fzp1 shifted": round(_example(3).corrected_ratio, 4),
    }
    expected = {"x_Fz": 0.2543, "x_Fzp1": 0.2386, "x_Fzp1 shifted": 0.3324}
    return record(expected, computed)


@register(_IEC60534_8_4, "IEC 60534-8-4:2005", "Jet diameter D_j (Eq. (4))")
def _chk_jet_diameter() -> Outcome:
    """Equation (4), the same in all three columns."""
    return numeric(0.01758, _example(1).jet_diameter, 5e-6, unit="m", places=5)


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Vena contracta velocity and stream power (Eqs. (5), (6))",
)
def _chk_velocity_and_power() -> Outcome:
    """Equations (5) and (6) across the three columns.

    Columns 2 and 3 share them: the two differ only in the threshold, and
    the threshold does not enter the velocity.
    """
    computed = {
        "U_vc 1": round(_example(1).velocity, 3),
        "U_vc 2": round(_example(2).velocity, 3),
        "W_m 1": round(_example(1).stream_power, 2),
        "W_m 2": round(_example(2).stream_power, 1),
    }
    expected = {"U_vc 1": 21.772, "U_vc 2": 28.801, "W_m 1": 6018.05, "W_m 2": 14042.1}
    return record(expected, computed)


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Acoustical efficiencies (Eqs. (8), (9))",
)
def _chk_efficiencies() -> Outcome:
    """The turbulent efficiency in all three columns and the cavitating one
    in the two that cavitate.

    Column 3's ``eta_cav`` is two decades below column 2's on the same flow:
    the shifted threshold is all that separates them, and Equation (9) is a
    fifth power of how far past it the valve is.
    """
    cavitating_two = _cavitating(
        _example(2).cavitation_efficiency, "eta_cav of example 2"
    )
    cavitating_three = _cavitating(
        _example(3).cavitation_efficiency, "eta_cav of example 3"
    )
    computed = {
        "eta_turb 1 (x1e6)": round(_example(1).turbulent_efficiency * 1e6, 3),
        "eta_turb 2 (x1e6)": round(_example(2).turbulent_efficiency * 1e6, 3),
        "eta_cav 2 (x1e6)": round(cavitating_two * 1e6, 3),
        "eta_cav 3 (x1e8)": round(cavitating_three * 1e8, 3),
    }
    expected = {
        "eta_turb 1 (x1e6)": 1.555,
        "eta_turb 2 (x1e6)": 2.057,
        "eta_cav 2 (x1e6)": 1.243,
        "eta_cav 3 (x1e8)": 1.992,
    }
    return record(expected, computed)


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Sound power W_a, examples 1 to 3 (Eqs. (7a), (7b))",
)
def _chk_sound_power() -> Outcome:
    """The turbulent branch in column 1 and the cavitating one in 2 and 3.

    Each carries Table 2's acoustic power ratio of 0,25, the share a globe
    valve radiates into the pipe rather than losing in its body.
    """
    computed = {f"example {i}": round(_example(i).sound_power, 5) for i in _EXAMPLES}
    expected = {"example 1": 0.00234, "example 2": 0.01158, "example 3": 0.00729}
    return record(expected, computed, unit="W")


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Internal level at the pipe wall, examples 1 to 3 (Eq. (10))",
)
def _chk_internal_level() -> Outcome:
    """Equation (10) across the three columns.

    All three are near 150 dB inside the pipe, which is what the water's
    impedance does to a milliwatt of sound power.
    """
    computed = {f"example {i}": round(_example(i).internal_level, 3) for i in _EXAMPLES}
    expected = {"example 1": 149.596, "example 2": 156.543, "example 3": 154.532}
    return record(expected, computed, unit="dB")


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Strouhal number and turbulent peak (Eqs. (11), (12))",
)
def _chk_turbulent_peak() -> Outcome:
    """Equation (12) in the form Table A.1 prints it, then Equation (11).

    Column 1's peak is printed 494,5 Hz where the unrounded chain gives
    494,64; columns 2 and 3 are exact to the last printed digit, so they are
    the ones pinned here.
    """
    computed = {
        "N_Str 2": round(_example(2).strouhal_number, 3),
        "N_Str 3": round(_example(3).strouhal_number, 3),
        "f_p,turb 2": round(_example(2).turbulent_peak, 2),
        "f_p,turb 3": round(_example(3).turbulent_peak, 2),
    }
    expected = {
        "N_Str 2": 0.399,
        "N_Str 3": 0.243,
        "f_p,turb 2": 654.35,
        "f_p,turb 3": 397.93,
    }
    return record(expected, computed)


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Cavitating peak frequency (Eq. (13))",
)
def _chk_cavitation_peak() -> Outcome:
    """Equation (13) in the two cavitating columns.

    The shifted threshold nearly doubles it, from 1 089 Hz to 1 973 Hz, on
    exactly the same flow: cavitation just past its onset is a hiss.
    """
    computed = {
        "example 2": round(
            _cavitating(_example(2).cavitation_peak, "f_p,cav of example 2"), 2
        ),
        "example 3": round(
            _cavitating(_example(3).cavitation_peak, "f_p,cav of example 3"), 2
        ),
    }
    expected = {"example 2": 1088.94, "example 3": 1973.43}
    return record(expected, computed, unit="Hz")


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Ring frequency and its transmission loss (Eqs. (14), (15))",
)
def _chk_pipe() -> Outcome:
    """The two properties of the pipe, shared by the three columns."""
    computed = {
        "f_r": round(_example(1).pipe_ring_frequency, 3),
        "TL_fr": round(_example(1).reference_transmission_loss, 2),
    }
    expected = {"f_r": 14860.406, "TL_fr": -44.71}
    return record(expected, computed)


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Turbulent transmission loss, examples 1 to 3 (Eqs. (16a), (16b))",
)
def _chk_turbulent_loss() -> Outcome:
    """Equation (16b) at each column's peak frequency, added to (15).

    The peak frequencies are all far below the pipe's 14,9 kHz ring, so the
    correction costs another 27 to 31 dB on top of the 44,7 the mass law
    already gives.

    The annex adds its own rounded intermediates here - -44,71 and -29,56
    make its -74,27 - where the unrounded chain gives -74,263. That is the
    whole of the 0,008 dB this row is allowed.
    """
    printed = {1: -74.27, 2: -71.84, 3: -76.16}
    _deviation, index = max(
        (abs(_example(i).turbulent_transmission_loss - value), i)
        for i, value in printed.items()
    )
    return numeric(
        printed[index],
        _example(index).turbulent_transmission_loss,
        0.01,
        unit="dB",
        places=3,
        expected_label=f"example {index}, the worst of the three",
    )


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Cavitating transmission loss, examples 2 and 3 (Eq. (17))",
)
def _chk_cavitating_loss() -> Outcome:
    """Equation (17), the one row of Table A.1 whose printed values its own
    printed intermediates do not reproduce.

    The annex prints -62,917 and -75,006; the equation with the numbers
    printed beside them gives -62,86 and -74,93. The offset is under 0,1 dB
    and does not reach the printed external levels, which round to the same
    figures either way.
    """

    def loss(index: int) -> float:
        return _cavitating(
            _example(index).cavitation_transmission_loss,
            f"TL_cav of example {index}",
        )

    worst = max(
        (abs(loss(i) - printed), i, printed)
        for i, printed in ((2, -62.917), (3, -75.006))
    )
    _deviation, index, printed = worst
    return numeric(
        printed,
        loss(index),
        0.1,
        unit="dB",
        places=3,
        expected_label=f"example {index}, the worse of the two printed rows",
    )


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Level 1 m from the pipe wall, examples 1 to 3 (Eqs. (18a), (18b))",
)
def _chk_external_level() -> Outcome:
    """The end of the method in all three columns.

    Columns 2 and 3 are the same valve at the same flow with a threshold
    0,1 apart, and they answer 81,0 and 66,9 dB: the 14 dB the closing prose
    of Annex A calls a significant prediction error.
    """
    computed = {f"example {i}": round(_example(i).external_level, 1) for i in _EXAMPLES}
    expected = {"example 1": 62.7, "example 2": 81.0, "example 3": 66.9}
    return record(expected, computed, unit="dB")


@register(
    _IEC60534_8_4,
    "IEC 60534-8-4:2005",
    "Frequency route at 8 kHz, examples 1 to 3 (Eqs. (19) to (22))",
)
def _chk_band_route() -> Outcome:
    """The whole band route of 5.4, at the one band Table A.1 evaluates.

    The band transmission loss is printed 51,76 dB without a minus sign in
    all three columns; only -51,76 reproduces the external levels printed
    two rows below it, which is what this row pins.
    """
    computed = {
        "L_pi(8k) 1": round(_at_band(1, "band_internal_level"), 1),
        "L_pi(8k) 2": round(_at_band(2, "band_internal_level"), 1),
        "L_pi(8k) 3": round(_at_band(3, "band_internal_level"), 1),
        "TL(8k)": round(_at_band(1, "band_transmission_loss"), 2),
        "L_pe(8k) 1": round(_at_band(1, "band_external_level"), 1),
        "L_pe(8k) 2": round(_at_band(2, "band_external_level"), 1),
        "L_pe(8k) 3": round(_at_band(3, "band_external_level"), 1),
    }
    expected = {
        "L_pi(8k) 1": 116.3,
        "L_pi(8k) 2": 141.9,
        "L_pi(8k) 3": 128.0,
        "TL(8k)": -51.76,
        "L_pe(8k) 1": 51.8,
        "L_pe(8k) 2": 77.4,
        "L_pe(8k) 3": 63.6,
    }
    return record(expected, computed, unit="dB")
