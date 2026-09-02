#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Sound power radiated into a duct by a fan, in-duct method: ISO 5136:2003.

A ducted fan does not radiate into a room: what leaves it travels down the
duct, so ISO 5136 measures it *inside* the duct. The fan is connected to an
anechoically terminated test duct on its inlet and/or outlet side, and a
microphone in that duct samples the one-third-octave sound pressure level at
three circumferential positions (clause 6.2.2), by multiplexing, or over one
continuous revolution (clauses 7.2.3 and 7.2.4). Three things stand between
that reading and the sound power. The microphone sits in a mean flow that
adds turbulent pressure fluctuations, so it is shielded by a sampling tube,
a nose cone or a foam ball; the shield has a frequency response of its own;
and above the first cut-on the duct carries higher-order modes to which a
sampling tube does not respond as it does to a plane wave. Clause 8 gathers
the three into one combined correction :math:`C` on the averaged level
(equations (9), (10) and (11)):

.. math::

   \overline{L_p} = 10 \lg\!\left[\frac{1}{n}\sum_{i=1}^{n} 10^{0.1 L_{pi}}\right]
   \mathrm{dB} + C, \qquad C = C_1 + C_2 + C_{3,4}
   \tag{Eqs 9, 10}

where :math:`C_1` is the microphone free-field correction from the
manufacturer's data, :math:`C_2` the frequency response correction of the
shield measured per clause 5.3.3.2 c) or 5.3.4.2, and :math:`C_{3,4}` the
combined mean flow velocity and modal correction. For a level already averaged
by multiplexing or a traverse, :math:`\overline{L_p} = \overline{L_{pm}} + C`
(equation (11)). The sound power then follows from the plane-wave relation of
clause 8.2:

.. math::

   L_W = \overline{L_p} + \left(10 \lg\frac{S}{S_0}
   - 10 \lg\frac{\rho c}{(\rho c)_0}\right) \mathrm{dB},
   \qquad S = \frac{\pi d^2}{4}, \quad S_0 = 1~\mathrm{m^2},
   \quad (\rho c)_0 = 400~\mathrm{N \cdot s/m^3}
   \tag{Eq. 12}

:math:`C_{3,4}` is the part of the standard that is not arithmetic. For the
sampling tube it is a polynomial in the mean flow velocity :math:`U`, in
metres per second, negative on the inlet side and positive on the outlet side
(clause 5.3.3.4, equation (7)):

.. math::

   C_{3,4} = \sum_{i=0}^{10} a_i U^i \tag{Eq. 7}

whose coefficients :math:`a_i` Annex A tabulates per one-third-octave band
and per range of test-duct diameter (Tables A.1 to A.6, 0,15 m to 2 m), an
empty cell being zero. The coefficients are normative for 50 Hz to 10 kHz and
:math:`|U| \le 40` m/s. The footnote of every table then adds two informative
extensions which do not combine: within 50 Hz to 10 kHz the rows also hold for
:math:`40 < |U| \le 60` m/s, while the rows for 12,5 kHz to 20 kHz are given
for :math:`|U| \le 40` m/s only, and carry their own ":math:`|U| \le 40` m/s"
band header in the print. A velocity beyond 40 m/s is therefore refused as
soon as a band above 10 kHz is asked for. For the omni-directional nose cone
and foam ball no modal data exist, and clause 5.3.4.3 replaces the polynomial
by the frequency-independent convective term (equation (8)):

.. math::

   C_{3,4} = 10 \lg \frac{1}{(1 - U/c)^2}~\mathrm{dB} \tag{Eq. 8}

with :math:`c` the speed of sound, which clause 5.3.4.3 prints as 340 m/s
"under normal conditions". A whole determination knows the duct air, so
:func:`sound_power_in_duct` evaluates Eq. (8) with the :math:`c` its
``temperature`` gives, the "speed of sound in the test duct" of Table 1;
the 340 m/s is the default of :func:`flow_modal_correction` called on its own.

The A-weighted sound power level is the energy sum of the band levels with the
:math:`C_j` of Table C.1 (Annex C, equation (C.1)), and the uncertainty to
be recorded is the reproducibility of Table 2, :math:`\sigma_R` per band for
the sampling tube, expanded to :math:`2\sigma_R` at 95 % coverage (clause 9.2).
Above 10 kHz the standard suggests the extrapolated values of Table 3 without
making them part of itself.

What is *not* a term of :math:`L_W`, and is therefore not computed here, is
the qualification of the facility and the instrument: the reflection
coefficient of the anechoic termination (Table 5, Annex F), the directivity
of the sampling tube (equation (6), Table 6), the signal-to-noise ratio
against turbulence (Annex B) and the duct geometry (clause 5.2). The
informative Annexes H and I extend the coefficient tables below 0,15 m and
above 2 m; the standard's own scope stops at those diameters and so does this
module.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._internal.levels_math import energy_mean, energy_sum
from .._internal.validation import (
    require_choice,
    require_finite_fields,
    require_per_band,
    require_positive,
    require_positive_array,
    require_ranks,
    require_same_length,
)
from ._shared import _S0, SoundPowerWarning
from .sound_power_reverberation import _speed_of_sound

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

__all__ = [
    "InDuctSoundPowerResult",
    "flow_modal_correction",
    "in_duct_reproducibility",
    "sound_power_in_duct",
]

#: The three microphone shields of clause 5.3 and their scope limits (1.1).
MicrophoneShield = Literal["sampling-tube", "nose-cone", "foam-ball"]

_SHIELDS: tuple[str, ...] = ("sampling-tube", "nose-cone", "foam-ball")

#: Reference characteristic impedance (rho c)_0 of Eq. (12), N s/m^3 (8.2).
_RHO_C_0 = 400.0
#: Speed of sound "under normal conditions" the standard states beside
#: Eq. (8) and Eq. (3), m/s (5.3.4.3, 3.10).
_C_NORMAL = 340.0
#: Specific gas constant of dry air, J/(kg K): the fluid density rho of
#: Eq. (12) from the static pressure and temperature of the duct air.
_R_DRY_AIR = 287.05
#: Celsius to kelvin.
_KELVIN = 273.15
#: Test-duct diameter range of the method, m (clause 1.1; Annexes H and I
#: extend it informatively and are not implemented).
_DUCT_DIAMETER_MIN = 0.15
_DUCT_DIAMETER_MAX = 2.0
#: Air temperature range of the method, degrees Celsius (clause 1.1).
_TEMPERATURE_MIN_C = -50.0
_TEMPERATURE_MAX_C = 70.0
#: Maximum mean flow velocity at the microphone head per shield, m/s
#: (clause 1.1). The sampling tube's is the normative limit of the Annex A
#: coefficients too; beyond it, up to 60 m/s, they are given for information
#: only (5.3.3.4 NOTE and the footnote of Tables A.1 to A.6).
_MAX_VELOCITY: dict[str, float] = {
    "sampling-tube": 40.0,
    "nose-cone": 20.0,
    "foam-ball": 15.0,
}
_SAMPLING_TUBE_INFORMATIVE_MAX_VELOCITY = 60.0
#: Highest one-third-octave band of the standard, Hz (3.8, clause 4): above it
#: the coefficients and the uncertainty are for information only.
_NORMATIVE_MAX_HZ = 10000.0
#: Least number of circumferential microphone positions (6.2.2, 8.1).
_MIN_POSITIONS = 3
#: Coverage factor of the expanded uncertainty (9.2, clause 4 NOTE 2).
_COVERAGE_FACTOR = 2.0

#: Table C.1: the A-weighting C_j of the 27 nominal one-third-octave bands,
#: j = 1 (50 Hz) to j_max = 27 (20 kHz), dB, "according to IEC 60651".
_TABLE_C1: dict[int, float] = {
    50: -30.2,
    63: -26.2,
    80: -22.5,
    100: -19.1,
    125: -16.1,
    160: -13.4,
    200: -10.9,
    250: -8.6,
    315: -6.6,
    400: -4.8,
    500: -3.2,
    630: -1.9,
    800: -0.8,
    1000: 0.0,
    1250: 0.6,
    1600: 1.0,
    2000: 1.2,
    2500: 1.3,
    3150: 1.2,
    4000: 1.0,
    5000: 0.5,
    6300: -0.1,
    8000: -1.1,
    10000: -2.5,
    12500: -4.3,
    16000: -6.6,
    20000: -9.3,
}
#: The band axis of the standard, in the order Table C.1 numbers it.
_NOMINAL_BANDS: tuple[int, ...] = tuple(_TABLE_C1)

#: Table 2: standard deviation of reproducibility sigma_R of the sampling tube
#: per one-third-octave band, dB (clause 4). "80 to 100" and "125 to 4 000"
#: are printed as ranges; they are unrolled here band by band.
_TABLE_2_SIGMA_R: dict[int, float] = {
    50: 3.5,
    63: 3.0,
    80: 2.5,
    100: 2.5,
    125: 2.0,
    160: 2.0,
    200: 2.0,
    250: 2.0,
    315: 2.0,
    400: 2.0,
    500: 2.0,
    630: 2.0,
    800: 2.0,
    1000: 2.0,
    1250: 2.0,
    1600: 2.0,
    2000: 2.0,
    2500: 2.0,
    3150: 2.0,
    4000: 2.0,
    5000: 2.5,
    6300: 3.0,
    8000: 3.5,
    10000: 4.0,
}
#: Table 3: the extrapolated sigma_R above 10 kHz, dB, which clause 4
#: "suggests" while saying those bands are not part of the standard.
_TABLE_3_SIGMA_R: dict[int, float] = {12500: 4.5, 16000: 5.0, 20000: 5.5}

# --- Annex A: coefficients a_i of Eq. (7) for the sampling tube -------------
# One tuple per printed row, (band, (a0, a1, ..., a_k)); the coefficients the
# print leaves empty are zero and are simply not listed. The first row of each
# table is printed "<= f" and serves every band at or below f. Values are as
# printed, comma decimal read as a point, "x 10-02" read as e-02.
_Rows = tuple[tuple[int, tuple[float, ...]], ...]

#: Table A.1: 0,15 m <= d < 0,2 m. The 800 Hz row prints no a0 between an
#: a0 of -5,00e-02 at 630 Hz and -2,09e-02 at 1 kHz; the footnote makes it
#: zero and that is what is used.
_TABLE_A1: _Rows = (
    (630, (-5.00e-02, 2.70e-02)),
    (800, (0.0, 2.97e-02)),
    (1000, (-2.09e-02, 2.85e-02, 1.18e-04)),
    (1250, (8.41e-01, 3.61e-02, 9.34e-05)),
    (1600, (7.79e-01, 5.01e-02, 1.38e-04)),
    (2000, (7.67e-01, 5.45e-02, 3.77e-04)),
    (2500, (1.59, 6.12e-02, 5.06e-04)),
    (3150, (2.40, 8.26e-02, 7.45e-04, -3.02e-06)),
    (4000, (3.43, 9.99e-02, 9.61e-04, -3.29e-06)),
    (5000, (3.98, 1.29e-01, 2.21e-03, -8.88e-06, -2.32e-07)),
    (6300, (4.87, 1.59e-01, 3.43e-03, -1.73e-05, -5.12e-07)),
    (8000, (6.09, 2.04e-01, 6.57e-03, -5.09e-05, -2.47e-06, 5.89e-09, 3.32e-10)),
    (
        10000,
        (
            6.95,
            2.54e-01,
            1.12e-02,
            -1.19e-04,
            -7.88e-06,
            3.39e-08,
            2.52e-09,
            -3.22e-12,
            -2.85e-13,
        ),
    ),
    (12500, (8.06, 3.04e-01, 1.68e-02, -2.06e-04, -1.59e-05, 6.99e-08, 5.07e-09)),
    (
        16000,
        (
            9.25,
            3.71e-01,
            2.75e-02,
            -4.42e-04,
            -4.90e-05,
            3.74e-07,
            3.73e-08,
            -1.06e-10,
            -9.89e-12,
        ),
    ),
    (
        20000,
        (
            1.06e01,
            4.46e-01,
            4.08e-02,
            -7.79e-04,
            -1.21e-04,
            1.25e-06,
            1.63e-07,
            -8.86e-10,
            -9.97e-11,
            2.21e-13,
            2.25e-14,
        ),
    ),
)

#: Table A.2: 0,2 m <= d < 0,3 m.
_TABLE_A2: _Rows = (
    (630, (-5.00e-02, 2.70e-02)),
    (800, (1.36e-01, 3.30e-02)),
    (1000, (1.75e-01, 4.08e-02)),
    (1250, (-3.32e-02, 4.32e-02, 1.35e-04)),
    (1600, (5.43e-01, 4.92e-02, 1.89e-04)),
    (2000, (1.29, 5.80e-02, 3.01e-04)),
    (2500, (1.91, 6.93e-02, 4.60e-04)),
    (3150, (2.64, 9.00e-02, 8.73e-04, -4.13e-06)),
    (4000, (3.88, 1.07e-01, 1.15e-03, -6.03e-06)),
    (5000, (4.50, 1.29e-01, 2.55e-03, -1.03e-05, -2.75e-07)),
    (6300, (5.54, 1.52e-01, 3.93e-03, -1.68e-05, -6.36e-07)),
    (8000, (6.85, 1.89e-01, 7.37e-03, -4.51e-05, -3.13e-06, 6.10e-09, 4.34e-10)),
    (
        10000,
        (
            7.82,
            2.29e-01,
            1.17e-02,
            -8.27e-05,
            -9.21e-06,
            2.52e-08,
            3.00e-09,
            -2.62e-12,
            -3.39e-13,
        ),
    ),
    (12500, (9.04, 2.75e-01, 1.56e-02, -1.07e-04, -1.70e-05, 3.13e-08, 5.71e-09)),
    (
        16000,
        (
            1.02e01,
            3.49e-01,
            2.26e-02,
            -1.94e-04,
            -4.60e-05,
            1.05e-07,
            3.74e-08,
            -2.33e-11,
            -1.02e-11,
        ),
    ),
    (
        20000,
        (
            1.18e01,
            4.59e-01,
            1.81e-02,
            -4.24e-04,
            -3.60e-05,
            3.70e-07,
            3.06e-08,
            -1.94e-10,
            -8.76e-12,
            4.09e-14,
        ),
    ),
)

#: Table A.3: 0,3 m <= d < 0,5 m.
_TABLE_A3: _Rows = (
    (400, (-5.00e-02, 2.70e-02)),
    (500, (-3.91e-01, 3.13e-02)),
    (630, (-6.13e-01, 3.32e-02)),
    (800, (-4.78e-01, 3.57e-02)),
    (1000, (-2.06e-01, 4.07e-02)),
    (1250, (3.80e-01, 4.71e-02, 8.89e-05)),
    (1600, (8.58e-01, 5.33e-02, 1.87e-04)),
    (2000, (1.58, 6.06e-02, 3.34e-04)),
    (2500, (2.46, 7.49e-02, 5.64e-04, -3.11e-06)),
    (3150, (3.51, 8.64e-02, 9.06e-04, -4.39e-06)),
    (4000, (4.75, 9.80e-02, 1.69e-03, -4.85e-06, -1.45e-07)),
    (5000, (5.62, 1.14e-01, 2.59e-03, -4.34e-06, -3.56e-07)),
    (6300, (6.77, 1.44e-01, 3.17e-03, -6.85e-06, -6.10e-07)),
    (8000, (8.09, 1.88e-01, 4.88e-03, -1.37e-05, -2.27e-06, -1.03e-09, 3.36e-10)),
    (10000, (9.12, 2.59e-01, 4.51e-03, -6.07e-05, -2.12e-06, 7.03e-09, 3.47e-10)),
    (12500, (9.84, 3.38e-01, 7.94e-03, -1.53e-04, -7.19e-06, 3.21e-08, 2.40e-09)),
    (
        16000,
        (
            1.08e01,
            4.47e-01,
            9.42e-03,
            -4.61e-04,
            -7.86e-06,
            3.02e-07,
            2.35e-09,
            -6.92e-11,
        ),
    ),
    (
        20000,
        (
            1.17e01,
            5.24e-01,
            1.74e-02,
            -7.12e-04,
            -2.95e-05,
            6.27e-07,
            2.18e-08,
            -1.91e-10,
            -5.64e-12,
        ),
    ),
)

#: Table A.4: 0,5 m <= d < 0,8 m; the table Annex D exercises.
_TABLE_A4: _Rows = (
    (250, (-5.00e-02, 2.70e-02)),
    (315, (-6.50e-01, 2.89e-02)),
    (400, (-4.36e-01, 3.01e-02)),
    (500, (-3.12e-01, 3.09e-02)),
    (630, (8.52e-02, 3.24e-02)),
    (800, (1.03, 3.57e-02)),
    (1000, (1.85, 3.80e-02)),
    (1250, (2.61, 4.34e-02, 1.08e-04)),
    (1600, (3.18, 5.30e-02, 1.32e-04)),
    (2000, (3.64, 6.67e-02, 1.57e-04)),
    (2500, (4.12, 8.36e-02, 2.72e-04)),
    (3150, (4.64, 1.12e-01, 6.78e-04, -6.27e-06)),
    (4000, (5.47, 1.30e-01, 1.29e-03, -8.74e-06, -1.48e-07)),
    (5000, (6.03, 1.53e-01, 1.91e-03, -1.17e-05, -2.80e-07)),
    (6300, (6.92, 1.84e-01, 2.37e-03, -1.99e-05, -3.93e-07)),
    (8000, (8.01, 2.34e-01, 4.22e-03, -5.79e-05, -1.74e-06, 7.63e-09, 2.46e-10)),
    (
        10000,
        (8.90, 2.96e-01, 4.86e-03, -1.37e-04, -2.16e-06, 4.39e-08, 3.29e-10, -5.11e-12),
    ),
    (12500, (9.57, 3.58e-01, 9.87e-03, -2.20e-04, -9.71e-06, 7.05e-08, 3.25e-09)),
    (
        16000,
        (
            1.05e01,
            4.50e-01,
            1.57e-02,
            -5.09e-04,
            -2.78e-05,
            3.98e-07,
            2.21e-08,
            -1.12e-10,
            -6.07e-12,
        ),
    ),
    (
        20000,
        (
            1.17e01,
            5.58e-01,
            1.70e-02,
            -1.01e-03,
            -2.93e-05,
            1.40e-06,
            2.26e-08,
            -9.09e-10,
            -6.11e-12,
            2.17e-13,
        ),
    ),
)

#: The one band of Annex A whose coefficient is a reading rather than a
#: transcription: see :func:`_warn_reconstructed_coefficient`.
_RECONSTRUCTED_BAND_HZ = 5000

#: Table A.5: 0,8 m <= d < 1,25 m. The a3 of the 5 000 Hz row is printed with
#: its leading digit missing, "- ,24 x 10-05"; it is read as -1,24e-05, the
#: value the neighbouring tables bracket (-1,17e-05 in A.4, -1,27e-05 in
#: A.6). Asking for that band raises a warning that says so. See
#: docs/ERRATA.md.
_TABLE_A5: _Rows = (
    (160, (-5.00e-02, 2.70e-02)),
    (200, (-1.04, 2.35e-02)),
    (250, (-7.07e-01, 2.62e-02)),
    (315, (-5.60e-01, 2.87e-02)),
    (400, (-1.10e-01, 3.01e-02)),
    (500, (6.61e-01, 3.09e-02)),
    (630, (1.34, 3.23e-02)),
    (800, (1.92, 3.72e-02)),
    (1000, (2.10, 4.33e-02)),
    (1250, (2.26, 5.37e-02)),
    (1600, (2.50, 6.30e-02, 1.33e-04)),
    (2000, (3.00, 7.07e-02, 2.66e-04)),
    (2500, (3.70, 8.07e-02, 3.91e-04)),
    (3150, (4.45, 1.05e-01, 6.32e-04, -4.55e-06)),
    (4000, (5.53, 1.28e-01, 8.01e-04, -7.67e-06)),
    (5000, (6.00, 1.54e-01, 1.74e-03, -1.24e-05, -2.32e-07)),
    (6300, (6.88, 1.92e-01, 2.33e-03, -3.11e-05, -3.94e-07, 2.69e-09)),
    (8000, (7.97, 2.37e-01, 4.25e-03, -5.96e-05, -1.78e-06, 7.91e-09, 2.57e-10)),
    (
        10000,
        (
            8.67,
            2.97e-01,
            6.89e-03,
            -1.35e-04,
            -5.29e-06,
            4.27e-08,
            1.81e-09,
            -4.89e-12,
            -2.15e-13,
        ),
    ),
    (12500, (9.56, 3.59e-01, 9.71e-03, -2.22e-04, -9.55e-06, 7.20e-08, 3.21e-09)),
    (
        16000,
        (
            1.05e01,
            4.51e-01,
            1.56e-02,
            -5.09e-04,
            -2.76e-05,
            3.97e-07,
            2.19e-08,
            -1.11e-10,
            -6.00e-12,
        ),
    ),
    (
        20000,
        (
            1.17e01,
            5.60e-01,
            1.68e-02,
            -1.02e-03,
            -2.88e-05,
            1.42e-06,
            2.22e-08,
            -9.29e-10,
            -5.98e-12,
            2.23e-13,
        ),
    ),
)

#: Table A.6: 1,25 m <= d <= 2 m.
_TABLE_A6: _Rows = (
    (100, (-5.00e-02, 2.70e-02)),
    (125, (-1.24, 2.05e-02)),
    (160, (-9.02e-01, 2.28e-02)),
    (200, (-8.46e-01, 2.42e-02)),
    (250, (-3.52e-01, 2.64e-02)),
    (315, (4.54e-01, 2.85e-02)),
    (400, (1.15, 3.02e-02)),
    (500, (1.37, 3.15e-02)),
    (630, (1.11, 3.45e-02)),
    (800, (9.80e-01, 4.11e-02)),
    (1000, (1.28, 4.53e-02)),
    (1250, (1.87, 5.17e-02)),
    (1600, (2.31, 6.08e-02, 1.33e-04)),
    (2000, (2.88, 7.08e-02, 2.39e-04)),
    (2500, (3.59, 8.22e-02, 3.70e-04)),
    (3150, (4.37, 1.06e-01, 5.76e-04, -4.46e-06)),
    (4000, (5.46, 1.27e-01, 7.93e-04, -7.43e-06)),
    (5000, (5.95, 1.55e-01, 1.73e-03, -1.27e-05, -2.32e-07)),
    (6300, (6.84, 1.93e-01, 2.32e-03, -3.10e-05, -3.93e-07, 2.62e-09)),
    (8000, (7.95, 2.38e-01, 4.21e-03, -6.04e-05, -1.77e-06, 8.08e-09, 2.56e-10)),
    (
        10000,
        (8.85, 2.97e-01, 4.82e-03, -1.36e-04, -2.16e-06, 4.31e-08, 3.31e-10, -4.96e-12),
    ),
    (12500, (9.56, 3.60e-01, 9.65e-03, -2.23e-04, -9.49e-06, 7.24e-08, 3.18e-09)),
    (
        16000,
        (
            1.05e01,
            4.52e-01,
            1.55e-02,
            -5.11e-04,
            -2.74e-05,
            3.99e-07,
            2.17e-08,
            -1.12e-10,
            -5.96e-12,
        ),
    ),
    (
        20000,
        (
            1.17e01,
            5.61e-01,
            1.67e-02,
            -1.03e-03,
            -2.86e-05,
            1.43e-06,
            2.20e-08,
            -9.34e-10,
            -5.93e-12,
            2.24e-13,
        ),
    ),
)

#: Annex A, Tables A.1 to A.6, keyed by the upper edge of their duct-diameter
#: range: a table serves the diameters below its edge and above the previous
#: one. The edges are the printed "d < 0,2", "d < 0,3", "d < 0,5", "d < 0,8",
#: "d < 1,25"; the last table is printed "d <= 2 m" and takes the top edge.
_ANNEX_A_TABLES: tuple[tuple[float, _Rows], ...] = (
    (0.2, _TABLE_A1),
    (0.3, _TABLE_A2),
    (0.5, _TABLE_A3),
    (0.8, _TABLE_A4),
    (1.25, _TABLE_A5),
    (_DUCT_DIAMETER_MAX, _TABLE_A6),
)


@dataclass(frozen=True)
class InDuctSoundPowerResult:
    r"""Result of an ISO 5136:2003 in-duct sound power determination.

    ``sound_power_level`` is the per-band :math:`L_W` of Eq. (12), and
    ``sound_power_level_a`` the A-weighted total of Annex C, Eq. (C.1), over
    the bands supplied. ``mean_pressure_level`` is the spatially averaged
    level before any correction, the bracket of Eq. (9) or the
    :math:`\overline{L_{pm}}` of Eq. (11); ``corrected_pressure_level`` is
    :math:`\overline{L_p}` after the combined correction. The three corrections
    are kept apart so the record clause 9.1 f) asks for can be made:
    ``microphone_correction`` (:math:`C_1`), ``shield_correction``
    (:math:`C_2`) and ``flow_modal_correction`` (:math:`C_{3,4}`), with
    ``combined_correction`` their sum (Eq. 10).

    ``reproducibility_standard_deviation`` is :math:`\sigma_R` of Table 2 per
    band (Table 3 above 10 kHz) and ``expanded_uncertainty`` is twice it, the
    95 % figure clause 9.2 says to record. ``information_only_band`` marks the
    bands the standard gives for information rather than as part of itself:
    those above 10 kHz, and every band when the sampling tube is used between
    40 m/s and 60 m/s (5.3.3.4 NOTE, clause 4). Table 2 is stated for the
    sampling tube; clause 4 NOTE 5 expects the figures to grow for the other
    shields and gives no others, so the same values are reported for them.

    ``duct_diameter`` and ``duct_area`` are :math:`d` and :math:`S`,
    ``characteristic_impedance`` is the :math:`\rho c` of the duct air and
    ``speed_of_sound`` its :math:`c`, ``flow_velocity`` is the signed :math:`U`
    (negative on the inlet side) and ``shield`` names the microphone shield.
    """

    frequencies: np.ndarray
    sound_power_level: np.ndarray
    mean_pressure_level: np.ndarray
    corrected_pressure_level: np.ndarray
    microphone_correction: np.ndarray
    shield_correction: np.ndarray
    flow_modal_correction: np.ndarray
    combined_correction: np.ndarray
    reproducibility_standard_deviation: np.ndarray
    expanded_uncertainty: np.ndarray
    information_only_band: np.ndarray
    duct_diameter: float
    duct_area: float
    characteristic_impedance: float
    speed_of_sound: float
    flow_velocity: float
    shield: str
    sound_power_level_a: float

    def __post_init__(self) -> None:
        """Reject a determination whose per-band quantities disagree.

        Every per-band field is read beside ``sound_power_level``: the
        spectrum figure labels its bars with ``frequencies``, and a report of
        the corrections of clause 9.1 f) prints one row per band. A field one
        entry short raises inside whichever reader indexes it first, and one
        entry long is silently dropped, so the lengths are pinned here.

        ``shield`` is pinned because the uncertainty statement and the
        velocity scope both turn on it. The scalars, the levels and the three
        correction columns are pinned finite because every one of them reaches
        a printed sheet as a plain number and none of them has a
        not-applicable reading a NaN could stand for: ``sound_power_in_duct``
        refuses a non-finite :math:`C_1` or :math:`C_2` at the door, and
        :math:`C_{3,4}` is always a finite polynomial value.

        :raises ValueError: if a per-band field disagrees with the rest,
            ``shield`` is not one of the three of clause 5.3, or a scalar or
            a level is not finite.
        """
        require_choice(self.shield, "shield", _SHIELDS)
        per_band = (
            "frequencies",
            "sound_power_level",
            "mean_pressure_level",
            "corrected_pressure_level",
            "microphone_correction",
            "shield_correction",
            "flow_modal_correction",
            "combined_correction",
            "reproducibility_standard_deviation",
            "expanded_uncertainty",
            "information_only_band",
        )
        require_ranks(self, **dict.fromkeys(per_band, 1))
        require_same_length(self, *per_band)
        require_finite_fields(
            self,
            "frequencies",
            "sound_power_level",
            "mean_pressure_level",
            "corrected_pressure_level",
            "microphone_correction",
            "shield_correction",
            "flow_modal_correction",
            "combined_correction",
            "reproducibility_standard_deviation",
            "expanded_uncertainty",
            "duct_diameter",
            "duct_area",
            "characteristic_impedance",
            "speed_of_sound",
            "flow_velocity",
            "sound_power_level_a",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the in-duct sound power spectrum with the A-weighted total.

        One bar per one-third-octave band of ``sound_power_level``, the
        :math:`L_{W\mathrm{A}}` of Annex C in the title. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the band :meth:`~matplotlib.axes.Axes.bar`.
        :return: The axes.
        :raises ValueError: If ``language`` is unknown.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_sound_power

        check_language(language)
        return plot_sound_power(self, ax=ax, language=language, **kwargs)


def _nominal_bands(frequencies: ArrayLike) -> np.ndarray:
    """The band axis, refused unless every entry is a nominal centre.

    The coefficients of Annex A, the A-weighting of Table C.1 and the
    reproducibility of Table 2 are all keyed by the nominal one-third-octave
    centre frequency, so a band the standard does not print cannot be served
    at all. Nominal means the printed integer: 1 000 Hz, not the exact
    1 000,0 Hz or the base-ten 1 000 Hz of IEC 61260-1, which are the same
    label once rounded.

    :param frequencies: The band centres, in hertz.
    :return: The centres as a ``float64`` array.
    :raises ValueError: if a centre is not one of the 27 nominal bands from
        50 Hz to 20 kHz.
    """
    freqs = require_positive_array(frequencies, "frequencies")
    for f in freqs:
        if round(float(f)) not in _TABLE_C1:
            msg = (
                "'frequencies' must be nominal one-third-octave centre "
                "frequencies from 50 Hz to 20 000 Hz (ISO 5136 Table C.1); "
                f"got {f:g} Hz."
            )
            raise ValueError(msg)
    return freqs


def _band_keys(frequencies: np.ndarray) -> list[int]:
    """The nominal integer key of each validated band centre."""
    return [round(float(f)) for f in frequencies]


def _check_shield(shield: str) -> str:
    return require_choice(shield, "shield", _SHIELDS)


def _as_scalar(value: object, name: str) -> float:
    """Coerce one measurement to a ``float``, naming the parameter when it is not.

    The scope guards below read a single number: a duct diameter, a mean flow
    velocity, a temperature, a static pressure. Handing them a per-band array
    or a string is an ordinary caller mistake, and left to :func:`float` it
    raises a bare ``TypeError`` from numpy or the interpreter that names
    neither the parameter nor the function; every refusal of this module is a
    ``ValueError`` that names its parameter, so the coercion is made here.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        msg = f"'{name}' must be a real number."
        raise ValueError(msg) from exc


def _check_duct_diameter(duct_diameter: float) -> float:
    """The test-duct diameter, refused outside the 0,15 m to 2 m of clause 1.1.

    The informative Annexes H and I carry coefficient tables for smaller and
    larger ducts, but the standard says they are not part of itself and this
    module does not reach past its scope.
    """
    d = require_positive(_as_scalar(duct_diameter, "duct_diameter"), "duct_diameter")
    if d < _DUCT_DIAMETER_MIN or d > _DUCT_DIAMETER_MAX:
        msg = (
            "'duct_diameter' must be between 0.15 m and 2 m, the test-duct "
            f"range of ISO 5136 (clause 1.1); got {d:g} m."
        )
        raise ValueError(msg)
    return d


def _check_flow_velocity(flow_velocity: float, shield: str) -> float:
    """The signed mean flow velocity, refused beyond the shield's limit.

    Clause 1.1 caps the mean flow velocity at the microphone head at 15 m/s
    for the foam ball, 20 m/s for the nose cone and 40 m/s for the sampling
    tube. For the sampling tube alone Annex A extends its coefficients to
    60 m/s for information, so that is where the refusal sits; between 40 and
    60 the result is flagged rather than refused.
    """
    u = _as_scalar(flow_velocity, "flow_velocity")
    if not math.isfinite(u):
        msg = "'flow_velocity' must be finite."
        raise ValueError(msg)
    limit = (
        _SAMPLING_TUBE_INFORMATIVE_MAX_VELOCITY
        if shield == "sampling-tube"
        else _MAX_VELOCITY[shield]
    )
    if abs(u) > limit:
        why = (
            "Annex A gives no coefficients beyond it"
            if shield == "sampling-tube"
            else "the limit of clause 1.1 for that shield"
        )
        msg = (
            f"'flow_velocity' must satisfy |U| <= {limit:g} m/s for the "
            f"{shield.replace('-', ' ')} ({why}); got {u:g} m/s."
        )
        raise ValueError(msg)
    return u


def _check_informative_bands(frequencies: np.ndarray, flow_velocity: float) -> None:
    r"""Refuse the corner of Annex A the tables leave empty.

    Tables A.1 to A.6 are split by a band header. Over 50 Hz to 10 kHz the
    rows carry the normative :math:`|U| \le 40` m/s and, for information, the
    extended :math:`|U| \le 60` m/s; the rows for 12,5 kHz to 20 kHz sit under
    a second header that reads ":math:`|U| \le 40` m/s", and the footnote
    grants them no velocity extension: "Also for information only, values are
    given for an extended frequency range, 12 500 Hz to 20 000 Hz, for flow
    velocities :math:`|U| \le 40`".

    The two extensions therefore do not compose. Evaluated past 40 m/s the
    high-band polynomials do not merely lose accuracy, they diverge: at
    60 m/s in a 0,5 m duct they return +100 dB at 12,5 kHz, -369 dB at 16 kHz
    and +268 dB at 20 kHz. Nothing downstream can tell that apart from a
    correction, so the combination is refused at the door.

    :raises ValueError: if a band above 10 kHz is asked for at |U| > 40 m/s.
    """
    limit = _MAX_VELOCITY["sampling-tube"]
    if abs(flow_velocity) <= limit:
        return
    above = frequencies[frequencies > _NORMATIVE_MAX_HZ]
    if above.size == 0:
        return
    msg = (
        f"'flow_velocity' must satisfy |U| <= {limit:g} m/s for the "
        "one-third-octave bands above 10 kHz: ISO 5136 Annex A gives the "
        "12,5 kHz to 20 kHz coefficients for that range only (the band "
        f"header and footnote of Tables A.1 to A.6); got {flow_velocity:g} "
        f"m/s with {above[0]:g} Hz in 'frequencies'."
    )
    raise ValueError(msg)


def _annex_a_rows(duct_diameter: float) -> _Rows:
    """The Annex A table whose diameter range holds *duct_diameter*."""
    for edge, rows in _ANNEX_A_TABLES:
        if duct_diameter < edge:
            return rows
    return _ANNEX_A_TABLES[-1][1]


def _warn_reconstructed_coefficient(rows: _Rows, freqs: np.ndarray) -> None:
    """Say so when a band is answered with a coefficient that is a reading.

    One cell of Annex A is not legible: the ``a3`` of the 5 000 Hz row of
    Table A.5 is printed without its leading digit. Every other coefficient in
    the annex is transcribed; this one is reconstructed, and the caller is
    entitled to know which of the two it is holding. The reading moves the
    correction by 0,64 dB per unit of the missing digit at 40 m/s and by
    0,034 dB at 15 m/s, so it matters at the top of the velocity range and
    barely at all at the bottom. No printed value crosses this cell, because
    Table D.1 is tabulated for a diameter served by Table A.4.
    """
    if rows is not _TABLE_A5 or not np.any(freqs == _RECONSTRUCTED_BAND_HZ):
        return
    warnings.warn(
        f"The {_RECONSTRUCTED_BAND_HZ:g} Hz coefficient a3 of Table A.5 "
        "(0,8 m <= d < 1,25 m) is printed without its leading digit and is "
        "read as -1,24e-05, the value Tables A.4 and A.6 bracket; the "
        "correction returned for that band is that reading, not a "
        "transcription (see docs/ERRATA.md).",
        SoundPowerWarning,
        stacklevel=3,
    )


def _annex_a_coefficients(rows: _Rows, band: int) -> tuple[float, ...]:
    """The a_i of one band: its own row, or the "<= f" row for the lower bands."""
    lowest_band, lowest = rows[0]
    if band <= lowest_band:
        return lowest
    return dict(rows)[band]


def flow_modal_correction(
    frequencies: ArrayLike,
    flow_velocity: float,
    duct_diameter: float,
    *,
    shield: MicrophoneShield = "sampling-tube",
    speed_of_sound: float = _C_NORMAL,
) -> np.ndarray:
    r"""Combined mean flow velocity and modal correction :math:`C_{3,4}`.

    For the sampling tube, the polynomial of clause 5.3.3.4 with the
    coefficients of Annex A for the band and the test-duct diameter, an empty
    cell of the print counting as zero (Eq. (7)):

    .. math::

       C_{3,4} = \sum_{i=0}^{10} a_i U^i

    For the omni-directional nose cone and foam ball, the frequency-independent
    convective term of clause 5.3.4.3 (Eq. (8)):

    .. math::

       C_{3,4} = 10 \lg \frac{1}{(1 - U/c)^2}~\mathrm{dB}

    :math:`U` is signed: negative for an inlet-side measurement, positive on
    the outlet side (Table 1 NOTE 2), so the same speed reads as a different
    correction on the two sides of the fan.

    One cell of Annex A is not legible, the :math:`a_3` of the 5 000 Hz row of
    the table that serves 0,8 m to 1,25 m, and asking for that band and that
    diameter emits a :class:`SoundPowerWarning` saying the coefficient is a
    reading rather than a transcription (see ``docs/ERRATA.md``). Every other
    coefficient in the annex is transcribed and none of them warns.

    :param frequencies: Nominal one-third-octave centre frequencies, in
        hertz, 50 Hz to 20 kHz.
    :param flow_velocity: Mean flow velocity :math:`U` at the microphone
        position, in metres per second, negative on the inlet side.
    :param duct_diameter: Test-duct diameter :math:`d`, in metres, 0,15 m
        to 2 m; it selects the Annex A table for the sampling tube and is
        checked against the scope for the other shields.
    :param shield: ``"sampling-tube"`` (default), ``"nose-cone"`` or
        ``"foam-ball"``.
    :param speed_of_sound: The :math:`c` of Eq. (8), in metres per second;
        the 340 m/s clause 5.3.4.3 states for normal conditions by default.
        :func:`sound_power_in_duct` passes the duct air's own :math:`c`
        instead. Unused by the sampling tube.
    :return: :math:`C_{3,4}` per band, in decibels.
    :raises ValueError: for a band that is not a nominal centre, a diameter
        outside 0,15 m to 2 m, a velocity beyond the shield's limit (60 m/s
        for the sampling tube, 20 m/s for the nose cone, 15 m/s for the foam
        ball), a velocity beyond 40 m/s asked for in a band above 10 kHz,
        where Annex A tabulates no coefficients, or a non-positive speed of
        sound.
    """
    freqs = _nominal_bands(frequencies)
    _check_shield(shield)
    _check_duct_diameter(duct_diameter)
    u = _check_flow_velocity(flow_velocity, shield)
    c = require_positive(speed_of_sound, "speed_of_sound")
    if shield != "sampling-tube":
        if abs(u) >= c:
            msg = "'flow_velocity' must be smaller in magnitude than 'speed_of_sound'."
            raise ValueError(msg)
        # Eq. (8): 10 lg[1/(1 - U/c)^2], one value for every band.
        value = 10.0 * math.log10(1.0 / (1.0 - u / c) ** 2)
        return np.full(freqs.shape, value, dtype=np.float64)
    _check_informative_bands(freqs, u)
    rows = _annex_a_rows(duct_diameter)
    _warn_reconstructed_coefficient(rows, freqs)
    return np.asarray(
        [
            np.polynomial.polynomial.polyval(u, _annex_a_coefficients(rows, band))
            for band in _band_keys(freqs)
        ],
        dtype=np.float64,
    )


def in_duct_reproducibility(frequencies: ArrayLike) -> np.ndarray:
    r"""Standard deviation of reproducibility :math:`\sigma_R` per band.

    Table 2 of ISO 5136:2003 for the sampling tube, 50 Hz to 10 kHz, which is
    what clause 9.2 doubles into the expanded uncertainty to be recorded at
    95 % coverage. For 12,5 kHz, 16 kHz and 20 kHz the extrapolated values of
    Table 3 are returned; clause 4 suggests them while saying that
    measurements above 10 kHz are not part of the standard, so a result that
    carries those bands says so in ``information_only_band``.

    :param frequencies: Nominal one-third-octave centre frequencies, in
        hertz, 50 Hz to 20 kHz.
    :return: :math:`\sigma_R` per band, in decibels.
    :raises ValueError: for a band that is not a nominal centre.
    """
    freqs = _nominal_bands(frequencies)
    table = {**_TABLE_2_SIGMA_R, **_TABLE_3_SIGMA_R}
    return np.asarray([table[band] for band in _band_keys(freqs)], dtype=np.float64)


def _check_air(temperature: float, static_pressure: float) -> tuple[float, float]:
    r"""The duct air's speed of sound and characteristic impedance.

    The temperature is refused outside the -50 degC to +70 degC the method is
    stated for (clause 1.1). The speed of sound is the ISO 3741 form the
    emission package already uses, and the density is the ideal gas at the
    static pressure and temperature of the duct, so that
    :math:`\rho c` is 413 N s/m^3 at 20 degC and 101,325 kPa against the
    400 N s/m^3 reference of Eq. (12).

    :return: ``(c, rho_c)`` in m/s and N s/m^3.
    """
    theta = _as_scalar(temperature, "temperature")
    if (
        not math.isfinite(theta)
        or theta < _TEMPERATURE_MIN_C
        or theta > _TEMPERATURE_MAX_C
    ):
        msg = (
            "'temperature' must be between -50 degC and 70 degC, the air "
            f"temperature range of ISO 5136 (clause 1.1); got {temperature!r}."
        )
        raise ValueError(msg)
    ps = require_positive(
        _as_scalar(static_pressure, "static_pressure"), "static_pressure"
    )
    c = _speed_of_sound(theta)
    # kPa to Pa in the numerator; the ideal-gas density of dry air.
    rho = ps * 1000.0 / (_R_DRY_AIR * (_KELVIN + theta))
    return c, rho * c


def _position_levels(levels: ArrayLike, n_bands: int) -> np.ndarray:
    r"""The measured levels as ``(positions, bands)`` or a ``(bands,)`` average.

    A 2-D array is one row per circumferential microphone position, to be
    energy-averaged by Eq. (9); a 1-D array is a level already averaged by
    multiplexing or a continuous traverse, the :math:`\overline{L_{pm}}` of
    Eq. (11). Fewer than the three positions of clause 6.2.2 is warned about,
    not refused: the arithmetic is the same and the shortfall is a matter of
    sampling the duct, which the caller may have done another way.
    """
    try:
        arr = np.asarray(levels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = "'levels' must be numeric."
        raise ValueError(msg) from exc
    if arr.ndim not in (1, 2) or arr.size == 0:
        msg = "'levels' must be a (bands,) spectrum or a (positions, bands) array."
        raise ValueError(msg)
    if arr.shape[-1] != n_bands:
        msg = (
            f"'levels' must carry one value per band ({n_bands} in "
            f"'frequencies'); got shape {arr.shape}."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = "'levels' must contain only finite values."
        raise ValueError(msg)
    if arr.ndim == 2 and arr.shape[0] < _MIN_POSITIONS:  # noqa: PLR2004
        warnings.warn(
            f"Only {arr.shape[0]} microphone position(s) were supplied; "
            f"ISO 5136 averages at least {_MIN_POSITIONS} circumferential "
            "positions (clauses 6.2.2 and 8.1).",
            SoundPowerWarning,
            stacklevel=3,
        )
    return arr


def sound_power_in_duct(
    levels: ArrayLike,
    frequencies: ArrayLike,
    duct_diameter: float,
    flow_velocity: float,
    *,
    shield: MicrophoneShield = "sampling-tube",
    microphone_correction: ArrayLike = 0.0,
    shield_correction: ArrayLike = 0.0,
    temperature: float = 20.0,
    static_pressure: float = 101.325,
) -> InDuctSoundPowerResult:
    r"""Sound power radiated into a test duct, in-duct method (ISO 5136:2003).

    ``levels`` is either a ``(positions, bands)`` array of the time-averaged
    sound pressure level at each circumferential microphone position, energy-
    averaged by Eq. (9), or a ``(bands,)`` spectrum already averaged by
    multiplexing or a continuous traverse (Eq. (11)). The combined correction
    :math:`C = C_1 + C_2 + C_{3,4}` of Eq. (10) is added to the average, with
    :math:`C_1` and :math:`C_2` supplied per band or as a scalar and
    :math:`C_{3,4}` from :func:`flow_modal_correction`, and the plane-wave
    relation of Eq. (12) gives the sound power in each band:

    .. math::

       L_W = \overline{L_p} + 10 \lg\frac{\pi d^2 / 4}{S_0}
       - 10 \lg\frac{\rho c}{(\rho c)_0}

    The A-weighted total follows Annex C over the bands supplied, and the
    uncertainty statement of clause 9.2, twice the reproducibility of
    Table 2, is carried per band.

    :param levels: Sound pressure levels, in decibels, ``(positions, bands)``
        or an already averaged ``(bands,)`` spectrum.
    :param frequencies: Nominal one-third-octave centre frequencies of the
        bands, in hertz, 50 Hz to 20 kHz.
    :param duct_diameter: Test-duct diameter :math:`d`, in metres, 0,15 m
        to 2 m.
    :param flow_velocity: Mean flow velocity :math:`U` at the microphone
        position, in metres per second; negative on the inlet side, positive
        on the outlet side.
    :param shield: Microphone shield, ``"sampling-tube"`` (default),
        ``"nose-cone"`` or ``"foam-ball"``.
    :param microphone_correction: :math:`C_1`, the manufacturer's free-field
        correction of the microphone, in decibels, per band or scalar.
    :param shield_correction: :math:`C_2`, the frequency response correction
        of the shield determined per clause 5.3.3.2 c) or 5.3.4.2, in
        decibels, per band or scalar.
    :param temperature: Air temperature in the duct, in degrees Celsius,
        -50 degC to 70 degC; sets :math:`c` and :math:`\rho`. The :math:`c`
        it sets is also the one Eq. (8) is evaluated with for the
        omni-directional shields, Table 1 defining :math:`c` as the speed of
        sound in the test duct; over the -50 degC to 70 degC of clause 1.1
        that moves :math:`C_{3,4}` by up to 0,08 dB against the 340 m/s
        :func:`flow_modal_correction` uses on its own.
    :param static_pressure: Static pressure in the duct, in kilopascals;
        sets :math:`\rho`.
    :return: :class:`InDuctSoundPowerResult`.
    :raises ValueError: for levels of the wrong shape or not finite, a band
        that is not a nominal centre, a diameter, velocity or temperature
        that is not a real number or is outside the scope of the standard, a
        velocity beyond 40 m/s asked for in a band above 10 kHz, a correction
        that is neither a scalar nor one value per band, or a non-positive
        static pressure.
    """
    freqs = _nominal_bands(frequencies)
    _check_shield(shield)
    d = _check_duct_diameter(duct_diameter)
    u = _check_flow_velocity(flow_velocity, shield)
    c, rho_c = _check_air(temperature, static_pressure)
    c1 = require_per_band(
        microphone_correction, "microphone_correction", freqs, "frequencies"
    )
    c2 = require_per_band(shield_correction, "shield_correction", freqs, "frequencies")
    if not np.all(np.isfinite(c1)):
        msg = "'microphone_correction' must contain only finite values."
        raise ValueError(msg)
    if not np.all(np.isfinite(c2)):
        msg = "'shield_correction' must contain only finite values."
        raise ValueError(msg)
    arr = _position_levels(levels, freqs.size)

    # Eq. (9) over the positions, or Eq. (11) on an averaged spectrum.
    mean_level = energy_mean(arr, axis=0) if arr.ndim == 2 else arr  # noqa: PLR2004
    c34 = flow_modal_correction(freqs, u, d, shield=shield, speed_of_sound=c)
    combined = c1 + c2 + c34  # Eq. (10)
    corrected = mean_level + combined
    # Eq. (12): the plane-wave relation with S = pi d^2 / 4.
    area = math.pi * d**2 / 4.0
    lw = corrected + 10.0 * math.log10(area / _S0) - 10.0 * math.log10(rho_c / _RHO_C_0)
    lw = np.asarray(lw, dtype=np.float64)

    sigma_r = in_duct_reproducibility(freqs)
    if shield != "sampling-tube":
        warnings.warn(
            f"The reproducibility reported for a {shield} is the one Table 2 "
            "tabulates, which clause 4 NOTE 5 states refers to the sampling "
            "tube only and 'can be expected to increase for other shields'; "
            "the standard puts no number on the increase, so sigma_R and the "
            "expanded uncertainty are a lower bound here.",
            SoundPowerWarning,
            stacklevel=2,
        )
    information_only = (freqs > _NORMATIVE_MAX_HZ) | np.full(
        freqs.shape, abs(u) > _MAX_VELOCITY["sampling-tube"]
    )
    # Annex C, Eq. (C.1): the energy sum with the C_j of Table C.1.
    cj = np.asarray([_TABLE_C1[band] for band in _band_keys(freqs)], dtype=np.float64)
    return InDuctSoundPowerResult(
        frequencies=freqs,
        sound_power_level=lw,
        mean_pressure_level=np.asarray(mean_level, dtype=np.float64),
        corrected_pressure_level=np.asarray(corrected, dtype=np.float64),
        microphone_correction=c1,
        shield_correction=c2,
        flow_modal_correction=c34,
        combined_correction=np.asarray(combined, dtype=np.float64),
        reproducibility_standard_deviation=sigma_r,
        expanded_uncertainty=_COVERAGE_FACTOR * sigma_r,
        information_only_band=information_only,
        duct_diameter=d,
        duct_area=area,
        characteristic_impedance=rho_c,
        speed_of_sound=c,
        flow_velocity=u,
        shield=shield,
        sound_power_level_a=energy_sum(lw + cj),
    )
