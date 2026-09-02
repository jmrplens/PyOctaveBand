#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""What every sound-power method has in common: the A-weighting band corrections
of ISO 3744:2010 Annex E, the reference area, the accuracy-grade vocabulary
and the qualification warning.

Each determination method carries its own standard and its own module (the
enveloping surface of ISO 3744/3746, the anechoic room of ISO 3745, the
reverberation room of ISO 3741, the intensity scanning of ISO 9614-2/-3), yet
they meet in the last step: the band sound power levels are combined into an
A-weighted total with the band corrections :math:`C_k` tabulated in ISO
3744:2010 Annex E (Tables E.1/E.2),

.. math::

   L_{W\mathrm{A}} = 10 \log_{10}\!\left[ \sum_k 10^{0.1 (L_{Wk} + C_k)} \right]
   \tag{Eq. E.1}

and the other standards adopt that same table by reference for their own
A-weighted totals (ISO 3741:2010 Annex F Eq. F.2, ISO 3745:2012 Annex C
Eq. C.1, ISO 9614-2:1996 clause 10.6 b). One table, transcribed once.

The radiation-impedance correction :math:`C_2` that carries a band level to
the reference meteorological conditions is shared the same way. ISO 3741:2010
clause 9.1.4 and ISO 3747:2010 Annex C print the same expression,

.. math::

   C_2 = -10 \log_{10}\frac{p_\mathrm{s}}{p_{\mathrm{s},0}}
   + 15 \log_{10}\frac{273{,}15 + \theta}{\theta_1},
   \qquad p_{\mathrm{s},0} = 101{,}325\ \mathrm{kPa},
   \quad \theta_1 = 296\ \mathrm{K}

digit for digit, so the guard on the two meteorological inputs and the
correction itself live here once.

The single event time-integrated levels of the sound energy determinations
meet here too: ISO 3744:2010 Eq. (19)/(20), ISO 3746:2010 Eq. (16)/(17) and
ISO 3741:2010 Eq. (22)/(23) are one pair of equations printed three times,
the energy mean of :math:`N_\mathrm{e}` events measured one at a time and
the :math:`-10 \log_{10} N_\mathrm{e}` of one measurement that encompasses
them all, so :func:`_single_event_mean` is transcribed once and the method
modules call it.

The reference area :math:`S_0 = 1` m^2 of the :math:`10 \log_{10}(S/S_0)`
surface term, the two accuracy grades the methods share (``'engineering'``,
grade 2 of ISO 3744 and ISO 9614-2, and ``'survey'``, grade 3 of ISO 3746 and
ISO 9614-2) and the warning class that every method raises when a
qualification limit is only just met are here for the same reason: the method
modules import them downward from a single definition, and none of them has to
import another method to get them.
"""

from __future__ import annotations

import warnings
from typing import Literal, cast

import numpy as np

from .._internal.levels_math import energy_mean
from .._internal.validation import require_positive
from .._internal.warnings import PhonometryWarning

_S0 = 1.0  #: Reference area, in square metres (ISO 3744, 8.2.5).
#: Reference static pressure of the ISO 3740 family, in kilopascals
#: (ISO 3741:2010 clause 4; ISO 3747:2010 Annex C prints 1,013 25 x 10^5 Pa).
_PS0 = 101.325
#: Reference temperature of the C2 radiation-impedance correction, in kelvin
#: (ISO 3741:2010 clause 9.1.4, ISO 3747:2010 Annex C).
_THETA1 = 296.0
#: Temperature floor in degC: sqrt(273 + theta) of the speed-of-sound formula
#: c = 20.05*sqrt(273 + theta) is zero or complex at or below it (ISO 3741, 9.1.4).
_ROUNDED_ABS_ZERO_C = -273.0

#: Reference duration, in seconds, of the single event time-integrated sound
#: pressure level: L_E = L_p,T + 10 lg(T/T0) (ISO 3744:2010, 3.4 NOTE 1).
_T0 = 1.0
#: Fewest single sound emission events a sound energy determination rests
#: on, whether measured one at a time or in one measurement encompassing
#: them all (ISO 3744:2010 8.3.1, ISO 3746:2010 8.4.1, ISO 3741:2010 8.5.1).
_MIN_EVENTS = 5

Grade = Literal["engineering", "survey"]


class SoundPowerWarning(PhonometryWarning):
    """Non-fatal qualification issue in any of the sound-power methods.

    Emitted for ISO 3744/3746 background margin below the criterion and for
    ``K2`` beyond the method's validity limit (8.2.3, 4.3.2); for ISO 3741
    reverberation-room qualification (room volume vs Table 1, mean absorption)
    and microphone/source-position sampling; and for ISO 9614-2 negative total
    partial power and unmet field-indicator criteria. Where a lower criterion
    is only just met the returned levels represent upper bounds and must be
    reported as such.
    """


# --- ISO 3744:2010 Annex E, A-weighting band corrections Ck (dB) ------------
#: Table E.1 - one-third-octave mid-band frequencies.
_CK_THIRD: dict[int, float] = {
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
}
#: Table E.2 - octave mid-band frequencies (subset of E.1 values).
_CK_OCTAVE: dict[int, float] = {
    63: -26.2,
    125: -16.1,
    250: -8.6,
    500: -3.2,
    1000: 0.0,
    2000: 1.2,
    4000: 1.0,
    8000: -1.1,
}


def _a_weighting_corrections(frequencies: np.ndarray) -> np.ndarray:
    """A-weighting band corrections Ck for the given nominal mid-band
    frequencies (ISO 3744:2010 Annex E, Tables E.1/E.2). The octave-band
    values (Table E.2) coincide with the one-third-octave values (Table E.1)
    at the shared mid-band frequencies, so a single lookup serves both.
    """
    nominal = [round(f) for f in frequencies]
    # Use the octave table (E.2) when every band is an octave mid-band centre,
    # otherwise the one-third-octave table (E.1); the two agree where they
    # overlap, so the result is identical either way.
    table = _CK_OCTAVE if set(nominal) <= set(_CK_OCTAVE) else _CK_THIRD
    ck = []
    for freq in nominal:
        if freq not in table:
            msg = (
                f"No ISO 3744 Annex E A-weighting value for {freq} Hz; "
                "expected a nominal octave or one-third-octave mid-band "
                "frequency (50 Hz to 10 kHz)."
            )
            raise ValueError(msg)
        ck.append(table[freq])
    return np.asarray(ck, dtype=np.float64)


def _validate_meteorology(temperature: float, static_pressure: float) -> None:
    r"""Guard the meteorological inputs before the log10/sqrt of C1/C2 and c.

    A non-finite or :math:`\le -273` degC temperature makes
    :math:`\sqrt{273 + \theta}` complex/zero, and a non-finite or non-positive
    static pressure makes :math:`\log_{10}(p_\mathrm{s}/p_{\mathrm{s}0})` undefined; both are rejected
    with a clean ``ValueError``.
    """
    if not np.isfinite(temperature) or temperature <= _ROUNDED_ABS_ZERO_C:
        msg = "'temperature' must be finite and greater than -273 degC."
        raise ValueError(msg)
    if not np.isfinite(static_pressure) or static_pressure <= 0.0:
        msg = "'static_pressure' must be finite and positive."
        raise ValueError(msg)


def _c2_correction(temperature: float, static_pressure: float) -> float:
    """Radiation-impedance correction ``C2`` (ISO 3741:2010 clause 9.1.4,
    ISO 3747:2010 Annex C, Eq. C.1/C.3).
    """
    return float(
        -10.0 * np.log10(static_pressure / _PS0)
        + 15.0 * np.log10((273.15 + temperature) / _THETA1)
    )


def _check_grade(grade: str) -> Grade:
    if grade not in ("engineering", "survey"):
        msg = "'grade' must be 'engineering' or 'survey'."
        raise ValueError(msg)
    return cast(Grade, grade)


def _single_event_mean(
    levels: np.ndarray, events: int | None, *, name: str, stacklevel: int
) -> np.ndarray:
    r"""Mean single event time-integrated level of one event, per position.

    ISO 3744:2010 Eq. (19)/(20), printed again as ISO 3746:2010 Eq. (16)/(17)
    and ISO 3741:2010 Eq. (22)/(23). With ``events`` ``None`` the first axis
    of ``levels`` holds the :math:`N_\mathrm{e}` events measured one at a
    time and the mean is the energy average over that axis:

    .. math::

       L'_{Ei(\mathrm{ST})} = 10 \log_{10}\!\left[ \frac{1}{N_\mathrm{e}}
       \sum_{q=1}^{N_\mathrm{e}} 10^{0.1 L'_{Ei,q(\mathrm{ST})}} \right]
       \tag{Eq. 19}

    With ``events`` :math:`= N_\mathrm{e}` the whole of ``levels`` is one
    measurement that encompasses :math:`N_\mathrm{e}` successive events, and
    the level of one event is that measurement less :math:`10 \log_{10}
    N_\mathrm{e}`:

    .. math::

       L'_{Ei(\mathrm{ST})} = L'_{Ei,N_\mathrm{e}(\mathrm{ST})}
       - 10 \log_{10} N_\mathrm{e} \tag{Eq. 20}

    Both standards ask for at least five events in either mode; fewer is
    accepted with a :class:`SoundPowerWarning`, the determination then being
    below the count the method is written for.

    :param levels: Per-event levels with the events on the first axis
        (``events`` ``None``), or the one measurement of ``events`` events.
    :param events: ``None`` for one event per entry of the first axis, else
        the number of events one measurement encompasses.
    :param name: The caller's parameter name, for the error messages.
    :param stacklevel: Where the warning is reported.
    :return: The mean level of one event, one axis fewer than ``levels`` when
        ``events`` is ``None``, the same shape otherwise.
    :raises ValueError: for non-finite levels, an empty event axis or a
        non-positive ``events``.
    """
    arr = np.asarray(levels, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    if events is None:
        if arr.ndim == 0 or arr.shape[0] == 0:
            msg = f"'{name}' must carry one entry per single event on its first axis."
            raise ValueError(msg)
        count = int(arr.shape[0])
        mean = energy_mean(arr, axis=0)
    else:
        if isinstance(events, bool) or not isinstance(events, (int, np.integer)):
            msg = "'events' must be an integer: the number of events one measurement encompasses."
            raise ValueError(msg)
        if events < 1:
            msg = "'events' must be a positive integer."
            raise ValueError(msg)
        count = int(events)
        mean = arr - 10.0 * np.log10(count)
    if count < _MIN_EVENTS:
        warnings.warn(
            f"Only {count} single event(s); the sound energy determination "
            f"requires at least {_MIN_EVENTS} (ISO 3744:2010 8.3.1, ISO "
            "3741:2010 8.5.1).",
            SoundPowerWarning,
            stacklevel=stacklevel,
        )
    return np.asarray(mean, dtype=np.float64)


def _background_exposure(
    background_levels: np.ndarray, integration_time: float | None
) -> np.ndarray:
    r"""The background's single event time-integrated level over ``T``.

    The correction Eq. (21) subtracts one energy from another, and the two
    must be referred to the same quantity: the event level is
    :math:`\int p^2\,\mathrm{d}t` re :math:`E_0 = p_0^2 \cdot 1` s, so the
    time-averaged background level the standard has measured over the same
    interval :math:`T` enters as its own exposure over :math:`T`,
    :math:`L_{p(\mathrm{B})} + 10 \log_{10}(T/T_0)` (clause 3.4 NOTE 1).

    :param background_levels: Time-averaged background levels, in decibels.
    :param integration_time: The interval ``T`` of the single event levels,
        in seconds; required.
    :return: The background exposure levels over ``T``, in decibels.
    :raises ValueError: if ``integration_time`` is missing or not positive.
    """
    if integration_time is None:
        msg = (
            "'integration_time' (the interval T of the single event levels, in "
            "seconds) is required with 'background_levels': the background is "
            "compared as its exposure over the same T (ISO 3744:2010 8.3.4, ISO 3741:2010 9.2.2)."
        )
        raise ValueError(msg)
    t = require_positive(integration_time, "integration_time")
    bg = np.asarray(background_levels, dtype=np.float64)
    if not np.all(np.isfinite(bg)):
        msg = "'background_levels' must contain only finite values."
        raise ValueError(msg)
    return bg + 10.0 * np.log10(t / _T0)
