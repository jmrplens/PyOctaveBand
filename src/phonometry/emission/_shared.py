#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
What every sound-power method has in common: the A-weighting band corrections
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

The reference area :math:`S_0 = 1` m^2 of the :math:`10 \log_{10}(S/S_0)`
surface term, the two accuracy grades the methods share (``'engineering'``,
grade 2 of ISO 3744 and ISO 9614-2, and ``'survey'``, grade 3 of ISO 3746 and
ISO 9614-2) and the warning class that every method raises when a
qualification limit is only just met are here for the same reason: the method
modules import them downward from a single definition, and none of them has to
import another method to get them.
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

from .._internal.warnings import PhonometryWarning

_S0 = 1.0  #: Reference area, in square metres (ISO 3744, 8.2.5).

Grade = Literal["engineering", "survey"]


class SoundPowerWarning(PhonometryWarning):
    """Non-fatal qualification issue in any of the sound-power methods.

    Emitted for ISO 3744/3746 background margin below the criterion and for
    ``K2`` beyond the method's validity limit (8.2.3, 4.3.2); for ISO 3741
    reverberation-room qualification (room volume vs Table 1, mean absorption)
    and microphone/source-position sampling; and for ISO 9614-2 negative total
    partial power and unmet field-indicator criteria. Where a lower criterion
    is only just met the returned levels represent upper bounds and must be
    reported as such."""


# --- ISO 3744:2010 Annex E, A-weighting band corrections Ck (dB) ------------
#: Table E.1 - one-third-octave mid-band frequencies.
_CK_THIRD: dict[int, float] = {
    50: -30.2, 63: -26.2, 80: -22.5, 100: -19.1, 125: -16.1, 160: -13.4,
    200: -10.9, 250: -8.6, 315: -6.6, 400: -4.8, 500: -3.2, 630: -1.9,
    800: -0.8, 1000: 0.0, 1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3,
    3150: 1.2, 4000: 1.0, 5000: 0.5, 6300: -0.1, 8000: -1.1, 10000: -2.5,
}
#: Table E.2 - octave mid-band frequencies (subset of E.1 values).
_CK_OCTAVE: dict[int, float] = {
    63: -26.2, 125: -16.1, 250: -8.6, 500: -3.2,
    1000: 0.0, 2000: 1.2, 4000: 1.0, 8000: -1.1,
}


def _a_weighting_corrections(frequencies: np.ndarray) -> np.ndarray:
    """A-weighting band corrections Ck for the given nominal mid-band
    frequencies (ISO 3744:2010 Annex E, Tables E.1/E.2). The octave-band
    values (Table E.2) coincide with the one-third-octave values (Table E.1)
    at the shared mid-band frequencies, so a single lookup serves both."""
    nominal = [round(f) for f in frequencies]
    # Use the octave table (E.2) when every band is an octave mid-band centre,
    # otherwise the one-third-octave table (E.1); the two agree where they
    # overlap, so the result is identical either way.
    table = _CK_OCTAVE if set(nominal) <= set(_CK_OCTAVE) else _CK_THIRD
    ck = []
    for freq in nominal:
        if freq not in table:
            raise ValueError(
                f"No ISO 3744 Annex E A-weighting value for {freq} Hz; "
                "expected a nominal octave or one-third-octave mid-band "
                "frequency (50 Hz to 10 kHz)."
            )
        ck.append(table[freq])
    return np.asarray(ck, dtype=np.float64)


def _check_grade(grade: str) -> Grade:
    if grade not in ("engineering", "survey"):
        raise ValueError("'grade' must be 'engineering' or 'survey'.")
    return cast(Grade, grade)
