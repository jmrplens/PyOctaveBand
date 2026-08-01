#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Airborne sound power from surface vibration (ISO/TS 7849-1/-2:2009).

The airborne sound power a machine radiates through the structure-borne
vibration of its outer surface is estimated from the surface vibratory velocity
and a **radiation factor** ``epsilon`` (the radiation efficiency). The radiated
power, in watts, is (ISO/TS 7849-1, Equation 6):

.. math::

   P = Z_c \, \langle v^2 \rangle \, S \, \epsilon

with :math:`Z_c` the characteristic impedance of air and
:math:`\langle v^2 \rangle` the mean-square vibratory velocity averaged over
the radiating area :math:`S`. The vibratory velocity is reported as a
**level**, in decibels, re :math:`v_0 = 5 \times 10^{-8}` m/s (Equation 3):

.. math::

   L_v = 10 \lg\frac{\langle v^2 \rangle}{v_0^2} = 20 \lg\frac{v}{v_0}

so the A-weighted sound power level follows in logarithmic form, in decibels
(ISO/TS 7849-1, Equation 12; ISO/TS 7849-2, Equation 15):

.. math::

   L_W = L_v + 10 \lg\frac{S}{S_0} + 10 \lg \epsilon
   + 10 \lg\frac{Z_{c,n}}{Z_{c,0}}

where :math:`S_0 = 1~\text{m}^2`, the normalized characteristic impedance
:math:`Z_{c,n} = 411~\text{N s/m}^3` (at 23 degC, 101.3 kPa) and the
reference acoustic impedance :math:`Z_{c,0} = 400~\text{N s/m}^3` give the
fixed :math:`10 \lg(411/400) = 0.118` dB term.

The two parts differ only in ``epsilon``:

* **Part 1 (survey)** assumes :math:`\epsilon = 1` and yields the *upper
  limit* ``L_W,max`` of the radiated power, needing only
  :math:`\langle v^2 \rangle` and ``S``.
* **Part 2 (engineering)** applies a frequency-band radiation factor
  ``epsilon_j`` determined (per ISO 9614) as
  :math:`\epsilon_j = P_j / (Z_{c,n} \langle v_j^2 \rangle S)` (Equation 8).

This module feeds the structure-borne source characterisation standards
(ISO 9611, EN 15657, EN 12354-5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

from .._internal.validation import require_positive

#: Reference vibratory velocity ``v0`` (ISO/TS 7849, Equation 3), m/s.
REFERENCE_VELOCITY: float = 5.0e-8
#: Normalized characteristic impedance ``Z_c,n`` at 23 degC, 101,3 kPa, N.s/m3.
NORMALIZED_IMPEDANCE: float = 411.0
#: Reference acoustic impedance of air ``Z_c,0``, N.s/m3.
REFERENCE_IMPEDANCE: float = 400.0
#: Reference sound power ``P0``, W.
REFERENCE_SOUND_POWER: float = 1.0e-12
#: Reference area ``S0``, m2.
REFERENCE_AREA: float = 1.0

#: ISO/TS 7849-1 Table 2: correction K1A (dB) for extraneous vibratory velocity
#: keyed by the integer level difference dLv (dB). dLv >= 10 -> 0; dLv < 3 -> 3.
_K1A_TABLE: dict[int, float] = {3: 3.0, 4: 2.0, 5: 2.0, 6: 1.0,
                                7: 1.0, 8: 1.0, 9: 1.0, 10: 0.0}


def velocity_level(
    velocity: ArrayLike, *, reference: float = REFERENCE_VELOCITY
) -> np.ndarray:
    r"""Vibratory velocity level (ISO/TS 7849-1, Eq. 3).

    :math:`L_v = 20 \lg(v/v_0)` with ``v0`` the reference velocity.

    :param velocity: R.m.s. vibratory velocity ``v`` (scalar or array), in m/s.
    :param reference: Reference velocity ``v0`` (Default: 5e-8 m/s).
    :return: The velocity level ``L_v``, in dB re ``v0``.
    :raises ValueError: for a non-positive reference.
    """
    reference = require_positive(reference, "reference")
    v = np.asarray(velocity, dtype=np.float64)
    return np.asarray(20.0 * np.log10(np.abs(v) / reference), dtype=np.float64)


def velocity_level_from_acceleration(
    peak_acceleration: ArrayLike,
    frequency: ArrayLike,
    *,
    reference: float = REFERENCE_VELOCITY,
) -> np.ndarray:
    r"""Velocity level from a sinusoidal acceleration (ISO/TS 7849-1, Eq. 8).

    :math:`L_v = 20 \lg\!\left( \frac{a_{\text{peak}}}{2\pi f v_0 \sqrt{2}}
    \right)`, used to convert a calibration acceleration to the equivalent
    r.m.s. velocity level.

    :param peak_acceleration: Peak acceleration ``a_peak`` (scalar or array),
        in m/s^2.
    :param frequency: Frequency ``f``, in hertz.
    :param reference: Reference velocity ``v0`` (Default: 5e-8 m/s).
    :return: The velocity level ``L_v``, in dB re ``v0``.
    :raises ValueError: for a non-positive frequency or reference.
    """
    reference = require_positive(reference, "reference")
    a = np.asarray(peak_acceleration, dtype=np.float64)
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive.")
    v_rms = np.abs(a) / (2.0 * np.pi * f * np.sqrt(2.0))
    return np.asarray(20.0 * np.log10(v_rms / reference), dtype=np.float64)


def mean_velocity_level(
    levels: ArrayLike, areas: ArrayLike | None = None
) -> float:
    """Mean vibratory velocity level over the surface (ISO/TS 7849-1, Eq. 10/11).

    With uniformly distributed positions (``areas`` is ``None``) this is the
    energetic average (Equation 10); with per-position partial areas it is the
    area-weighted average (Equation 11).

    :param levels: Velocity levels ``L_v,i`` at the positions, in dB.
    :param areas: Partial areas ``S_i`` (Equation 11), or ``None`` for the
        uniform energetic average (Equation 10).
    :return: The mean velocity level, in dB.
    """
    lv = np.asarray(levels, dtype=np.float64)
    energy = 10.0 ** (0.1 * lv)
    if areas is None:
        mean_energy = float(np.mean(energy))
    else:
        s = np.asarray(areas, dtype=np.float64)
        if s.shape != lv.shape:
            raise ValueError("'areas' must match the shape of 'levels'.")
        mean_energy = float(np.sum(s * energy) / np.sum(s))
    return float(10.0 * np.log10(mean_energy))


def radiation_factor(
    sound_power: ArrayLike,
    area: float,
    mean_square_velocity: ArrayLike,
    *,
    impedance: float = NORMALIZED_IMPEDANCE,
) -> np.ndarray:
    r"""A-weighted radiation factor ``epsilon`` (ISO/TS 7849-1 Eq. 4, -2 Eq. 8).

    :math:`\epsilon = P / (Z_c \langle v^2 \rangle S)`, the sound-radiation
    efficiency, from an independently measured radiated power (ISO 9614), the
    surface area and the mean-square vibratory velocity.

    :param sound_power: Radiated airborne sound power ``P`` (scalar or array),
        in W.
    :param area: Radiating surface area ``S``, in m^2 (> 0).
    :param mean_square_velocity: Mean-square vibratory velocity
        :math:`\langle v^2 \rangle`, in (m/s)^2.
    :param impedance: Characteristic impedance ``Z_c`` (Default: 411 N.s/m^3).
    :return: The radiation factor ``epsilon`` (dimensionless).
    :raises ValueError: for a non-positive area or impedance.
    """
    area = require_positive(area, "area")
    impedance = require_positive(impedance, "impedance")
    p = np.asarray(sound_power, dtype=np.float64)
    v2 = np.asarray(mean_square_velocity, dtype=np.float64)
    return np.asarray(p / (impedance * v2 * area), dtype=np.float64)


def radiated_sound_power_level(
    velocity_level: ArrayLike,
    area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
    reference_area: float = REFERENCE_AREA,
    normalized_impedance: float = NORMALIZED_IMPEDANCE,
    reference_impedance: float = REFERENCE_IMPEDANCE,
) -> np.ndarray:
    r"""Radiated sound power level (ISO/TS 7849-1 Eq. 12, -2 Eq. 15).

    .. math::

       L_W = L_v + 10 \lg\frac{S}{S_0} + 10 \lg \epsilon
       + 10 \lg\frac{Z_{c,n}}{Z_{c,0}}

    With the default ``radiation_factor = 1`` this is the Part 1 *upper limit*
    ``L_W,max``; pass a measured ``epsilon`` for the Part 2 engineering value.

    :param velocity_level: Mean vibratory velocity level ``L_v`` (scalar or
        array, e.g. per band), in dB re 5e-8 m/s.
    :param area: Radiating surface area ``S``, in m^2 (> 0).
    :param radiation_factor: Radiation factor ``epsilon`` (Default: 1.0).
    :param reference_area: Reference area ``S0`` (Default: 1 m^2).
    :param normalized_impedance: ``Z_c,n`` (Default: 411 N.s/m^3).
    :param reference_impedance: ``Z_c,0`` (Default: 400 N.s/m^3).
    :return: The sound power level ``L_W``, in dB re 1 pW.
    :raises ValueError: for a non-positive area, reference area or impedance.
    """
    area = require_positive(area, "area")
    reference_area = require_positive(reference_area, "reference_area")
    normalized_impedance = require_positive(normalized_impedance,
                                            "normalized_impedance")
    reference_impedance = require_positive(reference_impedance,
                                           "reference_impedance")
    lv = np.asarray(velocity_level, dtype=np.float64)
    eps = np.asarray(radiation_factor, dtype=np.float64)
    lw = (
        lv
        + 10.0 * np.log10(area / reference_area)
        + 10.0 * np.log10(eps)
        + 10.0 * np.log10(normalized_impedance / reference_impedance)
    )
    return np.asarray(lw, dtype=np.float64)


def extraneous_velocity_correction(level_difference: float) -> float:
    r"""Correction K1A for extraneous vibration (ISO/TS 7849-1, Table 2).

    :math:`\Delta L_v` is the difference between the operating and the
    extraneous vibratory velocity levels. The correction is subtracted from
    the measured level; per the standard :math:`\Delta L_v \ge 10` dB gives
    0 dB, and :math:`\Delta L_v < 3` dB uses the 3 dB value (the result is
    then an upper boundary). The level difference is rounded to the nearest
    integer decibel to index the standard's table.

    :param level_difference: Level difference :math:`\Delta L_v`, in dB.
    :return: The correction ``K1A`` to subtract, in dB.
    """
    if level_difference >= 10.0:
        return 0.0
    if level_difference < 3.0:
        return 3.0
    return _K1A_TABLE[round(level_difference)]


@dataclass(frozen=True)
class VibrationSoundPowerResult:
    """Sound power radiated by surface vibration (ISO/TS 7849).

    :ivar frequencies: Band centre frequencies, in hertz, or ``None`` for a
        single broadband value.
    :ivar velocity_level: Mean vibratory velocity level ``L_v`` per band, in dB.
    :ivar sound_power_level: Radiated sound power level ``L_W`` per band, in dB.
    :ivar radiation_factor: Radiation factor ``epsilon`` per band.
    :ivar area: Radiating surface area ``S``, in m^2.
    """

    velocity_level: np.ndarray
    sound_power_level: np.ndarray
    radiation_factor: np.ndarray
    area: float
    frequencies: np.ndarray | None = None

    @property
    def total_level(self) -> float:
        r"""Band-summed sound power level, in dB:
        :math:`10 \lg \sum_j 10^{0.1 L_{Wj}}`."""
        lw = np.asarray(self.sound_power_level, dtype=np.float64)
        return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * lw))))

    @property
    def sound_power_level_a(self) -> float:
        """A-weighted sound power level ``L_WA``, in dB re 1 pW.

        Combines the band levels with the A-weighting band corrections of
        ISO 3744:2010 Annex E (the standard tabulation reused by the vibration
        method) when band centre frequencies are known. Without band
        frequencies the result is an unweighted broadband level that cannot be
        A-weighted, so ``L_WA`` is undefined and ``nan`` is returned (the
        report then boxes the unweighted total ``L_W`` instead of an ``L_WA``
        claim, and no A-weighted verdict is drawn).
        """
        if self.frequencies is None:
            return float("nan")
        from .sound_power import _a_weighting_corrections

        lw = np.asarray(self.sound_power_level, dtype=np.float64)
        ck = _a_weighting_corrections(
            np.asarray(self.frequencies, dtype=np.float64)
        )
        return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * (lw + ck)))))

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the radiated sound power level per band.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.emission import plot_vibration_sound_power

        check_language(language)
        return plot_vibration_sound_power(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render an ISO/TS 7849 sound-power-from-vibration determination fiche.

        Writes a one-page sound-power test sheet: the standard-basis line naming
        the vibration method (the ISO/TS 7849-1 survey method with a fixed
        radiation factor when every band uses ``epsilon = 1``, otherwise the
        ISO/TS 7849-2 engineering method with a determined radiation factor), an
        optional metadata header (client, machine/source, test environment,
        instrumentation, climate, date), a per-band table (nominal
        octave/one-third-octave frequency, the surface vibratory velocity level
        ``Lv`` and the radiated band sound-power level ``LW``), the sound-power
        spectrum ``LW(f)`` with a nominal band axis, the boxed A-weighted sound
        power level ``LWA`` (dB re 1 pW) with the total ``LW``, the radiating
        area ``S`` and the applied method, an optional verdict row against a
        declared limit, and a measurement-basis strip stating the sound-power
        relation :math:`L_W = L_v + 10 \lg(S/S_0) + 10 \lg \epsilon
        + 10 \lg(Z_{c,n}/Z_{c,0})`.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the machine/source, ``test_room``
            the test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared A-weighted sound-power limit
            the fiche checks the result against (lower is better). The radiating
            area ``S`` comes from the result itself.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the radiation
            factor ``epsilon`` column.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso7849 import render_vibration_power_report

        return render_vibration_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def sound_power_from_vibration(
    velocity_level: ArrayLike,
    area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
    frequencies: ArrayLike | None = None,
) -> VibrationSoundPowerResult:
    """Bundle a sound-power-from-vibration determination (ISO/TS 7849).

    :param velocity_level: Mean vibratory velocity level ``L_v`` (per band), dB.
    :param area: Radiating surface area ``S``, in m^2 (> 0).
    :param radiation_factor: Radiation factor ``epsilon`` (Default: 1.0 -> the
        Part 1 upper limit); scalar or per band.
    :param frequencies: Band centre frequencies, in hertz, or ``None``.
    :return: The :class:`VibrationSoundPowerResult`.
    """
    lv = np.atleast_1d(np.asarray(velocity_level, dtype=np.float64))
    eps = np.broadcast_to(
        np.asarray(radiation_factor, dtype=np.float64), lv.shape
    ).astype(np.float64)
    lw = radiated_sound_power_level(lv, area, radiation_factor=eps)
    freq = None
    if frequencies is not None:
        freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        if freq.shape != lv.shape:
            raise ValueError(
                "'frequencies' must match the shape of 'velocity_level'."
            )
    return VibrationSoundPowerResult(
        velocity_level=lv,
        sound_power_level=np.asarray(lw, dtype=np.float64),
        radiation_factor=eps,
        area=float(area),
        frequencies=freq,
    )
