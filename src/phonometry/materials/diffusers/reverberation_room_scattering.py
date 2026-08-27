#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Random-incidence scattering coefficient in a reverberation room.

**ISO 17497-1:2004+A1:2014.** Four reverberation times (Table 2) taken with and
without the test sample, with a static and a rotating turntable, give two
Sabine-form absorption coefficients: the random-incidence absorption
coefficient ``alpha_s`` (Clause 8.1.1, Eq. (1)) and the specular absorption
coefficient ``alpha_spec`` (Clause 8.1.2, Eq. (4)). Their ratio yields the
scattering coefficient
:math:`s = (\alpha_{\mathrm{spec}} - \alpha_\mathrm{s}) / (1 - \alpha_\mathrm{s})`
(Clause 8.1.3, Eq. (5)). The turntable base plate is qualified through its
own scattering coefficient (Clause 8.1.4, Eq. (6)) against the Table 1
limits (Clause 6.2). Air properties come from the speed-of-sound and
energy-attenuation relations of Clause 8 (Eqs. (2)/(3), after ISO 9613-1),
and measurement accuracy from Annex A (Eqs. (A.1)-(A.5)).

One subject: everything the reverberation-room method needs, from the four
measured decay times to the scattering coefficient and its uncertainty. Part 2
of ISO 17497 is a different measurement in a free field and lives in
:mod:`phonometry.materials.diffusers.scattering_diffusion`; the two parts share
no formula, and the helpers are named per part so they are never mixed. Neither
part contains a numeric worked example.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    check_engine,
    require_equal_shapes,
    require_ranks,
    require_same_length,
)
from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._internal.types import Real
    from ..._report.metadata import ReportMetadata


__all__ = [
    "BASE_PLATE_BANDS",
    "BASE_PLATE_MAX_SCATTERING",
    "ScatteringDiffusionWarning",
    "ScatteringResult",
    "ScatteringUncertainty",
    "absorption_coefficient_uncertainty",
    "air_attenuation_coefficient",
    "base_plate_scattering",
    "check_base_plate_scattering",
    "random_incidence_absorption",
    "reverberation_time_uncertainty",
    "scattering_coefficient",
    "scattering_coefficient_spectrum",
    "scattering_coefficient_uncertainty",
    "specular_absorption_coefficient",
    "speed_of_sound",
]


#: Sabine/Eyring air constant of ISO 17497-1 Eqs. (1), (4), (6); shared with
#: ISO 354. Given as-is by the standard (approx. 24 * ln(10) / c-related), not
#: re-derived here.
_SABINE_CONSTANT = 55.3

#: Reference speed of sound of ISO 17497-1 Eq. (2), in m/s (343,2 exactly).
_C_REF = 343.2

#: Reference absolute temperature of ISO 17497-1 Eq. (2), in kelvin, i.e.
#: 273,15 + 20 degC = 293,15 K.
_T_REF_K = 293.15

#: Celsius-to-kelvin offset used by ISO 17497-1 Eq. (2) (273,15).
_T0 = 273.15

#: Decibels per neper, ``10 * lg(e)`` (approx. 4,343), converting the
#: ISO 9613-1 pressure attenuation coefficient (dB/m) to the energy
#: attenuation coefficient ``m`` (1/m) of ISO 17497-1 Eq. (3).
_DB_PER_NEPER = 10.0 * math.log10(math.e)

#: Minimum number ``N`` of spatially-averaged reverberation-time measurements
#: for the standard error of the mean of ISO 17497-1 Eq. (A.1); the
#: ``N (N - 1)`` denominator vanishes for a single measurement.
_MIN_MEASUREMENTS = 2

#: One-third-octave centre frequencies of ISO 17497-1 Table 1, in Hz
#: (equivalent full-scale ``f / N``), 100 Hz to 5000 Hz.
BASE_PLATE_BANDS: tuple[int, ...] = (
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
    4000,
    5000,
)

#: Maximum admissible scattering coefficient of the base plate alone
#: (ISO 17497-1:2004+A1:2014, Table 1, Clause 6.2), keyed by equivalent
#: full-scale one-third-octave centre frequency in Hz.
BASE_PLATE_MAX_SCATTERING: dict[int, float] = {
    100: 0.05,
    125: 0.05,
    160: 0.05,
    200: 0.05,
    250: 0.05,
    315: 0.05,
    400: 0.05,
    500: 0.05,
    630: 0.10,
    800: 0.10,
    1000: 0.10,
    1250: 0.15,
    1600: 0.15,
    2000: 0.15,
    2500: 0.20,
    3150: 0.20,
    4000: 0.20,
    5000: 0.25,
}


class ScatteringDiffusionWarning(PhonometryWarning):
    """Advisory for out-of-range scattering/diffusion measurement conditions."""


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------
def _positive_scalar(value: float, name: str) -> float:
    """Return ``value`` as a positive float or raise ``ValueError``."""
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        msg = f"'{name}' must be a positive, finite number."
        raise ValueError(msg)
    return v


def _positive_array(value: ArrayLike, name: str) -> Real:
    """Return ``value`` as a positive float array or raise ``ValueError``."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' values must be finite."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = f"'{name}' values must be positive."
        raise ValueError(msg)
    return arr


def _nonneg_array(value: ArrayLike, name: str) -> Real:
    """Return ``value`` as a non-negative float array or raise ``ValueError``."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' values must be finite."
        raise ValueError(msg)
    if np.any(arr < 0.0):
        msg = f"'{name}' values must be non-negative."
        raise ValueError(msg)
    return arr


# ---------------------------------------------------------------------------
# ISO 17497-1: air properties (Clause 8, Eqs. (2)/(3), after ISO 9613-1).
# ---------------------------------------------------------------------------
def speed_of_sound(temperature: ArrayLike) -> Real:
    r"""Speed of sound in air (ISO 17497-1:2004, Clause 8, Eq. (2)).

    :math:`c = 343.2 \sqrt{(273.15 + t) / 293.15}` (m/s).

    :param temperature: Air temperature ``t``, in **degrees Celsius** (scalar
        or per band).
    :return: Speed of sound ``c``, in metres per second.
    :raises ValueError: if any temperature is at or below -273.15 degC.
    """
    t = np.asarray(temperature, dtype=np.float64)
    kelvin = _T0 + t
    if np.any(kelvin <= 0.0):
        msg = "'temperature' must exceed -273.15 degC."
        raise ValueError(msg)
    return np.asarray(_C_REF * np.sqrt(kelvin / _T_REF_K), dtype=np.float64)


def air_attenuation_coefficient(pressure_attenuation_db_per_m: ArrayLike) -> Real:
    r"""Energy attenuation coefficient ``m`` (ISO 17497-1, Clause 8, Eq. (3)).

    :math:`m = \alpha / (10 \log_{10}(e)) \approx \alpha / 4.343` (1/m), where
    ``alpha`` is
    the sound-*pressure* attenuation coefficient in dB/m obtained from
    ISO 9613-1 using the measured temperature and relative humidity.

    :param pressure_attenuation_db_per_m: Pressure attenuation coefficient
        ``alpha`` from ISO 9613-1, in decibels per metre (scalar or per band).
    :return: Energy (power) attenuation coefficient ``m``, in reciprocal metres.
    :raises ValueError: if any value is negative or non-finite.
    """
    alpha = _nonneg_array(
        pressure_attenuation_db_per_m, "pressure_attenuation_db_per_m"
    )
    return np.asarray(alpha / _DB_PER_NEPER, dtype=np.float64)


# ---------------------------------------------------------------------------
# ISO 17497-1: Sabine-form absorption differences (Clause 8, Eqs. (1)/(4)/(6)).
# ---------------------------------------------------------------------------
def _sabine_absorption(
    volume: float,
    area: float,
    situation_a: tuple[ArrayLike, ArrayLike, ArrayLike],
    situation_b: tuple[ArrayLike, ArrayLike, ArrayLike],
) -> Real:
    r"""Sabine-form absorption difference between two measurement situations.

    Implements the common kernel of ISO 17497-1 Eqs. (1), (4) and (6):

    .. math::

       55.3 \, \frac{V}{S}
       \left( \frac{1}{c_b t_b} - \frac{1}{c_a t_a} \right)
       - \frac{4 V}{S} (m_b - m_a)

    Each situation is the tuple ``(c, T, m)`` of speed of sound (m/s),
    reverberation time (s) and energy attenuation coefficient (1/m). ``V`` and
    ``S`` are the room volume and sample area; the difference is taken as
    "situation b minus situation a" to match the printed equations.
    """
    vol = _positive_scalar(volume, "volume")
    surf = _positive_scalar(area, "area")
    c_a = _positive_array(situation_a[0], "c")
    t_a = _positive_array(situation_a[1], "T")
    m_a = _nonneg_array(situation_a[2], "m")
    c_b = _positive_array(situation_b[0], "c")
    t_b = _positive_array(situation_b[1], "T")
    m_b = _nonneg_array(situation_b[2], "m")
    ratio = vol / surf
    absorption = _SABINE_CONSTANT * ratio * (
        1.0 / (c_b * t_b) - 1.0 / (c_a * t_a)
    ) - 4.0 * ratio * (m_b - m_a)
    return np.asarray(absorption, dtype=np.float64)


def random_incidence_absorption(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    t1: ArrayLike,
    c2: ArrayLike,
    t2: ArrayLike,
    m1: ArrayLike = 0.0,
    m2: ArrayLike = 0.0,
) -> Real:
    r"""Random-incidence absorption coefficient ``alpha_s`` (ISO 17497-1, Eq. (1)).

    .. math::

       \alpha_\mathrm{s} = 55.3 \, \frac{V}{S}
       \left( \frac{1}{c_2 T_2} - \frac{1}{c_1 T_1} \right)
       - \frac{4 V}{S} (m_2 - m_1)

    Situation 1 is the empty room with the (static) base plate present;
    situation 2 adds the test sample, still without turntable rotation
    (Table 2, rows t1 and t2).

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c1: Speed of sound during ``t1``, in m/s (see :func:`speed_of_sound`).
    :param t1: Reverberation time without sample (base plate only), in seconds.
    :param c2: Speed of sound during ``t2``, in m/s.
    :param t2: Reverberation time with the test sample, in seconds.
    :param m1: Energy attenuation coefficient during ``t1``, in 1/m
        (see :func:`air_attenuation_coefficient`); defaults to 0.
    :param m2: Energy attenuation coefficient during ``t2``, in 1/m; defaults to 0.
    :return: Random-incidence absorption coefficient ``alpha_s`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c1, t1, m1), (c2, t2, m2))


def specular_absorption_coefficient(
    volume: float,
    area: float,
    *,
    c3: ArrayLike,
    t3: ArrayLike,
    c4: ArrayLike,
    t4: ArrayLike,
    m3: ArrayLike = 0.0,
    m4: ArrayLike = 0.0,
) -> Real:
    r"""Specular absorption coefficient ``alpha_spec`` (ISO 17497-1, Eq. (4)).

    .. math::

       \alpha_{\text{spec}} = 55.3 \, \frac{V}{S}
       \left( \frac{1}{c_4 T_4} - \frac{1}{c_3 T_3} \right)
       - \frac{4 V}{S} (m_4 - m_3)

    Situation 3 is the rotating base plate without the sample; situation 4 is
    the sample on the rotating turntable (Table 2, rows t3 and t4). The
    apparent (specular) absorption includes the energy lost to scattering.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c3: Speed of sound during ``t3``, in m/s.
    :param t3: Reverberation time, rotating base plate without sample, in seconds.
    :param c4: Speed of sound during ``t4``, in m/s.
    :param t4: Reverberation time, sample on the rotating turntable, in seconds.
    :param m3: Energy attenuation coefficient during ``t3``, in 1/m; defaults to 0.
    :param m4: Energy attenuation coefficient during ``t4``, in 1/m; defaults to 0.
    :return: Specular absorption coefficient ``alpha_spec`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c3, t3, m3), (c4, t4, m4))


def scattering_coefficient(
    alpha_spec: ArrayLike,
    alpha_s: ArrayLike,
    *,
    truncate_negative: bool = True,
) -> Real:
    r"""Random-incidence scattering coefficient ``s`` (ISO 17497-1, Eq. (5)).

    .. math::

       s = 1 - \frac{1 - \alpha_{\text{spec}}}{1 - \alpha_\mathrm{s}}
         = \frac{\alpha_{\text{spec}} - \alpha_\mathrm{s}}{1 - \alpha_\mathrm{s}}

    Following the presentation rule of Clause 8.3, negative results are
    truncated to 0 while values greater than 1 (which can occur through edge
    effects, Clause 6.3.2) are **kept** and reported. Rounding to 0,01 for a
    results table is left to the caller.

    :param alpha_spec: Specular absorption coefficient ``alpha_spec`` (Eq. (4)).
    :param alpha_s: Random-incidence absorption coefficient ``alpha_s`` (Eq. (1)).
    :param truncate_negative: If ``True`` (default), clip negative ``s`` to 0
        per Clause 8.3; values above 1 are never clipped.
    :return: Scattering coefficient ``s`` (per band).
    :raises ValueError: if any ``alpha_s`` equals 1 (undefined ratio).
    """
    spec = np.asarray(alpha_spec, dtype=np.float64)
    diff = np.asarray(alpha_s, dtype=np.float64)
    denom = 1.0 - diff
    if np.any(np.isclose(denom, 0.0)):
        msg = "'alpha_s' must not equal 1 (division by zero)."
        raise ValueError(msg)
    s = (spec - diff) / denom
    if truncate_negative:
        s = np.maximum(s, 0.0)
    return np.asarray(s, dtype=np.float64)


@dataclass(frozen=True)
class ScatteringResult:
    """A random-incidence scattering-coefficient spectrum (ISO 17497-1).

    :ivar frequencies: One-third-octave band centre frequencies, in hertz.
    :ivar scattering: Scattering coefficient ``s`` per band (Eq. (5)).
    :ivar random_incidence: Random-incidence absorption ``alpha_s`` (Eq. (1)).
    :ivar specular: Specular absorption ``alpha_spec`` (Eq. (4)).
    """

    frequencies: Real
    scattering: Real
    random_incidence: Real
    specular: Real

    def __post_init__(self) -> None:
        """Reject a spectrum whose columns disagree, is empty, or is non-finite.

        The ISO 17497-1 fiche prints one row per entry of ``frequencies``,
        reading the absorptions and the scattering coefficient at that row's
        position, so the columns must agree on how many bands there are.

        The spectrum must cover at least one band. Four length-0 columns agree
        with each other, and the sheet that comes out is a complete accredited
        page with an empty table under the fabricated headline "0 Hz to 0 Hz",
        which the ``lo``/``hi`` fall-backs write when there is no band to take
        a range from. :func:`scattering_coefficient_spectrum` refuses an empty
        axis for the same reason, so an empty spectrum can only be hand-built.

        Every value must also be finite. The method has no undeterminable
        band: :func:`scattering_coefficient_spectrum` derives ``s`` from the
        two Sabine-form absorptions of positive, finite reverberation times,
        so a NaN can only be smuggled into a hand-built result. Without this
        pin a NaN coefficient renders as a literal ``nan`` cell in the
        accredited per-band table, and a NaN band centre crashes the header's
        ``round(freqs.min())`` with a bare message naming neither the field
        nor the result.

        :raises ValueError: if the per-band columns disagree, the spectrum is
            empty, or any value is non-finite.
        """
        require_ranks(self, frequencies=1, scattering=1, random_incidence=1, specular=1)
        require_same_length(
            self, "frequencies", "scattering", "random_incidence", "specular"
        )
        if np.asarray(self.frequencies).size == 0:
            msg = (
                "ScatteringResult: 'frequencies' must carry at least one "
                "band; the spectrum is empty."
            )
            raise ValueError(msg)
        for name in ("frequencies", "scattering", "random_incidence", "specular"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), np.float64))):
                msg = f"ScatteringResult: '{name}' must contain only finite values."
                raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the scattering coefficient ``s`` versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_scattering_coefficient

        check_language(language)
        return plot_scattering_coefficient(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 17497-1 scattering-coefficient test-report fiche to a PDF.

        Writes a one-page accredited random-incidence scattering report
        (ISO 17497-1:2004+A1:2014): the standard-basis line, an optional
        metadata header block (client, specimen, test room, sample area ``S``,
        temperature, humidity ...), a two-panel body with the per-band table
        (frequency, the random-incidence absorption ``alpha_s`` and the
        scattering coefficient ``s``) beside the ``s(f)`` curve on a categorical
        band axis, and a footer with the fixed disclaimer. ISO 17497-1 is a
        characterisation, so there is no pass/fail verdict and no single-number
        rating.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche whose header shows only the
            measured frequency range. The applicable descriptive fields are
            ``client``, ``manufacturer``, ``specimen``, ``area``, ``room_volume``,
            ``mounting``, ``test_room``, ``test_date``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``measurement_standard``,
            ``laboratory``, ``operator``, ``report_id`` and ``notes``. The
            ``requirement`` field is ignored (ISO 17497-1 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the value table inserts the specular
            absorption ``alpha_spec`` column beside ``alpha_s`` and ``s``.
        :param language: Fiche language: ``"en"`` (default, English, decimal
            point) or ``"es"`` (Spanish, decimal comma).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        check_engine(engine)
        from ..._report.iso17497 import render_scattering_report

        return render_scattering_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def scattering_coefficient_spectrum(
    frequencies: ArrayLike,
    specular_absorption: ArrayLike,
    random_absorption: ArrayLike,
    *,
    truncate_negative: bool = True,
) -> ScatteringResult:
    """Scattering-coefficient spectrum ``s(f)`` (ISO 17497-1, Eq. (5)).

    Convenience wrapper over :func:`scattering_coefficient` that pairs the
    per-band specular ``alpha_spec`` (Eq. (4)) and random-incidence ``alpha_s``
    (Eq. (1)) absorptions with their band centres and returns a plottable
    :class:`ScatteringResult`.

    :param frequencies: One-third-octave band centres, in hertz (1-D).
    :param specular_absorption: Specular absorption ``alpha_spec`` per band.
    :param random_absorption: Random-incidence absorption ``alpha_s`` per band.
    :param truncate_negative: Clip negative ``s`` to 0 (Clause 8.3 default).
    :return: A :class:`ScatteringResult` with ``.plot()``.
    :raises ValueError: if the three inputs differ in shape, the band centres
        are empty or not 1-D, or any ``alpha_s`` equals 1.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    spec = np.atleast_1d(np.asarray(specular_absorption, dtype=np.float64))
    rand = np.atleast_1d(np.asarray(random_absorption, dtype=np.float64))
    if freq.ndim != 1 or freq.size == 0:
        msg = (
            "'frequencies' must be a non-empty 1-D sequence of band centres; "
            f"got shape {freq.shape}."
        )
        raise ValueError(msg)
    require_equal_shapes(
        "scattering_coefficient_spectrum",
        {
            "frequencies": freq.shape,
            "specular_absorption": spec.shape,
            "random_absorption": rand.shape,
        },
        "band",
    )
    s = scattering_coefficient(spec, rand, truncate_negative=truncate_negative)
    return ScatteringResult(
        frequencies=freq,
        scattering=np.asarray(s, dtype=np.float64),
        random_incidence=rand,
        specular=spec,
    )


def base_plate_scattering(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    t1: ArrayLike,
    c3: ArrayLike,
    t3: ArrayLike,
    m1: ArrayLike = 0.0,
    m3: ArrayLike = 0.0,
) -> Real:
    r"""Scattering coefficient of the base plate alone (ISO 17497-1, Eq. (6)).

    .. math::

       s_{\mathrm{base}} = 55.3 \frac{V}{S}
       \left( \frac{1}{c_3 T_3} - \frac{1}{c_1 T_1} \right)
       - \frac{4 V}{S} (m_3 - m_1)

    Ideally :math:`T_1 = T_3`; a slightly non-symmetrical base plate shortens
    ``t3``
    and this quality metric captures the resulting spurious scattering, which
    must not exceed the Table 1 limits (Clause 6.2). See
    :func:`check_base_plate_scattering`.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c1: Speed of sound during ``t1``, in m/s.
    :param t1: Reverberation time with the static base plate, in seconds.
    :param c3: Speed of sound during ``t3``, in m/s.
    :param t3: Reverberation time with the rotating base plate, in seconds.
    :param m1: Energy attenuation coefficient during ``t1``, in 1/m; defaults to 0.
    :param m3: Energy attenuation coefficient during ``t3``, in 1/m; defaults to 0.
    :return: Base-plate scattering coefficient ``s_base`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c1, t1, m1), (c3, t3, m3))


def check_base_plate_scattering(
    scattering: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> tuple[int, ...]:
    """Verify base-plate scattering against Table 1 (ISO 17497-1, Clause 6.2).

    Every band whose measured base-plate scattering coefficient exceeds the
    :data:`BASE_PLATE_MAX_SCATTERING` limit is collected and a single
    :class:`ScatteringDiffusionWarning` is issued when any band is over the
    limit.

    :param scattering: Measured base-plate scattering coefficients, either a
        mapping keyed by one-third-octave centre frequency (Hz) or a sequence
        of 18 values ordered as :data:`BASE_PLATE_BANDS`.
    :return: Tuple of the centre frequencies (Hz) that exceed the limit, in
        ascending order (empty if the base plate is compliant).
    :raises ValueError: for a mapping missing a band or a sequence of the
        wrong length.
    """
    if isinstance(scattering, Mapping):
        values: dict[int, float] = {}
        for band in BASE_PLATE_BANDS:
            if band in scattering:
                values[band] = float(scattering[band])
            elif float(band) in scattering:
                values[band] = float(scattering[float(band)])
            else:
                msg = (
                    f"scattering mapping is missing band {band} Hz; "
                    f"expected keys {BASE_PLATE_BANDS}"
                )
                raise ValueError(msg)
    else:
        arr = np.asarray(scattering, dtype=np.float64)
        if arr.ndim != 1 or arr.size != len(BASE_PLATE_BANDS):
            msg = (
                f"scattering must have {len(BASE_PLATE_BANDS)} values for "
                f"bands {BASE_PLATE_BANDS}, got shape {arr.shape}"
            )
            raise ValueError(msg)
        values = dict(zip(BASE_PLATE_BANDS, arr.tolist(), strict=True))
    exceeded = tuple(
        band
        for band in BASE_PLATE_BANDS
        if values[band] > BASE_PLATE_MAX_SCATTERING[band]
    )
    if exceeded:
        warnings.warn(
            "Base-plate scattering coefficient exceeds the ISO 17497-1 Table 1 "
            f"limit at {list(exceeded)} Hz; the turntable is not compliant "
            "(Clause 6.2).",
            ScatteringDiffusionWarning,
            stacklevel=2,
        )
    return exceeded


# ---------------------------------------------------------------------------
# ISO 17497-1: measurement uncertainty (Annex A, Eqs. (A.1)-(A.5)).
# ---------------------------------------------------------------------------
def reverberation_time_uncertainty(times: ArrayLike) -> Real:
    r"""Standard uncertainty of a reverberation time (ISO 17497-1, Eq. (A.1)).

    .. math::

       u = \sqrt{ \sum_i (T_i - \overline{T})^2 / (N (N - 1)) }

    with :math:`\overline{T}` the mean of
    the ``N`` spatially-averaged measurements (Eq. (A.2)); this is the standard
    error of the mean.

    :param times: The :math:`N \ge 2` reverberation-time measurements, in
        seconds.
    :return: Standard uncertainty ``u`` of the mean reverberation time (0-d).
    :raises ValueError: if fewer than two measurements are supplied.
    """
    arr = _positive_array(times, "times")
    if arr.ndim != 1:
        msg = "'times' must be a 1-D sequence of measurements."
        raise ValueError(msg)
    n = arr.size
    if n < _MIN_MEASUREMENTS:
        msg = "'times' needs at least two measurements (N >= 2)."
        raise ValueError(msg)
    mean = arr.mean()
    variance_of_mean = np.sum((arr - mean) ** 2) / (n * (n - 1))
    return np.asarray(np.sqrt(variance_of_mean), dtype=np.float64)


def absorption_coefficient_uncertainty(
    volume: float,
    area: float,
    *,
    c: ArrayLike,
    t_a: ArrayLike,
    u_a: ArrayLike,
    t_b: ArrayLike,
    u_b: ArrayLike,
) -> Real:
    r"""Uncertainty of a Sabine absorption coefficient (ISO 17497-1, A.3/A.4).

    .. math::

       u_\alpha = \frac{55.3 V}{c S}
       \sqrt{(u_b / t_b^2)^2 + (u_a / t_a^2)^2}

    With situations ``(t1, t2)`` this is ``u(alpha_s)`` (Eq. (A.3)); with
    ``(t3, t4)`` it is ``u(alpha_spec)`` (Eq. (A.4)). The unsubscripted ``c`` of
    the standard is taken as a single (mean) speed of sound.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c: Speed of sound ``c``, in m/s.
    :param t_a: Reverberation time of the first situation, in seconds.
    :param u_a: Standard uncertainty of ``t_a`` (Eq. (A.1)), in seconds.
    :param t_b: Reverberation time of the second situation, in seconds.
    :param u_b: Standard uncertainty of ``t_b`` (Eq. (A.1)), in seconds.
    :return: Combined standard uncertainty of the absorption coefficient (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    vol = _positive_scalar(volume, "volume")
    surf = _positive_scalar(area, "area")
    c_arr = _positive_array(c, "c")
    ta = _positive_array(t_a, "t_a")
    tb = _positive_array(t_b, "t_b")
    ua = _nonneg_array(u_a, "u_a")
    ub = _nonneg_array(u_b, "u_b")
    prefactor = _SABINE_CONSTANT * vol / (c_arr * surf)
    combined = np.sqrt((ub / tb**2) ** 2 + (ua / ta**2) ** 2)
    return np.asarray(prefactor * combined, dtype=np.float64)


@dataclass(frozen=True)
class ScatteringUncertainty:
    r"""Uncertainty of the scattering coefficient (ISO 17497-1, Annex A).

    :ivar u_scattering: Combined standard uncertainty ``u_s`` of the scattering
        coefficient (Eq. (A.5)).
    :ivar expanded: Expanded uncertainty :math:`U = 2 u_s` at 95 % confidence
        (Annex A).
    """

    u_scattering: Real
    expanded: Real


def scattering_coefficient_uncertainty(
    alpha_spec: ArrayLike,
    alpha_s: ArrayLike,
    u_alpha_spec: ArrayLike,
    u_alpha_s: ArrayLike,
) -> ScatteringUncertainty:
    r"""Uncertainty of the scattering coefficient (ISO 17497-1, Eq. (A.5)).

    .. math::

       u_s = \left\lvert
       \frac{\alpha_{\mathrm{spec}} - 1}{1 - \alpha_\mathrm{s}} \right\rvert
       \sqrt{\left( \frac{u_{\alpha_{\mathrm{spec}}}}
       {\alpha_{\mathrm{spec}} - 1} \right)^{2}
       + \left( \frac{u_{\alpha_\mathrm{s}}}{1 - \alpha_\mathrm{s}} \right)^{2}}

    with the expanded uncertainty :math:`U = 2 u_s` (95 % confidence).

    :param alpha_spec: Specular absorption coefficient ``alpha_spec`` (Eq. (4)).
    :param alpha_s: Random-incidence absorption coefficient ``alpha_s`` (Eq. (1)).
    :param u_alpha_spec: Standard uncertainty of ``alpha_spec`` (Eq. (A.4)).
    :param u_alpha_s: Standard uncertainty of ``alpha_s`` (Eq. (A.3)).
    :return: A :class:`ScatteringUncertainty` with ``u_s`` and
        :math:`U = 2 u_s`.
    :raises ValueError: if any ``alpha_s`` equals 1 or any ``alpha_spec`` equals 1.
    """
    spec = np.asarray(alpha_spec, dtype=np.float64)
    diff = np.asarray(alpha_s, dtype=np.float64)
    u_spec = _nonneg_array(u_alpha_spec, "u_alpha_spec")
    u_diff = _nonneg_array(u_alpha_s, "u_alpha_s")
    spec_term = spec - 1.0
    diff_term = 1.0 - diff
    if np.any(np.isclose(diff_term, 0.0)):
        msg = "'alpha_s' must not equal 1 (division by zero)."
        raise ValueError(msg)
    if np.any(np.isclose(spec_term, 0.0)):
        msg = "'alpha_spec' must not equal 1 (division by zero)."
        raise ValueError(msg)
    u_s = np.abs(spec_term / diff_term) * np.sqrt(
        (u_spec / spec_term) ** 2 + (u_diff / diff_term) ** 2
    )
    u_s_arr = np.asarray(u_s, dtype=np.float64)
    return ScatteringUncertainty(
        u_scattering=u_s_arr,
        expanded=np.asarray(2.0 * u_s_arr, dtype=np.float64),
    )
