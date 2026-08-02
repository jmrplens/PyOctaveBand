#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Random-incidence scattering and directional diffusion coefficients.

Two complementary free-field / reverberation-room surface descriptors are
implemented, each faithful to its own standard:

* **ISO 17497-1:2004+A1:2014** - random-incidence *scattering* coefficient in
  a reverberation room. Four reverberation times (Table 2) taken with and
  without the test sample, with a static and a rotating turntable, give two
  Sabine-form absorption coefficients: the random-incidence absorption
  coefficient ``alpha_s`` (Clause 8.1.1, Eq. (1)) and the specular absorption
  coefficient ``alpha_spec`` (Clause 8.1.2, Eq. (4)). Their ratio yields the
  scattering coefficient
  :math:`s = (\alpha_{\mathrm{spec}} - \alpha_s) / (1 - \alpha_s)`
  (Clause 8.1.3, Eq. (5)). The turntable base plate is qualified through its
  own scattering coefficient (Clause 8.1.4, Eq. (6)) against the Table 1
  limits (Clause 6.2). Air properties come from the speed-of-sound and
  energy-attenuation relations of Clause 8 (Eqs. (2)/(3), after ISO 9613-1),
  and measurement accuracy from Annex A (Eqs. (A.1)-(A.5)).

* **ISO 17497-2:2012** - directional *diffusion* coefficient in a free field.
  From the set of reflected sound-pressure levels ``L_i`` on a semicircle or
  hemisphere the autocorrelation diffusion coefficient ``d_theta`` is formed
  for equal-area receivers (Clause 8.1, Formula (5)) or with per-receiver area
  weights ``N_i`` (Formula (6)); the area weights follow from the solid-angle
  factors of Clause 8.3 (Formula (8)). Finite-panel effects are removed by
  normalising to the reference flat surface (Clause 8.2, Formula (7)), and the
  random-incidence coefficient is the (weighted) average of the directional
  coefficients over the source positions (Clause 8.4).

Neither part of ISO 17497 contains a numeric worked example; the two methods
are distinct measurements and the helpers are named per part so they are
never mixed.

The Part 2 diffusion coefficient is the design target of subwavelength diffuser
panels such as the metadiffusers of Jiménez, Cox, Romero-García & Groby
(2017, *Scientific Reports* 7, 5389, doi:10.1038/s41598-017-05710-5), whose
slow-sound resonant slots reach the diffusion of a Schroeder or
quadratic-residue diffuser in a fraction of the depth.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..._internal.types import Real
from ..._internal.warnings import PhonometryWarning, _warn_renamed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata


__all__ = [
    "BASE_PLATE_BANDS",
    "BASE_PLATE_MAX_SCATTERING",
    "TWO_DIMENSIONAL_SOURCE_WEIGHTS",
    "DiffusionResult",
    "DiffusionSpectrum",
    "ScatteringDiffusionWarning",
    "ScatteringResult",
    "ScatteringUncertainty",
    "absorption_coefficient_uncertainty",
    "air_attenuation_coefficient",
    "area_factors",
    "base_plate_scattering",
    "check_base_plate_scattering",
    "diffusion_spectrum",
    "directional_diffusion",
    "directional_diffusion_coefficient",
    "normalized_diffusion_coefficient",
    "random_incidence_absorption",
    "random_incidence_diffusion",
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

#: One-third-octave centre frequencies of ISO 17497-1 Table 1, in Hz
#: (equivalent full-scale ``f / N``), 100 Hz to 5000 Hz.
BASE_PLATE_BANDS: tuple[int, ...] = (
    100, 125, 160, 200, 250, 315, 400, 500, 630,
    800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
)

#: Maximum admissible scattering coefficient of the base plate alone
#: (ISO 17497-1:2004+A1:2014, Table 1, Clause 6.2), keyed by equivalent
#: full-scale one-third-octave centre frequency in Hz.
BASE_PLATE_MAX_SCATTERING: dict[int, float] = {
    100: 0.05, 125: 0.05, 160: 0.05, 200: 0.05, 250: 0.05, 315: 0.05,
    400: 0.05, 500: 0.05, 630: 0.10, 800: 0.10, 1000: 0.10, 1250: 0.15,
    1600: 0.15, 2000: 0.15, 2500: 0.20, 3150: 0.20, 4000: 0.20, 5000: 0.25,
}

#: Source-position weights for the two-dimensional (single-plane) random
#: incidence average of ISO 17497-2 Clause 8.4: the 0 deg source is weighted 1
#: and each of the four +/-30 deg, +/-60 deg sources is weighted 3.
TWO_DIMENSIONAL_SOURCE_WEIGHTS: tuple[int, ...] = (1, 3, 3, 3, 3)


class ScatteringDiffusionWarning(PhonometryWarning):
    """Advisory for out-of-range scattering/diffusion measurement conditions."""


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------
def _positive_scalar(value: float, name: str) -> float:
    """Return ``value`` as a positive float or raise ``ValueError``."""
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return v


def _positive_array(value: ArrayLike, name: str) -> Real:
    """Return ``value`` as a positive float array or raise ``ValueError``."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' values must be finite.")
    if np.any(arr <= 0.0):
        raise ValueError(f"'{name}' values must be positive.")
    return arr


def _nonneg_array(value: ArrayLike, name: str) -> Real:
    """Return ``value`` as a non-negative float array or raise ``ValueError``."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' values must be finite.")
    if np.any(arr < 0.0):
        raise ValueError(f"'{name}' values must be non-negative.")
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
        raise ValueError("'temperature' must exceed -273,15 degC.")
    return np.asarray(
        _C_REF * np.sqrt(kelvin / _T_REF_K), dtype=np.float64
    )


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
       \left( \frac{1}{c_b T_b} - \frac{1}{c_a T_a} \right)
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
    absorption = (
        _SABINE_CONSTANT
        * ratio
        * (1.0 / (c_b * t_b) - 1.0 / (c_a * t_a))
        - 4.0 * ratio * (m_b - m_a)
    )
    return np.asarray(absorption, dtype=np.float64)


def random_incidence_absorption(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    T1: ArrayLike,
    c2: ArrayLike,
    T2: ArrayLike,
    m1: ArrayLike = 0.0,
    m2: ArrayLike = 0.0,
) -> Real:
    r"""Random-incidence absorption coefficient ``alpha_s`` (ISO 17497-1, Eq. (1)).

    .. math::

       \alpha_s = 55.3 \, \frac{V}{S}
       \left( \frac{1}{c_2 T_2} - \frac{1}{c_1 T_1} \right)
       - \frac{4 V}{S} (m_2 - m_1)

    Situation 1 is the empty room with the (static) base plate present;
    situation 2 adds the test sample, still without turntable rotation
    (Table 2, rows T1 and T2).

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c1: Speed of sound during ``T1``, in m/s (see :func:`speed_of_sound`).
    :param T1: Reverberation time without sample (base plate only), in seconds.
    :param c2: Speed of sound during ``T2``, in m/s.
    :param T2: Reverberation time with the test sample, in seconds.
    :param m1: Energy attenuation coefficient during ``T1``, in 1/m
        (see :func:`air_attenuation_coefficient`); defaults to 0.
    :param m2: Energy attenuation coefficient during ``T2``, in 1/m; defaults to 0.
    :return: Random-incidence absorption coefficient ``alpha_s`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c1, T1, m1), (c2, T2, m2))


def specular_absorption_coefficient(
    volume: float,
    area: float,
    *,
    c3: ArrayLike,
    T3: ArrayLike,
    c4: ArrayLike,
    T4: ArrayLike,
    m3: ArrayLike = 0.0,
    m4: ArrayLike = 0.0,
) -> Real:
    r"""Specular absorption coefficient ``alpha_spec`` (ISO 17497-1, Eq. (4)).

    .. math::

       \alpha_{\text{spec}} = 55.3 \, \frac{V}{S}
       \left( \frac{1}{c_4 T_4} - \frac{1}{c_3 T_3} \right)
       - \frac{4 V}{S} (m_4 - m_3)

    Situation 3 is the rotating base plate without the sample; situation 4 is
    the sample on the rotating turntable (Table 2, rows T3 and T4). The
    apparent (specular) absorption includes the energy lost to scattering.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c3: Speed of sound during ``T3``, in m/s.
    :param T3: Reverberation time, rotating base plate without sample, in seconds.
    :param c4: Speed of sound during ``T4``, in m/s.
    :param T4: Reverberation time, sample on the rotating turntable, in seconds.
    :param m3: Energy attenuation coefficient during ``T3``, in 1/m; defaults to 0.
    :param m4: Energy attenuation coefficient during ``T4``, in 1/m; defaults to 0.
    :return: Specular absorption coefficient ``alpha_spec`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c3, T3, m3), (c4, T4, m4))


def scattering_coefficient(
    alpha_spec: ArrayLike,
    alpha_s: ArrayLike,
    *,
    truncate_negative: bool = True,
) -> Real:
    r"""Random-incidence scattering coefficient ``s`` (ISO 17497-1, Eq. (5)).

    .. math::

       s = 1 - \frac{1 - \alpha_{\text{spec}}}{1 - \alpha_s}
         = \frac{\alpha_{\text{spec}} - \alpha_s}{1 - \alpha_s}

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
        raise ValueError("'alpha_s' must not equal 1 (division by zero).")
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

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
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
    :raises ValueError: if the three inputs differ in length, are empty, or any
        ``alpha_s`` equals 1.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    spec = np.atleast_1d(np.asarray(specular_absorption, dtype=np.float64))
    rand = np.atleast_1d(np.asarray(random_absorption, dtype=np.float64))
    if (
        freq.ndim != 1
        or freq.size == 0
        or freq.shape != spec.shape
        or freq.shape != rand.shape
    ):
        raise ValueError(
            "'frequencies', 'specular_absorption' and 'random_absorption' must "
            "be non-empty, 1-D and equal-length."
        )
    s = scattering_coefficient(spec, rand, truncate_negative=truncate_negative)
    return ScatteringResult(
        frequencies=freq,
        scattering=np.asarray(s, dtype=np.float64),
        random_incidence=rand,
        specular=spec,
    )


@dataclass(frozen=True)
class DiffusionResult:
    """A measured polar response and its diffusion coefficient (ISO 17497-2).

    :ivar angles: Receiver angles of the polar response, in degrees.
    :ivar levels: Reflected sound-pressure level at each angle, in decibels.
    :ivar coefficient: Autocorrelation diffusion coefficient ``d`` (Formula (5)).
    """

    angles: Real
    levels: Real
    coefficient: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the polar response with the diffusion coefficient annotated.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        polar :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffusion_polar

        check_language(language)
        return plot_diffusion_polar(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 17497-2 polar-response test-report fiche to a PDF.

        Writes a one-page accredited free-field diffusion report for a single
        source position (ISO 17497-2:2012, Clause 8.5): the standard-basis line,
        an optional metadata header block, a two-panel body with the corrected
        polar-response table (receiver angle and reflected sound-pressure level
        ``L``, rounded to 0,1 dB) beside the semicircular polar plot, a boxed
        directional diffusion coefficient ``d_theta`` (Formula (5)/(6)) and a
        footer with the fixed disclaimer. ISO 17497-2 is a characterisation, so
        there is no pass/fail verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche. The applicable descriptive
            fields are ``client``, ``manufacturer``, ``specimen``, ``mounting``,
            ``test_room``, ``test_date``, ``temperature``, ``relative_humidity``,
            ``pressure``, ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (ISO 17497-2 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for signature parity; the polar-response fiche
            has no extended table, so it renders the same body.
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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso17497 import render_diffusion_polar_report

        return render_diffusion_polar_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class DiffusionSpectrum:
    """A diffusion-coefficient spectrum ``d(f)`` (ISO 17497-2, Clause 8.5).

    Where :class:`DiffusionResult` holds the polar response of a single
    one-third-octave band, this holds the diffusion coefficient across the
    measured bands, so it can be tabulated and plotted against frequency as
    Clause 8.5 requires. The per-band coefficient is a *directional* diffusion
    coefficient ``d_theta`` (Formula (5)/(6)) when it comes from one source
    position, or a *random-incidence* diffusion coefficient ``d`` when it is the
    per-band average of the directional coefficients over the source positions
    (Clause 8.4); the standard defines both as frequency-dependent quantities,
    so this carries a spectrum rather than a single number.

    :ivar frequencies: One-third-octave band centre frequencies, in hertz.
    :ivar diffusion: Diffusion coefficient ``d`` per band (directional per
        source, or random-incidence when averaged over source positions).
    :ivar normalized: Optional normalised diffusion coefficient ``d_n`` per band
        (Formula (7)), or ``None`` when the reference flat surface was not
        measured.
    """

    frequencies: Real
    diffusion: Real
    normalized: Real | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the diffusion coefficient ``d`` versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_diffusion_report

        check_language(language)
        return plot_diffusion_report(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 17497-2 diffusion-coefficient test-report fiche to a PDF.

        Writes a one-page accredited free-field diffusion report
        (ISO 17497-2:2012, Clause 8.5): the standard-basis line, an optional
        metadata header block, a two-panel body with the per-band table
        (frequency, the diffusion coefficient ``d`` and, when present, the
        normalised ``d_n``) beside the ``d(f)`` curve on a categorical band
        axis, a boxed characterisation headline over the tested frequency range,
        and a footer with the fixed disclaimer. ISO 17497-2 is a
        characterisation, so there is no pass/fail verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche whose header shows only the
            measured frequency range. The applicable descriptive fields are
            ``client``, ``manufacturer``, ``specimen``, ``mounting``,
            ``test_room``, ``test_date``, ``temperature``, ``relative_humidity``,
            ``pressure``, ``measurement_standard``, ``laboratory``, ``operator``,
            ``report_id`` and ``notes``. The ``requirement`` field is ignored
            (ISO 17497-2 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` and a normalised spectrum is present, the
            value table adds the normalised ``d_n`` column.
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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso17497 import render_diffusion_spectrum_report

        return render_diffusion_spectrum_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def diffusion_spectrum(
    frequencies: ArrayLike,
    diffusion: ArrayLike,
    *,
    normalized: ArrayLike | None = None,
) -> DiffusionSpectrum:
    """Diffusion-coefficient spectrum ``d(f)`` (ISO 17497-2, Clause 8.5).

    Pairs the per-band diffusion coefficients ``d`` with their band centres and
    returns a plottable, reportable :class:`DiffusionSpectrum`. The coefficient
    is the *directional* coefficient ``d_theta`` (Formula (5)/(6)) when it comes
    from a single source position, or the *random-incidence* coefficient ``d``
    when it is the per-band average of the directional coefficients over the
    source positions (Clause 8.4, e.g. via :func:`random_incidence_diffusion`
    band by band). The optional normalised coefficients ``d_n`` (Formula (7))
    are carried through when supplied.

    :param frequencies: One-third-octave band centres, in hertz (1-D).
    :param diffusion: Diffusion coefficient ``d`` per band.
    :param normalized: Optional normalised diffusion coefficient ``d_n`` per
        band; ``None`` when the reference flat surface was not measured.
    :return: A :class:`DiffusionSpectrum` with ``.plot()`` and ``.report()``.
    :raises ValueError: if the inputs differ in length, are empty or not 1-D.
    """
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    d = np.atleast_1d(np.asarray(diffusion, dtype=np.float64))
    if freq.ndim != 1 or freq.size == 0 or freq.shape != d.shape:
        raise ValueError(
            "'frequencies' and 'diffusion' must be non-empty, 1-D and "
            "equal-length."
        )
    d_n: Real | None = None
    if normalized is not None:
        d_n = np.atleast_1d(np.asarray(normalized, dtype=np.float64))
        if d_n.shape != freq.shape:
            raise ValueError(
                "'normalized' must match 'frequencies' in length."
            )
    return DiffusionSpectrum(
        frequencies=freq,
        diffusion=d,
        normalized=d_n,
    )


def directional_diffusion(
    angles: ArrayLike,
    levels: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> DiffusionResult:
    """Diffusion coefficient of a polar response (ISO 17497-2, Formula (5)/(6)).

    Convenience wrapper over :func:`directional_diffusion_coefficient` that
    keeps the receiver angles alongside the levels and returns a plottable
    :class:`DiffusionResult`.

    :param angles: Receiver angles of the polar response, in degrees (1-D).
    :param levels: Reflected sound-pressure level at each angle, in decibels.
    :param weights: Optional area weights ``N_i`` (Formula (8)); ``None`` uses
        the equal-area Formula (5).
    :return: A :class:`DiffusionResult` with ``.plot()``.
    :raises ValueError: if ``angles`` and ``levels`` differ in length or are
        shorter than two receivers.
    """
    ang = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    lev = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if ang.shape != lev.shape:
        raise ValueError("'angles' and 'levels' must have the same length.")
    d = float(directional_diffusion_coefficient(lev, area_weights=weights))
    return DiffusionResult(angles=ang, levels=lev, coefficient=d)


def base_plate_scattering(
    volume: float,
    area: float,
    *,
    c1: ArrayLike,
    T1: ArrayLike,
    c3: ArrayLike,
    T3: ArrayLike,
    m1: ArrayLike = 0.0,
    m3: ArrayLike = 0.0,
) -> Real:
    r"""Scattering coefficient of the base plate alone (ISO 17497-1, Eq. (6)).

    .. math::

       s_{\mathrm{base}} = 55.3 \frac{V}{S}
       \left( \frac{1}{c_3 T_3} - \frac{1}{c_1 T_1} \right)
       - \frac{4 V}{S} (m_3 - m_1)

    Ideally :math:`T_1 = T_3`; a slightly non-symmetrical base plate shortens
    ``T3``
    and this quality metric captures the resulting spurious scattering, which
    must not exceed the Table 1 limits (Clause 6.2). See
    :func:`check_base_plate_scattering`.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c1: Speed of sound during ``T1``, in m/s.
    :param T1: Reverberation time with the static base plate, in seconds.
    :param c3: Speed of sound during ``T3``, in m/s.
    :param T3: Reverberation time with the rotating base plate, in seconds.
    :param m1: Energy attenuation coefficient during ``T1``, in 1/m; defaults to 0.
    :param m3: Energy attenuation coefficient during ``T3``, in 1/m; defaults to 0.
    :return: Base-plate scattering coefficient ``s_base`` (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    return _sabine_absorption(volume, area, (c1, T1, m1), (c3, T3, m3))


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
                raise ValueError(
                    f"scattering mapping is missing band {band} Hz; "
                    f"expected keys {BASE_PLATE_BANDS}"
                )
    else:
        arr = np.asarray(scattering, dtype=np.float64)
        if arr.ndim != 1 or arr.size != len(BASE_PLATE_BANDS):
            raise ValueError(
                f"scattering must have {len(BASE_PLATE_BANDS)} values for "
                f"bands {BASE_PLATE_BANDS}, got shape {arr.shape}"
            )
        values = dict(zip(BASE_PLATE_BANDS, arr.tolist()))
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
        raise ValueError("'times' must be a 1-D sequence of measurements.")
    n = arr.size
    if n < 2:
        raise ValueError("'times' needs at least two measurements (N >= 2).")
    mean = arr.mean()
    variance_of_mean = np.sum((arr - mean) ** 2) / (n * (n - 1))
    return np.asarray(np.sqrt(variance_of_mean), dtype=np.float64)


def absorption_coefficient_uncertainty(
    volume: float,
    area: float,
    *,
    c: ArrayLike,
    T_a: ArrayLike,
    u_a: ArrayLike,
    T_b: ArrayLike,
    u_b: ArrayLike,
) -> Real:
    r"""Uncertainty of a Sabine absorption coefficient (ISO 17497-1, A.3/A.4).

    .. math::

       u_\alpha = \frac{55.3 V}{c S}
       \sqrt{(u_b / T_b^2)^2 + (u_a / T_a^2)^2}

    With situations ``(T1, T2)`` this is ``u(alpha_s)`` (Eq. (A.3)); with
    ``(T3, T4)`` it is ``u(alpha_spec)`` (Eq. (A.4)). The unsubscripted ``c`` of
    the standard is taken as a single (mean) speed of sound.

    :param volume: Reverberation-room volume ``V``, in cubic metres.
    :param area: Test-sample area ``S``, in square metres.
    :param c: Speed of sound ``c``, in m/s.
    :param T_a: Reverberation time of the first situation, in seconds.
    :param u_a: Standard uncertainty of ``T_a`` (Eq. (A.1)), in seconds.
    :param T_b: Reverberation time of the second situation, in seconds.
    :param u_b: Standard uncertainty of ``T_b`` (Eq. (A.1)), in seconds.
    :return: Combined standard uncertainty of the absorption coefficient (per band).
    :raises ValueError: for non-positive ``V``, ``S``, ``c`` or ``T``.
    """
    vol = _positive_scalar(volume, "volume")
    surf = _positive_scalar(area, "area")
    c_arr = _positive_array(c, "c")
    ta = _positive_array(T_a, "T_a")
    tb = _positive_array(T_b, "T_b")
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
       \frac{\alpha_{\mathrm{spec}} - 1}{1 - \alpha_s} \right\rvert
       \sqrt{\left( \frac{u_{\alpha_{\mathrm{spec}}}}
       {\alpha_{\mathrm{spec}} - 1} \right)^{2}
       + \left( \frac{u_{\alpha_s}}{1 - \alpha_s} \right)^{2}}

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
        raise ValueError("'alpha_s' must not equal 1 (division by zero).")
    if np.any(np.isclose(spec_term, 0.0)):
        raise ValueError("'alpha_spec' must not equal 1 (division by zero).")
    u_s = np.abs(spec_term / diff_term) * np.sqrt(
        (u_spec / spec_term) ** 2 + (u_diff / diff_term) ** 2
    )
    u_s_arr = np.asarray(u_s, dtype=np.float64)
    return ScatteringUncertainty(
        u_scattering=u_s_arr,
        expanded=np.asarray(2.0 * u_s_arr, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# ISO 17497-2: directional and random-incidence diffusion (Clause 8).
# ---------------------------------------------------------------------------
def directional_diffusion_coefficient(
    levels: ArrayLike,
    *,
    area_weights: ArrayLike | None = None,
) -> float:
    r"""Directional diffusion coefficient ``d_theta`` (ISO 17497-2, Formulas (5)/(6)).

    For a fixed source position and one-third-octave band, from the ``n``
    reflected sound-pressure levels ``L_i`` (dB). With equal-area receivers
    (``area_weights is None``, Formula (5)):

    .. math::

       d_\theta = \frac{\left( \sum_i p_i \right)^{2} - \sum_i p_i^{2}}
       {(n - 1) \sum_i p_i^{2}}

    where :math:`p_i = 10^{L_i / 10}`. When each receiver samples a different
    area (Formula (6)) the per-receiver weights ``N_i`` (from
    :func:`area_factors`) enter:

    .. math::

       d_\theta = \frac{\left( \sum_i p_i N_i \right)^{2}
       - \sum_i N_i p_i^{2}}
       {\left( \sum_i N_i - 1 \right) \sum_i N_i p_i^{2}}

    which reduces to Formula (5) for uniform weights. The coefficient is 0 when
    only one receiver has non-zero scattered energy and 1 when all receivers
    are equal.

    :param levels: The :math:`n \ge 2` reflected sound-pressure levels
        ``L_i``, in
        decibels (a level of ``-inf`` denotes a receiver with zero energy).
    :param area_weights: Optional per-receiver area weights ``N_i`` (Formula (8));
        ``None`` selects the equal-area Formula (5).
    :return: Directional diffusion coefficient ``d_theta`` (a scalar).
    :raises ValueError: for fewer than two receivers, a non-1-D input, a length
        mismatch, or non-positive total weight.
    """
    lvl = np.asarray(levels, dtype=np.float64)
    if lvl.ndim != 1:
        raise ValueError("'levels' must be a 1-D sequence of receiver SPLs.")
    n = lvl.size
    if n < 2:
        raise ValueError("'levels' needs at least two receivers (n >= 2).")
    p = np.power(10.0, lvl / 10.0)
    if area_weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(area_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.size != n:
            raise ValueError(
                "'area_weights' must match the number of receivers "
                f"({n}), got shape {weights.shape}."
            )
        if np.any(weights <= 0.0):
            raise ValueError("'area_weights' values must be positive.")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1.0:
        raise ValueError("The total area weight must exceed 1.")
    weighted_energy = np.sum(p * weights)
    weighted_energy_sq = np.sum(weights * p**2)
    if weighted_energy_sq <= 0.0:
        raise ValueError(
            "The polar response carries no energy (all levels -inf); the "
            "diffusion coefficient is undefined."
        )
    numerator = weighted_energy**2 - weighted_energy_sq
    denominator = (weight_sum - 1.0) * weighted_energy_sq
    return float(numerator / denominator)


def normalized_diffusion_coefficient(
    d_theta: ArrayLike,
    d_theta_reference: ArrayLike,
) -> Real:
    r"""Normalised directional diffusion coefficient (ISO 17497-2, Formula (7)).

    :math:`d_{\theta,n} = (d_\theta - d_{\theta,r}) / (1 - d_{\theta,r})`,
    removing the
    finite-panel diffusion of the reference flat surface ``d_theta_r`` (same
    projected footprint as the test surface). It maps
    :math:`d_\theta = d_{\theta,r}`
    to 0 and :math:`d_\theta = 1` to 1.

    :param d_theta: Directional diffusion coefficient of the test surface.
    :param d_theta_reference: Directional diffusion coefficient of the
        reference flat surface ``d_theta_r``.
    :return: Normalised directional diffusion coefficient ``d_theta_n``.
    :raises ValueError: if any reference coefficient equals 1 (undefined ratio).
    """
    d = np.asarray(d_theta, dtype=np.float64)
    d_ref = np.asarray(d_theta_reference, dtype=np.float64)
    denom = 1.0 - d_ref
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError(
            "'d_theta_reference' must not equal 1 (division by zero)."
        )
    return np.asarray((d - d_ref) / denom, dtype=np.float64)


def area_factors(
    elevations: ArrayLike,
    *,
    delta_theta: float,
    delta_phi: float | None = None,
) -> Real:
    r"""Per-receiver area weights ``N_i`` (ISO 17497-2, Clause 8.3, Formula (8)).

    For a hemispherical measurement the solid-angle area sampled by a receiver
    at elevation ``theta`` (with angular spacings ``delta_theta``,
    ``delta_phi``) is:

    .. math::

       \begin{aligned}
       A_i &= (4 \pi / \Delta\phi) \sin^{2}(\Delta\theta / 4)
       && \text{for } \theta = 0^\circ \\
       A_i &= 2 \sin(\theta) \sin(\Delta\theta / 2)
       && \text{for } \theta \ne 0^\circ, 90^\circ \\
       A_i &= \sin(\Delta\theta / 2)
       && \text{for } \lvert \theta \rvert = 90^\circ
       \end{aligned}

    and :math:`N_i = A_i / A_{\min}` (Formula (8)), with ``A_min`` the
    smallest ``A_i``. All angles are handled internally in **radians**; the
    :math:`\theta = 0` form in particular requires ``delta_phi`` in radians to
    be dimensionally consistent with the :math:`4 \pi` factor.

    :param elevations: Receiver elevation angles ``theta`` from the reference
        normal, in **degrees** (1-D), over the measurement domain
        :math:`0 \le \theta \le 90` (Figure 7). Formula (8) assumes a single
        receiver
        at :math:`\theta = 0` (the zenith); duplicate zenith entries would
        each take
        the full zenith area.
    :param delta_theta: Elevation spacing between adjacent receivers, in degrees
        (typically 5).
    :param delta_phi: Azimuth spacing between adjacent receivers, in degrees;
        defaults to ``delta_theta``. Required (implicitly) for the
        :math:`\theta = 0`
        receiver.
    :return: Per-receiver area weights ``N_i`` (dimensionless, min value 1).
    :raises ValueError: for a non-1-D input or non-positive spacings.
    """
    theta_deg = np.asarray(elevations, dtype=np.float64)
    if theta_deg.ndim != 1 or theta_deg.size == 0:
        raise ValueError(
            "'elevations' must be a non-empty 1-D sequence of angles."
        )
    d_theta = _positive_scalar(delta_theta, "delta_theta")
    d_phi = d_theta if delta_phi is None else _positive_scalar(
        delta_phi, "delta_phi"
    )
    theta = np.radians(theta_deg)
    d_theta_rad = math.radians(d_theta)
    d_phi_rad = math.radians(d_phi)

    at_zenith = np.isclose(theta_deg, 0.0)
    at_pole = np.isclose(np.abs(theta_deg), 90.0)
    general = ~at_zenith & ~at_pole

    areas = np.empty_like(theta_deg)
    areas[at_zenith] = (
        4.0 * np.pi / d_phi_rad * np.sin(d_theta_rad / 4.0) ** 2
    )
    areas[at_pole] = np.sin(d_theta_rad / 2.0)
    areas[general] = (
        2.0 * np.sin(theta[general]) * np.sin(d_theta_rad / 2.0)
    )
    a_min = np.min(areas)
    if a_min <= 0.0:
        raise ValueError("Area factors must be positive; check the angles.")
    return np.asarray(areas / a_min, dtype=np.float64)


def random_incidence_diffusion(
    directional_coefficients: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Random-incidence diffusion coefficient ``d`` (ISO 17497-2, Clause 8.4).

    The (normalised or non-normalised) directional coefficients are averaged
    over the source positions. Hemispherical measurements use **equal**
    weightings (``weights is None``); two-dimensional (single-plane)
    measurements use the source weighting of Clause 8.4 - weight 1 for the
    0 deg source and weight 3 for each of the four +/-30 deg, +/-60 deg sources
    (see :data:`TWO_DIMENSIONAL_SOURCE_WEIGHTS`).

    :param directional_coefficients: Directional diffusion coefficients
        ``d_theta`` (or ``d_theta_n``), one per source position (1-D).
    :param weights: Optional source-position weights; ``None`` averages with
        equal weight.
    :return: Random-incidence diffusion coefficient ``d`` (a scalar).
    :raises ValueError: for an empty or non-1-D input, a length mismatch, or
        non-positive total weight.
    """
    d = np.asarray(directional_coefficients, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(
            "'directional_coefficients' must be a 1-D sequence."
        )
    if d.size == 0:
        raise ValueError("'directional_coefficients' must not be empty.")
    if weights is None:
        w = np.ones(d.size, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != d.size:
            raise ValueError(
                "'weights' must match the number of source positions "
                f"({d.size}), got shape {w.shape}."
            )
        if np.any(w < 0.0):
            raise ValueError("'weights' values must be non-negative.")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("The total source weight must be positive.")
    return float(np.sum(w * d) / total)


# --- Deprecated alias (phonometry 3.1 rename; remove in 4.0) -------------

def __getattr__(name: str) -> Any:
    """PEP 562 shim warning for the renamed band constant."""
    if name == "BASE_PLATE_BANDS_HZ":
        _warn_renamed("BASE_PLATE_BANDS_HZ", "BASE_PLATE_BANDS")
        return BASE_PLATE_BANDS
    raise AttributeError(
        f"module 'phonometry.scattering_diffusion' has no attribute {name!r}"
    )
