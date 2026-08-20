#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Sound absorption in a reverberation room: BS EN ISO 354:2003.

The mean reverberation time of a reverberation room is measured empty and with
the test specimen installed. From those two reverberation times the equivalent
sound absorption area of the specimen is obtained via Sabine's equation, and for
a plane absorber the sound absorption coefficient follows by dividing by the
covered area (ISO 354:2003, Clauses 4 and 8.1).

Equivalent sound absorption area (ISO 354:2003, Eq. (5) empty room / Eq. (7)
with specimen; identical form):

.. math::

   A = \frac{55.3 V}{c T} - 4 V m

with ``V`` the room volume (m3), ``c`` the speed of sound (m/s), ``T`` the
reverberation time (s) and ``m`` the power attenuation coefficient of air (1/m).
The speed of sound follows Eq. (6), valid for 15 degC to 30 degC:

.. math::

   c = \left( 331 + 0.6\,t/{}^{\circ}\text{C} \right)~\text{m/s}

The equivalent sound absorption area of the specimen and its absorption
coefficient (ISO 354:2003, Eq. (8) and Eq. (9)):

.. math::

   A_\mathrm{T} = A_2 - A_1
   = 55.3 V \left( \frac{1}{c_2 T_2} - \frac{1}{c_1 T_1} \right)
   - 4 V (m_2 - m_1)

   \alpha_\mathrm{s} = \frac{A_\mathrm{T}}{S}

``alpha_s`` may exceed 1.0 (e.g. from diffraction/edge effects) and is not a
percentage (ISO 354:2003, Clause 3.7 NOTE 2); it is therefore never clamped.

The air attenuation coefficient ``m`` is defined by ISO 354 only through its
conversion from the ISO 9613-1 attenuation coefficient ``alpha`` (in dB/m)
(ISO 354:2003, 8.1.2.1):

.. math::

   m = \frac{\alpha}{10 \log_{10} e}

ISO 354 otherwise defers the calculation of ``alpha`` entirely to ISO 9613-1.
``m`` is therefore a user-supplied per-band parameter here (default 0, i.e. no air
correction); a caller holding ISO 9613-1 ``alpha`` values can convert them with
:func:`attenuation_from_alpha`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

#: Sabine constant of ISO 354:2003, Eq. (5)/(7) (55,3 exactly as printed).
_SABINE = 55.3
#: ``10 * lg e`` (ISO 354:2003, 8.1.2.1), lg = log10; ~= 4,342944819.
_TEN_LG_E = 10.0 * math.log10(math.e)
#: Validity range of the speed-of-sound Eq. (6), in degrees Celsius.
_EQ6_TEMPERATURE_RANGE = (15.0, 30.0)
#: Minimum reverberation-room volume, ISO 354:2003 clause 6.1.1 (m3).
_MIN_ROOM_VOLUME = 150.0
#: Plane-absorber sample-area limits, ISO 354:2003 clause 6.2.1.1 (m2).
_SAMPLE_AREA_MIN = 10.0
_SAMPLE_AREA_MAX = 12.0
#: Reference volume for scaling the upper sample-area limit, clause 6.2.1.1 (m3).
_SAMPLE_AREA_REF_VOLUME = 200.0


class AbsorptionWarning(PhonometryWarning):
    """Advisory for out-of-range or non-physical ISO 354 absorption inputs."""


def _warn_small_room(volume: float, *, stacklevel: int) -> None:
    r"""Advise when the room is below the ISO 354 clause 6.1.1 minimum volume.

    Clause 6.1.1 requires :math:`V \ge 150` m3 (new rooms are recommended to
    be at
    least 200 m3). Below that the modal density is too low for a reliable diffuse
    field; the result is still returned.
    """
    if volume < _MIN_ROOM_VOLUME:
        warnings.warn(
            f"Room volume {volume:g} m3 is below the {_MIN_ROOM_VOLUME:g} m3 "
            "minimum of ISO 354:2003 clause 6.1.1 (new rooms recommended "
            ">= 200 m3); the result is advisory.",
            AbsorptionWarning,
            stacklevel=stacklevel,
        )


def _warn_sample_area(sample_area: float, volume: float, *, stacklevel: int) -> None:
    r"""Advise when ``S`` is outside the ISO 354 clause 6.2.1.1 range.

    Clause 6.2.1.1 requires a plane-absorber area :math:`10 \le S \le 12` m2;
    for :math:`V > 200` m3 the upper limit is multiplied by
    :math:`(V/200)^{2/3}`. The result
    is still returned.
    """
    upper = _SAMPLE_AREA_MAX
    if volume > _SAMPLE_AREA_REF_VOLUME:
        upper *= (volume / _SAMPLE_AREA_REF_VOLUME) ** (2.0 / 3.0)
    if not _SAMPLE_AREA_MIN <= sample_area <= upper:
        warnings.warn(
            f"Sample area {sample_area:g} m2 is outside the ISO 354:2003 clause "
            f"6.2.1.1 range [{_SAMPLE_AREA_MIN:g}, {upper:g}] m2; the result is "
            "advisory.",
            AbsorptionWarning,
            stacklevel=stacklevel,
        )


def _speed_of_sound(temperature: float) -> float:
    """Speed of sound from air temperature (ISO 354:2003, Eq. (6)).

    :param temperature: Air temperature, in degrees Celsius (valid 15..30).
    :return: Propagation speed of sound, in metres per second.
    """
    return 331.0 + 0.6 * temperature


def _resolve_speed(temperature: float, speed_of_sound: float | None) -> float:
    """Return ``speed_of_sound`` if given, else Eq. (6) from ``temperature``.

    Warns when the speed is derived from a temperature outside the 15..30 degC
    validity range of Eq. (6). An explicit ``speed_of_sound`` bypasses the check.
    """
    if speed_of_sound is not None:
        if speed_of_sound <= 0.0:
            raise ValueError("'speed_of_sound' must be positive.")
        return float(speed_of_sound)
    lo, hi = _EQ6_TEMPERATURE_RANGE
    if not lo <= temperature <= hi:
        warnings.warn(
            f"Temperature {temperature} degC is outside the 15..30 degC validity "
            "range of ISO 354:2003 Eq. (6); speed of sound may be inaccurate.",
            AbsorptionWarning,
            stacklevel=3,
        )
    return _speed_of_sound(temperature)


def attenuation_from_alpha(alpha: ArrayLike) -> NDArray[np.float64]:
    r"""Air power attenuation coefficient ``m`` from ISO 9613-1 ``alpha``.

    Applies the ISO 354:2003 (8.1.2.1) conversion
    :math:`m = \alpha / (10 \log_{10} e)`,
    where ``alpha`` is the attenuation coefficient in decibels per metre used by
    ISO 9613-1 and ``m`` is the power attenuation coefficient in reciprocal
    metres entering Eq. (5)/(7)/(8). ISO 354 itself provides no ``alpha`` table
    or formula (it defers to ISO 9613-1); this helper only performs the unit
    conversion for a caller who already holds ``alpha`` values.

    :param alpha: Attenuation coefficient, in dB/m (scalar or per band).
    :return: Power attenuation coefficient ``m``, in 1/m.
    """
    a = np.asarray(alpha, dtype=np.float64)
    if np.any(a < 0.0):
        raise ValueError("'alpha' must be non-negative.")
    return a / _TEN_LG_E


def _validate_area_inputs(
    t: NDArray[np.float64], volume: float, m: NDArray[np.float64]
) -> None:
    if volume <= 0.0:
        raise ValueError("'volume' must be positive.")
    if np.any(t <= 0.0):
        raise ValueError("Reverberation times must be positive.")
    if np.any(m < 0.0):
        raise ValueError("Air attenuation coefficient 'm' must be non-negative.")
    if m.ndim != 0 and m.shape != t.shape:
        raise ValueError(
            "'m' must be a scalar or an array matching the shape of 't60'."
        )


def _absorption_area(
    t60: ArrayLike,
    volume: float,
    *,
    temperature: float,
    speed_of_sound: float | None,
    m: ArrayLike,
) -> NDArray[np.float64]:
    """Core Eq. (5)/(7) evaluation without the advisory volume warning.

    Shared by :func:`absorption_area` (which adds the ISO 354 clause 6.1.1
    volume advisory) and :func:`absorption_coefficient` (which advises the
    volume once for the pair of measurements)."""
    t = np.asarray(t60, dtype=np.float64)
    m_arr = np.asarray(m, dtype=np.float64)
    _validate_area_inputs(t, volume, m_arr)
    c = _resolve_speed(temperature, speed_of_sound)
    return _SABINE * volume / (c * t) - 4.0 * volume * m_arr


def absorption_area(
    t60: ArrayLike,
    volume: float,
    *,
    temperature: float = 20.0,
    speed_of_sound: float | None = None,
    m: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    r"""Equivalent sound absorption area of a room (ISO 354:2003, Eq. (5)/(7)).

    :math:`A = 55.3 V / (c T) - 4 V m`. This is Sabine's equation with the
    air-absorption term; it gives the empty-room area ``A1`` from ``T1`` or the
    with-specimen area ``A2`` from ``T2`` (both equations have identical form).

    :param t60: Reverberation time(s) ``T``, in seconds (scalar or per band).
    :param volume: Room volume ``V``, in cubic metres.
    :param temperature: Air temperature, in degrees Celsius, used to compute the
        speed of sound via Eq. (6) when ``speed_of_sound`` is not given
        (default 20 degC, i.e. c = 343 m/s). A temperature outside 15..30 degC
        emits an :class:`AbsorptionWarning`. A room volume below the 150 m3
        minimum of clause 6.1.1 likewise emits an advisory :class:`AbsorptionWarning`.
    :param speed_of_sound: Explicit speed of sound ``c``, in m/s; overrides
        ``temperature`` and Eq. (6) when supplied.
    :param m: Power attenuation coefficient of air ``m``, in 1/m (a scalar or an
        array matching the shape of ``t60``; default 0, i.e. no air correction).
        A per-band ``m`` whose shape differs from ``t60`` raises ``ValueError``.
        Obtain it from an ISO 9613-1 attenuation coefficient with
        :func:`attenuation_from_alpha`.
    :return: Equivalent sound absorption area ``A``, in square metres, with the
        shape of ``t60``.
    """
    area = _absorption_area(
        t60, volume, temperature=temperature, speed_of_sound=speed_of_sound, m=m
    )
    _warn_small_room(volume, stacklevel=3)
    return area


def absorption_coefficient(
    t1: ArrayLike,
    t2: ArrayLike,
    volume: float,
    sample_area: float,
    *,
    temperature1: float = 20.0,
    temperature2: float | None = None,
    speed_of_sound1: float | None = None,
    speed_of_sound2: float | None = None,
    m1: ArrayLike = 0.0,
    m2: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    r"""Sound absorption coefficient of a plane absorber (ISO 354:2003, Eq. (9)).

    Builds the equivalent sound absorption area of the specimen from Eq. (8),
    using the empty-room reverberation time ``T1`` and the with-specimen time
    ``T2``:

    .. math::

       A_\mathrm{T} = A_2 - A_1
       = 55.3 V \left( \frac{1}{c_2 T_2} - \frac{1}{c_1 T_1} \right)
       - 4 V (m_2 - m_1)

    then returns :math:`\alpha_\mathrm{s} = A_\mathrm{T} / S` (Eq. (9)).

    The two measurements may be at different temperatures; ``c1`` and ``c2`` are
    resolved independently. ``alpha_s`` is returned unclamped and may exceed 1.0
    (Clause 3.7 NOTE 2). Because adding an absorber must reduce the
    reverberation time, :math:`T_2 \ge T_1` (:math:`\alpha_\mathrm{s} \le 0`) is
    non-physical and emits an :class:`AbsorptionWarning`. A room volume below
    the 150 m3 minimum of clause 6.1.1, or a sample area outside the clause
    6.2.1.1 range (:math:`10~\text{m}^2 \le S \le 12~\text{m}^2`, upper limit
    scaled by :math:`(V/200)^{2/3}` when :math:`V > 200` m3), each emit an
    advisory :class:`AbsorptionWarning`.

    :param t1: Empty-room reverberation time(s) ``T1``, in seconds.
    :param t2: With-specimen reverberation time(s) ``T2``, in seconds.
    :param volume: Room volume ``V``, in cubic metres.
    :param sample_area: Area ``S`` covered by the test specimen, in square metres
        (for both-sides-exposed absorbers, the area of the two sides;
        Clause 3.7 NOTE 1).
    :param temperature1: Empty-room air temperature, in degrees Celsius
        (default 20). Used for ``c1`` via Eq. (6) unless ``speed_of_sound1`` is
        given.
    :param temperature2: With-specimen air temperature, in degrees Celsius;
        defaults to ``temperature1``. Used for ``c2`` unless ``speed_of_sound2``
        is given.
    :param speed_of_sound1: Explicit ``c1`` in m/s; overrides ``temperature1``.
    :param speed_of_sound2: Explicit ``c2`` in m/s; overrides ``temperature2``.
        Defaults to ``speed_of_sound1`` when that is given but ``c2`` is not, so
        overriding only ``c1`` applies the same speed to both measurements.
    :param m1: Empty-room air attenuation coefficient ``m1``, in 1/m (default 0).
    :param m2: With-specimen air attenuation coefficient ``m2``, in 1/m
        (default 0).
    :return: Sound absorption coefficient ``alpha_s`` with the broadcast shape of
        ``t1`` and ``t2``.
    """
    if sample_area <= 0.0:
        raise ValueError("'sample_area' must be positive.")
    if volume <= 0.0:
        raise ValueError("'volume' must be positive.")
    if temperature2 is None:
        temperature2 = temperature1
    if speed_of_sound2 is None:
        speed_of_sound2 = speed_of_sound1
    # Advisory setup checks (result still returned). Volume is advised here once
    # via the module-private core helper (which does not warn) to avoid
    # duplicate volume advisories from the two area evaluations.
    _warn_small_room(volume, stacklevel=2)
    _warn_sample_area(sample_area, volume, stacklevel=2)
    a1 = _absorption_area(
        t1,
        volume,
        temperature=temperature1,
        speed_of_sound=speed_of_sound1,
        m=m1,
    )
    a2 = _absorption_area(
        t2,
        volume,
        temperature=temperature2,
        speed_of_sound=speed_of_sound2,
        m=m2,
    )
    area_specimen = a2 - a1
    alpha_s = area_specimen / sample_area
    if np.any(alpha_s <= 0.0):
        warnings.warn(
            "alpha_s <= 0 (T2 >= T1): adding the specimen did not reduce the "
            "reverberation time; check the measurement (ISO 354:2003, 8.1.2/8.1.3).",
            AbsorptionWarning,
            stacklevel=2,
        )
    return alpha_s


# --- one-third-octave measurement result (ISO 354:2003, Clause 8) --------


@dataclass(frozen=True)
class SoundAbsorptionMeasurement:
    """A reverberation-room sound absorption measurement (ISO 354:2003).

    The one-third-octave outcome of a plane-absorber test: the mean
    reverberation time measured empty (``T1``) and with the specimen (``T2``),
    the equivalent sound absorption areas ``A1`` (Eq. (5)) and ``A2`` (Eq. (7))
    they give through Sabine's equation, and the sound absorption coefficient
    ``alpha_s`` (Eq. (8)/(9)) of the specimen. Build it with
    :func:`measure_sound_absorption`; the frozen instance then exposes
    :meth:`plot` (``alpha_s`` versus frequency) and :meth:`report` (an
    accredited ISO 354 test-report PDF).

    A single-number rating (the practical coefficient ``alpha_p`` and the
    weighted coefficient ``alpha_w``) is defined by ISO 11654, not ISO 354, and
    is therefore not produced here; pass ``alpha_s`` to
    :func:`~phonometry.materials.weighted_absorption_from_third_octave` for it.

    :ivar frequencies: One-third-octave band centre frequencies, in Hz
        (the ISO 354 range is 100 Hz to 5000 Hz).
    :ivar t_empty: Mean reverberation time of the empty room ``T1``, per band,
        in seconds.
    :ivar t_specimen: Mean reverberation time of the room with the specimen
        ``T2``, per band, in seconds.
    :ivar volume: Reverberation-room volume ``V``, in cubic metres.
    :ivar area: Area ``S`` covered by the test specimen, in square metres.
    :ivar temperature: Air temperature during the test, in degrees Celsius.
    :ivar humidity: Relative humidity during the test, in %, or ``None`` when
        not recorded. It is informational: humidity enters ISO 354 only through
        the air attenuation coefficient ``m`` (via ISO 9613-1), never directly.
    :ivar speed_of_sound: Propagation speed of sound ``c`` used in the Sabine
        inversion, in m/s (from Eq. (6) unless it was given explicitly).
    :ivar air_attenuation: Power attenuation coefficient of air ``m``, per band,
        in 1/m (``0`` when no air correction was applied).
    :ivar absorption_area_empty: Equivalent sound absorption area of the empty
        room ``A1`` (Eq. (5)), per band, in square metres.
    :ivar absorption_area_with_specimen: Equivalent sound absorption area of the
        room containing the specimen ``A2`` (Eq. (7)), per band, in square
        metres.
    :ivar alpha_s: Sound absorption coefficient ``alpha_s`` (Eq. (8)/(9)), per
        band. It may exceed 1,0 (Clause 3.7 NOTE 2) and is never clamped.
    """

    frequencies: NDArray[np.float64]
    t_empty: NDArray[np.float64]
    t_specimen: NDArray[np.float64]
    volume: float
    area: float
    temperature: float
    humidity: float | None
    speed_of_sound: float
    air_attenuation: NDArray[np.float64]
    absorption_area_empty: NDArray[np.float64]
    absorption_area_with_specimen: NDArray[np.float64]
    alpha_s: NDArray[np.float64]

    @property
    def equivalent_absorption_area(self) -> NDArray[np.float64]:
        r"""Equivalent sound absorption area :math:`A_\mathrm{T} = A_2 - A_1`.

        The ISO 354:2003 Eq. (8) quantity, per band, in square metres; dividing
        it by the specimen area ``S`` gives :attr:`alpha_s` (Eq. (9)).
        """
        return self.absorption_area_with_specimen - self.absorption_area_empty

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the sound absorption coefficient ``alpha_s`` versus frequency.

        Draws ``alpha_s`` over the one-third-octave band axis (ISO 354). Values
        above 1,0 are kept (Clause 3.7 NOTE 2), so the axis grows to show them.
        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the ``alpha_s`` curve ``plot`` call.
        :return: The axes.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_sound_absorption

        check_language(language)
        return plot_sound_absorption(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 354 sound-absorption test-report fiche to a PDF.

        Writes a one-page accredited reverberation-room report: the
        standard-basis line, an optional metadata header block (client,
        specimen, area ``S``, room volume ``V``, mounting, climate ...), a
        two-panel body with the one-third-octave ``alpha_s`` table beside the
        ``alpha_s`` curve, and a footer with the fixed disclaimer. ISO 354 is a
        characterisation, so there is no pass/fail verdict and no single-number
        rating (the weighted ``alpha_w`` is an ISO 11654 quantity, out of scope
        here).

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a body-and-disclaimer fiche without a metadata header. The
            ``requirement`` field is ignored (ISO 354 has no verdict).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table adds the reverberation times
            ``T1``/``T2`` and the equivalent absorption areas ``A1``/``A2``.
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
        from ..._report.iso354 import render_iso354_report

        return render_iso354_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def measure_sound_absorption(
    frequencies: ArrayLike,
    t_empty: ArrayLike,
    t_specimen: ArrayLike,
    *,
    volume: float,
    area: float,
    temperature: float = 20.0,
    humidity: float | None = None,
    speed_of_sound: float | None = None,
    m: ArrayLike = 0.0,
) -> SoundAbsorptionMeasurement:
    r"""Measure the sound absorption of a plane absorber (ISO 354:2003).

    Assembles a :class:`SoundAbsorptionMeasurement` from the one-third-octave
    reverberation times of the empty room (``T1``) and of the room with the
    specimen installed (``T2``). The equivalent sound absorption areas ``A1``
    and ``A2`` follow from Sabine's equation (Eq. (5)/(7), delegated to
    :func:`absorption_area`) and the sound absorption coefficient
    :math:`\alpha_\mathrm{s} = (A_2 - A_1)/S` from Eq. (8)/(9) (delegated to
    :func:`absorption_coefficient`); no formula is re-derived here.

    Both measurements are taken at the same air temperature and, for the air
    attenuation term, the same climatic conditions (ISO 354:2003, 6.3), so a
    single ``temperature`` and ``m`` apply to both. Use the lower-level
    :func:`absorption_coefficient` directly when the empty-room and
    with-specimen climates differ.

    :param frequencies: One-third-octave band centre frequencies, in Hz (the
        ISO 354 range is 100 Hz to 5000 Hz); a 1-D array matching ``t_empty``
        and ``t_specimen``.
    :param t_empty: Empty-room reverberation time ``T1``, per band, in seconds.
    :param t_specimen: With-specimen reverberation time ``T2``, per band, in
        seconds.
    :param volume: Reverberation-room volume ``V``, in cubic metres. A volume
        below the 150 m3 minimum of clause 6.1.1 emits an advisory
        :class:`AbsorptionWarning`.
    :param area: Area ``S`` covered by the test specimen, in square metres. An
        area outside the clause 6.2.1.1 range (10 m2 to 12 m2, upper limit
        scaled by :math:`(V/200)^{2/3}` for :math:`V > 200` m3) emits an
        advisory :class:`AbsorptionWarning`.
    :param temperature: Air temperature during the test, in degrees Celsius
        (default 20). Used for the speed of sound via Eq. (6) unless
        ``speed_of_sound`` is given; a temperature outside 15..30 degC emits an
        :class:`AbsorptionWarning`.
    :param humidity: Relative humidity during the test, in % (informational;
        recorded on the result but not used in the computation, which sees the
        climate only through ``m``). ``None`` leaves it unrecorded.
    :param speed_of_sound: Explicit speed of sound ``c``, in m/s; overrides
        ``temperature`` and Eq. (6) when supplied.
    :param m: Power attenuation coefficient of air ``m``, in 1/m (a scalar or a
        per-band array matching ``frequencies``; default 0, i.e. no air
        correction). Obtain it from an ISO 9613-1 attenuation coefficient with
        :func:`attenuation_from_alpha`.
    :return: A frozen :class:`SoundAbsorptionMeasurement`.
    :raises ValueError: If the frequency and reverberation-time arrays do not
        share one shape, or an input is non-physical (see
        :func:`absorption_coefficient`).
    """
    freqs = np.asarray(frequencies, dtype=np.float64)
    t1 = np.asarray(t_empty, dtype=np.float64)
    t2 = np.asarray(t_specimen, dtype=np.float64)
    if not (freqs.shape == t1.shape == t2.shape):
        raise ValueError(
            "'frequencies', 't_empty' and 't_specimen' must share one shape; "
            f"got {freqs.shape}, {t1.shape} and {t2.shape}."
        )
    m_arr = np.broadcast_to(np.asarray(m, dtype=np.float64), freqs.shape).astype(
        np.float64, copy=True
    )
    # Resolve the speed once (Eq. (6)); this emits the single temperature
    # advisory. Passing the resolved speed to the reused helpers below keeps
    # every advisory to exactly one, since both measurements share the climate.
    c = _resolve_speed(temperature, speed_of_sound)
    # A1/A2 reuse the Eq. (5)/(7) evaluation.
    a1 = _absorption_area(
        t1, volume, temperature=temperature, speed_of_sound=c, m=m_arr
    )
    a2 = _absorption_area(
        t2, volume, temperature=temperature, speed_of_sound=c, m=m_arr
    )
    # alpha_s reuses the validated Eq. (8)/(9) path (it also emits the volume,
    # sample-area and non-physical advisories exactly once).
    alpha_s = absorption_coefficient(
        t1,
        t2,
        volume,
        area,
        temperature1=temperature,
        speed_of_sound1=c,
        m1=m_arr,
        m2=m_arr,
    )
    return SoundAbsorptionMeasurement(
        frequencies=freqs,
        t_empty=t1,
        t_specimen=t2,
        volume=float(volume),
        area=float(area),
        temperature=float(temperature),
        humidity=None if humidity is None else float(humidity),
        speed_of_sound=c,
        air_attenuation=m_arr,
        absorption_area_empty=a1,
        absorption_area_with_specimen=a2,
        alpha_s=alpha_s,
    )
