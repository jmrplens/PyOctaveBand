#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
In-situ sound absorption of road surfaces (ISO 13472-1 / ISO 13472-2).

Two complementary standardised in-situ methods are supported here. They target
opposite ends of the absorption scale and are **not** interchangeable:

* **BS ISO 13472-1:2002** - *extended surface method*. A free-field impulse
  response is measured over the road, the direct (incident) and surface
  (reflected) components are separated in the time domain by the subtraction
  technique and an **Adrienne-type temporal window** (Clause 6.4), transformed
  to frequency, and ratioed. The normal-incidence sound absorption coefficient
  is (Clause 4.1):

  .. math::

     \alpha(f) = 1 - Q_W(f)
     = 1 - \frac{1}{K_r^{2}}
     \left\lvert \frac{H_\mathrm{r}(f)}{H_\mathrm{i}(f)} \right\rvert^{2}

  with ``Hi``/``Hr`` the incident/reflected transfer functions and ``Kr`` the
  geometrical-spreading factor :math:`K_r = (d_\mathrm{s} - d_\mathrm{m}) / (d_\mathrm{s} + d_\mathrm{m})` for the
  mandatory geometry :math:`d_\mathrm{s} = 1.25` m (source-to-plane) and
  :math:`d_\mathrm{m} = 0.25` m (mic-to-plane), giving :math:`K_r = 2/3` (Clause 4.2 /
  Annex C). The complex pressure reflection factor
  :math:`Q_p = (1/K_r)(H_\mathrm{r}/H_\mathrm{i})\, e^{+j 2 \pi f \Delta\tau}` with
  :math:`\Delta\tau = 2 d_\mathrm{m} / c` (Annex C) is available for theory comparison.
  A highly reflective reference surface removes the electro-acoustic chain
  error and the geometry factor by a ratio (Annex B), and non-normal incidence
  uses :math:`K_{r,\theta}` (Annex F).

* **BS ISO 13472-2:2010** - *spot method*. This is an **in-situ application of
  the ISO 10534-2 two-microphone impedance tube** for reflective surfaces
  (measured ``alpha`` below ~0.15). Part 2 does not restate the
  transfer-function / reflection-factor / absorption mathematics; Clauses 4,
  5.7 and 6.6 defer it to ISO 10534-2. The core computation therefore lives in
  :func:`~phonometry.materials.absorbers.impedance_tube.two_microphone_impedance` and is **not**
  reimplemented here. This module contributes only the Part-2 tube
  geometry/validity helpers (upper usable frequency, microphone-spacing bounds,
  the 250-1600 Hz one-third-octave working range) and the Annex A internal-loss
  system correction
  :math:`\alpha \approx \alpha_{\text{measured}} - \alpha_{\text{system}}`.

Sign / normalisation convention (ISO 13472-1): the forward transform is
NumPy's ``rfft`` with kernel :math:`e^{-j 2 \pi f t}` (unnormalised); every
quantity here is a ratio of two transforms from the same processing chain, so
the transform normalisation cancels (Clause 6.1). A pure time delay ``tau`` of
the reflected path scales its spectrum by :math:`e^{-j 2 \pi f \tau}`; the
optional phase restoration multiplies by :math:`e^{+j 2 \pi f \tau}` to
recover ``Qp`` (Annex C /
Annex G, resolving the Clause 4.1 NOTE shorthand to the frequency-dependent
form).

The window durations of the Adrienne window are **not** fixed by the standard:
Clause 6.4 mandates only a sharp leading edge, a 5 ms flat portion and a
cosine-squared or Blackman-Harris trailing edge, with the shape and lengths
**reported per measurement**. They are therefore configurable here and default
to a short leading edge, a 5 ms flat top and a Blackman-Harris trailing edge
(the Annex E example report); the historical fixed edge timings are **not**
hard-coded as if normative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._internal.types import Real
from ..._internal.warnings import PhonometryWarning

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

Complex = NDArray[np.complex128]

#: Mandatory source-to-reference-plane distance ``ds`` (ISO 13472-1, 4.2), m.
DEFAULT_SOURCE_HEIGHT = 1.25
#: Mandatory microphone-to-reference-plane distance ``dm`` (ISO 13472-1, 4.2), m.
DEFAULT_MIC_HEIGHT = 0.25
#: Speed of sound of the Annex A worked example (ISO 13472-1), in m/s.
DEFAULT_SPEED_OF_SOUND = 340.0

#: Flat-portion duration mandated by ISO 13472-1 Clause 6.4, in seconds (5 ms).
_ADRIENNE_FLAT = 5.0e-3
#: Default short leading-edge duration (implementation-defined, A1), in seconds.
_ADRIENNE_LEADING = 0.5e-3
#: Default Blackman-Harris trailing-edge duration (A1), in seconds.
_ADRIENNE_TRAILING = 5.0e-3
#: Four-term Blackman-Harris coefficients (ISO 13472-1 Clause 6.4 trailing edge).
_BLACKMAN_HARRIS = (0.35875, 0.48829, 0.14128, 0.01168)
_EDGE_SHAPES = ("blackman-harris", "cosine-squared")

#: One-third-octave working range of ISO 13472-1 (Scope), in hertz.
PART1_FREQUENCY_RANGE = (250.0, 4000.0)
#: One-third-octave working range of ISO 13472-2 (Scope), in hertz.
SPOT_FREQUENCY_RANGE = (250.0, 1600.0)
#: Narrow-band equivalent range of ISO 13472-2 (Clause 3.1 NOTE), in hertz.
SPOT_NARROW_BAND_RANGE = (220.0, 1800.0)

#: Nominal one-third-octave midband frequencies spanning both methods, in hertz.
_ONE_THIRD_OCTAVE_CENTERS = (
    250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
    2000.0, 2500.0, 3150.0, 4000.0,
)

#: ISO 13472-2 Clause 5.4.1 upper-frequency factor ``f_u = 0.58 c0 / d``.
_SPOT_FU_FACTOR = 0.58
#: ISO 13472-2 Clause 5.4.2 maximum-spacing factor ``s_max < 0.45 c0 / f_max``.
_SPOT_SMAX_FACTOR = 0.45
#: ISO 13472-2 Clause 5.4.2 minimum-spacing factor ``s_min > 0.05 c0 / f_min``.
_SPOT_SMIN_FACTOR = 0.05

__all__ = [
    "DEFAULT_MIC_HEIGHT",
    "DEFAULT_SOURCE_HEIGHT",
    "DEFAULT_SPEED_OF_SOUND",
    "PART1_FREQUENCY_RANGE",
    "SPOT_FREQUENCY_RANGE",
    "SPOT_NARROW_BAND_RANGE",
    "InsituAbsorptionResult",
    "RoadAbsorptionWarning",
    "absorption_reference_corrected",
    "adrienne_window",
    "check_spot_frequency_range",
    "geometric_spreading_factor",
    "geometric_spreading_factor_angle",
    "insitu_absorption_coefficient",
    "insitu_absorption_from_reflection",
    "insitu_absorption_spectrum",
    "insitu_reflection_factor",
    "max_sampled_area_radius",
    "msa_major_axis",
    "one_third_octave_absorption",
    "power_reflection_coefficient",
    "reflected_path_delay",
    "spot_internal_loss_correction",
    "spot_microphone_spacing_bounds",
    "spot_tube_upper_frequency",
]


class RoadAbsorptionWarning(PhonometryWarning):
    """Advisory for out-of-range in-situ road-absorption frequencies."""


# --------------------------------------------------------------------------- #
# Geometry (ISO 13472-1 Clause 4.1 / Annex C / Annex F)
# --------------------------------------------------------------------------- #
#: Rejection messages the entry points of this module share.
_FS_POSITIVE = "'fs' must be positive."
_SPEED_POSITIVE = "'speed_of_sound' must be positive."

def geometric_spreading_factor(
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
) -> float:
    r"""Geometrical-spreading factor ``Kr`` (ISO 13472-1:2002, Clause 4.1).

    :math:`K_r = (d_\mathrm{s} - d_\mathrm{m}) / (d_\mathrm{s} + d_\mathrm{m})`. It corrects the reflected path
    for the extra spherical spreading over the image-source distance
    :math:`d_\mathrm{s} + d_\mathrm{m}` relative to the direct distance :math:`d_\mathrm{s} - d_\mathrm{m}`
    (Annex C). The mandatory geometry :math:`d_\mathrm{s} = 1.25` m,
    :math:`d_\mathrm{m} = 0.25` m gives :math:`K_r = 2/3` (Clause 4.2).

    :param source_height: Source-to-reference-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-reference-plane distance ``dm``, in metres.
    :return: Geometrical-spreading factor ``Kr`` (dimensionless,
        :math:`0 < K_r < 1`).
    :raises ValueError: If ``ds`` or ``dm`` is not positive or
        :math:`d_\mathrm{s} \le d_\mathrm{m}`.
    """
    _check_geometry(source_height, mic_height)
    return float((source_height - mic_height) / (source_height + mic_height))


def geometric_spreading_factor_angle(
    incidence_angle: float,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
) -> float:
    r"""Oblique geometrical-spreading factor (ISO 13472-1, Annex F).

    :math:`K_{r,\theta}^2 = 1 - \cos^2(\theta) (1 - K_r^2)` with ``Kr`` the
    normal-incidence factor of :func:`geometric_spreading_factor`. At
    :math:`\theta = 0` the
    cosine is unity and ``Kr,theta`` collapses to ``Kr`` (Clause 4.1).

    :param incidence_angle: Incidence angle ``theta``, in **radians**.
    :param source_height: Source-to-reference-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-reference-plane distance ``dm``, in metres.
    :return: Oblique factor ``Kr,theta`` (positive root, dimensionless).
    """
    kr = geometric_spreading_factor(source_height, mic_height)
    cos_sq = float(np.cos(incidence_angle)) ** 2
    return float(np.sqrt(1.0 - cos_sq * (1.0 - kr**2)))


def reflected_path_delay(
    mic_height: float = DEFAULT_MIC_HEIGHT,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> float:
    r"""Reflected-path arrival delay ``dtau`` (ISO 13472-1:2002, Annex C).

    :math:`\Delta\tau = 2 d_\mathrm{m} / c` is the time difference between the direct
    and the surface-reflected impulses for the normal-incidence geometry; it
    is the delay undone by the phase-restoration term of
    :func:`insitu_reflection_factor`.

    :param mic_height: Microphone-to-reference-plane distance ``dm``, in metres.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :return: Delay ``dtau``, in seconds.
    :raises ValueError: If ``dm`` or ``c`` is not positive.
    """
    if mic_height <= 0.0:
        raise ValueError("'mic_height' must be positive.")
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_POSITIVE)
    return float(2.0 * mic_height / speed_of_sound)


# --------------------------------------------------------------------------- #
# Adrienne temporal window (ISO 13472-1 Clause 6.4)
# --------------------------------------------------------------------------- #
def _blackman_harris_full(length: int) -> NDArray[np.float64]:
    """Symmetric four-term Blackman-Harris window of ``length`` samples."""
    a0, a1, a2, a3 = _BLACKMAN_HARRIS
    k = np.arange(length, dtype=np.float64)
    x = 2.0 * np.pi * k / (length - 1)
    return np.asarray(
        a0 - a1 * np.cos(x) + a2 * np.cos(2.0 * x) - a3 * np.cos(3.0 * x),
        dtype=np.float64,
    )


def _edge(n_samples: int, shape: str, *, rising: bool) -> NDArray[np.float64]:
    """Monotonic half-window taper of ``n_samples`` samples in ``[0, 1]``."""
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)
    if shape == "blackman-harris":
        full = _blackman_harris_full(2 * n_samples)
        half = np.asarray(
            full[:n_samples] if rising else full[n_samples:], dtype=np.float64
        )
        # The symmetric window peaks (== 1) between samples ``n_samples-1`` and
        # ``n_samples``, so the half taper stops just short of 1 at its inner
        # end. Rescale so the junction with the flat top is exactly 1, keeping
        # the taper length unchanged and the flat-to-edge transition continuous.
        inner = half[-1] if rising else half[0]
        return np.asarray(half / inner, dtype=np.float64)
    # Cosine-squared (Hann) half-taper: sin^2 rising, cos^2 falling.
    k = np.arange(1, n_samples + 1, dtype=np.float64)
    arg = 0.5 * np.pi * k / n_samples
    taper = np.sin(arg) ** 2 if rising else np.cos(arg) ** 2
    return np.asarray(taper, dtype=np.float64)


def adrienne_window(
    fs: float | None = None,
    *,
    flat_duration: float = _ADRIENNE_FLAT,
    leading_duration: float = _ADRIENNE_LEADING,
    trailing_duration: float = _ADRIENNE_TRAILING,
    leading_edge: str = "blackman-harris",
    trailing_edge: str = "blackman-harris",
) -> Real:
    """Adrienne-type temporal window (ISO 13472-1:2002, Clause 6.4).

    Clause 6.4 mandates a **sharp leading edge**, a **5 ms flat portion** and a
    **cosine-squared or Blackman-Harris trailing edge** so as to suppress
    frequency-domain oscillations (Figure 4); the shape and the durations are
    *reported per measurement* rather than fixed by the standard. All three
    durations are therefore configurable. The window is the concatenation of a
    rising leading half-taper, a unit-valued flat portion and a falling trailing
    half-taper::

        w = [ rising_edge (0 -> 1) | flat (== 1) | trailing_edge (1 -> 0) ]

    The defaults (short 0.5 ms leading edge, 5 ms flat top, 5 ms Blackman-Harris
    trailing edge) follow the Annex E example report; they are **not** a
    normative fixed set of timings. The lower usable frequency scales as
    ``~ 1 / T_window`` (Clause 6.4), so report the durations used.

    :param fs: Sampling frequency, in hertz (ISO 13472-1 Clause 6.2 requires
        ``fs > 40 kHz`` in practice).
    :param flat_duration: Flat-portion duration, in seconds (default 5 ms).
    :param leading_duration: Leading-edge (rise) duration, in seconds; 0 gives a
        sharp step onset.
    :param trailing_duration: Trailing-edge (fall) duration, in seconds.
    :param leading_edge: Leading-edge shape, ``"blackman-harris"`` or
        ``"cosine-squared"``.
    :param trailing_edge: Trailing-edge shape, ``"blackman-harris"`` or
        ``"cosine-squared"``.
    :return: The time-domain window, one sample per ``1 / fs`` (length
        ``round((leading + flat + trailing) * fs)`` samples).
    :raises ValueError: If ``fs`` is missing or not positive, a duration is
        negative, the flat duration is not positive, or an edge shape is
        unknown.
    """
    if fs is None:
        raise ValueError("adrienne_window() missing required argument: 'fs'.")
    if fs <= 0.0:
        raise ValueError(_FS_POSITIVE)
    if flat_duration <= 0.0:
        raise ValueError("'flat_duration' must be positive.")
    if leading_duration < 0.0 or trailing_duration < 0.0:
        raise ValueError("Edge durations must be non-negative.")
    if leading_edge not in _EDGE_SHAPES or trailing_edge not in _EDGE_SHAPES:
        raise ValueError(f"Edge shapes must be one of {_EDGE_SHAPES}.")
    n_lead = round(leading_duration * fs)
    n_flat = round(flat_duration * fs)
    n_trail = round(trailing_duration * fs)
    if n_flat <= 0:
        raise ValueError("'flat_duration' is too short for 'fs'.")
    rising = _edge(n_lead, leading_edge, rising=True)
    flat = np.ones(n_flat, dtype=np.float64)
    falling = _edge(n_trail, trailing_edge, rising=False)
    return np.asarray(np.concatenate([rising, flat, falling]), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Reflection factor and absorption (ISO 13472-1 Clause 4.1 / Annex C)
# --------------------------------------------------------------------------- #
def _transfer_functions(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    n: int | None,
) -> tuple[Complex, Complex, int]:
    """Real FFTs of the (windowed) incident and reflected impulse responses.

    Returns the two spectra together with the time-domain FFT length actually
    used, so callers needing ``rfftfreq`` do not have to reconstruct it from the
    bin count (which is ambiguous for odd lengths).
    """
    hi_t = np.atleast_1d(np.asarray(incident_ir, dtype=np.float64))
    hr_t = np.atleast_1d(np.asarray(reflected_ir, dtype=np.float64))
    if hi_t.size == 0 or hr_t.size == 0:
        raise ValueError("Impulse responses must be non-empty.")
    length = n if n is not None else max(hi_t.size, hr_t.size)
    if length <= 0:
        raise ValueError("'n' must be positive.")
    hi = np.fft.rfft(hi_t, n=length)
    hr = np.fft.rfft(hr_t, n=length)
    return (
        np.asarray(hi, dtype=np.complex128),
        np.asarray(hr, dtype=np.complex128),
        length,
    )


def insitu_reflection_factor(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    incidence_angle: float = 0.0,
    fs: float | None = None,
    delay: float | None = None,
    n: int | None = None,
) -> Complex:
    r"""Complex pressure reflection factor ``r(f)`` (ISO 13472-1, Clause 4.1).

    :math:`r(f) = (1 / K_r) H_\mathrm{r}(f) / H_\mathrm{i}(f)` from the windowed reflected and
    incident
    impulse responses, with ``Hr``/``Hi`` their real FFTs and ``Kr`` the
    geometrical-spreading factor (or ``Kr,theta`` when ``incidence_angle`` is
    given, Annex F). When both ``fs`` and ``delay`` are supplied the
    reflected-path time offset is undone by
    :math:`\exp(+j 2 \pi f \, \text{delay})`, yielding
    the complex ``Qp`` of the Clause 4.1 NOTE (with ``delay`` set to
    :math:`\Delta\tau = 2 d_\mathrm{m} / c`,
    Annex C; the frequency-dependent form of Annex G).

    :param incident_ir: Windowed incident (direct-path) impulse response
        ``hi(t)``, real, one sample per ``1 / fs``.
    :param reflected_ir: Windowed reflected-path impulse response ``hr(t)``,
        real, same sampling as ``incident_ir``.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param incidence_angle: Incidence angle ``theta``, in radians (0 = normal).
    :param fs: Sampling frequency, in hertz; required with ``delay`` for phase
        restoration.
    :param delay: Reflected-path delay ``dtau`` to undo, in seconds; ``None``
        returns the raw spectral ratio.
    :param n: FFT length; defaults to the longer of the two impulse responses.
    :return: Complex reflection factor ``r(f)`` at the ``rfft`` frequency bins.
    :raises ValueError: On empty inputs, invalid geometry, or ``delay`` given
        without ``fs``.
    """
    kr = geometric_spreading_factor_angle(
        incidence_angle, source_height, mic_height
    )
    hi, hr, length = _transfer_functions(incident_ir, reflected_ir, n)
    r = (hr / hi) / kr
    if delay is not None:
        if fs is None:
            raise ValueError("'fs' is required to apply 'delay'.")
        if fs <= 0.0:
            raise ValueError(_FS_POSITIVE)
        # ``length`` is the exact time-domain length used for the FFTs, so
        # ``rfftfreq`` is correct for both even and odd inputs.
        freqs = np.fft.rfftfreq(length, d=1.0 / fs)
        r = r * np.exp(2j * np.pi * freqs * delay)
    return np.asarray(r, dtype=np.complex128)


def insitu_absorption_from_reflection(reflection: ArrayLike) -> Real:
    r"""Absorption coefficient from the reflection factor (ISO 13472-1, 4.1).

    :math:`\alpha = 1 - \lvert r \rvert^2`. With ``r`` already carrying the
    :math:`1 / K_r` factor
    (see :func:`insitu_reflection_factor`) this is the **reflection-factor route** to
    ``alpha`` and equals the direct energy route of
    :func:`insitu_absorption_coefficient`.

    :param reflection: Complex reflection factor ``r``.
    :return: Absorption coefficient ``alpha`` (real).
    """
    r = np.asarray(reflection, dtype=np.complex128)
    return np.asarray(1.0 - np.abs(r) ** 2, dtype=np.float64)


def power_reflection_coefficient(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    incidence_angle: float = 0.0,
    n: int | None = None,
) -> Real:
    r"""Sound-power reflection factor ``QW(f)`` (ISO 13472-1, 4.1 / Annex C).

    The **direct energy route**
    :math:`Q_W(f) = (1 / K_r^2) \lvert H_\mathrm{r}(f) / H_\mathrm{i}(f) \rvert^2`
    (``Kr,theta`` for oblique incidence). It equals
    :math:`\lvert r \rvert^2` from
    :func:`insitu_reflection_factor` but is formed from magnitudes only, so it is
    independent of any reflected-path time offset.

    :param incident_ir: Windowed incident impulse response ``hi(t)``, real.
    :param reflected_ir: Windowed reflected impulse response ``hr(t)``, real.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param incidence_angle: Incidence angle ``theta``, in radians.
    :param n: FFT length; defaults to the longer input.
    :return: Sound-power reflection factor ``QW(f)`` (real).
    """
    kr = geometric_spreading_factor_angle(
        incidence_angle, source_height, mic_height
    )
    hi, hr, _length = _transfer_functions(incident_ir, reflected_ir, n)
    ratio = np.abs(hr) / np.abs(hi)
    return np.asarray((ratio / kr) ** 2, dtype=np.float64)


def insitu_absorption_coefficient(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    incidence_angle: float = 0.0,
    n: int | None = None,
) -> Real:
    r"""Normal-incidence absorption coefficient ``alpha(f)`` (ISO 13472-1, 4.1).

    :math:`\alpha(f) = 1 - Q_W(f)
    = 1 - (1 / K_r^2) \lvert H_\mathrm{r}(f) / H_\mathrm{i}(f) \rvert^2` (the direct
    energy route via :func:`power_reflection_coefficient`; for oblique incidence
    ``Kr`` is replaced by ``Kr,theta``, Annex F). Non-specularly reflected energy
    is treated as absorbed, so ``alpha`` may be slightly overestimated
    (Clause 4.1).

    :param incident_ir: Windowed incident impulse response ``hi(t)``, real.
    :param reflected_ir: Windowed reflected impulse response ``hr(t)``, real.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param incidence_angle: Incidence angle ``theta``, in radians (0 = normal).
    :param n: FFT length; defaults to the longer input.
    :return: Absorption coefficient ``alpha(f)`` at the ``rfft`` frequency bins.
    """
    qw = power_reflection_coefficient(
        incident_ir,
        reflected_ir,
        source_height=source_height,
        mic_height=mic_height,
        incidence_angle=incidence_angle,
        n=n,
    )
    return np.asarray(1.0 - qw, dtype=np.float64)


def absorption_reference_corrected(
    road_reflection: ArrayLike,
    reference_reflection: ArrayLike,
) -> Real:
    r"""Reference-corrected road absorption (ISO 13472-1:2002, Annex B).

    Dividing the road and reference measured pressure reflection factors removes
    both the electro-acoustic chain error ``e(f)`` and, because the geometry is
    identical, the ``Kr`` factor:

    .. math::

       Q_{p,\text{road}}(f)
       = \frac{Q_{p,\text{road,meas}}(f)}{Q_{p,\text{ref,meas}}(f)}

       \alpha_{\text{road}}(f) = 1 - \left\lvert
       \frac{Q_{p,\text{road,meas}}(f)}{Q_{p,\text{ref,meas}}(f)}
       \right\rvert^{2}

    The reference surface is assumed totally reflecting
    (:math:`\lvert Q_{p,\text{ref}} \rvert = 1`, checked in an impedance tube
    to have absorption < 0.05, Annex B).

    :param road_reflection: Measured road pressure reflection factor
        ``Qp,road,meas`` (complex; the ``1 / Kr`` scaling need not be removed as
        it cancels).
    :param reference_reflection: Measured reference pressure reflection factor
        ``Qp,ref,meas`` (complex, same geometry and chain).
    :return: Reference-corrected road absorption coefficient ``alpha_road(f)``.
    """
    q_road = np.asarray(road_reflection, dtype=np.complex128)
    q_ref = np.asarray(reference_reflection, dtype=np.complex128)
    return np.asarray(1.0 - np.abs(q_road / q_ref) ** 2, dtype=np.float64)


# --------------------------------------------------------------------------- #
# One-third-octave presentation (ISO 13472-1 Clause 4.1 / ISO 13472-2 6.6)
# --------------------------------------------------------------------------- #
def one_third_octave_absorption(
    frequency: ArrayLike,
    absorption: ArrayLike,
    *,
    f_min: float = PART1_FREQUENCY_RANGE[0],
    f_max: float = PART1_FREQUENCY_RANGE[1],
    clip_negative: bool = True,
) -> tuple[Real, Real]:
    r"""Aggregate narrow-band absorption into one-third-octave bands.

    Both parts require a **linear average** of the narrow-band absorption over
    each one-third-octave band (ISO 13472-1 Clause 4.1; ISO 13472-2 Clause 6.6),
    keeping negative narrow-band values during the averaging. The band edges are
    the IEC base-two limits :math:`f_\mathrm{c} \cdot 2^{\pm 1/6}`. Negative one-third-octave
    results are set to zero when ``clip_negative`` is true (ISO 13472-2
    Clause 6.6 step 5); Part 1 does not mandate clipping, so pass
    ``clip_negative=False`` to reproduce its raw output.

    :param frequency: Narrow-band frequencies, in hertz.
    :param absorption: Narrow-band absorption values aligned with ``frequency``.
    :param f_min: Lowest band centre to report, in hertz (default 250 Hz).
    :param f_max: Highest band centre to report, in hertz (4000 Hz for Part 1;
        pass 1600 Hz for the Part-2 range).
    :param clip_negative: Set negative band results to zero (default ``True``).
    :return: Tuple ``(band_centres, band_absorption)``; a band with no
        narrow-band samples yields ``nan``.
    :raises ValueError: If ``frequency`` and ``absorption`` differ in length or
        are empty.
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    alpha = np.atleast_1d(np.asarray(absorption, dtype=np.float64))
    if freq.size == 0 or freq.shape != alpha.shape:
        raise ValueError(
            "'frequency' and 'absorption' must be non-empty and equal-length."
        )
    centres = np.array(
        [c for c in _ONE_THIRD_OCTAVE_CENTERS if f_min <= c <= f_max],
        dtype=np.float64,
    )
    lower = centres * 2.0 ** (-1.0 / 6.0)
    upper = centres * 2.0 ** (1.0 / 6.0)
    out = np.full(centres.shape, np.nan, dtype=np.float64)
    for i in range(centres.size):
        mask = (freq >= lower[i]) & (freq < upper[i])
        if np.any(mask):
            out[i] = float(np.mean(alpha[mask]))
    if clip_negative:
        out = np.where(np.isnan(out), out, np.maximum(out, 0.0))
    return centres, np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class InsituAbsorptionResult:
    """An in-situ one-third-octave absorption spectrum (ISO 13472-1).

    :ivar frequencies: One-third-octave band centre frequencies, in hertz.
    :ivar absorption: Sound-absorption coefficient ``alpha`` per band (a band
        with no contributing narrow-band samples is ``nan``).
    :ivar source_height: Source height the spectrum was reduced with, in
        metres, retained (with ``mic_height``) so :meth:`plot_geometry`
        can draw the set-up; ``None`` for hand-built results.
    :ivar mic_height: Microphone height, in metres, or ``None``.
    """

    frequencies: Real
    absorption: Real
    source_height: float | None = None
    mic_height: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the in-situ absorption spectrum ``alpha(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_insitu_absorption

        check_language(language)
        return plot_insitu_absorption(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the measurement set-up to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_insitu_result_geometry

        check_language(language)
        return plot_insitu_result_geometry(self, ax=ax, language=language, **kwargs)


def insitu_absorption_spectrum(
    incident_ir: ArrayLike,
    reflected_ir: ArrayLike,
    fs: float | None = None,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    incidence_angle: float = 0.0,
    n: int | None = None,
    f_min: float = PART1_FREQUENCY_RANGE[0],
    f_max: float = PART1_FREQUENCY_RANGE[1],
    clip_negative: bool = True,
) -> InsituAbsorptionResult:
    """In-situ one-third-octave absorption spectrum (ISO 13472-1, Clause 4.1).

    End-to-end convenience: the windowed incident and reflected impulse
    responses give the narrow-band absorption via
    :func:`insitu_absorption_coefficient`, which is then reduced to
    one-third-octave bands with :func:`one_third_octave_absorption` and wrapped
    in a plottable :class:`InsituAbsorptionResult`.

    :param incident_ir: Windowed incident (direct-path) impulse response ``hi``.
    :param reflected_ir: Windowed reflected-path impulse response ``hr``.
    :param fs: Sampling frequency, in hertz.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param incidence_angle: Incidence angle ``theta``, in radians (0 = normal).
    :param n: FFT length; defaults to the longer of the two impulse responses.
    :param f_min: Lowest band centre to report, in hertz (default 250 Hz).
    :param f_max: Highest band centre to report, in hertz (default 4000 Hz).
    :param clip_negative: Clip negative band results to zero (default ``True``).
    :return: An :class:`InsituAbsorptionResult` with ``.plot()``.
    :raises ValueError: On empty inputs, invalid geometry, or a missing or
        non-positive ``fs``.
    """
    if fs is None:
        raise ValueError(
            "insitu_absorption_spectrum() missing required argument: 'fs'."
        )
    if fs <= 0.0:
        raise ValueError(_FS_POSITIVE)
    hi_t = np.atleast_1d(np.asarray(incident_ir, dtype=np.float64))
    hr_t = np.atleast_1d(np.asarray(reflected_ir, dtype=np.float64))
    # Fix the FFT length explicitly so ``rfftfreq`` matches the transform used
    # inside ``insitu_absorption_coefficient`` for both even and odd inputs.
    length = n if n is not None else max(hi_t.size, hr_t.size)
    alpha = insitu_absorption_coefficient(
        hi_t,
        hr_t,
        source_height=source_height,
        mic_height=mic_height,
        incidence_angle=incidence_angle,
        n=length,
    )
    freq = np.fft.rfftfreq(length, d=1.0 / fs)
    centres, band = one_third_octave_absorption(
        freq, alpha, f_min=f_min, f_max=f_max, clip_negative=clip_negative
    )
    return InsituAbsorptionResult(
        frequencies=centres,
        absorption=band,
        source_height=float(source_height),
        mic_height=float(mic_height),
    )


# --------------------------------------------------------------------------- #
# Maximum sampled area (ISO 13472-1 Annex A / Annex F)
# --------------------------------------------------------------------------- #
def max_sampled_area_radius(
    window_width: float,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> float:
    r"""Radius of the maximum sampled area (ISO 13472-1:2002, Annex A).

    For normal incidence the maximum sampled area is a circle of radius (m):

    .. math::

       r = \frac{1}{d_\mathrm{s} + d_\mathrm{m} + c T_\mathrm{w}}
       \sqrt{(d_\mathrm{s} + d_\mathrm{m} + c T_\mathrm{w}/2)(d_\mathrm{s} + c T_\mathrm{w}/2)
       (2 d_\mathrm{m} + c T_\mathrm{w})(c T_\mathrm{w})}

    with ``Tw`` the width of the temporal window isolating the reflected wave.
    The Annex A worked example (:math:`d_\mathrm{s} = 1.25`, :math:`d_\mathrm{m} = 0.25`,
    :math:`c = 340` m/s, and the 5 ms flat window) gives
    :math:`r \approx 1.34` m.

    :param window_width: Temporal-window width ``Tw`` (reflected wave), seconds.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :return: Maximum-sampled-area radius ``r``, in metres.
    :raises ValueError: On non-positive geometry or window width.
    """
    _check_geometry(source_height, mic_height)
    if window_width <= 0.0:
        raise ValueError("'window_width' must be positive.")
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_POSITIVE)
    ds, dm, ctw = source_height, mic_height, speed_of_sound * window_width
    numerator = np.sqrt(
        (ds + dm + ctw / 2.0) * (ds + ctw / 2.0) * (2.0 * dm + ctw) * ctw
    )
    return float(numerator / (ds + dm + ctw))


def msa_major_axis(
    window_width: float,
    projected_distance: float,
    *,
    source_height: float = DEFAULT_SOURCE_HEIGHT,
    mic_height: float = DEFAULT_MIC_HEIGHT,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
) -> float:
    r"""Major axis of the oblique sampled-area ellipsoid (ISO 13472-1,
    Annex F).

    :math:`a = c\,T_\mathrm{w} + \sqrt{(d_\mathrm{s} + d_\mathrm{m})^2 + d_\mathrm{p}^2}` for the ellipsoid of
    revolution with the source and microphone at its foci; ``dp`` is the
    source-to-microphone distance projected on the reference plane.

    :param window_width: Temporal-window width ``Tw``, in seconds.
    :param projected_distance: Projected source-to-mic distance ``dp``, in metres.
    :param source_height: Source-to-plane distance ``ds``, in metres.
    :param mic_height: Microphone-to-plane distance ``dm``, in metres.
    :param speed_of_sound: Speed of sound ``c``, in metres per second.
    :return: Major axis ``a``, in metres.
    :raises ValueError: On non-positive window width, speed, or negative ``dp``.
    """
    if window_width <= 0.0:
        raise ValueError("'window_width' must be positive.")
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_POSITIVE)
    if projected_distance < 0.0:
        raise ValueError("'projected_distance' must be non-negative.")
    if source_height <= 0.0 or mic_height <= 0.0:
        raise ValueError("Heights must be positive.")
    return float(
        speed_of_sound * window_width
        + np.hypot(source_height + mic_height, projected_distance)
    )


# --------------------------------------------------------------------------- #
# ISO 13472-2 spot method: tube geometry / validity helpers.
# The core two-microphone DSP is NOT here: see
# phonometry.materials.absorbers.impedance_tube.two_microphone_impedance (ISO 10534-2, Clause 7).
# --------------------------------------------------------------------------- #
def spot_tube_upper_frequency(
    diameter: float, speed_of_sound: float = DEFAULT_SPEED_OF_SOUND
) -> float:
    r"""Upper usable frequency of the spot tube (ISO 13472-2:2010, 5.4.1).

    :math:`f_\mathrm{u} = 0.58 c_0 / d` (circular tube), the highest frequency at which
    only plane waves propagate. A 100 mm tube at :math:`c_0 = 340` m/s gives
    :math:`f_\mathrm{u} \approx 1972` Hz, comfortably above the 1800 Hz narrow-band
    top.

    :param diameter: Tube diameter ``d``, in metres.
    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :return: Upper usable frequency ``f_u``, in hertz.
    :raises ValueError: If ``d`` or ``c0`` is not positive.
    """
    if diameter <= 0.0:
        raise ValueError("'diameter' must be positive.")
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_POSITIVE)
    return float(_SPOT_FU_FACTOR * speed_of_sound / diameter)


def spot_microphone_spacing_bounds(
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    *,
    f_min: float = SPOT_NARROW_BAND_RANGE[0],
    f_max: float = SPOT_NARROW_BAND_RANGE[1],
) -> tuple[float, float]:
    r"""Microphone-spacing bounds ``(s_min, s_max)`` (ISO 13472-2:2010, 5.4.2).

    :math:`s_{\mathrm{max}} < 0.45 c_0 / f_{\mathrm{max}}` avoids spacing
    approaching half a wavelength at
    the top frequency, and :math:`s_{\mathrm{min}} > 0.05 c_0 / f_{\mathrm{min}}`
    keeps the spacing above
    5 % of a wavelength at the bottom frequency. For the narrow band
    220-1800 Hz (:math:`c_0 \approx 340` m/s) these give
    :math:`s_{\mathrm{max}} \approx 85` mm and
    :math:`s_{\mathrm{min}} \approx 77` mm, bracketing the nominal
    :math:`s = (81 \pm 4)` mm.

    :param speed_of_sound: Speed of sound ``c0``, in metres per second.
    :param f_min: Lowest frequency of interest ``f_min``, in hertz.
    :param f_max: Highest frequency of interest ``f_max``, in hertz.
    :return: Tuple ``(s_min, s_max)`` of the lower and upper spacing limits, m.
    :raises ValueError: On non-positive speed or frequency, or ``f_min >= f_max``.
    """
    if speed_of_sound <= 0.0:
        raise ValueError(_SPEED_POSITIVE)
    if f_min <= 0.0 or f_max <= 0.0:
        raise ValueError("Frequencies must be positive.")
    if f_min >= f_max:
        raise ValueError("'f_min' must be less than 'f_max'.")
    s_max = _SPOT_SMAX_FACTOR * speed_of_sound / f_max
    s_min = _SPOT_SMIN_FACTOR * speed_of_sound / f_min
    if s_min >= s_max:
        warnings.warn(
            f"No valid microphone spacing exists for {f_min:g}-{f_max:g} Hz: "
            f"s_min ({s_min:.4f} m) >= s_max ({s_max:.4f} m). Narrow the "
            "frequency range (the spot method's narrow band is 220-1800 Hz).",
            RoadAbsorptionWarning,
            stacklevel=2,
        )
    return s_min, s_max


def check_spot_frequency_range(frequency: ArrayLike) -> None:
    """Advise when a spot-method frequency is out of range (ISO 13472-2, Scope).

    The spot method is valid over the one-third-octave bands 250-1600 Hz
    (narrow-band 220-1800 Hz). Frequencies outside :data:`SPOT_FREQUENCY_RANGE`
    raise a :class:`RoadAbsorptionWarning`; results there are advisory.

    :param frequency: Frequencies to check, in hertz.
    :raises ValueError: If any frequency is negative.
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    if np.any(freq < 0.0):
        raise ValueError("'frequency' must be non-negative.")
    lo, hi = SPOT_FREQUENCY_RANGE
    if np.any(freq < lo) or np.any(freq > hi):
        warnings.warn(
            f"Frequencies outside the ISO 13472-2 spot range "
            f"[{lo:.0f}, {hi:.0f}] Hz; results there are advisory.",
            RoadAbsorptionWarning,
            stacklevel=2,
        )


def spot_internal_loss_correction(
    measured_absorption: ArrayLike,
    system_absorption: ArrayLike,
    *,
    clip_negative: bool = True,
) -> Real:
    r"""Internal-loss (system) correction (ISO 13472-2:2010, Annex A).

    The tube reads all thermal/viscous losses between the microphones and the
    surface as absorption. For reflective surfaces this is removed by
    subtracting the reading on a totally reflecting reference plate:

    .. math::

       \alpha(f) \approx \alpha_{\text{measured}}(f)
       - \alpha_{\text{system}}(f)

    Valid because both terms are small (measured < 0.15, internal < 0.03). This
    is the **subtractive** Part-2 correction and must not be confused with the
    Part-1 ratio correction (:func:`absorption_reference_corrected`). Negative
    one-third-octave results are set to zero when ``clip_negative`` is true
    (Clause 6.6 step 5).

    :param measured_absorption: Measured road absorption ``alpha_m(f)``.
    :param system_absorption: Reference-plate (system) absorption
        ``alpha_system(f)``, same bands.
    :param clip_negative: Set negative corrected values to zero (default ``True``).
    :return: Corrected absorption coefficient ``alpha(f)``.
    :raises ValueError: If the two inputs differ in shape.
    """
    alpha_m = np.atleast_1d(np.asarray(measured_absorption, dtype=np.float64))
    alpha_s = np.atleast_1d(np.asarray(system_absorption, dtype=np.float64))
    if alpha_m.shape != alpha_s.shape:
        raise ValueError(
            "'measured_absorption' and 'system_absorption' must share a shape."
        )
    corrected = alpha_m - alpha_s
    if clip_negative:
        corrected = np.maximum(corrected, 0.0)
    return np.asarray(corrected, dtype=np.float64)


def _check_geometry(source_height: float, mic_height: float) -> None:
    """Validate the source/microphone heights of the extended-surface geometry."""
    if source_height <= 0.0 or mic_height <= 0.0:
        raise ValueError("'source_height' and 'mic_height' must be positive.")
    if source_height <= mic_height:
        raise ValueError("'source_height' must exceed 'mic_height'.")
