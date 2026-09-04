#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Spherical-wave ground effect and advanced barrier diffraction.

This module extends the tabulated ground and barrier terms of ISO 9613-2 (see
:mod:`phonometry.environment.propagation.outdoor_propagation`) with the underlying wave
acoustics: the spherical-wave reflection coefficient of a finite-impedance
ground and the wave-theoretic diffraction of a screen, both in a homogeneous
(non-refracting, non-turbulent) atmosphere.

Ground effect (Weyl-Van der Pol)
--------------------------------
The sound field of a point source above a locally reacting ground is the sum of
a direct wave and a reflected wave weighted by the spherical-wave reflection
coefficient ``Q`` (Attenborough & Van Renterghem, *Predicting Outdoor Sound*
2e, 2021, Eq. (2.40a); Salomons, *Computational Atmospheric Acoustics*, 2001,
Eq. (3.2)):

.. math::

   p = \frac{e^{ikR_1}}{4 \pi R_1} + Q \, \frac{e^{ikR_2}}{4 \pi R_2}

with :math:`R_1` the source-receiver distance, :math:`R_2` the image-source
distance and (Attenborough Eq. (2.40c) / Salomons Eq. (D.58)):

.. math::

   Q = R_\mathrm{p} + (1 - R_\mathrm{p}) F(w)

   R_\mathrm{p} = \frac{Z \cos\theta - 1}{Z \cos\theta + 1}
   \tag{Salomons Eq. D.59}

   F(w) = 1 + i \sqrt{\pi} \, w \exp(-w^2) \operatorname{erfc}(-i w)
   \tag{Salomons Eq. D.60}

   w = \sqrt{i k R_2 / 2} \, \left( \cos\theta + \frac{1}{Z} \right)
   \tag{Salomons Eq. D.57}

Here ``Z`` is the normalized (by ``rho c``) surface impedance of the ground,
``theta`` is the angle of incidence from the ground normal
(:math:`\cos\theta = (h_\mathrm{s} + h_\mathrm{r})/R_2`) and :math:`F(w)` is the boundary-loss
factor written through the scaled complementary error function
:math:`\exp(-w^2) \operatorname{erfc}(-i w)`, i.e.
the Faddeeva function :func:`scipy.special.wofz`. The relative sound level (the
"excess attenuation", dB re free field) is:

.. math::

   \Delta L = 20 \log_{10} \left| 1 + Q \, \frac{R_1}{R_2} \,
   e^{i k (R_2 - R_1)} \right| \tag{Salomons Eq. 3.4}

Limits reproduced by the implementation: an acoustically hard ground
(:math:`|Z| \to \infty`) gives :math:`R_\mathrm{p} \to 1`, so
:math:`(1 - R_\mathrm{p}) \to 0` and :math:`Q \to 1`
regardless of the boundary loss (the ground wave vanishes), and
:math:`\Delta L` reaches
``+6 dB`` in phase (Salomons Sec. 3.4); at grazing incidence
(:math:`h_\mathrm{s}, h_\mathrm{r} \to 0`, :math:`\cos\theta \to 0`) :math:`R_\mathrm{p} \to -1`; and as
the range grows (:math:`R_2 \to \infty`) :math:`|w| \to \infty` and
:math:`F \to 0`. The ground impedance is taken in
the :math:`e^{-i \omega t}` time convention of Salomons, in which a passive
ground has :math:`\operatorname{Im}(Z) > 0`; it may be supplied directly or
derived from the porous models of :mod:`phonometry.materials`
(:func:`~phonometry.materials.delany_bazley` / :func:`~phonometry.materials.miki`),
which model a semi-infinite porous ground whose surface impedance equals the
characteristic impedance of the medium. The materials domain works in the
opposite :math:`e^{+j \omega t}` convention (:math:`\operatorname{Im}(Z) < 0`
for a passive medium), so
any impedance obtained from a porous model is conjugated internally before it
enters the formulas above.

Barrier diffraction
-------------------
Three levels of screening beyond the ISO 9613-2 ``Dz`` term are provided:

* the Kurze-Anderson closed form in the Fresnel number ``N`` (Bies, Hansen &
  Howard, *Engineering Noise Control* 5e, 2017, Eq. (5.138); Kurze & Anderson,
  1971),
  :math:`\Delta = 5 + 20 \log_{10}\!\left[ \sqrt{2 \pi N} / \tanh\sqrt{2 \pi N}
  \right]`,
  which tends to ``5 dB`` at :math:`N \to 0` and stays within about 1.5 dB of
  Maekawa's point-source curve for all ``N`` (a very good fit for
  :math:`N > 0.5`);

* the wave-theoretic insertion loss of a rigid thin screen (half-plane), the
  flat-wedge limit of the MacDonald / Hadden & Pierce solution
  (Attenborough Eqs. (9.19)-(9.20)), obtained from the auxiliary Fresnel
  functions and correctly giving ``6 dB`` at the shadow boundary (the field is
  halved);

* the coherent barrier-on-ground model that combines the four source-image /
  receiver-image diffracted paths with the spherical-wave reflection
  coefficient ``Q`` above (Attenborough Ch. 9; Bies Sec. 5.3.5), which shows the
  ground-barrier interference structure a purely energetic sum cannot.

Thick barriers (or two parallel thin screens) are handled by the double-edge
Fresnel number :math:`N = (2/\lambda)(A + B + e - d)` (Bies Eq. (5.157)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import fresnel, wofz

from ..._internal.validation import (
    check_engine,
    require_choice,
    require_positive,
    require_positive_array,
    require_ranks,
    require_same_length,
)
from ...materials.absorbers.porous import PUBLISHED_AIR, delany_bazley, miki

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata
    from ...fluids import Fluid
    from ...materials.absorbers.porous import PorousMediumResult

#: Nominal octave-band midband frequencies, in hertz (as ISO 9613-2 clause 1).
DEFAULT_FREQUENCIES: tuple[float, ...] = (
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    8000.0,
)
#: The diffraction models :func:`barrier_insertion_loss` implements, and the
#: only tags a :class:`BarrierInsertionLoss` may carry: the fiche and the plot
#: both name the model from this tag, so one they do not know would be printed
#: as the basis of a prediction the library never made.
_BARRIER_METHODS: tuple[str, ...] = ("kurze_anderson", "exact")
#: Tolerance on |x|, x = sqrt(2*pi*N), below which x/tanh(x) is replaced by its
#: x -> 0 limit of 1+0j (guards the 0/0 indeterminate form at N = 0).
_X_OVER_TANH_EPS = 1e-9
#: Illuminated-zone limit of Maekawa's curve: the Fresnel number below which
#: barrier diffraction is taken as negligible (0 dB).
_MAEKAWA_ILLUMINATED_LIMIT = -0.2

Complex = NDArray[np.complex128]
Real = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Ground impedance helpers
# --------------------------------------------------------------------------- #
def _normalized_ground_impedance(
    frequency: Real,
    impedance: ArrayLike | PorousMediumResult | None,
    flow_resistivity: float | None,
    model: str,
    fluid: Fluid,
) -> Complex:
    r"""Resolve the normalized (by ``rho c``) surface impedance of the ground.

    A semi-infinite porous ground is modelled as a half-space whose surface
    impedance equals the characteristic impedance of the medium, so the
    normalized surface impedance is exactly the normalized characteristic
    impedance of :func:`~phonometry.materials.delany_bazley` /
    :func:`~phonometry.materials.miki`.

    The returned impedance is in the :math:`e^{-i \omega t}` time convention
    of Salomons (a passive ground has :math:`\operatorname{Im}(Z) > 0`). The
    materials domain works in the opposite :math:`e^{+j \omega t}` convention
    (:math:`\operatorname{Im}(Z) < 0`), so anything
    obtained from a porous model (the ``flow_resistivity`` path or a
    ``PorousMediumResult``) is conjugated here; a plain ``impedance``
    scalar/array is taken as already :math:`e^{-i \omega t}` and passed
    through.
    """
    from ...materials.absorbers.porous import PorousMediumResult

    if (impedance is None) == (flow_resistivity is None):
        msg = "provide exactly one of 'impedance' or 'flow_resistivity'."
        raise ValueError(msg)
    if flow_resistivity is not None:
        sigma = require_positive(flow_resistivity, "flow_resistivity")
        if model == "delany_bazley":
            medium = delany_bazley(frequency, sigma, fluid=fluid)
        elif model == "miki":
            medium = miki(frequency, sigma, fluid=fluid)
        else:
            msg = f"unknown ground model {model!r}; options: 'delany_bazley', 'miki'."
            raise ValueError(msg)
        # Materials e^{+j omega t} -> Salomons e^{-i omega t}: conjugate.
        return np.asarray(np.conj(medium.normalized_impedance), dtype=np.complex128)
    if isinstance(impedance, PorousMediumResult):
        # Materials e^{+j omega t} -> Salomons e^{-i omega t}: conjugate.
        impedance = np.conj(impedance.normalized_impedance)
    return _resolve_impedance_array(impedance, frequency)


def _resolve_impedance_array(impedance: ArrayLike | None, frequency: Real) -> Complex:
    r"""Broadcast a scalar/array normalized impedance to the frequency shape.

    The impedance is in the :math:`e^{-i \omega t}` convention (a passive
    ground has :math:`\operatorname{Im}(Z) > 0`); a zero or non-finite
    impedance is rejected because :math:`1/Z`
    enters the numerical distance (an infinite ``Z`` would give ``inf/inf`` NaN
    rather than the intended hard-ground limit, which is a large finite ``Z``).
    """
    z = np.atleast_1d(np.asarray(impedance, dtype=np.complex128))
    if z.size == 1:
        z = np.full(frequency.shape, z[0], dtype=np.complex128)
    if z.shape != frequency.shape:
        msg = (
            "'impedance' must be a scalar or match the frequency vector "
            f"(got {z.shape}, expected {frequency.shape})."
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(z)):
        msg = "'impedance' must be finite."
        raise ValueError(msg)
    if not np.all(np.abs(z) > 0.0):
        msg = "'impedance' must be non-zero."
        raise ValueError(msg)
    return z


# --------------------------------------------------------------------------- #
# B1. Spherical-wave ground reflection (Weyl-Van der Pol)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SphericalGroundResult:
    r"""Spherical-wave ground-effect result (Weyl-Van der Pol).

    Every array is aligned with :attr:`frequencies`.

    :ivar frequencies: Frequencies, in hertz.
    :ivar excess_attenuation: Relative sound level :math:`\Delta L` (dB re
        free field, Salomons Eq. (3.4)); positive is enhancement (up to +6 dB
        over hard ground), negative is the ground-effect dip.
    :ivar reflection_coefficient: Spherical-wave reflection coefficient ``Q``
        (complex, Attenborough Eq. (2.40c)).
    :ivar plane_reflection_coefficient: Plane-wave reflection coefficient ``Rp``
        (complex, Salomons Eq. (D.59)).
    :ivar boundary_loss: Boundary-loss factor ``F(w)`` (complex, Eq. (D.60)).
    :ivar normalized_impedance: Normalized surface impedance ``Z`` used.
    :ivar r_direct: Direct source-receiver distance ``R1``, in metres.
    :ivar r_reflected: Image-source distance ``R2``, in metres.
    """

    frequencies: Real
    excess_attenuation: Real
    reflection_coefficient: Complex
    plane_reflection_coefficient: Complex
    boundary_loss: Complex
    normalized_impedance: Complex
    r_direct: float
    r_reflected: float

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the excess attenuation :math:`\Delta L` versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.environment import plot_spherical_ground

        return plot_spherical_ground(
            self, ax=ax, language=check_language(language), **kwargs
        )


def spherical_reflection_coefficient(
    frequencies: ArrayLike,
    normalized_impedance: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    speed_of_sound: float = PUBLISHED_AIR.speed_of_sound,
) -> Complex:
    r"""Spherical-wave reflection coefficient ``Q`` (Weyl-Van der Pol).

    Implements :math:`Q = R_\mathrm{p} + (1 - R_\mathrm{p}) F(w)` (Attenborough Eq. (2.40c);
    Salomons Eq. (D.58)) with the plane-wave coefficient ``Rp`` (Eq. (D.59)),
    the boundary-loss factor
    :math:`F(w) = 1 + i \sqrt{\pi} \, w \exp(-w^2) \operatorname{erfc}(-i w)`
    (Eq. (D.60), evaluated through :func:`scipy.special.wofz`) and the
    numerical distance
    :math:`w = \sqrt{i k R_2 / 2} \, (\cos\theta + 1/Z)` (Eq. (D.57)).

    :param frequencies: Frequencies, in hertz.
    :param normalized_impedance: Ground surface impedance normalized by
        ``rho c`` (complex, per frequency or scalar), in the
        :math:`e^{-i \omega t}` time convention (a passive ground has
        :math:`\operatorname{Im}(Z) > 0`).
    :param source_height: Source height ``hs`` above the ground, in metres.
    :param receiver_height: Receiver height ``hr`` above the ground, in metres.
    :param distance: Horizontal source-receiver distance, in metres.
    :param speed_of_sound: Speed of sound ``c``, in m/s.
    :return: Complex ``Q`` per frequency.
    :raises ValueError: If a height is negative, the distance is not positive,
        the impedance is zero, or its shape does not match the frequencies.
    """
    if source_height < 0.0 or receiver_height < 0.0:
        msg = "Source and receiver heights must be non-negative."
        raise ValueError(msg)
    if distance <= 0.0:
        msg = "'distance' must be positive."
        raise ValueError(msg)
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    f = require_positive_array(frequencies, "frequencies")
    z = _resolve_impedance_array(normalized_impedance, f)
    k = 2.0 * np.pi * f / speed_of_sound
    r2 = float(np.hypot(distance, source_height + receiver_height))
    cos_theta = (source_height + receiver_height) / r2
    rp = (z * cos_theta - 1.0) / (z * cos_theta + 1.0)
    w = np.sqrt(0.5j * k * r2) * (cos_theta + 1.0 / z)
    f_w = 1.0 + 1j * np.sqrt(np.pi) * w * wofz(w)
    return np.asarray(rp + (1.0 - rp) * f_w, dtype=np.complex128)


@overload
def ground_effect(
    frequencies: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    *,
    impedance: ArrayLike | PorousMediumResult,
    model: Literal["delany_bazley", "miki"] = ...,
    fluid: Fluid = ...,
) -> SphericalGroundResult: ...


@overload
def ground_effect(
    frequencies: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    *,
    flow_resistivity: float,
    model: Literal["delany_bazley", "miki"] = ...,
    fluid: Fluid = ...,
) -> SphericalGroundResult: ...


def ground_effect(
    frequencies: ArrayLike,
    source_height: float,
    receiver_height: float,
    distance: float,
    *,
    impedance: ArrayLike | PorousMediumResult | None = None,
    flow_resistivity: float | None = None,
    model: Literal["delany_bazley", "miki"] = "delany_bazley",
    fluid: Fluid = PUBLISHED_AIR,
) -> SphericalGroundResult:
    r"""Spherical-wave ground effect above a finite-impedance ground.

    Assembles the two-ray field
    :math:`p = e^{ikR_1}/(4 \pi R_1) + Q \, e^{ikR_2}/(4 \pi R_2)`
    with the spherical-wave reflection coefficient ``Q`` of
    :func:`spherical_reflection_coefficient` and reports the relative sound
    level
    :math:`\Delta L = 20 \log_{10}\left| 1 + Q (R_1/R_2) e^{i k (R_2 - R_1)}
    \right|` (Salomons Eq. (3.4)),
    i.e. the level re the free field.

    The ground surface impedance is either supplied through ``impedance`` (a
    normalized complex array/scalar, or a
    :class:`~phonometry.materials.PorousMediumResult`) or derived from an
    effective ``flow_resistivity`` (in Pa s/m2) via the ``model`` porous model
    of the materials domain. Exactly one of the two must be given.

    :param frequencies: Frequencies, in hertz.
    :param source_height: Source height ``hs``, in metres.
    :param receiver_height: Receiver height ``hr``, in metres.
    :param distance: Horizontal source-receiver distance, in metres.
    :param impedance: Normalized ground impedance (:math:`e^{-i \omega t}`
        convention, :math:`\operatorname{Im}(Z) > 0` for a passive ground), or
        a ``PorousMediumResult`` (which is conjugated internally from the
        materials' :math:`e^{+j \omega t}` convention).
    :param flow_resistivity: Effective flow resistivity ``sigma`` (Pa s/m2);
        grassland is about ``2e5`` (Salomons Sec. 3.1). The porous model raises
        a :class:`~phonometry.materials.PorousAbsorberWarning` when the lowest
        bands fall below its published fit range
        :math:`0.01 < \rho f / \sigma < 1`
        (it still extrapolates a value there).
    :param model: Porous model for ``flow_resistivity`` (``"delany_bazley"`` or
        ``"miki"``).
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid`
        (Default: :data:`PUBLISHED_AIR`, the air this model was published
        with). Pass a computed one, such as ``fluids.air(temperature_c=30.0,
        relative_humidity_percent=70.0)``, to work in the air of the room.
    :return: A :class:`SphericalGroundResult`.
    :raises ValueError: If neither or both of ``impedance``/``flow_resistivity``
        are given, a height is negative, or the distance is not positive.
    """
    if source_height < 0.0 or receiver_height < 0.0:
        msg = "Source and receiver heights must be non-negative."
        raise ValueError(msg)
    if distance <= 0.0:
        msg = "'distance' must be positive."
        raise ValueError(msg)
    speed_of_sound = fluid.speed_of_sound
    f = require_positive_array(frequencies, "frequencies")
    z = _normalized_ground_impedance(f, impedance, flow_resistivity, model, fluid)
    k = 2.0 * np.pi * f / speed_of_sound
    r1 = float(np.hypot(distance, source_height - receiver_height))
    r2 = float(np.hypot(distance, source_height + receiver_height))
    cos_theta = (source_height + receiver_height) / r2
    rp = (z * cos_theta - 1.0) / (z * cos_theta + 1.0)
    w = np.sqrt(0.5j * k * r2) * (cos_theta + 1.0 / z)
    f_w = 1.0 + 1j * np.sqrt(np.pi) * w * wofz(w)
    q = rp + (1.0 - rp) * f_w
    ratio = 1.0 + q * (r1 / r2) * np.exp(1j * k * (r2 - r1))
    d_l = 20.0 * np.log10(np.abs(ratio))
    return SphericalGroundResult(
        frequencies=f,
        excess_attenuation=np.asarray(d_l, dtype=np.float64),
        reflection_coefficient=np.asarray(q, dtype=np.complex128),
        plane_reflection_coefficient=np.asarray(rp, dtype=np.complex128),
        boundary_loss=np.asarray(f_w, dtype=np.complex128),
        normalized_impedance=z,
        r_direct=r1,
        r_reflected=r2,
    )


# --------------------------------------------------------------------------- #
# B4. Barrier diffraction
# --------------------------------------------------------------------------- #
def fresnel_number(
    source_to_edge: float,
    edge_to_receiver: float,
    direct_distance: float,
    frequencies: ArrayLike,
    speed_of_sound: float = PUBLISHED_AIR.speed_of_sound,
) -> Real:
    r"""Fresnel number :math:`N = (2/\lambda)(A + B - d)` (Bies Eq. (5.134)).

    ``A`` and ``B`` are the two segments of the shortest source-edge-receiver
    path and ``d`` is the straight source-receiver distance. ``N`` is positive
    when the receiver is in the shadow zone (:math:`A + B > d`) and negative
    in the bright zone.

    :param source_to_edge: Path segment ``A`` from source to edge, in metres.
    :param edge_to_receiver: Path segment ``B`` from edge to receiver, in metres.
    :param direct_distance: Straight source-receiver distance ``d``, in metres.
    :param frequencies: Frequencies, in hertz.
    :param speed_of_sound: Speed of sound ``c``, in m/s.
    :return: Fresnel number ``N`` per frequency.
    :raises ValueError: If a distance is not positive.
    """
    for name, value in (
        ("source_to_edge", source_to_edge),
        ("edge_to_receiver", edge_to_receiver),
        ("direct_distance", direct_distance),
    ):
        if value <= 0.0:
            msg = f"'{name}' must be positive."
            raise ValueError(msg)
    speed_of_sound = require_positive(speed_of_sound, "speed_of_sound")
    f = require_positive_array(frequencies, "frequencies")
    lam = speed_of_sound / f
    delta = source_to_edge + edge_to_receiver - direct_distance
    return np.asarray(2.0 * delta / lam, dtype=np.float64)


def kurze_anderson_attenuation(fresnel_number: ArrayLike) -> Real:
    r"""Kurze-Anderson barrier attenuation (Bies Eq. (5.138); Kurze & Anderson, 1971).

    .. math::

       \Delta = 5 + 20 \log_{10}\!\left[ \frac{\sqrt{2 \pi N}}
       {\tanh\sqrt{2 \pi N}} \right] \qquad \text{dB}

    For :math:`N \to 0` the ratio tends to 1 and :math:`\Delta \to 5` dB; for
    :math:`N < 0`
    (bright zone) the square root is imaginary and ``tanh`` becomes ``tan``, so
    the expression continues smoothly until, below :math:`N = -0.2` (the
    illuminated-zone limit of Maekawa's curve), the diffraction is taken as
    negligible (0 dB) rather than let the closed form oscillate through the
    tangent poles. It stays within about 1.5 dB of Maekawa's point-source curve
    for all ``N`` (a very good fit for :math:`N > 0.5`). The result is clamped
    at 0 dB (a barrier never amplifies).

    :param fresnel_number: Fresnel number ``N`` (scalar or array).
    :return: Attenuation ``Delta``, in decibels (>= 0), matching the input shape.
    :raises ValueError: for a non-numeric or non-finite ``fresnel_number``.
    """
    try:
        n = np.asarray(fresnel_number, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = "'fresnel_number' must be numeric."
        raise ValueError(msg) from exc
    # A NaN would pass through the closed form and come back as a NaN
    # "attenuation", silently breaking the documented >= 0 dB contract.
    if not np.all(np.isfinite(n)):
        msg = "'fresnel_number' must contain only finite values."
        raise ValueError(msg)
    # The formula is only meaningful for N > -0.2 (the illuminated-zone limit of
    # Maekawa's curve); beyond it the diffraction is negligible (0 dB), so the
    # intermediate argument is clipped at -0.2 to keep it clear of the tangent
    # poles of the imaginary-argument branch (this cannot change the result,
    # which is forced to 0 there below) instead of oscillating through them.
    n_clipped = np.maximum(n, _MAEKAWA_ILLUMINATED_LIMIT)
    x = np.sqrt(2.0 * np.pi * n_clipped.astype(np.complex128))
    # x / tanh(x) is real for both real and imaginary x; take the real part.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(x) < _X_OVER_TANH_EPS, 1.0 + 0.0j, x / np.tanh(x))
    delta = 5.0 + 20.0 * np.log10(np.abs(ratio))
    delta = np.where(n <= _MAEKAWA_ILLUMINATED_LIMIT, 0.0, delta)
    return np.asarray(np.maximum(delta, 0.0), dtype=np.float64)


def _auxiliary_fresnel(x: Real) -> tuple[Real, Real]:
    r"""Auxiliary Fresnel functions ``f(x)``, ``g(x)`` (Attenborough Eq. (9.20)).

    :math:`f = (1/2 - S) \cos(\pi x^2/2) - (1/2 - C) \sin(\pi x^2/2)` and
    :math:`g = (1/2 - C) \cos(\pi x^2/2) + (1/2 - S) \sin(\pi x^2/2)` with
    ``C``, ``S`` the Fresnel integrals of Eq. (9.11) (the :math:`\pi/2`
    convention of :func:`scipy.special.fresnel`).
    """
    s, c = fresnel(x)
    phase = 0.5 * np.pi * x**2
    cos_p, sin_p = np.cos(phase), np.sin(phase)
    f = (0.5 - s) * cos_p - (0.5 - c) * sin_p
    g = (0.5 - c) * cos_p + (0.5 - s) * sin_p
    return f, g


def _diffraction_integral(x: Real) -> Complex:
    r"""Diffraction integral of the flat-wedge screen solution (Eq. (9.19e)).

    :math:`A_\mathrm{D}(X) = \operatorname{sgn}(X) \, (f(|X|) - i \, g(|X|))`. At
    :math:`X = 0` (a source-edge-receiver exactly on the sight line) the sign
    is taken as ``+1``, the shadow-side (:math:`X \to 0^{+}`) limit, so the
    diffracted term stays continuous through the shadow boundary instead of
    vanishing.
    """
    f, g = _auxiliary_fresnel(np.abs(x))
    sign = np.where(x >= 0.0, 1.0, -1.0)
    return np.asarray(sign * (f - 1j * g), dtype=np.complex128)


def _screen_field(
    source: tuple[float, float],
    near_edge: tuple[float, float],
    far_edge: tuple[float, float],
    receiver: tuple[float, float],
    k: Real,
) -> Complex:
    r"""Complex diffracted field of a rigid thin or thick barrier (flat-wedge limit).

    Uses the compact Fresnel-integral form of the MacDonald / Hadden & Pierce
    solution (Attenborough Eq. (9.19)):

    .. math::

       p_\mathrm{d} = \frac{e^{ikR'}}{4 \pi R'} \, \frac{1+i}{2}
       \left[ A_\mathrm{D}(X_-) + A_\mathrm{D}(X_+) \right]

    with :math:`R' = A + e + B` the shortest diffracted path over the barrier
    top (``A`` source to near edge, ``e`` the top width between the two edges,
    ``B`` far edge to receiver; a thin screen has ``near_edge == far_edge``
    and :math:`e = 0`). :math:`X_-` uses the direct source-receiver detour and
    :math:`X_+` the detour of the source image across the near-edge screen
    plane; the geometry-invariant argument is
    :math:`X = \operatorname{sgn}(\delta)
    \sqrt{\delta (R' + R) / (\lambda R')}` with :math:`\delta = R' - R`.
    Lengthening :math:`R'` by the top width ``e`` reproduces the double-edge
    Fresnel number of Bies Eq. (5.157), so a thick barrier attenuates
    monotonically more than the thin screen of the same height. Exact for a
    rigid half-plane near grazing incidence (the barrier regime).

    Coordinates are 2-D ``(horizontal, height)`` with the screen the vertical
    half-plane below the near edge.
    """
    sx, sy = source
    nx, ny = near_edge
    fx, fy = far_edge
    rx, ry = receiver
    r0 = float(np.hypot(sx - nx, sy - ny))
    e_top = float(np.hypot(fx - nx, fy - ny))
    rr = float(np.hypot(rx - fx, ry - fy))
    r_path = r0 + e_top + rr
    r_direct = float(np.hypot(sx - rx, sy - ry))
    # Source image across the vertical screen plane x = nx.
    r_reflected = float(np.hypot((2.0 * nx - sx) - rx, sy - ry))
    lam = 2.0 * np.pi / k

    def arg(r_geom: float) -> Real:
        delta = r_path - r_geom
        val = np.sqrt(np.maximum((delta * (r_path + r_geom)) / (lam * r_path), 0.0))
        return np.asarray(np.sign(delta) * val, dtype=np.float64)

    x_minus = arg(r_direct)
    x_plus = arg(r_reflected)
    p0 = np.exp(1j * k * r_path) / (4.0 * np.pi * r_path)
    return np.asarray(
        p0
        * (0.5 + 0.5j)
        * (_diffraction_integral(x_minus) + _diffraction_integral(x_plus)),
        dtype=np.complex128,
    )


@dataclass(frozen=True)
class BarrierInsertionLoss:
    r"""Per-frequency barrier insertion loss (IL vs frequency).

    :ivar frequencies: Frequencies, in hertz.
    :ivar insertion_loss: Insertion loss
        :math:`\mathrm{IL} = 20 \log_{10} \lvert p_{\text{without}} /
        p_{\text{with}} \rvert`, in decibels, per frequency.
    :ivar fresnel_number: Fresnel number ``N`` per frequency (single-edge
        geometry; the double-edge ``N`` for a thick barrier).
    :ivar method: Diffraction model used, and the tag the fiche and the plot
        name it by: ``"kurze_anderson"`` or ``"exact"``, the two models the
        library implements. Any other tag is refused at construction.
    :ivar ground: Whether the coherent four-path ground model was applied.
    :ivar source_height: Source height the loss was computed for, in metres,
        retained (with the other five geometry fields) so
        :meth:`plot_geometry` can draw the section; appended after the
        original fields and ``None`` for hand-built results.
    :ivar barrier_distance: Source-to-barrier horizontal distance, in metres.
    :ivar barrier_height: Barrier height, in metres.
    :ivar receiver_distance: Source-to-receiver horizontal distance, in
        metres.
    :ivar receiver_height: Receiver height, in metres.
    :ivar thickness: Thick-barrier top width, in metres, or ``None``.
    """

    frequencies: Real
    insertion_loss: Real
    fresnel_number: Real
    method: str
    ground: bool
    source_height: float | None = None
    barrier_distance: float | None = None
    barrier_height: float | None = None
    receiver_distance: float | None = None
    receiver_height: float | None = None
    thickness: float | None = None

    def __post_init__(self) -> None:
        """Pin the model tag, and the loss finite on its frequency axis.

        :attr:`method` names the diffraction model the sheet cites as its
        basis, and both readers resolve it through a dict of model phrases
        with the tag itself as the silent default. Unpinned, any string
        passed construction and came out the other side as an accredited
        claim: ``method="ISO 9613-2 Dz"`` printed "Barrier insertion loss IL
        predicted with ISO 9613-2 Dz, a wave-acoustics complement to the
        tabulated ISO 9613-2:1996 screening term", naming as the applied
        model the very formula the fiche exists to distinguish itself from.
        The tag is pinned here to the two models
        :func:`barrier_insertion_loss` implements, so the dict lookups it
        feeds cannot miss.

        The fiche prints one table row per entry of :attr:`frequencies`,
        reading :attr:`insertion_loss` (and, verbose, the Fresnel number) at
        the same indices, and the plot draws one against the other. Neither
        reader names a field when the lengths disagree: a loss one long
        crashes as matplotlib's bare "x and y must have same first
        dimension", one short as an ``IndexError`` out of the table. This
        refusal is raised at construction and says which field is short.

        The loss must also cover at least one frequency. Three length-0 axes
        agree with each other and pass every count above, and the sheet then
        asks matplotlib to log-scale an axis with nothing on it, which comes
        back as "Data cannot be log-scaled because all values are <= 0" --
        a complaint about values, from a result that has none.

        Every value must also be finite. The one producer,
        :func:`barrier_insertion_loss`, computes the loss from a geometry
        and frequency axis it has already pinned finite, so a NaN here is
        never the library's own output; letting one through renders a
        ``nan`` table cell under a BOXED ``IL = nan dB`` headline
        (``np.mean`` computes through it), and with a declared requirement
        the verdict dies on ``display_round(nan)`` with an anonymous
        "cannot convert float NaN to integer".

        :raises ValueError: if the method is not a model the library
            implements, the loss covers no frequency, or the per-frequency
            fields disagree in length or carry a non-finite value.
        """
        require_choice(self.method, "method", _BARRIER_METHODS)
        require_ranks(self, frequencies=1, insertion_loss=1, fresnel_number=1)
        require_same_length(
            self, "frequencies", "insertion_loss", "fresnel_number", axis="frequency"
        )
        if np.asarray(self.frequencies).size == 0:
            msg = (
                "BarrierInsertionLoss: 'frequencies' must carry at least one "
                "frequency; the insertion loss is empty."
            )
            raise ValueError(msg)
        for name in ("frequencies", "insertion_loss", "fresnel_number"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.all(np.isfinite(value)):
                msg = f"BarrierInsertionLoss: '{name}' must contain only finite values."
                raise ValueError(msg)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the insertion loss versus frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.environment import plot_barrier_insertion_loss

        return plot_barrier_insertion_loss(
            self, ax=ax, language=check_language(language), **kwargs
        )

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Draw the source-barrier-receiver section to scale.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its geometry.
        """
        from ..._i18n import check_language
        from ..._plot.geometry import plot_barrier_result_geometry

        check_language(language)
        return plot_barrier_result_geometry(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a one-page barrier insertion-loss prediction fiche to a PDF.

        Writes a prediction sheet (clearly labelled a prediction, not a
        measurement) laid out like an outdoor-noise barrier calculation: the
        standard-basis line naming the diffraction model used (the
        wave-theoretic rigid-screen model for ``method="exact"`` or the
        Kurze-Anderson closed form for ``method="kurze_anderson"``, a
        wave-acoustics complement to the ISO 9613-2 screening term), an optional
        metadata header (source/situation, client, receiver position, date), a
        per-band table of the insertion loss ``IL`` (and, in verbose mode, the
        Fresnel number ``N``), the insertion-loss spectrum plot, a boxed mean
        insertion loss over the octave bands, an optional PASS/FAIL verdict
        against a declared minimum required insertion loss (a higher insertion
        loss is better) and a footer identity/disclaimer block.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``specimen`` the source/situation, ``client``,
            ``test_room`` the receiver position) and the footer identity. A
            supplied ``requirement`` is read as the minimum required mean
            insertion loss in dB.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When True, the per-band table adds the Fresnel number
            ``N`` column.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab or matplotlib is not installed
            (``pip install "phonometry[report,plot]"``).
        """
        from ..._i18n import check_language

        check_language(language)
        check_engine(engine)
        from ..._report.iso9613 import render_barrier_insertion_loss_report

        return render_barrier_insertion_loss_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _validate_barrier_geometry(
    source_height: float,
    barrier_distance: float,
    barrier_height: float,
    receiver_distance: float,
    receiver_height: float,
    thickness: float | None,
) -> float | None:
    """Validate the barrier geometry and return the checked top width.

    :return: The validated ``thickness`` (positive and finite) or ``None`` for a
        thin screen.
    :raises ValueError: on a non-positive or non-finite distance, a non-finite
        height, a mis-ordered geometry, a barrier below the line of sight, or a
        thick barrier whose far edge reaches the receiver.
    """
    # require_positive rejects NaN/inf as well as non-positive distances and
    # names the one that was wrong; a NaN height would sail through the
    # ordering comparisons below (every comparison against NaN is False) and
    # come back from the diffraction maths as an all-NaN insertion loss.
    require_positive(barrier_distance, "barrier_distance")
    require_positive(receiver_distance, "receiver_distance")
    for name, height in (
        ("source_height", source_height),
        ("barrier_height", barrier_height),
        ("receiver_height", receiver_height),
    ):
        if not np.isfinite(height):
            msg = f"'{name}' must be finite."
            raise ValueError(msg)
    if receiver_distance <= barrier_distance:
        msg = "'receiver_distance' must exceed 'barrier_distance'."
        raise ValueError(msg)
    if barrier_height <= max(source_height, receiver_height):
        msg = (
            "'barrier_height' must exceed the source and receiver heights "
            "(the receiver must be in the geometric shadow)."
        )
        raise ValueError(msg)
    if thickness is None:
        return None
    # require_positive rejects NaN/inf as well as non-positive widths.
    thickness = require_positive(thickness, "thickness")
    if barrier_distance + thickness >= receiver_distance:
        msg = (
            "'receiver_distance' must exceed 'barrier_distance + thickness' "
            "(the receiver must lie beyond the barrier's far edge)."
        )
        raise ValueError(msg)
    return thickness


def barrier_insertion_loss(
    frequencies: ArrayLike,
    source_height: float,
    barrier_distance: float,
    barrier_height: float,
    receiver_distance: float,
    receiver_height: float,
    *,
    method: Literal["kurze_anderson", "exact"] = "exact",
    thickness: float | None = None,
    ground_impedance: ArrayLike | PorousMediumResult | None = None,
    ground_flow_resistivity: float | None = None,
    ground_model: Literal["delany_bazley", "miki"] = "delany_bazley",
    fluid: Fluid = PUBLISHED_AIR,
) -> BarrierInsertionLoss:
    r"""Insertion loss of a thin, thick or ground-coupled barrier.

    The 2-D geometry places the source at ``(0, source_height)``, the (near)
    diffraction edge at ``(barrier_distance, barrier_height)`` and the receiver
    at ``(receiver_distance, receiver_height)``. Three models are available:

    * ``method="kurze_anderson"``: the closed form
      :func:`kurze_anderson_attenuation` of the Fresnel number
      :func:`fresnel_number` (Bies Eqs. (5.134)/(5.138)); with ``thickness`` the
      double-edge Fresnel number :math:`N = (2/\lambda)(A + B + e - d)` of
      Bies Eq. (5.157) is used, ``e`` being the top width.
    * ``method="exact"`` without ground: the wave-theoretic insertion loss of the
      rigid thin screen (:func:`_screen_field`, MacDonald / Hadden & Pierce),
      :math:`\mathrm{IL} = 20 \log_{10} |p_{\text{free}} / p_{\text{diffracted}}|`.
    * ``method="exact"`` with a ground (``ground_impedance`` or
      ``ground_flow_resistivity``): the coherent four-path model. The field with
      the barrier sums the four source-image / receiver-image diffracted paths,
      each ground reflection weighted by the spherical-wave coefficient ``Q``
      (:func:`spherical_reflection_coefficient`); the field without the barrier
      is the two-ray ground field. This exposes the ground-barrier interference
      structure (Attenborough Ch. 9; Bies Sec. 5.3.5). As a first-order
      simplification a single ``Q`` (evaluated over the overall
      source-receiver geometry) weights every bounce rather than a separate
      coefficient per image path; the model is coherent and reciprocal but not
      a full boundary-element solution.

    :param frequencies: Frequencies, in hertz.
    :param source_height: Source height, in metres.
    :param barrier_distance: Horizontal source-to-barrier distance, in metres.
    :param barrier_height: Barrier (edge) height, in metres.
    :param receiver_distance: Horizontal source-to-receiver distance, in metres
        (``> barrier_distance``).
    :param receiver_height: Receiver height, in metres.
    :param method: ``"kurze_anderson"`` or ``"exact"``.
    :param thickness: Top width ``e`` of a thick barrier (double diffraction),
        in metres; ``None`` for a thin screen.
    :param ground_impedance: Normalized ground impedance for the coherent ground
        model (``"exact"`` only), in the :math:`e^{-i \omega t}` convention
        (:math:`\operatorname{Im}(Z) > 0` for a passive ground); a
        ``PorousMediumResult`` is conjugated internally from the materials'
        :math:`e^{+j \omega t}` convention.
    :param ground_flow_resistivity: Effective flow resistivity ``sigma``
        (Pa s/m2) for the ground model, as an alternative to ``ground_impedance``.
    :param ground_model: Porous model for ``ground_flow_resistivity``.
    :param fluid: The medium, a :class:`~phonometry.fluids.Fluid`
        (Default: :data:`PUBLISHED_AIR`, the air this model was published
        with). Pass a computed one, such as ``fluids.air(temperature_c=30.0,
        relative_humidity_percent=70.0)``, to work in the air of the room.
    :return: A :class:`BarrierInsertionLoss`.
    :raises ValueError: On a non-positive/ordered geometry, or if a ground is
        requested with ``method="kurze_anderson"``.
    """
    thickness = _validate_barrier_geometry(
        source_height,
        barrier_distance,
        barrier_height,
        receiver_distance,
        receiver_height,
        thickness,
    )
    has_ground = ground_impedance is not None or ground_flow_resistivity is not None
    speed_of_sound = fluid.speed_of_sound
    f = require_positive_array(frequencies, "frequencies")
    k = 2.0 * np.pi * f / speed_of_sound

    source = (0.0, source_height)
    edge = (barrier_distance, barrier_height)
    receiver = (receiver_distance, receiver_height)
    # Single-edge geometry (near edge). A thick barrier adds the far edge at the
    # same height, separated by the top width e.
    a_seg = float(np.hypot(source[0] - edge[0], source[1] - edge[1]))
    far_edge = (barrier_distance + (thickness or 0.0), barrier_height)
    b_seg = float(np.hypot(receiver[0] - far_edge[0], receiver[1] - far_edge[1]))
    d_seg = float(np.hypot(receiver[0] - source[0], receiver[1] - source[1]))
    delta = a_seg + b_seg + (thickness or 0.0) - d_seg
    n = np.asarray(2.0 * delta / (speed_of_sound / f), dtype=np.float64)

    if method == "kurze_anderson":
        if has_ground:
            msg = "the coherent ground model requires method='exact'."
            raise ValueError(msg)
        il = kurze_anderson_attenuation(n)
    elif method == "exact":
        il = _exact_barrier_il(
            _ExactBarrierSetup(
                frequencies=f,
                wavenumber=k,
                source=source,
                near_edge=edge,
                far_edge=far_edge,
                receiver=receiver,
                thickness=thickness,
                receiver_distance=receiver_distance,
                fluid=fluid,
                ground_impedance=ground_impedance,
                ground_flow_resistivity=ground_flow_resistivity,
                ground_model=ground_model,
            )
        )
    else:
        msg = f"unknown method {method!r}; options: {_BARRIER_METHODS}."
        raise ValueError(msg)
    return BarrierInsertionLoss(
        frequencies=f,
        insertion_loss=np.asarray(il, dtype=np.float64),
        fresnel_number=n,
        method=method,
        ground=has_ground,
        source_height=float(source_height),
        barrier_distance=float(barrier_distance),
        barrier_height=float(barrier_height),
        receiver_distance=float(receiver_distance),
        receiver_height=float(receiver_height),
        thickness=float(thickness) if thickness is not None else None,
    )


def _diffracted_over_barrier(
    source: tuple[float, float],
    near_edge: tuple[float, float],
    far_edge: tuple[float, float],
    receiver: tuple[float, float],
    thickness: float | None,
    k: Real,
) -> Complex:
    """Diffracted field over a thin (``near_edge``) or thick (``near``/``far``
    edges) barrier for one source/receiver pair; see :func:`_screen_field`.
    """
    if thickness is None:
        return _screen_field(source, near_edge, near_edge, receiver, k)
    return _screen_field(source, near_edge, far_edge, receiver, k)


@dataclass(frozen=True)
class _ExactBarrierSetup:
    """Bundled geometry and environment for the wave-theoretic barrier field.

    Groups the many inputs of :func:`_exact_barrier_il` into one object; ``2-D``
    points are ``(horizontal, height)``.
    """

    frequencies: Real
    wavenumber: Real
    source: tuple[float, float]
    near_edge: tuple[float, float]
    far_edge: tuple[float, float]
    receiver: tuple[float, float]
    thickness: float | None
    receiver_distance: float
    fluid: Fluid
    ground_impedance: ArrayLike | PorousMediumResult | None
    ground_flow_resistivity: float | None
    ground_model: str

    @property
    def has_ground(self) -> bool:
        """Whether the coherent four-path ground model applies."""
        return (
            self.ground_impedance is not None
            or self.ground_flow_resistivity is not None
        )


def _exact_barrier_il(setup: _ExactBarrierSetup) -> Real:
    """Insertion loss from the wave-theoretic diffracted field (with/without ground)."""
    k = setup.wavenumber
    source, receiver = setup.source, setup.receiver
    near_edge, far_edge = setup.near_edge, setup.far_edge
    thickness = setup.thickness
    if not setup.has_ground:
        p_diff = _diffracted_over_barrier(
            source, near_edge, far_edge, receiver, thickness, k
        )
        r_direct = float(np.hypot(source[0] - receiver[0], source[1] - receiver[1]))
        p_free = np.exp(1j * k * r_direct) / (4.0 * np.pi * r_direct)
        return np.asarray(
            -20.0 * np.log10(np.abs(p_diff) / np.abs(p_free)), dtype=np.float64
        )

    z = _normalized_ground_impedance(
        setup.frequencies,
        setup.ground_impedance,
        setup.ground_flow_resistivity,
        setup.ground_model,
        setup.fluid,
    )
    hs, hr = source[1], receiver[1]
    src_img = (source[0], -hs)
    rec_img = (receiver[0], -hr)
    # Spherical-wave reflection coefficient for the source-side and receiver-side
    # ground bounces (evaluated over the full source-receiver geometry).
    q_src = spherical_reflection_coefficient(
        setup.frequencies,
        z,
        hs,
        hr,
        setup.receiver_distance,
        setup.fluid.speed_of_sound,
    )
    q_rec = q_src
    # Four diffracted paths: {source, source image} x {receiver, receiver image}.
    p_barrier = (
        _diffracted_over_barrier(source, near_edge, far_edge, receiver, thickness, k)
        + q_src
        * _diffracted_over_barrier(src_img, near_edge, far_edge, receiver, thickness, k)
        + q_rec
        * _diffracted_over_barrier(source, near_edge, far_edge, rec_img, thickness, k)
        + q_src
        * q_rec
        * _diffracted_over_barrier(src_img, near_edge, far_edge, rec_img, thickness, k)
    )
    # Field without the barrier: the two-ray ground field.
    r1 = float(np.hypot(setup.receiver_distance, hs - hr))
    r2 = float(np.hypot(setup.receiver_distance, hs + hr))
    p_free_2ray = np.asarray(
        np.exp(1j * k * r1) / (4.0 * np.pi * r1)
        + q_src * np.exp(1j * k * r2) / (4.0 * np.pi * r2),
        dtype=np.complex128,
    )
    return np.asarray(
        -20.0 * np.log10(np.abs(p_barrier) / np.abs(p_free_2ray)), dtype=np.float64
    )
