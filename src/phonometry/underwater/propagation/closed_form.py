#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Underwater sound propagation: propagation loss (closed-form).

Propagation loss ``PL``, :math:`N_\mathrm{PL}` (dB) is the difference between
the source level in a given direction and the mean-square sound pressure level
at the receiver, :math:`N_\mathrm{PL}(x) = L_\mathrm{S} - L_p(x)` (ISO 18405:2017,
3.4.1.4). Here it is the sum of geometrical spreading and volume absorption:

* :func:`spreading_loss` -- geometrical spreading, :math:`20 \log_{10} R`
  (spherical), :math:`10 \log_{10} R` (cylindrical) or spherical-then-cylindrical
  (``"practical"``).
* :func:`seawater_absorption` -- the volume absorption coefficient
  :math:`\alpha` in
  dB/km, from three coexisting formulations selectable through ``model``:
  Francois & Garrison (1982, the default and reference), Ainslie & McColm (1998,
  a legible simplification of it) and Thorp (1967, a frequency-only form).
* :func:`propagation_loss` -- the total
  :math:`\mathrm{PL} = \text{spreading} + \alpha R` versus range,
  returned as a :class:`PropagationLossResult` with a ``.plot()``.

Not *transmission loss*, which ISO 18405:2017, 3.4.1.3 reserves for the
reduction in a specified level between two specified points and whose Note 6 to
entry deprecates as a synonym of this quantity.

Sources (clean-room, implemented from the published equations): Francois &
Garrison, JASA 72 (1982) via Medwin & Clay; Ainslie & McColm, JASA 103 (1998);
Thorp (1967) via Etter (2003). Absorption is validated by the mutual agreement of
Francois-Garrison and Ainslie-McColm (~10 % as the latter's paper states;
marginally exceeded at the extreme corners of the stated domain, e.g. 10.4 % at
T = −6 °C / 1 MHz and 12.3 % at z = 7 km, a property of the published
simplification, both transcriptions verified digit-for-digit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..._internal.validation import (
    require_above_absolute_zero,
    require_ranks,
    require_same_length,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

_SPREADING_LAWS = ("spherical", "cylindrical", "practical")
_ABSORPTION_MODELS = ("francois-garrison", "ainslie-mccolm", "thorp")
#: Metres per kilometre.
_M_PER_KM = 1000.0
#: Francois-Garrison pure-water (A3) coefficient boundary, in degrees Celsius:
#: the "20 C switch" between the two published A3 cubics (see the note in
#: _francois_garrison -- they do not meet exactly there).
_FG_A3_SWITCH_T_C = 20.0


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        msg = f"'{name}' must be a positive, finite number."
        raise ValueError(msg)
    return scalar


def _positive_array(
    values: NDArray[np.float64] | list[float] | float, name: str
) -> NDArray[np.float64]:
    arr = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite and non-empty."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = f"'{name}' must be strictly positive."
        raise ValueError(msg)
    return arr


def spreading_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    *,
    law: str = "spherical",
    transition_range: float | None = None,
) -> NDArray[np.float64]:
    r"""Geometrical spreading loss, in dB.

    ``"spherical"`` gives :math:`20 \log_{10}(R)` (free field), ``"cylindrical"``
    gives :math:`10 \log_{10}(R)` (perfect waveguide) and ``"practical"`` is
    spherical up to ``transition_range`` ``R0`` then cylindrical:
    :math:`20 \log_{10}(R_0) + 10 \log_{10}(R/R_0)` (mode stripping in a channel).

    :param range_m: Range ``R`` from the source, in metres (scalar or array,
        strictly positive).
    :param law: ``"spherical"`` (default), ``"cylindrical"`` or ``"practical"``.
    :param transition_range: Transition range ``R0`` in metres; required for
        ``"practical"``.
    :return: Spreading loss per range, in dB.
    :raises ValueError: If the inputs are invalid.
    """
    r = _positive_array(range_m, "range_m")
    key = law.strip().lower()
    if key == "spherical":
        return 20.0 * np.log10(r)
    if key == "cylindrical":
        return 10.0 * np.log10(r)
    if key == "practical":
        if transition_range is None:
            msg = "'transition_range' is required for the 'practical' law."
            raise ValueError(msg)
        r0 = _positive(transition_range, "transition_range")
        return np.where(
            r <= r0, 20.0 * np.log10(r), 20.0 * np.log10(r0) + 10.0 * np.log10(r / r0)
        )
    msg = f"'law' must be one of {_SPREADING_LAWS}, got {law!r}."
    raise ValueError(msg)


def _thorp(f_khz: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0936 * (
        0.1 * f_khz**2 / (1.0 + f_khz**2) + 40.0 * f_khz**2 / (4100.0 + f_khz**2)
    )


#: The Celsius-to-kelvin offset Francois & Garrison print inside their two
#: relaxation frequencies. It is 273, not 273,15, and it is normative: the pole
#: of this model therefore sits 0,15 degC above absolute zero.
_FG_KELVIN_OFFSET = 273.0


def _francois_garrison(
    f_khz: NDArray[np.float64], t: float, s: float, z_m: float, ph: float
) -> NDArray[np.float64]:
    # Sound speed c, printed as "q = 1412 + 3.21T + 1.19S + 0.0167z" in the
    # Medwin & Clay transcription (Eq. (3.4.30), printed p. 110) although the
    # same block divides by c; the original paper names it c. See ERRATA.
    c = 1412.0 + 3.21 * t + 1.19 * s + 0.0167 * z_m
    # Boric-acid factor A1 = (8.86/c)*10^(0.78 pH - 5) per the original paper
    # (Francois & Garrison 1982 Part II, Eq. (10)/Fig. 7), whose own Table IV
    # reproduces only with 8.86; the Medwin & Clay transcription (Eq. (3.4.30),
    # printed p. 110) prints 8.68 (digit transposition), biasing
    # boric-dominated bands by up to 1.7 %.
    a1 = (8.86 / c) * 10.0 ** (0.78 * ph - 5.0)
    f1 = 2.8 * (s / 35.0) ** 0.5 * 10.0 ** (4.0 - 1245.0 / (273.0 + t))
    a2 = 21.44 * (s / c) * (1.0 + 0.025 * t)
    p2 = 1.0 - 1.37e-4 * z_m + 6.2e-9 * z_m**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (273.0 + t)) / (1.0 + 0.0018 * (s - 35.0))
    # The two published A3 cubics do not meet exactly at the 20 C switch
    # (step of 1e-7*f^2 dB/km, i.e. 0.1 dB/km at 1 MHz, 0.03 % of alpha there);
    # inherent in the Francois-Garrison coefficients -- do not "fix" it.
    if t <= _FG_A3_SWITCH_T_C:
        a3 = 4.937e-4 - 2.59e-5 * t + 9.11e-7 * t**2 - 1.50e-8 * t**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * t + 1.45e-7 * t**2 - 6.50e-10 * t**3
    p3 = 1.0 - 3.83e-5 * z_m + 4.9e-10 * z_m**2
    boric = a1 * f1 * f_khz**2 / (f_khz**2 + f1**2)
    mgso4 = a2 * p2 * f2 * f_khz**2 / (f_khz**2 + f2**2)
    water = a3 * p3 * f_khz**2
    return np.asarray(boric + mgso4 + water, dtype=np.float64)


def _ainslie_mccolm(
    f_khz: NDArray[np.float64], t: float, s: float, z_km: float, ph: float
) -> NDArray[np.float64]:
    f1 = 0.78 * (s / 35.0) ** 0.5 * np.exp(t / 26.0)
    f2 = 42.0 * np.exp(t / 17.0)
    boric = 0.106 * f1 * f_khz**2 / (f_khz**2 + f1**2) * np.exp((ph - 8.0) / 0.56)
    mgso4 = (
        0.52
        * (1.0 + t / 43.0)
        * (s / 35.0)
        * f2
        * f_khz**2
        / (f_khz**2 + f2**2)
        * np.exp(-z_km / 6.0)
    )
    water = 0.00049 * f_khz**2 * np.exp(-(t / 27.0 + z_km / 17.0))
    return np.asarray(boric + mgso4 + water, dtype=np.float64)


def seawater_absorption(
    frequency_hz: NDArray[np.float64] | list[float] | float,
    *,
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = "francois-garrison",
) -> NDArray[np.float64]:
    r"""Volume absorption coefficient :math:`\alpha`, in dB/km.

    :param frequency_hz: Acoustic frequency, in Hz (scalar or array).
    :param temperature: Temperature ``T``, in degrees Celsius.
    :param salinity: Salinity ``S``, in parts per thousand.
    :param depth: Depth, in metres (``>= 0``).
    :param ph: Acidity (used by Francois-Garrison and Ainslie-McColm; default 8).
    :param model: ``"francois-garrison"`` (default), ``"ainslie-mccolm"`` or
        ``"thorp"`` (the Thorp 1967 frequency-only form of Etter, valid below
        ~50 kHz; ignores ``temperature``/``salinity``/``depth``/``ph``).
    :return: Absorption coefficient per frequency, in dB/km.
    :raises ValueError: If ``model`` is unknown or an input is invalid.
    """
    f_khz = _positive_array(frequency_hz, "frequency_hz") / 1000.0
    t = float(temperature)
    s = float(salinity)
    z = float(depth)
    if not (np.isfinite(t) and np.isfinite(s) and np.isfinite(z) and np.isfinite(ph)):
        msg = "'temperature', 'salinity', 'depth' and 'ph' must be finite."
        raise ValueError(msg)
    if s < 0.0 or z < 0.0:
        msg = "'salinity' and 'depth' must be non-negative."
        raise ValueError(msg)
    require_above_absolute_zero(t, "temperature")
    key = model.strip().lower()
    if key == "thorp":
        return _thorp(f_khz)
    if key == "francois-garrison":
        # This model has a pole of its own, 0,15 degC above absolute zero: its
        # relaxation frequencies are written 1245/(273 + t) and 1990/(273 + t)
        # with the 273 the paper prints, so the whole band (-273,15, -273,0]
        # clears the check above and then overflows or divides by zero, neither
        # of which names the argument that caused it. The printed 273 is
        # normative and stays; the guard goes in front of it.
        if _FG_KELVIN_OFFSET + t <= 0.0:
            msg = (
                "'temperature' must exceed -273 degC for the francois-garrison "
                "model, whose relaxation frequencies are printed as "
                "1245/(273 + t) and 1990/(273 + t)."
            )
            raise ValueError(msg)
        return _francois_garrison(f_khz, t, s, z, float(ph))
    if key == "ainslie-mccolm":
        return _ainslie_mccolm(f_khz, t, s, z / _M_PER_KM, float(ph))
    msg = f"'model' must be one of {_ABSORPTION_MODELS}, got {model!r}."
    raise ValueError(msg)


@dataclass(frozen=True)
class PropagationLossResult:
    r"""Propagation loss versus range (closed-form).

    :ivar range_m: Ranges from the source, in metres.
    :ivar pl: Total propagation loss per range, in dB.
    :ivar spreading: Geometrical-spreading contribution per range, in dB.
    :ivar absorption: Volume-absorption contribution per range, in dB.
    :ivar frequency: The acoustic frequency, in Hz.
    :ivar absorption_coefficient: The absorption coefficient :math:`\alpha`,
        in dB/km.
    :ivar law: The spreading law used.
    :ivar model: The absorption model used.
    """

    range_m: NDArray[np.float64]
    pl: NDArray[np.float64]
    spreading: NDArray[np.float64]
    absorption: NDArray[np.float64]
    frequency: float
    absorption_coefficient: float
    law: str
    model: str

    def __post_init__(self) -> None:
        """Reject a loss curve whose terms do not run over the same ranges.

        The three curves are one decomposition, and it is a decomposition only
        range by range: ``pl`` is ``spreading + absorption`` at each entry of
        ``range_m``. That is what a reader takes it apart for -- to subtract
        the absorption and put back the one a different water would charge, or
        to read the spreading law's exponent off its own term -- and every one
        of those operations is elementwise. A term of another length is not
        obviously wrong to look at, since all three are losses in dB of
        entirely plausible size; it is wrong because each of its values then
        answers for a range the row beside it did not come from. The figure
        does stop on a length disagreement, all three curves being drawn
        against ``range_m``, but with matplotlib's own ``x and y must have
        same first dimension, but have shapes (4,) and (3,)`` -- shapes only,
        naming neither field, and raised by the drawing rather than by the
        result that was wrong all along.

        The ranks are pinned as well, because a count cannot see an extra
        axis. A ``pl`` of shape ``(n, 2)`` -- two waters stacked side by side,
        or ``(range, loss)`` pairs kept in one array -- carries one value per
        range on its first axis exactly as a single curve does, passes the
        length check, and reaches matplotlib, which draws every column it is
        handed: the figure comes back with a second total-loss curve and the
        ``Total PL`` legend row printed twice. A whole result of that shape is
        what :func:`propagation_loss` would build from ranges handed to it as
        a grid, the input check testing for finite positive values and not for
        rank: from a two-by-three grid the figure would come back with nine
        lines under three distinct legend labels, each one running down a
        column and so joining ranges that are not neighbours -- the first of
        them straight from 100 m to 1000 m, with the 200 m and 400 m between
        them handed to two other lines.

        :raises ValueError: if any curve disagrees with ``range_m``, or if one
            carries an axis beyond the range axis.
        """
        require_ranks(self, range_m=1, pl=1, spreading=1, absorption=1)
        require_same_length(
            self, "range_m", "pl", "spreading", "absorption", axis="range"
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the propagation loss versus range with its two contributions."""
        from ..._i18n import check_language
        from ..._plot.underwater import plot_propagation_loss

        return plot_propagation_loss(
            self, ax=ax, language=check_language(language), **kwargs
        )


def propagation_loss(
    range_m: NDArray[np.float64] | list[float] | float,
    frequency_hz: float,
    *,
    law: str = "spherical",
    temperature: float = 10.0,
    salinity: float = 35.0,
    depth: float = 0.0,
    ph: float = 8.0,
    model: str = "francois-garrison",
    transition_range: float | None = None,
) -> PropagationLossResult:
    r"""Total propagation loss
    :math:`\mathrm{PL} = \text{spreading} + \alpha R` versus range.

    :param range_m: Range(s) from the source, in metres (scalar or array).
    :param frequency_hz: Acoustic frequency, in Hz.
    :param law: Spreading law (see :func:`spreading_loss`).
    :param temperature: Temperature ``T``, in degrees Celsius.
    :param salinity: Salinity ``S``, in parts per thousand.
    :param depth: Depth, in metres.
    :param ph: Acidity (default 8).
    :param model: Absorption model (see :func:`seawater_absorption`).
    :param transition_range: Transition range for the ``"practical"`` law, in m.
    :return: A :class:`PropagationLossResult`.
    :raises ValueError: If the inputs are invalid.
    """
    f = _positive(frequency_hz, "frequency_hz")
    r = _positive_array(range_m, "range_m")
    spreading = spreading_loss(r, law=law, transition_range=transition_range)
    alpha = float(
        seawater_absorption(
            f,
            temperature=temperature,
            salinity=salinity,
            depth=depth,
            ph=ph,
            model=model,
        )[0]
    )
    absorption = alpha * (r / _M_PER_KM)
    return PropagationLossResult(
        range_m=r,
        pl=spreading + absorption,
        spreading=spreading,
        absorption=absorption,
        frequency=f,
        absorption_coefficient=alpha,
        law=law.strip().lower(),
        model=model.strip().lower(),
    )
