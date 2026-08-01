#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Steady-state sound field in a room: room constant, critical distance, level.

When a source of constant sound power runs in a room, the sound pressure level
at a receiver settles to a steady value made of two parts: the **direct field**
that falls with distance as any free-field source does, and the **reverberant
field** built up by the many wall reflections, which is (to the diffuse-field
approximation) the same everywhere in the room. This module gives the classical
statistical-acoustics relations between the source power, the room's absorption
and the received level (Bies, Hansen & Howard, *Engineering Noise Control* 5th
ed., 6.4; Kuttruff, *Room Acoustics* 6th ed., 5.6), the bridge between the sound
power of :mod:`phonometry.emission` and the received level indoors.

**Room constant** :math:`R = S \bar{\alpha} / (1 - \bar{\alpha})` (Bies
Equation (6.44)), with the total boundary area ``S`` and its area-weighted
mean Sabine absorption ``alpha_bar``
(:func:`phonometry.room.mean_absorption`). ``R`` has units of
area and measures how much reverberant field a given power builds up: a live
room (small ``alpha_bar``) has a small ``R`` and a loud reverberant field, a
dead room a large ``R``.

**Steady-state level** (Bies Equation (6.43)):

.. math::

   L_p = L_W + 10 \log_{10}\!\left( \frac{Q}{4 \pi r^2} + \frac{4}{R}
   \right) \quad \left[ +\, 10 \log_{10}\frac{\rho c}{400} \right]

with the source directivity factor ``Q`` (:math:`Q = 1` omnidirectional,
:math:`Q = 2` on a
hard floor, ...), the distance ``r`` and the room constant ``R``. The first
term inside the bracket is the direct field, the second the reverberant field.
Dropping the direct term (``distance=None``) leaves
:math:`L_p = L_W + 10 \log_{10}(4/R)`,
the reverberant field alone: the level far from the source, and the one a
diffuse-field calculation such as a room-to-room transmission
(:mod:`phonometry.noise_control.room_to_room`) works from.
The optional :math:`10 \log_{10}(\rho c / 400)` term corrects for a
characteristic impedance ``rho c`` differing from the reference 400 Pa s/m; it
is about ``+0.14 dB`` at 20 degC and is omitted by default (Bies notes the
``~0.1 dB`` it contributes).

**Critical distance** :math:`r_c = \sqrt{Q R / (16 \pi)}` is where the direct
and reverberant terms are equal (setting
:math:`Q / (4 \pi r^2) = 4 / R` in Equation
(6.43)); closer than ``rc`` the direct field dominates, farther the reverberant
field does. Kuttruff's reverberation distance (Equation (5.44),
:math:`r_c = \sqrt{A / (16 \pi)}` for :math:`Q = 1`) uses the Sabine
absorption area :math:`A = S \bar{\alpha}` in place of the room constant
:math:`R = A / (1 - \bar{\alpha})`;
the two coincide for a small ``alpha_bar`` and differ by the factor
:math:`1 - \bar{\alpha}` otherwise. This module uses the room constant, so
``rc`` is exactly the crossover of its own :func:`steady_state_spl`.

**Sound power model.** Both relations above assume the radiated power itself is
a property of the machine. Norton & Karczub, *Fundamentals of Noise and
Vibration Analysis for Engineers* 2nd ed., 4.6 (Table 4.5), point out that this
is only one of three positions: a *constant-power* source radiates the same
:math:`\Pi_0` wherever it stands, a *constant-volume* source is loaded by
nearby reflecting boundaries and radiates :math:`\Pi_0 Q` (up to ``+9 dB`` in
a corner, the conservative upper bound a design estimate uses), and a
*constant-pressure* source, a theoretical lower bound, radiates
:math:`\Pi_0 / Q`. Real machines sit
between the first two whenever the source is closer to the boundary than a
wavelength. ``source_model`` selects which of the three
:data:`SOURCE_POWER_MODELS` is applied.

**Schroeder frequency** :math:`f_s = 2000 \sqrt{T / V}` (Kuttruff Equation
(3.44),
``V`` in cubic metres, ``T`` in seconds) marks the boundary between the
modal low-frequency regime -- where discrete room modes rule and the diffuse
assumption of ``R`` and ``rc`` fails -- and the high-frequency regime of
overlapping modes where the statistical field of this module applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.types import as_float_or_array
from .._internal.validation import require_choice, require_positive

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _require_fraction_below_one(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate a mean absorption in ``(0, 1)`` (scalar or per-band)."""
    arr = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"'{name}' must be finite.")
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError(
            f"'{name}' must lie strictly in (0, 1): the room constant "
            "R = S*alpha/(1 - alpha) diverges as the mean absorption tends "
            "to 1 (a perfectly absorbing room has no reverberant field)."
        )
    return arr


def room_constant(
    surface_area: float, mean_absorption: ArrayLike
) -> np.ndarray | float:
    r"""Room constant :math:`R = S \bar{\alpha} / (1 - \bar{\alpha})`
    (Bies Equation (6.44)).

    :param surface_area: Total boundary area ``S`` of the room, m2.
    :param mean_absorption: Area-weighted mean Sabine absorption
        ``alpha_bar`` in ``(0, 1)`` (scalar or per-band); e.g. from
        :func:`phonometry.room.mean_absorption`.
    :return: The room constant ``R``, m2; a float for a scalar input,
        otherwise a per-band array.
    """
    surface_area = require_positive(surface_area, "surface_area")
    alpha = _require_fraction_below_one(mean_absorption, "mean_absorption")
    return as_float_or_array(surface_area * alpha / (1.0 - alpha))


def critical_distance(
    room_constant: ArrayLike, *, directivity: float = 1.0
) -> np.ndarray | float:
    r"""Critical (reverberation) distance
    :math:`r_c = \sqrt{Q R / (16 \pi)}`.

    The distance at which the direct and reverberant fields of
    :func:`steady_state_spl` are equal (Bies Equation (6.43) crossover;
    Kuttruff Equation (5.44) states the :math:`Q = 1` form with the Sabine
    absorption area :math:`A = S \bar{\alpha}` instead of the room constant
    ``R``, the two differing by :math:`1 - \bar{\alpha}`).

    :param room_constant: Room constant ``R``, m2 (scalar or per-band); from
        :func:`room_constant`.
    :param directivity: Source directivity factor ``Q`` (``1`` omnidirectional,
        ``2`` on one reflecting plane, ``4`` in an edge, ``8`` in a corner).
    :return: The critical distance ``rc``, m.
    """
    directivity = require_positive(directivity, "directivity")
    r = np.asarray(room_constant, dtype=np.float64)
    if np.any(r <= 0.0) or not np.all(np.isfinite(r)):
        raise ValueError("'room_constant' must be positive and finite.")
    return as_float_or_array(np.sqrt(directivity * r / (16.0 * np.pi)))


def schroeder_frequency(
    reverberation_time: ArrayLike, volume: float
) -> np.ndarray | float:
    r"""Schroeder frequency :math:`f_s = 2000 \sqrt{T / V}`
    (Kuttruff Equation (3.44)).

    The frequency above which room modes overlap (on average three
    eigenfrequencies per resonance half-width) so the statistical, diffuse
    field of this module applies; below it the sound field is ruled by discrete
    modes and ``R`` / ``rc`` lose their meaning.

    :param reverberation_time: Reverberation time ``T``, s (scalar or per-band).
    :param volume: Room volume ``V``, m3.
    :return: The Schroeder frequency, Hz.
    """
    volume = require_positive(volume, "volume")
    t = np.asarray(reverberation_time, dtype=np.float64)
    if np.any(t <= 0.0) or not np.all(np.isfinite(t)):
        raise ValueError("'reverberation_time' must be positive and finite.")
    return as_float_or_array(2000.0 * np.sqrt(t / volume))


#: Sound power models of Norton & Karczub 2e Table 4.5, as the exponent of the
#: directivity factor ``Q`` that multiplies the radiated power: a constant-power
#: source radiates ``Pi_0`` wherever it stands, a constant-volume source
#: ``Pi_0 Q`` (reflecting boundaries load it and it radiates *more*, the
#: conservative upper bound), a constant-pressure source ``Pi_0 / Q`` (the
#: theoretical lower bound).
SOURCE_POWER_MODELS: dict[str, float] = {
    "constant_power": 0.0,
    "constant_volume": 1.0,
    "constant_pressure": -1.0,
}


def steady_state_spl(
    sound_power_level: ArrayLike,
    distance: ArrayLike | None,
    room_constant: ArrayLike,
    *,
    directivity: float = 1.0,
    source_model: str = "constant_power",
    characteristic_impedance: float | None = None,
) -> np.ndarray | float:
    r"""Steady-state sound pressure level in a room (Bies Equation (6.43)).

    .. math::

       L_p = L_W + 10 \log_{10}\!\left( \frac{Q}{4 \pi r^2}
       + \frac{4}{R} \right)

    plus the optional :math:`10 \log_{10}(\rho c / 400)`
    characteristic-impedance term. The bracket sums
    the direct field :math:`Q / (4 \pi r^2)` and the (position-independent)
    reverberant field :math:`4 / R`. Passing ``distance=None`` drops the
    direct term and leaves the reverberant field alone,
    :math:`L_p = L_W + 10 \log_{10}(4 / R)`:
    the limit far from the source, and the level a diffuse-field calculation
    such as a room-to-room transmission uses.

    :param sound_power_level: Source sound power level ``Lw``, dB re 1 pW
        (scalar or per-band); e.g. from :mod:`phonometry.emission`.
    :param distance: Source-receiver distance ``r``, m (scalar or array), or
        ``None`` for the reverberant field alone.
    :param room_constant: Room constant ``R``, m2 (scalar or per-band); from
        :func:`room_constant`.
    :param directivity: Source directivity factor ``Q`` (default 1; ``2`` on
        one reflecting plane, ``4`` in an edge, ``8`` in a corner).
    :param source_model: Sound power model of Norton & Karczub 2e Table 4.5:
        ``"constant_power"`` (default, :math:`\Pi = \Pi_0`, the position of
        the source does not change its radiated power), ``"constant_volume"``
        (:math:`\Pi = \Pi_0 Q`, reflecting boundaries raise the radiated
        power by :math:`10 \log_{10} Q`; the conservative upper bound) or
        ``"constant_pressure"`` (:math:`\Pi = \Pi_0 / Q`, the theoretical
        lower bound). See :data:`SOURCE_POWER_MODELS`.
    :param characteristic_impedance: Air characteristic impedance ``rho c``,
        Pa s/m. When given, the :math:`10 \log_{10}(\rho c / 400)` term is
        added (about ``+0.14 dB`` at 20 degC where :math:`\rho c = 413`);
        ``None`` (default) omits it, matching the common textbook form.
    :return: The steady-state SPL ``Lp``, dB; a float for scalar inputs,
        otherwise an array broadcasting ``sound_power_level``, ``distance``
        and ``room_constant``.
    :raises ValueError: If ``distance`` or ``room_constant`` is not positive and
        finite, or ``source_model`` is not one of
        :data:`SOURCE_POWER_MODELS`.
    """
    lw = np.asarray(sound_power_level, dtype=np.float64)
    # Named r_const (not rc) to avoid confusion with the critical distance rc.
    r_const = np.asarray(room_constant, dtype=np.float64)
    directivity = require_positive(directivity, "directivity")
    model = require_choice(
        source_model, "source_model", tuple(SOURCE_POWER_MODELS)
    )
    if np.any(r_const <= 0.0) or not np.all(np.isfinite(r_const)):
        raise ValueError("'room_constant' must be positive and finite.")
    bracket = 4.0 / r_const
    if distance is not None:
        r = np.asarray(distance, dtype=np.float64)
        if np.any(r <= 0.0) or not np.all(np.isfinite(r)):
            raise ValueError("'distance' must be positive and finite.")
        bracket = directivity / (4.0 * np.pi * r**2) + bracket
    lp = (
        lw
        + SOURCE_POWER_MODELS[model] * 10.0 * np.log10(directivity)
        + 10.0 * np.log10(bracket)
    )
    if characteristic_impedance is not None:
        rho_c = require_positive(characteristic_impedance, "characteristic_impedance")
        lp = lp + 10.0 * np.log10(rho_c / 400.0)
    return as_float_or_array(lp)


@dataclass(frozen=True)
class SteadyFieldResult:
    r"""Steady-state SPL versus distance in a room, split direct / reverberant.

    :ivar distances: Source-receiver distances ``r``, m.
    :ivar direct: Direct-field level
        :math:`L_W + 10 \log_{10}(Q / (4 \pi r^2))` per distance, dB.
    :ivar reverberant: Reverberant-field level
        :math:`L_W + 10 \log_{10}(4 / R)`, dB
        (constant across distance; broadcast to the distance grid).
    :ivar total: Combined steady-state level (Bies Equation (6.43)), dB.
    :ivar critical_distance: Critical distance ``rc``, m, where direct equals
        reverberant.
    :ivar room_constant: Room constant ``R``, m2.
    :ivar sound_power_level: Source sound power level ``Lw``, dB re 1 pW.
    :ivar directivity: Source directivity factor ``Q``.
    """

    distances: np.ndarray
    direct: np.ndarray
    reverberant: np.ndarray
    total: np.ndarray
    critical_distance: float
    room_constant: float
    sound_power_level: float
    directivity: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot direct, reverberant and total SPL against distance.

        Marks the critical distance ``rc`` where the direct and reverberant
        fields cross. Requires matplotlib (``pip install phonometry[plot]``);
        returns the :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.room import plot_steady_field

        check_language(language)
        return plot_steady_field(self, ax=ax, language=language, **kwargs)


def steady_state_field(
    sound_power_level: float,
    surface_area: float,
    mean_absorption: float,
    *,
    distances: ArrayLike | None = None,
    directivity: float = 1.0,
    characteristic_impedance: float | None = None,
) -> SteadyFieldResult:
    r"""Steady-state SPL versus distance for one source in a room (Bies 6.4).

    Builds the room constant from ``surface_area`` and ``mean_absorption``
    (Bies Equation (6.44)), then evaluates the direct, reverberant and combined
    fields (Equation (6.43)) over a distance grid together with the critical
    distance (crossover of the two fields).

    :param sound_power_level: Source sound power level ``Lw``, dB re 1 pW.
    :param surface_area: Total boundary area ``S``, m2.
    :param mean_absorption: Mean Sabine absorption ``alpha_bar`` in ``(0, 1)``.
    :param distances: Distance grid ``r``, m; default 30 points log-spaced from
        one tenth of the critical distance to ten times it.
    :param directivity: Source directivity factor ``Q`` (default 1).
    :param characteristic_impedance: Optional ``rho c`` for the Bies
        :math:`10 \log_{10}(\rho c / 400)` term (``None`` omits it).
    :return: A :class:`SteadyFieldResult`.
    """
    lw = float(sound_power_level)
    r_const = float(room_constant(surface_area, mean_absorption))
    q = require_positive(directivity, "directivity")
    rc = float(critical_distance(r_const, directivity=q))

    if distances is None:
        r = np.geomspace(0.1 * rc, 10.0 * rc, 30)
    else:
        r = np.asarray(distances, dtype=np.float64)
        if r.ndim != 1 or r.size == 0:
            raise ValueError("'distances' must be a non-empty 1D array.")
        if np.any(r <= 0.0) or not np.all(np.isfinite(r)):
            raise ValueError("'distances' must be positive and finite.")

    offset = (
        10.0 * np.log10(require_positive(characteristic_impedance, "characteristic_impedance") / 400.0)
        if characteristic_impedance is not None
        else 0.0
    )
    direct = lw + 10.0 * np.log10(q / (4.0 * np.pi * r**2)) + offset
    reverberant = np.full_like(r, lw + 10.0 * np.log10(4.0 / r_const) + offset)
    total = np.asarray(
        steady_state_spl(
            lw, r, r_const, directivity=q,
            characteristic_impedance=characteristic_impedance,
        ),
        dtype=np.float64,
    )
    return SteadyFieldResult(
        distances=r,
        direct=direct,
        reverberant=reverberant,
        total=total,
        critical_distance=rc,
        room_constant=r_const,
        sound_power_level=lw,
        directivity=float(q),
    )
