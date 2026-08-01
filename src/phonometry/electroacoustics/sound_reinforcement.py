#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Gain before feedback of a sound-reinforcement system.

A public-address system is a closed loop: the loudspeaker feeds the audience,
but it also feeds the microphone that drives it. Long (*Architectural
Acoustics* 2nd ed., Chapter 18, Equations (18.13) to (18.24)) writes the loop
in terms of two decibel gains,

* the **open-loop system gain** :math:`Z_S` (Equation (18.17)), the level
  the loudspeaker produces at an average listener minus the level the
  talker produces at the microphone,

  .. math::

     Z_S = L_{H\text{-}L} - L_{T\text{-}M}

  so :math:`Z_S = -6` dB (a typical auditorium or church) means the
  amplified sound at the listener sits 6 dB below what the talker delivers
  to the microphone, i.e. a comfortable conversational level at twice the
  talker-to-microphone distance;

* the **feedback-loop gain** :math:`G_S` (Equation (18.18)), the part of
  that output that returns to the microphone,

  .. math::

     G_S = L_{H\text{-}M} - L_{H\text{-}L} + D_M(\theta)

  with :math:`D_M(\theta)` the directivity index of the microphone toward
  the loudspeaker *relative to* the talker (zero for an omnidirectional
  microphone, about -2 to -3 dB for a cardioid pointed at the talker).

Summing the infinite series of round trips (Equation (18.14)) makes the system
oscillate when the loop gain reaches unity, that is (Equation (18.16))

.. math::

   Z_S + G_S = 0

Long takes a **feedback stability margin** of 10 dB for an equalised system
(other authors quote 12 dB unequalised and 6 dB carefully equalised): 6 dB of
it covers a tone that adds in phase with a reflection from a hard surface, the
remaining 4 dB is safety. With several microphones open at once the returned
signals add at the mixer, which is accounted for by the *number of open
microphones* correction (Equation (18.23))
:math:`\Delta L_\text{nom} = 10 \log_{10} N_m`. The stability criterion is
then Equation (18.24),

.. math::

   Z_S + L_{H\text{-}M} + \Delta L_\text{nom}
   \le L_{H\text{-}L} - D_M(\theta) - 10

.. note::
   Long prints Equation (18.24) with :math:`+ D_M(\theta)` on the
   right-hand side, which contradicts Equations (18.20) to (18.22) it
   generalises (and would flip the benefit of a directional microphone
   into a penalty; see ``docs/ERRATA.md``). This module implements the
   sign of Equation (18.20), so that :math:`N_m = 1` reproduces Long's
   own special cases: with :math:`Z_S = -6` dB the criterion collapses to
   :math:`L_{H\text{-}M} \le L_{H\text{-}L} - D_M(\theta) - 4`
   (Equation (18.21)), which for an omnidirectional microphone puts the
   loudspeaker level at the microphone 4 dB below the average level in
   the audience, and for a cardioid (:math:`D_M = -2` dB) 2 dB below it
   (Equation (18.22)).

Note what the criterion does *not* contain: the loudspeaker type, its number
or its power. Only the sound field it produces at two points and the type and
orientation of the microphone matter. Long also excludes the reverberant field
deliberately, as it is uniform and therefore does not depend on where the
microphone is or where it points; the direct-to-reverberant ratio enters the
design separately, through the intelligibility table of his Table 18.3
(better than -3 dB excellent, -3 to -6 very good, -6 to -9 good, -9 to -12
fair, -12 to -15 poor, below -15 very poor).

This module is a closed-form design calculator anchored on those equations.
Long gives no fully numeric worked case for them, so the tests anchor on the
closed forms and on the special cases he states in words.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Feedback stability margin used by Long for an equalised system, dB.
DEFAULT_STABILITY_MARGIN = 10.0

#: Directivity index of a cardioid microphone toward the loudspeaker relative
#: to the talker, dB (Long, Chapter 18: "-2 or -3 dB of relative directivity").
CARDIOID_RELATIVE_DIRECTIVITY = -2.0


def open_microphone_correction(open_microphones: int) -> float:
    r"""Number-of-open-microphones correction (Long Equation (18.23)).

    :math:`\Delta L_\text{nom} = 10 \log_{10} N_m`: doubling the number
    of simultaneously open microphones costs about 3 dB of gain before
    feedback.

    :param open_microphones: Number :math:`N_m` of microphones open at
        once (>= 1).
    :return: The correction :math:`\Delta L_\text{nom}`, dB.
    """
    n = int(open_microphones)
    if n != open_microphones or n < 1:
        raise ValueError("'open_microphones' must be an integer of at least 1.")
    return 10.0 * math.log10(n)


def feedback_loop_gain(
    level_loudspeaker_at_microphone: float,
    level_loudspeaker_at_listener: float,
    *,
    microphone_directivity: float = 0.0,
) -> float:
    r"""Feedback-loop gain :math:`G_S` (Long Equation (18.18)).

    :math:`G_S = L_{H\text{-}M} - L_{H\text{-}L} + D_M(\theta)`.

    :param level_loudspeaker_at_microphone: Direct-field level
        :math:`L_{H\text{-}M}` the loudspeaker produces at the
        microphone, dB.
    :param level_loudspeaker_at_listener: Direct-field level
        :math:`L_{H\text{-}L}` the loudspeaker produces at an average
        listener, dB.
    :param microphone_directivity: Directivity index :math:`D_M(\theta)`
        of the microphone toward the loudspeaker relative to the talker,
        dB (0 for an omnidirectional microphone, about
        :data:`CARDIOID_RELATIVE_DIRECTIVITY` for a cardioid).
    :return: The feedback-loop gain :math:`G_S`, dB.
    """
    values = (
        float(level_loudspeaker_at_microphone),
        float(level_loudspeaker_at_listener),
        float(microphone_directivity),
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Levels and the directivity index must be finite.")
    l_hm, l_hl, d_m = values
    return l_hm - l_hl + d_m


@dataclass(frozen=True)
class FeedbackStabilityResult:
    r"""Gain structure and stability verdict of a reinforcement loop.

    :ivar open_loop_gain: Open-loop system gain :math:`Z_S`, dB
        (Equation (18.17)).
    :ivar feedback_loop_gain: Feedback-loop gain :math:`G_S`, dB
        (Equation (18.18)).
    :ivar nom_correction: Number-of-open-microphones correction
        :math:`\Delta L_\text{nom}`, dB (Equation (18.23)).
    :ivar loop_gain: Total loop gain
        :math:`Z_S + G_S + \Delta L_\text{nom}`, dB. The system
        oscillates at 0 dB (Equation (18.16)).
    :ivar stability_margin: Required margin below oscillation, dB.
    :ivar margin: Margin actually available, ``-loop_gain``, dB.
    :ivar headroom: Gain that may still be added before the required margin is
        used up, ``-stability_margin - loop_gain``, dB; negative when the
        criterion of Equation (18.24) is already violated.
    :ivar is_stable: Whether the criterion of Equation (18.24) holds.
    :ivar maximum_open_loop_gain: Largest :math:`Z_S` the loop tolerates,
        dB.
    :ivar maximum_level_at_microphone: Largest :math:`L_{H\text{-}M}` the
        loop tolerates, dB, for the given :math:`Z_S` (Equations (18.20)
        to (18.22)).
    :ivar level_loudspeaker_at_microphone: Input :math:`L_{H\text{-}M}`,
        dB.
    :ivar level_loudspeaker_at_listener: Input :math:`L_{H\text{-}L}`, dB.
    :ivar microphone_directivity: Input :math:`D_M(\theta)`, dB.
    :ivar open_microphones: Input :math:`N_m`.
    """

    open_loop_gain: float
    feedback_loop_gain: float
    nom_correction: float
    loop_gain: float
    stability_margin: float
    margin: float
    headroom: float
    is_stable: bool
    maximum_open_loop_gain: float
    maximum_level_at_microphone: float
    level_loudspeaker_at_microphone: float
    level_loudspeaker_at_listener: float
    microphone_directivity: float
    open_microphones: int

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        r"""Plot the gain structure against the oscillation and margin
        lines.

        Bars for :math:`Z_S`, :math:`G_S` and :math:`\Delta L_\text{nom}`
        accumulate into the total loop gain, with the 0 dB oscillation
        threshold of Equation (18.16) and the required stability margin
        marked. Requires matplotlib (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_feedback_stability

        check_language(language)
        return plot_feedback_stability(self, ax=ax, language=language, **kwargs)


def feedback_stability(
    open_loop_gain: float,
    level_loudspeaker_at_microphone: float,
    level_loudspeaker_at_listener: float,
    *,
    microphone_directivity: float = 0.0,
    open_microphones: int = 1,
    stability_margin: float = DEFAULT_STABILITY_MARGIN,
) -> FeedbackStabilityResult:
    r"""Stability of a reinforcement loop (Long Equations (18.16) to
    (18.24)).

    The loop gain :math:`Z_S + G_S + \Delta L_\text{nom}` is compared
    with the oscillation threshold of Equation (18.16) reduced by the
    stability margin, that is Equation (18.24) written with the sign of
    Equation (18.20) (see the module docstring).

    :param open_loop_gain: Open-loop system gain
        :math:`Z_S = L_{H\text{-}L} - L_{T\text{-}M}`, dB; about -6 dB
        for a typical auditorium or church.
    :param level_loudspeaker_at_microphone: Direct-field level
        :math:`L_{H\text{-}M}` produced by the loudspeaker system at the
        microphone, dB.
    :param level_loudspeaker_at_listener: Direct-field level
        :math:`L_{H\text{-}L}` produced by the loudspeaker system at an
        average listener, dB.
    :param microphone_directivity: Directivity index :math:`D_M(\theta)`
        of the microphone toward the loudspeaker relative to the talker,
        dB.
    :param open_microphones: Number :math:`N_m` of microphones open at
        once.
    :param stability_margin: Required margin below oscillation, dB (default
        :data:`DEFAULT_STABILITY_MARGIN`, Long's equalised-system value).
    :return: A :class:`FeedbackStabilityResult`.
    """
    z_s = float(open_loop_gain)
    if not math.isfinite(z_s):
        raise ValueError("'open_loop_gain' must be finite.")
    margin_required = float(stability_margin)
    if not math.isfinite(margin_required) or margin_required < 0.0:
        raise ValueError("'stability_margin' must be finite and non-negative.")

    g_s = feedback_loop_gain(
        level_loudspeaker_at_microphone,
        level_loudspeaker_at_listener,
        microphone_directivity=microphone_directivity,
    )
    d_nom = open_microphone_correction(open_microphones)
    loop = z_s + g_s + d_nom
    l_hm = float(level_loudspeaker_at_microphone)
    l_hl = float(level_loudspeaker_at_listener)
    d_m = float(microphone_directivity)

    return FeedbackStabilityResult(
        open_loop_gain=z_s,
        feedback_loop_gain=g_s,
        nom_correction=d_nom,
        loop_gain=loop,
        stability_margin=margin_required,
        margin=-loop,
        headroom=-margin_required - loop,
        is_stable=loop <= -margin_required,
        maximum_open_loop_gain=-margin_required - g_s - d_nom,
        maximum_level_at_microphone=l_hl - d_m - margin_required - z_s - d_nom,
        level_loudspeaker_at_microphone=l_hm,
        level_loudspeaker_at_listener=l_hl,
        microphone_directivity=d_m,
        open_microphones=int(open_microphones),
    )
