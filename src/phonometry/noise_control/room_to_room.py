#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Room-to-room noise reduction: source room, partition, receiving room, criterion.

The other classic bookkeeping exercise of noise control, beside the duct-borne
cascade of :mod:`phonometry.noise_control.duct_path`. A machine runs in one
room, a partition separates that room from an occupied one, and the question is
the octave-band spectrum in the occupied room and whether it meets a design
criterion. The answer is a short chain, and every link of it already exists in
the library: the reverberant level the machine builds up in the source room
(:func:`phonometry.room.steady_state_spl` over the room constant of
:func:`phonometry.room.room_constant`), the transmission loss of the partition
(measured, or predicted by :mod:`phonometry.building.prediction.panel_transmission`), the
absorption of the receiving room
(:func:`phonometry.room.equivalent_absorption_area`) and the rating of what
arrives (:func:`phonometry.room.noise_criterion`). What this module adds is the
one step that lives nowhere else, and the result object that carries the whole
path.

**The step.** Norton & Karczub, *Fundamentals of Noise and Vibration Analysis
for Engineers* 2nd ed., 4.9, balance the steady-state power crossing the
partition against the power the receiving room absorbs and the power that leaks
back, and arrive at Equation (4.101):

.. math::

   \mathrm{NR} = \mathrm{TL} -
   10 \log_{10}\!\left[\frac{S_\mathrm{w}}{S_2 \alpha_2 + \tau S_\mathrm{w}}\right],

with :math:`\mathrm{NR} = L_{p1} - L_{p2}` the noise reduction between the two
reverberant
fields, :math:`\mathrm{TL} = 10 \log_{10}(1/\tau)` the transmission loss of the
partition, ``S_w``
the area of the partition and ``S_2 alpha_2`` the equivalent absorption area of
the receiving room. The ``tau S_w`` term is the power the partition itself
passes back and is normally negligible beside the room absorption; it is
switched on with ``include_partition_transmission``.

The single most useful thing the equation says is that **the noise reduction is
not the transmission loss**. A large partition into a hard room delivers less
than its ``TL``; a small partition into a well-absorbing room delivers more.
Norton also warns that the measured noise reduction runs a few decibels below
the prediction because of flanking transmission through mechanical connections
and air leaks, which ``flanking_penalty`` applies as an explicit debit.

**The source-room level.** In a plant room the receiver of interest is the
partition, not a point near the machine, so the level that drives the
transmission is the reverberant field alone,
:math:`L_{p1} = L_W + 10 \log_{10}(4 / R_1)`. That is
:func:`phonometry.room.steady_state_spl` at ``distance=None``, and this module
delegates to it rather than repeating it. Norton's Table 4.5 adds the choice of
*sound power model*: a machine standing in the intersection of a floor and a
wall radiates more power than its free-space ``L_W`` if it behaves as a
constant-volume source, and ``source_model="constant_volume"`` is the
conservative upper bound a design estimate uses.

**The verdict.** :attr:`RoomToRoomResult.required_transmission_loss` inverts the
chain: given the source-room level and the criterion curve, the partition
transmission loss that would just meet it, band by band. That is the number a
partition is specified from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
)
from ._criterion import (
    as_band_spectrum,
    curve_of,
    exceedance_of,
    meets_target_of,
    rating_of,
    validate_target,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ..room.noise_criteria import NCResult, RCResult


@dataclass(frozen=True)
class SourceRoom:
    r"""The source room of the chain: what drives the partition.

    Two ways to say the same thing, and exactly one of them is given. Either
    the reverberant level in the source room is known (``level``), or the
    machine's sound power level is (``power_level``), in which case the room
    constant of the source room is needed to build the reverberant field
    :math:`L_{p1} = L_W + 10 \log_{10}(4 / R_1)` (Norton 2e, 4.7), and the
    sound power model of Table 4.5 decides whether the position of the source
    in the room raises or lowers the power it radiates.

    :param level: Reverberant sound pressure level in the source room
        ``L_p1``, dB (scalar or per band). Mutually exclusive with
        ``power_level``.
    :param power_level: Sound power level of the source ``L_W``, dB re 1 pW
        (scalar or per band). Requires ``room_constant``.
    :param room_constant: Room constant of the source room ``R_1``, m2
        (scalar or per band); from :func:`phonometry.room.room_constant`.
    :param directivity: Directivity factor ``Q`` of the source in the source
        room (``1`` in free space, ``2`` on one plane, ``4`` in an edge, ``8``
        in a corner). Only affects the level through ``model``, because the
        reverberant field itself is position-independent.
    :param model: Sound power model of Norton Table 4.5:
        ``"constant_power"`` (default, the radiated power does not depend on
        the source position), ``"constant_volume"`` (the conservative upper
        bound, the power rises by :math:`10 \log_{10} Q`) or
        ``"constant_pressure"`` (the lower bound, it falls by
        :math:`10 \log_{10} Q`).
    """

    level: ArrayLike | None = None
    power_level: ArrayLike | None = None
    room_constant: ArrayLike | None = None
    directivity: float = 1.0
    model: str = "constant_power"

    def __post_init__(self) -> None:
        """Refuse a source room that does not describe one source.

        The two ways of saying it are mutually exclusive, and the power-level
        way needs the room constant that turns a power into a level. Checked
        here rather than where the object is used, so the complaint arrives
        at the line that built it.
        """
        if (self.level is None) == (self.power_level is None):
            msg = (
                "give exactly one of 'level' (the source-room level L_p1) or "
                "'power_level' (the source sound power level L_W)."
            )
            raise ValueError(msg)
        if self.power_level is not None and self.room_constant is None:
            msg = (
                "'room_constant' is required with 'power_level'; build it "
                "with phonometry.room.room_constant."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class DesignCriterion:
    """The design criterion the receiving room is held to.

    The verdict half of the chain: which room-criterion family the received
    spectrum is rated against, the curve it has to stay under, and the
    allowance a design sheet keeps for the transmission the calculation does
    not model.

    :param family: Room-criterion family, ``"NC"`` (default) or ``"RC"``.
    :param target: The design criterion value (e.g. ``45`` for NC 45), or
        ``None`` for no target, which leaves the verdicts undefined.
    :param flanking_penalty: Decibels debited from the predicted noise
        reduction for flanking transmission through mechanical connections and
        air leaks (Norton's "a few dB"). Default ``0``.
    """

    family: str = "NC"
    target: float | None = None
    flanking_penalty: float = 0.0


@dataclass(frozen=True)
class RoomToRoomResult:
    r"""The room-to-room chain of one partition (Norton 2e, 4.9).

    Built by :func:`room_to_room_transmission`. The four spectra are the four
    rows a hand calculation writes down: the level in the source room, the
    transmission loss of the partition, the noise reduction the partition and
    the receiving room deliver together, and what arrives.

    :ivar frequencies: Octave-band centre frequencies, Hz.
    :ivar source_level: Reverberant sound pressure level in the source room
        ``L_p1``, dB.
    :ivar transmission_loss: Transmission loss of the partition ``TL``, dB.
    :ivar partition_area: Area of the partition ``S_w``, m2.
    :ivar receiving_absorption: Equivalent absorption area of the receiving
        room ``S_2 alpha_2`` per band, m2.
    :ivar noise_reduction: The delivered noise reduction ``NR`` per band, dB:
        Equation (4.101) less :attr:`flanking_penalty`.
    :ivar received_level: Reverberant sound pressure level in the receiving
        room :math:`L_{p2} = L_{p1} - \mathrm{NR}`, dB.
    :ivar flanking_penalty: The debit applied to the predicted noise reduction
        for flanking transmission and air leaks, dB.
    :ivar source_power_level: The source sound power level ``L_W`` the source
        level was built from, dB re 1 pW, or ``None`` when ``source.level`` was
        given directly.
    :ivar criterion: ``"NC"`` or ``"RC"``, the room-criterion family.
    :ivar target: The design criterion value (e.g. ``45`` for NC 45), or
        ``None``.
    :ivar label: A short human label of the chain.
    """

    frequencies: np.ndarray
    source_level: np.ndarray
    transmission_loss: np.ndarray
    partition_area: float
    receiving_absorption: np.ndarray
    noise_reduction: np.ndarray
    received_level: np.ndarray
    flanking_penalty: float
    source_power_level: np.ndarray | None
    criterion: str
    target: float | None
    label: str

    # -- verdicts ---------------------------------------------------------
    @property
    def rating(self) -> NCResult | RCResult:
        """The rating of the spectrum arriving in the receiving room.

        An :class:`~phonometry.room.noise_criteria.NCResult` for the ``"NC"``
        family, an :class:`~phonometry.room.noise_criteria.RCResult` for ``"RC"``.
        """
        return rating_of(self)

    @property
    def criterion_curve(self) -> np.ndarray | None:
        """The design curve of :attr:`target` at the analysis bands, dB.

        ``None`` when the chain declared no target. Comparable band by band
        with :attr:`received_level`.
        """
        return curve_of(self)

    @property
    def exceedance(self) -> np.ndarray | None:
        """How far the receiving room is over the criterion, band by band, dB.

        Positive where the curve is exceeded, ``None`` without a target. It is
        also the transmission-loss deficit of the partition, band for band.
        """
        return exceedance_of(self)

    @property
    def meets_target(self) -> bool | None:
        """``True`` when the receiving room stays under the curve everywhere.

        ``None`` without a target. The band-by-band test of a design sheet, not
        the two-step designation procedure behind :attr:`rating`.
        """
        return meets_target_of(self)

    # -- design inverse ---------------------------------------------------
    @property
    def required_transmission_loss(self) -> np.ndarray | None:
        r"""Partition ``TL`` that would just meet the criterion, per band, dB.

        The chain of Equation (4.101) solved for the transmission loss,

        .. math::

           \mathrm{TL}_\mathrm{req} = L_{p1} - L_{p2,\mathrm{target}}
           + 10 \log_{10}(S_\mathrm{w} / S_2 \alpha_2) + \text{penalty},

        with :math:`L_{p2,\mathrm{target}}` the design criterion curve. The
        ``tau S_w`` term is
        left out of the inverse (it depends on the answer), which is how the
        equation is used to specify a partition. ``None`` when no target was
        declared.
        """
        curve = self.criterion_curve
        if curve is None:
            return None
        return np.asarray(
            self.source_level
            - curve
            + 10.0 * np.log10(self.partition_area / self.receiving_absorption)
            + self.flanking_penalty,
            dtype=np.float64,
        )

    # -- tabular view -----------------------------------------------------
    def table(self) -> list[dict[str, Any]]:
        """The chain as a list of printed rows, source room to verdict.

        Each entry has ``label``, ``kind`` (one of ``"source_power"``,
        ``"source"``, ``"transmission_loss"``, ``"absorption"``,
        ``"noise_reduction"``, ``"flanking"``, ``"received"``, ``"criterion"``,
        ``"required"``) and ``values`` (a per-band array). The rows are the ones
        a hand calculation writes down, in that order.

        :return: The list of row dictionaries, in printing order.
        """
        n = self.frequencies.size
        rows: list[dict[str, Any]] = []
        if self.source_power_level is not None:
            rows.append(
                {
                    "label": "Source sound power level",
                    "kind": "source_power",
                    "values": self.source_power_level,
                }
            )
        rows.append(
            {
                "label": "Source room level",
                "kind": "source",
                "values": self.source_level,
            }
        )
        rows.append(
            {
                "label": "Partition transmission loss",
                "kind": "transmission_loss",
                "values": self.transmission_loss,
            }
        )
        rows.append(
            {
                "label": "Receiving-room absorption",
                "kind": "absorption",
                "values": self.receiving_absorption,
            }
        )
        if self.flanking_penalty:
            rows.append(
                {
                    "label": "Flanking penalty",
                    "kind": "flanking",
                    "values": np.full(n, -self.flanking_penalty),
                }
            )
        rows.append(
            {
                "label": "Noise reduction",
                "kind": "noise_reduction",
                "values": self.noise_reduction,
            }
        )
        rows.append(
            {
                "label": "Receiving room level",
                "kind": "received",
                "values": self.received_level,
            }
        )
        curve = self.criterion_curve
        required = self.required_transmission_loss
        if curve is not None and required is not None and self.target is not None:
            rows.append(
                {
                    "label": f"{self.criterion} {self.target:g}",
                    "kind": "criterion",
                    "values": curve,
                }
            )
            rows.append(
                {
                    "label": "Required transmission loss",
                    "kind": "required",
                    "values": required,
                }
            )
        return rows

    # -- presentation -----------------------------------------------------
    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the source-room level, the noise reduction and what arrives.

        The source-room and receiving-room spectra are drawn against the design
        criterion curve, with the band-by-band noise reduction on a twin axis.
        Requires matplotlib (``pip install phonometry[plot]``).

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :param kwargs: Forwarded to the received-level ``Axes.plot``.
        :return: The axes.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_room_to_room

        check_language(language)
        return plot_room_to_room(self, ax=ax, language=language, **kwargs)


def room_to_room_transmission(
    frequencies: ArrayLike,
    transmission_loss: ArrayLike,
    partition_area: float,
    receiving_absorption: ArrayLike,
    *,
    source: SourceRoom | None = None,
    include_partition_transmission: bool = False,
    criterion: DesignCriterion | None = None,
    label: str = "Room to room",
) -> RoomToRoomResult:
    r"""Sound transmission from one room to another (Norton 2e Equation (4.101)).

    Computes the noise reduction the partition and the receiving room deliver
    together, and the reverberant spectrum in the receiving room. The
    source-room level is either given directly as ``source.level`` or built
    from a sound power level and the source room's room constant, in which case
    the reverberant field alone is used (:func:`phonometry.room.steady_state_spl`
    at ``distance=None``), which is the level that drives the transmission
    across the partition.

    :param frequencies: Octave-band centre frequencies, Hz (1-D array).
    :param transmission_loss: Transmission loss of the partition ``TL``, dB;
        a scalar or one value per band. Measured, tabulated, or predicted by
        :mod:`phonometry.building.prediction.panel_transmission`.
    :param partition_area: Area of the partition between the rooms ``S_w``, m2.
    :param receiving_absorption: Equivalent absorption area of the receiving
        room ``S_2 alpha_2`` per band, m2; e.g. from
        :func:`phonometry.room.equivalent_absorption_area`.
    :param source: The source room (:class:`SourceRoom`): its level, or its
        sound power level with the room constant, directivity and sound power
        model that turn it into one. Exactly one of the two descriptions is
        required, so the empty default is rejected.
    :param include_partition_transmission: When ``True`` the ``tau S_w`` term of
        Equation (4.101) is added to the receiving-room absorption, with
        :math:`\tau = 10^{-\mathrm{TL}/10}`. Default ``False``, the form hand
        calculations
        use.
    :param criterion: The design criterion (:class:`DesignCriterion`): the
        family, the target curve and the flanking allowance. ``None`` is the
        default criterion, an ``"NC"`` family with no target.
    :param label: A short human label of the chain.
    :return: A :class:`RoomToRoomResult`.
    :raises ValueError: If the spectra do not share one value per band, if
        neither or both source descriptions are given, or if the criterion
        family or the sound power model is unknown.
    """
    from ..room.steady_field import SOURCE_POWER_MODELS, steady_state_spl

    source = SourceRoom() if source is None else source
    criterion = DesignCriterion() if criterion is None else criterion

    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        msg = "'frequencies' must be a non-empty 1-D array."
        raise ValueError(msg)
    n = f.size

    require_choice(source.model, "source.model", tuple(SOURCE_POWER_MODELS))
    lw: NDArray[np.float64] | None = None
    # SourceRoom.__post_init__ has already refused every other combination,
    # so these two branches are the only ones reachable.
    if source.power_level is not None and source.room_constant is not None:
        lw = as_band_spectrum(source.power_level, n, "source.power_level")
        lp1 = np.asarray(
            steady_state_spl(
                lw,
                None,
                as_band_spectrum(source.room_constant, n, "source.room_constant"),
                directivity=source.directivity,
                source_model=source.model,
            ),
            dtype=np.float64,
        )
    elif source.level is None:  # pragma: no cover - __post_init__ guards this
        msg = "'source.level' must be given."
        raise ValueError(msg)
    else:
        lp1 = as_band_spectrum(source.level, n, "source.level")

    tl = as_band_spectrum(transmission_loss, n, "transmission_loss")
    s_w = require_positive(partition_area, "partition_area")
    absorption = as_band_spectrum(receiving_absorption, n, "receiving_absorption")
    if np.any(absorption <= 0.0):
        msg = (
            "'receiving_absorption' must be positive: a receiving room with no "
            "absorption has no finite reverberant level."
        )
        raise ValueError(msg)
    penalty = require_non_negative(
        criterion.flanking_penalty, "criterion.flanking_penalty"
    )
    family, goal = validate_target(criterion.family, criterion.target)

    total_absorption = absorption
    if include_partition_transmission:
        total_absorption = absorption + s_w * 10.0 ** (-tl / 10.0)
    nr = tl - 10.0 * np.log10(s_w / total_absorption) - penalty
    return RoomToRoomResult(
        frequencies=f,
        source_level=lp1,
        transmission_loss=tl,
        partition_area=s_w,
        receiving_absorption=absorption,
        noise_reduction=nr,
        received_level=lp1 - nr,
        flanking_penalty=penalty,
        source_power_level=lw,
        criterion=family,
        target=goal,
        label=label,
    )
