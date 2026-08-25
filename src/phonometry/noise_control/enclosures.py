#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Insertion loss of a close or free-standing machine enclosure.

Wrapping a machine in a sealed enclosure reduces the radiated noise by the
transmission loss of its panels, *minus* a penalty for the reverberant build-up
inside the small, hard cavity. Bies, Hansen & Howard, *Engineering Noise
Control* 5th ed., §7.4.2 (Eqs. (7.103), (7.111)) write the net reduction as

.. math::

   \mathrm{IL} = R - C, \qquad
   C = 10 \log_{10}\!\left[0.3 +
   \frac{S_\mathrm{E} (1 - \alpha_\mathrm{i})}{S_\mathrm{i} \alpha_\mathrm{i}}\right],

where ``R`` is the field-incidence transmission loss of the enclosure panels,
``S_E`` the external surface area, ``S_i`` the internal surface area (including
the machine) and ``alpha_i`` the mean absorption of the enclosure interior. The
reverberant term is exactly ``S_E`` over the interior **room constant**
:math:`R_\mathrm{i} = S_\mathrm{i} \alpha_\mathrm{i} / (1 - \alpha_\mathrm{i})`
(:func:`phonometry.room.room_constant`), so

.. math::

   C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i}).

A hard interior (``alpha_i`` small) makes ``C`` large and wastes much of the
panel ``R``; lining the enclosure drives ``C`` toward its floor
:math:`10 \log_{10} 0.3 = -5.2` dB (a fully absorbing interior, where
:math:`\mathrm{IL} = R + 5.2`).
Bies terms this net reduction the enclosure *noise reduction*; it is the
insertion loss of the enclosure.

Norton & Karczub, *Fundamentals of Noise and Vibration Analysis for Engineers*
2nd ed., 4.10 (Equation (4.115)) derive the same design equation from the same
power balance but **without the ``0.3``**:

.. math::

   \mathrm{IL} = R - 10 \log_{10}(S_\mathrm{E} / R_\mathrm{i}),

so a well-lined enclosure keeps rising instead of levelling off at
:math:`R + 5.2`. The two agree to a few tenths of a decibel while
:math:`S_\mathrm{E} / R_\mathrm{i}` stays above about
unity (a hard or lightly lined interior) and diverge once the lining takes over,
where Bies' floor is the safer statement. ``model`` selects between them, and
the difference is the reason a published worked answer has to be reproduced with
the model its author used.

Turned around, the same equation answers the question a designer actually asks:
*given the noise reduction I need, what transmission loss must the panels have?*
That is :func:`enclosure_required_transmission_loss`, which returns the required
``R`` per band in the same :class:`EnclosureResult`.

**The panel transmission loss ``R`` is supplied by the caller** -- measured, or
predicted by a panel model -- as a per-band array, a callable of frequency, or
a panel prediction result (a :class:`phonometry.building.SoundReductionResult`
or :class:`phonometry.building.ApertureTransmissionResult`, matched structurally
so no dependency on ``building`` is introduced). This module never predicts
``R`` itself; it combines a given ``R`` with the interior absorption. The
interior room constant reuses :func:`phonometry.room.room_constant`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from .._internal.validation import (
    check_engine,
    require_choice,
    require_equal_shapes,
    require_positive,
    require_ranks,
    require_same_length,
)
from ..room.steady_field import room_constant

#: Interior build-up correction models. The value is the additive floor inside
#: the logarithm of ``C = 10 lg(floor + S_E / R_i)``: Bies, Hansen & Howard
#: (Equation (7.111)) carry ``0.3``, which caps the insertion loss of a fully
#: lined enclosure at ``R + 5.2 dB``; Norton & Karczub (Equation (4.115)) carry
#: none.
ENCLOSURE_MODELS: dict[str, float] = {"bies": 0.3, "norton": 0.0}

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from .._report.metadata import ReportMetadata


@dataclass(frozen=True)
class EnclosureResult:
    r"""Insertion loss of a machine enclosure over frequency (Bies §7.4.2).

    :ivar frequencies: Frequencies ``f``, Hz, or ``None`` if the panel ``R``
        was given as a bare per-band array with no frequency labels.
    :ivar panel_transmission_loss: The supplied panel transmission loss ``R``
        per band, dB.
    :ivar correction: The interior-build-up correction ``C`` per band, dB.
    :ivar insertion_loss: The net enclosure insertion loss
        :math:`\mathrm{IL} = R - C`, dB.
    :ivar external_area: External enclosure surface area ``S_E``, m2.
    :ivar internal_area: Internal surface area ``S_i``, m2.
    :ivar room_constant: Interior room constant ``R_i`` per band, m2.
    """

    frequencies: np.ndarray | None
    panel_transmission_loss: np.ndarray
    correction: np.ndarray
    insertion_loss: np.ndarray
    external_area: float
    internal_area: float
    room_constant: np.ndarray

    def __post_init__(self) -> None:
        """Reject a sheet whose columns do not all run over the same bands.

        The enclosure fiche prints one row per band -- nominal frequency, panel
        ``R``, correction ``C`` and net ``IL`` side by side, with the interior
        room constant ``R_i`` added in the verbose table -- and it sizes that
        table from ``insertion_loss`` alone. Of those columns only ``R_i``
        stays out of the embedded figure, which is what makes it the one that
        hides: the sheet draws this result's own plot, where ``R``, ``C`` and
        ``IL`` share a single x axis, so one of them off the ``IL`` length
        raises from matplotlib as the figure is drawn and no PDF is written at
        all. An ``R_i`` one entry *long* fills every printed row from a column
        that runs over other bands -- carry an extra band at the front and
        each row states the previous band's ``R_i`` -- while one entry *short*
        renders an ordinary-looking plain sheet and fails only when someone
        asks for the verbose one.

        :raises ValueError: if the per-band quantities disagree, or one of them
            carries an extra axis.
        """
        require_ranks(
            self,
            frequencies=1,
            panel_transmission_loss=1,
            correction=1,
            insertion_loss=1,
            room_constant=1,
        )
        require_same_length(
            self,
            "frequencies",
            "panel_transmission_loss",
            "correction",
            "insertion_loss",
            "room_constant",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the panel ``R``, correction ``C`` and net insertion loss.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_enclosure

        check_language(language)
        return plot_enclosure(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render a machine-enclosure insertion-loss fiche to ``path``.

        Writes a one-page enclosure-performance sheet: the method-basis line
        naming the Bies, Hansen & Howard insertion-loss model
        (Engineering Noise Control 5th ed., section 7.4.2), an optional metadata
        header (client, enclosed machine, test environment, instrumentation,
        climate, date), a per-band table (nominal frequency, the supplied panel
        transmission loss ``R``, the interior build-up correction ``C`` and the
        net insertion loss :math:`\mathrm{IL} = R - C`) beside the ``R``, ``C``
        and ``IL`` curves, the boxed mean insertion loss over the analysis
        bands with the external and internal surface areas, an optional verdict
        row against a declared minimum, and a method-basis strip stating
        :math:`\mathrm{IL} = R - C` with
        :math:`C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i})`.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the enclosed machine,
            ``test_room`` the test environment, ``instrumentation``,
            ``temperature``, ``relative_humidity``, ``pressure``, ``test_date``),
            the footer identity (``laboratory``, ``operator``, ``report_id``,
            ``notes``) and, via ``requirement``, a declared minimum mean
            insertion loss (more insertion loss is better). The surface areas
            come from the result itself.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the interior room
            constant ``R_i`` column.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from .._i18n import check_language

        check_language(language)
        check_engine(engine)
        from .._report.enclosure import render_enclosure_report

        return render_enclosure_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


class PanelTransmissionResult(Protocol):
    """A panel prediction result exposing a per-band transmission loss.

    A frozen result such as :class:`phonometry.building.SoundReductionResult`
    or :class:`phonometry.building.ApertureTransmissionResult` carries the
    predicted transmission loss in ``transmission_loss`` and its band centres
    in ``frequencies``. Matching structurally (a :class:`typing.Protocol`)
    lets :func:`enclosure_insertion_loss` accept such a result directly without
    importing the ``building`` package, so ``noise_control`` keeps no
    dependency on it.
    """

    transmission_loss: NDArray[np.float64]
    frequencies: NDArray[np.float64]


def _resolve_frequencies(
    frequencies: ArrayLike | None,
) -> NDArray[np.float64] | None:
    """Validate an optional 1-D positive frequency grid (Hz)."""
    if frequencies is None:
        return None
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if freqs.ndim != 1 or freqs.size == 0:
        msg = "'frequencies' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(freqs <= 0.0) or not np.all(np.isfinite(freqs)):
        msg = "'frequencies' must be positive and finite."
        raise ValueError(msg)
    return freqs


def _resolve_panel_r(
    values: ArrayLike | Callable[[NDArray[np.float64]], ArrayLike],
    freqs: NDArray[np.float64] | None,
    name: str = "panel_transmission_loss",
) -> NDArray[np.float64]:
    """Resolve a per-band decibel spectrum into a validated 1-D array."""
    if callable(values):
        if freqs is None:
            msg = f"'frequencies' is required when '{name}' is a callable."
            raise ValueError(msg)
        r = np.atleast_1d(np.asarray(values(freqs), dtype=np.float64))
    else:
        r = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if r.ndim != 1 or r.size == 0:
        msg = f"'{name}' must be a non-empty 1-D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(r)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    return r


def _resolve_interior(
    values: ArrayLike | Callable[[NDArray[np.float64]], ArrayLike],
    external_area: float,
    internal_area: float,
    internal_absorption: ArrayLike,
    frequencies: ArrayLike | None,
    model: str,
    name: str,
    owner: str,
) -> tuple[
    NDArray[np.float64] | None,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
]:
    """Shared geometry of the enclosure equation, in either direction.

    Validates the two areas, the interior absorption and the per-band spectrum
    ``values`` (a panel ``R`` for :func:`enclosure_insertion_loss`, a target
    ``IL`` for :func:`enclosure_required_transmission_loss`), and returns
    ``(frequencies, values, correction C, room constant R_i, S_E, S_i)``.

    *name* is the parameter the spectrum arrived under and *owner* the entry
    point the caller typed, so a band count that disagrees is reported in the
    caller's own vocabulary rather than in this helper's.
    """
    s_e = require_positive(external_area, "external_area")
    s_i = require_positive(internal_area, "internal_area")
    floor = ENCLOSURE_MODELS[require_choice(model, "model", tuple(ENCLOSURE_MODELS))]

    freqs = _resolve_frequencies(frequencies)
    v = _resolve_panel_r(values, freqs, name)

    alpha = np.asarray(internal_absorption, dtype=np.float64)
    if alpha.ndim > 1 or alpha.size == 0:
        msg = "'internal_absorption' must be a scalar or a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(alpha <= 0.0) or np.any(alpha >= 1.0) or not np.all(np.isfinite(alpha)):
        msg = "'internal_absorption' must lie strictly in (0, 1)."
        raise ValueError(msg)

    r_i = np.atleast_1d(np.asarray(room_constant(s_i, alpha), dtype=np.float64))
    try:
        r_i_b, v_b = np.broadcast_arrays(r_i, v)
    except ValueError:
        # numpy reports the two shapes it could not reconcile without saying
        # which argument carried which; name the culprit in the caller's words.
        msg = (
            f"'internal_absorption' must be a scalar or carry one value per "
            f"band ({v.size} in '{name}'); got shape {alpha.shape}."
        )
        raise ValueError(msg) from None
    if freqs is not None:
        # The band count is whatever the spectrum and the absorption broadcast
        # to, so either of them can be the one that disagrees with the labels.
        require_equal_shapes(
            owner,
            {
                "frequencies": freqs.shape,
                f"{name} / internal_absorption": v_b.shape,
            },
            "band",
        )
    correction = 10.0 * np.log10(floor + s_e / r_i_b)
    return (
        freqs,
        np.array(v_b, dtype=np.float64),
        np.array(correction, dtype=np.float64),
        np.array(r_i_b, dtype=np.float64),
        s_e,
        s_i,
    )


def enclosure_insertion_loss(
    panel_transmission_loss: (
        ArrayLike | Callable[[NDArray[np.float64]], ArrayLike] | PanelTransmissionResult
    ),
    external_area: float,
    internal_area: float,
    internal_absorption: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    model: str = "bies",
) -> EnclosureResult:
    r"""Net insertion loss of a machine enclosure (Bies Eqs. (7.103), (7.111)).

    :math:`\mathrm{IL} = R - C` with
    :math:`C = 10 \log_{10}(0.3 + S_\mathrm{E} / R_\mathrm{i})` and the interior room
    constant :math:`R_\mathrm{i} = S_\mathrm{i} \alpha_\mathrm{i} / (1 - \alpha_\mathrm{i})`.

    :param panel_transmission_loss: Panel transmission loss ``R`` per band, dB.
        One of: a per-band array (measured); a callable mapping a frequency
        array to per-band ``R`` (then ``frequencies`` is required); or a panel
        prediction result carrying ``transmission_loss`` and ``frequencies``,
        such as the :class:`~phonometry.building.SoundReductionResult` of
        :func:`phonometry.building.single_panel_transmission_loss` /
        :func:`phonometry.building.double_wall_transmission_loss` or the
        :class:`~phonometry.building.ApertureTransmissionResult` of
        :func:`phonometry.building.composite_transmission_loss` (its
        ``frequencies`` are then used unless *frequencies* is given). This
        function does not predict ``R`` itself.
    :param external_area: External enclosure surface area ``S_E``, m2.
    :param internal_area: Internal surface area ``S_i`` (including the machine),
        m2.
    :param internal_absorption: Mean interior absorption ``alpha_i`` in
        ``(0, 1)`` (scalar or per-band).
    :param frequencies: Band centre frequencies, Hz; required when
        ``panel_transmission_loss`` is a callable, optional otherwise (used to
        label the result and the plot).
    :param model: Interior build-up model, one of :data:`ENCLOSURE_MODELS`:
        ``"bies"`` (default) carries the ``0.3`` floor of Bies Equation (7.111),
        ``"norton"`` the bare :math:`C = 10 \log_{10}(S_\mathrm{E} / R_\mathrm{i})` of Norton & Karczub
        Equation (4.115).
    :return: An :class:`EnclosureResult`.
    """
    if not callable(panel_transmission_loss) and not isinstance(
        panel_transmission_loss, (np.ndarray, list, tuple)
    ):
        if hasattr(panel_transmission_loss, "transmission_loss") and hasattr(
            panel_transmission_loss, "frequencies"
        ):
            result = cast("PanelTransmissionResult", panel_transmission_loss)
            if frequencies is None:
                frequencies = result.frequencies
            panel_transmission_loss = np.asarray(
                result.transmission_loss, dtype=np.float64
            )
        elif not isinstance(panel_transmission_loss, (int, float)) and not hasattr(
            panel_transmission_loss, "__array__"
        ):
            # Anything else would fall through to numpy's float() coercion,
            # whose message names neither the parameter nor the accepted forms.
            msg = (
                "'panel_transmission_loss' must be a per-band array, a callable "
                "of the frequency array, or a panel result carrying "
                "'transmission_loss' and 'frequencies'; got "
                f"{type(panel_transmission_loss).__name__}."
            )
            raise TypeError(msg)

    freqs, r_b, correction, r_i_b, s_e, s_i = _resolve_interior(
        panel_transmission_loss,
        external_area,
        internal_area,
        internal_absorption,
        frequencies,
        model,
        name="panel_transmission_loss",
        owner="enclosure_insertion_loss",
    )
    return EnclosureResult(
        frequencies=freqs,
        panel_transmission_loss=r_b,
        correction=correction,
        insertion_loss=r_b - correction,
        external_area=s_e,
        internal_area=s_i,
        room_constant=r_i_b,
    )


def enclosure_required_transmission_loss(
    insertion_loss: ArrayLike | Callable[[NDArray[np.float64]], ArrayLike],
    external_area: float,
    internal_area: float,
    internal_absorption: ArrayLike,
    *,
    frequencies: ArrayLike | None = None,
    model: str = "bies",
) -> EnclosureResult:
    r"""Panel ``R`` an enclosure needs to deliver a given insertion loss.

    The design equation of :func:`enclosure_insertion_loss` solved the other
    way, :math:`R = \mathrm{IL} + C`: the enclosure geometry and its interior
    lining fix the
    build-up correction ``C``, and the panels have to make up the rest. This is
    the number an enclosure is specified from, and the form Norton & Karczub use
    in 4.10 when the target ``IL`` is the gap between the level a machine
    produces in the room and a noise-criterion curve.

    :param insertion_loss: Required insertion loss ``IL`` per band, dB (a
        per-band array, or a callable of the frequency array, in which case
        ``frequencies`` is required).
    :param external_area: External enclosure surface area ``S_E``, m2.
    :param internal_area: Internal surface area ``S_i``, m2, including the
        surface of the enclosed machine.
    :param internal_absorption: Mean interior absorption ``alpha_i`` in
        ``(0, 1)`` (scalar or per-band); e.g. from
        :func:`phonometry.room.mean_absorption` over the lined enclosure walls
        and the machine surface.
    :param frequencies: Band centre frequencies, Hz; required for a callable
        ``insertion_loss``, optional otherwise.
    :param model: Interior build-up model, one of :data:`ENCLOSURE_MODELS`
        (see :func:`enclosure_insertion_loss`).
    :return: An :class:`EnclosureResult` whose ``panel_transmission_loss`` is
        the **required** ``R`` and whose ``insertion_loss`` is the requested
        target, so it plots and reports like a forward calculation.
    """
    freqs, il_b, correction, r_i_b, s_e, s_i = _resolve_interior(
        insertion_loss,
        external_area,
        internal_area,
        internal_absorption,
        frequencies,
        model,
        name="insertion_loss",
        owner="enclosure_required_transmission_loss",
    )
    return EnclosureResult(
        frequencies=freqs,
        panel_transmission_loss=il_b + correction,
        correction=correction,
        insertion_loss=il_b,
        external_area=s_e,
        internal_area=s_i,
        room_constant=r_i_b,
    )
