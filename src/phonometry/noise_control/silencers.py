#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Reactive silencers by the four-pole (transmission-matrix) method.

A reactive silencer controls noise by *reflecting* it back to the source with
impedance discontinuities -- sudden area changes and side branches -- rather
than by dissipating it in absorptive material. The one-dimensional plane-wave
theory represents each acoustic element by a 2x2 **transfer (four-pole)
matrix** relating the sound pressure ``p`` and volume velocity ``S u`` at its
two ends, and a compound silencer is the ordered matrix product of its
elements (Bies, Hansen & Howard, *Engineering Noise Control* 5th ed., §8.8-8.9;
Munjal, *Acoustics of Ducts and Mufflers*).

**Transfer matrix** (Bies Eq. (8.133)), state vector ``[p, S u]`` with the
characteristic acoustic impedance :math:`Z = \rho c / S`. The plane-wave
element for
a straight duct of length ``L`` and area ``S`` is (Bies Eq. (8.143), no flow)

.. math::

   \begin{bmatrix}
   \cos(kL) & j (\rho c / S) \sin(kL) \\
   j (S / \rho c) \sin(kL) & \cos(kL)
   \end{bmatrix},
   \qquad k = \omega / c,

and a **side branch** of acoustic impedance ``Z_b`` is the shunt element
(Bies Eq. (8.144))

.. math::

   \begin{bmatrix}
   1 & 0 \\
   1 / Z_\mathrm{b} & 1
   \end{bmatrix}.

**Transmission loss** from the compound matrix ``T`` (Munjal, *Acoustics of
Ducts and Mufflers* 2nd ed., Eq. (3.27), no flow; reduces to Bies Eq. (8.148)
for equal inlet/outlet areas):

.. math::

   \mathrm{TL} = 10 \log_{10}\!\left[\frac{Z_n}{Z_1} \cdot \frac{1}{4}
   \left\lvert T_{11} + \frac{T_{12}}{Z_n} + Z_1 T_{21}
   + \frac{Z_1}{Z_n} T_{22} \right\rvert^2\right]

with :math:`Z_1 = \rho c / S_{\mathrm{in}}` and
:math:`Z_n = \rho c / S_{\mathrm{out}}`. A zero-length element
between unequal areas then reproduces the classic sudden-expansion result
:math:`\mathrm{TL} = 10 \log_{10}[(1 + m)^2 / (4 m)]` with
:math:`m = S_{\mathrm{out}} / S_{\mathrm{in}}`, and the TL is
the same from either side, as reciprocity of a passive two-port requires.
Bies Eq. (8.141) prints this formula with impedance ratios on ``T11`` and
``T22`` (:math:`Z_{A1}/Z_{An}` and :math:`Z_{An}/Z_{A1}`) instead of the
overall :math:`Z_n/Z_1`
prefactor; as printed it fails the sudden-expansion limit (see
``docs/ERRATA.md``). ``TL`` is the intrinsic attenuation for an anechoic
termination. The **insertion loss** for a source of internal impedance
``Z_s`` radiating into a termination impedance ``Z_r`` is the extra
attenuation of inserting the silencer in place of a direct connection,

.. math::

   \mathrm{IL} = 20 \log_{10}
   \left\lvert \frac{T_{11} Z_\mathrm{r} + T_{12} + Z_\mathrm{s} Z_\mathrm{r} T_{21} + Z_\mathrm{s} T_{22}}
   {Z_\mathrm{s} + Z_\mathrm{r}} \right\rvert,

which is ``0`` when the silencer reduces to a through connection
(:math:`T = I`)
and, for equal inlet/outlet areas, equals the transmission loss for the
anechoic reference :math:`Z_\mathrm{s} = Z_\mathrm{r} = \rho c / S` (with unequal areas the
direct
connection contains the same area jump, so its mismatch loss cancels from
the insertion loss but not from the transmission loss).

**Simple expansion chamber.** A chamber of area ``S_exp`` and length ``L``
between pipes of area ``S_duct`` has the closed-form transmission loss (Bies
Eq. (8.111)) with area ratio :math:`m = S_{\mathrm{exp}} / S_{\mathrm{duct}}`

.. math::

   \mathrm{TL} = 10 \log_{10}\!\left[1 + \frac{1}{4}
   \left(m - \frac{1}{m}\right)^2 \sin^2(kL)\right],

peaking at :math:`10 \log_{10}[1 + (1/4)(m - 1/m)^2]` when
:math:`kL = \pi/2, 3\pi/2, \ldots` and
dropping to ``0`` at :math:`kL = n \pi` (no dissipation). The four-pole product
reproduces this exactly, and the machinery extends to side-branch (Helmholtz,
quarter-wave) and extended-tube resonators that the closed form cannot cover.

**Layouts of your own.** Anything the four named devices do not cover is built
by cascading elements directly. :class:`SilencerChain` does that through the
same :func:`duct_matrix`, :func:`shunt_matrix` and :func:`cascade` calls while
keeping the arguments each element was given, which is what lets a hand-built
chain be drawn (:meth:`SilencerChain.plot_geometry`) and not only computed. The
drawing shows the ducts to scale and marks the branch points, because that is
the whole of what the elements declare: a duct element is handed a length and
an area, a shunt element only an impedance.

**Validity.** All of this is one-dimensional: it holds while the duct and the
chamber carry plane waves only, that is below the first higher-order-mode
cut-on frequency of the widest cross section
(:mod:`phonometry.noise_control.duct_modes`). Every result reports that
frequency as :attr:`ReactiveSilencerResult.plane_wave_limit` and raises a
:class:`~phonometry.noise_control.duct_modes.PlaneWaveWarning` when the
analysis grid reaches past it: the numbers are still returned, but above cut-on
they describe the plane-wave mode alone and a measurement will show the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._internal.validation import (
    check_engine,
    require_non_negative,
    require_positive,
    require_ranks,
    require_same_length,
)
from .duct_modes import plane_wave_limit as _plane_wave_limit
from .duct_modes import warn_above_plane_wave_limit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference air properties at 20 degC, 101.325 kPa.
_C_AIR = 343.0
_RHO_AIR = 1.206
#: Relative slack allowed when two extended tubes meet inside a chamber. Their
#: sum has to clear the chamber length by more than this fraction of it to count
#: as an overlap, which separates a geometry the user did not intend from the
#: last bits of a sum like 0.1 + 0.2. Nine orders of magnitude below the length
#: is far above the arithmetic and far below any dimension anyone draws.
_MEETING_SLACK = 1e-9
#: Smallest frequency grid on which an interior minimum of |Z_b| can exist:
#: two endpoints plus at least one interior candidate, since a least value
#: sitting on either end is a property of the grid rather than of the branch.
_MIN_GRID_FOR_INTERIOR_MINIMUM = 3

_Complex = NDArray[np.complex128]


def _frequencies(frequencies: ArrayLike) -> NDArray[np.float64]:
    """Validate a strictly positive, finite 1-D frequency grid (Hz)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        msg = "'frequencies' must be a non-empty 1-D array."
        raise ValueError(msg)
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        msg = "'frequencies' must be positive and finite."
        raise ValueError(msg)
    return f


def duct_matrix(
    frequencies: ArrayLike,
    length: float,
    area: float,
    *,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
) -> _Complex:
    """Four-pole matrix of a straight duct (Bies Eq. (8.143), no flow).

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param length: Duct length ``L``, m.
    :param area: Cross-sectional area ``S``, m2.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :return: A ``(n_freq, 2, 2)`` complex transfer-matrix array.
    """
    f = _frequencies(frequencies)
    length = require_non_negative(length, "length")
    area = require_positive(area, "area")
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    k = 2.0 * np.pi * f / c
    z = rho * c / area
    cos = np.cos(k * length)
    sin = np.sin(k * length)
    t = np.empty((f.size, 2, 2), dtype=np.complex128)
    t[:, 0, 0] = cos
    t[:, 0, 1] = 1j * z * sin
    t[:, 1, 0] = 1j * sin / z
    t[:, 1, 1] = cos
    return t


def shunt_matrix(branch_impedance: ArrayLike) -> _Complex:
    """Four-pole matrix of a side branch of impedance ``Z_b`` (Bies Eq. (8.144)).

    :param branch_impedance: Acoustic impedance ``Z_b`` of the branch,
        Pa s/m3 (1-D complex array over frequency).
    :return: A ``(n_freq, 2, 2)`` complex transfer-matrix array.
    """
    zb = np.atleast_1d(np.asarray(branch_impedance, dtype=np.complex128))
    if zb.ndim != 1 or zb.size == 0:
        msg = "'branch_impedance' must be a non-empty 1-D array."
        raise ValueError(msg)
    t = np.zeros((zb.size, 2, 2), dtype=np.complex128)
    t[:, 0, 0] = 1.0
    # A lossless branch at exact resonance has Z_b -> 0 (it shorts the duct);
    # 1/Z_b -> infinity gives the ideal infinite attenuation there.
    with np.errstate(divide="ignore", invalid="ignore"):
        t[:, 1, 0] = 1.0 / zb
    t[:, 1, 1] = 1.0
    return t


def _shorting_frequency(
    f: NDArray[np.float64], branch_impedance: _Complex
) -> float | None:
    """Grid frequency of least ``|Z_b|``, when it falls inside the grid.

    The one thing a bare shunt impedance says about itself: where on the
    analysis grid the branch comes closest to shorting the duct, which is
    where it takes the most out of the line. A least value sitting on either
    end of the grid is a property of the grid rather than of the branch, and
    is not reported.
    """
    if f.size < _MIN_GRID_FOR_INTERIOR_MINIMUM:
        return None
    magnitude = np.abs(branch_impedance)
    magnitude = np.where(np.isfinite(magnitude), magnitude, np.inf)
    if not bool(np.any(np.isfinite(magnitude))):
        return None
    index = int(np.argmin(magnitude))
    if index in (0, f.size - 1):
        return None
    return float(f[index])


def cascade(*matrices: _Complex) -> _Complex:
    """Cascade element four-pole matrices from inlet to outlet.

    The compound matrix is the ordered product ``T1 @ T2 @ ... @ Tn`` (the
    state at the inlet equals the compound matrix times the state at the
    outlet), broadcast over the frequency axis.

    :param matrices: One or more ``(n_freq, 2, 2)`` arrays sharing ``n_freq``.
    :return: The compound ``(n_freq, 2, 2)`` array.
    """
    if not matrices:
        msg = "cascade() needs at least one matrix."
        raise ValueError(msg)
    n = matrices[0].shape[0]
    if any(m.shape[0] != n for m in matrices[1:]):
        msg = "cascade() matrices must share the same frequency grid (n_freq)."
        raise ValueError(msg)
    total = matrices[0]
    for m in matrices[1:]:
        total = np.matmul(total, m)
    return total


def transmission_loss(
    transfer_matrix: _Complex,
    *,
    inlet_area: float,
    outlet_area: float,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
) -> NDArray[np.float64]:
    r"""Transmission loss of a four-pole element (Munjal Eq. (3.27), no flow).

    .. math::

       \mathrm{TL} = 10 \log_{10}\!\left[(Z_n/Z_1) \frac{1}{4}
       \lvert T_{11} + T_{12}/Z_n + Z_1 T_{21}
       + (Z_1/Z_n) T_{22} \rvert^2\right]

    with :math:`Z_1 = \rho c / S_{\mathrm{in}}` and
    :math:`Z_n = \rho c / S_{\mathrm{out}}` (Munjal, *Acoustics
    of Ducts and Mufflers* 2nd ed., Eq. (3.27)). Do not "restore" the Bies
    Eq. (8.141) weighting: as printed there the equation fails the
    sudden-expansion limit for unequal port areas (see ``docs/ERRATA.md``).

    :param transfer_matrix: A ``(n_freq, 2, 2)`` compound matrix.
    :param inlet_area: Inlet pipe area ``S_in``, m2.
    :param outlet_area: Outlet pipe area ``S_out``, m2.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :return: The transmission loss per frequency, dB.
    """
    s_in = require_positive(inlet_area, "inlet_area")
    s_out = require_positive(outlet_area, "outlet_area")
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    z1 = rho * c / s_in
    zn = rho * c / s_out
    t11 = transfer_matrix[:, 0, 0]
    t12 = transfer_matrix[:, 0, 1]
    t21 = transfer_matrix[:, 1, 0]
    t22 = transfer_matrix[:, 1, 1]
    term = t11 + t12 / zn + z1 * t21 + (z1 / zn) * t22
    return np.asarray(
        10.0 * np.log10((zn / z1) * 0.25 * np.abs(term) ** 2),
        dtype=np.float64,
    )


def insertion_loss(
    transfer_matrix: _Complex,
    *,
    source_impedance: ArrayLike,
    radiation_impedance: ArrayLike,
) -> NDArray[np.float64]:
    r"""Insertion loss of a four-pole element for given end impedances.

    The attenuation from inserting the element in place of a direct (zero
    length) connection between a source of internal impedance ``Z_s`` and a
    radiation (termination) impedance ``Z_r``:

    .. math::

       \mathrm{IL} = 20 \log_{10}
       \left\lvert \frac{T_{11} Z_\mathrm{r} + T_{12} + Z_\mathrm{s} Z_\mathrm{r} T_{21} + Z_\mathrm{s} T_{22}}
       {Z_\mathrm{s} + Z_\mathrm{r}} \right\rvert.

    :param transfer_matrix: A ``(n_freq, 2, 2)`` compound matrix.
    :param source_impedance: Source internal acoustic impedance ``Z_s``,
        Pa s/m3 (scalar or per-frequency, real or complex).
    :param radiation_impedance: Termination/radiation acoustic impedance
        ``Z_r``, Pa s/m3 (scalar or per-frequency).
    :return: The insertion loss per frequency, dB.
    """
    n = transfer_matrix.shape[0]
    zs = np.broadcast_to(np.asarray(source_impedance, dtype=np.complex128), (n,))
    zr = np.broadcast_to(np.asarray(radiation_impedance, dtype=np.complex128), (n,))
    t11 = transfer_matrix[:, 0, 0]
    t12 = transfer_matrix[:, 0, 1]
    t21 = transfer_matrix[:, 1, 0]
    t22 = transfer_matrix[:, 1, 1]
    num = t11 * zr + t12 + zs * zr * t21 + zs * t22
    return np.asarray(20.0 * np.log10(np.abs(num / (zs + zr))), dtype=np.float64)


def helmholtz_impedance(
    frequencies: ArrayLike,
    neck_area: float,
    neck_length: float,
    cavity_volume: float,
    *,
    resistance: float = 0.0,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
) -> _Complex:
    r"""Acoustic impedance of a Helmholtz side branch (Bies Eq. (8.152)).

    :math:`Z = R + j(\rho \omega l_\mathrm{e} / S_{\mathrm{neck}} -
    \rho c^2 / (\omega V))` with acoustic
    mass :math:`\rho l_\mathrm{e} / S_{\mathrm{neck}}` and compliance
    :math:`V / (\rho c^2)`; the resonance
    :math:`f_0 = (c / 2 \pi) \sqrt{S_{\mathrm{neck}} / (l_\mathrm{e} V)}`
    (Bies Eq. (8.46)) is where the
    reactance vanishes, leaving :math:`Z = R`: a lossless branch
    (``resistance = 0``) shorts the duct there, and a resistive one presents
    its resistance instead, which is what bounds the peak attenuation.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param neck_area: Neck cross-sectional area ``S_neck``, m2.
    :param neck_length: Effective neck length ``l_e`` (with end corrections), m.
    :param cavity_volume: Cavity volume ``V``, m3.
    :param resistance: Acoustic resistance ``R``, Pa s/m3 (default 0, lossless).
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :return: The complex branch impedance per frequency, Pa s/m3.
    """
    f = _frequencies(frequencies)
    s_neck = require_positive(neck_area, "neck_area")
    le = require_positive(neck_length, "neck_length")
    vol = require_positive(cavity_volume, "cavity_volume")
    r = require_non_negative(resistance, "resistance")
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    omega = 2.0 * np.pi * f
    reactance = rho * omega * le / s_neck - rho * c**2 / (omega * vol)
    return np.asarray(r + 1j * reactance, dtype=np.complex128)


def quarter_wave_impedance(
    frequencies: ArrayLike,
    length: float,
    area: float,
    *,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
) -> _Complex:
    r"""Acoustic impedance of a closed quarter-wave side branch (Bies Eq. (8.146)).

    :math:`Z = -j (\rho c / S) \cot(k l_\mathrm{e})`; the reactance vanishes at
    :math:`l_\mathrm{e} = \lambda / 4` (:math:`f = c / 4 l_\mathrm{e}`), where the closed tube
    presents a
    pressure node and shorts the duct.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param length: Effective tube length ``l_e`` (with end correction), m.
    :param area: Tube cross-sectional area ``S``, m2.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :return: The complex branch impedance per frequency, Pa s/m3.
    """
    f = _frequencies(frequencies)
    le = require_positive(length, "length")
    area = require_positive(area, "area")
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    k = 2.0 * np.pi * f / c
    z = rho * c / area
    # At the half-wave frequencies (k l_e = n pi) the closed tube is transparent
    # (Z -> infinity); the divide-by-zero is expected and its shunt 1/Z is 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        zb = -1j * z / np.tan(k * le)
    return np.asarray(zb, dtype=np.complex128)


@dataclass(frozen=True)
class ReactiveSilencerResult:
    """Transmission and insertion loss of a reactive silencer over frequency.

    :ivar frequencies: Frequencies ``f``, Hz.
    :ivar transmission_loss: Transmission loss per frequency, dB.
    :ivar insertion_loss: Insertion loss per frequency, dB, or ``None`` when no
        source/radiation impedance was supplied.
    :ivar transfer_matrix: The compound ``(n_freq, 2, 2)`` four-pole matrix.
    :ivar kind: A short label of the device (e.g. ``"expansion chamber"``).
    :ivar resonances: Notable resonance frequencies, Hz (e.g. the resonator
        tuning frequency), or ``None``.
    :ivar geometry: The defining geometry the constructor was called with
        (keys matching its keyword names, e.g. ``length``/``chamber_area``/
        ``pipe_area`` for a chamber), retained so :meth:`plot_geometry` can
        draw the device; appended after the original fields and ``None`` for
        hand-built results that were not assembled by a :class:`SilencerChain`.
    :ivar plane_wave_limit: The first higher-order-mode cut-on frequency of the
        widest cross section of the device, Hz (Norton & Karczub Eq. 7.6,
        :func:`phonometry.noise_control.duct_modes.plane_wave_limit`). The
        four-pole algebra of this module is one-dimensional and is valid below
        it; above it several modes propagate at once and the result describes
        the plane-wave mode only, which is why a
        :class:`~phonometry.noise_control.duct_modes.PlaneWaveWarning` is
        raised when the analysis reaches past it. ``None`` for hand-built
        results that do not retain their geometry.
    :ivar chain: The :class:`SilencerChain` that assembled this result, for a
        result built by :meth:`SilencerChain.result`, and ``None`` otherwise.
        A chain is a list of four-pole elements rather than a named device, so
        it carries its geometry element by element instead of in ``geometry``;
        :meth:`plot_geometry` draws whichever of the two is present.
    """

    frequencies: np.ndarray
    transmission_loss: np.ndarray
    insertion_loss: np.ndarray | None
    transfer_matrix: np.ndarray
    kind: str
    resonances: np.ndarray | None = None
    geometry: dict[str, float] | None = None
    plane_wave_limit: float | None = None
    chain: SilencerChain | None = None

    def __post_init__(self) -> None:
        """Reject a device whose curves do not all run over the same grid.

        :meth:`plot` draws :attr:`transmission_loss` and
        :attr:`insertion_loss` against :attr:`frequencies`, and the fiche of
        :meth:`report` embeds that same figure, so a curve of its own length
        is already loud in both directions: matplotlib refuses two axes of
        different length, and one value short is out of range for the table
        the fiche builds row by row before the figure is even drawn. What
        neither failure does is name the field, and neither happens until
        something is rendered; refusing here names the odd curve where the
        result was assembled.

        :attr:`transfer_matrix` is the quiet one, and the reason this belongs
        at construction rather than in a reader: nothing in the library opens
        it again once the result is built, so a stack covering half the grid
        travels through :meth:`plot` and an entirely ordinary PDF without a
        word. It is pinned to three axes because two of them are its own: the
        four-pole matrix is 2x2 at every frequency, and a stack that lost the
        frequency axis would still count two rows on its first one and pass
        any check that only counted.

        :attr:`resonances` is left out of the grid: it counts the notable
        frequencies of the device -- one tuning frequency for a Helmholtz
        resonator, the odd quarter-wave multiples that fall inside the range
        for a branch tube -- and has nothing to do with how finely the
        response was sampled.

        :raises ValueError: if the curves disagree, or the matrix is not a
            stack of four-poles over frequency.
        """
        require_ranks(
            self,
            frequencies=1,
            transmission_loss=1,
            insertion_loss=1,
            transfer_matrix=3,
            resonances=1,
        )
        require_same_length(
            self,
            "frequencies",
            "transmission_loss",
            "insertion_loss",
            "transfer_matrix",
            axis="frequency",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the transmission (and insertion) loss against frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_reactive_silencer

        check_language(language)
        return plot_reactive_silencer(self, ax=ax, language=language, **kwargs)

    def plot_geometry(self, ax: Axes | None = None, *, language: str = "en") -> Axes:
        """Draw the silencer cross-section to scale (dimensioned side cut).

        A named device is drawn from its ``geometry``; a result assembled by a
        :class:`SilencerChain` is drawn from that ``chain``, duct by duct.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :return: The axes.
        :raises ValueError: If the result retains neither its ``geometry`` nor
            the ``chain`` that built it.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_silencer_result_geometry

        check_language(language)
        return plot_silencer_result_geometry(self, ax=ax, language=language)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a reactive-silencer transmission-loss fiche to ``path``.

        Writes a one-page silencer-performance sheet: the method-basis line
        naming the plane-wave four-pole (transfer-matrix) method (Munjal,
        Acoustics of Ducts and Mufflers 2nd ed., Eq. (3.27); Bies, Hansen &
        Howard, Engineering Noise Control 5th ed., sections 8.8-8.9), an
        optional metadata header (client, device, test environment,
        instrumentation, climate, date), a per-band table (nominal frequency,
        the transmission loss ``TL`` and, when computed, the insertion loss
        ``IL``) beside the ``TL`` (and ``IL``) curves, the boxed mean
        transmission loss over the analysis bands with the peak transmission
        loss and the device kind, an optional verdict row against a declared
        minimum, and a method-basis strip stating the four-pole
        transmission-loss relation.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the device, ``test_room`` the
            test environment, ``instrumentation``, ``temperature``,
            ``relative_humidity``, ``pressure``, ``test_date``), the footer
            identity (``laboratory``, ``operator``, ``report_id``, ``notes``)
            and, via ``requirement``, a declared minimum mean transmission loss
            (more transmission loss is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for signature symmetry with the other fiches;
            the silencer table already shows the insertion loss when it was
            computed.
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
        from .._report.silencer import render_reactive_silencer_report

        return render_reactive_silencer_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _result(
    f: NDArray[np.float64],
    t: _Complex,
    *,
    inlet_area: float,
    outlet_area: float,
    c: float,
    rho: float,
    source_impedance: ArrayLike | None,
    radiation_impedance: ArrayLike | None,
    kind: str,
    resonances: NDArray[np.float64] | None = None,
    geometry: dict[str, float] | None = None,
    areas: Sequence[float] | None = None,
    chain: SilencerChain | None = None,
) -> ReactiveSilencerResult:
    """Assemble a :class:`ReactiveSilencerResult` from a compound matrix.

    Also determines the frequency at which the plane-wave assumption behind the
    whole four-pole formulation expires: the first higher-order-mode cut-on of
    the widest cross section the device presents (the chamber of an expansion
    silencer, the duct of a side-branch resonator), and warns when the analysis
    grid reaches past it. A named device declares those cross sections in its
    ``geometry``; a :class:`SilencerChain` has one per element and passes them
    in ``areas``.
    """
    limit: float | None = None
    if areas is None and geometry is not None:
        areas = [
            value
            for key, value in geometry.items()
            if key.endswith("_area") and value > 0.0
        ]
    if areas:
        limit = _plane_wave_limit(area=max(areas), speed_of_sound=c)
        warn_above_plane_wave_limit(f, limit, kind, stacklevel=4)
    tl = transmission_loss(
        t,
        inlet_area=inlet_area,
        outlet_area=outlet_area,
        speed_of_sound=c,
        density=rho,
    )
    il: NDArray[np.float64] | None = None
    if source_impedance is not None and radiation_impedance is not None:
        il = insertion_loss(
            t,
            source_impedance=source_impedance,
            radiation_impedance=radiation_impedance,
        )
    return ReactiveSilencerResult(
        frequencies=f,
        transmission_loss=tl,
        insertion_loss=il,
        transfer_matrix=t,
        kind=kind,
        resonances=resonances,
        geometry=geometry,
        plane_wave_limit=limit,
        chain=chain,
    )


def expansion_chamber(
    frequencies: ArrayLike,
    length: float,
    chamber_area: float,
    pipe_area: float,
    *,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult:
    r"""Simple expansion-chamber silencer (Bies Eq. (8.111) / four-pole).

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param length: Chamber length ``L``, m.
    :param chamber_area: Chamber cross-sectional area ``S_exp``, m2.
    :param pipe_area: Inlet/outlet pipe area ``S_duct``, m2.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :param source_impedance: Optional source impedance ``Z_s`` for the
        insertion loss, Pa s/m3.
    :param radiation_impedance: Optional radiation impedance ``Z_r`` for the
        insertion loss, Pa s/m3.
    :return: A :class:`ReactiveSilencerResult` (its ``transmission_loss``
        equals the closed form
        :math:`10 \log_{10}[1 + (1/4)(m - 1/m)^2 \sin^2(kL)]`).
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    require_positive(chamber_area, "chamber_area")
    require_positive(pipe_area, "pipe_area")
    t = duct_matrix(f, length, chamber_area, speed_of_sound=c, density=rho)
    return _result(
        f,
        t,
        inlet_area=pipe_area,
        outlet_area=pipe_area,
        c=c,
        rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="expansion chamber",
        geometry={
            "length": float(length),
            "chamber_area": float(chamber_area),
            "pipe_area": float(pipe_area),
        },
    )


def helmholtz_resonator(
    frequencies: ArrayLike,
    duct_area: float,
    neck_area: float,
    neck_length: float,
    cavity_volume: float,
    *,
    resistance: float = 0.0,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult:
    r"""Side-branch Helmholtz resonator on a duct (Bies Eqs. (8.144), (8.152)).

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param duct_area: Main-duct cross-sectional area ``S_d``, m2.
    :param neck_area: Resonator neck area ``S_neck``, m2.
    :param neck_length: Effective neck length ``l_e``, m.
    :param cavity_volume: Cavity volume ``V``, m3.
    :param resistance: Neck acoustic resistance ``R``, Pa s/m3 (default 0).
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :param source_impedance: Optional source impedance ``Z_s``, Pa s/m3.
    :param radiation_impedance: Optional radiation impedance ``Z_r``, Pa s/m3.
    :return: A :class:`ReactiveSilencerResult`; ``resonances`` holds
        :math:`f_0 = (c / 2 \pi) \sqrt{S_{\mathrm{neck}} / (l_\mathrm{e} V)}`.
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    s_d = require_positive(duct_area, "duct_area")
    zb = helmholtz_impedance(
        f,
        neck_area,
        neck_length,
        cavity_volume,
        resistance=resistance,
        speed_of_sound=c,
        density=rho,
    )
    t = shunt_matrix(zb)
    f0 = c / (2.0 * np.pi) * np.sqrt(neck_area / (neck_length * cavity_volume))
    return _result(
        f,
        t,
        inlet_area=s_d,
        outlet_area=s_d,
        c=c,
        rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="Helmholtz resonator",
        resonances=np.array([f0], dtype=np.float64),
        geometry={
            "duct_area": float(s_d),
            "neck_area": float(neck_area),
            "neck_length": float(neck_length),
            "cavity_volume": float(cavity_volume),
        },
    )


def quarter_wave_resonator(
    frequencies: ArrayLike,
    duct_area: float,
    length: float,
    branch_area: float,
    *,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult:
    r"""Closed quarter-wave side-branch tube on a duct (Bies Eqs. (8.144), (8.146)).

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param duct_area: Main-duct cross-sectional area ``S_d``, m2.
    :param length: Effective branch length ``l_e``, m.
    :param branch_area: Branch tube area ``S``, m2.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :param source_impedance: Optional source impedance ``Z_s``, Pa s/m3.
    :param radiation_impedance: Optional radiation impedance ``Z_r``, Pa s/m3.
    :return: A :class:`ReactiveSilencerResult`; ``resonances`` holds the odd
        multiples of :math:`f = c / (4 l_\mathrm{e})` within the frequency range.
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    s_d = require_positive(duct_area, "duct_area")
    le = require_positive(length, "length")
    zb = quarter_wave_impedance(f, le, branch_area, speed_of_sound=c, density=rho)
    t = shunt_matrix(zb)
    f_fundamental = c / (4.0 * le)
    odds = np.arange(1, int(2.0 * f.max() / f_fundamental) + 2, 2)
    res = np.asarray(odds * f_fundamental, dtype=np.float64)
    res = res[res <= f.max()]
    if res.size == 0:
        res = np.array([f_fundamental], dtype=np.float64)
    return _result(
        f,
        t,
        inlet_area=s_d,
        outlet_area=s_d,
        c=c,
        rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="quarter-wave resonator",
        resonances=res,
        geometry={
            "duct_area": float(s_d),
            "length": float(length),
            "branch_area": float(branch_area),
        },
    )


def extended_tube_chamber(
    frequencies: ArrayLike,
    length: float,
    chamber_area: float,
    pipe_area: float,
    *,
    inlet_extension: float = 0.0,
    outlet_extension: float = 0.0,
    speed_of_sound: float = _C_AIR,
    density: float = _RHO_AIR,
    source_impedance: ArrayLike | None = None,
    radiation_impedance: ArrayLike | None = None,
) -> ReactiveSilencerResult:
    r"""Extended-inlet/outlet expansion chamber (Bies §8.9.7).

    The inlet and outlet pipes extend a distance into the chamber, forming
    annular quarter-wave side branches (of area
    :math:`S_{\mathrm{exp}} - S_{\mathrm{duct}}` and lengths
    equal to the extensions, Bies Eq. (8.156)) at the two junctions. Tuning the
    extensions (classically :math:`L/4` and :math:`L/2`) places quarter-wave
    peaks that
    fill the :math:`kL = n \pi` troughs of the plain expansion chamber. With
    both extensions ``0`` the result reduces exactly to
    :func:`expansion_chamber`.

    The junction where each extended pipe ends is where its three ducts meet,
    so the straight chamber element cascaded between the two side branches is
    the length left over,
    :math:`L_c = L - L_a - L_b` (Bies Figure 8.19(a) and Example 8.2, where
    :math:`L = L_a + L_b + L_c`), and not the full chamber length. When the two
    extensions meet (:math:`L_a + L_b = L`) the straight element vanishes and
    the two annular branches shunt the same plane, which is the well-defined
    limit of the cascade; extensions that would overlap are rejected.

    :param frequencies: Frequencies ``f``, Hz (1-D array).
    :param length: Overall chamber length ``L``, m, extensions included.
    :param chamber_area: Chamber cross-sectional area ``S_exp``, m2.
    :param pipe_area: Inlet/outlet pipe area ``S_duct``, m2.
    :param inlet_extension: Inlet pipe extension into the chamber ``L_a``, m.
    :param outlet_extension: Outlet pipe extension into the chamber ``L_b``, m.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    :param source_impedance: Optional source impedance ``Z_s``, Pa s/m3.
    :param radiation_impedance: Optional radiation impedance ``Z_r``, Pa s/m3.
    :return: A :class:`ReactiveSilencerResult`.
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    s_exp = require_positive(chamber_area, "chamber_area")
    s_duct = require_positive(pipe_area, "pipe_area")
    length = require_non_negative(length, "length")
    la = require_non_negative(inlet_extension, "inlet_extension")
    lb = require_non_negative(outlet_extension, "outlet_extension")
    if s_exp <= s_duct:
        msg = "'chamber_area' must exceed 'pipe_area'."
        raise ValueError(msg)
    # The straight section between the two junction planes. Extensions that
    # meet exactly are the accepted limit, but binary arithmetic does not
    # respect that: 0.1 + 0.2 exceeds 0.3 by an ulp, so a chamber described in
    # round decimal metres would be refused for a geometry it does in fact
    # have. What is rejected is therefore a materially negative section, and
    # anything within rounding of zero is clamped to it, since a duct of
    # negative length is not a thing either.
    straight = length - la - lb
    if straight < -_MEETING_SLACK * max(length, la + lb):
        msg = (
            "the inlet and outlet extensions cannot together exceed the "
            "chamber length (they would overlap inside the chamber, leaving "
            "a straight chamber section of negative length)."
        )
        raise ValueError(msg)
    straight = max(straight, 0.0)
    annulus = s_exp - s_duct

    elements = []
    resonances = []
    if la > 0.0:
        elements.append(
            shunt_matrix(
                quarter_wave_impedance(f, la, annulus, speed_of_sound=c, density=rho)
            )
        )
        resonances.append(c / (4.0 * la))
    elements.append(duct_matrix(f, straight, s_exp, speed_of_sound=c, density=rho))
    if lb > 0.0:
        elements.append(
            shunt_matrix(
                quarter_wave_impedance(f, lb, annulus, speed_of_sound=c, density=rho)
            )
        )
        resonances.append(c / (4.0 * lb))
    t = cascade(*elements)
    res = np.array(resonances, dtype=np.float64) if resonances else None
    return _result(
        f,
        t,
        inlet_area=s_duct,
        outlet_area=s_duct,
        c=c,
        rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="extended-tube chamber",
        resonances=res,
        geometry={
            "length": float(length),
            "chamber_area": float(chamber_area),
            "pipe_area": float(pipe_area),
            "inlet_extension": float(inlet_extension),
            "outlet_extension": float(outlet_extension),
        },
    )


#: The ``kind`` of a result assembled element by element rather than from one
#: of the named devices; shared with the geometry renderer.
_KIND_CHAIN = "element chain"


@dataclass(frozen=True)
class SilencerChainElement:
    """One recorded element of a :class:`SilencerChain`.

    The element carries its four-pole matrix and, with it, whatever geometry
    the call that produced the matrix was given. That is the whole asymmetry
    of a hand-built chain: :func:`duct_matrix` is handed a length and an area,
    so a duct element knows its shape, while :func:`shunt_matrix` is handed an
    impedance and nothing else, so a shunt element has no shape to know.

    :ivar matrix: The element's ``(n_freq, 2, 2)`` four-pole matrix.
    :ivar length: Duct length ``L``, m, or ``None`` for a shunt element.
    :ivar area: Duct cross-sectional area ``S``, m2, or ``None`` for a shunt
        element.
    :ivar label: The name the element was given, or ``None``.
    :ivar shorting_frequency: For a shunt element, the analysis frequency at
        which ``|Z_b|`` is least (where the branch comes closest to shorting
        the duct), or ``None`` when that least value sits on an end of the
        analysis grid, and for every duct element.
    """

    matrix: np.ndarray
    length: float | None = None
    area: float | None = None
    label: str | None = None
    shorting_frequency: float | None = None

    @property
    def is_duct(self) -> bool:
        """Whether this is a duct element (an element with a declared area).

        :return: ``True`` for a duct element, ``False`` for a shunt element.
        """
        return self.area is not None


class SilencerChain:
    """A chain of four-pole elements that remembers the geometry it was given.

    :func:`duct_matrix`, :func:`shunt_matrix` and :func:`cascade` build any
    silencer layout the named devices do not cover, but they return bare
    matrices: the compound matrix of a hand-built chain is a stack of complex
    numbers, and nothing in it recalls that the first element was a 300 mm run
    of 200 mm duct. This class calls the same three functions and keeps the
    arguments, so the chain can be drawn (:meth:`plot_geometry`) as well as
    evaluated (:meth:`result`), and the drawing cannot drift from the model
    because one call produces both.

    Elements are added in order from inlet to outlet, and each adder returns
    the chain so the calls read as the device does::

        chain = (
            SilencerChain(frequencies)
            .duct(0.30, 0.0314)
            .shunt(quarter_wave_impedance(frequencies, 0.686, 0.0079))
            .duct(0.60, 0.1257)
            .duct(0.30, 0.0314)
        )

    What the drawing may show follows from what the elements know. A duct is
    drawn to scale from its declared length and area; a shunt declares an
    impedance, which fixes no length, no area and no volume, so it is marked
    at the station where it joins the run and nothing about its shape is
    invented (see :meth:`plot_geometry`).

    :param frequencies: Frequencies ``f``, Hz (1-D array), shared by every
        element of the chain.
    :param speed_of_sound: Speed of sound ``c``, m/s.
    :param density: Air density ``rho``, kg/m3.
    """

    def __init__(
        self,
        frequencies: ArrayLike,
        *,
        speed_of_sound: float = _C_AIR,
        density: float = _RHO_AIR,
    ) -> None:
        self._frequencies = _frequencies(frequencies)
        self._c = require_positive(speed_of_sound, "speed_of_sound")
        self._rho = require_positive(density, "density")
        self._elements: list[SilencerChainElement] = []

    @property
    def frequencies(self) -> NDArray[np.float64]:
        """The analysis frequencies shared by every element, Hz.

        :return: The frequency grid the chain was built on.
        """
        return self._frequencies

    @property
    def elements(self) -> tuple[SilencerChainElement, ...]:
        """The recorded elements, in order from inlet to outlet.

        :return: The elements added so far.
        """
        return tuple(self._elements)

    @property
    def transfer_matrix(self) -> _Complex:
        """The compound four-pole matrix of the chain.

        :return: The ordered product :func:`cascade` makes of the element
            matrices, ``(n_freq, 2, 2)``.
        :raises ValueError: If the chain is empty.
        """
        return cascade(*(element.matrix for element in self._elements))

    def duct(self, length: float, area: float) -> SilencerChain:
        """Append a straight duct of length ``L`` and area ``S``.

        :param length: Duct length ``L``, m. A zero-length duct is the
            identity matrix, so it is neither computed against nor drawn.
        :param area: Cross-sectional area ``S``, m2.
        :return: The chain, so the calls can be written one after another.
        """
        length = require_non_negative(length, "length")
        area = require_positive(area, "area")
        self._elements.append(
            SilencerChainElement(
                matrix=duct_matrix(
                    self._frequencies,
                    length,
                    area,
                    speed_of_sound=self._c,
                    density=self._rho,
                ),
                length=length,
                area=area,
            )
        )
        return self

    def shunt(
        self, branch_impedance: ArrayLike, *, label: str | None = None
    ) -> SilencerChain:
        """Append a side branch of acoustic impedance ``Z_b``.

        The branch is the only element that can be given a ``label``, because
        it is the only one the drawing cannot identify by its dimensions.

        :param branch_impedance: Acoustic impedance ``Z_b`` of the branch,
            Pa s/m3: one value per analysis frequency, or a scalar held
            constant over the grid.
        :param label: What the branch is, e.g.
            ``"Helmholtz resonator, 125 Hz"``. Rendered verbatim in the
            drawing, in whatever language it is written in.
        :return: The chain, so the calls can be written one after another.
        :raises ValueError: If ``branch_impedance`` is neither a scalar nor one
            value per analysis frequency.
        """
        zb = np.asarray(branch_impedance, dtype=np.complex128)
        if zb.ndim == 0:
            zb = np.full(self._frequencies.size, zb, dtype=np.complex128)
        if zb.shape != self._frequencies.shape:
            msg = (
                "'branch_impedance' must be a scalar or hold one value per "
                f"analysis frequency ({self._frequencies.size} values); got "
                f"shape {zb.shape}."
            )
            raise ValueError(msg)
        self._elements.append(
            SilencerChainElement(
                matrix=shunt_matrix(zb),
                label=label,
                shorting_frequency=_shorting_frequency(self._frequencies, zb),
            )
        )
        return self

    def result(
        self,
        *,
        inlet_area: float,
        outlet_area: float,
        source_impedance: ArrayLike | None = None,
        radiation_impedance: ArrayLike | None = None,
    ) -> ReactiveSilencerResult:
        """Evaluate the chain into a :class:`ReactiveSilencerResult`.

        The port areas are the pipes the chain is connected between, which
        ``transmission_loss`` needs and the chain itself does not contain: put
        them in the chain as duct elements if the drawing is to show them.

        :param inlet_area: Inlet pipe area ``S_in``, m2.
        :param outlet_area: Outlet pipe area ``S_out``, m2.
        :param source_impedance: Optional source impedance ``Z_s`` for the
            insertion loss, Pa s/m3.
        :param radiation_impedance: Optional radiation impedance ``Z_r`` for
            the insertion loss, Pa s/m3.
        :return: The result, carrying a snapshot of this chain so that it can
            be drawn as well as plotted and reported.
        :raises ValueError: If the chain is empty.
        """
        areas = [inlet_area, outlet_area]
        areas += [
            element.area for element in self._elements if element.area is not None
        ]
        return _result(
            self._frequencies,
            self.transfer_matrix,
            inlet_area=inlet_area,
            outlet_area=outlet_area,
            c=self._c,
            rho=self._rho,
            source_impedance=source_impedance,
            radiation_impedance=radiation_impedance,
            kind=_KIND_CHAIN,
            areas=areas,
            chain=self._snapshot(),
        )

    def plot_geometry(self, ax: Axes | None = None, *, language: str = "en") -> Axes:
        """Draw the chain: its ducts to scale, its branch points marked.

        Every duct is drawn at its declared length and equivalent circular
        diameter ``d = 2 sqrt(S / pi)``, so the runs, the area steps between
        them and the overall length are read off the page. A shunt element
        holds an impedance and no geometry at all, so it is not drawn as a
        stub of any length: it is marked with a leader at the station where it
        joins the run, carrying its label and, when the analysis grid resolves
        one, the frequency at which it comes closest to shorting the duct.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :param ax: Existing axes, or ``None`` to create a figure.
        :param language: Label language, ``"en"`` (default) or ``"es"``.
        :return: The axes.
        :raises ValueError: If the chain holds no duct of positive length, and
            so has no geometry and no scale to draw at.
        """
        from .._i18n import check_language
        from .._plot.geometry import plot_silencer_chain_geometry

        check_language(language)
        return plot_silencer_chain_geometry(self, ax=ax, language=language)

    def _snapshot(self) -> SilencerChain:
        """A copy of the chain as it stands, immune to later additions."""
        clone = SilencerChain(
            self._frequencies, speed_of_sound=self._c, density=self._rho
        )
        clone._elements.extend(self._elements)
        return clone
