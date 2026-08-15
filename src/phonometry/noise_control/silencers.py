#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Reactive silencers by the four-pole (transmission-matrix) method.

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
   1 / Z_b & 1
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
   \left\lvert \frac{T_{11} Z_r + T_{12} + Z_s Z_r T_{21} + Z_s T_{22}}
   {Z_s + Z_r} \right\rvert,

which is ``0`` when the silencer reduces to a through connection
(:math:`T = I`)
and, for equal inlet/outlet areas, equals the transmission loss for the
anechoic reference :math:`Z_s = Z_r = \rho c / S` (with unequal areas the
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

from .._internal.validation import require_non_negative, require_positive
from .duct_modes import plane_wave_limit as _plane_wave_limit
from .duct_modes import warn_above_plane_wave_limit

if TYPE_CHECKING:
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

_Complex = NDArray[np.complex128]


def _frequencies(frequencies: ArrayLike) -> NDArray[np.float64]:
    """Validate a strictly positive, finite 1-D frequency grid (Hz)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if f.ndim != 1 or f.size == 0:
        raise ValueError("'frequencies' must be a non-empty 1-D array.")
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("'frequencies' must be positive and finite.")
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
        raise ValueError("'branch_impedance' must be a non-empty 1-D array.")
    t = np.zeros((zb.size, 2, 2), dtype=np.complex128)
    t[:, 0, 0] = 1.0
    # A lossless branch at exact resonance has Z_b -> 0 (it shorts the duct);
    # 1/Z_b -> infinity gives the ideal infinite attenuation there.
    with np.errstate(divide="ignore", invalid="ignore"):
        t[:, 1, 0] = 1.0 / zb
    t[:, 1, 1] = 1.0
    return t


def cascade(*matrices: _Complex) -> _Complex:
    """Cascade element four-pole matrices from inlet to outlet.

    The compound matrix is the ordered product ``T1 @ T2 @ ... @ Tn`` (the
    state at the inlet equals the compound matrix times the state at the
    outlet), broadcast over the frequency axis.

    :param matrices: One or more ``(n_freq, 2, 2)`` arrays sharing ``n_freq``.
    :return: The compound ``(n_freq, 2, 2)`` array.
    """
    if not matrices:
        raise ValueError("cascade() needs at least one matrix.")
    n = matrices[0].shape[0]
    if any(m.shape[0] != n for m in matrices[1:]):
        raise ValueError(
            "cascade() matrices must share the same frequency grid (n_freq)."
        )
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
       \left\lvert \frac{T_{11} Z_r + T_{12} + Z_s Z_r T_{21} + Z_s T_{22}}
       {Z_s + Z_r} \right\rvert.

    :param transfer_matrix: A ``(n_freq, 2, 2)`` compound matrix.
    :param source_impedance: Source internal acoustic impedance ``Z_s``,
        Pa s/m3 (scalar or per-frequency, real or complex).
    :param radiation_impedance: Termination/radiation acoustic impedance
        ``Z_r``, Pa s/m3 (scalar or per-frequency).
    :return: The insertion loss per frequency, dB.
    """
    n = transfer_matrix.shape[0]
    zs = np.broadcast_to(
        np.asarray(source_impedance, dtype=np.complex128), (n,)
    )
    zr = np.broadcast_to(
        np.asarray(radiation_impedance, dtype=np.complex128), (n,)
    )
    t11 = transfer_matrix[:, 0, 0]
    t12 = transfer_matrix[:, 0, 1]
    t21 = transfer_matrix[:, 1, 0]
    t22 = transfer_matrix[:, 1, 1]
    num = t11 * zr + t12 + zs * zr * t21 + zs * t22
    return np.asarray(
        20.0 * np.log10(np.abs(num / (zs + zr))), dtype=np.float64
    )


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

    :math:`Z = R + j(\rho \omega l_e / S_{\mathrm{neck}} -
    \rho c^2 / (\omega V))` with acoustic
    mass :math:`\rho l_e / S_{\mathrm{neck}}` and compliance
    :math:`V / (\rho c^2)`; the resonance
    :math:`f_0 = (c / 2 \pi) \sqrt{S_{\mathrm{neck}} / (l_e V)}`
    (Bies Eq. (8.46)) is where the
    reactance vanishes and the branch shorts the duct.

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

    :math:`Z = -j (\rho c / S) \cot(k l_e)`; the reactance vanishes at
    :math:`l_e = \lambda / 4` (:math:`f = c / 4 l_e`), where the closed tube
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
        hand-built results.
    :ivar plane_wave_limit: The first higher-order-mode cut-on frequency of the
        widest cross section of the device, Hz (Norton & Karczub Eq. 7.6,
        :func:`phonometry.noise_control.duct_modes.plane_wave_limit`). The
        four-pole algebra of this module is one-dimensional and is valid below
        it; above it several modes propagate at once and the result describes
        the plane-wave mode only, which is why a
        :class:`~phonometry.noise_control.duct_modes.PlaneWaveWarning` is
        raised when the analysis reaches past it. ``None`` for hand-built
        results that do not retain their geometry.
    """

    frequencies: np.ndarray
    transmission_loss: np.ndarray
    insertion_loss: np.ndarray | None
    transfer_matrix: np.ndarray
    kind: str
    resonances: np.ndarray | None = None
    geometry: dict[str, float] | None = None
    plane_wave_limit: float | None = None

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the transmission (and insertion) loss against frequency.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.noise_control import plot_reactive_silencer

        check_language(language)
        return plot_reactive_silencer(self, ax=ax, language=language, **kwargs)

    def plot_geometry(
        self, ax: Axes | None = None, *, language: str = "en"
    ) -> Axes:
        """Draw the silencer cross-section to scale (dimensioned side cut).

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.

        :raises ValueError: If the result does not retain its ``geometry``.
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
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
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
) -> ReactiveSilencerResult:
    """Assemble a :class:`ReactiveSilencerResult` from a compound matrix.

    Also determines the frequency at which the plane-wave assumption behind the
    whole four-pole formulation expires: the first higher-order-mode cut-on of
    the widest cross section the device presents (the chamber of an expansion
    silencer, the duct of a side-branch resonator), and warns when the analysis
    grid reaches past it.
    """
    limit: float | None = None
    if geometry is not None:
        areas = [
            value
            for key, value in geometry.items()
            if key.endswith("_area") and value > 0.0
        ]
        if areas:
            limit = _plane_wave_limit(area=max(areas), speed_of_sound=c)
            warn_above_plane_wave_limit(f, limit, kind, stacklevel=4)
    tl = transmission_loss(
        t, inlet_area=inlet_area, outlet_area=outlet_area,
        speed_of_sound=c, density=rho,
    )
    il: NDArray[np.float64] | None = None
    if source_impedance is not None and radiation_impedance is not None:
        il = insertion_loss(
            t, source_impedance=source_impedance,
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
        f, t, inlet_area=pipe_area, outlet_area=pipe_area, c=c, rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance, kind="expansion chamber",
        geometry={
            "length": float(length), "chamber_area": float(chamber_area),
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
        :math:`f_0 = (c / 2 \pi) \sqrt{S_{\mathrm{neck}} / (l_e V)}`.
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    s_d = require_positive(duct_area, "duct_area")
    zb = helmholtz_impedance(
        f, neck_area, neck_length, cavity_volume, resistance=resistance,
        speed_of_sound=c, density=rho,
    )
    t = shunt_matrix(zb)
    f0 = c / (2.0 * np.pi) * np.sqrt(neck_area / (neck_length * cavity_volume))
    return _result(
        f, t, inlet_area=s_d, outlet_area=s_d, c=c, rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="Helmholtz resonator",
        resonances=np.array([f0], dtype=np.float64),
        geometry={
            "duct_area": float(s_d), "neck_area": float(neck_area),
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
        multiples of :math:`f = c / (4 l_e)` within the frequency range.
    """
    f = _frequencies(frequencies)
    c = require_positive(speed_of_sound, "speed_of_sound")
    rho = require_positive(density, "density")
    s_d = require_positive(duct_area, "duct_area")
    le = require_positive(length, "length")
    zb = quarter_wave_impedance(
        f, le, branch_area, speed_of_sound=c, density=rho
    )
    t = shunt_matrix(zb)
    f_fundamental = c / (4.0 * le)
    odds = np.arange(1, int(2.0 * f.max() / f_fundamental) + 2, 2)
    res = np.asarray(odds * f_fundamental, dtype=np.float64)
    res = res[res <= f.max()]
    if res.size == 0:
        res = np.array([f_fundamental], dtype=np.float64)
    return _result(
        f, t, inlet_area=s_d, outlet_area=s_d, c=c, rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="quarter-wave resonator",
        resonances=res,
        geometry={
            "duct_area": float(s_d), "length": float(length),
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
        raise ValueError("'chamber_area' must exceed 'pipe_area'.")
    # The straight section between the two junction planes. Extensions that
    # meet exactly are the accepted limit, but binary arithmetic does not
    # respect that: 0.1 + 0.2 exceeds 0.3 by an ulp, so a chamber described in
    # round decimal metres would be refused for a geometry it does in fact
    # have. What is rejected is therefore a materially negative section, and
    # anything within rounding of zero is clamped to it, since a duct of
    # negative length is not a thing either.
    straight = length - la - lb
    if straight < -_MEETING_SLACK * max(length, la + lb):
        raise ValueError(
            "the inlet and outlet extensions cannot together exceed the "
            "chamber length (they would overlap inside the chamber, leaving "
            "a straight chamber section of negative length)."
        )
    straight = max(straight, 0.0)
    annulus = s_exp - s_duct

    elements = []
    resonances = []
    if la > 0.0:
        elements.append(shunt_matrix(
            quarter_wave_impedance(f, la, annulus, speed_of_sound=c, density=rho)
        ))
        resonances.append(c / (4.0 * la))
    elements.append(
        duct_matrix(f, straight, s_exp, speed_of_sound=c, density=rho)
    )
    if lb > 0.0:
        elements.append(shunt_matrix(
            quarter_wave_impedance(f, lb, annulus, speed_of_sound=c, density=rho)
        ))
        resonances.append(c / (4.0 * lb))
    t = cascade(*elements)
    res = np.array(resonances, dtype=np.float64) if resonances else None
    return _result(
        f, t, inlet_area=s_duct, outlet_area=s_duct, c=c, rho=rho,
        source_impedance=source_impedance,
        radiation_impedance=radiation_impedance,
        kind="extended-tube chamber", resonances=res,
        geometry={
            "length": float(length), "chamber_area": float(chamber_area),
            "pipe_area": float(pipe_area),
            "inlet_extension": float(inlet_extension),
            "outlet_extension": float(outlet_extension),
        },
    )
