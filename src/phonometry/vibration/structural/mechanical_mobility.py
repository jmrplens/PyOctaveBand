#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Mechanical mobility and the frequency-response-function family (ISO 7626-1:2011).

Mechanical **mobility** is the complex ratio of a velocity response to the
excitation force that produces it, :math:`Y_{ij} = v_i / F_j` (ISO 7626-1,
3.1.2). It is
one of a family of motion-per-force frequency-response functions (FRFs); which
member is used depends only on whether the motion is expressed as displacement,
velocity or acceleration, and each has a force-per-motion reciprocal
(ISO 7626-1, Table 1):

===============  =====================  ===========  =========================
Motion           FRF (motion / force)   Unit          Reciprocal (force / motion)
===============  =====================  ===========  =========================
displacement     dynamic compliance /   m/N           dynamic stiffness  (N/m)
                 receptance ``H``
velocity         mobility ``Y``         m/(N.s)        impedance          (N.s/m)
acceleration     accelerance ``A``      1/kg           apparent mass      (kg)
===============  =====================  ===========  =========================

For a harmonic motion :math:`x e^{j \omega t}` the velocity is
:math:`j \omega x` and the acceleration :math:`-\omega^2 x`, so every FRF
follows from the receptance ``H``:

.. math::

   Y = j \omega H \qquad A = -\omega^2 H

   Z~\text{(impedance)} = \frac{1}{Y}

   M~\text{(apparent mass)} = \frac{1}{A}
   \quad \text{(Table 1 name: "effective mass")}

   K~\text{(dyn. stiffness)} = \frac{1}{H}

These element-wise reciprocals are the **free** quantities of ISO 7626-1,
3.1.4; the *blocked* matrix quantities of Table 1 do not invert element-wise
(:math:`Z_{ij} \ne 1/Y_{ij}` for multi-coordinate systems); see
:func:`convert_frf`.

:func:`convert_frf` moves between any two of the six FRFs through the receptance
pivot. A **driving-point** FRF has the response and force at the same point
(:math:`i = j`); a **transfer** FRF has them at different points.

The canonical closed-form reference is the single-degree-of-freedom resonator
of mass ``m``, viscous damping ``c`` and stiffness ``k``, whose receptance is
:math:`H(\omega) = 1 / (k - \omega^2 m + j \omega c)` (consistent with the
ISO 7626-1
Table 1 / 3.1.2 definitions). At its resonance :math:`\omega_0 = \sqrt{k/m}`
the driving-point mobility is purely real and equal to :math:`1/c` -- the
mobility peak
measures the damping. This module is the FRF backbone for the structure-borne
source and transmission standards (ISO 9611, ISO 10846, EN 15657, EN 12354-5).

**Measured FRFs (ISO 7626-2:2015).** The single-point measurement side is
covered by the library's spectral estimators: processing of random-excitation
records per ISO 7626-2, 8.1.3 -- the H1 estimator
:math:`H = G(\text{response},\text{force}) / G(\text{force},\text{force})`
-- and the ordinary coherence
:math:`\gamma^2 = |G_{xy}|^2 / (G_{xx} G_{yy})`
used for its data-quality checks are
:func:`phonometry.electroacoustics.frequency_response.transfer_function` (with
``estimator="H1"``, the default) and
:func:`phonometry.electroacoustics.frequency_response.coherence`; both are
exported at the package top level. This module adds the ISO 7626-2 acceptance
criteria on top of them: the operational rigid-mass calibration of 7.5.2
(:func:`rigid_mass_calibration_check`, +/- 5 %) and the Annex A normalized
random error with its < 5 % averaging criterion (:func:`random_error_percent`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike, NDArray

    from ..._report.metadata import ReportMetadata


from ..._internal.validation import (
    require_non_negative,
    require_positive,
    require_ranks,
    require_same_length,
)

# ---------------------------------------------------------------------------
# FRF taxonomy (ISO 7626-1 Table 1).
# ---------------------------------------------------------------------------

#: The six frequency-response functions, keyed by name. ``motion`` is the power
#: of ``j*omega`` relating the FRF to the receptance (0 displacement, 1 velocity,
#: 2 acceleration); ``inverse`` marks the force-per-motion reciprocals.
_FRF_TYPES: dict[str, tuple[int, bool]] = {
    "receptance": (0, False),  # dynamic compliance, x/F  [m/N]
    "mobility": (1, False),  # v/F                      [m/(N.s)]
    "accelerance": (2, False),  # a/F                      [1/kg]
    "dynamic_stiffness": (0, True),  # F/x                      [N/m]
    "impedance": (1, True),  # F/v                      [N.s/m]
    "apparent_mass": (2, True),  # F/a                      [kg]
}

#: SI unit strings for each FRF, for labelling.
FRF_UNITS: dict[str, str] = {
    "receptance": "m/N",
    "mobility": "m/(N·s)",
    "accelerance": "1/kg",
    "dynamic_stiffness": "N/m",
    "impedance": "N·s/m",
    "apparent_mass": "kg",
}


def _omega(frequency: ArrayLike) -> NDArray[np.float64]:
    r"""Angular frequency :math:`\omega = 2 \pi f` (rad/s); rejects f <= 0."""
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        msg = "'frequency' must be positive (mobility conversions divide by omega)."
        raise ValueError(msg)
    return 2.0 * np.pi * f


def _to_receptance(
    value: NDArray[np.complex128], omega: NDArray[np.float64], source: str
) -> NDArray[np.complex128]:
    r"""Reduce any FRF to the receptance :math:`H = x / F`."""
    power, inverse = _FRF_TYPES[source]
    # Undo the force-per-motion reciprocal, then the (j omega)**power factor.
    motion_per_force = 1.0 / value if inverse else value
    return motion_per_force / (1j * omega) ** power


def _from_receptance(
    receptance: NDArray[np.complex128], omega: NDArray[np.float64], target: str
) -> NDArray[np.complex128]:
    """Build any FRF from the receptance ``H``."""
    power, inverse = _FRF_TYPES[target]
    motion_per_force = receptance * (1j * omega) ** power
    result = 1.0 / motion_per_force if inverse else motion_per_force
    return np.asarray(result, dtype=np.complex128)


def convert_frf(
    value: ArrayLike,
    frequency: ArrayLike,
    source: str,
    target: str,
) -> np.ndarray:
    r"""Convert a frequency-response function between any two kinds (Table 1).

    .. note::
        The force-per-motion kinds returned here are arithmetic reciprocals of
        the motion-per-force FRFs: the **free** quantities of ISO 7626-1,
        3.1.4 (all other response coordinates unconstrained). They coincide
        with the **blocked** matrix quantities of Table 1 only for a scalar
        (single-coordinate) system: blocked matrices do not invert
        element-wise, :math:`Z_{ij} \ne 1/Y_{ij}` in general. For
        driving-point or
        single-path use (e.g. the ISO 10846-1 Table A.2 relations) the free
        forms are exactly what is needed; for multi-coordinate blocked
        quantities invert the full FRF matrix instead. ISO 7626-1 Table 1
        names ``F/a`` the **effective mass** (also known as apparent mass,
        the name used here).

    :param value: The (complex) FRF value(s) of kind *source*.
    :param frequency: Frequency ``f``, in hertz (scalar or array, broadcast with
        *value*).
    :param source: The FRF kind of *value* -- one of ``"receptance"``,
        ``"mobility"``, ``"accelerance"``, ``"dynamic_stiffness"``,
        ``"impedance"`` or ``"apparent_mass"``.
    :param target: The FRF kind to convert to (same set).
    :return: The FRF value(s) of kind *target*, as a complex array.
    :raises ValueError: for an unknown FRF name, a non-positive frequency, or
        a zero *value* (dead channel) when the conversion involves a
        force-per-motion reciprocal.
    """
    for name, role in ((source, "source"), (target, "target")):
        if name not in _FRF_TYPES:
            msg = f"unknown {role} FRF {name!r}; choose from {tuple(_FRF_TYPES)}."
            raise ValueError(msg)
    omega = _omega(frequency)
    val = np.asarray(value, dtype=np.complex128)
    if (_FRF_TYPES[source][1] or _FRF_TYPES[target][1]) and not np.all(
        np.abs(val) > 0.0
    ):
        msg = (
            "'value' contains zeros (dead channel); converting "
            f"{source!r} to {target!r} takes a reciprocal, which is "
            "undefined there."
        )
        raise ValueError(msg)
    receptance = _to_receptance(val, omega, source)
    return np.asarray(_from_receptance(receptance, omega, target), dtype=np.complex128)


# ---------------------------------------------------------------------------
# Single-degree-of-freedom closed-form reference (consistent with the
# ISO 7626-1 Table 1 / 3.1.2 FRF definitions).
# ---------------------------------------------------------------------------


def sdof_receptance(
    frequency: ArrayLike, mass: float, stiffness: float, damping: float
) -> np.ndarray:
    r"""Receptance of a viscously damped SDOF resonator (closed form).

    :math:`H(\omega) = 1 / (k - \omega^2 m + j \omega c)`, the textbook
    single-degree-of-freedom reference, expressed in the FRF taxonomy of
    ISO 7626-1 (Table 1 / 3.1.2 definitions).

    :param frequency: Frequency ``f``, in hertz (scalar or array).
    :param mass: Mass ``m``, in kg.
    :param stiffness: Stiffness ``k``, in N/m.
    :param damping: Viscous damping coefficient ``c``, in N.s/m (>= 0).
    :return: The complex receptance ``H``, in m/N.
    """
    mass = require_positive(mass, "mass")
    stiffness = require_positive(stiffness, "stiffness")
    damping = require_non_negative(damping, "damping")
    omega = _omega(frequency)
    denom = stiffness - omega**2 * mass + 1j * omega * damping
    return np.asarray(1.0 / denom, dtype=np.complex128)


def sdof_mobility(
    frequency: ArrayLike, mass: float, stiffness: float, damping: float
) -> np.ndarray:
    r"""Mobility of a viscously damped SDOF resonator: :math:`Y = j \omega H`.

    :param frequency: Frequency ``f``, in hertz.
    :param mass: Mass ``m``, in kg.
    :param stiffness: Stiffness ``k``, N/m.
    :param damping: Viscous damping ``c``, N.s/m.
    :return: The complex mobility ``Y``, in m/(N.s).
    """
    h = sdof_receptance(frequency, mass, stiffness, damping)
    return convert_frf(h, frequency, "receptance", "mobility")


def sdof_accelerance(
    frequency: ArrayLike, mass: float, stiffness: float, damping: float
) -> np.ndarray:
    r"""Accelerance of a viscously damped SDOF resonator: :math:`A = -\omega^2 H`.

    :param frequency: Frequency ``f``, in hertz.
    :param mass: Mass ``m``, in kg.
    :param stiffness: Stiffness ``k``, N/m.
    :param damping: Viscous damping ``c``, N.s/m.
    :return: The complex accelerance ``A``, in 1/kg.
    """
    h = sdof_receptance(frequency, mass, stiffness, damping)
    return convert_frf(h, frequency, "receptance", "accelerance")


def resonance_frequency(mass: float, stiffness: float) -> float:
    r"""Undamped natural frequency :math:`f_0 = (1/2\pi) \sqrt{k/m}` of the
    SDOF, in Hz.

    :param mass: Mass ``m``, in kg.
    :param stiffness: Stiffness ``k``, in N/m.
    :return: The natural frequency ``f0``, in hertz.
    """
    mass = require_positive(mass, "mass")
    stiffness = require_positive(stiffness, "stiffness")
    return float(np.sqrt(stiffness / mass) / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Measured-FRF acceptance checks (ISO 7626-2:2015).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidMassCalibrationResult:
    r"""Operational rigid-mass calibration check (ISO 7626-2:2015, 7.5.2).

    :ivar frequencies: Frequencies of the calibration FRF, in hertz.
    :ivar measured: Measured FRF magnitude per frequency (``1/kg`` for
        accelerance, ``m/(N.s)`` for mobility).
    :ivar expected: Known correct magnitude of the rigid calibration block per
        frequency: :math:`1/m` (accelerance) or :math:`1/(2 \pi f m)`
        (mobility).
    :ivar deviation: Relative deviation ``measured/expected - 1`` per frequency.
    :ivar within_tolerance: Per-frequency pass flag ``|deviation| <= tolerance``.
    :ivar passed: ``True`` if every frequency is within the tolerance.
    :ivar mass: Mass ``m`` of the calibration block, in kg.
    :ivar quantity: FRF kind checked (``"accelerance"`` or ``"mobility"``).
    :ivar tolerance: Relative tolerance applied (the standard's is 0.05).
    """

    frequencies: np.ndarray
    measured: np.ndarray
    expected: np.ndarray
    deviation: np.ndarray
    within_tolerance: np.ndarray
    passed: bool
    mass: float
    quantity: str
    tolerance: float

    def __post_init__(self) -> None:
        """Reject a calibration whose per-frequency quantities disagree.

        The check of 7.5.2 is a verdict on a frequency range, and
        :attr:`within_tolerance` is what carries it frequency by frequency.
        :meth:`plot` selects the passing points with ``frequencies[within]``
        and colours the rest as failures, so a mask of another length is
        numpy's complaint about a boolean index that does not match the array
        it indexes, raised from inside the plotter and naming neither field.
        :attr:`passed` protests less: nothing re-derives it, so a mask
        covering part of the measured range leaves an overall pass standing
        over the frequencies it happened to reach, and the drift the
        calibration exists to catch sits in the ones it did not.

        :raises ValueError: if the per-frequency quantities do not share one
            frequency axis.
        """
        require_ranks(
            self,
            frequencies=1,
            measured=1,
            expected=1,
            deviation=1,
            within_tolerance=1,
        )
        require_same_length(
            self,
            "frequencies",
            "measured",
            "expected",
            "deviation",
            "within_tolerance",
            axis="frequency",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | np.ndarray:
        """Plot the rigid-mass calibration check (ISO 7626-2, 7.5.2).

        The measured FRF magnitude against the known rigid-mass line with its
        tolerance band, and the relative deviation against the same band. With
        no ``ax`` a two-panel figure is drawn and its axes array returned; with
        ``ax`` the deviation diagnostic is drawn on it and that axes returned.

        Requires matplotlib (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language
        from ..._plot.vibration import plot_rigid_mass_calibration

        return plot_rigid_mass_calibration(
            self, ax=ax, language=check_language(language), **kwargs
        )


def rigid_mass_calibration_check(
    frf: ArrayLike,
    frequencies: ArrayLike,
    mass: float,
    *,
    quantity: str = "accelerance",
    tolerance: float = 0.05,
) -> RigidMassCalibrationResult:
    r"""Check an operational calibration on a rigid mass (ISO 7626-2, 7.5.2).

    The measured frequency response of a freely suspended rigid calibration
    block of known mass ``m`` shall agree within +/- 5 % with its known correct
    value: the accelerance magnitude :math:`\lvert A \rvert = 1/m` or the
    mobility magnitude
    :math:`\lvert Y \rvert = 1/(2 \pi f m)`. All components of the
    measurement chain (including
    the attachment hardware) are connected as in the test series, so a failure
    flags transducer, chain or attachment-compliance errors.

    :param frf: Measured calibration FRF (complex or magnitude, scalar or
        array), in 1/kg (accelerance) or m/(N.s) (mobility).
    :param frequencies: Frequencies of *frf*, in hertz (> 0, same shape).
    :param mass: Known mass ``m`` of the calibration block, in kg (> 0).
    :param quantity: ``"accelerance"`` (:math:`\lvert A \rvert = 1/m`) or
        ``"mobility"`` (:math:`\lvert Y \rvert = 1/(\omega m)`).
        (Default: ``"accelerance"``.)
    :param tolerance: Relative tolerance (Default: 0.05, the +/- 5 % of 7.5.2).
    :return: A :class:`RigidMassCalibrationResult` with per-band pass flags.
    :raises ValueError: for an unknown quantity, non-positive mass, tolerance
        or frequency, or mismatched shapes.
    """
    if quantity not in ("accelerance", "mobility"):
        msg = "'quantity' must be 'accelerance' or 'mobility'."
        raise ValueError(msg)
    mass = require_positive(mass, "mass")
    tolerance = require_positive(tolerance, "tolerance")
    freq = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    measured = np.abs(np.atleast_1d(np.asarray(frf, dtype=np.complex128)))
    if measured.shape != freq.shape:
        msg = "'frf' and 'frequencies' must have the same shape."
        raise ValueError(msg)
    omega = _omega(freq)
    if quantity == "accelerance":
        expected = np.full_like(freq, 1.0 / mass)
    else:
        expected = 1.0 / (omega * mass)
    deviation = measured / expected - 1.0
    within = np.abs(deviation) <= tolerance
    return RigidMassCalibrationResult(
        frequencies=freq,
        measured=np.asarray(measured, dtype=np.float64),
        expected=np.asarray(expected, dtype=np.float64),
        deviation=np.asarray(deviation, dtype=np.float64),
        within_tolerance=within,
        passed=bool(np.all(within)),
        mass=mass,
        quantity=quantity,
        tolerance=tolerance,
    )


def random_error_percent(coherence: ArrayLike, n_averages: int) -> np.ndarray:
    r"""Normalized random error of an averaged FRF magnitude (ISO 7626-2 Annex A).

    :math:`\epsilon = \sqrt{(1 - \gamma^2) / (2 n \gamma^2)}`, the normalized
    random
    error of the frequency-response-function magnitude estimated from ``n``
    averaged spectra with ordinary coherence :math:`\gamma^2` (the relation
    behind
    Figure A.2, from Bendat & Piersol). ISO 7626-2, 8.1.3 requires enough
    averages that the error at each resonance of a driving-point mobility is
    below 5 %: e.g. :math:`\gamma^2 = 0.8` needs about :math:`n = 75` spectra
    (:math:`\epsilon = 4.08` %), the Annex A example.

    :param coherence: Ordinary coherence :math:`\gamma^2` per frequency, in
        (0, 1].
    :param n_averages: Number of averaged spectra ``n`` (>= 1).
    :return: The normalized random error, in percent (same shape as input).
    :raises ValueError: for a coherence outside (0, 1] or ``n_averages < 1``.
    """
    n = int(n_averages)
    if n < 1:
        msg = "'n_averages' must be at least 1."
        raise ValueError(msg)
    gamma2 = np.asarray(coherence, dtype=np.float64)
    if np.any(gamma2 <= 0.0) or np.any(gamma2 > 1.0):
        msg = "'coherence' must lie in (0, 1]."
        raise ValueError(msg)
    eps = np.sqrt((1.0 - gamma2) / (2.0 * n * gamma2))
    return np.asarray(100.0 * eps, dtype=np.float64)


# ---------------------------------------------------------------------------
# Bundled FRF result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MobilityResult:
    """A measured or modelled mobility FRF over frequency.

    :ivar frequencies: Frequencies, in hertz.
    :ivar mobility: Complex mobility ``Y`` per frequency, in m/(N.s).
    :ivar driving_point: ``True`` if response and force are co-located (i = j).
    """

    frequencies: np.ndarray
    mobility: np.ndarray
    driving_point: bool = True

    def __post_init__(self) -> None:
        """Reject an FRF whose mobility does not lie on its frequency axis.

        The fiche boxes one number, the peak of ``|Y|``, and dates it: it
        takes the peak out of :attr:`mobility`, reads the frequency at that
        same index out of :attr:`frequencies`, and states the measured range
        from :attr:`frequencies` alone. Two axes of different lengths pair a
        peak drawn from one of them with a range drawn from the other, so the
        boxed resonance can sit outside the range printed beside it; where
        the peak falls past the end of the shorter axis it is instead an
        index error out of the renderer, naming neither field.

        :raises ValueError: if the mobility does not carry one value per
            frequency.
        """
        require_ranks(self, frequencies=1, mobility=1)
        require_same_length(self, "frequencies", "mobility", axis="frequency")

    @property
    def magnitude(self) -> np.ndarray:
        """Mobility magnitude ``|Y|``, in m/(N.s)."""
        return np.asarray(np.abs(self.mobility), dtype=np.float64)

    @property
    def phase(self) -> np.ndarray:
        """Mobility phase, in radians."""
        return np.asarray(np.angle(self.mobility), dtype=np.float64)

    def to(self, target: str) -> np.ndarray:
        r"""Convert the mobility to another FRF kind (see :func:`convert_frf`).

        The force-per-motion kinds are element-wise reciprocals, i.e. the
        *free* quantities of ISO 7626-1, 3.1.4: on a transfer FRF
        (``driving_point=False``), ``to("impedance")`` returns the free
        impedance :math:`1/Y_{ij}`, not the blocked matrix impedance of Table 1
        (which does not invert element-wise).
        """
        return convert_frf(self.mobility, self.frequencies, "mobility", target)

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the mobility magnitude ``|Y(f)|``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.vibration import plot_mobility

        return plot_mobility(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a mechanical-mobility measurement fiche to a PDF (ISO 7626).

        Writes a one-page mobility measurement report: the standard-basis line
        naming whether the mobility is a driving-point or a transfer FRF
        (ISO 7626-1:2011 definitions; measurement per ISO 7626-2:2015), an
        optional metadata header, a two-panel body with a compact table of the
        FRF's characteristic points (the FRF type, the frequency range, the peak
        frequency, the peak mobility magnitude and the phase there) beside the
        mobility magnitude spectrum ``|Y(f)|``, a boxed peak mobility ``|Y|``
        with the frequency it occurs at, and a footer identity/disclaimer block.

        Mechanical mobility is a continuous frequency-response function, so the
        fiche presents it as a spectrum plus a table of characteristic points; a
        mobility measurement is a characterisation, so there is no pass/fail
        verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``specimen`` is the tested structure) and the
            footer identity; the ``requirement`` field is ignored.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            mobility fiche has a single body layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the mobility spectrum, so both are required
            (``pip install "phonometry[report,plot]"``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from ..._report.iso7626 import render_mobility_report

        return render_mobility_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def sdof_mobility_result(
    frequency: ArrayLike, mass: float, stiffness: float, damping: float
) -> MobilityResult:
    """SDOF driving-point mobility bundled as a :class:`MobilityResult`.

    :param frequency: Frequencies ``f``, in hertz (array).
    :param mass: Mass ``m``, in kg.
    :param stiffness: Stiffness ``k``, N/m.
    :param damping: Viscous damping ``c``, N.s/m.
    :return: The :class:`MobilityResult` (driving point).
    """
    freq = np.asarray(frequency, dtype=np.float64)
    y = sdof_mobility(freq, mass, stiffness, damping)
    return MobilityResult(frequencies=freq, mobility=y, driving_point=True)
