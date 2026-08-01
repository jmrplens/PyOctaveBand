#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Dynamic transfer stiffness of resilient elements (ISO 10846-1/-2/-3).

The vibro-acoustic transfer property of a resilient element (a vibration
isolator, mount, bellows or hose) is its **dynamic transfer stiffness**: the
frequency-dependent ratio of the *blocking force* phasor ``F2,b`` on the output
(receiver) side to the displacement phasor ``u1`` on the input (source) side,
with the output blocked (ISO 10846-1, 3.7), in N/m:

.. math::

   k_{2,1} = \frac{F_{2,b}}{u_1}

For an isolator between two structures of large driving-point stiffness, the
force delivered to the receiver approximates this blocking force (ISO 10846-1,
Equation 7), so :math:`k_{2,1}` characterises the isolator's transmission.
Results are reported as a **level**, in dB, re the reference stiffness
:math:`k_0 = 1` N/m (ISO 10846-2 and -3, 3.17):

.. math::

   L_k = 10 \lg\!\left( \frac{|k_{2,1}|^2}{k_0^2} \right)
   = 20 \lg\!\left( \frac{|k_{2,1}|}{k_0} \right)

and, in the low-frequency range where inertial forces in the element are
negligible, the **loss factor** is the tangent of the phase angle of
:math:`k_{2,1}` (ISO 10846-1, 3.8):
:math:`\eta = \operatorname{Im}(k_{2,1}) / \operatorname{Re}(k_{2,1})`.

Two laboratory methods determine :math:`k_{2,1}`:

* **Direct method** (ISO 10846-2): measure the blocked output force ``F2,b``
  and the input displacement ``u1`` directly:
  :math:`k_{2,1} = F_{2,b} / u_1`.
* **Indirect method** (ISO 10846-3): load the output with a compact blocking
  mass ``m2`` and measure the vibration transmissibility
  :math:`T = u_2/u_1`; the blocking force is the mass's inertia force
  (ISO 10846-3, Equation 1):
  :math:`k_{2,1} = -(2\pi f)^2 (m_2 + m_f) T` for :math:`T \ll 1`,
  where ``mf`` is the mass of the output flange of the test element. The
  approximation is valid only where :math:`|T| \le 0.1` (Inequality (2):
  :math:`\Delta L_{1,2} \ge 20` dB) and while the blocking mass still
  behaves rigidly, :math:`10 \lg(m_{2,\mathrm{eff}}^2/m_2^2) \le 1` dB
  (Inequality (3)); see :func:`transfer_stiffness_indirect`.

The dynamic transfer stiffness is a member of the frequency-response-function
family (ISO 10846-1, Annex A / Table A.2):
:math:`k = j\omega Z = -\omega^2 m_{\mathrm{eff}}`,
so it converts to mechanical impedance and effective mass through
:func:`phonometry.convert_frf` (``"dynamic_stiffness"`` <-> ``"impedance"`` <->
``"apparent_mass"``). This module feeds the structure-borne source and building
prediction standards (ISO 9611, EN 15657, EN 12354-5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

from numpy.typing import ArrayLike, NDArray

from .._internal.validation import require_non_negative, require_positive
from .._internal.warnings import PhonometryWarning
from .mechanical_mobility import convert_frf

#: Reference dynamic stiffness for the level ``L_k`` (ISO 10846-2/-3, 3.17), N/m.
REFERENCE_STIFFNESS: float = 1.0

#: Validity limit on the vibration transmissibility magnitude for the indirect
#: method (ISO 10846-3:2002, 6.1, Inequality (2)): measurements are valid only
#: where ``DeltaL1,2 = La1 - La2 >= 20 dB``, i.e. ``|T| <= 0.1``, which keeps
#: the ``T << 1`` approximation of Formula (1) accurate within 1 dB (12 %).
TRANSMISSIBILITY_LIMIT: float = 0.1


def _omega(frequency: ArrayLike) -> NDArray[np.float64]:
    r"""Angular frequency :math:`\omega = 2\pi f` (rad/s); rejects f <= 0."""
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("'frequency' must be positive.")
    return 2.0 * np.pi * f


def transfer_stiffness_level(
    stiffness: ArrayLike, *, reference: float = REFERENCE_STIFFNESS
) -> np.ndarray:
    r"""Level of the dynamic transfer stiffness (ISO 10846-2/-3, 3.17).

    :math:`L_k = 20 \lg(|k_{2,1}| / k_0)` dB, with ``k0`` the reference
    stiffness.

    :param stiffness: Dynamic transfer stiffness :math:`k_{2,1}` (complex or
        real, scalar or array, non-zero), in N/m.
    :param reference: Reference stiffness ``k0`` (Default: 1 N/m), in N/m.
    :return: The level ``L_k``, in dB re ``k0``.
    :raises ValueError: for a non-positive reference or a zero stiffness
        magnitude (a dead channel has no level).
    """
    reference = require_positive(reference, "reference")
    magnitude = np.abs(np.asarray(stiffness, dtype=np.complex128))
    if not np.all(magnitude > 0.0):
        raise ValueError(
            "'stiffness' contains zero magnitudes; a zero (dead-channel) "
            "stiffness has no level."
        )
    return np.asarray(20.0 * np.log10(magnitude / reference), dtype=np.float64)


def loss_factor(stiffness: ArrayLike) -> np.ndarray:
    r"""Loss factor
    :math:`\eta = \operatorname{Im}(k_{2,1}) / \operatorname{Re}(k_{2,1})`
    (ISO 10846-1, 3.8).

    Valid in the low-frequency range where inertial forces in the element are
    negligible; it is the tangent of the phase angle of the transfer stiffness.

    :param stiffness: Dynamic transfer stiffness :math:`k_{2,1}` (complex,
        scalar or array, with a non-zero real part), in N/m.
    :return: The loss factor ``eta`` (dimensionless).
    :raises ValueError: for a purely imaginary stiffness
        (:math:`\operatorname{Re}(k_{2,1}) = 0`), for which the loss factor
        is undefined.
    """
    k = np.asarray(stiffness, dtype=np.complex128)
    if not np.all(np.abs(k.real) > 0.0):
        raise ValueError(
            "'stiffness' contains purely imaginary values (Re = 0); the loss "
            "factor eta = Im/Re is undefined there."
        )
    return np.asarray(k.imag / k.real, dtype=np.float64)


def transfer_stiffness_direct(
    blocking_force: ArrayLike, input_displacement: ArrayLike
) -> np.ndarray:
    r"""Dynamic transfer stiffness by the direct method (ISO 10846-2).

    :math:`k_{2,1} = F_{2,b} / u_1`, the blocked output force phasor over
    the input displacement phasor.

    :param blocking_force: Blocked output force phasor ``F2,b`` (complex), in N.
    :param input_displacement: Input displacement phasor ``u1`` (complex,
        non-zero), in m.
    :return: The dynamic transfer stiffness :math:`k_{2,1}`, in N/m.
    :raises ValueError: for a zero input displacement (dead input channel).
    """
    f2b = np.asarray(blocking_force, dtype=np.complex128)
    u1 = np.asarray(input_displacement, dtype=np.complex128)
    if not np.all(np.abs(u1) > 0.0):
        raise ValueError(
            "'input_displacement' contains zeros (dead input channel); the "
            "ratio k2,1 = F2,b/u1 is undefined there."
        )
    return np.asarray(f2b / u1, dtype=np.complex128)


def transfer_stiffness_indirect(
    frequency: ArrayLike,
    transmissibility: ArrayLike,
    blocking_mass: float,
    *,
    flange_mass: float = 0.0,
) -> np.ndarray:
    r"""Dynamic transfer stiffness by the indirect method (ISO 10846-3, Eq. 1).

    :math:`k_{2,1} = -(2\pi f)^2 (m_2 + m_f) T`: the blocking force is the
    inertia force of a compact blocking mass ``m2`` (plus the output flange
    mass ``mf``), derived from the measured vibration transmissibility
    :math:`T = u_2/u_1`. Valid for :math:`T \ll 1` (i.e. well above the
    mass/spring resonance).

    **Validity (ISO 10846-3, clause 6).** The :math:`T \ll 1` approximation
    of Formula (1) is required accurate within 1 dB, i.e. within 12 % of the
    calculated stiffness magnitude. This holds only where Inequality (2) is
    met: :math:`\Delta L_{1,2} = L_{a1} - L_{a2} \ge 20` dB, i.e.
    :math:`|T| \le 0.1` (:data:`TRANSMISSIBILITY_LIMIT`). Bands with
    ``|T|`` above that limit
    (routine near or below the mass/spring resonance) trigger a
    :class:`~phonometry.PhonometryWarning`; treat those bands as outside the
    valid frequency range of the test arrangement. The upper frequency limit
    ``f3`` additionally requires the blocking mass to vibrate as a rigid
    body: results are valid only while its effective mass ``m2,eff``,
    measured per Formula (4) as
    :math:`m_{2,\mathrm{eff}} = 2 F_2 / (a'_1 + a''_1)` (two accelerometers
    spaced :math:`D = \sqrt{S}` across the contact area), stays within 1 dB
    of the rigid mass, :math:`10 \lg(m_{2,\mathrm{eff}}^2 / m_2^2) \le 1` dB
    (Inequality (3), 6.2.3).

    :param frequency: Frequency ``f``, in hertz (scalar or array).
    :param transmissibility: Vibration transmissibility
        :math:`T = u_2/u_1` (complex,
        scalar or array; velocity and acceleration ratios have the same value).
    :param blocking_mass: Blocking mass ``m2``, in kg (> 0).
    :param flange_mass: Output-flange mass ``mf``, in kg (Default: 0.0).
    :return: The dynamic transfer stiffness :math:`k_{2,1}`, in N/m.
    :raises ValueError: for a non-positive frequency or blocking mass.
    :warns PhonometryWarning: where any :math:`\lvert T\rvert > 0.1`
        (Inequality (2) violated).
    """
    blocking_mass = require_positive(blocking_mass, "blocking_mass")
    flange_mass = require_non_negative(flange_mass, "flange_mass")
    omega = _omega(frequency)
    t = np.asarray(transmissibility, dtype=np.complex128)
    magnitude = np.abs(t)
    if np.any(magnitude > TRANSMISSIBILITY_LIMIT):
        worst = float(np.max(magnitude))
        warnings.warn(
            f"|T| up to {worst:.3g} exceeds {TRANSMISSIBILITY_LIMIT:g} "
            "(DeltaL1,2 < 20 dB): ISO 10846-3 Inequality (2) is violated and "
            "the T << 1 approximation of Formula (1) is no longer accurate "
            "within 1 dB (12 %); those bands lie outside the valid frequency "
            "range of the indirect method.",
            PhonometryWarning,
            stacklevel=2,
        )
    return np.asarray(
        -(omega**2) * (blocking_mass + flange_mass) * t, dtype=np.complex128
    )


def blocking_force_ratio(
    driving_point_stiffness: ArrayLike, termination_stiffness: ArrayLike
) -> np.ndarray:
    r"""Ratio of the delivered force to the blocking force (ISO 10846-1, Eq. 6).

    For an isolator driving a receiving structure, the output force for a
    given source displacement ``u1`` is
    :math:`F_2 = k_{2,1} u_1 / (1 + k_{2,2}/k_t)`
    (Equation (6)), where ``k2,2`` is the isolator's output driving-point
    stiffness (output blocked at the input) and ``kt`` the dynamic
    driving-point stiffness of the termination. This function returns

    .. math::

       \frac{F_2}{F_{2,b}} = \frac{1}{1 + k_{2,2}/k_t}

    the factor by which the delivered force deviates from the blocking force
    :math:`F_{2,b} = k_{2,1} u_1` of Equation (7). For
    :math:`|k_{2,2}| < 0.1 |k_t|` the ratio is within 10 % of unity
    (:math:`1/1.1 = 0.909` at the limit), which is the
    stiffness mismatch that justifies characterising an isolator by its
    blocked transfer stiffness alone.

    :param driving_point_stiffness: Output driving-point stiffness ``k2,2`` of
        the isolator (complex, scalar or array), in N/m.
    :param termination_stiffness: Driving-point stiffness ``kt`` of the
        receiving structure (complex, scalar or array, non-zero), in N/m.
    :return: The complex ratio ``F2/F2,b``.
    :raises ValueError: for a zero termination stiffness.
    """
    k22 = np.asarray(driving_point_stiffness, dtype=np.complex128)
    kt = np.asarray(termination_stiffness, dtype=np.complex128)
    if not np.all(np.abs(kt) > 0.0):
        raise ValueError("'termination_stiffness' must be non-zero.")
    return np.asarray(1.0 / (1.0 + k22 / kt), dtype=np.complex128)


def base_transmissibility(
    frequency: ArrayLike, mass: float, stiffness: float, damping: float = 0.0
) -> np.ndarray:
    r"""Transmissibility of a mass on an ideal resilient element (model).

    The output mass ``m`` on a massless Kelvin-Voigt element (spring ``k`` in
    parallel with a viscous damper ``c``) driven at the input has the
    base-excitation transmissibility

    .. math::

       T = \frac{u_2}{u_1}
       = \frac{k + j\omega c}{k - \omega^2 m + j\omega c}

    This ideal-element model is the counterpart of the indirect-method test
    arrangement (ISO 10846-3): feeding ``T`` into
    :func:`transfer_stiffness_indirect` with the same mass recovers the
    element's transfer stiffness :math:`k + j\omega c` in the high-frequency
    limit :math:`T \ll 1`.

    :param frequency: Frequency ``f``, in hertz (scalar or array).
    :param mass: Output mass ``m``, in kg.
    :param stiffness: Element stiffness ``k``, in N/m.
    :param damping: Viscous damping ``c``, in N.s/m (Default: 0.0).
    :return: The complex transmissibility ``T``.
    """
    mass = require_positive(mass, "mass")
    stiffness = require_positive(stiffness, "stiffness")
    damping = require_non_negative(damping, "damping")
    omega = _omega(frequency)
    numerator = stiffness + 1j * omega * damping
    denominator = stiffness - omega**2 * mass + 1j * omega * damping
    return np.asarray(numerator / denominator, dtype=np.complex128)


@dataclass(frozen=True)
class TransferStiffnessResult:
    """A dynamic transfer stiffness over frequency (ISO 10846).

    :ivar frequencies: Frequencies, in hertz.
    :ivar transfer_stiffness: Complex :math:`k_{2,1}` per frequency, in N/m.
    :ivar blocking_mass: Blocking mass ``m2`` used (indirect method), in kg, or
        ``None`` for the direct method.
    """

    frequencies: np.ndarray
    transfer_stiffness: np.ndarray
    blocking_mass: float | None = None

    @property
    def magnitude(self) -> np.ndarray:
        r"""Transfer-stiffness magnitude :math:`|k_{2,1}|`, in N/m."""
        return np.asarray(np.abs(self.transfer_stiffness), dtype=np.float64)

    @property
    def level(self) -> np.ndarray:
        """Transfer-stiffness level ``L_k`` re 1 N/m, in dB (3.17)."""
        return transfer_stiffness_level(self.transfer_stiffness)

    @property
    def loss_factor(self) -> np.ndarray:
        r"""Loss factor :math:`\eta = \operatorname{Im}/\operatorname{Re}`
        per frequency (3.8)."""
        return loss_factor(self.transfer_stiffness)

    def to(self, target: str) -> np.ndarray:
        r"""Convert :math:`k_{2,1}` to a related FRF (ISO 10846-1 Annex A).

        ``target`` is ``"impedance"`` (:math:`Z = k/(j\omega)`) or
        ``"apparent_mass"`` (:math:`m_{\mathrm{eff}} = -k/\omega^2`); see
        :func:`phonometry.convert_frf`.
        """
        return convert_frf(
            self.transfer_stiffness, self.frequencies, "dynamic_stiffness", target
        )

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the transfer-stiffness level ``L_k(f)``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.vibration import plot_transfer_stiffness

        return plot_transfer_stiffness(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        r"""Render a dynamic-transfer-stiffness fiche to a PDF (ISO 10846).

        Writes a one-page transfer-stiffness characterisation report for a
        resilient element: the standard-basis line naming the determination
        method (direct, ISO 10846-2:2008, or indirect blocking-mass,
        ISO 10846-3:2002; definition per ISO 10846-1:2008), an optional metadata
        header, a two-panel body with a compact table of the FRF's
        characteristic points (the method, the blocking mass for the indirect
        method, the frequency range, and the low-frequency stiffness plateau
        :math:`|k_{2,1}|`, its level ``L_k`` and the loss factor ``eta``
        there) beside
        the transfer-stiffness level spectrum ``L_k(f)``, a boxed low-frequency
        ``L_k`` with the stiffness magnitude and method alongside, and a footer
        identity/disclaimer block.

        Dynamic transfer stiffness is a continuous frequency-response function,
        so the fiche presents it as a spectrum plus a table of characteristic
        points; a transfer-stiffness determination is a characterisation, so
        there is no pass/fail verdict.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header identity (``specimen`` is the tested resilient element)
            and the footer identity; the ``requirement`` field is ignored.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            transfer-stiffness fiche has a single body layout, so it has no
            effect.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab or matplotlib is not installed. The
            fiche always embeds the ``L_k(f)`` spectrum, so both are required
            (``pip install "phonometry[report,plot]"``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.iso10846 import render_transfer_stiffness_report

        return render_transfer_stiffness_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def indirect_transfer_stiffness_result(
    frequency: ArrayLike,
    transmissibility: ArrayLike,
    blocking_mass: float,
    *,
    flange_mass: float = 0.0,
) -> TransferStiffnessResult:
    r"""Indirect-method transfer stiffness bundled as a :class:`TransferStiffnessResult`.

    See :func:`transfer_stiffness_indirect` for the ISO 10846-3 validity
    conditions (Inequalities (2) and (3)); bands with :math:`|T| > 0.1`
    trigger a :class:`~phonometry.PhonometryWarning`.

    :param frequency: Frequencies ``f``, in hertz (array).
    :param transmissibility: Vibration transmissibility
        :math:`T = u_2/u_1` (complex).
    :param blocking_mass: Blocking mass ``m2``, in kg (> 0).
    :param flange_mass: Output-flange mass ``mf``, in kg (Default: 0.0).
    :return: The :class:`TransferStiffnessResult` (indirect method).
    :warns PhonometryWarning: where any :math:`\lvert T\rvert > 0.1`
        (Inequality (2) violated).
    """
    freq = np.asarray(frequency, dtype=np.float64)
    k = transfer_stiffness_indirect(
        freq, transmissibility, blocking_mass, flange_mass=flange_mass
    )
    return TransferStiffnessResult(
        frequencies=freq, transfer_stiffness=k, blocking_mass=float(blocking_mass)
    )
