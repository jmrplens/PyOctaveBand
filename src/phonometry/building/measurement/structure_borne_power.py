#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Structure-borne sound power of building equipment (EN 15657:2018; ISO 9611).

Building service equipment (pumps, fans, boilers, sanitary appliances) injects
**structure-borne sound power** into the building structure it is fixed to.
EN 15657 measures it with the **reception-plate method**: the source is
mounted on a plate of known mass per unit area ``m`` and area ``S`` whose
structural loss factor ``eta`` is known, and the spatial-average vibratory
velocity level of the plate is measured.

The power a resonant plate dissipates equals
:math:`P = \omega\, \eta\, (m S)\, \langle v^2 \rangle`, so the power
**injected into that reception plate** is, in one-third-octave bands
(Formula 14):

.. math::

   L_{W\mathrm{s}} = 10 \log_{10}\frac{2 \pi f \eta\, m S}{f_0 m_0 S_0} + L_v - 60
   \qquad \text{dB re 1 pW}

with the references :math:`f_0 = 1` Hz, :math:`m_0 = 1` kg (printed as 1 kg;
the quantity it normalises is a mass per unit area, so the reference is
1 kg/m² -- see ``docs/ERRATA.md``), :math:`S_0 = 1` m²;
the fixed ``-60 dB`` term is
:math:`10 \log_{10}(v_0^2 / P_0)` for the EN 15657 velocity reference
:math:`v_0 = 10^{-9}` m/s and :math:`P_0 = 1` pW. The spatial mean velocity
level is the energetic average over the ``N`` plate positions (Formula 12):

.. math::

   L_v = 10 \log_{10}\!\left( \frac{1}{N} \sum 10^{L_{v,i}/10} \right)

and the plate loss factor follows from its structural reverberation time ``Ts``
(Formula 13, identical to the ISO 10848 total loss factor):

.. math::

   \eta = \frac{2.2}{f\, T_\mathrm{s}}

**Formula (14) is plate-specific, not a source descriptor.** The same source
injects a different power into a different receiver; feeding ``L_Ws`` directly
into EN 12354-5 as a characteristic level mis-states the receiving-room level
by up to ~20 dB. EN 15657 derives the plate-independent source quantities from
two test plates: a *low-mobility* plate (its point mobility and loss factor
are unchanged by the source) and a *high-mobility* plate (loaded by the
source):

- the **equivalent blocked force level** (Formula 15, dB re
  :math:`F_0 = 10^{-6}` N):
  :math:`L_{F\mathrm{b,eq}} = L_{Ws,low}
  - 10 \log_{10}( \operatorname{Re}\{Y_{R,low,eq}\} / Y_0 )`
  with the measured low-mobility-plate mobility and :math:`Y_0 = 1`
  m/(N.s);
- the **characteristic reception-plate power level** used by EN 12354-5
  (Formula 17), referred to the standard 10 cm concrete plate of
  characteristic mobility :math:`Y_{R,\infty,low} = 5 \cdot 10^{-6}`
  m/(N.s) (clause 7.2.4):
  :math:`L_{Wsn} = L_{F\mathrm{b,eq}} + 10 \log_{10}( Y_{R,\infty,low} / Y_0 )`;
- the **equivalent free velocity level** (Formula 18, dB re ``1e-9 m/s``)
  from the high-mobility plate, and the **source mobility** from both
  (Formula 19). ``L_Wsn`` plus the mobility corrections of EN 12354-5
  Annex I (see :mod:`phonometry.building.prediction.installed_structure_borne`) close
  the EN 15657 -> EN 12354-5 chain.

The source-side free velocity of ISO 9611:1996 is the direct-measurement
counterpart: velocity levels re :math:`v_0 = 5 \cdot 10^{-8}` m/s
(clause 7), averaged over
positions with the energy mean of its equation (9), implemented by
:func:`mean_free_velocity_level`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

    from ..._report.metadata import ReportMetadata


from ..._internal.validation import (
    require_per_band,
    require_positive,
    require_ranks,
    require_same_length,
)
from .flanking_transmission import total_loss_factor

#: EN 15657 vibratory velocity reference ``v0`` (= ISO 1683 10^-9 m/s), m/s.
REFERENCE_VELOCITY: float = 1.0e-9
#: Reference sound power ``P0``, W.
REFERENCE_SOUND_POWER: float = 1.0e-12
#: Reference mobility ``Y0`` of EN 15657 Formulae (15)/(17)/(18), m/(N.s).
REFERENCE_MOBILITY: float = 1.0
#: Reference force ``F0`` of the equivalent blocked force level (EN 15657), N.
REFERENCE_FORCE: float = 1.0e-6
#: Characteristic mobility ``Y_R,inf,low`` of the standard 10 cm concrete
#: reception plate (EN 15657:2018, clause 7.2.4), m/(N.s).
CHARACTERISTIC_PLATE_MOBILITY: float = 5.0e-6
#: ISO 9611:1996 free-velocity reference ``v0`` (clause 7), m/s.
FREE_VELOCITY_REFERENCE: float = 5.0e-8


def spatial_mean_velocity_level(levels: ArrayLike) -> float:
    r"""Spatial-average velocity level over the plate (EN 15657, Formula 12).

    :math:`L_v = 10 \log_{10}\left( \frac{1}{N} \sum 10^{L_{v,i}/10} \right)` --
    the energetic average of the
    per-position velocity levels.

    :param levels: Velocity levels ``L_v,i`` at the ``N`` positions, in dB.
    :return: The spatial mean velocity level, in dB.
    """
    lv = np.asarray(levels, dtype=np.float64)
    return float(10.0 * np.log10(np.mean(10.0 ** (0.1 * lv))))


def plate_loss_factor(
    frequency: ArrayLike, reverberation_time: ArrayLike
) -> np.ndarray:
    r"""Plate loss factor :math:`\eta = 2.2 / (f\, T_\mathrm{s})` (EN 15657, Formula 13).

    Estimated from the plate's structural reverberation time; identical to the
    ISO 10848 total loss factor.

    :param frequency: Band centre frequency ``f``, in hertz (per band).
    :param reverberation_time: Structural reverberation time ``Ts``, in s.
    :return: The loss factor ``eta`` (dimensionless) per band.
    """
    f = np.asarray(frequency, dtype=np.float64)
    ts = np.asarray(reverberation_time, dtype=np.float64)
    return total_loss_factor(f, ts)


def structure_borne_power_level(
    velocity_level: ArrayLike,
    frequency: ArrayLike,
    mass_per_area: float,
    area: float,
    loss_factor: ArrayLike,
    *,
    reference_velocity: float = REFERENCE_VELOCITY,
) -> np.ndarray:
    r"""Structure-borne power injected into the reception plate (EN 15657, Formula 14).

    The power a resonant reception plate dissipates, expressed as a level re
    1 pW:

    .. math::

       L_{W\mathrm{s}} = 10 \log_{10}\frac{2 \pi f \eta\, m S}{f_0 m_0 S_0} + L_v
       + 10 \log_{10}\frac{v_0^2}{P_0}

    With the EN 15657 reference :math:`v_0 = 10^{-9}` m/s the last term is
    -60 dB.

    .. note::
        This is the power injected into *this particular plate*, not a source
        descriptor: do **not** feed it into EN 12354-5 as the characteristic
        level ``L_Ws,c``. Convert it first via
        :func:`equivalent_blocked_force_level` (Formula 15) and
        :func:`characteristic_reception_plate_power` (Formula 17), then apply
        the EN 12354-5 Annex I mobility correction
        (:func:`phonometry.building.installed_power_from_reception_plate`).

    :param velocity_level: Spatial mean plate velocity level ``L_v`` (scalar or
        per band), in dB re ``v0``.
    :param frequency: Band centre frequency ``f``, in hertz.
    :param mass_per_area: Plate mass per unit area ``m``, in kg/m^2 (> 0).
    :param area: Plate area ``S``, in m^2 (> 0).
    :param loss_factor: Plate loss factor ``eta`` (scalar or per band, > 0).
    :param reference_velocity: Velocity reference ``v0`` (Default: 1e-9 m/s).
    :return: The structure-borne sound power level ``L_Ws``, in dB re 1 pW.
    :raises ValueError: for a non-positive mass, area, reference, frequency or
        loss factor.
    """
    mass_per_area = require_positive(mass_per_area, "mass_per_area")
    area = require_positive(area, "area")
    reference_velocity = require_positive(reference_velocity, "reference_velocity")
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f <= 0.0):
        msg = "'frequency' must be positive."
        raise ValueError(msg)
    lv = np.asarray(velocity_level, dtype=np.float64)
    eta = np.asarray(loss_factor, dtype=np.float64)
    if not np.all(np.isfinite(eta)) or np.any(eta <= 0.0):
        msg = "'loss_factor' must contain positive, finite values."
        raise ValueError(msg)
    offset = 10.0 * np.log10(reference_velocity**2 / REFERENCE_SOUND_POWER)
    lw = 10.0 * np.log10(2.0 * np.pi * f * eta * mass_per_area * area) + lv + offset
    return np.asarray(lw, dtype=np.float64)


@dataclass(frozen=True)
class StructureBornePowerResult:
    """Structure-borne sound power injected into a reception plate (EN 15657).

    The power level is specific to the measured plate; derive the
    plate-independent source quantities with
    :func:`equivalent_blocked_force_level` and
    :func:`characteristic_reception_plate_power` before using EN 12354-5.

    :ivar frequencies: Band centre frequencies, in hertz, or ``None``.
    :ivar power_level: Reception-plate injected power level ``L_Ws`` per band,
        in dB re 1 pW (Formula 14).
    :ivar velocity_level: Spatial mean plate velocity level ``L_v`` per band, dB.
    :ivar loss_factor: Plate loss factor ``eta`` per band.
    :ivar mass_per_area: Plate mass per unit area ``m``, in kg/m^2.
    :ivar area: Plate area ``S``, in m^2.
    """

    power_level: np.ndarray
    velocity_level: np.ndarray
    loss_factor: np.ndarray
    mass_per_area: float
    area: float
    frequencies: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject a plate result whose per-band quantities disagree.

        The EN 15657 fiche prints ``Lv``, ``eta`` and ``L_Ws`` as columns of
        one band table whose row count comes from ``L_Ws`` alone, and reads
        every other column at those same row indices. A companion array one
        entry longer is therefore not refused: whatever sits past the last row
        is dropped, and a surplus entry ahead of the others carries the whole
        column one band out of step with the labels beside it, so each printed
        value belongs to its neighbour's band. ``frequencies`` decides more
        than its own column: the band fraction is read off its first two
        entries and settles whether the sheet calls its rows octave or
        one-third-octave bands and rules them off in octave triplets, so a
        surplus entry at the head of it regroups every band the sheet shows.

        :raises ValueError: if any per-band quantity disagrees with the rest.
        """
        require_ranks(
            self,
            power_level=1,
            velocity_level=1,
            loss_factor=1,
            frequencies=1,
        )
        require_same_length(
            self, "power_level", "velocity_level", "loss_factor", "frequencies"
        )

    @property
    def total_level(self) -> float:
        r"""Band-summed power level :math:`10 \log_{10}(\sum 10^{0.1 L_{W\mathrm{s}}})`,
        in dB.
        """
        lw = np.asarray(self.power_level, dtype=np.float64)
        return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * lw))))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the characteristic structure-borne power level per band.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_structure_borne_power

        check_language(language)
        return plot_structure_borne_power(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an EN 15657 structure-borne sound power fiche to ``path``.

        Writes a one-page reception-plate characterization sheet: the
        standard-basis line naming the EN 15657:2018 reception-plate method
        (Formula 14), an optional metadata header (client, source equipment,
        test environment, instrumentation, climate, date), a per-band table
        (nominal octave/one-third-octave frequency, the spatial mean plate
        velocity level ``Lv`` and the injected structure-borne sound power
        level ``L_Ws``), the ``L_Ws(f)`` spectrum with a nominal band axis, the
        boxed band-summed total ``L_Ws`` (dB re 1 pW) with the plate mass per
        area ``m`` and area ``S``, an optional verdict row against a declared
        limit, and a basis strip stating Formula 14 and the conversion to the
        plate-independent source quantities (Formulae 15/17) required before
        EN 12354-5.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the source equipment,
            ``test_room`` the test environment, ``instrumentation``,
            ``temperature``, ``relative_humidity``, ``pressure``,
            ``test_date``), the footer identity (``laboratory``, ``operator``,
            ``report_id``, ``notes``) and, via ``requirement``, a declared
            upper limit on the total ``L_Ws`` (lower is better). The plate mass
            and area come from the result itself.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds the plate loss
            factor ``eta`` column.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"`` or ``language``
            is unknown.
        :raises ImportError: If reportlab (or, for the figure, matplotlib) is
            not installed (``pip install phonometry[report]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            raise ValueError(msg)
        from ..._report.en15657 import render_structure_borne_power_report

        return render_structure_borne_power_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def reception_plate_power(
    velocity_level: ArrayLike,
    frequency: ArrayLike,
    mass_per_area: float,
    area: float,
    *,
    loss_factor: ArrayLike | None = None,
    reverberation_time: ArrayLike | None = None,
) -> StructureBornePowerResult:
    r"""Reception-plate injected structure-borne sound power (EN 15657, clause 7).

    Provide the plate loss factor directly, or its structural reverberation
    time ``Ts`` (from which :math:`\eta = 2.2/(f\, T_\mathrm{s})` is computed,
    Formula 13).
    The result is the power injected into *this* plate (Formula 14); see the
    module docstring for the conversion chain to the EN 12354-5 source
    quantities.

    :param velocity_level: Spatial mean plate velocity level ``L_v`` (per band),
        in dB re 1e-9 m/s (see :func:`spatial_mean_velocity_level`).
    :param frequency: Band centre frequencies ``f``, in hertz.
    :param mass_per_area: Plate mass per unit area ``m``, in kg/m^2 (> 0).
    :param area: Plate area ``S``, in m^2 (> 0).
    :param loss_factor: Plate loss factor ``eta`` (per band), or ``None`` to
        derive it from ``reverberation_time``.
    :param reverberation_time: Structural reverberation time ``Ts``, in s, used
        when ``loss_factor`` is ``None``.
    :return: The :class:`StructureBornePowerResult`.
    :raises ValueError: if neither ``loss_factor`` nor ``reverberation_time``
        is given, or if ``velocity_level`` or ``loss_factor`` is neither a
        scalar nor one value per band.
    """
    freq = np.atleast_1d(np.asarray(frequency, dtype=np.float64))
    lv = require_per_band(velocity_level, "velocity_level", freq)
    if loss_factor is not None:
        eta = require_per_band(loss_factor, "loss_factor", freq)
    elif reverberation_time is not None:
        eta = np.asarray(plate_loss_factor(freq, reverberation_time), dtype=np.float64)
    else:
        msg = "provide either 'loss_factor' or 'reverberation_time'."
        raise ValueError(msg)
    lw = structure_borne_power_level(lv, freq, mass_per_area, area, eta)
    return StructureBornePowerResult(
        power_level=np.asarray(lw, dtype=np.float64),
        velocity_level=lv,
        loss_factor=eta,
        mass_per_area=float(mass_per_area),
        area=float(area),
        frequencies=freq,
    )


def equivalent_blocked_force_level(
    power_level: ArrayLike,
    plate_mobility: ArrayLike,
) -> np.ndarray:
    r"""Equivalent blocked force level, squared (EN 15657:2018, Formula 15).

    :math:`L_{F\mathrm{b,eq}} = L_{Ws,low}
    - 10 \log_{10}(\operatorname{Re}\{Y_{R,low,eq}\}/Y_0)`
    in dB re :math:`F_0 = 10^{-6}` N,
    from the power injected into the *low-mobility* reception plate
    (Formula 14) and the equivalent point mobility of that plate (the
    arithmetic mean of Re{Y} over the contact points, Formula 16).

    :param power_level: Injected power level ``L_Ws,low`` (per band), in dB re
        1 pW (see :func:`structure_borne_power_level`).
    :param plate_mobility: Equivalent plate mobility ``Y_R,low,eq`` (per
        band), in m/(N.s); complex values use their real part (Formula 16).
    :return: The equivalent blocked force level ``L_Fb,eq``, in dB re 1e-6 N.
    :raises ValueError: if ``Re{plate_mobility}`` is not positive and finite.
    """
    lw = np.asarray(power_level, dtype=np.float64)
    y = np.asarray(plate_mobility, dtype=np.complex128)
    y_re = np.real(y)
    if (
        not np.all(np.isfinite(y.real))
        or not np.all(np.isfinite(y.imag))
        or np.any(y_re <= 0.0)
    ):
        msg = "'plate_mobility' must be finite with a positive real part."
        raise ValueError(msg)
    return np.asarray(lw - 10.0 * np.log10(y_re / REFERENCE_MOBILITY), dtype=np.float64)


def characteristic_reception_plate_power(
    blocked_force_level: ArrayLike,
    *,
    characteristic_mobility: float = CHARACTERISTIC_PLATE_MOBILITY,
) -> np.ndarray:
    r"""Characteristic reception-plate power level (EN 15657:2018, Formula 17).

    :math:`L_{Wsn} = L_{F\mathrm{b,eq}} + 10 \log_{10}(|Y_{R,\infty,low}|/Y_0)`
    with the characteristic mobility of the standard 10 cm concrete reception
    plate :math:`Y_{R,\infty,low} = 5 \cdot 10^{-6}` m/(N.s) (clause 7.2.4) and
    :math:`Y_0 = 1` m/(N.s). This is the plate-independent source power level
    ``L_Ws,n`` that EN 12354-5 consumes (its Annex I mobility correction
    :func:`phonometry.building.installed_power_from_reception_plate` then
    refers it to the actual receiver).

    :param blocked_force_level: Equivalent blocked force level ``L_Fb,eq``
        (per band), in dB re 1e-6 N (see
        :func:`equivalent_blocked_force_level`).
    :param characteristic_mobility: ``Y_R,inf,low``, in m/(N.s)
        (Default: 5e-6, clause 7.2.4).
    :return: The characteristic power level ``L_Wsn``, in dB re 1 pW.
    :raises ValueError: for a non-positive characteristic mobility.
    """
    characteristic_mobility = require_positive(
        characteristic_mobility, "characteristic_mobility"
    )
    lf = np.asarray(blocked_force_level, dtype=np.float64)
    return np.asarray(
        lf + 10.0 * np.log10(characteristic_mobility / REFERENCE_MOBILITY),
        dtype=np.float64,
    )


def equivalent_free_velocity_level(
    power_level: ArrayLike,
    plate_mobility: ArrayLike,
) -> np.ndarray:
    r"""Equivalent free velocity level of the source (EN 15657:2018, Formula 18).

    :math:`L_{vf,eq} = L_{Ws,high} + 10 \log_{10}\left( |Y_{R,high,eq}|^2 /
    (\operatorname{Re}\{Y_{R,high,eq}\}\, Y_0) \right) + 60~\text{dB}`
    in dB re 1e-9 m/s, from the power injected into the
    *high-mobility* reception plate and its equivalent (complex) point
    mobility. The plus sign follows the printed formula and the physics
    (:math:`v_f^2 = P |Y|^2 / \operatorname{Re}\{Y\}`); it is what makes
    Formulae (15) and (18)
    combine through Formula (19) into :math:`|Y_{S,eq}| = v_f / F_b`.

    :param power_level: Injected power level ``L_Ws,high`` (per band), in dB
        re 1 pW.
    :param plate_mobility: Equivalent plate mobility ``Y_R,high,eq`` (per
        band, complex), in m/(N.s).
    :return: The equivalent free velocity level ``L_vf,eq``, in dB re 1e-9 m/s.
    :raises ValueError: if ``Re{plate_mobility}`` is not positive and finite.
    """
    lw = np.asarray(power_level, dtype=np.float64)
    y = np.asarray(plate_mobility, dtype=np.complex128)
    y_re = np.real(y)
    if (
        not np.all(np.isfinite(y.real))
        or not np.all(np.isfinite(y.imag))
        or np.any(y_re <= 0.0)
    ):
        msg = "'plate_mobility' must be finite with a positive real part."
        raise ValueError(msg)
    term = np.abs(y) ** 2 / (y_re * REFERENCE_MOBILITY)
    return np.asarray(lw + 10.0 * np.log10(term) + 60.0, dtype=np.float64)


def source_mobility_from_levels(
    free_velocity_level: ArrayLike,
    blocked_force_level: ArrayLike,
) -> np.ndarray:
    r"""Equivalent source mobility magnitude (EN 15657:2018, Formula 19).

    :math:`|Y_{S,eq}|^2 / Y_0^2 = 10^{(L_{vf,eq} - L_{F\mathrm{b,eq}})/10} \cdot
    10^{-6}`, the ratio
    of the free-velocity (re 1e-9 m/s) and blocked-force (re 1e-6 N)
    references makes the constant
    :math:`(10^{-9}/10^{-6})^2 = 10^{-6}`.

    :param free_velocity_level: Equivalent free velocity level ``L_vf,eq``
        (per band), in dB re 1e-9 m/s (Formula 10 or 18).
    :param blocked_force_level: Equivalent blocked force level ``L_Fb,eq``
        (per band), in dB re 1e-6 N (Formula 15).
    :return: The source mobility magnitude ``|Y_S,eq|``, in m/(N.s).
    """
    lv = np.asarray(free_velocity_level, dtype=np.float64)
    lf = np.asarray(blocked_force_level, dtype=np.float64)
    return np.asarray(
        REFERENCE_MOBILITY * np.sqrt(10.0 ** ((lv - lf) / 10.0) * 1.0e-6),
        dtype=np.float64,
    )


def mean_free_velocity_level(levels: ArrayLike) -> float:
    r"""Mean free velocity level over positions (ISO 9611:1996, equation (9)).

    :math:`\overline{L}_{vx} = 10 \log_{10}\left[ \frac{1}{N} \sum 10^{L_{vxi}/10}
    \right]`, the energy mean of the
    free-velocity levels measured at the ``N`` contact/attachment points of
    one direction ``x``, each in dB re the ISO 9611 free-velocity reference
    :math:`v_0 = 5 \cdot 10^{-8}` m/s (clause 7). The arithmetic is the
    energetic average
    (identical to EN 15657 Formula 12, :func:`spatial_mean_velocity_level`);
    only the reference differs.

    :param levels: Free velocity levels ``Lvxi`` at the ``N`` positions, in
        dB re 5e-8 m/s.
    :return: The mean free velocity level, in dB re 5e-8 m/s.
    """
    return spatial_mean_velocity_level(levels)
