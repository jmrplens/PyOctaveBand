#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Installed structure-borne sound from service equipment (EN 12354-5:2009).

EN 12354-5 predicts the sound pressure level in a receiving room caused by
building service equipment that injects **structure-borne sound** into the
building. The chain closes the structural-vibroacoustics series:

1. The source strength is its *characteristic structure-borne sound power level*
   ``L_Ws,c``. It is **not** the raw reception-plate power of EN 15657
   Formula (14): that plate-injected level must first be converted to the
   plate-independent ``L_Ws,n`` (EN 15657 Formulae (15)/(17); see
   :mod:`phonometry.building.measurement.structure_borne_power`) and then referred to the
   actual receiver with the Annex I mobility correction
   (:func:`installed_power_from_reception_plate`),
   :math:`L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,n}} + 10 \log_{10}( Y_{\infty,i} / Y_{\infty,\mathrm{rec}} )`
   with the reference plate mobility
   :math:`Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}` m/(N.s), or equivalently to the
   characteristic level
   :math:`L_{W\mathrm{s,c}} = L_{W\mathrm{s,n}} + 10 \log_{10}( Y_\mathrm{s} / Y_{\infty,\mathrm{rec}} )` with the
   source mobility (Annex I.3, Table I.8), from which ``D_C`` is subtracted.
2. Only part of that power is actually injected into the supporting element; the
   loss is the **coupling term** ``D_C`` (clause 4.4.3), positive in the usual
   mobility-mismatched cases (see :func:`coupling_term` for the exception),
   set by the source mobility ``Y_s`` and the receiver mobility ``Y_i``
   (Formula 19b):
   :math:`D_{\mathrm{C},i} = 10 \log_{10}\left( |Y_\mathrm{s} + Y_i|^2 / (|Y_\mathrm{s}|
   \operatorname{Re}\{Y_i\}) \right)`, which reduces to
   :math:`10 \log_{10}( |Y_\mathrm{s}| / \operatorname{Re}\{Y_i\} )` for a force source
   (high source mobility,
   Formula 19c) and to :math:`-10 \log_{10}( |Y_\mathrm{s}| \operatorname{Re}\{Z_i\} )`
   for a velocity source (low
   source mobility, Formula 19d). An elastic support adds its transfer
   mobility ``Y_k`` inside the modulus (Formula 19e).
3. The **installed** power level is then
   :math:`L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,c}} - D_{\mathrm{C},i}`
   (Formula 18b).
4. The normalised sound pressure level in the receiving room for one path (i->j)
   follows from the installed power, the structure-to-airborne adjustment term
   ``D_sa`` (clause 4.4.4), the flanking sound reduction index ``R_ij,ref`` and
   the element area (Formula 18a):
   :math:`L_{\mathrm{n,s},ij} = L_{W\mathrm{s,inst},i} - D_{\mathrm{sa},i} - R_{ij,\mathrm{ref}}
   - 10 \log_{10}(S_i/S_0) - 10 \log_{10}(A_0/4)`
   with :math:`S_0 = A_0 = 10` m²; the paths combine energetically
   (Formula 17).

The source and receiver mobilities/impedances are those of
:mod:`phonometry.vibration.structural.mechanical_mobility` and :mod:`phonometry.vibration.structural.transfer_stiffness`.

**The informative tables.** Every term of that chain is a number the user
would otherwise copy out of the standard, so the two tables of the informative
annexes are here as named lookups:

- **Table D.1** (Annex D) estimates the mobility of typical construction
  elements from their own dimensions, which is how clause D.1.3 builds up a
  *source* mobility ``Y_s`` for step 2 out of the machine's mass, feet,
  chassis panels and pipework: :func:`typical_element_mobility`, whose
  ``structure`` argument is the table's first column and whose keywords are
  its second (:data:`TABLE_D1_QUANTITIES`).
- **Table F.1** (Annex F) gives the octave-band force level ``L_F`` of the ISO
  tapping machine, the substitution source of clause D.1.2.3:
  :func:`tapping_machine_force_level`, with
  :func:`tapping_machine_characteristic_power_level` and
  :func:`tapping_machine_coupling_term` turning it into the ``L_Ws,c`` and
  ``D_C`` of step 3.

Annex F also supplies the two terms step 4 takes: the adjustment term ``D_sa``
of Formula (F.3) (:func:`structure_to_airborne_adjustment`) and the
multi-junction adjustment ``dK`` of clause F.1
(:func:`multi_junction_adjustment`), which the flanking reduction index
``R_ij,ref`` of a path more than one junction away is built with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

from numpy.typing import ArrayLike

from ..._internal.validation import (
    require_choice,
    require_non_negative,
    require_positive,
)
from .resilient_layers import TAPPING_HAMMER_MASS

#: Reference area ``S0 = A0`` of EN 12354-5 (Formula 18a), m^2.
REFERENCE_AREA: float = 10.0

#: Characteristic mobility ``Y_inf,rec`` of the EN 15657 reference reception
#: plate (10 cm concrete; EN 12354-5 Annex I / EN 15657:2018 7.2.4), m/(N.s).
REFERENCE_PLATE_MOBILITY: float = 5.0e-6


def _positive_real_part(values: ArrayLike, name: str) -> np.ndarray:
    """Validate that the real part of a mobility/impedance is positive."""
    arr = np.asarray(values, dtype=np.complex128)
    re = np.real(arr)
    if (
        not np.all(np.isfinite(arr.real))
        or not np.all(np.isfinite(arr.imag))
        or np.any(re <= 0.0)
    ):
        raise ValueError(
            f"'{name}' must be finite with a positive real part (a passive "
            "receiver dissipates power)."
        )
    return arr


def _positive_values(values: ArrayLike, name: str) -> np.ndarray:
    """Validate finite, strictly positive reals, preserving the input shape.

    The shape-preserving sibling of
    :func:`phonometry._internal.validation.require_positive_array`: the rest of
    this module returns a scalar for a scalar input, and the Annex D and
    Annex F lookups have to do the same.

    :raises ValueError: for a non-finite or non-positive value.
    """
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"'{name}' must be finite and strictly positive.")
    return arr


def _nonzero_magnitude(values: ArrayLike, name: str) -> np.ndarray:
    """Validate a finite, non-zero (complex) mobility magnitude."""
    arr = np.asarray(values, dtype=np.complex128)
    mag = np.abs(arr)
    if not np.all(np.isfinite(mag)) or not np.all(mag > 0.0):
        raise ValueError(f"'{name}' must be finite and non-zero.")
    return arr


def coupling_term(
    source_mobility: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    transfer_mobility: ArrayLike = 0.0,
) -> np.ndarray:
    r"""Coupling term ``D_C`` for a point excitation (EN 12354-5, Formula 19b/19e).

    :math:`D_\mathrm{C} = 10 \log_{10}\left( |Y_\mathrm{s} + Y_i + Y_k|^2 / (|Y_\mathrm{s}|
    \operatorname{Re}\{Y_i\}) \right)` -- the loss between
    the characteristic and the injected structure-borne power. ``Y_k`` is the
    transfer mobility of an elastic support (Formula 19e; 0 for a rigid
    connection, Formula 19b).

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero),
        in m/(N.s).
    :param receiver_mobility: Receiver point mobility ``Y_i`` (complex,
        positive real part).
    :param transfer_mobility: Elastic-support transfer mobility ``Y_k``
        (Default: 0.0).
    :return: The coupling term ``D_C``, in dB. Positive whenever the source
        and receiver mobilities are well mismatched (the usual installed
        case), but **not** guaranteed non-negative: near a mounting
        resonance where ``Y_s`` and ``Y_i`` are of comparable magnitude and
        opposite phase the numerator
        :math:`\lvert Y_\mathrm{s} + Y_i \rvert^2` collapses and ``D_C``
        goes negative (the installed power then exceeds the characteristic
        level; e.g. :math:`Y_\mathrm{s} = j \cdot 10^{-4}`,
        :math:`Y_i = 10^{-5} - j \cdot 10^{-4}` m/(N·s) gives
        :math:`D_\mathrm{C} \approx -10` dB).
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Y_i}`` is not
        positive and finite.
    """
    ys = _nonzero_magnitude(source_mobility, "source_mobility")
    yi = _positive_real_part(receiver_mobility, "receiver_mobility")
    yk = np.asarray(transfer_mobility, dtype=np.complex128)
    numerator = np.abs(ys + yi + yk) ** 2
    denominator = np.abs(ys) * np.real(yi)
    return np.asarray(10.0 * np.log10(numerator / denominator), dtype=np.float64)


def coupling_term_force_source(
    source_mobility: ArrayLike, receiver_mobility: ArrayLike
) -> np.ndarray:
    r"""Coupling term for a force source, high source mobility (Formula 19c).

    .. math::

       D_\mathrm{C} = 10 \log_{10}\frac{|Y_\mathrm{s}|}{\operatorname{Re}\{Y_i\}}

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero).
    :param receiver_mobility: Receiver point mobility ``Y_i`` (complex,
        positive real part).
    :return: The coupling term ``D_C``, in dB.
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Y_i}`` is not
        positive and finite.
    """
    ys = np.abs(_nonzero_magnitude(source_mobility, "source_mobility"))
    yi = np.real(_positive_real_part(receiver_mobility, "receiver_mobility"))
    return np.asarray(10.0 * np.log10(ys / yi), dtype=np.float64)


def coupling_term_velocity_source(
    source_mobility: ArrayLike, receiver_impedance: ArrayLike
) -> np.ndarray:
    r"""Coupling term for a velocity source, low source mobility (Formula 19d).

    .. math::

       D_\mathrm{C} = -10 \log_{10}\left( |Y_\mathrm{s}| \operatorname{Re}\{Z_i\} \right)

    :param source_mobility: Source point mobility ``Y_s`` (complex, non-zero).
    :param receiver_impedance: Receiver point impedance ``Z_i`` (complex,
        positive real part).
    :return: The coupling term ``D_C``, in dB.
    :raises ValueError: if ``Y_s`` is zero/non-finite or ``Re{Z_i}`` is not
        positive and finite.
    """
    ys = np.abs(_nonzero_magnitude(source_mobility, "source_mobility"))
    zi = np.real(_positive_real_part(receiver_impedance, "receiver_impedance"))
    return np.asarray(-10.0 * np.log10(ys * zi), dtype=np.float64)


# ---------------------------------------------------------------------------
# Annex D, Table D.1 -- mobility of typical construction elements
# ---------------------------------------------------------------------------
# EN 12354-5:2009, Table D.1 "Estimations for the mobility of typical
# construction elements" (BS EN 12354-5:2009, PDF page 48, printed folio 46).
# The table has three columns: the type of structure, the quantities that
# describe it, and the mobility magnitude |Y| in m/(N.s). Clause D.1.3 places
# it under "service equipment with known source mobility": the rows are
# estimates of the *source* mobility Y_s built up from the machine's own parts
# (total mass, feet, chassis panels, connected pipework), which is the input
# the coupling term of Formula (19b) needs and which no measurement of the
# equipment supplies on its own.
#
# The printed table gives magnitudes only, so every row returns a real,
# positive |Y|; :func:`coupling_term` accepts it directly wherever the phase is
# not being tracked. Frequency is not listed among the describing quantities
# because it is the band frequency of the prediction rather than a property of
# the element, but four of the six rows contain it.

#: EN 12354-5, Table D.1: the "Describing quantities" column, as the keyword
#: names each row of :func:`typical_element_mobility` requires. The printed
#: symbols are ``M`` [kg], ``rho`` [kg/m3], ``cL`` [m/s], ``S`` [m^2], ``t``
#: [m], ``w`` [m], ``r`` (radius) [m], ``s`` [N/m] and ``eta`` [-].
TABLE_D1_QUANTITIES: dict[str, tuple[str, ...]] = {
    "mass": ("mass",),
    "bar_end": ("density", "longitudinal_velocity", "area"),
    "beam": ("density", "longitudinal_velocity", "thickness", "width"),
    "plate": ("density", "longitudinal_velocity", "thickness"),
    "pipe": ("density", "longitudinal_velocity", "thickness", "radius"),
    "mass_spring": ("mass", "stiffness", "loss_factor"),
}

#: The Table D.1 rows whose printed expression contains the frequency ``f``.
#: The other two are frequency-independent, and passing ``frequency`` to them
#: would silently suggest a spectrum the table does not have.
_TABLE_D1_FREQUENCY_DEPENDENT: frozenset[str] = frozenset(
    {"mass", "beam", "pipe", "mass_spring"}
)


def _mass_mobility(frequency: ArrayLike, mass: float) -> np.ndarray:
    r"""Mobility of a mass (EN 12354-5, Table D.1, row "Mass").

    .. math::

       |Y| = \left[ 2 \pi f M \right]^{-1}

    :param frequency: Frequency ``f``, in hertz (scalar or per band, > 0).
    :param mass: Mass ``M``, in kilograms (> 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive frequency or mass.
    """
    f = _positive_values(frequency, "frequency")
    m = require_positive(mass, "mass")
    return np.asarray(1.0 / (2.0 * np.pi * f * m), dtype=np.float64)


def _bar_end_mobility(
    density: float, longitudinal_velocity: float, area: float
) -> np.ndarray:
    r"""Mobility at the end of a bar (EN 12354-5, Table D.1, row "Bar end").

    .. math::

       |Y| = \left[ \rho c_\mathrm{L} S \right]^{-1}

    Frequency-independent: the end of a bar loaded in compression behaves as
    the characteristic impedance of the quasi-longitudinal wave.

    :param density: Density ``rho``, in kg/m^3 (> 0).
    :param longitudinal_velocity: Quasi-longitudinal wave speed ``cL``, in m/s
        (> 0).
    :param area: Cross-sectional area ``S``, in m^2 (> 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive input.
    """
    rho = require_positive(density, "density")
    c_l = require_positive(longitudinal_velocity, "longitudinal_velocity")
    s = require_positive(area, "area")
    return np.asarray(1.0 / (rho * c_l * s), dtype=np.float64)


def _beam_mobility(
    frequency: ArrayLike,
    density: float,
    longitudinal_velocity: float,
    thickness: float,
    width: float,
) -> np.ndarray:
    r"""Mobility of a beam in bending (EN 12354-5, Table D.1, row "Beam").

    .. math::

       |Y| = \left[ 7{,}6\, \rho\, t\, w \sqrt{c_\mathrm{L} t f} \right]^{-1}

    :param frequency: Frequency ``f``, in hertz (scalar or per band, > 0).
    :param density: Density ``rho``, in kg/m^3 (> 0).
    :param longitudinal_velocity: Quasi-longitudinal wave speed ``cL``, in m/s
        (> 0).
    :param thickness: Beam thickness ``t`` in the bending direction, in metres
        (> 0).
    :param width: Beam width ``w``, in metres (> 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive input.
    """
    f = _positive_values(frequency, "frequency")
    rho = require_positive(density, "density")
    c_l = require_positive(longitudinal_velocity, "longitudinal_velocity")
    t = require_positive(thickness, "thickness")
    w = require_positive(width, "width")
    return np.asarray(
        1.0 / (7.6 * rho * t * w * np.sqrt(c_l * t * f)), dtype=np.float64
    )


def _plate_mobility(
    density: float, longitudinal_velocity: float, thickness: float
) -> np.ndarray:
    r"""Mobility of a plate in bending (EN 12354-5, Table D.1, row "Plate").

    .. math::

       |Y| = \left[ 2{,}3\, c_\mathrm{L}\, \rho\, t^2 \right]^{-1}

    Frequency-independent above the lowest plate resonance, and the same
    quantity Annex F Formula (F.4) writes as
    :math:`Y_{i,\infty} = 1 / (8\sqrt{m B'})`; see
    :func:`phonometry.vibration.structural.point_mobility.infinite_plate_mobility`
    for the stiffness-and-mass parameterisation of the same result.

    :param density: Density ``rho``, in kg/m^3 (> 0).
    :param longitudinal_velocity: Quasi-longitudinal wave speed ``cL``, in m/s
        (> 0).
    :param thickness: Plate thickness ``t``, in metres (> 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive input.
    """
    rho = require_positive(density, "density")
    c_l = require_positive(longitudinal_velocity, "longitudinal_velocity")
    t = require_positive(thickness, "thickness")
    return np.asarray(1.0 / (2.3 * c_l * rho * t**2), dtype=np.float64)


def _pipe_mobility(
    frequency: ArrayLike,
    density: float,
    longitudinal_velocity: float,
    thickness: float,
    radius: float,
) -> np.ndarray:
    r"""Mobility of a pipe wall (EN 12354-5, Table D.1, row "Pipe").

    .. math::

       |Y| = \left[ 63\, \rho\, t\, r \sqrt{c_\mathrm{L} r f} \right]^{-1}

    :param frequency: Frequency ``f``, in hertz (scalar or per band, > 0).
    :param density: Density ``rho``, in kg/m^3 (> 0).
    :param longitudinal_velocity: Quasi-longitudinal wave speed ``cL``, in m/s
        (> 0).
    :param thickness: Wall thickness ``t``, in metres (> 0).
    :param radius: Pipe radius ``r``, in metres (> 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive input.
    """
    f = _positive_values(frequency, "frequency")
    rho = require_positive(density, "density")
    c_l = require_positive(longitudinal_velocity, "longitudinal_velocity")
    t = require_positive(thickness, "thickness")
    r = require_positive(radius, "radius")
    return np.asarray(
        1.0 / (63.0 * rho * t * r * np.sqrt(c_l * r * f)), dtype=np.float64
    )


def _mass_spring_mobility(
    frequency: ArrayLike, mass: float, stiffness: float, loss_factor: float
) -> np.ndarray:
    r"""Mobility of a mass on a spring (Table D.1, row "Mass-spring").

    .. math::

       |Y| = \left[ \left( \frac{2 \pi f \eta}{s (1 + \eta^2)} \right)^2
       + \left( \frac{2 \pi f}{s (1 + \eta^2)}
       - \frac{1}{2 \pi f M} \right)^2 \right]^{1/2}

    The row for a machine on non-rigid feet, and the series sum of the spring
    mobility :math:`\mathrm{j}\omega / s(1 + \mathrm{j}\eta)` and the mass
    mobility :math:`1/\mathrm{j}\omega M`. The second bracket is the two
    reactances, and they cancel at the mass-spring resonance
    :math:`f_0 = (2\pi)^{-1}\sqrt{s(1 + \eta^2)/M}`, where the mobility drops
    to its damping-limited **minimum**
    :math:`2 \pi f_0 \eta / (s (1 + \eta^2))`. That is the frequency at which
    the mount injects the most power, since a small ``|Y_s|`` is a small
    ``D_C``. Passing ``loss_factor=0`` returns exactly zero there, which
    :func:`coupling_term` then rejects; the printed expression is written for
    a damped support.

    :param frequency: Frequency ``f``, in hertz (scalar or per band, > 0).
    :param mass: Mass ``M`` carried by the support, in kilograms (> 0).
    :param stiffness: Support stiffness ``s``, in N/m (> 0).
    :param loss_factor: Support loss factor ``eta``, dimensionless (>= 0).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for a non-positive frequency, mass or stiffness, or a
        negative loss factor.
    """
    f = _positive_values(frequency, "frequency")
    m = require_positive(mass, "mass")
    s = require_positive(stiffness, "stiffness")
    eta = require_non_negative(loss_factor, "loss_factor")
    compliance = 2.0 * np.pi * f / (s * (1.0 + eta**2))
    inertance = 1.0 / (2.0 * np.pi * f * m)
    return np.asarray(
        np.hypot(compliance * eta, compliance - inertance), dtype=np.float64
    )


def _table_d1_arguments(
    structure: str, supplied: dict[str, float | None], frequency: ArrayLike | None
) -> dict[str, float]:
    """Check the quantities supplied for one Table D.1 row against its column.

    :raises ValueError: If a describing quantity is missing, one that the row
        does not describe was supplied, or ``frequency`` does not match the
        row's dependence on it.
    """
    required = TABLE_D1_QUANTITIES[structure]
    missing = [k for k in required if supplied[k] is None]
    extra = [k for k, v in supplied.items() if v is not None and k not in required]
    if missing or extra:
        raise ValueError(
            f"Table D.1 row {structure!r} is described by "
            f"{', '.join(required)}"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; {', '.join(extra)} does not describe it" if extra else "")
            + "."
        )
    needs_frequency = structure in _TABLE_D1_FREQUENCY_DEPENDENT
    if needs_frequency and frequency is None:
        raise ValueError(
            f"Table D.1 row {structure!r} depends on frequency; pass 'frequency'."
        )
    if not needs_frequency and frequency is not None:
        raise ValueError(
            f"Table D.1 row {structure!r} is frequency-independent; do not "
            "pass 'frequency'."
        )
    return {k: float(supplied[k]) for k in required}  # type: ignore[arg-type]


def typical_element_mobility(
    structure: str,
    *,
    frequency: ArrayLike | None = None,
    mass: float | None = None,
    density: float | None = None,
    longitudinal_velocity: float | None = None,
    thickness: float | None = None,
    width: float | None = None,
    area: float | None = None,
    radius: float | None = None,
    stiffness: float | None = None,
    loss_factor: float | None = None,
) -> np.ndarray:
    r"""Named lookup of EN 12354-5, Table D.1 (mobility of typical elements).

    Returns the mobility magnitude ``|Y|`` the third column of Table D.1
    prints, in m/(N.s), for the row named by *structure*:

    - ``"mass"``, described by ``mass`` :math:`M` [kg]:
      :math:`\lvert Y \rvert = \left[ 2 \pi f M \right]^{-1}`.
    - ``"bar_end"``, described by ``density`` :math:`\rho` [kg/m3],
      ``longitudinal_velocity`` :math:`c_\mathrm{L}` [m/s] and ``area`` :math:`S`
      [m2]: :math:`\lvert Y \rvert = \left[ \rho c_\mathrm{L} S \right]^{-1}`.
    - ``"beam"``, described by :math:`\rho`, :math:`c_\mathrm{L}`, ``thickness``
      :math:`t` [m] and ``width`` :math:`w` [m]:
      :math:`\lvert Y \rvert = \left[ 7{,}6\, \rho t w \sqrt{c_\mathrm{L} t f}
      \right]^{-1}`.
    - ``"plate"``, described by :math:`\rho`, :math:`c_\mathrm{L}` and :math:`t`:
      :math:`\lvert Y \rvert = \left[ 2{,}3\, c_\mathrm{L} \rho t^2 \right]^{-1}`.
    - ``"pipe"``, described by :math:`\rho`, :math:`c_\mathrm{L}`, :math:`t` and
      ``radius`` :math:`r` [m]:
      :math:`\lvert Y \rvert = \left[ 63\, \rho t r \sqrt{c_\mathrm{L} r f}
      \right]^{-1}`.
    - ``"mass_spring"``, described by :math:`M`, ``stiffness`` :math:`s`
      [N/m] and ``loss_factor`` :math:`\eta` [-]:
      :math:`\lvert Y \rvert = \left[ \left( \frac{2 \pi f \eta}{s (1 +
      \eta^2)} \right)^2 + \left( \frac{2 \pi f}{s (1 + \eta^2)}
      - \frac{1}{2 \pi f M} \right)^2 \right]^{1/2}`.

    :data:`TABLE_D1_QUANTITIES` carries the "Describing quantities" column as
    the keyword names of this function, and only the quantities a row describes
    may be supplied.
    Frequency is not among them, because it is the band frequency of the
    prediction and not a property of the element, but it appears in four of the
    six expressions: ``"mass"``, ``"beam"``, ``"pipe"`` and ``"mass_spring"``
    require ``frequency`` and the other two reject it.

    Clause D.1.3 offers the table for building up a **source** mobility ``Y_s``
    from the machine's own parts, which is what :func:`coupling_term` needs and
    what a measurement of the equipment does not give. Annex F Formulae (F.4)
    to (F.6b) cover the *receiver* mobility ``Y_i`` of the supporting element;
    the ``"plate"`` row is the same quantity as Formula (F.4)
    :math:`Y_{i,\infty} = 1 / (8\sqrt{m B'})` and as
    :func:`phonometry.vibration.structural.point_mobility.infinite_plate_mobility`,
    written in :math:`\rho`, :math:`c_\mathrm{L}` and :math:`t` instead of mass and
    bending stiffness.

    The ``"mass_spring"`` row is the machine on non-rigid feet: its second
    bracket holds the two reactances, which cancel at the mass-spring
    resonance :math:`f_0 = (2\pi)^{-1}\sqrt{s(1 + \eta^2)/M}` and leave the
    mobility at its damping-limited minimum, the frequency at which the mount
    injects the most power. ``loss_factor=0`` returns exactly zero there,
    which :func:`coupling_term` then rejects.

    :param structure: Table D.1 row name.
    :param frequency: Frequency ``f``, in hertz, for the frequency-dependent
        rows only.
    :param mass: Mass ``M``, in kilograms (rows ``"mass"``, ``"mass_spring"``).
    :param density: Density ``rho``, in kg/m^3.
    :param longitudinal_velocity: Quasi-longitudinal wave speed ``cL``, in m/s.
    :param thickness: Thickness ``t``, in metres.
    :param width: Beam width ``w``, in metres (row ``"beam"``).
    :param area: Cross-sectional area ``S``, in m^2 (row ``"bar_end"``).
    :param radius: Pipe radius ``r``, in metres (row ``"pipe"``).
    :param stiffness: Support stiffness ``s``, in N/m (row ``"mass_spring"``).
    :param loss_factor: Support loss factor ``eta`` (row ``"mass_spring"``).
    :return: The mobility magnitude ``|Y|``, in m/(N.s).
    :raises ValueError: for an unknown row, a missing or surplus describing
        quantity, a ``frequency`` that the row does not take (or lacks), or a
        non-positive value.
    """
    row = require_choice(structure, "structure", tuple(TABLE_D1_QUANTITIES))
    args = _table_d1_arguments(
        row,
        {
            "mass": mass,
            "density": density,
            "longitudinal_velocity": longitudinal_velocity,
            "thickness": thickness,
            "width": width,
            "area": area,
            "radius": radius,
            "stiffness": stiffness,
            "loss_factor": loss_factor,
        },
        frequency,
    )
    if row == "mass":
        return _mass_mobility(frequency, args["mass"])  # type: ignore[arg-type]
    if row == "bar_end":
        return _bar_end_mobility(
            args["density"], args["longitudinal_velocity"], args["area"]
        )
    if row == "beam":
        return _beam_mobility(
            frequency,  # type: ignore[arg-type]
            args["density"],
            args["longitudinal_velocity"],
            args["thickness"],
            args["width"],
        )
    if row == "plate":
        return _plate_mobility(
            args["density"], args["longitudinal_velocity"], args["thickness"]
        )
    if row == "pipe":
        return _pipe_mobility(
            frequency,  # type: ignore[arg-type]
            args["density"],
            args["longitudinal_velocity"],
            args["thickness"],
            args["radius"],
        )
    return _mass_spring_mobility(
        frequency,  # type: ignore[arg-type]
        args["mass"],
        args["stiffness"],
        args["loss_factor"],
    )


# ---------------------------------------------------------------------------
# Annex F, Table F.1 -- force level of the ISO tapping machine
# ---------------------------------------------------------------------------
# EN 12354-5:2009, Table F.1 "Force level LF re 1 pN for the ISO tapping
# machine in octave bands" (BS EN 12354-5:2009, PDF page 61, printed folio 59).
# The table is two rows deep under one spanning header, "Octave band with
# centre frequency in Hz": the eight nominal centres and the eight force
# levels. The standard prints the first centre as "31"; it is the 31,5 Hz
# nominal octave band, and this module keys it as 31,5 Hz like every other
# band elsewhere in the library.
#
# Clause F.4.2 sets the conditions: the values are for the ISO tapping machine
# used as a substitution source in place of an electrodynamic shaker when
# measuring the normalised level difference D_Fp,n of Formula (F.9), and
# clause D.1.2.3 restricts them to low-mobility receiving structures. Only
# octave bands are tabulated; the one-third-octave counterpart comes from the
# closed form of :func:`tapping_machine_force_level_estimate`.
#
# The reference the table caption prints, "re 1 pN", is a defect: the values
# are re 1e-6 N, the reference force of ISO 1683 that EN 15657:2018 Formula
# (15) also uses. See docs/ERRATA.md. The tabulated numbers are unaffected.

#: EN 12354-5, Table F.1: the nominal octave-band centre frequencies of the
#: header row, in hertz. The standard prints the first as "31".
TABLE_F1_OCTAVE_BANDS: tuple[float, ...] = (
    31.5,
    63.0,
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
)

#: EN 12354-5, Table F.1: the force level ``L_F`` of the ISO tapping machine
#: per octave band of :data:`TABLE_F1_OCTAVE_BANDS`, in dB re 1e-6 N
#: (the caption prints "re 1 pN"; see ``docs/ERRATA.md``).
TABLE_F1_FORCE_LEVEL: tuple[float, ...] = (
    139.0,
    142.0,
    145.0,
    148.0,
    151.0,
    154.0,
    156.0,
    156.0,
)

#: Bandwidth coefficient of the closed form printed under Table F.1:
#: ``L_F = 10 lg(k f / 10^-12)`` with ``k = 2,5`` in octave bands and
#: ``k = 0,8`` in one-third-octave bands.
_TAPPING_MACHINE_COEFFICIENT: dict[str, float] = {"octave": 2.5, "third": 0.8}


def tapping_machine_force_level() -> np.ndarray:
    r"""Tabulated ISO tapping machine force level (EN 12354-5, Table F.1).

    The eight octave-band values of Table F.1, in the order of
    :data:`TABLE_F1_OCTAVE_BANDS` (31,5 Hz to 4 kHz). Clause F.4.2
    offers them for the tapping machine used in place of an electrodynamic
    shaker when measuring ``D_Fp,n`` (Formula F.9), and clause D.1.2.3
    restricts the source to low-mobility receiving structures.

    Feed them to :func:`tapping_machine_characteristic_power_level` for the
    ``L_Ws,c`` that :func:`installed_source_prediction` takes.

    :return: The force level ``L_F`` per octave band, in dB re 1e-6 N (the
        table caption prints "re 1 pN"; see ``docs/ERRATA.md``).
    """
    return np.asarray(TABLE_F1_FORCE_LEVEL, dtype=np.float64)


def tapping_machine_force_level_estimate(
    frequency: ArrayLike, *, bandwidth: str = "octave"
) -> np.ndarray:
    r"""Closed form printed under Table F.1 for the tapping machine force level.

    .. math::

       L_F = 10 \log_{10} \frac{2{,}5 f}{10^{-12}} \quad\text{(octave)},
       \qquad
       L_F = 10 \log_{10} \frac{0{,}8 f}{10^{-12}} \quad\text{(1/3 octave)}

    The standard qualifies this with "up till about 1000 Hz": it reproduces the
    first six tabulated values to the printed decibel, and above that it
    departs from the table, which
    flattens at 156 dB (the closed form gives 157 dB at 2 kHz and 160 dB at
    4 kHz). Use :func:`tapping_machine_force_level` for the octave bands the
    table covers and this only where it does not, chiefly the one-third-octave
    bands the standard does not tabulate.

    :param frequency: Band centre frequency ``f``, in hertz (> 0).
    :param bandwidth: ``"octave"`` (coefficient 2,5) or ``"third"``
        (coefficient 0,8).
    :return: The force level ``L_F``, in dB re 1e-6 N (the standard prints
        "re 1 pN"; see ``docs/ERRATA.md``).
    :raises ValueError: for a non-positive frequency or an unknown bandwidth.
    """
    band = require_choice(bandwidth, "bandwidth", tuple(_TAPPING_MACHINE_COEFFICIENT))
    f = _positive_values(frequency, "frequency")
    coefficient = _TAPPING_MACHINE_COEFFICIENT[band]
    return np.asarray(10.0 * np.log10(coefficient * f / 1.0e-12), dtype=np.float64)


def tapping_machine_characteristic_power_level(
    frequency: ArrayLike, force_level: ArrayLike
) -> np.ndarray:
    r"""Characteristic power level of the tapping machine (Formula D.9a).

    .. math::

       L_{W\mathrm{s,c}} = L_F - 5 - 10 \log_{10} f

    The standard notes the result is about 115 dB re 1 pW per one-third octave
    for the ISO tapping machine, treated in clause D.1.3 as a force source with
    the mass-like source mobility of its 0,5 kg hammers. Pair it with
    :func:`tapping_machine_coupling_term` and hand both to
    :func:`installed_source_prediction`.

    The formula only balances with the reference force ``F_0 = 1e-6`` N, since
    it carries no term for :math:`F_0^2 / W_0`; that is what pins the reading
    of Table F.1 against its own printed "re 1 pN" caption.

    :param frequency: Band centre frequency ``f``, in hertz (> 0).
    :param force_level: Force level ``L_F``, in dB re 1e-6 N (Table F.1 or
        :func:`tapping_machine_force_level_estimate`).
    :return: The characteristic power level ``L_Ws,c``, in dB re 1 pW.
    :raises ValueError: for a non-positive frequency.
    """
    f = _positive_values(frequency, "frequency")
    lf = np.asarray(force_level, dtype=np.float64)
    return np.asarray(lf - 5.0 - 10.0 * np.log10(f), dtype=np.float64)


def tapping_machine_coupling_term(
    frequency: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    hammer_mass: float = TAPPING_HAMMER_MASS,
) -> np.ndarray:
    r"""Coupling term of the tapping machine (EN 12354-5, Formula D.9b).

    .. math::

       D_{\mathrm{C},i} = -10 \log_{10}(\omega M Y_i)
       + 10 \log_{10}\left[ 1 + (\omega M Y_i)^2 \right]

    with :math:`\omega = 2 \pi f` and ``M`` the hammer mass of clause D.1.3,
    which the standard takes as 0,5 kg. It is the mass-like-source form of
    Formula (19b), for a machine standing on a plate-like element of real
    mobility ``Y_i``; Annex F Formulae (F.4) to (F.6b) estimate that ``Y_i``.

    :param frequency: Band centre frequency ``f``, in hertz (> 0).
    :param receiver_mobility: Real mobility ``Y_i`` of the supporting element,
        in m/(N.s) (> 0).
    :param hammer_mass: Source mass ``M``, in kilograms (Default:
        :data:`~phonometry.building.TAPPING_HAMMER_MASS`, the 0,5 kg of clause
        D.1.3 and ISO 10140-5).
    :return: The coupling term ``D_C,i``, in dB.
    :raises ValueError: for a non-positive frequency, mobility or mass.
    """
    f = _positive_values(frequency, "frequency")
    y_i = _positive_values(receiver_mobility, "receiver_mobility")
    m = require_positive(hammer_mass, "hammer_mass")
    omega_m_y = 2.0 * np.pi * f * m * y_i
    return np.asarray(
        -10.0 * np.log10(omega_m_y) + 10.0 * np.log10(1.0 + omega_m_y**2),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Annex F -- the terms Formula (18a) takes
# ---------------------------------------------------------------------------

#: Adjustment ``dK`` to the vibration reduction index for a path that crosses
#: more than one junction (EN 12354-5, clause F.1), in dB, keyed by the number
#: of junctions in the path. The standard gives 4 dB for two junctions and
#: 6 dB for three or more; a single junction is the invariant ``Kij`` of
#: EN 12354-1 and takes no adjustment.
_MULTI_JUNCTION_ADJUSTMENT: dict[int, float] = {1: 0.0, 2: 4.0, 3: 6.0}

#: Floor on the multi-junction ``Kij`` of clause F.1, in dB: "the resulting
#: value for Kij should normally not become less than - 5 dB", the point at
#: which the standard reads the junctions as transmitting the whole
#: structure-borne power.
MINIMUM_MULTI_JUNCTION_KIJ: float = -5.0


def multi_junction_adjustment(junctions: int) -> float:
    r"""Vibration reduction index adjustment ``dK`` (EN 12354-5, clause F.1).

    When the receiving room is more than one junction away from the equipment,
    Formula (F.1) sums the junction ``Kij`` along the path and subtracts an
    adjustment ``dK`` that covers the transmission by wave types other than
    bending waves. Clause F.1 estimates it from published data as 4 dB for two
    junctions and 6 dB for three or more, with the resulting ``Kij`` floored at
    :data:`MINIMUM_MULTI_JUNCTION_KIJ`.

    :param junctions: Number of junctions the transmission path crosses (>= 1).
    :return: The adjustment ``dK``, in dB (0,0 for a single junction).
    :raises ValueError: for fewer than one junction.
    """
    if junctions < 1:
        raise ValueError("'junctions' must be at least 1.")
    return _MULTI_JUNCTION_ADJUSTMENT[min(junctions, 3)]


def structure_to_airborne_adjustment(
    frequency: ArrayLike,
    critical_frequency: float,
    mass_per_area: float,
    *,
    radiation_factor: ArrayLike = 1.0,
) -> np.ndarray:
    r"""Adjustment term ``D_sa`` of a supporting element (Formula F.3).

    .. math::

       D_{\mathrm{sa},i} = 10 \log_{10} \frac{400 f_{\mathrm{c},i} \sigma_i}{m_i f^2}

    the ratio of injected structure-borne power to incident airborne power that
    leaves the same free-vibration energy in the element. Clause F.2 gives it
    for a force excitation perpendicular to a homogeneous supporting element,
    exact above the critical frequency (where the radiation factor saturates at
    1) and a good approximation over the whole range.

    ``D_sa`` is normally **negative**, and Formula (18a) subtracts it, so it
    raises the predicted level: this is the value
    :func:`structure_borne_pressure_level_path` takes as ``adjustment_term``,
    sign included.

    :param frequency: Band centre frequency ``f``, in hertz (> 0).
    :param critical_frequency: Critical frequency ``fc,i`` of the element, in
        hertz (> 0).
    :param mass_per_area: Mass per unit area ``mi`` of the element, in kg/m^2
        (> 0).
    :param radiation_factor: Radiation factor ``sigma_i`` of the element
        (Default: 1.0, its value above ``fc``; EN 12354-1:2000 Annex B
        estimates it below).
    :return: The adjustment term ``D_sa,i``, in dB.
    :raises ValueError: for a non-positive frequency, critical frequency, mass
        or radiation factor.
    """
    f = _positive_values(frequency, "frequency")
    f_c = require_positive(critical_frequency, "critical_frequency")
    m = require_positive(mass_per_area, "mass_per_area")
    sigma = _positive_values(radiation_factor, "radiation_factor")
    return np.asarray(
        10.0 * np.log10(400.0 * f_c * sigma / (m * f**2)), dtype=np.float64
    )


def installed_power_from_reception_plate(
    reception_plate_level: ArrayLike,
    receiver_mobility: ArrayLike,
    *,
    plate_mobility: float = REFERENCE_PLATE_MOBILITY,
) -> np.ndarray:
    r"""Mobility correction of the reception-plate power (EN 12354-5, Annex I).

    :math:`L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,n},i} + 10 \log_{10}( Y_{\infty,i} /
    Y_{\infty,\mathrm{rec}} )`, which refers the
    characteristic reception-plate power level ``L_Ws,n`` (EN 15657
    Formula (17), re the 10 cm concrete plate
    :math:`Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}` m/(N.s))
    to the characteristic mobility ``Y_inf,i`` of the actual receiving
    element (floor, wall), yielding the installed power of that element as in
    the Annex I.2 whirlpool example. The same correction with the *source*
    mobility instead of ``Y_inf,i`` yields the characteristic level
    ``L_Ws,c`` (Annex I.3, Table I.8), from which
    :func:`installed_structure_borne_power_level` subtracts ``D_C``.

    :param reception_plate_level: Power level to re-refer (per band), in dB re
        1 pW: either the characteristic level ``L_Ws,n`` (EN 15657 Formula 17,
        referred to the default 5e-6 m/(N.s) plate) or a raw Formula (14)
        plate power together with the mobility of the plate it was measured
        on, passed as ``plate_mobility``.
    :param receiver_mobility: Characteristic mobility ``Y_inf,i`` of the
        receiving element (per band; complex values use their magnitude), in
        m/(N.s).
    :param plate_mobility: Mobility the input level is referred to
        (Default: the EN 15657 reference plate,
        :math:`Y_{\infty,\mathrm{rec}} = 5 \cdot 10^{-6}` m/(N.s);
        pass the measured plate mobility when the input is a raw Formula (14)
        level).
    :return: The mobility-corrected power level, in dB re 1 pW.
    :raises ValueError: for a non-positive receiver or plate mobility.
    """
    plate_mobility = require_positive(plate_mobility, "plate_mobility")
    lw = np.asarray(reception_plate_level, dtype=np.float64)
    y_i = np.abs(np.asarray(receiver_mobility, dtype=np.complex128))
    if not np.all(np.isfinite(y_i)) or np.any(y_i <= 0.0):
        raise ValueError("'receiver_mobility' must be finite and non-zero.")
    return np.asarray(lw + 10.0 * np.log10(y_i / plate_mobility), dtype=np.float64)


def installed_structure_borne_power_level(
    characteristic_power_level: ArrayLike, coupling_term: ArrayLike
) -> np.ndarray:
    r"""Installed structure-borne power level (EN 12354-5, Formula 18b).

    .. math::

       L_{W\mathrm{s,inst},i} = L_{W\mathrm{s,c}} - D_{\mathrm{C},i}

    :param characteristic_power_level: Characteristic level ``L_Ws,c`` (per
        band), in dB: the EN 15657 reception-plate level converted with
        Formulae (15)/(17) and the source-mobility correction (see the module
        docstring), **not** the raw plate-injected Formula (14) level.
    :param coupling_term: Coupling term ``D_C,i`` (per band), in dB.
    :return: The installed structure-borne power level ``L_Ws,inst``, in dB.
    """
    lw = np.asarray(characteristic_power_level, dtype=np.float64)
    dc = np.asarray(coupling_term, dtype=np.float64)
    return np.asarray(lw - dc, dtype=np.float64)


def structure_borne_pressure_level_path(
    installed_power_level: ArrayLike,
    adjustment_term: ArrayLike,
    flanking_reduction_index: ArrayLike,
    element_area: float,
    *,
    reference_area: float = REFERENCE_AREA,
) -> np.ndarray:
    r"""Normalised structure-borne SPL for one path i->j (Formula 18a).

    .. math::

       L_{\mathrm{n,s},ij} = L_{W\mathrm{s,inst},i} - D_{\mathrm{sa},i} - R_{ij,\mathrm{ref}}
       - 10 \log_{10}\frac{S_i}{S_0} - 10 \log_{10}\frac{A_0}{4}

    :param installed_power_level: Installed power level ``L_Ws,inst,i``, in dB.
    :param adjustment_term: Structure-to-airborne adjustment ``D_sa,i`` (clause
        4.4.4 / Annex F), in dB.
    :param flanking_reduction_index: Flanking sound reduction index
        ``R_ij,ref`` re ``S0`` (EN 12354-1), in dB.
    :param element_area: Supporting-element area ``S_i``, in m^2 (> 0).
    :param reference_area: Reference area :math:`S_0 = A_0` (Default: 10 m^2).
    :return: The normalised path sound pressure level ``L_n,s,ij``, in dB.
    :raises ValueError: for a non-positive area.
    """
    element_area = require_positive(element_area, "element_area")
    reference_area = require_positive(reference_area, "reference_area")
    lw = np.asarray(installed_power_level, dtype=np.float64)
    dsa = np.asarray(adjustment_term, dtype=np.float64)
    rij = np.asarray(flanking_reduction_index, dtype=np.float64)
    lp = (
        lw
        - dsa
        - rij
        - 10.0 * np.log10(element_area / reference_area)
        - 10.0 * np.log10(reference_area / 4.0)
    )
    return np.asarray(lp, dtype=np.float64)


def total_structure_borne_pressure_level(path_levels: ArrayLike) -> np.ndarray:
    r"""Combine path sound pressure levels energetically (Formula 17).

    .. math::

       L_\mathrm{n,s} = 10 \log_{10}\!\left( \sum_j 10^{L_{\mathrm{n,s},ij}/10} \right)

    :param path_levels: Path levels ``L_n,s,ij``; sum is over the first axis
        (paths), broadcasting any trailing band axis.
    :return: The total normalised sound pressure level ``L_n,s``, in dB.
    """
    lp = np.asarray(path_levels, dtype=np.float64)
    return np.asarray(
        10.0 * np.log10(np.sum(10.0 ** (0.1 * lp), axis=0)), dtype=np.float64
    )


@dataclass(frozen=True)
class InstalledSourceResult:
    """Installed structure-borne sound prediction (EN 12354-5).

    :ivar frequencies: Band centre frequencies, in hertz, or ``None``.
    :ivar path_levels: Per-path normalised SPL ``L_n,s,ij`` (paths x bands), dB.
    :ivar total_level: Combined normalised SPL ``L_n,s`` per band, in dB.
    :ivar installed_power_level: Installed power level ``L_Ws,inst`` per band, dB.
    """

    path_levels: np.ndarray
    total_level: np.ndarray
    installed_power_level: np.ndarray
    frequencies: np.ndarray | None = None

    @property
    def overall_level(self) -> float:
        r"""Band-summed total level :math:`10 \log_{10}(\sum 10^{0.1 L_\mathrm{n,s}})`,
        in dB."""
        lt = np.atleast_1d(np.asarray(self.total_level, dtype=np.float64))
        return float(10.0 * np.log10(np.sum(10.0 ** (0.1 * lt))))

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes:
        """Plot the per-path and total normalised sound pressure levels.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from ..._i18n import check_language
        from ..._plot.building import plot_installed_structure_borne

        check_language(language)
        return plot_installed_structure_borne(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an EN 12354-5 installed structure-borne prediction fiche.

        Writes a one-page **prediction** sheet (an estimate, not a
        measurement): a prediction-basis line naming EN 12354-5:2009, an
        optional metadata header (client, source equipment, receiving room,
        instrumentation, climate, date), a per-band table (nominal
        octave/one-third-octave frequency, the installed structure-borne power
        level ``L_Ws,inst``, each transmission path's normalised SPL
        ``L_n,s,ij`` and the combined total ``L_n,s``), the per-path and total
        ``L_n,s(f)`` spectra, the boxed band-summed total ``L_n,s`` (dB) with
        the installed power total and the path count, an optional verdict row
        against a declared limit, and a basis strip stating Formulae 18a/17 and
        the prediction disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata` supplying
            the header (``client``, ``specimen`` the source equipment,
            ``test_room`` the receiving room, ``instrumentation``,
            ``temperature``, ``relative_humidity``, ``pressure``,
            ``test_date``), the footer identity (``laboratory``, ``operator``,
            ``report_id``, ``notes``) and, via ``requirement``, a declared
            upper limit on the overall ``L_n,s`` (lower is better).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True`` the per-band table adds one column per
            transmission path (up to five); otherwise only the installed power
            and the combined total are shown.
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
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.en12354_5 import render_installed_structure_borne_report

        return render_installed_structure_borne_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


#: Keys every transmission path must carry: the adjustment term ``D_sa``, the
#: flanking reduction index ``R_ij,ref`` and the element area ``S_i``.
_REQUIRED_PATH_KEYS = (
    "adjustment_term",
    "flanking_reduction_index",
    "element_area",
)
#: The subset of :data:`_REQUIRED_PATH_KEYS` that may carry one value per band.
_PER_BAND_PATH_KEYS = ("adjustment_term", "flanking_reduction_index")


def _widest_band_count(n_bands: int, paths: list[dict[str, Any]]) -> int:
    """Widen *n_bands* to the longest per-band term any path carries.

    Also checks that every path carries the required keys, naming the path and
    the keys it is missing.

    :raises ValueError: If a path is missing a required key.
    """
    for k, p in enumerate(paths):
        missing = [key for key in _REQUIRED_PATH_KEYS if key not in p]
        if missing:
            raise ValueError(
                f"path {k} is missing required key(s): {', '.join(missing)}."
            )
        for key in _PER_BAND_PATH_KEYS:
            n_bands = max(
                n_bands, np.atleast_1d(np.asarray(p[key], dtype=np.float64)).size
            )
    return n_bands


def _path_pressure_levels(
    lw_inst: np.ndarray, paths: list[dict[str, Any]], n_bands: int
) -> np.ndarray:
    """One row of per-band ``L_n,s`` per transmission path, broadcast to *n_bands*.

    :raises ValueError: If a per-band term matches neither one value nor
        ``n_bands``.
    """
    rows = []
    for k, p in enumerate(paths):
        for key in _PER_BAND_PATH_KEYS:
            values = np.atleast_1d(np.asarray(p[key], dtype=np.float64))
            if values.size not in (1, n_bands):
                raise ValueError(
                    f"path {k}: {key!r} has {values.size} bands, expected 1 "
                    f"or {n_bands} to match the other per-band inputs."
                )
        rows.append(
            np.broadcast_to(
                structure_borne_pressure_level_path(
                    lw_inst,
                    p["adjustment_term"],
                    p["flanking_reduction_index"],
                    p["element_area"],
                ),
                (n_bands,),
            )
        )
    return np.asarray(rows, dtype=np.float64)


def installed_source_prediction(
    characteristic_power_level: ArrayLike,
    coupling_term: ArrayLike,
    paths: list[dict[str, Any]],
    *,
    frequencies: ArrayLike | None = None,
) -> InstalledSourceResult:
    """Predict the installed structure-borne SPL over several paths (EN 12354-5).

    The band count is set by the widest per-band input (the characteristic
    power level, the ``coupling_term`` or any path's ``adjustment_term`` /
    ``flanking_reduction_index``); every
    per-band input must carry one value or that count, and single values
    broadcast across the bands (a single-number source level with per-band
    path data is valid, and the result's ``installed_power_level`` is
    broadcast to the band count).

    :param characteristic_power_level: Characteristic level ``L_Ws,c`` (per
        band or a single value), in dB.
    :param coupling_term: Coupling term ``D_C`` (per band or a single
        value), in dB.
    :param paths: One dict per transmission path with keys ``adjustment_term``
        (``D_sa``), ``flanking_reduction_index`` (``R_ij,ref``) and
        ``element_area`` (``S_i``), each per band where applicable.
    :param frequencies: Band centre frequencies, in hertz, or ``None``.
    :return: The :class:`InstalledSourceResult`.
    :raises ValueError: if ``paths`` is empty, a path is missing a required
        key, or a per-band input matches neither one value nor the band
        count.
    """
    if not paths:
        raise ValueError("'paths' must contain at least one transmission path.")
    lw_inst = np.atleast_1d(
        np.asarray(
            installed_structure_borne_power_level(
                characteristic_power_level, coupling_term
            ),
            dtype=np.float64,
        )
    )
    # The band count is set by the widest per-band input (the characteristic
    # level, the coupling term or any path's per-band terms); every other
    # per-band input must carry one value or that count, and single values
    # broadcast across the bands.
    n_bands = _widest_band_count(lw_inst.size, paths)
    if lw_inst.size not in (1, n_bands):
        raise ValueError(
            f"the source levels carry {lw_inst.size} bands, expected 1 or "
            f"{n_bands} to match the transmission paths."
        )
    path_levels = _path_pressure_levels(lw_inst, paths, n_bands)
    total = total_structure_borne_pressure_level(path_levels)
    freq = (
        None
        if frequencies is None
        else np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    )
    if freq is not None and freq.size != n_bands:
        raise ValueError(
            f"frequencies carries {freq.size} values, expected {n_bands} to "
            "match the per-band inputs."
        )
    return InstalledSourceResult(
        path_levels=path_levels,
        total_level=np.asarray(total, dtype=np.float64),
        installed_power_level=np.broadcast_to(lw_inst, (n_bands,)).copy(),
        frequencies=freq,
    )
