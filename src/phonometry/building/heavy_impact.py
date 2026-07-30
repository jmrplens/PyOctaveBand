#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Heavy and soft impact sources: rubber ball and bang machine
(ISO 16283-2:2020 Annex A, ISO 10140-5:2010 Annex F, JIS A 1418-2:2019,
ISO 717-2:2020 Annex D).

The ISO tapping machine is a *light* impact source: five 500 g hammers dropped
from 40 mm produce a short, hard, quasi-stationary excitation whose spectrum
peaks well above 100 Hz. Real heavy impacts in a dwelling, a child jumping off
a chair or an adult walking barefoot, are slow, soft and low-frequency, and the
tapping machine says almost nothing about them. The **standard heavy impact
sources** were introduced for exactly that: a hollow silicone **rubber ball**
dropped from 1 m, and the **bang machine**, a car tyre dropped from 85 cm. Both
excite the floor with a single-peak force pulse about 20 ms long that puts most
of its energy below 125 Hz, and both are rated by a **maximum** (Fast
time-weighted) sound pressure level rather than by an energy average.

**Impact force exposure level (ISO 16283-2 Formula (A.1) = JIS A 1418-2
Formula (1)).** A heavy source is specified not by its geometry but by the
octave-band energy of its force pulse::

    LFE = 10 lg[ (1/Tref) integral_t1^t2 F(t)**2 / F0**2 dt ]   dB re 1 N

with ``F0 = 1 N``, ``Tref = 1 s`` and ``t2 - t1`` the duration of the impact
force. :func:`impact_force_exposure_level` evaluates it from a sampled force
waveform; :func:`heavy_impact_source_limits` returns the printed
octave-band tolerance table and :func:`check_heavy_impact_source` verifies a
measured source against it.

The two source specifications are printed identically in ISO 16283-2:2020
Table A.1 and ISO 10140-5:2010 Table F.1 (rubber ball), and in
JIS A 1418-2:2019 Tables A.2 (rubber ball, *impact force characteristic 2*) and
A.1 (bang machine, *impact force characteristic 1*):

===========  ====================  =====================
Octave (Hz)  Rubber ball LFE (dB)  Bang machine LFE (dB)
===========  ====================  =====================
31,5         39,0 +/- 1,0          47,0 +/- 1,0
63           31,0 +/- 1,5          40,0 +/- 1,5
125          23,0 +/- 1,5          22,0 +/- 1,5
250          17,0 +/- 2,0          11,5 +/- 2,0
500          12,5 +/- 2,0          5,5 +/- 2,0
===========  ====================  =====================

**Standardized maximum impact sound pressure level (ISO 16283-2 Formulae (4),
(5) and (6)).** Because the rated quantity is a *maximum* of a Fast-weighted
level and not an energy average, the receiving room cannot be corrected with
the usual ``10 lg(T/T0)``: the Fast detector only ever sees the first
``1,7275 s`` worth of decay. The standard therefore uses::

    L'i,Fmax,V,T = Li,Fmax + 10 lg(V/V0) - 10 lg[ g(C) / g(C0) ]
    C0 = T0 / 1,7275                                            (5)
    C  = T  / 1,7275                                            (6)

with ``T0 = 0,5 s``, ``V0 = 50 m3`` and, writing Formula (4) in the compact
form used here,

    ``g(C) = (C**(1/(1-C)) - C**(-1/(1-1/C))) / (1 - 1/C)``.

``g`` is the peak of the Fast-weighted response to an exponentially decaying
burst; for ``T = T0`` the bracket collapses to 1 and the correction reduces to
the pure volume term ``10 lg(V/V0)``, as it must. See
:func:`standardized_maximum_impact_level` and
:func:`fast_reverberation_correction`.

**A-weighted rating (ISO 717-2:2020 Annex D, normative).** The single number is
not a shifted reference curve but an A-weighted sum (Formula (D.1))::

    XiA,Fmax = 10 lg( sum_j 10**((Xi,Fmax,j + Aj)/10) )

over the one-third-octave bands 50 Hz to 630 Hz **or** the octave bands 63 Hz to
500 Hz, with the Table D.3 A-weighting corrections, rounded half-up to an
integer. One-third-octave measurements are rated in one-third octaves, never by
first summing them into octaves. See
:func:`a_weighted_maximum_impact_level`; :func:`heavy_impact_octave_levels`
implements the separate octave-band conversion of ISO 16283-2 Formula (20),
used when the *measurement* is reported in octaves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .._internal.validation import (
    require_choice,
    require_finite_array,
    require_positive,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "HEAVY_IMPACT_A_WEIGHTING",
    "HEAVY_IMPACT_OCTAVE_BANDS",
    "HEAVY_IMPACT_SOURCES",
    "AWeightedMaximumImpactResult",
    "HeavyImpactSourceCheck",
    "HeavyImpactSourceSpec",
    "StandardizedMaximumImpactResult",
    "a_weighted_maximum_impact_level",
    "check_heavy_impact_source",
    "fast_reverberation_correction",
    "heavy_impact_octave_levels",
    "heavy_impact_source_limits",
    "heavy_impact_source_specification",
    "impact_force_exposure_level",
    "standardized_maximum_impact_level",
]

#: Octave-band centre frequencies of the heavy-impact source specification
#: (ISO 16283-2:2020 Table A.1; JIS A 1418-2:2019 Tables A.1 and A.2), in Hz.
HEAVY_IMPACT_OCTAVE_BANDS: tuple[float, ...] = (31.5, 63.0, 125.0, 250.0, 500.0)

#: Impact force exposure level ``LFE`` and its tolerance per octave band, in
#: dB re 1 N, transcribed digit-for-digit from the primary texts. The rubber
#: ball is printed identically in **ISO 16283-2:2020 Table A.1** (printed p. 23),
#: **ISO 10140-5:2010 Table F.1** and **JIS A 1418-2:2019 Table A.2** (impact
#: force characteristic 2); the bang machine only in **JIS A 1418-2:2019
#: Table A.1** (impact force characteristic 1, printed p. 6). Keyed by source
#: name, each value is the tuple of ``(LFE, tolerance)`` pairs ordered as
#: :data:`HEAVY_IMPACT_OCTAVE_BANDS`.
HEAVY_IMPACT_SOURCES: dict[str, tuple[tuple[float, float], ...]] = {
    "rubber_ball": (
        (39.0, 1.0),
        (31.0, 1.5),
        (23.0, 1.5),
        (17.0, 2.0),
        (12.5, 2.0),
    ),
    "bang_machine": (
        (47.0, 1.0),
        (40.0, 1.5),
        (22.0, 1.5),
        (11.5, 2.0),
        (5.5, 2.0),
    ),
}

#: A-weighting correction ``Aj`` for the one-third-octave bands 50 Hz to 630 Hz
#: and the octave bands 63 Hz to 500 Hz, transcribed from **ISO 717-2:2020
#: Table D.3** (printed p. 21). Keyed by band width, each value maps the band
#: centre frequency in Hz to ``Aj`` in dB.
HEAVY_IMPACT_A_WEIGHTING: dict[str, dict[float, float]] = {
    "third": {
        50.0: -30.3, 63.0: -26.2, 80.0: -22.4, 100.0: -19.1,
        125.0: -16.2, 160.0: -13.2, 200.0: -10.8, 250.0: -8.7,
        315.0: -6.6, 400.0: -4.8, 500.0: -3.2, 630.0: -1.9,
    },
    "octave": {63.0: -26.2, 125.0: -16.2, 250.0: -8.7, 500.0: -3.2},
}

#: Reference force ``F0`` of Formula (A.1), in newtons.
_FORCE_REFERENCE = 1.0
#: Reference time interval ``Tref`` of Formula (A.1), in seconds.
_TIME_REFERENCE = 1.0
#: Reference reverberation time ``T0`` of Formula (5), in seconds.
_T0 = 0.5
#: Reference receiving-room volume ``V0`` of Formula (4), in m3 (dwellings).
_V0 = 50.0
#: Fast time-weighting factor of Formulae (5) and (6): ``C = T / 1,7275``.
_FAST_FACTOR = 1.7275
#: Half-width around ``C = 1`` where ``g(C)`` is replaced by its limit ``1/e``
#: (the closed form is 0/0 there and loses precision nearby).
_G_SINGULARITY = 1e-6


@dataclass(frozen=True)
class HeavyImpactSourceSpec:
    """Printed specification of a standard heavy and soft impact source.

    The mechanical figures are those of the *example of construction* clauses,
    which are informative in ISO 16283-2 (A.2.2) and in JIS A 1418-2 (Annex B);
    the force exposure levels and the impact duration are normative.

    :ivar name: ``"rubber_ball"`` or ``"bang_machine"``.
    :ivar frequencies: Octave-band centre frequencies, in Hz.
    :ivar force_exposure_level: Nominal ``LFE`` per band, in dB re 1 N.
    :ivar tolerance: Tolerance on ``LFE`` per band, in dB.
    :ivar drop_height: Free-fall drop height, in m.
    :ivar drop_height_tolerance: Tolerance on the drop height, in m.
    :ivar effective_mass: Effective (equivalent) mass of the source, in kg.
    :ivar effective_mass_tolerance: Tolerance on the effective mass, in kg.
    :ivar restitution: Coefficient of restitution.
    :ivar restitution_tolerance: Tolerance on the coefficient of restitution.
    :ivar contact_time: Nominal duration of the single-peak force pulse, in s
        (JIS A 1418-2:2019 A.2 b): ``20 +/- 2 ms`` for both characteristics).
    :ivar contact_time_tolerance: Tolerance on the contact time, in s.
    :ivar description: One-line description of the source.
    """

    name: str
    frequencies: tuple[float, ...]
    force_exposure_level: tuple[float, ...]
    tolerance: tuple[float, ...]
    drop_height: float
    drop_height_tolerance: float
    effective_mass: float
    effective_mass_tolerance: float
    restitution: float
    restitution_tolerance: float
    contact_time: float
    contact_time_tolerance: float
    description: str


#: The two source specifications, assembled from the printed clauses. The
#: rubber-ball geometry is ISO 16283-2:2020 A.2.2 (hollow 180 mm ball, 30 mm
#: wall) = JIS A 1418-2:2019 B.3; the bang-machine geometry is JIS A 1418-2:2019
#: B.2 (tyre at (2,4 +/- 0,2) x 10^5 Pa, dropped from 85 cm), absent from ISO.
_SPECS: dict[str, HeavyImpactSourceSpec] = {
    "rubber_ball": HeavyImpactSourceSpec(
        name="rubber_ball",
        frequencies=HEAVY_IMPACT_OCTAVE_BANDS,
        force_exposure_level=tuple(v for v, _ in HEAVY_IMPACT_SOURCES["rubber_ball"]),
        tolerance=tuple(t for _, t in HEAVY_IMPACT_SOURCES["rubber_ball"]),
        drop_height=1.00,
        drop_height_tolerance=0.01,
        effective_mass=2.5,
        effective_mass_tolerance=0.1,
        restitution=0.8,
        restitution_tolerance=0.1,
        contact_time=0.020,
        contact_time_tolerance=0.002,
        description=(
            "Hollow silicone-rubber ball, 180 mm outer diameter, 30 mm wall, "
            "dropped in free fall from 100 cm (impact force characteristic 2)"
        ),
    ),
    "bang_machine": HeavyImpactSourceSpec(
        name="bang_machine",
        frequencies=HEAVY_IMPACT_OCTAVE_BANDS,
        force_exposure_level=tuple(v for v, _ in HEAVY_IMPACT_SOURCES["bang_machine"]),
        tolerance=tuple(t for _, t in HEAVY_IMPACT_SOURCES["bang_machine"]),
        drop_height=0.85,
        drop_height_tolerance=0.0,
        effective_mass=7.3,
        effective_mass_tolerance=0.2,
        restitution=0.8,
        restitution_tolerance=0.1,
        contact_time=0.020,
        contact_time_tolerance=0.002,
        description=(
            "Car tyre inflated to (2,4 +/- 0,2) x 10^5 Pa, dropped in free "
            "fall from 85 cm (impact force characteristic 1)"
        ),
    ),
}


def heavy_impact_source_specification(source: str = "rubber_ball") -> HeavyImpactSourceSpec:
    """Printed specification of a standard heavy and soft impact source.

    :param source: ``"rubber_ball"`` (ISO 16283-2 Annex A.2, ISO 10140-5
        Annex F.2, JIS A 1418-2 impact force characteristic 2) or
        ``"bang_machine"`` (JIS A 1418-2 impact force characteristic 1).
    :return: The :class:`HeavyImpactSourceSpec`.
    :raises ValueError: for an unknown source name.
    """
    key = require_choice(source, "source", tuple(_SPECS))
    return _SPECS[key]


def heavy_impact_source_limits(
    source: str = "rubber_ball",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Octave-band ``LFE`` tolerance band of a heavy impact source.

    :param source: ``"rubber_ball"`` or ``"bang_machine"``.
    :return: ``(frequencies, lower, upper)``: the five octave-band centre
        frequencies in Hz and the ``LFE`` tolerance limits in dB re 1 N.
    :raises ValueError: for an unknown source name.
    """
    spec = heavy_impact_source_specification(source)
    nominal = np.asarray(spec.force_exposure_level, dtype=np.float64)
    tol = np.asarray(spec.tolerance, dtype=np.float64)
    return (
        np.asarray(spec.frequencies, dtype=np.float64),
        nominal - tol,
        nominal + tol,
    )


def impact_force_exposure_level(
    force: ArrayLike,
    sample_rate: float,
    *,
    reference_force: float = _FORCE_REFERENCE,
    reference_time: float = _TIME_REFERENCE,
) -> float:
    """Impact force exposure level ``LFE`` of a force pulse (Formula (A.1)).

    ``LFE = 10 lg[(1/Tref) integral F(t)**2 / F0**2 dt]`` dB re 1 N
    (ISO 16283-2:2020 Formula (A.1) = ISO 10140-5:2010 Formula (F.2) =
    JIS A 1418-2:2019 Formula (1)). The integral is taken over the whole
    supplied record with the trapezoidal rule, so pass one isolated impact.

    :param force: Sampled instantaneous force ``F(t)``, in newtons (1-D).
    :param sample_rate: Sampling rate of *force*, in hertz (> 0).
    :param reference_force: Reference force ``F0``, in newtons (Default: 1 N).
    :param reference_time: Reference time interval ``Tref``, in seconds
        (Default: 1 s).
    :return: The impact force exposure level ``LFE``, in dB re 1 N.
    :raises ValueError: for a malformed record or a non-positive parameter.
    """
    f = require_finite_array(force, "force")
    if f.size < 2:
        raise ValueError("'force' must be a 1-D record of at least two samples.")
    fs = require_positive(sample_rate, "sample_rate")
    f0 = require_positive(reference_force, "reference_force")
    tref = require_positive(reference_time, "reference_time")
    energy = float(np.trapezoid((f / f0) ** 2, dx=1.0 / fs))
    if energy <= 0.0:
        raise ValueError("'force' must carry non-zero energy.")
    return float(10.0 * np.log10(energy / tref))


@dataclass(frozen=True)
class HeavyImpactSourceCheck:
    """Conformance of a measured heavy impact source to its printed spectrum.

    :ivar source: ``"rubber_ball"`` or ``"bang_machine"``.
    :ivar frequencies: Octave-band centre frequencies, in Hz.
    :ivar measured: Measured impact force exposure level ``LFE``, in dB re 1 N.
    :ivar nominal: Printed nominal ``LFE`` per band, in dB re 1 N.
    :ivar tolerance: Printed tolerance per band, in dB.
    :ivar deviation: ``measured - nominal`` per band, in dB.
    :ivar within_tolerance: Per-band boolean mask of conforming bands.
    :ivar passed: ``True`` when every band conforms.
    """

    source: str
    frequencies: np.ndarray
    measured: np.ndarray
    nominal: np.ndarray
    tolerance: np.ndarray
    deviation: np.ndarray
    within_tolerance: np.ndarray
    passed: bool

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the measured ``LFE`` against the printed tolerance band.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_heavy_impact_source

        check_language(language)
        return plot_heavy_impact_source(self, ax=ax, language=language, **kwargs)


def check_heavy_impact_source(
    force_exposure_level: ArrayLike, source: str = "rubber_ball"
) -> HeavyImpactSourceCheck:
    """Check a measured heavy impact source against its printed spectrum.

    Compares the five measured octave-band impact force exposure levels with
    the tolerance band of ISO 16283-2:2020 Table A.1 / ISO 10140-5:2010
    Table F.1 / JIS A 1418-2:2019 Table A.2 (rubber ball) or JIS A 1418-2:2019
    Table A.1 (bang machine).

    :param force_exposure_level: Measured ``LFE`` in the five octave bands
        31,5 Hz to 500 Hz, in dB re 1 N.
    :param source: ``"rubber_ball"`` or ``"bang_machine"``.
    :return: A :class:`HeavyImpactSourceCheck`.
    :raises ValueError: for an unknown source or a wrong number of bands.
    """
    spec = heavy_impact_source_specification(source)
    measured = require_finite_array(force_exposure_level, "force_exposure_level")
    n = len(spec.frequencies)
    if measured.shape != (n,):
        raise ValueError(
            f"'force_exposure_level' must hold {n} octave-band values "
            f"(31,5 Hz to 500 Hz)."
        )
    nominal = np.asarray(spec.force_exposure_level, dtype=np.float64)
    tol = np.asarray(spec.tolerance, dtype=np.float64)
    deviation = measured - nominal
    within = np.abs(deviation) <= tol
    return HeavyImpactSourceCheck(
        source=spec.name,
        frequencies=np.asarray(spec.frequencies, dtype=np.float64),
        measured=measured,
        nominal=nominal,
        tolerance=tol,
        deviation=deviation,
        within_tolerance=within,
        passed=bool(np.all(within)),
    )


def _fast_peak_factor(c: np.ndarray) -> np.ndarray:
    """``g(C)``: peak of the Fast-weighted response of a decaying burst.

    The bracket of ISO 16283-2:2020 Formula (4) is ``g(C)/g(C0)`` with
    ``g(C) = (C**(1/(1-C)) - C**(-1/(1-1/C))) / (1 - 1/C)``. ``g`` has a
    removable singularity at ``C = 1`` where its limit is ``1/e``.
    """
    out = np.empty_like(c)
    near = np.abs(c - 1.0) <= _G_SINGULARITY
    out[near] = 1.0 / np.e
    far = ~near
    cf = c[far]
    out[far] = (cf ** (1.0 / (1.0 - cf)) - cf ** (-1.0 / (1.0 - 1.0 / cf))) / (
        1.0 - 1.0 / cf
    )
    return out


def fast_reverberation_correction(
    reverberation_time: ArrayLike, *, reference_time: float = _T0
) -> np.ndarray:
    """Fast time-weighting reverberation correction (Formulae (4), (5), (6)).

    The term ``10 lg[g(C)/g(C0)]`` subtracted in ISO 16283-2:2020 Formula (4),
    with ``C = T/1,7275`` (Formula (6)) and ``C0 = T0/1,7275`` (Formula (5)).
    It is the *maximum-level* counterpart of the energy-average ``10 lg(T/T0)``:
    a Fast detector never integrates more than about 1,7 s of decay, so the
    correction saturates instead of growing without bound. It is exactly 0 dB
    when ``T = T0``.

    :param reverberation_time: Receiving-room reverberation time ``T`` per
        band, in seconds (> 0).
    :param reference_time: Reference reverberation time ``T0``, in seconds
        (Default: 0,5 s).
    :return: The correction per band, in dB.
    :raises ValueError: for a non-positive reverberation time.
    """
    t = require_finite_array(reverberation_time, "reverberation_time")
    if np.any(t <= 0.0):
        raise ValueError("'reverberation_time' must be positive and finite.")
    t0 = require_positive(reference_time, "reference_time")
    c = t / _FAST_FACTOR
    c0 = np.full_like(c, t0 / _FAST_FACTOR)
    return np.asarray(
        10.0 * np.log10(_fast_peak_factor(c) / _fast_peak_factor(c0)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class StandardizedMaximumImpactResult:
    """Standardized maximum impact sound pressure level (ISO 16283-2 3.16).

    :ivar frequencies: Band centre frequencies, in Hz, or ``None``.
    :ivar measured: Energy-averaged maximum level ``Li,Fmax`` per band, in dB.
    :ivar standardized: ``L'i,Fmax,V,T`` per band, in dB.
    :ivar volume_term: The volume correction ``10 lg(V/V0)``, in dB (scalar).
    :ivar reverberation_correction: The ``10 lg[g(C)/g(C0)]`` term subtracted
        per band, in dB.
    :ivar volume: Receiving-room volume ``V``, in m3.
    :ivar reverberation_time: Receiving-room reverberation time ``T`` per band,
        in seconds.
    """

    frequencies: np.ndarray | None
    measured: np.ndarray
    standardized: np.ndarray
    volume_term: float
    reverberation_correction: np.ndarray
    volume: float
    reverberation_time: np.ndarray

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot ``Li,Fmax`` and ``L'i,Fmax,V,T`` per band.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_standardized_maximum_impact

        check_language(language)
        return plot_standardized_maximum_impact(self, ax=ax, language=language, **kwargs)


def standardized_maximum_impact_level(
    level: ArrayLike,
    volume: float,
    reverberation_time: ArrayLike,
    *,
    frequency: ArrayLike | None = None,
    reference_time: float = _T0,
    reference_volume: float = _V0,
) -> StandardizedMaximumImpactResult:
    """Standardized maximum impact level ``L'i,Fmax,V,T`` (Formulae (4)-(6)).

    ``L'i,Fmax,V,T = Li,Fmax + 10 lg(V/V0) - 10 lg[g(C)/g(C0)]``
    (ISO 16283-2:2020, definition 3.16), the field quantity used to rate a
    floor excited by the rubber ball. The reverberation term is
    :func:`fast_reverberation_correction`.

    :param level: Energy-averaged maximum impact sound pressure level
        ``Li,Fmax`` per band, in dB.
    :param volume: Receiving-room volume ``V``, in m3 (> 0).
    :param reverberation_time: Receiving-room reverberation time ``T`` per
        band, in seconds (> 0); a scalar is broadcast over the bands.
    :param frequency: Optional band centre frequencies, in Hz.
    :param reference_time: Reference reverberation time ``T0``, in seconds
        (Default: 0,5 s).
    :param reference_volume: Reference volume ``V0``, in m3 (Default: 50 m3
        for dwellings).
    :return: A :class:`StandardizedMaximumImpactResult`.
    :raises ValueError: for mismatched shapes or non-positive inputs.
    """
    li = require_finite_array(level, "level")
    v = require_positive(volume, "volume")
    v0 = require_positive(reference_volume, "reference_volume")
    t = require_finite_array(reverberation_time, "reverberation_time")
    if t.size == 1:
        t = np.full(li.shape, float(t[0]))
    if t.shape != li.shape:
        raise ValueError("'reverberation_time' must be a scalar or match 'level'.")
    freqs: np.ndarray | None = None
    if frequency is not None:
        freqs = require_finite_array(frequency, "frequency")
        if freqs.shape != li.shape:
            raise ValueError("'frequency' must match 'level'.")
    correction = fast_reverberation_correction(t, reference_time=reference_time)
    volume_term = float(10.0 * np.log10(v / v0))
    return StandardizedMaximumImpactResult(
        frequencies=freqs,
        measured=li,
        standardized=np.asarray(li + volume_term - correction, dtype=np.float64),
        volume_term=volume_term,
        reverberation_correction=correction,
        volume=v,
        reverberation_time=t,
    )


def heavy_impact_octave_levels(level: ArrayLike) -> np.ndarray:
    """Combine one-third-octave maximum levels into octaves (Formula (20)).

    ``L'i,Fmax,V,T,oct = 10 lg( sum_{n=1..3} 10**(L'i,Fmax,V,T,1/3oct,n / 10) )``
    (ISO 16283-2:2020, printed p. 19). The input length must be a multiple of
    three and the bands must be in ascending order, three per octave.

    Note that ISO 717-2:2020 Annex D forbids using this conversion as a route
    to the single number: a one-third-octave measurement is rated in one-third
    octaves with :func:`a_weighted_maximum_impact_level`.

    :param level: One-third-octave levels in ascending order, in dB.
    :return: The octave-band levels, in dB.
    :raises ValueError: when the length is not a positive multiple of three.
    """
    x = require_finite_array(level, "level")
    if x.size % 3 != 0:
        raise ValueError("'level' must be a 1-D array whose length is a multiple of 3.")
    return np.asarray(
        10.0 * np.log10(np.sum(10.0 ** (x.reshape(-1, 3) / 10.0), axis=1)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class AWeightedMaximumImpactResult:
    """A-weighted maximum impact sound pressure level (ISO 717-2 Annex D).

    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar band: ``"third"`` (50-630 Hz) or ``"octave"`` (63-500 Hz).
    :ivar levels: Input maximum levels ``Xi,Fmax,j`` per band, in dB.
    :ivar a_weighting: Table D.3 correction ``Aj`` per band, in dB.
    :ivar corrected: ``Xi,Fmax,j + Aj`` per band, in dB.
    :ivar unrounded: The Formula (D.1) sum before rounding, in dB.
    :ivar rating: The single-number quantity ``XiA,Fmax``, rounded half-up to
        an integer, in dB.
    """

    frequencies: np.ndarray
    band: str
    levels: np.ndarray
    a_weighting: np.ndarray
    corrected: np.ndarray
    unrounded: float
    rating: int

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the band levels, their A-weighted contributions and the rating.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_a_weighted_maximum_impact

        check_language(language)
        return plot_a_weighted_maximum_impact(self, ax=ax, language=language, **kwargs)


def _round_half_up(value: float) -> int:
    """Round to an integer, halves away from zero (ISO 717-2 Annex D note a)."""
    return int(np.floor(value + 0.5)) if value >= 0.0 else -int(np.floor(-value + 0.5))


def a_weighted_maximum_impact_level(
    level: ArrayLike,
    frequency: ArrayLike | Sequence[float] | None = None,
    *,
    band: str | None = None,
) -> AWeightedMaximumImpactResult:
    """A-weighted maximum impact level ``XiA,Fmax`` (ISO 717-2:2020 Formula (D.1)).

    ``XiA,Fmax = 10 lg( sum_j 10**((Xi,Fmax,j + Aj)/10) )``, rounded half-up to
    an integer, over the one-third-octave bands 50 Hz to 630 Hz (12 values) or
    the octave bands 63 Hz to 500 Hz (4 values), with the A-weighting
    corrections of Table D.3. The same formula rates ``LiA,Fmax``,
    ``LiA,Fmax,V,T``, ``L'iA,Fmax`` and ``L'iA,Fmax,V,T``.

    :param level: Maximum impact sound pressure levels ``Xi,Fmax,j``, in dB,
        in ascending band order.
    :param frequency: Optional band centre frequencies, in Hz. When given they
        must be exactly the rating bands of the chosen band width.
    :param band: ``"third"`` or ``"octave"``; inferred from the number of
        values (12 -> third, 4 -> octave) when omitted.
    :return: An :class:`AWeightedMaximumImpactResult`.
    :raises ValueError: for a wrong number of values or mismatched bands.
    """
    x = require_finite_array(level, "level")
    if band is None:
        band = {12: "third", 4: "octave"}.get(x.size, "")
        if not band:
            raise ValueError(
                "'level' must hold 12 one-third-octave values (50-630 Hz) or "
                "4 octave values (63-500 Hz); pass 'band' to disambiguate."
            )
    chosen = require_choice(band, "band", ("third", "octave"))
    table = HEAVY_IMPACT_A_WEIGHTING[chosen]
    bands = np.asarray(sorted(table), dtype=np.float64)
    if x.shape != bands.shape:
        raise ValueError(
            f"'level' must hold {bands.size} values for band={chosen!r} "
            f"({bands[0]:g} Hz to {bands[-1]:g} Hz)."
        )
    if frequency is not None:
        given = require_finite_array(frequency, "frequency")
        if given.shape != bands.shape or not np.allclose(given, bands):
            raise ValueError(
                f"'frequency' must be the ISO 717-2 Annex D rating bands for "
                f"band={chosen!r}: {[float(b) for b in bands]}."
            )
    a_weighting = np.asarray([table[float(b)] for b in bands], dtype=np.float64)
    corrected = x + a_weighting
    unrounded = float(10.0 * np.log10(np.sum(10.0 ** (corrected / 10.0))))
    return AWeightedMaximumImpactResult(
        frequencies=bands,
        band=chosen,
        levels=x,
        a_weighting=a_weighting,
        corrected=corrected,
        unrounded=unrounded,
        rating=_round_half_up(unrounded),
    )
