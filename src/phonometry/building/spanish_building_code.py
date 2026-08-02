#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Spanish building code CTE DB-HR: global indices and requirement checks.

The *Documento Basico HR Proteccion frente al ruido* of the Spanish Codigo
Tecnico de la Edificacion states its requirements in A-weighted global
quantities that are close relatives of, but not identical to, the ISO 717-1
weighted ratings: ``RA``, ``RA,tr``, ``DnT,A`` and ``D2m,nT,Atr``.

**The global index (Annex A, Formulae (A.5) to (A.7)).** Instead of shifting a
reference curve, DB-HR weights the measured one-third-octave insulation with a
normalised A-weighted source spectrum and sums it energetically:

.. math::

   I_x = -10 \log_{10} \sum_i 10^{(L_{x,i} - X_i)/10} \quad [\mathrm{dBA}]

where ``X_i`` is the band insulation (``R``, ``R'``, ``DnT``, ``D2m,nT`` ...)
and ``L_x,i`` the normalised spectrum of the source. The sum runs over the
**eighteen** one-third-octave bands 100 Hz to 5 kHz, two more than ISO 717-1's
sixteen: the 4 kHz and 5 kHz bands.

**Which quantity a facade is assessed in (clause 3.1.3.4 point 1).** The name
of the global quantity depends on the dominant outdoor noise, and it is not a
free choice:

* dominant **railway** noise (or railway stations) is assessed as
  ``D2m,nT,A`` through Formula (A.5), using the Table A.4 railway spectrum;
* dominant **road traffic** or **aircraft** noise is assessed as
  ``D2m,nT,Atr`` through Formula (A.6), using the Table A.3 road-traffic or
  the Table A.2 aircraft spectrum.

Table H.1 prints the same split. The railway spectrum of Table A.4 happens to
be numerically identical to the road-traffic spectrum of Table A.3, so the two
routes give the same number for a rail-dominant site; the *name* of the
reported quantity still differs, which is why :func:`d2m_nt_a` and
:func:`d2m_nt_atr` are separate entry points rather than one function with a
spectrum switch.

**Relationship with the ISO 717-1 route.** DB-HR Annex H accepts
``DnT,w + C`` as an approximation of ``DnT,A`` (H.1), ``D2m,nT,w + C`` of
``D2m,nT,A`` for trains (H.2) and ``D2m,nT,w + Ctr`` of ``D2m,nT,Atr`` for
road traffic (H.3), valid while the two routes differ by less than 1 dB.
Annex H writes those terms as plain ``C`` and ``Ctr`` and refers their
definition to UNE-EN ISO 717-1 without narrowing the frequency range; the
reading that they must be the *enlarged-range* terms ``C100-5000`` and
``Ctr,100-5000``, because the DB-HR indices run to 5 kHz while the ISO 717-1
core range stops at 3150 Hz, is the one the Spanish literature makes explicit
(Aviles Lopez & Perera Martin, note to expressions [7.15] and [7.16]). The
library keeps both routes: this module implements the direct Formula
(A.5)-(A.7) route, which is the normative one for DB-HR, and
:func:`~phonometry.building.insulation.weighted_rating_extended` supplies the
ISO 717-1 route with the enlarged-range adaptation terms.

**Rounding.** DB-HR 3.1.3.1 point 4 requires the final values of the
quantities that define the requirements to be expressed rounded to an integer,
while intermediate values carry one decimal. The result objects therefore
expose the exact value, the one-decimal intermediate and the integer used in
the compliance check.

**Requirements (Clause 2).** Airborne insulation between rooms and against the
outside (Table 2.1, keyed on the day noise index ``Ld`` of the site), impact
sound pressure level, and reverberation time in classrooms, conference halls,
dining rooms and restaurants.

**Window size (Catalogo de Elementos Constructivos).** Windows are tested on
about 1,8 m2 specimens; larger windows insulate less, and the CEC corrects
``RA`` and ``RA,tr`` by 0 to -3 dB by total window area.

.. note::
   ``"sanitary"`` does not mean the same thing here as in
   :mod:`phonometry.environment.assessment.spain`. In Table 2.1 of DB-HR
   it is the *non-hospital* ambulatory health use of footnote (1) (medical
   practices, consulting rooms), deliberately distinct from ``"hospital"``;
   in the RD 1367/2007 tables it is the *hospitalario* use itself. Both names
   reach the top-level :mod:`phonometry` namespace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "DB_HR_FREQUENCIES",
    "DB_HR_NORMALISED_SPECTRA",
    "DbHrAssessment",
    "DbHrCheck",
    "DbHrGlobalIndexResult",
    "DbHrRequirement",
    "assess_db_hr",
    "check_db_hr_requirement",
    "d2m_nt_a",
    "d2m_nt_atr",
    "db_hr_airborne_requirement",
    "db_hr_facade_requirement",
    "db_hr_global_index",
    "db_hr_impact_requirement",
    "db_hr_party_wall_requirement",
    "db_hr_reverberation_requirement",
    "dnt_a",
    "ra",
    "ra_tr",
    "window_size_correction",
]

#: The eighteen one-third-octave band centre frequencies of DB-HR Annex A.
DB_HR_FREQUENCIES: tuple[float, ...] = (
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
    800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
)

#: Normalised A-weighted source spectra of DB-HR Annex A, in dBA.
#: ``"pink"`` is Table A.5 (``LAr,i``), ``"traffic"`` Table A.3 (``LAtr,i``),
#: ``"railway"`` Table A.4 (``LAef,i``, numerically equal to Table A.3) and
#: ``"aircraft"`` Table A.2 (``LAav,i``).
DB_HR_NORMALISED_SPECTRA: Mapping[str, tuple[float, ...]] = MappingProxyType({
    "pink": (
        -30.1, -27.1, -24.4, -21.9, -19.6, -17.6, -15.8, -14.2, -12.9,
        -11.8, -11.0, -10.4, -10.0, -9.8, -9.7, -9.8, -10.0, -10.5,
    ),
    "traffic": (
        -20.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0,
        -9.0, -8.0, -9.0, -10.0, -11.0, -13.0, -15.0, -16.0, -18.0,
    ),
    "railway": (
        -20.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0,
        -9.0, -8.0, -9.0, -10.0, -11.0, -13.0, -15.0, -16.0, -18.0,
    ),
    "aircraft": (
        -23.8, -20.2, -15.4, -13.1, -12.6, -10.4, -9.8, -9.5, -8.7,
        -9.5, -10.5, -11.0, -12.5, -14.9, -15.9, -18.6, -23.3, -29.9,
    ),
})

#: The DB-HR table each normalised spectrum is read from.
_SPECTRUM_TABLES: dict[str, str] = {
    "pink": "Table A.5 (LAr,i)",
    "traffic": "Table A.3 (LAtr,i)",
    "railway": "Table A.4 (LAef,i)",
    "aircraft": "Table A.2 (LAav,i)",
}

#: Window-area breakpoints and corrections of the CTE Catalogo de Elementos
#: Constructivos: (upper area bound in m2 inclusive, correction in dB).
_WINDOW_SIZE_CORRECTIONS: tuple[tuple[float, int], ...] = (
    (2.7, 0),
    (3.6, -1),
    (4.6, -2),
)
#: Correction beyond the last breakpoint, in dB.
_WINDOW_SIZE_CORRECTION_LARGE = -3

#: Band-centre matching tolerance (fractional).
_BAND_TOLERANCE = 0.06

#: DB-HR Table 2.1: required ``D2m,nT,Atr`` in dBA by ``Ld`` band. The two
#: series are the "bedroom-like" and the "living-room-like" columns; the table
#: pairs them differently for residential/hospital and for the other uses.
_FACADE_SERIES_A: tuple[int, ...] = (30, 32, 37, 42, 47)
_FACADE_SERIES_B: tuple[int, ...] = (30, 30, 32, 37, 42)

#: Upper ``Ld`` bound (inclusive) of each row of Table 2.1, in dBA.
_FACADE_LD_BOUNDS: tuple[float, ...] = (60.0, 65.0, 70.0, 75.0)

#: Which Table 2.1 series applies to each (building use, room type).
_FACADE_ROWS: dict[tuple[str, str], tuple[int, ...]] = {
    ("residential", "bedrooms"): _FACADE_SERIES_A,
    ("residential", "living"): _FACADE_SERIES_B,
    ("hospital", "bedrooms"): _FACADE_SERIES_A,
    ("hospital", "living"): _FACADE_SERIES_B,
    ("cultural", "living"): _FACADE_SERIES_A,
    ("cultural", "classrooms"): _FACADE_SERIES_B,
    ("sanitary", "living"): _FACADE_SERIES_A,
    ("sanitary", "classrooms"): _FACADE_SERIES_B,
    ("educational", "living"): _FACADE_SERIES_A,
    ("educational", "classrooms"): _FACADE_SERIES_B,
    ("administrative", "living"): _FACADE_SERIES_A,
    ("administrative", "classrooms"): _FACADE_SERIES_B,
}

#: Accepted spellings of a building use in Table 2.1.
_USE_ALIASES: dict[str, str] = {
    "dwelling": "residential",
    "housing": "residential",
    "hospitalario": "hospital",
    "docente": "educational",
    "teaching": "educational",
    "office": "administrative",
    "offices": "administrative",
}

#: Accepted spellings of a room type in Table 2.1.
_ROOM_ALIASES: dict[str, str] = {
    "bedroom": "bedrooms",
    "dormitorios": "bedrooms",
    "dormitories": "bedrooms",
    "living_room": "living",
    "living_rooms": "living",
    "estancias": "living",
    "classroom": "classrooms",
    "aulas": "classrooms",
}

#: Extra ``D2m,nT,Atr``, in dBA, when aircraft noise dominates (Clause 2.1.1
#: a) iv, fifth bullet).
_AIRCRAFT_INCREMENT = 4.0
#: ``Ld`` reduction, in dBA, for a facade sheltered from the dominant noise
#: (Clause 2.1.1 a) iv, fourth bullet).
_QUIET_FACADE_REDUCTION = 10.0


def _finite(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting non-finite input."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"'{name}' must be finite.")
    return scalar


def _positive(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting non-positive or non-finite input."""
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"'{name}' must be a positive, finite number.")
    return scalar


def _round_half_up(value: float, decimals: int = 0) -> float:
    """Round half-up to ``decimals`` places (DB-HR 3.1.3.1 point 4)."""
    scale = 10.0**decimals
    return math.floor(value * scale + 0.5) / scale


def _normalise(value: str, aliases: Mapping[str, str], name: str) -> str:
    """Lower-case ``value``, normalise separators and resolve aliases."""
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string; got {value!r}.")  # noqa: TRY004 - ValueError keeps the module validation errors uniform
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    return aliases.get(key, key)


# --------------------------------------------------------------------------- #
# Global indices (Annex A, Formulae (A.5) to (A.7))
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DbHrGlobalIndexResult:
    """A DB-HR A-weighted global insulation index.

    :ivar value: The index, in dBA, unrounded.
    :ivar intermediate: The index rounded to one decimal, the form DB-HR
        prescribes for intermediate quantities and for product specifications.
    :ivar reported: The index rounded to an integer, the form DB-HR requires
        for the quantities that define the requirements (3.1.3.1 point 4).
    :ivar name: The index name, e.g. ``"RA"`` or ``"D2m,nT,Atr"``.
    :ivar spectrum: Key of the normalised spectrum used.
    :ivar frequencies: Band centre frequencies, in Hz.
    :ivar band_values: The band insulation values ``X_i``, in dB.
    :ivar spectrum_levels: The normalised spectrum ``L_x,i``, in dBA.
    :ivar band_contributions: Per-band transmitted level ``L_x,i - X_i``, in
        dBA; the index is minus their energy sum.
    :ivar reference: The DB-HR Annex A table the normalised spectrum was read
        from.
    """

    value: float
    intermediate: float
    reported: int
    name: str
    spectrum: str
    frequencies: np.ndarray
    band_values: np.ndarray
    spectrum_levels: np.ndarray
    band_contributions: np.ndarray
    reference: str

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the band insulation with the weighted per-band contributions."""
        from .._i18n import check_language
        from .._plot.building import plot_db_hr_global_index

        return plot_db_hr_global_index(
            self, ax=ax, language=check_language(language), **kwargs
        )


def _match_bands(frequencies: np.ndarray) -> np.ndarray:
    """Indices of the eighteen DB-HR bands within ``frequencies``."""
    indices: list[int] = []
    for target in DB_HR_FREQUENCIES:
        hits = np.nonzero(np.abs(frequencies - target) <= _BAND_TOLERANCE * target)[0]
        if hits.size != 1:
            raise ValueError(
                "The input must contain exactly one band at each of the "
                "eighteen DB-HR one-third-octave centre frequencies "
                f"100 Hz to 5 kHz; {target:g} Hz matched {hits.size} bands."
            )
        indices.append(int(hits[0]))
    return np.asarray(indices, dtype=np.intp)


def db_hr_global_index(
    band_values: Sequence[float] | np.ndarray,
    spectrum: str = "pink",
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
    name: str | None = None,
) -> DbHrGlobalIndexResult:
    r"""A-weighted global insulation index per DB-HR Annex A.

    :math:`I_x = -10 \log_{10} \sum_i 10^{(L_{x,i} - X_i)/10}` over the eighteen
    one-third-octave bands 100 Hz to 5 kHz (Formulae (A.5) to (A.7)).

    :param band_values: The band insulation ``X_i``, in dB. Either exactly the
        eighteen DB-HR bands, or a longer spectrum together with
        ``frequencies``, from which the eighteen bands are selected.
    :param spectrum: Normalised spectrum key: ``"pink"`` (default),
        ``"traffic"``, ``"railway"`` or ``"aircraft"``.
    :param frequencies: Band centre frequencies, in Hz, one per value.
        ``None`` assumes exactly the eighteen DB-HR bands in order.
    :param name: Optional index name carried on the result (e.g. ``"R'A"``).
    :return: A :class:`DbHrGlobalIndexResult`.
    :raises ValueError: For an unknown spectrum, a wrong band count, or
        non-finite values.
    """
    key = str(spectrum).strip().lower()
    if key not in DB_HR_NORMALISED_SPECTRA:
        raise ValueError(
            "'spectrum' must be one of "
            f"{sorted(DB_HR_NORMALISED_SPECTRA)}; got {spectrum!r}."
        )
    values = np.asarray(band_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("'band_values' must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("'band_values' must contain finite values.")

    if frequencies is None:
        if values.size != len(DB_HR_FREQUENCIES):
            raise ValueError(
                "Without 'frequencies' the input must be the eighteen DB-HR "
                "one-third-octave bands 100 Hz to 5 kHz; got "
                f"{values.size} values."
            )
        freqs = np.asarray(DB_HR_FREQUENCIES, dtype=np.float64)
        # Copy so a later mutation of the caller's array cannot rewrite the
        # values this result reports.
        selected = values.copy()
    else:
        given = np.asarray(frequencies, dtype=np.float64)
        if given.shape != values.shape:
            raise ValueError(
                "'frequencies' must have one value per band of 'band_values'."
            )
        if not np.all(np.isfinite(given)) or np.any(given <= 0.0):
            raise ValueError("'frequencies' must contain positive values.")
        index = _match_bands(given)
        freqs = given[index]
        selected = values[index]

    levels = np.asarray(DB_HR_NORMALISED_SPECTRA[key], dtype=np.float64)
    contributions = levels - selected
    value = float(-10.0 * np.log10(np.sum(10.0 ** (contributions / 10.0))))
    return DbHrGlobalIndexResult(
        value=value,
        intermediate=_round_half_up(value, 1),
        reported=int(_round_half_up(value)),
        name=name if name is not None else "IA",
        spectrum=key,
        frequencies=freqs,
        band_values=selected,
        spectrum_levels=levels,
        band_contributions=contributions,
        reference=f"CTE DB-HR Annex A, {_SPECTRUM_TABLES[key]}",
    )


def ra(
    reduction_index: Sequence[float] | np.ndarray,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> DbHrGlobalIndexResult:
    """A-weighted global sound reduction index ``RA`` for pink noise.

    :param reduction_index: Band sound reduction index ``R`` (or ``R'``), in dB.
    :param frequencies: Band centre frequencies, in Hz (optional).
    :return: A :class:`DbHrGlobalIndexResult` named ``"RA"``.
    """
    return db_hr_global_index(reduction_index, "pink", frequencies=frequencies, name="RA")


def ra_tr(
    reduction_index: Sequence[float] | np.ndarray,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> DbHrGlobalIndexResult:
    """A-weighted global sound reduction index ``RA,tr`` for road-traffic noise.

    :param reduction_index: Band sound reduction index ``R`` (or ``R'``), in dB.
    :param frequencies: Band centre frequencies, in Hz (optional).
    :return: A :class:`DbHrGlobalIndexResult` named ``"RA,tr"``.
    """
    return db_hr_global_index(
        reduction_index, "traffic", frequencies=frequencies, name="RA,tr"
    )


def dnt_a(
    level_difference: Sequence[float] | np.ndarray,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
) -> DbHrGlobalIndexResult:
    """A-weighted standardized level difference ``DnT,A`` (Formula (A.7)).

    :param level_difference: Band standardized level difference ``DnT``, in dB.
    :param frequencies: Band centre frequencies, in Hz (optional).
    :return: A :class:`DbHrGlobalIndexResult` named ``"DnT,A"``.
    """
    return db_hr_global_index(
        level_difference, "pink", frequencies=frequencies, name="DnT,A"
    )


def d2m_nt_a(
    level_difference: Sequence[float] | np.ndarray,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
    spectrum: str = "pink",
) -> DbHrGlobalIndexResult:
    """A-weighted facade level difference ``D2m,nT,A`` (Formula (A.5)).

    The pink-noise global quantity of a facade, a roof or a floor in contact
    with the outside air. Clause 3.1.3.4 point 1 makes this the quantity a
    **railway-dominant** site is assessed in, evaluated through Formula (A.5)
    with the Table A.4 railway spectrum; Table H.1 prints the same split and
    Annex H (H.2) gives ``D2m,nT,w + C`` as its ISO 717-1 approximation.

    For a road-traffic or aircraft dominant site the quantity is
    ``D2m,nT,Atr`` instead: use :func:`d2m_nt_atr`.

    :param level_difference: Band standardized facade level difference
        ``D2m,nT``, in dB.
    :param frequencies: Band centre frequencies, in Hz (optional).
    :param spectrum: ``"pink"`` (default, Table A.5) or ``"railway"``
        (Table A.4), per the dominant outdoor noise.
    :return: A :class:`DbHrGlobalIndexResult` named ``"D2m,nT,A"``.
    :raises ValueError: For a spectrum Formula (A.5) does not cover.
    """
    key = str(spectrum).strip().lower()
    if key not in ("pink", "railway"):
        raise ValueError(
            "Formula (A.5) evaluates 'D2m,nT,A' for pink noise or, on a "
            "railway-dominant site, the Table A.4 railway spectrum; "
            f"got {spectrum!r}. Road-traffic and aircraft noise are assessed "
            "as 'D2m,nT,Atr' through Formula (A.6): use d2m_nt_atr()."
        )
    return db_hr_global_index(
        level_difference, key, frequencies=frequencies, name="D2m,nT,A"
    )


def d2m_nt_atr(
    level_difference: Sequence[float] | np.ndarray,
    *,
    frequencies: Sequence[float] | np.ndarray | None = None,
    spectrum: str = "traffic",
) -> DbHrGlobalIndexResult:
    """A-weighted facade level difference ``D2m,nT,Atr`` (Formula (A.6)).

    The road-traffic global quantity of a facade, a roof or a floor in contact
    with the outside air. Clause 3.1.3.4 point 1 makes this the quantity a
    **road-traffic or aircraft** dominant site is assessed in, the aircraft
    case evaluated through the same Formula (A.6) with the Table A.2 aircraft
    spectrum.

    A railway-dominant site is assessed as ``D2m,nT,A`` through Formula (A.5)
    instead: use :func:`d2m_nt_a`. The railway spectrum of Table A.4 is
    numerically identical to the road-traffic spectrum of Table A.3, so the
    two give the same number, but the quantity that the requirement and the
    report are stated in is not the same.

    :param level_difference: Band standardized facade level difference
        ``D2m,nT``, in dB.
    :param frequencies: Band centre frequencies, in Hz (optional).
    :param spectrum: ``"traffic"`` (default, Table A.3) or ``"aircraft"``
        (Table A.2), per the dominant outdoor noise.
    :return: A :class:`DbHrGlobalIndexResult` named ``"D2m,nT,Atr"``.
    :raises ValueError: For a spectrum Formula (A.6) does not cover.
    """
    key = str(spectrum).strip().lower()
    if key not in ("traffic", "aircraft"):
        raise ValueError(
            "Formula (A.6) evaluates 'D2m,nT,Atr' for road-traffic noise or, "
            "on an aircraft-dominant site, the Table A.2 aircraft spectrum; "
            f"got {spectrum!r}. Railway noise is assessed as 'D2m,nT,A' "
            "through Formula (A.5): use d2m_nt_a()."
        )
    return db_hr_global_index(
        level_difference, key, frequencies=frequencies, name="D2m,nT,Atr"
    )


def window_size_correction(area: float) -> int:
    """Window-size correction of ``RA`` / ``RA,tr``, in dB (CTE CEC).

    Windows are tested on specimens of about 1,8 m2; a larger window insulates
    less. The Catalogo de Elementos Constructivos of the CTE corrects the
    catalogue value by 0 dB up to 2,7 m2, -1 dB over 2,7 to 3,6 m2, -2 dB over
    3,6 to 4,6 m2 and -3 dB above 4,6 m2.

    :param area: Total window area, in m2 (positive).
    :return: The correction, in dB (0, -1, -2 or -3), to be *added* to the
        catalogue ``RA`` or ``RA,tr``.
    :raises ValueError: If ``area`` is not positive and finite.
    """
    value = _positive(area, "area")
    for bound, correction in _WINDOW_SIZE_CORRECTIONS:
        if value <= bound:
            return correction
    return _WINDOW_SIZE_CORRECTION_LARGE


# --------------------------------------------------------------------------- #
# Requirements (Clause 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DbHrRequirement:
    """A single DB-HR performance requirement.

    :ivar quantity: The quantity the requirement is stated in, e.g.
        ``"DnT,A"``, ``"L'nT,w"``, ``"RA"``, ``"T"`` or ``"A/V"``.
    :ivar limit: The limit value, in the quantity's own unit.
    :ivar direction: ``"min"`` when the quantity must be at least the limit,
        ``"max"`` when it must be at most the limit.
    :ivar unit: The unit the limit is expressed in.
    :ivar decimals: Decimal places the achieved value is rounded to before the
        comparison (DB-HR 3.1.3.1 point 4 for the dB quantities, 3.2.1 for the
        reverberation time).
    :ivar reference: The DB-HR clause the requirement comes from.
    :ivar description: The case it applies to, in plain words.
    """

    quantity: str
    limit: float
    direction: str
    unit: str
    decimals: int
    reference: str
    description: str

    def __post_init__(self) -> None:
        """Validate the comparison direction.

        :func:`check_db_hr_requirement` branches on ``"min"`` and treats
        anything else as ``"max"``, so an unchecked typo would produce a
        confident, inverted verdict.
        """
        if self.direction not in ("min", "max"):
            raise ValueError(
                "'direction' must be 'min' (the quantity must reach the "
                "limit) or 'max' (it must not exceed it); got "
                f"{self.direction!r}."
            )


@dataclass(frozen=True)
class DbHrCheck:
    """A DB-HR requirement checked against an achieved value.

    :ivar requirement: The :class:`DbHrRequirement` checked.
    :ivar value: The achieved value, unrounded.
    :ivar reported: The achieved value rounded as DB-HR prescribes.
    :ivar margin: ``reported - limit`` for a ``"min"`` requirement and
        ``limit - reported`` for a ``"max"`` one; non-negative when compliant.
    :ivar complies: Whether the requirement is met.
    """

    requirement: DbHrRequirement
    value: float
    reported: float
    margin: float
    complies: bool


@dataclass(frozen=True)
class DbHrAssessment:
    """A set of DB-HR requirement checks.

    :ivar checks: The individual :class:`DbHrCheck` results.
    """

    checks: tuple[DbHrCheck, ...]

    @property
    def complies(self) -> bool:
        """Whether every check is met."""
        return all(c.complies for c in self.checks)

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the achieved values against their DB-HR limits."""
        from .._i18n import check_language
        from .._plot.building import plot_db_hr_assessment

        return plot_db_hr_assessment(
            self, ax=ax, language=check_language(language), **kwargs
        )


def db_hr_facade_requirement(
    ld: float,
    building_use: str,
    room_type: str,
    *,
    dominant_noise: str = "road",
    quiet_facade: bool = False,
) -> DbHrRequirement:
    """Required facade insulation ``D2m,nT,Atr`` (DB-HR Table 2.1).

    The requirement is keyed on the day noise index ``Ld`` of the site (Annex I
    of RD 1513/2005) and on the building use and room type. Two rules modify
    it: a facade not directly exposed to the dominant noise (an enclosed
    courtyard, a quiet surrounding) is assessed with ``Ld`` reduced by 10 dBA,
    and where aircraft noise dominates the table value is increased by 4 dBA.

    When no official ``Ld`` is available, DB-HR prescribes 60 dBA for
    residential acoustic areas.

    :param ld: Day noise index ``Ld`` of the site, in dBA.
    :param building_use: ``"residential"``, ``"hospital"``, ``"cultural"``,
        ``"sanitary"`` (non-hospital health premises), ``"educational"`` or
        ``"administrative"``.
    :param room_type: ``"bedrooms"``, ``"living"`` (living areas) or
        ``"classrooms"``, per the columns of Table 2.1.
    :param dominant_noise: ``"road"`` (default), ``"railway"`` or
        ``"aircraft"``. This selects the quantity the requirement is stated
        in as well as the aircraft increment: ``"railway"`` gives a
        ``D2m,nT,A`` requirement (Formula (A.5)) and the other two a
        ``D2m,nT,Atr`` one (Formula (A.6)), per clause 3.1.3.4 point 1.
    :param quiet_facade: Assess with ``Ld`` reduced by 10 dBA.
    :return: The :class:`DbHrRequirement`, on ``D2m,nT,Atr`` or, for a
        railway-dominant site, on ``D2m,nT,A``.
    :raises ValueError: For an unknown use/room combination, an unknown
        dominant noise, or a non-finite ``Ld``.
    """
    value = _finite(ld, "ld")
    use = _normalise(building_use, _USE_ALIASES, "building_use")
    room = _normalise(room_type, _ROOM_ALIASES, "room_type")
    if (use, room) not in _FACADE_ROWS:
        raise ValueError(
            "Unknown (building_use, room_type) combination "
            f"{(building_use, room_type)!r}; Table 2.1 holds "
            f"{sorted(_FACADE_ROWS)}."
        )
    noise = str(dominant_noise).strip().lower()
    if noise not in ("road", "railway", "aircraft"):
        raise ValueError(
            "'dominant_noise' must be 'road', 'railway' or 'aircraft'; got "
            f"{dominant_noise!r}."
        )
    if quiet_facade:
        value -= _QUIET_FACADE_REDUCTION

    series = _FACADE_ROWS[(use, room)]
    row = len(_FACADE_LD_BOUNDS)
    for i, bound in enumerate(_FACADE_LD_BOUNDS):
        if value <= bound:
            row = i
            break
    limit = float(series[row])
    note = ""
    if noise == "aircraft":
        limit += _AIRCRAFT_INCREMENT
        note = ", aircraft noise dominant (+4 dBA)"
    if quiet_facade:
        note += ", facade sheltered from the dominant noise (Ld - 10 dBA)"
    # Clause 3.1.3.4 point 1: a railway-dominant site is assessed in
    # D2m,nT,A (Formula (A.5)), road traffic and aircraft in D2m,nT,Atr
    # (Formula (A.6)). Table 2.1 supplies the same numeric limit either way.
    quantity = "D2m,nT,A" if noise == "railway" else "D2m,nT,Atr"
    return DbHrRequirement(
        quantity=quantity,
        limit=limit,
        direction="min",
        unit="dBA",
        decimals=0,
        reference="CTE DB-HR Table 2.1",
        description=(
            f"{use} building, {room.replace('_', ' ')}, "
            f"Ld = {value:g} dBA{note}"
        ),
    )


#: DB-HR 2.1.1 airborne requirements, keyed by (receiving room, source room).
_AIRBORNE_ROWS: dict[tuple[str, str], tuple[str, float, str, str]] = {
    ("protected", "same_unit"): (
        "RA", 33.0, "CTE DB-HR 2.1.1 a) i",
        "partition walls within a dwelling of a private residential building",
    ),
    ("protected", "other_unit"): (
        "DnT,A", 50.0, "CTE DB-HR 2.1.1 a) ii",
        "protected room against a room of another dwelling or use unit",
    ),
    ("protected", "installations"): (
        "DnT,A", 55.0, "CTE DB-HR 2.1.1 a) iii",
        "protected room against a services or activity room",
    ),
    ("protected", "activity"): (
        "DnT,A", 55.0, "CTE DB-HR 2.1.1 a) iii",
        "protected room against a services or activity room",
    ),
    ("habitable", "same_unit"): (
        "RA", 33.0, "CTE DB-HR 2.1.1 b) i",
        "partition walls within a dwelling of a private residential building",
    ),
    ("habitable", "other_unit"): (
        "DnT,A", 45.0, "CTE DB-HR 2.1.1 b) ii",
        "habitable room against a room of another dwelling or use unit",
    ),
    ("habitable", "installations"): (
        "DnT,A", 45.0, "CTE DB-HR 2.1.1 b) iii",
        "habitable room against a services or activity room",
    ),
    ("habitable", "activity"): (
        "DnT,A", 45.0, "CTE DB-HR 2.1.1 b) iii",
        "habitable room against a services or activity room",
    ),
}

#: RA of the shared door or window, and of the surrounding enclosure, when the
#: two rooms share one (DB-HR 2.1.1). Keyed by (receiving room, source room).
_SHARED_OPENING_ROWS: dict[tuple[str, str], tuple[float, float, str]] = {
    ("protected", "other_unit"): (30.0, 50.0, "CTE DB-HR 2.1.1 a) ii"),
    ("habitable", "other_unit"): (20.0, 50.0, "CTE DB-HR 2.1.1 b) ii"),
    # 2.1.1 b) ii states the shared-opening variant only for residential
    # (public or private) and hospital buildings.
    ("habitable", "installations"): (30.0, 50.0, "CTE DB-HR 2.1.1 b) iii"),
    ("habitable", "activity"): (30.0, 50.0, "CTE DB-HR 2.1.1 b) iii"),
}


def db_hr_airborne_requirement(
    receiving_room: str, source_room: str, *, shared_opening: bool = False
) -> tuple[DbHrRequirement, ...]:
    """Airborne insulation requirements between two rooms (DB-HR 2.1.1).

    A *protected* room is a habitable room of special protection (bedrooms,
    living areas, classrooms, reading rooms, offices); a *habitable* room is
    any other occupied room. The source can be another room of the same use
    unit (``"same_unit"``, which puts the requirement on the partition ``RA``),
    a room of a different use unit (``"other_unit"``), or a services or
    activity room (``"installations"`` / ``"activity"``).

    When the two rooms share a door or window the DB-HR replaces the
    level-difference requirement with a pair of ``RA`` requirements on the
    opening and on the surrounding enclosure; ``shared_opening=True`` returns
    those instead.

    :param receiving_room: ``"protected"`` or ``"habitable"``.
    :param source_room: ``"same_unit"``, ``"other_unit"``, ``"installations"``
        or ``"activity"``.
    :param shared_opening: The two rooms share a door or window.
    :return: A tuple of :class:`DbHrRequirement`, of length one in the general
        case and two when ``shared_opening`` applies.
    :raises ValueError: For an unknown combination, or for ``shared_opening``
        in a case DB-HR does not contemplate (a protected room against a
        services or activity room, and any ``"same_unit"`` case).
    """
    receiving = str(receiving_room).strip().lower()
    source = str(source_room).strip().lower()
    key = (receiving, source)
    if key not in _AIRBORNE_ROWS:
        raise ValueError(
            "Unknown (receiving_room, source_room) combination "
            f"{(receiving_room, source_room)!r}; DB-HR 2.1.1 covers "
            f"{sorted(_AIRBORNE_ROWS)}."
        )
    if not shared_opening:
        quantity, limit, reference, description = _AIRBORNE_ROWS[key]
        return (
            DbHrRequirement(
                quantity=quantity,
                limit=limit,
                direction="min",
                unit="dBA",
                decimals=0,
                reference=reference,
                description=description,
            ),
        )
    if key not in _SHARED_OPENING_ROWS:
        raise ValueError(
            "DB-HR 2.1.1 states no shared-opening variant for "
            f"{(receiving_room, source_room)!r}."
        )
    opening_limit, enclosure_limit, reference = _SHARED_OPENING_ROWS[key]
    opening_note = (
        "shared door or window"
        if key != ("habitable", "other_unit")
        else (
            "shared door or window, residential (public or private) or "
            "hospital building"
        )
    )
    return (
        DbHrRequirement(
            quantity="RA",
            limit=opening_limit,
            direction="min",
            unit="dBA",
            decimals=0,
            reference=reference,
            description=opening_note,
        ),
        DbHrRequirement(
            quantity="RA",
            limit=enclosure_limit,
            direction="min",
            unit="dBA",
            decimals=0,
            reference=reference,
            description="enclosure surrounding the shared door or window",
        ),
    )


#: DB-HR 2.1.1 c): the two alternative routes for a party wall between two
#: buildings, keyed by the quantity the designer chooses to demonstrate.
_PARTY_WALL_ROUTES: dict[str, tuple[float, str]] = {
    "D2m,nT,Atr": (40.0, "each of the two leaves of the party wall"),
    "DnT,A": (50.0, "the two leaves of the party wall taken together"),
}


def db_hr_party_wall_requirement(quantity: str = "D2m,nT,Atr") -> DbHrRequirement:
    """Airborne requirement on a party wall between two buildings (DB-HR 2.1.1 c).

    For a habitable or protected room against a *medianeria*, DB-HR offers two
    alternative routes: either each of the two enclosing leaves reaches
    ``D2m,nT,Atr`` of at least 40 dBA, or the two leaves taken together reach
    ``DnT,A`` of at least 50 dBA. They are alternatives, not cumulative
    requirements, so this function returns the one route asked for.

    :param quantity: ``"D2m,nT,Atr"`` (default, the per-leaf route) or
        ``"DnT,A"`` (the combined route).
    :return: The :class:`DbHrRequirement` of the chosen route.
    :raises ValueError: For an unknown quantity.
    """
    key = str(quantity).strip()
    if key not in _PARTY_WALL_ROUTES:
        raise ValueError(
            "'quantity' must be 'D2m,nT,Atr' (per leaf, 40 dBA) or 'DnT,A' "
            f"(both leaves together, 50 dBA); got {quantity!r}."
        )
    limit, description = _PARTY_WALL_ROUTES[key]
    return DbHrRequirement(
        quantity=key,
        limit=limit,
        direction="min",
        unit="dBA",
        decimals=0,
        reference="CTE DB-HR 2.1.1 c)",
        description=f"party wall between two buildings, {description}",
    )


#: DB-HR 2.1.2 impact requirements, keyed by (receiving room, source room).
_IMPACT_ROWS: dict[tuple[str, str], tuple[float, str]] = {
    ("protected", "other_unit"): (65.0, "CTE DB-HR 2.1.2 a) i"),
    ("protected", "installations"): (60.0, "CTE DB-HR 2.1.2 a) ii"),
    ("protected", "activity"): (60.0, "CTE DB-HR 2.1.2 a) ii"),
    ("habitable", "installations"): (60.0, "CTE DB-HR 2.1.2 b) i"),
    ("habitable", "activity"): (60.0, "CTE DB-HR 2.1.2 b) i"),
}


def db_hr_impact_requirement(receiving_room: str, source_room: str) -> DbHrRequirement:
    """Impact sound requirement ``L'nT,w`` (DB-HR 2.1.2).

    The standardized impact sound pressure level in a protected room adjacent
    to a room of another use unit must not exceed 65 dB; against a services or
    activity room the limit is 60 dB, and 60 dB also applies to a habitable
    room against those. The requirement does not apply to a protected room
    horizontally adjacent to a staircase.

    :param receiving_room: ``"protected"`` or ``"habitable"``.
    :param source_room: ``"other_unit"``, ``"installations"`` or
        ``"activity"``.
    :return: The :class:`DbHrRequirement` on ``L'nT,w``.
    :raises ValueError: For a combination DB-HR states no impact limit for.
    """
    key = (str(receiving_room).strip().lower(), str(source_room).strip().lower())
    if key not in _IMPACT_ROWS:
        raise ValueError(
            "DB-HR 2.1.2 states no impact requirement for "
            f"{(receiving_room, source_room)!r}; it covers "
            f"{sorted(_IMPACT_ROWS)}."
        )
    limit, reference = _IMPACT_ROWS[key]
    return DbHrRequirement(
        quantity="L'nT,w",
        limit=limit,
        direction="max",
        unit="dB",
        decimals=0,
        reference=reference,
        description=f"{key[0]} room against a {key[1].replace('_', ' ')} room",
    )


#: DB-HR 2.2 reverberation-time limits: (limit in s, clause, description).
_REVERBERATION_ROWS: dict[str, tuple[float, str, str]] = {
    "classroom": (
        0.7, "CTE DB-HR 2.2 point 1 a)",
        "empty classroom or conference hall under 350 m3",
    ),
    "classroom_seated": (
        0.5, "CTE DB-HR 2.2 point 1 b)",
        "empty classroom or conference hall under 350 m3, seating included",
    ),
    "restaurant": (
        0.9, "CTE DB-HR 2.2 point 1 c)",
        "empty restaurant or dining room",
    ),
}

#: Equivalent absorption area per unit volume required in common areas, in
#: m2 per m3 (DB-HR 2.2 point 2).
_COMMON_AREA_ABSORPTION = 0.2


def db_hr_reverberation_requirement(
    room_use: str, *, furnished: bool = False
) -> DbHrRequirement:
    """Reverberation or absorption requirement (DB-HR 2.2).

    Classrooms and conference halls under 350 m3 must not exceed 0,7 s empty
    (0,5 s with the seating installed); restaurants and dining rooms 0,9 s.
    Common areas of residential-public, educational and hospital buildings
    that share doors with protected rooms must provide an equivalent absorption
    area of at least 0,2 m2 per cubic metre of room volume.

    :param room_use: ``"classroom"`` (or ``"conference_hall"``),
        ``"restaurant"`` (or ``"dining_room"``) or ``"common_area"``.
    :param furnished: For a classroom or conference hall, include the seating
        (the 0,5 s limit).
    :return: The :class:`DbHrRequirement`.
    :raises ValueError: For an unknown room use.
    """
    key = str(room_use).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "conference_hall": "classroom",
        "lecture_hall": "classroom",
        "aula": "classroom",
        "dining_room": "restaurant",
        "comedor": "restaurant",
        "common_areas": "common_area",
        "zonas_comunes": "common_area",
    }
    key = aliases.get(key, key)
    if key == "common_area":
        return DbHrRequirement(
            quantity="A/V",
            limit=_COMMON_AREA_ABSORPTION,
            direction="min",
            unit="m2/m3",
            decimals=2,
            reference="CTE DB-HR 2.2 point 2",
            description="common area sharing doors with protected rooms",
        )
    if key == "classroom" and furnished:
        key = "classroom_seated"
    if key not in _REVERBERATION_ROWS:
        raise ValueError(
            "'room_use' must be 'classroom', 'restaurant' or 'common_area' "
            f"(or a documented alias); got {room_use!r}."
        )
    limit, reference, description = _REVERBERATION_ROWS[key]
    return DbHrRequirement(
        quantity="T",
        limit=limit,
        direction="max",
        unit="s",
        decimals=1,
        reference=reference,
        description=description,
    )


def check_db_hr_requirement(value: float, requirement: DbHrRequirement) -> DbHrCheck:
    """Check an achieved value against a DB-HR requirement.

    The achieved value is first rounded as DB-HR prescribes for the quantity
    (an integer for the dB quantities that define the requirements, one decimal
    for a reverberation time), and the rounded value is compared with the limit.

    :param value: The achieved value, in the requirement's unit.
    :param requirement: The :class:`DbHrRequirement` to check against.
    :return: A :class:`DbHrCheck`.
    :raises ValueError: If ``value`` is not finite.
    """
    achieved = _finite(value, "value")
    reported = _round_half_up(achieved, requirement.decimals)
    if requirement.direction == "min":
        margin = reported - requirement.limit
    else:
        margin = requirement.limit - reported
    return DbHrCheck(
        requirement=requirement,
        value=achieved,
        reported=reported,
        margin=margin,
        complies=bool(margin >= -1e-9),
    )


def assess_db_hr(
    items: Sequence[tuple[float, DbHrRequirement]],
) -> DbHrAssessment:
    """Check a set of achieved values against their DB-HR requirements.

    :param items: ``(value, requirement)`` pairs.
    :return: A :class:`DbHrAssessment`.
    :raises ValueError: If ``items`` is empty or a value is not finite.
    """
    pairs = list(items)
    if not pairs:
        raise ValueError("At least one (value, requirement) pair is required.")
    return DbHrAssessment(
        checks=tuple(check_db_hr_requirement(value, req) for value, req in pairs)
    )
