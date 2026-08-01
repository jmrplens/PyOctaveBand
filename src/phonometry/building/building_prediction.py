#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Building acoustic performance prediction (EN 12354-1/-2:2000).

This is the **prediction** counterpart of the measurement modules
(:mod:`phonometry.lab_insulation` for laboratory ``R``/``Ln`` and
:mod:`phonometry.insulation` for field ``R'``/``L'n``). EN 12354 estimates the
*in-situ* apparent performance of a building from the laboratory performance of
its elements, adding the flanking transmission that a field measurement would
capture but a laboratory measurement suppresses.

Both parts have a *detailed* per-band model and a *simplified* single-number
model. This module implements the **simplified single-number model** (Part 1
Clause 4.4, Part 2 Clause 4.3): it takes the weighted single-number ratings of
the elements (``Rw`` of walls/floors, ``ΔRw``/``ΔLw`` of linings/coverings and
the ``Kij`` vibration-reduction indices of the junctions) and predicts the
apparent weighted rating (``R'w`` airborne, ``L'n,w`` impact). The simplified
model is exact for ``RA`` and a good approximation for ``R'w`` (Part 1
Clause 4.4.1), with a reported standard deviation of about 2 dB (Clause 5).

**Airborne, Formula (26).** The apparent weighted sound reduction index is the
energetic sum of the direct path ``Dd`` and, for every flanking element, the
three flanking paths ``Ff``, ``Df`` and ``Fd``:

.. math::

   R'_w = -10 \lg\!\left[ 10^{-R_{Dd,w}/10} + \sum 10^{-R_{Ff,w}/10}
   + \sum 10^{-R_{Df,w}/10} + \sum 10^{-R_{Fd,w}/10} \right]

with the direct path :math:`R_{Dd,w} = R_{s,w} + \Delta R_{Dd,w}` (Formula 27)
and each flanking path (Formula 28a)

.. math::

   R_{ij,w} = \frac{R_{i,w} + R_{j,w}}{2} + \Delta R_{ij,w} + K_{ij}
   + 10 \lg\frac{S_s}{l_0 l_f}

where :math:`l_0 = 1` m is the reference coupling length.

**Junctions, Annex E.** The vibration reduction index ``Kij`` of rigid cross
(E.3) and T (E.4) junctions, junctions with flexible interlayers (E.5),
lightweight façade junctions (E.6), junctions of lightweight double-leaf walls
with homogeneous elements (E.7) or with other coupled double-leaf walls (E.8),
and corners / thickness changes (E.9) are empirical functions of the mass
ratio :math:`M = \lg(m'_{\perp,i} / m'_i)`. A minimum value ``Kij,min``
follows from the Kij,min relation of Clause 4.4.2 (printed as Eq. (23)
in the BS EN 12354-1:2000 edition).

**Impact, Formula (21).** :math:`L'_{n,w} = L_{n,w,eq} - \Delta L_w + K` with
the bare-floor equivalent level ``Ln,w,eq`` (Annex B
:math:`164 - 35 \lg(m'/m'_0)`), the covering improvement ``ΔLw`` (ISO 717-2)
and the flanking correction ``K`` from Table 1.

Clause citations refer to EN 12354-1:2000 (airborne) or EN 12354-2:2000 (impact).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log10
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .._report.metadata import ReportMetadata

#: Reference coupling length ``l0`` in Formula (28a), in metres (Clause 4.4.1).
_L0 = 1.0

#: Reference frequency ``fref`` of Annex E, in hertz (Formula E.1).
_FREF = 1000.0

#: Frequency (Hz) at which the single-number ``Kij`` is read for the simplified
#: model (Clause 4.4.2): the 500 Hz value.
_K_FREQUENCY = 500.0

#: Default interlayer characteristic frequency ``f1`` of Formula (E.5), in hertz,
#: valid for ``E1/t1 ≈ 100 MN/m³``.
_F1_DEFAULT = 125.0

#: Reference mass per unit area ``m'0`` in ``Ln,w,eq = 164 − 35 lg(m'/m'0)``.
_M0 = 1.0

#: Constants of the bare-floor equivalent impact level (Part 2, Annex B).
_LN_EQ_A = 164.0
_LN_EQ_B = 35.0

#: Validity envelope of the Annex B closed form ``164 − 35 lg(m')``, in kg/m²
#: (Part 2, Annex B: homogeneous floors of 100 kg/m² to 600 kg/m²).
_LN_EQ_MASS_MIN = 100.0
_LN_EQ_MASS_MAX = 600.0

#: Standardization constant of Formula (3) (Part 2):
#: ``L'nT = L'n − 10 lg(0,16·V/(A0·T0)) = L'n − 10 lg(0,032·V)`` with
#: ``A0 = 10 m²`` and ``T0 = 0,5 s``. The standard's own Annex E.3 rounds the
#: factor to ``10 lg(V/30)`` (1/0,032 = 31,25 ≈ 30); this module keeps the
#: exact Formula (3) form.
_IMPACT_STANDARDIZATION = 0.032

#: Standardization constant of Formula (5b) (Part 1):
#: ``DnT = R' + 10 lg(0,16·V/(T0·Ss)) = R' + 10 lg(0,32·V/Ss)`` with
#: ``T0 = 0,5 s``. The Annex H.3 worked example rounds the factor to
#: ``10 lg(V/(3·Ss))`` (1/0,32 = 3,125 ≈ 3); this module keeps the exact
#: Formula (5b) form (both round to the same integer in H.3).
_AIRBORNE_STANDARDIZATION = 0.32

#: Characteristic frequency ``fk`` of the frequency-dependent lightweight
#: double-leaf junctions (Annex E, Formulas (E.7)/(E.8)), in hertz.
_FK_DOUBLE_LEAF = 500.0

JunctionType = Literal[
    "rigid_cross", "rigid_t", "flexible_t", "lightweight_facade",
    "lightweight_double_homogeneous", "lightweight_double_coupled",
    "corner", "thickness_change",
]
PathKind = Literal["through", "corner", "double_leaf"]

#: Table 1 of EN 12354-2:2000: flanking correction ``K`` (dB). Row key is the
#: separating-floor mass, column key the mean flanking-element mass (kg/m²).
_TABLE1_SEP = (100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900)
_TABLE1_FLK = (100, 150, 200, 250, 300, 350, 400, 450, 500)
_TABLE1_K = (
    (1, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 1, 0, 0, 0, 0, 0, 0, 0),
    (2, 1, 1, 0, 0, 0, 0, 0, 0),
    (2, 1, 1, 1, 0, 0, 0, 0, 0),
    (3, 2, 1, 1, 1, 0, 0, 0, 0),
    (3, 2, 1, 1, 1, 1, 0, 0, 0),
    (4, 2, 2, 1, 1, 1, 1, 0, 0),
    (4, 3, 2, 2, 1, 1, 1, 1, 1),
    (4, 3, 2, 2, 1, 1, 1, 1, 1),
    (5, 4, 3, 2, 2, 1, 1, 1, 1),
    (5, 4, 3, 3, 2, 2, 1, 1, 1),
    (6, 4, 4, 3, 2, 2, 2, 1, 1),
    (6, 5, 4, 3, 3, 2, 2, 2, 2),
)


@dataclass(frozen=True)
class FlankingPath:
    """One flanking transmission path (Ff, Df or Fd) of the simplified model.

    :ivar label: Human-readable path name, e.g. ``"floor-Ff"``.
    :ivar kind: Path type, one of ``"Ff"``, ``"Df"``, ``"Fd"``.
    :ivar r_ij_w: Weighted flanking sound reduction index ``Rij,w`` of the path,
        in dB (EN 12354-1 Formula 28a).
    """

    label: str
    kind: Literal["Ff", "Df", "Fd"]
    r_ij_w: float


@dataclass(frozen=True)
class PathContribution:
    """A transmission path with its share of the total transmitted energy.

    :ivar label: Path name (``"Dd"`` for the direct path).
    :ivar kind: ``"Dd"``, ``"Ff"``, ``"Df"`` or ``"Fd"``.
    :ivar r_w: Weighted sound reduction index of the path, in dB.
    :ivar fraction: Fraction of the total transmitted sound energy carried by
        this path (0 to 1); the dominant path has the largest fraction.
    """

    label: str
    kind: str
    r_w: float
    fraction: float


@dataclass(frozen=True)
class AirbornePredictionResult:
    """Predicted apparent airborne insulation (EN 12354-1:2000, Formula 26).

    :ivar r_prime_w: Apparent weighted sound reduction index ``R'w``, in dB.
    :ivar r_direct_w: Direct-path weighted index ``RDd,w``, in dB (Formula 27).
    :ivar paths: Per-path contributions in input order (direct path first, then
        the flanking paths as supplied), each with its share of the energy.
    :ivar dominant: The path carrying the most energy (``PathContribution``).
    """

    r_prime_w: float
    r_direct_w: float
    paths: tuple[PathContribution, ...]
    dominant: PathContribution

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the per-path shares of the transmitted energy.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_airborne_prediction

        check_language(language)
        return plot_airborne_prediction(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a predicted airborne insulation report to a PDF (EN 12354-1).

        Writes a one-page **prediction** report for the predicted apparent
        sound reduction index ``R'`` between rooms estimated by the EN/ISO
        12354-1:2000 simplified single-number model (Clause 4.4): a
        standard-basis line that states the sheet is a prediction from element
        data and not a measurement, an optional metadata header block, a
        two-panel body with the transmission-path table (the direct path and
        each flanking path's weighted index ``Rij,w``) beside the per-path
        share-of-energy plot, the boxed predicted rating ``R'w``, the
        prediction statement (with the model's ~2 dB standard deviation) and,
        when a requirement is supplied, a PASS/FAIL verdict (the apparent index
        passes at or above the requirement), followed by a footer.

        The applicable :class:`~phonometry.ReportMetadata` fields describe the
        predicted situation: ``specimen`` (the separating element),
        ``area`` (the separating-element area ``Ss``), ``source_volume`` /
        ``receiving_volume`` (the room geometry), ``client``, ``manufacturer``,
        ``test_room``, ``laboratory`` (the calculator / laboratory),
        ``operator``, ``report_id`` and ``test_date``. A summary of the
        flanking construction and the model assumptions is recorded in
        ``notes`` (free text), and ``requirement`` supplies the target ``R'w``.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a lightweight fiche (body, rating, statement, disclaimer).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the path table also shows each path's
            share of the transmitted sound energy.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown or ``language`` is not
            supported.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is "
                "supported."
            )
        from .._report.iso12354 import render_iso12354_airborne_report

        return render_iso12354_airborne_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


@dataclass(frozen=True)
class ImpactPredictionResult:
    """Predicted apparent impact insulation (EN 12354-2:2000, Formula 21).

    :ivar l_prime_n_w: Apparent weighted normalized impact sound pressure level
        ``L'n,w``, in dB.
    :ivar ln_w_eq: Bare-floor equivalent weighted level ``Ln,w,eq``, in dB.
    :ivar delta_l_w: Weighted covering improvement ``ΔLw``, in dB.
    :ivar k_correction: Flanking correction ``K``, in dB (Table 1).
    """

    l_prime_n_w: float
    ln_w_eq: float
    delta_l_w: float
    k_correction: float

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the Formula 21 terms and the resulting ``L'n,w``.

        Requires matplotlib (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes`.
        """
        from .._i18n import check_language
        from .._plot.building import plot_impact_prediction

        check_language(language)
        return plot_impact_prediction(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render a predicted impact insulation report to a PDF (EN 12354-2).

        Writes a one-page **prediction** report for the predicted apparent
        normalized impact sound pressure level ``L'n`` estimated by the EN/ISO
        12354-2:2000 simplified single-number model (Clause 4.3): a
        standard-basis line that states the sheet is a prediction from element
        data and not a measurement, an optional metadata header block, a
        two-panel body with the Formula (21) term table (the bare-floor
        equivalent level ``Ln,w,eq``, the covering improvement ``ΔLw`` and the
        flanking correction ``K``) beside the term plot, the boxed predicted
        rating ``L'n,w``, the prediction statement (with the model's ~2 dB
        standard deviation) and, when a requirement is supplied, a PASS/FAIL
        verdict (the apparent level passes at or below the requirement, a lower
        impact level being better), followed by a footer.

        The applicable :class:`~phonometry.ReportMetadata` fields describe the
        predicted situation: ``specimen`` (the separating floor), ``area`` (the
        floor area), ``mass_per_area`` (the bare floor's mass per unit area),
        ``receiving_volume`` (the receiving-room geometry), ``client``,
        ``manufacturer``, ``test_room``, ``laboratory`` (the calculator /
        laboratory), ``operator``, ``report_id`` and ``test_date``. A summary
        of the flanking construction and the model assumptions is recorded in
        ``notes`` (free text), and ``requirement`` supplies the target
        ``L'n,w``.

        :param path: Destination path of the PDF file.
        :param metadata: Optional :class:`~phonometry.ReportMetadata`; ``None``
            produces a lightweight fiche (body, rating, statement, disclaimer).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform ``.report()`` signature; the
            impact fiche has a single body layout, so it has no effect.
        :param language: Fiche language: ``"en"`` (default) or ``"es"``.
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is unknown or ``language`` is not
            supported.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is "
                "supported."
            )
        from .._report.iso12354 import render_iso12354_impact_report

        return render_iso12354_impact_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


def _check_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, raising if it is not finite."""
    v = float(value)
    if not isfinite(v):
        raise ValueError(f"'{name}' must be a finite number.")
    return v


def junction_vibration_reduction(
    junction_type: JunctionType,
    path: PathKind,
    mass_ratio: float,
    *,
    frequency: float = _K_FREQUENCY,
    f1: float = _F1_DEFAULT,
) -> float:
    r"""Vibration reduction index ``Kij`` of a junction (EN 12354-1 Annex E).

    Empirical ``Kij`` for common junctions as a function of the mass ratio
    :math:`M = \lg(m'_{\perp,i} / m'_i)` (Formula E.2), where ``mass_ratio``
    is :math:`m'_{\perp,i} / m'_i`, the mass per unit area of the
    perpendicular element over that of the element carrying the path.
    ``path`` selects the *through* branch (in-line elements,
    ``K13``), the *corner* branch (:math:`K_{12} = K_{23}`) or, for double-leaf
    separating walls, the *double-leaf* branch (``K24``, the path between the
    two flanking legs across the double leaf).

    Supported ``junction_type`` values and their formulas:

    - ``"rigid_cross"`` (E.3): through :math:`8.7 + 17.1 M + 5.7 M^2`;
      corner :math:`8.7 + 5.7 M^2`.
    - ``"rigid_t"`` (E.4): through :math:`5.7 + 14.1 M + 5.7 M^2`;
      corner :math:`5.7 + 5.7 M^2`.
    - ``"flexible_t"`` (E.5, wall junction with flexible interlayers): through
      :math:`5.7 + 14.1 M + 5.7 M^2 + 2 \Delta_1`; corner
      :math:`5.7 + 5.7 M^2 + \Delta_1` with
      :math:`\Delta_1 = 10 \lg(f/f_1)` for :math:`f > f_1` (else 0) and
      :math:`f_1 = 125` Hz for the typical interlayer
      :math:`E_1/t_1 \approx 100` MN/m³; double-leaf
      :math:`K_{24} = 3.7 + 14.1 M + 5.7 M^2` clamped to
      :math:`-4 \le K_{24} \le 0` dB.
      (The 2000 print states the clamp as "0 ≤ K24 ≤ −4 dB", an obvious
      misprint of the bounds' order.)
    - ``"lightweight_facade"`` (E.6): through :math:`\max(5 + 10 M, 5)`;
      corner :math:`10 + 10 |M|`.
    - ``"lightweight_double_homogeneous"`` (E.7, lightweight double-leaf wall
      joined to homogeneous elements): through
      :math:`\max(10 + 20 M - 3.3 \lg(f/f_k), 10)`; corner
      :math:`10 + 10 |M| + 3.3 \lg(f/f_k)`; double-leaf
      :math:`K_{24} = 3.0 + 14.1 M + 5.7 M^2` with :math:`f_k = 500` Hz.
      The K24 path is
      carried by the homogeneous element crossing the double leaf, so its
      per-path ``mass_ratio`` :math:`= m'_{\perp,i}/m'_i` is
      leaf-over-homogeneous and the
      validity condition (homogeneous over three times heavier than a leaf)
      reads ``mass_ratio`` :math:`< 1/3`. (The 2000 print states this line as
      :math:`3.0 - 14.1 M + 5.7 M^2` in the *figure-axis* variable
      :math:`M = \lg(m_2/m_1)` of Figure E.9, contradicting the annex's own
      per-path definition of M; both forms are numerically identical, and
      ISO 12354-1:2017 E.3.5 prints the per-path form implemented here. See
      ``docs/ERRATA.md``.)
    - ``"lightweight_double_coupled"`` (E.8, junction of lightweight coupled
      double-leaf walls): through
      :math:`\max(10 + 20 M - 3.3 \lg(f/f_k), 10)`;
      corner :math:`10 + 10 |M| - 3.3 \lg(f/f_k)`; with
      :math:`f_k = 500` Hz.
    - ``"corner"`` (E.9 A, two elements meeting at a corner): corner
      :math:`K_{12} = \max(15 |M| - 3, -2)` (:math:`= K_{21}`); the only
      path.
    - ``"thickness_change"`` (E.9 B, thickness change in an element): through
      :math:`K_{12} = 5 M^2 - 5` (:math:`= K_{21}`); the only path.

    :param junction_type: Junction geometry (see above).
    :param path: ``"through"`` (K13; also the single K12 path of a thickness
        change), ``"corner"`` (K12 = K23; also the single path of a corner) or
        ``"double_leaf"`` (K24).
    :param mass_ratio: :math:`m'_{\perp,i} / m'_i`, the mass per unit area of
        the perpendicular element over that of the element carrying the path
        (must be positive). The same per-path convention applies to every
        branch, including both ``double_leaf`` (K24) branches.
    :param frequency: Frequency at which ``Kij`` is evaluated, in Hz; only the
        ``"flexible_t"`` (through/corner) and the E.7/E.8 lightweight
        double-leaf junctions are frequency dependent. Defaults to 500 Hz, the
        value used by the simplified model (Clause 4.4.2), at which the
        E.7/E.8 :math:`\lg(f/f_k)` terms vanish.
    :param f1: Interlayer characteristic frequency for ``"flexible_t"``, in Hz.
    :return: ``Kij``, in dB.
    :raises ValueError: If ``mass_ratio`` is not positive, ``frequency``/``f1``
        are not positive, an unknown ``junction_type``/``path`` is given, the
        requested path does not exist for the junction, or the E.7 double-leaf
        branch is requested outside its :math:`m_2/m_1 > 3` validity.
    """
    ratio = _check_finite(mass_ratio, "mass_ratio")
    if ratio <= 0.0:
        raise ValueError("'mass_ratio' must be positive.")
    if _check_finite(frequency, "frequency") <= 0.0:
        raise ValueError("'frequency' must be positive.")
    if _check_finite(f1, "f1") <= 0.0:
        raise ValueError("'f1' must be positive.")
    if path not in ("through", "corner", "double_leaf"):
        raise ValueError("'path' must be 'through', 'corner' or 'double_leaf'.")

    m = log10(ratio)
    delta1 = 10.0 * log10(frequency / f1) if frequency > f1 else 0.0
    # Frequency term of the E.7/E.8 lightweight double-leaf junctions.
    lg_f_fk = log10(frequency / _FK_DOUBLE_LEAF)

    if junction_type == "rigid_cross":
        if path == "through":
            return 8.7 + 17.1 * m + 5.7 * m * m
        if path == "corner":
            return 8.7 + 5.7 * m * m
    if junction_type == "rigid_t":
        if path == "through":
            return 5.7 + 14.1 * m + 5.7 * m * m
        if path == "corner":
            return 5.7 + 5.7 * m * m
    if junction_type == "flexible_t":
        if path == "through":
            return 5.7 + 14.1 * m + 5.7 * m * m + 2.0 * delta1
        if path == "corner":
            return 5.7 + 5.7 * m * m + delta1
        # (E.5) K24, clamped to −4..0 dB (the print's "0 ≤ K24 ≤ −4 dB" read
        # with the bounds in ascending order).
        return min(max(3.7 + 14.1 * m + 5.7 * m * m, -4.0), 0.0)
    if junction_type == "lightweight_facade":
        if path == "through":
            return max(5.0 + 10.0 * m, 5.0)
        if path == "corner":
            return 10.0 + 10.0 * abs(m)
    if junction_type == "lightweight_double_homogeneous":
        if path == "through":
            return max(10.0 + 20.0 * m - 3.3 * lg_f_fk, 10.0)
        if path == "corner":
            return 10.0 + 10.0 * abs(m) + 3.3 * lg_f_fk
        # (E.7) K24 is given only where the homogeneous elements carrying
        # the 2-4 path are more than three times heavier than a leaf
        # (m2/m1 > 3), i.e. for a per-path mass_ratio = m'⊥,i/m'i < 1/3.
        if ratio >= 1.0 / 3.0:
            raise ValueError(
                "The E.7 double-leaf branch (K24) is given only for "
                "mass_ratio = m'⊥,i/m'i < 1/3 (the homogeneous element "
                "carries the 2-4 path, so the ratio is leaf mass over "
                "homogeneous-element mass and the printed condition "
                "m2/m1 > 3 reads mass_ratio < 1/3)."
            )
        # Per-path form (ISO 12354-1:2017 E.3.5); the 2000 print recasts the
        # same relation as 3,0 - 14,1 M + 5,7 M^2 in the Figure E.9 x-axis
        # variable lg(m2/m1), against its own Annex E definition of M (see
        # docs/ERRATA.md).
        return 3.0 + 14.1 * m + 5.7 * m * m
    if junction_type == "lightweight_double_coupled":
        if path == "through":
            return max(10.0 + 20.0 * m - 3.3 * lg_f_fk, 10.0)
        if path == "corner":
            return 10.0 + 10.0 * abs(m) - 3.3 * lg_f_fk
    if junction_type == "corner":
        if path == "corner":
            return max(15.0 * abs(m) - 3.0, -2.0)
        raise ValueError(
            "A 'corner' junction has the single path K12 = K21; use "
            "path='corner'."
        )
    if junction_type == "thickness_change":
        if path == "through":
            return 5.0 * m * m - 5.0
        raise ValueError(
            "A 'thickness_change' junction has the single in-line path "
            "K12 = K21; use path='through'."
        )
    if junction_type in (
        "rigid_cross", "rigid_t", "lightweight_facade",
        "lightweight_double_coupled",
    ):
        raise ValueError(
            f"Junction type {junction_type!r} has no 'double_leaf' (K24) "
            "branch in EN 12354-1 Annex E."
        )
    raise ValueError(
        "'junction_type' must be one of 'rigid_cross', 'rigid_t', "
        "'flexible_t', 'lightweight_facade', "
        "'lightweight_double_homogeneous', 'lightweight_double_coupled', "
        "'corner', 'thickness_change'."
    )


def junction_min_vibration_reduction(
    coupling_length: float, s_i: float, s_j: float
) -> float:
    r"""Minimum vibration reduction index ``Kij,min`` (EN 12354-1 Clause 4.4.2).

    Printed as Formula (29) in the EN 12354-1:2000 edition.

    :math:`K_{ij,\mathrm{min}} = 10 \lg[l_f \, l_0 \, (1/S_i + 1/S_j)]` with
    the reference coupling
    length :math:`l_0 = 1` m. When the tabulated ``Kij`` is below this value,
    the minimum is used (Clause 4.4.2).

    :param coupling_length: Common coupling length ``lf`` of the junction, in m.
    :param s_i: Area of element ``i``, in m².
    :param s_j: Area of element ``j``, in m².
    :return: ``Kij,min``, in dB.
    :raises ValueError: If any argument is not positive.
    """
    lf = _check_finite(coupling_length, "coupling_length")
    si = _check_finite(s_i, "s_i")
    sj = _check_finite(s_j, "s_j")
    if lf <= 0.0 or si <= 0.0 or sj <= 0.0:
        raise ValueError("'coupling_length', 's_i' and 's_j' must be positive.")
    return 10.0 * log10(lf * _L0 * (1.0 / si + 1.0 / sj))


def combine_linings(delta_a: float, delta_b: float) -> float:
    r"""Combine two lining improvements (EN 12354-1 Formulas 30/31).

    For two linings the total improvement is the larger value plus half the
    smaller: :math:`\Delta R = \max(a, b) + \min(a, b)/2`. For a single lining
    pass the other as ``0``.

    .. note::
        ISO 12354-1:2017 (Formulas (22)/(23)) adds a special case the 2000
        edition implemented here does not have: when *both* linings are
        negative (each lining worsens the insulation), half is taken of the
        *higher* value instead,
        :math:`\Delta R = \min(a, b) + \max(a, b)/2`, so two
        degrading linings degrade further (e.g. −2 and −4 dB combine to
        −5 dB under the 2017 rule, −4 dB under the 2000 rule used here).

    :param delta_a: Improvement of the first lining, in dB.
    :param delta_b: Improvement of the second lining, in dB.
    :return: Combined ``ΔR``/``ΔRij``, in dB.
    """
    a = _check_finite(delta_a, "delta_a")
    b = _check_finite(delta_b, "delta_b")
    return max(a, b) + min(a, b) / 2.0


def _coupling_term(separating_area: float, coupling_length: float) -> float:
    r"""The :math:`10 \lg(S_s/(l_0 l_f))` term of Formula (28a)."""
    ss = _check_finite(separating_area, "separating_area")
    lf = _check_finite(coupling_length, "coupling_length")
    if ss <= 0.0 or lf <= 0.0:
        raise ValueError(
            "'separating_area' and 'coupling_length' must be positive."
        )
    return 10.0 * log10(ss / (_L0 * lf))


def flanking_path(
    *,
    label: str,
    kind: Literal["Ff", "Df", "Fd"],
    r_source: float,
    r_receive: float,
    k_ij: float,
    separating_area: float,
    coupling_length: float,
    delta_r: float = 0.0,
    kij_min: float | None = None,
) -> FlankingPath:
    r"""Build one flanking path ``Rij,w`` (EN 12354-1 Formula 28a).

    .. math::

       R_{ij,w} = \frac{R_{i,w} + R_{j,w}}{2} + \Delta R_{ij,w} + K_{ij}
       + 10 \lg\frac{S_s}{l_0 l_f}

    with ``r_source`` and ``r_receive`` as :math:`R_{i,w}` / :math:`R_{j,w}`,
    ``delta_r`` as :math:`\Delta R_{ij,w}` and ``k_ij`` as :math:`K_{ij}`.
    The two element indices depend on the path: for ``Ff`` both are the flanking
    element (``RF,w``, ``Rf,w``); for ``Fd`` they are the flanking (source) and
    separating (receive) elements; for ``Df`` the separating (source) and
    flanking (receive) elements.

    When ``kij_min`` is given, ``k_ij`` is clamped up to it
    (``max(k_ij, kij_min)``) before the path is formed, enforcing the floor
    :math:`K_{ij} \ge K_{ij,\mathrm{min}}` of Clause 4.4.2 (compute
    ``kij_min`` with
    :func:`junction_min_vibration_reduction`). Left as ``None`` the raw ``k_ij``
    is used unchanged.

    :param label: Human-readable path name.
    :param kind: ``"Ff"``, ``"Df"`` or ``"Fd"``.
    :param r_source: Weighted sound reduction index of the source-side element.
    :param r_receive: Weighted sound reduction index of the receive-side element.
    :param k_ij: Vibration reduction index of this path, in dB.
    :param separating_area: Area ``Ss`` of the separating element, in m².
    :param coupling_length: Junction coupling length ``lf``, in m.
    :param delta_r: Combined lining improvement ``ΔRij,w`` for this path, in dB.
    :param kij_min: Optional ``Kij,min`` floor (Clause 4.4.2); ``k_ij`` is
        raised to it when it lies below. ``None`` disables the clamp.
    :return: The :class:`FlankingPath`.
    :raises ValueError: If ``kind`` is unknown, areas/lengths are not positive,
        or any value is non-finite.
    """
    if kind not in ("Ff", "Df", "Fd"):
        raise ValueError("'kind' must be 'Ff', 'Df' or 'Fd'.")
    rs = _check_finite(r_source, "r_source")
    rr = _check_finite(r_receive, "r_receive")
    kij = _check_finite(k_ij, "k_ij")
    if kij_min is not None:
        kij = max(kij, _check_finite(kij_min, "kij_min"))
    dr = _check_finite(delta_r, "delta_r")
    r_ij = (rs + rr) / 2.0 + dr + kij + _coupling_term(
        separating_area, coupling_length
    )
    return FlankingPath(label=label, kind=kind, r_ij_w=r_ij)


def flanking_element(
    *,
    label: str,
    r_flanking: float,
    r_separating: float,
    k_ff: float,
    k_fd: float,
    k_df: float,
    separating_area: float,
    coupling_length: float,
    delta_r_ff: float = 0.0,
    delta_r_fd: float = 0.0,
    delta_r_df: float = 0.0,
    flanking_area: float | None = None,
) -> tuple[FlankingPath, FlankingPath, FlankingPath]:
    r"""Build the three flanking paths (Ff, Df, Fd) of one flanking element.

    Convenience wrapper over :func:`flanking_path` for the common case where a
    flanking element is essentially the same on the source and receiving side
    (Clause 4.4.1). Returns the ``Ff``, ``Df`` and ``Fd`` paths that this element
    contributes across its junction with the separating element.

    **Kij,min (Clause 4.4.2).** When ``flanking_area`` is given, the mandatory
    floor :math:`K_{ij} \ge K_{ij,\mathrm{min}}` is applied automatically per
    path: ``KFf`` is clamped to :math:`10 \lg[l_f l_0 (2/S_F)]` (both
    junction elements are
    the flanking element) and ``KFd``/``KDf`` to
    :math:`10 \lg[l_f l_0 (1/S_F + 1/S_s)]` (flanking and separating
    element), via
    :func:`junction_min_vibration_reduction`. Without ``flanking_area`` the
    per-path floors cannot be formed from the available geometry, so the raw
    ``k_ff``/``k_fd``/``k_df`` are used unchanged; compute the floors
    yourself (or call :func:`flanking_path` with ``kij_min``) in that case to
    stay within Clause 4.4.2.

    :param label: Base name; paths are labelled ``"<label>-Ff"`` etc.
    :param r_flanking: Weighted sound reduction index of the flanking element.
    :param r_separating: Weighted sound reduction index of the separating element.
    :param k_ff: ``KFf`` vibration reduction index, in dB.
    :param k_fd: ``KFd`` vibration reduction index, in dB.
    :param k_df: ``KDf`` vibration reduction index, in dB.
    :param separating_area: Separating-element area ``Ss``, in m².
    :param coupling_length: Junction coupling length ``lf``, in m.
    :param delta_r_ff: Combined lining improvement for the Ff path, in dB.
    :param delta_r_fd: Combined lining improvement for the Fd path, in dB.
    :param delta_r_df: Combined lining improvement for the Df path, in dB.
    :param flanking_area: Flanking-element area :math:`S_F = S_f`, in m².
        Enables
        the automatic ``Kij,min`` clamp (Clause 4.4.2); ``None`` skips it.
    :return: The ``(Ff, Df, Fd)`` :class:`FlankingPath` triple.
    :raises ValueError: If a geometry value is not positive or an input is
        non-finite.
    """
    kij_min_ff: float | None = None
    kij_min_cross: float | None = None
    if flanking_area is not None:
        kij_min_ff = junction_min_vibration_reduction(
            coupling_length, flanking_area, flanking_area
        )
        kij_min_cross = junction_min_vibration_reduction(
            coupling_length, flanking_area, separating_area
        )
    ff = flanking_path(
        label=f"{label}-Ff", kind="Ff", r_source=r_flanking,
        r_receive=r_flanking, k_ij=k_ff, separating_area=separating_area,
        coupling_length=coupling_length, delta_r=delta_r_ff,
        kij_min=kij_min_ff,
    )
    df = flanking_path(
        label=f"{label}-Df", kind="Df", r_source=r_separating,
        r_receive=r_flanking, k_ij=k_df, separating_area=separating_area,
        coupling_length=coupling_length, delta_r=delta_r_df,
        kij_min=kij_min_cross,
    )
    fd = flanking_path(
        label=f"{label}-Fd", kind="Fd", r_source=r_flanking,
        r_receive=r_separating, k_ij=k_fd, separating_area=separating_area,
        coupling_length=coupling_length, delta_r=delta_r_fd,
        kij_min=kij_min_cross,
    )
    return ff, df, fd


def predicted_airborne_insulation(
    *,
    r_direct: float,
    flanking_paths: Sequence[FlankingPath] = (),
    delta_r_direct: float = 0.0,
) -> AirbornePredictionResult:
    r"""Predict the apparent airborne insulation ``R'w`` (EN 12354-1 Formula 26).

    Energetically combines the direct path
    :math:`R_{Dd,w} = R_{s,w} + \Delta R_{Dd,w}` (Formula 27, from
    ``r_direct`` and ``delta_r_direct``) with the supplied flanking paths:

    .. math::

       R'_w = -10 \lg\!\left[ 10^{-R_{Dd,w}/10}
       + \sum 10^{-R_{ij,w}/10} \right]

    With no flanking paths the result equals the direct path ``RDd,w``; each
    added path strictly lowers ``R'w``. The result exposes every path's share of
    the transmitted energy so the dominant path is visible.

    :param r_direct: Weighted sound reduction index of the separating element
        ``Rs,w``, in dB.
    :param flanking_paths: Flanking paths (see :func:`flanking_element`). May be
        empty for the direct-only case.
    :param delta_r_direct: Combined lining improvement ``ΔRDd,w`` on the
        separating element, in dB.
    :return: The :class:`AirbornePredictionResult`.
    :raises ValueError: If any input is non-finite.
    """
    r_dd = _check_finite(r_direct, "r_direct") + _check_finite(
        delta_r_direct, "delta_r_direct"
    )
    tau_direct = 10.0 ** (-r_dd / 10.0)
    tau_paths = [10.0 ** (-p.r_ij_w / 10.0) for p in flanking_paths]
    tau_total = tau_direct + sum(tau_paths)
    r_prime_w = -10.0 * log10(tau_total)

    contributions = [
        PathContribution(
            label="Dd", kind="Dd", r_w=r_dd, fraction=tau_direct / tau_total
        )
    ]
    contributions += [
        PathContribution(
            label=p.label, kind=p.kind, r_w=p.r_ij_w,
            fraction=tau / tau_total,
        )
        for p, tau in zip(flanking_paths, tau_paths)
    ]
    dominant = max(contributions, key=lambda c: c.fraction)
    return AirbornePredictionResult(
        r_prime_w=r_prime_w,
        r_direct_w=r_dd,
        paths=tuple(contributions),
        dominant=dominant,
    )


def equivalent_impact_level(mass_per_area: float) -> float:
    r"""Bare-floor equivalent weighted impact level ``Ln,w,eq`` (Part 2, Annex B).

    :math:`L_{n,w,eq} = 164 - 35 \lg(m'/m'_0)` with :math:`m'_0 = 1` kg/m²,
    the closed form
    used in the Annex E worked example for a homogeneous concrete floor. The
    Annex B relation is stated for homogeneous floors of 100 kg/m² to
    600 kg/m²; outside that envelope the value is an extrapolation and a
    :class:`UserWarning` is emitted.

    :param mass_per_area: Mass per unit area ``m'`` of the bare floor, in kg/m²
        (must be positive; the Annex B relation covers 100-600 kg/m²).
    :return: ``Ln,w,eq``, in dB.
    :raises ValueError: If ``mass_per_area`` is not positive.
    """
    m = _check_finite(mass_per_area, "mass_per_area")
    if m <= 0.0:
        raise ValueError("'mass_per_area' must be positive.")
    if not (_LN_EQ_MASS_MIN <= m <= _LN_EQ_MASS_MAX):
        warnings.warn(
            f"mass_per_area = {m:g} kg/m² lies outside the 100-600 kg/m² "
            "envelope of the EN 12354-2 Annex B relation "
            "Ln,w,eq = 164 - 35 lg(m'); the result is an extrapolation.",
            UserWarning,
            stacklevel=2,
        )
    return _LN_EQ_A - _LN_EQ_B * log10(m / _M0)


def impact_flanking_correction(
    separating_mass: float, flanking_mass: float
) -> int:
    """Flanking correction ``K`` from Table 1 (EN 12354-2:2000).

    Looks up ``K`` (dB) for the separating-floor mass and the mean mass of the
    homogeneous flanking elements, selecting the nearest tabulated row/column
    (the table is discrete; masses outside 100–900 / 100–500 kg/m² clamp to the
    nearest edge).

    :param separating_mass: Mass per unit area of the separating floor, in kg/m².
    :param flanking_mass: Mean mass per unit area of the homogeneous flanking
        elements not covered by additional layers, in kg/m².
    :return: The correction ``K``, in dB (a non-negative integer).
    :raises ValueError: If a mass is not positive.
    """
    sm = _check_finite(separating_mass, "separating_mass")
    fm = _check_finite(flanking_mass, "flanking_mass")
    if sm <= 0.0 or fm <= 0.0:
        raise ValueError("'separating_mass' and 'flanking_mass' must be positive.")
    # Nearest-neighbour selection on the discrete Table 1 grid. Both mass axes
    # are ascending, and ``min`` returns the first index reaching the smallest
    # distance, so a mass exactly halfway between two tabulated values (e.g.
    # 125 kg/m² between 100 and 150) ties to the *lower* tabulated mass.
    row = min(range(len(_TABLE1_SEP)), key=lambda i: abs(_TABLE1_SEP[i] - sm))
    col = min(range(len(_TABLE1_FLK)), key=lambda j: abs(_TABLE1_FLK[j] - fm))
    return _TABLE1_K[row][col]


def predicted_impact_insulation(
    *,
    ln_w_eq: float,
    delta_l_w: float = 0.0,
    k_correction: float = 0.0,
) -> ImpactPredictionResult:
    r"""Predict the apparent impact insulation ``L'n,w`` (EN 12354-2 Formula 21).

    :math:`L'_{n,w} = L_{n,w,eq} - \Delta L_w + K`. The bare-floor equivalent
    level may come from
    :func:`equivalent_impact_level` and the flanking correction from
    :func:`impact_flanking_correction`.

    :param ln_w_eq: Bare-floor equivalent weighted level ``Ln,w,eq``, in dB.
    :param delta_l_w: Weighted covering improvement ``ΔLw`` (ISO 717-2), in dB.
    :param k_correction: Flanking correction ``K`` (Table 1), in dB.
    :return: The :class:`ImpactPredictionResult`.
    :raises ValueError: If any input is non-finite.
    """
    ln_eq = _check_finite(ln_w_eq, "ln_w_eq")
    dl = _check_finite(delta_l_w, "delta_l_w")
    k = _check_finite(k_correction, "k_correction")
    return ImpactPredictionResult(
        l_prime_n_w=ln_eq - dl + k,
        ln_w_eq=ln_eq,
        delta_l_w=dl,
        k_correction=k,
    )


def standardized_impact_level(l_prime_n_w: float, volume: float) -> float:
    r"""Standardized apparent impact level ``L'nT,w`` (EN 12354-2 Formula 3).

    .. math::

       L'_{nT,w} = L'_{n,w} - 10 \lg\frac{0.16\,V}{A_0 T_0}
       = L'_{n,w} - 10 \lg(0.032\,V)

    with :math:`A_0 = 10` m² and :math:`T_0 = 0.5` s, the exact Formula (3)
    form. The standard's own Annex E.3 worked example rounds the factor to
    :math:`10 \lg(V/30)` (:math:`1/0.032 = 31.25 \approx 30`), 0.18 dB below
    the exact form; both round to the same integer rating in E.3.

    :param l_prime_n_w: Apparent weighted normalized impact level ``L'n,w``, dB.
    :param volume: Receiving-room volume ``V``, in m³ (must be positive).
    :return: ``L'nT,w``, in dB.
    :raises ValueError: If ``volume`` is not positive.
    """
    lnw = _check_finite(l_prime_n_w, "l_prime_n_w")
    v = _check_finite(volume, "volume")
    if v <= 0.0:
        raise ValueError("'volume' must be positive.")
    return lnw - 10.0 * log10(_IMPACT_STANDARDIZATION * v)


def standardized_level_difference(
    r_prime_w: float, volume: float, separating_area: float
) -> float:
    r"""Standardized level difference ``DnT,w`` from ``R'w`` (EN 12354-1 Formula 5b).

    .. math::

       D_{nT} = R' + 10 \lg\frac{0.16\,V}{T_0 S_s}
       = R' + 10 \lg\frac{0.32\,V}{S_s}

    with :math:`T_0 = 0.5` s, the exact Formula (5b) form, applied to the
    weighted single numbers of the simplified model (Clause 4.4). The Annex
    H.3 worked example rounds the factor to :math:`10 \lg(V/(3 S_s))`
    (:math:`1/0.32 = 3.125 \approx 3`), printing :math:`52.2 + 1.6 = 53.8` dB
    where the exact form gives 53.6 dB;
    both round to the same :math:`D_{nT,w} = 54` dB.

    :param r_prime_w: Apparent weighted sound reduction index ``R'w``, in dB
        (see :func:`predicted_airborne_insulation`).
    :param volume: Receiving-room volume ``V``, in m³ (must be positive).
    :param separating_area: Separating-element area ``Ss``, in m² (must be
        positive).
    :return: ``DnT,w``, in dB.
    :raises ValueError: If ``volume`` or ``separating_area`` is not positive.
    """
    rw = _check_finite(r_prime_w, "r_prime_w")
    v = _check_finite(volume, "volume")
    ss = _check_finite(separating_area, "separating_area")
    if v <= 0.0:
        raise ValueError("'volume' must be positive.")
    if ss <= 0.0:
        raise ValueError("'separating_area' must be positive.")
    return rw + 10.0 * log10(_AIRBORNE_STANDARDIZATION * v / ss)
