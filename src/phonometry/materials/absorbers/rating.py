#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Single-number rating of sound absorption (ISO 11654:1997).

From one-third-octave sound absorption coefficients ``alpha_s`` measured in
a reverberation room (ISO 354) this module forms the practical sound
absorption coefficient ``alpha_p`` per octave band (Clause 4.1), the
weighted sound absorption coefficient ``alpha_w`` by the reference-curve
shifting method (Clause 4.2), the shape indicators ``L``, ``M``, ``H``
(Clause 4.3) and the sound absorption class ``A`` to ``E`` of the
informative Table B.1 (Annex B).

**Practical absorption coefficient (Clause 4.1).** For each octave band the
practical coefficient is the arithmetic mean of the three one-third-octave
coefficients it contains:

.. math::

   \alpha_{p,i} = \frac{\alpha_{i1} + \alpha_{i2} + \alpha_{i3}}{3}

The mean is evaluated to the second decimal and then rounded in steps of
0.05 (the NOTE of Clause 4.1 gives :math:`0.92 \to 0.90`); rounded means
above 1.00 are set to 1.00. The five rating bands are 250, 500, 1000, 2000 and
4000 Hz, fed from the fifteen one-third octaves 200 Hz to 5000 Hz.

**Weighted absorption (Clause 4.2).** The fixed reference curve of Figure 1
(``{250: 0.80, 500: 1.00, 1000: 1.00, 2000: 1.00, 4000: 0.90}``) is shifted
downwards, towards the measured ``alpha_p``, in steps of 0,05 until the sum
of the unfavourable deviations -- taken only where the measured value lies
below the shifted curve, with magnitude ``curve - measured`` -- is not more
than 0,10. ``alpha_w`` is the shifted-curve value read at 500 Hz, i.e.
``1.00 - shift``.

**Shape indicators (Clause 4.3).** A shape indicator is appended in
parentheses whenever a practical coefficient exceeds the shifted reference
curve by 0,25 or more: ``L`` at 250 Hz, ``M`` at 500 Hz or 1000 Hz, ``H`` at
2000 Hz or 4000 Hz (e.g. ``0.60(M)``).

**Absorption class (Table B.1, informative).** ``alpha_w`` maps to a class:
A (0,90-1,00), B (0,80-0,85), C (0,60-0,75), D (0,30-0,55), E (0,15-0,25),
and "Not classified" (0,00-0,10). Because ``alpha_w`` is always a multiple
of 0,05 these ranges partition the grid exactly.

The rating is defined only over the whole reference range 250 Hz to
4000 Hz (Clause 1); the 125 Hz octave that is customarily plotted is not
part of the shift and is not produced here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..._internal.warnings import _warn_renamed

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from ..._report.metadata import ReportMetadata

__all__ = [
    "OCTAVE_BANDS",
    "REFERENCE_CURVE",
    "THIRD_OCTAVE_BANDS",
    "AbsorptionRatingResult",
    "absorption_class",
    "practical_absorption_coefficient",
    "weighted_absorption",
    "weighted_absorption_from_third_octave",
]

# --- ISO 11654:1997 fixed data ------------------------------------------

#: Octave rating bands, in Hz (Clause 4.1); the reference-curve bands.
OCTAVE_BANDS: tuple[int, ...] = (250, 500, 1000, 2000, 4000)

#: One-third-octave bands feeding the five octaves, in Hz (200 Hz to
#: 5000 Hz); each consecutive triple averages into one octave (Clause 4.1).
THIRD_OCTAVE_BANDS: tuple[int, ...] = (
    200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
    4000, 5000,
)

#: Reference absorption curve, Figure 1 (Clause 4.2), per octave band.
REFERENCE_CURVE: dict[int, float] = {
    250: 0.80, 500: 1.00, 1000: 1.00, 2000: 1.00, 4000: 0.90,
}

#: Index of the 500 Hz band, where ``alpha_w`` is read (Clause 4.2).
_INDEX_500 = 1

#: Reference curve as integer twentieths (units of 0,05) for exact,
#: float-safe grid arithmetic: 0.80, 1.00, 1.00, 1.00, 0.90.
_REFERENCE_UNITS: tuple[int, ...] = (16, 20, 20, 20, 18)

#: Unfavourable-deviation budget of Clause 4.2, in twentieths (0,10).
_UNFAVOURABLE_BUDGET_UNITS = 2

#: Shape-indicator excess threshold of Clause 4.3, in twentieths (0,25).
_SHAPE_THRESHOLD_UNITS = 5

#: Table B.1 absorption classes (Annex B): lowest twentieth of the class
#: mapped to its letter; "Not classified" is the residual below E.
_CLASS_TABLE: tuple[tuple[int, str], ...] = (
    (18, "A"),  # 0,90 - 1,00
    (16, "B"),  # 0,80 - 0,85
    (12, "C"),  # 0,60 - 0,75
    (6, "D"),   # 0,30 - 0,55
    (3, "E"),   # 0,15 - 0,25
)
_NOT_CLASSIFIED = "Not classified"


# --- rounding helpers ----------------------------------------------------


def _round_half_up(value: float) -> int:
    """Round to the nearest integer, halves away from zero."""
    if value < 0.0:
        return -math.floor(-value + 0.5)
    return math.floor(value + 0.5)


def _practical_round(mean: float) -> float:
    r"""Round an octave mean per Clause 4.1: to the second decimal, then in
    steps of 0.05, capped at 1.00 (the NOTE gives :math:`0.92 \to 0.90`)."""
    hundredths = _round_half_up(mean * 100.0)  # second decimal
    fives = _round_half_up(hundredths / 5.0) * 5  # nearest 0,05
    fives = max(0, min(100, fives))  # maximise to 1,00 (Clause 4.1)
    return fives / 100.0


def _to_units(alpha: float) -> int:
    """Snap an absorption coefficient to the 0,05 grid as an integer
    twentieth in ``[0, 20]`` (``1.00`` -> 20), for float-safe comparison."""
    return max(0, min(20, _round_half_up(alpha * 20.0)))


# --- input coercion ------------------------------------------------------


def _coerce(
    values: Mapping[Any, float] | Sequence[float] | ArrayLike,
    centers: tuple[int, ...],
    name: str,
) -> list[float]:
    """Return ``values`` ordered by ``centers``.

    Accepts a mapping keyed by band centre frequency (Hz) or a plain
    sequence already ordered to match ``centers``.
    """
    if isinstance(values, Mapping):
        out: list[float] = []
        for c in centers:
            if c in values:
                out.append(float(values[c]))
            elif float(c) in values:
                out.append(float(values[float(c)]))
            else:
                raise ValueError(
                    f"{name} mapping is missing band {c} Hz; "
                    f"expected keys {centers}"
                )
    else:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {arr.shape}.")
        out = arr.tolist()
        if len(out) != len(centers):
            raise ValueError(
                f"{name} must have {len(centers)} values for bands {centers}, "
                f"got {len(out)}"
            )
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} values must all be finite (no NaN or inf).")
    return out


# --- result --------------------------------------------------------------


@dataclass(frozen=True)
class AbsorptionRatingResult:
    r"""Weighted sound absorption rating (ISO 11654:1997).

    :ivar alpha_w: Weighted sound absorption coefficient ``alpha_w``, the
        shifted reference curve read at 500 Hz (Clause 4.2). A multiple of
        0,05 in ``[0, 1]``.
    :ivar shape_indicator: Concatenated shape indicators, ``L``/``M``/``H``
        in that order, or an empty string when none applies (Clause 4.3).
    :ivar absorption_class: Sound absorption class ``A``-``E`` or
        ``"Not classified"`` from Table B.1 (Annex B).
    :ivar shift: Downward shift applied to the reference curve, in
        absorption units (Clause 4.2); :math:`\alpha_w = 1.00 - \text{shift}`.
    :ivar unfavourable_sum: Sum of the unfavourable deviations at the final
        shift (Clause 4.2); at most 0.10.
    :ivar band_centers: Octave rating-band centre frequencies, in Hz
        (250 Hz to 4000 Hz).
    :ivar measured: Practical absorption coefficients ``alpha_p`` used for
        the rating (snapped to the 0,05 grid of Clause 4.1).
    :ivar shifted_reference: Reference curve of Figure 1 after the final
        shift.
    :ivar third_octave_alpha_s: The fifteen one-third-octave sound absorption
        coefficients ``alpha_s`` (200 Hz to 5000 Hz, ISO 354) the rating was
        built from, as given; ``None`` when the rating was formed from octave
        practical coefficients only. Carried so the fiche can show the full
        one-third-octave table every accredited ISO 354 certificate reports.
    :ivar third_octave_bands: The one-third-octave centre frequencies, in Hz,
        that pair with ``third_octave_alpha_s`` (the module
        :data:`THIRD_OCTAVE_BANDS`); ``None`` when ``third_octave_alpha_s`` is.
    """

    alpha_w: float
    shape_indicator: str
    absorption_class: str
    shift: float
    unfavourable_sum: float
    band_centers: NDArray[np.float64]
    measured: NDArray[np.float64]
    shifted_reference: NDArray[np.float64]
    third_octave_alpha_s: NDArray[np.float64] | None = None
    third_octave_bands: NDArray[np.float64] | None = None

    @property
    def rating_label(self) -> str:
        """The rating as reported in Clause 5.3, e.g. ``"0.60(M)"`` or
        ``"0.60"`` when no shape indicator applies."""
        if self.shape_indicator:
            return f"{self.alpha_w:.2f}({self.shape_indicator})"
        return f"{self.alpha_w:.2f}"

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the practical curve vs the shifted reference (ISO 11654).

        Unfavourable deviations (measured below the shifted reference) are
        shaded and ``alpha_w`` annotated. Requires matplotlib
        (``pip install phonometry[plot]``); returns the
        :class:`~matplotlib.axes.Axes` and never calls ``plt.show``.
        """
        from ..._i18n import check_language
        from ..._plot.materials import plot_weighted_absorption

        check_language(language)
        return plot_weighted_absorption(self, ax=ax, language=language, **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ISO 11654 sound-absorption rating fiche to a PDF.

        Writes a one-page accredited absorption report: the standard-basis
        line, an optional metadata header block, an absorption table beside the
        practical-versus-shifted-reference plot (the result's own :meth:`plot`),
        the boxed ``alpha_w`` result with its absorption class, an optional
        verdict row and a footer with the fixed disclaimer. When the rating was
        built by :func:`weighted_absorption_from_third_octave`, the table is the
        full ISO 354 one-third-octave ``alpha_s`` table with the octave
        ``alpha_p`` on the matching rows, as accredited certificates print it.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a
            prediction fiche (body, result and disclaimer only). A supplied
            ``requirement`` is read as the minimum ``alpha_w``.
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: When ``True``, the table adds the ISO 11654 evaluation
            columns (practical coefficient, shifted reference, unfavourable
            deviation) instead of the two-column ``f | alpha_p`` table.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from ..._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from ..._report.iso11654 import render_iso11654_report

        return render_iso11654_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


# --- public functions ----------------------------------------------------


def practical_absorption_coefficient(
    third_octave_alpha_s: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> NDArray[np.float64]:
    """Practical sound absorption coefficients ``alpha_p`` (ISO 11654 Clause 4.1).

    :param third_octave_alpha_s: The fifteen one-third-octave coefficients
        ``alpha_s`` from 200 Hz to 5000 Hz (ISO 354), as a sequence ordered
        low to high or a mapping keyed by band centre frequency (Hz).
        Values may exceed 1,00 (reverberation-room measurement); the
        resulting octave coefficient is capped at 1,00.
    :returns: The five octave practical coefficients for 250, 500, 1000,
        2000 and 4000 Hz, each the mean of its three one-third octaves
        rounded in steps of 0,05.
    :raises ValueError: if any coefficient is negative or the wrong number
        of values is supplied.
    """
    values = _coerce(third_octave_alpha_s, THIRD_OCTAVE_BANDS, "third_octave_alpha_s")
    if any(v < 0.0 for v in values):
        raise ValueError("alpha_s values must be non-negative")
    alpha_p = [
        _practical_round((values[3 * i] + values[3 * i + 1] + values[3 * i + 2]) / 3.0)
        for i in range(len(OCTAVE_BANDS))
    ]
    return np.asarray(alpha_p, dtype=np.float64)


def _shape_indicator(measured_units: list[int], reference_units: list[int]) -> str:
    """Shape indicators of Clause 4.3: a band exceeding the shifted
    reference by 0,25 or more contributes L (250 Hz), M (500/1000 Hz) or
    H (2000/4000 Hz)."""
    excess = [
        m - r >= _SHAPE_THRESHOLD_UNITS
        for m, r in zip(measured_units, reference_units)
    ]
    indicator = ""
    if excess[0]:
        indicator += "L"
    if excess[1] or excess[2]:
        indicator += "M"
    if excess[3] or excess[4]:
        indicator += "H"
    return indicator


def weighted_absorption(
    alpha_p: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> AbsorptionRatingResult:
    """Weighted sound absorption coefficient ``alpha_w`` (ISO 11654 Clause 4.2).

    The reference curve of Figure 1 is shifted downwards in steps of 0,05,
    towards the measured practical coefficients, until the sum of the
    unfavourable deviations (measured below the shifted curve) is at most
    0,10; ``alpha_w`` is the shifted curve read at 500 Hz.

    :param alpha_p: The five octave practical coefficients for 250, 500,
        1000, 2000 and 4000 Hz (e.g. from
        :func:`practical_absorption_coefficient`), as a sequence ordered
        low to high or a mapping keyed by band centre frequency (Hz).
        Inputs are snapped to the 0,05 grid of Clause 4.1.
    :returns: A frozen :class:`AbsorptionRatingResult` with ``alpha_w``, the
        shape indicators, the applied shift, the fitted reference curve and
        the absorption class.
    """
    measured = _coerce(alpha_p, OCTAVE_BANDS, "alpha_p")
    measured_units = [_to_units(v) for v in measured]

    # Shift the reference down in 0,05 steps until Clause 4.2 is satisfied.
    # The unfavourable sum is non-increasing in the shift, so the first
    # (smallest) qualifying shift is the answer.
    shift_units = 0
    while shift_units <= 20:
        shifted_units = [r - shift_units for r in _REFERENCE_UNITS]
        unfav_units = sum(
            max(0, s - m) for s, m in zip(shifted_units, measured_units)
        )
        if unfav_units <= _UNFAVOURABLE_BUDGET_UNITS:
            break
        shift_units += 1

    shifted_units = [r - shift_units for r in _REFERENCE_UNITS]
    alpha_w_units = shifted_units[_INDEX_500]
    indicator = _shape_indicator(measured_units, shifted_units)

    return AbsorptionRatingResult(
        alpha_w=alpha_w_units / 20.0,
        shape_indicator=indicator,
        absorption_class=absorption_class(alpha_w_units / 20.0),
        shift=shift_units / 20.0,
        unfavourable_sum=unfav_units / 20.0,
        band_centers=np.asarray(OCTAVE_BANDS, dtype=np.float64),
        measured=np.asarray([u / 20.0 for u in measured_units], dtype=np.float64),
        shifted_reference=np.asarray(
            [u / 20.0 for u in shifted_units], dtype=np.float64
        ),
    )


def weighted_absorption_from_third_octave(
    third_octave_alpha_s: Mapping[Any, float] | Sequence[float] | ArrayLike,
) -> AbsorptionRatingResult:
    """Weighted absorption rating from one-third-octave ``alpha_s`` (ISO 11654).

    Convenience wrapper over :func:`practical_absorption_coefficient` and
    :func:`weighted_absorption`: it forms the octave practical coefficients
    ``alpha_p`` (Clause 4.1) from the fifteen one-third-octave coefficients,
    rates them (Clause 4.2) and returns the same
    :class:`AbsorptionRatingResult` as :func:`weighted_absorption` would, but
    with the input ``alpha_s`` and its band centres retained on the result so a
    fiche can print the full one-third-octave table that every accredited
    ISO 354 certificate carries.

    :param third_octave_alpha_s: The fifteen one-third-octave coefficients
        ``alpha_s`` from 200 Hz to 5000 Hz (ISO 354), as a sequence ordered low
        to high or a mapping keyed by band centre frequency (Hz). Values may
        exceed 1,00 (reverberation-room measurement); the octave coefficient is
        capped at 1,00.
    :returns: A frozen :class:`AbsorptionRatingResult` identical to
        :func:`weighted_absorption` on the derived ``alpha_p``, additionally
        carrying ``third_octave_alpha_s`` (the input as given) and
        ``third_octave_bands`` (:data:`THIRD_OCTAVE_BANDS`).
    :raises ValueError: if any coefficient is negative or the wrong number of
        values is supplied.
    """
    values = _coerce(third_octave_alpha_s, THIRD_OCTAVE_BANDS, "third_octave_alpha_s")
    alpha_p = practical_absorption_coefficient(values)
    result = weighted_absorption(alpha_p)
    # `replace` is typed as returning a generic dataclass instance, so the
    # concrete type is restated for the reader and for the analyzers.
    rated: AbsorptionRatingResult = replace(
        result,
        third_octave_alpha_s=np.asarray(values, dtype=np.float64),
        third_octave_bands=np.asarray(THIRD_OCTAVE_BANDS, dtype=np.float64),
    )
    return rated


def absorption_class(alpha_w: float) -> str:
    """Sound absorption class for ``alpha_w`` (ISO 11654 Table B.1, Annex B).

    :param alpha_w: Weighted sound absorption coefficient (a multiple of
        0,05 in ``[0, 1]``).
    :returns: ``"A"``, ``"B"``, ``"C"``, ``"D"``, ``"E"`` or
        ``"Not classified"``.
    """
    units = _to_units(alpha_w)
    for lowest, letter in _CLASS_TABLE:
        if units >= lowest:
            return letter
    return _NOT_CLASSIFIED


# --- Deprecated aliases (phonometry 3.1 renames; remove in 4.0) ----------

#: Old constant name -> canonical name (units moved to the docstring).
_RENAMED_CONSTANTS: dict[str, str] = {
    "OCTAVE_BANDS_HZ": "OCTAVE_BANDS",
    "THIRD_OCTAVE_BANDS_HZ": "THIRD_OCTAVE_BANDS",
}


def __getattr__(name: str) -> Any:
    """PEP 562 shim warning for the renamed band constants."""
    try:
        canonical = _RENAMED_CONSTANTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'phonometry.absorption_rating' has no attribute {name!r}"
        ) from None
    _warn_renamed(name, canonical)
    return globals()[canonical]
