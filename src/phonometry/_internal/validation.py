#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Shared input-validation helpers (private).

Seeded by the library audit: modules used to hand-roll the same checks with
diverging NaN semantics and error-message styles. New code should validate
through these helpers; existing modules migrate as they are touched.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import ArrayLike


def require_positive(value: float, name: str) -> float:
    """Require a positive finite number (rejects NaN and infinities).

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite or non-positive value.
    """
    if not math.isfinite(value) or value <= 0.0:
        msg = f"'{name}' must be positive."
        raise ValueError(msg)
    return float(value)


#: Absolute zero, in degrees Celsius. The bound on a temperature that has to be
#: a temperature. It is not the same number as the ``-273`` that several ISO 3740
#: clauses print inside their own ``sqrt(273 + theta)``: those guards protect the
#: pole of the formula the clause states, which sits 0,15 degC higher, and they
#: stay where they are.
ABSOLUTE_ZERO_C = -273.15


def require_above_absolute_zero(value: float, name: str) -> float:
    """Require a finite temperature in degrees Celsius above absolute zero.

    The one bound a temperature always has. A model may state a narrower domain
    for itself, and that is a warning rather than a refusal, because a fit past
    its validated range is still arithmetic; a temperature at or below absolute
    zero is not a state at all, and the arithmetic below one is not wrong so much as
    meaningless. Left unguarded it does not raise, it returns: a sea-water sound
    speed comes back at -31 457 m/s, and an absorption at -300 degC comes back
    at a plausible-looking 0,495 dB/km.

    :param value: The temperature to validate, in degrees Celsius.
    :param name: Parameter name used in the error message.
    :return: The validated temperature as a ``float``.
    :raises ValueError: for a non-finite temperature, or one at or below
        -273,15 degC.
    """
    if not math.isfinite(value) or value <= ABSOLUTE_ZERO_C:
        msg = f"'{name}' must be finite and above -273.15 degC."
        raise ValueError(msg)
    return float(value)


def require_above_absolute_zero_array(x: ArrayLike, name: str) -> np.ndarray:
    """Require every temperature in ``x`` to be finite and above absolute zero.

    The array companion of :func:`require_above_absolute_zero`. A profile is
    where this bites hardest: one element at or below absolute zero among a
    hundred good ones still poisons the whole returned array, and the caller has no
    reason to look at any single element of it.

    :param x: Temperatures, in degrees Celsius.
    :param name: Parameter name used in the error message.
    :return: The temperatures as a ``float64`` array.
    :raises ValueError: if any element is non-finite or at or below
        -273,15 degC.
    """
    values = _as_float64(x, name)
    if not np.all(np.isfinite(values)) or np.any(values <= ABSOLUTE_ZERO_C):
        # Phrased so it reads for a single value and for a column alike: the
        # same helper serves 'temperature' and 'temperatures'.
        msg = f"'{name}' must be finite and above -273.15 degC."
        raise ValueError(msg)
    return values


def require_non_negative(value: float, name: str) -> float:
    """Require a non-negative finite number (rejects NaN and infinities).

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite or negative value.
    """
    if not math.isfinite(value) or value < 0.0:
        msg = f"'{name}' must be non-negative."
        raise ValueError(msg)
    return float(value)


def require_fraction(value: float, name: str) -> float:
    """Require a finite fraction in ``[0, 1)``.

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :return: The validated value as a ``float``.
    :raises ValueError: for a non-finite value or one outside ``[0, 1)``.
    """
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        msg = f"'{name}' must be in the range [0, 1)."
        raise ValueError(msg)
    return float(value)


def require_choice(value: str, name: str, options: tuple[str, ...]) -> str:
    """Require *value* to be one of *options*.

    :param value: The value to validate.
    :param name: Parameter name used in the error message.
    :param options: The accepted values.
    :return: The validated value.
    :raises ValueError: for a value not in *options*.
    """
    if value not in options:
        msg = f"'{name}' must be one of {options}; got {value!r}."
        raise ValueError(msg)
    return value


def check_engine(engine: str) -> None:
    """Require *engine* to name the one report engine there is.

    The counterpart of :func:`phonometry._i18n.check_language` for the
    ``engine`` parameter every ``report`` entry point takes. A single-choice
    check reads oddly until the choice is seen for what it is: ``"reportlab"``
    is the only engine today, and the parameter is the seam a second engine
    would arrive through. Gating it here keeps every entry point rejecting the
    same way, with one message to update when that day comes.

    :param engine: The requested report engine.
    :raises ValueError: for any value other than ``"reportlab"``.
    """
    if engine == "reportlab":
        return
    msg = f"Unknown report engine {engine!r}; only 'reportlab' is supported."
    raise ValueError(msg)


def _as_float64(x: ArrayLike, name: str) -> np.ndarray:
    """Convert *x* to ``float64``, naming the parameter when numpy cannot.

    A string element raises numpy's anonymous "could not convert string to
    float", and an unconvertible object a bare ``TypeError``; both would
    escape the array guards below without saying which parameter refused.

    A complex array is refused rather than converted. ``np.asarray(z,
    dtype=float64)`` does not fail on one: it drops the imaginary part and
    emits a ``ComplexWarning``, which is a warning and not an error, so
    ``np.array([1 + 2j])`` would arrive at the guards below as ``[1.0]`` and
    pass every one of them. A list of complex numbers already raised here, so
    the two spellings of the same input disagreed.
    """
    try:
        complex_input = np.iscomplexobj(x)
    except (TypeError, ValueError) as exc:
        # `iscomplexobj` builds the array to read its dtype, so a ragged list
        # fails here rather than below and would escape without naming the
        # parameter. Its refusal is the same refusal, in the same words.
        msg = f"'{name}' must be numeric."
        raise ValueError(msg) from exc
    if complex_input:
        msg = f"'{name}' must be real, not complex."
        raise ValueError(msg)
    try:
        return np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"'{name}' must be numeric."
        raise ValueError(msg) from exc


def require_positive_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce *x* to a 1-D float64 array of strictly positive, finite values.

    :param x: The input (scalar or 1-D array-like).
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array (at least 1-D).
    :raises ValueError: for a non-numeric, empty, multi-dimensional,
        non-finite or non-positive input.
    """
    arr = np.atleast_1d(_as_float64(x, name))
    if arr.ndim != 1 or arr.size == 0:
        msg = f"'{name}' must be a non-empty 1-D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must be finite."
        raise ValueError(msg)
    if np.any(arr <= 0.0):
        msg = f"'{name}' must be strictly positive."
        raise ValueError(msg)
    return arr


def require_finite_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce *x* to a non-empty 1-D float64 array of finite values.

    The sign-agnostic sibling of :func:`require_positive_array`, for the
    level-like quantities (dB values, corrections) that may legitimately be
    negative but never NaN.

    :param x: The input (scalar or 1-D array-like).
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array (at least 1-D).
    :raises ValueError: for a non-numeric, empty, multi-dimensional or
        non-finite input.
    """
    arr = np.atleast_1d(_as_float64(x, name))
    if arr.ndim != 1 or arr.size == 0:
        msg = f"'{name}' must be a non-empty 1-D array."
        raise ValueError(msg)
    if not np.all(np.isfinite(arr)):
        msg = f"'{name}' must contain only finite values."
        raise ValueError(msg)
    return arr


def require_axis_count(
    value: object, owner: str, name: str, axis: str = "band", *, rank: int | None
) -> int:
    """The number of entries along the first axis of *value*.

    ``len()``, not ``np.shape()``: the first axis of a tuple of per-band
    arrays is its length whatever the arrays hold, while ``np.shape`` would
    try to stack them and raise about an inhomogeneous shape for a bank whose
    bands carry filters of different orders.

    *rank* has no default and must be stated, because a count on its own says
    nothing about how many axes there are: an array of shape ``(bands, 2)``
    counts the right number of bands on its first axis and carries the second
    one onward untouched. Pass ``None`` only where the rank is pinned
    elsewhere, as :func:`require_ranks` does for a whole result at once.

    :param value: The field whose entries are counted.
    :param owner: Name of the type being checked, used in the error message.
    :param name: Field name used in the error message.
    :param axis: What the entries measure, singular, for the error message.
    :param rank: The number of axes the field must have, or ``None`` when that
        is checked elsewhere.
    :return: The number of entries.
    :raises ValueError: if *value* is a single number, or has another rank.
    """
    if rank is not None:
        require_axis_rank(value, owner, name, rank)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        msg = f"{owner}: '{name}' must carry one value per {axis}, not a single number."
        raise ValueError(msg) from None


def _rank(value: object) -> int:
    """The number of axes a field carries, as the guards count them.

    One place, because three callers ask the same question and numpy answers
    it badly for two kinds of field the library really holds.

    A sequence whose entries are arrays is one axis of entries, whatever each
    entry holds. A filter bank keeps one set of second-order sections per band
    and the sets need not be the same length; stacking them either raises
    about an inhomogeneous shape or, when the orders happen to agree, invents
    an axis nobody meant. Both cases have to answer the same, or a bank would
    be accepted or refused according to a coincidence between its bands.

    Everything else is measured. A nested list is as two-dimensional as the
    array it would become, and calling it one axis would refuse a correctly
    shaped grid handed in as lists while letting a list with an extra axis
    walk past the pin that exists to stop one.
    """
    if isinstance(value, (tuple, list)) and any(
        isinstance(entry, np.ndarray) for entry in value
    ):
        return 1
    try:
        return int(np.ndim(cast("ArrayLike", value)))
    except ValueError:
        # Ragged in some other shape: numpy has no rank to give, and the
        # sequence is still one axis of whatever it holds.
        return 1


def require_axis_rank(value: object, owner: str, name: str, rank: int) -> None:
    """Require *value* to have exactly *rank* axes.

    Counting an axis is not enough on its own: an array of shape
    ``(paths, bands, 2)`` has the right number of paths on its first axis and
    the right number of bands on its second, so every count agrees and the
    extra axis travels on into whatever indexes the field next.

    The rank is measured, not assumed from the type. A nested list is as
    two-dimensional as the array it would become, and taking it for a single
    axis would be wrong twice over: it would refuse a correctly shaped grid
    handed in as lists, and it would let a list carrying an extra axis walk
    past the very pin that exists to stop one.

    A sequence of arrays escapes measurement, ragged or not: see :func:`_rank`,
    which is where all three callers ask this question so that they cannot
    drift apart.

    :param value: The field whose axes are counted.
    :param owner: Name of the type being checked, used in the error message.
    :param name: Field name used in the error message.
    :param rank: The number of axes the field must have.
    :raises ValueError: if *value* has a different number of axes.
    """
    actual = _rank(value)
    if actual != rank:
        axes = "one axis" if rank == 1 else f"{rank} axes"
        why = (
            "An extra axis passes every count and reaches the reader intact."
            if actual > rank
            else "A value short of an axis is spread over the ones it was "
            "never measured on."
        )
        msg = f"{owner}: '{name}' must have {axes}; got {actual}. {why}"
        raise ValueError(msg)


def require_ranks(owner: object, **ranks: int) -> None:
    """Require each named field of *owner* to have the given number of axes.

    The companion of :func:`require_same_length`, which pins how long an axis
    is but not how many there are. ``None`` fields are skipped.

    A result whose every pinned field is a bare number is passed over whole.
    Several entry points take a single frequency and hand back nought-
    dimensional arrays throughout; that is not an extra axis, it is no axis.
    The exemption is all or nothing on purpose. Waiving it field by field
    would leave a lone scalar sitting beside real spectra wherever no length
    check happens to cover that field, and about ten fields here are pinned
    against an extra axis without being pinned against a missing one.

    :param owner: The instance whose fields are measured.
    :param ranks: Field name to the number of axes it must have.
    :raises ValueError: if a field has a different number of axes.
    """
    present = [
        (name, rank, value)
        for name, rank in ranks.items()
        if (value := getattr(owner, name)) is not None
    ]
    if all(_rank(value) == 0 for _, _, value in present):
        return
    for name, rank, value in present:
        require_axis_rank(value, type(owner).__name__, name, rank)


def require_equal_counts(
    owner: str, counts: Mapping[str, int], axis: str = "band"
) -> None:
    """Require every measured count to agree, naming each one that does not.

    The sibling of :func:`require_same_length` for the counts a caller has
    already measured: entries of a nested sequence, an axis picked out of a
    two-dimensional field, anything whose label is not a plain attribute name.

    The message lists every count it was given rather than picking out an odd
    one, because among peers no count is authoritative and a majority winner
    would be a guess. Where one quantity *is* authoritative, pass it and the
    single suspect: two entries make the shortest true message there is.

    :param owner: Name of the type being checked, used in the error message.
    :param counts: Label to count, in the order they should be reported.
    :param axis: What the counts measure, singular, for the error message.
    :raises ValueError: if two of the counts disagree.
    """
    if len(set(counts.values())) <= 1:
        return
    listed = ", ".join(f"'{label}' ({n})" for label, n in counts.items())
    msg = f"{owner}: {listed} must each carry one value per {axis}."
    raise ValueError(msg)


def require_same_length(
    owner: object, *fields: str | tuple[str, int], axis: str = "band"
) -> None:
    """Require the named fields of *owner* to share one length.

    Written for the ``__post_init__`` of a result dataclass, so that a result
    whose per-band arrays disagree cannot be built at all. Validating here
    rather than where the values are read covers the readers that do not
    exist yet, and both directions of the mistake: an array one short raises
    somewhere downstream, but an array one long is silently truncated by the
    table that prints it, under a total that summed the whole of it.

    A field may be given as ``"name"`` for a length along the first axis, or
    as ``("name", i)`` when the axis in question is not the first one, as it
    is not for a per-path-per-band array whose bands run along axis 1.

    ``None`` fields are skipped: an absent optional quantity is not a
    disagreement.

    A result whose every field is a bare number is left alone. Several entry
    points take a single frequency and hand back a result of nought-dimensional
    arrays, and there is no axis there to disagree about; only a mixture of
    scalars and arrays is a mistake, and that is what the count below reports.

    :param owner: The instance whose fields are compared.
    :param fields: Field names, or ``(name, axis_index)`` pairs.
    :param axis: What the lengths measure, singular, for the error message.
    :raises ValueError: if two of the fields disagree, or one is a scalar.
    """
    named = [(field if isinstance(field, tuple) else (field, 0)) for field in fields]
    present = [
        (name, index, value)
        for name, index in named
        if (value := getattr(owner, name)) is not None
    ]
    if all(_rank(value) == 0 for _, _, value in present):
        return
    counts: dict[str, int] = {}
    for name, index, value in present:
        if index == 0:
            counts[name] = require_axis_count(
                value, type(owner).__name__, name, axis, rank=None
            )
            continue
        shape = np.shape(value)
        if len(shape) <= index:
            msg = (
                f"{type(owner).__name__}: '{name}' must have an axis {index} "
                f"to carry one value per {axis}; got shape {shape}."
            )
            raise ValueError(msg)
        counts[f"{name} (axis {index})"] = shape[index]
    require_equal_counts(type(owner).__name__, counts, axis)


def require_equal_shapes(
    owner: str, shapes: Mapping[str, tuple[int, ...]], quantity: str = "point"
) -> None:
    """Require every shape to agree, naming each one that does not.

    What :func:`require_equal_counts` is to :func:`require_same_length`, this
    is to :func:`require_same_shape`: the form for values a caller is holding
    loose, where there is no instance to read the fields off. An entry point
    validating two of its own arguments is the usual case.

    Shape rather than length, because two grids of the same height agree on
    every count and can still disagree about every value in them.

    :param owner: Name of the entry point, used in the error message. The one
        the caller typed, not the private validator it reached.
    :param shapes: Label to shape, in the order they should be reported.
    :param quantity: What one element is, singular, for the error message.
    :raises ValueError: if two of the shapes disagree.
    """
    if len(set(shapes.values())) <= 1:
        return
    listed = ", ".join(f"'{label}' {shape}" for label, shape in shapes.items())
    msg = f"{owner}: {listed} must all have the same shape, one value per {quantity}."
    raise ValueError(msg)


def require_same_shape(owner: object, *fields: str, quantity: str = "point") -> None:
    """Require the named fields of *owner* to have exactly the same shape.

    The stricter sibling of :func:`require_same_length`, for fields derived
    from one another elementwise rather than laid side by side along one axis.
    A propagation loss and the signal excess computed from it agree point for
    point, and the solver that produced them may well have returned a grid
    over depth and range, so pinning a single axis would be both too weak,
    two grids of the same height can disagree everywhere else, and too strong,
    since it would refuse the grid the library itself hands back.

    ``None`` fields are skipped, and so is a set of fields that are all bare
    numbers.

    :param owner: The instance whose fields are measured.
    :param fields: Field names that must share one shape.
    :param quantity: What one element is, used in the error message.
    :raises ValueError: if two present fields have different shapes.
    """
    present = [(name, getattr(owner, name)) for name in fields]
    present = [(name, value) for name, value in present if value is not None]
    if all(np.ndim(value) == 0 for _, value in present):
        return
    shapes = {name: np.shape(value) for name, value in present}
    require_equal_shapes(type(owner).__name__, shapes, quantity)


def require_finite_fields(owner: object, *fields: str) -> None:
    """Require each named field of *owner* to carry only finite values.

    The value guard beside the shape guards of a result's ``__post_init__``,
    for the fields no producer can legitimately leave undetermined: a
    measurement surface area, a room volume, a single-number index. Where a
    quantity *is* legitimately undeterminable band by band, an ISO 9614 band
    of negative net power or an A-weighted total over no bands, it is marked
    as such by a flag and rendered as an em dash, and it must not be listed
    here.

    Shape guards alone do not cover this: a ``NaN`` has whatever shape it was
    stored in, so it passes every count and every rank, and reaches the fiche
    as the literal ``nan`` printed in an accredited table, or as a conversion
    error raised from inside a display rounder that names neither the field
    nor the result it belongs to.

    ``None`` fields are skipped: an absent optional quantity is not a
    non-finite one.

    A complex field is measured on both of its axes. Casting one to
    ``float64`` to run the check would drop the imaginary part on the way in,
    so a transfer stiffness whose damping term came back ``NaN`` beside an
    intact real part passed the guard untouched, and the phase quantities
    computed from it -- a magnitude, a loss factor -- carried the ``NaN``
    onward as though it had been measured. ``np.isfinite`` already reads a
    complex number as finite only when both components are, so the cast is
    made only for the fields that are not complex to begin with.

    That cast is also what refuses a field that is not numeric at all, and it
    is kept for the sake of it: a string or an object array reaches
    ``np.isfinite`` as an anonymous ``TypeError`` raised from inside numpy,
    which names neither the field nor the result and is not the
    ``ValueError`` this guard documents.

    :param owner: The instance whose fields are measured.
    :param fields: Field names that must be finite throughout.
    :raises ValueError: if a field carries a ``NaN`` or an infinity, or is
        not numeric.
    """
    for name in fields:
        value = getattr(owner, name)
        if value is None:
            continue
        try:
            array = np.asarray(value)
            if array.dtype.kind != "c":
                array = array.astype(np.float64)
        except (TypeError, ValueError) as exc:
            msg = f"{type(owner).__name__}: '{name}' must be numeric."
            raise ValueError(msg) from exc
        if np.all(np.isfinite(array)):
            continue
        what = "be finite" if array.ndim == 0 else "contain only finite values"
        msg = f"{type(owner).__name__}: '{name}' must {what}."
        raise ValueError(msg)


def is_class_designation(value: object, expected: Sequence[int]) -> bool:
    """Whether *value* is one of *expected* as a designation, not merely equal.

    A compliance class is a label rather than a measurement: a fiche prints it
    verbatim into ``Class 1`` and readers splice it into the
    ``margin_class1_db`` key they take the binding margin from. Equality with
    a class is therefore too weak a test. ``1.0`` labels a band ``Class 1.0``
    and builds ``margin_class1.0_db``, a key no band carries, and ``True`` is
    an ``int`` in Python and would label one ``Class True``. NumPy integers
    are admitted: they are the same designation and print as it.

    :param value: The candidate designation.
    :param expected: The classes the edition or standard defines.
    :return: ``True`` when *value* designates one of *expected*.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return False
    return int(value) in expected


def require_summary_class(
    owner: object,
    bands: Sequence[Mapping[str, object]],
    overall: object,
    expected: Sequence[int],
    *,
    field: str = "overall_class",
) -> None:
    """Pin a summary class against the per-band classes it restates.

    Two standards in the library derive one the same way, IEC 61260-1 for a
    filter bank and IEC 61043 for an intensity instrument: the strictest class
    every band meets, which is the *largest* per-band class because a band
    meeting the stricter one meets the looser too, and ``None`` as soon as one
    band meets no class at all. Their fiches print the two halves side by
    side, the per-band table from ``band["class"]`` and the boxed verdict from
    the summary, so a summary that does not restate the rows boxes ``Class 1 -
    COMPLIES`` above a table where a band reads ``Class 2``, with that band's
    negative margin quoted beside the word COMPLIES, and passes the instrument
    against a required class 1.

    Both the per-band classes and the summary are pinned as *designations*
    before the derivation runs, because the derivation cannot see the
    difference: ``1.0 == 1`` is true in Python, so a summary of ``1.0``
    restates a class 1 bank perfectly and still builds ``margin_class1.0_db``,
    a key no band carries, when a reader splices it.

    :param owner: The result instance, named in the messages.
    :param bands: The per-band verdict mappings.
    :param overall: The class stated for the whole instrument, or ``None``.
    :param expected: The classes the standard defines.
    :param field: Name of the summary field, for the message.
    An empty band list is refused a class of its own: a verdict attested over
    nothing is not a verdict, and the readers that take a binding margin from
    it die on an empty sequence instead.

    :raises ValueError: if a band carries no class or one that is no
        designation, if a class is claimed over no bands at all, or the
        summary is not the class the bands derive.
    """
    who = type(owner).__name__
    for band in bands:
        if "class" not in band:
            msg = (
                f"{who}: 'bands' must carry a 'class' per band; the "
                f"{_band_hz(band)} Hz entry has no 'class' key."
            )
            raise ValueError(msg)
        band_class = band["class"]
        if band_class is not None and not is_class_designation(band_class, expected):
            msg = (
                f"{who}: 'bands' must state a 'class' of {list(expected)} or "
                f"None per band; the {_band_hz(band)} Hz entry states "
                f"{band_class!r}."
            )
            raise ValueError(msg)
    if overall is not None and not is_class_designation(overall, expected):
        msg = (
            f"{who}: '{field}' must be a class of {list(expected)} or None; "
            f"got {overall!r}."
        )
        raise ValueError(msg)
    if not bands:
        if overall is not None:
            msg = (
                f"{who}: '{field}' is {overall!r} but 'bands' is empty: a "
                "class cannot be attested over no verified band. An "
                "instrument with no bands in range carries a summary of None."
            )
            raise ValueError(msg)
        return
    classes = [band["class"] for band in bands]
    derived = None if None in classes else max(cast("list[int]", classes))
    if overall != derived:
        binding = next(band for band in bands if band["class"] == derived)
        msg = (
            f"{who}: '{field}' must be the class the bands derive, the "
            "strictest class every band meets and None as soon as one band "
            f"meets none; got {overall!r} where the {_band_hz(binding)} Hz "
            f"band states {derived!r}."
        )
        raise ValueError(msg)


def _band_hz(band: Mapping[str, object]) -> str:
    """The band centre of *band* for a message, or ``?`` when it carries none."""
    freq = band.get("freq")
    return f"{freq:g}" if isinstance(freq, (int, float, np.number)) else "?"


def require_per_band(
    x: ArrayLike, name: str, bands: np.ndarray, bands_name: str = "frequency"
) -> np.ndarray:
    """Broadcast *x* over the band axis, saying whose length disagrees.

    A scalar applies to every band and an array of the band count is taken
    band by band. Nothing else is: broadcasting alone would also stretch a
    one-element array across every band, which is the shape a caller lands on
    by computing one value where the bands needed several, and the answer that
    comes back from repeating it looks perfectly ordinary.

    ``np.broadcast_to`` is left to do the stretching once the shape is known
    to be one of the two, because what it cannot do is name the culprit: it
    reports the two shapes it failed to reconcile, from inside its own C code,
    without saying which argument carried which.

    :param x: The per-band input (a scalar applies to every band).
    :param name: Parameter name used in the error message.
    :param bands: The band axis to match, already coerced and 1-D.
    :param bands_name: Parameter name of the band axis, for the message.
    :return: The broadcast ``float64`` array, of ``bands.shape``.
    :raises ValueError: if *x* is neither a scalar nor of ``bands``' length.
    """
    arr = _as_float64(x, name)
    if arr.ndim != 0 and arr.shape != bands.shape:
        msg = (
            f"'{name}' must be a scalar or carry one value per band "
            f"({bands.size} in '{bands_name}'); got shape {arr.shape}."
        )
        raise ValueError(msg)
    return np.broadcast_to(arr, bands.shape).astype(np.float64)


def require_1d_signal(x: ArrayLike, name: str = "signal") -> np.ndarray:
    """Coerce *x* to a float64 array and require a 1-D time series.

    Multichannel input is rejected rather than silently flattened (a raveled
    2-D signal concatenates the channels into one wrong series).

    :param x: The input signal.
    :param name: Parameter name used in the error message.
    :return: The validated ``float64`` array.
    :raises ValueError: for a non-1-D input.
    """
    arr = _as_float64(x, name)
    if arr.ndim != 1:
        msg = (
            f"{name} must be a 1-D time series; got shape {arr.shape}. "
            "Process multichannel signals one channel at a time."
        )
        raise ValueError(msg)
    return arr
