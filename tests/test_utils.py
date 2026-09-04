#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Tests for internal utility helpers."""

from typing import Any

import numpy as np
import pytest

from phonometry._internal.utils import _resample_to_length, _typesignal


def test_typesignal_converts_int_arrays_to_float64() -> None:
    """Integer inputs must be promoted to float64 to avoid overflow when squared."""
    x = np.array([1, -2, 3], dtype=np.int16)
    out = _typesignal(x)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [1.0, -2.0, 3.0])


def test_typesignal_preserves_float64_without_copy() -> None:
    """float64 arrays pass through unchanged (no copy)."""
    x = np.array([1.0, 2.0])
    assert _typesignal(x) is x


def test_typesignal_refuses_none_as_the_whole_signal() -> None:
    """None is not a signal, and converts to a NaN array rather than raising."""
    with pytest.raises(ValueError, match=r"'x' must be a signal, not None"):
        _typesignal(None)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("hello", id="string"),
        pytest.param({"a": 1.0}, id="dict"),
        pytest.param(object(), id="object"),
        pytest.param([[1.0, 2.0], [3.0]], id="ragged"),
    ],
)
def test_typesignal_refuses_what_is_not_numeric_by_name(bad: Any) -> None:  # noqa: ANN401 - the point is that anything at all may arrive here
    """What numpy cannot read as numbers is refused naming the parameter.

    Numpy answers a string or a ragged list in its own words and an object or
    a dict with a TypeError, which is not the exception the entry points
    document.
    """
    with pytest.raises(ValueError, match=r"'x' must be numeric"):
        _typesignal(bad)


def test_typesignal_refuses_a_complex_signal_saying_so() -> None:
    """A complex input gets its own message, because it is its own mistake.

    It used to share the "must be numeric" refusal, which was true of the list
    spelling only: numpy converts a complex *array* to float64 by dropping the
    imaginary part, under a warning rather than an error. Saying "not complex"
    tells a caller who passed an analytic signal what actually happened.
    """
    with pytest.raises(ValueError, match=r"'x' must be real, not complex"):
        _typesignal(1 + 2j)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([1.0, np.nan, 3.0], id="nan"),
        pytest.param([1.0, np.inf, 3.0], id="inf"),
        pytest.param([1.0, None, 3.0], id="none-inside-the-sequence"),
    ],
)
def test_typesignal_refuses_a_non_finite_sample(bad: Any) -> None:  # noqa: ANN401 - a list carrying a None is not a Sequence[float]
    """One poisoned sample is refused, whichever of the two routes it came by.

    A None inside the sequence and a NaN the caller passed are the same value
    after conversion, so the message speaks for both.
    """
    with pytest.raises(ValueError, match=r"'x' must contain only finite samples"):
        _typesignal(bad)


def test_typesignal_names_the_parameter_it_was_given() -> None:
    """The name is the caller's, so a refusal names an argument that exists."""
    with pytest.raises(ValueError, match=r"'reference' must be numeric"):
        _typesignal("hello", name="reference")


@pytest.mark.parametrize(
    ("legal", "shape"),
    [
        pytest.param([], (0,), id="empty"),
        pytest.param([1.0], (1,), id="single-sample"),
        pytest.param(np.zeros(16), (16,), id="silent-buffer-of-exact-zeros"),
        pytest.param(np.arange(8, dtype=np.int16), (8,), id="integer-array"),
        pytest.param(np.zeros((2, 8)), (2, 8), id="two-dimensional"),
        pytest.param(
            np.ma.masked_array([1.0, 2.0, 3.0], mask=[0, 1, 0]), (3,), id="masked"
        ),
        pytest.param(np.asfortranarray(np.zeros((2, 8))), (2, 8), id="fortran-ordered"),
        pytest.param(np.zeros(16)[::2], (8,), id="non-contiguous-slice"),
    ],
)
def test_typesignal_still_accepts_every_legal_shape_of_signal(
    legal: Any,  # noqa: ANN401 - the parametrization is over unrelated array-likes
    shape: tuple[int, ...],
) -> None:
    """The refusals must not have narrowed what a signal is allowed to be.

    A resolver under twenty-nine entry points refuses for all of them at once,
    so an empty record, one sample, a silence of exact zeros, an int16 take
    straight off a WAV, a (channels, samples) block, a masked array, a
    Fortran-ordered block and a strided view all have to survive it.
    """
    out = _typesignal(legal)
    assert out.shape == shape
    assert out.dtype == np.float64


def test_resample_to_length_padding_and_trimming() -> None:
    """Verify _resample_to_length pads and trims 1D and 2D signals."""
    x = np.arange(10, dtype=float)
    y = _resample_to_length(x, 1, 12)
    assert len(y) == 12
    np.testing.assert_array_equal(y[:10], x)
    assert np.all(y[10:] == 0)

    x2 = np.vstack([np.arange(10, dtype=float), np.arange(10, 20, dtype=float)])
    y2 = _resample_to_length(x2, 1, 12)
    assert y2.shape == (2, 12)
    np.testing.assert_array_equal(y2[:, :10], x2)
    assert np.all(y2[:, 10:] == 0)

    y3 = _resample_to_length(x, 1, 8)
    assert len(y3) == 8
    np.testing.assert_array_equal(y3, x[:8])

    y4 = _resample_to_length(x2, 1, 8)
    assert y4.shape == (2, 8)
    np.testing.assert_array_equal(y4, x2[:, :8])


@pytest.mark.parametrize(
    ("entry", "field"),
    [("resolve_samples", "x"), ("apply_calibration", "ref_signal")],
)
def test_a_factor_that_overflows_its_samples_is_refused(entry: str, field: str) -> None:
    """Two finite numbers whose product is not, and nothing saw it.

    The samples are checked for finiteness before the calibration factor is
    applied, and a check cannot see past the multiplication that follows it.
    Both operands are finite by contract, the factor because
    :class:`~phonometry.io.Signal` requires it, so the infinity was created
    between them and left as a value.

    The refusal names both, because neither is wrong on its own: a chain that
    overflows is built wrong somewhere along it, and saying only "non-finite"
    would send the reader to whichever end they looked at first.
    """
    from phonometry.io import Signal
    from phonometry.io._resolve import apply_calibration, resolve_samples

    samples = np.full(4, 1e300)
    signal = Signal(samples, 48000, calibration_factor=1e10)
    call = (
        (lambda: resolve_samples(signal))
        if entry == "resolve_samples"
        else (lambda: apply_calibration(signal, samples, name=field))
    )
    with pytest.raises(
        ValueError, match=rf"'{field}': the calibration factor overflows"
    ):
        call()


def test_a_calibration_the_instruments_actually_use_is_untouched() -> None:
    """The headroom is not close, and the guard must not pretend otherwise.

    A double reaches about 1.8e308, and the loudest quantity acoustics
    measures is a blast overpressure of a few bar. A full-scale sample scaled
    to pascals by a factor of 1e5 lands at 5e4, which is 1e303 short of the
    ceiling, so nothing an instrument produces comes near this refusal.
    """
    from phonometry.io import Signal
    from phonometry.io._resolve import resolve_samples

    scaled = resolve_samples(Signal(np.full(4, 0.5), 48000, calibration_factor=1e5))
    assert np.all(np.isfinite(scaled))
    assert float(np.max(scaled)) == pytest.approx(5.0e4)
