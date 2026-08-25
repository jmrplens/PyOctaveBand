#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Frequency-response and coherence estimators (Bendat & Piersol).

Two-channel (input/output) system identification from measured signals, using
the Welch-averaged cross- and auto-spectral densities. Following Bendat &
Piersol, *Random Data: Analysis and Measurement Procedures* (4th ed., 2010):

* the **H1** estimator :math:`H_1 = G_{xy} / G_{xx}` (unbiased when the
  noise is on the output),
* the **H2** estimator :math:`H_2 = G_{yy} / G_{yx}` (unbiased when the
  noise is on the input),
* the **ordinary coherence**
  :math:`\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} \cdot G_{yy})
  \in [0, 1]`, the fraction of the output power linearly explained by the
  input.

For a noiseless linear time-invariant path both estimators recover the true
transfer function and the coherence is unity; additive output noise biases H2
but not H1 and pulls the coherence down to
:math:`\mathrm{SNR} / (1 + \mathrm{SNR})`, which is the analytic oracle used
to verify the implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

# Shared Welch-core defaults and helpers (single source of truth for the
# segment policy across the spectral estimators).
from .._internal.validation import (
    require_equal_counts,
    require_ranks,
    require_same_length,
)
from ..io._resolve import apply_calibration, resolve_pair_fs
from ..signals.spectra import (
    _DEFAULT_OVERLAP,
    _MIN_SAMPLES,
    _default_nperseg,
    _noverlap_samples,
    _positive,
    _welch_pair,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from ..io._signal import Signal


def _validate_pair(
    x: Signal | NDArray[np.float64] | list[float],
    y: Signal | NDArray[np.float64] | list[float],
    owner: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the input/output record pair of *owner*.

    :param owner: Name of the entry point the caller invoked, so that the
        length complaint names the function that was typed rather than this
        validator, which two entry points share.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or ya.ndim != 1:
        msg = "'x' and 'y' must be one-dimensional."
        raise ValueError(msg)
    require_equal_counts(owner, {"x": xa.size, "y": ya.size}, "sample")
    if xa.size < _MIN_SAMPLES:
        msg = f"Signals too short for a spectral estimate: {xa.size} samples."
        raise ValueError(msg)
    if not (np.all(np.isfinite(xa)) and np.all(np.isfinite(ya))):
        msg = "'x' and 'y' must be finite."
        raise ValueError(msg)
    return apply_calibration(x, xa), apply_calibration(y, ya)


def _spectra(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    fs: float,
    nperseg: int,
    overlap: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return ``(freqs, Gxy, Gxx, Gyy)`` from Welch-averaged Hann segments.

    Thin adapter over the shared Welch core in
    :mod:`phonometry.signals.spectra` (same taper, overlap policy and
    detrend-off calibration; bit-identical to the previous local
    implementation).
    """
    return _welch_pair(x, y, fs, nperseg, _noverlap_samples(nperseg, overlap))


def _welch_params(
    n: int, fs: float, nperseg: int | None, overlap: float
) -> tuple[int, float]:
    """Validated Welch segment length and overlap fraction.

    The guards run before the ``int()``/``float()`` coercions, so a bad
    value is refused by name instead of dying inside the standard library,
    and a fractional ``nperseg`` is refused instead of silently truncated.
    The comparisons stay in exact int/float arithmetic and only floats are
    probed for finiteness, because ``float()`` and ``math.isfinite`` both
    overflow on an integer beyond float range; kept exact, such an integer
    falls through to the range check, which refuses it by name.
    """
    if (
        not isinstance(overlap, (int, float, np.integer, np.floating))
        or not 0.0 <= overlap < 1.0
    ):
        msg = "'overlap' must be in [0, 1)."
        raise ValueError(msg)
    if nperseg is not None and (
        not isinstance(nperseg, (int, float, np.integer, np.floating))
        or (isinstance(nperseg, (float, np.floating)) and not math.isfinite(nperseg))
        or int(nperseg) != nperseg
    ):
        msg = "'nperseg' must be a positive integer."
        raise ValueError(msg)
    seg = _default_nperseg(n, fs) if nperseg is None else int(nperseg)
    if seg < _MIN_SAMPLES or seg > n:
        msg = "'nperseg' must be between 32 and the signal length."
        raise ValueError(msg)
    return seg, float(overlap)


@dataclass(frozen=True)
class FrequencyResponseResult:
    r"""Estimated frequency response of an input/output path (Bendat &
    Piersol).

    :ivar frequencies: Frequency axis, in Hz.
    :ivar response: Complex frequency-response estimate ``H(f)``.
    :ivar magnitude_db: Magnitude :math:`20 \log_{10} \lvert H \rvert`, in dB.
    :ivar phase: Phase of ``H``, in radians (unwrapped).
    :ivar coherence: Ordinary coherence :math:`\gamma^2(f) \in [0, 1]`.
    :ivar estimator: Estimator used (``'H1'`` or ``'H2'``).
    """

    frequencies: NDArray[np.float64]
    response: NDArray[np.complex128]
    magnitude_db: NDArray[np.float64]
    phase: NDArray[np.float64]
    coherence: NDArray[np.float64]
    estimator: str

    def __post_init__(self) -> None:
        """Reject an estimate whose curves do not share one frequency axis.

        The five fields are one Welch frequency axis and the four quantities
        read off it, so the Bode magnitude is plotted against ``frequencies``
        point for point and the coherence panel beneath it is the quality
        flag of those same points. The three panels cut their
        positive-frequency mask from ``frequencies``, so a wrong length in a
        curve they draw surfaces as numpy's complaint about a boolean index
        and two axis sizes, which names neither the field nor the result.
        What no panel draws says nothing at all: ``response`` is read
        positionally at an index cut from ``frequencies`` and hands back
        another frequency's complex gain, and with an existing ``ax`` only the
        magnitude is drawn, leaving the phase and the coherence just as quiet.
        An extra axis passes every count, so the ranks are pinned too: a
        two-dimensional ``magnitude_db`` draws a second identical curve on the
        magnitude panel.

        :raises ValueError: if the five curves do not share one length, or if
            one of them carries more than one axis.
        """
        require_ranks(
            self,
            frequencies=1,
            response=1,
            magnitude_db=1,
            phase=1,
            coherence=1,
        )
        require_same_length(
            self,
            "frequencies",
            "response",
            "magnitude_db",
            "phase",
            "coherence",
            axis="frequency",
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the Bode magnitude/phase and the coherence.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.electroacoustics import plot_frequency_response

        check_language(language)
        return plot_frequency_response(self, ax=ax, language=language, **kwargs)


def transfer_function(
    x: Signal | NDArray[np.float64] | list[float],
    y: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    estimator: Literal["H1", "H2"] = "H1",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
) -> FrequencyResponseResult:
    r"""Estimate the frequency response from input ``x`` to output ``y``.

    :math:`H_1 = G_{xy} / G_{xx}` (the default; unbiased for output
    noise) or :math:`H_2 = G_{yy} / G_{yx}` (unbiased for input noise),
    from Welch-averaged Hann segments. The ordinary coherence
    :math:`\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} G_{yy})` is
    returned alongside as a data-quality indicator.

    :param x: Input (reference) signal, 1-D. Accepts a
        :class:`phonometry.io.Signal`, whose calibration is applied to the
        samples. The response is a ratio of the two records, so with factors
        ``c_x`` on ``x`` and ``c_y`` on ``y`` it comes out as
        ``(c_y / c_x) H``: a factor the two share cancels, calibrating only
        ``y`` multiplies the response by ``c_y`` and calibrating only ``x``
        divides it by ``c_x``. That is the right answer for a pair that is
        genuinely in two different units.
    :param y: Output (response) signal, 1-D, same length as ``x``. Same
        treatment as ``x``.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param estimator: ``'H1'`` (default) or ``'H2'``.
    :param nperseg: Welch segment length; ``None`` picks a length targeting
        about 4 Hz resolution (as in :mod:`~phonometry.emission.intensity`).
    :param overlap: Segment overlap fraction (default 0.5).
    :return: A :class:`FrequencyResponseResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    fs = resolve_pair_fs(x, y, fs)
    xa, ya = _validate_pair(x, y, "transfer_function")
    fs_v = _positive(fs, "fs")
    if estimator not in ("H1", "H2"):
        msg = "'estimator' must be 'H1' or 'H2'."
        raise ValueError(msg)
    seg, ovl = _welch_params(xa.size, fs_v, nperseg, overlap)

    freqs, gxy, gxx, gyy = _spectra(xa, ya, fs_v, seg, ovl)
    if estimator == "H1":
        response = np.divide(gxy, gxx, out=np.zeros_like(gxy), where=gxx > 0.0)
    else:
        # H2 = Gyy / Gyx = Gyy / conj(Gxy).
        gyx = np.conj(gxy)
        response = np.divide(
            gyy.astype(np.complex128),
            gyx,
            out=np.zeros_like(gxy),
            # |gyx|^2 > 0 guards division by a zero denominator without a
            # float equality check or the sqrt in np.abs.
            where=gyx.real**2 + gyx.imag**2 > 0.0,
        )
    denom = gxx * gyy
    coh = np.divide(np.abs(gxy) ** 2, denom, out=np.zeros_like(gxx), where=denom > 0.0)
    coh = np.clip(coh, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        magnitude_db = 20.0 * np.log10(np.abs(response))
    magnitude_db[~np.isfinite(magnitude_db)] = -np.inf
    phase = np.unwrap(np.angle(response))
    return FrequencyResponseResult(
        frequencies=freqs,
        response=response,
        magnitude_db=magnitude_db,
        phase=phase,
        coherence=coh,
        estimator=estimator,
    )


def coherence(
    x: Signal | NDArray[np.float64] | list[float],
    y: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Ordinary coherence :math:`\gamma^2(f)` between ``x`` and ``y``
    (Bendat & Piersol).

    :math:`\gamma^2 = \lvert G_{xy} \rvert^2 / (G_{xx} G_{yy}) \in
    [0, 1]`: unity for a noiseless linear path and
    :math:`\mathrm{SNR}/(1+\mathrm{SNR})` with additive output noise.
    Averaging over several segments is required for a meaningful estimate
    (a single segment gives :math:`\gamma^2 \equiv 1`).

    :param x: First signal, 1-D. Accepts a :class:`phonometry.io.Signal`,
        whose calibration is applied to the samples. The coherence is a
        normalised ratio, so it is unchanged whichever of the two is
        calibrated and by how much.
    :param y: Second signal, 1-D, same length as ``x``. Same treatment as
        ``x``.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param nperseg: Welch segment length; ``None`` picks a default.
    :param overlap: Segment overlap fraction (default 0.5).
    :return: ``(frequencies, gamma_squared)``.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    fs = resolve_pair_fs(x, y, fs)
    xa, ya = _validate_pair(x, y, "coherence")
    fs_v = _positive(fs, "fs")
    seg, ovl = _welch_params(xa.size, fs_v, nperseg, overlap)
    freqs, gxy, gxx, gyy = _spectra(xa, ya, fs_v, seg, ovl)
    denom = gxx * gyy
    coh = np.divide(np.abs(gxy) ** 2, denom, out=np.zeros_like(gxx), where=denom > 0.0)
    return freqs, np.clip(coh, 0.0, 1.0)
