#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""Multiple and partial coherence of a multiple-input/single-output system.

When several partially correlated sources drive one response, the ordinary
coherence of each source with the output is misleading: a source that only
*correlates* with the true cause inherits a spurious coherence through it.
Bendat & Piersol, *Random Data: Analysis and Measurement Procedures*
(4th ed., 2010, Chapter 7), resolve this with the multiple-input/output
(MISO) coherence functions, computed here from the Welch cross-spectral
machinery of :mod:`phonometry.signals.spectra`:

* the **ordinary coherence**
  :math:`\gamma^2_{iy} = \lvert G_{iy} \rvert^2 / (G_{ii} G_{yy})`
  (Eq. 7.109) of each input with the output, taken on its own;
* the **multiple coherence**
  :math:`\gamma^2_{y:x} = G_{vv}/G_{yy} = 1 - G_{nn}/G_{yy}` (Eq. 7.35):
  the fraction of the output autospectrum linearly explained by *all* inputs
  jointly, obtained from the input cross-spectral matrix :math:`G_{xx}` and
  the input-output vector :math:`G_{iy}` (matrix form
  :math:`\gamma^2_{y:x} = G_{iy}^{\mathsf{H}} G_{xx}^{-1} G_{iy} / G_{yy}`,
  Eqs. 7.170-7.192). For additive uncorrelated output noise of per-band
  signal-to-noise ratio :math:`\mathrm{SNR} = G_{vv}/G_{nn}` this is exactly
  :math:`\mathrm{SNR}/(1+\mathrm{SNR})`;
* the **partial coherence** :math:`\gamma^2_{iy \cdot (i-1)!} =
  \lvert G_{iy \cdot (i-1)!} \rvert^2 / (G_{ii \cdot (i-1)!} G_{yy})`
  (Eq. 7.87): the coherence of input ``i`` with the output once the linear
  effect of the inputs *before it in the conditioning order* has been
  removed. The 4th-edition definition uses the *total* output
  :math:`G_{yy}` in the denominator (not the conditioned output), which
  makes the partial coherences of the ordered inputs add up to the multiple
  coherence, :math:`\gamma^2_{y:x} = \sum \gamma^2_{iy \cdot (i-1)!}`
  (Eq. 7.116), and reduce exactly to the ordinary coherences when the
  inputs are mutually uncorrelated (Eq. 7.117);
* the **conditioned (residual) spectra** :math:`G_{ij \cdot r!}` computed by
  the Gaussian-elimination recursion :math:`G_{ij \cdot r!} =
  G_{ij \cdot (r-1)!} - G_{rj \cdot (r-1)!} G_{ir \cdot (r-1)!} /
  G_{rr \cdot (r-1)!}` (Eq. 7.94, base case Eq. 7.95), the Schur complement
  that removes the linear effect of the pivot input ``r`` from every
  remaining record;
* the **partial (cumulative) coherent output spectra**
  :math:`G_{v_i} = \lvert L_{iy} \rvert^2 G_{ii \cdot (i-1)!} =
  \gamma^2_{iy \cdot (i-1)!} G_{yy}` (Eq. 7.86): the share of output power
  the ``i``-th ordered input contributes, so that
  :math:`G_{yy} = \sum G_{v_i} + G_{nn}` (Eqs. 7.88-7.89, 7.121). Comparing
  them band by band answers "which source dominates here?".

The random errors follow Bendat & Piersol Section 9.3: conditioning on the
:math:`i-1` preceding inputs costs :math:`i-1` degrees of freedom, so the
``i``-th ordered input carries :math:`n_\mathrm{d} - (i-1)` effective averages
(Eqs. 9.100/9.101) and the ``q``-input multiple coherence carries
:math:`n_\mathrm{d} - (q-1)` (Eqs. 9.98/9.99).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..io._resolve import resolve_fs, resolve_pair_fs, resolve_samples
from ..io._signal import Signal
from .spectra import (
    _noverlap_samples,
    _positive,
    _segment_statistics,
    _validate_scaling,
    _validate_signal,
    _validate_welch_params,
    _welch_autospectrum,
    _welch_cross_spectrum,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

__all__ = [
    "MISOCoherenceResult",
    "miso_coherence",
]

#: Default overlap fraction between Welch segments (matches spectra.py).
_DEFAULT_OVERLAP = 0.5
#: Relative floor below which a conditioned pivot autospectrum is treated as
#: zero power (an empty or perfectly collinear input): its conditioning step
#: is skipped and its coherence contribution is zero.
_PIVOT_FLOOR_REL = 1e-12


# ---------------------------------------------------------------------------
# Cross-spectral matrix assembly (Welch core of spectra.py)
# ---------------------------------------------------------------------------


def _resolve_miso_fs(
    inputs: (
        Signal
        | Sequence[Signal | NDArray[np.float64] | list[float]]
        | NDArray[np.float64]
    ),
    output: Signal | NDArray[np.float64] | list[float],
    fs: float | None,
) -> float:
    """The one rate of a MISO measurement, from whichever side knows it.

    ``inputs`` may be the Signal itself (its channels are the records), or a
    sequence in which some element is one; ``output`` may be one too. Every
    Signal involved has to agree, because a MISO estimate cross-spectra them
    against each other, and two rates would mis-time every phase in the
    result.
    """
    carrier: Signal | NDArray[np.float64] | list[float]
    if isinstance(inputs, (Signal, np.ndarray)):
        carrier = inputs
    else:
        signals = [row for row in inputs if isinstance(row, Signal)]
        # Every one of them, not just the first: two inputs recorded at
        # different rates can hold the same number of samples, so the
        # length check downstream would not notice, and the conditioning
        # would cross-spectrum them on one frequency axis that is right
        # for at most one of them.
        rates = sorted({row.fs for row in signals})
        if len(rates) > 1:
            raise ValueError(
                f"'inputs' holds Signals recorded at different rates ({rates} "
                "Hz); a MISO estimate cross-spectra them against each other, "
                "so resample them to a common rate first"
            )
        carrier = signals[0] if signals else np.asarray(output)
    if isinstance(carrier, Signal) or isinstance(output, Signal):
        return float(
            resolve_pair_fs(carrier, output, fs, names=("inputs", "output"))
        )
    return float(resolve_fs(np.asarray(output), fs, name="output"))


def _validate_inputs(
    inputs: (
        Signal
        | Sequence[Signal | NDArray[np.float64] | list[float]]
        | NDArray[np.float64]
    ),
    output: Signal | NDArray[np.float64] | list[float],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Validate the input records and the output, all one-dimensional.

    Accepts a sequence of 1-D arrays or a 2-D ``(q, n)`` array of inputs.
    """
    rows: list[Signal | NDArray[np.float64] | list[float]]
    if isinstance(inputs, Signal):
        # A multichannel Signal *is* the set of input records: the q channels
        # of one recording are exactly what the model calls x1..xq. Its
        # calibration is applied once, here, because the rows that come out
        # are bare arrays and would otherwise reach the estimate in digital
        # units.
        rows = list(np.atleast_2d(resolve_samples(inputs)))
    elif isinstance(inputs, np.ndarray) and inputs.ndim == 2:
        rows = [np.ascontiguousarray(row, dtype=np.float64) for row in inputs]
    else:
        # A sequence may hold Signals of its own; each element keeps its own
        # calibration, applied by _validate_signal below.
        rows = list(inputs)
    if len(rows) < 2:
        raise ValueError("'inputs' must hold at least two input records.")
    xs = [_validate_signal(x, f"inputs[{i}]") for i, x in enumerate(rows)]
    ya = _validate_signal(output, "output")
    for i, x in enumerate(xs):
        if x.size != ya.size:
            raise ValueError(
                f"'inputs[{i}]' and 'output' must have the same length."
            )
    return xs, ya


def _validate_order(order: Sequence[int] | None, q: int) -> tuple[int, ...]:
    """Validate the conditioning order, a permutation of ``range(q)``."""
    if order is None:
        return tuple(range(q))
    perm = tuple(int(i) for i in order)
    if sorted(perm) != list(range(q)):
        raise ValueError(f"'order' must be a permutation of range({q}).")
    return perm


def _augmented_matrix(
    xs: list[NDArray[np.float64]],
    ya: NDArray[np.float64],
    fs: float,
    seg: int,
    nov: int,
    window: str,
    scaling: str,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Assemble the ``(F, q+1, q+1)`` Hermitian spectral matrix.

    Records ``0..q-1`` are the inputs and record ``q`` is the output, so
    ``M[:, i, j] = Gij``. Only the upper triangle is estimated by Welch; the
    lower triangle is filled by Hermitian symmetry (``Gji = Gij*``) and the
    diagonal autospectra are forced real. Returns ``(frequencies, M)``.
    """
    records = [*xs, ya]
    q1 = len(records)
    freqs, _ = _welch_cross_spectrum(
        records[0], records[0], fs, seg, nov, window, scaling
    )
    mat = np.zeros((freqs.size, q1, q1), dtype=np.complex128)
    for i in range(q1):
        _, gii = _welch_autospectrum(records[i], fs, seg, nov, window, scaling)
        mat[:, i, i] = gii.astype(np.complex128)
        for j in range(i + 1, q1):
            _, gij = _welch_cross_spectrum(
                records[i], records[j], fs, seg, nov, window, scaling
            )
            mat[:, i, j] = gij
            mat[:, j, i] = np.conj(gij)
    return freqs, mat


# ---------------------------------------------------------------------------
# Ordinary coherence and the Gaussian-elimination conditioning (B&P 7.3)
# ---------------------------------------------------------------------------


def _ordinary_coherences(
    mat: NDArray[np.complex128], q: int
) -> NDArray[np.float64]:
    """Ordinary coherence of each input with the output (Eq. 7.109)."""
    gyy = mat[:, q, q].real
    coh = np.zeros((q, mat.shape[0]), dtype=np.float64)
    for i in range(q):
        gii = mat[:, i, i].real
        denom = gii * gyy
        coh[i] = np.divide(
            np.abs(mat[:, i, q]) ** 2,
            denom,
            out=np.zeros_like(gyy),
            where=denom > 0.0,
        )
    return np.asarray(np.clip(coh, 0.0, 1.0), dtype=np.float64)


def _pivot_safe(piv: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Frequency bins whose pivot autospectrum is above the power floor.

    The floor is relative to the largest pivot on the axis
    (``_PIVOT_FLOOR_REL``), so a bin with negligible conditioned power (an
    empty or perfectly collinear input) is flagged out. The *same* mask
    gates both the coherent-output accumulation and the Schur update, so a
    near-singular pivot cannot credit output power that is never removed
    from the residual.
    """
    floor = _PIVOT_FLOOR_REL * float(np.max(piv)) if piv.size else 0.0
    return np.asarray(piv > floor, dtype=np.bool_)


def _schur_eliminate(
    mat: NDArray[np.complex128], r: int, safe: NDArray[np.bool_]
) -> None:
    r"""Remove the linear effect of pivot record ``r`` in place (Eq. 7.94).

    Updates the trailing block (records ``> r``) with the Schur complement
    :math:`G_{ij \cdot r!} = G_{ij \cdot (r-1)!} - G_{ir \cdot (r-1)!}
    G_{rj \cdot (r-1)!} / G_{rr \cdot (r-1)!}`. Only the
    frequency bins flagged by ``safe`` (pivot autospectrum above the power
    floor, see :func:`_pivot_safe`) are updated; the rest are left unchanged
    so their conditioning contribution is exactly null. ``safe`` is computed
    once by the caller and shared with the coherent-output accumulation, so
    the power-decomposition invariant holds bin by bin even for a
    near-singular pivot.
    """
    if not np.any(safe):
        return
    piv = mat[:, r, r].real
    col = mat[:, r + 1:, r]          # Gir·(r-1)!  (i > r)
    row = mat[:, r, r + 1:]          # Grj·(r-1)!  (j > r)
    update = col[:, :, None] * row[:, None, :]
    pv = np.where(safe, piv, 1.0)[:, None, None]
    mask = safe[:, None, None]
    mat[:, r + 1:, r + 1:] -= np.where(mask, update / pv, 0.0)


def _condition(
    mat: NDArray[np.complex128], order: tuple[int, ...]
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    r"""Ordered conditioning of the augmented matrix (Bendat & Piersol 7.3).

    Pivots on the inputs in ``order``, output last. Returns, indexed by the
    *original* input index:

    * ``partial`` - partial coherence :math:`\gamma^2_{iy \cdot (i-1)!}`
      (Eq. 7.87, total :math:`G_{yy}` denominator so the values sum to the
      multiple coherence, Eq. 7.116);
    * ``coherent`` - partial coherent output spectrum :math:`G_{v_i}`
      (Eq. 7.86);

    and, shared, ``noise`` - the residual output autospectrum
    :math:`G_{yy \cdot q!}` after all inputs are removed (Eq. 7.121).
    """
    q = len(order)
    n_freq = mat.shape[0]
    # Permute inputs into the conditioning order; the output stays last.
    perm = [*order, q]
    work = mat[:, perm][:, :, perm].copy()
    gyy = mat[:, q, q].real  # total output autospectrum (unconditioned)

    partial = np.zeros((q, n_freq), dtype=np.float64)
    coherent = np.zeros((q, n_freq), dtype=np.float64)
    for pos in range(q):
        gpp = work[:, pos, pos].real          # Gii·(i-1)!
        gpy = work[:, pos, q]                 # Giy·(i-1)!
        safe = _pivot_safe(gpp)
        num = np.abs(gpy) ** 2
        contrib = np.divide(
            num, gpp, out=np.zeros_like(gpp), where=safe
        )  # |Liy|²·Gii·(i-1)! = Gvᵢ (Eq. 7.86)
        orig = order[pos]
        coherent[orig] = contrib
        partial[orig] = np.divide(
            contrib, gyy, out=np.zeros_like(gyy), where=gyy > 0.0
        )
        _schur_eliminate(work, pos, safe)
    noise = work[:, q, q].real
    clipped = np.asarray(np.clip(partial, 0.0, 1.0), dtype=np.float64)
    return clipped, coherent, noise


# ---------------------------------------------------------------------------
# Statistical errors (Bendat & Piersol Section 9.3)
# ---------------------------------------------------------------------------


def _coherent_output_errors(
    partial: NDArray[np.float64], order: tuple[int, ...], nd: float
) -> NDArray[np.float64]:
    r"""Random error of each partial coherent output spectrum (Eq. 9.100).

    :math:`\varepsilon = (2-\gamma^2)^{1/2} /
    (\lvert \gamma \rvert \sqrt{n_\mathrm{d} - (i-1)})` for the ``i``-th ordered
    input, with :math:`\gamma^2 = \gamma^2_{iy \cdot (i-1)!}` its partial
    coherence. Bins with zero partial coherence, or an effective average
    count :math:`n_\mathrm{d} - (i-1) \le 0`, report ``inf`` (no usable estimate).
    """
    err = np.full_like(partial, np.inf)
    for pos, orig in enumerate(order):
        eff = nd - pos
        gamma2 = partial[orig]
        gamma = np.sqrt(gamma2)
        if eff > 0.0:
            err[orig] = np.divide(
                np.sqrt(2.0 - gamma2),
                gamma * float(np.sqrt(eff)),
                out=np.full_like(gamma, np.inf),
                where=gamma > 0.0,
            )
    return err


def _multiple_coherence_error(
    multiple: NDArray[np.float64], q: int, nd: float
) -> NDArray[np.float64]:
    r"""Random error of the multiple coherence estimate (Eq. 9.98).

    :math:`\varepsilon = \sqrt{2}\,(1-\gamma^2_{y:x}) /
    (\lvert \gamma_{y:x} \rvert \sqrt{n_\mathrm{d} - (q-1)})`.
    """
    eff = nd - (q - 1)
    gamma = np.sqrt(multiple)
    if eff <= 0.0:
        return np.full_like(multiple, np.inf)
    err = np.divide(
        np.sqrt(2.0) * (1.0 - multiple),
        gamma * float(np.sqrt(eff)),
        out=np.full_like(multiple, np.inf),
        where=gamma > 0.0,
    )
    return np.asarray(err, dtype=np.float64)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MISOCoherenceResult:
    r"""Multiple and partial coherence of a MISO system (B&P Chapter 7).

    Every per-input array is indexed by the *original* input index (the order
    in which the records were passed), so ``ordinary_coherence[i]`` and
    ``coherent_output_spectra[i]`` refer to the same physical source; the
    conditioning that produced the partial coherences is recorded in
    :attr:`order`.

    :ivar frequencies: One-sided frequency axis, in Hz.
    :ivar n_inputs: Number of inputs ``q`` (``q >= 2``).
    :ivar order: Conditioning order actually applied, as original input
        indices; ``partial_coherence[order[k]]`` is conditioned on the inputs
        ``order[:k]``.
    :ivar ordinary_coherence: :math:`\gamma^2_{iy}(f) \in [0, 1]` per input
        (Eq. 7.109), shape ``(q, F)``: each input against the output on its
        own.
    :ivar multiple_coherence: :math:`\gamma^2_{y:x}(f) \in [0, 1]`
        (Eq. 7.35): the fraction of output power explained by all inputs
        jointly. Equals the sum of the partial coherences (Eq. 7.116) and
        ``1 - noise_psd/output_psd``.
    :ivar partial_coherence: :math:`\gamma^2_{iy \cdot (i-1)!}(f) \in [0, 1]`
        per input (Eq. 7.87, total-output denominator), shape ``(q, F)``:
        the coherence of the input with the output once the linear effect of
        the inputs preceding it in :attr:`order` is removed.
    :ivar coherent_output_spectra: Partial coherent output spectrum
        :math:`G_{v_i}` per input (Eq. 7.86), shape ``(q, F)``: the output
        power the input contributes, with
        ``sum(coherent_output_spectra) + noise_psd = output_psd``.
    :ivar output_psd: Measured output autospectrum :math:`\hat{G}_{yy}(f)`.
    :ivar noise_psd: Residual (uncorrelated) output spectrum
        :math:`\hat{G}_{nn} = \hat{G}_{yy \cdot q!}` after removing every
        input (Eq. 7.121).
    :ivar multiple_coherence_random_error: Normalized random error of
        :math:`\gamma^2_{y:x}` (Eq. 9.98), using :math:`n_\mathrm{d} - (q-1)`
        effective averages.
    :ivar coherent_output_random_error: Normalized random error of each
        :math:`G_{v_i}` (Eq. 9.100), shape ``(q, F)``, using
        :math:`n_\mathrm{d} - (i-1)` effective averages for the ``i``-th ordered
        input.
    :ivar n_segments: Raw number of (possibly overlapped) segments averaged.
    :ivar n_averages: Effective number of independent averages :math:`n_\mathrm{d}`.
    :ivar resolution_bandwidth: Effective noise bandwidth :math:`B_\mathrm{e}`, in Hz.
    :ivar window: Taper name.
    :ivar nperseg: Segment length, in samples.
    :ivar overlap: Segment overlap fraction.
    :ivar scaling: ``'density'`` or ``'spectrum'``.
    """

    frequencies: NDArray[np.float64]
    n_inputs: int
    order: tuple[int, ...]
    ordinary_coherence: NDArray[np.float64]
    multiple_coherence: NDArray[np.float64]
    partial_coherence: NDArray[np.float64]
    coherent_output_spectra: NDArray[np.float64]
    output_psd: NDArray[np.float64]
    noise_psd: NDArray[np.float64]
    multiple_coherence_random_error: NDArray[np.float64]
    coherent_output_random_error: NDArray[np.float64]
    n_segments: int
    n_averages: float
    resolution_bandwidth: float
    window: str
    nperseg: int
    overlap: float
    scaling: str

    def dominant_input(self) -> NDArray[np.intp]:
        """Index of the input contributing the most output power per bin.

        Returns, for every frequency, the original input index whose partial
        coherent output spectrum :attr:`coherent_output_spectra` is largest -
        the source that dominates that band. Bins where every contribution is
        zero report the first input (index 0).

        :return: Integer array of length ``len(frequencies)``.
        """
        return np.asarray(
            np.argmax(self.coherent_output_spectra, axis=0), dtype=np.intp
        )

    def plot(
        self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any
    ) -> Axes | NDArray[Any]:
        """Plot the per-input coherent output spectra and multiple coherence.

        :param language: Label language, ``"en"`` (default) or ``"es"``.
        """
        from .._i18n import check_language
        from .._plot.signals import plot_miso_coherence

        check_language(language)
        return plot_miso_coherence(self, ax=ax, language=language, **kwargs)


def miso_coherence(
    inputs: (
        Signal
        | Sequence[Signal | NDArray[np.float64] | list[float]]
        | NDArray[np.float64]
    ),
    output: Signal | NDArray[np.float64] | list[float],
    fs: float | None = None,
    *,
    order: Sequence[int] | None = None,
    window: str = "hann",
    nperseg: int | None = None,
    overlap: float = _DEFAULT_OVERLAP,
    scaling: Literal["density", "spectrum"] = "density",
) -> MISOCoherenceResult:
    r"""Multiple and partial coherence of a MISO system (Bendat & Piersol 7).

    Estimates every auto- and cross-spectrum of the ``q`` inputs and the
    output by the shared Welch core of
    :func:`~phonometry.signals.spectra.cross_spectral_density` (Hann taper
    and 50 % overlap by default, no detrending), then:

    * reports the **ordinary coherence** of each input with the output
      (Eq. 7.109);
    * forms the **multiple coherence** :math:`\gamma^2_{y:x}` (Eq. 7.35)
      from the residual output spectrum left by the Gaussian-elimination
      conditioning of Section 7.3;
    * conditions the inputs in ``order`` to get the **partial coherences**
      :math:`\gamma^2_{iy \cdot (i-1)!}` (Eq. 7.87) and the **partial
      coherent output spectra** :math:`G_{v_i}` (Eq. 7.86), which decompose
      the output power source by source
      (:math:`\sum_i G_{v_i} + G_{nn} = G_{yy}`).

    The partial coherences and the coherent-output decomposition depend on
    the conditioning order; the ordinary and multiple coherences do not.
    Absent a physical basis, Bendat & Piersol (Section 7.2.4) recommend
    ordering the inputs by descending ordinary coherence with the output.

    :param inputs: The ``q`` input records (``q >= 2``), a sequence of
        equal-length 1-D arrays or a 2-D ``(q, n)`` array. Accepts a
        :class:`phonometry.io.Signal` whose channels are the records, or a
        sequence with Signals among its elements; their calibration is
        applied to the samples, and Signals recorded at different rates are
        refused rather than arbitrated.
    :param output: The output record, 1-D, same length as the inputs. Accepts a :class:`phonometry.io.Signal`, whose
        calibration is applied to the samples, so the output autospectrum
        and the coherent output spectra come out in Pa²/Hz, or Pa² for
        ``scaling='spectrum'``. Every coherence is a ratio and does not move.
    :param fs: Sample rate, in Hz. Required when both records are bare
        arrays; either may be a :class:`~phonometry.io.Signal` and supply it,
        and two Signals recorded at different rates are refused rather than
        arbitrated.
    :param order: Conditioning order as input indices (default ``0..q-1``).
    :param window: Segment taper (default Hann).
    :param nperseg: Welch segment length; ``None`` picks a default.
    :param overlap: Segment overlap fraction in [0, 1) (default 0.5).
    :param scaling: ``'density'`` (units²/Hz) or ``'spectrum'`` (units²).
    :return: A :class:`MISOCoherenceResult`.
    :raises ValueError: If the inputs or parameters are invalid.
    """
    xs, ya = _validate_inputs(inputs, output)
    q = len(xs)
    fs_v = _positive(_resolve_miso_fs(inputs, output, fs), "fs")
    scaling_v = _validate_scaling(scaling)
    perm = _validate_order(order, q)
    seg, ovl = _validate_welch_params(ya.size, fs_v, nperseg, overlap)
    nov = _noverlap_samples(seg, ovl)

    freqs, mat = _augmented_matrix(xs, ya, fs_v, seg, nov, window, scaling_v)
    k, nd, benbw = _segment_statistics(ya.size, seg, nov, window, fs_v)

    ordinary = _ordinary_coherences(mat, q)
    partial, coherent, noise = _condition(mat, perm)
    gyy = mat[:, q, q].real
    multiple = np.asarray(
        np.clip(
            np.divide(gyy - noise, gyy, out=np.zeros_like(gyy), where=gyy > 0.0),
            0.0,
            1.0,
        ),
        dtype=np.float64,
    )

    return MISOCoherenceResult(
        frequencies=freqs,
        n_inputs=q,
        order=perm,
        ordinary_coherence=ordinary,
        multiple_coherence=multiple,
        partial_coherence=partial,
        coherent_output_spectra=coherent,
        output_psd=gyy,
        noise_psd=noise,
        multiple_coherence_random_error=_multiple_coherence_error(
            multiple, q, nd
        ),
        coherent_output_random_error=_coherent_output_errors(
            partial, perm, nd
        ),
        n_segments=k,
        n_averages=nd,
        resolution_bandwidth=benbw,
        window=window,
        nperseg=seg,
        overlap=ovl,
        scaling=scaling_v,
    )
