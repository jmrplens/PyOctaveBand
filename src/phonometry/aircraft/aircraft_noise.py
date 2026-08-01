#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""
Aircraft noise certification: Effective Perceived Noise Level (ICAO Annex 16).

The EPNL is the noise-certification metric for transport-category aircraft. It
is built from a half-second spectral time history (24 one-third-octave bands,
50 Hz-10 kHz) in five steps, implementing **ICAO Annex 16 Vol. I, Appendix 2**
(the analytic formulation):

* :func:`perceived_noisiness` -- per-band perceived noisiness ``n`` (noys),
  the analytic piecewise noy law with the Table A2-3 constants.
* :func:`perceived_noise_level` --
  :math:`\mathrm{PNL} = 40 + (10/\lg 2) \cdot \lg N` from the total
  noisiness :math:`N = 0.85 \cdot n_{\mathrm{max}} + 0.15 \cdot \sum n`.
* :func:`tone_correction` -- the tone-correction factor ``C`` (the slope /
  "encircling" method) that penalises spectral irregularities.
* :func:`effective_perceived_noise_level` -- the end-to-end metric: per-record
  :math:`\mathrm{PNLT} = \mathrm{PNL} + C`, the maximum ``PNLTM``, the
  10 dB-down integration limits
  and the duration correction, giving
  :math:`\mathrm{EPNL} = \mathrm{PNLTM} + D`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from .._report.metadata import ReportMetadata

#: The 24 one-third-octave band centre frequencies (Hz) used by PNL/EPNL.
NOY_BANDS: NDArray[np.float64] = np.array(
    [
        50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0,
        315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0,
        2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
    ],
    dtype=np.float64,
)

_INF = np.inf
_NAN = np.nan
#: ICAO Annex 16 Vol. I Appendix 2 Table A2-3: analytic noy constants per band.
#: Columns: SPL(a), SPL(b), SPL(c), SPL(d), SPL(e), M(b), M(c), M(d), M(e).
_A2_3: NDArray[np.float64] = np.array(
    [
        [91.0, 64, 52, 49, 55, 0.043478, 0.030103, 0.079520, 0.058098],
        [85.9, 60, 51, 44, 51, 0.040570, 0.030103, 0.068160, 0.058098],
        [87.3, 56, 49, 39, 46, 0.036831, 0.030103, 0.068160, 0.052288],
        [79.0, 53, 47, 34, 42, 0.036831, 0.030103, 0.059640, 0.047534],
        [79.8, 51, 46, 30, 39, 0.035336, 0.030103, 0.053013, 0.043573],
        [76.0, 48, 45, 27, 36, 0.033333, 0.030103, 0.053013, 0.043573],
        [74.0, 46, 43, 24, 33, 0.033333, 0.030103, 0.053013, 0.040221],
        [74.9, 44, 42, 21, 30, 0.032051, 0.030103, 0.053013, 0.037349],
        [94.6, 42, 41, 18, 27, 0.030675, 0.030103, 0.053013, 0.034859],
        [_INF, 40, 40, 16, 25, 0.030103, _NAN, 0.053013, 0.034859],
        [_INF, 40, 40, 16, 25, 0.030103, _NAN, 0.053013, 0.034859],
        [_INF, 40, 40, 16, 25, 0.030103, _NAN, 0.053013, 0.034859],
        [_INF, 40, 40, 16, 25, 0.030103, _NAN, 0.053013, 0.034859],
        [_INF, 40, 40, 16, 25, 0.030103, _NAN, 0.053013, 0.034859],
        [_INF, 38, 38, 15, 23, 0.030103, _NAN, 0.059640, 0.034859],
        [_INF, 34, 34, 12, 21, 0.029960, _NAN, 0.053013, 0.040221],
        [_INF, 32, 32, 9, 18, 0.029960, _NAN, 0.053013, 0.037349],
        [_INF, 30, 30, 5, 15, 0.029960, _NAN, 0.047712, 0.034859],
        [_INF, 29, 29, 4, 14, 0.029960, _NAN, 0.047712, 0.034859],
        [_INF, 29, 29, 5, 14, 0.029960, _NAN, 0.053013, 0.034859],
        [_INF, 30, 30, 6, 15, 0.029960, _NAN, 0.053013, 0.034859],
        [_INF, 31, 31, 10, 17, 0.029960, 0.029960, 0.068160, 0.037349],
        [44.3, 37, 34, 17, 23, 0.042285, 0.029960, 0.079520, 0.037349],
        [50.7, 41, 37, 21, 29, 0.042285, 0.029960, 0.059640, 0.043573],
    ],
    dtype=np.float64,
)

#: Perceived-noise-level scale factor 10/log10(2).
_PNL_FACTOR = 10.0 / np.log10(2.0)


def _validate_spectrum(spl: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    sig = np.asarray(spl, dtype=np.float64)
    if sig.shape != (24,):
        raise ValueError("'spl' must contain the 24 one-third-octave-band levels.")
    if not np.all(np.isfinite(sig)):
        raise ValueError("'spl' must be finite.")
    return sig


def perceived_noisiness(spl: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    r"""Per-band perceived noisiness ``n`` in noys (ICAO Annex 16 App. 2 §4.7).

    The analytic piecewise noy law, using the Table A2-3 constants:

    * :math:`\mathrm{SPL} \ge \mathrm{SPL}(a)`:
      :math:`n = 10^{M(c) \cdot (\mathrm{SPL} - \mathrm{SPL}(c))}`
    * :math:`\mathrm{SPL}(b) \le \mathrm{SPL} < \mathrm{SPL}(a)`:
      :math:`n = 10^{M(b) \cdot (\mathrm{SPL} - \mathrm{SPL}(b))}`
    * :math:`\mathrm{SPL}(e) \le \mathrm{SPL} < \mathrm{SPL}(b)`:
      :math:`n = 0.3 \cdot 10^{M(e) \cdot (\mathrm{SPL} - \mathrm{SPL}(e))}`
    * :math:`\mathrm{SPL}(d) \le \mathrm{SPL} < \mathrm{SPL}(e)`:
      :math:`n = 0.1 \cdot 10^{M(d) \cdot (\mathrm{SPL} - \mathrm{SPL}(d))}`
    * :math:`\mathrm{SPL} < \mathrm{SPL}(d)`: :math:`n = 0` (below the noy
      floor)

    :param spl: The 24 one-third-octave-band sound pressure levels, in dB.
    :return: The per-band perceived noisiness, in noys.
    :raises ValueError: If the spectrum is not 24 finite levels.
    """
    sig = _validate_spectrum(spl)
    spl_a, spl_b, spl_c, spl_d, spl_e = _A2_3[:, 0], _A2_3[:, 1], _A2_3[:, 2], _A2_3[:, 3], _A2_3[:, 4]
    m_b, m_c, m_d, m_e = _A2_3[:, 5], _A2_3[:, 6], _A2_3[:, 7], _A2_3[:, 8]

    # Guard the "not applicable" M(c) (NaN) so it never feeds a live branch:
    # branch (a) is only selected where SPL ≥ SPL(a), which never holds when
    # SPL(a) = inf, so replacing NaN with 0 here is inert.
    m_c_safe = np.where(np.isnan(m_c), 0.0, m_c)

    n_a = 10.0 ** (m_c_safe * (sig - spl_c))
    n_b = 10.0 ** (m_b * (sig - spl_b))
    n_e = 0.3 * 10.0 ** (m_e * (sig - spl_e))
    n_d = 0.1 * 10.0 ** (m_d * (sig - spl_d))

    conditions = [
        sig >= spl_a,
        sig >= spl_b,
        sig >= spl_e,
        sig >= spl_d,
    ]
    choices = [n_a, n_b, n_e, n_d]
    return np.asarray(np.select(conditions, choices, default=0.0), dtype=np.float64)


def perceived_noise_level(spl: NDArray[np.float64] | list[float]) -> float:
    r"""Perceived noise level ``PNL`` (ICAO Annex 16 App. 2 §4.2), in PNdB.

    :math:`N = 0.85 \cdot n_{\mathrm{max}} + 0.15 \cdot \sum n` and
    :math:`\mathrm{PNL} = 40 + (10/\lg 2) \cdot \lg N`. If the total
    noisiness is not positive the PNL is defined as 0.

    :param spl: The 24 one-third-octave-band sound pressure levels, in dB.
    :return: The perceived noise level, in PNdB.
    :raises ValueError: If the spectrum is not 24 finite levels.
    """
    noys = perceived_noisiness(spl)
    total = float(0.85 * np.max(noys) + 0.15 * np.sum(noys))
    if total <= 0.0:
        return 0.0
    return float(40.0 + _PNL_FACTOR * np.log10(total))


#: First band of the tone-correction routine (band 3 = 80 Hz, 0-based index 2).
_TONE_START = 2


def _tone_background(
    spl: NDArray[np.float64] | list[float],
    start: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Tone-correction background level ``SPL''`` and excess ``F`` (App. 2 §4.3).

    Implements Steps 1-8 of the slope ("encircling") method: slopes, the
    encircling of irregular slopes, the adjusted spectrum ``SPL'``, its smoothed
    slopes, the background ``SPL''`` and the tone excess
    :math:`F = \mathrm{SPL} - \mathrm{SPL}''`.
    Bands 1-2 (50, 63 Hz) do not participate; the routine starts at 80 Hz.

    :return: ``(spl_background, excess_f)``, each length 24; entries below the
        start band are zero.
    """
    sig = _validate_spectrum(spl)
    n = 24
    start = _TONE_START if start is None else start

    # Step 1: slopes s(i) = SPL(i) − SPL(i−1), starting at band 4 (index 3).
    s = np.full(n, np.nan)
    for i in range(start + 1, n):
        s[i] = sig[i] - sig[i - 1]

    # Step 2: encircle a slope where |s(i) − s(i−1)| > 5.
    s_enc = np.zeros(n, dtype=bool)
    for i in range(start + 1, n):
        if not np.isnan(s[i]) and not np.isnan(s[i - 1]) and abs(s[i] - s[i - 1]) > 5.0:
            s_enc[i] = True

    # Step 3: mark which SPL value to adjust.
    spl_marked = np.zeros(n, dtype=bool)
    for i in range(start + 1, n):
        if not s_enc[i]:
            continue
        if s[i] > 0.0 and (np.isnan(s[i - 1]) or s[i] > s[i - 1]):
            spl_marked[i] = True
        elif s[i] <= 0.0 and not np.isnan(s[i - 1]) and s[i - 1] > 0.0:
            spl_marked[i - 1] = True

    # Step 4: adjusted spectrum SPL'.
    spl2 = sig.copy()
    for i in range(start, n):
        if spl_marked[i]:
            if i < n - 1:
                spl2[i] = 0.5 * (sig[i - 1] + sig[i + 1])
            else:  # last band: SPL'(24) = SPL(23) + s(23)
                spl2[i] = sig[i - 1] + s[i - 1]

    # Step 5: new slopes s'(i); s'(start) = s'(start+1); imaginary 25th slope.
    sp = np.full(n, np.nan)
    for i in range(start + 1, n):
        sp[i] = spl2[i] - spl2[i - 1]
    sp[start] = sp[start + 1]
    s_imaginary = sp[n - 1]

    # Step 6: 3-point running mean s̄(i), i = start … n−2.
    sbar = np.full(n, np.nan)
    for i in range(start, n - 1):
        third = sp[i + 2] if i + 2 < n else s_imaginary
        sbar[i] = (sp[i] + sp[i + 1] + third) / 3.0

    # Step 7: background SPL''(start) = SPL(start); accumulate the smoothed slopes.
    spl3 = np.full(n, np.nan)
    spl3[start] = sig[start]
    for i in range(start + 1, n):
        spl3[i] = spl3[i - 1] + sbar[i - 1]

    # Step 8: tone excess F = SPL − SPL''.
    excess = np.zeros(n)
    for i in range(start, n):
        excess[i] = sig[i] - spl3[i]

    spl3 = np.where(np.isnan(spl3), 0.0, spl3)
    return np.asarray(spl3, dtype=np.float64), np.asarray(excess, dtype=np.float64)


def _tone_factor(excess: float, frequency: float) -> float:
    """Tone-correction factor ``C`` from the excess ``F`` (App. 2 Table A2-2)."""
    if not np.isfinite(excess) or excess < 1.5:
        return 0.0
    if 500.0 <= frequency <= 5000.0:
        if excess < 3.0:
            return 2.0 * excess / 3.0 - 1.0
        if excess < 20.0:
            return excess / 3.0
        return 20.0 / 3.0
    if excess < 3.0:
        return excess / 3.0 - 0.5
    if excess < 20.0:
        return excess / 6.0
    return 10.0 / 3.0


def tone_correction(spl: NDArray[np.float64] | list[float], *,
                    start_band: int = _TONE_START) -> float:
    """Tone-correction factor ``C`` for a spectrum (ICAO Annex 16 App. 2 §4.3).

    The maximum over bands of the tone-correction factor derived from the tone
    excess ``F`` above the smoothed background, with the 1.5 dB threshold, the
    500 Hz / 5000 Hz frequency split and the 6⅔ dB cap of Table A2-2.

    :param spl: The 24 one-third-octave-band sound pressure levels, in dB.
    :param start_band: First band index of the slope analysis (App. 2 §4.3.1
        Step 1). The default 2 (80 Hz) is the aeroplane procedure; helicopters
        and tilt-rotors use 0 (50 Hz).
    :return: The tone-correction factor ``C``, in dB (0 if no tones qualify).
    :raises ValueError: If the spectrum is not 24 finite levels.
    """
    if not 0 <= int(start_band) <= 3:
        raise ValueError("'start_band' must be between 0 (50 Hz) and 3 (100 Hz).")
    _, excess = _tone_background(spl, start=int(start_band))
    factors = [_tone_factor(float(excess[i]), float(NOY_BANDS[i])) for i in range(24)]
    return float(max(factors))


#: EPNL duration reference time ``T0`` (s) and the level drop for the limits.
_EPNL_REFERENCE_TIME = 10.0
_TEN_DB_DOWN = 10.0


def _ten_db_down_limits(pnlt: NDArray[np.float64], threshold: float) -> tuple[int, int]:
    """The 10 dB-down record indices ``(kF, kL)`` of the integration window.

    ``kF``/``kL`` are the records whose PNLT is nearest to ``PNLTM − 10``. The
    search takes the **first** (from the start) and **last** (from the end)
    records that reach the threshold, then the nearer of the two samples
    bracketing each crossing. Spanning the outermost crossings is deliberate:
    App. 2 §4.5 (and FAR-36 §A36.4.5) require the *largest possible duration*
    for multi-lobed histories, so an interior dip below the threshold must not
    truncate the window. ``PNLTM`` is always ``≥ threshold``, so the search
    never comes up empty.
    """
    n = pnlt.size
    above = np.nonzero(pnlt >= threshold)[0]
    if above.size == 0:  # pragma: no cover - PNLTM >= threshold guarantees a hit
        return 0, n - 1

    first = int(above[0])
    if first > 0 and abs(pnlt[first - 1] - threshold) < abs(pnlt[first] - threshold):
        kf = first - 1
    else:
        kf = first

    last = int(above[-1])
    if last < n - 1 and abs(pnlt[last + 1] - threshold) < abs(pnlt[last] - threshold):
        kl = last + 1
    else:
        kl = last
    return kf, kl


def epnl_from_pnlt(
    pnlt: NDArray[np.float64] | list[float],
    dt: float | NDArray[np.float64] | list[float] = 0.5,
    *,
    reference_time: float = _EPNL_REFERENCE_TIME,
    tone_corrections: NDArray[np.float64] | list[float] | None = None,
) -> tuple[float, float, int, int]:
    r"""EPNL from a tone-corrected perceived-noise-level time history (App. 2 §4.5-4.6).

    .. math::

       \mathrm{EPNL} = 10 \cdot \lg\left( \sum_{k_F..k_L}
       10^{\mathrm{PNLT}(k)/10} \cdot \Delta t(k) \right)
       - 10 \cdot \lg(T_0)

    with the
    10 dB-down integration limits about the maximum ``PNLTM``. The exact
    :math:`-10 \cdot \lg(T_0)` form is used rather than the Annex's rounded
    constant 13 for
    uniform 0.5 s records (difference 0.0103 dB); the ETM Table 4-4 integrated
    reference reproduces the exact form to five decimals.

    When ``tone_corrections`` is given, the **bandsharing adjustment** ``ΔB``
    (App. 2 §4.4.2/4.4.3, ETM GM/AMC A2 4.4.2) is applied: if the tone
    correction at the PNLTM record is below the average of the records within
    one second of it (five records for the uniform 0.5 s cadence), ``ΔB`` is
    that shortfall, added to PNLTM before the 10 dB-down window is found and
    included in the reported EPNL.

    :param pnlt: The tone-corrected perceived noise levels ``PNLT(k)``, in PNdB.
    :param dt: Per-record duration, in s (scalar broadcast or per record).
    :param reference_time: Normalising time ``T0``, in s (default 10).
    :param tone_corrections: Optional tone corrections ``C(k)``, in dB, one per
        ``pnlt`` record; enables the bandsharing adjustment ``ΔB`` described
        above. Default ``None`` (no adjustment).
    :return: ``(epnl, pnltm, kF, kL)`` -- EPNL in EPNdB, the peak PNLTM, and the
        0-based 10 dB-down record indices (inclusive).
    :raises ValueError: If the inputs are invalid.
    """
    p = np.atleast_1d(np.asarray(pnlt, dtype=np.float64))
    if p.ndim != 1 or p.size < 1 or not np.all(np.isfinite(p)):
        raise ValueError("'pnlt' must be a non-empty finite 1-D sequence.")
    dt_raw = np.asarray(dt, dtype=np.float64)
    dt_arr = np.full(p.shape, float(dt_raw.reshape(-1)[0])) if dt_raw.size == 1 else dt_raw
    if dt_arr.shape != p.shape or np.any(dt_arr <= 0.0) or not np.all(np.isfinite(dt_arr)):
        raise ValueError("'dt' must be positive and match 'pnlt' in length.")
    if reference_time <= 0.0 or not np.isfinite(reference_time):
        raise ValueError("'reference_time' must be a positive, finite number.")

    km = int(np.argmax(p))
    delta_b = 0.0
    if tone_corrections is not None:
        c = np.atleast_1d(np.asarray(tone_corrections, dtype=np.float64))
        if c.shape != p.shape or not np.all(np.isfinite(c)):
            raise ValueError("'tone_corrections' must match 'pnlt' in length and be finite.")
        t_mid = np.cumsum(dt_arr) - 0.5 * dt_arr
        window = c[np.abs(t_mid - t_mid[km]) <= 1.0 + 1e-9]
        delta_b = max(0.0, float(np.mean(window)) - float(c[km]))
    pnltm = float(p[km]) + delta_b
    kf, kl = _ten_db_down_limits(p, pnltm - _TEN_DB_DOWN)
    energy = np.sum(10.0 ** (p[kf : kl + 1] / 10.0) * dt_arr[kf : kl + 1])
    epnl = float(10.0 * np.log10(energy) - 10.0 * np.log10(reference_time)) + delta_b
    return epnl, pnltm, kf, kl


@dataclass(frozen=True)
class EPNLResult:
    r"""Effective Perceived Noise Level of an aircraft flyover (ICAO Annex 16).

    :ivar frequencies: The 24 one-third-octave band centre frequencies, in Hz.
    :ivar times: Record times, in s.
    :ivar pnl: Perceived noise level per record, in PNdB.
    :ivar tone_correction: Tone-correction factor per record, in dB.
    :ivar pnlt: Tone-corrected perceived noise level per record, in PNdB.
    :ivar pnltm: Maximum tone-corrected perceived noise level, in PNdB,
        including the bandsharing adjustment ``ΔB`` (App. 2 §4.4.3).
    :ivar bandsharing_adjustment: The bandsharing adjustment ``ΔB``, in dB
        (zero unless the tone correction at PNLTM is suppressed).
    :ivar duration_correction: Duration correction
        :math:`D = \mathrm{EPNL} - \mathrm{PNLTM}`, in dB.
    :ivar epnl: Effective perceived noise level, in EPNdB.
    :ivar band_limits: The 0-based 10 dB-down record indices ``(kF, kL)``.
    """

    frequencies: NDArray[np.float64]
    times: NDArray[np.float64]
    pnl: NDArray[np.float64]
    tone_correction: NDArray[np.float64]
    pnlt: NDArray[np.float64]
    pnltm: float
    bandsharing_adjustment: float
    duration_correction: float
    epnl: float
    band_limits: tuple[int, int]

    def plot(self, ax: Axes | None = None, *, language: str = "en", **kwargs: Any) -> Axes:
        """Plot the PNL and PNLT time histories with PNLTM and the 10 dB-down band."""
        from .._i18n import check_language
        from .._plot.aircraft import plot_epnl

        return plot_epnl(self, ax=ax, language=check_language(language), **kwargs)

    def report(
        self,
        path: str,
        *,
        metadata: ReportMetadata | None = None,
        engine: str = "reportlab",
        verbose: bool = False,
        language: str = "en",
    ) -> str:
        """Render an ICAO Annex 16 EPNL certification fiche to a PDF.

        Writes a one-page aircraft-noise-certification data sheet: the
        standard-basis line (ICAO Annex 16 Vol. I Appendix 2), an optional
        TCDSN-style metadata header (aircraft, manufacturer / type-certificate
        holder, applicant, measurement point), a metrics table of the
        informational intermediate quantities (PNLTM, the duration correction
        ``D``, the 10 dB-down record window and, when non-zero, the bandsharing
        adjustment), the result's own landscape PNLT-versus-time :meth:`plot`,
        the boxed ``EPNL = X EPNdB`` result, a Level | Limit | Margin verdict
        row when a certification limit is supplied, a static reference-conditions
        strip and a footer with the fixed disclaimer.

        :param path: Destination path of the PDF file.
        :param metadata: Optional
            :class:`~phonometry.ReportMetadata`; ``None`` produces a prediction
            fiche (metrics, plot and result only, no verdict). A supplied
            ``requirement`` is read as the certification EPNL limit in EPNdB
            (the EPNL passes at or below it).
        :param engine: Rendering back end; only ``"reportlab"`` is supported.
        :param verbose: Accepted for a uniform signature; it has no effect on
            the single-layout EPNL fiche.
        :param language: Fiche language: ``"en"`` (default, English) or
            ``"es"`` (Spanish, with a comma decimal separator).
        :return: The written ``path`` as a :class:`str`.
        :raises ValueError: If ``engine`` is not ``"reportlab"``.
        :raises ImportError: If reportlab is not installed
            (``pip install phonometry[report]``), or matplotlib is missing for
            the embedded figure (``pip install phonometry[plot]``).
        """
        from .._i18n import check_language

        check_language(language)
        if engine != "reportlab":
            raise ValueError(
                f"Unknown report engine {engine!r}; only 'reportlab' is supported."
            )
        from .._report.annex16_epnl import render_annex16_epnl_report

        return render_annex16_epnl_report(
            self, path, metadata=metadata, verbose=verbose, language=language
        )


#: Tone-correction start band per certification procedure (App. 2 §4.3.1
#: Step 1): aeroplanes start the slope analysis at band 3 (80 Hz, 0-based
#: index 2); helicopters and tilt-rotors at band 1 (50 Hz, index 0).
_PROCEDURE_START_BAND: dict[str, int] = {"aeroplane": _TONE_START, "helicopter": 0}


def effective_perceived_noise_level(
    spectra: NDArray[np.float64] | list[list[float]],
    dt: float | NDArray[np.float64] | list[float] = 0.5,
    *,
    reference_time: float = _EPNL_REFERENCE_TIME,
    procedure: str = "aeroplane",
) -> EPNLResult:
    r"""Effective Perceived Noise Level from a spectral time history (ICAO Annex 16).

    Each record (row) is a 24-band one-third-octave spectrum sampled every
    ``dt`` seconds. The per-record ``PNL`` and tone correction ``C`` give
    :math:`\mathrm{PNLT} = \mathrm{PNL} + C`; the maximum ``PNLTM`` and the
    duration correction over
    the 10 dB-down window give :math:`\mathrm{EPNL} = \mathrm{PNLTM} + D`.

    :param spectra: Spectral time history, shape ``(K, 24)``, in dB.
    :param dt: Per-record duration, in s (scalar or per record, default 0.5).
    :param reference_time: Normalising time ``T0``, in s (default 10).
    :param procedure: Tone-correction procedure of App. 2 §4.3.1 Step 1:
        ``"aeroplane"`` (default) starts the slope analysis at the 80 Hz band
        (band 3), ``"helicopter"`` (helicopters and tilt-rotors) at the 50 Hz
        band (band 1), so rotor tones in the 50-80 Hz bands are not missed.
    :return: An :class:`EPNLResult`.
    :raises ValueError: If the input is not a ``(K, 24)`` finite array, or
        ``procedure`` is not ``"aeroplane"`` or ``"helicopter"``.
    """
    arr = np.asarray(spectra, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 24 or arr.shape[0] < 1:
        raise ValueError("'spectra' must have shape (K, 24).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("'spectra' must be finite.")
    start_band = _PROCEDURE_START_BAND.get(procedure)
    if start_band is None:
        raise ValueError(
            f"Unknown procedure {procedure!r}; use 'aeroplane' (tone "
            "correction from 80 Hz) or 'helicopter' (from 50 Hz, ICAO "
            "Annex 16 Vol I App. 2 4.3.1 Step 1)."
        )

    pnl = np.array([perceived_noise_level(row) for row in arr])
    corr = np.array([tone_correction(row, start_band=start_band) for row in arr])
    pnlt = pnl + corr

    epnl, pnltm, kf, kl = epnl_from_pnlt(
        pnlt, dt, reference_time=reference_time, tone_corrections=corr)

    k = arr.shape[0]
    dt_raw = np.asarray(dt, dtype=np.float64)
    if dt_raw.size == 1:
        times = np.arange(k, dtype=np.float64) * float(dt_raw.reshape(-1)[0])
    else:
        times = np.concatenate([[0.0], np.cumsum(dt_raw)[:-1]])
    return EPNLResult(
        frequencies=NOY_BANDS.copy(),
        times=times,
        pnl=pnl,
        tone_correction=corr,
        pnlt=pnlt,
        pnltm=pnltm,
        bandsharing_adjustment=float(pnltm - float(np.max(pnlt))),
        duration_correction=float(epnl - pnltm),
        epnl=epnl,
        band_limits=(kf, kl),
    )
