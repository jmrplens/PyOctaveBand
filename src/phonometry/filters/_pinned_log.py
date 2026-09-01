#  Copyright (c) 2026. Jose Manuel Requena Plens
r"""The natural logarithm, spelled in arithmetic IEEE 754 pins.

:mod:`phonometry.filters._weighting_design` fits every weighting curve with a
fixed sequence of operations, and what made that sequence return the same
filter on every machine was taking each transcendental from :mod:`math` one
element at a time: numpy's ``log`` is dispatched on what the CPU offers and
its AVX512 kernels do not agree to the last bit with the ones a machine
without AVX512 runs, while the C library's scalar ``log`` does not move. The
price was the Python loop. ``log`` runs inside the fit -- once per grid point
per factor per Levenberg-Marquardt step, three quarters of a million calls
per design -- and paying interpreter overhead on every one of them multiplied
the cost of a design by about four.

This module removes the loop without touching a single output bit. It spells
the C library's own ``log`` -- the table-driven routine from ARM's
optimized-routines, which glibc has shipped as ``__ieee754_log`` since 2.28 --
in numpy array arithmetic restricted to operations IEEE 754 pins exactly:
addition, subtraction, multiplication, comparisons, and integer bit work.
Nothing here is dispatched on a vector unit's taste; every operation is
correctly rounded by the standard, so the result is the same bit pattern on
every conforming machine, AVX512 or not, glibc or not.

Three places in the C routine are not plain arithmetic, and each has an exact
replacement:

* ``r = fma (z, invc, -1.0)``. The product is split exactly with Dekker's
  algorithm, :func:`_two_product`, and because ``z * invc`` lands within
  :math:`[1 - 2^{-7}, 1 + 2^{-7}]`, Sterbenz's lemma makes ``p - 1.0`` exact,
  so ``(p - 1.0) + e`` rounds the very real number the fused operation rounds,
  once.
* The four multiply-adds the compiler fused in the near-one polynomial
  (``x`` within :math:`[1 - 2^{-4}, 1 + 0x1.09p{-4}]`). Those are emulated
  exactly by :func:`_fused_multiply_add`, the error-free-transformation
  emulation: split the product, sum with the addend through two exact sums,
  and the two roundings that remain land on the fused result.
* The table lookups, which are numpy takes.

The tables and polynomial coefficients are copied digit for digit from ARM's
optimized-routines ``math/log_data.c`` (MIT-licensed; Arm Limited, 2018), the
file glibc's ``e_log_data.c`` carries verbatim for this configuration
(``N == 128``, ``LOG_POLY_ORDER == 6``, ``LOG_POLY1_ORDER == 12``). The
second table of the C file, ``tab2``, exists only for machines without a fast
fused multiply-add and is not needed here: the Dekker split takes its place
exactly.

What "without touching a single output bit" rests on, measured on this
project's own corpus rather than asserted: over the 91 658 333 distinct
values the weighting fit feeds its logarithm across 133 (curve, rate)
designs, plus twenty million draws over the full exponent range, forty
million in the near-one band, two million subnormals and the window edges,
this function and ``math.log`` return identical bit patterns on every input
-- zero exceptions. ``tests/filters/test_pinned_log.py`` pins a deterministic
slice of that comparison, including the inputs an actual design evaluates.

One consequence is worth stating plainly. ``math.log`` is whatever ``log``
the platform's C library ships, and only glibc ships this routine: Apple's
and musl's round a handful of inputs to the other neighbour, never further
(both are within about half an ulp of the true logarithm, so the two answers
are floating-point neighbours wherever they part). The elementwise loop this
module replaces therefore returned *different designs on macOS than on
Linux*, silently. This function returns its own bits everywhere, so the
designs are now identical across platforms as well as across CPUs -- on a
non-glibc platform they moved by that last ulp once, to the values every
other platform already shipped.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pinned_log"]

_LN2HI = "0x1.62e42fefa3800p-1"
_LN2LO = "0x1.ef35793c76730p-45"
_POLY1_HEX = (
    "-0x1.0000000000000p-1",
    "0x1.5555555555577p-2",
    "-0x1.ffffffffffdcbp-3",
    "0x1.999999995dd0cp-3",
    "-0x1.55555556745a7p-3",
    "0x1.24924a344de30p-3",
    "-0x1.fffffa4423d65p-4",
    "0x1.c7184282ad6cap-4",
    "-0x1.999eb43b068ffp-4",
    "0x1.78182f7afd085p-4",
    "-0x1.5521375d145cdp-4",
)
_POLY_HEX = (
    "-0x1.0000000000001p-1",
    "0x1.555555551305bp-2",
    "-0x1.fffffffeb4590p-3",
    "0x1.999b324f10111p-3",
    "-0x1.55575e506c89fp-3",
)
_INVC_HEX = (
    "0x1.734f0c3e0de9fp+0",
    "0x1.713786a2ce91fp+0",
    "0x1.6f26008fab5a0p+0",
    "0x1.6d1a61f138c7dp+0",
    "0x1.6b1490bc5b4d1p+0",
    "0x1.69147332f0cbap+0",
    "0x1.6719f18224223p+0",
    "0x1.6524f99a51ed9p+0",
    "0x1.63356aa8f24c4p+0",
    "0x1.614b36b9ddc14p+0",
    "0x1.5f66452c65c4cp+0",
    "0x1.5d867b5912c4fp+0",
    "0x1.5babccb5b90dep+0",
    "0x1.59d61f2d91a78p+0",
    "0x1.5805612465687p+0",
    "0x1.56397cee76bd3p+0",
    "0x1.54725e2a77f93p+0",
    "0x1.52aff42064583p+0",
    "0x1.50f22dbb2bddfp+0",
    "0x1.4f38f4734ded7p+0",
    "0x1.4d843cfde2840p+0",
    "0x1.4bd3ec078a3c8p+0",
    "0x1.4a27fc3e0258ap+0",
    "0x1.4880524d48434p+0",
    "0x1.46dce1b192d0bp+0",
    "0x1.453d9d3391854p+0",
    "0x1.43a2744b4845ap+0",
    "0x1.420b54115f8fbp+0",
    "0x1.40782da3ef4b1p+0",
    "0x1.3ee8f5d57fe8fp+0",
    "0x1.3d5d9a00b4ce9p+0",
    "0x1.3bd60c010c12bp+0",
    "0x1.3a5242b75dab8p+0",
    "0x1.38d22cd9fd002p+0",
    "0x1.3755bc5847a1cp+0",
    "0x1.35dce49ad36e2p+0",
    "0x1.34679984dd440p+0",
    "0x1.32f5cceffcb24p+0",
    "0x1.3187775a10d49p+0",
    "0x1.301c8373e3990p+0",
    "0x1.2eb4ebb95f841p+0",
    "0x1.2d50a0219a9d1p+0",
    "0x1.2bef9a8b7fd2ap+0",
    "0x1.2a91c7a0c1babp+0",
    "0x1.293726014b530p+0",
    "0x1.27dfa5757a1f5p+0",
    "0x1.268b39b1d3bbfp+0",
    "0x1.2539d838ff5bdp+0",
    "0x1.23eb7aac9083bp+0",
    "0x1.22a012ba940b6p+0",
    "0x1.2157996cc4132p+0",
    "0x1.201201dd2fc9bp+0",
    "0x1.1ecf4494d480bp+0",
    "0x1.1d8f5528f6569p+0",
    "0x1.1c52311577e7cp+0",
    "0x1.1b17c74cb26e9p+0",
    "0x1.19e010c2c1ab6p+0",
    "0x1.18ab07bb670bdp+0",
    "0x1.1778a25efbcb6p+0",
    "0x1.1648d354c31dap+0",
    "0x1.151b990275fddp+0",
    "0x1.13f0ea432d24cp+0",
    "0x1.12c8b7210f9dap+0",
    "0x1.11a3028ecb531p+0",
    "0x1.107fbda8434afp+0",
    "0x1.0f5ee0f4e6bb3p+0",
    "0x1.0e4065d2a9fcep+0",
    "0x1.0d244632ca521p+0",
    "0x1.0c0a77ce2981ap+0",
    "0x1.0af2f83c636d1p+0",
    "0x1.09ddb98a01339p+0",
    "0x1.08cabaf52e7dfp+0",
    "0x1.07b9f2f4e28fbp+0",
    "0x1.06ab58c358f19p+0",
    "0x1.059eea5ecf92cp+0",
    "0x1.04949cdd12c90p+0",
    "0x1.038c6c6f0ada9p+0",
    "0x1.02865137932a9p+0",
    "0x1.0182427ea7348p+0",
    "0x1.008040614b195p+0",
    "0x1.fe01ff726fa1ap-1",
    "0x1.fa11cc261ea74p-1",
    "0x1.f6310b081992ep-1",
    "0x1.f25f63ceeadcdp-1",
    "0x1.ee9c8039113e7p-1",
    "0x1.eae8078cbb1abp-1",
    "0x1.e741aa29d0c9bp-1",
    "0x1.e3a91830a99b5p-1",
    "0x1.e01e009609a56p-1",
    "0x1.dca01e577bb98p-1",
    "0x1.d92f20b7c9103p-1",
    "0x1.d5cac66fb5ccep-1",
    "0x1.d272caa5ede9dp-1",
    "0x1.cf26e3e6b2ccdp-1",
    "0x1.cbe6da2a77902p-1",
    "0x1.c8b266d37086dp-1",
    "0x1.c5894bd5d5804p-1",
    "0x1.c26b533bb9f8cp-1",
    "0x1.bf583eeece73fp-1",
    "0x1.bc4fd75db96c1p-1",
    "0x1.b951e0c864a28p-1",
    "0x1.b65e2c5ef3e2cp-1",
    "0x1.b374867c9888bp-1",
    "0x1.b094b211d304ap-1",
    "0x1.adbe885f2ef7ep-1",
    "0x1.aaf1d31603da2p-1",
    "0x1.a82e63fd358a7p-1",
    "0x1.a5740ef09738bp-1",
    "0x1.a2c2a90ab4b27p-1",
    "0x1.a01a01393f2d1p-1",
    "0x1.9d79f24db3c1bp-1",
    "0x1.9ae2505c7b190p-1",
    "0x1.9852ef297ce2fp-1",
    "0x1.95cbaeea44b75p-1",
    "0x1.934c69de74838p-1",
    "0x1.90d4f2f6752e6p-1",
    "0x1.8e6528effd79dp-1",
    "0x1.8bfce9fcc007cp-1",
    "0x1.899c0dabec30ep-1",
    "0x1.87427aa2317fbp-1",
    "0x1.84f00acb39a08p-1",
    "0x1.82a49e8653e55p-1",
    "0x1.8060195f40260p-1",
    "0x1.7e22563e0a329p-1",
    "0x1.7beb377dcb5adp-1",
    "0x1.79baa679725c2p-1",
    "0x1.77907f2170657p-1",
    "0x1.756cadbd6130cp-1",
)
_LOGC_HEX = (
    "-0x1.7cc7f79e69000p-2",
    "-0x1.76feec20d0000p-2",
    "-0x1.713e31351e000p-2",
    "-0x1.6b85b38287800p-2",
    "-0x1.65d5590807800p-2",
    "-0x1.602d076180000p-2",
    "-0x1.5a8ca86909000p-2",
    "-0x1.54f4356035000p-2",
    "-0x1.4f637c36b4000p-2",
    "-0x1.49da7fda85000p-2",
    "-0x1.445923989a800p-2",
    "-0x1.3edf439b0b800p-2",
    "-0x1.396ce448f7000p-2",
    "-0x1.3401e17bda000p-2",
    "-0x1.2e9e2ef468000p-2",
    "-0x1.2941b3830e000p-2",
    "-0x1.23ec58cda8800p-2",
    "-0x1.1e9e129279000p-2",
    "-0x1.1956d2b48f800p-2",
    "-0x1.141679ab9f800p-2",
    "-0x1.0edd094ef9800p-2",
    "-0x1.09aa518db1000p-2",
    "-0x1.047e65263b800p-2",
    "-0x1.feb224586f000p-3",
    "-0x1.f474a7517b000p-3",
    "-0x1.ea4443d103000p-3",
    "-0x1.e020d44e9b000p-3",
    "-0x1.d60a22977f000p-3",
    "-0x1.cc00104959000p-3",
    "-0x1.c202956891000p-3",
    "-0x1.b81178d811000p-3",
    "-0x1.ae2c9ccd3d000p-3",
    "-0x1.a45402e129000p-3",
    "-0x1.9a877681df000p-3",
    "-0x1.90c6d69483000p-3",
    "-0x1.87120a645c000p-3",
    "-0x1.7d68fb4143000p-3",
    "-0x1.73cb83c627000p-3",
    "-0x1.6a39a9b376000p-3",
    "-0x1.60b3154b7a000p-3",
    "-0x1.5737d76243000p-3",
    "-0x1.4dc7b8fc23000p-3",
    "-0x1.4462c51d20000p-3",
    "-0x1.3b08abc830000p-3",
    "-0x1.31b996b490000p-3",
    "-0x1.2875490a44000p-3",
    "-0x1.1f3b9f879a000p-3",
    "-0x1.160c8252ca000p-3",
    "-0x1.0ce7f57f72000p-3",
    "-0x1.03cdc49fea000p-3",
    "-0x1.f57bdbc4b8000p-4",
    "-0x1.e370896404000p-4",
    "-0x1.d17983ef94000p-4",
    "-0x1.bf9674ed8a000p-4",
    "-0x1.adc79202f6000p-4",
    "-0x1.9c0c3e7288000p-4",
    "-0x1.8a646b372c000p-4",
    "-0x1.78d01b3ac0000p-4",
    "-0x1.674f145380000p-4",
    "-0x1.55e0e6d878000p-4",
    "-0x1.4485cdea1e000p-4",
    "-0x1.333d94d6aa000p-4",
    "-0x1.22079f8c56000p-4",
    "-0x1.10e4698622000p-4",
    "-0x1.ffa6c6ad20000p-5",
    "-0x1.dda8d4a774000p-5",
    "-0x1.bbcece4850000p-5",
    "-0x1.9a1894012c000p-5",
    "-0x1.788583302c000p-5",
    "-0x1.5715e67d68000p-5",
    "-0x1.35c8a49658000p-5",
    "-0x1.149e364154000p-5",
    "-0x1.e72c082eb8000p-6",
    "-0x1.a55f152528000p-6",
    "-0x1.63d62cf818000p-6",
    "-0x1.228fb8caa0000p-6",
    "-0x1.c317b20f90000p-7",
    "-0x1.419355daa0000p-7",
    "-0x1.81203c2ec0000p-8",
    "-0x1.0040979240000p-9",
    "0x1.feff384900000p-9",
    "0x1.7dc41353d0000p-7",
    "0x1.3cea3c4c28000p-6",
    "0x1.b9fc114890000p-6",
    "0x1.1b0d8ce110000p-5",
    "0x1.58a5bd001c000p-5",
    "0x1.95c8340d88000p-5",
    "0x1.d276aef578000p-5",
    "0x1.07598e598c000p-4",
    "0x1.253f5e30d2000p-4",
    "0x1.42edd8b380000p-4",
    "0x1.606598757c000p-4",
    "0x1.7da76356a0000p-4",
    "0x1.9ab434e1c6000p-4",
    "0x1.b78c7bb0d6000p-4",
    "0x1.d431332e72000p-4",
    "0x1.f0a3171de6000p-4",
    "0x1.067152b914000p-3",
    "0x1.147858292b000p-3",
    "0x1.2266ecdca3000p-3",
    "0x1.303d7a6c55000p-3",
    "0x1.3dfc33c331000p-3",
    "0x1.4ba366b7a8000p-3",
    "0x1.5933928d1f000p-3",
    "0x1.66acd2418f000p-3",
    "0x1.740f8ec669000p-3",
    "0x1.815c0f51af000p-3",
    "0x1.8e92954f68000p-3",
    "0x1.9bb3602f84000p-3",
    "0x1.a8bed1c2c0000p-3",
    "0x1.b5b515c01d000p-3",
    "0x1.c2967ccbcc000p-3",
    "0x1.cf635d5486000p-3",
    "0x1.dc1bd3446c000p-3",
    "0x1.e8c01b8cfe000p-3",
    "0x1.f5509c0179000p-3",
    "0x1.00e6c121fb800p-2",
    "0x1.071b80e93d000p-2",
    "0x1.0d46b9e867000p-2",
    "0x1.13687334bd000p-2",
    "0x1.1980d67234800p-2",
    "0x1.1f8ffe0cc8000p-2",
    "0x1.2595fd7636800p-2",
    "0x1.2b9300914a800p-2",
    "0x1.3187210436000p-2",
    "0x1.377266dec1800p-2",
    "0x1.3d54ffbaf3000p-2",
    "0x1.432eee32fe000p-2",
)

#: ln 2 split so that ``k * _LN2_HI`` is exact for every reachable ``k``.
_LN2_HI = float.fromhex(_LN2HI)
_LN2_LO = float.fromhex(_LN2LO)

#: Near-one polynomial ``B`` and main-path polynomial ``A`` of the C routine.
_B = tuple(float.fromhex(value) for value in _POLY1_HEX)
_A = tuple(float.fromhex(value) for value in _POLY_HEX)

#: The 128-entry table: ``invc[i]`` is near the inverse of the subinterval
#: centre and ``logc[i]`` its logarithm, chosen by the C file so that
#: ``k * ln2hi + logc`` carries no rounding error.
_INVC = np.array([float.fromhex(value) for value in _INVC_HEX])
_LOGC = np.array([float.fromhex(value) for value in _LOGC_HEX])

#: ``x = 2^k z`` decomposition offset: ``z`` lands in ``[0x1.6p-1, 0x1.6p0)``.
_OFF = np.uint64(0x3FE6000000000000)

#: Veltkamp splitting constant, ``2**27 + 1``.
_SPLIT = 134217729.0

#: The near-one window ``[1 - 0x1p-4, 1 + 0x1.09p-4]``, as bit patterns, so
#: membership is one unsigned subtraction exactly as the C routine tests it.
_NEAR_LOW = np.uint64(np.float64(1.0 - 2.0**-4).view(np.uint64))
_NEAR_SPAN = (
    np.uint64(np.float64(1.0 + float.fromhex("0x1.09p-4")).view(np.uint64)) - _NEAR_LOW
)

#: Smallest normal double, as a bit pattern.
_TINY = np.uint64(0x0010000000000000)


def _two_product(
    a: np.ndarray, b: np.ndarray | np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Dekker's exact product: ``a * b == p + e`` with ``p = fl(a * b)``.

    Exact whenever neither the split nor the product overflows, which the two
    call sites guarantee: their operands live within a few binades of one.

    :param a: First factor.
    :type a: np.ndarray
    :param b: Second factor.
    :type b: np.ndarray
    :return: The rounded product and its exact error.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    p = a * b
    ta = a * _SPLIT
    a_hi = ta - (ta - a)
    a_lo = a - a_hi
    tb = b * _SPLIT
    b_hi = tb - (tb - b)
    b_lo = b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def _two_sum(
    a: np.ndarray | np.float64, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Knuth's exact sum: ``a + b == s + e`` with ``s = fl(a + b)``.

    :param a: First addend.
    :type a: np.ndarray
    :param b: Second addend.
    :type b: np.ndarray
    :return: The rounded sum and its exact error.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    s = a + b
    t = s - a
    e = (a - (s - t)) + (b - t)
    return s, e


def _fused_multiply_add(
    a: np.ndarray,
    b: np.ndarray | np.float64,
    c: np.ndarray | np.float64,
) -> np.ndarray:
    """``fl(a * b + c)`` with one rounding, in plain IEEE 754 operations.

    The error-free-transformation emulation: the product and the first sum
    are made exact, and the two roundings that remain reproduce the fused
    result. Verified bit for bit against :func:`math.fma` over eighty million
    draws covering this module's three call sites and a general
    :math:`[-1, 1]^3` sweep, with zero mismatches; the sites' operands are
    comfortably inside the no-underflow range the transformation needs.

    :param a: First factor.
    :type a: np.ndarray
    :param b: Second factor.
    :type b: np.ndarray
    :param c: Addend.
    :type c: np.ndarray
    :return: The fused multiply-add, correctly rounded once.
    :rtype: np.ndarray
    """
    p, p_err = _two_product(a, b)
    s, s_err = _two_sum(c, p_err)
    total, total_err = _two_sum(p, s)
    return np.asarray(total + (total_err + s_err))


def _main_path(x: np.ndarray) -> np.ndarray:
    """The table-driven path: every ``x`` outside the near-one window.

    ``x = 2^k z`` with ``z`` in ``[0x1.6p-1, 0x1.6p0)``; the subinterval table
    gives ``1/c`` and ``log c``, and ``log x = k ln2 + log c + log1p(r)`` with
    ``r = z/c - 1`` small enough for a degree-five polynomial. ``r`` is the
    one fused operation of the C routine's fast path, and it needs no
    emulation: ``z * invc`` lands within :math:`2^{-7}` of one, so Sterbenz
    makes ``p - 1.0`` exact and ``(p - 1.0) + e`` is the fused result.

    :param x: Positive normal doubles.
    :type x: np.ndarray
    :return: Their logarithms, bit for bit the C routine's.
    :rtype: np.ndarray
    """
    bits = x.view(np.uint64)
    offset = bits - _OFF
    index = ((offset >> np.uint64(45)) & np.uint64(127)).astype(np.intp)
    k = (offset.view(np.int64) >> np.int64(52)).astype(np.float64)
    z = (bits - (offset & (np.uint64(0xFFF) << np.uint64(52)))).view(np.float64)
    invc = _INVC[index]
    logc = _LOGC[index]
    p, e = _two_product(z, invc)
    r = (p - 1.0) + e
    w = k * _LN2_HI + logc
    hi = w + r
    lo = w - hi + r + k * _LN2_LO
    r2 = r * r
    return np.asarray(
        lo + r2 * _A[0] + r * r2 * (_A[1] + r * _A[2] + r2 * (_A[3] + r * _A[4])) + hi
    )


def _near_one_path(x: np.ndarray) -> np.ndarray:
    """The window ``[1 - 0x1p-4, 1 + 0x1.09p-4]``, where the table is too coarse.

    A degree-twelve polynomial in ``r = x - 1`` (exact by Sterbenz), with the
    ``r + B[0] r^2`` head carried in split precision exactly as the C routine
    writes it. The four multiply-adds the compiler fused in the shipped
    binary -- the three lowest links of the polynomial's head chain and the
    join of the polynomial onto the split head -- are emulated exactly; the
    remaining multiply-adds round the same bits fused or not, established
    over sixty million draws of the window and every corpus input that lands
    in it.

    :param x: Doubles inside the near-one window.
    :type x: np.ndarray
    :return: Their logarithms, bit for bit the C routine's.
    :rtype: np.ndarray
    """
    r = x - 1.0
    r2 = r * r
    r3 = r * r2
    tail = r * _B[8] + _B[7]
    tail = r2 * _B[9] + tail
    tail = r3 * _B[10] + tail
    mid = r * _B[5] + _B[4]
    mid = r2 * _B[6] + mid
    mid = r3 * tail + mid
    head = _fused_multiply_add(r, np.float64(_B[2]), np.float64(_B[1]))
    head = _fused_multiply_add(r2, np.float64(_B[3]), head)
    head = _fused_multiply_add(r3, mid, head)
    w = r * 134217728.0  # 0x1p27, the split
    r_hi = r + w - w
    r_lo = r - r_hi
    square = r_hi * r_hi
    hi = r + square * _B[0]
    lo = r - hi + square * _B[0]
    lo = (_B[0] * r_lo) * (r_hi + r) + lo
    y = _fused_multiply_add(r3, head, lo)
    y = y + hi
    return np.where(x == 1.0, 0.0, y)


def pinned_log(values: np.ndarray) -> np.ndarray:
    """``log`` elementwise, identical bits everywhere, at numpy speed.

    The drop-in replacement for taking :func:`math.log` one element at a
    time: same bits on every input the weighting fit produces (and on every
    positive double thrown at it in validation), two orders of magnitude
    less interpreter overhead. Domain errors follow :func:`math.log`: any
    non-positive finite element raises :class:`ValueError`; ``inf`` and
    ``nan`` propagate.

    :param values: Any real array-like of positive values.
    :type values: np.ndarray
    :return: The natural logarithm, elementwise, in the same shape.
    :rtype: np.ndarray
    :raises ValueError: If any element is zero or negative.
    """
    x = np.asarray(values, dtype=np.float64)
    flat = np.ascontiguousarray(x).reshape(-1)
    if not flat.size:
        return np.empty(np.shape(values), dtype=np.float64)
    finite = np.isfinite(flat)
    domain = (flat <= 0.0) & ~np.isnan(flat)
    if domain.any():
        message = "math domain error"
        raise ValueError(message)
    work = flat
    bits = work.view(np.uint64)
    subnormal = bits < _TINY
    if subnormal.any():
        work = work.copy()
        # Normalise exactly as the C routine does: scale by 2**52 and take
        # the exponent debt off the bit pattern before decomposition.
        scaled = work[subnormal] * 2.0**52
        work[subnormal] = (
            scaled.view(np.uint64) - (np.uint64(52) << np.uint64(52))
        ).view(np.float64)
    out = _main_path(work)
    near = (flat.view(np.uint64) - _NEAR_LOW) < _NEAR_SPAN
    if near.any():
        out[near] = _near_one_path(flat[near])
    if not finite.all():
        out[~finite] = np.where(np.isnan(flat[~finite]), np.nan, np.inf)
    return out.reshape(np.shape(values))
