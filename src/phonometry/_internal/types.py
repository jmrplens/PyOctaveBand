#  Copyright (c) 2026. Jose Manuel Requena Plens
"""
Shared typing aliases and array/scalar coercion helpers (private).

Seeded by the library audit: several modules re-declared the same
``Real`` alias and hand-copied the "0-d array becomes a float" return
idiom. New code should import from here; existing modules migrate as
they are touched.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: Real-valued float64 array alias shared across the library.
Real = NDArray[np.float64]


def as_float_or_array(x: np.ndarray | float) -> np.ndarray | float:
    """Return a Python ``float`` for a 0-d array or scalar, the array otherwise.

    :param x: The array (or scalar) produced by a vectorised computation.
    :return: ``float(x)`` when *x* is zero-dimensional, else *x* unchanged.
    """
    return float(x) if np.ndim(x) == 0 else x
