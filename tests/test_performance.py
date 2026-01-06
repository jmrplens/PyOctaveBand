#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Performance tests to verify coefficient reuse in OctaveFilterBank.
"""

import time

import numpy as np

from pyoctaveband import OctaveFilterBank, octavefilter


def test_filterbank_reuse_performance() -> None:
    """
    Verify that reusing OctaveFilterBank is faster than calling octavefilter multiple times.
    
    **Purpose:**
    The refactored class-based approach allows pre-calculating SOS coefficients.
    Subsequent filtering should be faster as it skips the design phase.

    **Verification:**
    - Perform 10 iterations of filtering using the functional API (which re-designs every time).
    - Perform 10 iterations of filtering using a pre-initialized `OctaveFilterBank`.
    - Measure and compare the total execution time.

    **Expectation:**
    - Class-based filtering (after initialization) must be significantly faster.
    - Total time (init + 10 filters) should also be less than 10 functional calls.
    """
    fs = 48000
    duration = 0.5
    rng = np.random.default_rng(42)
    x = rng.standard_normal(int(fs * duration))
    num_iterations = 10
    
    # 1. Using functional API (re-designs every time)
    start_func = time.time()
    for _ in range(num_iterations):
        octavefilter(x, fs)
    time_func = time.time() - start_func
    
    # 2. Using FilterBank class (designs once)
    start_class_init = time.time()
    bank = OctaveFilterBank(fs)
    time_class_init = time.time() - start_class_init
    
    start_class_filter = time.time()
    for _ in range(num_iterations):
        bank.filter(x)
    time_class_filter = time.time() - start_class_filter
    
    time_class_total = time_class_init + time_class_filter
    
    print(f"\nFunctional API Time: {time_func:.4f}s")
    print(f"Class Init Time: {time_class_init:.4f}s")
    print(f"Class Filter Only Time: {time_class_filter:.4f}s")
    print(f"Class Total Time: {time_class_total:.4f}s")
    
    # The class-based filtering (after init) should be significantly faster
    # than the functional API calls which include init every time.
    assert time_class_filter < time_func
    
    # Even including init, it should be comparable or better for multiple iterations
    assert time_class_total < time_func