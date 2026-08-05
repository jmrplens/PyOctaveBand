#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Measurement uncertainty: the worked examples of the GUM and its Supplement 1.

The uncertainty machinery every other module eventually leans on, pinned
to the source that defines it: the additive and Welch-Satterthwaite
examples, the end-gauge calibration of Annex H.1 with its per-input
contributions and effective degrees of freedom, the repeated-observation
example of Annex H.2, and the Monte Carlo coverage intervals that
Supplement 1 Tables 2 and 3 give for the same problems.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Measurement uncertainty - ISO/IEC Guide 98-3 (GUM) and Supplement 1.
# The additive model y = x1+x2+x3+x4 with u(xi)=1 has uc = 2.0 (Suppl 1, 9.2);
# the coverage factor at p=0.99 with 16 degrees of freedom is 2.92 (GUM Annex
# H.1 / Table G.2); equal contributions each with 10 degrees of freedom give a
# Welch-Satterthwaite effective dof of 40 (Annex G.4).
# ---------------------------------------------------------------------------
GUM_ADDITIVE_UC = 2.0  # combined standard uncertainty, additive model
GUM_COVERAGE_K99_16 = 2.92  # coverage factor t at p=0.99, v=16
GUM_WELCH_VEFF = 40.0  # Welch-Satterthwaite effective degrees of freedom

# GUM Annex H.1 end-gauge calibration, end to end: model
# l = lS + d - lS*(dalpha*theta + alphaS*dtheta) with the H.1.3 inputs
# (value, u, dof): lS = 50 000 623 nm (25, 18); d = 215 nm (9.7, 25.6);
# alphaS = 11.5e-6 /degC (1.2e-6, inf); theta = -0.1 degC (0.41, inf);
# dalpha = 0 (0.58e-6, 50); dtheta = 0 (0.029, 2). Published results
# (H.1.4-H.1.6): l = 50 000 838 nm; uc = 32 nm (unrounded 31.71);
# contributions (25, 9.7, 0, 0, 2.9, 16.7) nm -- alphaS and theta are
# genuinely flat directions at the estimates; veff = 16 (truncated from
# 16.66, G.4.2); U99 = 93 nm at k(0.99, 16) = 2.92 (interpolation at the
# untruncated veff, permitted by G.4.2 NOTE 1, gives 92.1 nm).
GUM_H1_INPUTS = [
    # (value, standard uncertainty, dof)
    (50_000_623.0, 25.0, 18.0),        # lS, nm
    (215.0, 9.7, 25.6),                # d, nm
    (11.5e-6, 1.2e-6, math.inf),       # alphaS, 1/degC
    (-0.1, 0.41, math.inf),            # theta, degC
    (0.0, 0.58e-6, 50.0),              # dalpha, 1/degC
    (0.0, 0.029, 2.0),                 # dtheta, degC
]
GUM_H1_VALUE = 50_000_838.0            # nm
GUM_H1_UC = 31.71                      # nm (printed 32)
GUM_H1_CONTRIBUTIONS = [25.0, 9.7, 0.0, 0.0, 2.9, 16.7]
GUM_H1_VEFF = 16.66                    # (printed truncated to 16)
GUM_H1_U99 = 92.1                      # nm at the untruncated veff (printed 93)

# GUM Annex H.2 simultaneous resistance/reactance measurement: the only
# published numeric oracle of the correlated Equation (16) path. Five
# simultaneous observation sets of (V / V, I / mA, phi / rad) from Table H.2;
# their means, standard deviations of the means and sample correlation
# coefficients (r(V,I) = -0.36, r(V,phi) = 0.86, r(I,phi) = -0.65 after
# 2-decimal print rounding) feed R = (V/I) cos phi, X = (V/I) sin phi,
# Z = V/I. Published results (Table H.3): R = 127.732 ohm, uc = 0.071;
# X = 219.847 ohm, uc = 0.295; Z = 254.260 ohm, uc = 0.236. The uc reproduce
# with the correlations computed from the observations; the 2-decimal printed
# r values give uc(R) = 0.070 (their rounding).
GUM_H2_OBSERVATIONS = [
    (5.007, 19.663, 1.0456),
    (4.994, 19.639, 1.0438),
    (5.005, 19.640, 1.0468),
    (4.990, 19.685, 1.0428),
    (4.999, 19.678, 1.0433),
]
GUM_H2_RESULTS = {                     # measurand: (value / ohm, uc / ohm)
    "R": (127.732, 0.071),
    "X": (219.847, 0.295),
    "Z": (254.260, 0.236),
}

# GUM Supplement 1 clause 9.2 additive model Y = X1+X2+X3+X4: the 95 %
# probabilistically symmetric coverage intervals. Table 2 (standard Gaussian
# inputs): +/-3.92 (analytic; GUF identical). Table 3 (rectangular inputs of
# unit standard deviation): u(y) = 2.00 and +/-3.88, analytically
# 2*sqrt(3)*(2 - (3/5)^(1/4)) = 3.8807 (Annex E).
GUMS1_TABLE2_INTERVAL_95 = 3.92
GUMS1_TABLE3_INTERVAL_95 = 3.88
GUMS1_TABLE3_U = 2.00
