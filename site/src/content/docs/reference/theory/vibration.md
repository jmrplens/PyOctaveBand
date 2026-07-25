---
title: "Vibration"
description: "Human vibration weightings, whole-body and hand-arm metrics and the multiple-shock spinal model behind phonometry."
references:
  - type: book
    authors: ["Griffin, M. J."]
    year: 1996
    title: "Handbook of human vibration"
    publisher: "Academic Press"
    url: "https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2"
    note: "ISBN 978-0-12-303041-2. The biodynamic and health-effect evidence behind the ISO 8041-1 weightings, the rms/MTVV/VDV dose measures and the spinal-injury rationale of the multiple-shock model."
  - type: book
    authors: ["Mansfield, N. J."]
    year: 2004
    title: "Human response to vibration"
    publisher: "CRC Press"
    url: "https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390"
    note: "ISBN 978-0-415-28239-0. A compact modern walkthrough of the ISO 2631-1 whole-body and ISO 5349 hand-arm evaluation chains summarised on this page."
  - type: book
    authors: ["Cremer, L.", "Heckl, M.", "Petersson, B. A. T."]
    year: 2005
    title: "Structure-borne sound: Structural vibrations and sound radiation at audio frequencies"
    edition: "3rd ed."
    publisher: "Springer"
    url: "https://link.springer.com/book/10.1007/b137728"
    note: "ISBN 978-3-540-22696-3. The driving-point power W = (1/2) |F|^2 Re{Y} (Eq. 5.23) and the closed-form infinite-structure mobilities and impedances (Table 5.1) of the point-mobility section."
  - type: book
    authors: ["Hopkins, C."]
    year: 2007
    title: "Sound insulation"
    publisher: "Butterworth-Heinemann"
    url: "https://www.routledge.com/Sound-Insulation/Hopkins/p/book/9780750665261"
    note: "ISBN 978-0-7506-6526-1. The bending-plate radiation-efficiency theory (Eqs 2.227-2.230, the Leppington/Maidanik high-frequency limit) behind the radiation-efficiency section."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2011
    title: "Mechanical vibration and shock — Experimental determination of mechanical mobility — Part 1: Basic terms and definitions, and transducer specifications"
    designation: "ISO 7626-1:2011"
    url: "https://www.iso.org/standard/50426.html"
    note: "The measured driving-point mobilities that the closed-form infinite-structure results of the point-mobility section are the theoretical companions of."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2009
    title: "Acoustics — Determination of airborne sound power levels emitted by machinery using vibration measurement — Part 1: Survey method using a fixed radiation factor"
    designation: "ISO/TS 7849-1:2009"
    url: "https://www.iso.org/standard/40537.html"
    note: "The radiation factor that equals the radiation efficiency, closing the sound-power-from-vibration chain of the radiation-efficiency section (survey method, fixed radiation factor)."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2009
    title: "Acoustics — Determination of airborne sound power levels emitted by machinery using vibration measurement — Part 2: Engineering method including determination of the adequate radiation factor"
    designation: "ISO/TS 7849-2:2009"
    url: "https://www.iso.org/standard/40538.html"
    note: "The engineering method of the same sound-power-from-vibration chain, determining the adequate radiation factor from measurement."
---

This page collects the theory behind human vibration: the ISO 8041-1 frequency weightings, the whole-body and hand-arm metrics of ISO 2631-1 and ISO 5349, the action and limit values of Directive 2002/44/EC, and the ISO 2631-5 multiple-shock spinal model. It is part of the [theory reference](/phonometry/reference/theory/).

## Human vibration (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directive 2002/44/EC)

Human response to vibration depends on frequency, axis and body part, so
acceleration is filtered by the frequency weightings of ISO 8041-1:2017 before
any metric. Each weighting is the analog cascade
$H(s) = H_h(s) H_l(s) H_t(s) H_s(s)$ (Formula 5): two-pole Butterworth
band-limiting high-pass and low-pass stages (Formulae 1/2), an
acceleration–velocity transition (Formula 3, carrying the only non-unity gain,
$K = 1.024$ for Wb) and an upward step (Formula 4), with the Table 3 corner
frequencies and Q factors; a corner at infinity collapses its stage to unity
(Table 3 NOTEs). Wk (vertical whole-body) and Wd (horizontal) of
ISO 2631-1, Wm (buildings, ISO 2631-2), Wb (rail, ISO 2631-4), Wc/We/Wj
(seat-back, rotational, head) and Wh (hand-arm, ISO 5349-1) plus Wf (motion
sickness) are all implemented from the exact cascade (the filter is applied
as the exact complex response via FFT, magnitude *and* phase, not a
bilinear-warped digital approximation) and the ISO 8041-1 Annex B design-goal
tables (B.1–B.9) are reproduced to 0.1 %.

The weighted metrics follow ISO 2631-1:1997: running rms with linear or
exponential integration (Eqs. 2/3), **MTVV** as its maximum (Eq. 4), the
fourth-power **VDV** $= (\int a_w^4\, dt)^{1/4}$ in m/s^1.75 (Eq. 5), the crest
factor with the basic method deemed adequate up to 9 (clause 6.2), and the
vibration total value $a_v = \sqrt{\sum_j k_j^2 a_{wj}^2}$ (Eq. 10). Hand-arm
exposure follows ISO 5349-1:2001: $a_{hv}$ (Eq. 1, all $k = 1$), daily
exposure $A(8) = a_{hv} \sqrt{T/T_0}$ with $T_0 = 8$ h (Eq. 2), partial
exposures combined in quadrature (ISO 5349-2:2001, Eqs. 1–3), and the Annex C
vascular-risk model $D_y = 31.8\ A(8)^{-1.06}$ for the years to 10 %
white-finger prevalence. The Directive 2002/44/EC action and limit values are
built in: hand-arm $A(8)$ 2.5/5.0 m/s², whole-body $A(8)$ 0.5/1.15 m/s² or
VDV 9.1/21.0 m/s^1.75 (Article 3). The ISO 5349-2 worked examples are
reproduced (E.2.1: 7.4 m/s² for 2.5 h → $A(8) = 4.1$ m/s²; E.3 forestry,
three tools → 3.6 m/s²), as are the ISO 5349-1 Table C.1 exposure-duration
rows.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting.svg" alt="The whole-body vertical weighting Wk in decibels over 0.4 to 100 Hz: a plateau near -6 dB below 2 Hz, a small +0.5 dB peak near 6 Hz and a roll-off to about -21 dB at 100 Hz" style="width:88%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_dark.svg" alt="The whole-body vertical weighting Wk in decibels over 0.4 to 100 Hz: a plateau near -6 dB below 2 Hz, a small +0.5 dB peak near 6 Hz and a roll-off to about -21 dB at 100 Hz" style="width:88%" loading="lazy">

*The Wk whole-body weighting realized from the ISO 8041-1 cascade.*

### Multiple shocks (ISO 2631-5)

Repeated shocks damage the lumbar spine through peak compression rather than
average energy, so ISO 2631-5:2018 replaces the Wk weighting with the
seat-to-spine transfer function of clause 5.2 (Formula 1: one complex zero and
six complex pole pairs, unity at DC, resonance near 5 Hz,
$|H| \approx 1.54$ at 5 Hz) and accumulates the positive spinal-response peaks with a
sixth-power (Palmgren-Miner) dose (clause 5.3, Formulae 3/4):

$$
D_z = 1.07 \left( \sum_i A_{z,i}^6 \right)^{1/6}, \qquad
D_{zd} = D_z\ (t_d / t_m)^{1/6}.
$$

Annex C converts the daily dose to a compressive stress $S_d = m_z D_{zd}$
($m_z = 0.029/0.025$ MPa per m/s² for the 82 kg male / 64 kg female), tracks
the age-declining ultimate strength $S_u = 6.75 - S_{age}(b + i)$ and forms
the cumulative stress variable $R$ (Formulae C.3/C.4), mapped to an injury
probability by the Table C.1 Weibull law $\Pi = 1 - e^{-(R/\alpha)^\beta}$.
The spinal filter is evaluated analytically in the frequency domain and
validated against the Annex D 256 Hz digital-filter tabulation within the
clause 5.2 tolerance; the Annex C worked example (five 40 m/s² shocks per day
over 20 years) is reproduced: $D_{zd} = 55.97$ m/s², $R = 1.22$,
$\Pi = 0.37$. The Annex A finite-element spinal model (distributed by ISO as
separate software) is out of scope.

See the [Human Vibration guide](/phonometry/guides/human-vibration/) and the
[Multiple-Shock Vibration guide](/phonometry/guides/multiple-shock-vibration/) for usage.

## Point mobilities and radiation efficiency (Cremer 5, Hopkins 2.9)

The vibrational power a point force injects into a structure is
$W = \tfrac12 |F|^2\,\mathrm{Re}\{Y\}$ (Cremer Eq. 5.23), so the driving-point
**mobility** $Y$ (the reciprocal of the impedance) governs how much energy the
structure absorbs. For infinite structures these are closed forms (Cremer
Table 5.1): an infinite thin plate is a pure resistance $Z = 8\sqrt{B'\,m''}$
(real, frequency independent, with $B'$ the bending stiffness per unit width and
$m''$ the mass per unit area), an infinite beam has
$Y = (1-\mathrm{j})/(4 m' c_B)$ (a 45-degree phase, falling as $\omega^{-1/2}$
through the bending wave speed $c_B$), and a longitudinal rod has
$Z = \rho c_L S$. These supply the receiver mobility EN 12354-5 needs when no
measurement exists, and are the theoretical companions of the measured ISO 7626
mobilities. How efficiently a bending plate then radiates the airborne power is
its **radiation efficiency** $\sigma$: below the critical frequency it radiates
weakly (edge and corner modes), and above it $\sigma \to (1 - f_c/f)^{-1/2} \to 1$
(Leppington/Maidanik, Hopkins Eqs 2.227-2.230). Because $\sigma$ is exactly the
radiation factor $\varepsilon$ of ISO 7849, predicting it closes the sound-power-
from-vibration chain without a power measurement, and it drives the resonant
transmission path of the
[panel sound insulation theory](/phonometry/reference/theory/rooms-buildings/).

See the [Predicting Panel Sound Insulation guide](/phonometry/guides/panel-sound-insulation/)
for usage.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility.svg" alt="Normalized receptance, mobility and accelerance magnitudes of a single-degree-of-freedom resonator on a log-log frequency axis, all peaking at the resonance" style="width:82%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility_dark.svg" alt="Normalized receptance, mobility and accelerance magnitudes of a single-degree-of-freedom resonator on a log-log frequency axis, all peaking at the resonance" style="width:82%" loading="lazy">

*Receptance, mobility and accelerance of a one-degree-of-freedom resonator: the same resonance seen through the three kinematic quantities.*
