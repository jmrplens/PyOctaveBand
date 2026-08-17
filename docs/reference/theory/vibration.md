← [Documentation index](../../README.md)

# Theory: Vibration

This page collects the theory behind human vibration: the ISO 8041-1 frequency weightings, the whole-body and hand-arm metrics of ISO 2631-1 and ISO 5349, the action and limit values of Directive 2002/44/EC, and the ISO 2631-5 multiple-shock spinal model. It is part of the [theory reference](index.md).

## Human vibration (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directive 2002/44/EC)

Human response to vibration depends on frequency, axis and body part, so
acceleration is filtered by the frequency weightings of ISO 8041-1:2017 before
any metric. Each weighting is the analog cascade
$H(s) = H_\mathrm{h}(s) H_\mathrm{l}(s) H_\mathrm{t}(s) H_\mathrm{s}(s)$ (Formula 5): two-pole Butterworth
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

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_family_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_family.svg" alt="All nine human-vibration weightings in decibels from 0.05 Hz to 1.5 kHz on a log frequency axis, colour-grouped into the whole-body weightings Wk, Wd and Wc, the rotational and recumbent weightings We and Wj, the building and rail weightings Wm and Wb, the motion-sickness weighting Wf peaking at 0.17 Hz, and the hand-arm weighting Wh peaking near 11 Hz; three bars above the curves mark the band each part of the family is tabulated over, 0.1 to 0.5 Hz for Wf, 0.5 to 80 Hz for the whole-body weightings and 6.3 to 1250 Hz for Wh" width="94%"></picture>

*The family the one cascade produces, a Table 3 parameter set per curve. `Wk`,
the vertical whole-body weighting quoted most often, is the flat-topped curve
peaking near 6 Hz and 21 dB down at 100 Hz; `Wd` peaks two and a half octaves
below it, because the body is far more compliant horizontally at low frequency;
`Wf` sits an order of magnitude lower in frequency again; and `Wh` extends three
decades above, which is why the three sets are tabulated over the three
different bands drawn above the curves.*

The weighted metrics follow ISO 2631-1:1997: running rms with linear or
exponential integration (Eqs. 2/3), **MTVV** as its maximum (Eq. 4), the
fourth-power **VDV** $= (\int a_\mathrm{w}^4\, dt)^{1/4}$ in m/s^1.75 (Eq. 5), the crest
factor with the basic method deemed adequate up to 9 (clause 6.2), and the
vibration total value $a_\mathrm{v} = \sqrt{\sum_j k_j^2 a_{\mathrm{w}j}^2}$ (Eq. 10). Hand-arm
exposure follows ISO 5349-1:2001: $a_\mathrm{hv}$ (Eq. 1, all $k = 1$), daily
exposure $A(8) = a_\mathrm{hv} \sqrt{T/T_0}$ with $T_0 = 8$ h (Eq. 2), partial
exposures combined in quadrature (ISO 5349-2:2001, Eqs. 1–3), and the Annex C
vascular-risk model $D_\mathrm{y} = 31.8\ A(8)^{-1.06}$ for the years to 10 %
white-finger prevalence. The Directive 2002/44/EC action and limit values are
built in: hand-arm $A(8)$ 2.5/5.0 m/s², whole-body $A(8)$ 0.5/1.15 m/s² or
VDV 9.1/21.0 m/s^1.75 (Article 3). The ISO 5349-2 worked examples are
reproduced (E.2.1: 7.4 m/s² for 2.5 h → $A(8) = 4.1$ m/s²; E.3 forestry,
three tools → 3.6 m/s²), as are the ISO 5349-1 Table C.1 exposure-duration
rows.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting.svg" alt="The whole-body vertical weighting Wk in decibels over 0.4 to 100 Hz: a plateau near -6 dB below 2 Hz, a small +0.5 dB peak near 6 Hz and a roll-off to about -21 dB at 100 Hz" width="88%"></picture>

*The Wk whole-body weighting realized from the ISO 8041-1 cascade.*

### Multiple shocks (ISO 2631-5)

Repeated shocks damage the lumbar spine through peak compression rather than
average energy, so ISO 2631-5:2018 replaces the Wk weighting with the
seat-to-spine transfer function of clause 5.2 (Formula 1: one complex zero and
six complex pole pairs, unity at DC, resonance near 5 Hz,
$|H| \approx 1.54$ at 5 Hz) and accumulates the positive spinal-response peaks with a
sixth-power (Palmgren-Miner) dose (clause 5.3, Formulae 3/4):

$$
D_\mathrm{z} = 1.07 \left( \sum_i A_{\mathrm{z},i}^6 \right)^{1/6}, \qquad
D_\mathrm{zd} = D_\mathrm{z}\ (t_\mathrm{d} / t_\mathrm{m})^{1/6}.
$$

Annex C converts the daily dose to a compressive stress $S_\mathrm{d} = m_\mathrm{z} D_\mathrm{zd}$
($m_\mathrm{z} = 0.029/0.025$ MPa per m/s² for the 82 kg male / 64 kg female), tracks
the age-declining ultimate strength $S_\mathrm{u} = 6.75 - S_\mathrm{age}(b + i)$ and forms
the cumulative stress variable $R$ (Formulae C.3/C.4), mapped to an injury
probability by the Table C.1 Weibull law $\Pi = 1 - e^{-(R/\alpha)^\beta}$.
The spinal filter is evaluated analytically in the frequency domain and
validated against the Annex D 256 Hz digital-filter tabulation within the
clause 5.2 tolerance; the Annex C worked example (five 40 m/s² shocks per day
over 20 years) is reproduced: $D_\mathrm{zd} = 55.97$ m/s², $R = 1.22$,
$\Pi = 0.37$. The Annex A finite-element spinal model (distributed by ISO as
separate software) is out of scope.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multiple_shock_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/multiple_shock.svg" alt="Left: the seat-to-spine transmissibility rising to about 1.6 near a 5 Hz resonance then rolling off to near zero by 80 Hz. Right: the Weibull probability of lumbar injury versus the stress variable R for male and female, with the 10, 50 and 90 percent risk levels and the Annex C male example at R = 1.22, about 37 percent" width="96%"></picture>

*The two objects of the model. Left, the clause 5.2 seat-to-spine
transmissibility: unity at DC, peaking at $|H| \approx 1.54$ near 5 Hz and
rolling off above it, which is why $W_\mathrm{k}$ had to be replaced for shocks. Right,
the Table C.1 Weibull law $\Pi(R)$ with the Annex C worked example marked at
$R = 1.22$, $\Pi = 0.37$ — the risk rises steeply over a narrow band of $R$, so
a dose that doubles does not double the probability.*

See the [Human Vibration guide](../../vibration/human/human-vibration.md) and the
[Multiple-Shock Vibration guide](../../vibration/human/multiple-shock-vibration.md) for usage.

## Point mobilities and radiation efficiency (Cremer 5, Hopkins 2.9)

The vibrational power a point force injects into a structure is
$W = \tfrac12 |F|^2\,\mathrm{Re}\{Y\}$ (Cremer Eq. 5.23), so the driving-point
**mobility** $Y$ (the reciprocal of the impedance) governs how much energy the
structure absorbs. For infinite structures these are closed forms (Cremer
Table 5.1): an infinite thin plate is a pure resistance $Z = 8\sqrt{B'\,m''}$
(real, frequency independent, with $B'$ the bending stiffness per unit width and
$m''$ the mass per unit area), an infinite beam has
$Y = (1-\mathrm{j})/(4 m' c_\mathrm{B})$ (a 45-degree phase, falling as $\omega^{-1/2}$
through the bending wave speed $c_\mathrm{B}$), and a longitudinal rod has
$Z = \rho c_\mathrm{L} S$. These supply the receiver mobility EN 12354-5 needs when no
measurement exists, and are the theoretical companions of the measured ISO 7626
mobilities. How efficiently a bending plate then radiates the airborne power is
its **radiation efficiency** $\sigma$: below the critical frequency it radiates
weakly (edge and corner modes), and above it $\sigma \to (1 - f_\mathrm{c}/f)^{-1/2} \to 1$
(Leppington/Maidanik, Hopkins Eqs 2.227-2.230). Because $\sigma$ is exactly the
radiation factor $\varepsilon$ of ISO 7849, predicting it closes the sound-power-
from-vibration chain without a power measurement, and it drives the resonant
transmission path of the
[panel sound insulation theory](rooms-buildings.md). Both are clean-room
from Cremer, Heckl & Petersson (2005) and Hopkins (2007).

See the [Predicting Panel Sound Insulation guide](../../buildings/design/panel-sound-insulation.md) for
usage.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/infinite_mobilities_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/infinite_mobilities.svg" alt="Two stacked panels against frequency from 0.5 Hz to 2 kHz on a log axis. Top: mobility magnitude for a single-degree-of-freedom resonator, which resonates, against three infinite-structure closed forms — a 140 mm concrete plate, flat at about 2.6 times ten to the minus six metres per newton-second; a 100 by 200 mm steel beam falling as the inverse square root of frequency; and a steel strut in longitudinal motion, flat. Bottom: the phase of each, zero for the plate and the rod, minus 45 degrees for the beam" width="92%"></picture>

*The three closed forms drawn against the finite resonator. The 140 mm concrete
plate is real and frequency-independent at $2.6 \times 10^{-6}$ m/(N·s); the
100 × 200 mm steel beam falls as $f^{-1/2}$ — a factor 63 over the twelve
octaves shown — at a constant −45°; the longitudinal strut is real and flat.
None of them resonates, which is exactly the difference from a finite structure
and the reason the closed forms are band-average substitutes rather than
point-by-point predictions.*

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility.svg" alt="Normalized receptance, mobility and accelerance magnitudes of a single-degree-of-freedom resonator on a log-log frequency axis, all peaking at the resonance" width="82%"></picture>

*Not one of the closed forms above: this is a **finite** one-degree-of-freedom
resonator, receptance, mobility and accelerance being the same resonance seen
through the three kinematic quantities. It is here as the contrast — the
infinite-structure results are frequency-independent or smoothly falling,
while anything finite resonates.*

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mobility_result_lines_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mobility_result_lines.svg" alt="Driving-point mobility magnitude of a single-degree-of-freedom resonator on log-log axes, climbing along the stiffness line below resonance, falling along the mass line above it, and peaking at one over the damping coefficient at the resonance" width="82%"></picture>

*The same point read as a diagnosis: below resonance the magnitude climbs the
**stiffness line** $\omega/k$, above it it falls along the **mass line**
$1/(\omega m)$, and the peak height is set by the damping alone. A real
structure has many such resonances, and the infinite-structure closed forms
above are the average the measured mobility oscillates about, not the value it
takes at a given frequency — which is why they are used with octave or
third-octave inputs and are least trustworthy in the lowest bands of a small
or lightly damped element.*

## References

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Publisher page](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  The biodynamic and health-effect evidence behind the ISO 8041-1 weightings,
  the rms/MTVV/VDV dose measures and the spinal-injury rationale of the
  multiple-shock model.
- Mansfield, N. J. (2004). *Human response to vibration*. CRC Press.
  ISBN 978-0-415-28239-0.
  [Publisher page](https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390).
  A compact modern walkthrough of the ISO 2631-1 whole-body and ISO 5349
  hand-arm evaluation chains summarised on this page.
