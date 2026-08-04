← [Documentation index](../../README.md)

# Levels and weighting

A sound level meter does three things to a calibrated signal, in order: it
**weights it in frequency** to mimic the ear's sensitivity, it **smooths it in
time** with a standardised ballistic, and it **integrates it into a level**.
The pages of this section implement exactly that chain, one page per
stage, following **IEC 61672-1:2013** closely enough that the weightings are
verified against the standard's own tolerance tables in CI.

[Frequency Weighting (A, C, Z)](weighting.md) covers the
first stage. The A-curve tracks hearing sensitivity at moderate levels and
dominates regulation; C is nearly flat and serves peaks and low-frequency
checks; Z is unweighted by definition. The rest of the family lives in
[Special Weightings (G, B, D, AU)](special-weightings.md):
the G-curve of **ISO 7196** extends the idea into infrasound, where
conventional weightings are blind, the historical B and D curves serve legacy
data, and AU rejects ultrasound from an audible-sound reading per IEC 61012.

[Time Weighting](time-weighting.md) covers the second stage:
the exponential Fast (125 ms), Slow (1 s) and Impulse ballistics that decide
how quickly a displayed level follows the sound. phonometry implements the
exact time constants, verified against the toneburst responses of the
standard.

[Integrated and Statistical Levels](levels.md) is the payoff:
the equivalent continuous level Leq and its A-weighted LAeq, the percentile
levels L10/L50/L90 that describe fluctuating noise, LCpeak and SEL, the noise
dose of IEC 61252, plus the octave spectrogram for visualising level against
time and band at once. This is the page where most practical measurements
end, and where the environmental and occupational sections pick up.

Turning those levels into a regulatory verdict is
[Environmental Levels (ISO 1996-1/-2)](../../environment/environmental-levels.md):
the day-evening-night level Lden and the rating levels of **ISO 1996-1** with
their adjustments, and the ISO 1996-2 determination chain of tonal
adjustment, residual-noise correction and the measurement uncertainty budget.

National frameworks build their own index on top of that chain, and
[Spanish Noise Regulation (RD 1367/2007)](../../environment/spanish-noise-regulation.md)
implements the Spanish one: the corrected level LKeq with its tonal,
low-frequency and impulsive corrections, the evaluation periods split into
noise phases, and the limit tables an activity is judged against.

## Pages in this section

- [Frequency Weighting (A, C, Z)](weighting.md): the
  IEC 61672-1 A/C/Z curves, the high-frequency accuracy mode and the class
  verification.
- [Special Weightings (G, B, D, AU)](special-weightings.md):
  the ISO 7196 infrasound G-weighting, the historical B and D curves and AU
  per IEC 61012.
- [Time Weighting](time-weighting.md): Fast, Slow and
  Impulse exponential ballistics.
- [Integrated and Statistical Levels](levels.md): Leq and
  LAeq, percentile levels, LCpeak/SEL, noise dose and octave spectrograms.
- [Environmental Levels (ISO 1996-1/-2)](../../environment/environmental-levels.md):
  Lden, Ldn and rating levels, tonal adjustment, residual noise and
  uncertainty.
- [Spanish Noise Regulation (RD 1367/2007)](../../environment/spanish-noise-regulation.md):
  the corrected level LKeq, the Kt/Kf/Ki corrections, the evaluation periods
  and noise phases, and the limit tables.
