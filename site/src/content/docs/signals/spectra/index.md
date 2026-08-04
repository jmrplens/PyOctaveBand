---
title: "Signals and spectra"
description: "Frequency- and time-domain signal analysis in phonometry: calibrated Welch spectral estimates with their statistical quality, cepstral analysis with echo detection and liftering, the envelope spectrum, and correlation, time-delay estimation and the Hilbert envelope."
---

Band levels answer *how much*; this section answers *what is in the signal*.
Where the rest of the core works in fractional octave bands, these pages work
with the fine-grained estimators of classical signal analysis: spectral
densities, correlation functions, delays and envelopes. They share one
discipline, taken from Bendat & Piersol: every estimate is **calibrated** (the
same dB SPL / dBFS reference frames as the rest of the library) and carries
its **statistical quality**, so a spectrum is not just a curve but a curve
with a confidence interval.

[Calibrated spectral analysis](/phonometry/signals/spectra/spectral-analysis/) is the
frequency-domain half. The Welch power and cross-spectral density estimators
report their effective number of averages, normalized random errors and
chi-square confidence intervals; the coherent output spectrum splits a
measured output into the part explained by an input and the part that is
noise, with a spectral signal-to-noise ratio; fractional-octave smoothing
bridges back to the banded world; and the colored-noise generators synthesize
white, pink, red, blue and violet test signals with an exact power-law slope.
[Multiple and partial coherence](/phonometry/signals/spectra/miso-coherence/) carries
that same cross-spectral machinery to several correlated sources at once: from
multiple inputs and one output it separates the coherence a source
genuinely contributes from the part it merely shares with another, and its
partial coherent output spectra say which source dominates each band.

[Time-frequency analysis](/phonometry/signals/spectra/time-frequency/) is the view in
between: the calibrated STFT spectrogram shows what happens *when* - a
passing siren, an impact, a run-up - with every cell reading an absolute
level in the same scaling as the Welch estimators, and the zoom FFT computes
the spectrum of a narrow band on an arbitrarily fine grid to separate tones
closer than a practical FFT bin.
[Cepstrum, echoes and the envelope spectrum](/phonometry/signals/spectra/cepstrum-echoes/)
works on the *shape* of the spectrum. The power, real and complex cepstrum
collapse periodic spectral ripple onto quefrency spikes, echo detection reads
a reflection's delay and coefficient off the cepstral peak, liftering splits
a log spectrum into smooth envelope and fine structure, and the envelope
spectrum turns amplitude modulations into discrete lines at the modulation
frequency.
[Time synchronous averaging](/phonometry/signals/spectra/synchronous-averaging/)
extracts a repetitive waveform of known period from asynchronous noise by
ensemble-averaging successive periods: the residual noise falls as the square
root of the number of averages, and choosing that number to place a comb node
on an interfering order rejects it far better than the habitual power of two.

[Correlation, time delay and envelope](/phonometry/signals/spectra/correlation-delay/)
is the time-domain half. Auto- and cross-correlation come with the
Bendat & Piersol normalizations and random errors; time-delay estimation
offers the direct correlator, the cross-spectrum phase slope and the
Knapp & Carter generalized cross-correlation weightings (Roth, SCOT, PHAT,
maximum likelihood); impulse responses can be delayed and aligned with
sub-sample precision; and the Hilbert transform yields the envelope with
instantaneous phase and frequency.

[Test signals and sample-rate tools](/phonometry/signals/spectra/test-signals/) is
the toolbox the other two lean on: tone bursts with the exact gating of
IEC 60268-1 (zero-crossing start, integral full periods, repetitive trains),
polyphase resampling behind an explicit anti-alias specification whose
designed filter travels with the result, and band-limited fractional delay
with a linear or circular boundary, sharing its kernel with the sub-sample
alignment of impulse responses.

[System measurement](/phonometry/signals/spectra/system-measurement/) turns the
toolbox toward measuring systems themselves: complementary Golay pairs
deconvolve a time-invariant system with zero correlation noise, the
Mueller & Massarani shaped sweeps put the excitation energy where a target
spectrum asks for it while keeping a swept sine's crest factor, and the
regularized spectral inversion converts a measured response into a safe
equalizer with an analytic bound on flatness and out-of-band gain.

These estimators feed the rest of the library: transfer functions and
distortion analysis build on the cross-spectral machinery, room impulse
response work leans on delay estimation and alignment, and the
[uncertainty pages](/phonometry/signals/metrology/)
supply the error-analysis vocabulary the estimates are stated in.

## Pages in this section

- [Calibrated spectral analysis](/phonometry/signals/spectra/spectral-analysis/):
  Welch PSD/CSD with chi-square confidence intervals, the coherent output
  spectrum and spectral SNR, 1/n-octave smoothing and exact-slope
  colored-noise generators.
- [Multiple and partial coherence](/phonometry/signals/spectra/miso-coherence/):
  the Bendat & Piersol multiple-input/output coherence functions for
  multiple correlated sources and one output, with the Gaussian-elimination
  conditioning that tells a genuine cause from a source that merely
  correlates with it, and the partial coherent output spectra that say which
  source dominates each band.
- [Time-frequency analysis](/phonometry/signals/spectra/time-frequency/): the
  calibrated STFT spectrogram in absolute units (dB SPL for pascals) with
  the time-versus-frequency resolution trade-off, and the zoom FFT that
  resolves tones closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](/phonometry/signals/spectra/cepstrum-echoes/):
  the power/real/complex cepstrum with quefrency analysis, echo detection
  with the reflection coefficient read off the peak, lowpass/highpass
  liftering, the homomorphic round trip and the envelope spectrum of
  amplitude modulations.
- [Time synchronous averaging](/phonometry/signals/spectra/synchronous-averaging/):
  extraction of a periodic waveform of known period by time domain averaging,
  the comb filter that describes the operation in the frequency domain, the
  square-root noise-reduction law, and the choice of the number of averages
  that places a comb node on an interfering order (McFadden 1987).
- [Machine fault frequencies](/phonometry/vibration/machinery/machine-diagnostics/):
  the kinematic fault-frequency families of rotating machinery (Norton &
  Karczub Section 8.4) drawn on top of a measured envelope spectrum: bearing
  BPFO, BPFI, BSF and cage frequencies, gear-mesh sidebands, induction-motor
  slip, pole-pass and rotor-slot harmonics, and blade-passing tones.
- [Correlation, time delay and envelope](/phonometry/signals/spectra/correlation-delay/):
  correlation estimates with their random errors, time-delay estimation by
  direct correlation, phase slope and GCC weightings, sub-sample
  impulse-response alignment, and the Hilbert envelope.
- [Test signals and sample-rate tools](/phonometry/signals/spectra/test-signals/):
  IEC 60268-1 tone bursts with exact gating, resampling with a stated
  anti-alias specification, and band-limited fractional delay.
- [System measurement](/phonometry/signals/spectra/system-measurement/):
  complementary Golay pairs with exactly noise-free deconvolution to an
  impulse response, sweeps that follow an arbitrary target magnitude
  spectrum by group-delay shaping, and the Kirkeby-regularized inversion
  of a measured response.
