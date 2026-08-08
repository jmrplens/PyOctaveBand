← [Documentation index](../../README.md)

# Signals and spectra

Band levels answer *how much*; this section answers *what is in the signal*.
Where the rest of the core works in fractional octave bands, these pages work
with the fine-grained estimators of classical signal analysis: spectral
densities, correlation functions, delays and envelopes. They share one
discipline, taken from Bendat & Piersol: every estimate is **calibrated** (the
same dB SPL / dBFS reference frames as the rest of the library) and carries
its **statistical quality**, so a spectrum is not just a curve but a curve
with a confidence interval.

Eight pages sit under that discipline, in six families: the frequency-domain
estimators ([spectral
analysis](spectral-analysis.md), [multiple and
partial coherence](miso-coherence.md)), the
time-frequency middle ground
([spectrograms](time-frequency.md)), the
spectrum-shape methods
([cepstrum](cepstrum-echoes.md)), the period-domain
methods ([synchronous
averaging](synchronous-averaging.md)), the
time-domain estimators ([correlation and
delay](correlation-delay.md)), and the stimulus and
system toolbox ([test
signals](test-signals.md), [system
measurement](system-measurement.md)).

They differ in what they estimate and share two decisions. The first is the
**segment length**. It fixes the resolution bandwidth of a PSD, the cell shape
of a spectrogram, the number of averages behind every confidence interval, the
longest delay a generalized cross-correlation can see and the degrees of
freedom a conditioned MISO estimate has left, so choosing it once and keeping
it is what makes a PSD, a coherence and a delay computed from the same record
mutually consistent. The second is **stationarity**. Every average on these
pages, and every error formula quoted beside it, assumes the process did not
drift while it was being recorded, which is exactly what the [data
qualification](../metrology/data-qualification.md) tests decide;
when a record fails them, the honest tools are the short-time views rather than
the averaged ones.

[Calibrated spectral analysis](spectral-analysis.md) is where
the frequency-domain family starts. The Welch power and cross-spectral density estimators
report their effective number of averages, normalized random errors and
chi-square confidence intervals; the coherent output spectrum splits a
measured output into the part explained by an input and the part that is
noise, with a spectral signal-to-noise ratio; fractional-octave smoothing
bridges back to the banded world; and the colored-noise generators synthesize
white, pink, red, blue and violet test signals with an exact power-law slope.
[Multiple and partial coherence](miso-coherence.md) carries
that same cross-spectral machinery to several correlated sources at once: from
multiple inputs and one output it separates the coherence a source
genuinely contributes from the part it merely shares with another, and its
partial coherent output spectra say which source dominates each band.

[Time-frequency analysis](time-frequency.md) is the view in
between: the calibrated STFT spectrogram shows what happens *when* - a
passing siren, an impact, a run-up - with every cell reading an absolute
level in the same scaling as the Welch estimators, and the zoom FFT computes
the spectrum of a narrow band on an arbitrarily fine grid to separate tones
closer than a practical FFT bin.
[Cepstrum, echoes and the envelope spectrum](cepstrum-echoes.md)
works on the *shape* of the spectrum. The power, real and complex cepstrum
collapse periodic spectral ripple onto quefrency spikes, echo detection reads
a reflection's delay and coefficient off the cepstral peak, liftering splits
a log spectrum into smooth envelope and fine structure, and the envelope
spectrum turns amplitude modulations into discrete lines at the modulation
frequency.
[Time synchronous averaging](synchronous-averaging.md)
extracts a repetitive waveform of known period from asynchronous noise by
ensemble-averaging successive periods: the residual noise falls as the square
root of the number of averages, and choosing that number to place a comb node
on an interfering order rejects it far better than the habitual power of two.

[Correlation, time delay and envelope](correlation-delay.md)
is where its time-domain counterpart starts. Auto- and cross-correlation come with the
Bendat & Piersol normalizations and random errors; time-delay estimation
offers the direct correlator, the cross-spectrum phase slope and the
Knapp & Carter generalized cross-correlation weightings (Roth, SCOT, PHAT,
maximum likelihood); impulse responses can be delayed and aligned with
sub-sample precision; and the Hilbert transform yields the envelope with
instantaneous phase and frequency.

[Test signals and sample-rate tools](test-signals.md) is
the toolbox underneath the rest: tone bursts with the exact gating of
IEC 60268-1 (zero-crossing start, integral full periods, repetitive trains)
that exercise detector ballistics, polyphase resampling behind an explicit
anti-alias specification whose designed filter travels with the result and
which every cross-rate comparison needs, and band-limited fractional delay
with a linear or circular boundary, whose kernel is shared by the
impulse-response alignment of
[Correlation, time delay and envelope](correlation-delay.md)
and the non-integer period alignment of
[Time synchronous averaging](synchronous-averaging.md).

[System measurement](system-measurement.md) turns the
toolbox toward measuring systems themselves: complementary Golay pairs
deconvolve a time-invariant system with zero correlation noise, the
Mueller & Massarani shaped sweeps put the excitation energy where a target
spectrum asks for it while keeping a swept sine's crest factor, and the
regularized spectral inversion converts a measured response into a safe
equalizer with an analytic bound on flatness and out-of-band gain.

These estimators feed the rest of the library: transfer functions and
distortion analysis build on the cross-spectral machinery, room impulse
response work leans on delay estimation and alignment, and the
[uncertainty pages](../metrology/index.md)
supply the error-analysis vocabulary the estimates are stated in.

## Pages in this section

- [Calibrated spectral analysis](spectral-analysis.md):
  Welch PSD/CSD with chi-square confidence intervals, the coherent output
  spectrum and spectral SNR, 1/n-octave smoothing and exact-slope
  colored-noise generators.
- [Multiple and partial coherence](miso-coherence.md):
  the Bendat & Piersol multiple-input/output coherence functions for
  multiple correlated sources and one output, with the Gaussian-elimination
  conditioning that tells a genuine cause from a source that merely
  correlates with it, and the partial coherent output spectra that say which
  source dominates each band.
- [Time-frequency analysis](time-frequency.md): the
  calibrated STFT spectrogram in absolute units (dB SPL for pascals) with
  the time-versus-frequency resolution trade-off, and the zoom FFT that
  resolves tones closer than a practical FFT bin.
- [Cepstrum, echoes and the envelope spectrum](cepstrum-echoes.md):
  the power/real/complex cepstrum with quefrency analysis, echo detection
  with the reflection coefficient read off the peak, lowpass/highpass
  liftering, the homomorphic round trip and the envelope spectrum of
  amplitude modulations.
- [Time synchronous averaging](synchronous-averaging.md):
  extraction of a periodic waveform of known period by time domain averaging,
  the comb filter that describes the operation in the frequency domain, the
  square-root noise-reduction law, and the choice of the number of averages
  that places a comb node on an interfering order (McFadden 1987).
- [Correlation, time delay and envelope](correlation-delay.md):
  correlation estimates with their random errors, time-delay estimation by
  direct correlation, phase slope and GCC weightings, sub-sample
  impulse-response alignment, and the Hilbert envelope.
- [Test signals and sample-rate tools](test-signals.md):
  IEC 60268-1 tone bursts with exact gating, resampling with a stated
  anti-alias specification, and band-limited fractional delay.
- [System measurement](system-measurement.md):
  complementary Golay pairs with exactly noise-free deconvolution to an
  impulse response, sweeps that follow an arbitrary target magnitude
  spectrum by group-delay shaping, and the Kirkeby-regularized inversion
  of a measured response.

## See also

Pages elsewhere on the site that this section leans on:

- [Machine fault frequencies](../../vibration/machinery/machine-diagnostics.md):
  the kinematic fault-frequency families of rotating machinery (Norton &
  Karczub Section 8.4) drawn on top of a measured envelope spectrum: bearing
  BPFO, BPFI, BSF and cage frequencies, gear-mesh sidebands, induction-motor
  slip, pole-pass and rotor-slot harmonics, and blade-passing tones.

## What this section does not cover

These are Bendat & Piersol's textbook estimators, not a certification method:
no page here carries clause numbers or acceptance limits, and no result is a
compliance verdict. Three capabilities a reader looks for are genuinely absent.
**Multiple arrivals are not separated.** `time_delay` and `echo_detection`
report the single largest peak, so a record with a direct path plus several
reflections needs manual peak-picking or repeated calls on narrowed bands.
**There is no multi-sensor geometry.** Delay estimation is pairwise; there is
no built-in TDOA solver, no beamformer and no source localisation, and a
multiple-output system needs one `miso_coherence` call per output.
**There are no perceptual features.** The cepstrum is the plain
linear-frequency one, with no mel warping and no MFCC variant. Two estimators
that share this section's Welch core are documented where they are used
instead: the transfer function and ordinary coherence on
[Electroacoustics](../../devices/electroacoustics/electroacoustics.md), and
the two-microphone intensity probe on [Sound
intensity](../../devices/emission/intensity.md).
