← [Documentation index](../../README.md)

# Measuring the Room Impulse Response

Every room-acoustic quantity on this site starts from the same measurement:
the **impulse response** (IR) between a source and a receiver. Get it right
and everything downstream (reverberation time, clarity, insulation, speech
metrics) is a matter of arithmetic; get it wrong and no later processing
recovers what the excitation or the microphone position threw away. This
guide covers the acquisition itself, per ISO 18233: the exponential sine
sweep and its deconvolution, the MLS correlation method, and where to place
sources and microphones so the averaged result means something. Turning the
measured IR into room parameters lives in
[Room Acoustics](room-acoustics.md); the same IR measured either side of a
partition becomes sound insulation in
[Field Insulation Measurement (ISO 16283)](../insulation/insulation-field.md).

## Two deterministic excitations (ISO 18233)

A room behaves, to a good approximation, as a **linear time-invariant**
system, so everything about it is contained in its IR. You could fire a
pistol and record the tail, but a deterministic excitation played through
a loudspeaker and *deconvolved* recovers the same IR with 20–30 dB more
effective signal-to-noise ratio (ISO 18233). Two excitations are provided.

**Exponential sine sweep (ESS, Annex B).** The instantaneous frequency
rises exponentially,

$$
f(t) = f_1 \left( \frac{f_2}{f_1} \right)^{t/T},
$$

so the time spent per octave is constant and the excitation mimics
pink noise (constant energy per fractional-octave band). The IR is
recovered by linear (zero-padded, non-circular) spectral division,

$$
H = \frac{Y\ \overline{X}}{|X|^2 + \varepsilon},
$$

with a small Tikhonov term $\varepsilon$ guarding the band edges where the
sweep has little energy. Because a low-to-high sweep places harmonic
distortion at *negative* arrival times, distortion separates cleanly from
the linear IR and is discarded by keeping only the causal part. The
`"farina"` method reaches the same result by convolving the recording with
the analytic inverse filter; it assumes the reference sweep was generated
with the default amplitude and fade, so use the spectral method for a
non-unit-amplitude or custom-fade sweep.

**Maximum-length sequence (MLS, Annex A).** An order-$N$ binary sequence of
length $2^N-1$ whose circular autocorrelation is a near-perfect delta; the
IR follows from circular cross-correlation of the recorded period with the
sequence. MLS excites at constant amplitude and is quick to average, but it
is more sensitive to time variance (draughts, temperature drift) and cannot
be fed as much power as a sweep. **Prefer the sweep** for rooms and
partitions; reach for MLS when the excitation must be periodic or the
hardware favours a two-level signal.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_ir_measurement_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_ir_measurement.svg" alt="ISO 18233 indirect measurement chain: an ESS sweep or MLS excitation drives a loudspeaker into the room, a microphone captures the response, and deconvolution (correlation or inverse filter) recovers the impulse response" width="92%"></picture>

```python
import numpy as np
from scipy.signal import fftconvolve
from phonometry import room

fs = 48000
# A 3 s, 20 Hz - 20 kHz sweep is a good broadband room excitation
# sweep: excitation you play through the loudspeaker
sweep = room.sweep_signal(fs, 20.0, 20000.0, 3.0)

# Deconvolve the recorded response back to the impulse response
system = np.zeros(fs); system[100] = 1.0; system[2000] = 0.4   # direct + reflection
# recorded: mic capture of the played sweep (here simulated by convolution with a synthetic room)
recorded = fftconvolve(sweep, system)
ir = room.impulse_response(recorded, sweep, fs, method="spectral")
print(int(np.argmax(np.abs(ir))))                    # 100: direct sound recovered
ir.plot()                     # waveform + Schroeder envelope (figure below)

# Farina inverse-filter variant (needs the sweep band)
ir_f = room.impulse_response(recorded, sweep, fs, method="farina", f_range=(20.0, 20000.0))

# Periodic MLS: excite with >= 2 periods, average, cross-correlate
mls = room.mls_signal(16)                                 # length 2**16 - 1 = 65535
rec = fftconvolve(np.tile(mls, 2), system)[: 2 * mls.size]
ir_m = room.mls_impulse_response(rec, mls)
print(int(np.argmax(np.abs(ir_m))))                  # 100
```

`sweep_signal`/`mls_signal` return plain arrays, ready to write to a WAV file
and play. `impulse_response`/`mls_impulse_response` return an
`ImpulseResponseResult`, a drop-in for the raw IR array (`np.asarray(ir)`,
indexing and `ir.size` all keep working, so `room_parameters(ir, fs)` is
unchanged) that also carries the sample rate and method and adds an `.plot()`.
Two more excitations from the transfer-function literature - complementary
Golay pairs with exactly noise-free deconvolution, and sweeps shaped to an
arbitrary target spectrum - live in the
[system-measurement guide](../../signals/spectra/system-measurement.md) and return the same
result types.

**The two excitations.** The exponential sweep sweeps its energy up the
spectrum over the whole signal, while the MLS is a flat-spectrum two-level
sequence, visible as the near-constant magnitude on the right.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/excitation_signals_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/excitation_signals.webp" alt="ISO 18233 excitation signals: the exponential sine sweep waveform and its spectrogram showing the exponential frequency rise, and a maximum-length sequence with its flat magnitude spectrum" width="96%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
from phonometry import room
from phonometry import plot_excitation

fs = 48000
sweep = room.sweep_signal(fs, 50.0, 20000.0, 1.0)   # ESS excitation
mls = room.mls_signal(12).astype(float)             # length 2**12 - 1

# One-liner: waveform + spectrogram (sweep), sequence + flat spectrum (MLS)
plot_excitation(sweep, fs, kind="sweep")
plot_excitation(mls, fs, kind="mls")

# By hand: the sweep spectrogram and the MLS magnitude spectrum
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.specgram(sweep, NFFT=1024, Fs=fs, noverlap=512)
ax1.set(xlabel="Time [s]", ylabel="Frequency [Hz]", title="Sweep spectrogram")
spec = np.abs(np.fft.rfft(mls))
freqs = np.fft.rfftfreq(mls.size, d=1.0 / fs)
ax2.semilogx(freqs[1:], 20 * np.log10(spec[1:] / np.median(spec[1:])))
ax2.set(xlabel="Frequency [Hz]", ylabel="Magnitude [dB]", title="MLS spectrum (flat)")
```

</details>

## The recovered impulse response

Deconvolving the recording gives the
broadband IR: the direct sound, discrete early reflections and the decaying
diffuse tail. Its `.plot()` shows the waveform above and the log-magnitude
envelope with the Schroeder energy-decay curve below: the straight decay
whose slope becomes the reverberation time in the
[Room Acoustics guide](room-acoustics.md).

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_sweep_deconvolution_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_sweep_deconvolution.gif" alt="Animation: the exponential sweep crosses the room while its spectrogram builds with delayed copies from the reflections, then the inverse filter collapses the whole recording into the impulse response" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_sweep_deconvolution.webm)

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impulse_response_dark.webp"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/impulse_response.webp" alt="Recovered room impulse response: the normalized waveform with the direct sound and reflections labelled, and below it the log-magnitude envelope in dB with the Schroeder energy-decay curve" width="88%"></picture>

<details>
<summary>Show the code for this figure</summary>

```python
import numpy as np
from scipy.signal import fftconvolve
from phonometry import room

fs = 48000
sweep = room.sweep_signal(fs, 20.0, 20000.0, 1.5)
# A synthetic room: direct sound + two reflections + a decaying diffuse tail
system = np.zeros(int(0.7 * fs))
system[80], system[1400], system[3100] = 1.0, 0.5, 0.32
ir = room.impulse_response(fftconvolve(sweep, system), sweep, fs, length=system.size)

# One-liner: waveform + log-magnitude / Schroeder decay
ir.plot()

# By hand: the normalized log-magnitude envelope in dB
import matplotlib.pyplot as plt
h = np.asarray(ir)
t = np.arange(h.size) / fs
plt.plot(t, 20 * np.log10(np.abs(h) / np.max(np.abs(h))))
plt.ylim(-80, 5)
plt.xlabel("Time [s]"); plt.ylabel("Level re peak [dB]")
```

</details>

### `sweep_signal()` / `inverse_filter()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `fs` | int | Hz | > 0 | Sampling frequency |
| `f1` | float | Hz | > 0, at/below lowest band | Sweep start frequency |
| `f2` | float | Hz | `f1 < f2 <= fs/2` | Sweep stop frequency |
| `seconds` | float | s | any; longer ⇒ more SNR | Sweep duration |
| `amplitude` | float | — | default `1.0` | Peak amplitude |
| `fade` | float | — | `[0, 0.5)`, default `0.01` | Half-Hann fade fraction (kills start/stop transients) |

### `impulse_response()` parameters

| Parameter | Type | Units | Range / default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `recorded` | 1D array | any | non-empty | Recorded system response |
| `reference` | 1D array | any | non-empty | The emitted sweep |
| `fs` | int | Hz | > 0 | Sample rate |
| `method` | str | — | `'spectral'` (default) / `'farina'` | `'farina'` requires `f_range` |
| `f_range` | (float, float) | Hz | default `None` | `(f1, f2)` of the sweep (Farina only) |
| `regularization` | float | — | default `1e-6` | Tikhonov term as a fraction of peak spectral energy |
| `length` | int, optional | samples | default `len(recorded)` | Samples of causal IR to return |
| `return_full` | bool | — | default `False` | Return the full sequence (distortion in the tail) |

`mls_signal(order)` takes an integer `order` in 2–20 (sequence length
$2^{\text{order}}-1$); `mls_impulse_response(recorded, mls, length=None)`
needs `recorded` to span an integer number of MLS periods.

## Where to measure

One IR characterises a single source–receiver pair; a
reported room parameter is the spatial average over several. ISO 3382-1
(performance spaces) asks for at least two source positions and microphones
spaced $\geq 2$ m apart, $\geq 1$ m from any surface, at $1.2$ m
(seated-ear) height; ISO 3382-2 fixes the minimum number of source,
microphone and source–microphone combinations per accuracy grade
(survey / engineering / precision) and asks for microphone positions that
avoid symmetric placements.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_room_measurement_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_room_measurement.svg" alt="Room-acoustics measurement setup: a top-view room plan with two loudspeaker source positions and six microphone positions with the ISO 3382-1 spacing rules, and the ISO 3382-2 table of minimum positions for the survey, engineering and precision grades" width="94%"></picture>

**Averaging across positions.** The reported per-band parameter is the
arithmetic mean over all source-microphone combinations, and the spread
across positions is part of the answer, not noise: quote it alongside the
mean whenever it exceeds the parameter's JND (ISO 3382-1 Table A.1), because a room can meet a
target on average while individual seats sit far outside it. Four placement
mistakes bias the mean itself:

- **Correlated positions.** Microphones closer than 2 m to each other
  sample nearly the same sound field twice, so the average looks more
  stable than it is. The 1 m minimum from any surface likewise avoids the
  pressure build-up near a boundary (and the comb filter of the animation
  below) that colours everything a too-close microphone records.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_comb_filtering_dark.gif"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_comb_filtering.gif" alt="Animation: as the microphone height changes, the delay between the direct sound and the floor reflection shifts and the comb filter in the frequency response moves with it, which is why measurement position matters near reflecting surfaces" width="640" height="360" loading="lazy"></picture>

[Watch the high-resolution video (WebM)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/anim_comb_filtering.webm)

- **Symmetric placements.** In a geometrically symmetric room, mirror-image
  positions receive mirror-image reflection patterns; averaging them adds
  no new information. This is why ISO 3382-2 asks for positions that do not
  sit on symmetry lines.
- **Too close to the source.** Inside the direct field EDT collapses and
  C80 saturates upward no matter what the room does. ISO 3382-2 therefore
  keeps every microphone at least $d_\mathrm{min} = 2\sqrt{V/(c\,\hat T)}$ from
  the source, with $\hat T$ an estimate of the expected reverberation time
  (about 2.2 m for a 200 m³ classroom with an expected 0.5 s).
- **Low-frequency luck.** Below the Schroeder frequency (see the
[Room Acoustics guide](room-acoustics.md)) each band
  holds only a handful of room modes, and a microphone on a node of one of
  them sees a different decay than a microphone on an antinode. The spread
  of the 63-125 Hz bands across positions is structurally larger; the cure
  is more positions, not a longer excitation.

Three practical limits close the chain. Playback and recording themselves —
sound-card I/O, level calibration of the chain — are outside the library,
which starts at the recorded array. ISO 18233's time-variance and distortion
diagnostics are not implemented either: the sweep separates distortion, but
no helper here quantifies it (that reading lives in the
[swept-sine distortion guide](../../devices/electroacoustics/swept-sine-distortion.md)).
And the position rules above are guidance prose: nothing checks how many
positions were measured or where.

## From the impulse response onward

The IR leaves this page in several directions. Band-filtered and
backward-integrated it becomes the decay curve whose slope is the
reverberation time: the [Room Acoustics guide](room-acoustics.md) picks up
exactly there, and its open-plan sibling walks the
[ISO 3382-3 line of workstations](open-plan-acoustics.md). Measured either
side of a partition, the same sweeps feed the level differences of
[field sound insulation](../insulation/insulation-field.md). And the harmonic distortion
that the exponential sweep pushes to negative arrival times is not always
discarded: read deliberately, it becomes the THD analysis of the
[swept-sine distortion guide](../../devices/electroacoustics/swept-sine-distortion.md).

## Quick answers

### Should I measure the room impulse response with a sine sweep or an MLS?

Both are ISO 18233 deconvolution methods that recover the impulse response with 20–30 dB more effective signal-to-noise ratio than an impulsive source. Prefer the exponential sine sweep (Annex B): it places harmonic distortion at negative arrival times, where it is discarded. Use MLS (Annex A) when the excitation must be periodic or the hardware favours a two-level signal; it is more sensitive to time variance.

## References

- International Organization for Standardization. (2006). *Acoustics —
  Application of new measurement methods in building and room acoustics*
  (ISO 18233:2006).
  [iso.org catalogue](https://www.iso.org/standard/40408.html).
  The swept-sine and MLS acquisition this page implements.
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40979.html).
  The source and microphone position requirements quoted in the
  where-to-measure rules.
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [iso.org catalogue](https://www.iso.org/standard/36201.html).
  The accuracy grades, position counts and the minimum source distance
  behind the position-averaging discussion.

## Standards

ISO 18233:2006 (application of new measurement methods: the swept-sine and
MLS acquisition of impulse responses); ISO 3382-1:2009 and ISO 3382-2:2008
(the source and microphone position requirements and accuracy grades applied
when planning where to measure). Validated against closed-form
deconvolution identities in the [conformance report](../../CONFORMANCE.md).

## See also

- [Room Acoustics](room-acoustics.md): the decay analysis and ISO 3382
  parameters computed from the IR this page acquires.
- [Open-Plan Office Acoustics (ISO 3382-3)](open-plan-acoustics.md): the
  speech-privacy line measured with the same acquisition.
- [System Measurement](../../signals/spectra/system-measurement.md): complementary Golay pairs and
  spectrally shaped sweeps, returning the same result types.
- [Swept-Sine Distortion & Phase](../../devices/electroacoustics/swept-sine-distortion.md): the Farina
  reading of the distortion the sweep separates.
- [Field Insulation Measurement (ISO 16283)](../insulation/insulation-field.md): the same
  measurement either side of a partition.
- [Image Sources & Steady-State Field](room-image-sources.md): simulated
  impulse responses to rehearse the processing chain on.
- API reference: [`room.impulse_response`](https://jmrplens.github.io/phonometry/reference/api/rooms/impulse-response/).
