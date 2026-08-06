← [Documentation index](../../README.md)

# Multichannel and Performance

Most real measurement sessions produce more than one channel: the two ears
of a dummy head, the microphone pair of an intensity probe, the several
positions of a room survey, the capsules of a beamforming array. The
convention for all of them is one array of shape `(channels, samples)`
processed in a single call: every function runs along the last (time) axis
and preserves the leading channel axis, so each row is analyzed exactly as
if it were the only one.

That per-channel guarantee is normative, not just convenient. The band
filters applied to each row are the IEC 61260-1 designs and the weightings
and detector ballistics are the IEC 61672-1 ones, unchanged from the
single-channel path; the vectorization batches the arithmetic across rows
(one filter design, one SciPy call) and never mixes them. This is the
textbook procedure for multiple data records (Bendat & Piersol 2010,
§10.4.2): analyze each record individually first, and compute anything
joint as a separate, explicit step.

Use this page when your channels are parallel recordings on the same clock
and you want per-channel spectra or levels. When the question is *between*
channels (what is the delay from A to B, how much of B is explained by A),
that is cross-channel analysis: see
[Correlation and delay](../spectra/correlation-delay.md) and
[Multiple and partial coherence](../spectra/miso-coherence.md), the implementations of
the Bendat & Piersol cross-correlation and multiple-input models.

## A stereo analysis in one call

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_multichannel_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/signal_response_multichannel.svg" alt="Stereo analysis: pink noise and logarithmic sweep resolved per channel in one-third-octave bands" width="80%"></picture>

*Simultaneous analysis of a Stereo signal: Left Channel (Pink Noise) vs Right
Channel (Log Sine Sweep).*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp
from phonometry import filters

# Stereo test signal: pink noise left, logarithmic sine sweep right
fs, duration = 48000, 5
t = np.linspace(0, duration, fs * duration, endpoint=False)
rng = np.random.default_rng(42)
spec = np.fft.rfft(rng.standard_normal(t.size))
spec[1:] /= np.sqrt(np.arange(1, spec.size))   # 1/f shaping: pink noise
left = np.fft.irfft(spec, t.size)
right = chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

x = np.stack([left, right])                    # (2, n_samples)
spl, freq = filters.octave_filter(x, fs, fraction=3, limits=[20, 20000])

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, levels, name in zip(axes, spl, ["Left: pink noise", "Right: log sweep"]):
    ax.semilogx(freq, levels, marker="o", label=name)
    ax.set_ylabel("Level [dB]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Frequency [Hz]")
plt.show()
```

</details>

The convention is consistent across the whole library: time is always the
**last axis**. This applies to `octave_filter`, `OctaveFilterBank`,
`weighting_filter`, `time_weighting`, `leq`, `laeq`, `ln_levels` and
`spectrogram`.

```python
import numpy as np
from phonometry import filters

# Two calibrated channels in Pa so the guide runs standalone
fs = 48000
t = np.arange(fs) / fs
left = 0.2 * np.sin(2 * np.pi * 1000 * t)
right = 0.1 * np.sin(2 * np.pi * 500 * t)

stereo = np.stack([left, right])          # (2, n_samples)
spl, freq = filters.octave_filter(stereo, fs, fraction=3)
# spl has shape (2, n_bands): one row per channel
```

## Accepted shapes, at a glance

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_multichannel_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_multichannel.svg" alt="Array-shape flow: a 1-D (samples,) input reduces to a scalar and a 2-D (channels, samples) input reduces to (channels,), because the operation runs along the last axis while the channel axis is preserved" width="80%"></picture>

| Input | Interpreted as | Typical output |
| :--- | :--- | :--- |
| `(n,)` 1D array | one channel | scalar level / `(bands,)` |
| `(ch, n)` 2D array | `ch` channels, `n` samples each | `(ch,)` levels / `(ch, bands)` |
| list of floats | one channel (converted) | as 1D |
| `(ch, n)` into `spectrogram` | multichannel STFT-style | `(ch, bands, frames)` |

Everything vectorizes across the leading channel axis: one filter design is
applied to all channels in a single SciPy call, so 8 channels cost far less
than 8 separate runs. Convention: **channels first**, like most DSP code
(`soundfile` returns `(n, ch)`: transpose with `x.T`).

## Per-channel semantics

Multichannel processing is strictly per channel: nothing is ever mixed,
summed or averaged across the channel axis. Three consequences worth
spelling out:

- **Combining channels is your decision.** Levels come back one per
  channel. If you need an array-average level (for instance the
  position-averaged levels of the room-acoustics standards), combine
  energies yourself: `10 * np.log10(np.mean(10 ** (spl / 10), axis=0))`,
  never the arithmetic mean of the dB values (see
  [Levels](../levels/levels.md) for why).
- **One `calibration_factor` means one sensitivity.** The scalar factor
  multiplies every channel, which is correct only if all channels share the
  same sensitivity. For an array of microphones with individual
  calibrations, scale the rows first, `x * factors[:, None]`, and leave
  `calibration_factor` at 1.
- **Stateful classes keep one state per channel.** In block processing the
  state array matches the channel count and a change of channel count
  resets it; see [Block Processing](block-processing.md).

## Performance: Vectorization and caching

`OctaveFilterBank` is the tool for repeated or streaming analysis: one bank
designs its filters once and applies them to every frame, and NumPy
broadcasting covers all channels of a frame in one filtering call per band,
with no Python loop over channels.

```python
import numpy as np
from phonometry import filters

# Two calibrated channels in Pa so the guide runs standalone
fs = 48000
t = np.arange(fs) / fs
left = 0.2 * np.sin(2 * np.pi * 1000 * t)
right = 0.1 * np.sin(2 * np.pi * 500 * t)
stereo = np.stack([left, right])          # (2, n_samples)

bank = filters.OctaveFilterBank(
    fs=48000, fraction=3, design=filters.FilterDesign(filter_type='butter'))

# Access computed properties
# bank.freq (center), bank.freq_d (lower), bank.freq_u (upper), bank.sos (coefficients)

# Process multiple signals efficiently
stream = [stereo]                 # your sequence of multichannel frames
for frame in stream:
    # detrend=True (default) removes DC offset to improve low-freq accuracy
    spl, freq = bank.filter(frame, detrend=True)
```

Additional performance notes:

- **Design cache**: `octave_filter()` reuses filter bank designs across calls
  with identical parameters (LRU cache, 32 entries), so calling it in a loop
  does not redesign the bank each time. `OctaveFilterBank` gives you explicit
  control over the design lifetime.
- **Multirate decimation**: low-frequency bands are filtered at a decimated
  rate, which is both faster and numerically more stable (see
  [Theory](../../reference/theory/signal-analysis.md)).
- **Optional numba**: the `impulse` time weighting kernel is JIT-compiled when
  numba is installed (`pip install phonometry[perf]`).

## See also

- [Correlation and delay](../spectra/correlation-delay.md): the cross-channel
  time-domain questions (delay, alignment) the per-channel path deliberately
  leaves to you.
- [Multiple and partial coherence](../spectra/miso-coherence.md): which of several
  correlated channels actually drives a response (Bendat & Piersol Ch. 7).
- [Block Processing](block-processing.md): the streaming counterpart, with
  one filter state per channel.
- [Levels](../levels/levels.md): the per-channel level metrics, and why dB values are
  combined energetically.

## References

- Bendat, J. S., & Piersol, A. G. (2010). *Random data: Analysis and
  measurement procedures* (4th ed.). Wiley.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  Section 10.4.2 (the procedure for analyzing multiple data records:
  individual per-record analysis first, joint cross-record analysis as a
  separate, deliberate step) and Chapter 7 (the multiple-input/output
  models that joint step leads to).
- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/5063).
  The band definitions each channel is filtered with; the multichannel path
  batches them unchanged.
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5708).
  The weighting and time-integration semantics applied per channel.

## Standards

IEC 61260-1:2014, *Electroacoustics — Octave-band and
fractional-octave-band filters — Part 1: Specifications*, and IEC 61672-1:2013,
*Electroacoustics — Sound level meters — Part 1: Specifications* —
multichannel support adds no normative content of its own: each channel is
filtered, weighted and time-integrated exactly as the single-channel standards
prescribe (see [Filter Banks](filter-banks.md) and [Levels](../levels/levels.md)); the
vectorization only batches the computation across the channel axis.
