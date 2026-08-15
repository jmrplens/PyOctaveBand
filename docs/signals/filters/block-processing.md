← [Documentation index](../../README.md)

# Block Processing

Some measurements never fit in memory: an hour-long environmental recording,
a live monitor that must report levels while the microphone is still
capturing, or an embedded logger that only ever sees one buffer at a time. In
all of these the signal has to be processed block by block, and the filters
must behave exactly as if they had seen the whole signal at once.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_block_processing_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/diagram_block_processing.svg" alt="Two lanes comparing block processing with the filter state carried across blocks, giving one continuous envelope, versus reset each block, where the envelope restarts from zero at every seam" width="86%"></picture>

The `OctaveFilterBank`, `WeightingFilter` (for A, C, or Z-weighting) and
`TimeWeighting` classes support block (streaming) processing: the internal
filter state is carried between calls, so concatenated block outputs match a
single full-signal pass.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/block_processing_continuity_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/block_processing_continuity.svg" alt="Stateful block processing matching the continuous result versus independent blocks restarting the filter transient at each boundary" width="80%"></picture>

*With `stateful=True` the concatenated block outputs match the continuous
result exactly; without state, every block boundary restarts the filter
transient.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from phonometry import filters

# 1 kHz octave band: four stateful blocks vs one continuous pass
fs, block = 8000, 1000
rng = np.random.default_rng(42)
x = rng.standard_normal(4 * block)
t = np.arange(x.size) / fs

bank = filters.OctaveFilterBank(
    fs, fraction=1, limits=[900, 1100],
    design=filters.FilterDesign(resample=False),
    block_processing=filters.BlockProcessing(stateful=True),
)
streamed = np.concatenate([
    bank.filter(x[i * block:(i + 1) * block], sigbands=True,
                detrend=False, calculate_level=False)[2][0]
    for i in range(4)
])
offline = filters.OctaveFilterBank(
    fs, fraction=1, limits=[900, 1100],
    design=filters.FilterDesign(resample=False),
).filter(x, sigbands=True, detrend=False, calculate_level=False)[2][0]
print(np.max(np.abs(streamed - offline)))     # 0.0 (bit-exact)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t, offline, linewidth=2.5, alpha=0.35, label="Continuous (whole signal)")
ax.plot(t, streamed, label="Stateful blocks (state carried)")
ax.axvline(block / fs, color="gray", linestyle=":", label="Block boundary")
ax.set(xlim=(0.11, 0.17), xlabel="Time [s]", ylabel="Amplitude")
ax.legend()
plt.show()
```

</details>

Create a stateful filter bank with
`block_processing=BlockProcessing(stateful=True)`. The internal state is
zero-initialized by default but may be initialized for step-response
steady-state (like `scipy.signal.sosfilt_zi`) with
`BlockProcessing(steady_ic=True)`. The options that must be disabled in
stateful mode are summarized in the
[constraints table](#stateful-mode-constraints) at the end of this guide.

## Example

This example streams a WAV file with [`soundfile`](https://pysoundfile.readthedocs.io/),
an optional dependency (`pip install soundfile`). Any block source works just as
well: `scipy.io.wavfile` plus manual slicing, or a live capture callback.

```python
import soundfile as sf
from phonometry import filters

fs = 48000
octave_filter = filters.OctaveFilterBank(
    fs, 1,
    design=filters.FilterDesign(resample=False),
    block_processing=filters.BlockProcessing(stateful=True),
)
afilter = filters.WeightingFilter(fs, "A", stateful=True)

for block in sf.blocks("measurement.wav", blocksize=256, overlap=0):

    # Apply A-filter
    weighted = afilter.filter(block)

    # Split into octave bands
    block_spl, _, block_output = octave_filter.filter(weighted, sigbands=True, detrend=False)

    # further signal processing
    ...
```

## Time weighting across blocks

Use the `TimeWeighting` class (state carried automatically):

```python
from phonometry import filters

tw = filters.TimeWeighting(fs, mode="fast")
# audio_blocks: successive frames of your microphone recording (Pa),
#   e.g. from sf.blocks("measurement.wav", ...) as in the block above.
for block in audio_blocks:
    envelope = tw.process(block)
```

Or manage the state yourself with the functional API; see
[Time Weighting](../levels/time-weighting.md#6-block-processing).

## Multichannel state

All stateful classes handle multichannel input `(channels, samples)`: the state
is allocated lazily on the first call to match the channel count, and reallocated
if the channel count changes (e.g. switching from stereo to mono resets the state).

## What the carried state is

Every filter in the library runs as a cascade of second-order sections in
transposed direct form II. For one section with coefficients
$(b_0, b_1, b_2, a_1, a_2)$:

$$
y[n] = b_0\,x[n] + z_1[n-1], \qquad
z_1[n] = b_1\,x[n] - a_1\,y[n] + z_2[n-1], \qquad
z_2[n] = b_2\,x[n] - a_2\,y[n]
$$

The pair $(z_1, z_2)$ per section is the filter's *entire* memory of the
past. `stateful=True` stores these values when a block ends and restores
them when the next begins, so the recursion cannot tell where one buffer
stopped and the next started: the concatenated output equals the
single-pass output exactly, not approximately. The `TimeWeighting` detector
carries even less, just its last envelope value $y[n-1]$, which is the same
value the functional API hands back as `initial_state` (see
[Time Weighting](../levels/time-weighting.md#6-block-processing)).

### Streaming pitfalls

- **Per-block preprocessing creates seams.** Anything computed from a single
  block that should be global, like detrending (removing the block's own
  mean) or normalization, gives each block a slightly different operation:
  the outputs no longer concatenate into the continuous result. This is why
  `detrend` must be `False` in stateful mode.
- **Not every metric streams.** Energy metrics accumulate cleanly (a
  running $L_\mathrm{eq}$ is a running energy sum), but rank statistics do not: the
  $L_{90}$ of a recording is not any combination of per-block $L_{90}$
  values. Stream the *envelope* and compute percentiles once, on the pooled
  result.
- **The first block still carries the onset transient.** State starts at
  rest, so the filter settles during the first instants exactly as in a
  single pass. Use `steady_ic=True` to start in step-response steady state,
  or discard the settling time once (not once per block).
- **One stream per object.** A stateful instance holds the memory of one
  signal; feeding two interleaved streams through it corrupts both. Create
  one filter object per stream: the design cost is paid once at
  construction, not per block.

## Real-time level meter pattern

The canonical streaming loop weights, envelopes and reports block by block,
carrying all state across calls:

```python
import numpy as np
from phonometry import filters

fs, block = 48000, 4800  # 100 ms blocks
aw = filters.WeightingFilter(fs, "A", stateful=True)
env = filters.TimeWeighting(fs, mode="fast")  # the class is inherently stateful

for x in audio_stream(block):            # your capture callback
    y = env.process(aw.filter(x))
    spl = 10 * np.log10(y[..., -1] / (2e-5) ** 2)  # instantaneous LAF
    display(spl)
```

### Stateful-mode constraints

| Option | Stateful behavior | Why |
| :--- | :--- | :--- |
| `detrend` | must be `False` | Per-block detrending creates boundary discontinuities |
| `resample` | must be `False` | The resampler is not stateful |
| `zero_phase` | unsupported | Forward-backward filtering needs the whole signal |
| `high_accuracy` (weighting) | resolves to `False` by default (the legacy bilinear design, see [Frequency Weighting](../levels/weighting.md)); explicitly passing `True` raises `ValueError` | The polyphase resampling inside is block-incompatible |
| `steady_ic` | optional | Starts the filters in step-response steady state |

## See also

- [Filter Banks](filter-banks.md): the offline bank streaming reproduces bit for bit, and the `zero_phase` mode streaming cannot use.
- [Multichannel and Performance](multichannel.md): one state per channel, and where the computation time actually goes.
- [Time Weighting](../levels/time-weighting.md#6-block-processing): the functional detector API with the state passed by hand.
- [Integrated and Statistical Levels](../levels/levels.md): which metrics accumulate across blocks and which must be recomputed on the pooled envelope.
- API reference: [`filters.weighting`](https://jmrplens.github.io/phonometry/reference/api/filters/weighting/) and [`filters.core`](https://jmrplens.github.io/phonometry/reference/api/filters/core/).
- Theory: [Filter Bank Design and Numerical Stability](../../reference/theory/signal-analysis.md#filter-bank-design--numerical-stability): why a bank is designed as second-order sections and decimated per band, which is what makes streaming state possible at all.

## References

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-time signal processing*
  (3rd ed.). Pearson. ISBN 978-0-13-198842-2.
  [Open Library record](https://openlibrary.org/isbn/9780131988422).
  The direct-form filter structures and state recursion behind the carried
  state equation above.

## Standards

IEC 61260-1:2014, *Electroacoustics — Octave-band and
fractional-octave-band filters — Part 1: Specifications*, and IEC 61672-1:2013,
*Electroacoustics — Sound level meters — Part 1: Specifications* — block
processing adds no normative content of its own: the streamed filters are the
same designs those standards govern (see [Filter Banks](filter-banks.md),
[Frequency Weighting](../levels/weighting.md) and [Time Weighting](../levels/time-weighting.md)),
and carrying the internal filter state across blocks is exactly what makes the
concatenated output identical to a single full-signal pass, so every class and
tolerance claim of the underlying pages holds unchanged in streaming use.
