← [Documentation index](../../README.md)

# Filter class verification (IEC 61260-1)

A filter bank becomes a measuring instrument only once its bands have been
proved against a specification. IEC 61260-1:2014 writes that specification
as an **acceptance mask**: a corridor of relative attenuation around each
mid frequency, narrow in the passband, opening into a minimum-attenuation
requirement far outside the band, with one corridor per performance class.
A bank "is class 1" when every band of it stays inside the class 1 corridor
at every normalized frequency, and the margin in decibels says by how much.

This page is the verification half of the octave-filtering topic: the 2014
mask and the per-band verdict, the stricter **class 0** kept alive by the
withdrawn IEC 61260:1995 and ANSI S1.11-2004 masks, a reading of what a
class actually buys in a measurement (passband error, stopband leakage,
uncertainty budget), and the one-page accredited fiche that turns the
verdict into a document. The design half, the band mathematics and the
parameter reference, is [Filter Banks](filter-banks.md), and the five
architectures with their compared responses are
[Filter Architecture Gallery](filter-gallery.md); the same machinery
applied to the frequency weightings is section 6 of
[Frequency Weighting](../levels/weighting.md).

## 1. Verifying the class against IEC 61260-1:2014

`verify_filter_class` checks every band of a bank against the acceptance
limits of **IEC 61260-1:2014** (Table 1, with the fractional-octave breakpoint
mapping and log-frequency interpolation from the standard) and reports the
performance class per band with its margin in dB:

```python
from phonometry import filters

bank = filters.OctaveFilterBank(fs=48000, fraction=3, order=6)
result = filters.verify_filter_class(bank)
print(result["overall_class"])          # 1
print(result["bands"][0])
# {'freq': 12.589254117941678, 'class': 1, 'checked_to_omega': 3.8127755266765493, 'margin_class1_db': 0.3999999999999595, 'margin_class2_db': 0.5999999999999595}
```

The Table 1 acceptance mask itself is public too: `class_limits(fraction,
filter_class, omega)` returns the minimum/maximum relative-attenuation
limits at normalized frequencies $\Omega = f/f_\mathrm{m}$, the same limits the verifier
and the figure below use.

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/class_mask_overlay_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/class_mask_overlay.svg" alt="Butterworth band response threading between the forbidden regions of the IEC 61260-1 class 1 acceptance mask" width="80%"></picture>

*The order-6 Butterworth response (blue) threads between the forbidden
regions: it must attenuate at least the red mask outside the band and no more
than the purple mask inside it.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfreqz
from phonometry import filters

fs = 48000
bank = filters.OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200])
idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
fm, fsd = bank.freq[idx], fs / bank.factor[idx]
w, h = sosfreqz(bank.sos[idx], worN=2**15, fs=fsd)
att = -20 * np.log10(np.abs(h) + 1e-12)
delta_a = att - np.interp(fm, w, att)     # relative attenuation

grid = np.logspace(np.log10(0.05), np.log10(8), 2000)
lo1, hi1 = filters.class_limits(1.0, 1, grid)     # class 1 min/max attenuation

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.fill_between(grid, -10, lo1, alpha=0.15, color="tab:red",
                label="Forbidden: too little attenuation")
finite = np.isfinite(hi1)
ax.fill_between(grid[finite], hi1[finite], 90, alpha=0.15, color="tab:purple",
                label="Forbidden: too much attenuation")
ax.plot(w / fm, delta_a, label="Butterworth order 6")
ax.set(xscale="log", xlim=(0.08, 8), ylim=(-6, 90),
       xlabel="Normalized frequency f / fm",
       ylabel="Relative attenuation [dB]")
ax.legend()
plt.show()
```

</details>

With default parameters (order 6), **Butterworth meets class 1**, and so does
**Chebyshev II**: its `attenuation` default is now `72` dB, clearing the 70 dB
far-stopband class 1 limit (scipy pins the cheby2 equiripple floor at exactly
`attenuation`, so any value $\ge 70\ \text{dB}$ qualifies; the 72 dB default
keeps the same +0.400 dB passband margin as Butterworth). Chebyshev I,
Elliptic and Bessel do
not meet class limits at order 6: passband ripple (cheby1/ellip) and slow
roll-off (bessel) violate the mask.

## 2. Class 0 (IEC 61260:1995 / ANSI S1.11-2004)

The tightest performance class, **class 0**, was defined by the earlier
**IEC 61260:1995** and its US twin **ANSI S1.11-2004** (both withdrawn/superseded
but still referenced for laboratory-grade instruments); IEC 61260-1:2014 dropped
it. Its class 1/2 masks differ slightly from the 2014 edition, so it lives behind
an `edition` switch rather than being mixed into the 2014 mask:

```python
from phonometry import filters

fs = 48000
bank = filters.OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200])

result = filters.verify_filter_class(bank, edition="1995")   # classes 0, 1, 2
print(result["overall_class"])          # 0  (the default Butterworth clears it)
print(result["bands"][0]["margin_class0_db"])
```

<picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/filter_class0_mask_dark.svg"><img src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/filter_class0_mask.svg" alt="Nested pass-band acceptance corridors for class 0, 1 and 2 of IEC 61260:1995 with the order-6 Butterworth response sitting inside the tightest class 0 corridor" width="80%"></picture>

*The class 0 corridor (±0.15 dB at mid-band) is the tightest; class 1 (±0.3 dB)
and class 2 (±0.5 dB) are progressively wider. The order-6 Butterworth threads
inside class 0 across the whole pass-band.*

<details>
<summary>Show the code for this figure</summary>

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfreqz
from phonometry import filters

fs = 48000
bank = filters.OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200])
idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
fm, fsd = bank.freq[idx], fs / bank.factor[idx]
w, h = sosfreqz(bank.sos[idx], worN=2**15, fs=fsd)
att = -20 * np.log10(np.abs(h) + 1e-12)
delta_a = att - np.interp(fm, w, att)

# Pass-band only: outside the band edges the maximum limit is +inf.
g = 10 ** (3 / 10)
grid = np.linspace(g ** -0.5, g ** 0.5, 1500)
pb = (w / fm >= g ** -0.5) & (w / fm <= g ** 0.5)

fig, ax = plt.subplots(figsize=(9, 5.5))
for cls in (2, 1, 0):                      # nested corridors, class 0 tightest
    lo, hi = filters.class_limits(1.0, cls, grid, edition="1995")
    ax.plot(grid, hi, label=f"Class {cls} corridor")
    ax.plot(grid, lo, color=ax.lines[-1].get_color())
ax.plot(w[pb] / fm, delta_a[pb], "k", lw=2, label="Butterworth order 6")
ax.set(xscale="log", xlim=(g ** -0.5, g ** 0.5), ylim=(-0.7, 6),
       xlabel="Normalized frequency f / fm",
       ylabel="Relative attenuation [dB]")
ax.legend()
plt.show()
```

</details>

## 3. What a class means physically

The masks are worst-case error bounds on a *measurement*, not abstract
grades:

- **In the passband** the corridor bounds how much the band can mis-read
  in-band content: a class 1 bank reads a mid-band tone within ±0.4 dB of
  its true level and a class 2 bank within ±0.6 dB (2014 Table 1; the
  stricter 1995 masks allowed ±0.3 dB for class 1 and ±0.15 dB for
  class 0). Toward the band edges the corridor widens, which is the honest
  admission that a tone sitting exactly on an edge is genuinely ambiguous
  between two bands (both read it about 3 dB down).
- **In the stopband** the minimum-attenuation mask bounds leakage from the
  rest of the spectrum: far from the band, class 1 demands at least 70 dB of
  relative attenuation (the reason the `cheby2` default is 72 dB). In energy
  terms, an out-of-band tone must be roughly 70 dB stronger than the band's
  own content before it doubles the band's energy reading (+3 dB). The
  practical consequence: measuring bands far below a dominant tone, the
  reading floors out at the leakage skirt about 70 dB down, and a steeper
  architecture (or higher order) is the only way to push that floor lower.
- **For the uncertainty budget**, the class is the filter's contribution to the
  measurement uncertainty: a class 1 bank adds up to a few tenths of a dB to
  a band level, comparable to a class 1 sound level meter's other tolerance
  terms, which is why instrument-grade chains specify the class of every
  stage rather than a single overall figure.

**Which architecture reaches which class?** The library's **default, Butterworth
order 6, meets class 0** in the configurations the conformance suite verifies
(octave and third-octave banks at 48 kHz), so no special setup is needed for
laboratory-grade banks in that range. The table below reports the best class
each architecture reaches under that same order-6 / 48 kHz setup; the other
architectures fall short of class 0 because they trade the IEC mask for a
different property *by construction*:

| Architecture | Best class (order 6, fs 48 kHz) | Why |
| :--- | :---: | :--- |
| `butter` (default) | **0** | Maximally-flat pass-band, monotone roll-off; fits the mask |
| `cheby2` | 1 | Flat pass-band but the mask relationship binds at class 1 |
| `cheby1` | — | Pass-band ripple violates the flatness limit |
| `ellip` | — | Pass- and stop-band ripple |
| `bessel` | — | Flat group delay bought with a slow roll-off |

So the sensible default is the common one (Butterworth order 6): it clears
class 0 in the verified configurations, while the alternative architectures are
deliberate opt-ins whose purpose (steeper roll-off, linear phase) works against
the class mask. Away from these settings (very high `fraction` or near-Nyquist
bands), always re-run `verify_filter_class` to confirm the class you need, and
raise the order if a band needs more margin.

## 4. The compliance fiche (`.report()`)

`filter_class_compliance(bank)` wraps the same verification as a result object
that exposes `.plot()` and `.report()`, so a type-test verdict can be rendered
as a one-page accredited fiche: a per-band classification table, the
worst-margin band's measured relative attenuation overlaid on the class
corridor, and the boxed overall class-compliance result. Pass a `required_class`
on the `ReportMetadata` to add a PASS/FAIL verdict row (a bank "meets class N"
when its achieved class is at least as strict, i.e. a class index of N or
lower). The fiche renders in English by default; pass `language="es"` for a
Spanish fiche (translated fixed strings and a comma decimal separator), e.g.
`result.report("iec61260_es.pdf", language="es")`.

```python
from phonometry import ReportMetadata, filters

bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[125, 4000])
result = filters.filter_class_compliance(bank)   # overall_class == 1
result.plot()   # the worst-margin band on its class corridor

result.report(
    "iec61260.pdf",
    metadata=ReportMetadata(
        specimen="1/1-octave filter bank",
        measurement_standard="IEC 61260-1:2014",
        required_class=1,                # class 1 (or stricter) required
    ),
)                                        # -> Class 1 - COMPLIES, PASS
```

The example fiche, regenerated with `make reports`, is kept rendered in the
repository. Click the preview to open the PDF:

[![One-page filter-class-compliance fiche: a metadata header, a per-band classification table listing each octave band's achieved class and binding margin, the worst-margin band's measured relative attenuation overlaid on the green class-1 acceptance corridor, the boxed Class 1 - COMPLIES (margin +0.40 dB) result and a PASS verdict against the required class 1](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61260_filter_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61260_filter_example.pdf)

*Filter class compliance fiche (`FilterComplianceResult.report`), the achieved
class with its binding margin in dB and the measured relative attenuation over
the IEC 61260-1:2014 Table 1 corridor.*

Passing `edition="1995"` verifies against the older IEC 61260:1995 /
ANSI S1.11-2004 mask, which keeps the stricter **class 0** that the 2014 edition
dropped; the default order-6 Butterworth bank can then be certified to class 0:

```python
bank = filters.OctaveFilterBank(fs=48000, fraction=1, order=6, limits=[250, 4000])
result = filters.filter_class_compliance(bank, edition="1995")   # overall_class == 0
result.plot()   # the class-0 corridor of the 1995 edition
result.report(
    "iec61260_1995.pdf",
    metadata=ReportMetadata(
        measurement_standard="IEC 61260:1995",
        required_class=0,                # class 0 (1995 edition) required
    ),
)                                        # -> Class 0 - COMPLIES, PASS
```

[![One-page filter-class-compliance fiche under the 1995 edition: a per-band classification table showing every octave band achieving class 0, the measured relative attenuation overlaid on the green class-0 acceptance corridor, the boxed Class 0 - COMPLIES (margin +0.15 dB) result and a PASS verdict against the required class 0](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61260_filter_1995_example.webp)](https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/reports/iec61260_filter_1995_example.pdf)

*Class 0 is retained by the IEC 61260:1995 / ANSI S1.11-2004 edition
(`edition="1995"`); the 2014 edition keeps only classes 1 and 2.*

## Quick answers

### Which filter architecture meets IEC 61260-1 class 1 with default settings?

With the default order 6, Butterworth meets class 1 of the IEC 61260-1:2014
Table 1 acceptance limits, and so does Chebyshev II: its default
`attenuation` of 72 dB clears the 70 dB far-stopband class 1 limit.
Chebyshev I, Elliptic and Bessel do not: passband ripple (cheby1, ellip)
and slow roll-off (bessel) violate the mask. `verify_filter_class` reports
the achieved class per band.

### What is class 0 and which standard defines it?

Class 0 is the tightest filter performance class, defined by IEC 61260:1995
and its US twin ANSI S1.11-2004 and dropped by IEC 61260-1:2014. Its
passband corridor allows only ±0.15 dB at mid-band, against ±0.3 dB for
class 1 in the 1995 masks. It stays available through `edition="1995"`, and
the default order-6 Butterworth bank meets class 0 in the verified 48 kHz
configurations.

## See also

- [Filter Banks](filter-banks.md): the band mathematics and the
  architectures whose class is verified here.
- [Frequency Weighting](../levels/weighting.md): the companion verification of the A,
  C and Z curves against the IEC 61672-1 tolerance tables, with the G, B, D and
  AU curves covered in [Special Weightings](../levels/special-weightings.md).
- [Sound level meter](../sound-level-meter.md): the instrument chain whose
  spectrum stage this class applies to.
- [Conformance report](../../CONFORMANCE.md): the verified configurations behind
  the class claims of this page.
- API reference: [`filters.compliance`](https://jmrplens.github.io/phonometry/reference/api/filters/compliance/).

## References

- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/5063).
  The Table 1 class 1 / class 2 acceptance limits verified here, with the
  fractional-octave breakpoint mapping and the log-frequency interpolation
  of the standard.
- International Electrotechnical Commission. (1995). *Electroacoustics —
  Octave-band and fractional-octave-band filters* (IEC 61260:1995).
  [IEC webstore](https://webstore.iec.ch/en/publication/5065).
  The withdrawn first edition whose Table 1 supplies the stricter class 0
  mask offered by `edition="1995"`.
- American National Standards Institute. (2004). *Specification for
  octave-band and fractional-octave-band analog and digital filters*
  (ANSI S1.11-2004). Acoustical Society of America.
  [ANSI webstore](https://webstore.ansi.org/standards/asa/ansis1112004).
  Its Table 1 class limits are identical to those of IEC 61260:1995 and back
  the same class 0 mask.

## Standards

IEC 61260-1:2014, *Electroacoustics — Octave-band and
fractional-octave-band filters — Part 1: Specifications*: the Table 1
class 1 / class 2 acceptance limits (with the fractional-octave breakpoint
mapping and log-frequency interpolation) verified in §1.
IEC 61260:1995 and ANSI S1.11-2004, *Octave-Band and Fractional-Octave-Band …
Filters*: the withdrawn edition's Table 1 (identical between the two)
supplies the stricter class 0 mask offered by ``edition="1995"`` and
verified in §2.

**Not covered.** `verify_filter_class` checks a *designed digital response*
against Table 1, not an instrument. IEC 61260-1's conformance tests for the
physical filter — overload recovery, filter linearity, environmental influences
— apply to hardware and are not run here; they belong to **IEC 61260-2:2016**
(pattern evaluation) and **IEC 61260-3:2016** (periodic tests), which this page
only summarises. Near Nyquist the bilinear transform warps the frequency axis
and the bank carries no correction for it, so the stopband mask beyond the
processing Nyquist is reported as `range_limited` rather than verified.

