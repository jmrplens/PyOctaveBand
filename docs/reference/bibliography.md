← [Documentation index](../README.md)

# Bibliography

Every guide in this documentation closes with two citation sections: a
`## References` section listing the books and papers that support the physics
on the page (APA style, one bullet per source, each with a DOI or an official
publisher link, and half a sentence on what the entry supports), followed by a
`## Standards` section naming the normative documents the page implements,
clause by clause. This page collects the References entries of all
guides in one place, grouped by domain: a curated reading list, and the single
source of truth for link checking. Each entry lists the guide pages that cite
it; the list grows as guides gain their References sections.

## General acoustics

- Kinsler, L. E., Frey, A. R., Coppens, A. B., & Sanders, J. V. (2000).
  *Fundamentals of acoustics* (4th ed.). Wiley. ISBN 978-0-471-84789-2.
  [Publisher page](https://www.wiley.com/en-us/Fundamentals+of+Acoustics%2C+4th+Edition-p-9780471847892).
  The standard first course in acoustics: plane and spherical waves, acoustic
  impedance and the level definitions assumed throughout the guides.
  Cited by [Integrated and Statistical Levels](../signals/levels/levels.md).
- Rossing, T. D. (Ed.). (2014). *Springer handbook of acoustics* (2nd ed.).
  Springer. ISBN 978-1-4939-0754-0.
  [doi:10.1007/978-1-4939-0755-7](https://doi.org/10.1007/978-1-4939-0755-7).
  A one-volume survey of every domain this library touches, from room
  acoustics to psychoacoustics and underwater sound; the cross-domain
  reference of first resort.
- Beranek, L. L., & Mellow, T. J. (2012). *Acoustics: Sound fields and
  transducers*. Academic Press. ISBN 978-0-12-391421-7.
  [doi:10.1016/C2011-0-05897-0](https://doi.org/10.1016/C2011-0-05897-0).
  Sound fields, radiation and electroacoustic transducers; supports the
  electroacoustics and sound-power material.
  Cited by [Electroacoustics](../devices/electroacoustics/electroacoustics.md),
  [Loudspeaker characterisation](../devices/electroacoustics/loudspeakers.md) and
  [Sound Power](../devices/emission/sound-power.md).

## Signal processing

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-time signal processing*
  (3rd ed.). Pearson. ISBN 978-0-13-198842-2.
  [Open Library record](https://openlibrary.org/isbn/9780131988422).
  The digital-filter theory behind the SOS cascades, the bilinear transform
  and the multirate decimation used by the filter banks.
  Cited by [Filter Banks](../signals/filters/filter-banks.md) and
  [Block Processing](../signals/filters/block-processing.md).
- Smith, J. O. *Introduction to digital filters with audio applications*
  (online book). Center for Computer Research in Music and Acoustics (CCRMA),
  Stanford University.
  [ccrma.stanford.edu/~jos/filters](https://ccrma.stanford.edu/~jos/filters/).
  Free companion treatment of digital-filter design and analysis, a good next
  step after the filter-bank guides.
  Cited by [Filter Banks](../signals/filters/filter-banks.md) and
  [Filter Architecture Gallery](../signals/filters/filter-gallery.md).
- Bristow-Johnson, R. *Audio EQ Cookbook*. Republished as a W3C Working Group
  Note (ed. R. Toy), 8 June 2021.
  [w3.org/TR/audio-eq-cookbook](https://www.w3.org/TR/audio-eq-cookbook/).
  The biquad coefficient recipes (peaking, shelves, low/high-pass, band-pass,
  notch, all-pass) and their Q / bandwidth / shelf-slope parameterization,
  implemented verbatim by `ParametricEQ`.
  Cited by [Filter Banks](../signals/filters/filter-banks.md).
- Bendat, J. S., & Piersol, A. G. (2010). *Random Data: Analysis and
  Measurement Procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  The statistical error analysis behind the calibrated spectral estimators,
  the correlation and time-delay formulas, the Hilbert-envelope chapter and
  the log-magnitude/phase Hilbert relation of the minimum-phase utilities.
  Cited by [Calibrated Spectral Analysis](../signals/spectra/spectral-analysis.md),
  [Correlation, Time Delay and Envelope](../signals/spectra/correlation-delay.md) and
  [Swept-sine distortion and phase utilities](../devices/electroacoustics/swept-sine-distortion.md).
- Harris, F. J. (1978). On the use of windows for harmonic analysis with the
  discrete Fourier transform. *Proceedings of the IEEE*, 66(1), 51-83.
  [doi:10.1109/PROC.1978.10837](https://doi.org/10.1109/PROC.1978.10837).
  The window figures of merit (Table 1) computed by `window_metrics`.
  Cited by [Calibrated Spectral Analysis](../signals/spectra/spectral-analysis.md).
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE*, 70(9), 1055-1096.
  [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433).
  The multitaper method: Slepian tapers, eigenspectra and the adaptive
  weights implemented by `multitaper_psd`.
  Cited by [Calibrated Spectral Analysis](../signals/spectra/spectral-analysis.md).
- Percival, D. B., & Walden, A. T. (1993). *Spectral Analysis for Physical
  Applications: Multitaper and Conventional Univariate Techniques*.
  Cambridge University Press. ISBN 978-0-521-43541-3.
  [doi:10.1017/CBO9780511622762](https://doi.org/10.1017/CBO9780511622762).
  The multitaper development (Chapter 7) behind `multitaper_psd` and the
  Slepian-sequence eigenvalue tables that anchor its test oracle.
  Cited by [Calibrated Spectral Analysis](../signals/spectra/spectral-analysis.md).
- Knapp, C. H., & Carter, G. C. (1976). The generalized correlation method
  for estimation of time delay. *IEEE Transactions on Acoustics, Speech,
  and Signal Processing*, 24(4), 320-327.
  [doi:10.1109/TASSP.1976.1162830](https://doi.org/10.1109/TASSP.1976.1162830).
  The GCC framework and the Roth/SCOT/PHAT/maximum-likelihood weightings of
  the time-delay estimators.
  Cited by [Correlation, Time Delay and Envelope](../signals/spectra/correlation-delay.md).
- Farina, A. (2000). Simultaneous measurement of impulse response and
  distortion with a swept-sine technique. *108th AES Convention*, Paris,
  preprint 5093.
  [AES E-Library record](https://www.aes.org/e-lib/browse.cfm?elib=10211).
  The exponential-sweep deconvolution and the L·ln(n) packing of the
  harmonic impulse responses ahead of the linear one.
  Cited by [Swept-sine distortion and phase utilities](../devices/electroacoustics/swept-sine-distortion.md).
- Novak, A., Lotton, P., & Simon, L. (2015). Synchronized swept-sine:
  Theory, application and implementation. *Journal of the Audio Engineering
  Society*, 63(10), 786-798.
  [doi:10.17743/jaes.2015.0071](https://doi.org/10.17743/jaes.2015.0071).
  The synchronization condition for coherent harmonic phases, the analytic
  inverse-filter spectrum and the fractional-sample harmonic separation.
  Cited by [Swept-sine distortion and phase utilities](../devices/electroacoustics/swept-sine-distortion.md).
- Müller, S., & Massarani, P. (2001). Transfer-function measurement with
  sweeps. *Journal of the Audio Engineering Society*, 49(6), 443-471.
  [AES E-Library record](https://www.aes.org/e-lib/browse.cfm?elib=10189).
  The sweep-measurement monograph behind the practice notes: inverse
  filters, fades and distortion rejection.
  Cited by [Swept-sine distortion and phase utilities](../devices/electroacoustics/swept-sine-distortion.md).

## Measurement instrumentation

- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/5063).
  The base-10 band edges and the class acceptance masks of the fractional
  octave banks.
  Cited by [Filter Banks](../signals/filters/filter-banks.md),
  [Filter Architecture Gallery](../signals/filters/filter-gallery.md),
  [Filter class verification](../signals/filters/filter-compliance.md) and
  [Multichannel and Performance](../signals/filters/multichannel.md).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5708).
  The A/C/Z weightings, the exponential time weightings and the level
  metrics of the sound level meter, with the tolerance tables used for
  verification.
  Cited by [Integrated and Statistical Levels](../signals/levels/levels.md),
  [Frequency Weighting (A, C, Z)](../signals/levels/weighting.md),
  [Time Weighting and Integration](../signals/levels/time-weighting.md) and
  [Multichannel and Performance](../signals/filters/multichannel.md).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 3: Periodic tests* (IEC 61672-3:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5710).
  The periodic laboratory verification of a sound level meter.
  Cited by [Calibration and dBFS](../signals/metrology/calibration.md).
- International Electrotechnical Commission. (2017). *Electroacoustics —
  Sound calibrators* (IEC 60942:2017).
  [IEC webstore](https://webstore.iec.ch/en/publication/30045).
  The calibrator classes, level tolerances and the short-term stability
  criterion applied to calibration recordings.
  Cited by [Calibration and dBFS](../signals/metrology/calibration.md).
- International Electrotechnical Commission. (2014). *Sound system equipment —
  Part 4: Microphones* (IEC 60268-4:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/32039).
  The rated microphone characteristics: free-field sensitivity and its level
  re 1 V/Pa, the frequency response and the effective frequency range against
  the tolerance limits, the directional pattern and the directivity index, the
  overload sound pressure level, the equivalent sound pressure level due to
  inherent noise, and the rated impedances and power supply.
  Cited by [Microphone characterisation](../devices/electroacoustics/microphones.md).
- International Electrotechnical Commission. (2007). *Sound system equipment —
  Part 5: Loudspeakers* (IEC 60268-5:2003+A1:2007).
  [IEC webstore](https://webstore.iec.ch/en/publication/1223).
  The rated loudspeaker characteristics: rated impedance, rated frequency
  range, characteristic sensitivity referred to 1 W at 1 m, the effective
  frequency range against the -10 dB band, the directivity index and the total
  harmonic distortion against frequency.
  Cited by [Loudspeaker characterisation](../devices/electroacoustics/loudspeakers.md).
- International Electrotechnical Commission. (1982). *Scales and sizes for
  plotting frequency characteristics and polar diagrams* (IEC 60263:1982).
  [IEC webstore](https://webstore.iec.ch/en/publication/1218).
  The scale proportions of the characteristic graphs: one frequency decade
  equal to 25 dB on the ordinate, and the polar diagram on a 25 dB
  reference-circle radius.
  Cited by [Loudspeaker characterisation](../devices/electroacoustics/loudspeakers.md) and
  [Microphone characterisation](../devices/electroacoustics/microphones.md).
- International Electrotechnical Commission. (1985). *Sound system
  equipment — Part 1: General* (IEC 60268-1:1985).
  [IEC webstore](https://webstore.iec.ch/en/publication/1204).
  Annex A, Clause A2: tone bursts starting at the zero crossing of the tone
  with an integral number of full periods, repetitive burst trains at a
  stated repetition rate, and the Table AII burst durations.
  Cited by [Test signals and sample-rate tools](../signals/spectra/test-signals.md).

## Broadcast programme loudness

- International Telecommunication Union. (2023). *Algorithms to measure
  audio programme loudness and true-peak audio level* (Recommendation
  ITU-R BS.1770-5).
  [ITU-R publication](https://www.itu.int/rec/R-REC-BS.1770).
  The K-weighting, channel weights, gating and true-peak algorithms.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).
- European Broadcasting Union. (2023). *Loudness normalisation and permitted
  maximum level of audio signals* (EBU R 128).
  [tech.ebu.ch/publications/r128](https://tech.ebu.ch/publications/r128).
  The −23.0 LUFS target, the −1 dBTP ceiling and the normalisation practice.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).
- European Broadcasting Union. (2023). *Loudness metering: 'EBU Mode'
  metering to supplement loudness normalisation* (EBU Tech 3341).
  [tech.ebu.ch/publications/tech3341](https://tech.ebu.ch/publications/tech3341).
  The M/S/I time scales and the minimum-requirements test signals.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).
- European Broadcasting Union. (2023). *Loudness range: A measure to
  supplement loudness normalisation* (EBU Tech 3342).
  [tech.ebu.ch/publications/tech3342](https://tech.ebu.ch/publications/tech3342).
  The LRA algorithm, its reference implementation and its test signals.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).
- European Broadcasting Union. (2023). *Guidelines for production of
  programmes in accordance with EBU R 128* (EBU Tech 3343).
  [tech.ebu.ch/publications/tech3343](https://tech.ebu.ch/publications/tech3343).
  The production practice behind the loudness normalisation numbers.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).
- Steinmetz, C. J., & Reiss, J. D. (2021). *pyloudnorm: A simple yet
  flexible loudness meter in Python*. 150th AES Convention.
  [github.com/csteinmetz1/pyloudnorm](https://github.com/csteinmetz1/pyloudnorm).
  An independent BS.1770 implementation, useful as a cross-check.
  Cited by [Programme loudness & true peak](../devices/broadcast/program-loudness.md).

## Sound power and intensity

- Fahy, F. J. (1995). *Sound intensity* (2nd ed.). E&FN Spon.
  ISBN 978-0-419-19810-9.
  [doi:10.4324/9780203475386](https://doi.org/10.4324/9780203475386).
  The monograph on sound energy flux: active and reactive intensity, the
  p-p estimator and its phase-mismatch error budget.
  Cited by [Sound power by intensity scanning](../devices/emission/sound-power-intensity.md) and
  [Sound Intensity (p-p)](../devices/emission/intensity.md).
- International Organization for Standardization. (2019). *Acoustics —
  Determination of sound power levels of noise sources — Guidelines for the
  use of basic standards* (ISO 3740:2019).
  [iso.org catalogue](https://www.iso.org/standard/45107.html).
  The selection guide for the sound-power family: grades, environments,
  source-size and background criteria.
  Cited by [Sound Power](../devices/emission/sound-power.md).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for reverberation test
  rooms* (ISO 3741:2010).
  [iso.org catalogue](https://www.iso.org/standard/52053.html).
  The precision reverberation-room method.
  Cited by [Sound power in the reverberation room](../devices/emission/sound-power-reverberation.md).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Engineering methods for an essentially free
  field over a reflecting plane* (ISO 3744:2010).
  [iso.org catalogue](https://www.iso.org/standard/52055.html).
  The enveloping-surface engineering method.
  Cited by [Sound power by pressure methods](../devices/emission/sound-power-pressure.md).
- International Organization for Standardization. (2012). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for anechoic rooms and
  hemi-anechoic rooms* (ISO 3745:2012).
  [iso.org catalogue](https://www.iso.org/standard/45362.html).
  The precision anechoic-room method.
  Cited by [Sound power by pressure methods](../devices/emission/sound-power-pressure.md).
- International Organization for Standardization. (1996). *Acoustics —
  Declaration and verification of noise emission values of machinery and
  equipment* (ISO 4871:1996).
  [iso.org catalogue](https://www.iso.org/standard/10868.html).
  What a manufacturer publishes once the power level is measured: the dual
  and single-number declaration forms, the declared value
  $L_{WAd} = L_{WA} + K_{WA}$ and the clause 6.2 verification procedure.
  Cited by [Sound Power](../devices/emission/sound-power.md).
- International Organization for Standardization. (1993). *Acoustics —
  Determination of sound power levels of noise sources using sound
  intensity — Part 1: Measurement at discrete points* (ISO 9614-1:1993).
  [iso.org catalogue](https://www.iso.org/standard/17427.html).
  The field indicators and the dynamic-capability criterion of intensity
  measurement.
  Cited by [Sound Intensity (p-p)](../devices/emission/intensity.md).
- International Electrotechnical Commission. (1993). *Electroacoustics —
  Instruments for the measurement of sound intensity — Measurements with
  pairs of pressure sensing microphones* (IEC 61043:1993; adopted in Europe
  as EN 61043:1994).
  [IEC webstore](https://webstore.iec.ch/en/publication/4353).
  The p-p instrument standard: the cross-spectral estimator and the
  residual pressure-intensity index.
  Cited by [Sound Intensity (p-p)](../devices/emission/intensity.md).

## Room acoustics

- Long, M. (2014). *Architectural acoustics* (2nd ed.). Academic Press.
  [doi:10.1016/C2012-0-03257-5](https://doi.org/10.1016/C2012-0-03257-5).
  The architectural-design companion to the measurement standards: the
  rectangular-room eigenfrequencies and the Morse/Pierce mode count
  (Chapter 8), the crowd self-noise of an occupied room (Chapter 17) and the
  gain-before-feedback criterion of a reinforcement system (Chapter 18).
  Cited by [Image sources and the steady-state room field](../buildings/rooms/room-image-sources.md),
  [Open-plan office acoustics](../buildings/rooms/open-plan-acoustics.md) and
  [Loudspeaker characterisation](../devices/electroacoustics/loudspeakers.md).
- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The reference monograph on sound fields in rooms: statistical decay
  theory, the Schroeder frequency, absorption and the perceptual room
  parameters.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md),
  [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md),
  [Image sources and the steady-state room field](../buildings/rooms/room-image-sources.md),
  [Sound absorption in enclosed spaces](../buildings/rooms/enclosed-space-absorption.md) and
  [2D FDTD wave simulation](../simulation/fdtd-simulation.md).
- Vorländer, M. (2020). *Auralization: Fundamentals of acoustics, modelling,
  simulation, algorithms and acoustic virtual reality* (2nd ed.). Springer.
  [doi:10.1007/978-3-030-51202-6](https://doi.org/10.1007/978-3-030-51202-6).
  The mirror-source room model and the source-path-receiver auralization
  chain.
  Cited by [Image sources and the steady-state room field](../buildings/rooms/room-image-sources.md).
- Allen, J. B., & Berkley, D. A. (1979). Image method for efficiently
  simulating small-room acoustics. *The Journal of the Acoustical Society of
  America*, 65(4), 943-950.
  [doi:10.1121/1.382599](https://doi.org/10.1121/1.382599).
  The efficient rectangular-room image-source construction and its
  reflection-count decomposition.
  Cited by [Image sources and the steady-state room field](../buildings/rooms/room-image-sources.md).
- Sabine, W. C. (1922). *Collected papers on acoustics*. Harvard University
  Press.
  [Free scan at the Internet Archive](https://archive.org/details/collectedpaperso00sabi).
  The founding reverberation experiments and the Sabine law.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Eyring, C. F. (1930). Reverberation time in "dead" rooms. *The Journal of
  the Acoustical Society of America*, 1(2A), 217-241.
  [doi:10.1121/1.1915175](https://doi.org/10.1121/1.1915175).
  The mean-free-path reverberation formula for strongly absorbing rooms.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Millington, G. (1932). A modified formula for reverberation. *The Journal
  of the Acoustical Society of America*, 4(1), 69-82.
  [doi:10.1121/1.1915588](https://doi.org/10.1121/1.1915588).
  The per-surface logarithmic reverberation formula.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Fitzroy, D. (1959). Reverberation formula which seems to be more accurate
  with nonuniform distribution of absorption. *The Journal of the
  Acoustical Society of America*, 31(7), 893-897.
  [doi:10.1121/1.1907814](https://doi.org/10.1121/1.1907814).
  The axial reverberation formula for anisotropic absorption.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Arau-Puchades, H. (1988). An improved reverberation formula. *Acustica*,
  65(4), 163-180.
  [Publisher record at Ingenta](https://www.ingentaconnect.com/content/dav/aaua/1988/00000065/00000004/art00003).
  The geometric-mean refinement of the axial reverberation formula.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Schroeder, M. R. (1965). New method of measuring reverberation time.
  *The Journal of the Acoustical Society of America*, 37(3), 409-412.
  [doi:10.1121/1.1909343](https://doi.org/10.1121/1.1909343).
  The backward integration of the squared impulse response into a decay
  curve.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- Hak, C. C. J. M., Wenmaekers, R. H. C., & van Luxemburg, L. C. J. (2012).
  Measuring room impulse responses: Impact of the decay range on derived
  room acoustic parameters. *Acta Acustica united with Acustica*, 98(6),
  907-915. [doi:10.3813/aaa.918574](https://doi.org/10.3813/aaa.918574).
  The impulse-to-noise-ratio (INR) analysis of decay-range requirements.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- Everest, F. A. (2001). *Master handbook of acoustics* (4th ed.).
  McGraw-Hill. ISBN 978-0-07-136097-5.
  [Open Library record](https://openlibrary.org/isbn/9780071360975).
  A practical room-acoustics handbook; its Fig. 7-22 worked example anchors
  the reverberation-prediction conformance suite.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Carrión Isbert, A. (1998). *Diseño acústico de espacios arquitectónicos*.
  Edicions UPC. ISBN 978-84-8301-252-9.
  [Open Library record](https://openlibrary.org/books/OL23159935M).
  A Spanish-language textbook on acoustic room design.
  Cited by [Reverberation-time prediction](../buildings/rooms/reverberation-prediction.md).
- Beranek, L. L. (1957). Revised criteria for noise in buildings. *Noise
  Control*, 3(1), 19-27.
  [doi:10.1121/1.2369239](https://doi.org/10.1121/1.2369239).
  The original NC curves and their speech-interference rationale.
  Cited by [Room-noise criteria](../buildings/rooms/room-noise.md).
- Kosten, C. W., & van Os, G. J. (1962). Community reaction criteria for
  external noises. In *The Control of Noise* (National Physical Laboratory
  Symposium No. 12, pp. 373-387). Her Majesty's Stationery Office.
  [Open Library record](https://openlibrary.org/books/OL58781133M).
  The NR curve family contrasted with NC.
  Cited by [Room-noise criteria](../buildings/rooms/room-noise.md).
- Blazier, W. E. (1997). RC Mark II: A refined procedure for rating the
  noise of heating, ventilating, and air-conditioning (HVAC) systems in
  buildings. *Noise Control Engineering Journal*, 45(6), 243-250.
  [doi:10.3397/1.2828446](https://doi.org/10.3397/1.2828446).
  The RC Mark II procedure later codified by ANSI/ASA S12.2 Annex D.
  Cited by [Room-noise criteria](../buildings/rooms/room-noise.md).
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40979.html).
  Room-parameter definitions, position requirements and just-noticeable
  differences.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [iso.org catalogue](https://www.iso.org/standard/36201.html).
  The accuracy grades and position counts of reverberation measurement.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- International Organization for Standardization. (2012). *Acoustics —
  Measurement of room acoustic parameters — Part 3: Open plan offices*
  (ISO 3382-3:2012; since revised as
  [ISO 3382-3:2022](https://www.iso.org/standard/77437.html)).
  [iso.org catalogue](https://www.iso.org/standard/46520.html).
  The open-plan speech-privacy quantities.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- International Organization for Standardization. (2006). *Acoustics —
  Application of new measurement methods in building and room acoustics*
  (ISO 18233:2006).
  [iso.org catalogue](https://www.iso.org/standard/40408.html).
  The swept-sine and MLS acquisition of impulse responses.
  Cited by [Room Acoustics](../buildings/rooms/room-acoustics.md).
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [iso.org catalogue](https://www.iso.org/standard/34545.html).
  The reverberation-room absorption measurement behind the surface data.
  Cited by [Sound absorption in enclosed spaces](../buildings/rooms/enclosed-space-absorption.md) and
  [Sound Absorption Measurement and Rating](../materials/absorbers/absorption-measurement.md).
- European Committee for Standardization. (2003). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 6: Sound absorption in enclosed spaces*
  (EN 12354-6:2003).
  [BSI Knowledge record (BS EN 12354-6:2003)](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-of-buildings-from-the-performance-of-elements-sound-absorption-in-enclosed-spaces).
  The absorption member of the EN 12354 prediction family.
  Cited by [Sound absorption in enclosed spaces](../buildings/rooms/enclosed-space-absorption.md).
- Acoustical Society of America. (2019). *Criteria for evaluating room
  noise* (ANSI/ASA S12.2-2019).
  [ANSI webstore](https://webstore.ansi.org/standards/asa/ansiasas122019).
  The normative NC tangency method and the RC Mark II rating of its
  informative Annex D, with its spectral tag.
  Cited by [Room-noise criteria](../buildings/rooms/room-noise.md).

## Materials and surfaces

- Allard, J. F., & Atalla, N. (2009). *Propagation of sound in porous media:
  Modelling sound absorbing materials* (2nd ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  The porous-material theory linking airflow resistivity, surface impedance
  and absorption.
  Cited by [Airflow Resistance](../materials/absorbers/airflow-resistance.md),
  [Impedance Tube](../materials/absorbers/impedance-tube.md) and
  [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3rd ed.). CRC Press.
  ISBN 978-1-4987-4099-9.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  The monograph on absorber and diffuser measurement and design, by the
  authors behind the ISO 17497-2 diffusion-coefficient method.
  Cited by
  [Sound Absorption Measurement and Rating](../materials/absorbers/absorption-measurement.md),
  [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md),
  [Diffusers and Their Coefficients](../materials/diffusers/diffusers.md),
  [Metadiffusers](../materials/diffusers/metadiffusers.md) and
  [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md).
- Hargreaves, T. J., Cox, T. J., Lam, Y. W., & D'Antonio, P. (2000). Surface
  diffusion coefficients for room acoustics: Free-field measures of
  single-plane diffusion. *The Journal of the Acoustical Society of America*,
  108(4), 1710-1720.
  [doi:10.1121/1.1310192](https://doi.org/10.1121/1.1310192).
  The free-field diffusion-coefficient method behind ISO 17497-2 and the
  published N = 7 QRD geometry of the worked example.
  Cited by [Diffusers and Their Coefficients](../materials/diffusers/diffusers.md).
- Audio Engineering Society. (2001). *AES information document for room
  acoustics and sound reinforcement systems — Characterization and
  measurement of surface scattering uniformity* (AES-4id-2001). *Journal of
  the Audio Engineering Society*, 49(3), 149-165.
  [AES standards in print](https://www.aes.org/publications/standards/list.cfm).
  The single-plane free-field diffusion-coefficient procedure that
  ISO 17497-2 later standardised.
  Cited by [Diffusers and Their Coefficients](../materials/diffusers/diffusers.md).
- Jiménez, N., Cox, T. J., Romero-García, V., & Groby, J.-P. (2017).
  Metadiffusers: Deep-subwavelength sound diffusers. *Scientific Reports*,
  7, 5389.
  [doi:10.1038/s41598-017-05710-5](https://doi.org/10.1038/s41598-017-05710-5).
  The metadiffuser model: resonator-loaded slits reproducing Schroeder phase
  profiles and ternary sequences from deep-subwavelength panels.
  Cited by [Metadiffusers](../materials/diffusers/metadiffusers.md).
- Jiménez, N., Cox, T. J., Groby, J.-P., & Romero-García, V. (2019). Beyond
  phase grating diffusers using locally-resonant metamaterials. *Proceedings
  of the 23rd International Congress on Acoustics (ICA 2019)*, Aachen.
  [Proceedings PDF](https://pub.dega-akustik.de/ICA2019/data/articles/000706.pdf).
  The congress companion to the metadiffuser paper: the transfer-matrix
  chain and the slow-sound dispersion picture.
  Cited by [Metadiffusers](../materials/diffusers/metadiffusers.md).
- Jiménez, N., Groby, J.-P., Pagneux, V., & Romero-García, V. (2017).
  Iridescent perfect absorption in critically-coupled acoustic metamaterials
  using the transfer matrix method. *Applied Sciences*, 7(6), 618.
  [doi:10.3390/app7060618](https://doi.org/10.3390/app7060618).
  The slit + Helmholtz-resonator transfer-matrix model and the
  critical-coupling condition.
  Cited by [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md) and
  [Metadiffusers](../materials/diffusers/metadiffusers.md).
- Jiménez, N., Huang, W., Romero-García, V., Pagneux, V., & Groby, J.-P.
  (2016). Ultra-thin metamaterial for perfect and quasi-omnidirectional
  sound absorption. *Applied Physics Letters*, 109(12), 121902.
  [doi:10.1063/1.4962328](https://doi.org/10.1063/1.4962328).
  The resonator impedance and radiation end corrections, and the published
  λ/88 perfect absorber.
  Cited by [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md) and
  [Metadiffusers](../materials/diffusers/metadiffusers.md).
- Jiménez, N., Umnova, O., & Groby, J.-P. (Eds.). (2021). *Acoustic waves in
  periodic structures, metamaterials, and porous media* (Topics in Applied
  Physics, Vol. 143). Springer.
  [doi:10.1007/978-3-030-84300-7](https://doi.org/10.1007/978-3-030-84300-7).
  The book-length treatment of resonant metamaterial absorbers and
  diffusers; the modern companion to Cox & D'Antonio.
  Cited by [Metadiffusers](../materials/diffusers/metadiffusers.md) and
  [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md).
- Stinson, M. R. (1991). The propagation of plane sound waves in narrow and
  wide circular tubes, and generalization to uniform tubes of arbitrary
  cross-sectional shape. *Journal of the Acoustical Society of America*,
  89(2), 550-558. [doi:10.1121/1.400379](https://doi.org/10.1121/1.400379).
  The visco-thermal effective parameters of the slit and the square necks
  and cavities.
  Cited by [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md).
- Mechel, F. P. (Ed.). (2008). *Formulas of acoustics* (2nd ed.). Springer.
  ISBN 978-3-540-76832-6.
  [doi:10.1007/978-3-540-76833-3](https://doi.org/10.1007/978-3-540-76833-3).
  The formula compendium: layer reflection, multilayer schemes, diffuse-field
  absorption integrals and the empirical porous-material relations.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Bies, D. A., Hansen, C. H., & Howard, C. Q. (2017). *Engineering noise
  control* (5th ed.). CRC Press. ISBN 978-1-4987-2405-0.
  [doi:10.1201/9781351228152](https://doi.org/10.1201/9781351228152).
  The engineering-noise-control reference; its Appendix D collects the
  porous-material coefficient sets and layered-construction recursions, and
  its Chapter 6 covers the steady-state room field and the room constant.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md) and
  [Image sources and the steady-state room field](../buildings/rooms/room-image-sources.md).
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105-116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  The original one-parameter empirical model for fibrous absorbers.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Miki, Y. (1990). Acoustical properties of porous materials - Modifications
  of Delany-Bazley models. *Journal of the Acoustical Society of Japan (E)*,
  11(1), 19-24. [doi:10.1250/ast.11.19](https://doi.org/10.1250/ast.11.19).
  The positive-real refit of the Delany-Bazley regression.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Johnson, D. L., Koplik, J., & Dashen, R. (1987). Theory of dynamic
  permeability and tortuosity in fluid-saturated porous media. *Journal of
  Fluid Mechanics*, 176, 379-402.
  [doi:10.1017/S0022112087000727](https://doi.org/10.1017/S0022112087000727).
  The dynamic-tortuosity theory behind the Johnson-Champoux-Allard model.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Maa, D.-Y. (1998). Potential of microperforated panel absorber. *Journal
  of the Acoustical Society of America*, 104(5), 2861-2866.
  [doi:10.1121/1.423870](https://doi.org/10.1121/1.423870).
  The exact microperforated-panel impedance, its end corrections and the
  design formulas.
  Cited by [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md) and
  [Metamaterial Absorbers](../materials/absorbers/metamaterial-absorbers.md).
- International Organization for Standardization. (1998). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 2: Transfer-function method* (ISO 10534-2:1998; adopted in
  Europe as EN ISO 10534-2:2001; since revised as
  [ISO 10534-2:2023](https://www.iso.org/standard/81294.html)).
  [iso.org catalogue](https://www.iso.org/standard/22851.html).
  The two-microphone transfer-function method and its plane-wave limits.
  Cited by [Impedance Tube](../materials/absorbers/impedance-tube.md).
- ASTM International. (2019). *Standard test method for normal incidence
  determination of porous material acoustical properties based on the
  transfer matrix method* (ASTM E2611-19, the edition implemented here;
  since revised as [ASTM E2611-24](https://store.astm.org/e2611-24.html)).
  [ASTM store](https://store.astm.org/e2611-19.html).
  The four-microphone transfer-matrix transmission-loss method.
  Cited by [Impedance Tube](../materials/absorbers/impedance-tube.md).
- International Organization for Standardization. (2018). *Acoustics —
  Determination of airflow resistance — Part 1: Static airflow method*
  (ISO 9053-1:2018).
  [iso.org catalogue](https://www.iso.org/standard/69869.html).
  The static airflow-resistance method and its reference velocity.
  Cited by [Airflow Resistance](../materials/absorbers/airflow-resistance.md).
- International Organization for Standardization. (2020). *Acoustics —
  Determination of airflow resistance — Part 2: Alternating airflow method*
  (ISO 9053-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/76744.html).
  The alternating airflow-resistance method with the Annex A effective ratio
  of specific heats.
  Cited by [Airflow Resistance](../materials/absorbers/airflow-resistance.md).
- International Organization for Standardization. (1996). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 1: Method using standing wave ratio* (ISO 10534-1:1996;
  implemented as its European adoption BS EN ISO 10534-1:2001).
  [iso.org catalogue](https://www.iso.org/standard/18603.html).
  The standing-wave-ratio method.
  Cited by [Impedance Tube](../materials/absorbers/impedance-tube.md).
- International Organization for Standardization. (1997). *Acoustics — Sound
  absorbers for use in buildings — Rating of sound absorption*
  (ISO 11654:1997).
  [iso.org catalogue](https://www.iso.org/standard/19583.html).
  The weighted sound-absorption rating $\alpha_\mathrm{w}$, its shape indicators and
  the absorption class.
  Cited by
  [Sound Absorption Measurement and Rating](../materials/absorbers/absorption-measurement.md).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 2: Sound absorption* (ISO 12999-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/68749.html).
  The reproducibility and repeatability uncertainties of the
  reverberation-room quantities and their single-number ratings.
  Cited by
  [Sound Absorption Measurement and Rating](../materials/absorbers/absorption-measurement.md).
- International Organization for Standardization. (2004). *Acoustics —
  Sound-scattering properties of surfaces — Part 1: Measurement of the
  random-incidence scattering coefficient in a reverberation room*
  (ISO 17497-1:2004+A1:2014, the edition implemented here).
  [iso.org catalogue](https://www.iso.org/standard/31397.html).
  The turntable scattering-coefficient method.
  Cited by [Diffusers and Their Coefficients](../materials/diffusers/diffusers.md).
- International Organization for Standardization. (2012). *Acoustics —
  Sound-scattering properties of surfaces — Part 2: Measurement of the
  directional diffusion coefficient in a free field* (ISO 17497-2:2012).
  [iso.org catalogue](https://www.iso.org/standard/55293.html).
  The goniometer diffusion-coefficient method.
  Cited by [Diffusers and Their Coefficients](../materials/diffusers/diffusers.md).
- International Organization for Standardization. (2002). *Acoustics —
  Measurement of sound absorption properties of road surfaces in situ —
  Part 1: Extended surface method* (ISO 13472-1:2002, the edition
  implemented here; since revised as ISO 13472-1:2022).
  [iso.org catalogue](https://www.iso.org/standard/35387.html).
  The subtraction technique with the Adrienne window and the sampled-area
  radius.
  Cited by [In-situ Road-Surface Absorption](../materials/surfaces/road-absorption.md).
- International Organization for Standardization. (2010). *Acoustics —
  Measurement of sound absorption properties of road surfaces in situ —
  Part 2: Spot method for reflective surfaces* (ISO 13472-2:2010, the
  edition implemented here; since revised as ISO 13472-2:2025).
  [iso.org catalogue](https://www.iso.org/standard/32304.html).
  The spot-tube method and its plane-wave and spacing limits.
  Cited by [In-situ Road-Surface Absorption](../materials/surfaces/road-absorption.md).

## Building acoustics

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The comprehensive treatment of airborne and impact sound insulation:
  measurement chains, flanking transmission and the EN 12354 prediction
  framework.
  Cited by [Field Insulation Measurement (ISO 16283)](../buildings/insulation/insulation-field.md),
  [Laboratory Insulation Measurement](../buildings/insulation/insulation-lab.md),
  [Predicting Sound Insulation (EN 12354)](../buildings/design/insulation-prediction.md),
  [Bending-wave transmission at plate junctions](../vibration/structural/junction-transmission.md) and
  [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md).
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  A compact textbook on sound transmission in buildings, from single and
  double constructions to floating floors.
  Cited by [Field Insulation Measurement (ISO 16283)](../buildings/insulation/insulation-field.md),
  [Laboratory Insulation Measurement](../buildings/insulation/insulation-lab.md),
  [Dynamic stiffness of resilient materials](../materials/resilient/dynamic-stiffness.md) and
  [Predicting Panel Sound Insulation](../buildings/design/panel-sound-insulation.md).
- International Organization for Standardization. (2020). *Acoustics —
  Rating of sound insulation in buildings and of building elements — Part 1:
  Airborne sound insulation* (ISO 717-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/77435.html).
  The reference-curve rating and the spectrum adaptation terms C and Ctr.
  Cited by [Field Insulation Measurement (ISO 16283)](../buildings/insulation/insulation-field.md).
- International Organization for Standardization. (2014). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 1: Airborne sound insulation* (ISO 16283-1:2014).
  [iso.org catalogue](https://www.iso.org/standard/55997.html).
  The field airborne measurement method.
  Cited by [Field Insulation Measurement (ISO 16283)](../buildings/insulation/insulation-field.md).
- International Organization for Standardization. (1989). *Acoustics —
  Determination of dynamic stiffness — Part 1: Materials used under floating
  floors in dwellings* (ISO 9052-1:1989).
  [iso.org catalogue](https://www.iso.org/standard/16620.html).
  The resonance method for the dynamic stiffness per unit area, identical to
  EN 29052-1.
  Cited by [Dynamic stiffness of resilient materials](../materials/resilient/dynamic-stiffness.md).

## Structure-borne sound

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  The standard monograph on structural vibration and its radiation:
  mobilities, power flow, vibration isolation, radiation efficiency and
  transmission across junctions.
  Cited by [Mechanical mobility and the FRF family](../vibration/structural/mechanical-mobility.md),
  [Dynamic transfer stiffness of resilient elements](../vibration/structural/transfer-stiffness.md),
  [Sound power from surface vibration](../devices/emission/vibration-sound-power.md),
  [Structure-borne sound power of building equipment](../buildings/design/structure-borne-power.md),
  [Installed structure-borne sound from equipment](../buildings/design/installed-structure-borne.md)
  and [Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).
- Cremer, L., Heckl, M., & Ungar, E. E. (1973). *Structure-borne sound:
  Structural vibrations and sound radiation at audio frequencies* (1st ed.).
  Springer. ISBN 978-3-540-06002-4.
  [doi:10.1007/978-3-662-10118-6](https://doi.org/10.1007/978-3-662-10118-6).
  The original derivation of the wave parameters χ and ψ and the bending-wave
  transmission coefficients for junctions of plates.
  Cited by [Bending-wave transmission at plate junctions](../vibration/structural/junction-transmission.md).
- Craik, R. J. M. (1996). *Sound transmission through buildings using
  statistical energy analysis*. Gower. ISBN 978-0-566-07572-5.
  The SEA treatment of airborne and structure-borne transmission in buildings,
  with the tabulated bending-wave transmission coefficients for X, T, L and
  in-line junctions.
  Cited by [Bending-wave transmission at plate junctions](../vibration/structural/junction-transmission.md).
- International Organization for Standardization. (2011). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 1: Basic terms and definitions, and transducer specifications*
  (ISO 7626-1:2011).
  [iso.org catalogue](https://www.iso.org/standard/50426.html).
  The FRF family and its free/blocked distinctions.
  Cited by [Mechanical mobility and the FRF family](../vibration/structural/mechanical-mobility.md).
- International Organization for Standardization. (2015). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 2: Measurements using single-point translation excitation with an
  attached vibration exciter* (ISO 7626-2:2015).
  [iso.org catalogue](https://www.iso.org/standard/62483.html).
  The attached-exciter measurement method and its acceptance criteria.
  Cited by [Mechanical mobility and the FRF family](../vibration/structural/mechanical-mobility.md).
- International Organization for Standardization. (2008). *Acoustics and
  vibration — Laboratory measurement of vibro-acoustic transfer properties of
  resilient elements — Part 1: Principles and guidelines* (ISO 10846-1:2008).
  [iso.org catalogue](https://www.iso.org/standard/38936.html).
  The blocking-force idealisation behind the dynamic transfer stiffness.
  Cited by [Dynamic transfer stiffness of resilient elements](../vibration/structural/transfer-stiffness.md).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 1: Survey method using a fixed radiation
  factor* (ISO/TS 7849-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40537.html).
  The upper-limit sound power from surface velocity with ε = 1.
  Cited by [Sound power from surface vibration](../devices/emission/vibration-sound-power.md).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 2: Engineering method including determination
  of the adequate radiation factor* (ISO/TS 7849-2:2009).
  [iso.org catalogue](https://www.iso.org/standard/40538.html).
  The engineering method with a measured band-wise radiation factor.
  Cited by [Sound power from surface vibration](../devices/emission/vibration-sound-power.md).
- International Organization for Standardization. (1996). *Acoustics —
  Characterization of sources of structure-borne sound with respect to sound
  radiation from connected structures — Measurement of velocity at the
  contact points of machinery when resiliently mounted* (ISO 9611:1996).
  [iso.org catalogue](https://www.iso.org/standard/17424.html).
  The free-velocity characterization of resiliently mounted sources.
  Cited by [Structure-borne sound power of building equipment](../buildings/design/structure-borne-power.md).

## Outdoor and environmental sound

- Salomons, E. M. (2001). *Computational atmospheric acoustics*. Kluwer
  Academic Publishers. ISBN 978-1-4020-0390-5.
  [doi:10.1007/978-94-010-0660-6](https://doi.org/10.1007/978-94-010-0660-6).
  The wave-based theory of outdoor sound (parabolic equation, fast field
  program, refraction, turbulence) behind the engineering approximations of
  ISO 9613-2.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md) and
  [Spherical ground effect and advanced barriers](../environment/propagation/ground-barriers.md).
- Attenborough, K., & Van Renterghem, T. (2021). *Predicting outdoor sound*
  (2nd ed.). CRC Press.
  [doi:10.1201/9780429470806](https://doi.org/10.1201/9780429470806).
  Ground impedance models, the spherical-wave reflection coefficient behind
  the ground dip, meteorological effects on barriers, and the chapter 4
  pressure-velocity FDTD reference model.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md),
  [Spherical ground effect and advanced barriers](../environment/propagation/ground-barriers.md),
  [Porous and Multilayer Absorbers](../materials/absorbers/porous-absorbers.md) and
  [2D FDTD wave simulation](../simulation/fdtd-simulation.md).
- Maekawa, Z. (1968). Noise reduction by screens. *Applied Acoustics*, 1(3),
  157-173.
  [doi:10.1016/0003-682X(68)90020-0](https://doi.org/10.1016/0003-682X(68)90020-0).
  The screen-attenuation chart against Fresnel number that barrier
  engineering formulas descend from.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md) and
  [Spherical ground effect and advanced barriers](../environment/propagation/ground-barriers.md).
- Kurze, U. J., & Anderson, G. S. (1971). Sound attenuation by barriers.
  *Applied Acoustics*, 4(1), 35-53.
  [doi:10.1016/0003-682X(71)90024-7](https://doi.org/10.1016/0003-682X(71)90024-7).
  The closed-form fit to Maekawa's chart in the Fresnel number.
  Cited by [Spherical ground effect and advanced barriers](../environment/propagation/ground-barriers.md).
- Hadden, W. J., & Pierce, A. D. (1981). Sound diffraction around screens and
  wedges for arbitrary point source locations. *Journal of the Acoustical
  Society of America*, 69(5), 1266-1276.
  [doi:10.1121/1.385809](https://doi.org/10.1121/1.385809).
  The exact wedge-diffraction solution whose flat-wedge (thin-screen) limit
  the barrier insertion loss uses.
  Cited by [Spherical ground effect and advanced barriers](../environment/propagation/ground-barriers.md).
- Kephalopoulos, S., Paviotti, M., & Anfosso-Lédée, F. (2012). *Common noise
  assessment methods in Europe (CNOSSOS-EU)* (EUR 25379 EN). Publications
  Office of the European Union.
  [doi:10.2788/31776](https://doi.org/10.2788/31776),
  [JRC repository](https://publications.jrc.ec.europa.eu/repository/handle/JRC72550).
  The common EU noise-mapping framework, contrasted with ISO 9613-2; its
  flow-resistivity ground classes are reused by the rotorcraft ground effect.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md) and
  [Rotorcraft noise](../aircraft/rotorcraft-noise.md).
- International Organization for Standardization. (1993). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 1: Calculation of
  the absorption of sound by the atmosphere* (ISO 9613-1:1993).
  [iso.org catalogue](https://www.iso.org/standard/17426.html).
  The pure-tone atmospheric attenuation coefficient.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md).
- International Organization for Standardization. (1996). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 2: General method
  of calculation* (ISO 9613-2:1996; revised in 2024, the 1996 method is the
  implemented one).
  [iso.org catalogue](https://www.iso.org/standard/20649.html).
  The implemented outdoor attenuation chain.
  Cited by [Outdoor Sound Propagation](../environment/propagation/outdoor-propagation.md).
- International Organization for Standardization. (2016). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 1:
  Basic quantities and assessment procedures* (ISO 1996-1:2016).
  [iso.org catalogue](https://www.iso.org/standard/59765.html).
  The environmental rating framework and its Table A.1 category adjustments.
  Cited by [Impulsive-sound prominence](../environment/assessment/impulsive-sound.md).
- International Organization for Standardization. (2017). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 2:
  Determination of sound pressure levels* (ISO 1996-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/59766.html).
  The environmental measurement standard: its Annex J adopts the engineering
  method for tonal audibility, and the audibility criterion IEC 61400-11
  reuses comes from the Annex C of its 2007 edition.
  Cited by [Objective audibility of tones](../perception/psychoacoustics/tone-audibility.md) and
  [Wind-turbine noise](../environment/sources/wind-turbine-noise.md).
- Nordtest. (2002). *Acoustics: Prominence of impulsive sounds and for
  adjustment of LAeq* (Nordtest Method NT ACOU 112).
  [nordtest.info](https://www.nordtest.info/wp/2002/05/01/acoustics-prominence-of-impulsive-sounds-and-for-adjustment-of-laeq-nt-acou-112/).
  The freely downloadable onset-rate prominence method.
  Cited by [Impulsive-sound prominence](../environment/assessment/impulsive-sound.md).
- International Organization for Standardization. (2022). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 3:
  Objective method for the measurement of prominence of impulsive sounds and
  for adjustment of LAeq* (ISO/PAS 1996-3:2022).
  [iso.org catalogue](https://www.iso.org/standard/77035.html).
  The objective onset-analysis method implemented by
  `environment.assessment.impulsive_sound`: the level history LpAF (Clause 4), the onset
  detection and least-squares onset rate (Clauses 3.3-3.5), the prominence
  (Clause 5, Formula 2) and the adjustment KI (Clause 6, Formula 3).
  Cited by [Impulsive-sound prominence](../environment/assessment/impulsive-sound.md).
- International Electrotechnical Commission. (2018). *Wind turbines —
  Part 11: Acoustic noise measurement techniques*
  (IEC 61400-11:2012+AMD1:2018 CSV).
  [IEC webstore](https://webstore.iec.ch/en/publication/63367).
  The apparent sound power geometry, wind-speed binning and tonal
  audibility of wind turbines.
  Cited by [Wind-turbine noise](../environment/sources/wind-turbine-noise.md).
- International Electrotechnical Commission. (2005). *Wind turbines —
  Part 14: Declaration of apparent sound power level and tonality values*
  (IEC TS 61400-14:2005).
  [IEC webstore](https://webstore.iec.ch/en/publication/5432).
  Declared values and their uncertainty for a batch of turbines.
  Cited by [Wind-turbine noise](../environment/sources/wind-turbine-noise.md).

## Aircraft noise

- International Civil Aviation Organization. (2017). *Annex 16 to the
  Convention on International Civil Aviation: Environmental protection —
  Volume I: Aircraft noise* (8th ed.).
  [ICAO store](https://store.icao.int/en/annex-16-environmental-protection-volume-i-aircraft-noise).
  The aircraft noise-certification standard whose Appendix 2 defines the
  EPNL procedure.
  Cited by [Aircraft noise](../aircraft/aircraft-noise.md).
- International Civil Aviation Organization. (2018). *Environmental technical
  manual — Volume I: Procedures for the noise certification of aircraft*
  (Doc 9501, 3rd ed.).
  [ICAO store](https://store.icao.int/en/environmental-technical-manual-volume-1-procedures-for-the-noise-certification-of-aircraft-doc-9501-1).
  The certification guidance whose worked examples (tone correction,
  integrated-method EPNL) serve as numeric oracles.
  Cited by [Aircraft noise](../aircraft/aircraft-noise.md).
- International Electrotechnical Commission. (1995). *Electroacoustics —
  Instruments for measurement of aircraft noise — Performance requirements for
  systems to measure one-third-octave-band sound pressure levels in noise
  certification of transport-category aeroplanes* (IEC 61265:1995; since
  revised as [IEC 61265:2018](https://webstore.iec.ch/en/publication/32635),
  the 1995 edition is the implemented one).
  [IEC webstore](https://webstore.iec.ch/en/publication/5076).
  The aircraft-noise measurement-system performance tolerances.
  Cited by [Aircraft noise](../aircraft/aircraft-noise.md).
- SAE International. (2013). *Application of pure-tone atmospheric absorption
  losses to one-third octave-band data* (SAE ARP 5534, reaffirmed 2021).
  [sae.org](https://www.sae.org/standards/content/arp5534/).
  The SAE-Method one-third-octave-band atmospheric absorption for aircraft
  flyover spectra.
  Cited by [Aircraft noise](../aircraft/aircraft-noise.md).
- SAE International. (2012). *Standard values of atmospheric absorption as a
  function of temperature and humidity* (SAE ARP 866B, stabilized 2012).
  [sae.org](https://www.sae.org/standards/content/arp866b/).
  The predecessor SAE atmospheric-absorption practice, source of the older
  50 dB-limited Approximate Method.
  Cited by [Aircraft noise](../aircraft/aircraft-noise.md).
- SAE International. (2006). *Method for predicting lateral attenuation of
  airplane noise* (SAE AIR 5662).
  [sae.org](https://www.sae.org/standards/content/air5662/).
  The soft-ground lateral-attenuation model adopted by ECAC Doc 29.
  Cited by [Airport noise](../aircraft/airport-noise.md).
- European Civil Aviation Conference. (2016). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 4th ed.),
  Volume 2: Technical guide.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf).
  The European airport noise-contour method: NPD interpolation and the
  single-event segment calculation.
  Cited by [Airport noise](../aircraft/airport-noise.md).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 5th ed.),
  Volume 3: Reference cases and verification framework.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_29_5th_Edition-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_NOISE_CONTOURS_AROUND_CIVIL_AIRPORTS-Volume_3-REFERENCE_CASES_AND_VERIFICATION_FRAMEWORK.pdf).
  The reference cases and workbook used to validate the single-event chain.
  Cited by [Airport noise](../aircraft/airport-noise.md).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing rotorcraft noise contours* (ECAC.CEAC Doc 32, 1st ed.).
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_32-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_ROTORCRAFT_NOISE_CONTOURS.pdf).
  The standard rotorcraft contour method built on the noise hemisphere.
  Cited by [Rotorcraft noise](../aircraft/rotorcraft-noise.md).
- Olsen, H., Tuinstra, M., & van Oosten, N. (2024). *Rotorcraft noise
  modelling guidance* (Research Project NOISE SC01, deliverable D1.5d,
  contract EASA.2020.FC.06). European Union Aviation Safety Agency.
  [EASA project page](https://www.easa.europa.eu/en/research-projects/environmental-research-rotorcraft-noise),
  [free PDF](https://www.easa.europa.eu/en/downloads/132005/en).
  The NORAH2 equation-level modelling guidance, whose tables and reference
  hemispheres serve as oracles.
  Cited by [Rotorcraft noise](../aircraft/rotorcraft-noise.md).
- Chien, C. F., & Soroka, W. W. (1975). Sound propagation along an impedance
  plane. *Journal of Sound and Vibration*, 43(1), 9-20.
  [doi:10.1016/0022-460X(75)90200-X](https://doi.org/10.1016/0022-460X(75)90200-X).
  The two-ray interference solution over an impedance plane behind the
  rotorcraft ground effect.
  Cited by [Rotorcraft noise](../aircraft/rotorcraft-noise.md).
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105-116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  The one-parameter flow-resistivity ground-impedance model.
  Cited by [Rotorcraft noise](../aircraft/rotorcraft-noise.md).

## Underwater sound

- Urick, R. J. (1983). *Principles of underwater sound* (3rd ed.).
  McGraw-Hill; reprinted 1996 by Peninsula Publishing.
  ISBN 978-0-932146-62-5.
  [Open Library record](https://openlibrary.org/books/OL9317725M).
  The classic monograph on underwater sound: level conventions, ship
  radiated noise and the sonar-equation framework.
  Cited by [Underwater acoustics](../underwater/underwater-acoustics.md) and
  [Underwater sound propagation](../underwater/underwater-propagation.md).
- Ainslie, M. A. (2010). *Principles of sonar performance modelling*.
  Springer.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  The systematic treatment of underwater acoustical quantities in the line
  that ISO 18405 standardised, the Weston energy-flux propagation regimes of
  shallow water, the sonar equations with seven fully numeric worked examples,
  and the orca audiogram.
  Cited by [Underwater acoustics](../underwater/underwater-acoustics.md),
  [Underwater sound propagation](../underwater/underwater-propagation.md)
  and [Marine-mammal noise exposure](../underwater/marine-mammal-exposure.md).
- Medwin, H., & Clay, C. S. (1998). *Fundamentals of acoustical oceanography*.
  Academic Press. ISBN 978-0-12-487570-8.
  [Publisher page](https://shop.elsevier.com/books/fundamentals-of-acoustical-oceanography/medwin/978-0-12-487570-8).
  Ocean acoustics from first principles; the fluid-fluid Rayleigh
  reflection coefficient of the seabed model.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Jensen, F. B., Kuperman, W. A., Porter, M. B., & Schmidt, H. (2011).
  *Computational ocean acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-1-4419-8678-8](https://doi.org/10.1007/978-1-4419-8678-8).
  The reference monograph on numerical propagation: normal modes, ray
  tracing and the parabolic equation.
  Cited by [Underwater propagation solvers](../underwater/underwater-solvers.md).
- Munk, W. H. (1974). Sound channel in an exponentially stratified ocean,
  with application to SOFAR. *The Journal of the Acoustical Society of
  America*, 55(2), 220-226.
  [doi:10.1121/1.1914492](https://doi.org/10.1121/1.1914492).
  The canonical deep-water sound-speed profile used by the solver examples.
  Cited by [Underwater propagation solvers](../underwater/underwater-solvers.md).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements: Part I: Pure water and magnesium sulfate contributions.
  *The Journal of the Acoustical Society of America*, 72(3), 896-907.
  [doi:10.1121/1.388170](https://doi.org/10.1121/1.388170).
  The pure-water and magnesium-sulfate halves of the reference seawater
  absorption model.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements. Part II: Boric acid contribution and equation for total
  absorption. *The Journal of the Acoustical Society of America*, 72(6),
  1879-1890.
  [doi:10.1121/1.388673](https://doi.org/10.1121/1.388673).
  The boric-acid term and the complete total-absorption equation.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Ainslie, M. A., & McColm, J. G. (1998). A simplified formula for viscous and
  chemical absorption in sea water. *The Journal of the Acoustical Society of
  America*, 103(3), 1671-1672.
  [doi:10.1121/1.421258](https://doi.org/10.1121/1.421258).
  The legible simplified seawater absorption formula.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Thorp, W. H. (1967). Analytic description of the low-frequency attenuation
  coefficient. *The Journal of the Acoustical Society of America*, 42(1), 270.
  [doi:10.1121/1.1910566](https://doi.org/10.1121/1.1910566).
  The frequency-only low-frequency absorption formula.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Chen, C.-T., & Millero, F. J. (1977). Speed of sound in seawater at high
  pressures. *The Journal of the Acoustical Society of America*, 62(5),
  1129-1135.
  [doi:10.1121/1.381646](https://doi.org/10.1121/1.381646).
  The UNESCO international-standard sound-speed equation.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Wong, G. S. K., & Zhu, S. (1995). Speed of sound in seawater as a function
  of salinity, temperature, and pressure. *The Journal of the Acoustical
  Society of America*, 97(3), 1732-1736.
  [doi:10.1121/1.413048](https://doi.org/10.1121/1.413048).
  The ITS-90 recast of the UNESCO sound-speed coefficients, the implemented
  form.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Del Grosso, V. A. (1974). New equation for the speed of sound in natural
  waters (with comparisons to other equations). *The Journal of the
  Acoustical Society of America*, 56(4), 1084-1091.
  [doi:10.1121/1.1903388](https://doi.org/10.1121/1.1903388).
  The alternative pressure-based sound-speed equation.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans.
  *The Journal of the Acoustical Society of America*, 70(3), 807-812.
  [doi:10.1121/1.386920](https://doi.org/10.1121/1.386920).
  The depth-based nine-term sound-speed equation.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Leroy, C. C., & Parthiot, F. (1998). Depth-pressure relationships in the
  oceans and seas. *The Journal of the Acoustical Society of America*, 103(3),
  1346-1352.
  [doi:10.1121/1.421275](https://doi.org/10.1121/1.421275).
  The depth-to-pressure conversion used by the sound-speed equations.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and
  sources. *The Journal of the Acoustical Society of America*, 34(12),
  1936-1956.
  [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155).
  The classic ambient-noise survey behind the wind and thermal spectrum
  components.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Carey, W. M., & Evans, R. B. (2011). *Ocean ambient noise: Measurement and
  theory*. Springer.
  [doi:10.1007/978-1-4419-7832-5](https://doi.org/10.1007/978-1-4419-7832-5).
  The modern treatment of ocean ambient noise: the wind "rule of fives" and
  the Mellen thermal-noise derivation.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- MacGillivray, A., & de Jong, C. (2021). A reference spectrum model for
  estimating source levels of marine shipping based on automated
  identification system data. *Journal of Marine Science and Engineering*,
  9(4), 369.
  [doi:10.3390/jmse9040369](https://doi.org/10.3390/jmse9040369).
  The open-access JOMOPANS-ECHO ship source-level model and its reference
  calculator.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- Wales, S. C., & Heitmeyer, R. M. (2002). An ensemble source spectra model
  for merchant ship-radiated noise. *The Journal of the Acoustical Society of
  America*, 111(3), 1211-1231.
  [doi:10.1121/1.1427355](https://doi.org/10.1121/1.1427355).
  The ensemble merchant-ship source-spectrum model.
  Cited by [Underwater sound propagation](../underwater/underwater-propagation.md).
- National Marine Fisheries Service (2018). *2018 Revision to: Technical
  Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal
  Hearing (Version 2.0)*. NOAA Technical Memorandum NMFS-OPR-59.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2023-05/TECHMEMOGuidance508.pdf).
  The auditory weighting parameters and PTS onset thresholds of the 2018
  guidance, with the Appendix D worked example.
  Cited by [Marine-mammal noise exposure](../underwater/marine-mammal-exposure.md).
- National Marine Fisheries Service (2024). *2024 Update to: Technical
  Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal
  Hearing (Version 3.0)*. NOAA Technical Memorandum NMFS-OPR-71.
  [NOAA Fisheries](https://www.fisheries.noaa.gov/s3/2024-11/Tech_Memo-Guidance_-3.0-_OCT-2024-508_OPR1.pdf).
  The current U.S. guidance: revised weighting parameters and the auditory
  injury onset criteria that supersede the 2018 PTS thresholds.
  Cited by [Marine-mammal noise exposure](../underwater/marine-mammal-exposure.md).
- Southall, B. L., Finneran, J. J., Reichmuth, C., Nachtigall, P. E.,
  Ketten, D. R., Bowles, A. E., Ellison, W. T., Nowacek, D. P., &
  Tyack, P. L. (2019). Marine mammal noise exposure criteria: Updated
  scientific recommendations for residual hearing effects. *Aquatic Mammals*,
  45(2), 125-232.
  [doi:10.1578/AM.45.2.2019.125](https://doi.org/10.1578/AM.45.2.2019.125).
  The peer-reviewed hearing groups, group audiograms and TTS/PTS onset
  criteria, with the errata of 45(5), 569-572.
  Cited by [Marine-mammal noise exposure](../underwater/marine-mammal-exposure.md).
- Finneran, J. J. (2016). *Auditory weighting functions and TTS/PTS exposure
  functions for marine mammals exposed to underwater noise*. Technical Report
  3026, SSC Pacific.
  [Report page](https://apps.dtic.mil/sti/citations/AD1026445).
  The band-pass weighting-function form and the audiogram equation that the
  NMFS and Southall criteria both adopt.
  Cited by [Marine-mammal noise exposure](../underwater/marine-mammal-exposure.md).

## Wave simulation

- Williams, E. G. (1999). *Fourier acoustics: Sound radiation and
  nearfield acoustical holography*. Academic Press.
  [doi:10.1016/B978-0-12-753960-7.X5000-1](https://doi.org/10.1016/B978-0-12-753960-7.X5000-1).
  The Helmholtz integral equation behind the near-to-far-field
  transformation, with the outgoing free-space Green function and the
  far-field limit.
  Cited by [2D FDTD wave simulation](../simulation/fdtd-simulation.md).
- Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
  velocity-stress finite-difference method. *Geophysics*, 51(4), 889-901.
  [doi:10.1190/1.1442147](https://doi.org/10.1190/1.1442147).
  The elastic velocity-stress scheme on the fully staggered cell, its
  Courant bound and dispersion relations, and the liquid as the shear-free
  limit.
  Cited by [Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).
- Moczo, P., Kristek, J., Galis, M., Pazak, P., & Balazovjech, M. (2007).
  The finite-difference and finite-element modeling of seismic wave
  propagation and earthquake motion. *Acta Physica Slovaca*, 57(2),
  177-406. The heterogeneous effective grid parameters (harmonic shear
  modulus, arithmetic density) and the stress-imaging free surface.
  Cited by [Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).
- Brekhovskikh, L. M., & Godin, O. A. (1990). *Acoustics of layered media
  I: Plane and quasi-plane waves*. Springer.
  [doi:10.1007/978-3-642-52369-4](https://doi.org/10.1007/978-3-642-52369-4).
  The fluid-solid oracles: the oblique reflection coefficient with mode
  conversion, the exact Scholte characteristic equation and the three-media
  layer transmission.
  Cited by [Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).
- van Vossen, R., Robertsson, J. O. A., & Chapman, C. H. (2002).
  Finite-difference modeling of wave propagation in a fluid-solid
  configuration. *Geophysics*, 67(2), 618-624.
  [doi:10.1190/1.1468623](https://doi.org/10.1190/1.1468623).
  The fluid-solid benchmark of the staggered scheme: the effective-parameter
  averages, the soft-bed Scholte configuration and the
  points-per-wavelength rule for interface waves.
  Cited by [Elastic waves and fluid-solid coupling](../simulation/elastic-waves.md).

## Speech

- Houtgast, T., & Steeneken, H. J. M. (1985). A review of the MTF concept in
  room acoustics and its use for estimating speech intelligibility in
  auditoria. *The Journal of the Acoustical Society of America*, 77(3),
  1069-1077. [doi:10.1121/1.392224](https://doi.org/10.1121/1.392224).
  The modulation-transfer framework the Speech Transmission Index is built on.
  Cited by [Speech Transmission Index](../perception/speech/speech-transmission.md).
- French, N. R., & Steinberg, J. C. (1947). Factors governing the
  intelligibility of speech sounds. *The Journal of the Acoustical Society of
  America*, 19(1), 90-119.
  [doi:10.1121/1.1916407](https://doi.org/10.1121/1.1916407).
  The articulation-band experiments behind the band-importance function of the
  Speech Intelligibility Index.
  Cited by [Speech Intelligibility Index](../perception/speech/speech-intelligibility.md).
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2011). An
  algorithm for intelligibility prediction of time-frequency weighted noisy
  speech. *IEEE Transactions on Audio, Speech, and Language Processing*,
  19(7), 2125-2136.
  [doi:10.1109/TASL.2011.2114881](https://doi.org/10.1109/TASL.2011.2114881).
  STOI: the shared one-third-octave front end, the normalisation and
  signal-to-distortion clipping, and the per-band envelope correlation the
  index averages.
  Cited by [Objective Intelligibility (STOI & ESTOI)](../perception/speech/objective-intelligibility.md).
- Taal, C. H., Hendriks, R. C., Heusdens, R., & Jensen, J. (2010). A
  short-time objective intelligibility measure for time-frequency weighted
  noisy speech. *2010 IEEE International Conference on Acoustics, Speech and
  Signal Processing (ICASSP)*, 4214-4217.
  [doi:10.1109/ICASSP.2010.5495701](https://doi.org/10.1109/ICASSP.2010.5495701).
  The short conference version of STOI.
  Cited by [Objective Intelligibility (STOI & ESTOI)](../perception/speech/objective-intelligibility.md).
- Jensen, J., & Taal, C. H. (2016). An algorithm for predicting the
  intelligibility of speech masked by modulated noise maskers. *IEEE/ACM
  Transactions on Audio, Speech, and Language Processing*, 24(11), 2009-2022.
  [doi:10.1109/TASLP.2016.2585878](https://doi.org/10.1109/TASLP.2016.2585878).
  ESTOI: the row- and column-normalised short-time spectrogram and its
  spectral-correlation intermediate index.
  Cited by [Objective Intelligibility (STOI & ESTOI)](../perception/speech/objective-intelligibility.md).

## Psychoacoustics

- Moore, B. C. J. (2013). *An introduction to the psychology of hearing*
  (6th ed.). Brill.
  [doi:10.1163/9789004252424](https://doi.org/10.1163/9789004252424).
  The standard textbook on auditory perception; pages 76-77 give the
  Glasberg and Moore (1990) ERB_N auditory-filter bandwidth and the Cam
  (ERB_N number) frequency scale the loudness models are written on.
  Cited by [Advanced loudness](../perception/psychoacoustics/advanced-loudness.md).
- Fletcher, H., & Munson, W. A. (1933). Loudness, its definition, measurement
  and calculation. *The Journal of the Acoustical Society of America*, 5(2),
  82-108. [doi:10.1121/1.1915637](https://doi.org/10.1121/1.1915637).
  The original equal-loudness measurements whose 40-phon contour became the
  A-weighting curve.
  Cited by [Frequency Weighting (A, C, Z)](../signals/levels/weighting.md) and
  [Loudness](../perception/psychoacoustics/loudness.md).
- International Organization for Standardization. (2023). *Acoustics —
  Normal equal-loudness-level contours* (ISO 226:2023).
  [iso.org catalogue](https://www.iso.org/standard/83117.html).
  The modern equal-loudness contours, successors of the Fletcher-Munson
  curves.
  Cited by [Frequency Weighting (A, C, Z)](../signals/levels/weighting.md) and
  [Loudness](../perception/psychoacoustics/loudness.md).
- Fastl, H., & Zwicker, E. (2007). *Psychoacoustics: Facts and models*
  (3rd ed.). Springer.
  [doi:10.1007/978-3-540-68888-4](https://doi.org/10.1007/978-3-540-68888-4).
  The psychoacoustic-annoyance model and the closed-form fluctuation strength
  for amplitude-modulated broadband noise.
  Cited by [Psychoacoustic annoyance](../perception/psychoacoustics/psychoacoustic-annoyance.md),
  [Loudness](../perception/psychoacoustics/loudness.md) and [Sound Quality Metrics](../perception/psychoacoustics/sound-quality.md).
- Osses Vecchi, A., García León, R., & Kohlrausch, A. (2016). Modelling the
  sensation of fluctuation strength. *Proceedings of Meetings on Acoustics*,
  28, 050005. [doi:10.1121/2.0000410](https://doi.org/10.1121/2.0000410).
  The fluctuation-strength signal model and its Table 1 literature values.
  Cited by [Psychoacoustic annoyance](../perception/psychoacoustics/psychoacoustic-annoyance.md).
- Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2025).
  *SQAT: a sound quality analysis toolbox for MATLAB* (open-source software).
  [github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT),
  [doi:10.5281/zenodo.7934709](https://doi.org/10.5281/zenodo.7934709).
  The open MATLAB reference used as the numeric oracle for the
  fluctuation-strength cross-checks.
  Cited by [Psychoacoustic annoyance](../perception/psychoacoustics/psychoacoustic-annoyance.md).
- Ecma International. (2024). *ECMA-418-1: Psychoacoustic metrics for ITT
  equipment — Part 1: Prominent discrete tones* (3rd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-418-1_3rd_edition_december_2024.pdf).
  The freely downloadable tone-to-noise ratio and prominence ratio methods.
  Cited by [Prominent Discrete Tones](../perception/psychoacoustics/tone-prominence.md).
- Ecma International. (2025). *ECMA-74: Measurement of airborne noise emitted
  by information technology and telecommunications equipment* (22nd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-74_22nd_edition_december_2025.pdf).
  The freely downloadable parent emission standard whose Annex D delegates
  tone assessment to ECMA-418-1.
  Cited by [Prominent Discrete Tones](../perception/psychoacoustics/tone-prominence.md).
- International Organization for Standardization. (2016). *Acoustics —
  Objective method for assessing the audibility of tones in noise —
  Engineering method* (ISO/PAS 20065:2016; withdrawn, superseded by
  [ISO/TS 20065:2022](https://www.iso.org/standard/81518.html)).
  [iso.org catalogue](https://www.iso.org/standard/66941.html).
  The engineering method for the objective audibility of tones; the
  implementation follows the 2016 PAS edition.
  Cited by [Objective audibility of tones](../perception/psychoacoustics/tone-audibility.md).

## Hearing and hearing conservation

- International Organization for Standardization. (2017). *Acoustics —
  Statistical distribution of hearing thresholds related to age and gender*
  (ISO 7029:2017). [iso.org catalogue](https://www.iso.org/standard/42916.html).
  The age model of the hearing threshold and its population spread.
  Cited by [Hearing threshold](../perception/hearing/hearing-threshold.md).
- International Organization for Standardization. (2005). *Acoustics —
  Reference zero for the calibration of audiometric equipment — Part 7:
  Reference threshold of hearing under free-field and diffuse-field listening
  conditions* (ISO 389-7:2005).
  [iso.org catalogue](https://www.iso.org/standard/38976.html).
  The audiometric zero as a sound pressure level.
  Cited by [Hearing threshold](../perception/hearing/hearing-threshold.md).
- International Organization for Standardization. (2013). *Acoustics —
  Estimation of noise-induced hearing loss* (ISO 1999:2013).
  [iso.org catalogue](https://www.iso.org/standard/45103.html).
  The NIPTS model, its distribution and the HTLAN combination.
  Cited by [Noise-induced hearing loss](../perception/hearing/noise-induced-hearing-loss.md).
- Passchier-Vermeer, W. (1974). Hearing loss due to continuous exposure to
  steady-state broad-band noise. *The Journal of the Acoustical Society of
  America*, 56(5), 1585–1593.
  [doi:10.1121/1.1903482](https://doi.org/10.1121/1.1903482).
  A field study of the noise exposure-response relations later codified in
  ISO 1999.
  Cited by [Noise-induced hearing loss](../perception/hearing/noise-induced-hearing-loss.md).
- National Institute for Occupational Safety and Health. (1998). *Criteria for
  a recommended standard: Occupational noise exposure — Revised criteria 1998*
  (DHHS/NIOSH Publication No. 98-126).
  [doi:10.26616/NIOSHPUB98126](https://doi.org/10.26616/NIOSHPUB98126),
  [free PDF](https://www.cdc.gov/niosh/docs/98-126/pdfs/98-126.pdf).
  The freely available criteria document behind the 85 dB(A) recommended
  exposure limit and the hearing-conservation and fence discussion.
  Cited by [Noise-induced hearing loss](../perception/hearing/noise-induced-hearing-loss.md) and
  [Occupational noise exposure](../perception/hearing/occupational-exposure.md).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of occupational noise exposure — Engineering method*
  (ISO 9612:2009). [iso.org catalogue](https://www.iso.org/standard/41718.html).
  The three measurement strategies and the Annex C uncertainty budget.
  Cited by [Occupational noise exposure](../perception/hearing/occupational-exposure.md).
- European Parliament and Council. (2003). *Directive 2003/10/EC on the
  minimum health and safety requirements regarding the exposure of workers to
  the risks arising from physical agents (noise)*. Official Journal of the
  European Union.
  [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/dir/2003/10/oj/eng).
  The EU exposure action and limit values for occupational noise.
  Cited by [Occupational noise exposure](../perception/hearing/occupational-exposure.md).

## Human vibration

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Publisher page](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  The standard monograph on whole-body and hand-transmitted vibration: the
  biodynamics, discomfort and health-effect evidence behind the weightings,
  dose measures and exposure-response guidance of the vibration guides.
  Cited by [Human Vibration](../vibration/human/human-vibration.md) and
  [Multiple-shock whole-body vibration](../vibration/human/multiple-shock-vibration.md).
- Mansfield, N. J. (2004). *Human response to vibration*. CRC Press.
  ISBN 978-0-415-28239-0.
  [Publisher page](https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390).
  A compact modern textbook on the ISO 2631-1 and ISO 5349 evaluation chains,
  from perception and comfort to the occupational exposure limits.
  Cited by [Human Vibration](../vibration/human/human-vibration.md).

## Metrology

- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Guide to the expression of uncertainty in measurement* (JCGM
  100:2008, the GUM). BIPM.
  [doi:10.59161/JCGM100-2008E](https://doi.org/10.59161/JCGM100-2008E),
  [free PDF](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
  The law of propagation of uncertainty implemented by the uncertainty module.
  Cited by [Measurement uncertainty](../signals/metrology/gum-uncertainty.md).
- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Supplement 1 to the "Guide to the expression of uncertainty in
  measurement" — Propagation of distributions using a Monte Carlo method*
  (JCGM 101:2008). BIPM.
  [doi:10.59161/JCGM101-2008](https://doi.org/10.59161/JCGM101-2008),
  [free PDF](https://www.bipm.org/documents/20126/2071204/JCGM_101_2008_E.pdf).
  The Monte Carlo propagation of distributions implemented by the Monte Carlo
  uncertainty engine.
  Cited by [Measurement uncertainty](../signals/metrology/gum-uncertainty.md).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 1: Sound insulation* (ISO 12999-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/73930.html).
  The domain-specific reproducibility budget for building-acoustics
  single-number ratings, the companion to the general GUM machinery.
  Cited by [Measurement uncertainty](../signals/metrology/gum-uncertainty.md).
