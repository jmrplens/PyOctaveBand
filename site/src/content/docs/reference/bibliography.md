---
title: "Bibliography"
description: "The books and papers behind the guides, grouped by domain: every entry with a verified DOI or official publisher link and a note on what it supports."
---

Every guide on this site closes with two citation sections: a **References**
section listing the books and papers that support the physics on the page
(APA style, one bullet per source, each with a DOI or an official publisher
link, and half a sentence on what the entry supports), followed by a
**Standards** section naming the normative documents the page implements,
clause by clause. This page collects the References entries of all guides in
one place, grouped by domain: a curated reading list, and the single source of
truth for link checking. Each entry lists the guide pages that cite it; the
list grows as guides gain their References sections.

## General acoustics

- Kinsler, L. E., Frey, A. R., Coppens, A. B., & Sanders, J. V. (2000).
  *Fundamentals of acoustics* (4th ed.). Wiley. ISBN 978-0-471-84789-2.
  [Publisher page](https://www.wiley.com/en-us/Fundamentals+of+Acoustics%2C+4th+Edition-p-9780471847892).
  The standard first course in acoustics: plane and spherical waves, acoustic
  impedance and the level definitions assumed throughout the guides.
  Cited by [Integrated and Statistical Levels](/phonometry/guides/levels/).
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
  Cited by [Electroacoustics](/phonometry/guides/electroacoustics/) and
  [Sound Power](/phonometry/guides/sound-power/).

## Signal processing

- Oppenheim, A. V., & Schafer, R. W. (2010). *Discrete-time signal processing*
  (3rd ed.). Pearson. ISBN 978-0-13-198842-2.
  [Open Library record](https://openlibrary.org/isbn/9780131988422).
  The digital-filter theory behind the SOS cascades, the bilinear transform
  and the multirate decimation used by the filter banks.
  Cited by [Filter Banks](/phonometry/guides/filter-banks/) and
  [Block Processing](/phonometry/guides/block-processing/).
- Smith, J. O. *Introduction to digital filters with audio applications*
  (online book). Center for Computer Research in Music and Acoustics (CCRMA),
  Stanford University.
  [ccrma.stanford.edu/~jos/filters](https://ccrma.stanford.edu/~jos/filters/).
  Free companion treatment of digital-filter design and analysis, a good next
  step after the filter-bank guides.
  Cited by [Filter Banks](/phonometry/guides/filter-banks/).
- Bendat, J. S., & Piersol, A. G. (2010). *Random data: Analysis and
  measurement procedures* (4th ed.). Wiley. ISBN 978-0-470-24877-5.
  [doi:10.1002/9781118032428](https://doi.org/10.1002/9781118032428).
  The reference for the Welch spectral estimators and their statistical
  quality, and for the multiple-input/output coherence functions of
  Chapter 7 (multiple and partial coherence, conditioned spectra) with the
  Section 9.3 error formulas implemented by `miso_coherence`.
  Cited by [Calibrated spectral analysis](/phonometry/guides/spectral-analysis/)
  and [Multiple and partial coherence](/phonometry/guides/miso-coherence/).
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  *Proceedings of the IEEE*, 70(9), 1055-1096.
  [doi:10.1109/PROC.1982.12433](https://doi.org/10.1109/PROC.1982.12433).
  The multitaper method: Slepian tapers, eigenspectra and the adaptive
  weights implemented by `multitaper_psd`.
  Cited by [Calibrated spectral analysis](/phonometry/guides/spectral-analysis/).
- Percival, D. B., & Walden, A. T. (1993). *Spectral analysis for physical
  applications: Multitaper and conventional univariate techniques*.
  Cambridge University Press. ISBN 978-0-521-43541-3.
  [doi:10.1017/CBO9780511622762](https://doi.org/10.1017/CBO9780511622762).
  The multitaper development (Chapter 7) behind `multitaper_psd` and the
  Slepian-sequence eigenvalue tables that anchor its test oracle.
  Cited by [Calibrated spectral analysis](/phonometry/guides/spectral-analysis/).

## Measurement instrumentation

- International Electrotechnical Commission. (2014). *Electroacoustics —
  Octave-band and fractional-octave-band filters — Part 1: Specifications*
  (IEC 61260-1:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/5063).
  The base-10 band edges and the class acceptance masks of the fractional
  octave banks.
  Cited by [Filter Banks](/phonometry/guides/filter-banks/) and
  [Multichannel and Performance](/phonometry/guides/multichannel/).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 1: Specifications* (IEC 61672-1:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5708).
  The A/C/Z weightings, the exponential time weightings and the level
  metrics of the sound level meter, with the tolerance tables used for
  verification.
  Cited by [Integrated and Statistical Levels](/phonometry/guides/levels/),
  [Frequency Weighting (A, C, G, Z)](/phonometry/guides/weighting/),
  [Time Weighting and Integration](/phonometry/guides/time-weighting/) and
  [Multichannel and Performance](/phonometry/guides/multichannel/).
- International Electrotechnical Commission. (2013). *Electroacoustics —
  Sound level meters — Part 3: Periodic tests* (IEC 61672-3:2013).
  [IEC webstore](https://webstore.iec.ch/en/publication/5710).
  The periodic laboratory verification of a sound level meter.
  Cited by [Calibration and dBFS](/phonometry/guides/calibration/).
- International Electrotechnical Commission. (2017). *Electroacoustics —
  Sound calibrators* (IEC 60942:2017).
  [IEC webstore](https://webstore.iec.ch/en/publication/30045).
  The calibrator classes, level tolerances and the short-term stability
  criterion applied to calibration recordings.
  Cited by [Calibration and dBFS](/phonometry/guides/calibration/).
- International Electrotechnical Commission. (2014). *Sound system equipment —
  Part 4: Microphones* (IEC 60268-4:2014).
  [IEC webstore](https://webstore.iec.ch/en/publication/32039).
  The rated microphone characteristics: free-field sensitivity and its level
  re 1 V/Pa, the frequency response and the effective frequency range against
  the tolerance limits, the directional pattern and the directivity index, the
  overload sound pressure level, the equivalent sound pressure level due to
  inherent noise, and the rated impedances and power supply.
  Cited by [Electroacoustics](/phonometry/guides/electroacoustics/).
- International Electrotechnical Commission. (2007). *Sound system equipment —
  Part 5: Loudspeakers* (IEC 60268-5:2003+A1:2007).
  [IEC webstore](https://webstore.iec.ch/en/publication/1223).
  The rated loudspeaker characteristics: rated impedance, rated frequency
  range, characteristic sensitivity referred to 1 W at 1 m, the effective
  frequency range against the -10 dB band, the directivity index and the total
  harmonic distortion against frequency.
  Cited by [Electroacoustics](/phonometry/guides/electroacoustics/).
- International Electrotechnical Commission. (1982). *Scales and sizes for
  plotting frequency characteristics and polar diagrams* (IEC 60263:1982).
  [IEC webstore](https://webstore.iec.ch/en/publication/1218).
  The scale proportions of the characteristic graphs: one frequency decade
  equal to 25 dB on the ordinate, and the polar diagram on a 25 dB
  reference-circle radius.
  Cited by [Electroacoustics](/phonometry/guides/electroacoustics/).

## Sound power and intensity

- Fahy, F. J. (1995). *Sound intensity* (2nd ed.). E&FN Spon.
  ISBN 978-0-419-19810-9.
  [doi:10.4324/9780203475386](https://doi.org/10.4324/9780203475386).
  The monograph on sound energy flux: active and reactive intensity, the
  p-p estimator and its phase-mismatch error budget.
  Cited by [Sound Power](/phonometry/guides/sound-power/) and
  [Sound Intensity (p-p)](/phonometry/guides/intensity/).
- International Organization for Standardization. (2019). *Acoustics —
  Determination of sound power levels of noise sources — Guidelines for the
  use of basic standards* (ISO 3740:2019).
  [iso.org catalogue](https://www.iso.org/standard/45107.html).
  The selection guide for the sound-power family: grades, environments,
  source-size and background criteria.
  Cited by [Sound Power](/phonometry/guides/sound-power/).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for reverberation test
  rooms* (ISO 3741:2010).
  [iso.org catalogue](https://www.iso.org/standard/52053.html).
  The precision reverberation-room method.
  Cited by [Sound Power](/phonometry/guides/sound-power/).
- International Organization for Standardization. (2010). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Engineering methods for an essentially free
  field over a reflecting plane* (ISO 3744:2010).
  [iso.org catalogue](https://www.iso.org/standard/52055.html).
  The enveloping-surface engineering method.
  Cited by [Sound Power](/phonometry/guides/sound-power/).
- International Organization for Standardization. (2012). *Acoustics —
  Determination of sound power levels and sound energy levels of noise
  sources using sound pressure — Precision methods for anechoic rooms and
  hemi-anechoic rooms* (ISO 3745:2012).
  [iso.org catalogue](https://www.iso.org/standard/45362.html).
  The precision anechoic-room method.
  Cited by [Sound Power](/phonometry/guides/sound-power/).
- International Organization for Standardization. (1996). *Acoustics —
  Declaration and verification of noise emission values of machinery and
  equipment* (ISO 4871:1996).
  [iso.org catalogue](https://www.iso.org/standard/10868.html).
  The noise-emission declaration: the dual/single-number forms,
  L_WAd = L_WA + K_WA and the clause 6.2 verification.
  Cited by [Sound Power](/phonometry/guides/sound-power/).
- International Organization for Standardization. (1993). *Acoustics —
  Determination of sound power levels of noise sources using sound
  intensity — Part 1: Measurement at discrete points* (ISO 9614-1:1993).
  [iso.org catalogue](https://www.iso.org/standard/17427.html).
  The field indicators and the dynamic-capability criterion of intensity
  measurement.
  Cited by [Sound Intensity (p-p)](/phonometry/guides/intensity/).
- International Electrotechnical Commission. (1993). *Electroacoustics —
  Instruments for the measurement of sound intensity — Measurements with
  pairs of pressure sensing microphones* (IEC 61043:1993; adopted in Europe
  as EN 61043:1994).
  [IEC webstore](https://webstore.iec.ch/en/publication/4353).
  The p-p instrument standard: the cross-spectral estimator and the
  residual pressure-intensity index.
  Cited by [Sound Intensity (p-p)](/phonometry/guides/intensity/).

## Room acoustics

- Kuttruff, H. (2016). *Room acoustics* (6th ed.). CRC Press.
  [doi:10.1201/9781315372150](https://doi.org/10.1201/9781315372150).
  The reference monograph on sound fields in rooms: statistical decay
  theory, the Schroeder frequency, absorption and the perceptual room
  parameters.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/),
  [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/) and
  [Sound absorption in enclosed spaces](/phonometry/guides/enclosed-space-absorption/).
- Sabine, W. C. (1922). *Collected papers on acoustics*. Harvard University
  Press.
  [Free scan at the Internet Archive](https://archive.org/details/collectedpaperso00sabi).
  The founding reverberation experiments and the Sabine law.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Eyring, C. F. (1930). Reverberation time in "dead" rooms. *The Journal of
  the Acoustical Society of America*, 1(2A), 217-241.
  [doi:10.1121/1.1915175](https://doi.org/10.1121/1.1915175).
  The mean-free-path reverberation formula for strongly absorbing rooms.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Millington, G. (1932). A modified formula for reverberation. *The Journal
  of the Acoustical Society of America*, 4(1), 69-82.
  [doi:10.1121/1.1915588](https://doi.org/10.1121/1.1915588).
  The per-surface logarithmic reverberation formula.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Fitzroy, D. (1959). Reverberation formula which seems to be more accurate
  with nonuniform distribution of absorption. *The Journal of the
  Acoustical Society of America*, 31(7), 893-897.
  [doi:10.1121/1.1907814](https://doi.org/10.1121/1.1907814).
  The axial reverberation formula for anisotropic absorption.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Arau-Puchades, H. (1988). An improved reverberation formula. *Acustica*,
  65(4), 163-180.
  [Publisher record at Ingenta](https://www.ingentaconnect.com/content/dav/aaua/1988/00000065/00000004/art00003).
  The geometric-mean refinement of the axial reverberation formula.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Schroeder, M. R. (1965). New method of measuring reverberation time.
  *The Journal of the Acoustical Society of America*, 37(3), 409-412.
  [doi:10.1121/1.1909343](https://doi.org/10.1121/1.1909343).
  The backward integration of the squared impulse response into a decay
  curve.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- Hak, C. C. J. M., Wenmaekers, R. H. C., & van Luxemburg, L. C. J. (2012).
  Measuring room impulse responses: Impact of the decay range on derived
  room acoustic parameters. *Acta Acustica united with Acustica*, 98(6),
  907-915. [doi:10.3813/aaa.918574](https://doi.org/10.3813/aaa.918574).
  The impulse-to-noise-ratio (INR) analysis of decay-range requirements.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- Everest, F. A. (2001). *Master handbook of acoustics* (4th ed.).
  McGraw-Hill. ISBN 978-0-07-136097-5.
  [Open Library record](https://openlibrary.org/isbn/9780071360975).
  A practical room-acoustics handbook; its Fig. 7-22 worked example anchors
  the reverberation-prediction conformance suite.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Carrión Isbert, A. (1998). *Diseño acústico de espacios arquitectónicos*.
  Edicions UPC. ISBN 978-84-8301-252-9.
  [Open Library record](https://openlibrary.org/books/OL23159935M).
  A Spanish-language textbook on acoustic room design.
  Cited by [Reverberation-time prediction](/phonometry/guides/reverberation-prediction/).
- Beranek, L. L. (1957). Revised criteria for noise in buildings. *Noise
  Control*, 3(1), 19-27.
  [doi:10.1121/1.2369239](https://doi.org/10.1121/1.2369239).
  The original NC curves and their speech-interference rationale.
  Cited by [Room-noise criteria](/phonometry/guides/room-noise/).
- Kosten, C. W., & van Os, G. J. (1962). Community reaction criteria for
  external noises. In *The Control of Noise* (National Physical Laboratory
  Symposium No. 12, pp. 373-387). Her Majesty's Stationery Office.
  [Open Library record](https://openlibrary.org/books/OL58781133M).
  The NR curve family contrasted with NC.
  Cited by [Room-noise criteria](/phonometry/guides/room-noise/).
- Blazier, W. E. (1997). RC Mark II: A refined procedure for rating the
  noise of heating, ventilating, and air-conditioning (HVAC) systems in
  buildings. *Noise Control Engineering Journal*, 45(6), 243-250.
  [doi:10.3397/1.2828446](https://doi.org/10.3397/1.2828446).
  The RC Mark II procedure later codified by ANSI/ASA S12.2 Annex D.
  Cited by [Room-noise criteria](/phonometry/guides/room-noise/).
- International Organization for Standardization. (2009). *Acoustics —
  Measurement of room acoustic parameters — Part 1: Performance spaces*
  (ISO 3382-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40979.html).
  Room-parameter definitions, position requirements and just-noticeable
  differences.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- International Organization for Standardization. (2008). *Acoustics —
  Measurement of room acoustic parameters — Part 2: Reverberation time in
  ordinary rooms* (ISO 3382-2:2008).
  [iso.org catalogue](https://www.iso.org/standard/36201.html).
  The accuracy grades and position counts of reverberation measurement.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- International Organization for Standardization. (2012). *Acoustics —
  Measurement of room acoustic parameters — Part 3: Open plan offices*
  (ISO 3382-3:2012).
  [iso.org catalogue](https://www.iso.org/standard/46520.html).
  The open-plan speech-privacy quantities.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- International Organization for Standardization. (2006). *Acoustics —
  Application of new measurement methods in building and room acoustics*
  (ISO 18233:2006).
  [iso.org catalogue](https://www.iso.org/standard/40408.html).
  The swept-sine and MLS acquisition of impulse responses.
  Cited by [Room Acoustics](/phonometry/guides/room-acoustics/).
- International Organization for Standardization. (2003). *Acoustics —
  Measurement of sound absorption in a reverberation room* (ISO 354:2003).
  [iso.org catalogue](https://www.iso.org/standard/34545.html).
  The reverberation-room absorption measurement behind the surface data.
  Cited by [Sound absorption in enclosed spaces](/phonometry/guides/enclosed-space-absorption/) and
  [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/).
- European Committee for Standardization. (2003). *Building acoustics —
  Estimation of acoustic performance of buildings from the performance of
  elements — Part 6: Sound absorption in enclosed spaces*
  (EN 12354-6:2003).
  [BSI Knowledge record (BS EN 12354-6:2003)](https://knowledge.bsigroup.com/products/building-acoustics-estimation-of-acoustic-performance-of-buildings-from-the-performance-of-elements-sound-absorption-in-enclosed-spaces).
  The absorption member of the EN 12354 prediction family.
  Cited by [Sound absorption in enclosed spaces](/phonometry/guides/enclosed-space-absorption/).
- Acoustical Society of America. (2019). *Criteria for evaluating room
  noise* (ANSI/ASA S12.2-2019).
  [ANSI webstore](https://webstore.ansi.org/standards/asa/ansiasas122019).
  The normative NC tangency method and the RC Mark II rating of its
  informative Annex D, with its spectral tag.
  Cited by [Room-noise criteria](/phonometry/guides/room-noise/).

## Materials and surfaces

- Allard, J. F., & Atalla, N. (2009). *Propagation of sound in porous media:
  Modelling sound absorbing materials* (2nd ed.). Wiley.
  ISBN 978-0-470-74661-5.
  [doi:10.1002/9780470747339](https://doi.org/10.1002/9780470747339).
  The porous-material theory linking airflow resistivity, surface impedance
  and absorption.
  Cited by [Airflow Resistance](/phonometry/guides/airflow-resistance/) and
  [Impedance Tube](/phonometry/guides/impedance-tube/).
- Cox, T. J., & D'Antonio, P. (2017). *Acoustic absorbers and diffusers:
  Theory, design and application* (3rd ed.). CRC Press.
  ISBN 978-1-4987-4099-9.
  [doi:10.1201/9781315369211](https://doi.org/10.1201/9781315369211).
  The monograph on absorber and diffuser measurement and design, by the
  authors behind the ISO 17497-2 diffusion-coefficient method.
  Cited by
  [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/) and
  [Surface Scattering, Diffusion and In-situ Absorption](/phonometry/guides/surface-scattering/).
- Jiménez, N., Umnova, O., & Groby, J.-P. (Eds.). (2021). *Acoustic waves in
  periodic structures, metamaterials, and porous media* (Topics in Applied
  Physics, Vol. 143). Springer.
  [doi:10.1007/978-3-030-84300-7](https://doi.org/10.1007/978-3-030-84300-7).
  An edited umbrella volume on resonant and periodic sound-absorbing and
  sound-diffusing structures, from the transfer-matrix and critical-coupling
  theory of metamaterial absorbers to deep-subwavelength diffusers; the modern
  metamaterials companion to Cox & D'Antonio.
- International Organization for Standardization. (1998). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 2: Transfer-function method* (ISO 10534-2:1998; adopted in
  Europe as EN ISO 10534-2:2001; since revised as
  [ISO 10534-2:2023](https://www.iso.org/standard/81294.html)).
  [iso.org catalogue](https://www.iso.org/standard/22851.html).
  The two-microphone transfer-function method and its plane-wave limits.
  Cited by [Impedance Tube](/phonometry/guides/impedance-tube/).
- ASTM International. (2019). *Standard test method for normal incidence
  determination of porous material acoustical properties based on the
  transfer matrix method* (ASTM E2611-19, the edition implemented here;
  since revised as [ASTM E2611-24](https://store.astm.org/e2611-24.html)).
  [ASTM store](https://store.astm.org/e2611-19.html).
  The four-microphone transfer-matrix transmission-loss method.
  Cited by [Impedance Tube](/phonometry/guides/impedance-tube/).
- International Organization for Standardization. (2018). *Acoustics —
  Determination of airflow resistance — Part 1: Static airflow method*
  (ISO 9053-1:2018).
  [iso.org catalogue](https://www.iso.org/standard/69869.html).
  The static airflow-resistance method and its reference velocity.
  Cited by [Airflow Resistance](/phonometry/guides/airflow-resistance/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination of airflow resistance — Part 2: Alternating airflow method*
  (ISO 9053-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/76744.html).
  The alternating airflow-resistance method with the Annex A effective ratio
  of specific heats.
  Cited by [Airflow Resistance](/phonometry/guides/airflow-resistance/).
- International Organization for Standardization. (1996). *Acoustics —
  Determination of sound absorption coefficient and impedance in impedance
  tubes — Part 1: Method using standing wave ratio* (ISO 10534-1:1996;
  implemented as its European adoption BS EN ISO 10534-1:2001).
  [iso.org catalogue](https://www.iso.org/standard/18603.html).
  The standing-wave-ratio method.
  Cited by [Impedance Tube](/phonometry/guides/impedance-tube/).
- International Organization for Standardization. (1997). *Acoustics — Sound
  absorbers for use in buildings — Rating of sound absorption*
  (ISO 11654:1997).
  [iso.org catalogue](https://www.iso.org/standard/19583.html).
  The weighted sound-absorption rating, its shape indicators and the
  absorption class.
  Cited by
  [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 2: Sound absorption* (ISO 12999-2:2020).
  [iso.org catalogue](https://www.iso.org/standard/68749.html).
  The reproducibility and repeatability uncertainties of the
  reverberation-room quantities and their single-number ratings.
  Cited by
  [Sound Absorption Measurement and Rating](/phonometry/guides/absorption-measurement/).
- International Organization for Standardization. (2004). *Acoustics —
  Sound-scattering properties of surfaces — Part 1: Measurement of the
  random-incidence scattering coefficient in a reverberation room*
  (ISO 17497-1:2004+A1:2014, the edition implemented here).
  [iso.org catalogue](https://www.iso.org/standard/31397.html).
  The turntable scattering-coefficient method.
  Cited by [Surface Scattering, Diffusion and In-situ Absorption](/phonometry/guides/surface-scattering/).
- International Organization for Standardization. (2012). *Acoustics —
  Sound-scattering properties of surfaces — Part 2: Measurement of the
  directional diffusion coefficient in a free field* (ISO 17497-2:2012).
  [iso.org catalogue](https://www.iso.org/standard/55293.html).
  The goniometer diffusion-coefficient method.
  Cited by [Surface Scattering, Diffusion and In-situ Absorption](/phonometry/guides/surface-scattering/).

## Building acoustics

- Hopkins, C. (2007). *Sound insulation*. Butterworth-Heinemann.
  ISBN 978-0-7506-6526-1.
  [doi:10.4324/9780080550473](https://doi.org/10.4324/9780080550473).
  The comprehensive treatment of airborne and impact sound insulation:
  measurement chains, flanking transmission and the EN 12354 prediction
  framework.
  Cited by [Field Insulation Measurement and Ratings](/phonometry/guides/insulation-field/),
  [Laboratory Insulation Measurement](/phonometry/guides/insulation-lab/) and
  [Predicting Sound Insulation (EN 12354)](/phonometry/guides/insulation-prediction/).
- Vigran, T. E. (2008). *Building acoustics*. CRC Press.
  ISBN 978-0-415-42853-8.
  [doi:10.1201/9781482266016](https://doi.org/10.1201/9781482266016).
  A compact textbook on sound transmission in buildings, from single and
  double constructions to floating floors.
  Cited by [Field Insulation Measurement and Ratings](/phonometry/guides/insulation-field/),
  [Laboratory Insulation Measurement](/phonometry/guides/insulation-lab/) and
  [Dynamic stiffness of resilient materials](/phonometry/guides/dynamic-stiffness/).
- International Organization for Standardization. (2020). *Acoustics —
  Rating of sound insulation in buildings and of building elements — Part 1:
  Airborne sound insulation* (ISO 717-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/77435.html).
  The reference-curve rating and the spectrum adaptation terms C and Ctr.
  Cited by [Field Insulation Measurement and Ratings](/phonometry/guides/insulation-field/).
- International Organization for Standardization. (2014). *Acoustics — Field
  measurement of sound insulation in buildings and of building elements —
  Part 1: Airborne sound insulation* (ISO 16283-1:2014).
  [iso.org catalogue](https://www.iso.org/standard/55997.html).
  The field airborne measurement method.
  Cited by [Field Insulation Measurement and Ratings](/phonometry/guides/insulation-field/).
- International Organization for Standardization. (1989). *Acoustics —
  Determination of dynamic stiffness — Part 1: Materials used under floating
  floors in dwellings* (ISO 9052-1:1989).
  [iso.org catalogue](https://www.iso.org/standard/16620.html).
  The resonance method for the dynamic stiffness per unit area, identical to
  EN 29052-1.
  Cited by [Dynamic stiffness of resilient materials](/phonometry/guides/dynamic-stiffness/).

## Structure-borne sound

- Cremer, L., Heckl, M., & Petersson, B. A. T. (2005). *Structure-borne
  sound: Structural vibrations and sound radiation at audio frequencies*
  (3rd ed.). Springer. ISBN 978-3-540-22696-3.
  [doi:10.1007/b137728](https://doi.org/10.1007/b137728).
  The standard monograph on structural vibration and its radiation:
  mobilities, power flow, vibration isolation, radiation efficiency and
  transmission across junctions.
  Cited by [Mechanical mobility and the FRF family](/phonometry/guides/mechanical-mobility/),
  [Transfer stiffness of resilient elements](/phonometry/guides/transfer-stiffness/),
  [Sound power from surface vibration](/phonometry/guides/vibration-sound-power/),
  [Structure-borne sound power of equipment](/phonometry/guides/structure-borne-power/)
  and [Installed structure-borne sound](/phonometry/guides/installed-structure-borne/).
- Cremer, L., Heckl, M., & Ungar, E. E. (1973). *Structure-borne sound:
  Structural vibrations and sound radiation at audio frequencies* (1st ed.).
  Springer. ISBN 978-3-540-06002-4.
  [doi:10.1007/978-3-662-10118-6](https://doi.org/10.1007/978-3-662-10118-6).
  The original derivation of the wave parameters χ and ψ and the bending-wave
  transmission coefficients for junctions of plates.
  Cited by [Bending-wave transmission at plate junctions](/phonometry/guides/junction-transmission/).
- Craik, R. J. M. (1996). *Sound transmission through buildings using
  statistical energy analysis*. Gower. ISBN 978-0-566-07572-5.
  The SEA treatment of airborne and structure-borne transmission in buildings,
  with the tabulated bending-wave transmission coefficients for X, T, L and
  in-line junctions.
  Cited by [Bending-wave transmission at plate junctions](/phonometry/guides/junction-transmission/).
- International Organization for Standardization. (2011). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 1: Basic terms and definitions, and transducer specifications*
  (ISO 7626-1:2011).
  [iso.org catalogue](https://www.iso.org/standard/50426.html).
  The FRF family and its free/blocked distinctions.
  Cited by [Mechanical mobility and the FRF family](/phonometry/guides/mechanical-mobility/).
- International Organization for Standardization. (2015). *Mechanical
  vibration and shock — Experimental determination of mechanical mobility —
  Part 2: Measurements using single-point translation excitation with an
  attached vibration exciter* (ISO 7626-2:2015).
  [iso.org catalogue](https://www.iso.org/standard/62483.html).
  The attached-exciter measurement method and its acceptance criteria.
  Cited by [Mechanical mobility and the FRF family](/phonometry/guides/mechanical-mobility/).
- International Organization for Standardization. (2008). *Acoustics and
  vibration — Laboratory measurement of vibro-acoustic transfer properties of
  resilient elements — Part 1: Principles and guidelines* (ISO 10846-1:2008).
  [iso.org catalogue](https://www.iso.org/standard/38936.html).
  The blocking-force idealisation behind the dynamic transfer stiffness.
  Cited by [Transfer stiffness of resilient elements](/phonometry/guides/transfer-stiffness/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 1: Survey method using a fixed radiation
  factor* (ISO/TS 7849-1:2009).
  [iso.org catalogue](https://www.iso.org/standard/40537.html).
  The upper-limit sound power from surface velocity with ε = 1.
  Cited by [Sound power from surface vibration](/phonometry/guides/vibration-sound-power/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of airborne sound power levels emitted by machinery using
  vibration measurement — Part 2: Engineering method including determination
  of the adequate radiation factor* (ISO/TS 7849-2:2009).
  [iso.org catalogue](https://www.iso.org/standard/40538.html).
  The engineering method with a measured band-wise radiation factor.
  Cited by [Sound power from surface vibration](/phonometry/guides/vibration-sound-power/).
- International Organization for Standardization. (1996). *Acoustics —
  Characterization of sources of structure-borne sound with respect to sound
  radiation from connected structures — Measurement of velocity at the
  contact points of machinery when resiliently mounted* (ISO 9611:1996).
  [iso.org catalogue](https://www.iso.org/standard/17424.html).
  The free-velocity characterization of resiliently mounted sources.
  Cited by [Structure-borne sound power of equipment](/phonometry/guides/structure-borne-power/).

## Outdoor and environmental sound

- Salomons, E. M. (2001). *Computational atmospheric acoustics*. Kluwer
  Academic Publishers. ISBN 978-1-4020-0390-5.
  [doi:10.1007/978-94-010-0660-6](https://doi.org/10.1007/978-94-010-0660-6).
  The wave-based theory of outdoor sound (parabolic equation, fast field
  program, refraction, turbulence) behind the engineering approximations of
  ISO 9613-2.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/).
- Attenborough, K., & Van Renterghem, T. (2021). *Predicting outdoor sound*
  (2nd ed.). CRC Press.
  [doi:10.1201/9780429470806](https://doi.org/10.1201/9780429470806).
  Ground impedance models, the spherical-wave reflection coefficient behind
  the ground dip, and meteorological effects on barriers.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/).
- Maekawa, Z. (1968). Noise reduction by screens. *Applied Acoustics*, 1(3),
  157-173.
  [doi:10.1016/0003-682X(68)90020-0](https://doi.org/10.1016/0003-682X(68)90020-0).
  The screen-attenuation chart against Fresnel number that barrier
  engineering formulas descend from.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/).
- Kephalopoulos, S., Paviotti, M., & Anfosso-Lédée, F. (2012). *Common noise
  assessment methods in Europe (CNOSSOS-EU)* (EUR 25379 EN). Publications
  Office of the European Union.
  [doi:10.2788/31776](https://doi.org/10.2788/31776),
  [JRC repository](https://publications.jrc.ec.europa.eu/repository/handle/JRC72550).
  The common EU noise-mapping framework, contrasted with ISO 9613-2; its
  flow-resistivity ground classes are reused by the rotorcraft ground effect.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/)
  and [Rotorcraft noise](/phonometry/guides/rotorcraft-noise/).
- International Organization for Standardization. (1993). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 1: Calculation of
  the absorption of sound by the atmosphere* (ISO 9613-1:1993).
  [iso.org catalogue](https://www.iso.org/standard/17426.html).
  The pure-tone atmospheric attenuation coefficient.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/).
- International Organization for Standardization. (1996). *Acoustics —
  Attenuation of sound during propagation outdoors — Part 2: General method
  of calculation* (ISO 9613-2:1996; revised in 2024, the 1996 method is the
  implemented one).
  [iso.org catalogue](https://www.iso.org/standard/20649.html).
  The implemented outdoor attenuation chain.
  Cited by [Outdoor Sound Propagation](/phonometry/guides/outdoor-propagation/).
- International Organization for Standardization. (2016). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 1:
  Basic quantities and assessment procedures* (ISO 1996-1:2016).
  [iso.org catalogue](https://www.iso.org/standard/59765.html).
  The environmental rating framework and its Table A.1 category adjustments.
  Cited by [Impulsive-sound prominence](/phonometry/guides/impulse-prominence/).
- International Organization for Standardization. (2017). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 2:
  Determination of sound pressure levels* (ISO 1996-2:2017).
  [iso.org catalogue](https://www.iso.org/standard/59766.html).
  The environmental measurement standard: its Annex J adopts the engineering
  method for tonal audibility, and the audibility criterion IEC 61400-11
  reuses comes from the Annex C of its 2007 edition.
  Cited by [Objective audibility of tones](/phonometry/guides/tone-audibility/) and
  [Wind-turbine noise](/phonometry/guides/wind-turbine-noise/).
- Nordtest. (2002). *Acoustics: Prominence of impulsive sounds and for
  adjustment of LAeq* (Nordtest Method NT ACOU 112).
  [nordtest.info](https://www.nordtest.info/wp/2002/05/01/acoustics-prominence-of-impulsive-sounds-and-for-adjustment-of-laeq-nt-acou-112/).
  The freely downloadable onset-rate prominence method.
  Cited by [Impulsive-sound prominence](/phonometry/guides/impulse-prominence/).
- International Organization for Standardization. (2022). *Acoustics —
  Description, measurement and assessment of environmental noise — Part 3:
  Objective method for the measurement of prominence of impulsive sounds and
  for adjustment of LAeq* (ISO/PAS 1996-3:2022).
  [iso.org catalogue](https://www.iso.org/standard/77035.html).
  The ISO successor built on the NT ACOU 112 prominence.
  Cited by [Impulsive-sound prominence](/phonometry/guides/impulse-prominence/).
- International Electrotechnical Commission. (2018). *Wind turbines —
  Part 11: Acoustic noise measurement techniques*
  (IEC 61400-11:2012+AMD1:2018 CSV).
  [IEC webstore](https://webstore.iec.ch/en/publication/63367).
  The apparent sound power geometry, wind-speed binning and tonal
  audibility of wind turbines.
  Cited by [Wind-turbine noise](/phonometry/guides/wind-turbine-noise/).
- International Electrotechnical Commission. (2005). *Wind turbines —
  Part 14: Declaration of apparent sound power level and tonality values*
  (IEC TS 61400-14:2005).
  [IEC webstore](https://webstore.iec.ch/en/publication/5432).
  Declared values and their uncertainty for a batch of turbines.
  Cited by [Wind-turbine noise](/phonometry/guides/wind-turbine-noise/).

## Aircraft noise

- International Civil Aviation Organization. (2017). *Annex 16 to the
  Convention on International Civil Aviation: Environmental protection —
  Volume I: Aircraft noise* (8th ed.).
  [ICAO store](https://store.icao.int/en/annex-16-environmental-protection-volume-i-aircraft-noise).
  The aircraft noise-certification standard whose Appendix 2 defines the
  EPNL procedure.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- International Civil Aviation Organization. (2018). *Environmental technical
  manual — Volume I: Procedures for the noise certification of aircraft*
  (Doc 9501, 3rd ed.).
  [ICAO store](https://store.icao.int/en/environmental-technical-manual-volume-1-procedures-for-the-noise-certification-of-aircraft-doc-9501-1).
  The certification guidance whose worked examples (tone correction,
  integrated-method EPNL) serve as numeric oracles.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- International Electrotechnical Commission. (1995). *Electroacoustics —
  Instruments for measurement of aircraft noise — Performance requirements for
  systems to measure one-third-octave-band sound pressure levels in noise
  certification of transport-category aeroplanes* (IEC 61265:1995; since
  revised as [IEC 61265:2018](https://webstore.iec.ch/en/publication/32635),
  the 1995 edition is the implemented one).
  [IEC webstore](https://webstore.iec.ch/en/publication/5076).
  The aircraft-noise measurement-system performance tolerances.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- SAE International. (2013). *Application of pure-tone atmospheric absorption
  losses to one-third octave-band data* (SAE ARP 5534, reaffirmed 2021).
  [sae.org](https://www.sae.org/standards/content/arp5534/).
  The SAE-Method one-third-octave-band atmospheric absorption for aircraft
  flyover spectra.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- SAE International. (2012). *Standard values of atmospheric absorption as a
  function of temperature and humidity* (SAE ARP 866B, stabilized 2012).
  [sae.org](https://www.sae.org/standards/content/arp866b/).
  The predecessor SAE atmospheric-absorption practice, source of the older
  50 dB-limited Approximate Method.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- SAE International. (2006). *Method for predicting lateral attenuation of
  airplane noise* (SAE AIR 5662).
  [sae.org](https://www.sae.org/standards/content/air5662/).
  The soft-ground lateral-attenuation model adopted by ECAC Doc 29.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- European Civil Aviation Conference. (2016). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 4th ed.),
  Volume 2: Technical guide.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-Doc_29_4th_edition_Dec_2016_Volume_2.pdf).
  The European airport noise-contour method: NPD interpolation and the
  single-event segment calculation.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing noise contours around civil airports* (ECAC.CEAC Doc 29, 5th ed.),
  Volume 3: Reference cases and verification framework.
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_29_5th_Edition-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_NOISE_CONTOURS_AROUND_CIVIL_AIRPORTS-Volume_3-REFERENCE_CASES_AND_VERIFICATION_FRAMEWORK.pdf).
  The reference cases and workbook used to validate the single-event chain.
  Cited by [Aircraft noise](/phonometry/guides/aircraft-noise/).
- European Civil Aviation Conference. (2026). *Report on standard method of
  computing rotorcraft noise contours* (ECAC.CEAC Doc 32, 1st ed.).
  [ECAC documents page](https://www.ecac-ceac.org/documents/ecac-documents-and-international-agreements),
  [free PDF](https://www.ecac-ceac.org/images/documents/ECAC-CEAC-DOC_32-REPORT_ON_STANDARD_METHOD_OF_COMPUTING_ROTORCRAFT_NOISE_CONTOURS.pdf).
  The standard rotorcraft contour method built on the noise hemisphere.
  Cited by [Rotorcraft noise](/phonometry/guides/rotorcraft-noise/).
- Olsen, H., Tuinstra, M., & van Oosten, N. (2024). *Rotorcraft noise
  modelling guidance* (Research Project NOISE SC01, deliverable D1.5d,
  contract EASA.2020.FC.06). European Union Aviation Safety Agency.
  [EASA project page](https://www.easa.europa.eu/en/research-projects/environmental-research-rotorcraft-noise),
  [free PDF](https://www.easa.europa.eu/en/downloads/132005/en).
  The NORAH2 equation-level modelling guidance, whose tables and reference
  hemispheres serve as oracles.
  Cited by [Rotorcraft noise](/phonometry/guides/rotorcraft-noise/).
- Chien, C. F., & Soroka, W. W. (1975). Sound propagation along an impedance
  plane. *Journal of Sound and Vibration*, 43(1), 9-20.
  [doi:10.1016/0022-460X(75)90200-X](https://doi.org/10.1016/0022-460X(75)90200-X).
  The two-ray interference solution over an impedance plane behind the
  rotorcraft ground effect.
  Cited by [Rotorcraft noise](/phonometry/guides/rotorcraft-noise/).
- Delany, M. E., & Bazley, E. N. (1970). Acoustical properties of fibrous
  absorbent materials. *Applied Acoustics*, 3(2), 105-116.
  [doi:10.1016/0003-682X(70)90031-9](https://doi.org/10.1016/0003-682X(70)90031-9).
  The one-parameter flow-resistivity ground-impedance model.
  Cited by [Rotorcraft noise](/phonometry/guides/rotorcraft-noise/).

## Underwater sound

- Urick, R. J. (1983). *Principles of underwater sound* (3rd ed.).
  McGraw-Hill; reprinted 1996 by Peninsula Publishing.
  ISBN 978-0-932146-62-5.
  [Open Library record](https://openlibrary.org/books/OL9317725M).
  The classic monograph on underwater sound: level conventions, ship
  radiated noise and the sonar-equation framework.
  Cited by [Underwater acoustics](/phonometry/guides/underwater-acoustics/) and
  [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Ainslie, M. A. (2010). *Principles of sonar performance modelling*.
  Springer.
  [doi:10.1007/978-3-540-87662-5](https://doi.org/10.1007/978-3-540-87662-5).
  The systematic treatment of underwater acoustical quantities in the line
  that ISO 18405 standardised.
  Cited by [Underwater acoustics](/phonometry/guides/underwater-acoustics/).
- Medwin, H., & Clay, C. S. (1998). *Fundamentals of acoustical oceanography*.
  Academic Press. ISBN 978-0-12-487570-8.
  [Publisher page](https://shop.elsevier.com/books/fundamentals-of-acoustical-oceanography/medwin/978-0-12-487570-8).
  Ocean acoustics from first principles; the fluid-fluid Rayleigh
  reflection coefficient of the seabed model.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Jensen, F. B., Kuperman, W. A., Porter, M. B., & Schmidt, H. (2011).
  *Computational ocean acoustics* (2nd ed.). Springer.
  [doi:10.1007/978-1-4419-8678-8](https://doi.org/10.1007/978-1-4419-8678-8).
  The reference monograph on numerical propagation: normal modes, ray
  tracing and the parabolic equation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements: Part I: Pure water and magnesium sulfate contributions.
  *The Journal of the Acoustical Society of America*, 72(3), 896-907.
  [doi:10.1121/1.388170](https://doi.org/10.1121/1.388170).
  The pure-water and magnesium-sulfate halves of the reference seawater
  absorption model.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Francois, R. E., & Garrison, G. R. (1982). Sound absorption based on ocean
  measurements. Part II: Boric acid contribution and equation for total
  absorption. *The Journal of the Acoustical Society of America*, 72(6),
  1879-1890.
  [doi:10.1121/1.388673](https://doi.org/10.1121/1.388673).
  The boric-acid term and the complete total-absorption equation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Ainslie, M. A., & McColm, J. G. (1998). A simplified formula for viscous and
  chemical absorption in sea water. *The Journal of the Acoustical Society of
  America*, 103(3), 1671-1672.
  [doi:10.1121/1.421258](https://doi.org/10.1121/1.421258).
  The legible simplified seawater absorption formula.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Thorp, W. H. (1967). Analytic description of the low-frequency attenuation
  coefficient. *The Journal of the Acoustical Society of America*, 42(1), 270.
  [doi:10.1121/1.1910566](https://doi.org/10.1121/1.1910566).
  The frequency-only low-frequency absorption formula.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Chen, C.-T., & Millero, F. J. (1977). Speed of sound in seawater at high
  pressures. *The Journal of the Acoustical Society of America*, 62(5),
  1129-1135.
  [doi:10.1121/1.381646](https://doi.org/10.1121/1.381646).
  The UNESCO international-standard sound-speed equation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Wong, G. S. K., & Zhu, S. (1995). Speed of sound in seawater as a function
  of salinity, temperature, and pressure. *The Journal of the Acoustical
  Society of America*, 97(3), 1732-1736.
  [doi:10.1121/1.413048](https://doi.org/10.1121/1.413048).
  The ITS-90 recast of the UNESCO sound-speed coefficients, the implemented
  form.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Del Grosso, V. A. (1974). New equation for the speed of sound in natural
  waters (with comparisons to other equations). *The Journal of the
  Acoustical Society of America*, 56(4), 1084-1091.
  [doi:10.1121/1.1903388](https://doi.org/10.1121/1.1903388).
  The alternative pressure-based sound-speed equation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Mackenzie, K. V. (1981). Nine-term equation for sound speed in the oceans.
  *The Journal of the Acoustical Society of America*, 70(3), 807-812.
  [doi:10.1121/1.386920](https://doi.org/10.1121/1.386920).
  The depth-based nine-term sound-speed equation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Leroy, C. C., & Parthiot, F. (1998). Depth-pressure relationships in the
  oceans and seas. *The Journal of the Acoustical Society of America*, 103(3),
  1346-1352.
  [doi:10.1121/1.421275](https://doi.org/10.1121/1.421275).
  The depth-to-pressure conversion used by the sound-speed equations.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Wenz, G. M. (1962). Acoustic ambient noise in the ocean: Spectra and
  sources. *The Journal of the Acoustical Society of America*, 34(12),
  1936-1956.
  [doi:10.1121/1.1909155](https://doi.org/10.1121/1.1909155).
  The classic ambient-noise survey behind the wind and thermal spectrum
  components.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Carey, W. M., & Evans, R. B. (2011). *Ocean ambient noise: Measurement and
  theory*. Springer.
  [doi:10.1007/978-1-4419-7832-5](https://doi.org/10.1007/978-1-4419-7832-5).
  The modern treatment of ocean ambient noise: the wind "rule of fives" and
  the Mellen thermal-noise derivation.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- MacGillivray, A., & de Jong, C. (2021). A reference spectrum model for
  estimating source levels of marine shipping based on automated
  identification system data. *Journal of Marine Science and Engineering*,
  9(4), 369.
  [doi:10.3390/jmse9040369](https://doi.org/10.3390/jmse9040369).
  The open-access JOMOPANS-ECHO ship source-level model and its reference
  calculator.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).
- Wales, S. C., & Heitmeyer, R. M. (2002). An ensemble source spectra model
  for merchant ship-radiated noise. *The Journal of the Acoustical Society of
  America*, 111(3), 1211-1231.
  [doi:10.1121/1.1427355](https://doi.org/10.1121/1.1427355).
  The ensemble merchant-ship source-spectrum model.
  Cited by [Underwater sound propagation](/phonometry/guides/underwater-propagation/).

## Speech

- Houtgast, T., & Steeneken, H. J. M. (1985). A review of the MTF concept in
  room acoustics and its use for estimating speech intelligibility in
  auditoria. *The Journal of the Acoustical Society of America*, 77(3),
  1069-1077. [doi:10.1121/1.392224](https://doi.org/10.1121/1.392224).
  The modulation-transfer framework the Speech Transmission Index is built on.
  Cited by [Speech Transmission Index](/phonometry/guides/speech-transmission/).
- French, N. R., & Steinberg, J. C. (1947). Factors governing the
  intelligibility of speech sounds. *The Journal of the Acoustical Society of
  America*, 19(1), 90-119.
  [doi:10.1121/1.1916407](https://doi.org/10.1121/1.1916407).
  The articulation-band experiments behind the band-importance function of the
  Speech Intelligibility Index.
  Cited by [Speech Intelligibility Index](/phonometry/guides/speech-intelligibility/).

## Psychoacoustics

- Fletcher, H., & Munson, W. A. (1933). Loudness, its definition, measurement
  and calculation. *The Journal of the Acoustical Society of America*, 5(2),
  82-108. [doi:10.1121/1.1915637](https://doi.org/10.1121/1.1915637).
  The original equal-loudness measurements whose 40-phon contour became the
  A-weighting curve.
  Cited by [Frequency Weighting (A, C, G, Z)](/phonometry/guides/weighting/)
  and [Loudness](/phonometry/guides/loudness/).
- International Organization for Standardization. (2023). *Acoustics —
  Normal equal-loudness-level contours* (ISO 226:2023).
  [iso.org catalogue](https://www.iso.org/standard/83117.html).
  The modern equal-loudness contours, successors of the Fletcher-Munson
  curves.
  Cited by [Frequency Weighting (A, C, G, Z)](/phonometry/guides/weighting/)
  and [Loudness](/phonometry/guides/loudness/).
- Fastl, H., & Zwicker, E. (2007). *Psychoacoustics: Facts and models*
  (3rd ed.). Springer.
  [doi:10.1007/978-3-540-68888-4](https://doi.org/10.1007/978-3-540-68888-4).
  The psychoacoustic-annoyance model and the closed-form fluctuation strength
  for amplitude-modulated broadband noise.
  Cited by [Psychoacoustic annoyance](/phonometry/guides/psychoacoustic-annoyance/),
  [Loudness](/phonometry/guides/loudness/) and
  [Sound Quality Metrics](/phonometry/guides/sound-quality/).
- Osses Vecchi, A., García León, R., & Kohlrausch, A. (2016). Modelling the
  sensation of fluctuation strength. *Proceedings of Meetings on Acoustics*,
  28, 050005. [doi:10.1121/2.0000410](https://doi.org/10.1121/2.0000410).
  The fluctuation-strength signal model and its Table 1 literature values.
  Cited by [Psychoacoustic annoyance](/phonometry/guides/psychoacoustic-annoyance/).
- Felix Greco, G., Merino-Martínez, R., Osses, A., & Lotinga, M. J. B. (2025).
  *SQAT: a sound quality analysis toolbox for MATLAB* (open-source software).
  [github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT),
  [doi:10.5281/zenodo.7934709](https://doi.org/10.5281/zenodo.7934709).
  The open MATLAB reference used as the numeric oracle for the
  fluctuation-strength cross-checks.
  Cited by [Psychoacoustic annoyance](/phonometry/guides/psychoacoustic-annoyance/).
- Ecma International. (2024). *ECMA-418-1: Psychoacoustic metrics for ITT
  equipment — Part 1: Prominent discrete tones* (3rd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-418-1_3rd_edition_december_2024.pdf).
  The freely downloadable tone-to-noise ratio and prominence ratio methods.
  Cited by [Prominent Discrete Tones](/phonometry/guides/tone-prominence/).
- Ecma International. (2025). *ECMA-74: Measurement of airborne noise emitted
  by information technology and telecommunications equipment* (22nd ed.).
  [Free PDF](https://ecma-international.org/wp-content/uploads/ECMA-74_22nd_edition_december_2025.pdf).
  The freely downloadable parent emission standard whose Annex D delegates
  tone assessment to ECMA-418-1.
  Cited by [Prominent Discrete Tones](/phonometry/guides/tone-prominence/).
- International Organization for Standardization. (2016). *Acoustics —
  Objective method for assessing the audibility of tones in noise —
  Engineering method* (ISO/PAS 20065:2016).
  [iso.org catalogue](https://www.iso.org/standard/66941.html).
  The engineering method for the objective audibility of tones.
  Cited by [Objective audibility of tones](/phonometry/guides/tone-audibility/).

## Hearing and hearing conservation

- International Organization for Standardization. (2017). *Acoustics —
  Statistical distribution of hearing thresholds related to age and gender*
  (ISO 7029:2017). [iso.org catalogue](https://www.iso.org/standard/42916.html).
  The age model of the hearing threshold and its population spread.
  Cited by [Hearing threshold](/phonometry/guides/hearing-threshold/).
- International Organization for Standardization. (2005). *Acoustics —
  Reference zero for the calibration of audiometric equipment — Part 7:
  Reference threshold of hearing under free-field and diffuse-field listening
  conditions* (ISO 389-7:2005).
  [iso.org catalogue](https://www.iso.org/standard/38976.html).
  The audiometric zero as a sound pressure level.
  Cited by [Hearing threshold](/phonometry/guides/hearing-threshold/).
- International Organization for Standardization. (2013). *Acoustics —
  Estimation of noise-induced hearing loss* (ISO 1999:2013).
  [iso.org catalogue](https://www.iso.org/standard/45103.html).
  The NIPTS model, its distribution and the HTLAN combination.
  Cited by [Noise-induced hearing loss](/phonometry/guides/noise-induced-hearing-loss/).
- Passchier-Vermeer, W. (1974). Hearing loss due to continuous exposure to
  steady-state broad-band noise. *The Journal of the Acoustical Society of
  America*, 56(5), 1585–1593.
  [doi:10.1121/1.1903482](https://doi.org/10.1121/1.1903482).
  A field study of the noise exposure-response relations later codified in
  ISO 1999.
  Cited by [Noise-induced hearing loss](/phonometry/guides/noise-induced-hearing-loss/).
- National Institute for Occupational Safety and Health. (1998). *Criteria for
  a recommended standard: Occupational noise exposure — Revised criteria 1998*
  (DHHS/NIOSH Publication No. 98-126).
  [doi:10.26616/NIOSHPUB98126](https://doi.org/10.26616/NIOSHPUB98126),
  [free PDF](https://www.cdc.gov/niosh/docs/98-126/pdfs/98-126.pdf).
  The freely available criteria document behind the 85 dB(A) recommended
  exposure limit and the hearing-conservation and fence discussion.
  Cited by [Noise-induced hearing loss](/phonometry/guides/noise-induced-hearing-loss/) and
  [Occupational noise exposure](/phonometry/guides/occupational-exposure/).
- International Organization for Standardization. (2009). *Acoustics —
  Determination of occupational noise exposure — Engineering method*
  (ISO 9612:2009). [iso.org catalogue](https://www.iso.org/standard/41718.html).
  The three measurement strategies and the Annex C uncertainty budget.
  Cited by [Occupational noise exposure](/phonometry/guides/occupational-exposure/).
- European Parliament and Council. (2003). *Directive 2003/10/EC on the
  minimum health and safety requirements regarding the exposure of workers to
  the risks arising from physical agents (noise)*. Official Journal of the
  European Union.
  [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/dir/2003/10/oj/eng).
  The EU exposure action and limit values for occupational noise.
  Cited by [Occupational noise exposure](/phonometry/guides/occupational-exposure/).

## Human vibration

- Griffin, M. J. (1996). *Handbook of human vibration*. Academic Press.
  ISBN 978-0-12-303041-2.
  [Publisher page](https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2).
  The standard monograph on whole-body and hand-transmitted vibration: the
  biodynamics, discomfort and health-effect evidence behind the weightings,
  dose measures and exposure-response guidance of the vibration guides.
  Cited by [Human Vibration](/phonometry/guides/human-vibration/) and
  [Multiple-shock whole-body vibration](/phonometry/guides/multiple-shock-vibration/).
- Mansfield, N. J. (2004). *Human response to vibration*. CRC Press.
  ISBN 978-0-415-28239-0.
  [Publisher page](https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390).
  A compact modern textbook on the ISO 2631-1 and ISO 5349 evaluation chains,
  from perception and comfort to the occupational exposure limits.
  Cited by [Human Vibration](/phonometry/guides/human-vibration/).

## Metrology

- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Guide to the expression of uncertainty in measurement* (JCGM
  100:2008, the GUM). BIPM.
  [doi:10.59161/JCGM100-2008E](https://doi.org/10.59161/JCGM100-2008E),
  [free PDF](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf).
  The law of propagation of uncertainty implemented by the uncertainty module.
  Cited by [Measurement uncertainty](/phonometry/guides/gum-uncertainty/).
- Joint Committee for Guides in Metrology. (2008). *Evaluation of measurement
  data — Supplement 1 to the "Guide to the expression of uncertainty in
  measurement" — Propagation of distributions using a Monte Carlo method*
  (JCGM 101:2008). BIPM.
  [doi:10.59161/JCGM101-2008](https://doi.org/10.59161/JCGM101-2008),
  [free PDF](https://www.bipm.org/documents/20126/2071204/JCGM_101_2008_E.pdf).
  The Monte Carlo propagation of distributions implemented by the Monte Carlo
  uncertainty engine.
  Cited by [Measurement uncertainty](/phonometry/guides/gum-uncertainty/).
- International Organization for Standardization. (2020). *Acoustics —
  Determination and application of measurement uncertainties in building
  acoustics — Part 1: Sound insulation* (ISO 12999-1:2020).
  [iso.org catalogue](https://www.iso.org/standard/73930.html).
  The domain-specific reproducibility budget for building-acoustics
  single-number ratings, the companion to the general GUM machinery.
  Cited by [Measurement uncertainty](/phonometry/guides/gum-uncertainty/).
