## What this folder contains

| File | Kind | Provenance |
| :--- | :--- | :--- |
| `rail_emission_cases.csv` | derived (extract of a published result set) | 123 of the 19 872 cases of `CNOSSOS_RAIL_EMISSION_TEST.xlsx`, the railway emission test workbook published by the European Commission on CIRCABC with the CNOSSOS-EU source module. Each row carries the full case description read straight out of the workbook, together with the workbook's own per-octave-band and total line-power levels, copied verbatim to the two decimals it prints. |
| `rail_vehicles_2015.csv` | transcribed | The vehicle definitions of `CNOSSOS_Rail_Vehicles.xml` v1.1 (catalogue date 2015-04-09), the vehicle database of the same Commission source module: number of axles and the reference into each spectrum table. |
| `rail_wavelength_tables_2015.csv` | transcribed | The spectra of that catalogue and of `CNOSSOS_Rail_Track.xml` v1.0 (2014-04-27) that are given against wavelength: wheel roughness, contact filter, rail roughness and impact roughness, on the 32-step 1/3-octave grid from 1 000 mm to 0,8 mm. |
| `rail_frequency_tables_2015.csv` | transcribed | The spectra of the same two catalogues that are given against frequency, in the 24 1/3-octave bands from 50 Hz to 10 kHz: wheel, track and superstructure transfer functions, traction sound power for constant speed and for idling, and the aerodynamic reference spectra. |
| `road_emission_cases.csv` | derived (extract of a published result set) | 60 of the 4 875 cases of `CNOSSOS_ROAD_EMISSION_TEST.xlsx`, the road emission test workbook published by the European Commission on CIRCABC with the CNOSSOS-EU source module. Each row carries the segment description recovered from the workbook (see below) and the workbook's own per-octave-band and total line-power levels, copied verbatim to the two decimals it prints. |
| `road_coefficients_2015.csv` | transcribed | Table F-1 of Appendix F of Annex II to Directive 2002/49/EC **as published in Commission Directive (EU) 2015/996** (OJ L 168, 1.7.2015, p. 124), that is the version the workbook was computed with and which Commission Delegated Directive (EU) 2021/1226 later replaced. Machine-transcribed from the Official Journal text. |
| `road_surfaces_2015.csv` | transcribed | Table F-4 of the same 2015 text (OJ L 168, 1.7.2015, pp. 125-129), keyed by the `NLxx` surface identifiers the workbook uses. Also superseded by (EU) 2021/1226. |

These are the **2015** coefficients, not the ones the library ships. The
workbook was computed in 2014 with the database that Commission Directive
(EU) 2015/996 went on to print in Appendix G of Annex II, before Commission
Delegated Directive (EU) 2021/1226 replaced Tables G-1b, G-2, G-3a, G-4 and
G-7. The coefficients of the current text live in `tests/reference_data.py`,
transcribed from the Official Journal, and pin the tables shipped in
`phonometry.environment.sources.cnossos_rail`.

Only the running conditions the Directive models are transcribed. The
catalogue carries four (constant, accelerating, decelerating, idling) where
Annex II 2.3.2 models two, and states that
`L_W,0,const,i = L_W,0,idling,i`; the accelerating and decelerating spectra are
a database extension with no counterpart in the method, so neither they nor the
workbook cases that use them are committed here.

Three of the workbook's twenty-three vehicles are left out as well, because
their rows record defects of the reference program rather than of the method:
vehicle 15 carries `n` instead of a number in its axle-count and length fields
and produces no rolling noise at all, and vehicles 24 and 25 have no traction
data, which makes visible an all-zero aerodynamic spectrum that the program
energy-sums below 200 km/h. The remaining twenty vehicles are reproduced
exactly: the shipped model reproduces all 34 560 published rows of those twenty
vehicles, 17 280 cases at both source heights, to 0,0055 dB.

## Source and authorship

- The workbook is `CNOSSOS_RAIL_EMISSION_TEST.xlsx`, delivered with the
  *Testing of Emission DLL's for CNOSSOS-EU Road, Rail and Industry Noise
  Sources* report (Stapelfeldt Ingenieurgesellschaft mbH, Dortmund, doc. rev.
  1406-2, 7 July 2014) as part of the CNOSSOS-EU source module of the Joint
  Research Centre of the European Commission. Its results were produced twice,
  once by the reference emission library of DGMR and once by an independent
  implementation, and the two agree to 0,01 dB. The XML catalogues are
  delivered with the same module. Everything is distributed through the
  Commission's CIRCABC document repository (<https://circabc.europa.eu/>,
  CNOSSOS-EU interest group), retrieved 2026-07-29.
- Official Journal texts are published by the Publications Office of the
  European Union; reuse is authorised under Commission Decision 2011/833/EU,
  provided the source is acknowledged, which the header of every file here and
  the comments in `tests/reference_data.py` do.
- The workbook is `CNOSSOS_ROAD_EMISSION_TEST.xlsx`, delivered with the
  *Testing of Emission DLL's for CNOSSOS-EU Road, Rail and Industry Noise
  Sources* report (Stapelfeldt Ingenieurgesellschaft mbH, Dortmund, doc. rev.
  1406-2, 7 July 2014) as part of the CNOSSOS-EU source module of the Joint
  Research Centre of the European Commission. It is distributed through the
  Commission's CIRCABC document repository
  (<https://circabc.europa.eu/>, CNOSSOS-EU interest group), retrieved
  2026-07-29. Its results were produced twice, once by the reference emission
  library of DGMR and once by an independent implementation, and agree to
  0,01 dB.
- The coefficient tables are the text of Commission Directive (EU) 2015/996
  (OJ L 168, 1.7.2015, p. 1). Official Journal texts are published by the
  Publications Office of the European Union; reuse is authorised under
  Commission Decision 2011/833/EU, provided the source is acknowledged, which
  the header of every file here does.
- **This repository's MIT licence does not cover these files.** They are
  redistributed under the Commission's reuse policy as Commission publications.

## Purpose and scope of use

`tests/environment/sources/test_cnossos_rail.py` feeds the shipped equations of
Annex II 2.3 the 2015 coefficient set and requires the workbook's levels back,
band by band, at both source heights. That is the only oracle that pins the
*equations*: the Directive itself prints no worked example anywhere in 2.3 or
Appendix G. The 123 committed cases cover all twenty usable vehicles, both
source heights, all three speeds, both running conditions, all nine
track-parameter combinations, all four receiver-angle pairs and both flow
rates, so the committed subset reaches every level of every factor the full set
varies; what it does not reach is the sheer number of their combinations.

What the agreement can and cannot resolve is worth stating, because it is a
property of the source model and not of the test selection. A coefficient that
sits twenty decibels below the term beside it does not move the total, so the
workbook constrains the loud rows of the database and not the quiet ones.
Corrupting every committed coefficient one at a time (sign flip, decimal shift
up and down, transposition with the neighbouring cell) and recomputing all
cases kills 1 078 of the 1 116 mutations of the roughness and contact-filter
tables, but only 3 010 of the 4 015 mutations of the transfer and traction
tables. The survivors are concentrated in rows the published test set never
brings to the surface: the all-zero and all-140 dB "min" and "max" placeholders
of the IMAGINE catalogue, the source-A column of the constant-speed traction,
which rolling noise always covers, and the two track transfer functions the
workbook only ever pairs with the 140 dB superstructure placeholder. A
mistranscription in one of those cells could not change a computed value, so it
could not turn a wrong model into a passing run either; it would simply go
unnoticed, and the coverage assertion in the test module keeps at least one
unmasked case for every committed row.

The published levels themselves are pinned hard: 3 805 of the 3 806 mutations
of the 984 committed band levels are caught. The single survivor transposes two
levels of a saturated case that differ by 0,01 dB, the last digit the workbook
prints.

Because the workbook predates the 2021 amendment, the cases are run with the
2015 vertical directivity, with the curve-squeal excess and the bridge constant
of the 2015 text supplied as data columns, and with the roughness speed floor
switched off, which is how the reference program behaves. Everything the 2021
amendment changed is therefore covered by the table transcriptions and by the
closed-form assertions in the same test module, not by these cases.

The workbook tabulates the **total** line power of a segment carrying all five
vehicle categories, while each of its rows names only the category whose flow
and speed that row varies. The traffic composition behind each block was
recovered by exhaustive search over the flow and speed grids of the workbook
and then verified against **all 4 875 rows**, which it reproduces to 0,005 dB,
i.e. inside the workbook's own rounding. `road_emission_cases.csv` therefore
writes the five flows out explicitly, so the committed cases are readable
without the reconstruction.

The coefficients of the current text, Tables F-1 to F-4 as they stand after
(EU) 2021/1226, are *not* here: they live in `tests/reference_data.py`, also
machine-transcribed from the Official Journal, and pin the tables shipped in
`phonometry.environment.sources.cnossos_road`.

`tests/environment/sources/test_cnossos_road.py` feeds the shipped equations of
Annex II 2.2 the superseded 2015 coefficient set and requires the workbook's
levels back, band by band. That is the only oracle that pins the *equations*:
the Directive itself prints no worked example anywhere in 2.2 or Appendix F.
The 60 committed cases span all thirteen road surfaces exercised by the
workbook, all five vehicle categories, all five gradient and junction
combinations, all four temperature and studded-tyre combinations and all twelve
speeds, so the committed subset reaches everything the full set does except the
sheer number of repetitions.

Deliberately excluded: the two workbooks themselves (10 MB for the railway set
and 9 MB for the road one, almost all of it repeated cases), the CNOSSOS-EU
reference source module and its libraries, and the Official Journal PDFs. The
data are **not** part of the `phonometry` package and are not installed with
it.

## Removal policy

If you represent the European Commission, the Joint Research Centre or
Stapelfeldt Ingenieurgesellschaft and consider that committing this subset
exceeds the intended reuse of the published material, please open an issue or
contact the maintainer (see `CITATION.cff`) and it will be removed promptly.
The suite would then lose its only oracle for the railway and road emission
equations alike: the Appendix F and G transcriptions and the closed-form
assertions would still run, but the end-to-end cases would have to go with
them.
