# EASA ANP database (bundled)

This directory ships the **full EASA/EUROCONTROL Aircraft Noise and Performance
(ANP) database, archive version 2.3**, as the default dataset for
`phonometry.aircraft.anp_fleet` and the reference oracle for the test suite.

## Source

- EASA ANP database, archive **version 2.3** (released 14 October 2020), the
  semicolon-delimited CSV export (`archive_anp_v2.3.zip`).
- Public source: <https://www.aircraftnoisemodel.org/> (EASA / EUROCONTROL).

The data was developed by the aircraft manufacturers and collaboratively
reviewed by the US DOT Volpe Center, US FAA, EASA and EUROCONTROL.

## Contents

The complete v2.3 CSV tables are reproduced verbatim (155 aircraft types):

| File | Content |
|------|---------|
| `Aircraft.csv` | Aircraft metadata (engine type, weights, NPD id, mounting) |
| `NPD_data.csv` | Noise-Power-Distance curves (SEL, LAmax, EPNL, PNLTM) |
| `Default_fixed_point_profiles.csv` | Ready-to-use fixed-point trajectories |
| `Default_departure_procedural_steps.csv` | Departure procedural-step profiles |
| `Default_approach_procedural_steps.csv` | Approach procedural-step profiles |
| `Default_weights.csv` | Default weights per stage length |
| `Aerodynamic_coefficients.csv` | Aerodynamic (flap-configuration) coefficients |
| `Jet_engine_coefficients.csv` | Jet-engine thrust coefficients |
| `Propeller_engine_coefficients.csv` | Propeller-engine coefficients |
| `Spectral_classes.csv` | Departure/approach spectral classes |

The `anp_fleet` loader consumes `Aircraft.csv`, `NPD_data.csv` and
`Default_fixed_point_profiles.csv` for the fixed-point trajectories, and the
engine, aerodynamic, weight and procedural-step tables for the ECAC Doc 29
Vol. 2 Appendix B performance model in `flight_performance`, which flies a
published procedure into a profile for aircraft that ship no fixed-point
trajectory. `Spectral_classes.csv` is shipped for completeness and is not read.

The files are the published CSVs with their upstream column layout and values
unchanged (only the `ANP2.3_` filename prefix is dropped), so their provenance
is verifiable against the published tables.

## Licence / redistribution

The ANP database is published by EASA/EUROCONTROL free of charge for aircraft
noise and performance modelling and is redistributed here on that basis, for
reference and non-commercial use. It is not sold and carries no commercial
benefit; the authoritative and complete dataset remains the one published at the
source above. Point `load_anp_database(path=...)` at a directory of another
release to use different data.
