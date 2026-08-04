← [Documentation index](../../README.md)

# Absorbers

An absorber can be characterised at three scales, and this subsection walks
them in order: the finished product in a reverberation room, the small sample
in an impedance tube, and the raw material in a flow rig, with the prediction
models that tie the three together and the metamaterial designs that push
them past the classical thickness rules.

[Sound Absorption Measurement and Rating](absorption-measurement.md)
is the product scale: the ISO 354 reverberation-room measurement of the
random-incidence coefficient α_s, the ISO 11654 weighted rating α_w with its
letter class that absorber datasheets quote, and the ISO 12999-2 measurement
uncertainty of both. It also answers the recurring question of when a
reverberation-room number and a tube number can, and cannot, be compared.

[Airflow Resistance](airflow-resistance.md) is the material
scale: the ISO 9053-1 static and ISO 9053-2 alternating determination of the
airflow resistance, specific resistance and resistivity σ, the parameter that
governs a porous absorber's low-frequency behaviour and anchors every porous
model downstream.

[Impedance Tube](impedance-tube.md) is the sample scale: the
complex reflection factor, surface impedance and absorption at normal
incidence, by the ISO 10534-1 standing-wave-ratio and ISO 10534-2
transfer-function methods, plus the ASTM E2611 four-microphone transmission
loss, and the virtual FDTD tube that cross-checks the wave solver against the
same reduction chains.

[Porous and Multilayer Absorbers](porous-absorbers.md) closes
the loop with prediction: the Delany-Bazley, Miki and Johnson-Champoux-Allard
equivalent-fluid models turn the measured resistivity into characteristic
impedance and wavenumber, and the transfer-matrix multilayer solver predicts
the absorption of a whole construction, at any incidence and in a diffuse
field, before anything is built.

[Metamaterial Absorbers](metamaterial-absorbers.md) is where
the prediction models leave the classical rules behind: slow-sound slit
panels loaded by Helmholtz resonators reach perfect absorption at critical
coupling from panels a fortieth of a wavelength deep, with the
transfer-matrix model, the design solver and the FDTD cross-check of the
meshed cell.

## Pages in this section

- [Sound Absorption Measurement and Rating](absorption-measurement.md):
  the ISO 354 reverberation-room measurement, the ISO 11654 weighted rating
  and class, and the ISO 12999-2 measurement uncertainty.
- [Airflow Resistance](airflow-resistance.md): the ISO 9053
  static and alternating methods for the airflow resistance and resistivity.
- [Impedance Tube](impedance-tube.md): normal-incidence
  absorption, surface impedance and ASTM E2611 transmission loss, plus the
  virtual FDTD tube.
- [Porous and Multilayer Absorbers](porous-absorbers.md):
  the equivalent-fluid models and the transfer-matrix multilayer solver with
  perforated, microperforated and membrane layers.
- [Metamaterial Absorbers](metamaterial-absorbers.md): the
  critical-coupling condition and the slow-sound slit panel with its design
  solver.
