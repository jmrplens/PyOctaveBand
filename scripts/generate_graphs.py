import os
import pathlib
import sys

# Deterministic figure output: pin every numerical thread pool to a single
# thread BEFORE numpy/scipy/numba import their backends, so multi-threaded
# reductions cannot reorder floating-point sums and perturb the rendered bytes
# across machines (the CI "Documentation figures" job runs on a different core
# count than a dev box, which is what made the heavy compute figures flaky).
for _threads_var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import math
from collections.abc import Callable, Sequence
from functools import cache, lru_cache
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import signal as scipy_signal

# Add src to path to use the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from phonometry import OctaveFilterBank
from phonometry._plot.common import (
    _register_field_dark_cmap,
    format_frequency_axis,
    theme_fill,
    theme_fill_alpha,
)

# Constants for professional styling
# ---------------------------------------------------------------------------
# Language support: every figure is also generated in Spanish ("_es" suffix).
# Translation happens at savefig time by walking the figure's Text artists,
# so the generator functions stay single-language (English) internally.
_LANG = "en"
_LANG_SUFFIX = ""

_ES_EXACT = {
    "Bearing Fault Lines on a Measured Envelope Spectrum":
        "Líneas de fallo del rodamiento sobre un espectro de envolvente medido",
    "envelope spectrum of the 2-4 kHz band":
        "espectro de envolvente de la banda 2-4 kHz",
    "predicted BPFO and harmonics": "BPFO previsto y sus armónicos",
    "predicted BPFI": "BPFI previsto",
    "predicted BSF": "BSF previsto",
    "shaft rate": "frecuencia de giro del eje",
    "Envelope amplitude": "Amplitud de la envolvente",
    "15 rollers, D = 34 mm, d = 6 mm, φ = 12.96°, 2000 r/min\n"
    "BPFO = 207 Hz, BPFI = 293 Hz\n"
    "the envelope lines fall on BPFO, not on BPFI: outer-race spall":
        "15 rodillos, D = 34 mm, d = 6 mm, φ = 12,96°, 2000 r/min\n"
        "BPFO = 207 Hz, BPFI = 293 Hz\n"
        "las líneas de la envolvente caen en BPFO, no en BPFI: "
        "descascarillado de la pista exterior",
    "Coupling Loss Factors: Prediction and Power Injection":
        "Factores de pérdidas por acoplamiento: predicción e inyección de potencia",
    "Predicted: Weld Against Bolts":
        "Predicción: soldadura frente a tornillos",
    "Measured: Power Injection, 500 Hz Octave":
        "Medida: inyección de potencia, octava de 500 Hz",
    "Coupling loss factor": "Factor de pérdidas por acoplamiento",
    "Loss factor": "Factor de pérdidas",
    r"welded line junction $\eta_{12}$":
        r"unión lineal soldada $\eta_{12}$",
    r"12 bolts, point connections $\eta_{12}$":
        r"12 tornillos, conexiones puntuales $\eta_{12}$",
    r"internal loss factor $\eta_1$": r"factor de pérdidas interno $\eta_1$",
    "Plateau Estimate Against the Physical Panel Model":
        "Estimación por meseta frente al modelo físico del panel",
    "coincidence plateau (A to B)": "meseta de coincidencia (A a B)",
    "physical model (mass law + coincidence + damping)":
        "modelo físico (ley de masas + coincidencia + amortiguamiento)",
    "plateau estimate (Norton Table 3.1)":
        "estimación por meseta (Norton, tabla 3.1)",
    "critical frequency fc": "frecuencia crítica fc",
    "Transmission loss TL [dB]": "Pérdida por transmisión TL [dB]",
    "identical below A; the plateau replaces the whole coincidence region":
        "idénticas por debajo de A; la meseta sustituye toda la región de "
        "coincidencia",
    "Corrugating a Sheet Flattens Its Sound Reduction Index":
        "Grecar una chapa aplana su índice de reducción acústica",
    r"coincidence range $f_{c1}$ to $f_{c2}$":
        r"rango de coincidencia $f_{c1}$ a $f_{c2}$",
    r"flat 1 mm sheet (isotropic, single $f_c$)":
        r"chapa plana de 1 mm (isótropa, un solo $f_c$)",
    "corrugated sheet (orthotropic, diffuse-field integral)":
        "chapa grecada (ortótropa, integral en campo difuso)",
    "Heckl's approximation": "aproximación de Heckl",
    "A Limp Frame Carries Its Own Inertia":
        "Un esqueleto flexible arrastra su propia inercia",
    r"Normalised effective density $\rho_e/\rho_0$":
        r"Densidad efectiva normalizada $\rho_e/\rho_0$",
    "rigid frame, real part": "esqueleto rígido, parte real",
    "rigid frame, imaginary part": "esqueleto rígido, parte imaginaria",
    "limp frame, real part": "esqueleto flexible, parte real",
    "limp frame, imaginary part": "esqueleto flexible, parte imaginaria",
    "Only an Elastic Frame Resonates":
        "Solo un esqueleto elástico resuena",
    r"Normalised surface impedance $Z_s/\rho_0 c_0$":
        r"Impedancia superficial normalizada $Z_s/\rho_0 c_0$",
    "Biot poroelastic, real part": "Biot poroelástico, parte real",
    "Biot poroelastic, imaginary part": "Biot poroelástico, parte imaginaria",
    "Glass wool, 100 mm glued to a rigid wall: porosity 0.94,\n"
    "flow resistivity 40 kPa s/m\u00b2, frame density 130 kg/m\u00b3,\n"
    "shear modulus 2.2 MPa":
        "Lana de vidrio, 100 mm pegada a una pared r\u00edgida: porosidad 0,94,\n"
        "resistividad al flujo 40 kPa s/m\u00b2, densidad del esqueleto 130 kg/m\u00b3,\n"
        "m\u00f3dulo de cizalla 2,2 MPa",
    "2D FDTD wavefront in a hall of columns":
        "Frente de onda FDTD 2D en una sala de columnas",
    "loudspeaker": "altavoz",
    "anechoic termination": "terminación anecoica",
    "rigid plug": "tapón rígido",
    "|p| envelope": "envolvente |p|",
    "The 2 cm metadiffuser scatters like the 27 cm QRD (2 kHz)":
        "El metadifusor de 2 cm dispersa como el QRD de 27 cm (2 kHz)",
    "Metadiffuser, panel 2 cm": "Metadifusor, panel de 2 cm",
    "Far field of the meshed metadiffuser vs the model (2 kHz)":
        "Campo lejano del metadifusor mallado frente al modelo (2 kHz)",
    "TMM + Fraunhofer model": "Modelo TMM + Fraunhofer",
    "FDTD + NTFF, panel meshed at 0.5 mm":
        "FDTD + NTFF, panel mallado a 0,5 mm",
    "Schroeder diffuser vs metadiffuser (2D FDTD)":
        "Difusor de Schroeder frente a metadifusor (FDTD 2D)",
    "QRD, wells down to 27 cm": "QRD, pozos de hasta 27 cm",
    "Metadiffuser, 2 cm panel": "Metadifusor, panel de 2 cm",
    "real slits and resonators meshed at 0.25 mm":
        "rendijas y resonadores mallados a 0,25 mm",
    "a collimated specular beam": "un haz especular colimado",
    "a wide scattered fan": "un abanico dispersado ancho",
    "the same fan, from 2 cm": "el mismo abanico, con 2 cm",
    "QRD, wells up to 27.4 cm": "QRD, pozos de hasta 27,4 cm",
    "The virtual impedance tube: standing waves read the absorption "
    "(2D FDTD)":
        "El tubo de impedancia virtual: las ondas estacionarias leen la "
        "absorción (FDTD 2D)",
    "Rigid end: deep minima, |r| ~ 1":
        "Extremo rígido: mínimos profundos, |r| ~ 1",
    "10 cm lossy sample: shallow minima":
        "Muestra disipativa de 10 cm: mínimos poco profundos",
    "The virtual transmission tube: what gets through (2D FDTD)":
        "El tubo de transmisión virtual: lo que se transmite (FDTD 2D)",
    "Empty tube: the packet crosses unchanged":
        "Tubo vacío: el paquete lo cruza intacto",
    "10 cm lossy layer: reflected + attenuated transmission":
        "Capa disipativa de 10 cm: reflexión + transmisión atenuada",
    "Position along the tube [m]": "Posición a lo largo del tubo [m]",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "Rayleigh wave along a free aluminium surface (elastic FDTD)":
        "Onda de Rayleigh en una superficie libre de aluminio (FDTD elástico)",
    "P wave front": "Frente de onda P",
    "S wave front": "Frente de onda S",
    "Rayleigh wave": "Onda de Rayleigh",
    "Scholte wave along a water-sediment interface (elastic FDTD)":
        "Onda de Scholte en una interfase agua-sedimento (FDTD elástico)",
    "Scholte wave, evanescent on both sides":
        "Onda de Scholte, evanescente a ambos lados",
    "direct water wave": "onda directa en el agua",
    "water 1500 m/s": "agua 1500 m/s",
    "seabed 3500 / 2000 m/s": "sedimento 3500 / 2000 m/s",
    "Particle velocity [m/s]": "Velocidad de partícula [m/s]",
    "Predicted diffusion from design (Cox & D'Antonio Fraunhofer model)":
        "Difusión predicha desde el diseño (modelo de Fraunhofer de Cox y D'Antonio)",
    "Predicted diffusion coefficient d":
        "Coeficiente de difusión predicho d",
    "N = 7 QRD design": "Diseño QRD N = 7",
    "Flat panel": "Panel plano",
    "Atmospheric Refraction: Ray Bending and the Acoustic Shadow":
        "Refracción atmosférica: curvatura de rayos y zona de sombra",
    "Sound rays (upward refraction)": "Rayos sonoros (refracción hacia arriba)",
    "Tone-Burst Test Signal (IEC 60268-1)":
        "Señal de prueba de salvas de tono (IEC 60268-1)",
    "Single 5 ms burst of 5 kHz tone (25 full periods)":
        "Salva única de 5 ms de tono de 5 kHz (25 períodos completos)",
    "Repetitive train: 10 bursts per second (duty cycle 5 %)":
        "Tren repetitivo: 10 salvas por segundo (ciclo de trabajo 5 %)",
    "Gating envelope": "Envolvente de conmutación",
    "Window Functions: The Spectral Trade-off (Harris 1978)":
        "Ventanas de análisis: el compromiso espectral (Harris 1978)",
    "Parametric EQ Biquads (RBJ Audio EQ Cookbook)":
        "Biquads de EQ paramétrico (RBJ Audio EQ Cookbook)",
    "Peaking +6 dB (Q = 1.4)": "Campana +6 dB (Q = 1.4)",
    "Low shelf +6 dB": "Shelving grave +6 dB",
    "High shelf -6 dB": "Shelving agudo -6 dB",
    "Low-pass (Q = 0.707)": "Paso bajo (Q = 0.707)",
    "High-pass (Q = 0.707)": "Paso alto (Q = 0.707)",
    "Band-pass (Q = 2)": "Paso banda (Q = 2)",
    "Notch (Q = 6)": "Rechazo de banda (Q = 6)",
    "Frequency offset [DFT bins]": "Desplazamiento en frecuencia [bins de la DFT]",
    "Level re main lobe [dB]": "Nivel re lóbulo principal [dB]",
    "GFPE relative sound level": "Nivel sonoro relativo GFPE",
    "Shadow-zone boundary": "Límite de la zona de sombra",
    "Effective sound speed [m/s]": "Velocidad efectiva del sonido [m/s]",
    "Theoretical panel sound insulation (Bies / Hopkins / Cremer)":
        "Aislamiento acústico teórico de paneles (Bies / Hopkins / Cremer)",
    "Single panel: mass law and coincidence":
        "Panel simple: ley de masas y coincidencia",
    "Double wall: mass-spring-mass resonance":
        "Pared doble: resonancia masa-muelle-masa",
    "Radiation efficiency of a bending plate":
        "Eficiencia de radiación de una placa en flexión",
    "Composite wall with a small aperture":
        "Pared compuesta con una abertura pequeña",
    "Sound reduction index R [dB]": "Índice de reducción acústica R [dB]",
    r"Radiation efficiency $\sigma$": r"Eficiencia de radiación $\sigma$",
    "field-incidence mass law": "ley de masas de campo",
    "single panel R (Sharp)": "R de panel simple (Sharp)",
    "double wall R": "R de pared doble",
    "single leaf (total mass)": "hoja simple (masa total)",
    "solid wall alone": "pared maciza sola",
    "wall + 1 % open slit": "pared + rendija 1 % abierta",
    "open-area limit": "límite de área abierta",
    "Aircraft Atmospheric Absorption (SAE ARP 5534)":
        "Absorción atmosférica aeronáutica (SAE ARP 5534)",
    "Attenuation [dB]": "Atenuación [dB]",
    "Expansion-chamber transmission loss (Bies Eq. 8.111)":
        "Pérdida de transmisión de cámara de expansión (Bies Ec. 8.111)",
    "Area ratio m = Sexp/Sduct": "Relación de áreas m = Sexp/Sduct",
    "Noise-Power-Distance Curves (ECAC Doc 29)":
        "Curvas nivel-potencia-distancia (ECAC Doc 29)",
    "Aircraft Departure SEL Contour (ECAC Doc 29)":
        "Contorno SEL de despegue (ECAC Doc 29)",
    "Aircraft noise contour (ECAC Doc 29)": "Contorno de ruido de aeronave (ECAC Doc 29)",
    "Start-of-Roll Directivity ΔSOR (ECAC Doc 29 §4.5.7)":
        "Directividad de inicio de rodaje ΔSOR (ECAC Doc 29 §4.5.7)",
    "Turbofan jet (Eq. 4-24a)": "Reactor turbofán (Ec. 4-24a)",
    "Turboprop (Eq. 4-24b)": "Turbohélice (Ec. 4-24b)",
    "90°\nabeam": "90°\ntravés",
    "180° behind": "180° detrás",
    "radial axis: ΔSOR [dB] relative to abeam  ·  dSOR = 300 m":
        "eje radial: ΔSOR [dB] relativo al través  ·  dSOR = 300 m",
    "Rotorcraft Ground Effect (ECAC Doc 32, Chien-Soroka)":
        "Efecto de suelo de rotorcraft (ECAC Doc 32, Chien-Soroka)",
    "Rotorcraft Flyover Time History (ECAC Doc 32)":
        "Historia temporal de sobrevuelo de rotorcraft (ECAC Doc 32)",
    "Recorded time [s]": "Tiempo registrado [s]",
    "Rotorcraft Terrain Screening (ECAC Doc 32 / NORAH2)":
        "Apantallamiento por terreno de rotorcraft (ECAC Doc 32 / NORAH2)",
    "Terrain profile": "Perfil del terreno",
    "Line of sight": "Línea de visión",
    "Diffraction edges": "Aristas de difracción",
    "Terrain screening (ECAC Doc 32 / NORAH2 guidance)":
        "Apantallamiento por terreno (ECAC Doc 32 / guía NORAH2)",
    "Section distance [m]": "Distancia en la sección [m]",
    "Height [m]": "Altura [m]",
    "Flat ground (no hill)": "Suelo plano (sin colina)",
    "Screened by the hill (Eq. 45-47)": "Apantallado por la colina (Ec. 45-47)",
    "Ground and screening adjustment [dB]": "Ajuste de suelo y apantallamiento [dB]",
    "A-weighted sound pressure level [dB(A)]":
        "Nivel de presión acústica ponderado A [dB(A)]",
    "Received level $L_A(t)$": "Nivel recibido $L_A(t)$",
    "One-third-octave-band centre frequency [Hz]":
        "Frecuencia central de banda de 1/3 de octava [Hz]",
    "Ground-effect adjustment ΔLg [dB]": "Ajuste por efecto de suelo ΔLg [dB]",
    "Hard (asphalt/concrete, class G)": "Duro (asfalto/hormigón, clase G)",
    "Soft (grass/pasture, class D)": "Blando (hierba/pasto, clase D)",
    "x [km]": "x [km]",
    "y [km]": "y [km]",
    "Slant distance [m]": "Distancia oblicua [m]",
    "Event level [dB]": "Nivel de evento [dB]",
    "markers: tabulated NPD nodes\nlines: log-linear interpolation":
        "marcadores: nodos NPD tabulados\nlíneas: interpolación log-lineal",
    "25 °C, 70% RH\nsolid: SAE band, dashed: pure-tone mid-band":
        "25 °C, 70% HR\ncontinuo: banda SAE, discontinuo: tono puro medio de banda",
    # Emitted by phonometry.metrology.filter_design._showfilter (not by this script);
    # do not remove as "orphans".
    "Filter Bank Frequency Response": "Respuesta en frecuencia del banco de filtros",
    "Amplitude [dB]": "Amplitud [dB]",
    "Level [dB]": "Nivel [dB]",
    "Time [s]": "Tiempo [s]",
    "Amplitude": "Amplitud",
    "Error [dB]": "Error [dB]",
    "Group delay [ms]": "Retardo de grupo [ms]",
    "Level re steady state [dB]": "Nivel re estado estacionario [dB]",
    # ISO 18233 excitation signals + recovered impulse response
    "ISO 18233 excitation signals": "Señales de excitación ISO 18233",
    "Exponential sine sweep — waveform":
        "Barrido sinusoidal exponencial — forma de onda",
    "Sweep spectrogram (exponential rise)":
        "Espectrograma del barrido (ascenso exponencial)",
    "MLS magnitude spectrum (flat)": "Espectro de magnitud de la MLS (plano)",
    "Recovered room impulse response (ISO 18233)":
        "Respuesta al impulso de la sala recuperada (ISO 18233)",
    "Amplitude (norm.)": "Amplitud (norm.)",
    "Level re peak [dB]": "Nivel re pico [dB]",
    "Magnitude [dB]": "Magnitud [dB]",
    "Sample": "Muestra",
    "direct sound": "sonido directo",
    "reflections": "reflexiones",
    "Log-magnitude envelope": "Envolvente log-magnitud",
    "Schroeder decay (EDC)": "Decaimiento de Schroeder (EDC)",
    "Normalized Response": "Respuesta normalizada",
    "Normalized frequency  f / fm": "Frecuencia normalizada  f / fm",
    "Relative attenuation \u0394A [dB]": "Atenuaci\u00f3n relativa \u0394A [dB]",
    "Sound pressure level [dB re 20 \u00b5Pa]": "Nivel de presi\u00f3n ac\u00fastica [dB re 20 \u00b5Pa]",
    "1/3 Octave Band Analysis": "An\u00e1lisis en bandas de octava 1/3",
    "1/12 Octave Spectrogram (Fast windows, 87.5% overlap)":
        "Espectrograma 1/12 de octava (ventanas Fast, 87,5 % de solape)",
    "4 kHz Toneburst Response vs IEC 61672-1 Table 4 (FAST)":
        "Respuesta a r\u00e1fagas de 4 kHz vs Tabla 4 de IEC 61672-1 (FAST)",
    "A-Weighting": "Ponderaci\u00f3n A",
    "A-Weighting (reference)": "Ponderaci\u00f3n A (referencia)",
    "B-Weighting (historical)": "Ponderaci\u00f3n B (hist\u00f3rica)",
    "C-Weighting": "Ponderaci\u00f3n C",
    "D-Weighting (aircraft, withdrawn)":
        "Ponderaci\u00f3n D (aeronaves, retirada)",
    "AU-Weighting (audible + ultrasound)":
        "Ponderaci\u00f3n AU (audible + ultrasonido)",
    "Z-Weighting (Flat)": "Ponderaci\u00f3n Z (plana)",
    "G-weighting (ISO 7196)": "Ponderaci\u00f3n G (ISO 7196)",
    "G Frequency Weighting for Infrasound (ISO 7196:1995)":
        "Ponderaci\u00f3n frecuencial G para infrasonido (ISO 7196:1995)",
    "Bessel": "Bessel",
    "Bilinear error": "Error del dise\u00f1o bilineal",
    "Butterworth": "Butterworth",
    "Butterworth (Flat)": "Butterworth (plano)",
    "Butterworth order 6 (1 kHz octave band)":
        "Butterworth de orden 6 (banda de octava de 1 kHz)",
    "Causal filtering (group delay)": "Filtrado causal (retardo de grupo)",
    "Chebyshev I": "Chebyshev I",
    "Chebyshev II": "Chebyshev II",
    "Class 1 lower limit @ 12.5 kHz": "L\u00edmite inferior de clase 1 @ 12,5 kHz",
    "Class 2 minimum attenuation": "Atenuaci\u00f3n m\u00ednima de clase 2",
    "Continuous (whole signal)": "Continuo (se\u00f1al completa)",
    "Elliptic": "El\u00edptico",
    "FAST envelope": "Envolvente FAST",
    "Fast (125ms)": "Fast (125 ms)",
    "Fast level $L_p(t)$": "Nivel Fast $L_p(t)$",
    "Filter Architecture Comparison (Order 6, 1kHz Band)":
        "Comparativa de arquitecturas de filtro (orden 6, banda de 1 kHz)",
    "Forbidden for class 1 (too little attenuation)":
        "Prohibido para clase 1 (atenuaci\u00f3n insuficiente)",
    "Forbidden for class 1 (too much attenuation)":
        "Prohibido para clase 1 (atenuaci\u00f3n excesiva)",
    # weighting_class_mask figure (IEC 61672-1 Table 3 verifier)
    "Weighting Deviation vs IEC 61672-1:2013 Table 3 Limits":
        "Desviaci\u00f3n de ponderaci\u00f3n vs l\u00edmites de la Tabla 3 de IEC 61672-1:2013",
    "Class 1 acceptance region": "Regi\u00f3n de aceptaci\u00f3n de clase 1",
    "Class 1 upper/lower limit": "L\u00edmite superior/inferior de clase 1",
    "Class 2 upper/lower limit": "L\u00edmite superior/inferior de clase 2",
    "A weighting deviation (48 kHz)": "Desviaci\u00f3n de ponderaci\u00f3n A (48 kHz)",
    "C weighting deviation (48 kHz)": "Desviaci\u00f3n de ponderaci\u00f3n C (48 kHz)",
    "Deviation from design goal [dB]": "Desviaci\u00f3n del objetivo de dise\u00f1o [dB]",
    "Frequency Weighting Curves (IEC 61672-1)":
        "Curvas de ponderaci\u00f3n frecuencial (IEC 61672-1)",
    # special_weighting_responses figure (B, D and AU against the A reference)
    "Special Weighting Curves (B, D, AU)":
        "Curvas de ponderaci\u00f3n especiales (B, D y AU)",
    "AU is 13 dB below A at 16 kHz":
        "AU queda 13 dB por debajo de A en 16 kHz",
    "Group Delay Comparison (1 kHz Octave Band, Order 6)":
        "Comparativa de retardo de grupo (banda de 1 kHz, orden 6)",
    "Hearing threshold $T_f$ (Table 1)": "Umbral de audici\u00f3n $T_f$ (Tabla 1)",
    "Hearing threshold $T_f$": "Umbral de audici\u00f3n $T_f$",
    "High Pass (LR4)": "Paso alto (LR4)",
    "IEC 61672-1 analytic curve": "Curva anal\u00edtica IEC 61672-1",
    "ISO 7196 Table 2 nominals": "Nominales de la Tabla 2 de ISO 7196",
    "Impulse (35ms/1.5s)": "Impulse (35 ms/1,5 s)",
    "Independent blocks (state reset)": "Bloques independientes (estado reiniciado)",
    "Input Burst (Normalized)": "R\u00e1faga de entrada (normalizada)",
    "Input burst (250 Hz)": "R\u00e1faga de entrada (250 Hz)",
    "Left Channel: Pink Noise": "Canal izquierdo: ruido rosa",
    "Linkwitz-Riley Crossover (4th Order @ 1kHz)":
        "Crossover Linkwitz-Riley (4\u00ba orden @ 1 kHz)",
    "Low Pass (LR4)": "Paso bajo (LR4)",
    "Multichannel Analysis (Stereo Input)": "An\u00e1lisis multicanal (entrada est\u00e9reo)",
    "No state: each block restarts the filter transient":
        "Sin estado: cada bloque reinicia el transitorio del filtro",
    "Normal Equal-Loudness-Level Contours (ISO 226:2023)":
        "L\u00edneas isof\u00f3nicas normales (ISO 226:2023)",
    "ISO 226:2023 defines 20 to 90 phon; above 80 phon the contour is defined only up to 4 kHz.":
        "ISO 226:2023 define de 20 a 90 fonios; por encima de 80 fonios la curva solo se define hasta 4 kHz.",
    "Original Signal (250 Hz + 1000 Hz Sum) @ 48 kHz":
        "Se\u00f1al original (suma de 250 Hz + 1000 Hz) @ 48 kHz",
    "Oversampled (high_accuracy=True)": "Sobremuestreado (high_accuracy=True)",
    "Plain bilinear (high_accuracy=False)": "Bilineal simple (high_accuracy=False)",
    "Raw PSD": "PSD sin filtrar",
    "Raw Signal Spectrum (PSD)": "Espectro de la se\u00f1al (PSD)",
    "Relative Attenuation vs IEC 61260-1:2014 Class Limits":
        "Atenuaci\u00f3n relativa vs l\u00edmites de clase de IEC 61260-1:2014",
    "Right Channel: Log Sine Sweep": "Canal derecho: barrido senoidal logar\u00edtmico",
    "Slow (1000ms)": "Slow (1000 ms)",
    "Stateful blocks (state carried)": "Bloques con estado (estado conservado)",
    "Statistical Levels L10 / L50 / L90 (Fast envelope)":
        "Niveles estad\u00edsticos L10 / L50 / L90 (envolvente Fast)",
    "Sum (Flat)": "Suma (plana)",
    "Time Weighting Ballistics (IEC 61672-1)":
        "Ponderaci\u00f3n temporal F/S/I (IEC 61672-1)",
    "Zero-Phase Filtering: Group Delay Elimination (250 Hz Band)":
        "Filtrado de fase cero: eliminaci\u00f3n del retardo de grupo (banda de 250 Hz)",
    "Zoom at -3 dB (Log Scale)": "Zoom en -3 dB (escala log)",
    "Zoom: A-weighting is positive (max +1.27 dB @ 2.5 kHz)":
        "Zoom: la ponderaci\u00f3n A es positiva (m\u00e1x +1,27 dB @ 2,5 kHz)",
    "block boundary:\nfilter transient restarts":
        "frontera de bloque:\nse reinicia el transitorio del filtro",
    "high_accuracy error": "Error con high_accuracy",
    "stateful=True: block outputs equal the continuous result":
        "stateful=True: los bloques igualan el resultado continuo",
    "zero_phase=True (aligned)": "zero_phase=True (alineado)",
    "0 dB @ 10 Hz": "0 dB @ 10 Hz",
    # filter_class0_mask figure (IEC 61260:1995 / ANSI S1.11-2004 class 0)
    "Pass-band Class 0/1/2 Limits (IEC 61260:1995 / ANSI S1.11-2004)":
        "Límites de clase 0/1/2 en banda de paso (IEC 61260:1995 / ANSI S1.11-2004)",
    "Class 0 corridor": "Corredor de clase 0",
    "Class 1 corridor": "Corredor de clase 1",
    "Class 2 corridor": "Corredor de clase 2",
    # intensity_insulation figure (ISO 15186-1)
    "ISO 15186-1 Intensity Sound Reduction Index (RI and RI,M)":
        "Índice de reducción acústica por intensidad ISO 15186-1 (RI y RI,M)",
    "Sound reduction index [dB]": "Índice de reducción acústica [dB]",
    "Kc adaptation": "Adaptación Kc",
    "RI (intensity)": "RI (intensidad)",
    # survey_insulation figure (ISO 10052)
    "ISO 10052 Survey Method: Reverberation-Index Correction":
        "Método de control ISO 10052: corrección por índice de reverberación",
    "Level difference [dB]": "Diferencia de nivel [dB]",
    "D (level difference)": "D (diferencia de nivel)",
    "DnT (standardized)": "DnT (estandarizada)",
    "octave bands, T0 = 0.5 s": "bandas de octava, T0 = 0,5 s",
    # absorption_uncertainty figure (ISO 12999-2)
    "ISO 12999-2 Sound Absorption Coefficient Uncertainty":
        "Incertidumbre del coeficiente de absorción acústica (ISO 12999-2)",
    "+/-U (k = 2), reproducibility": "±U (k = 2), reproducibilidad",
    "alpha_s (ISO 354)": "alpha_s (ISO 354)",
    # floor_covering_improvement figure (ISO 16251-1)
    "ISO 16251-1 Floor-Covering Impact Sound Improvement":
        "Mejora a impacto de revestimientos de suelo (ISO 16251-1)",
    "Improvement of impact sound insulation [dB]":
        "Mejora del aislamiento a impactos [dB]",
    "delta-L (improvement)": "delta-L (mejora)",
    # heavy_impact_sources figure (ISO 16283-2 / JIS A 1418-2 / ISO 717-2)
    "Standard heavy impact sources\n(ISO 16283-2 Table A.1, JIS A 1418-2 Tables A.1/A.2)":
        "Fuentes de impacto pesadas normalizadas\n(ISO 16283-2 Tabla A.1, "
        "JIS A 1418-2 Tablas A.1/A.2)",
    "Impact force exposure level LFE [dB re 1 N]":
        "Nivel de exposición a la fuerza de impacto LFE [dB re 1 N]",
    "rubber ball tolerance": "tolerancia de la pelota de caucho",
    "rubber ball nominal": "pelota de caucho nominal",
    "bang machine tolerance": "tolerancia de la máquina de neumático",
    "bang machine nominal": "máquina de neumático nominal",
    "A-weighted heavy-impact rating\n(ISO 717-2 Annex D, Table D.4 worked example)":
        "Índice de impacto pesado ponderado A\n(ISO 717-2 Anexo D, ejemplo de "
        "la Tabla D.4)",
    "Maximum impact sound pressure level [dB]":
        "Nivel máximo de presión acústica de impactos [dB]",
    "Li,Fmax (measured)": "Li,Fmax (medido)",
    "Li,Fmax + A (Table D.3)": "Li,Fmax + A (Tabla D.3)",
    # ceiling_plenum_flanking figure (Vigran 9.2.3 / ASTM E1414 / ASTM E413)
    "Suspended-ceiling plenum path\n(one-dimensional model, LR = 4.75 m, reflecting sidewalls)":
        "Trayecto por plenum de techo suspendido\n(modelo unidimensional, "
        "LR = 4.75 m, paredes laterales reflectantes)",
    "RS + RR (two ceilings)": "RS + RR (dos techos)",
    "Ceiling attenuation class\n(ASTM E1414/E413, CAC = 34 dB)":
        "Clase de atenuación de techo\n(ASTM E1414/E413, CAC = 34 dB)",
    "Normalized ceiling attenuation Dn,c [dB]":
        "Diferencia de niveles normalizada del techo Dn,c [dB]",
    "Dn,c (measured)": "Dn,c (medido)",
    "ASTM E413 contour, fitted": "curva ASTM E413 ajustada",
    "deficiencies": "deficiencias",
    # masonry_wall_ties figure (Hopkins 3.11.3.2 / 4.3.5.4.1)
    "Wall-tie structure-borne coupling\n(point-connection model, 2.5 ties/m2)":
        "Acoplamiento estructural por llaves de muro\n(modelo de unión puntual, "
        "2.5 llaves/m2)",
    "Coupling loss factor eta_ij": "Factor de pérdidas por acoplamiento eta_ij",
    "rigid connection (Yc = 0)": "unión rígida (Yc = 0)",
    "Ties stiffen the cavity\n(140 kg/m2 leaves, 75 mm cavity, 2.5 ties/m2)":
        "Las llaves rigidizan la cámara\n(hojas de 140 kg/m2, cámara de 75 mm, "
        "2.5 llaves/m2)",
    "cavity wall, no ties": "muro con cámara, sin llaves",
    "2.5 ties/m2, k = 2 MN/m": "2,5 llaves/m2, k = 2 MN/m",
    "combined-mass range added by the ties":
        "rango de masa combinada añadido por las llaves",
    # flanking_transmission figure (ISO 10848)
    "ISO 10848 Junction Vibration Reduction Index":
        "Índice de reducción de vibraciones de unión (ISO 10848)",
    "Vibration reduction index Kij [dB]":
        "Índice de reducción de vibraciones Kij [dB]",
    "Kij (ISO 10848)": "Kij (ISO 10848)",
    "mean Kij (200-1250 Hz)": "Kij medio (200-1250 Hz)",
    # tonal_audibility figure (ISO 1996-2)
    "ISO 1996-2 Tonal Adjustment": "Ajuste tonal ISO 1996-2",
    r"Tonal audibility $\Delta L_{ta}$ [dB]":
        r"Audibilidad tonal $\Delta L_{ta}$ [dB]",
    r"Tonal adjustment $K_t$ [dB]": r"Ajuste tonal $K_t$ [dB]",
    r"$K_t(\Delta L_{ta})$ (Formulae C.4-C.6)":
        r"$K_t(\Delta L_{ta})$ (Fórmulas C.4-C.6)",
    "Annex C.5 examples": "ejemplos del Anexo C.5",
    "mid-range tone": "tono de rango medio",
    # reverberation_models figure (Sabine / Eyring / Millington / Fitzroy / Arau)
    "Reverberation-time prediction models":
        "Modelos de predicción del tiempo de reverberación",
    # dynamic_stiffness figure (EN 29052-1)
    "EN 29052-1 Floating-Floor Resonance":
        "Resonancia de suelo flotante EN 29052-1",
    r"Dynamic stiffness per unit area $s'$ [MN/m³]":
        r"Rigidez dinámica por unidad de área $s'$ [MN/m³]",
    r"Natural frequency $f_0$ [Hz]": r"Frecuencia natural $f_0$ [Hz]",
    "design point": "punto de diseño",
    # junction_transmission figure (Hopkins 5.2.1.3, Cremer/Craik)
    "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)":
        "Transmisión de onda de flexión en una unión X rígida (Hopkins 5.2.1.3)",
    "Incidence angle [degrees]": "Ángulo de incidencia [grados]",
    r"Transmission coefficient $\tau$": r"Coeficiente de transmisión $\tau$",
    r"corner $\tau_{12}(\theta)$": r"esquina $\tau_{12}(\theta)$",
    r"straight $\tau_{13}(\theta)$": r"recta $\tau_{13}(\theta)$",
    "corner average": "media esquina",
    "straight average": "media recta",
    # mechanical_mobility figure (ISO 7626-1)
    "ISO 7626-1 Mechanical Mobility FRFs":
        "FRF de movilidad mecánica ISO 7626-1",
    "Normalized FRF magnitude": "Magnitud FRF normalizada",
    "Receptance $|H|$ (× k)": "Receptancia $|H|$ (× k)",
    r"Mobility $|Y|$ (× k/$\omega_0$)": r"Movilidad $|Y|$ (× k/$\omega_0$)",
    r"Accelerance $|A|$ (× k/$\omega_0^2$)":
        r"Acelerancia $|A|$ (× k/$\omega_0^2$)",
    "resonance $f_0$": "resonancia $f_0$",
    # transfer_stiffness figure (ISO 10846)
    "ISO 10846 Dynamic Transfer Stiffness":
        "Rigidez dinámica de transferencia ISO 10846",
    r"Transfer stiffness level $L_k$ [dB re 1 N/m]":
        r"Nivel de rigidez de transferencia $L_k$ [dB re 1 N/m]",
    r"true $L_k$ of $k_{2,1}=k+j\omega c$":
        r"$L_k$ real de $k_{2,1}=k+j\omega c$",
    r"indirect method $-(2\pi f)^2 m_2 T$":
        r"método indirecto $-(2\pi f)^2 m_2 T$",
    # rigid_mass_calibration figure (ISO 7626-2, 7.5.2)
    "ISO 7626-2 Rigid-Mass Calibration Check":
        "Verificación de calibración con masa rígida ISO 7626-2",
    "Accelerance $|A|$ [1/kg]": "Acelerancia $|A|$ [1/kg]",
    "Deviation [%]": "Desviación [%]",
    r"expected $|A| = 1/m$": r"esperado $|A| = 1/m$",
    r"$\pm$5 % tolerance band": r"banda de tolerancia $\pm$5 %",
    "within tolerance": "dentro de tolerancia",
    "out of tolerance": "fuera de tolerancia",
    # vibration_sound_power figure (ISO/TS 7849)
    "ISO/TS 7849 Sound Power from Surface Vibration":
        "Potencia acústica desde vibración superficial ISO/TS 7849",
    r"Sound power level $L_W$ [dB re 1 pW]":
        r"Nivel de potencia acústica $L_W$ [dB re 1 pW]",
    "Part 1 upper limit ($\\varepsilon$ = 1)":
        "Parte 1 límite superior ($\\varepsilon$ = 1)",
    "Part 2 engineering ($\\varepsilon$ measured)":
        "Parte 2 ingeniería ($\\varepsilon$ medido)",
    # structure_borne_power figure (EN 15657)
    "EN 15657 Reception-Plate Structure-Borne Sound Power":
        "Potencia acústica estructural en placa receptora EN 15657",
    r"Structure-borne power level $L_{Ws}$ [dB re 1 pW]":
        r"Nivel de potencia estructural $L_{Ws}$ [dB re 1 pW]",
    "low-mobility plate": "placa de baja movilidad",
    "high-mobility plate": "placa de alta movilidad",
    # installed_structure_borne figure (EN 12354-5)
    "EN 12354-5 Installed Structure-Borne Sound":
        "Ruido estructural instalado EN 12354-5",
    r"characteristic $L_{Ws,c}$ (EN 15657)":
        r"característica $L_{Ws,c}$ (EN 15657)",
    r"installed $L_{Ws,inst}$ = $L_{Ws,c}-D_C$":
        r"instalada $L_{Ws,inst}$ = $L_{Ws,c}-D_C$",
    "paths $L_{n,s,ij}$": "caminos $L_{n,s,ij}$",
    r"total $L_{n,s}$": r"total $L_{n,s}$",
    # tone_audibility figure (ISO/PAS 20065)
    "ISO/PAS 20065 Tonal Audibility": "Audibilidad tonal ISO/PAS 20065",
    r"Audibility $\Delta L$ [dB]": r"Audibilidad $\Delta L$ [dB]",
    r"threshold $\Delta L = 0$ dB": r"umbral $\Delta L = 0$ dB",
    # facade_prediction figure (EN 12354-3 Annex F)
    "EN 12354-3 Façade Sound Insulation (Annex F example)":
        "Aislamiento acústico de fachada EN 12354-3 (ejemplo del Anexo F)",
    "Reduction index / level difference [dB]":
        "Índice de reducción / diferencia de niveles [dB]",
    "Rp — wall": "Rp — muro",
    "Rp — window": "Rp — ventana",
    "Rp — skylight": "Rp — claraboya",
    "Rp — air inlet": "Rp — entrada de aire",
    "R′ (façade)": "R′ (fachada)",
    "air inlet limits the low bands": "la entrada de aire limita las bandas bajas",
    # Scattering coefficient spectrum (ISO 17497-1)
    "Random-incidence scattering coefficient (ISO 17497-1)":
        "Coeficiente de dispersión de incidencia aleatoria (ISO 17497-1)",
    "Scattering coefficient s": "Coeficiente de dispersión s",
    # In-situ road-surface absorption (ISO 13472-1)
    "In-situ road-surface absorption (ISO 13472-1)":
        "Absorción in situ de pavimentos (ISO 13472-1)",
    "Absorption coefficient alpha": "Coeficiente de absorción alpha",
    "Porous layer 50 mm (sigma = 20 kPa s/m2)":
        "Capa porosa de 50 mm (sigma = 20 kPa s/m2)",
    "Microperforated panel + 48 mm cavity":
        "Panel microperforado + cámara de 48 mm",
    "Perforated panel 6 mm + porous 25 mm + air":
        "Panel perforado 6 mm + poroso 25 mm + aire",
    "Membrane 2 kg/m2 + air + porous 38 mm":
        "Membrana 2 kg/m2 + aire + poroso 38 mm",
    "Helmholtz closed form": "Forma cerrada de Helmholtz",
    "Membrane closed form": "Forma cerrada de membrana",
    "Multilayer Absorber Prediction (Transfer-Matrix Method)":
        "Predicción de absorbentes multicapa (método de matrices de transferencia)",
    "Normal incidence, rigid backing, 50 mm total depth":
        "Incidencia normal, respaldo rígido, 50 mm de profundidad total",
    # Human vibration (ISO 8041-1 / ISO 2631 / ISO 5349 / 2002/44/EC)
    "Whole-body vertical weighting Wk (ISO 8041-1)":
        "Ponderación vertical de cuerpo entero Wk (ISO 8041-1)",
    "Weighting factor [dB]": "Factor de ponderación [dB]",
    "One-third-octave band [Hz]": "Banda de tercio de octava [Hz]",
    "Band audibility": "Audibilidad de banda",
    r"Band audibility $A_i$": r"Audibilidad de banda $A_i$",
    r"Importance-weighted $I_i\,A_i$ (scaled)":
        r"Ponderada por importancia $I_i\,A_i$ (escalada)",
    "r.m.s. acceleration [m/s$^2$]": "Aceleración eficaz [m/s$^2$]",
    "Unweighted $a_i$": "Sin ponderar $a_i$",
    "Weighted $W_i\\,a_i$ (Wk)": "Ponderada $W_i\\,a_i$ (Wk)",
    "Daily exposure A(8) [m/s$^2$]": "Exposición diaria A(8) [m/s$^2$]",
    "brush-saw": "desbrozadora",
    "felling": "tala",
    "stripping": "descortezado",
    # Precision sound power (ISO 3745 / ISO 9614-3)
    "Sound power level LW [dB]": "Nivel de potencia acústica LW [dB]",
    "Non-applicable band": "Banda no aplicable",
    "Stable tone (good coupling)": "Tono estable (buen acoplamiento)",
    "3% AM tone (loose coupling)": "Tono con AM del 3 % (acoplamiento flojo)",
    "IEC 60942:2017 class 1 limit (deviation from mean)":
        "L\u00edmite de clase 1 de IEC 60942:2017 (desviaci\u00f3n de la media)",
    "Calibration Tone Stability Check (IEC 60942:2017, 5.3.3)":
        "Comprobaci\u00f3n de estabilidad del tono de calibraci\u00f3n (IEC 60942:2017, 5.3.3)",
    "F-weighted level re mean [dB]": "Nivel con ponderaci\u00f3n F re media [dB]",
    "Fast level of the event": "Nivel Fast del evento",
    "Leq over the whole event": "Leq de todo el evento",
    "SEL: same energy in 1 s": "SEL: la misma energ\u00eda en 1 s",
    "equal energy": "igual energ\u00eda",
    "Sound Exposure Level: the event normalized to 1 s":
        "Nivel de exposici\u00f3n sonora: el evento normalizado a 1 s",
    "Level [dBFS]": "Nivel [dBFS]",
    "Hourly LAeq": "LAeq horario",
    "Lday (+0 dB)": "Ld\u00eda (+0 dB)",
    "Levening + 5 dB": "Ltarde + 5 dB",
    "Lnight + 10 dB": "Lnoche + 10 dB",
    "Day-Evening-Night Level Lden (ISO 1996-1)":
        "Nivel d\u00eda-tarde-noche Lden (ISO 1996-1)",
    "Hour of day": "Hora del d\u00eda",
    "Averaged FFT spectrum (Hann)": "Espectro FFT promediado (Hann)",
    "Critical band around the tone": "Banda cr\u00edtica en torno al tono",
    "Tone-to-Noise Ratio (ECMA-418-1, clause 11)":
        "Relaci\u00f3n tono-ruido (ECMA-418-1, apartado 11)",
    "Bin power [dB]": "Potencia por bin [dB]",
    "Specific Loudness Pattern (ISO 532-1 Zwicker)":
        "Patr\u00f3n de sonoridad espec\u00edfica (Zwicker, ISO 532-1)",
    "Critical-band rate z [Bark]": "Raz\u00f3n de banda cr\u00edtica z [Bark]",
    "Specific loudness N' [sone/Bark]":
        "Sonoridad espec\u00edfica N' [sonios/Bark]",
    "Shaded area = total loudness N": "\u00c1rea sombreada = sonoridad total N",
    "STI vs Reverberation Time (IEC 60268-16)":
        "STI frente al tiempo de reverberaci\u00f3n (IEC 60268-16)",
    "Reverberation time T60 [s]": "Tiempo de reverberaci\u00f3n T60 [s]",
    "Analytic Schroeder MTF (closed form)":
        "MTF de Schroeder anal\u00edtica (forma cerrada)",
    "Measured (sti_from_impulse_response)":
        "Medido (sti_from_impulse_response)",
    "Annex F rating": "Calificaci\u00f3n del Anexo F",
    "Measured": "Medido",
    "Octave-band center frequency [Hz]":
        "Frecuencia central de banda de octava [Hz]",
    "Octave-band sound pressure level [dB]":
        "Nivel de presi\u00f3n ac\u00fastica por banda de octava [dB]",
    "Rumble tol. (+5 dB)": "Tol. retumbo (+5 dB)",
    "Hiss tol. (+3 dB)": "Tol. siseo (+3 dB)",
    "ISO 7029 — age-related threshold (male)":
        "ISO 7029 — umbral por edad (hombres)",
    "ISO 389-7 — reference threshold of hearing":
        "ISO 389-7 — umbral de referencia de la audición",
    "Audiometric frequency [Hz]": "Frecuencia audiométrica [Hz]",
    "Median threshold deviation from age 18 [dB]":
        "Desviación mediana del umbral respecto a los 18 años [dB]",
    "Reference threshold [dB]": "Umbral de referencia [dB]",
    "Free-field (frontal)": "Campo libre (frontal)",
    "Diffuse-field": "Campo difuso",
    "ANSI S3.5-1997 — speech spectra by vocal effort":
        "ANSI S3.5-1997 — espectros de voz por esfuerzo vocal",
    "Speech spectrum level [dB SPL]": "Nivel del espectro de voz [dB SPL]",
    "SII vs vocal effort in a fixed noise":
        "SII frente al esfuerzo vocal en un ruido fijo",
    "Onset rate [dB/s]": "Tasa de crecimiento [dB/s]",
    "Predicted prominence $P$": "Prominencia prevista $P$",
    "Adjustment $K_I$ [dB]": "Ajuste $K_I$ [dB]",
    "Adjustment to $L_{Aeq}$": "Ajuste a $L_{Aeq}$",
    "Impulses": "Impulsos",
    "threshold $P = 5$": "umbral $P = 5$",
    "Transmissibility  seat $\\rightarrow$ spine":
        "Transmisibilidad  asiento $\\rightarrow$ columna",
    "Seat-to-spine transfer function":
        "Función de transferencia asiento-columna",
    "Stress variable $R$": "Variable de tensión $R$",
    "Probability of lumbar injury [%]": "Probabilidad de lesión lumbar [%]",
    "Injury probability (Annex C)": "Probabilidad de lesión (Anexo C)",
    "male": "hombre",
    "female": "mujer",
    "Equivalent absorption area $A$ [m$^2$]":
        "Área de absorción equivalente $A$ [m$^2$]",
    "Absorption area (Formula 1)": "Área de absorción (Fórmula 1)",
    "Reverberation time $T$ [s]": "Tiempo de reverberación $T$ [s]",
    "Reverberation time (Formula 5)": "Tiempo de reverberación (Fórmula 5)",
    "bare ceiling": "techo desnudo",
    "acoustic ceiling": "techo acústico",
    "Speech Intelligibility Index": "Índice de inteligibilidad del habla",
    "Normal": "Normal",
    "Raised": "Elevada",
    "Loud": "Fuerte",
    "Shout": "Grito",
    r"ISO 1999 — NIPTS at $L_{EX,8h}$ = 95 dB":
        r"ISO 1999 — NIPTS a $L_{EX,8h}$ = 95 dB",
    "ISO 1999 — HTLAN (male, age 60, 95 dB / 30 yr)":
        "ISO 1999 — HTLAN (hombres, 60 años, 95 dB / 30 años)",
    "Median NIPTS [dB]": "NIPTS mediana [dB]",
    "Hearing threshold level [dB]": "Nivel del umbral de audición [dB]",
    "10-90 % band (40 yr)": "Banda 10-90 % (40 años)",
    "Age (HTLA, ISO 7029)": "Edad (HTLA, ISO 7029)",
    "Noise (NIPTS)": "Ruido (NIPTS)",
    "Age + noise (HTLAN)": "Edad + ruido (HTLAN)",
    "GUM uncertainty budget": "Presupuesto de incertidumbre (GUM)",
    "Contribution to combined uncertainty [dB]":
        "Contribución a la incertidumbre combinada [dB]",
    "Monte Carlo (Suppl 1)": "Monte Carlo (Supl. 1)",
    "GUM Gaussian": "Gaussiana GUM",
    "95 % coverage interval": "Intervalo de cobertura 95 %",
    "A-weighted level [dB]": "Nivel ponderado A [dB]",
    "Probability density": "Densidad de probabilidad",
    "Reading": "Lectura",
    "Calibration": "Calibración",
    "Instrument": "Instrumento",
    "Position (Type A)": "Posición (Tipo A)",
    "Sound Intensity with a p-p Probe (IEC 61043)":
        "Intensidad acústica con sonda p-p (IEC 61043)",
    "Plane wave: Lp \u2248 LI": "Onda plana: Lp \u2248 LI",
    "Standing wave: reactive field": "Onda estacionaria: campo reactivo",
    "Pressure level Lp": "Nivel de presi\u00f3n Lp",
    "Intensity level LI": "Nivel de intensidad LI",
    "Schroeder Integration and Reverberation Time (ISO 3382)":
        "Integraci\u00f3n de Schroeder y tiempo de reverberaci\u00f3n (ISO 3382)",
    "Raw squared IR level": "Nivel de la RI al cuadrado",
    "Schroeder decay curve": "Curva de ca\u00edda de Schroeder",
    "T20 fit (\u22125 to \u221225 dB)": "Ajuste T20 (\u22125 a \u221225 dB)",
    "T30 fit (\u22125 to \u221235 dB)": "Ajuste T30 (\u22125 a \u221235 dB)",
    "EDT fit (0 to \u221210 dB)": "Ajuste EDT (0 a \u221210 dB)",
    "EDT slope": "Pendiente EDT",
    "ISO 717-1 Weighted Sound Reduction Index (Annex C example)":
        "\u00cdndice ponderado de reducci\u00f3n ac\u00fastica (ISO 717-1, ejemplo del Anexo C)",
    "Apparent sound reduction index R' [dB]":
        "\u00cdndice de reducci\u00f3n ac\u00fastica aparente R' [dB]",
    "Measured R' (third octave)": "R' medido (tercios de octava)",
    "Shifted reference curve (ISO 717-1)":
        "Curva de referencia desplazada (ISO 717-1)",
    "Unfavourable deviations": "Desviaciones desfavorables",
    "Sharpness Weighting g(z) (DIN 45692)":
        "Ponderación de agudeza g(z) (DIN 45692)",
    "Weighting g(z)": "Ponderación g(z)",
    "DIN 45692 g(z)": "g(z) DIN 45692",
    "von Bismarck (Annex B)": "von Bismarck (Anexo B)",
    "DIN knee\n15.8 Bark": "Codo DIN\n15,8 Bark",
    "Bismarck knee\n15 Bark": "Codo Bismarck\n15 Bark",
    "ISO 717-2 Weighted Normalized Impact Sound Level (Annex C example)":
        "Nivel de ruido de impactos normalizado y ponderado "
        "(ISO 717-2, ejemplo del Anexo C)",
    "Normalized impact sound pressure level Ln [dB]":
        "Nivel de presión acústica de impactos normalizado Ln [dB]",
    "Measured Ln (third octave)": "Ln medido (tercios de octava)",
    "Shifted reference curve (ISO 717-2)":
        "Curva de referencia desplazada (ISO 717-2)",
    "Unfavourable deviations (measured above reference)":
        "Desviaciones desfavorables (medido por encima de la referencia)",
    "Open-Plan Spatial Decay of Speech (ISO 3382-3)":
        "Decaimiento espacial del habla en oficina abierta (ISO 3382-3)",
    "Distance from the talker r [m]": "Distancia al hablante r [m]",
    "A-weighted SPL [dB]": "SPL ponderado A [dB]",
    "Measured Lp,A,S": "Lp,A,S medido",
    "STI vs distance": "STI vs distancia",
    # --- ERB_N / Cam auditory-filter scale (Glasberg & Moore 1990) ---
    "Auditory-Filter Bandwidth and the Cam Scale (Glasberg & Moore, 1990)":
        "Ancho de banda del filtro auditivo y escala Cam (Glasberg y Moore, 1990)",
    "Centre frequency [Hz]": "Frecuencia central [Hz]",
    "Equivalent rectangular bandwidth ERB$_N$ [Hz]":
        "Ancho de banda rectangular equivalente ERB$_N$ [Hz]",
    "ERB$_N$ (Glasberg & Moore, 1990)": "ERB$_N$ (Glasberg y Moore, 1990)",
    "One-third octave (23 % of f)": "Tercio de octava (23 % de f)",
    "ERB$_N$ number [Cam]": "Número ERB$_N$ [Cam]",
    "1 kHz = 15.59 Cam": "1 kHz = 15,59 Cam",
    # --- Advanced psychoacoustics figures (plan-17 block A) ---
    "Loudness Models Compared (1 kHz tone)":
        "Modelos de sonoridad comparados (tono de 1 kHz)",
    "Sound pressure level [dB SPL]": "Nivel de presión acústica [dB SPL]",
    "Total loudness N [sone]": "Sonoridad total N [sonios]",
    "Sottek ECMA-418-2": "Sottek ECMA-418-2",
    "Anchor: 1 kHz / 40 dB = 1 sone":
        "Anclaje: 1 kHz / 40 dB = 1 sonio",
    "Models diverge at high levels":
        "Los modelos divergen a niveles altos",
    "Sottek Specific Loudness (ECMA-418-2)":
        "Sonoridad específica de Sottek (ECMA-418-2)",
    "Specific loudness N' [sone_HMS/Bark]":
        "Sonoridad específica N' [sonios_HMS/Bark]",
    "Peak specific loudness": "Sonoridad específica máxima",
    "ECMA-418-2 Tonality T(t)": "Tonalidad T(t) (ECMA-418-2)",
    "Tonality T [tu_HMS]": "Tonalidad T [tu_HMS]",
    "ECMA-418-2 Roughness vs Modulation Frequency":
        "Aspereza vs frecuencia de modulación (ECMA-418-2)",
    "Modulation frequency f_mod [Hz]":
        "Frecuencia de modulación f_mod [Hz]",
    "Roughness R [asper]": "Aspereza R [asper]",
    "1 kHz carrier, 100 % AM": "Portadora de 1 kHz, AM del 100 %",
    "Sound Quality Metrics (ECMA-418-2 Sottek Hearing Model)":
        "Métricas de calidad sonora (modelo auditivo de Sottek, ECMA-418-2)",
    "Slow vs Fast Modulation Perception (ECMA-418-2 Sottek Hearing Model)":
        "Percepción de modulación lenta vs rápida (modelo auditivo de "
        "Sottek, ECMA-418-2)",
    "Fluctuation strength F (Clause 9, slow modulation)":
        "Intensidad de fluctuación F (cláusula 9, modulación lenta)",
    "Roughness R (Clause 7, fast modulation)":
        "Aspereza R (cláusula 7, modulación rápida)",
    "1 kHz carrier, 100 % AM, overall 60 dB SPL":
        "Portadora de 1 kHz, AM del 100 %, 60 dB SPL globales",
    "Time-Varying Loudness (ISO 532-3)":
        "Sonoridad variable en el tiempo (ISO 532-3)",
    "Loudness [sone]": "Sonoridad [sonios]",
    "1 kHz burst, 200 ms": "Ráfaga de 1 kHz, 200 ms",
    "Fast attack / release": "Ataque / relajación rápidos",
    "Slow integration": "Integración lenta",
    # Fluctuation strength + psychoacoustic annoyance (Fastl & Zwicker; Osses 2016)
    "Fluctuation Strength — 4 Hz Band-Pass Characteristic":
        "Intensidad de fluctuación — característica de paso de banda a 4 Hz",
    "Fluctuation strength F [vacil]": "Intensidad de fluctuación F [vacil]",
    "AM-tone F, signal model [vacil]":
        "F de tono AM, modelo de señal [vacil]",
    "4 Hz reference": "referencia 4 Hz",
    "Psychoacoustic Annoyance vs Loudness (Fastl & Zwicker)":
        "Molestia psicoacústica vs sonoridad (Fastl y Zwicker)",
    "Percentile loudness N5 [sone]": "Sonoridad percentil N5 [sonios]",
    "Psychoacoustic annoyance PA": "Molestia psicoacústica PA",
    "Baseline: S = 1.75 acum, F = R = 0":
        "Base: S = 1,75 acum, F = R = 0",
    "Sharp: S = 3.5 acum": "Aguda: S = 3,5 acum",
    "Rough + fluctuating: F = 1.2 vacil, R = 0.7 asper":
        "Áspera + fluctuante: F = 1,2 vacil, R = 0,7 asper",
    # Electroacoustics (IEC 60268-3 distortion; Bendat & Piersol response)
    "Harmonic Distortion of a Single-Tone Test (IEC 60268-3)":
        "Distorsión armónica de un ensayo con tono único (IEC 60268-3)",
    "Magnitude spectrum": "Espectro de magnitud",
    "Harmonics n·f₁": "Armónicos n·f₁",
    "Level re fundamental [dB]": "Nivel respecto al fundamental [dB]",
    "Frequency Response and Coherence (Bendat & Piersol)":
        "Respuesta en frecuencia y coherencia (Bendat y Piersol)",
    "True |H|": "|H| verdadero",
    "Estimated |H| (H1)": "|H| estimado (H1)",
    # Swept-sine harmonic separation (Farina 2000 / Novak et al. 2015)
    "Swept-Sine Harmonic Distortion by Order (Farina / Novak)":
        "Distorsión armónica por orden con barrido sinusoidal "
        "(Farina / Novak)",
    "Excitation frequency [Hz]": "Frecuencia de excitación [Hz]",
    "Distortion re fundamental [%]": "Distorsión respecto al fundamental [%]",
    "Total THD(f)": "THD(f) total",
    "2nd harmonic d₂(f)": "2º armónico d₂(f)",
    "3rd harmonic d₃(f)": "3er armónico d₃(f)",
    "Chebyshev asymptote (a₂/2)/H₁": "Asíntota de Chebyshev (a₂/2)/H₁",
    "Chebyshev asymptote (a₃/4)/H₁": "Asíntota de Chebyshev (a₃/4)/H₁",
    "one sweep separates every distortion order;\n"
    "each rolls off where its product n·f crosses the 3 kHz corner":
        "un solo barrido separa cada orden de distorsión;\n"
        "cada uno cae donde su producto n·f cruza el corte de 3 kHz",
    # Calibrated spectral analysis (Bendat & Piersol PSD/CSD core)
    "Calibrated Spectral Density of Pink Noise (Bendat & Piersol)":
        "Densidad espectral calibrada de ruido rosa (Bendat y Piersol)",
    "95 % chi-square confidence interval":
        "Intervalo de confianza chi-cuadrado del 95 %",
    "Welch PSD estimate": "Estimación de la PSD de Welch",
    "1/3-octave smoothed": "Suavizado en 1/3 de octava",
    "Exact -3.01 dB/octave power law":
        "Ley de potencias exacta de -3,01 dB/octava",
    # Thomson multitaper spectral density (Percival & Walden 1993)
    "Thomson Multitaper Density of a Short Record (Percival & Walden)":
        "Densidad multitaper de Thomson de un registro corto (Percival y Walden)",
    "Single Slepian taper ($K$ = 1, $\\nu$ = 2)":
        "Un solo taper de Slepian ($K$ = 1, $\\nu$ = 2)",
    "Multitaper estimate ($K$ = 7, adaptive)":
        "Estimación multitaper ($K$ = 7, adaptativa)",
    # Time-frequency analysis (Bendat & Piersol spectrogram + zoom FFT)
    "Calibrated Spectrogram in dB SPL (Bendat & Piersol)":
        "Espectrograma calibrado en dB SPL (Bendat y Piersol)",
    "a siren, an impact and a pink-noise floor:\n"
    "every cell reads an absolute level":
        "una sirena, un impacto y un fondo de ruido rosa:\n"
        "cada celda lee un nivel absoluto",
    "Zoom FFT Resolves Tones One Coarse Bin Apart (Bendat & Piersol)":
        "La FFT con zoom resuelve tonos a menos de un bin grueso "
        "(Bendat y Piersol)",
    "1024-point FFT (8 Hz bins)": "FFT de 1024 puntos (bins de 8 Hz)",
    "Zoom FFT of the same record": "FFT con zoom del mismo registro",
    "997 and 1000 Hz, 3 Hz apart:\n"
    "one lump on the 8 Hz grid,\n"
    "two exact lines on the zoom grid":
        "997 y 1000 Hz, separados 3 Hz:\n"
        "un solo bulto en la malla de 8 Hz,\n"
        "dos líneas exactas en la malla del zoom",
    # Cepstral analysis (echo detection) and envelope spectrum
    "Echo Detection on the Power Cepstrum (Quefrency Analysis)":
        "Detección de ecos en el cepstro de potencia (análisis de quefrencia)",
    "Power cepstrum": "Cepstro de potencia",
    "Searched band": "Banda de búsqueda",
    "True echo delay (8 ms)": "Retardo verdadero del eco (8 ms)",
    "Detected peak (height = reflection a)":
        "Pico detectado (altura = reflexión a)",
    "Quefrency [ms]": "Quefrencia [ms]",
    "Cepstrum": "Cepstro",
    "spectral ripple of period 1/(8 ms) collapses to one\n"
    "spike at 8 ms whose height reads the reflection":
        "el rizado espectral de período 1/(8 ms) colapsa en un\n"
        "pico a 8 ms cuya altura mide la reflexión",
    "Envelope Spectrum of an AM Tone (Bendat & Piersol 13.3)":
        "Espectro de la envolvente de un tono AM (Bendat y Piersol 13.3)",
    "Envelope spectrum": "Espectro de la envolvente",
    "Modulation frequency (25 Hz)": "Frecuencia de modulación (25 Hz)",
    r"Exact line amplitude $A_0 m$ = 0.4":
        r"Amplitud exacta de la línea $A_0 m$ = 0,4",
    "Modulation amplitude": "Amplitud de modulación",
    "the carrier is at 1 kHz; its amplitude modulation\n"
    "appears as one line at exactly $f_m$":
        "la portadora está en 1 kHz; su modulación de amplitud\n"
        "aparece como una línea exactamente en $f_m$",
    # Time synchronous averaging (McFadden 1987)
    "Periodic Waveform Extracted from Noise":
        "Forma de onda periódica extraída del ruido",
    "One noisy period": "Un período ruidoso",
    "Average of N = 40 periods": "Promedio de N = 40 períodos",
    "True periodic waveform": "Forma de onda periódica verdadera",
    "averaging N periods lowers the asynchronous\n"
    "noise by $\\sqrt{N}$ in amplitude":
        "promediar N períodos reduce el ruido asíncrono\n"
        "en $\\sqrt{N}$ en amplitud",
    "Rejecting a Tone by Choosing N (McFadden 1987)":
        "Rechazo de un tono eligiendo N (McFadden 1987)",
    "N = 32 (power of two)": "N = 32 (potencia de dos)",
    "N = 20 (node on 32.05)": "N = 20 (nodo en 32,05)",
    "Interfering tone (32.05)": "Tono interferente (32,05)",
    "Frequency [orders]": "Frecuencia [órdenes]",
    "Comb filter magnitude": "Magnitud del filtro peine",
    "N = 20 puts a node on 32.05 orders and removes\n"
    "it; the power-of-two N = 32 lets it through":
        "N = 20 sitúa un nodo en 32,05 órdenes y lo\n"
        "elimina; la potencia de dos N = 32 lo deja pasar",
    # Multiple-input coherence (Bendat & Piersol Ch. 7)
    "Multiple-Input Coherence: Which Source Dominates Each Band "
    "(Bendat & Piersol Ch. 7)":
        "Coherencia de múltiples entradas: qué fuente domina cada banda "
        "(Bendat y Piersol cap. 7)",
    "Measured output": "Salida medida",
    "Input 1 contribution": "Contribución de la entrada 1",
    "Input 2 contribution": "Contribución de la entrada 2",
    "Residual noise": "Ruido residual",
    "Coherent output [dB re 1/Hz]": "Salida coherente [dB re 1/Hz]",
    r"Input 2 ordinary $\gamma^2_{2y}$ (inflated by x1)":
        r"Entrada 2 ordinaria $\gamma^2_{2y}$ (inflada por x1)",
    r"Input 2 partial $\gamma^2_{2y\cdot 1}$ (x1 removed)":
        r"Entrada 2 parcial $\gamma^2_{2y\cdot 1}$ (x1 eliminada)",
    r"Multiple $\gamma^2_{y:x}$": r"Múltiple $\gamma^2_{y:x}$",
    "Coherence": "Coherencia",
    "conditioning removes the shared x1 component:\n"
    "the low-band ordinary coherence of x2 collapses":
        "el condicionamiento elimina la componente x1 compartida:\n"
        "la coherencia ordinaria de x2 en la banda baja se desploma",
    # Data qualification: trend and stationarity tests, Rice crossing statistics
    "Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)":
        "Test de tendencia no paramétrico por inversiones de orden (B&P 4.5.2)",
    "B&P Example 4.4: A = 86, accepted (no trend)":
        "Ejemplo 4.4 de B&P: A = 86, aceptado (sin tendencia)",
    "Added rising drift: A = 38, rejected (trend)":
        "Deriva ascendente añadida: A = 38, rechazado (tendencia)",
    "Sample index": "Índice de muestra",
    "Sequence value": "Valor de la secuencia",
    "20 observations; the count A of pairs i < j with x[i] > x[j]\n"
    "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
    "trend depresses A below the acceptance region":
        "20 observaciones; el conteo A de pares i < j con x[i] > x[j]\n"
        "debe caer en (64, 125] al nivel del 5 % (Tabla A.6). Una deriva\n"
        "ascendente reduce A por debajo de la región de aceptación",
    "Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)":
        "Test de estacionariedad por inversiones de orden (B&P 10.3.1.1)",
    "Steady noise: A = 91, accepted (stationary)":
        "Ruido estable: A = 91, aceptado (estacionario)",
    "+20 % gain ramp: A = 7, rejected (nonstationary)":
        "Rampa de ganancia del +20 %: A = 7, rechazado (no estacionario)",
    "Segment index": "Índice de segmento",
    "Segment mean square": "Media cuadrática por segmento",
    "20 segment mean squares; the count A of pairs i < j with\n"
    "x[i] > x[j] must fall in (64, 125] at the 5 % level (Table A.6)":
        "20 medias cuadráticas por segmento; el conteo A de pares i < j con\n"
        "x[i] > x[j] debe caer en (64, 125] al nivel del 5 % (Tabla A.6)",
    "Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)":
        "Tasas de cruce por nivel de ruido gaussiano de banda limitada "
        "(Rice)",
    r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Eq. 5.196)":
        r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Ec. 5.196)",
    "Measured crossing rate": "Tasa de cruces medida",
    "Level a [signal units]": "Nivel a [unidades de la señal]",
    "Crossings per second [1/s]": "Cruces por segundo [1/s]",
    "800-1200 Hz Gaussian band: 2014 zero crossings/s, an\n"
    r"apparent frequency $N_0/2 \approx$ 1007 Hz (B&P Example 5.13)":
        "banda gaussiana de 800-1200 Hz: 2014 cruces por cero/s, una\n"
        r"frecuencia aparente $N_0/2 \approx$ 1007 Hz (B&P Ejemplo 5.13)",
    "Peak-Height Distribution and the Irregularity Factor (Rice)":
        "Distribución de alturas de pico y factor de irregularidad (Rice)",
    "Rayleigh limit (r = 1, narrowband)":
        "Límite de Rayleigh (r = 1, banda estrecha)",
    "Gaussian limit (r = 0, wideband)":
        "Límite gaussiano (r = 0, banda ancha)",
    "Rice mixture at r = 0.746 (Eq. 5.223)":
        "Mezcla de Rice con r = 0,746 (Ec. 5.223)",
    "Empirical peak exceedance (0-2 kHz noise)":
        "Excedencia empírica de picos (ruido de 0-2 kHz)",
    r"Standardized peak height $z = a/\sigma_x$":
        r"Altura de pico estandarizada $z = a/\sigma_x$",
    "Prob[peak > z]": "Prob[pico > z]",
    r"low-pass noise: $r = N_0/2M = \sqrt{5}/3$; negative maxima exist,"
    "\nso the peak law sits between Gaussian and Rayleigh (B&P 5.5.4)":
        r"ruido paso bajo: $r = N_0/2M = \sqrt{5}/3$; existen máximos"
        " negativos,\nasí que la ley de picos queda entre la gaussiana y la"
        " de Rayleigh (B&P 5.5.4)",
    # Correlation and time-delay estimation (B&P / Knapp & Carter GCC)
    "Time-Delay Estimation: GCC-PHAT vs Direct Correlation (Knapp & Carter)":
        "Estimación del retardo: GCC-PHAT frente a correlación directa "
        "(Knapp y Carter)",
    "Direct cross-correlation": "Correlación cruzada directa",
    "True delay (20 samples)": "Retardo verdadero (20 muestras)",
    "Lag [ms]": "Retardo [ms]",
    "Normalized correlation": "Correlación normalizada",
    "colored signal: the plain correlator smears the peak,\n"
    "PHAT prewhitens the cross-spectrum and restores it":
        "señal coloreada: el correlador simple ensancha el pico,\n"
        "PHAT preblanquea el espectro cruzado y lo recupera",
    # Objective intelligibility: STOI vs ESTOI (Taal 2011 / Jensen & Taal 2016)
    "Short-Time Objective Intelligibility: STOI vs ESTOI":
        "Inteligibilidad objetiva de corto plazo: STOI frente a ESTOI",
    "STOI (Taal et al. 2011)": "STOI (Taal et al. 2011)",
    "ESTOI (Jensen & Taal 2016)": "ESTOI (Jensen y Taal 2016)",
    "Stationary masker": "Enmascarador estacionario",
    "Modulated (5 Hz gated) masker":
        "Enmascarador modulado (con puerta a 5 Hz)",
    "SNR [dB]": "SNR [dB]",
    "Intelligibility index": "Índice de inteligibilidad",
    "ESTOI rates the modulated masker higher: it credits the\n"
    "speech glimpsed in the quiet gaps. STOI barely separates them.":
        "ESTOI valora más alto el enmascarador modulado: acredita el\n"
        "habla vislumbrada en los silencios. STOI apenas los separa.",
    # Room acoustics: image-source reflectogram (Kuttruff 4.1 / Vorlander 11.4)
    "Image-Source Room Impulse Response: a 7x5x3 m room (order <= 10)":
        "Respuesta al impulso por fuentes imagen: sala de 7x5x3 m "
        "(orden <= 10)",
    "Reflection order": "Orden de reflexión",
    "Arrival time [ms]": "Tiempo de llegada [ms]",
    "Reflection level re direct [dB]": "Nivel de reflexión rel. directo [dB]",
    r"$1/r$ spreading envelope": r"Envolvente de divergencia $1/r$",
    "Reflections (image sources)": "Reflexiones (fuentes imagen)",
    "Direct sound (order 0)": "Sonido directo (orden 0)",
    "each reflection is a mirror image of the source;\n"
    "amplitude = product of wall reflection factors / (4 pi r)":
        "cada reflexión es una imagen especular de la fuente;\n"
        "amplitud = producto de factores de reflexión de pared / (4 pi r)",
    # Underwater acoustics (ISO 17208 ship radiated noise; ISO 18406 pile driving)
    "Ship Equivalent Monopole Source Level (ISO 17208-2)":
        "Nivel de fuente monopolar equivalente de buque (ISO 17208-2)",
    "Source level Ls": "Nivel de fuente Ls",
    "Radiated noise level": "Nivel de ruido radiado",
    "Surface correction ΔL [dB]": "Corrección de superficie ΔL [dB]",
    "Surface correction ΔL": "Corrección de superficie ΔL",
    "Level [dB re 1 µPa·m]": "Nivel [dB re 1 µPa·m]",
    "Percussive Pile-Driving Strike (ISO 18406)":
        "Golpe de hincado de pilotes por percusión (ISO 18406)",
    "Time [ms]": "Tiempo [ms]",
    "Pressure [Pa]": "Presión [Pa]",
    "FDTD probe pressure": "Presión en las sondas FDTD",
    "Number of strikes N": "Número de golpes N",
    "Cumulative SEL [dB re 1 µPa²·s]": "SEL acumulado [dB re 1 µPa²·s]",
    "ICAO Aircraft Flyover — Effective Perceived Noise Level (Annex 16)":
        "Sobrevuelo de aeronave ICAO — Nivel efectivo de ruido percibido (Anexo 16)",
    "Level [PNdB]": "Nivel [PNdB]",
    "10 dB-down window": "Ventana 10 dB por debajo",
    "Wind-Turbine Tonal Audibility (IEC 61400-11)":
        "Audibilidad tonal de aerogenerador (IEC 61400-11)",
    "Narrowband spectrum": "Espectro de banda estrecha",
    "Critical band": "Banda crítica",
    # Underwater propagation (plan-22 P1): transmission loss, sound speed, sonar.
    "Underwater Transmission Loss (Francois–Garrison)":
        "Pérdida por transmisión submarina (Francois–Garrison)",
    "Range [m]": "Distancia [m]",
    "Transmission loss [dB]": "Pérdida por transmisión [dB]",
    "Total transmission loss": "Pérdida por transmisión total",
    "Geometrical spreading": "Ensanchamiento geométrico",
    "Volume absorption": "Absorción de volumen",
    "Sea-Water Sound-Speed Profile (UNESCO)":
        "Perfil de velocidad del sonido en agua de mar (UNESCO)",
    "Sound speed [m/s]": "Velocidad del sonido [m/s]",
    "Depth [m]": "Profundidad [m]",
    "UNESCO sound speed": "Velocidad del sonido UNESCO",
    "Sound-channel axis": "Eje del canal sonoro",
    # Weston shallow-water propagation regimes (Ainslie section 9.1.1.2).
    "Weston Shallow-Water Propagation Regimes (Ainslie §9.1.1.2)":
        "Regímenes de propagación de Weston en aguas someras (Ainslie §9.1.1.2)",
    "Propagation loss [dB re 1 m²]": "Pérdida de propagación [dB re 1 m²]",
    "Spherical, 20 lg r": "Esférica, 20 lg r",
    "Cylindrical, 10 lg r": "Cilíndrica, 10 lg r",
    "Mode stripping, 15 lg r": "Descamado de modos, 15 lg r",
    "Single mode": "Modo único",
    "Composite propagation loss": "Pérdida de propagación compuesta",
    # Marine-mammal auditory weighting (NMFS 2024 v3.0).
    "Marine-Mammal Auditory Weighting (NMFS 2024, v3.0)":
        "Ponderación auditiva de mamíferos marinos (NMFS 2024, v3.0)",
    "Weighting amplitude W(f) [dB]": "Amplitud de ponderación W(f) [dB]",
    "Passive Sonar Equation": "Ecuación del sonar pasivo",
    "Signal excess [dB]": "Exceso de señal [dB]",
    "Signal excess": "Exceso de señal",
    "Detection limit (SE = 0)": "Límite de detección (SE = 0)",
    "Figure of merit": "Figura de mérito",
    # Underwater propagation (plan-22 P1 PR-2): seabed, ambient noise, traffic.
    "Seabed Reflection Loss (Rayleigh)":
        "Pérdida por reflexión en el fondo (Rayleigh)",
    "Grazing angle [°]": "Ángulo rasante [°]",
    "Bottom loss [dB]": "Pérdida por reflexión [dB]",
    "Bottom loss (sand)": "Pérdida por reflexión (arena)",
    "Seabed Reflection Coefficient (Rayleigh)":
        "Coeficiente de reflexión del fondo (Rayleigh)",
    "Reflection coefficient magnitude |R|":
        "Módulo del coeficiente de reflexión |R|",
    "Reflection coefficient magnitude |R| (sand)":
        "Módulo del coeficiente de reflexión |R| (arena)",
    "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650":
        "Agua ρ = 1000, c = 1500\nArena ρ = 1900, c = 1650",
    "Ocean Ambient Noise (Wenz)": "Ruido ambiental oceánico (Wenz)",
    "Spectrum level [dB re 1 µPa²/Hz]": "Nivel espectral [dB re 1 µPa²/Hz]",
    "Ship Traffic Source Level (JOMOPANS-ECHO)":
        "Nivel de fuente del tráfico marítimo (JOMOPANS-ECHO)",
    "Ray trace (Munk profile)": "Trazado de rayos (perfil de Munk)",
    "Source": "Fuente",
    "Range [km]": "Distancia [km]",
    "Parabolic equation (50 Hz)": "Ecuación parabólica (50 Hz)",
    "Modes vs PE (50 Hz, z = 120 m)": "Modos vs PE (50 Hz, z = 120 m)",
    "Normal modes": "Modos normales",
    "Parabolic equation": "Ecuación parabólica",
    "Source spectral density [dB re 1 µPa²/Hz at 1 m]":
        "Densidad espectral de fuente [dB re 1 µPa²/Hz a 1 m]",
    "Wind": "Viento",
    "Thermal": "Térmico",
    # Building acoustics (EN 12354-1 flanking prediction, ISO 12999-1 uncertainty)
    "EN 12354-1 Flanking Transmission (Annex H.3 example)":
        "Transmisión por flancos EN 12354-1 (ejemplo del Anexo H.3)",
    "Share of transmitted energy [%]": "Cuota de energía transmitida [%]",
    "ISO 12354-1 Detailed Model: Dominant Path per Band (Annex L)":
        "Modelo detallado ISO 12354-1: camino dominante por banda (Anexo L)",
    "other paths": "otros caminos",
    "R' (apparent)": "R' (aparente)",
    "Transmission path": "Camino de transmisión",
    "Dd — direct": "Dd — directo",
    "Ff — flanking–flanking": "Ff — flanco–flanco",
    "Fd — flanking–separating": "Fd — flanco–separador",
    "Df — separating–flanking": "Df — separador–flanco",
    "dominant path": "camino dominante",
    "ISO 12999-1 Measurement Uncertainty (situation B, airborne)":
        "Incertidumbre de medición ISO 12999-1 (situación B, aéreo)",
    "Measured R'": "R' medido",
    "Standard uncertainty ±u": "Incertidumbre típica ±u",
    "Expanded uncertainty ±U (95 %)": "Incertidumbre expandida ±U (95 %)",
    "R'w ± U (single number)": "R'w ± U (valor único)",
    # Outdoor propagation & occupational exposure (PR-C).
    "ISO 9613-1 Atmospheric Absorption α(f)":
        "Absorción atmosférica α(f) (ISO 9613-1)",
    "Attenuation coefficient α [dB/km]":
        "Coeficiente de atenuación α [dB/km]",
    "ISO 9613-1 atmospheric attenuation":
        "Atenuación atmosférica ISO 9613-1",
    r"Attenuation coefficient $\alpha$ [dB/km]":
        r"Coeficiente de atenuación $\alpha$ [dB/km]",
    "ISO 9613-2 Attenuation Breakdown (with a 4 m barrier)":
        "Desglose de la atenuación (ISO 9613-2, con barrera de 4 m)",
    "Octave-band centre frequency [Hz]":
        "Frecuencia central de banda de octava [Hz]",
    "1/3-octave band centre frequency [Hz]":
        "Frecuencia central de banda de tercio de octava [Hz]",
    "CNOSSOS-EU Railway Source Line Power (96 coaches/h at 160 km/h)":
        "Potencia de la línea fuente ferroviaria CNOSSOS-EU "
        "(96 coches/h a 160 km/h)",
    "Line power L'W,eq,line [dB re 1 pW/m]":
        "Potencia de la línea L'W,eq,line [dB re 1 pW/m]",
    "Both source heights": "Ambas alturas de fuente",
    "Source A, 0,5 m (rolling, impact, traction)":
        "Fuente A, 0,5 m (rodadura, impacto, tracción)",
    "Source B, 4,0 m (traction, aerodynamic)":
        "Fuente B, 4,0 m (tracción, aerodinámico)",
    "CNOSSOS-EU Total Effective Roughness against Speed "
    "(f = v/λ, λ in the wavelength domain)":
        "Rugosidad efectiva total CNOSSOS-EU frente a la velocidad "
        "(f = v/λ, λ en el dominio de longitud de onda)",
    "Total effective roughness LR,TOT [dB re 1 μm]":
        "Rugosidad efectiva total LR,TOT [dB re 1 μm]",
    "Attenuation A [dB]": "Atenuación A [dB]",
    "CNOSSOS-EU Road Source Line Power (urban arterial, 50 km/h)":
        "Potencia de la línea fuente viaria CNOSSOS-EU (vía urbana, 50 km/h)",
    "Line power L'W,eq,line [dB re 1 pW/m]":
        "Potencia de línea L'W,eq,line [dB re 1 pW/m]",
    "Total source line": "Línea fuente total",
    "Light vehicles (1)": "Vehículos ligeros (1)",
    "Medium heavy vehicles (2)": "Vehículos pesados medios (2)",
    "Heavy vehicles (3)": "Vehículos pesados (3)",
    "Motorcycles (4b)": "Motocicletas (4b)",
    "Light vehicles (1) — total": "Vehículos ligeros (1) — total",
    "Light vehicles (1) — rolling": "Vehículos ligeros (1) — rodadura",
    "Light vehicles (1) — propulsion": "Vehículos ligeros (1) — propulsión",
    "Heavy vehicles (3) — total": "Vehículos pesados (3) — total",
    "Heavy vehicles (3) — rolling": "Vehículos pesados (3) — rodadura",
    "Heavy vehicles (3) — propulsion": "Vehículos pesados (3) — propulsión",
    "CNOSSOS-EU Single-Vehicle Sound Power against Speed (reference conditions)":
        "Potencia acústica de un vehículo frente a la velocidad "
        "(CNOSSOS-EU, condiciones de referencia)",
    "Speed v [km/h]": "Velocidad v [km/h]",
    "A-weighted sound power LW,A [dB(A) re 1 pW]":
        "Potencia acústica ponderada A LW,A [dB(A) re 1 pW]",
    "Adiv — divergence": "Adiv — divergencia",
    "Aatm — atmospheric": "Aatm — atmosférica",
    "Agr — ground": "Agr — suelo",
    "Abar — barrier": "Abar — barrera",
    "A — total": "A — total",
    "Spherical-Wave Ground Effect (Weyl-Van der Pol)":
        "Efecto suelo de onda esférica (Weyl-Van der Pol)",
    "Level re free field [dB]": "Nivel respecto al campo libre [dB]",
    "Fresh snow (10 kPa·s·m⁻²)": "Nieve reciente (10 kPa·s·m⁻²)",
    "Forest floor (50 kPa·s·m⁻²)": "Suelo forestal (50 kPa·s·m⁻²)",
    "Grassland (200 kPa·s·m⁻²)": "Pradera (200 kPa·s·m⁻²)",
    "Asphalt (20 000 kPa·s·m⁻²)": "Asfalto (20 000 kPa·s·m⁻²)",
    "Hard-ground limit (+6 dB)": "Límite de suelo rígido (+6 dB)",
    "ISO 9612 Task-Based Exposure (Annex D)":
        "Exposición por tareas (ISO 9612, Anexo D)",
    "LEX,8h contribution [dB]": "Contribución a LEX,8h [dB]",
    "Measurement task": "Tarea de medición",
    "planning/breaks": "planificación/pausas",
    "welding": "soldadura",
    "cutting/grinding": "corte/amolado",
    "Daily LEX,8h": "LEX,8h diario",
    "LEX,8h + U (one-sided 95 %)": "LEX,8h + U (unilateral 95 %)",
    # Materials: absorption rating, airflow resistance, impedance tube
    "Shifted reference curve (ISO 11654)":
        "Curva de referencia desplazada (ISO 11654)",
    "Practical absorption alpha_p": "Absorción práctica alpha_p",
    "ISO 11654 Weighted Sound Absorption Coefficient (Annex A.2 example)":
        "Coeficiente de absorción acústica ponderado ISO 11654 (ejemplo del Anexo A.2)",
    "Sound absorption coefficient": "Coeficiente de absorción acústica",
    "Through-origin quadratic fit  dp = a u + b u^2":
        "Ajuste cuadrático por el origen  dp = a u + b u^2",
    "Measured pressure drop": "Caída de presión medida",
    "evaluation at 0.5 mm/s": "evaluación a 0,5 mm/s",
    "ISO 9053-1 Static-Method Airflow Resistance":
        "Resistencia al flujo de aire por el método estático (ISO 9053-1)",
    "Linear airflow velocity u [mm/s]": "Velocidad lineal del aire u [mm/s]",
    "Pressure drop dp [Pa]": "Caída de presión dp [Pa]",
    "Absorption coefficient alpha = 1 - |r|^2":
        "Coeficiente de absorción alpha = 1 - |r|^2",
    "Standing-wave level difference L_max - L_min [dB]":
        "Diferencia de nivel de onda estacionaria L_max - L_min [dB]",
    "Sound absorption coefficient alpha": "Coeficiente de absorción acústica alpha",
    "Perfect Absorption by Critical Coupling (Slow-Sound Panel)":
        "Absorción perfecta por acoplo crítico (panel de sonido lento)",
    "Critically coupled (perfect)": "Acoplo crítico (perfecto)",
    "Narrow slit (over-damped)": "Ranura estrecha (sobreamortiguada)",
    "Wide slit (under-damped)": "Ranura ancha (subamortiguada)",
    "design 300 Hz": "diseño 300 Hz",
    "Reflection factor magnitude |r|": "Módulo del factor de reflexión |r|",
    "ISO 10534-1 Standing-Wave-Ratio Method":
        "Método de la razón de onda estacionaria (ISO 10534-1)",
    # --- Tier-1 animation labels ---
    "tone burst": "ráfaga de tono",
    "Fast (125 ms)": "Rápida (125 ms)",
    "Slow (1000 ms)": "Lenta (1000 ms)",
    "Impulse (35 ms / 1.5 s)": "Impulso (35 ms / 1,5 s)",
    "Time-weighting ballistics (IEC 61672-1)":
        "Balística de la ponderación temporal (IEC 61672-1)",
    "Mean-square response (normalized)":
        "Respuesta cuadrática media (normalizada)",
    "RC exponential detector": "Detector exponencial RC",
    "input x(t)": "entrada x(t)",
    "square-law rectifier": "rectificador cuadrático",
    "stored charge (Fast shown)": "carga almacenada (se muestra Fast)",
    "charging": "cargando",
    "draining": "descargando",
    "τ = RC sets attack and decay": "τ = RC fija el ataque y la caída",
    "onset (> 10 dB/s)": "inicio (> 10 dB/s)",
    "Impulse onset detection (NT ACOU 112)":
        "Detección del inicio de impulso (NT ACOU 112)",
    "A-weighted level L_AF [dB]": "Nivel ponderado A L_AF [dB]",
    "L_AF (A-weighted, Fast)": "L_AF (ponderado A, Fast)",
    "detector: onset when dL/dt > 10 dB/s":
        "detector: inicio cuando dL/dt > 10 dB/s",
    "onset rate": "tasa de inicio",
    "level difference": "diferencia de nivel",
    "prominence": "prominencia",
    "adjustment": "ajuste",
    "add KI to the rating level": "sumar KI al nivel de evaluación",
    "pressure p": "presión p",
    "velocity u": "velocidad u",
    "intensity p·u": "intensidad p·u",
    "amplitude (normalized)": "amplitud (normalizada)",
    "Two-microphone p-p probe: instantaneous intensity p·u":
        "Sonda p-p de dos micrófonos: intensidad instantánea p·u",
    "Progressive wave — active": "Onda progresiva — activa",
    "Standing wave — reactive": "Onda estacionaria — reactiva",
    "p and u in phase": "p y u en fase",
    "p and u 90° apart": "p y u desfasados 90°",
    "spacer Δr": "separador Δr",
    "T20 fit": "ajuste T20",
    "T30 fit": "ajuste T30",
    "Schroeder backward integration (ISO 3382)":
        "Integración inversa de Schroeder (ISO 3382)",
    "← integrate from the tail": "← integrar desde la cola",
    "squared impulse response p²": "respuesta al impulso al cuadrado p²",
    "tail energy": "energía de la cola",
    "Room modes in a rigid 5 m × 3.5 m room (2D FDTD)":
        "Modos propios en una sala rígida de 5 m × 3,5 m (FDTD 2D)",
    "instantaneous p(x, y)": "p(x, y) instantánea",
    "RMS pressure (mode map)": "presión RMS (mapa modal)",
    "nodal lines (2,1)": "líneas nodales (2,1)",
    "source": "fuente",
    "same color scale": "misma escala de color",
    # --- Tier-1 animation labels (second batch) ---
    "Standing wave in the impedance tube (ISO 10534-2)":
        "Onda estacionaria en el tubo de impedancia (ISO 10534-2)",
    "Rigid termination": "Terminación rígida",
    "Porous sample": "Muestra porosa",
    "sample": "muestra",
    "rigid wall": "pared rígida",
    "incident": "incidente",
    "reflected": "reflejada",
    "sum p(x, t)": "suma p(x, t)",
    "envelope |p(x)|": "envolvente |p(x)|",
    "deep nodes": "nodos profundos",
    "shallow nodes": "nodos poco profundos",
    "Flanking transmission paths (EN 12354-1)":
        "Caminos de transmisión por flancos (EN 12354-1)",
    "source room": "recinto emisor",
    "receiving room": "recinto receptor",
    "direct, wall to wall": "directo, de muro a muro",
    "floor to floor": "de suelo a suelo",
    "floor to wall": "de suelo a muro",
    "wall to floor": "de muro a suelo",
    "junction: Kij attenuates each transfer":
        "unión: Kij atenúa cada transferencia",
    "R'w sums all paths — always below the wall alone":
        "R'w suma todos los caminos — siempre menor que el muro solo",
    "Intensity scanning over a box surface (ISO 9614-2)":
        "Barrido de intensidad sobre una superficie en caja (ISO 9614-2)",
    "p-p probe": "sonda p-p",
    "normal intensity I·n on the surface":
        "intensidad normal I·n en la superficie",
    "partial powers": "potencias parciales",
    "top": "superior",
    "front": "frontal",
    "back": "trasera",
    "left": "izquierda",
    "right": "derecha",
    "any enclosing surface gives the same P":
        "cualquier superficie envolvente da la misma P",
    "Sweep measurement and deconvolution (ISO 18233)":
        "Medición con barrido y deconvolución (ISO 18233)",
    "mic": "micro",
    "direct + reflections": "directo + reflexiones",
    "Frequency [kHz]": "Frecuencia [kHz]",
    "recorded sweep (spectrogram)": "barrido grabado (espectrograma)",
    "delayed copies = reflections": "copias retardadas = reflexiones",
    "impulse response": "respuesta al impulso",
    "direct": "directo",
    "⊛ inverse filter": "⊛ filtro inverso",
    "same information, different domain: sweep ⊛ inverse filter = impulse"
    " response":
        "la misma información en otro dominio: barrido ⊛ filtro inverso ="
        " respuesta al impulso",
    "Specific loudness N'(z) and its integral (ISO 532-1)":
        "Sonoridad específica N'(z) y su integral (ISO 532-1)",
    "1 kHz ≈ 8.5 Bark": "1 kHz ≈ 8,5 Bark",
    "upward spread of masking":
        "extensión del enmascaramiento hacia agudos",
    "1 kHz narrowband": "banda estrecha de 1 kHz",
    "One source, two rooms, one sound power":
        "Una fuente, dos salas, una única potencia acústica",
    "Anechoic room (ISO 3745)": "Cámara anecoica (ISO 3745)",
    "Reverberation room (ISO 3741)": "Cámara reverberante (ISO 3741)",
    "microphone sphere, r": "esfera de micrófonos, r",
    "direct sound only — no reflections":
        "solo sonido directo — sin reflexiones",
    "rotating microphone": "micrófono giratorio",
    "reflections build a diffuse field":
        "las reflexiones crean un campo difuso",
    "the room changes Lp, not the source power":
        "la sala cambia Lp, no la potencia de la fuente",
    "Comb filtering from a single reflection":
        "Filtrado en peine por una sola reflexión",
    "reflecting floor": "suelo reflectante",
    "image source": "fuente imagen",
    "arrival time [ms]": "tiempo de llegada [ms]",
    "amplitude": "amplitud",
    "delayed copy": "copia retardada",
    "response [dB]": "respuesta [dB]",
    "high mic: dense comb": "micro alto: peine denso",
    "lower: notches move up": "más bajo: los nulos suben",
    "on the floor: copies merge — no comb in band":
        "en el suelo: las copias se funden — sin peine en banda",
    "first notch above 8 kHz": "primer nulo por encima de 8 kHz",
    # --- FDTD animation labels (third batch) ---
    "Barrier diffraction into the shadow zone (2D FDTD)":
        "Difracción en una barrera hacia la zona de sombra (FDTD 2D)",
    "barrier": "barrera",
    "shadow zone": "zona de sombra",
    "rigid ground": "suelo rígido",
    "RMS level [dB re panel max]": "nivel RMS [dB re máx del panel]",
    "diffraction fills the shadow": "la difracción rellena la sombra",
    "deep, clean shadow": "sombra profunda y limpia",
    "each panel on its own dB scale": "cada panel en su propia escala dB",
    "Ground effect: direct + reflected interference (2D FDTD)":
        "Efecto de suelo: interferencia directa + reflejada (FDTD 2D)",
    "source (h = 1.5 m)": "fuente (h = 1,5 m)",
    "image source (ghost)": "fuente imagen (fantasma)",
    "receiver in a dip": "receptor en un mínimo",
    "instantaneous pressure": "presión instantánea",
    "RMS level: interference lobes": "nivel RMS: lóbulos de interferencia",
    "Level on the 8 m arc": "Nivel en el arco de 8 m",
    "elevation angle θ [°]": "ángulo de elevación θ [°]",
    "level [dB re max]": "nivel [dB re máx]",
    "image-source model": "modelo de fuente imagen",
    "predicted nulls": "mínimos previstos",
    "dips land exactly on the predicted nulls":
        "los mínimos caen exactamente en los nulos previstos",
    "FDTD": "FDTD",
    "SOFAR channel: sound trapped by the c(z) minimum (2D FDTD)":
        "Canal SOFAR: sonido atrapado por el mínimo de c(z) (FDTD 2D)",
    "Source on the channel axis (depth 400 m)":
        "Fuente en el eje del canal (400 m de profundidad)",
    "Source near the surface (depth 150 m)":
        "Fuente cerca de la superficie (150 m de profundidad)",
    "channel axis (c minimum)": "eje del canal (mínimo de c)",
    "trapped: wavefronts bend back to the axis":
        "atrapado: los frentes de onda se curvan de vuelta al eje",
    "leaks: energy escapes the channel":
        "fuga: la energía escapa del canal",
    "c(z) [m/s]": "c(z) [m/s]",
    "Flat panel vs Schroeder diffuser (2D FDTD)":
        "Panel plano frente a difusor de Schroeder (FDTD 2D)",
    "Flat rigid panel": "Panel rígido plano",
    "Schroeder diffuser (QRD, N = 7)": "Difusor de Schroeder (QRD, N = 7)",
    "incident plane wavefront": "frente de onda plano incidente",
    "sound field p": "campo sonoro p",
    "scattered field (total − incident)":
        "campo dispersado (total − incidente)",
    "specular beam": "haz especular",
    "scattered fan": "abanico dispersado",
    "receiver arc": "arco de receptores",
    "Programme Loudness Metering (EBU R 128)":
        "Medición de sonoridad de programa (EBU R 128)",
    "Loudness [LUFS]": "Sonoridad [LUFS]",
    "Momentary M (400 ms)": "Momentánea M (400 ms)",
    "Short-term S (3 s)": "Corto plazo S (3 s)",
    "ambience": "ambiente",
    "dialogue": "diálogo",
    "music": "música",
    "fade-out": "fundido",
    # Rooms & materials result figures (WP: rooms & materials)
    r"Normal incidence $\alpha(0°)$": r"Incidencia normal $\alpha(0°)$",
    "Probe pressure spectrum": "Espectro de presión en la sonda",
    "Analytic mode frequencies": "Frecuencias modales analíticas",
    "Rigid-box FDTD probe spectrum vs analytic modes":
        "Espectro FDTD en la sonda de una caja rígida frente a los modos analíticos",
    "Probe spectrum [dB re max]": "Espectro en la sonda [dB re máx]",
    # Building & structure-borne result figures (guide figure coverage).
    "ISO 717-1 Enlarged-Range Rating (Annex B)":
        "Índice ponderado con rango ampliado ISO 717-1 (Anexo B)",
    "enlarged range (Annex B)": "rango ampliado (Anexo B)",
    "measured R": "R medido",
    "shifted reference (100-3150 Hz)": "referencia desplazada (100-3150 Hz)",
    "shifted reference": "referencia desplazada",
    "unfavourable deviations": "desviaciones desfavorables",
    "ISO 16283-1 Field Airborne Insulation":
        "Aislamiento a ruido aéreo in situ ISO 16283-1",
    "ISO 16283-3 Field Facade Insulation":
        "Aislamiento de fachada in situ ISO 16283-3",
    "D2m,nT (standardized)": "D2m,nT (estandarizada)",
    "D2m,n (normalized)": "D2m,n (normalizada)",
    "R'45° (element)": "R'45° (elemento)",
    "Level difference / reduction index [dB]":
        "Diferencia de nivel / índice de reducción [dB]",
    "ISO 10052 Survey Method: Impact Sound":
        "Método de control ISO 10052: ruido de impactos",
    "Li (impact level)": "Li (nivel de impactos)",
    "L'nT (standardized)": "L'nT (estandarizado)",
    "Impact sound pressure level [dB]":
        "Nivel de presión acústica de impactos [dB]",
    "ISO 10140 Laboratory Insulation (flanking suppressed)":
        "Aislamiento en laboratorio ISO 10140 (flancos suprimidos)",
    "normalized Ln": "Ln normalizado",
    "Impact sound pressure level Ln [dB]":
        "Nivel de presión acústica de impactos Ln [dB]",
    "ISO 15186-1 Small-Element Insulation by Intensity":
        "Aislamiento de elementos pequeños por intensidad ISO 15186-1",
    "DI,n,e (element)": "DI,n,e (elemento)",
    "Element normalized level difference [dB]":
        "Diferencia de niveles normalizada de elemento [dB]",
    "ISO 10848 Airborne Flanking Transmission":
        "Transmisión aérea por flancos ISO 10848",
    "Dn,f (flanking)": "Dn,f (flancos)",
    "Normalized flanking level difference [dB]":
        "Diferencia de niveles normalizada de flancos [dB]",
    "EN 12354-2 Impact Sound Prediction (Annex E.3)":
        "Predicción de ruido de impactos EN 12354-2 (Anexo E.3)",
    "Level / correction [dB]": "Nivel / corrección [dB]",
    "EN 12354-4 Radiated Sound Power (Annex G)":
        "Potencia acústica radiada EN 12354-4 (Anexo G)",
    "radiated $L_W$ per octave": "$L_W$ radiada por octava",
    "Radiated sound power level [dB re 1 pW]":
        "Nivel de potencia acústica radiada [dB re 1 pW]",
    "Predicted Single-Panel Insulation Rated per ISO 717-1":
        "Aislamiento previsto de panel simple evaluado según ISO 717-1",
    "predicted R (Sharp)": "R previsto (Sharp)",
    "Wave-Approach Junction $K_{ij}$ (Hopkins Eq. 5.116)":
        "$K_{ij}$ de unión por el enfoque ondulatorio (Hopkins Ec. 5.116)",
    "X corner": "X esquina",
    "X straight": "X recta",
    "T-junction (1) corner": "unión en T (1) esquina",
    "L corner": "L esquina",
    "identical plates (τ = 1/12)": "placas idénticas (τ = 1/12)",
    "Thickness ratio h2/h1": "Relación de espesores h2/h1",
    "Vibration reduction index $K_{ij}$ [dB]":
        "Índice de reducción de vibraciones $K_{ij}$ [dB]",
    "Reading a Driving-Point Mobility (ISO 7626-1)":
        "Lectura de una movilidad en el punto de excitación (ISO 7626-1)",
    "driving-point $|Y(f)|$": "$|Y(f)|$ en el punto de excitación",
    r"stiffness line $\omega/k$": r"línea de rigidez $\omega/k$",
    r"mass line $1/(\omega m)$": r"línea de masa $1/(\omega m)$",
    "peak $|Y| = 1/c$ (damping)": "pico $|Y| = 1/c$ (amortiguamiento)",
    "Mobility $|Y|$ [m/(N·s)]": "Movilidad $|Y|$ [m/(N·s)]",
    # Atmospheric refraction (effective profiles, ray fan, GFPE range cut) and
    # wave-theoretic barrier insertion loss (ground-and-barriers guide).
    "Effective Sound-Speed Profiles (Salomons Eq. 4.5)":
        "Perfiles de velocidad efectiva del sonido (Salomons ec. 4.5)",
    "Downward refraction (b = +1 m/s)": "Refracción hacia abajo (b = +1 m/s)",
    "Upward refraction (b = -1 m/s)": "Refracción hacia arriba (b = -1 m/s)",
    "Sound Rays under Downward Refraction (b = +1 m/s)":
        "Rayos sonoros con refracción hacia abajo (b = +1 m/s)",
    "shallow rays are bent back to the ground\nand bounce on down-range":
        "los rayos rasantes se curvan de vuelta al suelo\ny rebotan a lo largo de la distancia",
    "GFPE Relative Level at the Receiver Height (400 Hz, 2 m)":
        "Nivel relativo GFPE a la altura del receptor (400 Hz, 2 m)",
    "Downward (b = +1 m/s)": "Hacia abajo (b = +1 m/s)",
    "Homogeneous (b = 0)": "Homogénea (b = 0)",
    "Upward (b = -1 m/s)": "Hacia arriba (b = -1 m/s)",
    "Wave-Theoretic Barrier Insertion Loss":
        "Pérdida por inserción de barrera (teoría ondulatoria)",
    "Kurze-Anderson (thin screen)": "Kurze-Anderson (pantalla delgada)",
    "Exact rigid half-plane": "Semiplano rígido exacto",
    "Exact + coherent ground (four paths)":
        "Exacto + suelo coherente (cuatro caminos)",
    "Kurze-Anderson grazing limit (5 dB)":
        "Límite rasante de Kurze-Anderson (5 dB)",
    "Insertion loss [dB]": "Pérdida por inserción [dB]",
    # --- WP emission & electroacoustics figures (result .plot() labels) ---
    "Carrier f₂": "Portadora f₂",
    "Sidebands f₂ ± n·f₁": "Bandas laterales f₂ ± n·f₁",
    "Level re carrier [dB]": "Nivel respecto a la portadora [dB]",
    "$R_1$ (resistance)": "$R_1$ (resistencia)",
    "$X_1$ (reactance)": "$X_1$ (reactancia)",
    r"Normalized radiation impedance $Z_r / \rho c S$":
        r"Impedancia de radiación normalizada $Z_r / \rho c S$",
    "Baffled circular piston radiation impedance":
        "Impedancia de radiación de un pistón circular con pantalla",
    "F2 (surface pressure-intensity)": "F2 (presión-intensidad superficial)",
    "F3 (negative partial power)": "F3 (potencia parcial negativa)",
    "Dynamic capability Ld": "Capacidad dinámica Ld",
    "F4 (non-uniformity)": "F4 (no uniformidad)",
    "Indicator [dB]": "Indicador [dB]",
    "Field non-uniformity F4": "No uniformidad del campo F4",
    "ISO 9614-1 field indicators": "Indicadores de campo ISO 9614-1",
    "Helmholtz resonator": "Resonador de Helmholtz",
    "Quarter-wave tube": "Tubo de cuarto de onda",
    "Side-branch resonators: transmission loss (Bies Eqs. 8.44, 8.46)":
        "Resonadores en derivación: pérdida de transmisión (Bies Ecs. 8.44, 8.46)",
    "Duct end reflection loss (ASHRAE Table 8.14)":
        "Pérdida por reflexión del extremo del conducto (ASHRAE Tabla 8.14)",
    "End reflection loss [dB]": "Pérdida por reflexión del extremo [dB]",
    "Duct diameter": "Diámetro del conducto",
    "Duct-borne noise into the room: supply, return and NC 30":
        "Ruido transmitido por conductos a la sala: impulsión, retorno y NC 30",
    "Supply": "Impulsión",
    "Return": "Retorno",
    "Plant room to operator room: what the wall delivers, and NC 45":
        "Sala de máquinas a sala de control: lo que aporta el muro, y NC 45",
    "Duct higher-order-mode cut-on: 254 mm steam line at 200 m/s":
        "Corte de modos superiores del conducto: línea de vapor de 254 mm a 200 m/s",
    "Panel R": "R del panel",
    "Interior correction C": "Corrección interior C",
    "Insertion loss (R - C)": "Pérdida por inserción (R - C)",
    "Machine enclosure insertion loss":
        "Pérdida por inserción de encapsulado de máquina",
    "Measured phase": "Fase medida",
    "Minimum phase (from |H|)": "Fase mínima (de |H|)",
    "Excess phase (all-pass)": "Fase de exceso (pasa-todo)",
    "Phase [rad]": "Fase [rad]",
    "Minimum-phase / all-pass decomposition":
        "Descomposición fase mínima / pasa-todo",
    "Group delay": "Retardo de grupo",
    "Excess group delay": "Retardo de grupo de exceso",
    "Momentary (400 ms)": "Momentánea (400 ms)",
    "Short-term (3 s)": "Corto plazo (3 s)",
    "Programme loudness (EBU R 128)": "Sonoridad de programa (EBU R 128)",
    # Core-metrology figures: resampling, cepstrum, correlation, spectra,
    # system measurement, synchronous averaging, data qualification.
    "Polyphase Resampling 44.1 kHz → 48 kHz: the Delivered Anti-Alias Filter":
        "Remuestreo polifásico 44,1 kHz → 48 kHz: "
        "el filtro antisolapamiento entregado",
    "Anti-alias filter $|H(f)|$": "Filtro antisolapamiento $|H(f)|$",
    "Passband edge": "Borde de la banda de paso",
    "Stopband edge (alias fold)":
        "Borde de la banda atenuada (pliegue de alias)",
    "Design attenuation -120 dB": "Atenuación de diseño -120 dB",
    "Rejected band (would fold back as aliases)":
        "Banda rechazada (se plegaría como alias)",
    "The Three Cepstrum Variants of One Echo-Carrying Record":
        "Las tres variantes del cepstro de un registro con eco",
    "Real cepstrum (exactly half the power)":
        "Cepstro real (exactamente la mitad del de potencia)",
    "Complex cepstrum": "Cepstro complejo",
    "first rahmonic at 8 ms:\nheight ≈ a on the power cepstrum":
        "primer rahmónico en 8 ms:\naltura ≈ a en el cepstro de potencia",
    "second rahmonic: $-a^2/2$": "segundo rahmónico: $-a^2/2$",
    "Liftering at 4 ms: Envelope Versus Echo Ripple":
        "Liftering a 4 ms: envolvente frente a rizado del eco",
    "Log spectrum of the record": "Espectro logarítmico del registro",
    "Lowpass lifter: spectral envelope":
        "Lifter paso bajo: envolvente espectral",
    "Highpass lifter: the echo ripple alone":
        "Lifter paso alto: solo el rizado del eco",
    "closed-form ripple bounds $20\\lg(1\\pm a)$":
        "cotas del rizado en forma cerrada $20\\lg(1\\pm a)$",
    "Correlation Normalizations of a Two-Sensor Delay Model":
        "Normalizaciones de la correlación de un modelo "
        "de dos sensores con retardo",
    "Coefficient $\\rho_{xy}(\\tau)$ (bounded by $\\pm 1$)":
        "Coeficiente $\\rho_{xy}(\\tau)$ (acotado por $\\pm 1$)",
    "true delay +12.5 ms": "retardo verdadero +12,5 ms",
    "Biased $1/N$ (tapers toward the ends)":
        "Sesgada $1/N$ (se atenúa hacia los extremos)",
    "Unbiased $1/(N-|r|)$ (variance grows at the ends)":
        "Insesgada $1/(N-|r|)$ (la varianza crece en los extremos)",
    "Lag [s]": "Retardo [s]",
    "Correlation": "Correlación",
    "Sub-Sample Impulse-Response Alignment":
        "Alineación submuestra de respuestas al impulso",
    "Reference IR": "RI de referencia",
    "Measured IR (delayed 7.37 samples)":
        "RI medida (retardada 7,37 muestras)",
    "Aligned IR (delay removed)": "RI alineada (retardo eliminado)",
    "Hilbert Envelope and Instantaneous Frequency":
        "Envolvente de Hilbert y frecuencia instantánea",
    "Signal": "Señal",
    "Envelope $A(t)$": "Envolvente $A(t)$",
    "Instantaneous frequency $f(t)$": "Frecuencia instantánea $f(t)$",
    "Instantaneous frequency [Hz]": "Frecuencia instantánea [Hz]",
    "carrier 250 Hz": "portadora de 250 Hz",
    "Cross-Spectral Density of a 2 ms Delay Path":
        "Densidad espectral cruzada de un camino con retardo de 2 ms",
    "$|\\hat{G}_{xy}|$ (Welch estimate)":
        "$|\\hat{G}_{xy}|$ (estimación de Welch)",
    "Unwrapped phase": "Fase desenrollada",
    "$\\pm 1$ s.d. band (Eq. 9.52)": "banda $\\pm 1$ d.e. (Ec. 9.52)",
    "slope $-2\\pi f\\tau$ ($\\tau$ = 2 ms)":
        "pendiente $-2\\pi f\\tau$ ($\\tau$ = 2 ms)",
    "Coherent Output Spectrum and Spectral SNR (Bendat & Piersol 9.2.2)":
        "Espectro de salida coherente y SNR espectral (Bendat y Piersol 9.2.2)",
    "$\\hat{G}_{yy}$ (measured output)": "$\\hat{G}_{yy}$ (salida medida)",
    "$\\hat{G}_{vv} = \\gamma^2\\hat{G}_{yy}$ (coherent part)":
        "$\\hat{G}_{vv} = \\gamma^2\\hat{G}_{yy}$ (parte coherente)",
    "$\\hat{G}_{nn}$ (uncorrelated noise)":
        "$\\hat{G}_{nn}$ (ruido no correlado)",
    "Spectral density [dB re 1/Hz]": "Densidad espectral [dB re 1/Hz]",
    "Spectral SNR [dB]": "SNR espectral [dB]",
    "closed form $10\\lg(2.56)$ = 4.1 dB":
        "forma cerrada $10\\lg(2{,}56)$ = 4,1 dB",
    "Golay-Pair Impulse Response: Exact Complementary Recovery":
        "Respuesta al impulso con par de Golay: "
        "recuperación complementaria exacta",
    "Recovered IR (golay_impulse_response)":
        "RI recuperada (golay_impulse_response)",
    "True system response": "Respuesta verdadera del sistema",
    "noise-free closed-form identity": "identidad exacta sin ruido",
    "TSA Noise Reduction: the $\\sqrt{N}$ Law (McFadden 1987)":
        "Reducción de ruido del promediado síncrono: "
        "la ley $\\sqrt{N}$ (McFadden 1987)",
    "Measured RMS error of the average":
        "Error RMS medido del promedio",
    "Ideal $\\sigma/\\sqrt{N}$": "$\\sigma/\\sqrt{N}$ ideal",
    "Number of averages N": "Número de promedios N",
    "RMS error of the averaged waveform":
        "Error RMS de la forma de onda promediada",
    "Runs Test About the Median (Wald & Wolfowitz)":
        "Test de rachas respecto a la mediana (Wald y Wolfowitz)",
    "Above the median": "Por encima de la mediana",
    "Below the median": "Por debajo de la mediana",
    "Sequence median": "Mediana de la secuencia",
    # anim_fdtd_slit_absorber
    "Slow-sound slit absorber at critical coupling (2D FDTD)":
        "Absorbedor de rendija de sonido lento en acoplamiento crítico "
        "(FDTD 2D)",
    "Critically coupled slit: the wave dies inside":
        "Rendija en acoplamiento crítico: la onda muere dentro",
    "Wide slit (detuned): the reflection returns":
        "Rendija ancha (desintonizada): la reflexión vuelve",
    "inside the cell": "dentro de la celda",
    # anim_fdtd_expansion_chamber
    # Kept short: the canvas clips a suptitle much beyond ~80 characters.
    "Expansion-chamber silencer: pass band vs stop band (2D FDTD)":
        "Silenciador de cámara de expansión: paso frente a rechazo "
        "(FDTD 2D)",
    "the chamber is acoustically invisible":
        "la cámara es acústicamente invisible",
    "the mismatch reflects the wave back up the pipe":
        "el desajuste refleja la onda de vuelta por el conducto",
    "Position along the duct [m]": "Posición a lo largo del conducto [m]",
    # anim_fdtd_aperture_slit
    "Sound through a wall aperture (2D FDTD)":
        "Sonido a través de una abertura en un muro (FDTD 2D)",
    "cylindrical re-radiation from the slit":
        "re-radiación cilíndrica desde la rendija",
    "the front passes: sharp-edged shadow":
        "el frente pasa: sombra de bordes nítidos",
    "RMS level [dB]": "Nivel RMS [dB]",
    # anim_fdtd_refraction
    "Atmospheric refraction: downwind duct, upwind shadow (2D FDTD)":
        "Refracción atmosférica: canal a favor del viento, sombra en "
        "contra (FDTD 2D)",
    "Downwind: sound speed grows with height":
        "A favor del viento: la velocidad del sonido crece con la altura",
    "Upwind: sound speed falls with height":
        "En contra del viento: la velocidad del sonido cae con la altura",
    "bent down: a duct hugs the ground, the receiver stays loud":
        "curvada hacia abajo: un canal pegado al suelo, el receptor sigue "
        "oyendo",
    "bent up: a shadow opens, the receiver goes quiet":
        "curvada hacia arriba: se abre una sombra, el receptor se queda en "
        "silencio",
    "source (h = 2 m)": "fuente (h = 2 m)",
    "receiver 350 m": "receptor a 350 m",
    "c_eff(z) [m/s]": "c_ef(z) [m/s]",
    # anim_elastic_plate_junction
    "Bending waves at an L-junction (elastic 2D FDTD)":
        "Ondas de flexión en una unión en L (FDTD elástico 2D)",
    "Straight plate: the packet just runs on":
        "Placa recta: el paquete sigue de largo",
    "L-junction: reflected, transmitted and mode-converted":
        "Unión en L: reflexión, transmisión y conversión de modo",
    "nothing comes back": "no vuelve nada",
    "free end": "extremo libre",
    "4 kHz tone burst at ▼": "salva de tono de 4 kHz en ▼",
    # anim_elastic_coincidence
    "Coincidence: the same steel plate, below and above f_c "
    "(elastic 2D FDTD)":
        "Coincidencia: la misma placa de acero, bajo y sobre f_c "
        "(FDTD elástico 2D)",
    "10 mm steel plate": "placa de acero de 10 mm",
}

_ES_PATTERNS = [
    # masonry_wall_ties legend entries (tie name + baked-in Table A4 stiffness).
    (r"^butterfly \((.+) MN/m\)$", r"mariposa (\1 MN/m)"),
    (r"^double triangle \((.+) MN/m\)$", r"doble triángulo (\1 MN/m)"),
    (r"^vertical twist \((.+) MN/m\)$", r"torsión vertical (\1 MN/m)"),
    # heavy_impact_sources info box (baked-in ISO 717-2 Table D.4 sum).
    (r"^unrounded sum = (\d+)\.(\d+) dB$",
     r"suma sin redondear = \1,\2 dB"),
    # experimental_sea_clf info box (baked-in input power).
    ((r"^platform driven, cylinder driven only through the joints\n"
      r"input power = (.+) W\n"
      r"coupling stays well below the damping: valid SEA$"),
     ("plataforma excitada, cilindro excitado solo a través de las uniones\n"
      "potencia inyectada = \\1 W\n"
      "el acoplamiento queda muy por debajo del amortiguamiento: SEA válido")),
    # plateau_transmission_loss info box (baked-in panel and plateau values).
    ((r"^6 mm float glass, m'' = (.+) kg/m², η = (.+)\n"
      r"plateau height 27 dB, B/A = 10 → A = (.+) Hz, B = (.+) Hz\n"
      r"identical below A; the plateau replaces the whole coincidence region$"),
     ("vidrio float de 6 mm, m'' = \\1 kg/m², η = \\2\n"
      "altura de meseta 27 dB, B/A = 10 → A = \\3 Hz, B = \\4 Hz\n"
      "idénticas por debajo de A; la meseta sustituye toda la región de "
      "coincidencia")),
    # slow_sound_absorber panel-depth annotation (baked-in wavelength ratio).
    (r"^Normal incidence, rigid backing, panel depth L = lambda/(\d+)$",
     r"Incidencia normal, respaldo rígido, profundidad del panel L = lambda/\1"),
    # psd_confidence_smoothing annotation (mathtext + baked-in numbers).
    (r"^\$n_d\$ = (\d+) averages, \$\\varepsilon_r\$ = (\d+)\.(\d+) %$",
     r"$n_d$ = \1 promedios, $\\varepsilon_r$ = \2,\3 %"),
    # multitaper_psd_confidence annotation (baked-in dof value).
    (r"^171 ms record, \$NW\$ = 4, \$\\bar\\nu\$ = (\d+)\.(\d+) equivalent dof$",
     r"registro de 171 ms, $NW$ = 4, $\\bar\\nu$ = \1,\2 g.d.l. equivalentes"),
    # window_functions_tradeoff legend entries (name + baked-in metrics).
    (r"^([a-z]+): ENBW (.+) bins, sidelobe (.+) dB$",
     r"\1: ENBW \2 bins, lóbulo lateral \3 dB"),
    (r"^alpha = (.+) at (.+) Hz$",
     r"alfa = \1 a \2 Hz"),
    (r"^TL = (.+) dB at (.+) Hz$",
     r"TL = \1 dB a \2 Hz"),
    (r"^Integrated I = (.+) LUFS$", "Integrada I = \\1 LUFS"),
    (r"^LRA = (.+) LU \(P10-P95\)$", "LRA = \\1 LU (P10-P95)"),
    (r"^f = 10 kHz, α = (.+) dB/km\npractical spreading \(R₀ = 1000 m\)$",
     "f = 10 kHz, α = \\1 dB/km\\nensanchamiento práctico (R₀ = 1000 m)"),
    (r"^f = 250 Hz, H = 50 m, medium sand\nψc = (.+)°, η = (.+) Np/rad, (.+) modes$",
     "f = 250 Hz, H = 50 m, arena media\\nψc = \\1°, η = \\2 Np/rad, \\3 modos"),
    (r"^Low-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de baja frecuencia (AUD INJ \1 dB)"),
    (r"^High-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de alta frecuencia (AUD INJ \1 dB)"),
    (r"^Very high-frequency cetaceans \(AUD INJ (.+) dB\)$",
     r"Cetáceos de muy alta frecuencia (AUD INJ \1 dB)"),
    (r"^Phocid pinnipeds \(water\) \(AUD INJ (.+) dB\)$",
     r"Pinnípedos fócidos (agua) (AUD INJ \1 dB)"),
    (r"^Otariid pinnipeds \(water\) \(AUD INJ (.+) dB\)$",
     r"Pinnípedos otáridos (agua) (AUD INJ \1 dB)"),
    (r"^SL = 140, NL = 60, DI = 15, DT = 8 dB\nfigure of merit = (.+) dB$",
     "SL = 140, NL = 60, DI = 15, DT = 8 dB\\nfigura de mérito = \\1 dB"),
    (r"^SAE band \((\d+) m\)$", r"banda SAE (\1 m)"),
    (r"^source (\d+) m, receiver (.+) m, offset (\d+) m$",
     r"fuente \1 m, receptor \2 m, offset \3 m"),
    ((r"^SEL (\d+)\.(\d+) dB\(A\)  ·  EPNL (\d+)\.(\d+) EPNdB\n"
     r"level flyover, 60 kt, 150 m, 120 m sideline, grass$"),
     ("SEL \\1,\\2 dB(A)  ·  EPNL \\3,\\4 EPNdB\n"
     "sobrevuelo nivelado, 60 kt, 150 m, 120 m lateral, hierba")),
    (r"^\$L_\{ASmax\}\$ = (\d+)\.(\d+) dB\(A\)$",
     r"$L_{ASmax}$ = \1,\2 dB(A)"),
    (r"^Diffracted path \(δ = (\d+)\.(\d+) m\)$",
     r"Camino difractado (δ = \1,\2 m)"),
    (r"^Diffracted path \(δ = (\d+),(\d+) m\)$",
     r"Camino difractado (δ = \1,\2 m)"),
    (r"^(\d+) yr$", r"\1 años"),
    # anim_elastic_coincidence titles and verdicts (library values baked in).
    (r"^f = f_c/2 = (\d+) Hz, 45° incidence$",
     r"f = f_c/2 = \1 Hz, incidencia a 45°"),
    (r"^f = 2 f_c = (\d+) Hz, 45° incidence$",
     r"f = 2 f_c = \1 Hz, incidencia a 45°"),
    (r"^below f_c: the mass law holds: (.+) dB \(it predicts (.+)\)$",
     r"bajo f_c: manda la ley de masas: \1 dB (predice \2)"),
    (r"^above f_c: trace matches λ_B: (.+) dB, the mass law said (.+)$",
     r"sobre f_c: la traza iguala λ_B: \1 dB, la ley de masas decía \2"),
    (r"^coincidence_frequency: f_c = (\d+) Hz \(10 mm steel\)$",
     r"coincidence_frequency: f_c = \1 Hz (acero de 10 mm)"),
    # Weak-field display-gain notes (_gain_note); the factor is measured off
    # the field, so the number and its dB equivalent ride through.
    (r"^air below the plate drawn ×(\d+) \(\+(\d+) dB\)$",
     r"aire bajo la placa dibujado ×\1 (+\2 dB)"),
    (r"^past the wall drawn ×(\d+) \(\+(\d+) dB\)$",
     r"tras el muro dibujado ×\1 (+\2 dB)"),
    # seabed_reflection / seabed_reflection_coefficient critical-angle legend.
    (r"^Critical angle \((\d+)\.(\d+)°\)$", r"Ángulo crítico (\1,\2°)"),
    # equal_loudness_contours per-contour annotations ("20 phon" ... "90 phon").
    (r"^(\d+) phon$", r"\1 fonios"),
    (r"^total \(limit\) (.+) dB$", r"total (límite) \1 dB"),
    (r"^total \(eng\.\) (.+) dB$", r"total (ing.) \1 dB"),
    # tone_audibility decisive legend (mathtext skips the decimal-comma pass).
    (r"^decisive \$\\Delta L\$ = (\d+)\.(\d+) dB @ (\d+)\.(\d+) Hz$",
     r"decisiva $\\Delta L$ = \1,\2 dB @ \3,\4 Hz"),
    (r"^Governing  \$K_I\$ = (\d+)\.(\d+) dB$", r"Determinante  $K_I$ = \1,\2 dB"),
    # The mathtext ($R$) makes the later decimal-comma pass skip this label, so
    # convert the decimal here as part of the translation.
    (r"^Example  \$R\$ = (\d+)\.(\d+)$", r"Ejemplo  $R$ = \1,\2"),
    (r"^Octave Band: (.+) Hz$", r"Banda de octava: \1 Hz"),
    (r"^(\d+) phon$", r"\1 fonios"),
    (r"^TNR = (.+) dB\n\(criterion (.+) dB\)$", "TNR = \\1 dB\\n(criterio \\2 dB)"),
    (r"^MLS — first (\d+) of (\d+) samples$",
     r"MLS — primeras \1 de \2 muestras"),
    (r"^Measured 1/(\d+) Octave Bands$", r"Bandas de 1/\1 de octava medidas"),
    (r"^IEC target (.+) dB$", r"Objetivo IEC \1 dB"),
    (r"^([\d.]+) ms burst$", "R\u00e1faga de \\1 ms"),
    (r"^A-Weighting High-Frequency Accuracy @ fs=(\d+) kHz$",
     "Precisi\u00f3n en alta frecuencia de la ponderaci\u00f3n A @ fs=\\1 kHz"),
    (r"^Impulse Response \((.+) Hz Band\) - Transient/Stability Comparison$",
     "Respuesta al impulso (banda de \\1 Hz) \u2014 transitorio y estabilidad"),
    (r"^1 kHz narrowband - N = (.+) sone$",
     "Banda estrecha de 1 kHz - N = \\1 sonios"),
    (r"^Flat broadband 60 dB - N = (.+) sone$",
     "Banda ancha plana a 60 dB - N = \\1 sonios"),
    (r"^Pressure-intensity index\n\u03b4pI = (.+) dB$",
     "\u00cdndice presi\u00f3n-intensidad\\n\u03b4pI = \\1 dB"),
    (r"^Reference curve shifted by (.+) dB$",
     r"Curva de referencia desplazada \1 dB"),
    (r"^Sum of unfavourable deviations = (.+) dB  \(limit 32\.0 dB\)$",
     "Suma de desviaciones desfavorables = \\1 dB  (l\u00edmite 32,0 dB)"),
    (r"^Aures \(Annex B, N = (.+) sone\)$",
     r"Aures (Anexo B, N = \1 sonios)"),
    (r"^Spatial decay D2,S = (.+) dB$",
     r"Decaimiento espacial D2,S = \1 dB"),
    (r"^Zwicker \(ISO 532-1\), N = (.+) sone$",
     r"Zwicker (ISO 532-1), N = \1 sonios"),
    (r"^Moore-Glasberg \(ISO 532-2\), N = (.+) sone$",
     r"Moore-Glasberg (ISO 532-2), N = \1 sonios"),
    (r"^Sottek \(ECMA-418-2\), N = (.+) sone$",
     r"Sottek (ECMA-418-2), N = \1 sonios"),
    (r"^1 kHz tone, 60 dB \(N = (.+) sone_HMS\)$",
     r"Tono de 1 kHz, 60 dB (N = \1 sonios_HMS)"),
    (r"^Tone in noise \(T = (.+) tu_HMS\)$",
     r"Tono en ruido (T = \1 tu_HMS)"),
    (r"^Pure noise \(T = (.+) tu_HMS\)$",
     r"Ruido puro (T = \1 tu_HMS)"),
    (r"^Peak R = (.+) asper @ (.+) Hz$",
     r"Máximo R = \1 asper @ \2 Hz"),
    (r"^AM broadband noise \(closed form, 60 dB\), peak (.+) vacil$",
     r"Ruido de banda ancha AM (forma cerrada, 60 dB), máximo \1 vacil"),
    (r"^AM tone \(signal model, 70 dB\), peak (.+) vacil$",
     r"Tono AM (modelo de señal, 70 dB), máximo \1 vacil"),
    (r"^Worked example \(PA = (.+)\)$",
     r"Ejemplo resuelto (PA = \1)"),
    (r"^PA = (.+)\nwS = (.+), wFR = (.+)$",
     "PA = \\1\\nwS = \\2, wFR = \\3"),
    (r"^Short-term loudness STL \(STL peak = (.+) sone\)$",
     r"Sonoridad a corto plazo STL (STL máx = \1 sonios)"),
    (r"^Long-term loudness LTL \(LTL peak = (.+) sone\)$",
     r"Sonoridad a largo plazo LTL (LTL máx = \1 sonios)"),
    (r"^floor-(.+)$", r"suelo-\1"),
    (r"^ceiling-(.+)$", r"techo-\1"),
    (r"^facade-(.+)$", r"fachada-\1"),
    (r"^wall-(.+)$", r"tabique-\1"),
    (r"^(.+) °C, (.+) % RH$", r"\1 °C, \2 % HR"),
    # Materials: absorption rating & airflow resistance annotations
    (r"^Reference curve shifted by ([\d.]+)$",
     r"Curva de referencia desplazada \1"),
    (r"^Sum of unfavourable deviations = (.+)  \(limit 0\.10\)$",
     "Suma de desviaciones desfavorables = \\1  (límite 0,10)"),
    (r"^Absorption class (.+)  \(shape indicator: (.+)\)$",
     r"Clase de absorción \1  (indicador de forma: \2)"),
    (r"^Specific airflow resistance R_s = (.+) Pa s/m$",
     r"Resistencia específica al flujo R_s = \1 Pa s/m"),
    (r"^Airflow resistivity sigma = (.+) Pa s/m\^2$",
     r"Resistividad al flujo sigma = \1 Pa s/m^2"),
    (r"^Linear term a = (.+) Pa s/m  \(= R_s at u -> 0\)$",
     r"Término lineal a = \1 Pa s/m  (= R_s en u -> 0)"),
    # Scattering / diffusion / precision power dynamic titles (numeric d / LWA)
    (r"^Directional diffusion  d = (.+)  \(ISO 17497-2\)$",
     r"Difusión direccional  d = \1  (ISO 17497-2)"),
    (r"^Precision sound power \(ISO 3745\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica de precisión (ISO 3745)  LWA = \1 dB(A)"),
    (r"^Precision intensity scanning \(ISO 9614-3\)  LWA = (.+) dB\(A\)$",
     r"Barrido de intensidad de precisión (ISO 9614-3)  LWA = \1 dB(A)"),
    (r"^Enveloping-surface sound power \(ISO 3744\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica por superficie envolvente (ISO 3744)  LWA = \1 dB(A)"),
    (r"^Reverberation-room sound power \(ISO 3741\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica en cámara reverberante (ISO 3741)  LWA = \1 dB(A)"),
    (r"^Intensity-scanning sound power \(ISO 9614-2\)  LWA = (.+) dB\(A\)$",
     r"Potencia acústica por barrido de intensidad (ISO 9614-2)  LWA = \1 dB(A)"),
    # Human-vibration dynamic titles (numeric a_w / A(8))
    (r"^Weighted seat acceleration \(ISO 2631-1\)  (.+)$",
     r"Aceleración ponderada del asiento (ISO 2631-1)  \1"),
    (r"^Hand-arm daily exposure \(ISO 5349 / 2002-44-EC\)  (.+)$",
     r"Exposición diaria mano-brazo (ISO 5349 / 2002-44-EC)  \1"),
    # Speech intelligibility dynamic title (numeric SII)
    (r"^Speech Intelligibility Index \(ANSI S3\.5-1997\)   SII = (.+)$",
     r"Índice de inteligibilidad del habla (ANSI S3.5-1997)   SII = \1"),
    # Room-noise criteria (ANSI S12.2-2019) dynamic titles/legends
    (r"^Noise Criteria — tangency method   NC-(.+)$",
     r"Criterios de ruido — método de tangencia   NC-\1"),
    (r"^Room Criteria Mark II   RC-(.+)$",
     r"Criterios de sala Mark II   RC-\1"),
    (r"^Tangent @ (.+) Hz$", r"Tangente @ \1 Hz"),
    (r"^Reference RC-(.+)$", r"Referencia RC-\1"),
    (r"^(\d+) yr$", r"\1 años"),
    (r"^10-90 % band \((\d+) yr\)$", r"banda 10-90 % (\1 años)"),
    # Tier-1 animation dynamic labels
    (r"^remaining energy: (.+) %$", r"energía restante: \1 %"),
    (r"^On the \(2,1\) mode: (.+) Hz$", r"En el modo (2,1): \1 Hz"),
    (r"^Off mode: (.+) Hz$", r"Fuera de modo: \1 Hz"),
    (r"^(.+) sone$", r"\1 sonios"),
    (r"^mean Lp = (.+) dB$", r"Lp medio = \1 dB"),
    (r"^first notch (.+) Hz$", r"primer nulo \1 Hz"),
    # FDTD animation dynamic labels (third batch)
    (r"^Low frequency: (.+)$", r"Baja frecuencia: \1"),
    (r"^High frequency: (.+)$", r"Alta frecuencia: \1"),
    (r"^insertion loss (.+) dB$", r"pérdida por inserción \1 dB"),
    (r"^diffusion coefficient d = (.+)$", r"coeficiente de difusión d = \1"),
    (r"^design frequency (.+) Hz$", r"frecuencia de diseño \1 Hz"),
    # 2D FDTD wave simulation (public API concept figure)
    (r"^FDTD pressure field at t = (.+) ms$",
     r"Campo de presión FDTD en t = \1 ms"),
    (r"^probe \((.+)\) m$", r"sonda (\1) m"),
    # Building & structure-borne result figures (dynamic values baked in).
    (r"^Airborne: Rw\(C;Ctr\) = (.+) dB$", r"Aéreo: Rw(C;Ctr) = \1 dB"),
    (r"^Impact: Ln,w\(CI\) = (.+) dB$", r"Impacto: Ln,w(CI) = \1 dB"),
    (r"^coincidence fc = (\d+) Hz$", r"coincidencia fc = \1 Hz"),
    (r"^\$L_\{WA\}\$ = (\d+)\.(\d+) dB\(A\)$", r"$L_{WA}$ = \1,\2 dB(A)"),
    # Their multi-line info boxes are single Text artists, so the whole
    # joined string is matched at once (values stay as capture groups).
    ((r"^Rw\(C;Ctr\) = (.+)\nC50-5000 = (.+)\n"
      r"rating on the core bands, terms on the full range$"),
     ("Rw(C;Ctr) = \\1\nC50-5000 = \\2\n"
      "índice en las bandas básicas, términos en el rango completo")),
    ((r"^Dls,2m,nT,w\(C;Ctr\) = (.+) dB\n45° loudspeaker method "
      r"\(-1\.5 dB on R'\)$"),
     "Dls,2m,nT,w(C;Ctr) = \\1 dB\nmétodo del altavoz a 45° (-1,5 dB en R')"),
    ((r"^L'nT,w\(CI\) = (.+) dB\nnote the minus sign: a live room lowers "
      r"L'nT$"),
     "L'nT,w(CI) = \\1 dB\natención al signo menos: un recinto vivo reduce L'nT"),
    ((r"^LW = Lp,in \+ Cd - R' \+ 10 lg\(S/S0\)\nwall 176 m² \+ industrial "
      r"door 24 m², Cd = -5 dB$"),
     ("LW = Lp,in + Cd - R' + 10 lg(S/S0)\n"
      "muro de 176 m² + puerta industrial de 24 m², Cd = -5 dB")),
    ((r"^Rw\(C;Ctr\) = (.+) dB\n6 mm float glass, m'' = 15 kg/m², "
      r"η = 0\.024$"),
     "Rw(C;Ctr) = \\1 dB\nvidrio flotado de 6 mm, m'' = 15 kg/m², η = 0,024"),
    ((r"^Kij = 10 lg\(1/τ̄\) \+ 5 lg\(fc2/1000\)\nconcrete, plate 1 fixed "
      r"at 100 mm$"),
     "Kij = 10 lg(1/τ̄) + 5 lg(fc2/1000)\nhormigón, placa 1 fija en 100 mm"),
    ((r"^below f0: stiffness-controlled, \|Y\| ~ ω/k\n"
      r"above f0: mass-controlled, \|Y\| ~ 1/\(ωm\)\n"
      r"f0 = (.+) Hz,  1/c = (.+) m/\(N·s\)$"),
     ("por debajo de f0: dominio de la rigidez, |Y| ~ ω/k\n"
      "por encima de f0: dominio de la masa, |Y| ~ 1/(ωm)\n"
      "f0 = \\1 Hz,  1/c = \\2 m/(N·s)")),
    # Programme-loudness .plot() legend lines (loudness_gating/loudness_range).
    (r"^Integrated (.+) LUFS$", r"Integrada \1 LUFS"),
    (r"^Ungated mean (.+) LUFS$", r"Media sin puerta \1 LUFS"),
    # Core-metrology figures: dynamic verdict / error strings
    (r"^r = (\d+) runs, accept \((\d+), (\d+)\]: trend-free$",
     r"r = \1 rachas, aceptación (\2, \3]: sin tendencia"),
    (r"^r = (\d+) runs, accept \((\d+), (\d+)\]: rejected$",
     r"r = \1 rachas, aceptación (\2, \3]: rechazada"),
    (r"^max \|recovered - true\| = (.+)$",
     r"máx |recuperada - verdadera| = \1"),
    (r"^estimated delay removed: (.+) samples$",
     r"retardo estimado eliminado: \1 muestras"),
    # FDTD second-batch clips: baked-number pills and titles
    (r"^slit h = (.+) mm$", r"rendija h = \1 mm"),
    (r"^Pass band: (\d+) Hz, kL = π$", r"Banda de paso: \1 Hz, kL = π"),
    (r"^Stop band peak: (\d+) Hz, kL = π/2$",
     r"Pico de rechazo: \1 Hz, kL = π/2"),
    (r"^Slit w = (\d+) mm \(λ/20\)$", r"Rendija w = \1 mm (λ/20)"),
    (r"^Opening w = (.+) m \(= λ\)$", r"Hueco w = \1 m (= λ)"),
    (r"^slit τ = (.+) \(Gomperts\)$", r"rendija τ = \1 (Gomperts)"),
    (r"^f = (\d+) Hz \(λ = (.+) m\)$", r"f = \1 Hz (λ = \2 m)"),
    (r"^shadow beyond ≈ (\d+) m \(ray model\)$",
     r"sombra más allá de ≈ \1 m (modelo de rayos)"),
    # cnossos_road_speed_law rolling/propulsion crossover annotation.
    (r"^crossover (\d+) km/h$", r"cruce \1 km/h"),
    # orthotropic_transmission_loss info box (library values baked in).
    ((r"^1 mm steel sheet, m'' = (.+) kg/m², flat fc = (.+) kHz\n"
      r"corrugated H = (\d+) mm, L = (\d+) mm, m'' = (.+) kg/m², "
      r"fc1 = (.+) Hz, fc2 = (.+) kHz\n"
      r"worst penalty (.+) dB at (.+) Hz, for a stiffer and only 9 % "
      r"heavier panel$"),
     ("chapa de acero de 1 mm, m'' = \\1 kg/m², fc plana = \\2 kHz\n"
      "grecada H = \\3 mm, L = \\4 mm, m'' = \\5 kg/m², "
      "fc1 = \\6 Hz, fc2 = \\7 kHz\n"
      "penalización máxima \\8 dB a \\9 Hz, con un panel más rígido y solo "
      "un 9 % más pesado")),
    # limp_frame_effective_density annotations (library values baked in).
    (r"^apparent total density rho_t/rho0 = (.+)$",
     r"densidad total aparente rho_t/rho0 = \1"),
    (r"^decoupling frequency (.+) Hz$", r"frecuencia de desacoplo \1 Hz"),
    # biot_frame_resonance annotation (library value baked in).
    (r"^frame quarter-wave resonance (.+) Hz$",
     r"resonancia de cuarto de onda del esqueleto \1 Hz"),
    ((r"^Soft fibrous layer: porosity (.+), flow resistivity (.+) kPa s/m², "
      r"frame density (.+) kg/m³$"),
     ("Capa fibrosa blanda: porosidad \\1, resistividad al flujo \\2 kPa s/m², "
      "densidad del esqueleto \\3 kg/m³")),
]


def set_lang(lang: str) -> None:
    """Switch the output language ('en' or 'es')."""
    global _LANG, _LANG_SUFFIX
    _LANG = lang
    _LANG_SUFFIX = "" if lang == "en" else f"_{lang}"


def _translate_figure(fig: Any) -> None:
    """Rewrite every Text artist of *fig* into the active language."""
    import re as _re

    import matplotlib.text as _mtext

    if _LANG == "en":
        return
    import re as _re2

    from matplotlib.ticker import FixedFormatter as _FxF
    from matplotlib.ticker import FuncFormatter as _FF
    from matplotlib.ticker import ScalarFormatter as _SF

    def _comma(s: str) -> str:
        # A letter immediately before the number marks a standard designation
        # (e.g. "S3.5"), not a decimal - leave those untouched.
        return _re2.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", s)

    def _tr_words(s: str) -> str:
        """Apply the exact / pattern lookups (no decimal comma) to *s*."""
        if s in _ES_EXACT:
            return _ES_EXACT[s]
        for pat, repl in _ES_PATTERNS:
            new, n = _re.subn(pat, repl, s)
            if n:
                return new
        return s

    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            fmt = axis.get_major_formatter()
            if isinstance(fmt, _FxF):
                # Translate categorical tick labels (e.g. path names) too;
                # numeric labels match nothing and only get the decimal comma.
                fmt.seq = [_comma(_tr_words(s)) for s in fmt.seq]
            elif isinstance(fmt, _FF) and not getattr(fmt, "_phonometry_comma", False):
                # Categorical labels (set_xticklabels installs a FuncFormatter)
                # need the word lookups too; numeric labels are untouched.
                wrapped = _FF(
                    lambda v, pos, _f=fmt: _comma(_tr_words(str(_f(v, pos))))
                )
                wrapped._phonometry_comma = True  # type: ignore[attr-defined]
                axis.set_major_formatter(wrapped)
            elif type(fmt) is _SF and axis.get_scale() == "linear":
                wrapped = _FF(lambda v, pos: _comma(f"{v:g}"))
                axis.set_major_formatter(wrapped)
    for artist in fig.findobj(_mtext.Text):
        s = artist.get_text()
        if not s:
            continue
        if s in _ES_EXACT:
            artist.set_text(_ES_EXACT[s])
        else:
            for pat, repl in _ES_PATTERNS:
                new, n = _re.subn(pat, repl, s)
                if n:
                    artist.set_text(new)
                    break
        # Spanish decimal comma, applied uniformly to every text artist
        # (tick labels included) except mathtext. The substitution itself is
        # conservative -- it only rewrites a bare ``digit.digit`` not adjacent
        # to further digits/dots -- so underscore-bearing unit tokens such as
        # ``sone_HMS`` / ``tu_HMS`` keep their identifier intact while genuine
        # decimals in the same label (e.g. ``8.0 sone_HMS``) still get commas.
        s = artist.get_text()
        if s and "$" not in s and _re.search(r"\d\.\d", s):
            # Clause/version numbers like 5.3.3 and standard designations like
            # "S3.5" (a letter immediately before the number) keep their dots.
            artist.set_text(
                _re.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", s)
            )


LABEL_FREQ_HZ = "Frequency [Hz]"
LABEL_LEVEL_DB = "Level [dB]"
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#d62728"
COLOR_TERTIARY = "#2ca02c"
COLOR_GRID = "#e0e0e0"
# Neutral for de-emphasised *data* (a context bar, a reference series): mid
# grey reads on both pages, unlike COLOR_GRID, which is tuned to disappear
# into the one it is drawn on.
COLOR_MUTED = "#9e9e9e"

# Theme state: every figure is generated twice (light + "_dark" suffix).
# COLOR_FG replaces literal black so annotations stay visible on dark bg.
COLOR_FG = "black"
# Solid backing for a text box that has to stay readable over the plotted
# data: a shade off the page rather than the page itself, so the box reads as
# a panel and its edge is not the only thing separating it from the figure.
COLOR_PANEL = "#f0f2f5"
_FILENAME_SUFFIX = ""


# The dark-theme wave-field colormap (berlin with its centre bias subtracted
# so zero maps to true black) now lives in the library, where the result
# ``.plot()`` renderers pick it whenever the axes background is dark; the
# clips address it by name, so it is registered eagerly here.
_register_field_dark_cmap()

# Theme-dependent wave-field styling, switched by set_theme(): the diverging
# colormap of the instantaneous-pressure/velocity panels (zero maps to the
# page background on both themes) and the ink/halo pair of the annotations
# drawn *inside* those fields. Sequential magma panels (RMS maps, level maps)
# are theme-independent and keep their hardcoded white ink.
CMAP_FIELD = "RdBu_r"
FIELD_INK = "black"
FIELD_STROKE = "white"

# Global matplotlib configuration
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def set_theme(dark: bool) -> None:
    """Switch between the light (default) and dark documentation themes."""
    global COLOR_FG, COLOR_GRID, COLOR_PANEL, _FILENAME_SUFFIX
    global CMAP_FIELD, FIELD_INK, FIELD_STROKE
    if dark:
        plt.style.use("dark_background")
        COLOR_FG = "white"
        COLOR_GRID = "#555555"
        COLOR_PANEL = "#1c2128"
        _FILENAME_SUFFIX = "_dark"
        CMAP_FIELD = "phonometry_field_dark"
        FIELD_INK = "white"
        FIELD_STROKE = "black"
    else:
        plt.style.use("default")
        COLOR_FG = "black"
        COLOR_GRID = "#e0e0e0"
        COLOR_PANEL = "#f0f2f5"
        _FILENAME_SUFFIX = ""
        CMAP_FIELD = "RdBu_r"
        FIELD_INK = "black"
        FIELD_STROKE = "white"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "figure.figsize": (10, 6),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


# Figures kept as raster (lossless WebP) because SVG would be strictly heavier,
# for two reasons:
#   * raster-backed (pcolormesh / specgram): the SVG would only wrap a base64
#     bitmap several times larger than the raster;
#   * dense time series (thousands of vertices): every sample becomes a path
#     coordinate, so the SVG runs 5.5-7.75x the raster (schroeder_decay
#     105->820 KB, calibration_stability 99->759 KB, impulse_response
#     134->747 KB, measured against the earlier PNG encoding).
# Both clear the ~4.5x "SVG no longer wins" bar; everything else is a vector plot
# written as deterministic SVG (moderate 2.4x cases like sel_concept /
# ln_levels_example stay SVG -- below the bar, and vector crispness is worth it).
_RASTER_FIGURES = frozenset(
    {
        "spectrogram_example",
        # imshow raster: the STFT cells would bloat an SVG (and the repo's
        # figure policy keeps rasters out of SVG files).
        "calibrated_spectrogram",
        "excitation_signals",
        "schroeder_decay",
        "calibration_stability",
        "impulse_response",
        "numerical_propagation",
        "atmospheric_refraction",
        "airport_contour",
        "fdtd_simulation",
        "elastic_halfspace_waves",
        "scholte_interface_wave",
        # Dense reflectogram: hundreds of image-source stems/markers make the
        # SVG far heavier than the raster (as for schroeder_decay above).
        "image_source_reflectogram",
        # Dense time series (tens of thousands of vertices), as for
        # calibration_stability above: the noise-floor traces of the
        # correlation normalizations make the SVG several times the raster.
        "correlation_normalizations",
    }
)


def save_figure(output_dir: str, filename: str, **kwargs: Any) -> None:
    """Translate, theme-suffix and save the current figure.

    Vector figures are written as SVG made byte-reproducible with a fixed
    ``svg.hashsalt`` (deterministic element ids), ``svg.fonttype = "none"``
    (text kept as ``<text>`` rather than freetype glyph outlines, so the
    output does not depend on the font build) and no date metadata -- so
    ``make graphs`` is stable and CI can diff it. The figures in
    :data:`_RASTER_FIGURES` stay raster (SVG would be heavier), written as
    lossless WebP: matplotlib renders the canvas to an in-memory PNG with the
    version-stamped ``Software`` chunk (and any date) stripped, and Pillow
    re-encodes those exact pixels with the exhaustive ``method=6`` search,
    whose output is fixed by the input, so the WebP bytes are reproducible
    across matplotlib builds and runs. In both cases ``filename`` may carry
    any extension; the real one is chosen here.
    """
    _translate_figure(plt.gcf())
    stem = os.path.splitext(filename)[0]
    ext = "webp" if stem in _RASTER_FIGURES else "svg"
    path = os.path.join(output_dir, f"{stem}{_LANG_SUFFIX}{_FILENAME_SUFFIX}.{ext}")
    if ext == "svg":
        plt.rcParams["svg.hashsalt"] = "phonometry"
        plt.rcParams["svg.fonttype"] = "none"
        kwargs.setdefault("metadata", {"Date": None})
        plt.savefig(path, **kwargs)
        # svg.fonttype='none' keeps text selectable, but matplotlib writes
        # mathtext runs ('$...$') with a bare font-family ('DejaVu Sans', no
        # generic), so viewers lacking that font fall back to a serif while
        # plain <text> (full chain ending in sans-serif) stays sans. Normalise
        # every declaration to the generic 'sans-serif' so all text renders in
        # the viewer's native sans font, consistently and viewer-independently.
        import re as _re
        svg_text = pathlib.Path(path).read_text(encoding="utf-8")
        svg_text = _re.sub(r"font-family:\s*[^;\"]+", "font-family: sans-serif", svg_text)
        pathlib.Path(path).write_text(svg_text, encoding="utf-8")
        return
    import io

    from PIL import Image

    # Render the raster figures at 300 dpi so they stay crisp on
    # high-density displays and when zoomed in the docs. Individual call
    # sites may still override this.
    kwargs.setdefault("dpi", 300)
    # Drop matplotlib's version-stamped Software chunk (and any date). Merge
    # instead of setdefault: a caller-supplied metadata dict must not silently
    # reintroduce the version-dependent chunks.
    kwargs["metadata"] = {"Software": None, "Date": None,
                          **kwargs.get("metadata", {})}
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="png", **kwargs)
        buffer.seek(0)
        with Image.open(buffer) as image:
            # Lossless WebP: identical pixels to the rendered canvas at roughly
            # half the optimized-PNG size (flat-color plot areas compress better
            # lossless than lossy). method=6 is the slowest, most exhaustive
            # encoder search; its output is a pure function of the input pixels,
            # so the committed bytes are byte-stable across runs.
            image.save(path, "WEBP", lossless=True, quality=100, method=6)



def apply_axis_styling(ax: Any, title: str, xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
    """Apply consistent styling to plots."""
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-")
    ax.grid(which="minor", color=COLOR_GRID, linestyle=":", alpha=0.4)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Standard Octave Ticks
    xticks = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    xticklabels = ["16", "31.5", "63", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)


def plot_psd(ax: Any, x: np.ndarray, fs: int, label: str = "Raw Signal PSD", color: str = "gray", alpha: float = 0.3) -> None:
    """Calculate and plot the Power Spectral Density of the raw signal."""
    # Use Welch's method for a smooth PSD estimate
    f, Pxx = scipy_signal.welch(x, fs, nperseg=4096)
    
    # Convert to dB (relative to max to match SPL scale roughly or just show shape)
    # Since SPL is calibrated differently, we just want to show the 'shape' of the spectrum
    # in the background. We can normalize Pxx to match the peak of the octave bands roughly
    # or just plot it as is if we had calibrated units.
    # Here we'll just plot relative dB
    
    # Avoid log(0)
    Pxx_db = 10 * np.log10(Pxx + 1e-12)
    
    # Normalize PSD peak to 0 dB for visualization shape, then shift down? 
    # Or better: don't normalize, just plot. But PSD density vs Octave Band Power (integrated) 
    # are different units (dB/Hz vs dB).
    # So we will plot it on a secondary Y axis or just scaled to fit nicely in background.
    
    # Let's shift it so its mean roughly aligns with the mean of the SPL for visualization
    # This is purely for qualitative comparison of "where the energy is".
    
    ax.semilogx(f, Pxx_db, color=color, alpha=alpha, linewidth=1, label=label, zorder=0)


def generate_filter_type_comparison(output_dir: str) -> None:
    """Compare different filter architectures with a zoom inset."""
    print("Generating filter_type_comparison.png...")
    fs = 48000
    fraction = 1
    order = 6
    
    # We want exactly the 1000Hz band
    limits = [800.0, 1200.0]
    
    filters = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I", COLOR_SECONDARY, "--"),
        ("cheby2", "Chebyshev II", COLOR_TERTIARY, ":"),
        ("ellip", "Elliptic", "#9467bd", "-."),
        ("bessel", "Bessel", "#8c564b", "-"),
    ]
    
    _, ax = plt.subplots(figsize=(10, 7))
    
    # Create inset axis for zoom (increased height to 45%)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="35%", height="45%", loc="upper left", borderpad=3)
    axins.set_xscale("log") # Explicitly set log scale
    
    for f_type, label, color, style in filters:
        bank = OctaveFilterBank(fs, fraction=fraction, order=order, limits=limits, filter_type=f_type)
        
        # Find index of 1000Hz band
        idx = np.argmin(np.abs(np.array(bank.freq) - 1000))
        
        fsd = fs / bank.factor[idx]
        w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=16384, fs=fsd)
        mag_db = 20 * np.log10(np.abs(h) + 1e-9)
        
        ax.semilogx(w, mag_db, label=label, color=color, linestyle=style)
        axins.plot(w, mag_db, color=color, linestyle=style)

    ax.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3, label="-3 dB")
    axins.axhline(-3, color=COLOR_FG, linestyle=":", alpha=0.3)
    
    apply_axis_styling(ax, "Filter Architecture Comparison (Order 6, 1kHz Band)", xlim=(100, 8000), ylim=(-80, 5))
    
    # Sub-plot styling (Zoom around 1kHz and -3dB)
    axins.set_xlim(650, 1500)
    axins.set_ylim(-4, 0.5)  # Adjusted: from -4 to 0.5
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom at -3 dB (Log Scale)", fontsize=9)

    # Fix x-ticks for log scale zoom to look right
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.xaxis.set_minor_formatter(NullFormatter())  # Hide minor tick labels
    axins.xaxis.get_major_formatter().set_scientific(False)  # Disable scientific notation
    axins.set_xticks([707, 1000, 1414])
    axins.set_xticklabels(["707", "1000", "1414"], fontsize=8)

    ax.legend(loc="lower right")
    save_figure(output_dir, "filter_type_comparison.png")
    plt.close()


def generate_filter_responses(output_dir: str) -> None:
    """Generate plots for the filter bank responses for different filter types."""
    fs = 48000
    
    # Filter types to generate
    filter_types = [
        ("butter", "butter"),
        ("cheby1", "cheby1"),
        ("cheby2", "cheby2"),
        ("ellip", "ellip"),
        ("bessel", "bessel"),
    ]

    configs = [
        (1, 6),
        (3, 6),
    ]

    for f_type_name, f_type in filter_types:
        for fraction, order in configs:
            filename = f"filter_{f_type_name}_fraction_{fraction}_order_{order}.png"
            print(f"Generating {filename}...")
            bank = OctaveFilterBank(fs=fs, fraction=fraction, order=order, limits=[12.0, 20000.0], filter_type=f_type)
            
            from phonometry.metrology.filter_design import _showfilter
            # Draw first, then save through save_figure so the Spanish
            # translation pass runs on the finished figure (it rewrites the
            # live figure's text artists right before the save).
            _showfilter(bank.sos, bank.freq, bank.freq_u, bank.freq_d, fs,
                        bank.factor, show=False, plot_file=None, close=False)
            save_figure(output_dir, filename, dpi=150,
                        bbox_inches="tight")
            plt.close("all")


def generate_signal_responses(output_dir: str) -> None:
    """Generate spectral analysis plots for a complex signal."""
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    freqs = [20, 100, 500, 2000, 4000, 15000]
    y = 100 * np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)

    for frac, filename, title in [
        (3, "signal_response_fraction_3.png", "1/3 Octave Band Analysis"),
    ]:
        print(f"Generating {filename}...")
        bank = OctaveFilterBank(fs=fs, fraction=frac, order=6, limits=[12.0, 20000.0])
        spl, freq = bank.filter(y)

        _, ax = plt.subplots()
        
        # Plot PSD of raw signal in background
        # We need to scale PSD to comparable levels. 
        # A simple hack for visualization is to align the max of PSD to max of SPL
        f_psd, Pxx = scipy_signal.welch(y, fs, nperseg=8192)
        Pxx_db = 10 * np.log10(Pxx + 1e-12)
        # Shift PSD to match SPL peak roughly
        Pxx_db += (np.max(spl) - np.max(Pxx_db)) - 5 # Shift slightly below
        
        ax.semilogx(f_psd, Pxx_db, color="gray", alpha=0.6, linewidth=1.2, label="Raw Signal Spectrum (PSD)", zorder=0)
        
        ax.semilogx(
            freq,
            spl,
            marker="o",
            markersize=5,
            linestyle="-",
            color=COLOR_PRIMARY,
            linewidth=1.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"Measured 1/{frac} Octave Bands"
        )
        apply_axis_styling(ax, title, xlim=(11, 25000))
        ax.legend(loc="lower right")
        save_figure(output_dir, filename)
        plt.close()


def generate_multichannel_response(output_dir: str) -> None:
    """Generate analysis plot for a stereo signal with separate subplots."""
    print("Generating signal_response_multichannel.png...")
    fs = 48000
    duration = 5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    rng = np.random.default_rng(42)
    # Channel 1: Pink Noise (Voss-McCartney simplified)
    # Good enough for visualization
    white = rng.standard_normal(len(t))
    b, a = scipy_signal.butter(1, 0.04) # -3dB/oct approx
    ch1 = scipy_signal.lfilter(b, a, white)
    ch1 = (ch1 - np.mean(ch1)) / np.max(np.abs(ch1))

    # Channel 2: Logarithmic Sine Sweep
    ch2 = scipy_signal.chirp(t, f0=50, t1=duration, f1=10000, method="logarithmic")

    x = np.vstack((ch1, ch2))
    bank = OctaveFilterBank(fs=fs, fraction=3, order=6, limits=[20.0, 20000.0])
    spl, freq = bank.filter(x)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Calculate PSDs for background
    f_psd1, Pxx1 = scipy_signal.welch(x[0], fs, nperseg=4096)
    Pxx_db1 = 10 * np.log10(Pxx1 + 1e-12)
    Pxx_db1 += (np.max(spl[0]) - np.max(Pxx_db1)) # Align peaks
    
    f_psd2, Pxx2 = scipy_signal.welch(x[1], fs, nperseg=4096)
    Pxx_db2 = 10 * np.log10(Pxx2 + 1e-12)
    Pxx_db2 += (np.max(spl[1]) - np.max(Pxx_db2)) # Align peaks

    # Plot Left Channel
    ax1.semilogx(f_psd1, Pxx_db1, color="gray", alpha=0.6, linewidth=1.2, label="Raw PSD", zorder=0)
    ax1.semilogx(
        freq,
        spl[0],
        marker="o",
        markersize=5,
        label="Left Channel: Pink Noise",
        color=COLOR_PRIMARY,
        linestyle="-",
        linewidth=1.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    # Use standard styling but override title
    apply_axis_styling(ax1, "Multichannel Analysis (Stereo Input)", xlim=(16, 20000))
    ax1.legend(loc="lower right")
    # Let Y-axis autoscale

    # Plot Right Channel
    ax2.semilogx(f_psd2, Pxx_db2, color="gray", alpha=0.6, linewidth=1.2, label="Raw PSD", zorder=0)
    ax2.semilogx(
        freq,
        spl[1],
        marker="s",
        markersize=5,
        label="Right Channel: Log Sine Sweep",
        color=COLOR_SECONDARY,
        linestyle="-",
        linewidth=1.5,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    apply_axis_styling(ax2, "", xlim=(16, 20000))
    ax2.set_title("") # Remove title from bottom plot
    ax2.legend(loc="lower right")
    # Let Y-axis autoscale

    plt.tight_layout()
    save_figure(output_dir, "signal_response_multichannel.png")
    plt.close()


def generate_decomposition_plot(output_dir: str) -> None:
    """Generate time-domain decomposition plot comparing two filter types (Butterworth vs Chebyshev II)."""
    print("Generating signal_decomposition.png with comparison (Butter vs Cheby2) @ 48kHz...")
    fs = 48000
    duration = 0.5
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Signal: sum of 250Hz and 1000Hz sines
    y = np.sin(2 * np.pi * 250 * t) + np.sin(2 * np.pi * 1000 * t)

    # Filter into 1/1 octave bands with two different architectures
    # We use Chebyshev II (flat passband, no ripple)
    bank_butter = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[100.0, 2000.0], filter_type="butter")
    bank_cheby2 = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[100.0, 2000.0], filter_type="cheby2")
    
    # Cast to 3-tuple to satisfy mypy unpacking
    _, freq, xb_butter = bank_butter.filter(y, sigbands=True)
    
    _, _, xb_cheby2 = bank_cheby2.filter(y, sigbands=True)

    if xb_butter is None or xb_cheby2 is None:
        raise ValueError("Signal bands should not be None")

    num_plots = len(xb_butter) + 2 # +1 for original, +1 for impulse response
    _fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2.2 * num_plots), sharex=False)


    # Fixed Y limits for decomposition
    y_lim = (-2.8, 2.8)

    # 1. Original Signal
    axes[0].plot(t, y, color=COLOR_FG, linewidth=1.5)
    axes[0].set_title("Original Signal (250 Hz + 1000 Hz Sum) @ 48 kHz", fontweight="bold")
    axes[0].set_ylim(y_lim)
    axes[0].set_xlim(0, 0.04)

    # 2. Filtered Bands Comparison
    for i, (f_center) in enumerate(freq):
        axes[i + 1].plot(t, xb_butter[i], color=COLOR_PRIMARY, linewidth=1.5, label="Butterworth (Flat)")
        axes[i + 1].plot(t, xb_cheby2[i], color=COLOR_SECONDARY, linewidth=1.2, linestyle="--", alpha=0.9, label="Chebyshev II")
        axes[i + 1].set_title(f"Octave Band: {f_center:.0f} Hz", fontsize=11, fontweight="bold")
        axes[i + 1].set_ylim(y_lim)
        axes[i + 1].set_xlim(0, 0.04)
        if i == 0:
            axes[i+1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    # 3. Impulse Response (Stability/Transient Visualization)
    impulse = np.zeros(len(t))
    impulse[0] = 1.0
    _, _, ir_butter = bank_butter.filter(impulse, sigbands=True)
    _, _, ir_cheby2 = bank_cheby2.filter(impulse, sigbands=True)
    
    idx_1000 = np.argmin(np.abs(np.array(freq) - 1000))
    axes[-1].plot(t, ir_butter[idx_1000], color=COLOR_PRIMARY, linewidth=1.5, label="Butterworth")
    axes[-1].plot(t, ir_cheby2[idx_1000], color=COLOR_SECONDARY, linewidth=1.2, linestyle="--", alpha=0.9, label="Chebyshev II")
    axes[-1].set_title(f"Impulse Response ({freq[idx_1000]:.0f} Hz Band) - Transient/Stability Comparison", fontweight="bold")
    axes[-1].set_xlim(0, 0.04)
    axes[-1].set_xlabel("Time [s]")
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.8)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, which="both", alpha=0.4, linestyle=":")

    plt.tight_layout()
    save_figure(output_dir, "signal_decomposition.png")
    plt.close()



def measure_weighting_response(
    fs: int, curve: str, freqs: "np.ndarray | None" = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure a weighting curve the way the docs figure plots it: impulse
    response through the real filter path, evaluated with ``freqz`` (DTFT
    of the measured response).

    The impulse is placed at the CENTER of the buffer: the high-accuracy
    weighting path resamples with polyphase FIRs, and an impulse at sample 0
    loses the anti-causal half of the interpolation kernel (edge truncation),
    which once shipped a figure with curves ~2.4 dB low. Centering only adds
    linear phase, which does not affect the magnitude.

    :param fs: Sample rate in Hz.
    :param curve: 'A', 'B', 'C', 'D', 'G', 'AU' or 'Z'.
    :param freqs: Optional exact frequencies to evaluate; defaults to a
        dense 8192-point grid for plotting.
    :return: Tuple (frequencies, magnitude in dB).
    """
    from phonometry import weighting_filter

    impulse = np.zeros(fs)
    impulse[fs // 2] = 1.0
    weighted = weighting_filter(impulse, fs, curve=curve)

    worn = 8192 if freqs is None else np.asarray(freqs, dtype=float)
    w, h = scipy_signal.freqz(weighted, [1], worN=worn, fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-9)
    if freqs is None:
        # Drop the 0 Hz point: it cannot be drawn on a log frequency axis.
        return w[1:], mag_db[1:]
    return w, mag_db


def _draw_weighting_curves(
    ax: Any,
    fs: int,
    curves: tuple[tuple[str, str, str, str, float], ...],
    inset: Any = None,
) -> None:
    """Draw ``(code, label, colour, linestyle, linewidth)`` curves measured at *fs*.

    Shared by the two weighting-family figures (the IEC 61672-1 A/C/Z chart and
    the special B/D/AU chart), which differ only in the curves they hold and in
    the axis they hold them on. Curves are drawn in the order given, so a wide
    reference line listed first sits behind the curves that follow it.
    """
    for code, label, color, style, width in curves:
        # measure_weighting_response is covered by tests/test_graph_measurements.py
        w, mag_db = measure_weighting_response(fs, code)
        ax.semilogx(w, mag_db, label=label, color=color, linestyle=style, linewidth=width)
        if inset is not None:
            inset.plot(w, mag_db, color=color, linestyle=style, linewidth=width)
    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)


def generate_weighting_responses(output_dir: str) -> None:
    """Plot the IEC 61672-1 A/C/Z weighting frequency responses."""
    print("Generating weighting_responses.png...")
    fs = 48000

    _, ax = plt.subplots(figsize=(10, 7))

    # Zoom inset: the A curve is POSITIVE (+1.27 dB max at ~2.5 kHz per
    # IEC 61672-1 Table 2), invisible at the full -72..15 dB scale, so all
    # three curves of this figure are redrawn in it.
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax, width="42%", height="34%", loc="lower center", borderpad=2)
    axins.set_xscale("log")

    # The special B, D and AU curves have their own figure
    # (generate_special_weighting_responses); the infrasound G curve has
    # generate_g_weighting_response.
    curves = (
        ("A", "A-Weighting", COLOR_PRIMARY, "-", 1.8),
        ("C", "C-Weighting", COLOR_SECONDARY, "-", 1.8),
        ("Z", "Z-Weighting (Flat)", COLOR_FG, "--", 1.8),
    )
    _draw_weighting_curves(ax, fs, curves, inset=axins)

    # -72 dB, not the -50 the six-curve version used: with only A, C and Z left
    # nothing forces the floor up, and A reaches -70.4 dB at 10 Hz, so the whole
    # curve now fits on the axis instead of running off the bottom-left corner.
    apply_axis_styling(ax, "Frequency Weighting Curves (IEC 61672-1)",
                       xlim=(10, 22000), ylim=(-72, 15))

    axins.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.4, linewidth=1)
    axins.set_xlim(500, 8000)
    axins.set_ylim(-3, 2)
    axins.grid(True, which="both", alpha=0.3)
    axins.set_title("Zoom: A-weighting is positive (max +1.27 dB @ 2.5 kHz)", fontsize=9)
    axins.annotate(
        "+1.27 dB", xy=(2500, 1.27), xytext=(4200, 1.55), fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    axins.xaxis.set_major_formatter(ScalarFormatter())
    axins.xaxis.set_minor_formatter(NullFormatter())
    axins.set_xticks([500, 1000, 2500, 5000, 8000])
    axins.set_xticklabels(["500", "1k", "2.5k", "5k", "8k"], fontsize=8)

    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "weighting_responses.png")
    plt.close()


def generate_special_weighting_responses(output_dir: str) -> None:
    """Plot the special B/D/AU weighting curves against the A reference."""
    print("Generating special_weighting_responses.png...")
    # 96 kHz, not the 48 kHz of the IEC 61672-1 figure: the point of AU is the
    # IEC 61012 U low-pass, specified up to 40 kHz, and a 48 kHz axis would cut
    # it off mid-slope (the guide says the same about measuring it).
    fs = 96000

    _, ax = plt.subplots(figsize=(10, 7))

    # A is drawn first, as a wide pale line, so it reads as the reference the
    # other three are described against: AU runs inside it up to 10 kHz and
    # then leaves, B stays above it, D crosses it at its 3.15 kHz hump.
    curves = (
        ("A", "A-Weighting (reference)", COLOR_MUTED, "-", 4.0),
        ("B", "B-Weighting (historical)", COLOR_TERTIARY, "--", 1.8),
        ("D", "D-Weighting (aircraft, withdrawn)", "#9467bd", "-.", 1.8),
        ("AU", "AU-Weighting (audible + ultrasound)", "#ff7f0e", "-", 1.8),
    )
    _draw_weighting_curves(ax, fs, curves)

    apply_axis_styling(ax, "Special Weighting Curves (B, D, AU)",
                       xlim=(10, 40000), ylim=(-90, 18))
    # apply_axis_styling stops labelling at 16 kHz; this axis runs into the
    # ultrasonic decade AU exists for, so it needs the 40 kHz end labelled.
    ax.set_xticks([16, 63, 250, 1000, 4000, 16000, 40000])
    ax.set_xticklabels(["16", "63", "250", "1k", "4k", "16k", "40k"])

    # The two numbers the guide quotes from the standards: the D hump of
    # IEC 537 (+11.5 dB at 3.15 kHz, NASA CR-3406 Table SLD-I) and the U
    # low-pass of IEC 61012 (-13 dB at 16 kHz, Table 1).
    ax.annotate(
        "+11.5 dB @ 3.15 kHz", xy=(3150, 11.6), xytext=(150, 13.5), fontsize=9,
        color="#9467bd", arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#9467bd"},
    )
    ax.annotate(
        "AU is 13 dB below A at 16 kHz", xy=(16000, -21.0), xytext=(700, -35.0),
        fontsize=9, color="#ff7f0e",
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#ff7f0e"},
    )

    ax.legend(loc="lower center", fontsize=9)
    save_figure(output_dir, "special_weighting_responses.png")
    plt.close()


def generate_g_weighting_response(output_dir: str) -> None:
    """Plot the ISO 7196 G-weighting curve against the Table 2 nominals."""
    print("Generating g_weighting_response.png...")
    from scipy import signal as sp_signal

    from phonometry import WeightingFilter

    fs = 48000
    # ISO 7196:1995 Table 2 - nominal one-third-octave frequency, response dB
    table2 = [
        (0.25, -88.0), (0.5, -64.3), (1.0, -43.0), (2.0, -28.3),
        (4.0, -16.0), (8.0, -4.0), (10.0, 0.0), (16.0, 7.7), (20.0, 9.0),
        (31.5, -4.0), (63.0, -28.0), (125.0, -52.0), (250.0, -76.0),
    ]
    freqs = np.logspace(np.log10(0.1), np.log10(1000), 800)
    sos = WeightingFilter(fs, "G").sos
    _, h = sp_signal.sosfreqz(sos, worN=freqs, fs=fs)
    mag_db = 20 * np.log10(np.abs(h))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, mag_db, color=COLOR_PRIMARY, label="G-weighting (ISO 7196)")
    tf = [f for f, _ in table2]
    tv = [v for _, v in table2]
    ax.plot(tf, tv, "o", color=COLOR_SECONDARY, markersize=5,
            label="ISO 7196 Table 2 nominals", zorder=5)
    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.axvline(10, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    ax.annotate("0 dB @ 10 Hz", xy=(10, 0), xytext=(20, -18), fontsize=9,
                arrowprops={"arrowstyle": "->", "lw": 0.8})
    apply_axis_styling(
        ax, "G Frequency Weighting for Infrasound (ISO 7196:1995)",
        xlim=(0.1, 1000), ylim=(-95, 15),
    )
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 125, 315, 1000]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "50", "125", "315", "1k"])
    ax.legend(loc="upper right")
    save_figure(output_dir, "g_weighting_response.png")
    plt.close()


def generate_equal_loudness_contours(output_dir: str) -> None:
    """Plot the ISO 226:2023 normal equal-loudness-level contours."""
    print("Generating equal_loudness_contours.png...")
    from phonometry import equal_loudness_contours

    _, ax = plt.subplots(figsize=(10, 7))
    # The result's own .plot() draws the contour family plus the hearing
    # threshold on a 1k/2k-labelled log frequency axis (ISO 226:2023 Formula 1).
    equal_loudness_contours().plot(ax=ax)
    ax.set_ylim(-10, 130)
    ax.set_title("Normal Equal-Loudness-Level Contours (ISO 226:2023)",
                 fontweight="bold", pad=12)
    save_figure(output_dir, "equal_loudness_contours.png")
    plt.close()


def generate_time_weighting_plot(output_dir: str) -> None:
    """Visualize Fast, Slow and Impulse time weighting response to a burst."""
    print("Generating time_weighting_analysis.png...")
    fs = 1000
    t = np.linspace(0, 4, fs * 4, endpoint=False)
    
    # 500ms burst of noise starting at 1.0s
    rng = np.random.default_rng(42)
    x = np.zeros_like(t)
    start_idx = int(fs * 1.0)
    end_idx = int(fs * 1.5)
    x[start_idx:end_idx] = rng.standard_normal(end_idx - start_idx)
    
    from phonometry import time_weighting
    
    # Square for energy
    x_sq = x**2
    fast = time_weighting(x, fs, mode="fast")
    slow = time_weighting(x, fs, mode="slow")
    impulse = time_weighting(x, fs, mode="impulse")
    
    _, ax = plt.subplots()
    # Normalize for better visualization
    # We normalized x_sq to peak at 1 for the plot
    peak = np.max(x_sq)
    x_sq /= peak
    fast /= peak
    slow /= peak
    impulse /= peak
    
    ax.plot(t, x_sq, color="#9e9e9e", alpha=0.6, label="Input Burst (Normalized)")
    ax.plot(t, fast, color=COLOR_PRIMARY, label="Fast (125ms)")
    ax.plot(t, slow, color=COLOR_SECONDARY, label="Slow (1000ms)")
    ax.plot(t, impulse, color="purple", linestyle="-.", linewidth=1.5, label="Impulse (35ms/1.5s)")
    
    ax.set_title("Time Weighting Ballistics (IEC 61672-1)", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Response")
    ax.legend(loc="upper right")
    ax.set_xlim(0.8, 3.5)
    save_figure(output_dir, "time_weighting_analysis.png")
    plt.close()


def generate_crossover_plot(output_dir: str) -> None:
    """Visualize Linkwitz-Riley 4th Order Crossover."""
    print("Generating crossover_lr4.png...")
    fs = 48000
    
    from phonometry import linkwitz_riley
    
    # Frequency analysis
    # Measure response using IR
    impulse = np.zeros(fs)
    impulse[0] = 1.0
    lp_ir, hp_ir = linkwitz_riley(impulse, fs, freq=1000, order=4)
    
    w, h_lp = scipy_signal.freqz(lp_ir, worN=8192, fs=fs)
    _, h_hp = scipy_signal.freqz(hp_ir, worN=8192, fs=fs)
    
    _, ax = plt.subplots()
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp) + 1e-9), color=COLOR_PRIMARY, label="Low Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_hp) + 1e-9), color=COLOR_SECONDARY, label="High Pass (LR4)")
    ax.semilogx(w, 20 * np.log10(np.abs(h_lp + h_hp) + 1e-9), color=COLOR_FG, linestyle="--", label="Sum (Flat)")

    apply_axis_styling(ax, "Linkwitz-Riley Crossover (4th Order @ 1kHz)", xlim=(20, 20000), ylim=(-60, 5))
    ax.legend(loc="lower right")
    save_figure(output_dir, "crossover_lr4.png")
    plt.close()


def generate_parametric_eq_family(output_dir: str) -> None:
    """Magnitude responses of the RBJ Audio EQ Cookbook biquad family."""
    print("Generating parametric_eq_family.png...")
    fs = 48000

    from phonometry import EQSection, ParametricEQ

    family = [
        (EQSection("peaking", 1000.0, gain_db=6.0, q=1.4),
         "Peaking +6 dB (Q = 1.4)", COLOR_PRIMARY, "-"),
        (EQSection("lowshelf", 125.0, gain_db=6.0),
         "Low shelf +6 dB", COLOR_TERTIARY, "-"),
        (EQSection("highshelf", 4000.0, gain_db=-6.0),
         "High shelf -6 dB", "#9467bd", "-"),
        (EQSection("lowpass", 10000.0),
         "Low-pass (Q = 0.707)", COLOR_SECONDARY, "--"),
        (EQSection("highpass", 50.0),
         "High-pass (Q = 0.707)", "#8c564b", "--"),
        (EQSection("bandpass", 500.0, q=2.0),
         "Band-pass (Q = 2)", "#ff7f0e", "-."),
        (EQSection("notch", 2000.0, q=6.0),
         "Notch (Q = 6)", "#17becf", "-."),
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    for section, label, color, style in family:
        res = ParametricEQ(fs, section).response(f_min=20.0, f_max=20000.0)
        ax.semilogx(res.frequencies, res.magnitude_db,
                    label=label, color=color, linestyle=style)

    ax.axhline(0, color=COLOR_FG, linestyle=":", alpha=0.3, linewidth=1)
    apply_axis_styling(ax, "Parametric EQ Biquads (RBJ Audio EQ Cookbook)",
                       xlim=(20, 20000), ylim=(-27, 9))
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.set_ylabel("Magnitude [dB]")
    ax.legend(loc="lower center", fontsize=9, ncols=2)
    save_figure(output_dir, "parametric_eq_family.png")
    plt.close()


def generate_spectrogram_example(output_dir: str) -> None:
    """Visualize OctaveFilterBank.spectrogram on a time-varying signal."""
    print("Generating spectrogram_example.png...")
    fs = 48000
    duration = 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Log sweep + two tone bursts to show time-frequency localization
    rng = np.random.default_rng(42)
    x = 0.5 * scipy_signal.chirp(t, f0=80, t1=duration, f1=8000, method="logarithmic")
    x[int(1.0 * fs):int(1.3 * fs)] += np.sin(2 * np.pi * 4000 * t[int(1.0 * fs):int(1.3 * fs)])
    x[int(2.5 * fs):int(2.8 * fs)] += np.sin(2 * np.pi * 250 * t[int(2.5 * fs):int(2.8 * fs)])
    x += 0.01 * rng.standard_normal(len(t))

    # 1/12-octave bands stepped at 1/8 of the window: the Fast (125 ms)
    # integration still sets the time resolution, but the mesh is sampled four
    # times finer on each axis than the band spacing and hop it replaces, so
    # the sweep reads as a line instead of a staircase of cells.
    bank = OctaveFilterBank(fs=fs, fraction=12, order=6, limits=[50.0, 12000.0])
    levels, freq, times = bank.spectrogram(x, window_time=0.125, overlap=0.875)

    _, ax = plt.subplots()
    mesh = ax.pcolormesh(times, freq, levels, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_title("1/12 Octave Spectrogram (Fast windows, 87.5% overlap)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_FREQ_HZ)
    yticks = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    ax.set_yticks(yticks)
    ax.set_yticklabels(["63", "125", "250", "500", "1k", "2k", "4k", "8k"])
    plt.colorbar(mesh, ax=ax, label=LABEL_LEVEL_DB)
    save_figure(output_dir, "spectrogram_example.png")
    plt.close()


def generate_ln_levels_example(output_dir: str) -> None:
    """Visualize statistical LN levels over the Fast envelope."""
    print("Generating ln_levels_example.png...")
    fs = 8000
    duration = 30.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    from phonometry import ln_levels, time_weighting

    # Fluctuating "traffic-like" noise: background + random events
    rng = np.random.default_rng(42)
    x = 0.05 * rng.standard_normal(len(t))
    for _ in range(12):
        start = rng.uniform(1, duration - 3)
        length = rng.uniform(0.5, 2.0)
        idx = (t >= start) & (t < start + length)
        envelope = np.hanning(int(idx.sum()))
        x[idx] += envelope * rng.uniform(0.3, 1.0) * rng.standard_normal(int(idx.sum()))

    envelope_ms = time_weighting(x, fs, mode="fast")
    level_t = 10 * np.log10(np.maximum(envelope_ms, 1e-12) / (2e-5) ** 2)
    stats = ln_levels(x, fs, n=(10, 50, 90))

    _, ax = plt.subplots()
    ax.plot(t, level_t, color=COLOR_PRIMARY, linewidth=0.8, label="Fast level $L_p(t)$")
    for n_value, color, style in [(10, COLOR_SECONDARY, "--"), (50, COLOR_FG, "-"), (90, COLOR_TERTIARY, "-.")]:
        ax.axhline(
            float(stats[n_value]), color=color, linestyle=style, linewidth=1.5,
            label=f"L{n_value} = {float(stats[n_value]):.1f} dB",
        )
    ax.set_title("Statistical Levels L10 / L50 / L90 (Fast envelope)", fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.set_xlim(0, duration)
    ax.legend(loc="lower right")
    save_figure(output_dir, "ln_levels_example.png")
    plt.close()


def generate_zero_phase_comparison(output_dir: str) -> None:
    """Compare causal vs zero-phase band filtering of a tone burst."""
    print("Generating zero_phase_comparison.png...")
    fs = 48000
    duration = 0.15
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 250 Hz tone burst in the middle of the frame
    x = np.zeros_like(t)
    start, end = int(0.05 * fs), int(0.10 * fs)
    x[start:end] = np.sin(2 * np.pi * 250 * t[start:end]) * np.hanning(end - start)

    bank = OctaveFilterBank(fs=fs, fraction=1, order=6, limits=[200.0, 300.0])
    _, _, bands_fwd = bank.filter(x, sigbands=True, calculate_level=False)
    _, _, bands_zp = bank.filter(x, sigbands=True, calculate_level=False, zero_phase=True)

    _, ax = plt.subplots()
    ax.plot(t, x, color="gray", alpha=0.5, linewidth=1.0, label="Input burst (250 Hz)")
    ax.plot(t, bands_fwd[0], color=COLOR_PRIMARY, linewidth=1.3, label="Causal filtering (group delay)")
    ax.plot(t, bands_zp[0], color=COLOR_SECONDARY, linewidth=1.3, linestyle="--", label="zero_phase=True (aligned)")
    ax.set_title("Zero-Phase Filtering: Group Delay Elimination (250 Hz Band)", fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    save_figure(output_dir, "zero_phase_comparison.png")
    plt.close()


def generate_weighting_accuracy_hf(output_dir: str) -> None:
    """Compare A-weighting HF accuracy: analytic vs bilinear vs high_accuracy."""
    print("Generating weighting_accuracy_hf.png...")
    fs = 48000

    from phonometry import WeightingFilter

    freqs = np.logspace(np.log10(1000), np.log10(20000), 40)

    def analytic_a(f: np.ndarray) -> np.ndarray:
        ra = (12194**2 * f**4) / (
            (f**2 + 20.6**2)
            * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
            * (f**2 + 12194**2)
        )
        return np.asarray(20 * np.log10(ra) + 2.0)

    def measured_gains(wf: WeightingFilter) -> np.ndarray:
        gains = []
        for f0 in freqs:
            tt = np.arange(int(fs * 0.2)) / fs
            x = np.sin(2 * np.pi * f0 * tt)
            y = wf.filter(x)
            n0 = int(0.05 * fs)  # skip filter transient
            gains.append(20 * np.log10(np.std(y[n0:]) / np.std(x[n0:])))
        return np.array(gains)

    legacy = measured_gains(WeightingFilter(fs, "A", high_accuracy=False))
    accurate = measured_gains(WeightingFilter(fs, "A"))
    reference = analytic_a(freqs)

    _, (ax, ax_err) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax.semilogx(freqs, reference, color=COLOR_FG, linewidth=2, label="IEC 61672-1 analytic curve")
    ax.semilogx(freqs, legacy, color=COLOR_SECONDARY, linestyle="--", label="Plain bilinear (high_accuracy=False)")
    ax.semilogx(freqs, accurate, color=COLOR_PRIMARY, linestyle="-.", label="Oversampled (high_accuracy=True)")
    ax.set_title(f"A-Weighting High-Frequency Accuracy @ fs={fs//1000} kHz", fontweight="bold", pad=12)
    ax.set_ylabel(LABEL_LEVEL_DB)
    ax.legend(loc="lower left")

    ax_err.semilogx(freqs, legacy - reference, color=COLOR_SECONDARY, linestyle="--", label="Bilinear error")
    ax_err.semilogx(freqs, accurate - reference, color=COLOR_PRIMARY, linestyle="-.", label="high_accuracy error")
    ax_err.axhline(-2.5, color="gray", linestyle=":", label="Class 1 lower limit @ 12.5 kHz")
    ax_err.set_ylabel("Error [dB]")
    ax_err.set_xlabel(LABEL_FREQ_HZ)
    ax_err.set_ylim(-8, 2)
    ax_err.legend(loc="lower left")

    for a in (ax, ax_err):
        xticks = [1000, 2000, 4000, 8000, 12500, 16000, 20000]
        a.set_xticks(xticks)
        a.set_xticklabels(["1k", "2k", "4k", "8k", "12.5k", "16k", "20k"])

    plt.tight_layout()
    save_figure(output_dir, "weighting_accuracy_hf.png")
    plt.close()


def generate_group_delay_comparison(output_dir: str) -> None:
    """Group delay of the 1 kHz band for every architecture (docs: filter-banks)."""
    print("Generating group_delay_comparison.png...")
    fs = 48000
    limits = [800.0, 1200.0]

    filters = [
        ("butter", "Butterworth", COLOR_PRIMARY, "-"),
        ("cheby1", "Chebyshev I", COLOR_SECONDARY, "--"),
        ("cheby2", "Chebyshev II", COLOR_TERTIARY, ":"),
        ("ellip", "Elliptic", "#9467bd", "-."),
        ("bessel", "Bessel", "#8c564b", "-"),
    ]

    _, ax = plt.subplots()
    for f_type, label, color, style in filters:
        bank = OctaveFilterBank(fs, fraction=1, order=6, limits=limits, filter_type=f_type)
        idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
        fsd = fs / bank.factor[idx]
        # Group delay of an SOS cascade = sum of the sections' group delays.
        w = np.logspace(np.log10(500), np.log10(2000), 1024)
        gd = np.zeros_like(w)
        for section in bank.sos[idx]:
            _w_s, gd_s = scipy_signal.group_delay((section[:3], section[3:]), w=w, fs=fsd)
            gd += gd_s
        ax.semilogx(w, gd / fsd * 1000, label=label, color=color, linestyle=style)

    ax.set_title("Group Delay Comparison (1 kHz Octave Band, Order 6)", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Group delay [ms]")
    ax.set_xlim(500, 2000)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([500, 707, 1000, 1414, 2000])
    ax.set_xticklabels(["500", "707", "1k", "1.41k", "2k"])
    ax.legend(loc="upper right")
    save_figure(output_dir, "group_delay_comparison.png")
    plt.close()


def generate_tone_burst_iec(output_dir: str) -> None:
    """FAST envelope response to 4 kHz tonebursts vs IEC 61672-1 Table 4 targets."""
    print("Generating tone_burst_iec.png...")
    fs = 48000

    from phonometry import time_weighting

    cases = [(0.2, -1.0), (0.05, -4.8), (0.01, -11.1)]  # Table 4, class 1 rows
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    t_all = np.arange(int(fs * 2.0)) / fs
    steady = np.sin(2 * np.pi * 4000 * t_all)
    ref = time_weighting(steady, fs, mode="fast")[int(1.5 * fs):].mean()

    for ax, (duration, target) in zip(axes, cases):
        burst = np.zeros_like(t_all)
        start = int(0.5 * fs)
        burst[start:start + round(duration * fs)] = steady[start:start + round(duration * fs)]
        env_db = 10 * np.log10(np.maximum(time_weighting(burst, fs, mode="fast") / ref, 1e-6))

        ax.plot(t_all, env_db, color=COLOR_PRIMARY, linewidth=1.3, label="FAST envelope")
        ax.axhline(target, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2,
                   label=f"IEC target {target} dB")
        ax.set_title(f"{duration * 1000:g} ms burst", fontsize=11, fontweight="bold")
        ax.set_xlim(0.4, 1.4)
        ax.set_ylim(-30, 3)
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("Level re steady state [dB]")

    fig.suptitle("4 kHz Toneburst Response vs IEC 61672-1 Table 4 (FAST)", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "tone_burst_iec.png")
    plt.close()


def generate_block_processing_continuity(output_dir: str) -> None:
    """Stateful vs stateless block processing (docs: block-processing)."""
    print("Generating block_processing_continuity.png...")
    fs = 8000
    n_blocks, block = 4, 1000
    rng = np.random.default_rng(42)
    x = rng.standard_normal(n_blocks * block)
    t = np.arange(len(x)) / fs

    def band_output(stateful: bool) -> np.ndarray:
        bank = OctaveFilterBank(fs, fraction=1, limits=[900, 1100],
                                stateful=stateful, resample=False)
        if stateful:
            parts = [
                bank.filter(x[i * block:(i + 1) * block], sigbands=True,
                            detrend=False, calculate_level=False)[2][0]
                for i in range(n_blocks)
            ]
        else:
            parts = []
            for i in range(n_blocks):
                b2 = OctaveFilterBank(fs, fraction=1, limits=[900, 1100], resample=False)
                parts.append(b2.filter(x[i * block:(i + 1) * block], sigbands=True,
                                       detrend=False, calculate_level=False)[2][0])
        return np.concatenate(parts)

    continuous = OctaveFilterBank(fs, fraction=1, limits=[900, 1100], resample=False).filter(
        x, sigbands=True, detrend=False, calculate_level=False)[2][0]
    y_stateful = band_output(stateful=True)
    y_stateless = band_output(stateful=False)

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    zoom = slice(int(0.9 * block), int(1.4 * block))  # around the first boundary

    ax1.plot(t[zoom], continuous[zoom], color=COLOR_FG, linewidth=2.2, alpha=0.35,
             label="Continuous (whole signal)")
    ax1.plot(t[zoom], y_stateful[zoom], color=COLOR_PRIMARY, linewidth=1.1,
             label="Stateful blocks (state carried)")
    ax1.set_title("stateful=True: block outputs equal the continuous result",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(t[zoom], continuous[zoom], color=COLOR_FG, linewidth=2.2, alpha=0.35,
             label="Continuous (whole signal)")
    ax2.plot(t[zoom], y_stateless[zoom], color=COLOR_SECONDARY, linewidth=1.1,
             label="Independent blocks (state reset)")
    ax2.axvline(block / fs, color=COLOR_FG, linestyle=":", alpha=0.6)
    # The callout lands on top of the dense trace, so it carries a solid
    # panel of its own instead of relying on the gaps between the waveforms.
    ax2.annotate("block boundary:\nfilter transient restarts", xy=(block / fs, 0),
                 xytext=(block / fs + 0.02, ax2.get_ylim()[0] * 0.55 if ax2.get_ylim()[0] < 0 else -1),
                 fontsize=9.5, color=COLOR_FG, zorder=6,
                 bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                       "edgecolor": COLOR_GRID},
                 arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax2.set_title("No state: each block restarts the filter transient",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time [s]")
    ax2.legend(loc="upper right", fontsize=9)
    for a in (ax1, ax2):
        a.set_ylabel("Amplitude")

    plt.tight_layout()
    save_figure(output_dir, "block_processing_continuity.png")
    plt.close()


def generate_class_mask_overlay(output_dir: str) -> None:
    """Band response against the IEC 61260-1:2014 class limit mask."""
    print("Generating class_mask_overlay.png...")
    fs = 48000

    from phonometry.metrology.compliance import class_limits

    bank = OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200], filter_type="butter")
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2 ** 15, fs=fsd)
    attenuation = -20 * np.log10(np.abs(h) + 1e-12)
    a_ref = float(np.interp(fm, w, attenuation))
    omega = w / fm
    valid = (omega > 0.05) & (omega < 8)
    omega, delta_a = omega[valid], (attenuation - a_ref)[valid]

    grid = np.logspace(np.log10(0.05), np.log10(8), 2000)
    lo1, hi1 = class_limits(1.0, 1, grid)
    lo2, _ = class_limits(1.0, 2, grid)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Forbidden regions for class 1: below the minimum required attenuation
    # (stop band) and above the maximum allowed attenuation (pass band).
    ax.fill_between(grid, -10, lo1, color=theme_fill(COLOR_SECONDARY, ax),
                    zorder=0,
                    label="Forbidden for class 1 (too little attenuation)")
    finite = np.isfinite(hi1)
    ax.fill_between(grid[finite], hi1[finite], 90,
                    color=theme_fill("#9467bd", ax), zorder=0,
                    label="Forbidden for class 1 (too much attenuation)")
    ax.plot(grid, lo2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.2,
            label="Class 2 minimum attenuation")

    ax.plot(omega, delta_a, color=COLOR_PRIMARY, linewidth=1.6,
            label="Butterworth order 6 (1 kHz octave band)")

    ax.set_xscale("log")
    ax.set_xlim(0.08, 8)
    ax.set_ylim(-6, 90)
    ax.set_title("Relative Attenuation vs IEC 61260-1:2014 Class Limits", fontweight="bold", pad=12)
    ax.set_xlabel("Normalized frequency  f / fm")
    ax.set_ylabel("Relative attenuation ΔA [dB]")
    ax.set_xticks([0.125, 0.25, 0.5, 0.707, 1, 1.414, 2, 4, 8])
    ax.set_xticklabels(["0.125", "0.25", "0.5", "0.707", "1", "1.41", "2", "4", "8"])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "class_mask_overlay.png")
    plt.close()


def generate_filter_class0_mask(output_dir: str) -> None:
    """Pass-band class 0/1/2 maximum corridors (IEC 61260:1995 / ANSI S1.11-2004)."""
    print("Generating filter_class0_mask...")
    fs = 48000
    from phonometry.metrology.compliance import class_limits

    bank = OctaveFilterBank(fs, fraction=1, order=6, limits=[800, 1200], filter_type="butter")
    idx = int(np.argmin(np.abs(np.array(bank.freq) - 1000)))
    fm = bank.freq[idx]
    fsd = fs / bank.factor[idx]
    w, h = scipy_signal.sosfreqz(bank.sos[idx], worN=2 ** 15, fs=fsd)
    attenuation = -20 * np.log10(np.abs(h) + 1e-12)
    a_ref = float(np.interp(fm, w, attenuation))
    omega = w / fm

    # Restrict to the pass-band [G**-1/2, G**+1/2] where a finite max applies
    # (beyond the band edges the maximum limit is +inf, so plotting there would
    # misleadingly show the filter's natural roll-off "exceeding" a corridor).
    g_octave = 10 ** (3 / 10)  # octave ratio G (IEC 61260)
    edge_lo, edge_hi = g_octave ** -0.5, g_octave ** 0.5
    pb = (omega >= edge_lo) & (omega <= edge_hi)
    omega, delta_a = omega[pb], (attenuation - a_ref)[pb]
    grid = np.linspace(edge_lo, edge_hi, 1500)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Nested min/max corridors: class 0 (+-0.15 dB reference) is the tightest.
    for cls, colour, name in ((2, COLOR_TERTIARY, "Class 2 corridor"),
                              (1, COLOR_SECONDARY, "Class 1 corridor"),
                              (0, COLOR_PRIMARY, "Class 0 corridor")):
        lo, hi = class_limits(1.0, cls, grid, edition="1995")
        ax.plot(grid, hi, color=colour, linewidth=1.4, label=name)
        ax.plot(grid, lo, color=colour, linewidth=1.4)
    ax.plot(omega, delta_a, color=COLOR_FG, linewidth=2.2,
            label="Butterworth order 6 (1 kHz octave band)")

    ax.set_xscale("log")
    ax.set_xlim(edge_lo, edge_hi)
    ax.set_ylim(-0.7, 6)
    ax.set_title("Pass-band Class 0/1/2 Limits (IEC 61260:1995 / ANSI S1.11-2004)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Normalized frequency  f / fm")
    ax.set_ylabel("Relative attenuation ΔA [dB]")
    ax.set_xticks([0.707, 0.841, 1, 1.189, 1.414])
    ax.set_xticklabels(["0.707", "0.841", "1", "1.189", "1.414"])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())  # keep only explicit ticks
    ax.grid(which="major", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", fontsize=9)
    save_figure(output_dir, "filter_class0_mask.png")
    plt.close()


def generate_weighting_class_mask(output_dir: str) -> None:
    """A/C weighting deviation against the IEC 61672-1:2013 Table 3 mask."""
    print("Generating weighting_class_mask.png...")
    from phonometry import (
        WeightingFilter,
        verify_weighting_class,
        weighting_class_limits,
    )

    freqs, lower1, upper1 = weighting_class_limits(1)
    _, lower2, upper2 = weighting_class_limits(2)
    floor, ceil = -7.0, 7.0  # plotting bounds; -inf limits clip to the floor
    lo1 = np.clip(lower1, floor, ceil)
    lo2 = np.clip(lower2, floor, ceil)

    _, ax = plt.subplots(figsize=(10, 6.5))
    # Allowed corridor for class 1 (between lower and upper limit).
    ax.fill_between(freqs, lo1, upper1, color=theme_fill(COLOR_PRIMARY, ax),
                    step="mid", label="Class 1 acceptance region")
    ax.plot(freqs, upper1, color=COLOR_SECONDARY, linewidth=1.3, drawstyle="steps-mid",
            label="Class 1 upper/lower limit")
    ax.plot(freqs, lo1, color=COLOR_SECONDARY, linewidth=1.3, drawstyle="steps-mid")
    ax.plot(freqs, upper2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.1,
            drawstyle="steps-mid", label="Class 2 upper/lower limit")
    ax.plot(freqs, lo2, color=COLOR_TERTIARY, linestyle=":", linewidth=1.1,
            drawstyle="steps-mid")

    for curve, colour, marker in (("A", COLOR_PRIMARY, "o"), ("C", "#9467bd", "s")):
        result = verify_weighting_class(WeightingFilter(48000, curve))
        f = np.array([b["freq"] for b in result["bands"]])
        dev = np.array([b["deviation_db"] for b in result["bands"]])
        ax.plot(f, dev, color=colour, linewidth=1.6, marker=marker, markersize=4,
                label=f"{curve} weighting deviation (48 kHz)")

    ax.set_xscale("log")
    ax.set_xlim(10, 20000)
    ax.set_ylim(floor, ceil)
    format_frequency_axis(ax, 10, 20000)
    ax.set_title("Weighting Deviation vs IEC 61672-1:2013 Table 3 Limits",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Deviation from design goal [dB]")
    ax.grid(which="both", color=COLOR_GRID, linestyle=":", alpha=0.4)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    save_figure(output_dir, "weighting_class_mask.png")
    plt.close()


def generate_calibration_stability(output_dir: str) -> None:
    """Stable vs unstable calibration tone against the IEC 60942 limit."""
    print("Generating calibration_stability.png...")
    from phonometry import time_weighting

    fs = 48000
    seconds = 6.0
    tt = np.arange(int(fs * seconds)) / fs
    stable = 0.5 * np.sin(2 * np.pi * 1000 * tt)
    # 3 % amplitude modulation at 2 Hz: ~0.14 dB deviation, clearly over
    unstable = 0.5 * (1 + 0.03 * np.sin(2 * np.pi * 2.0 * tt)) * np.sin(2 * np.pi * 1000 * tt)

    _, ax = plt.subplots(figsize=(10, 6))
    skip = fs  # discard the F-integrator attack (~8*tau = 1 s)
    for x, color, label in [
        (stable, COLOR_PRIMARY, "Stable tone (good coupling)"),
        (unstable, COLOR_SECONDARY, "3% AM tone (loose coupling)"),
    ]:
        env = time_weighting(x, fs, mode="fast")[skip:]
        level = 10 * np.log10(np.maximum(env, np.finfo(float).eps))
        rel = level - np.mean(level)
        ax.plot(tt[skip:], rel, color=color, linewidth=1.4, label=label)

    ax.axhline(0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(-0.07, color=COLOR_FG, linestyle="--", linewidth=1.2, alpha=0.7,
               label="IEC 60942:2017 class 1 limit (deviation from mean)")
    ax.fill_between([1, seconds], -0.07, 0.07, color=COLOR_PRIMARY, alpha=0.06)
    ax.set_title("Calibration Tone Stability Check (IEC 60942:2017, 5.3.3)",
                 fontweight="bold", pad=12)
    ax.set_xlim(1, seconds)
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("F-weighted level re mean [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "calibration_stability.png")
    plt.close()


def generate_sel_concept(output_dir: str) -> None:
    """SEL: the whole event compressed into one second of equal energy."""
    print("Generating sel_concept.png...")
    from phonometry import leq, sel, time_weighting

    fs = 48000
    seconds = 8.0
    tt = np.arange(int(fs * seconds)) / fs
    rng = np.random.default_rng(11)
    # A vehicle pass-by: noise with a gaussian energy envelope
    envelope = np.exp(-0.5 * ((tt - 4.0) / 1.1) ** 2)
    x = envelope * rng.standard_normal(tt.size) * 0.3

    env = time_weighting(x, fs, mode="fast")
    level = 10 * np.log10(np.maximum(env, 1e-12))
    l_sel = float(sel(x, fs, dbfs=True))
    l_eq = float(leq(x, fs, dbfs=True))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tt, level, color=COLOR_PRIMARY, linewidth=1.2,
            label="Fast level of the event")
    ax.hlines(l_eq, 0, seconds, color=COLOR_TERTIARY, linestyle="--",
              linewidth=1.6, label="Leq over the whole event")
    # SEL: same energy squeezed into 1 s (drawn as a 1 s block)
    ax.fill_between([3.5, 4.5], -55, l_sel, color=COLOR_SECONDARY, alpha=0.25)
    ax.hlines(l_sel, 3.5, 4.5, color=COLOR_SECONDARY, linewidth=2.2,
              label="SEL: same energy in 1 s")
    ax.annotate("equal energy", xy=(4.5, l_sel - 3), xytext=(5.6, l_sel - 1),
                fontsize=10, arrowprops={"arrowstyle": "->", "lw": 0.9})
    ax.set_title("Sound Exposure Level: the event normalized to 1 s",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, seconds)
    ax.set_ylim(-55, l_sel + 6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [dBFS]")
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "sel_concept.png")
    plt.close()


def generate_lden_profile(output_dir: str) -> None:
    """A 24 h urban level profile with the Lden period weightings."""
    print("Generating lden_profile.png...")
    from phonometry import lden

    hours = np.arange(24)
    # Typical urban road profile (synthetic hourly LAeq, dB)
    laeq_h = np.array([48, 46, 45, 45, 46, 50, 56, 64, 66, 65, 63, 63,
                       64, 63, 63, 64, 65, 66, 65, 64, 63, 62, 61, 50],
                      dtype=float)

    def _period_leq(idx: "np.ndarray") -> float:
        return float(10 * np.log10(np.mean(10 ** (0.1 * laeq_h[idx]))))

    ld = _period_leq(np.arange(7, 19))    # day 07-19
    le = _period_leq(np.arange(19, 23))   # evening 19-23
    ln_ = _period_leq(np.r_[np.arange(23, 24), np.arange(0, 7)])  # night 23-07
    l_den = lden(ld, le, ln_)

    _, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(7, 19, color=theme_fill(COLOR_TERTIARY, ax))
    ax.axvspan(19, 23, color=theme_fill("#e8a838", ax))
    ax.axvspan(23, 24, color=theme_fill(COLOR_PRIMARY, ax))
    ax.axvspan(0, 7, color=theme_fill(COLOR_PRIMARY, ax))
    ax.step(np.r_[hours, 24], np.r_[laeq_h, laeq_h[-1]], where="post",
            color=COLOR_FG, linewidth=1.6, label="Hourly LAeq")
    ax.hlines(ld, 7, 19, color=COLOR_TERTIARY, linestyle="--", linewidth=2,
              label="Lday (+0 dB)")
    ax.hlines(le + 5, 19, 23, color="#e8a838", linestyle="--", linewidth=2,
              label="Levening + 5 dB")
    ax.hlines(ln_ + 10, 23, 24, color=COLOR_PRIMARY, linestyle="--", linewidth=2)
    ax.hlines(ln_ + 10, 0, 7, color=COLOR_PRIMARY, linestyle="--", linewidth=2,
              label="Lnight + 10 dB")
    ax.hlines(l_den, 0, 24, color=COLOR_SECONDARY, linewidth=2.4,
              label=f"Lden = {l_den:.1f} dB")
    ax.set_title("Day-Evening-Night Level Lden (ISO 1996-1)",
                 fontweight="bold", pad=12)
    ax.set_xlim(0, 24)
    ax.set_ylim(42, 80)
    ax.set_xticks([0, 4, 7, 12, 16, 19, 23])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Level [dB]")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    save_figure(output_dir, "lden_profile.png")
    plt.close()


def generate_tonality_spectrum(output_dir: str) -> None:
    """Annotated spectrum for the tone-to-noise ratio method."""
    print("Generating tonality_spectrum.png...")
    from phonometry import tone_to_noise_ratio
    from phonometry.psychoacoustics.tonality import _averaged_spectrum, _critical_band

    fs = 48000
    rng = np.random.default_rng(21)
    tt = np.arange(fs * 30) / fs
    x = (np.sqrt(2) * 0.1 * np.sin(2 * np.pi * 1000 * tt)
         + 0.05 * rng.standard_normal(tt.size))
    result = tone_to_noise_ratio(x, fs)
    freqs, power, _ = _averaged_spectrum(x - np.mean(x), fs, 1.0)
    f1, f2, _ = _critical_band(result.frequency)

    _, ax = plt.subplots(figsize=(10, 6))
    sel_band = (freqs > 700) & (freqs < 1400)
    db = 10 * np.log10(np.maximum(power, 1e-18))
    ax.plot(freqs[sel_band], db[sel_band], color=COLOR_PRIMARY, linewidth=1.0,
            label="Averaged FFT spectrum (Hann)")
    ax.axvspan(f1, f2, color=COLOR_TERTIARY, alpha=0.15,
               label="Critical band around the tone")
    ax.axvline(result.frequency, color=COLOR_SECONDARY, linewidth=1.4,
               linestyle="--")
    ax.annotate(
        f"TNR = {result.ratio_db:.1f} dB\n(criterion {result.criterion_db:.1f} dB)",
        xy=(result.frequency, db.max() - 2), xytext=(1120, db.max() - 8),
        fontsize=11, arrowprops={"arrowstyle": "->", "lw": 1.0},
    )
    ax.set_title("Tone-to-Noise Ratio (ECMA-418-1, clause 11)",
                 fontweight="bold", pad=12)
    ax.set_xlim(700, 1400)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Bin power [dB]")
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "tonality_spectrum.png")
    plt.close()


def generate_loudness_pattern(output_dir: str) -> None:
    """Specific loudness N'(z) of a narrowband vs a broadband sound."""
    print("Generating loudness_pattern.png...")
    from phonometry import loudness_zwicker_from_spectrum

    # 28 one-third-octave band levels, 25 Hz .. 12.5 kHz (ISO 532-1
    # clause 5.3). Index 16 is the 1 kHz band.
    narrow_levels = np.full(28, -60.0)
    narrow_levels[16] = 60.0
    narrow = loudness_zwicker_from_spectrum(narrow_levels)
    flat = loudness_zwicker_from_spectrum(np.full(28, 60.0))

    z = np.arange(1, 241) * 0.1  # 0.1-Bark steps up to 24 Bark

    _, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(z, flat.specific, color=COLOR_SECONDARY, alpha=0.25)
    ax.plot(z, flat.specific, color=COLOR_SECONDARY, linewidth=1.6,
            label=f"Flat broadband 60 dB - N = {flat.loudness:.1f} sone")
    ax.fill_between(z, narrow.specific, color=COLOR_PRIMARY, alpha=0.35)
    ax.plot(z, narrow.specific, color=COLOR_PRIMARY, linewidth=1.6,
            label=f"1 kHz narrowband - N = {narrow.loudness:.1f} sone")

    peak_z = float(z[np.argmax(narrow.specific)])
    ax.annotate(
        "Shaded area = total loudness N",
        xy=(peak_z + 0.6, float(narrow.specific.max()) * 0.45),
        xytext=(12.5, float(narrow.specific.max()) * 0.75),
        fontsize=10, arrowprops={"arrowstyle": "->", "lw": 0.9},
    )
    ax.set_title("Specific Loudness Pattern (ISO 532-1 Zwicker)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Critical-band rate z [Bark]")
    ax.set_ylabel("Specific loudness N' [sone/Bark]")
    ax.set_xlim(0, 24)
    # Headroom above the tallest pattern so the legend stays clear of it.
    ax.set_ylim(0, float(flat.specific.max()) * 1.28)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "loudness_pattern.png")
    plt.close()


def generate_sti_curve(output_dir: str) -> None:
    """STI vs reverberation time: pipeline points vs the analytic MTF."""
    print("Generating sti_vs_t60.png...")
    from phonometry import sti_from_impulse_response

    fs = 48000
    t60_points = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]

    # The 14 full-STI modulation frequencies and the male alpha/beta
    # factors of IEC 60268-16 Ed.5 Table A.1 (phonometry.hearing.sti keeps them
    # private, so they are restated here for the analytic reference).
    mod_freqs = np.array([0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50,
                          3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5])
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])

    def analytic_sti(t60: float) -> float:
        # Schroeder MTF of an exponential decay: m(F) = 1/sqrt(1+(2*pi*F*T/13.8)^2)
        m = 1.0 / np.sqrt(1.0 + (2 * np.pi * mod_freqs * t60 / 13.8) ** 2)
        snr_eff = np.clip(10 * np.log10(m / (1 - m)), -15.0, 15.0)
        mti = np.full(7, ((snr_eff + 15.0) / 30.0).mean())
        return float(np.dot(alpha, mti) - np.dot(beta, np.sqrt(mti[:-1] * mti[1:])))

    rng = np.random.default_rng(2026)
    measured = []
    for t60 in t60_points:
        t = np.arange(int(2 * t60 * fs)) / fs
        ir = rng.standard_normal(t.size) * np.exp(-6.9077 * t / t60)
        measured.append(sti_from_impulse_response(ir, fs).sti)

    t_dense = np.logspace(np.log10(0.25), np.log10(6.0), 200)
    sti_dense = [analytic_sti(float(t)) for t in t_dense]

    _, ax = plt.subplots(figsize=(10, 6))
    # Annex F qualification bands (informative): edges 0.36 .. 0.76.
    edges = [0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76]
    letters = ["U", "J", "I", "H", "G", "F", "E", "D", "C", "B", "A", "A+"]
    y_min, y_max = 0.15, 0.95
    bounds = [y_min] + edges + [y_max]
    cmap = plt.get_cmap("RdYlGn")
    for i, letter in enumerate(letters):
        lo, hi = bounds[i], bounds[i + 1]
        ax.axhspan(lo, hi, color=theme_fill(cmap(i / (len(letters) - 1)), ax),
                   lw=0, zorder=0)
        ax.text(0.985, (lo + hi) / 2, letter, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8, color=COLOR_FG, alpha=0.7)
    ax.text(0.92, 0.985, "Annex F rating", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=COLOR_FG, alpha=0.7)

    ax.plot(t_dense, sti_dense, color=COLOR_PRIMARY, linestyle="--",
            linewidth=1.5, label="Analytic Schroeder MTF (closed form)")
    ax.plot(t60_points, measured, "o", color=COLOR_SECONDARY, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6,
            label="Measured (sti_from_impulse_response)")

    ax.set_xscale("log")
    ax.set_title("STI vs Reverberation Time (IEC 60268-16)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Reverberation time T60 [s]")
    ax.set_ylabel("STI")
    ax.set_xlim(0.25, 6.0)
    ax.set_ylim(y_min, y_max)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(t60_points)
    ax.set_xticklabels(["0.3", "0.5", "0.8", "1.2", "2", "3", "5"])
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "sti_vs_t60.png")
    plt.close()


def generate_intensity_demo(output_dir: str) -> None:
    """p-p intensity: plane progressive wave vs reactive standing wave."""
    print("Generating intensity_demo.png...")
    from phonometry import sound_intensity

    fs = 48000
    dr, c = 0.012, 343.0
    duration = 4.0
    n = int(fs * duration)

    # Broadband noise, band-limited and scaled to ~70 dB SPL, in pascals.
    rng = np.random.default_rng(2026)
    noise = rng.standard_normal(n)
    sos = scipy_signal.butter(4, [80.0, 6000.0], btype="bandpass", fs=fs, output="sos")
    noise = scipy_signal.sosfilt(sos, noise)
    noise *= 0.063 / np.std(noise)

    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    k = 2 * np.pi * freqs / c

    # Plane progressive wave: microphone 2 sees the wave dr/c later.
    p1_plane = noise
    p2_plane = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * dr / c), n)
    plane = sound_intensity(p1_plane, p2_plane, fs, dr, fraction=3, limits=[100.0, 5000.0])

    # Standing wave: equal counter-propagating waves, probe centred at x0.
    x0 = 0.30
    def standing_pressure(pos: float) -> np.ndarray:
        return np.fft.irfft(spectrum * 2.0 * np.cos(k * pos), n)
    standing = sound_intensity(
        standing_pressure(x0 - dr / 2), standing_pressure(x0 + dr / 2),
        fs, dr, fraction=3, limits=[100.0, 5000.0],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, res, title in [
        (ax1, plane, "Plane wave: Lp ≈ LI"),
        (ax2, standing, "Standing wave: reactive field"),
    ]:
        ax.semilogx(res.frequency, res.pressure_level, marker="o", markersize=5,
                    color=COLOR_PRIMARY, linewidth=1.5, markerfacecolor="white",
                    markeredgewidth=1.3, label="Pressure level Lp")
        ax.semilogx(res.frequency, res.intensity_level, marker="s", markersize=5,
                    color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
                    markerfacecolor="white", markeredgewidth=1.3,
                    label="Intensity level LI")
        apply_axis_styling(ax, title, xlim=(90, 5600), ylim=(0, 85))
        # The standard octave ticks extend past the band range: re-clamp.
        ax.set_xlim(90, 5600)
        dpi_db = round(float(res.total_pressure_intensity_index), 1) + 0.0
        ax.text(0.05, 0.33, f"Pressure-intensity index\nδpI = {dpi_db:.1f} dB",
                transform=ax.transAxes, fontsize=10, va="bottom", color=COLOR_FG)
        ax.legend(loc="upper right", fontsize=9)
    ax2.set_ylabel("")

    fig.suptitle("Sound Intensity with a p-p Probe (IEC 61043)", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "intensity_demo.png")
    plt.close()


def generate_schroeder_decay(output_dir: str) -> None:
    """Schroeder backward integration with T20/T30/EDT regressions (ISO 3382)."""
    print("Generating schroeder_decay.png...")
    from phonometry import decay_curve, room_parameters
    from phonometry.room.room_acoustics import (
        _EDT_RANGE,
        _T20_RANGE,
        _T30_RANGE,
        _onset_index,
    )

    fs = 48000
    reverb_t = 1.2  # target reverberation time (s)
    duration = 2.5
    rng = np.random.default_rng(2026)
    t = np.arange(int(duration * fs)) / fs
    # Exponential decay (T = 1.2 s) excited by white noise + a realistic
    # background-noise floor (~-45 dB re the peak envelope).
    ir = rng.standard_normal(t.size) * np.exp(-6.9077 * t / reverb_t)
    ir = ir + rng.standard_normal(t.size) * 10.0 ** (-45.0 / 20.0)

    # Library outputs: the annotated numbers are exactly these.
    time, level = decay_curve(ir, fs)
    res = room_parameters(ir, fs, limits=None)
    edt, t20, t30 = float(res.edt[0]), float(res.t20[0]), float(res.t30[0])

    # Raw squared-IR level trace (onset-trimmed, normalized to its peak):
    # the noisy line the backward integration smooths into the decay curve.
    p2 = ir.astype(np.float64) ** 2
    p2 = p2[_onset_index(p2):]
    t_raw = np.arange(p2.size) / fs
    raw_db = 10.0 * np.log10(np.maximum(p2, p2.max() * 1e-12) / p2.max())

    def fit_line(decay_range: tuple[float, float]) -> tuple[float, float]:
        """Least-squares (slope, intercept) over an evaluation range,
        replicating room_acoustics._fit_decay_time so the drawn line has
        slope -60/T with the annotated T."""
        mask = (level <= -decay_range[0]) & (level >= -decay_range[1])
        slope, intercept = np.polyfit(time[mask], level[mask], 1)
        return float(slope), float(intercept)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(t_raw, raw_db, color="gray", alpha=0.28, linewidth=0.6, zorder=0,
            label="Raw squared IR level")
    ax.plot(time, level, color=COLOR_PRIMARY, linewidth=2.4, zorder=5,
            label="Schroeder decay curve")

    lines = [
        (_EDT_RANGE, "#9467bd", "-", "EDT fit (0 to −10 dB)", (0.0, -13.0)),
        (_T20_RANGE, COLOR_SECONDARY, "--", "T20 fit (−5 to −25 dB)", (0.0, -60.0)),
        (_T30_RANGE, COLOR_TERTIARY, "-.", "T30 fit (−5 to −35 dB)", (0.0, -60.0)),
    ]
    for decay_range, color, style, label, (lo, hi) in lines:
        slope, intercept = fit_line(decay_range)
        t_lo, t_hi = (lo - intercept) / slope, (hi - intercept) / slope
        ax.plot([t_lo, t_hi], [lo, hi], color=color, linestyle=style,
                linewidth=1.7, zorder=4, label=label)

    # Evaluation levels -5 / -25 / -35 dB and the decay-curve crossings.
    for target in (-5.0, -25.0, -35.0):
        ax.axhline(target, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
        t_cross = float(np.interp(target, level[::-1], time[::-1]))
        ax.plot(t_cross, target, "o", color=COLOR_FG, markersize=5, zorder=6)
        # Place level labels clear of the upper-right legend.
        ax.text(1.40, target + 0.8, f"{target:.0f} dB", ha="left",
                va="bottom", fontsize=8, color=COLOR_FG, alpha=0.85)

    ax.text(0.12, -7.0, "EDT slope", fontsize=8, color="#9467bd", rotation=0)
    facecolor = plt.rcParams["axes.facecolor"]
    ax.text(0.04, 0.06,
            f"EDT = {edt:.2f} s\nT20 = {t20:.2f} s\nT30 = {t30:.2f} s",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=11,
            bbox={"boxstyle": "round", "facecolor": facecolor,
                  "edgecolor": COLOR_FG, "alpha": 0.85})

    ax.set_title("Schroeder Integration and Reverberation Time (ISO 3382)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level re steady state [dB]")
    ax.set_xlim(0, duration)
    ax.set_ylim(-65, 3)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "schroeder_decay.png")
    plt.close()


def generate_excitation_signals(output_dir: str) -> None:
    """ISO 18233 excitations: ESS waveform + spectrogram and MLS + spectrum."""
    print("Generating excitation_signals.png...")
    from phonometry import mls_signal, sweep_signal

    fs = 48000
    f1, f2, secs = 50.0, 20000.0, 1.0
    sweep = sweep_signal(fs, f1, f2, secs)
    t = np.arange(sweep.size) / fs
    mls = mls_signal(12)  # length 2**12 - 1 = 4095

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    (ax_sw, ax_sp), (ax_ml, ax_ms) = axes

    # Exponential sine sweep: time-domain waveform.
    ax_sw.plot(t, sweep, color=COLOR_PRIMARY, linewidth=0.5)
    ax_sw.set_title("Exponential sine sweep — waveform", fontweight="bold")
    ax_sw.set_xlabel("Time [s]")
    ax_sw.set_ylabel("Amplitude")
    ax_sw.set_xlim(0.0, secs)
    ax_sw.set_ylim(-1.2, 1.2)
    ax_sw.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # Sweep spectrogram: the exponential frequency rise.
    ax_sp.specgram(sweep, NFFT=1024, Fs=fs, noverlap=512, cmap="magma")
    ax_sp.set_title("Sweep spectrogram (exponential rise)", fontweight="bold")
    ax_sp.set_xlabel("Time [s]")
    ax_sp.set_ylabel("Frequency [Hz]")
    ax_sp.set_ylim(0.0, fs / 2)

    # MLS: first samples of the bipolar sequence.
    show = 100
    ax_ml.step(np.arange(show), mls[:show], where="mid", color=COLOR_PRIMARY,
               linewidth=1.2)
    ax_ml.set_title(f"MLS — first {show} of {mls.size} samples", fontweight="bold")
    ax_ml.set_xlabel("Sample")
    ax_ml.set_ylabel("Amplitude")
    ax_ml.set_ylim(-1.4, 1.4)
    ax_ml.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)

    # MLS magnitude spectrum: essentially flat (white excitation).
    spec = np.abs(np.fft.rfft(mls))
    freqs = np.fft.rfftfreq(mls.size, d=1.0 / fs)
    ax_ms.semilogx(freqs[1:], 20.0 * np.log10(spec[1:] / np.median(spec[1:])),
                   color=COLOR_SECONDARY, linewidth=0.7)
    ax_ms.set_title("MLS magnitude spectrum (flat)", fontweight="bold")
    ax_ms.set_xlabel("Frequency [Hz]")
    ax_ms.set_ylabel("Magnitude [dB]")
    ax_ms.set_xlim(20.0, fs / 2)
    format_frequency_axis(ax_ms, 20.0, fs / 2)
    ax_ms.set_ylim(-12.0, 12.0)
    ax_ms.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)

    fig.suptitle("ISO 18233 excitation signals", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "excitation_signals.png")
    plt.close()


def generate_impulse_response(output_dir: str) -> None:
    """ISO 18233 recovered IR: waveform + log-magnitude / Schroeder decay."""
    print("Generating impulse_response.png...")
    from scipy.signal import fftconvolve

    from phonometry import impulse_response, sweep_signal

    fs = 48000
    sweep = sweep_signal(fs, 20.0, 20000.0, 1.5)

    # A synthetic room: direct sound, two early reflections and an
    # exponentially decaying diffuse tail (T ~ 0.6 s) plus a low noise floor.
    rng = np.random.default_rng(2026)
    n = int(0.7 * fs)
    system = np.zeros(n)
    system[80] = 1.0                       # direct sound
    system[1400] = 0.5                     # early reflection
    system[3100] = 0.32                    # second reflection
    tail_t = np.arange(n) / fs
    system += rng.standard_normal(n) * np.exp(-6.9077 * tail_t / 0.6) * 0.08
    system += rng.standard_normal(n) * 10.0 ** (-60.0 / 20.0)

    recorded = fftconvolve(sweep, system)
    ir = impulse_response(recorded, sweep, fs, length=n)

    h = np.asarray(ir, dtype=np.float64)
    time = np.arange(h.size) / fs
    peak = float(np.max(np.abs(h)))
    tiny = np.finfo(np.float64).tiny
    env_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny) / peak)
    energy = np.cumsum(h[::-1] ** 2)[::-1]
    edc_db = 10.0 * np.log10(np.maximum(energy, tiny) / energy[0])

    _fig, (ax_w, ax_d) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_w.plot(time, h / peak, color=COLOR_PRIMARY, linewidth=0.7)
    ax_w.set_title("Recovered room impulse response (ISO 18233)",
                   fontweight="bold", pad=10)
    ax_w.set_ylabel("Amplitude (norm.)")
    ax_w.set_ylim(-1.1, 1.1)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.annotate("direct sound", xy=(80 / fs, 1.0), xytext=(0.06, 0.86),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_FG, "alpha": 0.7})
    ax_w.annotate("reflections", xy=(1400 / fs, 0.5), xytext=(0.20, 0.62),
                  textcoords="axes fraction", fontsize=9, color=COLOR_FG,
                  arrowprops={"arrowstyle": "->", "color": COLOR_FG, "alpha": 0.7})

    ax_d.plot(time, env_db, color="#9ecae1", linewidth=0.7,
              label="Log-magnitude envelope")
    ax_d.plot(time, edc_db, color=COLOR_SECONDARY, linewidth=1.9,
              label="Schroeder decay (EDC)")
    ax_d.set_xlabel("Time [s]")
    ax_d.set_ylabel("Level re peak [dB]")
    ax_d.set_xlim(0.0, n / fs)
    ax_d.set_ylim(-80.0, 5.0)
    ax_d.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_d.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "impulse_response.png")
    plt.close()


def generate_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 weighted rating: measured R', shifted reference, deviations."""
    print("Generating insulation_rating.png...")
    from phonometry.building.insulation import (
        _INDEX_500_THIRD,
        _REF_THIRD_OCTAVE,
        weighted_rating,
    )

    # ISO 717-1 Annex C worked example (Table C.1), 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])

    result = weighted_rating(measured)
    reference = np.asarray(_REF_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Rw

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(freqs, measured, shifted, where=(measured < shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-1)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R' (third octave)")

    # Rw is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(f"Rw = {result.rating} dB", xy=(500, result.rating),
                xytext=(560, result.rating - 9), fontsize=12, fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, (f"Rw (C ; Ctr) = {result.rating} "
               f"({result.c:+d} ; {result.ctr:+d}) dB")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-1 Weighted Sound Reduction Index (Annex C example)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 44)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_rating.png")
    plt.close()


def generate_impact_rating(output_dir: str) -> None:
    """ISO 717-2 weighted impact rating: measured Ln, shifted reference, CI."""
    print("Generating impact_rating.png...")
    from phonometry.building.insulation import (
        _INDEX_500_THIRD,
        _REF_IMPACT_THIRD_OCTAVE,
        weighted_impact_rating,
    )

    # ISO 717-2 Annex C worked example (Table C.1): laboratory bare massive
    # floor, one-third octave 100 Hz .. 3150 Hz.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                         73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])

    result = weighted_impact_rating(measured)
    reference = np.asarray(_REF_IMPACT_THIRD_OCTAVE, dtype=float)
    shift = result.rating - _REF_IMPACT_THIRD_OCTAVE[_INDEX_500_THIRD]
    shifted = reference + shift  # shifted reference read at 500 Hz == Ln,w

    _, ax = plt.subplots(figsize=(10, 6.5))
    # For impact sound an unfavourable deviation occurs where the MEASURED
    # curve lies ABOVE the reference (opposite sign to ISO 717-1 airborne).
    ax.fill_between(freqs, shifted, measured, where=(measured > shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations (measured above reference)")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=4, zorder=3,
                label="Shifted reference curve (ISO 717-2)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured Ln (third octave)")

    # Ln,w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.rating, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    # The annotation sits in the clear gap between the rising measured curve
    # and the flat low-frequency reference plateau; the string is identical
    # in both languages, so the placement holds for every variant.
    ax.annotate(f"Ln,w = {result.rating} dB", xy=(500, result.rating),
                xytext=(135, result.rating - 4.2), fontsize=12,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, f"Reference curve shifted by {shift} dB"),
        (0.90, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.1f}"
               f" dB  (limit 32.0 dB)")),
        (0.83, f"Ln,w = {result.rating} dB ; CI = {result.ci:+d} dB"),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 717-2 Weighted Normalized Impact Sound Level "
                 "(Annex C example)", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized impact sound pressure level Ln [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(55, 86)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    save_figure(output_dir, "impact_rating.png")
    plt.close()


def generate_sharpness_weighting(output_dir: str) -> None:
    """DIN 45692 sharpness weighting g(z): DIN vs Aures vs von Bismarck."""
    print("Generating sharpness_weighting.png...")
    from phonometry.psychoacoustics.sharpness import _Z, _g_aures, _g_bismarck, _g_din

    z = _Z                       # 0.1 .. 24 Bark, 0.1-Bark steps
    total_n = 4.0                # reference loudness for the Aures variant (sone)
    g_din = _g_din(z)
    g_bismarck = _g_bismarck(z)
    g_aures = _g_aures(z, total_n)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.semilogy(z, g_din, color=COLOR_PRIMARY, linewidth=2.2,
                label="DIN 45692 g(z)")
    ax.semilogy(z, g_bismarck, color=COLOR_TERTIARY, linewidth=1.7,
                linestyle="--", label="von Bismarck (Annex B)")
    ax.semilogy(z, g_aures, color=COLOR_SECONDARY, linewidth=1.7,
                linestyle="-.", label=f"Aures (Annex B, N = {total_n:.0f} sone)")

    # DIN weighting is flat (g = 1) up to 15.8 Bark, von Bismarck up to 15.
    ax.axhline(1.0, color=COLOR_FG, linestyle="-", alpha=0.15, linewidth=1)
    ax.axvline(15.8, color=COLOR_PRIMARY, linestyle=":", alpha=0.5, linewidth=1)
    ax.axvline(15.0, color=COLOR_TERTIARY, linestyle=":", alpha=0.5, linewidth=1)
    ax.annotate("DIN knee\n15.8 Bark", xy=(15.8, 1.0), xytext=(10.2, 2.3),
                fontsize=9, color=COLOR_PRIMARY, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_PRIMARY})
    ax.annotate("Bismarck knee\n15 Bark", xy=(15.0, 1.0), xytext=(7.0, 0.5),
                fontsize=9, color=COLOR_TERTIARY, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_TERTIARY})

    ax.set_title("Sharpness Weighting g(z) (DIN 45692)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Critical-band rate z [Bark]")
    ax.set_ylabel("Weighting g(z)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0.4, 30)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "sharpness_weighting.png")
    plt.close()


def generate_open_plan_decay(output_dir: str) -> None:
    """ISO 3382-3 spatial decay: speech SPL and STI vs source distance."""
    print("Generating open_plan_decay.png...")
    from phonometry import open_plan_metrics

    # The worked example from the room-acoustics guide (matches its numbers).
    r = np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0])   # distances (m)
    lp = 65.0 - 7.0 * np.log2(r)                      # A-weighted speech level (dB)
    sti = 0.70 - 0.03 * r                             # STI per position
    m = open_plan_metrics(r, lp, sti)

    # Reconstruct the two regressions the metrics come from (2-16 m window).
    b_log = -m.d2s / np.log10(2.0)                    # slope vs lg(r/r0)
    a_lp = m.lp_as_4m - b_log * np.log10(4.0)
    d_sti, c_sti = np.polyfit(r, sti, 1)              # STI vs distance

    _fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xscale("log")
    rr = np.logspace(np.log10(2.0), np.log10(16.0), 100)
    line_spl, = ax.plot(rr, a_lp + b_log * np.log10(rr), color=COLOR_PRIMARY,
                        linestyle="--", linewidth=1.8,
                        label=f"Spatial decay D2,S = {m.d2s:.1f} dB")
    pts_spl, = ax.plot(r, lp, "o", color=COLOR_PRIMARY, markersize=7,
                       markerfacecolor="white", markeredgewidth=1.6,
                       label="Measured Lp,A,S")
    ax.axvline(4.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    mark_4m, = ax.plot(4.0, m.lp_as_4m, "D", color=COLOR_SECONDARY, markersize=9,
                       zorder=6, label=f"Lp,A,S,4m = {m.lp_as_4m:.0f} dB")
    ax.annotate(f"Lp,A,S,4m = {m.lp_as_4m:.0f} dB", xy=(4.0, m.lp_as_4m),
                xytext=(4.7, m.lp_as_4m + 4.5), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax.set_title("Open-Plan Spatial Decay of Speech (ISO 3382-3)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Distance from the talker r [m]")
    ax.set_ylabel("A-weighted SPL [dB]", color=COLOR_PRIMARY)
    ax.set_xlim(1.7, 18.0)
    ax.set_ylim(30, 62)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([2, 3, 4, 6, 8, 12, 16])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "12", "16"])
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.4)

    # Right axis: STI vs distance with the distraction / privacy crossings.
    ax2 = ax.twinx()
    rr2 = np.linspace(1.7, 18.0, 100)
    line_sti, = ax2.plot(rr2, c_sti + d_sti * rr2, color=COLOR_TERTIARY,
                         linewidth=1.7, label="STI vs distance")
    ax2.plot(r, sti, "s", color=COLOR_TERTIARY, markersize=5,
             markerfacecolor="white", markeredgewidth=1.3)
    for dist, level, name in [(m.rd, 0.50, "rD"), (m.rp, 0.20, "rP")]:
        ax2.axhline(level, color=COLOR_FG, linestyle=":", alpha=0.25, linewidth=1)
        ax2.plot(dist, level, "v", color=COLOR_SECONDARY, markersize=9, zorder=6)
        ax2.annotate(f"{name} = {dist:.1f} m", xy=(dist, level),
                     xytext=(dist * 0.62, level + 0.03), fontsize=9,
                     color=COLOR_SECONDARY,
                     arrowprops={"arrowstyle": "->", "lw": 0.9,
                                 "color": COLOR_SECONDARY})
    ax2.set_ylabel("STI", color=COLOR_TERTIARY)
    ax2.set_ylim(0.1, 0.75)

    handles = [pts_spl, line_spl, mark_4m, line_sti]
    ax.legend(handles, [str(h.get_label()) for h in handles],
              loc="upper right", fontsize=9)
    save_figure(output_dir, "open_plan_decay.png")
    plt.close()


def generate_rectangular_room_modes(output_dir: str) -> None:
    """Mode ladder and modal density of Long's 7 x 5 x 3 m example room.

    One concept: where the eigenfrequencies of a rectangular room sit, which
    family each belongs to, and where the Schroeder frequency ends the modal
    regime.
    """
    print("Generating rectangular_room_modes...")
    from phonometry import room_modes

    # Long, Architectural Acoustics 2nd ed., Table 8.1 (printed p. 325).
    result = room_modes(
        (7.0, 5.0, 3.0), max_frequency=200.0, speed_of_sound=344.0,
        reverberation_time=0.8,
    )
    result.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6.5)
    plt.tight_layout()
    save_figure(output_dir, "rectangular_room_modes.svg")
    plt.close()


def generate_restaurant_crowd_noise(output_dir: str) -> None:
    """Restaurant self-noise against occupancy for three absorption areas.

    One concept: the crowd generates its own background, and only absorption
    keeps it below the level at which cross-table conversation fails.
    """
    print("Generating restaurant_crowd_noise...")
    from phonometry import crowd_noise

    # Long, Chapter 17 (printed pp. 665-666): a hard room of about 20 metric
    # sabins, the same room with a partly absorptive ceiling, and with the
    # full alpha 0.9 ceiling of his 13.7 x 13.7 m example (+170 sabins).
    result = crowd_noise([20.0, 95.0, 190.0], distance=1.2)
    ax = result.plot(language=_LANG)
    ax.set_xlim(1.0, 20.0)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "restaurant_crowd_noise.svg")
    plt.close()


def generate_sound_reinforcement_geometry(output_dir: str) -> None:
    """The four points of a reinforcement feedback loop, schematically.

    One concept: the distances that set the two direct-field levels
    ``L(H-M)`` and ``L(H-L)`` of Long's stability criterion. The layout is
    schematic and each path carries its own length, because a talker 0.3 m
    from the microphone and a listener 12 m from the loudspeaker cannot share
    one usable drawing scale.
    """
    print("Generating sound_reinforcement_geometry...")
    from phonometry import plot_sound_reinforcement_geometry

    _fig, ax = plt.subplots(figsize=(10, 4.6))
    plot_sound_reinforcement_geometry(0.3, 4.0, 12.0, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sound_reinforcement_geometry.svg")
    plt.close()


def generate_erb_bandwidth(output_dir: str) -> None:
    """Auditory-filter bandwidth against centre frequency, with the Cam scale.

    One concept: how wide the ear's analysis filter is at each frequency
    (Glasberg and Moore, 1990), against the constant-percentage one-third
    octave for scale, with the Cam axis that counts those widths.
    """
    print("Generating erb_bandwidth...")
    from phonometry import cam_from_frequency, erb_bandwidth

    f = np.geomspace(50.0, 16000.0, 400)
    erb = np.asarray(erb_bandwidth(f))
    third_octave = f * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(f, erb, color=COLOR_PRIMARY, linewidth=2.2,
              label="ERB$_N$ (Glasberg & Moore, 1990)")
    ax.loglog(f, third_octave, color=COLOR_SECONDARY, linewidth=1.6,
              linestyle="--", label="One-third octave (23 % of f)")
    ax.set_title("Auditory-Filter Bandwidth and the Cam Scale "
                 "(Glasberg & Moore, 1990)", fontweight="bold", pad=12)
    ax.set_xlabel("Centre frequency [Hz]")
    ax.set_ylabel("Equivalent rectangular bandwidth ERB$_N$ [Hz]")
    ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    format_frequency_axis(ax, 50.0, 16000.0)
    from matplotlib.ticker import FixedFormatter, FixedLocator
    y_ticks = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in y_ticks]))
    ax.legend(loc="upper left", fontsize=9)

    cam_1k = float(np.asarray(cam_from_frequency(1000.0))[()])
    erb_1k = float(np.asarray(erb_bandwidth(1000.0))[()])
    ax.plot([1000.0], [erb_1k], "o", color=COLOR_PRIMARY, markersize=8,
            markerfacecolor="white", markeredgewidth=1.6, zorder=6)
    ax.annotate(f"1 kHz = {cam_1k:.2f} Cam", xy=(1000.0, erb_1k),
                xytext=(1400.0, 0.45 * erb_1k), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    # Top axis: the Cam scale, which counts ERB_N widths along frequency.
    from matplotlib.ticker import NullFormatter, NullLocator

    from phonometry import frequency_from_cam

    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    cam_ticks = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    ax2.set_xticks(np.asarray(frequency_from_cam(cam_ticks)))
    ax2.set_xticklabels([f"{c:.0f}" for c in cam_ticks])
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("ERB$_N$ number [Cam]", color=COLOR_TERTIARY)
    save_figure(output_dir, "erb_bandwidth.svg")
    plt.close()


# ---------------------------------------------------------------------------
# Advanced psychoacoustics (plan-17 block A): the ECMA-418-2 Sottek model
# (loudness, tonality, roughness) and the Moore-Glasberg ISO 532-2/-3 models.
# The heavy computations (ECMA loudness ~5 s/call, tonality ~8 s/call) are
# cached so they run once and are reused across the four themed/language
# passes rather than four times over.
# ---------------------------------------------------------------------------
_P_REF = 2e-5  # reference sound pressure [Pa]
_FS_PSY = 48000  # ECMA-418-2 / ISO 532 operate at 48 kHz


def _pure_tone(freq: float, spl_db: float, dur: float,
               fs: int = _FS_PSY) -> np.ndarray:
    """Calibrated sinusoid: sound pressure in pascals at *spl_db* dB SPL."""
    t = np.arange(round(dur * fs)) / fs
    amp = _P_REF * 10.0 ** (spl_db / 20.0) * np.sqrt(2.0)
    return np.asarray(amp * np.sin(2.0 * np.pi * freq * t))


@cache
def _loudness_models_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Total loudness (sone) vs level for the three loudness models."""
    from phonometry import (
        loudness_ecma,
        loudness_moore_glasberg_from_spectrum,
        loudness_zwicker,
    )

    levels = np.arange(20.0, 81.0, 10.0)  # 20..80 dB SPL
    zw, mg, ec = [], [], []
    for spl in levels:
        x = _pure_tone(1000.0, float(spl), 1.0)
        zw.append(loudness_zwicker(x, _FS_PSY, stationary=True).loudness)
        mg.append(
            loudness_moore_glasberg_from_spectrum([(1000.0, float(spl))]).loudness
        )
        ec.append(loudness_ecma(x, _FS_PSY).loudness)
    return levels, np.array(zw), np.array(mg), np.array(ec)


def generate_loudness_models_comparison(output_dir: str) -> None:
    """Zwicker vs Moore-Glasberg vs Sottek loudness for a 1 kHz tone."""
    print("Generating loudness_models_comparison.png...")
    levels, zw, mg, ec = _loudness_models_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(levels, zw, "o-", color=COLOR_PRIMARY, linewidth=2.0, markersize=6,
            label=f"Zwicker (ISO 532-1), N = {zw[2]:.1f} sone")
    ax.plot(levels, mg, "s--", color=COLOR_TERTIARY, linewidth=1.8, markersize=6,
            label=f"Moore-Glasberg (ISO 532-2), N = {mg[2]:.1f} sone")
    ax.plot(levels, ec, "^-.", color=COLOR_SECONDARY, linewidth=1.8, markersize=6,
            label=f"Sottek (ECMA-418-2), N = {ec[2]:.1f} sone")

    # The three models are anchored to 1 sone at 1 kHz / 40 dB SPL.
    ax.axhline(1.0, color=COLOR_FG, linestyle=":", alpha=0.35, linewidth=1)
    ax.plot(40.0, 1.0, "o", color=COLOR_FG, markersize=9,
            markerfacecolor="none", markeredgewidth=1.6, zorder=5)
    ax.annotate("Anchor: 1 kHz / 40 dB = 1 sone",
                xy=(40.0, 1.0), xytext=(21.5, 6.5), fontsize=10,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": COLOR_FG})
    ax.annotate("Models diverge at high levels",
                xy=(80.0, float(zw[-1])), xytext=(52.0, 13.5), fontsize=9,
                color=COLOR_FG, ha="center",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_title("Loudness Models Compared (1 kHz tone)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Sound pressure level [dB SPL]")
    ax.set_ylabel("Total loudness N [sone]")
    ax.set_xlim(18, 82)
    ax.set_ylim(0, float(zw[-1]) * 1.08)
    ax.set_xticks([20, 30, 40, 50, 60, 70, 80])
    ax.legend(loc="upper left", fontsize=9)
    save_figure(output_dir, "loudness_models_comparison.png")
    plt.close()


@cache
def _sottek_specific_data() -> tuple[np.ndarray, np.ndarray, float]:
    """ECMA-418-2 specific loudness N'(z) of a 1 kHz / 60 dB tone."""
    from phonometry.psychoacoustics import loudness_ecma

    el = loudness_ecma(_pure_tone(1000.0, 60.0, 1.0), _FS_PSY)
    return el.bark.copy(), el.specific_loudness.copy(), float(el.loudness)


def generate_sottek_specific_loudness(output_dir: str) -> None:
    """ECMA-418-2 (Sottek) specific loudness N'(z) over the Bark-rate scale."""
    print("Generating sottek_specific_loudness.png...")
    bark, spec, total = _sottek_specific_data()

    _, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(bark, spec, color=COLOR_PRIMARY, alpha=0.30)
    ax.plot(bark, spec, color=COLOR_PRIMARY, linewidth=1.8,
            label=f"1 kHz tone, 60 dB (N = {total:.1f} sone_HMS)")

    peak_i = int(np.argmax(spec))
    ax.annotate("Peak specific loudness",
                xy=(float(bark[peak_i]), float(spec[peak_i])),
                xytext=(float(bark[peak_i]) + 4.5, float(spec[peak_i]) * 0.92),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_title("Sottek Specific Loudness (ECMA-418-2)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Critical-band rate z [Bark]")
    ax.set_ylabel("Specific loudness N' [sone_HMS/Bark]")
    ax.set_xlim(0, float(bark[-1]))
    ax.set_ylim(0, float(spec.max()) * 1.25)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "sottek_specific_loudness.png")
    plt.close()


@cache
def _tonality_data() -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    """ECMA-418-2 tonality T(t) for a 1 kHz tone-in-noise vs pure noise."""
    from phonometry.psychoacoustics import tonality_ecma

    rng = np.random.default_rng(2026)
    dur = 2.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    noise = rng.standard_normal(t.size)
    noise = noise / np.sqrt(np.mean(noise ** 2)) * _P_REF * 10.0 ** (50.0 / 20.0)
    tone = _P_REF * 10.0 ** (50.0 / 20.0) * np.sqrt(2.0) * np.sin(2.0 * np.pi * 1000.0 * t)

    tin = tonality_ecma(tone + noise, _FS_PSY)
    pn = tonality_ecma(noise, _FS_PSY)
    return (tin.time.copy(), tin.tonality_vs_time.copy(), float(tin.tonality),
            pn.time.copy(), pn.tonality_vs_time.copy(), float(pn.tonality))


@cache
def _roughness_sweep_data() -> tuple[np.ndarray, np.ndarray]:
    """ECMA-418-2 roughness R vs AM frequency, 1 kHz carrier, 100 % AM, 60 dB."""
    from phonometry.psychoacoustics import roughness_ecma

    dur = 1.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    fmods = np.array([20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
                      100.0, 120.0, 150.0, 180.0, 200.0])
    r = []
    for fm in fmods:
        am = (1.0 + 1.0 * np.sin(2.0 * np.pi * fm * t)) * np.sin(2.0 * np.pi * 1000.0 * t)
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (60.0 / 20.0)
        r.append(roughness_ecma(am, _FS_PSY).roughness)
    return fmods, np.array(r)


def generate_tonality_roughness_demo(output_dir: str) -> None:
    """Two-panel ECMA-418-2 sound-quality demo: tonality T(t) and roughness."""
    print("Generating tonality_roughness_demo.png...")
    (t_tin, tv_tin, t_single, _t_pn, tv_pn, pn_single) = _tonality_data()
    fmods, r = _roughness_sweep_data()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 8.5))

    # -- Top: time-dependent tonality, tone-in-noise vs pure noise -----------
    ax0.plot(t_tin, tv_tin, color=COLOR_PRIMARY, linewidth=1.8,
             label=f"Tone in noise (T = {t_single:.2f} tu_HMS)")
    ax0.plot(_t_pn, tv_pn, color=COLOR_SECONDARY, linewidth=1.8,
             label=f"Pure noise (T = {pn_single:.2f} tu_HMS)")
    ax0.set_title("ECMA-418-2 Tonality T(t)", fontweight="bold", pad=10)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Tonality T [tu_HMS]")
    ax0.set_xlim(0, float(t_tin[-1]))
    ax0.set_ylim(0, max(1.0, float(tv_tin.max()) * 1.30))
    ax0.legend(loc="upper right", fontsize=9)

    # -- Bottom: roughness vs modulation frequency (peak near 70 Hz) ---------
    ax1.plot(fmods, r, "o-", color=COLOR_TERTIARY, linewidth=2.0, markersize=6,
             label="1 kHz carrier, 100 % AM")
    peak_i = int(np.argmax(r))
    ax1.plot(fmods[peak_i], r[peak_i], "o", color=COLOR_SECONDARY, markersize=9,
             markerfacecolor="none", markeredgewidth=1.6, zorder=5)
    ax1.annotate(f"Peak R = {r[peak_i]:.1f} asper @ {fmods[peak_i]:.0f} Hz",
                 xy=(float(fmods[peak_i]), float(r[peak_i])),
                 xytext=(105.0, float(r[peak_i]) * 0.95), fontsize=10,
                 color=COLOR_FG,
                 arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax1.set_title("ECMA-418-2 Roughness vs Modulation Frequency",
                  fontweight="bold", pad=10)
    ax1.set_xlabel("Modulation frequency f_mod [Hz]")
    ax1.set_ylabel("Roughness R [asper]")
    ax1.set_xlim(10, 210)
    ax1.set_ylim(0, float(r.max()) * 1.25)
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle("Sound Quality Metrics (ECMA-418-2 Sottek Hearing Model)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(output_dir, "tonality_roughness_demo.png")
    plt.close()


@cache
def _fs_ecma_sweep_data() -> tuple[np.ndarray, np.ndarray]:
    """ECMA-418-2 Clause 9 F of a 1 kHz / 60 dB / 100 %-AM tone vs f_mod.

    Cached (language/theme independent): the Sottek fluctuation-strength
    chain is run once for the modulation-frequency sweep. The overall level
    convention (60 dB SPL of the modulated signal) matches the Clause 9
    calibration.
    """
    from phonometry.psychoacoustics import fluctuation_strength_ecma

    dur = 3.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    fmods = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0,
                      24.0, 32.0])
    f_vals = []
    for fm in fmods:
        am = (1.0 + np.sin(2.0 * np.pi * fm * t)) * carrier
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (60.0 / 20.0)
        f_vals.append(
            fluctuation_strength_ecma(am, float(_FS_PSY)).fluctuation_strength
        )
    return fmods, np.array(f_vals)


def generate_hms_modulation_bandpass(output_dir: str) -> None:
    """Complementary modulation band-passes of the Sottek Hearing Model.

    Fluctuation strength (ECMA-418-2 Clause 9, maximum near 4 Hz) and
    roughness (Clause 7, maximum near 70 Hz) computed for the same
    1 kHz / 100 % AM / 60 dB signal family over the modulation rate.
    """
    print("Generating hms_modulation_bandpass...")
    fm_fs, f_vals = _fs_ecma_sweep_data()
    fm_r, r_vals = _roughness_sweep_data()

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Per-metric identity colors (teal = fluctuation strength, brown =
    # roughness), matching the result .plot() renderers; legible on both
    # themes, kept literal on purpose.
    ax.semilogx(fm_fs, f_vals, "o-", color="#17becf", linewidth=2.2,
                markersize=6, label="Fluctuation strength F (Clause 9, slow modulation)")
    ax.semilogx(fm_r, r_vals, "s-", color="#8c564b", linewidth=2.2,
                markersize=6, label="Roughness R (Clause 7, fast modulation)")

    i_f = int(np.argmax(f_vals))
    i_r = int(np.argmax(r_vals))
    ax.annotate(f"F = {f_vals[i_f]:.2f} vacil_HMS @ {fm_fs[i_f]:.0f} Hz",
                xy=(float(fm_fs[i_f]), float(f_vals[i_f])),
                xytext=(float(fm_fs[i_f]) * 1.6, float(f_vals[i_f]) * 1.06),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.annotate(f"R = {r_vals[i_r]:.2f} asper @ {fm_r[i_r]:.0f} Hz",
                xy=(float(fm_r[i_r]), float(r_vals[i_r])),
                xytext=(float(fm_r[i_r]) * 1.5, float(r_vals[i_r]) * 1.06),
                fontsize=10, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_xlabel("Modulation frequency f_mod [Hz]")
    ax.set_ylabel("F [vacil_HMS] / R [asper]")
    ax.set_title("Slow vs Fast Modulation Perception (ECMA-418-2 Sottek Hearing Model)",
                 fontweight="bold", pad=12)
    top = max(float(np.max(f_vals)), float(np.max(r_vals)))
    ax.set_ylim(0.0, top * 1.22)
    ax.set_xlim(0.4, 260.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 70, 140, 250])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "70", "140", "250"])
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.03, "1 kHz carrier, 100 % AM, overall 60 dB SPL",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "hms_modulation_bandpass.svg")
    plt.close()


@cache
def _fluctuation_am_tone_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Osses 2016 signal-model F of a 1 kHz / 70 dB / 100 %-AM tone vs f_mod.

    Cached (language/theme independent): the signal model is run once for the
    modulation-frequency sweep {1, 2, 4, 8, 16, 32} Hz. Reproduces the band-pass
    sensation with its maximum at 4 Hz (Osses 2016 Table 1 trend).
    """
    from phonometry.psychoacoustics import fluctuation_strength

    dur = 2.0
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    carrier = np.sin(2.0 * np.pi * 1000.0 * t)
    fmods = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    f_vals = []
    for fm in fmods:
        am = (1.0 + np.sin(2.0 * np.pi * fm * t)) * carrier
        am = am / np.sqrt(np.mean(am ** 2)) * _P_REF * 10.0 ** (70.0 / 20.0)
        f_vals.append(fluctuation_strength(am, float(_FS_PSY)).fluctuation_strength)
    return fmods, np.array(f_vals)


def generate_fluctuation_strength(output_dir: str) -> None:
    """Fluctuation strength F vs modulation frequency: the 4 Hz band-pass peak."""
    print("Generating fluctuation_strength...")
    from phonometry import fluctuation_strength_am_noise

    # Exact closed form (Fastl & Zwicker Eq. 10.2) for AM broadband noise at
    # 60 dB, 100 % modulation, swept over f_mod on a log axis.
    fmod = np.logspace(np.log10(0.5), np.log10(32.0), 240)
    f_bbn = np.array([fluctuation_strength_am_noise(60.0, 1.0, fm) for fm in fmod])
    bbn_peak = int(np.argmax(f_bbn))

    # Osses 2016 signal model on an AM tone (70 dB), same modulation sweep.
    fm_tone, f_tone = _fluctuation_am_tone_sweep()
    tone_peak = int(np.argmax(f_tone))

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(fmod, f_bbn, color=COLOR_PRIMARY, linewidth=2.4,
                label=(f"AM broadband noise (closed form, 60 dB), "
                       f"peak {f_bbn[bbn_peak]:.1f} vacil"))
    ax.plot(fmod[bbn_peak], f_bbn[bbn_peak], "o", color=COLOR_PRIMARY,
            markersize=8, markerfacecolor="white", markeredgewidth=1.6, zorder=6)
    ax.axvline(4.0, color=COLOR_FG, linestyle="--", linewidth=1.0, alpha=0.7,
               label="4 Hz reference")
    ax.set_xlabel("Modulation frequency f_mod [Hz]")
    ax.set_ylabel("Fluctuation strength F [vacil]")
    ax.set_ylim(0.0, float(f_bbn.max()) * 1.18)
    ax.set_title("Fluctuation Strength — 4 Hz Band-Pass Characteristic",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32])
    ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32"])

    # Overlay the signal-model AM-tone sweep on a secondary axis (its absolute
    # scale differs from the broadband closed form, but the band-pass shape and
    # 4 Hz maximum coincide).
    ax2 = ax.twinx()
    ax2.plot(fm_tone, f_tone, "s--", color=COLOR_TERTIARY, linewidth=1.8,
             markersize=7, label=(f"AM tone (signal model, 70 dB), "
                                  f"peak {f_tone[tone_peak]:.2f} vacil"))
    ax2.set_ylabel("AM-tone F, signal model [vacil]", color=COLOR_TERTIARY)
    ax2.tick_params(axis="y", labelcolor=COLOR_TERTIARY)
    ax2.set_ylim(0.0, float(f_tone.max()) * 1.18)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    info = [
        "F = 5.8 (1.25 m - 0.25)(0.05 L - 1)",
        "    / [(fmod/5)^2 + 4/fmod + 1.5]  vacil",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "fluctuation_strength.svg")
    plt.close()


def generate_psychoacoustic_annoyance(output_dir: str) -> None:
    """Psychoacoustic annoyance PA vs loudness N5 for three sensation profiles."""
    print("Generating psychoacoustic_annoyance...")
    from phonometry.psychoacoustics import psychoacoustic_annoyance

    n5 = np.linspace(4.0, 60.0, 200)
    # (label, sharpness [acum], fluctuation strength [vacil], roughness [asper],
    #  colour, linestyle).
    profiles = [
        ("Baseline: S = 1.75 acum, F = R = 0", 1.75, 0.0, 0.0,
         COLOR_FG, "--"),
        ("Sharp: S = 3.5 acum", 3.5, 0.0, 0.0, COLOR_PRIMARY, "-"),
        ("Rough + fluctuating: F = 1.2 vacil, R = 0.7 asper", 2.0, 1.2, 0.7,
         COLOR_TERTIARY, "-"),
    ]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for label, s, f, r, color, ls in profiles:
        pa = np.array([psychoacoustic_annoyance(v, s, f, r).annoyance for v in n5])
        lw = 1.6 if ls == "--" else 2.4
        alpha = 0.7 if ls == "--" else 1.0
        ax.plot(n5, pa, color=color, linestyle=ls, linewidth=lw, alpha=alpha,
                label=label)

    # Worked example: N5 = 30 sone, S = 2.0 acum, F = 0.5 vacil, R = 0.3 asper.
    ex = psychoacoustic_annoyance(30.0, 2.0, 0.5, 0.3)
    ax.plot([30.0], [ex.annoyance], "o", color=COLOR_SECONDARY, markersize=10,
            markerfacecolor="white", markeredgewidth=2.0, zorder=6,
            label=f"Worked example (PA = {ex.annoyance:.2f})")
    ax.annotate(f"PA = {ex.annoyance:.2f}\nwS = {ex.w_s:.3f}, wFR = {ex.w_fr:.3f}",
                xy=(30.0, ex.annoyance), xytext=(33.0, ex.annoyance * 0.72),
                fontsize=9, color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})

    ax.set_xlabel("Percentile loudness N5 [sone]")
    ax.set_ylabel("Psychoacoustic annoyance PA")
    ax.set_xlim(0.0, 62.0)
    ax.set_ylim(0.0, None)
    ax.set_title("Psychoacoustic Annoyance vs Loudness (Fastl & Zwicker)",
                 fontweight="bold", pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "PA = N5 (1 + sqrt(wS^2 + wFR^2))",
        "wS  = (S - 1.75) 0.25 lg(N5 + 10)",
        "wFR = (2.18 / N5^0.4)(0.4 F + 0.6 R)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "psychoacoustic_annoyance.svg")
    plt.close()


_FS_ELECTRO = 48000  # audio sample rate for the electroacoustic demos


def generate_distortion(output_dir: str) -> None:
    """Annotated harmonic spectrum with the THD of a synthetic amplifier output."""
    print("Generating distortion...")
    from phonometry import harmonic_analysis

    fs = _FS_ELECTRO
    n = fs  # 1 s -> 1 Hz bins; every harmonic lands on a bin
    t = np.arange(n) / fs
    f0 = 1000.0
    # A 1 kHz fundamental with a decaying harmonic series over a broadband
    # noise floor, the kind of output an amplifier under a single-tone test
    # produces. The noise makes THD+N exceed the harmonic-only THD (it also
    # counts the noise), while SINAD reports the noise-and-distortion headroom.
    amps = {1: 1.0, 2: 0.02, 3: 0.012, 4: 0.006, 5: 0.003}
    sig = sum(a * np.sin(2 * np.pi * k * f0 * t) for k, a in amps.items())
    rng = np.random.default_rng(2026)
    sig = sig + rng.standard_normal(n) * 1.2e-2

    res = harmonic_analysis(sig, fs, f0, n_harmonics=len(amps))

    # Magnitude spectrum (coherent-gain normalised) in dB re the fundamental.
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(sig * window)) * 2.0 / np.sum(window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    ref = np.max(spectrum)
    spec_db = 20.0 * np.log10(np.maximum(spectrum, 1e-12) / ref)

    _fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.plot(freqs, spec_db, color=COLOR_PRIMARY, linewidth=1.0, alpha=0.8,
            label="Magnitude spectrum")
    hz = np.asarray(res.harmonic_frequencies)
    ha = np.asarray(res.harmonic_amplitudes)
    hdb = 20.0 * np.log10(np.maximum(ha, 1e-12) / ha[0])
    ax.plot(hz, hdb, "o", color=COLOR_SECONDARY, markersize=7, zorder=6,
            label="Harmonics n·f₁")
    for k, (fk, lk) in enumerate(zip(hz, hdb), start=1):
        ax.annotate(f"n={k}", xy=(fk, lk), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8,
                    color=COLOR_FG)

    ax.set_xlim(0.0, 9000.0)
    ax.set_ylim(-100.0, 8.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level re fundamental [dB]")
    ax.set_title("Harmonic Distortion of a Single-Tone Test (IEC 60268-3)",
                 fontweight="bold", pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        f"THD (F) = {res.thd_f * 100:.2f}%",
        f"THD (R) = {res.thd_r * 100:.2f}%",
        f"THD+N   = {res.thd_plus_noise * 100:.2f}%",
        f"SINAD   = {res.sinad_db:.1f} dB",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "distortion.svg")
    plt.close()


def generate_frequency_response(output_dir: str) -> None:
    """Bode magnitude and coherence of an estimated frequency response (H1)."""
    print("Generating frequency_response...")
    from scipy import signal as sp_signal

    from phonometry import transfer_function

    fs = _FS_ELECTRO
    n = 400000
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    # A resonant second-order band-pass "device under test".
    b, a = sp_signal.butter(2, [400.0, 4000.0], btype="band", fs=fs)
    y = sp_signal.lfilter(b, a, x)
    # Additive output noise pulls the coherence down where the signal is weak.
    y = y + rng.standard_normal(n) * np.sqrt(np.mean(y**2)) * 0.05

    res = transfer_function(x, y, fs, estimator="H1")
    _, h_true = sp_signal.freqz(b, a, worN=res.frequencies, fs=fs)
    pos = res.frequencies > 0.0
    freqs = res.frequencies[pos]
    true_db = 20.0 * np.log10(np.maximum(np.abs(h_true[pos]), 1e-12))

    _fig, (ax_mag, ax_coh) = plt.subplots(
        2, 1, figsize=(10, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax_mag.semilogx(freqs, true_db, color=COLOR_FG, linestyle="--",
                    linewidth=1.6, alpha=0.7, label="True |H|")
    ax_mag.semilogx(freqs, res.magnitude_db[pos], color=COLOR_PRIMARY,
                    linewidth=1.8, label="Estimated |H| (H1)")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_ylim(-80.0, 5.0)
    ax_mag.set_title("Frequency Response and Coherence (Bendat & Piersol)",
                     fontweight="bold", pad=12)
    ax_mag.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_mag.set_axisbelow(True)
    ax_mag.legend(loc="lower center", fontsize=9)

    ax_coh.semilogx(freqs, res.coherence[pos], color=COLOR_TERTIARY,
                    linewidth=1.8)
    ax_coh.set_ylabel(r"Coherence $\gamma^2$")
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylim(0.0, 1.05)
    ax_coh.set_xlim(20.0, fs / 2.0)
    ax_coh.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_coh.set_axisbelow(True)
    for _axf in (ax_mag, ax_coh):
        format_frequency_axis(_axf, 20.0, fs / 2.0)
    plt.tight_layout()
    save_figure(output_dir, "frequency_response.svg")
    plt.close()


def generate_swept_sine_thd(output_dir: str) -> None:
    """THD(f) by order from one synchronized sweep (Farina / Novak)."""
    print("Generating swept_sine_thd...")
    from scipy import signal as sp_signal

    from phonometry import swept_sine_distortion, synchronized_sweep_signal

    fs = 48000
    f1, f2, seconds = 20.0, 6000.0, 4.0
    a2, a3 = 0.12, 0.08
    # Hammerstein chain: a memoryless cubic polynomial (exact Chebyshev
    # harmonic levels) followed by a 3 kHz low-pass, so each order rolls off
    # where its own product n*f crosses the filter corner.
    x = synchronized_sweep_signal(fs, f1, f2, seconds)
    b, a = sp_signal.butter(2, 3000.0, fs=fs)
    y = sp_signal.lfilter(b, a, x + a2 * x**2 + a3 * x**3)
    res = swept_sine_distortion(y, fs, f1, f2, seconds, n_harmonics=3)

    sel = (res.thd_frequencies >= 30.0) & (res.thd_frequencies <= 2800.0)
    freqs = res.thd_frequencies[sel]
    h1_ref = 1.0 + 3.0 * a3 / 4.0  # Chebyshev fundamental gain

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(freqs, 100.0 * res.thd[sel], color=COLOR_PRIMARY,
              linewidth=2.0, label="Total THD(f)")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[0][sel],
              color=COLOR_SECONDARY, linewidth=1.5, linestyle="--",
              label="2nd harmonic d₂(f)")
    ax.loglog(freqs, 100.0 * res.distortion_ratios[1][sel],
              color=COLOR_TERTIARY, linewidth=1.5, linestyle="--",
              label="3rd harmonic d₃(f)")
    ax.axhline(100.0 * (a2 / 2.0) / h1_ref, color=COLOR_SECONDARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote (a₂/2)/H₁")
    ax.axhline(100.0 * (a3 / 4.0) / h1_ref, color=COLOR_TERTIARY,
               linestyle=":", linewidth=1.2, alpha=0.8,
               label="Chebyshev asymptote (a₃/4)/H₁")
    ax.set_xlabel("Excitation frequency [Hz]")
    ax.set_ylabel("Distortion re fundamental [%]")
    ax.set_title("Swept-Sine Harmonic Distortion by Order (Farina / Novak)",
                 fontweight="bold", pad=12)
    ax.set_xlim(30.0, 2800.0)
    ax.set_ylim(0.05, 20.0)
    format_frequency_axis(ax, 30.0, 2800.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            "one sweep separates every distortion order;\n"
            "each rolls off where its product n·f crosses the 3 kHz corner",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "swept_sine_thd.svg")
    plt.close()


def generate_piston_directivity(output_dir: str) -> None:
    """Far-field beam pattern of a baffled circular piston at three ka values."""
    print("Generating piston_directivity...")
    from phonometry import piston_directivity_pattern

    piston_directivity_pattern([3.0, 8.0, 16.0]).plot(language=_LANG)
    save_figure(output_dir, "piston_directivity.svg")
    plt.close()


def _loudspeaker_datasheet_example() -> Any:
    """The IEC 60268-5 loudspeaker result shared by the section-7 .plot() figures."""
    from phonometry import loudspeaker_characteristics, radiating_piston

    freqs = np.geomspace(30, 24000, 320)
    spl = 87.0 + 1.2 * np.sin(2 * np.log2(freqs / 900.0))
    spl -= 10 * np.log10(1 + (50.0 / freqs) ** 6)       # low-frequency roll-off
    spl -= 10 * np.log10(1 + (freqs / 16000.0) ** 7)    # high-frequency roll-off
    fz = np.geomspace(20, 20000, 260)
    thd_f = np.geomspace(50, 5000, 140)
    return loudspeaker_characteristics(
        freqs, spl, rated_impedance=8.0, sensitivity_band=(200.0, 4000.0),
        impedance=(fz, 6.6 + 24 * np.exp(-(np.log2(fz / 52.0) ** 2) / 0.12)),
        distortion=(thd_f, 0.3 + 2.6 * np.exp(-(np.log2(thd_f / 70.0) ** 2) / 0.45)),
        directivity=radiating_piston(0.075, np.array([1000.0, 2000.0, 4000.0]),
                                     angles=np.radians(np.linspace(0, 90, 46))),
        polar_frequency=2000.0,
    )


def _microphone_datasheet_example() -> Any:
    """The IEC 60268-4 microphone result shared by the section-8 .plot() figures."""
    from phonometry import microphone_characteristics

    freqs = np.geomspace(20, 20000, 400)
    response = -10 * np.log10(1 + (30.0 / freqs) ** 4)      # low-frequency roll-off
    response -= 10 * np.log10(1 + (freqs / 19000.0) ** 8)   # high-frequency roll-off
    response += 2.0 * np.exp(-(np.log2(freqs / 9000.0) ** 2) / 0.3)  # presence region
    angles = np.linspace(0, 179, 359)
    cardioid = 20 * np.log10((1 + np.cos(np.radians(angles))) / 2)
    noise_f = np.geomspace(20, 20000, 31)
    spl_axis = np.linspace(100, 140, 81)
    return microphone_characteristics(
        freqs, response, 12.5, tolerance_db=3.0,          # 12.5 mV/Pa at 1 kHz
        rated_impedance=150.0, minimum_load_impedance=1000.0,
        noise_voltage=1.25e-6, max_spl_thd_percent=0.5,
        noise_spectrum=(noise_f, 6.0 + 12.0 * np.log10(1000.0 / noise_f)),
        distortion=(spl_axis, 0.5 * 10 ** ((spl_axis - 130.0) * 0.08)),
        polar=(angles, cardioid), polar_frequency=1000.0,
        powering="Phantom P48 (IEC 61938)", supply_current_ma=3.1,
    )


def generate_loudspeaker_response(output_dir: str) -> None:
    """On-axis SPL response with tolerance band and effective range (IEC 60268-5)."""
    print("Generating loudspeaker_response...")
    _loudspeaker_datasheet_example().plot(quantity="response", language=_LANG)
    save_figure(output_dir, "loudspeaker_response.svg")
    plt.close()


def generate_loudspeaker_impedance(output_dir: str) -> None:
    """Impedance modulus with the rated and 80 %-of-rated lines (IEC 60268-5)."""
    print("Generating loudspeaker_impedance...")
    _loudspeaker_datasheet_example().plot(quantity="impedance", language=_LANG)
    save_figure(output_dir, "loudspeaker_impedance.svg")
    plt.close()


def generate_loudspeaker_thd(output_dir: str) -> None:
    """Total harmonic distortion against frequency (IEC 60268-5)."""
    print("Generating loudspeaker_thd...")
    _loudspeaker_datasheet_example().plot(quantity="thd", language=_LANG)
    save_figure(output_dir, "loudspeaker_thd.svg")
    plt.close()


def generate_loudspeaker_directivity(output_dir: str) -> None:
    """Polar directivity on the IEC 60263 25 dB reference circle (IEC 60268-5)."""
    print("Generating loudspeaker_directivity...")
    _loudspeaker_datasheet_example().plot(quantity="directivity", language=_LANG)
    save_figure(output_dir, "loudspeaker_directivity.svg")
    plt.close()


def generate_microphone_response(output_dir: str) -> None:
    """Free-field response with tolerance band and reference markers (IEC 60268-4)."""
    print("Generating microphone_response...")
    _microphone_datasheet_example().plot(quantity="response", language=_LANG)
    save_figure(output_dir, "microphone_response.svg")
    plt.close()


def generate_microphone_directivity(output_dir: str) -> None:
    """Cardioid directional pattern on the 25 dB reference circle (IEC 60268-4)."""
    print("Generating microphone_directivity...")
    _microphone_datasheet_example().plot(quantity="directivity", language=_LANG)
    save_figure(output_dir, "microphone_directivity.svg")
    plt.close()


def generate_microphone_noise(output_dir: str) -> None:
    """Inherent-noise equivalent band-level spectrum (IEC 60268-4)."""
    print("Generating microphone_noise...")
    _microphone_datasheet_example().plot(quantity="noise", language=_LANG)
    save_figure(output_dir, "microphone_noise.svg")
    plt.close()


def generate_microphone_distortion(output_dir: str) -> None:
    """Total harmonic distortion against sound pressure level (IEC 60268-4)."""
    print("Generating microphone_distortion...")
    _microphone_datasheet_example().plot(quantity="distortion", language=_LANG)
    save_figure(output_dir, "microphone_distortion.svg")
    plt.close()


def generate_psd_confidence_smoothing(output_dir: str) -> None:
    """Calibrated PSD of pink noise: chi-square CI plus 1/3-oct smoothing."""
    print("Generating psd_confidence_smoothing...")
    from phonometry import (
        fractional_octave_smoothing,
        noise_signal,
        power_spectral_density,
    )

    fs = 48000.0
    x = noise_signal(fs, 20.0, color="pink", seed=11)
    res = power_spectral_density(x, fs, nperseg=4096)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    smooth = fractional_octave_smoothing(res.frequencies, res.psd, 3.0)[band]
    # The exact -3.01 dB/oct power law through the level at 1 kHz.
    i0 = int(np.argmin(np.abs(freqs - 1000.0)))
    ref_db = 10.0 * np.log10(smooth[i0]) - 10.0 * np.log10(freqs / freqs[i0])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.0, alpha=0.85, label="Welch PSD estimate")
    ax.semilogx(freqs, 10.0 * np.log10(smooth), color=COLOR_SECONDARY,
                linewidth=2.2, label="1/3-octave smoothed")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title("Calibrated Spectral Density of Pink Noise (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.985, 0.965,
            f"$n_d$ = {res.n_averages:.0f} averages, "
            f"$\\varepsilon_r$ = {100.0 * res.random_error:.1f} %",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "psd_confidence_smoothing.svg")


def generate_multitaper_psd_confidence(output_dir: str) -> None:
    """Thomson multitaper PSD of a short record vs a single-taper estimate."""
    print("Generating multitaper_psd_confidence...")
    from phonometry import multitaper_psd, noise_signal

    fs = 48000.0
    n = 8192  # a genuinely short record: 171 ms
    x = noise_signal(fs, n / fs, color="pink", seed=11)
    single = multitaper_psd(x, fs, n_tapers=1, adaptive=False)
    res = multitaper_psd(x, fs)
    band = (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    freqs = res.frequencies[band]
    # The exact -3.01 dB/oct power law through the mean level around 1 kHz.
    anchor = (freqs >= 800.0) & (freqs <= 1250.0)
    level_1k = float(np.mean(10.0 * np.log10(res.psd[band][anchor])))
    ref_db = level_1k - 10.0 * np.log10(freqs / 1000.0)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(freqs, 10.0 * np.log10(single.psd[band]), color="gray",
                alpha=0.45, linewidth=0.7,
                label="Single Slepian taper ($K$ = 1, $\\nu$ = 2)")
    ax.fill_between(
        freqs,
        10.0 * np.log10(res.ci_lower[band]),
        10.0 * np.log10(res.ci_upper[band]),
        color=COLOR_PRIMARY, alpha=0.28, lw=0.0,
        label="95 % chi-square confidence interval")
    ax.semilogx(freqs, 10.0 * np.log10(res.psd[band]), color=COLOR_PRIMARY,
                linewidth=1.2, label="Multitaper estimate ($K$ = 7, adaptive)")
    ax.semilogx(freqs, ref_db, color=COLOR_FG, linestyle="--", linewidth=1.4,
                alpha=0.7, label="Exact -3.01 dB/octave power law")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB re 1/Hz]")
    ax.set_title(
        "Thomson Multitaper Density of a Short Record (Percival & Walden)",
        fontweight="bold", pad=12)
    ax.set_xlim(20.0, 20000.0)
    format_frequency_axis(ax, 20.0, 20000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    nu = float(np.mean(res.degrees_of_freedom[1:-1]))
    ax.text(0.985, 0.965,
            f"171 ms record, $NW$ = 4, "
            f"$\\bar\\nu$ = {nu:.1f} equivalent dof",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "multitaper_psd_confidence.svg")
def generate_calibrated_spectrogram(output_dir: str) -> None:
    """Calibrated STFT spectrogram of a nonstationary signal, in dB SPL."""
    print("Generating calibrated_spectrogram...")
    from phonometry import noise_signal, spectrogram

    fs = 16000.0
    duration = 4.0
    t = np.arange(int(fs * duration)) / fs
    p_ref = 2e-5

    # A siren sweeping 600-1200 Hz at 70 dB SPL, an impact at t = 2.5 s
    # and a pink-noise floor at 45 dB SPL, all in pascals.
    siren_rms = p_ref * 10.0 ** (70.0 / 20.0)
    phase = 2.0 * np.pi * 900.0 * t - 600.0 * np.cos(np.pi * t)
    x = siren_rms * np.sqrt(2.0) * np.cos(phase)
    rng = np.random.default_rng(9)
    n_imp = int(0.06 * fs)
    impact = rng.standard_normal(n_imp) * np.exp(
        -np.arange(n_imp) / (0.012 * fs)
    )
    x[int(2.5 * fs):int(2.5 * fs) + n_imp] += 0.4 * impact
    x += noise_signal(fs, duration, color="pink",
                      rms=p_ref * 10.0 ** (45.0 / 20.0), seed=10)

    res = spectrogram(x, fs, nperseg=1024, overlap=0.75, scaling="spectrum")
    level = 10.0 * np.log10(res.power / p_ref**2)
    vmax = float(np.ceil(level.max()))

    fig, ax = plt.subplots(figsize=(10, 6.2))
    half_hop = 0.5 * res.hop / fs
    df = float(res.frequencies[1] - res.frequencies[0])
    img = ax.imshow(
        level, cmap="magma", vmin=vmax - 55.0, vmax=vmax, aspect="auto",
        origin="lower", interpolation="nearest",
        extent=(float(res.times[0]) - half_hop,
                float(res.times[-1]) + half_hop,
                0.0, float(res.frequencies[-1]) + 0.5 * df),
    )
    fig.colorbar(img, ax=ax, label="Sound pressure level [dB SPL]")
    ax.set_ylim(0.0, 3000.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(LABEL_FREQ_HZ)
    ax.set_title("Calibrated Spectrogram in dB SPL (Bendat & Piersol)",
                 fontweight="bold", pad=12)
    ax.text(0.02, 0.965,
            "a siren, an impact and a pink-noise floor:\n"
            "every cell reads an absolute level",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="white")
    plt.tight_layout()
    save_figure(output_dir, "calibrated_spectrogram.png")
    plt.close()


def generate_zoom_fft_resolution(output_dir: str) -> None:
    """Zoom FFT resolving two tones closer than a coarse FFT bin."""
    print("Generating zoom_fft_resolution...")
    from scipy import signal as sp_signal

    from phonometry import zoom_fft

    fs = 8192.0
    t = np.arange(8192) / fs  # 1 s record: 1 Hz true resolution
    x = 0.8 * np.cos(2.0 * np.pi * 997.0 * t) + 0.5 * np.cos(
        2.0 * np.pi * 1000.0 * t
    )

    # Coarse view: a 1024-point FFT of the same record (8 Hz bins).
    w = sp_signal.get_window("hann", 1024)
    coarse = 2.0 * np.abs(np.fft.rfft(x[:1024] * w)) / np.sum(w)
    coarse_f = np.fft.rfftfreq(1024, 1.0 / fs)
    band = (coarse_f >= 950.0) & (coarse_f <= 1050.0)

    # 0.25 Hz grid: four points per record-length resolution, so the two
    # mainlobes are drawn as smooth curves with their exact peaks.
    res = zoom_fft(x, fs, 980.0, 1016.0, n_points=145)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coarse_f[band], 20.0 * np.log10(np.maximum(coarse[band], 1e-12)),
            color=COLOR_SECONDARY, marker="o",
            ms=4.0, lw=1.2, ls="--", label="1024-point FFT (8 Hz bins)")
    ax.plot(res.frequencies,
            20.0 * np.log10(np.maximum(res.amplitude, 1e-12)),
            color=COLOR_PRIMARY, lw=1.6,
            label="Zoom FFT of the same record")
    for f0 in (997.0, 1000.0):
        ax.axvline(f0, color=COLOR_FG, ls=":", lw=1.0, alpha=0.6)
    ax.set_xlim(950.0, 1050.0)
    ax.set_ylim(-70.0, 5.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Zoom FFT Resolves Tones One Coarse Bin Apart "
                 "(Bendat & Piersol)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.965,
            "997 and 1000 Hz, 3 Hz apart:\n"
            "one lump on the 8 Hz grid,\n"
            "two exact lines on the zoom grid",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "zoom_fft_resolution.svg")


def generate_tone_burst_train(output_dir: str) -> None:
    """IEC 60268-1 tone bursts: one gated burst and the repetitive train."""
    print("Generating tone_burst_train...")
    from phonometry import tone_burst

    fs = 48000.0
    single = tone_burst(fs, 5000.0, 25, pre_silence=0.001, post_silence=0.001)
    train = tone_burst(fs, 5000.0, 25, repetitions=4, repetition_rate=10.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
    t_ms = 1e3 * np.arange(single.signal.size) / single.fs
    axes[0].plot(t_ms, single.signal, color=COLOR_PRIMARY, linewidth=0.9)
    axes[0].plot(t_ms, single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[0].plot(t_ms, -single.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[0].set_title("Single 5 ms burst of 5 kHz tone (25 full periods)",
                      fontweight="bold")
    axes[0].set_xlabel("Time [ms]")
    axes[0].legend(loc="upper right", fontsize=9)

    t_s = np.arange(train.signal.size) / train.fs
    axes[1].plot(t_s, train.signal, color=COLOR_PRIMARY, linewidth=0.5)
    axes[1].plot(t_s, train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--", label="Gating envelope")
    axes[1].plot(t_s, -train.envelope, color=COLOR_SECONDARY, linewidth=1.6,
                 linestyle="--")
    axes[1].set_title("Repetitive train: 10 bursts per second (duty cycle 5 %)",
                      fontweight="bold")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    fig.suptitle("Tone-Burst Test Signal (IEC 60268-1)", fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "tone_burst_train.svg")
    plt.close()


def generate_window_functions_tradeoff(output_dir: str) -> None:
    """Window spectra with their Harris figures of merit in the legend."""
    print("Generating window_functions_tradeoff...")
    from phonometry import window_metrics

    n, oversample = 1024, 256
    cases = [
        ("boxcar", COLOR_FG, "-"),
        ("hann", COLOR_PRIMARY, "-"),
        ("hamming", COLOR_SECONDARY, "-"),
        ("blackman", COLOR_TERTIARY, "-"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for name, color, style in cases:
        res = window_metrics(name, n)
        spectrum = np.abs(np.fft.rfft(res.taps, n=n * oversample))
        level = 20.0 * np.log10(spectrum / spectrum[0])
        bins = np.arange(level.size) / oversample
        shown = bins <= 16.0
        ax.plot(bins[shown], level[shown], color=color, linestyle=style,
                linewidth=1.4, alpha=0.9,
                label=(f"{name}: ENBW {res.enbw_bins:.2f} bins, "
                       f"sidelobe {res.highest_sidelobe_db:.1f} dB"))
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(-100.0, 5.0)
    ax.set_xlabel("Frequency offset [DFT bins]")
    ax.set_ylabel("Level re main lobe [dB]")
    ax.set_title("Window Functions: The Spectral Trade-off (Harris 1978)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "window_functions_tradeoff.svg")
    plt.close()


def generate_program_loudness(output_dir: str) -> None:
    """EBU R 128 metering of a synthetic programme: M, S, I and LRA."""
    print("Generating program_loudness...")
    from scipy import signal as sp_signal

    from phonometry import program_loudness

    fs = 48000
    rng = np.random.default_rng(1770)
    # A one-minute synthetic programme: quiet ambience, two dialogue
    # passages around the -23 LUFS target, a louder music sting and a
    # fade-out. Pink-ish noise stands in for programme material.
    sections = [("ambience", -38.0, 8.0), ("dialogue", -23.0, 16.0),
                ("music", -17.0, 12.0), ("dialogue", -25.0, 16.0),
                ("fade-out", -45.0, 8.0)]
    sos = sp_signal.butter(2, 2000.0, fs=fs, output="sos")
    chunks = []
    for _, level, duration in sections:
        n = int(duration * fs)
        noise = sp_signal.sosfilt(sos, rng.standard_normal(n))
        noise /= np.sqrt(np.mean(noise**2))
        # Slow, aperiodic amplitude modulation so the momentary trace
        # breathes like real programme material.
        t = np.arange(n) / fs
        wobble = 1.0 + 0.22 * np.sin(2.0 * np.pi * 0.9 * t) \
            + 0.14 * np.sin(2.0 * np.pi * 2.83 * t + 1.0)
        chunks.append(10.0 ** (level / 20.0) * noise * wobble)
    x = np.concatenate(chunks)
    # Loudness-normalise the programme to the R 128 target of -23.0 LUFS,
    # exactly what a broadcast workflow does before delivery.
    raw = program_loudness(np.vstack([x, x]), fs)
    x *= 10.0 ** ((-23.0 - raw.integrated) / 20.0)
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.axhspan(res.lra_low, res.lra_high, color=theme_fill(COLOR_TERTIARY, ax),
               zorder=0,
               label=f"LRA = {res.loudness_range:.1f} LU (P10-P95)")
    ax.plot(res.momentary_time, res.momentary, color="#9e9e9e",
            linewidth=0.8, label="Momentary M (400 ms)")
    ax.plot(res.short_term_time, res.short_term, color=COLOR_PRIMARY,
            linewidth=2.2, label="Short-term S (3 s)")
    ax.axhline(res.integrated, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.8, label=f"Integrated I = {res.integrated:.1f} LUFS")

    # Label the programme sections along the top.
    t0 = 0.0
    for name, _, duration in sections:
        ax.text(t0 + duration / 2.0, -11.0, name, ha="center", va="top",
                fontsize=8.5, color=COLOR_FG, alpha=0.75, style="italic")
        t0 += duration
        if t0 < 60.0:
            ax.axvline(t0, color=COLOR_GRID, linewidth=0.8)

    info = [
        f"I     = {res.integrated:6.1f} LUFS",
        f"LRA   = {res.loudness_range:6.1f} LU",
        f"max M = {res.max_momentary:6.1f} LUFS",
        f"max S = {res.max_short_term:6.1f} LUFS",
        f"TPmax = {res.true_peak:6.1f} dBTP",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_xlim(0.0, 60.0)
    ax.set_ylim(-58.0, -10.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loudness [LUFS]")
    ax.set_title("Programme Loudness Metering (EBU R 128)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "program_loudness.svg")
    plt.close()


def generate_k_weighting_response(output_dir: str) -> None:
    """K-weighting magnitude frequency response (ITU-R BS.1770-5 Annex 1)."""
    print("Generating k_weighting_response...")
    from phonometry import k_weighting_response

    _fig, ax = plt.subplots(figsize=(10, 6))
    k_weighting_response().plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "k_weighting_response.svg")
    plt.close()


def generate_gcc_phat_delay(output_dir: str) -> None:
    """GCC-PHAT vs direct correlation for TDE on a colored signal pair."""
    print("Generating gcc_phat_delay...")
    from scipy import signal as sp_signal

    from phonometry import noise_signal, time_delay

    fs = 8192.0
    delay = 20  # samples -> 2.44 ms
    # Colored common signal: a Butterworth roll-off keeps some power in
    # every band (the Knapp & Carter condition for a usable PHAT phase).
    b, a = sp_signal.butter(2, 800.0 / (fs / 2.0))
    s = sp_signal.lfilter(b, a, noise_signal(fs, 4.0, color="white", seed=10))
    noise_x = noise_signal(fs, 4.0, color="white", rms=0.02, seed=11)
    noise_y = noise_signal(fs, 4.0, color="white", rms=0.02, seed=12)
    x = s + noise_x
    y = np.roll(s, delay) + noise_y

    direct = time_delay(x, y, fs, method="direct", max_delay=0.01)
    phat = time_delay(x, y, fs, method="gcc", weighting="phat",
                      nperseg=2048, max_delay=0.01)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(1e3 * direct.lags,
            direct.correlation / np.max(np.abs(direct.correlation)),
            color=COLOR_PRIMARY, linewidth=1.6,
            label="Direct cross-correlation")
    ax.plot(1e3 * phat.lags, phat.correlation, color=COLOR_SECONDARY,
            linewidth=1.6, label="GCC-PHAT")
    ax.axvline(1e3 * delay / fs, color=COLOR_FG, linestyle="--",
               linewidth=1.4, alpha=0.7, label="True delay (20 samples)")
    ax.set_xlabel("Lag [ms]")
    ax.set_ylabel("Normalized correlation")
    ax.set_title("Time-Delay Estimation: GCC-PHAT vs Direct Correlation "
                 "(Knapp & Carter)", fontweight="bold", pad=12)
    ax.set_xlim(-10.0, 10.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.965,
            "colored signal: the plain correlator smears the peak,\n"
            "PHAT prewhitens the cross-spectrum and restores it",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "gcc_phat_delay.svg")
    plt.close()


def generate_cepstrum_echo(output_dir: str) -> None:
    """Echo detection on the power cepstrum of an IR with one reflection."""
    print("Generating cepstrum_echo...")
    from scipy import signal as sp_signal

    from phonometry import echo_detection, noise_signal

    fs = 48000.0
    delay_s = 0.008  # one floor reflection 8 ms after the direct sound
    a = 0.5
    # Broadband direct sound: a band-passed click, plus the scaled echo and
    # a -80 dB noise floor.
    n = 12000
    impulse = np.zeros(n)
    impulse[0] = 1.0
    b, bb = sp_signal.butter(2, [0.004, 0.9], btype="bandpass")
    direct = sp_signal.lfilter(b, bb, impulse)
    ir = direct + a * np.roll(direct, round(delay_s * fs))
    ir += noise_signal(fs, n / fs, color="white", rms=1e-4, seed=13)

    res = echo_detection(ir, fs, min_quefrency=0.002)

    _fig, ax = plt.subplots(figsize=(10, 6))
    half = res.nfft // 2 + 1
    ax.plot(1e3 * res.quefrencies[:half], res.cepstrum[:half],
            color=COLOR_PRIMARY, linewidth=1.1, label="Power cepstrum")
    ax.axvspan(1e3 * res.search_range[0], 1e3 * res.search_range[1],
               color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Searched band")
    ax.axvline(1e3 * delay_s, color=COLOR_FG, linestyle="--", linewidth=1.3,
               alpha=0.7, label="True echo delay (8 ms)")
    ax.plot([1e3 * res.delay], [res.reflection_coefficient], "v",
            color=COLOR_SECONDARY, markersize=10,
            label="Detected peak (height = reflection a)")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(-0.3, 0.65)
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_title("Echo Detection on the Power Cepstrum (Quefrency Analysis)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.60,
            "spectral ripple of period 1/(8 ms) collapses to one\n"
            "spike at 8 ms whose height reads the reflection",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_echo.svg")
    plt.close()


def generate_envelope_spectrum(output_dir: str) -> None:
    """Envelope spectrum of an AM tone in noise: the line at fm."""
    print("Generating envelope_spectrum...")
    from phonometry import envelope_spectrum, noise_signal

    fs = 8192.0
    seconds = 4.0
    t = np.arange(int(seconds * fs)) / fs
    a0, m, fm = 1.0, 0.4, 25.0
    x = a0 * (1.0 + m * np.cos(2.0 * np.pi * fm * t)) * np.cos(
        2.0 * np.pi * 1000.0 * t
    )
    x += noise_signal(fs, seconds, color="white", rms=0.03, seed=8)

    res = envelope_spectrum(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.frequencies, res.amplitude, color=COLOR_PRIMARY,
            linewidth=1.4, label="Envelope spectrum")
    ax.axvline(fm, color=COLOR_FG, linestyle="--", linewidth=1.3, alpha=0.7,
               label="Modulation frequency (25 Hz)")
    ax.axhline(a0 * m, color=COLOR_SECONDARY, linestyle=":", linewidth=1.4,
               label=r"Exact line amplitude $A_0 m$ = 0.4")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 0.5)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Modulation amplitude")
    ax.set_title("Envelope Spectrum of an AM Tone (Bendat & Piersol 13.3)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.985, 0.70,
            "the carrier is at 1 kHz; its amplitude modulation\n"
            "appears as one line at exactly $f_m$",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "envelope_spectrum.svg")
    plt.close()


def generate_synchronous_average(output_dir: str) -> None:
    """TSA: a periodic waveform pulled from noise, and the comb filter."""
    print("Generating synchronous_average...")
    from phonometry import (
        comb_filter_response,
        noise_signal,
        time_synchronous_average,
    )

    fs = 8192.0
    period = 1.0 / 32.0  # one revolution: 256 samples at this rate
    m = 256
    n_avg = 40
    phase = np.arange((n_avg + 1) * m) / m
    periodic = (
        np.cos(2.0 * np.pi * phase)
        + 0.5 * np.cos(2.0 * np.pi * 3.0 * phase + 0.4)
        - 0.3 * np.cos(2.0 * np.pi * 6.0 * phase)
    )
    signal = periodic + noise_signal(fs, phase.size / fs, rms=0.9, seed=11)
    res = time_synchronous_average(signal, fs, period, n_averages=n_avg)
    true_one = periodic[:m]

    _fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel (a): one noisy period against the recovered average.
    t_ms = 1e3 * res.times
    ax0.plot(t_ms, signal[:m], color=COLOR_GRID, linewidth=1.0,
             label="One noisy period")
    ax0.plot(t_ms, res.period_waveform, color=COLOR_PRIMARY, linewidth=1.8,
             label=f"Average of N = {n_avg} periods")
    ax0.plot(t_ms, true_one, color=COLOR_SECONDARY, linestyle="--",
             linewidth=1.2, label="True periodic waveform")
    ax0.set_xlim(0.0, 1e3 * period)
    ax0.set_xlabel("Time [ms]")
    ax0.set_ylabel("Amplitude")
    ax0.set_title("Periodic Waveform Extracted from Noise",
                  fontweight="bold", pad=10)
    ax0.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax0.set_axisbelow(True)
    ax0.legend(loc="upper right", fontsize=8)
    ax0.text(0.02, 0.03,
             "averaging N periods lowers the asynchronous\n"
             "noise by $\\sqrt{N}$ in amplitude",
             transform=ax0.transAxes, va="bottom", ha="left", fontsize=8.5,
             color=COLOR_FG)

    # Panel (b): comb filter, node selection at 32.05 orders.
    orders = np.linspace(31.0, 33.0, 4000)
    freqs = orders / period
    c20 = comb_filter_response(freqs, period, 20)
    c32 = comb_filter_response(freqs, period, 32)
    ax1.plot(orders, c32, color=COLOR_TERTIARY, linewidth=1.2,
             label="N = 32 (power of two)")
    ax1.plot(orders, c20, color=COLOR_PRIMARY, linewidth=1.4,
             label="N = 20 (node on 32.05)")
    ax1.axvline(32.05, color=COLOR_SECONDARY, linestyle=":", linewidth=1.3,
                label="Interfering tone (32.05)")
    ax1.set_xlim(31.0, 33.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel("Frequency [orders]")
    ax1.set_ylabel("Comb filter magnitude")
    ax1.set_title("Rejecting a Tone by Choosing N (McFadden 1987)",
                  fontweight="bold", pad=10)
    ax1.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.text(0.02, 0.55,
             "N = 20 puts a node on 32.05 orders and removes\n"
             "it; the power-of-two N = 32 lets it through",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             color=COLOR_FG)

    plt.tight_layout()
    save_figure(output_dir, "synchronous_average.svg")
    plt.close()


def generate_miso_coherence(output_dir: str) -> None:
    """MISO coherence: which correlated source dominates each band."""
    print("Generating miso_coherence...")
    from scipy import signal as sp_signal

    from phonometry import miso_coherence, noise_signal

    fs = 8192.0
    seconds = 32.0
    # x1 drives a low-frequency path; x2 = 0.7*x1 + independent noise drives a
    # high-frequency path. x2 is correlated with x1, so its ORDINARY coherence
    # with the output is inflated in the low band (borrowed through x1), while
    # its PARTIAL coherence is clean once x1 is conditioned out.
    x1 = noise_signal(fs, seconds, color="white", seed=1)
    x2 = 0.7 * x1 + noise_signal(fs, seconds, color="white", seed=2)
    low = sp_signal.butter(4, 400.0, fs=fs, output="sos")
    high = sp_signal.butter(4, 1500.0, btype="high", fs=fs, output="sos")
    noise = noise_signal(fs, seconds, color="white", rms=0.05, seed=3)
    y = sp_signal.sosfilt(low, x1) + sp_signal.sosfilt(high, x2) + noise
    res = miso_coherence([x1, x2], y, fs, nperseg=2048)

    f = res.frequencies
    band = (f >= 20.0) & (f <= 4000.0)
    fb = f[band]

    _fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7.4), sharex=True)

    def db(v: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return 10.0 * np.log10(v)

    ax_top.semilogx(fb, db(res.output_psd[band]), color="gray", linewidth=1.4,
                    label="Measured output")
    floor = db(res.output_psd[band]).min() - 5.0
    for i, color in ((0, COLOR_PRIMARY), (1, COLOR_TERTIARY)):
        level = db(res.coherent_output_spectra[i][band])
        # Both contributions rise from the same floor, so they stay
        # translucent and only their opacity follows the page.
        ax_top.fill_between(fb, floor, level, color=color, lw=0.0,
                            alpha=theme_fill_alpha(color, ax_top))
        ax_top.semilogx(fb, level, color=color, linewidth=1.4,
                        label=f"Input {i + 1} contribution")
    ax_top.semilogx(fb, db(res.noise_psd[band]), color=COLOR_SECONDARY,
                    linestyle="--", linewidth=1.1, label="Residual noise")
    ax_top.set_ylim(floor, db(res.output_psd[band]).max() + 3.0)
    ax_top.set_ylabel("Coherent output [dB re 1/Hz]")
    ax_top.set_title(
        "Multiple-Input Coherence: Which Source Dominates Each Band "
        "(Bendat & Piersol Ch. 7)", fontweight="bold", pad=12)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower center", fontsize=8.5, ncol=2)

    ax_bot.semilogx(fb, res.ordinary_coherence[1][band], color=COLOR_TERTIARY,
                    linestyle=":", linewidth=1.6,
                    label=r"Input 2 ordinary $\gamma^2_{2y}$ (inflated by x1)")
    ax_bot.semilogx(fb, res.partial_coherence[1][band], color=COLOR_TERTIARY,
                    linewidth=1.8,
                    label=r"Input 2 partial $\gamma^2_{2y\cdot 1}$ (x1 removed)")
    ax_bot.semilogx(fb, res.multiple_coherence[band], color=COLOR_FG,
                    linewidth=1.4, alpha=0.8,
                    label=r"Multiple $\gamma^2_{y:x}$")
    ax_bot.set_ylim(0.0, 1.05)
    ax_bot.set_xlim(20.0, 4000.0)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_ylabel("Coherence")
    ax_bot.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_bot.set_axisbelow(True)
    ax_bot.legend(loc="center right", fontsize=8.5)
    ax_bot.text(0.015, 0.05,
                "conditioning removes the shared x1 component:\n"
                "the low-band ordinary coherence of x2 collapses",
                transform=ax_bot.transAxes, va="bottom", ha="left",
                fontsize=8.5, color=COLOR_FG)
    for ax in (ax_top, ax_bot):
        format_frequency_axis(ax, 20.0, 4000.0)
    plt.tight_layout()
    save_figure(output_dir, "miso_coherence.svg")
    plt.close()


def generate_trend_test(output_dir: str) -> None:
    """Reverse arrangement trend test: B&P Example 4.4 vs a rising drift."""
    print("Generating trend_test...")
    from phonometry import trend_test

    # B&P Example 4.4: twenty observations, A = 86, accepted (no trend).
    example = np.array([
        5.2, 6.2, 3.7, 6.4, 3.9, 4.0, 3.9, 5.3, 4.0, 4.6,
        5.9, 6.5, 4.3, 5.7, 3.1, 5.6, 5.2, 3.9, 6.2, 5.0,
    ])
    # The same fluctuations with a slow upward drift: fewer reverse
    # arrangements (A = 38, below the Table A.6 lower bound of 64), rejected.
    drifting = example + np.linspace(0.0, 4.0, example.size)
    res_flat = trend_test(example)
    res_drift = trend_test(drifting)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, example.size + 1)
    ax.plot(index, res_flat.values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="B&P Example 4.4: A = 86, accepted (no trend)")
    ax.plot(index, res_drift.values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="Added rising drift: A = 38, rejected (trend)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sequence value")
    ax.set_title("Nonparametric Trend Test by Reverse Arrangements (B&P 4.5.2)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 observations; the count A of pairs i < j with x[i] > x[j]\n"
            "must fall in (64, 125] at the 5 % level (Table A.6). A rising\n"
            "trend depresses A below the acceptance region",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "trend_test.svg")
    plt.close()


def generate_stationarity_test(output_dir: str) -> None:
    """Reverse arrangement stationarity test: steady noise vs a gain ramp."""
    print("Generating stationarity_test...")
    from phonometry import stationarity_test

    fs = 8192.0
    n = 1 << 16
    steady = np.random.default_rng(42).standard_normal(n)
    # The B&P Example 10.3 scenario: the same noise with a slow +20 % gain
    # increase over the record.
    ramp = np.random.default_rng(42).standard_normal(n) * np.linspace(
        1.0, 1.2, n
    )
    res_steady = stationarity_test(steady, fs)
    res_ramp = stationarity_test(ramp, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(1, res_steady.n_segments + 1)
    ax.plot(index, res_steady.segment_values, "o-", color=COLOR_PRIMARY,
            linewidth=1.2, markersize=5,
            label="Steady noise: A = 91, accepted (stationary)")
    ax.plot(index, res_ramp.segment_values, "s-", color=COLOR_SECONDARY,
            linewidth=1.2, markersize=5,
            label="+20 % gain ramp: A = 7, rejected (nonstationary)")
    ax.set_xticks(index[1::2])
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Segment mean square")
    ax.set_title("Stationarity Test by Reverse Arrangements (B&P 10.3.1.1)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.02, 0.80,
            "20 segment mean squares; the count A of pairs i < j with\n"
            "x[i] > x[j] must fall in (64, 125] at the 5 % level (Table A.6)",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "stationarity_test.svg")
    plt.close()


def _bandlimited_gaussian_figure_record(
    seed: int, fs: float, n: int, f1: float, f2: float
) -> np.ndarray:
    """Exactly bandlimited unit-variance Gaussian noise (FFT synthesis)."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = rng.standard_normal(freqs.size) + 1j * rng.standard_normal(
        freqs.size
    )
    spec[(freqs < f1) | (freqs > f2)] = 0.0
    x = np.fft.irfft(spec, n)
    return np.asarray(x / np.std(x))


def generate_rice_level_crossings(output_dir: str) -> None:
    """Measured level-crossing rates against the Rice curve."""
    print("Generating rice_level_crossings...")
    from phonometry import level_crossing_rate

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(0, fs, 1 << 19, 800.0, 1200.0)
    res = level_crossing_rate(x, fs, levels=np.linspace(-3.5, 3.5, 29))

    _fig, ax = plt.subplots(figsize=(10, 6))
    order = np.argsort(res.levels)
    ax.plot(res.levels[order], res.rice_rates[order], color=COLOR_SECONDARY,
            linewidth=1.6,
            label=r"Rice: $N_0\,\exp(-a^2/2\sigma_x^2)$ (Eq. 5.196)")
    ax.plot(res.levels, res.rates, "o", color=COLOR_PRIMARY, markersize=6,
            label="Measured crossing rate")
    ax.set_yscale("log")
    ax.set_xlabel("Level a [signal units]")
    ax.set_ylabel("Crossings per second [1/s]")
    ax.set_title("Level-Crossing Rates of Bandlimited Gaussian Noise (Rice)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.975, 0.965,
            "800-1200 Hz Gaussian band: 2014 zero crossings/s, an\n"
            r"apparent frequency $N_0/2 \approx$ 1007 Hz"
            " (B&P Example 5.13)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_level_crossings.svg")
    plt.close()


def generate_rice_peak_distribution(output_dir: str) -> None:
    """Peak-height exceedance between the Gaussian and Rayleigh limits."""
    print("Generating rice_peak_distribution...")
    from phonometry import peak_statistics
    from phonometry.metrology.random_data import _rice_peak_exceedance

    fs = 20480.0
    x = _bandlimited_gaussian_figure_record(3, fs, 1 << 19, 0.0, 2000.0)
    res = peak_statistics(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    peaks = res.peak_values
    exceedance = 1.0 - np.arange(1, peaks.size + 1) / peaks.size
    z = np.linspace(-2.5, 4.5, 400)
    ax.plot(z, _rice_peak_exceedance(z, 1.0), color=COLOR_FG, linewidth=1.0,
            linestyle="--", alpha=0.6, label="Rayleigh limit (r = 1, narrowband)")
    ax.plot(z, _rice_peak_exceedance(z, 0.0), color=COLOR_FG, linewidth=1.0,
            linestyle=":", alpha=0.6, label="Gaussian limit (r = 0, wideband)")
    ax.plot(z, res.peak_exceedance(z), color=COLOR_SECONDARY, linewidth=1.7,
            label="Rice mixture at r = 0.746 (Eq. 5.223)")
    ax.plot(peaks, exceedance, drawstyle="steps-post", color=COLOR_PRIMARY,
            linewidth=1.2, label="Empirical peak exceedance (0-2 kHz noise)")
    ax.set_yscale("log")
    ax.set_xlim(-2.5, 4.5)
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r"Standardized peak height $z = a/\sigma_x$")
    ax.set_ylabel("Prob[peak > z]")
    ax.set_title("Peak-Height Distribution and the Irregularity Factor (Rice)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.975, 0.965,
            r"low-pass noise: $r = N_0/2M = \sqrt{5}/3$;"
            " negative maxima exist,\nso the peak law sits between"
            " Gaussian and Rayleigh (B&P 5.5.4)",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "rice_peak_distribution.svg")
    plt.close()


def generate_stoi_intelligibility(output_dir: str) -> None:
    """STOI vs ESTOI over SNR for stationary and modulated maskers."""
    print("Generating stoi_intelligibility...")
    from phonometry import stoi

    fs = 10000  # the STOI internal rate: no resampling, faster and exact
    rng = np.random.default_rng(20)
    t = np.arange(3 * fs) / fs

    # A speech-like clean signal: amplitude-modulated formant-ish tones.
    clean = np.zeros_like(t)
    for f0 in (200.0, 400.0, 700.0, 1100.0, 1800.0, 2600.0):
        depth = 0.5 * (1.0 + np.sin(2 * np.pi * rng.uniform(2.0, 6.0) * t
                                    + rng.uniform(0.0, 2 * np.pi)))
        clean += depth * np.sin(2 * np.pi * f0 * t + rng.uniform(0.0, 2 * np.pi))
    p_clean = float(np.sqrt(np.mean(clean**2)))

    # Two maskers: a stationary Gaussian noise and the same noise deeply gated
    # at 5 Hz (a modulated masker with quiet gaps, as in Jensen & Taal 2016).
    base_noise = rng.standard_normal(clean.size)
    gate = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 5.0 * t)))  # 0/1 square gate
    modulated = base_noise * (0.05 + 0.95 * gate)

    snrs = np.arange(-15.0, 20.1, 5.0)

    def curve(masker: np.ndarray, extended: bool) -> np.ndarray:
        p_m = float(np.sqrt(np.mean(masker**2)))
        out = []
        for snr in snrs:
            g = p_clean / (p_m * 10.0 ** (snr / 20.0))
            out.append(stoi(clean, clean + g * masker, fs, extended=extended).value)
        return np.asarray(out)

    fig, (ax_stoi, ax_estoi) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, extended, title in (
        (ax_stoi, False, "STOI (Taal et al. 2011)"),
        (ax_estoi, True, "ESTOI (Jensen & Taal 2016)"),
    ):
        ax.plot(snrs, curve(base_noise, extended), "o-", color=COLOR_PRIMARY,
                linewidth=1.7, label="Stationary masker")
        ax.plot(snrs, curve(modulated, extended), "s--", color=COLOR_SECONDARY,
                linewidth=1.7, label="Modulated (5 Hz gated) masker")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("SNR [dB]")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper left", fontsize=9)
    ax_stoi.set_ylabel("Intelligibility index")
    ax_estoi.text(0.985, 0.03,
                  "ESTOI rates the modulated masker higher: it credits the\n"
                  "speech glimpsed in the quiet gaps. STOI barely separates them.",
                  transform=ax_estoi.transAxes, va="bottom", ha="right",
                  fontsize=8.5, color=COLOR_FG)
    fig.suptitle("Short-Time Objective Intelligibility: STOI vs ESTOI",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "stoi_intelligibility.svg")
    plt.close()


def generate_image_source_reflectogram(output_dir: str) -> None:
    """Image-source reflectogram: the synthetic RIR as reflections by order."""
    print("Generating image_source_reflectogram...")
    from phonometry import image_source_rir

    dims = (7.0, 5.0, 3.0)
    res = image_source_rir(dims, (2.0, 1.6, 1.5), (5.2, 3.4, 1.7),
                           0.12, fs=48000, max_order=10)
    times = np.asarray(res.times) * 1e3  # ms
    amp = np.asarray(res.amplitudes)
    orders = np.asarray(res.orders)
    direct = float(np.max(np.abs(amp)))
    level = 20.0 * np.log10(np.maximum(np.abs(amp), 1e-30) / direct)
    dist = np.asarray(res.distances)
    d0 = float(dist[int(np.argmin(dist))])
    envelope = 20.0 * np.log10(np.maximum(d0 / dist, 1e-30))

    window = times <= 120.0
    fig, ax = plt.subplots(figsize=(10, 6))
    env_sort = np.argsort(times[window])
    ax.plot(times[window][env_sort], envelope[window][env_sort],
            color=COLOR_GRID, linestyle="--", linewidth=1.2,
            label=r"$1/r$ spreading envelope", zorder=1)
    is_direct = orders == 0
    refl = window & ~is_direct
    sc = ax.scatter(times[refl], level[refl], c=orders[refl], cmap="viridis",
                    s=18, alpha=0.85, zorder=3,
                    label="Reflections (image sources)")
    d = window & is_direct
    ax.vlines(times[d], -60.0, level[d], color=COLOR_PRIMARY, linewidth=2.0,
              zorder=4)
    ax.plot(times[d], level[d], "o", color=COLOR_PRIMARY, ms=9, zorder=5,
            label="Direct sound (order 0)")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Reflection order")
    ax.set_xlabel("Arrival time [ms]")
    ax.set_ylabel("Reflection level re direct [dB]")
    ax.set_title("Image-Source Room Impulse Response: a 7x5x3 m room "
                 "(order <= 10)", fontweight="bold", pad=12)
    ax.set_xlim(0.0, 120.0)
    ax.set_ylim(-60.0, 5.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.015, 0.03,
            "each reflection is a mirror image of the source;\n"
            "amplitude = product of wall reflection factors / (4 pi r)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "image_source_reflectogram.svg")
    plt.close()


def generate_ship_source_level(output_dir: str) -> None:
    """Ship equivalent monopole source level and the ΔL surface correction."""
    print("Generating ship_source_level...")
    from phonometry import monopole_source_level

    # One-third-octave centres 20 Hz-20 kHz and a plausible broadband ship RNL
    # that rolls off with frequency; draught 6 m -> source depth 4.2 m.
    freqs = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
                      400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
                      4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000],
                     dtype=float)
    rnl = 175.0 - 12.0 * np.log10(freqs / 20.0)
    res = monopole_source_level(rnl, freqs, draught=6.0)

    _fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.semilogx(freqs, res.source_level, "o-", color=COLOR_PRIMARY, linewidth=2.0,
                markersize=4, label="Source level Ls")
    ax.semilogx(freqs, res.radiated_noise_level, "s--", color=COLOR_SECONDARY,
                linewidth=1.6, markersize=3, alpha=0.8, label="Radiated noise level")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB re 1 µPa·m]")
    ax.set_title("Ship Equivalent Monopole Source Level (ISO 17208-2)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.semilogx(freqs, res.surface_correction, ":", color=COLOR_TERTIARY,
                  linewidth=2.0, label="Surface correction ΔL")
    twin.set_ylabel("Surface correction ΔL [dB]")
    # After twinx() re-initialises the shared x-axis with the default log
    # locator, so the octave-band labelling is not reset to 10^n ticks.
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, loc="lower left", fontsize=9)

    info = [
        "Ls = LRN + ΔL",
        "ΔL = -10 lg[(2u^4+14u^2)/(14+2u^2+u^4)]",
        "u = k d_s,  d_s = 0.7 D = 4.2 m",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8.5, color=COLOR_FG,
            family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "ship_source_level.svg")
    plt.close()


def generate_pile_driving(output_dir: str) -> None:
    """Pile-driving strike waveform, single-strike SEL and cumulative-SEL growth."""
    print("Generating pile_driving...")
    from phonometry import cumulative_sel_identical, pile_strike_metrics

    fs = 48000
    dur = 0.3
    t = np.arange(int(dur * fs)) / fs
    # An impulsive strike: a short rise then an exponentially decaying ring.
    envelope = np.where(t < 0.01, t / 0.01, np.exp(-(t - 0.01) / 0.04))
    pressure = 8000.0 * envelope * np.sin(2.0 * np.pi * 180.0 * t)
    res = pile_strike_metrics(pressure, fs)

    # Cumulative SEL growth over a driving sequence of identical strikes.
    strikes = np.arange(1, 2001)
    sel_cum = np.array([cumulative_sel_identical(res.single_strike_sel, int(n))
                        for n in strikes])

    _fig, (ax_w, ax_c) = plt.subplots(
        2, 1, figsize=(10, 7.2),
        gridspec_kw={"height_ratios": [1.4, 1.0]})
    ax_w.plot(t * 1e3, pressure, color=COLOR_PRIMARY, linewidth=0.8)
    peak_idx = int(np.argmax(np.abs(pressure)))
    ax_w.plot([t[peak_idx] * 1e3], [pressure[peak_idx]], "o", color=COLOR_SECONDARY,
              markersize=8, label=f"Peak = {res.peak_spl:.0f} dB re 1 µPa")
    ax_w.set_xlabel("Time [ms]")
    ax_w.set_ylabel("Pressure [Pa]")
    ax_w.set_title("Percussive Pile-Driving Strike (ISO 18406)",
                   fontweight="bold", pad=12)
    ax_w.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_w.set_axisbelow(True)
    ax_w.legend(loc="upper right", fontsize=9)

    ax_c.semilogx(strikes, sel_cum, color=COLOR_TERTIARY, linewidth=2.2)
    ax_c.set_xlabel("Number of strikes N")
    ax_c.set_ylabel("Cumulative SEL [dB re 1 µPa²·s]")
    ax_c.set_title(
        f"SEL_ss = {res.single_strike_sel:.0f} dB;  "
        f"SEL_cum = SEL_ss + 10 lg(N)", fontsize=10)
    ax_c.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_c.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "pile_driving.svg")
    plt.close()


def generate_epnl(output_dir: str) -> None:
    """ICAO aircraft-flyover EPNL: PNL/PNLT time history with the 10 dB-down window."""
    print("Generating epnl...")
    from phonometry import NOY_BANDS, effective_perceived_noise_level

    k = 41
    dt = 0.5
    idx = np.arange(k)
    # Broadband flyover spectrum with a mid-frequency emphasis, modulated by a
    # Gaussian overall-level envelope; a fan tone in the 2500 Hz band adds a
    # tone correction near the closest-point-of-approach.
    shape = 15.0 * np.exp(-((np.log10(NOY_BANDS) - np.log10(400.0)) ** 2) / 0.5)
    gain = 30.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 5.0**2)) - 5.0
    spectra = (55.0 + shape)[None, :] + gain[:, None]
    spectra[:, 17] += 12.0 * np.exp(-((idx - 20.0) ** 2) / (2 * 6.0**2))
    res = effective_perceived_noise_level(spectra, dt)
    kf, kl = res.band_limits

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(res.times[kf], res.times[kl], color=COLOR_TERTIARY, alpha=0.15,
               label="10 dB-down window")
    ax.plot(res.times, res.pnl, color="#8c8c8c", linestyle="--", linewidth=1.4,
            label="PNL")
    ax.plot(res.times, res.pnlt, color=COLOR_PRIMARY, linewidth=2.2, label="PNLT")
    km = int(np.argmax(res.pnlt))
    ax.plot([res.times[km]], [res.pnltm], "o", color=COLOR_SECONDARY, markersize=9,
            label=f"PNLTM = {res.pnltm:.1f} PNdB")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Level [PNdB]")
    ax.set_title(
        "ICAO Aircraft Flyover — Effective Perceived Noise Level (Annex 16)",
        fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"EPNL = {res.epnl:.1f} EPNdB\nD = {res.duration_correction:+.1f} dB",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "epnl.svg")
    plt.close()


def generate_wind_turbine_tonality(output_dir: str) -> None:
    """IEC 61400-11 wind-turbine tonal audibility: narrowband spectrum + masking."""
    print("Generating wind_turbine_tonality...")
    from phonometry import wind_turbine_tonality
    from phonometry.environmental.wind_turbine_noise import _critical_band_edges

    # A narrowband spectrum: a shaped broadband floor with a blade-passing-style
    # tone near 200 Hz, at 2 Hz resolution.
    df = 2.0
    freqs = np.arange(50.0, 400.0 + df, df)
    floor = 42.0 - 6.0 * np.log10(freqs / 100.0)
    tone_bin = int(np.argmin(np.abs(freqs - 200.0)))
    levels = floor.copy()
    levels[tone_bin] += 22.0
    res = wind_turbine_tonality(levels, freqs, tone_frequency=200.0)

    _fig, ax = plt.subplots(figsize=(10, 6))
    band_lo, band_hi = _critical_band_edges(res.tone_frequency)
    ax.axvspan(band_lo, band_hi, color=COLOR_TERTIARY, alpha=0.15,
               label="Critical band")
    ax.plot(freqs, levels, color=COLOR_PRIMARY, linewidth=1.0,
            label="Narrowband spectrum")
    ax.axhline(res.masking_level, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"Masking level = {res.masking_level:.1f} dB")
    ax.plot([res.tone_frequency], [res.tone_level], "o", color=COLOR_SECONDARY,
            markersize=9, label=f"Tone = {res.tone_level:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.set_title("Wind-Turbine Tonal Audibility (IEC 61400-11)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            f"Tonal audibility ΔLₐ = {res.tonal_audibility:.1f} dB\n"
            f"{'audible' if res.is_audible else 'not audible'}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "wind_turbine_tonality.svg")
    plt.close()


def generate_underwater_transmission_loss(output_dir: str) -> None:
    """Underwater TL vs range: geometrical spreading + volume absorption."""
    print("Generating underwater_transmission_loss...")
    from phonometry import transmission_loss

    ranges = np.linspace(10.0, 20_000.0, 400)
    res = transmission_loss(
        ranges, 10_000.0, law="practical", transition_range=1000.0,
        temperature=10.0, salinity=35.0, depth=100.0, model="francois-garrison",
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.range_m, res.tl, color=COLOR_PRIMARY, linewidth=2.0,
            label="Total transmission loss")
    ax.plot(res.range_m, res.spreading, color="#8c8c8c", linestyle="--", linewidth=1.4,
            label="Geometrical spreading")
    ax.plot(res.range_m, res.absorption, color=COLOR_SECONDARY, linestyle=":", linewidth=1.6,
            label="Volume absorption")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Underwater Transmission Loss (Francois–Garrison)",
                 fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.02, 0.05,
            f"f = 10 kHz, α = {res.absorption_coefficient:.2f} dB/km\n"
            "practical spreading (R₀ = 1000 m)",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "underwater_transmission_loss.svg")
    plt.close()


def generate_weston_regimes(output_dir: str) -> None:
    """Weston's four shallow-water propagation regimes and their boundaries."""
    print("Generating weston_regimes...")
    from phonometry import weston_propagation_loss

    # A 50 m shallow-water site over medium sand at 250 Hz, the frequency and
    # sediment pair Ainslie uses to illustrate the transition (Figure 9.7).
    ranges = np.logspace(1.0, 5.3, 500)
    res = weston_propagation_loss(ranges, 250.0, 50.0, seabed="sand",
                                  source_depth=10.0, receiver_depth=25.0)
    bounds = res.boundaries
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xscale("log")
    # Each law is drawn from a third of a decade before it takes over: an
    # asymptotic form extrapolated far outside its own regime is meaningless
    # (the single-mode formula would sit below free field at 10 m).
    for curve, onset, label, color, style in (
        (res.spherical, 0.0, "Spherical, 20 lg r", "#8c8c8c", ":"),
        (res.cylindrical, bounds.spherical_to_cylindrical,
         "Cylindrical, 10 lg r", COLOR_SECONDARY, "--"),
        (res.mode_stripping, bounds.cylindrical_to_mode_stripping,
         "Mode stripping, 15 lg r", COLOR_TERTIARY, "-."),
        (res.single_mode, bounds.mode_stripping_to_single_mode,
         "Single mode", "#9467bd", (0, (3, 1, 1, 1))),
    ):
        shown = np.where(res.range_m >= onset / 3.0, curve, np.nan)
        ax.plot(res.range_m, shown, linestyle=style, linewidth=1.3, color=color, label=label)
    ax.plot(res.range_m, res.propagation_loss, color=COLOR_PRIMARY, linewidth=2.6,
            label="Composite propagation loss")
    for boundary, name in (
        (bounds.spherical_to_cylindrical, "H/2ψc"),
        (bounds.cylindrical_to_mode_stripping, "r_CS"),
        (bounds.mode_stripping_to_single_mode, "r_MS"),
    ):
        ax.axvline(boundary, color=COLOR_SECONDARY, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.annotate(name, xy=(boundary, 22.0), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=COLOR_SECONDARY)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Propagation loss [dB re 1 m²]")
    ax.set_title("Weston Shallow-Water Propagation Regimes (Ainslie §9.1.1.2)",
                 fontweight="bold", pad=12)
    ax.set_ylim(130.0, 18.0)
    ax.set_xlim(float(ranges[0]), float(ranges[-1]))
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05,
            f"f = 250 Hz, H = 50 m, medium sand\n"
            f"ψc = {np.degrees(bounds.critical_angle):.1f}°, "
            f"η = {bounds.reflection_loss_gradient:.2f} Np/rad, "
            f"{bounds.mode_count:.0f} modes",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "weston_regimes.svg")
    plt.close()


def generate_marine_mammal_weighting(output_dir: str) -> None:
    """NMFS 2024 auditory weighting functions for the five in-water groups."""
    print("Generating marine_mammal_weighting...")
    from phonometry import auditory_weighting, exposure_criteria

    freqs = np.logspace(1.0, 5.4, 700)
    groups = (
        ("LF", "Low-frequency cetaceans", COLOR_PRIMARY, "-"),
        ("HF", "High-frequency cetaceans", COLOR_SECONDARY, "--"),
        ("VHF", "Very high-frequency cetaceans", COLOR_TERTIARY, "-."),
        ("PW", "Phocid pinnipeds (water)", "#9467bd", ":"),
        ("OW", "Otariid pinnipeds (water)", "#8c564b", (0, (3, 1, 1, 1))),
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    for group, label, color, style in groups:
        res = auditory_weighting(freqs, group, guidance="nmfs-2024")
        crit = exposure_criteria(group, guidance="nmfs-2024", impulsive=True)
        ax.semilogx(res.frequencies, res.weighting, color=color, linestyle=style,
                    linewidth=2.0,
                    label=f"{label} (AUD INJ {crit.injury_sel:.0f} dB)")
    ax.axhline(0.0, color="#8c8c8c", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Weighting amplitude W(f) [dB]")
    ax.set_title("Marine-Mammal Auditory Weighting (NMFS 2024, v3.0)",
                 fontweight="bold", pad=12)
    ax.set_ylim(-75.0, 5.0)
    ax.set_xlim(10.0, 250e3)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9, ncol=2, framealpha=0.92)
    plt.tight_layout()
    save_figure(output_dir, "marine_mammal_weighting.svg")
    plt.close()


def generate_underwater_sound_speed(output_dir: str) -> None:
    """Sea-water sound-speed profile (UNESCO): mixed layer, thermocline, deep channel."""
    print("Generating underwater_sound_speed...")
    from phonometry import sound_speed_profile

    depths = np.linspace(0.0, 3000.0, 121)
    # A warm mixed layer (18 °C to 80 m), a thermocline down to 4 °C at 1000 m,
    # then an isothermal deep layer; the pressure term then lifts c with depth.
    temps = 4.0 + 14.0 / (1.0 + (np.maximum(depths - 80.0, 0.0) / 250.0) ** 2)
    prof = sound_speed_profile(depths, temps, 35.0, model="unesco")
    axis_depth = depths[int(np.argmin(prof.sound_speed))]
    _fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(prof.sound_speed, prof.depth, color=COLOR_PRIMARY, linewidth=2.0,
            label="UNESCO sound speed")
    ax.axhline(axis_depth, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
               label="Sound-channel axis")
    ax.set_xlabel("Sound speed [m/s]")
    ax.set_ylabel("Depth [m]")
    ax.set_title("Sea-Water Sound-Speed Profile (UNESCO)",
                 fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "underwater_sound_speed.svg")
    plt.close()


def generate_sonar_equation(output_dir: str) -> None:
    """Passive sonar equation: signal excess vs transmission loss."""
    print("Generating sonar_equation...")
    from phonometry import passive_sonar_equation

    tl = np.linspace(40.0, 120.0, 400)
    res = passive_sonar_equation(140.0, tl, 60.0, directivity_index=15.0,
                                 detection_threshold=8.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.transmission_loss, res.signal_excess, color=COLOR_PRIMARY, linewidth=2.0,
            label="Signal excess")
    ax.axhline(0.0, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
               label="Detection limit (SE = 0)")
    ax.axvline(res.figure_of_merit, color="#8c8c8c", linestyle=":", linewidth=1.6,
               label="Figure of merit")
    ax.set_xlabel("Transmission loss [dB]")
    ax.set_ylabel("Signal excess [dB]")
    ax.set_title("Passive Sonar Equation", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05,
            f"SL = 140, NL = 60, DI = 15, DT = 8 dB\n"
            f"figure of merit = {res.figure_of_merit:.1f} dB",
            transform=ax.transAxes, va="bottom", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "sonar_equation.svg")
    plt.close()


def generate_seabed_reflection(output_dir: str) -> None:
    """Seabed reflection loss vs grazing angle, marking the critical angle."""
    print("Generating seabed_reflection...")
    from phonometry import bottom_reflection_loss

    phi = np.linspace(0.0, 90.0, 361)
    res = bottom_reflection_loss(phi, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.grazing_angle, res.reflection_loss, color=COLOR_PRIMARY, linewidth=2.0,
            label="Bottom loss (sand)")
    if res.critical_angle is not None:
        ax.axvline(res.critical_angle, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
                   label=f"Critical angle ({res.critical_angle:.1f}°)")
    ax.set_xlabel("Grazing angle [°]")
    ax.set_ylabel("Bottom loss [dB]")
    ax.set_title("Seabed Reflection Loss (Rayleigh)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.95,
            "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "seabed_reflection.svg")
    plt.close()


def generate_seabed_reflection_coefficient(output_dir: str) -> None:
    """Seabed reflection-coefficient magnitude |R| vs grazing angle."""
    print("Generating seabed_reflection_coefficient...")
    from phonometry import seabed_reflection

    phi = np.linspace(0.0, 90.0, 361)
    res = seabed_reflection(phi, rho1=1000.0, c1=1500.0, rho2=1900.0, c2=1650.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(res.grazing_angle, res.magnitude, color=COLOR_PRIMARY, linewidth=2.0,
            label="Reflection coefficient magnitude |R| (sand)")
    if res.critical_angle is not None:
        ax.axvline(res.critical_angle, color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
                   label=f"Critical angle ({res.critical_angle:.1f}°)")
    ax.set_xlabel("Grazing angle [°]")
    ax.set_ylabel("Reflection coefficient magnitude |R|")
    ax.set_title("Seabed Reflection Coefficient (Rayleigh)", fontweight="bold", pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.02, 0.95,
            "Water ρ = 1000, c = 1500\nSand ρ = 1900, c = 1650",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "seabed_reflection_coefficient.svg")
    plt.close()


def generate_ocean_ambient_noise(output_dir: str) -> None:
    """Wenz ambient-noise curves: wind + thermal energy sum vs frequency."""
    print("Generating ocean_ambient_noise...")
    from phonometry.underwater import ocean_ambient_noise

    freqs = np.logspace(2, 5.5, 300)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # Label the wind/thermal components only once to avoid repeated legend rows.
    for i, (u, color) in enumerate(((5.0, COLOR_SECONDARY), (20.0, COLOR_PRIMARY))):
        res = ocean_ambient_noise(freqs, wind_speed_knots=u)
        _plot_ambient_curve(res, u, color, label_components=(i == 0))
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Spectrum level [dB re 1 µPa²/Hz]")
    ax.set_title("Ocean Ambient Noise (Wenz)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ocean_ambient_noise.svg")
    plt.close()


def _plot_ambient_curve(res: object, wind_speed: float, color: str,
                        label_components: bool = False) -> None:
    ax = plt.gca()
    ax.plot(res.frequency, res.spectrum_level, color=color, linewidth=2.0,  # type: ignore[attr-defined]
            label=f"Total ({wind_speed:.0f} kn)")
    ax.plot(res.frequency, res.wind, color=color, linewidth=1.0, linestyle="--", alpha=0.6,  # type: ignore[attr-defined]
            label="Wind" if label_components else None)
    ax.plot(res.frequency, res.thermal, color="#8c8c8c", linewidth=1.0, linestyle=":", alpha=0.8,  # type: ignore[attr-defined]
            label="Thermal" if label_components else None)


def generate_ship_traffic_noise(output_dir: str) -> None:
    """JOMOPANS-ECHO ship source-level spectra for three vessel classes."""
    print("Generating ship_traffic_noise...")
    from phonometry import ship_source_spectrum

    _fig, ax = plt.subplots(figsize=(10, 6))
    cases = (
        ("containership", 18.0, 300.0, COLOR_PRIMARY),
        ("cruise", 17.1, 250.0, COLOR_SECONDARY),
        ("tug", 3.7, 30.0, "#8c8c8c"),
    )
    for vessel_class, speed, length, color in cases:
        s = ship_source_spectrum(speed, length, vessel_class=vessel_class)
        ax.plot(s.frequency, s.source_psd, color=color, linewidth=2.0,
                label=f"{vessel_class} ({speed:.0f} kn, {length:.0f} m)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Source spectral density [dB re 1 µPa²/Hz at 1 m]")
    ax.set_title("Ship Traffic Source Level (JOMOPANS-ECHO)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    _sf = np.asarray(s.frequency, dtype=float)
    format_frequency_axis(ax, float(_sf.min()), float(_sf.max()))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ship_traffic_noise.svg")
    plt.close()


def generate_numerical_propagation(output_dir: str) -> None:
    """Three numerical solvers: ray paths, the PE field and modes-vs-PE loss."""
    print("Generating numerical_propagation...")
    from phonometry import normal_modes, parabolic_equation, ray_trace

    # A Munk deep-water sound-speed profile for the ray / PE panels.
    zprof = np.linspace(0.0, 5000.0, 60)
    eta = 2.0 * (zprof - 1300.0) / 1300.0
    cprof = 1500.0 * (1.0 + 0.00737 * (eta - 1.0 + np.exp(-eta)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Ray trace through the Munk profile (vectorised over rays).
    rt = ray_trace(zprof, cprof, source_depth=1000.0,
                   launch_angles_deg=np.linspace(-12.0, 12.0, 13),
                   max_range=100_000.0, n_steps=6000)
    for i in range(rt.ranges.shape[0]):
        axes[0].plot(rt.ranges[i] / 1000.0, rt.depths[i], color=COLOR_PRIMARY,
                     linewidth=0.6, alpha=0.7)
    axes[0].plot([0.0], [1000.0], "o", color=COLOR_SECONDARY, label="Source")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Range [km]")
    axes[0].set_ylabel("Depth [m]")
    axes[0].set_title("Ray trace (Munk profile)", fontweight="bold")
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) Parabolic-equation transmission-loss field for the same environment;
    # imshow renders a single raster image (the figure is a raster WebP).
    pe_field = parabolic_equation(50.0, zprof, cprof, source_depth=1000.0,
                                  max_range=100_000.0, range_step=25.0,
                                  n_depth_points=1024)
    tl = pe_field.transmission_loss
    vmax = float(np.percentile(tl[np.isfinite(tl)], 95))
    tl = np.where(np.isfinite(tl), tl, vmax)  # clip the infinite zero-range column
    img = axes[1].imshow(tl, cmap="viridis_r", vmin=vmax - 50.0, vmax=vmax,
                         aspect="auto", origin="upper", interpolation="bilinear",
                         extent=(0.0, 100.0, float(zprof[-1]), 0.0))
    fig.colorbar(img, ax=axes[1], label="Transmission loss [dB]")
    axes[1].set_xlabel("Range [km]")
    axes[1].set_ylabel("Depth [m]")
    axes[1].set_title("Parabolic equation (50 Hz)", fontweight="bold")

    # (c) Transmission loss vs range: modes and PE agree for a shallow gradient.
    r = np.linspace(100.0, 20_000.0, 400)
    nm = normal_modes(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                      receiver_depth=120.0, ranges_m=r, n_depth_points=800)
    pe = parabolic_equation(50.0, [0.0, 200.0], [1500.0, 1530.0], source_depth=30.0,
                            max_range=20_000.0, range_step=20.0, n_depth_points=512)
    zi = int(np.argmin(np.abs(pe.depths - 120.0)))
    axes[2].plot(nm.ranges / 1000.0, nm.transmission_loss, color=COLOR_PRIMARY,
                 linewidth=1.0, label="Normal modes")
    axes[2].plot(pe.ranges / 1000.0, pe.transmission_loss[zi], color=COLOR_SECONDARY,
                 linewidth=0.8, alpha=0.7, label="Parabolic equation")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Range [km]")
    axes[2].set_ylabel("Transmission loss [dB]")
    axes[2].set_title("Modes vs PE (50 Hz, z = 120 m)", fontweight="bold")
    axes[2].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "numerical_propagation.png")
    plt.close()


def generate_aircraft_atmospheric_absorption(output_dir: str) -> None:
    """SAE ARP 5534 band vs pure-tone mid-band atmospheric attenuation."""
    print("Generating aircraft_atmospheric_absorption...")
    from phonometry import sae_band_attenuation

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)  # 50 Hz - 10 kHz thirds
    _fig, ax = plt.subplots(figsize=(10, 6))
    for s, color in ((1000.0, COLOR_SECONDARY), (7620.0, COLOR_PRIMARY)):
        res = sae_band_attenuation(freqs, s, temperature=25.0, relative_humidity=70.0)
        ax.plot(res.frequency, res.band_attenuation, color=color, linewidth=2.0,
                marker="o", markersize=3, label=f"SAE band ({s:.0f} m)")
        ax.plot(res.frequency, res.midband_attenuation, color=color, linewidth=1.0,
                linestyle="--", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Attenuation [dB]")
    ax.set_title("Aircraft Atmospheric Absorption (SAE ARP 5534)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.5, 0.95, "25 °C, 70% RH\nsolid: SAE band, dashed: pure-tone mid-band",
            transform=ax.transAxes, va="top", ha="center", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "aircraft_atmospheric_absorption.svg")
    plt.close()


def generate_airport_noise(output_dir: str) -> None:
    """ECAC Doc 29 noise-power-distance interpolation for two power settings."""
    print("Generating airport_noise...")
    from phonometry import npd_curve

    # A schematic NPD table (SEL vs slant distance) for two thrust settings.
    powers = [12000.0, 20000.0]
    distances = [200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0, 6300.0, 10000.0]
    levels = [
        [98.5, 92.0, 88.2, 83.6, 76.8, 69.4, 63.9, 56.8],
        [107.2, 100.9, 97.2, 92.7, 86.0, 78.5, 72.9, 65.6],
    ]
    _fig, ax = plt.subplots(figsize=(10, 6))
    for p, color in ((20000.0, COLOR_PRIMARY), (12000.0, COLOR_SECONDARY)):
        res = npd_curve(powers, distances, levels, p)
        ax.plot(res.distance, res.level, color=color, linewidth=2.0,
                label=f"P = {p:.0f} N")
        ax.plot(res.table_distances, res.table_levels, "o", color=color, markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Slant distance [m]")
    ax.set_ylabel("Event level [dB]")
    ax.set_title("Noise-Power-Distance Curves (ECAC Doc 29)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05, "markers: tabulated NPD nodes\nlines: log-linear interpolation",
            transform=ax.transAxes, va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "airport_noise.svg")
    plt.close()


def generate_airport_contour(output_dir: str) -> None:
    """ECAC Doc 29 single-event SEL contour for a departure flight path."""
    print("Generating airport_contour...")
    from phonometry import noise_contour

    powers = [8000.0, 12000.0]
    distances = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0, 7680.0]
    sel = [[98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0, 56.0],
           [104.0, 98.0, 92.0, 86.0, 80.0, 74.0, 68.0, 62.0]]
    lmax = [[94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0, 52.0],
            [100.0, 94.0, 88.0, 82.0, 76.0, 70.0, 64.0, 58.0]]
    vref = 160.0 * 0.514444
    # Departure: ground roll then climb along +x.
    xs = np.linspace(0.0, 18000.0, 40)
    z = np.clip((xs - 1500.0) * 0.11, 0.0, 2500.0)
    power = np.where(xs < 3000.0, 12000.0, 10000.0)
    path = np.column_stack([xs, np.zeros_like(xs), z, power, np.full_like(xs, vref)])
    ground_roll = xs[:-1] < 1500.0  # takeoff roll: segments still on the runway
    res = noise_contour(path, powers, distances, sel, lmax, ground_roll=ground_roll,
                        x=np.linspace(-2500.0, 20000.0, 56), y=np.linspace(-6000.0, 6000.0, 44))
    ax = res.plot()
    plt.gcf().set_size_inches(10, 5.5)
    ax.set_title("Aircraft Departure SEL Contour (ECAC Doc 29)", fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "airport_contour.png")
    plt.close()


def generate_airport_sor(output_dir: str) -> None:
    """ECAC Doc 29 start-of-roll directivity: the rearward jet/turboprop lobe."""
    print("Generating airport_sor...")
    from phonometry import start_of_roll_directivity

    dsor = 300.0  # < 762 m normalising distance: no distance de-emphasis
    # ΔSOR is defined only in the rearward arc (ψ from 90° abeam to 180° directly
    # behind, symmetric left/right), so only that half-disc is drawn. Azimuth is
    # measured clockwise from the nose; 90°/270° = abeam, 180° = directly behind.
    az = np.linspace(90.0, 270.0, 361)
    psi = np.where(az <= 180.0, az, 360.0 - az)
    jet = np.array([start_of_roll_directivity(p, dsor, "jet") for p in psi])
    prop = np.array([start_of_roll_directivity(p, dsor, "turboprop") for p in psi])

    # A polar Axes always reserves the full circle's square bounding box, which
    # wastes the upper half for a rearward half-disc. So the half-rose is drawn
    # by hand on a plain, equal-aspect Axes: the radius encodes ΔSOR offset from
    # the −16 dB origin, and the y-limits crop tightly to a wide rectangle.
    r0 = -16.0                                   # radial origin (dB)
    def _xy(a_deg: "np.ndarray | float", rr: "np.ndarray | float") -> tuple[Any, Any]:
        t = np.radians(a_deg)                    # azimuth clockwise from nose (up)
        return rr * np.sin(t), rr * np.cos(t)

    _fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for a in (90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0):  # radial spokes
        sx, sy = _xy(a, np.array([2.0, 16.0]))
        ax.plot(sx, sy, color=COLOR_FG, linestyle="--", linewidth=0.9, alpha=0.28, zorder=0)
    for g in (-4.0, -8.0, -12.0):                # inner radial grid arcs (dB)
        gx, gy = _xy(az, g - r0)
        ax.plot(gx, gy, color=COLOR_FG, linestyle="--", linewidth=0.9, alpha=0.28, zorder=0)
    # The 0 dB (abeam-reference) arc is the outer boundary: draw it solid and bold.
    bx, by = _xy(az, 0.0 - r0)
    ax.plot(bx, by, color=COLOR_FG, linestyle="-", linewidth=1.4, alpha=0.55, zorder=2)
    for a, lbl in ((90.0, "90°\nabeam"), (120.0, "120°"), (150.0, "150°"),
                   (180.0, "180° behind"), (210.0, "150°"), (240.0, "120°"),
                   (270.0, "90°\nabeam")):
        tx, ty = _xy(a, 19.4)
        ax.text(tx, ty, lbl, fontsize=9, color=COLOR_FG, ha="center", va="center")
    for data, color, label in ((jet, COLOR_PRIMARY, "Turbofan jet (Eq. 4-24a)"),
                               (prop, COLOR_SECONDARY, "Turboprop (Eq. 4-24b)")):
        dx, dy = _xy(az, data - r0)
        # The two lobes overlap and the guide arcs run under them, so this one
        # keeps a translucent fill; only its opacity follows the page.
        ax.fill(dx, dy, color=color, alpha=theme_fill_alpha(color, ax), zorder=1)
        ax.plot(dx, dy, color=color, linewidth=2.2, zorder=3, label=label)
    for g in (0.0, -4.0, -8.0, -12.0):           # radial dB labels down the centre
        ax.text(0.6, -(g - r0), f"{g:.0f}", fontsize=8, color=COLOR_FG, ha="left",
                va="center", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-21.0, 21.0)
    ax.set_ylim(-20.2, 1.5)
    ax.set_title("Start-of-Roll Directivity ΔSOR (ECAC Doc 29 §4.5.7)",
                 fontweight="bold", pad=6)
    ax.text(0.0, 1.0, "radial axis: ΔSOR [dB] relative to abeam  ·  dSOR = 300 m",
            fontsize=9, color=COLOR_FG, ha="center", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=9,
              frameon=False)
    plt.tight_layout()
    save_figure(output_dir, "airport_sor.svg")
    plt.close()


def generate_rotorcraft_ground_effect(output_dir: str) -> None:
    """ECAC Doc 32 ground-effect ΔLg vs frequency for soft vs hard ground."""
    print("Generating rotorcraft_ground_effect...")
    from phonometry import ground_effect_adjustment

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    hs, hr, dp = 150.0, 1.5, 500.0                         # overflight geometry
    grass = ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="D")
    asphalt = ground_effect_adjustment(freqs, hs, hr, dp, flow_resistivity="G")

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax.plot(freqs, asphalt, color=COLOR_PRIMARY, lw=2.0, marker="o", ms=3,
            label="Hard (asphalt/concrete, class G)")
    ax.plot(freqs, grass, color=COLOR_SECONDARY, lw=2.0, marker="s", ms=3,
            label="Soft (grass/pasture, class D)")
    ax.set_xscale("log")
    ax.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax.set_ylabel("Ground-effect adjustment ΔLg [dB]")
    ax.set_title("Rotorcraft Ground Effect (ECAC Doc 32, Chien-Soroka)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax.set_axisbelow(True)
    format_frequency_axis(ax, float(freqs.min()), float(freqs.max()))
    ax.legend(loc="lower left", fontsize=9)
    ax.text(0.98, 0.05, f"source {hs:.0f} m, receiver {hr:.1f} m, offset {dp:.0f} m",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_ground_effect.svg")
    plt.close()



def generate_rotorcraft_flyover_event(output_dir: str) -> None:
    """ECAC Doc 32 single-event LA(t) time history of a level flyover."""
    print("Generating rotorcraft_flyover_event...")
    from phonometry import RotorcraftHemisphere, rotorcraft_event_level

    # Synthetic helicopter-like hemisphere on the standard 31-band 10 deg grid:
    # low-frequency dominated spectrum with a mild forward-lobed directivity.
    freqs = 1000.0 * 10.0 ** (np.arange(-20, 11) / 10.0)     # 10 Hz-10 kHz
    az = np.arange(-90.0, 91.0, 10.0)
    po = np.arange(0.0, 181.0, 10.0)
    spectrum = (88.0 - 12.0 * np.log10(freqs / 100.0) ** 2)  # broad LF hump
    direct = -0.045 * np.abs(po - 80.0)                      # forward lobe
    levels = spectrum[None, None, :] + direct[None, :, None] \
        - 0.02 * np.abs(az)[:, None, None]
    hemisphere = RotorcraftHemisphere(freqs, az, po, levels)

    speed = 30.87                                            # 60 kt, in m/s
    t = np.arange(0.0, 130.01, 0.5)
    track = np.column_stack([np.zeros_like(t), speed * (t - 65.0),
                             np.full_like(t, 150.0)])
    event = rotorcraft_event_level(
        [hemisphere], [speed], [0.0], t, track, (120.0, 0.0),
        flow_resistivity="D")

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(event.times, event.a_levels, color=COLOR_PRIMARY, lw=2.0,
            label="Received level $L_A(t)$")
    k = int(np.argmax(event.a_levels))
    ax.plot(event.times[k], event.la_max, "o", color=COLOR_SECONDARY, ms=7,
            label=f"$L_{{ASmax}}$ = {event.la_max:.1f} dB(A)")
    window = event.a_levels >= event.la_max - 10.0
    idx = np.nonzero(window)[0]
    ax.axvspan(event.times[idx[0]], event.times[idx[-1]], zorder=0,
               color=theme_fill(COLOR_PRIMARY, ax),
               label="10 dB-down window")
    ax.axhline(event.la_max - 10.0, color=COLOR_FG, lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel("Recorded time [s]")
    ax.set_ylabel("A-weighted sound pressure level [dB(A)]")
    ax.set_title("Rotorcraft Flyover Time History (ECAC Doc 32)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.05,
            f"SEL {event.sel:.1f} dB(A)  ·  EPNL {event.epnl:.1f} EPNdB\n"
            "level flyover, 60 kt, 150 m, 120 m sideline, grass",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": COLOR_GRID, "alpha": 0.6})
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_flyover_event.svg")
    plt.close()



def generate_rotorcraft_terrain_screening(output_dir: str) -> None:
    """ECAC Doc 32 / NORAH2 terrain screening: section geometry and adjustment."""
    print("Generating rotorcraft_terrain_screening...")
    from phonometry import ground_effect_adjustment, terrain_screening_adjustment

    freqs = 1000.0 * 10.0 ** (np.arange(-13, 11) / 10.0)   # 50 Hz-10 kHz thirds
    d = np.array([0.0, 150.0, 260.0, 300.0, 340.0, 420.0, 600.0])
    z = np.array([0.0, 4.0, 48.0, 62.0, 40.0, 8.0, 2.0])
    src = (0.0, 90.0)                                       # helicopter
    rcv = (600.0, 2.0 + 1.2)                                # microphone at 1.2 m
    res = terrain_screening_adjustment(freqs, src, rcv, d, z, flow_resistivity="D")
    flat = ground_effect_adjustment(freqs, src[1], 1.2, rcv[0], flow_resistivity="D")

    _fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                  gridspec_kw={"height_ratios": [1.1, 1.0]})
    res.plot(ax=ax)   # the user-facing section geometry
    ax.set_title("Rotorcraft Terrain Screening (ECAC Doc 32 / NORAH2)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    ax2.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.5)
    ax2.plot(freqs, flat, color=COLOR_TERTIARY, lw=1.6, ls="--", marker="s", ms=3,
             label="Flat ground (no hill)")
    ax2.plot(freqs, res.adjustment, color=COLOR_PRIMARY, lw=2.0, marker="o",
             ms=3, label="Screened by the hill (Eq. 45-47)")
    ax2.set_xscale("log")
    ax2.set_xlabel("One-third-octave-band centre frequency [Hz]")
    ax2.set_ylabel("Ground and screening adjustment [dB]")
    ax2.grid(color=COLOR_GRID, linestyle="--", alpha=0.6, which="both")
    ax2.set_axisbelow(True)
    format_frequency_axis(ax2, float(freqs.min()), float(freqs.max()))
    ax2.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "rotorcraft_terrain_screening.svg")
    plt.close()


@cache
def _time_loudness_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ISO 532-3 STL(t)/LTL(t) for a 1 kHz / 60 dB burst (on 200-400 ms)."""
    from phonometry.psychoacoustics import loudness_moore_glasberg_time

    dur = 0.8
    t = np.arange(int(dur * _FS_PSY)) / _FS_PSY
    sig = np.zeros_like(t)
    on = (t >= 0.2) & (t < 0.4)
    sig[on] = _P_REF * 10.0 ** (60.0 / 20.0) * np.sqrt(2.0) * np.sin(
        2.0 * np.pi * 1000.0 * t[on]
    )
    tv = loudness_moore_glasberg_time(sig, _FS_PSY)
    return (tv.time.copy(), tv.short_term_loudness.copy(),
            tv.long_term_loudness.copy())


def generate_moore_glasberg_time_loudness(output_dir: str) -> None:
    """ISO 532-3 short-term vs long-term loudness for a 1 kHz tone burst."""
    print("Generating moore_glasberg_time_loudness.png...")
    time, stl, ltl = _time_loudness_data()

    _, ax = plt.subplots(figsize=(10, 6))
    # Shade the burst window (200-400 ms).
    ax.axvspan(0.2, 0.4, color=theme_fill(COLOR_FG, ax), linewidth=0, zorder=0)
    ax.plot(time, stl, color=COLOR_PRIMARY, linewidth=1.8,
            label=f"Short-term loudness STL (STL peak = {stl.max():.1f} sone)")
    ax.plot(time, ltl, color=COLOR_SECONDARY, linewidth=2.0,
            label=f"Long-term loudness LTL (LTL peak = {ltl.max():.1f} sone)")

    ax.annotate("1 kHz burst, 200 ms", xy=(0.3, 0.0),
                xytext=(0.3, float(stl.max()) * 1.02), fontsize=10,
                color=COLOR_FG, ha="center")
    ax.annotate("Fast attack / release",
                xy=(float(time[int(np.argmax(stl))]), float(stl.max())),
                xytext=(0.45, float(stl.max()) * 0.82), fontsize=9,
                color=COLOR_PRIMARY,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_PRIMARY})
    ax.annotate("Slow integration",
                xy=(0.55, float(np.interp(0.55, time, ltl))),
                xytext=(0.58, float(ltl.max()) * 0.55), fontsize=9,
                color=COLOR_SECONDARY,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_SECONDARY})

    ax.set_title("Time-Varying Loudness (ISO 532-3)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Loudness [sone]")
    ax.set_xlim(0, float(time[-1]))
    ax.set_ylim(0, float(stl.max()) * 1.18)
    ax.legend(loc="upper right", fontsize=9)
    save_figure(output_dir, "moore_glasberg_time_loudness.png")
    plt.close()


def generate_prediction_flanking_demo(output_dir: str) -> None:
    """EN 12354-1 simplified flanking prediction (Annex H.3 worked example)."""
    print("Generating prediction_flanking_demo.png...")
    from phonometry.building.building_prediction import (
        FlankingPath,
        flanking_element,
        predicted_airborne_insulation,
    )

    # Annex H.3 inputs: separating wall Rs,w = 57 dB, Ss = 11.5 m², four
    # flanking elements. Columns: (label, Rw, KFf, KFd=KDf, coupling length lf).
    elements = [
        ("floor", 49, 12.4, 8.9, 4.50),
        ("ceiling", 46, 14.4, 9.2, 4.50),
        ("facade", 42, 12.6, 6.7, 2.55),
        ("wall", 33, 33.5, 15.7, 2.55),
    ]
    paths: list[FlankingPath] = []
    for name, rw, k_ff, k_side, lf in elements:
        ff, df, fd = flanking_element(
            label=name, r_flanking=float(rw), r_separating=57.0,
            k_ff=k_ff, k_fd=k_side, k_df=k_side,
            separating_area=11.5, coupling_length=lf,
        )
        paths.extend((ff, df, fd))
    result = predicted_airborne_insulation(r_direct=57.0, flanking_paths=paths)

    # Sort every path (direct + 12 flanking) by its share of the transmitted
    # energy, largest first.
    contribs = sorted(result.paths, key=lambda c: c.fraction, reverse=True)
    labels = [c.label for c in contribs]
    fracs = [c.fraction * 100.0 for c in contribs]
    df_orange = "#ff7f0e"
    kind_color = {
        "Dd": COLOR_TERTIARY, "Ff": COLOR_PRIMARY,
        "Fd": COLOR_SECONDARY, "Df": df_orange,
    }
    colors = [kind_color[c.kind] for c in contribs]

    direct_share = next(c.fraction for c in result.paths if c.kind == "Dd") * 100.0
    flank_share = 100.0 - direct_share

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    bars = ax.bar(range(len(fracs)), fracs, color=colors, edgecolor=COLOR_FG,
                  linewidth=0.7, zorder=3)
    bars[0].set_linewidth(2.2)  # highlight the dominant path
    ax.annotate("dominant path", xy=(0, fracs[0]), xytext=(1.5, fracs[0] + 3.5),
                fontsize=10, fontweight="bold", color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 1.1})

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_xlabel("Transmission path")
    ax.set_ylim(0, max(fracs) + 9.0)
    ax.set_title("EN 12354-1 Flanking Transmission (Annex H.3 example)",
                 fontweight="bold", pad=12)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLOR_TERTIARY, edgecolor=COLOR_FG, label="Dd — direct"),
        Patch(facecolor=COLOR_PRIMARY, edgecolor=COLOR_FG,
              label="Ff — flanking–flanking"),
        Patch(facecolor=COLOR_SECONDARY, edgecolor=COLOR_FG,
              label="Fd — flanking–separating"),
        Patch(facecolor=df_orange, edgecolor=COLOR_FG,
              label="Df — separating–flanking"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    rw_dd = result.r_direct_w
    rpw = result.r_prime_w
    lines = [
        f"Rw (Dd) = {rw_dd:.1f} dB",
        f"R'w = {rpw:.1f} dB",
        f"R'w − Rw = {rpw - rw_dd:.1f} dB",
        f"Dd {direct_share:.1f} %   ΣFf,Fd,Df {flank_share:.1f} %",
    ]
    ax.text(0.985, 0.62, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "prediction_flanking_demo.png")
    plt.close()


def generate_facade_prediction(output_dir: str) -> None:
    """EN 12354-3 façade airborne insulation prediction (Annex F worked example)."""
    print("Generating facade_prediction.png...")
    from phonometry import FacadeElement, facade_sound_reduction

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Annex F elements: double wall, two windows (area, R) + a small air inlet (Dn,e).
    elements = [
        FacadeElement(name="wall", area=6.0, r=[41, 46, 52, 58, 64]),
        FacadeElement(name="window", area=4.5, r=[23, 22, 30, 36, 37]),
        FacadeElement(name="skylight", area=0.5, r=[24, 27, 30, 33, 30]),
        FacadeElement(name="air inlet", dn_e=[28, 23, 25, 38, 44]),
    ]
    result = facade_sound_reduction(
        elements, area=11.3, volume=50.0, frequencies=bands, bands="octave"
    )

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Per-element partial indices Rp: thin, faded; they set the transmission floor.
    el_colors = [COLOR_PRIMARY, COLOR_SECONDARY, "#9467bd", "#ff7f0e"]
    for (name, rp), colour in zip(result.element_r.items(), el_colors):
        ax.plot(x, rp, "--", color=colour, linewidth=1.1, alpha=0.65,
                marker=".", markersize=6, label=f"Rp — {name}")
    # Façade apparent reduction R' and standardized level difference D2m,nT.
    ax.plot(x, result.r_prime, "-", color=COLOR_FG, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="R′ (façade)")
    ax.plot(x, result.d_2m_nt, "-", color=COLOR_TERTIARY, linewidth=2.2, marker="s",
            markersize=6, zorder=5, label="D2m,nT")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Reduction index / level difference [dB]")
    ax.set_title("EN 12354-3 Façade Sound Insulation (Annex F example)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)

    info = [
        f"R′tr,s,w = {result.r_tr_s_w} dB   (Ctr = {result.c_tr})",
        f"D2m,nT,w = {result.d_2m_nt_w} dB",
        "air inlet limits the low bands",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_prediction.png")
    plt.close()


def generate_intensity_insulation(output_dir: str) -> None:
    """ISO 15186-1 intensity SRI and the Kc-modified index RI,M = RI + Kc."""
    print("Generating intensity_insulation...")
    from phonometry import adaptation_term_kc, intensity_sound_reduction

    # 16 one-third-octave bands (100-3150 Hz); reuse the ISO 717-1 Annex C
    # airborne shape as the intensity SRI target (RI,w = 30 dB, a light wall).
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    ri = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                   28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    lp1, sm, s = 85.0, 12.0, 10.0
    # Levels that make Formula (7) land on RI, then modify with Kc (Annex B).
    l_in = lp1 - 6.0 - 10.0 * np.log10(sm / s) - ri
    kc = adaptation_term_kc(freqs)
    result = intensity_sound_reduction(
        np.full(16, lp1), l_in, measurement_area=sm, area=s, kc=kc
    )
    assert result.r_i_modified is not None
    assert result.rating is not None and result.rating_modified is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the Kc adaptation lift between RI and RI,M (largest at low bands).
    ax.fill_between(x, result.r_i, result.r_i_modified, color=COLOR_TERTIARY,
                    alpha=0.18, zorder=0, label="Kc adaptation")
    ax.plot(x, result.r_i, "-", color=COLOR_PRIMARY, linewidth=2.6, marker="o",
            markersize=6, zorder=5, label="RI (intensity)")
    ax.plot(x, result.r_i_modified, "--", color=COLOR_TERTIARY, linewidth=2.2,
            marker="s", markersize=6, zorder=5, label="RI,M = RI + Kc")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(f)}" for f in freqs], rotation=45, ha="right",
                       fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound reduction index [dB]")
    ax.set_title("ISO 15186-1 Intensity Sound Reduction Index (RI and RI,M)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    # Data-only info box (language-neutral); the Kc lift is explained by the
    # shaded "Kc adaptation" legend entry, which the ES translator handles.
    info = [
        f"RI,w = {result.rating.rating} dB",
        f"RI,M,w = {result.rating_modified.rating} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_insulation.png")
    plt.close()


def generate_survey_insulation(output_dir: str) -> None:
    """ISO 10052 survey method: the reverberation-index correction D -> DnT."""
    print("Generating survey_insulation...")
    from phonometry import reverberation_index, survey_airborne_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # A masonry partition: raw level difference D and the measured receiving-
    # room reverberation time T per octave band.
    l1 = np.array([88.0, 90.0, 92.0, 92.0, 90.0])
    l2 = np.array([55.0, 51.0, 47.0, 41.0, 35.0])
    t = np.array([0.7, 0.6, 0.5, 0.45, 0.4])
    k = reverberation_index(t)
    res = survey_airborne_insulation(l1, l2, k, volume=50.0)
    assert res.rating is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Shade the reverberation-index correction k between D and DnT.
    ax.fill_between(x, res.d, res.d_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="k = 10 lg(T/T0)")
    ax.plot(x, res.d, "--", color=COLOR_PRIMARY, linewidth=1.8, marker="o",
            markersize=6, zorder=5, label="D (level difference)")
    ax.plot(x, res.d_nt, "-", color=COLOR_FG, linewidth=2.6, marker="s",
            markersize=6, zorder=5, label="DnT (standardized)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 10052 Survey Method: Reverberation-Index Correction",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w = {res.rating.rating} dB  (C = {res.rating.c})",
        "octave bands, T0 = 0.5 s",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_insulation.png")
    plt.close()


def generate_floor_covering_improvement(output_dir: str) -> None:
    """ISO 16251-1 floor-covering impact-sound improvement spectrum ΔL with ΔLw."""
    print("Generating floor_covering_improvement...")
    from phonometry import impact_improvement

    freqs = [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
             630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0]
    # A real textile carpet measured on the CSTB mock-up: the improvement
    # spectrum digitized from Figure 4 of Foret, Chene & Guigou-Carter, Forum
    # Acusticum 2011 (ISO 16251-1 series). The published weighted improvement
    # is delta-Lw = 29 dB.
    bare = np.full(16, 78.0)
    covering = bare - np.array([5, 8, 10, 14, 18, 23, 30, 31, 39, 49,
                                53, 57, 60, 67, 68, 71], dtype=float)
    res = impact_improvement(bare, covering, freqs)
    assert res.delta_lw is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(x, 0.0, res.improvement, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0)
    ax.plot(x, res.improvement, "-", color=COLOR_PRIMARY, linewidth=2.4,
            marker="o", markersize=6, zorder=5, label="delta-L (improvement)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Improvement of impact sound insulation [dB]")
    ax.set_ylim(bottom=0.0)
    ax.set_title("ISO 16251-1 Floor-Covering Impact Sound Improvement",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"delta-Lw = {res.delta_lw} dB  (ISO 717-2)",
        "one-third octave, mock-up (a0 = 1e-6 m/s^2)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "floor_covering_improvement.png")
    plt.close()



def generate_heavy_impact_sources(output_dir: str) -> None:
    """Rubber-ball and bang-machine LFE spectra with their printed tolerances."""
    print("Generating heavy_impact_sources...")
    from phonometry import (
        a_weighted_maximum_impact_level,
        heavy_impact_source_limits,
        heavy_impact_source_specification,
    )

    _fig, (ax_src, ax_rate) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    x = np.arange(5)
    for source, colour in (
        ("rubber_ball", COLOR_PRIMARY), ("bang_machine", COLOR_SECONDARY)
    ):
        spec = heavy_impact_source_specification(source)
        _f, lower, upper = heavy_impact_source_limits(source)
        label = source.replace("_", " ")
        ax_src.fill_between(x, lower, upper, color=colour, alpha=0.30, zorder=1,
                            label=f"{label} tolerance")
        ax_src.plot(x, spec.force_exposure_level, "-o", color=colour, linewidth=2.4,
                    markersize=7, zorder=4, label=f"{label} nominal")
    ax_src.set_xticks(x)
    ax_src.set_xticklabels(["31.5", "63", "125", "250", "500"])
    ax_src.set_xlabel(LABEL_FREQ_HZ)
    ax_src.set_ylabel("Impact force exposure level LFE [dB re 1 N]")
    ax_src.set_title("Standard heavy impact sources\n(ISO 16283-2 Table A.1, "
                     "JIS A 1418-2 Tables A.1/A.2)", fontweight="bold", pad=10)
    ax_src.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_src.set_axisbelow(True)
    ax_src.legend(loc="upper right", fontsize=9)

    # ISO 717-2:2020 Table D.4: a field measurement in octave bands.
    levels = [65.3, 64.5, 58.0, 55.8]
    res = a_weighted_maximum_impact_level(levels)
    xr = np.arange(4)
    ax_rate.bar(xr - 0.19, res.levels, width=0.36, color=COLOR_PRIMARY,
                label="Li,Fmax (measured)", zorder=3)
    ax_rate.bar(xr + 0.19, res.corrected, width=0.36, color=COLOR_TERTIARY,
                label="Li,Fmax + A (Table D.3)", zorder=3)
    ax_rate.axhline(res.rating, color=COLOR_SECONDARY, linewidth=2.0, zorder=4,
                    label=f"LiA,Fmax = {res.rating} dB")
    ax_rate.set_xticks(xr)
    ax_rate.set_xticklabels(["63", "125", "250", "500"])
    ax_rate.set_xlabel(LABEL_FREQ_HZ)
    ax_rate.set_ylabel("Maximum impact sound pressure level [dB]")
    ax_rate.set_title("A-weighted heavy-impact rating\n(ISO 717-2 Annex D, "
                      "Table D.4 worked example)", fontweight="bold", pad=10)
    ax_rate.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_rate.set_axisbelow(True)
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.text(
        0.02, 0.03,
        f"unrounded sum = {res.unrounded:.6f} dB",
        transform=ax_rate.transAxes, va="bottom", ha="left", fontsize=10,
        color=COLOR_FG,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
              "edgecolor": COLOR_GRID},
    )

    plt.tight_layout()
    save_figure(output_dir, "heavy_impact_sources.png")
    plt.close()


def generate_ceiling_plenum_flanking(output_dir: str) -> None:
    """Ceiling/plenum flanking path Rcl and the CAC of an accredited report."""
    print("Generating ceiling_plenum_flanking...")
    from phonometry import (
        ceiling_attenuation_class,
        plenum_flanking_reduction_index,
    )

    _fig, (ax_path, ax_cac) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Vigran Figs. 9.11-9.13 geometry: LS = LR = 4,75 m, plenum h = 0,43 m,
    # 9,5 mm plasterboard ceiling, reflecting plenum sidewalls.
    freqs = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    ceiling = np.array([17.0, 21.0, 25.0, 29.0, 32.0, 30.0, 38.0])
    x = np.arange(freqs.size)
    for depth, colour, style in (
        (0.43, COLOR_PRIMARY, "-"), (0.86, COLOR_TERTIARY, "--")
    ):
        res = plenum_flanking_reduction_index(
            ceiling, ceiling, ceiling_length=4.75, plenum_height=depth,
            frequency=freqs,
        )
        ax_path.plot(x, res.reduction_index, style, color=colour, linewidth=2.4,
                     marker="o", markersize=6, zorder=4,
                     label=f"Rcl, plenum h = {depth:g} m")
    ax_path.plot(x, 2.0 * ceiling, ":", color=COLOR_SECONDARY, linewidth=2.0,
                 marker="s", markersize=5, zorder=3, label="RS + RR (two ceilings)")
    ax_path.set_xticks(x)
    ax_path.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax_path.set_xlabel(LABEL_FREQ_HZ)
    ax_path.set_ylabel("Sound reduction index [dB]")
    ax_path.set_title("Suspended-ceiling plenum path\n(one-dimensional model, "
                      "LR = 4.75 m, reflecting sidewalls)",
                      fontweight="bold", pad=10)
    ax_path.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_path.set_axisbelow(True)
    ax_path.legend(loc="upper left", fontsize=9)

    # ALA 16-091-4 (2016), tested to ASTM E1414/E1414M-11a: CAC 34.
    dnc = np.array([14.4, 18.6, 21.7, 24.1, 23.4, 30.3, 33.7, 35.2,
                    41.6, 44.2, 42.1, 36.8, 35.7, 36.0, 36.9, 37.9])
    cac = ceiling_attenuation_class(dnc)
    xc = np.arange(dnc.size)
    ax_cac.fill_between(xc, cac.measured, cac.shifted_reference,
                        where=(cac.measured < cac.shifted_reference).tolist(),
                        color=COLOR_SECONDARY, alpha=0.25, interpolate=True,
                        zorder=1, label="deficiencies")
    ax_cac.plot(xc, cac.measured, "-o", color=COLOR_PRIMARY, linewidth=2.4,
                markersize=6, zorder=4, label="Dn,c (measured)")
    ax_cac.plot(xc, cac.shifted_reference, "--s", color=COLOR_TERTIARY,
                linewidth=2.0, markersize=5, zorder=3,
                label="ASTM E413 contour, fitted")
    ax_cac.set_xticks(xc[::2])
    ax_cac.set_xticklabels(["125", "200", "315", "500", "800", "1.25k",
                            "2k", "3.15k"], rotation=45, fontsize=8)
    ax_cac.set_xlabel(LABEL_FREQ_HZ)
    ax_cac.set_ylabel("Normalized ceiling attenuation Dn,c [dB]")
    ax_cac.set_title(f"Ceiling attenuation class\n(ASTM E1414/E413, "
                     f"CAC = {cac.rating} dB)", fontweight="bold", pad=10)
    ax_cac.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_cac.set_axisbelow(True)
    ax_cac.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "ceiling_plenum_flanking.png")
    plt.close()


def generate_masonry_wall_ties(output_dir: str) -> None:
    """Wall-tie coupling loss factor and the resonance shift it causes."""
    print("Generating masonry_wall_ties...")
    from phonometry import (
        double_wall_transmission_loss,
        wall_tie_coupling_loss_factor,
        wall_tie_stiffness,
        wall_tie_stiffness_per_area,
    )

    _fig, (ax_clf, ax_tl) = plt.subplots(1, 2, figsize=(13.0, 5.6))

    # Two 100 mm masonry leaves, 150 and 170 kg/m2 (Hopkins Fig. 5.30
    # flanking-laboratory cavity wall), 2,5 ties per m2.
    freq = np.logspace(np.log10(50.0), np.log10(5000.0), 200)
    stiffness1 = 150.0 * 2000.0**2 * 0.1**2 / 12.0
    stiffness2 = 170.0 * 2000.0**2 * 0.1**2 / 12.0
    ties = ("butterfly", "double_triangle", "vertical_twist")
    colours = (COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY)
    for tie, colour in zip(ties, colours, strict=True):
        clf = wall_tie_coupling_loss_factor(
            freq, 150.0, 170.0, stiffness1, stiffness2,
            ties_per_area=2.5, tie=tie,
        )
        label = f"{tie.replace('_', ' ')} ({wall_tie_stiffness(tie)[1] / 1e6:g} MN/m)"
        ax_clf.loglog(freq, clf.coupling_loss_factor, color=colour, linewidth=2.4,
                      zorder=4, label=label)
    rigid = wall_tie_coupling_loss_factor(
        freq, 150.0, 170.0, stiffness1, stiffness2, ties_per_area=2.5
    )
    ax_clf.loglog(freq, rigid.rigid_coupling_loss_factor, "--", color=COLOR_MUTED,
                  linewidth=2.0, zorder=3, label="rigid connection (Yc = 0)")
    ax_clf.set_xticks([50, 125, 250, 500, 1000, 2000, 4000])
    ax_clf.set_xticklabels(["50", "125", "250", "500", "1k", "2k", "4k"])
    ax_clf.set_xlim(50.0, 5000.0)
    # Plain ASCII decade labels: the mathtext 10^-n exponent would put a
    # U+2212 minus in the SVG, which not every reader's sans-serif font has.
    ax_clf.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    ax_clf.set_yticklabels(["1e-8", "1e-6", "1e-4", "1e-2", "1"])
    ax_clf.set_ylim(1e-8, 2.0)
    ax_clf.set_xlabel(LABEL_FREQ_HZ)
    ax_clf.set_ylabel("Coupling loss factor eta_ij")
    ax_clf.set_title("Wall-tie structure-borne coupling\n(point-connection model, "
                     "2.5 ties/m2)", fontweight="bold", pad=10)
    ax_clf.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_clf.set_axisbelow(True)
    ax_clf.legend(loc="lower left", fontsize=9)

    # Hopkins Fig. 4.35: two 140 kg/m2 leaves, empty 75 mm cavity, 2,5 ties/m2
    # of s_75mm = 2e6 N/m, which lifts fmsm from 26 Hz to 50 Hz.
    bands = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0,
                      125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0])
    per_area = wall_tie_stiffness_per_area(2.5, 2.0e6)
    plain = double_wall_transmission_loss(bands, 140.0, 140.0, 0.075)
    tied = double_wall_transmission_loss(
        bands, 140.0, 140.0, 0.075, tie_stiffness_per_area=per_area
    )
    xb = np.arange(bands.size)
    ax_tl.plot(xb, plain.transmission_loss, "-o", color=COLOR_PRIMARY,
               linewidth=2.4, markersize=6, zorder=4, label="cavity wall, no ties")
    ax_tl.plot(xb, tied.transmission_loss, "--s", color=COLOR_SECONDARY,
               linewidth=2.4, markersize=6, zorder=4, label="2.5 ties/m2, k = 2 MN/m")
    positions = []
    for curve, colour in ((plain, COLOR_PRIMARY), (tied, COLOR_SECONDARY)):
        f0 = curve.resonance_frequency
        assert f0 is not None
        pos = float(np.interp(np.log10(f0), np.log10(bands), xb))
        positions.append(pos)
        ax_tl.axvline(pos, color=colour, linestyle=":", linewidth=1.8, zorder=2)
        ax_tl.annotate(
            f"fmsm = {f0:.0f} Hz", xy=(pos, 0.46),
            xycoords=("data", "axes fraction"), ha="right" if positions[:1] else "left",
            va="bottom", fontsize=9, color=colour, rotation=90,
        )
    ax_tl.axvspan(positions[0], positions[1], color=COLOR_SECONDARY, alpha=0.30,
                  zorder=1, label="combined-mass range added by the ties")
    ax_tl.set_xticks(xb[::2])
    ax_tl.set_xticklabels(["20", "31.5", "50", "80", "125", "200", "315", "500"])
    ax_tl.set_xlabel(LABEL_FREQ_HZ)
    ax_tl.set_ylabel("Sound reduction index R [dB]")
    ax_tl.set_title("Ties stiffen the cavity\n(140 kg/m2 leaves, 75 mm cavity, "
                    "2.5 ties/m2)", fontweight="bold", pad=10)
    ax_tl.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax_tl.set_axisbelow(True)
    ax_tl.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    save_figure(output_dir, "masonry_wall_ties.png")
    plt.close()


def generate_flanking_transmission(output_dir: str) -> None:
    """ISO 10848 vibration reduction index Kij per band with the mean K̄ij."""
    print("Generating flanking_transmission...")
    from phonometry import vibration_reduction_index

    freqs = [100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0,
             800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
    # A rigid T-junction of two heavy walls: measured direction-averaged velocity
    # level difference rising gently with frequency (typical laboratory data).
    dv = np.array([4.5, 4.8, 5.2, 5.6, 6.0, 6.5, 7.0, 7.6, 8.1, 8.7, 9.2, 9.8,
                   10.3, 10.9, 11.4, 11.9, 12.3, 12.7])
    res = vibration_reduction_index(
        dv, junction_length=4.0, area_i=12.0, area_j=10.0,
        frequency=freqs,
        structural_reverberation_time_i=0.35,
        structural_reverberation_time_j=0.40,
    )
    assert res.single_number is not None

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(x, res.k_ij, "-", color=COLOR_PRIMARY, linewidth=2.4, marker="o",
            markersize=6, zorder=5, label="Kij (ISO 10848)")
    ax.axhline(res.single_number, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=4, label="mean Kij (200-1250 Hz)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Vibration reduction index Kij [dB]")
    ax.set_title("ISO 10848 Junction Vibration Reduction Index",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "rigid T-junction, two heavy walls",
        "lij = 4 m, Si = 12 m^2, Sj = 10 m^2",
        "Formula (13), one-third octave",
        f"mean Kij = {res.single_number:.1f} dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_transmission.png")
    plt.close()


def generate_reverberation_models(output_dir: str) -> None:
    """Sabine/Eyring/Millington/Fitzroy/Arau reverberation time over octaves."""
    print("Generating reverberation_models...")
    from phonometry import air_attenuation_m, reverberation_time_models

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    # A 10 x 7 x 3.5 m room (V = 245 m3, S = 259 m2) with a strongly anisotropic
    # absorption distribution: a very absorptive floor/ceiling pair (carpet plus
    # an acoustic ceiling) against hard end walls and lightly treated side walls.
    # This is where the axial models (Fitzroy, Arau-Puchades) part company with
    # the isotropic Sabine and Eyring estimates.
    alpha_x = [0.06, 0.07, 0.08, 0.09, 0.10, 0.10]   # hard end walls
    alpha_y = [0.12, 0.14, 0.16, 0.18, 0.20, 0.20]   # lightly treated side walls
    alpha_z = [0.30, 0.50, 0.65, 0.78, 0.82, 0.80]   # carpet + acoustic ceiling
    m = air_attenuation_m(bands, 20.0, 50.0)
    res = reverberation_time_models(
        (10.0, 7.0, 3.5), (alpha_x, alpha_y, alpha_z),
        air_attenuation=m, frequencies=bands,
    )

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    styles = [
        ("Sabine", res.sabine, COLOR_SECONDARY, "s", 1.8),
        ("Eyring", res.eyring, COLOR_TERTIARY, "^", 1.8),
        ("Millington-Sette", res.millington_sette, "#9467bd", "v", 1.8),
        ("Fitzroy", res.fitzroy, "#ff7f0e", "D", 1.8),
        ("Arau-Puchades", res.arau_puchades, COLOR_PRIMARY, "o", 2.6),
    ]
    for label, curve, color, marker, lw in styles:
        ax.plot(x, curve, color=color, linewidth=lw, marker=marker,
                markersize=6, label=label, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Reverberation time $T$ [s]")
    ax.set_title("Reverberation-time prediction models", fontweight="bold", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "room 10 x 7 x 3.5 m",
        "V = 245 m^3, S = 259 m^2",
        "anisotropic: absorptive floor/ceiling",
        "c0 = 343 m/s, air at 20 C / 50 % RH",
    ]
    ax.text(0.015, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "reverberation_models.svg")
    plt.close()


def generate_dynamic_stiffness(output_dir: str) -> None:
    """EN 29052-1 floating-floor natural frequency f0(s') for typical floors."""
    print("Generating dynamic_stiffness...")
    from phonometry import natural_frequency

    s_mn = np.logspace(np.log10(2.0), np.log10(100.0), 300)   # MN/m3
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    # Two typical floating-floor masses per unit area (light vs heavy screed).
    for m, color, label in ((40.0, COLOR_SECONDARY, "m' = 40 kg/m^2"),
                             (120.0, COLOR_PRIMARY, "m' = 120 kg/m^2")):
        f0 = np.asarray(natural_frequency(s_mn * 1e6, m), dtype=float)
        ax.plot(s_mn, f0, color=color, linewidth=2.2, label=label)

    # A worked design point: s' = 10 MN/m3 on the 120 kg/m2 floor.
    s0, m0 = 10.0, 120.0
    f00 = float(natural_frequency(s0 * 1e6, m0))
    ax.scatter([s0], [f00], color=COLOR_TERTIARY, s=90, zorder=6,
               label=f"design point ({s0:g} MN/m^3, {f00:.0f} Hz)")
    ax.plot([s0, s0], [0, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)
    ax.plot([s_mn[0], s0], [f00, f00], color=COLOR_GRID, ls=":", lw=1.0, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dynamic stiffness per unit area $s'$ [MN/m³]")
    ax.set_ylabel(r"Natural frequency $f_0$ [Hz]")
    ax.set_title("EN 29052-1 Floating-Floor Resonance", fontweight="bold", pad=12)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=10)

    info = [
        "f0 = (1/2pi) sqrt(s'/m')  (Formula 2)",
        "s'  = s't + s'a  (clause 8.2)",
        "s't = 4 pi^2 m't fr^2  (Formula 4)",
        "s'a = p0/(d eps) ~ 111/d MN/m^3  (NOTE)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "dynamic_stiffness.svg")
    plt.close()


def generate_junction_transmission(output_dir: str) -> None:
    """Hopkins 5.2.1.3 bending-wave transmission at a rigid X-junction."""
    print("Generating junction_transmission...")
    from phonometry import junction_transmission

    # X-junction between a 100 mm and a 200 mm concrete plate (cL = 3200 m/s,
    # rho = 2400 kg/m^3 -> rho_s = 240 and 480 kg/m^2).
    res = junction_transmission("X", 0.1, 3200.0, 240.0, 0.2, 3200.0, 480.0)
    assert res.straight is not None and res.straight_average is not None
    angles = res.angles_deg

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(angles, res.corner, color=COLOR_PRIMARY, linewidth=2.0,
            label=r"corner $\tau_{12}(\theta)$")
    ax.plot(angles, res.straight, color=COLOR_SECONDARY, linewidth=2.0,
            label=r"straight $\tau_{13}(\theta)$")
    ax.axhline(res.corner_average, color=COLOR_PRIMARY, linestyle="--",
               linewidth=1.3, label="corner average")
    ax.axhline(res.straight_average, color=COLOR_SECONDARY, linestyle=":",
               linewidth=1.3, label="straight average")

    ax.set_xlabel("Incidence angle [degrees]")
    ax.set_ylabel(r"Transmission coefficient $\tau$")
    ax.set_title(
        "Bending-wave transmission at a rigid X-junction (Hopkins 5.2.1.3)",
        fontweight="bold", pad=12,
    )
    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "X-junction: 100 mm / 200 mm concrete",
        f"chi = {res.chi:.3f},  psi = {res.psi:.3f}",
        f"corner avg = {res.corner_average:.4f}",
        f"straight avg = {res.straight_average:.4f}",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "junction_transmission.svg")
    plt.close()


def generate_mechanical_mobility(output_dir: str) -> None:
    """ISO 7626-1 receptance/mobility/accelerance of a SDOF resonator."""
    print("Generating mechanical_mobility...")
    from phonometry import (
        convert_frf,
        resonance_frequency,
        sdof_receptance,
    )

    m, k, c = 2.0, 8000.0, 5.0
    f0 = resonance_frequency(m, k)
    freq = np.logspace(np.log10(f0 / 20.0), np.log10(f0 * 20.0), 600)
    w0 = 2.0 * np.pi * f0
    h = sdof_receptance(freq, m, k, c)
    y = convert_frf(h, freq, "receptance", "mobility")
    a = convert_frf(h, freq, "receptance", "accelerance")
    # Normalise each FRF to O(1) near resonance so all three share one axis.
    curves = [
        (np.abs(h) * k, COLOR_PRIMARY, "Receptance $|H|$ (× k)"),
        (np.abs(y) * k / w0, COLOR_SECONDARY, r"Mobility $|Y|$ (× k/$\omega_0$)"),
        (np.abs(a) * k / w0**2, COLOR_TERTIARY, r"Accelerance $|A|$ (× k/$\omega_0^2$)"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for mag, color, label in curves:
        ax.loglog(freq, mag, color=color, linewidth=2.0, label=label)
    ax.axvline(f0, color=COLOR_GRID, linestyle="--", linewidth=1.2,
               label="resonance $f_0$")

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Normalized FRF magnitude")
    ax.set_title("ISO 7626-1 Mechanical Mobility FRFs", fontweight="bold", pad=12)
    ax.set_xlim(freq[0], freq[-1])
    format_frequency_axis(ax, float(freq[0]), float(freq[-1]))
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        "SDOF: m = 2 kg, k = 8000 N/m, c = 5 N.s/m",
        "H = 1/(k - w^2 m + j w c)",
        "Y = j w H,   A = -w^2 H  (Table 1)",
        f"f0 = {f0:.1f} Hz,  |Y(f0)| = 1/c",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "mechanical_mobility.svg")
    plt.close()


def generate_transfer_stiffness(output_dir: str) -> None:
    """ISO 10846 dynamic transfer stiffness: true vs indirect-method recovery."""
    print("Generating transfer_stiffness...")
    from phonometry import (
        base_transmissibility,
        transfer_stiffness_indirect,
        transfer_stiffness_level,
    )

    # Kelvin-Voigt isolator k + jwc, loaded by a blocking mass m2.
    k, c, m2 = 1.0e6, 120.0, 8.0
    f0 = np.sqrt(k / m2) / (2.0 * np.pi)
    freq = np.logspace(np.log10(f0 / 5.0), np.log10(f0 * 40.0), 600)
    w = 2.0 * np.pi * freq

    k_true = k + 1j * w * c                                # exact transfer stiffness
    t = base_transmissibility(freq, m2, k, c)              # mass-loaded transmissibility
    k_indirect = transfer_stiffness_indirect(freq, t, m2)  # ISO 10846-3 Eq. (1)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.semilogx(freq, transfer_stiffness_level(k_true), color=COLOR_PRIMARY,
                linewidth=2.2, label=r"true $L_k$ of $k_{2,1}=k+j\omega c$")
    ax.semilogx(freq, transfer_stiffness_level(k_indirect), color=COLOR_SECONDARY,
                linewidth=2.0, linestyle="--",
                label=r"indirect method $-(2\pi f)^2 m_2 T$")
    ax.axvline(f0, color=COLOR_GRID, linestyle=":", linewidth=1.2,
               label="resonance $f_0$")
    ax.axvspan(freq[0], 3.0 * f0, color=theme_fill(COLOR_FG, ax), zorder=0)

    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Transfer stiffness level $L_k$ [dB re 1 N/m]")
    ax.set_title("ISO 10846 Dynamic Transfer Stiffness", fontweight="bold", pad=12)
    ax.set_xlim(freq[0], freq[-1])
    format_frequency_axis(ax, float(freq[0]), float(freq[-1]))
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "Kelvin-Voigt: k = 1 MN/m, c = 120 N.s/m",
        f"blocking mass m2 = 8 kg,  f0 = {f0:.1f} Hz",
        "indirect valid for T << 1  (f >> f0)",
        "shaded: T not small -> method invalid",
    ]
    ax.text(0.985, 0.05, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "transfer_stiffness.svg")
    plt.close()


def generate_rigid_mass_calibration(output_dir: str) -> None:
    """ISO 7626-2 (7.5.2) operational rigid-mass calibration check."""
    print("Generating rigid_mass_calibration...")
    from phonometry import rigid_mass_calibration_check

    # A 10 kg calibration block: the accelerance must be a flat |A| = 1/m over
    # frequency. The measured chain has a mild ripple and drifts above the
    # +/-5 % band towards a few kHz (a transducer/attachment-compliance error,
    # exactly what the check is meant to catch).
    m = 10.0
    freq = np.logspace(np.log10(20.0), np.log10(5000.0), 400)
    expected = 1.0 / m
    ripple = 0.015 * np.sin(2.0 * np.pi * np.log10(freq))
    drift = 0.05 * (freq / 2500.0) ** 2
    measured = expected * (1.0 + ripple + drift)
    res = rigid_mass_calibration_check(measured, freq, mass=m)
    within = res.within_tolerance
    tol = res.tolerance

    _fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 7.0),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
    )
    # Upper panel: measured accelerance against the rigid-mass line + band.
    ax_top.fill_between(freq, res.expected * (1.0 - tol), res.expected * (1.0 + tol),
                        color=COLOR_SECONDARY, alpha=0.15,
                        label=r"$\pm$5 % tolerance band")
    ax_top.semilogx(freq, res.expected, color=COLOR_SECONDARY, linestyle="--",
                    linewidth=1.6, label=r"expected $|A| = 1/m$")
    ax_top.semilogx(freq, res.measured, color=COLOR_PRIMARY, linewidth=2.0,
                    label="within tolerance")
    ax_top.semilogx(freq[~within], res.measured[~within], linestyle="none",
                    marker="o", markersize=4, color=COLOR_SECONDARY,
                    label="out of tolerance")
    ax_top.set_ylabel("Accelerance $|A|$ [1/kg]")
    ax_top.set_title("ISO 7626-2 Rigid-Mass Calibration Check",
                     fontweight="bold", pad=12)
    ax_top.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax_top.legend(loc="upper left", fontsize=9)

    # Lower panel: the relative deviation against the same +/-5 % band, where
    # the few-percent tolerance is actually readable.
    ax_bot.axhspan(-100.0 * tol, 100.0 * tol, color=COLOR_SECONDARY, alpha=0.15)
    ax_bot.axhline(0.0, color=COLOR_GRID, linestyle=":", linewidth=1.0)
    ax_bot.semilogx(freq, 100.0 * res.deviation, color=COLOR_PRIMARY,
                    linewidth=2.0)
    ax_bot.semilogx(freq[~within], 100.0 * res.deviation[~within],
                    linestyle="none", marker="o", markersize=4,
                    color=COLOR_SECONDARY)
    ax_bot.set_xlabel(LABEL_FREQ_HZ)
    ax_bot.set_ylabel("Deviation [%]")
    ax_bot.set_xlim(freq[0], freq[-1])
    ax_bot.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)

    format_frequency_axis(ax_top, float(freq[0]), float(freq[-1]))
    format_frequency_axis(ax_bot, float(freq[0]), float(freq[-1]))

    info = [
        "calibration block m = 10 kg",
        "|A| = 1/m = 0.100 1/kg  (7.5.2)",
        "criterion: agree within +/- 5 %",
        "high-f drift -> attachment error",
    ]
    ax_top.text(0.985, 0.05, "\n".join(info), transform=ax_top.transAxes,
                va="bottom", ha="right", fontsize=10, color=COLOR_FG,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                      "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "rigid_mass_calibration.svg")
    plt.close()


def generate_vibration_sound_power(output_dir: str) -> None:
    """ISO/TS 7849 sound power from surface vibration: upper limit vs engineering."""
    print("Generating vibration_sound_power...")
    from phonometry import radiated_sound_power_level

    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    # A plausible surface velocity level spectrum and a measured radiation factor.
    lv = np.array([78.0, 82.0, 85.0, 83.0, 79.0, 74.0])
    eps = np.array([0.20, 0.45, 0.75, 0.95, 1.00, 1.00])
    area = 1.6

    lw_max = radiated_sound_power_level(lv, area)                    # Part 1, eps=1
    lw_eng = radiated_sound_power_level(lv, area, radiation_factor=eps)  # Part 2

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x - 0.2, lw_max, width=0.4, color=COLOR_SECONDARY, edgecolor=COLOR_FG,
           linewidth=0.6, label="Part 1 upper limit ($\\varepsilon$ = 1)")
    ax.bar(x + 0.2, lw_eng, width=0.4, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.6, label="Part 2 engineering ($\\varepsilon$ measured)")

    total_max = 10.0 * np.log10(np.sum(10.0 ** (0.1 * lw_max)))
    total_eng = 10.0 * np.log10(np.sum(10.0 ** (0.1 * lw_eng)))
    ax.axhline(total_max, color=COLOR_SECONDARY, ls="--", lw=1.2,
               label=f"total (limit) {total_max:.1f} dB")
    ax.axhline(total_eng, color=COLOR_PRIMARY, ls="--", lw=1.2,
               label=f"total (eng.) {total_eng:.1f} dB")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Sound power level $L_W$ [dB re 1 pW]")
    ax.set_title("ISO/TS 7849 Sound Power from Surface Vibration",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LW = Lv + 10 lg(S/S0) + 10 lg(e) + 10 lg(411/400)",
        f"S = {area:g} m2,  S0 = 1 m2",
        "Part 1: e = 1 -> upper limit LW,max",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "vibration_sound_power.svg")
    plt.close()


def generate_structure_borne_power(output_dir: str) -> None:
    """EN 15657 structure-borne sound power injected into the reception plate."""
    print("Generating structure_borne_power...")
    from phonometry import reception_plate_power

    bands = np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3150.0])
    # A pump-like source on a low-mobility (heavy) and a high-mobility (light)
    # reception plate; the two determinations should agree within the method.
    lv_low = np.array([88.0, 90.0, 87.0, 84.0, 80.0, 76.0, 71.0])
    lv_high = lv_low + 6.0                      # lighter plate vibrates more
    res_low = reception_plate_power(lv_low, bands, mass_per_area=600.0, area=2.0,
                                    reverberation_time=0.8)
    res_high = reception_plate_power(lv_high, bands, mass_per_area=150.0, area=2.0,
                                     reverberation_time=0.5)

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x - 0.2, res_low.power_level, width=0.4, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.6, label="low-mobility plate")
    ax.bar(x + 0.2, res_high.power_level, width=0.4, color=COLOR_SECONDARY,
           edgecolor=COLOR_FG, linewidth=0.6, label="high-mobility plate")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Structure-borne power level $L_{Ws}$ [dB re 1 pW]")
    ax.set_title("EN 15657 Reception-Plate Structure-Borne Sound Power",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LWs = 10 lg(2 pi f eta m S) + Lv - 60 dB",
        "eta = 2.2/(f Ts),  v0 = 1 nm/s",
        "reception-plate method (clause 7)",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "structure_borne_power.svg")
    plt.close()


def generate_installed_structure_borne(output_dir: str) -> None:
    """EN 12354-5 installed structure-borne sound: characteristic power to SPL."""
    print("Generating installed_structure_borne...")
    from phonometry import (
        coupling_term,
        installed_source_prediction,
        installed_structure_borne_power_level,
    )

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    lws_c = np.array([78.0, 82.0, 84.0, 81.0, 77.0, 72.0, 66.0])   # EN 15657 source
    # Frequency-dependent source / receiver point mobilities (illustrative).
    ys = (2.0e-4 + 1.0e-4j) * (bands / 250.0)
    yi = (3.0e-5 + 1.0e-5j) * np.ones_like(bands)
    dc = np.array([float(coupling_term(a, b)) for a, b in zip(ys, yi)])
    lws_inst = installed_structure_borne_power_level(lws_c, dc)
    paths = [
        {"adjustment_term": 6.0,
         "flanking_reduction_index": np.array([44., 47., 50., 53., 56., 59., 62.]),
         "element_area": 12.0},
        {"adjustment_term": 7.0,
         "flanking_reduction_index": np.array([46., 49., 52., 55., 58., 61., 64.]),
         "element_area": 9.0},
    ]
    res = installed_source_prediction(lws_c, dc, paths, frequencies=bands)

    x = np.arange(bands.size)
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(x, lws_c, color=COLOR_SECONDARY, marker="o", lw=2.0,
            label=r"characteristic $L_{Ws,c}$ (EN 15657)")
    ax.plot(x, lws_inst, color=COLOR_TERTIARY, marker="s", lw=2.0,
            label=r"installed $L_{Ws,inst}$ = $L_{Ws,c}-D_C$")
    for k, p in enumerate(res.path_levels):
        ax.plot(x, p, color=COLOR_GRID, lw=1.0, ls=":", marker=".",
                label="paths $L_{n,s,ij}$" if k == 0 else None)
    ax.plot(x, res.total_level, color=COLOR_PRIMARY, marker="D", lw=2.4,
            label=r"total $L_{n,s}$")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Level [dB]")
    ax.set_title("EN 12354-5 Installed Structure-Borne Sound",
                 fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "DC = 10 lg(|Ys+Yi|^2 / (|Ys| Re Yi))",
        "Ln,s,ij = LWs,inst - Dsa - Rij - 10 lg(Si/S0) - 10 lg(A0/4)",
        "Ln,s = 10 lg(sum 10^(Ln,s,ij/10)),  S0 = A0 = 10 m2",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "installed_structure_borne.svg")
    plt.close()


def generate_tone_audibility(output_dir: str) -> None:
    """ISO/PAS 20065 tonal audibility: per-tone ΔL of the Annex E example."""
    print("Generating tone_audibility...")
    from phonometry import assess_tones

    # Annex E combustion-engine example, spectrum 1 (Tables E.2/E.3),
    # line spacing Δf = 2.7 Hz. Each tuple is (fT, LS, LT).
    tones = [
        (118.4, 48.91, 64.56), (137.3, 49.22, 67.96), (158.8, 50.50, 68.63),
        (314.9, 52.85, 68.50), (433.4, 58.29, 73.17), (592.2, 59.53, 78.31),
        (629.8, 59.71, 75.00), (643.3, 61.98, 79.75), (1582.7, 54.16, 71.07),
    ]
    freqs = [t[0] for t in tones]
    res = assess_tones(freqs, [t[2] for t in tones], [t[1] for t in tones], 2.7)

    x = np.arange(len(freqs))
    decisive = int(np.argmax(res.audibilities))
    colors = [COLOR_PRIMARY] * len(freqs)
    colors[decisive] = COLOR_SECONDARY

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x, res.audibilities, width=0.7, color=colors, edgecolor=COLOR_FG,
           linewidth=0.6)
    ax.axhline(0.0, color=COLOR_FG, ls="--", lw=1.0,
               label=r"threshold $\Delta L = 0$ dB")
    ax.bar([decisive], [res.audibilities[decisive]], width=0.7,
           color=COLOR_SECONDARY, edgecolor=COLOR_FG, linewidth=0.6,
           label=(rf"decisive $\Delta L$ = {res.decisive_audibility:.1f} dB "
                  rf"@ {res.decisive_frequency:g} Hz"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Audibility $\Delta L$ [dB]")
    ax.set_title("ISO/PAS 20065 Tonal Audibility", fontweight="bold", pad=12)
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "dfc = 25 + 75 (1 + 1.4 (fT/1000)^2)^0.69",
        "LG = LS + 10 lg(dfc/df),  av = -2 - lg(1 + (f/502)^2.5)",
        "dL = LT - LG - av  (combustion engine, Annex E)",
    ]
    ax.text(0.015, 0.02, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="left", fontsize=8.5, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "tone_audibility.svg")
    plt.close()


def generate_absorption_uncertainty(output_dir: str) -> None:
    """ISO 12999-2 absorption-coefficient uncertainty: alpha_s with a +/-U ribbon."""
    print("Generating absorption_uncertainty...")
    from phonometry import sound_absorption_coefficient_uncertainty

    # The standard's worked Example (Table 4): a measured sound absorption
    # coefficient alpha_s per one-third-octave band and its reproducibility
    # expanded uncertainty at k = 2.
    freqs = np.array([63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                      630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000],
                     dtype=float)
    alpha_s = np.array([0.33, 0.35, 0.39, 0.38, 0.37, 0.36, 0.36, 0.36, 0.43,
                        0.49, 0.58, 0.63, 0.68, 0.71, 0.73, 0.75, 0.77, 0.79,
                        0.81, 0.81])
    res = sound_absorption_coefficient_uncertainty(alpha_s, freqs, confidence=0.95)
    u = res.expanded_uncertainty

    x = np.arange(len(freqs))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(x, alpha_s - u, alpha_s + u, color=COLOR_TERTIARY, alpha=0.22,
                    zorder=0, label="+/-U (k = 2), reproducibility")
    ax.plot(x, alpha_s, "-", color=COLOR_PRIMARY, linewidth=2.4, marker="o",
            markersize=6, zorder=5, label="alpha_s (ISO 354)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(b)}" for b in freqs], rotation=45, fontsize=8)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound absorption coefficient")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("ISO 12999-2 Sound Absorption Coefficient Uncertainty",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "sigma_R = m alpha_s + n  (Table 1)",
        "U = k u,  k = 2  (95 %)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=11, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "absorption_uncertainty.png")
    plt.close()


def generate_insulation_uncertainty_demo(output_dir: str) -> None:
    """ISO 12999-1 per-band + single-number measurement uncertainty (situation B)."""
    print("Generating insulation_uncertainty_demo.png...")
    from phonometry.building.building_uncertainty import (
        band_uncertainty,
        insulation_coverage_factor,
        insulation_expanded_uncertainty,
        single_number_uncertainty,
    )
    from phonometry.building.insulation import weighted_rating

    # Reuse the ISO 717-1 Annex C measured R' curve (100 Hz .. 3150 Hz); its
    # weighted rating is R'w = 30 dB.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    measured = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                         28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    rating = weighted_rating(measured).rating

    # Per-band standard uncertainty u (ISO 12999-1 Table 2, situation B); match
    # each measured band to its tabulated value, then expand at k = 1.96 (95 %).
    band = band_uncertainty("airborne", "B")
    band_f, band_u = band.to_arrays()
    idx = [int(np.argmin(np.abs(band_f - f))) for f in freqs]
    u_band = band_u[idx]
    k = insulation_coverage_factor(0.95)
    exp_band = np.array(
        [insulation_expanded_uncertainty(float(v), 0.95) for v in u_band]
    )

    # Single-number expanded uncertainty for the rating.
    u_single = single_number_uncertainty("r_w", "B")
    exp_single = insulation_expanded_uncertainty(u_single, 0.95)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Nested bands: the inner one is placed twice as far from the page as the
    # outer, so the two read as distinct steps of the same hue.
    ax.fill_between(freqs, measured - exp_band, measured + exp_band,
                    color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
                    label="Expanded uncertainty ±U (95 %)")
    ax.fill_between(freqs, measured - u_band, measured + u_band,
                    color=theme_fill(COLOR_PRIMARY, ax, delta_e=26.0), zorder=0.1,
                    label="Standard uncertainty ±u")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.9,
                markersize=5, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Measured R'")

    # Single-number R'w with its expanded uncertainty, read at 500 Hz.
    ax.errorbar(500, rating, yerr=exp_single, fmt="D", color=COLOR_SECONDARY,
                markersize=9, capsize=6, elinewidth=1.8, zorder=6,
                label="R'w ± U (single number)")
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.35, zorder=0)

    # Word-free box (the situation and band meanings are in the title/legend, so
    # translation reduces to the automatic decimal-comma substitution).
    box = [
        f"R'w = {rating} ± {exp_single:.1f} dB",
        f"U = k·u ,  k = {k:g} (95 %)",
    ]
    ax.text(0.03, 0.97, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_title("ISO 12999-1 Measurement Uncertainty (situation B, airborne)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Apparent sound reduction index R' [dB]")
    ax.set_xscale("log")
    ax.set_xlim(90, 3600)
    ax.set_ylim(8, 42)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(
        ["100", "125", "160", "200", "250", "315", "400", "500", "630", "800",
         "1k", "1.25k", "1.6k", "2k", "2.5k", "3.15k"], fontsize=8)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "insulation_uncertainty_demo.png")
    plt.close()


def generate_air_absorption_alpha(output_dir: str) -> None:
    """ISO 9613-1 pure-tone atmospheric attenuation coefficient alpha(f)."""
    print("Generating air_absorption_alpha.png...")
    from phonometry import air_attenuation

    freqs = np.logspace(np.log10(50.0), np.log10(10000.0), 400)
    # Four representative (temperature, relative humidity) conditions spanning
    # the relaxation behaviour: the reference, a dry warm day, a cold humid day
    # and a hot humid day. alpha is returned in dB/m; plot in dB/km (Table 1).
    conditions = [
        (20.0, 50.0, COLOR_PRIMARY),
        (20.0, 10.0, COLOR_SECONDARY),
        (0.0, 70.0, COLOR_TERTIARY),
        (30.0, 80.0, "#ff7f0e"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for temp, rh, color in conditions:
        alpha_km = air_attenuation(freqs, temp, rh) * 1000.0
        ax.loglog(freqs, alpha_km, color=color, linewidth=2.0,
                  label=f"{temp:g} °C, {rh:g} % RH")
    ax.set_title("ISO 9613-1 Atmospheric Absorption α(f)", fontweight="bold",
                 pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Attenuation coefficient α [dB/km]")
    ax.set_xlim(50.0, 10000.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    save_figure(output_dir, "air_absorption_alpha.png")
    plt.close()


def generate_atmospheric_attenuation(output_dir: str) -> None:
    """ISO 9613-1 atmospheric attenuation via the AtmosphericAttenuation.plot()."""
    print("Generating atmospheric_attenuation.png...")
    from phonometry import atmospheric_attenuation

    freqs = np.logspace(np.log10(50.0), np.log10(10000.0), 400)
    _, ax = plt.subplots(figsize=(10, 6.2))
    # The result's own .plot() draws alpha in dB/km on a 1k/2k-labelled log
    # frequency axis for the reference 20 degC / 50 % RH atmosphere (ISO 9613-1).
    atmospheric_attenuation(freqs, temperature=20.0, relative_humidity=50.0).plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_attenuation.png")
    plt.close()


def generate_outdoor_attenuation_breakdown(output_dir: str) -> None:
    """ISO 9613-2 per-term octave-band attenuation breakdown, with a barrier."""
    print("Generating outdoor_attenuation_breakdown.png...")
    from phonometry import Barrier, outdoor_propagation_attenuation

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # A point source 200 m away over porous ground (G = 1), screened by a 4 m
    # barrier midway (source and receiver 1,5 m high; the diffraction geometry
    # gives dss = dsr ~ 100 m over the raised edge).
    barrier = Barrier(source_to_edge=101.0, edge_to_receiver=101.0)
    att = outdoor_propagation_attenuation(
        200.0, 1.5, 1.5, bands, ground_source=1.0, ground_middle=1.0,
        ground_receiver=1.0, barrier=barrier, temperature=15.0, relative_humidity=70.0,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    # Separate positive and negative cumulative baselines so a negative term
    # (Agr is a net gain at 63 Hz here) stacks below zero instead of being
    # drawn on top of the previous bars; the signed heights sum to a_total.
    pos_bottom = np.zeros(len(bands))
    neg_bottom = np.zeros(len(bands))
    for term, color, label in [
        (att.a_div, COLOR_PRIMARY, "Adiv — divergence"),
        (att.a_atm, COLOR_TERTIARY, "Aatm — atmospheric"),
        (att.a_gr, "#9467bd", "Agr — ground"),
        (att.a_bar, "#ff7f0e", "Abar — barrier"),
    ]:
        bottom = np.where(term >= 0.0, pos_bottom, neg_bottom)
        ax.bar(x, term, bottom=bottom, color=color, edgecolor=COLOR_FG,
               linewidth=0.6, label=label, zorder=3)
        pos_bottom += np.maximum(term, 0.0)
        neg_bottom += np.minimum(term, 0.0)
    ax.plot(x, att.a_total, marker="D", color=COLOR_SECONDARY, linewidth=2.0,
            markersize=6, markerfacecolor="white", markeredgewidth=1.4,
            zorder=5, label="A — total")

    ax.set_title("ISO 9613-2 Attenuation Breakdown (with a 4 m barrier)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Attenuation A [dB]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "outdoor_attenuation_breakdown.png")
    plt.close()


def generate_cnossos_road_emission(output_dir: str) -> None:
    """CNOSSOS-EU road source-line power of an urban arterial traffic mix."""
    print("Generating cnossos_road_emission.png...")
    from phonometry import (
        JunctionType,
        RoadSurface,
        RoadTraffic,
        RoadVehicleCategory,
        road_source_power,
    )

    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # A two-lane urban arterial on a thin-layer surface: 1 200 light vehicles,
    # 90 medium heavy and 45 heavy vehicles per hour at 50 km/h, plus 60
    # motorcycles, 60 m before a signalised crossing, on a 3 % upgrade, at the
    # 12 degC yearly mean of a temperate city.
    traffic = [
        RoadTraffic(RoadVehicleCategory.LIGHT, 1200.0, 50.0),
        RoadTraffic(RoadVehicleCategory.MEDIUM_HEAVY, 90.0, 50.0),
        RoadTraffic(RoadVehicleCategory.HEAVY, 45.0, 50.0),
        RoadTraffic(RoadVehicleCategory.MOTORCYCLES, 60.0, 50.0),
    ]
    result = road_source_power(
        traffic, surface=RoadSurface.THIN_LAYER_A, temperature=12.0, gradient=3.0,
        junction_distance=60.0, junction_type=JunctionType.CROSSING,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.bar(x, result.total_line_power, color=COLOR_MUTED, edgecolor=COLOR_FG,
           linewidth=0.6, label="Total source line", zorder=2)
    labels = {
        "1": "Light vehicles (1)",
        "2": "Medium heavy vehicles (2)",
        "3": "Heavy vehicles (3)",
        "4b": "Motorcycles (4b)",
    }
    colors = [COLOR_PRIMARY, COLOR_TERTIARY, COLOR_SECONDARY, "#9467bd"]
    markers = ["o", "s", "D", "^"]
    for row, category, color, marker in zip(
        result.line_power, result.categories, colors, markers, strict=True
    ):
        ax.plot(x, row, color=color, marker=marker, linewidth=1.8, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3, zorder=5,
                label=labels[category.value])

    ax.set_title(
        "CNOSSOS-EU Road Source Line Power (urban arterial, 50 km/h)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Line power L'W,eq,line [dB re 1 pW/m]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_ylim(45.0, 90.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_emission.png")
    plt.close()


def generate_cnossos_road_speed_law(output_dir: str) -> None:
    """Rolling and propulsion noise against speed, and where they cross over."""
    print("Generating cnossos_road_speed_law.png...")
    from phonometry import (
        CNOSSOS_A_WEIGHTING,
        road_propulsion_noise,
        road_rolling_noise,
        road_vehicle_sound_power,
    )

    weights = np.asarray(CNOSSOS_A_WEIGHTING)

    def a_weighted(bands: np.ndarray) -> float:
        return float(10.0 * np.log10(np.sum(10.0 ** ((bands + weights) / 10.0))))

    speeds = np.linspace(20.0, 130.0, 221)
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    for category, color, name in [
        ("1", COLOR_PRIMARY, "Light vehicles (1)"),
        ("3", COLOR_SECONDARY, "Heavy vehicles (3)"),
    ]:
        rolling = np.array([a_weighted(road_rolling_noise(category, v)) for v in speeds])
        propulsion = np.array(
            [a_weighted(road_propulsion_noise(category, v)) for v in speeds]
        )
        total = np.array([a_weighted(road_vehicle_sound_power(category, v)) for v in speeds])
        ax.plot(speeds, total, color=color, linewidth=2.4, zorder=5, label=f"{name} — total")
        ax.plot(speeds, rolling, color=color, linewidth=1.3, linestyle="--", zorder=4,
                label=f"{name} — rolling")
        ax.plot(speeds, propulsion, color=color, linewidth=1.3, linestyle=":", zorder=4,
                label=f"{name} — propulsion")
        crossing = int(np.argmin(np.abs(rolling - propulsion)))
        ax.plot(speeds[crossing], rolling[crossing], marker="o", markersize=8,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.6,
                zorder=6)
        ax.annotate(
            f"crossover {speeds[crossing]:.0f} km/h",
            xy=(speeds[crossing], rolling[crossing]),
            xytext=(6, -16), textcoords="offset points", fontsize=9, color=color,
        )

    ax.set_title(
        "CNOSSOS-EU Single-Vehicle Sound Power against Speed (reference conditions)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Speed v [km/h]")
    ax.set_ylabel("A-weighted sound power LW,A [dB(A) re 1 pW]")
    ax.set_xlim(20.0, 130.0)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_road_speed_law.png")
    plt.close()


def generate_ground_effect_spherical(output_dir: str) -> None:
    """Spherical-wave ground effect (Weyl-Van der Pol) for four ground types."""
    print("Generating ground_effect_spherical.png...")
    import warnings as _warnings

    from phonometry import ground_effect

    freqs = np.geomspace(50.0, 4000.0, 400)
    # Effective flow resistivities (kPa s/m2 -> Pa s/m2) after Attenborough Ch. 2
    # / Salomons Sec. 3.1: fresh snow, forest floor, grassland and a near-hard
    # surface (asphalt / compacted soil).
    grounds = [
        ("Fresh snow (10 kPa·s·m⁻²)", 10e3, COLOR_TERTIARY),
        ("Forest floor (50 kPa·s·m⁻²)", 50e3, "#9467bd"),
        ("Grassland (200 kPa·s·m⁻²)", 200e3, COLOR_PRIMARY),
        ("Asphalt (20 000 kPa·s·m⁻²)", 20000e3, COLOR_SECONDARY),
    ]
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for label, sigma, color in grounds:
            res = ground_effect(freqs, 1.0, 1.5, 50.0, flow_resistivity=sigma)
            ax.plot(freqs, res.excess_attenuation, color=color, linewidth=1.8,
                    label=label, zorder=3)
    ax.axhline(6.0, color=COLOR_FG, linestyle=":", linewidth=1.0, alpha=0.7,
               label="Hard-ground limit (+6 dB)")
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_title("Spherical-Wave Ground Effect (Weyl-Van der Pol)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level re free field [dB]")
    ax.set_xlim(50.0, 4000.0)
    ax.set_ylim(-20.0, 8.0)
    format_frequency_axis(ax, 50.0, 4000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ground_effect_spherical.png")
    plt.close()


def generate_atmospheric_refraction(output_dir: str) -> None:
    """Atmospheric refraction: curved rays and the GFPE shadow-zone field."""
    print("Generating atmospheric_refraction.png...")
    import warnings as _warnings

    from phonometry import (
        atmospheric_parabolic_equation,
        atmospheric_ray_paths,
        log_linear_sound_speed_profile,
        shadow_zone_distance,
    )

    # Upward-refracting surface layer (b = -1 m/s) over grassland: rays curve
    # up and leave an acoustic shadow near the ground (Salomons Sec. 4.4/4.6).
    c0, b = 340.0, -1.0
    prof = log_linear_sound_speed_profile(b, ground_speed=c0, max_height=60.0)
    zs = 2.0
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True)

    # (a) Ray fan from a near-ground source.
    rays = atmospheric_ray_paths(prof, source_height=zs,
                                 launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                                 max_range=600.0, n_steps=3000)
    for i in range(rays.heights.shape[0]):
        axes[0].plot(rays.ranges[i], rays.heights[i], color=COLOR_PRIMARY,
                     lw=0.8, alpha=0.7, zorder=2)
    axes[0].plot([0.0], [zs], "o", color=COLOR_SECONDARY, ms=7, zorder=4,
                 label="Source")
    # The linear-gradient shadow distance (grazing ray at hr = zs) as a guide.
    grad = (prof.speed_at(10.0) - c0) / 10.0
    x_sh = shadow_zone_distance(float(grad), zs, zs, ground_speed=c0)
    axes[0].axvline(x_sh, color=COLOR_SECONDARY, ls="--", lw=1.2, zorder=3,
                    label="Shadow-zone boundary")
    axes[0].set_ylabel("Height [m]")
    axes[0].set_ylim(0.0, 40.0)
    axes[0].set_title("Sound rays (upward refraction)", fontweight="bold", pad=8)
    axes[0].grid(which="both", color=COLOR_GRID, ls="--", alpha=0.5, zorder=0)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper right", fontsize=9)

    # (b) GFPE relative-level field over the same atmosphere and ground.
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        pe = atmospheric_parabolic_equation(400.0, prof, source_height=zs,
                                            flow_resistivity=200e3,
                                            max_range=600.0, max_height=40.0)
    dl = np.where(np.isfinite(pe.relative_level), pe.relative_level, np.nan)
    img = axes[1].imshow(
        dl, cmap="RdBu_r", vmin=-30.0, vmax=6.0, aspect="auto", origin="lower",
        interpolation="bilinear",
        extent=(float(pe.ranges[0]), float(pe.ranges[-1]),
                float(pe.heights[0]), float(pe.heights[-1])),
        zorder=1,
    )
    axes[1].axvline(x_sh, color=COLOR_FG, ls="--", lw=1.2, alpha=0.8, zorder=2)
    axes[1].plot([0.0], [zs], "o", color="k", ms=6, zorder=3)
    axes[1].set_ylabel("Height [m]")
    axes[1].set_xlabel("Range [m]")
    axes[1].set_ylim(0.0, 40.0)
    axes[1].set_title("GFPE relative sound level", fontweight="bold", pad=8)
    fig.colorbar(img, ax=axes[1], label="Level re free field [dB]", pad=0.01)

    fig.suptitle("Atmospheric Refraction: Ray Bending and the Acoustic Shadow",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_refraction.png")
    plt.close()


def generate_exposure_uncertainty(output_dir: str) -> None:
    """ISO 9612 Annex D task-based exposure with its expanded uncertainty."""
    print("Generating exposure_uncertainty.png...")
    from phonometry.hearing.occupational_exposure import Task, task_based_exposure

    tasks = [
        Task(samples=(70.0,), duration_hours=1.5, label="planning/breaks"),
        Task(samples=(80.1, 82.2, 79.6), duration_hours=5.0,
             duration_range=(4.0, 6.0), label="welding"),
        Task(samples=(86.5, 92.4, 89.3, 93.2, 87.8, 86.2), duration_hours=1.5,
             duration_range=(1.0, 2.0), label="cutting/grinding"),
    ]
    result = task_based_exposure(tasks, include_duration_uncertainty=False,
                                 warn=False)
    labels = [t.label for t in result.tasks]
    contribs = [t.lex_8h_contribution for t in result.tasks]
    lex = result.lex_8h
    upper = result.upper_limit  # LEX,8h + U

    x = np.arange(len(labels))
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(x, contribs, color=COLOR_PRIMARY, edgecolor=COLOR_FG, linewidth=0.7,
           width=0.6, zorder=3, label="Measurement task")
    for xi, c in zip(x, contribs):
        ax.text(float(xi), c - 2.5, f"{c:.1f}", ha="center", va="top",
                fontsize=9, color="white", fontweight="bold")

    # Daily energy-summed level and its one-sided 95 % upper limit LEX,8h + U.
    ax.axhspan(lex, upper, color=COLOR_SECONDARY, alpha=0.14, zorder=0,
               label="LEX,8h + U (one-sided 95 %)")
    ax.axhline(lex, color=COLOR_SECONDARY, linewidth=2.0, zorder=4,
               label="Daily LEX,8h")
    ax.axhline(upper, color=COLOR_SECONDARY, linewidth=1.2, linestyle="--",
               zorder=4)

    box = [
        f"LEX,8h = {lex:.1f} dB",
        f"U = {result.expanded_uncertainty:.1f} dB (k = 1.65)",
        f"LEX,8h + U = {upper:.1f} dB",
    ]
    ax.text(0.03, 0.78, "\n".join(box), transform=ax.transAxes, va="top",
            ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})

    ax.set_title("ISO 9612 Task-Based Exposure (Annex D)", fontweight="bold",
                 pad=12)
    ax.set_xlabel("Measurement task")
    ax.set_ylabel("LEX,8h contribution [dB]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, upper + 10.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "exposure_uncertainty.png")
    plt.close()


def generate_absorption_rating(output_dir: str) -> None:
    """ISO 11654 alpha_w: practical curve, shifted reference, deviations (Annex A.2)."""
    print("Generating absorption_rating.png...")
    from phonometry import weighted_absorption

    # ISO 11654:1997 Annex A.2 worked example -> alpha_w = 0.60(M).
    alpha_p = [0.35, 1.00, 0.65, 0.60, 0.55]
    result = weighted_absorption(alpha_p)
    freqs = np.asarray(result.band_centers, dtype=float)
    measured = np.asarray(result.measured, dtype=float)
    shifted = np.asarray(result.shifted_reference, dtype=float)

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(freqs, measured, shifted, where=(measured < shifted).tolist(),
                    interpolate=True, color=COLOR_SECONDARY, alpha=0.25,
                    zorder=1, label="Unfavourable deviations")
    ax.semilogx(freqs, shifted, marker="s", color=COLOR_FG, linewidth=1.6,
                linestyle="--", markersize=5, zorder=3,
                label="Shifted reference curve (ISO 11654)")
    ax.semilogx(freqs, measured, marker="o", color=COLOR_PRIMARY, linewidth=1.8,
                markersize=6, markerfacecolor="white", markeredgewidth=1.4,
                zorder=4, label="Practical absorption alpha_p")

    # alpha_w is the shifted reference read at 500 Hz.
    ax.axvline(500, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(500, result.alpha_w, "D", color=COLOR_SECONDARY, markersize=9, zorder=6)
    ax.annotate(f"alpha_w = {result.rating_label}", xy=(500, result.alpha_w),
                xytext=(600, result.alpha_w - 0.16), fontsize=12, fontweight="bold",
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    # Placed low-left, clear of the practical curve that peaks top-centre.
    for dy, text in (
        (0.30, f"Reference curve shifted by {result.shift:.2f}"),
        (0.23, (f"Sum of unfavourable deviations = {result.unfavourable_sum:.2f}"
               f"  (limit 0.10)")),
        (0.16, (f"Absorption class {result.absorption_class}  "
               f"(shape indicator: {result.shape_indicator or 'none'})")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 11654 Weighted Sound Absorption Coefficient (Annex A.2 example)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound absorption coefficient")
    ax.set_xscale("log")
    ax.set_xlim(220, 4600)
    ax.set_ylim(0.0, 1.08)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(freqs)
    ax.set_xticklabels(["250", "500", "1k", "2k", "4k"], fontsize=9)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "absorption_rating.png")
    plt.close()


def generate_airflow_resistance(output_dir: str) -> None:
    """ISO 9053-1 static method: dp vs u, through-origin quadratic fit, R_s at 0.5 mm/s."""
    print("Generating airflow_resistance.png...")
    from phonometry import static_airflow_resistance

    # A porous specimen (area 100 mm dia, 50 mm thick) measured stepwise. The
    # pressure drop is slightly super-linear in velocity; the through-origin
    # quadratic fit dp = a*u + b*u**2 recovers R_s = a at the reference 0.5 mm/s.
    area = float(np.pi) * (0.05 ** 2)  # 100 mm diameter cell
    r_s_true, curvature = 1.6e4, 4.0e5  # Pa*s/m, Pa*s2/m2
    u = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0]) * 1e-3  # m/s
    dp = r_s_true * u + curvature * u**2
    result = static_airflow_resistance(u, dp, area=area, thickness=0.05)

    u_fit = np.linspace(0.0, 13e-3, 200)
    dp_fit = result.linear_coefficient * u_fit + result.quadratic_coefficient * u_fit**2

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(u_fit * 1e3, dp_fit, color=COLOR_PRIMARY, linewidth=1.8, zorder=2,
            label="Through-origin quadratic fit  dp = a u + b u^2")
    ax.plot(u * 1e3, dp, "o", color=COLOR_SECONDARY, markersize=7,
            markerfacecolor="white", markeredgewidth=1.6, zorder=4,
            label="Measured pressure drop")

    u_ref = result.evaluation_velocity
    ax.axvline(u_ref * 1e3, color=COLOR_FG, linestyle=":", alpha=0.4)
    ax.plot(u_ref * 1e3, result.pressure_drop, "D", color=COLOR_TERTIARY,
            markersize=9, zorder=6)
    ax.annotate("evaluation at 0.5 mm/s", xy=(u_ref * 1e3, result.pressure_drop),
                xytext=(2.0, result.pressure_drop + 40), fontsize=10,
                arrowprops={"arrowstyle": "->", "lw": 1.0})

    for dy, text in (
        (0.97, (f"Specific airflow resistance R_s = {result.specific_resistance:.0f}"
               f" Pa s/m")),
        (0.90, f"Airflow resistivity sigma = {result.resistivity:.0f} Pa s/m^2"),
        (0.83, (f"Linear term a = {result.linear_coefficient:.0f} Pa s/m"
               f"  (= R_s at u -> 0)")),
    ):
        ax.text(0.03, dy, text, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=COLOR_FG)

    ax.set_title("ISO 9053-1 Static-Method Airflow Resistance", fontweight="bold",
                 pad=12)
    ax.set_xlabel("Linear airflow velocity u [mm/s]")
    ax.set_ylabel("Pressure drop dp [Pa]")
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    save_figure(output_dir, "airflow_resistance.png")
    plt.close()


def generate_impedance_tube(output_dir: str) -> None:
    """ISO 10534-1 standing-wave-ratio method: alpha and |r| vs level difference."""
    print("Generating impedance_tube.png...")
    from phonometry import (
        standing_wave_absorption,
        standing_wave_ratio_from_level,
        standing_wave_reflection_magnitude,
    )

    # Level difference between pressure maximum and minimum (Eq. 15): a large dL
    # means a strong reflection (little absorption); dL -> 0 is a perfect absorber.
    level_diff = np.linspace(0.5, 40.0, 300)
    swr = np.array([standing_wave_ratio_from_level(dl) for dl in level_diff])
    alpha = np.array([standing_wave_absorption(s) for s in swr])
    r_mag = np.array([standing_wave_reflection_magnitude(s) for s in swr])

    _, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(level_diff, alpha, color=COLOR_PRIMARY, linewidth=2.0, zorder=3,
            label="Absorption coefficient alpha = 1 - |r|^2")
    ax.set_xlabel("Standing-wave level difference L_max - L_min [dB]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, 40.0)

    ax_r = ax.twinx()
    ax_r.plot(level_diff, r_mag, color=COLOR_SECONDARY, linewidth=1.8,
              linestyle="--", zorder=2, label="Reflection factor magnitude |r|")
    ax_r.set_ylabel("Reflection factor magnitude |r|")
    ax_r.set_ylim(0.0, 1.02)

    # Mark the didactic anchor: dL = 9.54 dB -> s = 3 -> |r| = 0.5 -> alpha = 0.75.
    dl_anchor = 20.0 * float(np.log10(3.0))
    ax.plot(dl_anchor, 0.75, "D", color=COLOR_TERTIARY, markersize=9, zorder=6)
    # Text sits in the lens that opens between the diverging alpha and |r| curves.
    ax.annotate("s = 3 -> |r| = 0.5 -> alpha = 0.75",
                xy=(dl_anchor, 0.75), xytext=(15.0, 0.44),
                fontsize=10, arrowprops={"arrowstyle": "->", "lw": 1.0})

    ax.set_title("ISO 10534-1 Standing-Wave-Ratio Method", fontweight="bold", pad=12)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    save_figure(output_dir, "impedance_tube.png")
    plt.close()


def generate_porous_absorber_designs(output_dir: str) -> None:
    """Multilayer absorber packages predicted by the transfer-matrix method.

    Normal-incidence absorption of four 50-mm-deep constructions built from
    the same porous model (Miki) and the resonant sheet layers: a plain
    porous layer, a Maa microperforated panel over a cavity, a perforated
    panel over porous + air, and a limp membrane over a porous-filled
    cavity. One concept: the same 2x2 layer matrices predict broadband and
    resonant absorbers alike.
    """
    print("Generating porous_absorber_designs...")
    import warnings as _warnings

    from phonometry import (
        AirLayer,
        MembraneLayer,
        MicroperforatedPlateLayer,
        PerforatedPlateLayer,
        PorousAbsorberWarning,
        PorousLayer,
        helmholtz_resonance_frequency,
        layered_absorber,
        membrane_resonance_frequency,
        miki,
    )
    from phonometry.materials.porous_absorber import Layer

    f = np.logspace(np.log10(50.0), np.log10(5000.0), 500)
    with _warnings.catch_warnings():
        # The 50 Hz decade end sits below the published Miki fit range on
        # purpose (the figure shows the bass behaviour of the resonators).
        _warnings.simplefilter("ignore", PorousAbsorberWarning)
        med = miki(f, 20000.0)
        med_light = miki(f, 10000.0)
        cases: list[tuple[str, list[Layer], str, str]] = [
            ("Porous layer 50 mm (sigma = 20 kPa s/m2)",
             [PorousLayer(0.05, med)], COLOR_PRIMARY, "-"),
            ("Microperforated panel + 48 mm cavity",
             [MicroperforatedPlateLayer(0.5e-3, 0.15e-3, 0.008),
              AirLayer(0.048)], COLOR_SECONDARY, "-"),
            ("Perforated panel 6 mm + porous 25 mm + air",
             [PerforatedPlateLayer(0.006, 0.0025, 0.05),
              PorousLayer(0.025, med), AirLayer(0.019)], COLOR_TERTIARY, "-"),
            ("Membrane 2 kg/m2 + air + porous 38 mm",
             [MembraneLayer(2.0), AirLayer(0.01),
              PorousLayer(0.038, med_light)], "#9467bd", "-"),
        ]
        _fig, ax = plt.subplots(figsize=(10, 6.2))
        for label, layers, color, ls in cases:
            res = layered_absorber(f, layers)
            ax.semilogx(f, res.absorption, ls, color=color, linewidth=2.2,
                        label=label)
    # Closed-form resonance anchors of the two classical designs.
    f_helm = helmholtz_resonance_frequency(
        cavity_depth=0.044, plate_thickness=0.006, hole_radius=0.0025,
        open_area=0.05,
    )
    f_mem = membrane_resonance_frequency(surface_density=2.0, cavity_depth=0.048)
    for f0, color, label in (
        (f_helm, COLOR_TERTIARY, "Helmholtz closed form"),
        (f_mem, "#9467bd", "Membrane closed form"),
    ):
        ax.axvline(f0, color=color, linestyle=":", linewidth=1.1, alpha=0.8)
        ax.text(f0 * 1.04, 0.44, label, rotation=90, va="bottom",
                ha="left", fontsize=8.5, color=color)

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(50.0, 5000.0)
    ax.set_title("Multilayer Absorber Prediction (Transfer-Matrix Method)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([63, 125, 250, 500, 1000, 2000, 4000])
    ax.set_xticklabels(["63", "125", "250", "500", "1k", "2k", "4k"])
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.03, "Normal incidence, rigid backing, 50 mm total depth",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "porous_absorber_designs.svg")
    plt.close()


def generate_limp_frame_effective_density(output_dir: str) -> None:
    """Rigid-frame against limp-frame effective density (Allard & Atalla 11.3.4).

    One concept: a light frame carries inertia. The rigid-frame equivalent
    fluid lets the imaginary part of the effective density run away as
    sigma/(j w) below the decoupling frequency, because a motionless frame
    forbids rigid-body motion of the sample; the limp correction of
    Eqs. (11.53)-(11.55) converges instead on the apparent total density
    rho_t = rho1 + phi rho0. Above the decoupling frequency the two models
    coincide, which is the plot's own check on the correction.
    """
    print("Generating limp_frame_effective_density...")
    from phonometry import (
        decoupling_frequency,
        johnson_champoux_allard,
        limp_frame,
    )

    # Allard & Atalla Table 11.2 (printed p. 254): the soft fibrous material
    # behind their Figure 11.2.
    porosity, resistivity, frame_density = 0.98, 25.0e3, 30.0
    f = np.linspace(1.0, 2000.0, 800)
    rigid = johnson_champoux_allard(
        f, resistivity, porosity=porosity, tortuosity=1.02,
        viscous_length=90e-6, thermal_length=180e-6,
    )
    limp = limp_frame(rigid, frame_density, porosity=porosity)
    rho0 = rigid.air_density
    total = frame_density + porosity * rho0
    f_d = decoupling_frequency(
        resistivity, porosity=porosity, frame_density=frame_density
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(f, np.real(rigid.effective_density) / rho0, "--",
            color=COLOR_PRIMARY, linewidth=1.8, label="rigid frame, real part")
    ax.plot(f, np.imag(rigid.effective_density) / rho0, ":",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, imaginary part")
    ax.plot(f, np.real(limp.effective_density) / rho0, "-",
            color=COLOR_SECONDARY, linewidth=2.4, label="limp frame, real part")
    ax.plot(f, np.imag(limp.effective_density) / rho0, "-",
            color=COLOR_TERTIARY, linewidth=2.4,
            label="limp frame, imaginary part")
    ax.axhline(total / rho0, color=COLOR_FG, linestyle="-", linewidth=1.0,
               alpha=0.45)
    ax.axvline(f_d, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    ax.annotate(
        f"apparent total density rho_t/rho0 = {total / rho0:.1f}",
        xy=(1960.0, total / rho0), xytext=(1960.0, total / rho0 - 1.4),
        ha="right", va="top", fontsize=9, color=COLOR_FG,
    )
    ax.annotate(f"decoupling frequency {f_d:.0f} Hz", xy=(f_d, -18.0),
                xytext=(f_d + 60.0, -18.0), ha="left", va="center",
                fontsize=9, color=COLOR_FG)

    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(-30.0, 30.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised effective density $\rho_e/\rho_0$")
    ax.set_title("A Limp Frame Carries Its Own Inertia", fontweight="bold",
                 pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.015, 0.03,
            "Soft fibrous layer: porosity 0.98, "
            "flow resistivity 25 kPa s/m², frame density 30 kg/m³",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "limp_frame_effective_density.svg")
    plt.close()


def generate_biot_frame_resonance(output_dir: str) -> None:
    """The frame resonance an equivalent fluid cannot produce (A&A 6.6.3).

    One concept: the surface impedance of a glass-wool layer glued to a rigid
    wall, computed with the full Biot theory and with the same material treated
    as a rigid-frame equivalent fluid. The two curves lie on top of each other
    everywhere except around the quarter-wavelength resonance of the
    frame-borne wave, where the Biot prediction develops the dip-and-peak in
    the real part and the sharp maximum in the imaginary part that Allard &
    Atalla measure and plot in their Figure 6.10. The closed form Eq. (6.110)
    marks where that resonance is expected.
    """
    print("Generating biot_frame_resonance...")
    from phonometry import (
        PoroelasticLayer,
        PorousLayer,
        frame_quarter_wave_resonance,
        johnson_champoux_allard,
        layered_absorber,
    )

    # Allard & Atalla Table 6.1 (printed p. 124): the glass wool "Domisol
    # Coffrage", with the characteristic lengths of their printed p. 123.
    porosity, tortuosity, resistivity = 0.94, 1.06, 40.0e3
    frame_density, shear_modulus = 130.0, 2.2e6 * (1.0 + 0.1j)
    thickness = 0.10
    f = np.linspace(200.0, 1500.0, 1300)
    medium = johnson_champoux_allard(
        f, resistivity, porosity=porosity, tortuosity=tortuosity,
        viscous_length=0.56e-4, thermal_length=1.1e-4,
    )
    rigid = layered_absorber(f, [PorousLayer(thickness, medium)])
    biot = layered_absorber(
        f,
        [PoroelasticLayer(thickness, medium, porosity, tortuosity,
                          frame_density, shear_modulus)],
    )
    f_r = frame_quarter_wave_resonance(
        thickness, shear_modulus=shear_modulus, poisson_ratio=0.0,
        frame_density=frame_density,
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(f, np.real(rigid.normalized_impedance), "--",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, real part")
    ax.plot(f, np.imag(rigid.normalized_impedance), ":",
            color=COLOR_PRIMARY, linewidth=1.8,
            label="rigid frame, imaginary part")
    ax.plot(f, np.real(biot.normalized_impedance), "-",
            color=COLOR_SECONDARY, linewidth=2.4,
            label="Biot poroelastic, real part")
    ax.plot(f, np.imag(biot.normalized_impedance), "-",
            color=COLOR_TERTIARY, linewidth=2.4,
            label="Biot poroelastic, imaginary part")
    ax.axvline(f_r, color=COLOR_FG, linestyle=":", linewidth=1.2, alpha=0.7)
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    ax.annotate(
        f"frame quarter-wave resonance {f_r:.0f} Hz",
        xy=(f_r, 2.7), xytext=(f_r + 25.0, 2.7),
        ha="left", va="center", fontsize=9, color=COLOR_FG,
    )

    ax.set_xlim(200.0, 1500.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel(r"Normalised surface impedance $Z_s/\rho_0 c_0$")
    ax.set_title("Only an Elastic Frame Resonates", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.015, 0.03,
            "Glass wool, 100 mm glued to a rigid wall: porosity 0.94,\n"
            "flow resistivity 40 kPa s/m², frame density 130 kg/m³,\n"
            "shear modulus 2.2 MPa",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "biot_frame_resonance.svg")
    plt.close()


def generate_slow_sound_absorber(output_dir: str) -> None:
    """Perfect absorption by critical coupling in a slow-sound slit panel.

    A single Helmholtz resonator loads a thin closed slit; the critical
    coupling design tunes the cavity length and slit height so the intrinsic
    visco-thermal losses exactly balance the leakage, giving alpha = 1 at
    300 Hz. Detuning the slit height (more or less loss) breaks the balance
    and lowers the peak. One concept: perfect absorption is a loss-versus-
    leakage balance in a deep-subwavelength (L = lambda/38) panel.
    """
    print("Generating slow_sound_absorber.svg...")
    from phonometry import (
        HelmholtzResonator,
        critical_coupling_design,
        slit_helmholtz_absorber,
    )

    lattice_step = 3.0e-2
    period = 5.0e-2
    f0 = 300.0
    base = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    design = critical_coupling_design(
        f0, base, lattice_step=lattice_step, period=period,
    )
    h0 = design.slit_height
    f = np.linspace(150.0, 500.0, 700)
    cases = [
        (h0, COLOR_SECONDARY, "-", "Critically coupled (perfect)"),
        (0.6 * h0, COLOR_PRIMARY, "--", "Narrow slit (over-damped)"),
        (1.7 * h0, COLOR_TERTIARY, "--", "Wide slit (under-damped)"),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    for height, color, ls, label in cases:
        res = slit_helmholtz_absorber(
            f, design.resonator, slit_height=height,
            lattice_step=lattice_step, period=period,
        )
        ax.plot(f, res.absorption, ls, color=color, linewidth=2.2, label=label)
    ax.axvline(f0, color=COLOR_FG, linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(f0 * 1.01, 0.05, "design 300 Hz", rotation=90, va="bottom",
            ha="left", fontsize=8.5, color=COLOR_FG)
    panel_depth = lattice_step  # slit depth L = N a with N = 1
    ratio = (343.0 / f0) / panel_depth
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Sound absorption coefficient alpha")
    ax.set_ylim(0.0, 1.08)
    ax.set_xlim(150.0, 500.0)
    ax.set_title("Perfect Absorption by Critical Coupling (Slow-Sound Panel)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.985, 0.03,
            f"Normal incidence, rigid backing, panel depth L = lambda/{ratio:.0f}",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "slow_sound_absorber.svg")
    plt.close()


def generate_absorber_stack_geometry(output_dir: str) -> None:
    """To-scale cross-section of a three-layer absorber design.

    The microperforated plate + air gap + porous layer stack drawn the way a
    lab manual would: each layer with its material fill and dimensioned
    thickness, the rigid backing at the right and the sound arriving from the
    left. One concept: the geometry behind the multilayer absorption curve.
    """
    print("Generating absorber_stack_geometry...")
    from phonometry import materials

    frequency = np.linspace(200.0, 4000.0, 100)
    medium = materials.miki(frequency, 20000.0)
    layers: list[Any] = [
        materials.MicroperforatedPlateLayer(0.001, 0.0002, 0.01),
        materials.AirLayer(0.03),
        materials.PorousLayer(0.05, medium),
    ]
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_absorber_stack(layers, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "absorber_stack_geometry.svg")
    plt.close()


def generate_slit_absorber_geometry(output_dir: str) -> None:
    """To-scale cross-section of the critically-coupled slit panel.

    One period of the slow-sound metamaterial absorber of the companion
    absorption figure: the 300 Hz critical-coupling design (single Helmholtz
    resonator loading a thin slit, panel depth lambda/38). One concept: what
    the deep-subwavelength panel actually looks like.
    """
    print("Generating slit_absorber_geometry...")
    from phonometry import (
        HelmholtzResonator,
        critical_coupling_design,
        materials,
    )

    base = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    design = critical_coupling_design(
        300.0, base, lattice_step=3.0e-2, period=5.0e-2,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_slit_absorber_geometry(
        [design.resonator], ax=ax, slit_height=design.slit_height,
        lattice_step=3.0e-2, period=5.0e-2, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "slit_absorber_geometry.svg")
    plt.close()


def generate_qrd_geometry(output_dir: str) -> None:
    """To-scale well profile of the N = 7 QRD of the polar-response figure.

    Two periods of the commercial N = 7 quadratic-residue diffuser (42 wells
    across 3,6 m, so an 85,7 mm pitch split into an 80,7 mm well plus a 5 mm
    fin; deepest well 0,2 m, design frequency 490 Hz) drawn as the physical
    profile the Fraunhofer prediction models. One concept: the
    quadratic-residue depth sequence as a real, buildable surface.
    """
    print("Generating qrd_geometry...")
    from phonometry import materials

    depths = materials.qrd_well_depths(7, 490.0, speed_of_sound=343.0)
    pitch = 3.6 / 42
    fin = 0.005
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_qrd_geometry(
        depths, pitch - fin, ax=ax, periods=2, fin_width=fin, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "qrd_geometry.svg")
    plt.close()


#: Table-1 metadiffuser rows (slit h, neck l_n, cavity l_c, neck w_n,
#: cavity w_c, in mm), shared by the figures and the animation mesh.
_METADIFFUSER_T1_ROWS = (
    (14.7, 13.0, 16.4, 6.2, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (30.9, 9.1, 4.3, 3.5, 9.0),
    (15.7, 13.3, 17.0, 6.3, 9.0),
    (20.3, 18.0, 20.7, 3.2, 9.0),
)


def _qr_metadiffuser_wells() -> tuple[Any, float, float]:
    """The published 2 cm quadratic-residue metadiffuser (wells, L, d)."""
    from phonometry import HelmholtzResonator, MetadiffuserWell

    rows = _METADIFFUSER_T1_ROWS
    wells = [
        MetadiffuserWell(
            h * 1e-3,
            (HelmholtzResonator(ln * 1e-3, wn * 1e-3, lc * 1e-3, wc * 1e-3),)
            * 2,
        )
        for h, ln, lc, wn, wc in rows
    ]
    return wells, 0.02, 0.07


def generate_metadiffuser_polar(output_dir: str) -> None:
    """A 2 cm metadiffuser scatters like the 27 cm QRD it mimics.

    Far-field polar responses at 2 kHz of the five-slit metadiffuser panel
    and of the quadratic-residue diffuser (design frequency 500 Hz, wells up
    to 27.4 cm deep) whose reflection-phase profile it reproduces, both with
    six repetitions. One concept: the deep-subwavelength panel replaces a
    13.7 times thicker classical diffuser.
    """
    print("Generating metadiffuser_polar...")
    from phonometry import (
        metadiffuser_polar_response,
        predict_diffuser_polar_response,
        quadratic_residue_sequence,
    )

    wells, depth, period = _qr_metadiffuser_wells()
    sequence = np.roll(quadratic_residue_sequence(5), -1)
    qrd_depths = sequence * (343.0 / 500.0) / (2 * 5)
    meta = metadiffuser_polar_response(
        2000.0, wells, depth=depth, period=period, periods=6,
    )
    qrd = predict_diffuser_polar_response(
        period, 2000.0, depths=qrd_depths, periods=6,
        include_obliquity=False,
    )
    _fig, ax = plt.subplots(
        figsize=(10, 6.2), subplot_kw={"projection": "polar"},
    )
    meta.plot(
        ax=ax, color=COLOR_SECONDARY, marker="", linewidth=2.2,
        label="Metadiffuser, panel 2 cm", language=_LANG,
    )
    qrd.plot(
        ax=ax, color=COLOR_PRIMARY, marker="", linewidth=1.6,
        linestyle="--", label="QRD, wells up to 27.4 cm", language=_LANG,
    )
    ax.set_title(
        "The 2 cm metadiffuser scatters like the 27 cm QRD (2 kHz)",
        pad=18, fontweight="bold",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_polar.svg")
    plt.close()


def generate_metadiffuser_geometry(output_dir: str) -> None:
    """To-scale cross-section of the published 2 cm metadiffuser panel.

    One period of the five-slit quadratic-residue metadiffuser: numbered
    slits open at the face, each loaded by two Helmholtz resonators
    shelved sideways into the septum, over a rigid backing. One concept:
    the whole 35 cm x 2 cm panel that replaces a 27 cm deep diffuser.
    """
    print("Generating metadiffuser_geometry...")
    from phonometry.materials import plot_metadiffuser_panel_geometry

    wells, depth, period = _qr_metadiffuser_wells()
    _fig, ax = plt.subplots(figsize=(10, 3.4))
    plot_metadiffuser_panel_geometry(
        wells, ax=ax, depth=depth, period=period, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_geometry.svg")
    plt.close()


@lru_cache(maxsize=1)
def _meshed_metadiffuser_ntff_levels() -> tuple[Any, Any]:
    """Far-field polar levels of the meshed Table-1 panel at 2 kHz.

    A dedicated continuous-wave FDTD run at dx = 0.5 mm (every slit, neck
    and cavity of the real panel resolved by at least four cells, unit-cell
    reflection phases within 2 degrees of a 0.25 mm run): a plane wave from
    just inside the top sponge drives the panel to steady state, a closed
    contour probe accumulates the 2 kHz pressure and normal-velocity
    phasors on the fly, and the 2D Kirchhoff-Helmholtz integral turns them
    into the far-field pattern. The incident wave extinguishes in the
    exterior integral, so no reference run is needed. Returns the polar
    angles [deg from the panel normal] and the levels re the peak [dB].
    """
    from phonometry.simulation import (
        FDTD2D,
        CWSource,
        PlaneWaveSource,
        far_field_from_contour,
    )

    dx, sponge, f0 = 0.0005, 60, _META_F0
    gap, marg, front = (round(m / dx) for m in (0.030, 0.010, 0.020))
    face_cells = round(5 * _META_PITCH / dx)
    lat = marg + gap + sponge
    nx = face_cells + 2 * lat
    r_face = sponge + gap + front
    slab = round(0.023 / dx)
    ny = r_face + slab + marg + gap + sponge
    mask = np.zeros((ny, nx), dtype=bool)
    mask[r_face:r_face + slab, lat:lat + face_cells] = True
    for n, (h, l_n, l_c, w_n, w_c) in enumerate(_METADIFFUSER_T1_ROWS):
        x_slit = (n + 0.12) * _META_PITCH
        c0s = lat + round(x_slit / dx)
        c1s = lat + round((x_slit + h * 1e-3) / dx)
        mask[r_face:r_face + round(0.02 / dx), c0s:c1s] = False
        for m in range(2):
            y_m = (m + 0.5) * 0.01
            x_neck = x_slit + h * 1e-3
            r0 = r_face + round((y_m - 0.5e-3 * w_n) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_n) / dx)
            mask[r0:r1, c1s:lat + round((x_neck + l_n * 1e-3) / dx)] = False
            r0 = r_face + round((y_m - 0.5e-3 * w_c) / dx)
            r1 = r_face + round((y_m + 0.5e-3 * w_c) / dx)
            mask[r0:r1, lat + round((x_neck + l_n * 1e-3) / dx):
                 lat + round((x_neck + (l_n + l_c) * 1e-3) / dx)] = False
    # cfl 0.9 trims the step count; at 340 cells per wavelength the
    # numerical dispersion is negligible at any stable Courant number.
    sim = FDTD2D(343.0, dx, shape=(ny, nx), sponge_width=sponge, cfl=0.9,
                 obstacle_mask=mask)
    sim.add_source(PlaneWaveSource("down", CWSource(0, 0, f0).value,
                                   offset=sponge))
    probe = sim.add_contour_probe(lat - marg, lat + face_cells + marg - 1,
                                  r_face - front, r_face + slab + marg - 1,
                                  frequencies=[f0])
    sim.run(round(8e-3 / sim.dt))            # transient out (ramp + ring-up)
    probe.reset()
    sim.run(round(10.0 / f0 / sim.dt))       # ten whole periods of DFT
    angles = np.arange(-90.0, 90.1, 5.0)
    pattern = far_field_from_contour(
        probe.phasors(f0), angles - 90.0,
        origin=((lat + face_cells / 2.0) * dx, r_face * dx))
    magnitude = np.abs(pattern)
    levels = 20.0 * np.log10(np.maximum(magnitude / magnitude.max(), 1e-10))
    return angles, levels


def generate_metadiffuser_ntff_polar(output_dir: str) -> None:
    """The meshed metadiffuser radiates the far field the model predicts.

    Far-field polar response of the Table-1 metadiffuser at 2 kHz twice
    over: the TMM + Fraunhofer prediction of the library and a full 2D
    FDTD simulation of the meshed panel reduced through the
    Kirchhoff-Helmholtz near-to-far-field integral. One concept: the
    model chain is validated end to end by an independent full-wave
    solver, the library-side counterpart of the paper's TMM-vs-FEM
    cross-check.
    """
    print("Generating metadiffuser_ntff_polar...")
    from phonometry import metadiffuser_polar_response

    wells, depth, period = _qr_metadiffuser_wells()
    model = metadiffuser_polar_response(
        2000.0, wells, depth=depth, period=period, periods=1,
    )
    angles, levels = _meshed_metadiffuser_ntff_levels()
    _fig, ax = plt.subplots(
        figsize=(10, 6.2), subplot_kw={"projection": "polar"},
    )
    model.plot(
        ax=ax, color=COLOR_PRIMARY, marker="", linewidth=1.6,
        linestyle="--", label="TMM + Fraunhofer model", language=_LANG,
    )
    ax.plot(
        np.radians(angles), levels, color=COLOR_SECONDARY, linewidth=2.2,
        label="FDTD + NTFF, panel meshed at 0.5 mm",
    )
    ax.set_ylim(-40.0, 2.0)
    ax.set_title(
        "Far field of the meshed metadiffuser vs the model (2 kHz)",
        pad=18, fontweight="bold",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "metadiffuser_ntff_polar.svg")
    plt.close()


def generate_impedance_tube_geometry(output_dir: str) -> None:
    """To-scale side view of a 100 mm ISO 10534-2 impedance tube.

    Loudspeaker, the two flush microphones (s = 50 mm, x1 = 150 mm), the
    sample against its rigid backing, the circular cross-section and the
    plane-wave working range those dimensions buy. One concept: the tube
    geometry fixes the usable frequency band.
    """
    print("Generating impedance_tube_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_impedance_tube_geometry(
        ax=ax, spacing=0.05, x1=0.15, diameter=0.10, shape="circular",
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "impedance_tube_geometry.svg")
    plt.close()


def generate_transmission_tube_geometry(output_dir: str) -> None:
    """To-scale side view of a 100 mm ASTM E2611 transmission tube.

    Four flush microphones around the specimen (l1 = 100 mm, s1 = 50 mm,
    l2 = 200 mm, s2 = 50 mm), the changeable termination of the two-load
    method and the ASTM working range. One concept: where the four
    microphones of the transfer-matrix method actually sit.
    """
    print("Generating transmission_tube_geometry...")
    from phonometry import materials

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    materials.plot_transmission_tube_geometry(
        ax=ax, l1=0.10, s1=0.05, l2=0.20, s2=0.05, thickness=0.05,
        diameter=0.10, shape="circular", language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "transmission_tube_geometry.svg")
    plt.close()


def generate_helmholtz_resonator_geometry(output_dir: str) -> None:
    """To-scale cross-section of the 300 Hz slow-sound Helmholtz resonator.

    The square-section resonator of the critical-coupling design (neck 1 x 3
    mm, cavity 30 x 27 mm) with its four defining dimensions. One concept:
    the resonator geometry the slit-panel figures build on.
    """
    print("Generating helmholtz_resonator_geometry...")
    from phonometry import HelmholtzResonator

    resonator = HelmholtzResonator(
        neck_length=1.0e-3, neck_side=3.0e-3,
        cavity_length=30.0e-3, cavity_side=27.0e-3,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    resonator.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "helmholtz_resonator_geometry.svg")
    plt.close()


def generate_expansion_chamber_geometry(output_dir: str) -> None:
    """To-scale cross-section of the 4:1 expansion chamber of the TL figure.

    The 0,3 m chamber with 0,04 m2 over a 0,01 m2 pipe, drawn with the
    equivalent circular diameters. One concept: the geometry behind the
    expansion-chamber transmission-loss curve.
    """
    print("Generating expansion_chamber_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "expansion chamber", ax=ax, length=0.3, chamber_area=0.04,
        pipe_area=0.01, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "expansion_chamber_geometry.svg")
    plt.close()


def generate_helmholtz_branch_geometry(output_dir: str) -> None:
    """To-scale cross-section of the side-branch Helmholtz resonator.

    The resonator of the side-branch TL figure (1 cm2 neck of 2 cm on a
    1 L cavity over a 0,01 m2 duct), cavity drawn as its equal-volume cube.
    One concept: the branch geometry that shorts the duct at its tuning.
    """
    print("Generating helmholtz_branch_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "Helmholtz resonator", ax=ax, duct_area=0.01, neck_area=1e-4,
        neck_length=0.02, cavity_volume=1e-3, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "helmholtz_branch_geometry.svg")
    plt.close()


def generate_quarter_wave_geometry(output_dir: str) -> None:
    """To-scale cross-section of the quarter-wave side branch.

    The 0,3 m closed tube (20 cm2) of the side-branch TL figure on the same
    0,01 m2 duct. One concept: a quarter-wave stub is just a closed tube of
    the right length.
    """
    print("Generating quarter_wave_geometry...")
    from phonometry import plot_silencer_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_silencer_geometry(
        "quarter-wave resonator", ax=ax, duct_area=0.01, length=0.3,
        branch_area=2e-3, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "quarter_wave_geometry.svg")
    plt.close()


def generate_image_source_plan(output_dir: str) -> None:
    """Plan view of the image-source lattice of the reflectogram example.

    The 7 x 5 x 3 m room with its source and receiver, and every image up
    to third order coloured by reflection order over the mirror-room grid.
    One concept: where the reflections of the reflectogram actually come
    from.
    """
    print("Generating image_source_plan...")
    from phonometry import image_source_rir

    res = image_source_rir(
        (7.0, 5.0, 3.0), (2.0, 1.6, 1.5), (5.2, 3.4, 1.7), 0.3,
        fs=16000, max_order=3,
    )
    _fig, ax = plt.subplots(figsize=(10, 6.6))
    res.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "image_source_plan.svg")
    plt.close()


def generate_barrier_geometry(output_dir: str) -> None:
    """To-scale section of the barrier of the insertion-loss figure.

    The 4 m screen 50 m from a 1 m source with the receiver 1,5 m high at
    100 m: direct and diffracted paths and the path-length difference that
    drives the Fresnel number. One concept: the geometry the insertion-loss
    methods share.
    """
    print("Generating barrier_geometry...")
    from phonometry import plot_barrier_geometry

    _fig, ax = plt.subplots(figsize=(10, 5.4))
    plot_barrier_geometry(
        ax=ax, source_height=1.0, barrier_distance=50.0,
        barrier_height=4.0, receiver_distance=100.0, receiver_height=1.5,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "barrier_geometry.svg")
    plt.close()


def generate_microphone_positions_hemisphere(output_dir: str) -> None:
    """The ISO 3744 engineering hemisphere microphone array in 3-D.

    The tabulated 10-microphone array on a 2 m hemisphere over one
    reflecting plane, numbered as in the standard. One concept: where the
    sound-power microphones actually sit.
    """
    print("Generating microphone_positions_hemisphere...")
    from phonometry import measurement_positions, plot_microphone_positions

    positions = measurement_positions("hemisphere", radius=2.0)
    plt.figure(figsize=(9.0, 7.0))
    ax = plt.subplot(projection="3d")
    plot_microphone_positions(positions, ax=ax, radius=2.0, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "microphone_positions_hemisphere.svg")
    plt.close()


def generate_aperture_slit_geometry(output_dir: str) -> None:
    """To-scale section of a 2 mm slit through a 100 mm wall.

    The slit of the aperture transmission-loss example: a deep, narrow gap
    whose depth resonances puncture the wall's insulation. One concept: the
    tiny geometry behind a large leak.
    """
    print("Generating aperture_slit_geometry...")
    from phonometry import plot_aperture_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_aperture_geometry(0.1, ax=ax, width=0.002, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "aperture_slit_geometry.svg")
    plt.close()


def generate_piston_baffle_geometry(output_dir: str) -> None:
    """The baffled piston to scale with its high-frequency lobe.

    A 10 cm piston in an infinite baffle with the normalised far-field
    directivity overlaid at ka about 7 (where the main lobe has narrowed).
    One concept: the piston the radiation-impedance and directivity curves
    describe.
    """
    print("Generating piston_baffle_geometry...")
    from phonometry import radiating_piston

    result = radiating_piston(
        0.1, np.array([500.0, 2000.0, 4000.0]),
        angles=np.linspace(-np.pi / 2.0, np.pi / 2.0, 181),
    )
    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    result.plot_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "piston_baffle_geometry.svg")
    plt.close()


def generate_plenum_geometry(output_dir: str) -> None:
    """Plenum chamber section honouring the acoustic geometry exactly.

    A 1,2 m line of sight at 20 degrees off the inlet axis, a 0,09 m2
    outlet and 6 m2 of lined walls: the two geometric parameters of the
    attenuation formula are drawn exactly and the areas annotated. One
    concept: what the plenum formula actually measures.
    """
    print("Generating plenum_geometry...")
    from phonometry import plot_plenum_geometry

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    plot_plenum_geometry(0.09, 1.2, 6.0, ax=ax, angle=0.35, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "plenum_geometry.svg")
    plt.close()


def generate_fdtd_domain_geometry(output_dir: str) -> None:
    """The configured FDTD domain drawn before any time stepping.

    A 4,5 x 3 m domain with sponge layers left and right, an impedance top
    edge, a rigid floor, a thin obstacle, one source and two probes: the
    setup drawing that precedes every simulation. One concept: check the
    domain before you run it.
    """
    print("Generating fdtd_domain_geometry...")
    from phonometry.simulation import FDTD2D, GaussianPulse

    mask = np.zeros((60, 90), dtype=bool)
    mask[25:35, 40:44] = True
    sim = FDTD2D(
        343.0, 0.05, shape=(60, 90), sponge_width=8,
        sponge_sides=("left", "right"), edge_impedance={"top": 413.0},
        obstacle_mask=mask,
    )
    sim.add_source(GaussianPulse(10, 30, width=1e-3))
    _fig, ax = plt.subplots(figsize=(10, 6.6))
    sim.plot_geometry(ax=ax, probes=[(3.0, 1.5), (4.0, 2.0)],
                      language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_domain_geometry.svg")
    plt.close()


def generate_facade_elevation_geometry(output_dir: str) -> None:
    """Composite facade elevation with the element areas to scale.

    A masonry wall of 6 m2 with a 1,5 m2 window and its 0,3 m2 roller
    shutter box, the classic composite of the facade prediction. One
    concept: the areas the composite sound reduction index weighs.
    """
    print("Generating facade_elevation_geometry...")
    from phonometry import FacadeElement, plot_facade_elements

    elements = [
        FacadeElement("Masonry wall", area=6.0, r=[50.0] * 5),
        FacadeElement("Window", area=1.5, r=[30.0] * 5),
        FacadeElement("Roller shutter box", area=0.3, r=[22.0] * 5),
    ]
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_facade_elements(elements, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "facade_elevation_geometry.svg")
    plt.close()


def generate_double_wall_geometry(output_dir: str) -> None:
    """Mass-spring-mass double wall to scale.

    Two 12,5 mm plasterboard leaves (8,8 kg/m2 each) on a 100 mm stud
    cavity, the classic lightweight double wall, with its mass-air-mass
    resonance annotated. One concept: the geometry behind the double-wall
    resonance dip.
    """
    print("Generating double_wall_geometry...")
    from phonometry import mass_spring_mass_resonance, plot_double_wall_geometry

    f0 = mass_spring_mass_resonance(8.8, 8.8, 0.1)
    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_double_wall_geometry(
        8.8, 8.8, 0.1, ax=ax, resonance_frequency=f0, language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "double_wall_geometry.svg")
    plt.close()


def generate_junction_plate_geometry(output_dir: str) -> None:
    """A T-junction of heavyweight plates to scale.

    A 140 mm concrete floor ending against a continuous 200 mm wall (the
    T-junction whose perpendicular plates are the identical pair), the
    incident bending wave marked. One concept: the junction the
    transmission coefficients describe.
    """
    print("Generating junction_plate_geometry...")
    from phonometry import plot_junction_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_junction_geometry("T2", 0.14, 0.2, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "junction_plate_geometry.svg")
    plt.close()


def generate_insitu_setup_geometry(output_dir: str) -> None:
    """The in-situ road absorption set-up to scale.

    Loudspeaker at 1,25 m, microphone at 0,25 m on the same vertical, and
    the 1,34 m sampled-area radius of the standard 5 ms window on the
    surface. One concept: what the subtraction technique actually samples.
    """
    print("Generating insitu_setup_geometry...")
    from phonometry import plot_insitu_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.0))
    plot_insitu_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "insitu_setup_geometry.svg")
    plt.close()


def generate_dynamic_stiffness_rig_geometry(output_dir: str) -> None:
    """The dynamic-stiffness resonance rig to scale.

    The standard 200 mm square resilient specimen under the 8 kg load
    plate, exciter above and accelerometer on the plate. One concept: the
    little mass-spring oscillator behind s' and the floating-floor f0.
    """
    print("Generating dynamic_stiffness_rig_geometry...")
    from phonometry import plot_dynamic_stiffness_rig

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_dynamic_stiffness_rig(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "dynamic_stiffness_rig_geometry.svg")
    plt.close()


def generate_diffusion_goniometer_geometry(output_dir: str) -> None:
    """The free-field diffusion goniometer in plan, to scale.

    The 37-microphone semicircle at 5 m, the source at 10 m on the normal
    and the sample at the centre. One concept: where the polar responses
    of the diffusion coefficient are measured.
    """
    print("Generating diffusion_goniometer_geometry...")
    from phonometry import plot_goniometer_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.6))
    plot_goniometer_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "diffusion_goniometer_geometry.svg")
    plt.close()


def generate_radiation_plate_geometry(output_dir: str) -> None:
    """The baffled plate of the radiation-efficiency model, to scale.

    The 1,5 x 1,25 m simply supported plate in its rigid baffle, the
    geometry the sigma(f) curves describe. One concept: the radiator
    behind the radiation-efficiency plateau and coincidence peak.
    """
    print("Generating radiation_plate_geometry...")
    from phonometry import plot_plate_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 6.2))
    plot_plate_geometry(1.5, 1.25, ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "radiation_plate_geometry.svg")
    plt.close()


def generate_open_plan_line_geometry(output_dir: str) -> None:
    """The open-plan measurement line to scale.

    Six microphones from 2 m to 16 m across the workstations with the
    distraction and privacy distances marked. One concept: the single
    line every ISO 3382-3 quantity comes from.
    """
    print("Generating open_plan_line_geometry...")
    from phonometry import plot_open_plan_geometry

    _fig, ax = plt.subplots(figsize=(10, 4.6))
    plot_open_plan_geometry(
        [2.0, 4.0, 6.0, 8.0, 12.0, 16.0], ax=ax, rd=6.5, rp=13.0,
        language=_LANG,
    )
    plt.tight_layout()
    save_figure(output_dir, "open_plan_line_geometry.svg")
    plt.close()


def generate_pp_probe_geometry(output_dir: str) -> None:
    """The face-to-face p-p intensity probe to scale.

    Two phase-matched half-inch microphones on the classic 12 mm solid
    spacer, the intensity axis through both. One concept: the finite
    difference the p-p method is built on.
    """
    print("Generating pp_probe_geometry...")
    from phonometry import plot_pp_probe_geometry

    _fig, ax = plt.subplots(figsize=(9.0, 5.4))
    plot_pp_probe_geometry(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "pp_probe_geometry.svg")
    plt.close()


def generate_scattering_coefficient(output_dir: str) -> None:
    """ISO 17497-1: scattering coefficient s(f) from a per-band measurement."""
    print("Generating scattering_coefficient.png...")
    from phonometry import scattering_coefficient_spectrum

    # A realistic reverberation-room measurement reduced to two absorption
    # spectra over the 13 one-third-octave bands 250-4000 Hz: the random-
    # incidence absorption alpha_s (stationary sample) and the specular
    # absorption alpha_spec (rotating turntable). A diffuser scatters more with
    # frequency, so alpha_spec climbs above alpha_s and s(f) = (alpha_spec -
    # alpha_s)/(1 - alpha_s) rises smoothly from near 0 towards 0.8.
    freqs = np.array(
        [250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000],
        dtype=float,
    )
    alpha_s = np.full_like(freqs, 0.10)
    alpha_spec = 0.11 + 0.75 * (np.log10(freqs / 250.0) / np.log10(4000.0 / 250.0))
    result = scattering_coefficient_spectrum(freqs, alpha_spec, alpha_s)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.semilogx(result.frequencies, result.scattering, color=COLOR_PRIMARY,
                linewidth=1.9, marker="o", markersize=6, markerfacecolor="white",
                markeredgewidth=1.4, zorder=3)
    ax.set_title("Random-incidence scattering coefficient (ISO 17497-1)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Scattering coefficient s")
    ax.set_xlim(freqs.min() * 0.9, freqs.max() * 1.1)
    ax.set_ylim(0.0, 1.0)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([250, 500, 1000, 2000, 4000])
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "scattering_coefficient.png")
    plt.close()


def generate_diffusion_polar(output_dir: str) -> None:
    """ISO 17497-2: polar reflected response and its diffusion coefficient d."""
    print("Generating diffusion_polar.png...")
    from phonometry import (
        directional_diffusion,
        predict_diffuser_polar_response,
        qrd_well_depths,
    )

    # Reflected sound-pressure levels L_i(theta) on the standard 37-point
    # semicircle (-90 to 90 deg, 5 deg spacing) of a published diffuser
    # geometry: the N = 7 QRD, 6 periods, 3.6 m wide, 0.2 m deep array of
    # Cox & D'Antonio 3e Appendix B section 7 (the commercial N = 7 QRD of
    # Hargreaves et al. 2000, Table I), predicted with the library's own
    # Fraunhofer far-field model at 1000 Hz, normal incidence, and rounded to
    # the 1e-3 dB precision of the committed tests/reference_data.py arc. Six
    # periods of a periodic QRD concentrate the reflected energy into grating
    # lobes, so the ISO 17497-2 Formula (5) coefficient d is modest.
    angles = np.arange(-90.0, 90.5, 5.0)
    depths = qrd_well_depths(7, 490.0, speed_of_sound=343.0)  # deepest 0.2 m
    predicted = predict_diffuser_polar_response(
        3.6 / 42, 1000.0, depths=depths, periods=6, speed_of_sound=343.0,
    )
    result = directional_diffusion(angles, np.round(predicted.levels, 3))

    _fig, ax = plt.subplots(figsize=(8.0, 7.5),
                           subplot_kw={"projection": "polar"})
    # The theta-* setters live on PolarAxes, not the base Axes type.
    polar: Any = ax
    theta = np.radians(result.angles)
    polar.plot(theta, result.levels, color=COLOR_PRIMARY, linewidth=1.9,
               marker="o", markersize=4, zorder=3)
    # Translucent so the polar grid keeps reading through the lobe.
    polar.fill(theta, result.levels, color=COLOR_PRIMARY, zorder=1,
               alpha=theme_fill_alpha(COLOR_PRIMARY, ax))
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_thetamin(-90)
    polar.set_thetamax(90)
    polar.set_title(
        f"Directional diffusion  d = {result.coefficient:.2f}  (ISO 17497-2)",
        fontweight="bold", pad=20,
    )
    plt.tight_layout()
    save_figure(output_dir, "diffusion_polar.png")
    plt.close()


def generate_diffuser_prediction(output_dir: str) -> None:
    """Predicted diffusion d(f): an N = 7 QRD design versus a flat panel."""
    print("Generating diffuser_prediction.png...")
    from phonometry import (
        predicted_diffusion_spectrum,
        qrd_well_depths,
    )

    # An N = 7 quadratic residue diffuser, design frequency 500 Hz, 10 cm wells,
    # five periods (Cox & D'Antonio far-field Fraunhofer model). The predicted
    # diffusion is compared band by band with the flat reference panel of the
    # same footprint: the QRD spreads the reflected energy far more evenly, so
    # its diffusion coefficient sits well above the near-specular flat panel.
    freqs = np.array([250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
                      2000, 2500, 3150, 4000, 5000], float)
    depths = qrd_well_depths(7, 500.0)
    qrd = predicted_diffusion_spectrum(0.10, freqs, depths=depths, periods=5)
    flat = predicted_diffusion_spectrum(
        0.10, freqs, depths=np.zeros_like(depths), periods=5, normalize=False
    )

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs, qrd.diffusion, color=COLOR_PRIMARY, linewidth=1.9,
                marker="o", markersize=4, label="N = 7 QRD design")
    ax.semilogx(freqs, flat.diffusion, color=COLOR_SECONDARY, linewidth=1.9,
                marker="s", markersize=4, linestyle="--", label="Flat panel")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Predicted diffusion coefficient d")
    format_frequency_axis(ax, 250.0, 5000.0)
    ax.set_title(
        "Predicted diffusion from design (Cox & D'Antonio Fraunhofer model)",
        fontweight="bold",
    )
    ax.legend(loc="best")
    plt.tight_layout()
    save_figure(output_dir, "diffuser_prediction.png")
    plt.close()


def generate_insitu_absorption(output_dir: str) -> None:
    """ISO 13472-1: in-situ one-third-octave absorption spectrum alpha(f)."""
    print("Generating insitu_absorption.png...")
    from phonometry import geometric_spreading_factor, insitu_absorption_spectrum

    # A synthetic-but-realistic in-situ measurement. The incident impulse hi is
    # a unit spike; the road reflection is hr = Kr * r0 * roll(hi, shift) with
    # Kr the geometrical-spreading factor (2/3 for ds=1.25 m, dm=0.25 m), a
    # mildly frequency-dependent r0 realised by a gentle low-pass (a porous
    # surface reflects less as frequency rises), and the reflected-path delay
    # shift = round(2 dm / c * fs). The library forms the narrow-band
    # alpha = 1 - (1/Kr^2)|Hr/Hi|^2 and reduces it to one-third-octave bands.
    fs, n = 48000.0, 8192
    kr = geometric_spreading_factor()  # (ds - dm)/(ds + dm) = 2/3
    hi = np.zeros(n)
    hi[0] = 1.0
    r0 = 0.85
    taps = scipy_signal.firwin(41, 1200.0, fs=fs)
    taps = taps / taps.sum()
    shift = round(2.0 * 0.25 / 340.0 * fs)  # reflected-path delay 2 dm / c
    hr = kr * r0 * np.roll(scipy_signal.lfilter(taps, 1.0, hi), shift)
    result = insitu_absorption_spectrum(hi, hr, fs)

    freqs = result.frequencies
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, np.nan_to_num(result.absorption), width=0.7,
           color=COLOR_PRIMARY, edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title("In-situ road-surface absorption (ISO 13472-1)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Absorption coefficient alpha")
    ax.set_ylim(0.0, 1.0)
    ax.text(0.04, 0.94, "Kr = 2/3\nalpha = 1 - (1/Kr^2)|Hr/Hi|^2",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "insitu_absorption.png")
    plt.close()


def generate_sound_power_pressure_result(output_dir: str) -> None:
    """ISO 3744: enveloping-surface LW spectrum from hemisphere pressure levels."""
    print("Generating sound_power_pressure_result.png...")
    from phonometry import sound_power_pressure

    # The sound-power guide's section-1 example: octave-band SPL at the 10
    # hemisphere positions of ISO 3744 (Annex B) around a machine on one
    # reflecting plane, with a flat 55 dB background, corrected for background
    # (K1) and for the test room (K2 from T = 0.6 s, V = 300 m^3). The library
    # forms LW = Lp_bar - K1 - K2 + 10 lg(S/S0) per band and the A-weighted
    # total LWA.
    freqs = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = np.array([70.0, 74.0, 78.0, 80.0, 79.0, 76.0, 72.0, 66.0])
    rng = np.random.default_rng(0)
    levels = base + rng.normal(0.0, 0.5, size=(10, 8))
    background = np.full((10, 8), 55.0)
    result = sound_power_pressure(
        levels, "hemisphere", radius=1.5, reflecting_planes=1,
        background_levels=background, frequencies=freqs,
        reverberation_time=0.6, volume=300.0,
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Enveloping-surface sound power (ISO 3744)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_pressure_result.png")
    plt.close()


def generate_sound_power_reverberation_result(output_dir: str) -> None:
    """ISO 3741: reverberation-room LW spectrum (direct method)."""
    print("Generating sound_power_reverberation_result.png...")
    from phonometry.emission import sound_power_reverberation

    # The sound-power guide's section-2 example: one-third-octave mean room
    # SPL from 100 Hz to 10 kHz in a qualified 200 m^3 reverberation room with
    # T60 = 2 s, carried to LW through the Sabine absorption area, the
    # Waterhouse correction and the meteorological corrections C1/C2
    # (ISO 3741 Eq. 20).
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                      1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                      10000], dtype=float)
    lp = np.linspace(80.0, 70.0, freqs.size)
    t60 = np.full(freqs.size, 2.0)
    result = sound_power_reverberation(
        lp, t60, volume=200.0, surface_area=220.0, frequencies=freqs,
        temperature=20.0, static_pressure=101.0,
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Reverberation-room sound power (ISO 3741)  LWA = {lwa:.1f} dB(A)",
        fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_reverberation_result.png")
    plt.close()


def generate_sound_power_intensity_result(output_dir: str) -> None:
    """ISO 9614-2: intensity-scanning LW spectrum from segment sweeps."""
    print("Generating sound_power_intensity_result.png...")
    from phonometry.emission import sound_power_intensity

    # The sound-power guide's section-3 example: two repeated intensity sweeps
    # over 6 surface segments and 6 octave bands, with the segment surface SPL
    # and the probe's pressure-residual intensity index. The partial powers
    # In_i * Si sum to the band LW; every band passes the field-indicator
    # criteria at engineering grade here (no SoundPowerWarning fires).
    freqs = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    areas = np.full(6, 0.5)
    rng = np.random.default_rng(0)
    scan1 = np.abs(rng.normal(1e-4, 2e-5, size=(6, 6)))
    scan2 = scan1 * (1.0 + rng.normal(0.0, 0.02, size=(6, 6)))
    pressure = np.full((6, 6), 80.0)
    result = sound_power_intensity(
        scan1, areas, normal_intensity_2=scan2, pressure_levels=pressure,
        pressure_residual_index=12.0, frequencies=freqs,
        band_type="octave", grade="engineering",
    )

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Plot only the determinable (finite-LW) bands; an undeterminable band
    # (net inflow -> NaN) is left as a gap rather than faked to 0 dB. All six
    # bands are finite with this synthetic data, so this is future-proofing.
    finite = np.isfinite(lw)
    ax.bar(positions[finite], lw[finite], width=0.7, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Intensity-scanning sound power (ISO 9614-2)  LWA = {lwa:.1f} dB(A)",
        fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "sound_power_intensity_result.png")
    plt.close()


def generate_precision_anechoic_power(output_dir: str) -> None:
    """ISO 3745: precision LW spectrum from a hemisphere pressure measurement."""
    print("Generating precision_anechoic_power.png...")
    from phonometry import sound_power_anechoic

    # A mid-frequency-peaked machine measured over the 40-position hemisphere
    # array (ISO 3745 Annex E) in a hemi-anechoic room. levels_positions is the
    # (40, NB) surface pressure spectrum: a base machine spectrum peaked near
    # 1 kHz plus a small per-position spatial variation. The library forms the
    # surface-averaged LW = Lp_bar + 10 lg(S/S0) + C1+C2+C3 and the A-weighted
    # total LWA.
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=float)
    base = 70.0 + 8.0 * np.exp(-(np.log2(freqs / 1000.0) ** 2) / 2.0)
    rng = np.random.default_rng(7)
    levels = base[None, :] + rng.normal(0.0, 1.0, (40, freqs.size))
    result = sound_power_anechoic(levels, "hemisphere", radius=1.0,
                                  frequencies=freqs)

    lw = result.sound_power_level
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, lw, width=0.7, color=COLOR_PRIMARY, edgecolor=COLOR_FG,
           linewidth=0.7, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Precision sound power (ISO 3745)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "precision_anechoic_power.png")
    plt.close()


def generate_intensity_scan_power(output_dir: str) -> None:
    """ISO 9614-3: precision LW spectrum by intensity scanning (with a NaN band)."""
    print("Generating intensity_scan_power.png...")
    import warnings

    from phonometry import sound_power_intensity_precision

    # Four partial surfaces scanned over five one-third-octave bands. Each cell
    # of partial_intensity is the signed normal intensity In_i (W/m^2) already
    # reduced to the two-scan result; areas are the partial-surface areas Si.
    # The 250 Hz band has net-negative power (more energy flowing in than out),
    # so ISO 9614-3 flags it not-applicable (clause 9.2) and it is hatched.
    freqs = np.array([250, 500, 1000, 2000, 4000], dtype=float)
    areas = np.array([0.5, 1.0, 0.75, 0.5])
    base_intensity = np.array([2.0e-6, 8.0e-6, 2.0e-5, 1.0e-5, 3.0e-6])
    per_segment = np.array([1.0, 1.1, 0.9, 1.05])
    partial_intensity = base_intensity[None, :] * per_segment[:, None]
    # A locally reactive 250 Hz band: the segment intensities cancel to a
    # net-negative total.
    partial_intensity[:, 0] = np.array([2.0e-6, -3.0e-6, -4.0e-6, -1.0e-6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sound_power_intensity_precision(partial_intensity, areas,
                                                 frequencies=freqs)

    lw = result.sound_power_level
    neg = result.not_applicable_band
    lwa = result.sound_power_level_a
    positions = np.arange(freqs.size, dtype=float)
    _fig, ax = plt.subplots(figsize=(10, 6.3))
    # Determinate bands: a solid LW bar. Non-applicable bands carry no LW (NaN),
    # so instead of a zero-height bar they are flagged by a full-height greyed,
    # hatched span - clearly a marker, not a plotted value (ISO 9614-3, 9.2).
    ax.bar(positions[~neg], np.nan_to_num(lw)[~neg], width=0.7, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.7, zorder=3)
    for pos, is_neg in zip(positions, neg):
        if is_neg:
            ax.axvspan(pos - 0.35, pos + 0.35, facecolor="#888888", alpha=0.28,
                       hatch="//", edgecolor="#888888", linewidth=0.8, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(f"Precision intensity scanning (ISO 9614-3)  LWA = {lwa:.1f} dB(A)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Sound power level LW [dB]")
    ax.set_ylim(0.0, float(np.nanmax(lw)) + 8.0)
    from matplotlib.patches import Patch
    handle = Patch(facecolor="#888888", alpha=0.28, hatch="//",
                   edgecolor="#888888", label="Non-applicable band")
    ax.legend(handles=[handle], loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "intensity_scan_power.png")
    plt.close()


def generate_vibration_weighting(output_dir: str) -> None:
    """ISO 8041-1: the whole-body vertical weighting Wk over its band."""
    print("Generating vibration_weighting.png...")
    from phonometry import frequency_weighting

    # A user evaluates the principal ISO 2631-1 weighting Wk on a fine
    # frequency grid across the whole-body band (0,4-100 Hz). The result is the
    # ISO 8041-1 cascade H(f): a gentle +0,5 dB peak near 6 Hz, a band-limiting
    # roll-off below 0,4 Hz and above ~16 Hz.
    freqs = np.geomspace(0.4, 100.0, 240)
    result = frequency_weighting("Wk", freqs)

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.semilogx(result.frequencies, result.magnitude_db, color=COLOR_PRIMARY,
                linewidth=1.9, zorder=3)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_title("Whole-body vertical weighting Wk (ISO 8041-1)",
                 fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Weighting factor [dB]")
    ax.set_xlim(0.4, 100.0)
    ax.set_ylim(-40.0, 5.0)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100])
    # Explicit string labels install a FixedFormatter so the Spanish pass can
    # apply the decimal comma (a log-axis ScalarFormatter would not be caught).
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "vibration_weighting.png")
    plt.close()


def generate_weighted_acceleration(output_dir: str) -> None:
    """ISO 2631-1: measured seat spectrum weighted to a_w (Eq. (9))."""
    print("Generating weighted_acceleration.png...")
    from phonometry import weighted_acceleration

    # A measured vertical seat-pan acceleration spectrum (r.m.s. per one-third
    # octave, m/s^2) from a vehicle seat: energy concentrated in the 2-8 Hz
    # whole-body range. Weighting it with Wk gives the health-relevant a_w.
    freqs = np.array([1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0,
                      10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 63.0, 80.0])
    accel = np.array([0.18, 0.24, 0.33, 0.46, 0.52, 0.55, 0.48, 0.39, 0.31,
                      0.26, 0.21, 0.17, 0.13, 0.10, 0.078, 0.060, 0.045,
                      0.028, 0.020])
    result = weighted_acceleration(accel, freqs, "Wk")

    positions = np.arange(freqs.size, dtype=float)
    width = 0.4
    _fig, ax = plt.subplots(figsize=(10.5, 6.3))
    ax.bar(positions - width / 2, result.band_accelerations, width,
           color="#9e9e9e", edgecolor=COLOR_FG, linewidth=0.5,
           label="Unweighted $a_i$", zorder=2)
    ax.bar(positions + width / 2, result.weighted, width, color=COLOR_PRIMARY,
           edgecolor=COLOR_FG, linewidth=0.5, label="Weighted $W_i\\,a_i$ (Wk)",
           zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.set_title(
        f"Weighted seat acceleration (ISO 2631-1)  $a_w$ = {result.overall:.3f} "
        "m/s$^2$", fontweight="bold", pad=12)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("r.m.s. acceleration [m/s$^2$]")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "weighted_acceleration.png")
    plt.close()


def generate_daily_vibration_exposure(output_dir: str) -> None:
    """ISO 5349 + Directive 2002/44/EC: A(8) vs the EAV/ELV thresholds."""
    print("Generating daily_vibration_exposure.png...")
    from phonometry import daily_vibration_exposure

    # A forestry worker's day across three chain-saw tasks (the ISO 5349-2
    # Annex E.3 worked example): each task's a_hv and duration give a partial
    # exposure A_i(8); they combine to A(8) = 3,6 m/s^2, assessed against the
    # hand-arm action (2,5) and limit (5,0) values of Directive 2002/44/EC.
    result = daily_vibration_exposure(
        [4.6, 6.0, 3.6],
        [2 * 3600.0, 1 * 3600.0, 2 * 3600.0],
        kind="hav",
        labels=["brush-saw", "felling", "stripping"],
    )

    labels = [*result.labels, "A(8)"]
    values = [*result.partials.tolist(), result.a8]
    positions = np.arange(len(values), dtype=float)
    colors = ["#9e9e9e"] * result.partials.size + [COLOR_PRIMARY]
    _fig, ax = plt.subplots(figsize=(9.5, 6.3))
    ax.bar(positions, values, width=0.62, color=colors, edgecolor=COLOR_FG,
           linewidth=0.6, zorder=3)
    eav = result.assessment.action_value
    elv = result.assessment.limit_value
    ax.axhline(eav, color=COLOR_TERTIARY, linestyle="--", linewidth=1.6,
               label=f"EAV = {eav:g} m/s$^2$", zorder=2)
    ax.axhline(elv, color=COLOR_SECONDARY, linestyle="--", linewidth=1.6,
               label=f"ELV = {elv:g} m/s$^2$", zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Daily exposure A(8) [m/s$^2$]")
    ax.set_ylim(0.0, elv * 1.2)
    ax.set_title(
        f"Hand-arm daily exposure (ISO 5349 / 2002-44-EC)  A(8) = "
        f"{result.a8:.2f} m/s$^2$", fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    save_figure(output_dir, "daily_vibration_exposure.png")
    plt.close()


def generate_speech_intelligibility(output_dir: str) -> None:
    """ANSI S3.5-1997: band audibility and the SII in broadband noise."""
    print("Generating speech_intelligibility.png...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum

    # Standard normal-effort speech in a descending broadband masking noise
    # (an office/ventilation-like spectrum): the band-audibility function A_i
    # is partial across the band, and the importance-weighted contribution
    # I_i*A_i (ANSI S3.5-1997 clause 6) sums to the index SII.
    speech = standard_speech_spectrum("normal")
    noise = np.array([38.0, 37.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0,
                      22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0])
    result = speech_intelligibility_index(speech, noise)

    freqs = result.frequencies
    positions = np.arange(freqs.size)
    weighted = result.band_audibility * result.band_importance

    _fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.bar(positions, result.band_audibility, width=0.8, color=COLOR_PRIMARY,
           alpha=0.35, zorder=2, label=r"Band audibility $A_i$")
    ax.bar(positions, weighted / weighted.max(), width=0.45, color=COLOR_PRIMARY,
           zorder=3, label=r"Importance-weighted $I_i\,A_i$ (scaled)")
    ax.set_title(
        f"Speech Intelligibility Index (ANSI S3.5-1997)   SII = {result.sii:.2f}",
        fontweight="bold", pad=12)
    ax.set_xlabel("One-third-octave band [Hz]")
    ax.set_ylabel("Band audibility")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax.grid(which="major", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    save_figure(output_dir, "speech_intelligibility.png")
    plt.close()


def generate_sii_vocal_efforts(output_dir: str) -> None:
    """ANSI S3.5-1997 Table 3 standard speech spectra by vocal effort."""
    print("Generating sii_vocal_efforts.png...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum
    from phonometry.hearing.sii import BAND_CENTERS, VOCAL_EFFORTS

    freqs = BAND_CENTERS
    # Distinct hues (not COLOR_GRID, which blends into the gridlines and is
    # near-invisible on a light background) for the four ordered efforts.
    colours = {"normal": COLOR_TERTIARY, "raised": "#7f7f7f",
               "loud": COLOR_PRIMARY, "shout": COLOR_SECONDARY}
    _fig, (ax_s, ax_i) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: the four standard speech spectra. ---
    for effort in VOCAL_EFFORTS:
        ax_s.plot(freqs, standard_speech_spectrum(effort), "o-",
                  color=colours[effort], label=effort.capitalize())
    ax_s.set_xscale("log")
    ax_s.set_xticks(list(freqs))
    ax_s.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_s.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_s.set_xlabel("One-third-octave band [Hz]")
    ax_s.set_ylabel("Speech spectrum level [dB SPL]")
    ax_s.set_title("ANSI S3.5-1997 — speech spectra by vocal effort",
                   fontweight="bold", pad=10)
    ax_s.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_s.set_axisbelow(True)
    ax_s.legend(loc="upper right")

    # --- Right: SII in a fixed broadband noise rises with vocal effort. ---
    noise = np.array([48.0, 47.0, 46.0, 44.0, 42.0, 40.0, 38.0, 36.0, 34.0,
                      32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0])
    indices = [speech_intelligibility_index(e, noise).sii for e in VOCAL_EFFORTS]
    positions = np.arange(len(VOCAL_EFFORTS))
    bar_colours = [colours[e] for e in VOCAL_EFFORTS]
    ax_i.bar(positions, indices, width=0.6, color=bar_colours, zorder=2)
    for x, v in zip(positions, indices):
        ax_i.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                  fontweight="bold")
    ax_i.set_xticks(positions)
    ax_i.set_xticklabels([e.capitalize() for e in VOCAL_EFFORTS])
    ax_i.set_ylim(0.0, 1.0)
    ax_i.set_ylabel("Speech Intelligibility Index")
    ax_i.set_title("SII vs vocal effort in a fixed noise",
                   fontweight="bold", pad=10)
    ax_i.grid(which="major", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_i.set_axisbelow(True)

    plt.tight_layout()
    save_figure(output_dir, "sii_vocal_efforts.png")
    plt.close()


def generate_standard_speech_spectrum(output_dir: str) -> None:
    """ANSI S3.5-1997 Table 3 standard speech spectra via StandardSpeechSpectrum.plot()."""
    print("Generating standard_speech_spectrum...")
    from phonometry import standard_speech_spectra

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws one line per vocal effort on the categorical
    # one-third-octave band axis (160 Hz to 8000 Hz), each higher effort lifting
    # the whole speech spectrum (ANSI S3.5-1997 Table 3).
    standard_speech_spectra().plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "standard_speech_spectrum.svg")
    plt.close()


def generate_sii_band_procedures(output_dir: str) -> None:
    """The band-importance functions of the four ANSI S3.5-1997 procedures."""
    print("Generating sii_band_procedures...")
    from phonometry import SII_METHODS, sii_procedure

    _, ax = plt.subplots(figsize=(10, 6))
    # Each procedure's own .plot() steps Ii across its tabulated band limits on
    # a shared logarithmic frequency axis, so the 6-band octave function and
    # the 21-band critical-band function are directly comparable: the same
    # importance is spread over very different bandwidths (Tables 1 to 4).
    for method in SII_METHODS:
        sii_procedure(method).plot(ax=ax, language=_LANG, linewidth=1.8)
    plt.tight_layout()
    save_figure(output_dir, "sii_band_procedures.svg")
    plt.close()


def generate_impulse_prominence(output_dir: str) -> None:
    """NT ACOU 112: predicted prominence and the LAeq adjustment."""
    print("Generating impulse_prominence.png...")
    from phonometry import (
        impulse_adjustment,
        impulse_prominence,
        predicted_prominence,
    )
    from phonometry.environmental.impulse_prominence import ADJUSTMENT_THRESHOLD

    _fig, (ax_p, ax_k) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: P vs onset rate for three level differences (Formula 1). ---
    orate = np.logspace(1, 4, 200)  # 10 to 10000 dB/s
    # Distinct hues (not COLOR_GRID, which is near-invisible on a light ground).
    for ld, colour in ((5.0, COLOR_TERTIARY), (15.0, COLOR_PRIMARY),
                       (30.0, COLOR_SECONDARY)):
        ax_p.plot(orate, predicted_prominence(orate, np.full_like(orate, ld)),
                  color=colour, label=f"LD = {ld:g} dB")
    ax_p.set_xscale("log")
    ax_p.set_xlabel("Onset rate [dB/s]")
    ax_p.set_ylabel("Predicted prominence $P$")
    ax_p.set_title(r"$P = 3\,\lg(\mathrm{OR}) + 2\,\lg(\mathrm{LD})$",
                   fontweight="bold", pad=10)
    ax_p.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_p.set_axisbelow(True)
    ax_p.legend(loc="upper left")

    # --- Right: the adjustment KI(P) with example impulses. ---
    result = impulse_prominence([1200.0, 300.0, 60.0], [32.0, 18.0, 11.0])
    grid = np.linspace(0.0, 16.0, 200)
    ax_k.plot(grid, impulse_adjustment(grid), color=COLOR_PRIMARY,
              label=r"$K_I = 1.8\,(P-5)$")
    ax_k.axvline(ADJUSTMENT_THRESHOLD, color="#7f7f7f", linestyle=":",
                 label=f"threshold $P = {ADJUSTMENT_THRESHOLD:g}$")
    ax_k.scatter(result.per_impulse, impulse_adjustment(result.per_impulse),
                 color="#aec7e8", zorder=3, label="Impulses")
    ax_k.scatter([result.prominence], [result.adjustment], color=COLOR_SECONDARY,
                 marker="*", s=140, zorder=4,
                 label=f"Governing  $K_I$ = {result.adjustment:.1f} dB")
    ax_k.set_xlabel("Predicted prominence $P$")
    ax_k.set_ylabel("Adjustment $K_I$ [dB]")
    ax_k.set_title("Adjustment to $L_{Aeq}$", fontweight="bold", pad=10)
    ax_k.set_ylim(bottom=0.0)
    ax_k.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_k.set_axisbelow(True)
    ax_k.legend(loc="upper left")

    plt.tight_layout()
    save_figure(output_dir, "impulse_prominence.png")
    plt.close()


def generate_tonal_audibility(output_dir: str) -> None:
    """ISO 1996-2: tonal adjustment Kt(ΔLta) with the Annex C.5 examples."""
    print("Generating tonal_audibility...")
    from phonometry import assess_tonal_audibility, tonal_adjustment

    # The four ISO 1996-2:2007 Annex C.5 worked examples: (Lpt, Lpn, fc).
    examples = [(46.7, 37.3, 4000.0), (54.1, 45.2, 430.0),
                (53.6, 45.5, 755.0), (54.6, 45.5, 308.0)]
    assessed = [assess_tonal_audibility(lpt, lpn, fc) for lpt, lpn, fc in examples]
    # A synthetic mid-range tone to exercise the sloped branch.
    mid = assess_tonal_audibility(50.0, 44.0, 500.0)

    grid = np.linspace(0.0, 15.0, 300)
    curve = np.array([tonal_adjustment(d) for d in grid])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(grid, curve, "-", color=COLOR_PRIMARY, linewidth=2.4, zorder=5,
            label=r"$K_t(\Delta L_{ta})$ (Formulae C.4-C.6)")
    for x in (4.0, 10.0):
        ax.axvline(x, color=COLOR_GRID, linestyle=":", alpha=0.8, zorder=1)
    ax.scatter([a.audibility for a in assessed], [a.adjustment for a in assessed],
               color=COLOR_SECONDARY, marker="o", s=70, zorder=6,
               label="Annex C.5 examples")
    ax.scatter([mid.audibility], [mid.adjustment], color=COLOR_TERTIARY,
               marker="*", s=150, zorder=7, label="mid-range tone")

    ax.set_xlabel(r"Tonal audibility $\Delta L_{ta}$ [dB]")
    ax.set_ylabel("Tonal adjustment $K_t$ [dB]")
    ax.set_ylim(-0.3, 6.6)
    ax.set_title("ISO 1996-2 Tonal Adjustment", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    info = [
        "Kt = 0            (dLta < 4)",
        "Kt = dLta - 4  (4 <= dLta <= 10)",
        "Kt = 6            (dLta > 10)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG, family="monospace",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "tonal_audibility.png")
    plt.close()


def generate_multiple_shock(output_dir: str) -> None:
    """ISO 2631-5: seat-to-spine transmissibility and the injury probability."""
    print("Generating multiple_shock.png...")
    from phonometry import (
        compression_dose,
        dose_from_peaks,
        injury_probability,
        injury_risk,
        seat_to_spine_transfer,
    )
    from phonometry.vibration.multiple_shock_vibration import (
        MZ_MALE,
        RISK_THRESHOLDS_MALE,
    )

    _fig, (ax_h, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: seat-to-spine transmissibility |H(f)| (Formula 1). ---
    freq = np.logspace(np.log10(0.5), np.log10(80.0), 400)
    ax_h.plot(freq, np.abs(seat_to_spine_transfer(freq)), color=COLOR_PRIMARY,
              label=r"$|H(f)|$")
    ax_h.axhline(1.0, color=COLOR_GRID, linestyle="--", alpha=0.7)
    ax_h.set_xscale("log")
    ax_h.set_xlabel("Frequency [Hz]")
    ax_h.set_ylabel("Transmissibility  seat $\\rightarrow$ spine")
    ax_h.set_title("Seat-to-spine transfer function", fontweight="bold", pad=10)
    ax_h.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_h.set_axisbelow(True)
    format_frequency_axis(ax_h, float(freq[0]), float(freq[-1]))
    ax_h.legend(loc="upper right")

    # --- Right: injury probability Pi(R) with the Annex C male example. ---
    grid = np.linspace(0.0, 3.0, 300)
    sexes: tuple[tuple[Literal["male", "female"], str], ...] = (
        ("male", COLOR_PRIMARY),
        ("female", COLOR_SECONDARY),
    )
    for sex, colour in sexes:
        prob = 100.0 * injury_probability(grid, sex=sex)
        ax_r.plot(grid, prob, color=colour, label=f"{sex}")
    # The worked example: five 40 m/s2 peaks, 82 kg male -> R = 1.22.
    sd = compression_dose(dose_from_peaks([40.0] * 5), mz=MZ_MALE)
    r_male = injury_risk(sd, start_age=20, years=20, days_per_year=120, sex="male")
    for level, r_val in zip((10, 50, 90), RISK_THRESHOLDS_MALE):
        ax_r.axhline(level, color="#7f7f7f", linestyle=":", lw=0.8)
        ax_r.plot([r_val, r_val], [0.0, level], color="#7f7f7f", linestyle=":", lw=0.8)
    ax_r.scatter([r_male], [100.0 * injury_probability(r_male, sex="male")],
                 color=COLOR_TERTIARY, marker="*", s=160, zorder=4,
                 label=f"Example  $R$ = {r_male:.2f}")
    ax_r.set_xlabel("Stress variable $R$")
    ax_r.set_ylabel("Probability of lumbar injury [%]")
    ax_r.set_title("Injury probability (Annex C)", fontweight="bold", pad=10)
    ax_r.set_xlim(left=0.0)
    ax_r.set_ylim(0.0, 100.0)
    ax_r.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="lower right")

    plt.tight_layout()
    save_figure(output_dir, "multiple_shock.png")
    plt.close()


def generate_enclosed_space_absorption(output_dir: str) -> None:
    """EN 12354-6: absorption area and reverberation time of a room."""
    print("Generating enclosed_space_absorption.png...")
    from phonometry import enclosed_space_reverberation
    from phonometry.room.enclosed_space_absorption import OCTAVE_BANDS

    # A 5 x 4 x 3 m office (60 m3): hard plaster walls and floor; the ceiling is
    # either bare plaster or lined with an absorbing acoustic tile.
    volume = 60.0
    plaster = [0.02, 0.03, 0.03, 0.04, 0.05, 0.05, 0.05]
    tile = [0.15, 0.35, 0.65, 0.85, 0.90, 0.90, 0.85]
    walls_floor = [(54.0, plaster), (20.0, plaster)]  # walls + floor
    bare = enclosed_space_reverberation(
        [*walls_floor, (20.0, plaster)], volume, air_condition="20C_50-70")
    treated = enclosed_space_reverberation(
        [*walls_floor, (20.0, tile)], volume, air_condition="20C_50-70")

    _fig, (ax_a, ax_t) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    freq = OCTAVE_BANDS
    labels = [f"{f:g}" if f < 1000 else f"{f / 1000:g}k" for f in freq]

    for res, colour, name in ((bare, COLOR_SECONDARY, "bare ceiling"),
                              (treated, COLOR_PRIMARY, "acoustic ceiling")):
        ax_a.semilogx(freq, res.absorption_area, color=colour, marker="o", label=name)
        ax_t.semilogx(freq, res.reverberation_time, color=colour, marker="o",
                      label=name)
    for ax, ylab, title in (
        (ax_a, "Equivalent absorption area $A$ [m$^2$]", "Absorption area (Formula 1)"),
        (ax_t, "Reverberation time $T$ [s]", "Reverberation time (Formula 5)"),
    ):
        ax.set_xticks(freq)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Octave-band centre frequency [Hz]")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylim(bottom=0.0)
        ax.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "enclosed_space_absorption.png")
    plt.close()


def generate_room_noise_criteria(output_dir: str) -> None:
    """ANSI S12.2-2019: NC tangency rating and RC Mark II classification."""
    print("Generating room_noise_criteria.png...")
    from phonometry import noise_criterion, room_criterion
    from phonometry.room.room_noise import NC_CURVES, NC_INDICES, OCTAVE_BANDS

    # A ventilation-dominated room spectrum: the low-frequency bands rise well
    # above the sloped RC reference (a rumble tag under RC Mark II) while the
    # mid bands set the NC tangency.
    spectrum = np.array([62.0, 62.0, 59.0, 57.0, 52.0, 42.0, 35.0, 29.0, 24.0, 19.0])
    nc = noise_criterion(spectrum)
    rc = room_criterion(spectrum)

    _fig, (ax_nc, ax_rc) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: NC curves + tangency rating. ---
    for row, idx in zip(NC_CURVES, NC_INDICES):
        ax_nc.plot(OCTAVE_BANDS, row, color=COLOR_GRID, lw=0.8, zorder=1)
        ax_nc.annotate(f"{idx:.0f}", (OCTAVE_BANDS[-1], row[-1]),
                       fontsize=7, color="#999999", va="center")
    ax_nc.plot(OCTAVE_BANDS, spectrum, "o-", color=COLOR_PRIMARY, zorder=3,
               label="Measured")
    gov = spectrum[OCTAVE_BANDS == nc.governing_frequency][0]
    ax_nc.plot([nc.governing_frequency], [gov], "D", color=COLOR_SECONDARY,
               ms=9, zorder=4, label=f"Tangent @ {nc.governing_frequency:g} Hz")
    ax_nc.set_xscale("log")
    ax_nc.set_xticks(list(OCTAVE_BANDS))
    ax_nc.set_xticklabels([f"{f:g}" for f in OCTAVE_BANDS], rotation=45, ha="right")
    ax_nc.set_xlabel("Octave-band center frequency [Hz]")
    ax_nc.set_ylabel("Octave-band sound pressure level [dB]")
    ax_nc.set_title(f"Noise Criteria — tangency method   NC-{nc.rating:g}",
                    fontweight="bold", pad=10)
    ax_nc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_nc.set_axisbelow(True)
    ax_nc.legend(loc="upper right")

    # --- Right: RC Mark II reference + rumble/hiss tolerances. ---
    ref = rc.reference_curve
    low = OCTAVE_BANDS <= 500.0
    high = OCTAVE_BANDS >= 1000.0
    ax_rc.plot(OCTAVE_BANDS, ref, "s--", color="#7f7f7f",
               label=f"Reference RC-{rc.rating}")
    ax_rc.fill_between(OCTAVE_BANDS[low], ref[low], ref[low] + 5.0, zorder=0,
                       color=theme_fill("#ff7f0e", ax_rc),
                       label="Rumble tol. (+5 dB)")
    ax_rc.fill_between(OCTAVE_BANDS[high], ref[high], ref[high] + 3.0, zorder=0,
                       color=theme_fill(COLOR_PRIMARY, ax_rc),
                       label="Hiss tol. (+3 dB)")
    ax_rc.plot(OCTAVE_BANDS, spectrum, "o-", color=COLOR_PRIMARY, zorder=3,
               label="Measured")
    ax_rc.set_xscale("log")
    ax_rc.set_xticks(list(OCTAVE_BANDS))
    ax_rc.set_xticklabels([f"{f:g}" for f in OCTAVE_BANDS], rotation=45, ha="right")
    ax_rc.set_xlabel("Octave-band center frequency [Hz]")
    ax_rc.set_ylabel("Octave-band sound pressure level [dB]")
    ax_rc.set_title(f"Room Criteria Mark II   {rc.label}",
                    fontweight="bold", pad=10)
    ax_rc.grid(which="both", axis="y", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_rc.set_axisbelow(True)
    ax_rc.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "room_noise_criteria.png")
    plt.close()


def generate_hearing_threshold(output_dir: str) -> None:
    """ISO 7029 age-related threshold and ISO 389-7 reference threshold."""
    print("Generating hearing_threshold.png...")
    from phonometry import age_threshold, reference_threshold
    from phonometry.hearing.threshold import AUDIOMETRIC_FREQUENCIES

    freqs = AUDIOMETRIC_FREQUENCIES
    _fig, (ax_age, ax_ref) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: ISO 7029 median threshold by age (male) + 10-90 % band @70. ---
    ages = [(20, "#9e9e9e"), (40, "#7f7f7f"), (60, COLOR_PRIMARY),
            (80, COLOR_SECONDARY)]
    for age, color in ages:
        r = age_threshold(age, "male", 0.5)
        ax_age.plot(freqs, r.median, "o-", color=color, label=f"{age} yr")
    r70 = age_threshold(70, "male", 0.5)
    z90 = 1.2816
    ax_age.fill_between(freqs, r70.median - z90 * r70.spread_lower,
                        r70.median + z90 * r70.spread_upper,
                        color=theme_fill(COLOR_PRIMARY, ax_age), zorder=0,
                        label="10-90 % band (70 yr)")
    ax_age.set_xscale("log")
    ax_age.set_xticks(list(freqs))
    ax_age.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_age.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_age.invert_yaxis()
    ax_age.set_xlabel("Audiometric frequency [Hz]")
    ax_age.set_ylabel("Median threshold deviation from age 18 [dB]")
    ax_age.set_title("ISO 7029 — age-related threshold (male)",
                     fontweight="bold", pad=10)
    ax_age.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_age.set_axisbelow(True)
    ax_age.legend(loc="lower left")

    # --- Right: ISO 389-7 reference threshold, free vs diffuse field. ---
    ax_ref.plot(freqs, reference_threshold("free-field"), "o-",
                color=COLOR_PRIMARY, label="Free-field (frontal)")
    ax_ref.plot(freqs, reference_threshold("diffuse-field"), "s--",
                color=COLOR_SECONDARY, label="Diffuse-field")
    ax_ref.set_xscale("log")
    ax_ref.set_xticks(list(freqs))
    ax_ref.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_ref.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_ref.set_xlabel("Audiometric frequency [Hz]")
    ax_ref.set_ylabel("Reference threshold [dB]")
    ax_ref.set_title("ISO 389-7 — reference threshold of hearing",
                     fontweight="bold", pad=10)
    ax_ref.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_ref.set_axisbelow(True)
    ax_ref.legend(loc="upper left")

    plt.tight_layout()
    save_figure(output_dir, "hearing_threshold.png")
    plt.close()


def generate_noise_induced_hearing_loss(output_dir: str) -> None:
    """ISO 1999 noise-induced permanent threshold shift and HTLAN combination."""
    print("Generating noise_induced_hearing_loss.png...")
    from phonometry import htlan, nipts
    from phonometry.hearing.noise_induced_hearing_loss import NIPTS_FREQUENCIES

    freqs = NIPTS_FREQUENCIES
    _fig, (ax_n, ax_h) = plt.subplots(1, 2, figsize=(12.5, 5.6))

    # --- Left: median NIPTS growth with exposure duration at 95 dB. ---
    durations = [(10, "#9e9e9e"), (20, "#7f7f7f"), (30, COLOR_PRIMARY),
                 (40, COLOR_SECONDARY)]
    for years, color in durations:
        r = nipts(95.0, years, 0.5)
        ax_n.plot(freqs, r.median, "o-", color=color, label=f"{years} yr")
    r40 = nipts(95.0, 40.0, 0.5)
    z90 = 1.2816
    ax_n.fill_between(freqs, np.maximum(r40.median - z90 * r40.spread_lower, 0.0),
                      r40.median + z90 * r40.spread_upper,
                      color=theme_fill(COLOR_SECONDARY, ax_n), zorder=0,
                      label="10-90 % band (40 yr)")
    ax_n.set_xscale("log")
    ax_n.set_xticks(list(freqs))
    ax_n.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_n.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_n.invert_yaxis()
    ax_n.set_xlabel("Audiometric frequency [Hz]")
    ax_n.set_ylabel("Median NIPTS [dB]")
    ax_n.set_title(r"ISO 1999 — NIPTS at $L_{EX,8h}$ = 95 dB",
                   fontweight="bold", pad=10)
    ax_n.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_n.set_axisbelow(True)
    ax_n.legend(loc="lower left")

    # --- Right: HTLAN = age + noise for a 60-year-old worker, 95 dB / 30 yr. ---
    h = htlan(60, "male", 95.0, 30.0, 0.5)
    ax_h.plot(freqs, h.htla, "o-", color=COLOR_PRIMARY,
              label="Age (HTLA, ISO 7029)")
    ax_h.plot(freqs, h.nipts, "^-", color="#ff7f0e", label="Noise (NIPTS)")
    ax_h.plot(freqs, h.threshold, "s--", color=COLOR_SECONDARY,
              label="Age + noise (HTLAN)")
    ax_h.set_xscale("log")
    ax_h.set_xticks(list(freqs))
    ax_h.set_xticklabels([f"{f:g}" for f in freqs], rotation=45, ha="right")
    ax_h.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_h.invert_yaxis()
    ax_h.set_xlabel("Audiometric frequency [Hz]")
    ax_h.set_ylabel("Hearing threshold level [dB]")
    ax_h.set_title("ISO 1999 — HTLAN (male, age 60, 95 dB / 30 yr)",
                   fontweight="bold", pad=10)
    ax_h.grid(which="both", color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_h.set_axisbelow(True)
    ax_h.legend(loc="lower left")

    plt.tight_layout()
    save_figure(output_dir, "noise_induced_hearing_loss.png")
    plt.close()


def generate_uncertainty(output_dir: str) -> None:
    """GUM uncertainty budget and Monte Carlo distribution (Guide 98-3 + S1)."""
    print("Generating uncertainty_budget.png...")
    import phonometry as ph

    # A-weighted level: reading plus calibration, instrument and positional
    # corrections (all zero-mean); the model is their sum.
    quantities = [
        ph.Quantity(74.0, 0.0, name="Reading"),
        ph.rectangular(0.0, 0.20, name="Calibration"),
        ph.rectangular(0.0, 0.30, name="Instrument"),
        ph.Quantity(0.0, 0.35, dof=9, name="Position (Type A)"),
    ]
    model = lambda a, b, c, d: a + b + c + d

    result = ph.combine_uncertainty(model, quantities)
    mc = ph.monte_carlo(model, quantities, trials=1_000_000, coverage=0.95, seed=1)
    k, big = result.expanded(0.95)

    _fig, (ax_b, ax_m) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # --- Left: uncertainty budget (contributions). ---
    contrib = result.contributions
    names = list(result.names)
    pos = np.arange(len(names))
    ax_b.barh(pos, contrib, color=COLOR_PRIMARY, zorder=2)
    ax_b.axvline(result.combined_uncertainty, color=COLOR_SECONDARY, ls="--",
                 label=f"$u_c$ = {result.combined_uncertainty:.3f} dB")
    ax_b.set_yticks(pos)
    ax_b.set_yticklabels(names)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Contribution to combined uncertainty [dB]")
    ax_b.set_title("GUM uncertainty budget", fontweight="bold", pad=10)
    ax_b.grid(which="major", axis="x", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right")

    # --- Right: Monte Carlo output vs the GUM Gaussian. ---
    rng = np.random.default_rng(1)
    samples = (74.0 + rng.uniform(-0.20, 0.20, 200000)
               + rng.uniform(-0.30, 0.30, 200000)
               + rng.normal(0.0, 0.35, 200000))
    ax_m.hist(samples, bins=120, density=True, color=COLOR_PRIMARY, alpha=0.35,
              label="Monte Carlo (Suppl 1)")
    grid = np.linspace(samples.min(), samples.max(), 400)
    gauss = (np.exp(-0.5 * ((grid - result.value) / result.combined_uncertainty) ** 2)
             / (result.combined_uncertainty * np.sqrt(2 * np.pi)))
    ax_m.plot(grid, gauss, color=COLOR_SECONDARY, lw=2, label="GUM Gaussian")
    ax_m.axvspan(mc.interval[0], mc.interval[1], zorder=0,
                 color=theme_fill(COLOR_PRIMARY, ax_m),
                 label="95 % coverage interval")
    ax_m.set_xlabel("A-weighted level [dB]")
    ax_m.set_ylabel("Probability density")
    ax_m.set_title(f"Y = {result.value:.2f} dB,  U = {big:.2f} dB (k = {k:.2f})",
                   fontweight="bold", pad=10)
    ax_m.grid(color=COLOR_GRID, linestyle="-", alpha=0.4)
    ax_m.set_axisbelow(True)
    ax_m.legend(loc="upper right")

    plt.tight_layout()
    save_figure(output_dir, "uncertainty_budget.png")
    plt.close()


def generate_fdtd_simulation(output_dir: str) -> None:
    """2D FDTD simulation: diffraction snapshot and the probe histories."""
    print("Generating fdtd_simulation.png...")
    from phonometry import GaussianPulse, fdtd_simulation

    # A 3.0 x 2.0 m free field (absorbing edges) with a thin rigid barrier:
    # probe A sees the direct pulse plus the barrier reflection, probe B sits
    # in the shadow and only receives the wave diffracted around the edge.
    dx = 0.01
    mask = np.zeros((200, 300), dtype=bool)
    mask[60:, 150:154] = True
    res = fdtd_simulation(
        343.0, dx, 9.0e-3, shape=(200, 300),
        sources=[GaussianPulse(ix=60, iy=100, width=3.0e-4)],
        probes=[(100, 100), (240, 100)],
        obstacle_mask=mask,
        boundaries="absorbing", absorbing_layer_cells=30,
        snapshot_every=75,
    )

    _fig, (ax_f, ax_p) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})
    res.plot(kind="snapshot", frame=7, ax=ax_f)
    res.plot(ax=ax_p)
    ax_p.set_title("FDTD probe pressure", fontweight="bold", pad=10)
    ax_f.set_title(ax_f.get_title(), fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "fdtd_simulation.png")
    plt.close()


def generate_elastic_halfspace_waves(output_dir: str) -> None:
    """Elastic FDTD: P, S and Rayleigh waves in an aluminium half-space."""
    print("Generating elastic_halfspace_waves...")
    from phonometry import ForceSource, GaussianPulse, elastic_fdtd_simulation

    # A 0.6 x 0.3 m aluminium block with a free upper surface, struck by a
    # short vertical force at the middle of that surface (Lamb's problem):
    # one snapshot shows the three wave types at their own speeds, the
    # Rayleigh wave hugging the surface just behind the S front.
    c_p, c_s, rho, dx = 6320.0, 3130.0, 2700.0, 0.001
    width = 8e-6
    duration = 7.3e-5
    cfl = 0.6
    dt = cfl * dx / (c_p * np.sqrt(2.0))
    steps = round(duration / dt)
    res = elastic_fdtd_simulation(
        c_p, c_s, dx, duration, rho=rho, shape=(300, 600), cfl=cfl,
        sources=[ForceSource(ix=300, iy=0, direction="y", amplitude=1e6,
                             waveform=GaussianPulse(0, 0, width=width).value)],
        boundaries={"top": "free"},
        snapshot_every=steps, snapshot_field="vy",
    )

    _fig, ax = plt.subplots(figsize=(9.5, 5.4))
    assert res.snapshots is not None and res.snapshot_times is not None
    vmax = 0.18 * float(np.abs(res.snapshots[-1]).max())
    res.plot(kind="snapshot", frame=-1, ax=ax, vmin=-vmax, vmax=vmax)

    # Wavefront radii from the pulse centre time (t0 = 4 * width).
    t_eff = float(res.snapshot_times[-1]) - 4.0 * width
    x0 = (300 + 0.5) * dx
    theta = np.linspace(0.0, np.pi, 181)
    for radius in (c_p * t_eff, c_s * t_eff):
        ax.plot(x0 + radius * np.cos(theta), radius * np.sin(theta),
                linestyle=":", linewidth=1.0, color=COLOR_FG, alpha=0.65)
    ann: dict[str, Any] = {
        "fontsize": 9, "color": COLOR_FG,
        "arrowprops": {"arrowstyle": "->", "color": COLOR_FG, "lw": 0.9}}
    r_p, r_s = c_p * t_eff, c_s * t_eff
    ax.annotate("P wave front",
                xy=(x0 + r_p * np.cos(np.radians(55)),
                    r_p * np.sin(np.radians(55))),
                xytext=(0.585, 0.15), ha="right", **ann)
    ax.annotate("S wave front",
                xy=(x0 + r_s * np.cos(np.radians(35)),
                    r_s * np.sin(np.radians(35))),
                xytext=(0.525, 0.038), ha="right", **ann)
    ax.annotate("Rayleigh wave",
                xy=(x0 - 0.92 * r_s, 0.004), xytext=(0.045, 0.06), **ann)
    ax.set_title("Rayleigh wave along a free aluminium surface "
                 "(elastic FDTD)", fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "elastic_halfspace_waves.png")
    plt.close()


@cache
def _scholte_interface_result() -> Any:
    """Water over a soft seabed, explosive shot near the contact (cached).

    Language/theme independent: the van Vossen (2002) benchmark media
    (water 1500/1000 over a 3500/2000/2500 solid), a 50 Hz Ricker
    explosion 10 m above the interface, run until the Scholte train has
    crawled ~330 m along the contact.
    """
    from phonometry import ExplosionSource, elastic_fdtd_simulation

    ny, nx, dx = 200, 500, 1.0
    c_p = np.full((ny, nx), 1500.0)
    c_s = np.zeros((ny, nx))
    rho = np.full((ny, nx), 1000.0)
    c_p[100:], c_s[100:], rho[100:] = 3500.0, 2000.0, 2500.0
    f0, t0 = 50.0, 0.030
    duration = 0.232
    dt = 0.6 * dx / (3500.0 * np.sqrt(2.0))
    steps = round(duration / dt)

    def ricker(t: float) -> float:
        a = (np.pi * f0 * (t - t0)) ** 2
        return float((1.0 - 2.0 * a) * np.exp(-a))

    return elastic_fdtd_simulation(
        c_p, c_s, dx, duration, rho=rho,
        sources=[ExplosionSource(ix=60, iy=89, waveform=ricker,
                                 amplitude=1e3)],
        boundaries="absorbing", absorbing_layer_cells=20,
        snapshot_every=steps, snapshot_field="vy",
    )


def generate_scholte_interface_wave(output_dir: str) -> None:
    """Elastic FDTD: Scholte wave crawling along a water-seabed contact."""
    print("Generating scholte_interface_wave...")
    res = _scholte_interface_result()

    _fig, ax = plt.subplots(figsize=(9.5, 4.6))
    assert res.snapshots is not None
    vmax = 0.22 * float(np.abs(res.snapshots[-1]).max())
    res.plot(kind="snapshot", frame=-1, ax=ax, vmin=-vmax, vmax=vmax)
    # The light RdBu_r field stays light everywhere, so its in-axes
    # annotations keep a fixed dark ink rather than COLOR_FG; on the dark
    # theme the renderer picks the black-centred field, which takes the
    # white FIELD_INK of the animated clips.
    ink = FIELD_INK if _FILENAME_SUFFIX else "#3a3a3a"
    ax.axhline(100.0, color=ink, linestyle=":", linewidth=1.0, alpha=0.75)
    ann: dict[str, Any] = {
        "fontsize": 9, "color": ink,
        "arrowprops": {"arrowstyle": "->", "color": ink, "lw": 0.9}}
    ax.annotate("Scholte wave, evanescent on both sides",
                xy=(350.0, 108.0), xytext=(120.0, 170.0), **ann)
    ax.annotate("direct water wave", xy=(300.0, 35.0),
                xytext=(120.0, 14.0), **ann)
    txt: dict[str, Any] = {"fontsize": 9, "color": ink, "alpha": 0.9}
    ax.text(12.0, 32.0, "water 1500 m/s", **txt)
    ax.text(12.0, 192.0, "seabed 3500 / 2000 m/s", **txt)
    ax.set_title("Scholte wave along a water-sediment interface "
                 "(elastic FDTD)", fontweight="bold", pad=10)

    plt.tight_layout()
    save_figure(output_dir, "scholte_interface_wave.png")
    plt.close()


def generate_panel_insulation_concept(output_dir: str) -> None:
    """Theoretical panel sound insulation: the four PR-I predictions."""
    print("Generating panel_insulation_concept.png...")
    from phonometry import (
        coincidence_frequency,
        composite_transmission_loss,
        double_wall_transmission_loss,
        mass_law_transmission_loss,
        mass_spring_mass_resonance,
        plate_bending_stiffness,
        radiation_efficiency,
        single_panel_transmission_loss,
    )

    bands = np.array(
        [50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
         1250, 1600, 2000, 2500, 3150, 4000, 5000], dtype=float
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Single panel: field-incidence mass law and the coincidence dip.
    bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = coincidence_frequency(15.0, bp)
    ml = mass_law_transmission_loss(bands, 15.0, incidence="field")
    sharp = single_panel_transmission_loss(
        bands, 15.0, critical_frequency=fc, loss_factor=0.024
    )
    ax = axes[0, 0]
    ax.semilogx(bands, ml, color=COLOR_TERTIARY, ls="--", lw=1.6,
                label="field-incidence mass law")
    ax.semilogx(bands, sharp.transmission_loss, color=COLOR_PRIMARY, lw=2.0,
                marker="o", markersize=3, label="single panel R (Sharp)")
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_c$")
    ax.set_title("Single panel: mass law and coincidence",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (b) Double wall: mass-spring-mass resonance and cavity gain.
    dw = double_wall_transmission_loss(bands, 12.0, 12.0, 0.075)
    single = mass_law_transmission_loss(bands, 24.0, incidence="field")
    f0 = mass_spring_mass_resonance(12.0, 12.0, 0.075)
    ax = axes[0, 1]
    ax.semilogx(bands, single, color=COLOR_TERTIARY, ls="--", lw=1.6,
                label="single leaf (total mass)")
    ax.semilogx(bands, dw.transmission_loss, color=COLOR_PRIMARY, lw=2.0,
                marker="o", markersize=3, label="double wall R")
    ax.axvline(f0, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_0$")
    ax.set_title("Double wall: mass-spring-mass resonance",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (c) Radiation efficiency of a bending plate.
    sigma = radiation_efficiency(bands, 1.5, 1.25, fc)
    ax = axes[1, 0]
    ax.loglog(bands, sigma.radiation_efficiency, color=COLOR_PRIMARY, lw=2.0,
              marker="o", markersize=3, label=r"$\sigma(f)$")
    ax.axhline(1.0, color=COLOR_FG, ls=":", lw=0.9, alpha=0.5, label="$\\sigma = 1$")
    ax.axvline(fc, color=COLOR_SECONDARY, ls=":", lw=1.2, label="$f_c$")
    ax.set_title("Radiation efficiency of a bending plate",
                 fontweight="bold", pad=10)
    ax.set_ylabel(r"Radiation efficiency $\sigma$")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    # (d) Composite wall with a small aperture (open-area cap).
    wall = sharp.transmission_loss
    open_area = 0.01
    n = bands.size
    composite = np.array([
        float(composite_transmission_loss(
            [1.0 - open_area, open_area], [wall[i], 0.0]))
        for i in range(n)
    ])
    ax = axes[1, 1]
    ax.semilogx(bands, wall, color=COLOR_PRIMARY, lw=2.0, marker="o",
                markersize=3, label="solid wall alone")
    ax.semilogx(bands, composite, color=COLOR_SECONDARY, lw=2.0, marker="s",
                markersize=3, label="wall + 1 % open slit")
    ax.axhline(10.0 * np.log10(1.0 / open_area), color=COLOR_FG, ls=":", lw=0.9,
               alpha=0.5, label="open-area limit")
    ax.set_title("Composite wall with a small aperture",
                 fontweight="bold", pad=10)
    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_xlabel("Frequency [Hz]")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    format_frequency_axis(ax, float(bands.min()), float(bands.max()))

    fig.suptitle(
        "Theoretical panel sound insulation (Bies / Hopkins / Cremer)",
        fontweight="bold", fontsize=14,
    )
    plt.tight_layout()
    save_figure(output_dir, "panel_insulation_concept.png")
    plt.close()


def generate_silencer_expansion_chamber(output_dir: str) -> None:
    """Expansion-chamber transmission loss for four area ratios (Bies 8.111)."""
    print("Generating silencer_expansion_chamber.svg...")
    from phonometry import expansion_chamber

    freqs = np.linspace(20.0, 2000.0, 2000)
    pipe_area, length = 0.01, 0.3
    ratios = (2.0, 4.0, 8.0, 16.0)
    colors = (COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#9467bd")

    _fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for m, color in zip(ratios, colors):
        res = expansion_chamber(freqs, length, m * pipe_area, pipe_area)
        peak = 10.0 * np.log10(1.0 + 0.25 * (m - 1.0 / m) ** 2)
        ax.plot(freqs, res.transmission_loss, color=color, lw=1.8,
                label=f"m = {int(m)}  →  {peak:.1f} dB")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Expansion-chamber transmission loss (Bies Eq. 8.111)",
                 fontweight="bold", pad=10)
    ax.set_xlim(20.0, 2000.0)
    ax.set_ylim(0.0, 20.0)
    format_frequency_axis(ax, 20.0, 2000.0)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", title="Area ratio m = Sexp/Sduct")
    plt.tight_layout()
    save_figure(output_dir, "silencer_expansion_chamber.svg")
    plt.close()


# Every documentation figure, in the order the sequential generator has always
# produced them. This registry is the single source of truth for both the
# sequential path (``generate_all``) and the parallel runner (``--jobs``),
# and for the ``--figure`` name filter.


def generate_regularized_inversion(output_dir: str) -> None:
    """Kirkeby inversion of a loudspeaker-like band-pass response."""
    print("Generating regularized_inversion...")
    from scipy import signal as sp_signal

    from phonometry import regularized_inverse_filter

    fs = 48000.0
    b, bb = sp_signal.butter(2, [100.0, 8000.0], btype="bandpass", fs=fs)
    imp = np.zeros(2048)
    imp[0] = 1.0
    h = sp_signal.lfilter(b, bb, imp)

    res = regularized_inverse_filter(h, fs, f_range=(200.0, 4000.0))

    freqs = res.frequencies
    pos = freqs > 0.0
    tiny = np.finfo(np.float64).tiny
    h_mag = np.abs(res.response_spectrum)
    peak = float(np.max(h_mag))
    inv_mag = np.abs(res.spectrum)
    eq_mag = h_mag * inv_mag

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(h_mag[pos], tiny) / peak),
                color=COLOR_PRIMARY, linewidth=1.4,
                label="Measured response $|H|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(inv_mag[pos] * peak, tiny)),
                color=COLOR_SECONDARY, linewidth=1.4,
                label=r"Inverse filter $|H_{\mathrm{inv}}|$")
    ax.semilogx(freqs[pos],
                20.0 * np.log10(np.maximum(eq_mag[pos], tiny)),
                color=COLOR_TERTIARY, linewidth=1.8,
                label=r"Equalized $|H \cdot H_{\mathrm{inv}}|$")
    ax.axvspan(200.0, 4000.0, color=theme_fill(COLOR_PRIMARY, ax), zorder=0,
               label="Equalized band (200 Hz - 4 kHz)")
    ax.set_xlim(20.0, fs / 2.0)
    ax.set_ylim(-50.0, 15.0)
    format_frequency_axis(ax, 20.0, fs / 2.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Regularized Spectral Inversion (Kirkeby Frequency-"
                 "Dependent Regularization)", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9)
    ax.text(0.985, 0.97,
            "unity in-band; outside, the frequency-dependent\n"
            "regularization caps the gain instead of boosting noise",
            transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
            color=COLOR_FG)
    plt.tight_layout()
    save_figure(output_dir, "regularized_inversion.svg")
    plt.close()


def generate_shaped_sweep(output_dir: str) -> None:
    """A pink shaped sweep: waveform and spectrum against the target."""
    print("Generating shaped_sweep...")
    from scipy import signal as sp_signal

    from phonometry import shaped_sweep_signal

    fs = 48000
    res = shaped_sweep_signal(fs, 50.0, 5000.0, 2.0, target="pink")
    x = np.asarray(res)
    t = np.arange(x.size) / fs

    nperseg = 8192
    freqs_w, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg,
                                   noverlap=3 * nperseg // 4)
    tiny = np.finfo(np.float64).tiny
    band_w = (freqs_w >= 50.0) & (freqs_w <= 5000.0)
    welch_db = 10.0 * np.log10(np.maximum(psd, tiny))
    welch_db -= float(np.max(welch_db[band_w]))
    grid = res.frequencies
    band_g = (grid >= 50.0) & (grid <= 5000.0)
    target_db = 20.0 * np.log10(np.maximum(res.magnitude, tiny))
    target_db -= float(np.max(target_db[band_g]))

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(t, x, color=COLOR_PRIMARY, linewidth=0.5)
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0.0, float(t[-1]))
    axes[0].set_title("Shaped Sweep with an Arbitrary Target Spectrum "
                      "(Group-Delay Synthesis)", fontweight="bold", pad=12)
    axes[0].grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    axes[0].set_axisbelow(True)
    axes[0].text(0.985, 0.95,
                 "nearly constant envelope: the energy shaping lives\n"
                 "in the dwell time, not in the amplitude",
                 transform=axes[0].transAxes, va="top", ha="right",
                 fontsize=8.5, color=COLOR_FG)

    posw = freqs_w > 0.0
    axes[1].semilogx(freqs_w[posw], welch_db[posw], color=COLOR_PRIMARY,
                     linewidth=1.3, label="Welch spectrum of the sweep")
    posg = grid > 0.0
    axes[1].semilogx(grid[posg], target_db[posg], color=COLOR_SECONDARY,
                     linewidth=1.5, linestyle="--",
                     label="Pink target (-3 dB per octave)")
    axes[1].axvspan(50.0, 5000.0, color=theme_fill(COLOR_PRIMARY, axes[1]),
                    zorder=0, label="Sweep band (50 Hz - 5 kHz)")
    axes[1].set_xlim(20.0, 20000.0)
    axes[1].set_ylim(-60.0, 8.0)
    format_frequency_axis(axes[1], 20.0, 20000.0)
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].set_ylabel("Level re in-band max [dB]")
    axes[1].grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    axes[1].set_axisbelow(True)
    axes[1].legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "shaped_sweep.svg")
    plt.close()


def generate_sound_absorption_measurement(output_dir: str) -> None:
    """ISO 354: reverberation-room alpha_s spectrum from the two decay times."""
    print("Generating sound_absorption_measurement...")
    from phonometry import materials

    # The materials guide's ISO 354 example: one-third-octave reverberation
    # times of a 200 m^3 room, empty (T1) and with a 10.8 m^2 porous absorber
    # sample installed (T2), inverted through Sabine to the alpha_s spectrum.
    freqs = np.array([100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                      1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000], float)
    t_empty = np.array([9.0, 9.0, 8.8, 8.6, 8.4, 8.2, 8.0, 7.8, 7.5, 7.2,
                        6.9, 6.6, 6.2, 5.8, 5.4, 5.0, 4.6, 4.2])
    t_specimen = np.array([8.4, 8.2, 7.7, 7.2, 6.5, 5.7, 4.9, 4.2, 3.6, 3.15,
                           2.85, 2.65, 2.55, 2.5, 2.55, 2.6, 2.7, 2.85])
    m = materials.measure_sound_absorption(
        freqs, t_empty, t_specimen, volume=200.0, area=10.8, temperature=20.0
    )
    m.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "sound_absorption_measurement.svg")
    plt.close()


def generate_impedance_tube_result(output_dir: str) -> None:
    """ISO 10534-2: alpha(f) and |r| of a porous sample in a 100 mm tube."""
    print("Generating impedance_tube_result...")
    from phonometry import materials

    # A 50 mm porous absorber (Miki model, sigma = 20 kPa s/m^2) measured in a
    # 100 mm tube with 100 mm microphone spacing (working band roughly 170 Hz
    # to 1.5 kHz): the layer model provides the true reflection factor, from
    # which the measured transfer function H12 is synthesised and reduced back
    # through the Clause 7 chain.
    f = np.linspace(200.0, 1500.0, 260)
    med = materials.miki(f, 20000.0)
    layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
    spacing, x1, c0 = 0.10, 0.20, 343.2
    k0 = materials.tube_wavenumber(f, c0)
    x2 = x1 - spacing
    r_true = layer.reflection
    h12 = (np.exp(1j * k0 * x2) + r_true * np.exp(-1j * k0 * x2)) / \
          (np.exp(1j * k0 * x1) + r_true * np.exp(-1j * k0 * x1))
    result = materials.two_microphone_impedance(
        h12, frequency=f, spacing=spacing, x1=x1, speed_of_sound=c0,
        characteristic_impedance=407.0, diameter=0.10,
    )
    result.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "impedance_tube_result.svg")
    plt.close()


def generate_transfer_matrix_tl(output_dir: str) -> None:
    """ASTM E2611: TL and hard-backed absorption from the four-pole matrix."""
    print("Generating transfer_matrix_tl...")
    from phonometry import materials

    # The chain matrix of a 50 mm porous layer (Miki, sigma = 20 kPa s/m^2)
    # exposed by the layer solver, read back through the ASTM E2611 machinery:
    # the four-pole entries give the normal-incidence transmission loss and the
    # hard-backed absorption of the same specimen.
    f = np.linspace(200.0, 1600.0, 300)
    med = materials.miki(f, 20000.0)
    layer = materials.layered_absorber(f, [materials.PorousLayer(0.05, med)])
    chain = layer.transfer_matrix
    tm = materials.TransferMatrix(
        t11=chain[0, 0], t12=chain[0, 1], t21=chain[1, 0], t22=chain[1, 1]
    )
    tm.plot(f, 407.0, language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "transfer_matrix_tl.svg")
    plt.close()


def generate_porous_medium_model(output_dir: str) -> None:
    """Miki equivalent fluid: normalised Zc and k of a porous material."""
    print("Generating porous_medium_model...")
    from phonometry import materials

    # The porous-absorbers guide's model example: a rockwool-class material of
    # flow resistivity 20 kPa s/m^2 evaluated with the Miki (1990) regression.
    f = np.geomspace(100.0, 5000.0, 260)
    med = materials.miki(f, 20000.0)
    med.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "porous_medium_model.svg")
    plt.close()


def generate_mpp_absorption_peak(output_dir: str) -> None:
    """Maa (1998) Fig. 5 microperforated panel: resonant absorption peak."""
    print("Generating mpp_absorption_peak...")
    from phonometry import materials

    # Maa's own design: d = t = 0.2 mm, holes every 2.5 mm, 6 cm cavity. The
    # viscous losses in the submillimetre holes absorb without any porous
    # material; the peak reaches alpha = 0.96 near 677 Hz.
    eps = (np.pi / 4.0) * (0.2 / 2.5) ** 2
    f = np.linspace(100.0, 4000.0, 1200)
    res = materials.layered_absorber(
        f, [materials.MicroperforatedPlateLayer(0.2e-3, 0.1e-3, eps),
            materials.AirLayer(0.06)],
    )
    ax = res.plot(language=_LANG)
    format_frequency_axis(ax, 100.0, 4000.0)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "mpp_absorption_peak.svg")
    plt.close()


def generate_diffuse_field_absorption(output_dir: str) -> None:
    """Paris integral: random-incidence vs normal-incidence absorption."""
    print("Generating diffuse_field_absorption...")
    from phonometry import materials

    # A 50 mm porous layer (Miki, sigma = 20 kPa s/m^2): the diffuse-field
    # coefficient exceeds the normal-incidence one at low frequency because
    # the oblique waves travel a longer path inside the layer.
    f = np.geomspace(125.0, 4000.0, 200)
    layers: list[Any] = [materials.PorousLayer(0.05, materials.miki(f, 20000.0))]
    normal = materials.layered_absorber(f, layers)
    diffuse = materials.diffuse_field_absorption(f, layers)
    ax = diffuse.plot(language=_LANG)
    ax.plot(f, normal.absorption, ls="--", color=COLOR_SECONDARY,
            label=r"Normal incidence $\alpha(0°)$")
    ax.legend(loc="best", fontsize="small")
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "diffuse_field_absorption.svg")
    plt.close()


def generate_steady_state_field(output_dir: str) -> None:
    """Bies steady-state room field: direct, reverberant, total and rc."""
    print("Generating steady_state_field...")
    from phonometry import room

    # A 90 dB re 1 pW source in a 12 x 8 x 4 m workshop (S = 352 m^2) with a
    # mean Sabine absorption of 0.15: the total level follows the 1/r^2 direct
    # field close in and flattens onto the reverberant plateau beyond rc.
    field = room.steady_state_field(
        sound_power_level=90.0, surface_area=352.0, mean_absorption=0.15,
    )
    field.plot(language=_LANG)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    save_figure(output_dir, "steady_state_field.svg")
    plt.close()


def generate_room_parameters_bands(output_dir: str) -> None:
    """ISO 3382: per-band EDT/T20/T30 and C50/C80 of a synthetic room IR."""
    print("Generating room_parameters_bands...")
    from phonometry import room

    # A synthetic room impulse response with a frequency-dependent decay:
    # octave-band noise carriers whose T60 falls from 1.4 s at 125 Hz to
    # 0.7 s at 4 kHz, the classic behaviour of a furnished, treated room.
    fs = 48000
    rng = np.random.default_rng(3382)
    t = np.arange(int(1.6 * fs)) / fs
    t60 = {125.0: 1.4, 250.0: 1.25, 500.0: 1.1,
           1000.0: 1.0, 2000.0: 0.85, 4000.0: 0.7}
    ir = np.zeros_like(t)
    for fc, t60_band in t60.items():
        sos = scipy_signal.butter(
            4, [fc / np.sqrt(2.0), fc * np.sqrt(2.0)], btype="bandpass",
            fs=fs, output="sos",
        )
        carrier = scipy_signal.sosfilt(sos, rng.standard_normal(t.size))
        ir += carrier * np.exp(-3.0 * np.log(10.0) / t60_band * t)
    res = room.room_parameters(ir, fs)
    res.plot(language=_LANG)
    plt.gcf().set_size_inches(12.5, 5.4)
    plt.tight_layout()
    save_figure(output_dir, "room_parameters_bands.svg")
    plt.close()


def generate_fdtd_room_modes(output_dir: str) -> None:
    """Rigid-box FDTD probe spectrum against the analytic room modes."""
    print("Generating fdtd_room_modes...")
    from phonometry import simulation

    # The fdtd-simulation guide's oracle run: a rigid 1.0 x 0.7 m box excited
    # by a Gaussian pulse; the probe spectrum peaks at the analytic rigid-room
    # mode frequencies f = (c/2) sqrt((nx/Lx)^2 + (ny/Ly)^2).
    lx, ly, dx, c = 1.0, 0.7, 0.02, 343.0
    nx, ny = round(lx / dx), round(ly / dx)
    res = simulation.fdtd_simulation(
        c, dx, 0.35, shape=(ny, nx),
        sources=[simulation.GaussianPulse(ix=7, iy=5, width=2.0e-4)],
        probes=[(nx - 4, ny - 3)],
    )
    p = res.pressures[0]
    spec = np.abs(np.fft.rfft(p * np.hanning(p.size), n=8 * p.size))
    freqs = np.fft.rfftfreq(8 * p.size, res.dt)
    sel = (freqs >= 100.0) & (freqs <= 450.0)
    level = 20.0 * np.log10(spec[sel] / np.max(spec[sel]))

    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs[sel], level, color=COLOR_PRIMARY, linewidth=1.6,
            label="Probe pressure spectrum", zorder=3)
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    for i, (mx, my) in enumerate(modes):
        f_mode = 0.5 * c * float(np.hypot(mx / lx, my / ly))
        ax.axvline(f_mode, color=COLOR_SECONDARY, linestyle=":", linewidth=1.2,
                   zorder=2,
                   label="Analytic mode frequencies" if i == 0 else None)
        ax.annotate(f"({mx},{my})", xy=(f_mode, 1.5), ha="center", fontsize=9,
                    color=COLOR_FG)
    ax.set_title("Rigid-box FDTD probe spectrum vs analytic modes",
                 fontweight="bold", pad=14)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Probe spectrum [dB re max]")
    ax.set_xlim(100.0, 450.0)
    ax.set_ylim(-60.0, 6.0)
    ax.grid(which="major", color=COLOR_GRID, linestyle="-", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "fdtd_room_modes.svg")
# ---------------------------------------------------------------------------
# Building & structure-borne result figures (WP: guide figure coverage).
# Each figure is a realistic user result: example data -> real function ->
# result object, drawn the way the result's own .plot() presents it.
# ---------------------------------------------------------------------------
_THIRD_OCTAVE_16 = [100, 125, 160, 200, 250, 315, 400, 500,
                    630, 800, 1000, 1250, 1600, 2000, 2500, 3150]


def _band_index_axis(ax: Any, freqs: Sequence[float] | np.ndarray,
                     fontsize: int = 8) -> np.ndarray:
    """Rotated nominal band labels on an index axis (band-data figures)."""
    x = np.arange(len(freqs))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in np.asarray(freqs)],
                       rotation=45, fontsize=fontsize)
    ax.set_xlabel(LABEL_FREQ_HZ)
    return x


def generate_extended_insulation_rating(output_dir: str) -> None:
    """ISO 717-1 Annex B enlarged-range rating of the Annex C Table C.2 wall."""
    print("Generating extended_insulation_rating...")
    from phonometry import weighted_rating_extended

    # ISO 717-1 Annex C: the 16-band R spectrum (Table C.1) enlarged to
    # 50 Hz - 5 kHz with the Table C.2 values.
    r_core = [20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
              28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5]
    freqs = [50, 63, 80, *(_THIRD_OCTAVE_16), 4000, 5000]
    ext = weighted_rating_extended([18.7, 19.2, 20.0, *r_core, 26.8, 29.2],
                                   freqs)
    assert ext.measured is not None and ext.core.shifted_reference is not None
    assert ext.core.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, freqs)
    core = slice(3, 19)                     # the 16 core bands 100-3150 Hz
    enlarged = theme_fill(COLOR_FG, ax)
    ax.axvspan(-0.5, 2.5, color=enlarged, zorder=0,
               label="enlarged range (Annex B)")
    ax.axvspan(18.5, 20.5, color=enlarged, zorder=0)
    ax.plot(x, ext.measured, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="measured R")
    ax.plot(x[core], ext.core.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5,
            label="shifted reference (100-3150 Hz)")
    unfav = ext.core.measured < ext.core.shifted_reference
    ax.fill_between(x[core], ext.core.measured, ext.core.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_title("ISO 717-1 Enlarged-Range Rating (Annex B)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Rw(C;Ctr) = {ext.rating:g}({ext.c:g};{ext.ctr:g})",
        f"C50-5000 = {ext.c_50_5000:g},  Ctr,50-5000 = {ext.ctr_50_5000:g}",
        "rating on the core bands, terms on the full range",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "extended_insulation_rating.svg")
    plt.close()


def generate_field_airborne_insulation(output_dir: str) -> None:
    """ISO 16283-1 field airborne chain: D and DnT with the T correction."""
    print("Generating field_airborne_insulation...")
    from phonometry import airborne_insulation, weighted_rating

    # A separating wall between dwellings: source/receiving levels and the
    # measured receiving-room T per one-third-octave band.
    l1 = np.array([92.3, 93.1, 94.0, 94.4, 94.8, 95.0, 95.2, 95.4,
                   95.3, 95.1, 94.8, 94.4, 93.9, 93.3, 92.5, 91.6])
    d = np.array([38.2, 40.1, 42.6, 45.2, 47.8, 50.1, 52.3, 54.0,
                  55.6, 57.1, 58.2, 59.0, 59.6, 60.1, 60.3, 59.8])
    t2 = np.array([0.62, 0.58, 0.55, 0.53, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.45, 0.44, 0.43, 0.43, 0.42])
    field = airborne_insulation(l1, l1 - d, t2, area=12.5, volume=30.4)
    assert field.r_prime is not None
    w = weighted_rating(field.dnt)
    w_rp = weighted_rating(field.r_prime)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.fill_between(x, field.d, field.dnt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="10 lg(T/T0)")
    ax.plot(x, field.d, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D (level difference)")
    ax.plot(x, field.dnt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=5, label="DnT (standardized)")

    ax.set_ylabel("Level difference [dB]")
    ax.set_title("ISO 16283-1 Field Airborne Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"DnT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        f"R'w = {w_rp.rating} dB   (S = 12.5 m², V = 30.4 m³)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "field_airborne_insulation.svg")
    plt.close()


def generate_facade_field_insulation(output_dir: str) -> None:
    """ISO 16283-3 façade quantities D2m, D2m,nT, D2m,n and R'45 per band."""
    print("Generating facade_field_insulation...")
    from phonometry import facade_insulation, weighted_rating

    # A dwelling façade under the 45-degree loudspeaker method: the outdoor
    # level 2 m in front, the receiving-room level and T per band.
    l1_2m = np.array([76.0, 77.0, 78.0, 78.5, 79.0, 79.0, 79.0, 79.0,
                      78.5, 78.0, 77.5, 77.0, 76.5, 76.0, 75.0, 74.0])
    d2m = np.array([24.0, 25.5, 27.0, 28.5, 30.0, 31.5, 33.0, 34.5,
                    36.0, 37.0, 38.0, 38.5, 39.0, 39.0, 38.5, 38.0])
    t2 = np.array([0.65, 0.62, 0.58, 0.55, 0.52, 0.50, 0.49, 0.48,
                   0.47, 0.46, 0.45, 0.44, 0.43, 0.43, 0.42, 0.42])
    fac = facade_insulation(l1_2m, l1_2m - d2m, t2, volume=32.0, area=10.8,
                            surface_level=l1_2m + 3.0, method="loudspeaker",
                            frequencies=np.asarray(_THIRD_OCTAVE_16, float))
    assert fac.d_2m_n is not None and fac.r_prime is not None
    w = weighted_rating(fac.d_2m_nt)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, fac.d_2m_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=5,
            zorder=6, label="D2m,nT (standardized)")
    ax.plot(x, fac.d_2m, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=5, zorder=5, label="D2m = L1,2m - L2")
    ax.plot(x, fac.d_2m_n, ":", color=COLOR_SECONDARY, linewidth=1.8,
            marker=".", zorder=5, label="D2m,n (normalized)")
    ax.plot(x, fac.r_prime, "-.", color=COLOR_TERTIARY, linewidth=1.8,
            marker="^", markersize=5, zorder=5, label="R'45° (element)")

    ax.set_ylabel("Level difference / reduction index [dB]")
    ax.set_title("ISO 16283-3 Field Facade Insulation",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Dls,2m,nT,w(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        "45° loudspeaker method (-1.5 dB on R')",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "facade_field_insulation.svg")
    plt.close()


def generate_survey_impact_insulation(output_dir: str) -> None:
    """ISO 10052 survey impact method: Li -> L'nT with the k correction."""
    print("Generating survey_impact_insulation...")
    from phonometry import reverberation_index, survey_impact_insulation

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0]
    # Tapping machine on the floor above: octave-band receiving-room levels
    # and the measured receiving-room reverberation time.
    li = np.array([66.0, 64.0, 62.0, 60.0, 55.0])
    k = reverberation_index([0.70, 0.60, 0.50, 0.45, 0.40])
    res = survey_impact_insulation(li, k, volume=50.0)
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands, fontsize=10)
    ax.fill_between(x, res.l_i, res.l_nt, color=COLOR_TERTIARY, alpha=0.18,
                    zorder=0, label="-k = -10 lg(T/T0)")
    ax.plot(x, res.l_i, "--o", color=COLOR_PRIMARY, linewidth=1.8,
            markersize=6, zorder=5, label="Li (impact level)")
    ax.plot(x, res.l_nt, "-s", color=COLOR_FG, linewidth=2.4, markersize=6,
            zorder=5, label="L'nT (standardized)")

    ax.set_ylabel("Impact sound pressure level [dB]")
    ax.set_title("ISO 10052 Survey Method: Impact Sound",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)

    info = [
        f"L'nT,w(CI) = {res.rating.rating}({res.rating.ci}) dB",
        "note the minus sign: a live room lowers L'nT",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "survey_impact_insulation.svg")
    plt.close()


def generate_lab_insulation_result(output_dir: str) -> None:
    """ISO 10140 laboratory quantities: the airborne R and the impact Ln."""
    print("Generating lab_insulation_result...")
    from phonometry import lab_airborne_insulation, lab_impact_insulation

    # ISO 717-1 Annex C wall measured in an ISO 10140 suite (S = 10 m2,
    # V = 50 m3, T = 0.8 s) and the ISO 717-2 Annex C floor tapping levels.
    r = np.array([20.4, 16.3, 17.7, 22.6, 22.4, 22.7, 24.8, 26.6,
                  28.0, 30.5, 31.8, 32.5, 33.4, 33.0, 31.0, 25.5])
    l1 = np.full(16, 90.0)
    t2 = np.full(16, 0.8)
    lab = lab_airborne_insulation(l1, l1 - r, t2, area=10.0, volume=50.0)
    li = np.array([62.1, 63.2, 63.5, 66.2, 68.5, 70.0, 71.7, 73.1,
                   73.8, 73.5, 73.8, 73.3, 73.1, 73.0, 72.4, 71.2])
    imp = lab_impact_insulation(li, t2, volume=50.0)
    assert lab.rating is not None and imp.rating is not None

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax in (ax1, ax2):
        _band_index_axis(ax, _THIRD_OCTAVE_16, fontsize=7)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
    x = np.arange(16)

    ax1.plot(x, lab.r, "-o", color=COLOR_PRIMARY, linewidth=2.2,
             markersize=5, zorder=5, label="measured R")
    assert lab.rating.shifted_reference is not None
    ax1.plot(x, lab.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax1.set_ylabel("Sound reduction index R [dB]")
    ax1.set_title(f"Airborne: Rw(C;Ctr) = {lab.rating.rating}"
                  f"({lab.rating.c};{lab.rating.ctr}) dB", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)

    ax2.plot(x, imp.l_n, "-o", color=COLOR_SECONDARY, linewidth=2.2,
             markersize=5, zorder=5, label="normalized Ln")
    assert imp.rating.shifted_reference is not None
    ax2.plot(x, imp.rating.shifted_reference, "--s", color=COLOR_FG,
             linewidth=1.6, markersize=4, zorder=5, label="shifted reference")
    ax2.set_ylabel("Impact sound pressure level Ln [dB]")
    ax2.set_title(f"Impact: Ln,w(CI) = {imp.rating.rating}"
                  f"({imp.rating.ci}) dB", fontsize=11)
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("ISO 10140 Laboratory Insulation (flanking suppressed)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "lab_insulation_result.svg")
    plt.close()


def generate_intensity_element_insulation(output_dir: str) -> None:
    """ISO 15186-1 element-normalized level difference of a small element."""
    print("Generating intensity_element_insulation...")
    from phonometry import intensity_element_normalized_difference

    # A trickle ventilator in a masonry wall: source-room SPL 85 dB and the
    # normal intensity level over the Sm = 12 m2 measurement surface.
    l_in = np.array([57.9, 62.0, 60.6, 55.7, 55.9, 55.6, 53.5, 51.7,
                     50.3, 47.8, 46.5, 45.8, 44.9, 45.3, 47.3, 52.8])
    res = intensity_element_normalized_difference(
        np.full(16, 85.0), l_in, measurement_area=12.0, n=1
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_i_n_e, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="DI,n,e (element)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Element normalized level difference [dB]")
    ax.set_title("ISO 15186-1 Small-Element Insulation by Intensity",
                 fontweight="bold", pad=12)
    ax.margins(y=0.1)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"DI,n,e,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "DI,n,e = Lp1 - 6 - [LIn + 10 lg(Sm/A0)] + 10 lg N",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "intensity_element_insulation.svg")
    plt.close()


def generate_flanking_level_difference(output_dir: str) -> None:
    """ISO 10848 overall flanking descriptor Dn,f with its Dn,f,w rating."""
    print("Generating flanking_level_difference...")
    from phonometry import normalized_flanking_level_difference

    # A lightweight junction measured in the laboratory: source-room level,
    # receiving-room level over the flanking path and the absorption area.
    l1 = np.full(16, 80.0)
    dnf_target = np.array([48, 49, 50, 51, 52, 54, 55, 57,
                           58, 59, 60, 61, 62, 63, 64, 65], dtype=float)
    res = normalized_flanking_level_difference(
        l1, l1 - dnf_target, absorption_area=np.full(16, 10.0)
    )
    assert res.rating is not None
    assert res.rating.shifted_reference is not None
    assert res.rating.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.d_n_f, "-o", color=COLOR_PRIMARY, linewidth=2.2,
            markersize=5, zorder=5, label="Dn,f (flanking)")
    ax.plot(x, res.rating.shifted_reference, "--s", color=COLOR_FG,
            linewidth=1.8, markersize=5, zorder=5, label="shifted reference")
    unfav = res.rating.measured < res.rating.shifted_reference
    ax.fill_between(x, res.rating.measured, res.rating.shifted_reference,
                    where=unfav.tolist(), color=COLOR_SECONDARY, alpha=0.25,
                    interpolate=True, zorder=1,
                    label="unfavourable deviations")

    ax.set_ylabel("Normalized flanking level difference [dB]")
    ax.set_title("ISO 10848 Airborne Flanking Transmission",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        (f"Dn,f,w(C;Ctr) = {res.rating.rating}"
         f"({res.rating.c};{res.rating.ctr}) dB"),
        "Dn,f = L1 - L2 - 10 lg(A/A0)",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "flanking_level_difference.svg")
    plt.close()


def generate_impact_prediction_terms(output_dir: str) -> None:
    """EN 12354-2 Annex E.3 impact prediction as its Formula (21) terms."""
    print("Generating impact_prediction_terms...")
    from phonometry import (
        equivalent_impact_level,
        impact_flanking_correction,
        predicted_impact_insulation,
    )

    # EN 12354-2 Annex E.3: a 0.14 m concrete floor (m' = 322 kg/m2) with a
    # floating floor (delta-Lw = 33 dB), mean flanking mass 145 kg/m2.
    ln_eq = float(equivalent_impact_level(322.0))
    k = float(impact_flanking_correction(322.0, 145.0))
    imp = predicted_impact_insulation(ln_w_eq=ln_eq, delta_l_w=33.0,
                                      k_correction=k)

    labels = ["$L_{n,w,eq}$", r"$-\Delta L_w$", "$+K$", "$L'_{n,w}$"]
    values = [imp.ln_w_eq, -imp.delta_l_w, imp.k_correction, imp.l_prime_n_w]
    # COLOR_MUTED rather than the gridline grey for the starting term: a bar
    # is a read value, so it has to hold up against the page on both themes.
    colors = [COLOR_MUTED, COLOR_TERTIARY, COLOR_SECONDARY, COLOR_PRIMARY]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(np.arange(4), values, color=colors, zorder=5)
    ax.axhline(0.0, color=COLOR_FG, linewidth=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(labels, fontsize=12)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:+.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, value),
                    xytext=(0, 5 if value >= 0 else -14),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=COLOR_FG)
    ax.set_ylim(min(values) - 9.0, max(values) + 9.0)

    ax.set_ylabel("Level / correction [dB]")
    ax.set_title("EN 12354-2 Impact Sound Prediction (Annex E.3)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)

    info = [
        "L'n,w = Ln,w,eq - ΔLw + K",
        f"Ln,w,eq = 164 - 35 lg(m'/m'0) = {ln_eq:.1f} dB",
        f"L'n,w = {imp.l_prime_n_w:.1f} dB → 45 dB",
    ]
    ax.text(0.985, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "impact_prediction_terms.svg")
    plt.close()


def generate_detailed_prediction_paths(output_dir: str) -> None:
    """ISO 12354-1 Annex L: which transmission path dominates each band."""
    print("Generating detailed_prediction_paths...")
    import sys
    from pathlib import Path

    tests = str(Path(__file__).resolve().parent.parent / "tests")
    if tests not in sys.path:
        sys.path.insert(0, tests)
    import iso12354_building as bld

    from phonometry import (
        detailed_airborne_prediction,
        direct_reduction_index,
        floating_floor_improvement,
        in_situ_element,
    )

    # The heavy homogeneous building of ISO 12354-1:2017 Annex L: a 220 mm
    # concrete separating floor with a floating floor, two 365 mm AAC external
    # walls and two 200 mm calcium-silicate internal walls. The building and
    # its twelve flanking paths come from the shared test fixture, which the
    # conformance report reads too, so the figure cannot drift from the rows.
    bands = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
                      800, 1000, 1250, 1600, 2000, 2500, 3150], dtype=float)
    situ = {k: in_situ_element(e, bands) for k, e in bld.elements().items()}
    delta = floating_floor_improvement(
        bands, resonance_frequency=bld.floating_floor_resonance()
    )
    res = detailed_airborne_prediction(
        bands,
        direct_index=direct_reduction_index(
            situ["floor"].sound_reduction_index, delta_r_source=delta),
        flanking_paths=bld.airborne_paths(situ, delta),
    )
    assert res.rating is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, bands)
    order = list(np.argsort(-res.fractions.max(axis=1)))
    palette = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, "#9467bd",
               "#aec7e8", "#ff9896"]
    bottom = np.zeros(x.size)
    for colour, k in zip(palette, order[:6]):
        share = 100.0 * res.fractions[k]
        ax.bar(x, share, bottom=bottom, width=0.85, color=colour,
               edgecolor="none", zorder=2, label=res.paths[k].label)
        bottom = bottom + share
    rest = order[6:]
    if rest:
        share = 100.0 * res.fractions[rest].sum(axis=0)
        # The pooled segment is de-emphasised *data*, so it takes the mid grey
        # that reads on both pages, not the gridline grey that hides in one.
        ax.bar(x, share, bottom=bottom, width=0.85, color=COLOR_MUTED,
               edgecolor="none", zorder=2, label="other paths")
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Share of transmitted energy [%]")
    ax.set_title("ISO 12354-1 Detailed Model: Dominant Path per Band (Annex L)",
                 fontweight="bold", pad=12)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(x, res.r_prime, "-o", color=COLOR_FG, linewidth=2.0,
              markersize=4, zorder=5, label="R' (apparent)")
    twin.set_ylabel("Apparent sound reduction index R' [dB]")
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = twin.get_legend_handles_labels()
    ax.legend(handles + extra, labels + extra_labels, loc="upper left",
              fontsize=9, ncol=3)

    info = [
        "R' = -10 lg(Σ 10^(-Rij/10))",
        f"R'w (C; Ctr) = {res.rating.rating} ({res.rating.c}; {res.rating.ctr}) dB",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "detailed_prediction_paths.svg")
    plt.close()


def generate_radiated_power_outdoor(output_dir: str) -> None:
    """EN 12354-4 Annex G radiated sound power of a wall with a door."""
    print("Generating radiated_power_outdoor...")
    from phonometry import FacadeElement, radiated_sound_power

    # EN 12354-4 Annex G, side 1: a 10 x 20 m concrete wall segment with a
    # 6 x 4 m industrial door, inside level Lp,in and Cd = -5 dB.
    bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    seg = radiated_sound_power(
        [FacadeElement("wall", area=176.0, r=[32, 36, 36, 33, 39, 49, 57, 63]),
         FacadeElement("door", area=24.0, r=[21, 23, 28, 30, 30, 30, 30, 30])],
        lp_in=[70, 74, 76, 72, 70, 67, 62, 57], area=200.0, c_d=-5.0,
        r_prime_cap=40.0, octave_bands=bands)
    assert seg.l_w_dba is not None

    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.bar(x, seg.l_w, color=COLOR_PRIMARY, alpha=0.85, zorder=5,
           label="radiated $L_W$ per octave")
    ax.axhline(seg.l_w_dba, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.6, zorder=6,
               label=f"$L_{{WA}}$ = {seg.l_w_dba:.1f} dB(A)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Radiated sound power level [dB re 1 pW]")
    ax.set_ylim(0.0, float(np.max(seg.l_w)) * 1.35)
    ax.set_title("EN 12354-4 Radiated Sound Power (Annex G)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    info = [
        "LW = Lp,in + Cd - R' + 10 lg(S/S0)",
        "wall 176 m² + industrial door 24 m², Cd = -5 dB",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG, zorder=10,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "radiated_power_outdoor.svg")
    plt.close()


def generate_single_panel_rating(output_dir: str) -> None:
    """Sharp single-panel R(f) prediction rated per ISO 717-1 (6 mm glass)."""
    print("Generating single_panel_rating...")
    from phonometry import (
        coincidence_frequency,
        plate_bending_stiffness,
        single_panel_transmission_loss,
    )

    # 6 mm float glass: E = 62 GPa, rho = 2500 kg/m3, nu = 0.24, eta = 0.024.
    bands = np.asarray(_THIRD_OCTAVE_16, dtype=float)
    mass = 2500.0 * 0.006
    bp = plate_bending_stiffness(6.2e10, 0.006, 0.24)
    fc = float(coincidence_frequency(mass, bp))
    res = single_panel_transmission_loss(bands, mass, critical_frequency=fc,
                                         loss_factor=0.024)
    w = res.rating()
    assert w.shifted_reference is not None and w.measured is not None

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, _THIRD_OCTAVE_16)
    ax.plot(x, res.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5, label="predicted R (Sharp)")
    ax.plot(x, w.shifted_reference, "--s", color=COLOR_FG, linewidth=1.8,
            markersize=5, zorder=5, label="shifted reference")
    unfav = w.measured < w.shifted_reference
    ax.fill_between(x, w.measured, w.shifted_reference, where=unfav.tolist(),
                    color=COLOR_SECONDARY, alpha=0.25, interpolate=True,
                    zorder=1, label="unfavourable deviations")
    idx_fc = float(np.interp(np.log10(fc), np.log10(bands), x))
    ax.axvline(idx_fc, color=COLOR_TERTIARY, linestyle=":", linewidth=1.6,
               zorder=4, label=f"coincidence fc = {fc:.0f} Hz")

    ax.set_ylabel("Sound reduction index R [dB]")
    ax.set_title("Predicted Single-Panel Insulation Rated per ISO 717-1",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        f"Rw(C;Ctr) = {w.rating}({w.c};{w.ctr}) dB",
        "6 mm float glass, m'' = 15 kg/m², η = 0.024",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "single_panel_rating.svg")
    plt.close()


def generate_junction_kij_thickness(output_dir: str) -> None:
    """Wave-approach Kij versus the plate thickness ratio (Hopkins Eq. 5.116)."""
    print("Generating junction_kij_thickness...")
    from phonometry import junction_transmission, wave_vibration_reduction_index

    # Concrete plates (cL = 3200 m/s, rho = 2400 kg/m3): plate 1 fixed at
    # 100 mm, plate 2 swept from 50 mm to 400 mm.
    h1, cl, rho = 0.1, 3200.0, 2400.0
    ratios = np.linspace(0.5, 4.0, 36)
    curves: dict[str, list[float]] = {
        "X corner": [], "X straight": [], "T-junction (1) corner": [],
        "L corner": [],
    }
    for ratio in ratios:
        h2 = h1 * float(ratio)
        res_x = junction_transmission("X", h1, cl, rho * h1, h2, cl, rho * h2)
        assert res_x.straight_average is not None
        curves["X corner"].append(res_x.corner_reduction_index)
        curves["X straight"].append(float(wave_vibration_reduction_index(
            res_x.straight_average, res_x.critical_frequency2)))
        res_t = junction_transmission("T1", h1, cl, rho * h1, h2, cl, rho * h2)
        curves["T-junction (1) corner"].append(res_t.corner_reduction_index)
        res_l = junction_transmission("L", h1, cl, rho * h1, h2, cl, rho * h2)
        curves["L corner"].append(res_l.corner_reduction_index)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    styles = [("-", COLOR_PRIMARY), ("--", COLOR_PRIMARY),
              ("-", COLOR_SECONDARY), ("-", COLOR_TERTIARY)]
    for (label, values), (ls, color) in zip(curves.items(), styles):
        ax.plot(ratios, values, ls, color=color, linewidth=2.0, label=label)
    # The identical-plate X-junction: Kij = 10 lg 12 + 5 lg(fc2/1000).
    res_eq = junction_transmission("X", h1, cl, rho * h1, h1, cl, rho * h1)
    ax.scatter([1.0], [res_eq.corner_reduction_index], color=COLOR_FG, s=70,
               zorder=6, label="identical plates (τ = 1/12)")

    ax.set_xticks(np.arange(0.5, 4.01, 0.5))
    ax.set_xticklabels([f"{v:.1f}" for v in np.arange(0.5, 4.01, 0.5)])
    ax.set_xlabel("Thickness ratio h2/h1")
    ax.set_ylabel("Vibration reduction index $K_{ij}$ [dB]")
    ax.set_title("Wave-Approach Junction $K_{ij}$ (Hopkins Eq. 5.116)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    info = [
        "Kij = 10 lg(1/τ̄) + 5 lg(fc2/1000)",
        "concrete, plate 1 fixed at 100 mm",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "junction_kij_thickness.svg")
    plt.close()


def generate_bearing_fault_envelope(output_dir: str) -> None:
    """Predicted bearing fault lines over a measured envelope spectrum."""
    print("Generating bearing_fault_envelope...")
    from phonometry import bearing_fault_frequencies, envelope_spectrum, noise_signal

    # Norton problem 8.5 geometry: fifteen rollers, 34 mm pitch diameter,
    # 6 mm rollers, 12.96 deg contact angle, 2000 r/min.
    faults = bearing_fault_frequencies(2000.0, 15, 6.0, 34.0,
                                       contact_angle_deg=12.96)
    bpfo, fs_shaft = faults["BPFO"], faults.shaft_rate

    # A spalled outer race: one impact per BPFO period ringing a 3 kHz housing
    # resonance, load-modulated once per revolution, buried in broadband noise.
    fs, seconds = 20000.0, 2.0
    t = np.arange(int(fs * seconds)) / fs
    impacts = np.zeros_like(t)
    for k in range(int(seconds * bpfo)):
        idx = round(k / bpfo * fs)
        if idx < impacts.size:
            impacts[idx] = 1.0 + 0.35 * np.cos(2.0 * np.pi * fs_shaft * idx / fs)
    tau = np.arange(int(0.004 * fs)) / fs
    ring = np.exp(-tau / 6.0e-4) * np.sin(2.0 * np.pi * 3000.0 * tau)
    x = np.convolve(impacts, ring)[: t.size] * 0.6
    x += 0.35 * np.sin(2.0 * np.pi * fs_shaft * t)          # residual unbalance
    x += noise_signal(fs, seconds, color="white", rms=0.25, seed=17)

    res = envelope_spectrum(x, fs, band=(2000.0, 4000.0))
    keep = res.frequencies <= 4.6 * bpfo
    freq, amp = res.frequencies[keep], res.amplitude[keep]

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.plot(freq, amp, color=COLOR_PRIMARY, linewidth=1.1,
            label="envelope spectrum of the 2-4 kHz band")
    top = float(amp.max())
    for order in range(1, 5):
        line = order * bpfo
        ax.axvline(line, color=COLOR_SECONDARY, linestyle="--", linewidth=1.2,
                   alpha=0.85, zorder=2,
                   label="predicted BPFO and harmonics" if order == 1 else None)
        ax.annotate(f"{order}×BPFO" if order > 1 else "BPFO",
                    xy=(line, 1.02 * top), xytext=(3, 0),
                    textcoords="offset points", rotation=90, fontsize=8.5,
                    color=COLOR_SECONDARY, ha="left", va="bottom")
    for name, colour in (("BPFI", COLOR_TERTIARY), ("BSF", "#9467bd")):
        ax.axvline(faults[name], color=colour, linestyle=":", linewidth=1.3,
                   alpha=0.9, zorder=2, label=f"predicted {name}")
    ax.axvline(fs_shaft, color=COLOR_FG, linestyle="-.", linewidth=1.0,
               alpha=0.6, zorder=2, label="shaft rate")

    ax.set_xlim(0.0, 4.6 * bpfo)
    ax.set_ylim(0.0, 1.55 * top)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Envelope amplitude")
    ax.set_title("Bearing Fault Lines on a Measured Envelope Spectrum",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        "15 rollers, D = 34 mm, d = 6 mm, φ = 12.96°, 2000 r/min",
        f"BPFO = {bpfo:.0f} Hz, BPFI = {faults['BPFI']:.0f} Hz",
        "the envelope lines fall on BPFO, not on BPFI: outer-race spall",
    ]
    ax.text(0.985, 0.47, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "bearing_fault_envelope.svg")
    plt.close()


def generate_experimental_sea_clf(output_dir: str) -> None:
    """Measured and predicted coupling loss factors of a plate junction."""
    print("Generating experimental_sea_clf...")
    from phonometry import (
        coupling_loss_factor,
        cylindrical_shell_modal_density,
        flat_plate_modal_density,
        plate_bending_stiffness,
        point_connection_coupling_loss_factor,
        power_injection_clf,
        right_angle_transmission_coefficient,
    )
    from phonometry.vibration.point_mobility import plate_bending_wave_speed

    rho, nu, young = 2700.0, 0.33, 7.1e10
    c_l = math.sqrt(young / (rho * (1.0 - nu**2)))
    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0])

    # Two aluminium plates at right angles: 3 mm x 2.5 m x 1.2 m coupled to
    # 5.5 mm x 2.0 m x 1.2 m along the 1.2 m edge (Norton problem 6.13).
    h1, h2, area1, length = 0.003, 0.0055, 2.5 * 1.2, 1.2
    tau = right_angle_transmission_coefficient(
        h1, h2, density1=rho, density2=rho, wave_speed1=c_l, wave_speed2=c_l)
    c_b = plate_bending_wave_speed(bands, plate_bending_stiffness(young, h1, nu),
                                   rho * h1)
    welded = np.array([float(coupling_loss_factor(tau, 2.0 * c, length, f, area1))
                       for c, f in zip(c_b, bands, strict=True)])
    bolted = point_connection_coupling_loss_factor(
        bands, 12, thickness1=h1, thickness2=h2, surface_density1=rho * h1,
        surface_density2=rho * h2, wave_speed1=c_l, wave_speed2=c_l,
        plate_area1=area1)

    # The satellite platform and cylinder of Norton problem 6.10, inverted from
    # its measured energies in the 500 Hz octave.
    t_p, t_c, radius, cyl_len = 0.005, 0.003, 0.75, 2.0
    area_c = 2.0 * math.pi * radius * cyl_len
    area_p = 3.5 * 3.0 - math.pi * radius**2
    sea = power_injection_clf(
        500.0,
        rho * t_p * area_p * 0.0272**2,
        rho * t_c * area_c * 0.0132**2,
        4.4e-3, 2.4e-3,
        flat_plate_modal_density(area_p, t_p, c_l),
        float(cylindrical_shell_modal_density(500.0, area_c, t_c, radius, c_l)[0]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    ax = axes[0]
    x = _band_index_axis(ax, bands, fontsize=9)
    ax.semilogy(x, welded, "-o", color=COLOR_PRIMARY, linewidth=2.0,
                markersize=5, label=r"welded line junction $\eta_{12}$")
    ax.semilogy(x, bolted, "-s", color=COLOR_SECONDARY, linewidth=2.0,
                markersize=5, label=r"12 bolts, point connections $\eta_{12}$")
    ax.axhline(1.0e-2, color=COLOR_FG, linestyle="--", linewidth=1.2,
               alpha=0.7, label=r"internal loss factor $\eta_1$")
    ax.set_ylabel("Coupling loss factor")
    ax.set_title("Predicted: Weld Against Bolts", fontweight="bold", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    labels = [r"$\eta_1$", r"$\eta_2$", r"$\eta_{12}$", r"$\eta_{21}$"]
    values = [4.4e-3, 2.4e-3, float(sea.coupling_loss_factor12[0]),
              float(sea.coupling_loss_factor21[0])]
    # COLOR_MUTED, not COLOR_GRID: this is a de-emphasised *bar*, and the
    # grid colour is tuned to disappear into the page it is drawn on.
    colours = [COLOR_FG, COLOR_MUTED, COLOR_PRIMARY, COLOR_SECONDARY]
    ax.bar(labels, values, color=colours, edgecolor=COLOR_FG, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(1.0e-4, 1.0e-2)
    ax.set_ylabel("Loss factor")
    ax.set_title("Measured: Power Injection, 500 Hz Octave",
                 fontweight="bold", pad=10)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, axis="y", which="both")
    ax.set_axisbelow(True)
    for label, value in zip(labels, values, strict=True):
        ax.annotate(f"{value:.2e}", xy=(label, value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=COLOR_FG)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        "platform driven, cylinder driven only through the joints",
        f"input power = {float(sea.input_power[0]):.2f} W",
        r"coupling stays well below the damping: valid SEA",
    ]
    ax.text(0.98, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    fig.suptitle("Coupling Loss Factors: Prediction and Power Injection",
                 fontweight="bold", fontsize=13)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    save_figure(output_dir, "experimental_sea_clf.svg")
    plt.close()


def generate_plateau_transmission_loss(output_dir: str) -> None:
    """Plateau-method TL estimate against the full physical model."""
    print("Generating plateau_transmission_loss...")
    from phonometry import (
        plateau_transmission_loss,
        single_panel_transmission_loss,
    )

    # 6 mm float glass, 2.47 kg/m2 per mm (Norton Table 3.1); its critical
    # frequency from the same book's problem 3.13 is 2033 Hz.
    thickness_mm, f_c, eta = 6.0, 2033.0, 0.02
    mass = 2.47 * thickness_mm
    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000]
    bands = np.asarray(nominal, dtype=float)
    quick = plateau_transmission_loss(bands, material="glass",
                                      thickness_mm=thickness_mm,
                                      field_correction=5.5)
    physical = single_panel_transmission_loss(bands, mass,
                                              critical_frequency=f_c,
                                              loss_factor=eta)

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    assert quick.plateau_start is not None and quick.plateau_end is not None
    idx_a = float(np.interp(np.log10(quick.plateau_start), np.log10(bands), x))
    idx_b = float(np.interp(np.log10(quick.plateau_end), np.log10(bands), x))
    ax.axvspan(idx_a, idx_b, color=theme_fill(COLOR_SECONDARY, ax), lw=0, zorder=0,
               label="coincidence plateau (A to B)")
    ax.plot(x, physical.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5,
            label="physical model (mass law + coincidence + damping)")
    ax.plot(x, quick.transmission_loss, "--s", color=COLOR_SECONDARY,
            linewidth=2.0, markersize=5, zorder=5,
            label="plateau estimate (Norton Table 3.1)")
    idx_fc = float(np.interp(np.log10(f_c), np.log10(bands), x))
    ax.axvline(idx_fc, color=COLOR_TERTIARY, linestyle=":", linewidth=1.6,
               zorder=4, label="critical frequency fc")

    ax.set_ylabel("Transmission loss TL [dB]")
    ax.set_title("Plateau Estimate Against the Physical Panel Model",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    info = [
        f"6 mm float glass, m'' = {mass:.1f} kg/m², η = {eta:g}",
        (f"plateau height 27 dB, B/A = 10 → A = {quick.plateau_start:.0f} Hz, "
         f"B = {quick.plateau_end:.0f} Hz"),
        "identical below A; the plateau replaces the whole coincidence region",
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "plateau_transmission_loss.svg")
    plt.close()


def generate_orthotropic_transmission_loss(output_dir: str) -> None:
    """Corrugating a sheet flattens its transmission loss (Vigran 6.5.3).

    One concept: the same steel sheet, flat and corrugated. The flat plate has
    a single coincidence frequency far above the audio range and follows the
    mass law; corrugating it drags the lower coincidence frequency down by more
    than a decade and opens a range over which R collapses, despite the 9 %
    extra mass the developed length adds.
    """
    print("Generating orthotropic_transmission_loss...")
    from phonometry import (
        coincidence_frequency,
        corrugated_plate_mass_factor,
        corrugated_plate_stiffness,
        orthotropic_critical_frequencies,
        orthotropic_transmission_loss,
        plate_bending_stiffness,
        single_panel_transmission_loss,
    )

    # Vigran's worked example (printed p. 96): 1 mm steel, corrugation of total
    # height 20 mm (H = 10 mm) at a 100 mm pitch, E = 2,1e11 Pa, nu = 0,3.
    modulus, thickness, poisson, eta = 2.1e11, 1.0e-3, 0.3, 0.011
    amplitude, pitch, mass_flat = 0.010, 0.100, 7.8
    flat_b = plate_bending_stiffness(modulus, thickness, poisson)
    flat_fc = coincidence_frequency(mass_flat, flat_b)
    b_x, b_z, _b_xz = corrugated_plate_stiffness(
        thickness, amplitude, pitch, youngs_modulus=modulus,
        poisson_ratio=poisson,
    )
    mass_corr = mass_flat * corrugated_plate_mass_factor(amplitude, pitch)
    fc1, fc2 = orthotropic_critical_frequencies(mass_corr, b_x, b_z)

    nominal = [*_THIRD_OCTAVE_16, 4000, 5000, 6300, 8000, 10000, 12500, 16000]
    bands = np.asarray(nominal, dtype=float)
    flat = single_panel_transmission_loss(
        bands, mass_flat, critical_frequency=flat_fc, loss_factor=eta,
    )
    corrugated = orthotropic_transmission_loss(
        bands, mass_corr, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, loss_factor=eta,
    )
    heckl = orthotropic_transmission_loss(
        bands, mass_corr, critical_frequency_lower=fc1,
        critical_frequency_upper=fc2, method="heckl",
    )

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    x = _band_index_axis(ax, nominal)
    idx_1 = float(np.interp(np.log10(fc1), np.log10(bands), x))
    idx_2 = float(np.interp(np.log10(fc2), np.log10(bands), x))
    ax.axvspan(idx_1, idx_2, color=theme_fill(COLOR_TERTIARY, ax), lw=0, zorder=0,
               label=r"coincidence range $f_{c1}$ to $f_{c2}$")
    ax.plot(x, flat.transmission_loss, "-o", color=COLOR_PRIMARY,
            linewidth=2.2, markersize=5, zorder=5,
            label=r"flat 1 mm sheet (isotropic, single $f_c$)")
    ax.plot(x, corrugated.transmission_loss, "-s", color=COLOR_SECONDARY,
            linewidth=2.2, markersize=5, zorder=5,
            label="corrugated sheet (orthotropic, diffuse-field integral)")
    ax.plot(x, heckl.transmission_loss, "--", color=COLOR_TERTIARY,
            linewidth=1.8, zorder=4, label="Heckl's approximation")

    ax.set_ylabel("Transmission loss TL [dB]")
    ax.set_title("Corrugating a Sheet Flattens Its Sound Reduction Index",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    panel = "#f0f2f5" if COLOR_FG == "black" else "#1c2128"
    worst = int(np.argmax(flat.transmission_loss - corrugated.transmission_loss))
    penalty = float(
        flat.transmission_loss[worst] - corrugated.transmission_loss[worst]
    )
    # Plain symbol names, not mathtext: the Spanish variant rewrites decimal
    # points to commas everywhere except in mathtext strings.
    info = [
        (f"1 mm steel sheet, m'' = {mass_flat:.1f} kg/m², flat "
         f"fc = {flat_fc / 1000.0:.1f} kHz"),
        (f"corrugated H = 10 mm, L = 100 mm, m'' = {mass_corr:.1f} kg/m², "
         f"fc1 = {fc1:.0f} Hz, fc2 = {fc2 / 1000.0:.1f} kHz"),
        (f"worst penalty {penalty:.0f} dB at {nominal[worst]:g} Hz, "
         "for a stiffer and only 9 % heavier panel"),
    ]
    ax.text(0.985, 0.03, "\n".join(info), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=9, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": panel,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "orthotropic_transmission_loss.svg")
    plt.close()


def generate_mobility_result_lines(output_dir: str) -> None:
    """ISO 7626 driving-point mobility with its stiffness and mass lines."""
    print("Generating mobility_result_lines...")
    from phonometry import sdof_mobility_result

    m, k, c = 2.0, 8000.0, 5.0
    f = np.logspace(np.log10(0.5), np.log10(200.0), 400)
    res = sdof_mobility_result(f, mass=m, stiffness=k, damping=c)
    w = 2.0 * np.pi * f
    f0 = float(res.frequencies[int(np.argmax(res.magnitude))])

    _fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.loglog(f, res.magnitude, color=COLOR_PRIMARY, linewidth=2.2,
              label="driving-point $|Y(f)|$")
    ax.loglog(f, w / k, ":", color=COLOR_SECONDARY, linewidth=1.6,
              label=r"stiffness line $\omega/k$")
    ax.loglog(f, 1.0 / (w * m), ":", color=COLOR_TERTIARY, linewidth=1.6,
              label=r"mass line $1/(\omega m)$")
    ax.axhline(1.0 / c, color=COLOR_GRID, linestyle="--", linewidth=1.2)
    ax.scatter([f0], [1.0 / c], color=COLOR_FG, s=60, zorder=6,
               label="peak $|Y| = 1/c$ (damping)")

    ax.set_xlim(float(f[0]), float(f[-1]))
    ax.set_ylim(1e-5, 1.0)
    format_frequency_axis(ax, float(f[0]), float(f[-1]))
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Mobility $|Y|$ [m/(N·s)]")
    ax.set_title("Reading a Driving-Point Mobility (ISO 7626-1)",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=9, ncol=2)

    info = [
        "below f0: stiffness-controlled, |Y| ~ ω/k",
        "above f0: mass-controlled, |Y| ~ 1/(ωm)",
        f"f0 = {f0:.1f} Hz,  1/c = {1.0 / c:.2f} m/(N·s)",
    ]
    ax.text(0.015, 0.97, "\n".join(info), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    plt.tight_layout()
    save_figure(output_dir, "mobility_result_lines.svg")
def generate_tone_prominence_assessment(output_dir: str) -> None:
    """ECMA-418-1 TNR of a fan tone against the clause 11.5 criterion."""
    print("Generating tone_prominence_assessment...")
    from phonometry import psychoacoustics

    # A 250 Hz fan tone recorded in broadband machinery noise: 10 s at 48 kHz,
    # the tone 15.1 dB above the masking noise of its critical band against a
    # 13.0 dB criterion, so it is prominent with about 2 dB to spare.
    fs = 48000
    rng = np.random.default_rng(4)
    t = np.arange(10 * fs) / fs
    x = (np.sqrt(2) * 0.011 * np.sin(2 * np.pi * 250.0 * t)
         + 0.03 * rng.standard_normal(t.size))
    res = psychoacoustics.tone_to_noise_ratio(x, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "tone_prominence_assessment.svg")
    plt.close()


def generate_tone_audibility_levels(output_dir: str) -> None:
    """ISO/PAS 20065 tone levels above the critical-band masking noise."""
    print("Generating tone_audibility_levels...")
    from phonometry import psychoacoustics

    # ISO/PAS 20065 Annex E combustion-engine spectrum 1 (Delta f = 2.7 Hz):
    # the levels view of the same assessment the audibility bars summarise.
    ft = [118.4, 137.3, 158.8, 314.9, 433.4, 592.2, 629.8, 643.3, 1582.7]
    lt = [64.56, 67.96, 68.63, 68.50, 73.17, 78.31, 75.00, 79.75, 71.07]
    ls = [48.91, 49.22, 50.50, 52.85, 58.29, 59.53, 59.71, 61.98, 54.16]
    res = psychoacoustics.assess_tones(ft, lt, ls, 2.7)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, view="levels", language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "tone_audibility_levels.svg")
    plt.close()


def generate_impulsive_sound_onsets(output_dir: str) -> None:
    """ISO/PAS 1996-3 LpAF history with the detected impulse onsets."""
    print("Generating impulsive_sound_onsets...")
    import warnings as _warnings

    from phonometry import environmental

    # Three hammer strikes over a 55 dB(A) background, 6 s at 48 kHz: the
    # objective chain samples LpAF, detects the onsets and rates the source.
    fs = 48000
    rng = np.random.default_rng(7)
    t = np.arange(int(6.0 * fs)) / fs
    background = rng.standard_normal(t.size)
    background *= 2e-5 * 10 ** (55 / 20) / np.sqrt(np.mean(background**2))
    signal = background.copy()
    for onset_time in (1.0, 2.6, 4.2):
        decay = np.exp(-(t - onset_time) / 0.08) * (t >= onset_time)
        strike = decay * rng.standard_normal(t.size)
        window = (t >= onset_time) & (t < onset_time + 0.1)
        strike *= 2e-5 * 10 ** (95 / 20) / np.sqrt(np.mean(strike[window] ** 2))
        signal += strike
    with _warnings.catch_warnings():
        # The synthetic interval is shorter than the assessment period.
        _warnings.simplefilter("ignore")
        res = environmental.impulsive_sound_adjustment(signal, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "impulsive_sound_onsets.svg")
    plt.close()


def generate_moore_glasberg_specific_loudness(output_dir: str) -> None:
    """ISO 532-2 specific loudness over the ERB-number (Cam) scale."""
    print("Generating moore_glasberg_specific_loudness...")
    from phonometry import psychoacoustics

    # The definitional anchor of the sone: a 1 kHz tone at 40 dB SPL, free
    # field, binaural -> N = 1 sone, with the excitation pattern spreading
    # around the tone's ERB.
    fs = 48000
    x = (np.sqrt(2) * 2e-5 * 10 ** (40 / 20)
         * np.sin(2 * np.pi * 1000 * np.arange(fs) / fs))
    res = psychoacoustics.loudness_moore_glasberg(
        x, fs, field="free", presentation="binaural")

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "moore_glasberg_specific_loudness.svg")
    plt.close()


def generate_sottek_specific_tonality(output_dir: str) -> None:
    """ECMA-418-2 average specific tonality T'(z) of a 1 kHz tone."""
    print("Generating sottek_specific_tonality...")
    from phonometry import psychoacoustics

    fs = 48000
    t = np.arange(int(1.2 * fs)) / fs
    x = np.sqrt(2) * 2e-5 * 10 ** (40 / 20) * np.sin(2 * np.pi * 1000 * t)
    res = psychoacoustics.tonality_ecma(x, fs, field="free")

    _fig, ax = plt.subplots(figsize=(10, 6))
    # A supplied axes draws the specific-tonality panel alone (the tonality
    # concentrated in the tone's critical band), not the T(l) trace.
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sottek_specific_tonality.svg")
    plt.close()


def generate_fluctuation_strength_specific(output_dir: str) -> None:
    """Specific fluctuation strength over the Bark axis (Osses 2016 model)."""
    print("Generating fluctuation_strength_specific...")
    from phonometry import psychoacoustics

    # The reference-like stimulus: a 1 kHz tone at 70 dB SPL, fully amplitude
    # modulated at 4 Hz, where the sensation peaks.
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    am = (1.0 + np.sin(2 * np.pi * 4.0 * t)) * np.sin(2 * np.pi * 1000 * t)
    am = am / np.sqrt(np.mean(am**2)) * 2e-5 * 10 ** (70 / 20)
    res = psychoacoustics.fluctuation_strength(am, float(fs))

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "fluctuation_strength_specific.svg")
    plt.close()


def generate_sii_hearing_loss(output_dir: str) -> None:
    """ANSI S3.5-1997 band audibility with a sloping hearing loss."""
    print("Generating sii_hearing_loss...")
    from phonometry import speech_intelligibility_index, standard_speech_spectrum

    # The same speech and office noise as the reference SII figure, heard by a
    # listener with a sloping high-frequency loss: the consonant-bearing bands
    # fall below the raised internal noise and the index drops from 0.46 to
    # 0.36.
    speech = standard_speech_spectrum("normal")
    noise = np.array([38.0, 37.0, 36.0, 34.0, 32.0, 30.0, 28.0, 26.0, 24.0,
                      22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0])
    threshold = np.array([5.0, 5.0, 5.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0,
                          22.0, 28.0, 35.0, 42.0, 48.0, 55.0, 60.0, 65.0, 70.0])
    res = speech_intelligibility_index(speech, noise, threshold=threshold)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sii_hearing_loss.svg")
    plt.close()


def generate_age_threshold_fractiles(output_dir: str) -> None:
    """ISO 7029 age-related threshold with its 10-90 % fractile band."""
    print("Generating age_threshold_fractiles...")
    from phonometry import hearing

    # A 70-year-old man at the worst-hearing decile: the median presbycusis
    # slope with the population spread around it.
    res = hearing.age_threshold(70, "male", fractile=0.9)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "age_threshold_fractiles.svg")
    plt.close()


def generate_nipts_audiogram(output_dir: str) -> None:
    """ISO 1999 NIPTS spectrum of a long, loud exposure with its spread."""
    print("Generating nipts_audiogram...")
    from phonometry import hearing

    # 40 years at an 8 h-normalised 95 dB(A), most-susceptible tenth: the
    # 4 kHz notch of noise damage.
    res = hearing.nipts(95.0, 40.0, fractile=0.9)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "nipts_audiogram.svg")
    plt.close()


def generate_stoi_band_scores(output_dir: str) -> None:
    """Per-band intermediate correlation behind a STOI index."""
    print("Generating stoi_band_scores...")
    from scipy import signal as sp_signal

    from phonometry import stoi

    # Speech-like material (band-limited noise with a 3.5 Hz syllabic
    # envelope) in a flat masker at 0 dB SNR: the low bands lose most of the
    # envelope correlation, the consonant bands keep it.
    fs = 10000
    rng = np.random.default_rng(11)
    t = np.arange(4 * fs) / fs
    b, a = sp_signal.butter(2, [200 / (fs / 2), 4000 / (fs / 2)], btype="band")
    carrier = sp_signal.lfilter(b, a, rng.standard_normal(t.size))
    clean = carrier * (0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 3.5 * t)) ** 2)
    masker = rng.standard_normal(clean.size)
    gain = np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(masker**2))
    res = stoi(clean, clean + gain * masker, fs)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "stoi_band_scores.svg")
    plt.close()


def generate_sti_band_mti(output_dir: str) -> None:
    """IEC 60268-16 per-band modulation transfer index of a real room."""
    print("Generating sti_band_mti...")
    from phonometry import sti_from_impulse_response

    # A reverberant hall (T60 = 0.9 s) with a 15 dB speech-to-noise ratio:
    # the per-band MTI bars behind the single STI number and its rating.
    fs = 48000
    rng = np.random.default_rng(0)
    n = np.arange(fs)
    ir = rng.standard_normal(fs) * np.exp(-6.9078 * n / fs / 0.9)
    res = sti_from_impulse_response(ir, fs, snr=15.0)

    _fig, ax = plt.subplots(figsize=(10, 6))
    res.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "sti_band_mti.svg")
def generate_atmospheric_sound_speed_profiles(output_dir: str) -> None:
    """Log-linear effective sound-speed profiles: downward vs upward refraction."""
    print("Generating atmospheric_sound_speed_profiles...")
    from phonometry import log_linear_sound_speed_profile

    down = log_linear_sound_speed_profile(+1.0, ground_speed=340.0, max_height=60.0)
    up = log_linear_sound_speed_profile(-1.0, ground_speed=340.0, max_height=60.0)
    _fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.plot(down.sound_speeds, down.heights, color=COLOR_PRIMARY, linewidth=2.0,
            label="Downward refraction (b = +1 m/s)", zorder=3)
    ax.plot(up.sound_speeds, up.heights, color=COLOR_SECONDARY, linewidth=2.0,
            label="Upward refraction (b = -1 m/s)", zorder=3)
    ax.axvline(340.0, color=COLOR_FG, linestyle=":", linewidth=0.9, alpha=0.6)
    ax.set_xlabel("Effective sound speed [m/s]")
    ax.set_ylabel("Height [m]")
    ax.set_ylim(0.0, 60.0)
    ax.set_title("Effective Sound-Speed Profiles (Salomons Eq. 4.5)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_sound_speed_profiles.svg")
    plt.close()


def generate_atmospheric_ray_fan(output_dir: str) -> None:
    """Downward-refraction ray fan: every ray returns to the ground and bounces."""
    print("Generating atmospheric_ray_fan...")
    from phonometry import atmospheric_ray_paths, log_linear_sound_speed_profile

    profile = log_linear_sound_speed_profile(+1.0, ground_speed=340.0,
                                             max_height=60.0)
    zs = 2.0
    rays = atmospheric_ray_paths(profile, source_height=zs,
                                 launch_angles_deg=np.linspace(-8.0, 8.0, 17),
                                 max_range=600.0, n_steps=600)
    _fig, ax = plt.subplots(figsize=(11.0, 5.6))
    for i in range(rays.heights.shape[0]):
        ax.plot(rays.ranges[i], rays.heights[i], color=COLOR_PRIMARY, lw=0.8,
                alpha=0.7, zorder=2)
    ax.plot([0.0], [zs], "o", color=COLOR_SECONDARY, ms=7, zorder=4,
            label="Source")
    ax.axhline(0.0, color=COLOR_FG, lw=1.0, alpha=0.7)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Height [m]")
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(0.0, 40.0)
    ax.set_title("Sound Rays under Downward Refraction (b = +1 m/s)",
                 fontweight="bold", pad=12)
    ax.text(0.985, 0.94, "shallow rays are bent back to the ground\nand bounce on down-range",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            color=COLOR_FG)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_ray_fan.svg")
    plt.close()


def generate_atmospheric_pe_range(output_dir: str) -> None:
    """GFPE relative level vs range at 2 m for three refraction conditions."""
    print("Generating atmospheric_pe_range...")
    import warnings as _warnings

    from phonometry import (
        atmospheric_parabolic_equation,
        linear_sound_speed_profile,
        log_linear_sound_speed_profile,
        shadow_zone_distance,
    )

    c0, zs, zr = 340.0, 2.0, 2.0
    cases = [
        (log_linear_sound_speed_profile(+1.0, ground_speed=c0, max_height=60.0),
         COLOR_PRIMARY, "-", "Downward (b = +1 m/s)"),
        (linear_sound_speed_profile(0.0, ground_speed=c0, max_height=60.0),
         COLOR_FG, "--", "Homogeneous (b = 0)"),
        (log_linear_sound_speed_profile(-1.0, ground_speed=c0, max_height=60.0),
         COLOR_SECONDARY, "-", "Upward (b = -1 m/s)"),
    ]
    _fig, ax = plt.subplots(figsize=(11.0, 6.2))
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        for profile, color, ls, label in cases:
            pe = atmospheric_parabolic_equation(400.0, profile, source_height=zs,
                                                flow_resistivity=200e3,
                                                max_range=600.0, max_height=40.0)
            ax.plot(pe.ranges, pe.level_at_height(zr), color=color, ls=ls,
                    lw=1.6, label=label, zorder=3)
    # The closed-form shadow boundary of the equivalent linear upward gradient
    # (the 10 m mean gradient of the log profile), as in the page-top figure.
    up_prof = cases[2][0]
    grad = (up_prof.speed_at(10.0) - c0) / 10.0
    x_sh = shadow_zone_distance(float(grad), zs, zr, ground_speed=c0)
    ax.axvline(x_sh, color=COLOR_SECONDARY, ls=":", lw=1.2, zorder=2,
               label="Shadow-zone boundary")
    ax.axhline(0.0, color=COLOR_FG, lw=0.8, alpha=0.6)
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Level re free field [dB]")
    ax.set_xlim(0.0, 600.0)
    ax.set_ylim(-40.0, 10.0)
    ax.set_title("GFPE Relative Level at the Receiver Height (400 Hz, 2 m)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "atmospheric_pe_range.svg")
    plt.close()


def generate_barrier_insertion_loss_methods(output_dir: str) -> None:
    """Wave-theoretic barrier insertion loss: Kurze-Anderson, exact, exact + ground."""
    print("Generating barrier_insertion_loss_methods...")
    import warnings as _warnings

    from phonometry import barrier_insertion_loss

    # A 4 m barrier 50 m from a 1 m source, receiver 1.5 m high at 100 m
    # (the geometry of the ground-barriers guide snippets).
    freqs = np.geomspace(50.0, 5000.0, 240)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        il_ka = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="kurze_anderson")
        il_ex = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="exact")
        il_gr = barrier_insertion_loss(freqs, 1.0, 50.0, 4.0, 100.0, 1.5,
                                       method="exact",
                                       ground_flow_resistivity=2e5)
    _fig, ax = plt.subplots(figsize=(11.0, 6.4))
    ax.plot(freqs, il_ka.insertion_loss, color=COLOR_TERTIARY, lw=1.8,
            ls="--", label="Kurze-Anderson (thin screen)", zorder=3)
    ax.plot(freqs, il_ex.insertion_loss, color=COLOR_PRIMARY, lw=1.8,
            label="Exact rigid half-plane", zorder=3)
    ax.plot(freqs, il_gr.insertion_loss, color=COLOR_SECONDARY, lw=1.4,
            alpha=0.9, label="Exact + coherent ground (four paths)", zorder=2)
    ax.axhline(5.0, color=COLOR_FG, ls=":", lw=1.0, alpha=0.7,
               label="Kurze-Anderson grazing limit (5 dB)")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Insertion loss [dB]")
    ax.set_xlim(50.0, 5000.0)
    format_frequency_axis(ax, 50.0, 5000.0)
    ax.set_title("Wave-Theoretic Barrier Insertion Loss",
                 fontweight="bold", pad=12)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "barrier_insertion_loss_methods.svg")
def generate_modulation_distortion(output_dir: str) -> None:
    """IEC 60268-3 14.12.7 modulation sidebands via ModulationDistortionResult.plot()."""
    print("Generating modulation_distortion.svg...")
    from phonometry import modulation_distortion

    # The standard two-tone test: a large low tone f1 = 60 Hz and a small high
    # tone f2 = 7 kHz (4:1), captured at the output of a weakly non-linear
    # amplifier (a memoryless polynomial stands in for the device). One second
    # at 48 kHz puts every tone on an FFT bin (coherent sampling).
    fs = 48000
    t = np.arange(fs) / fs
    x = 1.0 * np.sin(2.0 * np.pi * 60.0 * t) + 0.25 * np.sin(2.0 * np.pi * 7000.0 * t)
    y = x + 0.04 * x**2 + 0.012 * x**3

    res = modulation_distortion(y, fs, 60.0, 7000.0)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the carrier (0 dB reference) and the four
    # modulation sidebands at f2 +/- f1 and f2 +/- 2 f1, annotated with the
    # per-order d2/d3 and the SMPTE combined RMS.
    res.plot(ax=ax)
    ax.set_xlim(6600.0, 7400.0)
    plt.tight_layout()
    save_figure(output_dir, "modulation_distortion.svg")
    plt.close()


def generate_piston_radiation_impedance(output_dir: str) -> None:
    """Baffled-piston R1/X1 against ka via RadiatingPistonResult.plot()."""
    print("Generating piston_radiation_impedance.svg...")
    from phonometry import radiating_piston

    # A 75 mm-radius piston (a typical mid-woofer cone) over the audio band:
    # ka runs from well below 0.1 (mass-controlled) past 10 (resistive).
    res = radiating_piston(radius=0.075, frequencies=np.geomspace(20.0, 20000.0, 400))
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the normalized radiation resistance R1 and
    # reactance X1 against ka (the classic Beranek & Mellow figure).
    res.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "piston_radiation_impedance.svg")
    plt.close()


def generate_field_indicators(output_dir: str) -> None:
    """ISO 9614-1 F2/F3/F4 per band vs dynamic capability via FieldIndicators.plot()."""
    print("Generating field_indicators.svg...")
    from phonometry import emission

    # A 10-position discrete-point scan (ISO 9614-1) of a machine in an
    # ordinary room. The surface pressure is nearly uniform; the normal
    # intensity per band is set so the pressure-intensity gap widens towards
    # low frequency (the field turns reactive), with two inward-flowing
    # positions in the 125 Hz band (F3 rises above F2 there).
    freqs = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    delta_pi = np.array([10.5, 8.5, 6.0, 4.5, 3.5, 3.0])   # target Lp - L|In| per band
    rng = np.random.default_rng(9614)
    lp = 78.0 + rng.normal(0.0, 0.4, (10, freqs.size))
    i_mean = 10.0 ** ((78.0 - delta_pi) / 10.0) * 1.0e-12
    i_n = i_mean[None, :] * (1.0 + rng.normal(0.0, 0.18, (10, freqs.size)))
    # Two positions see net inward flow in the 125 Hz band; rescale the rest so
    # the algebraic band mean keeps its target (the band stays determinable).
    i_n[:2, 0] = -0.35 * i_mean[0]
    i_n[2:, 0] *= (10.0 * i_mean[0] - i_n[:2, 0].sum()) / i_n[2:, 0].sum()

    fi = emission.field_indicators(lp, i_n, freqs)
    ld = emission.dynamic_capability_index(18.0)   # delta_pI0 = 18 dB, K = 10 dB
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws F2 and F3 per band, the dynamic
    # capability Ld (criterion 1: adequate where Ld > F2) and F4 on a twin axis.
    fi.plot(ax=ax, dynamic_capability=ld)
    plt.tight_layout()
    save_figure(output_dir, "field_indicators.svg")
    plt.close()


def generate_intensity_class(output_dir: str) -> None:
    """Measured delta_pI0 of a p-p chain over the IEC 61043 Table 2 masks."""
    print("Generating intensity_class.svg...")
    from phonometry import metrology

    # A complete instrument verified with the common 12 mm spacer, so the
    # printed 25 mm minima come down by 10 lg(12/25) = -3,2 dB (Table 2,
    # Note 1). The measured index is modelled from the physics behind the
    # table: a residual channel phase mismatch phi_s turns into
    # delta_pI0 = 10 lg(kd/phi_s), so a mismatch that is constant in degrees
    # already buys 10 dB per decade of index - which is exactly the slope of
    # the Table 2 requirement below 250 Hz. Above 1 kHz the mismatch of a real
    # chain grows with frequency instead of staying put, so the index levels
    # off where the requirement does.
    spacing = 0.012
    freqs, _, _ = metrology.residual_index_limits("instrument", spacing=spacing)
    phase_mismatch = 0.05 * np.maximum(1.0, freqs / 1000.0)   # degrees
    measured = metrology.residual_index_from_phase_mismatch(
        phase_mismatch, freqs, spacing
    )
    # A vent resonance of the capsules costs 4 dB around 100 Hz, the one band
    # that drops out of class 1 and drags the whole verdict down to class 2.
    measured = measured - 4.0 * np.exp(-(((np.log(freqs / 100.0)) / 0.25) ** 2))

    result = metrology.intensity_class_compliance(measured, freqs, spacing=spacing)
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() shades the pass region of the achieved class and
    # rings the bands that cost the chain the next class up.
    result.plot(ax=ax, language=_LANG)
    plt.tight_layout()
    save_figure(output_dir, "intensity_class.svg")
    plt.close()


def generate_silencer_side_branch(output_dir: str) -> None:
    """Helmholtz and quarter-wave side branches shorting the duct at tuning."""
    print("Generating silencer_side_branch.svg...")
    from phonometry import helmholtz_resonator, quarter_wave_resonator

    freqs = np.linspace(20.0, 600.0, 4000)
    hr = helmholtz_resonator(freqs, duct_area=0.01, neck_area=1e-4,
                             neck_length=0.02, cavity_volume=1e-3)
    qw = quarter_wave_resonator(freqs, duct_area=0.01, length=0.3,
                                branch_area=2e-3)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, hr.transmission_loss, color=COLOR_PRIMARY, lw=1.8,
            label="Helmholtz resonator")
    ax.plot(freqs, qw.transmission_loss, color=COLOR_SECONDARY, lw=1.8,
            ls="--", label="Quarter-wave tube")
    for resonances in (hr.resonances, qw.resonances):
        if resonances is not None:
            ax.axvline(float(resonances[0]), color=COLOR_TERTIARY, ls=":", lw=1.0)
    ax.set_xlim(20.0, 600.0)
    ax.set_ylim(0.0, 50.0)
    format_frequency_axis(ax, 20.0, 600.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("Side-branch resonators: transmission loss (Bies Eqs. 8.44, 8.46)",
                 fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    save_figure(output_dir, "silencer_side_branch.svg")
    plt.close()


def generate_hvac_end_reflection(output_dir: str) -> None:
    """ASHRAE duct end reflection loss for three diameters (HvacSpectrumResult)."""
    print("Generating hvac_end_reflection.svg...")
    from phonometry.noise_control import hvac

    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
    _fig, ax = plt.subplots(figsize=(10, 6))
    for diameter, color in ((0.15, COLOR_PRIMARY), (0.30, COLOR_SECONDARY),
                            (0.60, COLOR_TERTIARY)):
        er = hvac.end_reflection_loss(bands, diameter=diameter,
                                      termination="flush")
        ax.semilogx(np.asarray(er.frequencies), np.asarray(er.values), "o-",
                    color=color, lw=1.8, ms=4,
                    label=f"D = {int(diameter * 1000)} mm")
    ax.set_xlim(50.0, 2500.0)
    ax.set_ylim(bottom=0.0)
    format_frequency_axis(ax, 50.0, 2500.0)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("End reflection loss [dB]")
    ax.set_title("Duct end reflection loss (ASHRAE Table 8.14)",
                 fontweight="bold", pad=10)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(loc="upper right", fontsize="small", title="Duct diameter")
    plt.tight_layout()
    save_figure(output_dir, "hvac_end_reflection.svg")
    plt.close()


def _long_duct_paths() -> tuple[Any, Any]:
    """The supply and return paths of Long's worked HVAC sheet (Table 14.9).

    Every row is the one printed in the sheet, including the manufacturer data
    for the silencers and the terminal devices, so the figure shows what the
    published calculation delivers into the room rather than a re-derivation.
    """
    from phonometry import DuctElement, duct_path

    bands = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
    fan = [90.0, 86.0, 82.0, 79.0, 77.0, 75.0, 71.0, 61.0]
    source = "Fan, centrifugal FC, 5000 cfm, 2 in w.g."
    supply = duct_path(
        bands, fan,
        [
            DuctElement("Elbow, 36 x 24 in, unlined",
                        [0, 1, 2, 3, 3, 3, 3, 3],
                        [41, 39, 36, 29, 20, 6, 0, 0], code="2"),
            DuctElement("Silencer, 3 ft, standard pressure drop",
                        [7, 12, 16, 28, 35, 35, 28, 17],
                        [49, 43, 44, 42, 42, 45, 35, 24], code="3"),
            DuctElement("Duct, 36 x 24 in, 5 ft, 1 in lining",
                        [2, 2, 3, 7, 15, 12, 11, 9], code="4"),
            DuctElement("Split, 25 per cent", 6.0, code="5"),
            DuctElement("Duct, 18 x 12 in, 6 ft, 1 in lining",
                        [3, 3, 5, 11, 25, 22, 16, 13], code="6"),
            DuctElement("Flexible duct, 12 in, 6 ft",
                        [14, 14, 16, 15, 17, 22, 16, 13], code="7"),
            DuctElement("Rectangular diffuser, 312 cfm", None,
                        [33, 32, 29, 23, 15, 4, 0, 0], code="8"),
        ],
        room_effect=[6, 6, 5, 5, 6, 7, 6, 6],
        source_label=source, target=30.0, label="Supply",
    )
    ret = duct_path(
        bands, fan,
        [
            DuctElement("Elbow, 36 x 24 in, unlined",
                        [0, 1, 2, 3, 3, 3, 3, 3],
                        [43, 42, 39, 33, 24, 12, 0, 0], code="2"),
            DuctElement("Silencer, 5 ft, low-frequency type",
                        [16, 21, 35, 41, 41, 28, 21, 15],
                        [51, 49, 53, 56, 56, 59, 60, 53], code="3"),
            DuctElement("Elbow, 36 x 24 in, lined, 1 in",
                        [1, 2, 3, 4, 5, 6, 8, 10],
                        [39, 38, 34, 28, 18, 4, 0, 0], code="4"),
            DuctElement("Plenum, 800 sq ft, 50 per cent lined",
                        [12, 13, 19, 20, 20, 20, 21, 21], code="5"),
            DuctElement("Rectangular grille, 24 x 24 in, 563 cfm", None,
                        [30, 29, 26, 20, 12, 1, 0, 0], code="6"),
        ],
        room_effect=[9, 8, 6, 8, 8, 8, 9, 10],
        source_label=source, target=30.0, label="Return",
    )
    return supply, ret


def generate_duct_path_cascade(output_dir: str) -> None:
    """Long Table 14.9 supply + return duct paths against NC 30."""
    print("Generating duct_path_cascade.svg...")
    from phonometry import combine_duct_paths

    supply, ret = _long_duct_paths()
    total = combine_duct_paths([supply, ret], label="Supply + return")

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws each contributing path, the received
    # spectrum and the design criterion curve.
    total.plot(ax=ax, language=_LANG)
    ax.set_ylim(-6.0, 62.0)
    ax.set_title("Duct-borne noise into the room: supply, return and NC 30",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    save_figure(output_dir, "duct_path_cascade.svg")
    plt.close()


def _norton_plant_room_chain() -> Any:
    """Norton problem 4.18: a blower in a plant room, into the operator room.

    Every number is the one printed in the problem statement, so the figure
    shows the published calculation rather than a re-derivation: the blower
    sound power level, the ceiling, floor and wall absorption of the
    8 x 10 x 3 m plant room, the transmission loss of the separating wall and
    the carpeted floor of the 5 x 5 x 3 m operator room.
    """
    from phonometry import (
        equivalent_absorption_area,
        mean_absorption,
        room_constant,
        room_to_room_transmission,
    )

    bands = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
    ceiling = [0.07, 0.20, 0.40, 0.52, 0.60, 0.67]
    walls = [0.03, 0.03, 0.03, 0.04, 0.05, 0.07]
    plant = [
        (80.0, [0.01, 0.01, 0.015, 0.02, 0.02, 0.02]),
        (80.0, ceiling),
        (108.0, walls),
    ]
    operator = [
        (25.0, [0.08, 0.24, 0.57, 0.69, 0.71, 0.73]),
        (25.0, ceiling),
        (60.0, walls),
    ]
    return room_to_room_transmission(
        bands,
        [39.0, 42.0, 50.0, 58.0, 63.0, 67.0],
        5.0 * 3.0,
        equivalent_absorption_area(operator),
        source_power_level=[105.0, 103.0, 98.0, 108.0, 107.0, 109.0],
        source_room_constant=room_constant(268.0, mean_absorption(plant)),
        # The blower sits on the floor along the middle of a wall (Q = 4) and
        # the problem asks for a conservative estimate, i.e. the constant-volume
        # sound power model of Norton Table 4.5.
        source_directivity=4.0,
        source_model="constant_volume",
        target=45.0,
        label="Plant room to operator room",
    )


def generate_room_to_room_chain(output_dir: str) -> None:
    """Norton problem 4.18: plant room to operator room against NC 45."""
    print("Generating room_to_room_chain.svg...")
    result = _norton_plant_room_chain()

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws both reverberant spectra, the criterion
    # curve and the band-by-band noise reduction on the twin axis.
    result.plot(ax=ax, language=_LANG)
    ax.set_ylim(20.0, 115.0)
    ax.set_title(
        "Plant room to operator room: what the wall delivers, and NC 45",
        fontweight="bold", pad=10,
    )
    plt.tight_layout()
    save_figure(output_dir, "room_to_room_chain.svg")
    plt.close()


def generate_duct_mode_cut_on(output_dir: str) -> None:
    """Higher-order-mode cut-on ladder of a 254 mm steam line at 200 m/s."""
    print("Generating duct_mode_cut_on.svg...")
    from phonometry import circular_duct_cut_on

    # Norton & Karczub problem 7.1: a 254 mm circular duct carrying steam
    # (c = 405 m/s) at a mean flow velocity of 200 m/s, i.e. M = 0.494, where
    # the sqrt(1 - M^2) shift separates the two ladders visibly.
    modes = circular_duct_cut_on(0.254, flow_velocity=200.0,
                                 speed_of_sound=405.0, count=6)

    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the still-air ladder, the with-flow
    # ladder and the plane-wave band below the first cut-on.
    modes.plot(ax=ax, language=_LANG)
    ax.set_title("Duct higher-order-mode cut-on: 254 mm steam line at 200 m/s",
                 fontweight="bold", pad=10)
    plt.tight_layout()
    save_figure(output_dir, "duct_mode_cut_on.svg")
    plt.close()


def generate_enclosure_insertion_loss(output_dir: str) -> None:
    """Machine-enclosure IL = R - C per band via EnclosureResult.plot()."""
    print("Generating enclosure_insertion_loss.svg...")
    from phonometry import enclosure_insertion_loss

    # A measured panel transmission loss combined with a modest interior
    # lining (mean absorption 0.3): the reverberant build-up inside the small
    # hard cavity spends part of the panel R.
    bands = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])
    panel_r = np.array([18.0, 24.0, 30.0, 36.0, 42.0, 46.0])
    enc = enclosure_insertion_loss(
        panel_r, external_area=6.0, internal_area=5.0,
        internal_absorption=0.3, frequencies=bands,
    )
    _fig, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the panel R, the interior correction C
    # and the net insertion loss IL = R - C.
    enc.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "enclosure_insertion_loss.svg")
    plt.close()


def generate_phase_decomposition(output_dir: str) -> None:
    """Minimum-phase / all-pass split of a delayed band-pass response."""
    print("Generating phase_decomposition.svg...")
    from scipy import signal as sp_signal

    from phonometry import phase_decomposition

    # A strictly minimum-phase equalizer response (an RBJ +6 dB peaking biquad
    # at 1 kHz) measured through a 2.5 ms latency (a digital processing
    # delay): the minimum-phase part carries what an equalizer can invert, the
    # all-pass excess carries the pure delay, read directly off the flat
    # excess group delay.
    fs = 48000.0
    delay = int(0.0025 * fs)                      # 2.5 ms
    gain_a = 10.0 ** (6.0 / 40.0)                 # +6 dB peaking gain
    w0 = 2.0 * np.pi * 1000.0 / fs
    alpha = np.sin(w0) / (2.0 * 1.0)              # Q = 1
    b = np.array([1.0 + alpha * gain_a, -2.0 * np.cos(w0), 1.0 - alpha * gain_a])
    a = np.array([1.0 + alpha / gain_a, -2.0 * np.cos(w0), 1.0 - alpha / gain_a])
    imp = np.zeros(16384)
    imp[delay] = 1.0
    ir = sp_signal.lfilter(b / a[0], a / a[0], imp)

    res = phase_decomposition(np.fft.rfft(ir), fs)
    # The result's own .plot() draws three stacked panels: |H|, the measured /
    # minimum / excess phases and the total and excess group delays.
    res.plot()
    plt.gcf().set_size_inches(9.0, 8.0)
    plt.tight_layout()
    save_figure(output_dir, "phase_decomposition.svg")
    plt.close()


def _r128_noise_programme(sections: list[tuple[float, float]], fs: int) -> np.ndarray:
    """Deterministic band-limited noise programme from (level dBFS, seconds)."""
    from scipy import signal as sp_signal

    rng = np.random.default_rng(3341)
    sos = sp_signal.butter(2, 2000.0, fs=fs, output="sos")
    chunks = []
    for level, seconds in sections:
        noise = sp_signal.sosfilt(sos, rng.standard_normal(int(seconds * fs)))
        noise /= np.sqrt(np.mean(noise**2))
        chunks.append(10.0 ** (level / 20.0) * noise)
    return np.concatenate(chunks)


def generate_loudness_gating(output_dir: str) -> None:
    """BS.1770 gating: a long quiet tail does not drag the integrated loudness."""
    print("Generating loudness_gating.svg...")
    from phonometry import program_loudness

    # 20 s of programme material on the -23 LUFS target followed by 40 s of
    # quiet room ambience ~29 LU lower: the -70 LUFS absolute gate keeps
    # everything, but the relative gate (10 LU below the survivors) drops the
    # tail, so the integrated loudness stays on the foreground while the
    # ungated mean sinks. The programme is loudness-normalised to -23.0 LUFS
    # first, exactly as a delivery workflow would.
    fs = 48000
    from phonometry import integrated_loudness

    x = _r128_noise_programme([(-23.0, 20.0), (-52.0, 40.0)], fs)
    x *= 10.0 ** ((-23.0 - integrated_loudness(np.vstack([x, x]), fs)) / 20.0)
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # The result's own .plot() draws the momentary and short-term traces, the
    # integrated line and the LRA band.
    res.plot(ax=ax)
    finite = res.momentary[np.isfinite(res.momentary)]
    ungated = 10.0 * np.log10(np.mean(10.0 ** (finite / 10.0)))
    ax.axhline(ungated, color=COLOR_TERTIARY, ls="-.", lw=1.6,
               label=f"Ungated mean {ungated:.1f} LUFS")
    ax.legend(loc="center right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "loudness_gating.svg")
    plt.close()


def generate_loudness_range(output_dir: str) -> None:
    """EBU Tech 3342 loudness range on its two-step reference case (LRA = 10 LU)."""
    print("Generating loudness_range.svg...")
    from phonometry import program_loudness

    # EBU Tech 3342 test case 1: 20 s at -20 dBFS then 20 s at -30 dBFS. The
    # short-term distribution has two plateaus 10 LU apart, and the P10-P95
    # spread reads exactly LRA = 10.0 LU (the shaded band of the plot).
    fs = 48000
    t = np.arange(20 * fs) / fs
    tone = np.sin(2.0 * np.pi * 1000.0 * t)
    x = np.concatenate([10.0 ** (-20.0 / 20.0) * tone,
                        10.0 ** (-30.0 / 20.0) * tone])
    res = program_loudness(np.vstack([x, x]), fs)

    _fig, ax = plt.subplots(figsize=(10.5, 5.8))
    # The result's own .plot() shades the loudness range between its 10th and
    # 95th percentile edges under the momentary / short-term / integrated traces.
    res.plot(ax=ax)
    plt.tight_layout()
    save_figure(output_dir, "loudness_range.svg")
def generate_resampling_antialias(output_dir: str) -> None:
    """Polyphase resampling: the delivered anti-alias filter vs its spec."""
    print("Generating resampling_antialias...")
    from phonometry import noise_signal, resample_signal

    x = noise_signal(44100, 5.0, color="pink", seed=1)
    res = resample_signal(x, 44100, 48000)      # 120 dB alias rejection

    fs_up = res.original_fs * res.up
    freqs, h = scipy_signal.freqz(res.filter_taps, worN=1 << 18, fs=fs_up)
    tiny = np.finfo(np.float64).tiny
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), tiny))
    f_lo, f_hi = res.stopband_edge_hz / 8.0, 4.0 * res.stopband_edge_hz
    view = (freqs > 0.0) & (freqs <= f_hi)

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(freqs[view], mag_db[view], color=COLOR_PRIMARY, linewidth=1.2,
                label="Anti-alias filter $|H(f)|$")
    ax.axvline(res.passband_edge_hz, color=COLOR_TERTIARY, linestyle="--",
               linewidth=1.4, label="Passband edge")
    ax.axvline(res.stopband_edge_hz, color=COLOR_SECONDARY, linestyle="--",
               linewidth=1.4, label="Stopband edge (alias fold)")
    ax.axhline(-res.stopband_attenuation_db, color=COLOR_FG, linestyle=":",
               linewidth=1.2, alpha=0.7, label="Design attenuation -120 dB")
    ax.axvspan(res.stopband_edge_hz, f_hi, color=theme_fill(COLOR_SECONDARY, ax),
               zorder=0,
               label="Rejected band (would fold back as aliases)")
    ax.set_xlim(f_lo, f_hi)
    ax.set_ylim(-170.0, 10.0)
    format_frequency_axis(ax, f_lo, f_hi)
    ax.set_xlabel(LABEL_FREQ_HZ)
    ax.set_ylabel("Magnitude [dB]")
    ax.set_title("Polyphase Resampling 44.1 kHz → 48 kHz: "
                 "the Delivered Anti-Alias Filter", fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "resampling_antialias.svg")
    plt.close()


def generate_cepstrum_variants(output_dir: str) -> None:
    """Power, real and complex cepstra of one echo-carrying record."""
    print("Generating cepstrum_variants...")
    from phonometry import cepstrum

    fs = 48000.0
    # A band-limited source wavelet (its cepstrum concentrates below 1 ms)
    # plus one reflection at 8 ms with reflection coefficient a = 0.5.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # echo: 8 ms, a = 0.5

    _fig, ax = plt.subplots(figsize=(10, 6))
    axins = ax.inset_axes((0.60, 0.28, 0.26, 0.42))
    variants = (
        ("power", COLOR_PRIMARY, "-", "Power cepstrum"),
        ("real", COLOR_TERTIARY, "--", "Real cepstrum (exactly half the power)"),
        ("complex", COLOR_SECONDARY, ":", "Complex cepstrum"),
    )
    for kind, colour, style, label in variants:
        res = cepstrum(x, fs, kind=kind)
        q_ms = 1e3 * res.quefrencies
        mask = (q_ms > 0.5) & (q_ms <= 20.0)
        ax.plot(q_ms[mask], res.cepstrum[mask], color=colour, linestyle=style,
                linewidth=1.1, label=label)
        zoom = (q_ms > 7.8) & (q_ms <= 8.2)
        axins.plot(q_ms[zoom], res.cepstrum[zoom], color=colour,
                   linestyle=style, linewidth=1.3, marker="o", markersize=3)
    axins.set_xlim(7.85, 8.15)
    axins.set_ylim(-0.05, 0.55)
    axins.tick_params(labelsize=7)
    axins.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor=COLOR_FG, alpha=0.5)
    ax.annotate("first rahmonic at 8 ms:\nheight ≈ a on the power cepstrum",
                xy=(8.0, 0.5), xytext=(2.5, 0.42), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.annotate("second rahmonic: $-a^2/2$",
                xy=(16.0, -0.125), xytext=(12.0, -0.22), fontsize=9,
                color=COLOR_FG,
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": COLOR_FG})
    ax.set_xlabel("Quefrency [ms]")
    ax.set_ylabel("Cepstrum")
    ax.set_xlim(0.5, 20.0)
    ax.set_ylim(-0.3, 0.6)
    ax.set_title("The Three Cepstrum Variants of One Echo-Carrying Record",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cepstrum_variants.svg")
    plt.close()


def generate_lifter_split(output_dir: str) -> None:
    """Lowpass/highpass liftering of a log spectrum with an 8 ms echo."""
    print("Generating lifter_split...")
    from phonometry import lifter

    fs = 48000.0
    # The same wavelet-plus-echo record as the cepstrum-variants figure:
    # a smooth source envelope carrying a pure 8 ms echo ripple (a = 0.5),
    # so the highpass ripple swings exactly between the closed forms.
    b, a_coef = scipy_signal.butter(2, 0.3)
    s = np.zeros(4096)
    s[37: 37 + 256] = scipy_signal.lfilter(
        b, a_coef, np.r_[1.0, np.zeros(255)]
    )
    x = s + 0.5 * np.roll(s, 384)                # the same 8 ms echo

    low = lifter(x, fs, cutoff=0.004, mode="lowpass")
    high = lifter(x, fs, cutoff=0.004, mode="highpass")
    band = (low.frequencies >= 500.0) & (low.frequencies <= 2000.0)

    _fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogx(low.frequencies[band], low.spectrum_db[band],
                     color="#7f7f7f", linewidth=0.7, alpha=0.8,
                     label="Log spectrum of the record")
    axes[0].semilogx(low.frequencies[band], low.liftered_db[band],
                     color=COLOR_PRIMARY, linewidth=2.0,
                     label="Lowpass lifter: spectral envelope")
    axes[0].set_ylabel("Magnitude [dB]")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].semilogx(high.frequencies[band], high.liftered_db[band],
                     color=COLOR_SECONDARY, linewidth=1.1,
                     label="Highpass lifter: the echo ripple alone")
    for bound in (20.0 * np.log10(1.5), 20.0 * np.log10(0.5)):
        axes[1].axhline(bound, color=COLOR_TERTIARY, linestyle="--",
                        linewidth=1.2,
                        label=(r"closed-form ripple bounds $20\lg(1\pm a)$"
                               if bound > 0 else None))
    axes[1].set_ylabel("Magnitude [dB]")
    axes[1].set_xlabel(LABEL_FREQ_HZ)
    axes[1].legend(loc="upper right", fontsize=9)

    for ax in axes:
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 500.0, 2000.0)
    axes[0].set_title("Liftering at 4 ms: Envelope Versus Echo Ripple",
                      fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "lifter_split.svg")
    plt.close()


def generate_correlation_normalizations(output_dir: str) -> None:
    """The three correlation normalizations of a two-sensor delay model."""
    print("Generating correlation_normalizations...")
    from phonometry import correlation, noise_signal

    fs = 8192.0
    delay = 102                                   # 12.45 ms
    x = noise_signal(fs, 2.0, seed=4)
    interference = noise_signal(fs, 2.0, rms=0.5, seed=5)
    y = 0.8 * np.concatenate([np.zeros(delay), x[:-delay]]) + interference

    coeff = correlation(x, y, fs, normalization="coefficient", max_lag=0.05)
    biased = correlation(x, y, fs, normalization="biased")
    unbiased = correlation(x, y, fs, normalization="unbiased")

    _fig, (ax_c, ax_n) = plt.subplots(2, 1, figsize=(10, 7))
    ax_c.plot(1e3 * coeff.lags, coeff.values, color=COLOR_PRIMARY,
              linewidth=1.1,
              label=r"Coefficient $\rho_{xy}(\tau)$ (bounded by $\pm 1$)")
    ax_c.axvline(1e3 * delay / fs, color=COLOR_SECONDARY, linestyle="--",
                 linewidth=1.2, label="true delay +12.5 ms")
    ax_c.set_xlabel("Lag [ms]")
    ax_c.set_ylabel("Correlation")
    ax_c.set_ylim(-1.05, 1.05)
    ax_c.legend(loc="upper left", fontsize=9)

    ax_n.plot(unbiased.lags, unbiased.values, color=COLOR_SECONDARY,
              linewidth=0.5, alpha=0.9,
              label=r"Unbiased $1/(N-|r|)$ (variance grows at the ends)")
    ax_n.plot(biased.lags, biased.values, color=COLOR_PRIMARY,
              linewidth=0.5,
              label=r"Biased $1/N$ (tapers toward the ends)")
    ax_n.set_xlabel("Lag [s]")
    ax_n.set_ylabel("Correlation")
    ax_n.legend(loc="upper left", fontsize=9)

    for ax in (ax_c, ax_n):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_c.set_title("Correlation Normalizations of a Two-Sensor Delay Model",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "correlation_normalizations.svg")
    plt.close()


def generate_ir_alignment(output_dir: str) -> None:
    """Sub-sample alignment of an impulse response onto a reference."""
    print("Generating ir_alignment...")
    from phonometry import align_impulse_responses, fractional_delay

    fs = 48000.0
    t = np.arange(int(0.03 * fs)) / fs
    rng = np.random.default_rng(6)
    # Band-limited reference pulse: a 2 kHz Gaussian tone burst at 5 ms.
    ir_a = scipy_signal.gausspulse(t - 0.005, fc=2000.0, bw=0.5)
    ir_b = fractional_delay(ir_a, 7.37)[: ir_a.size]
    ir_b += 0.005 * rng.standard_normal(ir_a.size)

    res = align_impulse_responses(ir_b, ir_a, fs)
    t_ms = 1e3 * t

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms, res.reference, color=COLOR_PRIMARY, linewidth=1.6,
            label="Reference IR")
    ax.plot(t_ms, ir_b, color="#7f7f7f", linewidth=1.0, linestyle="--",
            alpha=0.8, label="Measured IR (delayed 7.37 samples)")
    ax.plot(t_ms, res.aligned[: t.size], color=COLOR_SECONDARY,
            linewidth=1.0, linestyle=":", label="Aligned IR (delay removed)")
    ax.text(0.985, 0.05,
            f"estimated delay removed: {res.delay_samples:.2f} samples",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG)
    ax.set_xlim(2.0, 9.0)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sub-Sample Impulse-Response Alignment",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "ir_alignment.svg")
    plt.close()


def generate_hilbert_envelope(output_dir: str) -> None:
    """Hilbert envelope and instantaneous frequency of a decaying mode."""
    print("Generating hilbert_envelope...")
    from phonometry import envelope

    fs = 8192.0
    t = np.arange(int(0.4 * fs)) / fs
    rng = np.random.default_rng(7)
    decay = np.exp(-t / 0.1)                     # a struck 250 Hz mode
    x = decay * np.sin(2.0 * np.pi * 250.0 * t)
    x += 0.001 * rng.standard_normal(t.size)

    # The envelope of a narrowband signal is low-frequency: the anti-aliased
    # x8 decimation keeps the outputs compact without losing the decay.
    res = envelope(x, fs, decimation_factor=8)

    _fig, (ax_e, ax_f) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_e.plot(t, res.signal, color=COLOR_PRIMARY, linewidth=0.6,
              label="Signal")
    ax_e.plot(res.times, res.envelope, color=COLOR_SECONDARY, linewidth=1.8,
              label="Envelope $A(t)$")
    ax_e.plot(res.times, -res.envelope, color=COLOR_SECONDARY, linewidth=1.8)
    ax_e.set_ylabel("Amplitude")
    ax_e.legend(loc="upper right", fontsize=9)

    ax_f.plot(res.times, res.instantaneous_frequency, color=COLOR_PRIMARY,
              linewidth=0.9, label="Instantaneous frequency $f(t)$")
    ax_f.axhline(250.0, color=COLOR_TERTIARY, linestyle="--", linewidth=1.4,
                 label="carrier 250 Hz")
    ax_f.set_ylim(230.0, 270.0)
    ax_f.set_ylabel("Instantaneous frequency [Hz]")
    ax_f.set_xlabel("Time [s]")
    ax_f.legend(loc="upper right", fontsize=9)

    for ax in (ax_e, ax_f):
        ax.set_xlim(0.0, 0.3)
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
    ax_e.set_title("Hilbert Envelope and Instantaneous Frequency",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "hilbert_envelope.svg")
    plt.close()


def generate_cross_spectral_density_delay(output_dir: str) -> None:
    """Cross-spectral density of a delay path: magnitude and linear phase."""
    print("Generating cross_spectral_density...")
    from phonometry import cross_spectral_density, noise_signal

    fs = 8000.0
    tau = 0.002                                   # 2 ms = 16 samples
    delay = int(tau * fs)
    x = noise_signal(fs, 8.0, seed=8)
    noise = noise_signal(fs, 8.0, rms=0.3, seed=9)
    y = 0.9 * np.concatenate([np.zeros(delay), x[:-delay]]) + noise

    res = cross_spectral_density(x, y, fs)
    tiny = np.finfo(np.float64).tiny
    band = (res.frequencies >= 20.0) & (res.frequencies <= 3500.0)
    freqs = res.frequencies[band]

    _fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_m.semilogx(freqs, 10.0 * np.log10(np.maximum(res.magnitude[band], tiny)),
                  color=COLOR_PRIMARY, linewidth=1.1,
                  label=r"$|\hat{G}_{xy}|$ (Welch estimate)")
    ax_m.set_ylabel("Magnitude [dB]")
    ax_m.legend(loc="lower left", fontsize=9)

    ax_p.semilogx(freqs, res.phase[band], color=COLOR_PRIMARY, linewidth=1.1,
                  label="Unwrapped phase")
    ax_p.fill_between(freqs, res.phase[band] - res.phase_std[band],
                      res.phase[band] + res.phase_std[band],
                      color=COLOR_PRIMARY, alpha=0.25,
                      label=r"$\pm 1$ s.d. band (Eq. 9.52)")
    ax_p.semilogx(freqs, -2.0 * np.pi * freqs * tau, color=COLOR_SECONDARY,
                  linestyle="--", linewidth=1.4,
                  label=r"slope $-2\pi f\tau$ ($\tau$ = 2 ms)")
    ax_p.set_ylabel("Phase [rad]")
    ax_p.set_xlabel(LABEL_FREQ_HZ)
    ax_p.legend(loc="lower left", fontsize=9)

    for ax in (ax_m, ax_p):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 3500.0)
    ax_m.set_title("Cross-Spectral Density of a 2 ms Delay Path",
                   fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "cross_spectral_density.svg")
    plt.close()


def generate_coherent_output_snr(output_dir: str) -> None:
    """Coherent output spectrum split and the spectral SNR."""
    print("Generating coherent_output_snr...")
    from phonometry import coherent_output_spectrum, noise_signal

    fs = 48000.0
    x = noise_signal(fs, 8.0, color="white", seed=1)
    noise = noise_signal(fs, 8.0, color="white", rms=0.5, seed=2)
    y = 0.8 * x + noise                      # SNR = 0.64/0.25 per band

    res = coherent_output_spectrum(x, y, fs, nperseg=2048)
    tiny = np.finfo(np.float64).tiny
    band = np.flatnonzero(
        (res.frequencies >= 20.0) & (res.frequencies <= 20000.0)
    )
    # Thin the flat spectra to ~1000 log-spaced bins: on a log axis the
    # linear Welch grid bunches thousands of vertices at the right edge
    # without adding visual information (and bloats the SVG).
    band = band[np.unique(np.geomspace(1, band.size, 1000).astype(int) - 1)]
    freqs = res.frequencies[band]

    _fig, (ax_g, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for values, colour, style, label in (
        (res.output_psd, COLOR_PRIMARY, "-",
         r"$\hat{G}_{yy}$ (measured output)"),
        (res.coherent_psd, COLOR_TERTIARY, "--",
         r"$\hat{G}_{vv} = \gamma^2\hat{G}_{yy}$ (coherent part)"),
        (res.noise_psd, COLOR_SECONDARY, ":",
         r"$\hat{G}_{nn}$ (uncorrelated noise)"),
    ):
        ax_g.semilogx(freqs,
                      10.0 * np.log10(np.maximum(values[band], tiny)),
                      color=colour, linestyle=style, linewidth=1.1,
                      label=label)
    ax_g.set_ylabel("Spectral density [dB re 1/Hz]")
    ax_g.legend(loc="lower left", fontsize=9)

    ax_s.semilogx(freqs, res.snr_db[band], color=COLOR_PRIMARY,
                  linewidth=1.1, label="Spectral SNR [dB]")
    ax_s.axhline(10.0 * np.log10(0.64 / 0.25), color=COLOR_SECONDARY,
                 linestyle="--", linewidth=1.4,
                 label=r"closed form $10\lg(2.56)$ = 4.1 dB")
    ax_s.set_ylabel("Spectral SNR [dB]")
    ax_s.set_xlabel(LABEL_FREQ_HZ)
    ax_s.legend(loc="lower left", fontsize=9)

    for ax in (ax_g, ax_s):
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
        ax.set_axisbelow(True)
        format_frequency_axis(ax, 20.0, 20000.0)
    ax_g.set_title(
        "Coherent Output Spectrum and Spectral SNR (Bendat & Piersol 9.2.2)",
        fontweight="bold", pad=12)
    plt.tight_layout()
    save_figure(output_dir, "coherent_output_snr.svg")
    plt.close()


def generate_golay_ir(output_dir: str) -> None:
    """Golay-pair impulse response: exact complementary recovery."""
    print("Generating golay_ir...")
    from phonometry import golay_impulse_response, golay_pair

    fs = 48000
    pair = golay_pair(14)                        # two 16384-sample codes
    b, a = scipy_signal.butter(2, [200.0, 2000.0], btype="bandpass", fs=fs)
    length = pair[0].size
    rec_a = scipy_signal.lfilter(b, a, np.tile(pair[0], 3))[2 * length:]
    rec_b = scipy_signal.lfilter(b, a, np.tile(pair[1], 3))[2 * length:]

    ir = np.asarray(golay_impulse_response(rec_a, rec_b, pair, fs=fs))
    impulse = np.zeros(length)
    impulse[0] = 1.0
    true_ir = scipy_signal.lfilter(b, a, impulse)
    err = float(np.max(np.abs(ir - true_ir)))

    t_ms = 1e3 * np.arange(length) / fs
    view = t_ms <= 6.0
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_ms[view], ir[view], color=COLOR_PRIMARY, linewidth=1.6,
            label="Recovered IR (golay_impulse_response)")
    ax.plot(t_ms[view], true_ir[view], color=COLOR_SECONDARY, linewidth=1.2,
            linestyle="--", label="True system response")
    ax.text(0.985, 0.05,
            f"max |recovered - true| = {err:.1e}\n"
            "noise-free closed-form identity",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=9,
            color=COLOR_FG,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": COLOR_PANEL,
                  "edgecolor": COLOR_GRID})
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Golay-Pair Impulse Response: Exact Complementary Recovery",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "golay_ir.svg")
    plt.close()


def generate_tsa_noise_reduction(output_dir: str) -> None:
    """TSA: measured error of the average against the ideal 1/sqrt(N)."""
    print("Generating tsa_noise_reduction...")
    from phonometry import time_synchronous_average

    fs = 8192.0
    samples = 256
    period = samples / fs
    m = np.arange(samples) / fs
    true = (np.cos(2.0 * np.pi * m / period)
            + 0.5 * np.cos(2.0 * np.pi * 3.0 * m / period + 0.7)
            + 0.25 * np.cos(2.0 * np.pi * 5.0 * m / period + 1.1))
    rng = np.random.default_rng(5)
    n_max = 128
    x = np.tile(true, n_max) + rng.standard_normal(n_max * samples)

    counts = [1, 2, 4, 8, 16, 32, 64, 128]
    errors = []
    for n in counts:
        res = time_synchronous_average(x[: n * samples], fs, period,
                                       n_averages=n)
        errors.append(float(np.sqrt(np.mean(
            (res.period_waveform - true) ** 2))))

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(counts, errors, "o-", color=COLOR_PRIMARY, linewidth=1.6,
              markersize=6, label="Measured RMS error of the average")
    ax.loglog(counts, 1.0 / np.sqrt(np.asarray(counts, dtype=np.float64)),
              color=COLOR_SECONDARY, linestyle="--", linewidth=1.4,
              label=r"Ideal $\sigma/\sqrt{N}$")
    ax.set_xticks(counts)
    ax.set_xticklabels([str(n) for n in counts])
    ax.set_xlabel("Number of averages N")
    ax.set_ylabel("RMS error of the averaged waveform")
    ax.set_title(r"TSA Noise Reduction: the $\sqrt{N}$ Law (McFadden 1987)",
                 fontweight="bold", pad=12)
    ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "tsa_noise_reduction.svg")
    plt.close()


def generate_runs_test(output_dir: str) -> None:
    """Runs test about the median: accepted noise vs rejected alternation."""
    print("Generating runs_test...")
    from phonometry import trend_test

    rng = np.random.default_rng(3)
    sequences = [rng.standard_normal(40), np.tile([1.0, -1.0], 10)]

    _fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, seq in zip(axes, sequences):
        res = trend_test(seq, method="runs")
        idx = np.arange(1, seq.size + 1)
        median = float(np.median(seq))
        above = seq > median
        ax.plot(idx, seq, color="#7f7f7f", linewidth=0.7, alpha=0.7,
                zorder=1)
        ax.scatter(idx[above], seq[above], color=COLOR_PRIMARY, s=28,
                   zorder=3, label="Above the median")
        ax.scatter(idx[~above], seq[~above], color=COLOR_SECONDARY, s=28,
                   zorder=3, label="Below the median")
        ax.axhline(median, color=COLOR_FG, linestyle="--", linewidth=1.1,
                   alpha=0.7, label="Sequence median")
        verdict = "trend-free" if res.trend_free else "rejected"
        ax.set_title(f"r = {res.statistic} runs, accept "
                     f"({res.bounds[0]}, {res.bounds[1]}]: {verdict}",
                     fontweight="bold", pad=10)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sequence value")
        ax.grid(color=COLOR_GRID, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", fontsize=8.5)
    plt.suptitle("Runs Test About the Median (Wald & Wolfowitz)",
                 fontweight="bold")
    plt.tight_layout()
    save_figure(output_dir, "runs_test.svg")
    plt.close()


def generate_rd1367_activity_assessment(output_dir: str) -> None:
    """RD 1367/2007 activity assessment against the Annex III Table B1 limits.

    The worked case of Aviles Lopez & Perera Martin, Manual de acustica
    ambiental y arquitectonica, Ejemplos 3.1 to 3.3: an activity on residential
    land whose day period splits into 2 h shut down, 6 h with a noisy machine
    (LKeq,Ti = 59 dB) and 4 h with the remaining sources (54 dB), giving
    LKeq,d = 57 dB and, over 303 operating days, LK,d = 56 dB. Against the
    55 dB limit of area type a the phase (+5 dB) and daily (+3 dB) criteria are
    met but the annual one is not.
    """
    print("Generating rd1367_activity_assessment.png...")
    from phonometry import NoisePhase, activity_limits, assess_activity

    day = [
        NoisePhase(2.0, 0.0, label="closed"),
        NoisePhase(6.0, 50.0, kt=6.0, kf=3.0),
        NoisePhase(4.0, 48.0, kt=3.0, kf=3.0),
    ]
    evening = [NoisePhase(2.0, 48.0, kt=3.0, kf=3.0), NoisePhase(2.0, 0.0)]
    verdict = assess_activity(
        {"day": day, "evening": evening}, activity_limits("a"), operating_days=303
    )

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the three Article 25.1 b indices per
    # period with their own allowances marked on each group.
    verdict.plot(ax=ax, language=_LANG)
    ax.set_ylim(0, 72)
    save_figure(output_dir, "rd1367_activity_assessment.png")
    plt.close()


def generate_dbhr_global_index(output_dir: str) -> None:
    """CTE DB-HR global index R'A of a measured wall (Manual Ejemplo 7.2).

    The published eighteen-band apparent sound reduction index R' of a field
    test, weighted with the normalised pink-noise spectrum of DB-HR Annex A
    Table A.5. The energy sum of the per-band transmitted level L_Ar,i - R'_i
    sets the global index, R'A = 51,4 dBA.
    """
    print("Generating dbhr_global_index.png...")
    from phonometry import ra

    r_prime = [
        36.2, 41.5, 36.9, 40.4, 44.7, 42.4, 45.7, 46.1, 47.1,
        52.3, 54.3, 57.5, 57.8, 57.3, 59.0, 62.8, 64.7, 65.3,
    ]

    _, ax = plt.subplots(figsize=(10, 6))
    # The result's own .plot() draws the band insulation as bars and the
    # weighted per-band transmitted level as a line on a 1k/2k-labelled axis.
    ra(r_prime).plot(ax=ax, language=_LANG)
    save_figure(output_dir, "dbhr_global_index.png")
    plt.close()


def _cnossos_rail_scene() -> tuple[Any, Any]:
    """A four-axle disc-braked coach on a normally maintained ballasted track.

    The vehicle is the 920 mm wheel of Table G-3b at the 50 kN wheel load of
    Table G-2, the track a concrete mono-block sleeper on a medium-stiffness
    rail pad with one joint per 100 m, which is the default Annex II prescribes
    for jointed track.
    """
    from phonometry import (
        BrakeType,
        ContactFilter,
        RailRoughnessClass,
        RailwayTrack,
        RollingStock,
        TrackTransferClass,
        TractionVehicle,
        WheelDiameter,
        aerodynamic_sound_power,
        contact_filter,
        impact_roughness_single,
        rail_roughness,
        track_transfer,
        traction_sound_power,
        wheel_roughness,
        wheel_transfer,
    )

    stock = RollingStock(
        axles=4,
        wheel_roughness=wheel_roughness(BrakeType.NON_TREAD),
        contact_filter=contact_filter(ContactFilter.LOAD_50_DIAMETER_920),
        wheel_transfer=wheel_transfer(WheelDiameter.MM_920),
        traction=traction_sound_power(TractionVehicle.ELECTRIC_MULTIPLE_UNIT),
        aerodynamic=aerodynamic_sound_power(),
    )
    track = RailwayTrack(
        rail_roughness=rail_roughness(RailRoughnessClass.NORMAL),
        track_transfer=track_transfer(TrackTransferClass.MONOBLOCK_MEDIUM),
        impact_roughness=impact_roughness_single(),
    )
    return stock, track


def generate_cnossos_rail_emission(output_dir: str) -> None:
    """CNOSSOS-EU railway source-line power at the two equivalent heights."""
    print("Generating cnossos_rail_emission.png...")
    from phonometry import RailwayVehicle, railway_source_power

    stock, track = _cnossos_rail_scene()
    bands = np.array([63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    # 96 coaches per hour at 160 km/h: eight eight-car trains an hour on a
    # conventional main line, seen broadside and slightly from above.
    result = railway_source_power(
        RailwayVehicle(stock, flow_rate=96.0, speed=160.0), track, phi=90.0, psi=10.0,
    )
    x = np.arange(len(bands))
    _fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.bar(x, result.total_line_power, color=COLOR_MUTED, edgecolor=COLOR_FG,
           linewidth=0.6, label="Both source heights", zorder=2)
    for row, color, marker, label in zip(
        result.line_power, (COLOR_PRIMARY, COLOR_SECONDARY), ("o", "s"),
        ("Source A, 0,5 m (rolling, impact, traction)",
         "Source B, 4,0 m (traction, aerodynamic)"), strict=True,
    ):
        ax.plot(x, row, color=color, marker=marker, linewidth=1.8, markersize=6,
                markerfacecolor="white", markeredgewidth=1.3, zorder=5, label=label)

    ax.set_title(
        "CNOSSOS-EU Railway Source Line Power (96 coaches/h at 160 km/h)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("Octave-band centre frequency [Hz]")
    ax.set_ylabel("Line power L'W,eq,line [dB re 1 pW/m]")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b:g}" for b in bands])
    ax.set_ylim(30.0, 95.0)
    ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_emission.png")
    plt.close()


def generate_cnossos_rail_roughness_shift(output_dir: str) -> None:
    """The roughness spectrum sliding along the frequency axis with speed."""
    print("Generating cnossos_rail_roughness_shift.png...")
    from phonometry import (
        RAILWAY_THIRD_OCTAVE_BANDS,
        BrakeType,
        ContactFilter,
        RailRoughnessClass,
        contact_filter,
        rail_roughness,
        roughness_to_frequency,
        total_effective_roughness,
        wheel_roughness,
    )

    freqs = np.asarray(RAILWAY_THIRD_OCTAVE_BANDS)
    rail = rail_roughness(RailRoughnessClass.NORMAL)
    wheel = wheel_roughness(BrakeType.NON_TREAD)
    filt = contact_filter(ContactFilter.LOAD_50_DIAMETER_920)

    _fig, ax = plt.subplots(figsize=(11, 6.4))
    for speed, color, marker in (
        (60.0, COLOR_PRIMARY, "o"), (160.0, COLOR_TERTIARY, "s"),
        (300.0, COLOR_SECONDARY, "D"),
    ):
        total = total_effective_roughness(
            roughness_to_frequency(rail[1], rail[0], speed),
            roughness_to_frequency(wheel[1], wheel[0], speed),
            roughness_to_frequency(filt[1], filt[0], speed),
        )
        ax.semilogx(freqs, total, color=color, marker=marker, linewidth=1.8,
                    markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                    label=f"v = {speed:g} km/h", zorder=4)
    ax.axhline(0.0, color=COLOR_MUTED, linewidth=0.9, zorder=1)
    ax.set_title(
        "CNOSSOS-EU Total Effective Roughness against Speed "
        "(f = v/λ, λ in the wavelength domain)",
        fontweight="bold", pad=12,
    )
    ax.set_xlabel("1/3-octave band centre frequency [Hz]")
    ax.set_ylabel("Total effective roughness LR,TOT [dB re 1 μm]")
    ax.set_xlim(50.0, 10000.0)
    format_frequency_axis(ax, 50.0, 10000.0)
    ax.grid(which="both", color=COLOR_GRID, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    save_figure(output_dir, "cnossos_rail_roughness_shift.png")
    plt.close()


_FIGURE_FUNCS: tuple[Callable[[str], None], ...] = (
    generate_filter_type_comparison,
    generate_filter_responses,
    generate_signal_responses,
    generate_multichannel_response,
    generate_decomposition_plot,
    generate_weighting_responses,
    generate_special_weighting_responses,
    generate_g_weighting_response,
    generate_equal_loudness_contours,
    generate_time_weighting_plot,
    generate_crossover_plot,
    generate_parametric_eq_family,
    # Feature documentation plots (levels, spectrogram, zero-phase, weighting accuracy)
    generate_spectrogram_example,
    generate_ln_levels_example,
    generate_zero_phase_comparison,
    generate_weighting_accuracy_hf,
    # Docs-enrichment plots (group delay, IEC toneburst, block continuity, class mask)
    generate_group_delay_comparison,
    generate_tone_burst_iec,
    generate_block_processing_continuity,
    generate_class_mask_overlay,
    generate_filter_class0_mask,
    generate_weighting_class_mask,
    generate_calibration_stability,
    generate_sel_concept,
    generate_lden_profile,
    generate_rd1367_activity_assessment,
    generate_dbhr_global_index,
    generate_tonality_spectrum,
    # Psychoacoustics / intensity plots (loudness, STI, p-p intensity)
    generate_loudness_pattern,
    generate_sti_curve,
    generate_intensity_demo,
    # Room / building acoustics plots (ISO 18233 excitations + IR, Schroeder
    # decay, ISO 717-1/-2 ratings)
    generate_excitation_signals,
    generate_impulse_response,
    generate_schroeder_decay,
    generate_insulation_rating,
    generate_impact_rating,
    # Building-acoustics prediction / uncertainty (EN 12354-1, ISO 12999-1)
    generate_prediction_flanking_demo,
    generate_facade_prediction,
    generate_intensity_insulation,
    generate_survey_insulation,
    generate_floor_covering_improvement,
    generate_heavy_impact_sources,
    generate_ceiling_plenum_flanking,
    generate_masonry_wall_ties,
    generate_flanking_transmission,
    generate_reverberation_models,
    generate_dynamic_stiffness,
    generate_mechanical_mobility,
    generate_junction_transmission,
    generate_bearing_fault_envelope,
    generate_experimental_sea_clf,
    generate_plateau_transmission_loss,
    generate_orthotropic_transmission_loss,
    generate_transfer_stiffness,
    generate_rigid_mass_calibration,
    generate_vibration_sound_power,
    generate_structure_borne_power,
    generate_installed_structure_borne,
    generate_tone_audibility,
    generate_absorption_uncertainty,
    generate_insulation_uncertainty_demo,
    # Outdoor propagation & occupational exposure (ISO 9613-1/2, ISO 9612)
    generate_air_absorption_alpha,
    generate_atmospheric_attenuation,
    generate_outdoor_attenuation_breakdown,
    generate_cnossos_road_emission,
    generate_cnossos_road_speed_law,
    generate_ground_effect_spherical,
    generate_atmospheric_refraction,
    # CNOSSOS-EU railway source emission (Directive 2002/49/EC Annex II, 2.3).
    generate_cnossos_rail_emission,
    generate_cnossos_rail_roughness_shift,
    generate_exposure_uncertainty,
    # Materials: absorption rating, airflow resistance, impedance tube
    # (ISO 11654, ISO 9053-1/-2, ISO 10534-1/-2, ASTM E2611)
    generate_absorption_rating,
    generate_airflow_resistance,
    generate_impedance_tube,
    # Porous materials & multilayer absorbers (Mechel / Bies / Cox & D'Antonio)
    generate_porous_absorber_designs,
    generate_limp_frame_effective_density,
    generate_biot_frame_resonance,
    generate_absorber_stack_geometry,
    # Slow-sound slit + Helmholtz-resonator perfect absorbers (Jimenez et al.)
    generate_metadiffuser_geometry,
    generate_metadiffuser_polar,
    generate_metadiffuser_ntff_polar,
    generate_slow_sound_absorber,
    generate_slit_absorber_geometry,
    generate_helmholtz_resonator_geometry,
    # Scattering/diffusion, in-situ road absorption, precision sound power
    # (ISO 17497-1/-2, ISO 13472-1, ISO 3745 / ISO 9614-3)
    generate_scattering_coefficient,
    generate_diffusion_polar,
    generate_diffuser_prediction,
    generate_qrd_geometry,
    generate_impedance_tube_geometry,
    generate_transmission_tube_geometry,
    generate_insitu_absorption,
    generate_precision_anechoic_power,
    generate_intensity_scan_power,
    # Sound power result spectra for the three most-used routes
    # (ISO 3744 enveloping surface, ISO 3741 reverberation room,
    # ISO 9614-2 intensity scanning)
    generate_sound_power_pressure_result,
    generate_sound_power_reverberation_result,
    generate_sound_power_intensity_result,
    # Human vibration (ISO 8041-1, ISO 2631-1/-2/-4, ISO 5349-1/-2,
    # Directive 2002/44/EC): frequency weighting, weighted a_w, daily A(8)
    generate_vibration_weighting,
    generate_weighted_acceleration,
    generate_daily_vibration_exposure,
    # Speech intelligibility (ANSI S3.5-1997): band audibility and the SII.
    generate_speech_intelligibility,
    generate_sii_vocal_efforts,
    generate_standard_speech_spectrum,
    generate_sii_band_procedures,
    generate_impulse_prominence,
    # Room-noise criteria (ANSI S12.2-2019): NC tangency and RC Mark II.
    generate_room_noise_criteria,
    # Hearing threshold (ISO 7029 age-related, ISO 389-7 reference).
    generate_hearing_threshold,
    # Noise-induced hearing loss (ISO 1999 NIPTS and HTLAN).
    generate_noise_induced_hearing_loss,
    # Multiple-shock whole-body vibration (ISO 2631-5 Clause 5 + Annex C).
    generate_tonal_audibility,
    generate_multiple_shock,
    # Sound absorption in enclosed spaces (EN 12354-6 Clause 4).
    generate_enclosed_space_absorption,
    # Measurement uncertainty (GUM Guide 98-3 + Supplement 1 Monte Carlo).
    generate_uncertainty,
    # Psychoacoustics / open-plan plots (sharpness weighting, spatial decay)
    generate_sharpness_weighting,
    generate_open_plan_decay,
    # Rectangular room modes (Long Ch. 8), restaurant crowd self-noise
    # (Long Ch. 17) and the ERB_N / Cam auditory-filter scale.
    generate_rectangular_room_modes,
    generate_restaurant_crowd_noise,
    generate_erb_bandwidth,
    # Advanced psychoacoustics: ECMA-418-2 Sottek model and Moore-Glasberg
    # ISO 532-2/-3 loudness (models, specific loudness, sound quality
    # metrics, time-varying loudness).
    generate_loudness_models_comparison,
    generate_sottek_specific_loudness,
    generate_tonality_roughness_demo,
    # ECMA-418-2 fluctuation strength (Clause 9) vs roughness (Clause 7):
    # the complementary slow/fast modulation band-passes.
    generate_hms_modulation_bandpass,
    generate_moore_glasberg_time_loudness,
    # Fluctuation strength (Fastl & Zwicker Eq. 10.2 + Osses 2016 signal model)
    # and psychoacoustic annoyance (Fastl & Zwicker Eqs 16.2-16.4).
    generate_fluctuation_strength,
    generate_psychoacoustic_annoyance,
    # Electroacoustics: distortion metrics (IEC 60268-3) and
    # frequency-response / coherence estimators (Bendat & Piersol).
    generate_distortion,
    generate_frequency_response,
    # Swept-sine harmonic separation: THD(f) by order from one synchronized
    # sweep (Farina 2000 / Novak et al. 2015).
    generate_swept_sine_thd,
    # Far-field directivity (beam) pattern of a baffled circular piston.
    generate_piston_directivity,
    # Single-concept rated-characteristic .plot() figures, sharing their panel
    # drawing with the .report() fiches (IEC 60268-5 loudspeaker, IEC 60268-4
    # microphone).
    generate_loudspeaker_response,
    generate_loudspeaker_impedance,
    generate_loudspeaker_thd,
    generate_loudspeaker_directivity,
    generate_microphone_response,
    generate_microphone_directivity,
    generate_microphone_noise,
    generate_microphone_distortion,
    # Calibrated spectral analysis: PSD with chi-square confidence interval
    # and 1/3-octave smoothing on exact-slope pink noise (Bendat & Piersol).
    generate_psd_confidence_smoothing,
    # Thomson multitaper density of a short record with its chi-square
    # band against the single-taper estimate (Percival & Walden 1993).
    generate_multitaper_psd_confidence,
    # Time-frequency analysis: calibrated STFT spectrogram in dB SPL and
    # the zoom FFT resolving sub-bin tone pairs (Bendat & Piersol).
    generate_calibrated_spectrogram,
    generate_zoom_fft_resolution,
    # Signal toolbox: IEC 60268-1 tone bursts (single and repetitive train)
    # and the window figures of merit (Harris 1978).
    generate_tone_burst_train,
    generate_window_functions_tradeoff,
    # Broadcast: programme loudness and true peak (ITU-R BS.1770-5 /
    # EBU R 128 with Tech 3341/3342).
    generate_program_loudness,
    generate_k_weighting_response,
    # Correlation / time-delay estimation: GCC-PHAT vs the direct
    # correlator on a colored signal pair (Knapp & Carter 1976).
    generate_gcc_phat_delay,
    # Cepstral analysis: echo detection on the power cepstrum (Havelock
    # Chs. 27/87) and the envelope spectrum of an AM tone (B&P 13.3).
    generate_cepstrum_echo,
    generate_envelope_spectrum,
    # Time synchronous averaging of a periodic waveform in noise (McFadden 1987).
    generate_synchronous_average,
    # Multiple-input/output coherence (Bendat & Piersol Ch. 7).
    generate_miso_coherence,
    # Data qualification: reverse arrangement trend and stationarity tests
    # level-crossing / peak statistics (Bendat & Piersol 10.3 / 5.5).
    generate_trend_test,
    generate_stationarity_test,
    generate_rice_level_crossings,
    generate_rice_peak_distribution,
    # Regularized inversion (Kirkeby) and the Mueller-Massarani shaped sweep.
    generate_regularized_inversion,
    generate_shaped_sweep,
    # Objective intelligibility: STOI vs ESTOI over SNR for stationary and
    # modulated maskers (Taal et al. 2011 / Jensen & Taal 2016).
    generate_stoi_intelligibility,
    # Room acoustics: the synthetic image-source room impulse response as a
    # reflectogram of mirror-image reflections by order (Kuttruff 4.1).
    generate_image_source_reflectogram,
    generate_image_source_plan,
    generate_open_plan_line_geometry,
    generate_sound_reinforcement_geometry,
    # Underwater acoustics: ship radiated noise / monopole source
    # level (ISO 17208) and pile-driving sound exposure (ISO 18406).
    generate_ship_source_level,
    generate_pile_driving,
    # Aircraft noise: ICAO Annex 16 Effective Perceived Noise Level.
    generate_epnl,
    # Wind-turbine noise: IEC 61400-11 tonal audibility.
    generate_wind_turbine_tonality,
    # Underwater propagation: transmission loss, sound-speed profile and the
    # sonar equation.
    generate_underwater_transmission_loss,
    generate_weston_regimes,
    generate_underwater_sound_speed,
    generate_sonar_equation,
    # Underwater fauna: regulatory auditory weighting (NMFS 2024).
    generate_marine_mammal_weighting,
    # Underwater propagation: seabed reflection, ambient noise (Wenz) and
    # ship-traffic source level (JOMOPANS-ECHO).
    generate_seabed_reflection,
    generate_seabed_reflection_coefficient,
    generate_ocean_ambient_noise,
    generate_ship_traffic_noise,
    # Underwater propagation: numerical solvers (modes/rays/PE).
    generate_numerical_propagation,
    # Aircraft atmospheric absorption: SAE ARP 5534 band method.
    generate_aircraft_atmospheric_absorption,
    # Airport noise: ECAC Doc 29 noise-power-distance curves.
    generate_airport_noise,
    generate_airport_contour,
    generate_airport_sor,
    generate_rotorcraft_ground_effect,
    generate_rotorcraft_flyover_event,
    generate_rotorcraft_terrain_screening,
    # 2D FDTD wave simulation (public API concept figure).
    generate_fdtd_simulation,
    # Elastic P-SV FDTD: half-space snapshot with P/S/Rayleigh fronts.
    generate_elastic_halfspace_waves,
    # Elastic FDTD fluid-solid coupling: the Scholte interface wave.
    generate_scholte_interface_wave,
    # Theoretical panel sound insulation (single/double wall, radiation, slit).
    generate_panel_insulation_concept,
    generate_silencer_expansion_chamber,
    generate_expansion_chamber_geometry,
    # Rooms & materials result plots: ISO 354 measurement, ISO 10534-2 tube
    # result, ASTM E2611 transfer matrix, porous models, MPP peak, Paris
    # integral, Bies steady-state field, ISO 3382 per-band parameters and the
    # rigid-box FDTD mode oracle.
    generate_sound_absorption_measurement,
    generate_impedance_tube_result,
    generate_transfer_matrix_tl,
    generate_porous_medium_model,
    generate_mpp_absorption_peak,
    generate_diffuse_field_absorption,
    generate_steady_state_field,
    generate_room_parameters_bands,
    generate_fdtd_room_modes,
    # Building & structure-borne result figures (guide figure coverage):
    # ISO 717 enlarged range, ISO 16283 field chains, ISO 10052 survey impact,
    # ISO 10140 laboratory quantities, ISO 15186-1 small elements, ISO 10848
    # flanking descriptors, EN 12354-2/-4 predictions, the rated Sharp panel,
    # the wave-approach junction Kij and the ISO 7626 mobility lines.
    generate_extended_insulation_rating,
    generate_field_airborne_insulation,
    generate_facade_field_insulation,
    generate_survey_impact_insulation,
    generate_lab_insulation_result,
    generate_intensity_element_insulation,
    generate_flanking_level_difference,
    generate_impact_prediction_terms,
    generate_detailed_prediction_paths,
    generate_radiated_power_outdoor,
    generate_single_panel_rating,
    generate_junction_kij_thickness,
    generate_mobility_result_lines,
    # Perception, hearing and speech: single-concept result figures drawn by
    # the results' own .plot() (ECMA-418-1 tone prominence, ISO/PAS 20065 tone
    # levels, ISO/PAS 1996-3 onsets, ISO 532-2 and ECMA-418-2 patterns, the
    # Osses fluctuation strength, ANSI S3.5 audibility with hearing loss,
    # ISO 7029 / ISO 1999 thresholds, STOI band scores and the STI MTI bars).
    generate_tone_prominence_assessment,
    generate_tone_audibility_levels,
    generate_impulsive_sound_onsets,
    generate_moore_glasberg_specific_loudness,
    generate_sottek_specific_tonality,
    generate_fluctuation_strength_specific,
    generate_sii_hearing_loss,
    generate_age_threshold_fractiles,
    generate_nipts_audiogram,
    generate_stoi_band_scores,
    generate_sti_band_mti,
    # Atmospheric refraction (profiles, ray fan, GFPE range cut) and
    # wave-theoretic barrier insertion loss.
    generate_atmospheric_sound_speed_profiles,
    generate_atmospheric_ray_fan,
    generate_atmospheric_pe_range,
    generate_barrier_insertion_loss_methods,
    generate_barrier_geometry,
    generate_facade_elevation_geometry,
    generate_double_wall_geometry,
    generate_junction_plate_geometry,
    generate_insitu_setup_geometry,
    generate_dynamic_stiffness_rig_geometry,
    generate_diffusion_goniometer_geometry,
    generate_radiation_plate_geometry,
    generate_pp_probe_geometry,
    # Emission & electroacoustics: modulation sidebands, piston radiation
    # impedance, ISO 9614-1 field indicators, side-branch silencers, HVAC end
    # reflection, machine enclosures, phase decomposition and R 128 loudness.
    generate_modulation_distortion,
    generate_piston_radiation_impedance,
    generate_piston_baffle_geometry,
    generate_microphone_positions_hemisphere,
    generate_aperture_slit_geometry,
    generate_fdtd_domain_geometry,
    generate_field_indicators,
    generate_intensity_class,
    generate_silencer_side_branch,
    generate_helmholtz_branch_geometry,
    generate_quarter_wave_geometry,
    generate_plenum_geometry,
    generate_hvac_end_reflection,
    generate_duct_path_cascade,
    generate_room_to_room_chain,
    generate_duct_mode_cut_on,
    generate_enclosure_insertion_loss,
    generate_phase_decomposition,
    generate_loudness_gating,
    generate_loudness_range,
    # Core-metrology figures (resampling, cepstral analysis, correlation,
    # cross-spectra, Golay recovery, synchronous averaging, runs test).
    generate_resampling_antialias,
    generate_cepstrum_variants,
    generate_lifter_split,
    generate_correlation_normalizations,
    generate_ir_alignment,
    generate_hilbert_envelope,
    generate_cross_spectral_density_delay,
    generate_coherent_output_snr,
    generate_golay_ir,
    generate_tsa_noise_reduction,
    generate_runs_test,
)


def generate_all(img_dir: str) -> None:
    """Generate every documentation figure for the currently active theme."""
    for func in _FIGURE_FUNCS:
        func(img_dir)


# ====================================================================# Animations (Tier 1 pilot)
# ---------------------------------------------------------------------------
# Deterministic FuncAnimation clips of the level-vs-time phenomena the library
# already computes. Each is rendered to WebM (site <video>) in all four
# language x theme variants, and to an animated GIF for the English GitHub
# docs (both themes). Every WebM also gets a JPEG poster frame (the held
# verdict state near the end of the clip) so the site can embed the video
# with ``preload="none"`` and still show a meaningful still while nothing is
# fetched. Posters are ``.jpg`` on purpose: `make graphs` clears only
# ``*.svg``/``*.png``/``*.webp`` and scripts/check_figures.py
# regenerates/compares only those, so the posters ride with the WebM/GIF
# outside the figure pipeline. Kept out of generate_all()/`make graphs` so
# ordinary figure regeneration stays fast; produced by `make animations` (or `make posters` to re-extract the
# stills from the committed WebMs without re-rendering).
# ====================================================================
_ANIM_FPS = 20
_ANIM_SECONDS = 12
_ANIM_FRAMES = _ANIM_FPS * _ANIM_SECONDS
# Closing hold appended to each schematic timeline: the settled verdict frame
# stays on screen ~2 s so it can be read before the loop restarts.
_ANIM_HOLD = 2 * _ANIM_FPS
# The wave-field clips run on their own frame budgets: on the shared 12 s
# timeline a wavefront advances several cells per frame and the field
# strobes. Each FDTD/elastic clip therefore sizes its own window from the
# two animation norms -- the captured span is 1.2 times the full round-trip
# flight (source to the deepest reflector, wells and cavities included, and
# on to the farthest visible point, at the slowest speed on the path) and
# the capture stride keeps at least 48 frames per period of the highest
# carrier -- and states that arithmetic next to its frame count. This 18 s
# value is the starting budget the older clips were sized against and the
# default for clips whose own note does not override it.
_FDTD_ANIM_FRAMES = 18 * _ANIM_FPS
_ANIM_FIGSIZE = (8.0, 4.5)   # inches at _ANIM_DPI -> 2400 x 1350 px
_ANIM_DPI = 300
# The GitHub-docs GIF is a compact fallback for the smooth site WebM: a lower
# frame rate, smaller frame and capped palette keep even the detail-dense
# clips (the p·u oscillations) near half a megabyte.
_GIF_FPS = 12
_GIF_SCALE = 640
_GIF_COLORS = 64
# Rounded box of the verdict/annotation pills drawn over the clips.
_ANIM_PILL_BOX = "round,pad=0.25"

# VP9 encoding for the site WebM. Beyond the constant-quality base
# (``-crf 40 -b:v 0``) the screen-content tuning, row multithreading and the
# ``good`` deadline cut the encode time roughly in half at the same file size
# (the flat schematic / field content compresses identically). ``-cpu-used``
# trades encode speed for compression effort; on this content the size and
# quality are essentially flat from 2 through 4, so the run uses the faster
# setting. ``-threads`` is capped so eight parallel Python workers do not
# oversubscribe the box. Animations are not byte-compared (unlike the
# SVG/WebP figures), so the mild run-to-run variance these knobs introduce is
# harmless.
_ANIM_CPU_USED = 4
_ANIM_ENCODE_THREADS = 3


def _vp9_extra_args() -> list[str]:
    """ffmpeg args for the tuned constant-quality VP9 WebM encode."""
    return [
        "-b:v", "0", "-crf", "40",
        "-tune-content", "screen", "-row-mt", "1", "-deadline", "good",
        "-cpu-used", str(_ANIM_CPU_USED), "-threads", str(_ANIM_ENCODE_THREADS),
        "-pix_fmt", "yuv420p", "-an", "-loglevel", "error",
    ]


def _translate_str(s: str) -> str:
    """Translate one label to the active language (exact + pattern + comma).

    Mirrors :func:`_translate_figure` for a single string, so animation labels
    -- which are rewritten every frame and never pass through ``save_figure``
    -- can be localised at creation time instead.
    """
    import re as _re

    if _LANG == "en" or not s:
        return s
    if s in _ES_EXACT:
        out = _ES_EXACT[s]
    else:
        out = s
        for pat, repl in _ES_PATTERNS:
            new, n = _re.subn(pat, repl, s)
            if n:
                out = new
                break
    if "$" not in out and _re.search(r"\d\.\d", out):
        out = _re.sub(r"(?<![\d.A-Za-z])(\d+)\.(\d+)(?![.\d])", r"\1,\2", out)
    return out


def _anim_path(output_dir: str, stem: str, ext: str) -> str:
    """Animation output path with the active language + theme suffixes."""
    return os.path.join(output_dir, f"{stem}{_LANG_SUFFIX}{_FILENAME_SUFFIX}.{ext}")


def _anim_figure() -> Any:
    """A fixed-size themed figure for a clip (constant canvas across frames)."""
    return plt.figure(figsize=_ANIM_FIGSIZE, dpi=_ANIM_DPI, layout="constrained")


def _grid_axes(ax: Any) -> None:
    """Apply the standard documentation grid to a data axes."""
    ax.grid(True, color=COLOR_GRID, linestyle="--", alpha=0.5)


def _schematic_axes(ax: Any, xlim: tuple[float, float],
                    ylim: tuple[float, float], *, equal: bool = False) -> None:
    """Turn *ax* into a bare drawing canvas (no ticks, frame or grid)."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")


def _render_clip(fig: Any, update: Callable[[int], tuple[Any, ...]],
                 output_dir: str, stem: str, *, frames: int | None = None,
                 fps: float | None = None, gif_fps: int | None = None,
                 poster_ss: float | None = None) -> None:
    """Shared clip tail: drive *update* through FuncAnimation and encode.

    Static artists (tick labels included) get the same Spanish pass as the
    figures; per-frame labels are translated by ``_translate_str`` inside
    each clip's update function. ``frames``/``fps`` override the shared
    timeline for clips that need their own budget (the FDTD room modes);
    ``gif_fps`` is forwarded to the GitHub-GIF palette pass.
    """
    from matplotlib.animation import FuncAnimation

    _translate_figure(fig)
    n_frames = _ANIM_FRAMES if frames is None else frames
    rate = _ANIM_FPS if fps is None else fps
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / rate, blit=False)
    _save_animation(anim, fig, output_dir, stem, fps=rate, gif_fps=gif_fps, poster_ss=poster_ss)


# --- shared schematic vocabulary (mics, gauges, flow boxes, arrows) --------


def _draw_mic(ax: Any, x: float, y: float, *, direction: int = 1,
              size: float = 1.0, label: str = "", angle: float = 0.0) -> None:
    """A measurement-microphone symbol: rounded body plus capsule head.

    ``direction = +1`` points the capsule toward +x, ``-1`` toward -x. Draw
    on an equal-aspect schematic axes so the capsule stays round. ``angle``
    rotates the whole symbol (degrees, counterclockwise) around ``(x, y)``
    -- e.g. ``-90`` points a ``direction = +1`` capsule straight down; the
    label stays horizontal and rides on the rotated anchor.
    """
    from matplotlib import transforms
    from matplotlib.patches import Circle, FancyBboxPatch

    tr = ax.transData
    if angle:
        tr = transforms.Affine2D().rotate_deg_around(x, y, angle) + tr
    body_w, body_h = 1.1 * size, 0.42 * size
    head_r = 0.26 * size
    body = FancyBboxPatch(
        (x - body_w / 2, y - body_h / 2), body_w, body_h,
        boxstyle="round,pad=0.03", facecolor=COLOR_GRID,
        edgecolor=COLOR_FG, lw=1.2)
    body.set_transform(tr)
    ax.add_patch(body)
    hx = x + direction * (body_w / 2 + head_r * 0.85)
    head = Circle((hx, y), head_r, facecolor="none", edgecolor=COLOR_FG,
                  lw=1.2)
    head.set_transform(tr)
    ax.add_patch(head)
    for frac in (-0.45, 0.0, 0.45):
        half = head_r * float(np.sqrt(1.0 - frac * frac)) * 0.9
        (grille,) = ax.plot([hx - half, hx + half], [y + frac * head_r] * 2,
                            color=COLOR_FG, lw=0.6)
        grille.set_transform(tr)
    if label:
        rad = np.deg2rad(angle)
        dx, dy = 0.0, body_h * 1.7
        lx = x + dx * np.cos(rad) - dy * np.sin(rad)
        ly = y + dx * np.sin(rad) + dy * np.cos(rad)
        ax.text(lx, ly, label, ha="center", va="bottom",
                color=COLOR_FG, fontsize=11)


def _draw_speaker(ax: Any, x: float, y: float, *, size: float = 1.0,
                  direction: int = 1) -> None:
    """A loudspeaker symbol: cabinet plus cone, radiating toward
    ``direction`` (+1 = +x, -1 = -x). ``(x, y)`` is the cone mouth centre."""
    from matplotlib.patches import Polygon, Rectangle

    d = float(direction)
    cone_l, cone_h = 0.42 * size, 0.55 * size
    box_l, box_h = 0.5 * size, 0.36 * size
    xb = x - d * cone_l
    ax.add_patch(Polygon(
        [(x, y - cone_h), (x, y + cone_h), (xb, y + box_h), (xb, y - box_h)],
        closed=True, facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    ax.add_patch(Rectangle((min(xb, xb - d * box_l), y - box_h),
                           box_l, 2 * box_h, facecolor=COLOR_GRID,
                           edgecolor=COLOR_FG, lw=1.2))


def _make_wavefronts(ax: Any, x: float, y: float, color: str, n: int = 4,
                     *, theta1: float = 0.0, theta2: float = 360.0,
                     lw: float = 1.6) -> list[Any]:
    """``n`` expanding wavefront arcs centred on ``(x, y)``, initially hidden.

    Drive them with :func:`_set_wavefronts`; ``theta1``/``theta2`` restrict
    the arc span (degrees) for sources radiating into a half or quarter space.
    """
    from matplotlib.patches import Arc

    arcs = []
    for _ in range(n):
        arc = Arc((x, y), 0.01, 0.01, theta1=theta1, theta2=theta2,
                  edgecolor=color, lw=lw)
        arc.set_visible(False)
        ax.add_patch(arc)
        arcs.append(arc)
    return arcs


def _set_wavefronts(arcs: list[Any], radii: list[float], rmax: float,
                    *, alpha: float = 0.9, color: Any = None) -> list[Any]:
    """Resize each wavefront arc; fronts fade toward ``rmax`` and hide beyond.

    ``radii`` pairs with ``arcs``; non-positive radii hide the arc. Returns
    the arcs (for blit-style artist lists).
    """
    for arc, r in zip(arcs, radii, strict=True):
        if r <= 0.0 or r > rmax:
            arc.set_visible(False)
            continue
        arc.set_visible(True)
        arc.width = 2.0 * r
        arc.height = 2.0 * r
        arc.set_alpha(alpha * max(0.0, 1.0 - r / rmax))
        if color is not None:
            arc.set_edgecolor(color)
    return arcs


def _polyline_point(pts: Any, frac: float) -> tuple[float, float]:
    """The point a fraction ``frac`` of the arc length along a polyline.

    ``pts`` is an ``(N, 2)`` array of vertices; ``frac`` is clipped to
    [0, 1]. Used to move pulses/probes along schematic paths at constant
    speed regardless of how the vertices are spaced.
    """
    p = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = float(np.clip(frac, 0.0, 1.0)) * cum[-1]
    i = int(np.searchsorted(cum[1:], s, side="right"))
    i = min(i, len(seg) - 1)
    t = 0.0 if seg[i] == 0.0 else (s - cum[i]) / seg[i]
    x = p[i, 0] + t * (p[i + 1, 0] - p[i, 0])
    y = p[i, 1] + t * (p[i + 1, 1] - p[i, 1])
    return float(x), float(y)


def _make_gauge(ax: Any, cx: float, cy: float, r: float, label: str,
                color: str, lo: str = "", hi: str = "") -> dict[str, Any]:
    """A semicircular meter dial; move the needle with :func:`_set_gauge`.

    ``lo``/``hi`` are optional scale-endpoint labels (left and right end of
    the arc), so a reader can anchor the needle position to numbers.
    """
    from matplotlib.patches import Arc

    ax.add_patch(Arc((cx, cy), 2 * r, 2 * r, theta1=0.0, theta2=180.0,
                     edgecolor=COLOR_FG, lw=1.4))
    for frac in np.linspace(0.0, 1.0, 5):
        a = np.pi * (1.0 - float(frac))
        ax.plot([cx + 0.86 * r * np.cos(a), cx + r * np.cos(a)],
                [cy + 0.86 * r * np.sin(a), cy + r * np.sin(a)],
                color=COLOR_FG, lw=0.8)
    if lo:
        ax.text(cx - 1.12 * r, cy - 0.02 * r, lo, ha="center", va="top",
                color=COLOR_FG, fontsize=7)
    if hi:
        ax.text(cx + 1.12 * r, cy - 0.02 * r, hi, ha="center", va="top",
                color=COLOR_FG, fontsize=7)
    (needle,) = ax.plot([cx, cx - 0.78 * r], [cy, cy], color=color, lw=2.4,
                        solid_capstyle="round")
    ax.plot([cx], [cy], marker="o", ms=4.5, color=color)
    ax.text(cx, cy - 0.30 * r, label, ha="center", va="top", color=color,
            fontsize=11, fontweight="bold")
    value = ax.text(cx, cy - 0.72 * r, "", ha="center", va="top",
                    color=COLOR_FG, fontsize=9, family="monospace")
    return {"needle": needle, "value": value, "cx": cx, "cy": cy, "r": r}


def _set_gauge(gauge: dict[str, Any], frac: float, text: str) -> tuple[Any, Any]:
    """Point the needle at ``frac`` in [0, 1] (left to right) + set readout."""
    a = np.pi * (1.0 - float(np.clip(frac, 0.0, 1.0)))
    cx, cy, r = gauge["cx"], gauge["cy"], gauge["r"]
    gauge["needle"].set_data([cx, cx + 0.78 * r * np.cos(a)],
                             [cy, cy + 0.78 * r * np.sin(a)])
    gauge["value"].set_text(text)
    return gauge["needle"], gauge["value"]


def _flow_box(ax: Any, cx: float, cy: float, w: float, h: float,
              title: str) -> dict[str, Any]:
    """A processing-pipeline box, dimmed until :func:`_light_box` lights it."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         facecolor="none",
                         edgecolor=to_rgba(COLOR_FG, 0.35), lw=1.6)
    ax.add_patch(box)
    title_t = ax.text(cx, cy + 0.22 * h, title, ha="center", va="center",
                      color=COLOR_FG, alpha=0.55, fontsize=9)
    value_t = ax.text(cx, cy - 0.22 * h, "", ha="center", va="center",
                      color=COLOR_FG, fontsize=9, fontweight="bold",
                      family="monospace")
    return {"box": box, "title": title_t, "value": value_t}


def _light_box(box: dict[str, Any], value: str, color: str,
               fill: bool = False) -> None:
    """Light a pipeline box up: colored edge, full-strength title, a value."""
    from matplotlib.colors import to_rgba

    box["box"].set_edgecolor(color)
    box["box"].set_linewidth(2.2)
    box["box"].set_facecolor(to_rgba(color, 0.15) if fill else "none")
    box["title"].set_alpha(1.0)
    box["value"].set_text(value)
    box["value"].set_color(color)


def _dim_box(box: dict[str, Any]) -> None:
    """Return a pipeline box to its dimmed, not-yet-evaluated state."""
    from matplotlib.colors import to_rgba

    box["box"].set_edgecolor(to_rgba(COLOR_FG, 0.35))
    box["box"].set_linewidth(1.6)
    box["box"].set_facecolor("none")
    box["title"].set_alpha(0.55)
    box["value"].set_text("")


def _make_arrow(ax: Any, color: str, scale: float = 14.0) -> Any:
    """An updatable arrow patch; reposition it with ``set_positions``."""
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), arrowstyle="-|>",
                            mutation_scale=scale, color=color, lw=2.0,
                            shrinkA=0.0, shrinkB=0.0)
    ax.add_patch(arrow)
    return arrow


def _draw_resistor(ax: Any, x0: float, x1: float, y: float) -> None:
    """A zigzag resistor symbol on a horizontal wire from *x0* to *x1*."""
    n = 6
    xs = np.linspace(x0, x1, 2 * n + 1)
    ys = np.full(xs.size, y)
    ys[1:-1] = y + 0.22 * np.where(np.arange(1, 2 * n) % 2 == 1, 1.0, -1.0)
    ax.plot(xs, ys, color=COLOR_FG, lw=1.4)


def _poster_ss_for(webm: str) -> float | None:
    """Per-clip poster-time override; None keeps the end-of-hold default.

    The pillar-hall banner poster is grabbed mid-flight (simulation time
    6 ms, the front threading the middle of the hall) instead of the
    settled last frame, which for a single travelling packet is an almost
    empty field.
    """
    if _PILLAR_STEM in os.path.basename(webm):
        import fdtd2d
        dt = fdtd2d.FDTD2D(343.0, _PILLAR_DX, shape=(3, 8)).dt
        dt_frame = _PILLAR_EVERY * dt
        return float((6.0e-3 / dt_frame - _PILLAR_WARM) / _PILLAR_FPS)
    return None


def _extract_poster(webm: str, poster_ss: float | None = None) -> str:
    """Extract the deferred-loading poster still from a rendered WebM.

    The frame is grabbed half a second before the end of the clip, i.e.
    inside the closing hold, so the poster shows the settled verdict state.
    The output sits next to the WebM as ``<stem>_poster.jpg``; JPEG keeps it
    outside the SVG/WebP figure pipeline (`make graphs` deletion glob and the
    scripts/check_figures.py regeneration compare).
    """
    import subprocess

    poster = webm.removesuffix(".webm") + "_poster.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error"]
        + (["-ss", f"{poster_ss:.2f}"] if poster_ss is not None
           else ["-sseof", "-0.5"])
        + ["-i", webm, "-frames:v", "1", "-q:v", "3", "-update", "1",
           poster],
        check=True)
    return poster


def generate_posters(output_dir: str) -> None:
    """Re-extract every animation poster from the already-rendered WebMs.

    Used by ``--posters`` (`make posters`) to refresh the stills without the
    slow re-encode of the clips themselves.
    """
    import glob

    webms = sorted(glob.glob(os.path.join(output_dir, "anim_*.webm")))
    if not webms:
        raise RuntimeError(f"no anim_*.webm files found in {output_dir}; "
                           "run `make animations` first")
    for webm in webms:
        poster = _extract_poster(webm, _poster_ss_for(webm))
        print(f"  {os.path.basename(webm)} -> {os.path.basename(poster)}")


def _gpu_encode_target() -> dict[str, str] | None:
    """The remote AV1 (NVENC) encode target from .env, if enabled and alive.

    Opt-in through ``PHONO_GPU_ENCODE=av1`` next to the ``PHONO_GPU_*``
    host settings: VP9 has no NVIDIA encoder, so the hardware path writes
    AV1 into the same WebM container (the site embeds are codec-agnostic;
    the GitHub GIF is still derived locally). Any doubt (no .env, host
    down, flag off) returns ``None`` and the clips keep the CPU VP9 path.
    """
    import subprocess

    try:
        import fdtd_gpu_remote

        fdtd_gpu_remote.load_env()
    except ImportError:
        return None
    if os.environ.get("PHONO_GPU_ENCODE", "").lower() != "av1":
        return None
    host = os.environ.get("PHONO_GPU_HOST", "")
    user = os.environ.get("PHONO_GPU_USER", "root")
    image = os.environ.get("PHONO_GPU_FFMPEG_IMAGE", "linuxserver/ffmpeg:latest")
    if not host:
        return None
    probe = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "--",
         f"{user}@{host}", "true"], capture_output=True, check=False)
    if probe.returncode != 0:
        return None
    workdir = os.environ.get("PHONO_GPU_WORKDIR", "/tmp/phonometry-gpu")
    return {"target": f"{user}@{host}", "image": image, "workdir": workdir}


def _av1_nvenc_extra_args() -> list[str]:
    """ffmpeg args for the remote AV1 NVENC encode (WebM container)."""
    return [
        "-c:v", "av1_nvenc",
        # cq 44 calibrated against the VP9 crf-40 size on the metadiffuser
        # clip (cq 42 = parity, 46 = -22 %): smaller files, same resolution.
        "-b:v", "0", "-cq", "44", "-preset", "p6",
        "-pix_fmt", "yuv420p", "-an", "-loglevel", "error",
    ]


def _write_gpu_ffmpeg_wrapper(target: str, image: str, workdir: str) -> str:
    """A throwaway ffmpeg shim that runs the encode on the GPU host.

    Matplotlib's writer spawns ``ffmpeg <args> <output>`` and streams raw
    frames on stdin; the shim forwards stdin over ssh into the NVENC
    container, which writes the WebM to a file in the remote work
    directory (the matroska muxer needs a seekable target), and the shim
    then fetches it back with scp and cleans up.
    """
    import shlex
    import tempfile

    q_target = shlex.quote(target)
    q_image = shlex.quote(image)
    q_dir = shlex.quote(workdir)
    script = f"""#!/bin/bash
set -euo pipefail
out="${{@: -1}}"
args=("${{@:1:$#-1}}")
rid="enc-$$-$RANDOM.webm"
quoted=$(printf ' %q' "${{args[@]}}")
ssh -o BatchMode=yes -- {q_target} \
    "mkdir -p {q_dir} && docker run --rm -i --gpus all \
     -v {q_dir}:/work {q_image}$quoted /work/$rid"
scp -q -- {q_target}:{q_dir}/"$rid" "$out"
ssh -o BatchMode=yes -- {q_target} "rm -f {q_dir}/$rid"
"""
    with tempfile.NamedTemporaryFile(
            "w", suffix=".sh", prefix="ffmpeg-gpu-", delete=False) as handle:
        handle.write(script)
    os.chmod(handle.name, 0o755)
    return handle.name


def _save_animation(anim: Any, fig: Any, output_dir: str, stem: str,
                    make_gif: bool = True, *, fps: float | None = None,
                    gif_fps: int | None = None,
                    poster_ss: float | None = None) -> None:
    """Write *anim* to WebM (always), its poster JPEG and, for English, a GIF.

    The GIF is derived from the just-written WebM with an ffmpeg palette pass
    so the GitHub docs get a compact, self-contained loop; the site embeds the
    WebM directly. Every WebM variant also gets a ``_poster.jpg`` still (see
    :func:`_extract_poster`) so the site ``<video>`` embeds can defer loading
    (``preload="none"``) behind a meaningful frame. ``fps`` overrides the
    shared WebM rate and ``gif_fps`` the GIF sampling rate (long clips drop
    GIF frames to stay within GitHub-friendly file sizes). ``savefig.bbox``
    is forced to ``standard`` so every frame keeps the same canvas size (a
    ``tight`` box would jitter and break the encoder).
    """
    import subprocess

    from matplotlib.animation import FFMpegWriter

    webm = _anim_path(output_dir, stem, "webm")
    rate = _ANIM_FPS if fps is None else fps
    # Matplotlib types the writer rate as an int, but the writer only
    # formats it into ffmpeg's ``-r`` and a fractional rate is legal there.
    # A clip whose capture stride was cut by a non-integer factor plays at
    # the matching fractional rate so its pacing stays exact.
    rate_arg = cast(int, rate)
    gpu = _gpu_encode_target()
    saved = False
    if gpu is not None:
        wrapper = _write_gpu_ffmpeg_wrapper(gpu["target"], gpu["image"],
                                            gpu["workdir"])
        try:
            writer = FFMpegWriter(fps=rate_arg, codec="av1_nvenc",
                                  extra_args=_av1_nvenc_extra_args())
            with plt.rc_context({"savefig.bbox": "standard",
                                 "animation.ffmpeg_path": wrapper}):
                anim.save(webm, writer=writer, dpi=_ANIM_DPI,
                          savefig_kwargs={"facecolor": fig.get_facecolor()})
            saved = True
        except (RuntimeError, subprocess.CalledProcessError, OSError) as exc:
            print(f"  [gpu-encode] {stem}: {exc}; falling back to VP9")
        finally:
            os.remove(wrapper)
    if not saved:
        writer = FFMpegWriter(fps=rate_arg, codec="libvpx-vp9",
                              extra_args=_vp9_extra_args())
        with plt.rc_context({"savefig.bbox": "standard"}):
            anim.save(webm, writer=writer, dpi=_ANIM_DPI,
                      savefig_kwargs={"facecolor": fig.get_facecolor()})
    _extract_poster(webm, poster_ss)
    made_gif = False
    if make_gif and _LANG == "en":
        gif = _anim_path(output_dir, stem, "gif")
        palette = os.path.join(output_dir, f".{stem}{_FILENAME_SUFFIX}_pal.png")
        vf = (f"fps={_GIF_FPS if gif_fps is None else gif_fps},"
              f"scale={_GIF_SCALE}:-1:flags=lanczos")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-vf",
             f"{vf},palettegen=max_colors={_GIF_COLORS}:stats_mode=diff",
             "-update", "1", palette],
            check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-i", palette,
             "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
             "-loop", "0", gif], check=True)
        os.remove(palette)
        made_gif = True
    plt.close(fig)
    theme = "dark" if _FILENAME_SUFFIX else "light"
    print(f"  {stem} [{_LANG} {theme}] -> webm + poster"
          + (" + gif" if made_gif else ""))


def animate_time_weighting_ballistics(output_dir: str) -> None:
    """A tone burst through the IEC 61672-1 RC detector: the capacitor fills
    and drains while the F/S/I meter needles follow their own ballistics."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    from phonometry import time_weighting

    T = _translate_str
    fs = 8000
    t = np.linspace(0, 4.0, int(fs * 4.0), endpoint=False)
    # A steady 250 Hz tone burst (unit amplitude) from 1.0 s to 2.5 s. The
    # carrier is high enough that even the 35 ms Impulse detector smooths the
    # squared ripple, so each detector shows its own clean rise and decay
    # (a noise burst would drown the ballistics in fluctuation).
    x = np.zeros_like(t)
    on = (t >= 1.0) & (t < 2.5)
    x[on] = np.sin(2 * np.pi * 250 * t[on])
    # Each detector is normalized to its own steady reading of the burst, so
    # the clip compares pure ballistics: a real meter shows the same level on
    # F, S and I for a steady tone (the asymmetric Impulse kernel otherwise
    # rides the carrier ripple toward its peaks and would sit ~2.6 dB high).
    i_ss = int(2.45 * fs)   # inside the burst, all detectors settled
    fast = time_weighting(x, fs, mode="fast")
    slow = time_weighting(x, fs, mode="slow")
    imp = time_weighting(x, fs, mode="impulse")
    fast /= fast[i_ss]
    slow /= 0.5             # Slow has not settled by 2.45 s; use the tone MS
    imp /= imp[i_ss]
    # Slope of the Fast output for the charging/draining indicator, smoothed
    # over 50 ms so the residual 500 Hz detector ripple cannot flip its sign.
    kernel = np.ones(400) / 400.0
    dfast = np.gradient(np.convolve(fast, kernel, mode="same"), t)
    col_imp = "#7e57c2"

    fig = _anim_figure()
    fig.suptitle(T("Time-weighting ballistics (IEC 61672-1)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35],
                          height_ratios=[1.0, 1.15])
    ax_s = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax_s, (0.0, 10.0), (0.0, 10.0))
    ax_g = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_g, (0.0, 3.4), (-0.55, 0.68), equal=True)
    ax_t = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_t)

    # --- schematic: input strip feeding the square-law + RC detector ------
    ax_s.text(5.0, 10.0, T("RC exponential detector"), ha="center", va="top",
              color=COLOR_FG, fontsize=11, fontweight="bold")
    # Display carrier slowed to ~4 Hz so the burst reads as a waveform, not
    # a filled block (the detectors run on the real 250 Hz burst).
    x_vis = np.where(on, np.sin(2 * np.pi * 4.0 * t), 0.0)
    ax_s.plot(0.6 + 8.8 * t[::4] / 4.0, 8.2 + 0.7 * x_vis[::4],
              color=COLOR_PRIMARY, lw=1.0, alpha=0.9)
    ax_s.text(9.4, 9.05, T("input x(t)"), ha="right", va="bottom",
              color=COLOR_FG, fontsize=9)
    (strip_cur,) = ax_s.plot([0.6, 0.6], [7.4, 9.0], color=COLOR_FG,
                             lw=1.2, alpha=0.7)
    ax_s.annotate("", xy=(0.6, 6.0), xytext=(0.6, 7.3),
                  arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.2})
    wire = {"color": COLOR_FG, "lw": 1.4}
    ax_s.plot([0.6, 1.3], [5.6, 5.6], **wire)
    ax_s.add_patch(FancyBboxPatch((1.3, 5.0), 1.4, 1.2,
                                  boxstyle="round,pad=0.04",
                                  facecolor="none", edgecolor=COLOR_FG, lw=1.4))
    ax_s.text(2.0, 5.6, "$x^2$", ha="center", va="center", color=COLOR_FG,
              fontsize=13)
    ax_s.text(2.0, 4.55, T("square-law rectifier"), ha="center", va="top",
              color=COLOR_FG, fontsize=8.5)
    ax_s.plot([2.7, 3.2], [5.6, 5.6], **wire)
    _draw_resistor(ax_s, 3.2, 5.0, 5.6)
    ax_s.text(4.1, 6.1, "R", ha="center", va="bottom", color=COLOR_FG,
              fontsize=11)
    ax_s.plot([5.0, 8.1], [5.6, 5.6], **wire)
    ax_s.plot([6.3], [5.6], marker="o", ms=4, color=COLOR_FG)
    ax_s.annotate("", xy=(9.0, 5.6), xytext=(8.1, 5.6),
                  arrowprops={"arrowstyle": "-|>", "color": COLOR_FG, "lw": 1.4})
    ax_s.text(8.55, 6.0, r"$10\,\log_{10}$", ha="center", va="bottom",
              color=COLOR_FG, fontsize=9)
    ax_s.text(9.15, 5.6, "dB", ha="left", va="center", color=COLOR_FG,
              fontsize=10)
    # capacitor drawn as a tank so its state of charge is visible
    ax_s.plot([6.3, 6.3], [5.6, 4.7], **wire)
    ax_s.plot([5.6, 7.0], [4.7, 4.7], color=COLOR_FG, lw=2.2)
    ax_s.plot([5.6, 7.0], [3.2, 3.2], color=COLOR_FG, lw=2.2)
    ax_s.text(7.2, 3.95, "C", ha="left", va="center", color=COLOR_FG,
              fontsize=11)
    # tank walls, animated fill and a liquid-level line so partial charge
    # reads as partial even in a still frame
    for xw in (5.6, 7.0):
        ax_s.plot([xw, xw], [3.2, 4.7], color=COLOR_GRID, lw=1.0, ls=":")
    cap_fill = Rectangle((5.62, 3.24), 1.36, 0.0, facecolor=COLOR_PRIMARY,
                         alpha=0.8 if _FILENAME_SUFFIX else 0.55,
                         edgecolor="none")
    ax_s.add_patch(cap_fill)
    (cap_level,) = ax_s.plot([5.62, 6.98], [3.24, 3.24], color=COLOR_PRIMARY,
                             lw=2.0)
    ax_s.plot([6.3, 6.3], [3.2, 2.62], **wire)
    for gw, gy in ((0.7, 2.62), (0.44, 2.48), (0.18, 2.34)):
        ax_s.plot([6.3 - gw / 2, 6.3 + gw / 2], [gy, gy], color=COLOR_FG,
                  lw=1.4)
    ax_s.text(6.3, 2.05, T("stored charge (Fast shown)"), ha="center",
              va="top", color=COLOR_FG, fontsize=8.5)
    charge_arrow = _make_arrow(ax_s, COLOR_TERTIARY, scale=12.0)
    charge_txt = ax_s.text(4.95, 3.9, "", ha="right", va="center",
                           color=COLOR_FG, fontsize=8.5)
    ax_s.text(5.0, 1.3, T("τ = RC sets attack and decay"), ha="center",
              va="center", color=COLOR_FG, fontsize=9)
    ax_s.text(5.0, 0.6, "F 125 ms · S 1000 ms · I 35/1500 ms", ha="center",
              va="center", color=COLOR_FG, fontsize=8.5, alpha=0.85)

    # --- meter gauges + response traces -----------------------------------
    # One shared 0..1.2 scale, spelled out on the first dial only (the
    # endpoint labels of adjacent dials would otherwise collide).
    gauges = [_make_gauge(ax_g, 0.6, 0.0, 0.5, "F", COLOR_PRIMARY,
                          lo="0", hi="1.2"),
              _make_gauge(ax_g, 1.7, 0.0, 0.5, "S", COLOR_SECONDARY),
              _make_gauge(ax_g, 2.8, 0.0, 0.5, "I", col_imp)]
    ax_t.set_xlim(0.5, 4.0)
    ax_t.set_ylim(0, 1.25)
    ax_t.axvspan(1.0, 2.5, color=COLOR_GRID, alpha=0.4, lw=0)
    ax_t.text(1.75, 1.19, T("tone burst"), ha="center", va="top",
              color=COLOR_FG, fontsize=8.5, alpha=0.8)
    (l_f,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.0,
                       label=T("Fast (125 ms)"))
    (l_s,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.0,
                       label=T("Slow (1000 ms)"))
    (l_i,) = ax_t.plot([], [], color=col_imp, lw=1.7, ls="-.",
                       label=T("Impulse (35 ms / 1.5 s)"))
    cursor = ax_t.axvline(0.5, color=COLOR_FG, lw=1.0, alpha=0.45)
    ax_t.set_xlabel(T("Time [s]"))
    ax_t.set_ylabel(T("Mean-square response (normalized)"), fontsize=8)
    ax_t.legend(loc="upper right", fontsize=7.5)

    tmin, tmax = 0.5, 4.0
    # Sweep the burst-and-decay, then hold the settled three-trace comparison.
    sweep = _ANIM_FRAMES - _ANIM_HOLD

    def update(k: int) -> tuple[Any, ...]:
        tc = tmin + (tmax - tmin) * min(k, sweep - 1) / (sweep - 1)
        i = max(0, min(t.size - 1, round(tc * fs)))
        xc = 0.6 + 8.8 * tc / 4.0
        strip_cur.set_data([xc, xc], [7.4, 9.0])
        level = 3.24 + 1.42 * min(float(fast[i]), 1.02)
        cap_fill.set_height(level - 3.24)
        cap_level.set_data([5.62, 6.98], [level, level])
        slope = float(dfast[i])
        if abs(slope) > 0.05:
            charging = slope > 0
            charge_arrow.set_visible(True)
            y0, y1 = (4.55, 3.35) if charging else (3.35, 4.55)
            charge_arrow.set_positions((5.2, y0), (5.2, y1))
            charge_arrow.set_color(COLOR_TERTIARY if charging
                                   else COLOR_SECONDARY)
            charge_txt.set_text(T("charging") if charging else T("draining"))
        else:
            charge_arrow.set_visible(False)
            charge_txt.set_text("")
        arts: list[Any] = [strip_cur, cap_fill, cap_level, charge_arrow,
                           charge_txt]
        for gauge, val in zip(gauges, (fast[i], slow[i], imp[i]), strict=True):
            # Normalize onto the dial's 0..1.2 scale so the needle position
            # matches the endpoint labels and the numeric readout.
            arts += _set_gauge(gauge, float(val) / 1.2, T(f"{val:.2f}"))
        m = t <= tc
        l_f.set_data(t[m], fast[m])
        l_s.set_data(t[m], slow[m])
        l_i.set_data(t[m], imp[m])
        cursor.set_xdata([tc, tc])
        arts += [l_f, l_s, l_i, cursor]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_time_weighting")


def animate_onset_detection(output_dir: str) -> None:
    """The NT ACOU 112 detector scanning L_AF with a magnifier; the
    OR -> LD -> P -> KI decision chain lights up once the onset is found."""
    from matplotlib.patches import Ellipse

    from phonometry import impulse_adjustment, predicted_prominence

    T = _translate_str
    fs = 500
    t = np.linspace(0, 3.0, int(fs * 3.0), endpoint=False)
    ls, le = 55.0, 85.0    # start and end level of the onset, dB
    t0, rise = 1.0, 0.15   # onset at 1.0 s, lasting 150 ms
    laf = np.full_like(t, ls)
    ramp = (t >= t0) & (t < t0 + rise)
    laf[ramp] = ls + (le - ls) * 0.5 * (1 - np.cos(np.pi * (t[ramp] - t0) / rise))
    after = t >= t0 + rise
    laf[after] = ls + (le - ls) * np.exp(-(t[after] - t0 - rise) / 0.6)
    grad = np.gradient(laf, t)
    onset_rate = (le - ls) / rise      # 200 dB/s
    level_diff = le - ls               # 30 dB
    prom = float(predicted_prominence(onset_rate, level_diff))
    ki = float(impulse_adjustment(prom))
    is_onset = grad > 10.0             # clauses 4.5-4.7

    fig = _anim_figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 0.9])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.55, 3.0)
    ax.set_ylim(42, 100)
    ax.plot(t, laf, color=COLOR_PRIMARY, lw=1.8,
            label=T("L_AF (A-weighted, Fast)"))
    (hot,) = ax.plot([], [], color=COLOR_SECONDARY, lw=4.0,
                     solid_capstyle="round", label=T("onset (> 10 dB/s)"))
    rx, ry = 0.13, 6.5
    lens = Ellipse((0.7, ls), width=2 * rx, height=2 * ry, facecolor="none",
                   edgecolor=COLOR_FG, lw=2.2, zorder=5)
    ax.add_patch(lens)
    (handle,) = ax.plot([], [], color=COLOR_FG, lw=3.2,
                        solid_capstyle="round", zorder=5)
    (tangent,) = ax.plot([], [], color=COLOR_FG, lw=1.6, ls="--", zorder=6)
    lens_txt = ax.text(0.7, ls, "", ha="center", va="bottom", color=COLOR_FG,
                       fontsize=9, zorder=6)
    ax.text(0.02, 0.05, T("detector: onset when dL/dt > 10 dB/s"),
            transform=ax.transAxes, ha="left", va="bottom", color=COLOR_FG,
            fontsize=9)
    ax.set_title(T("Impulse onset detection (NT ACOU 112)"), fontweight="bold")
    ax.set_xlabel(T("Time [s]"))
    ax.set_ylabel(T("A-weighted level L_AF [dB]"), fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    ax_b = fig.add_subplot(gs[1])
    _schematic_axes(ax_b, (0.0, 10.0), (0.0, 2.3))
    titles = [T("onset rate"), T("level difference"), T("prominence"),
              T("adjustment")]
    boxes = [_flow_box(ax_b, cx, 1.35, 1.95, 1.35, title)
             for cx, title in zip((1.5, 3.9, 6.3, 8.7), titles, strict=True)]
    for xa in (2.5, 4.9, 7.3):
        ax_b.annotate("", xy=(xa + 0.42, 1.35), xytext=(xa, 1.35),
                      arrowprops={"arrowstyle": "-|>", "color": COLOR_GRID,
                                  "lw": 1.6})
    verdict = ax_b.text(5.0, 0.28, "", ha="center", va="center",
                        color=COLOR_SECONDARY, fontsize=10, fontweight="bold")
    values = (f"OR = {onset_rate:.0f} dB/s", f"LD = {level_diff:.0f} dB",
              f"P = {prom:.1f}", f"KI = {ki:.1f} dB")

    # Nonuniform sweep: slow motion while the magnifier crosses the onset.
    # The sweep ends _ANIM_HOLD frames early; np.interp clamps past the last
    # knot, so the lit OR -> LD -> P -> KI chain and the verdict hold ~2 s.
    end_k = float(_ANIM_FRAMES - 1 - _ANIM_HOLD)
    knots_k = (0.0, 0.22 * end_k, 0.52 * end_k, end_k)
    knots_t = (0.62, 0.985, 1.19, 3.0)

    def update(k: int) -> tuple[Any, ...]:
        tc = float(np.interp(k, knots_k, knots_t))
        # Park the magnifier just short of the right edge so its lens and
        # handle stay fully inside the panel during the closing hold (tc
        # reaches 3.0 s, the xlim). The decision chain below still keys off
        # the true tc, so only the glyph position is clamped.
        tc_view = min(max(tc, 0.55 + rx), 3.0 - rx * 1.9)
        i = max(0, min(t.size - 1, round(tc_view * fs)))
        y0, g = float(laf[i]), float(grad[i])
        detecting = g > 10.0
        color = COLOR_SECONDARY if detecting else COLOR_FG
        lens.set_center((tc_view, y0))
        lens.set_edgecolor(color)
        dxl = rx * 0.72
        if abs(g) * dxl > ry * 0.72:
            dxl = ry * 0.72 / abs(g)
        tangent.set_data([tc_view - dxl, tc_view + dxl],
                         [y0 - g * dxl, y0 + g * dxl])
        tangent.set_color(color)
        handle.set_data([tc_view + rx * 0.75, tc_view + rx * 1.9],
                        [y0 - ry * 0.75, y0 - ry * 1.9])
        lens_txt.set_position((tc_view, y0 + ry * 1.35))
        lens_txt.set_text(T(f"dL/dt = {g:.0f} dB/s"))
        lens_txt.set_color(color)
        hot_m = (t <= tc) & is_onset
        hot.set_data(t[hot_m], laf[hot_m])
        arts: list[Any] = [lens, handle, tangent, lens_txt, hot]
        for j, (box, val) in enumerate(zip(boxes, values, strict=True)):
            if tc >= 1.3 + 0.4 * j:
                _light_box(box, T(val),
                           COLOR_SECONDARY if j == 3 else COLOR_PRIMARY,
                           fill=j == 3)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        verdict.set_text(T("add KI to the rating level") if tc >= 2.6 else "")
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_onset_detection")


def animate_instantaneous_intensity(output_dir: str) -> None:
    """A p-p probe with p/u phasors: the instantaneous intensity arrow flips
    while its running average settles to net flow (active) or zero (reactive)."""
    from matplotlib.patches import Circle

    T = _translate_str
    t = np.linspace(0, 3.0, 600)
    w = 2 * np.pi * 2.0
    # Progressive: p and u in phase -> p·u >= 0, non-zero mean (net flow).
    # Standing (at a point): p and u 90 deg out of phase -> p·u averages zero.
    cases = [
        (T("Progressive wave — active"), 0.0, T("p and u in phase")),
        (T("Standing wave — reactive"), np.pi / 2, T("p and u 90° apart")),
    ]
    fig = _anim_figure()
    fig.suptitle(T("Two-microphone p-p probe: instantaneous intensity p·u"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    dial_c, dial_r = (1.55, 3.05), 1.25
    i_axis_y, i_scale = 1.05, 2.35
    panels: list[dict[str, Any]] = []
    for col, (title, phi, caption) in enumerate(cases):
        ax_s = fig.add_subplot(gs[0, col])
        _schematic_axes(ax_s, (0.0, 10.0), (0.0, 4.9), equal=True)
        ax_s.set_title(title, fontsize=10.5, fontweight="bold")
        ax_s.add_patch(Circle(dial_c, dial_r, facecolor="none",
                              edgecolor=COLOR_GRID, lw=1.2))
        p_ph = _make_arrow(ax_s, COLOR_PRIMARY, scale=11.0)
        u_ph = _make_arrow(ax_s, COLOR_TERTIARY, scale=11.0)
        ax_s.text(dial_c[0] - dial_r - 0.15, dial_c[1] + dial_r - 0.15, "p",
                  color=COLOR_PRIMARY, fontsize=11, ha="right", va="center",
                  fontweight="bold")
        ax_s.text(dial_c[0] - dial_r - 0.15, dial_c[1] - dial_r + 0.15, "u",
                  color=COLOR_TERTIARY, fontsize=11, ha="right", va="center",
                  fontweight="bold")
        # Tucked up under the phasor dial so it clears the intensity
        # number-line below (the longer Spanish caption otherwise crowds it).
        ax_s.text(dial_c[0], 1.72, caption, ha="center", va="top",
                  color=COLOR_FG, fontsize=8.0)
        _draw_mic(ax_s, 4.6, 3.4, direction=1, size=1.05, label="$p_1$")
        _draw_mic(ax_s, 7.4, 3.4, direction=-1, size=1.05, label="$p_2$")
        ax_s.annotate("", xy=(6.85, 2.6), xytext=(5.15, 2.6),
                      arrowprops={"arrowstyle": "<->", "color": COLOR_FG,
                                  "lw": 1.0})
        ax_s.text(6.0, 2.38, T("spacer Δr"), ha="center", va="top",
                  color=COLOR_FG, fontsize=8.5)
        ax_s.plot([3.4, 8.6], [i_axis_y, i_axis_y], color=COLOR_GRID,
                  lw=1.0, ls="--")
        ax_s.plot([6.0, 6.0], [i_axis_y - 0.15, i_axis_y + 0.15],
                  color=COLOR_FG, lw=1.0, alpha=0.7)
        ax_s.text(6.0, i_axis_y + 0.22, "0", ha="center", va="bottom",
                  color=COLOR_FG, fontsize=8, alpha=0.8)
        ax_s.text(3.15, i_axis_y, "−", ha="right", va="center",
                  color=COLOR_FG, fontsize=10, alpha=0.8)
        ax_s.text(8.85, i_axis_y, "+", ha="left", va="center",
                  color=COLOR_FG, fontsize=10, alpha=0.8)
        ax_s.text(3.4, 1.55, T("I(t) = p·u"), ha="left", va="bottom",
                  color=COLOR_SECONDARY, fontsize=9)
        i_arrow = _make_arrow(ax_s, COLOR_SECONDARY, scale=16.0)
        (mean_marker,) = ax_s.plot([], [], marker="^", ms=7, color=COLOR_FG)
        mean_lab = ax_s.text(6.0, 0.42, "⟨I⟩", ha="center", va="top",
                             color=COLOR_FG, fontsize=8.5)

        ax_tr = fig.add_subplot(gs[1, col])
        _grid_axes(ax_tr)
        ax_tr.set_xlim(0, 3.0)
        ax_tr.set_ylim(-1.15, 1.15)
        p_sig = np.cos(w * t)
        u_sig = np.cos(w * t - phi)
        ax_tr.plot(t, p_sig, color=COLOR_PRIMARY, alpha=0.55, lw=1.1,
                   label=T("pressure p"))
        ax_tr.plot(t, u_sig, color=COLOR_TERTIARY, alpha=0.55, lw=1.1,
                   label=T("velocity u"))
        (iline,) = ax_tr.plot([], [], color=COLOR_SECONDARY, lw=2.0,
                              label=T("intensity p·u"))
        mline = ax_tr.axhline(0.0, color=COLOR_FG, ls="--", lw=1.1, alpha=0.7)
        txt = ax_tr.text(0.5, 0.02, "", transform=ax_tr.transAxes,
                         ha="center", va="bottom", family="monospace",
                         fontsize=10, color=COLOR_FG)
        ax_tr.set_xlabel(T("Time [s]"))
        if col == 0:
            ax_tr.set_ylabel(T("amplitude (normalized)"), fontsize=9)
        ax_tr.legend(loc="upper right", fontsize=7.5)
        panels.append({"ax": ax_tr, "phi": phi, "I": p_sig * u_sig,
                       "p_ph": p_ph, "u_ph": u_ph, "i_arrow": i_arrow,
                       "mean_marker": mean_marker, "mean_lab": mean_lab,
                       "iline": iline, "mline": mline, "txt": txt,
                       "fill": None})

    # Six carrier periods sweep by, then the settled averages (net flow vs
    # zero) hold so the active/reactive contrast can be read.
    sweep = _ANIM_FRAMES - _ANIM_HOLD

    def update(k: int) -> tuple[Any, ...]:
        tc = 3.0 * min(k, sweep - 1) / (sweep - 1)
        idx = max(1, int(np.searchsorted(t, tc)))
        # Average over whole periods once one is complete, so the reactive
        # mean pins to exactly zero instead of hovering near it.
        n_per = np.floor(tc * 2.0)
        idx_mean = idx if n_per < 1 else max(1, int(np.searchsorted(
            t, float(n_per) / 2.0)))
        ph = w * tc
        cx, cy = dial_c
        arts: list[Any] = []
        for pn in panels:
            pv = float(np.cos(ph))
            uv = float(np.cos(ph - pn["phi"]))
            pn["p_ph"].set_positions(
                (cx, cy), (cx + 1.05 * dial_r * np.cos(ph),
                           cy + 1.05 * dial_r * np.sin(ph)))
            pn["u_ph"].set_positions(
                (cx, cy), (cx + 0.88 * dial_r * np.cos(ph - pn["phi"]),
                           cy + 0.88 * dial_r * np.sin(ph - pn["phi"])))
            ival = pv * uv
            tip = 6.0 + i_scale * ival
            if abs(tip - 6.0) < 1e-3:
                tip = 6.0 + 1e-3
            pn["i_arrow"].set_positions((6.0, i_axis_y), (tip, i_axis_y))
            mean = float(np.mean(pn["I"][:idx_mean]))
            if abs(mean) < 5e-3:
                mean = 0.0     # avoid a distracting "-0.00" readout
            xm = 6.0 + i_scale * mean
            pn["mean_marker"].set_data([xm], [i_axis_y - 0.22])
            pn["mean_lab"].set_position((xm, i_axis_y - 0.5))
            pn["iline"].set_data(t[:idx], pn["I"][:idx])
            if pn["fill"] is not None:
                pn["fill"].remove()
            pn["fill"] = pn["ax"].fill_between(t[:idx], 0.0, pn["I"][:idx],
                                               color=COLOR_SECONDARY,
                                               alpha=0.22)
            pn["mline"].set_ydata([mean, mean])
            pn["txt"].set_text(T(f"⟨p·u⟩ = {mean:+.2f}"))
            arts += [pn["p_ph"], pn["u_ph"], pn["i_arrow"],
                     pn["mean_marker"], pn["mean_lab"],
                     pn["iline"], pn["fill"], pn["mline"], pn["txt"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_instantaneous_intensity")


def animate_schroeder(output_dir: str) -> None:
    """Backward integration of p²(t): the tail energy fills up on one axis
    while the decay curve and its T20/T30 fits emerge on a companion axis."""
    from matplotlib.patches import Patch

    from phonometry import decay_curve, room_parameters
    from phonometry.room.room_acoustics import _T20_RANGE, _T30_RANGE, _onset_index

    T = _translate_str
    fs, reverb_t = 48000, 1.2
    rng = np.random.default_rng(2026)
    t = np.arange(int(2.0 * fs)) / fs
    ir = (rng.standard_normal(t.size) * np.exp(-6.9077 * t / reverb_t)
          + rng.standard_normal(t.size) * 10.0 ** (-45.0 / 20.0))
    time, level = decay_curve(ir, fs)
    res = room_parameters(ir, fs, limits=None)
    t20, t30 = float(res.t20[0]), float(res.t30[0])
    p2 = ir.astype(np.float64) ** 2
    p2 = p2[_onset_index(p2):]
    t_raw = np.arange(p2.size) / fs
    raw_db = 10.0 * np.log10(np.maximum(p2, p2.max() * 1e-12) / p2.max())

    # Regression lines drawn once the sweep finishes: slope -60/T over each
    # evaluation range, extended to the -60 dB crossing at t = T.
    def _fit(rng_db: tuple[float, float]) -> tuple[float, float] | None:
        mask = (level <= -rng_db[0]) & (level >= -rng_db[1])
        if int(mask.sum()) < 2:
            return None
        slope, intercept = np.polyfit(time[mask], level[mask], 1)
        if slope >= 0.0:   # a non-decaying fit has no meaningful -60 dB crossing
            return None
        return float(slope), float(intercept)

    fits = []
    for rng_db, color, style, key in (
        (_T20_RANGE, COLOR_SECONDARY, "--", "T20 fit"),
        (_T30_RANGE, COLOR_TERTIARY, "-.", "T30 fit"),
    ):
        fit = _fit(rng_db)
        if fit is None:
            continue
        slope, intercept = fit
        fits.append((color, style, key,
                     (-intercept / slope, (-60.0 - intercept) / slope)))
    tmax = float(time.max())
    xmax = max([tmax, *(f[3][1] for f in fits)]) * 1.03
    # Tail-energy fraction E(t)/E(0) on the onset-trimmed squared IR, and a
    # display-decimated copy of the raw level for the mechanism panel.
    cum = np.cumsum(p2[::-1])[::-1]
    e_frac = cum / cum[0]
    ds = slice(None, None, 8)

    fig = _anim_figure()
    fig.suptitle(T("Schroeder backward integration (ISO 3382)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.3])
    ax_e = fig.add_subplot(gs[0])
    _grid_axes(ax_e)
    ax_e.set_xlim(0, xmax)
    ax_e.set_ylim(-65, 3)
    ax_e.tick_params(labelbottom=False)
    sweep_max = float(t_raw[-1])   # start the front at the very end of p²
    fill_alpha = 0.5 if _FILENAME_SUFFIX else 0.35
    ax_e.plot(t_raw[ds], raw_db[ds], color="gray", alpha=0.4, lw=0.6)
    e_fill = {"art": None}
    front_e = ax_e.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    # Sits well below the upper-right legend (a higher anchor ran under it).
    front_txt = ax_e.text(sweep_max, -22.0, T("← integrate from the tail"),
                          ha="left", va="top", color=COLOR_FG, fontsize=9)
    e_txt = ax_e.text(0.02, 0.06, "", transform=ax_e.transAxes, ha="left",
                      va="bottom", family="monospace", fontsize=10,
                      color=COLOR_FG)
    ax_e.set_ylabel(T("Level [dB]"), fontsize=9)
    ax_e.legend(handles=[
        Patch(facecolor="gray", alpha=0.4,
              label=T("squared impulse response p²")),
        Patch(facecolor=COLOR_SECONDARY, alpha=fill_alpha,
              label=T("tail energy") + r"  $E(t)=\int_t^{\infty}p^2\,d\tau$"),
    ], loc="upper right", fontsize=8.5)

    ax_d = fig.add_subplot(gs[1], sharex=ax_e)
    _grid_axes(ax_d)
    ax_d.set_ylim(-65, 3)
    (curve,) = ax_d.plot([], [], color=COLOR_PRIMARY, lw=2.4,
                         label=T("Schroeder decay curve"))
    # The fit lines only exist in the closing frames, so they are announced
    # by the color-matched T20/T30 readouts instead of a premature legend.
    fit_lines = []
    for color, style, _key, span in fits:
        (fl,) = ax_d.plot([], [], color=color, ls=style, lw=1.7)
        fit_lines.append((fl, span))
    front_d = ax_d.axvline(sweep_max, color=COLOR_FG, lw=1.3, alpha=0.55)
    ax_d.text(0.02, 0.08, r"$L(t) = 10\,\log_{10}\,E(t)\,/\,E(0)$",
              transform=ax_d.transAxes, ha="left", va="bottom",
              color=COLOR_FG, fontsize=10)
    ann_fits = [
        ax_d.text(0.56, 0.16, "", transform=ax_d.transAxes, ha="left",
                  va="bottom", family="monospace", fontsize=11,
                  color=COLOR_SECONDARY),
        ax_d.text(0.56, 0.05, "", transform=ax_d.transAxes, ha="left",
                  va="bottom", family="monospace", fontsize=11,
                  color=COLOR_TERTIARY),
    ]
    ax_d.set_xlabel(T("Time [s]"))
    ax_d.set_ylabel(T("Level [dB]"), fontsize=9)
    ax_d.legend(loc="upper right", fontsize=8.5)

    # Sweep for 80% of the frames; the T20/T30 fits and readouts then hold
    # for the remaining ~2.4 s so the verdict can be read.
    reveal = int(_ANIM_FRAMES * 0.8)

    def update(k: int) -> tuple[Any, ...]:
        xf = sweep_max * (1.0 - k / (reveal - 1)) if k < reveal else 0.0
        m = time >= xf
        curve.set_data(time[m], level[m])
        front_e.set_xdata([xf, xf])
        front_d.set_xdata([xf, xf])
        if e_fill["art"] is not None:
            e_fill["art"].remove()
        mr = t_raw[ds] >= xf
        e_fill["art"] = ax_e.fill_between(t_raw[ds][mr], -65.0, raw_db[ds][mr],
                                          color=COLOR_SECONDARY,
                                          alpha=fill_alpha, lw=0)
        idx = min(int(np.searchsorted(t_raw, xf)), e_frac.size - 1)
        e_txt.set_text(T(f"remaining energy: {100.0 * e_frac[idx]:.1f} %"))
        front_txt.set_position((min(xf, xmax * 0.72) + 0.012 * xmax, -22.0))
        front_txt.set_visible(k < reveal)
        arts: list[Any] = [curve, front_e, front_d, e_fill["art"], e_txt,
                           front_txt, *ann_fits]
        if k >= reveal:
            for (fl, (t_lo, t_hi)), ann, name, val in zip(
                    fit_lines, ann_fits, ("T20", "T30"), (t20, t30),
                    strict=False):
                fl.set_data([t_lo, t_hi], [0.0, -60.0])
                ann.set_text(T(f"{name} = {val:.2f} s"))
                arts.append(fl)
        else:
            for ann in ann_fits:
                ann.set_text("")
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_schroeder")


# Room-modes timeline. Mesh: the highest carrier is the off-mode drive at
# 91.2 Hz (lambda = 3.76 m, lambda / 8 = 0.47 m) and the smallest geometric
# dimension is the 3.5 m room side (3.5 / 4 = 0.88 m), so the rule
# dx = min(0.88, 0.47) m allows 0.47 m; dx = 0.01 m sits 47 times finer,
# because the mode map has to be drawn, not just resolved.
# Flight: the source at (0.25, 0.25) m reaches the deepest reflector -- the
# opposite corner (5, 3.5) -- after 5.755 m and the farthest visible point
# (the near corner) 6.103 m later, so 11.859 m / 343 m/s x 1.2 = 41.5 ms.
# Sampling: 18 solver steps per frame = 222.6 us gives 49.3 frames per
# 91.2 Hz period (>= 48). The captured window keeps the committed clip's
# full 350 ms -- the flight floor is only a floor, and this clip needs
# 8.4 times it (3.5 amplitude time constants of the T60 = 0.7 s room,
# tau = T60 / 6.9077 = 101 ms) because the resonance has to be seen
# *growing* all the way into its settled mode map. 1 572 frames cover that
# window, played at 13/3 of the shared 20 fps -- exactly the factor by
# which the stride shrank from the committed clip's 78 solver steps per
# frame -- so the pacing stays the 19.30 ms of simulation per second of
# playback the committed clip had, and the clip runs 18.1 s (it ran 18 s).
_ROOM_EVERY = 18
_ROOM_FRAMES = 1572
#: 13/3 of the shared rate, matching the stride cut (see the note above).
_ROOM_FPS = _ANIM_FPS * 13.0 / 3.0


@lru_cache(maxsize=1)
def _room_mode_fields(
        n_frames: int = _ROOM_FRAMES) -> tuple[Any, Any, Any, float, float]:
    """Run the two FDTD room simulations once per process (all variants).

    Returns instantaneous-pressure frames, running-RMS frames (both float32,
    stacked ``(2, n_frames, ny, nx)``), the frame times, and the on-mode and
    off-mode drive frequencies. ``n_frames`` sets the captured window at the
    fixed ``_ROOM_EVERY`` capture stride (see the timeline note above).
    """
    import fdtd2d

    lx, ly, c0 = 5.0, 3.5, 343.0
    dx = 0.01                                    # 500 x 350 cells
    ny, nx = round(ly / dx), round(lx / dx)
    f_mode = 0.5 * c0 * float(np.hypot(2 / lx, 1 / ly))   # (2,1) ~ 84.3 Hz
    f_next = 0.5 * c0 * (2 / ly)                          # (0,2) = 98 Hz
    f_off = 0.5 * (f_mode + f_next)              # between resonances
    t60 = 0.7
    every = _ROOM_EVERY
    steps = every * n_frames
    p_all, r_all = [], []
    times = np.zeros(0)
    for f in (f_mode, f_off):
        sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx), damping=6.9077 / t60)
        sim.add_source(fdtd2d.CWSource(ix=25, iy=25, frequency=f,
                                       ramp_cycles=2.0))
        # Running mean square with a two-period time constant: the pattern
        # (the mode map) builds up as the resonance settles.
        beta = float(np.exp(-sim.dt * f / 2.0))
        ms = np.zeros_like(sim.p)
        ps: list[Any] = []
        rs: list[Any] = []
        ts: list[float] = []
        for i in range(steps):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if (i + 1) % every == 0 and len(ps) < n_frames:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                rs.append(np.sqrt(ms[::2, ::2]).astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        r_all.append(np.stack(rs))
        times = np.asarray(ts)
    return np.stack(p_all), np.stack(r_all), times, f_mode, f_off


def animate_fdtd_room_modes(output_dir: str) -> None:
    """On-mode vs off-mode CW drive in a rigid 5 x 3.5 m room (2D FDTD):
    on resonance the (2,1) standing-wave pattern grows until it dominates
    the field; off resonance the forced response stays weak and never
    organises into that nodal structure."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, r_all, times, f_mode, f_off = _room_mode_fields()
    half = p_all.shape[1] // 2
    vmax_p = float(np.quantile(np.abs(p_all[0][half:]), 0.995))
    vmax_r = float(np.quantile(r_all[0][-1], 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Room modes in a rigid 5 m × 3.5 m room (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T(f"On the (2,1) mode: {f_mode:.1f} Hz"),
              T(f"Off mode: {f_off:.1f} Hz")]
    ims: list[Any] = []
    for col in range(2):
        ax_p = fig.add_subplot(gs[0, col])
        ax_p.grid(False)
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap=CMAP_FIELD,
                           vmin=-vmax_p, vmax=vmax_p, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
        ax_p.plot([0.25], [0.25], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
        ax_p.text(0.45, 0.22, T("source"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.tick_params(labelsize=7, labelbottom=False)
        ax_r = fig.add_subplot(gs[1, col])
        ax_r.grid(False)
        im_r = ax_r.imshow(r_all[col][0], origin="lower",
                           extent=(0.0, 5.0, 0.0, 3.5), cmap="magma",
                           vmin=0.0, vmax=vmax_r, interpolation="bilinear")
        ax_r.tick_params(labelsize=7)
        ax_r.set_xlabel("x [m]", fontsize=8)
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS pressure (mode map)"), fontsize=9)
            for xn in (1.25, 3.75):
                ax_r.axvline(xn, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.axhline(1.75, color="white", ls="--", lw=1.0, alpha=0.75)
            ax_r.text(0.12, 3.3, T("nodal lines (2,1)"), color="white",
                      fontsize=8, ha="left", va="top")
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(0.12, 3.3, T("same color scale"), color="white",
                      fontsize=8, ha="left", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.985, 0.955, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(r_all[col][k])
        t_txt.set_text(T(f"t = {times[k] * 1000.0:3.0f} ms"))
        return (*ims, t_txt)

    # Own frame budget and rate (see the _ROOM_FRAMES timeline note):
    # 1 572 frames at 49.3 per acoustic period, played at 260/3 fps. The
    # GitHub GIF samples this long clip at a reduced rate so the
    # palette-quantized file stays well under 4 MB.
    _render_clip(fig, update, output_dir, "anim_fdtd_room_modes",
                 frames=int(p_all.shape[1]), fps=_ROOM_FPS, gif_fps=5)


# ---------------------------------------------------------------------------
# FDTD wave-field clips (barrier, ground effect, SOFAR duct, diffuser).
# Every simulation runs once per process behind an lru_cache and its frames
# are re-rendered for the four language x theme variants; each clip keeps
# >= 48 captured frames per acoustic period of its highest carrier, raising
# its playback rate with the capture stride so the wavefronts read as
# continuous motion at the pacing the clip was composed for.
# ---------------------------------------------------------------------------


def _fdtd_cw_capture(
        sim: Any, frequency: float, every: int, n_frames: int,
        decimate: int = 2) -> tuple[Any, Any, Any, Any]:
    """Drive a CW simulation and capture instantaneous + running-RMS frames.

    The running mean square uses a two-period time constant, so the RMS map
    (the lobe/shadow pattern) builds up as the field settles. Returns the
    float32 frame stacks (decimated), the frame times and the final
    full-resolution RMS map for physics probes.
    """
    beta = float(np.exp(-sim.dt * frequency / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[Any] = []
    rs: list[Any] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::decimate, ::decimate].astype(np.float32))
            rs.append(np.sqrt(ms[::decimate, ::decimate]).astype(np.float32))
            ts.append(sim.time)
    return np.stack(ps), np.stack(rs), np.asarray(ts), np.sqrt(ms)


def _rms_to_db(rms_frames: Any, *, floor: float = -40.0) -> Any:
    """RMS frame stack -> dB re the final frame's maximum, clipped at floor."""
    ref = float(rms_frames[-1].max())
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms_frames / ref)
    return np.clip(db, floor, 0.0).astype(np.float32)


# --- weak-field display gain ----------------------------------------------
# A transmitted or shadowed field can sit tens of dB below the incident one
# that sets the colour scale, and a single linear diverging ramp cannot show
# both: give the quiet side half the ramp and the loud side saturates into a
# flat slab, hiding the wavefronts the clip exists to show. Compressing the
# ramp instead (signed log, asinh, gamma) has the same ceiling -- a factor of
# 250 between the two sides is a factor of 250 whatever the transfer curve --
# and it costs the loud side its shape as well.
# The treatment used across the clips is therefore a per-region display gain:
# the quiet region is drawn amplified by a fixed factor picked from the field
# itself, so each region uses the full ramp, the *shape* of both fields
# survives, and the panel states the factor (and the dB it stands for) in
# writing. The measured level annotations stay the physical ones, so nothing
# on screen is silently rescaled.
# The ladder is coarse on purpose (1-1.5-2-3-5-7 per decade): a round factor is
# readable in an annotation and stays put when the field is re-simulated. It
# starts at 5 (14 dB) so a region that already uses a fifth of the ramp is
# left alone -- the caveat such a panel would have to print costs more than
# the contrast it would buy.
_GAIN_STEPS = (5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0,
               200.0, 300.0, 500.0, 700.0, 1000.0)


def _weak_field_gain(weak: Any, vmax: float, *,
                     quantile: float = 0.999) -> float:
    """Rounded display gain that lifts a weak field onto a readable colour.

    ``weak`` is the quiet region of the field (any shape), ``vmax`` the
    colour-scale half-range the loud region set. The gain is the largest rung
    of :data:`_GAIN_STEPS` at or under the factor that makes the ``quantile``
    amplitude of ``weak`` fill the ramp, so the quiet region gets the whole
    colour scale and at most a thousandth of its samples saturate. A region
    already within 14 dB of the ramp is returned at 1.0 (no gain, and so no
    annotation).
    """
    peak = float(np.quantile(np.abs(weak), quantile))
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    raw = float(vmax) / peak
    return max((g for g in _GAIN_STEPS if g <= raw), default=1.0)


def _gain_note(region: str, gain: float) -> str:
    """English annotation for a region drawn with a display gain.

    Empty for a unit gain (nothing to declare). The dB equivalent rides along
    so the reader can put the compression next to the level annotations.
    """
    if gain <= 1.0:
        return ""
    return f"{region} drawn ×{gain:g} (+{20.0 * np.log10(gain):.0f} dB)"


_BARRIER_FREQS = (100.0, 500.0)
# Receiver low over the ground: there the ground bounce reinforces both
# frequencies almost equally (tiny path difference), so the insertion-loss
# contrast is pure diffraction instead of ground-interference lobes.
_BARRIER_RECEIVER = (9.0, 0.5)
# Barrier timeline. Mesh: the highest carrier is 500 Hz (lambda = 0.686 m,
# lambda / 8 = 86 mm) and the smallest resolved geometric dimension is the
# 2.5 m screen height (2.5 / 4 = 0.63 m), so the rule dx = min(0.63, 0.086)
# m allows 86 mm; dx = 20 mm sits 4.3 times finer. (The screen is three
# cells thick, but its thickness is not an acoustic dimension: a rigid
# knife edge only has to be impermeable, which the 10^6:1 density contrast
# makes it, and the diffraction the clip shows is set by the edge height.)
# Flight: the source at (2.0, 0.5) m reaches the deepest reflector -- the
# rigid ground 0.5 m below -- and from there the farthest visible point,
# the top right corner (12, 7), is 12.207 m away, so 12.707 m / 343 m/s
# x 1.2 = 44.45 ms.
# Sampling: every solver step is captured, 24.74 us apart, i.e. 80.9 frames
# per 500 Hz period (>= 48; the grid offers nothing between this and the
# 40.4 of a two-step stride). 1 800 frames cover the 44.53 ms window,
# played at 120 fps -- six times the 20 fps of the committed clip, exactly
# the factor by which the stride shrank from its 6 solver steps per frame
# -- so the pacing stays the 2.969 ms of simulation per second of playback
# the committed clip had, and the clip runs 15 s (the old 18 s clip spent
# its extra seconds past the flight, on the 53.4 ms window trimmed here).
_BARRIER_EVERY = 1
_BARRIER_FRAMES = 1800
_BARRIER_FPS = 120


@lru_cache(maxsize=1)
def _barrier_fields(
        n_frames: int = _BARRIER_FRAMES) -> tuple[Any, Any, Any, Any]:
    """Two CW barrier-diffraction runs (low/high frequency), cached.

    A 12 m x 7 m half-space over rigid ground with a thin rigid barrier
    (5 000:1 density contrast) 2.5 m tall at x = 5.5 m; absorbing sponges
    on the two sides and the sky. Each frequency also gets a barrier-free
    reference run over the same ground, so the receiver annotation is a
    true insertion loss (patch-averaged around the receiver to keep the
    ground-interference lobes out of the number). After the captured clip
    both runs keep stepping, uncaptured, into genuine steady state and the
    insertion loss is an exact RMS over the final two full periods of that
    extended window (see the settle comment below). Returns the
    instantaneous-pressure frames, RMS maps in dB re each run's own
    maximum, the frame times and the per-frequency insertion losses.
    """
    import fdtd2d

    c0, dx = 343.0, 0.02
    ny, nx = 350, 600                      # 7 m x 12 m
    rho = np.full((ny, nx), 1.2)
    bx = round(5.5 / dx)              # thin barrier: 3 cells = 6 cm
    rho[:round(2.5 / dx), bx:bx + 3] = 1.2e6
    every = _BARRIER_EVERY
    # Receiver patch: 0.6 m x 0.6 m around the shadow-zone receiver,
    # energy-averaged so residual interference fringes average out.
    rx, ry = _BARRIER_RECEIVER
    patch = (slice(int((ry - 0.3) / dx), int((ry + 0.3) / dx) + 1),
             slice(int((rx - 0.3) / dx), int((rx + 0.3) / dx) + 1))
    p_all, db_all, ils = [], [], []
    times = np.zeros(0)
    for f in _BARRIER_FREQS:
        # Row 0 is the ground (displayed at the bottom via origin="lower"),
        # so the absorbing sides are left/right and the *high* rows ("bottom"
        # in the imshow-origin naming of fdtd2d); the ground stays rigid.
        rms_patch = []
        for rho_map in (rho, None):
            sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx),
                                rho=1.2 if rho_map is None else rho_map,
                                sponge_width=40,
                                sponge_sides=("left", "right", "bottom"))
            sim.add_source(fdtd2d.CWSource(ix=100, iy=25, frequency=f,
                                           ramp_cycles=2.0))
            if rho_map is not None:
                ps, rs, times, _ = _fdtd_cw_capture(sim, f, every, n_frames)
                p_all.append(ps)
                db_all.append(_rms_to_db(rs))
            else:
                # Barrier-free reference: same steps, no frames captured.
                for _ in range(every * n_frames):
                    sim.step()
            # The clip ends at 44.5 ms, but at 100 Hz the field behind the
            # barrier has not settled by then: after the 20 ms source ramp
            # the diffracted and ground-bounced paths over the edge keep
            # building the receiver level for several more periods. Step
            # both runs on, uncaptured, to ~113 ms -- where the measured
            # insertion loss sits within 0.05 dB of its value 30 ms later
            # -- and measure an exact RMS over the last two full periods,
            # so neither run's transient biases the published number.
            period = round(1.0 / (f * sim.dt))
            settle = round(0.113 / sim.dt) - sim.n
            acc = np.zeros_like(sim.p)
            for i in range(settle):
                sim.step()
                if i >= settle - 2 * period:
                    acc += sim.p**2
            rms = np.sqrt(acc / (2 * period))
            rms_patch.append(float(np.sqrt(np.mean(rms[patch] ** 2))))
        ils.append(20.0 * float(np.log10(rms_patch[1] / rms_patch[0])))
    return np.stack(p_all), np.stack(db_all), times, tuple(ils)


def animate_fdtd_barrier(output_dir: str) -> None:
    """A point source behind a thin rigid barrier on reflecting ground
    (2D FDTD), at 100 Hz and 500 Hz side by side: the long wavelength
    diffracts around the edge and fills the shadow zone, the short one is
    cast into a deep, clean shadow -- why barriers fail at low frequency."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, db_all, times, ils = _barrier_fields()
    half = p_all.shape[1] // 2
    lam = tuple(343.0 / f for f in _BARRIER_FREQS)
    rx, ry = _BARRIER_RECEIVER

    fig = _anim_figure()
    fig.suptitle(T("Barrier diffraction into the shadow zone (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [
        T(f"Low frequency: {_BARRIER_FREQS[0]:.0f} Hz "
          f"(λ ≈ {lam[0]:.1f} m)"),
        T(f"High frequency: {_BARRIER_FREQS[1]:.0f} Hz "
          f"(λ ≈ {lam[1]:.2f} m)"),
    ]
    verdicts = [T("diffraction fills the shadow"), T("deep, clean shadow")]
    ims: list[Any] = []
    il_txts: list[Any] = []
    for col in range(2):
        vmax = float(np.quantile(np.abs(p_all[col][half:]), 0.995))
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 12.0, 0.0, 7.0), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_r = ax_r.imshow(db_all[col][0], origin="lower",
                           extent=(0.0, 12.0, 0.0, 7.0), cmap="magma",
                           vmin=-40.0, vmax=0.0, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_p, ax_r):
            ax.grid(False)
            ax.set_ylim(-0.5, 7.0)
            ax.add_patch(Rectangle((0.0, -0.5), 12.0, 0.5,
                                   facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                                   lw=0.8, hatch="///"))
            # Theme-independent bar: mid-gray with a white edge stays
            # visible on the near-white RdBu row and the black magma row.
            ax.add_patch(Rectangle((5.44, 0.0), 0.18, 2.5,
                                   facecolor="#707070", edgecolor="white",
                                   lw=0.6))
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("x [m]", fontsize=8)
        ax_p.plot([2.0], [0.5], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
        ax_p.text(2.25, 0.55, T("source"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.text(5.53, 2.7, T("barrier"), ha="center", va="bottom",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_p.text(9.2, 1.1, T("shadow zone"), ha="center", va="center",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline)
        ax_r.text(11.7, 6.4, verdicts[col], ha="right", va="top",
                  color="white", fontsize=8)
        ax_r.plot([rx], [ry], marker="o", ms=5, color="white",
                  markeredgecolor="black", markeredgewidth=0.8)
        il_txts.append(
            ax_r.text(rx, ry + 0.45, "", ha="center", va="bottom",
                      color="white", fontsize=7.5))
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB re panel max]"), fontsize=8)
            ax_p.text(0.25, -0.27, T("rigid ground"), ha="left",
                      va="center", color=COLOR_FG, fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.2",
                            "facecolor": fig.get_facecolor(),
                            "edgecolor": "none"})
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(0.3, 5.45, T("each panel on its own dB scale"),
                      color="white", fontsize=7, ha="left", va="top")
        ims += [im_p, im_r]
    # Top-left margin: the field panels run their x-axis (ticks + "x [m]")
    # to the very bottom-right corner, so a bottom readout collides with the
    # tick labels; the top-left stays clear of the centred column titles.
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(p_all[col][k])
            ims[2 * col + 1].set_data(db_all[col][k])
            # The measured insertion loss appears once the field has
            # actually reached and settled at the receiver, so the number
            # never precedes its cause.
            il_txts[col].set_text(
                T(f"insertion loss {ils[col]:.0f} dB")
                if times[k] >= 0.032 else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *il_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_barrier",
                 frames=int(p_all.shape[1]), fps=_BARRIER_FPS, gif_fps=5)


_GROUND_FREQ = 400.0
_GROUND_H = 1.5
_GROUND_ARC_R = 8.0
# Ground-effect timeline. Mesh: the carrier is 400 Hz (lambda = 0.858 m,
# lambda / 8 = 107 mm) and the smallest geometric dimension is the 1.5 m
# source height (1.5 / 4 = 0.38 m), so the rule dx = min(0.38, 0.107) m
# allows 107 mm; dx = 20 mm sits 5.4 times finer, which the 8 m sampling
# arc needs to stay smooth.
# Flight: the source at (1.6, 1.5) m reaches the deepest reflector -- the
# rigid ground 1.5 m below -- and from there the farthest visible point,
# the top right corner (14, 8), is 14.757 m away, so 16.257 m / 343 m/s
# x 1.2 = 56.88 ms.
# Sampling: 2 solver steps per frame = 49.48 us gives 50.5 frames per
# 400 Hz period (>= 48). 1 150 frames cover the 56.90 ms window, played at
# 80 fps -- four times the 20 fps of the committed clip, exactly the
# factor by which the stride shrank from its 8 solver steps per frame --
# so the pacing stays the 3.958 ms of simulation per second of playback
# the committed clip had, and the clip runs 14.4 s (the old 18 s clip
# spent its extra seconds past the flight, on the 71.2 ms window trimmed
# here).
_GROUND_EVERY = 2
_GROUND_FRAMES = 1150
_GROUND_FPS = 80


@lru_cache(maxsize=1)
def _ground_effect_fields(
    n_frames: int = _GROUND_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """One CW run of a point source 1.5 m over rigid ground, cached.

    Returns instantaneous frames, RMS maps in dB, frame times, the arc
    angles [deg], the per-frame arc levels [dB re the final maximum], the
    two-path image-source model levels on the same arc, and the predicted
    null angles where the path difference is an odd multiple of lambda/2.
    """
    import fdtd2d

    c0, dx = 343.0, 0.02
    ny, nx = 400, 700                      # 8 m x 14 m
    sim = fdtd2d.FDTD2D(c0, dx, shape=(ny, nx), sponge_width=40,
                        sponge_sides=("left", "right", "bottom"))
    sim.add_source(fdtd2d.CWSource(ix=80, iy=75, frequency=_GROUND_FREQ,
                                   ramp_cycles=2.0))
    every = _GROUND_EVERY
    lam = c0 / _GROUND_FREQ
    theta = np.arange(1.0, 62.5, 0.5)
    rad = np.radians(theta)
    arc_x = 1.6 + _GROUND_ARC_R * np.cos(rad)
    arc_y = _GROUND_ARC_R * np.sin(rad)
    ix_arc = np.round(arc_x / dx).astype(int)
    iy_arc = np.round(arc_y / dx).astype(int)

    beta = float(np.exp(-sim.dt * _GROUND_FREQ / 2.0))
    ms = np.zeros_like(sim.p)
    ps: list[Any] = []
    rs: list[Any] = []
    ts: list[float] = []
    arc: list[Any] = []
    for _ in range(every * n_frames):
        sim.step()
        ms = beta * ms + (1.0 - beta) * sim.p**2
        if sim.n % every == 0 and len(ps) < n_frames:
            ps.append(sim.p[::2, ::2].astype(np.float32))
            rs.append(np.sqrt(ms[::2, ::2]).astype(np.float32))
            ts.append(sim.time)
            arc.append(np.sqrt(ms[iy_arc, ix_arc]))
    arc_rms = np.stack(arc)
    with np.errstate(divide="ignore"):
        arc_db = 20.0 * np.log10(arc_rms / float(arc_rms[-1].max()))
    arc_db = np.clip(arc_db, -45.0, None)

    # Two-path image-source model on the same arc (2D line source: 1/sqrt(r)
    # spreading), and the predicted nulls where r2 - r1 = (m + 1/2) lambda.
    def paths(th: Any) -> tuple[Any, Any]:
        x, y = (_GROUND_ARC_R * np.cos(np.radians(th)),
                _GROUND_ARC_R * np.sin(np.radians(th)))
        r1 = np.hypot(x, y - _GROUND_H)
        r2 = np.hypot(x, y + _GROUND_H)
        return r1, r2

    r1, r2 = paths(theta)
    k = 2.0 * np.pi / lam
    h = (np.exp(1j * k * r1) / np.sqrt(r1)
         + np.exp(1j * k * r2) / np.sqrt(r2))
    model_db = 20.0 * np.log10(np.abs(h) / float(np.abs(h).max()))
    th_fine = np.arange(0.5, 62.4, 0.02)
    r1f, r2f = paths(th_fine)
    delta = r2f - r1f
    nulls: list[float] = []
    for m in range(4):
        target = (m + 0.5) * lam
        cross = np.nonzero(np.diff(np.sign(delta - target)))[0]
        nulls.extend(float(np.interp(target, delta[c:c + 2],
                                     th_fine[c:c + 2])) for c in cross)
    return (np.stack(ps), _rms_to_db(np.stack(rs)), np.asarray(ts),
            theta, arc_db.astype(np.float32), model_db, tuple(sorted(nulls)))


def animate_fdtd_ground_effect(output_dir: str) -> None:
    """A 400 Hz point source 1.5 m above rigid ground (2D FDTD): the direct
    and ground-reflected wavefronts interfere and the lobe pattern forms;
    the ghosted image source below the ground explains the geometry, and the
    level sampled on an 8 m arc converges to the two-path model with its
    predicted nulls."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    (p_frames, db_frames, times, theta, arc_db, model_db,
     nulls) = _ground_effect_fields()
    half = p_frames.shape[0] // 2
    vmax = float(np.quantile(np.abs(p_frames[half:]), 0.995))

    fig = _anim_figure()
    fig.suptitle(T("Ground effect: direct + reflected interference "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1.0])
    ax_p = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[1, 0])
    ax_l = fig.add_subplot(gs[:, 1])

    im_p = ax_p.imshow(p_frames[0], origin="lower",
                       extent=(0.0, 14.0, 0.0, 8.0), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, interpolation="bilinear")
    im_r = ax_r.imshow(db_frames[0], origin="lower",
                       extent=(0.0, 14.0, 0.0, 8.0), cmap="magma",
                       vmin=-40.0, vmax=0.0, interpolation="bilinear")
    rad = np.radians(theta)
    arc_x = 1.6 + _GROUND_ARC_R * np.cos(rad)
    arc_y = _GROUND_ARC_R * np.sin(rad)
    for ax in (ax_p, ax_r):
        ax.grid(False)
        ax.set_ylim(-1.9, 8.0)
        ax.fill_between([0.0, 14.0], -1.9, 0.0, facecolor=COLOR_GRID,
                        edgecolor=COLOR_FG, lw=0.8, hatch="///")
        ax.tick_params(labelsize=7)
    ax_p.tick_params(labelbottom=False)
    ax_r.set_xlabel("x [m]", fontsize=8)
    ax_p.set_ylabel(T("instantaneous pressure"), fontsize=9)
    ax_r.set_ylabel(T("RMS level: interference lobes"), fontsize=8)
    # Source, ghosted image source and the mirror geometry.
    ax_p.plot([1.6], [_GROUND_H], marker="o", ms=5, color=COLOR_TERTIARY,
              markeredgecolor=FIELD_STROKE, markeredgewidth=0.8)
    ax_p.text(1.95, 1.65, T("source (h = 1.5 m)"), ha="left", va="center",
              color=FIELD_INK, fontsize=7.5, path_effects=outline)
    ax_p.plot([1.6], [-_GROUND_H], marker="o", ms=5, mfc="none",
              color=COLOR_TERTIARY, alpha=0.85)
    ax_p.plot([1.6, 1.6], [_GROUND_H, -_GROUND_H], ls=":",
              color=COLOR_TERTIARY, lw=1.0, alpha=0.7)
    hatch_box = {"boxstyle": "round,pad=0.2",
                 "facecolor": fig.get_facecolor(), "edgecolor": "none"}
    ax_p.text(1.95, -1.5, T("image source (ghost)"), ha="left",
              va="center", color=COLOR_FG, fontsize=7.5, bbox=hatch_box)
    ax_p.text(13.7, -0.95, T("rigid ground"), ha="right", va="center",
              color=COLOR_FG, fontsize=6.5, bbox=hatch_box)
    ax_p.text(0.3, 7.6, f"f = {_GROUND_FREQ:.0f} Hz", ha="left", va="top",
              color=FIELD_INK, fontsize=8, path_effects=outline)
    # Sampling arc and the receiver sitting in the first-order dip.
    ax_r.plot(arc_x, arc_y, ls=":", color="white", lw=0.9, alpha=0.6)
    th_dip = nulls[1]
    dip_x = 1.6 + _GROUND_ARC_R * np.cos(np.radians(th_dip))
    dip_y = _GROUND_ARC_R * np.sin(np.radians(th_dip))
    ax_r.plot([dip_x], [dip_y], marker="o", ms=5, color="white",
              markeredgecolor="black", markeredgewidth=0.8)
    # Right-aligned to the left of the dot: the Spanish label is longer
    # and would clip at the panel edge on the other side.
    ax_r.text(dip_x - 0.35, dip_y + 0.42, T("receiver in a dip"),
              ha="right", va="center", color="white", fontsize=7.5)
    # Level vs elevation angle, converging to the image-source model.
    _grid_axes(ax_l)
    ax_l.set_title(T("Level on the 8 m arc"), fontsize=10,
                   fontweight="bold")
    ax_l.set_xlabel(T("elevation angle θ [°]"), fontsize=8)
    ax_l.set_ylabel(T("level [dB re max]"), fontsize=8)
    ax_l.set_xlim(0.0, 63.0)
    ax_l.set_ylim(-34.0, 3.0)
    ax_l.tick_params(labelsize=7)
    for i, th in enumerate(n for n in nulls if n <= 62.0):
        ax_l.axvline(th, color=COLOR_SECONDARY, ls=":", lw=1.2,
                     label=T("predicted nulls") if i == 0 else None)
    ax_l.plot(theta, model_db, ls="--", color=COLOR_FG, lw=1.3, alpha=0.75,
              label=T("image-source model"))
    (l_sim,) = ax_l.plot([], [], color=COLOR_PRIMARY, lw=2.0, label="FDTD")
    # Same white-dot styling as the field receiver, so the two views of
    # the receiver are visually the same object.
    ax_l.plot([th_dip], [float(np.interp(th_dip, theta, model_db))],
              marker="o", ms=5, color="white", markeredgecolor="black",
              markeredgewidth=0.8)
    ax_l.legend(fontsize=7, loc="center right")
    # The strip above 0 dB is data-free, so the closing caption fits there.
    # The longer Spanish verdict spans the whole panel at 7.5 pt, so it drops
    # a step to keep a margin on both sides.
    verdict_txt = ax_l.text(0.5, 0.975, "", transform=ax_l.transAxes,
                            ha="center", va="top", color=COLOR_FG,
                            fontsize=7.5 if _LANG == "en" else 6.5,
                            fontweight="bold")
    # Bottom-left corner: the arc panel's wide x-label owns bottom-right.
    t_txt = fig.text(0.015, 0.02, "", ha="left", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.83 * p_frames.shape[0])

    def update(k: int) -> tuple[Any, ...]:
        im_p.set_data(p_frames[k])
        im_r.set_data(db_frames[k])
        l_sim.set_data(theta, arc_db[k])
        verdict_txt.set_text(
            T("dips land exactly on the predicted nulls")
            if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (im_p, im_r, l_sim, verdict_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_ground_effect",
                 frames=int(p_frames.shape[0]), fps=_GROUND_FPS, gif_fps=5)


_DUCT_AXIS = 400.0                       # channel-axis depth [m]
_DUCT_SRC_DEPTHS = (400.0, 150.0)        # on the axis / near the surface


def _duct_profile(z: Any) -> Any:
    """Munk-style sound-speed profile with an exaggerated gradient.

    The canonical SOFAR shape (Munk 1974): a minimum at the channel axis,
    an exponential thermocline above and a near-linear pressure gradient
    below, ``c = c1 (1 + eps (eta + exp(-eta) - 1))``. The perturbation is
    scaled far beyond the real ocean's (eps = 0.35 vs ~0.0074, capped at
    +250 m/s) so a ray cycle fits the 2.4 km of this domain instead of
    tens of kilometres -- the mechanism is the point, not the scale.
    """
    c1, eps, b_scale = 1480.0, 0.35, 300.0
    eta = 2.0 * (np.asarray(z, dtype=np.float64) - _DUCT_AXIS) / b_scale
    c = c1 * (1.0 + eps * (eta + np.exp(-eta) - 1.0))
    return np.minimum(c, c1 + 250.0)


# SOFAR-duct timeline. Mesh: the pulse carries useful energy to ~19 Hz,
# where the slowest water on the axis (1 480 m/s) gives lambda = 77.9 m and
# lambda / 8 = 9.7 m; the geometry has no feature smaller than the 300 m
# profile scale (300 / 4 = 75 m), so the rule dx = min(75, 9.7) m allows
# 9.7 m and dx = 2 m sits 4.9 times finer, which the refracted wavefronts
# need to stay smooth over 2.4 km.
# Flight: nothing in this scene reflects (sponges all round, the turning
# points are refractive), so the flight is the source out to the farthest
# visible point, the corner (2 400, 800) m, over the slowest speed on the
# path (1 480 m/s on the channel axis). The near-surface source at
# (200, 150) m is the binding one at 2 294 m -- the on-axis source is
# 2 236 m away -- so 2 294 / 1 480 x 1.2 = 1.860 s, against the old
# 1.589 s window, which cut the wavefronts off before the frame edge.
# Sampling: every solver step is captured, 490.5 us apart, i.e. 107.3
# frames per 19 Hz period (>= 48; a two-step stride would already drop to
# 53.6, still compliant, but this grid's dt is the natural floor). 3 793
# frames cover the 1.860 s window, played at 180 fps -- nine times the
# 20 fps of the committed clip, exactly the factor by which the stride
# shrank from its 9 solver steps per frame -- so the pacing stays the
# 88.29 ms of simulation per second of playback the committed clip had,
# and the clip runs 21.1 s (the old 18 s clip covered only its shorter
# 1.589 s window at the same speed).
_DUCT_EVERY = 1
_DUCT_FRAMES = 3793
_DUCT_FPS = 180


@lru_cache(maxsize=1)
def _ducting_fields(
    n_frames: int = _DUCT_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Two pulse runs in the SOFAR-like channel (on/off axis), cached.

    A 2 400 m x 800 m ocean slice with the exaggerated Munk profile of
    :func:`_duct_profile` and sponges on all four sides. Each source is a
    zero-mean Gaussian doublet (two opposite-sign pulses 33 ms apart, peak
    energy near 13 Hz). Returns the pressure frame stacks, the
    time-integrated energy maps in dB (the closing verdict overlay), the
    frame times, the depth grid, the c(z) profile and the full-resolution
    energy maps for the trapping physics check.
    """
    import fdtd2d

    dx = 2.0
    ny, nx = 400, 1200                     # 800 m depth x 2400 m range
    z = (np.arange(ny) + 0.5) * dx
    c_prof = _duct_profile(z)
    c_map = np.repeat(c_prof[:, np.newaxis], nx, axis=1)
    every = _DUCT_EVERY
    width, offset = 0.028, 0.033
    p_all, e_all = [], []
    times = np.zeros(0)
    for depth in _DUCT_SRC_DEPTHS:
        sim = fdtd2d.FDTD2D(c_map, dx, rho=1025.0, sponge_width=30)
        iy = round(depth / dx)
        sim.add_source(fdtd2d.GaussianPulse(ix=100, iy=iy, width=width))
        sim.add_source(fdtd2d.GaussianPulse(ix=100, iy=iy, width=width,
                                            t0=4.0 * width + offset,
                                            amplitude=-1.0))
        energy = np.zeros_like(sim.p)
        ps: list[Any] = []
        ts: list[float] = []
        for _ in range(every * n_frames):
            sim.step()
            energy += sim.p**2
            if sim.n % every == 0 and len(ps) < n_frames:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        e_all.append(energy)
        times = np.asarray(ts)
    energy_maps = np.stack(e_all)
    # Verdict overlay: the time-integrated p**2 map in dB (shared scale),
    # i.e. everywhere the pulse has carried energy over the whole run. The
    # reference comes from the far half of the domain so the near-source
    # blast saturates instead of washing out the duct-band contrast.
    ref = float(np.quantile(energy_maps[:, :, nx // 4:], 0.999))
    with np.errstate(divide="ignore"):
        e_db = 10.0 * np.log10(energy_maps[:, ::2, ::2] / ref)
    e_db = np.clip(e_db, -20.0, 0.0).astype(np.float32)
    return np.stack(p_all), e_db, times, z, c_prof, energy_maps


def animate_fdtd_ducting(output_dir: str) -> None:
    """A low-frequency pulse in a SOFAR-like underwater sound channel
    (2D FDTD): launched on the channel axis the wavefronts refract back
    toward the sound-speed minimum and stay trapped; launched near the
    surface the energy crosses the channel and leaks away to depth. The
    closing seconds crossfade to the time-integrated energy map, so the
    verdict frame shows the whole path history."""
    from matplotlib import patheffects

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_prof, _ = _ducting_fields()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, :half]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("SOFAR channel: sound trapped by the c(z) minimum "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.22, 1.0])
    titles = [T("Source on the channel axis (depth 400 m)"),
              T("Source near the surface (depth 150 m)")]
    verdicts = [T("trapped: wavefronts bend back to the axis"),
                T("leaks: energy escapes the channel")]
    extent = (0.0, 2400.0, 800.0, 0.0)
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    for row, depth in enumerate(_DUCT_SRC_DEPTHS):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_prof, z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot([float(np.interp(depth, z, c_prof))], [depth],
                  marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor="white", markeredgewidth=0.8)
        ax_c.axhline(_DUCT_AXIS, color=COLOR_FG, ls="--", lw=0.9, alpha=0.6)
        ax_c.set_ylim(800.0, 0.0)
        ax_c.set_xlim(1460.0, 1750.0)
        ax_c.set_xticks([1480.0, 1730.0])
        ax_c.set_ylabel(T("Depth [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T("c(z) [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(p_all[row][0], origin="upper", extent=extent,
                         cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                         aspect="auto", interpolation="bilinear")
        # The verdict overlay: fades in over the last seconds (and the
        # poster frame), replacing the instantaneous wavefronts with the
        # time-integrated energy paths.
        im_e = ax_f.imshow(e_db[row], origin="upper", extent=extent,
                           cmap="magma", vmin=-20.0, vmax=0.0,
                           aspect="auto", interpolation="bilinear",
                           alpha=0.0, zorder=2.5)
        ax_f.set_title(titles[row], fontsize=10, fontweight="bold")
        ax_f.axhline(_DUCT_AXIS, color="#888888", ls="--", lw=0.9,
                     alpha=0.8, zorder=3)
        ax_f.plot([200.0], [depth], marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor=FIELD_STROKE, markeredgewidth=0.8,
                  zorder=4)
        ax_f.text(240.0, depth - 25.0, T("source"), ha="left", va="bottom",
                  color=FIELD_INK, fontsize=7.5, path_effects=outline,
                  zorder=4)
        # A translucent dark pill keeps this label legible once the bright
        # magma energy overlay fades in over the channel axis (a plain white
        # stroke washes out against the near-white high-energy region).
        ax_f.text(2360.0, 425.0, T("channel axis (c minimum)"), ha="right",
                  va="top", color="white", fontsize=7, zorder=4,
                  bbox={"boxstyle": "round,pad=0.2", "facecolor": "black",
                        "alpha": 0.45, "edgecolor": "none"})
        v_txt = ax_f.text(60.0, 770.0, "", ha="left", va="bottom",
                          color=FIELD_INK, fontsize=8, path_effects=outline,
                          zorder=4)
        ax_f.tick_params(labelsize=7, labelleft=False)
        if row == 0:
            ax_f.tick_params(labelbottom=False)
        else:
            ax_f.set_xlabel(T("Range [m]"), fontsize=8)
        ims.append(im)
        ims_e.append(im_e)
        v_txts.append(v_txt)
    # Right margin, below the suptitle: the lower field panel carries the
    # "Range [m]" x-axis to the bottom-right corner, so a bottom readout
    # collides with its tick labels; this spot clears both the tick labels
    # and the (long) centred suptitle.
    t_txt = fig.text(0.988, 0.90, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.83 * p_all.shape[1])    # ~17.5 s: pulse has crossed
    captions_on = int(0.38 * p_all.shape[1])   # first refocus is visible

    def update(k: int) -> tuple[Any, ...]:
        alpha = min(1.0, max(0.0, (k - reveal) / 12.0))
        for row in range(2):
            ims[row].set_data(p_all[row][k])
            ims_e[row].set_alpha(alpha)
            v_txts[row].set_text(verdicts[row] if k >= captions_on else "")
        t_txt.set_text(T(f"t = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_ducting",
                 frames=int(p_all.shape[1]), fps=_DUCT_FPS, gif_fps=5)


#: Virtual-tube sample: the equivalent fluid of the solver cross-checks
#: (slower, denser, lossy; k = (omega - j sigma)/c at the real rho c).
_VTUBE_C2, _VTUBE_RHO2, _VTUBE_SIGMA = 0.6 * 343.0, 3.6, 600.0
_VTUBE_DX = 0.0025
_VTUBE_F = 850.0                          # CW inside the 100 mm plane range


def _anim_speaker(ax: Any, x0: float, y_mid: float, bore: float, *,
                  tip_inset: float | None = None,
                  label_y: float | None = None) -> None:
    """Drive loudspeaker of the FDTD tube/duct clips: magnet block plus
    cone, the cone tip on the ``x0`` plane, centred on ``y_mid`` for the
    given ``bore``. The tip stops ``tip_inset`` short of each bore edge
    (3 % of the bore when omitted); a "loudspeaker" caption is drawn at
    ``label_y`` when given (``None`` skips it)."""
    from matplotlib.patches import Polygon, Rectangle

    if tip_inset is None:
        tip_inset = 0.03 * bore
    magnet_w, cone_w = 0.05, 0.045
    ax.add_patch(Rectangle((x0 - magnet_w - cone_w, y_mid - 0.32 * bore),
                           magnet_w, 0.64 * bore, facecolor="#9a9a9a",
                           edgecolor=COLOR_FG, linewidth=0.8, zorder=4))
    ax.add_patch(Polygon(
        [(x0 - cone_w, y_mid - 0.20 * bore),
         (x0 - cone_w, y_mid + 0.20 * bore),
         (x0, y_mid + 0.5 * bore - tip_inset),
         (x0, y_mid - 0.5 * bore + tip_inset)],
        closed=True, facecolor="#e8b98a", edgecolor=COLOR_FG,
        linewidth=0.8, zorder=4))
    if label_y is not None:
        ax.text(x0 - magnet_w - cone_w, label_y, "loudspeaker",
                ha="left", va="top", fontsize=7.5)


def _anim_tube_hardware(ax: Any, length: float, *, bore: float = 0.1,
                        sample: tuple[float, float] | None = None,
                        termination: str = "rigid",
                        mics: tuple[tuple[float, str], ...] = (),
                        label_speaker: bool = True) -> None:
    """Draw the tube as hardware around the FDTD bore: walls, loudspeaker,
    sample block, microphones and the termination, all to scale (metres)."""
    from matplotlib.patches import Rectangle

    wall = 0.014
    grey = "#9a9a9a"
    for y0 in (-wall, bore):
        ax.add_patch(Rectangle((0.0, y0), length, wall, facecolor=grey,
                               edgecolor=COLOR_FG, linewidth=0.7, zorder=3))
    # Loudspeaker driving the left end: magnet + cone into the bore.
    _anim_speaker(ax, 0.0, 0.5 * bore, bore,
                  label_y=-0.55 * bore if label_speaker else None)
    if sample is not None:
        ax.add_patch(Rectangle((sample[0], 0.0), sample[1] - sample[0],
                               bore, facecolor=COLOR_SECONDARY, alpha=0.30,
                               hatch="..", edgecolor=COLOR_SECONDARY,
                               linewidth=0.0, zorder=2.6))
    if termination == "rigid":
        ax.add_patch(Rectangle((length, -wall), 0.030, bore + 2 * wall,
                               facecolor="#5a5a5a", edgecolor=COLOR_FG,
                               linewidth=0.8, zorder=3))
        ax.text(length + 0.030, -0.55 * bore, "rigid plug", ha="right",
                va="top", fontsize=7.5)
    else:
        ax.add_patch(Rectangle((length, -wall), 0.045, bore + 2 * wall,
                               facecolor=grey, hatch="////",
                               edgecolor=COLOR_FG, linewidth=0.8, zorder=3))
        ax.text(length + 0.045, -0.55 * bore, "anechoic termination",
                ha="right", va="top", fontsize=7.5)
    for x_m, label in mics:
        ax.plot([x_m, x_m], [bore + wall, bore + wall + 0.35 * bore],
                color=COLOR_FG, lw=1.2, zorder=5)
        ax.plot([x_m], [bore + wall + 0.50 * bore], marker="o", ms=7,
                markerfacecolor=COLOR_PRIMARY, markeredgecolor=COLOR_FG,
                markeredgewidth=0.7, zorder=5)
        ax.text(x_m, bore + wall + 0.78 * bore, label, ha="center",
                va="bottom", fontsize=7.5, color=COLOR_PRIMARY)


# Impedance-tube timeline. Mesh: the carrier is 850 Hz, where the air
# wavelength is 0.404 m (lambda / 8 = 50 mm) and the equivalent-fluid
# sample's is 0.242 m (lambda / 8 = 30 mm); the smallest geometric
# dimension is the 0.1 m bore (0.1 / 4 = 25 mm), so the rule
# dx = min(25, 30) mm allows 25 mm and dx = 2.5 mm sits ten times finer,
# which the 10 cm sample (40 cells) and the envelope readout need.
# Flight: the wave enters at x = 0, crosses 1.1 m of air and 0.1 m of
# sample to the rigid plug and comes back to the farthest visible point
# (the left edge), i.e. 2.2 m at 343 m/s plus 0.2 m at the sample's
# 205.8 m/s = 7.39 ms, x 1.2 = 8.86 ms.
# Sampling: 7 solver steps per frame = 21.6 us gives 54.3 frames per 850 Hz
# period (>= 48), and 410 frames cover the 8.87 ms window (the old one ran
# 16.7 ms at 25 frames per period), played at 15/7 of the shared 20 fps --
# exactly the factor by which the stride shrank from the committed clip's
# 15 solver steps per frame -- so the pacing stays the 0.928 ms of
# simulation per second of playback the committed clip had, and the flight
# takes 9.6 s (the old 18 s clip ran on well past it). The clip then holds
# the settled frame for the shared 2 s (86 frames at this rate): the
# reflected front reaches the left edge on the last captured frame, so the
# standing-wave envelope -- a running maximum over the trailing period --
# is only complete right at the end, and the absorption pill is revealed
# there rather than mid-flight.
_VTUBE_EVERY = 7
_VTUBE_ACTIVE = 410
#: 15/7 of the shared rate, matching the stride cut (see the note above).
_VTUBE_FPS = _ANIM_FPS * 15.0 / 7.0
#: The shared 2 s closing hold, counted at this clip's own frame rate.
_VTUBE_HOLD = round(2.0 * _VTUBE_FPS)
_VTUBE_FRAMES = _VTUBE_ACTIVE + _VTUBE_HOLD


@lru_cache(maxsize=1)
def _impedance_tube_fields(
    n_frames: int = _VTUBE_ACTIVE,
) -> tuple[Any, Any, Any, Any]:
    """CW build-up in the virtual impedance tube, empty vs sample, cached.

    Two 1,2 m x 0,1 m tubes: a rho c edge on the left carries a sustained
    one-way 850 Hz plane wave in; the right end is rigid. The lower tube
    ends in a 10 cm equivalent-fluid sample. Returns the frame stacks, the
    running envelope stacks max|p(x)| over the trailing period, the frame
    times and the analytic absorption of the sample at the drive frequency.
    """
    import fdtd2d

    dx = _VTUBE_DX
    ny, nx = 40, 480                       # 0.1 m x 1.2 m
    every = _VTUBE_EVERY
    sample_cells = round(0.10 / dx)
    runs = []
    for with_sample in (False, True):
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        damping = np.zeros((ny, nx))
        if with_sample:
            c_map[:, nx - sample_cells:] = _VTUBE_C2
            rho_map[:, nx - sample_cells:] = _VTUBE_RHO2
            damping[:, nx - sample_cells:] = _VTUBE_SIGMA
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=damping,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_VTUBE_F)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        period_frames = max(1, round(1.0 / (_VTUBE_F * every * sim.dt)))
        ps: list[Any] = []
        env: list[Any] = []
        ts: list[float] = []
        recent: list[Any] = []
        while len(ps) < n_frames:
            sim.step()
            if sim.n % every == 0:
                row = np.abs(sim.p[ny // 2, :]).astype(np.float32)
                recent.append(row)
                if len(recent) > period_frames:
                    recent.pop(0)
                ps.append(sim.p[::2, ::2].astype(np.float32))
                env.append(np.max(np.stack(recent), axis=0))
                ts.append(sim.time)
        runs.append((np.stack(ps), np.stack(env)))
    times = np.asarray(ts)
    # Analytic absorption of the sample at the drive frequency.
    omega = 2.0 * np.pi * _VTUBE_F
    k2 = (omega - 1j * _VTUBE_SIGMA) / _VTUBE_C2
    z2 = _VTUBE_RHO2 * _VTUBE_C2
    zs = -1j * z2 / np.tan(k2 * 0.10)
    r = (zs - 1.2 * 343.0) / (zs + 1.2 * 343.0)
    alpha = float(1.0 - abs(r) ** 2)
    return runs[0], runs[1], times, alpha


def animate_fdtd_impedance_tube(output_dir: str) -> None:
    """The virtual impedance tube (2D FDTD): a loudspeaker drives a
    sustained plane tone into a rigid-walled tube; against the rigid plug
    the standing wave grows deep nulls (|r| ~ 1), while a 10 cm lossy
    sample in front of the same plug leaves the minima shallow: the ISO
    10534-2 microphone pair reads exactly that envelope. One concept: the
    standing-wave ratio IS the absorption measurement."""
    T = _translate_str
    (p_e, env_e), (p_s, env_s), times, alpha = _impedance_tube_fields()
    vmax = float(np.quantile(np.abs(p_e), 0.999))
    length, bore = 1.2, 0.1
    env_base, env_h, env_max = 0.24, 0.34, 2.2
    fig = _anim_figure()
    fig.suptitle(T("The virtual impedance tube: standing waves read the "
                   "absorption (2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Rigid end: deep minima, |r| ~ 1"),
              T("10 cm lossy sample: shallow minima")]
    mics = ((length - 0.10 - 0.20, "1"), (length - 0.10 - 0.15, "2"))
    ims: list[Any] = []
    lines: list[Any] = []
    x_env = (np.arange(p_e.shape[2] * 2) + 0.5) * _VTUBE_DX
    env_from = 16                          # hide the injection-line step
    for ax, title, with_sample in ((axes[0], titles[0], False),
                                   (axes[1], titles[1], True)):
        ax.grid(False)
        im = ax.imshow(np.zeros((20, 240)), origin="lower",
                       extent=(0.0, length, 0.0, bore),
                       cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="bilinear", zorder=2)
        _anim_tube_hardware(
            ax, length, bore=bore,
            sample=(length - 0.10, length) if with_sample else None,
            termination="rigid", mics=mics,
        )
        # The standing-wave envelope, drawn above the hardware.
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.text(0.005, env_base + 0.015, "|p| envelope", fontsize=7.5,
                ha="left", va="bottom", color=COLOR_FG, alpha=0.8)
        (line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.9, zorder=6)
        ax.set_aspect(0.42, adjustable="box")
        ax.set_xlim(-0.125, length + 0.075)
        ax.set_ylim(-0.150, env_base + env_h + 0.045)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ims.append(im)
        lines.append(line)
    axes[1].set_xlabel(T("Position along the tube [m]"), fontsize=9)
    a_txt = axes[1].text(0.03, env_base + env_h - 0.015, "", ha="left",
                         va="top", fontsize=9, color="white", zorder=7,
                         bbox={"boxstyle": _ANIM_PILL_BOX,
                               "facecolor": "black", "alpha": 0.55,
                               "edgecolor": "none"})
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The envelope is only settled once the reflected front has run back
    # down the whole tube (one round trip plus a period of the trailing
    # maximum), so the absorption pill is revealed at the very end of the
    # flight and read during the closing hold.
    n_active = len(times)
    reveal = n_active - 1

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        for im, line, (p_all, env) in zip(ims, lines,
                                          ((p_e, env_e), (p_s, env_s))):
            im.set_data(p_all[kf])
            env_row = np.repeat(env[kf], 2)[: x_env.size]
            line.set_data(x_env[env_from:],
                          env_base + env_row[env_from:] / env_max * env_h)
        a_txt.set_text(
            T(f"alpha = {alpha:.2f} at {_VTUBE_F:.0f} Hz")
            if k >= reveal else ""
        )
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.1f} ms"))
        return (*ims, *lines, a_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_impedance_tube",
                 frames=n_active + _VTUBE_HOLD, fps=_VTUBE_FPS, gif_fps=8)


# Transmission-tube timeline. Same 2.5 mm mesh and the same 850 Hz carrier
# as the impedance tube, for the same reason (bore / 4 = 25 mm against
# lambda_sample / 8 = 30 mm).
# Flight: the packet starts at x = 0.35 m; its transmitted half crosses
# 1.15 m of air and the 0.1 m sample to the far end of the frame
# (1.15 / 343 + 0.1 / 205.8 = 3.84 ms) while its reflected half runs
# 1.25 m of air back to the left edge (3.64 ms), so 1.2 x 3.84 ms =
# 4.61 ms of flight -- the old 3.2 ms window froze while both halves were
# still travelling.
# Sampling: 7 solver steps per frame = 21.6 us gives 54.3 frames per 850 Hz
# period (>= 48; the committed clip's 4-step stride gave 95, more than the
# norm asks, so the capture is coarsened to it), and 213 frames carry the
# flight, played at 4/7 of the shared 20 fps -- exactly the inverse of the
# factor by which the stride grew -- so the pacing stays the 0.247 ms of
# simulation per second of playback the committed clip had, and the flight
# takes 18.6 s. The clip then holds the settled verdict frame for the
# shared 2 s (23 frames at this rate).
_TTUBE_EVERY = 7
_TTUBE_ACTIVE = 213
#: 4/7 of the shared rate, undoing the stride coarsening (note above).
_TTUBE_FPS = _ANIM_FPS * 4.0 / 7.0
#: The shared 2 s closing hold, counted at this clip's own frame rate.
_TTUBE_HOLD = round(2.0 * _TTUBE_FPS)
_TTUBE_FRAMES = _TTUBE_ACTIVE + _TTUBE_HOLD
#: Frame the closing hold (and with it the deferred-loading poster) freezes
#: on: t = 3.2 ms, where the reflected and transmitted halves are fully
#: separated and both still inside the tube. Both ends are anechoic, so by
#: the end of the flight the packets have been absorbed and the tube is
#: quiet -- a correct last frame, but an empty verdict. The time readout
#: follows the frame back, so nothing on screen contradicts anything else.
_TTUBE_VERDICT = 148


@lru_cache(maxsize=1)
def _transmission_tube_fields(
    n_frames: int = _TTUBE_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """A carrier packet crossing the virtual transmission tube, cached.

    Two 1,6 m x 0,1 m tubes with anechoic rho c ends: the lower one holds
    a 10 cm equivalent-fluid layer mid-tube, so the packet splits into a
    reflected and an attenuated transmitted part. Returns the two frame
    stacks, the frame times and the analytic transmission loss at the
    carrier frequency.
    """
    import fdtd2d

    dx = _VTUBE_DX
    ny, nx = 40, 640                       # 0.1 m x 1.6 m
    every = _TTUBE_EVERY
    face = 320
    sample_cells = round(0.10 / dx)
    runs = []
    for with_sample in (False, True):
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        damping = np.zeros((ny, nx))
        if with_sample:
            sl = slice(face, face + sample_cells)
            c_map[:, sl] = _VTUBE_C2
            rho_map[:, sl] = _VTUBE_RHO2
            damping[:, sl] = _VTUBE_SIGMA
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=damping,
                            edge_impedance={"left": 1.2 * 343.0,
                                            "right": 1.2 * 343.0})
        sim.add_plane_wave("right", center=0.35, width=0.10,
                           wavelength=343.0 / _VTUBE_F)
        ps: list[Any] = []
        ts: list[float] = []
        active = min(n_frames, _TTUBE_ACTIVE)
        while len(ps) < active:
            sim.step()
            if sim.n % every == 0:
                ps.append(sim.p[::2, ::2].astype(np.float32))
                ts.append(sim.time)
        while len(ps) < n_frames:          # end hold on the verdict frame
            ps.append(ps[_TTUBE_VERDICT])
            ts.append(ts[_TTUBE_VERDICT])
        runs.append(np.stack(ps))
    omega = 2.0 * np.pi * _VTUBE_F
    k2 = (omega - 1j * _VTUBE_SIGMA) / _VTUBE_C2
    kd = k2 * 0.10
    z2 = _VTUBE_RHO2 * _VTUBE_C2
    rc = 1.2 * 343.0
    combo = (np.cos(kd) + 1j * z2 * np.sin(kd) / rc
             + rc * 1j * np.sin(kd) / z2 + np.cos(kd))
    tl = float(20.0 * np.log10(abs(combo) / 2.0))
    return runs[0], runs[1], np.asarray(ts), tl


def animate_fdtd_transmission_tube(output_dir: str) -> None:
    """The virtual transmission tube (2D FDTD): the loudspeaker end fires a
    carrier packet down a rigid-walled tube with an anechoic termination.
    The empty tube passes it unchanged; a 10 cm lossy layer mid-tube splits
    it into a reflection and a weakened transmission that the four ASTM
    E2611 microphones resolve. One concept: transmission loss is what fails
    to come out the other side."""
    T = _translate_str
    p_e, p_s, times, tl = _transmission_tube_fields()
    vmax = float(np.quantile(np.abs(p_e), 0.999))
    length, bore = 1.6, 0.1
    fig = _anim_figure()
    fig.suptitle(T("The virtual transmission tube: what gets through "
                   "(2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Empty tube: the packet crosses unchanged"),
              T("10 cm lossy layer: reflected + attenuated transmission")]
    mics = ((0.60, "1"), (0.65, "2"), (1.05, "3"), (1.10, "4"))
    ims: list[Any] = []
    for ax, title, with_sample in ((axes[0], titles[0], False),
                                   (axes[1], titles[1], True)):
        ax.grid(False)
        im = ax.imshow(np.zeros((20, 320)), origin="lower",
                       extent=(0.0, length, 0.0, bore),
                       cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="bilinear", zorder=2)
        _anim_tube_hardware(
            ax, length, bore=bore,
            sample=(0.80, 0.90) if with_sample else None,
            termination="anechoic", mics=mics,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.125, length + 0.10)
        ax.set_ylim(-0.150, 0.255)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        ims.append(im)
    axes[1].set_xlabel(T("Position along the tube [m]"), fontsize=9)
    tl_txt = axes[1].text(0.02, 0.205, "", ha="left", va="top",
                          fontsize=9, color="white", zorder=7,
                          bbox={"boxstyle": _ANIM_PILL_BOX,
                                "facecolor": "black", "alpha": 0.55,
                                "edgecolor": "none"})
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.55 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        ims[0].set_data(p_e[k])
        ims[1].set_data(p_s[k])
        tl_txt.set_text(
            T(f"TL = {tl:.1f} dB at {_VTUBE_F:.0f} Hz")
            if k >= reveal else ""
        )
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
        return (*ims, tl_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_transmission_tube",
                 frames=len(times), fps=_TTUBE_FPS, gif_fps=8)


_QRD_DESIGN_F = 343.0 / 0.56             # ~612 Hz: lambda0 = 0.56 m
_QRD_SLAB = (2.085, 3.915, 0.55, 0.85)   # x0, x1, y0, y1 of the panel slab


def _qrd_wells() -> list[tuple[float, float, float]]:
    """The QRD well openings: (x0, x1, depth) per well, panel-relative.

    An N = 7 quadratic-residue sequence (n^2 mod 7), well depths
    ``s_n * lambda0 / (2 N)`` with the 0.56 m design wavelength, 12 cm
    wells split by 1 cm rigid fins, two periods across the 1.83 m panel.
    """
    seq = (0, 1, 4, 2, 2, 4, 1)
    unit = 0.56 / (2 * len(seq))          # = 4 cm per residue step
    well_w, fin_w = 0.12, 0.01
    wells = []
    x = _QRD_SLAB[0]
    for _ in range(2):
        for s in seq:
            wells.append((x + fin_w, x + fin_w + well_w, s * unit))
            x += fin_w + well_w
    return wells


# Diffuser timeline. Mesh: the carrier is the 612 Hz design frequency
# (lambda = 0.56 m, lambda / 8 = 70 mm) and the smallest acoustic dimension
# of the panel is the 4 cm quadratic-residue depth unit (40 / 4 = 10 mm),
# so the rule dx = min(10, 70) mm gives exactly the 10 mm mesh used here.
# (The 1 cm fins between wells are one cell wide: they are the discrete
# stand-in for the infinitely thin rigid separators of Schroeder theory,
# and one cell of 10^6:1 density contrast already makes them impermeable.
# What has to be resolved is the well -- 12 cells across, 4 to 16 deep.)
# Flight: the packet starts 3.2 m up and runs 2.51 m down to the deepest
# well bottom (0.85 - 0.16 = 0.69 m); the worst-placed of those deepest
# wells sits at x = 3.715 m (see _qrd_wells), and from there the scattered
# fan has 4.685 m to the farthest visible corner of the cropped frame
# (0.4, 4.0). So 7.195 m / 343 m/s x 1.2 = 25.17 ms, against the old
# 22.27 ms window.
# Sampling: 2 solver steps per frame = 24.74 us gives 66.0 frames per
# period of the 612 Hz carrier (>= 48; 30.0 at the 1.35 kHz edge of the
# packet spectrum, itself well over the old 12-frame floor). 1 018 frames
# cover the 25.18 ms window, played at 50 fps -- 5/2 of the 20 fps of the
# committed clip, exactly the factor by which the stride shrank from its
# 5 solver steps per frame -- so the pacing stays the 1.237 ms of
# simulation per second of playback the committed clip had, and the clip
# runs 20.4 s (the old 18 s clip covered only its shorter 22.27 ms window
# at the same speed).
_DIFF_EVERY = 2
_DIFF_FRAMES = 1018
_DIFF_FPS = 50


@lru_cache(maxsize=1)
def _diffusion_fields(
    n_frames: int = _DIFF_FRAMES,
) -> tuple[Any, Any, Any, Any, Any]:
    """Plane-wave packet onto a flat panel vs a QRD, cached (three runs).

    A 6 m x 4.4 m free-field box (sponges all around) with a floating
    1.83 m panel: solid slab vs the same slab with quadratic-residue wells
    (5 000:1 density contrast builds the rigid geometry). A downward
    plane-wave packet (carrier at the 612 Hz design frequency) is injected
    as an initial condition with matched leapfrog velocities, so a single
    clean wavefront travels down. A third, panel-free reference run gives
    the incident field, so ``scattered = total - incident`` exactly.

    Returns the two total-field frame stacks, the scattered-field envelope
    trails [dB] (a fading |total - incident| history, so the specular beam
    and the diffuse fan persist into the verdict frame), frame times, the
    arc angles [deg], and the scattered energy levels on the arc [dB] per
    panel (flat, QRD) for the diffusion-coefficient check.
    """
    import fdtd2d

    c0, dx = 343.0, 0.01
    ny, nx = 440, 600                      # 4.4 m x 6.0 m
    x0, x1, y0, y1 = _QRD_SLAB
    rho_flat = np.full((ny, nx), 1.2)
    rho_flat[round(y0 / dx):round(y1 / dx),
             round(x0 / dx):round(x1 / dx)] = 1.2e6
    rho_qrd = rho_flat.copy()
    for wx0, wx1, d in _qrd_wells():
        if d > 0.0:
            rho_qrd[round((y1 - d) / dx):round(y1 / dx),
                    round(wx0 / dx):round(wx1 / dx)] = 1.2
    rho_ref = np.full((ny, nx), 1.2)

    # Downgoing packet: carrier at the design wavelength under a Gaussian
    # envelope (~10 % spectral amplitude beyond 1.15 kHz).
    y_pkt, sig, lam0 = 3.2, 0.3, 0.56

    # Receiver arc over the panel, ISO 17497-2 goniometer style.
    theta = np.arange(15.0, 165.5, 1.0)
    rad = np.radians(theta)
    ix_arc = np.round((3.0 + 2.2 * np.cos(rad)) / dx).astype(int)
    iy_arc = np.round((y1 + 2.2 * np.sin(rad)) / dx).astype(int)

    every = _DIFF_EVERY
    sims = []
    for rho in (rho_flat, rho_qrd, rho_ref):
        sim = fdtd2d.FDTD2D(c0, dx, rho=rho, shape=(ny, nx),
                            sponge_width=40)
        # One-way carrier-under-Gaussian packet toward the panel, the
        # leapfrog-consistent initial condition the solver now provides.
        sim.add_plane_wave("up", center=y_pkt, width=sig, wavelength=lam0)
        sims.append(sim)
    # The three runs advance in lockstep so the scattered field
    # (total - incident, exact by linearity) is available at every step
    # for the arc energies and for the fading envelope trails (6 ms
    # half-life). Below the panel face the difference is just the ghost
    # of the unblocked incident wave, so the trail is masked there.
    decay = float(2.0 ** (-sims[0].dt / 0.006))
    face_row = round(y1 / dx)
    trails = [np.zeros_like(sims[0].p) for _ in range(2)]
    arc_e = np.zeros((2, theta.size))
    tot_frames: list[list[Any]] = [[], []]
    trail_frames: list[list[Any]] = [[], []]
    ts: list[float] = []
    for _ in range(every * n_frames):
        for sim in sims:
            sim.step()
        for j in range(2):
            scat = sims[j].p - sims[2].p
            scat[:face_row, :] = 0.0
            arc_e[j] += scat[iy_arc, ix_arc] ** 2
            np.maximum(trails[j] * decay, np.abs(scat), out=trails[j])
        if sims[0].n % every == 0 and len(ts) < n_frames:
            for j in range(2):
                tot_frames[j].append(
                    sims[j].p[::2, ::2].astype(np.float32))
                trail_frames[j].append(
                    trails[j][::2, ::2].astype(np.float32))
            ts.append(sims[0].time)
    times = np.asarray(ts)
    tot = np.stack([np.stack(f) for f in tot_frames])
    trail = np.stack([np.stack(f) for f in trail_frames])
    ref = float(trail[:, trail.shape[1] // 3:].max())
    with np.errstate(divide="ignore"):
        trail_db = 20.0 * np.log10(trail / ref)
    trail_db = np.clip(trail_db, -30.0, 0.0).astype(np.float32)
    # Arc levels floored 45 dB under each panel's own peak.
    levels = []
    for j in range(2):
        with np.errstate(divide="ignore"):
            lvl = 10.0 * np.log10(arc_e[j] / float(arc_e[j].max()))
        levels.append(np.maximum(lvl, -45.0))
    return tot, trail_db, times, theta, np.stack(levels)


def animate_fdtd_diffusion(output_dir: str) -> None:
    """A plane wavefront hitting a flat rigid panel vs a Schroeder QRD
    (2D FDTD, ISO 17497-2 goniometer style): the flat panel throws a
    collimated specular beam back, the diffuser's phase-step wells spray
    the same energy into a wide fan; the scattered field (total minus
    incident) and the arc diffusion coefficients make the contrast
    quantitative."""
    from matplotlib import patheffects
    from matplotlib.patches import Polygon

    from phonometry import directional_diffusion_coefficient

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    tot_all, trail_db, times, theta, levels = _diffusion_fields()
    d_coef = [directional_diffusion_coefficient(lv) for lv in levels]
    x0, x1, y0, y1 = _QRD_SLAB
    vmax = float(np.quantile(np.abs(tot_all[:, 0]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Flat panel vs Schroeder diffuser (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T("Flat rigid panel"), T("Schroeder diffuser (QRD, N = 7)")]
    beams = [T("specular beam"), T("scattered fan")]
    # Panel cross-sections: the flat slab, and the staircase along the QRD
    # surface (wells carved into the same slab).
    flat_poly = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    qrd_poly: list[tuple[float, float]] = [(x0, y0), (x0, y1)]
    for wx0, wx1, d in _qrd_wells():
        qrd_poly += [(wx0, y1), (wx0, y1 - d), (wx1, y1 - d), (wx1, y1)]
    qrd_poly += [(x1, y1), (x1, y0)]
    polys = [flat_poly, qrd_poly]
    rad = np.radians(theta)
    arc_x = 3.0 + 2.2 * np.cos(rad)
    arc_y = y1 + 2.2 * np.sin(rad)

    ims: list[Any] = []
    d_txts: list[Any] = []
    for col in range(2):
        ax_t = fig.add_subplot(gs[0, col])
        ax_s = fig.add_subplot(gs[1, col])
        im_t = ax_t.imshow(tot_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 4.4), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_s = ax_s.imshow(trail_db[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 4.4), cmap="magma",
                           vmin=-30.0, vmax=0.0, interpolation="bilinear")
        ax_t.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_t, ax_s):
            ax.grid(False)
            ax.add_patch(Polygon(polys[col], closed=True,
                                 facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                                 lw=1.0))
            # Crop the absorbing sponge zones out of view, so the frame
            # edge is physical field, not the boundary-layer artefacts.
            ax.set_xlim(0.4, 5.6)
            ax.set_ylim(0.0, 4.0)
            ax.tick_params(labelsize=7)
        ax_t.tick_params(labelbottom=False)
        ax_s.set_xlabel("x [m]", fontsize=8)
        ax_t.text(3.0, 3.6, T("incident plane wavefront"), ha="center",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_t.annotate("", xy=(3.0, 3.1), xytext=(3.0, 3.55),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_s.plot(arc_x, arc_y, ls=":", color="white", lw=0.9, alpha=0.65)
        ax_s.text(3.0, 0.15, beams[col], ha="center", va="bottom",
                  color="white", fontsize=7.5)
        d_txt = ax_s.text(5.45, 3.82, "", ha="right", va="top",
                          color="white", fontsize=8.5, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(T("sound field p"), fontsize=9)
            ax_s.set_ylabel(T("scattered field (total − incident)"),
                            fontsize=8)
            ax_s.text(0.78, 2.05, T("receiver arc"), ha="left", va="bottom",
                      color="white", fontsize=6.5, rotation=64.0)
        else:
            ax_t.tick_params(labelleft=False)
            ax_s.tick_params(labelleft=False)
            ax_t.text(3.0, 0.15, T(f"design frequency "
                                   f"{_QRD_DESIGN_F:.0f} Hz"), ha="center",
                      va="bottom", color=FIELD_INK, fontsize=6.5,
                      path_effects=outline)
        ims += [im_t, im_s]
        d_txts.append(d_txt)
    t_txt = fig.text(0.985, 0.02, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.8 * tot_all.shape[1])   # arc energy has settled by here

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(tot_all[col][k])
            ims[2 * col + 1].set_data(trail_db[col][k])
            d_txts[col].set_text(
                T(f"diffusion coefficient d = {d_coef[col]:.2f}")
                if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.1f} ms"))
        return (*ims, *d_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_diffusion",
                 frames=int(tot_all.shape[1]), fps=_DIFF_FPS, gif_fps=5)


_META_DX = 0.00025
_META_NY, _META_NX = 4800, 6400
_META_XL, _META_FACE = 0.45, 0.36
_META_PITCH, _META_PERIODS = 0.07, 2
_META_F0 = 2000.0


def _metadiffuser_panel_mask(rho: Any) -> None:
    """Carve the Table-1 metadiffuser (two periods) into a dense slab.

    The slab spans the panel depth L = 2 cm plus a 3 mm back wall under
    the face line; each well is a vertical slit from the face with its two
    resonators shelved sideways into the septum, the same layout the
    to-scale drawing uses (slit at 0.12 d into the cell, lattice a = L/2).
    """
    dx, y1 = _META_DX, _META_FACE
    rows = _METADIFFUSER_T1_ROWS
    depth, back = 0.02, 0.003
    rho[round((y1 - depth - back) / dx):round(y1 / dx),
        round(_META_XL / dx):round((_META_XL + _META_PERIODS * 5
                                    * _META_PITCH) / dx)] = 1.2e6
    lattice = depth / 2
    for period in range(_META_PERIODS):
        for n, (h, ln, lc, wn, wc) in enumerate(rows):
            x0 = _META_XL + (period * 5 + n) * _META_PITCH
            x_slit = x0 + 0.12 * _META_PITCH
            c0s, c1s = round(x_slit / dx), round((x_slit + h * 1e-3) / dx)
            rho[round((y1 - depth) / dx):round(y1 / dx), c0s:c1s] = 1.2
            for m in range(2):
                y_m = y1 - depth + (2 - m - 0.5) * lattice
                x_neck = x_slit + h * 1e-3
                r0 = round((y_m - 0.5 * wn * 1e-3) / dx)
                r1 = round((y_m + 0.5 * wn * 1e-3) / dx)
                rho[r0:r1, c1s:round((x_neck + ln * 1e-3) / dx)] = 1.2
                r0 = round((y_m - 0.5 * wc * 1e-3) / dx)
                r1 = round((y_m + 0.5 * wc * 1e-3) / dx)
                rho[r0:r1, round((x_neck + ln * 1e-3) / dx):
                    round((x_neck + (ln + lc) * 1e-3) / dx)] = 1.2


def _meta_qrd_wells() -> list[tuple[float, float, float]]:
    """QRD well openings (x0, x1, depth) matched to the metadiffuser.

    The same N = 5 residue order as the Table-1 metadiffuser
    (s = 1, 4, 4, 1, 0), designed at 500 Hz: depths s lambda0 / (2 N) up
    to 27.4 cm, 65 mm wells split by 5 mm fins on the 7 cm pitch.
    """
    unit = (343.0 / 500.0) / 10.0
    wells: list[tuple[float, float, float]] = []
    for period in range(_META_PERIODS):
        for n, s_n in enumerate((1, 4, 4, 1, 0)):
            x0 = _META_XL + (period * 5 + n) * _META_PITCH
            wells.append((x0 + 0.005, x0 + _META_PITCH, s_n * unit))
    return wells


def _meta_rho(kind: str) -> Any:
    """Density map of one run: ``flat``, ``qrd``, ``meta`` or ``ref``."""
    dx, ny, nx = _META_DX, _META_NY, _META_NX
    y1 = _META_FACE
    x_r = _META_XL + _META_PERIODS * 5 * _META_PITCH
    rho = np.full((ny, nx), 1.2)
    if kind == "qrd":
        rho[round(0.05 / dx):round(y1 / dx),
            round(_META_XL / dx):round(x_r / dx)] = 1.2e6
        for wx0, wx1, d in _meta_qrd_wells():
            if d > 0.0:
                rho[round((y1 - d) / dx):round(y1 / dx),
                    round(wx0 / dx):round(wx1 / dx)] = 1.2
    elif kind == "meta":
        _metadiffuser_panel_mask(rho)
    elif kind == "flat":
        # A flat rigid slab with the metadiffuser's exact silhouette, so
        # the fans can only come from the slits, not from the outline.
        rho[round((y1 - 0.023) / dx):round(y1 / dx),
            round(_META_XL / dx):round(x_r / dx)] = 1.2e6
    return rho


def _meta_taper() -> Any:
    """Lateral cosine taper of the incident packet (free-field edges).

    The wavefront is flat over the panels and dies smoothly well before
    the lateral sponges, so the exterior boundaries only ever absorb
    outgoing scattered waves and the incident front stays plane: no edge
    arcs in the total field, and the reference run subtracts exactly.
    """
    x = (np.arange(_META_NX) + 0.5) * _META_DX
    x0, x1, x2, x3 = 0.10, 0.30, 1.30, 1.50
    w = np.zeros(_META_NX)
    rise = (x >= x0) & (x < x1)
    w[rise] = 0.5 - 0.5 * np.cos(np.pi * (x[rise] - x0) / (x1 - x0))
    w[(x >= x1) & (x <= x2)] = 1.0
    fall = (x > x2) & (x <= x3)
    w[fall] = 0.5 + 0.5 * np.cos(np.pi * (x[fall] - x2) / (x3 - x2))
    return w


_META_FRAMES = 600   # 6.13 ms / 10.2 us per frame (49 frames per period)


def _meta_pair_worker(kind: str, n_frames: int) -> tuple[Any, Any, Any]:
    """One panel run in lockstep with its own free-field reference.

    Each worker process pays for its private incident-field run, but the
    three workers advance in parallel, so the wall time of the cached
    field computation is about two simulations instead of four.
    """
    import fdtd2d

    c0, dx = 343.0, _META_DX
    y1 = _META_FACE
    lam0 = c0 / _META_F0
    # Duration rule: full flight source -> deepest well bottom -> top of
    # the visible frame, (0.714 + 1.034) m / c * 1.2 = 6.12 ms, sampled at
    # four times the 12 frames-per-period visual floor (49/period at 2 kHz).
    every = 33
    sims = []
    for rho in (_meta_rho(kind), _meta_rho("ref")):
        sim = fdtd2d.FDTD2D(c0, dx, rho=rho, shape=(_META_NY, _META_NX),
                            sponge_width=200)
        sim.add_plane_wave("up", center=0.80, width=0.05, wavelength=lam0)
        taper = _meta_taper()
        sim.p *= taper[np.newaxis, :]
        sim.vy *= taper[np.newaxis, :]
        sims.append(sim)
    decay = float(2.0 ** (-sims[0].dt / 0.0015))
    face_row = round(y1 / dx)
    trail = np.zeros_like(sims[0].p)
    tot_frames: list[Any] = []
    trail_frames: list[Any] = []
    ts: list[float] = []
    for _ in range(every * n_frames):
        for sim in sims:
            sim.step()
        scat = sims[0].p - sims[1].p
        scat[:face_row, :] = 0.0
        np.maximum(trail * decay, np.abs(scat), out=trail)
        if sims[0].n % every == 0 and len(ts) < n_frames:
            tot_frames.append(sims[0].p[::20, ::20].astype(np.float32))
            trail_frames.append(trail[::20, ::20].astype(np.float32))
            ts.append(sims[0].time)
    return np.stack(tot_frames), np.stack(trail_frames), np.asarray(ts)


@lru_cache(maxsize=1)
def _metadiffuser_fields(
    n_frames: int = _META_FRAMES,
) -> tuple[Any, Any, Any]:
    """Plane-wave packet onto a deep QRD vs the 2 cm metadiffuser, cached.

    A 1.6 m x 1.2 m free-field box at dx = 0.25 mm, fine enough to mesh the
    real millimetre slits, necks and cavities of the Table-1 metadiffuser
    (density contrast 1e6:1 builds the panels). Each panel (flat control
    with the metadiffuser's silhouette, deep QRD, metadiffuser) advances
    in lockstep with a free-field reference inside its own worker process,
    so ``total - incident`` is exact; a downgoing carrier packet at the
    2 kHz evaluation frequency plays the goniometer source. Returns the total
    frame stacks, the fading scattered-envelope trails [dB] and the frame
    times. The quantitative far-field comparison lives in the companion
    polar figure; this clip shows the near fields of the meshed panels.
    """
    import multiprocessing as mp

    tot, trail, times = None, None, None
    try:
        # The GPU host of .env turns the half-hour CPU computation into a
        # couple of minutes (the runner falls back by itself, but the CPU
        # pool below is faster than its single-process local mode).
        import fdtd_gpu_remote

        fdtd_gpu_remote.load_env()
        config = fdtd_gpu_remote.RemoteConfig.from_env()
        use_gpu = bool(config.host) and fdtd_gpu_remote.remote_available(
            config)
    except (ImportError, OSError, ValueError):
        use_gpu = False
    if use_gpu:
        import fdtd2d

        # Same duration rule as the CPU worker: 6.12 ms at 49 frames/period.
        every = 33
        stride = 20
        dt = fdtd2d.FDTD2D(343.0, _META_DX, shape=(4, 8)).dt
        sample_steps = [every * k for k in range(1, n_frames + 1)]
        lam0 = 343.0 / _META_F0
        frames: dict[str, Any] = {}
        for kind in ("flat", "qrd", "meta", "ref"):
            job = fdtd_gpu_remote.build_job(
                343.0, _META_DX, steps=every * n_frames,
                sample_steps=sample_steps, shape=(_META_NY, _META_NX),
                rho=_meta_rho(kind), sponge_width=200,
                plane_waves=[{"direction": "up", "center": 0.80,
                              "width": 0.05, "wavelength": lam0}],
                init_scale_x=_meta_taper(),
                sample_stride=stride, sample_dtype="float32",
            )
            # The four 19 800-step field jobs run ~10-12 min each on the
            # GPU host; give each submit an hour before falling back.
            frames[kind] = fdtd_gpu_remote.submit(
                job, config, timeout=3600.0)["frames"]
        face_row = round(_META_FACE / _META_DX) // stride
        decay = float(2.0 ** (-(every * dt) / 0.0015))
        tot_list, trail_list = [], []
        for kind in ("flat", "qrd", "meta"):
            scat = frames[kind] - frames["ref"]
            scat[:, :face_row, :] = 0.0
            running = np.zeros_like(scat[0])
            history = np.empty_like(scat)
            for k in range(scat.shape[0]):
                np.maximum(running * decay, np.abs(scat[k]), out=running)
                history[k] = running
            tot_list.append(frames[kind].astype(np.float32))
            trail_list.append(history)
        tot = np.stack(tot_list)
        trail = np.stack(trail_list)
        times = np.asarray(sample_steps, dtype=np.float64) * dt
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=3) as pool:
            parts = pool.starmap(
                _meta_pair_worker,
                [(kind, n_frames) for kind in ("flat", "qrd", "meta")],
            )
        tot = np.stack([part[0] for part in parts])
        trail = np.stack([part[1] for part in parts])
        times = parts[0][2]
    ref = float(trail[:, trail.shape[1] // 3:].max()) or 1.0
    with np.errstate(divide="ignore"):
        trail_db = 20.0 * np.log10(trail / ref)
    trail_db = np.clip(trail_db, -30.0, 0.0).astype(np.float32)
    return tot, trail_db, times


def animate_fdtd_metadiffuser(output_dir: str) -> None:
    """The 27 cm deep Schroeder QRD vs the 2 cm metadiffuser that mimics
    it, next to a flat control slab (2D FDTD at 0.25 mm, real slits, necks
    and cavities meshed): the same 2 kHz wavefront leaves the same kind of
    scattered fan, from a panel 13.7 times thinner."""
    from matplotlib import patheffects
    from matplotlib.patches import Polygon, Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    tot_all, trail_db, times = _metadiffuser_fields()
    y1 = _META_FACE
    x_l = _META_XL
    x_r = x_l + _META_PERIODS * 5 * _META_PITCH
    vmax = float(np.quantile(np.abs(tot_all[:, 0]), 0.999))

    fig = _anim_figure()
    fig.suptitle(T("Schroeder diffuser vs metadiffuser (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 3)
    titles = [T("Flat rigid panel"), T("QRD, wells down to 27 cm"),
              T("Metadiffuser, 2 cm panel")]
    qrd_poly: list[tuple[float, float]] = [(x_l, 0.05), (x_l, y1)]
    for wx0, wx1, d in _meta_qrd_wells():
        qrd_poly += [(wx0, y1), (wx0, y1 - d), (wx1, y1 - d), (wx1, y1)]
    qrd_poly += [(x_r, y1), (x_r, 0.05)]
    xc = 0.5 * (x_l + x_r)

    ims: list[Any] = []
    d_txts: list[Any] = []
    for col in range(3):
        ax_t = fig.add_subplot(gs[0, col])
        ax_s = fig.add_subplot(gs[1, col])
        im_t = ax_t.imshow(tot_all[col][0], origin="lower",
                           extent=(0.0, 1.6, 0.0, 1.2), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_s = ax_s.imshow(trail_db[col][0], origin="lower",
                           extent=(0.0, 1.6, 0.0, 1.2), cmap="magma",
                           vmin=-30.0, vmax=0.0, interpolation="bilinear")
        ax_t.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_t, ax_s):
            ax.grid(False)
            if col == 1:
                ax.add_patch(Polygon(qrd_poly, closed=True,
                                     facecolor=COLOR_GRID,
                                     edgecolor=COLOR_FG, lw=0.8))
            else:
                ax.add_patch(Rectangle((x_l, y1 - 0.023), x_r - x_l, 0.023,
                                       facecolor=COLOR_GRID,
                                       edgecolor=COLOR_FG, lw=0.8))
            ax.set_xlim(0.06, 1.54)
            ax.set_ylim(0.0, 1.12)
            ax.tick_params(labelsize=7)
        ax_t.tick_params(labelbottom=False)
        ax_s.set_xlabel("x [m]", fontsize=8)
        if col == 1:
            ax_t.text(xc, 0.97, T("incident plane wavefront"), ha="center",
                      va="bottom", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline)
            ax_t.annotate("", xy=(xc, 0.83), xytext=(xc, 0.955),
                          arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                      "lw": 1.2})
        d_txt = ax_s.text(xc, 1.03, "", ha="center", va="top",
                          color="white", fontsize=7.5, fontweight="bold")
        if col == 0:
            ax_t.set_ylabel(T("sound field p"), fontsize=9)
            ax_s.set_ylabel(T("scattered field (total − incident)"),
                            fontsize=8)
        else:
            ax_t.tick_params(labelleft=False)
            ax_s.tick_params(labelleft=False)
        if col == 1:
            ax_t.annotate("", xy=(x_r + 0.045, y1 - 0.274),
                          xytext=(x_r + 0.045, y1),
                          arrowprops={"arrowstyle": "-", "color": COLOR_FG,
                                      "lw": 1.6})
            ax_t.text(x_r + 0.07, y1 - 0.14, "27 cm", ha="left",
                      va="center", fontsize=7, color=COLOR_FG)
        if col == 2:
            ax_t.text(xc, 0.06, T("real slits and resonators meshed at "
                                  "0.25 mm"), ha="center", va="bottom",
                      color=FIELD_INK, fontsize=6.5, path_effects=outline)
            ax_t.annotate("", xy=(x_r + 0.045, y1 - 0.023),
                          xytext=(x_r + 0.045, y1),
                          arrowprops={"arrowstyle": "-", "color": COLOR_FG,
                                      "lw": 1.6})
            ax_t.text(x_r + 0.07, y1 - 0.012, "2 cm", ha="left",
                      va="center", fontsize=7, color=COLOR_FG)
        ims += [im_t, im_s]
        d_txts.append(d_txt)
    t_txt = fig.text(0.985, 0.02, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.8 * tot_all.shape[1])

    verdicts = [T("a collimated specular beam"),
                T("a wide scattered fan"),
                T("the same fan, from 2 cm")]

    def update(k: int) -> tuple[Any, ...]:
        for col in range(3):
            ims[2 * col].set_data(tot_all[col][k])
            ims[2 * col + 1].set_data(trail_db[col][k])
            d_txts[col].set_text(verdicts[col] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1000.0:4.2f} ms"))
        return (*ims, *d_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_metadiffuser",
                 frames=int(tot_all.shape[1]), gif_fps=8)


# --- README banner: a wavefront in a hall of columns (2D FDTD) -------------
# A wide 4:1 clip made for the top of the README: an 800 Hz plane wavefront
# sweeps down a rigid-walled hall through a staggered field of rigid columns,
# diffracting around every pillar until the whole hall is filled with the
# interference of the scattered wavelets. Banner canvas: 8.0 x 2.0 in at
# 300 dpi = 2400 x 600 px; the visible window is exactly 4 m x 1 m so the
# field is shown at true physical aspect.
_PILLAR_DX = 0.0025                   # [m]
_PILLAR_LX, _PILLAR_LY = 4.4, 1.0     # domain [m]; 0.2 m sponge ends hidden
_PILLAR_NX = round(_PILLAR_LX / _PILLAR_DX)   # 1760
_PILLAR_NY = round(_PILLAR_LY / _PILLAR_DX)   # 400
_PILLAR_F0 = 800.0                    # carrier [Hz], lambda = 42.9 cm
_PILLAR_VIEW = (0.2, 4.2)             # visible x window [m] (4:1 exact)
_PILLAR_FIGSIZE = (8.0, 2.0)          # in at _ANIM_DPI -> 2400 x 600 px
# Mesh rule: the smallest geometric dimension is the 6.6 cm column-wall
# aperture (6.6 / 4 = 1.65 cm) against lambda / 8 = 5.36 cm at 800 Hz, so
# the bound is 1.65 cm; dx = 2.5 mm sits 6.6x finer (>= 44 cells per
# column diameter) for the definition the banner format demands.
# Duration rule: the packet crosses injection (x = 0.30 m) to the far
# absorbing end (4.40 m) in 12.0 ms and the norm floor is x 1.2 = 14.3 ms;
# the captured window runs 22.3 ms so the multiple-scattering coda is
# also seen decaying to silence. With dt = 0.6 dx / (c sqrt(2)) = 3.09 us
# and one stored frame every 8 steps (24.7 us), the 1.25 ms carrier
# period is sampled 50.5 times (>= 48 frames per period). The first 40
# frames (~1.0 ms of free travel before the first columns) are trimmed,
# leaving 900 frames played at 40 fps: the same 0.99 ms of simulation
# per clip second as the approved preview pacing, 22.5 s + hold.
_PILLAR_EVERY = 8
_PILLAR_FRAMES = 900
_PILLAR_FPS = 40
_PILLAR_WARM = 40                     # frames of free travel trimmed
_PILLAR_STEM = "anim_fdtd_pillar_hall"


def _pillar_layout() -> list[tuple[float, float, float]]:
    """Column centres and radii [m] of the banner hall, deterministic.

    A staggered lattice (0.30 m pitch, three rows) with seeded jitter,
    radius spread and random vacancies, so the hall reads as an irregular
    colonnade rather than a perfect crystal, porous enough that the
    wavetrain threads through and fills the right half instead of
    mirroring off a solid wall of columns. The 10-17 cm column diameters
    sit around a quarter to half of the 42.9 cm carrier wavelength, where
    a rigid cylinder scatters strongly and casts a readable shadow. The
    tightest apertures of the seeded layout are 12.1 cm between column
    surfaces and 6.6 cm between a column and a wall, so
    dx = min(smallest aperture / 4, lambda / 8) allows up to 1.6 cm; the
    2.5 mm grid is three times finer than that bound and rasterises the
    cylinders at >= 44 cells per diameter.
    """
    rng = np.random.default_rng(20260728)
    pitch = 0.30
    out: list[tuple[float, float, float]] = []
    for j, cy in enumerate((0.17, 0.50, 0.83)):
        for cx in np.arange(1.15, 3.46, pitch):
            x = float(cx) + (pitch / 2.0 if j % 2 else 0.0)
            drop = rng.random() < 0.18
            jx, jy = rng.uniform(-0.03, 0.03, 2)
            r = rng.uniform(0.05, 0.085)
            if drop:
                continue
            out.append((x + float(jx), float(cy) + float(jy), float(r)))
    return out


def _pillar_mask() -> Any:
    """Rasterise the columns into the rigid-cell mask of the banner run."""
    xs = (np.arange(_PILLAR_NX) + 0.5) * _PILLAR_DX
    ys = (np.arange(_PILLAR_NY) + 0.5) * _PILLAR_DX
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros((_PILLAR_NY, _PILLAR_NX), dtype=bool)
    for cx, cy, r in _pillar_layout():
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return mask


@lru_cache(maxsize=1)
def _pillar_fields(n_frames: int = _PILLAR_FRAMES) -> tuple[Any, Any]:
    """Field frames of the banner run: rigid hall, absorbing ends, cached.

    The top and bottom edges stay rigid (the hall walls), the left and
    right ends absorb through 0.2 m sponges hidden outside the visible
    window, and the columns are rigid rasterised cells. The incident wave
    is a single one-way plane-wave packet (Gaussian envelope one carrier
    wavelength wide) launched just before the first columns, so one front
    sweeps the hall, each pillar sheds a visible reflection into the
    structured coda behind it, and the hall then empties through the
    absorbing ends; a mild bulk damping (25 1/s, a ~0.28 s T60 stand-in
    for air and wall absorption) keeps the decay physical. 704 k cells
    and 7520 steps, still comfortably a CPU job, so the GPU runner is
    not involved.
    """
    import fdtd2d

    sim = fdtd2d.FDTD2D(
        343.0, _PILLAR_DX, shape=(_PILLAR_NY, _PILLAR_NX),
        sponge_width=80, sponge_sides=("left", "right"),
        damping=25.0, obstacle_mask=_pillar_mask())

    lam0 = 343.0 / _PILLAR_F0
    sim.add_plane_wave("right", center=0.30, width=0.08, wavelength=lam0)
    for _ in range(_PILLAR_WARM * _PILLAR_EVERY):
        sim.step()
    frames = np.empty((n_frames, _PILLAR_NY, _PILLAR_NX), dtype=np.float32)
    ts = np.empty(n_frames)
    for k in range(n_frames):
        for _ in range(_PILLAR_EVERY):
            sim.step()
        frames[k] = sim.p
        ts[k] = sim.time
    return frames, ts


def animate_fdtd_pillar_hall(output_dir: str) -> None:
    """README banner: an 800 Hz plane wavefront sweeps through a hall of
    rigid columns (2D FDTD at 2.5 mm): every pillar diffracts the front and
    the scattered wavelets interfere until they fill the whole hall."""
    from matplotlib.patches import Circle

    T = _translate_str
    frames, ts = _pillar_fields()
    n_active = frames.shape[0]
    dark = bool(_FILENAME_SUFFIX)
    cmap = CMAP_FIELD
    # Colour scale: 55 % of the packet's global peak, so the travelling
    # front saturates boldly while each column's shed reflection sits in
    # the mid ramp and the decaying coda fades instead of clipping.
    vmax = 0.55 * float(np.max(np.abs(frames)))
    fig = plt.figure(figsize=_PILLAR_FIGSIZE, dpi=_ANIM_DPI)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    im = ax.imshow(frames[0], origin="lower",
                   extent=(0.0, _PILLAR_LX, 0.0, _PILLAR_LY), cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    for cx, cy, r in _pillar_layout():
        ax.add_patch(Circle((cx, cy), r, facecolor=COLOR_GRID,
                            edgecolor=COLOR_FG, lw=0.8))
    ax.set_xlim(*_PILLAR_VIEW)
    ax.set_ylim(0.0, _PILLAR_LY)
    ax.set_aspect("auto")
    ax.grid(False)
    ax.axis("off")
    caption_bbox = {"facecolor": "black" if dark else "white",
                    "alpha": 0.55, "edgecolor": "none", "pad": 2.0}
    ax.text(0.012, 0.055, T("2D FDTD wavefront in a hall of columns"),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            color=COLOR_FG, bbox=caption_bbox)
    t_txt = ax.text(0.988, 0.055, "", transform=ax.transAxes, ha="right",
                    va="bottom", family="monospace", fontsize=9,
                    color=COLOR_FG, bbox=caption_bbox)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        im.set_data(frames[kf])
        t_txt.set_text(T(f"t = {ts[kf] * 1e3:5.2f} ms"))
        return (im, t_txt)

    # GIF budget: the full-frame wave motion compresses poorly, so the
    # README GIF decimates the 40 fps WebM 5x (gif_fps = 8) at the shared
    # 640 px width, landing at 6.2 MB (light) and 6.6 MB (dark), under
    # the ~8 MB GitHub autoplay budget.
    _render_clip(fig, update, output_dir, _PILLAR_STEM,
                 frames=n_active + _ANIM_HOLD, fps=_PILLAR_FPS, gif_fps=8,
                 poster_ss=_poster_ss_for(_PILLAR_STEM + ".webm"))


def animate_standing_wave_tube(output_dir: str) -> None:
    """ISO 10534-2 impedance tube: the incident and reflected waves travel
    inside a drawn tube and their sum forms the standing-wave envelope; a
    rigid termination (deep nodes) is compared with a porous sample
    (shallow nodes) sampled by the two wall microphones."""
    from matplotlib.patches import Polygon, Rectangle

    T = _translate_str
    lam = 3.5                       # display wavelength inside the tube
    k = 2.0 * np.pi / lam
    x0, xs = 1.5, 11.0              # tube mouth and sample face
    x = np.linspace(x0, xs, 400)
    amp = 0.42                      # per-wave display amplitude
    # Mic positions from the sample face (ISO 10534-2: two flush wall mics
    # near the specimen, spacing < half a wavelength).
    mic_xi = (0.90 * lam, 0.65 * lam)
    cases = [
        (T("Rigid termination"), 1.0),
        (T("Porous sample"), 0.55),
    ]

    fig = _anim_figure()
    fig.suptitle(T("Standing wave in the impedance tube (ISO 10534-2)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 1)
    panels: list[dict[str, Any]] = []
    for row, (title, refl) in enumerate(cases):
        ax = fig.add_subplot(gs[row])
        _schematic_axes(ax, (0.0, 14.2), (-1.75, 1.85), equal=True)
        ax.text(0.1, 1.7, title, ha="left", va="top", color=COLOR_FG,
                fontsize=10, fontweight="bold")
        # Tube walls, loudspeaker and termination
        for yw in (-1.0, 1.0):
            ax.plot([x0 - 0.1, xs + (0.8 if refl < 1.0 else 0.0)], [yw, yw],
                    color=COLOR_FG, lw=1.8)
        _draw_speaker(ax, x0 - 0.05, 0.0, size=1.55)
        if refl < 1.0:
            # porous specimen slab in front of the rigid backing
            ax.add_patch(Rectangle((xs, -1.0), 0.8, 2.0,
                                   facecolor=COLOR_TERTIARY, alpha=0.45,
                                   edgecolor=COLOR_FG, lw=1.0))
            wall_x = xs + 0.8
            ax.text(xs + 0.4, -1.15, T("sample"), ha="center", va="top",
                    color=COLOR_FG, fontsize=8)
        else:
            wall_x = xs
            ax.text(xs + 0.25, -1.15, T("rigid wall"), ha="center", va="top",
                    color=COLOR_FG, fontsize=8)
        ax.add_patch(Polygon(
            [(wall_x, -1.0), (wall_x, 1.0), (wall_x + 0.35, 1.0),
             (wall_x + 0.35, -1.0)], closed=True, facecolor="none",
            edgecolor=COLOR_FG, lw=1.2, hatch="///"))
        # Two flush wall microphones pointing down into the tube
        for j, xi in enumerate(mic_xi):
            xm = xs - xi
            _draw_mic(ax, xm, 1.42, direction=1, size=0.62, angle=-90.0)
            ax.text(xm, 1.62, f"$p_{j + 1}$", ha="center", va="bottom",
                    color=COLOR_FG, fontsize=9)
        # Waves: incident, reflected (fades in), their sum and the envelope
        (l_inc,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.2, alpha=0.85)
        (l_ref,) = ax.plot([], [], color=COLOR_TERTIARY, lw=1.2, alpha=0.0)
        (l_sum,) = ax.plot([], [], color=COLOR_SECONDARY, lw=2.2)
        env = amp * np.sqrt(1.0 + refl**2
                            + 2.0 * refl * np.cos(2.0 * k * (xs - x)))
        env_lines = [ax.plot(x, s * env, color=COLOR_FG, lw=1.0, ls="--",
                             alpha=0.0)[0] for s in (1.0, -1.0)]
        (mic_dots,) = ax.plot([], [], marker="o", ms=5, ls="none",
                              color=COLOR_SECONDARY)
        # left of the shared legend row, clear of the bottom panel's strip
        note = ax.text(2.5, -1.42, "", ha="center", va="top",
                       color=COLOR_FG, fontsize=9)
        readout = ax.text(13.9, 1.7, "", ha="right", va="top",
                          color=COLOR_FG, fontsize=9, family="monospace")
        panels.append({"refl": refl, "l_inc": l_inc, "l_ref": l_ref,
                       "l_sum": l_sum, "env": env, "env_lines": env_lines,
                       "mic_dots": mic_dots, "note": note,
                       "readout": readout})

    legend_ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    legend_ax.axis("off")
    for xl, color, lab in ((0.30, COLOR_PRIMARY, T("incident")),
                           (0.44, COLOR_TERTIARY, T("reflected")),
                           (0.58, COLOR_SECONDARY, T("sum p(x, t)")),
                           (0.72, COLOR_FG, T("envelope |p(x)|"))):
        legend_ax.plot([xl, xl + 0.025], [0.028, 0.028], color=color,
                       lw=2.0, ls="--" if color == COLOR_FG else "-",
                       transform=legend_ax.transAxes)
        legend_ax.text(xl + 0.032, 0.028, lab, ha="left", va="center",
                       color=COLOR_FG, fontsize=8.5,
                       transform=legend_ax.transAxes)

    f_disp = 0.55                   # display oscillation frequency [Hz]
    sweep = _ANIM_FRAMES - _ANIM_HOLD
    t_ref, t_env = 3.5, 7.0         # phase starts [s of clip time]

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf, sweep - 1) / _ANIM_FPS
        ph = 2.0 * np.pi * f_disp * tc
        arts: list[Any] = []
        a_ref = float(np.clip((tc - t_ref) / 1.2, 0.0, 1.0))
        a_env = float(np.clip((tc - t_env) / 1.2, 0.0, 1.0))
        for pn in panels:
            refl = pn["refl"]
            # xi is the distance to the sample face; the incident wave
            # cos(ph + k*xi) travels toward +x (into the sample) and the
            # reflected wave cos(ph - k*xi) back toward the loudspeaker.
            xi = xs - x
            p_inc = amp * np.cos(ph + k * xi)
            p_ref = refl * amp * np.cos(ph - k * xi)
            pn["l_inc"].set_data(x, p_inc)
            pn["l_ref"].set_data(x, p_ref)
            pn["l_ref"].set_alpha(0.85 * a_ref)
            pn["l_sum"].set_data(x, p_inc + a_ref * p_ref)
            for line in pn["env_lines"]:
                line.set_alpha(0.8 * a_env)
            if a_env > 0.0:
                xm = np.array([xs - m for m in mic_xi])
                pim = amp * np.cos(ph + k * (xs - xm))
                prm = refl * amp * np.cos(ph - k * (xs - xm))
                pn["mic_dots"].set_data(xm, pim + a_ref * prm)
                pn["note"].set_text(
                    T("deep nodes") if refl >= 1.0 else T("shallow nodes"))
                pn["note"].set_alpha(a_env)
                pn["readout"].set_text(
                    T(f"|R| = {refl:.2f}   α = {1.0 - refl**2:.2f}"))
                pn["readout"].set_alpha(a_env)
            else:
                pn["mic_dots"].set_data([], [])
                pn["note"].set_text("")
                pn["readout"].set_text("")
            arts += [pn["l_inc"], pn["l_ref"], pn["l_sum"], pn["mic_dots"],
                     pn["note"], pn["readout"], *pn["env_lines"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_standing_wave_tube")


def animate_flanking_paths(output_dir: str) -> None:
    """EN 12354-1 junction schematic: energy pulses leave the source room
    over the Dd, Ff, Fd and Df paths, shrinking at the junction, and each
    path label lights up as its pulse re-radiates into the receiving room."""
    from matplotlib.patches import Circle, Rectangle

    T = _translate_str
    fig = _anim_figure()
    fig.suptitle(T("Flanking transmission paths (EN 12354-1)"),
                 fontweight="bold")
    ax = fig.add_subplot()
    _schematic_axes(ax, (0.0, 14.2), (0.0, 7.6), equal=True)

    # Cross-section: source room | separating wall | receiving room, with a
    # continuous floor slab running through the junction.
    wall_x0, wall_x1 = 6.6, 7.1
    floor_y0, floor_y1 = 1.0, 1.6
    ax.add_patch(Rectangle((wall_x0, floor_y1), wall_x1 - wall_x0, 5.0,
                           facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    ax.add_patch(Rectangle((0.7, floor_y0), 12.8, floor_y1 - floor_y0,
                           facecolor=COLOR_GRID, edgecolor=COLOR_FG, lw=1.2))
    for xr in (0.7, 13.5):
        ax.plot([xr, xr], [floor_y1, 6.6], color=COLOR_FG, lw=1.4)
    ax.plot([0.7, 13.5], [6.6, 6.6], color=COLOR_FG, lw=1.4)
    ax.text(3.6, 6.35, T("source room"), ha="center", va="top",
            color=COLOR_FG, fontsize=10)
    ax.text(10.3, 6.35, T("receiving room"), ha="center", va="top",
            color=COLOR_FG, fontsize=10)
    # Element letters: capital = source side, lowercase = receiving side
    for xt, yt, s in ((wall_x0 - 0.25, 4.3, "D"), (wall_x1 + 0.25, 4.3, "d"),
                      (3.6, 1.3, "F"), (10.3, 1.3, "f")):
        ax.text(xt, yt, s, ha="center", va="center", color=COLOR_FG,
                fontsize=12, fontweight="bold", fontstyle="italic")
    _draw_speaker(ax, 2.6, 3.6, size=1.35)

    src = (2.9, 3.6)
    col_df = "#7e57c2"
    # Path polylines (source -> element -> junction/wall -> radiator) with a
    # per-path arrival strength standing in for the junction attenuation Kij.
    paths: list[dict[str, Any]] = [
        {"key": "Dd", "color": COLOR_PRIMARY, "arrive": 0.62,
         "pts": [src, (wall_x0, 3.6), (wall_x1, 3.6), (8.6, 3.6)],
         "rad": (wall_x1, 3.6), "span": (-70.0, 70.0),
         "desc": T("direct, wall to wall")},
        {"key": "Ff", "color": COLOR_SECONDARY, "arrive": 0.40,
         "pts": [src, (4.4, floor_y1), (4.4, 1.3), (10.0, 1.3),
                 (10.0, floor_y1)],
         "rad": (10.0, floor_y1), "span": (20.0, 160.0),
         "desc": T("floor to floor")},
        {"key": "Fd", "color": COLOR_TERTIARY, "arrive": 0.30,
         "pts": [src, (4.9, floor_y1), (4.9, 1.3), (6.85, 1.45),
                 (6.85, 3.0), (wall_x1, 3.0), (8.4, 3.0)],
         "rad": (wall_x1, 3.0), "span": (-70.0, 70.0),
         "desc": T("floor to wall")},
        {"key": "Df", "color": col_df, "arrive": 0.30,
         "pts": [src, (wall_x0, 2.4), (6.85, 2.3), (6.85, 1.3),
                 (9.4, 1.3), (9.4, floor_y1)],
         "rad": (9.4, floor_y1), "span": (20.0, 160.0),
         "desc": T("wall to floor")},
    ]
    junction = (6.85, 1.3)
    for pn in paths:
        pts = np.asarray(pn["pts"])
        ax.plot(pts[:, 0], pts[:, 1], color=pn["color"], lw=1.0, ls=":",
                alpha=0.35)
        (trail,) = ax.plot([], [], color=pn["color"], lw=2.0, alpha=0.85)
        pulse = Circle((0.0, 0.0), 0.16, facecolor=pn["color"],
                       edgecolor="none", visible=False)
        ax.add_patch(pulse)
        pn["trail"], pn["pulse"] = trail, pulse
        rad_x, rad_y = pn["rad"]
        pn["arcs"] = _make_wavefronts(ax, rad_x, rad_y, pn["color"], n=3,
                                      theta1=pn["span"][0],
                                      theta2=pn["span"][1], lw=1.4)
        pn["pts_a"] = pts
    junc_txt = ax.text(junction[0] + 0.3, junction[1] - 0.55, "",
                       ha="left", va="top", color=COLOR_FG, fontsize=8.5)
    # Path legend column: lights up as each pulse arrives. The longer Spanish
    # descriptions start a little further left so they clear the right wall.
    labels = []
    lab_x = 10.15 if _LANG == "en" else 9.6
    for j, pn in enumerate(paths):
        yt = 5.6 - 0.55 * j
        lab = ax.text(lab_x, yt, f"{pn['key']} — {pn['desc']}", ha="left",
                      va="center", color=pn["color"], fontsize=8.5,
                      fontweight="bold", alpha=0.35)
        labels.append(lab)
    verdict = ax.text(7.05, 0.35, "", ha="center", va="center",
                      color=COLOR_FG, fontsize=10, fontweight="bold")

    travel = 2.1                    # seconds a pulse takes over its path
    starts = (0.4, 2.6, 4.8, 7.0)   # launch time of each pulse [s]
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        junc_txt.set_text("")
        for pn, t0, lab in zip(paths, starts, labels, strict=True):
            frac = (tc - t0) / travel
            pts = pn["pts_a"]
            if frac <= 0.0:
                pn["pulse"].set_visible(False)
                pn["trail"].set_data([], [])
                _set_wavefronts(pn["arcs"], [0.0] * 3, 1.0)
            elif frac < 1.0:
                px, py = _polyline_point(pts, frac)
                pn["pulse"].set_center((px, py))
                pn["pulse"].set_visible(True)
                # Pulse shrinks and dims along the path: the transmitted
                # energy that is left after the element and the junction.
                scale = 1.0 - (1.0 - pn["arrive"]) * frac
                pn["pulse"].set_radius(0.17 * scale)
                pn["pulse"].set_alpha(0.35 + 0.65 * scale)
                n_tr = max(2, int(frac * 60))
                tr = np.array([_polyline_point(pts, f)
                               for f in np.linspace(0.0, frac, n_tr)])
                pn["trail"].set_data(tr[:, 0], tr[:, 1])
                pn["trail"].set_alpha(0.3 + 0.5 * scale)
                _set_wavefronts(pn["arcs"], [0.0] * 3, 1.0)
                if pn["key"] != "Dd" and 0.35 < frac < 0.75:
                    junc_txt.set_text(
                        T("junction: Kij attenuates each transfer"))
            else:
                pn["pulse"].set_visible(False)
                age = (tc - t0) - travel
                radii = [0.55 * (age - 0.35 * i) for i in range(3)]
                _set_wavefronts(pn["arcs"], radii, 1.8,
                                alpha=0.4 + 0.5 * pn["arrive"])
                lab.set_alpha(1.0)
            arts += [pn["pulse"], pn["trail"], *pn["arcs"], lab]
        verdict.set_text(
            T("R'w sums all paths — always below the wall alone")
            if tc >= 9.4 else "")
        arts += [junc_txt, verdict]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_flanking_paths")


def animate_intensity_scan_power(output_dir: str) -> None:
    """ISO 9614-2 sound power: a p-p probe traces the serpentine scan over
    the top face of the measurement box while the normal-intensity arrows
    appear behind it, and the partial powers of the five faces accumulate
    into the L_W meter."""
    from matplotlib.patches import Circle, Polygon

    T = _translate_str

    # Parallel projection of the measurement box (w x d x h metres).
    bw, bd, bh = 5.2, 2.6, 3.2
    ox, oy = 0.52, 0.30              # projected offset per metre of depth
    x0, y0 = 1.0, 1.1

    def proj(u: float, v: float, w: float) -> tuple[float, float]:
        return (x0 + u + ox * v, y0 + w + oy * v)

    fig = _anim_figure()
    fig.suptitle(T("Intensity scanning over a box surface (ISO 9614-2)"),
                 fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
    ax = fig.add_subplot(gs[0])
    _schematic_axes(ax, (0.0, 9.4), (0.0, 6.4), equal=True)

    # Box edges (hidden rear edges dashed)
    edges = [((0, 0, 0), (bw, 0, 0)), ((bw, 0, 0), (bw, bd, 0)),
             ((bw, bd, 0), (0, bd, 0)), ((0, bd, 0), (0, 0, 0)),
             ((0, 0, bh), (bw, 0, bh)), ((bw, 0, bh), (bw, bd, bh)),
             ((bw, bd, bh), (0, bd, bh)), ((0, bd, bh), (0, 0, bh)),
             ((0, 0, 0), (0, 0, bh)), ((bw, 0, 0), (bw, 0, bh)),
             ((bw, bd, 0), (bw, bd, bh)), ((0, bd, 0), (0, bd, bh))]
    for a, b in edges:
        xa, ya = proj(*a)
        xb, yb = proj(*b)
        ax.plot([xa, xb], [ya, yb], color=COLOR_FG, lw=1.1, alpha=0.65)
    # The machine under test inside the box (a plain block on the floor)
    mx, md = bw / 2, bd / 2
    m_w, m_d, m_h = 1.5, 1.0, 1.0
    base = [(mx - m_w / 2, md - m_d / 2), (mx + m_w / 2, md - m_d / 2),
            (mx + m_w / 2, md + m_d / 2), (mx - m_w / 2, md + m_d / 2)]
    top = [proj(u, v, m_h) for u, v in base]
    front = [proj(base[0][0], base[0][1], 0.0),
             proj(base[1][0], base[1][1], 0.0),
             proj(base[1][0], base[1][1], m_h),
             proj(base[0][0], base[0][1], m_h)]
    side = [proj(base[1][0], base[1][1], 0.0),
            proj(base[2][0], base[2][1], 0.0),
            proj(base[2][0], base[2][1], m_h),
            proj(base[1][0], base[1][1], m_h)]
    # Opaque theme-blended face tints so the box edges behind the machine
    # do not show through it (or through its label).
    from matplotlib.colors import to_rgb
    bg = np.asarray(to_rgb(plt.rcParams["figure.facecolor"]))
    grid_rgb = np.asarray(to_rgb(COLOR_GRID))
    for poly, al in ((front, 0.75), (side, 0.55), (top, 0.9)):
        tint = tuple(al * grid_rgb + (1.0 - al) * bg)
        ax.add_patch(Polygon(poly, closed=True, facecolor=tint,
                             edgecolor=COLOR_FG, lw=1.0))
    ax.text(*proj(mx, md, 0.45), T("source"), ha="center", va="center",
            color=COLOR_FG, fontsize=8.5)
    src_arcs = _make_wavefronts(ax, *proj(mx, md, m_h), COLOR_TERTIARY,
                                n=3, theta1=15.0, theta2=165.0, lw=1.2)

    # Serpentine scan over the TOP face: passes along u at stepped v.
    n_pass = 4
    vs = np.linspace(0.25, bd - 0.25, n_pass)
    serp: list[tuple[float, float]] = []
    for i, v in enumerate(vs):
        us = (0.3, bw - 0.3) if i % 2 == 0 else (bw - 0.3, 0.3)
        serp += [(us[0], float(v)), (us[1], float(v))]
    serp_a = np.asarray(serp)
    (scan_line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.6, alpha=0.8)
    probe = Circle((0.0, 0.0), 0.11, facecolor=COLOR_PRIMARY,
                   edgecolor=COLOR_FG, lw=1.0, zorder=6, visible=False)
    ax.add_patch(probe)
    probe_lab = ax.text(0.0, 0.0, T("p-p probe"), ha="left", va="bottom",
                        color=COLOR_FG, fontsize=8.5, visible=False)
    # Normal-intensity arrows on the top face: longer where the source is
    # closer (I·n ~ cos(theta) / r^2), revealed as the probe passes.
    gu, gv = np.meshgrid(np.linspace(0.55, bw - 0.55, 6),
                         np.linspace(0.4, bd - 0.4, 3))
    serp_fine = np.array([_polyline_point(serp_a, fr)
                          for fr in np.linspace(0.0, 1.0, 300)])
    arrows: list[dict[str, Any]] = []
    for u, v in zip(gu.ravel(), gv.ravel(), strict=True):
        r2 = (u - mx) ** 2 + (v - md) ** 2 + (bh - m_h) ** 2
        inten = (bh - m_h) / r2 ** 1.5      # cos(theta) / r^2, unnormalised
        # the serpentine fraction at which the probe passes this grid point
        near = int(np.argmin(np.hypot(serp_fine[:, 0] - u,
                                      serp_fine[:, 1] - v)))
        arrows.append({"i": float(inten), "frac": near / 299.0,
                       "u": float(u), "v": float(v)})
    i_max = max(ad["i"] for ad in arrows)
    for ad in arrows:
        xa, ya = proj(ad["u"], ad["v"], bh)
        ln = 0.28 + 0.75 * ad["i"] / i_max
        arr = _make_arrow(ax, COLOR_SECONDARY, scale=9.0)
        arr.set_positions((xa, ya), (xa, ya + ln))
        arr.set_visible(False)
        ad["arrow"] = arr
    ax.text(0.15, 6.25, T("normal intensity I·n on the surface"),
            ha="left", va="top", color=COLOR_SECONDARY, fontsize=9)

    # Right column: per-face partial powers into the L_W meter.
    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.4), (0.0, 6.4))
    ax_m.text(2.2, 6.25, T("partial powers"), ha="center", va="top",
              color=COLOR_FG, fontsize=10, fontweight="bold")
    # Face shares of the total power for a source at the box centre floor:
    # top sees the most, the four sides split the rest (plausible shares).
    faces = [("top", T("top"), 0.34), ("front", T("front"), 0.20),
             ("back", T("back"), 0.20), ("left", T("left"), 0.13),
             ("right", T("right"), 0.13)]
    p_total = 1.6e-3                 # W -> L_W = 92.0 dB
    boxes = {}
    for j, (key, lab, _share) in enumerate(faces):
        boxes[key] = _flow_box(ax_m, 1.1, 5.35 - 0.95 * j, 1.9, 0.8, lab)
    gauge = _make_gauge(ax_m, 3.3, 1.1, 0.85, "$L_W$", COLOR_SECONDARY,
                        lo="80", hi="95")
    ax_m.text(3.3, 2.75, r"$P = \sum_i \int I{\cdot}n\ dS_i$", ha="center",
              va="bottom", color=COLOR_FG, fontsize=10)
    # verdict rides under the box drawing, where the full width is free
    verdict = ax.text(4.7, 0.35, "", ha="center", va="center",
                      color=COLOR_SECONDARY, fontsize=10,
                      fontweight="bold")

    scan_end, face_step = 6.2, 0.85  # top-face scan, then one face per step
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # source breathing wavefronts above the machine
        age = (tc * 0.5) % 0.6
        _set_wavefronts(src_arcs, [0.9 * (age + 0.2 * i) for i in range(3)],
                        0.9, alpha=0.5)
        arts += src_arcs
        frac = min(tc / scan_end, 1.0)
        px, pv = _polyline_point(serp_a, frac)
        xpr, ypr = proj(px, pv, bh)
        probe.set_center((xpr, ypr + 0.06))
        probe.set_visible(tc > 0.0 and frac < 1.0)
        # name the probe during the first pass only, riding below the face
        # so the revealed intensity arrows never strike the label
        probe_lab.set_position((xpr + 0.28, ypr - 0.52))
        probe_lab.set_visible(probe.get_visible() and tc < 1.6)
        n_tr = max(2, int(frac * 90))
        tr = np.array([_polyline_point(serp_a, f)
                       for f in np.linspace(0.0, frac, n_tr)])
        trp = np.array([proj(u, v, bh) for u, v in tr])
        scan_line.set_data(trp[:, 0], trp[:, 1])
        # reveal the normal arrows the probe has already passed
        for a in arrows:
            a["arrow"].set_visible(a["frac"] <= frac)
            arts.append(a["arrow"])
        arts += [probe, probe_lab, scan_line]
        # accumulate the partial powers: top face during the scan, then the
        # remaining faces one by one
        acc = 0.0
        for j, (key, _lab, share) in enumerate(faces):
            if j == 0:
                got = share * frac
            else:
                got = share * float(np.clip(
                    (tc - scan_end - (j - 1) * face_step) / face_step,
                    0.0, 1.0))
            acc += got
            if got >= share - 1e-9:
                _light_box(boxes[key],
                           T(f"{got * p_total * 1e3:.2f} mW"),
                           COLOR_PRIMARY, fill=j == 0)
            elif got > 0.0:
                _light_box(boxes[key],
                           T(f"{got * p_total * 1e3:.2f} mW"),
                           COLOR_PRIMARY)
            else:
                _dim_box(boxes[key])
            b = boxes[key]
            arts += [b["box"], b["title"], b["value"]]
        if acc > 0.0:
            lw_now = 10.0 * np.log10(acc * p_total / 1e-12)
            arts += _set_gauge(gauge, (lw_now - 80.0) / 15.0,
                               T(f"{lw_now:.1f} dB"))
        verdict.set_text(
            T("any enclosing surface gives the same P") if acc >= 0.999
            else "")
        arts.append(verdict)
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_intensity_scan_power")


def animate_sweep_deconvolution(output_dir: str) -> None:
    """ISO 18233 swept-sine measurement: the exponential sweep crosses a
    drawn room while its spectrogram builds, then the inverse filter
    collapses the whole record into the impulse response."""
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from scipy.signal import fftconvolve, spectrogram

    from phonometry import impulse_response, sweep_signal

    T = _translate_str
    fs = 8000
    sweep_len, f1, f2 = 2.5, 60.0, 3500.0
    sweep = sweep_signal(fs, f1, f2, sweep_len)
    # Synthetic room: direct sound, two discrete echoes far enough apart
    # (90/160 ms >> the 64 ms spectrogram window) that the recorded sweep
    # shows them as visibly separate delayed copies of the main ridge,
    # plus a diffuse tail.
    rng = np.random.default_rng(18233)
    ir_len = int(0.45 * fs)
    system = np.zeros(ir_len)
    for delay_ms, g in ((12.0, 1.0), (90.0, 0.55), (160.0, 0.35)):
        system[int(delay_ms * 1e-3 * fs)] = g
    tail_t = np.arange(ir_len) / fs
    system += (0.05 * rng.standard_normal(ir_len)
               * np.exp(-6.9077 * tail_t / 0.5) * (tail_t > 0.012))
    recorded = fftconvolve(sweep, system)
    rec_dur = recorded.size / fs
    _freqs, times, sxx = spectrogram(recorded, fs, nperseg=512,
                                    noverlap=384)
    sxx_db = 10.0 * np.log10(np.maximum(sxx, sxx.max() * 1e-8) / sxx.max())
    ir = impulse_response(recorded, sweep, fs, method="spectral",
                          length=int(0.2 * fs))
    t_ir = np.arange(ir.ir.size) / fs * 1e3
    ir_n = ir.ir / np.max(np.abs(ir.ir))

    fig = _anim_figure()
    fig.suptitle(T("Sweep measurement and deconvolution (ISO 18233)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.15])
    ax_r = fig.add_subplot(gs[0, :])
    _schematic_axes(ax_r, (0.0, 18.0), (0.0, 3.4), equal=True)
    room_box = Rectangle((1.0, 0.3), 16.0, 2.8, facecolor="none",
                         edgecolor=COLOR_FG, lw=1.4)
    ax_r.add_patch(room_box)
    _draw_speaker(ax_r, 2.6, 1.7, size=1.1)
    _draw_mic(ax_r, 14.6, 1.7, direction=-1, size=0.85, label=T("mic"))
    # direct and reflected ray paths
    ax_r.plot([2.8, 13.9], [1.7, 1.7], color=COLOR_FG, lw=1.0, ls="--",
              alpha=0.5)
    for ry in (0.3, 3.1):
        ax_r.plot([2.8, 8.3, 13.9], [1.7, ry, 1.75], color=COLOR_FG,
                  lw=0.9, ls=":", alpha=0.45)
    ax_r.text(8.3, 0.05, T("direct + reflections"), ha="center", va="bottom",
              color=COLOR_FG, fontsize=8)
    arcs = _make_wavefronts(ax_r, 2.8, 1.7, COLOR_PRIMARY, n=4,
                            theta1=-75.0, theta2=75.0)
    for arc in arcs:            # wavefronts stay inside the drawn room
        arc.set_clip_path(room_box)
    freq_txt = ax_r.text(2.0, 2.55, "", ha="left", va="bottom",
                         color=COLOR_FG, fontsize=9, family="monospace")
    note = ax_r.text(9.0, 0.8, "", ha="center", va="center",
                     color=COLOR_FG, fontsize=9.5, fontstyle="italic",
                     visible=False,
                     bbox={"boxstyle": "round,pad=0.35",
                           "facecolor": plt.rcParams["figure.facecolor"],
                           "edgecolor": "none", "alpha": 0.85})

    ax_s = fig.add_subplot(gs[1, 0])
    ax_s.grid(False)
    disp = np.full_like(sxx_db, sxx_db.min())
    im = ax_s.imshow(disp, origin="lower", aspect="auto",
                     extent=(0.0, float(times[-1]), 0.0, fs / 2 / 1e3),
                     cmap="magma", norm=Normalize(-60.0, 0.0),
                     interpolation="bilinear")
    ax_s.set_xlabel(T("Time [s]"), fontsize=8)
    ax_s.set_ylabel(T("Frequency [kHz]"), fontsize=8)
    ax_s.tick_params(labelsize=7)
    ax_s.set_title(T("recorded sweep (spectrogram)"), fontsize=9)
    ridge_txt = ax_s.text(0.97, 0.08, "", transform=ax_s.transAxes,
                          ha="right", va="bottom", color="white", fontsize=8)

    ax_i = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_i)
    ax_i.set_xlim(0.0, 200.0)
    ax_i.set_ylim(-1.05, 1.05)
    (ir_line,) = ax_i.plot([], [], color=COLOR_PRIMARY, lw=1.0)
    ax_i.set_xlabel(T("Time [ms]"), fontsize=8)
    ax_i.set_title(T("impulse response"), fontsize=9)
    ax_i.tick_params(labelsize=7)
    tap_marks = [ax_i.annotate("", xy=(ms, g * 1.0),
                               xytext=(ms + 22.0, min(g + 0.35, 0.95)),
                               fontsize=7.5, color=COLOR_FG, ha="left",
                               arrowprops={"arrowstyle": "-",
                                           "color": COLOR_FG, "lw": 0.7},
                               visible=False)
                 for ms, g in ((12.0, 1.0), (90.0, 0.55))]
    tap_marks[0].set_text(T("direct"))
    tap_marks[1].set_text(T("reflections"))
    deconv = _flow_box(ax_i, 140.0, -0.72, 85.0, 0.42,
                       T("⊛ inverse filter"))

    play_end, dec_t = 6.5, 7.6      # sweep playback, deconvolution moment
    sweep_frames = _ANIM_FRAMES - _ANIM_HOLD

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf, sweep_frames - 1) / _ANIM_FPS
        arts: list[Any] = []
        t_sig = min(tc / play_end, 1.0) * rec_dur
        # room wavefronts coloured by the instantaneous sweep frequency
        if tc < play_end:
            f_now = f1 * (f2 / f1) ** min(t_sig / sweep_len, 1.0)
            cmap = plt.get_cmap("plasma")
            col = cmap(float(np.log(f_now / f1) / np.log(f2 / f1)))
            age = (tc * 2.2) % 1.0
            _set_wavefronts(arcs, [11.5 * ((age + 0.25 * i) % 1.0)
                                   for i in range(4)], 11.5, color=col)
            freq_txt.set_text(f"f = {f_now:5.0f} Hz")
        else:
            _set_wavefronts(arcs, [0.0] * 4, 1.0)
            freq_txt.set_text("")
        arts += [*arcs, freq_txt]
        # spectrogram builds in sync with the playback
        n_col = int(np.searchsorted(times, t_sig))
        disp[:, :n_col] = sxx_db[:, :n_col]
        im.set_data(disp)
        ridge_txt.set_text(T("delayed copies = reflections")
                           if tc >= 3.4 else "")
        arts += [im, ridge_txt]
        # deconvolution: the record collapses into the impulse response
        if tc >= dec_t:
            _light_box(deconv, "", COLOR_SECONDARY, fill=True)
            reveal = float(np.clip((tc - dec_t) / 1.2, 0.0, 1.0))
            n = max(2, int(reveal * ir_n.size))
            ir_line.set_data(t_ir[:n], ir_n[:n])
            for mk in tap_marks:
                mk.set_visible(reveal >= 1.0)
            if reveal >= 1.0:
                note.set_text(
                    T("same information, different domain: sweep ⊛ inverse"
                      " filter = impulse response"))
                note.set_visible(True)
        else:
            _dim_box(deconv)
            ir_line.set_data([], [])
            note.set_text("")
            note.set_visible(False)
        arts += [ir_line, deconv["box"], deconv["title"], deconv["value"],
                 note, *tap_marks]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_sweep_deconvolution")


def animate_specific_loudness(output_dir: str) -> None:
    """ISO 532-1 specific loudness: the N'(z) pattern of a 1 kHz narrowband
    sound builds along the Bark axis as the band level steps up, and the
    area under the pattern integrates to the total loudness in sone."""
    T = _translate_str
    from phonometry import loudness_zwicker_from_spectrum

    bands = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
             500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
             6300, 8000, 10000, 12500]
    i_1k = bands.index(1000)
    levels = np.arange(40, 86)
    patterns: dict[int, Any] = {}
    totals: dict[int, float] = {}
    for lv in levels:
        third = [-60.0] * 28
        third[i_1k] = float(lv)
        res = loudness_zwicker_from_spectrum(third, field="free")
        patterns[int(lv)] = res.specific
        totals[int(lv)] = float(res.loudness)
    z = np.arange(1, 241) * 0.1

    fig = _anim_figure()
    fig.suptitle(T("Specific loudness N'(z) and its integral (ISO 532-1)"),
                 fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1.0])
    ax = fig.add_subplot(gs[0])
    _grid_axes(ax)
    ax.set_xlim(0.0, 24.0)
    # headroom above the 85 dB pattern (max N' ~ 5.4 sone/Bark)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel(T("Critical-band rate z [Bark]"))
    ax.set_ylabel(T("Specific loudness N' [sone/Bark]"), fontsize=9)
    (line,) = ax.plot([], [], color=COLOR_PRIMARY, lw=2.2)
    fill = {"art": None}
    ax.axvline(8.5, color=COLOR_FG, lw=0.9, ls=":", alpha=0.6)
    ax.text(8.75, 5.85, T("1 kHz ≈ 8.5 Bark"), ha="left", va="top",
            color=COLOR_FG, fontsize=8)
    spread = ax.annotate(T("upward spread of masking"), xy=(13.0, 1.1),
                         xytext=(16.0, 3.2), fontsize=8.5, color=COLOR_FG,
                         arrowprops={"arrowstyle": "->", "color": COLOR_FG,
                                     "lw": 1.0}, visible=False)
    level_txt = ax.text(0.02, 0.965, "", transform=ax.transAxes, ha="left",
                        va="top", color=COLOR_FG, fontsize=11,
                        family="monospace")

    ax_m = fig.add_subplot(gs[1])
    _schematic_axes(ax_m, (0.0, 4.0), (0.0, 8.6))
    _draw_speaker(ax_m, 0.55, 7.7, size=0.7)
    ax_m.text(1.15, 7.7, T("1 kHz narrowband"), ha="left", va="center",
              color=COLOR_FG, fontsize=8.5)
    gauge = _make_gauge(ax_m, 2.0, 4.6, 1.05, "N", COLOR_SECONDARY,
                        lo="0", hi=T("20 sone"))
    ax_m.text(2.0, 6.5, r"$N = \int_0^{24} N'(z)\, dz$", ha="center",
              va="bottom", color=COLOR_FG, fontsize=11)
    steps = [45, 65, 85]
    step_boxes = [_flow_box(ax_m, 2.0, 2.6 - 0.95 * j, 3.4, 0.8,
                            T(f"{lv} dB"))
                  for j, lv in enumerate(steps)]

    # Level trajectory: three plateaus with 1 s ramps between them.
    knots_t = (0.0, 2.9, 3.9, 6.6, 7.6, 10.0)
    knots_l = (45.0, 45.0, 65.0, 65.0, 85.0, 85.0)
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    # First plateau: an explicit left-to-right integration sweep.
    int_t0, int_t1 = 0.4, 2.6

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        lv = round(float(np.interp(tc, knots_t, knots_l)))
        spec = patterns[lv]
        line.set_data(z, spec)
        if tc <= int_t1:
            z_lim = float(np.interp(tc, (int_t0, int_t1), (0.0, 24.0)))
        else:
            z_lim = 24.0
        m = z <= z_lim
        if fill["art"] is not None:
            fill["art"].remove()
        fill["art"] = ax.fill_between(z[m], 0.0, spec[m],
                                      color=COLOR_PRIMARY, alpha=0.3, lw=0)
        n_part = float(np.trapezoid(spec[m], z[m])) if m.any() else 0.0
        n_show = totals[lv] if z_lim >= 24.0 else n_part
        level_txt.set_text(f"L = {lv} dB")
        spread.set_visible(lv >= 80)
        arts: list[Any] = [line, fill["art"], level_txt, spread]
        arts += _set_gauge(gauge, n_show / 20.0, T(f"{n_show:.1f} sone"))
        for j, (lv_s, box) in enumerate(zip(steps, step_boxes, strict=True)):
            done = (tc >= (int_t1, 5.2, 8.9)[j])
            if done:
                _light_box(box, T(f"N = {totals[lv_s]:.1f} sone"),
                           COLOR_SECONDARY if j == 2 else COLOR_PRIMARY,
                           fill=j == 2)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_specific_loudness")


def animate_power_two_rooms(output_dir: str) -> None:
    """The same source in an anechoic room (ISO 3745 free field) and in a
    reverberation room (ISO 3741 diffuse build-up): both microphone
    pressures differ, both routes converge to the same sound power L_W."""
    from matplotlib.patches import Circle, Polygon, Rectangle

    T = _translate_str
    lw_true = 92.0
    r_mic = 2.05                   # projected mic-ring radius (anechoic)
    lp_free = 77.5                 # L_W - 10 lg(4 pi 1.5^2)
    lp_diff = 86.0                 # L_W - 10 lg V + 10 lg T + 14

    fig = _anim_figure()
    fig.suptitle(T("One source, two rooms, one sound power"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    _schematic_axes(ax_a, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_r = fig.add_subplot(gs[0, 1])
    _schematic_axes(ax_r, (0.0, 8.0), (0.0, 6.6), equal=True)
    ax_b = fig.add_subplot(gs[1, :])
    _schematic_axes(ax_b, (0.0, 16.0), (0.0, 2.6))

    # --- anechoic room: wedges on every wall, mic ring, free field -------
    ax_a.set_title(T("Anechoic room (ISO 3745)"), fontsize=10,
                   fontweight="bold")
    ax_a.add_patch(Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none",
                             edgecolor=COLOR_FG, lw=1.4))
    wedge = 0.42
    for xw in np.arange(0.85, 7.3, 0.55):
        ax_a.add_patch(Polygon([(xw - 0.22, 0.4), (xw + 0.22, 0.4),
                                (xw, 0.4 + wedge)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
        ax_a.add_patch(Polygon([(xw - 0.22, 6.0), (xw + 0.22, 6.0),
                                (xw, 6.0 - wedge)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
    for yw in np.arange(0.85, 5.9, 0.55):
        ax_a.add_patch(Polygon([(0.6, yw - 0.22), (0.6, yw + 0.22),
                                (0.6 + wedge, yw)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
        ax_a.add_patch(Polygon([(7.4, yw - 0.22), (7.4, yw + 0.22),
                                (7.4 - wedge, yw)], closed=True,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.6))
    ca = (4.0, 3.2)
    _draw_speaker(ax_a, ca[0] - 0.05, ca[1], size=0.85)
    # offset by 15 deg so no dot lands on the caption lines at the bottom
    mic_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False) + np.pi / 12
    for ang in mic_angles:
        ax_a.plot([ca[0] + r_mic * np.cos(ang)],
                  [ca[1] + r_mic * np.sin(ang)], marker="o", ms=4,
                  color=COLOR_PRIMARY)
    ax_a.text(ca[0], ca[1] - r_mic + 0.14, T("microphone sphere, r"),
              ha="center", va="bottom", color=COLOR_PRIMARY, fontsize=8)
    arcs_a = _make_wavefronts(ax_a, *ca, COLOR_TERTIARY, n=4)
    note_a = ax_a.text(4.0, 0.92, T("direct sound only — no reflections"),
                       ha="center", va="bottom", color=COLOR_FG, fontsize=8,
                       alpha=0.0)
    lp_a = ax_a.text(6.9, 5.5, "", ha="right", va="top", color=COLOR_FG,
                     fontsize=9, family="monospace")

    # --- reverberation room: bare walls, diffuse build-up, one mic path --
    ax_r.set_title(T("Reverberation room (ISO 3741)"), fontsize=10,
                   fontweight="bold")
    # animated fill (the diffuse level building up) + a fixed outline
    room = Rectangle((0.6, 0.4), 6.8, 5.6, facecolor=COLOR_PRIMARY,
                     edgecolor="none", alpha=0.0)
    ax_r.add_patch(room)
    ax_r.add_patch(Rectangle((0.6, 0.4), 6.8, 5.6, facecolor="none",
                             edgecolor=COLOR_FG, lw=1.4))
    # a tilted diffuser panel hanging in the volume
    ax_r.plot([1.3, 2.7], [4.7, 5.3], color=COLOR_FG, lw=2.4, alpha=0.8)
    cr = (1.7, 1.3)
    _draw_speaker(ax_r, cr[0] - 0.05, cr[1], size=0.85)
    # bouncing ray trails (specular folding inside the rectangle)
    rays = [{"v": (2.9, 1.9), "trail": ax_r.plot(
        [], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0]},
        {"v": (2.2, -2.6), "trail": ax_r.plot(
            [], [], color=COLOR_TERTIARY, lw=1.1, alpha=0.75)[0]}]
    mic_path_c, mic_path_r = (4.4, 3.4), 1.35
    ax_r.add_patch(Circle(mic_path_c, mic_path_r, facecolor="none",
                          edgecolor=COLOR_PRIMARY, lw=0.9, ls="--",
                          alpha=0.8))
    (mic_dot,) = ax_r.plot([], [], marker="o", ms=6, color=COLOR_PRIMARY)
    ax_r.text(mic_path_c[0], mic_path_c[1] - mic_path_r - 0.28,
              T("rotating microphone"), ha="center", va="top",
              color=COLOR_PRIMARY, fontsize=8)
    note_r = ax_r.text(4.0, 0.62, T("reflections build a diffuse field"),
                       ha="center", va="bottom", color=COLOR_FG, fontsize=8,
                       alpha=0.0)
    lp_r = ax_r.text(6.9, 5.5, "", ha="right", va="top", color=COLOR_FG,
                     fontsize=9, family="monospace")

    def _fold(p: float, lo: float, hi: float) -> float:
        span = hi - lo
        q = (p - lo) % (2.0 * span)
        return lo + (q if q <= span else 2.0 * span - q)

    # --- bottom strip: the two formulas converge on one L_W --------------
    box_a = _flow_box(ax_b, 3.4, 1.55, 6.2, 1.15,
                      r"$L_W = \bar{L}_p + 10\,\lg(4\pi r^2/S_0)$")
    box_r = _flow_box(ax_b, 12.6, 1.55, 6.2, 1.15,
                      r"$L_W = \bar{L}_p + 10\,\lg V - 10\,\lg T - 14$")
    lw_box = _flow_box(ax_b, 8.0, 1.3, 2.6, 1.3, "$L_W$")
    arr_a = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    arr_r = _make_arrow(ax_b, COLOR_SECONDARY, scale=13.0)
    for arr in (arr_a, arr_r):
        arr.set_visible(False)
    verdict = ax_b.text(8.0, 0.12, "", ha="center", va="bottom",
                        color=COLOR_SECONDARY, fontsize=10,
                        fontweight="bold")

    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS
    t_meter, t_form, t_conv = 4.5, 7.0, 9.4

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        arts: list[Any] = []
        # free field: wavefronts die before the wedges
        age = (tc * 0.55) % 1.0
        _set_wavefronts(arcs_a, [2.9 * ((age + 0.25 * i) % 1.0)
                                 for i in range(4)], 2.9, alpha=0.8)
        arts += arcs_a
        note_a.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        # diffuse field: rays bounce, the background level rises
        build = float(np.clip(tc / 4.0, 0.0, 1.0))
        room.set_alpha(0.16 * (1.0 - np.exp(-3.0 * build)))
        for ray in rays:
            ts = np.linspace(max(0.0, tc - 0.9), tc, 24)
            xs = [_fold(cr[0] + ray["v"][0] * s, 0.7, 7.3) for s in ts]
            ys = [_fold(cr[1] + ray["v"][1] * s, 0.5, 5.9) for s in ts]
            ray["trail"].set_data(xs, ys)
            arts.append(ray["trail"])
        ang = 2.0 * np.pi * 0.12 * tc
        mic_dot.set_data([mic_path_c[0] + mic_path_r * np.cos(ang)],
                         [mic_path_c[1] + mic_path_r * np.sin(ang)])
        note_r.set_alpha(min(tc / 2.0, 1.0) * 0.9)
        arts += [room, mic_dot, note_a, note_r]
        # microphone readings appear, then each formula computes L_W
        if tc >= t_meter:
            lp_a.set_text(T(f"mean Lp = {lp_free:.1f} dB"))
            lp_r.set_text(T(f"mean Lp = {lp_diff:.1f} dB"))
        else:
            lp_a.set_text("")
            lp_r.set_text("")
        # Mathtext subscript for the sound-power symbol; localise the decimal
        # comma by hand because ``$`` disables the ``_translate_str`` comma pass.
        lw_val = f"{lw_true:.1f}" if _LANG == "en" else f"{lw_true:.1f}".replace(".", ",")
        for box, on in ((box_a, tc >= t_form), (box_r, tc >= t_form + 0.6)):
            if on:
                _light_box(box, f"$L_W$ = {lw_val} dB", COLOR_PRIMARY)
            else:
                _dim_box(box)
            arts += [box["box"], box["title"], box["value"]]
        if tc >= t_conv:
            _light_box(lw_box, T(f"{lw_true:.1f} dB"), COLOR_SECONDARY,
                       fill=True)
            arr_a.set_positions((6.6, 1.55), (7.3, 1.4))
            arr_r.set_positions((9.4, 1.55), (8.7, 1.4))
            arr_a.set_visible(True)
            arr_r.set_visible(True)
            verdict.set_text(
                T("the room changes Lp, not the source power"))
        else:
            _dim_box(lw_box)
            arr_a.set_visible(False)
            arr_r.set_visible(False)
            verdict.set_text("")
        arts += [lp_a, lp_r, lw_box["box"], lw_box["title"],
                 lw_box["value"], arr_a, arr_r, verdict]
        return tuple(arts)

    _render_clip(fig, update, output_dir, "anim_power_two_rooms")


def animate_comb_filtering(output_dir: str) -> None:
    """Direct sound plus one floor reflection at a microphone: as the mic
    height changes, the delayed copy shifts and the comb filter in the
    frequency response moves with it, which is why measurement position matters
    near reflecting surfaces."""
    T = _translate_str
    c0 = 343.0
    xs_, hs = 0.6, 1.5              # source position (x, height)
    xm = 4.2                        # mic x position
    refl_g = 0.8                    # floor reflection factor

    fig = _anim_figure()
    fig.suptitle(T("Comb filtering from a single reflection"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    _schematic_axes(ax, (0.0, 5.6), (-2.3, 2.6), equal=True)
    ax.axhline(0.0, color=COLOR_FG, lw=2.0)
    ax.fill_between([0.0, 5.6], -2.3, 0.0, color=COLOR_GRID, alpha=0.35,
                    lw=0)
    ax.text(5.5, -0.34, T("reflecting floor"), ha="right", va="top",
            color=COLOR_FG, fontsize=8.5)
    _draw_speaker(ax, xs_, hs, size=0.8)
    ax.plot([xs_ - 0.25, xs_ - 0.25], [0.0, hs - 0.28], color=COLOR_FG,
            lw=1.2)
    # image source below the floor
    _draw_speaker(ax, xs_, -hs, size=0.8)
    for art in ax.patches[-2:]:
        art.set_alpha(0.3)
    ax.text(xs_ + 0.4, -hs, T("image source"), ha="left", va="center",
            color=COLOR_FG, fontsize=8.5, alpha=0.75)
    (l_dir,) = ax.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (l_ref,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.6, ls="--")
    (l_img,) = ax.plot([], [], color=COLOR_SECONDARY, lw=1.0, ls=":",
                       alpha=0.5)
    (mic_stand,) = ax.plot([], [], color=COLOR_FG, lw=1.2)
    (mic_dot,) = ax.plot([], [], marker="o", ms=8, color=COLOR_GRID,
                         markeredgecolor=COLOR_FG, markeredgewidth=1.2)
    mic_lab = ax.text(xm + 0.18, 0.0, T("mic"), ha="left", va="center",
                      color=COLOR_FG, fontsize=8.5)
    delta_txt = ax.text(0.15, 2.45, "", ha="left", va="top",
                        color=COLOR_FG, fontsize=9.5, family="monospace")
    stage_txt = ax.text(3.1, -2.05, "", ha="center", va="center",
                        color=COLOR_FG, fontsize=9, fontweight="bold")

    ax_t = fig.add_subplot(gs[0, 1])
    _grid_axes(ax_t)
    ax_t.set_xlim(9.0, 25.0)
    ax_t.set_ylim(0.0, 1.15)
    ax_t.set_xlabel(T("arrival time [ms]"), fontsize=8)
    ax_t.set_ylabel(T("amplitude"), fontsize=8)
    ax_t.tick_params(labelsize=7)
    (stem_d,) = ax_t.plot([], [], color=COLOR_PRIMARY, lw=2.6,
                          solid_capstyle="butt")
    (stem_r,) = ax_t.plot([], [], color=COLOR_SECONDARY, lw=2.6,
                          solid_capstyle="butt")
    tau_ann = ax_t.annotate("", xy=(0.0, 0.9), xytext=(0.0, 0.9),
                            arrowprops={"arrowstyle": "<->",
                                        "color": COLOR_FG, "lw": 1.0})
    tau_txt = ax_t.text(0.0, 0.98, "τ", ha="center", va="bottom",
                        color=COLOR_FG, fontsize=9)
    ax_t.text(0.97, 0.92, T("direct"), transform=ax_t.transAxes, ha="right",
              va="top", color=COLOR_PRIMARY, fontsize=8)
    ax_t.text(0.97, 0.80, T("delayed copy"), transform=ax_t.transAxes,
              ha="right", va="top", color=COLOR_SECONDARY, fontsize=8)

    ax_f = fig.add_subplot(gs[1, 1])
    _grid_axes(ax_f)
    f = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    ax_f.set_xscale("log")
    ax_f.set_xlim(50.0, 8000.0)
    ax_f.set_ylim(-16.0, 8.0)
    ax_f.set_xlabel(T("Frequency [Hz]"), fontsize=8)
    ax_f.set_ylabel(T("response [dB]"), fontsize=8)
    ax_f.tick_params(labelsize=7)
    (comb,) = ax_f.plot([], [], color=COLOR_PRIMARY, lw=1.8)
    (notch_dot,) = ax_f.plot([], [], marker="v", ms=7, ls="none",
                             color=COLOR_SECONDARY)
    notch_txt = ax_f.text(0.03, 0.97, "", transform=ax_f.transAxes,
                          ha="left", va="top", color=COLOR_SECONDARY,
                          fontsize=8.5, family="monospace")

    # Mic height trajectory: three plateaus (high, mid, on the floor).
    knots_t = (0.0, 2.6, 3.8, 6.2, 7.4, 10.0)
    knots_h = (1.5, 1.5, 0.6, 0.6, 0.02, 0.02)
    stages = ((2.0, T("high mic: dense comb")),
              (5.6, T("lower: notches move up")),
              (9.0, T("on the floor: copies merge — no comb in band")))
    sweep_s = (_ANIM_FRAMES - _ANIM_HOLD) / _ANIM_FPS

    def update(kf: int) -> tuple[Any, ...]:
        tc = min(kf / _ANIM_FPS, sweep_s)
        hm = float(np.interp(tc, knots_t, knots_h))
        r1 = float(np.hypot(xm - xs_, hm - hs))
        r2 = float(np.hypot(xm - xs_, hm + hs))
        tau = (r2 - r1) / c0
        # geometry: direct ray, floor bounce and the image-source ray
        xb = xs_ + (xm - xs_) * hs / (hs + hm)
        l_dir.set_data([xs_, xm], [hs, hm])
        l_ref.set_data([xs_, xb, xm], [hs, 0.0, hm])
        l_img.set_data([xs_, xm], [-hs, hm])
        mic_stand.set_data([xm, xm], [0.0, hm])
        mic_dot.set_data([xm], [hm])
        mic_lab.set_position((xm + 0.18, hm + 0.16))
        # three decimals keep the two readouts mutually consistent on
        # screen even for the near-zero floor geometry (Δ = c·τ)
        delta_txt.set_text(
            T(f"Δ = {r2 - r1:.3f} m   τ = {tau * 1e3:.3f} ms"))
        # time domain: two arrivals separated by tau
        t1, t2 = r1 / c0 * 1e3, r2 / c0 * 1e3
        stem_d.set_data([t1, t1], [0.0, 1.0])
        stem_r.set_data([t2, t2], [0.0, refl_g])
        tau_ann.xy = (t1, 0.9)
        tau_ann.set_position((t2, 0.9))
        tau_txt.set_position(((t1 + t2) / 2.0, 0.93))
        # hide both the label and the double arrow once the two arrivals
        # merge, otherwise the collapsed arrow reads as a stray glyph
        tau_txt.set_visible(t2 - t1 > 0.6)
        tau_ann.set_visible(t2 - t1 > 0.6)
        # frequency domain: |1 + g e^{-j 2 pi f tau}|
        h = np.abs(1.0 + refl_g * np.exp(-2j * np.pi * f * tau))
        comb.set_data(f, 20.0 * np.log10(h))
        f1n = 1.0 / (2.0 * tau) if tau > 0.0 else np.inf
        if f1n <= 8000.0:
            notch_dot.set_data([f1n], [20.0 * np.log10(1.0 - refl_g) + 1.2])
            notch_txt.set_text(T(f"first notch {f1n:.0f} Hz"))
        else:
            notch_dot.set_data([], [])
            notch_txt.set_text(T("first notch above 8 kHz"))
        stage = ""
        for t_s, s in stages:
            if tc >= t_s - 1.4:
                stage = s
        stage_txt.set_text(stage)
        return (l_dir, l_ref, l_img, mic_stand, mic_dot, mic_lab, delta_txt,
                stem_d, stem_r, tau_ann, tau_txt, comb, notch_dot,
                notch_txt, stage_txt)

    _render_clip(fig, update, output_dir, "anim_comb_filtering")


# ---------------------------------------------------------------------------
# FDTD wave-field clips, second batch (slit absorber, expansion chamber,
# wall aperture, atmospheric refraction). Same conventions as the first
# batch: each simulation runs once per process behind an lru_cache, every
# clip keeps >= 48 captured frames per period of its highest carrier, and
# every number printed on screen comes from the library models, never from
# the simulation itself.
# ---------------------------------------------------------------------------


_SLIT_ABS_F0 = 300.0                     # critical-coupling design frequency
_SLIT_ABS_TUBE = 0.60                    # air column before the panel face
_SLIT_ABS_PERIOD = 5.0e-2                # panel period d = the tube bore
_SLIT_ABS_LATTICE = 3.0e-2               # lattice step a = panel depth (N=1)
_SLIT_ABS_DETUNE = 1.7                   # wide-slit factor of the alpha figure


@lru_cache(maxsize=1)
def _slit_absorber_design() -> Any:
    """The 300 Hz critical-coupling design of the slow-sound figures."""
    from phonometry import HelmholtzResonator, critical_coupling_design

    base = HelmholtzResonator(neck_length=1.0e-3, neck_side=3.0e-3,
                              cavity_length=30.0e-3, cavity_side=27.0e-3)
    return critical_coupling_design(_SLIT_ABS_F0, base,
                                    lattice_step=_SLIT_ABS_LATTICE,
                                    period=_SLIT_ABS_PERIOD)


def _lossy_fluid_params(rho_c: complex,
                        kappa_c: complex) -> tuple[float, float, float]:
    """Real-coefficient FDTD equivalent of a complex effective fluid.

    The solver carries plane waves through a uniform lossy region as
    ``k = (omega - j sigma)/c`` at the real impedance ``rho c``, so a
    visco-thermal effective density/bulk-modulus pair maps onto the phase
    speed ``omega / Re(k)``, the density ``Re(Z)/c`` and the decay rate
    ``sigma = -Im(k) c`` at the drive frequency. Returns ``(c, rho, sigma)``.
    """
    omega = 2.0 * np.pi * _SLIT_ABS_F0
    k = omega * complex(np.sqrt(rho_c / kappa_c))
    z = complex(np.sqrt(rho_c * kappa_c))
    c_ph = omega / k.real
    return c_ph, z.real / c_ph, -k.imag * c_ph


def _slit_absorber_cell_rows(dx: float, slit_cells: int,
                             ny: int) -> dict[str, tuple[int, int]]:
    """Row/column spans of the meshed cell (slit at the top of the period,
    neck and cavity below it, mirroring ``plot_slit_absorber_geometry``)."""
    res = _slit_absorber_design().resonator
    neck_cells = round(res.neck_length / dx)
    cav_cells = round(res.cavity_length / dx)
    slit0 = ny - slit_cells
    neck0 = slit0 - neck_cells
    return {"slit": (slit0, ny), "neck": (neck0, slit0),
            "cavity": (neck0 - cav_cells, neck0)}


@lru_cache(maxsize=1)
def _slit_absorber_fields(
    n_frames: int = _FDTD_ANIM_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """CW build-up against the meshed critical-coupling cell, cached.

    Two runs of a 0.60 m plane-wave tube whose bore is one 50 mm panel
    period, driven at 300 Hz through a rho c left edge; the right end is
    the panel itself: slit, neck and cavity carved out of the rigid body
    with the obstacle mask at ``dx = h/4`` (the slit is the smallest
    feature) and filled with the library's visco-thermal effective fluids
    mapped onto real ``(c, rho, sigma)`` triplets. The neck and cavity are
    square ducts of side ``w`` in 3D, so the 2D slice scales their density
    (and with it ``kappa = rho c^2``) by the out-of-plane ratio ``a / w``,
    keeping the 3D cell's acoustic mass and compliance per unit depth. Run
    one is the critical design; run two detunes only the slit height by
    the wide-slit factor of the absorption figure. ``cfl = 0.95``: at
    lambda/4700 the numerical dispersion is negligible and the wide margin
    of the default would just double the run time. Returns per-run (tube
    frames, cell-zoom frames, the settled |p| envelope), the frame times
    and the library absorption at 300 Hz for both slit heights.
    """
    import fdtd2d

    from phonometry import slit_helmholtz_absorber
    from phonometry.materials import (
        rectangular_duct_properties,
        slit_effective_properties,
    )

    design = _slit_absorber_design()
    res = design.resonator
    h0 = design.slit_height
    # Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
    # carrier) = min(h/4 = 0.244 mm, 1.143 m / 8 = 143 mm) -> the slit
    # height governs and the grid is h/4 = 0.244 mm (lambda/4677).
    dx = h0 / 4.0
    ny = round(_SLIT_ABS_PERIOD / dx)
    face = round(_SLIT_ABS_TUBE / dx)
    nx = face + round(_SLIT_ABS_LATTICE / dx)
    zoom0 = round((_SLIT_ABS_TUBE - 0.015) / dx)
    x_mouth = _SLIT_ABS_TUBE + 0.5 * _SLIT_ABS_LATTICE
    neck_c = (round((x_mouth - 0.5 * res.neck_side) / dx),
              round((x_mouth + 0.5 * res.neck_side) / dx))
    cav_c = (round((x_mouth - 0.5 * res.cavity_side) / dx),
             round((x_mouth + 0.5 * res.cavity_side) / dx))
    freq = np.array([_SLIT_ABS_F0])
    neck_f = _lossy_fluid_params(
        *(p.item() for p in rectangular_duct_properties(
            freq, side=res.neck_side)))
    cav_f = _lossy_fluid_params(
        *(p.item() for p in rectangular_duct_properties(
            freq, side=res.cavity_side)))
    # 2D-slice calibration: the meshed cell's slot-shaped mouths carry
    # smaller end-correction masses than the square ducts of the 3D
    # model, and in this slow-sound cell the resonance is the slit
    # inertance against the cavity compliance, so the whole cell lands
    # ~11 % high: a broadband two-microphone probe of the meshed cell
    # puts its absorption peak at 333 Hz instead of 300. Rescaling the
    # cavity sound speed by 300/333 = 0.90 restores the design
    # resonance; the same probe then measures alpha = 0.99 at 300.0 Hz,
    # the critical coupling the annotation quotes. The rescale is fitted
    # at the critical slit height only and does not transfer exactly to
    # the detuned row (the wide-slit cell measures |R| ~ 0.95 where the
    # 3D model gives 0.81); the visual contrast and the qualitative
    # message are the correct ones, and the alpha pill quotes the
    # library model's value.
    cav_f = (0.90 * cav_f[0], cav_f[1], cav_f[2])
    scale_neck = _SLIT_ABS_LATTICE / res.neck_side
    scale_cav = _SLIT_ABS_LATTICE / res.cavity_side

    alphas = (float(design.absorption),
              float(slit_helmholtz_absorber(
                  freq, res, slit_height=_SLIT_ABS_DETUNE * h0,
                  lattice_step=_SLIT_ABS_LATTICE,
                  period=_SLIT_ABS_PERIOD).absorption[0]))
    # Clip duration per the deepest-reflector rule: d(source -> cavity
    # bottom through slit mouth and neck) = 0.60 + 0.015 + 0.001 + 0.0447
    # = 0.6607 m, the same back to the farthest visible point (the frame's
    # left edge), over the slowest medium on the path (the slit fluid,
    # 313.7 m/s): t = 1.2 * 2 * 0.6607 / 313.7 = 5.06 ms -> every = 30
    # (5.17 ms captured, 232 frames per 300 Hz period >= 48). The window
    # is far shorter than the resonator ring-up (~6 ms time constant), so
    # both runs first settle uncaptured for 20 ms and the clip shows the
    # steady field in slow motion with its exact settled envelope.
    every = 30
    runs = []
    times = np.zeros(0)
    for h in (h0, _SLIT_ABS_DETUNE * h0):
        slit_f = _lossy_fluid_params(
            *(p.item() for p in slit_effective_properties(
                freq, slit_height=h)))
        rows = _slit_absorber_cell_rows(dx, round(h / dx), ny)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[:, face:] = True
        c_map = np.full((ny, nx), 343.0)
        rho_map = np.full((ny, nx), 1.2)
        sig_map = np.zeros((ny, nx))
        for name, (c_e, rho_e, sig_e), scale, cols in (
                ("slit", slit_f, 1.0, (face, nx)),
                ("neck", neck_f, scale_neck, neck_c),
                ("cavity", cav_f, scale_cav, cav_c)):
            r0, r1 = rows[name]
            sl = (slice(r0, r1), slice(cols[0], cols[1]))
            mask[sl] = False
            c_map[sl] = c_e
            rho_map[sl] = rho_e * scale
            sig_map[sl] = sig_e
        sim = fdtd2d.FDTD2D(c_map, dx, rho=rho_map, damping=sig_map,
                            cfl=0.95, obstacle_mask=mask,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_SLIT_ABS_F0)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value, offset=2))
        mid = ny // 2
        # Settle to steady state, sampling the exact standing-wave
        # envelope over the final full period.
        period = round(1.0 / (_SLIT_ABS_F0 * sim.dt))
        settle = round(0.020 / sim.dt)
        env = np.zeros(sim.p[mid, :face:2].shape, dtype=np.float32)
        for i in range(settle):
            sim.step()
            if i >= settle - period:
                np.maximum(env, np.abs(sim.p[mid, :face:2]), out=env)
        tube: list[Any] = []
        cell: list[Any] = []
        ts: list[float] = []
        while len(tube) < n_frames:
            sim.step()
            if sim.n % every == 0:
                tube.append(sim.p[::4, ::4].astype(np.float32))
                cell.append(sim.p[:, zoom0:].astype(np.float32))
                ts.append(sim.time)
        runs.append((np.stack(tube), np.stack(cell), env))
        times = np.asarray(ts)
    return runs[0], runs[1], times, alphas


def animate_fdtd_slit_absorber(output_dir: str) -> None:
    """Inside the slow-sound slit absorber (2D FDTD): a 300 Hz plane tone
    meets the meshed critical-coupling cell, whose sub-millimetre slit and
    Helmholtz resonator are resolved on the grid and filled with the
    library's visco-thermal effective fluids. At the design slit height
    the loss/leakage balance swallows the wave (alpha = 1, flat envelope);
    widening the slit breaks the balance and the reflection rebuilds the
    standing wave. A zoom shows the field crawling through the slit."""
    from matplotlib.patches import Rectangle

    T = _translate_str
    (t_c, z_c, e_c), (t_w, z_w, e_w), times, alphas = _slit_absorber_fields()
    design = _slit_absorber_design()
    res = design.resonator
    h0 = design.slit_height
    dx = h0 / 4.0
    ny = round(_SLIT_ABS_PERIOD / dx)
    nx = round(_SLIT_ABS_TUBE / dx) + round(_SLIT_ABS_LATTICE / dx)
    x_end = nx * dx
    x_zoom0 = round((_SLIT_ABS_TUBE - 0.015) / dx) * dx
    bore = ny * dx
    half = t_c.shape[0] // 2
    # Color scale from the TUBE columns only: at critical coupling the
    # cavity rings at several times the incident amplitude and would
    # otherwise wash the travelling/standing waves out of the map (the
    # cell interior saturating instead is the point).
    tube_cols = round(_SLIT_ABS_TUBE / dx) // 4
    vmax = float(np.quantile(np.abs(t_c[half:, :, :tube_cols]), 0.999))
    env_base, env_h, env_max = 0.074, 0.034, 2.3
    x_env = (np.arange(e_c.shape[0]) + 0.5) * 2.0 * dx
    env_from = 10                          # hide the injection-line step

    fig = _anim_figure()
    fig.suptitle(T("Slow-sound slit absorber at critical coupling "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 0.52])
    titles = [T("Critically coupled slit: the wave dies inside"),
              T("Wide slit (detuned): the reflection returns")]
    heights = (h0, _SLIT_ABS_DETUNE * h0)
    envs = (e_c, e_w)
    ims: list[Any] = []
    ims_zoom: list[Any] = []
    a_txts: list[Any] = []
    for row, (title, h) in enumerate(zip(titles, heights, strict=True)):
        rows = _slit_absorber_cell_rows(dx, round(h / dx), ny)
        # Cell silhouette: the rigid part of the panel as five rectangles
        # around the slit/neck/cavity openings, so both views show the
        # field inside the openings through the gaps.
        y_slit0 = rows["slit"][0] * dx
        y_neck = tuple(r * dx for r in rows["neck"])
        y_cav = tuple(r * dx for r in rows["cavity"])
        x_face = _SLIT_ABS_TUBE
        x_mouth = _SLIT_ABS_TUBE + 0.5 * _SLIT_ABS_LATTICE
        x_neck = (x_mouth - 0.5 * res.neck_side,
                  x_mouth + 0.5 * res.neck_side)
        x_cav = (x_mouth - 0.5 * res.cavity_side,
                 x_mouth + 0.5 * res.cavity_side)
        body = [
            (x_face, 0.0, x_end - x_face, y_cav[0]),
            (x_face, y_cav[0], x_cav[0] - x_face, y_cav[1] - y_cav[0]),
            (x_cav[1], y_cav[0], x_end - x_cav[1], y_cav[1] - y_cav[0]),
            (x_face, y_neck[0], x_neck[0] - x_face, y_neck[1] - y_neck[0]),
            (x_neck[1], y_neck[0], x_end - x_neck[1],
             y_neck[1] - y_neck[0]),
        ]
        ax = fig.add_subplot(gs[row, 0])
        ax.grid(False)
        im = ax.imshow(np.zeros((2, 2)), origin="lower",
                       extent=(0.0, x_end, 0.0, bore), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="bilinear", zorder=2)
        _anim_slit_tube_walls(ax, x_end, bore, speaker=row == 0)
        for rect in body:
            ax.add_patch(Rectangle(rect[:2], rect[2], rect[3],
                                   facecolor="#8b8b8b",
                                   edgecolor=COLOR_FG, lw=0.4, zorder=3))
        # Dashed magnifier frame around the cell, echoed on the zoom axes.
        ax.add_patch(Rectangle((x_zoom0, 0.0), x_end - x_zoom0, bore,
                               facecolor="none", edgecolor=COLOR_FG,
                               ls="--", lw=0.8, zorder=5))
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.text(0.005, env_base + 0.004, "|p| envelope", fontsize=7.5,
                ha="left", va="bottom", color=COLOR_FG, alpha=0.8)
        # The settled standing-wave envelope, static: the clip captures
        # the steady state, where it no longer changes.
        ax.plot(x_env[env_from:],
                env_base + envs[row][env_from:] / env_max * env_h,
                color=COLOR_PRIMARY, lw=1.8, zorder=6)
        ax.set_xlim(-0.115, x_end + 0.022)
        # Headroom above the envelope trace keeps the alpha pill clear
        # of the curve in both rows.
        ax.set_ylim(-0.033, env_base + env_h + 0.026)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(T("Position along the tube [m]"), fontsize=9)
        a_txt = ax.text(0.03, env_base + env_h + 0.023, "", ha="left",
                        va="top", fontsize=9, color="white", zorder=7,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        # Zoom: the last 45 mm at full grid resolution, geometry to scale.
        ax_z = fig.add_subplot(gs[row, 1])
        ax_z.grid(False)
        im_z = ax_z.imshow(np.zeros((2, 2)), origin="lower",
                           extent=(x_zoom0, x_end, 0.0, bore),
                           cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                           aspect="equal", interpolation="nearest",
                           zorder=2)
        for rect in body:
            ax_z.add_patch(Rectangle(rect[:2], rect[2], rect[3],
                                     facecolor="#8b8b8b",
                                     edgecolor=COLOR_FG, lw=0.5, zorder=3))
        for spine in ax_z.spines.values():
            spine.set_linestyle("--")
            spine.set_edgecolor(COLOR_FG)
        ax_z.set_xlim(x_zoom0, x_end)
        ax_z.set_ylim(0.0, bore)
        ax_z.set_xticks([])
        ax_z.set_yticks([])
        ax_z.annotate(
            T(f"slit h = {h * 1e3:.2f} mm"),
            xy=(x_zoom0 + 0.006, y_slit0 + 0.5 * h),
            xytext=(x_zoom0 + 0.0035, bore - 0.0095),
            fontsize=7, color=COLOR_FG, ha="left", va="top",
            arrowprops={"arrowstyle": "-", "color": COLOR_FG, "lw": 0.7},
            zorder=6,
            bbox={"boxstyle": "round,pad=0.2",
                  "facecolor": fig.get_facecolor(), "alpha": 0.55,
                  "edgecolor": "none"})
        ax_z.text(x_mouth, 0.5 * (y_cav[0] + y_cav[1]),
                  T("Helmholtz resonator"), ha="center", va="center",
                  fontsize=6.5, color=COLOR_FG, zorder=6,
                  bbox={"boxstyle": "round,pad=0.2",
                        "facecolor": fig.get_facecolor(), "alpha": 0.55,
                        "edgecolor": "none"})
        if row == 0:
            ax_z.set_title(T("inside the cell"), fontsize=8.5)
        ims.append(im)
        ims_zoom.append(im_z)
        a_txts.append(a_txt)
    # Bottom-right corner: the top corners belong to the suptitle and the
    # zoom-column title, and the x-label of the tube column sits centred
    # well to the left of this spot.
    t_txt = fig.text(0.988, 0.015, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The captured field is already steady, so the verdict can come early.
    reveal = int(0.30 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        for i, (t_all, z_all, alpha) in enumerate(
                ((t_c, z_c, alphas[0]), (t_w, z_w, alphas[1]))):
            ims[i].set_data(t_all[k])
            ims_zoom[i].set_data(z_all[k])
            a_txts[i].set_text(
                T(f"alpha = {alpha:.2f} at {_SLIT_ABS_F0:.0f} Hz")
                if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
        return (*ims, *ims_zoom, *a_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_slit_absorber",
                 frames=len(times), gif_fps=8)


def _anim_slit_tube_walls(ax: Any, length: float, bore: float, *,
                          speaker: bool = True) -> None:
    """Tube walls and drive loudspeaker for the slit-absorber clip (the
    shared ``_anim_tube_hardware`` labels its termination as a plug, but
    here the termination IS the meshed panel, drawn by the caller)."""
    from matplotlib.patches import Rectangle

    wall = 0.007
    grey = "#9a9a9a"
    for y0 in (-wall, bore):
        ax.add_patch(Rectangle((0.0, y0), length, wall, facecolor=grey,
                               edgecolor=COLOR_FG, linewidth=0.7, zorder=3))
    _anim_speaker(ax, 0.0, 0.5 * bore, bore,
                  label_y=-0.30 * bore if speaker else None)


_CHAMBER_L = 0.30                        # chamber length of the TL example
_CHAMBER_M = 4.0                         # expansion area ratio
_CHAMBER_PIPE_A = 0.01                   # pipe area of the TL example [m2]
# 2D pipe height. The four-pole TL depends only on m and L, but the 2D
# junction end correction grows with the bore; a 50 mm pipe (a small
# muffler with a 200 x 300 mm chamber) keeps the sudden-area-change
# phase error at kL = pi below ~0.15 dB so the simulated pass band
# actually matches the model's TL = 0 within the line width.
_CHAMBER_BORE = 0.05
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
# highest carrier) = min(50 mm / 4 = 12.5 mm, 0.6 m / 8 = 75 mm); the
# grid runs finer still (2.5 mm, 20 cells across the pipe bore) so the
# junction fields render smoothly at negligible cost.
_CHAMBER_DX = 0.0025


def _expansion_chamber_freqs() -> tuple[float, float]:
    """The two CW carriers: full transmission at ``kL = pi`` and the TL
    peak at ``kL = pi/2`` of the 0.30 m chamber (Bies Eq. (8.111))."""
    return 343.0 / (2.0 * _CHAMBER_L), 343.0 / (4.0 * _CHAMBER_L)


@lru_cache(maxsize=1)
def _expansion_chamber_fields(
    n_frames: int = _FDTD_ANIM_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """Two CW runs through the m = 4 expansion chamber, cached.

    A 1.4 m duct at the L and m of the noise-control guide's example
    (50 mm pipe, 0.30 m chamber, area ratio 4 as the 2D height ratio),
    walls built from a 10^6:1 density contrast so the sustained plane wave
    can be injected across the full left edge (an obstacle mask would
    reject the injection line). rho c edges at both ends make the source
    non-reflecting and the termination anechoic, so the outlet carries
    pure transmission. Returns the wall-masked frame stacks (pass band,
    stop band), the settled centreline |p| envelopes, the frame times and
    the library transmission losses at the two carriers.
    """
    import fdtd2d

    from phonometry import expansion_chamber

    dx = _CHAMBER_DX
    pipe_cells = round(_CHAMBER_BORE / dx)               # 20 cells
    ny = round(_CHAMBER_M * _CHAMBER_BORE / dx)          # 80 cells
    nx = round(1.4 / dx)
    c0 = round(0.55 / dx)
    c1 = c0 + round(_CHAMBER_L / dx)
    p0 = (ny - pipe_cells) // 2
    air = np.zeros((ny, nx), dtype=bool)
    air[p0:p0 + pipe_cells, :] = True
    air[:, c0:c1] = True
    rho_map = np.where(air, 1.2, 1.2e6)
    f_pass, f_peak = _expansion_chamber_freqs()
    tls = expansion_chamber(
        np.array([f_pass, f_peak]), _CHAMBER_L,
        _CHAMBER_M * _CHAMBER_PIPE_A, _CHAMBER_PIPE_A).transmission_loss
    # Clip duration per the deepest-reflector rule: d(source -> chamber
    # outlet plate, the deepest reflecting feature) = 0.85 m, plus 0.85 m
    # back to the farthest visible field point (the duct inlet at x = 0):
    # t = 1.2 * 1.7 / 343 = 5.95 ms -> every = 6 (6.68 ms captured, 94
    # frames per 572 Hz period >= 48). The window is shorter than the
    # standing wave's build-up, so both runs settle uncaptured for 30 ms
    # first and the clip shows the steady field with its exact envelope.
    every = 6
    runs = []
    times = np.zeros(0)
    for f in (f_pass, f_peak):
        sim = fdtd2d.FDTD2D(343.0, dx, shape=(ny, nx), rho=rho_map,
                            edge_impedance={"left": 1.2 * 343.0,
                                            "right": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=f)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        mid = ny // 2
        period = round(1.0 / (f * sim.dt))
        settle = round(0.030 / sim.dt)
        env = np.zeros(sim.p[mid, ::2].shape, dtype=np.float32)
        for i in range(settle):
            sim.step()
            if i >= settle - period:
                np.maximum(env, np.abs(sim.p[mid, ::2]), out=env)
        ps: list[Any] = []
        ts: list[float] = []
        while len(ps) < n_frames:
            sim.step()
            if sim.n % every == 0:
                # The dense wall cells ring with the injection line;
                # blank them to NaN (transparent under imshow) so the
                # display and its color scale only see the air path.
                ps.append(np.where(air, sim.p, np.nan)[::2, ::2]
                          .astype(np.float32))
                ts.append(sim.time)
        runs.append((np.stack(ps), env))
        times = np.asarray(ts)
    return runs[0], runs[1], times, (float(tls[0]), float(tls[1]))


def animate_fdtd_expansion_chamber(output_dir: str) -> None:
    """The reactive expansion chamber (2D FDTD): the same silencer at its
    two characteristic frequencies. At kL = pi the chamber is a
    half-wavelength resonator and the wave crosses as if the chamber were
    not there (TL = 0); at kL = pi/2 the two area jumps reflect in phase
    (the pi round trip across the chamber returns both echoes to the
    inlet in phase) and the wave is sent back up the inlet, leaving the
    outlet at less than half amplitude (the 6.5 dB four-pole TL peak).
    No absorption anywhere: a purely reactive silencer."""
    T = _translate_str
    (p_pass, e_pass), (p_stop, e_stop), times, tls = (
        _expansion_chamber_fields())
    f_pass, f_peak = _expansion_chamber_freqs()
    dx = _CHAMBER_DX
    ny = round(_CHAMBER_M * _CHAMBER_BORE / dx)
    nx = round(1.4 / dx)
    length, height = nx * dx, ny * dx
    pipe_y = ((height - _CHAMBER_BORE) / 2.0,
              (height + _CHAMBER_BORE) / 2.0)
    env_base, env_h, env_max = 0.245, 0.105, 2.1
    x_env = (np.arange(p_pass.shape[2]) + 0.5) * 2.0 * dx
    env_from = 8                           # hide the injection-line step
    vmaxes = [float(np.nanquantile(np.abs(p[p.shape[0] // 2:]), 0.999))
              for p in (p_pass, p_stop)]

    fig = _anim_figure()
    fig.suptitle(T("Expansion-chamber silencer: pass band vs stop band "
                   "(2D FDTD)"), fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T(f"Pass band: {f_pass:.0f} Hz, kL = π"),
              T(f"Stop band peak: {f_peak:.0f} Hz, kL = π/2")]
    verdicts = [T("the chamber is acoustically invisible"),
                T("the mismatch reflects the wave back up the pipe")]
    ims: list[Any] = []
    tl_txts: list[Any] = []
    v_txts: list[Any] = []
    for (ax, title, vmax), env in zip(
            zip(axes, titles, vmaxes, strict=True), (e_pass, e_stop),
            strict=True):
        ax.grid(False)
        im = ax.imshow(np.zeros((2, 2)), origin="lower",
                       extent=(0.0, length, 0.0, height), cmap=CMAP_FIELD,
                       vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="bilinear", zorder=2)
        _anim_chamber_hardware(ax, length, height, pipe_y,
                               (0.55, 0.55 + _CHAMBER_L))
        ax.axhline(env_base, color=COLOR_GRID, lw=0.8, zorder=1)
        ax.text(0.005, env_base - 0.008, "|p| envelope", fontsize=7.5,
                ha="left", va="top", color=COLOR_FG, alpha=0.8)
        # The settled envelope, static: the clip captures steady state.
        ax.plot(x_env[env_from:],
                env_base + env[env_from:] / env_max * env_h,
                color=COLOR_PRIMARY, lw=1.8, zorder=6)
        ax.set_xlim(-0.115, length + 0.065)
        # The verdict/TL line sits above the tallest envelope hump
        # (1.85 of 2.1 full scale), so neither text strikes the curve.
        ax.set_ylim(-0.030, env_base + env_h + 0.062)
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)
        tl_txts.append(
            ax.text(length + 0.055, env_base + env_h + 0.050, "",
                    ha="right", va="top", fontsize=9, color="white",
                    zorder=7,
                    bbox={"boxstyle": _ANIM_PILL_BOX,
                          "facecolor": "black", "alpha": 0.55,
                          "edgecolor": "none"}))
        v_txts.append(
            ax.text(0.02, env_base + env_h + 0.046, "", ha="left",
                    va="top", fontsize=8, color=COLOR_FG, zorder=6))
        ims.append(im)
    axes[1].set_xlabel(T("Position along the duct [m]"), fontsize=9)
    t_txt = fig.text(0.988, 0.93, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    # The captured field is already steady, so the verdict can come early.
    reveal = int(0.30 * len(times))

    def update(k: int) -> tuple[Any, ...]:
        for i, (p_all, tl, f) in enumerate(
                ((p_pass, tls[0], f_pass), (p_stop, tls[1], f_peak))):
            ims[i].set_data(p_all[k])
            tl_txts[i].set_text(
                T(f"TL = {tl:.1f} dB at {f:.0f} Hz")
                if k >= reveal else "")
            v_txts[i].set_text(verdicts[i] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1e3:5.1f} ms"))
        return (*ims, *tl_txts, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_expansion_chamber",
                 frames=len(times), gif_fps=8)


def _anim_chamber_hardware(ax: Any, length: float, height: float,
                           pipe_y: tuple[float, float],
                           chamber_x: tuple[float, float]) -> None:
    """Draw the silencer as hardware: inlet/outlet pipe walls, the chamber
    shell with its end plates, the drive loudspeaker and the anechoic
    termination, all to scale (metres)."""
    from matplotlib.patches import Rectangle

    wall = 0.012
    grey = "#9a9a9a"
    x0, x1 = chamber_x
    for y_pipe in pipe_y:
        y0 = y_pipe - (wall if y_pipe == pipe_y[0] else 0.0)
        for xa, xb in ((0.0, x0), (x1, length)):
            ax.add_patch(Rectangle((xa, y0), xb - xa, wall,
                                   facecolor=grey, edgecolor=COLOR_FG,
                                   linewidth=0.7, zorder=3))
    for y0 in (-wall, height):             # chamber shell
        ax.add_patch(Rectangle((x0 - wall, y0), x1 - x0 + 2 * wall, wall,
                               facecolor=grey, edgecolor=COLOR_FG,
                               linewidth=0.7, zorder=3))
    for xe in (x0 - wall, x1):             # end plates (annular in 3D)
        for ya, yb in ((0.0, pipe_y[0]), (pipe_y[1], height)):
            ax.add_patch(Rectangle((xe, ya), wall, yb - ya,
                                   facecolor=grey, edgecolor=COLOR_FG,
                                   linewidth=0.7, zorder=3))
    # Loudspeaker into the inlet.
    bore = pipe_y[1] - pipe_y[0]
    _anim_speaker(ax, 0.0, 0.5 * (pipe_y[0] + pipe_y[1]), bore,
                  tip_inset=0.003, label_y=pipe_y[0] - 0.02)
    ax.add_patch(Rectangle((length, pipe_y[0] - wall), 0.045,
                           bore + 2 * wall, facecolor=grey, hatch="////",
                           edgecolor=COLOR_FG, linewidth=0.8, zorder=3))
    ax.text(length + 0.045, pipe_y[0] - 0.02, "anechoic termination",
            ha="right", va="top", fontsize=7.5)
    ax.text(0.5 * (x0 + x1), height + wall + 0.006,
            f"L = {_CHAMBER_L:.2f} m · m = {_CHAMBER_M:.0f}", ha="center",
            va="bottom", fontsize=7.5, color=COLOR_FG)


_APERTURE_F = 686.0                      # lambda = 0.50 m exactly
_APERTURE_WIDTHS = (0.025, 0.50)         # sub-lambda slit / lambda-sized gap
_APERTURE_DEPTH = 0.10                   # wall thickness across the opening
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the carrier)
# = min(25 mm / 4 = 6.25 mm, 0.5 m / 8 = 62.5 mm) -> the narrow slit governs
# (lambda/80, 4 cells across it).
_APERTURE_DX = 0.00625
_APERTURE_EVERY = 3
_APERTURE_FRAMES = 962
#: 8/3 of the shared rate, matching the stride cut (see the note below).
_APERTURE_FPS = _ANIM_FPS * 8.0 / 3.0


@lru_cache(maxsize=1)
def _aperture_fields(
    n_frames: int = _APERTURE_FRAMES,
) -> tuple[Any, Any, Any, Any]:
    """Two CW runs against the slotted wall, cached.

    A 6 m x 5 m free field (sponges on the far side and above/below, kept
    out of the final framing) with a rigid 0.10 m wall at x = 2 m holding
    a centred opening: 25 mm (lambda/20) in run one, 0.50 m (= lambda) in
    run two. A sustained 686 Hz plane wave enters through the rho c left
    edge; the wall reflection travels back out through the same matched
    edge. ``dx = 6.25 mm`` keeps four cells across the narrow slit.
    Returns the frame stacks, the running-RMS maps in dB on a SHARED
    scale (both referenced to the strongest final RMS, so the two
    transmitted fields are directly comparable), the frame times and the
    library transmission of the narrow slit at the drive frequency.
    """
    import fdtd2d

    from phonometry import slit_transmission_coefficient

    dx = _APERTURE_DX
    ny, nx = 800, 960                      # 5 m x 6 m
    wall_c = (round(2.0 / dx), round((2.0 + _APERTURE_DEPTH) / dx))
    # Clip duration per the deepest-reflector rule: d(source -> far wall
    # face through the opening) = 2.1 m, plus d(slit exit -> farthest
    # visible frame corner (5.72, 0.7)) = 4.04 m: t = 1.2 * 6.14 / 343 =
    # 21.5 ms captured from cold, so the clip is the transit story itself.
    # Sampling: 3 solver steps per frame = 23.19 us gives 62.9 frames per
    # 686 Hz period (>= 48; the old 8-step stride gave 23.6 and a 4-step
    # one would stop at 47.1). 962 frames cover the 22.31 ms window,
    # played at 8/3 of the shared 20 fps -- exactly the factor by which the
    # stride shrank, so both the 18.0 s length and the 1.237 ms of
    # simulation per second of playback are the ones the committed clip
    # already had.
    every = _APERTURE_EVERY
    p_all, r_all = [], []
    times = np.zeros(0)
    for width in _APERTURE_WIDTHS:
        gap = round(width / dx)
        mask = np.zeros((ny, nx), dtype=bool)
        mask[:, wall_c[0]:wall_c[1]] = True
        mask[(ny - gap) // 2:(ny + gap) // 2, wall_c[0]:wall_c[1]] = False
        sim = fdtd2d.FDTD2D(343.0, dx, shape=(ny, nx), sponge_width=40,
                            sponge_sides=("top", "bottom", "right"),
                            obstacle_mask=mask,
                            edge_impedance={"left": 1.2 * 343.0})
        tone = fdtd2d.CWSource(0, 0, frequency=_APERTURE_F)
        sim.add_source(fdtd2d.PlaneWaveSource("right", tone.value,
                                              offset=2))
        ps, rs, times, _ = _fdtd_cw_capture(sim, _APERTURE_F, every,
                                            n_frames)
        p_all.append(ps)
        r_all.append(rs)
    rms = np.stack(r_all)
    ref = float(max(r[-1].max() for r in r_all))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms / ref)
    db_all = np.clip(db, -40.0, 0.0).astype(np.float32)
    res = slit_transmission_coefficient(
        np.array([_APERTURE_F]), _APERTURE_WIDTHS[0], _APERTURE_DEPTH,
        field="normal")
    tau = float(res.transmission_coefficient[0])
    return np.stack(p_all), db_all, times, tau


def animate_fdtd_aperture_slit(output_dir: str) -> None:
    """A plane wavefront against a rigid wall with an opening (2D FDTD),
    sub-wavelength versus wavelength-sized side by side: the 25 mm slit
    re-radiates the little it lets through as a cylindrical wave into a
    nearly uniform half space, while the 0.50 m gap passes the front
    almost intact and casts sharp-edged shadows. The Gomperts model of
    the library gives the narrow slit's transmission coefficient. The
    shadow side of each instantaneous panel rides its own annotated
    display gain so the slit's re-radiation is visible next to the
    standing wave facing the wall; the RMS row keeps one shared scale."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, db_all, times, tau = _aperture_fields()
    lam = 343.0 / _APERTURE_F
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[0][half:]), 0.999))
    wall_x = (2.0, 2.0 + _APERTURE_DEPTH)
    # What the lambda/20 slit passes is ~30 dB under the standing wave that
    # faces the wall and sets the colour scale, so on the incident ramp the
    # cylindrical re-radiation this panel exists to show is a black field.
    # The shadow side of each panel therefore carries its own display gain
    # (see :func:`_weak_field_gain`), stated on the panel; the gain is per
    # panel because the wavelength-sized opening needs none, and the RMS row
    # below keeps the single shared scale that compares the two.
    past = round((wall_x[1] / 6.0) * p_all.shape[3])
    gains = [_weak_field_gain(p_all[col][half::4, :, past:], vmax)
             for col in range(2)]

    fig = _anim_figure()
    fig.suptitle(T("Sound through a wall aperture (2D FDTD)"),
                 fontweight="bold")
    gs = fig.add_gridspec(2, 2)
    titles = [T(f"Slit w = {_APERTURE_WIDTHS[0] * 1e3:.0f} mm (λ/20)"),
              T(f"Opening w = {_APERTURE_WIDTHS[1]:.2f} m (= λ)")]
    verdicts = [T("cylindrical re-radiation from the slit"),
                T("the front passes: sharp-edged shadow")]
    ims: list[Any] = []
    tau_txt: Any = None
    for col in range(2):
        gap = _APERTURE_WIDTHS[col]
        ax_p = fig.add_subplot(gs[0, col])
        ax_r = fig.add_subplot(gs[1, col])
        im_p = ax_p.imshow(p_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 5.0), cmap=CMAP_FIELD,
                           vmin=-vmax, vmax=vmax, interpolation="bilinear")
        im_r = ax_r.imshow(db_all[col][0], origin="lower",
                           extent=(0.0, 6.0, 0.0, 5.0), cmap="magma",
                           vmin=-40.0, vmax=0.0, interpolation="bilinear")
        ax_p.set_title(titles[col], fontsize=10, fontweight="bold")
        for ax in (ax_p, ax_r):
            ax.grid(False)
            for y0, y1 in ((0.0, 2.5 - gap / 2.0), (2.5 + gap / 2.0, 5.0)):
                ax.add_patch(Rectangle((wall_x[0], y0),
                                       _APERTURE_DEPTH, y1 - y0,
                                       facecolor="#707070",
                                       edgecolor="white", lw=0.5))
            # Crop the sponge layers (and the injection seam) out of view,
            # so the frame edge is physical field.
            ax.set_xlim(0.08, 5.72)
            ax.set_ylim(0.7, 4.3)
            ax.tick_params(labelsize=7)
        ax_p.tick_params(labelbottom=False)
        ax_r.set_xlabel("x [m]", fontsize=8)
        ax_p.annotate("", xy=(1.15, 3.75), xytext=(0.55, 3.75),
                      arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                  "lw": 1.2})
        ax_p.text(0.85, 3.85, T("incident plane wavefront"), ha="left",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline)
        ax_p.text(2.24, 1.45, T("rigid wall"), ha="left", va="center",
                  color=FIELD_INK, fontsize=7, path_effects=outline)
        if (note := _gain_note("past the wall", gains[col])):
            ax_p.text(5.62, 4.28, T(note), ha="right", va="top",
                      color=FIELD_INK, fontsize=6.5, path_effects=outline)
        ax_r.text(3.85, 0.85, verdicts[col], ha="center", va="bottom",
                  color="white", fontsize=7.5, zorder=6,
                  bbox={"boxstyle": _ANIM_PILL_BOX,
                        "facecolor": "black", "alpha": 0.45,
                        "edgecolor": "none"})
        if col == 0:
            ax_p.set_ylabel(T("instantaneous p(x, y)"), fontsize=9)
            ax_r.set_ylabel(T("RMS level [dB]"), fontsize=9)
            ax_p.text(0.2, 0.85,
                      T(f"f = {_APERTURE_F:.0f} Hz (λ = {lam:.2f} m)"),
                      ha="left", va="bottom", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline)
            tau_txt = ax_r.text(5.55, 4.1, "", ha="right", va="top",
                                fontsize=8.5, color="white", zorder=7,
                                bbox={"boxstyle": _ANIM_PILL_BOX,
                                      "facecolor": "black", "alpha": 0.55,
                                      "edgecolor": "none"})
        else:
            ax_p.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
            ax_r.text(5.55, 4.1, T("same color scale"), color="white",
                      fontsize=7, ha="right", va="top")
        ims += [im_p, im_r]
    t_txt = fig.text(0.012, 0.985, "", ha="left", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.5 * p_all.shape[1])

    def shadow_gained(col: int, k: int) -> Any:
        """Frame *k* of panel *col* with the shadow side amplified.

        The gain is applied frame by frame: the field stack is memoised and
        shared by the four language/theme variants, so it must not be
        rescaled in place.
        """
        frame = p_all[col][k]
        if gains[col] <= 1.0:
            return frame
        out = frame.copy()
        out[:, past:] *= gains[col]
        return out

    def update(k: int) -> tuple[Any, ...]:
        for col in range(2):
            ims[2 * col].set_data(shadow_gained(col, k))
            ims[2 * col + 1].set_data(db_all[col][k])
        tau_txt.set_text(
            T(f"slit τ = {tau:.2f} (Gomperts)") if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k] * 1e3:4.1f} ms"))
        return (*ims, tau_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_fdtd_aperture_slit",
                 frames=int(p_all.shape[1]), fps=_APERTURE_FPS, gif_fps=8)


_REFR_B = 1.0                            # log-profile strength [m/s]
_REFR_C0 = 340.0                         # ground-level effective speed
_REFR_SRC = (30.0, 2.0)                  # source range/height [m]
_REFR_RECV = 350.0                       # receiver distance from the source
_REFR_F = 50.0                           # CW drive: lambda ~ 6.9 m
# Mesh rule: dx = min(smallest scene dimension / 4, lambda/8 at the
# carrier) = min(2 m source height / 4 = 0.5 m, 6.86 m / 8 = 0.857 m);
# the grid runs finer still (0.3 m, lambda/23) so the long-range phase
# stays clean over the ~60-wavelength domain.
_REFR_DX = 0.3
_REFR_EVERY = 1
_REFR_FRAMES = 1440
_REFR_FPS = 80


def _refraction_profiles() -> tuple[Any, ...]:
    """The guide's realistic logarithmic surface-layer profiles: downwind
    (+1 m/s) and upwind (-1 m/s) effective sound speed."""
    from phonometry import log_linear_sound_speed_profile

    return tuple(
        log_linear_sound_speed_profile(sign * _REFR_B,
                                       ground_speed=_REFR_C0,
                                       max_height=140.0)
        for sign in (1.0, -1.0)
    )


@lru_cache(maxsize=1)
def _refraction_fields(
    n_frames: int = _REFR_FRAMES,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Two CW runs through the refracting atmosphere, cached.

    A 452 m x 132 m slice over rigid ground with the guide's logarithmic
    effective-speed profiles (downwind +1 m/s, upwind -1 m/s) written into
    the sound-speed map row by row; sponges on the sides and the sky. A
    50 Hz CW source (lambda ~ 6.9 m, short against the ~110 m ray-model
    shadow distance, so the shadow actually forms) sits 2 m over the
    ground, the guide's source height. A low-frequency PULSE cannot show
    this: within the 30-periods-per-clip frame budget its wavelength
    reaches ~20 m and diffraction refills the shadow (a measured contrast
    of barely 2 dB). Instead both runs step uncaptured to steady state
    (~1.3 s, the transit of the domain) and the clip then shows the
    settled wave field streaming through the refracting atmosphere at 13.6
    frames per period. Returns the frame stacks, the steady running-RMS
    maps in dB compensated for cylindrical spreading (the verdict
    overlay, on a shared scale), the frame times, the height grid and the
    two c(z) profiles sampled on it, and the library ray fans traced
    through the same profiles.
    """
    import fdtd2d

    from phonometry import atmospheric_ray_paths

    dx = _REFR_DX
    ny, nx = 440, 1507                     # 132 m x 452 m
    z = (np.arange(ny) + 0.5) * dx
    profiles = _refraction_profiles()
    # Clip duration per the deepest-reflector rule: d(source -> ground)
    # = 2 m plus d(ground -> farthest visible frame corner (430, 105))
    # = 413.6 m at the slowest c on the path (upwind, ~333 m/s) gives
    # t = 1.2 * 415.6 / 333 = 1.50 s of flight, which the 1.30 s of
    # uncaptured settling plus this 0.53 s window covers with margin: the
    # wave has crossed the whole domain before the first captured frame
    # and the clip then shows the settled streaming field.
    # Sampling: every solver step is captured, 366.6 us apart downwind and
    # 375.4 us apart upwind (each run's own c_max sets its dt), i.e. 54.6
    # and 53.3 frames per 50 Hz period (>= 48; the old 4-step stride gave
    # 13.6). 1 440 frames, played at 80 fps -- four times the shared
    # 20 fps, exactly the factor by which the stride shrank, so both the
    # 18.0 s length and the 29.3 ms of simulation per second of playback
    # are the ones the committed clip already had.
    every = _REFR_EVERY
    rays = [
        atmospheric_ray_paths(prof, source_height=_REFR_SRC[1],
                              launch_angles_deg=angles, max_range=430.0,
                              n_steps=900)
        for prof, angles in zip(
            profiles,
            # Downwind: a shallow fan (the log profile turns anything
            # under ~8 deg back down, so the duct is thin) plus one
            # escaping ray; upwind: the same fan bends up and away.
            ((-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
             (-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0)), strict=True)
    ]
    p_all, r_all = [], []
    times = np.zeros(0)
    for prof in profiles:
        c_prof = prof.speed_at(z)
        c_map = np.repeat(c_prof[:, np.newaxis], nx, axis=1)
        # Row 0 is the ground (rigid, no sponge); "bottom" is the sky in
        # the low-row-origin naming of fdtd2d, as in the barrier clip.
        sim = fdtd2d.FDTD2D(c_map, dx, sponge_width=40,
                            sponge_sides=("left", "right", "bottom"))
        ix, iy = round(_REFR_SRC[0] / dx), round(_REFR_SRC[1] / dx)
        sim.add_source(fdtd2d.CWSource(ix=ix, iy=iy, frequency=_REFR_F,
                                       ramp_cycles=2.0))
        beta = float(np.exp(-sim.dt * _REFR_F / 2.0))
        ms = np.zeros_like(sim.p)
        settle = round(1.30 / sim.dt)      # domain transit + ring-up
        for _ in range(settle):
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
        ps: list[Any] = []
        ts: list[float] = []
        while len(ps) < n_frames:
            sim.step()
            ms = beta * ms + (1.0 - beta) * sim.p**2
            if sim.n % every == 0:
                ps.append(sim.p[::3, ::3].astype(np.float32))
                ts.append(sim.time)
        p_all.append(np.stack(ps))
        r_all.append(np.sqrt(ms))
        times = np.asarray(ts)
    rms_maps = np.stack(r_all)
    # Verdict overlay: the steady RMS map times sqrt(r) to undo the 2D
    # cylindrical spreading; what remains is pure refraction (the bright
    # downwind ground duct and interference lobes, the dark upwind shadow
    # wedge) on a shared dB scale referenced away from the source blast.
    xs, zs = _REFR_SRC
    xg, zg = np.meshgrid((np.arange(nx) + 0.5) * dx, z)
    r_map = np.maximum(np.hypot(xg - xs, zg - zs), 5.0)
    comp = rms_maps * np.sqrt(r_map)
    ref = float(np.quantile(comp[:, :, nx // 4:], 0.999))
    with np.errstate(divide="ignore"):
        e_db = 20.0 * np.log10(comp[:, ::3, ::3] / ref)
    e_db = np.clip(e_db, -30.0, 0.0).astype(np.float32)
    c_profs = np.stack([prof.speed_at(z) for prof in profiles])
    return np.stack(p_all), e_db, times, z, c_profs, rays


def animate_fdtd_refraction(output_dir: str) -> None:
    """Atmospheric refraction (2D FDTD): the same steady 50 Hz source 2 m
    over rigid ground, downwind and upwind. With the effective sound
    speed growing with height the wavefronts bend back down and stream
    along the ground, keeping the 350 m receiver loud; with it falling
    they lift off the surface and an acoustic shadow opens at the ground.
    The library's ray fans, traced through the same c(z) profiles,
    overlay the fields; the closing seconds crossfade to the
    spreading-compensated RMS map."""
    from matplotlib import patheffects

    from phonometry import shadow_zone_distance

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_all, e_db, times, z, c_profs, rays = _refraction_fields()
    profiles = _refraction_profiles()
    half = p_all.shape[1] // 2
    vmax = float(np.quantile(np.abs(p_all[:, half // 2:half]), 0.999))
    # Shadow-boundary estimate exactly as the guide does it: the log
    # profile's linear-equivalent gradient over the first 10 m, then the
    # closed-form grazing-ray distance.
    grad = float(profiles[1].speed_at(10.0) - profiles[1].speed_at(0.0))
    x_shadow = shadow_zone_distance(grad / 10.0, _REFR_SRC[1],
                                    _REFR_SRC[1], ground_speed=_REFR_C0)
    extent = (0.0, 1507 * _REFR_DX, 0.0, 440 * _REFR_DX)
    src_x, src_h = _REFR_SRC
    recv_x = src_x + _REFR_RECV

    fig = _anim_figure()
    fig.suptitle(T("Atmospheric refraction: downwind duct, upwind shadow "
                   "(2D FDTD)"), fontweight="bold")
    gs = fig.add_gridspec(2, 2, width_ratios=[0.20, 1.0])
    titles = [T("Downwind: sound speed grows with height"),
              T("Upwind: sound speed falls with height")]
    verdicts = [T("bent down: a duct hugs the ground, the receiver "
                  "stays loud"),
                T("bent up: a shadow opens, the receiver goes quiet")]
    ims: list[Any] = []
    ims_e: list[Any] = []
    v_txts: list[Any] = []
    ray_lines: list[Any] = []
    for row in range(2):
        ax_c = fig.add_subplot(gs[row, 0])
        _grid_axes(ax_c)
        ax_c.plot(c_profs[row], z, color=COLOR_PRIMARY, lw=1.6)
        ax_c.plot([float(np.interp(src_h, z, c_profs[row]))], [src_h],
                  marker="o", ms=5, color=COLOR_TERTIARY,
                  markeredgecolor="white", markeredgewidth=0.8)
        ax_c.set_ylim(0.0, 105.0)
        ax_c.set_xlim(332.0, 348.0)
        ax_c.set_xticks([335.0, 345.0])
        ax_c.set_ylabel(T("Height [m]"), fontsize=8)
        ax_c.tick_params(labelsize=6)
        if row == 1:
            ax_c.set_xlabel(T("c_eff(z) [m/s]"), fontsize=7)

        ax_f = fig.add_subplot(gs[row, 1])
        ax_f.grid(False)
        im = ax_f.imshow(p_all[row][0], origin="lower", extent=extent,
                         cmap=CMAP_FIELD, vmin=-vmax, vmax=vmax,
                         aspect="auto", interpolation="bilinear")
        im_e = ax_f.imshow(e_db[row], origin="lower", extent=extent,
                           cmap="magma", vmin=-30.0, vmax=0.0,
                           aspect="auto", interpolation="bilinear",
                           alpha=0.0, zorder=2.5)
        ax_f.set_title(titles[row], fontsize=10, fontweight="bold")
        ax_f.set_xlim(14.0, 430.0)
        ax_f.set_ylim(-7.0, 105.0)
        ax_f.fill_between([14.0, 430.0], -7.0, 0.0, facecolor=COLOR_GRID,
                          edgecolor=COLOR_FG, lw=0.8, hatch="///")
        # Library ray fan through the same profile, revealed once the
        # wavefront has drawn the geometry it explains.
        lines_row = []
        result = rays[row]
        for i in range(result.heights.shape[0]):
            (ln,) = ax_f.plot(src_x + result.ranges[i],
                              result.heights[i], color="white",
                              lw=0.8, alpha=0.0, zorder=3.4,
                              path_effects=[patheffects.withStroke(
                                  linewidth=1.5, foreground="#40404060")])
            lines_row.append(ln)
        ray_lines.append(lines_row)
        ax_f.plot([src_x], [src_h], marker="o", ms=5,
                  color=COLOR_TERTIARY, markeredgecolor=FIELD_STROKE,
                  markeredgewidth=0.8, zorder=4)
        ax_f.text(src_x + 8.0, src_h + 4.0, T("source (h = 2 m)"),
                  ha="left", va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline, zorder=4)
        ax_f.plot([recv_x], [src_h], marker="o", ms=5, color=FIELD_STROKE,
                  markeredgecolor=FIELD_INK, markeredgewidth=0.8, zorder=4)
        ax_f.text(recv_x, src_h + 6.0, T("receiver 350 m"), ha="center",
                  va="bottom", color=FIELD_INK, fontsize=7.5,
                  path_effects=outline, zorder=4)
        if row == 0:
            ax_f.text(20.0, -3.5, T("rigid ground"), ha="left",
                      va="center", color=COLOR_FG, fontsize=6.5,
                      bbox={"boxstyle": "round,pad=0.2",
                            "facecolor": fig.get_facecolor(),
                            "edgecolor": "none"})
            ax_f.text(22.0, 97.0, T(f"f = {_REFR_F:.0f} Hz"), ha="left",
                      va="top", color=FIELD_INK, fontsize=7.5,
                      path_effects=outline, zorder=4)
        else:
            # The ray-model shadow boundary of the library, where the
            # grazing ray leaves the ground for good.
            ax_f.axvline(src_x + x_shadow, color="#888888", ls="--",
                         lw=0.9, alpha=0.8, zorder=3)
            ax_f.text(src_x + x_shadow + 6.0, 84.0,
                      T(f"shadow beyond ≈ {x_shadow:.0f} m (ray model)"),
                      ha="left", va="top", color=FIELD_INK, fontsize=7,
                      path_effects=outline, zorder=4)
        v_txt = ax_f.text(424.0, 97.0, "", ha="right", va="top",
                          color="white", fontsize=8, zorder=4,
                          bbox={"boxstyle": _ANIM_PILL_BOX,
                                "facecolor": "black", "alpha": 0.45,
                                "edgecolor": "none"})
        ax_f.tick_params(labelsize=7, labelleft=False)
        if row == 0:
            ax_f.tick_params(labelbottom=False)
        else:
            ax_f.set_xlabel(T("Range [m]"), fontsize=8)
        ims.append(im)
        ims_e.append(im_e)
        v_txts.append(v_txt)
        if row == 1:
            # Inside the lower field, top-left: the Spanish suptitle and
            # row titles leave no free margin at the figure corners, and
            # this corner stays clear of the shadow-boundary annotation.
            t_txt = ax_f.text(20.0, 97.0, "", ha="left", va="top",
                              family="monospace", fontsize=9,
                              color="white", zorder=4,
                              bbox={"boxstyle": _ANIM_PILL_BOX,
                                    "facecolor": "black", "alpha": 0.45,
                                    "edgecolor": "none"})
    # The field is already steady when the clip starts, so the ray fan can
    # come in early and the RMS verdict follows once the streaming motion
    # has been established.
    rays_on = int(0.18 * p_all.shape[1])
    reveal = int(0.80 * p_all.shape[1])

    def update(k: int) -> tuple[Any, ...]:
        alpha_e = min(1.0, max(0.0, (k - reveal) / 12.0))
        alpha_r = min(0.65, max(0.0, (k - rays_on) / 20.0))
        arts: list[Any] = []
        for row in range(2):
            ims[row].set_data(p_all[row][k])
            ims_e[row].set_alpha(alpha_e)
            for ln in ray_lines[row]:
                ln.set_alpha(alpha_r)
                arts.append(ln)
            v_txts[row].set_text(verdicts[row] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[k]:5.2f} s"))
        return (*ims, *ims_e, *arts, *v_txts, t_txt)

    # gif_fps 4: the steady CW field moves everywhere in every frame, the
    # worst case for GIF palette coding; 8 fps left the fallback near 8 MB
    # where every other FDTD clip stays under 4 MB.
    _render_clip(fig, update, output_dir, "anim_fdtd_refraction",
                 frames=int(p_all.shape[1]), fps=_REFR_FPS, gif_fps=4)


# --- Elastic FDTD clips: plate junction and coincidence --------------------
# Both clips run the library's 2D P-SV solver (phonometry.simulation
# .elastic_fdtd) on the same 10 mm steel plate: body-wave speeds and density
# below give the plane-strain plate modulus E' = 4 mu (lambda + mu) /
# (lambda + 2 mu) that the solver itself propagates, so every derived number
# (bending speed, coincidence frequency) matches the simulated field.
_EL_CP, _EL_CS, _EL_RHO = 5900.0, 3200.0, 7850.0
_EL_H = 0.010
# Mesh rule: dx = min(smallest geometric dimension / 4, shortest relevant
# wavelength / 8). The 10 mm plate thickness governs both clips
# (h / 4 = 2.5 mm), well below lambda_B / 8 (19.5 mm at the 4 kHz junction
# carrier, 25 mm at 2 f_c in the coincidence clip) and lambda_air / 8
# (17.8 mm at 2 f_c), and it puts the required 4 cells across the plate.
_EL_DX = _EL_H / 4.0
#: Solver time step at the default CFL 0.6: dt = cfl dx / (c_p_max sqrt(2)).
_EL_DT = 0.6 * _EL_DX / (_EL_CP * float(np.sqrt(2.0)))
_EL_C0, _EL_RHO0 = 343.0, 1.205


def _elastic_plate_bp_m2() -> tuple[float, float]:
    """(B', m'') of the 10 mm steel plate from the solver's own moduli."""
    mu = _EL_RHO * _EL_CS**2
    lam = _EL_RHO * (_EL_CP**2 - 2.0 * _EL_CS**2)
    e_apparent = 4.0 * mu * (lam + mu) / (lam + 2.0 * mu)
    return e_apparent * _EL_H**3 / 12.0, _EL_RHO * _EL_H


# Junction clip geometry [m]: a semi-infinite horizontal plate (it enters
# through the left sponge and, in the control panel, leaves through the
# right one) carrying a 4 kHz bending packet from x = 0.45 into an
# L-junction at x = 1.05 with a 0.35 m plate hanging down from it.
_EJ_F0 = 4000.0
_EJ_SRC_X, _EJ_JUNC_X = 0.45, 1.05
_EJ_PLATE_Y = 0.36
_EJ_VERT_LEN = 0.35
_EJ_SPONGE = 120                      # 0.30 m absorbing skirt on every side
_EJ_NY, _EJ_NX = 440, 680             # 1.10 m x 1.70 m
_EJ_VIEW = (0.31, 1.39, 0.31, 0.78)   # x0, x1, y0, y1 kept in frame


@lru_cache(maxsize=1)
def _plate_junction_fields() -> tuple[Any, Any, Any, Any, int]:
    """|v| frame stacks of the straight-plate and L-junction runs, cached.

    Two identical elastic runs except for the geometry past x = 1.05: the
    control plate continues into the right sponge (an infinite plate), the
    other stops against a perpendicular plate of the same 10 mm steel. A
    vertical tone-burst force on the surface launches the A0 bending
    packet (Gaussian envelope, sigma = T/2, t0 = 4 sigma). Besides the
    fields, the mid-plane velocity profiles of each plate (vy along the
    horizontal plates, vx down the vertical one) are recorded per frame,
    to be drawn as exaggerated deflection lines over the thin plates.

    Timeline per the animation norms: the frame step is 28 solver steps =
    5.03 us, i.e. 49.7 frames per 4 kHz carrier period (>= the 48 = 4 x
    12-frame visual minimum), and the active span is the source build-up
    t0 plus 1.2 x the full flight tour [source -> junction -> free end ->
    junction -> source] = 2 x (0.60 + 0.35) = 1.90 m at the bending group
    speed c_g = 2 c_B(4 kHz) = 1251 m/s: t = 0.5 ms + 1.2 x 1.52 ms =
    2.32 ms -> 462 active frames.
    """
    from phonometry import ElasticFDTD2D, ForceSource

    dx = _EL_DX
    bp, m2 = _elastic_plate_bp_m2()
    c_b = float((bp / m2) ** 0.25 * np.sqrt(2.0 * np.pi * _EJ_F0))
    c_g = 2.0 * c_b
    sigma_t = 0.5 / _EJ_F0
    t0 = 4.0 * sigma_t
    tour = 2.0 * ((_EJ_JUNC_X - _EJ_SRC_X) + _EJ_VERT_LEN)
    every = 28
    n_active = int(np.ceil((t0 + 1.2 * tour / c_g) / (every * _EL_DT)))

    r0 = round(_EJ_PLATE_Y / dx)                    # plate rows r0..r0+3
    j0 = round(_EJ_JUNC_X / dx)                     # junction columns
    r_end = r0 + round((_EL_H + _EJ_VERT_LEN) / dx)

    def burst(t: float) -> float:
        return float(np.exp(-(((t - t0) / sigma_t) ** 2))
                     * np.sin(2.0 * np.pi * _EJ_F0 * (t - t0)))

    x0, x1, y0, y1 = _EJ_VIEW
    rows = slice(round(y0 / dx), round(y1 / dx), 2)
    cols = slice(round(x0 / dx), round(x1 / dx), 2)
    runs = []
    for with_junction in (False, True):
        c_p = np.full((_EJ_NY, _EJ_NX), _EL_C0)
        c_s = np.zeros((_EJ_NY, _EJ_NX))
        rho = np.full((_EJ_NY, _EJ_NX), _EL_RHO0)
        h_cols = slice(0, j0 + 4) if with_junction else slice(0, _EJ_NX)
        c_p[r0:r0 + 4, h_cols] = _EL_CP
        c_s[r0:r0 + 4, h_cols] = _EL_CS
        rho[r0:r0 + 4, h_cols] = _EL_RHO
        if with_junction:
            c_p[r0:r_end, j0:j0 + 4] = _EL_CP
            c_s[r0:r_end, j0:j0 + 4] = _EL_CS
            rho[r0:r_end, j0:j0 + 4] = _EL_RHO
        sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho,
                            sponge_width=_EJ_SPONGE)
        # Vertical force on the top surface (the vy face between the air
        # row r0 - 1 and the first plate row): Lamb-style A0 launcher.
        sim.add_source(ForceSource(ix=round(_EJ_SRC_X / dx), iy=r0 - 1,
                                   direction="y", amplitude=1e4,
                                   waveform=burst))
        frames: list[Any] = []
        hl: list[Any] = []
        vl: list[Any] = []
        ts: list[float] = []
        for _ in range(n_active):
            for _s in range(every):
                sim.step()
            v = np.hypot(sim.collocated("vx"), sim.collocated("vy"))
            frames.append(v[rows, cols].astype(np.float32))
            # Mid-plane plate velocities: the vy face row and (junction
            # run) the vx face column through the middle of each plate.
            hl.append(sim.vy[r0 + 1, :].astype(np.float32))
            vl.append(sim.vx[:, j0 + 1].astype(np.float32))
            ts.append(sim.time)
        runs.append((np.stack(frames), np.stack(hl), np.stack(vl)))
    lines = (runs[0][1], runs[1][1], runs[1][2])
    return runs[0][0], runs[1][0], lines, np.asarray(ts), n_active


def animate_elastic_plate_junction(output_dir: str) -> None:
    """A 4 kHz bending packet on a 10 mm steel plate (elastic 2D FDTD):
    on the straight control plate it just runs on; at an L-junction with an
    identical perpendicular plate it splits into a reflected and a
    transmitted bending wave plus a fast in-plane precursor, and the
    verdict pill quotes the closed-form junction_transmission coefficient
    the guide computes for this corner."""
    from matplotlib.patches import Rectangle

    from phonometry import junction_transmission

    T = _translate_str
    v_ctrl, v_junc, lines, times, n_active = _plate_junction_fields()
    hl_ctrl, hl_junc, vl_junc = lines
    bp, m2 = _elastic_plate_bp_m2()
    c_pl = float(np.sqrt(12.0 * bp / (m2 * _EL_H**2)))
    res = junction_transmission("L", _EL_H, c_pl, m2, _EL_H, c_pl, m2)
    tau0 = float(res.corner[0])
    kij = float(res.corner_reduction_index)
    x0, x1, y0, y1 = _EJ_VIEW
    vmax = float(np.quantile(v_junc, 0.9995))
    # Exaggerated plate-deflection overlays: the plate is only 10 mm
    # thick on screen, so the signed mid-plane velocity is drawn as a
    # deflected line (a fixed gain shared by all three lines).
    gain = 0.045 / float(np.quantile(np.abs(hl_junc), 0.999))
    dx = _EL_DX
    j0 = round(_EJ_JUNC_X / dx)
    r0 = round(_EJ_PLATE_Y / dx)
    r_end = r0 + round((_EL_H + _EJ_VERT_LEN) / dx)
    c0, c1 = round(x0 / dx), round(x1 / dx)
    x_h = (np.arange(c0, c1) + 0.5) * dx           # vy faces, cropped
    x_hj = (np.arange(c0, j0 + 4) + 0.5) * dx      # up to the junction
    y_v = (np.arange(r0, r_end) + 0.5) * dx        # vx faces, vertical
    y_mid = _EJ_PLATE_Y + 0.5 * _EL_H
    x_mid = _EJ_JUNC_X + 2.0 * dx

    fig = _anim_figure()
    fig.suptitle(T("Bending waves at an L-junction (elastic 2D FDTD)"),
                 fontweight="bold")
    axes = fig.subplots(2, 1, sharex=True)
    titles = [T("Straight plate: the packet just runs on"),
              T("L-junction: reflected, transmitted and mode-converted")]
    verdicts = [T("nothing comes back"),
                T(f"junction_transmission('L'): τ(0°) = {tau0:.2f}, "
                  f"K12 = {kij:.1f} dB")]
    ims: list[Any] = []
    v_txts: list[Any] = []
    ln_arts: list[Any] = []
    for ax, title, data, with_junction in (
            (axes[0], titles[0], v_ctrl, False),
            (axes[1], titles[1], v_junc, True)):
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper", extent=(x0, x1, y1, y0),
                       cmap="magma", vmin=0.0, vmax=vmax,
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, fontsize=10, fontweight="bold")
        # The 10 mm plates, drawn as thin open rectangles so the field
        # inside stays visible.
        px1 = _EJ_JUNC_X + 4 * _EL_DX if with_junction else x1
        ax.add_patch(Rectangle((x0, _EJ_PLATE_Y), px1 - x0, _EL_H,
                               facecolor="none", edgecolor=COLOR_FG,
                               lw=0.7, alpha=0.8))
        if with_junction:
            y_end = _EJ_PLATE_Y + _EL_H + _EJ_VERT_LEN
            ax.add_patch(Rectangle((_EJ_JUNC_X, _EJ_PLATE_Y),
                                   4 * _EL_DX, y_end - _EJ_PLATE_Y,
                                   facecolor="none", edgecolor=COLOR_FG,
                                   lw=0.7, alpha=0.8))
            ax.text(_EJ_JUNC_X - 0.055, y_end - 0.005, T("free end"),
                    ha="right", va="bottom", color="white", fontsize=7)
        ax.plot([_EJ_SRC_X], [_EJ_PLATE_Y - 0.008], marker="v", ms=5,
                color=COLOR_TERTIARY, markeredgecolor="white",
                markeredgewidth=0.6, zorder=4)
        if not with_junction:
            ax.text(x0 + 0.015, y1 - 0.02, T("4 kHz tone burst at ▼"),
                    ha="left", va="bottom", color="white", fontsize=7.5,
                    zorder=5,
                    bbox={"boxstyle": _ANIM_PILL_BOX, "facecolor": "black",
                          "alpha": 0.5, "edgecolor": "none"})
        # The exaggerated deflection lines riding on the plates.
        if with_junction:
            (ln_h,) = ax.plot(x_hj, np.full(x_hj.size, y_mid), color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            (ln_v,) = ax.plot(np.full(y_v.size, x_mid), y_v, color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            ln_arts += [ln_h, ln_v]
        else:
            (ln_h,) = ax.plot(x_h, np.full(x_h.size, y_mid), color="white",
                              lw=1.1, alpha=0.85, zorder=3.5)
            ln_arts.append(ln_h)
        ax.set_ylim(y1, y0)
        ax.tick_params(labelsize=7)
        v_txt = ax.text(x1 - 0.015, y1 - 0.02, "", ha="right", va="bottom",
                        color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.5,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    axes[1].set_xlabel("x [m]", fontsize=8)
    t_txt = fig.text(0.985, 0.965, "", ha="right", va="top",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.62 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        ims[0].set_data(v_ctrl[kf])
        ims[1].set_data(v_junc[kf])
        ln_arts[0].set_ydata(y_mid + gain * hl_ctrl[kf][c0:c1])
        ln_arts[1].set_ydata(y_mid + gain * hl_junc[kf][c0:j0 + 4])
        ln_arts[2].set_xdata(x_mid + gain * vl_junc[kf][r0:r_end])
        for row, v_txt in enumerate(v_txts):
            v_txt.set_text(verdicts[row] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.2f} ms"))
        return (*ims, *ln_arts, *v_txts, t_txt)

    _render_clip(fig, update, output_dir, "anim_elastic_plate_junction",
                 frames=n_active + _ANIM_HOLD, gif_fps=6)


# Coincidence clip geometry [m]: the same 10 mm steel plate lying flat at
# y = 0.65 across the whole domain (its ends die in the side sponges: an
# infinite plate), air above and below, driven from a phase-graded source
# line at y = 0.32 that radiates a sustained plane wave 45 degrees down.
# The two panels differ only in frequency: f_c / 2 and 2 f_c, with
# sin(theta) = sqrt(f_c / f) putting the exact trace match of the second
# panel at 45 degrees.
_EC_PLATE_Y = 0.65
_EC_SRC_Y = 0.32
_EC_THETA = 45.0
_EC_SPONGE = 120
_EC_NY, _EC_NX = 564, 1036            # 1.41 m x 2.59 m
# The view starts at x = 1.06 so that even its bottom-left corner sits
# inside the steady 45-degree beam of the source aperture (the stationary
# source point of a view point (x, y) is at x - (y - y_src) tan(theta),
# which must stay right of the sponge edge at x = 0.30).
_EC_VIEW = (1.06, 2.06, 0.345, 1.02)
# Numerical-dispersion compensation of the simulated plate. The trace-match
# resonance of a steel plate in air is razor sharp (its scale is
# 2 rho0 c0 / (omega m'' cos theta) ~ 1e-3 here), while the 4-cell discrete
# plate propagates A0 a few percent slower than the Kirchhoff continuum
# (thickness shear plus grid dispersion): driven at the continuum 2 f_c the
# discrete plate would sit far off its own coincidence and transmit nothing.
# The plate's body-wave speeds are therefore raised by the factor measured
# with the cross-spectrum phase-speed rig of the flexural-dispersion test
# (same dx, same 4-cell immersed plate: c_A0 = 463.1 m/s at 2 f_c against
# the 485.1 m/s of the continuum Kirchhoff plate; c_B scales as
# sqrt(c_pl), so c_pl needs (485.1 / 463.1)**2 = 1.0971, re-verified with
# the same rig on the compensated material). This puts the discrete
# plate's bending wavelength at the 2 f_c carrier back on the continuum
# value that every displayed library number describes; the below-f_c mass
# law is untouched by it (that only sees m'').
_EC_CB_TUNE = 1.0971
# The transmitted wave stays faint on the incident colour scale even at
# coincidence: air drives steel so weakly that the resonant bending wave
# needs a radiation build-up length of c_g m'' cos(theta) / (2 rho0 c0)
# ~ 65 m, so over the ~2 m lit span the coincidence beam only climbs to
# ~13 dB above the mass law (growing visibly along x). A steady-scene scan
# of the tune factor (+-4 % around the value below, same domain and
# source) moves the under-plate level by < 1.2 dB, confirming the level is
# this aperture-limited build-up and not residual detuning. Both panels
# therefore draw the air below the plate with the display gain
# :func:`_weak_field_gain` measures off the settled field -- one gain for
# the two, since the clip's whole point is that they transmit the same
# level -- and print it.
# Colour half-range of the instantaneous panels, on the normalisation that
# puts the incident wave at unit amplitude: 1.6 leaves the standing wave
# above the plate (incident plus near-total reflection) room to breathe.
_EC_VLIM = 1.6


@lru_cache(maxsize=1)
def _coincidence_fields() -> tuple[Any, Any, Any, list[float], int]:
    """The below-f_c and above-f_c oblique plane-wave runs, cached.

    1D pre-validation (same dx, same 4-cell immersed plate, normal
    incidence): the measured transmission of a narrowband packet follows
    the library mass_law_transmission_loss within 0.07 dB at f_c / 2 and
    0.17 dB at 2 f_c (narrowband spectral amplitudes at the carrier;
    peak-amplitude estimates stay within 0.5 dB, limited by the packet's
    finite bandwidth), so the air-steel contrast is trusted at the -48 dB
    transmission this clip displays.

    Timeline per the animation norms: the frame step is 52 solver steps =
    8.52 us, i.e. 48.6 frames per period of the 2 f_c carrier (>= the 4 x
    12-frame visual minimum; four times that at f_c / 2), and the active
    span is the 0.62 ms source ramp plus 1.2 x the flight from the source
    line to the farthest visible point at the bottom of the view,
    (1.02 - 0.321) m / (c0 cos 45) = 2.88 ms -> 4.08 ms -> 479 active
    frames. Returns the two normalized frame stacks (incident amplitude
    = 1), the frame times, the measured transmitted levels [dB re
    incident] and the active frame count.
    """
    from phonometry import ElasticFDTD2D, coincidence_frequency

    dx = _EL_DX
    bp, m2 = _elastic_plate_bp_m2()
    fc = coincidence_frequency(m2, bp)
    freqs = (0.5 * fc, 2.0 * fc)
    theta = np.radians(_EC_THETA)
    r0 = round(_EC_PLATE_Y / dx)
    src_row = round(_EC_SRC_Y / dx)
    y_src = (src_row + 0.5) * dx
    # The compensated plate raises c_p_max and with it the solver step, so
    # the 48-frames-per-period pacing is derived from the actual dt.
    cp_plate = _EL_CP * _EC_CB_TUNE
    dt = 0.6 * dx / (cp_plate * float(np.sqrt(2.0)))
    every = int((1.0 / freqs[1]) / (48.0 * dt))
    ramp_t = 1.5 / freqs[1]
    flight = (_EC_VIEW[3] - y_src) / (_EL_C0 * float(np.cos(theta)))
    n_active = int(np.ceil((ramp_t + 1.2 * flight) / (every * dt)))
    steps = n_active * every

    src_cols = np.arange(_EC_SPONGE, _EC_NX - _EC_SPONGE)
    x_src = (src_cols + 0.5) * dx
    x0, x1, y0, y1 = _EC_VIEW
    rows = slice(round(y0 / dx), round(y1 / dx), 2)
    cols = slice(round(x0 / dx), round(x1 / dx), 2)
    # Incident-amplitude probe: reached by the direct wave at ~0.74 ms,
    # by the first plate reflection at ~1.98 ms.
    p_row, p_col = round(0.50 / dx), round(1.65 / dx)
    w0, w1 = 1.05e-3, 1.90e-3
    # Transmitted-level box under the plate, averaged over the last
    # quarter of the active frames (the settled state).
    b_rows = slice(round(0.75 / dx), round(0.95 / dx), 2)

    stacks = []
    trans_db: list[float] = []
    for f0 in freqs:
        c_p = np.full((_EC_NY, _EC_NX), _EL_C0)
        c_s = np.zeros((_EC_NY, _EC_NX))
        rho = np.full((_EC_NY, _EC_NX), _EL_RHO0)
        c_p[r0:r0 + 4] = _EL_CP * _EC_CB_TUNE
        c_s[r0:r0 + 4] = _EL_CS * _EC_CB_TUNE
        rho[r0:r0 + 4] = _EL_RHO
        sim = ElasticFDTD2D(c_p, c_s, dx, rho=rho, sponge_width=_EC_SPONGE)
        omega = 2.0 * np.pi * f0
        phase_x = omega * np.sin(theta) * x_src / _EL_C0
        frames: list[Any] = []
        ts: list[float] = []
        inc = 0.0
        for i in range(steps):
            sim.step()
            t = sim.time
            r = min(t / ramp_t, 1.0)
            drive = (r * r * (3.0 - 2.0 * r)) * np.sin(omega * t - phase_x)
            sim.txx[src_row, src_cols] -= drive
            sim.tyy[src_row, src_cols] -= drive
            if w0 <= t <= w1:
                inc = max(inc, abs(float(sim.p[p_row, p_col])))
            if (i + 1) % every == 0:
                frames.append(sim.p[rows, cols].astype(np.float32))
                ts.append(t)
        stack = np.stack(frames) / inc
        tail = stack[3 * n_active // 4:]
        # The stored rows start at y0; index the under-plate box within.
        rb0 = (b_rows.start - rows.start) // 2
        rb1 = (b_rows.stop - rows.start) // 2
        rms = float(np.sqrt(np.mean(tail[:, rb0:rb1, :] ** 2)))
        trans_db.append(20.0 * float(np.log10(rms * np.sqrt(2.0))))
        stacks.append(stack)
    return stacks[0], stacks[1], np.asarray(ts), trans_db, n_active


def animate_elastic_coincidence(output_dir: str) -> None:
    """A sustained oblique plane wave in air on a 10 mm steel plate
    (elastic 2D FDTD, fluid-solid coupling in the same solver): below the
    coincidence frequency the mass law rules and the far side stays dark;
    at 2 f_c and 45 degrees the acoustic trace matches the free bending
    wavelength and the plate re-radiates a 45-degree beam that grows
    along the lit span, holding the transmitted level at the f_c/2 figure
    where the mass law demanded 12 dB more. The two frequencies and the
    angle come from the library's coincidence_frequency; the verdicts
    quote the oblique mass law (Bies Eq. 7.41) each panel is judged
    against. Both panels draw the air below the plate with the same
    annotated display gain, so the transmitted field is legible next to an
    incident wave some 45 dB louder."""
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle

    from phonometry import coincidence_frequency

    T = _translate_str
    outline = [patheffects.withStroke(linewidth=2.0,
                                      foreground=FIELD_STROKE)]
    p_lo, p_hi, times, trans_db, n_active = _coincidence_fields()
    bp, m2 = _elastic_plate_bp_m2()
    fc = coincidence_frequency(m2, bp)
    x0, x1, y0, y1 = _EC_VIEW
    # The air below the plate rides an annotated display gain measured off
    # the settled field of both runs at once (see the _EC_VLIM note); the
    # field stacks are cached and shared by the four variants, so the gain
    # goes on copies.
    dx = _EL_DX
    i_below = (round((_EC_PLATE_Y + _EL_H) / dx) - round(y0 / dx)) // 2
    settled = slice(3 * n_active // 4, n_active)
    gain = _weak_field_gain(
        np.concatenate([p_lo[settled, i_below:, :].ravel(),
                        p_hi[settled, i_below:, :].ravel()]), _EC_VLIM)
    p_lo = p_lo.copy()
    p_hi = p_hi.copy()
    for stack in (p_lo, p_hi):
        stack[:, i_below:, :] *= gain

    fig = _anim_figure()
    fig.suptitle(T("Coincidence: the same steel plate, below and above "
                   "f_c (elastic 2D FDTD)"), fontweight="bold")
    axes = fig.subplots(1, 2, sharey=True)
    titles = [T(f"f = f_c/2 = {fc / 2:.0f} Hz, 45° incidence"),
              T(f"f = 2 f_c = {2 * fc:.0f} Hz, 45° incidence")]
    # Each panel is judged against the oblique mass law (Bies Eq. 7.41):
    # TL(theta) = 10 lg[1 + (pi f m'' cos(theta) / rho0 c0)^2]. Below f_c
    # the measured level lands on it; above f_c the trace-matched plate
    # re-radiates and beats it by ~13 dB, so the two panels transmit the
    # same level even though the mass law demands 12 dB more at 4x the
    # frequency.
    theta = np.radians(_EC_THETA)
    ml = [10.0 * float(np.log10(1.0 + (np.pi * f * m2 * np.cos(theta)
                                       / (_EL_RHO0 * _EL_C0)) ** 2))
          for f in (0.5 * fc, 2.0 * fc)]
    verdicts = [T(f"below f_c: the mass law holds: {trans_db[0]:.0f} dB "
                  f"(it predicts {-ml[0]:.0f})"),
                T(f"above f_c: trace matches λ_B: {trans_db[1]:.0f} dB, "
                  f"the mass law said {-ml[1]:.0f}")]
    ims: list[Any] = []
    v_txts: list[Any] = []
    for col, (ax, title, data) in enumerate(
            ((axes[0], titles[0], p_lo), (axes[1], titles[1], p_hi))):
        ax.grid(False)
        im = ax.imshow(data[0], origin="upper", extent=(x0, x1, y1, y0),
                       cmap=CMAP_FIELD, vmin=-_EC_VLIM, vmax=_EC_VLIM,
                       aspect="equal", interpolation="bilinear")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.add_patch(Rectangle((x0, _EC_PLATE_Y), x1 - x0, _EL_H,
                               facecolor=COLOR_GRID, edgecolor=COLOR_FG,
                               lw=0.8, zorder=3))
        if col == 0:
            ax.text(x0 + 0.02, _EC_PLATE_Y - 0.012,
                    T("10 mm steel plate"), ha="left", va="bottom",
                    color=FIELD_INK, fontsize=7.5, path_effects=outline,
                    zorder=4)
            ax.set_ylabel("y [m]", fontsize=8)
        ax.text(x0 + 0.02, _EC_PLATE_Y + _EL_H + 0.012,
                T(_gain_note("air below the plate", gain)), ha="left",
                va="top", color=FIELD_INK, fontsize=6.5,
                path_effects=outline, zorder=4)
        # Incidence arrow at 45 degrees.
        ax.annotate("", xy=(x0 + 0.34, y0 + 0.30),
                    xytext=(x0 + 0.13, y0 + 0.09),
                    arrowprops={"arrowstyle": "-|>", "color": FIELD_INK,
                                "lw": 1.4}, zorder=4)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        v_txt = ax.text(x1 - 0.02, y1 - 0.02, "", ha="right", va="bottom",
                        color="white", fontsize=8, zorder=5,
                        bbox={"boxstyle": _ANIM_PILL_BOX,
                              "facecolor": "black", "alpha": 0.55,
                              "edgecolor": "none"})
        ims.append(im)
        v_txts.append(v_txt)
    fc_txt = fig.text(0.5, 0.035,
                      T(f"coincidence_frequency: f_c = {fc:.0f} Hz "
                        f"(10 mm steel)"), ha="center", va="bottom",
                      fontsize=9, color=COLOR_FG)
    t_txt = fig.text(0.985, 0.035, "", ha="right", va="bottom",
                     family="monospace", fontsize=10, color=COLOR_FG)
    reveal = int(0.75 * n_active)

    def update(k: int) -> tuple[Any, ...]:
        kf = min(k, n_active - 1)
        ims[0].set_data(p_lo[kf])
        ims[1].set_data(p_hi[kf])
        for col, v_txt in enumerate(v_txts):
            v_txt.set_text(verdicts[col] if k >= reveal else "")
        t_txt.set_text(T(f"t = {times[kf] * 1e3:5.2f} ms"))
        return (*ims, *v_txts, fc_txt, t_txt)

    _render_clip(fig, update, output_dir, "anim_elastic_coincidence",
                 frames=n_active + _ANIM_HOLD, gif_fps=5)


_ANIMATIONS: dict[str, Callable[[str], None]] = {
    "anim_time_weighting": animate_time_weighting_ballistics,
    "anim_onset_detection": animate_onset_detection,
    "anim_instantaneous_intensity": animate_instantaneous_intensity,
    "anim_schroeder": animate_schroeder,
    "anim_fdtd_room_modes": animate_fdtd_room_modes,
    "anim_fdtd_barrier": animate_fdtd_barrier,
    "anim_fdtd_ground_effect": animate_fdtd_ground_effect,
    "anim_fdtd_ducting": animate_fdtd_ducting,
    "anim_fdtd_diffusion": animate_fdtd_diffusion,
    "anim_fdtd_metadiffuser": animate_fdtd_metadiffuser,
    "anim_fdtd_pillar_hall": animate_fdtd_pillar_hall,
    "anim_fdtd_impedance_tube": animate_fdtd_impedance_tube,
    "anim_fdtd_transmission_tube": animate_fdtd_transmission_tube,
    "anim_standing_wave_tube": animate_standing_wave_tube,
    "anim_flanking_paths": animate_flanking_paths,
    "anim_intensity_scan_power": animate_intensity_scan_power,
    "anim_sweep_deconvolution": animate_sweep_deconvolution,
    "anim_specific_loudness": animate_specific_loudness,
    "anim_power_two_rooms": animate_power_two_rooms,
    "anim_comb_filtering": animate_comb_filtering,
    "anim_fdtd_slit_absorber": animate_fdtd_slit_absorber,
    "anim_fdtd_expansion_chamber": animate_fdtd_expansion_chamber,
    "anim_fdtd_aperture_slit": animate_fdtd_aperture_slit,
    "anim_fdtd_refraction": animate_fdtd_refraction,
    "anim_elastic_plate_junction": animate_elastic_plate_junction,
    "anim_elastic_coincidence": animate_elastic_coincidence,
}


#: Cached field builders of the clips whose simulation dominates their cost,
#: keyed by clip name. Calling one fills its ``lru_cache`` in the current
#: process, which is what lets :func:`_render_anim_variants` fork the four
#: language/theme variants off a single field computation: the frame stacks
#: are never written after they are built, so the children share them
#: copy-on-write instead of paying for the simulation (or the memory) again.
_ANIM_FIELDS: dict[str, Callable[[], Any]] = {
    "anim_fdtd_room_modes": _room_mode_fields,
    "anim_fdtd_barrier": _barrier_fields,
    "anim_fdtd_ground_effect": _ground_effect_fields,
    "anim_fdtd_ducting": _ducting_fields,
    "anim_fdtd_diffusion": _diffusion_fields,
    "anim_fdtd_metadiffuser": _metadiffuser_fields,
    "anim_fdtd_impedance_tube": _impedance_tube_fields,
    "anim_fdtd_transmission_tube": _transmission_tube_fields,
    "anim_fdtd_slit_absorber": _slit_absorber_fields,
    "anim_fdtd_expansion_chamber": _expansion_chamber_fields,
    "anim_fdtd_aperture_slit": _aperture_fields,
    "anim_fdtd_refraction": _refraction_fields,
    "anim_elastic_plate_junction": _plate_junction_fields,
    "anim_elastic_coincidence": _coincidence_fields,
}

# A clip rename must not silently drop its field builder; fail fast.
if not _ANIM_FIELDS.keys() <= _ANIMATIONS.keys():
    _unknown_field = sorted(_ANIM_FIELDS.keys() - _ANIMATIONS.keys())
    raise RuntimeError(f"animation names not in _ANIMATIONS: {_unknown_field}")


def _render_anim_variant(clip: str, output_dir: str, lang: str,
                         dark: bool) -> None:
    """Render one language/theme variant of *clip* (fork child entry)."""
    set_lang(lang)
    set_theme(dark)
    try:
        _ANIMATIONS[clip](output_dir)
    finally:
        plt.close("all")


def _render_anim_variants(clip: str, output_dir: str) -> None:
    """Render all four language/theme variants of one clip.

    Where the clip has a registered field builder and the platform can
    fork, the field is computed once here and the four variants then render
    concurrently in forked children that share it: the encoding of a
    600-frame field clip is the long pole, and four of them cost about as
    much wall time as one. Everything else (no builder, no fork) falls back
    to rendering the variants one after another in this process, which is
    what the pool workers of a full run do anyway.
    """
    import multiprocessing as mp

    builder = _ANIM_FIELDS.get(clip)
    if builder is None or "fork" not in mp.get_all_start_methods():
        for lang, dark in _VARIANTS:
            _render_anim_variant(clip, output_dir, lang, dark)
        return
    builder()
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_render_anim_variant,
                    args=(clip, output_dir, lang, dark))
        for lang, dark in _VARIANTS
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    failed = [p.exitcode for p in procs if p.exitcode]
    if failed:
        raise RuntimeError(f"{clip}: {len(failed)} variant(s) failed "
                           f"(exit codes {failed})")


def generate_animations(output_dir: str, names: list[str] | None = None,
                        *, variants: bool = False) -> None:
    """Render the Tier-1 animations, by default in the active language/theme.

    ``names`` (clip stems, e.g. ``anim_schroeder``) restricts the run to a
    subset, used by ``--anim`` to re-render a single clip after review
    fixes without paying for the whole batch. With ``variants`` each clip is
    rendered in all four language x theme variants off one field
    computation (:func:`_render_anim_variants`) instead of once in whatever
    language and theme the caller has set.
    """
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found on PATH; it is required to encode the "
            "animation WebM/GIF outputs. Install ffmpeg and retry."
        )
    if names:
        unknown = sorted(set(names) - _ANIMATIONS.keys())
        if unknown:
            available = ", ".join(sorted(_ANIMATIONS))
            raise SystemExit(
                f"unknown animation(s) {unknown}; available: {available}")
        clips = list(names)
    else:
        clips = list(_ANIMATIONS)
    for clip in clips:
        if variants:
            print(f"--- Generating {clip} (4 variants) ---")
            _render_anim_variants(clip, output_dir)
        else:
            _ANIMATIONS[clip](output_dir)


# ====================================================================# Command line / parallel figure generation
# ---------------------------------------------------------------------------
# Every asset is produced four times: light/dark theme x English/Spanish
# ("_dark" / "_es" / "_es_dark" suffixes) so both site languages follow the
# user's mode. The figures are independent (each renders from its own seeded
# data into its own per-variant files), so the (figure, language, theme)
# tasks are distributed over a process pool. Language and theme are applied
# inside the worker before each figure runs, exactly as the sequential loop
# did, so the output bytes are identical either way.
# ====================================================================
_VARIANTS: tuple[tuple[str, bool], ...] = (
    ("en", False),
    ("en", True),
    ("es", False),
    ("es", True),
)

# Figures whose expensive signal analysis is memoised with ``lru_cache``
# across the four language/theme variants (the ECMA-418-2 / ISO 532
# psychoacoustic computations). The caches live per process, so these render
# all four variants as ONE task in a single worker -- four separate tasks
# would recompute the same analysis in four workers (~150 s of duplicated
# CPU). Every other figure is cheap to recompute and parallelises per
# variant.
_GROUPED_FIGURES = frozenset(
    {
        "loudness_models_comparison",
        "sottek_specific_loudness",
        "tonality_roughness_demo",
        "moore_glasberg_time_loudness",
        "fluctuation_strength",
        "metadiffuser_ntff_polar",
    }
)

# Approximate task cost in seconds (measured sequentially on a 12-core dev
# box; for the grouped figures the value is the whole four-variant task).
# Used only to submit the heaviest tasks to the pool first so none of them
# lands at the tail and stretches the run; unlisted figures are treated as
# fast and staleness is harmless.
_FIGURE_WEIGHTS: dict[str, float] = {
    "metadiffuser_ntff_polar": 260.0,
    "loudness_models_comparison": 25.5,
    "tonality_roughness_demo": 19.7,
    "sottek_specific_loudness": 4.1,
    "filter_responses": 2.8,
    "moore_glasberg_time_loudness": 2.6,
    "numerical_propagation": 2.5,
    "fluctuation_strength": 2.2,
    "weighting_responses": 1.1,
    "special_weighting_responses": 2.6,
    "sti_curve": 1.7,
    "schroeder_decay": 1.3,
    "excitation_signals": 1.2,
    "crossover_plot": 1.1,
    "fdtd_room_modes": 2.0,
}

# A registry rename must not silently degroup a cached figure (a 4x
# recompute) or drop its scheduling weight; fail fast on import instead.
_REGISTRY_NAMES = frozenset(
    f.__name__.removeprefix("generate_") for f in _FIGURE_FUNCS
)
if not (_GROUPED_FIGURES | _FIGURE_WEIGHTS.keys()) <= _REGISTRY_NAMES:
    _unknown = sorted((_GROUPED_FIGURES | _FIGURE_WEIGHTS.keys()) - _REGISTRY_NAMES)
    raise RuntimeError(f"figure names not in _FIGURE_FUNCS: {_unknown}")


def _run_figure_task(
    func_name: str, variants: tuple[tuple[str, bool], ...], img_dir: str
) -> str:
    """Render one figure in the given language/theme variants (worker entry).

    The prologue applies each variant's language and theme to this worker's
    module globals, exactly like the sequential loop does before calling the
    generator; the epilogue closes every figure so a task cannot leak pyplot
    state (or figure memory) into the next task that reuses the process.
    """
    func: Callable[[str], None] = getattr(sys.modules[__name__], func_name)
    try:
        for lang, dark in variants:
            set_lang(lang)
            set_theme(dark)
            func(img_dir)
    finally:
        plt.close("all")
    return func_name


def _default_jobs() -> int:
    """Default worker count: the available cores minus two, capped at 8.

    Two cores are left free so the machine (or the CI runner's other duties)
    stays responsive, and the cap bounds peak memory: every worker holds one
    figure's data arrays plus a matplotlib canvas (~a few hundred MB for the
    heaviest compute figures).
    """
    cpus = os.process_cpu_count() or 2
    return max(1, min(cpus - 2, 8))


def _generate_figures_parallel(
    img_dir: str, funcs: list[Callable[[str], None]], jobs: int
) -> None:
    """Render ``funcs`` in all four variants on a ``jobs``-wide process pool.

    Workers are spawned (not forked) so each starts from a pristine
    interpreter: a fresh import of this module pins the numerical thread
    pools and leaves no inherited pyplot/rcParams state, keeping every task
    bit-reproducible regardless of scheduling order.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

    tasks: list[tuple[str, tuple[tuple[str, bool], ...]]] = []
    for func in funcs:
        if func.__name__.removeprefix("generate_") in _GROUPED_FIGURES:
            tasks.append((func.__name__, _VARIANTS))
        else:
            tasks.extend((func.__name__, (variant,)) for variant in _VARIANTS)
    # Heaviest tasks first: they hit idle workers immediately instead of
    # serialising the tail of the run.
    tasks.sort(
        key=lambda t: -_FIGURE_WEIGHTS.get(t[0].removeprefix("generate_"), 0.0)
    )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {
            pool.submit(_run_figure_task, name, variants, img_dir): (name, variants)
            for name, variants in tasks
        }
        _done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for future in not_done:
            future.cancel()
    # The pool has drained here (the `with` block shut it down), so every
    # non-cancelled future is settled; scanning them all also aggregates
    # failures from tasks that were still running when wait() returned on
    # the first exception.
    failures: list[str] = []
    cancelled = 0
    for future, (name, variants) in futures.items():
        if future.cancelled():
            cancelled += 1
            continue
        if (exc := future.exception()) is not None:
            labels = ", ".join(
                f"{lang} {'dark' if dark else 'light'}" for lang, dark in variants
            )
            # The executor chains the worker's formatted traceback as
            # __cause__; keep it, or a failure only says what raised, not
            # where in the generators.
            detail = f"{exc!r}"
            if exc.__cause__ is not None:
                detail += f"\n{exc.__cause__}"
            failures.append(f"{name} [{labels}]: {detail}")
    if failures:
        skipped = (
            f"\n  ({cancelled} queued tasks cancelled, not attempted)" if cancelled else ""
        )
        raise RuntimeError(
            "figure generation failed:\n  " + "\n  ".join(sorted(failures)) + skipped
        )


# Approximate per-clip render cost (all four variants, the FDTD simulation
# amortised once per clip through the ``lru_cache``). Used only to submit the
# heaviest clips to the pool first so a long-pole FDTD clip does not land at
# the tail; the exact values are irrelevant to correctness. The wave-field
# clips dominate (a full 2D simulation plus four WebM encodes each) and the
# weights track their captured frame counts and mesh sizes -- the SOFAR duct
# at 3 793 frames and the sub-millimetre slit absorber, whose simulation
# alone runs three quarters of an hour, are the long poles. The schematics
# are cheap and unlisted clips are treated as light.
_ANIM_WEIGHTS: dict[str, float] = {
    "anim_fdtd_room_modes": 1500.0,
    "anim_fdtd_barrier": 1000.0,
    "anim_fdtd_ground_effect": 700.0,
    "anim_fdtd_ducting": 1600.0,
    "anim_fdtd_diffusion": 750.0,
    "anim_fdtd_metadiffuser": 540.0,
    "anim_fdtd_pillar_hall": 200.0,
    "anim_standing_wave_tube": 130.0,
    "anim_sweep_deconvolution": 90.0,
    "anim_power_two_rooms": 80.0,
    "anim_specific_loudness": 80.0,
    "anim_comb_filtering": 70.0,
    "anim_flanking_paths": 70.0,
    "anim_instantaneous_intensity": 65.0,
    "anim_intensity_scan_power": 60.0,
    "anim_onset_detection": 55.0,
    "anim_time_weighting": 55.0,
    "anim_schroeder": 55.0,
    "anim_fdtd_impedance_tube": 300.0,
    "anim_fdtd_transmission_tube": 200.0,
    "anim_fdtd_slit_absorber": 1500.0,
    "anim_fdtd_aperture_slit": 800.0,
    "anim_fdtd_refraction": 900.0,
    "anim_fdtd_expansion_chamber": 250.0,
    "anim_elastic_plate_junction": 420.0,
    "anim_elastic_coincidence": 640.0,
}

# A clip rename must not silently drop its scheduling weight; fail fast.
if not _ANIM_WEIGHTS.keys() <= _ANIMATIONS.keys():
    _unknown_anim = sorted(_ANIM_WEIGHTS.keys() - _ANIMATIONS.keys())
    raise RuntimeError(f"animation names not in _ANIMATIONS: {_unknown_anim}")


def _run_anim_task(clip: str, img_dir: str) -> str:
    """Render one clip in all four language/theme variants (worker entry).

    All four variants render in the SAME worker so a clip's expensive FDTD
    simulation -- memoised with ``lru_cache`` -- is computed once and reused
    across the variants, exactly as the sequential loop's per-process cache
    did. Splitting the variants across workers would recompute the field
    four times. Each variant applies its language/theme to this worker's
    module globals before rendering, and the epilogue closes every figure so
    pyplot state cannot leak into the next clip that reuses the process.
    """
    func = _ANIMATIONS[clip]
    try:
        for lang, dark in _VARIANTS:
            set_lang(lang)
            set_theme(dark)
            func(img_dir)
    finally:
        plt.close("all")
        # The banner's 2.5 mm frame stack is ~2.5 GB; do not let it stay
        # pinned in this worker while it renders the next clip.
        _pillar_fields.cache_clear()
    return clip


def _generate_animations_parallel(
    img_dir: str, clips: list[str], jobs: int
) -> None:
    """Render ``clips`` (all four variants each) on a ``jobs``-wide pool.

    Mirrors :func:`_generate_figures_parallel`: spawned workers each import a
    pristine module (numerical thread pools pinned to one thread), one grouped
    task per clip, heaviest clips submitted first so the long-pole FDTD field
    clips are not left to serialise at the tail of the run.
    """
    import multiprocessing as mp
    from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

    tasks = sorted(clips, key=lambda c: -_ANIM_WEIGHTS.get(c, 0.0))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        futures = {
            pool.submit(_run_anim_task, clip, img_dir): clip for clip in tasks
        }
        _done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
        for future in not_done:
            future.cancel()
    failures: list[str] = []
    cancelled = 0
    for future, clip in futures.items():
        if future.cancelled():
            cancelled += 1
            continue
        if (exc := future.exception()) is not None:
            detail = f"{exc!r}"
            if exc.__cause__ is not None:
                detail += f"\n{exc.__cause__}"
            failures.append(f"{clip}: {detail}")
    if failures:
        skipped = (
            f"\n  ({cancelled} queued clips cancelled, not attempted)"
            if cancelled else ""
        )
        raise RuntimeError(
            "animation generation failed:\n  "
            + "\n  ".join(sorted(failures)) + skipped
        )


def _select_figures(names: list[str] | None) -> list[Callable[[str], None]]:
    """Resolve ``--figure`` names (with or without the ``generate_`` prefix)."""
    if not names:
        return list(_FIGURE_FUNCS)
    by_name = {f.__name__.removeprefix("generate_"): f for f in _FIGURE_FUNCS}
    selected = []
    for raw in names:
        name = raw.removeprefix("generate_")
        if name not in by_name:
            available = ", ".join(sorted(by_name))
            raise SystemExit(f"unknown figure {raw!r}; available: {available}")
        selected.append(by_name[name])
    return selected


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: figures by default, clips behind ``--animations``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate the documentation figures (and animations) "
        "in all four language x theme variants."
    )
    parser.add_argument(
        "--animations", action="store_true",
        help="render only the Tier-1 animation clips (slow ffmpeg encoding, "
        "kept out of the default figure run)",
    )
    parser.add_argument(
        "--posters", action="store_true",
        help="re-extract only the animation poster stills from the "
        "already-rendered WebM files (no clip re-encoding)",
    )
    parser.add_argument(
        "--anim", action="append", default=None, metavar="NAME",
        help="with --animations, render only this clip (repeatable; use "
        "the output stem, e.g. --anim anim_schroeder)",
    )
    parser.add_argument(
        "--all", dest="do_all", action="store_true",
        help="render both the figures and the animations",
    )
    parser.add_argument(
        "--jobs", type=int, default=None, metavar="N",
        help="worker processes for the figures (default: cores minus two, "
        "capped at 8; 1 renders sequentially in-process)",
    )
    parser.add_argument(
        "--figure", action="append", default=None, metavar="NAME",
        help="render only this figure (repeatable; the generate_ prefix is "
        "optional, e.g. --figure loudness_models_comparison)",
    )
    args = parser.parse_args(argv)

    img_dir = ".github/images"
    os.makedirs(img_dir, exist_ok=True)

    do_figs = not (args.animations or args.posters) or args.do_all
    do_anim = args.animations or args.do_all

    if args.anim and not do_anim:
        parser.error("--anim requires --animations (or --all)")
    if args.posters and not do_anim:
        print("--- Re-extracting animation posters ---")
        generate_posters(img_dir)
    jobs = args.jobs if args.jobs is not None else _default_jobs()
    if jobs < 1:
        parser.error("--jobs must be >= 1")

    if do_figs:
        funcs = _select_figures(args.figure)
        if jobs == 1:
            for lang, dark in _VARIANTS:
                set_lang(lang)
                set_theme(dark)
                print(f"--- Generating {lang} {'dark' if dark else 'light'} theme figures ---")
                for func in funcs:
                    func(img_dir)
        else:
            print(f"--- Generating figures ({len(funcs)} x 4 variants, {jobs} jobs) ---")
            _generate_figures_parallel(img_dir, funcs, jobs)

    if do_anim:
        if args.anim or jobs == 1:
            # A targeted re-render (``--anim``) and an explicit ``--jobs 1``
            # walk the clips one at a time: each clip's field is simulated
            # once and its four language/theme variants then render off that
            # single computation (see _render_anim_variants).
            generate_animations(img_dir, args.anim, variants=True)
        else:
            import shutil
            if shutil.which("ffmpeg") is None:
                raise RuntimeError(
                    "ffmpeg was not found on PATH; it is required to encode "
                    "the animation WebM/GIF outputs. Install ffmpeg and retry."
                )
            clips = list(_ANIMATIONS)
            print(f"--- Generating animations ({len(clips)} clips "
                  f"x 4 variants, {jobs} jobs) ---")
            _generate_animations_parallel(img_dir, clips, jobs)

    print("Graphics generated successfully.")


if __name__ == "__main__":
    main()
