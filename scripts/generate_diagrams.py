#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Deterministic SVG generator for the experimental-setup diagrams used in the
documentation. Every diagram is emitted in a light and a dark variant
(``*_dark.svg``) with the same palette as the matplotlib figures, so the
docs can theme-switch them exactly like the raster WebP plots.

Run directly or via ``make graphs``.
"""

from __future__ import annotations

import os

# Deterministic output: pin numerical thread pools to a single thread before any
# numeric backend initializes (see generate_graphs.py for the rationale).
for _threads_var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import itertools
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    suffix: str
    bg: str
    fg: str
    muted: str
    panel: str
    primary: str
    secondary: str
    accent: str


LIGHT = Theme(
    suffix="", bg="#ffffff", fg="#1a1a1a", muted="#666666", panel="#f0f2f5",
    primary="#1f77b4", secondary="#d62728", accent="#2ca02c",
)
DARK = Theme(
    suffix="_dark", bg="#0d1117", fg="#e6e6e6", muted="#9a9a9a", panel="#1c2128",
    primary="#4da3d8", secondary="#e46a6a", accent="#5abf5a",
)

_FONT = "Segoe UI, Helvetica, Arial, sans-serif"
_MONO = "Consolas, Menlo, monospace"

# Spanish variants of every user-visible string. Strings not in the table
# (numbers, unit-only labels, code identifiers) are shared between languages.
_ES: dict[str, str] = {
    "Calibration chain — from calibrator to physical units":
        "Cadena de calibración — del calibrador a unidades físicas",
    # Speech Intelligibility Index (ANSI S3.5-1997)
    "Speech Intelligibility Index computation flow (ANSI S3.5-1997)":
        "Flujo de cálculo del índice de inteligibilidad del habla (ANSI S3.5-1997)",
    # Room-noise rating methods (ANSI/ASA S12.2-2019)
    "Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II":
        "Calificación del ruido de salas (S12.2-2019): NC y RC Mark II",
    "Octave-band sound pressure levels  L(f)":
        "Niveles de presión acústica por banda de octava  L(f)",
    "NC — tangency method": "NC — método de tangencia",
    "Table 1 curves": "curvas de la Tabla 1",
    "NC value in each band": "Valor NC en cada banda",
    "curve level = L(f) at that f": "nivel de la curva = L(f) en esa f",
    "NC = highest curve touched": "NC = curva más alta tocada",
    "note the governing band": "anotar la banda determinante",
    "NC-NN (band)": "NC-NN (banda)",
    "RC Mark II  (Annex D)": "RC Mark II  (Anexo D)",
    "−5 dB/octave curves": "curvas de −5 dB/octava",
    "RC = round(LMF)   (clause D.4)": "RC = redondeo(LMF)   (cláusula D.4)",
    "Spectral tag  (clause D.3)": "Etiqueta espectral  (cláusula D.3)",
    "R  rumble: a band ≤ 500 Hz exceeds RC by > 5 dB":
        "R  retumbo: una banda ≤ 500 Hz supera RC en > 5 dB",
    "H  hiss: a band ≥ 1000 Hz exceeds RC by > 3 dB":
        "H  siseo: una banda ≥ 1000 Hz supera RC en > 3 dB",
    "N  neutral: within both tolerances":
        "N  neutro: dentro de ambas tolerancias",
    # Hearing threshold (ISO 7029 / ISO 389-7)
    "Hearing-threshold model (ISO 7029 age distribution, ISO 389-7 zero)":
        "Modelo del umbral de audición (ISO 7029 por edad, cero ISO 389-7)",
    "Age Y,  sex,  population fractile Q":
        "Edad Y,  sexo,  fractil poblacional Q",
    "audiometric frequencies 125 Hz – 8000 Hz":
        "frecuencias audiométricas 125 Hz – 8000 Hz",
    "Median deviation from age 18   (ISO 7029, 4.2)":
        "Desviación mediana respecto a los 18 años   (ISO 7029, 4.2)",
    "dHmd = a · (Y − 18) ^ b   (Table 1, by sex)":
        "dHmd = a · (Y − 18) ^ b   (Tabla 1, por sexo)",
    "Spread su / sl   (ISO 7029, 4.3)":
        "Dispersión su / sl   (ISO 7029, 4.3)",
    "degree-5 polynomials in (Y − 18)   (Tables 2–5)":
        "polinomios de grado 5 en (Y − 18)   (Tablas 2–5)",
    "Fractile threshold   (ISO 7029, 4.4)":
        "Umbral del fractil   (ISO 7029, 4.4)",
    "dHQ = dHmd + z(Q) * s   (su if Q >= 0.5, else sl)":
        "dHQ = dHmd + z(Q) * s   (su si Q >= 0.5, si no sl)",
    "Expected hearing threshold level (dB HL)":
        "Nivel del umbral de audición esperado (dB HL)",
    "referenced to the audiometric zero":
        "referido al cero audiométrico",
    "Audiometric zero = ISO 389-7 reference threshold":
        "Cero audiométrico = umbral de referencia ISO 389-7",
    "free-field / diffuse-field (Table 1) — the dB HL / dB SPL zero":
        "campo libre / campo difuso (Tabla 1) — el cero dB HL / dB SPL",
    # Measurement uncertainty (ISO/IEC Guide 98-3 / Supplement 1)
    "Uncertainty: GUM propagation vs Monte Carlo (Guide 98-3)":
        "Incertidumbre: propagación GUM frente a Monte Carlo (Guía 98-3)",
    "Measurement model  y = f(x_1, …, x_N)":
        "Modelo de medida  y = f(x_1, …, x_N)",
    "input estimates x_i with standard uncertainties u(x_i)":
        "estimaciones de entrada x_i con incertidumbres típicas u(x_i)",
    "Law of propagation  (GUM 5)": "Ley de propagación  (GUM 5)",
    "sensitivity c_i = ∂f / ∂x_i": "sensibilidad c_i = ∂f / ∂x_i",
    "Combine in quadrature": "Combinación en cuadratura",
    "uc² = Σ c_i² u²(x_i) + correlation":
        "uc² = Σ c_i² u²(x_i) + correlación",
    "Effective dof  (Annex G.4)": "Grados de libertad efectivos  (Anexo G.4)",
    "v_eff — Welch–Satterthwaite": "v_eff — Welch–Satterthwaite",
    "U = k · uc": "U = k · uc",
    "k = t_p(v_eff)   (clause 6)": "k = t_p(v_eff)   (cláusula 6)",
    "Monte Carlo  (Suppl. 1, 7)": "Monte Carlo  (Supl. 1, 7)",
    "draw x_i from its PDF g(x_i)": "muestrear x_i de su PDF g(x_i)",
    "Propagate M trials": "Propagar M ensayos",
    "y_r = f(x_1r, …, x_Nr)": "y_r = f(x_1r, …, x_Nr)",
    "Sort {y_r}, take fractiles": "Ordenar {y_r}, tomar fractiles",
    "prob.-symmetric 95 % interval": "intervalo simétrico en prob. al 95 %",
    "coverage interval": "intervalo de cobertura",
    "[y_low, y_high]   (clause 7.7)": "[y_low, y_high]   (cláusula 7.7)",
    # Noise-induced hearing loss (ISO 1999)
    "Noise-induced hearing loss (ISO 1999): NIPTS and HTLAN":
        "Pérdida auditiva inducida por ruido (ISO 1999): NIPTS y HTLAN",
    "Age Y,  sex,  fractile Q": "Edad Y,  sexo,  fractil Q",
    "database A = ISO 7029": "base de datos A = ISO 7029",
    "Exposure L_EX,8h,  t years": "Exposición L_EX,8h,  t años",
    "normalized to 8 h / 5 days": "normalizada a 8 h / 5 días",
    "Age threshold  H  (HTLA)": "Umbral por edad  H  (HTLA)",
    "ISO 7029 fractile, dB": "fractil ISO 7029, dB",
    "Median NIPTS  N50  (6.3.1)": "NIPTS mediana  N50  (6.3.1)",
    "N50 = [u + v·lg(t/t0)]·(L − L0)²":
        "N50 = [u + v·lg(t/t0)]·(L − L0)²",
    "Fractile NIPTS  N  (6.3.2)": "NIPTS del fractil  N  (6.3.2)",
    "N = N50 + z·(du if z ≥ 0 else dl)":
        "N = N50 + z·(du si z ≥ 0, si no dl)",
    "HTLAN   H' = H + N − H·N / 120": "HTLAN   H' = H + N − H·N / 120",
    "threshold from age and noise  (Formula 1, 6.1)":
        "umbral por edad y ruido  (Fórmula 1, 6.1)",
    # Impulsive-sound prominence (NT ACOU 112)
    "Impulsive-sound prominence and LAeq adjustment (NT ACOU 112)":
        "Prominencia de sonidos impulsivos y ajuste de LAeq (NT ACOU 112)",
    "A-weighted level history  L_pAF  (time weighting F)":
        "Historia del nivel ponderado A  L_pAF  (ponderación F)",
    "an onset = a stretch where the gradient exceeds 10 dB/s (clauses 4.5-4.7)":
        "un arranque = tramo donde el gradiente supera 10 dB/s (cláusulas 4.5-4.7)",
    "Per impulse: onset rate OR and level difference LD":
        "Por impulso: tasa de crecimiento OR y diferencia de nivel LD",
    "OR = onset slope [dB/s],   LD = Le − Ls [dB]":
        "OR = pendiente de crecimiento [dB/s],   LD = Le − Ls [dB]",
    "Predicted prominence  P   (clause 7, Formula 1)":
        "Prominencia prevista  P   (cláusula 7, Fórmula 1)",
    "P = 3·lg(OR) + 2·lg(LD);   highest P over 30 min governs":
        "P = 3·lg(OR) + 2·lg(LD);   la P más alta en 30 min gobierna",
    "Adjustment  KI   (clause 8, Formula 2)":
        "Ajuste  KI   (cláusula 8, Fórmula 2)",
    "KI = 1.8·(P − 5) dB for P > 5, else 0":
        "KI = 1.8·(P − 5) dB si P > 5, si no 0",
    "Rating level  LAr,T = 10·lg( (1/T) Σ Δt·10^((LAeq+KI)/10) )":
        "Nivel de evaluación  LAr,T = 10·lg( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
    "impulse-adjusted level over the reference time  (Note 1)":
        "nivel ajustado por impulsos sobre el tiempo de referencia  (Nota 1)",
    "Vertical seat acceleration  az(t)":
        "Aceleración vertical del asiento  az(t)",
    "band-limited per ISO 2631-1  (0.4 Hz to 100 Hz)":
        "limitada en banda según ISO 2631-1  (0,4 Hz a 100 Hz)",
    "Spinal response  Az(t)  (clause 5.2, Formula 1/2)":
        "Respuesta de la columna  Az(t)  (cláusula 5.2, Fórmula 1/2)",
    "seat-to-spine transfer function H(f): 1 zero, 6 poles":
        "función de transferencia asiento-columna H(f): 1 cero, 6 polos",
    "Acceleration dose  Dz = 1.07·(Σ Az,i^6)^(1/6)  (Formula 3)":
        "Dosis de aceleración  Dz = 1.07·(Σ Az,i^6)^(1/6)  (Fórmula 3)",
    "Az,i = positive peaks;   daily dose Dzd = Dz·(td/tm)^(1/6)":
        "Az,i = picos positivos;   dosis diaria Dzd = Dz·(td/tm)^(1/6)",
    "Compressive stress  Sd = mz·Dzd  (Annex C, Formula C.1)":
        "Tensión compresiva  Sd = mz·Dzd  (Anexo C, Fórmula C.1)",
    "mz = 0.029 (male) / 0.025 (female) MPa per m/s²":
        "mz = 0.029 (hombre) / 0.025 (mujer) MPa por m/s²",
    "Stress variable  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)":
        "Variable de tensión  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)",
    "Su = 6.75 − Sage·(b+i) MPa, cumulated over exposure years (C.3/C.4)":
        "Su = 6.75 − Sage·(b+i) MPa, acumulada sobre los años de exposición (C.3/C.4)",
    "Injury probability  P(R) = 1 − exp(−(R/α)^β)  (Formula C.5)":
        "Probabilidad de lesión  P(R) = 1 − exp(−(R/α)^β)  (Fórmula C.5)",
    "Weibull risk of lumbar injury, by sex (Table C.1/C.2)":
        "riesgo de lesión lumbar de Weibull, por sexo (Tabla C.1/C.2)",
    "Multiple-shock spinal-response dose and injury risk (ISO 2631-5)":
        "Dosis espinal por choques múltiples y riesgo de lesión (ISO 2631-5)",
    "Surfaces  (Si, αs,i)": "Superficies  (Si, αs,i)",
    "area and absorption per band": "área y absorción por banda",
    "Objects  (Vobj)": "Objetos  (Vobj)",
    "Aobj = Vobj^(2/3)  (Formula 4)": "Aobj = Vobj^(2/3)  (Fórmula 4)",
    "Equivalent absorption area  A  (clause 4.3, Formula 1)":
        "Área de absorción equivalente  A  (cláusula 4.3, Fórmula 1)",
    "A = Σ αs,i·Si + Σ Aobj + Aair;   Aair = 4·m·V·(1 − ψ)  (Formula 2)":
        "A = Σ αs,i·Si + Σ Aobj + Aair;   Aair = 4·m·V·(1 − ψ)  (Fórmula 2)",
    "Object fraction  ψ = Σ Vobj / V   (Formula 3)":
        "Fracción de objetos  ψ = Σ Vobj / V   (Fórmula 3)",
    "air absorption negligible below 1 kHz for V < 200 m³":
        "absorción del aire despreciable bajo 1 kHz si V < 200 m³",
    "Reverberation time  T = 55.3/c₀ · V·(1 − ψ) / A  (Formula 5)":
        "Tiempo de reverberación  T = 55.3/c₀ · V·(1 − ψ) / A  (Fórmula 5)",
    "c₀ = 345.6 m/s so 55.3/c₀ = 0.16  (clause 4.4)":
        "c₀ = 345.6 m/s, así 55.3/c₀ = 0.16  (cláusula 4.4)",
    "Absorption area and reverberation time of a room (EN 12354-6)":
        "Área de absorción y tiempo de reverberación de una sala (EN 12354-6)",
    "Speech  Ei'": "Habla  Ei'",
    "Noise  Ni'": "Ruido  Ni'",
    "Threshold  Ti'": "Umbral  Ti'",
    "spectrum level (dB)": "nivel espectral (dB)",
    "Self-masking + spread of masking": "Automáscara + propagación de la máscara",
    "Zi   (clause 5.4)": "Zi   (cláusula 5.4)",
    "Equivalent disturbance Di": "Perturbación equivalente Di",
    "max(masking, internal noise) (5.6)":
        "máx(máscara, ruido interno) (5.6)",
    "Band audibility Ai = (Ei' − Di + 15)/30":
        "Audibilidad de banda Ai = (Ei' − Di + 15)/30",
    "clipped to [0, 1]   (clause 5.8)": "acotada a [0, 1]   (cláusula 5.8)",
    "band importance I_i (Table 3)  ·  index in [0, 1]  (clause 6)":
        "importancia de banda I_i (Tabla 3)  ·  índice en [0, 1]  (cláusula 6)",
    # Human vibration (ISO 2631-1 / ISO 8041-1 / 2002-44-EC)
    "Whole-body vibration measurement chain (ISO 2631-1 / ISO 8041-1)":
        "Cadena de medición de vibración de cuerpo entero (ISO 2631-1 / ISO 8041-1)",
    "vibration input": "entrada de vibración",
    "Seat/body interface": "Interfaz asiento/cuerpo",
    "Triaxial accelerometer": "Acelerómetro triaxial",
    "Band limiting + Wk / Wd": "Limitación de banda + Wk / Wd",
    "weighting (ISO 8041-1)": "ponderación (ISO 8041-1)",
    "Weighted r.m.s. a_w  &  VDV": "a_w eficaz ponderada  y  VDV",
    "(ISO 2631-1)": "(ISO 2631-1)",
    "assessed vs EAV / ELV (Directive 2002/44/EC)":
        "evaluada frente a EAV / ELV (Directiva 2002/44/CE)",
    "Sound calibrator": "Calibrador acústico",
    "Microphone +": "Micrófono +",
    "preamplifier": "preamplificador",
    "Audio interface": "Interfaz de audio",
    "Pa per": "Pa por",
    "digital unit": "unidad digital",
    "Stability: |max − mean| and |min − mean| ≤ 0.07 dB":
        "Estabilidad: |máx − media| y |mín − media| ≤ 0,07 dB",
    "(IEC 60942:2017 Table 2, class 1) — else CalibrationWarning":
        "(IEC 60942:2017 Tabla 2, clase 1) — si no, CalibrationWarning",
    "Environmental noise measurement positions (ISO 1996-2)":
        "Posiciones de medida de ruido ambiental (ISO 1996-2)",
    "Building façade": "Fachada del edificio",
    "A — free field": "A — campo libre",
    "B — 2 m from façade": "B — a 2 m de la fachada",
    "C — flush-mounted": "C — enrasado en fachada",
    "4.0 ± 0.2 m": "4,0 ± 0,2 m",
    "Emission measurement positions (ECMA-74)":
        "Posiciones de medida de emisión (ECMA-74)",
    "Operator — seated (P2)": "Operador — sentado (P2)",
    "Bystanders — top view": "Observadores — vista en planta",
    "height 1.50 m": "altura 1,50 m",
    "0.25 m": "0,25 m",
    "1.20 m": "1,20 m",
    "1.00 m": "1,00 m",
    "phonometry processing chain": "Cadena de procesado de phonometry",
    "Signal": "Señal",
    "Calibrate": "Calibrar",
    "Weighting": "Ponderación",
    "Octave": "Octavas",
    "bands 1/b": "bandas 1/b",
    "Ballistics": "Temporal",
    "Metrics": "Métricas",
    "Multirate decimation in the octave filter bank":
        "Decimación multitasa en el banco de filtros de octava",
    "16 kHz band": "Banda de 16 kHz",
    "1 kHz band": "Banda de 1 kHz",
    "63 Hz band": "Banda de 63 Hz",
    "no decimation": "sin decimación",
    "Anti-alias": "Antialias",
    "Low bands are filtered at a decimated rate: the relative":
        "Las bandas graves se filtran a frecuencia decimada: el ancho",
    "bandwidth stays wide, so the SOS stays numerically healthy.":
        "relativo se mantiene amplio y las SOS siguen bien condicionadas.",
    "Two-microphone (p-p) intensity probe":
        "Sonda de intensidad p-p (dos micrófonos)",
    "measurement axis / intensity direction":
        "eje de medida / dirección de la intensidad",
    "u from the p2−p1 gradient": "u a partir del gradiente p2−p1",
    "STI measurement chain (IEC 60268-16)":
        "Cadena de medida STI (IEC 60268-16)",
    "Source": "Fuente",
    "STIPA signal": "Señal STIPA",
    "Room": "Sala",
    "reverberation + noise": "reverberación + ruido",
    "Microphone": "Micrófono",
    "Analysis": "Análisis",
    "m(F) drops": "m(F) cae",
    "Airborne sound insulation setup (ISO 16283-1)":
        "Montaje de aislamiento acústico aéreo (ISO 16283-1)",
    "Source room": "Recinto emisor",
    "Receiving room": "Recinto receptor",
    "Test partition": "Partición de ensayo",
    "Loudspeaker": "Altavoz",
    "microphone positions": "posiciones de micrófono",
    "≥ 1.0 m": "≥ 1,0 m",
    "≥ 0.7 m": "≥ 0,7 m",
    "≥ 0.5 m": "≥ 0,5 m",
    "7.6 a) ≥ 0.7 m between microphone positions":
        "7.6 a) ≥ 0,7 m entre posiciones de micrófono",
    "7.6 b) ≥ 0.5 m to room boundaries":
        "7.6 b) ≥ 0,5 m a los límites del recinto",
    "7.6 c) ≥ 1.0 m to the loudspeaker":
        "7.6 c) ≥ 1,0 m al altavoz",
    "7.2.2 ≥ 1.0 m loudspeaker to separating partition":
        "7.2.2 ≥ 1,0 m del altavoz a la partición separadora",
    "Impulse-response measurement chain (ISO 18233)":
        "Cadena de medición de la respuesta al impulso (ISO 18233)",
    "Excitation": "Excitación",
    "ESS sweep / MLS": "Barrido ESS / MLS",
    "Deconvolution": "Deconvolución",
    "correlation /": "correlación /",
    "inverse filter": "filtro inverso",
    "acoustic path": "trayecto acústico",
    "The room response h(t) is recovered by deconvolving the microphone signal.":
        "La respuesta de la sala h(t) se recupera deconvolucionando "
        "la señal del micrófono.",
    # d10 - ISO 3744/3746 sound power measurement surfaces
    "ISO 3744 / 3746 sound power measurement surfaces":
        "Superficies de medición de potencia acústica (ISO 3744 / 3746)",
    "Hemispherical surface": "Superficie hemisférica",
    "Reflecting plane": "Plano reflectante",
    "Measurement surface": "Superficie de medición",
    "Parallelepiped surface": "Superficie de paralelepípedo",
    "radius r ≥ 2 d₀": "radio r ≥ 2 d₀",
    "measurement distance d": "distancia de medición d",
    "10 key positions (Table B.1)": "10 posiciones clave (Tabla B.1)",
    "one plane · S = 2πr²": "un plano · S = 2πr²",
    "one plane · S = 4(ab+bc+ca)": "un plano · S = 4(ab+bc+ca)",
    # d11 - ISO 16283-2 impact sound insulation setup
    "ISO 16283-2 impact sound insulation setup":
        "Montaje de aislamiento de ruido de impactos (ISO 16283-2)",
    "Source room (upper)": "Recinto emisor (superior)",
    "Receiving room (lower)": "Recinto receptor (inferior)",
    "Separating floor": "Forjado separador",
    "Tapping machine": "Máquina de impactos",
    "Microphone positions": "Posiciones de micrófono",
    "structure-borne impact": "impacto estructural",
    "radiated impact sound": "ruido de impactos radiado",
    "Impact sound insulation": "Aislamiento de impactos",
    "Li = energy-averaged": "Li = promedio en energía",
    "band level (Formula 10)": "del nivel de banda (Fórmula 10)",
    "A = 0.16 V/T  (Sabine)": "A = 0,16 V/T  (Sabine)",
    "T₀ = 0.5 s , A₀ = 10 m²": "T₀ = 0,5 s , A₀ = 10 m²",
    # d12 - sound power methods comparison
    "Sound power methods compared": "Métodos de potencia acústica comparados",
    "Free field over a reflecting plane":
        "Campo libre sobre plano reflectante",
    "Reverberation test room": "Sala reverberante de ensayo",
    "In situ — any environment": "In situ — cualquier entorno",
    "Grade 2 / 3 (engineering / survey)":
        "Grado 2 / 3 (ingeniería / control)",
    "Grade 1 (precision)": "Grado 1 (precisión)",
    "Sound pressure · enveloping surface":
        "Presión acústica · superficie envolvente",
    "Sound pressure · diffuse field": "Presión acústica · campo difuso",
    "Sound intensity · scanning": "Intensidad acústica · barrido de intensidad",
    "K2A ≤ 4 dB (3744) / ≤ 7 dB (3746)":
        "K2A ≤ 4 dB (3744) / ≤ 7 dB (3746)",
    "V ≥ 200 m³ , qualified room": "V ≥ 200 m³ , sala cualificada",
    "no negative-power bands": "sin bandas de potencia negativa",
    "Method": "Método",
    "Environment": "Entorno",
    "Accuracy": "Exactitud",
    # d13 - EN 12354 direct and flanking transmission paths
    "Direct and flanking transmission paths (EN 12354)":
        "Caminos de transmisión directa y por flancos (EN 12354)",
    "Separating element (D, d)": "Elemento separador (D, d)",
    "Flanking element (F, f)": "Elemento de flanco (F, f)",
    "junction": "unión",
    "Dd — direct path: separating element both sides":
        "Dd — camino directo: elemento separador en ambos lados",
    "Ff — flanking–flanking: flanking element both sides":
        "Ff — flanco–flanco: elemento de flanco en ambos lados",
    "Fd — flanking (source) → separating (receiving)":
        "Fd — flanco (emisor) → separador (receptor)",
    "Df — separating (source) → flanking (receiving)":
        "Df — separador (emisor) → flanco (receptor)",
    "R'w = −10 lg Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)":
        "R'w = −10 lg Σ 10^(−Rij,w /10) dB   (EN 12354-1, Fórmula 26)",
    # d14 - ISO 9613-2 outdoor propagation geometry
    "ISO 9613-2 source–barrier–receiver geometry":
        "Geometría fuente–barrera–receptor (ISO 9613-2)",
    "Receiver": "Receptor",
    "Barrier": "Barrera",
    "Ground (Gs, Gm, Gr)": "Suelo (Gs, Gm, Gr)",
    "diffracted path": "trayecto difractado",
    "direct path (blocked)": "trayecto directo (bloqueado)",
    "z = dss + dsr − d   (path difference)":
        "z = dss + dsr − d   (diferencia de camino)",
    "Dz = 10 lg[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)":
        "Dz = 10 lg[ 3 + (C₂/λ) C₃ z Kmet ]   (Ec. 14)",
    # Impedance tube (ISO 10534) setup
    "Impedance tube: two-microphone method (ISO 10534-2)":
        "Tubo de impedancia: método de dos micrófonos (ISO 10534-2)",
    "Test specimen": "Probeta de ensayo",
    "Rigid backing": "Terminación rígida",
    "incident": "incidente",
    "reflected": "reflejada",
    "H₁₂ → reflection factor r (Eq. 17), absorption α = 1 − |r|² (Eq. 18), "
    "Z/ρc₀ = (1+r)/(1−r) (Eq. 19)":
        "H₁₂ → factor de reflexión r (Ec. 17), absorción α = 1 − |r|² (Ec. 18), "
        "Z/ρc₀ = (1+r)/(1−r) (Ec. 19)",
    "Working range f_l < f < f_u set by the microphone spacing s "
    "and the tube diameter (Clause 6.1)":
        "Rango útil f_l < f < f_u fijado por la separación s de micrófonos "
        "y el diámetro del tubo (Cláusula 6.1)",
    "ASTM E2611: two further microphones behind the specimen also "
    "give the transmission loss":
        "ASTM E2611: dos micrófonos más tras la probeta dan también "
        "la pérdida por transmisión",
    # Four-microphone tube (ASTM E2611) setup
    "Four-microphone transmission-loss tube (ASTM E2611)":
        "Tubo de pérdida por transmisión de cuatro micrófonos (ASTM E2611)",
    "Termination": "Terminación",
    "(2 loads)": "(2 cargas)",
    "Decompose A, B (upstream) and C, D (downstream) → transfer matrix T (Eq. 22)":
        "Descomponer A, B (aguas arriba) y C, D (aguas abajo) → "
        "matriz de transferencia T (Ec. 22)",
    "TL = 20 log₁₀ |(T₁₁ + T₁₂/ρc + ρc·T₂₁ + T₂₂) / 2|   (Eq. 26)":
        "TL = 20 log₁₀ |(T₁₁ + T₁₂/ρc + ρc·T₂₁ + T₂₂) / 2|   (Ec. 26)",
    "Two-load method: repeat with two terminations; the one-load "
    "variant uses a single anechoic end":
        "Método de dos cargas: repetir con dos terminaciones; la variante "
        "de una carga usa un único extremo anecoico",
    # Airflow resistance (ISO 9053) setup
    "Airflow resistance: static and alternating methods (ISO 9053-1/-2)":
        "Resistencia al flujo: métodos estático y alternante (ISO 9053-1/-2)",
    "Static method (ISO 9053-1)": "Método estático (ISO 9053-1)",
    "specimen (A, d)": "probeta (A, d)",
    "laminar flow  q_v": "flujo laminar  q_v",
    "manom.": "manóm.",
    "R = Δp / q_v   (through-origin fit at 0.5 mm/s)":
        "R = Δp / q_v   (ajuste por el origen a 0,5 mm/s)",
    "Alternating method (ISO 9053-2)": "Método alternante (ISO 9053-2)",
    "cavity": "cavidad",
    "specimen / airtight": "probeta / cierre estanco",
    "piston  f = 1–4 Hz": "pistón  f = 1–4 Hz",
    "R from L_p,s − L_p,t   (κ′ per Annex A)":
        "R por L_p,s − L_p,t   (κ′ según Anexo A)",
    # d15 - ISO 17497-1 random-incidence scattering (reverberation room)
    "Random-incidence scattering in a reverberation room (ISO 17497-1)":
        "Dispersión a incidencia aleatoria en cámara reverberante (ISO 17497-1)",
    "Reverberation room": "Cámara reverberante",
    "Turntable (test sample)": "Plataforma giratoria (probeta)",
    "Rotating boom source": "Fuente en brazo giratorio",
    "stationary → α_s": "estática → α_s",
    "rotating → α_spec": "girando → α_spec",
    "Stationary sample → α_s (Eq. 1)   ·   rotating / averaged → α_spec (Eq. 4)":
        "Probeta estática → α_s (Ec. 1)   ·   girando / promediada → α_spec (Ec. 4)",
    "s = (α_spec − α_s) / (1 − α_s)   (Eq. 5)":
        "s = (α_spec − α_s) / (1 − α_s)   (Ec. 5)",
    "α from 55.3·(V/S)·(1/cT) − 4(V/S)m   (Sabine, Table 2 rows T1–T4)":
        "α con 55,3·(V/S)·(1/cT) − 4(V/S)m   (Sabine, filas T1–T4 de la Tabla 2)",
    "Base-plate check: s_base ≤ Table 1 limit (Clause 6.2)":
        "Placa base: s_base ≤ límite de la Tabla 1 (Cláusula 6.2)",
    # d16 - ISO 17497-2 free-field diffusion goniometer
    "Free-field diffusion goniometer (ISO 17497-2)":
        "Goniómetro de difusión en campo libre (ISO 17497-2)",
    "Test sample": "Probeta de ensayo",
    "Turntable": "Plataforma giratoria",
    "Fixed source": "Fuente fija",
    "polar response L_i": "respuesta polar L_i",
    "receiver arc (5° steps)": "arco de receptores (pasos de 5°)",
    "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / [(n−1)·Σ(10^(L_i/10))²]   (Formula 5)":
        "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / [(n−1)·Σ(10^(L_i/10))²]   (Fórmula 5)",
    "d_n = (d − d_ref) / (1 − d_ref)   (Formula 7)":
        "d_n = (d − d_ref) / (1 − d_ref)   (Fórmula 7)",
    "5° receiver steps · turntable rotates the sample · source fixed":
        "pasos de 5° entre receptores · la plataforma gira la probeta · fuente fija",
    # d17 - ISO 13472-1 in-situ road absorption, subtraction technique
    "In-situ road absorption — subtraction technique (ISO 13472-1)":
        "Absorción in situ de carreteras — técnica de sustracción (ISO 13472-1)",
    "Road surface": "Superficie de la carretera",
    "direct  ds−dm": "directo  ds−dm",
    "reflected  ds+dm": "reflejado  ds+dm",
    "to image source (ds below)": "hacia fuente imagen (ds por debajo)",
    "ds = 1.25 m": "ds = 1,25 m",
    "dm = 0.25 m": "dm = 0,25 m",
    "Free-field reference": "Referencia en campo libre",
    "Hi: no ground reflection in the window":
        "Hi: sin reflexión del suelo en la ventana",
    "Kr = (ds − dm)/(ds + dm) = 2/3   (Clause 4.1)":
        "Kr = (ds − dm)/(ds + dm) = 2/3   (Cláusula 4.1)",
    "α(f) = 1 − (1/Kr²)·|Hr/Hi|²   ·   Δτ = 2 dm / c":
        "α(f) = 1 − (1/Kr²)·|Hr/Hi|²   ·   Δτ = 2 dm / c",
    "Adrienne time window isolates the reflected response Hr":
        "La ventana temporal Adrienne aísla la respuesta reflejada Hr",
    # d18 - ISO 13472-2 in-situ road absorption, spot method
    "In-situ road absorption — spot method (ISO 13472-2)":
        "Absorción in situ de carreteras — método puntual (ISO 13472-2)",
    "Road surface (test sample)": "Superficie de carretera (probeta)",
    "Spot method (ISO 13472-2)": "Método puntual (ISO 13472-2)",
    "f_u = 0.58 c₀ / d   (Clause 5.4.1)":
        "f_u = 0,58 c₀ / d   (Cláusula 5.4.1)",
    "0.05 c₀/f_min < s < 0.45 c₀/f_max   (Clause 5.4.2)":
        "0,05 c₀/f_min < s < 0,45 c₀/f_max   (Cláusula 5.4.2)",
    "Working range: 250–1600 Hz (1/3-octave)":
        "Rango útil: 250–1600 Hz (1/3 de octava)",
    "Two-microphone transfer function H₁₂":
        "Función de transferencia de dos micrófonos H₁₂",
    "→ ISO 10534-2 decomposition → α(f)":
        "→ descomposición ISO 10534-2 → α(f)",
    "Tube sealed onto the road; plane waves only below f_u":
        "Tubo sellado sobre la carretera; solo ondas planas por debajo de f_u",
    # d19 - ISO 3745 precision sound power (anechoic / hemi-anechoic room)
    "Precision sound power in an anechoic room (ISO 3745)":
        "Potencia acústica de precisión en sala anecoica (ISO 3745)",
    "Reflecting plane (hemi-anechoic)": "Plano reflectante (semianecoica)",
    "Anechoic wedges": "Cuñas anecoicas",
    "Source (DUT)": "Fuente (DUT)",
    "20 / 40 mic positions": "20 / 40 posiciones de micrófono",
    "radius r": "radio r",
    "S = 2πr² (hemi-anechoic) · 4πr² (anechoic)":
        "S = 2πr² (semianecoica) · 4πr² (anecoica)",
    "K1: per-position background correction":
        "K1: corrección de ruido de fondo por posición",
    "C1, C2, C3: meteorological corrections (ps, θ, a(f))":
        "C1, C2, C3: correcciones meteorológicas (ps, θ, a(f))",
    # d20 - ISO 9614-3 precision sound intensity scanning
    "Precision sound intensity scanning (ISO 9614-3)":
        "Barrido de intensidad acústica de precisión (ISO 9614-3)",
    "Measurement surface (segments S_i)": "Superficie de medición (segmentos S_i)",
    "p-p probe": "sonda p-p",
    "serpentine scan": "barrido en serpentina",
    "I_n (normal intensity)": "I_n (intensidad normal)",
    "P = Σ I_n,i · S_i   (partial powers per segment)":
        "P = Σ I_n,i · S_i   (potencias parciales por segmento)",
    "Field indicators: F_pIn , FT , FS":
        "Indicadores de campo: F_pIn , FT , FS",
    "Five acceptance criteria (Annex C); band invalid if P < 0":
        "Cinco criterios de aceptación (Anexo C); banda no válida si P < 0",
    # d_room - ISO 3382-1/-2 room-acoustics measurement setup
    "Room-acoustics measurement setup (ISO 3382-1 / ISO 3382-2)":
        "Configuración de medición de acústica de salas (ISO 3382-1 / ISO 3382-2)",
    "Room plan (top view)": "Planta de la sala (vista superior)",
    "Microphone position": "Posición de micrófono",
    "Loudspeaker source": "Fuente (altavoz)",
    "ISO 3382-1 (positions):": "ISO 3382-1 (posiciones):",
    "• ≥ 2 source positions": "• ≥ 2 posiciones de fuente",
    "• mics ≥ 2 m apart": "• micrófonos ≥ 2 m entre sí",
    "• ≥ 1 m from surfaces": "• ≥ 1 m de las superficies",
    "• mic height 1.2 m": "• altura del micrófono 1,2 m",
    "ISO 3382-2 — reverberation-time measurement grades":
        "ISO 3382-2 — grados de medición del tiempo de reverberación",
    "Source pos.": "Pos. fuente",
    "Mic pos.": "Pos. micróf.",
    "Source–mic comb.": "Comb. fuente–micróf.",
    "Decays / comb.": "Decaim. / comb.",
    "Survey": "Control",
    "Engineering": "Ingeniería",
    "Precision": "Precisión",
    # --- Tanda 11: new diagrams -------------------------------------------
    "Exponential-detector chain of the time weightings (IEC 61672-1)":
        "Cadena del detector exponencial de las ponderaciones temporales "
        "(IEC 61672-1)",
    "Block processing: carrying the filter state versus resetting it":
        "Procesado por bloques: conservar el estado del filtro frente a "
        "reiniciarlo",
    "Array-shape flow through a per-channel operation":
        "Flujo de formas de array en una operación por canal",
    "Open-plan office spatial decay of speech (ISO 3382-3)":
        "Caída espacial del habla en oficina diáfana (ISO 3382-3)",
    "Measurement uncertainty from tables to expanded U (ISO 12999-1)":
        "Incertidumbre de medición: de las tablas a la U expandida (ISO 12999-1)",
    "Single-number sound-absorption rating (ISO 11654)":
        "Valoración de la absorción acústica en índice único (ISO 11654)",
    "Zwicker loudness model chain (ISO 532-1)":
        "Cadena del modelo de sonoridad de Zwicker (ISO 532-1)",
    # equal-loudness -> A-weighting
    "Why A-weighting: an equal-loudness contour, inverted (ISO 226)":
        "Por qué la ponderación A: una isófona invertida (ISO 226)",
    "Equal-loudness contours (ISO 226)": "Líneas isofónicas (ISO 226)",
    "Frequency [Hz]": "Frecuencia [Hz]",
    "40 phon": "40 fonios",
    "invert": "invertir",
    "0 dB at 1 kHz": "0 dB en 1 kHz",
    "A-weighting (IEC 61672-1)": "Ponderación A (IEC 61672-1)",
    "inverted 40-phon contour": "isófona de 40 fonios invertida",
    "A is the 40-phon contour flipped into a realizable filter: quiet sounds, "
    "where the ear discards bass hardest.":
        "A es la isófona de 40 fonios convertida en un filtro realizable: "
        "niveles bajos, donde el oído descarta más los graves.",
    "The match is deliberately loose (a 1930s convention, not a loudness "
    "model); C mirrors the flatter ~100-phon contour.":
        "Coincidencia laxa a propósito (convención de los años 30, no un "
        "modelo de sonoridad); C sigue la isófona plana de ~100 fonios.",
    # time-weighting
    "a first-order low-pass on the squared signal → the mean-square envelope":
        "un paso bajo de primer orden sobre la señal al cuadrado → la "
        "envolvente cuadrática media",
    "band signal": "señal de banda",
    "square": "cuadrado",
    "one-pole RC": "RC de un polo",
    "time constant τ": "constante de tiempo τ",
    "to decibels": "a decibelios",
    "time-weighted level": "nivel con ponderación temporal",
    "Fast (F)": "Rápida (F)",
    "Slow (S)": "Lenta (S)",
    "Impulse (I)": "Impulso (I)",
    "35 ms rise · 1500 ms fall": "35 ms subida · 1500 ms bajada",
    # block-processing
    "State carried across blocks — TimeWeighting.process()":
        "Estado conservado entre bloques — TimeWeighting.process()",
    "y[-1] (or the sosfilt zi vector) seeds the next block → identical to "
    "one continuous call":
        "y[-1] (o el vector zi de sosfilt) inicializa el bloque siguiente → "
        "idéntico a una llamada continua",
    "State reset each block — reset() or a fresh call":
        "Estado reiniciado en cada bloque — reset() o una llamada nueva",
    "every block restarts from rest → spurious discontinuities at the seams":
        "cada bloque arranca desde reposo → discontinuidades espurias en las "
        "uniones",
    "block 1": "bloque 1",
    "block 2": "bloque 2",
    "block 3": "bloque 3",
    # multichannel
    "1-D:  (samples,)": "1-D:  (muestras,)",
    "scalar": "escalar",
    "2-D:  (channels, samples)": "2-D:  (canales, muestras)",
    "(channels,)": "(canales,)",
    "reduce along": "reducir sobre",
    "axis = −1  (time)": "eje = −1  (tiempo)",
    "the channel axis 0": "el eje de canal 0",
    "rides through untouched": "pasa intacto",
    "A mono call returns a scalar; a C-channel call returns C results.":
        "Una llamada mono devuelve un escalar; una de C canales devuelve C "
        "resultados.",
    "Band metrics widen the reduced axis instead: (…, bands).":
        "Las métricas por banda ensanchan el eje reducido: (…, bandas).",
    # open-plan
    "source": "fuente",
    "spatial-decay fit range (2 m to 16 m)":
        "rango de ajuste de caída espacial (2 m a 16 m)",
    "spatial decay rate": "tasa de caída espacial",
    "dB per doubling · Cl. 6.2": "dB por duplicación · Cl. 6.2",
    "speech level at 4 m": "nivel de habla a 4 m",
    "A-weighted · Cl. 3.3": "ponderado A · Cl. 3.3",
    "distraction distance": "distancia de distracción",
    "fitted STI = 0.50 · Cl. 3.6": "STI ajustado = 0,50 · Cl. 3.6",
    "privacy distance": "distancia de privacidad",
    "fitted STI = 0.20 · Cl. 3.7": "STI ajustado = 0,20 · Cl. 3.7",
    # ISO 12999-1
    "Standard uncertainty  u  — reproducibility read from the tables":
        "Incertidumbre típica  u  — reproducibilidad leída de las tablas",
    "bands: Tables 2/4 · ratings: Tables 3/5 · situation A (σR) / B (σsitu) / "
    "C (σr)":
        "bandas: Tablas 2/4 · índices: Tablas 3/5 · situación A (σR) / "
        "B (σsitu) / C (σr)",
    "Reduce by  m  independent measurements   u/√m   (Formula A.7)":
        "Reducir con  m  mediciones independientes   u/√m   (Fórmula A.7)",
    "and combine model with reality per Annex A when predicting":
        "y combinar modelo con realidad según el Anexo A al predecir",
    "Combine uncorrelated contributions   uc = √(Σ u_i²)   (Formula C.2)":
        "Combinar contribuciones no correlacionadas   uc = √(Σ u_i²)   "
        "(Fórmula C.2)",
    "single-number combination of Annex B uses Formula B.2":
        "la combinación de índice único del Anexo B usa la Fórmula B.2",
    "Expand   U = k·u   (Formula 2),   k from Table 8   (k ≥ 1)":
        "Expandir   U = k·u   (Fórmula 2),   k de la Tabla 8   (k ≥ 1)",
    "the coverage factor depends on the reported quantity and situation":
        "el factor de cobertura depende de la magnitud reportada y la situación",
    "Report   Y = y ± U   (Formula 3)": "Reportar   Y = y ± U   (Fórmula 3)",
    "two-sided coverage factor": "factor de cobertura bilateral",
    "Declare conformity   (Formulae 4/5)": "Declarar conformidad   (Fórmulas 4/5)",
    "one-sided coverage factor": "factor de cobertura unilateral",
    # ISO 11654
    "Measured  αs  at one-third octaves, 200 Hz to 5000 Hz":
        "αs medido en tercios de octava, 200 Hz a 5000 Hz",
    "from a reverberation room (ISO 354)": "en cámara reverberante (ISO 354)",
    "Practical  αp  per octave band, 250 Hz to 4000 Hz  (Clause 4.1)":
        "αp práctico por banda de octava, 250 Hz a 4000 Hz  (Cláusula 4.1)",
    "mean of the three one-third octaves, rounded to 0.05":
        "media de los tres tercios de octava, redondeado a 0,05",
    "Shift the reference curve in 0.05 steps to best fit  (Clause 4.2)":
        "Desplazar la curva de referencia en pasos de 0,05 hasta el mejor "
        "ajuste  (Cláusula 4.2)",
    "sum of unfavourable deviations kept ≤ 0.10":
        "suma de desviaciones desfavorables ≤ 0,10",
    "Weighted coefficient  αw = shifted reference at 500 Hz":
        "Coeficiente ponderado  αw = referencia desplazada a 500 Hz",
    "Shape indicators (L, M, H) where  αp − reference ≥ 0.25":
        "Indicadores de forma (L, M, H) donde  αp − referencia ≥ 0,25",
    "Sound absorption class  A to E   (Table B.1, Annex B)":
        "Clase de absorción acústica  A a E   (Tabla B.1, Anexo B)",
    "or “Not classified” when αw falls below the class-E band":
        "o «No clasificado» cuando αw cae por debajo de la banda de clase E",
    # Zwicker
    "28 one-third-octave band levels, 25 Hz to 12.5 kHz":
        "28 niveles de banda de tercio de octava, 25 Hz a 12,5 kHz",
    "from a spectrum, or from a calibrated signal via the Annex A filterbank":
        "de un espectro, o de una señal calibrada mediante el banco de filtros "
        "del Anexo A",
    "Equal-loudness correction and lower critical bands  "
    "(Clause 5.4, Table A.3)":
        "Corrección de igual sonoridad y bandas críticas inferiores  "
        "(Cláusula 5.4, Tabla A.3)",
    "the 11 lowest bands grouped into 3 critical bands, 25-250 Hz":
        "las 11 bandas más bajas agrupadas en 3 bandas críticas, 25-250 Hz",
    "Core loudness of the 20 critical bands  (Tables A.4-A.7)":
        "Sonoridad de núcleo de las 20 bandas críticas  (Tablas A.4-A.7)",
    "a₀ transmission (A.4), diffuse-field DDF (A.5), threshold in quiet "
    "LTQ (A.6)":
        "transmisión a₀ (A.4), DDF de campo difuso (A.5), umbral en silencio "
        "LTQ (A.6)",
    "Specific loudness  N′(z)  over 0.1-Bark steps to 24 Bark":
        "Sonoridad específica  N′(z)  en pasos de 0,1 Bark hasta 24 Bark",
    "upper masking slopes added band to band (Table A.9)":
        "pendientes de enmascaramiento superior sumadas banda a banda (Tabla A.9)",
    "Total loudness  N = ∫ N′(z) dz  [sone]":
        "Sonoridad total  N = ∫ N′(z) dz  [sone]",
    "loudness level  LN = 40 + 10·log₂ N  [phon]":
        "nivel de sonoridad  LN = 40 + 10·log₂ N  [phon]",
    # Loudspeaker free-field sensitivity (IEC 60268-5)
    "Loudspeaker free-field sensitivity measurement (IEC 60268-5)":
        "Sensibilidad de altavoz en campo libre (IEC 60268-5)",
    "Reference axis": "Eje de referencia",
    "Measurement microphone": "Micrófono de medición",
    "Amplifier": "Amplificador",
    "2.83 V (8 Ω)": "2,83 V (8 Ω)",
    "Characteristic sensitivity: Lp at 1 m for 1 W into the rated impedance":
        "Sensibilidad característica: Lp a 1 m para 1 W en la impedancia nominal",
    "Up = √(R · 1 W): 2.83 V is 1 W into 8 Ω but 2 W into 4 Ω (+3 dB)":
        "Up = √(R · 1 W): 2,83 V es 1 W en 8 Ω pero 2 W en 4 Ω (+3 dB)",
    "Lp(1 m) = Lp(r) + 20 lg(r / 1 m)   (far field, inverse-distance law)":
        "Lp(1 m) = Lp(r) + 20 lg(r / 1 m)   (campo lejano, ley 1/r)",
    "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 lg(M / 1 V/Pa) dB":
        "Micrófono (IEC 60268-4): M en mV/Pa, o LM = 20 lg(M / 1 V/Pa) dB",
    # Occupational noise exposure (ISO 9612)
    "Occupational noise exposure measurement (ISO 9612)":
        "Medición de la exposición al ruido en el trabajo (ISO 9612)",
    "Worn instrument (Clause 12.3)": "Instrumento portado (apartado 12.3)",
    "≈ 0.04 m": "≈ 0,04 m",
    "above the shoulder": "sobre el hombro",
    "≥ 0.1 m from the ear canal,": "≥ 0,1 m del canal auditivo,",
    "most-exposed side": "lado del oído más expuesto",
    "Personal sound exposure meter": "Exposímetro sonoro personal",
    "(IEC 61252)": "(IEC 61252)",
    "Measurement strategies (Clauses 9–11)":
        "Estrategias de medición (apartados 9–11)",
    "Working day": "Jornada laboral",
    "Task-based (Clause 9)": "Basada en tareas (apartado 9)",
    "split the day into tasks — ≥ 3 samples (│) per task, plus each duration":
        "dividir la jornada en tareas — ≥ 3 muestras (│) y la duración por tarea",
    "Job-based (Clause 10)": "Basada en la función (apartado 10)",
    "N ≥ 5 random samples over the homogeneous exposure group":
        "N ≥ 5 muestras aleatorias sobre el grupo de exposición homogéneo",
    "Full-day (Clause 11)": "Jornada completa (apartado 11)",
    "the whole shift, at least 3 times (5 if the days differ by > 3 dB)":
        "toda la jornada, al menos 3 veces (5 si los días difieren en > 3 dB)",
    "Task 1": "Tarea 1",
    "Task 2": "Tarea 2",
    "Task 3": "Tarea 3",
    "day 1": "día 1",
    "choose by work pattern (Table B.1)  →  LEX,8h + Annex C uncertainty":
        "según el patrón de trabajo (Tabla B.1)  →  LEX,8h + U del Anexo C",
    # Dynamic-stiffness resonance rig (ISO 9052-1)
    "Dynamic-stiffness resonance rig (ISO 9052-1)":
        "Banco de resonancia de rigidez dinámica (ISO 9052-1)",
    "Resonance rig": "Banco de resonancia",
    "Rigid foundation": "Base rígida",
    "Load plate": "Placa de carga",
    "Resilient specimen": "Probeta resiliente",
    "Exciter": "Excitador",
    "Accelerometer": "Acelerómetro",
    "Mass-spring model": "Modelo masa-resorte",
    "resonance read from the response peak":
        "la resonancia se lee del pico de la respuesta",
    "s′t = 4π² m′t fr²   (Formula 4)": "s′t = 4π² m′t fr²   (Fórmula 4)",
    "then f₀ = (1/2π)·√(s′/m′) for the installed floating floor   (Formula 2)":
        "luego f₀ = (1/2π)·√(s′/m′) para el suelo flotante instalado   "
        "(Fórmula 2)",
    # Mechanical-mobility rig (ISO 7626)
    "Mechanical-mobility measurement on a beam (ISO 7626)":
        "Medición de movilidad mecánica sobre una viga (ISO 7626)",
    "soft elastic suspension": "suspensión elástica blanda",
    "Structure under test (free-free beam)":
        "Estructura bajo ensayo (viga libre-libre)",
    "Impedance head": "Cabeza de impedancia",
    "F and a at the drive point": "F y a en el mismo punto",
    "driving point:  Yii = vi / Fi": "punto de excitación:  Yii = vi / Fi",
    "transfer:  Yji = vj / Fi": "transferencia:  Yji = vj / Fi",
    "Y(f) = v/F  [m/(N·s)] · attached exciter (Part 2) · impact hammer (Part 5)":
        "Y(f) = v/F  [m/(N·s)] · excitador acoplado (Parte 2) · martillo de "
        "impacto (Parte 5)",
    "same measurement, three FRFs: x/F receptance · v/F mobility · a/F accelerance":
        "una misma medición, tres FRF: x/F receptancia · v/F movilidad · "
        "a/F acelerancia",
    # Dynamic transfer stiffness (ISO 10846)
    "Dynamic transfer stiffness: direct and indirect methods (ISO 10846)":
        "Rigidez de transferencia: métodos directo e indirecto (ISO 10846)",
    "Direct method (Part 2)": "Método directo (Parte 2)",
    "Indirect method (Part 3)": "Método indirecto (Parte 3)",
    "excitation mass": "masa de excitación",
    "isolator under test": "aislador bajo ensayo",
    "force transducer": "transductor de fuerza",
    "blocking mass m₂": "masa de bloqueo m₂",
    "soft support": "apoyo blando",
    "output blocked:  u₂ ≈ 0 → measure F₂,b":
        "salida bloqueada:  u₂ ≈ 0 → se mide F₂,b",
    "measure T = u₂ / u₁  (small)": "se mide T = u₂ / u₁  (pequeña)",
    "valid where ΔL₁,₂ = La₁ − La₂ ≥ 20 dB, i.e. |T| ≤ 0.1   (Part 3, Inequality 2)":
        "válido donde ΔL₁,₂ = La₁ − La₂ ≥ 20 dB, es decir |T| ≤ 0,1   "
        "(Parte 3, Desigualdad 2)",
    "the blocking force approximates the force delivered to a stiff receiver (Part 1, Eq. 7)":
        "la fuerza de bloqueo aproxima la fuerza entregada a un receptor "
        "rígido (Parte 1, Ec. 7)",
    # Reception plate (EN 15657)
    "Reception-plate measurement of structure-borne power (EN 15657)":
        "Medición en placa receptora de la potencia estructural (EN 15657)",
    "Source under test (pump, fan, boiler …)":
        "Fuente bajo ensayo (bomba, ventilador, caldera …)",
    "Reception plate  (m, S, η)": "Placa receptora  (m, S, η)",
    "velocity positions → Lv": "posiciones de velocidad → Lv",
    "injected structure-borne power": "potencia estructural inyectada",
    "resilient supports": "apoyos resilientes",
    "Plate power balance": "Balance de potencia de la placa",
    "η = 2.2 / (f·Ts)   (Formula 13)": "η = 2,2 / (f·Ts)   (Fórmula 13)",
    "+ Lv − 60   (Formula 14)": "+ Lv − 60   (Fórmula 14)",
    "→ source quantities (Formulae 15–19):":
        "→ magnitudes de fuente (Fórmulas 15–19):",
    "equivalent blocked force L_Fb,eq ,": "fuerza bloqueada equivalente L_Fb,eq ,",
    "L_Wsn consumed by EN 12354-5": "L_Wsn que consume EN 12354-5",
    "spatial average:  Lv = 10 lg[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)":
        "promedio espacial:  Lv = 10 lg[(1/N)·Σ 10^(Lv,i/10)]   (Fórmula 12)",
    # Installed structure-borne sound (EN 12354-5)
    "Installed structure-borne sound paths (EN 12354-5)":
        "Vías del sonido estructural de equipos instalados (EN 12354-5)",
    "Service equipment (pump)": "Equipo de servicio (bomba)",
    "coupling D_C   (Formula 19b)": "acoplamiento D_C   (Fórmula 19b)",
    "path along the slab into the wall  (i → j)":
        "vía por el forjado hacia la pared  (i → j)",
    "excited floor radiates (path i = j)":
        "el forjado excitado radia (vía i = j)",
    "Prediction cascade": "Cascada de predicción",
    "characteristic power (EN 15657)": "potencia característica (EN 15657)",
    "coupling at the contacts (19b)": "acoplamiento en los contactos (19b)",
    "installed power (18b)": "potencia instalada (18b)",
    "per transmission path (18a)": "por vía de transmisión (18a)",
    "energetic sum L_n,s (17)": "suma energética L_n,s (17)",
    "each path i → j: excited element i, radiating element j in the receiving room":
        "cada vía i → j: elemento excitado i, elemento radiante j en el "
        "recinto receptor",
    # Wind-turbine noise measurement geometry (IEC 61400-11)
    "Wind-turbine noise measurement geometry (IEC 61400-11)":
        "Geometría de medida del ruido de aerogenerador (IEC 61400-11)",
    "Wind": "Viento",
    "rotor centre": "centro del rotor",
    "Microphone on a ground board": "Micrófono sobre placa en el suelo",
    "Met mast": "Mástil meteorológico",
    "wind speed + direction": "velocidad y dirección del viento",
    "Plan view (Figure 3)": "Planta (Figura 3)",
    "reference position 1 (downwind)": "referencia 1 (a sotavento)",
    "optional positions 2–4": "posiciones opcionales 2–4",
    "R1 = √(H² + R0²)   slant distance, rotor centre → microphone":
        "R1 = √(H² + R0²)   distancia oblicua, centro del rotor → micrófono",
    "LWA,i = Lp,i − 6 + 10 lg(4π R1² / S0)   (Formula 26, S0 = 1 m²)":
        "LWA,i = Lp,i − 6 + 10 lg(4π R1² / S0)   (Fórmula 26, S0 = 1 m²)",
    "the −6 dB removes the board's pressure doubling; board-to-R1 angle φ = 25°–40°":
        "los −6 dB descuentan la duplicación de presión de la placa; "
        "ángulo placa–R1 φ = 25°–40°",
    # Ground reflection (image source)
    "Ground reflection: direct ray, image source and path difference":
        "Reflexión del suelo: rayo directo, fuente imagen y diferencia de camino",
    "image source": "fuente imagen",
    "direct ray  r1": "rayo directo  r1",
    "reflected ray": "rayo reflejado",
    "equal angles": "ángulos iguales",
    "path difference  δ = r2 − r1": "diferencia de camino  δ = r2 − r1",
    "phase difference  Δφ = 2π δ / λ  (+ arg Q)":
        "diferencia de fase  Δφ = 2π δ / λ  (+ arg Q)",
    "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = ground reflection coefficient)":
        "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = coeficiente de reflexión del suelo)",
    "in phase (δ ≈ nλ): up to +6 dB    ·    out of phase (δ ≈ λ/2 on hard ground): a deep dip":
        "en fase (δ ≈ nλ): hasta +6 dB    ·    en oposición (δ ≈ λ/2 sobre "
        "suelo duro): un mínimo profundo",
    # 2D FDTD wave simulation
    "Domain  c(x, y), ρ(x, y), dx": "Dominio  c(x, y), ρ(x, y), dx",
    "square cells; dt from the Courant number":
        "celdas cuadradas; dt desde el número de Courant",
    "Geometry and boundaries": "Geometría y contornos",
    "rigid, impedance or absorbing edges; obstacles":
        "bordes rígidos, de impedancia o absorbentes; obstáculos",
    "Sources  s(t) injected at cells  (Eq. 4.11-4.12 grid)":
        "Fuentes  s(t) inyectadas en celdas  (malla de Ec. 4.11-4.12)",
    "Gaussian pulse, ramped tone or arbitrary sampled signal":
        "pulso gaussiano, tono con rampa o señal muestreada arbitraria",
    "Staggered-grid leapfrog update  (Eqs. 4.11-4.12)":
        "Actualización leapfrog en malla escalonada  (Ecs. 4.11-4.12)",
    "v ← v − (dt/ρ·dx)·grad p,  then  p ← p − (ρc²·dt/dx)·div v":
        "v ← v − (dt/ρ·dx)·grad p,  y luego  p ← p − (ρc²·dt/dx)·div v",
    "stable while  CN = c·dt·√2/dx ≤ 1  (Eqs. 4.13-4.14)":
        "estable mientras  CN = c·dt·√2/dx ≤ 1  (Ecs. 4.13-4.14)",
    "resolve ≥ 10 cells per wavelength to keep dispersion low":
        "resolver ≥ 10 celdas por longitud de onda para baja dispersión",
    "2D acoustic FDTD wave simulation (staggered leapfrog)":
        "Simulación de ondas FDTD acústica 2D (leapfrog escalonado)",
    "FDTDResult:  probe histories p(t), field snapshots, .plot()":
        "FDTDResult:  historias de sonda p(t), instantáneas del campo, "
        ".plot()",
    "deterministic: same inputs, bit-identical outputs":
        "determinista: mismas entradas, salidas idénticas bit a bit",
    # Sound level meter chain (IEC 61672-1)
    "Sound level meter measurement chain (IEC 61672-1)":
        "Cadena de medición del sonómetro (IEC 61672-1)",
    "Sound calibrator (class 1)": "Calibrador acústico (clase 1)",
    "coupled for": "acoplado para",
    "the level check": "verificar el nivel",
    "Windscreen": "Pantalla antiviento",
    "Microphone + preamplifier": "Micrófono + preamplificador",
    "free-field capsule, high-impedance stage":
        "cápsula de campo libre, etapa de alta impedancia",
    "Frequency weighting  A / C / Z": "Ponderación frecuencial  A / C / Z",
    "all three are 0 dB at 1 kHz; class 1: ±0.7 dB":
        "las tres valen 0 dB a 1 kHz; clase 1: ±0,7 dB",
    "Squaring + time weighting  F / S":
        "Cuadrado + ponderación temporal  F / S",
    "exponential detector: τF = 125 ms, τS = 1 s":
        "detector exponencial: τF = 125 ms, τS = 1 s",
    "Display": "Pantalla",
    "LAF(t), LAS(t) in dB re 20 µPa": "LAF(t), LAS(t) en dB re 20 µPa",
    # Laboratory sound insulation suite (ISO 10140)
    "Laboratory sound insulation suite (ISO 10140)":
        "Cámaras de aislamiento acústico de laboratorio (ISO 10140)",
    "structural break": "junta estructural",
    "Test element in the test opening":
        "Elemento de ensayo en la abertura de ensayo",
    "moving microphone": "micrófono móvil",
    "sweep radius ≥ 1 m": "radio de barrido ≥ 1 m",
    "Test opening ≈ 10 m² (3.75 m × 2.7 m); shorter edge ≥ 2.3 m":
        "Abertura de ensayo ≈ 10 m² (3,75 m × 2,7 m); lado menor ≥ 2,3 m",
    "Room volumes ≥ 50 m³, differing by at least 10 %":
        "Volúmenes de sala ≥ 50 m³, con diferencia de al menos el 10 %",
    "Continuously moving microphone: sweep radius ≥ 1 m, traverse ≥ 15 s":
        "Micrófono en movimiento continuo: radio de barrido ≥ 1 m, "
        "recorrido ≥ 15 s",
    "5.0 m": "5,0 m",
    "4.6 m": "4,6 m",
    "4.1 m": "4,1 m",
    "3.75 m": "3,75 m",
    # Junction vibration measurement (ISO 10848)
    "Junction vibration measurement on L- and T-junctions (ISO 10848)":
        "Medición de vibración en uniones en L y en T (ISO 10848)",
    "L-junction": "Unión en L",
    "T-junction": "Unión en T",
    "Shaker or hammer on element i": "Excitador o martillo sobre el elemento i",
    "accelerometers on i and j": "acelerómetros en i y j",
    "lij ≥ 2.3 m": "lij ≥ 2,3 m",
    "concrete plates 140 mm to 200 mm thick":
        "placas de hormigón de 140 mm a 200 mm de espesor",
    "lij ≥ 2.3 m along the junction; element sizes 3.0 m ≤ li < 6.0 m":
        "lij ≥ 2,3 m a lo largo de la unión; dimensiones de elemento "
        "3,0 m ≤ li < 6,0 m",
    "≥ 4 excitation positions on i; accelerometers ≥ 0.25 m from edges, ≥ 0.5 m apart":
        "≥ 4 posiciones de excitación en i; acelerómetros a ≥ 0,25 m de los "
        "bordes y ≥ 0,5 m entre sí",
    "Kij = D̄v,ij + 10 lg( lij / √(ai·aj) ),   ai = equivalent absorption length":
        "Kij = D̄v,ij + 10 lg( lij / √(ai·aj) ),   ai = long. de "
        "absorción equiv.",
    # Sound power from surface vibration (ISO/TS 7849)
    "Sound power from surface vibration (ISO/TS 7849)":
        "Potencia acústica a partir de la vibración superficial (ISO/TS 7849)",
    "Vibrating measurement surface S": "Superficie de medición vibrante S",
    "Machine under test": "Máquina en ensayo",
    "radiated airborne sound": "sonido aéreo radiado",
    "Number of positions N": "Número de posiciones N",
    "one accelerometer per cell of area S/N":
        "un acelerómetro por celda de área S/N",
    "Survey sound power": "Potencia acústica de control",
    "ε = 1 assumed → upper limit LWA,max":
        "se asume ε = 1 → límite superior LWA,max",
    "normal surface velocity, A-weighted r.m.s.":
        "velocidad normal eficaz, ponderada A",
    "2.5 m": "2,5 m",
    "1.6 m": "1,6 m",
    # Ship radiated-noise measurement geometry (ISO 17208-1)
    "Ship radiated-noise measurement geometry (ISO 17208-1)":
        "Geometría de medición del ruido radiado por buques (ISO 17208-1)",
    "Ship under test": "Buque en ensayo",
    "Surface buoy": "Boya de superficie",
    "vertical array of 3 hydrophones": "array vertical de 3 hidrófonos",
    "ballast": "lastre",
    "sea floor": "fondo marino",
    "Plan view": "Vista en planta",
    "course": "rumbo",
    "data window": "ventana de datos",
    "dCPA ≥ 100 m (or 1·L)": "dCPA ≥ 100 m (o 1·L)",
    "water depth ≥ 150 m (or 1.5·L)": "profundidad ≥ 150 m (o 1,5·L)",
    "Four runs, two per side; levels averaged while the ship crosses the data window":
        "Cuatro pasadas, dos por banda; niveles promediados mientras el buque "
        "cruza la ventana de datos",
    "Hydrophone depths from the 15°, 30° and 45° depression angles at r = dCPA; L = ship length":
        "Profundidades de hidrófono según los ángulos de depresión de 15°, "
        "30° y 45° a r = dCPA; L = eslora",
    # SOFAR channel (deep sound channel)
    "The SOFAR channel: a deep-ocean sound waveguide":
        "El canal SOFAR: una guía de ondas del océano profundo",
    "Sound-speed profile c(z)": "Perfil de velocidad del sonido c(z)",
    "Ray paths near the axis": "Trayectorias de rayos cerca del eje",
    "sea surface": "superficie del mar",
    "source on the channel axis": "fuente en el eje del canal",
    "rays that stay in the channel meet no surface or bottom loss":
        "los rayos que permanecen en el canal no sufren pérdidas en "
        "superficie ni en fondo",
    "c rises toward the surface (temperature) and toward the bottom (pressure); the minimum traps sound":
        "c aumenta hacia la superficie (temperatura) y hacia el fondo "
        "(presión); el mínimo atrapa el sonido",
    "rays launched within about ±12° of the axis stay trapped and can cross entire oceans":
        "los rayos lanzados a menos de unos ±12° del eje quedan atrapados "
        "y pueden cruzar océanos enteros",
}


class SVG:
    """Tiny element accumulator with technical-drawing helpers."""

    def __init__(self, width: int, height: int, th: Theme, lang: str = "en") -> None:
        self.w, self.h, self.th = width, height, th
        self.lang = lang
        self.parts: list[str] = []

    def tr(self, s: str) -> str:
        """Translate a user-visible string for the current language."""
        return _ES.get(s, s) if self.lang == "es" else s

    # -- primitives -------------------------------------------------------
    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def rect(self, x: float, y: float, w: float, h: float, fill: str,
             stroke: str = "none", rx: float = 0.0, sw: float = 1.5,
             dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             sw: float = 1.5, dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="{stroke}" stroke-width="{sw}"{d} stroke-linecap="round"/>')

    def circle(self, cx: float, cy: float, r: float, fill: str,
               stroke: str = "none", sw: float = 1.5) -> None:
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def ellipse(self, cx: float, cy: float, rx: float, ry: float,
                fill: str = "none", stroke: str = "none", sw: float = 1.5,
                dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def text(self, x: float, y: float, s: str, size: int = 20,
             fill: str = "", anchor: str = "middle", bold: bool = False,
             mono: bool = False, italic: bool = False) -> None:
        s = self.tr(s)
        # Escape XML metacharacters so labels may contain <, > and & literally.
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        fill = fill or self.th.fg
        w = ' font-weight="600"' if bold else ""
        i = ' font-style="italic"' if italic else ""
        fam = _MONO if mono else _FONT
        self.add(f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{size}" '
                 f'fill="{fill}" text-anchor="{anchor}"{w}{i}>{s}</text>')

    def path(self, d: str, fill: str = "none", stroke: str = "none",
             sw: float = 1.5, dash: str = "") -> None:
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw}" stroke-linejoin="round"{dd}/>')

    # -- technical helpers -------------------------------------------------
    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
              sw: float = 1.6) -> None:
        """Straight arrow with a filled head at (x2, y2)."""
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        L, W = 9.0, 3.6
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        px, py = -math.sin(ang), math.cos(ang)
        self.line(x1, y1, bx, by, stroke, sw)
        self.path(f"M {x2:.1f} {y2:.1f} L {bx + W * px:.1f} {by + W * py:.1f} "
                  f"L {bx - W * px:.1f} {by - W * py:.1f} Z", fill=stroke)

    def dim(self, x1: float, y1: float, x2: float, y2: float, label: str,
            offset: float = 0.0, size: int = 18, label_side: str = "left") -> None:
        """Dimension between two measured points, drafting style.

        The dimension line is placed ``offset`` px away (perpendicular);
        dashed witness lines connect it to the measured points. With
        ``offset=0`` the caller is responsible for any witness lines.
        """
        th = self.th
        horizontal = abs(y2 - y1) < abs(x2 - x1)
        if horizontal:
            y = y1 + offset
            if offset:
                self.line(x1, y1, x1, y, th.muted, 0.9, dash="3,3")
                self.line(x2, y2, x2, y, th.muted, 0.9, dash="3,3")
            mid = (x1 + x2) / 2
            self.arrow(mid - 4, y, x1, y, th.muted, 1.2)
            self.arrow(mid + 4, y, x2, y, th.muted, 1.2)
            self.text(mid, y - 7, label, size, th.fg, "middle")
        else:
            x = x1 + offset
            if offset:
                self.line(x1, y1, x, y1, th.muted, 0.9, dash="3,3")
                self.line(x2, y2, x, y2, th.muted, 0.9, dash="3,3")
            mid = (y1 + y2) / 2
            self.arrow(x, mid - 4, x, y1, th.muted, 1.2)
            self.arrow(x, mid + 4, x, y2, th.muted, 1.2)
            # Label beside the line, on whichever side is clear of the
            # measured object (masts, people, furniture).
            if label_side == "right":
                self.text(x + 9, mid + 6, label, size, th.fg, "start")
            else:
                self.text(x - 9, mid + 6, label, size, th.fg, "end")

    def mic(self, x: float, capsule_top: float, ground: float,
            scale: float = 1.0) -> None:
        """Measurement microphone on a stand that reaches the ground.

        ``capsule_top`` is the y of the capsule tip (the measurement point).
        """
        th, s = self.th, scale
        cap_h, body_h = 12 * s, 34 * s
        self.rect(x - 4 * s, capsule_top, 8 * s, cap_h, th.fg, rx=2.5 * s)
        self.rect(x - 6 * s, capsule_top + cap_h, 12 * s, body_h, th.primary, rx=4 * s)
        self.line(x, capsule_top + cap_h + body_h, x, ground, th.fg, 2.2)
        self.line(x - 16 * s, ground, x + 16 * s, ground, th.fg, 2.2)

    def person(self, x: float, y: float, h: float = 90.0, seated: bool = False) -> None:
        """Simple engineering-style human silhouette; (x, y) = feet."""
        th = self.th
        r = h * 0.10
        if not seated:
            self.circle(x, y - h + r, r, th.muted)
            self.line(x, y - h + 2 * r, x, y - h * 0.35, th.muted, 3)
            self.line(x, y - h * 0.75, x - h * 0.18, y - h * 0.5, th.muted, 2.4)
            self.line(x, y - h * 0.75, x + h * 0.18, y - h * 0.5, th.muted, 2.4)
            self.line(x, y - h * 0.35, x - h * 0.13, y, th.muted, 2.4)
            self.line(x, y - h * 0.35, x + h * 0.13, y, th.muted, 2.4)
        else:
            self.circle(x, y - h + r, r, th.muted)
            self.line(x, y - h + 2 * r, x, y - h * 0.45, th.muted, 3)       # torso
            self.line(x, y - h * 0.45, x + h * 0.30, y - h * 0.45, th.muted, 2.4)  # thigh
            self.line(x + h * 0.30, y - h * 0.45, x + h * 0.30, y, th.muted, 2.4)  # shin
            self.line(x, y - h * 0.70, x + h * 0.22, y - h * 0.55, th.muted, 2.4)  # arm

    def ground(self, y: float, x1: float, x2: float, hatch: int = 24) -> None:
        th = self.th
        self.line(x1, y, x2, y, th.fg, 2.2)
        x = x1
        while x < x2:
            self.line(x, y, x - 8, y + 9, th.muted, 1.1)
            x += hatch

    def render(self, title: str) -> str:
        th = self.th
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                f'<rect width="{self.w}" height="{self.h}" fill="{th.bg}"/>'
                f'<text x="{self.w / 2}" y="30" font-family="{_FONT}" '
                f'font-size="26" font-weight="600" fill="{th.fg}" '
                f'text-anchor="middle">{self.tr(title)}</text>')
        return head + "".join(self.parts) + "</svg>"


def _write(output_dir: str, name: str, build: Callable[[SVG, Theme], None], title: str,
           height: int = 560) -> None:
    for lang, lang_suffix in (("en", ""), ("es", "_es")):
        for th in (LIGHT, DARK):
            svg = SVG(900, height, th, lang)
            build(svg, th)
            path = os.path.join(output_dir, f"{name}{lang_suffix}{th.suffix}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg.render(title))
    print(f"Generated {name}.svg (+dark, +es, +es_dark)")


# ---------------------------------------------------------------------------
# d1 - Calibration chain (IEC 60942)
# ---------------------------------------------------------------------------

def _d1(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # Calibrator on top of the microphone (left column)
    mx = 150.0
    cal_y = 110.0
    s.text(mx, cal_y - 22, "Sound calibrator", 22, th.fg, bold=True)
    s.rect(mx - 62, cal_y, 124, 86, th.panel, th.fg, rx=10, sw=2)
    s.text(mx, cal_y + 38, "94.0 dB", 26, th.secondary, bold=True, mono=True)
    s.text(mx, cal_y + 66, "1 kHz", 20, th.muted, mono=True)
    s.rect(mx - 15, cal_y + 86, 30, 12, th.fg, rx=3)   # coupler cavity
    s.mic(mx, cal_y + 98, gy, 1.3)

    # Signal chain
    boxes = [(400, "Microphone +", "preamplifier"), (650, "Audio interface", "(ADC)")]
    by, bw, bh = 176.0, 210.0, 78.0
    prev_x = mx + 62
    for bx, l1, l2 in boxes:
        s.rect(bx - bw / 2, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(bx, by + 33, l1, 22, th.fg, bold=True)
        s.text(bx, by + 60, l2, 22, th.fg, bold=True)
        s.arrow(prev_x, by + bh / 2, bx - bw / 2 - 6, by + bh / 2, th.fg, 2)
        prev_x = bx + bw / 2 + 6
    s.arrow(prev_x, by + bh / 2, 862, by + bh / 2, th.accent, 2.4)
    s.text(796, by + bh / 2 + 34, "Pa per", 20, th.accent, mono=True)
    s.text(796, by + bh / 2 + 58, "digital unit", 20, th.accent, mono=True)

    # Stability annotation, clearly separated below the chain
    s.rect(250, 340, 560, 96, "none", th.secondary, rx=12, dash="6,5")
    s.text(530, 376, "Stability: |max − mean| and |min − mean| ≤ 0.07 dB", 22, th.secondary, bold=True)
    s.text(530, 408, "(IEC 60942:2017 Table 2, class 1) — else CalibrationWarning", 20, th.fg)


# ---------------------------------------------------------------------------
# d2 - Environmental noise microphone positions (ISO 1996-2)
# ---------------------------------------------------------------------------

def _d2(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # Building facade (right)
    fx = 700.0
    s.rect(fx, 120, 160, gy - 120, th.panel, th.fg, sw=2)
    for wy in range(158, int(gy) - 50, 78):
        s.rect(fx + 24, wy, 38, 46, th.bg, th.muted, rx=3, sw=1.2)
        s.rect(fx + 96, wy, 38, 46, th.bg, th.muted, rx=3, sw=1.2)
    s.text(fx + 80, 104, "Building façade", 22, th.fg, bold=True)

    # Source (left): car on a road
    s.rect(60, gy - 9, 140, 9, th.muted)
    s.path(f"M 88 {gy - 30} L 106 {gy - 48} L 146 {gy - 48} L 164 {gy - 30} Z", fill=th.secondary)
    s.rect(80, gy - 32, 96, 14, th.secondary, rx=5)
    s.circle(102, gy - 13, 9, th.fg)
    s.circle(156, gy - 13, 9, th.fg)
    for r in (44, 76, 108):
        s.path(f"M {168 + r * 0.5} {gy - 34 - r * 0.55} "
               f"A {r} {r} 0 0 1 {168 + r * 0.87} {gy - 34 + r * 0.1}",
               stroke=th.accent, sw=1.6)

    # Position A: free field, capsule 4 m above ground
    ax = 330.0
    a_cap = gy - 230.0
    s.mic(ax, a_cap, gy, 1.15)
    s.dim(ax, gy, ax, a_cap, "4.0 ± 0.2 m", offset=-60, size=20)
    s.text(ax - 20, a_cap - 58, "A — free field", 22, th.fg, bold=True)
    s.text(ax - 20, a_cap - 30, "0 dB", 22, th.accent, bold=True, mono=True)

    # Position B: 2 m in front of the facade, dimension at capsule height
    bx = fx - 108.0
    b_cap = gy - 230.0
    s.mic(bx, b_cap, gy, 1.15)
    s.dim(bx, b_cap + 6, fx, b_cap + 6, "2 m", offset=-14, size=20)
    s.text(bx - 30, b_cap - 58, "B — 2 m from façade", 22, th.fg, bold=True)
    s.text(bx - 30, b_cap - 30, "−3 dB", 22, th.secondary, bold=True, mono=True)

    # Position C: flush-mounted on the facade, below B's dimension zone
    cy = gy - 120.0
    s.circle(fx + 3, cy, 7, th.fg)
    # The leader crosses mic B's mast (plain line crossing, standard
    # drafting); the label itself sits in the clear zone between masts.
    s.line(fx - 2, cy + 5, 470, cy + 60, th.muted, 1.4)
    s.text(462, cy + 84, "C — flush-mounted", 22, th.fg, bold=True)
    s.text(462, cy + 110, "−6 dB", 22, th.secondary, bold=True, mono=True)


# ---------------------------------------------------------------------------
# d3 - Operator / bystander microphone positions (ECMA-74, clause 8.6)
# ---------------------------------------------------------------------------

def _d3(s: SVG, th: Theme) -> None:
    gy = 470.0
    s.ground(gy, 40, 860)

    # --- Left: seated operator at table-top equipment (side view) ---------
    s.text(240, 72, "Operator — seated (P2)", 24, th.fg, bold=True)
    tx = 80.0
    table_y = gy - 150.0
    s.line(tx + 18, gy, tx + 18, table_y, th.fg, 3)
    s.line(tx + 232, gy, tx + 232, table_y, th.fg, 3)
    s.line(tx, table_y, tx + 250, table_y, th.fg, 4)
    s.rect(tx + 16, table_y - 76, 118, 76, th.panel, th.primary, rx=8, sw=2)
    s.text(tx + 75, table_y - 32, "EUT", 22, th.primary, bold=True)
    eut_front = tx + 134.0

    # microphone: capsule tip at 1.20 m, 0.25 m from the EUT front face
    mx = eut_front + 76.0
    cap = gy - 268.0
    s.mic(mx, cap, table_y, 1.1)
    s.line(mx - 18, table_y, mx + 18, table_y, th.fg, 2.2)
    s.dim(eut_front, table_y - 76, mx, cap, "0.25 m", offset=-36, size=20)
    s.dim(mx + 210, gy, mx + 210, cap, "1.20 m", offset=0, size=20, label_side="right")
    s.line(mx + 10, cap, mx + 210, cap, th.muted, 0.9, dash="3,3")  # witness to capsule

    # seated operator on a chair, clear of both dimensions
    px = mx + 120.0
    seat_y = gy - 115.0
    s.line(px - 28, seat_y, px + 32, seat_y, th.muted, 3)
    s.line(px - 24, seat_y, px - 24, gy, th.muted, 2.6)
    s.line(px + 28, seat_y, px + 28, gy, th.muted, 2.6)
    s.line(px + 32, seat_y, px + 32, seat_y - 86, th.muted, 2.6)
    s.circle(px, gy - 240, 15, th.muted)
    s.line(px, gy - 225, px + 6, seat_y, th.muted, 3.4)
    s.line(px + 6, seat_y, px - 34, seat_y - 2, th.muted, 2.8)
    s.line(px - 34, seat_y - 2, px - 34, gy, th.muted, 2.8)
    s.line(px - 1, gy - 205, px - 38, gy - 178, th.muted, 2.6)

    # --- Right: bystander positions (top view), equal face distances ------
    cx, cyv = 700.0, 270.0
    s.text(cx, 72, "Bystanders — top view", 24, th.fg, bold=True)
    s.text(cx, 100, "height 1.50 m", 20, th.muted)
    s.rect(cx - 52, cyv - 40, 104, 80, th.panel, th.primary, rx=8, sw=2)
    s.text(cx, cyv + 8, "EUT", 22, th.primary, bold=True)
    g = 92.0  # face-to-microphone distance, equal on all four sides
    for pxx, pyy in [(cx, cyv - 40 - g), (cx, cyv + 40 + g),
                     (cx - 52 - g, cyv), (cx + 52 + g, cyv)]:
        s.circle(pxx, pyy, 8, th.secondary)
        s.circle(pxx, pyy, 2.8, th.bg)
    s.dim(cx + 52, cyv - 20, cx + 52 + g, cyv, "1.00 m", offset=-44, size=20)


# ---------------------------------------------------------------------------
# d4 - Library signal chain
# ---------------------------------------------------------------------------

def _d4(s: SVG, th: Theme) -> None:
    stages = [
        ("Signal", "x, fs", th.fg),
        ("Calibrate", "→ Pa", th.primary),
        ("Weighting", "A/C/G/Z", th.primary),
        ("Octave", "bands 1/b", th.primary),
        ("Ballistics", "F / S / I", th.primary),
        ("Metrics", "Leq, LN…", th.accent),
    ]
    bw, bh, gap = 136.0, 92.0, 12.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 170.0
    for i, (title, sub, color) in enumerate(stages):
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        s.text(x + bw / 2, y + 40, title, 22, th.fg, bold=True)
        s.text(x + bw / 2, y + 68, sub, 19, color, mono=True)
        if i < len(stages) - 1:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap




# ---------------------------------------------------------------------------
# d5 - Multirate decimation inside the filter bank
# ---------------------------------------------------------------------------

def _d5(s: SVG, th: Theme) -> None:
    # Input on the left
    s.rect(36, 150, 136, 70, th.panel, th.fg, rx=10, sw=2)
    s.text(104, 180, "Signal", 22, th.fg, bold=True)
    s.text(104, 205, "fs = 48 kHz", 18, th.muted, mono=True)

    rows = [
        (120.0, "16 kHz band", "fs", "no decimation", th.secondary),
        (230.0, "1 kHz band", "fs / 8", "6 kHz", th.primary),
        (340.0, "63 Hz band", "fs / 64", "750 Hz", th.accent),
    ]
    for y, band, rate, eff, color in rows:
        bx = 455.0
        if "no" not in eff:
            s.arrow(172, 185, 240, y + 35, th.fg, 1.6)
            s.rect(250, y, 150, 70, th.panel, th.muted, rx=10, sw=1.6)
            s.text(325, y + 30, "Anti-alias", 20, th.fg)
            s.text(325, y + 54, "LPF + \u2193M", 18, th.muted, mono=True)
            s.arrow(400, y + 35, 448, y + 35, th.fg, 1.6)
        else:
            s.arrow(172, 185, 448, y + 35, th.fg, 1.6)
        s.rect(bx, y, 190, 70, th.panel, color, rx=10, sw=2)
        s.text(bx + 95, y + 30, band, 20, th.fg, bold=True)
        s.text(bx + 95, y + 54, f"SOS @ {rate}", 18, color, mono=True)
        s.text(660, y + 40, eff, 18, th.muted, "start", mono=True)

    s.text(450, 480, "Low bands are filtered at a decimated rate: the relative", 20, th.fg)
    s.text(450, 508, "bandwidth stays wide, so the SOS stays numerically healthy.", 20, th.fg)


# ---------------------------------------------------------------------------
# d6 - Two-microphone (p-p) intensity probe (IEC 61043)
# ---------------------------------------------------------------------------

def _d6(s: SVG, th: Theme) -> None:
    ay = 232.0  # probe axis height

    # Measurement axis / intensity direction (drawn first, under the probe)
    s.line(70, ay, 820, ay, th.accent, 1.4, dash="10,4,2,4")
    s.arrow(820, ay, 852, ay, th.accent, 1.8)
    s.text(450, 305, "measurement axis / intensity direction", 18, th.accent)

    # Two opposed capsules facing each other with a spacer between the tips
    for side in (-1, 1):
        # bodies: 180..320 and 580..720; capsules: 320..400 / 500..580;
        # tips (grilles): 400..414 / 486..500; gap 414..486 = Δr
        bx = 180.0 if side < 0 else 580.0
        cx = 320.0 if side < 0 else 500.0
        tx = 400.0 if side < 0 else 486.0
        s.rect(bx, ay - 28, 140, 56, th.panel, th.primary, rx=10, sw=2)
        s.rect(cx, ay - 20, 80, 40, th.fg, rx=4)
        s.rect(tx, ay - 16, 14, 32, th.muted, rx=2)
    s.rect(414, ay - 6, 72, 12, th.panel, th.muted, rx=4, sw=1.2)  # spacer

    s.text(360, ay - 38, "p1", 20, th.fg, mono=True, bold=True)
    s.text(540, ay - 38, "p2", 20, th.fg, mono=True, bold=True)

    # Δr dimension between the capsule tips, drafting style
    s.dim(414, ay - 16, 486, ay - 16, "Δr = 12 mm", offset=-66, size=18)

    # p-p estimator notes near the capsules
    s.text(280, 365, "u from the p2−p1 gradient", 19, th.muted, mono=True)
    s.text(620, 365, "p = (p1+p2)/2", 19, th.muted, mono=True)


# ---------------------------------------------------------------------------
# d7 - STI measurement chain (IEC 60268-16)
# ---------------------------------------------------------------------------

def _d7(s: SVG, th: Theme) -> None:
    stages = [
        ("Source", "STIPA signal", th.fg),
        ("Room", "reverberation + noise", th.secondary),
        ("Microphone", "", th.primary),
        ("Analysis", "MTF → TI → STI", th.accent),
    ]
    bw, bh, gap = 192.0, 96.0, 20.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 150.0
    for i, (title, sub, color) in enumerate(stages):
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        if sub:
            s.text(x + bw / 2, y + 42, title, 22, th.fg, bold=True)
            if "→" in sub:
                s.text(x + bw / 2, y + 70, sub, 18, color, mono=True)
            else:
                s.text(x + bw / 2, y + 70, sub, 18, color)
        else:
            s.text(x + bw / 2, y + bh / 2 + 7, title, 22, th.fg, bold=True)
        if i == 1:  # the room degrades the modulation transfer function
            cx = x + bw / 2
            s.line(cx, y + bh, cx, y + bh + 18, th.muted, 1.2, dash="3,3")
            s.text(cx, y + bh + 40, "m(F) drops", 18, th.muted, italic=True)
        if i < len(stages) - 1:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap


# ---------------------------------------------------------------------------
# d8 - Airborne sound insulation setup (ISO 16283-1)
# ---------------------------------------------------------------------------

def _d8(s: SVG, th: Theme) -> None:
    top, bot = 90.0, 470.0

    # Two rooms in plan view separated by the test partition.
    s.rect(70, top, 375, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(465, top, 365, bot - top, th.panel, th.fg, rx=6, sw=3)
    s.rect(445, top, 20, bot - top, th.secondary, th.fg, sw=2)  # partition (S)
    s.text(455, 80, "Test partition", 20, th.secondary, bold=True)

    s.text(90, top + 32, "Source room", 22, th.fg, bold=True, anchor="start")
    s.text(90, top + 58, "L₁", 20, th.muted, anchor="start")
    s.text(486, top + 32, "Receiving room", 22, th.fg, bold=True, anchor="start")
    s.text(486, top + 58, "L₂ , T", 20, th.muted, anchor="start")

    # Loudspeaker in a corner of the source room (bottom-left).
    lsx, lsy = 150.0, 405.0
    for r in (40, 66, 92):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.accent, sw=1.6)
    s.rect(lsx - 26, lsy - 30, 52, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 10, 12, th.primary)
    s.circle(lsx, lsy - 10, 5, th.bg)
    s.circle(lsx, lsy + 16, 7, th.primary)
    s.text(lsx, lsy + 52, "Loudspeaker", 20, th.fg, bold=True)

    # Microphone positions (five per room, in the central zone).
    src_mics = [(150, 315), (255, 250), (360, 300), (300, 360), (390, 205)]
    rec_mics = [(590, 160), (653, 160), (560, 290), (690, 380), (785, 300)]
    for mics in (src_mics, rec_mics):
        for mx, my in mics:
            s.circle(mx, my, 8, th.fg)
            s.circle(mx, my, 3, th.bg)
    s.text(268, 172, "microphone positions", 18, th.muted)
    s.text(636, 430, "microphone positions", 18, th.muted)

    # Normative minimum separations (ISO 16283-1, 7.6 and 7.2.2).
    s.dim(150, 395, 150, 317, "≥ 1.0 m", offset=-42, size=20)          # 7.6c
    s.dim(178, 405, 443, 405, "≥ 1.0 m", offset=0, size=20)            # 7.2.2
    s.dim(590, 160, 653, 160, "≥ 0.7 m", offset=42, size=20)           # 7.6a
    s.dim(785, 300, 830, 300, "≥ 0.5 m", offset=-46, size=20)          # 7.6b

    # Clause legend.
    for y, txt in (
        (505, "7.6 a) ≥ 0.7 m between microphone positions"),
        (531, "7.6 b) ≥ 0.5 m to room boundaries"),
        (557, "7.6 c) ≥ 1.0 m to the loudspeaker"),
        (583, "7.2.2 ≥ 1.0 m loudspeaker to separating partition"),
    ):
        s.text(80, y, txt, 18, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# d9 - ISO 18233 indirect impulse-response measurement chain
# ---------------------------------------------------------------------------

def _d9(s: SVG, th: Theme) -> None:
    bw, bh = 200.0, 96.0
    xs = (120.0, 350.0, 580.0)
    y1, y2 = 110.0, 300.0

    def box(x: float, y: float, title: str, subs: list[str], color: str,
            mono: bool) -> None:
        s.rect(x, y, bw, bh, th.panel, color, rx=12, sw=2)
        t_size = 20 if len(title) > 11 else 22
        if subs:
            s.text(x + bw / 2, y + 38, title, t_size, th.fg, bold=True)
            if len(subs) == 1:
                s.text(x + bw / 2, y + 66, subs[0], 18, color,
                       mono=mono, italic=mono)
            else:
                s.text(x + bw / 2, y + 62, subs[0], 18, color)
                s.text(x + bw / 2, y + 82, subs[1], 18, color)
        else:
            s.text(x + bw / 2, y + bh / 2 + 7, title, t_size, th.fg, bold=True)

    # Row 1 (left to right): the physical excitation path.
    box(xs[0], y1, "Excitation", ["ESS sweep / MLS"], th.primary, False)
    box(xs[1], y1, "Loudspeaker", [], th.fg, False)
    box(xs[2], y1, "Room", ["h(t)"], th.secondary, True)
    s.arrow(xs[0] + bw, y1 + bh / 2, xs[1] - 2, y1 + bh / 2, th.fg, 2)
    s.arrow(xs[1] + bw, y1 + bh / 2, xs[2] - 2, y1 + bh / 2, th.fg, 2)

    # Serpentine connector: the acoustic field couples Room -> Microphone.
    cx = xs[2] + bw / 2
    s.arrow(cx, y1 + bh, cx, y2 - 2, th.muted, 2)
    s.text(cx - 12, (y1 + bh + y2) / 2 + 5, "acoustic path", 18, th.muted,
           anchor="end", italic=True)

    # Row 2 (right to left): recover the impulse response by deconvolution.
    box(xs[2], y2, "Microphone", [], th.primary, False)
    box(xs[1], y2, "Deconvolution", ["correlation /", "inverse filter"],
        th.accent, False)
    box(xs[0], y2, "IR", ["ĥ(t)"], th.accent, True)
    s.arrow(xs[2], y2 + bh / 2, xs[1] + bw + 2, y2 + bh / 2, th.fg, 2)
    s.arrow(xs[1], y2 + bh / 2, xs[0] + bw + 2, y2 + bh / 2, th.fg, 2)

    s.text(450, 425,
           "The room response h(t) is recovered by deconvolving the "
           "microphone signal.", 18, th.fg)


def _box_solid(s: SVG, th: Theme, bx: float, gy: float, hw: float, dp: float,
               ht: float, stroke: str = "", fill: str = "") -> None:
    """Small oblique-projected box standing on the plane at ``(bx, gy)``.

    ``hw`` is the front half-width, ``ht`` the height, ``dp`` the depth
    (oblique offset). Draws the top, front and right visible faces.
    """
    stroke = stroke or th.primary
    fill = fill or th.panel
    dxo, dyo = dp * 0.72, dp * 0.55
    ftl, ftr = (bx - hw, gy - ht), (bx + hw, gy - ht)
    fbr = (bx + hw, gy)
    btl = (bx - hw + dxo, gy - ht - dyo)
    btr = (bx + hw + dxo, gy - ht - dyo)
    bbr = (bx + hw + dxo, gy - dyo)
    # top face (lighter) then right face (shaded) then front face
    s.path(f"M {ftl[0]} {ftl[1]} L {ftr[0]} {ftr[1]} L {btr[0]} {btr[1]} "
           f"L {btl[0]} {btl[1]} Z", fill=fill, stroke=stroke, sw=1.8)
    s.path(f"M {ftr[0]} {ftr[1]} L {fbr[0]} {fbr[1]} L {bbr[0]} {bbr[1]} "
           f"L {btr[0]} {btr[1]} Z", fill=th.panel, stroke=stroke, sw=1.8)
    s.rect(bx - hw, gy - ht, 2 * hw, ht, fill, stroke, sw=1.8)


def _box_wire(s: SVG, th: Theme, bx: float, gy: float, hw: float, dp: float,
              ht: float, color: str, dash: str = "7,5") -> None:
    """Dashed oblique wireframe box (measurement surface) on the plane."""
    dxo, dyo = dp * 0.72, dp * 0.55
    fbl, fbr = (bx - hw, gy), (bx + hw, gy)
    ftl, ftr = (bx - hw, gy - ht), (bx + hw, gy - ht)
    bbl = (bx - hw + dxo, gy - dyo)
    bbr = (bx + hw + dxo, gy - dyo)
    btl = (bx - hw + dxo, gy - ht - dyo)
    btr = (bx + hw + dxo, gy - ht - dyo)
    for a, b in ((fbl, fbr), (fbr, ftr), (ftr, ftl), (ftl, fbl),
                 (bbl, bbr), (bbr, btr), (btr, btl), (btl, bbl),
                 (fbl, bbl), (fbr, bbr), (ftl, btl), (ftr, btr)):
        s.line(a[0], a[1], b[0], b[1], color, 1.5, dash=dash)


# ---------------------------------------------------------------------------
# d10 - ISO 3744/3746 sound power measurement surfaces
# ---------------------------------------------------------------------------

def _d_surfaces(s: SVG, th: Theme) -> None:
    # ===== Left panel: hemispherical surface over a reflecting plane =====
    cx, gy, R = 235.0, 420.0, 150.0
    s.text(cx, 74, "Hemispherical surface", 22, th.fg, bold=True)

    # Reflecting plane (hatched line through the equator / footprint centre).
    s.ground(gy, 55, 430)
    s.text(70, gy + 34, "Reflecting plane", 17, th.muted, anchor="start")

    # Hemisphere: dashed footprint ellipse + solid dome silhouette.
    ky = 0.30
    s.ellipse(cx, gy, R, R * ky, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.primary, sw=2.4)

    # Source box at the centre O.
    _box_solid(s, th, cx, gy, 30, 24, 34)
    s.circle(cx, gy, 3.4, th.fg)

    # Ten key microphone positions (ISO 3744 Table B.1), oblique-projected.
    b1 = [(0.16, -0.96, 0.22), (0.78, -0.60, 0.20), (0.78, 0.55, 0.31),
          (0.16, 0.90, 0.41), (-0.83, 0.32, 0.45), (-0.83, -0.40, 0.38),
          (-0.26, -0.65, 0.71), (0.74, -0.07, 0.67), (-0.26, 0.50, 0.83),
          (0.10, -0.10, 0.99)]
    labelled = {1, 8, 10}
    pts = []
    for x, y, z in b1:
        px = cx + R * x + 42 * y
        py = gy - 34 * y - R * z
        pts.append((px, py))
    # radius r drawn to position 8 (a mid-height point on the surface).
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text((cx + r8[0]) / 2 + 10, (gy + r8[1]) / 2 + 4, "radius r ≥ 2 d₀",
           17, th.accent, anchor="start")
    for i, (px, py) in enumerate(pts, start=1):
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
        if i in labelled:
            s.text(px, py - 12, str(i), 16, th.fg, bold=True)
    s.text(cx, gy + 62, "10 key positions (Table B.1)", 17, th.muted)
    s.text(cx, gy + 86, "one plane · S = 2πr²", 18, th.primary, bold=True, mono=True)

    # ===== Right panel: parallelepiped measurement surface =====
    bx2, gy2 = 675.0, 420.0
    s.text(bx2, 74, "Parallelepiped surface", 22, th.fg, bold=True)
    s.ground(gy2, 500, 872)

    # Source box (solid) enclosed by the measurement box (dashed wireframe).
    _box_solid(s, th, bx2, gy2, 46, 40, 58)
    _box_wire(s, th, bx2, gy2, 96, 90, 108, th.accent)
    s.text(bx2, gy2 + 40, "Measurement surface", 17, th.muted)
    s.text(bx2, gy2 + 64, "one plane · S = 4(ab+bc+ca)", 18, th.accent,
           bold=True, mono=True)

    # Measurement distance d: vertical clearance between the source top face
    # and the enveloping measurement surface (labelled arrow + caption above).
    s.text(bx2, 208, "measurement distance d", 18, th.secondary, bold=True)
    s.dim(bx2, gy2 - 108, bx2, gy2 - 58, "d", offset=0, size=20,
          label_side="right")


# ---------------------------------------------------------------------------
# d11 - ISO 16283-2 impact sound insulation setup
# ---------------------------------------------------------------------------

def _d_impact(s: SVG, th: Theme) -> None:
    bx0, bx1 = 90.0, 620.0          # building left / right walls
    top = 82.0
    floor_top, floor_bot = 292.0, 316.0  # separating floor slab
    bot = 512.0                     # receiving-room floor

    # Building shell and the two stacked rooms.
    s.rect(bx0, top, bx1 - bx0, floor_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, floor_bot, bx1 - bx0, bot - floor_bot, th.panel, th.fg, sw=2.5)
    s.rect(bx0, floor_top, bx1 - bx0, floor_bot - floor_top, th.secondary,
           th.fg, sw=2)  # separating floor / ceiling
    s.text(bx0 + 16, top + 30, "Source room (upper)", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx0 + 16, bot - 16, "Receiving room (lower)", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx1 - 12, floor_top - 8, "Separating floor", 17, th.secondary,
           bold=True, anchor="end")

    # Tapping machine standing on the separating floor (five hammers).
    mx = bx0 + 165.0
    body_y = floor_top - 40.0
    s.rect(mx - 60, body_y, 120, 28, th.primary, th.fg, rx=5, sw=2)
    for hx in range(-40, 41, 20):
        s.line(mx + hx, body_y + 28, mx + hx, floor_top - 2, th.fg, 2.4)
        s.circle(mx + hx, floor_top - 2, 4.2, th.fg)
    s.line(mx - 54, body_y + 28, mx - 54, floor_top, th.fg, 2)   # legs
    s.line(mx + 54, body_y + 28, mx + 54, floor_top, th.fg, 2)
    s.text(mx, body_y - 12, "Tapping machine", 19, th.fg, bold=True)

    # Structure-borne path through the slab, radiated into the room below.
    s.arrow(mx, floor_bot + 2, mx, floor_bot + 42, th.secondary, 2.2)
    s.text(mx - 12, floor_bot + 30, "structure-borne impact", 15, th.secondary,
           anchor="end", italic=True)
    for r in (46, 74, 102):
        s.path(f"M {mx - r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f} "
               f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {floor_bot + 44 + r * 0.5:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(mx, bot - 44, "radiated impact sound", 15, th.accent, italic=True)

    # Microphone positions on the receiving-room floor.
    for off in (300, 400, 500):
        s.mic(bx0 + off, bot - 120, bot, 0.95)
    s.text(bx0 + 400, floor_bot + 42, "Microphone positions", 16, th.muted)

    # Normative relations (right column); no invented spacing dimensions.
    lx = 648.0
    s.text(lx, 118, "Impact sound insulation", 18, th.fg, bold=True,
           anchor="start")
    box_items = [
        (160, "L′nT = Li − 10 lg(T/T₀)", th.primary),
        (192, "L′n = Li + 10 lg(A/A₀)", th.primary),
        (224, "A = 0.16 V/T  (Sabine)", th.muted),
        (256, "T₀ = 0.5 s , A₀ = 10 m²", th.accent),
    ]
    for y, txt, col in box_items:
        s.text(lx, y, txt, 15, col, anchor="start", mono=True,
               bold=(col != th.muted))
    s.rect(lx - 10, 292, 236, 100, "none", th.muted, rx=10, dash="6,5")
    s.text(lx, 320, "Li = energy-averaged", 15, th.fg, anchor="start")
    s.text(lx, 342, "band level (Formula 10)", 15, th.fg, anchor="start")
    s.text(lx, 374, "ISO 717-2 → Ln,w , CI", 16, th.secondary, anchor="start",
           bold=True)


# ---------------------------------------------------------------------------
# d12 - Sound power methods comparison infographic
# ---------------------------------------------------------------------------

def _d_methods(s: SVG, th: Theme) -> None:
    cols = [
        ("ISO 3744 / 3746", "Free field over a reflecting plane",
         "Grade 2 / 3 (engineering / survey)",
         "Sound pressure · enveloping surface",
         "LW = L̄p + 10lg(S/S₀) − K1 − K2",
         "K2A ≤ 4 dB (3744) / ≤ 7 dB (3746)", th.primary, "hemi"),
        ("ISO 3741", "Reverberation test room",
         "Grade 1 (precision)",
         "Sound pressure · diffuse field",
         "LW ← L̄p , T , V",
         "V ≥ 200 m³ , qualified room", th.accent, "reverb"),
        ("ISO 9614-2", "In situ — any environment",
         "Grade 2 / 3 (engineering / survey)",
         "Sound intensity · scanning",
         "LW = 10lg |Σ IᵢSᵢ| / W₀",
         "no negative-power bands", th.secondary, "probe"),
    ]
    cw, gap = 270.0, 15.0
    x0 = (900 - (3 * cw + 2 * gap)) / 2
    ctop, cbot = 66.0, 540.0
    for i, (name, env, grade, method, formula, note, col, pic) in enumerate(cols):
        x = x0 + i * (cw + gap)
        cxc = x + cw / 2
        s.rect(x, ctop, cw, cbot - ctop, th.panel, col, rx=14, sw=2.4)
        s.rect(x, ctop, cw, 44, col, col, rx=14, sw=0)
        s.rect(x, ctop + 22, cw, 22, col, "none")  # square off header bottom
        s.text(cxc, ctop + 30, name, 22, th.bg, bold=True)

        # Mini-pictogram band.
        py = ctop + 120.0
        if pic == "hemi":
            R = 58.0
            s.ellipse(cxc, py + 30, R, R * 0.3, "none", th.muted, 1.2, dash="4,3")
            s.path(f"M {cxc - R} {py + 30} A {R} {R} 0 0 1 {cxc + R} {py + 30}",
                   stroke=col, sw=2.2)
            s.line(cxc - R, py + 30, cxc + R, py + 30, th.muted, 1.4)
            _box_solid(s, th, cxc, py + 30, 12, 10, 16, stroke=col)
            for ang in (35, 90, 145):
                import math
                a = math.radians(ang)
                s.circle(cxc + R * math.cos(a), py + 30 - R * math.sin(a), 4.5,
                         th.secondary)
        elif pic == "reverb":
            s.rect(cxc - 58, py - 26, 116, 84, "none", col, rx=6, sw=2.2)
            for k in range(3):
                yy = py - 12 + k * 22
                s.path(f"M {cxc - 44} {yy} q 12 -12 24 0 q 12 12 24 0 q 12 -12 24 0",
                       stroke=th.muted, sw=1.6)
            s.circle(cxc - 40, py + 44, 6, th.secondary)   # RSS / source
        else:  # probe scanning a surface
            s.rect(cxc - 56, py - 30, 112, 92, "none", col, rx=6, sw=2.0, )
            # serpentine scan path
            s.path(f"M {cxc - 44} {py - 16} L {cxc + 40} {py - 16} "
                   f"L {cxc + 40} {py + 4} L {cxc - 44} {py + 4} "
                   f"L {cxc - 44} {py + 24} L {cxc + 40} {py + 24}",
                   stroke=th.accent, sw=1.7)
            s.circle(cxc + 40, py + 24, 5, th.secondary)
            s.text(cxc, py + 54, "I⊥", 17, col, bold=True, mono=True)

        # Attribute rows.
        rows = [(py + 96, env, th.fg, False),
                (py + 128, grade, col, True),
                (py + 160, method, th.muted, False)]
        for yy, txt, cc, bold in rows:
            s.text(cxc, yy, txt, 14, cc, bold=bold)

        # Headline formula in a boxed footer.
        s.rect(x + 10, cbot - 96, cw - 20, 46, "none", col, rx=8, dash="5,4")
        s.text(cxc, cbot - 67, formula, 14, th.fg, bold=True, mono=True)
        s.text(cxc, cbot - 26, note, 14, th.muted)


# ---------------------------------------------------------------------------
# d13 - EN 12354 direct + flanking transmission paths across a junction
# ---------------------------------------------------------------------------

def _d_flanking(s: SVG, th: Theme) -> None:
    dark = bool(th.suffix)
    # Four legible path colours (green / blue / red / orange), independent of
    # the neutral structural fills so every path stands out in both themes.
    c_dd = th.accent
    c_ff = th.primary
    c_fd = th.secondary
    c_df = "#f0a94e" if dark else "#d9820e"

    room_top, room_bot = 96.0, 372.0
    slab_top, slab_bot = 372.0, 402.0
    slab_cy = (slab_top + slab_bot) / 2.0
    wall_l, wall_r, wx = 434.0, 466.0, 450.0
    wall_bot = 430.0                       # wall runs on past the slab (cross)
    bl, br = 70.0, 830.0
    jx, jy = wx, slab_cy                   # junction node

    # --- structural shell: two rooms, separating wall, flanking slab --------
    s.rect(bl, room_top, wall_l - bl, room_bot - room_top, th.panel, th.fg, sw=2.5)
    s.rect(wall_r, room_top, br - wall_r, room_bot - room_top, th.panel, th.fg, sw=2.5)
    # Flanking element (continuous slab through the junction).
    s.rect(bl, slab_top, br - bl, slab_bot - slab_top, th.panel, th.fg, sw=2)
    for hx in range(int(bl) + 16, int(br), 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    # Separating element (vertical wall, drawn on top -> rigid cross junction).
    s.rect(wall_l, room_top, wall_r - wall_l, wall_bot - room_top, th.secondary,
           th.fg, sw=2)

    s.text(bl + 16, room_top + 34, "Source room", 22, th.fg, bold=True, anchor="start")
    s.text(bl + 16, room_top + 60, "L₁", 20, th.muted, anchor="start")
    s.text(wall_r + 16, room_top + 34, "Receiving room", 22, th.fg, bold=True, anchor="start")
    s.text(wall_r + 16, room_top + 60, "L₂ , T", 20, th.muted, anchor="start")
    s.text(wx, room_top - 8, "Separating element (D, d)", 18, th.secondary, bold=True)
    s.text(bl + 16, slab_bot + 22, "Flanking element (F, f)", 18, th.fg, bold=True, anchor="start")

    # Loudspeaker (airborne excitation) in the source room, mic in receiving.
    lsx, lsy = 140.0, 300.0
    for r in (30, 50, 70):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.muted, sw=1.4)
    s.rect(lsx - 22, lsy - 26, 44, 52, th.panel, th.fg, rx=5, sw=2)
    s.circle(lsx, lsy - 8, 10, th.fg)
    s.circle(lsx, lsy - 8, 4, th.bg)
    s.circle(lsx, lsy + 14, 6, th.fg)
    s.text(lsx, lsy + 50, "Loudspeaker", 18, th.fg, bold=True)
    s.mic(786.0, 236.0, room_bot, 0.9)
    s.text(786.0, 220.0, "Microphone", 18, th.fg, bold=True)

    # --- transmission paths -------------------------------------------------
    # Dd: straight through the separating element, well above the slab.
    ddy = 172.0
    s.arrow(250.0, ddy, 648.0, ddy, c_dd, 3.0)
    s.text(300.0, ddy - 12, "Dd", 24, c_dd, bold=True)

    # Ff: down onto the flanking slab, along it through the junction, up again.
    s.line(250.0, 284.0, 250.0, slab_cy, c_ff, 2.8)
    s.line(250.0, slab_cy, 650.0, slab_cy, c_ff, 2.8)
    s.arrow(650.0, slab_cy, 650.0, 288.0, c_ff, 2.8)
    s.text(662.0, 300.0, "Ff", 24, c_ff, bold=True, anchor="start")

    # Fd: flanking element (source) -> junction -> radiates from the wall.
    s.line(330.0, 320.0, 330.0, slab_cy, c_fd, 2.8)
    s.line(330.0, slab_cy, 444.0, slab_cy, c_fd, 2.8)
    s.line(444.0, slab_cy, 444.0, 296.0, c_fd, 2.8)
    s.arrow(444.0, 296.0, 556.0, 236.0, c_fd, 2.8)
    s.text(560.0, 230.0, "Fd", 24, c_fd, bold=True, anchor="start")

    # Df: separating wall (source) -> junction -> radiates from the slab.
    s.line(392.0, 236.0, 456.0, 296.0, c_df, 2.8)
    s.line(456.0, 296.0, 456.0, slab_cy, c_df, 2.8)
    s.line(456.0, slab_cy, 614.0, slab_cy, c_df, 2.8)
    s.arrow(614.0, slab_cy, 614.0, 316.0, c_df, 2.8)
    s.text(626.0, 322.0, "Df", 24, c_df, bold=True, anchor="start")

    # Junction node on top of everything.
    s.circle(jx, jy, 6.5, th.bg, th.fg, 2.2)
    s.text(360.0, slab_bot + 22, "junction", 16, th.muted, italic=True)
    s.line(392.0, slab_bot + 17, jx - 7, jy + 3, th.muted, 0.9, dash="3,3")

    # --- legend + master formula (Formula 26) -------------------------------
    rows = [
        (c_dd, "Dd — direct path: separating element both sides"),
        (c_ff, "Ff — flanking–flanking: flanking element both sides"),
        (c_fd, "Fd — flanking (source) → separating (receiving)"),
        (c_df, "Df — separating (source) → flanking (receiving)"),
    ]
    ly = 452.0
    for col, txt in rows:
        s.line(bl + 4, ly - 6, bl + 44, ly - 6, col, 4.0)
        s.text(bl + 58, ly, txt, 19, th.fg, anchor="start")
        ly += 32
    s.text(450.0, ly + 12,
           "R'w = −10 lg Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)",
           19, th.muted, bold=True)


def _d_outdoor(s: SVG, th: Theme) -> None:
    c_diff = th.accent          # diffracted (over-the-top) ray
    c_direct = th.muted         # blocked direct ray
    gy = 430.0                  # ground line
    s.ground(gy, 60.0, 840.0)
    s.text(66.0, gy + 26.0, "Ground (Gs, Gm, Gr)", 18, th.muted, anchor="start")

    # --- source (loudspeaker) on the left, acoustic centre at (sx, sy) -------
    sx, sy = 150.0, 300.0
    for r in (26, 44, 62):
        s.path(f"M {sx + r * 0.22:.1f} {sy - r:.1f} "
               f"A {r} {r} 0 0 1 {sx + r:.1f} {sy - r * 0.22:.1f}",
               stroke=th.muted, sw=1.3)
    s.rect(sx - 20, sy - 24, 40, 48, th.panel, th.fg, rx=5, sw=2)
    s.circle(sx, sy - 6, 9, th.fg)
    s.circle(sx, sy - 6, 3.5, th.bg)
    s.circle(sx, sy + 14, 6, th.fg)
    s.line(sx, sy + 24, sx, gy, th.fg, 2.0)          # mast to the ground
    s.text(sx, sy - 74, "Source", 20, th.fg, bold=True)

    # --- barrier in the middle, top edge at (ex, ey) -------------------------
    ex, ey = 450.0, 150.0
    bw = 16.0
    s.rect(ex - bw / 2, ey, bw, gy - ey, th.secondary, th.fg, sw=2)
    s.text(ex + 16.0, (ey + gy) / 2 + 6.0, "Barrier", 20, th.secondary,
           bold=True, anchor="start")
    s.circle(ex, ey, 5.5, th.bg, th.fg, 2.0)          # diffraction edge node

    # --- receiver (microphone) on the right, capsule at (rx, ry) -------------
    rx, ry = 770.0, 288.0
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 20, th.fg, bold=True)

    # --- rays ---------------------------------------------------------------
    # Direct (blocked) ray straight through the barrier.
    s.line(sx + 14, sy - 6, rx, ry + 6, c_direct, 1.8, dash="7,6")
    s.text(285.0, sy + 40.0, "direct path (blocked)", 16, c_direct,
           anchor="middle", italic=True)
    # Diffracted ray up to the top edge, then down to the receiver.
    s.line(sx + 12, sy - 12, ex, ey, c_diff, 3.0)
    s.arrow(ex, ey, rx, ry + 2, c_diff, 3.0)
    s.text(300.0, 208.0, "dss", 18, c_diff, anchor="middle")
    s.text(610.0, 200.0, "dsr", 18, c_diff, anchor="middle")
    s.text(ex, ey - 22.0, "diffracted path", 17, c_diff, bold=True)

    # --- heights (witness dimensions) ---------------------------------------
    s.dim(sx - 44, gy, sx - 44, sy - 6, "hs", offset=0, label_side="left")
    s.line(sx - 44, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 44, sy - 6, sx, sy - 6, th.muted, 0.9, dash="3,3")
    s.dim(rx + 40, gy, rx + 40, ry + 6, "hr", offset=0, label_side="right")
    s.line(rx, gy, rx + 40, gy, th.muted, 0.9, dash="3,3")
    s.line(rx, ry + 6, rx + 40, ry + 6, th.muted, 0.9, dash="3,3")

    # --- master relations ---------------------------------------------------
    s.text(450.0, gy + 58.0, "z = dss + dsr − d   (path difference)", 19,
           th.fg, bold=True)
    s.text(450.0, gy + 84.0,
           "Dz = 10 lg[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)", 18, th.muted)


def _d_impedance_tube(s: SVG, th: Theme) -> None:
    """ISO 10534-2 two-microphone impedance tube (side view)."""
    tube_top, tube_bot, mid = 215.0, 335.0, 275.0
    tube_l, tube_r = 165.0, 778.0
    back_w, spec_w = 20.0, 48.0
    spec_l = tube_r - back_w - spec_w

    # Tube body.
    s.rect(tube_l, tube_top, tube_r - tube_l, tube_bot - tube_top, th.bg, th.fg, sw=3)

    # Loudspeaker sealed to the left end, cone opening into the tube.
    s.rect(72, mid - 46, 70, 92, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M 142 {mid - 18} L 142 {mid + 18} L {tube_l} {tube_bot} "
           f"L {tube_l} {tube_top} Z", fill=th.panel, stroke=th.primary, sw=2)
    s.circle(120, mid, 11, th.primary)
    s.text(118, tube_bot + 42, "Loudspeaker", 20, th.fg, bold=True)

    # Test specimen and rigid backing at the right end.
    s.rect(tube_r - back_w, tube_top, back_w, tube_bot - tube_top, th.fg)
    s.rect(spec_l, tube_top, spec_w, tube_bot - tube_top, th.panel, th.secondary, sw=2)
    for hx in range(int(spec_l) + 8, int(spec_l + spec_w), 11):
        s.line(hx, tube_bot - 4, hx - 16, tube_top + 4, th.secondary, 1.0)
    s.text(spec_l + spec_w / 2, tube_top - 14, "Test specimen", 19, th.secondary, bold=True)
    s.text(tube_r - back_w / 2, tube_bot + 42, "Rigid backing", 18, th.muted)

    # Two microphones flush in the top wall (mic 1 = farther from specimen).
    m1x, m2x = 460.0, 555.0
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2")):
        s.rect(mx - 7, tube_top - 20, 14, 20, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 28, lab, 18, th.fg, bold=True)

    # Plane-wave arrows inside the tube.
    s.arrow(tube_l + 30, mid - 18, spec_l - 16, mid - 18, th.accent, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid - 26, "incident", 17, th.accent)
    s.arrow(spec_l - 16, mid + 20, tube_l + 30, mid + 20, th.secondary, 2.2)
    s.text((tube_l + spec_l) / 2 - 40, mid + 38, "reflected", 17, th.secondary)

    # Dimensions: x1 (specimen face -> far mic) above, spacing s below.
    s.dim(spec_l, tube_top, m1x, tube_top, "x₁", offset=-58, size=19)
    s.dim(m1x, tube_bot, m2x, tube_bot, "s", offset=70, size=19)

    # Governing relations and range.
    for y, txt, col in (
        (438, ("H₁₂ → reflection factor r (Eq. 17), "
              "absorption α = 1 − |r|² (Eq. 18), "
              "Z/ρc₀ = (1+r)/(1−r) (Eq. 19)"), th.fg),
        (466, ("Working range f_l < f < f_u set by the microphone spacing s "
              "and the tube diameter (Clause 6.1)"), th.muted),
        (492, ("ASTM E2611: two further microphones behind the specimen also "
              "give the transmission loss"), th.muted),
    ):
        s.text(450, y, txt, 18, col)


def _d_astm_tube(s: SVG, th: Theme) -> None:
    """ASTM E2611 four-microphone transmission-loss tube (side view)."""
    tube_top, tube_bot, mid = 225.0, 345.0, 285.0
    tube_l, tube_r = 140.0, 825.0
    spec_l, spec_r = 453.0, 497.0
    m1x, m2x, m3x, m4x = 250.0, 360.0, 590.0, 700.0

    # Tube body.
    s.rect(tube_l, tube_top, tube_r - tube_l, tube_bot - tube_top, th.bg, th.fg, sw=3)

    # Loudspeaker sealed to the left end.
    s.rect(56, mid - 42, 62, 84, th.panel, th.primary, rx=6, sw=2)
    s.path(f"M 118 {mid - 16} L 118 {mid + 16} L {tube_l} {tube_bot} "
           f"L {tube_l} {tube_top} Z", fill=th.panel, stroke=th.primary, sw=2)
    s.circle(96, mid, 10, th.primary)
    s.text(96, tube_bot + 40, "Source", 19, th.fg, bold=True)

    # Adjustable termination (two loads) at the right end.
    s.rect(tube_r - 20, tube_top, 20, tube_bot - tube_top, th.fg)
    s.text(tube_r - 10, tube_bot + 40, "Termination", 17, th.muted)
    s.text(tube_r - 10, tube_bot + 60, "(2 loads)", 17, th.muted)

    # Test specimen at the centre.
    s.rect(spec_l, tube_top, spec_r - spec_l, tube_bot - tube_top, th.panel,
           th.secondary, sw=2)
    for hx in range(int(spec_l) + 7, int(spec_r), 10):
        s.line(hx, tube_bot - 4, hx - 14, tube_top + 4, th.secondary, 1.0)
    s.text((spec_l + spec_r) / 2, tube_bot + 40, "Test specimen", 18,
           th.secondary, bold=True)

    # Four microphones flush in the top wall (1,2 upstream; 3,4 downstream).
    for mx, lab in ((m1x, "Mic 1"), (m2x, "Mic 2"), (m3x, "Mic 3"), (m4x, "Mic 4")):
        s.rect(mx - 6, tube_top - 18, 12, 18, th.fg, rx=3)
        s.circle(mx, tube_top, 5, th.primary)
        s.text(mx, tube_top - 26, lab, 16, th.fg, bold=True)

    # Up- and downstream travelling waves.
    s.arrow(tube_l + 26, mid - 16, spec_l - 8, mid - 16, th.accent, 2.0)
    s.arrow(spec_l - 8, mid + 18, tube_l + 26, mid + 18, th.secondary, 2.0)
    s.arrow(spec_r + 8, mid - 16, tube_r - 26, mid - 16, th.accent, 2.0)
    s.arrow(tube_r - 26, mid + 18, spec_r + 8, mid + 18, th.secondary, 2.0)
    s.text(tube_l + 40, mid - 22, "A", 17, th.accent, bold=True)
    s.text(tube_l + 40, mid + 34, "B", 17, th.secondary, bold=True)
    s.text(tube_r - 40, mid - 22, "C", 17, th.accent, bold=True)
    s.text(tube_r - 40, mid + 34, "D", 17, th.secondary, bold=True)

    # Dimensions: spacings s1/s2 below; specimen offsets l1/l2 and thickness d above.
    s.dim(m1x, tube_bot, m2x, tube_bot, "s₁", offset=62, size=18)
    s.dim(m3x, tube_bot, m4x, tube_bot, "s₂", offset=62, size=18)
    # l1, l2 are both measured from the specimen FRONT face (x = 0), matching
    # wave_decomposition/transfer_matrix_two_load; l2 therefore spans the specimen.
    s.dim(m2x, tube_top, spec_l, tube_top, "l₁", offset=-42, size=18)
    s.dim(spec_l, tube_top, m3x, tube_top, "l₂", offset=-58, size=18)
    s.dim(spec_l, tube_top - 78, spec_r, tube_top - 78, "d", offset=0, size=17)
    s.line(spec_l, tube_top, spec_l, tube_top - 78, th.muted, 0.9, dash="3,3")
    s.line(spec_r, tube_top, spec_r, tube_top - 78, th.muted, 0.9, dash="3,3")

    # Governing relations.
    for y, txt, col in (
        (452, ("Decompose A, B (upstream) and C, D (downstream) → "
              "transfer matrix T (Eq. 22)"), th.fg),
        (480, "TL = 20 log₁₀ |(T₁₁ + T₁₂/ρc + ρc·T₂₁ + T₂₂) / 2|   (Eq. 26)",
         th.muted),
        (506, ("Two-load method: repeat with two terminations; the one-load "
              "variant uses a single anechoic end"), th.muted),
    ):
        s.text(450, y, txt, 17, col)


def _d_airflow(s: SVG, th: Theme) -> None:
    """ISO 9053-1 static and ISO 9053-2 alternating airflow-resistance rigs."""
    # --- Left panel: static (DC) method -----------------------------------
    s.rect(55, 70, 385, 430, th.panel, th.fg, rx=8, sw=2)
    s.text(247, 100, "Static method (ISO 9053-1)", 21, th.fg, bold=True)

    cx = 200.0
    holder_l, holder_r = cx - 45, cx + 45
    top_y, bot_y = 170.0, 430.0
    # Vertical specimen holder (tube).
    s.line(holder_l, top_y, holder_l, bot_y, th.fg, 2.5)
    s.line(holder_r, top_y, holder_r, bot_y, th.fg, 2.5)
    # Specimen (hatched disc) in the middle.
    spec_y, spec_h = 285.0, 46.0
    s.rect(holder_l, spec_y, 90, spec_h, th.bg, th.secondary, sw=2)
    for hy in range(int(spec_y) + 8, int(spec_y + spec_h), 10):
        s.line(holder_l + 4, hy, holder_r - 4, hy - 8, th.secondary, 1.0)
    s.text(cx, spec_y + spec_h + 22, "specimen (A, d)", 17, th.secondary, bold=True)
    # Steady laminar flow up through the holder.
    s.arrow(cx, bot_y - 6, cx, spec_y + spec_h + 34, th.accent, 2.4)
    s.arrow(cx, spec_y - 12, cx, top_y + 8, th.accent, 2.4)
    s.text(cx, bot_y + 22, "laminar flow  q_v", 18, th.accent, bold=True)
    # Differential manometer across the specimen (pressure taps).
    tap_x = holder_r + 8
    s.line(holder_r, spec_y - 4, tap_x + 40, spec_y - 4, th.primary, 1.6)
    s.line(holder_r, spec_y + spec_h + 4, tap_x + 40, spec_y + spec_h + 4, th.primary, 1.6)
    s.rect(tap_x + 40, spec_y - 26, 74, spec_h + 44, th.bg, th.primary, rx=8, sw=2)
    s.text(tap_x + 77, spec_y + 8, "Δp", 22, th.primary, bold=True, mono=True)
    s.text(tap_x + 77, spec_y + 34, "manom.", 15, th.muted)
    s.text(247, 478, "R = Δp / q_v   (through-origin fit at 0.5 mm/s)",
           16, th.fg, bold=True)

    # --- Right panel: alternating (AC) method -----------------------------
    s.rect(460, 70, 385, 430, th.panel, th.fg, rx=8, sw=2)
    s.text(652, 100, "Alternating method (ISO 9053-2)", 21, th.fg, bold=True)

    cav_l, cav_r = 590.0, 715.0
    cav_top, cav_bot = 160.0, 360.0
    # Cavity walls.
    s.rect(cav_l, cav_top, cav_r - cav_l, cav_bot - cav_top, th.bg, th.fg, sw=2.5)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 - 6, "cavity", 18, th.fg)
    s.text((cav_l + cav_r) / 2, (cav_top + cav_bot) / 2 + 18, "V", 20, th.fg,
           bold=True, italic=True)
    # Specimen / airtight termination on top.
    s.rect(cav_l, cav_top - 26, cav_r - cav_l, 26, th.bg, th.secondary, sw=2)
    for hx in range(int(cav_l) + 8, int(cav_r), 11):
        s.line(hx, cav_top - 4, hx - 14, cav_top - 22, th.secondary, 1.0)
    s.text((cav_l + cav_r) / 2, cav_top - 36, "specimen / airtight", 16,
           th.secondary, bold=True)
    # Piston at the bottom, oscillating.
    s.rect(cav_l, cav_bot, cav_r - cav_l, 26, th.panel, th.primary, sw=2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 58, (cav_l + cav_r) / 2, cav_bot + 30,
            th.primary, 2.2)
    s.arrow((cav_l + cav_r) / 2, cav_bot + 30, (cav_l + cav_r) / 2, cav_bot + 58,
            th.primary, 2.2)
    s.text((cav_l + cav_r) / 2, cav_bot + 80, "piston  f = 1–4 Hz", 18,
           th.primary, bold=True)
    # Microphone in the cavity wall.
    s.circle(cav_r + 2, (cav_top + cav_bot) / 2, 6, th.fg)
    s.line(cav_r + 2, (cav_top + cav_bot) / 2, cav_r + 60,
           (cav_top + cav_bot) / 2, th.muted, 1.4)
    s.text(cav_r + 66, (cav_top + cav_bot) / 2 + 6, "L_p", 20, th.fg,
           bold=True, mono=True, anchor="start")
    s.text(652, 478, "R from L_p,s − L_p,t   (κ′ per Annex A)",
           16, th.fg, bold=True)


def _rot_arrow(s: SVG, cx: float, cy: float, r: float, a0_deg: float,
               a1_deg: float, color: str, sw: float = 2.0,
               ry: float | None = None) -> None:
    """Curved rotation indicator: an elliptical arc with a head at ``a1``."""
    import math
    ryy = r if ry is None else ry
    a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
    x0, y0 = cx + r * math.cos(a0), cy + ryy * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + ryy * math.sin(a1)
    large = 1 if abs(a1_deg - a0_deg) > 180 else 0
    sweep = 1 if a1_deg > a0_deg else 0
    s.path(f"M {x0:.1f} {y0:.1f} A {r:.1f} {ryy:.1f} 0 {large} {sweep} "
           f"{x1:.1f} {y1:.1f}", stroke=color, sw=sw)
    tang = a1 + (math.pi / 2 if sweep else -math.pi / 2)
    L, W = 10.0, 4.4
    bx, by = x1 - L * math.cos(tang), y1 - L * math.sin(tang)
    px, py = -math.sin(tang), math.cos(tang)
    s.path(f"M {x1:.1f} {y1:.1f} L {bx + W * px:.1f} {by + W * py:.1f} "
           f"L {bx - W * px:.1f} {by - W * py:.1f} Z", fill=color)


# ---------------------------------------------------------------------------
# d15 - ISO 17497-1 random-incidence scattering (reverberation room)
# ---------------------------------------------------------------------------

def _d_scattering_reverb(s: SVG, th: Theme) -> None:
    """ISO 17497-1 scattering coefficient in a reverberation room."""
    gy = 400.0
    # Reverberation room with non-parallel walls (skew quadrilateral).
    s.path("M 60 80 L 782 66 L 796 400 L 72 400 Z", fill=th.panel,
           stroke=th.fg, sw=3)
    s.text(80, 106, "Reverberation room", 20, th.fg, bold=True, anchor="start")

    # --- Turntable carrying the test sample (left, in perspective) --------
    tx, tyc = 285.0, 366.0
    s.ellipse(tx, tyc, 150, 26, th.panel, th.primary, 2.2)      # turntable
    s.ellipse(tx, tyc - 12, 82, 15, th.bg, th.secondary, 2.2)   # test sample
    for hx in range(int(tx) - 60, int(tx) + 60, 12):            # sample hatch
        s.line(hx, tyc - 10, hx + 10, tyc - 18, th.secondary, 1.0)
    s.text(tx, gy + 22, "Turntable (test sample)", 17, th.fg, bold=True)
    _rot_arrow(s, tx, tyc, 150, 205, 340, th.accent, 2.2, ry=26)
    s.text(445, tyc + 6, "rotating → α_spec", 15, th.accent, anchor="start")
    s.text(tx, tyc - 42, "stationary → α_s", 15, th.muted)

    # --- Rotating boom loudspeaker source (upper right) -------------------
    pvx, pvy = 560.0, 100.0
    spx, spy = 668.0, 202.0
    s.circle(pvx, pvy, 5, th.fg)
    s.line(pvx, pvy, spx, spy, th.fg, 3)
    s.rect(spx - 26, spy - 26, 40, 52, th.panel, th.primary, rx=6, sw=2)
    s.circle(spx - 6, spy, 11, th.primary)
    s.circle(spx - 6, spy, 4, th.bg)
    _rot_arrow(s, pvx, pvy, 118, -18, 46, th.accent, 2.0)
    s.text(spx + 8, spy + 46, "Rotating boom source", 18, th.fg, bold=True)

    # --- Microphone on a stand in the room --------------------------------
    s.mic(468.0, 246.0, gy, 1.0)
    s.text(468.0, 234.0, "Microphone", 18, th.fg, bold=True)

    # --- Governing relations ----------------------------------------------
    for y, txt, col, bold in (
        (448, ("Stationary sample → α_s (Eq. 1)   ·   "
              "rotating / averaged → α_spec (Eq. 4)"), th.fg, True),
        (478, "s = (α_spec − α_s) / (1 − α_s)   (Eq. 5)", th.accent, True),
        (508, ("α from 55.3·(V/S)·(1/cT) − 4(V/S)m   "
              "(Sabine, Table 2 rows T1–T4)"), th.muted, False),
        (534, "Base-plate check: s_base ≤ Table 1 limit (Clause 6.2)",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


# ---------------------------------------------------------------------------
# d16 - ISO 17497-2 free-field diffusion goniometer
# ---------------------------------------------------------------------------

def _d_diffusion_goniometer(s: SVG, th: Theme) -> None:
    """ISO 17497-2 directional diffusion coefficient (goniometer)."""
    import math
    gy, cx, R = 430.0, 450.0, 300.0
    s.ground(gy, 90, 810)

    # Semicircular receiver arc (0 deg right .. 180 deg left, zenith at top).
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.muted, sw=1.8)
    ends = {0, 90, 180}
    for ang in range(0, 181, 15):
        a = math.radians(ang)
        px, py = cx + R * math.cos(a), gy - R * math.sin(a)
        s.circle(px, py, 6.5, th.primary)
        s.circle(px, py, 2.2, th.bg)
    # Label the two horizon receivers and the zenith one.
    s.text(cx + R + 4, gy - 4, "L_n", 17, th.fg, anchor="start")
    s.text(cx - R - 4, gy - 4, "L_1", 17, th.fg, anchor="end")
    s.text(cx, gy - R - 14, "L_i", 17, th.fg)
    s.text(cx + 150, gy - 250, "receiver arc (5° steps)", 16, th.muted)
    _ = ends

    # Polar (scattered) response lobe about the sample centre.
    pts = []
    for ang in range(0, 181, 6):
        a = math.radians(ang)
        rr = 92.0 + 42.0 * abs(math.sin(3.0 * a))
        pts.append((cx + rr * math.cos(a), gy - rr * math.sin(a)))
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    s.path(d, stroke=th.accent, sw=2.0)
    s.text(cx + 96, gy - 150, "polar response L_i", 16, th.accent)

    # Fixed source, off to the upper left, illuminating the sample.
    sa = math.radians(155.0)
    sxx, syy = cx + (R + 44) * math.cos(sa), gy - (R + 44) * math.sin(sa)
    s.rect(sxx - 26, syy - 22, 52, 44, th.panel, th.primary, rx=6, sw=2)
    s.circle(sxx + 20, syy, 10, th.primary)
    s.circle(sxx + 20, syy, 4, th.bg)
    s.text(sxx, syy - 32, "Fixed source", 17, th.fg, bold=True)
    s.arrow(sxx + 26, syy + 6, cx - 74, gy - 12, th.accent, 2.0)

    # Test sample on the turntable at the arc centre.
    s.rect(cx - 72, gy - 13, 144, 13, th.bg, th.secondary, sw=2)
    for hx in range(int(cx) - 64, int(cx) + 64, 12):
        s.line(hx, gy - 3, hx + 9, gy - 11, th.secondary, 1.0)
    s.text(cx, gy - 20, "Test sample", 16, th.secondary, bold=True)
    s.ellipse(cx, gy + 8, 88, 12, "none", th.primary, 1.8)
    _rot_arrow(s, cx, gy + 8, 88, 200, 340, th.primary, 1.8, ry=12)
    s.text(cx + 150, gy + 12, "Turntable", 16, th.fg, bold=True, anchor="start")

    # Governing relations.
    s.text(450, 476,
           "d = [(Σ10^(L_i/10))² − Σ(10^(L_i/10))²] / "
           "[(n−1)·Σ(10^(L_i/10))²]   (Formula 5)", 17, th.fg, bold=True)
    s.text(450, 506, "d_n = (d − d_ref) / (1 − d_ref)   (Formula 7)", 18,
           th.accent, bold=True)
    s.text(450, 534,
           "5° receiver steps · turntable rotates the sample · source fixed",
           17, th.muted)


# ---------------------------------------------------------------------------
# d17 - ISO 13472-1 in-situ road absorption, subtraction technique
# ---------------------------------------------------------------------------

def _d_insitu_subtraction(s: SVG, th: Theme) -> None:
    """ISO 13472-1 extended-surface (subtraction) in-situ absorption."""
    gy = 415.0
    # Road surface (the reference plane) under the main measurement.
    s.ground(gy, 55, 590)
    s.text(66, gy + 30, "Road surface", 16, th.muted, anchor="start")

    sx = 250.0
    src_y, mic_y = gy - 235.0, gy - 47.0        # ds : dm = 1.25 : 0.25 m
    s.line(sx, src_y, sx, gy, th.muted, 1.0, dash="4,4")   # normal axis

    # Loudspeaker (source) at ds above the surface.
    s.rect(sx - 30, src_y - 30, 60, 60, th.panel, th.primary, rx=6, sw=2)
    s.circle(sx, src_y, 12, th.primary)
    s.circle(sx, src_y, 5, th.bg)
    s.text(sx, src_y - 42, "Loudspeaker", 18, th.fg, bold=True)

    # Microphone at dm above the surface.
    s.rect(sx - 6, mic_y - 9, 12, 18, th.fg, rx=3)
    s.circle(sx, mic_y - 9, 5, th.primary)
    s.text(sx + 16, mic_y + 5, "Microphone", 15, th.fg, anchor="start")

    # Direct ray (source -> mic), drawn offset to the left of the axis.
    s.arrow(sx - 7, src_y + 22, sx - 7, mic_y - 12, th.accent, 2.0)
    s.text(sx - 60, (src_y + mic_y) / 2, "direct  ds−dm", 15, th.accent,
           anchor="end")
    # Road-reflected ray: source -> surface point -> mic (shallow V, offset).
    gpx = sx + 74.0
    s.line(sx + 8, src_y + 24, gpx, gy, th.secondary, 2.0)
    s.arrow(gpx, gy, sx + 8, mic_y + 6, th.secondary, 2.0)
    s.text(gpx + 8, gy - 96, "reflected  ds+dm", 15, th.secondary,
           anchor="start")
    # Dashed continuation toward the image source below the plane.
    s.line(gpx, gy, sx + 34, gy + 66, th.muted, 1.2, dash="5,4")
    s.text(sx + 40, gy + 60, "to image source (ds below)", 14, th.muted,
           anchor="start")

    # Height dimensions ds and dm.
    s.dim(sx - 72, gy, sx - 72, src_y, "ds = 1.25 m", offset=0,
          label_side="left", size=17)
    s.line(sx - 72, gy, sx, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 72, src_y, sx - 30, src_y, th.muted, 0.9, dash="3,3")
    s.dim(sx + 122, gy, sx + 122, mic_y, "dm = 0.25 m", offset=0,
          label_side="right", size=17)
    s.line(sx, mic_y, sx + 122, mic_y, th.muted, 0.9, dash="3,3")

    # --- Free-field reference (right): source + mic high, no ground -------
    s.line(615, 90, 615, gy + 40, th.muted, 1.2, dash="6,5")
    fx = 730.0
    fs_y, fm_y = 150.0, 292.0
    s.rect(fx - 28, fs_y - 26, 56, 52, th.panel, th.primary, rx=6, sw=2)
    s.circle(fx, fs_y, 11, th.primary)
    s.circle(fx, fs_y, 4, th.bg)
    s.rect(fx - 6, fm_y - 9, 12, 18, th.fg, rx=3)
    s.circle(fx, fm_y - 9, 5, th.primary)
    s.arrow(fx, fs_y + 28, fx, fm_y - 14, th.accent, 2.0)
    s.text(fx, fs_y - 40, "Free-field reference", 17, th.fg, bold=True)
    s.text(fx, fm_y + 34, "Hi: no ground reflection in the window", 14,
           th.muted)

    # Governing relations.
    s.text(450, 502, "Kr = (ds − dm)/(ds + dm) = 2/3   (Clause 4.1)", 18,
           th.fg, bold=True)
    s.text(450, 528, "α(f) = 1 − (1/Kr²)·|Hr/Hi|²   ·   Δτ = 2 dm / c", 18,
           th.accent, bold=True)
    s.text(450, 552, "Adrienne time window isolates the reflected response Hr",
           16, th.muted)


# ---------------------------------------------------------------------------
# d18 - ISO 13472-2 in-situ road absorption, spot method
# ---------------------------------------------------------------------------

def _d_spot_tube(s: SVG, th: Theme) -> None:
    """ISO 13472-2 spot method: short tube sealed onto the road surface."""
    gy = 430.0
    cx, hw, y_top = 235.0, 72.0, 120.0

    # Road surface (the test sample) with the tube sealed onto it.
    s.ground(gy, 60, 430)
    s.text(72, gy + 30, "Road surface (test sample)", 15, th.muted,
           anchor="start")

    # Tube walls.
    s.line(cx - hw, y_top, cx - hw, gy, th.fg, 3)
    s.line(cx + hw, y_top, cx + hw, gy, th.fg, 3)
    # Sealing rings where the tube meets the road.
    s.rect(cx - hw - 7, gy - 9, 14, 18, th.muted, rx=2)
    s.rect(cx + hw - 7, gy - 9, 14, 18, th.muted, rx=2)

    # Loudspeaker cap at the top.
    s.rect(cx - hw, y_top - 40, 2 * hw, 40, th.panel, th.primary, sw=2)
    s.circle(cx, y_top - 20, 12, th.primary)
    s.circle(cx, y_top - 20, 5, th.bg)
    s.text(cx, y_top - 52, "Loudspeaker", 18, th.fg, bold=True)

    # Two microphones flush in the right wall, spacing s.
    m1y, m2y = gy - 158.0, gy - 82.0
    for my, lab in ((m1y, "Mic 1"), (m2y, "Mic 2")):
        s.rect(cx + hw - 4, my - 7, 12, 14, th.fg, rx=3)
        s.circle(cx + hw, my, 4, th.primary)
        s.text(cx + hw + 16, my + 5, lab, 15, th.fg, anchor="start")

    # Plane-wave travel down and reflection back up.
    s.arrow(cx - 34, y_top + 16, cx - 34, gy - 26, th.accent, 2.0)
    s.arrow(cx - 8, gy - 26, cx - 8, y_top + 16, th.secondary, 2.0)

    # Dimensions: tube diameter d (across) and mic spacing s (down).
    s.dim(cx - hw, y_top + 18, cx + hw, y_top + 18, "d", offset=0, size=18)
    s.dim(cx + hw + 62, m1y, cx + hw + 62, m2y, "s", offset=0,
          label_side="right", size=18)
    s.line(cx + hw + 10, m1y, cx + hw + 62, m1y, th.muted, 0.9, dash="3,3")
    s.line(cx + hw + 10, m2y, cx + hw + 62, m2y, th.muted, 0.9, dash="3,3")

    # Right panel: usable frequency range and DSP method.
    s.rect(430, 118, 430, 300, "none", th.muted, rx=12, dash="6,5")
    s.text(645, 152, "Spot method (ISO 13472-2)", 20, th.fg, bold=True)
    for y, txt, col in (
        (196, "f_u = 0.58 c₀ / d   (Clause 5.4.1)", th.accent),
        (232, "0.05 c₀/f_min < s < 0.45 c₀/f_max   (Clause 5.4.2)", th.accent),
        (268, "Working range: 250–1600 Hz (1/3-octave)", th.fg),
        (312, "Two-microphone transfer function H₁₂", th.fg),
        (344, "→ ISO 10534-2 decomposition → α(f)", th.primary),
    ):
        s.text(645, y, txt, 18, col, bold=(col is th.primary))
    s.text(645, 396, "Tube sealed onto the road; plane waves only below f_u",
           15, th.muted)


# ---------------------------------------------------------------------------
# d19 - ISO 3745 precision sound power (anechoic / hemi-anechoic room)
# ---------------------------------------------------------------------------

def _d_precision_anechoic(s: SVG, th: Theme) -> None:
    """ISO 3745 precision sound power on a (hemi-)spherical array."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges lining the ceiling and the two side walls.
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    for wy in range(int(y0) + 30, int(gy) - 36, 40):
        s.path(f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    s.text(200, 120, "Anechoic wedges", 15, th.muted, anchor="start")

    # Reflecting floor (hemi-anechoic room).
    s.ground(gy, x0, x1)
    s.text(70, gy - 8, "Reflecting plane (hemi-anechoic)", 15, th.muted,
           anchor="start")

    # Source (DUT) at the centre of the reflecting plane.
    cx, R = 450.0, 200.0
    _box_solid(s, th, cx, gy, 34, 26, 40)
    s.circle(cx, gy, 3.4, th.fg)
    s.text(cx + 52, gy - 14, "Source (DUT)", 17, th.fg, bold=True,
           anchor="start")

    # Hemispherical measurement surface of radius r.
    s.ellipse(cx, gy, R, R * 0.16, "none", th.muted, 1.3, dash="5,4")
    s.path(f"M {cx - R} {gy} A {R} {R} 0 0 1 {cx + R} {gy}",
           stroke=th.primary, sw=2.4)

    # Ten normative microphone positions (ISO 3744/3745 Annex B), projected.
    b1 = [(0.16, -0.96, 0.22), (0.78, -0.60, 0.20), (0.78, 0.55, 0.31),
          (0.16, 0.90, 0.41), (-0.83, 0.32, 0.45), (-0.83, -0.40, 0.38),
          (-0.26, -0.65, 0.71), (0.74, -0.07, 0.67), (-0.26, 0.50, 0.83),
          (0.10, -0.10, 0.99)]
    pts = [(cx + R * x + 46 * y, gy - 30 * y - R * z) for x, y, z in b1]
    r8 = pts[7]
    s.line(cx, gy, r8[0], r8[1], th.accent, 1.6, dash="6,4")
    s.text((cx + r8[0]) / 2 + 8, (gy + r8[1]) / 2 + 2, "radius r", 16,
           th.accent, anchor="start")
    for px, py in pts:
        s.circle(px, py, 6.5, th.secondary)
        s.circle(px, py, 2.2, th.bg)
    s.text(688, 300, "20 / 40 mic positions", 16, th.muted, anchor="start")

    # Governing relations.
    for y, txt, col, bold in (
        (514, "LW = ⟨Lp⟩ + 10 lg(S/S0) + C1 + C2 + C3", th.fg, True),
        (540, "S = 2πr² (hemi-anechoic) · 4πr² (anechoic)", th.primary, True),
        (564, "K1: per-position background correction", th.muted, False),
        (587, "C1, C2, C3: meteorological corrections (ps, θ, a(f))",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


# ---------------------------------------------------------------------------
# d20 - ISO 9614-3 precision sound intensity scanning
# ---------------------------------------------------------------------------

def _d_intensity_scan(s: SVG, th: Theme) -> None:
    """ISO 9614-3 precision sound power by intensity scanning."""
    gy, bx = 470.0, 360.0

    # Measurement surface (dashed wireframe) enclosing the source.
    _box_wire(s, th, bx, gy, 150, 120, 240, th.primary)
    _box_solid(s, th, bx, gy, 45, 34, 70)
    s.text(bx, gy - 82, "Source", 18, th.fg, bold=True)
    s.text(bx, 214, "Measurement surface (segments S_i)", 17, th.primary,
           bold=True)

    # Segment grid on the front face (3 x 3 segments Sᵢ).
    fl, fr, ft, fb = bx - 150, bx + 150, gy - 240, gy
    for gx in (fl + 100, fl + 200):
        s.line(gx, ft, gx, fb, th.muted, 1.2, dash="4,4")
    for gyy in (ft + 80, ft + 160):
        s.line(fl, gyy, fr, gyy, th.muted, 1.2, dash="4,4")
    s.text(fl + 50, ft + 46, "S_i", 18, th.fg, bold=True)

    # Serpentine scan path across the segment-row centres.
    ys = (ft + 40, ft + 120, ft + 200)
    px = [(fl + 30, ys[0]), (fr - 30, ys[0]), (fr - 30, ys[1]),
          (fl + 30, ys[1]), (fl + 30, ys[2]), (fr - 30, ys[2])]
    for (ax, ay), (bxx, byy) in itertools.pairwise(px):
        s.line(ax, ay, bxx, byy, th.accent, 2.0, dash="2,3")
    s.arrow(px[-2][0] + 60, px[-1][1], px[-1][0], px[-1][1], th.accent, 2.0)
    s.text(fr + 8, ys[2] + 6, "serpentine scan", 15, th.accent, anchor="start")

    # A p-p intensity probe on the scan path.
    ppx, ppy = bx, ys[1]
    s.line(ppx, ppy, ppx + 46, ppy - 26, th.fg, 2.2)
    s.circle(ppx, ppy - 6, 5, th.fg)
    s.circle(ppx, ppy + 6, 5, th.fg)
    s.text(ppx + 52, ppy - 30, "p-p probe", 15, th.fg, anchor="start")

    # Normal-intensity arrows exiting the left column of segments.
    for yy in ys:
        s.arrow(fl, yy, fl - 34, yy + 8, th.secondary, 2.0)
    s.text(fl - 40, ys[1] + 30, "I_n (normal intensity)", 15, th.secondary,
           anchor="end")

    # Governing relations.
    for y, txt, col, bold in (
        (505, "P = Σ I_n,i · S_i   (partial powers per segment)", th.fg, True),
        (533, "LW = 10 lg(P/P0),  P0 = 1 pW", th.accent, True),
        (559, "Field indicators: F_pIn , FT , FS", th.primary, True),
        (583, "Five acceptance criteria (Annex C); band invalid if P < 0",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


def _d_human_vibration(s: SVG, th: Theme) -> None:
    """Whole-body vibration measurement chain (ISO 2631-1 / ISO 8041-1)."""
    gy = 510.0
    # --- Left: a seated person on a vibrating seat, triaxial accelerometer ---
    s.ground(gy, 40, 350)
    # Seat: cushion, backrest and support leg.
    s.rect(118, 424, 132, 18, th.panel, th.fg, rx=4, sw=2)      # cushion
    s.rect(118, 336, 16, 90, th.panel, th.fg, rx=3, sw=2)       # backrest
    s.line(184, 442, 184, gy, th.fg, 2.4)                       # pedestal
    # A wavy "vibration" arrow rising into the seat base.
    s.arrow(184, gy - 4, 184, 452, th.secondary, 2.4)
    s.text(184, gy - 12, "vibration input", 17, th.secondary, "middle", italic=True)
    s.person(178, gy, 176, seated=True)
    # Triaxial accelerometer at the seat/body interface with its x, y, z axes.
    ox, oy = 176.0, 420.0
    s.rect(ox - 9, oy - 8, 18, 16, th.secondary, th.fg, rx=2, sw=1.5)
    s.arrow(ox, oy - 8, ox, oy - 58, th.accent, 2.0)            # z (vertical)
    s.text(ox + 8, oy - 54, "z", 18, th.accent, "start", bold=True)
    s.arrow(ox + 9, oy, ox + 62, oy, th.accent, 2.0)            # x (fore-aft)
    s.text(ox + 66, oy + 5, "x", 18, th.accent, "start", bold=True)
    s.arrow(ox - 7, oy + 6, ox - 44, oy + 34, th.accent, 2.0)   # y (lateral)
    s.text(ox - 52, oy + 44, "y", 18, th.accent, "end", bold=True)
    s.text(150, gy + 34, "Seat/body interface", 18, th.fg, "middle")

    # --- Right: the vertical signal-processing chain ---
    cx, bw, bh = 650.0, 320.0, 72.0
    x0 = cx - bw / 2
    chain = [
        (96.0, "Triaxial accelerometer", "a_x , a_y , a_z  (m/s²)"),
        (206.0, "Band limiting + Wk / Wd", "weighting (ISO 8041-1)"),
        (316.0, "Weighted r.m.s. a_w  &  VDV", "(ISO 2631-1)"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(cx, by + 31, l1, 21, th.fg, "middle", bold=True)
        s.text(cx, by + 56, l2, 18, th.muted, "middle")
    s.arrow(cx, 168, cx, 206, th.fg, 2.0)
    s.arrow(cx, 278, cx, 316, th.fg, 2.0)
    # Feed the setup into the chain.
    s.arrow(252, oy, x0 - 6, 132, th.fg, 2.0)

    # --- Bottom: dominant axis, daily exposure and the Directive assessment ---
    # The Directive's whole-body A(8) is based on the HIGHEST frequency-
    # weighted axis value (1,4 a_wx, 1,4 a_wy, a_wz), Annex Part B point 1 -
    # not on the ISO 2631-1 Eq. (10) vector total a_v.
    s.arrow(cx, 388, cx, 424, th.fg, 2.0)
    s.rect(400, 424, 470, 78, "none", th.secondary, rx=12, sw=2, dash="6,5")
    s.text(635, 452, "A(8) = max(1.4·a_wx , 1.4·a_wy , a_wz)·√(T/T₀)",
           20, th.fg, "middle", bold=True)
    s.text(635, 480, "assessed vs EAV / ELV (Directive 2002/44/EC)",
           18, th.secondary, "middle")


def _d_speech_intelligibility(s: SVG, th: Theme) -> None:
    """SII computation flow (ANSI S3.5-1997, one-third-octave method)."""
    # --- Top: three equivalent-spectrum-level inputs (per 1/3-octave band) ---
    inputs = [
        (150.0, "Speech  Ei'", th.primary),
        (450.0, "Noise  Ni'", th.secondary),
        (750.0, "Threshold  Ti'", th.accent),
    ]
    iw, ih, iy = 220.0, 66.0, 40.0
    for cx, label, col in inputs:
        s.rect(cx - iw / 2, iy, iw, ih, th.panel, col, rx=10, sw=2)
        s.text(cx, iy + 28, label, 21, th.fg, "middle", bold=True)
        s.text(cx, iy + 51, "spectrum level (dB)", 16, th.muted, "middle")
        s.arrow(cx, iy + ih, cx, 150, th.fg, 1.8)

    # --- Vertical processing chain (ANSI S3.5-1997 clause 5) ---
    cx, bw, bh = 450.0, 470.0, 70.0
    x0 = cx - bw / 2
    chain = [
        (150.0, "Self-masking + spread of masking", "Zi   (clause 5.4)"),
        (264.0, "Equivalent disturbance Di", "max(masking, internal noise) (5.6)"),
        (378.0, "Band audibility Ai = (Ei' − Di + 15)/30", "clipped to [0, 1]   (clause 5.8)"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.fg, rx=12, sw=2)
        s.text(cx, by + 30, l1, 20, th.fg, "middle", bold=True)
        s.text(cx, by + 54, l2, 17, th.muted, "middle")
    s.arrow(cx, 220, cx, 264, th.fg, 2.0)
    s.arrow(cx, 334, cx, 378, th.fg, 2.0)

    # --- Band-importance weighting and the final index ---
    s.arrow(cx, 448, cx, 486, th.fg, 2.0)
    s.rect(x0, 486, bw, 74, "none", th.primary, rx=12, sw=2.4)
    s.text(cx, 516, "SII = Σ I_i A_i", 26, th.fg, "middle", bold=True)
    s.text(cx, 542, "band importance I_i (Table 3)  ·  index in [0, 1]  (clause 6)",
           16, th.primary, "middle")


def _d_room_measurement(s: SVG, th: Theme) -> None:
    """Room-acoustics measurement layout (ISO 3382-1 positions, ISO 3382-2 grades).

    A top-view room plan with two source positions and six microphone
    positions plus the ISO 3382-1 spacing rules, and a table of the
    ISO 3382-2:2008 Table 1 minimum position counts for the three grades.
    """
    # --- Room plan (top view) ------------------------------------------------
    rx, ry, rw, rh = 60.0, 96.0, 500.0, 300.0
    s.rect(rx, ry, rw, rh, th.panel, th.fg, rx=6, sw=2.4)
    s.text(rx + 10, ry - 12, "Room plan (top view)", 20, th.fg, "start", bold=True)

    # Two loudspeaker source positions (ISO 3382-1: at least two).
    def _speaker(x: float, y: float, label: str) -> None:
        s.rect(x - 13, y - 11, 26, 22, th.primary, th.fg, rx=3, sw=1.6)
        s.circle(x, y, 5, th.bg, th.fg, 1.2)
        s.text(x, y - 18, label, 18, th.primary, "middle", bold=True)

    _speaker(rx + 70, ry + 70, "S1")
    _speaker(rx + rw - 80, ry + rh - 70, "S2")

    # Six microphone positions, asymmetric (ISO 3382-1: >= 2 m apart,
    # >= 1 m from surfaces; >= 3 receivers per source in ISO 3382-2 precision).
    mics = [
        (rx + 180, ry + 90, "M1"),
        (rx + 300, ry + 55, "M2"),
        (rx + 420, ry + 130, "M3"),
        (rx + 250, ry + 220, "M4"),
        (rx + 380, ry + 250, "M5"),
        (rx + 130, ry + 210, "M6"),
    ]
    for mx, my, label in mics:
        s.circle(mx, my, 7, th.secondary, th.fg, 1.4)
        s.text(mx + 12, my + 6, label, 17, th.fg, "start", bold=True)

    # Spacing annotations.
    m1 = (rx + 180, ry + 90)
    m2 = (rx + 300, ry + 55)
    s.line(m1[0], m1[1], m2[0], m2[1], th.accent, 1.6, dash="5,4")
    s.text((m1[0] + m2[0]) / 2, (m1[1] + m2[1]) / 2 - 8,
           "≥ 2 m", 17, th.accent, "middle", bold=True)
    m6 = (rx + 130, ry + 210)
    s.arrow(m6[0], m6[1] + 9, m6[0], ry + rh, th.muted, 1.4)
    s.text(m6[0] - 8, (m6[1] + ry + rh) / 2 + 6, "≥ 1 m", 16, th.fg, "end")
    # Minimum source-receiver distance guideline.
    s.line(rx + 70, ry + 70, m1[0], m1[1], th.primary, 1.3, dash="4,4")

    # Legend + ISO 3382-1 rules, to the right of the plan.
    lx = rx + rw + 24
    s.circle(lx + 8, ry + 16, 7, th.secondary, th.fg, 1.4)
    s.text(lx + 24, ry + 22, "Microphone position", 17, th.fg, "start")
    s.rect(lx, ry + 40, 16, 14, th.primary, th.fg, rx=2, sw=1.4)
    s.text(lx + 24, ry + 52, "Loudspeaker source", 17, th.fg, "start")
    for i, line in enumerate((
        "ISO 3382-1 (positions):",
        "• ≥ 2 source positions",
        "• mics ≥ 2 m apart",
        "• ≥ 1 m from surfaces",
        "• mic height 1.2 m",
        "d_min = 2√(V/cT)",
    )):
        bold = i == 0 or line.startswith("d_min")
        s.text(lx, ry + 88 + i * 30, line, 17, th.fg, "start", bold=bold)

    # --- ISO 3382-2 Table 1: minimum measurement positions per grade ---------
    ty = ry + rh + 46.0
    s.text(60, ty - 14, "ISO 3382-2 — reverberation-time measurement grades",
           20, th.fg, "start", bold=True)
    cols = [
        (70.0, "Method", "start"),
        (330.0, "Source pos.", "middle"),
        (470.0, "Mic pos.", "middle"),
        (630.0, "Source–mic comb.", "middle"),
        (820.0, "Decays / comb.", "middle"),
    ]
    rows = [
        ("Survey", "≥ 1", "≥ 2", "2", "1"),
        ("Engineering", "≥ 2", "≥ 2", "6", "2"),
        ("Precision", "≥ 2", "≥ 3", "12", "3"),
    ]
    tw, th_row = 840.0, 40.0
    s.rect(60, ty, tw, th_row * (len(rows) + 1), "none", th.fg, rx=6, sw=1.8)
    s.rect(60, ty, tw, th_row, th.panel, th.fg, rx=6, sw=1.8)
    for cx, label, anchor in cols:
        s.text(cx, ty + 26, label, 17, th.fg, anchor, bold=True)
    for r, row in enumerate(rows):
        yy = ty + th_row * (r + 1)
        if r < len(rows) - 1:
            s.line(60, yy + th_row, 60 + tw, yy + th_row, th.muted, 1.0)
        for (cx, _, anchor), value in zip(cols, row):
            col = th.primary if cx == 70.0 else th.fg
            s.text(cx, yy + 26, value, 17, col, anchor, bold=(cx == 70.0))


def _d_room_noise(s: SVG, th: Theme) -> None:
    """Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II.

    From a single octave-band spectrum, two parallel lanes: the NC tangency
    method (Table 1) and the RC Mark II rating and spectral tag (Annex D).
    """
    # --- Shared input spectrum ----------------------------------------------
    cx = 450.0
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Octave-band sound pressure levels  L(f)", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "16 Hz – 8000 Hz", 15, th.muted, "middle")

    lxc, rxc = 232.0, 668.0
    s.arrow(cx, 118, lxc, 158, th.fg, 1.8)
    s.arrow(cx, 118, rxc, 158, th.fg, 1.8)

    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 14, th.muted, "middle")

    # --- Left lane: NC tangency method (Table 1) ----------------------------
    _step(lxc, 158, "NC — tangency method", "Table 1 curves", th.primary)
    _step(lxc, 256, "NC value in each band", "curve level = L(f) at that f", th.fg)
    _step(lxc, 354, "NC = highest curve touched", "note the governing band", th.fg)
    s.arrow(lxc, 220, lxc, 256, th.fg, 1.8)
    s.arrow(lxc, 318, lxc, 354, th.fg, 1.8)
    s.arrow(lxc, 416, lxc, 470, th.fg, 1.8)
    s.rect(lxc - bw / 2, 470, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(lxc, 505, "NC-NN (band)", 23, th.fg, "middle", bold=True)

    # --- Right lane: RC Mark II rating and tag (Annex D) ---------------------
    _step(rxc, 158, "RC Mark II  (Annex D)", "−5 dB/octave curves", th.secondary)
    _step(rxc, 256, "LMF = (L500 + L1000 + L2000) / 3", "RC = round(LMF)   (clause D.4)",
          th.fg)
    s.arrow(rxc, 220, rxc, 256, th.fg, 1.8)
    s.arrow(rxc, 318, rxc, 354, th.fg, 1.8)
    # Spectral-tag rule box (clause D.3).
    s.rect(rxc - bw / 2, 354, bw, 116, th.panel, th.fg, rx=10, sw=2)
    s.text(rxc, 379, "Spectral tag  (clause D.3)", 18, th.fg, "middle", bold=True)
    for i, line in enumerate((
        "R  rumble: a band ≤ 500 Hz exceeds RC by > 5 dB",
        "H  hiss: a band ≥ 1000 Hz exceeds RC by > 3 dB",
        "N  neutral: within both tolerances",
    )):
        s.text(rxc - bw / 2 + 18, 403 + i * 22, line, 14, th.fg, "start")
    s.arrow(rxc, 470, rxc, 490, th.fg, 1.8)
    s.rect(rxc - bw / 2, 490, bw, 58, "none", th.secondary, rx=10, sw=2.4)
    s.text(rxc, 525, "RC-NN(A)", 23, th.fg, "middle", bold=True)


def _d_hearing_threshold(s: SVG, th: Theme) -> None:
    """Hearing-threshold model: ISO 7029 age distribution + ISO 389-7 zero."""
    cx = 450.0
    # --- Inputs --------------------------------------------------------------
    iw, ih = 540.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Age Y,  sex,  population fractile Q", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "audiometric frequencies 125 Hz – 8000 Hz", 15, th.muted,
           "middle")
    s.arrow(cx, 118, cx, 152, th.fg, 1.8)

    bw, bh = 620.0, 60.0
    x0 = cx - bw / 2

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 26, l1, 18, th.fg, "middle", bold=True)
        s.text(cx, y + 47, l2, 14, th.muted, "middle")

    # --- ISO 7029 chain ------------------------------------------------------
    _step(152, "Median deviation from age 18   (ISO 7029, 4.2)",
          "dHmd = a · (Y − 18) ^ b   (Table 1, by sex)", th.primary)
    _step(244, "Spread su / sl   (ISO 7029, 4.3)",
          "degree-5 polynomials in (Y − 18)   (Tables 2–5)", th.fg)
    _step(336, "Fractile threshold   (ISO 7029, 4.4)",
          "dHQ = dHmd + z(Q) * s   (su if Q >= 0.5, else sl)", th.fg)
    s.arrow(cx, 212, cx, 244, th.fg, 1.8)
    s.arrow(cx, 304, cx, 336, th.fg, 1.8)
    s.arrow(cx, 396, cx, 430, th.fg, 1.8)

    # --- Output + ISO 389-7 reference ---------------------------------------
    s.rect(x0, 430, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 456, "Expected hearing threshold level (dB HL)", 19, th.fg,
           "middle", bold=True)
    s.text(cx, 476, "referenced to the audiometric zero", 14, th.primary,
           "middle")
    s.rect(x0, 506, bw, 52, th.panel, th.secondary, rx=10, sw=2)
    s.text(cx, 530, "Audiometric zero = ISO 389-7 reference threshold",
           17, th.fg, "middle", bold=True)
    s.text(cx, 549, "free-field / diffuse-field (Table 1) — the dB HL / dB SPL zero",
           14, th.muted, "middle")


def _d_uncertainty(s: SVG, th: Theme) -> None:
    """Two routes to measurement uncertainty (ISO/IEC Guide 98-3 and Suppl. 1).

    From a shared measurement model and its input estimates, two parallel
    lanes: the GUM law of propagation of uncertainty (clause 5) and the
    Monte Carlo propagation of distributions (Supplement 1, clause 7).
    """
    # --- Shared measurement model + inputs ----------------------------------
    cx = 450.0
    iw, ih = 560.0, 62.0
    s.rect(cx - iw / 2, 56, iw, ih, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 84, "Measurement model  y = f(x_1, …, x_N)", 20, th.fg,
           "middle", bold=True)
    s.text(cx, 106, "input estimates x_i with standard uncertainties u(x_i)",
           15, th.muted, "middle")

    lxc, rxc = 232.0, 668.0
    s.arrow(cx, 118, lxc, 158, th.fg, 1.8)
    s.arrow(cx, 118, rxc, 158, th.fg, 1.8)

    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 14, th.muted, "middle")

    # --- Left lane: GUM law of propagation (clause 5) -----------------------
    _step(lxc, 158, "Law of propagation  (GUM 5)",
          "sensitivity c_i = ∂f / ∂x_i", th.primary)
    _step(lxc, 250, "Combine in quadrature",
          "uc² = Σ c_i² u²(x_i) + correlation", th.fg)
    _step(lxc, 342, "Effective dof  (Annex G.4)",
          "v_eff — Welch–Satterthwaite", th.fg)
    s.arrow(lxc, 220, lxc, 250, th.fg, 1.8)
    s.arrow(lxc, 312, lxc, 342, th.fg, 1.8)
    s.arrow(lxc, 404, lxc, 434, th.fg, 1.8)
    s.rect(lxc - bw / 2, 434, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(lxc, 462, "U = k · uc", 22, th.fg, "middle", bold=True)
    s.text(lxc, 482, "k = t_p(v_eff)   (clause 6)", 14, th.muted, "middle")

    # --- Right lane: Monte Carlo (Supplement 1, clause 7) -------------------
    _step(rxc, 158, "Monte Carlo  (Suppl. 1, 7)",
          "draw x_i from its PDF g(x_i)", th.secondary)
    _step(rxc, 250, "Propagate M trials",
          "y_r = f(x_1r, …, x_Nr)", th.fg)
    _step(rxc, 342, "Sort {y_r}, take fractiles",
          "prob.-symmetric 95 % interval", th.fg)
    s.arrow(rxc, 220, rxc, 250, th.fg, 1.8)
    s.arrow(rxc, 312, rxc, 342, th.fg, 1.8)
    s.arrow(rxc, 404, rxc, 434, th.fg, 1.8)
    s.rect(rxc - bw / 2, 434, bw, 58, "none", th.secondary, rx=10, sw=2.4)
    s.text(rxc, 462, "coverage interval", 22, th.fg, "middle", bold=True)
    s.text(rxc, 482, "[y_low, y_high]   (clause 7.7)", 14, th.muted, "middle")


def _d_nihl(s: SVG, th: Theme) -> None:
    """Noise-induced hearing loss (ISO 1999:2013): NIPTS and HTLAN.

    Two converging lanes (the age component H (HTLA, database A = ISO 7029)
    and the noise component N (NIPTS, Formulae 2-7)) combine into the hearing
    threshold associated with age and noise (HTLAN, Formula 1).
    """
    cx = 450.0
    lxc, rxc = 232.0, 668.0
    bw, bh = 372.0, 62.0

    def _step(cxx: float, y: float, l1: str, l2: str, color: str) -> None:
        s.rect(cxx - bw / 2, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cxx, y + 27, l1, 18, th.fg, "middle", bold=True)
        if l2:
            s.text(cxx, y + 48, l2, 13, th.muted, "middle")

    # --- Inputs -------------------------------------------------------------
    _step(lxc, 56, "Age Y,  sex,  fractile Q", "database A = ISO 7029", th.fg)
    _step(rxc, 56, "Exposure L_EX,8h,  t years",
          "normalized to 8 h / 5 days", th.fg)

    # --- Left lane: age component H (HTLA) ----------------------------------
    s.arrow(lxc, 118, lxc, 150, th.fg, 1.8)
    _step(lxc, 150, "Age threshold  H  (HTLA)",
          "ISO 7029 fractile, dB", th.primary)

    # --- Right lane: noise component N (NIPTS) ------------------------------
    s.arrow(rxc, 118, rxc, 150, th.fg, 1.8)
    _step(rxc, 150, "Median NIPTS  N50  (6.3.1)",
          "N50 = [u + v·lg(t/t0)]·(L − L0)²", th.secondary)
    s.arrow(rxc, 212, rxc, 244, th.fg, 1.8)
    _step(rxc, 244, "Fractile NIPTS  N  (6.3.2)",
          "N = N50 + z·(du if z ≥ 0 else dl)", th.fg)

    # --- Converge into HTLAN ------------------------------------------------
    box_y = 372.0
    s.arrow(lxc, 212, cx - 118.0, box_y, th.fg, 1.8)
    s.arrow(rxc, 306, cx + 118.0, box_y, th.fg, 1.8)
    s.rect(cx - bw / 2, box_y, bw, 66, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, box_y + 29, "HTLAN   H' = H + N − H·N / 120", 20, th.fg,
           "middle", bold=True)
    s.text(cx, box_y + 51, "threshold from age and noise  (Formula 1, 6.1)",
           13, th.muted, "middle")


def _d_impulse_prominence(s: SVG, th: Theme) -> None:
    """Impulsive-sound prominence and the LAeq adjustment (NT ACOU 112:2002)."""
    cx = 450.0
    bw, bh = 640.0, 60.0
    x0 = cx - bw / 2

    # --- Input --------------------------------------------------------------
    s.rect(x0, 56, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 82, "A-weighted level history  L_pAF  (time weighting F)", 19,
           th.fg, "middle", bold=True)
    s.text(cx, 103, "an onset = a stretch where the gradient exceeds 10 dB/s "
           "(clauses 4.5-4.7)", 13, th.muted, "middle")
    s.arrow(cx, 116, cx, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 26, l1, 18, th.fg, "middle", bold=True)
        s.text(cx, y + 47, l2, 13, th.muted, "middle")

    _step(150, "Per impulse: onset rate OR and level difference LD",
          "OR = onset slope [dB/s],   LD = Le − Ls [dB]", th.primary)
    _step(242, "Predicted prominence  P   (clause 7, Formula 1)",
          "P = 3·lg(OR) + 2·lg(LD);   highest P over 30 min governs", th.fg)
    _step(334, "Adjustment  KI   (clause 8, Formula 2)",
          "KI = 1.8·(P − 5) dB for P > 5, else 0", th.secondary)
    s.arrow(cx, 210, cx, 242, th.fg, 1.8)
    s.arrow(cx, 302, cx, 334, th.fg, 1.8)
    s.arrow(cx, 394, cx, 426, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 426, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 452, "Rating level  LAr,T = 10·lg( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
           18, th.fg, "middle", bold=True)
    s.text(cx, 473, "impulse-adjusted level over the reference time  (Note 1)",
           13, th.muted, "middle")


def _d_multiple_shock(s: SVG, th: Theme) -> None:
    """Multiple-shock spinal-response dose and injury risk (ISO 2631-5:2018)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Input --------------------------------------------------------------
    s.rect(x0, 48, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 72, "Vertical seat acceleration  az(t)", 19, th.fg, "middle",
           bold=True)
    s.text(cx, 92, "band-limited per ISO 2631-1  (0.4 Hz to 100 Hz)", 13,
           th.muted, "middle")
    s.arrow(cx, 106, cx, 136, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(136, "Spinal response  Az(t)  (clause 5.2, Formula 1/2)",
          "seat-to-spine transfer function H(f): 1 zero, 6 poles", th.primary)
    _step(224, "Acceleration dose  Dz = 1.07·(Σ Az,i^6)^(1/6)  (Formula 3)",
          "Az,i = positive peaks;   daily dose Dzd = Dz·(td/tm)^(1/6)", th.fg)
    _step(312, "Compressive stress  Sd = mz·Dzd  (Annex C, Formula C.1)",
          "mz = 0.029 (male) / 0.025 (female) MPa per m/s²", th.fg)
    _step(400, "Stress variable  R = [Σ (Sd·N^(1/6) / (Su − Sstat))^6]^(1/6)",
          "Su = 6.75 − Sage·(b+i) MPa, cumulated over exposure years (C.3/C.4)",
          th.secondary)
    for y0, y1 in ((196, 224), (284, 312), (372, 400), (460, 488)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 488, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 513, "Injury probability  P(R) = 1 − exp(−(R/α)^β)  (Formula C.5)",
           17, th.fg, "middle", bold=True)
    s.text(cx, 533, "Weibull risk of lumbar injury, by sex (Table C.1/C.2)", 13,
           th.muted, "middle")


def _d_enclosed_space_absorption(s: SVG, th: Theme) -> None:
    """Absorption area and reverberation time of a room (EN 12354-6:2003)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Inputs (two feeder boxes) -----------------------------------------
    iw = 320.0
    s.rect(cx - bw / 2, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx - bw / 2 + iw / 2, 72, "Surfaces  (Si, αs,i)", 17, th.fg, "middle",
           bold=True)
    s.text(cx - bw / 2 + iw / 2, 92, "area and absorption per band", 13, th.muted,
           "middle")
    s.rect(cx + bw / 2 - iw, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx + bw / 2 - iw / 2, 72, "Objects  (Vobj)", 17, th.fg, "middle",
           bold=True)
    s.text(cx + bw / 2 - iw / 2, 92, "Aobj = Vobj^(2/3)  (Formula 4)", 13,
           th.muted, "middle")
    s.arrow(cx - bw / 2 + iw / 2, 106, cx - 60, 150, th.fg, 1.8)
    s.arrow(cx + bw / 2 - iw / 2, 106, cx + 60, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(150, "Equivalent absorption area  A  (clause 4.3, Formula 1)",
          "A = Σ αs,i·Si + Σ Aobj + Aair;   Aair = 4·m·V·(1 − ψ)  (Formula 2)",
          th.primary)
    _step(238, "Object fraction  ψ = Σ Vobj / V   (Formula 3)",
          "air absorption negligible below 1 kHz for V < 200 m³", th.fg)
    s.arrow(cx, 210, cx, 238, th.fg, 1.8)
    s.arrow(cx, 296, cx, 324, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 324, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 349, "Reverberation time  T = 55.3/c₀ · V·(1 − ψ) / A  (Formula 5)",
           17, th.fg, "middle", bold=True)
    s.text(cx, 369, "c₀ = 345.6 m/s so 55.3/c₀ = 0.16  (clause 4.4)", 13,
           th.muted, "middle")


def _d_time_weighting(s: SVG, th: Theme) -> None:
    """Exponential-detector chain of the sound-level time weightings (IEC 61672-1)."""
    stages = [
        ("p(t)", "band signal", th.fg),
        ("( · )²", "square", th.primary),
        ("one-pole RC", "time constant τ", th.primary),
        ("10·lg(·/p₀²)", "to decibels", th.accent),
        ("L_τ(t)", "time-weighted level", th.secondary),
    ]
    bw, bh, gap = 150.0, 90.0, 12.0
    total = len(stages) * bw + (len(stages) - 1) * gap
    x = (900 - total) / 2
    y = 108.0
    last = len(stages) - 1
    for i, (title, sub, color) in enumerate(stages):
        fill = "none" if i in (0, last) else th.panel
        s.rect(x, y, bw, bh, fill, color, rx=12, sw=2.2)
        s.text(x + bw / 2, y + 38, title, 21, th.fg, "middle", bold=True)
        s.text(x + bw / 2, y + 64, sub, 14, color, "middle")
        if i < last:
            s.arrow(x + bw + 1, y + bh / 2, x + bw + gap - 2, y + bh / 2, th.fg, 2)
        x += bw + gap

    # Discrete realization of the detector.
    s.rect(130, 246, 640, 70, th.panel, th.muted, rx=10, sw=1.6)
    s.text(450, 275, "y[n] = α·x²[n] + (1 − α)·y[n−1],   α = 1 − e^(−1/(fs·τ))",
           18, th.fg, "middle", bold=True, mono=True)
    s.text(450, 299, "a first-order low-pass on the squared signal → the mean-square "
           "envelope", 14, th.muted, "middle")

    # The three standardized time constants.
    chips = [
        ("Fast (F)", "τ = 125 ms", th.primary),
        ("Slow (S)", "τ = 1000 ms", th.accent),
        ("Impulse (I)", "35 ms rise · 1500 ms fall", th.secondary),
    ]
    cw, cgap = 210.0, 15.0
    cx = (900 - (len(chips) * cw + (len(chips) - 1) * cgap)) / 2
    for title, sub, color in chips:
        s.rect(cx, 350, cw, 74, "none", color, rx=10, sw=2.2)
        s.text(cx + cw / 2, 380, title, 18, th.fg, "middle", bold=True)
        s.text(cx + cw / 2, 404, sub, 14, th.muted, "middle")
        cx += cw + cgap


def _d_block_processing(s: SVG, th: Theme) -> None:
    """Streaming block processing: carrying the filter state versus resetting it."""
    import math

    x0, blk_w, nblk, amp = 150.0, 190.0, 3, 66.0

    def _lane(gy: float, reset: bool, color: str) -> None:
        s.line(x0, gy, x0 + nblk * blk_w, gy, th.muted, 1.4)
        for k in range(nblk + 1):
            bx = x0 + k * blk_w
            s.line(bx, gy - amp - 16, bx, gy + 12, th.muted, 1.0, dash="3,4")
        for k in range(nblk):
            pts = []
            for j in range(31):
                frac = j / 30.0
                t = frac if reset else (k + frac)
                v = 1.0 - math.exp(-t / 0.9)
                pts.append((x0 + (k + frac) * blk_w, gy - amp * v))
            d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
            s.path(d, stroke=color, sw=2.6)
            s.text(x0 + (k + 0.5) * blk_w, gy + 30, f"block {k + 1}", 13,
                   th.muted, "middle")
        if reset:
            # Mark the discontinuity where each block restarts from rest.
            v_end = 1.0 - math.exp(-1.0 / 0.9)
            for k in range(1, nblk):
                bx = x0 + k * blk_w
                s.line(bx, gy - amp * v_end, bx, gy, th.secondary, 1.6, dash="2,3")
        else:
            # A small tag shows the carried state seeding the next block.
            for k in range(1, nblk):
                bx = x0 + k * blk_w
                s.rect(bx - 27, gy - amp - 40, 54, 22, th.bg, color, rx=6, sw=1.4)
                s.text(bx, gy - amp - 25, "y[-1]", 12, th.fg, "middle", mono=True)

    s.text(450, 62, "State carried across blocks — TimeWeighting.process()", 19,
           th.fg, "middle", bold=True)
    s.text(450, 84, "y[-1] (or the sosfilt zi vector) seeds the next block → identical "
           "to one continuous call", 13, th.muted, "middle")
    _lane(200.0, reset=False, color=th.primary)

    s.text(450, 300, "State reset each block — reset() or a fresh call", 19,
           th.fg, "middle", bold=True)
    s.text(450, 322, "every block restarts from rest → spurious discontinuities at "
           "the seams", 13, th.muted, "middle")
    _lane(430.0, reset=True, color=th.secondary)


def _d_multichannel(s: SVG, th: Theme) -> None:
    """How array shapes flow through a per-channel operation (time axis last)."""
    cell = 22.0

    def _grid(gx: float, gy: float, rows: int, cols: int, color: str) -> None:
        for r in range(rows):
            for c in range(cols):
                s.rect(gx + c * cell, gy + r * cell, cell, cell, th.panel, color, sw=1.3)

    # 1-D lane.
    _grid(64, 120, 1, 8, th.primary)
    s.text(64 + 4 * cell, 108, "1-D:  (samples,)", 15, th.fg, "middle", bold=True)
    s.rect(610, 120, cell, cell, "none", th.accent, sw=2)
    s.text(610 + cell / 2, 108, "scalar", 15, th.fg, "middle", bold=True)

    # 2-D lane.
    _grid(64, 250, 3, 8, th.primary)
    s.text(64 + 4 * cell, 238, "2-D:  (channels, samples)", 15, th.fg, "middle",
           bold=True)
    for r in range(3):
        s.rect(610, 250 + r * cell, cell, cell, "none", th.accent, sw=2)
    s.text(610 + cell / 2, 238, "(channels,)", 15, th.fg, "middle", bold=True)

    # Shared processing box.
    s.rect(360, 96, 190, 200, th.panel, th.fg, rx=12, sw=2)
    s.text(455, 178, "reduce along", 17, th.fg, "middle", bold=True)
    s.text(455, 202, "axis = −1  (time)", 17, th.primary, "middle", bold=True, mono=True)
    s.text(455, 236, "the channel axis 0", 14, th.muted, "middle")
    s.text(455, 256, "rides through untouched", 14, th.muted, "middle")

    s.arrow(64 + 8 * cell + 4, 131, 358, 150, th.fg, 1.6)
    s.arrow(64 + 8 * cell + 4, 283, 358, 244, th.fg, 1.6)
    s.arrow(552, 150, 606, 131, th.fg, 1.6)
    s.arrow(552, 244, 606, 283, th.fg, 1.6)

    s.text(450, 350, "A mono call returns a scalar; a C-channel call returns C results.",
           15, th.fg, "middle")
    s.text(450, 374, "Band metrics widen the reduced axis instead: (…, bands).",
           14, th.muted, "middle")


def _d_open_plan(s: SVG, th: Theme) -> None:
    """ISO 3382-3 open-plan measurement line and its single-number quantities."""
    ly = 150.0
    lx0, lx1 = 120.0, 812.0
    # Talker/source near the origin.
    s.person(lx0, ly, h=70)
    s.text(lx0, ly + 22, "source", 13, th.muted, "middle")
    s.text(lx0, ly + 40, "(r₀ = 1 m)", 13, th.muted, "middle")
    # Measurement line with workstations and positions.
    s.line(lx0 + 26, ly - 30, lx1, ly - 30, th.fg, 1.8, dash="6,5")
    dists = [(0.18, "2 m"), (0.36, "4 m"), (0.56, "8 m"), (0.78, "12 m"), (0.98, "16 m")]
    for frac, lab in dists:
        px = lx0 + 26 + frac * (lx1 - lx0 - 26)
        s.rect(px - 22, ly + 4, 44, 26, th.panel, th.muted, rx=4, sw=1.3)  # desk
        s.circle(px, ly - 30, 5, th.primary)  # measurement position
        s.text(px, ly - 42, lab, 13, th.fg, "middle")
    # Evaluation-range bracket (2 m to 16 m).
    bx0 = lx0 + 26 + 0.18 * (lx1 - lx0 - 26)
    bx1 = lx0 + 26 + 0.98 * (lx1 - lx0 - 26)
    s.line(bx0, ly + 52, bx1, ly + 52, th.accent, 1.6)
    s.line(bx0, ly + 46, bx0, ly + 58, th.accent, 1.6)
    s.line(bx1, ly + 46, bx1, ly + 58, th.accent, 1.6)
    s.text((bx0 + bx1) / 2, ly + 74, "spatial-decay fit range (2 m to 16 m)", 14,
           th.accent, "middle")

    chips = [
        ("D₂,S", "spatial decay rate", "dB per doubling · Cl. 6.2", th.primary),
        ("Lp,A,S,4m", "speech level at 4 m", "A-weighted · Cl. 3.3", th.primary),
        ("rD", "distraction distance", "fitted STI = 0.50 · Cl. 3.6", th.secondary),
        ("rP", "privacy distance", "fitted STI = 0.20 · Cl. 3.7", th.secondary),
    ]
    cw, cgap = 190.0, 14.0
    cx = (900 - (len(chips) * cw + (len(chips) - 1) * cgap)) / 2
    for sym, name, note, color in chips:
        s.rect(cx, 320, cw, 118, th.panel, color, rx=10, sw=2)
        s.text(cx + cw / 2, 356, sym, 22, th.fg, "middle", bold=True)
        s.text(cx + cw / 2, 384, name, 15, color, "middle", bold=True)
        s.text(cx + cw / 2, 412, note, 12, th.muted, "middle")
        cx += cw + cgap


def _d_iso12999(s: SVG, th: Theme) -> None:
    """ISO 12999-1 uncertainty: from tabulated reproducibility to the expanded U."""
    cx = 450.0
    bw, bh = 664.0, 60.0
    x0 = cx - bw / 2

    s.rect(x0, 48, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 72, "Standard uncertainty  u  — reproducibility read from the tables",
           18, th.fg, "middle", bold=True)
    s.text(cx, 92, "bands: Tables 2/4 · ratings: Tables 3/5 · situation A (σR) / "
           "B (σsitu) / C (σr)", 13, th.muted, "middle")
    s.arrow(cx, 108, cx, 138, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(138, "Reduce by  m  independent measurements   u/√m   (Formula A.7)",
          "and combine model with reality per Annex A when predicting", th.fg)
    _step(226, "Combine uncorrelated contributions   uc = √(Σ u_i²)   (Formula C.2)",
          "single-number combination of Annex B uses Formula B.2", th.primary)
    _step(314, "Expand   U = k·u   (Formula 2),   k from Table 8   (k ≥ 1)",
          "the coverage factor depends on the reported quantity and situation",
          th.secondary)
    for y0, y1 in ((198, 226), (286, 314)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 374, cx, 404, th.fg, 1.8)

    # Two-sided reporting vs one-sided conformity.
    hw = 320.0
    s.rect(x0, 404, hw, 66, "none", th.primary, rx=10, sw=2.2)
    s.text(x0 + hw / 2, 430, "Report   Y = y ± U   (Formula 3)", 16, th.fg,
           "middle", bold=True)
    s.text(x0 + hw / 2, 452, "two-sided coverage factor", 13, th.muted, "middle")
    s.rect(cx + bw / 2 - hw, 404, hw, 66, "none", th.secondary, rx=10, sw=2.2)
    s.text(cx + bw / 2 - hw / 2, 430, "Declare conformity   (Formulae 4/5)", 16,
           th.fg, "middle", bold=True)
    s.text(cx + bw / 2 - hw / 2, 452, "one-sided coverage factor", 13, th.muted, "middle")


def _d_iso11654(s: SVG, th: Theme) -> None:
    """ISO 11654 single-number absorption rating: from αs to the absorption class."""
    cx = 450.0
    bw, bh = 664.0, 54.0
    x0 = cx - bw / 2

    s.rect(x0, 46, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 68, "Measured  αs  at one-third octaves, 200 Hz to 5000 Hz", 18,
           th.fg, "middle", bold=True)
    s.text(cx, 88, "from a reverberation room (ISO 354)", 13, th.muted, "middle")
    s.arrow(cx, 100, cx, 128, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 23, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 42, l2, 13, th.muted, "middle")

    _step(128, "Practical  αp  per octave band, 250 Hz to 4000 Hz  (Clause 4.1)",
          "mean of the three one-third octaves, rounded to 0.05", th.primary)
    _step(206, "Shift the reference curve in 0.05 steps to best fit  (Clause 4.2)",
          "sum of unfavourable deviations kept ≤ 0.10", th.fg)
    _step(284, "Weighted coefficient  αw = shifted reference at 500 Hz", "", th.fg)
    _step(362, "Shape indicators (L, M, H) where  αp − reference ≥ 0.25", "", th.secondary)
    for y0, y1 in ((100, 128), (182, 206), (260, 284), (338, 362)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 416, cx, 444, th.fg, 1.8)

    s.rect(x0, 444, bw, 58, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 469, "Sound absorption class  A to E   (Table B.1, Annex B)", 17,
           th.fg, "middle", bold=True)
    s.text(cx, 489, "or “Not classified” when αw falls below the class-E band",
           13, th.muted, "middle")


def _d_zwicker(s: SVG, th: Theme) -> None:
    """ISO 532-1 Zwicker loudness: from band levels to N (sone) and LN (phon)."""
    cx = 450.0
    bw, bh = 668.0, 58.0
    x0 = cx - bw / 2

    s.rect(x0, 46, bw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(cx, 70, "28 one-third-octave band levels, 25 Hz to 12.5 kHz", 18,
           th.fg, "middle", bold=True)
    s.text(cx, 90, "from a spectrum, or from a calibrated signal via the Annex A "
           "filterbank", 13, th.muted, "middle")
    s.arrow(cx, 104, cx, 132, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(132, "Equal-loudness correction and lower critical bands  "
          "(Clause 5.4, Table A.3)",
          "the 11 lowest bands grouped into 3 critical bands, 25-250 Hz",
          th.primary)
    _step(218, "Core loudness of the 20 critical bands  (Tables A.4-A.7)",
          "a₀ transmission (A.4), diffuse-field DDF (A.5), threshold in quiet "
          "LTQ (A.6)", th.fg)
    _step(304, "Specific loudness  N′(z)  over 0.1-Bark steps to 24 Bark",
          "upper masking slopes added band to band (Table A.9)", th.secondary)
    for y0, y1 in ((190, 218), (276, 304)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)
    s.arrow(cx, 362, cx, 392, th.fg, 1.8)

    s.rect(x0, 392, bw, 60, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 417, "Total loudness  N = ∫ N′(z) dz  [sone]", 17, th.fg, "middle",
           bold=True)
    s.text(cx, 438, "loudness level  LN = 40 + 10·log₂ N  [phon]", 14, th.muted,
           "middle")


# ISO 226:2023 Table 1 (p. 4): frequency Hz -> (alpha_f, L_U dB, T_f dB), the
# same parameters the library's ``equal_loudness_contour`` implements.
_ISO226_TABLE1: tuple[tuple[float, float, float, float], ...] = (
    (20.0, 0.635, -31.5, 78.1), (25.0, 0.602, -27.2, 68.7),
    (31.5, 0.569, -23.1, 59.5), (40.0, 0.537, -19.3, 51.1),
    (50.0, 0.509, -16.1, 44.0), (63.0, 0.482, -13.1, 37.5),
    (80.0, 0.456, -10.4, 31.5), (100.0, 0.433, -8.2, 26.5),
    (125.0, 0.412, -6.3, 22.1), (160.0, 0.391, -4.6, 17.9),
    (200.0, 0.373, -3.2, 14.4), (250.0, 0.357, -2.1, 11.4),
    (315.0, 0.343, -1.2, 8.6), (400.0, 0.330, -0.5, 6.2),
    (500.0, 0.320, 0.0, 4.4), (630.0, 0.311, 0.4, 3.0),
    (800.0, 0.303, 0.5, 2.2), (1000.0, 0.300, 0.0, 2.4),
    (1250.0, 0.295, -2.7, 3.5), (1600.0, 0.292, -4.2, 1.7),
    (2000.0, 0.290, -1.2, -1.3), (2500.0, 0.290, 1.4, -4.2),
    (3150.0, 0.289, 2.3, -6.0), (4000.0, 0.289, 1.0, -5.4),
    (5000.0, 0.289, -2.3, -1.5), (6300.0, 0.293, -7.2, 6.0),
    (8000.0, 0.303, -11.2, 12.6), (10000.0, 0.323, -10.9, 13.9),
    (12500.0, 0.354, -3.5, 12.3),
)


def _iso226_spl(alpha_f: float, l_u: float, t_f: float, phon: float) -> float:
    """ISO 226:2023 Formula (1): SPL of a pure tone at loudness level ``phon``."""
    import math
    term = (4.0e-10) ** (0.3 - alpha_f) * (10 ** (0.03 * phon) - 10 ** 0.072) \
        + 10 ** (alpha_f * (t_f + l_u) / 10)
    return 10 / alpha_f * math.log10(term) - l_u


def _a_weight_db(f: float) -> float:
    """IEC 61672-1 Annex E analytic A-weighting, normalized to 0 dB at 1 kHz."""
    import math

    def gain(x: float) -> float:
        f1, f2, f3, f4 = 20.599, 107.653, 737.862, 12194.217
        return (f4 ** 2 * x ** 4) / ((x ** 2 + f1 ** 2)
                                     * math.sqrt((x ** 2 + f2 ** 2) * (x ** 2 + f3 ** 2))
                                     * (x ** 2 + f4 ** 2))

    return 20 * math.log10(gain(f) / gain(1000.0))


def _d_equal_loudness_weighting(s: SVG, th: Theme) -> None:
    """Equal-loudness contours (ISO 226) inverted into the A-curve (IEC 61672-1)."""
    import math

    f_lo, f_hi = 20.0, 12500.0

    def make_fx(px0: float, px1: float) -> Callable[[float], float]:
        span = math.log10(f_hi) - math.log10(f_lo)

        def fx(f: float) -> float:
            return px0 + (math.log10(f) - math.log10(f_lo)) / span * (px1 - px0)

        return fx

    def axes(bx0: float, by0: float, bx1: float, by1: float,
             fx: Callable[[float], float]) -> None:
        s.rect(bx0, by0, bx1 - bx0, by1 - by0, "none", th.muted, sw=1.2)
        for f, lab in ((20.0, "20"), (100.0, "100"), (1000.0, "1k"), (10000.0, "10k")):
            x = fx(f)
            s.line(x, by1, x, by1 + 5, th.muted, 1.2)
            s.text(x, by1 + 21, lab, 13, th.muted, "middle")
        s.text((bx0 + bx1) / 2, by1 + 42, "Frequency [Hz]", 14, th.fg, "middle")

    # --- Left panel: the equal-loudness contours ----------------------------
    lx0, lx1, ly0, ly1 = 70.0, 390.0, 110.0, 430.0
    l_fx = make_fx(lx0, lx1)

    def l_fy(db: float) -> float:  # 0 dB at the bottom, 120 dB at the top
        return ly1 - db / 120.0 * (ly1 - ly0)

    s.text((lx0 + lx1) / 2, 92, "Equal-loudness contours (ISO 226)", 17, th.fg,
           "middle", bold=True)
    axes(lx0, ly0, lx1, ly1, l_fx)
    for db in (0, 40, 80, 120):
        y = l_fy(db)
        s.line(lx0 - 5, y, lx0, y, th.muted, 1.2)
        s.text(lx0 - 9, y + 5, str(db), 13, th.muted, "end")
    s.text(lx0 - 20, ly0 - 12, "dB SPL", 12, th.muted, "middle")

    for phon in (20, 40, 60, 80):
        main = phon == 40
        color = th.primary if main else th.muted
        pts = [(l_fx(f), l_fy(_iso226_spl(a, lu, tf, phon)))
               for f, a, lu, tf in _ISO226_TABLE1]
        d = "M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        s.path(d, stroke=color, sw=2.8 if main else 1.5)
        # Label each contour above its 160 Hz point, where the curves spread.
        yl = l_fy(_iso226_spl(0.391, -4.6, 17.9, phon)) - 10
        if main:
            s.text(l_fx(160.0), yl, "40 phon", 14, th.primary, "middle", bold=True)
        else:
            s.text(l_fx(160.0), yl, str(phon), 12, th.muted, "middle")

    # --- Middle: the inversion step ------------------------------------------
    s.text(462, 226, "invert", 16, th.fg, "middle", bold=True)
    s.text(462, 248, "0 dB at 1 kHz", 13, th.muted, "middle")
    s.arrow(400, 272, 546, 272, th.fg, 2.2)

    # --- Right panel: inverted contour vs the A-curve ------------------------
    rx0, rx1 = 558.0, 872.0
    r_fx = make_fx(rx0, rx1)

    def r_fy(db: float) -> float:  # +10 dB at the top, -70 dB at the bottom
        return ly0 + (10.0 - db) / 80.0 * (ly1 - ly0)

    s.text((rx0 + rx1) / 2, 92, "A-weighting (IEC 61672-1)", 17, th.fg,
           "middle", bold=True)
    axes(rx0, ly0, rx1, ly1, r_fx)
    for db in (0, -20, -40, -60):
        y = r_fy(db)
        s.line(rx0 - 5, y, rx0, y, th.muted, 1.2)
        s.text(rx0 - 9, y + 5, str(db), 13, th.muted, "end")
    s.text(rx0 - 20, ly0 - 12, "dB", 12, th.muted, "middle")
    s.line(rx0, r_fy(0.0), rx1, r_fy(0.0), th.muted, 0.9, dash="3,4")

    # Inverted 40-phon contour, relative to its 1 kHz value (dashed reference).
    inv = [(r_fx(f), r_fy(-(_iso226_spl(a, lu, tf, 40.0) - 40.0)))
           for f, a, lu, tf in _ISO226_TABLE1]
    s.path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in inv),
           stroke=th.secondary, sw=2.0, dash="6,5")
    # The A-curve itself, sampled densely on the log axis.
    n = 60
    aw = [(r_fx(f), r_fy(_a_weight_db(f)))
          for f in (f_lo * (f_hi / f_lo) ** (i / (n - 1)) for i in range(n))]
    s.path("M " + " L ".join(f"{px:.1f} {py:.1f}" for px, py in aw),
           stroke=th.primary, sw=2.8)

    # Legend, bottom right of the panel (the curves live top left).
    s.line(636, 386, 668, 386, th.primary, 2.8)
    s.text(676, 391, "A-weighting (IEC 61672-1)", 13, th.fg, "start")
    s.line(636, 408, 668, 408, th.secondary, 2.0, dash="6,5")
    s.text(676, 413, "inverted 40-phon contour", 13, th.fg, "start")

    # --- Footer ---------------------------------------------------------------
    s.text(450, 507, "A is the 40-phon contour flipped into a realizable filter: "
           "quiet sounds, where the ear discards bass hardest.", 14, th.fg, "middle")
    s.text(450, 529, "The match is deliberately loose (a 1930s convention, not a "
           "loudness model); C mirrors the flatter ~100-phon contour.", 13,
           th.muted, "middle")


def _d_loudspeaker_freefield(s: SVG, th: Theme) -> None:
    """IEC 60268-5 loudspeaker sensitivity on the reference axis (free field)."""
    x0, y0, x1, gy = 60.0, 70.0, 840.0, 470.0
    s.rect(x0, y0, x1 - x0, gy - y0, th.bg, th.fg, sw=3)

    # Anechoic wedges on all four boundaries (full free field: no floor).
    for wx in range(int(x0) + 4, int(x1) - 36, 40):
        s.path(f"M {wx} {y0} L {wx + 40} {y0} L {wx + 20} {y0 + 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {wx} {gy} L {wx + 40} {gy} L {wx + 20} {gy - 28} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    for wy in range(int(y0) + 30, int(gy) - 64, 40):
        s.path(f"M {x0} {wy} L {x0} {wy + 40} L {x0 + 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
        s.path(f"M {x1} {wy} L {x1} {wy + 40} L {x1 - 28} {wy + 20} Z",
               fill=th.panel, stroke=th.muted, sw=1.0)
    s.text(210, 122, "Anechoic wedges", 15, th.muted, anchor="start")

    # Loudspeaker cabinet on a stand, reference point on the front baffle.
    ax_y, fx = 275.0, 250.0
    s.line(219, ax_y + 70, 219, 462, th.fg, 2.2)
    s.line(199, 462, 239, 462, th.fg, 2.2)
    s.rect(fx - 62, ax_y - 70, 62, 140, th.panel, th.primary, rx=6, sw=2)
    s.circle(fx - 18, ax_y, 14, th.primary)
    s.circle(fx - 18, ax_y, 5.5, th.bg)
    s.text(219, ax_y - 84, "Loudspeaker", 18, th.fg, bold=True)
    for r in (26, 44, 62):
        s.path(f"M {fx + r * 0.34:.1f} {ax_y - r * 0.94:.1f} "
               f"A {r} {r} 0 0 1 {fx + r * 0.34:.1f} {ax_y + r * 0.94:.1f}",
               stroke=th.accent, sw=1.5)

    # Reference axis through the reference point, out to the right.
    s.circle(fx, ax_y, 3.4, th.fg)
    s.line(fx, ax_y, 782, ax_y, th.muted, 1.4, dash="7,5")
    s.arrow(760, ax_y, 792, ax_y, th.muted, 1.4)
    s.text(724, ax_y + 24, "Reference axis", 15, th.muted)

    # Measurement microphone on axis, capsule facing the loudspeaker.
    mx = 620.0
    s.line(mx + 23, ax_y + 6, mx + 23, 462, th.fg, 2.2)
    s.line(mx + 7, 462, mx + 39, 462, th.fg, 2.2)
    s.rect(mx, ax_y - 6, 46, 12, th.primary, rx=4)
    s.rect(mx - 12, ax_y - 4, 12, 8, th.fg, rx=2.5)
    s.text(mx + 24, ax_y - 24, "Measurement microphone", 17, th.fg, bold=True)

    # Reference distance, drafting style, between baffle and capsule tip.
    s.dim(fx, ax_y, mx - 12, ax_y, "r = 1 m", offset=92)

    # Drive: amplifier delivering 1 W into the rated impedance.
    s.rect(85, 383, 140, 54, th.panel, th.primary, rx=8, sw=2)
    s.text(155, 405, "Amplifier", 17, th.fg, bold=True)
    s.text(155, 427, "2.83 V (8 Ω)", 15, th.secondary, mono=True)
    s.line(155, 383, 155, 345, th.fg, 1.6)
    s.line(155, 345, fx - 62, 345, th.fg, 1.6)

    # Governing relations.
    for y, txt, col, bold in (
        (508, "Characteristic sensitivity: Lp at 1 m for 1 W into the rated impedance",
         th.fg, True),
        (534, "Up = √(R · 1 W): 2.83 V is 1 W into 8 Ω but 2 W into 4 Ω (+3 dB)",
         th.secondary, True),
        (559, "Lp(1 m) = Lp(r) + 20 lg(r / 1 m)   (far field, inverse-distance law)",
         th.primary, True),
        (583, "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 lg(M / 1 V/Pa) dB",
         th.muted, False),
    ):
        s.text(450, y, txt, 19 if bold else 18, col, bold=bold)


def _d_dosimeter(s: SVG, th: Theme) -> None:
    """ISO 9612 occupational exposure: worn-dosimeter microphone position
    (Clause 12.3) and the three measurement strategies (Clauses 9-11)."""
    # --- Left: worker with a shoulder-mounted personal exposimeter ---------
    s.text(195, 84, "Worn instrument (Clause 12.3)", 21, th.fg, bold=True)
    gy = 560.0
    s.ground(gy, 40, 330)
    px = 150.0
    s.person(px, gy, 300)
    head_y = gy - 300 + 30.0            # head-circle centre
    sh_y = gy - 300 * 0.75              # shoulder joint (arm attachment)

    # Microphone capsule ~0.04 m above the shoulder, on the most-exposed side.
    mx = px + 46.0
    cap_y = sh_y - 30.0
    s.line(px + 6, sh_y - 6, mx + 12, sh_y + 6, th.muted, 2.4)  # shoulder slope
    s.rect(mx - 5, cap_y, 10, 14, th.fg, rx=3)                  # capsule
    s.line(mx, cap_y + 14, mx, sh_y, th.primary, 2.2)           # stub mount
    # Cable from the capsule mount to the body-worn meter.
    s.path(f"M {mx:.0f} {sh_y:.0f} C {mx + 26:.0f} {sh_y + 56:.0f} "
           f"{px + 40:.0f} {gy - 130:.0f} {px + 26:.0f} {gy - 116:.0f}",
           stroke=th.muted, sw=1.6)
    s.rect(px + 12, gy - 118, 30, 44, th.panel, th.primary, rx=5, sw=2)
    s.circle(px + 27, gy - 104, 3.5, th.primary)
    s.text(185, gy + 44, "Personal sound exposure meter", 19, th.fg)
    s.text(185, gy + 68, "(IEC 61252)", 17, th.muted)

    # Dimension: capsule height above the shoulder.
    s.dim(mx + 44, sh_y, mx + 44, cap_y, "≈ 0.04 m", offset=0, size=18,
          label_side="right")
    s.line(mx + 5, cap_y, mx + 44, cap_y, th.muted, 0.9, dash="3,3")
    s.line(mx + 12, sh_y + 2, mx + 44, sh_y, th.muted, 0.9, dash="3,3")
    s.text(mx + 53, sh_y + 22, "above the shoulder", 15, th.muted, "start")
    # Distance to the ear-canal entrance.
    s.line(px + 24, head_y + 8, mx - 4, cap_y + 4, th.secondary, 1.4,
           dash="5,4")
    s.text(px, head_y - 82, "≥ 0.1 m from the ear canal,", 17,
           th.secondary)
    s.text(px, head_y - 62, "most-exposed side", 17, th.secondary)

    # --- Right: the three sampling strategies as day timelines -------------
    s.text(620, 84, "Measurement strategies (Clauses 9–11)", 22, th.fg,
           bold=True)
    x0, x1 = 390.0, 850.0
    bw = x1 - x0
    ax_y = 132.0
    s.line(x0, ax_y, x1, ax_y, th.muted, 1.4)
    for hh in range(0, 9, 2):
        tx = x0 + bw * hh / 8.0
        s.line(tx, ax_y - 4, tx, ax_y + 4, th.muted, 1.4)
        s.text(tx, ax_y + 22, f"{hh} h", 15, th.muted, mono=True)
    s.text(620, ax_y - 12, "Working day", 17, th.muted)

    def strip(y: float, title: str, caption: str) -> None:
        s.text(x0, y - 10, title, 19, th.fg, "start", bold=True)
        s.text(x0, y + 68, caption, 16, th.muted, "start", italic=True)

    # Strategy 1: task-based; the day split into tasks, >= 3 samples each.
    y1 = 190.0
    strip(y1, "Task-based (Clause 9)",
          "split the day into tasks — ≥ 3 samples (│) per task, plus each duration")
    edges = [0.0, 0.1875, 0.8125, 1.0]      # the Annex D welder: 1.5 h / 5 h / 1.5 h
    cols = [th.accent, th.primary, th.secondary]
    for k in range(3):
        xa, xb = x0 + bw * edges[k], x0 + bw * edges[k + 1]
        s.rect(xa, y1, xb - xa, 44, th.panel, cols[k], rx=6, sw=2)
        s.text((xa + xb) / 2, y1 + 27, f"Task {k + 1}", 17, th.fg)
        for frac in (0.25, 0.5, 0.75):
            sx = xa + (xb - xa) * frac
            s.line(sx, y1 + 34, sx, y1 + 42, cols[k], 2.2)

    # Strategy 2: job-based; random samples over the homogeneous group.
    y2 = 300.0
    strip(y2, "Job-based (Clause 10)",
          "N ≥ 5 random samples over the homogeneous exposure group")
    s.rect(x0, y2, bw, 44, "none", th.muted, rx=6, sw=1.6, dash="5,4")
    for frac in (0.05, 0.24, 0.46, 0.65, 0.86):
        s.rect(x0 + bw * frac, y2 + 6, bw * 0.06, 32, th.panel, th.primary,
               rx=4, sw=2)

    # Strategy 3: full-day; the whole shift, repeated on several days.
    y3 = 410.0
    strip(y3, "Full-day (Clause 11)",
          "the whole shift, at least 3 times (5 if the days differ by > 3 dB)")
    s.rect(x0, y3, bw, 24, th.panel, th.primary, rx=6, sw=2)
    s.text(x0 + bw / 2, y3 + 17, "day 1", 14, th.fg)
    s.rect(x0 + 8, y3 + 30, bw - 16, 7, th.panel, th.primary, rx=3, sw=1.2)
    s.rect(x0 + 16, y3 + 43, bw - 32, 7, th.panel, th.primary, rx=3, sw=1.2)

    # All three land in the same deliverable.
    s.text(620, 520, "choose by work pattern (Table B.1)  →  LEX,8h + Annex C uncertainty",
           17, th.fg)


# ---------------------------------------------------------------------------
# Shared structure-borne rig parts (ISO 9052-1 / ISO 7626 / ISO 10846 /
# EN 15657 / EN 12354-5 diagrams)
# ---------------------------------------------------------------------------

def _spring_v(s: SVG, x: float, y1: float, y2: float, color: str,
              coils: int = 4, width: float = 13.0, sw: float = 2.2) -> None:
    """Vertical zig-zag spring between (x, y1) and (x, y2), straight leads."""
    lead = min(10.0, (y2 - y1) * 0.12)
    n = 2 * coils
    step = (y2 - y1 - 2 * lead) / n
    d = [f"M {x:.1f} {y1:.1f}", f"L {x:.1f} {y1 + lead:.1f}"]
    for k in range(1, n):
        dx = width if k % 2 else -width
        d.append(f"L {x + dx:.1f} {y1 + lead + k * step:.1f}")
    d.append(f"L {x:.1f} {y2 - lead:.1f}")
    d.append(f"L {x:.1f} {y2:.1f}")
    s.path(" ".join(d), stroke=color, sw=sw)


def _accel(s: SVG, x: float, y: float, size: float = 14.0) -> None:
    """Accelerometer block standing on the surface point (x, y)."""
    th = s.th
    s.rect(x - size / 2, y - size, size, size, th.secondary, th.fg, rx=2.5,
           sw=1.3)
    s.line(x, y - size, x, y - size - 8, th.fg, 1.3)


def _exciter(s: SVG, x: float, y: float, stinger: float = 22.0,
             w: float = 74.0, h: float = 48.0, up: bool = False) -> None:
    """Electrodynamic exciter body with a stinger driving the point (x, y).

    ``up=False`` hangs the body above the drive point (stinger down);
    ``up=True`` stands it below the drive point (stinger up).
    """
    th = s.th
    sgn = 1.0 if up else -1.0
    s.line(x, y, x, y + sgn * stinger, th.fg, 2.2)
    top = y + stinger if up else y - stinger - h
    s.rect(x - w / 2, top, w, h, th.panel, th.primary, rx=9, sw=2)


def _motion_arrows(s: SVG, x: float, cy: float, half: float, color: str,
                   sw: float = 2.0) -> None:
    """Up-down double arrow marking a vibratory motion at x, centred on cy."""
    s.arrow(x, cy, x, cy - half, color, sw)
    s.arrow(x, cy, x, cy + half, color, sw)


# ---------------------------------------------------------------------------
# Dynamic-stiffness resonance rig (ISO 9052-1 / EN 29052-1)
# ---------------------------------------------------------------------------

def _d_dynamic_stiffness_rig(s: SVG, th: Theme) -> None:
    """ISO 9052-1 rig: exciter and accelerometer on the load plate over the
    resilient specimen, read as a mass-spring resonance."""
    # ===== Left: rig cross-section =====
    s.text(240, 74, "Resonance rig", 22, th.fg, bold=True)
    gy = 466.0
    s.ground(gy, 50, 430)
    s.text(56, gy + 34, "Rigid foundation", 17, th.muted, anchor="start")

    x0, x1 = 150.0, 330.0
    spec_top, plate_h = 400.0, 26.0
    plate_top = spec_top - plate_h
    # Resilient specimen (soft diagonal hatching).
    s.rect(x0, spec_top, x1 - x0, gy - spec_top, th.panel, th.accent, sw=2)
    for hx in range(int(x0) + 14, int(x1) + 1, 22):
        s.line(hx, spec_top, hx - 12, gy, th.accent, 0.9)
    # Load plate on top of the specimen.
    s.rect(x0 - 12, plate_top, x1 - x0 + 24, plate_h, th.panel, th.primary,
           rx=3, sw=2.2)
    s.text(x1 + 26, plate_top + 19, "Load plate", 18, th.fg, anchor="start",
           bold=True)
    s.text(x1 + 26, plate_top + 43, "m′t = 200 kg/m²", 15, th.muted,
           anchor="start")
    s.text(x1 + 26, spec_top + 40, "Resilient specimen", 17, th.fg,
           anchor="start")
    s.text(x1 + 26, spec_top + 62, "200 mm × 200 mm", 15, th.muted,
           anchor="start")
    s.dim(x0, spec_top, x0, gy, "d", offset=-30, size=18)
    # Exciter, drive force and accelerometer on the plate.
    _exciter(s, 205.0, plate_top)
    s.text(205, plate_top - 100, "Exciter", 18, th.fg, bold=True)
    _motion_arrows(s, 256.0, plate_top - 36, 24.0, th.secondary)
    s.text(268, plate_top - 30, "F(t)", 16, th.secondary, anchor="start",
           mono=True)
    _accel(s, 300.0, plate_top)
    s.text(x1 + 26, plate_top - 14, "Accelerometer", 16, th.fg, anchor="start")
    s.line(309, plate_top - 8, x1 + 20, plate_top - 18, th.muted, 1.1,
           dash="3,3")

    # ===== Right: the mass-spring reading =====
    s.text(680, 74, "Mass-spring model", 22, th.fg, bold=True)
    mx = 680.0
    s.rect(mx - 60, 120, 120, 62, th.panel, th.primary, rx=8, sw=2.2)
    s.text(mx, 158, "m′t", 22, th.fg, mono=True, bold=True)
    _spring_v(s, mx, 182, 288, th.accent, coils=4)
    s.text(mx + 26, 240, "s′t", 20, th.accent, anchor="start", mono=True,
           bold=True)
    s.ground(288, mx - 70, mx + 70)
    _motion_arrows(s, mx - 92, 151, 26, th.secondary)

    # Response curve with the resonance read at its peak.
    ax0, ax1, base = 540.0, 850.0, 420.0
    s.line(ax0, base, ax1, base, th.muted, 1.4)
    s.line(ax0, base, ax0, 330.0, th.muted, 1.4)
    pk = 660.0
    s.path(f"M {ax0 + 6} {base - 12} C {pk - 60} {base - 16} {pk - 34} 336 "
           f"{pk} 334 C {pk + 34} 336 {pk + 70} {base - 8} {ax1 - 6} {base - 4}",
           stroke=th.primary, sw=2.4)
    s.line(pk, base, pk, 336, th.muted, 1.2, dash="4,3")
    s.text(pk, base + 22, "fr", 18, th.secondary, mono=True, bold=True)
    s.text((ax0 + ax1) / 2, base + 48, "resonance read from the response peak",
           15, th.muted, italic=True)

    # Headline relations.
    s.text(450, 524, "s′t = 4π² m′t fr²   (Formula 4)", 21, th.primary,
           bold=True, mono=True)
    s.text(450, 550,
           "then f₀ = (1/2π)·√(s′/m′) for the installed floating floor   (Formula 2)",
           16, th.muted, mono=True)


# ---------------------------------------------------------------------------
# Mechanical-mobility rig (ISO 7626)
# ---------------------------------------------------------------------------

def _d_mobility_rig(s: SVG, th: Theme) -> None:
    """ISO 7626 rig: free-free beam, exciter + impedance head at the driving
    point, accelerometer at a transfer point, impact-hammer variant."""
    cy_top, beam_top, beam_h = 116.0, 286.0, 26.0
    beam_bot = beam_top + beam_h
    # Ceiling with soft suspension.
    s.line(150, cy_top, 730, cy_top, th.fg, 2.2)
    for hx in range(162, 730, 26):
        s.line(hx, cy_top, hx - 9, cy_top - 9, th.muted, 1.1)
    for sx in (168.0, 712.0):
        _spring_v(s, sx, cy_top, beam_top, th.muted, coils=3, width=8.0,
                  sw=1.6)
    s.text(196, 142, "soft elastic suspension", 15, th.muted, anchor="start")

    # Beam under test.
    s.rect(150, beam_top, 580, beam_h, th.panel, th.fg, sw=2.2)
    s.text(470, beam_bot + 32, "Structure under test (free-free beam)", 19,
           th.fg, bold=True)

    # Driving point: exciter below the beam through an impedance head.
    dx = 248.0
    s.rect(dx - 14, beam_bot, 28, 16, th.secondary, th.fg, rx=3, sw=1.6)
    s.line(dx, beam_bot + 16, dx, 412, th.fg, 2.2)
    s.rect(dx - 37, 412, 74, 48, th.panel, th.primary, rx=9, sw=2)
    s.text(dx, 486, "Exciter", 18, th.fg, bold=True)
    s.text(60, 380, "Impedance head", 16, th.fg, anchor="start", bold=True)
    s.text(60, 402, "F and a at the drive point", 14, th.muted,
           anchor="start")
    s.line(dx - 16, beam_bot + 8, 154, 362, th.muted, 1.1, dash="3,3")
    s.arrow(dx + 18, 396, dx + 18, 340, th.secondary, 2.2)
    s.text(dx + 28, 372, "Fi", 16, th.secondary, anchor="start", mono=True)
    s.arrow(dx, beam_top - 4, dx, beam_top - 46, th.accent, 2.2)
    s.text(dx - 14, beam_top - 34, "vi", 16, th.accent, anchor="end",
           mono=True)
    s.text(210, 218, "driving point:  Yii = vi / Fi", 16, th.fg,
           anchor="start", mono=True)

    # Transfer point: accelerometer further along the beam.
    tx = 430.0
    _accel(s, tx, beam_top)
    s.arrow(tx, beam_top - 28, tx, beam_top - 56, th.accent, 2.2)
    s.text(tx + 12, beam_top - 40, "vj", 16, th.accent, anchor="start",
           mono=True)
    s.text(tx + 60, 192, "transfer:  Yji = vj / Fi", 16, th.fg, mono=True)

    # Impact-hammer variant striking the beam.
    hx2 = 600.0
    s.line(hx2 + 60, 172, hx2 + 6, 244, th.fg, 2.4)
    s.rect(hx2 - 16, 238, 32, 20, th.panel, th.fg, rx=4, sw=2)
    s.arrow(hx2, 262, hx2, beam_top - 6, th.secondary, 1.8)

    # FRF family footer.
    s.text(450, 520,
           "Y(f) = v/F  [m/(N·s)] · attached exciter (Part 2) · impact hammer (Part 5)",
           17, th.fg, mono=True)
    s.text(450, 546,
           "same measurement, three FRFs: x/F receptance · v/F mobility · a/F accelerance",
           15, th.muted)


# ---------------------------------------------------------------------------
# Dynamic transfer stiffness of resilient elements (ISO 10846)
# ---------------------------------------------------------------------------

def _d_transfer_stiffness_rig(s: SVG, th: Theme) -> None:
    """ISO 10846: isolator between the driven input mass and a blocked output
    (direct, force transducer) or a blocking mass (indirect)."""
    for cx, head in ((250.0, "Direct method (Part 2)"),
                     (650.0, "Indirect method (Part 3)")):
        s.text(cx, 78, head, 22, th.fg, bold=True)
        _exciter(s, cx, 158.0, stinger=18.0)
        # Driven input mass with its input displacement u1.
        s.rect(cx - 80, 158, 160, 44, th.panel, th.fg, rx=6, sw=2.2)
        s.text(cx, 186, "excitation mass", 16, th.fg)
        _motion_arrows(s, cx - 96, 180, 24, th.secondary)
        s.text(cx - 110, 186, "u₁", 18, th.secondary, anchor="end", mono=True)
        # Isolator under test.
        _spring_v(s, cx, 202, 310, th.accent, coils=4)
        s.text(cx + 28, 260, "isolator under test", 16, th.accent,
               anchor="start")

    # ===== Direct output: blocked, force transducer on a rigid foundation ===
    cx = 250.0
    s.rect(cx - 30, 310, 60, 18, th.secondary, th.fg, rx=3, sw=1.6)
    s.text(cx + 40, 322, "force transducer", 15, th.secondary, anchor="start")
    s.rect(cx - 105, 328, 210, 26, th.panel, th.fg, sw=2)
    s.ground(354, cx - 125, cx + 125)
    s.text(cx, 388, "Rigid foundation", 15, th.muted)
    s.text(cx, 470, "output blocked:  u₂ ≈ 0 → measure F₂,b", 16, th.fg,
           mono=True)
    s.text(cx, 500, "k₂,₁ = F₂,b / u₁", 20, th.primary, bold=True, mono=True)

    # ===== Indirect output: blocking mass on soft supports ==================
    cx = 650.0
    s.rect(cx - 85, 310, 170, 60, th.panel, th.fg, rx=6, sw=2.4)
    s.text(cx, 346, "blocking mass m₂", 17, th.fg)
    _accel(s, cx + 55, 310)
    s.text(cx + 72, 296, "a₂", 15, th.secondary, anchor="start", mono=True)
    for sx in (cx - 50.0, cx + 50.0):
        _spring_v(s, sx, 370, 430, th.muted, coils=3, width=8.0, sw=1.6)
    s.ground(430, cx - 115, cx + 115)
    s.text(cx + 70, 408, "soft support", 14, th.muted, anchor="start")
    s.text(cx, 470, "measure T = u₂ / u₁  (small)", 16, th.fg, mono=True)
    s.text(cx, 500, "k₂,₁ = −(2πf)²·(m₂+mf)·T", 20, th.primary, bold=True,
           mono=True)

    # Validity footer (Part 3 clause 6, Part 1 Eq. 7).
    s.text(450, 556,
           "valid where ΔL₁,₂ = La₁ − La₂ ≥ 20 dB, i.e. |T| ≤ 0.1   (Part 3, Inequality 2)",
           17, th.muted)
    s.text(450, 582,
           "the blocking force approximates the force delivered to a stiff receiver (Part 1, Eq. 7)",
           15, th.muted, italic=True)


# ---------------------------------------------------------------------------
# Reception plate (EN 15657)
# ---------------------------------------------------------------------------

def _d_reception_plate(s: SVG, th: Theme) -> None:
    """EN 15657 reception plate: source machine on a resiliently supported
    plate, averaged plate velocity, plate power balance."""
    # ===== Source machine standing on the plate =====
    mx = 260.0
    s.text(mx, 150, "Source under test (pump, fan, boiler …)", 20, th.fg,
           bold=True)
    s.rect(mx - 75, 218, 150, 70, th.panel, th.fg, rx=8, sw=2.2)
    s.rect(mx - 48, 190, 58, 28, th.panel, th.muted, rx=6, sw=1.8)
    s.circle(mx + 40, 240, 12, th.bg, th.muted, 1.8)
    for fx in (mx - 52.0, mx + 52.0):
        s.rect(fx - 8, 288, 16, 14, th.fg, rx=2)
        s.arrow(fx, 306, fx, 326, th.secondary, 2.4)
    s.text(310, 356, "injected structure-borne power", 15, th.secondary,
           italic=True)

    # ===== Reception plate on resilient supports =====
    s.rect(100, 302, 460, 32, th.panel, th.primary, rx=3, sw=2.4)
    s.text(455, 324, "Reception plate  (m, S, η)", 16, th.fg, bold=True)
    for ax_ in (140.0, 190.0, 400.0, 500.0):
        _accel(s, ax_, 302)
    s.text(560, 272, "velocity positions → Lv", 16, th.secondary,
           anchor="end")
    for sx in (150.0, 510.0):
        _spring_v(s, sx, 334, 430, th.accent, coils=3)
    s.ground(430, 80, 580)
    s.text(330, 404, "resilient supports", 14, th.muted)

    # ===== Right column: the power balance and the source quantities =====
    s.text(735, 150, "Plate power balance", 20, th.fg, bold=True)
    s.rect(590, 172, 292, 148, "none", th.muted, rx=10, dash="6,5")
    s.text(735, 206, "P = ω·η·(m·S)·⟨v²⟩", 19, th.primary, bold=True,
           mono=True)
    s.text(735, 238, "η = 2.2 / (f·Ts)   (Formula 13)", 15, th.fg, mono=True)
    s.text(735, 270, "L_Ws = 10 lg(2πf·η·m·S / f₀m₀S₀)", 14, th.fg, mono=True)
    s.text(735, 296, "+ Lv − 60   (Formula 14)", 14, th.fg, mono=True)
    s.text(735, 366, "→ source quantities (Formulae 15–19):", 15, th.fg,
           bold=True)
    s.text(735, 394, "equivalent blocked force L_Fb,eq ,", 15, th.muted)
    s.text(735, 418, "L_Wsn consumed by EN 12354-5", 15, th.muted)

    # Footer: the spatial velocity average.
    s.text(450, 516,
           "spatial average:  Lv = 10 lg[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)",
           17, th.fg, mono=True)


# ---------------------------------------------------------------------------
# Installed structure-borne sound paths (EN 12354-5)
# ---------------------------------------------------------------------------

def _d_installed_paths(s: SVG, th: Theme) -> None:
    """EN 12354-5: service equipment on a floor slab, structure-borne paths
    into the receiving room below, and the prediction cascade."""
    bx0, bx1 = 80.0, 590.0
    top, slab_top, slab_bot, bot = 92.0, 296.0, 324.0, 528.0

    # Rooms, continuous floor slab and flanking wall (drawn over the slab).
    s.rect(bx0, top, bx1 - bx0, slab_top - top, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_bot, bx1 - bx0, bot - slab_bot, th.panel, th.fg, sw=2.5)
    s.rect(bx0, slab_top, bx1 - bx0 + 26, slab_bot - slab_top, th.panel,
           th.fg, sw=2)
    for hx in range(int(bx0) + 16, int(bx1) + 26, 34):
        s.line(hx, slab_top, hx - 12, slab_bot, th.muted, 0.9)
    s.rect(bx1, top, 26, bot - top, th.panel, th.fg, sw=2)
    s.text(bx0 + 16, top + 32, "Source room", 21, th.fg, bold=True,
           anchor="start")
    s.text(bx0 + 16, bot - 18, "Receiving room", 21, th.fg, bold=True,
           anchor="start")

    # Service equipment on resilient mounts on the slab.
    mx = 210.0
    s.rect(mx - 55, 238, 110, 42, th.panel, th.fg, rx=7, sw=2.2)
    s.rect(mx - 34, 216, 40, 22, th.panel, th.muted, rx=5, sw=1.6)
    for fx in (mx - 36.0, mx + 36.0):
        _spring_v(s, fx, 280, slab_top, th.accent, coils=2, width=6.0, sw=1.6)
    s.text(mx, 200, "Service equipment (pump)", 19, th.fg, bold=True)
    s.text(mx + 78, 268, "coupling D_C   (Formula 19b)", 15, th.secondary,
           anchor="start")

    # Path i = j: the excited slab radiates into the room below.
    s.arrow(mx, slab_bot + 2, mx, slab_bot + 40, th.secondary, 2.4)
    for r in (40, 66, 92):
        s.path(f"M {mx - r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f} "
               f"A {r} {r} 0 0 0 {mx + r * 0.72:.1f} {slab_bot + 42 + r * 0.5:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(mx, 484, "excited floor radiates (path i = j)", 15, th.secondary,
           italic=True)

    # Path i -> j: along the slab, through the junction, down the wall.
    s.line(mx + 40, 310, 596, 310, th.primary, 2.6)
    s.line(603, 310, 603, 420, th.primary, 2.6)
    s.arrow(603, 420, 574, 420, th.primary, 2.6)
    s.circle(603, 310, 5, th.bg, th.fg, 2)
    for r in (30, 52):
        s.path(f"M {588 - r * 0.5:.1f} {420 - r * 0.72:.1f} "
               f"A {r} {r} 0 0 0 {588 - r * 0.5:.1f} {420 + r * 0.72:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(584, 288, "path along the slab into the wall  (i → j)", 15,
           th.primary, anchor="end")

    # ===== Right column: the prediction cascade =====
    s.text(760, 120, "Prediction cascade", 20, th.fg, bold=True)
    steps = [
        ("L_Ws,c", "characteristic power (EN 15657)", th.fg),
        ("− D_C", "coupling at the contacts (19b)", th.secondary),
        ("L_Ws,inst", "installed power (18b)", th.fg),
        ("− D_sa − R_ij,ref", "per transmission path (18a)", th.primary),
        ("10 lg Σ 10^(L_n,s,ij/10)", "energetic sum L_n,s (17)", th.accent),
    ]
    y = 164.0
    for k, (term, caption, col) in enumerate(steps):
        s.text(760, y, term, 19, col, bold=True, mono=True)
        s.text(760, y + 22, caption, 14, th.muted)
        if k < len(steps) - 1:
            s.arrow(760, y + 34, 760, y + 56, th.muted, 1.6)
        y += 84

    # Footer: what a path is.
    s.text(335, 574,
           "each path i → j: excited element i, radiating element j in the receiving room",
           16, th.muted)


# ---------------------------------------------------------------------------
# Wind-turbine noise measurement geometry (IEC 61400-11)
# ---------------------------------------------------------------------------

def _d_wind_turbine(s: SVG, th: Theme) -> None:
    """IEC 61400-11 apparent-sound-power geometry: downwind ground-board
    microphone at R0 = H + D/2, slant distance R1 to the rotor centre and
    the Figure 3 plan-view position pattern."""
    import math
    gy = 470.0
    s.ground(gy, 40.0, 668.0)

    # Wind arrow on the upwind side.
    s.arrow(52.0, 108.0, 148.0, 108.0, th.accent, 2.6)
    s.text(100.0, 88.0, "Wind", 18, th.accent, bold=True)

    # --- met mast: anemometer cups + wind vane -----------------------------
    mmx = 108.0
    s.line(mmx, gy, mmx, gy - 96.0, th.fg, 2.2)
    s.line(mmx - 12, gy - 96.0, mmx + 12, gy - 96.0, th.fg, 1.8)
    s.circle(mmx - 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)   # cup
    s.circle(mmx + 12, gy - 102.0, 4.5, th.panel, th.fg, 1.4)   # cup
    s.line(mmx, gy - 82.0, mmx + 20, gy - 82.0, th.fg, 1.6)     # vane arm
    s.path(f"M {mmx + 20:.0f} {gy - 87:.0f} L {mmx + 34:.0f} {gy - 82:.0f} "
           f"L {mmx + 20:.0f} {gy - 77:.0f} Z", fill=th.secondary)
    s.text(mmx, gy - 118.0, "Met mast", 16, th.fg, bold=True)
    s.text(mmx, gy + 30.0, "wind speed + direction", 14, th.muted)

    # --- turbine: tower, nacelle and the rotor edge-on ----------------------
    tx = 262.0                    # tower vertical centreline
    hub_y = 168.0                 # rotor centre => H = 302 px
    rr = 104.0                    # rotor radius (D = 208 px)
    s.path(f"M {tx - 10:.0f} {gy:.0f} L {tx - 5:.0f} {hub_y + 12:.0f} "
           f"L {tx + 5:.0f} {hub_y + 12:.0f} L {tx + 10:.0f} {gy:.0f} Z",
           fill=th.panel, stroke=th.fg, sw=1.8)
    s.rect(tx - 28, hub_y - 12, 56, 24, th.panel, th.fg, rx=6, sw=1.8)
    rx_ = tx - 34.0               # rotor plane (upwind of the tower)
    s.ellipse(rx_, hub_y, 9.0, rr, stroke=th.muted, sw=1.3, dash="6,5")
    s.line(rx_, hub_y - 8, rx_ - 4, hub_y - rr + 4, th.fg, 3.2)   # blade up
    s.line(rx_, hub_y + 8, rx_ + 3, hub_y + rr - 4, th.fg, 3.2)   # blade down
    s.circle(rx_, hub_y, 6.5, th.fg)
    s.text(tx + 36, hub_y - 26, "rotor centre", 15, th.fg, anchor="start")
    s.line(rx_ + 8, hub_y - 6, tx + 33, hub_y - 21, th.muted, 0.9)

    # Rotor diameter D across the swept ellipse.
    dx_ = rx_ - 58.0
    s.dim(dx_, hub_y - rr, dx_, hub_y + rr, "D", offset=0, label_side="left")
    s.line(rx_ - 8, hub_y - rr, dx_, hub_y - rr, th.muted, 0.9, dash="3,3")
    s.line(rx_ - 8, hub_y + rr, dx_, hub_y + rr, th.muted, 0.9, dash="3,3")

    # Hub height H, downwind of the tower.
    hx_ = tx + 56.0
    s.dim(hx_, gy, hx_, hub_y, "H", offset=0, label_side="right")
    s.line(tx + 10, gy, hx_, gy, th.muted, 0.9, dash="3,3")
    s.line(tx + 28, hub_y, hx_, hub_y, th.muted, 0.9, dash="3,3")

    # --- downwind microphone on a ground board ------------------------------
    mx = 640.0
    s.ellipse(mx, gy - 3.0, 36.0, 7.0, fill=th.panel, stroke=th.fg, sw=1.6)
    s.rect(mx - 16, gy - 10.0, 20, 6, th.fg, rx=2.5)              # capsule flat
    s.rect(mx + 4, gy - 11.0, 10, 8, th.primary, rx=2)            # body
    s.text(mx - 84, gy - 42.0, "Microphone on a ground board", 16, th.fg,
           bold=True, anchor="end")

    # Slant distance R1 from the rotor centre to the microphone.
    s.line(rx_, hub_y, mx - 12, gy - 8.0, th.primary, 2.2, dash="9,6")
    s.text(430.0, 296.0, "R1", 19, th.primary, bold=True)
    # Board-to-R1 inclination angle (25°..40°).
    ang = math.atan2(gy - 8.0 - hub_y, mx - 12 - rx_)   # slope of R1
    r_arc = 52.0
    axp = mx - 12 - r_arc * math.cos(ang)
    ayp = gy - 8.0 - r_arc * math.sin(ang)
    s.path(f"M {mx - 12 - r_arc:.1f} {gy - 8:.1f} "
           f"A {r_arc:.0f} {r_arc:.0f} 0 0 1 {axp:.1f} {ayp:.1f}",
           stroke=th.muted, sw=1.3)
    s.text(mx - 74, gy - 22.0, "φ", 17, th.muted)

    # Horizontal reference distance R0.
    s.dim(tx, gy, mx, gy, "R0 = H + D/2", offset=40)

    # --- plan-view inset: the Figure 3 position pattern ---------------------
    pcx, pcy, pr = 794.0, 218.0, 76.0
    s.text(pcx, 104.0, "Plan view (Figure 3)", 17, th.fg, bold=True)
    s.arrow(pcx - 60.0, 118.0, pcx - 60.0, 150.0, th.accent, 2.2)  # wind, from top
    s.circle(pcx, pcy, pr, "none", th.muted, 1.2)
    s.line(pcx - pr - 8, pcy, pcx + pr + 8, pcy, th.muted, 1.0, dash="4,4")
    s.line(pcx, pcy - pr - 8, pcx, pcy + pr + 8, th.muted, 1.0, dash="4,4")
    s.line(pcx - 16, pcy, pcx + 16, pcy, th.fg, 3.0)              # rotor, plan
    s.circle(pcx, pcy, 4.0, th.fg)
    # Reference position 1, downwind (diamond).
    p1x, p1y = pcx, pcy + pr
    s.path(f"M {p1x:.0f} {p1y - 7:.0f} L {p1x + 7:.0f} {p1y:.0f} "
           f"L {p1x:.0f} {p1y + 7:.0f} L {p1x - 7:.0f} {p1y:.0f} Z",
           fill=th.secondary)
    s.text(p1x + 13, p1y + 5, "1", 15, th.secondary, anchor="start", bold=True)
    # Optional positions 2 and 4 at ±60° from downwind, 3 upwind.
    for lbl, adeg, lx, ly, anch in (
        ("2", 150.0, -12.0, 4.0, "end"),
        ("3", 270.0, 12.0, 4.0, "start"),
        ("4", 30.0, 12.0, 4.0, "start"),
    ):
        pxx = pcx + pr * math.cos(math.radians(adeg))
        pyy = pcy + pr * math.sin(math.radians(adeg))
        s.circle(pxx, pyy, 5.5, th.bg, th.fg, 1.6)
        s.text(pxx + lx, pyy + ly, lbl, 15, th.muted, anchor=anch)
        s.line(pcx, pcy, pxx, pyy, th.muted, 0.9, dash="3,4")
    s.line(pcx, pcy, p1x, p1y, th.muted, 0.9, dash="3,4")
    s.text(pcx - 34, pcy + 52.0, "60°", 13, th.muted)
    s.text(pcx + 34, pcy + 52.0, "60°", 13, th.muted)
    s.text(pcx, pcy + pr + 30.0, "reference position 1 (downwind)", 14,
           th.secondary)
    s.text(pcx, pcy + pr + 50.0, "optional positions 2–4", 14, th.muted)

    # --- governing relations -------------------------------------------------
    s.text(450.0, 560.0,
           "R1 = √(H² + R0²)   slant distance, rotor centre → microphone",
           19, th.fg, bold=True)
    s.text(450.0, 588.0,
           "LWA,i = Lp,i − 6 + 10 lg(4π R1² / S0)   (Formula 26, S0 = 1 m²)",
           19, th.primary, bold=True)
    s.text(450.0, 614.0,
           "the −6 dB removes the board's pressure doubling; board-to-R1 angle φ = 25°–40°",
           16, th.muted)


# ---------------------------------------------------------------------------
# Ground reflection: direct ray, image source, path difference
# ---------------------------------------------------------------------------

def _d_ground_reflection(s: SVG, th: Theme) -> None:
    """Two-path ground interference: source, receiver, direct ray, the
    specular reflection unfolded through the image source, and the path
    difference that sets the interference phase (ISO 9613-2 ground effect,
    Chien-Soroka geometry)."""
    gy = 372.0
    sx, sy = 170.0, 232.0          # source (hs = 140 px)
    rx, ry = 700.0, 282.0          # receiver capsule tip (hr = 90 px)
    ix, iy = sx, gy + (gy - sy)    # image source, mirrored below the ground
    # Specular point: equal angles, found by unfolding through the image.
    bx = sx + (rx - sx) * (gy - sy) / ((gy - sy) + (gy - ry))

    s.ground(gy, 60.0, 840.0)

    # Source: point with radiating arcs.
    for r in (22, 38, 54):
        s.path(f"M {sx + r * 0.30:.1f} {sy - r * 0.95:.1f} "
               f"A {r} {r} 0 0 1 {sx + r * 0.95:.1f} {sy - r * 0.30:.1f}",
               stroke=th.muted, sw=1.3)
    s.circle(sx, sy, 8.0, th.fg)
    s.text(sx, sy - 66.0, "Source", 20, th.fg, bold=True)
    s.text(sx - 14, sy + 24, "S", 15, th.fg, anchor="end", mono=True)
    s.line(sx, sy + 8, sx, gy, th.fg, 2.0)

    # Receiver: measurement microphone.
    s.mic(rx, ry, gy, 1.0)
    s.text(rx, ry - 18.0, "Receiver", 20, th.fg, bold=True)
    s.text(rx - 18, ry + 10.0, "R", 15, th.fg, anchor="end", mono=True)

    # Direct ray r1.
    s.arrow(sx + 10, sy, rx - 8, ry - 2, th.primary, 2.6)
    s.text(430.0, 236.0, "direct ray  r1", 17, th.primary, bold=True)

    # Reflected ray via the specular point (equal angles).
    s.line(sx + 6, sy + 7, bx, gy, th.accent, 2.6)
    s.arrow(bx, gy, rx - 6, ry + 4, th.accent, 2.6)
    s.text(330.0, gy - 34.0, "reflected ray", 17, th.accent, bold=True)
    # Equal grazing angles at the bounce.
    s.path(f"M {bx - 34:.1f} {gy:.1f} A 34 34 0 0 1 {bx - 26:.1f} {gy - 21:.1f}",
           stroke=th.muted, sw=1.2)
    s.path(f"M {bx + 34:.1f} {gy:.1f} A 34 34 0 0 0 {bx + 26:.1f} {gy - 21:.1f}",
           stroke=th.muted, sw=1.2)
    s.text(bx, gy - 40.0, "equal angles", 14, th.muted)

    # Image source: ghosted mirror of the source below the ground.
    s.circle(ix, iy, 8.0, "none", th.secondary, 1.8)
    s.line(sx, gy, ix, iy - 8, th.secondary, 1.2, dash="4,4")
    s.text(ix + 18, iy + 5, "image source", 16, th.secondary, anchor="start")
    s.text(ix - 16, iy + 5, "S′", 15, th.secondary, anchor="end", mono=True)
    # The unfolded path S' -> R is straight through the bounce point: r2.
    s.line(ix, iy - 6, bx, gy, th.secondary, 1.6, dash="7,5")
    s.line(bx, gy, rx - 6, ry + 4, th.secondary, 1.2, dash="2,6")
    s.text(380.0, iy - 62.0, "r2 = |S′R|", 16, th.secondary, mono=True)

    # Heights.
    s.dim(sx - 46, gy, sx - 46, sy, "hs", offset=0, label_side="left")
    s.line(sx - 46, gy, sx - 8, gy, th.muted, 0.9, dash="3,3")
    s.line(sx - 46, sy, sx - 8, sy, th.muted, 0.9, dash="3,3")
    s.dim(rx + 42, gy, rx + 42, ry, "hr", offset=0, label_side="right")
    s.line(rx + 8, gy, rx + 42, gy, th.muted, 0.9, dash="3,3")
    s.line(rx + 8, ry, rx + 42, ry, th.muted, 0.9, dash="3,3")

    # Governing relations (top block, clear of the geometry).
    s.text(560.0, 88.0, "path difference  δ = r2 − r1", 20, th.fg, bold=True)
    s.text(560.0, 114.0, "phase difference  Δφ = 2π δ / λ  (+ arg Q)", 18,
           th.fg)
    s.text(560.0, 142.0,
           "p ∝ e^(jkr1)/r1 + Q · e^(jkr2)/r2   (Q = ground reflection coefficient)",
           16, th.muted)
    s.text(560.0, 168.0,
           "in phase (δ ≈ nλ): up to +6 dB    ·    out of phase (δ ≈ λ/2 on hard ground): a deep dip",
           15, th.muted)


def _d_fdtd(s: SVG, th: Theme) -> None:
    """2D acoustic FDTD pipeline (Attenborough & Van Renterghem 2021, Ch. 4)."""
    cx = 450.0
    bw, bh = 660.0, 58.0
    x0 = cx - bw / 2

    # --- Inputs (two feeder boxes) -----------------------------------------
    iw = 320.0
    s.rect(x0, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(x0 + iw / 2, 72, "Domain  c(x, y), ρ(x, y), dx", 17, th.fg,
           "middle", bold=True)
    s.text(x0 + iw / 2, 92, "square cells; dt from the Courant number",
           13, th.muted, "middle")
    s.rect(x0 + bw - iw, 48, iw, bh, th.panel, th.fg, rx=10, sw=2)
    s.text(x0 + bw - iw / 2, 72, "Geometry and boundaries", 17, th.fg,
           "middle", bold=True)
    s.text(x0 + bw - iw / 2, 92,
           "rigid, impedance or absorbing edges; obstacles", 13,
           th.muted, "middle")
    s.arrow(x0 + iw / 2, 106, cx - 60, 150, th.fg, 1.8)
    s.arrow(x0 + bw - iw / 2, 106, cx + 60, 150, th.fg, 1.8)

    def _step(y: float, l1: str, l2: str, color: str) -> None:
        s.rect(x0, y, bw, bh, th.panel, color, rx=10, sw=2)
        s.text(cx, y + 25, l1, 17, th.fg, "middle", bold=True)
        s.text(cx, y + 45, l2, 13, th.muted, "middle")

    _step(150, "Sources  s(t) injected at cells  (Eq. 4.11-4.12 grid)",
          "Gaussian pulse, ramped tone or arbitrary sampled signal", th.fg)
    _step(238, "Staggered-grid leapfrog update  (Eqs. 4.11-4.12)",
          "v ← v − (dt/ρ·dx)·grad p,  then  p ← p − (ρc²·dt/dx)·div v",
          th.primary)
    _step(326, "stable while  CN = c·dt·√2/dx ≤ 1  (Eqs. 4.13-4.14)",
          "resolve ≥ 10 cells per wavelength to keep dispersion low",
          th.secondary)
    for y0, y1 in ((208, 238), (296, 326), (384, 414)):
        s.arrow(cx, y0, cx, y1, th.fg, 1.8)

    # --- Output -------------------------------------------------------------
    s.rect(x0, 414, bw, bh, "none", th.primary, rx=10, sw=2.4)
    s.text(cx, 439, "FDTDResult:  probe histories p(t), field snapshots, .plot()",
           17, th.fg, "middle", bold=True)
    s.text(cx, 459, "deterministic: same inputs, bit-identical outputs", 13,
           th.muted, "middle")


# ---------------------------------------------------------------------------
# Sound level meter chain (IEC 61672-1)
# ---------------------------------------------------------------------------

def _d_slm_chain(s: SVG, th: Theme) -> None:
    """IEC 61672-1 sound level meter: the acoustic front end on the left
    (windscreen, microphone, coupled class 1 calibrator) feeding the
    four-stage level chain on the right."""
    gy = 508.0
    s.ground(gy, 40.0, 330.0)

    # --- Acoustic calibrator, coupled onto the capsule for the level check --
    mx = 165.0
    s.text(mx, 68.0, "Sound calibrator (class 1)", 20, th.fg, bold=True)
    s.rect(mx - 62, 82.0, 124, 84, th.panel, th.fg, rx=10, sw=2)
    s.text(mx, 118.0, "94.0 dB", 26, th.secondary, bold=True, mono=True)
    s.text(mx, 146.0, "1 kHz", 20, th.muted, mono=True)
    s.rect(mx - 15, 166.0, 30, 12, th.fg, rx=3)            # coupler cavity
    s.arrow(mx, 184.0, mx, 236.0, th.secondary, 2.0)
    s.text(mx - 16, 202.0, "coupled for", 15, th.muted, anchor="end",
           italic=True)
    s.text(mx - 16, 222.0, "the level check", 15, th.muted, anchor="end",
           italic=True)

    # --- Microphone on a stand; the windscreen is fitted for measurement ----
    cap_top = 248.0
    s.mic(mx, cap_top, gy, 1.25)
    s.ellipse(mx, cap_top + 18, 42, 42, "none", th.muted, 1.6, dash="5,4")
    s.line(mx - 30, cap_top + 49, mx - 65, cap_top + 76, th.muted, 1.0)
    s.text(mx - 79, cap_top + 94, "Windscreen", 17, th.fg)

    # --- The four-stage level chain (vertical) ------------------------------
    cx, bw, bh = 610.0, 400.0, 78.0
    x0 = cx - bw / 2
    chain = [
        (96.0, "Microphone + preamplifier",
         "free-field capsule, high-impedance stage"),
        (208.0, "Frequency weighting  A / C / Z",
         "all three are 0 dB at 1 kHz; class 1: ±0.7 dB"),
        (320.0, "Squaring + time weighting  F / S",
         "exponential detector: τF = 125 ms, τS = 1 s"),
    ]
    for by, l1, l2 in chain:
        s.rect(x0, by, bw, bh, th.panel, th.primary, rx=12, sw=2)
        s.text(cx, by + 33, l1, 21, th.fg, bold=True)
        s.text(cx, by + 59, l2, 17, th.muted)
    s.rect(x0, 432.0, bw, bh, "none", th.accent, rx=12, sw=2.4)
    s.text(cx, 465.0, "Display", 21, th.fg, bold=True)
    s.text(cx, 491.0, "LAF(t), LAS(t) in dB re 20 µPa", 17, th.accent)
    for y0 in (174.0, 286.0, 398.0):
        s.arrow(cx, y0, cx, y0 + 32, th.fg, 2.0)
    # Sound pressure into the front end.
    s.arrow(226.0, 268.0, x0 - 8, 135.0, th.fg, 2.0)


# ---------------------------------------------------------------------------
# Laboratory sound insulation suite (ISO 10140)
# ---------------------------------------------------------------------------

def _d_insulation_lab(s: SVG, th: Theme) -> None:
    """ISO 10140 laboratory transmission suite in plan view: two
    structurally decoupled reverberant rooms, the test element mounted in
    the ~10 m2 test opening, a corner loudspeaker in the source room and a
    continuously moving (rotating) microphone in each room."""
    top = 92.0
    sc = 72.0                       # px per metre
    src_bot = top + 4.4 * sc        # source room 5.0 m x 4.4 m
    rec_bot = top + 4.1 * sc        # receiving room 4.6 m x 4.1 m
    rec_r = 470.0 + 4.6 * sc

    # Room shells (separate structures).
    s.rect(70, top, 360, src_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.rect(470, top, rec_r - 470, rec_bot - top, th.panel, th.fg, rx=4, sw=3)
    s.text(90, top + 30, "Source room", 21, th.fg, bold=True, anchor="start")
    s.text(90, top + 56, "V₁ ≈ 59 m³", 17, th.muted, anchor="start", mono=True)
    s.text(486, top + 30, "Receiving room", 21, th.fg, bold=True, anchor="start")
    s.text(486, top + 56, "V₂ ≈ 51 m³", 17, th.muted, anchor="start", mono=True)

    # Test opening (3.75 m in plan) with the specimen mounted; filler stubs
    # from each shell with an air gap between them (the structural break).
    op_t, op_b = 110.0, 380.0
    s.rect(430, top, 14, op_t - top, th.panel, th.fg, sw=1.6)
    s.rect(430, op_b, 14, src_bot - op_b, th.panel, th.fg, sw=1.6)
    s.rect(456, top, 14, op_t - top, th.panel, th.fg, sw=1.6)
    s.rect(456, op_b, 14, rec_bot - op_b, th.panel, th.fg, sw=1.6)
    s.rect(438, op_t, 24, op_b - op_t, th.panel, th.secondary, sw=2)
    for hy in range(int(op_t) + 12, int(op_b), 16):
        s.line(440, hy + 10, 460, hy - 4, th.secondary, 1.0)
    s.text(450, 66, "structural break", 15, th.muted, italic=True)
    s.line(450, 72, 450, 102, th.muted, 1.0, dash="3,3")
    s.text(450, 452, "Test element in the test opening", 17, th.secondary,
           bold=True)
    s.line(450, op_b + 4, 450, 436, th.muted, 1.0, dash="3,3")

    # Loudspeaker in a corner of the source room.
    lsx, lsy = 150.0, 350.0
    for r in (36, 60, 84):
        s.path(f"M {lsx + r * 0.22:.1f} {lsy - r:.1f} "
               f"A {r} {r} 0 0 1 {lsx + r:.1f} {lsy - r * 0.22:.1f}",
               stroke=th.accent, sw=1.5)
    s.rect(lsx - 24, lsy - 27, 48, 54, th.panel, th.primary, rx=6, sw=2)
    s.circle(lsx, lsy - 9, 11, th.primary)
    s.circle(lsx, lsy - 9, 4.5, th.bg)
    s.circle(lsx, lsy + 15, 6, th.primary)
    s.text(lsx + 4, lsy + 48, "Loudspeaker", 17, th.fg, bold=True)

    # Continuously moving (rotating) microphone in each room: the sweep
    # circle, the boom and the microphone on its tip.
    for mcx, mcy, a_mic in ((285.0, 200.0, 40.0), (640.0, 215.0, 150.0)):
        import math
        s.ellipse(mcx, mcy, sc, sc, "none", th.muted, 1.3, dash="6,5")
        pxm = mcx + sc * math.cos(math.radians(a_mic))
        pym = mcy - sc * math.sin(math.radians(a_mic))
        s.line(mcx, mcy, pxm, pym, th.fg, 2.0)
        s.circle(mcx, mcy, 4, th.fg)
        s.circle(pxm, pym, 7.5, th.secondary)
        s.circle(pxm, pym, 2.6, th.bg)
        _rot_arrow(s, mcx, mcy, sc + 12, -78, -8, th.accent, 1.8)
    s.text(285, 298, "moving microphone", 16, th.fg)
    s.text(285, 320, "sweep radius ≥ 1 m", 15, th.muted)
    s.text(640, 313, "moving microphone", 16, th.fg)

    # Dimensions (72 px per metre).
    s.dim(70, src_bot, 430, src_bot, "5.0 m", offset=30, size=18)
    s.dim(470, rec_bot, rec_r, rec_bot, "4.6 m", offset=30 + src_bot - rec_bot,
          size=18)
    s.dim(rec_r, top, rec_r, rec_bot, "4.1 m", offset=32, size=18,
          label_side="right")
    s.dim(430, op_t, 430, op_b, "3.75 m", offset=-24, size=17)

    # Normative facility limits.
    for y, txt in (
        (508.0, "Test opening ≈ 10 m² (3.75 m × 2.7 m); shorter edge ≥ 2.3 m"),
        (536.0, "Room volumes ≥ 50 m³, differing by at least 10 %"),
        (564.0, "Continuously moving microphone: sweep radius ≥ 1 m, traverse ≥ 15 s"),
    ):
        s.text(80, y, txt, 18, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# Junction vibration measurement, L- and T-junctions (ISO 10848)
# ---------------------------------------------------------------------------

def _plate_top(s: SVG, th: Theme, x0: float, y0: float, w: float, dp: float,
               t: float) -> None:
    """Horizontal slab in oblique projection: top face, front face and the
    visible right end face. ``(x0, y0)`` = front-left corner of the top face,
    ``w`` its width, ``dp`` its oblique depth and ``t`` its thickness."""
    dxo, dyo = dp * 0.72, dp * 0.55
    s.path(f"M {x0} {y0} L {x0 + w} {y0} L {x0 + w + dxo} {y0 - dyo} "
           f"L {x0 + dxo} {y0 - dyo} Z", fill=th.panel, stroke=th.fg, sw=1.8)
    s.rect(x0, y0, w, t, th.panel, th.fg, sw=1.8)
    s.path(f"M {x0 + w} {y0} L {x0 + w} {y0 + t} L {x0 + w + dxo} {y0 + t - dyo} "
           f"L {x0 + w + dxo} {y0 - dyo} Z", fill=th.panel, stroke=th.fg, sw=1.8)


def _plate_up(s: SVG, th: Theme, x0: float, y_base: float, t: float,
              h: float, dp: float) -> None:
    """Vertical slab in oblique projection standing on ``y_base``: front
    edge face, top face and the visible right-hand surface."""
    dxo, dyo = dp * 0.72, dp * 0.55
    y_top = y_base - h
    s.rect(x0, y_top, t, h, th.panel, th.fg, sw=1.8)
    s.path(f"M {x0} {y_top} L {x0 + t} {y_top} L {x0 + t + dxo} {y_top - dyo} "
           f"L {x0 + dxo} {y_top - dyo} Z", fill=th.panel, stroke=th.fg, sw=1.8)
    s.path(f"M {x0 + t} {y_top} L {x0 + t} {y_base} "
           f"L {x0 + t + dxo} {y_base - dyo} L {x0 + t + dxo} {y_top - dyo} Z",
           fill=th.panel, stroke=th.fg, sw=1.8)


def _accel_wall(s: SVG, x: float, y: float, size: float = 13.0) -> None:
    """Accelerometer block mounted on a wall surface seen in oblique view."""
    th = s.th
    s.rect(x - size / 2, y - size / 2, size, size, th.secondary, th.fg,
           rx=2.5, sw=1.3)
    s.line(x + size / 2, y, x + size / 2 + 8, y, th.fg, 1.3)


def _d_junction_rig(s: SVG, th: Theme) -> None:
    """ISO 10848 junction rig: an L- and a T-junction of concrete plates,
    structure-borne excitation on element i, accelerometers on i and j and
    the junction length l_ij along the corner line."""
    gy = 430.0
    dp = 170.0
    dxo, dyo = dp * 0.72, dp * 0.55

    # ===== Left: L-junction (wall on the left end of the floor plate) =====
    s.text(280, 86, "L-junction", 21, th.fg, bold=True)
    _plate_top(s, th, 140, gy, 230, dp, 16)
    _plate_up(s, th, 140, gy, 16, 180, dp)
    # Junction line along the corner, highlighted, with its length label.
    s.line(156, gy, 156 + dxo, gy - dyo, th.accent, 2.6)
    s.text(58, 474, "lij ≥ 2.3 m", 17, th.fg, anchor="start")
    s.line(126, 466, 152, 438, th.muted, 1.0)

    # Exciter on the floor (element i), accelerometers on i and j.
    _exciter(s, 330, 396)
    _accel(s, 250, 410)
    _accel(s, 380, 380)
    _accel_wall(s, 205, 300)
    _accel_wall(s, 236, 262)
    s.text(196, 420, "i", 22, th.primary, bold=True, italic=True)
    s.text(178, 200, "j", 22, th.secondary, bold=True, italic=True)
    # Transmission path across the corner.
    s.path("M 300 402 Q 214 400 208 330", stroke=th.accent, sw=2.0)
    s.arrow(209.0, 344.0, 208.0, 322.0, th.accent, 2.0)
    s.text(194, 356, "Dv,ij", 16, th.accent, anchor="end", mono=True)

    # ===== Right: T-junction (wall standing mid-way on the floor) =========
    s.text(690, 86, "T-junction", 21, th.fg, bold=True)
    _plate_top(s, th, 520, gy, 220, dp, 16)
    _plate_up(s, th, 620, gy, 16, 180, dp)
    s.line(636, gy, 636 + dxo, gy - dyo, th.accent, 2.6)
    _exciter(s, 566, 404)
    _accel(s, 588, 422)
    _accel_wall(s, 685, 290)
    _accel(s, 762, 384)
    s.text(533, 423, "i", 22, th.primary, bold=True, italic=True)
    s.text(658, 200, "j", 22, th.secondary, bold=True, italic=True)
    s.text(806, 400, "j", 22, th.secondary, bold=True, italic=True)
    s.path("M 612 418 Q 690 434 756 400", stroke=th.accent, sw=2.0)
    s.arrow(742.0, 407.0, 760.0, 398.0, th.accent, 2.0)
    s.path("M 606 406 Q 646 394 654 330", stroke=th.accent, sw=2.0)
    s.arrow(655.0, 344.0, 654.0, 322.0, th.accent, 2.0)

    # Exciter label shared by both panels.
    s.text(450, 250, "Shaker or hammer on element i", 17, th.fg)
    s.line(376, 258, 348, 322, th.muted, 1.0)
    s.line(524, 258, 556, 348, th.muted, 1.0)

    # Plate thickness leader (the lines stop above the caption text).
    s.text(450, 496, "concrete plates 140 mm to 200 mm thick", 17, th.muted)
    s.line(322, 477, 300, 442, th.muted, 1.0)
    s.line(578, 477, 600, 442, th.muted, 1.0)

    # Normative relations.
    s.text(80, 536, "lij ≥ 2.3 m along the junction; element sizes 3.0 m ≤ li < 6.0 m",
           18, th.fg, anchor="start")
    s.text(80, 564, "≥ 4 excitation positions on i; accelerometers ≥ 0.25 m from edges, ≥ 0.5 m apart",
           18, th.fg, anchor="start")
    s.text(80, 596, "Kij = D̄v,ij + 10 lg( lij / √(ai·aj) ),   ai = equivalent absorption length",
           18, th.primary, anchor="start", bold=True, mono=True)


# ---------------------------------------------------------------------------
# Sound power from surface vibration (ISO/TS 7849)
# ---------------------------------------------------------------------------

def _d_vibration_sound_power(s: SVG, th: Theme) -> None:
    """ISO/TS 7849 surface-velocity method: the machine's radiating surface
    divided into N equal cells, one accelerometer per cell centre, and the
    survey sound power from the mean velocity level over the area S."""
    gy = 470.0
    s.ground(gy, 50.0, 560.0)

    # Machine body with the vibrating measurement surface on top.
    bx, hw, dp, ht = 270.0, 140.0, 115.0, 170.0
    _box_solid(s, th, bx, gy, hw, dp, ht)
    fx0, fx1 = bx - hw, bx + hw          # top-face front edge
    fy = gy - ht
    dxo, dyo = dp * 0.72, dp * 0.55

    # Measurement grid: 5 x 2 cells on the top face, a dot per cell centre.
    for i in range(1, 5):
        gx = fx0 + i * (2 * hw) / 5
        s.line(gx, fy, gx + dxo, fy - dyo, th.muted, 1.0)
    s.line(fx0 + dxo / 2, fy - dyo / 2, fx1 + dxo / 2, fy - dyo / 2,
           th.muted, 1.0)
    pts = []
    for r_ in (0.25, 0.75):
        for i in range(5):
            u = (i + 0.5) / 5
            pts.append((fx0 + u * 2 * hw + r_ * dxo, fy - r_ * dyo))
    for px_, py_ in pts:
        s.circle(px_, py_, 5, th.secondary)
        s.circle(px_, py_, 1.8, th.bg)
    # One accelerometer drawn explicitly, with its vibratory motion.
    _accel(s, pts[5][0], pts[5][1] - 4)
    _motion_arrows(s, pts[5][0], pts[5][1] - 46, 16, th.secondary)
    s.text(250, 150, "Vibrating measurement surface S", 19, th.fg, bold=True)
    s.line(310, 160, 340, 228, th.muted, 1.0)

    # Radiated sound from the surface.
    for r in (36, 60, 84):
        s.path(f"M {475 + r * 0.30:.1f} {370 - r:.1f} "
               f"A {r} {r} 0 0 1 {475 + r:.1f} {370 - r * 0.30:.1f}",
               stroke=th.accent, sw=1.6)
    s.text(672, 432, "radiated airborne sound", 16, th.accent)
    s.line(618, 424, 570, 372, th.muted, 1.0)

    # Dimensions of the surface (2.5 m x 1.6 m -> S = 4 m2).
    s.dim(fx0, gy, fx1, gy, "2.5 m", offset=32, size=18)
    s.arrow(fx1 + 20, gy + 18, fx1 + dxo + 14, gy - dyo + 18, th.muted, 1.2)
    s.arrow(fx1 + dxo + 14, gy - dyo + 18, fx1 + 20, gy + 18, th.muted, 1.2)
    s.text(fx1 + dxo - 4, gy - dyo + 46, "1.6 m", 18, th.fg, anchor="start")
    s.text(bx, 540, "Machine under test", 18, th.fg, bold=True)

    # Number of measurement positions and the survey relation.
    lx = 575.0
    s.text(lx, 110, "Number of positions N", 19, th.fg, bold=True, anchor="start")
    for y, txt in ((140, "S < 1 m²   →   5"),
                   (166, "1 m² ≤ S ≤ 10 m²  →  10"),
                   (192, "S > 10 m²  →  S / S₀")):
        s.text(lx, y, txt, 16, th.fg, anchor="start", mono=True)
    s.text(lx, 220, "one accelerometer per cell of area S/N", 15, th.muted,
           anchor="start")
    s.text(lx, 284, "Survey sound power", 19, th.fg, bold=True, anchor="start")
    s.text(lx, 314, "LWA = LvA + 10 lg(S/S₀) + 10 lg ε", 15, th.primary,
           anchor="start", bold=True, mono=True)
    s.text(lx, 342, "ε = 1 assumed → upper limit LWA,max", 15, th.muted,
           anchor="start")
    s.text(lx, 368, "normal surface velocity, A-weighted r.m.s.", 15,
           th.muted, anchor="start")


# ---------------------------------------------------------------------------
# Ship radiated-noise measurement geometry (ISO 17208-1)
# ---------------------------------------------------------------------------

def _d_hydrophone_deployment(s: SVG, th: Theme) -> None:
    """ISO 17208-1 deep-water geometry: ship transiting past a buoy-suspended
    vertical array of three hydrophones at 15/30/45 degree depression angles,
    lateral CPA distance of at least 100 m, plus the plan-view data window."""
    import math
    surf = 150.0
    sc = 2.6                              # px per metre
    shx = 190.0                           # ship reference point at the CPA
    bx = shx + 100 * sc                   # buoy: dCPA = 100 m away

    # Sea surface as a gentle wave.
    dsur = f"M 50 {surf}"
    x = 50.0
    while x < 590:
        dsur += f" Q {x + 8:.0f} {surf - 5:.0f} {x + 16:.0f} {surf:.0f}"
        dsur += f" Q {x + 24:.0f} {surf + 5:.0f} {x + 32:.0f} {surf:.0f}"
        x += 32
    s.path(dsur, stroke=th.primary, sw=1.8)

    # Ship (side profile) at the closest point of approach.
    s.path(f"M 108 132 L 262 132 L 276 {surf} L 254 166 L 130 166 "
           f"L 108 {surf} Z", fill=th.panel, stroke=th.fg, sw=2)
    s.rect(122, 104, 44, 28, th.panel, th.fg, rx=3, sw=1.6)
    s.text(212, 88, "Ship under test", 18, th.fg, bold=True, anchor="end")
    s.circle(shx, surf, 3.5, th.fg)

    # Surface buoy with the suspended array and its ballast.
    s.circle(bx, 148, 10, th.panel, th.fg, 2)
    s.line(bx, 138, bx, 120, th.fg, 1.6)
    s.path(f"M {bx:.0f} 120 L {bx - 18:.0f} 125 L {bx:.0f} 130 Z",
           fill=th.secondary)
    s.text(bx + 18, 122, "Surface buoy", 16, th.fg, anchor="start", bold=True)
    s.line(bx, 158, bx, 448, th.fg, 2)
    s.rect(bx - 8, 448, 16, 22, th.fg, rx=2)
    s.text(bx, 490, "ballast", 14, th.muted)

    # Hydrophones at the depths set by the three depression angles.
    hyd = [(15, "≈ 27 m"), (30, "≈ 58 m"), (45, "= 100 m")]
    for ang, dlab in hyd:
        dy = 100 * math.tan(math.radians(ang)) * sc
        hy = surf + dy
        s.line(shx, surf, bx, hy, th.muted, 1.1, dash="5,4")
        s.circle(bx, hy, 7, th.secondary)
        s.circle(bx, hy, 2.5, th.bg)
        s.text(bx + 16, hy + 5, dlab, 15, th.fg, anchor="start", mono=True)
        lx_ = 305.0
        ly_ = surf + (lx_ - shx) * math.tan(math.radians(ang))
        s.text(lx_, ly_ - 7, f"{ang}°", 15, th.muted)
    s.text(bx + 16, surf + 100 * math.tan(math.radians(30)) * sc + 32,
           "vertical array of 3 hydrophones", 15, th.muted, anchor="start")

    # Lateral distance at the CPA and the water depth.
    s.dim(shx, 100, bx, 100, "dCPA ≥ 100 m (or 1·L)", offset=0, size=17)
    s.line(shx, 130, shx, 106, th.muted, 0.9, dash="3,3")
    s.line(bx, 116, bx, 106, th.muted, 0.9, dash="3,3")
    s.ground(540, 50, 600)
    s.text(90, 570, "sea floor", 14, th.muted, anchor="start")
    s.dim(70, surf, 70, 540, "water depth ≥ 150 m (or 1.5·L)", offset=0,
          size=16, label_side="right")

    # Plan view: course, CPA and the +/-30 degree data window.
    s.text(750, 130, "Plan view", 17, th.fg, bold=True)
    s.arrow(640, 170, 860, 170, th.fg, 2.0)
    s.text(852, 156, "course", 14, th.muted, anchor="end")
    s.rect(676, 162, 28, 14, th.panel, th.fg, rx=3, sw=1.4)
    s.circle(750, 170, 3.5, th.fg)
    # dCPA line drawn in two runs so it does not cross the label below.
    s.line(750, 170, 750, 184, th.muted, 1.1, dash="5,4")
    s.line(750, 210, 750, 330, th.muted, 1.1, dash="5,4")
    s.text(758, 256, "dCPA", 14, th.fg, anchor="start", mono=True)
    s.circle(750, 330, 6, th.secondary)
    s.circle(750, 330, 2.2, th.bg)
    win = 160 * math.tan(math.radians(30))
    s.line(750, 330, 750 - win, 170, th.muted, 1.0, dash="3,4")
    s.line(750, 330, 750 + win, 170, th.muted, 1.0, dash="3,4")
    s.text(750, 296, "±30°", 14, th.muted)
    s.line(750 - win, 178, 750 + win, 178, th.accent, 3.0)
    s.text(750, 200, "data window", 15, th.accent)

    # Normative context.
    s.text(80, 594, "Four runs, two per side; levels averaged while the ship crosses the data window",
           17, th.fg, anchor="start")
    s.text(80, 620, "Hydrophone depths from the 15°, 30° and 45° depression angles at r = dCPA; L = ship length",
           17, th.fg, anchor="start")


# ---------------------------------------------------------------------------
# SOFAR channel (deep sound channel)
# ---------------------------------------------------------------------------

def _d_sofar_channel(s: SVG, th: Theme) -> None:
    """The deep sound channel: measured North Atlantic values (sound speed
    1524 m/s at the surface, minimum near 1492 m/s at the 1200 m axis,
    1527 m/s at 4800 m) and rays oscillating about the channel axis."""
    import math
    surf, bot = 100.0, 520.0
    ax_y = surf + 1200.0 / 4800.0 * (bot - surf)      # channel axis, 1200 m

    # Ocean frame: surface, seabed and the left depth axis.
    s.line(60, surf, 850, surf, th.fg, 2.2)
    s.text(845, 88, "sea surface", 14, th.muted, anchor="end")
    s.ground(bot, 60, 850)
    s.line(90, surf, 90, bot, th.muted, 1.4)
    for dy_, ly_, dlab in ((surf, surf - 8, "0 m"), (ax_y, ax_y + 5, "1200 m"),
                           (bot, bot - 8, "4800 m")):
        s.line(84, dy_, 90, dy_, th.muted, 1.4)
        s.text(78, ly_, dlab, 14, th.fg, anchor="end", mono=True)

    # Channel axis (the sound-speed minimum).
    s.line(90, ax_y, 850, ax_y, th.muted, 1.2, dash="7,5")

    # --- Left: the sound-speed profile c(z) --------------------------------
    s.text(195, 76, "Sound-speed profile c(z)", 18, th.fg, bold=True)
    def cx_of(c: float) -> float:                     # 1480..1540 m/s
        return 90.0 + (c - 1480.0) / 60.0 * 180.0
    x_s, x_m, x_b = cx_of(1524), cx_of(1492), cx_of(1527)
    s.path(f"M {x_s:.1f} {surf:.1f} Q {x_m + 24:.1f} {surf + 52:.1f} "
           f"{x_m:.1f} {ax_y:.1f} Q {x_m + 14:.1f} {ax_y + 160:.1f} "
           f"{x_b:.1f} {bot:.1f}", stroke=th.primary, sw=2.6)
    s.circle(x_s, surf, 3.5, th.primary)
    s.circle(x_m, ax_y, 3.5, th.primary)
    s.circle(x_b, bot, 3.5, th.primary)
    s.text(x_s + 10, surf + 22, "1524 m/s", 14, th.fg, anchor="start", mono=True)
    s.text(x_m + 10, ax_y + 24, "≈ 1492 m/s", 14, th.fg, anchor="start", mono=True)
    s.text(x_b + 10, bot - 12, "1527 m/s", 14, th.fg, anchor="start", mono=True)

    # --- Right: rays trapped about the axis --------------------------------
    s.text(600, 76, "Ray paths near the axis", 18, th.fg, bold=True)
    sx = 315.0
    s.circle(sx, ax_y, 6, th.secondary)
    s.circle(sx, ax_y, 2.2, th.bg)
    s.text(310, 130, "source on the channel axis", 15, th.fg, anchor="start")
    s.line(322, 136, sx + 1, ax_y - 8, th.muted, 1.0)
    for amp, lam, col in ((45.0, 260.0, th.accent), (68.0, 310.0, th.primary),
                          (90.0, 360.0, th.secondary)):
        d = f"M {sx:.1f} {ax_y:.1f}"
        xr = sx
        while xr < 833:
            xr += 7
            yr = ax_y + amp * math.sin(2 * math.pi * (xr - sx) / lam)
            d += f" L {xr:.1f} {yr:.1f}"
        s.path(d, stroke=col, sw=1.8)
        y_end = ax_y + amp * math.sin(2 * math.pi * (840 - sx) / lam)
        y_prev = ax_y + amp * math.sin(2 * math.pi * (833 - sx) / lam)
        s.arrow(833.0, y_prev, 841.0, y_end, col, 1.8)
    s.text(575, 420, "rays that stay in the channel meet no surface or bottom loss",
           16, th.muted, italic=True)

    # Physics of the channel.
    s.text(80, 560, "c rises toward the surface (temperature) and toward the bottom (pressure); the minimum traps sound",
           17, th.fg, anchor="start")
    s.text(80, 588, "rays launched within about ±12° of the axis stay trapped and can cross entire oceans",
           17, th.fg, anchor="start")


DIAGRAMS = {
    "diagram_calibration_setup": (_d1, "Calibration chain — from calibrator to physical units", 560),
    "diagram_env_measurement": (_d2, "Environmental noise measurement positions (ISO 1996-2)", 560),
    "diagram_tonality_positions": (_d3, "Emission measurement positions (ECMA-74)", 560),
    "diagram_signal_chain": (_d4, "phonometry processing chain", 400),
    "diagram_multirate": (_d5, "Multirate decimation in the octave filter bank", 560),
    "diagram_pp_probe": (_d6, "Two-microphone (p-p) intensity probe", 460),
    "diagram_sti_chain": (_d7, "STI measurement chain (IEC 60268-16)", 400),
    "diagram_insulation_setup": (
        _d8, "Airborne sound insulation setup (ISO 16283-1)", 600),
    "diagram_ir_measurement": (
        _d9, "Impulse-response measurement chain (ISO 18233)", 440),
    "diagram_sound_power_surfaces": (
        _d_surfaces, "ISO 3744 / 3746 sound power measurement surfaces", 640),
    "diagram_impact_setup": (
        _d_impact, "ISO 16283-2 impact sound insulation setup", 600),
    "sound_power_methods": (
        _d_methods, "Sound power methods compared", 620),
    "diagram_flanking_paths": (
        _d_flanking, "Direct and flanking transmission paths (EN 12354)", 640),
    "diagram_outdoor_geometry": (
        _d_outdoor, "ISO 9613-2 source–barrier–receiver geometry", 560),
    "diagram_impedance_tube": (
        _d_impedance_tube, "Impedance tube: two-microphone method (ISO 10534-2)", 520),
    "diagram_astm_tube": (
        _d_astm_tube, "Four-microphone transmission-loss tube (ASTM E2611)", 560),
    "diagram_airflow_resistance": (
        _d_airflow, "Airflow resistance: static and alternating methods (ISO 9053-1/-2)", 540),
    "diagram_scattering_reverb": (
        _d_scattering_reverb,
        "Random-incidence scattering in a reverberation room (ISO 17497-1)", 560),
    "diagram_diffusion_goniometer": (
        _d_diffusion_goniometer,
        "Free-field diffusion goniometer (ISO 17497-2)", 580),
    "diagram_insitu_subtraction": (
        _d_insitu_subtraction,
        "In-situ road absorption — subtraction technique (ISO 13472-1)", 560),
    "diagram_spot_tube": (
        _d_spot_tube,
        "In-situ road absorption — spot method (ISO 13472-2)", 540),
    "diagram_precision_anechoic": (
        _d_precision_anechoic,
        "Precision sound power in an anechoic room (ISO 3745)", 600),
    "diagram_intensity_scan": (
        _d_intensity_scan,
        "Precision sound intensity scanning (ISO 9614-3)", 600),
    "diagram_human_vibration": (
        _d_human_vibration,
        "Whole-body vibration measurement chain (ISO 2631-1 / ISO 8041-1)", 580),
    "diagram_speech_intelligibility": (
        _d_speech_intelligibility,
        "Speech Intelligibility Index computation flow (ANSI S3.5-1997)", 600),
    "diagram_room_measurement": (
        _d_room_measurement,
        "Room-acoustics measurement setup (ISO 3382-1 / ISO 3382-2)", 620),
    "diagram_room_noise": (
        _d_room_noise,
        "Room-noise rating methods (ANSI/ASA S12.2-2019): NC and RC Mark II", 580),
    "diagram_hearing_threshold": (
        _d_hearing_threshold,
        "Hearing-threshold model (ISO 7029 age distribution, ISO 389-7 zero)", 600),
    "diagram_uncertainty": (
        _d_uncertainty,
        "Uncertainty: GUM propagation vs Monte Carlo (Guide 98-3)", 540),
    "diagram_nihl": (
        _d_nihl,
        "Noise-induced hearing loss (ISO 1999): NIPTS and HTLAN", 470),
    "diagram_ntacou112": (
        _d_impulse_prominence,
        "Impulsive-sound prominence and LAeq adjustment (NT ACOU 112)", 520),
    "diagram_iso2631_5": (
        _d_multiple_shock,
        "Multiple-shock spinal-response dose and injury risk (ISO 2631-5)", 580),
    "diagram_en12354_6": (
        _d_enclosed_space_absorption,
        "Absorption area and reverberation time of a room (EN 12354-6)", 410),
    "diagram_time_weighting": (
        _d_time_weighting,
        "Exponential-detector chain of the time weightings (IEC 61672-1)", 460),
    "diagram_block_processing": (
        _d_block_processing,
        "Block processing: carrying the filter state versus resetting it", 510),
    "diagram_multichannel": (
        _d_multichannel,
        "Array-shape flow through a per-channel operation", 410),
    "diagram_open_plan": (
        _d_open_plan,
        "Open-plan office spatial decay of speech (ISO 3382-3)", 500),
    "diagram_iso12999": (
        _d_iso12999,
        "Measurement uncertainty from tables to expanded U (ISO 12999-1)", 500),
    "diagram_iso11654": (
        _d_iso11654,
        "Single-number sound-absorption rating (ISO 11654)", 520),
    "diagram_zwicker": (
        _d_zwicker,
        "Zwicker loudness model chain (ISO 532-1)", 490),
    "diagram_equal_loudness_weighting": (
        _d_equal_loudness_weighting,
        "Why A-weighting: an equal-loudness contour, inverted (ISO 226)",
        560),
    "diagram_loudspeaker_freefield": (
        _d_loudspeaker_freefield,
        "Loudspeaker free-field sensitivity measurement (IEC 60268-5)", 600),
    "diagram_dosimeter_iso9612": (
        _d_dosimeter,
        "Occupational noise exposure measurement (ISO 9612)", 640),
    "diagram_dynamic_stiffness_rig": (
        _d_dynamic_stiffness_rig,
        "Dynamic-stiffness resonance rig (ISO 9052-1)", 560),
    "diagram_mobility_rig": (
        _d_mobility_rig,
        "Mechanical-mobility measurement on a beam (ISO 7626)", 560),
    "diagram_transfer_stiffness_rig": (
        _d_transfer_stiffness_rig,
        "Dynamic transfer stiffness: direct and indirect methods (ISO 10846)", 600),
    "diagram_reception_plate": (
        _d_reception_plate,
        "Reception-plate measurement of structure-borne power (EN 15657)", 560),
    "diagram_installed_paths": (
        _d_installed_paths,
        "Installed structure-borne sound paths (EN 12354-5)", 620),
    "diagram_wind_turbine_iec61400": (
        _d_wind_turbine,
        "Wind-turbine noise measurement geometry (IEC 61400-11)", 640),
    "diagram_ground_reflection": (
        _d_ground_reflection,
        "Ground reflection: direct ray, image source and path difference", 560),
    "diagram_fdtd": (
        _d_fdtd,
        "2D acoustic FDTD wave simulation (staggered leapfrog)", 500),
    "diagram_slm_chain": (
        _d_slm_chain,
        "Sound level meter measurement chain (IEC 61672-1)", 560),
    "diagram_insulation_lab": (
        _d_insulation_lab,
        "Laboratory sound insulation suite (ISO 10140)", 600),
    "diagram_junction_rig": (
        _d_junction_rig,
        "Junction vibration measurement on L- and T-junctions (ISO 10848)", 620),
    "diagram_vibration_sound_power": (
        _d_vibration_sound_power,
        "Sound power from surface vibration (ISO/TS 7849)", 580),
    "diagram_hydrophone_deployment": (
        _d_hydrophone_deployment,
        "Ship radiated-noise measurement geometry (ISO 17208-1)", 640),
    "diagram_sofar_channel": (
        _d_sofar_channel,
        "The SOFAR channel: a deep-ocean sound waveguide", 620),
}


def generate_all(output_dir: str = ".github/images") -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, (builder, title, height) in DIAGRAMS.items():
        _write(output_dir, name, builder, title, height)


if __name__ == "__main__":
    generate_all()
