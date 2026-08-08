#  Copyright (c) 2026. Jose Manuel Requena Plens
"""Spanish variants of every user-visible string the diagrams draw.

A table rather than code, which is why it is a module of its own: the
builders write their labels in English and :meth:`SVG.tr` looks the
Spanish rendering up here. A string that is absent is deliberate, not
missing; numbers, unit-only labels and code identifiers read the same in
both languages and are shared.
"""

from __future__ import annotations

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
    "N50 = [u + v·log10(t/t0)]·(L − L0)²":
        "N50 = [u + v·log10(t/t0)]·(L − L0)²",
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
    "P = 3·log10(OR) + 2·log10(LD);   highest P over 30 min governs":
        "P = 3·log10(OR) + 2·log10(LD);   la P más alta en 30 min gobierna",
    "Adjustment  KI   (clause 8, Formula 2)":
        "Ajuste  KI   (cláusula 8, Fórmula 2)",
    "KI = 1.8·(P − 5) dB for P > 5, else 0":
        "KI = 1.8·(P − 5) dB si P > 5, si no 0",
    "Rating level  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )":
        "Nivel de evaluación  LAr,T = 10·log10( (1/T) Σ Δt·10^((LAeq+KI)/10) )",
    "impulse-adjusted level over the reference time  (Note 1)":
        "nivel ajustado por impulsos sobre el tiempo de referencia  (Nota 1)",
    "Vertical seat acceleration  az(t)":
        "Aceleración vertical del asiento  az(t)",
    "conditioned per 5.1.3:  HP 0.01 Hz (2nd order) / LP 80 Hz (4th order)":
        "acondicionada según 5.1.3:  PA 0,01 Hz (2.º orden) / PB 80 Hz "
        "(4.º orden)",
    "not the ISO 2631-1 0.4 Hz / 100 Hz filters":
        "no los filtros de 0,4 Hz / 100 Hz de ISO 2631-1",
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
    "Reverberation test room": "Cámara reverberante de ensayo",
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
    "V ≥ 200 m³ , qualified room": "V ≥ 200 m³ , cámara cualificada",
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
    "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Formula 26)":
        "R'w = −10 log10 Σ 10^(−Rij,w /10) dB   (EN 12354-1, Fórmula 26)",
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
    "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Eq. 14)":
        "Dz = 10 log10[ 3 + (C₂/λ) C₃ z Kmet ]   (Ec. 14)",
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
    "Turntable and base plate": "Plataforma giratoria y placa base",
    "sample on the plate for T2 and T4":
        "probeta sobre la placa en T2 y T4",
    "the only thing that moves": "lo único que se mueve",
    # "≥ 1.0 m" (the turntable wall clearance) is already in the table above.
    "fixed sources (≥ 2)": "fuentes fijas (≥ 2)",
    "fixed microphones (≥ 3)": "micrófonos fijos (≥ 3)",
    "T1 base plate, static  ·  T2 sample, static  →  α_s (Eq. 1)":
        "T1 placa base, estática  ·  T2 probeta, estática  →  α_s (Ec. 1)",
    "T3 base plate, rotating  ·  T4 sample, rotating  →  α_spec (Eq. 4)":
        "T3 placa base, girando  ·  T4 probeta, girando  →  α_spec (Ec. 4)",
    "s = (α_spec − α_s) / (1 − α_s)   (Eq. 5)":
        "s = (α_spec − α_s) / (1 − α_s)   (Ec. 5)",
    "α from 55.3·(V/S)·(1/cT) − 4(V/S)m  ·  the base plate must pass the "
    "Table 1 ceiling":
        "α con 55,3·(V/S)·(1/cT) − 4(V/S)m  ·  la placa base debe cumplir el "
        "límite de la Tabla 1",
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
        "Potencia acústica de precisión en cámara anecoica (ISO 3745)",
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
    "what open_plan_metrics returns": "lo que devuelve open_plan_metrics",
    "Clause 4 also requires the average A-weighted background "
    "noise  Lp,A,B  (Cl. 6.4)":
        "La cláusula 4 exige además el ruido de fondo medio ponderado A  "
        "Lp,A,B  (cl. 6.4)",
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
    "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (far field, inverse-distance law)":
        "Lp(1 m) = Lp(r) + 20 log10(r / 1 m)   (campo lejano, ley 1/r)",
    "Microphone (IEC 60268-4): M in mV/Pa, or LM = 20 log10(M / 1 V/Pa) dB":
        "Micrófono (IEC 60268-4): M en mV/Pa, o LM = 20 log10(M / 1 V/Pa) dB",
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
    "spatial average:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Formula 12)":
        "promedio espacial:  Lv = 10 log10[(1/N)·Σ 10^(Lv,i/10)]   (Fórmula 12)",
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
    "LWA,i = Lp,i − 6 + 10 log10(4π R1² / S0)   (Formula 26, S0 = 1 m²)":
        "LWA,i = Lp,i − 6 + 10 log10(4π R1² / S0)   (Fórmula 26, S0 = 1 m²)",
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
    "Kij = D̄v,ij + 10 log10( lij / √(ai·aj) ),   ai = equivalent absorption length":
        "Kij = D̄v,ij + 10 log10( lij / √(ai·aj) ),   ai = long. de "
        "absorción equiv.",
    # Sound power from surface vibration (ISO/TS 7849)
    "Sound power from surface vibration (ISO/TS 7849)":
        "Potencia acústica a partir de la vibración superficial (ISO/TS 7849)",
    "Vibrating measurement surface S": "Superficie de medición vibrante S",
    "Machine under test": "Máquina en ensayo",
    "radiated airborne sound": "sonido aéreo radiado",
    "Initial number of positions N": "Número inicial de posiciones N",
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
    # Atmospheric refraction (Salomons / Attenborough & Van Renterghem)
    "Atmospheric refraction: downwind multipath and the upwind shadow":
        "Refracción atmosférica: multitrayecto y sombra por el viento",
    "wind u(z)": "viento u(z)",
    "acoustic shadow": "sombra acústica",
    "1.5 m": "1,5 m",
    "Upwind: rays bend up; beyond ≈ 220 m a ground shadow opens and the level collapses by over 20 dB":
        "Contra el viento: los rayos suben; desde ≈ 220 m se abre una "
        "sombra y el nivel cae más de 20 dB",
    "Downwind: rays bend down; the receiver hears the direct and the ground-bounced arrival (multipath)":
        "A favor del viento: los rayos bajan y llegan la directa y la "
        "rebotada en el suelo (multitrayecto)",
    "a ±0.1 (m/s)/m gradient curves rays with radius Rc = c0/|g| ≈ 3.4 km; source hs = 2 m, receiver hr = 1.5 m":
        "gradiente de ±0,1 (m/s)/m → radio Rc = c0/|g| ≈ 3,4 km; fuente "
        "hs = 2 m, receptor hr = 1,5 m",
    # Aircraft noise certification points (ICAO Annex 16, Chapter 3)
    "Aircraft noise certification points (ICAO Annex 16, Chapter 3)":
        "Certificación de ruido de aeronaves (Anexo 16 OACI, Capítulo 3)",
    "Side view": "Vista lateral",
    "start of roll": "inicio de rodadura",
    "runway": "pista",
    "take-off": "despegue",
    "approach": "aproximación",
    "Approach reference point": "Punto de referencia de aproximación",
    "Flyover reference point": "Punto de referencia de sobrevuelo",
    "Lateral reference line": "Línea lateral de referencia",
    "where take-off noise is greatest": "donde el ruido de despegue es máximo",
    "symmetric lateral point (measured on both sides)":
        "punto lateral simétrico (se mide a ambos lados)",
    "Microphones 1.2 m above the ground; the certification metric at the three points is EPNL, in EPNdB":
        "Micrófonos a 1,2 m sobre el suelo; la métrica de certificación en "
        "los tres puntos es el EPNL, en EPNdB",
    "Lateral: full take-off power · Flyover: 6.5 km from brake release · Approach: 3° ± 0.5° glide path":
        "Lateral: máxima potencia de despegue · Sobrevuelo: a 6,5 km de "
        "soltar frenos · Aproximación: senda de 3° ± 0,5°",
    "the approach point lies 120 m below the 3° path, which meets the ground 300 m beyond the threshold":
        "el punto de aproximación queda 120 m bajo la senda de 3°, que corta "
        "el suelo 300 m más allá del umbral",
    # Helicopter overflight certification (ICAO Annex 16, Chapter 8)
    "Helicopter overflight noise certification (ICAO Annex 16, Chapter 8)":
        "Certificación de ruido de helicópteros (Anexo 16 OACI, Capítulo 8)",
    "level flight at 0.9 VH": "vuelo nivelado a 0,9 VH",
    "centre microphone": "micrófono central",
    "track": "trayectoria",
    "3 microphones on a line perpendicular to the track":
        "3 micrófonos en una línea perpendicular a la trayectoria",
    "Speed: the least of 0.9 VH, 0.9 VNE, 0.45 VH + 120 km/h and 0.45 VNE + 120 km/h":
        "Velocidad: la menor de 0,9 VH, 0,9 VNE, 0,45 VH + 120 km/h y "
        "0,45 VNE + 120 km/h",
    "EPNL in EPNdB at the three points; at least six overflights, headwind and tailwind in equal number":
        "EPNL en EPNdB en los tres puntos; al menos seis pasadas, con viento "
        "en cara y en cola a partes iguales",
    "microphones 1.2 m above ground; the sideline pair sees the overhead helicopter at 45° (slant ≈ 212 m)":
        "micrófonos a 1,2 m del suelo; el par lateral ve el helicóptero en "
        "la vertical con 45° (oblicua ≈ 212 m)",
    # Swept-sine distortion (Farina / Novak)
    "Swept-sine distortion: deconvolution and harmonic pre-arrivals":
        "Distorsión por barrido senoidal: deconvolución y prellegadas "
        "armónicas",
    "Exponential sweep x(t)": "Barrido exponencial x(t)",
    "20 Hz → 6 kHz in T = 4 s": "20 Hz → 6 kHz en T = 4 s",
    "Device under test": "Dispositivo en ensayo",
    "weakly nonlinear: gain + harmonics":
        "débilmente no lineal: ganancia + armónicos",
    "Recording y(t)": "Grabación y(t)",
    "sweep + distortion products": "barrido + productos de distorsión",
    "Deconvolve with the inverse filter":
        "Deconvolución con el filtro inverso",
    "time-reversed sweep with a +6 dB/octave tilt":
        "barrido invertido en el tiempo, con +6 dB/octava",
    "h1 (linear), t = 0": "h1 (lineal), t = 0",
    "harmonic orders arrive early,": "los órdenes armónicos llegan antes,",
    "each in its own window": "cada uno en su propia ventana",
    "L·ln 2 = 0.49 s": "L·ln 2 = 0,49 s",
    "L·ln 3 = 0.77 s": "L·ln 3 = 0,77 s",
    "time": "tiempo",
    "L = T / ln(f2/f1) = 0.70 s here; the order-n products compress L·ln n ahead of the linear response":
        "L = T / ln(f2/f1) = 0,70 s aquí; los productos de orden n se "
        "comprimen L·ln n antes de la respuesta lineal",
    "window each arrival  →  H1(f), H2(f), H3(f), …  →  THD(f) = √( Σ |Hn(nf)|² ) / |H1(f)|":
        "enventanar cada llegada  →  H1(f), H2(f), H3(f), …  →  "
        "THD(f) = √( Σ |Hn(nf)|² ) / |H1(f)|",
    # Two-channel FRF measurement (H1 estimator and coherence)
    "Two-channel FRF measurement: the H1 estimator and coherence":
        "Medición de FRF a dos canales: el estimador H1 y la coherencia",
    "Signal generator": "Generador de señal",
    "broadband noise or a sweep": "ruido de banda ancha o barrido",
    "Power amplifier": "Amplificador",
    "Loudspeaker under test": "Altavoz en ensayo",
    "measurement microphone": "micrófono de medición",
    "Channel 1: reference x(t)": "Canal 1: referencia x(t)",
    "the electrical drive signal": "la señal eléctrica de excitación",
    "Channel 2: response y(t)": "Canal 2: respuesta y(t)",
    "acoustic output at the microphone": "salida acústica en el micrófono",
    "Dual-channel FFT analysis (Welch)":
        "Análisis FFT de dos canales (Welch)",
    "Hann segments, 50 % overlap  →  Gxx(f), Gyy(f), Gxy(f)":
        "segmentos Hann, 50 % de solape  →  Gxx(f), Gyy(f), Gxy(f)",
    "unbiased with output noise; H2 = Gyy/Gyx for input noise":
        "insesgado ante ruido a la salida (H2 = Gyy/Gyx a la entrada)",
    "1 for a noiseless linear path; less with output noise":
        "1 en un camino lineal sin ruido; menor con ruido a la salida",
    "trust |H1| only where γ² stays near 1: coherence dips flag noise, distortion or an unresolved delay":
        "fiarse de |H1| solo donde γ² ronda 1: las caídas delatan ruido, "
        "distorsión o retardo sin resolver",
    # Test-signal family panel
    "The test-signal family at a glance":
        "La familia de señales de ensayo de un vistazo",
    "White noise": "Ruido blanco",
    "Pink noise": "Ruido rosa",
    "Sweeps: linear vs exponential": "Barridos: lineal frente a exponencial",
    "Tone burst": "Salva de tono",
    "flat PSD: 0 dB/octave": "DEP plana: 0 dB/octava",
    "equal power per hertz": "igual potencia por hercio",
    "−3 dB/octave PSD": "DEP de −3 dB/octava",
    "equal power per octave": "igual potencia por octava",
    "flat, line spectrum": "espectro de rayas plano",
    "binary ±1, period 2^m − 1 samples": "binaria ±1, periodo 2^m − 1 muestras",
    "linear": "lineal",
    "exponential": "exponencial",
    "exponential: equal time (and energy) per octave; linear: equal time per hertz":
        "exponencial: igual tiempo (y energía) por octava; lineal: igual "
        "tiempo por hercio",
    "whole periods, starting at": "periodos completos, empezando en",
    "a zero crossing (IEC 60268-1)": "un paso por cero (IEC 60268-1)",
    "25 periods of 5 kHz = 5 ms": "25 periodos de 5 kHz = 5 ms",
    "every stimulus is deterministic and repeatable; synchronous averaging then lowers uncorrelated noise":
        "todos los estímulos son deterministas y repetibles; promediar "
        "pasadas reduce el ruido no correlacionado",
    "sweeps separate harmonic distortion, MLS smears it across the period, bursts probe dynamics":
        "los barridos separan la distorsión armónica, la MLS la reparte por "
        "el periodo y las salvas sondean la dinámica",
    # Welch PSD pipeline (Bendat & Piersol)
    "The Welch PSD pipeline: segment, taper, average (Bendat & Piersol)":
        "PSD de Welch: segmentar, enventanar, promediar (Bendat & Piersol)",
    "Record x(t) — fs = 48 kHz, 20 s of pink noise":
        "Registro x(t) — fs = 48 kHz, 20 s de ruido rosa",
    "960 000 samples, calibrated end to end: pascals in, Pa²/Hz out":
        "960 000 muestras, calibrado de extremo a extremo: pascales dentro, "
        "Pa²/Hz fuera",
    "Split into 50 %-overlapped segments — nperseg = 4096":
        "División en segmentos con 50 % de solape — nperseg = 4096",
    "467 segments of 85.3 ms; bin spacing Δf = fs/4096 = 11.7 Hz":
        "467 segmentos de 85,3 ms; separación de bins Δf = fs/4096 = "
        "11,7 Hz",
    "Hann taper on every segment": "Ventana de Hann en cada segmento",
    "ENBW = 1.5 bins → resolution bandwidth Be = 1.5·Δf = 17.6 Hz":
        "ENBW = 1,5 bins → ancho de banda de resolución Be = 1,5·Δf = "
        "17,6 Hz",
    "One-sided |FFT|² periodogram of each segment, then average":
        "Periodograma unilateral |FFT|² de cada segmento, y promedio",
    "overlap correlation (Welch 1967): 467 segments → n_d = 442 effective averages":
        "correlación por solape (Welch 1967): 467 segmentos → n_d = 442 "
        "promedios efectivos",
    "Gxx(f) with its chi-square confidence interval":
        "Gxx(f) con su intervalo de confianza chi-cuadrado",
    "random error εr = 1/√n_d = 4.8 %;  2·n_d ≈ 885 degrees of freedom":
        "error aleatorio εr = 1/√n_d = 4,8 %;  2·n_d ≈ 885 grados de "
        "libertad",
    "The trade-off: segment length buys resolution or stability, never both":
        "El compromiso: el segmento compra resolución o estabilidad, "
        "nunca ambas",
    "longer segments → finer Be but fewer averages (larger εr); shorter → the reverse":
        "segmentos más largos → Be más fino pero menos promedios (mayor "
        "εr); más cortos → lo contrario",
    # MISO coherence (Bendat & Piersol Chapter 7)
    "MISO coherence: from correlated sources to per-source contributions":
        "Coherencia MISO: de fuentes correladas a contribuciones por fuente",
    "Input x1": "Entrada x1",
    "white noise": "ruido blanco",
    "Input x2 = 0.7·x1 + noise": "Entrada x2 = 0,7·x1 + ruido",
    "correlated with x1": "correlada con x1",
    "Path H1(f)": "Camino H1(f)",
    "low-pass, 400 Hz": "paso bajo, 400 Hz",
    "Path H2(f)": "Camino H2(f)",
    "high-pass, 1.5 kHz": "paso alto, 1,5 kHz",
    "noise n(t)": "ruido n(t)",
    "Output y(t)": "Salida y(t)",
    "Welch cross-spectral matrix — Gxx (2×2) and Gxy, nperseg = 2048":
        "Matriz de espectros cruzados de Welch — Gxx (2×2) y Gxy, "
        "nperseg = 2048",
    "conditioning: Schur steps Gij·r! (Eq. 7.94), inputs ordered by descending ordinary coherence":
        "condicionamiento: pasos de Schur Gij·r! (Ec. 7.94), entradas "
        "ordenadas por coherencia ordinaria descendente",
    "Multiple and partial coherence": "Coherencia múltiple y parcial",
    "input 2 in the 100-300 Hz band: ordinary 0.32 → partial 0.00":
        "entrada 2 en la banda de 100-300 Hz: ordinaria 0,32 → parcial "
        "0,00",
    "multiple γ²y:x = 1 − Gnn/Gyy ≈ 1.00 (100-300 Hz)":
        "múltiple γ²y:x = 1 − Gnn/Gyy ≈ 1,00 (100-300 Hz)",
    "Contribution of each source": "Contribución de cada fuente",
    "Gvi = γ²iy·(i−1)!·Gyy per input": "Gvi = γ²iy·(i−1)!·Gyy por entrada",
    "ΣGvi + Gnn = Gyy, band by band": "ΣGvi + Gnn = Gyy, banda a banda",
    "each conditioning step spends one average: the i-th ordered input carries n_d − (i − 1); here n_d = 242":
        "cada paso de condicionamiento gasta un promedio: la entrada "
        "i-ésima ordenada lleva n_d − (i − 1); aquí n_d = 242",
    "average generously before reading a small partial coherence as zero":
        "promedia con generosidad antes de leer como cero una coherencia "
        "parcial pequeña",
    # Time-frequency tiling trade-off
    "The time-frequency trade-off: two tilings of the same record":
        "El compromiso tiempo-frecuencia: dos teselados del mismo registro",
    "Short window — nperseg = 256": "Ventana corta — nperseg = 256",
    "Long window — nperseg = 1024": "Ventana larga — nperseg = 1024",
    "T_B = 16 ms,  Be ≈ 1/T_B = 62.5 Hz":
        "T_B = 16 ms,  Be ≈ 1/T_B = 62,5 Hz",
    "T_B = 64 ms,  Be ≈ 15.6 Hz": "T_B = 64 ms,  Be ≈ 15,6 Hz",
    "sharp click, smeared tone": "clic nítido, tono emborronado",
    "sharp tone, smeared click": "tono nítido, clic emborronado",
    "tone": "tono",
    "click": "clic",
    "each cell is one unaveraged estimate: Be·T_B ≈ 1 and εr = 1 (n_d = 1)":
        "cada celda es una estimación sin promediar: Be·T_B ≈ 1 y εr = 1 "
        "(n_d = 1)",
    "the record fixes the product; nperseg only chooses how to spend it (fs = 16 kHz here)":
        "el registro fija el producto; nperseg solo elige cómo gastarlo "
        "(aquí fs = 16 kHz)",
    # Cepstrum chain (Havelock Ch. 27)
    "The cepstrum chain: an echo becomes a quefrency spike":
        "La cadena del cepstro: un eco se vuelve un pico en quefrencia",
    "Signal with one echo": "Señal con un eco",
    "Ripply spectrum |X(f)|": "Espectro ondulado |X(f)|",
    "cosine ripple of period": "ondulación coseno de periodo",
    "Take the log: ln |X|²": "Logaritmo: ln |X|²",
    "the multiplicative echo": "el eco multiplicativo",
    "becomes an additive ripple": "se vuelve ondulación aditiva",
    "Inverse FFT": "FFT inversa",
    "quefrency axis, in seconds": "eje de quefrencia, en segundos",
    "the cepstrum": "el cepstro",
    "quefrency": "quefrencia",
    "source wavelet,": "ondícula de la fuente,",
    "below 2 ms": "bajo 2 ms",
    "a = 0.5,  t0 = 8 ms": "a = 0,5;  t0 = 8 ms",
    "a = 0.5 at t0 = 8 ms": "a = 0,5 en t0 = 8 ms",
    "−a²/2 = −0.125": "−a²/2 = −0,125",
    "lifter cutoff 4 ms": "corte del lifter en 4 ms",
    "lowpass: envelope": "paso bajo: envolvente",
    "highpass: the echo ripple alone":
        "paso alto: solo la ondulación del eco",
    "rahmonics at n·t0 with heights a, −a²/2, a³/3, …, whatever the source spectrum does":
        "rahmónicos en n·t0 con alturas a, −a²/2, a³/3, …, haga lo que "
        "haga el espectro de la fuente",
    "the highpass ripple swings between 20·log10(1 ± a) = +3.5 and −6.0 dB; echo_detection reads t0 and a off the peak":
        "la ondulación paso alto oscila entre 20·log10(1 ± a) = +3,5 y "
        "−6,0 dB; echo_detection lee t0 y a del pico",
    # Time synchronous averaging (McFadden 1987)
    "Time synchronous averaging: trigger, slice, average":
        "Promediado síncrono temporal: disparo, troceado, promedio",
    "Tachometer: one trigger pulse per revolution":
        "Tacómetro: un pulso de disparo por revolución",
    "T = 1/32 s = 256 samples": "T = 1/32 s = 256 muestras",
    "Recording y(t) at fs = 8192 Hz: the synchronous signature buried in noise":
        "Registro y(t) a fs = 8192 Hz: la firma síncrona sepultada en ruido",
    "slice at every trigger": "trocear en cada disparo",
    "N aligned blocks": "N bloques alineados",
    "one period T each": "de un periodo T cada uno",
    "Coherent average": "Promedio coherente",
    "N = 40 here": "aquí N = 40",
    "The periodic part survives": "La parte periódica sobrevive",
    "comb teeth of unit gain": "dientes del peine de ganancia uno",
    "at every order k/T": "en cada orden k/T",
    "Asynchronous noise falls as 1/√N":
        "El ruido asíncrono cae como 1/√N",
    "power −10·log10 N = −16 dB for N = 40;  amplitude gain √N = 6.3":
        "potencia −10·log10 N = −16 dB con N = 40;  ganancia en amplitud "
        "√N = 6,3",
    "Residual": "Residual",
    "record − tiled average:": "registro − promedio repetido:",
    "everything not synchronous": "todo lo no síncrono",
    "a tone on a non-integer order is only attenuated: choose N so a comb node lands on it":
        "un tono en un orden no entero solo se atenúa: elige N para que un "
        "nodo del peine caiga sobre él",
    "McFadden's example: N = 20 nulls the 32.05-order tone (20·32.05 = 641); the habitual N = 32 does not":
        "el ejemplo de McFadden: N = 20 anula el tono de orden 32,05 "
        "(20·32,05 = 641); el habitual N = 32 no",
    # Correlation-based time-delay estimation (Knapp & Carter)
    "Time-delay estimation: two microphones and one correlation peak":
        "Estimación del retardo: dos micrófonos y un pico de correlación",
    "Δr = c·τ0 ≈ 0.84 m  (c = 343 m/s)":
        "Δr = c·τ0 ≈ 0,84 m  (c = 343 m/s)",
    "mic 1 — x(t)": "micro 1 — x(t)",
    "mic 2 — y(t)": "micro 2 — y(t)",
    "spacing d": "separación d",
    "cross-correlation against lag — y(t) = α·x(t − τ0) + n(t)":
        "correlación cruzada frente al retardo — y(t) = α·x(t − τ0) + n(t)",
    "direct correlator: broad peak": "correlador directo: pico ancho",
    "GCC-PHAT: sharp spike": "GCC-PHAT: pico estrecho",
    "τ0 = 20 samples / 8192 Hz = 2.44 ms":
        "τ0 = 20 muestras / 8192 Hz = 2,44 ms",
    "parabolic peak interpolation + ×16 local upsampling → error below 0.002 samples":
        "interpolación parabólica del pico + sobremuestreo local ×16 → "
        "error por debajo de 0,002 muestras",
    "the 'phase' route reads the same τ0 from the slope of the unwrapped cross-spectrum phase":
        "la vía 'phase' lee el mismo τ0 de la pendiente de la fase "
        "desenrollada del espectro cruzado",
    # Data qualification decision flow (Bendat & Piersol 10.3)
    "Data qualification: the stationarity decision (Bendat & Piersol 10.3)":
        "Calificación de datos: decisión de estacionariedad (B&P 10.3)",
    "Time record x(t)": "Registro temporal x(t)",
    "before trusting any PSD, Leq or GUM average":
        "antes de confiar en cualquier promedio PSD, Leq o GUM",
    "Mean square per interval — N = 20 equal segments":
        "Media cuadrática por intervalo — N = 20 segmentos iguales",
    "each interval long against the record's lowest frequencies; also rms, mean or variance":
        "cada intervalo largo frente a las frecuencias más bajas del "
        "registro; también rms, media o varianza",
    "Reverse arrangement count A": "Recuento de inversiones A",
    "pairs i < j with x_i > x_j; trend-free mean μ_A = N(N−1)/4 = 95":
        "pares i < j con x_i > x_j; media sin tendencia μ_A = N(N−1)/4 = 95",
    "(Table A.6, α = 0.05)": "(Tabla A.6, α = 0,05)",
    "yes": "sí",
    "Nonstationary: do not average": "No estacionario: no promediar",
    "+20 % gain ramp: A = 7 → rejected":
        "rampa de ganancia del +20 %: A = 7 → rechazado",
    "split at the change, or go short-time (spectrogram)":
        "divide en el cambio, o pasa a corto plazo (espectrograma)",
    "Stationary: analyse": "Estacionario: analizar",
    "steady noise: A = 91 → accepted":
        "ruido estable: A = 91 → aceptado",
    "the chi-square CIs and error formulas hold":
        "los IC chi-cuadrado y las fórmulas de error valen",
    "the runs test (method=\"runs\") is the two-sided companion: too many runs is as suspect as too few":
        "el test de rachas (method=\"runs\") es el compañero bilateral: "
        "demasiadas rachas son tan sospechosas como muy pocas",
    "a frequency glide can hide from the mean square: test statistic=\"mean\" or band-filtered copies too":
        "un deslizamiento en frecuencia puede esconderse de la media "
        "cuadrática: prueba statistic=\"mean\" o copias filtradas por "
        "bandas",
    # Sound-quality metric family (DIN 45692 + ECMA-418-2)
    "Sound quality beyond loudness: four calibrated sensations":
        "Calidad sonora más allá de la sonoridad: cuatro sensaciones",
    "Calibrated signal x(t) in pascals":
        "Señal calibrada x(t) en pascales",
    "any sample rate: each metric resamples to 48 kHz internally":
        "cualquier fs: cada métrica remuestrea internamente a 48 kHz",
    "Specific loudness N'(z)": "Sonoridad específica N'(z)",
    "Zwicker pattern over 24 Bark": "patrón de Zwicker sobre 24 Bark",
    "Sottek Hearing Model front end (ECMA-418-2)":
        "Etapa de entrada del modelo de Sottek (ECMA-418-2)",
    "outer/middle-ear filter + 53 auditory bands (Bark_HMS)":
        "filtro de oído externo/medio + 53 bandas auditivas (Bark_HMS)",
    "Sharpness S": "Agudeza S",
    "g(z)-weighted first moment": "primer momento ponderado",
    "of N'(z), with k = 0.108": "por g(z) de N'(z), k = 0,108",
    "critical-band-wide noise": "ruido de una banda crítica",
    "at 1 kHz, 60 dB": "a 1 kHz, 60 dB",
    "→ S = 1.00 acum": "→ S = 1,00 acum",
    "Tonality T": "Tonalidad T",
    "ECMA-418-2 clause 6": "ECMA-418-2, cláusula 6",
    "band autocorrelation finds": "la autocorrelación por banda",
    "periodic components": "detecta componentes periódicas",
    "1 kHz tone at 40 dB": "tono de 1 kHz a 40 dB",
    "→ T = 1.000 tu_HMS (999 Hz)": "→ T = 1,000 tu_HMS (999 Hz)",
    "Roughness R": "Aspereza R",
    "ECMA-418-2 clause 7": "ECMA-418-2, cláusula 7",
    "fast envelope modulation,": "modulación rápida de la envolvente,",
    "band-pass peaking near 70 Hz": "paso banda con pico hacia 70 Hz",
    "1 kHz, 100 % AM at 70 Hz, 60 dB": "1 kHz, AM 100 % a 70 Hz, 60 dB",
    "→ R = 0.9999 asper": "→ R = 0,9999 asper",
    "Fluctuation strength F": "Intensidad de fluctuación F",
    "ECMA-418-2 clause 9 (HSA)": "ECMA-418-2, cláusula 9 (HSA)",
    "slow envelope modulation,": "modulación lenta de la envolvente,",
    "band-pass peaking near 4 Hz": "paso banda con pico hacia 4 Hz",
    "1 kHz, 100 % AM at 4 Hz, 60 dB": "1 kHz, AM 100 % a 4 Hz, 60 dB",
    "→ F = 0.9957 vacil_HMS": "→ F = 0,9957 vacil_HMS",
    "Downstream, the sensations combine into annoyance":
        "Aguas abajo, las sensaciones se combinan en molestia",
    "N5, S, R and F feed the Fastl and Zwicker psychoacoustic annoyance PA = N5·(1 + √(wS² + wFR²))":
        "N5, S, R y F alimentan la molestia psicoacústica de Fastl y "
        "Zwicker PA = N5·(1 + √(wS² + wFR²))",
    # Tone audibility (ISO/PAS 20065)
    "Tone audibility: from spectrum to penalty (ISO/PAS 20065)":
        "Audibilidad tonal: del espectro al ajuste (ISO/PAS 20065)",
    "Narrow-band FFT spectrum — line spacing Δf = 2.7 Hz":
        "Espectro FFT de banda estrecha — resolución Δf = 2,7 Hz",
    "Annex E engine spectrum; peak detected at fT = 137.3 Hz (not on a slope)":
        "espectro del motor del Anexo E; pico detectado en fT = 137,3 Hz "
        "(no en una ladera)",
    "Critical band about the tone — Δfc = 101.36 Hz":
        "Banda crítica en torno al tono — Δfc = 101,36 Hz",
    "geometric placement: corners 95.67 and 197.04 Hz, √(f1·f2) = fT":
        "colocación geométrica: esquinas en 95,67 y 197,04 Hz, "
        "√(f1·f2) = fT",
    "Levels from the spectrum lines in the band":
        "Niveles desde las líneas del espectro en la banda",
    "masking noise LS = 49.22 dB (iterative mean); tone LT = 67.96 dB (energy sum)":
        "ruido enmascarante LS = 49,22 dB (media iterativa); tono "
        "LT = 67,96 dB (suma energética)",
    "Masking threshold seen by the ear":
        "Umbral de enmascaramiento visto por el oído",
    "LG = LS + 10·log10(Δfc/Δf) = 64.97 dB;  masking index av = −2.02 dB":
        "LG = LS + 10·log10(Δfc/Δf) = 64,97 dB;  índice de enmascaramiento "
        "av = −2,02 dB",
    "Audibility ΔL = LT − LG − av = 5.01 dB":
        "Audibilidad ΔL = LT − LG − av = 5,01 dB",
    "the largest ΔL of the nine tones: the decisive audibility of this spectrum":
        "el mayor ΔL de los nueve tonos: la audibilidad decisiva de este "
        "espectro",
    "From audibility to penalty (ISO 1996-2:2017 Annex J)":
        "De la audibilidad al ajuste (ISO 1996-2:2017, Anexo J)",
    "energy mean of the five spectra ΔL = 6.98 dB → tonal adjustment Kt = 4 dB (Table J.1)":
        "media energética de los cinco espectros ΔL = 6,98 dB → ajuste "
        "tonal Kt = 4 dB (Tabla J.1)",
    # Psychoacoustic annoyance (Fastl & Zwicker)
    "Psychoacoustic annoyance: four sensations, one scalar":
        "Molestia psicoacústica: cuatro sensaciones, un escalar",
    "S = 2.0 acum": "S = 2,0 acum",
    "sharpness (DIN 45692)": "agudeza (DIN 45692)",
    "counts only above 1.75 acum": "solo cuenta sobre 1,75 acum",
    "N5 = 30 sone": "N5 = 30 sone",
    "percentile loudness (ISO 532-1)": "sonoridad percentil (ISO 532-1)",
    "exceeded 5 % of the time": "superada el 5 % del tiempo",
    "F = 0.5 vacil": "F = 0,5 vacil",
    "fluctuation strength": "intensidad de fluctuación",
    "slow modulation, ≈ 4 Hz": "modulación lenta, ≈ 4 Hz",
    "R = 0.3 asper": "R = 0,3 asper",
    "roughness": "aspereza",
    "fast modulation, ≈ 70 Hz": "modulación rápida, ≈ 70 Hz",
    "Sharpness weighting wS = 0.1001":
        "Ponderación de la agudeza wS = 0,1001",
    "wS = (S − 1.75) · 0.25 · log10(N5 + 10)":
        "wS = (S − 1,75) · 0,25 · log10(N5 + 10)",
    "zero for S ≤ 1.75 acum": "cero para S ≤ 1,75 acum",
    "Roughness and fluctuation wFR = 0.2125":
        "Aspereza y fluctuación wFR = 0,2125",
    "wFR = 2.18 / N5^0.4 · (0.4·F + 0.6·R)":
        "wFR = 2,18 / N5^0,4 · (0,4·F + 0,6·R)",
    "roughness weighs more: 0.6 against 0.4":
        "la aspereza pesa más: 0,6 frente a 0,4",
    "PA = N5 · (1 + √(wS² + wFR²)) = 37.05":
        "PA = N5 · (1 + √(wS² + wFR²)) = 37,05",
    "Fastl and Zwicker Eq. 16.2 (origin Widmann 1992)":
        "Fastl y Zwicker, Ec. 16.2 (origen Widmann 1992)",
    "a neutral sound (S ≤ 1.75 acum, F = R = 0) sits on the baseline PA = N5":
        "un sonido neutro (S ≤ 1,75 acum, F = R = 0) queda en la línea "
        "base PA = N5",
    "sharpness, roughness and fluctuation only ever lift the annoyance above the loudness":
        "la agudeza, la aspereza y la fluctuación solo elevan la "
        "molestia por encima de la sonoridad",
    # Objective intelligibility (STOI / ESTOI)
    "STOI and ESTOI: correlating clean against degraded speech":
        "STOI y ESTOI: correlación entre habla limpia y degradada",
    "Clean reference x(t) and degraded version y(t)":
        "Referencia limpia x(t) y versión degradada y(t)",
    "the guide's example: speech-like material in a flat masker at 0 dB SNR":
        "el ejemplo de la guía: material tipo habla en un enmascarador "
        "plano a 0 dB de SNR",
    "Resample to 10 kHz and drop the silent frames":
        "Remuestreo a 10 kHz y descarte de tramas silenciosas",
    "frames 40 dB below the loudest clean frame carry no intelligibility":
        "las tramas 40 dB bajo la trama limpia más alta no aportan "
        "inteligibilidad",
    "Short-time DFT: 256-sample Hann frames, 50 % overlap":
        "DFT de corto plazo: tramas Hann de 256 muestras, 50 % de solape",
    "magnitudes grouped into 15 one-third-octave bands from 150 Hz":
        "magnitudes agrupadas en 15 bandas de tercio de octava desde "
        "150 Hz",
    "384 ms segments — 30 frames, the unit of comparison":
        "Segmentos de 384 ms — 30 tramas, la unidad de comparación",
    "long enough to hold the slow modulations that carry speech":
        "lo bastante largos para las modulaciones lentas que llevan el "
        "habla",
    "STOI: envelope correlation": "STOI: correlación de envolventes",
    "per band and segment; normalise,": "por banda y segmento; normaliza,",
    "clip at −15 dB, then average": "recorta en −15 dB y promedia",
    "ESTOI: spectral correlation": "ESTOI: correlación espectral",
    "row- and column-normalised segments;":
        "segmentos normalizados por filas y columnas;",
    "credits glimpses in modulated maskers":
        "acredita los atisbos en enmascaradores modulados",
    "STOI = 0.727 for the example": "STOI = 0,727 para el ejemplo",
    "the lowest band keeps 0.27 of the correlation; above 1.9 kHz it reaches 0.90":
        "la banda más baja conserva 0,27 de la correlación; sobre "
        "1,9 kHz llega a 0,90",
    # Programme loudness (ITU-R BS.1770 / EBU R 128)
    "Programme loudness: the BS.1770 / R 128 metering chain":
        "Sonoridad de programa: la cadena de medición BS.1770 / R 128",
    "Programme x — channel weights Gi: 1.0 front, 1.41 surround":
        "Programa x — pesos de canal Gi: 1,0 frontales, 1,41 envolventes",
    "anchor: a 0 dB FS 997 Hz sine on one front channel reads −3.01 LKFS":
        "ancla: un seno de 997 Hz a 0 dB FS en un canal frontal marca "
        "−3,01 LKFS",
    "K-weighting: +4 dB spherical-head shelf + RLB high-pass":
        "Ponderación K: estante de +4 dB (cabeza esférica) + paso alto RLB",
    "LK = −0.691 + 10·log10 Σ Gi·zi;  LKFS ≡ LUFS, 1 LU = 1 dB":
        "LK = −0,691 + 10·log10 Σ Gi·zi;  LKFS ≡ LUFS, 1 LU = 1 dB",
    "Mean square in 400 ms blocks, 75 % overlap":
        "Media cuadrática en bloques de 400 ms, 75 % de solape",
    "absolute gate: blocks below −70 LUFS are dropped":
        "puerta absoluta: se descartan los bloques bajo −70 LUFS",
    "Relative gate: −10 LU below the survivors":
        "Puerta relativa: −10 LU bajo los supervivientes",
    "example: 10 s at −23 dBFS + 30 s of quiet → threshold −39.0 LUFS":
        "ejemplo: 10 s a −23 dBFS + 30 s de silencio → umbral −39,0 LUFS",
    "Integrated loudness I = −23.1 LUFS: the tail is gated out":
        "Sonoridad integrada I = −23,1 LUFS: la cola queda fuera",
    "EBU R 128 target −23.0 LUFS; tolerance ±0.2 LU in QC, ±1.0 LU live":
        "objetivo EBU R 128 −23,0 LUFS; tolerancia ±0,2 LU en QC, "
        "±1,0 LU en directo",
    "Loudness range LRA = P95 − P10": "Rango de sonoridad LRA = P95 − P10",
    "short-term 3 s windows, deeper −20 LU gate":
        "ventanas de corto plazo de 3 s, puerta más honda de −20 LU",
    "10.0 LU on the Tech 3342 two-step case":
        "10,0 LU en el caso de dos escalones de Tech 3342",
    "True peak: 4× oversampling, in dBTP":
        "Pico verdadero: sobremuestreo 4×, en dBTP",
    "the fs/4 tone: sample peak −3.01 dB, true peak +0.12 dBTP":
        "el tono a fs/4: pico muestral −3,01 dB, pico verdadero "
        "+0,12 dBTP",
    "R 128 production ceiling −1 dBTP":
        "techo de producción de R 128: −1 dBTP",
    "K-weighted": "ponderada K",
    "raw signal": "señal sin filtrar",
    "the gates keep quiet passages from dragging the foreground down":
        "las puertas evitan que los pasajes silenciosos arrastren el "
        "primer plano",
    "ungated, the same 40 s example would read near −29 LUFS":
        "sin puertas, el mismo ejemplo de 40 s marcaría cerca de −29 LUFS",
    # Reverberation-time prediction (Sabine / Eyring)
    "Predicting the reverberation time: Sabine against Eyring":
        "Predicción del tiempo de reverberación: Sabine frente a Eyring",
    "Room 10 × 7 × 3.5 m — V = 245 m³, S = 259 m²":
        "Sala de 10 × 7 × 3,5 m — V = 245 m³, S = 259 m²",
    "hard end walls, lightly treated side walls, carpet and acoustic ceiling":
        "testeros duros, laterales con tratamiento ligero, moqueta y "
        "techo acústico",
    "mean absorption ᾱ runs from 0.21 at 125 Hz to 0.51 at 4 kHz":
        "la absorción media ᾱ va de 0,21 a 125 Hz a 0,51 a 4 kHz",
    "T = 0.161·V / (Σ Si·αi + 4mV)": "T = 0,161·V / (Σ Si·αi + 4mV)",
    "low, even absorption (ᾱ up to ≈ 0.2);":
        "absorción baja y uniforme (ᾱ hasta ≈ 0,2);",
    "stays finite even at α = 1": "queda finita incluso con α = 1",
    "T = 0.161·V / (−S·ln(1 − ᾱ) + 4mV)":
        "T = 0,161·V / (−S·ln(1 − ᾱ) + 4mV)",
    "strong, even absorption;": "absorción fuerte y uniforme;",
    "reaches T = 0 at total absorption":
        "llega a T = 0 con absorción total",
    "Predicted T60 per octave band": "T60 predicho por banda de octava",
    "0.74": "0,74", "0.47": "0,47", "0.37": "0,37", "0.31": "0,31",
    "0.30": "0,30", "0.66": "0,66", "0.39": "0,39", "0.29": "0,29",
    "0.23": "0,23", "0.21": "0,21", "0.22": "0,22",
    "Eyring runs 11 to 29 % shorter here: ᾱ is past Sabine's comfort zone":
        "Eyring sale entre un 11 y un 29 % más corto: ᾱ excede la zona "
        "cómoda de Sabine",
    "Domain of validity: a diffuse field that stays diffuse while it decays":
        "Dominio de validez: un campo difuso que sigue difuso mientras "
        "decae",
    "below the Schroeder frequency, in coupled volumes and in corridor-like rooms no single T60 exists":
        "bajo la frecuencia de Schroeder, en volúmenes acoplados y en "
        "salas tipo pasillo no existe un T60 único",
    # Panel between rooms (mass law and coincidence)
    "Panel between rooms: mass law and the coincidence dip":
        "Panel entre recintos: ley de masas y valle de coincidencia",
    "Panel under test: 12.5 mm plasterboard":
        "Panel en ensayo: yeso laminado de 12,5 mm",
    "diffuse incidence": "incidencia difusa",
    "transmitted": "transmitido",
    "bending wave at fc": "onda de flexión en fc",
    "12.5 mm": "12,5 mm",
    "m″ = 8.8 kg/m²": "m″ = 8,8 kg/m²",
    "fc = 2.6 kHz": "fc = 2,6 kHz",
    "+6 dB/octave": "+6 dB/octava",
    "predicted R(f)": "R(f) predicho",
    "Diffuse-field mass law: R rises 6 dB per octave and 6 dB per doubling of m″":
        "Ley de masas en campo difuso: R sube 6 dB por octava y 6 dB por "
        "duplicación de m″",
    "At fc = (c₀²/2π) √(m″/B′) = 2619 Hz the free bending wave matches the trace wavelength":
        "En fc = (c₀²/2π) √(m″/B′) = 2619 Hz la onda libre de flexión iguala "
        "la longitud de onda de traza",
    "Sharp's prediction rates at Rw = 27 dB; the dip collects the unfavourable deviations":
        "La predicción de Sharp puntúa Rw = 27 dB; el valle concentra las "
        "desviaciones desfavorables",
    # Porous layer on a rigid wall
    "Porous absorber on a rigid wall: microstructure to absorption":
        "Absorbente poroso sobre pared rígida: microestructura y absorción",
    "Porous layer (mineral wool)": "Capa porosa (lana mineral)",
    "plane wave, normal incidence": "onda plana, incidencia normal",
    "reflected: |R|² = 1 − α = 0.09": "reflejado: |R|² = 1 − α = 0,09",
    "microstructure (zoom)": "microestructura (ampliada)",
    "fibre frame": "esqueleto de fibras",
    "air in the pores: φ = 0.98": "aire en los poros: φ = 0,98",
    "σ = 20 kPa·s/m²  (flow resistivity)":
        "σ = 20 kPa·s/m²  (resistividad al flujo)",
    "φ = 0.98  (porosity)": "φ = 0,98  (porosidad)",
    "α∞ = 1.0  (tortuosity)": "α∞ = 1,0  (tortuosidad)",
    "Λ = Λ′ = 87 µm  (viscous / thermal lengths)":
        "Λ = Λ′ = 87 µm  (longitudes viscosa y térmica)",
    "JCA equivalent fluid: the five parameters give Zc and k; a hard-backed layer has Zs = −j Zc cot(kd)":
        "Fluido equivalente JCA: los cinco parámetros dan Zc y k; con "
        "respaldo rígido Zs = −j Zc cot(kd)",
    "α = 1 − |R|² = 0.91 at 1 kHz for this 50 mm layer":
        "α = 1 − |R|² = 0,91 a 1 kHz para esta capa de 50 mm",
    "viscous friction in the pores and heat exchange with the frame dissipate the sound energy":
        "la fricción viscosa en los poros y el intercambio de calor con el "
        "esqueleto disipan la energía sonora",
    # Barrier diffraction over ground (Fresnel number)
    "Barrier diffraction over ground: the Fresnel number at work":
        "Difracción en barrera sobre el suelo: el número de Fresnel en acción",
    "A = 50.09 m": "A = 50,09 m",
    "B = 50.06 m": "B = 50,06 m",
    "direct d = 100.00 m (blocked)": "directo d = 100,00 m (bloqueado)",
    "1.0 m": "1,0 m",
    "4.0 m": "4,0 m",
    "path difference δ = A + B − d = 0.15 m; Fresnel number N = 2δ/λ = 0.44 at 500 Hz":
        "diferencia de caminos δ = A + B − d = 0,15 m; número de Fresnel "
        "N = 2δ/λ = 0,44 a 500 Hz",
    "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10.0 dB at 500 Hz":
        "Kurze–Anderson: Δbar = 5 + 20 log10( √(2πN) / tanh √(2πN) ) = 10,0 dB "
        "a 500 Hz",
    "N grows with frequency: the same screen gives 15.5 dB at 2 kHz (vertical scale exaggerated)":
        "N crece con la frecuencia: la misma pantalla da 15,5 dB a 2 kHz "
        "(escala vertical exagerada)",
    # Image-source lattice in plan
    "Image-source lattice in plan: first reflections of a 7 × 5 m room":
        "Fuentes imagen en planta: primeras reflexiones (sala de 7 × 5 m)",
    "10.7 ms": "10,7 ms",
    "17.3 ms": "17,3 ms",
    "20.5 ms": "20,5 ms",
    "21.6 ms": "21,6 ms",
    "24.6 ms": "24,6 ms",
    "25.6 ms": "25,6 ms",
    "1st order": "1.er orden",
    "2nd order": "2.º orden",
    "the image sees": "la imagen ve",
    "a straight path": "un camino recto",
    "plan at the source plane z = 1.5 m": "planta en el plano de la fuente z = 1,5 m",
    "7.0 m": "7,0 m",
    "every reflection is the free-field arrival of a mirror image: t = r/c, √(1−α) per bounce, 1/(4πr) spreading":
        "cada reflexión llega como campo libre de su imagen: t = r/c, √(1−α) "
        "por rebote, esparcimiento 1/(4πr)",
    "in-plane images up to order 2 shown; the full lattice adds floor, ceiling and outer mirror rooms":
        "imágenes en planta hasta orden 2; la retícula completa añade suelo, "
        "techo y salas más lejanas",
    # Noise control at source, path and receiver
    "Noise control at the source, along the path and at the receiver":
        "Control de ruido en la fuente, en el camino y en el receptor",
    "1 · At the source": "1 · En la fuente",
    "2 · Along the path": "2 · En el camino",
    "3 · At the receiver": "3 · En el receptor",
    "Enclosure": "Encapsulamiento",
    "Machine": "Máquina",
    "Operator cabin": "Cabina del operario",
    "expansion chamber": "cámara de expansión",
    "lined elbow": "codo revestido",
    "open end": "extremo abierto",
    "enclosure IL = R − C": "IL del encapsulamiento = R − C",
    "25 dB at 500 Hz": "25 dB a 500 Hz",
    "silencer TL peak 6.5 dB at 286 Hz (m = 4)":
        "pico de TL del silenciador: 6,5 dB a 286 Hz (m = 4)",
    "lined elbow 6 dB at 1 kHz; open end 18 dB at 63 Hz":
        "codo revestido: 6 dB a 1 kHz; extremo abierto: 18 dB a 63 Hz",
    "cabin IL = R − C": "IL de la cabina = R − C",
    "31 dB at 1 kHz": "31 dB a 1 kHz",
    "0.30 m": "0,30 m",
    "the classic ranking: quiet the source first, treat the path next, shield the receiver last":
        "la jerarquía clásica: primero la fuente, después el camino y por "
        "último el receptor",
    "enclosure and cabin share IL = R − C, with C = 10 log10(0.3 + S_E/R_i) = 4.9 dB for a lined interior (ᾱ = 0.3)":
        "encapsulado y cabina: IL = R − C, con C = 10 log10(0,3 + S_E/R_i) = "
        "4,9 dB (interior revestido, ᾱ = 0,3)",
    "reactive silencer: TL = 10 log10[1 + ¼ (m − 1/m)² sin²(kL)], peaking where the 0.3 m chamber is a quarter wavelength":
        "silenciador reactivo: TL = 10 log10[1 + ¼ (m − 1/m)² sin²(kL)], "
        "máximo donde la cámara de 0,3 m mide λ/4",
    # Sound level meter pipeline (IEC 61672-1), one function per stage
    "The sound level meter pipeline: one function per stage":
        "La cadena del sonómetro: una función por etapa",
    "Calibrator tone": "Tono del calibrador",
    "94 dB at 1 kHz  (IEC 60942)": "94 dB a 1 kHz  (IEC 60942)",
    "Measurement recording": "Grabación de medición",
    "same microphone, same gain": "mismo micrófono, misma ganancia",
    "the factor S in pascals per digital unit":
        "el factor S en pascales por unidad digital",
    "Calibrated pressure   p(t) = S · x(t)   in pascals":
        "Presión calibrada   p(t) = S · x(t)   en pascales",
    "every level function takes S as calibration_factor=":
        "toda función de nivel acepta S como calibration_factor=",
    "Display and statistics": "Pantalla y estadística",
    "exponential detector, τF = 125 ms":
        "detector exponencial, τF = 125 ms",
    "Integrated levels": "Niveles integrados",
    "energy average, no ballistics": "promedio energético, sin balística",
    "Band spectrum": "Espectro por bandas",
    "IEC 61260-1 band edges": "bordes de banda IEC 61260-1",
    "dB re 20 µPa": "dB re 20 µPa",
    "one-third-octave band levels": "niveles por tercio de octava",
    "Class verification against the acceptance limits":
        "Verificación de clase frente a los límites de aceptación",
    "verify_weighting_class (IEC 61672-1 Table 3)  ·  verify_filter_class (IEC 61260-1 Table 1)":
        "verify_weighting_class (Tabla 3 de IEC 61672-1)  ·  "
        "verify_filter_class (Tabla 1 de IEC 61260-1)",
    # Calibration data flow (IEC 60942)
    "Calibration data flow: one factor, every level function":
        "Flujo de la calibración: un factor, todas las funciones de nivel",
    "Calibrator recording": "Grabación del calibrador",
    "1 kHz tone through the chain": "tono de 1 kHz por la misma cadena",
    "the same chain, untouched": "la misma cadena, sin tocar",
    "nothing in the chain may change between the two":
        "nada de la cadena puede cambiar entre una y otra",
    "fs enables the IEC 60942 stability check":
        "con fs comprueba la estabilidad (IEC 60942)",
    "pascals per digital unit": "pascales por unidad digital",
    "every level function accepts calibration_factor=":
        "toda función de nivel acepta calibration_factor=",
    "one factor for the whole library": "un solo factor para toda la biblioteca",
    "Levels in dB SPL": "Niveles en dB SPL",
    "No calibrator?": "¿Sin calibrador?",
    "S = 1, samples read as Pa": "S = 1, muestras como Pa",
    "use dbfs=True for dBFS": "usa dbfs=True para dBFS",
    # Filter bank data flow: decimation decision and band outputs
    "Inside a band: the decimation decision and the biquad cascade":
        "Dentro de una banda: diezmado y cascada de biquads",
    "Input signal  x(t)": "Señal de entrada  x(t)",
    "sample rate fs": "frecuencia de muestreo fs",
    "Room to decimate?": "¿Cabe diezmar?",
    "fs / 2 ≥ 1.25 · f_upper": "fs / 2 ≥ 1,25 · f_sup",
    "M = floor[(fs / 2) / (1.25 · f_upper)]":
        "M = floor[(fs / 2) / (1,25 · f_sup)]",
    "poles stay clear of z = 1": "los polos se alejan de z = 1",
    "SOS band filter at fs / M": "Filtro SOS de banda a fs / M",
    "SOS band filter at fs": "Filtro SOS de banda a fs",
    "cascaded biquads": "biquads en cascada",
    "designed on the IEC 61260-1 band edges":
        "diseñado sobre los bordes IEC 61260-1",
    "Every band filter": "Todo filtro de banda",
    "is a biquad cascade": "es una cascada de biquads",
    "not one high-order": "no un par (b, a)",
    "(b, a) pair": "de orden alto",
    "Band level": "Nivel de banda",
    "RMS or peak, in dB re 20 µPa": "RMS o pico, en dB re 20 µPa",
    "sigbands=True also returns the band signal at fs":
        "sigbands=True devuelve además la señal de banda a fs",
    "the decimated branch is interpolated back with resample_poly(M, 1)":
        "la rama diezmada se interpola de vuelta con resample_poly(M, 1)",
}
