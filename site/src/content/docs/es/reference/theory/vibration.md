---
title: "Vibración"
description: "Ponderaciones de vibración en humanos, métricas de cuerpo entero y mano-brazo y el modelo espinal de choques múltiples detrás de phonometry."
references:
  - type: book
    authors: ["Griffin, M. J."]
    year: 1996
    title: "Handbook of human vibration"
    publisher: "Academic Press"
    url: "https://shop.elsevier.com/books/handbook-of-human-vibration/griffin/978-0-12-303041-2"
    note: "ISBN 978-0-12-303041-2. La evidencia biodinámica y de efectos sobre la salud que sustenta las ponderaciones de ISO 8041-1, las medidas de dosis rms/MTVV/VDV y el razonamiento de lesión espinal del modelo de choques múltiples."
  - type: book
    authors: ["Mansfield, N. J."]
    year: 2004
    title: "Human response to vibration"
    publisher: "CRC Press"
    url: "https://www.routledge.com/Human-Response-to-Vibration/Mansfield/p/book/9780415282390"
    note: "ISBN 978-0-415-28239-0. Un recorrido moderno y compacto por las cadenas de evaluación de cuerpo completo de ISO 2631-1 y mano-brazo de ISO 5349 resumidas en esta página."
  - type: book
    authors: ["Cremer, L.", "Heckl, M.", "Petersson, B. A. T."]
    year: 2005
    title: "Structure-borne sound: Structural vibrations and sound radiation at audio frequencies"
    edition: "3.ª ed."
    publisher: "Springer"
    url: "https://link.springer.com/book/10.1007/b137728"
    note: "ISBN 978-3-540-22696-3. La potencia en el punto de excitación W = (1/2) |F|^2 Re{Y} (Ec. 5.23) y las movilidades e impedancias en forma cerrada de estructuras infinitas (Tabla 5.1) de la sección de movilidades puntuales."
  - type: book
    authors: ["Hopkins, C."]
    year: 2007
    title: "Sound insulation"
    publisher: "Butterworth-Heinemann"
    url: "https://www.routledge.com/Sound-Insulation/Hopkins/p/book/9780750665261"
    note: "ISBN 978-0-7506-6526-1. La teoría de la eficiencia de radiación de placas en flexión (Ecs. 2.227-2.230, el límite de alta frecuencia de Leppington/Maidanik) que sustenta la sección de eficiencia de radiación."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2011
    title: "Mechanical vibration and shock — Experimental determination of mechanical mobility — Part 1: Basic terms and definitions, and transducer specifications"
    designation: "ISO 7626-1:2011"
    url: "https://www.iso.org/standard/50426.html"
    note: "Las movilidades medidas en el punto de excitación de las que los resultados en forma cerrada de estructuras infinitas de la sección de movilidades puntuales son las contrapartes teóricas."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2009
    title: "Acoustics — Determination of airborne sound power levels emitted by machinery using vibration measurement — Part 1: Survey method using a fixed radiation factor"
    designation: "ISO/TS 7849-1:2009"
    url: "https://www.iso.org/standard/40537.html"
    note: "El factor de radiación que coincide con la eficiencia de radiación, cerrando la cadena de potencia acústica a partir de la vibración de la sección de eficiencia de radiación (método de sondeo, factor de radiación fijo)."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2009
    title: "Acoustics — Determination of airborne sound power levels emitted by machinery using vibration measurement — Part 2: Engineering method including determination of the adequate radiation factor"
    designation: "ISO/TS 7849-2:2009"
    url: "https://www.iso.org/standard/40538.html"
    note: "El método de ingeniería de la misma cadena de potencia acústica a partir de la vibración, que determina el factor de radiación adecuado a partir de la medida."
---

Esta página reúne la teoría de la vibración en humanos: las ponderaciones frecuenciales de ISO 8041-1, las métricas de cuerpo entero y mano-brazo de ISO 2631-1 e ISO 5349, los valores de acción y límite de la Directiva 2002/44/CE, y el modelo espinal de choques múltiples de ISO 2631-5. Forma parte de la [referencia de teoría](/phonometry/es/reference/theory/).

## Vibración en humanos (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directiva 2002/44/CE)

La respuesta humana a la vibración depende de la frecuencia, el eje y la parte
del cuerpo, así que la aceleración se filtra con las ponderaciones
frecuenciales de ISO 8041-1:2017 antes de cualquier métrica. Cada ponderación
es la cascada analógica $H(s) = H_h(s) H_l(s) H_t(s) H_s(s)$ (Fórmula 5):
etapas Butterworth de dos polos de limitación de banda paso-alto y paso-bajo
(Fórmulas 1/2), una transición aceleración–velocidad (Fórmula 3, que porta la
única ganancia no unitaria, $K = 1{,}024$ para Wb) y un escalón ascendente
(Fórmula 4), con las frecuencias de esquina y los factores Q de la Tabla 3;
una esquina en infinito colapsa su etapa a la unidad (NOTAs de la Tabla 3). Wk
(cuerpo completo vertical) y Wd (horizontal) de ISO 2631-1, Wm (edificios,
ISO 2631-2), Wb (ferrocarril, ISO 2631-4), Wc/We/Wj (respaldo, rotacional,
cabeza) y Wh (mano-brazo, ISO 5349-1) más Wf (mareo) están implementadas desde
la cascada exacta (el filtro se aplica como la respuesta compleja exacta vía
FFT, magnitud *y* fase, no como una aproximación digital deformada por la
bilineal) y las tablas de objetivos de diseño del Anexo B de ISO 8041-1
(B.1–B.9) se reproducen al 0,1 %.

Las métricas ponderadas siguen ISO 2631-1:1997: rms móvil con integración
lineal o exponencial (Ecs. 2/3), el **MTVV** como su máximo (Ec. 4), el
**VDV** de cuarta potencia $= (\int a_w^4\, dt)^{1/4}$ en m/s^1,75 (Ec. 5), el
factor de cresta con el método básico considerado adecuado hasta 9
(cláusula 6.2) y el valor total de vibración
$a_v = \sqrt{\sum_j k_j^2 a_{wj}^2}$ (Ec. 10). La exposición mano-brazo sigue
ISO 5349-1:2001: $a_{hv}$ (Ec. 1, todos los $k = 1$), exposición diaria
$A(8) = a_{hv} \sqrt{T/T_0}$ con $T_0 = 8$ h (Ec. 2), exposiciones parciales
combinadas en cuadratura (ISO 5349-2:2001, Ecs. 1–3), y el modelo de riesgo
vascular del Anexo C, $D_y = 31{,}8\ A(8)^{-1{,}06}$, para los años hasta una
prevalencia del 10 % de dedo blanco. Los valores de acción y límite de la
Directiva 2002/44/CE están incorporados: mano-brazo $A(8)$ 2,5/5,0 m/s²,
cuerpo completo $A(8)$ 0,5/1,15 m/s² o VDV 9,1/21,0 m/s^1,75 (Artículo 3).
Los ejemplos resueltos de ISO 5349-2 se reproducen (E.2.1: 7,4 m/s² durante
2,5 h → $A(8) = 4{,}1$ m/s²; E.3 forestal, tres herramientas → 3,6 m/s²),
igual que las filas de duración de exposición de la Tabla C.1 de ISO 5349-1.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_es.svg" alt="La ponderación vertical de cuerpo completo Wk en decibelios de 0,4 a 100 Hz: una meseta cerca de -6 dB por debajo de 2 Hz, un pequeño pico de +0,5 dB cerca de 6 Hz y una caída hasta cerca de -21 dB a 100 Hz" style="width:88%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/vibration_weighting_es_dark.svg" alt="La ponderación vertical de cuerpo completo Wk en decibelios de 0,4 a 100 Hz: una meseta cerca de -6 dB por debajo de 2 Hz, un pequeño pico de +0,5 dB cerca de 6 Hz y una caída hasta cerca de -21 dB a 100 Hz" style="width:88%" loading="lazy">

*La ponderación de cuerpo completo Wk obtenida de la cascada de ISO 8041-1.*

### Choques múltiples (ISO 2631-5)

Los choques repetidos dañan la columna lumbar por la compresión de pico y no
por la energía media, así que ISO 2631-5:2018 sustituye la ponderación Wk por
la función de transferencia asiento-columna de la cláusula 5.2 (Fórmula 1: un
cero complejo y seis pares de polos complejos, unidad en continua, resonancia
cerca de 5 Hz, $|H| \approx 1{,}54$ a 5 Hz) y acumula los picos positivos de la
respuesta espinal con una dosis de sexta potencia (Palmgren-Miner)
(cláusula 5.3, Fórmulas 3/4):

$$
D_z = 1{,}07 \left( \sum_i A_{z,i}^6 \right)^{1/6}, \qquad
D_{zd} = D_z\ (t_d / t_m)^{1/6}.
$$

El Anexo C convierte la dosis diaria en una tensión compresiva
$S_d = m_z D_{zd}$ ($m_z = 0{,}029/0{,}025$ MPa por m/s² para el hombre de
82 kg / la mujer de 64 kg), sigue la resistencia última decreciente con la
edad $S_u = 6{,}75 - S_{age}(b + i)$ y forma la variable de tensión acumulada
$R$ (Fórmulas C.3/C.4), llevada a una probabilidad de lesión por la ley de
Weibull de la Tabla C.1, $\Pi = 1 - e^{-(R/\alpha)^\beta}$. El filtro espinal
se evalúa analíticamente en el dominio de la frecuencia y se valida frente a
la tabulación del filtro digital a 256 Hz del Anexo D dentro de la tolerancia
de la cláusula 5.2; el ejemplo resuelto del Anexo C (cinco choques de 40 m/s²
al día durante 20 años) se reproduce: $D_{zd} = 55{,}97$ m/s², $R = 1{,}22$,
$\Pi = 0{,}37$. El modelo espinal de elementos finitos del Anexo A
(distribuido por ISO como software aparte) queda fuera de alcance.

Consulta la [guía de vibración en humanos](/phonometry/es/guides/human-vibration/) y la
[guía de vibración con choques múltiples](/phonometry/es/guides/multiple-shock-vibration/) para su uso.

## Movilidades puntuales y eficiencia de radiación (Cremer 5, Hopkins 2.9)

La potencia vibratoria que una fuerza puntual inyecta en una estructura es
$W = \tfrac12 |F|^2\,\mathrm{Re}\{Y\}$ (Cremer Ec. 5.23), así que la
**movilidad** puntual $Y$ (la inversa de la impedancia) gobierna cuánta energía
absorbe la estructura. Para estructuras infinitas son formas cerradas (Cremer
Tabla 5.1): una placa delgada infinita es una resistencia pura
$Z = 8\sqrt{B'\,m''}$ (real, independiente de la frecuencia, con $B'$ la rigidez
a flexión por unidad de anchura y $m''$ la masa por unidad de superficie), una
viga infinita tiene $Y = (1-\mathrm{j})/(4 m' c_B)$ (fase de 45 grados,
decayendo como $\omega^{-1/2}$ a través de la velocidad de flexión $c_B$), y una
barra longitudinal tiene $Z = \rho c_L S$. Aportan la movilidad de receptor que
EN 12354-5 necesita cuando no hay medida, y son las contrapartes teóricas de las
movilidades medidas de ISO 7626. La eficiencia con que una placa en flexión
radia la potencia aérea es su **eficiencia de radiación** $\sigma$: por debajo de
la frecuencia crítica radia poco (modos de borde y esquina), y por encima
$\sigma \to (1 - f_c/f)^{-1/2} \to 1$ (Leppington/Maidanik, Hopkins
Ec. 2.227-2.230). Como $\sigma$ es exactamente el factor de radiación
$\varepsilon$ de ISO 7849, predecirla cierra la cadena de potencia acústica a
partir de la vibración sin una medida de potencia, y alimenta el camino de
transmisión resonante de la
[teoría de aislamiento de paneles](/phonometry/es/reference/theory/rooms-buildings/).

Consulta la guía [Predicción del aislamiento de paneles](/phonometry/es/guides/panel-sound-insulation/)
para su uso.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility_es.svg" alt="Magnitudes normalizadas de receptancia, movilidad y acelerancia de un resonador de un grado de libertad en un eje de frecuencia log-log, todas con máximo en la resonancia" style="width:82%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/mechanical_mobility_es_dark.svg" alt="Magnitudes normalizadas de receptancia, movilidad y acelerancia de un resonador de un grado de libertad en un eje de frecuencia log-log, todas con máximo en la resonancia" style="width:82%" loading="lazy">

*Receptancia, movilidad y acelerancia de un resonador de un grado de libertad: la misma resonancia vista a través de las tres magnitudes cinemáticas.*
