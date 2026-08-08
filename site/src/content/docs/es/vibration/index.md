---
title: "Vibración y ruido estructural"
description: "La vibración como fuente de ruido en edificios, como exposición humana y como diagnóstico de la propia máquina: el vocabulario de FRF (ISO 7626), la transmisión de onda de flexión en uniones, la rigidez de transferencia de aisladores (ISO 10846), la potencia acústica desde vibración (ISO/TS 7849), la cadena de equipos EN 15657 y EN 12354-5, las métricas de vibración en humanos de ISO 2631 e ISO 5349, y las frecuencias cinemáticas de fallo de la maquinaria rotativa."
---

La vibración importa a la acústica tres veces. Primero como **fuente de
sonido**: una bomba o un ventilador atornillados a un edificio inyectan
potencia estructural que viaja por muros y forjados y se vuelve a radiar como
ruido audible varias salas más allá. Segundo como **exposición humana** por
derecho propio: la vibración transmitida a una persona de pie, sentada o que
empuña una herramienta se mide, pondera y limita de forma muy parecida al
ruido, con sus propias métricas y valores legales de acción. Tercero como
**diagnóstico de la propia máquina**: el mismo espectro que alimenta las dos
primeras preguntas nombra además el rodamiento, el engranaje o el álabe que lo
produjo, porque toda periodicidad que hay en él pertenece a algo que gira,
engrana o pasa, a una frecuencia que fija la geometría.

Las páginas de **fuentes de ruido estructural** siguen la cadena de la fuente
en orden. La familia de funciones de respuesta en frecuencia de ISO 7626
(receptancia, movilidad, acelerancia) es el vocabulario; los coeficientes de
transmisión del enfoque ondulatorio de una unión de placas describen la
estructura por la que corre después esa potencia; la rigidez de transferencia
de ISO 10846 caracteriza los elementos resilientes que interrumpen ese camino;
ISO/TS 7849 estima la potencia aérea que una
superficie vibrante radia directamente; EN 15657 mide la potencia estructural
que una máquina inyecta en una placa de recepción; y EN 12354-5 lo ensambla
todo en el nivel de presión acústica predicho en un recinto receptor. Esa
predicción final es también donde esta sección entrega el testigo a los
modelos de [aislamiento acústico](/phonometry/es/buildings/insulation/)
de la sección de edificación.

Las páginas de **vibración en humanos** comparten la filosofía de medida de
un sonómetro, aplicada a la aceleración: ponderaciones frecuenciales que
reflejan la respuesta del cuerpo, promedios móviles e integrados, y
magnitudes de dosis comparadas con los valores de acción y límite de la
Directiva 2002/44/CE, más el modelo dedicado de respuesta espinal para
vibración con choques repetidos.

Empieza por
[Movilidad mecánica y la familia de FRF](/phonometry/es/vibration/structural/mechanical-mobility/)
si te interesa el ruido que una máquina causa en un edificio, por
[Vibración en humanos](/phonometry/es/vibration/human/human-vibration/) si te
interesa la dosis que recibe una persona, o por [Frecuencias de fallo en
máquinas](/phonometry/es/vibration/machinery/machine-diagnostics/) si te
interesa el estado de la propia máquina.

## [Fuentes de ruido estructural](/phonometry/es/vibration/structural/)

Del vocabulario de FRF al nivel predicho en el recinto receptor.

- [Movilidad mecánica y la familia de FRF (ISO 7626-1)](/phonometry/es/vibration/structural/mechanical-mobility/):
  receptancia, movilidad y acelerancia con sus recíprocas, y el resonador de
  referencia de un grado de libertad.
- [Transmisión de onda de flexión en uniones de placas (Cremer/Craik/Hopkins)](/phonometry/es/vibration/structural/junction-transmission/):
  los coeficientes del enfoque ondulatorio independientes de la frecuencia para
  uniones rígidas en X, T, L y en línea, su media angular y el factor de
  pérdidas por acoplamiento y el Kij derivados.
- [Rigidez dinámica de transferencia (ISO 10846)](/phonometry/es/vibration/structural/transfer-stiffness/):
  la rigidez de transferencia dinámica de aisladores de vibración por los
  métodos directo e indirecto.
- [Potencia acústica desde vibración (ISO/TS 7849)](/phonometry/es/devices/emission/vibration-sound-power/):
  potencia aérea radiada a partir de la velocidad superficial y un factor de
  radiación.
- [Potencia acústica estructural de equipos (EN 15657)](/phonometry/es/buildings/design/structure-borne-power/):
  el método de la placa de recepción y las magnitudes de fuente
  independientes de la placa.
- [Ruido estructural instalado (EN 12354-5)](/phonometry/es/buildings/design/installed-structure-borne/):
  el nivel de presión acústica en el recinto receptor predicho desde las
  movilidades de fuente y receptor.

## [Vibración en humanos](/phonometry/es/vibration/human/)

La vibración transmitida al cuerpo humano, de la exposición diaria al riesgo
de lesión lumbar.

- [Vibración en humanos](/phonometry/es/vibration/human/human-vibration/):
  ponderaciones de cuerpo completo y mano-brazo (ISO 8041-1), medidas r.m.s.
  y de dosis (ISO 2631-1), exposición diaria A(8) (ISO 5349) y los valores de
  la Directiva 2002/44/CE.
- [Vibración con choques múltiples (ISO 2631-5)](/phonometry/es/vibration/human/multiple-shock-vibration/):
  el modelo de respuesta espinal y la probabilidad de lesión lumbar para
  vibración con múltiples choques.

## [Maquinaria](/phonometry/es/vibration/machinery/)

Convertir un espectro de vibración en un diagnóstico de la máquina que lo produjo.

- [Frecuencias de fallo en máquinas](/phonometry/es/vibration/machinery/machine-diagnostics/):
  las frecuencias características de rodamientos, engranajes y ejes, y el
  análisis de envolvente que las encuentra bajo el ruido de banda ancha de una
  máquina en marcha.

## Qué no cubre esta sección

**Aquí no se somete a ensayo de tipo ningún instrumento.** El objeto propio de
la ISO 8041-1, el diseño y los ensayos de tipo de los vibrómetros para vibración
en humanos, no está implementado: de ella solo se toman las ponderaciones
frecuenciales, así que un veredicto de clase para un vibrómetro de mano no es
algo que esta biblioteca pueda dar.

**Tampoco se emite ningún veredicto de severidad para una máquina.** Las páginas
de maquinaria predicen *dónde* estaría una línea, nunca si está presente ni si
la máquina tiene un problema: los criterios de amplitud que convierten una línea
presente en una valoración — las tendencias del factor de cresta y de la
curtosis, y las bandas de severidad en velocidad de las ISO 10816 / ISO 20816 —
quedan fuera de la biblioteca, igual que el equilibrado de rotores (ISO 21940) y
el seguimiento de órdenes.

**Ni tampoco para un edificio.** La edición de 2003 de la ISO 2631-2 suprimió a
propósito los valores orientativos de su predecesora, así que no hay magnitudes
admisibles de vibración en edificios con las que comparar; lo que da la
biblioteca es la magnitud ponderada, y el juicio se queda con quien evalúa y con
el código nacional.

Dos predicciones estructurales son idealizaciones y no mediciones. Los
coeficientes de transmisión de unión son un resultado en forma cerrada para una
unión rígida y simplemente apoyada — el índice de reducción vibracional *medido*
de la ISO 10848 es [Transmisión por flancos en
laboratorio](/phonometry/es/buildings/insulation/flanking-lab/) — y la página de
FRF devuelve recíprocas libres elemento a elemento, correctas para el punto de
excitación o para una sola vía pero no para una matriz de FRF completa, sin
procesado de excitación por martillo de impacto (ISO 7626-5) y sin magnitudes
matriciales bloqueadas. En la página de aisladores, las Partes 4 y 5 de la
ISO 10846 no están implementadas, y dos de las comprobaciones de validez de la
norma (la desigualdad de masa de bloqueo y el criterio de linealidad del
apartado 7.6) se describen pero no se calculan por ti.

## Antes y después de estas páginas

Toda magnitud de aquí parte de un registro de aceleración y de una estimación
espectral, así que el filtrado, las curvas de ponderación y los estimadores
espectrales que hay detrás están en [Análisis de
señal](/phonometry/es/signals/), y [Análisis
espectral](/phonometry/es/signals/spectra/spectral-analysis/) es la página sobre
la que se construye el diagnóstico de maquinaria. Las derivaciones están en
[Teoría de la vibración](/phonometry/es/reference/theory/vibration/): las
ponderaciones de vibración en humanos, el modelo de choques de la ISO 2631-5 y
las movilidades puntuales y la eficiencia de radiación.

Si has llegado aquí desde una búsqueda y quieres la forma de la biblioteca
entera, [¿Qué necesitas medir?](/phonometry/es/start/tasks/) la indexa por el
trabajo y [Todas las guías](/phonometry/es/start/guides/) lista todas las
páginas con una línea sobre cada una.
