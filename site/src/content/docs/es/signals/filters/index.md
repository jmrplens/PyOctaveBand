---
title: "Filtrado en octavas"
description: "El análisis en bandas de octava fraccional en phonometry: los bancos de filtros ANSI S1.11 / IEC 61260-1 y sus arquitecturas, la verificación de clase de IEC 61260-1, el procesado por bloques con estado para señales en streaming, y el análisis multicanal vectorizado."
---

El análisis acústico rara vez quiere una FFT en bruto: las normas, los
índices y el propio oído trabajan en **bandas de octava fraccional**,
intervalos de frecuencia cuya anchura crece proporcionalmente con la
frecuencia. phonometry las implementa como bancos de filtros recursivos cuyos
diseños se verifican contra las tolerancias de clase de
**IEC 61260-1:2014**. El banco Butterworth por defecto, y las alternativas
Chebyshev II y Bessel, sitúan sus **puntos de -3 dB exactamente en los bordes
de banda de ANSI S1.11**, así que sus niveles de banda son directamente
comparables; las dos arquitecturas de rizado constante (Chebyshev I,
elíptica) colocan ahí su borde de rizado y en consecuencia leen unas décimas
de decibelio altas en todas las bandas, y por eso una campaña debe fijar una
arquitectura y mantenerla.

La página fundacional es
[Bancos de filtros](/phonometry/es/signals/filters/filter-banks/). Cubre la matemática
de las bandas, cómo se descompone una señal en bandas de 1/1, 1/3 o 1/b
arbitrario, el ecualizador paramétrico y el modo de fase cero fuera de línea
para análisis donde el retardo del filtro no debe emborronar el resultado.
Por dentro, cada banco es una cascada de secciones de segundo orden con
diezmado multitasa, que es lo que mantiene numéricamente estables las bandas
de baja frecuencia. Elegir entre las cinco arquitecturas (Butterworth,
Chebyshev I/II, elíptica y Bessel) es el trabajo de la
[Galería de arquitecturas de filtro](/phonometry/es/signals/filters/filter-gallery/):
qué intercambian sus respuestas en frecuencia entre sí, la galería completa
de respuestas y el uso de cada arquitectura, más el crossover Linkwitz-Riley.

Demostrar que un banco diseñado cumple esas tolerancias es el trabajo de
[Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/signals/filters/filter-compliance/):
la máscara de aceptación de la Tabla 1 banda a banda, la clase 0 más estricta
de la edición retirada de 1995, qué aporta una clase de prestaciones en una
medida y la ficha de conformidad acreditada de una página.

Las otras dos páginas escalan esa base por dos ejes independientes.
[Procesado por bloques](/phonometry/es/signals/filters/block-processing/) la escala en
*tiempo*: las señales que no caben en memoria (grabaciones de horas,
monitorización en vivo, registradores embebidos) se procesan búfer a búfer
arrastrando el estado de los filtros, de modo que el resultado es idéntico
bit a bit a procesar la señal entera de una vez.
[Multicanal y rendimiento](/phonometry/es/signals/filters/multichannel/) la escala en
*canales*: los arrays de micrófonos y las grabaciones multicanal se analizan
vectorizados, una llamada para todos los canales, con notas sobre dónde se va
realmente el tiempo de cálculo.

Léelas en ese orden. Todo lo que viene después (niveles, sonoridad,
parámetros de sala) consume las señales o los niveles de banda que estas
páginas producen.

## Páginas de esta sección

- [Bancos de filtros](/phonometry/es/signals/filters/filter-banks/): la matemática de
  las bandas, los parámetros del banco, el ecualizador paramétrico, la
  descomposición en bandas y el filtrado de fase cero.
- [Galería de arquitecturas de filtro](/phonometry/es/signals/filters/filter-gallery/):
  las cinco arquitecturas comparadas, la galería de respuestas y el uso de
  cada arquitectura.
- [Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/signals/filters/filter-compliance/):
  la máscara de aceptación de la Tabla 1, la clase 0 y la ficha de
  conformidad.
- [Procesado por bloques](/phonometry/es/signals/filters/block-processing/): flujos en
  streaming con estado de filtro arrastrado.
- [Multicanal y rendimiento](/phonometry/es/signals/filters/multichannel/): análisis
  multicanal vectorizado y notas de rendimiento.

## Qué no cubre esta sección

`verify_filter_class` comprueba una respuesta digital diseñada frente a la
Tabla 1 de IEC 61260-1. Los ensayos de conformidad de la norma sobre el filtro
físico (recuperación de sobrecarga, linealidad, las magnitudes de influencia
ambientales) se aplican a un instrumento y no están implementados, así que un
veredicto de clase de aquí es una afirmación sobre el diseño y no sobre un
aparato. Cerca de Nyquist la transformada bilineal deforma el eje de
frecuencia y el banco no lleva ninguna corrección para ello, a diferencia de
la opción `high_accuracy` de los filtros de ponderación: la banda atenuada más
allá del Nyquist de procesado se informa como `range_limited` en lugar de
verificarse, así que mantén el borde de banda superior holgadamente por debajo
de Nyquist o sube `fs`. Dos operaciones no admiten streaming: el filtrado de
fase cero hacia delante y hacia atrás necesita la señal entera, y los
estadísticos de rango como L90 hay que calcularlos una sola vez sobre la
envolvente conjunta. Y el camino por canal nunca mezcla canales: el retardo
entre dos micrófonos, o cuánto de un canal explica un segundo, son
[Correlación, retardo y envolvente](/phonometry/es/signals/spectra/correlation-delay/) y
[Coherencia múltiple y parcial](/phonometry/es/signals/spectra/miso-coherence/).
