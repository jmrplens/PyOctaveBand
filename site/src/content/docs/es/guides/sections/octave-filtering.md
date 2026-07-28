---
title: "Filtrado en octavas"
description: "El análisis en bandas de octava fraccional en phonometry: los bancos de filtros ANSI S1.11 / IEC 61260-1 y sus arquitecturas, la verificación de clase de IEC 61260-1, el procesado por bloques con estado para señales en streaming, y el análisis multicanal vectorizado."
---

El análisis acústico rara vez quiere una FFT en bruto: las normas, los
índices y el propio oído trabajan en **bandas de octava fraccional**,
intervalos de frecuencia cuya anchura crece proporcionalmente con la
frecuencia. phonometry las implementa como bancos de filtros recursivos cuyos
**puntos de -3 dB caen exactamente en los bordes de banda de ANSI S1.11**, de
modo que los niveles de banda son comparables sea cual sea la arquitectura de
filtro que los calcule, y cuyos diseños se verifican contra las tolerancias
de clase de **IEC 61260-1:2014**.

La página fundacional es
[Bancos de filtros](/phonometry/es/guides/filter-banks/). Cubre las cinco
arquitecturas (Butterworth, Chebyshev I/II, elíptica y Bessel), qué
intercambian sus respuestas en frecuencia entre sí, cómo se descompone una
señal en bandas de 1/1, 1/3 o 1/b arbitrario, y el modo de fase cero fuera de
línea para análisis donde el retardo del filtro no debe emborronar el
resultado. Por dentro, cada banco es una cascada de secciones de segundo
orden con diezmado multitasa, que es lo que mantiene numéricamente estables
las bandas de baja frecuencia.

Demostrar que un banco diseñado cumple esas tolerancias es el trabajo de
[Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/guides/filter-compliance/):
la máscara de aceptación de la Tabla 1 banda a banda, la clase 0 más estricta
de la edición retirada de 1995, qué aporta una clase de prestaciones en una
medida y la ficha de conformidad acreditada de una página.

Las otras dos páginas escalan esa base por dos ejes independientes.
[Procesado por bloques](/phonometry/es/guides/block-processing/) la escala en
*tiempo*: las señales que no caben en memoria (grabaciones de horas,
monitorización en vivo, registradores embebidos) se procesan búfer a búfer
arrastrando el estado de los filtros, de modo que el resultado es idéntico
bit a bit a procesar la señal entera de una vez.
[Multicanal y rendimiento](/phonometry/es/guides/multichannel/) la escala en
*canales*: los arrays de micrófonos y las grabaciones multicanal se analizan
vectorizados, una llamada para todos los canales, con notas sobre dónde se va
realmente el tiempo de cálculo.

Léelas en ese orden. Todo lo que viene después (niveles, sonoridad,
parámetros de sala) consume las señales o los niveles de banda que estas
páginas producen.

## Páginas de esta sección

- [Bancos de filtros](/phonometry/es/guides/filter-banks/): las cinco
  arquitecturas de filtro, respuestas en frecuencia, descomposición en bandas
  y filtrado de fase cero.
- [Verificación de clase de filtros (IEC 61260-1)](/phonometry/es/guides/filter-compliance/):
  la máscara de aceptación de la Tabla 1, la clase 0 y la ficha de
  conformidad.
- [Procesado por bloques](/phonometry/es/guides/block-processing/): flujos en
  streaming con estado de filtro arrastrado.
- [Multicanal y rendimiento](/phonometry/es/guides/multichannel/): análisis
  multicanal vectorizado y notas de rendimiento.
