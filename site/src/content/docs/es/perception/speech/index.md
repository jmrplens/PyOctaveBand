---
title: "Habla"
description: "Los dos índices normalizados de inteligibilidad del habla y las distintas preguntas que responden: el STI de IEC 60268-16, que valora un canal de transmisión, y el SII de ANSI S3.5, que valora una condición de escucha."
---

Las dos páginas de esta sección reducen la inteligibilidad del habla a un
número en [0, 1], y el arte está en saber qué número responde a tu pregunta.
El **índice de transmisión del habla** (STI) valora un *canal de
transmisión*: una sala, un sistema de megafonía, un interfono. El **índice de
inteligibilidad del habla** (SII) valora una *condición de escucha*: este
espectro de habla, en este ruido, oído por este oyente. Un aula reverberante
es un problema de STI; el ajuste de un audífono o un aviso de cabina oído
sobre el ruido de los motores es un problema de SII.

La diferencia física está en lo que modela cada índice. El STI
(**IEC 60268-16**) trabaja sobre la *envolvente* del habla: la
inteligibilidad se degrada cuando la reverberación y el ruido aplanan las
modulaciones lentas de intensidad del habla, y el índice mide cuánta de esa
modulación sobrevive al canal, mediante la función de transferencia de
modulación. Puede calcularse indirectamente desde una respuesta al impulso
medida o medirse directamente con la señal de ensayo STIPA.
[Índice de transmisión del habla (STI)](/phonometry/es/perception/speech/speech-transmission/)
cubre la física de la modulación, ambos métodos y las bandas de calificación
del Anexo F.

El SII (**ANSI S3.5-1997**) trabaja sobre la *audibilidad*: la
inteligibilidad se predice a partir de cuánta parte del espectro portador de
habla supera el umbral efectivo del oyente, banda a banda, ponderada por la
importancia de cada banda para el habla. El ruido, el autoenmascaramiento, la
extensión ascendente del enmascaramiento y el umbral de audición del propio
oyente entran explícitamente, y por eso el SII se extiende con naturalidad a
la pérdida auditiva.
[Índice de inteligibilidad del habla](/phonometry/es/perception/speech/speech-intelligibility/)
cubre los cuatro procedimientos por bandas de la norma (bandas críticas,
bandas críticas de contribución equitativa, tercio de octava y octava),
incluidos los espectros normalizados de habla desde el esfuerzo vocal normal
hasta el grito.

Un tercer par de medidas, **STOI** y **ESTOI**, responde a otra pregunta:
dada una referencia limpia *y* una versión degradada o procesada de la misma
habla, ¿cómo de inteligible es el resultado? Valoran el propio procesado, por
lo que son la vara de medir habitual de la reducción de ruido y la separación
de fuentes.

Los dos conectan de forma natural con el resto de la biblioteca: el STI
consume las respuestas al impulso de
[Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/), y el SII consume
los umbrales de audición cuantificados en
[Umbral de audición](/phonometry/es/perception/hearing/hearing-threshold/).

## Páginas de esta sección

- [Índice de transmisión del habla (STI)](/phonometry/es/perception/speech/speech-transmission/):
  la función de transferencia de modulación de IEC 60268-16, el método
  indirecto desde una respuesta al impulso y la medición directa STIPA.
- [Índice de inteligibilidad del habla](/phonometry/es/perception/speech/speech-intelligibility/):
  el método de importancia y audibilidad de banda de ANSI S3.5-1997, en ruido
  y con pérdida auditiva.
- [Inteligibilidad objetiva (STOI y ESTOI)](/phonometry/es/perception/speech/objective-intelligibility/):
  las medidas basadas en correlación para habla ruidosa con ponderación
  tiempo-frecuencia, a partir de un par limpio/degradado.
