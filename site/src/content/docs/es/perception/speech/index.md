---
title: "Habla"
description: "Las tres medidas objetivas de la inteligibilidad del habla y las preguntas que responden: el STI de IEC 60268-16, que valora un canal de transmisión, el SII de ANSI S3.5, que valora una condición de escucha, y STOI/ESTOI, que valoran un procesado a partir de un par limpio y degradado."
---

Las tres páginas de esta sección reducen la inteligibilidad del habla a un
número en [0, 1], y el arte está en saber qué número responde a tu pregunta.
El **índice de transmisión del habla** (STI) valora un *canal de
transmisión*: una sala, un sistema de megafonía, un interfono. El **índice de
inteligibilidad del habla** (SII) valora una *condición de escucha*: este
espectro de habla, en este ruido, oído por este oyente. Un aula reverberante
es un problema de STI; el ajuste de un audífono o un aviso de cabina oído
sobre el ruido de los motores es un problema de SII.

El rango [0, 1] compartido es una coincidencia de la normalización, no una
escala común, y 0,6 significa tres cosas distintas en los tres. Un **STI** de
0,6 cae en la banda D de la escala de calificación del Anexo F de
IEC 60268-16, cuyas once letras van de la U por debajo de 0,36 a la A+ a partir
de 0,76; eso es una buena sala de conferencias, y una especificación de alarma
por voz suele fijar su mínimo un par de bandas más abajo. Un **SII** de 0,6
significa que en torno al 60 % del espectro del habla ponderado por importancia
es audible para ese oyente en ese ruido; el índice es una fracción por
construcción y no lleva escala de calificación normalizada alguna. Un **STOI**
de 0,6 no tiene significado absoluto: la correspondencia entre el índice y un
porcentaje de palabras entendidas se ajusta para cada corpus de pruebas de
escucha y está deliberadamente sin implementar, de modo que el STOI solo se lee
como diferencia entre dos procesadores sobre el mismo material. Nunca sustituyas
un índice por otro en una especificación y, cuando un requisito cite una cifra,
comprueba a qué norma pertenece antes de calcular nada.

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

Los dos índices normalizados conectan de forma natural con el resto de la
biblioteca: el STI consume las respuestas al impulso de
[Acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/), y el SII consume
los umbrales de audición cuantificados en
[Umbral de audición](/phonometry/es/perception/hearing/hearing-threshold/). El
STOI y el ESTOI también tienen algo aguas arriba, pero de otra clase: toman
formas de onda, así que lo que los alimenta es lo que haya producido la
grabación limpia y la degradada, y por eso quedan al lado de las herramientas
de procesado de señal de [Señales y
espectros](/phonometry/es/signals/spectra/) y no al lado de una norma de
medida.

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

## Qué no cubre esta sección

**Aquí no se examina a ningún oyente ni se predice una puntuación.** El STOI
devuelve el índice basado en correlación y no el porcentaje de palabras
entendidas, porque la correspondencia logística se ajusta para cada corpus de
pruebas de escucha; el SII devuelve una fracción de audibilidad y no una
puntuación; y ninguna página de aquí reproduce una prueba subjetiva de
inteligibilidad. **Tampoco se adquiere ninguna señal**: la página del STI
implementa la señal directa STIPA y el cálculo indirecto desde una respuesta al
impulso, pero la medición directa completa con las 14 frecuencias de modulación
del apartado 6.3 no está implementada, así que una cadena con distorsión severa
necesita equipo de medida y no esta biblioteca.

Dentro del SII hay dos límites de cobertura que conviene comprobar antes de
usarlo: los espectros de habla de esfuerzo vocal elevado, alto y de grito solo
se incluyen para el procedimiento por tercios de octava, y las funciones de
importancia de banda tabuladas son el compromiso de habla promedio de cada
tabla, con las alternativas por material del Anexo B en tus manos a través del
argumento `band_importance=`. No se ofrece remuestreo entre los cuatro
procedimientos por bandas: cada uno se alimenta con espectros en sus propias
bandas. Y la opción de habla femenina no falta en el STI: la Edición 5 de
IEC 60268-16 la eliminó, así que no queda nada por implementar.
