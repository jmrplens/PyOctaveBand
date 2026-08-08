---
title: "Psicoacústica"
description: "Las métricas perceptivas del sonido: modelos de sonoridad (ISO 532, ECMA-418-2), agudeza, tonalidad, aspereza e intensidad de fluctuación, los dos métodos de evaluación tonal (prominencia ECMA-418-1 y audibilidad ISO/PAS 20065), y la molestia psicoacústica de Fastl y Zwicker."
---

La psicoacústica sustituye la pregunta "¿cuántos decibelios?" por "¿qué
percibe el oyente?". Su magnitud base es la **sonoridad**: una magnitud
perceptiva en sonios, calculada por modelos auditivos que tienen en cuenta el
filtrado, el enmascaramiento y la compresión del oído. Sobre la sonoridad se
asientan las sensaciones de **calidad sonora** que distinguen dos sonidos
igual de fuertes: la agudeza (énfasis en altas frecuencias), la tonalidad
(tonos discretos audibles), la aspereza (modulación rápida) y la intensidad
de fluctuación (modulación lenta). Y encima, una métrica combinada de **molestia** que pesa la sonoridad, la agudeza, la aspereza y la intensidad de fluctuación en un único escalar.

Toda métrica de aquí es una magnitud fijada por un **sonido de referencia** y no
por una unidad física, y conocer el ancla es lo que hace legible una cifra:
1 sonio es un tono de 1 kHz a 40 dB SPL; 1 acum, un ruido del ancho de una banda
crítica a 1 kHz y 60 dB; 1 asper, una portadora de 1 kHz totalmente modulada a
70 Hz a 60 dB; 1 vacil, esa misma portadora modulada a 4 Hz; y 1 tu_HMS, un tono
de 1 kHz a 40 dB. Están tabuladas juntas, al lado de las escalas de habla y de
audición, bajo "Cómo leer las cifras" en la
[página de la sección](/phonometry/es/perception/).

Las dos familias de páginas difieren en propósito, y esa diferencia decide qué
se puede concluir. La sonoridad, la agudeza, la aspereza y la intensidad de
fluctuación son **magnitudes abiertas** para comparar diseños: no hay nota de
aprobado, y la afirmación útil es siempre una comparación. Las dos páginas
tonales terminan en un **veredicto frente a un criterio**, porque existen para
justificar una declaración o una penalización. ECMA-418-2 queda entre ambas:
acompaña su tonalidad (0,4 tu_HMS sobre una banda), su aspereza (0,2 asper) y
su intensidad de fluctuación (0,2 vacil_HMS) con criterios de prominencia
informativos, que es lo más parecido a una nota de aprobado dentro de la familia
de las magnitudes. Todas comparten un prerrequisito: una señal calibrada de
forma absoluta en pascales, porque toda métrica de aquí depende del nivel.

[Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/) es la página fundacional: el
método de referencia Zwicker de ISO 532-1 con su ficha de una página, junto
con las curvas isofónicas de ISO 226:2023 que anclan la escala perceptiva
para tonos puros. Las familias de modelos más recientes, Moore-Glasberg según
ISO 532-2/-3 y el modelo auditivo de Sottek de ECMA-418-2, continúan en
[Sonoridad avanzada](/phonometry/es/perception/psychoacoustics/advanced-loudness/), que además
lleva la tabla de elección de modelo.
[Métricas de calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/) añade la
agudeza según DIN 45692 y la tonalidad, la aspereza y la intensidad de
fluctuación de ECMA-418-2, que comparten el frontal auditivo de Sottek.

Los tonos en ruido reciben dos páginas dedicadas porque se les hacen dos
preguntas distintas.
[Tonos discretos prominentes (ECMA-418-1)](/phonometry/es/perception/psychoacoustics/tone-prominence/)
responde a una pregunta de ruido de producto: ¿es este tono *prominente*
según los criterios de razón tono-ruido y razón de prominencia usados en las
declaraciones de equipos informáticos?
[Audibilidad objetiva de tonos en ruido (ISO/PAS 20065)](/phonometry/es/perception/psychoacoustics/tone-audibility/)
responde a una ambiental: ¿en cuántos decibelios supera el tono su umbral de
enmascaramiento?, la audibilidad que alimenta la penalización tonal de
ISO 1996-2.

[Molestia psicoacústica e intensidad de fluctuación](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/)
cierra la cadena con el modelo de Fastl y Zwicker, que combina sonoridad,
agudeza, aspereza y la sensación de modulación lenta de la intensidad de
fluctuación en un único valor de molestia. Léela en último lugar: tres de sus
cuatro entradas salen de las páginas anteriores, y la cuarta, la intensidad de
fluctuación, la aporta ella misma, tanto en la forma cerrada de Fastl y Zwicker
como en el modelo de señal de Osses 2016. La intensidad de fluctuación de
ECMA-418-2 de la página de calidad sonora es un modelo normativo más de esa
misma sensación, bajo otro nombre de unidad.

## Páginas de esta sección

- [Sonoridad](/phonometry/es/perception/psychoacoustics/loudness/): la sonoridad Zwicker de
  ISO 532-1 en sonios, más las curvas isofónicas de ISO 226:2023.
- [Sonoridad avanzada (ISO 532-2/-3, ECMA-418-2)](/phonometry/es/perception/psychoacoustics/advanced-loudness/):
  los modelos de sonoridad Moore-Glasberg y Sottek y la tabla de elección de
  modelo.
- [Métricas de calidad sonora](/phonometry/es/perception/psychoacoustics/sound-quality/): agudeza
  (DIN 45692) y tonalidad, aspereza e intensidad de fluctuación de
  ECMA-418-2.
- [Tonos discretos prominentes (ECMA-418-1)](/phonometry/es/perception/psychoacoustics/tone-prominence/):
  razones tono-ruido y de prominencia con veredictos de prominencia.
- [Audibilidad objetiva de tonos en ruido (ISO/PAS 20065)](/phonometry/es/perception/psychoacoustics/tone-audibility/):
  la audibilidad de un tono sobre el umbral de enmascaramiento, que alimenta
  el ajuste tonal de ISO 1996-2.
- [Molestia psicoacústica e intensidad de fluctuación](/phonometry/es/perception/psychoacoustics/psychoacoustic-annoyance/):
  el modelo de molestia de Fastl y Zwicker y los modelos de intensidad de
  fluctuación que consume.

## Qué no cubre esta sección

**Aquí todo es monoaural.** Las combinaciones binaurales que ECMA-418-2 define
para la sonoridad, la aspereza y la intensidad de fluctuación no están
implementadas, así que una grabación estéreo o binaural se analiza canal a
canal, y ningún modelo de aquí tiene en cuenta la localización ni la liberación
espacial del enmascaramiento. Quedan fuera además dos refinamientos opcionales:
la ponderación de entropía del apartado 7.1.6, que necesita una señal externa
de velocidad de giro, y el pequeño ajuste que permite la nota al pie 47.

**Un veredicto nunca está completo.** El indicador `prominent` que devuelven las
funciones de prominencia tonal es solo el criterio numérico; ECMA-418-1 exige
además la confirmación auditiva y el filtro del umbral inferior de audición, y
ambos quedan a cargo de quien llama. El módulo de audibilidad de tonos es
agnóstico a la ponderación y **no** aplica la ponderación A que exige el
apartado 5.3.2, así que pondera A el espectro antes de pasárselo; y toma un
espectro de banda estrecha ya calculado en lugar de construirlo desde una
grabación.

Conviene conocer dos desviaciones documentadas. ISO 532-3 prescribe remuestrear
a 32 kHz antes del análisis FFT deslizante; esta implementación trabaja a la
frecuencia de muestreo nativa, una desviación que se mantiene dentro de la
incertidumbre expandida de la norma pero que deberías deshacer remuestreando
primero si te importa la conformidad estricta apartado a apartado. Y el modelo
de señal de intensidad de fluctuación de Osses 2016 está validado solo para
estímulos modulados en amplitud, con un suelo documentado: un tono estacionario
de 1 kHz marca unos 0,09 vacil en lugar de 0.

Por último, ninguna de estas métricas es una respuesta comunitaria: la molestia
de aquí es una sensación de laboratorio calculada a partir de una señal,
mientras que la molestia que declara un vecindario es una magnitud de encuesta
social, que se trata con los indicadores de [Medio ambiente y
transporte](/phonometry/es/environment/).
