---
title: "Acústica submarina"
description: "El sonido en el mar: los niveles de referencia de ISO 18405, el ruido radiado por buques (ISO 17208) y la exposición por hincado de pilotes (ISO 18406), la propagación desde la pérdida de transmisión en forma cerrada y la ecuación del sonar hasta el ruido ambiente y los solvers numéricos, y los criterios auditivos de mamíferos marinos que convierten un nivel en una evaluación."
---

La acústica submarina funciona con la misma física que la acústica aérea pero
a otra escala y con otra referencia: los niveles se expresan re **1 µPa** (no
20 µPa), la exposición re 1 µPa²·s, y el propio medio, con su velocidad del
sonido dependiente de la profundidad, refracta el sonido en canales que lo
transportan durante kilómetros. Esta sección cubre la disciplina siguiendo la
cadena fuente-camino-receptor del resto de la biblioteca.

La mitad de **fuente**, en
[Acústica submarina: ruido radiado e hincado de pilotes](/phonometry/es/underwater/underwater-acoustics/),
establece la terminología de ISO 18405 (niveles SPL, SEL y de pico y sus
referencias) y la aplica a dos casos de medición regulados: los buques, con
el nivel de ruido radiado de ISO 17208-1 y el nivel de fuente monopolar
equivalente de ISO 17208-2 mediante la corrección de superficie del espejo
de Lloyd, y el hincado percusivo de pilotes, con la exposición sonora de un
golpe, de pico y acumulada de ISO 18406.

La mitad de **camino** abarca ahora dos páginas. La primera,
[Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/),
predice lo que el mar hace con ese sonido en forma cerrada: divergencia
geométrica más absorción volumétrica (Francois-Garrison, Ainslie-McColm o
Thorp), los cuatro regímenes de aguas someras de Weston con sus distancias de
transición, la velocidad del sonido en agua de mar por cuatro formulaciones,
las ecuaciones de sonar pasivo y activo con la inversión del alcance de
detección, la pérdida por reflexión en el lecho marino de Rayleigh, y el
espectro de ruido ambiente de Wenz con el tráfico marítimo JOMOPANS-ECHO. Cuando la refracción y los contornos deciden la
respuesta,
[Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/)
calcula el campo en su lugar: la expansión en modos normales, el trazado de
rayos y la ecuación parabólica split-step de Fourier, con la guía para
elegir entre ellos y las formas cerradas.

Una mitad de **receptor** cierra el círculo.
[Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/)
toma el nivel que producen una fuente y un camino y pregunta qué le hace a los
animales que lo oyen: los audiogramas de grupo de Southall et al., las
funciones de ponderación auditiva regulatorias de la guía NMFS, los criterios
de inicio de TTS y de lesión, y la exposición acumulada ponderada de una
campaña de hincado medida frente a ellos.

Lee las páginas en ese orden: los niveles de referencia van primero porque
todo resultado de propagación se expresa en ellos, y los criterios de
exposición van al final porque consumen ambos. Excepcionalmente en este sitio,
la teoría de estas páginas vive junto a las guías en lugar de en la referencia
de teoría.

## Páginas de esta sección

- [Acústica submarina: ruido radiado e hincado de pilotes](/phonometry/es/underwater/underwater-acoustics/):
  niveles de referencia ISO 18405, ruido radiado por buques y nivel de fuente
  monopolar ISO 17208, y exposición por hincado de pilotes ISO 18406.
- [Propagación submarina del sonido](/phonometry/es/underwater/underwater-propagation/):
  pérdida de transmisión, velocidad del sonido, ecuación del sonar, reflexión
  en el lecho y ruido ambiente oceánico, en forma cerrada.
- [Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/):
  los solvers de modos normales, trazado de rayos y ecuación parabólica de
  la guía de ondas estratificada, cada uno validado contra una forma cerrada
  exacta, y cómo elegir modelo de propagación.
- [Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/):
  audiogramas de grupo, las funciones de ponderación auditiva regulatorias con
  la versión de guía seleccionable, los criterios de inicio de TTS y de lesión,
  y una evaluación trabajada de hincado de pilotes.
