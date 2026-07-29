---
title: "Niveles y ponderación"
description: "De la señal ponderada al número reportado: las ponderaciones frecuenciales de IEC 61672-1 y las curvas especiales, las balísticas Fast/Slow/Impulse, los niveles integrados, estadísticos y de dosis, y los indicadores ambientales de ISO 1996 construidos sobre ellos."
---

Un sonómetro hace tres cosas a una señal calibrada, en orden: la **pondera en
frecuencia** para imitar la sensibilidad del oído, la **suaviza en el tiempo**
con una balística normalizada y la **integra en un nivel**. Las páginas
de esta sección implementan exactamente esa cadena, una página por etapa,
siguiendo **IEC 61672-1:2013** tan de cerca que las ponderaciones se
verifican en CI contra las propias tablas de tolerancia de la norma.

[Ponderación frecuencial (A, C, Z)](/phonometry/es/guides/weighting/)
cubre la primera etapa. La curva A sigue la sensibilidad del oído a niveles
moderados y domina la regulación; la C es casi plana y sirve para picos y
comprobaciones de baja frecuencia; la Z no pondera por definición. El resto
de la familia vive en
[Ponderaciones especiales (G, B, D, AU)](/phonometry/es/guides/special-weightings/):
la curva G de **ISO 7196** extiende la idea al infrasonido, donde las
ponderaciones convencionales son ciegas, las curvas históricas B y D sirven a
los datos antiguos, y la AU rechaza el ultrasonido de una lectura de sonido
audible según IEC 61012.

[Ponderación temporal](/phonometry/es/guides/time-weighting/) cubre la
segunda etapa: las balísticas exponenciales Fast (125 ms), Slow (1 s) e
Impulse que deciden con qué rapidez el nivel mostrado sigue al sonido.
phonometry implementa las constantes de tiempo exactas, verificadas contra
las respuestas a ráfagas de tono de la norma.

[Niveles integrados y estadísticos](/phonometry/es/guides/levels/) es la
recompensa: el nivel continuo equivalente Leq y su versión ponderada A LAeq,
los niveles percentiles L10/L50/L90 que describen el ruido fluctuante, LCpeak
y SEL, la dosis de ruido de IEC 61252, más el espectrograma de octava para
visualizar nivel contra tiempo y banda a la vez. Es la página donde terminan
la mayoría de las mediciones prácticas, y donde arrancan las secciones
ambiental y laboral.

Convertir esos niveles en un veredicto regulatorio es
[Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/guides/environmental-levels/):
el nivel día-tarde-noche Lden y los niveles de valoración de **ISO 1996-1**
con sus ajustes, y la cadena de determinación de ISO 1996-2 con el ajuste
tonal, la corrección de ruido residual y el presupuesto de incertidumbre de
la medición.

## Páginas de esta sección

- [Ponderación frecuencial (A, C, Z)](/phonometry/es/guides/weighting/):
  las curvas A/C/Z de IEC 61672-1, el modo de precisión en alta frecuencia y
  la verificación de clase.
- [Ponderaciones especiales (G, B, D, AU)](/phonometry/es/guides/special-weightings/):
  la ponderación G para infrasonido de ISO 7196, las curvas históricas B y D
  y la AU según IEC 61012.
- [Ponderación temporal](/phonometry/es/guides/time-weighting/): las
  balísticas exponenciales Fast, Slow e Impulse.
- [Niveles integrados y estadísticos](/phonometry/es/guides/levels/): Leq y
  LAeq, niveles percentiles, LCpeak/SEL, dosis de ruido y espectrogramas de
  octava.
- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/guides/environmental-levels/):
  Lden, Ldn y niveles de valoración, ajuste tonal, ruido residual e
  incertidumbre.
