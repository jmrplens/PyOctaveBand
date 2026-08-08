---
title: "Medio ambiente y transporte"
description: "El sonido en exteriores y las fuentes que lo dominan: el modelo de propagación ISO 9613, el nivel de evaluación de ISO 1996 y su aplicación española, el ajuste por sonidos impulsivos NT ACOU 112, y los modelos de fuente CNOSSOS-EU de carretera y ferrocarril y de aerogeneradores de IEC 61400-11."
---

El ruido ambiental es un problema fuente-camino-receptor estirado a lo largo
de cientos de metros de aire libre. Esta sección cubre los tres. Las páginas
de **propagación** se ocupan del camino: ISO 9613 predice, banda a banda,
cuánto nivel sobrevive a la divergencia, la absorción del aire, el suelo y
cualquier barrera de camino al receptor, y los modelos ondulatorios de suelo y
de refracción dicen cuándo ese método de ingeniería deja de bastar.

Las páginas de **evaluación** se ocupan de lo que pasa una vez ha llegado el
sonido: el nivel de evaluación de ISO 1996 y los indicadores día-tarde-noche,
su aplicación española en el RD 1367/2007, y el ajuste de NT ACOU 112 que
cuantifica cuándo el carácter impulsivo hace un sonido recibido más molesto de
lo que sugiere su LAeq.

Las páginas de **fuentes** se ocupan del otro extremo: lo que emite, descrito
como lo quiere un modelo ambiental. CNOSSOS-EU da al tráfico viario y al
ferrocarril una potencia de fuente por banda y por categoría, e IEC 61400-11
valora un aerogenerador por su potencia acústica aparente y su audibilidad
tonal. Lo que los une es el patrón: un descriptor de fuente cuidadosamente
normalizado que el modelo de camino de arriba atenúa después.

Esta sección se apoya en el núcleo de la biblioteca, pero solo hasta el nivel
de periodo.
[Niveles integrados y estadísticos](/phonometry/es/signals/levels/levels/)
aporta el LAeq, los percentiles y los niveles de suceso de cada periodo de
referencia; lo que convierte esos niveles de periodo en Lden, Ldn y el nivel de
evaluación, con la corrección por tonalidad, la corrección por ruido residual y
el balance de incertidumbre encima, es
[Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/),
en esta misma sección. La
absorción atmosférica que consume todo modelo de propagación se comparte con
las páginas de salas y materiales. Empieza por
[Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
introduce la contabilidad fuente-camino-receptor que las páginas de
transporte reutilizan.

Los tres trabajos para los que se usa normalmente esta sección combinan las
subsecciones de manera distinta. Un **mapa estratégico de ruido** es un modelo
de fuente CNOSSOS, un modelo de propagación y el Lden. La **evaluación de una
instalación o de una licencia** es una potencia acústica medida (determinada en
[Fuentes y dispositivos](/phonometry/es/devices/emission/)), la ISO 9613-2 hasta
la vivienda más próxima, y el nivel de evaluación de ISO 1996 con sus ajustes.
Una **inspección de actividad** es un sonómetro en un punto receptor y el
RD 1367/2007, o la reglamentación nacional que corresponda, sin ningún modelo
de propagación en la cadena.

## [Evaluación y normativa](/phonometry/es/environment/assessment/)

Con qué se compara el sonido recibido, una vez ha llegado.

- [Niveles ambientales (ISO 1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/):
  los indicadores día-tarde-noche, el nivel de evaluación y las correcciones
  que convierten un LAeq medido en uno evaluado.
- [Normativa española de ruido (RD 1367/2007)](/phonometry/es/environment/assessment/spanish-noise-regulation/):
  la aplicación nacional de esa cadena, con sus propios límites y sus propias
  correcciones por tonalidad, baja frecuencia e impulsividad.
- [Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/environment/assessment/impulsive-sound/):
  la prominencia predicha de los sonidos impulsivos y el ajuste graduado
  añadido al LAeq.

## [Sonido en exteriores](/phonometry/es/environment/propagation/)

El camino de una fuente exterior a un receptor, y el carácter de lo que
llega.

- [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
  absorción atmosférica (ISO 9613-1) y el método general de ISO 9613-2 con su
  desglose de atenuación término a término, incluidos el término de suelo
  tabulado y el término de apantallamiento por barrera.
- [Efecto suelo esférico y barreras avanzadas](/phonometry/es/environment/propagation/ground-barriers/):
  la acústica ondulatoria que hay debajo de esos dos ajustes — el coeficiente
  de reflexión de onda esférica de Weyl-Van der Pol sobre un suelo de
  impedancia finita, y la difracción de pantallas por teoría ondulatoria.
- [Refracción atmosférica: rayos y GFPE](/phonometry/es/environment/propagation/atmospheric-refraction/):
  cómo los gradientes de viento y temperatura curvan un rayo hacia dentro o
  fuera de una zona de sombra.

## [Fuentes ambientales](/phonometry/es/environment/sources/)

Lo que emite, descrito como lo quiere un modelo ambiental: una potencia de
fuente por banda, lista para que el camino de arriba la atenúe.

- [Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/):
  la potencia de rodadura y propulsión de un flujo de tráfico, por categoría
  de vehículo y por banda.
- [Emisión de la fuente ferroviaria CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-rail-emission/):
  el equivalente para ferrocarril, con sus alturas de fuente y sus aportes de
  rodadura, tracción y aerodinámico.
- [Ruido de aerogeneradores: potencia y audibilidad tonal](/phonometry/es/environment/sources/wind-turbine-noise/):
  el nivel de potencia acústica aparente de IEC 61400-11 y la cadena de
  audibilidad tonal.

Las aeronaves son la otra fuente de transporte con métricas fijadas
internacionalmente, y tienen un área propia:
[Ruido de aeronaves](/phonometry/es/aircraft/).

## Qué no cubre esta sección

Del Anexo II de CNOSSOS-EU solo está implementado el lado de la fuente, y solo
dos de sus cuatro fuentes. La **fuente industrial** del apartado 2.4 y el
Apéndice H, y la **fuente de aeronaves** de los apartados 2.6 y 2.7, no están
implementadas; el ruido de aeronaves lo cubren los métodos de la OACI y de la
ECAC en [Ruido de aeronaves](/phonometry/es/aircraft/), que es una familia de
modelos distinta, y una máquina que no es un vehículo se caracteriza como una
potencia acústica en [Fuentes y
dispositivos](/phonometry/es/devices/emission/). Tampoco está implementado el
**método de propagación de CNOSSOS** del apartado 2.5: el camino de aquí es la
ISO 9613-2, un modelo distinto, así que una cadena construida con fuentes
CNOSSOS y la propagación de esta biblioteca no es un cálculo CNOSSOS y no debe
declararse como tal.

Nada de aquí es un motor de mapas. No hay modelo de terreno, ni geometría de
ciudad, ni capa GIS: las funciones de propagación toman una fuente, un receptor
y el suelo entre ambos, los dos modelos de refracción suponen un suelo plano en
z = 0, y cómo se descompone una línea fuente en fuentes puntuales lo declara
fuera de alcance el propio CNOSSOS. Del lado de la evaluación, la biblioteca
empieza donde termina el sonómetro — las posiciones del receptor y las
correcciones de fachada de ISO 1996-2, y los procedimientos de medición del
anexo IV del RD 1367/2007 (posiciones de micrófono, duración de las series,
número de mediciones), no están implementados, solo la aritmética que sigue una
vez aplicados. La zonificación acústica, los mapas de ruido y los planes de
acción de la Ley 37/2003 son instrumentos de planificación, no cálculos.

## Antes y después de estas páginas

Toda evaluación de aquí es un $L_{eq}$ corregido, así que la calibración, la
ponderación y la integración temporal que lo producen están en [Análisis de
señal](/phonometry/es/signals/), y [Construye un
sonómetro](/phonometry/es/signals/sound-level-meter/) recorre esa cadena de
principio a fin en una sola página ejecutable. Las derivaciones están en la
[teoría de medio ambiente y
transporte](/phonometry/es/reference/theory/environment-transport/): los
descriptores de ISO 1996-1, el criterio de prominencia de NT ACOU 112 y los
términos de atenuación de ISO 9613.

Si has llegado aquí desde una búsqueda y quieres la forma de la biblioteca
entera, [¿Qué necesitas medir?](/phonometry/es/start/tasks/) la indexa por el
trabajo y [Todas las guías](/phonometry/es/start/guides/) lista todas las
páginas con una línea sobre cada una.
