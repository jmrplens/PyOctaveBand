---
title: "Ruido de aeronaves"
description: "Ruido de aeronaves con métricas fijadas internacionalmente: la cadena de certificación EPNL del Anexo 16 de la ICAO, la maquinaria de contornos de aeropuerto de ECAC Doc 29, el método del hemisferio para rotorcraft de ECAC Doc 32 y la base de datos ANP de EASA que los alimenta."
---

El ruido de aeronaves se calcula con métodos negociados internacionalmente de
dos clases. La **certificación** fija un único número por tipo de aeronave
hasta el último decimal, en los puntos de referencia que una norma coloca
alrededor de la pista. Los **métodos de contornos** toman esa flota
certificada y predicen lo que un aeropuerto hace al terreno que lo rodea. Las
cuatro páginas de esta sección cubren ambas clases, y comparten una misma
anatomía: un **descriptor de fuente** rigurosamente normalizado (una historia
temporal espectral, una tabla nivel-potencia-distancia o un hemisferio de
ruido), más **ajustes de propagación** normalizados que colocan la fuente en
un receptor.

[Ruido de aeronaves: nivel efectivo de ruido percibido](/phonometry/es/aircraft/aircraft-noise/)
cubre la certificación de ala fija. El **EPNL** del Anexo 16 de la ICAO
condensa una historia temporal en tercios de octava de un sobrevuelo en un
único valor en EPNdB a través de la molestia percibida, una corrección tonal
y una corrección de duración; la página añade el verificador de sistemas de
medida IEC 61265 y la absorción atmosférica SAE ARP 5534 usada en la cadena
de certificación.
[Ruido de aeropuertos (ECAC Doc 29)](/phonometry/es/aircraft/airport-noise/)
recoge el avión desde ahí: las tablas nivel-potencia-distancia, las
correcciones por segmento de una trayectoria (impedancia, atenuación lateral,
instalación del motor, duración, fracción de ruido y directividad de inicio
de rodaje) y el contorno de evento único sobre una malla de tierra.

[Ruido de rotorcraft: el método del hemisferio](/phonometry/es/aircraft/rotorcraft-noise/)
cubre los helicópteros, cuya fuerte directividad derrota a un nivel de fuente
de un solo número. ECAC Doc 32 describe en cambio la fuente como un
**hemisferio de ruido** (niveles de banda sobre una malla de ángulos de
emisión a una distancia de referencia de 60 m), propaga cada rayo con
divergencia esférica, absorción atmosférica y el efecto de suelo de
Chien-Soroka, interpola entre las condiciones de vuelo medidas a lo largo de
la trayectoria e integra el historial recibido en el SEL, LASmax y EPNL de
evento único y sus contornos en malla de tierra.

[La base de datos ANP de flota](/phonometry/es/aircraft/anp-fleet/) cierra el
círculo de los dos anteriores: las tablas nivel-potencia-distancia y las
trayectorias por defecto que EASA y EUROCONTROL publican para los tipos de
aeronave reales, listas para alimentar la cadena de Doc 29 sin escribir una
tabla a mano.

La física compartida conecta hacia fuera: la absorción atmosférica viene del
mismo modelo ISO 9613-1 que
[Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/),
y la misma lógica de ensayo de tipo gobierna el
[Ruido de aerogeneradores](/phonometry/es/environment/sources/wind-turbine-noise/),
que está archivado con las demás fuentes ambientales: su nivel de potencia
acústica aparente de IEC 61400-11 y su cadena de audibilidad tonal responden a
la misma pregunta para una fuente que no es una aeronave. Ese ensayo de
tonalidad es a su vez primo de los métodos de
[Psicoacústica](/phonometry/es/perception/psychoacoustics/).

Empieza por la pregunta. Para contrastar un avión con un límite de
certificación, o para entender de dónde salen los números publicados de un
tipo, empieza por la página del EPNL. Para predecir lo que hace un movimiento
en una dirección concreta de la calle, usa la página de Doc 29, con la página
de ANP aportando los datos de la aeronave. Para helicópteros, la página del
hemisferio sustituye a las dos. Lee las páginas de ala fija en ese orden: la
del EPNL define la métrica certificada, la de Doc 29 convierte aviones
certificados en contornos sobre el terreno a partir de tablas escritas a mano,
y la de ANP sustituye esas tablas escritas a mano por los datos de flota
publicados. La página de rotorcraft se sostiene sola (otra norma y otro modelo
de fuente) y puede leerse la primera si vienes por los helicópteros.

Las tres métricas no son intercambiables. El EPNL es una métrica de
*certificación* de un avión en un punto prescrito; el SEL y el LASmax son
métricas de evaluación de *evento único* en un receptor cualquiera; ninguna es
el índice de largo plazo con el que finalmente se juzga un estudio de usos del
suelo. Y el límite: esta sección no calcula índices acumulados de múltiples
eventos, no sintetiza tablas NPD a partir de datos de motor, no modela
operaciones de rotorcraft en estacionario, ralentí o rodaje, y no toca el
estampido sónico.

## Páginas de esta sección

- [Ruido de aeronaves: nivel efectivo de ruido percibido](/phonometry/es/aircraft/aircraft-noise/):
  la cadena EPNL del Anexo 16 de la ICAO, el verificador IEC 61265 y la
  absorción SAE ARP 5534.
- [Ruido de aeropuertos (ECAC Doc 29)](/phonometry/es/aircraft/airport-noise/):
  el motor NPD, la cadena de segmentos de evento único y el contorno de SEL
  en malla de tierra.
- [Ruido de rotorcraft: el método del hemisferio](/phonometry/es/aircraft/rotorcraft-noise/):
  el modelo de fuente de hemisferio de ruido de ECAC Doc 32, sus ajustes de
  propagación y las métricas y contornos de evento único.
- [La base de datos ANP de flota](/phonometry/es/aircraft/anp-fleet/): las
  tablas de EASA con curvas nivel-potencia-distancia y trayectorias por defecto
  que ejecutan la cadena de Doc 29 con una aeronave real.
