---
title: "Ruido de aeronaves"
description: "Ruido de aeronaves con métricas fijadas internacionalmente: la cadena de certificación EPNL del Anexo 16 de la ICAO, la maquinaria de contornos de aeropuerto de ECAC Doc 29, el método del hemisferio para rotorcraft de ECAC Doc 32 y la base de datos ANP de EASA que los alimenta."
---

Las aeronaves son fuentes de ruido lo bastante importantes como para tener
métricas propias negociadas internacionalmente, cada una fijada hasta el
último decimal por un marco de certificación. Las cuatro páginas de esta sección
implementan esos marcos, y comparten una misma anatomía: un **descriptor de fuente** rigurosamente
normalizado, más **ajustes de propagación** normalizados que colocan la
fuente en un receptor.

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
