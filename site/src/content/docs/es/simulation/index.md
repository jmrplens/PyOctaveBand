---
title: "Simulación de ondas"
description: "Calcular el propio campo sonoro: un solucionador FDTD 2D determinista sobre una malla escalonada presión-velocidad, con fuentes, sondas, obstáculos rasterizados y contornos por lado, más su compañero elástico P-SV con ondas de Rayleigh, acoplamiento fluido-sólido y ondas de interfase de Scholte."
---

La mayor parte de esta biblioteca predice un número; esta sección calcula el
**propio campo de ondas**. Un solucionador de diferencias finitas en el
dominio del tiempo (FDTD) integra las ecuaciones acústicas lineales sobre una
malla 2D, de modo que la reflexión, la difracción, la interferencia, el
comportamiento modal y la refracción en medios inhomogéneos emergen de
primeros principios, y su compañero elástico lleva el mismo esquema a los
sólidos. Ambos solucionadores son deterministas (entradas idénticas dan
salidas idénticas bit a bit en la misma plataforma), están validados contra
oráculos analíticos y sirven además como motor de contraste para los modelos
en forma cerrada de las demás secciones.

La sección se divide según los medios que simula. La página acústica
explica el método numérico (el esquema leapfrog escalonado y su límite de
estabilidad de Courant), los bloques de construcción (fuentes, sondas,
obstáculos y condiciones de contorno, incluido el borde de impedancia real
de reacción local), la transformación de campo cercano a lejano, cuándo una
simulación ondulatoria merece su coste, qué puede y qué no puede decir un
dominio 2D sobre un problema 3D, y cómo la dispersión numérica fija la
regla de resolución de celdas por longitud de onda. La página elástica
extiende la misma malla escalonada a los sólidos: ondas de cizalla,
superficies libres con ondas de Rayleigh, y el acoplamiento fluido-sólido
con conversión de modo, ondas de interfase de Scholte y transmisión de
placas sumergidas.

Una buena forma de leerla es junto a las páginas de forma cerrada que
contrasta: las frecuencias modales de
[acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) reaparecen como
picos en el espectro simulado de una sala, la pérdida por inserción de
[efecto del suelo y barreras](/phonometry/es/environment/propagation/ground-barriers/) puede
rederivarse colocando un obstáculo en el dominio, y la curvatura de rayos de
[refracción atmosférica](/phonometry/es/environment/propagation/atmospheric-refraction/)
emerge de un perfil de velocidad del sonido dependiente de la altura. Cuando
una geometría es demasiado irregular para esos modelos (salas de forma
extraña, barreras múltiples, suelo de impedancia mixta), la simulación es el
recurso que aún da una respuesta cuantitativa; cuando existe una forma
cerrada, prefiérela, y usa el solucionador para verificar las hipótesis en
las que se apoya.

Esos contrastes no son solo argumentos: quince de las animaciones de esta
documentación son salida de estos dos solucionadores, y están archivadas en las
guías cuya física zanjan y no aquí. Los modos de sala creciendo en resonancia y
fuera de ella aparecen en
[acústica de salas](/phonometry/es/buildings/rooms/room-acoustics/) y en
[predicción de la reverberación](/phonometry/es/buildings/rooms/reverberation-prediction/),
que además lleva la sala de columnas que convierte un solo frente de onda en un
campo mezclado; la difracción por barrera a dos longitudes de onda, en
[propagación en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/)
y en [efecto del suelo y barreras](/phonometry/es/environment/propagation/ground-barriers/);
la refracción a favor y en contra del viento, en
[refracción atmosférica](/phonometry/es/environment/propagation/atmospheric-refraction/);
el patrón de lóbulos del efecto del suelo, en propagación en exteriores y en
[ruido de aeropuerto](/phonometry/es/aircraft/airport-noise/); los tubos de onda
estacionaria y de transmisión, en
[el tubo de impedancia](/phonometry/es/materials/absorbers/impedance-tube/); los
paneles QRD y de metadifusor, en
[difusores](/phonometry/es/materials/diffusers/diffusers/) y
[metadifusores](/phonometry/es/materials/diffusers/metadiffusers/); el absorbente
de rendija, en [absorbentes metamateriales](/phonometry/es/materials/absorbers/metamaterial-absorbers/);
la cámara de expansión, en [silenciadores](/phonometry/es/devices/noise-control/silencers/);
la abertura en la pared, en
[aislamiento acústico de paneles](/phonometry/es/buildings/design/panel-sound-insulation/);
y el canal SOFAR, en
[propagación submarina](/phonometry/es/underwater/underwater-propagation/). El
solucionador elástico aporta dos más: el paquete de flexión que entra en una
unión en L, en
[transmisión en uniones](/phonometry/es/vibration/structural/junction-transmission/),
y la placa en coincidencia, en aislamiento acústico de paneles. Las dos aparecen
también en la página elástica de más abajo, donde se explica el solucionador que
las produjo.

## Páginas de esta sección

- [Simulación de ondas FDTD 2D](/phonometry/es/simulation/fdtd-simulation/): el
  método FDTD presión-velocidad en malla escalonada según el capítulo 4 de
  Attenborough y Van Renterghem (2021), sus fuentes, sondas, obstáculos y
  condiciones de contorno, la cadena de campo cercano a lejano, los límites
  del 2D y la regla de dispersión numérica.
- [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/simulation/elastic-waves/):
  el solucionador compañero velocidad-esfuerzo P-SV (Virieux 1986) sobre la
  misma malla, con superficies libres por imagen de esfuerzos, ondas de
  Rayleigh, conversión de modo, ondas de interfase de Scholte y transmisión
  de placas sumergidas, cada una validada contra su forma cerrada exacta.
