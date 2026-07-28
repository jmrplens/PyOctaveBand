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
[acústica de salas](/phonometry/es/guides/room-acoustics/) reaparecen como
picos en el espectro simulado de una sala, la pérdida por inserción de
[efecto del suelo y barreras](/phonometry/es/guides/ground-barriers/) puede
rederivarse colocando un obstáculo en el dominio, y la curvatura de rayos de
[refracción atmosférica](/phonometry/es/guides/atmospheric-refraction/)
emerge de un perfil de velocidad del sonido dependiente de la altura. Cuando
una geometría es demasiado irregular para esos modelos (salas de forma
extraña, barreras múltiples, suelo de impedancia mixta), la simulación es el
recurso que aún da una respuesta cuantitativa; cuando existe una forma
cerrada, prefiérela, y usa el solucionador para verificar las hipótesis en
las que se apoya.

## Páginas de esta sección

- [Simulación de ondas FDTD 2D](/phonometry/es/guides/fdtd-simulation/): el
  método FDTD presión-velocidad en malla escalonada según el capítulo 4 de
  Attenborough y Van Renterghem (2021), sus fuentes, sondas, obstáculos y
  condiciones de contorno, la cadena de campo cercano a lejano, los límites
  del 2D y la regla de dispersión numérica.
- [Ondas elásticas y acoplamiento fluido-sólido](/phonometry/es/guides/elastic-waves/):
  el solucionador compañero velocidad-esfuerzo P-SV (Virieux 1986) sobre la
  misma malla, con superficies libres por imagen de esfuerzos, ondas de
  Rayleigh, conversión de modo, ondas de interfase de Scholte y transmisión
  de placas sumergidas, cada una validada contra su forma cerrada exacta.
