---
title: "Sonido en exteriores"
description: "El sonido en su camino por el aire libre: el modelo de propagación ISO 9613 con su desglose de atenuación término a término, el efecto suelo esférico por acústica ondulatoria y la difracción de barreras avanzada que hay debajo, y la atmósfera refractante que ambos suponen inexistente: rayos curvos, zonas de sombra y la ecuación parabólica."
---

Esta sección es el **camino**: lo que le pasa a un sonido entre una fuente de
potencia conocida y un receptor a cientos de metros. Sus tres páginas van del
método de ingeniería a la física que aproxima, y de ahí a lo único que ambos
dan por supuesto que no ocurre.

[Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/)
es el método de ingeniería. Partiendo de la **potencia acústica** de la fuente,
la ISO 9613-2 resta, banda de octava a banda de octava, cada mecanismo que
atenúa el sonido por el camino: la divergencia geométrica, la absorción
atmosférica (suministrada por el coeficiente de tono puro de **ISO 9613-1**), el
efecto del suelo y el apantallamiento por barreras, con una corrección
meteorológica para promedios a largo plazo. La página mantiene visible el
desglose término a término, de modo que una predicción nunca es una caja negra:
se ve exactamente qué mecanismo compra cuántos decibelios a qué frecuencia.
Empieza aquí — las otras dos páginas existen para decir cuándo puede uno fiarse
de sus términos.

[Efecto suelo esférico y barreras avanzadas](/phonometry/es/environment/propagation/ground-barriers/)
baja por debajo de dos de esos términos hasta la acústica ondulatoria que
ajustan: el coeficiente de reflexión de onda esférica de Weyl-Van der Pol de un
suelo de impedancia finita, y la difracción de barreras por el número de Fresnel
de Kurze-Anderson, el semiplano rígido exacto, las barreras gruesas y la barrera
coherente de cuatro caminos sobre el suelo. Lo que resuelve es la interferencia
dependiente de la frecuencia que los términos por banda de octava suavizan: un
valle del efecto suelo es una cancelación entre un camino directo y uno
reflejado, a una frecuencia que depende de la geometría y de la impedancia del
suelo, y una corrección tabulada no puede saber dónde cae. Abre esta página
cuando la respuesta la domine el suelo o una pantalla, o cuando haya que
defender el resultado por bandas de octava frente a una medición.

[Refracción atmosférica: rayos y GFPE](/phonometry/es/environment/propagation/atmospheric-refraction/)
retira la hipótesis sobre la que se construyen las dos páginas anteriores. La
velocidad del sonido cambia con la altura, así que los rayos son curvos y no
rectos, y que eso importe es, sobre todo, cuestión de distancia: un gradiente
representativo de la capa superficial curva los rayos con un radio de unos
3,4 km, de modo que en los primeros cien metros los modelos homogéneos son
precisos, y más allá de unos cientos de metros manda la geometría. A favor del
viento, o bajo una inversión nocturna, los rayos se cierran sobre el suelo y
mantienen el nivel arriba; contra el viento el mismo perfil abre una sombra
acústica en la que el nivel se derrumba 20 dB o más. Esa asimetría — la misma
máquina a la misma distancia, con decenas de decibelios de diferencia según el
lado en que uno se sitúe — es lo que la ISO 9613-2 fija por decreto en su
convención de propagación favorable y comprime en la corrección meteorológica
escalar. Esta página la calcula, con rayos curvos y distancias de zona de sombra
en forma cerrada, y con la ecuación parabólica de la función de Green como campo
de referencia.

Léelas en ese orden. La evaluación en la que termina un nivel predicho no está
aquí: los niveles por periodo salen de [Niveles integrados y
estadísticos](/phonometry/es/signals/levels/levels/), y el Lden, el Ldn y el
nivel de evaluación, de [Niveles ambientales (ISO
1996-1/-2)](/phonometry/es/environment/assessment/environmental-levels/), en la
subsección de [evaluación](/phonometry/es/environment/assessment/). Las
potencias de fuente de las que parte una predicción están en
[Fuentes ambientales](/phonometry/es/environment/sources/) para carretera,
ferrocarril y aerogeneradores, en
[Potencia acústica e intensidad](/phonometry/es/devices/emission/) para una
máquina, y en [Ruido de aeronaves](/phonometry/es/aircraft/) para las aeronaves.

## Páginas de esta sección

- [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
  la absorción atmosférica de ISO 9613-1 y el método general de ISO 9613-2
  con desglose de atenuación por término y banda de octava.
- [Efecto suelo esférico y barreras avanzadas](/phonometry/es/environment/propagation/ground-barriers/):
  la reflexión de onda esférica de Weyl-Van der Pol en el suelo y la difracción
  de barreras por teoría ondulatoria (Kurze-Anderson, semiplano rígido exacto,
  barreras gruesas y la barrera coherente de cuatro caminos sobre el suelo).
- [Refracción atmosférica: rayos y GFPE](/phonometry/es/environment/propagation/atmospheric-refraction/):
  la propia atmósfera refractante: perfiles efectivos de velocidad del sonido,
  rayos curvados con sus zonas de sombra, y la ecuación parabólica de función
  de Green.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/)
  y [Emisión de la fuente ferroviaria CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-rail-emission/),
  ambas en [Fuentes ambientales](/phonometry/es/environment/sources/): la
  potencia acústica direccional por metro de línea fuente de la que parte una
  predicción.
- [Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/environment/assessment/impulsive-sound/),
  en [Evaluación y normativa](/phonometry/es/environment/assessment/): el
  ajuste por carácter que se aplica al nivel una vez ha llegado.

## Qué no cubre esta sección

Son modelos punto a punto, no un motor de mapas. Cada llamada toma una fuente,
un receptor y el suelo entre ambos; no hay perfil de elevación del terreno, ni
geometría de edificios, ni capa GIS, los dos modelos de refracción asumen un
suelo plano a altura cero y un perfil que varía solo con la altura y no a lo
largo del camino, y cómo se descompone una línea fuente en fuentes puntuales lo
declara fuera de alcance el propio CNOSSOS. El **método de propagación de
CNOSSOS-EU del apartado 2.5 no está implementado**: es un modelo distinto de la
ISO 9613-2, así que un cálculo que empareje potencias de fuente CNOSSOS con el
camino de aquí no es un cálculo CNOSSOS. El modelo coherente de barrera sobre
suelo pondera sus cuatro caminos difractados con un único coeficiente de
reflexión calculado sobre la geometría global, así que es coherente y recíproco
pero no es una solución por elementos de contorno, y ningún modelo de aquí
calcula la dispersión por turbulencia: la ISO 9613-2 la absorbe en los topes
fijos de su término de apantallamiento, y las páginas de acústica ondulatoria y
de refracción asumen sin más una atmósfera no turbulenta. Nada de estas páginas
produce una evaluación: ni Lden, ni valor límite, ni veredicto — eso es
[Evaluación y normativa](/phonometry/es/environment/assessment/).
