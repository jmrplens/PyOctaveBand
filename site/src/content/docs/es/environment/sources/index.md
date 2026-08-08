---
title: "Fuentes ambientales"
description: "Los descriptores de fuente de los que parte una predicción ambiental: la emisión de carretera y ferrocarril de CNOSSOS-EU según el Anexo II de la Directiva 2002/49/CE, y la potencia acústica aparente y la audibilidad tonal de un aerogenerador según IEC 61400-11."
---

Un modelo de propagación no admite una máquina; admite un **descriptor de
fuente** con una geometría fijada. Para el tráfico eso significa una línea
fuente incoherente que lleva una potencia acústica por metro a una altura
normalizada; para un aerogenerador significa una potencia acústica aparente
referida a una fuente puntual equivalente en el centro del rotor. La altura, el
rango de bandas y la directividad forman parte de la definición, no son
detalles de la medición, y por eso un método de emisión es una norma por
derecho propio y no un paso preliminar. Lo que produce cada página de aquí es
ese descriptor, en la forma que consume
[Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/).

[Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/)
implementa el apartado 2.2 del Anexo II de la Directiva 2002/49/CE en su texto
consolidado: la Directiva (UE) 2015/996 con la corrección de errores del DO L 5
de 2018, que restituye el rango de octavas de 63 Hz a 8 kHz que el apartado
original contradecía, y modificada por la Directiva Delegada (UE) 2021/1226,
que sustituye por completo las Tablas F-1 y F-4 y deja la fuente vigente unos
2,5 a 3,5 dB(A) más ruidosa que la de 2015 — así que cualquier comparación con
bibliografía anterior a 2021 arrastra ese desfase. Cada vehículo es una fuente
puntual a 0,05 m sobre el pavimento, con la primera reflexión en el pavimento
ya dentro de su potencia. Por categoría (ligeros, pesados medios, pesados,
ciclomotores, motocicletas) se suman en energía un término de rodadura y otro
de propulsión, se corrigen por pavimento, temperatura del aire, neumáticos con
clavos y pendiente, se ajustan en las proximidades de los cruces, y se
convierten en una potencia direccional por metro de línea fuente.

[Emisión de la fuente ferroviaria CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-rail-emission/)
implementa el apartado 2.3 con el mismo patrón, pero con **dos** líneas fuente
equivalentes, a 0,5 m y a 4,0 m sobre la cabeza del carril, porque las fuentes
físicas radian desde alturas distintas. Empieza un paso más atrás que el método
viario: en los espectros de rugosidad de rueda y carril, pasados por el filtro
de contacto y por las funciones de transferencia del vehículo y de la vía, con
la conversión de longitud de onda a frecuencia a la velocidad del tren que hace
que la aritmética ferroviaria sea distinta de la viaria. El ruido de impacto en
juntas y desvíos, el chirrido en curva, la tracción, el ruido aerodinámico por
encima de 200 km/h y un término de puente se asignan cada uno a la altura desde
la que radian.

[Ruido de aerogeneradores: potencia acústica y audibilidad tonal](/phonometry/es/environment/sources/wind-turbine-noise/)
es la IEC 61400-11, donde el descriptor se **mide** en vez de tabularse. Con el
micrófono sobre una placa en el suelo a la distancia horizontal R0 = H + D/2, la
potencia acústica aparente por banda se sigue del nivel de presión medido y de
la distancia oblicua al centro del rotor, con los −6 dB de la fórmula dando
cuenta de la duplicación de presión sobre la placa; los resultados se clasifican
por velocidad de viento normalizada. Esa misma página lleva la audibilidad tonal
que decide si un tono de paso de pala, de multiplicadora o de generador destaca
sobre su ruido de enmascaramiento, y termina en una ficha de evaluación
`.report()`.

Lee primero la página de carretera aunque tu trabajo sea ferroviario: introduce
la contabilidad por líneas fuente y el encaje del Anexo II que la página de
ferrocarril reutiliza. La de aerogeneradores es independiente de ambas.

## Páginas de esta sección

- [Emisión de la fuente de tráfico viario CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-road-emission/):
  las potencias acústicas de rodadura y propulsión por categoría de vehículo,
  sus correcciones por pavimento, temperatura, neumáticos con clavos, pendiente
  y cruces, y la potencia direccional por metro de línea fuente.
- [Emisión de la fuente ferroviaria CNOSSOS-EU](/phonometry/es/environment/sources/cnossos-rail-emission/):
  de la rugosidad y las funciones de transferencia a las dos líneas fuente
  equivalentes a 0,5 m y 4,0 m, con los términos de impacto, chirrido,
  tracción, aerodinámico y de puente.
- [Ruido de aerogeneradores: potencia acústica y audibilidad tonal](/phonometry/es/environment/sources/wind-turbine-noise/):
  la potencia acústica aparente de IEC 61400-11 referida al centro del rotor,
  su clasificación por velocidad de viento y la cadena de audibilidad tonal,
  con la ficha de evaluación.

## Véase también

Páginas de otras áreas del sitio en las que se apoya esta sección:

- [Propagación del sonido en exteriores](/phonometry/es/environment/propagation/outdoor-propagation/):
  el modelo de camino al que está hecho para alimentar todo descriptor de aquí.
- [Potencia acústica e intensidad](/phonometry/es/devices/emission/): cómo se
  caracteriza una máquina que no es un vehículo.

## Qué no cubre esta sección

Faltan dos de las cuatro fuentes de CNOSSOS, por omisión y no por descuido: la
**fuente industrial** del apartado 2.4 y el Apéndice H no está implementada, y
tampoco la **fuente de aeronaves** de los apartados 2.6 y 2.7 — el ruido de
aeronaves lo cubren los métodos de la ICAO y de la ECAC en [Ruido de
aeronaves](/phonometry/es/aircraft/), que es una familia de modelos
completamente distinta. Tampoco lo está el **método de propagación de CNOSSOS**
del apartado 2.5: difiere del modelo ISO 9613-2 que implementa esta biblioteca,
así que emparejar estas potencias de fuente con [Propagación del sonido en
exteriores](/phonometry/es/environment/propagation/outdoor-propagation/) no da
un resultado CNOSSOS. Dentro de los dos métodos que sí están, tres huecos
vienen de los propios documentos fuente: la categoría abierta de vehículos 5 no
tiene coeficientes en el Apéndice F y no se modeliza, las clases de rugosidad
ferroviaria N y B no llevan espectro en el Apéndice G y las tiene que aportar
el Estado miembro, y cómo se descompone una línea fuente en fuentes puntuales
lo declara fuera de alcance el propio método. Las cocheras, las estaciones y la
megafonía son fuentes ferroviarias según el 2.3.3 pero se tratan por el método
industrial, así que tampoco están aquí.
