---
title: "Los mismos bits en cualquier máquina"
description: "Por qué la coma flotante deriva entre hosts, qué fija phonometry para que un filtro diseñado sea el mismo filtro byte a byte en cualquier CPU, BLAS, número de hilos o plataforma, y la evidencia medida tras la afirmación."
---

Un filtro de ponderación diseñado por phonometry es el mismo filtro, byte a
byte, en cualquier máquina: cualquier CPU, con o sin AVX-512; cualquier juego
de kernels BLAS; cualquier número de hilos; cualquier semilla de hash; y,
desde que se fijó el logaritmo, la biblioteca C de cualquier plataforma. Los
coeficientes que diseña un portátil son los que diseña un clúster, así que un
veredicto de conformidad, una figura publicada o un filtro embarcado en una
cadena de medida es un artefacto reproducible y no el relato de una máquina
sobre sí misma.

Esa frase no es el comportamiento normal del Python científico, y esta página
registra lo que costó, con los números.

## Por qué la coma flotante deriva entre hosts

IEEE 754 fija cada operación *individual*: dados dos dobles, su suma, resta,
producto, cociente y raíz cuadrada son el mismo patrón de bits en toda
máquina conforme. Todo lo que va más allá de una operación es por donde se
escapa la reproducibilidad, por tres puertas distintas.

**Las reducciones pueden reasociarse.** Una suma de muchos términos no tiene
orden definido, y un BLAS elige el que conviene a los registros vectoriales
que encuentra en tiempo de ejecución. phonometry midió lo que eso cuesta
sobre su propio ajuste de ponderaciones: dos juegos de kernels de OpenBLAS
coincidieron en todas las decisiones de aceptación de la búsqueda de
Levenberg-Marquardt durante treinta y nueve pasos consecutivos mientras los
costes que comparaban derivaban de 3&nbsp;&times;&nbsp;10⁻¹³ a
1&nbsp;&times;&nbsp;10⁻⁴ de separación, y las dos ejecuciones aterrizaron
después en **filtros distintos**: 5&nbsp;&times;&nbsp;10⁻⁵ de separación en
el coeficiente principal, 0,002&nbsp;dB en respuesta, unos visibles
0,0225&nbsp;pt en una curva dibujada. Un optimizador en un valle plano
amplifica lo que haga el último bit.

**Las trascendentes vectorizadas se despachan.** numpy elige los kernels de
`log`, `exp`, `tan`, `power`, `arctan` y `sin` según lo que ofrece la CPU, y
los kernels AVX-512 no coinciden hasta el último bit con los que corre una
máquina sin AVX-512. Medido bajo Intel SDE emulando Skylake-X contra el mismo
host en nativo: cada una de esas seis devuelve un digest distinto sobre las
mismas entradas. La fuga llega más lejos que los sitios de llamada obvios:
`numpy.geomspace` es `10 ** linspace(...)`, así que la propia rejilla de
frecuencias sobre la que se evalúa cada residuo del ajuste salía distinta
bajo AVX-512, y todos los pasos aguas abajo la heredaban.

**La libm de la plataforma es una elección de plataforma.** `math.log` es el
`log` que traiga la biblioteca C. El de glibc y el de Apple son ambos exactos
hasta cerca de media ulp, lo que significa que en un puñado de entradas
redondean legítimamente a vecinos *opuestos*, así que un bucle de diseño
construido sobre `math.log` devolvía filtros distintos en macOS y en Linux
sin nada en ningún sitio que lo dijera.

## Qué fija phonometry

El camino de diseño determinista responde a cada puerta en su terreno, en
`filters._weighting_design` y `filters._pinned_log`:

- **Sus propias reducciones.** Cada suma que el ajuste compara o con la que
  resuelve se pliega por pares en una forma que fija el código, y las
  ecuaciones normales se resuelven por eliminación gaussiana escrita a mano,
  el mismo algoritmo que corre el `dgesv` de LAPACK con el orden ya fuera de
  la elección de una biblioteca. El plegado por pares es además el orden
  *más exacto*: un redondeo O(&epsilon;&nbsp;log&nbsp;n) contra el
  O(&epsilon;&nbsp;n) de un acumulador corrido, así que no se sacrificó
  nada.
- **Aritmética real donde la compleja se despacha.** El producto complejo y
  el módulo complejo fusionan operaciones de forma distinta por CPU, así que
  los dos puntos donde el camino entregaba a numpy una expresión compleja
  están escritos en aritmética real.
- **Trascendentes fijadas.** Fuera de la iteración, cada trascendente se
  toma de la biblioteca C una vez por diseño. Dentro, el logaritmo corre
  tres cuartos de millón de veces por diseño, y el bucle que lo fijaba costó
  un factor cuatro; ahora es el propio algoritmo de `log` de glibc, el de
  tablas, deletreado en operaciones numpy que IEEE&nbsp;754 fija
  exactamente, aritmética simple, trabajo de bits enteros y consultas de
  tabla, con los multiply-add fusionados del original reproducidos por
  transformaciones exactas libres de error. Sus bits son suyos en toda
  máquina, que es más fuerte que llamar a la libm de nadie.
- **La rejilla deletreada.** La rejilla geométrica de frecuencias se genera
  término a término con los extremos fijados, así que el ajuste evalúa las
  mismas frecuencias en todas partes.

## La evidencia

Cada afirmación de arriba está medida, y la mayoría las re-miden tests que
corren en CI:

- **91.658.333 entradas, cero excepciones.** Cada valor distinto que el
  corpus de diseño entero pasa a su logaritmo se capturó y se comparó bit a
  bit contra el `log` de glibc: idéntico en todos, más alrededor de cien
  millones de sorteos adversariales sobre el rango completo de exponentes,
  la banda cercana a uno, los números subnormales y los bordes de las ramas. Una
  búsqueda adversarial encontró también el límite honesto: en torno a una
  entrada de cada treinta millones, en una banda estrecha, cuyo logaritmo
  verdadero queda a media ulp de ambos vecinos y donde las dos
  implementaciones se separan en el último bit. Ninguna puede alcanzar un
  diseño, porque en el camino de diseño la rutina fijada *es* la
  definición.
- **El corpus es byte-idéntico.** Los diseños de siete curvas a diecinueve
  frecuencias de muestreo, 133 en total, dan el mismo SHA-256 único antes y
  después de vectorizar el logaritmo: ningún coeficiente, figura ni valor de
  conformidad publicado se movió.
- **Digest a digest entre entornos.** Los diseños son bit-idénticos entre
  este host y el mismo host bajo emulación AVX-512, bajo cada juego de
  kernels de OpenBLAS seleccionable, cada número de hilos y cada semilla de
  hash; las comparaciones entorno contra entorno están fijadas como tests.
- **Idénticos entre plataformas.** Con el logaritmo fijado, los diseños de
  macOS se movieron esa última ulp una sola vez, hasta los valores que las
  demás plataformas ya publicaban, y la diferencia silenciosa entre Linux y
  macOS desapareció.

## Compruébalo tú

Dos diseños son el mismo diseño, y los bytes tienen nombre:

```python
import hashlib
import numpy as np
from phonometry import filters

una = filters.WeightingFilter(48000, "A").sos
otra = filters.WeightingFilter(48000, "A").sos
print(np.array_equal(una, otra))                        # True
print(hashlib.sha256(una.tobytes()).hexdigest()[:16])   # 991833ff389afe91
```

La segunda línea es la clave: ese digest no es «lo que salió en mi máquina»,
es el digest de la ponderación A a 48&nbsp;kHz de esta versión en cualquier
máquina. Si una versión futura mueve un coeficiente a propósito, el digest se
mueve con ella y el cambio es un suceso documentado, nunca una propiedad de
tu hardware.

## Alcance

La garantía cubre el camino de diseño determinista de ponderaciones de punta
a punta, y el procesamiento por bloques compone con ella: una cascada con
estado alimentada bloque a bloque es bit-idéntica a una sola pasada sobre el
registro entero, así que el streaming no gasta la garantía. Las funciones de
análisis generales construidas sobre numpy y SciPy en su forma corriente
siguen siendo reproducibles en una misma máquina pero pueden diferir en los
últimos bits entre CPUs, que es la condición normal del Python científico;
el tratamiento determinista se aplica donde importa la identidad de un
filtro.

El relato de ingeniería completo vive con el código, en los docstrings de
módulo de
[`filters/_weighting_design.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/_weighting_design.py)
y
[`filters/_pinned_log.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/_pinned_log.py),
y los tests que lo sostienen están en
[`tests/filters/test_weighting_design.py`](https://github.com/jmrplens/phonometry/blob/main/tests/filters/test_weighting_design.py)
y
[`tests/filters/test_pinned_log.py`](https://github.com/jmrplens/phonometry/blob/main/tests/filters/test_pinned_log.py).

## Véase también

- [Informe de conformidad](/phonometry/es/reference/conformance/): contra qué se comprueban los números; esta página es la razón de que las comprobaciones lean igual en todas partes.
- [Erratas de las fuentes publicadas](/phonometry/es/reference/errata/): la otra mitad de la historia de la evidencia: donde lo que está mal es el valor esperado impreso.
- [Ponderaciones en frecuencia](/phonometry/es/signals/levels/weighting/): la guía de las curvas que realiza el diseño determinista.
- [Procesamiento por bloques](/phonometry/es/signals/filters/block-processing/): la identidad de streaming en la que se apoya la sección de alcance.
