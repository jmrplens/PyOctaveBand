---
title: "Erratas de las fuentes publicadas"
description: "Defectos encontrados en las normas, los documentos de guía y los libros de los que parte la biblioteca: erratas de imprenta, ejemplos resueltos que contradicen su propio articulado y qué hace la biblioteca con cada uno."
---

Implementar una norma en sala limpia significa volver a deducir cada fórmula,
constante y ejemplo resuelto a partir del documento fuente y no del código de
otra persona. Hecho sobre cientos de documentos, ese proceso encuentra
defectos en las propias fuentes: un ejemplo resuelto que contradice su
articulado, una constante a la que la composición tipográfica le comió un
dígito, una referencia cruzada que apunta a la ecuación equivocada.

Esta página es el registro de esos hallazgos. Cada entrada nombra la edición
impresa y el punto exacto, cita lo que dice el documento, muestra por qué no
puede ser correcto, aporta la evidencia independiente y declara qué lectura
implementa la biblioteca y qué test de regresión la fija. Un defecto listado
aquí nunca es un defecto del *método*: en todos los casos la lectura
pretendida se ha podido establecer a partir del propio documento o de la
física.

Léela junto al
[informe de conformidad](/phonometry/es/reference/conformance/), que muestra
los números que calcula la biblioteca; esta página explica el puñado de sitios
donde lo que está mal es el valor esperado impreso.

:::note
Esta página reproduce la edición española del registro, traducida entrada a
entrada. La redacción autoritativa es la inglesa, que es la que se ha
comunicado o se comunicará a los organismos emisores; las citas textuales, las
matemáticas y los valores impresos se reproducen sin traducir, tal como los
imprime cada fuente.
:::

El registro vive en
[`docs/ERRATA.md`](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md)
con su edición española en
[`docs/ERRATA.es.md`](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.es.md),
y esta última se trasplanta aquí en tiempo de compilación con
`make site-reports`, que además exige que las dos ediciones lleven las mismas
entradas en el mismo orden, así que ninguna pareja puede discrepar.

<!-- BEGIN GENERATED BODY - transplanted from docs/ERRATA.es.md by scripts/generate_site_reports.py (`make site-reports`). Edit the source document, never the text below. -->

Durante la implementación en sala limpia de esta biblioteca, cada fórmula,
constante y ejemplo resuelto se vuelve a deducir y a recalcular de forma
independiente a partir de los documentos fuente. Ese proceso saca a la luz, de
cuando en cuando, defectos de las propias fuentes: erratas de imprenta,
ejemplos resueltos que contradicen su propio texto normativo y redacciones
ambiguas. Este fichero registra cada caso confirmado con la evidencia, lo que
hace la biblioteca al respecto y si se ha notificado.

El registro cubre todos los tipos de fuente publicada de los que parte la
biblioteca: normas (ISO, IEC, EN), documentos de guía e informes técnicos
(EASA, ECAC, NRL), libros y artículos de revista. Las fuentes no normativas
quedan marcadas como tales en su entrada.

Las entradas describen las ediciones impresas concretas que se citan. Un
defecto listado aquí no es un defecto del método; en todos los casos la
lectura pretendida se ha podido establecer a partir del propio documento o de
la física, y la biblioteca implementa esa lectura. Cuando la lectura cambia
algún número que la biblioteca da, la entrada nombra la comprobación o el test
que lo fija; cuando el defecto es una etiqueta, una referencia cruzada o una
tabla que la biblioteca nunca lee, la entrada deja constancia de que no hizo
falta ningún cambio.

Leyenda de estados: **sin notificar** (registrado solo aquí) / **notificado**
(comunicado al organismo emisor, con fecha y referencia).

Una afirmación que dependa de los caracteres exactos de una fórmula,
constante, coeficiente, símbolo, desigualdad o celda de tabla se verifica
contra **la página tal como está impresa**, y su punto de Evidencia cita esa
página por índice de página del PDF y folio impreso. El texto extraído puede
localizar una página; nunca se cita como «el impreso», porque las capas de
texto de los PDF borran glifos en silencio (la mayoría de las fuentes citadas
aquí no emiten ningún `√`, así que `f_T/√2` se extrae como `f_T/2`). El
desplazamiento de páginas de cada documento se establece empíricamente, porque
difiere entre documentos y deriva entre capítulos de un mismo libro. Las
entradas que descansan en otra cosa, un recálculo o la comparación de dos
frases, lo dicen en un aviso inicial o figuran en la lista de excepciones de
[`scripts/check_errata_evidence.py`](https://github.com/jmrplens/phonometry/blob/main/scripts/check_errata_evidence.py),
que es la comprobación que hace cumplir la regla; véase
[CONTRIBUTING.md](https://github.com/jmrplens/phonometry/blob/main/CONTRIBUTING.md#6-filing-an-errata-entry).

Esta edición española traduce la prosa del registro entrada a entrada. La
redacción autoritativa es la inglesa de [ERRATA.md](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md), que es la que
se ha comunicado o se comunicará a los organismos emisores; las citas
textuales, las matemáticas y los valores impresos se reproducen aquí sin
traducir, tal como los imprime cada fuente. `make site-reports` mantiene las
dos ediciones con las mismas entradas y en el mismo orden.


---

## ISO 717-2:2020, Anexo C, ejemplo C.1 (C_I del suelo desnudo)

- **Ubicación:** Anexo C, Tabla C.1 (p. 17 impresa) y el cálculo de $C_I$
  impreso en la misma celda.
- **El impreso:** $L_{n,\text{sum}} = 83{,}523\,8\ldots = 84\ \text{dB}$ y
  $C_I = 84 - 15 - 79 = -10\ \text{dB}$ para el ejemplo del suelo desnudo.
- **El problema:** dos defectos independientes en la misma celda. (a) El
  apartado A.2.1 define $C_I$ desde la suma energética de 100 Hz a 2500 Hz
  (las quince primeras bandas de tercio de octava); el valor impreso solo se
  reproduce si se incluye la banda de 3150 Hz, en contradicción con A.2.1. La
  suma correcta de 100 Hz a 2500 Hz es 83,2613 dB, redondeada 83, lo que da
  $C_I = -11$. (b) Incluso leída como la suma de dieciséis bandas, los dígitos
  impresos están mal en la última cifra: la columna $L_n$ del suelo desnudo
  suma 83,523 4 dB, no los 83,523 **8** dB impresos. El defecto queda
  confinado a esa celda, porque la columna con revestimiento de la misma tabla
  imprime $L_{n,\text{sum}} = 76{,}059\,3\ldots$ y se recalcula a
  76,059 29 dB, reproduciendo todos los dígitos impresos. Ni (a) ni (b)
  cambian los 84 dB redondeados, así que solo (a) mueve $C_I$.
- **Evidencia:** recálculo independiente de ambas sumas a partir de los
  niveles por banda impresos (16 bandas 83,523 38 dB, 15 bandas 83,261 27 dB,
  con revestimiento 16 bandas 76,059 29 dB); la edición de 2013 del mismo
  ejemplo imprime $C_I = -11$. Verificado en la página 23 del PDF (p. 17
  impresa) y la página 17 del PDF (p. 11 impresa) de ISO 717-2:2020, y en la
  página 22 del PDF (p. 14 impresa) de ISO 717-2:2013.
- **Comportamiento de la biblioteca:** implementa A.2.1 tal como está escrito
  y fija $C_I = -11$ con el impreso de 2013 como oráculo
  ([`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data), comprobación de
  conformidad «ISO 717-2 Annex C, Table C.1»).
- **Estado:** sin notificar.

## ISO 717-2:2020, Anexo C, ejemplo C.2 (suelo revestido: valor de 800 Hz y cadena del CI)

- **Ubicación:** Anexo C, Tabla C.2 (p. 18 impresa), el ejemplo resuelto de
  $\Delta L_w$ / $\Delta L_\text{lin}$.
- **El impreso:** (a) el valor del suelo de referencia a 800 Hz está impreso
  como 71,0 dB; (b) la línea del $C_I$ imprime
  $L_{n,\text{sum}} = 75{,}252\,7\ldots = 75\ \text{dB}$ y
  $C_I = 75 - 15 - 63 = -3\ \text{dB}$, que alimenta
  $\Delta L_\text{lin} = 78 - 11 - (63 - 3) = 7\ \text{dB}$.
- **El problema:** dos defectos independientes. (a) El suelo de referencia de
  la Tabla 4 normativa vale 71,5 dB a 800 Hz, y la propia columna es una rampa
  limpia de +0,5 dB por tercio de octava desde 67,0 dB a 100 Hz hasta 72,0 dB
  a 1000 Hz, que los 71,0 dB impresos rompen repitiendo la celda de 630 Hz. La
  errata se propaga por su propia fila y hasta el total de la tabla, tres
  celdas más que la tabla imprime y que una revisión anterior de esta entrada
  no nombraba: la celda $L_{n,r,0} - \Delta L$ a 800 Hz está impresa como
  64,0 dB ($= 71{,}0 - 7{,}0$) donde 71,5 da 64,5; la desviación desfavorable
  está impresa como 3,0 dB ($= 64{,}0 - 61$) donde la celda corregida da 3,5;
  y el `Sum 27,9` impreso es la suma de las trece desviaciones desfavorables
  incluyendo ese 3,0, donde la cadena corregida da 28,4. Nada de eso mueve la
  valoración: 28,4 dB sigue por debajo del criterio de desplazamiento de
  32,0 dB, así que $L_{n,w,r} = 63\ \text{dB}$ y
  $\Delta L_w = 15\ \text{dB}$ en cualquier caso. (b) Los 75,2527 dB impresos
  son exactamente la suma energética de la *columna equivocada sobre el rango
  equivocado*: el suelo medido «with covering» sobre las dieciséis bandas
  de 100 Hz a 3150 Hz. A.2.1 define $C_I$ desde el suelo de referencia con
  revestimiento (la columna $L_{n,r,0} - \Delta L$) de 100 Hz a 2500 Hz (15
  bandas), lo que da 75,674 dB (cadena impresa) o 75,710 dB (celda de 800 Hz
  corregida), ambos redondean a 76 dB, así que
  $C_{I,r} = 76 - 15 - 63 = -2$ en cualquier caso, dando
  $C_{I,\Delta} = -11 - (-2) = -9$ y $\Delta L_\text{lin} = 6\ \text{dB}$, no
  la cadena impresa de −3 / −8 / 7 dB.
- **Evidencia:** recálculo independiente de todas las sumas candidatas y de
  todas las celdas de la fila de 800 Hz a partir de los valores por banda
  impresos; el 75,2527 impreso se reproduce con todos sus dígitos solo como la
  suma de 16 bandas de la columna con revestimiento, y todas las demás celdas
  de las columnas $L_{n,r,0} - \Delta L$ y de desviación se reproducen
  exactamente desde el suelo de referencia impreso, así que la fila de 800 Hz
  es la única que no. Verificado en la página 24 del PDF (p. 18 impresa) y la
  página 13 del PDF (p. 7 impresa) de ISO 717-2:2020.
- **Comportamiento de la biblioteca:** deriva el suelo de referencia revestido
  desde los valores normativos de la Tabla 4 y suma según A.2.1, fijando
  $\Delta L_w = 15\ \text{dB}$ y $C_{I,\Delta} = -9$; la comprobación de
  conformidad anota la procedencia explícitamente.
- **Estado:** sin notificar.

## ISO 2631-5:2018, ejemplos resueltos del Anexo C (fórmula masculina desplegada, R femenino)

- **Ubicación:** Anexo C: el ejemplo resuelto masculino desplegado (varón de
  82 kg, $m_z = 0{,}029\ \text{MPa}/(\text{m/s}^2)$, p. 19 impresa) y la
  NOTA 5 (mujer de 64 kg, $m_z = 0{,}025\ \text{MPa}/(\text{m/s}^2)$, p. 20
  impresa).
- **El impreso:** (a) el ejemplo masculino se despliega como

  $$
  R = \left\{ \sum_{i=0}^{20-1}
  \left[ \frac{1{,}62\ \text{MPa}\,(120)^{1/6}}
  {6{,}75\ \text{MPa} - 0{,}052\ \text{MPa}\,(20+i)} \right]^{6}
  \right\}^{1/6} \approx 1{,}22
  $$

  y (b) la NOTA 5 declara $R = 0{,}97$ para el caso femenino.
- **El problema:** dos defectos independientes. (a) La fórmula masculina
  desplegada omite el término $-S_{\text{stat},i}$ que la Fórmula (C.3)
  normativa pone en el denominador, y que el propio anexo fija en
  $S_\text{stat} = 0{,}029 \cdot 9{,}81 = 0{,}281\ \text{MPa}$ en la frase que
  sigue a la lista de definiciones de la Fórmula (C.3). Evaluada exactamente
  como está desplegada, la suma da $R = 1{,}1497$, que se imprime 1,15, no el
  1,22 impreso; restaurar el término que falta da 1,2168 con el
  $S_\text{stat} = 0{,}281\ \text{MPa}$ impreso y 1,2177 con el exacto
  $m_z \cdot 9{,}81 = 0{,}2845\ \text{MPa}$, es decir, el 1,22 impreso en
  cualquiera de los dos casos. El *resultado* impreso es por tanto correcto y
  la *fórmula* impresa no. (b) El recálculo exacto de la Fórmula (C.3) con los
  propios datos de la NOTA 5 ($m_z = 0{,}025$, coeficiente de edad 0,039,
  $b = 20$, $n = 20$, $N = 120$) da $R = 0{,}9621$, que redondea a 0,96; el
  mismo código reproduce el ejemplo masculino exactamente, y el
  $S_d = 1{,}40\ \text{MPa}$ de la nota coincide con el exacto 1,3992, así que
  la discrepancia queda confinada al último dígito del $R$ femenino impreso.
- **Evidencia:** recálculo término a término de la suma de C.3 bajo ambas
  lecturas del denominador, con el ejemplo masculino como discriminador: el
  1,22 impreso solo es alcanzable con $-S_\text{stat}$, y el 1,15 solo sin él.
  Verificado en las páginas 23 (p. 17 impresa), 24 (p. 18 impresa), 25 (p. 19
  impresa) y 26 (p. 20 impresa) del PDF de ISO 2631-5:2018.
- **Comportamiento de la biblioteca:** implementa la Fórmula (C.3) tal como
  está escrita, con $-S_\text{stat}$; el ancla masculina fija 1,22 y el ancla
  de test femenina conserva el 0,97 impreso con una tolerancia que documenta
  el 0,9621 recalculado.
- **Estado:** sin notificar.

## Ainslie (2010), la Ec. (4.6) frente a su propio folio 177, y el exponente de la Ec. (4.13)

- **Ubicación:** *Principles of Sonar Performance Modelling* (Springer 2010),
  Ec. (4.6) en el folio impreso 127; la densidad del agua de mar citada en el
  apartado 4.4 del folio impreso 177; Ec. (4.13) del folio impreso 135.
- **Lo impreso:** la Ec. (4.6) da la densidad del agua de mar como
  $\hat\rho = 1027 + 4{,}3\times10^{-7}\hat P_\mathrm{w} + 0{,}75[S-35] -
  0{,}16[\hat T-10] - 0{,}004[\hat T-10]^2$, atribuida a Pierce (1989,
  p. 34), con las unidades fijadas por las Ecs. (4.7) a (4.10) del folio 128:
  presión en pascales, temperatura en grados Celsius, densidad en kg/m³. La
  Ec. (4.4) del folio 127 define esa presión como
  $P_\mathrm{w}(z) = P_\mathrm{atm} + \int_0^z \rho g\,\mathrm{d}\zeta$,
  y la Ec. (4.11) del folio 128 la evalúa en $98\,066{,}5 \times 1{,}04 =
  101\,989{,}16$ Pa en superficie. El folio 177 enuncia después, para los
  cocientes que escalan las correlaciones de sedimento de Bachman, «*standard
  conditions involving atmospheric pressure, a temperature of 23 °C, and
  salinity 35*» con $\rho_\mathrm{w} = 1024{,}2$ kg/m³.
- **El problema:** dos defectos de distinta naturaleza.

  (a) El 1024,2 del folio 177 no se sigue de la Ec. (4.6) leída con la
  Ec. (4.4). A 23 °C, salinidad 35 y una atmósfera, la ecuación da 1024,287 9,
  que imprime 1024,3. El 1024,2 impreso es lo que da la ecuación con su término
  de presión a cero, es decir, leyendo $P_\mathrm{w}$ como presión manométrica
  contra la definición que enuncia el mismo capítulo. La diferencia son
  0,043 9 kg/m³, o 4,3 partes por cien mil.

  (b) La Ec. (4.13), que despeja la (4.6) para estimar la salinidad a partir de
  una densidad medida, imprime el coeficiente de presión como
  $4{,}3\times10^{-5}$ donde la (4.6) tiene $4{,}3\times10^{-7}$. Dos órdenes
  de magnitud, y no es otra magnitud reformulada: es el mismo coeficiente en el
  mismo papel. Arrastrado a 23 °C da 1028,63 kg/m³ frente a 1024,29, un error
  del 0,42 %.
- **Evidencia:** la Ec. (4.6) evaluada en las condiciones enunciadas con la
  presión de la Ec. (4.11), contra el valor que imprime el folio 177; y los dos
  exponentes impresos comparados directamente. Verificado en las páginas 157,
  158, 165 y 207 (pp. 127, 128, 135 y 177 impresas) del PDF de la edición
  Springer de 2010.
- **Comportamiento de la biblioteca:** implementa la Ec. (4.6) con la presión
  absoluta que define su propia Ec. (4.4), porque una definición impresa manda
  sobre la cita redondeada de un valor derivado tres capítulos más allá. La
  discrepancia queda por debajo de toda tolerancia de esta biblioteca, así que
  nada depende de la elección; lo que sí dependía era de elegir bando en
  silencio. La Ec. (4.13) no se implementa
  ([`tests/fluids/test_water.py`](https://github.com/jmrplens/phonometry/blob/main/tests/fluids/test_water.py),
  comprobaciones de conformidad «Sea water (Ainslie 2010)»).
- **Estado:** sin comunicar.

## ISO 9053-2:2020, Anexo A.3 (dos propiedades del aire atribuidas a un documento que no las imprime)

- **Ubicación:** Anexo A.3, folio impreso 13 (página 17 del PDF) para los cuatro
  primeros valores y folio impreso 14 (página 18 del PDF) para el quinto.
- **Lo impreso:** «The following physical properties for air, valid at 23 °C,
  101,325 kPa and 50 % RH, are used for the calculation (values from
  IEC 61094-2:2009):», y a continuación $c_0 = 345{,}9$ m/s, $\rho_0 = 1{,}186$
  kg/m³, $\kappa = 1{,}400\,8$, $k_\mathrm{a} = 0{,}023\,55$ J/(s·m·K) y, a la
  vuelta, $C_\mathrm{P} = 938{,}7$ J/(kg·K).
- **El problema:** dos de los cinco no son valores de IEC 61094-2:2009. La
  Tabla F.1 de esa norma (folio impreso 40) tabula exactamente cinco magnitudes
  en ese estado: $\rho$, $c_0$, $\kappa$, $\eta$ y la **difusividad** térmica
  $\alpha_t = 2{,}115\,317 \times 10^{-5}$ m²/s. No tabula ni la conductividad
  térmica ni el calor específico; ésos aparecen en el Anexo F sólo como las dos
  expresiones de la cláusula F.6, que no imprimen valor. Los tres valores del
  Anexo A.3 que sí cuadran son precisamente las tres celdas de la Tabla F.1
  redondeadas a cuatro cifras ($345{,}866\,52 \to 345{,}9$;
  $1{,}186\,084\,8 \to 1{,}186$; $1{,}400\,757\,3 \to 1{,}400\,8$). Los dos
  que no cuadran son precisamente las dos magnitudes que la Tabla F.1 no imprime:
  evaluada en ese mismo estado, la cláusula F.6 da
  $k_\mathrm{a} = 0{,}025\,434\,1$ J/(s·m·K) y $C_\mathrm{P} = 1013{,}74$
  J/(kg·K), cada uno mayor que el par impreso por el mismo factor 1,0800.

  El factor común no es casualidad ni una diferencia de unidades. El par está
  anclado a la difusividad tabulada: $0{,}023\,55 / (1{,}186 \times
  2{,}115\,317 \times 10^{-5}) = 938{,}708\,5$, que imprime 938,7. O sea, uno
  de los dos vino de otro sitio y el otro se retrocalculó por la Fórmula (F.5)
  para que $\alpha_t$ siguiera saliendo. Cuál de los dos es el ajeno lo decide la
  termodinámica y no una preferencia: $C_\mathrm{P} = 938{,}7$ J/(kg·K) son
  27,19 J/(mol·K), por debajo del suelo del rotor rígido diatómico
  $(7/2)R = 29{,}10$ J/(mol·K), así que no es aire a ninguna temperatura, en
  ninguna unidad, ni por masa ni por mol, y la expresión del Anexo F para
  $C_\mathrm{P}$ no baja de unos 1013 J/(kg·K) en todo el intervalo de 200 K a
  400 K. La conductividad 0,023 55 J/(s·m·K), en cambio, sí es una conductividad
  real del aire: es la que da la expresión del Anexo F cerca de −1,4 °C, fuera
  del dominio de 15 °C a 27 °C que el propio Anexo F imprime.
- **Consecuencia para el ejemplo del anexo:** ninguna. La Fórmula (A.5) usa
  $k_\mathrm{a}$ y $C_\mathrm{P}$ sólo a través de la combinación
  $k_\mathrm{a}/(\rho_0 c_0 C_\mathrm{P})$, donde el factor común se cancela,
  así que los dos pares dan el $b = 1{,}83 \times 10^{-3}$ m y el
  $\kappa' = 1{,}370$ impresos. El defecto es invisible dentro del Anexo A.3 y
  sólo aparece al leer cualquiera de las dos constantes por separado, como
  documento al que se atribuye haberla publicado.
- **Evidencia:** las dos páginas impresas contra la Tabla F.1 (folio impreso 40)
  y la cláusula F.6 (folio impreso 39) de IEC 61094-2:2009; las expresiones de la
  cláusula F.6 evaluadas a 23 °C, 101 325 Pa y 50 % de humedad relativa, que
  reproducen el $\alpha_t$ impreso a $1{,}0 \times 10^{-7}$ relativo; el calor
  molar que implican los 938,7 J/(kg·K) contra el suelo diatómico.
  IEC 61094-2:2009 no es referencia normativa de ISO 9053-2:2020: aparece sólo
  como entrada [4] de la bibliografía. Verificado en la página 17 (p. 13 impresa)
  y la página 18 (p. 14 impresa) del PDF de ISO 9053-2:2020, y en la página 42
  (p. 40 impresa) y la página 41 (p. 39 impresa) del PDF de BS EN 61094-2:2009.
- **Comportamiento de la biblioteca:** las filas de conformidad que reproducen el
  Anexo A.3 pasan los cinco valores que el anexo imprime, así que reproducen la
  norma en vez de limitarse a coincidir con ella. Los valores por defecto que
  recibe quien llama son ese mismo estado del aire calculado desde el Anexo F de
  IEC 61094-2:2009, que es lo que el anexo dice estar usando; los dos caen sobre
  el $b$ y el $\kappa'$ impresos ([`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data),
  comprobaciones de conformidad «ISO 9053-2:2020 Annex A.3»).
- **Estado:** sin comunicar.

## EN 12354-1:2000 Fórmula (E.5) / ISO 12354-1:2017 E.3.4 (errata de la acotación de K24)

- **Ubicación:** EN 12354-1:2000, Anexo E, el bloque de uniones de pared con
  capas elásticas intermedias impreso bajo la Figura E.5 y numerado Fórmula
  (E.5) (p. 46 impresa), y la NOTA 4 de E.3.4 de ISO 12354-1:2017. El Anexo E
  de la edición de 2000 solo tiene dos cláusulas numeradas, E.1 «Determination
  methods» y E.2 «Empirical data», así que «E.5» es un número de fórmula, no
  de cláusula; una revisión anterior de esta entrada lo citaba como cláusula.
- **El impreso:** $K_{24} = 3{,}7 + 14{,}1 M + 5{,}7 M^{2}\ \text{dB}$;
  $0 \le K_{24} \le -4\ \text{dB}$ ; $0\ \text{dB / octave}$, es decir, la
  acotación del término de unión $K_{24}$ es un intervalo vacío; la edición de
  2017 repite la errata de 2000 al pie de la letra.
- **El problema:** el intervalo es imposible tal como está impreso; la figura
  que lo acompaña y la física (el término es una reducción acotada por abajo)
  indican $-4\ \text{dB} \le K_{24} \le 0\ \text{dB}$.
- **Evidencia:** la familia de curvas de la Figura E.5 en la misma página
  recorre la rama de $K_{24}$ desde 0 dB hasta cerca de −4 dB sobre las
  relaciones de masas dibujadas, que es el intervalo leído en el otro orden.
  Verificado en la página 48 del PDF (p. 46 impresa) de EN 12354-1:2000 y la
  página 52 del PDF (p. 46 impresa) de ISO 12354-1:2017.
- **Comportamiento de la biblioteca:** implementa la acotación como
  $-4 \le K_{24} \le 0$ con una nota de errata en el docstring.
- **Estado:** sin notificar.

## EN 12354-1:2000, Figura E.9 (E.7) (K24 expresado en la relación de masas del eje de la figura)

- **Ubicación:** Anexo E, Figura E.9 / Fórmula (E.7) (unión de pared ligera de
  doble hoja con elementos homogéneos), la línea de $K_{24}$.
- **El impreso:** $K_{24} = 3{,}0 - 14{,}1 M + 5{,}7 M^{2}\ \text{dB}$ (para
  $m_2/m_1 > 3$), bajo una figura cuyo eje x es $m_2/m_1$.
- **El problema:** el Anexo E define $M$ por trayectoria de transmisión como
  $M = \lg(m'_{\perp,i}/m'_i)$ (elemento perpendicular sobre el elemento que
  lleva la trayectoria). La trayectoria 2→4 de $K_{24}$ la lleva el elemento
  homogéneo ($m_2 = m_4$) con la hoja ($m_1$) perpendicular, así que el $M$
  por trayectoria es $\log_{10}(m_1/m_2)$, pero la línea impresa de $K_{24}$
  solo casa con la curva de su propia figura cuando $M$ se lee como la
  variable del eje x, $\log_{10}(m_2/m_1)$ (p. ej. −2,4 dB en $m_2/m_1 = 3$,
  −5,4 dB en 10). Leída con el $M$ que declara el anexo, la línea contradice
  la figura en $28{,}2 \cdot |\log_{10}(m_2/m_1)|\ \text{dB}$. La otra línea
  de $K_{24}$ de la misma edición (Figura E.5, Fórmula (E.5)) *sí* sigue el
  $M$ por trayectoria declarado, así que los dos impresos de $K_{24}$ de la
  edición de 2000 usan convenciones distintas en silencio. ISO 12354-1:2017
  E.3.5 imprime la relación de forma consistente en la convención por
  trayectoria de su Fórmula (E.3),
  $K_{24} = 3{,}0 + 14{,}1 M + 5{,}7 M^{2}$; las dos ediciones coinciden
  numéricamente (una revisión anterior de esta entrada leyó el impreso de 2017
  como una errata de signo; la re-deducción contra las figuras de ambas
  ediciones muestra que es un cambio de convención, no un defecto del texto de
  2017).
- **Evidencia:** evaluación numérica de ambas formas contra la curva de la
  Figura E.9. Verificado en la página 44 del PDF (p. 42 impresa), la página 48
  del PDF (p. 46 impresa) y la página 50 del PDF (p. 48 impresa) de
  EN 12354-1:2000, y en la página 53 del PDF (p. 47 impresa) de
  ISO 12354-1:2017, cuyo E.3.5 imprime su línea de K24 junto a una Figura E.7
  que no lleva ningún eje de relación de masas.
- **Comportamiento de la biblioteca:** implementa la convención por
  trayectoria de manera uniforme (`junction_vibration_reduction`,
  mass_ratio = $m'_{\perp,i}/m'_i$ para todas las ramas), así que la rama de
  doble hoja de E.7 toma relaciones hoja-sobre-homogéneo por debajo de 1/3 y
  evalúa $3{,}0 + 14{,}1 M + 5{,}7 M^2$.
- **Estado:** sin notificar.

## EN 12354-2:2000, Fórmula (3) frente al Anexo E.3 (nivel de impacto estandarizado)

- **Ubicación:** Fórmula (3) y ejemplo resuelto E.3.
- **El impreso:** la Fórmula (3) define
  $L'_{nT} = L'_n - 10 \lg(0{,}16 \cdot V/(A_0 \cdot T_0))$, que se reduce
  exactamente a $L'_n - 10 \lg(0{,}032 \cdot V)$, es decir, un volumen de
  referencia de $31{,}25\ \text{m}^3$. El Anexo E.3 declara «from equation
  (3): $L'_{nT,w} = L'_{n,w} - 10 \lg(V/30)$».
- **El problema:** el $V/30$ del anexo es un redondeo de la propia constante
  de la fórmula; las dos variantes difieren en 0,177 dB constantes.
- **Evidencia:** álgebra directa; ambas variantes recalculadas para el caso
  de E.3 (42,959 frente a 42,782 dB, ambas redondean a 43 en ese ejemplo).
  Verificado en la página 7 del PDF (p. 5 impresa) y la página 34 del PDF
  (p. 32 impresa) de EN 12354-2:2000.
- **Comportamiento de la biblioteca:** implementa la forma exacta
  $0{,}032 \cdot V$ y documenta el redondeo del anexo.
- **Estado:** sin notificar.

## EN 12354-3:2000, Fórmula (5) (forma reducida de la diferencia de niveles normalizada)

- **Ubicación:** apartado 3.1.5 «Relations between quantities», Fórmula (5)
  (p. 6 impresa).
- **El impreso:**
  $D_{2m,n} = D_{2m,nT} - 10 \lg[0{,}16\,V/(T_0 A_0)] = D_{2m,nT} - 10 \lg 0{,}32\,V\ \text{dB}$.
- **El problema:** la forma reducida está mal por un factor de diez. Seis
  líneas más arriba, la lista de símbolos del apartado 3.1.4 define $A_0$
  como «the reference equivalent sound absorption area, in square metres, for
  dwellings given as 10 m²», y la lista de símbolos del apartado 3.1.3 de la
  página anterior define $T_0$ como «the reference reverberation time, in
  seconds, for dwellings given as 0,5 s». Así que
  $0{,}16/(T_0 A_0) = 0{,}16/5 = 0{,}032$, no 0,32. Aplicada como está
  impresa, la forma reducida desplaza toda diferencia de niveles de fachada
  normalizada en exactamente $10\log_{10} 10 = 10\ \text{dB}$. El análogo
  exacto de la parte compañera, la Fórmula (3) de EN 12354-2:2000, imprime la
  misma álgebra correctamente:
  $L'_{nT} = L'_n - 10 \lg[0{,}16\,V/(A_0 T_0)] = L'_n - 10 \lg 0{,}032\,V\ \text{dB}$.
  ISO 12354-3:2017 retiró la forma reducida por completo: su Fórmula (5)
  imprime solo $D_{2m,n} = D_{2m,nT} - 10 \lg[C_\text{sab} V/(A_0 T_0)]$ con
  $C_\text{sab} = 0{,}16\ \text{s/m}$.
- **Evidencia:** álgebra directa con los propios $A_0$ y $T_0$ de la norma, y
  la comparación lado a lado con la Fórmula (3) de la Parte 2, reducida
  correctamente. Verificado en la página 8 del PDF (p. 6 impresa) y la página
  7 del PDF (p. 5 impresa) de EN 12354-3:2000, en la página 7 del PDF (p. 5
  impresa) de EN 12354-2:2000 para su Fórmula (3), y en la página 12 del PDF
  (p. 6 impresa) de ISO 12354-3:2017 para las Fórmulas (4) y (5) de 2017.
- **Comportamiento de la biblioteca:** no le afecta. Ningún camino de código
  implementa la forma reducida: el modelo de fachada calcula $D_{2m,nT}$
  desde la Fórmula (13)
  ([`facade.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/prediction/facade.py)), y el
  método de inspección convierte con la forma sin reducir
  $D_{2m,n} = D_{2m} + k + 10\log_{10}[A_0 T_0/(0{,}16 V)]$ del apartado 3.15
  de ISO 10052
  ([`survey_insulation.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/measurement/survey_insulation.py)).
  Las dos constantes de estandarización que *sí* están pre-plegadas en otros
  puntos de la biblioteca son ambas correctas: $0{,}032$ para la forma de
  impacto de la Parte 2 y $0{,}32$ para la forma aérea de la Parte 1,
  $D_{nT} = R' + 10\log_{10}(0{,}16 V/(T_0 S_s))$, cuyo denominador es un
  área y no $A_0$.
- **Estado:** sin notificar.

## EN 12354-3:2000, Fórmula (13) frente a su propio ejemplo del Anexo F (la constante «6»)

- **Ubicación:** apartado 4.1, Fórmula (13) (p. 9 impresa), contra el ejemplo
  resuelto del Anexo F (pp. 27-28 impresas).
- **El impreso:** la Fórmula (13) da
  $D_{2m,nT} = R' + \Delta L_\text{fs} + 10 \lg[V/(6 T_0 S)]\ \text{dB}$,
  mientras que la tabla de resultados de F.1.3 imprime una fila de
  $D_{2m,nT}$ que es exactamente $R' + 1{,}5\ \text{dB}$ en las cinco bandas
  de octava y en la columna de valor único (25,9/23,0/26,4/36,9/39,0 contra
  24,4/21,5/24,9/35,4/37,5, y 29,3 contra 27,8).
- **El problema:** en esta constante el *ejemplo* es autoconsistente y la
  *fórmula* es la discrepante. (Dos celdas de la misma tabla del anexo no se
  siguen de sus filas de elementos, que es el asunto de la entrada siguiente;
  la fila impresa de $+1,5$ dB se cumple en todas las bandas de todos modos,
  así que los dos defectos son independientes.) Con los propios datos del
  ejemplo ($V = 50\ \text{m}^3$, $S = 11{,}3\ \text{m}^2$,
  $T_0 = 0{,}5\ \text{s}$, $\Delta L_\text{fs} = 0$), la forma de Sabine da
  $10\log_{10}[0{,}16 \cdot 50/(0{,}5 \cdot 11{,}3)] = 1{,}5104\ \text{dB}$,
  que es la fila impresa de +1,5 dB; la Fórmula (13) tal como está impresa da
  $10\log_{10}[50/(6 \cdot 0{,}5 \cdot 11{,}3)] = 1{,}6877\ \text{dB}$. La
  brecha es la constante: el «6» de la Fórmula (13) es un
  $1/0{,}16 = 6{,}25$ redondeado, y
  $10\log_{10}(6{,}25/6) = 0{,}177\ \text{dB}$ es exactamente la
  discrepancia. ISO 12354-3:2017 lo sustituyó por una constante de Sabine
  explícita, imprimiendo la Fórmula (4) como
  $D_{2m,nT} = R' + \Delta L_\text{fs} + [10 \lg(C_\text{sab} V/(T_0 S))]$
  con $C_\text{sab} = 0{,}16\ \text{s/m}$, que es la constante que el ejemplo
  de 2000 ya usaba. Una revisión anterior de esta entrada atribuía la fila de
  1,5 dB al ejemplo; la atribución es al revés.
- **Evidencia:** evaluación de ambas constantes contra las filas impresas del
  Anexo F, que concuerdan con 0,16 dentro de los 0,05 dB que lleva la tabla y
  discrepan del 6 redondeado en 0,18 dB uniformes; y la refundición de 2017,
  que adopta la constante del ejemplo. El resultado de valor único del
  ejemplo, $D_{2m,nT,w} = 33\ \text{dB}$, es insensible a la diferencia y se
  reproduce en cualquier caso. Verificado en las páginas 11 (p. 9 impresa),
  29 (p. 27 impresa) y 30 (p. 28 impresa) del PDF de EN 12354-3:2000, y en la
  página 12 del PDF (p. 6 impresa) de ISO 12354-3:2017.
- **Comportamiento de la biblioteca:** implementa la Fórmula (13) tal como
  está impresa, con el 6 redondeado; los datos de test dejan constancia de
  que las filas del Anexo F siguen la constante exacta 0,16 y quedan 0,18 dB
  por debajo del modelo.
- **Estado:** sin notificar.

## EN 12354-3:2000, Anexo F.1.3 (las celdas de R' de 1 kHz y 2 kHz)

- **Ubicación:** Anexo F, tabla F.1.3 «Results for façade» (p. 28 impresa),
  la fila `R' (equation 10)`.
- **El impreso:** $R'$ = 24,4 / 21,5 / 24,9 / 35,4 / 37,5 dB a 125 / 250 /
  500 / 1000 / 2000 Hz.
- **El problema:** las dos últimas celdas no se siguen de las propias filas
  de elementos de la tabla. La Fórmula (10),
  $R' = -10\log_{10} \sum \tau_{e,i}$, aplicada a las cuatro columnas de
  $-10\log_{10} \tau_e$ impresas justo encima, da 24,41 / 21,50 / 24,86 /
  **35,78** / **37,99** dB. Las tres primeras celdas se reproducen dentro de
  los 0,05 dB que lleva la tabla; las celdas de 1 kHz y 2 kHz están impresas
  0,4 dB y 0,5 dB por debajo.
- **Evidencia:** suma energética de las filas de elementos impresas banda a
  banda (1 kHz: 60,7 / 40,0 / 46,6 / 38,5 dB; 2 kHz: 66,7 / 41,0 / 43,6 /
  44,5 dB). La fila de $D_{2m,nT}$ de debajo es un $R' + 1{,}5\ \text{dB}$
  uniforme en todas las bandas, incluidas esas dos, así que hereda el mismo
  desplazamiento, y el resultado de valor único $D_{2m,nT,w} = 33\ \text{dB}$
  es insensible a él y sigue reproduciéndose. Verificado en la página 30 del
  PDF (p. 28 impresa) de EN 12354-3:2000.
- **Comportamiento de la biblioteca:** los datos de test anotan la
  inconsistencia junto al ancla afectada.
- **Estado:** sin notificar.

## EN 12354-5:2009, Tabla F.1 y apartado F.4.2 (fuerza de referencia impresa como 1 pN)

- **Ubicación:** Anexo F, apartado F.4.2: la lista de símbolos de la Fórmula
  (F.9), la frase que introduce la forma cerrada y el pie de la Tabla F.1
  (p. 59 impresa).
- **El impreso:** «$L_F$ is the force level in the source room, in dB re
  1 pN»; «$L_F = 10\lg 2{,}5f/10^{-12}$ dB re 1 pN or
  $L_F = 10\lg 0{,}8f/10^{-12}$ dB re 1 pN for one-third octave bands»; y
  «Table F.1 – Force level $L_F$ re 1 pN for the ISO tapping machine in
  octave bands», cuyas ocho celdas leen 139, 142, 145, 148, 151, 154, 156 y
  156 dB.
- **El problema:** la fuerza de referencia de esos niveles es $10^{-6}$ N, no
  1 pN. Tres lecturas independientes concuerdan, y ninguna es compatible con
  la referencia impresa. **(a) La propia álgebra del anexo.** Un nivel de
  potencia re 1 pW construido desde un nivel de fuerza y una movilidad es
  $L_W = L_F + 10\lg(F_0^2 Y / W_0)$. La Fórmula (D.5a) imprime
  $L_{Ws,c} = L_{F,eq} + 10\lg Y_s$ y la Fórmula (D.9a) imprime
  $L_{Ws,c} = L_F - 5 - 10\lg f$, que es la misma expresión evaluada en la
  movilidad de fuente de tipo masa $Y_s = (2\pi f M)^{-1}$ de un martillo de
  máquina de impactos de 0,5 kg. Ninguna lleva término alguno de
  $F_0^2/W_0$, así que ambas solo cuadran cuando
  $F_0^2 / W_0 = 1\ \text{s}^{-1}$, es decir $F_0 = 10^{-6}$ N; leídas re
  1 pN, cada una se quedaría 120 dB corta respecto al nivel que define. La
  contrapartida en velocidad, la Fórmula (D.10a), sí imprime su término de
  referencia $10\lg(v_\text{ref}^2/W_\text{ref})$ y declara que el resultado
  cancela el $10\lg Z_s$ exactamente, cosa que hace con los $10^{-9}$ m/s que
  la propia norma da como referencia del nivel de velocidad en el apartado
  F.4.2. El anexo es por tanto explícito y correcto con la referencia de
  velocidad y calla la de fuerza. **(b) La máquina que produce la tabla.** La
  máquina de impactos ISO deja caer martillos de 0,5 kg desde 40 mm a diez
  impactos por segundo, así que cada impacto transfiere un momento de
  0,443 N·s y la fuerza es un tren de impulsos de 10 Hz cada uno de cuyos
  armónicos lleva 6,26 N r.m.s. Sumando los armónicos que caen dentro de cada
  banda de octava salen 139,4 / 142,4 / 145,4 / 148,4 / 151,4 / 154,4 dB re
  $10^{-6}$ N de 31,5 Hz a 1 kHz, reproduciendo las seis primeras celdas de
  la Tabla F.1 con margen de 0,5 dB; las celdas de 2 kHz y 4 kHz quedan por
  debajo de esa recta, que es la caída que la propia norma señala con «up
  till about 1000 Hz». Re 1 pN, las mismas celdas describirían fuerzas de
  decenas de micronewtons, que ninguna máquina de impactos produce. **(c) La
  norma compañera.** La Fórmula (15) de EN 15657:2018, que es de donde salen
  en primer lugar los datos de fuente estructural del Anexo D, escribe la
  misma conversión de fuerza a potencia «in dB re $F_0 = 10^{-6}$ N», y
  $10^{-6}$ N es la fuerza de referencia preferida de ISO 1683.
- **Evidencia:** verificado en las páginas 61 y 62 del PDF (pp. 59 y 60
  impresas) de BS EN 12354-5:2009, que llevan el apartado F.4.2 con la lista
  de símbolos de la Fórmula (F.9), la forma cerrada, la Tabla F.1 completa y
  la lista de símbolos de la Fórmula (F.11) con su referencia de velocidad de
  $10^{-9}$ m/s; y en las páginas 45, 48 y 50 del PDF (pp. 43, 46 y 48
  impresas) de la misma edición, que llevan las Fórmulas (D.5a), (D.9a) y
  (D.10a).
- **Comportamiento de la biblioteca:** publica las celdas impresas sin
  cambios y las documenta re $10^{-6}$ N. `tapping_machine_force_level`
  devuelve los ocho valores de la Tabla F.1,
  `tapping_machine_force_level_estimate` la forma cerrada y
  `tapping_machine_characteristic_power_level` la Fórmula (D.9a) tal como
  está impresa;
  `test_table_f1_is_referred_to_1e_6_newton_not_1_piconewton` fija la lectura
  contra la mecánica de la máquina.
- **Estado:** sin notificar.

## EN 12354-5:2009, leyenda de la Figura D.3 (tres curvas bajo un mismo símbolo)

- **Ubicación:** Anexo D, la leyenda de la Figura D.3 (p. 47 impresa).
- **El impreso:** tres filas de leyenda, cada una etiquetada con el mismo
  símbolo: $L_{Ws,c,A} = 124\ \text{dB}$, $L_{Ws,c,A} = 119\ \text{dB}$ y
  $L_{Ws,c,A} = 102\ \text{dB}$.
- **El problema:** el propio pie de la figura lee «Structure-borne sound
  power for the ISO-tapping machine: characteristic source power, installed
  power on a wooden floor and installed power on a concrete floor; the
  A-weighted power level is also indicated». Solo la primera curva es una
  potencia característica; las otras dos son potencias instaladas y sus
  totales ponderados A son $L_{Ws,\text{inst},A}$. Las curvas dibujadas
  zanjan la asignación: la primera es plana en torno a 114,5 dB re 1 pW, que
  es el resultado independiente de la frecuencia de la Fórmula (D.9a) para la
  máquina de impactos, mientras que las otras dos crecen con la frecuencia y
  quedan por debajo, la del suelo de hormigón la más baja, como exige
  $L_{Ws,c} - D_{C,i}$.
- **Evidencia:** verificado en la página 49 del PDF (p. 47 impresa) de
  BS EN 12354-5:2009, la página que lleva la Figura D.3 con su leyenda y su
  pie.
- **Comportamiento de la biblioteca:** no hizo falta ninguno; ningún valor se
  lee de la Figura D.3.
  `test_formula_d9a_is_flat_at_about_115_db_per_third_octave` fija la curva
  característica plana a la que pertenece la primera fila de la leyenda.
- **Estado:** sin notificar.

## ISO 12354-1:2017 Tabla L.3 / ISO 12354-2:2017 Tabla G.3 (sumas de perímetro)

- **Ubicación:** el bloque de datos de entrada bajo la Tabla L.3 (p. 81
  impresa) y el bloque idéntico bajo la Tabla G.3 (p. 38 impresa), que lista
  la suma de absorción perimetral $\sum l_k \alpha_k$ de la Fórmula (C.1)
  para el ejemplo resuelto.
- **El impreso:** un valor por *tipo* de elemento: suelo separador 2,364 m
  ($S = 20\ \text{m}^2$), pared exterior 2,375 m ($S = 11\ \text{m}^2$),
  pared interior 1,840 m ($S = 13{,}75\ \text{m}^2$).
- **El problema:** la Fórmula (C.1) necesita una suma por *elemento*, y el
  ejemplo tiene cinco elementos con tres áreas distintas. Solo dos de los
  tres valores impresos reproducen las columnas que se supone que gobiernan:
  2,375 m con $S = 11\ \text{m}^2$ da la pared exterior 1 exacta, y 1,840 m
  con $S = 13{,}75\ \text{m}^2$ da la pared interior **2** exacta. Los
  2,364 m impresos del suelo separador no reproducen su propia columna en
  ninguna banda (0,074 9 contra los 0,083 1 impresos a 50 Hz, 0,026 4 contra
  0,029 0 a 500 Hz); 2,659 m sí, en todas las bandas. Los dos elementos sin
  valor impreso necesitan 2,548 m (pared exterior 2,
  $S = 13{,}75\ \text{m}^2$) y 1,636 m (pared interior 1,
  $S = 11\ \text{m}^2$).
- **Evidencia:** las cinco sumas re-deducidas desde la Fórmula (C.4),
  $\alpha_k = \sum_j \sqrt{f_{c,j}/f_\text{ref}}\ 10^{-K_{ij}/10}$, sobre la
  propia geometría de uniones del ejemplo con los índices del Anexo E sin
  redondear: 2,659 / 2,375 / 2,548 / 1,636 / 1,839 m. La deducción devuelve
  los dos valores impresos que son autoconsistentes con sus propias columnas
  (2,375 m, y 1,839 m contra los 1,840 m impresos) y aporta los tres que
  faltan o están mal, y entonces todas las columnas de
  $\eta_\text{tot,situ}$ de la Tabla L.3 / G.3 se reproducen a
  $5 \cdot 10^{-5}$. Los valores impresos aplicados al elemento equivocado
  del mismo tipo fallan por mucho más que ese redondeo: 2,375 m en la pared
  exterior 2 da 0,108 5 contra los 0,114 9 impresos a 50 Hz, y 1,840 m en la
  pared interior 1 da 0,085 0 contra 0,077 0.
- **Comportamiento de la biblioteca:** `in_situ_total_loss_factor` toma
  $\sum l_k \alpha_k$ como entrada y `perimeter_absorption_coefficient`
  implementa la Fórmula (C.4); la fixture del Anexo L deduce las cinco sumas
  por esa vía en lugar de usar el bloque impreso, y lo dice
  ([`tests/building/prediction/test_detailed_model.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_detailed_model.py)).
- **Estado:** sin notificar.

## ISO 12354-1:2017 Tabla L.3 / ISO 12354-2:2017 Tabla G.3 (ηint de la pared exterior)

- **Ubicación:** el mismo bloque de datos de entrada, línea de la pared
  exterior.
- **El impreso:** $\eta_\text{int} = 0{,}013$ para las paredes exteriores de
  hormigón celular curado en autoclave de 365 mm.
- **El problema:** la propia especificación de elementos del ejemplo, y la
  Tabla B.3 del Anexo B para el hormigón celular curado en autoclave, dan
  0,012 5. Solo 0,012 5 reproduce el $\eta_\text{tot,situ}$ tabulado: a
  500 Hz la Fórmula (C.1) da
  $0{,}012\,5 + 0{,}001\,41 + 0{,}034\,57 = 0{,}048\,5$, el valor impreso,
  donde 0,013 daría 0,049 0.
- **Evidencia:** recálculo término a término de la Fórmula (C.1) para ambas
  paredes exteriores en todas las bandas con cada $\eta_\text{int}$
  candidato.
- **Comportamiento de la biblioteca:** la fixture del Anexo L usa 0,012 5.
- **Estado:** sin notificar.

## ISO 12354-1:2017, Tabla L.4 (segundo bloque de trayectoria etiquetado 2d)

- **Ubicación:** Anexo L, Tabla L.4 (p. 82 impresa), el bloque derecho
  encabezado «Transmission path 2d».
- **El impreso:** el bloque da $\alpha_{i,\text{situ}}$ = 6,3 a 14,1,
  $D_{v,ij,\text{situ}}$ = 11,0 a 13,6 y $R_{ij}$ = 43,9 a 84,6 dB.
- **El problema:** esos son los números de la trayectoria **4d** (pared
  interior 2 al suelo separador), no de la 2d (pared exterior 2). La Tabla
  L.1 del mismo anexo imprime la columna $R_{4d}$ entera, de 43,9 a 84,6 dB,
  y la columna $R_{ij}$ del bloque es esa columna celda a celda. Lo que lo
  zanja banda a banda son las otras dos columnas, que no admiten confusión:
  la pared exterior 2 tiene $\alpha_{i,\text{situ}} = 10{,}3\ \text{m}$ a
  50 Hz ($S = 13{,}75\ \text{m}^2$, $\eta_\text{tot} = 0{,}114\,9$) mientras
  que la pared interior 2 tiene 6,3 m ($\eta_\text{tot} = 0{,}070\,3$), el
  valor impreso; y $D_{v,ij,\text{situ}}$ sigue el $K_{ij}$ de suelo a pared
  interior de 8,8 dB, que da 11,0 a 13,6 dB, no el de suelo a pared exterior
  de 6,4 dB, que da 9,6 a 11,9 dB.
- **Evidencia:** recálculo independiente de las Fórmulas (10), (11) y (15)
  para ambas trayectorias candidatas en todas las bandas. La trayectoria 4d
  reproduce las tres columnas del bloque, $\alpha_{i,\text{situ}}$ a 0,05 m y
  $D_{v,ij,\text{situ}}$ y $R_{ij}$ a 0,05 dB, que es la resolución impresa.
  La trayectoria 2d se aparta de la columna $R_{ij}$ del bloque entre 0,1 dB
  y 7,0 dB según la banda, y se acerca más entre 100 Hz y 160 Hz (0,5 / 0,5 /
  0,1 dB), así que $R_{ij}$ por sí sola no identifica la trayectoria en esas
  bandas; $\alpha_{i,\text{situ}}$ (10,3 contra 6,3 m a 50 Hz) y
  $D_{v,ij,\text{situ}}$ (de 1,4 dB a 1,7 dB de separación en todas las
  bandas) sí.
- **Comportamiento de la biblioteca:** el test que afirma el bloque lo
  construye como trayectoria 4d y nombra el etiquetado erróneo.
- **Estado:** sin notificar.

## ISO 12354-1:2017, Tabla L.1 (índices globales no enteros)

- **Ubicación:** Anexo L, Tabla L.1 (p. 79 impresa), la fila de $R_w$ y la
  frase que la sigue, y la fila correspondiente de $L_{n,w}$ de la Tabla G.1
  de ISO 12354-2:2017.
- **El impreso:** la fila de $R_w$ da un decimal para cada trayectoria
  (75,1 / 84,5 / 70,6 / … y 57,8 en la columna del total) mientras que la
  frase inmediatamente debajo declara
  $R'_w\,(C\,;\,C_\text{tr}) = 57{,}9\ (-2\,;\,-8)\ \text{dB}$.
- **El problema:** ISO 717-1 valora desplazando la curva de referencia **en
  pasos de 1 dB**, así que un índice global es un entero; los valores
  impresos con un decimal son la curva de referencia desplazada *de forma
  continua* hasta que la suma de desviaciones desfavorables vale exactamente
  32,0 dB. La fila aérea de $R_w$ de la Tabla L.1 *trunca* ese valor continuo
  a un decimal mientras que la frase de debajo redondea, y por eso la misma
  magnitud aparece dos veces como 57,8 y 57,9; la fila de impacto de
  $L_{n,w}$ de la Tabla G.1 redondea en cambio (29,58 se imprime 29,6 y 40,98
  como 41,0), así que el truncamiento es una propiedad solo de la fila aérea.
  Los términos de adaptación espectral heredan el desplazamiento: con el
  índice ISO 717-1 de 57 dB son $C = -1$ y $C_\text{tr} = -7$, y el
  (−2 ; −8) impreso es exactamente el par desplazado por los mismos 0,86 dB.
- **Evidencia:** una resolución por desplazamiento continuo de la curva de
  referencia de ISO 717-1 contra los espectros por banda impresos reproduce
  todos los valores impresos de ambas filas ($R_{Dd}$ 75,12 contra 75,1;
  $R_{D1}$ 84,54 contra 84,5; $R_{11}$ 70,66 contra 70,6; el total 57,86
  contra 57,8 / 57,9; en el lado de impacto $L_{n,Df1}$ 29,58 contra 29,6 y
  el total 40,98 contra 41,0), mientras que los índices ISO 717-1 en pasos de
  1 dB de los mismos espectros son 75, 84, 70 y 57 dB. Verificado en la
  página 85 del PDF (p. 79 impresa) de ISO 12354-1:2017.
- **Comportamiento de la biblioteca:** `weighted_rating` /
  `weighted_impact_rating` implementan ISO 717-1/-2 tal como están escritas,
  así que el modelo detallado devuelve $R'_w = 57\ \text{dB}$ y
  $L'_{n,w} = 41\ \text{dB}$ ($C_I = 2$) para el ejemplo; el test fija esos
  valores y documenta los impresos.
- **Estado:** sin notificar.

## ISO 12354-2:2017, Tabla G.1 (columnas de flancos de 50 Hz a 80 Hz)

- **Ubicación:** Anexo G, Tabla G.1 (p. 36 impresa), las cuatro columnas de
  $L_{n,Df}$, filas de 50 Hz, 63 Hz y 80 Hz.
- **El impreso:** $L_{n,Df1}$ = 47,3 / 44,9 / 46,2 dB.
- **El problema:** la Tabla G.4 del mismo anexo imprime la misma trayectoria
  Df para la pared exterior 1, desde las mismas entradas, como 47,8 / 45,9 /
  47,0 dB. Las dos tablas no pueden estar bien a la vez, y de 100 Hz hacia
  arriba coinciden exactamente.
- **Evidencia:** la Fórmula (12) evaluada desde las propias columnas de la
  Tabla G.3 del anexo ($L_{n,\text{situ}}$, $R_\text{situ}$) y las columnas
  $D_{v,ij,\text{situ}}$ y $\Delta L_\text{situ}$ de la Tabla G.4 da 47,80 /
  45,85 / 46,95 dB, reproduciendo los 47,8 / 45,9 / 47,0 impresos de la Tabla
  G.4 a 0,05 dB y la Tabla G.1 solo de 100 Hz hacia arriba. Llevando el mismo
  recálculo por toda la cadena, la pared exterior 2 queda baja entre 0,5 dB y
  1,0 dB en las mismas tres bandas y las dos paredes interiores bajas hasta
  0,5 dB a 50 Hz y 63 Hz (sus celdas de 80 Hz coinciden). De 100 Hz hacia
  arriba ninguna columna de flancos se desvía más de 0,15 dB. Corregir las
  celdas afectadas sube el total impreso $L'_n$ solo ligeramente: de 58,6 a
  58,7 dB a 50 Hz, de 57,0 a 57,2 dB a 63 Hz, de 55,9 a 56,1 dB a 80 Hz.
- **Comportamiento de la biblioteca:** el test afirma la Tabla G.4 completa,
  la columna directa de la Tabla G.1 en todo el rango, y las columnas de
  flancos de la Tabla G.1 de 100 Hz hacia arriba, nombrando la discrepancia.
- **Estado:** sin notificar.

## ISO 12354-2:2017, Tabla G.8 (Kij de unión y m'i)

- **Ubicación:** Anexo G, Tabla G.8 (p. 40 impresa), la unión rígida en T de
  pared interior con pared exterior.
- **El impreso:** la fila «Int. wall 1/2 - Ext. wall 1/2» da
  $K_{ij} = 6{,}6\ \text{dB}$; la fila de debajo, «Ext. wall 1/2 - Ext. wall
  1/2», da $m'_i = 2{,}19\ \text{kg/m}^2$.
- **El problema:** dos erratas independientes. La rama de esquina de la T
  rígida $K_{12} = 5{,}7 + 5{,}7 M^2$ con
  $M = \log_{10}(360/219) = 0{,}215\,6$ da 5,97, es decir **6,0**, y la Tabla
  L.8 de ISO 12354-1:2017 imprime 6,0 para la unión idéntica del ejemplo
  idéntico. Y la masa por unidad de área de la pared exterior es
  $219{,}0\ \text{kg/m}^2$ en todo el ejemplo, no 2,19 (un factor 100).
- **Evidencia:** evaluación de la rama de esquina según el Anexo E; las demás
  filas de la misma tabla y todo el Anexo L de ISO 12354-1 usan
  $219{,}0\ \text{kg/m}^2$. Verificado en la página 46 del PDF (p. 40
  impresa) de ISO 12354-2:2017, cuyas columnas de masa de la Tabla G.8 se
  encabezan `m'i` y `m'orthogonal`, y la página 89 del PDF (p. 83 impresa) de
  ISO 12354-1:2017.
- **Comportamiento de la biblioteca:** usa 6,0 dB y
  $219{,}0\ \text{kg/m}^2$.
- **Estado:** sin notificar.

## ISO 12354-2:2017, Tabla G.6 (fila mal etiquetada)

- **Ubicación:** Anexo G, Tabla G.6 (p. 40 impresa), unión rígida en cruz de
  pared interior con suelo separador.
- **El impreso:** una fila etiquetada «Ext. wall 1/2 – Int. wall 1/2» con
  `m'i` = 360,0, `m'orthogonal` = 484,0 y $K_{ij} = 11{,}0\ \text{dB}$.
- **El problema:** la Tabla G.6 describe la unión en cruz de *pared interior
  con suelo separador*; ninguna pared exterior concurre en ella. Las masas y
  el valor son los de la trayectoria en línea de la pared interior, y la
  Tabla L.6 de ISO 12354-1:2017 imprime la misma fila correctamente como
  «Int. wall 1/2 - Int. wall 1/2».
- **Evidencia:** la rama pasante de la cruz rígida
  $8{,}7 + 17{,}1 M + 5{,}7 M^2$ con $M = \log_{10}(484/360)$ da 10,99, el
  11,0 impreso, para la pared interior. Verificado en la página 46 del PDF
  (p. 40 impresa) de ISO 12354-2:2017 y la página 89 del PDF (p. 83 impresa)
  de ISO 12354-1:2017.
- **Comportamiento de la biblioteca:** trata la fila como la trayectoria en
  línea de la pared interior.
- **Estado:** sin notificar.

## ISO 12354-1:2017 Tabla L.10 / ISO 12354-2:2017 Tabla G.10 (etiqueta de elemento)

- **Ubicación:** la tabla de datos de entrada del modelo simplificado de
  ambas partes, cuarta fila: Tabla L.10 (p. 84 impresa) y Tabla G.10 (p. 41
  impresa).
- **El impreso:** ISO 12354-1 imprime «Internal wall 4 (F = f = 4)»;
  ISO 12354-2 imprime «Internal wall 4 (f4)»: las dos partes etiquetan la
  fila de forma distinta, y una revisión anterior de esta entrada citaba la
  forma de la Parte 1 para ambas.
- **El problema:** el ejemplo tiene dos paredes interiores; el elemento de
  índice $F = f = 4$ es la pared interior **2**
  ($5{,}00\ \text{m} \times 2{,}75\ \text{m}$, $S = 13{,}75\ \text{m}^2$),
  como la etiquetan las tablas del modelo detallado de los mismos anexos.
- **Evidencia:** los propios $S = 13{,}75\ \text{m}^2$ y
  $l_{ij} = 5{,}0\ \text{m}$ de la fila casan con la pared interior 2 de la
  Tabla L.1 / G.1. Verificado en la página 90 del PDF (p. 84 impresa) de
  ISO 12354-1:2017 y en la página 47 del PDF (p. 41 impresa) de
  ISO 12354-2:2017, con las etiquetas de columna del modelo detallado leídas
  en la página 85 del PDF (p. 79 impresa) de ISO 12354-1:2017 y en la página
  42 del PDF (p. 36 impresa) de ISO 12354-2:2017.
- **Comportamiento de la biblioteca:** no hizo falta ninguno; los números no
  se ven afectados.
- **Estado:** sin notificar.

## ISO 12354-1:2017, Tabla D.1 (1 600 Hz cubierto por dos filas)

- **Ubicación:** Anexo D, Tabla D.1 (p. 39 impresa), en la que se lee la
  mejora del índice global de reducción sonora de un revestimiento interior a
  partir de su frecuencia de resonancia.
- **El impreso:** las dos últimas filas son «630 to 1 600 -> -10» y «1 600 <=
  f0 <= 5 000 -> -5».
- **El problema:** 1 600 Hz pertenece a ambas filas, con valores distintos, y
  el apartado D.2.2 exige que $f_0$ sea «rounded to the centre frequency of
  the one-third-octave band in which fo falls», así que 1 600 Hz es un valor
  en el que la tabla se lee de verdad y no un borde inalcanzable. Como el
  redondeo es obligatorio, la ambigüedad no es un punto único: toda
  frecuencia de resonancia bruta de la banda de 1 600 Hz, es decir de
  1 412,5 Hz a 1 778,3 Hz (bordes de banda de ISO 266), cae en él. Todos los
  demás límites de la tabla son centros de banda distintos (200, 250, 315,
  400, 500 Hz), y ningún otro par de filas se solapa.
- **Evidencia:** la propia tabla impresa, en la página 45 del PDF (p. 39
  impresa) de ISO 12354-1:2017: las dos filas van regladas por separado y
  comparten el extremo al pie de la letra, «630 to 1 600» y «1 600 <= f0 <=
  5 000». Ninguna de las dos filas puede descartarse, porque de 630 Hz a
  1 250 Hz no hay otra entrada y de 2 000 Hz a 5 000 Hz tampoco. La edición
  predecesora da la lectura anterior, sin ambigüedad: la Tabla D.3 de
  EN 12354-1:2000, verificada en la página 43 del PDF (p. 41 impresa) de esa
  edición, imprime el mismo par de filas como «630 - 1 600 -> -10» y
  «> 1 600 -> -5», estrictamente mayor, así que en 2000 exactamente 1 600 Hz
  tomaba -10 dB sin nada que decidir. La reescritura de 2017 sustituyó
  «> 1 600» por «1 600 <= f0 <= 5 000» dejando «630 to 1 600» intacto, que es
  lo que crea el solape; qué pretendía la reescritura en el extremo
  compartido, el texto no lo dice.
- **Comportamiento de la biblioteca:** `weighted_lining_improvement` devuelve
  los -10 dB más conservadores exactamente a 1 600 Hz y -5 dB por encima, la
  lectura de 2000, con la ambigüedad nombrada en el docstring y fijada en
  [`tests/building/prediction/test_resilient_layers.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_resilient_layers.py).
- **Estado:** sin notificar.

- **Relacionado, no una errata:** la NOTA 1 de la misma tabla pone un suelo
  de 0 dB a la rama de 30 Hz a 160 Hz,
  $74{,}4 - 20\log_{10}(f_0) - R_w/2$. Dentro de la caja de validez que el
  apartado D.2.2 declara para la tabla
  ($30\ \text{Hz} \le f_0 \le 160\ \text{Hz}$,
  $20\ \text{dB} \le R_w \le 60\ \text{dB}$) la rama nunca lo alcanza: su
  mínimo es $74{,}4 - 20\log_{10}(160) - 60/2 = 0{,}32\ \text{dB}$. El suelo
  es por tanto inactivo para toda entrada declarada de la tabla, pero no
  siempre lo fue: la edición de 2000 tabulaba la rama baja como cuatro filas
  discretas que terminaban en «160 -> 28 - Rw/2», cuyo mínimo es
  $28 - 60/2 = -2\ \text{dB}$, así que la NOTA 1 operaba allí. El ajuste
  continuo de 2017 queda 2,3 dB por encima en esa esquina y dejó la nota
  vestigial. La biblioteca conserva el suelo porque la nota sigue impresa.

## ISO 15186-1, apartado 3.9, Fórmula (8) (signo del término 10 lg N)

- **Ubicación:** apartado 3.9, Fórmula (8) (p. 3 impresa), la diferencia de
  niveles normalizada de elemento por intensidad para N elementos pequeños de
  edificación medidos juntos. El impreso leído aquí es
  **BS EN ISO 15186-1:2003**, la adopción británica de texto idéntico; la
  entrada llevaba antes el encabezado «:2000», el año de la edición ISO que
  citan los docstrings de la biblioteca, que no es la copia que se leyó.
- **El impreso:**
  $D_{I,n,e} = L_{p1} - 6 - (L_{In} + 10 \lg(S_m/A_0) + 10 \lg(N))$, es
  decir, el término $10 \lg N$ se resta.
- **El problema:** el signo restado no puede deducirse. Medir $N$ unidades
  idénticas dentro de una superficie de medición eleva la potencia
  transmitida (y con ella $L_{In} + 10\log_{10} S_m$) en $10\log_{10} N$, así
  que recuperar el $D_{I,n,e}$ por unidad exige *sumar* $10\log_{10} N$. El
  equivalente en presión, la Fórmula (6) de ISO 10140-2:2010, imprime
  exactamente esa corrección
  ($D_{n,e} = L_1 - L_2 + 10\log_{10}(nA_0/A)$), y la Fórmula (12) de
  ISO 15186-2:2010 imprime la Fórmula (8) sin término alguno de $N$ (el caso
  $N = 1$, con el que ambos signos concuerdan). Tal como está impreso,
  instalar más unidades *bajaría* el índice por unidad en $20\log_{10} N$
  respecto al valor deducible.
- **Evidencia:** deducción desde la relación de sala receptora de campo
  difuso $L_2 = L_W + 10\log_{10}(4/A)$ contra la Fórmula (6) de
  ISO 10140-2:2010; contraste con la Fórmula (12) de ISO 15186-2:2010 y
  Hopkins, *Sound Insulation* (2007), Ec. 3.45. Verificado en la página 11
  del PDF (p. 3 impresa) de BS EN ISO 15186-1:2003, con el contraste leído en
  la página 11 del PDF (p. 11 impresa) de ISO 10140-2:2010. **La parte 3 de la
  propia serie lo zanja:** la ISO 15186-3:2002, apartado 3.9, Fórmula (8),
  enuncia la misma magnitud como
  $D_{I,n,e} = L_{pS} - 9 - [L_{In} - 10 \lg(A_0/S_m) - 10 \lg N]$, cuyo
  corchete lleva el $10 \lg N$ con el signo exterior contrario, es decir el
  $+10\log_{10} N$ que aquí se deduce. Leído en la página 10 del PDF (p. 4
  impresa) de BS EN ISO 15186-3:2010.
- **Comportamiento de la biblioteca:** implementa la forma por unidad
  deducible (`intensity_element_normalized_difference`, $+10\log_{10} N$) y
  emite un aviso siempre que $n > 1$, donde el resultado se desvía del
  impreso.
- **Estado:** sin notificar.

## ISO 15186-3:2002, anexo A, Tabla A.1 (la columna del sándwich de acero no se reproduce con sus propios datos)

- **Ubicación:** anexo A (normativo), A.2 y Tabla A.1, «Calculated sound
  reduction index (at 1 013 hPa and 23 °C)», el ejemplo de calificación con
  el que un laboratorio comprueba su instalación. El ejemplar leído es **BS EN
  ISO 15186-3:2010**, la adopción británica de texto idéntico de la ISO
  15186-3:2002, página 18 del PDF (p. 12 impresa).
- **El impreso:** dos columnas de seis valores en tercios de octava, de 50 Hz
  a 160 Hz. La columna del cartón-yeso va encabezada por «10 kg/m²» sobre un
  «Test opening 10 m²» y da 10,7 / 11,9 / 13,4 / 14,8 / 16,3 / 17,9. La del
  acero va encabezada por «17 kg/m²» sobre un «Test opening 1,25 m × 1,50 m» y
  da 21,3 / 21,2 / 21,7 / 22,7 / 23,8 / 25,1. A.2 añade que «the dimensions of
  the free part of the panel are 1,162 m × 1,412 m».
- **El problema:** ninguna lectura de los datos impresos junto a la columna del
  acero la reproduce. Con el hueco de ensayo (1,875 m²) y la masa declarada,
  los seis valores calculados quedan entre 1,27 dB y 0,72 dB por debajo de los
  impresos. Esa dispersión de 0,55 dB entre ambos extremos descarta cualquier
  masa superficial con esa área, porque un error de masa desplaza
  $R_0 = 20 \lg(\pi f m / \rho c)$ lo mismo en todas las bandas. Con la parte
  libre del panel (1,640744 m²) el residuo queda casi plano, de media
  0,562 dB, pero aun así se abre 0,102 dB de un extremo a otro, que es el ancho
  entero del decimal impreso, así que tampoco es el desplazamiento constante
  que dejaría una masa equivocada por sí sola.

  Ningún dato aislado la cierra dentro de los 0,05 dB que admite una impresión
  de un decimal. La mejor masa superficial sola, sobre la parte libre, es
  18,13 kg/m² y deja 0,051 dB; la mejor presión estática sola son 950 hPa y
  deja 0,051 dB; la mejor temperatura sola, sobre el hueco de ensayo, son 63 °C
  y deja 0,052 dB. Las dos últimas contradicen el encabezado, que fija el clima
  en 1 013 hPa y 23 °C, y la columna del cartón-yeso sí se reproduce justo con
  ese clima, así que las dos columnas no pueden leerse con climas distintos.

  La única lectura que sí reproduce los seis valores mueve dos datos a la vez:
  un área de unos 1,654 m², cercana a la parte libre pero no igual a ella,
  junto con una masa superficial de unos 18,16 kg/m². Esa masa no está al
  alcance de la probeta descrita. El acero macizo de 2,2 mm da entre
  16,9 kg/m² y 17,3 kg/m², y la hoja es un sándwich acero/resina/acero, así que
  su masa superficial queda por fuerza por debajo. La columna del cartón-yeso
  de esa misma tabla, con las mismas fórmulas y el mismo clima, reproduce sus
  seis valores dentro de 0,050 dB.
- **Evidencia:** Fórmulas (A.1) a (A.5) evaluadas a los 1 013 hPa y 23 °C
  declarados, leídas en las páginas 17 y 18 del PDF (pp. 11 y 12 impresas) de
  BS EN ISO 15186-3:2010. La ISO 140-3:1995, C.2.4, que A.2 cita como origen de
  la probeta, describe la hoja de acero/resina/acero de 2,2 mm pero no declara
  masa superficial alguna, así que los 17 kg/m² no vienen de ahí. No consta
  corrigendum al anexo A.
- **Comportamiento de la biblioteca:** `limp_panel_reduction_index` implementa
  las Fórmulas (A.1) a (A.5) tal como se imprimen. La suite de conformidad las
  ancla solo en la columna del cartón-yeso; la del acero no se usa como
  oráculo a propósito.
- **Estado:** sin notificar.

## ISO 10848-1:2006, apartado 8.1.1, Fórmula (20) (π espurio en la frecuencia crítica)

- **Ubicación:** apartado 8.1.1, Fórmula (20), la frecuencia crítica de placa
  delgada que usa el criterio de flancos de la instalación de ensayo de la
  Fórmula (19).
- **El impreso:** $f_c = c_0^{2} / (1{,}8\ c_L \cdot h \cdot \pi)$.
- **El problema:** la constante 1,8 es ya el
  $2\pi/\sqrt{12} \approx 1{,}814$ redondeado de la relación de dispersión de
  placa delgada, así que el $\pi$ extra la cuenta dos veces y desplazaría
  $f_c$ en un factor $\pi$ (p. ej. un elemento de hormigón de 100 mm con
  $c_L = 3500\ \text{m/s}$: 187 Hz sin el $\pi$, 59 Hz con él, lejos de
  cualquier valle de coincidencia medido).
- **Evidencia:** deducción desde la relación de dispersión de placa delgada
  (Hopkins, *Sound Insulation* (2007), Ec. 2.201,
  $f_c = c_0^2/(1{,}8 c_L h)$); ISO 12354-1:2017 imprime la misma forma sin
  $\pi$ en sus definiciones de símbolos ($f_c = c_0^2/(1{,}8 c_L t)$).
- **Comportamiento de la biblioteca:** implementa la forma sin $\pi$
  (`phonometry.building.measurement.flanking_transmission.critical_frequency`),
  con una nota de errata en el docstring.
- **Estado:** corregido aguas arriba. ISO 10848-1:2017 (segunda edición)
  imprime la forma sin $\pi$ en su Fórmula (5),
  $f_c = c_0^2/(1{,}8 h c_L)$, confirmando el impreso de 2006 como errata. No
  hace falta notificación. La entrada se conserva porque la biblioteca cita
  la edición de 2006, cuyo impreso lleva el defecto; la edición de 2017 queda
  como confirmación.

## UNE-EN 15657:2018, apartado 7.1, Fórmula (14) (masa de referencia dimensionalmente inconsistente con la magnitud que normaliza)

- **Ubicación:** apartado 7.1, la frase que introduce la Fórmula (14) (p. 14
  impresa) y la propia Fórmula (14) (p. 15 impresa), el nivel de potencia
  estructural inyectado en la placa de recepción.
- **El impreso:** la frase lee «a partir del nivel de velocidad promediado
  espacialmente de la placa $L_v$, de la **masa por unidad de superficie**
  $m$, del área de la placa $S$ y del factor de pérdida $\eta$, utilizando
  $f_0 = 1$ Hz, $m_0 = 1$ kg y $S_0 = 1$ m² como referencias», sobre
  $L_{Ws} = \left(10\lg\left(\dfrac{2\pi f m \eta S}{f_0 \cdot m_0 \cdot S_0}\right)\right)\text{dB} + L_v - 60\ \text{dB}$.
- **El problema:** la misma frase define $m$ como una masa por unidad de
  superficie, en kg/m², y su referencia $m_0$ como 1 kg. Con $m$ en kg/m² y
  $S$ en m², el grupo $2\pi f\,\eta\,m\,S / (f_0 m_0 S_0)$ solo es
  adimensional si $m_0$ es 1 kg/m²; tal como está impreso arrastra un m⁻²
  suelto. La constante de cierre confirma la lectura pretendida:
  $10\lg(f_0 m_0 S_0 v_0^2 / P_0) = -60$ dB con $v_0 = 10^{-9}$ m/s y
  $P_0 = 1$ pW cierra en vatios solo cuando $f_0 m_0 S_0$ tiene las unidades
  de una densidad superficial por un área por una frecuencia. El resultado
  numérico no se ve afectado, porque $10\lg(1) = 0$ sea cual sea la unidad
  adjunta, que es por lo que el desliz sobrevive a un ejemplo resuelto.
- **Evidencia:** análisis dimensional de la Fórmula (14) contra la definición
  de $m$ en la frase que la precede, y contra la constante de $-60$ dB con la
  que cierra; la frase y la fórmula se leyeron como imágenes, no desde el
  texto extraído. Verificado en la página 14 del PDF (p. 14 impresa) y la
  página 15 del PDF (p. 15 impresa) de UNE-EN 15657:2018. Solo se leyó la
  adopción en español, así que esta entrada no establece si el impreso inglés
  de EN 15657:2018 lleva la misma referencia.
- **Comportamiento de la biblioteca:** no hizo falta ningún cambio.
  `characteristic_reception_plate_power` toma `mass_per_area` en kg/m² y
  reproduce los propios valores resueltos de la norma, así que la lectura
  pretendida es la implementada; la guía y el docstring conservan la
  referencia impresa y nombran esta entrada a su lado.
- **Estado:** sin notificar.

## ISO 12999-1:2020, Tabla 4 (falta la fila de 500 Hz)

- **Ubicación:** Tabla 4 (incertidumbres in situ por banda).
- **El impreso:** la tabla de la edición de 2020 omite la fila de 500 Hz que
  la edición de 2014 imprime (situación B 1,2 dB / situación C 0,8 dB).
- **El problema:** probable omisión editorial; las filas circundantes no
  cambian entre ediciones y el texto no menciona retirar la banda.
- **Evidencia:** comparación lado a lado de los impresos de 2014 y 2020.
- **Comportamiento de la biblioteca:** sigue el impreso de 2020 tal como está
  publicado, con la omisión documentada en el módulo.
- **Estado:** sin notificar.

## ISO 12999-2:2020, redacción del apartado 8 frente a las Tablas 4 y 5

- **Ubicación:** apartado 8 **«Reporting uncertainties»** (pp. 5-6 impresas),
  la lista de símbolos bajo la Fórmula (10), contra las Tablas 4 y 5
  resueltas (p. 7 impresa). Una revisión anterior de esta entrada llamaba al
  apartado «expression of results», que no es su título impreso.
- **El impreso:** la lista de símbolos define $u$ como «the standard
  uncertainty determined in accordance with Clause 5, Clause 6 or Clause 7
  **rounded to two decimal digits for absorption coefficients** or one
  decimal digit for all other quantities», y la Fórmula (10) forma después
  $U = k \cdot u$.
- **El problema:** las propias Tablas 4 y 5 del documento solo se reproducen
  cuando $U$ se calcula desde el $u$ sin redondear y se redondea al final.
  Ninguna de las dos tablas imprime columna alguna de $u$ (cada una lleva
  solo el coeficiente $\alpha_s$ o $\alpha_p$ y $\pm U$ con $k = 2$), así que
  los valores impresos de $U$ son toda la evidencia, y 11 de los 25 son
  inalcanzables bajo la redacción literal del apartado.
- **Evidencia:** recálculo de las 25 entradas (Tabla 4: 20 filas, Tabla 5: 5
  filas) desde la Fórmula (1) con las constantes de la Tabla 1 y desde la
  Fórmula (4) con las constantes de la Tabla 2, bajo ambos convenios.
  Redondear al final reproduce 25 de 25; redondear primero falla 11 de 25
  (63, 125, 160, 200, 250, 1250, 1600, 2000, 3150 y 4000 Hz de la Tabla 4, y
  250 Hz de la Tabla 5). Una revisión anterior de esta entrada citaba el
  recuento como «10 de 20», que no es ni el numerador correcto ni el número
  correcto de entradas. Verificado en las páginas 9 (p. 3 impresa), 10 (p. 4
  impresa), 11 (p. 5 impresa) y 13 (p. 7 impresa) del PDF de
  ISO 12999-2:2020.
- **Comportamiento de la biblioteca:** redondea al final, casando con las
  tablas; el convenio está documentado y probado.
- **Estado:** sin notificar.

## ISO 12999-2:2020, Tabla 5 (datos de banda de octava bajo un encabezado de tercio de octava)

- **Ubicación:** apartado 8, Tabla 5 «Example for the practical sound
  absorption coefficient, αp, and its expanded uncertainty under
  reproducibility conditions» (p. 7 impresa).
- **El impreso:** la columna de frecuencias de la Tabla 5 se encabeza
  **«One-third octave midband frequency / Hz»** y sus filas son 250, 500,
  1 000, 2 000 y 4 000 Hz.
- **El problema:** esas cinco frecuencias son la serie de bandas de
  **octava** de ISO 11654, que es sobre la que se define el coeficiente de
  absorción sonora práctico $\alpha_p$; no son una serie de tercios de
  octava, y entre ellas no falta ningún tercio de octava. El documento se
  contradice sobre la misma magnitud dos páginas antes: la Tabla 2, que
  aporta las constantes $m$ y $n$ de la Fórmula (4) para exactamente estas
  cinco frecuencias, se encabeza «Octave midband frequency». El mismo texto
  de encabezado figura sobre la Tabla 4 de la misma página, donde es
  correcto: esa tabla lleva una serie genuina de tercios de octava, de 63 Hz
  a 5 000 Hz en 20 filas.
- **Evidencia:** las cinco frecuencias tabuladas mismas, y el encabezado
  «Octave midband frequency» de la Tabla 2 para las mismas constantes de
  $\alpha_p$. Verificado en la página 13 del PDF (p. 7 impresa) y la página
  11 del PDF (p. 5 impresa) de ISO 12999-2:2020.
- **Comportamiento de la biblioteca:** `_TABLE2` en
  [`uncertainty.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/materials/absorbers/uncertainty.py)
  está indexada por frecuencia central de *octava*, siguiendo la Tabla 2 y la
  definición de $\alpha_p$ de ISO 11654 y no el encabezado de la Tabla 5.
- **Estado:** sin notificar.

## ISO 10052:2021, encabezado del rango de volúmenes de la Tabla 4

- **Ubicación:** Tabla 4 (estimador del índice de reverberación), encabezado
  del rango de volúmenes.
- **El impreso:** el encabezado lee «60 ≤ V < 150» mientras que el cuerpo del
  texto dice que el método se aplica a salas «up to 150 m³».
- **El problema:** el límite $V = 150\ \text{m}^3$ queda incluido por el
  texto y excluido por el encabezado.
- **Evidencia:** comparación directa del encabezado y el texto del apartado.
- **Comportamiento de la biblioteca:** acepta $V = 150$ (sigue el texto), con
  la ambigüedad anotada.
- **Estado:** sin notificar.

## ISO 16283-1:2014, apartado 6 (un tiempo de reverberación de la sala emisora)

- **Ubicación:** apartado 6 «General», el párrafo sobre el tiempo de
  reverberación (p. 6 impresa).
- **El impreso:** «For the reverberation time, the low-frequency procedure
  shall be used for the 50 Hz, 63 Hz, and 80 Hz one-third octave bands in
  **the source and/or receiving room** when its volume is smaller than 25 m³
  (calculated to the nearest cubic metre).»
- **El problema:** ISO 16283-1 no mide ningún tiempo de reverberación de la
  sala emisora, así que no hay nada en la sala emisora a lo que aplicar un
  procedimiento de tiempo de reverberación. El primer párrafo del mismo
  apartado, cinco párrafos y una NOTA antes, lista las mediciones requeridas
  como «the sound pressure levels in both rooms with the source(s) operating,
  the background noise in the receiving room ... and the reverberation times
  **in the receiving room**». El apartado 10, que es donde de verdad se
  especifican los procedimientos de tiempo de reverberación, dice lo mismo
  cuatro veces: su título es «Reverberation time **in the receiving room**
  (default and low-frequency procedure)», su apartado 10.1 acota todo el
  apartado a «the receiving room», su apartado 10.3 bifurca según si «the
  receiving room has a volume larger than or equal to 25 m³», y su apartado
  10.4 aplica el procedimiento de baja frecuencia «when **the receiving
  room** volume is smaller than 25 m³». La locución es correcta dos párrafos
  más arriba, uno de ellos la NOTA, donde le corresponde: el *nivel de
  presión sonora* sí se mide en ambas salas y su procedimiento de baja
  frecuencia sí aplica a cualquiera de las dos. Se arrastró hasta la frase
  del tiempo de reverberación, donde solo existe una sala. Las otras dos
  partes imprimen la misma frase con una sola sala: el apartado 6 de
  ISO 16283-2:2020 y el apartado 6 de ISO 16283-3:2016 leen ambos «in the
  receiving room when its volume is smaller than 25 m³», así que la Parte 1
  es la discrepante de las tres.
- **Evidencia:** la frase en la página 12 del PDF (p. 6 impresa) de
  ISO 16283-1:2014, idéntica en la página 14 del PDF (p. 6 impresa) de
  BS EN ISO 16283-1:2014; el apartado 10 y sus subapartados en las páginas 23
  y 24 del PDF (pp. 17 y 18 impresas) del mismo documento; y la versión de
  una sola sala de la frase en la página 13 del PDF (p. 7 impresa) de
  ISO 16283-2:2020 y la página 16 del PDF (p. 10 impresa) de
  ISO 16283-3:2016.
- **Comportamiento de la biblioteca:** la sustitución por la octava de 63 Hz
  es una operación de sala receptora en todas las partes, siguiendo el
  apartado 10; un procedimiento de sala emisora que lleve un tiempo de
  reverberación de la octava de 63 Hz se rechaza, y una llamada de sala
  emisora no toma tiempo de reverberación alguno. El procedimiento de esquina
  para el *nivel*, que es el párrafo al que pertenece la locución, sí admite
  ambas salas en ISO 16283-1 y el punto de entrada aéreo ofrece ambas.
- **Estado:** sin notificar.

## ISO 16283-2:2020, apartado 8.3 (una sala emisora en una medición de impacto)

- **Ubicación:** apartado 8.3 «Microphone positions», último párrafo (p. 15
  impresa).
- **El impreso:** «For the 50 Hz, 63 Hz and 80 Hz one-third octave bands,
  calculate the low-frequency energy-average sound pressure level **for the
  source and/or receiving room** according to 8.5.»
- **El problema:** una medición de impacto no tiene nivel de presión sonora
  de sala emisora que calcular. Todas las demás formulaciones del mismo
  procedimiento en la misma parte nombran una sola sala: el apartado 6 lo
  introduce como usado «in the receiving room when its volume is smaller than
  25 m³» (p. 6 impresa), el apartado 8.1 repite «in the receiving room»
  (p. 14 impresa), el apartado 8.5 construye $L_\text{i,Corner}$ desde
  esquinas de la sala receptora (p. 16 impresa), y las Fórmulas (1) y (3), a
  las que la misma frase remite al lector, están escritas en $L_\text{i}$, el
  nivel de presión sonora de impacto promedio energético en la sala
  receptora. La locución es correcta allí de donde viene: el apartado 8.3 de
  ISO 16283-1 dice «for the source and/or receiving room» de una medición
  aérea, donde ambas salas sí llevan un nivel. Se copió a la parte de impacto
  y sobrevivió a la revisión sin cambios.
- **Evidencia:** la frase en la página 21 del PDF (p. 15 impresa) de
  ISO 16283-2:2020 junto a la misma frase en la página 23 del PDF (p. 15
  impresa) del texto ISO/DIS 16283-2 circulado como BSI DPC 13/30269186 DC, y
  el original aéreo en la página 21 del PDF (p. 15 impresa) de
  ISO 16283-1:2014.
- **Comportamiento de la biblioteca:** el punto de entrada de impacto toma un
  procedimiento de baja frecuencia de sala receptora y nada más, siguiendo
  los apartados 6, 8.1 y 8.5; solo el punto de entrada aéreo, donde el
  apartado 8.1 de ISO 16283-1 sí admite ambas salas, ofrece uno de sala
  emisora.
- **Estado:** sin notificar.

## ISO 16283-2:2020, apartado 10.3 (una sala receptora de exactamente 25 m³)

- **Ubicación:** apartado 10.3 «Default procedure» del tiempo de
  reverberación (p. 18 impresa).
- **El impreso:** «for all one-third octave bands between 50 Hz and 5 000 Hz
  when the receiving room has a volume **larger than** 25 m³ (calculated to
  the nearest cubic metre) and between 100 Hz and 5 000 Hz when the receiving
  room has a volume smaller than 25 m³ (calculated to the nearest cubic
  metre)».
- **El problema:** una sala receptora que redondea a exactamente 25 m³ no cae
  en ninguna de las dos ramas, así que el apartado no declara rango de
  frecuencias para ella. Las otras dos partes imprimen «larger than **or
  equal to** 25 m³» en la frase por lo demás idéntica, que cierra el límite.
  La lectura pretendida no está en duda: el disparador de los apartados 8.1 y
  10.4 es «smaller than 25 m³» en las tres partes, así que 25 m³ pertenece a
  la rama mayor y toma el rango por defecto completo de 50 Hz a 5 000 Hz.
- **Evidencia:** página 24 del PDF (p. 18 impresa) de ISO 16283-2:2020,
  contra la página 24 del PDF (p. 18 impresa) de ISO 16283-1:2014 y la página
  24 del PDF (p. 18 impresa) de ISO 16283-3:2016, que llevan ambas el «or
  equal to». El hueco no es un desliz de 2020 ni un artefacto de un borrador:
  el texto ISO/DIS en la página 26 del PDF (p. 18 impresa) de
  BSI DPC 13/30269186 DC ya leía igual, y también la edición anterior
  publicada, cuyo apartado 10.3 en la página 25 del PDF (p. 25 impresa) de
  UNE-EN ISO 16283-2:2016, la traducción española de ISO 16283-2:2015, lee
  «un volumen **superior a** 25 m³» sin «o igual a». La redacción ha
  permanecido sin cambios a través de dos ediciones y una revisión.
- **Comportamiento de la biblioteca:** el predicado disparador es el
  «smaller than 25 m³» estricto que comparten las tres partes, así que una
  sala de exactamente 25 m³ toma el procedimiento por defecto en todas las
  partes y no existe hueco.
- **Estado:** sin notificar.

## ISO 17208-2:2019, cobertura de bandas de la incertidumbre del apartado 5

- **Ubicación:** apartado 5 (incertidumbres expandidas representativas), p. 4
  impresa.
- **El impreso:** «5 dB for the low frequency (10 Hz to 100 Hz) bands, 3 dB
  for the mid frequency (125 Hz to 16 000 Hz) bands, and 4 dB for the high
  frequency (**>20 000 Hz**) bands».
- **El problema:** la propia banda de tercio de octava de 20 kHz queda sin
  asignar: el rango medio termina en 16 kHz *inclusive* y el rango alto
  empieza estrictamente por encima de 20 kHz. ISO 17208-1:2016, de la que el
  apartado 5 dice tomar los valores, imprime los mismos tres rangos con
  «**≥20 000 Hz**», que cierra el hueco; la Parte 2 degradó el $\ge$ a un
  $>$. La banda de 20 kHz no es un caso límite para este documento: la Tabla
  1 de ISO 17208-1 exige que la medición cubra «20 000 Hz (minimum)» como su
  banda de tercio de octava superior. Una revisión anterior de esta entrada
  decía «nada cubre de 16 kHz a 20 kHz inclusive», que está mal por el
  extremo inferior: 16 kHz sí está cubierto.
- **Evidencia:** los dos apartados lado a lado. Verificado en la página 10
  del PDF (p. 4 impresa) de ISO 17208-2:2019 y la página 22 del PDF (p. 16
  impresa) de ISO 17208-1:2016.
- **Comportamiento de la biblioteca:** aplica el valor conservador de 4 dB de
  la banda alta desde la banda de 20 kHz hacia arriba, siguiendo la Parte 1,
  con el hueco documentado.
- **Estado:** sin notificar.

## ECMA-418-1:2024 (3.ª edición), NOTA 2 del apartado 4.1.1 (límite superior del rango de tonos discretos)

- **Ubicación:** apartado **4.1.1** «frequency range of interest», NOTA 2
  (p. 2 impresa). Una revisión anterior de esta entrada citaba el apartado
  4.1.2, que es la definición de «ITT equipment» y no dice nada de
  frecuencias.
- **El impreso:** «From viewpoint of test implementation by using FFT
  analyser, the frequency range of discrete tones are between 89,1 Hz and
  11 220 Hz inclusive, referred to *the discrete tone frequency range of
  interest*.»
- **El problema:** todas las fórmulas y tablas de la norma trabajan hasta
  11 200 Hz: los ajustes de bordes de banda de las Tablas 2 y 3 se declaran
  para $11\,200 \ge f_t > 1\,600$, y los apartados 10, 12.3 y 12.4 permiten
  datos FFT con $f_1 < 89{,}1\ \text{Hz}$ y $f_2 > 11\,200\ \text{Hz}$. Los
  dos números son la misma cantidad a distinta precisión y no un error
  tipográfico: $10\,000 \cdot 2^{1/6} = 11\,224{,}6\ \text{Hz}$ es el borde
  superior de la banda de tercio de octava de 10 kHz que cierra el rango de
  interés, que redondea a 11 220 Hz con cuatro cifras significativas y a
  11 200 Hz con tres. Una revisión anterior de esta entrada lo llamaba errata
  y añadía que «ningún otro apartado menciona 11 220 Hz»; la última marca del
  eje x de la Figura 6 (p. 20 impresa) está etiquetada 11220. Lo que sí lleva
  el apartado 4.1 es un defecto estructural: 4.1.2 «ITT equipment» repite al
  pie de la letra la NOTA 1 de 4.1.1 («This range was selected to be
  identical to that of ECMA-74:2022, 3.1.3»), aunque 4.1.2 no define rango
  alguno, y el apartado 10 remite después a «NOTE 1 of 4.1.2» para el rango
  de tonos discretos, que es la nota duplicada y no la NOTA 2 que lo declara.
- **Evidencia:** la aritmética de arriba, y los rangos de las Tablas 2/3 y el
  eje de la Figura 6 leídos junto a la NOTA 2. Verificado en la página 10 del
  PDF (p. 2 impresa), la página 18 del PDF (p. 10 impresa), la página 25 del
  PDF (p. 17 impresa) y la página 28 del PDF (p. 20 impresa) de
  ECMA-418-1:2024 (3.ª edición).
- **Comportamiento de la biblioteca:** usa el rango internamente consistente
  de $89{,}1\ \text{Hz}$ a 11 200 Hz (extremo superior exclusivo según las
  fórmulas), con una nota en el código en
  [`tonality.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/quality/tonality.py).
- **Estado:** sin notificar.

## ECMA-418-1:2024 (3.ª edición), Fórmula (21) (término constante repetido)

- **Ubicación:** apartado 12.3, Fórmula (21) (p. 17 impresa), el ajuste de
  curva de la frecuencia del borde inferior $f_{1,L}$ de la banda crítica
  inferior.
- **El impreso:** $f_{1,L} = C_{L,0} + C_{L,0} f_t + C_{L,2} f_t^{2}$.
- **El problema:** el coeficiente lineal repite el término constante. La
  lista de símbolos inmediatamente debajo de la fórmula declara «$C_{L,0}$,
  $C_{L,1}$, $C_{L,2}$ are constants given in Table 2», la Tabla 2 tabula una
  columna $C_{L,1}$, y la Fórmula (22) paralela para el borde superior de
  banda imprime $f_{2,U} = C_{U,0} + C_{U,1} f_t + C_{U,2} f_t^{2}$
  correctamente. La errata es numéricamente fatal, no cosmética: en el rango
  de ajuste central ($171{,}4 \le f_t \le 1\,600$) la Tabla 2 da
  $C_{L,0} = -149{,}5$ y $C_{L,1} = 1{,}001$, así que la forma impresa
  devuelve $-149{,}5 - 149{,}5 f_t - 6{,}90 \cdot 10^{-5} f_t^2$, negativa en
  todas partes, en lugar de un borde de banda un poco por debajo de $f_t$.
- **Evidencia:** la fórmula, su propia lista de símbolos y la Tabla 2 en una
  misma página, con la Fórmula (22) como control consistente. Verificado en
  la página 25 del PDF (p. 17 impresa) de ECMA-418-1:2024 (3.ª edición).
- **Comportamiento de la biblioteca:** implementa la lectura con $C_{L,1}$,
  que es la única que devuelve un borde de banda usable, con una nota en el
  código en
  [`tonality.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/quality/tonality.py).
- **Estado:** sin notificar.

## ECMA-418-1:2024 (3.ª edición), apartado 11.3 (referencias de campo sin resolver)

- **Ubicación:** apartado 11.3 «Determination of masking noise level» (p. 12
  impresa), la frase que introduce el ancho de banda crítico.
- **El impreso:** «The critical bandwidth Δf_c is determined from Formula
  **Error! Reference source not found.Error! Reference source not found.**
  with f_0 set equal to the frequency of the discrete tone under
  investigation, f_t».
- **El problema:** dos referencias de campo de procesador de textos sin
  resolver quedaron compuestas, en negrita, en el lugar de los números de
  fórmula, y salieron publicadas en la tercera edición. Los destinos
  pretendidos son inequívocos por el resto de la frase, que pasa a nombrar
  las Fórmulas (4) y (5) o (7) y (8) para los bordes de banda: el ancho de
  banda crítico mismo es la Fórmula (2), y la Fórmula (3) es la relación
  $f_2 - f_1 = \Delta f_c$ que lo convierte en bordes de banda.
- **Evidencia:** el apartado tal como está impreso. Verificado en la página
  20 del PDF (p. 12 impresa), la página 18 del PDF (p. 10 impresa) y la
  página 30 del PDF (p. 22 impresa) de ECMA-418-1:2024 (3.ª edición).
- **Comportamiento de la biblioteca:** no hizo falta ninguno; la biblioteca
  implementa el ancho de banda crítico desde las Fórmulas (3)/(6)
  directamente.
- **Estado:** sin notificar.

## ECMA-418-2:2025 (4.ª edición), apartado 5.1.5.2 (índice del último bloque)

- **Ubicación:** apartado 5.1.5.2, la segmentación de la señal rellenada con
  ceros para los tamaños de bloque de aspereza/intensidad de fluctuación.
- **El impreso:** el índice del último bloque se da como
  $l_\text{last} = \lceil (n + s_b)/s_h \rceil$.
- **El problema:** la fórmula es internamente inconsistente: los bloques
  colocados en ese índice desbordan la señal rellenada con ceros que define
  el apartado 5.1.2.2, y la rejilla temporal resultante de la Fórmula (103)
  deja de ser monótona. La única lectura autoconsistente es detenerse en el
  último bloque que cabe dentro de la señal rellenada y alinearlo a ras con
  su final.
- **Evidencia:** evaluación directa de los índices de comienzo de bloque
  contra la longitud rellenada para los tamaños de bloque/salto del apartado
  7.1.1; la lectura a ras del final reproduce la calibración de aspereza del
  apartado 7 ($1\ \text{asper}$) a $0{,}9999$.
- **Comportamiento de la biblioteca:** implementa la lectura a ras del final
  con una nota en el código en
  [`roughness_ecma.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/quality/roughness_ecma.py).
- **Estado:** sin notificar.

## ECMA-418-2:2025 (4.ª edición), apartado 9.1.4, Fórmula (127) (fase del núcleo HSA)

- **Ubicación:** apartado 9.1.4, Fórmula (127), el núcleo espectral de la
  ventana de análisis de envolvente que usa el High-resolution Spectral
  Analysis.
- **El impreso:** el factor de fase del núcleo es
  $\exp(-j \cdot 2\pi \cdot f_n(k) \cdot (\tilde{s}_b - n_{ze} + n_{zb} - 1))$.
- **El problema:** el núcleo es, por construcción, la DFT de la ventana de
  análisis rectangular de la Fórmula (120) modulada a la tasa candidata; ese
  es el modelo que la Fórmula (124) ajusta al espectro DFT medido. Esa DFT
  tiene la fase
  $\exp(-j \cdot \pi \cdot f_n \cdot (\tilde{s}_b - n_{ze} + n_{zb} - 1))$;
  el factor impreso la duplica (y es además inconsistente con los argumentos
  en $\pi$ de los términos seno impresos de la misma fórmula). Con la fase
  impresa el modelo ajustado no puede reproducir el espectro de una sinusoide
  enventanada sin ruido, contradiciendo la propia afirmación del apartado de
  que el HSA alcanza «theoretically infinite resolution for signals without
  noise».
- **Evidencia:** deducción independiente de la DFT de la ventana más
  recálculo numérico: con $\pi$ el ajuste por mínimos cuadrados recupera la
  parte constante, las amplitudes y las fases de envolventes sintéticas sin
  ruido a precisión de máquina y el residuo de la Fórmula (135) se anula; con
  el $2\pi$ impreso el núcleo se desvía de la DFT de la ventana en cantidades
  del orden del propio núcleo y el residuo se queda del orden de la energía
  de la señal.
- **Comportamiento de la biblioteca:** implementa la lectura con $\pi$,
  fijada por un test de regresión sobre la recuperación exacta de pares de
  líneas sintéticos.
- **Estado:** sin notificar.

## ECMA-418-2:2025 (4.ª edición), apartado 9.1.5, Fórmula (144) (desplazamiento de bin)

- **Ubicación:** apartado 9.1.5, Fórmula (144), la tasa de modulación de un
  máximo local del espectro de potencia de la envolvente.
- **El impreso:** la tasa es el centroide de tres bins ponderado por amplitud
  de la posición del pico **menos uno**, escalado por $\Delta f$.
- **El problema:** el apartado 9.1.4 (bajo la Fórmula (122)) define el índice
  espectral $k$ como el que mapea a la tasa de modulación
  $k \cdot \tilde{r}_s/\tilde{s}_b$ con $k$ empezando en 0. Un máximo local
  simétrico en el bin $k$ tiene centroide $k$, y la fórmula impresa le asigna
  entonces la tasa $(k - 1) \cdot \Delta f$, un bin entero
  ($0{,}73\ \text{Hz}$) por debajo, lo que a tasas de intensidad de
  fluctuación es fatal (una modulación verdadera de $1{,}46\ \text{Hz}$ se
  reportaría como $0{,}73\ \text{Hz}$). El desplazamiento solo es consistente
  con posiciones de línea espectral basadas en 1, contradiciendo la propia
  definición de $k$ de la norma.
- **Evidencia:** contraste de la Fórmula (144) contra el mapeo de $k$ a tasa
  declarado bajo la Fórmula (122).
- **Comportamiento de la biblioteca:** usa el centroide directamente (sin
  desplazamiento) con el $k$ basado en 0 de la Fórmula (122).
- **Estado:** sin notificar.

## ECMA-418-2:2025 (4.ª edición), apartado 9.1.7 (unidades de las constantes de ajuste fino)

- **Ubicación:** apartado 9.1.7, Fórmulas (149)-(152), el ajuste fino por
  Newton amortiguado de la tasa de modulación dominante.
- **El impreso:** paso diferencial $\Delta x = 10^{-5}$, tope del paso
  amortiguado $2 \cdot 10^{-4}$, tolerancia de parada $10^{-7}$ y un límite
  de 40 iteraciones, con el punto de partida
  $x_0 = \tilde{f}_{c,i_\text{max}}$ (una tasa en Hz) y la comprobación de
  fallo
  $|f_{c,1,\text{opt}} - \tilde{f}_{c,i_\text{max}}| > 1{,}25 \cdot \Delta f$.
- **El problema:** las constantes no llevan unidades. Leídas en Hz, el paso
  amortiguado queda topado a $5 \cdot 10^{-5}\ \text{Hz}$ por iteración
  ($2 \cdot 10^{-3}\ \text{Hz}$ en las 40 iteraciones), así que el ajuste no
  puede moverse apreciablemente y la comprobación de fallo de
  $1{,}25 \cdot \Delta f$ ($\approx 0{,}92\ \text{Hz}$) es inalcanzable; el
  apartado entero sería inerte. Leídas como tasas de modulación normalizadas
  $f/\tilde{r}_s$ (la variable en la que se expresan las frecuencias del
  núcleo de la Fórmula (127)), las mismas constantes dan un tope amortiguado
  por iteración de $0{,}075\ \text{Hz}$ ($\approx 2{,}9\ \text{Hz}$ en las 39
  iteraciones), una tolerancia de parada de
  $1{,}5 \cdot 10^{-4}\ \text{Hz}$ y una comprobación de fallo alcanzable,
  todo consistente con el propósito del apartado.
- **Evidencia:** análisis dimensional de las constantes impresas contra la
  resolución espectral de $0{,}7324\ \text{Hz}$ y el umbral de fallo.
- **Comportamiento de la biblioteca:** aplica las constantes como tasas de
  modulación normalizadas.
- **Estado:** sin notificar.

## ECMA-418-2:2025 (4.ª edición), introducción del apartado 9 (referencia cruzada rota)

- **Ubicación:** apartado 9, tercer párrafo de la introducción, sobre la
  predicción de sonoridad basada en HSA.
- **El impreso:** «loudness scaling is improved by using HSA-based loudness
  prediction (see Clause 0)».
- **El problema:** «Clause 0» no existe; el escalado de sonoridad basado en
  HSA se describe en el apartado 9.1.10 (una referencia de campo sin
  resolver).
- **Evidencia:** el propio índice de apartados de la norma.
- **Comportamiento de la biblioteca:** no hizo falta ninguno (el destino
  pretendido es inequívoco).
- **Estado:** sin notificar.

## ISO/PAS 20065:2016, apartado 5.3.4 (pendiente de los flancos de un tono destacado)

- **Ubicación:** apartado 5.3.4, Fórmulas (10)/(11) (p. 9 impresa), la
  pendiente mínima de los flancos de un tono destacado.
- **El impreso:** los dos flancos se escalan de forma distinta:
  $\Delta L_u = (f_T/2) \cdot (L_{T\text{max}} - L_u)/(f_T - f_u) \ge 24\ \text{dB}$
  y
  $\Delta L_o = f_T \cdot (L_{T\text{max}} - L_o)/(f_o - f_T) \ge 24\ \text{dB}$.
- **El problema:** la norma madre DIN 45681:2005-03 imprime $f_T/\sqrt{2}$ en
  **ambos** flancos (Gleichungen (10)/(11), p. 14 impresa), y su programa de
  referencia ejecutable del Anhang J hace lo mismo (`Frequenz(i)/Sqr(2)`).
  Los dos impresos no pueden satisfacerse a la vez. Ninguno de los factores
  ISO es el de DIN: en el flanco inferior $1/2 < 1/\sqrt{2}$, así que el
  impreso ISO devuelve una diferencia de nivel $\sqrt{2}$ **menor** y es por
  tanto **más estricto**; en el flanco superior el divisor falta por
  completo, así que el impreso ISO devuelve $\sqrt{2}$ **mayor** y es **más
  laxo**. Una revisión anterior de esta entrada tenía las dos direcciones al
  revés y describía el flanco superior como «a la mitad», cuando en realidad
  el divisor falta, no está a la mitad. Los tonos límite con pendiente de un
  solo flanco entre $24/\sqrt{2} = 17$ y
  $24 \cdot \sqrt{2} = 34\ \text{dB/octave}$ cambian de clasificación entre
  las dos lecturas.
- **Evidencia:** comparación lado a lado del impreso ISO, el impreso de
  DIN 45681 y el programa del Anhang J de DIN. Los radicales de DIN son
  exactamente el caso para el que existe la regla de la página: `pdftotext`
  pierde el glifo `√` de ambas fórmulas DIN, así que el texto extraído lee
  `f_T/2` y casa con el impreso ISO, mientras que la página misma lee
  `f_T/√2`. Verificado en la página 13 del PDF (p. 9 impresa) de ISO/PAS
  20065:2016 y la página 14 del PDF (p. 14 impresa) de DIN 45681:2005-03.
- **Comportamiento de la biblioteca:** sigue la lectura DIN/$\sqrt{2}$ (casa
  con la única referencia ejecutable), con la elección registrada en
  [`tone_audibility.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/quality/tone_audibility.py).
- **Estado:** sin notificar.

## DIN 45681:2005-03, Anhang I, Tabelle I.6, fila «6 FG»

- **Ubicación:** Anhang I, Beispiel I.2 (motor de combustión, espectro
  $j = 1$), Tabelle I.6, la fila combinada «6 FG» de los tres tonos
  $k = 6/7/8$ ($592{,}2$ / $629{,}8$ / $643{,}3\ \text{Hz}$, niveles de tono
  $78{,}31$ / $75{,}00$ / $79{,}75\ \text{dB}$).
- **El impreso:** $L_T = 81{,}11\ \text{dB}$ junto con
  $\Delta L = 9{,}12\ \text{dB}$ (con $L_S = 59{,}53$, $L_G = 76{,}16$,
  $a_v = -2{,}40$ a $592{,}2\ \text{Hz}$).
- **El problema:** las dos celdas se contradicen. El
  $\Delta L = 9{,}12\ \text{dB}$ impreso solo se reproduce desde la suma
  energética *simple* de la Fórmula (17) de los tres niveles de tono
  ($82{,}873\,4\ \text{dB}$): $82{,}87 - 76{,}16 + 2{,}40 = 9{,}11$. El
  $L_T = 81{,}11\ \text{dB}$ impreso es esa misma suma menos exactamente
  $1{,}763\ \text{dB}$, y tomado al pie de la letra daría
  $\Delta L = 7{,}35\ \text{dB}$.
- **Evidencia:** recálculo desde los niveles por tono impresos de la Tabelle
  I.6. El desplazamiento es el discriminador y es una constante, no una
  deduplicación: $82{,}873\,4 - 81{,}11 = 1{,}763\ \text{dB}$, y
  $1{,}76\ \text{dB}$ es $10\log_{10} 1{,}5$, la propia corrección de ancho
  de banda efectivo de Hanning de la norma (apartado 5.3.2). El mismo
  desplazamiento aparece en la fila «5 FG» de la Tabelle I.10 (p. 46
  impresa), donde los dos tonos miembros a $705{,}2$ y $732{,}1\ \text{Hz}$
  tienen $L_T = 55{,}12$ y $54{,}23\ \text{dB}$, suman
  $57{,}708\ \text{dB}$, y se imprimen como $55{,}95\ \text{dB}$,
  $1{,}758\ \text{dB}$ más bajo, y allí el $\Delta L = 3{,}22\ \text{dB}$
  impreso sigue al $L_T$ impreso exactamente
  ($55{,}95 - 55{,}28 + 2{,}55 = 3{,}22$), así que la fila de la Tabelle I.10
  es internamente consistente y la de la Tabelle I.6 no. La tercera fila
  combinada, «2 FG» de la misma Tabelle I.6, no lleva desplazamiento alguno:
  sus tres niveles miembros $64{,}56$ / $67{,}96$ / $68{,}63\ \text{dB}$
  suman $72{,}149\ \text{dB}$ contra unos $72{,}15\ \text{dB}$ impresos, y su
  $\Delta L$ los sigue. Una revisión anterior de esta entrada atribuía la
  celda de $81{,}11\ \text{dB}$ a la deduplicación de líneas compartidas de
  la Anmerkung 2; ese diagnóstico no se sostiene, porque una deduplicación
  quita una cantidad arbitraria de energía mientras que todos los
  desplazamientos observados aquí son los mismos 1,76 dB. Verificado en la
  página 41 del PDF (p. 41 impresa) y la página 46 del PDF (p. 46 impresa) de
  DIN 45681:2005-03.
- **Comportamiento de la biblioteca:** `combined_tone_level` sigue la
  Anmerkung 2 (líneas compartidas contadas una vez), que reproduce el oráculo
  impreso de «2 FG»; para la fila «6 FG» solo se fija la cadena de
  $\Delta L$, con la contradicción registrada en `tests/reference_data/`.
- **Estado:** sin notificar.

## DIN 45681:2005-03, Anhang I, Tabellen I.2 y I.10 (índice de espectro equivocado en un encabezado de columna)

- **Ubicación:** Anhang I, los encabezados de columna de la Tabelle I.2
  (p. 37 impresa, espectro $j = 2$) y la Tabelle I.10 (p. 46 impresa,
  espectro $j = 24$).
- **El impreso:** todas las columnas de la Tabelle I.2 llevan el subíndice
  del índice de espectro 2 (`f_T 2,k`, `f_1 2,k`, `f_2 2,k`, `L_S 2,k`,
  `L_T 2,k`, `L_G 2,k`, `a_v 2,k`, `u_2,k`) salvo la columna de
  audibilidad, que se encabeza **`ΔL_1,k`**. Todas las columnas de la Tabelle
  I.10 llevan el subíndice 24 (`f_T 24,k`, `ΔL 24,k`, `f_1 24,k`,
  `f_2 24,k`, `L_S 24,k`, `L_T 24,k`, `L_G 24,k`, `u 24,k`) salvo la columna
  de enmascaramiento, que se encabeza **`a_v 1,k`**.
- **El problema:** ambas tablas llevan el índice de espectro del *primer*
  espectro en una columna. El propio pie de la Tabelle I.2 lee «des zweiten
  Spektrums (j = 2)» y el de la Tabelle I.10 «des 24. Spektrums (j = 24)», y
  los valores del cuerpo pertenecen a esos espectros: la columna $\Delta L$
  de la Tabelle I.2 es la audibilidad de los tonos de $j = 2$
  ($8{,}53\ \text{dB}$ a $627{,}2\ \text{Hz}$, que la Anmerkung bajo la tabla
  llama «die maßgebliche Differenz ΔL_2»), y la columna $a_v$ de la Tabelle
  I.10 es el índice de enmascaramiento de los tonos de $j = 24$. El índice 1
  está bien en exactamente una tabla del anexo, la Tabelle I.6, que es la
  tabla de $j = 1$ del Beispiel I.2 y lleva `ΔL_1,k` y `a_v 1,k`
  legítimamente.
- **Evidencia:** los propios pies de las tablas, los subíndices de sus
  columnas vecinas y la Anmerkung bajo cada una. Verificado en la página 37
  del PDF (p. 37 impresa), la página 46 del PDF (p. 46 impresa) y la página
  41 del PDF (p. 41 impresa) de DIN 45681:2005-03.
- **Comportamiento de la biblioteca:** no hizo falta ninguno; los números no
  se ven afectados. Las fixtures de regresión indexan ambas tablas por el
  espectro de su pie.
- **Estado:** sin notificar.

## IEC 60268-1:1985, Appendix A, Figura A1 (último condensador en derivación impreso como 41.47 nF)

- **Ubicación:** Appendix A, «Noise weighting network and quasi-peak meter»,
  Figura A1 «Weighting network» (p. 29 impresa), plano 0641/85. El impreso
  francés del mismo arte (p. 28 impresa) lleva el mismo valor como
  `41,47 nF`.
- **El impreso:** el último condensador en derivación de la escalera, el que
  cruza la entrada del amplificador de 600 Ω, está etiquetado **41.47 nF**.
- **El problema:** debería ser **31.47 nF**, que es lo que la Figura 1a de
  ITU-R BS.468-4 imprime para la misma red. Todos los demás elementos de la
  Figura A1 casan exactamente con la Figura 1a de BS.468-4: fuente de 600 Ω,
  13.85 nF, 12.88 mH, 26.82 nF, 33.06 nF, 9.21 nF, 26.49 mH y Z = 600 Ω. La
  lectura pretendida no está en duda, porque el documento se contradice a sí
  mismo: evaluada contra la Table AI, impresa dos páginas antes en el mismo
  anexo, la escalera con 31.47 nF reproduce las 21 filas con un máximo de
  **0.050 dB** y no viola tolerancia alguna, mientras que la escalera con
  41.47 nF se va hasta **2.252 dB** (a 31 500 Hz) con un error cuadrático
  medio de 1.055 dB y **rompe la propia columna de tolerancias de la Table AI
  en siete frecuencias**, todas de 8 000 Hz a 20 000 Hz: −0.40 dB contra
  ±0.40 a 8 kHz, −0.74 contra ±0.60 a 9 kHz, −1.16 contra ±0.80 a 10 kHz,
  −1.85 contra ±1.20 a 12.5 kHz, −1.98 contra ±1.40 a 14 kHz, −2.05 contra
  ±1.60 a 16 kHz y −2.12 contra ±2.00 a 20 kHz. Barrer el condensador para
  minimizar el error contra la Table AI aterriza en 31.4798 nF.
- **Evidencia:** las dos escaleras evaluadas de forma independiente por un
  producto de cadenas ABCD sobre los siete elementos reactivos impresos entre
  la fuente y la carga de 600 Ω impresas, normalizadas a 1 kHz, y comparadas
  fila a fila con la Table AI y su columna de tolerancias. Ni la enmienda
  Amendment 1:1988 (que sustituye solo la Table AII) ni la Amendment 2:1988
  (que sustituye el subapartado 12.1, sobre producir un campo magnético
  alterno uniforme) tocan la Figura A1, así que la errata sigue en pie en el
  documento vigente con sus enmiendas. Verificado en la página 31 del PDF
  (p. 29 impresa) y la página 29 del PDF (p. 27 impresa), que lleva la Table
  AI, de IEC 60268-1:1985, y en la página 1 del PDF (p. 1 impresa) de la
  Recomendación ITU-R BS.468-4.
- **Comportamiento de la biblioteca:** no le afecta. La red de ponderación se
  construye desde los valores de componentes de la Figura 1a de BS.468-4 en
  [`filters/weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/weighting.py), con
  31.47 nF, y las filas de la Table 1 son el oráculo. La entrada importa
  porque el subapartado 14.12.11 de IEC 60268-3:2013 remite al lector a «a
  weighting network complying with Appendix A of IEC 60268-1», así que una
  implementación en sala limpia arrancada desde IEC 60268-3 aterriza en el
  condensador equivocado.
- **Estado:** sin notificar.

## IEC 60268-1:1985, Appendix A, Table AII (fila del límite inferior corrida una columna)

- **Ubicación:** Appendix A, Table AII, la característica dinámica con
  ráfagas de tono del medidor de cuasi-pico, fila «Limited values — lower
  limit» (p. 31 impresa).
- **El impreso:** la fila del límite inferior (%) lee
  `13.5 | 22.4 | 34 | 41 | 44 | 44 | 50 | 68` para las columnas de 1, 2, 5,
  10, 20, 50, 100 y 200 ms, mientras que la fila (dB) impresa justo debajo
  lee `−17.4 | −13.0 | −9.3 | −7.7 | −7.1 | −6.0 | −4.7 | −3.3`.
- **El problema:** las celdas de 50 ms y 100 ms contradicen sus propias
  celdas en dB. −6.0 dB es 50.1 %, no 44 %, y −4.7 dB es 58.2 %, no 50 %. La
  fila de porcentajes se ha corrido una columna a la derecha desde 50 ms en
  adelante, arrastrando los valores de 20 ms y 50 ms a las dos celdas
  siguientes; la fila en dB y la celda de 200 ms se quedaron donde les
  corresponde. La Table 2 de ITU-R BS.468-4 imprime `... | 44 | 50 | 58 | 68`
  para las mismas cuatro columnas.
- **Evidencia:** las dos filas de la misma tabla leídas una contra otra, y
  contra la fila correspondiente de la Table 2 de ITU-R BS.468-4.
  **Corregido por la Amendment 1:1988-01**, cuya hoja inglesa se encabeza
  «Page 31 / Replace Table AII by the following:» e imprime la fila del
  límite inferior como `13.5 | 22.4 | 34 | 41 | 44 | 50 | 58 | 68`, casando
  con BS.468-4; todas las demás celdas de la tabla de sustitución son
  idénticas al impreso base, así que esta fila es todo el contenido
  sustantivo de la enmienda. Verificado en la página 33 del PDF (p. 31
  impresa) de IEC 60268-1:1985, en la página 3 del PDF (p. 3 impresa) de la
  Amendment 1:1988 de IEC 60268-1:1985, y en la página 4 del PDF (p. 4
  impresa) de la Recomendación ITU-R BS.468-4.
- **Comportamiento de la biblioteca:** no le afecta. Las once ventanas de
  aceptación están transcritas de las Tables 2 y 3 de BS.468-4 en
  [`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data), que concuerdan con la
  tabla IEC enmendada. Se registra porque el documento base sin enmendar es
  el que probablemente tenga un lector, y ensancha las ventanas de aceptación
  de 50 ms y 100 ms en 1.1 dB y 1.3 dB por abajo.
- **Estado:** sin notificar (corregido por el organismo emisor en 1988).

## ITU-R BS.468-4, Table 2, límite superior de 5 ms (la celda en dB debería leer −6.7)

- **Ubicación:** apartado 2.1, Table 2, «Limiting values — upper limit», la
  columna de 5 ms (p. 4 impresa). El mismo par de celdas está impreso
  idéntico en la Table AII de IEC 60268-1:1985 y en la tabla de la Amendment
  1:1988 que la sustituye, así que el defecto se hereda del texto del CCIR y
  no lo introduce ninguna de las dos ediciones.
- **El impreso:** `46` en la fila (%) y `−6.6` en la fila (dB).
- **El problema:** las dos discrepan. 46 % es 20 lg(0.46) = **−6.745 dB**, y
  −6.6 dB es 46.8 %. Se auditaron las 33 celdas de las Tables 2 y 3 contra su
  propia contraparte; 32 concuerdan dentro de 0.050 dB, el redondeo de un
  porcentaje de dos cifras significativas, y esta se va 0.145 dB. El límite
  *inferior* vecino de 5 ms (`34`, `−9.3`) se va 0.070 dB y es benigno,
  porque 34 % como porcentaje redondeado de dos cifras cubre de 33.5 % a
  34.5 %, es decir de −9.500 dB a −9.241 dB, y −9.3 cae dentro. La celda
  superior no es benigna: 46 % cubre de 45.5 % a 46.5 %, es decir de
  −6.840 dB a −6.651 dB, lo que excluye −6.6.
- **Qué celda está mal:** la de dB. Leída en porcentajes, la ventana de
  aceptación es un −1.4 dB / +1.2 dB muy estable en torno a la lectura de
  referencia para todas las duraciones de 5 ms a 200 ms (+1.18 a +1.24 dB por
  arriba, −1.37 a −1.45 dB por abajo). 46 % sitúa el límite superior de 5 ms
  +1.214 dB por encima de su referencia, sobre ese patrón; 46.774 %, que es
  lo que significa −6.6 dB, lo situaría +1.360 dB por encima, fuera de él.
  Así que 46 % está bien y la celda en dB debería leer **−6.7**.
- **Evidencia:** 20 lg de cada porcentaje impreso comparado con la celda en
  dB impresa a su lado, para las 24 celdas de la Table 2 y las 9 de la Table
  3, y los desplazamientos de los límites superior e inferior en torno a la
  fila de referencia recalculados en las cinco duraciones de 5 ms a 200 ms.
  Verificado en la página 4 del PDF (p. 4 impresa) de la Recomendación ITU-R
  BS.468-4, en la página 33 del PDF (p. 31 impresa) de IEC 60268-1:1985, y en
  la página 3 del PDF (p. 3 impresa) de la Amendment 1:1988 de
  IEC 60268-1:1985.
- **Comportamiento de la biblioteca:** las filas de porcentajes son primarias
  y las filas en dB se derivan de ellas, que es la decisión que esta entrada
  fuerza. Las once ventanas de aceptación se almacenan como porcentajes en
  [`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data) y se comprueban como
  porcentajes, en la suite de tests y en las filas de conformidad «ITU-R
  BS.468-4 Table 2» y «ITU-R BS.468-4 Table 3».
- **Estado:** sin notificar.

## IEC 60268-1:1985, Appendix A, Table AI (tolerancia de 16 000 Hz impresa como ±1.65)

- **Ubicación:** Appendix A, Table AI, la columna de tolerancias, fila de
  16 000 Hz (p. 27 impresa; el Tableau AI francés en la p. 26 impresa imprime
  el mismo valor).
- **El impreso:** `±1.65 1)`.
- **El problema:** la Table 1 de ITU-R BS.468-4 y la Table 1 de AES17-2015
  imprimen ambas **±1.6** para la misma fila, y la propia nota 1) de la tabla
  es lo que lo zanja: las tolerancias marcadas «are obtained by a linear
  interpolation on a logarithmic graph on the basis of values specified for
  the frequencies used to define the mask, i.e. 31.5 Hz, 100 Hz, 1 000 Hz,
  5 000 Hz, 6 300 Hz, and 20 000 Hz». Interpolado con esa regla entre
  (6 300 Hz, 0 dB) y (20 000 Hz, ±2.0 dB), 16 000 Hz da 1.6137 dB, que
  redondea a 1.6 con un decimal y a 1.61 con dos. Ningún redondeo de la regla
  produce 1.65, y ningún par de anclas alternativo tampoco: tomar la recta de
  6 300 Hz a 31 500 Hz da en cambio 1.6216 dB. El valor es además anómalo
  dentro de su propia columna, que está citada a un decimal en todos los
  demás sitios.
- **Evidencia:** la regla de la nota aplicada a las 14 filas marcadas de la
  misma columna, que reproduce todas y cada una (63 Hz 1.400, 200 Hz 0.8495,
  400 Hz 0.6990, 800 Hz 0.5485, 3 150 y 4 000 Hz 0.5000, 7 100 Hz 0.2070,
  8 000 Hz 0.4136, 9 000 Hz 0.6175, 10 000 Hz 0.7999, 12 500 Hz 1.1863,
  14 000 Hz 1.3825, 31 500 Hz 2.7865) y solo 16 000 Hz discrepa de lo
  impreso. No lo corrigen la Amendment 1:1988 ni la Amendment 2:1988.
  Verificado en la página 29 del PDF (p. 27 impresa) de IEC 60268-1:1985 y en
  la página 2 del PDF (p. 2 impresa) de la Recomendación ITU-R BS.468-4.
- **Comportamiento de la biblioteca:** no hizo falta ninguno. La máscara de
  tolerancias se toma de la Table 1 de BS.468-4, y la curva digital realizada
  se sujeta de todos modos a una cota mucho más estrecha que la máscara: la
  máscara gobierna un instrumento de medición compuesto por el amplificador y
  la red, no la desviación de un filtro respecto de la curva nominal.
- **Estado:** sin notificar.

## IEC 60268-3:2013, apartado 14.12.9.2 f) (denominador del DIM)

- **Ubicación:** apartado 14.12.9.2, punto f) (p. 39 impresa), la fórmula de
  la distorsión de intermodulación dinámica $d_\text{DIM}$.
- **El impreso:**
  $d_\text{DIM} = (\sum_{i=1}^{9} {U'_i}^{2})^{1/2} / U_2 \times 100\ \%$.
- **El problema:** el denominador es uno de los nueve términos de su propio
  numerador. La Table 2 del mismo apartado (p. 38 impresa) define $U_2$ como
  la componente de intermodulación a $f_s - 2f_q = 8{,}70\ \text{kHz}$, y el
  punto d) define $U_1, U_2, \ldots U_i$ como exactamente esas componentes,
  así que la suma $i = 1\ldots9$ recorre $U_1 \ldots U_9$ e incluye $U_2$.
  Entretanto, el apartado definitorio 14.12.9.1 declara la razón de la suma
  r.m.s. de las tensiones de productos de intermodulación de la Table 2 «to
  the amplitude of the output voltage at the frequency f_s», es decir, la
  componente senoidal de 15 kHz $U_s$, el convenio de Otala, y el punto d)
  mide «the amplitudes of the sinusoidal signal $U_s$» precisamente para que
  pueda usarse, cosa que la fórmula de f) no hace nunca. El denominador
  debería ser $U_s$. Una revisión anterior de esta entrada decía que «U2 se
  usa en todo 14.12 para la tensión total de salida»; eso es falso, tanto en
  el impreso inglés como en el francés.
- **Evidencia:** la Table 2, el punto d) y el punto f) leídos juntos en ambas
  columnas de idioma de la edición bilingüe; la literatura histórica del DIM
  (Otala) define la razón respecto a la amplitud del seno. Verificado en la
  página 41 del PDF (p. 39 impresa), la página 40 del PDF (p. 38 impresa),
  que lleva la Table 2, y la página 102 del PDF (p. 100 impresa), que lleva
  el mismo punto f) en la columna francesa, de IEC 60268-3:2013.
- **Comportamiento de la biblioteca:** sigue la definición de 14.12.9.1
  (referencia = la amplitud de salida a $f_s$), con un comentario en el
  código junto a la medición de referencia en
  [`distortion.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/electroacoustics/distortion.py).
- **Estado:** sin notificar.

## IEC 60268-16:2011, Table M.1 (la fila beta declara el término de redundancia equivocado)

- **Ubicación:** Annex M, Table M.1 «Example calculation», paso 4, la fila
  etiquetada «Sum of beta\*$MTI$ = $MTI_k$ $\times$ beta weighting» (p. 67
  impresa), justo debajo de la fila alfa homóloga.
- **El impreso:** la etiqueta lee $MTI_k \times \beta_k$, y las siete celdas
  de la fila leen 0,059 | 0,052 | 0,045 | 0,008 | 0,037 | 0,081 | 0,000,
  sumadas en la página siguiente como $\sum \text{beta*}MTI = 0{,}282$.
- **El problema:** la etiqueta y las celdas declaran cantidades distintas, y
  la equivocada es la etiqueta. El apartado A.5.6 de la misma edición (p. 47
  impresa) define el índice como
  $STI = \sum_{k=1}^{7} \alpha_k \times MTI_k -
  \sum_{k=1}^{6} \beta_k \times \sqrt{MTI_k \times MTI_{k+1}}$: el término de
  redundancia es la *media geométrica de dos bandas adyacentes*, no el
  $MTI_k$ de la propia banda. Leído contra la propia fila de MTI de la tabla,
  $\beta_k \times MTI_k$ da 0,062 | 0,051 | 0,044 | 0,008 | 0,036 | 0,076,
  sumando 0,277, que discrepa de cinco de las seis celdas impresas y del
  total impreso. $\beta_k\sqrt{MTI_k MTI_{k+1}}$ reproduce las seis celdas y
  el total de 0,282. La séptima celda no forma parte de ninguna de las dos
  lecturas: la suma de redundancia se detiene en $k = 6$ porque la banda de
  8 kHz no tiene banda por encima con la que emparejarse, así que su 0,000 es
  el marcador de posición de una columna sin pareja de redundancia, no un
  término. La fila alfa de encima, etiquetada de la misma manera, es
  correcta, porque allí la etiqueta y A.5.6 sí concuerdan.
- **Evidencia:** ambas lecturas recalculadas desde la propia fila de MTI del
  paso 4c de la tabla y comparadas celda a celda con la fila impresa y con su
  total impreso; A.5.6 leído contra la etiqueta. El defecto no mueve la
  respuesta de este ejemplo, ya que $1{,}040 - 0{,}277 = 0{,}763$ y
  $1{,}040 - 0{,}282 = 0{,}758$ se imprimen ambos como el STI 0,76 con el que
  termina la tabla, que es como una etiqueta que contradice la fórmula
  normativa sobrevive a un ejemplo resuelto. Verificado en la página 69 del
  PDF (p. 67 impresa) y la página 49 del PDF (p. 47 impresa) de
  IEC 60268-16:2011.
- **Comportamiento de la biblioteca:** implementa A.5.6 con el término de
  redundancia tal como allí está impreso, en
  [`_index_from_corrected_mtf`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/speech/sti.py); el test de
  factores de ponderación por pares de A.2.2 lo fija de forma independiente,
  y las filas de conformidad «IEC 60268-16:2020 A.2.2» e «IEC 60268-16 Annex
  M» leen ambas el índice que produce.
- **Estado:** sin notificar.

## IEC 60268-16:2011, Table M.1 (I_k tabulado un millón de veces sus vecinos)

- **Ubicación:** Annex M, Table M.1, la fila «Combined squared sound pressure
  $I_k$, MPa$^2$» del paso 2 (p. 64 impresa) y del paso 3 (p. 65 impresa),
  leída con las filas $I_{am,k}$ e $I_{rt,k}$ de debajo.
- **El impreso:** para la señal de 77,9 dB de la banda de 125 Hz, el paso 2
  imprime $I_k$ = 61,7, y cuatro filas más abajo imprime $I_{rt,k}$ = 40 000
  para el umbral de recepción de 46 dB de la misma banda.
- **El problema:** dos defectos en una fila. La unidad es imposible: a
  77,9 dB re 20 µPa la presión sonora al cuadrado es
  $0{,}0247\ \text{Pa}^2$, así que la celda no puede ser 61,7 MPa$^2$ bajo
  ninguna lectura del prefijo. Lo que la fila tabula en realidad es la razón
  de intensidades adimensional $10^{L/10} = 61\,722\,596$ dividida por
  $10^{6}$. Y ese divisor no se aplica a las dos cantidades que la norma suma
  a $I_k$ en las filas inmediatamente siguientes: $I_{am,k}$ e $I_{rt,k}$ se
  tabulan como la razón simple, siendo 40 000 el
  $10^{4,6} = 39\,811$ redondeado, sin dividir. Un lector que forme
  $I_k + I_{am,k} + I_{rt,k}$ desde las celdas tal como están impresas
  infravalora su primer término en $10^{6}$. La fila impresa «adjustment to
  remove masking and threshold» es la comprobación: 1,019 a 500 Hz es
  $(I_k + I_{am,k} + I_{rt,k})/I_k$ solo una vez que $I_k$ se restituye a
  26 305 192; formada desde las celdas tal como están impresas, la misma
  expresión lee 19 279.
- **Evidencia:** todas las celdas de ambas filas de $I_k$ recalculadas como
  $10^{L/10}$ desde los niveles combinados impresos encima, y todas las
  celdas de las filas $I_{am,k}$ e $I_{rt,k}$ recalculadas como
  $amf_k \times I_{k-1}$ y $10^{ART_k/10}$; el primer conjunto se reproduce a
  $10^{-6}$ del valor calculado y los otros dos a $10^{0}$. Verificado en la
  página 66 del PDF (p. 64 impresa) y la página 67 del PDF (p. 65 impresa) de
  IEC 60268-16:2011.
- **Comportamiento de la biblioteca:** lleva las tres cantidades en una sola
  escala, la razón simple a $p_0^2 = (20\ \mu\text{Pa})^2$, en la corrección
  de [`sti.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/speech/sti.py); la transcripción en
  [`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data) conserva las celdas
  impresas al pie de la letra y nombra el $10^{6}$ por el que las reescala, y
  la fila de conformidad «IEC 60268-16 Annex M» lee el ajuste que alimentan.
- **Estado:** sin notificar.

## IEC 60268-16:2011, Table M.1 (I_am,k del paso 3 a 250 Hz)

- **Ubicación:** Annex M, Table M.1, paso 3, la fila $I_{am,k}$, columna de
  250 Hz (p. 65 impresa).
- **El impreso:** 2 850 000, el mismo valor que la celda de 500 Hz de al
  lado.
- **El problema:** la celda no redondea desde la cantidad que nombra. Con los
  niveles operacionales impresos dos filas más arriba, $I_{am,k}$ a 250 Hz es
  el factor de enmascaramiento auditivo de la banda de 125 Hz por la
  intensidad combinada de esa banda,
  $0{,}01463507 \times 195\,339\,273 = 2\,858\,804$, que a las tres cifras
  significativas a las que se imprime la fila lee 2 860 000. La celda de
  500 Hz es correcta: su 2 852 252 sí se imprime como 2 850 000. Las dos
  celdas solo se reproducen juntas arrastrando el
  $amf \times 1000 = 14{,}6$ *redondeado* de la fila de arriba en lugar del
  propio factor, y el paso 2 demuestra que eso no es lo que hace la tabla, ya
  que sus dos celdas correspondientes se imprimen separadas, como 508 000 y
  507 000, cosa que solo da el factor sin redondear.
- **Evidencia:** ambas celdas recalculadas desde los niveles operacionales de
  habla y ruido impresos, y el par del paso 2 recalculado de la misma manera
  como control. El defecto no cambia nada aguas abajo: la corrección de
  enmascaramiento y umbral de la banda es 0,985552 con el valor correcto
  contra 0,985596 con el impreso, y la fila imprime 0,986 en cualquier caso.
  Verificado en la página 67 del PDF (p. 65 impresa) y la página 66 del PDF
  (p. 64 impresa) de IEC 60268-16:2011.
- **Comportamiento de la biblioteca:** calcula $I_{am,k}$ desde el factor de
  enmascaramiento sin redondear. La transcripción en
  [`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data) conserva la celda
  impresa y el test
  `test_annex_m_step3_masking_intensity_at_250_hz_is_the_printed_erratum`
  afirma el valor calculado contra 2 858 804 y contra el impreso, para que la
  única celda de la tabla que no es un oráculo no pueda convertirse en uno
  sin hacer ruido.
- **Estado:** sin notificar.

## UNE-EN 61043:1999, apartado 6.1 (el rango de frecuencias de la clase 2, perdido en la traducción)

- **Ubicación:** apartado 6.1 «Rango de frecuencias», la frase de la clase 2,
  de UNE-EN 61043 (abril de 1999), que se declara «la versión oficial, en
  español, de la Norma Europea EN 61043 de enero 1994, que a su vez adopta la
  Norma Internacional CEI 61043:1993».
- **El impreso:** una sola frase, «Los procesadores de clase 2 deberán
  cubrir, al menos, el rango desde 45 Hz a 5,6 kHz en bandas de octava.»
- **El problema:** el texto EN/IEC da a los procesadores de clase 2 dos
  rangos alternativos, no uno: «Class 2 processors shall, at least, cover the
  range from 45 Hz to 7,1 kHz in one-third octave bands, **or** the range
  from 45 Hz to 5,6 kHz in one octave bands» (BS EN 61043:1994, apartado
  6.1). La traducción pierde la primera alternativa. La omisión es normativa
  y no editorial: elimina una de las dos maneras de satisfacer el apartado
  6.1, y un lector solo del texto español concluiría que la clase 2 está
  *definida* sobre bandas de octava, de modo que una cadena de tercios de
  octava verificada sobre las 22 bandas tabuladas de 50 Hz a
  $6{,}3\ \text{kHz}$ no podría acreditar la clase 2 en todo su rango.
- **Evidencia:** lectura lado a lado del apartado 6.1 en ambos impresos. La
  frase de la clase 1 es equivalente palabra por palabra en los dos
  documentos, así que la divergencia queda confinada a la frase de la clase
  2. El impreso español además se contradice a sí mismo: su Tabla 2 tabula el
  índice presión-intensidad residual para procesadores de clase 2 en los 22
  centros de tercio de octava, y su Nota 2, traducida fielmente («Para
  procesadores con análisis en bandas de octavas únicamente, los requisitos
  se aplican únicamente a las frecuencias centrales de las bandas de
  octava»), aparta los procesadores de solo octavas como caso especial. Ambas
  son redundantes si todo procesador de clase 2 es de bandas de octava.
- **Comportamiento de la biblioteca:** implementa la lectura EN/IEC.
  [`verify_intensity_class`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/intensity_compliance.py)
  trata el conjunto completo de 22 bandas de tercio de octava como
  acreditación de cualquiera de las dos clases, y el conjunto de 7 bandas de
  octava (63 Hz a 4 kHz) como alternativa de clase 2 que nunca acredita la
  clase 1, con ambas ramas fijadas por tests de regresión
  ([`tests/emission/test_intensity_compliance.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_intensity_compliance.py)).
- **Estado:** sin notificar (traducción nacional, no el texto del organismo
  emisor).

## UNE-EN ISO 9614-1:2010, apartado 9.1 (el signo perdido de «signed magnitude» en la traducción)

- **Ubicación:** apartado 9.1, la lista de símbolos bajo la Fórmula (11)
  $P_i = I_{\mathrm{n}i} \cdot S_i$, de UNE-EN ISO 9614-1 (marzo de 2010),
  que se declara «la versión en español de la Norma Europea EN ISO
  9614-1:2009», la adopción europea de ISO 9614-1:1993.
- **El impreso:** «$I_{\mathrm{n}i}$ es el **módulo** de la componente de la
  intensidad acústica normal medida en la posición $i$ sobre la superficie de
  medida». El original ISO lee «$I_{\mathrm{n}i}$ is the **signed
  magnitude** of the normal sound intensity component measured at position
  $i$ on the measurement surface».
- **El problema:** *módulo* es el valor absoluto, así que el calificador que
  llevaba el signo ha desaparecido, y el signo es de lo que depende el resto
  del método. El impreso español se contradice entonces dos veces. El mismo
  apartado 9.1 da, dos párrafos por debajo de esa línea, la conversión a
  aplicar cuando el nivel de una posición se escribe $(-)\,XX$ dB:
  $I_{\mathrm{n}i} = -I_0 \times 10^{XX/10}$, un $I_{\mathrm{n}i}$ negativo.
  El apartado 3.6.1, que define la mismísima cantidad que la Fórmula (11)
  calcula, llama a $I_{\mathrm{n}i}$ «la componente normal, **con su signo**,
  de la intensidad acústica medida en la posición $i$», y A.2.3 lo llama «el
  **valor algebraico** de la componente de intensidad acústica normal». Y el
  apartado 9.2 hace de que $\sum_i P_i$ *sea negativa* la condición que deja
  una banda de frecuencia fuera del método, cosa que ninguna suma de módulos
  y áreas positivas puede ser jamás. Leído como módulo, el método pierde lo
  único para lo que sirve medir en puntos discretos: separar la energía que
  sale de la fuente de la energía que vuelve a entrar por parte de la
  superficie, que es lo que $F_3$ (Fórmulas (A.6) y (A.7)) y $F_4$ (Fórmulas
  (A.8) y (A.9)) están construidos para cuantificar desde la media algebraica
  del mismo $I_{\mathrm{n}i}$.
- **Evidencia:** los dos impresos de la misma lista de símbolos, puestos lado
  a lado, y los tres apartados españoles leídos unos contra otros. Páginas
  10, 18 y 22 del PDF (pp. 10, 18 y 22 impresas) de UNE-EN ISO 9614-1:2010;
  página 12 del PDF (p. 7 impresa) de ISO 9614-1:1993, donde el calificador
  está presente.
- **Comportamiento de la biblioteca:** implementa la lectura con signo en
  todo el recorrido, que es el texto ISO.
  [`sound_power_intensity_points`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_intensity_points.py)
  suma potencias parciales con signo, marca las bandas cuya suma no es
  positiva como fuera del método, y reporta $F_3 - F_2$ como el exceso que
  produce el flujo entrante; `normal_intensity_from_levels` lleva el $(-)$
  del impreso como argumento separado, porque el nivel impreso nunca lo
  contiene. Fijado por
  `test_a_genuinely_negative_partial_power_is_kept_and_summed` y los tests de
  conversión con signo en
  [`tests/emission/test_sound_power_intensity_points.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_intensity_points.py).
- **Estado:** sin notificar (traducción nacional, no el texto del organismo
  emisor: un lector que trabaje desde la edición ISO no tiene nada que
  sortear).

## UNE-EN ISO 9614-1:2010, apartado A.2.3 (barras de módulo sobre el nivel de intensidad algebraico)

- **Ubicación:** Anexo A, apartado A.2.3, la lista «donde» bajo la Fórmula
  (A.6) $F_3 = \overline{L_p} - \overline{L_{I_\mathrm{n}}}$.
- **El impreso:** la segunda entrada de la lista está compuesta
  $\overline{L_{|I_\mathrm{n}|}}$, con las barras de valor absoluto, y lee
  «es el valor algebraico del nivel de intensidad acústica superficial, en
  decibelios, calculado a partir de la ecuación (A.7)». La Fórmula (A.7),
  tres líneas más abajo en la misma página, está etiquetada
  $\overline{L_{I_\mathrm{n}}}$, sin las barras.
- **El problema:** el símbolo con barras es el de A.2.2, el nivel del
  *módulo* medio de la Fórmula (A.5), que es exactamente lo que resta $F_2$.
  Con las barras, $F_3$ y $F_2$ serían el mismo indicador y todo A.2.3 sería
  redundante; la frase junto al símbolo dice «valor algebraico» y apunta a
  (A.7), que toma la media algebraica. El original ISO imprime la misma
  entrada sin las barras y la describe como «the surface normal signed
  intensity level», así que las barras son composición propia de la
  traducción.
- **Evidencia:** el símbolo tal como está compuesto en las dos ediciones, y
  la (A.7) sin barras en la misma página que la entrada con barras. Página 22
  del PDF (p. 22 impresa) de UNE-EN ISO 9614-1:2010; página 15 del PDF (p. 10
  impresa) de ISO 9614-1:1993.
- **Comportamiento de la biblioteca:** no hizo falta ninguno.
  `field_indicators` en
  [`intensity.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/intensity.py) forma $F_3$ desde
  la media algebraica de la Fórmula (A.7) y $F_2$ desde el módulo medio de la
  Fórmula (A.5), que es lo que hace de $F_3 - F_2$ el exceso de flujo
  entrante sobre el que está escrita la compuerta del Anexo B. Registrado
  como defecto de etiqueta.
- **Estado:** sin notificar (traducción nacional, no el texto del organismo
  emisor).

## ISO 9614-1:1993, apartado B.1.3 ($F_4$ remitido a A.2.3, que define $F_3$)

- **Ubicación:** Anexo B, apartado B.1.3, la frase que introduce las dos
  evaluaciones separadas de $F_4$ que consume la Fórmula (B.4).
- **El impreso:** «Calculate indicator $F_4$ separately according to
  **A.2.3**», sobre los dos puntos «a) for the segment subset $N_\alpha$
  having total area $S_\alpha$, and» y «b) for the remaining segments». La
  edición española reproduce el mismo número de apartado: «Calcular el
  indicador $F_4$ separadamente de acuerdo al apartado A.2.3 para: a) el
  subconjunto de segmentos $N_\alpha$ con área total $S_\alpha$, y b) los
  segmentos restantes.»
- **El problema:** A.2.3 es «Negative partial power indicator», que define
  $F_3$ por las Fórmulas (A.6) y (A.7). $F_4$ es A.2.4, «Field
  non-uniformity indicator», Fórmulas (A.8) y (A.9). Seguida tal como está
  impresa, la referencia calcula el indicador equivocado para $F_4(\alpha)$ y
  $F_4(1-\alpha)$, y esos son los que dimensionan las posiciones nuevas en la
  Fórmula (B.4). Ambas ediciones llevan la misma numeración de apartados, así
  que el defecto es del organismo emisor.
- **Evidencia:** la referencia y los títulos de A.2.3 y A.2.4 leídos uno
  contra otro. Páginas 18 y 15 a 16 del PDF (pp. 13 y 10 a 11 impresas) de
  ISO 9614-1:1993; la misma frase en la página 24 del PDF (p. 24 impresa) de
  UNE-EN ISO 9614-1:2010.
- **Comportamiento de la biblioteca:** sigue el destino pretendido.
  $F_4(\alpha)$ y $F_4(1-\alpha)$ se calculan según A.2.4 en
  [`partial_power_concentration`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_intensity_points.py).
  La referencia no cambia ningún número que la biblioteca reporte, así que no
  hizo falta ningún otro cambio.
- **Estado:** sin notificar (defecto de referencia cruzada, sin consecuencia
  numérica).

## UNE-EN ISO 9614-1:2010, apartado 10.5 c) (un número de ecuación sustituido por un capítulo que no existe)

- **Ubicación:** apartado 10.5 c), «Datos acústicos», el requisito de informe
  que acompaña al nivel de una banda que no satisface el criterio 2.
- **El impreso:** «Una referencia a la incertidumbre prevista en el nivel de
  potencia acústica determinada para cada banda de frecuencia en la que no se
  satisfaga el criterio 2 del anexo B, **de acuerdo a la ecuación (véase el
  capítulo B.3)**.» El original ISO lee «A statement of the predicted
  uncertainty in the sound power level determined for each frequency band, in
  which criterion 2 of annex B is not satisfied, **according to equation
  (B.3)**.»
- **El problema:** el número que identificaba la ecuación se ha convertido en
  una referencia cruzada y ha cambiado por el camino. «De acuerdo a la
  ecuación ( )» no nombra ecuación alguna, y lo que el paréntesis nombra en
  su lugar no forma parte del documento: el Anexo B se divide en B.1, con
  B.1.1 a B.1.5, y B.2, y ahí se acaba, así que no hay capítulo B.3 que
  consultar. El requisito es inutilizable tal como está impreso a menos que
  el lector reconozca la Fórmula (B.3), el intervalo de confianza del 95 %
  $10 \lg (1 \pm 2 F_4 / \sqrt{N})$, que el apartado B.1.2 introduce con esta
  misma condición adjunta.
- **Evidencia:** los dos impresos del mismo punto, y las divisiones del Anexo
  B según corren sus títulos. Páginas 20 y 23 a 26 del PDF (pp. 20 y 23 a 26
  impresas) de UNE-EN ISO 9614-1:2010; página 14 del PDF (p. 9 impresa) de
  ISO 9614-1:1993, donde el número de ecuación está presente.
- **Comportamiento de la biblioteca:** reporta el intervalo de la Fórmula
  (B.3) para todas las bandas, así que la declaración que pide el apartado
  10.5 c) puede hacerse sobre cualquier banda que la necesite.
  `confidence_interval` en
  [`DiscretePointIntensityResult`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_intensity_points.py)
  lleva el par, y `criterion_2` dice a qué bandas aplica el requisito. El
  defecto no cambia ningún número, solo adónde se manda al lector a buscar la
  fórmula.
- **Estado:** sin notificar (traducción nacional, no el texto del organismo
  emisor).

## ISO 9614-1:1993, Tabla B.3 (las acciones c y d reclaman ambas $F_3 - F_2 = 1$ dB)

- **Ubicación:** Tabla B.3, «Actions to be taken to increase grade of
  accuracy of determination», las celdas de criterio de las filas de las
  acciones c y d.
- **El impreso:** la acción c está condicionada a «Criterion 2 not satisfied
  and 1 dB $\leq (F_3 - F_2) \leq$ 3 dB»; la acción d a «Criterion 2 not
  satisfied and $(F_3 - F_2) \leq$ 1 dB, and the procedure of 8.3.2 either
  fails or is not selected». Ambas desigualdades están impresas no estrictas,
  en las dos ediciones.
- **El problema:** las dos filas se solapan exactamente en
  $F_3 - F_2 = 1$ dB, donde la tabla prescribe dos acciones distintas para un
  mismo estado: aumentar la densidad de posiciones uniformemente (c), o
  alejar la superficie y conservar las posiciones (d). Una tabla de decisión
  normativa no es implementable mientras eso se mantenga. El documento lo
  zanja en otro lugar: el quinto rombo de decisión de la Figura B.1 es
  «$(F_3 - F_2) \leq$ 1 dB ?», y su rama **Yes** es la que lleva al
  procedimiento opcional y a la acción d, así que 1 dB pertenece a d y c
  empieza por encima. El apartado 8.3.2 concuerda, abriendo el procedimiento
  opcional «if $F_3 - F_2 \leq$ 1 dB».
- **Evidencia:** las dos celdas de criterio, el rombo y sus ramas, y la
  condición del apartado 8.3.2. Páginas 19, 20 y 12 del PDF (pp. 14, 15 y 7
  impresas) de ISO 9614-1:1993; los mismos tres lugares en las páginas 26, 27
  y 17 del PDF (pp. 26, 27 y 17 impresas) de UNE-EN ISO 9614-1:2010.
- **Comportamiento de la biblioteca:** sigue la Figura B.1 y el apartado
  8.3.2. `required_actions` en
  [`DiscretePointIntensityResult`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_intensity_points.py)
  responde a una banda que falla el criterio 2 con la acción c por encima de
  1 dB y la acción d a 1 dB y por debajo, fijado en el propio límite por
  `test_action_d_is_the_action_at_exactly_one_decibel` en
  [`tests/emission/test_sound_power_intensity_points.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_intensity_points.py).
- **Estado:** sin notificar.

## ISO 9614-1:1993, ecuaciones (A.1) y (A.8) (la intensidad normalizadora sin su barra)

- **Ubicación:** Anexo A, apartado A.2.1, ecuación (A.1) del indicador de
  variabilidad temporal $F_1$, y apartado A.2.4, ecuación (A.8) del indicador
  de no uniformidad del campo $F_4$.
- **El impreso:** ambas ecuaciones abren con el factor $1/I_\mathrm{n}$, un
  símbolo sin barra, mientras que la desviación dentro de la suma está
  escrita contra un $\overline{I_\mathrm{n}}$ claramente con barra:
  $F_1 = \frac{1}{I_\mathrm{n}} \sqrt{\frac{1}{M-1}\sum_k (I_{\mathrm{n}k} -
  \overline{I_\mathrm{n}})^2}$ y
  $F_4 = \frac{1}{I_\mathrm{n}} \sqrt{\frac{1}{N-1}\sum_i (I_{\mathrm{n}i} -
  \overline{I_\mathrm{n}})^2}$. Las dos ediciones las componen igual.
- **El problema:** las listas de símbolos que siguen definen solo el de la
  barra («$\overline{I_\mathrm{n}}$ is the mean value of $I_\mathrm{n}$ for
  $M$ short-time-average samples», A.2.1; «$\overline{I_\mathrm{n}}$ is the
  surface normal sound intensity calculated from equation (A.9)», A.2.4). El
  $I_\mathrm{n}$ sin barra es la intensidad normal en un punto del apartado
  3.4, así que tal como está impreso un coeficiente de variación se divide
  por un valor único sin especificar en lugar de por la media en torno a la
  cual se toma su propio numerador. Ambos indicadores son coeficientes de
  variación y no admiten otra normalización.
- **Evidencia:** las dos ecuaciones y las listas de símbolos bajo ellas,
  donde la barra falta sobre el divisor y está intacta sobre el símbolo de
  dentro de la suma. Páginas 15 y 16 del PDF (pp. 10 y 11 impresas) de
  ISO 9614-1:1993; las mismas dos ecuaciones en las páginas 21 y 22 del PDF
  (pp. 21 y 22 impresas) de UNE-EN ISO 9614-1:2010.
- **Comportamiento de la biblioteca:** no hizo falta ninguno. El coeficiente
  de variación tras `field_indicators` y `temporal_variability_indicator` en
  [`intensity.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/intensity.py) divide por la
  media algebraica, y rechaza una media que no sea positiva en lugar de
  dividir por ella. Registrado como defecto tipográfico.
- **Estado:** sin notificar (tipográfico).

## UNE-EN ISO 9614-1:2010, Nota 11 del apartado B.1.3 (la mitad de un nivel, y una recomendación vuelta requisito)

- **Ubicación:** Nota 11, inmediatamente después del bloque de la Fórmula
  (B.4) del apartado B.1.3, que matiza la elección del factor $C$ de la Tabla
  B.2 para una determinación ponderada A.
- **El impreso:** «Si la contribución total al **nivel de** potencia acústica
  ponderado A de las bandas de tercio de octava en el margen de frecuencias
  de 800 Hz a 5 000 Hz es menos de la mitad del **nivel total**, entonces
  **deben** usarse los valores de $C$ para las bandas de tercio de octava de
  200 Hz a 630 Hz.» El original ISO lee «If the total contribution to the
  A-weighted sound **power** from the one-third-octave bands in the frequency
  range 800 Hz to 5 000 Hz is less than half the total **power**, then the
  values of $C$ for the one-third-octave band 200 Hz to 630 Hz **should** be
  used.»
- **El problema:** dos desviaciones en una frase. La mitad de un *nivel* no
  es una operación definida, así que el impreso español declara una condición
  que no puede evaluarse tal como está escrita; el original condiciona sobre
  la mitad de la *potencia*, que es una contribución 3 dB o más por debajo
  del total y es decidible. Y *should*, una recomendación bajo las reglas de
  redacción ISO/IEC, se convierte en *deben*, que se lee como requisito, así
  que los dos impresos ni siquiera coinciden en si la sustitución es
  opcional.
- **Evidencia:** los dos impresos de la misma nota. Página 25 del PDF (p. 25
  impresa) de UNE-EN ISO 9614-1:2010; página 18 del PDF (p. 13 impresa) de
  ISO 9614-1:1993.
- **Comportamiento de la biblioteca:** implementa la lectura en potencia, y
  aplica la sustitución siempre que la condición se cumple en lugar de
  dejarla al llamante, lo que satisface ambos impresos. `_a_weighted_factor`
  en
  [`sound_power_intensity_points.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_intensity_points.py)
  compara la contribución ponderada A sumada de las bandas de 800 Hz a 5 kHz
  con la mitad de la contribución total y lee la fila de 200 Hz a 630 Hz de
  la Tabla B.2 cuando se queda corta.
- **Estado:** sin notificar (traducción nacional, no el texto del organismo
  emisor).

## ISO 3744:2010, 8.3.4, Ecuación (21) (un nivel integrado en el tiempo comparado con uno promediado en el tiempo)

- **Ubicación:** apartado 8.3.4, Ecuación (21) y la lista de símbolos que la
  sigue (página 31 del PDF, p. 25 impresa) de ISO 3744:2010, leída contra las
  definiciones de los apartados 3.3 y 3.4 (página 9 del PDF, p. 3 impresa). La
  misma construcción se imprime como Ecuación (25) de ISO 3741:2010 (página 33
  del PDF, p. 24 impresa, con su lista de símbolos en la página 34 del PDF,
  p. 25 impresa) como Ecuación (14) de ISO 3747:2010 (páginas 22 y 23 del
  PDF, pp. 13 y 14 impresas) y como Ecuación (15) de ISO 3746:2010 en su
  apartado 8.4.2 (página 25 del PDF, p. 16 impresa), que es la vía de grado
  *survey* que toma la biblioteca con `grade='survey'`.
- **El impreso:** $K_1 = -10 \lg\left(1 - 10^{-0{,}1\,\Delta L_E}\right)$ dB
  con $\Delta L_E = \overline{L'_{E(\mathrm{ST})}} - \overline{L_{p(\mathrm{B})}}$,
  donde $\overline{L'_{E(\mathrm{ST})}}$ «is the mean frequency-band or
  A-weighted single event time-integrated sound pressure level» y
  $\overline{L_{p(\mathrm{B})}}$ «is the mean frequency-band or A-weighted
  time-averaged sound pressure level of the background noise», seguido de
  «The integration time $T = t_2 - t_1$ and other measurement parameters shall
  be the same for the measurement of the single event time-integrated sound
  pressure level $L'_{Ei(\mathrm{ST})}$ and of the background noise level
  $L_{pi(\mathrm{B})}$.»
- **El problema:** los dos niveles no comparten magnitud de referencia, y la
  corrección resta una energía de otra. Por el apartado 3.4, $L_E$ es
  $\int_{t_1}^{t_2} p^2\,\mathrm{d}t$ re $E_0 = (20\ \mu\text{Pa})^2\,\text{s}$;
  por el apartado 3.3, $L_{p,T}$ es $\frac{1}{T}\int_{t_1}^{t_2} p^2\,\mathrm{d}t$
  re $p_0^2$. Su diferencia es un cociente de energías solo cuando $T = 1$ s.
  Sobre el intervalo común $T$ el fondo aporta la energía
  $L_{p(\mathrm{B})} + 10 \lg(T/T_0)$ (la identidad de la NOTA 1 del apartado
  3.4), así que el $\Delta L_E$ impreso supera al cociente de energías
  señal-fondo en $10 \lg(T/T_0)$ y $K_1$ se subestima para todo $T > 1$ s: una
  ráfaga cuya energía está 6 dB por encima de la del fondo en un intervalo de
  10 s se lee 16 dB por encima y no recibe corrección, donde el criterio de
  8.2.3 sitúa $K_1$ en su mayor valor admisible, 1,3 dB. La cadena gemela de
  8.2, donde ambos niveles están promediados en el tiempo, no tiene ese
  término, y 8.3.3 exige promediar los niveles de suceso aislado «in the same
  way as for the time-averaged sound pressure levels described in 8.2.2», así
  que la lectura pretendida es aquella bajo la cual las dos cadenas coinciden
  para una fuente estacionaria durante $T$, $L_J = L_W + 10 \lg(T/T_0)$, y es
  la lectura bajo la cual la insistencia en un único tiempo de integración
  para ambas mediciones sirve de algo.
- **Evidencia:** Verificado en la página 31 del PDF (p. 25 impresa) de
  ISO 3744:2010 para la ecuación y su lista de símbolos, y en la página 9 del
  PDF (p. 3 impresa) para las definiciones de los apartados 3.3 y 3.4 con la
  NOTA 1; la misma construcción leída en las páginas 33 y 34 del PDF (pp. 24 y
  25 impresas) de BS EN ISO 3741:2010 y en las páginas 22 y 23 del PDF (pp. 13
  y 14 impresas) de BS EN ISO 3747:2010.
- **Comportamiento de la biblioteca:** `sound_energy_pressure`,
  `sound_energy_reverberation` y `sound_energy_comparison` comparan el fondo
  como su exposición sobre el mismo intervalo,
  $L_{p(\mathrm{B})} + 10 \lg(T/T_0)$, y exigen `integration_time` con el fondo
  de la fuente bajo ensayo, que es el que se compara contra un nivel de suceso.
  La fuente de referencia de `sound_energy_comparison` es estacionaria, así que
  `background_levels_ref` se corrige con la regla promediada en el tiempo de
  9.1.2 y no lleva ventana; los criterios y el tope de 8.2.3 (y de 9.1.2 en ISO 3741) se
  aplican entonces a ese margen. `tests/emission/test_sound_energy.py` fija
  $K_1 = 1{,}2563$ dB para una ráfaga de 78 dB sobre un fondo de 62 dB en una
  ventana de 10 s, y $L_J = L_W + 10 \lg(T/T_0)$ campo a campo en ambas
  familias; el informe de conformidad lleva la identidad como «ISO 3744:2010
  Eq. 23 / clause 3.4 NOTE 1».
- **Estado:** sin notificar.

## ISO 3744:2010, 8.3.4 (la corrección llamada K_1i en el texto y K_1 en la Ecuación (21))

- **Ubicación:** apartado 8.3.4, primera frase y Ecuación (21) (página 31 del
  PDF, p. 25 impresa).
- **El impreso:** «The background noise correction, $K_{1i}$, shall be
  calculated using Equation (21):» seguido de
  $K_1 = -10 \lg\left(1 - 10^{-0{,}1\,\Delta L_E}\right)$ dB con $\Delta L_E$
  formado a partir de las dos medias sobre la superficie de medición,
  $\overline{L'_{E(\mathrm{ST})}}$ y $\overline{L_{p(\mathrm{B})}}$.
- **El problema:** la frase nombra una corrección por posición y la ecuación
  define una sola a partir de medias superficiales. El apartado gemelo 8.2.3
  nombra $K_1$ en ambos sitios y lo forma a partir de las mismas medias
  superficiales (Ecuación (16)), y 8.3.5 resta el $K_1$ sin subíndice en la
  Ecuación (22). El subíndice es la convención por micrófono de los apartados
  9.1.2 y 9.2.2 de ISO 3741:2010 ($K_{1i}$, Ecuaciones (14) y (25)), donde
  cada posición se corrige antes del promedio, y no pertenece a este
  apartado.
- **Evidencia:** Verificado en la página 31 del PDF (p. 25 impresa) de
  ISO 3744:2010, contra el apartado 8.2.3 en la página 29 del PDF (p. 23
  impresa).
- **Comportamiento de la biblioteca:** `sound_energy_pressure` forma un $K_1$
  por banda a partir de las medias superficiales, tal como lo imprime la
  Ecuación (21) y como hace `sound_power_pressure` con la Ecuación (16); no se
  aplica ninguna corrección por posición en la cadena de ISO 3744. No hizo
  falta ningún cambio.
- **Estado:** sin notificar.

## ISO/PAS 1996-3:2022, apartado 5 (referencias cruzadas de r y d)

- **Ubicación:** apartado 5, Fórmula (2), las definiciones de los símbolos de
  la prominencia
  $P = 3\log_{10}[r/(\text{dB/s})] + 2\log_{10}(d/\text{dB})$.
- **El impreso:** «r is the onset rate (OR) as defined in 3.4» y «d is the
  level difference (LD) as defined in 3.5».
- **El problema:** las dos referencias cruzadas están intercambiadas. Los
  propios términos y definiciones del documento fijan 3.4 como la *diferencia
  de niveles* LD («difference in decibels of L_pAF between the level of the
  end point L_e and the level of the starting point L_s of the onset») y 3.5
  como la *velocidad de aparición* OR («slope in decibels per second of the
  straight line that gives the best approximation to the onset»). Leída al
  pie de la letra, la Fórmula (2) tomaría tres veces el logaritmo de una
  diferencia de niveles más dos veces el logaritmo de una pendiente,
  invirtiendo los pesos que el método asigna a las dos cantidades. Los
  nombres desarrollados de la misma lista («the onset rate (OR)», «the level
  difference (LD)») y las unidades dadas para cada uno («dB/s» para $r$,
  «dB» para $d$) hacen inequívoca la lectura pretendida.
- **Evidencia:** lectura lado a lado de 3.4, 3.5 y la lista de símbolos del
  apartado 5; las unidades impresas con cada símbolo contradicen los números
  de apartado impresos con ellos.
- **Comportamiento de la biblioteca:** implementa la lectura desarrollada,
  ponderando la velocidad de aparición por 3 y la diferencia de niveles por 2
  (`predicted_prominence` en
  [`impulsive_sound.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/assessment/impulsive_sound.py)),
  que es además la forma de NT ACOU 112:2002 que el PAS arrastra.
- **Estado:** sin notificar.

## ISO 3744:2010, H.4.2.7 (la corrección por altitud y el divisor que lleva debajo)

- **Ubicación:** Anexo H (informativo), H.4.2.7 «Meteorological and radiation
  impedance corrections», el párrafo que dimensiona $u_{C_1+C_2}$ a partir de
  la corrección del Anexo G.
- **El impreso:** «At 120 m altitude and 23 °C the correction is zero and at
  500 m altitude the correction is 0,6 dB. Assuming a triangular distribution
  for this uncertainty, the standard deviation is
  $s_\mathrm{met} = 0{,}6/\sqrt{6} = 0{,}3\ \mathrm{dB}$».
- **El problema:** dos defectos independientes en un mismo par de frases.
  (a) El Anexo G, que es normativo y al que ese párrafo remite, da
  $C_1 + C_2 = 0{,}394$ dB a 500 m y 23,0 °C, no 0,6 dB. La lectura se valida
  a sí misma: esas dos mismas ecuaciones dan $-4{,}6 \times 10^{-5}$ dB a 120 m
  y 23,0 °C, que es el «cero» que imprime la misma frase, así que las
  constantes y los términos de temperatura se están leyendo como la norma
  pretende. Los 0,6 dB se alcanzan hacia los 697 m a 23,0 °C, o a 500 m solo
  si el aire está a 30,1 °C.
  (b) $0{,}6/\sqrt{6} = 0{,}245$, no 0,3. El cociente no da el resultado que
  se imprime a su lado: 0,3 dB es exactamente $0{,}6/2$, así que o el divisor
  o el resultado está mal. Para una distribución triangular de semianchura $a$
  la desviación típica es $a/\sqrt{6}$, que es el divisor que la frase nombra.
- **Evidencia:** H.4.2.7 leído en la página 82 del PDF (p. 73 impresa), contra
  las Ecuaciones (G.1) y (G.2) del Anexo G con $a = 2{,}2560 \times 10^{-5}$
  m$^{-1}$, $b = 5{,}2553$, $\theta_0 = 314$ K y $\theta_1 = 296$ K en las
  páginas 73 y 74 del PDF (pp. 64 y 65 impresas), todo de BS EN ISO 3744:2010.
  Ambos valores se recalcularon solo a partir de las ecuaciones impresas.
- **Comportamiento de la biblioteca:** el presupuesto de incertidumbre del
  Anexo H no está modelado, así que ningún número publicado depende de
  ninguna de las dos cifras. La corrección del Anexo G sí se evalúa desde las
  Ecuaciones (G.1) y (G.2) en
  [`reference_atmosphere_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power.py),
  y la comprobación de conformidad «ISO 3744:2010 Annex G / H.4.2.7» fija la
  mitad del párrafo que sí es correcta: la corrección se anula a 120 m y 23 °C.
- **Estado:** sin notificar.

## ISO 9613-2:1996, Tabla 2 (celda de 15 °C / 80 % / 1 kHz)

- **Ubicación:** Tabla 2, «Atmospheric attenuation coefficient α for octave
  bands of noise», fila de 15 °C / 80 % de humedad relativa, columna de
  1 kHz.
- **El impreso:** $\alpha = 4{,}1\ \text{dB/km}$.
- **El problema:** la Tabla 2 es un extracto redondeado de ISO 9613-1, a la
  que el propio apartado remite («For values of α at atmospheric conditions
  not covered in table 2, see ISO 9613-1»). Evaluar la fórmula de tono puro
  de ISO 9613-1 a 1 kHz, $15\ ^\circ\text{C}$, $80\ \%$ de HR y
  $101{,}325\ \text{kPa}$ da $4{,}1511\ \text{dB/km}$, que redondea a
  $4{,}2$, no al $4{,}1$ impreso. Las celdas vecinas de la misma fila
  redondean correctamente (2 kHz: $8{,}338$ -> impreso $8{,}3$; 4 kHz:
  $23{,}86$ -> $23{,}7$ en el centro de banda exacto), igual que las celdas
  de 1 kHz de las otras filas ($15\ ^\circ\text{C}$ / $50\ \%$: $4{,}164$ ->
  impreso $4{,}2$), así que el defecto queda confinado a esta celda.
- **Evidencia:** evaluación independiente del coeficiente de ISO 9613-1 a la
  frecuencia nominal y a la del centro de banda exacto
  ($4{,}1511\ \text{dB/km}$ en ambos casos, siendo 1 kHz las dos).
- **Comportamiento de la biblioteca:** no le afecta. La biblioteca nunca lee
  la Tabla 2: calcula $A_\text{atm}$ desde la fórmula de ISO 9613-1
  directamente
  ([`air_absorption.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/propagation/air_absorption.py)),
  así que da $4{,}15\ \text{dB/km}$ para esta condición.
- **Estado:** sin notificar.

## ISO/TR 17534-3:2015, Tabla 20 (q atribuida a la nota al pie equivocada de la Tabla 3 de la ISO 9613-2)

- **Ubicación:** Tabla 20, «Single number step by step results» del caso de
  prueba T08, la fila que nombra el factor de solape de la región media $q$.
- **Lo impreso:** `q (ISO 9613-2:1996, Table 3, footnote 1)`.
- **El problema:** la nota al pie 1 de la Tabla 3 de la ISO 9613-2:1996 trata
  de qué factor de suelo y qué altura toma cada región exterior («For
  calculating $A_s$, take $G = G_s$ and $h = h_s$...»). No dice nada de $q$. El
  factor $q$ lo define la nota al pie 2 de esa misma tabla, que es adonde la
  propia guía manda al lector en los otros cuatro sitios donde imprime la fila:
  la Tabla 3 (T01), la Tabla 8 (T04), la Tabla 14 (T06) y la Tabla 22 (T09)
  dicen todas «Table 3 footnote 2». La Tabla 20 es la única ocurrencia que dice
  footnote 1, y el valor que lleva, $q = 0{,}23$, es el que produce la nota 2.
- **Evidencia:** verificado en la página 23 del PDF (p. 17 impresa) de
  ISO/TR 17534-3:2015, junto con las cuatro ocurrencias coherentes de esa misma
  fila en las páginas 13, 16, 20 y 41 del PDF (pp. 7, 10, 14 y 35 impresas) de
  esa misma edición; las dos notas al pie a las que apunta se leyeron en la
  página 10 del PDF (p. 8 impresa) de ISO 9613-2:1996.
- **Comportamiento de la biblioteca:** no le afecta. El desliz tipográfico está
  en una referencia cruzada, no en un número, y
  [`ground_attenuation`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/propagation/outdoor_propagation.py)
  implementa $q$ a partir de la nota 2, que es lo que reproduce el 0,23
  impreso.
- **Estado:** sin notificar.

## VDI 2081 Blatt 1:2001, apartado 6.7.3 (la lista de símbolos de la ecuación (36) remite A a la propia ecuación (36))

- **Ubicación:** apartado 6.7.3, la lista de símbolos bajo la ecuación (36), la
  entrada del área de absorción equivalente ``A``.
- **Lo impreso:** «A  äquivalente Absorptionsfläche; in m², Gleichung (36)» /
  «A  is the equivalent absorption area; in m², Equation (36)».
- **El problema:** la ecuación (36) es la ecuación de nivel a la que pertenece
  la lista, $L_P = L_W + 10\lg[Q/(4\pi r^2) + 4/A]$, en la que ``A`` es un
  dato de entrada. No define ``A``. La guía lo define dos veces más abajo en el
  mismo apartado: la ecuación (37), $A = 0{,}163\,V/T$, y la ecuación (39),
  $A = \sum \alpha_i S_n + \sum A_n$. La referencia es circular, y está en las
  dos columnas de idioma, así que es un desliz de composición del original y no
  de la traducción.
- **Evidencia:** verificado en la página 43 del PDF (p. 43 impresa) de la
  VDI 2081 Blatt 1:2001-07, con las ecuaciones (37) y (39) en la página 44 del
  PDF (p. 44 impresa) de la misma tirada.
- **Comportamiento de la biblioteca:** no le afecta. El desliz está en una
  referencia cruzada, no en un número:
  [`room_effect`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) toma ``A`` como
  argumento y
  [`sabine_absorption_area`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/steady_field.py)
  implementa la ecuación (37).
- **Estado:** sin reportar.

## VDI 2081 Blatt 1:2001, apartado 6.7.3 (la columna inglesa llama esférica a una propagación semiesférica)

- **Ubicación:** apartado 6.7.3, la frase que dice dónde empieza el campo
  reverberante, justo después de la ecuación (36b).
- **Lo impreso:** en alemán, «Der Nachhallbereich beginnt bei
  **halbkugelförmiger** Schallausbreitung in einer Entfernung, die größer ist
  als $r_H = 0{,}2\sqrt{A}$»; en inglés, «The reverberation area begins as a
  **spherical** sound propagation at a distance which is greater than
  $r_H = 0.2\sqrt{A}$».
- **El problema:** *halbkugelförmig* es semiesférica, no esférica, y la
  constante impresa da la razón al alemán. El radio de reverberación es
  $r_H = \sqrt{Q A / 16\pi}$, que vale $0{,}199\sqrt{A}$ con la $Q = 2$ de un
  semiespacio y $0{,}141\sqrt{A}$ con la $Q = 1$ del espacio entero. Sólo la
  primera redondea al $0{,}2$ impreso. Quien siga la columna inglesa tomará
  $0{,}2\sqrt{A}$ por el radio esférico y situará el campo reverberante un 41 %
  más lejos de lo que toca.
- **Evidencia:** verificado en la página 44 del PDF (p. 44 impresa) de la
  VDI 2081 Blatt 1:2001-07, leyendo las dos columnas de la misma frase una al
  lado de la otra.
- **Comportamiento de la biblioteca:** no le afecta.
  [`critical_distance`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/steady_field.py) toma ``Q`` como
  argumento y deja dicha la lectura semiesférica en su propio texto.
- **Estado:** sin reportar.

## ANSI S3.5-1997, ejemplos resueltos del Anexo C (erratas oficiales del WG S3-79)

> **Sin verificar contra la página.** ANSI S3.5-1997 no está en la biblioteca
> local (se tiene la viñeta del paquete de R `SII`, no la norma), así que lo
> que esta entrada llama «el impreso» es la descripción del propio grupo de
> trabajo, no una página que este proyecto haya leído. Los recálculos de
> abajo son independientes y sí se reproducen, pero los *caracteres impresos*
> descansan solo en la lista de erratas. La norma está en la lista de
> adquisiciones pendientes del mantenedor; cuando llegue una copia, la
> entrada debe re-verificarse contra el impreso de las pp. 21-22 impresas y
> retirarse este aviso.

- **Ubicación:** Anexo C, Tabla C.1 (ejemplo resuelto en bandas de octava,
  p. 21) y Tabla C.2 (ejemplo resuelto en tercios de octava, p. 22) de la
  impresión de 1997.
- **El impreso (según las erratas del grupo de trabajo):** (a) Tabla C.1,
  fila $i = 5$, el factor de distorsión por nivel $L_i$ bajo el Step 6 está
  impreso como $0.10$; (b) Tabla C.2, primera fila, la pendiente de
  autoenmascaramiento del habla $C_i$ está impresa como $-45.59$.
- **El problema:** ambas celdas contradicen las propias fórmulas normativas
  de la norma. (a) El apartado 5.7 con las entradas del ejemplo
  ($E'_5 = 20\ \text{dB}$, $U_5 = 9.33\ \text{dB}$) da
  $L_5 = 1 - (20 - 9.33 - 10)/160 = 0.9958$, que a dos decimales se imprime
  $1.00$, no $0.10$. (b) El apartado 5.4 con las entradas del ejemplo
  ($B_1 = 40\ \text{dB}$, $f_1 = 160\ \text{Hz}$) da
  $C_1 = -80 + 0.6 (40 + 10\log_{10} 160 - 6.353) = -46.587$, que se imprime
  $-46.59$, no $-45.59$; la columna $Z_i$ del ejemplo solo es consistente con
  la pendiente corregida ($Z_2$ recalcula a $34.658$ = impreso 34.66 dB,
  mientras que la pendiente con la errata daría 34.76 dB). El ejemplo de la
  Tabla C.1 es el procedimiento de bandas de octava y el de la Tabla C.2 el
  de tercios de octava, así que hay una celda afectada de cada uno.
- **Evidencia:** la lista oficial de erratas publicada por el grupo de
  trabajo S3-79 de la ASA, el comité que mantiene ANSI S3.5, en su sitio de
  soporte (sii.to): «Page 21, Table C1, row i=5, column Li under Step 6: the
  value printed as 0.10 should be changed to 1.00» y «Page 22, Table C2, the
  first row of numbers, value −45.59 should be −46.59»; más el recálculo
  independiente de ambas celdas desde los apartados normativos (arriba). La
  misma lista lleva cinco correcciones más (la grafía de una referencia, la
  redacción de los pies de las Tablas 1-4 registrada en la entrada siguiente,
  la ganancia de inserción $G_i$ que falta en la Ec. 23, y dos arreglos del
  Anexo B, una referencia cruzada «B16» que debería leer «B15» y un cambio de
  redacción sobre la aproximación audiovisual); ninguna de ellas toca una
  fórmula que esta biblioteca implemente. La fuente es la lista de erratas
  del WG S3-79 en sii.to/html/errata.html (capturada el 2026-07-30,
  re-comprobada en vivo el 2026-08-04). No es la página impresa y no puede
  sustituirla, que es por lo que esta entrada lleva el aviso de arriba.
- **Comportamiento de la biblioteca:** no le afecta; la biblioteca calcula
  los valores corregidos desde los apartados normativos y siempre lo hizo.
  Sus anclas del Anexo C.2
  ([`tests/reference_data/`](https://github.com/jmrplens/phonometry/blob/main/tests/reference_data),
  `ANSIS3_5_ANNEX_C1*` y `ANSIS3_5_ANNEX_C2*`) fijan la cadena consistente
  con las erratas de ambos ejemplos, contrastada a doble precisión con la
  propia implementación de referencia del grupo de trabajo `SII.C` y sus
  resultados de casos de prueba publicados. La celda de la Tabla C.1 está
  fijada directamente: el factor de distorsión por nivel del apartado 5.7
  para la fila $i = 5$ del ejemplo de bandas de octava del Anexo C.1 calcula
  a $0.99581$, que se imprime como el $1.00$ corregido.
- **Estado:** correcciones publicadas por el grupo de trabajo emisor; nada
  que notificar aguas arriba.

## ANSI S3.5-1997, pies de las Tablas 1 a 4 (errata oficial del WG S3-79)

> **Sin verificar contra la página.** Como en la entrada anterior,
> ANSI S3.5-1997 no está en la biblioteca local, así que la redacción de los
> cuatro pies se toma de la lista de erratas del grupo de trabajo y no de una
> página que este proyecto haya leído. El argumento de que las tablas no
> llevan columna de umbral es independiente y sí se sostiene contra las
> constantes transcritas. Re-verificar contra el impreso de las pp. 3-5
> impresas cuando se adquiera la norma.

- **Ubicación:** los pies de las Tablas 1, 2, 3 y 4 (pp. 3-5 de la impresión
  de 1997), las tablas de constantes de los cuatro procedimientos por bandas:
  banda crítica (21 bandas), banda crítica de contribución igual (17 bandas),
  tercio de octava (18 bandas) y octava (6 bandas).
- **El impreso (según las erratas del grupo de trabajo):** cada pie lista las
  cantidades que la tabla tabula e incluye la locución «hearing threshold
  levels,».
- **El problema:** ninguna de las cuatro tablas tabula un nivel de umbral de
  audición. Cada una lleva la frecuencia central de banda (y, para las Tablas
  1, 2 y 4, los límites de banda), la función de importancia de banda $I_i$,
  el nivel espectral estándar del habla $U_i$ por esfuerzo vocal y el nivel
  espectral de ruido interno de referencia $X_i$. El nivel de umbral de
  audición $T'_i$ es una *entrada del usuario* del procedimiento (apartado
  5.5, donde el nivel espectral de ruido interno equivalente es
  $X'_i = X_i + T'_i$), que es exactamente la cantidad que el pie invita al
  lector a buscar en la tabla y a confundir con $X_i$.
- **Evidencia:** la lista oficial de erratas publicada por el grupo de
  trabajo S3-79 de la ASA, el comité que mantiene ANSI S3.5, en su sitio de
  soporte (sii.to): «Pages 3-5, Tables 1-4: In each of the **figure**
  captions the phrase 'hearing threshold levels,' should be deleted» (la
  lista de erratas del WG S3-79 en sii.to/html/errata.html, capturada el
  2026-07-30, re-comprobada en vivo el 2026-08-04; una revisión anterior de
  esta entrada omitía la palabra «figure» de la cita); más las propias
  tablas, que no tienen tal columna.
- **Comportamiento de la biblioteca:** no le afecta. Las cuatro tablas están
  implementadas con las columnas que de verdad llevan, expuestas por
  procedimiento por `sii_procedure()` como `band_importance`,
  `speech_spectrum` ($U_i$) e `internal_noise` ($X_i$), y el umbral de
  audición sigue siendo el argumento `threshold=` de
  `speech_intelligibility_index`
  ([`src/phonometry/speech/sii.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/speech/sii.py)).
- **Estado:** corrección publicada por el grupo de trabajo emisor; nada que
  notificar aguas arriba.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), Ec. (27)

- **Ubicación:** sección A.4.2, Ec. (27) (coeficiente de absorción
  atmosférica) y la frase que define sus símbolos, p. 21 impresa.
- **El impreso:** la Ec. (27) empareja el coeficiente
  $6.6928 \cdot 10^{-6}$ con $f_{rO}$ y $1.3415 \cdot 10^{-6}$ con $f_{rN}$,
  y la frase de debajo lee «the variables f_rN = 75692 Hz and f_rO = 630.7 Hz
  represent the vibrational relaxation frequencies of oxygen and nitrogen
  respectively».
- **El problema:** los dos subíndices están intercambiados en la frase de
  definición. Los *valores* casan con los *nombres* que les da (75 692 Hz es
  la frecuencia de relajación del oxígeno y 630.7 Hz la del nitrógeno en las
  condiciones de referencia), pero están asignados a los símbolos opuestos,
  así que la ecuación tal como está impresa multiplica el coeficiente del
  oxígeno por la frecuencia de relajación del nitrógeno y viceversa. Evaluada
  así da 14.2 dB/km a 500 Hz contra el valor de 3.1 dB/km de la propia Tabla
  4 de la guía; con $f_{rO}$ y $f_{rN}$ intercambiados da 3.07 dB/km,
  reproduciendo la Tabla 4 y el coeficiente de tono puro de ISO 9613-1 a
  0.02 dB/km. Una revisión anterior de esta entrada citaba el valor impreso
  como 14.3 dB/km y planteaba el defecto como un emparejamiento erróneo de
  los coeficientes y no como subíndices intercambiados en la definición.
- **Evidencia:** evaluación numérica de la Ec. (27) con la asignación impresa
  y con la asignación intercambiada, contra la celda de 500 Hz de la Tabla 4
  de la misma página. Verificado en la página 20 del PDF (p. 21 impresa) de
  NORAH2 SC01.D1.5d (EASA.2020.FC.06):2024.
- **Comportamiento de la biblioteca:** implementa el emparejamiento correcto;
  el docstring del módulo lleva una nota defensiva para que la errata no se
  transcriba como «arreglo».
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), Ec. (21)

- **Ubicación:** sección A.3.3, Ec. (21) (ángulo de la trayectoria de vuelo).
- **El impreso:** $\gamma = \text{acos}(\Delta Z/\Delta S)$.
- **El problema:** el arcocoseno de la razón de ascenso a trayectoria
  devuelve el complemento del ángulo de trayectoria ($90^\circ$ en vuelo
  nivelado, donde $\gamma$ debe ser $0^\circ$) y contradice el propio uso de
  $\gamma$ como ángulo de ascenso/descenso en toda la sección A.3. El Doc 32
  de la CEAC, 1.ª ed., Ec. (10), imprime la forma correcta,
  $\gamma = \text{atan}(\Delta Z/\Delta S)$ con el $\Delta S$ horizontal de
  su Ec. (8).
- **Evidencia:** evaluación en vuelo nivelado; contraste contra la Ec. (10)
  del Doc 32 y contra los ficheros de entrada del prototipo NORAH2, cuyas
  columnas ``Vang`` son ángulos de ascenso/descenso ($0^\circ$ en segmentos
  nivelados).
- **Comportamiento de la biblioteca:** ``flight_path_kinematics`` implementa
  la forma ``atan`` del Doc 32; el docstring del resultado lleva la nota
  defensiva.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), triangulación del §A.3.1

- **Ubicación:** sección A.3.1, pasos 2 a 4 (interpolación de condiciones de
  vuelo), contra las tablas de consulta de triangulación distribuidas con la
  base de datos NORAH2 (``*_triangulation.int``).
- **El impreso:** los pasos 2 y 3 normalizan las condiciones de la base de
  datos (rangos, con $F_{fc} = 2$ en el ángulo de trayectoria) y el paso 4
  calcula «the Delaunay triangulation for the database flight conditions
  γ̄_j and V̄_j», es decir, la de los puntos normalizados, ofreciendo una
  tabla de consulta como equivalente.
- **El problema:** las tablas de consulta distribuidas con la base de datos
  (que la guía dice que forman parte de los datos de hemisferios y no deben
  editarse) son la triangulación de Delaunay de las condiciones $(V, \gamma)$
  brutas, no de las normalizadas: para el conjunto del R22, 14 de los 27
  triángulos distribuidos difieren de la triangulación de Delaunay de las
  condiciones normalizadas. Una triangulación de Delaunay no es invariante
  bajo la normalización anisótropa, así que las dos prescripciones
  seleccionan triángulos envolventes distintos para parte de la envolvente.
  Los pesos de distancia de las Ecs. (7)/(8) sí usan las coordenadas
  normalizadas en el prototipo (verificado contra sus salidas mezcladas).
- **Evidencia:** recálculo de ambas triangulaciones para la base de datos del
  R22; reproducción bin a bin de la selección de hemisferios por paso del
  prototipo con las tablas distribuidas, y de sus niveles mezclados con pesos
  en el espacio normalizado, a 0.05 dB.
- **Comportamiento de la biblioteca:** ``flight_condition_weights`` sigue el
  método impreso (Delaunay de las condiciones normalizadas) por defecto y
  acepta la tabla de consulta de la base de datos vía ``triangles``, que
  reproduce la implementación de referencia exactamente.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), Ec. (46)

- **Ubicación:** sección A.4.5, Ec. (46) (efecto de suelo del lado de la
  fuente ponderado por la difracción).
- **El impreso:** el exponente de ponderación lee
  $(\Delta L_{g,s'} - \Delta L_{d,s})/20$.
- **El problema:** no existe término alguno $\Delta L_{g,s'}$; la prosa
  justo debajo de la ecuación define $\Delta L_{d,s'}$ como «the attenuation
  due to the diffraction between the image source S′ and R», la Ec. (47)
  compañera del lado del receptor imprime el término paralelo correctamente
  como $\Delta L_{d,r'}$, y el método CNOSSOS-EU en el que se basa la sección
  escribe $\Delta_\text{ground}(S,O)$ con $\Delta_\text{dif}(S',R)$ en esa
  posición. El subíndice $g$ es una errata por $d$.
- **Evidencia:** consistencia interna de la sección (su propia prosa y la
  Ec. (47)) y la fuente CNOSSOS-EU de las ecuaciones.
- **Comportamiento de la biblioteca:** implementa el término de difracción de
  la fuente imagen $\Delta L_{d,s'}$ tal como lo define la prosa.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), referencias cruzadas del §A.4.5

- **Ubicación:** sección A.4.5, las definiciones bajo la Ec. (46) (p. 32
  impresa) y la Ec. (47) (p. 33 impresa).
- **El impreso:** cuatro referencias cruzadas a la eq. 44, en **tres**
  redacciones distintas: «calculated as per eq. 44» para $\Delta L_{d,s'}$ y
  de nuevo para $\Delta L_{d,s}$ bajo la Ec. (46); «calculated as in eq. 44»
  para $\Delta L_{d,r'}$ bajo la Ec. (47); y «calculated as in Subsection
  eq. 44» para $\Delta L_{d,s}$ bajo la Ec. (47). Una revisión anterior de
  esta entrada citaba las cuatro con la primera redacción.
- **El problema:** la Ec. (44) es el coeficiente de difracción múltiple
  $C''$; la atenuación por difracción es la Ec. (42). Las cuatro referencias
  cruzadas apuntan al coeficiente auxiliar en lugar de a la fórmula que
  describen, y la cuarta lleva además un «Subsection» colgado sin número de
  subsección detrás.
- **Evidencia:** los términos son atenuaciones en dB, que solo produce la
  Ec. (42); la Ec. (44) es un coeficiente adimensional que consume la
  Ec. (42). Verificado en las páginas 31 y 32 del PDF (pp. 32 y 33 impresas)
  de NORAH2 SC01.D1.5d (EASA.2020.FC.06):2024.
- **Comportamiento de la biblioteca:** evalúa los términos de difracción del
  camino imagen y directo con la Ec. (42), usando la Ec. (44) para $C''$
  dentro de ella.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), §A.3.5 Approach 3 (base del ralentí a régimen pleno)

- **Ubicación:** sección A.3.5, Approach 3, paso 3 (p. 18 impresa), contra la
  fila «Fl. idle» de la Tabla 3 (pp. 18-19 impresas).
- **El impreso:** el paso lee «add offset of 12 dB\* to derive out of ground
  hover from the in-ground hover disk, -12 dB\* to derive reduced-rpm idle
  from in-ground hover disk, and -2.5 dB\* to derive full-rpm idle from out
  of ground hover»; la tabla imprime
  $LA_{\mathrm{FL.idle}}(\theta) = LA_{\mathrm{HIGE}}(\theta) - 2.5\ \mathrm{dB}^*$.
- **El problema:** la prosa deriva el ralentí a régimen pleno del
  estacionario fuera del efecto suelo donde la tabla lo deriva del
  estacionario en efecto suelo, y las dos prescripciones caen a 12 dB una de
  otra (por la prosa,
  $LA_{\mathrm{HOGE}}(\theta) - 2.5 = LA_{\mathrm{HIGE}}(\theta) + 9.5$; por
  la tabla, $LA_{\mathrm{HIGE}}(\theta) - 2.5$). Solo la tabla conserva el
  orden físico de las condiciones (ralentí a régimen pleno por encima del
  ralentí a régimen reducido, ambos por debajo del estacionario en efecto
  suelo). El párrafo que introduce estas fases, al final de la sección A.3.3
  (p. 17 impresa), está él mismo sin terminar («For specific phases of a
  flight such as, turns, hover, taxiing»), apuntando a una pasada de edición
  que la sección no recibió.
- **Evidencia:** las correcciones distribuidas con la base de datos pública
  V2.0.74 son todas relativas al disco de estacionario en efecto suelo
  (`Fullrpmidle -2` en el fichero de consulta de interpolación de todos los
  tipos), concordando con la tabla y no con la prosa. Verificado en las
  páginas 16, 17 y 18 del PDF (pp. 17, 18 y 19 impresas) de NORAH2
  SC01.D1.5d (EASA.2020.FC.06):2024.
- **Comportamiento de la biblioteca:** `hover_derived_hemisphere` aplica
  todos los desplazamientos de la Tabla 3 desde el hemisferio de estacionario
  en efecto suelo, como imprime la tabla; el docstring declara la condición
  base explícitamente.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), asignación de rodaje del §A.3.5

- **Ubicación:** sección A.3.5, último párrafo (p. 19 impresa).
- **El impreso:** «To include taxiing for helicopters with and without wheels
  into the noise calculation the measured and derived hemispheres for
  in-ground hover and full-rpm idle respectively should be employed.»
- **El problema:** leído al pie de la letra, el «respectively» empareja el
  helicóptero con ruedas con la fuente de estacionario en efecto suelo y el
  sin ruedas con el ralentí a régimen pleno, que es lo contrario de las
  operaciones que modela: un helicóptero sin ruedas solo puede rodar en
  estacionario en efecto suelo, y uno con ruedas rueda por el suelo sobre sus
  ruedas con el rotor a ralentí gobernado, sin producir sustentación. Las dos
  listas se leen transpuestas. Ningún oráculo lo zanja (la publicación
  pública no distribuye caso de verificación de rodaje), así que el
  emparejamiento se corrige solo desde la física de las operaciones.
- **Evidencia:** comparación interna de las dos listas de la prosa contra las
  operaciones que nombran. Verificado en la página 18 del PDF (p. 19 impresa)
  de NORAH2 SC01.D1.5d (EASA.2020.FC.06):2024.
- **Comportamiento de la biblioteca:** ninguna función se ve afectada (la
  regla selecciona entre dos hemisferios que el lector ya ha construido); la
  guía de rotorcraft documenta el emparejamiento físico, rodaje sin ruedas
  sobre el hemisferio de estacionario en efecto suelo y rodaje con ruedas
  sobre el de ralentí a régimen pleno, con esta salvedad.
- **Estado:** sin notificar.

## Guía de rotorcraft NORAH2 SC01.D1.5d (EASA.2020.FC.06), desplazamientos de la Tabla 3 frente a las correcciones distribuidas

- **Ubicación:** Tabla 3, columna de Approach 3 (pp. 18-19 impresas), contra
  el bloque `&CORRECTIONS` de los ficheros de consulta de interpolación
  distribuidos con la publicación pública NORAH2 V2.0.74.
- **El impreso:** desplazamientos de +12 dB\* (estacionario fuera del efecto
  suelo), -12 dB\* (ralentí a régimen reducido) y -2.5 dB\* (ralentí a
  régimen pleno) desde el disco de estacionario en efecto suelo, con la nota
  del asterisco de que se derivaron de mediciones con micrófonos invertidos
  sobre placas de suelo y «may not be valid for other microphone setups».
- **El problema:** la base de datos de referencia sobre la que se construye
  la guía distribuye valores distintos: cada uno de los once ficheros de
  consulta de triangulación por tipo (``*_triangulation.int``) de la
  publicación pública lleva `Corr_dB` 8, -10 y -2 para las mismas tres
  operaciones, así que las constantes publicadas y la base de datos discrepan
  en 4, 2 y 0.5 dB. La guía, cuya sección A.3.1 declara los datos de consulta
  distribuidos parte de la base de datos de hemisferios y no editables, no
  menciona la diferencia, y su nota cuestiona la validez de los valores
  publicados sin nombrar los que de verdad se distribuyen.
- **Evidencia:** los bloques `&CORRECTIONS` idénticos de los once ficheros de
  consulta de triangulación (``*_triangulation.int``) de la publicación
  pública V2.0.74; las constantes publicadas verificadas en las páginas 17 y
  18 del PDF (pp. 18 y 19 impresas) de NORAH2 SC01.D1.5d
  (EASA.2020.FC.06):2024.
- **Comportamiento de la biblioteca:** `hover_derived_hemisphere` usa por
  defecto las constantes publicadas de la Tabla 3 y acepta una corrección
  medida o de base de datos como ``offset_db``; el caso de verificación de
  estacionario de extremo a extremo pasa el +8 dB de la base de datos
  explícitamente, y el docstring registra la divergencia.
- **Estado:** sin notificar.

## RANDI 3.1 Physics Description (NRL, Breeding et al.), Tabla 2

- **Ubicación:** Tabla 2 (niveles de fuente de buques representativos).
- **El impreso:** dos celdas se desvían de las propias Ecs. (2) a (5) del
  informe evaluadas con las longitudes y velocidades medias de la Tabla 1: el
  valor de Merchant a 25 Hz (unos 3 dB alto) y el de Tanker a 300 Hz (en
  torno a 1 dB bajo). La fila de Fishing Vessel no es reproducible desde las
  medias de la Tabla 1 en absoluto (un desplazamiento constante de unos
  3.8 dB sugiere entradas supuestas distintas).
- **El problema:** el informe no declara las entradas exactas usadas para la
  Tabla 2, y dos celdas contradicen sus propias ecuaciones mientras que todas
  las celdas de Large Tanker y Super Tanker concuerdan a 0.06 dB.
- **Evidencia:** recálculo de las 25 celdas desde las Ecs. (2) a (5).
- **Comportamiento de la biblioteca:** el test de regresión fija las filas
  reproducibles y excluye las celdas contradictorias con la justificación en
  el test.
- **Estado:** sin notificar (informe técnico, no una norma).

## Osses, García & Kohlrausch (2016), modelo de intensidad de fluctuación, Ec. (3)

- **Ubicación:** Ec. (3), la transformación a razón de banda crítica (Bark)
  del frontal de patrón de excitación.
- **El impreso:**
  $z(f) = 13 \cdot \arctan(0.76 \cdot 10^{-4} \cdot f) + 3.5 \cdot \arctan((f/7500)^2)$.
- **El problema:** el primer coeficiente es el $0.76 \cdot 10^{-3}$ de
  Zwicker-Terhardt con el exponente mal impreso. Las propias anclas del
  artículo desmienten el impreso: declara
  $0.5\ \text{Bark} = 50\ \text{Hz}$ y
  $23.5\ \text{Bark} = 13.2\ \text{kHz}$ (sección 2.1.2) y
  $15\ \text{Bark} = 2.7\ \text{kHz}$ (sección 3.1), todas las cuales exigen
  $10^{-3}$. Con $10^{-4}$, $z(1\ \text{kHz}) = 1.05$ en lugar de
  $8.51\ \text{Bark}$ y los 47 centros de filtro del modelo abarcarían de
  491 Hz a 20 kHz en lugar de 50 Hz a 13.2 kHz.
- **Evidencia:** evaluación de la Ec. (3) bajo ambos exponentes contra las
  anclas Bark/frecuencia impresas del artículo. El rango impreso de la
  sección 2.1.2, «0.5 Bark (50 Hz) to 23.5 Bark (13.2 kHz)», y el ancla de la
  sección 3.1, «15 Bark (2.7 kHz)», se reproducen todos bajo el
  $0.76 \cdot 10^{-3}$ de Zwicker-Terhardt (50.6 Hz, 13.07 kHz y 2.71 kHz) y
  ninguno bajo el exponente impreso. Verificado en la página 4 del PDF (p. 4
  impresa) de Osses, García & Kohlrausch, ICA:2016, con las anclas en la
  página 7 del PDF (p. 7 impresa) del mismo artículo.
- **Comportamiento de la biblioteca:** implementa $0.76 \cdot 10^{-3}$ con
  una nota junto a la fórmula; el test de barrido de frecuencia portadora
  cazaría una regresión al valor impreso
  ([`fluctuation_strength.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/psychoacoustics/quality/fluctuation_strength.py)).
- **Estado:** sin notificar (artículo de congreso, no una norma).

## Medwin & Clay, Fundamentals of Acoustical Oceanography (1998), Ec. (3.4.30) (coeficiente del ácido bórico)

- **Ubicación:** el término de ácido bórico de Francois-Garrison tal como lo
  transcribe el libro, **Ec. (3.4.30), p. 110 impresa**. Una revisión
  anterior de esta entrada citaba la Ec. 3.4.29, que es la suma de absorción
  total de los tres términos en la p. 109 impresa; el bloque del ácido bórico
  es la ecuación siguiente.
- **El impreso:**
  $A_1 = (8.68/c) \cdot 10^{0.78\,\text{pH} - 5}\ \text{dB km}^{-1}\ \text{kHz}^{-1}$.
- **El problema:** el artículo original (Francois & Garrison 1982, JASA 72,
  Parte II, Ec. (10) y Fig. 7) imprime 8.86; los dígitos están traspuestos.
  Solo 8.86 reproduce la propia Tabla IV del artículo: con 8.68 las celdas
  dominadas por el bórico de $0.6$ a 30 kHz quedan hasta un $1.7\,\%$ por
  debajo de los totales impresos (peor caso relativo 2 kHz, 10 °C, $S = 35$:
  $0.1209$ contra los 0.123 dB/km impresos).
- **Evidencia:** recálculo de todas las celdas muestreadas de la Tabla IV
  bajo ambos coeficientes contra los valores impresos del artículo.
  Verificado en la página 131 del PDF (p. 110 impresa) de Medwin & Clay,
  Fundamentals of Acoustical Oceanography (1998), y en las páginas 8 y 9 del
  PDF (pp. 1886 y 1887 impresas) de Francois & Garrison (1982), JASA 72,
  Parte II, que imprimen el propio
  $A_1 = (8.86/c) \cdot 10^{0.78\,\text{pH} - 5}$ del artículo.
- **Comportamiento de la biblioteca:** implementa el 8.86 del artículo con
  una nota defensiva; el conjunto fijado de la Tabla IV incluye las filas
  dominadas por el bórico.
- **Estado:** sin notificar (libro, no una norma).

## Medwin & Clay (1998), Ec. (3.4.30) (la velocidad del sonido impresa como q)

- **Ubicación:** el mismo bloque de la Ec. (3.4.30), p. 110 impresa, su
  última línea.
- **El impreso:** $q = 1412 + 3.21T + 1.19 S + 0.0167 z\ \text{m/s}$.
- **El problema:** la cantidad que el bloque necesita es la velocidad del
  sonido $c$, que es por lo que dividen las dos líneas de arriba
  ($A_1 = 8.68/c$, y $A_2 = 21.44 S/c$ en el bloque del sulfato de magnesio
  de la misma página). Ningún símbolo $q$ se define en ninguna parte de la
  sección, así que el sistema transcrito no está cerrado: un lector que siga
  los símbolos impresos no tiene valor para $c$. Francois & Garrison 1982
  Parte II imprime el mismo polinomio como
  $c = 1412 + 3.21 T + 1.19 S + 0.0167 D$, introducido por «where c is the
  sound speed (m/s), given approximately by».
- **Evidencia:** el propio uso de $c$ del bloque dos líneas más arriba, y el
  artículo fuente. Verificado en la página 131 del PDF (p. 110 impresa) y la
  página 130 del PDF (p. 109 impresa) de Medwin & Clay (1998), y en la página
  8 del PDF (p. 1886 impresa) de Francois & Garrison 1982 Parte II (JASA 72).
- **Comportamiento de la biblioteca:** no le afecta; el modelo de absorción
  toma la velocidad del sonido del mismo polinomio bajo el nombre `c`.
- **Estado:** sin notificar (libro, no una norma).

---

## Maa (1998), «Potential of microperforated panel absorber», JASA 104(5), Ec. (5b)

- **Ubicación:** Ec. (5b), el coeficiente de reactancia de masa del panel
  microperforado, impreso como
  $k_m = 1 + [1 + k^2/2]^{-1/2} + 0.85\,d/t$.
- **El impreso:** el primer término entre corchetes lee
  $(1 + k^2/2)^{-1/2}$.
- **El problema:** la Ec. (4) del mismo artículo, de la que se factoriza
  (5b), imprime el término como $(3^2 + k^2/2)^{-1/2}$, y solo esa forma
  reproduce el límite de Crandall a $k$ bajo,
  $Z_1 \to (4/3) j\omega\rho_0 t$, de la propia Ec. (3a) del artículo: a
  $k \to 0$ la (5b) impresa da un factor de masa interna de 2 en lugar de
  4/3. La propia Fig. 1 del artículo lo confirma: con
  $0.85 \cdot d/t = 0.85$ el $k_m$ dibujado arranca cerca de $2.2$
  ($= 4/3 + 0.85$) a $k = 0.1$, no en $2.85$.
- **Evidencia:** recálculo de ambas variantes del corchete contra la Ec. (4),
  la Ec. (3a) y la curva de la Fig. 1; la solución exacta de Bessel de la
  Ec. (2) concuerda con la Ec. (4) dentro del $\sim 6\,\%$ que Maa declara
  solo con la forma $3^2$ (la forma 1 yerra en $>30\,\%$ a $k$ bajo).
  Verificado en la página 2 del PDF (p. 2862 impresa) de Maa (1998),
  «Potential of microperforated panel absorber», JASA 104(5), que lleva la
  Ec. (4) y la Ec. (5b) a quince líneas una de otra en la misma columna.
- **Comportamiento de la biblioteca:** implementa la Ec. (2) exacta (sin
  aproximación), así que la errata no entra en el código; el test de
  regresión ``test_maa_exact_vs_wide_range_approximation`` fija la solución
  exacta a la forma corregida de la Ec. (4).
- **Estado:** sin notificar (artículo de revista; la forma correcta aparece
  en los artículos anteriores de Maa de 1975/1987 y en literatura
  secundaria).

## Jiménez, Groby, Pagneux & Romero-García (2017), Appl. Sci. 7(6), 618, Ecs. (7)-(8)

- **Ubicación:** Ecs. (7) y (8), la densidad efectiva y el módulo de
  compresibilidad viscotérmicos de conducto rectangular (la serie de Stinson,
  usada para los cuellos y cavidades cuadrados del absorbente de ranura +
  resonador de Helmholtz).
- **El impreso:** la constante normalizadora inicial de ambas series es 4:
  $\rho_\text{eff} = -\rho_0 \cdot a^2 b^2/(4 \cdot G_\rho^2 \cdot \Sigma)$
  y el factor homólogo $4 \cdot (\gamma - 1) \cdot G_\kappa^2/(a^2 b^2)$
  dentro de $\kappa_\text{eff}$.
- **El problema:** la constante correcta es 64 (un error de factor 16). Solo
  64 reproduce los límites exactos del modelo: al desvanecerse las capas
  límite $\rho_\text{eff} \to \rho_0$ y $\kappa_\text{eff} \to \kappa_0$ (el
  4 impreso da $16 \cdot \rho_0$), y en continua el
  $j\omega \cdot \rho_\text{eff}$ del conducto cuadrado tiende a la
  resistividad de flujo de Poiseuille exacta de Shah-London: el valor de la
  serie $a^6/(64 \cdot S_0) = 28.4542$ casa con $fRe/2 = 28.455$ (en unidades
  de $\eta/a^2$), donde $S_0$ es la suma doble de modos transversales a
  $G = 0$; el 4 impreso da dieciséis veces eso.
- **Evidencia:** evaluación de ambas constantes contra los límites sin capa
  límite y el valor exacto de conducto cuadrado de Shah-London; el límite de
  conducto ancho de la serie también casa con el propio modelo de ranura de
  los artículos (Ec. (6)) solo con 64.
- **Comportamiento de la biblioteca:** implementa 64 con una nota en el
  docstring; los límites están fijados en
  [`tests/materials/absorbers/test_slow_sound.py`](https://github.com/jmrplens/phonometry/blob/main/tests/materials/absorbers/test_slow_sound.py)
  y la comprobación de conformidad «Poiseuille limit (Stinson 1991)».
- **Estado:** sin notificar (artículo de revista, no una norma).

## Jiménez et al. (2017), Appl. Sci. 7(6), 618 / Sci. Rep. 7, 5389, término de radiación de la ranura

- **Ubicación:** Appl. Sci. Ec. (3), la impedancia de radiación
  característica de las ranuras, y la reimpresión idéntica en los Methods del
  artículo de metadifusores (Sci. Rep. 7, 5389, Ec. (5)).
- **El impreso:**
  $Z_{\Delta l_\text{slit}} = -i\omega \cdot \Delta l_\text{slit} \cdot \rho_0/(\phi t \cdot S_0)$.
- **El problema:** el término modela la masa de radiación añadida de la boca
  de la ranura, pero el prefactor $-i\omega$ impreso es una expresión del
  convenio temporal opuesto ($e^{-i\omega t}$), inconsistente con la cadena
  de matrices de transferencia por lo demás en $e^{+i\omega t}$ de los
  artículos (las matrices de ranura con $+i$ fuera de la diagonal de la
  Appl. Sci. Ec. (2) y la impedancia de resonador tipo cotangente con $-i$).
  Transcrita literalmente en esa cadena, la corrección sube la resonancia del
  panel de ranuras donde una masa añadida debe bajarla: para una ranura de
  1 mm con paso de red de 30 mm y periodo de 50 mm el pico de absorción se
  mueve de 378.6 Hz a 386.8 Hz tal como está impreso, contra 370.8 Hz con el
  signo de masa. Las correcciones de extremo de cuello del mismo modelo se
  comportan correctamente (bajan la resonancia del resonador).
- **Evidencia:** evaluación numérica de ambos signos de la corrección contra
  el panel sin corregir; la dirección de las correcciones de extremo de
  cuello de los mismos artículos como control consistente.
- **Comportamiento de la biblioteca:** usa el signo de masa añadida
  ($+j\omega$ en el convenio $e^{+j\omega t}$ de la biblioteca), conjugando
  el término impreso exactamente igual que conjuga la serie de conducto de
  Stinson de los artículos; dirección y pico están fijados por
  ``test_slit_radiation_correction_lowers_resonance`` en
  [`tests/materials/absorbers/test_slow_sound.py`](https://github.com/jmrplens/phonometry/blob/main/tests/materials/absorbers/test_slow_sound.py).
- **Estado:** sin notificar (artículos de revista, no normas).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Tabla 5.1

- **Ubicación:** Tabla 5.1, «Coefficient and exponent values in the Delany
  and Bazley, Miki and modified Miki models», fila «Miki [6,7]», coeficiente
  $r$.
- **El impreso:** $r = 0.0109$.
- **El problema:** la fuente original (Miki 1990, J. Acoust. Soc. Jpn (E)
  11(1), Ec. (34)) imprime
  $\beta(f) = (\omega/c_0)[1 + 0.109 \cdot (f/\sigma)^{-0.618}]$; la tabla
  pierde un dígito. Con 0.0109 la parte real del número de onda de Miki a
  $f/\sigma = 0.01$ es $1.19$ en lugar de $2.88$, inconsistente con la fila
  de Delany-Bazley de la misma tabla ($3.10$ desde sus propios $r = 0.0862$,
  $s = -0.693$) y con la fila del «modified Miki» que el propio libro deriva
  de ella.
- **Evidencia:** comprobación dígito a dígito contra el artículo original de
  Miki (1990) (Ecs. (30)–(34)) y cálculo cruzado de ambas variantes en el
  borde del rango de ajuste. Verificado en la página 168 del PDF (p. 149
  impresa) de Attenborough & Van Renterghem, Predicting Outdoor Sound
  2e:2021, y en la página 4 del PDF (p. 22 impresa) de Miki, J. Acoust. Soc.
  Jpn (E) 11(1):1990.
- **Comportamiento de la biblioteca:** implementa el 0.109 original de Miki;
  el punto de digitalización $f/\sigma = 0.1$ está fijado en
  ``tests/reference_data/`` y en la comprobación de conformidad «Miki 1990
  Eqs. (30)-(34)».
- **Estado:** sin notificar (libro, no una norma).

## Attenborough & Van Renterghem, Predicting Outdoor Sound 2e (2021), Ec. (5.13)

- **Ubicación:** Ec. (5.13), la densidad compleja volumétrica de
  Johnson-Champoux-Allard, con
  $G(\Lambda) = \sqrt{1 - 4iT\eta\rho_0\omega/(R_S^2\Lambda^2\Omega^2)}$.
- **El impreso:** la tortuosidad $T$ aparece a la primera potencia dentro de
  $G(\Lambda)$.
- **El problema:** Johnson et al. (1987) y la formulación JCA estándar (Cox &
  D'Antonio 3e Ec. (6.19); Allard & Atalla) llevan ahí
  $T^2 = \alpha_\infty^2$. El impreso a la primera potencia rompe la asíntota
  de alta frecuencia que define la longitud característica viscosa: con $T^2$
  la densidad tiende a $(T\rho_0/\Omega)(1 + (1 - j)\delta_v/\Lambda)$ con
  $\delta_v = \sqrt{2\eta/\rho_0\omega}$, mientras que la forma impresa
  tiende a una corrección $\delta_v/(\Lambda\sqrt{T})$, que para $T = 2$
  supone un error del $29\,\%$ en el término de capa límite para la misma
  $\Lambda$.
- **Evidencia:** desarrollo asintótico de ambas variantes contra la
  definición de $\Lambda$ de Johnson et al. y contra la Ec. (6.19) de Cox &
  D'Antonio; el test JCA de alta frecuencia de la biblioteca fija el
  comportamiento con $T^2$. Verificado en la página 173 del PDF (p. 154
  impresa) de Predicting Outdoor Sound 2e:2021.
- **Comportamiento de la biblioteca:** implementa la forma estándar con $T^2$
  (Cox & D'Antonio Ec. (6.19)); la asíntota está fijada en
  ``test_high_frequency_density_asymptote``.
- **Estado:** sin notificar (libro, no una norma).

## Bies, Hansen & Howard, Engineering Noise Control 5e (2017), Ec. (8.141)

- **Ubicación:** sección 8.9.1, Ec. (8.141) (p. 461 impresa), la pérdida de
  transmisión de un silenciador desde los elementos de su matriz de cuatro
  polos total.
- **El impreso:**
  $$
  TL = 10 \lg\left[ \left(\frac{1+M_n}{1+M_1}\right)^2 \cdot \tfrac{1}{4} \cdot
  \left| \frac{Z_{A1}}{Z_{An}} T_{11} + \frac{T_{12}}{Z_{An}}
  + Z_{A1} T_{21} + \frac{Z_{An}}{Z_{A1}} T_{22} \right|^2 \right],
  $$
  es decir, con la razón de impedancias $Z_{A1}/Z_{An}$ ponderando $T_{11}$ y
  su inversa ponderando $T_{22}$.
- **El problema:** la fuente que la propia ecuación cita (Munjal, *Acoustics
  of Ducts and Mufflers* 2e, Ec. (3.27), p. 105) lleva el prefactor global
  $Z_{An}/Z_{A1}$ (equivalentemente $\sqrt{S_1/S_n}$ dentro de una forma en
  $20\log_{10}$) con $T_{11}$ sin ponderar y $Z_{A1}/Z_{An}$ sobre $T_{22}$.
  Tal como está impresa, la Ec. (8.141) falla el límite de expansión brusca:
  un elemento de longitud cero ($T = I$) entre $S_1 = 0.01\ \text{m}^2$ y
  $S_n = 0.02\ \text{m}^2$ es una expansión brusca de área con el clásico
  $TL = 10\log_{10}[(1+m)^2/(4m)] = 0.512\ \text{dB}$ ($m = S_n/S_1 = 2$),
  pero la ecuación impresa da
  $\tfrac{1}{4} \cdot (Z_{A1}/Z_{An} + Z_{An}/Z_{A1})^2 = 1.938\ \text{dB}$.
  Leer las razones como un prefactor global $Z_{A1}/Z_{An}$ también está mal:
  da 6.532 dB sobre el mismo oráculo y viola la reciprocidad ($11.34$ contra
  -0.70 dB para una cámara de expansión entre tubos desiguales; una TL
  negativa para un elemento pasivo). La errata es invisible siempre que las
  áreas de entrada y salida son iguales, donde todas las variantes se reducen
  a la Ec. (8.148).
- **Evidencia:** evaluación numérica del elemento identidad de longitud cero
  y de una cámara de expansión de puertos desiguales bajo la forma impresa,
  el prefactor invertido y la Ec. (3.27) de Munjal; solo la forma de Munjal
  reproduce el clásico de la expansión brusca (0.512 dB, en ambos sentidos) y
  es recíproca.
- **Comportamiento de la biblioteca:** `transmission_loss` en
  [`silencers.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/silencers.py) implementa
  la Ec. (3.27) de Munjal, con el límite de expansión brusca y la
  reciprocidad de la TL fijados por tests de regresión
  ([`tests/noise_control/test_silencers.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_silencers.py))
  y una nota defensiva junto a la fórmula.
- **Estado:** sin notificar (libro, no una norma).

## Long, Architectural Acoustics 2e (2014), Ec. (18.24) (signo de la directividad del micrófono)

- **Ubicación:** capítulo 18, «Multiple Open Microphones», Ec. (18.24)
  (p. 699 impresa), el criterio de estabilidad de ganancia antes de
  realimentación generalizado a varios micrófonos abiertos.
- **El impreso:**
  $Z_S + L_{H-M} + \Delta L_\text{nom} \le L_{H-L} \boldsymbol{+} D_M(\theta) - 10$,
  con el índice de directividad del micrófono entrando en el lado derecho con
  signo más.
- **El problema:** la Ec. (18.24) es la generalización a número de micrófonos
  abiertos de la Ec. (18.20) (p. 698 impresa), que lee
  $Z_S + L_{H-M} \le L_{H-L} \boldsymbol{-} D_M(\theta) - 10$ y que se sigue
  a su vez de la condición de oscilación Ec. (18.19),
  $Z_S + L_{H-M} = L_{H-L} - D_M(\theta)$, obtenida sustituyendo la ganancia
  del lazo de realimentación $G_S = L_{H-M} - L_{H-L} + D_M(\theta)$
  (Ec. (18.18)) en $Z_S + G_S = 0$ (Ec. (18.16)). Poner $N_m = 1$ hace
  $\Delta L_\text{nom} = 0$, así que la Ec. (18.24) debe reducirse a la
  Ec. (18.20) y no lo hace. El signo importa físicamente: $D_M(\theta)$ es
  «usually negative» en la propia definición de Long (en torno a $-2$ a
  -3 dB para un cardioide apuntado al orador), así que tal como está impreso
  un micrófono direccional *costaría* ganancia antes de realimentación en
  lugar de comprarla, invirtiendo la propia conclusión del capítulo de que
  «it is prudent to incorporate a cardioid or hypercardioid microphone into a
  system».
- **Evidencia:** la ecuación impresa lee
  $Z_S + L_{H-M} + \Delta L_\text{nom} \le L_{H-L} + D_M(\theta) - 10$,
  contra $Z_S + L_{H-M} \le L_{H-L} - D_M(\theta) - 10$ dos páginas antes,
  donde la misma posición lleva un menos. (Una revisión anterior de esta
  entrada citaba la extracción de `pdftotext`,
  `Z S þ L HM þ DL nom  L HL þ D M ðqÞ  10`, en la que `þ` es la ligadura que
  este PDF usa para «+» y todos los signos menos se han perdido por completo;
  esa extracción no puede distinguir un más de un menos y nunca debió ser la
  evidencia.) Verificado en la página 697 del PDF (p. 699 impresa) y la
  página 696 del PDF (p. 698 impresa) de Long, Architectural Acoustics 2e
  (2014). El signo menos es el que reproduce los propios casos particulares
  resueltos de Long a $N_m = 1$: con $Z_S = -6\ \text{dB}$, la Ec. (18.21) da
  $L_{H-M} \le L_{H-L} - D_M(\theta) - 4$ (un micrófono omnidireccional 4 dB
  por debajo del nivel medio de la audiencia), y la Ec. (18.22) da
  $L_{H-M} \le L_{H-L} - 2$ para un cardioide a $D_M = -2\ \text{dB}$.
  Ninguno de los dos casos particulares es recuperable desde la Ec. (18.24)
  impresa.
- **Comportamiento de la biblioteca:** `feedback_stability` en
  [`sound_reinforcement.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/electroacoustics/sound_reinforcement.py)
  implementa el signo de la Ec. (18.20), con una nota junto al criterio.
  Ambos casos particulares de Long están fijados por tests de regresión
  ([`tests/electroacoustics/test_sound_reinforcement.py`](https://github.com/jmrplens/phonometry/blob/main/tests/electroacoustics/test_sound_reinforcement.py))
  y por las comprobaciones de conformidad «Long, Architectural Acoustics 2e,
  Eq. (18.21)» y «Eq. (18.22)».
- **Estado:** sin notificar (libro, no una norma, así que no normativo).

## Long, Architectural Acoustics 2e (2014), Ec. (17.53) (constante de la cota de comunicación)

- **Ubicación:** capítulo 17, «Restaurant Design», Ec. (17.53) (p. 666
  impresa), la absorción mínima por mesa ocupada para una comunicación entre
  mesas adecuada.
- **El impreso:** $A_\text{tab} > 6.33 r_s^2$.
- **El problema:** la cota es la Ec. (17.52),
  $L_\text{SN} = 10\log_{10}[Q/(4\pi r^2)] + 10\log_{10}[A_\text{tab}/4]$,
  resuelta para $A_\text{tab}$ en el umbral declarado
  $L_\text{SN} > -6\ \text{dB}$, lo que da
  $A_\text{tab} > 16\pi \cdot 10^{-0.6} r_s^2/Q$. Con el $Q = 2$ que el
  capítulo usa para un orador, esa constante es 6.3130, no 6.33. La brecha es
  del $0.27\,\%$, es decir, el último dígito impreso: 6.33 es lo que devuelve
  $16\pi \cdot 10^{-0.6}/2$ si $10^{-0.6}$ se arrastra grueso como 0.252 en
  lugar de 0.251 19. Se clasifica como discrepancia de redondeo y no como
  error estructural de la fórmula, ya que la fórmula misma queda confirmada
  por su compañera (abajo) y ninguna suposición alternativa consistente
  reproduce 6.33 (exigiría $Q = 1.995$).
- **Evidencia:** la Ec. (17.54) inmediatamente siguiente es la misma forma
  cerrada en el umbral de privacidad $L_\text{SN} < -9\ \text{dB}$, y su
  constante impresa 3.16 es exactamente lo que da
  $16\pi \cdot 10^{-0.9}/2 = 3.1640$, confirmando la fórmula y el $Q = 2$.
  Solo la constante de -6 dB está desviada. Lo que *no* discrimina es la
  prosa de Long un párrafo después, «at least 6.3 or more square meters (68
  sq ft) of absorption per table»: 6.313 m² son $67.95\ \text{ft}^2$ y
  6.33 m² son $68.14\ \text{ft}^2$, así que ambos se imprimen como 68 sq ft,
  y ambos redondean a 6.3 m². Una revisión anterior de esta entrada ofrecía
  esa conversión como corroboración. Verificado en la página 665 del PDF
  (p. 666 impresa) de Long, Architectural Acoustics 2e (2014).
- **Comportamiento de la biblioteca:** `absorption_per_table` en
  [`crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/crowd_noise.py) calcula la cota
  desde la Ec. (17.52) en lugar de cablear ninguna de las dos constantes, así
  que ambas cotas se mantienen mutuamente consistentes; el valor 6.313 y el
  3.16 impreso están fijados por tests de regresión
  ([`tests/room/test_crowd_noise.py`](https://github.com/jmrplens/phonometry/blob/main/tests/room/test_crowd_noise.py)) y
  la constante 3.16 por la comprobación de conformidad «Long, Architectural
  Acoustics 2e, Eq. (17.54)».
- **Estado:** sin notificar (libro, no una norma, así que no normativo);
  clasificado como discrepancia de redondeo y no como defecto estructural.

## Long, Architectural Acoustics 2e (2014), Tabla 14.7 (filas de codos redondos)

- **Ubicación:** capítulo 14, Tabla 14.7, «Insertion Loss of Round Elbows»
  (p. 541 impresa), indexada por el producto frecuencia-anchura $f w$ (kHz
  por pulgadas).
- **El impreso:** solo cuatro filas: $f w < 1.9$ → 0 dB; $1.9 < f w < 3.8$ →
  1 dB; $3.8 < f w < 7.5$ → 2 dB; $f w > 15$ → 3 dB.
- **El problema:** la banda $7.5 < f w < 15$ no tiene fila alguna, así que la
  tabla salta de $3.8 < f w < 7.5$ directamente a $f w > 15$. Un cálculo por
  conductos cae en esa banda con normalidad: un codo de $24\ \text{in}$ a
  500 Hz tiene $f w = 12$.
- **Evidencia:** los mismos datos adaptados de la misma fuente ASHRAE
  aparecen en Bies, Hansen & Howard, *Engineering Noise Control* 5e, Tabla
  8.11, indexados por $W/\lambda$ ($= 0.074\,f w$). Su columna de codos
  redondos tiene seis filas, 0/1/2/3/3/3, y da 3 dB para
  $0.55 \le W/\lambda < 1.11$, que es exactamente la banda $7.5 < f w < 15$
  que Long omite. Las cuatro filas de Long se mapean sobre las seis de Bies
  así: las tres primeras coinciden entrada por entrada, la cuarta
  ($f w > 15$, 3 dB) fusiona legítimamente las dos filas superiores idénticas
  de Bies, y la banda sin fila es la cuarta de Bies. Una revisión anterior de
  esta entrada decía que «las Tablas 14.5 y 14.6 llevan ambas seis filas» y
  que «las otras cinco filas de las dos tablas coinciden entrada por
  entrada»; en la página, la Tabla 14.5 lleva seis filas y la Tabla 14.6
  cinco (fusiona las mismas dos bandas superiores idénticas, legítimamente),
  y la Tabla 14.7 imprime cuatro, así que ninguno de los dos recuentos es
  correcto. Verificado en la página 542 del PDF (p. 541 impresa) y la página
  541 del PDF (p. 540 impresa) de Long, Architectural Acoustics 2e (2014).
- **Comportamiento de la biblioteca:** `elbow_insertion_loss` en
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) lleva la columna
  redonda de seis filas con 3 dB en la banda que falta, fijada por
  `test_elbow_tables_by_frequency_width_product`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py)).
- **Estado:** sin notificar (libro, no una norma).

## Long, Architectural Acoustics 2e (2014), Ec. 13.28 (unidades de U_G)

- **Ubicación:** capítulo 13, Ec. 13.28 (p. 521 impresa), el coeficiente de
  caída de presión normalizado
  $\xi = 334.9 \cdot \Delta P/(\rho_0 U_G^2)$ del modelo de potencia sonora
  de difusores.
- **El impreso:** la nomenclatura bajo la ecuación da «U_G = flow velocity
  prior to the diffuser (ft/min)» y, en la línea siguiente, «= Q/(60·S_G)
  (for Q in cfm)».
- **El problema:** las dos declaraciones se contradicen. $Q$ en ft³/min
  dividido por $60 S_G$ es una velocidad en **ft/s**, no ft/min, y solo la
  lectura en ft/s hace correcta la constante: $334.9/\rho_0$ con
  $\rho_0 = 0.075\ \text{lb/ft}^3$ es $4465 \cdot \Delta P/U^2$, que es la
  relación estándar de presión de velocidad $\Delta P/(U/4005)^2$ solo cuando
  $U$ se convierte desde ft/s. Leído como ft/min el coeficiente sale 3600
  veces demasiado pequeño. La propia Ec. 13.27 declara $U_G$ en ft/s, así que
  la etiqueta «(ft/min)» bajo la Ec. 13.28 es la discordante.
- **Evidencia:** comprobación dimensional de $Q/(60 S_G)$; reconstrucción de
  la constante $334.9/\rho_0$ desde la relación de presión de velocidad; y la
  frecuencia de pico. Lo que **no** discrimina es el nivel global: la
  Ec. 13.27 lleva $30\log_{10}\xi + 60\log_{10} U_G$, y sustituir la
  Ec. 13.28 hace que la velocidad se cancele idénticamente,
  $30\log_{10}\xi + 60\log_{10} U_G = 30\log_{10}(334.9 \Delta P/\rho_0)$.
  Para el difusor de impulsión de la Tabla 14.9 ($S_G = 4\ \text{ft}^2$,
  $Q = 312$ cfm, $\Delta P = 0.05$ in w.g.) ambas lecturas devuelven por
  tanto el mismo $L_W = 45.18\ \text{dB}$. Una revisión anterior de esta
  entrada afirmaba que la lectura en ft/min «lo falla por 100 dB», cosa
  aritméticamente imposible para una cantidad que no depende de la velocidad
  en absoluto. Lo que sí discrimina es la Ec. 13.32, $f_P = 48.8 U_G$, que es
  el único otro lugar donde entra $U_G$: leída en ft/s la velocidad de
  aproximación es $1.3\ \text{ft/s}$ y el pico cae a 63.4 Hz, es decir, en la
  octava de 63 Hz, así que la forma de la Ec. 13.31 pone 33.4 dB en esa banda
  contra los 33 impresos; leída en ft/min es $78\ \text{ft/min}$, el pico se
  mueve a 3 806 Hz, y la misma forma pone -8.2 dB en la banda de 63 Hz.
  Verificado en la página 522 del PDF (p. 521 impresa) de Long, Architectural
  Acoustics 2e (2014).
- **Comportamiento de la biblioteca:** `diffuser_sound_power` en
  [`hvac.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) lee $U_G$ en ft/s
  internamente (SI en la interfaz), con la fila de la Tabla 14.9 fijada por
  `test_diffuser_sound_power_reproduces_the_table_14_9_row`
  ([`tests/noise_control/test_hvac_long.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_hvac_long.py))
  y la comprobación de conformidad «Long 2e Eqs. 13.27-13.33».
- **Estado:** sin notificar (libro, no una norma).

## Vigran, Building Acoustics (2008), pie de la Figura 8.37 (exponente de la rigidez de la moqueta)

- **Fuente no normativa** (libro).
- **Ubicación:** sección 8.4.2, el pie de la Figura 8.37 en la p. 320 impresa
  / p. 341 del PDF, que etiqueta las curvas de mejora predicha de dos
  revestimientos de suelo colocados sobre un suelo pesado.
- **El impreso:** «Predicted improvement with a linear model: stiffness of
  carpet squares 3.2·10^6 N/m, vinyl covering 5.2·10^6 N/m.» (Vigran escribe
  el separador decimal como punto.)
- **El problema:** el exponente de la moqueta está un orden de magnitud alto.
  El cuerpo del texto que introduce la figura, en la p. 321 impresa, dice de
  las losetas de moqueta que «we have assumed that the covering has the same
  stiffness as used in Figure 8.36», y la Figura 8.36 está etiquetada
  $s = 3.2 \cdot 10^{5}\ \text{N/m}$ dentro del gráfico, el mismo valor que
  el cuerpo del texto de la p. 320 impresa le da. El valor del vinilo del
  mismo pie es correcto.
- **Evidencia:** la p. 320 impresa declara $3.2 \cdot 10^{5}\ \text{N/m}$
  «giving a resonance frequency f0 of approximately 130 Hz with a hammer mass
  of 0.5 kg», y $\sqrt{3.2 \cdot 10^{5}/0.5}/(2\pi) = 127.3\ \text{Hz}$ lo
  reproduce mientras que
  $\sqrt{3.2 \cdot 10^{6}/0.5}/(2\pi) = 402.6\ \text{Hz}$ es una frecuencia
  que no aparece en ningún lugar de la sección. La misma aritmética aplicada
  al valor del vinilo del pie da
  $\sqrt{5.2 \cdot 10^{6}/0.5}/(2\pi) = 513.3\ \text{Hz}$ contra los
  «approximately 510 Hz» impresos en la p. 321, lo que fija la fórmula y la
  masa de martillo que usó el autor. Gráficamente, las dos curvas de
  predicción a trazos de la Fig. 8.37 están a unas dos octavas una de otra,
  casando con la razón de rigideces
  $5.2 \cdot 10^{6}/(3.2 \cdot 10^{5}) = 16.25$ (un factor 4.03 en
  frecuencia) y no con
  $5.2 \cdot 10^{6}/(3.2 \cdot 10^{6}) = 1.63$ (un factor 1.27). Verificado
  en la página 341 del PDF (p. 320 impresa) de Vigran, Building
  Acoustics:2008, en la que ambos exponentes del pie leen 6 sin ambigüedad y
  el cuerpo del texto de la misma página lee
  $3.2 \cdot 10^{5}\ \text{N/m}$, con el argumento circundante leído en la
  página 340 del PDF (p. 319 impresa) y la página 342 del PDF (p. 321
  impresa) de la misma edición.
- **Comportamiento de la biblioteca:** no hizo falta ninguno; la biblioteca
  toma la rigidez del revestimiento del usuario a través de
  `covering_contact_stiffness`, y las frecuencias de corte impresas en las
  que está anclada vienen de Hopkins y no de este pie.
- **Estado:** sin notificar.

## Norton & Karczub, Fundamentals of Noise and Vibration Analysis for Engineers 2e (2003), Ec. (6.56)

- **Ubicación:** sección 6.6.1, Ec. (6.56), el factor de pérdida por
  acoplamiento de dos placas homogéneas unidas por $N$ conexiones puntuales
  (p. 418 impresa).
- **El impreso:** el corchete del denominador
  $(\rho_{s1}^2 h_1^2 c_{L1}^2 + \rho_{s2}^2 h_2^2 c_{L2}^2)$ aparece a la
  primera potencia.
- **El problema:** tal como está impresa, la expresión no es adimensional. El
  prefactor $4 N h_1 c_{L1}/(\sqrt{3}\,\omega S_1)$ ya tiene las dimensiones
  de $\text{m}^2\,\text{s}^{-1}$ sobre $\text{m}^2\,\text{s}^{-1}$, es decir,
  la unidad, así que la razón restante de los dos productos entre corchetes
  debe ser adimensional también. Eso exige que la suma esté al cuadrado,
  $A_1 A_2/(A_1 + A_2)^2$.
- **Evidencia:** la propia respuesta del libro al problema 6.13 (p. 617
  impresa). Con el denominador al cuadrado el par de aluminio de doce pernos
  da $\eta_{12} = 1.43 \cdot 10^{-2}$ a 125 Hz contra el
  $1.44 \cdot 10^{-2}$ impreso, y casa con toda la columna de 125 Hz a 2 kHz
  a mejor del $0.7\,\%$; con el denominador impreso (sin cuadrado) el
  resultado no es un factor de pérdida en absoluto. Verificado en la página
  438 del PDF (p. 418 impresa) de Norton & Karczub, Fundamentals of Noise and
  Vibration Analysis for Engineers 2e:2003.
- **Comportamiento de la biblioteca:**
  `point_connection_coupling_loss_factor` en
  [`junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/structural/junction_transmission.py)
  implementa la forma al cuadrado, con la columna impresa fijada por un test
  de regresión
  ([`tests/vibration/structural/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_junction_transmission.py))
  y una nota junto a la fórmula.
- **Estado:** sin notificar (libro, no una norma).

## Norton & Karczub 2e (2003), respuesta del problema 6.13 (columna eta_21)

- **Ubicación:** respuestas a los problemas, problema 6.13 (p. 617 impresa),
  las dos columnas de $\eta_{21}$ de las tablas soldada y atornillada.
- **El impreso:** para las dos placas de aluminio (placa 1: 3 mm,
  2.5 m × 1.2 m; placa 2: 5.5 mm, 2.0 m × 1.2 m) la respuesta da, a 125 Hz,
  $\eta_{21} = 5.77 \cdot 10^{-3}$ (soldada) y $2.64 \cdot 10^{-2}$
  (atornillada).
- **El problema:** ambas columnas son exactamente la columna $\eta_{12}$
  correspondiente multiplicada por $h_2/h_1 = 1.833$. La relación de
  consistencia SEA es $n_1 \eta_{12} = n_2 \eta_{21}$ (Ec. 6.8) con la
  densidad modal de placa plana $n = S\sqrt{12}/(2 c_L h)$ de la Ec. (6.25),
  así que el factor correcto es
  $n_1/n_2 = (S_1 h_2)/(S_2 h_1) = 2.292$. La columna impresa pierde la razón
  de áreas de placa $S_1/S_2 = 1.25$.
- **Evidencia:** la razón de las columnas impresas es 1.8333 a cinco dígitos
  en todas las bandas de ambas tablas, que es $h_2/h_1$ exactamente; las
  propias columnas de $\eta_{12}$ se reproducen desde las Ecs. (6.52) a
  (6.56) a mejor del 0.7 %. Verificado en la página 637 del PDF (p. 617
  impresa) de Norton & Karczub 2e:2003, la página que lleva ambas tablas de
  respuestas.
- **Comportamiento de la biblioteca:** las columnas de $\eta_{12}$ se usan
  como oráculo de regresión; $\eta_{21}$ se obtiene de la Ec. (6.8) con las
  densidades modales completas, y un test fija la razón 2.292 explícitamente
  ([`tests/vibration/structural/test_junction_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_junction_transmission.py)).
- **Estado:** sin notificar (libro, no una norma).

## Norton & Karczub 2e (2003), problema 6.10 (área de la plataforma)

- **Ubicación:** problemas, problema 6.10 (pp. 593-594 impresas) y su
  respuesta (p. 617 impresa): una plataforma de satélite acoplada a un
  cilindro de aluminio, octava de 500 Hz, respuestas impresas
  $\eta_{12} = 4.26 \cdot 10^{-4}$, $\eta_{21} = 3.92 \cdot 10^{-4}$ y
  $\Pi_\text{in} = 1.31\ \text{W}$.
- **El impreso:** el enunciado da la plataforma de aluminio como «5 mm thick
  and 3.5 m × 3 m», es decir, 10.5 m².
- **El problema:** esa área es inconsistente con las tres respuestas
  impresas. La Ec. (6.12) fija
  $E_1/E_2 = (\eta_2 + \eta_{21})/\eta_{12} = 6.554$ desde los factores de
  pérdida impresos solos, mientras que la geometría enunciada con las
  velocidades impresas (27.2 y 13.2 mm/s) da 7.88. La razón de energías es
  independiente de las densidades modales y de la velocidad de onda, así que
  ninguna elección de esas puede reconciliarla; solo el área de la plataforma
  puede. El área que las respuestas implican es 8.73 m², que es
  $3.5 \times 3$ menos la huella $\pi(0.75\ \text{m})^2$ del cilindro que la
  Fig. P6.10 muestra atravesando la plataforma.
- **Evidencia:** con 8.73 m² la inversión de las Ecs. (6.15), (6.8) y (6.10)
  devuelve $\eta_{12} = 4.256 \cdot 10^{-4}$,
  $\eta_{21} = 3.910 \cdot 10^{-4}$ y $\Pi_\text{in} = 1.306\ \text{W}$, es
  decir, las tres respuestas impresas dentro del 0.4 %; la energía y la
  densidad modal del propio cilindro salen sin cambios en cualquier caso.
  Verificado en la página 613 del PDF (p. 593 impresa), que lleva el
  enunciado y sus dimensiones, y la página 637 del PDF (p. 617 impresa), que
  lleva las tres respuestas, de Norton & Karczub 2e:2003.
- **Comportamiento de la biblioteca:** `power_injection_clf` en
  [`experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/vibration/structural/experimental_sea.py)
  implementa la inversión tal como está publicada; el test de regresión usa
  el área libre de la plataforma y documenta la discrepancia
  ([`tests/vibration/structural/test_experimental_sea.py`](https://github.com/jmrplens/phonometry/blob/main/tests/vibration/structural/test_experimental_sea.py)).
- **Estado:** sin notificar (libro, no una norma).

## Norton & Karczub 2e (2003), problema 3.14 (factor de pérdida estructural)

- **Ubicación:** problemas, problema 3.14 (p. 580 impresa) y su respuesta
  (p. 611 impresa): la pérdida de transmisión por bandas de octava de un
  panel de aglomerado de 20 mm.
- **El impreso:** el enunciado da al panel un factor de pérdida estructural
  de «~1.5 × 10⁻²»; la respuesta da 27 dB a 8 kHz y 38.6 dB a 16 kHz.
- **El problema:** esos dos valores están por encima de la frecuencia crítica
  del panel (4885 Hz para el aglomerado del Appendix 4,
  $f_c t = 97.7\ \text{m/s}$) y siguen por tanto la Ec. (3.110) de Cremer,
  que contiene $10\log_{10}(\eta)$. Con $\eta = 1.5 \cdot 10^{-2}$ la
  ecuación da 37.0 dB y 48.5 dB, diez decibelios por encima de las respuestas
  impresas; con $\eta = 1.5 \cdot 10^{-3}$ da 27.0 dB y 38.5 dB.
- **Evidencia:** el desplazamiento de 10 dB es exactamente una década de
  $10\log_{10}(\eta)$, y la dependencia en frecuencia del par impreso fija de
  forma independiente $f_c = 4939\ \text{Hz}$ contra el valor del Appendix 4
  de 4885 Hz. Los ocho valores por debajo de la coincidencia se reproducen
  exactamente desde la Ec. (3.104) y no involucran $\eta$. La discrepancia es
  una década en un exponente impreso, así que las dos cifras se leyeron como
  imágenes y no a través de la capa de texto. Verificado en la página 600 del
  PDF (p. 580 impresa) y la página 631 del PDF (p. 611 impresa) de Norton &
  Karczub 2e (2003).
- **Comportamiento de la biblioteca:** el test de regresión usa
  $\eta = 1.5 \cdot 10^{-3}$, el valor que las respuestas impresas exigen
  ([`tests/building/prediction/test_panel_transmission.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_panel_transmission.py)).
- **Estado:** sin notificar (libro, no una norma).

---

## Vigran, Building Acoustics (2008), Ec. (9.18) (coeficiente del lado receptor)

- **Ubicación:** sección 9.2.3.2, Ec. (9.18) (p. 339 impresa), el factor de
  transmisión del modelo unidimensional de plénum de techo suspendido según
  Mechel (1980).
- **El impreso:** el denominador lee $m_S L_S \cdot m_R L_R h$ con el $m_R$
  **sin prima**, mientras que el exponente de la misma expresión lleva el
  $m'_R = m_R + s_R \tau_R / h$ con prima de la Ec. (9.17).
- **El problema:** los dos lados del plénum se integran de la misma manera.
  La integral del lado receptor es
  $\int_0^{L_R} \exp(-\varepsilon m'_R x)\,dx = (1 - \exp(-\varepsilon m'_R L_R))/(\varepsilon m'_R)$,
  así que el factor que la normaliza debe ser $m'_R L_R$, exactamente igual
  que el del lado emisor es $m_S L_S$. Leída al pie de la letra, la expresión
  impresa no es un factor de transmisión en absoluto: arrastra un
  $m'_R/m_R = 1 + s_R \tau_R/(h m_R)$ espurio, así que crece sin cota cuando
  cae el amortiguamiento del plénum. Dos consecuencias son visibles con
  entradas corrientes ($L_S = L_R = 5\ \text{m}$, $h = 0.6\ \text{m}$,
  $R_S = R_R = 25\ \text{dB}$, $\varepsilon = 2$, $s_S = s_R = 0.5$): el
  modelo **diverge cuando el amortiguamiento del plénum se desvanece**, dando
  $R_\text{cl} = 40.26\ \text{dB}$ a $m_R = 0.01\ \text{1/m}$ pero solo
  26.48 dB a $10^{-4}$ y 6.64 dB a $10^{-6}$, contra los 40.85 dB finitos que
  la lectura deducida devuelve para el mismo plénum desnudo, donde el término
  de fuga $s_R \tau_R/h$ acota el camino; un plénum sin absorbente alguno se
  predice por tanto arbitrariamente peor que el valor limitado por la fuga en
  lugar de igual a él. También **rompe la conservación de la energía**,
  devolviendo $\tau_\text{cl} = 4.45$ a $R_S = R_R = 6\ \text{dB}$,
  $m_R = 0.01$ y $\tau_\text{cl} = 829$ a $R_S = R_R = 0\ \text{dB}$,
  $m_R = 10^{-3}$.
- **Evidencia:** con $m'_R$ en el denominador todas esas patologías
  desaparecen: $\tau_\text{cl}$ se aplana sobre el valor limitado por la fuga
  cuando el amortiguamiento se desvanece, donde la forma impresa sigue
  creciendo, y está acotado por arriba por 1 porque
  $(1 - \exp(-\varepsilon m'_R L_R))/(\varepsilon m'_R L_R) \le 1$, y se
  reduce al propio resultado de atenuación pequeña de Vigran, la Ec. (9.19)
  $\tau_\text{cl} = \varepsilon^2 \tau_S \tau_R L_R/(4h)$, siempre que
  $m_S L_S$ y $m'_R L_R$ son ambos pequeños. Con el $m_R$ impreso el mismo
  límite recoge el factor $m'_R/m_R$, que diverge, así que la Ec. (9.18) tal
  como está impresa no se reduce a la Ec. (9.19) en absoluto: las dos
  ecuaciones que el libro presenta como pareja son inconsistentes entre sí.
  Verificado en la página 361 del PDF (p. 339 impresa) de Vigran, Building
  Acoustics:2008, que muestra el denominador llevando el $m_R$ sin prima
  mientras el exponente de la misma expresión lleva el $m'_R$ con prima de la
  Ec. (9.17).
- **Comportamiento de la biblioteca:** `plenum_flanking_reduction_index` en
  [`ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/building/prediction/ceiling_plenum.py)
  implementa el $m'_R$ deducido en el exponente y en el denominador, con la
  lectura documentada junto a la fórmula, y rechaza un factor de transmisión
  por encima de la unidad en lugar de reportar un índice de reducción sonora
  negativo. Los tests fijan la física que el modelo debe (monotonía en el
  amortiguamiento, la cota $\tau_\text{cl} \le 1$, el tamaño del término de
  fuga de la Ec. (9.17) en un techo realista) y la única propiedad que separa
  las dos lecturas: un plénum desnudo no peor que el valor sin amortiguar de
  la Ec. (9.20)
  ([`tests/building/prediction/test_ceiling_plenum.py`](https://github.com/jmrplens/phonometry/blob/main/tests/building/prediction/test_ceiling_plenum.py)).
- **Estado:** sin notificar (libro, no una norma). El artículo original de
  Mechel de 1980, que Vigran reproduce, no estuvo disponible para comprobar
  si la errata se origina allí.

## Real Decreto 1367/2007, Anexo IV A.3.3 (tablas de umbrales de Kf y Ki)

- **Ubicación:** Anexo IV, sección A.3.3, las tablas de corrección $K_f$
  (baja frecuencia) y $K_i$ (impulsiva), fila central de cada una.
- **El impreso:** ambas tablas imprimen la fila de 3 dB como «Si 10 > Lf <=
  15» y «Si 10 > Li <= 15» respectivamente (BOE-A-2007-18397, texto
  consolidado).
- **El problema:** la condición tal como está impresa es insatisfacible. Lee
  «10 mayor que Lf» y «Lf como mucho 15» a la vez, lo que seleccionaría
  niveles por debajo de 10 dB, pero la fila de arriba ya asigna esos a 0 dB
  («Si Lf <= 10») y la fila de abajo cubre «Si Lf > 15». Las tres filas solo
  particionan el rango bajo la lectura $10 < L_f \le 15$, así que el «>» es
  una inversión tipográfica de «<».
- **Evidencia:** las filas que la encierran no dejan otra lectura
  consistente; la construcción idéntica aparece en ambas tablas, y las tablas
  equivalentes de los reglamentos de ruido autonómicos que transponen este
  Anexo imprimen `10 < Lf <= 15`. Verificado en la página 26 del PDF (p. 26
  impresa) de Real Decreto 1367/2007, texto consolidado BOE-A-2007-18397, en
  la que el «>» de ambas filas centrales es inequívoco contra los glifos
  «<=» de la misma celda.
- **Comportamiento de la biblioteca:**
  [`low_frequency_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/assessment/spain.py)
  e `impulsive_correction` implementan $10 < L \le 15$, con un test de
  regresión fijando las tres ramas en los límites de 10 dB y 15 dB.
- **Estado:** sin notificar (reglamento nacional, no un organismo de
  normalización).

---

## Directiva (UE) 2015/996 de la Comisión, Anexo II 2.2.1 (rango de bandas de octava de la fuente de carretera)

- **Ubicación:** el Anexo, punto 2.2.1, segundo párrafo bajo el encabezado
  «Traffic flow» (DO L 168, 1.7.2015, p. 8).
- **El impreso:** «these sound power levels are calculated for each octave
  band i from 125 Hz to 4 kHz».
- **El problema:** el modelo de fuente de carretera contradice su propia base
  de datos de coeficientes. Todas las tablas dependientes de banda del
  Appendix F, tanto en el texto de 2015 como en la versión sustituida por la
  (UE) 2021/1226, están impresas sobre las ocho bandas de octava de **63 Hz a
  8 kHz** (la Tabla F-3 no tiene columnas de frecuencia), y el punto 2.1.1
  del mismo Anexo define el rango de frecuencias del método como 63 Hz a
  8 kHz. Un cálculo restringido a 125 Hz - 4 kHz descartaría en silencio las
  bandas de 63 Hz y 8 kHz, que el Appendix F tabula como todas las demás.
- **Evidencia:** corregido por la corrección de errores publicada en el
  DO L 5, 10.1.2018, p. 35, que lee íntegra: 'On page 8, in the Annex, in
  point 2.2.1, in the second paragraph under the heading "Traffic flow": for:
  "each octave band i from 125 Hz to 4 kHz", read: "each octave band i from
  63 Hz to 8 kHz"'. La misma corrección de errores añade además «octave
  bands» al rango de frecuencias de 2.1.1. Verificado en la página 8 del PDF
  (p. L 168/8 impresa) de la Directiva (UE) 2015/996:2015 para la
  restricción impresa, en la página 1 del PDF (p. L 5/35 impresa) de la
  corrección de errores para ambos puntos, y en la página 4 del PDF
  (p. L 168/4 impresa) y la página 124 del PDF (p. L 168/124 impresa) de la
  Directiva para el rango conforme.
- **Comportamiento de la biblioteca:**
  [`cnossos_road`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/environment/sources/cnossos_road.py)
  trabaja sobre la rejilla corregida de 63 Hz a 8 kHz
  (`ROAD_OCTAVE_BANDS`), fijada por
  `test_octave_bands_are_the_corrected_range` y por los casos del libro de
  cálculo, cuyos niveles publicados cubren las ocho bandas.
- **Estado:** corregido por el organismo emisor (corrección de errores del 10
  de enero de 2018); registrado porque el texto de 2015 sin corregir sigue
  siendo el que más se descarga y se cita.

---

## Ainslie, Principles of Sonar Performance Modelling (2010), Ec. (9.57)

*Libro, no una norma.*

- **Ubicación:** sección 9.1.1.2.4 (p. 457 impresa), el alcance de transición
  entre los regímenes de decapado de modos y de modo único del modelo de
  flujo de Weston.
- **El impreso:** $r_\text{MS} \approx k^2 H_e^3/(9\eta)$, donde $H$ es la
  profundidad del agua, $H_e$ la profundidad efectiva de Weston de la
  Ec. (9.55), $k = \omega/c_w$ y $\eta$ el gradiente de pérdida por
  reflexión.
- **El problema:** la frase inmediatamente encima prescribe la deducción,
  «estimated by equating θ_n and θ_eff with n = 3/2». Los dos ángulos están a
  cuatro páginas impresas uno de otro, no en la misma página como declaraba
  una revisión anterior de esta entrada:
  - Ec. (9.47), $\theta_\text{eff} = (\pi H/(4\eta r))^{1/2}$, **p. 453
    impresa**, con la **profundidad de agua verdadera $H$** (procede de la
    integral multitrayecto Ec. (9.46), cuyo prefactor $1/(rH)$ es el área de
    cilindro $A_\text{CS} = 2\pi r H$ de la Ec. (9.44), así que $H$ es la
    profundidad que cuenta rebotes de fondo);
  - Ec. (9.56), $\theta_n \approx n\pi/(k H_e)$, **p. 457 impresa**, con la
    **profundidad efectiva $H_e$** (los ángulos de modo los fija la frontera
    aparente de presión nula).

  Igualarlos a $n = 3/2$ da $\pi H/(4\eta r) = 9\pi^2/(4k^2H_e^2)$, es decir
  **$r_\text{MS} = k^2H_e^2H/(9\pi\eta)$**. La forma impresa es mayor en
  $\pi H_e/H$. El factor $\pi$ es incondicional: sobrevive incluso si se
  sustituye $H_e$ por $H$ en la Ec. (9.47), que es presumiblemente como
  surgió el $H_e^3$ impreso, y esa lectura daría $k^2H_e^3/(9\pi\eta)$,
  todavía $\pi$ por debajo del impreso. El residuo $H_e/H$ es la propia
  sustitución de profundidad, y tiende a 1 en alta frecuencia. La otra
  transición de la misma sección, la Ec. (9.50)
  $r_\text{CS} = \pi H/(4\eta\psi_c^2)$, sigue su propia deducción
  exactamente (es donde se cruzan la Ec. (9.42) y la Ec. (9.49)), así que el
  defecto queda confinado a la Ec. (9.57).
- **Evidencia:** la re-deducción simbólica de arriba, comprobada
  numéricamente para $H = 50\ \text{m}$, $f = 250\ \text{Hz}$,
  $c_w = 1500\ \text{m/s}$ sobre el lecho de arena de la Tabla 9.1
  ($\eta = 0.28\ \text{Np/rad}$, $\psi_c = 33.56^\circ$,
  $H_e = 53.63\ \text{m}$, $k = 1.047\ \text{m}^{-1}$):

  | | $r_\text{MS}$ | $\theta_\text{eff}$ allí (Ec. 9.47) |
  |---|---|---|
  | deducción, $k^2H_e^2H/(9\pi\eta)$ | 19.9 km | 4.808° |
  | Ec. (9.57) impresa, $k^2H_e^3/(9\eta)$ | 67.1 km | 2.619° |

  La razón $67.1/19.9$ es $\pi H_e/H = 3.3695$ a todos los dígitos
  arrastrados. La columna de ángulos es una comprobación independiente que no
  depende de cómo se lea la deducción: los dos primeros ángulos de modo de la
  Ec. (9.56) son $\theta_1 = 3.205^\circ$ y $\theta_2 = 6.410^\circ$, así que
  $\theta_{3/2} = 4.808^\circ$. En el alcance deducido el ángulo efectivo es
  exactamente $\theta_{3/2}$, a mitad de camino entre los dos primeros modos,
  que es lo que el texto pide. En el alcance impreso ha caído a 2.619°, **por
  debajo del propio $\theta_1$**: el segundo modo se habría decapado mucho
  antes, así que ese alcance no puede ser donde empieza el régimen de modo
  único. Ambas fórmulas impresas están confirmadas en la página 483 del PDF
  (p. 453 impresa) y la página 487 del PDF (p. 457 impresa) de Ainslie,
  Principles of Sonar Performance Modelling (2010).
- **Comportamiento de la biblioteca:** `weston_regime_boundaries` en
  [`propagation/weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/propagation/weston_regimes.py)
  implementa el $k^2H_e^2H/(9\pi\eta)$ consistente con la deducción, que es
  además lo que mantiene $\theta_\text{eff}$ definido con $H$ en todos los
  puntos donde el módulo evalúa la Ec. (9.47). La regla de igualación está
  fijada por
  `test_mode_stripping_boundary_equates_theta_eff_with_mode_3_over_2`, que
  reconstruye ambos ángulos desde las ecuaciones impresas y no desde la
  implementación, y la definición compartida de $\theta_\text{eff}$ por
  `test_composite_loss_and_the_boundary_use_the_same_effective_angle` (ambos
  en
  [`tests/underwater/propagation/test_weston_regimes.py`](https://github.com/jmrplens/phonometry/blob/main/tests/underwater/propagation/test_weston_regimes.py)).
- **Estado:** sin notificar (libro, no una norma).

---

## NMFS (2024) Updated Technical Guidance v3.0, Tabla 5 / Tabla ES2 (C de los otáridos)

*Documento de guía regulatoria, no una norma.*

- **Ubicación:** Tabla 5 (p. 25 impresa), repetida como Tabla ES2 (p. 3
  impresa) y de nuevo como Tabla 8 (p. 35 impresa): el parámetro de
  ponderación auditiva $C$ del grupo de pinnípedos otáridos en el agua
  (OW / OCW).
- **El impreso:** $C = 1.37\ \text{dB}$.
- **El problema:** el valor correcto es 1.36 dB. El propio NMFS lo declara en
  la nota de la tabla: «During the public comment period, an error was
  identified with the Navy's rounding, where this value should be 1.36,
  instead of 1.37. Because this is such a minor error and to remain
  consistent with the Navy, NMFS decided rely upon the value the Navy
  originally provided.» El documento publica por tanto el dígito equivocado a
  sabiendas.
- **Evidencia:** recálculo independiente de $C$ desde su propia definición,
  el pico negado de $W(f)$, con los parámetros de la misma fila $a = 1.58$,
  $b = 5$, $f_1 = 2.53\ \text{kHz}$, $f_2 = 43.8\ \text{kHz}$:
  $C = 1.3643\ \text{dB}$, que redondea a 1.36. El umbral TTS ponderado
  publicado de la misma fila ($179\ \text{dB} = K + C$ con $K = 178$) no se
  ve afectado por el tercer dígito. El mismo recálculo reproduce todas las
  demás filas de la tabla a los dos decimales impresos, así que la fila OW es
  la única que no redondea desde sus propios parámetros. Verificado en la
  página 36 del PDF (p. 25 impresa), la página 14 del PDF (p. 3 impresa) y la
  página 46 del PDF (p. 35 impresa) de NMFS Updated Technical Guidance
  v3.0:2024, las tres llevando 1.37 con la nota idéntica.
- **Comportamiento de la biblioteca:**
  [`bioacoustics/weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/bioacoustics/weighting.py)
  implementa 1.36 y mantiene el 1.37 impreso disponible como
  `WeightingParameters.c_db_as_printed`, para que una evaluación que deba
  reproducir la tabla publicada al pie de la letra aún pueda. Fijado por
  `test_nmfs_2024_otariid_c_uses_the_corrected_1_36`.
- **Estado:** sin notificar (el organismo emisor ya lo tiene documentado).

---

## Southall et al. (2019), Aquatic Mammals 45(2), Tabla 7 (SPL de pico impulsivo)

*Artículo de revista con revisión por pares, no una norma.*

- **Ubicación:** Tabla 7 (p. 156 impresa), los criterios de umbral TTS y PTS
  para ruido impulsivo; las dos filas de carnívoros en el aire PCA y OCA.
- **El impreso:** PCA SPL de pico TTS 138 y SPL de pico PTS 144; OCA SPL de
  pico TTS 161 y SPL de pico PTS 167
  $\text{dB re } 20\ \mu\text{Pa}$.
- **El problema:** los cuatro son errores tipográficos. Las propias erratas
  de los autores (*Aquatic Mammals* 45(5), 569-572,
  DOI 10.1578/AM.45.5.2019.569) nombran los cuatro en la p. 569 impresa,
  «There are four typographical errors in Table 7 on page 156», y reimprimen
  la tabla corregida en la p. 570 impresa: PCA 155 y 161, OCA 170 y 176. Las
  mismas erratas corrigen además la columna encabezada «B» de la Tabla 5 al
  parámetro b de la Ec. (2), que igualmente llaman error tipográfico.
- **Evidencia:** las erratas mismas, que nombran cada valor erróneo y su
  sustituto, corroboradas por la propia regla de extrapolación del artículo.
  Nótese primero lo que *no* discrimina. La regla de pico PTS = pico TTS +
  6 dB de la p. 155 impresa la satisface también el par impreso
  ($144 - 138 = 6$, igual que $161 - 155 = 6$), así que no dice nada sobre
  qué par es el correcto. Tampoco la duplicación visible en las filas
  impresas, donde la entrada de SPL de pico TTS es igual a la entrada **SEL**
  de umbral PTS de la misma fila (PCA 123 / 138 / 138 / 144 y OCA 146 / 161 /
  161 / 167, leyendo SEL TTS, pico TTS, SEL PTS, pico PTS): para estas dos
  filas en el aire esa igualdad la fuerzan dos reglas que el artículo declara
  en la p. 155 impresa, ambas sumando 15 dB al mismo SEL TTS base, así que se
  cumpliría fueran cuales fueran los valores SEL. Una revisión anterior de
  esta entrada leía esa igualdad como la firma de un corrimiento de columna;
  es en cambio la tabla impresa siendo internamente consistente con el propio
  método en el aire del artículo, que es lo que hace de las erratas lo único
  que zanja el asunto.
  - **El valor.** Los números corregidos están cerca de lo que produce la
    regla de extrapolación del artículo, con la salvedad de que la regla no
    se declara para estas filas. La p. 155 impresa fija el umbral TTS de SPL
    de pico impulsivo de un grupo sin datos directos en el umbral de audición
    a la frecuencia de mejor sensibilidad $f_0$ más 159 dB, y restringe esa
    regla explícitamente a los grupos en el agua: «For other species groups
    **in water** (LF, SI, PCW, and OCW), 159 dB was added to the value of the
    hearing threshold at f₀». La desarrolla para PCW: «Peak SPL TTS onset was
    estimated as 212 dB re 1 µPa (53 dB at f₀ + 159 dB)». Evaluar el
    audiograma de grupo de la Tabla 2 en el $f_0$ de la Tabla 4 reproduce las
    tres filas en el agua que las erratas no tocan (SI 219.6 contra un 220
    publicado; PCW 212.5 contra 212; OCW 226.1 contra 226), lo que valida la
    regla donde el artículo la aplica. Extenderla a las dos filas de
    carnívoros en el aire, cosa que el artículo no hace, da PCA
    $-4.6\ \text{dB re } 20\ \mu\text{Pa}$ a 2.3 kHz y OCA
    $11.4\ \text{dB re } 20\ \mu\text{Pa}$ a 10 kHz, de donde **154.4** y
    **170.4**. Esos reproducen los corregidos 155 y 170 con margen de 0.6 dB
    y quedan a 16 dB y 9 dB de los impresos 138 y 161, que es lo que los hace
    corroborantes y no confirmatorios; nótese que 154.4 redondea a 154, no a
    155, y una revisión anterior de esta entrada afirmaba que redondeaba al
    valor corregido.
  - **Una segunda inconsistencia, sin reparar.** La p. 155 impresa declara
    que para los carnívoros en el aire específicamente «a nominal 15 dB
    offset is used ... between the SEL-based TTS threshold and the peak
    SPL-based threshold», lo que reproduce los *impresos* 138 y 161 desde la
    columna SEL. Esa frase, no la regla de +159 dB, es la que el propio
    método del artículo aplica a PCA y OCA. Las erratas resuelven el
    conflicto a favor de valores consistentes con la regla de +159 dB, así
    que dejan sin efecto la frase además de la tabla; la frase queda en pie
    en el artículo.
  Verificado en la página 31 del PDF (p. 155 impresa) y la página 32 del PDF
  (p. 156 impresa) de Southall et al. (2019), Aquatic Mammals 45(2), que
  llevan la restricción «in water (LF, SI, PCW, and OCW)», el desplazamiento
  de 15 dB en el aire en el mismo párrafo, las dos formulaciones de la regla
  de +6 dB, y la Tabla 7 del artículo con la fila PCA 123 / 138 / 138 / 144 y
  la fila OCA 146 / 161 / 161 / 167. Las erratas son una publicación propia,
  *Aquatic Mammals* 45(5), 569-572, encuadernada al final de la copia que
  distribuyen los autores: verificadas allí en la página 109 del PDF (p. 569
  impresa), que nombra los cuatro valores y sus sustitutos, y la página 110
  del PDF (p. 570 impresa), que reimprime la Tabla 7 con PCA 123 / 155 / 138
  / 161 y OCA 146 / 170 / 161 / 176.
- **Comportamiento de la biblioteca:** los valores corregidos por las erratas
  son los implementados en
  [`bioacoustics/weighting.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/underwater/bioacoustics/weighting.py),
  fijados por `test_southall_table_7_errata_values_are_implemented`, con la
  propia regla de +159 dB comprobada contra el audiograma en
  `test_southall_impulsive_peak_spl_is_threshold_at_f0_plus_159_db` para los
  grupos en el agua a los que el artículo la restringe y, por separado y con
  la extrapolación etiquetada como tal, para PCA y OCA.
- **Estado:** notificado por los propios autores (erratas publicadas en
  2019).

## Directiva (UE) 2015/996, Anexo II 2.3.2 (conversión de rugosidad en km/h)

- **Ubicación:** el párrafo «Definition» de *Wheel and rail roughness*
  (DO L 168, 1.7.2015, p. 19) y el primer párrafo tras la fórmula (2.3.11)
  (p. 21).
- **El impreso:** «it shall be converted to a frequency spectrum f = v/λ,
  where f is the centre band frequency of a given 1/3 octave band in Hz, λ is
  the wavelength in m, and **v is the train speed in km/h**», y, para el
  ruido de impacto, «using the relation λ = v/f, where f is the 1/3 octave
  band centre frequency in Hz and **v is the s-th vehicle speed of the t-th
  vehicle type in km/h**».
- **El problema:** dimensionalmente imposible. Una frecuencia en hercios es
  una velocidad en metros por segundo dividida por una longitud de onda en
  metros; leer la velocidad en km/h en $f = v/\lambda$ multiplica todas las
  frecuencias por 3,6, situando el espectro de rugosidad entero un factor 3,6
  demasiado alto en frecuencia, que es más de una octava y media.
- **Evidencia:** verificado en la página 19 del PDF (p. L 168/19 impresa) y
  la página 21 del PDF (p. L 168/21 impresa) de la Directiva (UE) 2015/996.
  La corrección de errores del DO L 5, 10.1.2018, p. 35 sustituye «km/h» por
  «m/s» en ambos lugares. La ecuación de flujo (2.3.2) sí toma de verdad su
  velocidad en km/h, que es lo que hace plausible la errata.
- **Comportamiento de la biblioteca:** `roughness_to_frequency` convierte la
  velocidad a m/s antes de dividir, como está corregido, y su docstring lo
  dice. La implementación de referencia que la Comisión publicó con el módulo
  fuente hace lo mismo, y los 123 casos de libro de cálculo commiteados no se
  reproducirían de otro modo.
- **Estado:** sin notificar (corregido por el organismo emisor en 2018).

## Directiva (UE) 2015/996, Appendix G, Tabla G-1, segunda tabla (símbolo equivocado)

- **Ubicación:** Tabla G-1, «Coefficients Lr,TR,i and Lr,VEH,i for rail and
  wheel roughness», segunda tabla (DO L 168, 1.7.2015, pp. 130-131).
- **El impreso:** la segunda tabla está encabezada **$L_{r,VEH,i}$**, el
  mismo símbolo que la primera.
- **El problema:** sus dos columnas son «EN ISO 3095:2013 (Well maintained
  and very smooth)» y «Average network (Normally maintained smooth)», que son
  las clases de rugosidad de carril E y M del dígito 2 del descriptor de vía
  de la Tabla [2.3.b]. La tabla es la rugosidad de **carril** $L_{r,TR,i}$,
  la cantidad que el propio título de la tabla anuncia y que de otro modo
  falta en el Appendix G.
- **Evidencia:** verificado en la página 130 del PDF (p. L 168/130 impresa)
  de la Directiva (UE) 2015/996:2015, la página que lleva el encabezado de la
  segunda tabla. La corrección de errores del DO L 5, 10.1.2018 la retitula
  $L_{r,TR,i}$, y la Directiva Delegada (UE) 2021/1226 de la Comisión, punto
  (20)(a) del Anexo, la reimprime bajo ese símbolo cuando la sustituye,
  verificado en la página 35 del PDF (p. L 269/99 impresa) de esa Directiva.
- **Comportamiento de la biblioteca:** `rail_roughness` devuelve la segunda
  tabla de G-1 como la rugosidad de carril de (2.3.7) y `wheel_roughness`
  devuelve la primera como la rugosidad de rueda, que es la única asignación
  bajo la cual las clases de la Tabla [2.3.b] pueden alcanzarse siquiera.
- **Estado:** sin notificar (corregido por el organismo emisor en 2018).

## Directiva (UE) 2015/996, Appendix G, Tabla G-5, fila de 6 350 Hz (muesca de 50 dB)

- **Ubicación:** Tabla G-5, «Coefficients LW,0,idling for traction noise», la
  fila de 6 350 Hz del par «Diesel locomotive (c. 2 200 kW)» (DO L 168,
  1.7.2015, p. 138).
- **El impreso:** Source A **31,4** dB y Source B **30,7** dB.
- **El problema:** ambos están unos 50 dB por debajo de sus propios vecinos
  en la misma columna: 90,5 / 89,5 dB a 5 000 Hz y 81,2 / 80,6 dB a
  8 000 Hz. Ninguna fuente de tracción física tiene una muesca de 50 dB de un
  tercio de octava de ancho, y ninguna otra columna de la tabla tiene nada
  comparable. Se perdió el dígito inicial 8.
- **Evidencia:** verificado en la página 138 del PDF (p. L 168/138 impresa)
  de la Directiva (UE) 2015/996:2015, que lleva las filas de 5 000, 6 350 y
  8 000 Hz y el encabezado de columna «Diesel locomotive (c. 2 200 kW)». La
  Directiva Delegada (UE) 2021/1226 de la Comisión, punto (20)(f) del Anexo,
  verificado en la página 39 del PDF (p. L 269/103 impresa) de esa Directiva,
  sustituye la 4.ª columna, 25.ª fila por «81,4» y la 5.ª columna, 25.ª fila
  por «80,7», restaurando la caída monótona. Los mismos dos valores aparecen
  como 31,41 y 30,71 en el fichero de catálogo IMAGINE que la Comisión
  distribuye con su módulo fuente de referencia, así que el error es anterior
  a la Directiva.
- **Comportamiento de la biblioteca:** publica los corregidos 81,4 / 80,7 y
  los fija, junto con la afirmación de que ningún valor dista más de 10 dB de
  cualquiera de sus vecinos, en
  `test_table_g5_carries_the_2021_correction_at_6300_hz`.
- **Estado:** sin notificar (corregido por el organismo emisor en 2021).

## Directiva (UE) 2015/996, Appendix G, etiquetas de bandas y longitudes de onda

- **Ubicación:** la columna de frecuencias de las Tablas G-3, G-5 y G-6 y la
  columna de longitudes de onda de la Tabla G-1 (DO L 168, 1.7.2015,
  pp. 129-140).
- **El impreso:** los centros de tercio de octava están etiquetados
  **316 Hz**, **3 160 Hz** y **6 350 Hz**, y las longitudes de onda
  **120 mm**, **12 mm**, **3,2 mm** y **1,2 mm**.
- **El problema:** ninguna de las dos series es la preferente. Los centros
  nominales de tercio de octava de IEC 61260-1 son 315, 3 150 y 6 300 Hz, y
  los números preferentes R10 en torno a esas longitudes de onda son 125,
  12,5, 3,15 y 1,25 mm. Los propios ficheros de catálogo de la Comisión,
  distribuidos con el módulo fuente de referencia, usan la serie de
  longitudes de onda preferente en todo su recorrido.
- **Evidencia:** verificado en las páginas 129, 130, 131, 133, 134, 135, 137
  y 138 del PDF (pp. L 168/129 a L 168/138 impresas) de la Directiva (UE)
  2015/996:2015, que llevan todas las apariciones de las cuatro longitudes de
  onda y de las tres etiquetas de banda. La Directiva Delegada (UE) 2021/1226
  de la Comisión, punto (20)(c) del Anexo, sustituye la sección
  $L_{H,TR,i}$ de la Tabla G-3 de raíz y los puntos (20)(d), (f) y (g)
  sustituyen las tres etiquetas de frecuencia en las secciones restantes y en
  las Tablas G-5 y G-6; las tablas que sustituye de raíz llevan las
  longitudes de onda preferentes. Pero el punto (20)(a) sustituye solo «the
  second table» de la Tabla G-1, así que las etiquetas de longitud de onda
  120, 12, 3,2 y 1,2 mm siguen en pie en la **primera** tabla de G-1, la
  rugosidad de rueda $L_{r,VEH,i}$, que es la única tabla que las conserva.
- **Comportamiento de la biblioteca:** la rejilla de frecuencias es la de
  IEC 61260-1 en todo su recorrido. Las rejillas de longitudes de onda se
  mantienen como están impresas, una por tabla, y cada espectro de rugosidad
  se remuestrea sobre su propia rejilla en lugar de forzarse a una común, que
  es para lo que están `_WAVELENGTHS_WHEEL` y `_WAVELENGTHS_STANDARD`; la
  diferencia entre las dos está fijada por
  `test_wheel_roughness_keeps_the_non_standard_wavelength_grid`.
- **Estado:** sin notificar (etiquetas de frecuencia corregidas por el
  organismo emisor en 2021; las etiquetas de longitud de onda de la rugosidad
  de rueda siguen en pie).

## Directiva (UE) 2015/996, Anexo II 2.3.2, chirrido en curva (extremos sin asignar)

- **Ubicación:** el párrafo *Squeal* (DO L 168, 1.7.2015, p. 21).
- **El impreso:** «The emission level to be used is determined for curves
  with radius below **or equal to** 500 m and for sharper curves and
  branch-outs of points with radii below 300 m», y después «squeal noise
  shall be considered by adding 8 dB for **R < 300 m** and 5 dB for
  **300 m < R < 500 m**».
- **El problema:** los dos intervalos abiertos dejan $R = 300\ \text{m}$ y
  $R = 500\ \text{m}$ sin exceso alguno, y $R = 500\ \text{m}$ está
  explícitamente dentro del alcance que el mismo párrafo acaba de fijar. Una
  curva de 500 m se cae por tanto de una regla escrita para incluirla.
- **Evidencia:** verificado en la página 21 del PDF (p. L 168/21 impresa) de
  la Directiva (UE) 2015/996:2015, en la que las dos desigualdades de la
  frase de la regla son estrictas mientras que la frase de alcance encima de
  ellas lee «below or equal to 500 m». La Directiva Delegada (UE) 2021/1226
  de la Comisión, punto (4)(b) del Anexo, verificado en la página 4 del PDF
  (p. L 269/68 impresa) de esa Directiva, sustituye el párrafo por una tabla
  cuyos intervalos son cerrados, «R <= 300 m» y «300 m < R <= 500 m».
- **Comportamiento de la biblioteca:** `curve_squeal_excess` implementa la
  tabla de 2021, así que $R = 300\ \text{m}$ devuelve 8 dB y
  $R = 500\ \text{m}$ devuelve 5 dB; los límites están fijados en
  `test_curve_squeal_rule_of_2021`.
- **Estado:** sin notificar (corregido por el organismo emisor en 2021).

---

## Allard & Atalla, Propagation of Sound in Porous Media 2e (2009), Ec. (6.85)

*Libro, no una norma.*

- **Ubicación:** sección 6.5.2 (p. 123 impresa), la segunda forma de la razón
  de velocidades de la onda de cizalla $\mu_3$.
- **El impreso:**
  $\mu_3 = (N\delta_3^2 - \omega^2\rho_{11})/(\omega^2\rho_{22})$, ofrecida
  como alternativa a la Ec. (6.84), $\mu_3 = -\rho_{12}/\rho_{22}$.
- **El problema:** las dos formas impresas no son iguales. Sustituir el
  número de onda de cizalla de la Ec. (6.83),
  $\delta_3^2 = (\omega^2/N)(\rho_{11}\rho_{22} - \rho_{12}^2)/\rho_{22}$, en
  la Ec. (6.85) impresa da $-\rho_{12}^2/\rho_{22}^2$, que es la Ec. (6.84)
  multiplicada por el factor espurio $\rho_{12}/\rho_{22}$. El denominador
  debería leer $\omega^2\rho_{12}$.
- **Evidencia:** la propia deducción del libro. La Ec. (6.80), p. 122
  impresa, es
  $-\omega^2\rho_{11}\psi_s - \omega^2\rho_{12}\psi_f = N\nabla^2\psi_s = -N\delta_3^2\psi_s$,
  así que $(N\delta_3^2 - \omega^2\rho_{11})\psi_s = \omega^2\rho_{12}\psi_f$
  y por tanto
  $\mu_3 = \psi_f/\psi_s = (N\delta_3^2 - \omega^2\rho_{11})/(\omega^2\rho_{12})$.
  Con esa lectura las dos formas concuerdan idénticamente allí donde
  $\rho_{12}$ no es cero; a $\rho_{12} = 0$ el cociente corregido es $0/0$
  mientras que la Ec. (6.84) sigue definida y da
  $\mu_3 = -\rho_{12}/\rho_{22} = 0$, que es el valor a usar allí. La forma
  impresa difiere en cambio de la Ec. (6.84) en el factor
  $\rho_{12}/\rho_{22}$, así que coincide con ella solo donde esa razón es
  exactamente 0 o exactamente 1. Con
  $\rho_{12}/\rho_{22} = \rho_0/(\phi\rho_\text{eq}) - 1$ esos dos casos
  piden $\rho_\text{eq} = \rho_0/\phi$ y $\rho_\text{eq} = \rho_0/(2\phi)$,
  ambos reales; la densidad efectiva de un medio poroso con pérdidas es
  compleja, así que ninguno se cumple jamás. Verificado en la página 132 del
  PDF (p. 123 impresa) de Allard & Atalla, Propagation of Sound in Porous
  Media 2e:2009, que lleva ambas formas impresas, y en la página opuesta para
  la Ec. (6.80).
- **Comportamiento de la biblioteca:** `biot_waves` implementa la Ec. (6.84)
  tal como está impresa, y
  `test_shear_velocity_ratio_matches_the_corrected_second_printed_form` la
  comprueba contra la Ec. (6.85) corregida sobre cuatro décadas de
  frecuencia, y afirma además que la forma exactamente como está impresa
  discrepa.
- **Estado:** sin notificar.

---

## Allard & Atalla 2e (2009), Ec. (11.48) y Tabla 11.1 (capa poroelástica)

*Libro, no una norma.*

- **Ubicación:** sección 11.3.3 (pp. 251-252 impresas), la tensión normal del
  fluido $\sigma_{33}^f$ de una capa poroelástica y la matriz $[\Gamma]$ que
  alimenta.
- **El impreso:** la Ec. (11.48) lee

  $$
  \sigma_{33}^f = \sum_i (Q + R\mu_i)(k_t^2 + k_{i3}^2)
  \left\{ -(A_i - A'_i)\cos(k_{i3}x_3) + j(A_i - A'_i)\sin(k_{33}x_3) \right\}
  $$

  y la Tabla 11.1 escribe $k_{i3}$ en las dos columnas que llevan $\mu_1$,
  $D_1$ y $E_1$.
- **El problema:** dos erratas independientes en la misma ecuación, más un
  desliz de subíndice en la tabla.
  - Falta el coeficiente de la amplitud *simétrica* $(A_i + A'_i)$: la
    Ec. (11.48) adjunta ambos términos a $(A_i - A'_i)$, lo que dejaría la
    primera y la tercera columna de $[\Gamma]$ sin entrada alguna de
    $\sigma_{33}^f$, contradiciendo la Tabla 11.1, cuya fila 6 imprime
    $-E_1 \cos(k_{13}x_3)$ y $-E_2 \cos(k_{23}x_3)$ exactamente en esas
    columnas. El primer término es $-(A_i + A'_i)\cos(k_{i3} x_3)$.
  - El seno lleva $k_{33}$, la componente del número de onda de *cizalla*,
    dentro de una suma sobre las dos ondas de compresión $i = 1, 2$. Debe ser
    $k_{i3}$. La Tabla 11.1 da de nuevo la lectura pretendida: su fila 6
    tiene $j E_1 \sin(k_{13}x_3)$ y $j E_2 \sin(k_{23}x_3)$, y cero en ambas
    columnas de cizalla, porque una onda de cizalla no produce dilatación y
    por tanto tampoco $\sigma_{33}^f$.
  - La Tabla 11.1 imprime el subíndice corredero $k_{i3}$ en sus dos primeras
    columnas, que pertenecen a la primera onda de compresión sola: los
    $\mu_1$, $D_1$ y $E_1$ de las mismas columnas hacen de $k_{13}$ la única
    lectura consistente.
- **Evidencia:** las dos lecturas de arriba las fuerza la Tabla 11.1, que la
  misma página declara ser la tabulación de las Ecs. (11.37), (11.38) y
  (11.46)-(11.48). Son también lo que da la relación tensión-deformación
  Ec. (11.41),
  $\sigma_{33}^f = R\,\mathrm{div}\,u_f + Q\,\mathrm{div}\,u_s$, cuando los
  potenciales de desplazamiento de las Ecs. (11.22)-(11.25) se derivan
  directamente. Verificado en la página 257 del PDF (p. 251 impresa) y la
  página 258 del PDF (p. 252 impresa) de Allard & Atalla, Propagation of
  Sound in Porous Media 2e:2009, que llevan la Ec. (11.48) y las dos columnas
  de la Tabla 11.1 tal como están impresas.
- **Comportamiento de la biblioteca:** la $[\Gamma]$ de la Tabla 11.1 está
  implementada con las lecturas corregidas, y
  `test_gamma_matches_the_field_rebuilt_from_the_potentials` comprueba sus
  treinta y seis entradas a tres frecuencias, tres profundidades y tres
  ángulos de incidencia contra el campo reconstruido desde las
  Ecs. (11.22)-(11.28) sin pasar por la tabla.
- **Estado:** sin notificar.

---

## Allard & Atalla 2e (2009), sección 6.6.3 (espesor de la segunda muestra)

*Libro, no una norma.*

- **Ubicación:** sección 6.6.3, p. 129 impresa, las dos muestras de lana de
  vidrio cuyas impedancias de superficie medidas y predichas son las Figuras
  6.10 y 6.11.
- **El impreso:** la primera frase dice que las impedancias se muestran «for
  l = 10 cm and l = 5.4 cm»; dos frases después el pico de la segunda muestra
  se sitúa a «860 Hz for l = 5.6 cm», y el pie de la Figura 6.11 dice «l =
  5.6 cm».
- **El problema:** los dos espesores no pueden estar bien a la vez.
- **Evidencia:** textual, y solo textual. Dos declaraciones impresas llevan
  5.6 cm, la frase del pico de 860 Hz y el pie independiente de la Figura
  6.11, contra una que lleva 5.4 cm; un solo desliz en la frase inicial es la
  explicación más corta que el mismo desliz cometido dos veces. Los números
  **no** lo zanjan, y esta entrada no pretende que lo hagan. El libro no da
  regla de localización de picos, y la respuesta sigue a la regla elegida:
  - Tomando el pico como el máximo de $\text{Im}(Z_s)$, la Ec. (6.107) sobre
    la lana de vidrio de la Tabla 6.1, completamente especificada, da
    863.5 Hz para 5.6 cm (+0.4 % contra los 860 impresos) y 896.2 Hz para
    5.4 cm (+4.2 %), lo que favorece 5.6 cm. Pero la misma regla sitúa la
    muestra indiscutida de 10 cm a 480.0 Hz contra sus 470 impresos, un sesgo
    de +2.1 % del mismo tamaño que el efecto que se resuelve.
  - Tomando el pico como el máximo de $|Z_s - Z_{s,\text{rigid}}|$, que es la
    desviación que el mismo párrafo describe («close to each other, except
    around the peaks which are not predicted by the one-wave model»), la
    muestra de 10 cm aterriza a 469.2 Hz (-0.2 %) y **ambas** frecuencias
    impresas salen entonces del par (10 cm, 5.4 cm): 861.2 Hz para 5.4 cm
    (+0.1 %) contra 831.0 Hz para 5.6 cm (-3.4 %). Esa regla favorece 5.4 cm.
  - Escalar el pico de 10 cm tampoco ayuda, y se inclina al lado contrario de
    la conclusión: $470 \times (10/5.4) = 870\ \text{Hz}$ está a 10 Hz de los
    860 publicados, $470 \times (10/5.6) = 839\ \text{Hz}$ a 21 Hz.
  - La concordancia de «860 Hz» con «5.6 cm» es en todo caso parcialmente
    circular, ya que ambos están en la misma cláusula: contrasta esa frase
    contra sí misma, no cuál de las dos frases es la errata.
  Verificado en la página 138 del PDF (p. 129 impresa) de Allard & Atalla,
  Propagation of Sound in Porous Media 2e:2009, en la que el 5.4 cm solitario
  y el 5.6 cm de la cláusula de 860 Hz están en la misma página, y en la
  página 139 del PDF (p. 130 impresa) de la misma edición para el pie de la
  Figura 6.11, la segunda frase que lleva 5.6 cm.
- **Comportamiento de la biblioteca:** registrado, sin efecto en la
  implementación.
  `test_impedance_peak_of_the_thin_layer_resolves_the_printed_thickness`
  fija el pico de 5.6 cm contra los 860 Hz publicados bajo la regla de
  $\text{Im}(Z_s)$ y comprueba que la lectura de 5.4 cm es la peor de las dos
  bajo esa regla.
- **Estado:** sin notificar, y la más débil de las cuatro entradas de aquí:
  la conclusión descansa en la lectura dos-contra-uno de la página impresa,
  no en un cálculo.

---

## Allard & Atalla 2e (2009), sección 6.5.4 (la razón de velocidades de la onda del esqueleto)

*Libro, no una norma.*

- **Ubicación:** sección 6.5.4, p. 125 impresa, la única frase del libro que
  cita valores calculados de $\mu_b$ para la lana de vidrio de la Tabla 6.1.
- **El impreso:** «The ratio modulus $|\mu_b|$ of the velocities of the frame
  and the air for the frame-borne wave decreases from 1.0 at 50 Hz to 0.82 at
  1500 Hz.»
- **El problema:** los dos valores citados son la *parte real* de $\mu_b$, no
  su módulo. $\mu_b$ es complejo, y la frase nombra el módulo
  explícitamente.
- **Evidencia:** sobre el material de la Tabla 6.1, completamente
  especificado, el modelo da $\mu_b(1500\ \text{Hz}) = 0.811 + 0.473j$. Su
  parte real es **0.811**, a 1.1 % del 0.82 impreso; su módulo es **0.939**,
  a 14.5 %. Leída como la parte real, la frase acierta en ambos extremos y
  describe un descenso monótono: $\text{Re}(\mu_b)$ es 1.002 a 50 Hz y pasa
  por 0.82 a 1467 Hz, a 2.2 % de los 1500 Hz impresos. Leída como el módulo
  no acierta en ninguno: $|\mu_b|$ es 1.002 a 50 Hz pero *sube* a 1.008 hacia
  400 Hz antes de darse la vuelta, y solo alcanza 0.82 a 2634 Hz, un 76 % por
  encima de la frecuencia impresa. Ninguna lectura admisible de las entradas
  impresas cierra esa brecha. Con el factor de pérdida a 0 o a 0.2, la
  longitud viscosa a la mitad o al doble, $\Lambda' = 2\Lambda$ en lugar del
  $1.1 \cdot 10^{-4}\ \text{m}$ impreso, la resistividad a la mitad o al
  doble, la tortuosidad a 1 o el coeficiente de Poisson a 0.3,
  $|\mu_b(1500)|$ se mueve solo entre 0.874 y 1.073. La más cercana de las
  ocho, 0.874 a factor de pérdida cero, sigue a 6.6 % del 0.82 impreso, y
  pierde por completo el cruce de ramas a 495 Hz de la misma sección; la
  única variante que conserva ese cruce ($\Lambda' = 2\Lambda$, 495.2 Hz)
  deja $|\mu_b|$ en 0.937. Leer la frase como $\text{Re}(\mu_b)$ no necesita
  variante alguna. Verificado en la página 134 del PDF (p. 125 impresa) de
  Allard & Atalla, Propagation of Sound in Porous Media 2e:2009, que lleva la
  frase y su 0.82.
- **Comportamiento de la biblioteca:** `biot_waves` calcula $\mu_b$ desde la
  Ec. (6.71) tal como está impresa. La fila de conformidad y
  `test_frame_borne_velocity_ratio_matches_the_two_published_values` están
  escritas contra $\text{Re}(\mu_b)$, y lo dicen.
- **Estado:** sin notificar.

---

## ECAC Doc 29, 5.ª ed., Volumen 2, Appendix B, Ec. (B-41) (deceleración en descenso)

- **Ubicación:** Appendix B, sección B7.1.1, la deceleración $a$ definida
  bajo la Ec. (B-41), en la página que lleva la Ec. (B-40) y la Ec. (B-41).
- **El impreso:**
  $a = k^2 \left(\left(\left(\mathrm{Pt1(NextSeg)}_{TAS} - w\right)/\cos\gamma\right)^2 - \left(\left(\mathrm{Point1}_{TAS} - w\right)/\cos\gamma\right)^2\right) / \left(2\left(\mathrm{Point1\_Height} - \mathrm{Pt1(NextSeg)\_Height}\right)/\sin\gamma\right)$,
  es decir, ambas velocidades respecto al suelo divididas por $\cos\gamma$
  sobre dos veces la longitud oblicua del segmento.
- **El problema:** la pendiente de descenso se cuenta dos veces. La
  deceleración media a lo largo de la trayectoria de vuelo es el cambio del
  cuadrado de la velocidad a lo largo del camino sobre dos veces la longitud
  del camino; la expresión impresa convierte las velocidades a valores a lo
  largo del camino *y* usa una longitud de camino que ya es la oblicua, así
  que sobreestima $|a|$ en $1/\cos^2\gamma$. La Ec. (B-21) de la 4.ª edición
  es autoconsistente, y el denominador no es lo que cambió: lee
  $2 \cdot \Delta s/\cos\gamma$ con $\Delta s$ «the ground distance covered»,
  que es la misma longitud oblicua que la 5.ª edición escribe como
  $2\left(\mathrm{Point1\_Height} - \mathrm{Pt1(NextSeg)\_Height}\right)/\sin\gamma$.
  Lo que cambió es el numerador. La Ec. (B-22) de la 4.ª edición define las
  velocidades que divide como *velocidades respecto al suelo*,
  $V = V_C\cos\gamma/\sqrt{\sigma} - w$, es decir, la velocidad verdadera
  resuelta en el plano horizontal, así que dividir cada una por $\cos\gamma$
  restituye correctamente una velocidad a lo largo del camino. La 5.ª edición
  alimenta la Ec. (B-41) con la propia $\mathrm{TAS}$ de los puntos del
  perfil, que ya es a lo largo del camino, y conservó la división. Los
  propios resultados de referencia del Doc 29 lo deciden: de los doce puntos
  del caso 2D del Volumen 3 Parte 2, volado enteramente con ese tipo de paso,
  nueve se alcanzan por la deceleración. Las velocidades respecto al suelo
  simples reproducen el empuje tabulado en cada uno de los doce, peor
  desviación 0.047 lb, mientras que las velocidades divididas impresas se
  quedan cortas en los nueve, por 6.05, 5.47, 4.02, 3.81, 4.56, 4.45, 0.29,
  6.35 y 6.41 lb en orden de perfil: siempre bajas, y nunca dentro de las
  0.05 lb de precisión impresa del propio libro de cálculo. El término de
  resistencia junto a ella sí conserva el $\cos\gamma$ que imprime la misma
  ecuación, cosa que los mismos puntos confirman a las mismas 0.05 lb.
- **Evidencia:** reproducción de la hoja `D1-(Arrival_Results)` del Volumen 3
  Parte 2, caso 2D, bajo cada lectura. Verificado en la página 104 del PDF
  (p. B-31 impresa) de ECAC.CEAC Doc 29, 5.ª ed., Volumen 2: Technical guide,
  que lleva la Ec. (B-40), la Ec. (B-41) y la deceleración bajo ella, y en la
  página 90 del PDF (p. B-15 impresa) de ECAC.CEAC Doc 29, 4.ª edición,
  Volumen 2, que lleva la Ec. (B-21) y, justo debajo, la Ec. (B-22) que
  define sus $V_1$ y $V_2$ como velocidades respecto al suelo y decide así la
  lectura.
- **Comportamiento de la biblioteca:** `flight_performance` calcula la
  deceleración desde las velocidades respecto al suelo simples sobre la
  longitud oblicua, y el docstring del helper lleva la desviación y los
  números de arriba. `test_arrival_case_reproduces_every_profile_point` fija
  los 124 puntos de llegada, y la fila de conformidad *ECAC Doc 29 Appendix B
  approach thrust* fija el empuje de descenso del caso 2A.
- **Estado:** sin notificar.

## ECAC Doc 29, 5.ª ed., Volumen 2, Appendix B, Ec. (B-18) (gradiente de pista)

- **Ubicación:** Appendix B, sección B6.1.1, la aceleración media $a$
  definida bajo la Ec. (B-18).
- **El impreso:** «$a$ is the average acceleration (ft/s$^2$) along the
  runway, equal to:
  $a = \left(V_C/\sqrt{\sigma}\right)^2/\left(2 \cdot s_{TOw}\right)$», con
  «$V_C$ is the Calibrated Airspeed (**kt**) at *Point2*» y $s_{TOw}$ en pies
  en la misma página.
- **El problema:** la expresión se declara en ft/s$^2$ y evalúa en kt$^2$/ft.
  El factor que falta es $k^2 = 1.68781^2 = 2.8487$, el cuadrado de la
  constante de nudos a pies por segundo que el Doc 29 fija en B2.2 y lleva
  explícitamente en la Ec. (B-24) y la Ec. (B-41), que construyen
  aceleraciones con el mismo tipo de expresión. No es cosmético: $a$ entra
  solo a través de $a/(a - g\,G_R)$, así que infravalorarla en 2.85
  sobrevalora la corrección de gradiente, y en una pendiente ascendente del
  1 % con $V_{CTO} = 162.65$ kt y $s_{TOw} = 4900$ ft los 7.69 ft/s$^2$
  dimensionalmente correctos dan un factor de 1.0437 contra el 1.1353 de la
  lectura literal: un 8.8 % de distancia de despegue. La 4.ª edición lleva la
  misma omisión, así que es heredada y no introducida, e imprime
  $\left(V_C\sqrt{\sigma}\right)^2$ donde la 5.ª imprime
  $\left(V_C/\sqrt{\sigma}\right)^2$; solo la colocación de la 5.ª edición es
  una velocidad que el avión tiene, ya que $V_C/\sqrt{\sigma}$ es la
  velocidad verdadera de la Ec. (B-7).
- **Evidencia:** análisis dimensional contra la Ec. (B-24) y la Ec. (B-41)
  del mismo documento. Verificado en la página 90 del PDF (p. B-17 impresa)
  de ECAC.CEAC Doc 29, 5.ª ed., Volumen 2: Technical guide, y, para la mitad
  heredada, en la página 86 del PDF (p. B-11 impresa) de ECAC.CEAC Doc 29,
  4.ª edición, Volumen 2, donde la misma definición está bajo la Ec. (B-11) y
  lee
  $\left(V_C\cdot\sqrt{\sigma}\right)^2/\left(2\cdot s_{TOw}\right)$,
  ft/s$^2$. Esta no puede arbitrarse contra los resultados de referencia: la
  hoja de casos de salida `C8-(Departure_Cases)` del Volumen 3 Parte 2 no
  tiene columna de gradiente de pista, así que los 17 casos de referencia se
  vuelan a $G_R = 0$, donde la Ec. (B-18) es la identidad.
- **Comportamiento de la biblioteca:** `flight_performance` restituye $k^2$ y
  toma la colocación de $\sqrt{\sigma}$ de la 5.ª edición; el docstring del
  helper declara ambas desviaciones y que ningún caso de referencia puede
  detectar ninguna de las dos.
- **Estado:** sin notificar.

## ECAC Doc 29, 5.ª ed., Volumen 2, Appendix B, Ec. (B-21) (velocidad a mitad de paso)

- **Ubicación:** Appendix B, sección B6.1.2, el empuje neto corregido a mitad
  de paso $\overline{CNT}$ definido bajo la Ec. (B-21), en la rama que lo
  calcula desde la Ec. (B-12), es decir, para todo avión que lleva la tabla
  de coeficientes de hélice. B4.1 y B4.2 reparten los turbohélices entre
  ellas sin declarar regla, así que este no es el mismo conjunto que «los
  turbohélices»: de los 20 de ANP v2.3, 11 están en la tabla de hélice y
  llegan a la Ec. (B-12), los otros 9 están en la tabla de reactores y llegan
  a la Ec. (B-9), y los 8 aviones de pistón están todos en la tabla de
  hélice.
- **El impreso:** $\overline{CNT}$ «is the Corrected Net Thrust of the
  aircraft when being located at mid-step, i.e. at the altitude
  $Alt = E_{Apt} + \left(\mathrm{Point\ 1\_Height} + \mathrm{Point\ 2\_Height}\right)/2$»,
  y después, bajo «In the case of Eq. B-12,»,
  $V_T = \sqrt{0.5\left(\left(\mathrm{Point\ 2\_TAS}\right)^2 + \left(\mathrm{Point\ 1\_TAS}\right)^2\right)}$,
  la media cuadrática de las dos velocidades verdaderas de los extremos.
- **El problema:** la velocidad contradice la altitud nombrada una línea más
  arriba. Un paso Climb se vuela a una velocidad calibrada mantenida, así que
  la velocidad verdadera a la altitud de mitad de paso está fijada y es
  $V_C/\sqrt{\sigma}$ evaluada en la $\sigma$ de mitad de paso (Ec. B-7); la
  media cuadrática de los dos valores de los extremos es un número distinto.
  Las dos ramas de la misma lista describen por tanto dos aviones distintos
  en el único punto que ambas llaman mitad de paso: la rama de reactores
  imprime
  $V_C = \mathrm{Point\ 2\_TAS} \cdot \sqrt{\sigma_{Point\ 2}}$, que es la
  propia velocidad calibrada mantenida del paso y sitúa así el avión a
  $V_C/\sqrt{\sigma}$ a media subida, mientras que la rama de la Ec. (B-12)
  lo sitúa en la media cuadrática de los extremos. La media se lee además
  como trasplantada. La sección B6.1.3, para el paso Accelerate, está
  construida exactamente sobre esta media cuadrática y es autoconsistente con
  ella, dando a *ambas* ramas la misma
  $\overline{V_T} = \sqrt{\left(V_{T2}^2 + V_{T1}^2\right)/2}$ y
  convirtiéndola para la forma de reactores con la
  $\sqrt{\sigma_{Alt}}$ de mitad de paso; B6.1.2 conserva la línea de la
  Ec. (B-12) pero sustituye la línea de reactores por una cantidad de
  Point 2, y solo una de las dos sobrevive a la sustitución. De los
  candidatos el impreso es el mayor: a $V_C$ constante la velocidad verdadera
  crece convexamente con la altitud, así que la media cuadrática supera a la
  media aritmética, que supera al valor de media altitud. La Ec. (B-12) hace
  el empuje inversamente proporcional a $V_T$, así que la velocidad impresa
  infravalora el empuje a mitad de paso, infravalora $\sin\gamma$ y tiende la
  subida larga.
- **Evidencia:** reproducción de la hoja `D2-(Departure_Results)` del
  Volumen 3 Parte 2 bajo cada lectura. Los cuatro casos de salida de
  turbohélice son los únicos datos de referencia que llegan a esta rama, y
  son unánimes. En el caso 56 el último punto del perfil está impreso a
  400814.3 ft: la lectura de altitud de mitad de paso lo deja a 0.001 ft, la
  media aritmética 323.944 ft largo y la media cuadrática impresa 544.944 ft
  largo, contra los 0.15 ft con los que las distancias de salida casan por lo
  demás. Los casos 8, 28 y 68 dejan el mismo punto final 172.333, 223.003 y
  544.944 ft largo bajo la lectura impresa y 102.535, 132.673 y 323.944 ft
  largo bajo la media aritmética, siempre largo y nunca cerca de la precisión
  impresa; la lectura de altitud de mitad de paso queda a 0.049 ft de todos
  los puntos de los cuatro casos, peor caso el 28. La desviación crece con la
  altura del paso, como debe un error de convexidad: en el caso 8 la lectura
  impresa deja $V_T$ 0.0225 kt por encima del valor de mitad de paso en la
  subida de 1500 ft y 0.1066 kt por encima en la de 2500 ft. Verificado en la
  página 92 del PDF (p. B-19 impresa) de ECAC.CEAC Doc 29, 5.ª ed.,
  Volumen 2: Technical guide, que lleva la frase de la altitud de mitad de
  paso, la $V_C$ de la rama de reactores y la $V_T$ de la rama de la
  Ec. (B-12) en una misma página, y en la página 95 del PDF (p. B-22 impresa)
  del mismo documento para el par de B6.1.3 del que la línea de la Ec. (B-21)
  parece extraída.
- **Comportamiento de la biblioteca:** `flight_performance` evalúa la forma
  de hélice a la velocidad verdadera que el avión tiene a la altitud de mitad
  de paso, y el comentario del helper del paso Climb cita la expresión
  impresa, dice que el modelo se aparta de ella y apunta aquí.
  `test_departure_case_reproduces_every_profile_point` fija los 190 puntos de
  salida, cuatro de cuyos casos se vuelan sobre la Ec. (B-12).
- **Estado:** sin notificar. De las tres desviaciones del Appendix B
  registradas aquí esta es la que los resultados de referencia deciden con
  más nitidez, y la única que cambia un perfil distribuido.

## ANSI S1.4-1983, Table V, celda de 20 Hz tipo 2 (un signo más que perdió su barra)

- **Ubicación:** apartado 5.2, Table V «Tolerance limits on relative response
  levels for sound at random incidence measured on an instrument's
  calibration range», fila de 20 Hz, columna de tipo 2 (p. 6 impresa).
- **El impreso:** la celda lee «**+ 3**», sin segundo término. Sus vecinas de
  columna a 10, 12.5 y 16 Hz leen «+ 5, − ∞», y las celdas de tipo 0 y tipo 1
  de su propia fila leen «± 2» y «± 2.5».
- **El problema:** la tabla tiene una sola notación para un límite solo
  superior, un par «+ n, − ∞», y se usa tres filas por encima de esta celda
  en la misma columna. Esta celda no usa ni esa notación ni el «± n» de su
  fila, así que es o un límite escrito en una forma que la tabla no usa en
  ningún otro sitio o un «±» cuya barra no llegó a imprimirse. La Table V de
  IEC 651:1979, de la que esta tabla es la contraparte estadounidense y con
  la que la columna de tipo 2 concuerda en las otras treinta y tres filas,
  imprime «**±3**» exactamente en esta celda. La lectura pretendida es
  ±3 dB.
- **Evidencia:** la celda y sus vecinas de columna, leídas en la página 16
  del PDF (p. 6 impresa) de ANSI S1.4-1983, contra la misma celda en la
  página 10 del PDF (p. 8 impresa, marcada «[IEC page 19]») de BS 5969:1981,
  la adopción británica idéntica de IEC 651:1979.
- **Comportamiento de la biblioteca:** `_ANSI_S14_TABLE5_12` en
  [`weighting_compliance.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/filters/weighting_compliance.py)
  y su gemela de `reference_data` llevan −3 dB como límite inferior de tipo 2
  a 20 Hz, la más estricta de las dos lecturas, con la nota al lado.
  `test_b_masks_match_reference_data` fija las dos transcripciones una contra
  otra. Ningún veredicto distribuido se mueve: la ponderación B realizada
  queda 0,05 dB por debajo de la nominal a 20 Hz y pasa cualquiera de las dos
  lecturas.
- **Estado:** sin notificar.


---

## ISO 3747:2010, E.4.2.6.2 (el signo del nivel de campo directo)

- **Ubicación:** Anexo E (informativo), E.4.2.6.2 «Excess sound pressure,
  measurement distance effect, $\delta_r$», la frase que da la presión
  radiada directamente y las dos frases que se construyen sobre ella.
- **El impreso:** «the directly radiated pressure is approximately
  $L_{p,\mathrm{direct}} = L_W + 10 \lg(2\pi r^2/r_0^2)$ dB. Rearranging
  Equation (A.1) using $L_{p,\mathrm{direct}}$, gives
  $L_{p(\mathrm{RSS}),r} = L_{p,\mathrm{direct}} + \Delta L_f - 3$ dB», y el
  coeficiente de sensibilidad que sigue,
  $c_r = 10^{-0{,}1(\Delta L_f - 3\ \mathrm{dB})}\,8{,}7/r$.
- **El problema:** el campo directo de una fuente sobre un plano reflectante
  decae con la distancia, $L_{p,\mathrm{direct}} = L_W - 10 \lg(2\pi
  r^2/r_0^2)$ dB; el signo más impreso lo hace crecer. Las dos frases
  siguientes solo se sostienen con el signo menos. Sustituyendo
  $L_{p,\mathrm{direct}} = L_W - 20 \lg(r/r_0) - 8$ dB en la Ec. (A.1)
  reordenada, $L_{p(\mathrm{RSS}),r} = L_{W(\mathrm{RSS})} + \Delta L_f - 11 -
  20 \lg(r/r_0)$ dB, sale el $L_{p,\mathrm{direct}} + \Delta L_f - 3$ dB
  impreso, mientras que el signo más da
  $L_{p,\mathrm{direct}} + \Delta L_f - 19\ \mathrm{dB} - 40 \lg(r/r_0)$; y el
  $8{,}7/r$ del coeficiente de sensibilidad es $20/(r \ln 10)$, la derivada de
  $-20 \lg r$, así que el $c_r$ impreso es la derivada de la forma con signo
  menos. Una errata de signo en un anexo informativo.
- **Evidencia:** las tres frases consecutivas de E.4.2.6.2 leídas una contra
  otra y contra la Ec. (A.1). Verificado en la página 47 del PDF (p. 38
  impresa) y en la página 30 del PDF (p. 21 impresa) de BS EN ISO 3747:2010.
- **Comportamiento de la biblioteca:** el presupuesto de incertidumbre del
  Anexo E no está modelado; la biblioteca evalúa la Ec. (A.1) tal como está
  impresa
  ([`excess_sound_pressure_level`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_situ.py)),
  a la que la errata no afecta. No cambia ningún número.
- **Estado:** sin notificar.

## ISO 3747:2010, E.4.2.5 (la corrección por altitud citada contra el Anexo C)

- **Ubicación:** Anexo E (informativo), E.4.2.5 «Radiation impedance
  correction, $C_2$», las frases que dimensionan $u_{C_2}$.
- **El impreso:** «For altitudes less than 500 m above sea level, no
  meteorological correction is required. At 120 m altitude and 23 °C, the
  correction is 0 dB and at 500 m altitude, the correction is 0,6 dB.
  Assuming a triangular distribution for this uncertainty, the standard
  deviation is $u_{C_2} = 0{,}6/\sqrt{6} = 0{,}3$ dB.»
- **El problema:** el Anexo C, normativo, define la corrección como
  $C_2 = -10 \lg(p_\mathrm{s}/p_{\mathrm{s},0}) + 15 \lg[(273{,}15 +
  \theta)/\theta_\mathrm{ref}]$ con la presión estática de la Ec. (C.2),
  $p_\mathrm{s} = p_{\mathrm{s},0}\,(1 - aH_\mathrm{a})^b$. A 23 °C eso da
  0,07 dB a 120 m ($p_\mathrm{s}$ = 99,89 kPa, de los que el término de
  presión $-10 \lg(p_\mathrm{s}/p_{\mathrm{s},0})$ son 0,06 dB) y 0,26 dB a
  500 m ($p_\mathrm{s}$ = 95,46 kPa), no los 0,6 dB impresos, y la aritmética
  impresa a continuación tampoco cierra: $0{,}6/\sqrt{6} = 0{,}245$, impreso
  0,3. En el Anexo C no aparece ninguna altitud por debajo de la cual «no se
  requiera corrección meteorológica». El ejemplo informativo es inconsistente
  con el anexo normativo que cita.
- **Evidencia:** recálculo de la Ec. (C.2) y de $C_2$ a partir de las
  constantes impresas ($a$ = 2,2560 × 10⁻⁵ m⁻¹, $b$ = 5,255 3,
  $p_{\mathrm{s},0}$ = 1,013 25 × 10⁵ Pa, $\theta_\mathrm{ref}$ = 296 K).
  Verificado en la página 46 del PDF (p. 37 impresa) y en la página 36 del
  PDF (p. 27 impresa) de BS EN ISO 3747:2010.
- **Comportamiento de la biblioteca:** implementa el Anexo C tal como está
  impreso:
  [`static_pressure_from_altitude`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_situ.py)
  evalúa la Ec. (C.2) y el `c2` del resultado la corrección, así que un
  emplazamiento a 500 m recibe los 0,26 dB que da el anexo. El presupuesto
  del Anexo E no está modelado. Fijado por
  `test_static_pressure_from_altitude_eq_c2` en
  [`tests/emission/test_sound_power_in_situ.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_in_situ.py)
  y por la comprobación de conformidad «ISO 3747:2010 Eq. C.2».
- **Estado:** sin notificar.

## ISO 3747:2010, Tabla E.2 (el exceso que perdió su delta)

- **Ubicación:** Anexo E (informativo), Tabla E.2 «Uncertainty budget for
  determinations of $\sigma_{R0}$...», la celda del coeficiente de
  sensibilidad de la fila $\delta_r$ (distancia de medición).
- **El impreso:** $c_i = 10^{-0{,}1(L_f - 3)}\,8{,}7/r$.
- **El problema:** la magnitud del exponente es el *exceso* de nivel de
  presión acústica sobre el campo libre, el $\Delta L_f$ de la Ec. (A.1), no
  un nivel $L_f$; ninguna magnitud llamada $L_f$ está definida en la norma.
  El apartado E.4.2.6.2, que deduce ese mismo coeficiente, lo imprime como
  $c_r = 10^{-0{,}1(\Delta L_f - 3\ \mathrm{dB})}\,8{,}7/r$, y su caso extremo
  resuelto ($\Delta L_f$ = 7,1 dB, $r$ = 6 m) reproduce el 0,6 que allí se
  cita solo con el exceso en el exponente ($10^{-0{,}41} \times 8{,}7/6 =
  0{,}564$). En la tabla se perdió la delta.
- **Evidencia:** la celda de la tabla leída contra el texto que la deduce,
  verificado en la página 44 del PDF (p. 35 impresa) y en la página 47 del PDF
  (p. 38 impresa) de BS EN ISO 3747:2010. La Tabla E.2 es propia de esta
  parte: la fila correspondiente de ISO 3744:2010 lleva el coeficiente de
  campo libre $c_S = 8{,}7/r$ sin factor de exceso alguno, así que el desliz
  no viene heredado de la familia.
- **Comportamiento de la biblioteca:** el presupuesto de incertidumbre del
  Anexo E no está modelado, y el exceso se evalúa desde la Ec. (A.1) en
  [`excess_sound_pressure_level`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_situ.py).
  No cambia ningún número.
- **Estado:** sin notificar.

## ISO 3747:2010, Tabla E.2 (el coeficiente de muestreo que su propio apartado contradice)

- **Ubicación:** Anexo E (informativo), Tabla E.2 «Uncertainty budget for
  determinations of $\sigma_{R0}$...», la celda del coeficiente de
  sensibilidad de la fila $\delta_\mathrm{mic}$ (muestreo).
- **El impreso:** $c_i = 0{,}5$.
- **El problema:** el apartado E.4.2.6.3, que deduce esa misma fila, imprime lo
  contrario junto con su razón: «Sampling directly affects the total
  uncertainty so $c_\mathrm{mic} = 1$». El presupuesto de E.4.2.12 se pone del
  lado del apartado y no de la tabla: su sexto término es $0{,}7^2$, que es la
  contribución de 0,7 dB que cita E.4.2.6.3 tomada con $c_\mathrm{mic} = 1$. La
  fila vecina zanja que el 0,5 no es una convención general de las filas de
  instrumentación, porque E.4.2.7 fija $c_\mathrm{slm} = 0{,}5$ *y se lo gana*:
  las lecturas repetidas con un mismo sonómetro dejan que los errores
  sistemáticos se cancelen, lo que reduce el coeficiente a la mitad, y el
  apartado reproduce entonces el término que la propia tabla suma
  ($0{,}5 \times 0{,}5 = 0{,}25$ dB, citado allí como 0,3 dB para cada una de
  las dos fuentes, y $\sqrt{0{,}3^2 + 0{,}3^2} = 0{,}42$ dB, el 0,4 que suma
  E.4.2.12). La fila de muestreo no lleva deducción semejante, y no puede
  llevarla: $\delta_\mathrm{mic}$ está definido sobre la *diferencia*
  $\Delta L'_{p(\mathrm{ST-RSS})} = L'_{p(\mathrm{ST})} - L'_{p(\mathrm{RSS})}$,
  que ya abarca las dos fuentes, así que no hay una segunda contribución que
  partir por la mitad. La familia da la razón al apartado: la fila
  $\delta_\mathrm{mic}$ correspondiente de la Tabla H.2 de ISO 3744:2010 lleva
  $c_i = 1$, y su H.4.2.9 imprime también $c_\mathrm{mic} = 1$.
- **Evidencia:** la celda de la tabla, el apartado que la deduce y el
  presupuesto que la suma, leídos en las páginas 44, 47 y 50 del PDF (pp. 35,
  38 y 41 impresas) de BS EN ISO 3747:2010; la comparación con la familia en
  las páginas 79 y 82 del PDF (pp. 70 y 73 impresas) de BS EN ISO 3744:2010.
- **Comportamiento de la biblioteca:** el presupuesto de incertidumbre del
  Anexo E no está modelado. La reproducibilidad que publica la biblioteca es
  el $\sigma_{R0}$ tabulado de la Tabla 2, leído por grado de exactitud.
  No cambia ningún número.
- **Estado:** sin notificar.

## ISO 3747:2010, E.4.2.3 (la ecuación de la que se toma la derivada)

- **Ubicación:** Anexo E (informativo), E.4.2.3 «Sound pressure measurement
  repeatability, $\overline{L'_{p(\mathrm{ST})}}$», la frase que introduce el
  coeficiente de sensibilidad $c_{L'_{p(\mathrm{ST})}}$.
- **El impreso:** «It is obtained from the derivative of
  $L_{W\mathrm{ref,atm}}$ [Equation (E.1)], with respect to
  $\overline{L'_{p(\mathrm{ST})}}$.»
- **El problema:** la Ecuación (E.1) es la desviación típica de las
  condiciones de funcionamiento y montaje, $\sigma_\mathrm{omc} =
  \sqrt{\frac{1}{N-1}\sum (L_{p,j} - L_{p\mathrm{av}})^2}$, que no contiene
  ningún $L_{W\mathrm{ref,atm}}$ y no puede derivarse respecto de
  $\overline{L'_{p(\mathrm{ST})}}$. El modelo que lleva
  $L_{W\mathrm{ref,atm}}$ es la Ecuación (E.2), impresa en la página
  siguiente, y derivarla (con $K_1$ sustituido desde la Ec. 7) sí da el
  $c_{L'_{p(\mathrm{ST})}} = 1 + 1/(10^{0,1\Delta L_p} - 1)$ impreso. Una
  errata de referencia cruzada: (E.1) por (E.2).
- **Evidencia:** verificado en la página 45 del PDF (p. 36 impresa), que lleva
  la frase y el coeficiente, contra la página 41 del PDF (p. 32 impresa) para
  la Ec. (E.1) y la página 42 del PDF (p. 33 impresa) para la Ec. (E.2), de
  BS EN ISO 3747:2010. ISO 3741:2010 imprime el mismo coeficiente como «la
  derivada de $L_W$ respecto de $L'_{p(\mathrm{ST})}$», sin número de
  ecuación, así que el número equivocado es propio de esta parte.
- **Comportamiento de la biblioteca:** el presupuesto de incertidumbre del
  Anexo E no está modelado, así que ningún número de la biblioteca depende de
  él. Se registra para que un lector futuro que siga la deducción no acabe en
  la ecuación equivocada.
- **Estado:** sin notificar.

## ISO 5136:2003, Tabla A.5, fila de 5 000 Hz (falta el primer dígito de $a_3$)

- **Ubicación:** Anexo A, Tabla A.5, «Values of coefficients $a_i$ for the
  determination of the combined mean flow velocity and modal correction
  $C_{3,4}$ of the sampling tube for duct diameters 0,8 m $\le d <$ 1,25 m»,
  fila de 5 000 Hz, columna $a_3$.
- **El impreso:** $- ,24 \times 10^{-05}$: un signo menos, un espacio, una
  coma decimal y dos dígitos, sin ningún dígito antes de la coma. Todas las
  demás celdas de las doce tablas de coeficientes de los Anexos A, H e I
  imprimen un dígito antes de la coma.
- **El problema:** el coeficiente no puede leerse del documento, y la fila
  está dentro del rango normativo de la norma (5 000 Hz, $|U| \le 40$ m/s).
  El $a_3$ de la misma banda en las dos tablas vecinas es
  $-1{,}17 \times 10^{-5}$ (Tabla A.4, de 0,5 m a 0,8 m) y
  $-1{,}27 \times 10^{-5}$ (Tabla A.6, de 1,25 m a 2 m), que encierran
  $-1{,}24 \times 10^{-5}$; un primer dígito de 2 o mayor movería $C_{3,4}$
  a 40 m/s en 0,64 dB por unidad del dígito ($a_3 U^3$ con
  $U^3 = 6{,}4 \times 10^4$), cosa que ninguna banda ni tabla vecina
  respalda.
- **Evidencia:** la celda tal como está impresa. Página 39 del PDF (p. 29
  impresa) de ISO 5136:2003, frente a la misma celda de la Tabla A.4 en la
  página 38 del PDF (p. 28 impresa) y de la Tabla A.6 en la página 40 del
  PDF (p. 30 impresa).
- **Comportamiento de la biblioteca:** lee $-1{,}24 \times 10^{-5}$, el
  valor que encierran las vecinas, en `_TABLE_A5` de
  [`sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_duct.py).
  El comentario de la tabla y
  `test_table_a5_5000_hz_reads_the_missing_digit_as_one` en
  [`tests/emission/test_sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_in_duct.py)
  dicen que es una lectura y no el impreso; un ejemplar de la norma en el
  que el dígito haya sobrevivido lo zanjaría.
- **Estado:** sin notificar.

## ISO 5136:2003, Anexo D, Anexo H y Anexo I ($C_{3,4}$ «according to Equation (3)»)

- **Ubicación:** la primera frase del Anexo D, y la frase del Anexo H y del
  Anexo I que introduce sus tablas de coeficientes.
- **El impreso:** «For $d$ = 0,5 m, the values of the coefficients $a_i$ for
  the calculation of $C_{3,4}$ according to Equation (3) are given in Table
  A.4» (Anexo D); «Values for the coefficients $a_i$ necessary to compute the
  mean flow velocity-modal corrections $C_{3,4}$ according to Equation (3)
  are given in Tables H.1 to H.3» (Anexo H) y «... in Tables I.1 to I.3»
  (Anexo I).
- **El problema:** la Ecuación (3) es la frecuencia de corte del primer modo
  transversal, $f_{1,0} = 0{,}586\,(c/D)\sqrt{1 - (U/c)^2}$, en la
  definición de 3.10. El polinomio en $U$ cuyos coeficientes contienen las
  tablas es la Ecuación (7) del apartado 5.3.3.4. El mismo número erróneo se
  imprime tres veces.
- **Evidencia:** páginas 45, 64 y 68 del PDF (pp. 35, 54 y 58 impresas) de
  ISO 5136:2003, frente a la Ecuación (3) en la página 16 del PDF (p. 6
  impresa) y la Ecuación (7) en la página 28 del PDF (p. 18 impresa).
- **Comportamiento de la biblioteca:** evalúa la Ecuación (7);
  [`flow_modal_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_duct.py)
  la cita. Ningún número cambia.
- **Estado:** sin notificar (defecto de referencia cruzada, sin consecuencia
  numérica).

## ISO 5136:2003, Anexo B, B.2 paso 4 ($\Delta L_{\max}$ «given in Table C.1»)

- **Ubicación:** Anexo B, apartado B.2, «Comparative procedure using a
  microphone fitted with a nose cone and a microphone fitted with a sampling
  tube», paso 4.
- **El impreso:** «Check whether the difference between the circumferentially
  averaged sound pressure levels obtained with the nose cone and the sampling
  tube ($\overline{L_{p\mathrm{NC}}} - \overline{L_{p\mathrm{ST}}}$) is
  smaller than or equal to the maximum allowable difference $\Delta L_{\max}$
  given in Table C.1.»
- **El problema:** la Tabla C.1 es la ponderación A $C_j$ del Anexo C y no
  contiene ningún $\Delta L_{\max}$. La tabla de la diferencia máxima
  admisible frente a la supresión del ruido de turbulencia
  $\Delta L_\mathrm{t}$ del tubo de muestreo es la Tabla B.1, en la página
  siguiente al paso, y el párrafo dos por encima de los pasos ya remite a
  ella («see Table B.1»).
- **Evidencia:** página 41 del PDF (p. 31 impresa) de ISO 5136:2003, con la
  Tabla B.1 en la página 42 del PDF (p. 32 impresa) y la Tabla C.1 en la
  página 44 del PDF (p. 34 impresa).
- **Comportamiento de la biblioteca:** el procedimiento de relación
  señal-ruido del Anexo B es una cualificación de la medición, no un término
  de $L_W$, y no está implementado. No hizo falta ningún cambio.
- **Estado:** sin notificar (defecto de referencia cruzada, sin consecuencia
  numérica).

## ISO 5136:2003, Anexo B, B.1 («the determination of the combined mean flow velocity»)

- **Ubicación:** Anexo B, apartado B.1, «General», la primera frase.
- **El impreso:** «Two procedures for the determination of the combined mean
  flow velocity are given in B.2 and B.3.»
- **El problema:** el anexo se titula «Determination of the signal-to-noise
  ratio of sound vs. turbulent pressure fluctuation in the test duct», y B.2
  y B.3 determinan esa relación; nada en el anexo determina una «combined
  mean flow velocity», una expresión que es un fragmento de la «combined
  mean flow velocity and modal correction» del apartado 5.3.3.4. La frase
  también cuenta dos procedimientos donde el anexo, con el método de
  coherencia con el que cierra, da tres.
- **Evidencia:** página 41 del PDF (p. 31 impresa) de ISO 5136:2003, el
  título del anexo y la frase en la misma página, y el procedimiento de
  coherencia en la página 43 del PDF (p. 33 impresa).
- **Comportamiento de la biblioteca:** el Anexo B no está implementado; nada
  que cambiar.
- **Estado:** sin notificar (defecto de redacción).

## ISO 5136:2003, apartado 7.4 NOTA (el «diámetro hidráulico» $D_\mathrm{h} = \sqrt{S_{\mathrm{f}2}/\pi}$)

- **Ubicación:** apartado 7.4, la NOTA que sigue a la regla del conducto de
  impulsión para ventiladores grandes de la categoría de instalación D.
- **El impreso:** «The hydraulic diameter of the fan outlet area,
  $S_{\mathrm{f}2}$, is given by $D_\mathrm{h} = \sqrt{S_{\mathrm{f}2}/\pi}$».
- **El problema:** $\sqrt{S/\pi}$ es el radio del círculo de área $S$; su
  diámetro es $\sqrt{4S/\pi} = 2\sqrt{S/\pi}$. Seguida tal como está
  impresa, la longitud «2 $D_\mathrm{h}$» que el apartado pide al conducto de
  impulsión es un diámetro equivalente, no dos, y si la regla pretendida son
  dos diámetros o dos radios no puede zanjarse desde el documento.
- **Evidencia:** página 33 del PDF (p. 23 impresa) de ISO 5136:2003.
- **Comportamiento de la biblioteca:** las longitudes de conducto de los
  apartados 5.2 y 7.4 son geometría de la instalación y no se calculan; nada
  que cambiar.
- **Estado:** sin notificar.

## ISO 5136:2003, Tabla A.2, cabecera de coeficientes (la columna $a_9$ se titula $a9_0$)

- **Ubicación:** Anexo A, Tabla A.2, «Values of coefficients $a_i$ for the
  determination of the combined mean flow velocity and modal correction
  $C_{3,4}$ of the sampling tube for duct diameters 0,2 m $\le d <$ 0,3 m»,
  la fila de cabecera de las columnas de coeficientes, décima columna.
- **El impreso:** una $a$ en cursiva, un 9 en cursiva sobre la línea base y
  un 0 como subíndice, entre un $a_8$ y un $a_{10}$ de la misma fila que sí
  llevan su índice como subíndice.
- **El problema:** un subíndice cero de más en una columna que es $a_9$. La
  misma columna se titula $a_9$ en las Tablas A.1 y A.3 a A.6, la NOTA de
  cada una de ellas suma $a_i U^i$ de $i = 0$ a $i = 10$ sobre las once
  columnas que tiene la fila, y la única celda que esta contiene, el
  $4{,}09 \times 10^{-14}$ de la fila de 20 000 Hz, es el coeficiente de
  $U^9$: un $a_{90}$ no tendría lugar alguno en esa suma.
- **Evidencia:** página 36 del PDF (p. 26 impresa) de ISO 5136:2003, frente a
  la fila de cabecera de la Tabla A.1 en la página 35 del PDF (p. 25
  impresa).
- **Comportamiento de la biblioteca:** la columna se lee como $a_9$.
  `_TABLE_A2` en
  [`sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_duct.py)
  lleva la fila de 20 000 Hz como los diez coeficientes $a_0$ a $a_9$, y
  `test_table_a2_20_khz_row_reads_the_last_column_as_a9` en
  [`tests/emission/test_sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_in_duct.py)
  desarrolla la fila. No cambia ningún valor de coeficiente.
- **Estado:** sin notificar (tipográfico, sin consecuencia numérica).

## ISO 5136:2003, Tabla A.6, fila de 16 000 Hz ($a_1$ impreso con el signo de multiplicación duplicado)

- **Ubicación:** Anexo A, Tabla A.6, «... for duct diameters 1,25 m $\le d
  \le$ 2 m», fila de 16 000 Hz, columna $a_1$.
- **El impreso:** $4{,}52 \times\!\times 10^{-01}$, dos signos de
  multiplicación donde todas las demás celdas imprimen uno.
- **El problema:** solo tipográfico; la mantisa y el exponente son legibles y
  el valor es $4{,}52 \times 10^{-1}$, en línea con el
  $4{,}51 \times 10^{-1}$ de la Tabla A.5 y el $4{,}52 \times 10^{-1}$ de la
  Tabla I.1 en la misma banda. La fila está en el rango informativo por
  encima de 10 kHz.
- **Evidencia:** página 40 del PDF (p. 30 impresa) de ISO 5136:2003.
- **Comportamiento de la biblioteca:** $4{,}52 \times 10^{-1}$ en
  `_TABLE_A6` de
  [`sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_duct.py).
- **Estado:** sin notificar (tipográfico, sin consecuencia numérica).

## ISO 5136:2003, Tabla I.2 (continuación), fila de 20 000 Hz (los exponentes de $a_8$ y $a_9$)

- **Ubicación:** Anexo I, Tabla I.2, «... for duct diameters 3,55 m $\le d
  \le$ 5 m», la página de continuación, fila de 20 000 Hz, columnas $a_8$ y
  $a_9$.
- **El impreso:** $a_8 = -5{,}88 \times 10^{-10}$ y
  $a_9 = 2{,}25 \times 10^{-10}$.
- **El problema:** a $U$ = 40 m/s el $a_9$ impreso aporta por sí solo
  $2{,}25 \times 10^{-10} \times 40^9 \approx 5{,}9 \times 10^4$ dB a
  $C_{3,4}$, cosa que ninguna corrección puede ser. La misma fila de las
  tablas vecinas imprime $a_8 = -5{,}90 \times 10^{-12}$ y
  $a_9 = 2{,}25 \times 10^{-13}$ (Tabla I.1) y
  $a_9 = 2{,}25 \times 10^{-13}$ (Tabla I.3), así que los exponentes son
  $-12$ y $-13$ y al impreso le faltan dos y tres décadas. El Anexo I es
  informativo y la fila está en el rango informativo por encima de 10 kHz.
- **Evidencia:** página 72 del PDF (p. 62 impresa) de ISO 5136:2003, frente
  a la misma fila de la Tabla I.1 en la página 70 del PDF (p. 60 impresa) y
  de la Tabla I.3 en la página 74 del PDF (p. 64 impresa).
- **Comportamiento de la biblioteca:** los Anexos informativos H e I quedan
  fuera del alcance que la norma se fija a sí misma (de 0,15 m a 2 m) y no se
  implementan; un conducto de más de 2 m se rechaza. Se registra para que
  una implementación del Anexo I no arrastre los exponentes tal como están
  impresos.
- **Estado:** sin notificar.

## ISO 4869-2:2018, Tabla C.1 (la reimpresión que contradice a la tabla que reimprime)

- **Ubicación:** anexo C (informativo), Tabla C.1, «A-weighted octave-band
  sound pressure levels, $L_{p,\mathrm{A}f(k)i}$, **from Table 2**», página 17
  del PDF (p. 11 impresa), frente a la Tabla 2 normativa que cita, página 11
  del PDF (p. 5 impresa).
- **El impreso:** las dos tablas traen los mismos ocho ruidos de referencia
  sobre las mismas siete bandas de octava, y siete de las ocho filas coinciden
  dígito a dígito. La sexta dice 82,0 / **89,3** / **93,3** / 95,6 / 93,0 /
  90,1 / 83,0 en la Tabla 2 y 82,0 / **89,4** / **93,5** / 95,6 / 93,0 / 90,1 /
  83,0 en la Tabla C.1. Difieren las celdas de 250 Hz y 500 Hz; ninguna más.
- **El problema:** la Tabla C.1 declara en su propio encabezado que procede de
  la Tabla 2, así que una de las dos está mal, y los resultados del propio
  anexo dicen cuál. La Fórmula (15) aplicada a los dieciséis valores de
  atenuación de la Tabla A.1 con la fila de la Tabla 2 reproduce exactos los
  dieciséis $PNR_{j6}$ de la Tabla C.2; con la fila de la Tabla C.1, trece de
  los dieciséis se quedan 0,1 dB cortos. La Tabla 2 es, por tanto, la lectura
  con la que se calculó el ejemplo, y además es la normativa, siendo la
  Tabla C.1 una reimpresión informativa. Quien tome los espectros de
  referencia del anexo C, donde están junto al ejemplo trabajado, obtiene los
  valores $H$ y $M$ de un protector una décima de decibelio por debajo.
- **Evidencia:** Fórmula (15), página 11 del PDF (p. 5 impresa), evaluada
  sobre la Tabla A.1, página 15 del PDF (p. 9 impresa), contra la sexta fila
  de la Tabla C.2, página 18 del PDF (p. 12 impresa), todo ello de la
  ISO 4869-2:2018.
- **Comportamiento de la biblioteca:** `HML_REFERENCE_NOISES` lleva la
  Tabla 2. La suite de tests calcula esa misma fila con los valores de la
  Tabla C.1 y comprueba que falla trece de los dieciséis impresos, de modo que
  las dos lecturas no pueden intercambiarse en silencio.
- **Estado:** sin notificar.

## VDI 2081 Blatt 1:2001-07, apartado 6.4 (la columna inglesa dice lo contrario que la alemana)

- **Ubicación:** folio impreso 40 (página 40 del PDF), apartado 6.4
  "Verzweigungen" / "Junctions", la frase inmediatamente posterior a la
  ecuación (35).
- **Lo impreso:** la columna alemana dice "Diese in Bild 27 dargestellte Senkung
  des Schallleistungspegels ist **frequenzunabhängig**". La columna inglesa de
  la misma página, traduciendo esa misma frase, dice "This sound power level
  reduction shown in Figure 27 **depends on the frequency**".
- **El problema:** las dos dicen cosas opuestas, y la que manda es la alemana:
  la portada de toda directriz VDI declara que la versión alemana es la
  vinculante y que no se garantiza la traducción inglesa. La alemana es además
  la que concuerda con el resto del documento. La figura 27 de esa misma página
  representa $\Delta L_W$ frente al cociente de secciones
  $S_1 / \sum S_{1,2,3}$ y no tiene eje de frecuencia; la propia ecuación (35),
  $\Delta L_W = |10 \lg (S_1 / \sum_i S_i)|$, no contiene la frecuencia; y el
  ejemplo resuelto de VDI 2081 Blatt 2:2005-05 imprime la reducción de nivel de
  una ramificación como un único número y no como espectro de octava, en las
  tres que tiene (tabla 1, elementos 3, 7 y 16, folios impresos 13 y 15:
  $5{,}6$, $4{,}8$ y $3{,}0$ dB).
- **Mecanismo probable:** el prefijo negativo de "frequenzunabhängig" no está en
  la traducción, lo que convierte "independiente de la frecuencia" en su
  contrario. El resto de la frase no difiere.
- **Consecuencia:** quien trabaje sólo con la columna inglesa buscará una
  dependencia con la frecuencia que ni la ecuación ni la figura tienen, y puede
  concluir que la directriz está incompleta en vez de que la frase está mal
  traducida.
- **Evidencia:** las dos columnas de la misma página impresa leídas una contra
  otra; la figura 27 de esa página; la ecuación (35) que la precede; y las tres
  filas de ramificación del ejemplo resuelto del Blatt 2. Verificado en la
  página 40 del PDF (p. impresa 40) de VDI 2081 Blatt 1:2001-07 y en las páginas
  13 y 15 del PDF (pp. impresas 13 y 15) de VDI 2081 Blatt 2:2005-05.
- **Comportamiento de la biblioteca:**
  [`split_loss`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) con
  `model="vdi2081"` devuelve un solo valor para la ramificación, que es la
  lectura alemana, y reproduce las tres ramificaciones impresas del ejemplo.
- **Estado:** sin comunicar. Los dos impresos están sustituidos (Blatt 1:2022-04
  y Blatt 2:2022-10) y no se dispone de ninguno de los sucesores, así que aquí
  no consta si la traducción se corrigió.

## VDI 2081 Blatt 2:2005-05, tabla 1, elemento 2 (el diámetro hidráulico que imprime no es con el que calcula)

- **Ubicación:** tabla 1, folio impreso 12 (página 12 del PDF), elemento 2, el
  silenciador de bafles: las filas "Hydr. Durchmesser $d_\mathrm{h}$ (m)" y
  "Strouhalzahl $St$".
- **Lo impreso:** $d_\mathrm{h} = 0{,}171$ m, y los ocho números de Strouhal
  $0{,}9$, $1{,}7$, $3{,}4$, $6{,}8$, $13{,}5$, $27{,}0$, $54{,}0$ y $108{,}0$
  en las octavas de 63 Hz a 8 kHz, para una ranura libre $s = 0{,}100$ m, una
  altura de bafle $H = 0{,}600$ m y una velocidad de ranura $v = 14{,}81$ m/s.
- **El problema:** las dos filas se contradicen. El apartado 7.2.4.2 del
  Blatt 1 define $St = f_\mathrm{m} d_\mathrm{h} / v_\mathrm{i}$, así que el
  $d_\mathrm{h}$ impreso y el $St$ impreso se determinan mutuamente. Con los
  $0{,}171$ m impresos los ocho números serían $0{,}73$, $1{,}45$, $2{,}89$,
  $5{,}79$, $11{,}57$, $23{,}15$, $46{,}29$ y $92{,}59$: ninguno redondea sobre
  la fila impresa. Con $d_\mathrm{h} = 2s = 0{,}200$ m salen $0{,}851$,
  $1{,}688$, $3{,}376$, $6{,}752$, $13{,}504$, $27{,}009$, $54{,}018$ y
  $108{,}035$, que redondean sobre los ocho.

  Los dos valores son defendibles como diámetro hidráulico, y por eso esto es
  una incoherencia interna y no un número mal puesto: $4A/P$ de una ranura de
  $0{,}100$ m por $0{,}600$ m vale $0{,}171$ m, mientras que el límite de
  placas paralelas al que tiende una ranura larga y estrecha es
  $2s = 0{,}200$ m. La tabla imprime el primero y calcula con el segundo.
- **Consecuencia:** seguir el $d_\mathrm{h}$ impreso no reproduce ni la fila de
  Strouhal ni el espectro de ruido que va debajo. Con $2s$ el elemento entero
  sale hasta el último decimal impreso: $L_\mathrm{WA} = 52$ dB por la
  ecuación (49) y los ocho niveles de octava de $62{,}7$ a $35{,}6$ dB por las
  ecuaciones (46), (50) y (51), el peor de ellos a 0,046 dB de su celda.
- **Evidencia:** las dos filas del mismo elemento impreso leídas contra el
  apartado 7.2.4.2 del Blatt 1 (folio impreso 53); los dos diámetros candidatos
  evaluados en las ocho octavas; y el espectro de ruido recalculado con cada
  uno. Verificado en la página 12 del PDF (p. impresa 12) de VDI 2081
  Blatt 2:2005-05 y en la página 53 del PDF (p. impresa 53) de VDI 2081
  Blatt 1:2001-07.
- **Comportamiento de la biblioteca:**
  [`silencer_self_noise`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/hvac.py) con
  `model="vdi2081"` toma la ranura libre y usa $2s$, así que reproduce el
  ejemplo resuelto. El docstring dice cuál de los dos toma.
- **Estado:** sin comunicar. Los dos impresos están sustituidos y no se dispone
  de ninguno de los sucesores.

## VDI 2081 Blatt 2:2005-05, tabla 1, elemento 2 (una referencia cruzada al apartado equivocado)

- **Ubicación:** tabla 1, folio impreso 12 (página 12 del PDF), elemento 2, el
  recuadro que dice "Tabelle aus VDI 2081 Blatt 1/7.3.2" junto a los
  coeficientes $a_1$, $a_2$, $b_1$ y $b_2$.
- **Lo impreso:** los coeficientes $0{,}255$, $0{,}015$, $-2{,}82$ y $-2{,}91$
  se atribuyen al apartado 7.3.2 del Blatt 1.
- **El problema:** el apartado 7.3 de VDI 2081 Blatt 1:2001-07 es
  "Luftschalldämmung eines Bauteils", el aislamiento a ruido aéreo de un
  elemento constructivo, y no tiene tal tabla. Los coeficientes están impresos
  en el apartado **7.2.3.2**, "Kulissenschalldämpfer", en el folio impreso 52,
  cuya tabla da exactamente esos cuatro valores en la fila de 200 mm, que es el
  espesor de bafle del elemento.
- **Consecuencia:** quien siga la referencia aterriza en otro capítulo. Los
  valores en sí son correctos.
- **Evidencia:** el apartado citado y el real, leídos los dos de las páginas
  impresas. Verificado en la página 12 del PDF (p. impresa 12) de VDI 2081
  Blatt 2:2005-05 y en la página 52 del PDF (p. impresa 52) de VDI 2081
  Blatt 1:2001-07.
- **Comportamiento de la biblioteca:** ninguno; la biblioteca cita el apartado
  7.2.3.2.
- **Estado:** sin comunicar.

## ISO 11200:2014, anexo B (los dos casos calculan la misma desviación típica de dos maneras distintas)

- **Ubicación:** anexo B, tabla B.1 en el folio impreso 27 (página 33 del PDF) y
  tabla B.3 en el folio impreso 30 (página 36 del PDF). Las dos llevan una fila
  rotulada igual, «Standard deviation of the three values measured,
  $\sigma_{omc}$».
- **Lo impreso:** la tabla B.1 lista las tres lecturas 94,5 dB; 94,3 dB;
  93,8 dB y da $\sigma_{omc} = 0{,}3$ dB. La tabla B.3 lista 79,0 dB; 80,2 dB;
  82,9 dB y da $\sigma_{omc} = 2$ dB.
- **El problema:** usan estimadores distintos. La Ecuación (C.1), impresa
  idéntica en ISO 11201:2010, ISO 11202:2010 e ISO 11204:2010, es la desviación
  típica **muestral**,

  $$
  \sigma_\mathrm{omc} = \sqrt{\frac{1}{N-1}
  \sum_{j=1}^{N} \left( L'_{p,j} - \overline{L'_p} \right)^2}
  $$

  Con $1/(N-1)$ la primera terna da 0,3606 dB, que redondea a **0,4** y no a
  los 0,3 que imprime la tabla; la segunda da 1,9975 dB, que redondea al
  **2,0** que sí imprime. Con $1/N$ la primera da 0,2944 → **0,3**, el valor
  impreso, y la segunda 1,6310 → 1,6, que no está impreso. La tabla B.1 divide
  por tanto entre $N$ y la B.3 entre $N-1$, en el mismo anexo, bajo el mismo
  rótulo y para la misma magnitud.
- **Consecuencia:** no es cosmético, porque el valor se propaga. La tabla B.1
  sigue imprimiendo $\sigma_\mathrm{tot} = 1{,}5$ dB y $U = 2{,}4$ dB a partir
  de $\sigma_{R0} = 1{,}5$ dB. Con los 0,4 dB que da la Ecuación (C.1),
  $\sigma_\mathrm{tot} = \sqrt{1{,}5^2 + 0{,}4^2} = 1{,}552 \to 1{,}6$ dB y
  $U = 1{,}6 \times 1{,}552 = 2{,}48 \to 2{,}5$ dB, porque el factor de
  cobertura se aplica al total sin redondear y no al decibelio con que se
  informa. Quien reproduzca el ejemplo desde las ecuaciones no obtiene la
  incertidumbre que el ejemplo publica.
- **Mecanismo probable:** tres lecturas son la muestra más pequeña que la
  ecuación admite, y es justo donde los dos divisores más se separan:
  $\sqrt{3/2}$ es un 22 % de diferencia. La función de desviación típica
  poblacional de una hoja de cálculo toma $1/N$ por defecto, y con tres puntos
  el desliz basta para cambiar el decibelio redondeado.
- **Evidencia:** las dos tablas leídas en la página impresa, no en el texto
  extraído. Verificado en las páginas 33 y 36 del PDF (folios impresos 27 y 30)
  de ISO 11200:2014, contra la Ecuación (C.1) en la página 32 del PDF (folio
  impreso 26) de ISO 11201:2010.
- **Comportamiento de la biblioteca:**
  [`operating_standard_deviation`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/workstation.py)
  implementa la Ecuación (C.1) tal como se imprime, con $1/(N-1)$. Reproduce la
  tabla B.3 y deliberadamente no reproduce los 0,3 dB de la tabla B.1;
  `tests/emission/test_workstation.py` fija las dos mitades para que la
  elección no se mueva.
- **Estado:** sin comunicar.

## ISO 3382-1:2009, A.2.1 (el mismo símbolo nombra dos niveles distintos, con una página de por medio)

- **Ubicación:** anexo A (informativo), A.2.1. La lista «where» bajo las
  Ecuaciones (A.2) y (A.3) en el folio impreso 13 (página 21 del PDF), y la
  lista «where» bajo la Ecuación (A.5) en el folio impreso 14 (página 22 del
  PDF).
- **Lo impreso:** el folio 13 dice «$L_{pE}$ is the sound pressure exposure
  level of $p(t)$», siendo $p(t)$ «the instantaneous sound pressure of the
  impulse response measured at the measurement point», es decir, el receptor de
  la sala bajo ensayo. El folio 14, dentro de la NOTA 1, dice «$L_{pE}$ is the
  spatial-average sound pressure exposure level measured in the reverberation
  room».
- **El problema:** un símbolo, dos magnitudes, el mismo apartado, sin
  subíndice que las distinga y sin nota que avise de la reutilización. La
  segunda es una calibración de la fuente en laboratorio; la primera es la
  medición para la que existe todo el anexo.
- **Consecuencia:** sustituir la (A.5) en la (A.1) tal como están impresos los
  símbolos da

  $$
  G = L_{pE} - L_{pE,10} = L_{pE} - \left[ L_{pE} + 10 \lg (A/S_0) - 37 \right]
    = 37 - 10 \lg (A/S_0)\ \text{dB},
  $$

  donde la sala ha desaparecido y la fuerza sonora depende sólo del área de
  absorción de la cámara reverberante en que se calibró la fuente. La
  sustitución es la que invitan los símbolos impresos, y no tiene sentido.
- **Evidencia:** verificado en las páginas 21 y 22 del PDF (folios impresos 13
  y 14) de BS EN ISO 3382-1:2009.
- **Comportamiento de la biblioteca:**
  [`reverberation_room_reference_level`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py)
  llama a su argumento `reverberation_room_level`, y el nivel de la sala nunca
  llega hasta él: lo mide
  [`sound_strength`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) a partir de la
  respuesta que se pasa como `ir`. Nada impide que quien llama escriba la
  sustitución a mano, pero ninguna variable hace los dos papeles, y los dos
  nombres dicen cuál es cuál.
- **Estado:** sin comunicar.

## ISO 3382-1:2009, A.2.1 (un barrido de directividad «cada 12,5 grados» que no cierra la circunferencia)

- **Ubicación:** anexo A (informativo), A.2.1, la nota inmediatamente bajo la
  Ecuación (A.4), folio impreso 13 (página 21 del PDF).
- **Lo impreso:** «When making such a measurement in a free field, it is
  necessary to make the measurement at every 12,5° around the sound source and
  to calculate the energy-mean value of the sound pressure exposure levels in
  order to average the directivity of the sound source.»
- **El problema:** $360 / 12{,}5 = 28{,}8$. No hay número entero de pasos de
  12,5° que cierre una vuelta: 28 pasos llegan a 350° y dejan un hueco de 10°,
  29 se pasan hasta 362,5°. La instrucción no puede seguirse literalmente.
- **Consecuencia:** dos laboratorios que «midan cada 12,5°» pueden usar
  conjuntos de acimuts distintos y, para una fuente en el límite de
  directividad de la tabla 1 (±6 dB a 4 kHz), sus medias energéticas difieren.
  El nivel de referencia $L_{pE,10}$ al que llevan todas las rutas de A.2.1 no
  es, por tanto, reproducible sólo desde la instrucción impresa. El barrido de
  cualificación de la fuente de la propia norma, en 4.2.1, usa 5°, que divide
  360 exactamente en 72.
- **Evidencia:** verificado en la página 21 del PDF (folio impreso 13) de
  BS EN ISO 3382-1:2009.
- **Comportamiento de la biblioteca:**
  [`directivity_energy_average`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) toma la
  lectura que la nota sí sostiene: un muestreo uniforme de la vuelta completa
  no más grueso que el paso impreso, es decir al menos
  $\lceil 360 / 12{,}5 \rceil = 29$ acimuts, combinados con la media energética
  que la nota pide. Menos acimuts levantan `ValueError` en lugar de promediar
  una vuelta que nunca se cerró.
- **Estado:** sin comunicar.

## ISO 3382-1:2009, C.2.1 y C.2.2 (la prosa de los dos soportes de escenario se deja un límite de integración)

- **Ubicación:** anexo C (informativo), C.2.1 en el folio impreso 23 (página
  31 del PDF) y C.2.2 en el folio impreso 24 (página 32 del PDF), cada uno en
  la frase que presenta su propia ecuación.
- **Lo impreso:** C.2.1 define el soporte temprano como «the ratio, in
  decibels, of the reflected energy **within the first 0,1 s** relative to
  the direct sound», e imprime

  $$
  ST_\mathrm{Early} = 10 \lg \left[
      \frac{\int_{0,020}^{0,100} p^2(t)\ \mathrm{d}t}
           {\int_{0}^{0,010} p^2(t)\ \mathrm{d}t} \right]\ \mathrm{dB}.
  $$

  C.2.2 define el soporte tardío como «the ratio, in decibels, of the
  reflected energy **after the first 0,1 s** relative to the direct sound», e
  imprime

  $$
  ST_\mathrm{Late} = 10 \lg \left[
      \frac{\int_{0,100}^{1,000} p^2(t)\ \mathrm{d}t}
           {\int_{0}^{0,010} p^2(t)\ \mathrm{d}t} \right]\ \mathrm{dB}.
  $$

- **El problema:** ninguna de las dos frases describe la ecuación que tiene
  al lado. La Ecuación (C.1) empieza en 0,020 s, no en los 0,010 s donde
  termina la ventana del sonido directo, así que el intervalo entre ambos no
  cuenta ni en el numerador ni en el denominador y la prosa no menciona el
  hueco. La Ecuación (C.2) se detiene en 1,000 s, donde la prosa no pone
  ningún límite superior.
- **Consecuencia:** las dos mueven un número, y la primera lo mueve más. En
  un decaimiento exponencial de $T = 2$ s, quien entienda «within the first
  0,1 s» como que empieza donde acaba la ventana del sonido directo recoge
  además el intervalo de 10 ms a 20 ms, que es un 17 % más de energía y
  **0,68 dB** en $ST_\mathrm{Early}$, frente a la desviación típica de 1 dB
  que C.2.4 estima para una sola lectura. El techo que le falta a la (C.2)
  cuesta 0,01 dB en esa misma sala, porque un decaimiento de 2 s ya ha caído
  30 dB al segundo, y llega a 0,2 dB con $T = 4$ s y a 1,0 dB con $T = 8$ s:
  es la catedral, y no la sala de conciertos, lo que separa esa segunda
  omisión.
- **Evidencia:** verificado en las páginas 31 y 32 del PDF (folios impresos
  23 y 24) de BS EN ISO 3382-1:2009.
- **Comportamiento de la biblioteca:**
  [`stage_support`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) integra los límites
  impresos, que son los de
  [`EARLY_SUPPORT_WINDOW_S`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) y
  `LATE_SUPPORT_WINDOW_S`. `tests/room/test_auditorium_stage.py` deja caer una
  llegada en el hueco y otra pasado el techo y exige que ninguna cambie nada.
- **Estado:** sin comunicar.

## ISO 3382-1:2009, Tabla 1 y A.4 (los mismos límites son máximos en un apartado y mínimos en el otro)

- **Ubicación:** el título de la Tabla 1 y el párrafo de 4.2.1 que la
  precede, folio impreso 3 (página 11 del PDF), frente al cuarto párrafo de
  A.4, folio impreso 19 (página 27 del PDF).
- **Lo impreso:** 4.2.1 dice «Table 1 lists the **maximum** acceptable
  deviations from omnidirectionality when averaged over "gliding" 30° arcs in
  a free sound field», y el título de la propia tabla reza «Table 1 —
  **Maximum** deviation of directivity of source in decibels for excitation
  with octave bands of pink noise and measured in free field». A.4 dice «If
  the source directivity is close to the **minimum** limits given in Table 1,
  the measurement should be repeated with the source turned in at least three
  steps totally».
- **El problema:** una tabla, dos palabras opuestas para lo que son sus
  números. Los valores son techos, como dicen su propio título y 4.2.1, y A.4
  los llama suelos.
- **Consecuencia:** la frase de A.4 es la que le dice a un laboratorio cuándo
  hacer trabajo de más, y leída tal cual dice lo contrario de lo que quiere
  decir. Una fuente «close to the minimum limits» sería una casi perfecta, que
  es justo el caso que no necesita repetición ninguna; lo que pide A.4 es
  repetir el barrido de una fuente que apenas pasa el techo, porque ahí es
  donde la orientación de la fuente empieza a importarle a la respuesta. Quien
  tome la palabra al pie de la letra repite la medición con las fuentes
  equivocadas y se la salta con las que la necesitan.
- **Evidencia:** verificado en las páginas 11 y 27 del PDF (folios impresos 3
  y 19) de BS EN ISO 3382-1:2009.
- **Comportamiento de la biblioteca:**
  [`MAX_SOURCE_DIRECTIVITY_DEVIATION_DB`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py)
  y [`source_directivity_limit`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) los
  llevan como los máximos que su propio título hace de ellos. La repetición
  con tres orientaciones de A.4 es un procedimiento, no un cálculo, y la
  biblioteca no lo implementa.
- **Estado:** sin comunicar.

## ISO 3382-1:2009, 4.2.1 (un promedio deslizante cuya ventana no tiene fase declarada)

- **Ubicación:** 4.2.1, el párrafo inmediatamente anterior a la Tabla 1, folio
  impreso 3 (página 11 del PDF).
- **Lo impreso:** «Table 1 lists the maximum acceptable deviations from
  omnidirectionality when averaged over "gliding" 30° arcs in a free sound
  field. In case a turntable cannot be used, measurements per 5° should be
  performed, followed by "gliding" averages, each covering six neighbouring
  points.»
- **El problema:** seis puntos de 5° cubren 30° de arco leídos como seis
  sectores, y 25° leídos como la distancia entre el primero y el último, así
  que las dos frases sólo concuerdan con la lectura por sectores. Y, sobre
  todo, nada dice dónde se sitúan esos seis puntos respecto del arco que
  promedian: la ventana puede adelantarse a su acimut, retrasarse o quedar
  centrada, y el apartado no elige. Tampoco dice cómo se combinan los seis,
  aunque la referencia con la que se comparan sea explícitamente «a 360°
  energetic average».
- **Consecuencia:** en una vuelta completa las ventanas de seis puntos son un
  mismo conjunto cíclico se ancle la ventana por donde se ancle, así que la
  fase desplaza hasta media ventana, 15° del patrón, la orientación con la
  que se reporta cada desviación, y deja las desviaciones intactas. En una
  fuente cercana a su límite de la Tabla 1 eso sigue decidiendo si la
  desviación máxima se reporta sobre un lóbulo o entre dos, que es la
  orientación que la A.4 pide luego girar y volver a medir. Los otros dos
  silencios sí mueven el número: la lectura del span y la ley de combinación
  cambian lo que promedia cada arco, así que dos laboratorios que sigan los
  dos el apartado pueden dar desviaciones máximas distintas para una misma
  fuente, y la norma no da forma de saber cuál de los dos la leyó bien.
- **Evidencia:** verificado en la página 11 del PDF (folio impreso 3) de
  BS EN ISO 3382-1:2009.
- **Comportamiento de la biblioteca:**
  [`gliding_directivity_deviation`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/room/auditorium.py) toma
  la lectura por sectores, promedia los arcos energéticamente igual que la
  referencia, y empieza cada ventana en el acimut contra el que se reporta,
  dando la vuelta al círculo. Su docstring dice que las tres son elecciones.
- **Estado:** sin comunicar.

## IEC 60534-8-3:2010, anexo A (el factor geométrico de tubería se imprime redondeado, y el anexo no calculó con el valor redondeado)

- **Ubicación:** anexo A (informativo), A.2, el bloque «Given data» del folio
  impreso 32 (página 34 del PDF) de BS EN 60534-8-3:2011, frente a la fila de
  la Ecuación (2) de la Tabla A.1 en ese mismo folio.
- **Lo impreso:** los datos de partida dicen «Piping geometry factor:
  $F_\mathrm{p} = 0{,}98$», bajo el encabezado «The following values are used
  in, or determined from, calculations based on IEC 60534-2-1». La Tabla A.1
  imprime luego $p_{vc} = 567\,787$ Pa para el ejemplo 1, y cinco valores más
  para las demás columnas, a partir de
  $p_{vc} = p_1\left[1 - x/(F_{LP}/F_P)^2\right]$ con $F_{LP} = 0{,}792$.
- **El problema:** las dos cosas no pueden ser ciertas a la vez. Despejar
  $(F_{LP}/F_P)^2$ de la Ecuación (2) en cada pareja impresa da 0,647 829,
  0,647 827, 0,647 821, 0,647 829 y 0,647 833 en las cinco columnas que
  imprimen valor, que es $F_p = 0{,}984$ con cuatro cifras en todas ellas. Con
  el 0,98 impreso sale 0,653 128 y $p_{vc} = 571\,294$ Pa, a 3 507 Pa de la
  cifra impresa. El valor es calculado, no un dato: el propio anexo dice que
  viene de la IEC 60534-2-1, y con el coeficiente de pérdida de carga que
  imprime, $\Sigma\zeta = 0{,}86$, sale $F_p = 0{,}984$ para el caso DN 100.
  Es decir, el anexo calculó con tres decimales e imprimió dos.
- **Consecuencia:** se mueve todo lo que viene después. Con el 0,98 impreso
  las cuatro fronteras de régimen salen $x_C = 0{,}287$, $\alpha = 0{,}786$ y
  $x_B = 0{,}578$ frente a los 0,285, 0,784 y 0,576 impresos, y la potencia
  acústica del ejemplo 1 sale 21,9 W frente a los 22,3 W impresos. Nada queda
  muy lejos, y nada reproduce: quien contraste su implementación con el anexo
  A usando el número que el anexo A imprime no cuadra ni una fila.
- **Evidencia:** los datos de partida y los seis valores de $p_{vc}$ leídos de
  la página impresa. Verificado en las páginas 33 y 34 del PDF (pp. impresas 31
  y 32) de BS EN 60534-8-3:2011.
- **Comportamiento de la biblioteca:**
  [`valve_aerodynamic_noise`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/valves.py) toma
  el cociente como argumento y no guarda ningún valor propio; las filas de
  conformidad y `tests/noise_control/test_valves.py` pasan $0{,}792/0{,}984$ y
  dicen por qué en el fixture.
- **Estado:** sin comunicar.

## IEC 60534-8-3:2010, Tabla A.1 (un diámetro de orificio equivalente diez veces menor, desmentido por la fila de debajo)

- **Ubicación:** anexo A (informativo), Tabla A.1, la fila de la Ecuación (8c)
  del folio impreso 33 (página 35 del PDF), frente a la fila de la Ecuación
  (8a) impresa justo debajo.
- **Lo impreso:** las seis columnas de la fila (8c) dicen $d_0 = 0.010$ m. Los
  datos de partida del folio 31 dan $N_\mathrm{O} = 6$ aberturas de jaula y
  $A = 0{,}00137$ m² para una de ellas, y la Ecuación (8c) es
  $d_o = \sqrt{4 N_o A/\pi}$.
- **El problema:** $\sqrt{4 \times 6 \times 0{,}00137/\pi} = 0{,}102$ m, no
  0,010 m. Los dos numerales son las mismas tres cifras en otro orden. La fila
  de debajo resuelve cuál es: la Ecuación (8a) es $F_d = d_H/d_o$, la fila
  (8b) imprime $d_H = 0{,}030$ m y la fila (8a) imprime $F_d = 0{,}30$ en las
  seis columnas. 0,030/0,102 es 0,30; 0,030/0,010 es 3,0.
- **Consecuencia:** quien tome el $d_o$ impreso obtiene un modificador de
  estilo de válvula de 3,0, un diámetro de chorro diez veces mayor por la
  Ecuación (9) y una frecuencia de pico diez veces menor, lo que desplaza el
  espectro interno de la Ecuación (19) más de tres octavas. El resto de la
  tabla está calculado con 0,102 m, así que el error se queda en esa celda.
- **Evidencia:** las filas (8b), (8c) y (8a) leídas de la página impresa.
  Verificado en las páginas 33 y 35 del PDF (pp. impresas 31 y 33) de
  BS EN 60534-8-3:2011.
- **Comportamiento de la biblioteca:**
  [`valve_style_modifier`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/valves.py)
  implementa (8b) y (8c) tal como están impresas y devuelve 0,296 para la
  jaula del anexo, que redondea al $F_d$ impreso; el test que lleva el nombre
  de esta entrada fija las dos lecturas para que el $d_o$ impreso no vuelva.
- **Estado:** sin comunicar.

## IEC 60534-8-3:2010, Tabla A.2 (dos factores de frecuencia con el exponente equivocado en una potencia de diez)

- **Ubicación:** anexo A (informativo), A.3, la columna $G_x$ de la Tabla A.2
  del folio impreso 43 (página 45 del PDF), bandas 5 y 10 de 33.
- **Lo impreso:** la columna va $G_{x,4} = 5.6 \times 10^{-9}$,
  $G_{x,5} = 1.4 \times 10^{-9}$, $G_{x,6} = 3.6 \times 10^{-8}$, y más abajo
  $G_{x,9} = 5.8 \times 10^{-7}$, $G_{x,10} = 1.4 \times 10^{-7}$,
  $G_{x,11} = 3.5 \times 10^{-6}$.
- **El problema:** por debajo de la frecuencia de coincidencia interna la
  Tabla 6 hace $G_x$ proporcional a $f_i^4$, así que la columna tiene que
  crecer de forma monótona, y lo hace en todas las bandas menos en esas dos,
  donde baja. Recalcular la Tabla 6 para esta tubería da
  $1{,}4 \times 10^{-8}$ en la banda 5 y $1{,}4 \times 10^{-6}$ en la 10: la
  mantisa está bien en las dos y el exponente es una unidad menor.
- **Consecuencia:** ninguna para el resto del anexo, y eso es lo que lo
  resuelve. Las pérdidas por transmisión impresas dos filas más abajo,
  $TL_5 = -86{,}1$ dB y $TL_{10} = -76{,}2$ dB, son las que da la Ecuación
  (20a) con los factores corregidos; con los impresos saldrían $-96{,}1$ dB y
  $-86{,}3$ dB. La Tabla A.2 calculó con los buenos e imprimió los malos, y
  quien monte un oráculo sólo con la columna $G_x$ hereda un error de 10 dB en
  dos bandas.
- **Evidencia:** la columna $G_x$ leída de la página impresa. Verificado en las
  páginas 45 y 46 del PDF (pp. impresas 43 y 44) de BS EN 60534-8-3:2011; las
  24 pérdidas por transmisión impresas en la segunda son las que la biblioteca
  reproduce con menos de 0,07 dB de diferencia.
- **Comportamiento de la biblioteca:**
  [`pipe_transmission_loss`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/noise_control/valves.py)
  calcula $G_x$ a partir de la Tabla 6, y la fila de conformidad «Pipe
  transmission loss, example 7, 24 bands» reproduce todas las pérdidas
  impresas, cosa que los $G_x$ impresos no permitirían.
- **Estado:** sin comunicar.

## Propiedades de las fuentes, relacionadas, que no son erratas

Registradas aquí para prevenir futuros «arreglos» que romperían la
concordancia con las fuentes publicadas:

- **ISO 12354-1:2017 Tabla L.8 / ISO 12354-2:2017 Tabla G.8, primera fila:**
  la fila etiquetada «Int. wall 1/2 – Ext. wall 1/2» imprime
  $m'_i = 219{,}0\ \text{kg/m}^2$ y $m'_{\perp i}$ (Parte 2:
  $m'_\text{orthogonal}$) $= 360{,}0\ \text{kg/m}^2$, que es la asignación de
  una trayectoria *que sale de la pared exterior*, la dirección opuesta a la
  que da la propia etiqueta de la fila. Leída en la dirección de la fila, el
  elemento que lleva la trayectoria es la pared interior, así que $m'_i$
  debería ser 360,0 y la masa perpendicular 219,0. Es un desliz de etiquetado
  y nada más: la rama es la rama de **esquina** de la T rígida
  $K_{12} = 5{,}7 + 5{,}7 M^2$, donde solo entra $M^2$, así que ambas
  asignaciones devuelven los mismos 5,965 → 6,0 dB. La segunda fila de cada
  tabla, «Ext. wall 1/2 – Ext. wall 1/2», es la rama pasante
  $5{,}7 + 14{,}1 M + 5{,}7 M^2$, donde el signo de $M$ sí importa, y está
  etiquetada y poblada de forma consistente ($M = \log_{10}(360/219)$ da
  9,006 → los 9,0 impresos). Verificado en la página 89 del PDF (p. 83
  impresa) de ISO 12354-1:2017 y la página 46 del PDF (p. 40 impresa) de
  ISO 12354-2:2017. No se registra como errata porque ningún número depende
  de ello; se registra aquí para que un lector futuro no «corrija» el
  convenio por trayectoria de la biblioteca para casar con la fila impresa.
- **Término de agua pura de Francois-Garrison:** las dos cúbicas de $A_3$
  publicadas no se encuentran exactamente en el cambio de 20 °C (un escalón
  de $1 \cdot 10^{-7} f^2\ \text{dB/km}$, 0.1 dB/km a 1 MHz). Inherente a los
  coeficientes publicados.
- **Simplificación de Ainslie-McColm:** la afirmación del artículo de estar
  «within 10 % of Francois-Garrison» se excede marginalmente en las esquinas
  extremas de su dominio declarado (10.4 % a −6 °C / 1 MHz; 12.3 % a 7 km de
  profundidad). Una propiedad del ajuste publicado; ambas transcripciones
  verificadas dígito a dígito.
- **CNOSSOS-EU Anexo II 2.3, número de ecuación que falta:** la sección
  ferroviaria numera sus fórmulas (2.3.1), (2.3.2), (2.3.4), (2.3.5)..., sin
  ninguna (2.3.3) en todo el Anexo II. Verificado en la página 17 del PDF
  (p. L 168/17 impresa) de la Directiva (UE) 2015/996:2015, donde (2.3.2) y
  (2.3.4) están una encima de la otra. No falta nada del método; solo salta
  la numeración.
- **Corrección de errores de CNOSSOS-EU de 2018, códigos de columna de la
  Tabla G-3:** se reporta que la corrección de errores encabeza las siete
  columnas de $L_{r,TR}$ «B/S B/M B/H B/S B/M B/H B/H», donde las tres
  primeras deberían leer «M/S M/M M/H» y la última «W», y la Directiva
  Delegada (UE) 2021/1226 de la Comisión, punto (20)(c) del Anexo, sí
  sustituye ese encabezado por los códigos corregidos más una columna D
  nueva. Se deja sin registrar porque la propia corrección de errores se
  publica solo como HTML en EUR-Lex, así que no pudo obtenerse aquí ninguna
  página impresa suya, y este registro no recoge una afirmación sobre un
  símbolo impreso que no se haya leído de la página. El impreso de 2015 de la
  misma tabla, que sí se leyó, lleva encabezados descriptivos («Mono-block
  sleeper on soft rail pad» y demás) y ningún defecto.
- **Long, Architectural Acoustics 2e, capítulo 17, nivel en la mesa
  adyacente:** el ejemplo del restaurante declara que «at an adjacent table
  3 m (10 ft) away, the direct field level from our conversation is about
  54 dB», donde su propia Ec. (17.50) con los $Q = 2$ y
  $L_W = 70\ \text{dB}$ que producen sus 60 dB a 1.2 m da 52.5 dB. Se deja
  sin registrar porque la lectura pretendida no puede establecerse desde el
  libro: 54 dB es también lo que la misma ecuación da a 2.5 m (54.1 dB, y
  2.5 m es la separación de mesas que el párrafo siguiente deriva), y lo que
  daría una sola duplicación de distancia de 6 dB desde los 60 dB
  redondeados, mientras que el «3 m (10 ft)» impreso es autoconsistente en
  ambas unidades y se repite en el párrafo anterior. `speech_direct_level`
  evalúa la Ec. (17.50) tal como está impresa, así que devuelve 52.5 dB allí;
  no «corregirla» hacia 54 dB.
- **Constante EPNL del Anexo 16 de la OACI:** la constante redondeada 13 del
  Anexo para registros uniformes de 0.5 s difiere de la forma exacta
  $-10\log_{10}(T_0)$ en 0.0103 dB; la biblioteca usa la forma exacta, que la
  referencia integrada del ETM reproduce a cinco decimales.
- **Filas de elementos de la Tabla 14.9 de Long:** la hoja resuelta de ruido
  por conductos del capítulo 14 la produjo un programa comercial, como
  declara el texto que la introduce, y varias de sus filas de elementos no se
  siguen de las tablas impresas a su lado: la fila del ventilador
  (90/86/82/79/77/75/71/61 dB) no es lo que da la Ec. 13.1 con las constantes
  de curvatura hacia delante de la Tabla 13.5 en ese punto de trabajo
  (99/99/89/84/82/77/72/67 dB, y tampoco un desplazamiento de nivel de ello),
  y la fila del conducto flexible (14/14/16/15/17/22/16/13 dB) no es la
  entrada de la Tabla 14.4 para 12 in por 6 ft (3/5/10/15/17/16/9 dB). La
  biblioteca implementa las ecuaciones y tablas impresas, y usa la hoja solo
  para lo que fija de verdad, la aritmética de la cascada; sus filas de
  elementos se introducen tal como están publicadas en
  [`tests/noise_control/test_duct_path.py`](https://github.com/jmrplens/phonometry/blob/main/tests/noise_control/test_duct_path.py).
  El propio redondeo de la hoja tampoco es siempre autoconsistente (la fila 3
  de impulsión imprime un *Sum* de 49 dB a 500 Hz donde $76 - 28 = 48$, y
  después un *Combined* consistente con 48), que es por lo que la comparación
  corre al 1 dB que la hoja impresa lleva.
- **ISO 3747:2010 Tabla E.1, las etiquetas de grado de precisión:** la tabla
  informativa de ejemplos de $\sigma_\mathrm{tot}$ etiqueta sus tres filas
  «0,5 (accuracy grade 1)», «1,5 (accuracy grade 2)» y «3 (accuracy
  grade 3)», mientras que la Tabla 2, normativa, de esta parte da
  $\sigma_{R0}$ = 4,0 dB para el grado 3 de control y el campo de aplicación
  de ISO 3747 cubre solo los grados 2 y 3. Es la ilustración compartida de la
  familia ISO 3740, no una afirmación sobre este método: ISO 3744:2010 imprime
  en su Tabla H.1 la tabla idéntica, con las mismas filas, etiquetas y celdas
  de $\sigma_\mathrm{tot}$, e ISO 3744 cubre solo el grado 2. Verificado en la
  página 42 del PDF (p. 33 impresa) y en la página 27 del PDF (p. 18 impresa)
  de BS EN ISO 3747:2010. La biblioteca lee $\sigma_{R0}$ de la Tabla 2
  normativa (1,5 dB y 4,0 dB, comprobación de conformidad «ISO 3747:2010
  Table 2 / Eq. 22») y usa la Tabla E.1 solo por su fila
  $\sigma_\mathrm{tot}$ = 1,6 / 2,5 / 4,3 frente a $\sigma_{R0}$ = 1,5 dB,
  donde ambas tablas coinciden. No «corregir» la fila de 3 dB a 4,0 dB:
  pertenece a la ilustración de la familia, no a la Tabla 2 de esta parte.
- **ISO 3747:2010 Anexo C, $\theta_\mathrm{ref}$ = 296 K:** el anexo imprime
  la temperatura de referencia de la corrección de impedancia de radiación
  como 296 K junto a una condición de referencia de 23,0 °C, que son
  296,15 K, así que exactamente en las condiciones de referencia
  $C_2 = 15 \lg(296{,}15/296) = +0{,}003\,3$ dB y no cero. El apartado 9.1.4
  de ISO 3741:2010 e ISO 3744:2010 imprimen el mismo $\theta_1$ = 296 K, así
  que es el redondeo de la familia y no una errata de una parte; la
  biblioteca conserva los 296 K en el `C2` compartido de
  [`sound_power_reverberation.py`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_reverberation.py)
  y fija el residuo (comprobación de conformidad «ISO 3747:2010 Annex C»).
  No «corregirlo» a 296,15 K.
- **ISO 3747:2010 Ec. (14), el margen de ruido de fondo del suceso aislado:**
  $\Delta L_{Ei} = L'_{Ei,q(\mathrm{ST})} - L_{pi(\mathrm{B})}$ resta un nivel
  de ruido de fondo promediado en el tiempo de un nivel de suceso aislado
  integrado en el tiempo, pidiendo solo que ambos se midan con el mismo
  tiempo de integración $T$. La diferencia es un margen verdadero para
  $T$ = 1 s; para un $T$ mayor el ruido de fondo contiene $10 \lg(T/T_0)$ dB
  más energía sobre el intervalo del suceso (apartado 3.4, NOTA 1). La
  Ec. (25) de ISO 3741:2010 y el apartado 8.3.4 de ISO 3744:2010 imprimen la
  misma línea, verificado en la página 23 del PDF (p. 14 impresa) de BS EN
  ISO 3747:2010 y en las páginas correspondientes de las dos normas hermanas,
  así que es la convención de la familia y no se registra contra una parte.
  La biblioteca aplica la Ec. (14) tal como está impresa por defecto y ofrece
  `integration_time` en
  [`sound_energy_in_situ`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_situ.py)
  para llevar antes el ruido de fondo al intervalo del suceso.

- **ISO 5136:2003, apartado 5.3.4.3, el signo de la Ecuación (8):** el
  apartado dice que las correcciones del cono aerodinámico y la bola de
  espuma «are estimated to be negative and of small magnitude», y a
  continuación imprime $C_{3,4} = 10 \lg[1/(1 - U/c)^2]$ dB, que es positiva
  siempre que $U > 0$: a los 20 m/s que se permiten al cono aerodinámico, con
  $c$ = 340 m/s, $+0{,}53$ dB en el lado de impulsión y $-0{,}50$ dB en el de
  aspiración. El signo de la ecuación es el que da la onda plana convectada:
  el flujo de energía de una onda que viaja con el flujo es $(1 + M)^2$ veces
  $p^2/\rho c$, de modo que para una presión dada la potencia es mayor aguas
  abajo y menor aguas arriba. No se registra como errata porque la frase con
  la que se cierra ese mismo párrafo reconcilia las dos: «With this
  simplification, the sound power level obtained by using the nose cone or
  foam ball is expected to be higher than the true sound power level.» La
  corrección negativa es la modal, que no está disponible y se descarta; la
  Ecuación (8) es la parte convectiva que se conserva, y la norma dice en la
  misma frase que lo que queda sesga el $L_W$ hacia arriba. Leído en la
  página 29 del PDF (p. 19 impresa) de ISO 5136:2003. Se registra aquí para
  que nadie «corrija» el signo de la Ecuación (8), que
  [`flow_modal_correction`](https://github.com/jmrplens/phonometry/blob/main/src/phonometry/emission/sound_power_in_duct.py)
  implementa tal como está impresa y que fija
  `test_eq8_omnidirectional_shields` en
  [`tests/emission/test_sound_power_in_duct.py`](https://github.com/jmrplens/phonometry/blob/main/tests/emission/test_sound_power_in_duct.py).

<!-- END GENERATED BODY -->
