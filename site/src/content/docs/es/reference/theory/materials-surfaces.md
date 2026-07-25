---
title: "Materiales y superficies"
description: "Dispersión superficial y difusión, absorción in situ de pavimentos y caracterización de materiales acústicos detrás de phonometry."
references:
  - type: book
    authors: ["Cox, T. J.", "D'Antonio, P."]
    year: 2017
    title: "Acoustic absorbers and diffusers: Theory, design and application"
    edition: "3.ª ed."
    publisher: "CRC Press"
    doi: "10.1201/9781315369211"
    note: "ISBN 978-1-4987-4099-9. La medida y el diseño de absorbentes y difusores, de los autores del método del coeficiente de difusión de ISO 17497-2."
  - type: book
    authors: ["Allard, J. F.", "Atalla, N."]
    year: 2009
    title: "Propagation of sound in porous media: Modelling sound absorbing materials"
    edition: "2.ª ed."
    publisher: "Wiley"
    doi: "10.1002/9780470747339"
    note: "ISBN 978-0-470-74661-5. La teoría del material poroso que enlaza las magnitudes de resistencia al flujo y de tubo de impedancia de la sección de caracterización."
  - type: article
    authors: ["Jiménez, N.", "Cox, T. J.", "Romero-García, V.", "Groby, J.-P."]
    year: 2017
    title: "Metadiffusers: Deep-subwavelength sound diffusers"
    journal: "Scientific Reports"
    volume: 7
    pages: "5389"
    doi: "10.1038/s41598-017-05710-5"
    note: "Acceso abierto. Difusores de panel de sublongitud de onda profunda cuyas ranuras resonantes de sonido lento alcanzan la difusión de un difusor de Schroeder o de residuo cuadrático en una fracción del espesor; en coautoría con Cox, autor del método del coeficiente de difusión de ISO 17497-2."
  - type: article
    authors: ["Jiménez, N.", "Groby, J.-P.", "Pagneux, V.", "Romero-García, V."]
    year: 2017
    title: "Iridescent perfect absorption in critically-coupled acoustic metamaterials using the transfer matrix method"
    journal: "Applied Sciences"
    volume: 7
    issue: 6
    pages: "618"
    doi: "10.3390/app7060618"
    note: "Acceso abierto. Un tutorial del método de la matriz de transferencia para absorbentes resonantes con terminación rígida y la condición de acoplamiento crítico para la absorción perfecta, con la misma maquinaria por capas que la predicción multicapa del tubo de impedancia de aquí."
  - type: article
    authors: ["Jiménez, N.", "Romero-García, V.", "Groby, J.-P."]
    year: 2018
    title: "Perfect absorption of sound by rigidly-backed high-porous materials"
    journal: "Acta Acustica united with Acustica"
    volume: 104
    issue: 3
    pages: "396-409"
    doi: "10.3813/AAA.919183"
    note: "La condición de acoplamiento crítico aplicada a una capa de absorbente de alta porosidad con terminación rígida, que conecta los modelos de material poroso de la sección de caracterización con la absorción perfecta a una sola frecuencia."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2004
    title: "Acoustics — Sound-scattering properties of surfaces — Part 1: Measurement of the random-incidence scattering coefficient in a reverberation room"
    designation: "ISO 17497-1:2004+A1:2014"
    url: "https://www.iso.org/standard/31397.html"
    note: "La edición implementada aquí: el método del coeficiente de dispersión con mesa giratoria y su cadena de incertidumbre del Anexo A."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2012
    title: "Acoustics — Sound-scattering properties of surfaces — Part 2: Measurement of the directional diffusion coefficient in a free field"
    designation: "ISO 17497-2:2012"
    url: "https://www.iso.org/standard/55293.html"
    note: "El coeficiente de difusión direccional en campo libre y su ponderación por áreas de ángulo sólido."
  - type: standard
    organization: "International Organization for Standardization"
    year: 1998
    title: "Acoustics — Determination of sound absorption coefficient and impedance in impedance tubes — Part 2: Transfer-function method"
    designation: "ISO 10534-2:1998"
    url: "https://www.iso.org/standard/22851.html"
    note: "El método de la función de transferencia con dos micrófonos de la sección de tubo de impedancia. Adoptada en Europa como EN ISO 10534-2:2001; revisada después como ISO 10534-2:2023 (https://www.iso.org/standard/81294.html)."
  - type: standard
    organization: "ASTM International"
    year: 2019
    title: "Standard test method for normal incidence determination of porous material acoustical properties based on the transfer matrix method"
    designation: "ASTM E2611-19"
    url: "https://store.astm.org/e2611-19.html"
    note: "La descomposición por matriz de transferencia con cuatro micrófonos y su pérdida por transmisión. La edición implementada aquí; revisada después como ASTM E2611-24 (https://store.astm.org/e2611-24.html)."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2018
    title: "Acoustics — Determination of airflow resistance — Part 1: Static airflow method"
    designation: "ISO 9053-1:2018"
    url: "https://www.iso.org/standard/69869.html"
    note: "El método estático de resistencia al flujo de aire y su velocidad de referencia."
---

Esta página reúne la teoría de materiales y superficies: la dispersión y difusión superficial, la absorción in situ de pavimentos y la caracterización de materiales acústicos, del índice ponderado de absorción a la resistencia al flujo de aire y el tubo de impedancia. Forma parte de la [referencia de teoría](/phonometry/es/reference/theory/). La medición en cámara reverberante de ISO 354 que alimenta el índice de ISO 11654 se cubre en [Salas y edificación](/phonometry/es/reference/theory/rooms-buildings/).

## Dispersión superficial y difusión (ISO 17497-1, ISO 17497-2)

### Coeficiente de dispersión a incidencia aleatoria (ISO 17497-1)

Una superficie rugosa reparte la energía reflejada en una parte especular y
otra dispersa; el coeficiente de dispersión $s$ es la fracción de energía no
especular. ISO 17497-1:2004+A1:2014 lo mide en una cámara reverberante con la
muestra sobre una mesa giratoria: cuatro tiempos de reverberación, con la
mesa parada y girando, cada uno sin y con la muestra (Tabla 2), dan la
absorción a incidencia aleatoria $\alpha_s$ (cláusula 8.1.1, Fórmula 1) y la
absorción *especular* $\alpha_{spec}$ (cláusula 8.1.2, Fórmula 4). La
rotación decorrelaciona las reflexiones dispersas entre decaimientos, de modo
que se promedian y se registran como "absorción" extra, y el coeficiente de
dispersión se sigue (cláusula 8.1.3, Fórmula 5):

$$
s = \frac{\alpha_{spec} - \alpha_s}{1 - \alpha_s},
$$

siendo cada $\alpha$ una diferencia de Sabine entre dos condiciones,
$55{,}3 (V/S) [1/(c_b T_b) - 1/(c_a T_a)] - 4 (V/S)(m_b - m_a)$, con
$c = 343{,}2 \sqrt{(273{,}15 + t)/293{,}15}$ (Fórmula 2) y $m$ de ISO 9613-1
mediante $m = \alpha_{dB}/(10 \lg e)$ (Fórmula 3). La propia placa base debe
dispersar poco: la Tabla 1 acota su coeficiente (Fórmula 6) a 0,05–0,25 entre
100 Hz y 5 kHz (cláusula 6.2). El $s$ negativo se trunca a cero para su
presentación (cláusula 8.3), pero los valores por encima de 1 en bandas
cercanas al rasante se conservan (cláusula 6.3.2). La cadena de incertidumbre
del Anexo A ($u_\alpha$, Fórmulas A.3/A.4; $u_s$, Fórmula A.5; $U = 2 u_s$)
está implementada. Como la norma no imprime ningún ejemplo resuelto, el
oráculo es una cadena sintética de extremo a extremo ($V = 200$ m³,
$S = 10$ m², $T = 8{,}0/6{,}0/7{,}5/5{,}0$ s → $s = 0{,}093$) más el valor a
mano de la Fórmula A.5, $u_s = 0{,}0297$.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/scattering_coefficient_es.svg" alt="El coeficiente de dispersión de incidencia aleatoria s de una superficie difusora sobre las 13 bandas de tercio de octava de 250 a 4000 Hz, creciendo suavemente desde casi cero a baja frecuencia hasta 0,84 a 4 kHz" style="width:88%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/scattering_coefficient_es_dark.svg" alt="El coeficiente de dispersión de incidencia aleatoria s de una superficie difusora sobre las 13 bandas de tercio de octava de 250 a 4000 Hz, creciendo suavemente desde casi cero a baja frecuencia hasta 0,84 a 4 kHz" style="width:88%" loading="lazy">

*Un coeficiente de dispersión de incidencia aleatoria que crece con la frecuencia a medida que la rugosidad se hace comparable con la longitud de onda.*

### Coeficiente de difusión direccional (ISO 17497-2)

ISO 17497-2:2012 mide, en campo libre, cuán uniformemente reparte una
superficie su respuesta polar reflejada sobre $n$ micrófonos. El coeficiente
basado en autocorrelación (cláusula 8.1, Fórmula 5) es

$$
d_\theta = \frac{\left( \sum_i p_i \right)^2 - \sum_i p_i^2}{(n - 1) \sum_i p_i^2},
\qquad p_i = 10^{L_i/10},
$$

1 para una respuesta perfectamente uniforme y tendiendo a 0 para un único
lóbulo especular; la Fórmula 6 es la forma ponderada por área con
$N_i = A_i / A_{min}$ a partir de los factores de ángulo sólido de la
Fórmula 8 ($A_i = (4\pi/\Delta\phi) \sin^2(\Delta\theta/4)$ en el cénit).
Normalizar frente a un reflector plano de referencia del mismo tamaño elimina
la difracción de borde (cláusula 8.2, Fórmula 7):
$d_{\theta,n} = (d_\theta - d_{\theta,r})/(1 - d_{\theta,r})$. El valor a
incidencia aleatoria promedia los ángulos de fuente con pesos 1:3:3:3:3 para
0°, ±30°, ±60° (cláusula 8.4). Anclajes: el arco de 37 receptores predicho
por el modelo para el QRD N = 7 publicado de seis periodos (Cox & D'Antonio
3.ª ed., Apéndice B; Hargreaves et al. 2000, Tabla I) a 1000 Hz da
$d_\theta = 0{,}1099$, su referencia plana $0{,}0049$ y
$d_{\theta,n} = 0{,}1055$; las predicciones del modelo promediadas por banda
coinciden con la difusión normalizada BEM publicada del Apéndice B en las
bandas de 200-400 Hz dentro de 0,01 (anclaje de baja frecuencia: la
desviación absoluta media en banda ancha 100-5000 Hz ronda 0,09); factor de
área del cénit 1,5710.

Consulta la [guía de dispersión superficial](/phonometry/es/guides/surface-scattering/) para su uso.

## Absorción in situ de superficies de carretera (ISO 13472-1, ISO 13472-2)

ISO 13472-1:2002 (método de superficie extendida) recupera la absorción a
incidencia normal de un pavimento en su emplazamiento, con un micrófono sobre
él: las componentes directa y reflejada de una respuesta al impulso se separan
con la **técnica de sustracción** y la **ventana de Adrienne** (cláusula 6.4:
un flanco de subida abrupto, una meseta obligatoria de 5 ms y un flanco de
bajada Blackman-Harris), y

$$
\alpha(f) = 1 - \frac{1}{K_r^2} \left| \frac{H_r(f)}{H_i(f)} \right|^2,
\qquad K_r = \frac{d_s - d_m}{d_s + d_m} = \frac{2}{3}
$$

para la geometría obligatoria $d_s = 1{,}25$ m, $d_m = 0{,}25$ m
(cláusula 4.2, Anexo C); $K_r$ es el cociente de expansión esférica entre el
camino directo y el de la imagen. Referenciar la medición del pavimento a otra
sobre una superficie muy reflectante cancela toda la cadena electroacústica
junto con $K_r$ (Anexo B). La ventana de 5 ms acota el área muestreada (forma
cerrada del Anexo A: radio ≈ 1,34 m para la geometría normalizada) y el rango
válido es de 250 Hz a 4 kHz en tercios de octava. ISO 13472-2:2010 (método
puntual, 250–1600 Hz) acopla en cambio un pequeño tubo de impedancia a la
superficie y remite las matemáticas al método de función de transferencia de
ISO 10534-2 de más abajo (sus cláusulas 4/5.7/6.6); la implementación
reutiliza ese módulo, añadiendo la geometría y los límites de validez de la
Parte 2 ($f_u = 0{,}58\ c_0/d$; cotas de separación de micrófonos
$0{,}45\ c_0/f_{max}$ y $0{,}05\ c_0/f_{min}$, cláusula 5.4) y la corrección
sustractiva del Anexo A por pérdidas internas del sistema.

Consulta la [guía de dispersión superficial](/phonometry/es/guides/surface-scattering/) para su uso.

## Caracterización de materiales acústicos (ISO 11654, ISO 9053-1/2, ISO 10534-1/2, ASTM E2611)

### Absorción acústica ponderada (ISO 11654)

ISO 11654:1997 condensa una curva de absorción en tercios de octava de ISO 354
en un solo número. El coeficiente práctico $\alpha_p$ promedia los tres
tercios de cada octava de 250 Hz a 4 kHz y redondea a pasos de 0,05
(cláusula 4.1). La curva de referencia (0,80, 1,00, 1,00, 1,00, 0,90 en
250–4000 Hz) se desplaza después hacia abajo en pasos de 0,05 hasta que la
suma de desviaciones desfavorables, contadas solo donde la medición queda
*por debajo* de la curva desplazada, es $\le 0{,}10$; $\alpha_w$ es la curva
desplazada a 500 Hz (cláusula 4.2). Un indicador de forma señala el exceso de
absorción $\ge 0{,}25$ sobre la curva desplazada: L a 250 Hz, M a
500/1000 Hz, H a 2000/4000 Hz (cláusula 4.3), y el Anexo B informativo asigna
$\alpha_w$ a las clases de absorción A–E. Como toda magnitud es múltiplo de
0,05, la implementación hace toda la aritmética de la malla en veinteavos
enteros, dejando la búsqueda del desplazamiento y las fronteras de clase
exactas y a salvo del punto flotante. Los dos ejemplos resueltos del Anexo A
se reproducen: $\alpha_p = (0{,}35, 0{,}70, 0{,}65, 0{,}60, 0{,}55)$ →
$\alpha_w = 0{,}60$, clase C; y subir 500 Hz a 1,00 mantiene
$\alpha_w = 0{,}60$ pero añade el indicador, "0,60(M)".

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_rating_es.svg" alt="Valoración ponderada de absorción acústica de ISO 11654: el espectro de absorción práctica trazado frente a la curva de referencia desplazada de 250 Hz a 4000 Hz, con la desviación desfavorable a 250 Hz sombreada y el coeficiente ponderado alpha_w leído a 500 Hz" style="width:80%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/absorption_rating_es_dark.svg" alt="Valoración ponderada de absorción acústica de ISO 11654: el espectro de absorción práctica trazado frente a la curva de referencia desplazada de 250 Hz a 4000 Hz, con la desviación desfavorable a 250 Hz sombreada y el coeficiente ponderado alpha_w leído a 500 Hz" style="width:80%" loading="lazy">

*La valoración de ISO 11654: absorción práctica frente a la referencia desplazada, con la desviación desfavorable sombreada y el coeficiente ponderado leído a 500 Hz.*

### Resistencia al flujo de aire (ISO 9053-1/2)

La resistividad al flujo $\sigma = R\,A/d$ es el parámetro de transporte clave
de un absorbente poroso. ISO 9053-1:2018 (método estático) hace pasar un flujo
estacionario por la muestra y ajusta $\Delta p = a\,u + b\,u^2$ por el origen
(cláusula 7.5); como $R_s = \Delta p / u = a + b\,u$, el coeficiente lineal es
la resistencia específica a velocidad nula, informada a la velocidad de
referencia $u = 0{,}5$ mm/s. ISO 9053-2:2020 (método alternante) sustituye el
caudalímetro por un pistón a ~2 Hz y un micrófono en una cavidad cerrada
(cláusula 8.7, Fórmula 2):

$$
R = \kappa'\ \frac{p_s}{2 \pi f V}\ \frac{h_t}{h_s}\ 10^{(L_{ps} - L_{pt})/20}
$$

Solo entra una *diferencia* de niveles, así que el equipo de medición sonora
no necesita calibración absoluta. El exponente efectivo $\kappa'$ (Anexo A,
Fórmula A.7) corrige el $\kappa$ adiabático por la conducción de calor a las
paredes a través de la capa límite térmica $b = \sqrt{2 c_0 l_h / \omega}$
(Fórmulas A.4/A.5). El ejemplo resuelto del Anexo A.3 (cilindro cerrado de
100 mm a 2 Hz: $b = 1{,}83$ mm, $\kappa' = 1{,}370 = 0{,}978\,\kappa$) se
reproduce, y se aplican las salvaguardas de validez de la Fórmula 3 (cociente
de transferencia < 0,3) y la Fórmula 4 (margen de 10 dB sobre el fondo).

### Tubo de impedancia (ISO 10534-1, ISO 10534-2, ASTM E2611)

Un tubo por debajo de su frecuencia de corte ($f d < 0{,}58\ c_0$ circular,
$< 0{,}50\ c_0$ rectangular; límites de separación de micrófonos
$f s < 0{,}45\ c_0$ y $f > c_0/(20 s)$; cláusulas 4.2–4.5) solo propaga ondas
planas, así que el factor de reflexión superficial de una muestra es
plenamente observable. ISO 10534-2 (método de función de transferencia)
compara la función de transferencia medida entre dos micrófonos $H_{12}$ con
las analíticas incidente y reflejada $H_I = e^{-j k_0 s}$,
$H_R = e^{+j k_0 s}$ (Anexo D) para dar (cláusula 7, Ec. 17):

$$
r = \frac{H_{12} - H_I}{H_R - H_{12}}\ e^{2 j k_0 x_1}, \qquad
\alpha = 1 - |r|^2, \qquad \frac{Z}{\rho c_0} = \frac{1 + r}{1 - r},
$$

con la cota inferior de atenuación del número de onda complejo
$k_0'' = 1{,}94 \times 10^{-2} \sqrt{f}/(c_0 d)$ (Ec. A.18). ISO 10534-1
(método de la razón de onda estacionaria) es el clásico en forma cerrada:
$|r| = (s - 1)/(s + 1)$ a partir de la razón máximo/mínimo
$s = 10^{\Delta L/20}$ y la fase desde la posición del primer mínimo
(Ecs. 12–26); una SWR de 3 da exactamente $|r| = 0{,}5$ y
$\alpha = 0{,}75$. ASTM E2611-19 añade la transmisión: cuatro micrófonos
descomponen los campos aguas arriba y abajo en las ondas $A, B, C, D$
(Ecs. 17–20) y una resolución con dos cargas (o con una carga para muestra
simétrica) da la **matriz de transferencia** 2×2 de la muestra,
$[p; u]_0 = T\,[p; u]_d$ (Ecs. 16/22–24), de la que sale la pérdida por
transmisión a incidencia normal con terminación anecoica (Ecs. 25/26)

$$
TL = 20 \lg \frac{\left| T_{11} + T_{12}/\rho c + \rho c\ T_{21} + T_{22} \right|}{2},
$$

más la reflexión con fondo rígido
$R = (T_{11} - \rho c\,T_{21})/(T_{11} + \rho c\,T_{21})$ (Ec. 27), el número
de onda del material $\arccos(T_{11})/d$ (Ec. 29) y la impedancia
característica $\sqrt{T_{12}/T_{21}}$ (Ec. 30). Las tres normas conservan
deliberadamente su propio convenio de signos y sus unidades de temperatura
(ISO en kelvin, ASTM en Celsius), y las resoluciones de cargas casi singulares
emiten un aviso. Como ninguna norma imprime un ejemplo numérico, los oráculos
son identidades físicas: la matriz analítica de una capa de aire
($\det T = 1$, $T_{11} = T_{22}$, TL = 0 dB, $|R| = 1$ con fondo rígido),
recorridos de ida y vuelta sintéticos que recuperan una $r$ conocida y la
recuperación con dos cargas de una muestra recíproca asimétrica.

Consulta la [guía de materiales](/phonometry/es/guides/materials/) para su uso.
