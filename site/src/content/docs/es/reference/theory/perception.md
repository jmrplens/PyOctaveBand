---
title: "Percepción y audición"
description: "Líneas isofónicas, modelos de sonoridad, métricas de calidad sonora, prominencia tonal, inteligibilidad del habla y estadística de la audición detrás de phonometry."
references:
  - type: article
    authors: ["Fletcher, H.", "Munson, W. A."]
    year: 1933
    title: "Loudness, its definition, measurement and calculation"
    journal: "The Journal of the Acoustical Society of America"
    volume: 5
    issue: 2
    pages: "82-108"
    doi: "10.1121/1.1915637"
    note: "Las mediciones originales de igual sonoridad que sustentan el concepto de isófona de la primera sección."
  - type: book
    authors: ["Fastl, H.", "Zwicker, E."]
    year: 2007
    title: "Psychoacoustics: Facts and models"
    edition: "3.ª ed."
    publisher: "Springer"
    doi: "10.1007/978-3-540-68888-4"
    note: "La psicoacústica de bandas críticas, enmascaramiento y sonoridad que subyace al modelo de Zwicker y a las sensaciones de agudeza y aspereza."
  - type: article
    authors: ["Houtgast, T.", "Steeneken, H. J. M."]
    year: 1985
    title: "A review of the MTF concept in room acoustics and its use for estimating speech intelligibility in auditoria"
    journal: "The Journal of the Acoustical Society of America"
    volume: 77
    issue: 3
    pages: "1069-1077"
    doi: "10.1121/1.392224"
    note: "El marco de transferencia de modulación de la sección del STI."
  - type: article
    authors: ["French, N. R.", "Steinberg, J. C."]
    year: 1947
    title: "Factors governing the intelligibility of speech sounds"
    journal: "The Journal of the Acoustical Society of America"
    volume: 19
    issue: 1
    pages: "90-119"
    doi: "10.1121/1.1916407"
    note: "Los experimentos por bandas de articulación que sustentan la función de importancia de banda del SII."
  - type: article
    authors: ["Passchier-Vermeer, W."]
    year: 1974
    title: "Hearing loss due to continuous exposure to steady-state broad-band noise"
    journal: "The Journal of the Acoustical Society of America"
    volume: 56
    issue: 5
    pages: "1585-1593"
    doi: "10.1121/1.1903482"
    note: "Un estudio de campo de las relaciones exposición-respuesta codificadas después en la ISO 1999."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2023
    title: "Acoustics — Normal equal-loudness-level contours"
    designation: "ISO 226:2023"
    url: "https://www.iso.org/standard/83117.html"
    note: "El modelo de isófonas de las Fórmulas (1)/(2) y los parámetros de la Tabla 1 de la sección de igual sonoridad."
  - type: standard
    organization: "Ecma International"
    year: 2024
    title: "Psychoacoustic metrics for ITT equipment — Part 1: Prominent discrete tones"
    designation: "ECMA-418-1:2024 (3.ª ed.)"
    url: "https://ecma-international.org/wp-content/uploads/ECMA-418-1_3rd_edition_december_2024.pdf"
    note: "El modelo de banda crítica y los procedimientos TNR y PR de la sección de prominencia tonal. El PDF enlazado es la descarga gratuita."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2005
    title: "Acoustics — Reference zero for the calibration of audiometric equipment — Part 7: Reference threshold of hearing under free-field and diffuse-field listening conditions"
    designation: "ISO 389-7:2005"
    url: "https://www.iso.org/standard/38976.html"
    note: "Los umbrales de referencia de campo libre y campo difuso de la Tabla 1 de la sección de umbral de audición."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2017
    title: "Acoustics — Statistical distribution of hearing thresholds related to age and gender"
    designation: "ISO 7029:2017"
    url: "https://www.iso.org/standard/42916.html"
    note: "El desplazamiento mediano dependiente de la edad y las dispersiones por fractiles del modelo de presbiacusia."
  - type: standard
    organization: "International Organization for Standardization"
    year: 2013
    title: "Acoustics — Estimation of noise-induced hearing loss"
    designation: "ISO 1999:2013"
    url: "https://www.iso.org/standard/45103.html"
    note: "El modelo de NIPTS, sus fractiles y la suma comprimida HTLAN de la sección de pérdida auditiva."
---

Esta página reúne la teoría de la audición y la psicoacústica: las líneas isofónicas, los modelos de sonoridad de Zwicker, Moore-Glasberg y Sottek, las métricas de calidad sonora (tonalidad, aspereza y agudeza), la prominencia tonal, las métricas del habla STI y SII, y la estadística de los umbrales de audición y de la pérdida auditiva. Forma parte de la [referencia de teoría](/phonometry/es/reference/theory/).

## Líneas isofónicas (ISO 226:2023)

Un tono tiene un *nivel de sonoridad* de $L_N$ fonios cuando se juzga igual de sonoro que un tono puro de 1 kHz a $L_N$ dB SPL. La Fórmula (1) de ISO 226:2023 (cláusula 4.1, p. 2) da el SPL de un tono puro a la frecuencia $f$ que alcanza el nivel de sonoridad $L_N$:

$$
L_f = \frac{10}{\alpha_f} \log_{10}\left[ \left(4 \cdot 10^{-10}\right)^{0{,}3 - \alpha_f} \left( 10^{\ 0{,}03 L_N} - 10^{\ 0{,}072} \right) + 10^{\ \alpha_f (T_f + L_U)/10} \right] - L_U
$$

La Fórmula (2) (cláusula 4.2) la invierte, devolviendo el nivel de sonoridad de un tono a SPL $L_f$:

$$
L_N = \frac{100}{3} \log_{10}\left[ \frac{10^{\ \alpha_f (L_f + L_U)/10} - 10^{\ \alpha_f (T_f + L_U)/10}}{\left(4 \cdot 10^{-10}\right)^{0{,}3 - \alpha_f}} + 10^{\ 0{,}072} \right]
$$

Los tres parámetros provienen de la Tabla 1 (p. 4), tabulados en las 29 frecuencias preferentes de tercio de octava de ISO 266 desde 20 Hz hasta 12,5 kHz:

- $\alpha_f$: exponente de la percepción de sonoridad a la frecuencia $f$,
- $L_U$: magnitud de la función de transferencia lineal, normalizada en 1 kHz ($L_U = 0$ en 1 kHz),
- $T_f$: umbral de audición en $f$, en dB.

La norma **no especifica interpolación** entre las frecuencias tabuladas. La Fórmula (1) está especificada para **20 fonios a 90 fonios** entre 20 Hz y 4 kHz, y solo hasta **80 fonios entre 5 kHz y 12,5 kHz**; por encima de 80 fonios la línea isofónica se detiene, por tanto, en 4 kHz. Los valores fuera de estos límites obtenidos con la Fórmula (2) son extrapolaciones que la norma califica de meramente informativas.

Consulta la [guía de sonoridad](/phonometry/es/guides/loudness/) para su uso.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/equal_loudness_contours_es.svg" alt="Curvas isofónicas normales de ISO 226:2023 de 20 a 90 fonios con la curva del umbral de audición" style="width:80%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/equal_loudness_contours_es_dark.svg" alt="Curvas isofónicas normales de ISO 226:2023 de 20 a 90 fonios con la curva del umbral de audición" style="width:80%" loading="lazy">

*Las isófonas de ISO 226:2023 según la Fórmula (1), de 20 a 90 fonios, con el umbral de audición.*

## Prominencia tonal: TNR y PR (ECMA-418-1)

Ambos métodos operan sobre un espectro de potencia con ventana de Hann promediado en RMS (cláusulas 11.1 / 12.1) y usan el modelo de bandas críticas de la cláusula 10. El ancho de banda crítico centrado en un tono a $f$ es (Fórmula 2):

$$
\Delta f_c = 25{,}0 + 75{,}0 \left(1{,}0 + 1{,}4 \left(\tfrac{f}{1000}\right)^2\right)^{0{,}69} \ \text{Hz}
$$

Los bordes de banda se colocan **aritméticamente** para $f \le 500$ Hz (Fórmulas 4–5): $f_{1,2} = f \mp \Delta f_c / 2$, y **geométricamente** por encima (Fórmulas 7–8): $f_1 = -\Delta f_c/2 + \sqrt{\Delta f_c^2 + 4 f^2}/2$, $f_2 = f_1 + \Delta f_c$.

**TNR** (cláusula 11). La banda del tono abarca los mínimos espectrales a ambos lados del pico dentro del 15 % de $\Delta f_c$ (cláusula 11.2). A la potencia del tono se le resta la recta que conecta los bins de los bordes de banda (Fórmula 9): sobre los $N$ bins de la banda del tono, $P_t = \sum_k P_k - (P_{\text{lo}} + P_{\text{hi}})\ N/2$. La potencia del ruido enmascarante es la potencia restante de la banda crítica reescalada al ancho de banda crítico completo (Fórmula 10): $P_n = (P_{\text{band}} - P_t) \cdot \Delta f_c / \Delta f_{\text{band}}$, y $\mathrm{TNR} = 10\log_{10}(P_t/P_n)$ (Fórmula 11). El criterio de prominencia (Fórmulas 12–13) es

$$
\mathrm{TNR}_{\text{crit}} = \begin{cases} 8{,}0 + 8{,}33 \log_{10}(1000/f_t) \ \text{dB} & f_t < 1\ \text{kHz} \\ 8{,}0 \ \text{dB} & f_t \ge 1\ \text{kHz} \end{cases}
$$

**PR** (cláusula 12) compara el nivel de la banda crítica centrada en el tono, $L_M$, con la potencia media de las dos bandas críticas **contiguas** $L_L$, $L_U$ (bordes según las Fórmulas ajustadas 21–22 con las Tablas 2–3): $\mathrm{PR} = 10\log_{10} P_M - 10\log_{10}\left[(P_L + P_U)/2\right]$ (Fórmula 23). Para $f_t \le 171{,}4$ Hz la banda inferior se trunca en 20 Hz y su potencia se reescala a un **ancho de banda de 100 Hz** (Fórmula 24). El criterio (Fórmulas 25–26) es 9,0 dB para $f_t \ge 1$ kHz, y crece como $9{,}0 + 10{,}0\log_{10}(1000/f_t)$ por debajo. Los tonos se evalúan dentro del rango de interés de 89,1 Hz – 11,2 kHz (cláusulas 11.5 / 12.6).

Consulta la [guía de tonos discretos prominentes](/phonometry/es/guides/tone-prominence/) para su uso.

## Sonoridad de Zwicker (ISO 532-1)

El oído analiza el sonido en **bandas críticas**: regiones de frecuencia dentro de las cuales la energía se suma antes de formarse la sonoridad. La **escala Bark** transforma la frecuencia en razón de banda crítica $z$, de 0 a 24 Bark, e ISO 532-1:2017 muestrea la sonoridad específica $N'(z)$ en pasos de 0,1 Bark (240 valores). La implementación es un port de sala limpia del programa de referencia normativo de la norma (Anexo A.4) y procede por etapas:

1. **Niveles de tercio de octava**: 28 bandas, de 25 Hz a 12,5 kHz (el banco de filtros del Anexo A a 48 kHz, Tablas A.1/A.2). Para sonidos variables en el tiempo, las salidas de banda al cuadrado se suavizan con tres paso-bajos en cascada con $\tau = 2/(3 f_c)$ ($f_c$ limitada a 1 kHz) y se muestrean cada 2 ms.
2. **Agrupación en baja frecuencia**: las 11 bandas hasta 250 Hz reciben las correcciones isofónicas de la Tabla A.3 y se suman en las tres primeras bandas críticas (25–80, 100–160, 200–250 Hz).
3. **Transmisión a0**: la corrección de transferencia del oído externo/medio de la Tabla A.4 (más la diferencia de campo difuso de la Tabla A.5 con `field='diffuse'`) produce los niveles de banda crítica $L_E$.
4. **Sonoridad núcleo**: cada una de las 20 bandas críticas se transforma con los niveles de umbral en silencio $L_{TQ}$ de la Tabla A.6 (tras la adaptación de ancho de banda DCB de la Tabla A.7):

   $$
   N_c = \max\left(0,\ 0{,}0635 \cdot 10^{0{,}025 L_{TQ}} \left[ \left( 1 - s + s \cdot 10^{(L_E - L_{TQ})/10} \right)^{0{,}25} - 1 \right]\right) \ \text{sonos/Bark}, \qquad s = 0{,}25
   $$

   (la forma que da el programa de referencia a la transformación de sonoridad de Zwicker; las bandas por debajo del umbral aportan cero).

5. **Pendientes**: las pendientes superiores de enmascaramiento dependientes del nivel (inclinación por rango de sonoridad específica y banda crítica, Tablas A.8/A.9) añaden flancos decrecientes hacia $z$ mayores; la sonoridad total es el área bajo el patrón:

   $$
   N = \int_0^{24} N'(z)\ dz \ \ \text{sonos}
   $$

Para sonidos variables en el tiempo, un decaimiento temporal no lineal (constantes de tiempo 5/15/75 ms, cláusula 6.3) y la ponderación dependiente de la duración de la sonoridad total (paso-bajos de 3,5 ms y 70 ms ponderados 0,47/0,53, cláusula 6.4) preceden a la salida de sonoridad frente al tiempo a 500 Hz y a los percentiles N5/N10 (cláusula 6.5).

**Sonos y fonios** quedan ligados por el ancla de 1 kHz (1 sono = 40 fonios; cláusula 5.6):

$$
N = 2^{(L_N - 40)/10} \ \text{sonos} \qquad \Longleftrightarrow \qquad L_N = 40 + 10 \log_2 N \ \text{fonios} \qquad (N \ge 1)
$$

por debajo de 1 sono el programa de referencia usa $L_N = 40 (N + 0{,}0005)^{0{,}35}$, con suelo en 3 fonios.

Consulta la [guía de sonoridad](/phonometry/es/guides/loudness/) para su uso.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/loudness_pattern_es.svg" alt="Patrones de sonoridad específica sobre la escala Bark para un sonido de banda estrecha de 1 kHz y un sonido de banda ancha con el mismo nivel de banda" style="width:80%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/loudness_pattern_es_dark.svg" alt="Patrones de sonoridad específica sobre la escala Bark para un sonido de banda estrecha de 1 kHz y un sonido de banda ancha con el mismo nivel de banda" style="width:80%" loading="lazy">

*Sonoridad específica N′(z) sobre el eje de Bark: la energía repartida entre muchas bandas críticas suma más sonos que el mismo nivel de banda concentrado en una sola.*

## Modelos avanzados de sonoridad y calidad sonora

ISO 532-1 es uno de tres modelos de sonoridad; dos familias más recientes refinan el front-end auditivo y añaden las métricas de calidad sonora tonalidad y aspereza.

### Sonoridad de Moore-Glasberg (ISO 532-2:2017, ISO 532-3:2023)

En lugar de las bandas críticas fijas de Zwicker, el modelo de Moore-Glasberg forma un **patrón de excitación** continuo sobre la escala del número ERB ("Cam") usando filtros auditivos **exponenciales redondeados (roex)** dependientes del nivel. En función de la desviación de frecuencia normalizada $g = |f - f_c| / f_c$ respecto a un filtro centrado en $f_c$, la ponderación del filtro es

$$
W(g) = (1 + p\ g)\ e^{-p\ g}
$$

donde la pendiente $p$ crece con el nivel de la fuente, ensanchando el flanco inferior a medida que sube el nivel (ISO 532-2, Fórmulas 2–5); esto reproduce la propagación ascendente del enmascaramiento. Al pasar la intensidad del estímulo por cada filtro se obtiene la excitación $E(i)$, y una ley compresiva la transforma en la **sonoridad específica** $N'(i)$ en sonos/Cam (Fórmulas 7–9), de la forma de nivel medio

$$
N'(i) = C \left[ \left( G\ \frac{E(i)}{E_0} + A \right)^{\alpha} - A^{\alpha} \right]
$$

con la constante de calibración $C = 0{,}0617$ sonos/Cam (ISO 532-2; $0{,}063$ en ISO 532-3). La sonoridad total es el área bajo el patrón,

$$
N = \int N'(i)\ di \ \ \text{sonos}
$$

y una etapa de inhibición binaural (Fórmulas 10–13) combina los oídos de modo que un sonido diótico es más sonoro que el mismo sonido en un solo oído. El ancla de 1 kHz / 40 dB SPL da exactamente 1 sono.

**ISO 532-3** lo hace variable en el tiempo. Un espectro deslizante de seis FFT paralelas con ventana de Hann (longitudes de segmento de 2 a 64 ms, cada una aportando su propio rango de frecuencias, actualizado cada $T_0 = 1$ ms) alimenta la misma cadena de excitación y sonoridad específica, integrada por dos suavizadores de primer orden en cascada con $\alpha = 1 - e^{-T_0 / \tau}$,

$$
S(t) = \alpha\ x(t) + (1 - \alpha)\ S(t - 1)
$$

usando una constante de tiempo rápida en el ataque y una más lenta en la relajación. Esto produce la **sonoridad de corto plazo** $S'(t)$ (ataque/relajación en torno a 20–30 ms) y la **sonoridad de largo plazo** $S''(t)$ (en torno a 0,1–0,75 s); la sonoridad de largo plazo de pico $N_{\max} = \max_t S''(t)$ predice la sonoridad de sonidos de hasta unos 5 s.

### Modelo de Sottek (ECMA-418-2:2025)

ECMA-418-2 construye sus tres métricas sobre un único front-end auditivo (Cláusula 5): un filtro de oído externo/medio, un banco de 53 filtros paso-banda solapados de tipo gammatone espaciados en la escala Bark_HMS ($z = 0{,}5$ a $26{,}5$), rectificación de media onda y un RMS de bloque corto $\tilde{p}(l, z)$ por banda $z$ y bloque temporal $l$. Una no linealidad compresiva (Fórmula 23) convierte el RMS de banda en la **sonoridad de base específica** $N'_{\mathrm{basis}}(l, z)$, cuya constante de calibración $c_N$ fija un tono de 1 kHz / 40 dB SPL en 1 sone_HMS. La sonoridad ensambla las sonoridades tonal y de ruido (abajo) sobre bandas y tiempo (Fórmulas 113–117); crece unas $1{,}65\times$ por cada 10 dB, más lentamente que el factor de 2 de Zwicker, una propiedad intrínseca de la suma de Sottek.

### Tonalidad: autocorrelación de la señal de banda (ECMA-418-2)

Una componente tonal es periódica, así que sobrevive en la **función de autocorrelación** (ACF) de la señal rectificada de una banda mientras que el ruido de banda ancha se descorrelaciona. Para cada banda, la ACF no sesgada del bloque es

$$
\phi_z(m) = \frac{1}{M - m} \sum_{n=0}^{M - 1 - m} p_z(n)\ p_z(n + m)
$$

Una estimación espectral con ventana de $\phi_z$ separa una **sonoridad tonal** $N'_{\mathrm{tonal}}(l, z)$ de la **sonoridad de ruido** $N'_{\mathrm{noise}}(l, z)$ (Fórmulas 36–48). La tonalidad específica es la sonoridad tonal escalada por una compuerta suave de relación señal-ruido $q(l)$ (Fórmulas 49–51),

$$
T'(l, z) = c_T\ q(l)\ N'_{\mathrm{tonal}}(l, z)
$$

y el valor único $T$ (tu_HMS) es el promedio temporal con compuerta del máximo por bloque sobre las bandas (Fórmulas 61–64). La constante $c_T$ fija el tono de 1 kHz / 40 dB en 1 tu_HMS, y la banda del pico de la ACF da la frecuencia tonal $f_{\mathrm{ton}}$.

### Aspereza: modulación de la envolvente (ECMA-418-2)

La aspereza es la sensación de modulación de amplitud rápida (aproximadamente 20–300 Hz), máxima cerca de 70 Hz. A partir de la envolvente de cada banda $p_E(n)$ (magnitud de Hilbert) se forma un espectro de modulación ponderado por una función de tasa de modulación con pico cerca de 70 Hz y por la profundidad de modulación; correlacionando la modulación entre bandas vecinas y aplicando el filtrado temporal especificado se obtiene la **aspereza específica** $R'(l_{50}, z)$ y la aspereza dependiente del tiempo

$$
R(l_{50}) = \sum_z R'(l_{50}, z) \ \ \text{asper}
$$

(Fórmulas 65–111). El valor único $R$ es el percentil 90 de $R(l_{50})$ en el tiempo (Cláusula 7.1.10); la constante $c_R$ (Fórmula 104) calibra el sonido de referencia (un portador de 1 kHz modulado en amplitud al 100 % a 70 Hz y 60 dB SPL) a 1 asper.

### Agudeza (DIN 45692)

La agudeza (*sharpness*) condensa en un solo número el énfasis en alta frecuencia de un sonido: el primer momento ponderado por $g(z)$ del patrón de sonoridad específica estacionaria de ISO 532-1 (DIN 45692:2009, Ecuación 1):

$$
S = k\ \frac{\int_0^{24} N'(z)\ g(z)\ z\ dz}{\int_0^{24} N'(z)\ dz} \ \text{acum}, \qquad
g(z) = \begin{cases} 1 & z \le 15{,}8\ \text{Bark} \\ 0{,}15\ e^{0{,}42 (z - 15{,}8)} + 0{,}85 & z > 15{,}8\ \text{Bark} \end{cases}
$$

evaluado sobre la misma malla de 240 puntos a 0,1 Bark. La constante $k$ no está codificada a mano, sino que se deriva del requisito de calibración (cláusula 6): un ruido de banda estrecha de una banda crítica de ancho, 920–1080 Hz a 60 dB SPL, puntúa exactamente 1 acum, y la $k = 0{,}108$ derivada cae dentro de la ventana normativa $0{,}105 \le k < 0{,}115$ (cláusula 5.2). Las ponderaciones informativas del Anexo B se ofrecen bajo el mismo anclaje de 1 acum: von Bismarck (codo en 15 Bark, $0{,}2\ e^{0{,}308(z-15)} + 0{,}8$) y Aures (dependiente de la sonoridad, $g(z) = 0{,}078\ (e^{0{,}171 z}/z)\ N/\ln(0{,}05 N + 1)$). Los objetivos de banda estrecha de la Tabla A.2 se reproducen dentro de la tolerancia de la cláusula 6 (5 % o 0,05 acum): 0,38 acum a 250 Hz, 1,00 a 1 kHz, 1,78 a 2,5 kHz, 2,82 a 4 kHz.

Consulta la [guía de métricas de calidad sonora](/phonometry/es/guides/sound-quality/) para su uso.

## Transferencia de modulación y STI (IEC 60268-16)

La inteligibilidad del habla viaja en las modulaciones lentas de intensidad de la envolvente del habla. La **función de transferencia de modulación (MTF)** $m(F)$ de un canal de transmisión es la razón entre la profundidad de modulación recibida y la emitida de la envolvente de intensidad por banda de octava a la frecuencia de modulación $F$; el STI completo la evalúa en las 14 frecuencias de modulación en tercios de octava de 0,63 a 12,5 Hz en las siete bandas de octava de 125 Hz a 8 kHz (A.2.2). Desde una respuesta al impulso medida, la **forma cerrada de Schroeder** la da directamente (método indirecto):

$$
m_k(f_m) = \frac{\left| \int_0^{\infty} h_k^2(t)\ e^{-j 2 \pi f_m t}\ dt \right|}{\int_0^{\infty} h_k^2(t)\ dt}
$$

El ruido de fondo estacionario multiplica el $m$ de cada banda por la razón de intensidades (el término de ruido):

$$
m'_k = m_k \cdot \frac{I_k}{I_k + I_{n,k}} = \frac{m_k}{1 + 10^{-\mathrm{SNR}_k/10}}
$$

y cuando se conocen los niveles absolutos por banda, la corrección completa $m'_k = m_k I_k / (I_k + I_{am,k} + I_{rt,k} + I_{n,k})$ añade la intensidad de enmascaramiento auditivo $I_{am,k}$ (desde la banda de octava inmediatamente inferior, Tabla A.2) y el umbral absoluto de recepción $I_{rt,k}$ (Tabla A.3). Cada $m$ corregido se transforma en una **SNR efectiva**, recortada al rango de ±15 dB donde la inteligibilidad varía realmente, y después en un índice de transmisión (A.5.4/A.5.5):

$$
\mathrm{SNR}_{\mathrm{eff}} = 10 \log_{10} \frac{m}{1 - m}\ \text{dB}, \qquad \mathrm{TI} = \frac{\mathrm{SNR}_{\mathrm{eff}} + 15}{30}
$$

El MTI de banda es la media de los TI sobre las frecuencias de modulación, y el STI pondera las bandas con los factores masculinos $\alpha_k$, $\beta_k$ de la Tabla A.1 de la Ed. 5 (A.5.6):

$$
\mathrm{STI} = \sum_{k=1}^{7} \alpha_k\ \mathrm{MTI}_k - \sum_{k=1}^{6} \beta_k \sqrt{\mathrm{MTI}_k\ \mathrm{MTI}_{k+1}}
$$

truncado a 1,0. STIPA (Anexo B) muestrea la misma física con solo dos frecuencias de modulación por banda (Tabla B.1) sobre una señal de prueba con índice de modulación de fuente 0,55; las profundidades recibidas se miden por correlación seno/coseno de las envolventes de intensidad $I_k(t)$ filtradas paso-bajo a ~100 Hz sobre un número entero de periodos de modulación:

$$
m_{dr} = \frac{2 \sqrt{\left( \sum_t I_k(t) \sin 2 \pi f_m t \right)^2 + \left( \sum_t I_k(t) \cos 2 \pi f_m t \right)^2}}{\sum_t I_k(t)}, \qquad m = \frac{m_{dr}}{0{,}55}
$$

Consulta la [guía del índice de transmisión del habla](/phonometry/es/guides/speech-transmission/) para su uso.

## Índice de inteligibilidad del habla (ANSI S3.5)

Mientras el STI caracteriza un canal de transmisión, el SII (ANSI S3.5-1997) predice la inteligibilidad a partir de lo que el oyente puede oír realmente: 18 bandas de tercio de octava de 160 Hz a 8 kHz, cada una con su importancia de banda $I_i$ (Tabla 3, $\sum I_i = 1$, con máximo cerca de 2 kHz). Todas las entradas son niveles espectrales equivalentes (cláusulas 3.11/3.55). El habla se enmascara a sí misma hacia arriba: el espectro de enmascaramiento de cada banda $Z_i$ (cláusula 5.4) acumula las bandas inferiores con pendientes $C_i = -80 + 0{,}6\,(B_i + 10 \lg f_i - 6{,}353)$ dB, y la perturbación es el **mayor** entre el enmascaramiento y el suelo auditivo, $D_i = \max(Z_i, X'_i)$ (cláusula 5.6), con $X'_i = X_i + T'_i$ el espectro de ruido interno de referencia más el desplazamiento del umbral de audición del oyente (cláusulas 5.5/5.6). La audibilidad de banda recorta el margen habla-perturbación al intervalo $[0, 1]$ (cláusula 5.8), un factor de distorsión de nivel descuenta la presentación excesivamente alta (cláusula 5.7), y el índice suma (cláusula 6):

$$
A_i = \operatorname{clip}\Big( \frac{E'_i - D_i + 15}{30},\ 0,\ 1 \Big), \qquad
L_i = \operatorname{clip}\Big( 1 - \frac{E'_i - U_i - 10}{160},\ 0,\ 1 \Big), \qquad
\mathrm{SII} = \sum_{i=1}^{18} I_i\ L_i\ A_i .
$$

Los espectros normalizados de habla de la Tabla 3 para los esfuerzos vocales normal, elevado, fuerte y grito están incorporados (25,01 / 33,86 / 42,16 / 51,31 dB a 1 kHz); el $U_i$ del factor de distorsión de nivel es siempre el espectro de esfuerzo normal. Los anclajes: el espectro de esfuerzo normal en silencio con audición normal puntúa SII ≈ 0,996, los valores de referencia del espectro de enmascaramiento se reproducen a $10^{-4}$, y los espectros por esfuerzo vocal están verificados de forma cruzada frente a las implementaciones de referencia de Google y CRAN.

Consulta la [guía de inteligibilidad del habla](/phonometry/es/guides/speech-intelligibility/) para su uso.

## Umbrales de audición y presbiacusia (ISO 389-7, ISO 7029)

ISO 389-7:2005 Tabla 1 fija el umbral de audición de referencia de adultos jóvenes otológicamente normales: el SPL en campo libre y campo difuso que corresponde a 0 dB HL en las 11 frecuencias audiométricas de 125 Hz a 8 kHz (22,1 dB a 125 Hz en ambos campos, 2,4/0,8 dB libre/difuso a 1 kHz, divergiendo en alta frecuencia hasta 12,6 frente a 6,8 dB a 8 kHz). ISO 7029:2017 describe cómo se desplaza estadísticamente ese umbral con la edad: la desviación mediana respecto a los 18 años es (cláusula 4.2, Tabla 1)

$$
\Delta H_{md} = a\ (Y - 18)^b \ \text{dB},
$$

y cualquier fractil sigue un modelo gaussiano bilateral (cláusula 4.4), $\Delta H_Q = \Delta H_{md} + z(Q)\ s$, usando la dispersión superior $s_u$ para $z \ge 0$ (peor que la mediana) y la inferior $s_l$ en caso contrario, cada una un polinomio de grado 5 en $Y - 18$ por sexo y frecuencia (cláusula 4.3, Tablas 2–5). A los 18 años toda desviación es cero por construcción. Las fórmulas están establecidas hasta los 80 años a 2 kHz y por debajo, y hasta los 70 años por encima; más allá la evaluación es una extrapolación. Anclajes: a los 60 años las medianas evalúan a 7,85 dB (hombre, 1 kHz), 20,21 dB (hombre, 4 kHz) y 15,32 dB (mujer, 4 kHz), reproduciendo la fórmula de la Tabla 1 a $10^{-3}$.

Consulta la [guía de umbral de audición](/phonometry/es/guides/hearing-threshold/) para su uso.

<img class="light-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hearing_threshold_es.svg" alt="Dos paneles. Izquierda: la desviación mediana del umbral de audición de ISO 7029 para hombres a los 20, 40, 60 y 80 años en un eje de audiograma invertido, con la banda del 10 al 90 por ciento en torno a la curva de 70 años; la pérdida se acentúa hacia las altas frecuencias y con la edad. Derecha: el umbral de referencia de campo libre y campo difuso de ISO 389-7, que coinciden por debajo de 1 kHz y divergen por encima, con un mínimo cerca de 3 a 4 kHz" style="width:96%" loading="lazy"><img class="dark-only" src="https://raw.githubusercontent.com/jmrplens/phonometry/main/.github/images/hearing_threshold_es_dark.svg" alt="Dos paneles. Izquierda: la desviación mediana del umbral de audición de ISO 7029 para hombres a los 20, 40, 60 y 80 años en un eje de audiograma invertido, con la banda del 10 al 90 por ciento en torno a la curva de 70 años; la pérdida se acentúa hacia las altas frecuencias y con la edad. Derecha: el umbral de referencia de campo libre y campo difuso de ISO 389-7, que coinciden por debajo de 1 kHz y divergen por encima, con un mínimo cerca de 3 a 4 kHz" style="width:96%" loading="lazy">

*La desviación mediana con la edad de ISO 7029 con su banda de fractiles (izquierda) y los umbrales de referencia de campo libre y difuso de ISO 389-7 (derecha).*

## Pérdida auditiva inducida por ruido (ISO 1999)

ISO 1999:2013 predice el desplazamiento permanente del umbral que acumula una población expuesta a ruido. El desplazamiento mediano inducido por ruido (NIPTS) para 10–40 años de exposición es (cláusula 6.3.1, Fórmula 2, Tabla 1):

$$
N_{50} = \big[ u + v \lg(t/t_0) \big]\ (L_{EX,8h} - L_0)^2, \qquad t_0 = 1\ \text{año},
$$

cuadrático en el exceso sobre el nivel de inicio dependiente de la frecuencia $L_0$ (75 dB a 4 kHz, la banda más sensible, hasta 93 dB a 500 Hz) y cero por debajo; para menos de 10 años escala como $\lg(t+1)/\lg 11$ (Fórmula 3). Los fractiles añaden la dispersión, $N_Q = N_{50} + z\ d_{u,l}$ con $d = (X + Y \lg t)(L_{EX,8h} - L_0)^2$ (cláusula 6.3.2, Fórmulas 4–7, Tablas 2/3), acotada en cero; la convención cuenta la fracción de la población con el desplazamiento *menor*, así que $Q = 0{,}9$ es el decil más susceptible (rango fiable 0,05–0,95). El nivel de umbral de audición asociado a edad y ruido (HTLAN) combina el NIPTS con la componente de edad de ISO 7029 al mismo fractil mediante la suma comprimida (cláusula 6.1, Fórmula 1):

$$
H' = H + N - \frac{H\ N}{120}.
$$

Los ejemplos resueltos del Anexo D (Tablas D.1–D.4; p. ej. 100 dB / 40 años a 3 kHz: 29/38/60 dB en los fractiles 0,10/0,50/0,90) se reproducen exactamente con el redondeo a enteros de la norma, y el valor a mano de la Fórmula 2 a 4 kHz / 20 años / 90 dB es $N_{50} = 12{,}94$ dB.

Consulta la [guía de pérdida auditiva inducida por ruido](/phonometry/es/guides/noise-induced-hearing-loss/) para su uso.
