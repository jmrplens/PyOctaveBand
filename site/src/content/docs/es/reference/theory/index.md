---
title: "Teoría"
description: "Dónde viven las deducciones: seis páginas por dominio que devuelven cada método compartido al apartado, la ecuación y la tabla de los que sale, y las áreas cuya teoría se queda junto a la guía que la usa."
---

La referencia de teoría reúne las deducciones, las referencias de apartado y las
decisiones de diseño de las áreas cuya matemática **comparten muchas guías**.
Una página de teoría devuelve un método implementado al apartado, la ecuación y
la tabla de la norma o del libro de los que sale, enuncia la física que hay
detrás de cada término de corrección y las hipótesis que lo acotan, y da los
valores de referencia contra los que comprueba la batería de validación. No
muestra flujos de trabajo: para eso están las guías. Lo natural es llegar aquí
*desde* una guía cuando hay que justificar un término, o leer primero una página
de dominio al decidir si un método es aplicable siquiera.

Los seis dominios no son independientes. **El análisis de señal** sostiene todo
lo demás, porque el resto de páginas consume de él niveles de banda, curvas de
ponderación e integración temporal; **percepción y audición** explica las curvas
que varias otras páginas usan luego como ponderaciones; **salas y edificación**,
**materiales y superficies** y **medio ambiente y transporte** son los tres
dominios de aplicación, y se reparten entre ellos la maquinaria de potencia
acústica y de absorción; **vibración** aporta las magnitudes estructurales sobre
las que se apoyan la predicción de paneles y la de flancos.

Varias áreas se quedan con su teoría **dentro de sus guías** en lugar de aquí,
porque si no la deducción y el único método al que sirve quedarían separados
para nada. Es el caso de los módulos submarinos
([Acústica submarina](/phonometry/es/underwater/underwater-acoustics/),
[Propagación submarina](/phonometry/es/underwater/underwater-propagation/),
[Solvers numéricos de propagación submarina](/phonometry/es/underwater/underwater-solvers/) y
[Exposición a ruido de mamíferos marinos](/phonometry/es/underwater/marine-mammal-exposure/));
de los métodos de certificación y de curvas de ruido de
[Ruido de aeronaves](/phonometry/es/aircraft/); de los modelos de emisión viaria
y ferroviaria de CNOSSOS-EU en
[Fuentes ambientales](/phonometry/es/environment/sources/); de las medidas
electroacústicas de IEC 60268 y la cadena de radiodifusión de BS.1770 en
[Electroacústica](/phonometry/es/devices/electroacoustics/) y
[Radiodifusión](/phonometry/es/devices/broadcast/); de los modelos de
silenciador, de conductos y de recinto a recinto de [Control de
ruido](/phonometry/es/devices/noise-control/); y de los solvers FDTD y elástico,
cuyo método numérico, cota de estabilidad y regla de dispersión se desarrollan
en las propias páginas de [simulación de ondas](/phonometry/es/simulation/).
Todo lo de abajo va listado con las secciones que aloja cada página de dominio.

## [Análisis de señal](/phonometry/es/reference/theory/signal-analysis/)

De dónde salen los bordes de banda, las respuestas en magnitud de los filtros y
las curvas de ponderación, y las razones numéricas por las que el banco se
construye como una cascada diezmada de secciones de segundo orden; también las
deducciones de integración temporal, de intensidad y de incertidumbre GUM.

- [Frecuencias de banda de octava (ANSI S1.11 / IEC 61260)](/phonometry/es/reference/theory/signal-analysis/#frecuencias-de-banda-de-octava-ansi-s111--iec-61260)
- [Resolución frecuencial vs separación de bins FFT](/phonometry/es/reference/theory/signal-analysis/#resolución-frecuencial-vs-separación-de-bins-fft)
- [Respuestas en magnitud |H(jw)|](/phonometry/es/reference/theory/signal-analysis/#respuestas-en-magnitud-hjw)
- [Diseño del banco y estabilidad numérica](/phonometry/es/reference/theory/signal-analysis/#diseño-del-banco-y-estabilidad-numérica)
- [Curvas de ponderación (IEC 61672-1)](/phonometry/es/reference/theory/signal-analysis/#curvas-de-ponderación-iec-61672-1)
- [Integración temporal](/phonometry/es/reference/theory/signal-analysis/#integración-temporal)
- [Ponderación G (ISO 7196)](/phonometry/es/reference/theory/signal-analysis/#ponderación-g-iso-7196)
- [Métricas de evento y de dosis](/phonometry/es/reference/theory/signal-analysis/#métricas-de-evento-y-de-dosis)
- [Intensidad acústica (IEC 61043)](/phonometry/es/reference/theory/signal-analysis/#intensidad-acústica-iec-61043)
- [Incertidumbre de medida (ISO/IEC Guide 98-3: GUM y Suplemento 1)](/phonometry/es/reference/theory/signal-analysis/#incertidumbre-de-medida-isoiec-guide-98-3-gum-y-suplemento-1)

## [Percepción y audición](/phonometry/es/reference/theory/perception/)

La más larga de las seis: las líneas isofónicas, los modelos de patrón de
excitación y de enmascaramiento que hay detrás de la sonoridad y de la calidad
sonora, la cadena de transferencia de modulación del STI y la construcción por
importancia de banda y audibilidad del SII, y después la estadística de umbrales
de audición y el modelo de daño levantado sobre ella.

- [Líneas isofónicas (ISO 226:2023)](/phonometry/es/reference/theory/perception/#líneas-isofónicas-iso-2262023)
- [Prominencia tonal: TNR y PR (ECMA-418-1)](/phonometry/es/reference/theory/perception/#prominencia-tonal-tnr-y-pr-ecma-418-1)
- [Sonoridad de Zwicker (ISO 532-1)](/phonometry/es/reference/theory/perception/#sonoridad-de-zwicker-iso-532-1)
- [Modelos avanzados de sonoridad y calidad sonora](/phonometry/es/reference/theory/perception/#modelos-avanzados-de-sonoridad-y-calidad-sonora)
- [Transferencia de modulación y STI (IEC 60268-16)](/phonometry/es/reference/theory/perception/#transferencia-de-modulación-y-sti-iec-60268-16)
- [Índice de inteligibilidad del habla (ANSI S3.5)](/phonometry/es/reference/theory/perception/#índice-de-inteligibilidad-del-habla-ansi-s35)
- [Umbrales de audición y presbiacusia (ISO 389-7, ISO 7029)](/phonometry/es/reference/theory/perception/#umbrales-de-audición-y-presbiacusia-iso-389-7-iso-7029)
- [Pérdida auditiva inducida por ruido (ISO 1999)](/phonometry/es/reference/theory/perception/#pérdida-auditiva-inducida-por-ruido-iso-1999)

## [Salas y edificación](/phonometry/es/reference/theory/rooms-buildings/)

Dos secciones largas en lugar de muchas cortas: las curvas de criterio de
ANSI S12.2, y un tratamiento consolidado de las normas de sala y de edificación
que comparten la misma matemática de reverberación y de curvas de referencia.

- [Criterios de ruido de salas (ANSI S12.2)](/phonometry/es/reference/theory/rooms-buildings/#criterios-de-ruido-de-salas-ansi-s122)
- [Acústica de salas y edificación (ISO 18233, ISO 3382, ISO 16283, ISO 10140, EN 12354, ISO 12999, ISO 717, ISO 354)](/phonometry/es/reference/theory/rooms-buildings/#acústica-de-salas-y-edificación-iso-18233-iso-3382-iso-16283-iso-10140-en-12354-iso-12999-iso-717-iso-354)

## [Materiales y superficies](/phonometry/es/reference/theory/materials-surfaces/)

Las normas de caracterización y no los modelos de predicción: qué mide cada uno
de los coeficientes de dispersión y de difusión y por qué no hay que
intercambiarlos, cómo una medida in situ de un firme separa la reflexión en el
tiempo, y las definiciones que hay detrás de las magnitudes de absorción y de
impedancia de laboratorio.

- [Dispersión superficial y difusión (ISO 17497-1, ISO 17497-2)](/phonometry/es/reference/theory/materials-surfaces/#dispersión-superficial-y-difusión-iso-17497-1-iso-17497-2)
- [Absorción in situ de superficies de carretera (ISO 13472-1, ISO 13472-2)](/phonometry/es/reference/theory/materials-surfaces/#absorción-in-situ-de-superficies-de-carretera-iso-13472-1-iso-13472-2)
- [Caracterización de materiales acústicos (ISO 11654, ISO 9053-1/2, ISO 10534-1/2, ASTM E2611)](/phonometry/es/reference/theory/materials-surfaces/#caracterización-de-materiales-acústicos-iso-11654-iso-9053-12-iso-10534-12-astm-e2611)

## [Medio ambiente y transporte](/phonometry/es/reference/theory/environment-transport/)

Los descriptores y los términos de atenuación: cómo se construyen los
indicadores de ISO 1996-1, qué mide el criterio de prominencia de NT ACOU 112,
de dónde sale cada término de ISO 9613 y, archivadas aquí y no en dispositivos
porque la matemática es la misma, las deducciones de determinación de la
potencia acústica y la incertidumbre de exposición al ruido en el trabajo de
ISO 9612.

- [Descriptores ambientales (ISO 1996-1)](/phonometry/es/reference/theory/environment-transport/#descriptores-ambientales-iso-1996-1)
- [Prominencia de sonidos impulsivos (NT ACOU 112)](/phonometry/es/reference/theory/environment-transport/#prominencia-de-sonidos-impulsivos-nt-acou-112)
- [Propagación en exteriores y exposición al ruido en el trabajo (ISO 9613-1/2, ISO 9612)](/phonometry/es/reference/theory/environment-transport/#propagación-en-exteriores-y-exposición-al-ruido-en-el-trabajo-iso-9613-12-iso-9612)
- [Determinación de la potencia acústica (ISO 3744/3745/3746, ISO 3741, ISO 9614-2/3)](/phonometry/es/reference/theory/environment-transport/#determinación-de-la-potencia-acústica-iso-374437453746-iso-3741-iso-9614-23)

## [Vibración](/phonometry/es/reference/theory/vibration/)

La más corta: las ponderaciones y las medidas de dosis de vibración en humanos
con el modelo espinal de ISO 2631-5, más los resultados de movilidad puntual y
de eficiencia de radiación que usan las páginas de ruido estructural.

- [Vibración en humanos (ISO 8041-1, ISO 2631-1/2, ISO 5349-1/2, Directiva 2002/44/CE)](/phonometry/es/reference/theory/vibration/#vibración-en-humanos-iso-8041-1-iso-2631-12-iso-5349-12-directiva-200244ce)
- [Choques múltiples (ISO 2631-5)](/phonometry/es/reference/theory/vibration/#choques-múltiples-iso-2631-5)
- [Movilidades puntuales y eficiencia de radiación (Cremer 5, Hopkins 2.9)](/phonometry/es/reference/theory/vibration/#movilidades-puntuales-y-eficiencia-de-radiación-cremer-5-hopkins-29)
