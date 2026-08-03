---
title: "Acerca de"
description: "Quién mantiene phonometry, cómo se construye cada métrica a partir del texto de la norma que la rige y se comprueba en CI, cómo informar de un error y cómo citar el software."
---

Esta página existe porque una librería que te pide que confíes en sus números
te debe el nombre de quien está detrás, el método del que salen esos números y
una forma de avisarme cuando alguno esté mal.

## Quién mantiene phonometry

Soy José Manuel Requena Plens. Escribí phonometry (publicada originalmente como
PyOctaveBand) y la mantengo: las implementaciones, la batería de tests, este
sitio de documentación y las publicaciones de versiones. No hay ninguna empresa
detrás ni ningún equipo, así que si algo de este sitio está mal, me toca a mí
corregirlo.

Puedes comprobar quién soy por cualquiera de estas vías:

- **ORCID:** [0000-0003-1250-6212](https://orcid.org/0000-0003-1250-6212)
- **Google Scholar:** [scholar.google.com/citations?user=9b0kPaUAAAAJ](https://scholar.google.com/citations?user=9b0kPaUAAAAJ)
- **ResearchGate:** [Jose Requena Plens](https://www.researchgate.net/profile/Jose-Requena-Plens-2)
- **GitHub:** [@jmrplens](https://github.com/jmrplens)
- **Sitio personal:** [jmrp.io](https://jmrp.io)

La misma identidad está publicada también en
[MathWorks File Exchange](https://www.mathworks.com/matlabcentral/profile/authors/5890853),
[LinkedIn](https://www.linkedin.com/in/jmrplens),
[Mastodon](https://mstdn.jmrp.io/@jmrplens) y
[Keyoxide](https://keyoxide.org/0A993B268654DBBA52B7E8D3FCF653391E2C91FC),
donde figura la clave OpenPGP con la que firmo mi trabajo.

Mi formación es de acústica y de tratamiento de señal, y está registrada en
lugar de autodescrita: soy Ingeniero de Telecomunicación especializado en
Sonido e Imagen por la Universidad de Alicante (2011-2018), cursé el Máster en
Ingeniería Acústica de la Universitat Politècnica de València (2018-2019) y
trabajé como investigador en acústica en la UPV de 2020 a 2023, donde publiqué
sobre metasuperficies acústicas, difusores de sonido y predicción de campos
acústicos; los artículos están listados en los perfiles de Google Scholar y
ORCID de arriba. Hoy trabajo en I+D industrial como ingeniero de firmware y
software.

La librería nació de la parte de medición de ese trabajo: necesitaba una y
otra vez niveles por bandas de octava y magnitudes de sonómetro que pudiera
defender frente a una tabla de tolerancias, así que los construí bien una sola
vez en lugar de volver a deducirlos en cada proyecto.

## Cómo construyo la librería

El método es la razón de ser del proyecto, y es deliberadamente estrecho:

1. **Implemento cada métrica a partir del texto de la norma que la rige**,
   apartado a apartado, y no a partir de una descripción secundaria ni de otra
   implementación. Cuando una fórmula es ambigua, la ambigüedad se resuelve con
   el propio documento o con la física, y la lectura que he elegido queda
   escrita.
2. **Transcribo a la batería de tests los valores de referencia de la propia
   norma.** Cuando una norma publica un ejemplo resuelto, una tabla de
   tolerancias o un conjunto de respuestas nominales (los valores de
   ponderación de la Tabla 3 de IEC 61672-1, las respuestas a ráfagas tonales
   de su Tabla 4, los límites de filtro de la Tabla 1 de IEC 61260-1, los
   contornos del Anexo B de ISO 226 y demás), esos números pasan a ser los
   valores esperados de tests reales.
3. **CI los exige en cada cambio.** Una regresión que saque un valor calculado
   fuera del límite de aceptación de la norma tumba la compilación, de modo que
   la afirmación no puede dejar de ser cierta en silencio entre versiones.

No todas las normas publican un ejemplo numérico. Cuando no lo hacen, la
implementación se ancla a las expresiones en forma cerrada del texto normativo
y se fija con un caso sintetizado a un resultado conocido, lo que es una
garantía más débil que un ejemplo resuelto transcrito y así se etiqueta en el
informe. Prefiero decirlo con claridad antes que redondearlo al alza.

Los números son públicos. El
[informe de conformidad](/phonometry/es/reference/conformance/) recoge, para
cada comprobación, la norma y el apartado, el valor esperado normativo, el
valor que calcula realmente la librería, la diferencia y un veredicto de apto o
no apto; CI lo regenera en cada pull request, así que no puede desviarse del
código. [Por qué phonometry](/phonometry/es/start/why-phonometry/) recorre
el método sobre un caso resuelto concreto (la ponderación temporal según
IEC 61672-1:2013) si quieres verlo aplicado de principio a fin antes de
confiar en él de forma general.

## Cómo informar de un error

Infórmalo, por favor. Un número equivocado del que nadie me avisa se queda
equivocado.

Abre una incidencia en
[github.com/jmrplens/phonometry/issues](https://github.com/jmrplens/phonometry/issues).
El informe más útil me indica la norma y el apartado contra el que estás
comprobando, el valor esperado, el valor que ha dado phonometry y un fragmento
de código breve que lo reproduzca. Si no puedes compartir la señal, con los
parámetros suele bastarme para reconstruir un caso.

Si el gestor de incidencias no es una opción para ti, escríbeme directamente a
[mail@jmrp.io](mailto:mail@jmrp.io); aun así la incidencia es el mejor
canal, porque el informe y la corrección quedan públicos junto al código al
que afectan.

Eso incluye los errores que encuentres en las propias normas. Volver a deducir
las fórmulas y recalcular los ejemplos resueltos a partir de los documentos
originales saca a la luz de vez en cuando defectos en las propias fuentes:
erratas de imprenta, ejemplos resueltos que contradicen su propio texto
normativo, redacciones ambiguas. No los sorteo en silencio. Cada caso
confirmado queda registrado en el
[registro de erratas](https://github.com/jmrplens/phonometry/blob/main/docs/ERRATA.md)
con la localización, la evidencia, la lectura que implementa la librería y si
se ha comunicado al organismo emisor. El registro cubre por igual normas (ISO,
IEC, EN), documentos de orientación e informes técnicos, libros de texto y
artículos de revista. Si no estás de acuerdo con alguna de esas lecturas, esa
es justo la clase de incidencia que quiero recibir.

## Cómo citar este software

Si phonometry ha contribuido a un trabajo publicado, cítalo, por favor. El
registro archivado y su DOI están en Zenodo:

**[doi.org/10.5281/zenodo.21215280](https://doi.org/10.5281/zenodo.21215280)**

Ese es el DOI de concepto: siempre resuelve a la versión archivada más
reciente, y cada versión tiene además su propio DOI en ese mismo registro. Cita
la versión que hayas ejecutado realmente.

APA:

> Requena-Plens, J. M. (2026). *phonometry: acoustic measurement toolkit for
> Python* (Versión 3.3.0) [Software].
> https://doi.org/10.5281/zenodo.21215280

BibTeX:

```bibtex
@software{requenaplens_phonometry,
  author  = {Requena-Plens, Jos{\'e} M.},
  title   = {phonometry: acoustic measurement toolkit for Python
             (formerly PyOctaveBand)},
  year    = {2026},
  version = {3.3.0},
  doi     = {10.5281/zenodo.21215280},
  url     = {https://jmrplens.github.io/phonometry/},
  license = {MIT}
}
```

Ambas entradas están derivadas de
[`CITATION.cff`](https://github.com/jmrplens/phonometry/blob/main/CITATION.cff)
en el repositorio, que son los metadatos autorizados y el fichero que leen
GitHub y Zenodo. Si usas un gestor de referencias, importa ese fichero en lugar
de copiar el bloque de arriba, y ajusta la versión y el año a la publicación
que hayas utilizado.

## Licencia

phonometry se distribuye bajo la
[licencia MIT](https://github.com/jmrplens/phonometry/blob/main/LICENSE), de
modo que puedes usarla en trabajos comerciales y académicos, modificarla y
redistribuirla, siempre que el aviso de copyright y el texto de la licencia
viajen con ella. El software se entrega sin garantía: está verificado frente a
las normas que implementa, tal como se describe más arriba, pero no es un
instrumento calibrado y no tiene ninguna acreditación. Decidir si un resultado
es adecuado para tu propósito sigue siendo cosa tuya.

Las normas en sí tienen los derechos de autor de sus organismos emisores y no
se redistribuyen aquí. Esta documentación cita apartados y valores de
referencia en la medida necesaria para explicar y verificar las
implementaciones; no sustituye a la compra de los documentos.
