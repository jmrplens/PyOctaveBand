← [Documentation index](../README.md)

# Support the project

phonometry is maintained by one person, and its real cost is not hosting or
tooling: it is the primary sources. Every metric in the library is implemented
from the governing standard's own text, and standards are paid documents that
go out of print, get superseded and occasionally resist every legitimate
acquisition route there is. This page lists the ways to help, from free to
concrete.

## Use it, and tell me when a number looks wrong

The cheapest support is using the library and reporting the number that does
not match your reference. A report that names a standard and a clause is the
most valuable issue this project receives: the
[errata register](../ERRATA.md) exists because sources get read closely, and
several of its entries began as a user's raised eyebrow.
[Open an issue](https://github.com/jmrplens/phonometry/issues) with what you
measured and what you expected.

## Sponsor

[GitHub Sponsors](https://github.com/sponsors/jmrplens) funds the project
directly, and the money goes where the cost is: buying the standards and
editions the implementations are written from. A one-off sponsorship that
names one of the documents below buys exactly that document, and the entry
leaves the table when it lands.

## The sources I could not obtain

Everything below stands between the library and work it would otherwise do.
Each entry has survived the whole acquisition effort: my own library, the
official stores, the open-access indexes and every other legitimate channel I
could find, re-checked on the date shown. Three routes help:

- **Fund the purchase.** A one-off sponsorship naming the document; where a
  link goes to an official store, the price is on that page, and for the rest
  name the item and I will quote it in the issue.
- **Verify against your licensed copy.** If your lab or employer already
  holds one, you can check the implementation and its reference values
  against your copy without sending me anything. The repository never
  redistributes documents; what lands in the tree are the derived numeric
  values, which is how the [conformance suite](../CONFORMANCE.md) is built.
- **Point me at a channel I missed.** A lending library, an author's own
  copy of a thesis, a proceedings archive that actually resolves.
  [Open an issue](https://github.com/jmrplens/phonometry/issues).

One route does not help: sending scans or PDFs of paywalled documents. I
cannot accept them, and the project's credibility rests on its sources being
clean.

| Source | What it would unlock | Last checked |
| :--- | :--- | :--- |
| [IEC 60268-16:2020 (Ed. 5)](https://webstore.iec.ch/en/publication/26771) | The STI edition in force. The library implements the numerically identical Ed. 4 chain and holds Ed. 5 store previews and Corrigendum 1:2025; the full text would let the four new annexes and the ten-page Annex M be verified instead of described | 2026-09-01 |
| [IEC 61252:2025](https://webstore.iec.ch/en/publication/68929) | The edition that superseded IEC 61252:1993+A1+A2 on 17 October 2025. The suite transcribes the 1993+A2 text for `sound_exposure()` and `lex_8h()`; the 2025 text is needed to review the delta | 2026-09-01 |
| Widmann (1992), *Ein Modell der psychoakustischen Lästigkeit von Schallen*, TUM doctoral thesis | The primary source of the psychoacoustic-annoyance model. The model is implemented from Fastl & Zwicker (2006); the thesis, 116 printed pages that predate electronic deposit, would let the register cite the origin at first hand | 2026-09-01 |
| Bowdler & Leventhall (eds.), *Wind Turbine Noise* (Multi-Science, 2011) | The standard reference behind the wind-turbine noise guide; only its book reviews circulate | 2026-09-01 |
| Crispin, Blasco, Ingelaere & Van Damme, *The vibration reduction index Kij: laboratory measurements versus predictions EN 12354-1* ([doi:10.1201/9781003078852-125](https://doi.org/10.1201/9781003078852-125)) | Measured junction Kij values as an independent oracle for the ISO 10848 and EN 12354-1 implementations | 2026-09-01 |
| Hopkins, *Sound insulation measurements with intensity techniques: flanking transmission* (IoA 1997, doi 10.25144/19134) | Intensity-measured flanking data beside the ISO 15186 implementation; the publisher's own link has been dead for years | 2026-09-01 |

| Recuero, *Ingeniería acústica* (Paraninfo) | Canonical Spanish textbook; print only, and it would enrich the terminology and worked examples of the Spanish guides | 2026-09-01 |
| Arau, *ABC de la acústica arquitectónica* (CEAC, 2007) | Spanish room-acoustics authority behind the Arau-Puchades formula the library implements; print only | 2026-09-01 |

Second-hand print copies count, through the same routes.
