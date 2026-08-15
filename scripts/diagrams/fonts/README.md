# Vendored plate faces: Liberation Sans 2.1.5

These four TrueType files are the primary text faces of the experimental-setup
plates (`scripts/diagrams/`). The plates bake every label to glyph outlines at
generation time (`scripts/diagrams/outline.py`), so the committed artwork
depends on these exact binaries: a different build of the same face would move
every advance and rewrite every outline in the corpus.

Liberation Sans is metric-compatible with Arial/Helvetica, which is what the
plates' hand-tuned composition encodes -- the Spanish string table was reworded
to Helvetica-metric pixel widths, and Liberation reproduces those advances
(median width ratio 1.000 against the committed rendering). Glyphs the face
lacks fall back per-cluster to the matching DejaVu Sans face shipped inside the
pinned matplotlib wheel; no fontconfig lookup is ever consulted, every face is
loaded by file path.

## Provenance

Official upstream release 2.1.5 of the Liberation fonts, tarball
`liberation-fonts-ttf-2.1.5.tar.gz` attached to
<https://github.com/liberationfonts/liberation-fonts/releases/tag/2.1.5>
(tarball sha256
`7191c669bf38899f73a2094ed00f7b800553364f90e2637010a69c0e268f25d0`).
The Debian/Fedora packages rebuild the sources and produce different bytes;
only the upstream release binaries are used here.

| File | sha256 |
| --- | --- |
| `LiberationSans-Regular.ttf` | `76d04c18ea243f426b7de1f3ad208e927008f961dc5945e5aad352d0dfde8ee8` |
| `LiberationSans-Bold.ttf` | `788abee4c806d660e8aee46689dd8540cd4bb98da03dcc9d171ce3efd99a9173` |
| `LiberationSans-Italic.ttf` | `e5bae5c4cde31f22142753855f4f8fb86da6ff39955ed3c0a11248b0d16948b0` |
| `LiberationSans-BoldItalic.ttf` | `698da70fc191cc5f33ad4d6d3fe830fe4624b898ea2e3169955928b7c491f1ee` |
| `LICENSE` | `93fed46019c38bbe566b479d22148e2e8a1e85ada614accb0211c37b2c61c19b` |

`LICENSE` is the SIL Open Font License 1.1 from the same tarball, under which
the fonts are redistributed here (the `.github/brand/fonts/` precedent).

## Updating

Do not update casually: any change to these files is a wholesale regeneration
of all committed plates and is reviewed as such, exactly like a matplotlib or
FreeType bump (see the pin comment in `requirements-figures.txt`). If a bump
is ever needed, replace all four faces from an official upstream release,
refresh the hashes above, regenerate, and review the full corpus.
