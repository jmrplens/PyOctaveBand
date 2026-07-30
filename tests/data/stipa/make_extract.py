#  Copyright (c) 2026. Jose M. Requena-Plens
"""Build ``stipa_certified_extract.zip`` from a local copy of the stipa.info
IEC 60268-16 verification test signals.

The certified bench is 49 mono 48 kHz WAV files, 133 MB of PCM that does not
deflate (the signals are near full scale and broadband). Committing it as-is
is out of the question, so this script writes a **lossless re-encoding of a
27-signal subset**: each signal is stored as the k-th order difference of its
int16 samples, LZMA-compressed inside a zip, together with a manifest holding
the SHA-256 of the original sample bytes. Decoding is ``k`` cumulative sums,
and the test suite checks the digest, so what the tests analyse is bit-exact
with what stipa.info publishes.

Run it against a local copy of the full set (see ``README.md`` for the
selection rationale)::

    python tests/data/stipa/make_extract.py /path/to/stipa-verification
"""

from __future__ import annotations

import hashlib
import json
import lzma
import pathlib
import sys
import zipfile

import numpy as np
from scipy.io import wavfile

#: Signals kept in the committed extract, per verification suite. The
#: rationale for each choice is documented in ``README.md``.
SELECTION: tuple[str, ...] = (
    # Annex C.3.2 - direct-method modulation depth: the null case, the worst
    # measured STI deviation and the top of the dynamic range.
    "Annex C.3.2/STIPA-sinecarrier-M=0.wav",
    "Annex C.3.2/STIPA-sinecarrier-M=0.9.wav",
    "Annex C.3.2/STIPA-sinecarrier-M=1.wav",
    # Annex C.3.3 - indirect method: the complete suite (it compresses well).
    "Annex C.3.3/STIPA-expdecay-RT60=0.125.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=0.25.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=0.5.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=1.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=2.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=4.wav",
    "Annex C.3.3/STIPA-expdecay-RT60=8.wav",
    # Annex C.4.2 - filter-bank slope: the complete suite (two pure tones per
    # file, so the encoded form is a few kilobytes each).
    "Annex C.4.2/Filtertest_highslope 125.wav",
    "Annex C.4.2/Filtertest_highslope 250.wav",
    "Annex C.4.2/Filtertest_highslope 500.wav",
    "Annex C.4.2/Filtertest_highslope 1000.wav",
    "Annex C.4.2/Filtertest_highslope 2000.wav",
    "Annex C.4.2/Filtertest_highslope 4000.wav",
    "Annex C.4.2/Filtertest_highslope 8000.wav",
    "Annex C.4.2/Filtertest_lowslope 125.wav",
    "Annex C.4.2/Filtertest_lowslope 250.wav",
    "Annex C.4.2/Filtertest_lowslope 500.wav",
    "Annex C.4.2/Filtertest_lowslope 1000.wav",
    "Annex C.4.2/Filtertest_lowslope 2000.wav",
    "Annex C.4.2/Filtertest_lowslope 4000.wav",
    "Annex C.4.2/Filtertest_lowslope 8000.wav",
    # Annex A.2.2 - weighting/redundancy factors: the lowest and the
    # highest-STI band pair.
    "Annex A.2.2 - weight factor test/STIPA-sine-pair[125+250]STI=0.13.wav",
    "Annex A.2.2 - weight factor test/STIPA-sine-pair[1000+2000]STI=0.53.wav",
    # Annex A.3.1.2 - filter-bank phase distortion: the worst-case point of
    # the normative TI = 0,1 .. 0,9 range.
    (
        "Annex A.3.1.2 - filter bank phase test/"
        "STIPA-sine-edge-carriers-TI=0.9[m=0.94065].wav"
    ),
)

#: Difference orders tried when encoding; the smallest output wins.
_ORDERS = (0, 1, 2, 3, 4)

#: The whole bench is 48 kHz mono; anything else is the wrong download.
_FS = 48000

#: Fixed member timestamp, so rebuilding the archive is byte-reproducible
#: (the zip format would otherwise stamp the current time into every entry).
_EPOCH = (1980, 1, 1, 0, 0, 0)


def _member(name: str) -> zipfile.ZipInfo:
    """A deterministic archive entry for ``name``."""
    info = zipfile.ZipInfo(name, date_time=_EPOCH)
    info.compress_type = zipfile.ZIP_LZMA
    info.external_attr = 0o644 << 16
    return info


def encode(samples: np.ndarray, order: int) -> bytes:
    """Return the ``order``-th difference of ``samples`` as little-endian i4."""
    d = samples.astype(np.int64)
    for _ in range(order):
        d = np.diff(d, prepend=np.int64(0))
    return d.astype("<i4").tobytes()


def best_order(samples: np.ndarray) -> int:
    """Return the difference order that compresses ``samples`` smallest."""
    sizes = {
        order: len(lzma.compress(encode(samples, order), preset=6))
        for order in _ORDERS
    }
    return min(sizes, key=lambda order: sizes[order])


def build(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Write the LZMA-compressed extract of :data:`SELECTION` to ``destination``.

    The archive is built beside ``destination`` and moved into place only once
    it has closed cleanly, so a missing or malformed source file cannot leave
    the committed extract truncated.
    """
    staging = destination.with_suffix(destination.suffix + ".tmp")
    try:
        _write(source, staging)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    staging.replace(destination)
    print(f"{destination}: {len(SELECTION)} signals, {destination.stat().st_size} bytes")


def _write(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Build the archive at ``destination`` (see :func:`build`)."""
    entries = []
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_LZMA
    ) as archive:
        for relative in SELECTION:
            path = source / relative
            fs, samples = wavfile.read(path)
            if fs != _FS:
                raise SystemExit(f"{relative}: expected {_FS} Hz, got {fs}")
            if samples.dtype != np.int16 or samples.ndim != 1:
                raise SystemExit(f"{relative}: expected 16-bit mono PCM")
            order = best_order(samples)
            member = f"{relative}.i4"
            archive.writestr(_member(member), encode(samples, order))
            entries.append(
                {
                    "path": relative,
                    "member": member,
                    "fs": int(fs),
                    "samples": int(samples.size),
                    "order": order,
                    "sha256": hashlib.sha256(
                        samples.astype("<i2").tobytes()
                    ).hexdigest(),
                }
            )
        archive.writestr(
            _member("manifest.json"),
            json.dumps(
                {
                    "format": "phonometry-stipa-extract/1",
                    "source": (
                        "IEC 60268-16 STIPA verification test signals, "
                        "Embedded Acoustics BV (stipa.info)"
                    ),
                    "encoding": (
                        "int16 PCM, stored as the n-th order difference in "
                        "little-endian int32; decode with n cumulative sums"
                    ),
                    "signals": entries,
                },
                indent=1,
            ),
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <stipa-verification-directory>")
    build(
        pathlib.Path(sys.argv[1]),
        pathlib.Path(__file__).with_name("stipa_certified_extract.zip"),
    )
