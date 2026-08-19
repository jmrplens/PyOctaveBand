#  Copyright (c) 2026. Jose Manuel Requena Plens
"""EN/ES language option of the psychoacoustics ``.plot()`` renderers."""

from __future__ import annotations

import numpy as np
import pytest

from phonometry import psychoacoustics


def _result():
    fs = 48000
    t = np.arange(int(fs * 0.3)) / fs
    x = np.sqrt(2.0) * 0.02 * np.sin(2.0 * np.pi * 1000.0 * t)
    return psychoacoustics.loudness_zwicker(x, fs, stationary=True)


def test_spanish_labels_and_comma_decimals() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    res = _result()

    ax_en = res.plot(language="en")
    assert ax_en.get_xlabel() == "Critical-band rate $z$ [Bark]"

    ax_es = res.plot(language="es")
    # The symbol is drawn as mathematics and is the same symbol in both
    # languages; only the prose around it is translated.
    assert ax_es.get_xlabel() == "Razón de banda crítica $z$ [Bark]"
    assert "sonios" in ax_es.get_title()
    # Spanish uses a comma decimal separator in the annotated numbers.
    assert "," in ax_es.get_title()


def test_unknown_language_raises() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    result = _result()
    with pytest.raises(ValueError, match="Unknown language"):
        result.plot(language="xx")
