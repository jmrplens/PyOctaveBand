#  Copyright (c) 2026. Jose Manuel Requena Plens

"""Calibrated spectrogram and zoom FFT (Bendat & Piersol Ch. 12 / 11.5.4).

Clean-room oracles only: calibrated tone levels, the Parseval/COLA energy
identity, bin-by-bin consistency with the library's own Welch estimator,
and the Bendat & Piersol demodulate-decimate-DFT zoom chain reproduced
independently and matched to machine precision.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from phonometry import signals

REF_PRESSURE = 2e-5


def _tone(fs: float, duration: float, freq: float, amplitude: float) -> np.ndarray:
    t = np.arange(round(fs * duration)) / fs
    return amplitude * np.cos(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# Spectrogram: calibration
# ---------------------------------------------------------------------------


class TestSpectrogramCalibration:
    def test_tone_reads_its_spl_in_every_column(self) -> None:
        """A 1125 Hz tone at 80 dB SPL reads 80 dB in every column.

        fs = 48 kHz, nperseg = 4096 puts 1125 Hz exactly on bin 96, so
        the 'spectrum' column value at that bin is the tone's mean square
        A^2/2 = rms^2 and 10*lg(rms^2/p0^2) is the SPL. The only error is
        the Hann leakage of the negative-frequency image, ~1e-10 dB here.
        """
        fs = 48000.0
        rms = REF_PRESSURE * 10.0 ** (80.0 / 20.0)  # 80 dB SPL
        x = _tone(fs, 1.0, 1125.0, rms * np.sqrt(2.0))
        res = signals.spectrogram(x, fs, nperseg=4096, scaling="spectrum")
        bin_index = int(np.argmin(np.abs(res.frequencies - 1125.0)))
        assert res.frequencies[bin_index] == pytest.approx(1125.0)
        levels = 10.0 * np.log10(res.power[bin_index] / REF_PRESSURE**2)
        np.testing.assert_allclose(levels, 80.0, atol=1e-6)

    def test_tone_peaks_at_its_frequency(self) -> None:
        fs = 8192.0
        x = _tone(fs, 2.0, 1000.0, 1.0)
        res = signals.spectrogram(x, fs, nperseg=1024)
        ridge = res.frequencies[np.argmax(res.power, axis=0)]
        np.testing.assert_allclose(ridge, 1000.0)

    def test_column_average_is_the_welch_estimate(self) -> None:
        """Averaging the columns reproduces power_spectral_density."""
        rng = np.random.default_rng(20260721)
        fs = 8192.0
        x = rng.standard_normal(4 * 8192)
        for scaling in ("density", "spectrum"):
            res = signals.spectrogram(x, fs, nperseg=512, scaling=scaling)  # type: ignore[arg-type]
            psd = signals.power_spectral_density(x, fs, nperseg=512, scaling=scaling)  # type: ignore[arg-type]
            np.testing.assert_allclose(res.frequencies, psd.frequencies)
            np.testing.assert_allclose(np.mean(res.power, axis=1), psd.psd, rtol=1e-10)
            assert res.n_segments == psd.n_segments
            assert res.resolution_bandwidth == pytest.approx(psd.resolution_bandwidth)

    def test_parseval_energy_identity_with_cola_taper(self) -> None:
        """STFT power integrates to the record energy (Hann, 75 % overlap).

        With the squared Hann taper overlap-adding to the constant
        sum(w^2)/hop at 75 % overlap, summing the frequency-integrated
        'density' columns times the hop duration equals the time-domain
        energy of a burst fully covered by the segmentation - exactly
        (Parseval per segment plus the COLA identity, no approximation).
        """
        rng = np.random.default_rng(7)
        fs = 8192.0
        nperseg = 256
        n = 8192
        x = np.zeros(n)
        x[2048:4096] = rng.standard_normal(2048)  # interior burst
        res = signals.spectrogram(x, fs, nperseg=nperseg, overlap=0.75)
        df = res.frequencies[1] - res.frequencies[0]
        hop_seconds = res.hop / fs
        stft_energy = hop_seconds * float(np.sum(res.power)) * df
        time_energy = float(np.sum(x**2)) / fs
        assert stft_energy == pytest.approx(time_energy, rel=1e-12)

    def test_linear_chirp_ridge_tracks_the_instantaneous_frequency(self) -> None:
        """The spectrogram crest of a linear chirp follows f0 + beta*t."""
        fs = 8192.0
        duration = 4.0
        f0, f1 = 200.0, 2000.0
        beta = (f1 - f0) / duration
        t = np.arange(int(fs * duration)) / fs
        x = np.cos(2.0 * np.pi * (f0 * t + 0.5 * beta * t**2))
        res = signals.spectrogram(x, fs, nperseg=512)
        ridge = res.frequencies[np.argmax(res.power, axis=0)]
        expected = f0 + beta * res.times
        df = fs / 512
        assert np.max(np.abs(ridge - expected)) <= df

    def test_result_fields(self) -> None:
        fs = 1000.0
        res = signals.spectrogram(np.ones(1000), fs, nperseg=100, overlap=0.5)
        assert isinstance(res, signals.SpectrogramResult)
        assert res.power.shape == (51, res.n_segments)
        assert res.times.shape == (res.n_segments,)
        assert res.hop == 50
        assert res.time_resolution == pytest.approx(0.1)
        assert res.random_error == 1.0
        assert res.window == "hann"
        assert res.scaling == "density"


class TestSpectrogramValidation:
    def test_rejects_bad_overlap(self) -> None:
        x = np.ones(1024)
        with pytest.raises(ValueError, match=r"'overlap' must be in \[0, 1\)"):
            signals.spectrogram(x, 1000.0, overlap=1.0)

    def test_rejects_bad_scaling(self) -> None:
        x = np.ones(1024)
        with pytest.raises(ValueError, match=r"'scaling' must be 'density'"):
            signals.spectrogram(x, 1000.0, scaling="amplitude")  # type: ignore[arg-type]

    def test_rejects_short_signal(self) -> None:
        x = np.ones(8)
        with pytest.raises(ValueError, match=r"Signal too short for a spectrogram"):
            signals.spectrogram(x, 1000.0)

    def test_rejects_bad_nperseg(self) -> None:
        x = np.ones(1024)
        with pytest.raises(
            ValueError, match=r"'nperseg' must be between .* and the signal length"
        ):
            signals.spectrogram(x, 1000.0, nperseg=2048)


def test_silent_recording_renders_an_empty_spectrogram() -> None:
    """A silent recording is a legal input, so its figure must draw.

    Every cell is zero power and so -inf dB: the colour range has no
    strongest cell to anchor to, and taking the maximum of the finite
    levels used to reduce an empty array, stopping the plot with numpy's
    "zero-size array to reduction operation maximum". The fallback anchors
    the range at 0 dB and every cell sits below the 80 dB floor.
    """
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    res = signals.spectrogram(np.zeros(48000), fs=48000)
    assert float(np.max(res.power)) == 0.0
    ax = res.plot()
    image = ax.get_images()[0]
    assert image.get_clim() == (-80.0, 0.0)
    assert np.all(np.isneginf(np.ma.getdata(image.get_array())))
    plt.close("all")


# ---------------------------------------------------------------------------
# Zoom FFT: machine-precision oracles
# ---------------------------------------------------------------------------


def _demodulate_decimate_dft(
    x: np.ndarray, fs: float, f_min: float, decimation: int, k: int
) -> complex:
    """Bendat & Piersol Eqs. (11.128)-(11.130), computed literally.

    Demodulate by exp(-j*2*pi*f_min*n/fs), decimate by ``decimation``
    (plain subsampling; the band-limited tone needs no filter) and read
    bin ``k`` of the DFT of the decimated record.
    """
    n = np.arange(x.size)
    v = x * np.exp(-2j * np.pi * f_min * n / fs)
    v_dec = v[::decimation]
    m = np.arange(v_dec.size)
    return complex(np.sum(v_dec * np.exp(-2j * np.pi * k * m / v_dec.size)))


class TestZoomFFTExactness:
    def test_tone_amplitude_and_frequency_match_the_decimated_band_dft(
        self,
    ) -> None:
        """Machine-precision oracle per B&P 11.5.4.

        A 1100 Hz tone (on both the record's bin grid and the zoom grid)
        analysed over 1000-1256 Hz must read amplitude A and frequency
        1100 Hz exactly, and match the amplitude recovered by the
        demodulate-decimate-DFT chain of Eqs. (11.128)-(11.130) to
        machine precision.
        """
        fs = 8192.0
        n = 4096
        amp = 0.7
        t = np.arange(n) / fs
        x = amp * np.cos(2.0 * np.pi * 1100.0 * t + 0.3)
        res = signals.zoom_fft(
            x, fs, f_min=1000.0, f_max=1256.0, n_points=257, window="boxcar"
        )

        # Frequency recovered exactly on the zoom grid.
        peak = int(np.argmax(res.amplitude))
        assert res.frequencies[peak] == 1100.0

        # Amplitude recovered to machine precision.
        assert res.amplitude[peak] == pytest.approx(amp, rel=1e-12)

        # B&P chain: band 1000-1256 Hz -> B = 256 Hz, decimation
        # d = fs/(2B) = 16, tone at bin (1100-1000)/2 Hz = 50 of the
        # 256-point decimated DFT.
        v = _demodulate_decimate_dft(x, fs, 1000.0, 16, 50)
        amp_bp = 2.0 * abs(v) / (n / 16)
        assert res.amplitude[peak] == pytest.approx(amp_bp, rel=1e-12)

    def test_matches_the_full_length_dft_on_the_record_grid(self) -> None:
        """Chirp-Z samples equal np.fft.rfft bins where the grids meet."""
        rng = np.random.default_rng(11)
        fs = 8192.0
        n = 4096
        x = rng.standard_normal(n)
        # Zoom grid at the record resolution fs/n = 2 Hz: every zoom
        # point sits on an rfft bin.
        res = signals.zoom_fft(
            x, fs, f_min=500.0, f_max=756.0, n_points=129, window="boxcar"
        )
        bins = np.round(res.frequencies * n / fs).astype(int)
        full = np.fft.rfft(x)[bins] * 2.0 / n
        np.testing.assert_allclose(res.spectrum, full, rtol=1e-9, atol=1e-12)

    def test_resolves_two_tones_inside_one_base_fft_bin(self) -> None:
        """Two tones 3 Hz apart, one 1024-point-FFT bin (8 Hz) wide.

        With a 1 s record (1 Hz resolution) the zoom FFT separates the
        pair and, both tones sitting on the zoom grid where the Hann
        transform of the other is null, reads both amplitudes exactly.
        """
        fs = 8192.0
        t = np.arange(8192) / fs
        x = 0.8 * np.cos(2.0 * np.pi * 997.0 * t) + 0.5 * np.cos(
            2.0 * np.pi * 1000.0 * t
        )
        res = signals.zoom_fft(x, fs, f_min=980.0, f_max=1016.0, n_points=37)
        i1 = int(np.argmin(np.abs(res.frequencies - 997.0)))
        i2 = int(np.argmin(np.abs(res.frequencies - 1000.0)))
        assert res.amplitude[i1] == pytest.approx(0.8, rel=1e-6)
        assert res.amplitude[i2] == pytest.approx(0.5, rel=1e-6)
        # A valley separates the peaks: the pair is resolved.
        valley = res.amplitude[i1 + 1 : i2]
        assert np.max(valley) < 0.5

    def test_default_grid_matches_the_record_resolution(self) -> None:
        fs = 1000.0
        res = signals.zoom_fft(np.ones(2000), fs, f_min=100.0, f_max=200.0)
        assert res.bin_spacing == pytest.approx(fs / 2000.0)

    def test_power_is_the_tone_mean_square(self) -> None:
        fs = 8192.0
        x = _tone(fs, 0.5, 1024.0, 2.0)
        res = signals.zoom_fft(x, fs, f_min=1000.0, f_max=1048.0, window="boxcar")
        peak = int(np.argmax(res.power))
        assert res.frequencies[peak] == 1024.0
        assert res.power[peak] == pytest.approx(2.0, rel=1e-12)  # A^2/2

    def test_result_fields(self) -> None:
        res = signals.zoom_fft(
            np.ones(1000), 1000.0, f_min=100.0, f_max=200.0, n_points=51
        )
        assert isinstance(res, signals.ZoomFFTResult)
        assert res.n_points == 51
        assert res.frequencies[0] == 100.0
        assert res.frequencies[-1] == 200.0
        assert res.spectrum.shape == (51,)
        assert res.amplitude.shape == (51,)
        assert res.power.shape == (51,)
        assert res.window == "hann"
        # Hann ENBW = 1.5 * fs / N.
        assert res.resolution_bandwidth == pytest.approx(1.5)


class TestZoomFFTValidation:
    def test_rejects_inverted_band(self) -> None:
        x = np.ones(1024)
        with pytest.raises(
            ValueError, match=r"The zoom band must satisfy 0 <= f_min < f_max <= fs/2"
        ):
            signals.zoom_fft(x, 1000.0, f_min=300.0, f_max=200.0)

    def test_rejects_band_above_nyquist(self) -> None:
        x = np.ones(1024)
        with pytest.raises(
            ValueError, match=r"The zoom band must satisfy 0 <= f_min < f_max <= fs/2"
        ):
            signals.zoom_fft(x, 1000.0, f_min=100.0, f_max=600.0)

    def test_rejects_negative_f_min(self) -> None:
        x = np.ones(1024)
        with pytest.raises(
            ValueError, match=r"The zoom band must satisfy 0 <= f_min < f_max <= fs/2"
        ):
            signals.zoom_fft(x, 1000.0, f_min=-10.0, f_max=200.0)

    def test_rejects_single_point_grid(self) -> None:
        x = np.ones(1024)
        with pytest.raises(ValueError, match=r"'n_points' must be at least 2"):
            signals.zoom_fft(x, 1000.0, f_min=100.0, f_max=200.0, n_points=1)

    def test_rejects_short_signal(self) -> None:
        x = np.ones(8)
        with pytest.raises(ValueError, match=r"Signal too short for a zoom FFT"):
            signals.zoom_fft(x, 1000.0, f_min=100.0, f_max=200.0)

    def test_spectral_quantities_must_sit_on_the_zoom_grid(self) -> None:
        """A quantity off the grid is refused when built, not when read.

        Only ``power`` is drawn, and it surfaces as matplotlib's "x and y
        must have same first dimension" and two bare shapes, naming neither
        the field nor the result. ``spectrum`` and ``amplitude`` are drawn
        by nothing, so a dropped grid point renders the ordinary figure
        while the peak the caller reads moves onto a neighbour's frequency.
        An extra axis passes every count, so the ranks are pinned too.
        """
        good = signals.zoom_fft(
            np.ones(1000), 1000.0, f_min=100.0, f_max=200.0, n_points=51
        )
        per_point = "one value per grid point"
        cases = (
            ("frequencies", good.frequencies[:-1], rf"\(50\).*{per_point}"),
            ("spectrum", good.spectrum[:-1], rf"\(50\).*{per_point}"),
            (
                "spectrum",
                np.append(good.spectrum, good.spectrum[-1]),
                rf"\(52\).*{per_point}",
            ),
            ("amplitude", good.amplitude[:-1], rf"\(50\).*{per_point}"),
            (
                "amplitude",
                np.append(good.amplitude, good.amplitude[-1]),
                rf"\(52\).*{per_point}",
            ),
            ("power", good.power[:-1], rf"\(50\).*{per_point}"),
            ("power", np.append(good.power, good.power[-1]), rf"\(52\).*{per_point}"),
            ("amplitude", np.column_stack([good.amplitude] * 2), "must have one axis"),
            ("power", np.column_stack([good.power] * 2), "must have one axis"),
        )
        for field, value, fragment in cases:
            with pytest.raises(ValueError, match=rf"'{field}' {fragment}"):
                dataclasses.replace(good, **{field: value})


# ---------------------------------------------------------------------------
# DC / Nyquist calibration corners
# ---------------------------------------------------------------------------


class TestSingleComponentBins:
    def test_spectrogram_dc_reads_the_squared_offset(self) -> None:
        fs = 1000.0
        res = signals.spectrogram(
            np.full(1000, 3.0), fs, nperseg=100, scaling="spectrum"
        )
        np.testing.assert_allclose(res.power[0], 9.0, rtol=1e-12)

    def test_zoom_fft_dc_reads_the_offset(self) -> None:
        res = signals.zoom_fft(
            np.full(1000, 3.0), 1000.0, f_min=0.0, f_max=100.0, window="boxcar"
        )
        assert res.amplitude[0] == pytest.approx(3.0, rel=1e-12)
        assert res.power[0] == pytest.approx(9.0, rel=1e-12)
