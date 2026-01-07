#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Comprehensive benchmark and validation script for PyOctaveBand.
Evaluates performance, numerical precision, and standards compliance.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal as scipy_signal

from pyoctaveband import OctaveFilterBank, octavefilter


def benchmark_isolation_and_precision(filter_type: str, target_freq: float = 1000, fs: int = 48000) -> Dict[str, float]:
    """Evaluate spectral isolation and numerical precision relative to theory."""
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Pure sine wave with peak amplitude 1.0
    # Theoretical RMS = 1/sqrt(2) approx 0.7071
    # Theoretical dBFS = 20 * log10(1/sqrt(2)) approx -3.0103 dB
    x = np.sin(2 * np.pi * target_freq * t)
    theo_dbfs = 20 * np.log10(1 / np.sqrt(2))
    
    # Analyze with 1-octave bands in dBFS mode
    spl, freq = octavefilter(x, fs, fraction=1, limits=[62, 16000], filter_type=filter_type, dbfs=True)

    freq_arr = np.array(freq)
    closest_idx = np.argmin(np.abs(freq_arr - target_freq))
    peak_val = spl[closest_idx]
    
    precision_error = abs(peak_val - theo_dbfs)
    
    results = {
        "peak": float(peak_val),
        "precision_error": float(precision_error),
    }
    
    # Calculate attenuation at +/- 1 and +/- 2 octaves
    if closest_idx - 1 >= 0:
        results["-1_oct"] = float(peak_val - spl[closest_idx - 1])
    if closest_idx + 1 < len(spl):
        results["+1_oct"] = float(peak_val - spl[closest_idx + 1])
    if closest_idx - 2 >= 0:
        results["-2_oct"] = float(peak_val - spl[closest_idx - 2])
    if closest_idx + 2 < len(spl):
        results["+2_oct"] = float(peak_val - spl[closest_idx + 2])
        
    return results

def benchmark_stability_and_latency(filter_type: str, fs: int = 48000) -> Dict[str, float]:
    """Evaluate filter stability (ringing) and latency (group delay peak)."""
    x = np.zeros(fs)
    x[0] = 1.0
    
    start_time = time.perf_counter()
    res = octavefilter(x, fs, fraction=1, sigbands=True, filter_type=filter_type)
    _, freq, signals = res
    exec_time = time.perf_counter() - start_time
    
    max_tail_energy = 0.0
    max_latency_ms = 0.0
    
    # Analyze 1kHz band for latency specifically
    idx_1k = np.argmin(np.abs(np.array(freq) - 1000))
    
    for i, band_sig in enumerate(signals):
        # Stability: energy in the last 100ms
        tail = band_sig[-int(fs*0.1):]
        energy = np.sum(tail**2)
        max_tail_energy = max(max_tail_energy, float(energy))
        
        # Latency: peak of IR
        if i == idx_1k:
            peak_sample = np.argmax(np.abs(band_sig))
            max_latency_ms = (peak_sample / fs) * 1000
        
    return {
        "tail_energy": max_tail_energy,
        "exec_time": exec_time,
        "latency_ms": max_latency_ms
    }

def benchmark_multichannel_scaling(fs: int = 48000, max_channels: int = 16) -> List[Tuple[int, float]]:
    """Measure how performance scales with multiple channels (Vectorization test)."""
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    results = []
    
    # Pre-create bank to measure only filter execution
    bank = OctaveFilterBank(fs, fraction=3) # 1/3 octave is more intensive
    
    for n in [1, 2, 4, 8, 16]:
        if n > max_channels: break
        # Stack channels
        x = np.tile(np.sin(2 * np.pi * 1000 * t), (n, 1))
        
        # Warmup
        bank.filter(x)
        
        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            bank.filter(x)
        avg_time = (time.perf_counter() - start) / iters
        results.append((n, avg_time))
        
    return results

def benchmark_advanced_metrics(filter_type: str, fs: int = 48000) -> Dict[str, float]:
    """Evaluate advanced metrics: Flatness, Ripple, and Group Delay variance."""
    bank = OctaveFilterBank(fs, fraction=1, limits=[500, 2000], filter_type=filter_type)
    
    idx_1k = np.argmin(np.abs(np.array(bank.freq) - 1000))
    f_low = bank.freq_d[idx_1k]
    f_high = bank.freq_u[idx_1k]
    # Central passband (80%) to measure ripple without edge decay
    f_center_low = 1000 * (10**(-0.3/2)) * 1.1 
    f_center_high = 1000 * (10**(0.3/2)) * 0.9
    
    fsd = fs / bank.factor[idx_1k]
    
    # 1. Passband Ripple (Central 80%)
    w, h = scipy_signal.sosfreqz(bank.sos[idx_1k], worN=8192, fs=fsd)
    central_mask = (w >= f_center_low) & (w <= f_center_high)
    mag_db = 20 * np.log10(np.abs(h[central_mask]) + 1e-12)
    ripple = np.max(mag_db) - np.min(mag_db)
    
    # 2. Group Delay Variance (Phase linearity)
    # Avoid DC by starting frequency grid at 100Hz
    w_gd = np.linspace(f_low, f_high, 500)
    gd_total = np.zeros_like(w_gd)
    for section in bank.sos[idx_1k]:
        _, gd_sec = scipy_signal.group_delay((section[:3], section[3:]), w=w_gd, fs=fsd)
        gd_total += gd_sec
    
    gd_std = np.std(gd_total) * 1000  # in ms
    
    # 3. Total Power Flatness (Crossover Sum)
    f_grid = np.geomspace(500, 2000, 500)
    total_power = np.zeros_like(f_grid)
    for i in range(len(bank.freq)):
        fs_i = fs / bank.factor[i]
        _, h_i = scipy_signal.sosfreqz(bank.sos[i], worN=f_grid, fs=fs_i)
        total_power += np.abs(h_i)**2
    
    flatness_error = 10 * np.log10(np.max(total_power) / np.min(total_power))
    
    return {
        "ripple_db": float(ripple),
        "gd_std_ms": float(gd_std),
        "flatness_db": float(flatness_error)
    }

def main() -> None:
    filters = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    fs = 48000
    
    print("Starting Comprehensive Filter Benchmark...")
    
    markdown: list[str] = []
    markdown.append("# PyOctaveBand: Technical Benchmark Report")
    markdown.append(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    markdown.append(f"\n**Environment:** fs={fs}Hz, Python optimized with Numba.")

    # 1. Isolation & Precision
    markdown.append("\n## 1. Spectral Isolation & Numerical Precision")
    markdown.append("Evaluates how well the filter isolates a 1kHz tone and its numerical accuracy relative to the theoretical RMS (-3.01 dBFS).")
    header = (
        "| Filter Type | Peak (dBFS) | Precision Error (dB) | Atten. -1 Oct | Atten. +1 Oct | Atten. -2 Oct |"
    )
    markdown.append(header)
    markdown.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    
    for f in filters:
        iso = benchmark_isolation_and_precision(f, fs=fs)
        row = (
            f"| {f} | {iso['peak']:.4f} | {iso['precision_error']:.2e} | "
            f"{iso.get('-1_oct',0):.1f} dB | {iso.get('+1_oct',0):.1f} dB | "
            f"{iso.get('-2_oct',0):.1f} dB |"
        )
        markdown.append(row)

    # 2. Advanced Signal Theory Metrics
    markdown.append("\n## 2. Signal Theory & Quality Metrics")
    markdown.append("Validation of passband ripple (central 80%), phase linearity (GD Std), and total reconstruction flatness.")
    markdown.append("| Filter Type | Passband Ripple | GD Std Dev (ms) | Flatness Error | Recommended For |")
    markdown.append("|:---|:---:|:---:|:---:|:---:|")
    
    for f in filters:
        adv = benchmark_advanced_metrics(f, fs=fs)
        
        # Logic for recommendation
        if f == "butter": rec = "Standard Audio"
        elif f == "bessel": rec = "Transient Analysis"
        elif f in ["cheby2", "ellip"]: rec = "Out-of-band Rejection"
        else: rec = "High Selectivity"
            
        markdown.append(f"| {f} | {adv['ripple_db']:.4f} dB | {adv['gd_std_ms']:.3f} ms | {adv['flatness_db']:.2f} dB | {rec} |")

    # 3. Stability & Latency
    markdown.append("\n## 3. Stability, Latency & Speed")
    markdown.append("IR Tail Energy < 1e-6 indicates high numerical stability. Latency is measured at the 1kHz band peak.")
    markdown.append("| Filter Type | IR Tail Energy | Latency (ms) | Status | Exec Time (ms) |")
    markdown.append("|:---|:---:|:---:|:---:|:---:|")
    
    for f in filters:
        st = benchmark_stability_and_latency(f, fs=fs)
        status = "✅ Stable" if st['tail_energy'] < 1e-6 else "⚠️ Ringing"
        markdown.append(f"| {f} | {st['tail_energy']:.2e} | {st['latency_ms']:.2f} | {status} | {st['exec_time']*1000:.2f} |")

    # 4. Vectorization Scaling
    markdown.append("\n## 4. Multichannel Performance (Vectorization)")
    markdown.append("Measures the average execution time for a 1-second signal through a 1/3 Octave Filter Bank.")
    markdown.append("| Channels | Total Time (ms) | Time per Channel (ms) | Speedup Factor |")
    markdown.append("|:---|:---:|:---:|:---:|")
    
    scaling = benchmark_multichannel_scaling(fs=fs)
    base_time = scaling[0][1]
    for n, t_exec in scaling:
        t_ms = t_exec * 1000
        t_per_ch = t_ms / n
        speedup = (base_time * n) / t_exec
        markdown.append(f"| {n} | {t_ms:.2f} | {t_per_ch:.2f} | {speedup:.2f}x |")

    markdown.append("\n## 5. Architecture Summary")
    markdown.append("- **Butterworth:** Maximally flat passband. Standard for acoustic measurement.")
    markdown.append("- **Chebyshev I:** High selectivity, but introduces ripples in the passband.")
    markdown.append("- **Chebyshev II:** Flat passband with ripples in the stopband (excellent for isolation).")
    markdown.append("- **Elliptic:** Minimum transition width at the cost of ripple in both regions.")
    markdown.append("- **Bessel:** Linear phase response. Lowest latency and best transient preservation.")

    with open("filter_benchmark_report.md", "w") as f_out:
        f_out.write("\n".join(markdown))
    
    print("\nBenchmark report generated: filter_benchmark_report.md")

if __name__ == "__main__":
    main()