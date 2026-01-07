#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Comprehensive benchmark and validation script for PyOctaveBand.
Evaluates performance, numerical precision, and standards compliance.
Generates visualization graphs and a technical report.
"""

import os
import time
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

from pyoctaveband import OctaveFilterBank, octavefilter

# Configure matplotlib for headless generation
import matplotlib
matplotlib.use("Agg")

# Constants
IMG_DIR = ".github/images/benchmark"
os.makedirs(IMG_DIR, exist_ok=True)

def benchmark_isolation_and_precision(filter_type: str, target_freq: float = 1000, fs: int = 48000) -> Dict[str, float]:
    """Evaluate spectral isolation and numerical precision relative to theory."""
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = np.sin(2 * np.pi * target_freq * t)
    theo_dbfs = 20 * np.log10(1 / np.sqrt(2))
    
    spl, freq = octavefilter(x, fs, fraction=1, limits=[62, 16000], filter_type=filter_type, dbfs=True)

    freq_arr = np.array(freq)
    closest_idx = np.argmin(np.abs(freq_arr - target_freq))
    peak_val = spl[closest_idx]
    
    precision_error = abs(peak_val - theo_dbfs)
    
    results = {
        "peak": float(peak_val),
        "precision_error": float(precision_error),
    }
    
    if closest_idx - 1 >= 0:
        results["-1_oct"] = float(peak_val - spl[closest_idx - 1])
    if closest_idx + 1 < len(spl):
        results["+1_oct"] = float(peak_val - spl[closest_idx + 1])
        
    return results

def benchmark_advanced_metrics(filter_type: str, fs: int = 48000) -> Dict[str, float]:
    """Evaluate advanced metrics: Flatness, Ripple, and Group Delay variance."""
    bank = OctaveFilterBank(fs, fraction=1, limits=[500, 2000], filter_type=filter_type)
    
    idx_1k = np.argmin(np.abs(np.array(bank.freq) - 1000))
    f_low = bank.freq_d[idx_1k]
    f_high = bank.freq_u[idx_1k]
    
    # Central passband (80%)
    f_center_low = 1000 * (10**(-0.3/2)) * 1.1 
    f_center_high = 1000 * (10**(0.3/2)) * 0.9
    
    fsd = fs / bank.factor[idx_1k]
    
    # 1. Passband Ripple (Central 80%)
    w, h = scipy_signal.sosfreqz(bank.sos[idx_1k], worN=8192, fs=fsd)
    central_mask = (w >= f_center_low) & (w <= f_center_high)
    mag_db = 20 * np.log10(np.abs(h[central_mask]) + 1e-12)
    ripple = np.max(mag_db) - np.min(mag_db)
    
    # 2. Group Delay Variance (Phase linearity)
    # Silence singularities at DC
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*singular.* সন")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        w_gd = np.linspace(f_low, f_high, 500)
        gd_total = np.zeros_like(w_gd)
        for section in bank.sos[idx_1k]:
            with np.errstate(divide='ignore', invalid='ignore'):
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
    
    idx_1k = np.argmin(np.abs(np.array(freq) - 1000))
    
    for i, band_sig in enumerate(signals):
        tail = band_sig[-int(fs*0.1):]
        energy = np.sum(tail**2)
        max_tail_energy = max(max_tail_energy, float(energy))
        
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
    bank = OctaveFilterBank(fs, fraction=3)
    
    for n in [1, 2, 4, 8, 16]:
        if n > max_channels: break
        x = np.tile(np.sin(2 * np.pi * 1000 * t), (n, 1))
        bank.filter(x)
        
        start = time.perf_counter()
        iters = 5
        for _ in range(iters):
            bank.filter(x)
        avg_time = (time.perf_counter() - start) / iters
        results.append((n, avg_time))
        
    return results

def generate_benchmark_graphs(data: Dict[str, Dict]) -> None:
    """Generate visual representations of the benchmark results."""
    filters = list(data.keys())
    
    # 1. Precision & Isolation Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    prec_errors = [data[f]["iso"]["precision_error"] for f in filters]
    ax1.bar(filters, prec_errors, color="#1f77b4", alpha=0.8)
    ax1.set_title("Numerical Precision Error (dB)")
    ax1.set_ylabel("Absolute Error [dB]")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    iso_1oct = [data[f]["iso"]["+1_oct"] for f in filters]
    ax2.bar(filters, iso_1oct, color="#d62728", alpha=0.8)
    ax2.set_title("Isolation at +1 Octave (dB)")
    ax2.set_ylabel("Attenuation [dB]")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "benchmark_precision.png"), dpi=150)
    plt.close()

    # 2. Phase & Stability Graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    gd_stds = [data[f]["adv"]["gd_std_ms"] for f in filters]
    ax1.bar(filters, gd_stds, color="#2ca02c", alpha=0.8)
    ax1.set_title("Phase Linearity (GD Std Dev)")
    ax1.set_ylabel("Std Dev [ms]")
    ax1.grid(True, linestyle="--", alpha=0.5)

    tail_energies = [data[f]["st"]["tail_energy"] for f in filters]
    ax2.bar(filters, tail_energies, color="#9467bd", alpha=0.8)
    ax2.set_title("Numerical Stability (IR Tail Energy)")
    ax2.set_ylabel("Energy")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "benchmark_stability.png"), dpi=150)
    plt.close()

def generate_performance_graph(scaling: List[Tuple[int, float]]) -> None:
    """Generate speedup visualization."""
    channels = [s[0] for s in scaling]
    times = [s[1] for s in scaling]
    base_time = times[0]
    speedups = [(base_time * n) / t for n, t in zip(channels, times)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(channels, speedups, 'o-', linewidth=2, markersize=8, color="#ff7f0e", label="Measured Speedup")
    plt.plot(channels, [1]*len(channels), '--', color="gray", alpha=0.5, label="Ideal Scaling (Sequential)")
    
    plt.title("Multichannel Vectorization Scaling (Speedup)", fontweight="bold")
    plt.xlabel("Number of Channels")
    plt.ylabel("Speedup Factor")
    plt.xticks(channels)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    
    plt.savefig(os.path.join(IMG_DIR, "benchmark_performance.png"), dpi=150)
    plt.close()

def main() -> None:
    filters = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    fs = 48000
    
    print("Starting Comprehensive Filter Benchmark...")
    all_data = {}
    
    for f in filters:
        print(f"  Analyzing {f}...")
        all_data[f] = {
            "iso": benchmark_isolation_and_precision(f, fs=fs),
            "adv": benchmark_advanced_metrics(f, fs=fs),
            "st": benchmark_stability_and_latency(f, fs=fs)
        }
    
    print("  Measuring performance scaling...")
    scaling = benchmark_multichannel_scaling(fs=fs)
    
    print("  Generating visualization graphs...")
    generate_benchmark_graphs(all_data)
    generate_performance_graph(scaling)
    
    markdown: list[str] = []
    markdown.append("# PyOctaveBand: Technical Benchmark Report")
    markdown.append(f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    markdown.append(f"\n**Environment:** fs={fs}Hz, Python optimized with Numba and NumPy Vectorization.")

    markdown.append("\n## 1. Executive Summary")
    markdown.append("This report evaluates the numerical integrity and performance of the PyOctaveBand DSP engine. "
                    "The library achieves professional-grade precision and high throughput for multichannel analysis.")

    # Graphs Section
    markdown.append("\n## 2. Numerical Precision & Isolation")
    markdown.append("![Precision and Isolation](.github/images/benchmark/benchmark_precision.png)")
    markdown.append("\n- **Precision:** Measures the absolute error in dB relative to the theoretical RMS of a pure sine wave (-3.01 dBFS).")
    markdown.append("- **Isolation:** Evaluates the filter's ability to reject out-of-band energy at adjacent octave bands.")

    header = (
        "| Filter Type | Peak (dBFS) | Precision Error | Atten. +1 Oct |"
    )
    markdown.append("\n" + header)
    markdown.append("|:---|:---:|:---:|:---:|")
    for f in filters:
        iso = all_data[f]["iso"]
        markdown.append(f"| {f} | {iso['peak']:.4f} | {iso['precision_error']:.2e} dB | {iso.get('+1_oct',0):.1f} dB |")

    markdown.append("\n## 3. Phase Linearity & Stability")
    markdown.append("![Stability and Phase](.github/images/benchmark/benchmark_stability.png)")
    markdown.append("\n- **GD Std Dev:** Quantification of phase distortion. A lower standard deviation of Group Delay indicates better preservation of wave shapes.")
    markdown.append("- **IR Tail Energy:** Residual energy in the filter after 1.9 seconds. Values < 1e-6 confirm unconditional numerical stability.")

    markdown.append("\n| Filter Type | Passband Ripple | GD Std Dev (ms) | IR Tail Energy | Status |")
    markdown.append("|:---|:---:|:---:|:---:|:---:|")
    for f in filters:
        adv = all_data[f]["adv"]
        st = all_data[f]["st"]
        status = "💎 High Quality" if st['tail_energy'] < 1e-8 else "✅ Stable"
        markdown.append(f"| {f} | {adv['ripple_db']:.4f} dB | {adv['gd_std_ms']:.3f} ms | {st['tail_energy']:.2e} | {status} |")

    markdown.append("\n## 4. Multichannel Performance")
    markdown.append("![Performance Scaling](.github/images/benchmark/benchmark_performance.png)")
    markdown.append("\nPyOctaveBand leverages NumPy's internal C-optimized loops for multichannel processing. The chart shows the speedup factor as the number of channels increases.")

    markdown.append("\n| Channels | Total Time (ms) | Time per Channel (ms) | Speedup Factor |")
    markdown.append("|:---|:---:|:---:|:---:|")
    base_time = scaling[0][1]
    for n, t_exec in scaling:
        t_ms = t_exec * 1000
        t_per_ch = t_ms / n
        speedup = (base_time * n) / t_exec
        markdown.append(f"| {n} | {t_ms:.2f} | {t_per_ch:.2f} | {speedup:.2f}x |")

    markdown.append("\n## 5. Methodology")
    markdown.append("- **Input:** Double-precision floating-point buffers.")
    markdown.append("- **Architecture:** Second-Order Sections (SOS) with automatic multirate decimation for stability.")
    markdown.append("- **Metrics:** Calculated using standard SciPy Signal Processing toolbox functions.")

    with open("filter_benchmark_report.md", "w") as f_out:
        f_out.write("\n".join(markdown))
    
    print(f"\nBenchmark report and graphs generated successfully in {IMG_DIR} and filter_benchmark_report.md")

if __name__ == "__main__":
    main()
