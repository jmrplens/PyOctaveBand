#  Copyright (c) 2026. Jose M. Requena-Plens
"""
Professional technical benchmark for PyOctaveBand.
Validates numerical precision, stability, phase linearity, and performance 
using High-Resolution Audio parameters (96kHz, 10s).
"""

import os
import platform
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

# High-Resolution Audio Parameters
FS_BENCH = 96000
DURATION_BENCH = 10.0
IMG_DIR = ".github/images/benchmark"
os.makedirs(IMG_DIR, exist_ok=True)

def get_cpu_info() -> str:
    """Attempt to get a readable CPU model name."""
    try:
        if platform.system() == "Linux":
            import subprocess
            # Use list of arguments instead of shell=True for security
            output = subprocess.check_output(["grep", "model name", "/proc/cpuinfo"]).decode()
            for line in output.splitlines():
                if "model name" in line:
                    return line.split(":")[1].strip()
        return platform.processor()
    except Exception:
        return "Unknown Processor"

def benchmark_isolation_and_precision(filter_type: str, target_freq: float = 1000, fs: int = FS_BENCH) -> Dict[str, float]:
    """Evaluate spectral isolation and numerical precision relative to theory."""
    t = np.linspace(0, DURATION_BENCH, int(fs * DURATION_BENCH), endpoint=False)
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
    
    if closest_idx + 1 < len(spl):
        results["+1_oct"] = float(peak_val - spl[closest_idx + 1])
        
    return results

def benchmark_advanced_metrics(filter_type: str, fs: int = FS_BENCH) -> Dict[str, float]:
    """Evaluate advanced metrics: Ripple, Phase, and Flatness."""
    bank = OctaveFilterBank(fs, fraction=1, limits=[500, 2000], filter_type=filter_type)
    
    idx_1k = np.argmin(np.abs(np.array(bank.freq) - 1000))
    f_low = bank.freq_d[idx_1k]
    f_high = bank.freq_u[idx_1k]
    f_center_low = 1000 * (10**(-0.3/2)) * 1.1 
    f_center_high = 1000 * (10**(0.3/2)) * 0.9
    
    fsd = fs / bank.factor[idx_1k]
    
    # 1. Ripple
    w, h = scipy_signal.sosfreqz(bank.sos[idx_1k], worN=8192, fs=fsd)
    central_mask = (w >= f_center_low) & (w <= f_center_high)
    mag_db = 20 * np.log10(np.abs(h[central_mask]) + 1e-12)
    ripple = np.max(mag_db) - np.min(mag_db)
    
    # 2. Group Delay
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        w_gd = np.linspace(f_low, f_high, 500)
        gd_total = np.zeros_like(w_gd)
        for section in bank.sos[idx_1k]:
            with np.errstate(divide='ignore', invalid='ignore'):
                _, gd_sec = scipy_signal.group_delay((section[:3], section[3:]), w=w_gd, fs=fsd)
                gd_total += gd_sec
    gd_std = np.std(gd_total) * 1000
    
    return {
        "ripple_db": float(ripple),
        "gd_std_ms": float(gd_std)
    }

def benchmark_stability_and_latency(filter_type: str, fs: int = FS_BENCH) -> Dict[str, float]:
    """Evaluate stability and peak latency."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, int(fs * DURATION_BENCH)))
    
    start_time = time.perf_counter()
    bank = OctaveFilterBank(fs, fraction=1, limits=[62, 16000], filter_type=filter_type)
    res = bank.filter(x, sigbands=True)
    _, freq, signals = res
    exec_time = time.perf_counter() - start_time
    
    idx_1k = np.argmin(np.abs(np.array(freq) - 1000))
    tail_len = int(fs * 0.1)
    # signals is a list of [channels, samples]
    tail = signals[idx_1k][0, -tail_len:]
    energy_floor = np.sum(tail**2) / tail_len
    
    impulse = np.zeros((1, 8192))
    impulse[0, 0] = 1.0
    ir = bank._filter_and_resample(impulse, idx_1k)
    peak_sample = np.argmax(np.abs(ir[0]))
    latency_ms = (peak_sample / fs) * 1000
        
    return {
        "tail_energy": float(energy_floor),
        "exec_time": exec_time,
        "latency_ms": float(latency_ms)
    }

def benchmark_crossover(fs: int = FS_BENCH) -> Dict[str, float]:
    """Specialized benchmark for Linkwitz-Riley crossover."""
    from pyoctaveband import linkwitz_riley
    fc = 1000
    impulse = np.zeros(fs)
    impulse[0] = 1.0
    lp, hp = linkwitz_riley(impulse, fs, freq=fc, order=4)
    
    w, h_lp = scipy_signal.freqz(lp, worN=16384, fs=fs)
    _, h_hp = scipy_signal.freqz(hp, worN=16384, fs=fs)
    h_sum = h_lp + h_hp
    
    mag_sum_db = 20 * np.log10(np.abs(h_sum) + 1e-12)
    mask = (w >= fc/2) & (w <= fc*2)
    flatness_error = np.max(np.abs(mag_sum_db[mask]))
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(w, 20 * np.log10(np.abs(h_lp) + 1e-9), label="Low Pass (LR4)")
    plt.semilogx(w, 20 * np.log10(np.abs(h_hp) + 1e-9), label="High Pass (LR4)")
    plt.semilogx(w, mag_sum_db, label="Sum (Flatness)", color="black", linestyle="--")
    plt.title("Linkwitz-Riley Crossover Verification (96kHz)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.xlim(100, 10000)
    plt.ylim(-60, 5)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, "benchmark_crossover.png"), dpi=150)
    plt.close()
    
    return {"flatness_error": float(flatness_error)}

def benchmark_multichannel_scaling(fs: int = FS_BENCH, max_channels: int = 16) -> List[Tuple[int, float]]:
    """Measure vectorization efficiency."""
    duration = DURATION_BENCH
    rng = np.random.default_rng(42)
    results = []
    
    bank = OctaveFilterBank(fs, fraction=3) 
    
    # Pure noise for multichannel benchmark
    noise_frame = rng.standard_normal(int(fs * duration))
    
    for n in [1, 2, 4, 8, 16]:
        if n > max_channels:
            break
        x = np.tile(noise_frame, (n, 1))
        
        # Warmup
        bank.filter(x)
        
        start = time.perf_counter()
        iters = 3 # Fewer iters due to long signal
        for _ in range(iters):
            bank.filter(x)
        avg_time = (time.perf_counter() - start) / iters
        results.append((n, avg_time))
        
    return results

def main() -> None:
    filters = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    fs = FS_BENCH
    
    print(f"Starting Professional Benchmark (fs={fs}Hz, dur={DURATION_BENCH}s)...")
    all_data = {}
    for f in filters:
        print(f"  Analyzing {f}...")
        all_data[f] = {
            "iso": benchmark_isolation_and_precision(f, fs=fs),
            "adv": benchmark_advanced_metrics(f, fs=fs),
            "st": benchmark_stability_and_latency(f, fs=fs)
        }
    
    print("  Analyzing Crossover...")
    xr_data = benchmark_crossover(fs=fs)
    print("  Measuring scaling...")
    scaling = benchmark_multichannel_scaling(fs=fs)
    
    # Graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(filters, [all_data[f]["iso"]["precision_error"] for f in filters], color="#1f77b4")
    ax1.set_title("Precision Error (dB)")
    ax1.set_yscale("log")
    ax2.bar(filters, [all_data[f]["iso"]["+1_oct"] for f in filters], color="#d62728")
    ax2.set_title("Isolation +1 Oct (dB)")
    plt.savefig(os.path.join(IMG_DIR, "benchmark_precision.png"))
    plt.close()

    ch = [s[0] for s in scaling]
    sp = [(scaling[0][1]*s[0])/s[1] for s in scaling]
    plt.figure(figsize=(8, 5))
    plt.plot(ch, sp, 'o-', color="#ff7f0e", label="Measured Speedup")
    plt.plot(ch, [1]*len(ch), '--', color="gray", alpha=0.5)
    plt.title("Multichannel Speedup (NumPy Vectorization)")
    plt.xlabel("Channels")
    plt.ylabel("Factor")
    plt.grid(True, linestyle=":")
    plt.savefig(os.path.join(IMG_DIR, "benchmark_performance.png"))
    plt.close()

    markdown = [
        "# PyOctaveBand: Technical Benchmark Report",
        f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## 1. Environment",
        f"- **OS:** {platform.system()} {platform.release()}",
        f"- **CPU:** {get_cpu_info()}",
        f"- **DSP:** {fs/1000}kHz, {DURATION_BENCH}s signal",
        "\n## 2. Crossover (Linkwitz-Riley)",
        "![Crossover](.github/images/benchmark/benchmark_crossover.png)",
        f"\n- **Flatness Error:** {xr_data['flatness_error']:.6f} dB (Target < 0.01)",
        "\n## 3. Precision & Isolation",
        "![Precision](.github/images/benchmark/benchmark_precision.png)",
        "\n| Type | Error (dB) | Isolation | Ripple | GD Std (ms) |",
        "|:---|:---:|:---:|:---:|:---:|"
    ]
    for f in filters:
        d = all_data[f]
        markdown.append(f"| {f} | {d['iso']['precision_error']:.2e} | {d['iso']['+1_oct']:.1f} dB | {d['adv']['ripple_db']:.4f} dB | {d['adv']['gd_std_ms']:.3f} |")

    markdown.append("\n## 4. Performance")
    markdown.append("![Performance](.github/images/benchmark/benchmark_performance.png)")
    markdown.append("\n| Channels | Exec Time (s) | Speedup |")
    markdown.append("|:---|:---:|:---:|")
    for n, t_exec in scaling:
        markdown.append(f"| {n} | {t_exec:.3f} | {(scaling[0][1]*n)/t_exec:.2f}x |")

    with open("filter_benchmark_report.md", "w") as f_out:
        f_out.write("\n".join(markdown))
    print("Done.")

if __name__ == "__main__":
    main()