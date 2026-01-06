import time

import numpy as np

from pyoctaveband import octavefilter


def benchmark_isolation(filter_type, target_freq=1000, fs=48000):
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    x = np.sin(2 * np.pi * target_freq * t)
    
    # Analyze with 1-octave bands
    spl, freq = octavefilter(x, fs, fraction=1, limits=[62, 16000], filter_type=filter_type)
    
    freq_arr = np.array(freq)
    closest_idx = np.argmin(np.abs(freq_arr - target_freq))
    peak_val = spl[closest_idx]
    
    # Calculate attenuation at +/- 1 and +/- 2 octaves (indices since fraction=1)
    results = {"peak": peak_val}
    
    if closest_idx - 1 >= 0:
        results["-1_oct"] = peak_val - spl[closest_idx - 1]
    if closest_idx + 1 < len(spl):
        results["+1_oct"] = peak_val - spl[closest_idx + 1]
    if closest_idx - 2 >= 0:
        results["-2_oct"] = peak_val - spl[closest_idx - 2]
    if closest_idx + 2 < len(spl):
        results["+2_oct"] = peak_val - spl[closest_idx + 2]
        
    return results

def benchmark_stability(filter_type, fs=48000):
    x = np.zeros(fs)
    x[0] = 1.0
    
    start_time = time.time()
    _, _, signals = octavefilter(x, fs, fraction=1, sigbands=True, filter_type=filter_type)
    exec_time = time.time() - start_time
    
    max_tail_energy = 0
    for band_sig in signals:
        tail = band_sig[-int(fs*0.1):]
        energy = np.sum(tail**2)
        max_tail_energy = max(max_tail_energy, energy)
        
    return max_tail_energy, exec_time

def main():
    filters = ["butter", "cheby1", "cheby2", "ellip", "bessel"]
    
    markdown = []
    markdown.append("# Filter Architecture Benchmark Report")
    markdown.append("\nThis report compares the performance and characteristics of the available filter types.")
    
    markdown.append("\n## 1. Spectral Isolation (at 1kHz)")
    header = (
        "| Filter Type | Peak SPL (dB) | Atten. -1 Oct (dB) | "
        "Atten. +1 Oct (dB) | Atten. -2 Oct (dB) | Atten. +2 Oct (dB) |"
    )
    markdown.append(header)
    markdown.append("|---|---|---|---|---|---|")
    
    for f in filters:
        iso = benchmark_isolation(f)
        row = (
            f"| {f} | {iso.get('peak',0):.2f} | {iso.get('-1_oct',0):.1f} | "
            f"{iso.get('+1_oct',0):.1f} | {iso.get('-2_oct',0):.1f} | "
            f"{iso.get('+2_oct',0):.1f} |"
        )
        markdown.append(row)

    markdown.append("\n## 2. Stability and Performance")
    markdown.append("| Filter Type | Max IR Tail Energy | Stability Status | Avg. Execution Time (s) |")
    markdown.append("|---|---|---|---|")
    
    for f in filters:
        tail_energy, exec_time = benchmark_stability(f)
        status = "✅ Stable" if tail_energy < 1e-6 else "⚠️ Ringing"
        markdown.append(f"| {f} | {tail_energy:.2e} | {status} | {exec_time:.4f} |")

    markdown.append("\n## 3. Analysis Summary")
    markdown.append("- **Butterworth:** Best compromise, maximally flat passband.")
    markdown.append("- **Chebyshev I:** Steeper roll-off than Butterworth but with passband ripple.")
    markdown.append("- **Chebyshev II:** Flat passband, ripple in the stopband.")
    markdown.append("- **Elliptic:** Steepest transition but ripples in both passband and stopband.")
    markdown.append("- **Bessel:** Best phase response and minimal ringing (group delay), but slowest roll-off.")

    with open("filter_benchmark_report.md", "w") as f:
        f.write("\n".join(markdown))
    
    print("Benchmark report generated: filter_benchmark_report.md")

if __name__ == "__main__":
    main()
