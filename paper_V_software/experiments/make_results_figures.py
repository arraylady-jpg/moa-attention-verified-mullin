import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------- Figure 1: CPU thread-count crossover ----------
cpu_rows = list(csv.DictReader(open("real_results/results_paper1_delta_cpu_21181232.csv")))

by_n = {}
for r in cpu_rows:
    n = int(r["N"])
    by_n.setdefault(n, {})[int(r["threads"])] = float(r["seconds"])

fig, ax = plt.subplots(figsize=(7, 5))
ns = sorted(by_n)
cmap = plt.cm.viridis(np.linspace(0, 1, len(ns)))
for color, n in zip(cmap, ns):
    threads = sorted(by_n[n])
    times = [by_n[n][t] for t in threads]
    ax.loglog(threads, times, marker="o", color=color, label=f"N={n}")

ax.set_xlabel("threads")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper I forward pass, Delta CPU (job 21181232)\nOptimal thread count depends on N")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_cpu_threads.pdf")
print("Saved fig_paper1_cpu_threads.pdf")

# ---------- Figure 2: GPU scaling with O(n) and O(n^2) reference slopes ----------
gpu_rows = list(csv.DictReader(open("real_results/results_paper1_gpu_delta_gpuA100x4_21181233.csv")))
gpu_ns = [int(r["N"]) for r in gpu_rows]
gpu_times = [float(r["seconds"]) for r in gpu_rows]

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(gpu_ns, gpu_times, marker="o", color="#993C1D", linewidth=2, markersize=8, label="Measured (A100)")

# Reference slopes anchored at the first point
n0, t0 = gpu_ns[0], gpu_times[0]
ref_n = np.array(gpu_ns, dtype=float)
ref_o_n = t0 * (ref_n / n0)
ref_o_n2 = t0 * (ref_n / n0) ** 2
ax.loglog(ref_n, ref_o_n, "--", color="gray", alpha=0.6, label="O(n) reference")
ax.loglog(ref_n, ref_o_n2, ":", color="gray", alpha=0.8, label="O(n^2) reference")

ax.set_xlabel("N")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper I forward pass, Delta A100 (job 21181233)\nMeasured growth vs. O(n) and O(n^2) references")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_gpu_scaling.pdf")
print("Saved fig_paper1_gpu_scaling.pdf")

# ---------- Figure 3: CORRECTED A100 vs H200, post-fix real data ----------
import math

a100_rows = list(csv.DictReader(open("real_results/results_paper1_gpu_delta_gpuA100x4_21202354.csv")))
h200_rows = list(csv.DictReader(open("real_results/results_paper1_gpu_delta_gpuH200x8_21202355.csv")))

a100_ns = [int(r["N"]) for r in a100_rows]
a100_times = [float(r["seconds"]) for r in a100_rows]
h200_ns = [int(r["N"]) for r in h200_rows]
h200_times = [float(r["seconds"]) for r in h200_rows]

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(a100_ns, a100_times, marker="o", color="#993C1D", linewidth=2, markersize=7, label="A100 (fixed, job 21202354)")
ax.loglog(h200_ns, h200_times, marker="s", color="#0F6E56", linewidth=2, markersize=7, label="H200 (job 21202355)")

n0, t0 = a100_ns[0], a100_times[0]
ref_n = np.array(a100_ns, dtype=float)
ref_o_n2 = t0 * (ref_n / n0) ** 2
ax.loglog(ref_n, ref_o_n2, ":", color="gray", alpha=0.7, label="O(n^2) reference")

ax.set_xlabel("N")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper I forward pass, post-fix real GPU data\nA100 vs H200, both converge toward O(n^2)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_gpu_fixed_comparison.pdf")
print("Saved fig_paper1_gpu_fixed_comparison.pdf")

# ---------- Figure 4: local scaling exponent vs N, A100 and H200 ----------
a100_ns_full = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
a100_times_full = [0.000242, 0.000383, 0.000788, 0.002899, 0.008808, 0.031952, 0.116893, 0.457188]
h200_ns_full = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
h200_times_full = [0.000166, 0.000235, 0.000450, 0.001759, 0.005421, 0.027075, 0.104600, 0.431261]

def local_exponents(ns, times):
    mids, exps = [], []
    for i in range(1, len(ns)):
        ratio = times[i] / times[i-1]
        exps.append(math.log2(ratio))
        mids.append((ns[i] + ns[i-1]) / 2)  # midpoint for x-axis placement
    return mids, exps

a100_mid, a100_exp = local_exponents(a100_ns_full, a100_times_full)
h200_mid, h200_exp = local_exponents(h200_ns_full, h200_times_full)

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(a100_ns_full[1:], a100_exp, marker="o", color="#993C1D", linewidth=2, markersize=8, label="A100 (job 21202354)")
ax.semilogx(h200_ns_full[1:], h200_exp, marker="s", color="#0F6E56", linewidth=2, markersize=8, label="H200 (job 21202355)")
ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.5, label="O(n^2) target (exponent = 2)")

ax.set_xlabel("N (right edge of each doubling interval)")
ax.set_ylabel("local scaling exponent")
ax.set_title("Paper I forward pass: local exponent converges to 2\nDirect visual confirmation the timing fix works")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(0, 2.5)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_local_exponent.pdf")
print("Saved fig_paper1_local_exponent.pdf")

# ---------- Figure 5: DRAM traffic, predicted vs measured ----------
categories = ["Read", "Write", "Total"]
predicted = [6.291, 69.206, 75.497]
measured = [9.045, 142.333, 151.378]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
bars1 = ax.bar(x - width/2, predicted, width, label="Predicted (naive DNF count)", color="#5DCAA5")
bars2 = ax.bar(x + width/2, measured, width, label="Measured (ncu, real A100)", color="#993C1D")

for bar, val in zip(bars1, predicted):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}", ha="center", fontsize=9)
for bar, val in zip(bars2, measured):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f"{val:.1f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("DRAM traffic (MB)")
ax.set_title("Paper I forward pass, N=2048: predicted vs measured DRAM traffic\nWrites hit hardest (2.06x) -- consistent with ncu's coalescing diagnostics")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_dram_traffic.pdf")
print("Saved fig_paper1_dram_traffic.pdf")

# ---------- Figure 6: occupancy comparison, N=64 vs N=2048 (corrected finding) ----------
categories = ["Waves per SM\n(x10 for scale)", "Achieved occupancy (%)", "% of theoretical\nceiling reached"]
n64_vals = [0.17*10, 7.407128, 16.9]
n2048_vals = [5.42*10, 41.731774, 95.4]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(7.5, 5))
bars1 = ax.bar(x - width/2, n64_vals, width, label="N=64 (warmup kernel)", color="#C97B4A")
bars2 = ax.bar(x + width/2, n2048_vals, width, label="N=2048 (the actual timed run)", color="#2B6E8C")

for bar, val, orig in zip(bars1, n64_vals, [0.17, 7.41, 16.9]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{orig:.2f}", ha="center", fontsize=9)
for bar, val, orig in zip(bars2, n2048_vals, [5.42, 41.73, 95.4]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f"{orig:.2f}", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("value (waves scaled x10 for visibility)")
ax.set_title("GPU occupancy: N=64 (warmup) is severely under-saturated,\nN=2048 (the actual measurement) is not")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_occupancy_correction.pdf")
print("Saved fig_paper1_occupancy_correction.pdf")

# ---------- Figure 7: wall-clock vs device-timer cross-validation ----------
fig, ax = plt.subplots(figsize=(6, 4.5))
device_time = 0.026406
host_overhead = 0.031952 - 0.026406
ax.barh(["N=2048\nwall time"], [device_time], color="#5DCAA5", label="Device time (ncu gpu__time_duration)")
ax.barh(["N=2048\nwall time"], [host_overhead], left=[device_time], color="#C97B4A", label="Host-side overhead (malloc, transfer setup, etc.)")
ax.set_xlabel("time (s)")
ax.set_title("Wall clock (0.032s) vs GPU device timer (0.026s) at N=2048\n17.4% of measured wall time is host-side, not GPU compute")
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, 0.036)
for i, (d, h) in enumerate([(device_time, host_overhead)]):
    ax.text(d/2, i, f"{d:.4f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax.text(d + h/2, i, f"{h:.4f}s", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
fig.tight_layout()
fig.savefig("results_paper/fig_paper1_device_vs_wallclock.pdf")
print("Saved fig_paper1_device_vs_wallclock.pdf")

# ---------- Figure 8 (Paper II): fused vs naive forward+backward, N=4096 trend ----------
import csv as _csv

_forward = {}
with open("real_results/results_paper1_delta_cpu_21181232.csv") as f:
    for r in _csv.DictReader(f):
        _forward[(int(r["N"]), int(r["threads"]))] = float(r["seconds"])

_backward, _fused = {}, {}
with open("real_results/results_paper2_delta_cpu_21226440.csv") as f:
    for r in _csv.DictReader(f):
        key = (int(r["N"]), int(r["threads"]))
        if r["impl"] == "backward":
            _backward[key] = float(r["seconds"])
        else:
            _fused[key] = float(r["seconds"])

threads_list = [1,2,4,8,16,32,64,128]
naive_4096 = [_forward[(4096,t)] + _backward[(4096,t)] for t in threads_list]
fused_4096 = [_fused[(4096,t)] for t in threads_list]

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(threads_list, naive_4096, marker="o", color="#993C1D", linewidth=2, label="Naive: forward + backward (separate)")
ax.loglog(threads_list, fused_4096, marker="s", color="#0F6E56", linewidth=2, label="Fused (Algorithm 2)")
ax.set_xlabel("threads")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper II, N=4096, Delta CPU (jobs 21181232, 21226440)\nFused's advantage grows with thread count: 1.01x -> 0.72x naive time")
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper2_fused_vs_naive_n4096.pdf")
print("Saved fig_paper2_fused_vs_naive_n4096.pdf")

# ---------- Figure 9 (Paper II): GPU naive vs fused -- opposite of CPU pattern ----------
fwd_a100 = {}
with open("real_results/results_paper1_gpu_delta_gpuA100x4_21202354.csv") as f:
    for r in _csv.DictReader(f):
        fwd_a100[int(r["N"])] = float(r["seconds"])
fwd_h200 = {}
with open("real_results/results_paper1_gpu_delta_gpuH200x8_21202355.csv") as f:
    for r in _csv.DictReader(f):
        fwd_h200[int(r["N"])] = float(r["seconds"])

def load_bf(path):
    bw, fu = {}, {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            n = int(r["N"])
            if r["impl"] == "backward":
                bw[n] = float(r["seconds"])
            else:
                fu[n] = float(r["seconds"])
    return bw, fu

bw_a100, fu_a100 = load_bf("real_results/results_paper2_gpu_delta_gpuA100x4_21225789.csv")
bw_h200, fu_h200 = load_bf("real_results/results_paper2_gpu_delta_gpuH200x8_21225790.csv")

ns_gpu = sorted(fwd_a100)
ratio_a100 = [fu_a100[n] / (fwd_a100[n] + bw_a100[n]) for n in ns_gpu]
ratio_h200 = [fu_h200[n] / (fwd_h200[n] + bw_h200[n]) for n in ns_gpu]

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(ns_gpu, ratio_a100, marker="o", color="#993C1D", linewidth=2, label="A100 (jobs 21202354, 21225789)")
ax.semilogx(ns_gpu, ratio_h200, marker="s", color="#0F6E56", linewidth=2, label="H200 (jobs 21202355, 21225790)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity (fused = naive)")
ax.set_xlabel("N")
ax.set_ylabel("fused time / naive time")
ax.set_title("Paper II GPU: fused is SLOWER than naive at every size\n(opposite of the CPU N=4096 result)")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper2_gpu_ratio.pdf")
print("Saved fig_paper2_gpu_ratio.pdf")

# ---------- Figure 10 (Paper II): atomic instruction count, backward vs fused -- CONFIRMED 2x ----------
categories = ["Atomic instructions\n(GK / GV+GK)", "Device time\n(sum of kernels)"]
backward_vals = [536870912/1e6, 222.98]
fused_vals = [1073741824/1e6, 439.63]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(6.5, 5))
bars1 = ax.bar(x - width/2, backward_vals, width, label="backward (atomics: GK only)", color="#2B6E8C")
bars2 = ax.bar(x + width/2, fused_vals, width, label="fused (atomics: GV and GK)", color="#993C1D")

ax.text(x[0]-width/2, backward_vals[0]+15, f"{backward_vals[0]:.0f}M", ha="center", fontsize=9)
ax.text(x[0]+width/2, fused_vals[0]+15, f"{fused_vals[0]:.0f}M", ha="center", fontsize=9)
ax.text(x[1]-width/2, backward_vals[1]+8, f"{backward_vals[1]:.1f}ms", ha="center", fontsize=9)
ax.text(x[1]+width/2, fused_vals[1]+8, f"{fused_vals[1]:.1f}ms", ha="center", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("value (instructions in millions, or ms)")
ax.set_title("Paper II, N=2048: atomic instructions and device time\nfused has EXACTLY 2.00x backward's atomics, 1.97x its time")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper2_atomic_confirmation.pdf")
print("Saved fig_paper2_atomic_confirmation.pdf")

# ---------- Figure: conceptual diagram, naive vs memory-optimal (fused) data flow ----------
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def box(ax, x, y, w, h, text, color, fontsize=9, textcolor="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor="black", linewidth=1.2, zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, zorder=3, wrap=True)

def arrow(ax, x1, y1, x2, y2, color="black", style="-", lw=1.5):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                          color=color, linewidth=lw, linestyle=style, zorder=1)
    ax.add_patch(a)

# Title labels for the two columns
ax.text(2.3, 9.5, "Naive (unfused)", ha="center", fontsize=12, fontweight="bold")
ax.text(7.7, 9.5, "Fused (memory-optimal)", ha="center", fontsize=12, fontweight="bold")

# --- Naive path (left) ---
box(ax, 1.1, 8.2, 2.4, 0.7, "Q, K, V", "#DCE6F1")
arrow(ax, 2.3, 8.2, 2.3, 7.5)
box(ax, 1.1, 6.8, 2.4, 0.7, "compute QK^T", "#DCE6F1")
arrow(ax, 2.3, 6.8, 2.3, 6.1)
box(ax, 0.9, 5.2, 2.8, 0.9, "materialize A\n(n x n)", "#E8A87C", fontsize=9)
arrow(ax, 2.3, 5.2, 2.3, 4.5, color="#993C1D", lw=2.2)
ax.text(3.9, 4.85, "write to\nDRAM", fontsize=7.5, color="#993C1D", ha="left")
box(ax, 0.9, 3.6, 2.8, 0.9, "A sits in DRAM\n(n^2 bytes)", "#F4C7A1", fontsize=8.5)
arrow(ax, 2.3, 3.6, 2.3, 2.9, color="#993C1D", lw=2.2)
ax.text(3.9, 3.25, "read back\nfrom DRAM", fontsize=7.5, color="#993C1D", ha="left")
box(ax, 0.9, 2.0, 2.8, 0.9, "softmax +\nbackward pass", "#DCE6F1", fontsize=8.5)
arrow(ax, 2.3, 2.0, 2.3, 1.3)
box(ax, 1.1, 0.5, 2.4, 0.7, "GQ, GK, GV", "#C9E4CA")

# --- Fused path (right) ---
box(ax, 6.5, 8.2, 2.4, 0.7, "Q, K, V", "#DCE6F1")
arrow(ax, 7.7, 8.2, 7.7, 7.5)
box(ax, 6.3, 6.4, 2.8, 1.4, "compute scores,\nconsume immediately\n(A never leaves\nregisters/cache)", "#B8DCC5", fontsize=8)
ax.text(9.3, 7.1, "no DRAM\nround-trip", fontsize=7.5, color="#0F6E56", ha="left", style="italic")
arrow(ax, 7.7, 6.4, 7.7, 1.3, color="#0F6E56", lw=2.2)
box(ax, 6.5, 0.5, 2.4, 0.7, "GQ, GK, GV", "#C9E4CA")

ax.set_title("The core MoA idea: eliminate intermediate array materialization\n" +
             "This is what Paper II's fusion does -- and what Section 3.3/4.4's real DRAM and atomic-count\n" +
             "measurements directly confirm on hardware", fontsize=10)

fig.tight_layout()
fig.savefig("results_paper/fig_concept_naive_vs_fused.pdf")
print("Saved fig_concept_naive_vs_fused.pdf")

# ---------- Figure: paper series map (I -> II -> III/IV) ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

def pbox(ax, x, y, w, h, title, sub, color, status_color, status_text):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        facecolor=color, edgecolor="black", linewidth=1.3, zorder=2)
    ax.add_patch(p)
    ax.text(x+w/2, y+h*0.68, title, ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
    ax.text(x+w/2, y+h*0.38, sub, ha="center", va="center", fontsize=7.5, zorder=3)
    ax.text(x+w/2, y+h*0.12, status_text, ha="center", va="center", fontsize=7,
             color=status_color, fontweight="bold", zorder=3)

pbox(ax, 0.3, 2.0, 2.1, 1.6, "Paper I", "Forward pass\n(attention scores)", "#DCE6F1", "#0F6E56", "COMPLETE")
pbox(ax, 2.9, 2.0, 2.1, 1.6, "Paper II", "Backward + fused\n(eliminates A)", "#DCE6F1", "#0F6E56", "COMPLETE")
pbox(ax, 5.5, 3.0, 2.1, 1.6, "Paper III", "Decode / inference\n(KV-cache, O(n))", "#DCE6F1", "#0F6E56", "COMPLETE")
pbox(ax, 5.5, 0.5, 2.1, 1.6, "Paper IV", "Full block\n(norm+MLP+attn)", "#DCE6F1", "#0F6E56", "COMPLETE")
pbox(ax, 8.1, 1.75, 1.6, 1.6, "Deploy", "real-world\ninference/training", "#C9E4CA", "black", "")

arrow(ax, 2.4, 2.8, 2.9, 2.8, lw=2)
arrow(ax, 5.0, 2.8, 5.5, 3.5, lw=2)
arrow(ax, 5.0, 2.8, 5.5, 1.3, lw=2)
arrow(ax, 7.6, 3.5, 8.1, 2.9, lw=2)
arrow(ax, 7.6, 1.3, 8.1, 2.1, lw=2)

ax.set_title("How the four papers build toward one goal:\nmemory-optimal kernels for the complete transformer block, validated on real hardware",
             fontsize=10)
fig.tight_layout()
fig.savefig("results_paper/fig_paper_series_map.pdf")
print("Saved fig_paper_series_map.pdf")

# ---------- Figure (Paper III): the 128-thread NUMA anomaly ----------
p3 = {}
with open("real_results/timings_cpu_paper3_delta_21249667.csv") as f:
    for r in _csv.DictReader(f):
        p3[(int(r["threads"]), int(r["n"]))] = float(r["time_s"])

ns_p3 = sorted(set(n for t, n in p3 if t in (64, 128)))
t64_vals = [p3[(64, n)] for n in ns_p3]
t128_vals = [p3[(128, n)] for n in ns_p3]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(ns_p3, t64_vals, marker="o", color="#2B6E8C", linewidth=2, label="64 threads (fits within 1 socket, 4 NUMA domains)")
ax.loglog(ns_p3, t128_vals, marker="s", color="#993C1D", linewidth=2, label="128 threads (spans both sockets, all 8 NUMA domains)")
ax.set_xlabel("N (KV-cache length)")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper III decode, Delta CPU (job 21249667): 128-thread NUMA penalty\nUp to 535x slower at small N -- resolves once N amortizes cross-NUMA cost", fontsize=10.5)
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper3_numa_anomaly.pdf")
print("Saved fig_paper3_numa_anomaly.pdf")

# ---------- Figure (Paper II): GV-atomic fix confirmed -- before vs after, both shapes ----------
def load_bf2(path):
    bw, fu = {}, {}
    with open(path) as f:
        for r in _csv.DictReader(f):
            n = int(r["N"])
            if r["impl"] == "backward":
                bw[n] = float(r["seconds"])
            else:
                fu[n] = float(r["seconds"])
    return bw, fu

bw_a100_after, fu_a100_after = load_bf2("real_results/results_paper2_gpu_delta_gpuA100x4_21250395.csv")
bw_h200_after, fu_h200_after = load_bf2("real_results/results_paper2_gpu_delta_gpuH200x8_21250396.csv")

ns_after = sorted(bw_a100_after)
ratio_a100_before = [fu_a100[n] / (fwd_a100[n] + bw_a100[n]) for n in ns_after]
ratio_h200_before = [fu_h200[n] / (fwd_h200[n] + bw_h200[n]) for n in ns_after]
ratio_a100_after = [fu_a100_after[n] / (fwd_a100[n] + bw_a100_after[n]) for n in ns_after]
ratio_h200_after = [fu_h200_after[n] / (fwd_h200[n] + bw_h200_after[n]) for n in ns_after]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogx(ns_after, ratio_a100_before, marker="o", color="#993C1D", linewidth=2, linestyle="--", alpha=0.6, label="A100, before fix (jobs 21202354/21225789)")
ax.semilogx(ns_after, ratio_h200_before, marker="s", color="#C97B4A", linewidth=2, linestyle="--", alpha=0.6, label="H200, before fix (jobs 21202355/21225790)")
ax.semilogx(ns_after, ratio_a100_after, marker="o", color="#0F6E56", linewidth=2.5, label="A100, after fix (job 21250395)")
ax.semilogx(ns_after, ratio_h200_after, marker="s", color="#2B6E8C", linewidth=2.5, label="H200, after fix (job 21250396)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity (fused = naive)")
ax.set_xlabel("N")
ax.set_ylabel("fused time / naive time")
ax.set_title("Paper II GPU: eliminating the GV atomic reverses the result completely\nBefore: fused always slower. After: fused always faster, both shapes.")
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper2_gv_fix_confirmed.pdf")
print("Saved fig_paper2_gv_fix_confirmed.pdf")

# ---------- Figure (Paper III): Delta vs Anvil, two different thread-count phenomena ----------
p3_anvil = {}
with open("real_results/timings_cpu_paper3_anvil_20001576.csv") as f:
    for r in _csv.DictReader(f):
        p3_anvil[(int(r["threads"]), int(r["n"]))] = float(r["time_s"])

ns_common = sorted(set(n for t, n in p3_anvil if t in (32, 64)))
anvil_ratio = [p3_anvil[(64, n)] / p3_anvil[(32, n)] for n in ns_common]

ns_delta = sorted(set(n for t, n in p3 if t in (64, 128)))
delta_ratio = [p3[(128, n)] / p3[(64, n)] for n in ns_delta]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogx(ns_delta, delta_ratio, marker="o", color="#993C1D", linewidth=2, label="Delta: 128 vs 64 threads (128 real cores, 8 NUMA domains)")
ax.semilogx(ns_common, anvil_ratio, marker="s", color="#2B6E8C", linewidth=2, label="Anvil: 64 vs 32 threads (32 real cores, SMT off -- 2x oversubscription)")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity")
ax.set_yscale("log")
ax.set_xlabel("N (KV-cache length)")
ax.set_ylabel("time ratio (higher thread count / lower)")
ax.set_title("Paper III decode: two clusters, two different penalties\nDelta: NUMA memory-locality (535x). Anvil: oversubscription (<3x, reverses)", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper3_delta_vs_anvil.pdf")
print("Saved fig_paper3_delta_vs_anvil.pdf")

# ---------- Figure (Paper III): decode GPU cold-start fix, confirmed on real hardware ----------
p3gpu = {}
with open("real_results/results_paper3_gpu_delta_gpuA100x4_21264888.csv") as f:
    for r in _csv.DictReader(f):
        p3gpu[int(r["n"])] = float(r["seconds"])

ns_p3gpu = sorted(p3gpu)
times_p3gpu = [p3gpu[n] for n in ns_p3gpu]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(ns_p3gpu, times_p3gpu, marker="o", color="#0F6E56", linewidth=2, markersize=7, label="Post-fix (job 21264888, real data)")
# Reference: the old flawed run's approximate flat range (0.30-0.40s), from the
# original job 21180661 -- only the two endpoints and the qualitative "flat, slight
# upward trend" description are on record; not fabricating the missing intermediate
# values, so this is drawn as a shaded reference band, not a full comparison line.
ax.axhspan(0.30, 0.40, color="#993C1D", alpha=0.15, label="Pre-fix flat range (job 21180661, ~0.30-0.40s at every N)")
ax.axhline(0.383, color="#993C1D", linestyle="--", linewidth=1.2, alpha=0.6)

n0, t0 = ns_p3gpu[0], times_p3gpu[0]
ref_n = np.array(ns_p3gpu, dtype=float)
ref_o_n = t0 * (ref_n / n0)
ax.loglog(ref_n, ref_o_n, ":", color="gray", alpha=0.6, label="O(n) reference")

ax.set_xlabel("N (KV-cache length)")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper III decode GPU, cold-start fix confirmed on real A100\nReal scaling replaces the flat cold-start-dominated line")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper3_gpu_fix_confirmed.pdf")
print("Saved fig_paper3_gpu_fix_confirmed.pdf")

# ---------- Figure (Paper IV): C vs Fortran, CPU and GPU, both directions ----------
p4cpu = {}
with open("real_results/results_paper4_delta_cpu_21273806.csv") as f:
    for r in _csv.DictReader(f):
        key = (r["impl"], r["lang"], int(r["N"]), int(r["threads"]))
        p4cpu[key] = float(r["seconds"])

p4gpu = {}
with open("real_results/results_paper4_gpu_delta_gpuA100x4_21273959.csv") as f:
    for r in _csv.DictReader(f):
        key = (r["impl"], r["lang"], int(r["N"]))
        p4gpu[key] = float(r["seconds"])

ns_mlp = [64,128,256,512,1024,2048,4096,8192,16384,32768,65536]
mlp_ratio = [p4cpu[("mlp","C",n,1)]/p4cpu[("mlp","Fortran",n,1)] for n in ns_mlp]

ns_gpu = [64,128,256,512,1024,2048,4096]
gpu_ratio = [p4gpu[("fused","C-OpenACC",n)]/p4gpu[("fused","Fortran-OpenACC",n)] for n in ns_gpu]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogx(ns_mlp, mlp_ratio, marker="o", color="#2B6E8C", linewidth=2, label="CPU, MLP kernel (threads=1): C/Fortran time ratio")
ax.semilogx(ns_gpu, gpu_ratio, marker="s", color="#993C1D", linewidth=2, label="GPU, fused norm+MLP kernel: C/Fortran time ratio")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity (C = Fortran)")
ax.set_xlabel("N")
ax.set_ylabel("C time / Fortran time")
ax.set_title("Paper IV: C vs Fortran runs in OPPOSITE directions on CPU vs GPU\nCPU: C consistently faster (ratio<1). GPU: C consistently slower (ratio>1).")
ax.legend(fontsize=8, loc="center right")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper4_c_vs_fortran.pdf")
print("Saved fig_paper4_c_vs_fortran.pdf")

# ---------- Figure (Paper IV): full-block real CPU scaling + thread behavior ----------
p4cpu_v2 = {}
with open("real_results/results_paper4_delta_cpu_21294300.csv") as f:
    for r in _csv.DictReader(f):
        if r["impl"] == "fullblock":
            key = (int(r["N"]), int(r["threads"]))
            p4cpu_v2[key] = float(r["seconds"])

ns_fb = [64,128,256,512,1024,2048,4096]
fb_t1 = [p4cpu_v2[(n,1)] for n in ns_fb]
fb_best = []
for n in ns_fb:
    times = {t: p4cpu_v2[(n,t)] for t in [1,2,4,8,16,32,64,128] if (n,t) in p4cpu_v2}
    fb_best.append(min(times.values()))

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.loglog(ns_fb, fb_t1, marker="o", color="#993C1D", linewidth=2, label="1 thread")
ax.loglog(ns_fb, fb_best, marker="s", color="#0F6E56", linewidth=2, label="best thread count at each N")
ax.set_xlabel("N")
ax.set_ylabel("wall time (s)")
ax.set_title("Paper IV full-block, real Delta CPU data, 5-repeat average (job 21294300)\nN=2048->4096 jump (12.5x) CONFIRMED real: persists under averaging, isolated to this transition", fontsize=9.5)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper4_fullblock_cpu.pdf")
print("Saved fig_paper4_fullblock_cpu.pdf")

# ---------- Figure (Paper IV): M_block real DRAM traffic validation ----------
categories = ["Predicted\n(M_block)", "Measured\n(ncu, real A100)"]
values_mb = [10.486, 15.737]
colors = ["#2B6E8C", "#993C1D"]

fig, ax = plt.subplots(figsize=(6, 5.5))
bars = ax.bar(categories, values_mb, color=colors, width=0.5)
for bar, v in zip(bars, values_mb):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.2f} MB", ha="center", fontsize=10)
ax.set_ylabel("DRAM traffic (MB)")
ax.set_title("Paper IV: $M_{block}$ real DRAM traffic validation\nB=2, N=256, D=64, DFF=256 -- measured is 1.50x predicted")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper4_mblock_dram.pdf")
print("Saved fig_paper4_mblock_dram.pdf")

# ---------- Figure: GK atomic removal, real results ----------
old_bw, old_fu = {}, {}
with open("real_results/results_paper2_gpu_delta_gpuA100x4_21250395.csv") as f:
    for r in _csv.DictReader(f):
        n = int(r["N"])
        if r["impl"]=="backward": old_bw[n]=float(r["seconds"])
        else: old_fu[n]=float(r["seconds"])

new_bw, new_fu = {}, {}
with open("real_results/results_paper2_gpu_delta_gpuA100x4_21294299.csv") as f:
    for r in _csv.DictReader(f):
        n = int(r["N"])
        if r["impl"]=="backward": new_bw[n]=float(r["seconds"])
        else: new_fu[n]=float(r["seconds"])

ns_gk = sorted(old_bw)
bw_ratio = [new_bw[n]/old_bw[n] for n in ns_gk]
fu_ratio = [new_fu[n]/old_fu[n] for n in ns_gk]

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogx(ns_gk, bw_ratio, marker="o", color="#0F6E56", linewidth=2, label="backward alone: uniformly faster")
ax.semilogx(ns_gk, fu_ratio, marker="s", color="#993C1D", linewidth=2, label="fused: faster at small/medium N, SLOWER at N>=4096")
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Parity (no change)")
ax.set_xlabel("N")
ax.set_ylabel("New time / Old time (GK atomic removed / GK atomic present)")
ax.set_title("Removing GK's atomic too: real A100 results\nUniformly good for backward alone; mixed for fused (net win still holds vs naive)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper2_gk_removal.pdf")
print("Saved fig_paper2_gk_removal.pdf")

# ---------- Figure: C vs Fortran stall-reason comparison, dominant kernel ----------
kernels = ["norm\n(93)", "QKV/proj\n(99/103)", "MLP-fwd\n(104/112)", "MLP-bwd\n(122/142)"]
c_times = [0.008736, 0.005728, 3.448448, 24.578528]
f_times = [0.009056, 0.005792, 4.112992, 7.764768]

fig, ax = plt.subplots(figsize=(7.5, 5))
x = np.arange(len(kernels))
width = 0.35
ax.bar(x - width/2, c_times, width, label="C", color="#2B6E8C")
ax.bar(x + width/2, f_times, width, label="Fortran", color="#993C1D")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(kernels)
ax.set_ylabel("profiled time (s, log scale)")
ax.set_title("C vs Fortran, per-kernel profiled time at N=1024 (job 21299927)\nMLP-backward dominates both totals; C is 3.17x slower there specifically")
ax.legend()
ax.grid(True, axis="y", which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("results_paper/fig_paper4_stall_compare.pdf")
print("Saved fig_paper4_stall_compare.pdf")
