import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results_cpu.csv")))

series = {}
for r in rows:
    key = (r["impl"], r["lang"])
    series.setdefault(key, {"N": [], "s": []})
    series[key]["N"].append(int(r["N"]))
    series[key]["s"].append(float(r["seconds"]))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

styles = {
    ("rmsnorm", "C"): dict(color="#0F6E56", marker="o", label="RMSNorm, C"),
    ("rmsnorm", "Fortran"): dict(color="#5DCAA5", marker="s", label="RMSNorm, Fortran90"),
    ("mlp", "C"): dict(color="#993C1D", marker="o", label="Gated MLP, C"),
    ("mlp", "Fortran"): dict(color="#D85A30", marker="s", label="Gated MLP, Fortran90"),
}

for key, ax, title in [
    (("rmsnorm", "C"), axes[0], "RMSNorm forward+backward"),
    (("mlp", "C"), axes[1], "Gated MLP forward+backward"),
]:
    pass

for k in [("rmsnorm", "C"), ("rmsnorm", "Fortran")]:
    d = series[k]
    axes[0].loglog(d["N"], d["s"], **styles[k])
axes[0].set_title("RMSNorm forward+backward")
axes[0].set_xlabel("sequence length $n$ ($B{=}2$, $D{=}64$)")
axes[0].set_ylabel("wall time (s)")
axes[0].legend(fontsize=8)
axes[0].grid(True, which="both", alpha=0.3)

for k in [("mlp", "C"), ("mlp", "Fortran")]:
    d = series[k]
    axes[1].loglog(d["N"], d["s"], **styles[k])
axes[1].set_title("Gated MLP forward+backward")
axes[1].set_xlabel("sequence length $n$ ($B{=}2$, $D{=}64$, $D_{ff}{=}256$)")
axes[1].set_ylabel("wall time (s)")
axes[1].legend(fontsize=8)
axes[1].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig("scaling_plot.pdf")
print("Saved scaling_plot.pdf")
