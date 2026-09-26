"""
compare_machine_shapes.py

Ingests result CSVs from different machine shapes (Anvil A100/H100,
Delta gpuA100x4/gpuA100x8/gpuA40x4/gpuH200x8, and CPU runs on either
cluster) and produces:
  1. A combined long-format CSV (all runs, all machines, one row each).
  2. A per-(paper,N) comparison table (wall time and, where derivable,
     speedup relative to the slowest machine at that point).
  3. A log-log scaling plot, one line per machine, for a chosen
     paper/kernel.

USAGE
  python3 compare_machine_shapes.py --out combined.csv \
      results_paper1_cpu_anvil_12345.csv \
      results_paper1_gpu_delta_gpuA100x4_67890.csv:machine=delta-gpuA100x4 \
      results_paper1_gpu_delta_gpuH200x8_67891.csv:machine=delta-gpuH200x8 \
      timings_cpu_paper3_delta_11111.csv:machine=delta-cpu:schema=decode

Each input is a path, optionally suffixed with ":machine=<label>" (for
files whose script never added a machine column -- moa_decode_bench.c's
timings_cpu.csv and run_cpu_sweep.sh's results_cpu.csv are both like
this, since they're unmodified original tools, not new machine-shape-
aware wrappers) and/or ":schema=decode" (moa_decode_bench.c's CSV has a
different column layout: threads,n,dk,dv,time_s,gbps,traffic_mb,speedup_vs_seq
instead of the impl,lang,B,N,D,[threads,]seconds,[machine] layout every
other file in this project uses).

This script has been tested against synthetic mock data with the exact
schemas each real script would produce (see make_mock_data() below, run
via --self-test), NOT against real experiment output, since no
experiments have been run on real hardware as of this writing. Real
column values, ranges, and edge cases (e.g. a run that errored out
partway through a sweep) may need adjustments this script doesn't yet
handle.
"""
import argparse
import csv
import sys
from collections import defaultdict

STANDARD_COLUMNS = ["impl", "lang", "B", "N", "D", "DFF", "threads", "seconds", "machine"]


def parse_input_spec(spec):
    """Split 'path.csv:machine=label:schema=decode' into (path, overrides)."""
    parts = spec.split(":")
    path = parts[0]
    overrides = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"malformed input spec segment '{part}' in '{spec}'")
        k, v = part.split("=", 1)
        overrides[k] = v
    return path, overrides


def load_standard(path, overrides):
    """Load a file already in the standard impl,lang,B,N,D,[DFF,][threads,]seconds[,machine] layout."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {
                "impl": r.get("impl", ""),
                "lang": r.get("lang", ""),
                "B": r.get("B", ""),
                "N": r.get("N", r.get("n", "")),
                "D": r.get("D", ""),
                "DFF": r.get("DFF", ""),
                "threads": r.get("threads", "1"),
                "seconds": r.get("seconds", ""),
                "machine": overrides.get("machine", r.get("machine", "unknown")),
            }
            rows.append(row)
    return rows


def load_decode_schema(path, overrides):
    """Load moa_decode_bench.c's native CSV layout:
    threads,n,dk,dv,time_s,gbps,traffic_mb,speedup_vs_seq"""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "impl": "decode",
                "lang": "C",
                "B": "1",
                "N": r["n"],
                "D": r.get("dk", ""),
                "DFF": "",
                "threads": r["threads"],
                "seconds": r["time_s"],
                "machine": overrides.get("machine", "unknown"),
            })
    return rows


def load_file(spec):
    path, overrides = parse_input_spec(spec)
    schema = overrides.get("schema", "standard")
    if schema == "decode":
        return load_decode_schema(path, overrides)
    elif schema == "standard":
        return load_standard(path, overrides)
    else:
        raise ValueError(f"unknown schema '{schema}'")


def write_combined(rows, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STANDARD_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_comparison_table(rows, impl_filter=None):
    """Group by (impl, N), print seconds per machine side by side."""
    grouped = defaultdict(dict)
    machines = set()
    for r in rows:
        if impl_filter and r["impl"] != impl_filter:
            continue
        key = (r["impl"], r["N"])
        grouped[key][r["machine"]] = r["seconds"]
        machines.add(r["machine"])

    machines = sorted(machines)
    print(f"{'impl':12s} {'N':>10s} " + " ".join(f"{m:>18s}" for m in machines))
    for (impl, n), by_machine in sorted(grouped.items(), key=lambda kv: (kv[0][0], int(kv[0][1]) if kv[0][1].isdigit() else 0)):
        row_str = f"{impl:12s} {n:>10s} "
        for m in machines:
            v = by_machine.get(m, "-")
            row_str += f"{v:>18s} "
        print(row_str)


def plot_scaling(rows, impl_filter, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_machine = defaultdict(lambda: {"N": [], "s": []})
    for r in rows:
        if r["impl"] != impl_filter:
            continue
        try:
            n = int(r["N"])
            s = float(r["seconds"])
        except (ValueError, TypeError):
            continue
        by_machine[r["machine"]]["N"].append(n)
        by_machine[r["machine"]]["s"].append(s)

    if not by_machine:
        print(f"No data found for impl='{impl_filter}', skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, (machine, d) in enumerate(sorted(by_machine.items())):
        pairs = sorted(zip(d["N"], d["s"]))
        ns = [p[0] for p in pairs]
        ss = [p[1] for p in pairs]
        ax.loglog(ns, ss, marker=markers[i % len(markers)], label=machine)
    ax.set_xlabel("N")
    ax.set_ylabel("wall time (s)")
    ax.set_title(f"{impl_filter}: machine shape comparison")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved {out_path}")


def make_mock_data(tmpdir):
    """Generate synthetic CSVs matching each real script's exact schema,
    for self-testing this analysis script without real hardware data."""
    import os
    os.makedirs(tmpdir, exist_ok=True)

    # Standard schema, with machine column (e.g. delta_paper1_gpu_sweep.sbatch output)
    p1 = os.path.join(tmpdir, "mock_paper1_gpu_a100x4.csv")
    with open(p1, "w") as f:
        f.write("impl,lang,B,N,D,seconds,machine\n")
        for n, t in [(64, 0.001), (128, 0.002), (256, 0.005), (512, 0.012)]:
            f.write(f"forward,C-OpenACC,2,{n},64,{t},delta-gpuA100x4\n")

    p2 = os.path.join(tmpdir, "mock_paper1_gpu_h200x8.csv")
    with open(p2, "w") as f:
        f.write("impl,lang,B,N,D,seconds,machine\n")
        for n, t in [(64, 0.0006), (128, 0.0011), (256, 0.0027), (512, 0.0065)]:
            f.write(f"forward,C-OpenACC,2,{n},64,{t},delta-gpuH200x8\n")

    # Standard schema, WITHOUT machine column (e.g. run_cpu_sweep.sh raw output)
    p3 = os.path.join(tmpdir, "mock_paper4_cpu_anvil.csv")
    with open(p3, "w") as f:
        f.write("impl,lang,B,N,D,DFF,threads,seconds\n")
        for n, t in [(64, 0.0001), (128, 0.0002), (256, 0.0004)]:
            f.write(f"rmsnorm,C,2,{n},64,0,128,{t}\n")

    # Decode's native schema (moa_decode_bench.c)
    p4 = os.path.join(tmpdir, "mock_timings_decode.csv")
    with open(p4, "w") as f:
        f.write("threads,n,dk,dv,time_s,gbps,traffic_mb,speedup_vs_seq\n")
        for n, t in [(1024, 0.0002), (2048, 0.0003), (4096, 0.0011)]:
            f.write(f"128,{n},64,64,{t},4.0,0.5,1.0\n")

    return [
        f"{p1}",
        f"{p2}",
        f"{p3}:machine=anvil-cpu",
        f"{p4}:machine=delta-cpu:schema=decode",
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="input CSV specs, see module docstring")
    ap.add_argument("--out", default="combined_machine_shapes.csv", help="combined output CSV path")
    ap.add_argument("--plot-impl", default=None, help="impl name to plot (e.g. 'forward', 'rmsnorm')")
    ap.add_argument("--plot-out", default="machine_shape_comparison.png", help="plot output path")
    ap.add_argument("--table-impl", default=None, help="impl name to filter the printed table by")
    ap.add_argument("--self-test", action="store_true", help="run against synthetic mock data instead of real files")
    args = ap.parse_args()

    if args.self_test:
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="moa_mock_")
        print(f"Self-test mode: generating mock data in {tmpdir}")
        args.inputs = make_mock_data(tmpdir)
        args.plot_impl = args.plot_impl or "forward"
        args.table_impl = args.table_impl or "forward"

    if not args.inputs:
        ap.error("no input files given (use --self-test to try with synthetic data)")

    all_rows = []
    for spec in args.inputs:
        rows = load_file(spec)
        print(f"Loaded {len(rows)} rows from {spec}")
        all_rows.extend(rows)

    write_combined(all_rows, args.out)
    print(f"\nCombined {len(all_rows)} rows -> {args.out}\n")

    print("=== Comparison table ===")
    print_comparison_table(all_rows, impl_filter=args.table_impl)

    if args.plot_impl:
        plot_scaling(all_rows, args.plot_impl, args.plot_out)


if __name__ == "__main__":
    main()
