#!/bin/bash
#SBATCH --job-name=moa-paper2-gpu-h100
#SBATCH --account=cis261396-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=H100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=moa_paper2_gpu_h100_%j.out
#SBATCH --error=moa_paper2_gpu_h100_%j.err
#
# Paper II, GPU: backward pass (Algorithm 1) and fused forward+backward
# (Algorithm 2), compiled with nvc for real GPU offload. Same structure
# as Paper IV's anvil_gpu_sweep_h100.sbatch and Paper I's
# anvil_paper1_gpu_sweep_h100.sbatch.
#
# backward_gpu_bench.c and fused_attn_gpu_bench.c were verified with
# --check against CPU serial/composed references before this script was
# written: GV/GQ/GK (and Out for the fused kernel) matched to exact
# 0.000e+00 (GCC host-fallback).
#
# NOTE: fused_attn_gpu_bench.c is named to disambiguate from Paper IV's
# fused_gpu_bench.c (norm+MLP) -- different kernels, same naming risk
# if you're copying files between directories.
#
# CONFIRMED via sinfo/sacctmgr on Anvil (login07): "gpu" partition and
# "cis261396-gpu" account are both real. "wholenode" (CPU partition,
# used elsewhere) is also confirmed real -- the earlier squeue showing
# "batch" turned out to be a different cluster (NCSA Delta), not Anvil.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396-gpu` above if your allocation ID differs.
#   - Confirm `--constraint=H100` and `--partition=gpu` are correct.
#   - Confirm the nvhpc module name/version with `module avail nvhpc`.
#   - Copy or clone backward_gpu_bench.c and fused_attn_gpu_bench.c into
#     the submission directory first.
#
# Usage: sbatch anvil_paper2_gpu_sweep_h100.sbatch [N_START] [N_END] [B] [D]

module purge
module load nvhpc

# Fail loudly on any error (missing source file, failed compile, failed run)
# instead of silently continuing to write garbage/empty CSV rows -- this is
# exactly the class of bug that produced empty rows in an earlier real run
# (source .c files weren't copied to the submission directory; the script
# kept going past the compile failure and wrote blank CSV lines instead of
# stopping). Copy/clone the required source files into this directory
# BEFORE submitting -- see the ACTION NEEDED section above.
set -euo pipefail

MISSING_FILES=()
[ -f "backward_gpu_bench.c" ] || MISSING_FILES+=("backward_gpu_bench.c(from experiments/)")
[ -f "fused_attn_gpu_bench.c" ] || MISSING_FILES+=("fused_attn_gpu_bench.c(from experiments/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-64}
N_END=${2:-8192}
B=${3:-2}
D=${4:-64}

echo "Anvil H100 (h node). Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel backward_gpu_bench.c -o backward_gpu_bench_h100 -lm
nvc -O2 -acc -Minfo=accel fused_attn_gpu_bench.c -o fused_attn_gpu_bench_h100 -lm

OUT="results_paper2_gpu_h100_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,seconds,repeats" > "$OUT"

echo "Sweeping N: $N_START -> $N_END (doubling), in ONE process each. N capped at 8192 (fixed scratch buffer, raised from 512 after v2 fix)."
./backward_gpu_bench_h100   "$B" "$N_START" "$N_END" "$D" >> "$OUT"
./fused_attn_gpu_bench_h100 "$B" "$N_START" "$N_END" "$D" >> "$OUT"

echo "Done. Results in $OUT"
