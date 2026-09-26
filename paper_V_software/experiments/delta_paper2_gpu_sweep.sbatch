#!/bin/bash
#SBATCH --job-name=moa-delta-paper2-gpu
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=moa_delta_paper2_gpu_%j.out
#SBATCH --error=moa_delta_paper2_gpu_%j.err
#
# Paper II, Delta GPU: backward (Algorithm 1) and fused (Algorithm 2),
# compiled with nvc. Direct port of anvil_paper2_gpu_sweep_a100.sbatch.
#
# MACHINE SHAPE COMPARISON: override partition at submit time, same
# pattern as delta_paper1_gpu_sweep.sbatch:
#   sbatch --partition=gpuA100x4  delta_paper2_gpu_sweep.sbatch   # default
#   sbatch --partition=gpuA100x8  delta_paper2_gpu_sweep.sbatch
#   sbatch --partition=gpuA40x4   delta_paper2_gpu_sweep.sbatch
#   sbatch --partition=gpuH200x8  delta_paper2_gpu_sweep.sbatch
# gpuMI100x8 (AMD/ROCm) excluded, same reason as Paper I's script.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-gpu` and partition names with sacctmgr/sinfo.
#   - Confirm nvhpc module version (25.3 confirmed available).
#   - Copy or clone backward_gpu_bench.c and fused_attn_gpu_bench.c first.
#
# Usage: sbatch [--partition=<shape>] delta_paper2_gpu_sweep.sbatch [N_START] [N_END] [B] [D]

module purge
module load nvhpc/26.5

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

SHAPE="${SLURM_JOB_PARTITION:-unknown}"
echo "Delta $SHAPE. Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel backward_gpu_bench.c -o backward_gpu_bench_delta -lm
nvc -O2 -acc -Minfo=accel fused_attn_gpu_bench.c -o fused_attn_gpu_bench_delta -lm

OUT="results_paper2_gpu_delta_${SHAPE}_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,seconds,repeats,machine" > "$OUT"

echo "Sweeping N: $N_START -> $N_END (doubling), in ONE process each. N capped at 8192 (fixed scratch buffer, raised from 512 after v2 fix)."
./backward_gpu_bench_delta "$B" "$N_START" "$N_END" "$D" | while IFS= read -r ROW; do
  echo "${ROW},delta-${SHAPE}" >> "$OUT"
done
./fused_attn_gpu_bench_delta "$B" "$N_START" "$N_END" "$D" | while IFS= read -r ROW; do
  echo "${ROW},delta-${SHAPE}" >> "$OUT"
done

echo "Done. Results in $OUT"
