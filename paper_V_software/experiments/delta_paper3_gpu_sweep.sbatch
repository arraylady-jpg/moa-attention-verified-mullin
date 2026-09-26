#!/bin/bash
#SBATCH --job-name=moa-delta-paper3-gpu
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=moa_delta_paper3_gpu_%j.out
#SBATCH --error=moa_delta_paper3_gpu_%j.err
#
# Paper III, Delta GPU: decode kernel, compiled with nvc. Direct port
# of anvil_paper3_gpu_sweep_a100.sbatch.
#
# MACHINE SHAPE COMPARISON: same override pattern as Papers I-II.
# Decode is O(n), not O(n^2), so this is the cheapest of the four
# papers to sweep across all four NVIDIA shapes.
#   sbatch --partition=gpuA100x4  delta_paper3_gpu_sweep.sbatch   # default
#   sbatch --partition=gpuA100x8  delta_paper3_gpu_sweep.sbatch
#   sbatch --partition=gpuA40x4   delta_paper3_gpu_sweep.sbatch
#   sbatch --partition=gpuH200x8  delta_paper3_gpu_sweep.sbatch
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-gpu` and partition names with sacctmgr/sinfo.
#   - Confirm nvhpc module version (25.3 confirmed available).
#   - Copy or clone decode_gpu_bench.c into the submission directory first.
#
# Usage: sbatch [--partition=<shape>] delta_paper3_gpu_sweep.sbatch [N_START] [N_END] [DK] [DV]

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
[ -f "decode_gpu_bench.c" ] || MISSING_FILES+=("decode_gpu_bench.c(from experiments/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-1024}
N_END=${2:-1048576}
DK=${3:-64}
DV=${4:-64}

SHAPE="${SLURM_JOB_PARTITION:-unknown}"
echo "Delta $SHAPE. Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel decode_gpu_bench.c -o decode_gpu_bench_delta -lm

OUT="results_paper3_gpu_delta_${SHAPE}_${SLURM_JOB_ID}.csv"
echo "impl,lang,n,dk,dv,seconds,machine" > "$OUT"

echo "Sweeping n: $N_START -> $N_END (doubling), dk=$DK dv=$DV, in ONE process"
echo "(context-init cost paid once via warmup, not once per size -- see"
echo "decode_gpu_bench.c revision history for why this matters)."
./decode_gpu_bench_delta "$N_START" "$N_END" "$DK" "$DV" | while IFS= read -r ROW; do
  echo "${ROW},delta-${SHAPE}" >> "$OUT"
done

echo "Done. Results in $OUT"
