#!/bin/bash
#SBATCH --job-name=moa-delta-paper4-gpu
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=moa_delta_paper4_gpu_%j.out
#SBATCH --error=moa_delta_paper4_gpu_%j.err
#
# Paper IV, Delta GPU: fused norm+MLP kernel (C+Fortran) and the
# complete block kernel (C only). Direct port of anvil_gpu_sweep_a100.sbatch
# and the full-block portion of the Anvil GPU scripts.
#
# MACHINE SHAPE COMPARISON: same override pattern as Papers I-III.
#   sbatch --partition=gpuA100x4  delta_paper4_gpu_sweep.sbatch   # default
#   sbatch --partition=gpuA100x8  delta_paper4_gpu_sweep.sbatch
#   sbatch --partition=gpuA40x4   delta_paper4_gpu_sweep.sbatch
#   sbatch --partition=gpuH200x8  delta_paper4_gpu_sweep.sbatch
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-gpu` and partition names with sacctmgr/sinfo.
#   - Confirm nvhpc module version (25.3 confirmed available).
#   - Copy or clone benchmarking/fused_gpu_bench.{c,f90} and
#     benchmarking/full_block_gpu_bench.c into the submission directory first.
#
# Usage: sbatch [--partition=<shape>] delta_paper4_gpu_sweep.sbatch [N_START] [N_END] [B] [D] [DFF] [FULL_BLOCK_N_END]

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
[ -f "full_block_gpu_bench.c" ] || MISSING_FILES+=("full_block_gpu_bench.c(from benchmarking/)")
[ -f "fused_gpu_bench.c" ] || MISSING_FILES+=("fused_gpu_bench.c(from benchmarking/)")
[ -f "fused_gpu_bench.f90" ] || MISSING_FILES+=("fused_gpu_bench.f90(from benchmarking/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-64}
N_END=${2:-4096}
B=${3:-2}
D=${4:-64}
DFF=${5:-256}
FULL_BLOCK_N_END=${6:-512}

SHAPE="${SLURM_JOB_PARTITION:-unknown}"
echo "Delta $SHAPE. Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc/nvfortran for real GPU offload..."
nvc -O2 -acc -Minfo=accel fused_gpu_bench.c -o fused_gpu_bench_delta -lm
nvfortran -O2 -acc -Minfo=accel fused_gpu_bench.f90 -o fused_gpu_bench_f90_delta
nvc -O2 -acc -Minfo=accel full_block_gpu_bench.c -o full_block_gpu_bench_delta -lm

OUT="results_paper4_gpu_delta_${SHAPE}_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,DFF,seconds,machine" > "$OUT"

echo "Sweeping N: $N_START -> $N_END (doubling), C binaries in ONE process each."
echo "Full-block N capped at $FULL_BLOCK_N_END."
./fused_gpu_bench_delta "$B" "$N_START" "$N_END" "$D" "$DFF" | while IFS= read -r ROW; do
  echo "${ROW},delta-${SHAPE}" >> "$OUT"
done
if [ "$FULL_BLOCK_N_END" -ge "$N_START" ]; then
  ./full_block_gpu_bench_delta "$B" "$N_START" "$FULL_BLOCK_N_END" "$D" "$DFF" | while IFS= read -r ROW; do
    echo "${ROW},delta-${SHAPE}" >> "$OUT"
  done
fi

# fused_gpu_bench_f90_delta (Fortran) now has the same warmup/single-process-sweep
# fix as the C binaries above, applied after real Delta A100 data (job 21180661)
# revealed the cold-start-dominance bug in decode_gpu_bench.c and, on inspection,
# in every other *_gpu_bench.c/.f90 file in this project -- see EXPERIMENT_PLAN.md.
./fused_gpu_bench_f90_delta "$B" "$N_START" "$N_END" "$D" "$DFF" | tr -d ' ' | while IFS= read -r ROW; do
  echo "${ROW},delta-${SHAPE}" >> "$OUT"
done

echo "Done. Results in $OUT"
