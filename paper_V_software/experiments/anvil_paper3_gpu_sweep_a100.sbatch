#!/bin/bash
#SBATCH --job-name=moa-paper3-gpu-a100
#SBATCH --account=cis261396-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=moa_paper3_gpu_a100_%j.out
#SBATCH --error=moa_paper3_gpu_a100_%j.err
#
# Paper III, GPU: decode kernel (single query against n cached
# keys/values), compiled with nvc for real GPU offload. Same structure
# as the other Anvil GPU scripts in this directory.
#
# REVISION: real Delta A100 data (job 21180661) showed wall time flat
# (~0.3-0.4s) across a 1024x range of n -- wrong for an O(n) kernel.
# Root cause: the original design launched one process per n, and CUDA
# context creation (~300ms) dominated every single measurement. Fixed
# by having decode_gpu_bench.c sweep n internally within one process
# (context-init cost paid once via an untimed warmup); this script now
# calls the binary once with the full range instead of looping externally.
#
# decode_gpu_bench.c ports moa_decode_omp's three-pass structure
# (scores+max, softmax, weighted sum) from
# attention/moa_decode_bench.c to OpenACC. Verified with --check against
# a CPU serial reference before this script was written: out matched to
# exact 0.000e+00 across multiple sizes (GCC host-fallback).
#
# DEVIATION FROM SOURCE: decode_gpu_bench.c uses double, not the
# original moa_decode_bench.c's float -- see the .c file's header
# comment for why.
#
# CONFIRMED via sinfo/sacctmgr on Anvil (login07): "gpu" partition and
# "cis261396-gpu" account are both real. "wholenode" (CPU partition,
# used elsewhere) is also confirmed real -- the earlier squeue showing
# "batch" turned out to be a different cluster (NCSA Delta), not Anvil.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396-gpu` above if your allocation ID differs.
#   - Confirm `--constraint=A100` and `--partition=gpu` are correct.
#   - Confirm the nvhpc module name/version with `module avail nvhpc`.
#   - Copy or clone decode_gpu_bench.c into the submission directory first.
#
# Usage: sbatch anvil_paper3_gpu_sweep_a100.sbatch [N_START] [N_END] [DK] [DV]

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

echo "Anvil A100 (g node). Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel decode_gpu_bench.c -o decode_gpu_bench_a100 -lm

OUT="results_paper3_gpu_a100_${SLURM_JOB_ID}.csv"
echo "impl,lang,n,dk,dv,seconds" > "$OUT"

echo "Sweeping n: $N_START -> $N_END (doubling), dk=$DK dv=$DV, in ONE process"
echo "(context-init cost paid once via warmup, not once per size -- see"
echo "decode_gpu_bench.c revision history for why this matters)."
echo "Decode is O(n) not O(n^2), so this can push n much larger than"
echo "Papers I/II/IV's attention-forward sweeps before becoming impractical."
./decode_gpu_bench_a100 "$N_START" "$N_END" "$DK" "$DV" >> "$OUT"

echo "Done. Results in $OUT"
