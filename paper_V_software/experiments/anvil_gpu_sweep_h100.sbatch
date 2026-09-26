#!/bin/bash
#SBATCH --job-name=moa-gpu-sweep-h100
#SBATCH --account=cis261396-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=H100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=moa_gpu_sweep_h100_%j.out
#SBATCH --error=moa_gpu_sweep_h100_%j.err
#
# Experiment B (H100): real GPU bandwidth/latency for the fused norm+MLP
# kernel and the complete block kernel, compiled with nvc/nvfortran
# (NVIDIA HPC SDK) instead of GCC's OpenACC host-fallback used in the
# development sandbox. This is also the first real compilation test of
# these files under a genuine OpenACC-to-GPU compiler.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396-gpu` above if your allocation ID differs.
#   - Confirm `--constraint=H100` is still correct (Anvil's H100 nodes
#     are tagged with the `H100` feature per `sfeatures`; H100 nodes use
#     `H100` -- see anvil_gpu_sweep_h100.sbatch for that target).
#   - Confirm the nvhpc module name/version with `module avail nvhpc`.
#   - Copy or clone benchmarking/fused_gpu_bench.{c,f90},
#     full_block_gpu_bench.c into the submission directory first.
#
# Usage: sbatch anvil_gpu_sweep_h100.sbatch [N_START] [N_END] [B] [D] [DFF] [FULL_BLOCK_N_END]

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

echo "Anvil H100 (g node). Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvaccelinfo | head -20 || echo "(nvaccelinfo failed or unavailable, continuing anyway -- diagnostic only)"

echo "Building with nvc/nvfortran for real GPU offload..."
nvc -O2 -acc -Minfo=accel fused_gpu_bench.c -o fused_gpu_bench_h100 -lm
nvfortran -O2 -acc -Minfo=accel fused_gpu_bench.f90 -o fused_gpu_bench_f90_h100
nvc -O2 -acc -Minfo=accel full_block_gpu_bench.c -o full_block_gpu_bench_h100 -lm

OUT="results_gpu_h100_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,DFF,seconds" > "$OUT"

echo "Sweeping N: $N_START -> $N_END (doubling), C binaries in ONE process each."
echo "Full-block N capped at $FULL_BLOCK_N_END."
./fused_gpu_bench_h100 "$B" "$N_START" "$N_END" "$D" "$DFF" >> "$OUT"
if [ "$FULL_BLOCK_N_END" -ge "$N_START" ]; then
  ./full_block_gpu_bench_h100 "$B" "$N_START" "$FULL_BLOCK_N_END" "$D" "$DFF" >> "$OUT"
fi

# fused_gpu_bench_f90_h100 (Fortran) now has the same warmup/single-process-sweep
# fix as the C binaries above (see fused_gpu_bench.f90's revision history).
./fused_gpu_bench_f90_h100 "$B" "$N_START" "$N_END" "$D" "$DFF" | tr -d ' ' >> "$OUT"

echo "Done. Results in $OUT"
