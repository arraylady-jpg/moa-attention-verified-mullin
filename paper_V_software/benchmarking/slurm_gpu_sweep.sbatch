#!/bin/bash
#SBATCH --job-name=moa-fused-gpu-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=moa_gpu_sweep_%j.out
#SBATCH --error=moa_gpu_sweep_%j.err
#
# Sweeps problem size (sequence length N, doubling from N_START to N_END) for
# the fused RMSNorm+gated MLP OpenACC kernel (C and Fortran), and separately
# for the complete transformer block (Norm+QKV+Attention+Norm+FFN, C only --
# see note below), on ONE GPU.
#
# IMPORTANT LIMITATION: none of these programs implement multi-GPU domain
# decomposition. "Number of processors increasing to max" therefore does not
# apply to this script the way it does to the CPU OpenMP sweep: there is no
# thread/rank count to scale on one GPU kernel launch. If multi-GPU scaling is
# wanted, this script would need an MPI+OpenACC or NCCL-based decomposition of
# the (B,N) token dimension across GPUs, which is not yet implemented -- see
# Section 10 ("Remaining") of the paper. This script sweeps SIZE ONLY.
#
# IMPORTANT LIMITATION: full_block_gpu_bench.c uses fixed-size stack scratch
# arrays (arow, ga_row, GUrow, GVrow) capped at 512 elements, so N and DFF
# must each stay <= 512 for that program specifically -- use
# FULL_BLOCK_N_END below to cap it independently of the norm/MLP sweep range.
# There is also no Fortran90 counterpart to full_block_gpu_bench yet (listed
# as remaining work in Section 10 of the paper), so the full-block portion
# of this sweep is C only.
#
# Adjust module names below for your cluster's actual GPU compiler module
# (this uses NVIDIA HPC SDK's nvc/nvfortran; substitute pgcc/pgfortran or your
# site's equivalent if different).
#
# Usage: sbatch slurm_gpu_sweep.sbatch [N_START] [N_END] [B] [D] [DFF] [FULL_BLOCK_N_END]

module purge
module load nvhpc || module load pgi || echo "WARNING: no nvhpc/pgi module found, adjust for your site"

# Fail loudly on any error, same fix applied to every other script in this
# package after two real Delta jobs (21180084, 21180325) silently reported
# COMPLETED with no data because a missing source file's compile failure
# went unchecked. This script was in a different folder (benchmarking/, not
# experiments/) and was missed in that earlier fix pass -- added now.
set -euo pipefail

MISSING_FILES=()
[ -f "fused_gpu_bench.c" ] || MISSING_FILES+=("fused_gpu_bench.c")
[ -f "fused_gpu_bench.f90" ] || MISSING_FILES+=("fused_gpu_bench.f90")
[ -f "full_block_gpu_bench.c" ] || MISSING_FILES+=("full_block_gpu_bench.c")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "All three live in this same benchmarking/ folder -- copy them here first."
  exit 1
fi

N_START=${1:-64}
N_END=${2:-4096}
B=${3:-2}
D=${4:-64}
DFF=${5:-256}
FULL_BLOCK_N_END=${6:-512}

OUT="results_gpu_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,DFF,seconds" > "$OUT"

echo "Building with nvc/nvfortran for GPU offload..."
nvc -O2 -acc -Minfo=accel fused_gpu_bench.c -o fused_gpu_bench_gpu -lm
nvfortran -O2 -acc -Minfo=accel fused_gpu_bench.f90 -o fused_gpu_bench_f90_gpu
nvc -O2 -acc -Minfo=accel full_block_gpu_bench.c -o full_block_gpu_bench_gpu -lm

echo "Sweeping N: $N_START -> $N_END (doubling), C binaries in ONE process each."
echo "Full-block N capped at $FULL_BLOCK_N_END (fixed 512-element scratch buffers)."

./fused_gpu_bench_gpu "$B" "$N_START" "$N_END" "$D" "$DFF" >> "$OUT"
if [ "$FULL_BLOCK_N_END" -ge "$N_START" ]; then
  ./full_block_gpu_bench_gpu "$B" "$N_START" "$FULL_BLOCK_N_END" "$D" "$DFF" >> "$OUT"
fi

# fused_gpu_bench_f90_gpu (Fortran) now has the same warmup/single-process-sweep
# fix as the C binaries above (see fused_gpu_bench.f90's revision history).
./fused_gpu_bench_f90_gpu "$B" "$N_START" "$N_END" "$D" "$DFF" | tr -d ' ' >> "$OUT"

echo "Done. Results in $OUT"
