#!/bin/bash
#SBATCH --job-name=moa-profile-paper4
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=moa_profile_paper4_%j.out
#SBATCH --error=moa_profile_paper4_%j.err
#
# Profile Paper IV's complete-block GPU kernel (full_block_gpu_bench.c)
# with nsys and ncu to extract measured DRAM bytes read+written, for
# comparison against the closed-form M_block = 24nd + 4nd_ff
# prediction -- the same methodology delta_profile_paper1.sbatch used
# for M_fwd (see that script and results_paper/moa_experimental_
# validation.tex Section 3.3 for the worked example this follows).
#
# N chosen deliberately smaller than the real GPU sweep's default
# range (job 21273959 went to N=512 unprofiled in ~3s): ncu's full
# instrumentation set added roughly 200-300x overhead for comparably-
# sized kernels in this project's prior profiling runs (Paper II's
# backward kernel: 0.17s unprofiled -> 37.6s under ncu, job 21247666).
# At the same ratio, N=512 here could mean 10+ minutes just for the
# ncu step; N=256 (unprofiled ~0.78s per job 21273959) is used by
# default instead, to keep this tractable within the time limit while
# still exercising the full norm->QKV->attention->norm->FFN pipeline.
#
# DCGM conflict fix applied from the start this time (not discovered
# the hard way as it was for Paper II, job 21247572's first failed
# attempt) -- ncu calls are wrapped in dcgmi profile --pause/--resume.
#
# This produces raw profiler output (.nsys-rep, .ncu-rep); turning
# that into a predicted-vs-measured comparison against M_block is a
# follow-up analysis step, not done by this script.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-gpu` and `gpuA100x4` are still correct.
#   - Confirm nsys/ncu are on PATH after `module load nvhpc/26.5`.
#   - Copy or clone full_block_gpu_bench.c into the submission
#     directory first (from benchmarking/).
#
# Usage: sbatch delta_profile_paper4.sbatch [B] [N] [D] [DFF]

module purge
module load nvhpc/26.5
set -euo pipefail

MISSING_FILES=()
[ -f "full_block_gpu_bench.c" ] || MISSING_FILES+=("full_block_gpu_bench.c(from benchmarking/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  exit 1
fi

B=${1:-2}
N=${2:-256}
D=${3:-64}
DFF=${4:-256}

echo "Delta gpuA100x4. Job $SLURM_JOB_ID. Profiling full-block at B=$B N=$N D=$D DFF=$DFF"
which nsys ncu || echo "WARNING: nsys/ncu not found on PATH"

echo "Building full_block_gpu_bench.c with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel full_block_gpu_bench.c -o full_block_gpu_bench_profile -lm

echo "Profiling with nsys (system-level trace, kernel-launch timing)..."
nsys profile -o "paper4_nsys_${SLURM_JOB_ID}" ./full_block_gpu_bench_profile "$B" "$N" "$N" "$D" "$DFF"

srun --ntasks-per-node 1 dcgmi profile --pause

echo "Profiling with ncu (kernel-level DRAM traffic counters)..."
ncu --set full -o "paper4_ncu_${SLURM_JOB_ID}" ./full_block_gpu_bench_profile "$B" "$N" "$N" "$D" "$DFF"

srun --ntasks-per-node 1 dcgmi profile --resume

echo "Done. Profiler output: paper4_nsys_${SLURM_JOB_ID}.nsys-rep, paper4_ncu_${SLURM_JOB_ID}.ncu-rep"
echo "Next step (not automated here): extract 'dram__bytes_read.sum' and"
echo "'dram__bytes_write.sum' from the ncu report for the full_block_once"
echo "kernel, and compare against Paper IV's M_block = 24nd + 4nd_ff"
echo "prediction at B=$B, N=$N, D=$D, DFF=$DFF."
