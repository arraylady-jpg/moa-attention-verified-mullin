#!/bin/bash
#SBATCH --job-name=moa-profile-paper1
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=moa_profile_paper1_%j.out
#SBATCH --error=moa_profile_paper1_%j.err
#
# Profile Paper I's forward-pass GPU kernel with Nsight Systems (nsys)
# and Nsight Compute (ncu) to extract measured DRAM bytes read+written
# per forward call, for comparison against the closed-form M_fwd
# prediction (Paper I -- see that paper for M_fwd's exact form; not
# restated here to avoid transcribing it incorrectly from memory).
#
# Unlike Paper IV's anvil_profile_block.sbatch, there is no separate
# fixed-size file-I/O verification program for Paper I's GPU kernel to
# profile -- forward_gpu_bench.c (the dynamic-size benchmark itself,
# with in-process random data generation) is profiled directly instead,
# at a single representative N rather than a full sweep, since profiler
# overhead makes sweeping impractical under nsys/ncu.
#
# This produces raw profiler output (.nsys-rep, .ncu-rep); turning that
# into a predicted-vs-measured comparison against M_fwd is a follow-up
# analysis step, not done by this script.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-gpu` and `gpuA100x4` are still correct.
#   - Confirm nsys/ncu are on PATH after `module load nvhpc/26.5`.
#   - Copy or clone forward_gpu_bench.c into the submission directory first.
#
# Usage: sbatch delta_profile_paper1.sbatch [B] [N] [D]

module purge
module load nvhpc/26.5

set -euo pipefail

MISSING_FILES=()
[ -f "forward_gpu_bench.c" ] || MISSING_FILES+=("forward_gpu_bench.c(from experiments/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  exit 1
fi

B=${1:-2}
N=${2:-2048}
D=${3:-64}

echo "Delta gpuA100x4. Job $SLURM_JOB_ID on $SLURM_NODELIST"
which nsys ncu || echo "WARNING: nsys/ncu not found on PATH after module load nvhpc/26.5 -- check module list"

echo "Building forward_gpu_bench.c with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel forward_gpu_bench.c -o forward_gpu_bench_profile -lm

echo "Profiling with nsys (system-level trace, kernel-launch timing) at B=$B N=$N D=$D..."
nsys profile -o "paper1_nsys_${SLURM_JOB_ID}" ./forward_gpu_bench_profile "$B" "$N" "$N" "$D" 1

srun --ntasks-per-node 1 dcgmi profile --pause

echo "Profiling with ncu (kernel-level DRAM traffic counters)..."
ncu --set full -o "paper1_ncu_${SLURM_JOB_ID}" ./forward_gpu_bench_profile "$B" "$N" "$N" "$D" 1

srun --ntasks-per-node 1 dcgmi profile --resume

echo "Done. Profiler output: paper1_nsys_${SLURM_JOB_ID}.nsys-rep, paper1_ncu_${SLURM_JOB_ID}.ncu-rep"
echo "Next step (not automated here): extract 'dram__bytes_read.sum' and"
echo "'dram__bytes_write.sum' from the ncu report for the forward kernel,"
echo "and compare against Paper I's M_fwd prediction at B=$B, N=$N, D=$D."
