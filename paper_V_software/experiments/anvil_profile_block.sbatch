#!/bin/bash
#SBATCH --job-name=moa-profile-block
#SBATCH --account=cis261396-gpu
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=moa_profile_block_%j.out
#SBATCH --error=moa_profile_block_%j.err
#
# Experiment C: profile the full-block GPU kernel with Nsight Systems
# (nsys) and Nsight Compute (ncu) to extract measured DRAM bytes
# read+written per forward+backward call, for comparison against the
# closed-form M_block prediction from Section 9 of the paper:
#   M_block_fwd+bwd = 24nd + 4nd_ff
#
# This produces raw profiler output (.nsys-rep, .ncu-rep); turning that
# into the predicted-vs-measured table/plot described in
# EXPERIMENT_PLAN.md Section 6 is a follow-up analysis step, not done
# by this script.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396-gpu` above if your allocation ID differs.
#   - Confirm nsys/ncu are on PATH after `module load nvhpc` (they ship
#     with the CUDA toolkit component of the HPC SDK on most sites,
#     but confirm with `which nsys ncu` before relying on this).
#   - Copy or clone verification/full_block_gpu_kernel.c and its
#     data-generation scripts (reference_full_block.py, or use fixed
#     small-size correctness data) into the submission directory first
#     -- this profiles the FIXED-SIZE verification kernel, not the
#     dynamic-size benchmark, since profiler overhead dominates at these
#     small sizes anyway and the verification kernel already has known-
#     correct reference output to sanity-check against.
#
# Usage: sbatch anvil_profile_block.sbatch

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
[ -f "full_block_gpu_kernel.c" ] || MISSING_FILES+=("full_block_gpu_kernel.c(from verification/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

echo "Anvil A100 (g node). Job $SLURM_JOB_ID on $SLURM_NODELIST"
which nsys ncu || echo "WARNING: nsys/ncu not found on PATH after module load nvhpc -- check module list"

echo "Building full_block_gpu_kernel.c with nvc for real GPU offload..."
nvc -O2 -acc -Minfo=accel full_block_gpu_kernel.c -o full_block_gpu_kernel_a100 -lm

echo "Generating reference input data (requires python3 + torch)..."
python3 reference_full_block.py

echo "Profiling with nsys (system-level trace, kernel-launch timing)..."
nsys profile -o "full_block_nsys_${SLURM_JOB_ID}" ./full_block_gpu_kernel_a100

echo "Profiling with ncu (kernel-level DRAM traffic counters)..."
ncu --set full -o "full_block_ncu_${SLURM_JOB_ID}" ./full_block_gpu_kernel_a100

echo "Done. Profiler output: full_block_nsys_${SLURM_JOB_ID}.nsys-rep, full_block_ncu_${SLURM_JOB_ID}.ncu-rep"
echo "Next step (not automated here): extract 'dram__bytes_read.sum' and"
echo "'dram__bytes_write.sum' from the ncu report per kernel, sum across"
echo "the full forward+backward call, and compare against M_block = 24nd + 4nd_ff"
echo "at this run's B, N, D, D_ff (see EXPERIMENT_PLAN.md Section 6)."
