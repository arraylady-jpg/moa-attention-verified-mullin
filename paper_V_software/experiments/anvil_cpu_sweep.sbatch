#!/bin/bash
#SBATCH --job-name=moa-cpu-sweep
#SBATCH --account=cis261396-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=moa_cpu_sweep_%j.out
#SBATCH --error=moa_cpu_sweep_%j.err
#
# Experiment A: CPU OpenMP scaling, real multi-core speedup.
# Runs the existing benchmarking/run_cpu_sweep.sh at Anvil scale: thread
# count 1 -> 96 doubling (a full 'ai' partition node's 96 cores), sequence length
# N = 64 -> 65536 doubling.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396` above if your allocation ID differs.
#   - `ai` partition (NOT `wholenode` -- corrected after Anvil rejected
#     a real `wholenode` submission with "You are submitting a job to
#     a non-AI partition while using an AI allocation") and account
#     `cis261396-ai` (no bare `cis261396` account exists). `ai` nodes
#     have 96 CPUs, not `wholenode`'s 128. A second real rejection
#     followed (`QOSMinGRES`): `scontrol show partition ai` revealed
#     the `ai` partition IS the H100 GPU nodes (`Nodes=h[000-020]`,
#     same hardware as the `gpu` partition), billed by GPU
#     (`TRESBillingWeights=GRES/gpu=1.0`), under a QOS (`part-ai`) that
#     requires every job -- even this CPU-only one -- to request at
#     least one GPU. Fixed with `--gres=gpu:1` below (the GPU sits
#     idle; only CPU cores are used). See EXPERIMENT_PLAN.md Section 2
#     for the full story.
#   - Copy or clone benchmarking/*.c, *.f90, run_cpu_sweep.sh into the
#     submission directory before running.
#
# Usage: sbatch anvil_cpu_sweep.sbatch [N_START] [N_END] [B] [D] [DFF] [FULL_BLOCK_N_END]

module purge
module load gcc

# Fail loudly on any error (missing source file, failed compile, failed run)
# instead of silently continuing to write garbage/empty CSV rows -- this is
# exactly the class of bug that produced empty rows in an earlier real run
# (source .c files weren't copied to the submission directory; the script
# kept going past the compile failure and wrote blank CSV lines instead of
# stopping). Copy/clone the required source files into this directory
# BEFORE submitting -- see the ACTION NEEDED section above.
set -euo pipefail

MISSING_FILES=()
[ -f "run_cpu_sweep.sh" ] || MISSING_FILES+=("run_cpu_sweep.sh(from benchmarking/)")
[ -f "rmsnorm_bench.c" ] || MISSING_FILES+=("rmsnorm_bench.c(from benchmarking/)")
[ -f "mlp_bench.c" ] || MISSING_FILES+=("mlp_bench.c(from benchmarking/)")
[ -f "rmsnorm_bench.f90" ] || MISSING_FILES+=("rmsnorm_bench.f90(from benchmarking/)")
[ -f "mlp_bench.f90" ] || MISSING_FILES+=("mlp_bench.f90(from benchmarking/)")
[ -f "full_block_bench.c" ] || MISSING_FILES+=("full_block_bench.c(from benchmarking/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-64}
N_END=${2:-65536}
B=${3:-2}
D=${4:-64}
DFF=${5:-256}
FULL_BLOCK_N_END=${6:-4096}

echo "Anvil ai partition, 96 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Sweeping N: $N_START -> $N_END, threads: 1 -> 96, full-block N capped at $FULL_BLOCK_N_END"

chmod +x run_cpu_sweep.sh
./run_cpu_sweep.sh "$N_START" "$N_END" "$B" "$D" "$DFF" 96 "$FULL_BLOCK_N_END"

mv results_cpu.csv "results_cpu_anvil_${SLURM_JOB_ID}.csv"
echo "Done. Results in results_cpu_anvil_${SLURM_JOB_ID}.csv"
