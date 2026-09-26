#!/bin/bash
#SBATCH --job-name=moa-delta-paper4-cpu
#SBATCH --account=bibg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=moa_delta_paper4_cpu_%j.out
#SBATCH --error=moa_delta_paper4_cpu_%j.err
#
# Paper IV, Delta CPU: RMSNorm, gated MLP (C+Fortran90), and the
# complete block (C only). Direct port of anvil_cpu_sweep.sbatch,
# calling the existing benchmarking/run_cpu_sweep.sh unmodified.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-cpu` and `cpu` partition with sacctmgr/sinfo.
#   - Copy or clone benchmarking/*.c, *.f90, run_cpu_sweep.sh first.
#
# Usage: sbatch delta_paper4_cpu_sweep.sbatch [N_START] [N_END] [B] [D] [DFF] [FULL_BLOCK_N_END]

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

echo "Delta cpu partition, 128 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Sweeping N: $N_START -> $N_END, threads: 1 -> 128, full-block N capped at $FULL_BLOCK_N_END"

chmod +x run_cpu_sweep.sh
./run_cpu_sweep.sh "$N_START" "$N_END" "$B" "$D" "$DFF" 128 "$FULL_BLOCK_N_END"

mv results_cpu.csv "results_paper4_delta_cpu_${SLURM_JOB_ID}.csv"
echo "Done. Results in results_paper4_delta_cpu_${SLURM_JOB_ID}.csv"
echo "NOTE: this file has no 'machine' column (unmodified run_cpu_sweep.sh output);"
echo "tag it delta-cpu manually when combining with Anvil results."
