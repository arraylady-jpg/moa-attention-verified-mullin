#!/bin/bash
#SBATCH --job-name=moa-delta-paper3-cpu
#SBATCH --account=bibg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --output=moa_delta_paper3_cpu_%j.out
#SBATCH --error=moa_delta_paper3_cpu_%j.err
#
# Paper III, Delta CPU: decode kernel. Direct port of
# anvil_paper3_decode_sweep.sbatch, using the pre-existing
# moa_decode_bench.c (no new code needed, same as the Anvil version).
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-cpu` and `cpu` partition with sacctmgr/sinfo.
#   - Copy or clone attention/moa_decode_bench.c first.
#
# Usage: sbatch delta_paper3_cpu_sweep.sbatch [n_start] [n_max] [dk] [dv] [repeats]

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
[ -f "moa_decode_bench.c" ] || MISSING_FILES+=("moa_decode_bench.c(from attention/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-1024}
N_MAX=${2:-1048576}
DK=${3:-64}
DV=${4:-64}
REPEATS=${5:-5}

echo "Delta cpu partition, 128 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Decode sweep: n=$N_START -> $N_MAX (doubling), dk=$DK, dv=$DV, threads 1 -> 128 (doubling)"

gcc -O3 -fopenmp -o moa_decode_bench moa_decode_bench.c -lm

./moa_decode_bench 128 "$N_START" "$N_MAX" "$DK" "$DV" "$REPEATS"

mv timings_cpu.csv "timings_cpu_paper3_delta_${SLURM_JOB_ID}.csv"
echo "Done. Results in timings_cpu_paper3_delta_${SLURM_JOB_ID}.csv"
echo "NOTE: this file has no 'machine' column (unmodified original source);"
echo "tag it delta-cpu manually when combining with Anvil results."
