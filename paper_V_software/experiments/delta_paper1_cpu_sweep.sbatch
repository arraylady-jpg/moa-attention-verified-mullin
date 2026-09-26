#!/bin/bash
#SBATCH --job-name=moa-delta-paper1-cpu
#SBATCH --account=bibg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=moa_delta_paper1_cpu_%j.out
#SBATCH --error=moa_delta_paper1_cpu_%j.err
#
# Paper I, Delta CPU: forward pass. Direct port of anvil_paper1_cpu_sweep.sbatch
# -- same forward_bench.c, same math, different cluster. Delta's `cpu`
# partition nodes are 128 cores (confirmed via sinfo on dt-login03).
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-cpu` is still your correct account with
#     `sacctmgr show assoc user=$USER format=account,partition -p`.
#   - Confirm `cpu` partition core count with `sinfo -p cpu`.
#   - Copy or clone forward_bench.c into the submission directory first.
#
# Usage: sbatch delta_paper1_cpu_sweep.sbatch [N_START] [N_END] [B] [D]

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
[ -f "forward_bench.c" ] || MISSING_FILES+=("forward_bench.c(from experiments/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  echo "Copy them from the package (note: not all live in this experiments/ folder"
  echo "-- some are in attention/, benchmarking/, or verification/) before resubmitting."
  exit 1
fi

N_START=${1:-64}
N_END=${2:-8192}
B=${3:-2}
D=${4:-64}

echo "Delta cpu partition, 128 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Sweeping N: $N_START -> $N_END (doubling), threads: 1 -> 128 (doubling)"

gcc -O2 -fopenmp forward_bench.c -o forward_bench -lm

OUT="results_paper1_delta_cpu_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,threads,seconds,machine" > "$OUT"

N=$N_START
while [ "$N" -le "$N_END" ]; do
  T=1
  LAST_T=0
  while [ "$T" -le 128 ]; do
    echo "  N=$N threads=$T"
    ROW=$(./forward_bench "$B" "$N" "$D" "$T")
    echo "${ROW},delta-cpu" >> "$OUT"
    LAST_T=$T
    T=$((T * 2))
  done
  if [ "$LAST_T" -ne 128 ]; then
    ROW=$(./forward_bench "$B" "$N" "$D" 128)
    echo "${ROW},delta-cpu" >> "$OUT"
  fi
  N=$((N * 2))
done

echo "Done. Results in $OUT"
