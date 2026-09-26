#!/bin/bash
#SBATCH --job-name=moa-delta-paper2-cpu
#SBATCH --account=bibg-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --output=moa_delta_paper2_cpu_%j.out
#SBATCH --error=moa_delta_paper2_cpu_%j.err
#
# Paper II, Delta CPU: backward (Algorithm 1) and fused (Algorithm 2).
# Direct port of anvil_paper2_cpu_sweep.sbatch.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Confirm `bibg-delta-cpu` and `cpu` partition with sacctmgr/sinfo.
#   - Copy or clone backward_bench.c and fused_bench.c first.
#
# Usage: sbatch delta_paper2_cpu_sweep.sbatch [N_START] [N_END] [B] [D]

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
[ -f "backward_bench.c" ] || MISSING_FILES+=("backward_bench.c(from experiments/)")
[ -f "fused_bench.c" ] || MISSING_FILES+=("fused_bench.c(from experiments/)")
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
echo "NOTE: backward's A array is O(N^2) -- may take longer per step than Paper IV's O(N) kernels."

gcc -O2 -fopenmp backward_bench.c -o backward_bench -lm
gcc -O2 -fopenmp fused_bench.c -o fused_bench -lm

OUT="results_paper2_delta_cpu_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,threads,seconds,machine" > "$OUT"

N=$N_START
while [ "$N" -le "$N_END" ]; do
  T=1
  LAST_T=0
  while [ "$T" -le 128 ]; do
    echo "  N=$N threads=$T"
    ROW1=$(./backward_bench "$B" "$N" "$D" "$T"); echo "${ROW1},delta-cpu" >> "$OUT"
    ROW2=$(./fused_bench     "$B" "$N" "$D" "$T"); echo "${ROW2},delta-cpu" >> "$OUT"
    LAST_T=$T
    T=$((T * 2))
  done
  if [ "$LAST_T" -ne 128 ]; then
    ROW1=$(./backward_bench "$B" "$N" "$D" 128); echo "${ROW1},delta-cpu" >> "$OUT"
    ROW2=$(./fused_bench     "$B" "$N" "$D" 128); echo "${ROW2},delta-cpu" >> "$OUT"
  fi
  N=$((N * 2))
done

echo "Done. Results in $OUT"
