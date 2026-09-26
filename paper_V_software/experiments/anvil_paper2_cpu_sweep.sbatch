#!/bin/bash
#SBATCH --job-name=moa-paper2-sweep
#SBATCH --account=cis261396-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=moa_paper2_sweep_%j.out
#SBATCH --error=moa_paper2_sweep_%j.err
#
# Paper II: backward pass (Algorithm 1) and fused forward+backward
# (Algorithm 2) -- the series' most load-bearing result, cited by
# Papers III and IV. Validates M_bwd = n^2 + 4nd_k + 3nd_v and
# M_fwd+bwd = 4nd_k + 4nd_v (d_k = d_v = D here, single head).
#
# backward_bench.c and fused_bench.c were built this session by porting
# moa_backward_dnf and moa_fused_dnf out of attention/moa_attention.c
# into parameterized, timed, OpenMP-parallel harnesses -- same math,
# verified to match a serial/composed reference to machine precision
# (0.000e+00 to ~1e-16) at every thread count tested (1,2,4,8) before
# this script was written.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Replace `cis261396` above if your allocation ID differs.
#   - `ai` partition (NOT `wholenode`) and account `cis261396-ai`
#     -- corrected after Anvil rejected a real `wholenode` submission
#     ("You are submitting a job to a non-AI partition while using an
#     AI allocation"). A second real rejection followed (`QOSMinGRES`):
#     `scontrol show partition ai` revealed the `ai` partition IS the
#     H100 GPU nodes (`Nodes=h[000-020]`, same hardware as the `gpu`
#     partition), billed by GPU (`TRESBillingWeights=GRES/gpu=1.0`),
#     under a QOS (`part-ai`) that requires every job -- even this
#     CPU-only OpenMP one -- to request at least one GPU. Fixed with
#     `--gres=gpu:1` below (the GPU sits idle; only CPU cores are used
#     by this program). See EXPERIMENT_PLAN.md Section 2 for the full story.
#   - Copy or clone backward_bench.c and fused_bench.c into the
#     submission directory first.
#
# Usage: sbatch anvil_paper2_cpu_sweep.sbatch [N_START] [N_END] [B] [D]

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

echo "Anvil ai partition, 96 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Sweeping N: $N_START -> $N_END (doubling), threads: 1 -> 96 (doubling)"
echo "Fixed: B=$B D=$D. NOTE: backward's A array is O(N^2) -- this sweep"
echo "will take substantially longer per step than Paper IV's O(N) kernels"
echo "as N grows; consider a smaller N_END than the default if time-limited."

gcc -O2 -fopenmp backward_bench.c -o backward_bench -lm
gcc -O2 -fopenmp fused_bench.c -o fused_bench -lm

OUT="results_paper2_anvil_${SLURM_JOB_ID}.csv"
echo "impl,lang,B,N,D,threads,seconds" > "$OUT"

N=$N_START
while [ "$N" -le "$N_END" ]; do
  T=1
  LAST_T=0
  while [ "$T" -le 96 ]; do
    echo "  N=$N threads=$T"
    ./backward_bench "$B" "$N" "$D" "$T" >> "$OUT"
    ./fused_bench     "$B" "$N" "$D" "$T" >> "$OUT"
    LAST_T=$T
    T=$((T * 2))
  done
  if [ "$LAST_T" -ne 96 ]; then
    ./backward_bench "$B" "$N" "$D" 96 >> "$OUT"
    ./fused_bench     "$B" "$N" "$D" 96 >> "$OUT"
  fi
  N=$((N * 2))
done

echo "Done. Results in $OUT"
