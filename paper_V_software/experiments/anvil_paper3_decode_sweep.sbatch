#!/bin/bash
#SBATCH --job-name=moa-paper3-decode-sweep
#SBATCH --account=cis261396-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=moa_paper3_decode_%j.out
#SBATCH --error=moa_paper3_decode_%j.err
#
# Paper III: decode-time attention benchmark. Unlike Papers I/II (no
# bench code exists yet) and Paper IV (bench code exists but needed
# building this session), this one requires ZERO new code --
# attention/moa_decode_bench.c is already a complete, working,
# parameterized, self-verifying OpenMP benchmark. This script is a
# thin sbatch wrapper around the existing binary, nothing more.
#
# Decode's per-token cost is O(n) in cached sequence length (not O(n^2)
# like Paper IV's attention-forward sweep), so this can push n much
# larger before becoming impractical -- default sweep goes to n=1048576.
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
#   - Copy or clone attention/moa_decode_bench.c into the submission
#     directory first.
#
# Usage: sbatch anvil_paper3_decode_sweep.sbatch [n_start] [n_max] [dk] [dv] [repeats]

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

echo "Anvil ai partition, 96 cores. Job $SLURM_JOB_ID on $SLURM_NODELIST"
echo "Decode sweep: n=$N_START -> $N_MAX (doubling), dk=$DK, dv=$DV, threads 1 -> 96 (doubling)"

gcc -O3 -fopenmp -o moa_decode_bench moa_decode_bench.c -lm

./moa_decode_bench 96 "$N_START" "$N_MAX" "$DK" "$DV" "$REPEATS"

mv timings_cpu.csv "timings_cpu_paper3_anvil_${SLURM_JOB_ID}.csv"
echo "Done. Results in timings_cpu_paper3_anvil_${SLURM_JOB_ID}.csv"
