#!/bin/bash
#SBATCH --job-name=moa-profile-paper2-gk
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=moa_profile_paper2_gk_%j.out
#SBATCH --error=moa_profile_paper2_gk_%j.err
#
# Tests the specific hypothesis raised in results_paper/moa_experimental_
# validation.tex Section 4.6: that fused_attn_gpu_bench.c's N>=4096
# slowdown after removing GK's atomic (job 21294299, confirmed real:
# 1.04x at N=4096, 1.32x at N=8192) is caused by the new Pass 5 (GK,
# atomic-free) re-reading the full GS array a second time -- added
# memory traffic that outweighs the removed atomic-contention cost at
# large N, though not at small/medium N where the fix still wins.
#
# Profiles two variants of the SAME kernel at N=4096 (where the real
# slowdown was measured) and compares DRAM traffic directly:
#   - fused_attn_gpu_bench.c (v4, current, atomic-free, extra GS re-read)
#   - fused_attn_gpu_bench_v3atomic.c (v3 reconstruction, GK atomic,
#     no extra re-read -- built solely for this comparison, not a
#     replacement for the real file)
# Both verified to exact 0.000e+00 correctness before this script was
# written.
#
# If the hypothesis is correct: v4 should show measurably more DRAM
# bytes moved than v3atomic at N=4096. If DRAM traffic is similar
# despite the real timing difference, the slowdown's cause lies
# elsewhere (e.g. occupancy, launch overhead of the extra kernel) and
# this hypothesis should be considered wrong, not just unconfirmed.
#
# v2 of this script: the first attempt used `ncu --set full`, which
# OOM-killed (SIGKILL, exit 9) partway into v4's very first profiling
# pass. Root cause: `--set full`'s comprehensive metric collection
# needs ~50+ replay passes per kernel, each requiring the profiler to
# back up and restore full GPU memory state -- at N=4096, AROW and GS
# alone are ~256MB each, and that backup overhead compounds across
# every pass. Switched to `--metrics <specific list>`, which needs far
# fewer passes since only the requested counters must be scheduled,
# reducing replay-buffer memory pressure substantially. Not yet
# confirmed this avoids the OOM at N=4096 -- this is the first attempt
# with the fix applied.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Copy or clone both fused_attn_gpu_bench.c AND
#     fused_attn_gpu_bench_v3atomic.c into the submission directory.
#
# Usage: sbatch delta_profile_paper2_gk.sbatch [B] [N] [D]

module purge
module load nvhpc/26.5
set -euo pipefail

MISSING_FILES=()
[ -f "fused_attn_gpu_bench.c" ] || MISSING_FILES+=("fused_attn_gpu_bench.c")
[ -f "fused_attn_gpu_bench_v3atomic.c" ] || MISSING_FILES+=("fused_attn_gpu_bench_v3atomic.c")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  exit 1
fi

B=${1:-2}
N=${2:-4096}
D=${3:-64}

echo "Delta gpuA100x4. Job $SLURM_JOB_ID. Profiling fused at B=$B N=$N D=$D, both variants."
which nsys ncu || echo "WARNING: nsys/ncu not found on PATH"

echo "Building v4 (current, atomic-free, extra GS re-read)..."
nvc -O2 -acc -Minfo=accel fused_attn_gpu_bench.c -o fused_v4_profile -lm

echo "Building v3atomic (GK atomic, no extra re-read)..."
nvc -O2 -acc -Minfo=accel fused_attn_gpu_bench_v3atomic.c -o fused_v3atomic_profile -lm

srun --ntasks-per-node 1 dcgmi profile --pause

echo "=== Profiling v4 (atomic-free) at N=$N ==="
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum -o "paper2gk_v4_ncu_${SLURM_JOB_ID}" ./fused_v4_profile "$B" "$N" "$N" "$D" 1

echo "=== Profiling v3atomic (GK atomic) at N=$N ==="
ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum -o "paper2gk_v3atomic_ncu_${SLURM_JOB_ID}" ./fused_v3atomic_profile "$B" "$N" "$N" "$D" 1

srun --ntasks-per-node 1 dcgmi profile --resume

echo "Done. Profiler output:"
echo "  paper2gk_v4_ncu_${SLURM_JOB_ID}.ncu-rep (atomic-free, current)"
echo "  paper2gk_v3atomic_ncu_${SLURM_JOB_ID}.ncu-rep (GK atomic, reconstruction)"
echo "Next step (not automated here): extract dram__bytes_read.sum and"
echo "dram__bytes_write.sum from both reports (careful: Mbyte/Kbyte"
echo "pre-scaling, not raw bytes) and compare totals."
