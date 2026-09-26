#!/bin/bash
#SBATCH --job-name=moa-stall-compare
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=moa_stall_compare_%j.out
#SBATCH --error=moa_stall_compare_%j.err
#
# Follow-up to delta_sass_compare.sbatch (job 21299788), which found
# Fortran generates MORE SASS instructions in every category than C
# (11.4% more total) despite running FASTER on real GPU hardware
# (Paper IV Section 6.2, 1.45-2.81x) -- ruling out a simple
# instruction-count explanation without providing a replacement one.
#
# This profiles WHERE each language's kernel actually spends its
# cycles, via ncu's per-warp stall-reason breakdown, rather than just
# what instructions exist. Key metrics:
#   - long_scoreboard: warps stalled waiting on global memory (the
#     classic "memory latency" stall -- if Fortran hides this better
#     despite more instructions, that's a real, different explanation
#     than instruction count)
#   - short_scoreboard: stalled waiting on shared memory/MIO
#   - mio_throttle: memory-I/O pipe saturated
#   - not_selected: warp is eligible to issue but scheduler picked a
#     different one (usually a GOOD sign -- means enough warps are
#     ready to hide latency)
#   - no_instruction: warp scheduler has nothing ready to issue at all
#     (usually a BAD sign -- low occupancy or poor instruction-level
#     parallelism)
#   - achieved occupancy, for context
#
# Learned from job 21299053's OOM under --set full at N=4096: this
# uses a small, targeted --metrics list from the start, not the
# comprehensive set, to avoid the same replay-buffer memory pressure.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Copy or clone both fused_gpu_bench.c and fused_gpu_bench.f90
#     into the submission directory (from benchmarking/).
#
# Usage: sbatch delta_stall_compare.sbatch [B] [N] [D] [DFF]

module purge
module load nvhpc/26.5
set -euo pipefail

MISSING_FILES=()
[ -f "fused_gpu_bench.c" ] || MISSING_FILES+=("fused_gpu_bench.c(from benchmarking/)")
[ -f "fused_gpu_bench.f90" ] || MISSING_FILES+=("fused_gpu_bench.f90(from benchmarking/)")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: required source file(s) not found in $(pwd):"
  for item in "${MISSING_FILES[@]}"; do echo "  - $item"; done
  exit 1
fi

B=${1:-2}
N=${2:-1024}
D=${3:-64}
DFF=${4:-256}

METRICS="smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio,smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.sum"

echo "Delta gpuA100x4. Job $SLURM_JOB_ID. Comparing stall reasons for C vs Fortran fused kernel at N=$N."
which ncu || echo "WARNING: ncu not found on PATH"

echo "=== Building C version ==="
nvc -O2 -acc -Minfo=accel fused_gpu_bench.c -o fused_c_stall -lm

echo "=== Building Fortran version ==="
nvfortran -O2 -acc -Minfo=accel fused_gpu_bench.f90 -o fused_f90_stall

srun --ntasks-per-node 1 dcgmi profile --pause

echo "=== Profiling C at N=$N ==="
ncu --metrics "$METRICS" -o "stall_c_${SLURM_JOB_ID}" ./fused_c_stall "$B" "$N" "$N" "$D" "$DFF"

echo "=== Profiling Fortran at N=$N ==="
ncu --metrics "$METRICS" -o "stall_f90_${SLURM_JOB_ID}" ./fused_f90_stall "$B" "$N" "$N" "$D" "$DFF"

srun --ntasks-per-node 1 dcgmi profile --resume

echo "Done. Profiler output: stall_c_${SLURM_JOB_ID}.ncu-rep, stall_f90_${SLURM_JOB_ID}.ncu-rep"
echo "Next step (not automated here): ncu --import <file> --page raw --csv > dump.csv,"
echo "then extract and compare the stall-reason columns between the two."
