#!/bin/bash
#SBATCH --job-name=moa-profile-paper2
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=moa_profile_paper2_%j.out
#SBATCH --error=moa_profile_paper2_%j.err
# Profiles backward_gpu_bench.c and fused_attn_gpu_bench.c at the same N,
# to test why fused was slower than naive on real GPU data (jobs 21225789/90).
# See EXPERIMENT_PLAN.md and results_paper/moa_experimental_validation.tex
# Section 4.3 for the full hypothesis (atomic contention on GV+GK vs GK only).

module purge
module load nvhpc/26.5
set -euo pipefail

MISSING_FILES=()
[ -f "backward_gpu_bench.c" ] || MISSING_FILES+=("backward_gpu_bench.c")
[ -f "fused_attn_gpu_bench.c" ] || MISSING_FILES+=("fused_attn_gpu_bench.c")
if [ ${#MISSING_FILES[@]} -gt 0 ]; then
  echo "ERROR: missing source file(s): ${MISSING_FILES[@]}"
  exit 1
fi

B=${1:-2}
N=${2:-2048}
D=${3:-64}

echo "Delta gpuA100x4. Job $SLURM_JOB_ID. Profiling at B=$B N=$N D=$D"
which nsys ncu || echo "WARNING: nsys/ncu not on PATH"

nvc -O2 -acc -Minfo=accel backward_gpu_bench.c -o backward_gpu_bench_profile -lm
nvc -O2 -acc -Minfo=accel fused_attn_gpu_bench.c -o fused_attn_gpu_bench_profile -lm

srun --ntasks-per-node 1 dcgmi profile --pause

echo "=== backward (atomics: GK only) ==="
nsys profile -o "paper2_backward_nsys_${SLURM_JOB_ID}" ./backward_gpu_bench_profile "$B" "$N" "$N" "$D" 1
ncu --set full -o "paper2_backward_ncu_${SLURM_JOB_ID}" ./backward_gpu_bench_profile "$B" "$N" "$N" "$D" 1

echo "=== fused (atomics: GV and GK) ==="
nsys profile -o "paper2_fused_nsys_${SLURM_JOB_ID}" ./fused_attn_gpu_bench_profile "$B" "$N" "$N" "$D" 1
ncu --set full -o "paper2_fused_ncu_${SLURM_JOB_ID}" ./fused_attn_gpu_bench_profile "$B" "$N" "$N" "$D" 1

srun --ntasks-per-node 1 dcgmi profile --resume

echo "Done. Output: paper2_backward_ncu_${SLURM_JOB_ID}.ncu-rep, paper2_fused_ncu_${SLURM_JOB_ID}.ncu-rep"
