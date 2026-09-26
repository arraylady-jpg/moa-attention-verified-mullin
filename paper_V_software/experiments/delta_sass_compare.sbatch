#!/bin/bash
#SBATCH --job-name=moa-sass-compare
#SBATCH --account=bibg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=moa_sass_compare_%j.out
#SBATCH --error=moa_sass_compare_%j.err
#
# Investigates the open question from results_paper/moa_experimental_
# validation.tex Section 6.1-6.2: the same denotational fused norm+MLP
# computation runs faster in C than Fortran on CPU (1.2-1.85x), but
# faster in Fortran than C on GPU (1.45-2.81x) -- a real, confirmed
# reversal never diagnosed to a specific cause. Unlike every other
# investigation in this project, this one requires inspecting actual
# generated machine code (SASS), not timing or DRAM-traffic profiling,
# since the question is about code generation itself, not runtime
# behavior of a fixed kernel.
#
# Extracts SASS (compiled GPU machine code) from both fused_gpu_bench.c
# and fused_gpu_bench.f90's real GPU binaries via cuobjdump, then
# produces a simple instruction-category breakdown (memory ops, FMA,
# branches, total count) per kernel, since raw SASS is too large to
# usefully diff by eye. This is a coarse first look, not a definitive
# diagnosis -- if the category counts look similar despite the real
# timing difference, the cause likely lies in something SASS-level
# counting won't show (e.g. instruction scheduling, register bank
# conflicts, memory access pattern within otherwise-similar
# instruction mix), and this would need a different tool (ncu's
# per-instruction stall-reason breakdown) to go further.
#
# ACTION NEEDED BEFORE SUBMITTING:
#   - Copy or clone both fused_gpu_bench.c and fused_gpu_bench.f90
#     into the submission directory (from benchmarking/).
#
# Usage: sbatch delta_sass_compare.sbatch [B] [N] [D] [DFF]

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
N=${2:-256}
D=${3:-64}
DFF=${4:-256}

echo "Delta gpuA100x4. Job $SLURM_JOB_ID. Comparing SASS for C vs Fortran fused kernel."
which cuobjdump nvdisasm || echo "WARNING: cuobjdump/nvdisasm not found on PATH"

echo "=== Building C version ==="
nvc -O2 -acc -Minfo=accel fused_gpu_bench.c -o fused_c_sass -lm

echo "=== Building Fortran version ==="
nvfortran -O2 -acc -Minfo=accel fused_gpu_bench.f90 -o fused_f90_sass

echo "=== Extracting SASS from C binary ==="
cuobjdump --dump-sass fused_c_sass > sass_c_${SLURM_JOB_ID}.txt 2>&1 || echo "cuobjdump on C binary failed, see above"

echo "=== Extracting SASS from Fortran binary ==="
cuobjdump --dump-sass fused_f90_sass > sass_f90_${SLURM_JOB_ID}.txt 2>&1 || echo "cuobjdump on Fortran binary failed, see above"

echo "=== Instruction category counts: C ==="
echo "Total instructions: $(grep -cE '^\s+/\*[0-9a-f]+\*/' sass_c_${SLURM_JOB_ID}.txt || echo 0)"
echo "FFMA/DFMA (fused multiply-add): $(grep -cE '(FFMA|DFMA)' sass_c_${SLURM_JOB_ID}.txt || echo 0)"
echo "LDG/STG (global memory load/store): $(grep -cE '(LDG|STG)' sass_c_${SLURM_JOB_ID}.txt || echo 0)"
echo "LDS/STS (shared memory load/store): $(grep -cE '(LDS|STS)' sass_c_${SLURM_JOB_ID}.txt || echo 0)"
echo "Branches (BRA/BSSY/BSYNC): $(grep -cE '(BRA|BSSY|BSYNC)' sass_c_${SLURM_JOB_ID}.txt || echo 0)"

echo "=== Instruction category counts: Fortran ==="
echo "Total instructions: $(grep -cE '^\s+/\*[0-9a-f]+\*/' sass_f90_${SLURM_JOB_ID}.txt || echo 0)"
echo "FFMA/DFMA (fused multiply-add): $(grep -cE '(FFMA|DFMA)' sass_f90_${SLURM_JOB_ID}.txt || echo 0)"
echo "LDG/STG (global memory load/store): $(grep -cE '(LDG|STG)' sass_f90_${SLURM_JOB_ID}.txt || echo 0)"
echo "LDS/STS (shared memory load/store): $(grep -cE '(LDS|STS)' sass_f90_${SLURM_JOB_ID}.txt || echo 0)"
echo "Branches (BRA/BSSY/BSYNC): $(grep -cE '(BRA|BSSY|BSYNC)' sass_f90_${SLURM_JOB_ID}.txt || echo 0)"

echo "Done. Full SASS dumps: sass_c_${SLURM_JOB_ID}.txt, sass_f90_${SLURM_JOB_ID}.txt"
echo "(These can be large -- check file size before trying to paste them directly.)"
