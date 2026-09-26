#!/bin/bash
# Sweeps problem size (sequence length N, doubling from N_START to N_END) and
# thread count (doubling from 1 to MAX_THREADS, auto-detected via nproc unless
# overridden) across CPU/OpenMP benchmarks: rmsnorm and mlp (C and Fortran90),
# plus the complete transformer block (C only -- see note below).
#
# Usage: ./run_cpu_sweep.sh [N_START] [N_END] [B] [D] [DFF] [MAX_THREADS]
# Defaults: N_START=64 N_END=4096 B=2 D=64 DFF=256 MAX_THREADS=$(nproc)
#
# Output: results_cpu.csv, columns: impl,lang,B,N,D,DFF,threads,seconds
#
# NOTE: full_block_bench includes attention, whose cost is O(N^2) in the
# sequence length (score matrix), unlike rmsnorm/mlp which are O(N). At
# large N_END this sublayer will dominate total sweep time; reduce N_END
# for the full-block portion if that's a problem on your machine (see
# FULL_BLOCK_N_END below).
#
# NOTE: full_block_bench.f90 does not exist yet (Fortran90 counterpart to
# full_block_verify.c/full_block_gpu_kernel.c is listed as remaining work
# in Section 10 of the paper) -- only the C full-block benchmark runs here.

set -e

N_START=${1:-64}
N_END=${2:-4096}
B=${3:-2}
D=${4:-64}
DFF=${5:-256}
MAX_THREADS=${6:-$(nproc)}
FULL_BLOCK_N_END=${7:-$N_END}

OUT=results_cpu.csv
echo "impl,lang,B,N,D,DFF,threads,seconds" > "$OUT"

echo "Building..."
gcc -O2 -fopenmp rmsnorm_bench.c -o rmsnorm_bench -lm
gcc -O2 -fopenmp mlp_bench.c     -o mlp_bench -lm
gfortran -O2 -fopenmp rmsnorm_bench.f90 -o rmsnorm_bench_f90
gfortran -O2 -fopenmp mlp_bench.f90     -o mlp_bench_f90
gcc -O2 -fopenmp full_block_bench.c -o full_block_bench -lm

echo "Sweeping N: $N_START -> $N_END (doubling), threads: 1 -> $MAX_THREADS (doubling)"
echo "Fixed: B=$B D=$D DFF=$DFF. Full-block N capped at $FULL_BLOCK_N_END."

N=$N_START
while [ "$N" -le "$N_END" ]; do
  T=1
  LAST_T=0
  while [ "$T" -le "$MAX_THREADS" ]; do
    echo "  N=$N threads=$T"
    ./rmsnorm_bench    "$B" "$N" "$D"        "$T" >> "$OUT"
    ./mlp_bench        "$B" "$N" "$D" "$DFF" "$T" >> "$OUT"
    ./rmsnorm_bench_f90 "$B" "$N" "$D"        "$T" | tr -d ' ' >> "$OUT"
    ./mlp_bench_f90     "$B" "$N" "$D" "$DFF" "$T" | tr -d ' ' >> "$OUT"
    if [ "$N" -le "$FULL_BLOCK_N_END" ]; then
      ./full_block_bench "$B" "$N" "$D" "$DFF" "$T" >> "$OUT"
    fi
    LAST_T=$T
    T=$((T * 2))
  done
  # doubling may not land exactly on MAX_THREADS (e.g. 6 cores -> 1,2,4 skips 6);
  # always include the true max as a final explicit run.
  if [ "$LAST_T" -ne "$MAX_THREADS" ]; then
    echo "  N=$N threads=$MAX_THREADS (max, explicit)"
    ./rmsnorm_bench    "$B" "$N" "$D"        "$MAX_THREADS" >> "$OUT"
    ./mlp_bench        "$B" "$N" "$D" "$DFF" "$MAX_THREADS" >> "$OUT"
    ./rmsnorm_bench_f90 "$B" "$N" "$D"        "$MAX_THREADS" | tr -d ' ' >> "$OUT"
    ./mlp_bench_f90     "$B" "$N" "$D" "$DFF" "$MAX_THREADS" | tr -d ' ' >> "$OUT"
    if [ "$N" -le "$FULL_BLOCK_N_END" ]; then
      ./full_block_bench "$B" "$N" "$D" "$DFF" "$MAX_THREADS" >> "$OUT"
    fi
  fi
  N=$((N * 2))
done

echo "Done. Results in $OUT"
