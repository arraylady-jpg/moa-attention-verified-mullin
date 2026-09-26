# Benchmarking: CPU (sequential/OpenMP) and GPU (OpenACC/SLURM) sweeps

These are performance-timing benchmarks (dynamic sizes, no file I/O), distinct
from the correctness-verification programs in ../verification/ (fixed
B=2,N=4,D=8,D_ff=16, checked against PyTorch to machine precision). The
computational kernels here are identical to the verified ones; only the
random input generation and the timing/CLI wrapper differ.

## CPU: sequential and OpenMP ("multiprocessing"), C and Fortran90

    ./run_cpu_sweep.sh [N_START] [N_END] [B] [D] [DFF] [MAX_THREADS]
    # defaults:        64        4096    2  64  256   $(nproc)

Sweeps sequence length N by DOUBLING from N_START to N_END, and thread count
by DOUBLING from 1 (sequential/single-thread baseline) to MAX_THREADS (full
"multiprocessing" via OpenMP), auto-detected via `nproc` unless overridden.
If MAX_THREADS isn't a power of 2, an explicit final run at the true max is
still included so you always see the real ceiling, not just the nearest
power of 2 below it.

NOTE ON "SEQUENTIAL": the threads=1 row uses the OpenMP code path with one
thread, not a separately-compiled serial binary. This is standard practice
and the OpenMP runtime overhead at 1 thread is negligible, but if you want a
true no-OpenMP-runtime serial baseline for the paper's numbers specifically,
say so and I'll add `_serial.c`/`.f90` variants with the pragmas stripped.

Runs all four programs (rmsnorm x {C,Fortran}, mlp x {C,Fortran}) at every
(N, threads) combination and appends one CSV row each to `results_cpu.csv`:

    impl,lang,B,N,D,DFF,threads,seconds

DFF is 0 for rmsnorm rows (not applicable) so the schema stays uniform across
both benchmarks.

Doubling was chosen for both N and thread count over an arithmetic step
(N+=k, threads+=2) since geometric/doubling sweeps are the HPC-standard
convention for scaling studies (log-log speedup plots). Say so if you
actually wanted arithmetic steps and I'll change the loop increment.

## GPU: OpenACC via SLURM

    sbatch slurm_gpu_sweep.sbatch [N_START] [N_END] [B] [D] [DFF]

Sweeps N by doubling on a single allocated GPU (`--gres=gpu:1`), building
with `nvc`/`nvfortran` (NVIDIA HPC SDK) and running the fused
RMSNorm+gated-MLP OpenACC kernel in both C and Fortran. Writes
`results_gpu_<jobid>.csv` with columns `impl,lang,B,N,D,DFF,seconds`.

Adjust the `module load` line for your cluster's actual module name if it
isn't `nvhpc`.

IMPORTANT LIMITATION: this sweeps SIZE ONLY, not GPU count. The kernels here
are single-GPU, single-kernel-launch programs with no multi-GPU domain
decomposition (no MPI, no NCCL, no manual token-dimension partitioning
across devices). "Number of processors increasing to max" from the CPU sweep
does not have a direct GPU analogue in the current code -- there is no
thread/rank count to scale within one GPU kernel launch. If you want real
multi-GPU scaling numbers, that requires implementing an MPI+OpenACC (or
NCCL) decomposition of the (B,N) token dimension across GPUs first, which
is not yet built -- listed as remaining work in Section 10 of the paper. If
your cluster has multiple GPU node sizes available (e.g. 1, 2, 4, 8 GPU
nodes) and you want a script that submits one job per node size purely to
see if per-GPU time is stable as you add GPUs, say so and I'll add that as
an array-job wrapper -- but it would still run the same single-GPU kernel on
each GPU independently (embarrassingly parallel over problem replicas), not
a true decomposition of one large problem across GPUs.

## Verification vs. benchmarking, one more time

- ../verification/*_verify.{c,f90}: fixed small size, checked against PyTorch
  to ~1e-16, this is the correctness proof.
- ./*_bench.{c,f90}: dynamic size, randomly initialized, no correctness check
  (same kernel code, just no PyTorch comparison at these sizes) -- this is
  the performance-scaling tool.

## Full block: Norm + QKV projection + Attention + residual + Norm + FFN + residual

    gcc -O2 -fopenmp full_block_bench.c -o full_block_bench -lm
    ./full_block_bench B N D DFF THREADS

    gcc -O2 -fopenacc full_block_gpu_bench.c -o full_block_gpu_bench -lm
    ./full_block_gpu_bench B N D DFF

Same computation as ../verification/full_block_verify.c and
full_block_gpu_kernel.c, with dynamic sizes and no file I/O, for the
sweep scripts. NOTE: attention is O(N^2) in sequence length (score
matrix), unlike the O(N) rmsnorm/mlp kernels -- at large N this
sublayer dominates total time, hence run_cpu_sweep.sh's separate
FULL_BLOCK_N_END cap, and full_block_gpu_bench.c's fixed 512-element
scratch buffers (N, DFF <= 512 for that program specifically).

NOT YET DONE: Fortran90 counterpart (full_block_bench.f90 /
full_block_gpu_bench.f90 do not exist).
