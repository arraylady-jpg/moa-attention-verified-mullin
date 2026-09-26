# Results: Papers I-IV, Multi-Architecture Benchmarking

STATUS AS OF THIS UPDATE: real CPU scaling data exists for Paper I
(Delta, job 21181232) -- see Section 2 below. Everything else is still
a `[TODO: fill in]` placeholder. Paper I's GPU shape comparison is
pending (job 21181233 submitted, results not yet received).

## 1. What was measured

| Paper | Kernel(s) | CPU experiment | GPU experiment |
|---|---|---|---|
| I | forward pass | `forward_bench.c`/`.f90` | `forward_gpu_bench.c` |
| II | backward (Alg. 1), fused (Alg. 2) | `backward_bench.c`/`.f90`, `fused_bench.c`/`.f90` | `backward_gpu_bench.c`, `fused_attn_gpu_bench.c` |
| III | decode | `moa_decode_bench.c` (pre-existing) | `decode_gpu_bench.c` |
| IV | RMSNorm, gated MLP, complete block | `benchmarking/run_cpu_sweep.sh` | `fused_gpu_bench.c`/`.f90`, `full_block_gpu_bench.c` |

Machines: Anvil (A100, H100), Delta (A100x4, A100x8, A40x4, H200x8; MI100x8
excluded, AMD/ROCm not covered by these OpenACC kernels).

## 2. Per-paper results

### Paper I -- forward pass

**CPU scaling** (Delta `cpu` partition, real data, job 21181232,
`real_results/results_paper1_delta_cpu_21181232.csv`): the real finding
is a thread-count crossover that depends on problem size, not a single
"more threads is always better" story.

| N | Best thread count | Speedup at best vs. 1 thread | 128 threads vs. best |
|---|---|---|---|
| 64 | 4 | 1.4x | 14.5x SLOWER |
| 128 | 8 | 4.0x | 5.5x slower |
| 256 | 16 | 7.3x | 3.8x slower |
| 512 | 32 | 16.4x | 2.3x slower |
| 1024 | 64 | 23.4x | 1.7x slower |
| 2048 | 128 | 43.3x | -- (128 optimal) |
| 4096 | 128 | 53.5x | -- |
| 8192 | 128 | 60.6x | -- |

At N=64, using all 128 threads is 14.5x *slower* than the actual sweet
spot (4 threads) -- thread-spawn/teardown overhead dominates when
there's too little work per thread. The crossover to "more threads
always helps" happens somewhere between N=1024 and N=2048. At N=8192,
128 threads gives 60.6x speedup (47% parallel efficiency on 128 cores
-- reasonable for a memory-bandwidth-bound kernel, not surprising it's
below ideal linear scaling).

**Cost-model consistency check:** at 1 thread, wall time roughly
quadruples with each doubling of N (0.0013 -> 0.0067 -> 0.0271 ->
0.1388 -> 0.445 -> 2.13 -> 7.49 -> 24.57 seconds) -- exactly the O(n^2)
signature expected from the N x N attention score matrix, matching
Paper I's forward-pass cost claim directly.

**Practical implication for anyone deploying this:** the right thread
count is a function of sequence length, not a fixed cluster-wide
setting -- worth stating explicitly in Paper I if this cost function
ever needs a "how to configure this in production" section.

**GPU shape comparison**: [TODO -- job 21181233 (Delta gpuA100x4) was
submitted; results not yet received as of this update.]

**Does it match M_fwd?** [TODO -- no profiler (nsys/ncu) run yet for
Paper I specifically.]

### Paper II -- backward, fused

**CPU scaling**: [TODO]

**GPU shape comparison**: [TODO -- this is the highest-value comparison
in the whole set, since M_fwd+bwd = 4nd_k+4nd_v eliminating the n^2
term is the series' most-cited result. If real hardware confirms the
fused kernel scales linearly where backward-alone scales quadratically,
that's the headline finding for this paper.]

**Does it match M_bwd / M_fwd+bwd?** [TODO]

### Paper III -- decode

**CPU scaling**: [TODO -- decode is O(n), so this should scale to much
larger n than Papers I/II before becoming impractical; note the actual
n range achieved.]

**GPU shape comparison**: [TODO]

**KV-cache / inference-cost claim**: [TODO -- tie back to the "longer
context or fewer GPUs at fixed VRAM" framing from the DARPA pitch
material, now with real numbers instead of the qualitative claim.]

### Paper IV -- transformer block

**CPU scaling** (RMSNorm, MLP, full block): [TODO]

**GPU shape comparison**: [TODO]

**Does it match M_block = 24nd + 4nd_ff?** [TODO -- this was the
specific target of `anvil_profile_block.sbatch`; report the
predicted-vs-measured percentage error directly.]

## 3. Cross-paper comparison: best outcome

[TODO once data exists. Questions to answer here, not before:]
- Which machine shape gives the best wall-clock time per paper, and is
  it consistent across papers or does the ranking flip (e.g. H200
  fastest for compute-bound MLP but A100 competitive for memory-bound
  RMSNorm)?
- Where does the real measured speedup diverge from what the
  memory-optimal cost functions predict, and is that divergence
  explained by compiler behavior, kernel launch overhead, or something
  the DNF/ONF derivation didn't account for?
- Does any machine shape fail to show the claimed transpose-elimination
  benefit (i.e., does a "naive" comparison baseline need to be added to
  make the claim checkable at all -- none of the current bench programs
  include a deliberately naive/unfused baseline to compare against)?

## 4. What we could do better: machine-shape-driven follow-ups

[TODO once data exists, but likely candidates to investigate based on
what Delta's partition list actually offers:]
- **A40 vs A100 vs H200 cost-efficiency**, not just speed -- if SU cost
  differs by partition, the "best" shape for a production deployment
  argument isn't necessarily the fastest one.
- **Multi-GPU decomposition** -- every kernel in this package is
  single-GPU. `gpuA100x8`/`gpuH200x8` nodes have 8 GPUs; none of that
  capacity is used. Whether an MPI+OpenACC or NCCL decomposition of the
  (B,N) token dimension is worth building depends on whether
  single-GPU numbers show a real bottleneck at large N.
- **AMD/ROCm port for `gpuMI100x8`** -- currently unreachable by any
  kernel here; would need a HIP or OpenMP-target rewrite, not just an
  `nvc` recompile.
- **Fortran90 GPU parity** -- every GPU kernel is C-only; if Fortran GPU
  numbers differ meaningfully from C (they shouldn't, if `nvc`/
  `nvfortran` codegen is comparable, but that's an empirical claim, not
  an assumption) that's worth its own note.

## 5. Raw data

[TODO: list the actual CSV filenames once produced, e.g.:]
```
results_paper1_cpu_anvil_XXXXX.csv
results_paper1_gpu_delta_gpuA100x4_XXXXX.csv
results_paper1_gpu_delta_gpuH200x8_XXXXX.csv
...
```
Combined via:
```
python3 compare_machine_shapes.py --out combined_all.csv \
  results_paper1_cpu_anvil_XXXXX.csv:machine=anvil-cpu \
  results_paper1_gpu_delta_gpuA100x4_XXXXX.csv \
  ...
```
