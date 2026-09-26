# MoA Transformer Block: paper, code, and real-hardware validation

Six folders:

- **paper/** -- "The Transformer Block at the Theoretical Minimum"
  (`.tex`/`.pdf`/`.bib`/`.bbl`), covering RMSNorm and the gated MLP
  (SwiGLU) forward/backward/fused derivations, the transpose-elimination
  lemma, the whole-block cost function (including the QKV projection
  term), and verification results.

- **verification/** -- correctness-checked C, Fortran90, and OpenACC
  implementations of every kernel in the paper, matched against PyTorch
  double-precision autograd to ~1e-16. Includes both the standalone
  RMSNorm/MLP kernels and the complete integrated block (Norm -> QKV
  projection -> Attention -> residual -> Norm -> FFN -> residual).
  Start here: `README.md` for exact run order.

- **benchmarking/** -- dynamic-size, timed (not correctness-checked)
  versions of the same kernels, plus `run_cpu_sweep.sh` and
  `slurm_gpu_sweep.sbatch` for sweeping problem size and thread/GPU
  count.

- **attention/** -- the pre-existing Papers I-III attention code
  (`moa_attention.c`, forward/backward/fused DNF and ONF; and
  `moa_decode_bench.c`, the inference-time decode benchmark), provided
  by the author and integrated into `verification/full_block_*` and
  `benchmarking/full_block_*`.

- **experiments/** -- `EXPERIMENT_PLAN.md`, organized per-paper (I-IV),
  real SLURM scripts and real result CSVs for both Purdue Anvil (account
  `cis261396-ai`/`cis261396-gpu`) and NCSA Delta (account
  `bibg-delta-cpu`/`bibg-delta-gpu`), both under ACCESS allocation
  CIS261396. `results_paper/` contains the full validation writeup
  (`moa_experimental_validation.tex`, the detailed working-notes
  version) and the submission-shaped paper
  (`moa_real_hardware_validation_submission.tex`), plus every figure
  and the SLURM job-identifier appendix. Start here:
  `EXPERIMENT_PLAN.md` for the per-paper experiment log, or
  `results_paper/moa_real_hardware_validation_submission.pdf` for the
  finished paper.

## Status: all four papers' real-hardware validation is complete

Every item the original version of this README tracked as open has
since been closed, on real hardware, on both clusters:

- **Real GPU hardware**: every kernel (forward, backward, fused
  attention, decode, fused norm+MLP, the complete block) has been run
  and profiled on real NVIDIA A100/H100/H200 GPUs on both Anvil and
  Delta, not GCC's host-fallback.
- **Real multi-core CPU speedup**: every kernel has been swept across
  real thread counts (1 to 128) on real Delta/Anvil CPU nodes, not a
  single-core sandbox.
- **$M_{\mathrm{block}}$ validated against real hardware counters**:
  profiled with `ncu` (job 21289516), measured DRAM traffic compared
  directly against the analytical prediction (1.50x gap, partially
  explained by measured memory-coalescing inefficiency).
- **Fortran90 full-block counterpart**: exists and was run on real GPU
  hardware (job 21273959), the basis for the paper's C-versus-Fortran
  finding.

Three headline results came out of this validation effort, detailed in
`results_paper/moa_real_hardware_validation_submission.pdf`:
1. A real GPU performance regression (atomic-memory contention in the
   fused attention kernel) was diagnosed to four decimal places
   (2.0000x more atomic instructions) and fixed, reversing the result
   completely on both tested GPU shapes.
2. The identical kernel derivation produces a 535x NUMA-locality
   penalty on one cluster's topology versus under 3x oversubscription
   cost on another's.
3. A genuine, only partially resolved compiler-level anomaly: the same
   denotational computation runs faster in C than Fortran on CPU, and
   faster in Fortran than C on GPU.

This package's scope is explicitly a validation study of MoA-derived
designs against their own formally-derived predictions -- it does not
benchmark these designs against externally-optimized production
kernels (see `results_paper/moa_real_hardware_validation_submission.pdf`,
Section "Scope and Limitations").
