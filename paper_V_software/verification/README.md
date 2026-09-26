# Verification programs for Sections 3-10

## CPU verification (C and Fortran90, both OpenMP), grid B=2 N=4 D=8 D_ff=16

    python3 gen_inputs.py          # -> data/*.bin shared inputs
    python3 reference.py           # PyTorch double-precision autograd -> data/ref_*.bin

    gcc -O2 -fopenmp rmsnorm_verify.c -o rmsnorm_verify -lm
    gcc -O2 -fopenmp mlp_verify.c     -o mlp_verify -lm
    OMP_NUM_THREADS=4 ./rmsnorm_verify   # -> data/c_*.bin
    OMP_NUM_THREADS=4 ./mlp_verify

    gfortran -O2 -fopenmp rmsnorm_verify.f90 -o rmsnorm_verify_f90
    gfortran -O2 -fopenmp mlp_verify.f90     -o mlp_verify_f90
    OMP_NUM_THREADS=4 ./rmsnorm_verify_f90   # -> data/f_*.bin
    OMP_NUM_THREADS=4 ./mlp_verify_f90

    python3 compare.py              # prints max abs error per tensor, both languages, vs 1.2e-4

Result: all 8 tensors x 2 languages match PyTorch to ~1e-16 (machine
precision), confirmed identical at OMP_NUM_THREADS = 1, 2, 4, 8.

Both languages parallelize the per-token forward and backward loops.
The gradient-reduction loops that accumulate across tokens (G_gamma in
RMSNorm; G_Wgate, G_Wup, G_Wdown in the MLP) use explicit OpenMP array
reductions (`reduction(+:arr[0:n])` in C, `reduction(+:arr)` in Fortran)
rather than a serial post-pass, so the parallel and serial code paths
are identical code.

CAVEAT: this environment exposes a single CPU core, so the multi-thread
runs confirm the reductions are race-free, not that they scale. A
multi-core timing run is listed as remaining work in Section 10.

## Fused OpenACC kernel, chained RMSNorm -> gated MLP

    python3 chain_reference.py      # PyTorch reference for the chained pipeline -> data/chain_ref_*.bin
    gcc -O2 -fopenacc fused_gpu_kernel.c -o fused_gpu_kernel -lm
    ./fused_gpu_kernel              # -> data/g_Z.bin, data/g_F.bin

Result: Z matches to 4.4e-16, F matches to 1.3e-15.

CAVEAT: this environment has no CUDA-capable GPU and no nvc/pgcc
toolchain, so fused_gpu_kernel.c was compiled and run via GCC's OpenACC
host-fallback (`-fopenacc`) rather than true GPU offload. Every
`#pragma acc` directive is standard OpenACC and should offload
unmodified under nvc/nvc++ or an nvptx-enabled GCC build, but the
numbers produced here are a correctness check only -- not a GPU
bandwidth/latency result. A real GPU run is listed as remaining work in
Section 10 of the paper.

## Not yet done
- Multi-core OpenMP timing run to measure real parallel speedup.
- Real GPU execution of fused_gpu_kernel.c / full_block_gpu_kernel.c with
  bandwidth/latency measurement.
- Fortran90 counterpart to full_block_gpu_kernel.c (full_block_verify.f90
  now exists -- see below -- but the GPU/OpenACC full-block kernel does
  not yet have a Fortran90 version).
- End-to-end benchmarking against the M_block prediction (Section 9/10).

## Full block: Norm + QKV projection + Attention + residual + Norm + FFN + residual

    python3 reference_full_block.py   # PyTorch reference -> data/fb_ref_*.bin (10 tensors)
    gcc -O2 -o full_block_verify full_block_verify.c -lm
    ./full_block_verify                # -> data/fb_c_*.bin
    gcc -O2 -fopenacc full_block_gpu_kernel.c -o full_block_gpu_kernel -lm
    ./full_block_gpu_kernel            # -> data/fb_g_*.bin
    python3 compare_full_block.py      # checks BOTH C and GPU against fb_ref_*, prints max abs error per tensor vs 1.2e-4

Result: all 10 tensors (forward output Y; backward gradients G_X, both
G_gamma, all three QKV projection weight gradients, all three FFN
weight gradients) match PyTorch to ~1e-16 in BOTH the plain C and the
OpenACC implementations. Confirmed by running the entire sequence above
in order, from a clean data/ directory, together with gen_inputs.py /
reference.py / chain_reference.py / compare.py -- all 36 checks (16
standalone rmsnorm/mlp + 20 full-block) pass simultaneously.

IMPORTANT -- file naming convention: reference_full_block.py,
full_block_verify.c, and full_block_gpu_kernel.c all use an "fb_"
filename prefix (data/fb_X.bin, data/fb_ref_Y.bin, data/fb_c_GX.bin,
data/fb_g_GWq.bin, etc.), deliberately kept separate from the plain
X.bin/Wg.bin/Wu.bin/Wd.bin/etc. used by gen_inputs.py and the
standalone rmsnorm_verify.c/mlp_verify.c. An earlier version of
reference_full_block.py reused those exact filenames and silently
overwrote the shared input data every time it ran, causing
rmsnorm_verify.c/mlp_verify.c to load the wrong X (the full block's
random draw instead of gen_inputs.py's) on any subsequent run and fail
compare.py -- not because those programs were wrong, but because of the
file collision. If you add new full-block scripts, keep the fb_ prefix
convention to avoid reintroducing this.

This required adding a QKV projection layer (Z1 -> Q,K,V via learned
D x D weights Wq,Wk,Wv) that does not exist in ../attention/moa_attention.c,
since that file's DNF functions take Q,K,V as already-given inputs. The
projection's forward/backward and its own transpose-elimination
(Wq^T,Wk^T,Wv^T,Z1^T all reordered-index reads, never materialized) are
implemented inline in full_block_verify.c and full_block_gpu_kernel.c,
matching the same Section 8 lemma extended to these new weights (see
Section 9 of the paper for the added M_proj cost term).

CAVEAT (OpenACC): same as fused_gpu_kernel.c -- compiled/run via GCC's
host-fallback, correctness-only, not a GPU benchmark.

## Fortran90: full_block_verify.f90

Ported from full_block_verify.c, same flat-array indexing convention as
rmsnorm_verify.f90/mlp_verify.f90. Reads the same data/fb_*.bin files
reference_full_block.py writes; run reference_full_block.py first.

    gfortran -O2 -o full_block_verify_f90 full_block_verify.f90
    ./full_block_verify_f90   # -> data/fb_f_Y.bin, data/fb_f_GX.bin, etc.

Compare against data/fb_ref_*.bin using compare_full_block_f90.py (same
pattern as compare_full_block.py for the C version, substituting the
fb_f_ prefix for fb_c_/fb_g_). All 10 tensors match to ~1e-16.

One real bug was caught and fixed while porting, worth noting since
it's a recurring Fortran hazard in this project: Fortran is
case-insensitive, so the scalar temporaries `r1`, `x`, and `gv` (used
for the norm-1 statistic accumulator, a general-purpose loop temp, and
a gradient accumulator) silently collided with the array names `R1`
(the post-attention residual), `X` (the block input), and `GV` (the
attention value-gradient) respectively. The compiler did not catch
this cleanly -- it surfaced as a cascade of unrelated "no IMPLICIT
type" errors elsewhere in the file. Renamed the scalars to `rn1`,
`xval`, `gvv`, recompiled, and reverified against the reference before
calling it done.

