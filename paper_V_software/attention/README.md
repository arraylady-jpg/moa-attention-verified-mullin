# Attention (Papers I-III) - existing programs, not created in this conversation

These are the pre-existing C implementations from Papers I-II (forward,
backward, fused attention DNF/ONF) and Paper III (decode/inference CPU
benchmark), provided by the author. They are included here so the full
transformer block -- norm (Paper IV) + attention (Papers I-III) + norm
(Paper IV) + FFN (Paper IV) -- lives in one place.

## moa_attention.c

Six functions: forward/backward/fused, each in DNF and ONF (tiled) form.
Verified internally (run `./moa_attention`): all six agree to exact
machine precision (0.0000e+00) on B=2, N=4, D=8, T=4.

    gcc -O2 -lm -o moa_attention moa_attention.c
    ./moa_attention

## moa_decode_bench.c

OpenMP CPU benchmark for the MoA decode kernel (inference-time, KV-cache
style single-query attention against n cached keys/values), sweeping
thread count and array size, self-verifying against a sequential
reference each run.

    gcc -O3 -fopenmp -o moa_decode_bench moa_decode_bench.c -lm
    ./moa_decode_bench <max_threads> <n_start> <n_max> <dk> <dv> [repeats]

## Status

Integrated with the norm/MLP fused kernel as of this update: see
../verification/full_block_verify.c and
../verification/full_block_gpu_kernel.c for the complete
norm -> QKV projection -> attention -> residual -> norm -> FFN ->
residual block, verified against PyTorch to machine precision in both
plain C and OpenACC. A Fortran90 counterpart to these two files does
not exist yet (remaining work, Section 10 of the paper).
