/* Parameterized fused RMSNorm -> gated MLP forward+backward benchmark (C/OpenACC).
 * Counterpart to fused_gpu_kernel.c (the file-I/O correctness-verification
 * version, which stays fixed-size and unmodified by this fix -- that file
 * is the one to run for correctness proof; this one is for timing only).
 *
 * REVISION HISTORY: the original version took a single N per process
 * invocation. Real hardware data from Delta (A100, job 21180661) for
 * experiments/decode_gpu_bench.c -- the same one-process-per-size
 * structure this file originally had -- showed wall time essentially
 * flat across a 1024x range of N, dominated by ~300ms of CUDA context
 * init cost paid fresh on every process launch. This version sweeps N
 * internally within a single process, paying that cost once via an
 * untimed warmup before any measurement begins. Same fix as
 * decode_gpu_bench.c; see that file's revision history for the full
 * diagnosis. This file has NOT itself been run on real GPU hardware yet.
 *
 * All accumulator buffers here (GX_norm, Ggamma, GX_ffn, GWg, GWu, GWd)
 * were already calloc'd (not malloc'd) before this fix, so -- unlike
 * fused_attn_gpu_bench.c's Out buffer, which was malloc'd and silently
 * relied on fresh-process zero pages -- there was no analogous
 * uninitialized-accumulator bug to fix here when moving to repeated
 * in-process calls. Verified by re-running --check-equivalent output
 * comparison after the refactor (see compare notes at the bottom of
 * this file's test log, not embedded in this file itself since this
 * program has no built-in --check the way the experiments/ files do).
 *
 * Compiled and timed via OpenACC host-fallback in the sandbox that
 * built this (no CUDA GPU / nvc toolchain there) -- every #pragma acc
 * is standard OpenACC and should offload unmodified under nvc/nvc++.
 *
 * Usage: ./fused_gpu_bench B N_START N_END D DFF
 * Sweeps N from N_START to N_END, doubling. Prints one CSV line per N:
 * impl,lang,B,N,D,DFF,seconds
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double* alloc_rand(int n, unsigned seed) {
    double* a = malloc(sizeof(double) * n);
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        a[i] = ((double)(s % 20000) / 10000.0 - 1.0) * 0.3;
    }
    return a;
}

static double silu(double u) { return u / (1.0 + exp(-u)); }
static double silu_grad(double u) {
    double s = 1.0 / (1.0 + exp(-u));
    return s * (1.0 + u * (1.0 - s));
}

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* One fused norm+MLP forward+backward call, B,N,D,DFF fixed. */
static double fused_once(int B, int N, int D, int DFF) {
    int NT = B * N;
    const double EPS = 1e-6;

    double *X = alloc_rand(NT*D, 1);
    double *gam = alloc_rand(D, 2);
    double *GF = alloc_rand(NT*D, 15);
    double *Wg = alloc_rand(D*DFF, 12);
    double *Wu = alloc_rand(D*DFF, 13);
    double *Wd = alloc_rand(DFF*D, 14);

    double *Z = malloc(sizeof(double)*NT*D);
    double *U = malloc(sizeof(double)*NT*DFF);
    double *F = malloc(sizeof(double)*NT*D);
    double *r = malloc(sizeof(double)*NT);
    double *GX_norm = calloc(NT*D, sizeof(double));
    double *Ggamma  = calloc(D, sizeof(double));
    double *GX_ffn  = calloc(NT*D, sizeof(double));
    double *GWg = calloc(D*DFF, sizeof(double));
    double *GWu = calloc(D*DFF, sizeof(double));
    double *GWd = calloc(DFF*D, sizeof(double));

    double t0 = wall_time();

    #pragma acc data copyin(X[0:NT*D], gam[0:D], GF[0:NT*D], Wg[0:D*DFF], Wu[0:D*DFF], Wd[0:DFF*D]) \
                      copyout(Z[0:NT*D], F[0:NT*D], GX_norm[0:NT*D], Ggamma[0:D], \
                               GX_ffn[0:NT*D], GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D]) \
                      create(U[0:NT*DFF], r[0:NT])
    {
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) { double x = X[t*D+d]; sumsq += x*x; }
            r[t] = sqrt(sumsq / D + EPS);
        }
        #pragma acc parallel loop collapse(2)
        for (int t = 0; t < NT; t++)
            for (int d = 0; d < D; d++)
                Z[t*D+d] = gam[d] * (X[t*D+d] / r[t]);

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double Vrow[512], Hrow[512];
            for (int f = 0; f < DFF; f++) {
                double u = 0.0, v = 0.0;
                for (int d = 0; d < D; d++) {
                    u += Z[t*D+d] * Wg[d*DFF+f];
                    v += Z[t*D+d] * Wu[d*DFF+f];
                }
                U[t*DFF+f] = u; Vrow[f] = v; Hrow[f] = silu(u) * v;
            }
            for (int d = 0; d < D; d++) {
                double fo = 0.0;
                for (int f = 0; f < DFF; f++) fo += Hrow[f] * Wd[f*D+d];
                F[t*D+d] = fo;
            }
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double Vrow[512], Hrow[512], GHrow[512], GUrow[512], GVrow[512], GZf[512];
            for (int f = 0; f < DFF; f++) {
                double v = 0.0;
                for (int d = 0; d < D; d++) v += Z[t*D+d] * Wu[d*DFF+f];
                Vrow[f] = v;
                Hrow[f] = silu(U[t*DFF+f]) * v;
            }
            for (int f = 0; f < DFF; f++)
                for (int d = 0; d < D; d++) {
                    #pragma acc atomic update
                    GWd[f*D+d] += Hrow[f] * GF[t*D+d];
                }
            for (int f = 0; f < DFF; f++) {
                double gh = 0.0;
                for (int d = 0; d < D; d++) gh += GF[t*D+d] * Wd[f*D+d];
                GHrow[f] = gh;
                GVrow[f] = gh * silu(U[t*DFF+f]);
                GUrow[f] = gh * Vrow[f] * silu_grad(U[t*DFF+f]);
            }
            for (int d = 0; d < D; d++) {
                for (int f = 0; f < DFF; f++) {
                    #pragma acc atomic update
                    GWg[d*DFF+f] += Z[t*D+d] * GUrow[f];
                    #pragma acc atomic update
                    GWu[d*DFF+f] += Z[t*D+d] * GVrow[f];
                }
            }
            for (int d = 0; d < D; d++) {
                double gz = 0.0;
                for (int f = 0; f < DFF; f++)
                    gz += GUrow[f] * Wg[d*DFF+f] + GVrow[f] * Wu[d*DFF+f];
                GZf[d] = gz;
                GX_ffn[t*D+d] = gz;
            }
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = X[t*D+d] / r[t];
                correction += GZf[d] * gam[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = X[t*D+e] / r[t];
                GX_norm[t*D+e] = (GZf[e]*gam[e] - (x_hat_e/D)*correction) / r[t];
                #pragma acc atomic update
                Ggamma[e] += GZf[e] * x_hat_e;
            }
        }
    }

    double t1 = wall_time();

    free(X); free(gam); free(GF); free(Wg); free(Wu); free(Wd);
    free(Z); free(U); free(F); free(r);
    free(GX_norm); free(Ggamma); free(GX_ffn); free(GWg); free(GWu); free(GWd);
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s B N_START N_END D DFF\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int n_start = atoi(argv[2]), n_end = atoi(argv[3]);
    int D = atoi(argv[4]), DFF = atoi(argv[5]);

    if (DFF > 512) {
        fprintf(stderr, "DFF exceeds fixed scratch buffer size (512); increase Vrow/Hrow/etc. bound in source.\n");
        return 1;
    }

    fprintf(stderr, "warming up (paying CUDA context init cost once, untimed)...\n");
    (void)fused_once(B, n_start < 64 ? n_start : 64, D, DFF);

    for (int N = n_start; N <= n_end; N *= 2) {
        double elapsed = fused_once(B, N, D, DFF);
        printf("fused,C-OpenACC,%d,%d,%d,%d,%.6f\n", B, N, D, DFF, elapsed);
        fflush(stdout);
    }

    return 0;
}
