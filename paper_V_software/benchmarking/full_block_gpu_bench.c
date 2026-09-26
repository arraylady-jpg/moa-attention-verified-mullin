/* full_block_gpu_bench.c
 * Parameterized, timed OpenACC benchmark for the complete transformer block:
 * Norm -> QKV projection -> Attention -> residual -> Norm -> FFN -> residual.
 * Same computation as full_block_gpu_kernel.c (the fixed-size, file-I/O
 * correctness-verification version, unmodified by this fix); this one
 * takes dynamic sizes and times the run, no file I/O.
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
 * v2: the N-dimension scratch buffers (arow, ga_row) were hardcoded to
 * 512, capping the whole sweep at N<=512 -- inconsistent with every
 * other real-hardware sweep in this project (Papers I-II raised the
 * same class of buffer to 8192 after their own real hardware runs).
 * Caught before this file was ever submitted to real hardware, by
 * checking `delta_paper4_gpu_sweep.sbatch`'s actual default N_END
 * (4096) against this file's stated cap (512) rather than assuming
 * they matched. Raised to MAX_N=8192 to match the CPU sweep's full
 * range and Papers I-II's precedent. Not yet confirmed on real GPU
 * hardware since this change.
 *
 * All accumulator buffers (GX, GR1, GZ1, GZ2, GQ, GK, GV, Ggamma1,
 * Ggamma2, GWq, GWk, GWv, GWg, GWu, GWd) were already calloc'd before
 * this fix, so -- unlike experiments/fused_attn_gpu_bench.c's Out
 * buffer, which was malloc'd and silently relied on fresh-process zero
 * pages -- there was no analogous uninitialized-accumulator bug here.
 * Verified by calling full_block_once() 3x at the same N within one
 * process and confirming bit-identical output each time (no cross-call
 * state leakage) before this file was considered done.
 *
 * Compiled/run via OpenACC host-fallback in the sandbox that built
 * this (no CUDA GPU / nvc toolchain there).
 *
 * Usage: ./full_block_gpu_bench B N_START N_END D DFF
 * Sweeps N from N_START to N_END, doubling. Prints one CSV line per N:
 * impl,lang,B,N,D,DFF,seconds
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_N 8192

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

static double full_block_once(int B, int N, int D, int DFF) {
    const double EPS = 1e-6;
    int NT = B * N;
    long NN = (long)B * N * N;

    double *X = alloc_rand(NT*D, 1);
    double *gamma1 = alloc_rand(D, 2);
    double *gamma2 = alloc_rand(D, 3);
    double *Wq = alloc_rand(D*D, 4);
    double *Wk = alloc_rand(D*D, 5);
    double *Wv = alloc_rand(D*D, 6);
    double *Wg = alloc_rand(D*DFF, 7);
    double *Wu = alloc_rand(D*DFF, 8);
    double *Wd = alloc_rand(DFF*D, 9);
    double *GY = alloc_rand(NT*D, 10);

    double *Z1 = malloc(sizeof(double)*NT*D);
    double *r1 = malloc(sizeof(double)*NT);
    double *Q  = malloc(sizeof(double)*NT*D);
    double *K  = malloc(sizeof(double)*NT*D);
    double *V  = malloc(sizeof(double)*NT*D);
    double *A  = malloc(sizeof(double)*NN);
    double *R1 = malloc(sizeof(double)*NT*D);
    double *Z2 = malloc(sizeof(double)*NT*D);
    double *r2 = malloc(sizeof(double)*NT);
    double *U  = malloc(sizeof(double)*NT*DFF);
    double *Vg = malloc(sizeof(double)*NT*DFF);
    double *H  = malloc(sizeof(double)*NT*DFF);
    double *Y  = malloc(sizeof(double)*NT*D);
    double *GX  = calloc(NT*D, sizeof(double));
    double *GR1 = calloc(NT*D, sizeof(double));
    double *GZ1 = calloc(NT*D, sizeof(double));
    double *GZ2 = calloc(NT*D, sizeof(double));
    double *GQ  = calloc(NT*D, sizeof(double));
    double *GK  = calloc(NT*D, sizeof(double));
    double *GV  = calloc(NT*D, sizeof(double));
    double *Ggamma1 = calloc(D, sizeof(double));
    double *Ggamma2 = calloc(D, sizeof(double));
    double *GWq = calloc(D*D, sizeof(double));
    double *GWk = calloc(D*D, sizeof(double));
    double *GWv = calloc(D*D, sizeof(double));
    double *GWg = calloc(D*DFF, sizeof(double));
    double *GWu = calloc(D*DFF, sizeof(double));
    double *GWd = calloc(DFF*D, sizeof(double));

    const double scale = 1.0 / sqrt((double)D);

    double t0 = wall_time();

    #pragma acc data \
      copyin(X[0:NT*D], gamma1[0:D], gamma2[0:D], Wq[0:D*D], Wk[0:D*D], Wv[0:D*D], \
             Wg[0:D*DFF], Wu[0:D*DFF], Wd[0:DFF*D], GY[0:NT*D]) \
      create(Z1[0:NT*D], r1[0:NT], Q[0:NT*D], K[0:NT*D], V[0:NT*D], A[0:NN], \
             R1[0:NT*D], Z2[0:NT*D], r2[0:NT], U[0:NT*DFF], Vg[0:NT*DFF], H[0:NT*DFF], \
             GR1[0:NT*D], GZ1[0:NT*D], GZ2[0:NT*D], GQ[0:NT*D], GK[0:NT*D], GV[0:NT*D]) \
      copyout(Y[0:NT*D], GX[0:NT*D], Ggamma1[0:D], Ggamma2[0:D], \
              GWq[0:D*D], GWk[0:D*D], GWv[0:D*D], GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])
    {
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) { double x = X[t*D+d]; sumsq += x*x; }
            r1[t] = sqrt(sumsq / D + EPS);
            for (int d = 0; d < D; d++)
                Z1[t*D+d] = gamma1[d] * (X[t*D+d] / r1[t]);
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            for (int d2 = 0; d2 < D; d2++) {
                double q=0, k=0, v=0;
                for (int d = 0; d < D; d++) {
                    q += Z1[t*D+d] * Wq[d*D+d2];
                    k += Z1[t*D+d] * Wk[d*D+d2];
                    v += Z1[t*D+d] * Wv[d*D+d2];
                }
                Q[t*D+d2] = q; K[t*D+d2] = k; V[t*D+d2] = v;
            }
        }

        #pragma acc parallel loop
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double arow[MAX_N];
                double m = -1e300;
                for (int ic = 0; ic < N; ic++) {
                    double dot = 0.0;
                    for (int j = 0; j < D; j++)
                        dot += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                    arow[ic] = scale * dot;
                    if (arow[ic] > m) m = arow[ic];
                }
                double s = 0.0;
                for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic]-m); s += arow[ic]; }
                for (int ic = 0; ic < N; ic++) { arow[ic] /= s; A[(long)(b*N+ir)*N+ic] = arow[ic]; }
                for (int d = 0; d < D; d++) {
                    double o = 0.0;
                    for (int ic = 0; ic < N; ic++)
                        o += arow[ic] * V[(b*N+ic)*D+d];
                    R1[(b*N+ir)*D+d] = X[(b*N+ir)*D+d] + o;
                }
            }
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) { double x = R1[t*D+d]; sumsq += x*x; }
            r2[t] = sqrt(sumsq / D + EPS);
            for (int d = 0; d < D; d++)
                Z2[t*D+d] = gamma2[d] * (R1[t*D+d] / r2[t]);
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            for (int f = 0; f < DFF; f++) {
                double u=0, v=0;
                for (int d = 0; d < D; d++) {
                    u += Z2[t*D+d] * Wg[d*DFF+f];
                    v += Z2[t*D+d] * Wu[d*DFF+f];
                }
                U[t*DFF+f] = u; Vg[t*DFF+f] = v;
                H[t*DFF+f] = silu(u) * v;
            }
            for (int d = 0; d < D; d++) {
                double fo = 0.0;
                for (int f = 0; f < DFF; f++)
                    fo += H[t*DFF+f] * Wd[f*D+d];
                Y[t*D+d] = R1[t*D+d] + fo;
            }
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double GUrow[512], GVrow[512];
            for (int f = 0; f < DFF; f++) {
                double gh = 0.0;
                for (int d = 0; d < D; d++)
                    gh += GY[t*D+d] * Wd[f*D+d];
                GVrow[f] = gh * silu(U[t*DFF+f]);
                GUrow[f] = gh * Vg[t*DFF+f] * silu_grad(U[t*DFF+f]);
            }
            for (int f = 0; f < DFF; f++)
                for (int d = 0; d < D; d++) {
                    #pragma acc atomic update
                    GWd[f*D+d] += H[t*DFF+f] * GY[t*D+d];
                }
            for (int d = 0; d < D; d++)
                for (int f = 0; f < DFF; f++) {
                    #pragma acc atomic update
                    GWg[d*DFF+f] += Z2[t*D+d] * GUrow[f];
                    #pragma acc atomic update
                    GWu[d*DFF+f] += Z2[t*D+d] * GVrow[f];
                }
            double gz2;
            for (int d = 0; d < D; d++) {
                gz2 = 0.0;
                for (int f = 0; f < DFF; f++)
                    gz2 += GUrow[f]*Wg[d*DFF+f] + GVrow[f]*Wu[d*DFF+f];
                GZ2[t*D+d] = gz2;
            }
            for (int d = 0; d < D; d++)
                GR1[t*D+d] = GY[t*D+d];
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = R1[t*D+d] / r2[t];
                correction += GZ2[t*D+d] * gamma2[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = R1[t*D+e] / r2[t];
                GR1[t*D+e] += (GZ2[t*D+e]*gamma2[e] - (x_hat_e/D)*correction) / r2[t];
                #pragma acc atomic update
                Ggamma2[e] += GZ2[t*D+e] * x_hat_e;
            }
        }

        #pragma acc parallel loop
        for (int i = 0; i < NT*D; i++)
            GX[i] = GR1[i];

        #pragma acc parallel loop
        for (int b = 0; b < B; b++) {
            for (int ic = 0; ic < N; ic++)
                for (int d = 0; d < D; d++) {
                    double gv = 0.0;
                    for (int ir = 0; ir < N; ir++)
                        gv += A[(long)(b*N+ir)*N+ic] * GR1[(b*N+ir)*D+d];
                    GV[(b*N+ic)*D+d] = gv;
                }
            for (int ir = 0; ir < N; ir++) {
                double ga_row[MAX_N];
                for (int ic = 0; ic < N; ic++) {
                    double g = 0.0;
                    for (int d = 0; d < D; d++)
                        g += GR1[(b*N+ir)*D+d] * V[(b*N+ic)*D+d];
                    ga_row[ic] = g;
                }
                double corr = 0.0;
                for (int ic = 0; ic < N; ic++)
                    corr += A[(long)(b*N+ir)*N+ic] * ga_row[ic];
                for (int ic = 0; ic < N; ic++) {
                    double gs = scale * A[(long)(b*N+ir)*N+ic] * (ga_row[ic] - corr);
                    for (int j = 0; j < D; j++) {
                        GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                        GK[(b*N+ic)*D+j] += gs * Q[(b*N+ir)*D+j];
                    }
                }
            }
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            for (int d = 0; d < D; d++)
                for (int d2 = 0; d2 < D; d2++) {
                    #pragma acc atomic update
                    GWq[d*D+d2] += Z1[t*D+d] * GQ[t*D+d2];
                    #pragma acc atomic update
                    GWk[d*D+d2] += Z1[t*D+d] * GK[t*D+d2];
                    #pragma acc atomic update
                    GWv[d*D+d2] += Z1[t*D+d] * GV[t*D+d2];
                }
            for (int d = 0; d < D; d++) {
                double g = 0.0;
                for (int d2 = 0; d2 < D; d2++)
                    g += GQ[t*D+d2]*Wq[d*D+d2] + GK[t*D+d2]*Wk[d*D+d2] + GV[t*D+d2]*Wv[d*D+d2];
                GZ1[t*D+d] = g;
            }
        }

        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = X[t*D+d] / r1[t];
                correction += GZ1[t*D+d] * gamma1[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = X[t*D+e] / r1[t];
                GX[t*D+e] += (GZ1[t*D+e]*gamma1[e] - (x_hat_e/D)*correction) / r1[t];
                #pragma acc atomic update
                Ggamma1[e] += GZ1[t*D+e] * x_hat_e;
            }
        }
    }

    double t1 = wall_time();

    free(X); free(gamma1); free(gamma2); free(Wq); free(Wk); free(Wv); free(Wg); free(Wu); free(Wd); free(GY);
    free(Z1); free(r1); free(Q); free(K); free(V); free(A); free(R1); free(Z2); free(r2); free(U); free(Vg); free(H); free(Y);
    free(GX); free(GR1); free(GZ1); free(GZ2); free(GQ); free(GK); free(GV);
    free(Ggamma1); free(Ggamma2); free(GWq); free(GWk); free(GWv); free(GWg); free(GWu); free(GWd);

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

    if (n_end > MAX_N) {
        fprintf(stderr, "N_END exceeds fixed scratch buffer size (%d) used for arow/ga_row; increase MAX_N in source.\n", MAX_N);
        return 1;
    }
    if (DFF > 512) {
        fprintf(stderr, "DFF exceeds fixed scratch buffer size (512); increase in source.\n");
        return 1;
    }

    fprintf(stderr, "warming up (paying CUDA context init cost once, untimed)...\n");
    (void)full_block_once(B, n_start < 64 ? n_start : 64, D, DFF);

    for (int N = n_start; N <= n_end; N *= 2) {
        double elapsed = full_block_once(B, N, D, DFF);
        printf("fullblock,C-OpenACC,%d,%d,%d,%d,%.6f\n", B, N, D, DFF, elapsed);
        fflush(stdout);
    }

    return 0;
}
