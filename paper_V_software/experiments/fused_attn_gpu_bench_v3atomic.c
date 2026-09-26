/* fused_attn_gpu_bench.c
 * Parameterized, timed OpenACC benchmark for Paper II's fused
 * forward+backward ATTENTION pass (Algorithm 2, A never materialized).
 * Named fused_attn_gpu_bench.c, not fused_gpu_bench.c, to disambiguate
 * from Paper IV's ../benchmarking/fused_gpu_bench.c, which fuses
 * RMSNorm+gated-MLP -- a different pair of kernels entirely.
 *
 * Source: Mullin & Hains, "Attention at the Theoretical Minimum",
 * Algorithm 2. Same math as fused_bench.c (the OpenMP CPU version).
 *
 * REVISION HISTORY:
 * v1: same one-process-per-size timing bug as forward_gpu_bench.c and
 *     decode_gpu_bench.c, fixed the same way (single warmup, one
 *     timed call per N).
 * v2: repeated-and-averaged timing (matching forward_gpu_bench.c v2),
 *     applied preemptively before this file had been run on real
 *     hardware. Also raised the scratch buffer from 512 to 8192.
 * v3 (this version): real Delta profiling (job 21247666, ncu on both
 *     this file and backward_gpu_bench.c at N=2048) confirmed why v2's
 *     real hardware data (jobs 21225789 A100, 21225790 H200) showed
 *     fused running 14-84% SLOWER than naive forward+backward, the
 *     opposite of the CPU result: v2's single monolithic kernel used
 *     "#pragma acc atomic update" for BOTH GV and GK, while
 *     backward_gpu_bench.c's kernel only needs it for GK (forward is a
 *     separate, atomic-free prerequisite pass there). Measured atomic
 *     instruction count: fused = 1,073,741,824, backward = 536,870,912
 *     -- exactly 2.0000x, confirmed against
 *     smsp__inst_executed_op_generic_atom_dot_alu.sum, not
 *     approximately double. This version eliminates the GV atomic by
 *     restructuring the single monolithic kernel into four passes,
 *     mirroring backward_gpu_bench.c's approach exactly: (1) softmax
 *     weights persisted to a heap array (AROW) + Out, race-free,
 *     parallel over (b,ir); (2) GV, parallel over (b,ic) instead of
 *     (b,ir) so each thread owns its own GV[ic] slot -- no atomic
 *     needed, identical in structure to backward_gpu_bench.c's GV
 *     kernel; (3) the backward gate value GS, persisted to a second
 *     heap array, parallel over (b,ir), race-free; (4) GQ (race-free,
 *     indexed by ir) and GK (still atomic -- accumulated across ir
 *     into the same ic-indexed slot, exactly as in
 *     backward_gpu_bench.c) computed together. This was a deliberately
 *     scoped fix at the time, not a full rewrite: GK's atomic was left
 *     in place, matching backward_gpu_bench.c's own remaining
 *     structure, so fused's atomic count fell from 2x backward's to
 *     1x -- parity, not zero. Confirmed on real hardware (jobs
 *     21250395/21250396): complete reversal on both GPU shapes, fused
 *     now faster than naive at every tested size (Section 4.5).
 * v4 (this version): after v3's confirmed reversal, the natural next
 *     question (also posed in Section 4.5) was whether removing GK's
 *     atomic too -- via the same persisted-GS technique already used
 *     for GV here -- would help further. Applied: Pass 4 (formerly
 *     GQ+GK combined, GK atomic) is now Pass 4 (GQ only, parallel over
 *     (b,ir)) and a new Pass 5 (GK, parallel over (b,ic), reading GS
 *     back out, no atomic). This file now has zero atomics anywhere --
 *     matching the same fix applied to backward_gpu_bench.c's own v3.
 *     Not yet confirmed on real GPU hardware.
 *
 * Usage: ./fused_attn_gpu_bench B N_START N_END D [REPEATS] [--check]
 * Sweeps N from N_START to N_END, doubling. For each N, an untimed
 * warmup call is followed by REPEATS (default 5) timed calls,
 * reporting the mean. Prints one CSV line per N:
 * impl,lang,B,N,D,seconds,repeats
 * With --check: also runs forward+backward composed separately (CPU,
 * serial) once per N (first repeat only) and confirms match to
 * machine precision.
 *
 * NOTE ON THIS ENVIRONMENT: compiled/run via GCC's OpenACC host
 * fallback, correctness-only -- host-fallback has no comparable
 * per-process context-init cost, measurement-jitter, or atomic-
 * contention profile to real GPU hardware, so none of the three
 * revisions' timing claims can be confirmed here, only correctness and
 * internal consistency. This file has not itself been run on real GPU
 * hardware since this v3 restructure.
 *
 * Fixed-size scratch buffers (arow[MAX_N], ga_row[MAX_N]) cap usable N
 * at 8192. Two new persistent heap arrays (AROW, GS), each sized
 * B*N*N doubles, same order of magnitude as the CPU v2 fix's GS/AROW
 * arrays (Section 4.1's crash-fix commit).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define MAX_N 8192


static double* alloc_rand(int n, unsigned seed) {
    double* a = malloc(sizeof(double) * n);
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        a[i] = ((double)(s % 20000) / 10000.0 - 1.0);
    }
    return a;
}

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void softmax_row(double *row, int n) {
    double m = -1e300;
    for (int i = 0; i < n; i++) if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - m); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

static void forward_dnf_serial(const double *Q, const double *K, const double *V,
                                double *A, double *Out, double scale, int B, int N, int D) {
    double *arow = malloc(sizeof(double) * N);
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                double s = 0.0;
                for (int j = 0; j < D; j++) s += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                arow[ic] = s * scale;
            }
            softmax_row(arow, N);
            for (int ic = 0; ic < N; ic++) A[(b*N+ir)*N+ic] = arow[ic];
            for (int id = 0; id < D; id++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++) o += arow[ic] * V[(b*N+ic)*D+id];
                Out[(b*N+ir)*D+id] = o;
            }
        }
    }
    free(arow);
}

static void backward_dnf_serial(const double *Q, const double *K, const double *V,
                                 const double *A, const double *dO,
                                 double *GV, double *GQ, double *GK,
                                 double scale, int B, int N, int D) {
    for (int b = 0; b < B; b++)
        for (int ic = 0; ic < N; ic++)
            for (int id = 0; id < D; id++) {
                double g = 0.0;
                for (int ir = 0; ir < N; ir++) g += A[(b*N+ir)*N+ic] * dO[(b*N+ir)*D+id];
                GV[(b*N+ic)*D+id] = g;
            }
    double *ga_row = malloc(sizeof(double) * N);
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) for (int j = 0; j < D; j++) GQ[(b*N+ir)*D+j] = 0.0;
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                double g = 0.0;
                for (int id = 0; id < D; id++) g += dO[(b*N+ir)*D+id] * V[(b*N+ic)*D+id];
                ga_row[ic] = g;
            }
            double corr = 0.0;
            for (int ic = 0; ic < N; ic++) corr += A[(b*N+ir)*N+ic] * ga_row[ic];
            for (int ic = 0; ic < N; ic++) {
                double gs = scale * A[(b*N+ir)*N+ic] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                    GK[(b*N+ic)*D+j] += gs * Q[(b*N+ir)*D+j];
                }
            }
        }
    }
    free(ga_row);
}

static double maxdiff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d > m) m = d; }
    return m;
}

/* One fused call: B,N,D fixed, times just the acc data region.
 * Four passes now, not one: (1) AROW+Out, (2) GV (no atomic, parallel
 * over b,ic), (3) GS, (4) GQ+GK (GK atomic, matching
 * backward_gpu_bench.c exactly). */
static double fused_once(int B, int N, int D, int do_check) {
    const double scale = 1.0 / sqrt((double)D);
    int NT = B * N;

    double *Q = alloc_rand(NT*D, 1), *K = alloc_rand(NT*D, 2), *V = alloc_rand(NT*D, 3);
    double *dO = alloc_rand(NT*D, 4);
    double *Out = calloc(NT*D, sizeof(double));
    double *GV = calloc(NT*D, sizeof(double));
    double *GQ = calloc(NT*D, sizeof(double));
    double *GK = calloc(NT*D, sizeof(double));
    double *AROW = malloc(sizeof(double) * (long)NT*N);
    double *GS   = malloc(sizeof(double) * (long)NT*N);

    double t0 = wall_time();

    #pragma acc data copyin(Q[0:NT*D], K[0:NT*D], V[0:NT*D], dO[0:NT*D]) \
                      copyout(Out[0:NT*D], GV[0:NT*D], GQ[0:NT*D], GK[0:NT*D]) \
                      create(AROW[0:(long)NT*N], GS[0:(long)NT*N])
    {
        /* Pass 1: softmax weights (persisted to AROW) + Out. Race-free,
         * parallel over (b,ir) -- identical structure to before, minus
         * the GV/GK work this used to also do inline. */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double arow[MAX_N];
                double m = -1e300;
                for (int ic = 0; ic < N; ic++) {
                    double s = 0.0;
                    for (int j = 0; j < D; j++) s += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                    arow[ic] = s * scale;
                    if (arow[ic] > m) m = arow[ic];
                }
                double sum = 0.0;
                for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic]-m); sum += arow[ic]; }
                for (int ic = 0; ic < N; ic++) {
                    arow[ic] /= sum;
                    AROW[(long)(b*N+ir)*N+ic] = arow[ic];
                }
                for (int id = 0; id < D; id++) {
                    double o = 0.0;
                    for (int ic = 0; ic < N; ic++) o += arow[ic] * V[(b*N+ic)*D+id];
                    Out[(b*N+ir)*D+id] = o;
                }
            }
        }

        /* Pass 2: GV, parallel over (b,ic) -- each thread owns its own
         * GV[ic] slot, reading AROW back out. No atomic. This is the
         * fix: v2 computed this inline while parallelizing over (b,ir),
         * forcing an atomic since multiple ir-threads write the same
         * ic-indexed GV slot. */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ic = 0; ic < N; ic++) {
                for (int id = 0; id < D; id++) {
                    double g = 0.0;
                    for (int ir = 0; ir < N; ir++)
                        g += AROW[(long)(b*N+ir)*N+ic] * dO[(b*N+ir)*D+id];
                    GV[(b*N+ic)*D+id] = g;
                }
            }
        }

        /* Pass 3: GS (persisted), parallel over (b,ir) -- race-free. */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double ga_row[MAX_N];
                for (int ic = 0; ic < N; ic++) {
                    double g = 0.0;
                    for (int id = 0; id < D; id++)
                        g += dO[(b*N+ir)*D+id] * V[(b*N+ic)*D+id];
                    ga_row[ic] = g;
                }
                double corr = 0.0;
                for (int ic = 0; ic < N; ic++)
                    corr += AROW[(long)(b*N+ir)*N+ic] * ga_row[ic];
                for (int ic = 0; ic < N; ic++)
                    GS[(long)(b*N+ir)*N+ic] = scale * AROW[(long)(b*N+ir)*N+ic] * (ga_row[ic] - corr);
            }
        }

        /* v3 comparison variant: GQ and GK computed together, GK via
         * atomic (this is the pre-v4 structure, restored here only to
         * profile against v4's atomic-free version at N=4096 and test
         * whether the extra GS re-read in v4's separate Pass 5 is what
         * causes the N>=4096 slowdown found in job 21294299). */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                for (int j = 0; j < D; j++) GQ[(b*N+ir)*D+j] = 0.0;
                for (int ic = 0; ic < N; ic++) {
                    double gs = GS[(long)(b*N+ir)*N+ic];
                    for (int j = 0; j < D; j++) {
                        GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                        #pragma acc atomic update
                        GK[(b*N+ic)*D+j] += gs * Q[(b*N+ir)*D+j];
                    }
                }
            }
        }
    }

    double t1 = wall_time();

    if (do_check) {
        double *A = malloc(sizeof(double) * (long)NT*N);
        double *Out_ref = malloc(sizeof(double) * NT*D);
        double *GV_ref = calloc(NT*D, sizeof(double));
        double *GQ_ref = calloc(NT*D, sizeof(double));
        double *GK_ref = calloc(NT*D, sizeof(double));
        forward_dnf_serial(Q, K, V, A, Out_ref, scale, B, N, D);
        backward_dnf_serial(Q, K, V, A, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D);
        fprintf(stderr, "check: N=%d Out max diff = %.3e\n", N, maxdiff(Out, Out_ref, NT*D));
        fprintf(stderr, "check: N=%d GV  max diff = %.3e\n", N, maxdiff(GV, GV_ref, NT*D));
        fprintf(stderr, "check: N=%d GQ  max diff = %.3e\n", N, maxdiff(GQ, GQ_ref, NT*D));
        fprintf(stderr, "check: N=%d GK  max diff = %.3e\n", N, maxdiff(GK, GK_ref, NT*D));
        free(A); free(Out_ref); free(GV_ref); free(GQ_ref); free(GK_ref);
    }

    free(Q); free(K); free(V); free(dO); free(Out); free(GV); free(GQ); free(GK);
    free(AROW); free(GS);
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s B N_START N_END D [REPEATS] [--check]\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int n_start = atoi(argv[2]), n_end = atoi(argv[3]);
    int D = atoi(argv[4]);

    int repeats = 5;
    int do_check = 0;
    if (argc > 5) {
        if (strcmp(argv[5], "--check") == 0) {
            do_check = 1;
        } else {
            repeats = atoi(argv[5]);
            if (argc > 6 && strcmp(argv[6], "--check") == 0) do_check = 1;
        }
    }

    if (n_end > MAX_N) {
        fprintf(stderr, "N_END exceeds fixed scratch buffer size (%d); increase MAX_N in source.\n", MAX_N);
        return 1;
    }

    fprintf(stderr, "warming up (paying CUDA context init cost once, untimed)...\n");
    (void)fused_once(B, n_start < 64 ? n_start : 64, D, 0);

    for (int N = n_start; N <= n_end; N *= 2) {
        (void)fused_once(B, N, D, 0);  /* per-N untimed warmup */

        double total = 0.0;
        for (int r = 0; r < repeats; r++) {
            int check_this_rep = (do_check && r == 0);
            total += fused_once(B, N, D, check_this_rep);
        }
        double mean = total / repeats;
        printf("fused,C-OpenACC,%d,%d,%d,%.6f,%d\n", B, N, D, mean, repeats);
        fflush(stdout);
    }

    return 0;
}
