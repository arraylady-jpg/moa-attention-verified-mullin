/* backward_gpu_bench.c
 * Parameterized, timed OpenACC benchmark for Paper II's backward pass.
 *
 * Source: Mullin & Hains, "Attention at the Theoretical Minimum",
 * Algorithm 1 (backward, A from DRAM). Same math as backward_bench.c
 * (the OpenMP CPU version).
 *
 * REVISION HISTORY:
 * v1: same one-process-per-size timing bug as forward_gpu_bench.c and
 *     decode_gpu_bench.c, fixed the same way (single warmup, one
 *     timed call per N).
 * v2: Paper I's real A100 data (forward_gpu_bench.c, job 21181233)
 *     showed that even with the v1 fix, a single timed measurement
 *     per N has no protection against system jitter -- the local
 *     scaling exponent came out noisy until forward_gpu_bench.c was
 *     revised to average over multiple timed repeats, matching
 *     moa_decode_bench.c's CPU pattern. Applied here too. Also raised
 *     the scratch buffer from 512 to 8192 (MAX_N). Confirmed on real
 *     A100/H200 hardware, jobs 21225789/21225790.
 * v3 (this version): after fused_attn_gpu_bench.c's GV-atomic fix was
 *     confirmed to reverse Paper II's GPU regression completely (jobs
 *     21250395/21250396, results_paper/moa_experimental_validation.tex
 *     Section 4.5), the natural next question is whether this file's
 *     own remaining GK atomic -- deliberately left in place during
 *     that fix as the scoped baseline -- could be eliminated the same
 *     way. Unlike GV (which only needs A and dO, both directly
 *     indexable by (ir,ic)), GK depends on ga_row and corr, both
 *     computed per-ir, so a direct (b,ic)-parallelization isn't
 *     possible without first persisting the intermediate gate value
 *     GS[ir,ic] = scale*A[ir,ic]*(ga_row[ic]-corr) to a heap array --
 *     the same technique already used for fused_attn_gpu_bench.c's GV
 *     fix and for backward_bench.c's own OpenMP crash fix (Section
 *     4.1). Applied here: GQ+GK's single combined kernel becomes two
 *     passes, (1) parallel over (b,ir): compute GQ (race-free, own
 *     slot) and persist GS; (2) parallel over (b,ic): read GS back and
 *     compute GK (race-free, own slot) -- no atomic anywhere in this
 *     file anymore. Not yet confirmed on real GPU hardware.
 *
 * Usage: ./backward_gpu_bench B N_START N_END D [REPEATS] [--check]
 * Sweeps N from N_START to N_END, doubling. For each N, an untimed
 * warmup call is followed by REPEATS (default 5) timed calls,
 * reporting the mean. Prints one CSV line per N:
 * impl,lang,B,N,D,seconds,repeats
 * With --check: checks once per N (first repeat only), not every repeat.
 *
 * NOTE ON THIS ENVIRONMENT: compiled/run via GCC's OpenACC host
 * fallback, correctness-only -- host-fallback has no comparable
 * per-process context-init cost, measurement-jitter, or atomic-
 * contention profile to real GPU hardware, so none of the three
 * revisions' timing claims can be confirmed here, only correctness and
 * internal consistency.
 *
 * Fixed-size scratch buffer (arow[MAX_N], ga_row[MAX_N]) caps usable N
 * at 8192. New persistent heap array (GS), sized B*N*N doubles, same
 * order of magnitude as fused_attn_gpu_bench.c's v3 fix and
 * backward_bench.c's own GS/AROW arrays.
 *
 * GV is independent per (b,ic,id) -- no atomic needed (unchanged from
 * v2). GQ needs neither (indexed only by ir, unchanged). GK, as of
 * this v3, is also atomic-free, via the persisted-GS technique above.
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
                                double *A, double scale, int B, int N, int D) {
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

/* One backward call: B,N,D fixed, times just the backward acc data region
 * (forward is a prerequisite, computed on GPU but not timed, same as before). */
static double backward_once(int B, int N, int D, int do_check) {
    const double scale = 1.0 / sqrt((double)D);
    int NT = B * N;

    double *Q = alloc_rand(NT*D, 1), *K = alloc_rand(NT*D, 2), *V = alloc_rand(NT*D, 3);
    double *dO = alloc_rand(NT*D, 4);
    double *A = malloc(sizeof(double) * (long)NT*N);
    double *GV = calloc(NT*D, sizeof(double));
    double *GQ = calloc(NT*D, sizeof(double));
    double *GK = calloc(NT*D, sizeof(double));
    double *GS = malloc(sizeof(double) * (long)NT*N);

    #pragma acc data copyin(Q[0:NT*D], K[0:NT*D]) copyout(A[0:(long)NT*N])
    {
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
                for (int ic = 0; ic < N; ic++) A[(long)(b*N+ir)*N+ic] = arow[ic] / sum;
            }
        }
    }

    double t0 = wall_time();

    #pragma acc data copyin(Q[0:NT*D], K[0:NT*D], V[0:NT*D], dO[0:NT*D], A[0:(long)NT*N]) \
                      copyout(GV[0:NT*D], GQ[0:NT*D], GK[0:NT*D]) \
                      create(GS[0:(long)NT*N])
    {
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++)
            for (int ic = 0; ic < N; ic++)
                for (int id = 0; id < D; id++) {
                    double g = 0.0;
                    for (int ir = 0; ir < N; ir++)
                        g += A[(long)(b*N+ir)*N+ic] * dO[(b*N+ir)*D+id];
                    GV[(b*N+ic)*D+id] = g;
                }

        /* Pass 1: GQ (race-free, own slot) + persist GS. Parallel over
         * (b,ir), same structure as before this fix -- the only change
         * is that GK's computation is no longer inline here. */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double ga_row[MAX_N];
                for (int ic = 0; ic < N; ic++) {
                    double g = 0.0;
                    for (int id = 0; id < D; id++) g += dO[(b*N+ir)*D+id] * V[(b*N+ic)*D+id];
                    ga_row[ic] = g;
                }
                double corr = 0.0;
                for (int ic = 0; ic < N; ic++) corr += A[(long)(b*N+ir)*N+ic] * ga_row[ic];
                for (int j = 0; j < D; j++) GQ[(b*N+ir)*D+j] = 0.0;
                for (int ic = 0; ic < N; ic++) {
                    double gs = scale * A[(long)(b*N+ir)*N+ic] * (ga_row[ic] - corr);
                    GS[(long)(b*N+ir)*N+ic] = gs;
                    for (int j = 0; j < D; j++)
                        GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                }
            }
        }

        /* Pass 2: GK, parallel over (b,ic) -- each thread owns its own
         * GK[ic] slot, reading GS back out. No atomic. This is the fix:
         * v2 computed this inline in Pass 1's (b,ir)-parallelized loop,
         * forcing an atomic since multiple ir-threads write the same
         * ic-indexed GK slot. */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ic = 0; ic < N; ic++) {
                for (int j = 0; j < D; j++) {
                    double g = 0.0;
                    for (int ir = 0; ir < N; ir++)
                        g += GS[(long)(b*N+ir)*N+ic] * Q[(b*N+ir)*D+j];
                    GK[(b*N+ic)*D+j] = g;
                }
            }
        }
    }

    double t1 = wall_time();

    if (do_check) {
        double *A_ref = malloc(sizeof(double) * (long)NT*N);
        double *GV_ref = calloc(NT*D, sizeof(double));
        double *GQ_ref = calloc(NT*D, sizeof(double));
        double *GK_ref = calloc(NT*D, sizeof(double));
        forward_dnf_serial(Q, K, V, A_ref, scale, B, N, D);
        backward_dnf_serial(Q, K, V, A_ref, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D);
        fprintf(stderr, "check: N=%d GV max diff = %.3e\n", N, maxdiff(GV, GV_ref, NT*D));
        fprintf(stderr, "check: N=%d GQ max diff = %.3e\n", N, maxdiff(GQ, GQ_ref, NT*D));
        fprintf(stderr, "check: N=%d GK max diff = %.3e\n", N, maxdiff(GK, GK_ref, NT*D));
        free(A_ref); free(GV_ref); free(GQ_ref); free(GK_ref);
    }

    free(Q); free(K); free(V); free(dO); free(A); free(GV); free(GQ); free(GK); free(GS);
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
    (void)backward_once(B, n_start < 64 ? n_start : 64, D, 0);

    for (int N = n_start; N <= n_end; N *= 2) {
        (void)backward_once(B, N, D, 0);  /* per-N untimed warmup */

        double total = 0.0;
        for (int r = 0; r < repeats; r++) {
            int check_this_rep = (do_check && r == 0);
            total += backward_once(B, N, D, check_this_rep);
        }
        double mean = total / repeats;
        printf("backward,C-OpenACC,%d,%d,%d,%.6f,%d\n", B, N, D, mean, repeats);
        fflush(stdout);
    }

    return 0;
}
