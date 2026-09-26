/* backward_bench.c
 * Parameterized, timed OpenMP benchmark for Paper II's backward pass.
 *
 * Source: Mullin & Hains, "Attention at the Theoretical Minimum",
 * Algorithm 1 (backward, A from DRAM). Math ported directly from
 * moa_backward_dnf in attention/moa_attention.c (which verifies this
 * exact computation to machine precision against moa_backward_onf at
 * fixed size B=2,N=4,D=8) -- this file adds dynamic sizing via argv,
 * OpenMP parallelization, and timing; it does not change the math.
 *
 * Usage: ./backward_bench B N D THREADS
 * Prints one CSV line: impl,lang,B,N,D,threads,seconds
 *
 * Validates Paper II's claim: M_bwd = n^2 + 4nd_k + 3nd_v
 * (here d_k = d_v = D, single head).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#define QKV(b,i,d) ((b)*N*D + (i)*D + (d))
#define AA(b,i,j)  ((b)*N*N + (i)*N + (j))

static double* alloc_rand(int n, unsigned long seed) {
    double* a = malloc(sizeof(double) * n);
    unsigned long s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        a[i] = (double)(s & 0x7FFFFFFFULL) / (double)0x7FFFFFFF - 1.0;
    }
    return a;
}

static void softmax_row(double *row, int n) {
    double m = -DBL_MAX;
    for (int i = 0; i < n; i++) if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - m); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

/* forward_dnf, needed to produce A before backward can read it "from DRAM" */
static void forward_dnf(const double *Q, const double *K, const double *V,
                         double *A, double *Out, double scale, int B, int N, int D) {
    double *arow = malloc(sizeof(double) * N);
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                double s = 0.0;
                for (int j = 0; j < D; j++)
                    s += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                arow[ic] = s * scale;
            }
            softmax_row(arow, N);
            for (int ic = 0; ic < N; ic++) A[AA(b,ir,ic)] = arow[ic];
            for (int id = 0; id < D; id++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++) o += arow[ic] * V[QKV(b,ic,id)];
                Out[QKV(b,ir,id)] = o;
            }
        }
    }
    free(arow);
}

/* backward_dnf (Algorithm 1), OpenMP over (b,ir) then (b,ic).
 *
 * REVISION HISTORY: the original version used
 * "reduction(+:GK[0:B*N*D])" to handle GK's cross-ir accumulation.
 * Real Delta hardware (job 21225788) segfaulted (exit 139/SIGSEGV) at
 * N=8192, D=64: B*N*D*8 bytes = exactly 8MB, matching the cluster's
 * 8MB default stack limit (`ulimit -s` = 8192) exactly. GCC's libgomp
 * places the per-thread private copy for an OpenMP array reduction on
 * the stack, not the heap -- at this size it overflows the stack
 * outright. Reproduced locally (this sandbox has the same 8MB
 * ulimit -s) before attempting a fix, confirming this diagnosis rather
 * than guessing.
 *
 * Fixed by eliminating the reduction entirely: the intermediate
 * gs = scale*A*(ga_row-corr) value is precomputed once into a
 * heap-allocated N x N x B buffer (GS, the same order of size as A,
 * already heap-allocated elsewhere in this program), then GQ (indexed
 * by ir, race-free as before) and GK (now parallelized over (b,ic)
 * with an inner loop over ir, race-free by construction since each
 * (b,ic) thread owns its own GK[ic] slot) are each computed by reading
 * GS back out. No OpenMP reduction clause of any kind remains in this
 * file. */
static void backward_dnf_omp(const double *Q, const double *K, const double *V,
                              const double *A, const double *dO,
                              double *GV, double *GQ, double *GK,
                              double scale, int B, int N, int D) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
        for (int ic = 0; ic < N; ic++)
            for (int id = 0; id < D; id++) {
                double g = 0.0;
                for (int ir = 0; ir < N; ir++)
                    g += A[AA(b,ir,ic)] * dO[QKV(b,ir,id)];
                GV[QKV(b,ic,id)] = g;
            }

    double *GS = malloc(sizeof(double) * (size_t)B*N*N);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            double *ga_row = malloc(sizeof(double) * N);
            for (int ic = 0; ic < N; ic++) {
                double g = 0.0;
                for (int id = 0; id < D; id++)
                    g += dO[QKV(b,ir,id)] * V[QKV(b,ic,id)];
                ga_row[ic] = g;
            }
            double corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += A[AA(b,ir,ic)] * ga_row[ic];
            for (int ic = 0; ic < N; ic++)
                GS[AA(b,ir,ic)] = scale * A[AA(b,ir,ic)] * (ga_row[ic] - corr);
            free(ga_row);
        }
    }

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            for (int j = 0; j < D; j++) GQ[QKV(b,ir,j)] = 0.0;
            for (int ic = 0; ic < N; ic++) {
                double gs = GS[AA(b,ir,ic)];
                for (int j = 0; j < D; j++)
                    GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < N; ic++) {
            for (int j = 0; j < D; j++) {
                double sum = 0.0;
                for (int ir = 0; ir < N; ir++)
                    sum += GS[AA(b,ir,ic)] * Q[QKV(b,ir,j)];
                GK[QKV(b,ic,j)] = sum;
            }
        }
    }

    free(GS);
}

/* Serial reference (no OpenMP), for --check verification of the
 * parallel version above. Same math, straightforwardly ordered. */
static void backward_dnf_serial(const double *Q, const double *K, const double *V,
                                 const double *A, const double *dO,
                                 double *GV, double *GQ, double *GK,
                                 double scale, int B, int N, int D) {
    for (int b = 0; b < B; b++)
        for (int ic = 0; ic < N; ic++)
            for (int id = 0; id < D; id++) {
                double g = 0.0;
                for (int ir = 0; ir < N; ir++)
                    g += A[AA(b,ir,ic)] * dO[QKV(b,ir,id)];
                GV[QKV(b,ic,id)] = g;
            }
    double *ga_row = malloc(sizeof(double) * N);
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++)
            for (int j = 0; j < D; j++) GQ[QKV(b,ir,j)] = 0.0;
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                double g = 0.0;
                for (int id = 0; id < D; id++)
                    g += dO[QKV(b,ir,id)] * V[QKV(b,ic,id)];
                ga_row[ic] = g;
            }
            double corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += A[AA(b,ir,ic)] * ga_row[ic];
            for (int ic = 0; ic < N; ic++) {
                double gs = scale * A[AA(b,ir,ic)] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
                    GK[QKV(b,ic,j)] += gs * Q[QKV(b,ir,j)];
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

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s B N D THREADS [--check]\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]), N = atoi(argv[2]), D = atoi(argv[3]), THREADS = atoi(argv[4]);
    int do_check = (argc > 5 && strcmp(argv[5], "--check") == 0);
    omp_set_num_threads(THREADS);
    const double scale = 1.0 / sqrt((double)D);

    double *Q = alloc_rand(B*N*D, 1), *K = alloc_rand(B*N*D, 2), *V = alloc_rand(B*N*D, 3);
    double *dO = alloc_rand(B*N*D, 4);
    double *A = malloc(sizeof(double) * B*N*N);
    double *Out = malloc(sizeof(double) * B*N*D);
    double *GV = calloc(B*N*D, sizeof(double));
    double *GQ = calloc(B*N*D, sizeof(double));
    double *GK = calloc(B*N*D, sizeof(double));

    /* Forward is a prerequisite (Algorithm 1 reads A "from DRAM"), not timed here. */
    forward_dnf(Q, K, V, A, Out, scale, B, N, D);

    double t0 = omp_get_wtime();
    backward_dnf_omp(Q, K, V, A, dO, GV, GQ, GK, scale, B, N, D);
    double t1 = omp_get_wtime();

    printf("backward,C,%d,%d,%d,%d,%.6f\n", B, N, D, THREADS, t1 - t0);

    if (do_check) {
        double *GV_ref = calloc(B*N*D, sizeof(double));
        double *GQ_ref = calloc(B*N*D, sizeof(double));
        double *GK_ref = calloc(B*N*D, sizeof(double));
        backward_dnf_serial(Q, K, V, A, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D);
        fprintf(stderr, "check: GV max diff = %.3e\n", maxdiff(GV, GV_ref, B*N*D));
        fprintf(stderr, "check: GQ max diff = %.3e\n", maxdiff(GQ, GQ_ref, B*N*D));
        fprintf(stderr, "check: GK max diff = %.3e\n", maxdiff(GK, GK_ref, B*N*D));
        free(GV_ref); free(GQ_ref); free(GK_ref);
    }

    free(Q); free(K); free(V); free(dO); free(A); free(Out); free(GV); free(GQ); free(GK);
    return 0;
}
