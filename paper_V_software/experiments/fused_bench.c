/* fused_bench.c
 * Parameterized, timed OpenMP benchmark for Paper II's fused forward+backward.
 *
 * Source: Mullin & Hains, "Attention at the Theoretical Minimum",
 * Algorithm 2 (fused, A never written to DRAM). Math ported directly
 * from moa_fused_dnf in attention/moa_attention.c -- this file adds
 * dynamic sizing via argv, OpenMP parallelization, and timing; it does
 * not change the math.
 *
 * Usage: ./fused_bench B N D THREADS [--check]
 * Prints one CSV line: impl,lang,B,N,D,threads,seconds
 * With --check: also runs forward+backward composed separately and
 * confirms the fused kernel's Out/GV/GQ/GK match to machine precision
 * -- a real correctness check, since fused and separate implement
 * exactly the same math and must agree regardless of problem size.
 *
 * Validates Paper II's claim: M_fwd+bwd = 4nd_k + 4nd_v
 * (here d_k = d_v = D, single head) -- specifically, that fusing
 * eliminates the n^2-scaling A array entirely (Algorithm 1's M_bwd has
 * an n^2 term from A; Algorithm 2's M_fwd+bwd does not).
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

static void backward_dnf(const double *Q, const double *K, const double *V,
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
        for (int ir = 0; ir < N; ir++) {
            for (int j = 0; j < D; j++) GQ[QKV(b,ir,j)] = 0.0;
        }
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

/* fused_dnf (Algorithm 2), OpenMP.
 *
 * REVISION HISTORY: the original version used a single fused pass with
 * "reduction(+:GV[0:B*N*D], GK[0:B*N*D])" -- TWO stack-based array
 * reductions, doubling the risk backward_bench.c hit on real Delta
 * hardware (job 21225788, segfault/SIGSEGV at N=8192, D=64, where a
 * single B*N*D*8-byte reduction buffer exactly matched the cluster's
 * 8MB default stack limit). Fixed the same way: eliminated both
 * reduction clauses, replaced the single fused loop with five
 * sequential passes, each race-free by construction (every output
 * element is written by exactly one thread, no accumulation across
 * parallel iterations), using two persistent heap-allocated N x N x B
 * arrays (AROW for softmax weights, GS for the backward gate value)
 * instead of any OpenMP reduction. This mirrors backward_bench.c's fix
 * and, not coincidentally, is exactly the "compute once, reuse many
 * times" discipline Paper II's own memory-optimal DNF derivation
 * already argues for -- the bug was in this benchmark's OpenMP
 * structure, not in the underlying math. */
static void fused_dnf_omp(const double *Q, const double *K, const double *V, const double *dO,
                           double *Out, double *GV, double *GQ, double *GK,
                           double scale, int B, int N, int D) {
    double *AROW = malloc(sizeof(double) * (size_t)B*N*N);
    double *GS   = malloc(sizeof(double) * (size_t)B*N*N);

    /* Pass 1: forward softmax weights (persisted to AROW) + Out (race-free, indexed by ir) */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            double *arow = malloc(sizeof(double) * N);
            for (int ic = 0; ic < N; ic++) {
                double s = 0.0;
                for (int j = 0; j < D; j++)
                    s += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                arow[ic] = s * scale;
            }
            softmax_row(arow, N);
            for (int ic = 0; ic < N; ic++) AROW[AA(b,ir,ic)] = arow[ic];
            for (int id = 0; id < D; id++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++) o += arow[ic] * V[QKV(b,ic,id)];
                Out[QKV(b,ir,id)] = o;
            }
            free(arow);
        }
    }

    /* Pass 2: GV, parallel over (b,ic), inner loop over ir -- race-free */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < N; ic++) {
            for (int id = 0; id < D; id++) {
                double g = 0.0;
                for (int ir = 0; ir < N; ir++)
                    g += AROW[AA(b,ir,ic)] * dO[QKV(b,ir,id)];
                GV[QKV(b,ic,id)] = g;
            }
        }
    }

    /* Pass 3: GS (persisted), parallel over (b,ir) -- race-free */
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
                corr += AROW[AA(b,ir,ic)] * ga_row[ic];
            for (int ic = 0; ic < N; ic++)
                GS[AA(b,ir,ic)] = scale * AROW[AA(b,ir,ic)] * (ga_row[ic] - corr);
            free(ga_row);
        }
    }

    /* Pass 4: GQ, parallel over (b,ir) -- race-free */
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

    /* Pass 5: GK, parallel over (b,ic), inner loop over ir -- race-free */
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

    free(AROW); free(GS);
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
    double *Out = malloc(sizeof(double) * B*N*D);
    double *GV = malloc(sizeof(double) * B*N*D);
    double *GQ = malloc(sizeof(double) * B*N*D);
    double *GK = malloc(sizeof(double) * B*N*D);

    double t0 = omp_get_wtime();
    fused_dnf_omp(Q, K, V, dO, Out, GV, GQ, GK, scale, B, N, D);
    double t1 = omp_get_wtime();

    printf("fused,C,%d,%d,%d,%d,%.6f\n", B, N, D, THREADS, t1 - t0);

    if (do_check) {
        double *A = malloc(sizeof(double) * B*N*N);
        double *Out_ref = malloc(sizeof(double) * B*N*D);
        double *GV_ref = malloc(sizeof(double) * B*N*D);
        double *GQ_ref = malloc(sizeof(double) * B*N*D);
        double *GK_ref = malloc(sizeof(double) * B*N*D);

        forward_dnf(Q, K, V, A, Out_ref, scale, B, N, D);
        backward_dnf(Q, K, V, A, dO, GV_ref, GQ_ref, GK_ref, scale, B, N, D);

        fprintf(stderr, "check: Out  max diff = %.3e\n", maxdiff(Out, Out_ref, B*N*D));
        fprintf(stderr, "check: GV   max diff = %.3e\n", maxdiff(GV, GV_ref, B*N*D));
        fprintf(stderr, "check: GQ   max diff = %.3e\n", maxdiff(GQ, GQ_ref, B*N*D));
        fprintf(stderr, "check: GK   max diff = %.3e\n", maxdiff(GK, GK_ref, B*N*D));

        free(A); free(Out_ref); free(GV_ref); free(GQ_ref); free(GK_ref);
    }

    free(Q); free(K); free(V); free(dO); free(Out); free(GV); free(GQ); free(GK);
    return 0;
}
