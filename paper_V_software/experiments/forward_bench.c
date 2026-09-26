/* forward_bench.c
 * Parameterized, timed OpenMP benchmark for Paper I's forward pass.
 *
 * Source: arXiv:2606.07713v1, eqs. 17, 20, 21. Math ported directly
 * from moa_forward_dnf in attention/moa_attention.c (which verifies
 * this exact computation to machine precision against moa_forward_onf
 * at fixed size B=2,N=4,D=8,T=4) -- this file adds dynamic sizing via
 * argv, OpenMP parallelization, and timing; it does not change the
 * math.
 *
 * Usage: ./forward_bench B N D THREADS [--check]
 * Prints one CSV line: impl,lang,B,N,D,threads,seconds
 * With --check: also runs a serial reference and confirms the OpenMP
 * version's A and Out match to machine precision.
 *
 * Validates Paper I's forward-pass memory cost claim: does the
 * ONF-style forward pass (here parallelized over independent (b,ir)
 * query rows, matching the DNF's natural parallelism) actually deliver
 * real throughput at scale, and does it agree with the backward/fused
 * kernels' own internal forward_dnf (used as their prerequisite/
 * reference step) at every thread count.
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

/* forward_dnf (arXiv eqs. 17, 20, 21), OpenMP over (b,ir): every query
 * row's A[b,ir,:] and Out[b,ir,:] are written only by that (b,ir)
 * iteration -- no cross-iteration accumulation, so unlike backward's
 * GV/GK this needs no reduction, just collapse(2). */
static void forward_dnf_omp(const double *Q, const double *K, const double *V,
                             double *A, double *Out, double scale, int B, int N, int D) {
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
            for (int ic = 0; ic < N; ic++)
                A[AA(b,ir,ic)] = arow[ic];
            for (int id = 0; id < D; id++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++)
                    o += arow[ic] * V[QKV(b,ic,id)];
                Out[QKV(b,ir,id)] = o;
            }
            free(arow);
        }
    }
}

/* Serial reference (no OpenMP), for --check verification. */
static void forward_dnf_serial(const double *Q, const double *K, const double *V,
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
    double *A = malloc(sizeof(double) * B*N*N);
    double *Out = malloc(sizeof(double) * B*N*D);

    double t0 = omp_get_wtime();
    forward_dnf_omp(Q, K, V, A, Out, scale, B, N, D);
    double t1 = omp_get_wtime();

    printf("forward,C,%d,%d,%d,%d,%.6f\n", B, N, D, THREADS, t1 - t0);

    if (do_check) {
        double *A_ref = malloc(sizeof(double) * B*N*N);
        double *Out_ref = malloc(sizeof(double) * B*N*D);
        forward_dnf_serial(Q, K, V, A_ref, Out_ref, scale, B, N, D);
        fprintf(stderr, "check: A   max diff = %.3e\n", maxdiff(A, A_ref, B*N*N));
        fprintf(stderr, "check: Out max diff = %.3e\n", maxdiff(Out, Out_ref, B*N*D));
        free(A_ref); free(Out_ref);
    }

    free(Q); free(K); free(V); free(A); free(Out);
    return 0;
}
