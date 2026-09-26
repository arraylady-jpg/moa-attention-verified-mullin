/* Parameterized RMSNorm forward+backward benchmark (C/OpenMP).
 * Usage: ./rmsnorm_bench B N D THREADS
 * Prints one CSV line: impl,lang,B,N,D,DFF,threads,seconds
 * DFF is not used by RMSNorm; printed as 0 for a uniform CSV schema
 * across rmsnorm_bench and mlp_bench.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double* alloc_rand(int n, unsigned seed) {
    double* a = malloc(sizeof(double) * n);
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        a[i] = ((double)(s % 20000) / 10000.0 - 1.0) * 0.5;
    }
    return a;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s B N D THREADS\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int N = atoi(argv[2]);
    int D = atoi(argv[3]);
    int THREADS = atoi(argv[4]);
    const double EPS = 1e-6;

    omp_set_num_threads(THREADS);

    int NT = B * N;
    double* X     = alloc_rand(NT*D, 1);
    double* gamma = alloc_rand(D, 2);
    double* GZ    = alloc_rand(NT*D, 3);
    double* Z     = malloc(sizeof(double) * NT*D);
    double* GX    = malloc(sizeof(double) * NT*D);
    double* Ggamma = calloc(D, sizeof(double));
    double* r     = malloc(sizeof(double) * NT);

    double t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double sumsq = 0.0;
        for (int d = 0; d < D; d++) {
            double x = X[t*D+d];
            sumsq += x*x;
        }
        r[t] = sqrt(sumsq / D + EPS);
    }
    #pragma omp parallel for collapse(2)
    for (int t = 0; t < NT; t++)
        for (int d = 0; d < D; d++)
            Z[t*D+d] = gamma[d] * (X[t*D+d] / r[t]);

    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double correction = 0.0;
        for (int d = 0; d < D; d++) {
            double x_hat_d = X[t*D+d] / r[t];
            correction += GZ[t*D+d] * gamma[d] * x_hat_d;
        }
        for (int e = 0; e < D; e++) {
            double x_hat_e = X[t*D+e] / r[t];
            GX[t*D+e] = (GZ[t*D+e] * gamma[e] - (x_hat_e / D) * correction) / r[t];
        }
    }
    #pragma omp parallel for reduction(+:Ggamma[0:D])
    for (int t = 0; t < NT; t++)
        for (int d = 0; d < D; d++)
            Ggamma[d] += GZ[t*D+d] * (X[t*D+d] / r[t]);

    double t1 = omp_get_wtime();

    printf("rmsnorm,C,%d,%d,%d,0,%d,%.6f\n", B, N, D, THREADS, t1 - t0);

    free(X); free(gamma); free(GZ); free(Z); free(GX); free(Ggamma); free(r);
    return 0;
}
