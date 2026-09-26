/* Parameterized gated MLP (SwiGLU) forward+backward benchmark (C/OpenMP).
 * Usage: ./mlp_bench B N D DFF THREADS
 * Prints one CSV line: impl,lang,B,N,D,DFF,threads,seconds
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
        a[i] = ((double)(s % 20000) / 10000.0 - 1.0) * 0.3;
    }
    return a;
}

static double silu(double u) { return u / (1.0 + exp(-u)); }
static double silu_grad(double u) {
    double s = 1.0 / (1.0 + exp(-u));
    return s * (1.0 + u * (1.0 - s));
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s B N D DFF THREADS\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int N = atoi(argv[2]);
    int D = atoi(argv[3]);
    int DFF = atoi(argv[4]);
    int THREADS = atoi(argv[5]);

    omp_set_num_threads(THREADS);

    int NT = B * N;
    double* X  = alloc_rand(NT*D, 11);
    double* Wg = alloc_rand(D*DFF, 12);
    double* Wu = alloc_rand(D*DFF, 13);
    double* Wd = alloc_rand(DFF*D, 14);
    double* GF = alloc_rand(NT*D, 15);
    double* U  = malloc(sizeof(double) * NT*DFF);
    double* F  = malloc(sizeof(double) * NT*D);
    double* GX  = calloc(NT*D, sizeof(double));
    double* GWg = calloc(D*DFF, sizeof(double));
    double* GWu = calloc(D*DFF, sizeof(double));
    double* GWd = calloc(DFF*D, sizeof(double));

    double t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double* Vrow = malloc(sizeof(double)*DFF);
        double* Hrow = malloc(sizeof(double)*DFF);
        for (int f = 0; f < DFF; f++) {
            double u = 0.0, v = 0.0;
            for (int d = 0; d < D; d++) {
                u += X[t*D+d] * Wg[d*DFF+f];
                v += X[t*D+d] * Wu[d*DFF+f];
            }
            U[t*DFF+f] = u;
            Vrow[f] = v;
            Hrow[f] = silu(u) * v;
        }
        for (int d = 0; d < D; d++) {
            double fo = 0.0;
            for (int f = 0; f < DFF; f++)
                fo += Hrow[f] * Wd[f*D+d];
            F[t*D+d] = fo;
        }
        free(Vrow); free(Hrow);
    }

    #pragma omp parallel for reduction(+:GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])
    for (int t = 0; t < NT; t++) {
        double* Vrow = malloc(sizeof(double)*DFF);
        double* Hrow = malloc(sizeof(double)*DFF);
        double* GHrow = malloc(sizeof(double)*DFF);
        double* GUrow = malloc(sizeof(double)*DFF);
        double* GVrow = malloc(sizeof(double)*DFF);
        for (int f = 0; f < DFF; f++) {
            double v = 0.0;
            for (int d = 0; d < D; d++)
                v += X[t*D+d] * Wu[d*DFF+f];
            Vrow[f] = v;
            Hrow[f] = silu(U[t*DFF+f]) * v;
        }
        for (int f = 0; f < DFF; f++)
            for (int d = 0; d < D; d++)
                GWd[f*D+d] += Hrow[f] * GF[t*D+d];
        for (int f = 0; f < DFF; f++) {
            double gh = 0.0;
            for (int d = 0; d < D; d++)
                gh += GF[t*D+d] * Wd[f*D+d];
            GHrow[f] = gh;
            GVrow[f] = gh * silu(U[t*DFF+f]);
            GUrow[f] = gh * Vrow[f] * silu_grad(U[t*DFF+f]);
        }
        for (int d = 0; d < D; d++)
            for (int f = 0; f < DFF; f++) {
                GWg[d*DFF+f] += X[t*D+d] * GUrow[f];
                GWu[d*DFF+f] += X[t*D+d] * GVrow[f];
            }
        for (int d = 0; d < D; d++) {
            double gx = 0.0;
            for (int f = 0; f < DFF; f++)
                gx += GUrow[f] * Wg[d*DFF+f] + GVrow[f] * Wu[d*DFF+f];
            GX[t*D+d] = gx;
        }
        free(Vrow); free(Hrow); free(GHrow); free(GUrow); free(GVrow);
    }

    double t1 = omp_get_wtime();

    printf("mlp,C,%d,%d,%d,%d,%d,%.6f\n", B, N, D, DFF, THREADS, t1 - t0);

    free(X); free(Wg); free(Wu); free(Wd); free(GF); free(U); free(F);
    free(GX); free(GWg); free(GWu); free(GWd);
    return 0;
}
