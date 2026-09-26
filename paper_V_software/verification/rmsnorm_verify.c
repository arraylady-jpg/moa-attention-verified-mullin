#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define B 2
#define N 4
#define D 8
#define EPS 1e-6

static double* load_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    double* arr = malloc(sizeof(double) * count);
    fread(arr, sizeof(double), count, f);
    fclose(f);
    return arr;
}

static void save_bin(const char* path, double* arr, int count) {
    FILE* f = fopen(path, "wb");
    fwrite(arr, sizeof(double), count, f);
    fclose(f);
}

int main() {
    double* X = load_bin("data/X.bin", B*N*D);
    double* gamma = load_bin("data/gamma.bin", D);
    double* GZ = load_bin("data/GZ.bin", B*N*D);

    double* Z = malloc(sizeof(double) * B*N*D);
    double* GX = malloc(sizeof(double) * B*N*D);
    double* Ggamma = calloc(D, sizeof(double));
    double* r = malloc(sizeof(double) * B*N);

    /* ---- Forward DNF (Section 3): reduction pass, then elementwise pass ---- */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) {
                double x = X[(b*N+n)*D + d];
                sumsq += x * x;
            }
            r[b*N+n] = sqrt(sumsq / D + EPS);
        }
    }
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++)
        for (int n = 0; n < N; n++)
            for (int d = 0; d < D; d++) {
                double x = X[(b*N+n)*D + d];
                Z[(b*N+n)*D + d] = gamma[d] * (x / r[b*N+n]);
            }

    save_bin("data/c_Z.bin", Z, B*N*D);

    /* ---- Backward DNF (Section 4): fused pass, x_hat recomputed, never stored ---- */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            double rr = r[b*N+n];
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = X[(b*N+n)*D + d] / rr; /* recomputed, not stored */
                correction += GZ[(b*N+n)*D + d] * gamma[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = X[(b*N+n)*D + e] / rr; /* recomputed again */
                GX[(b*N+n)*D + e] = (GZ[(b*N+n)*D + e] * gamma[e]
                                      - (x_hat_e / D) * correction) / rr;
            }
        }
    }

    /* G_gamma: reduction over batch and sequence, parallel with array reduction */
    #pragma omp parallel for collapse(2) reduction(+:Ggamma[0:D])
    for (int b = 0; b < B; b++)
        for (int n = 0; n < N; n++) {
            double rr = r[b*N+n];
            for (int d = 0; d < D; d++) {
                double x_hat_d = X[(b*N+n)*D + d] / rr;
                Ggamma[d] += GZ[(b*N+n)*D + d] * x_hat_d;
            }
        }

    save_bin("data/c_GX_norm.bin", GX, B*N*D);
    save_bin("data/c_Ggamma.bin", Ggamma, D);

    printf("RMSNorm forward+backward complete. Z[0,0..3] = %f %f %f %f\n",
           Z[0], Z[1], Z[2], Z[3]);

    free(X); free(gamma); free(GZ); free(Z); free(GX); free(Ggamma); free(r);
    return 0;
}
