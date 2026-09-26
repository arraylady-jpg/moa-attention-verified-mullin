#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define B 2
#define N 4
#define D 8
#define DFF 16

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

static double silu(double u) { return u / (1.0 + exp(-u)); }
static double silu_grad(double u) {
    double s = 1.0 / (1.0 + exp(-u));
    return s * (1.0 + u * (1.0 - s));
}

int main() {
    int NT = B * N;
    double* X  = load_bin("data/X.bin", NT*D);
    double* Wg = load_bin("data/Wg.bin", D*DFF);
    double* Wu = load_bin("data/Wu.bin", D*DFF);
    double* Wd = load_bin("data/Wd.bin", DFF*D);
    double* GF = load_bin("data/GF.bin", NT*D);

    double* U = malloc(sizeof(double) * NT*DFF); /* retained (Section 7 tradeoff) */
    double* F = malloc(sizeof(double) * NT*D);

    /* ---- Forward DNF (Section 6): U = X x Wg, V = X x Wu (not retained), H = silu(U)*V, F = H x Wd ---- */
    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double Vrow[DFF], Hrow[DFF];
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
            double f_out = 0.0;
            for (int f = 0; f < DFF; f++)
                f_out += Hrow[f] * Wd[f*D+d];
            F[t*D+d] = f_out;
        }
    }
    save_bin("data/c_F.bin", F, NT*D);

    /* ---- Backward DNF (Section 6-7): V,H recomputed on the fly from retained U, no transpose materialized ---- */
    double* GX  = calloc(NT*D, sizeof(double));
    double* GWg = calloc(D*DFF, sizeof(double));
    double* GWu = calloc(D*DFF, sizeof(double));
    double* GWd = calloc(DFF*D, sizeof(double));

    #pragma omp parallel for reduction(+:GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])
    for (int t = 0; t < NT; t++) {
        double Vrow[DFF], Hrow[DFF], GHrow[DFF], GUrow[DFF], GVrow[DFF];
        /* recompute V and H from X and retained U (transpose-free psi-style access) */
        for (int f = 0; f < DFF; f++) {
            double v = 0.0;
            for (int d = 0; d < D; d++)
                v += X[t*D+d] * Wu[d*DFF+f];
            Vrow[f] = v;
            Hrow[f] = silu(U[t*DFF+f]) * v;
        }
        /* G_Wd = H^T x G_F  -- realized via reordered index access, H never transposed */
        for (int f = 0; f < DFF; f++)
            for (int d = 0; d < D; d++)
                GWd[f*D+d] += Hrow[f] * GF[t*D+d];

        /* G_H = G_F x Wd^T -- Wd accessed transposed via reordered index, not materialized */
        for (int f = 0; f < DFF; f++) {
            double gh = 0.0;
            for (int d = 0; d < D; d++)
                gh += GF[t*D+d] * Wd[f*D+d];
            GHrow[f] = gh;
        }

        for (int f = 0; f < DFF; f++) {
            GVrow[f] = GHrow[f] * silu(U[t*DFF+f]);
            GUrow[f] = GHrow[f] * Vrow[f] * silu_grad(U[t*DFF+f]);
        }

        /* G_Wg = X^T x G_U, G_Wu = X^T x G_V -- X accessed transposed via reordered index */
        for (int d = 0; d < D; d++)
            for (int f = 0; f < DFF; f++) {
                GWg[d*DFF+f] += X[t*D+d] * GUrow[f];
                GWu[d*DFF+f] += X[t*D+d] * GVrow[f];
            }

        /* G_X = G_U x Wg^T + G_V x Wu^T -- Wg, Wu accessed transposed via reordered index */
        for (int d = 0; d < D; d++) {
            double gx = 0.0;
            for (int f = 0; f < DFF; f++)
                gx += GUrow[f] * Wg[d*DFF+f] + GVrow[f] * Wu[d*DFF+f];
            GX[t*D+d] = gx;
        }
    }

    save_bin("data/c_GX_ffn.bin", GX, NT*D);
    save_bin("data/c_GWg.bin", GWg, D*DFF);
    save_bin("data/c_GWu.bin", GWu, D*DFF);
    save_bin("data/c_GWd.bin", GWd, DFF*D);

    printf("Gated MLP forward+backward complete. F[0,0..3] = %f %f %f %f\n",
           F[0], F[1], F[2], F[3]);

    free(X); free(Wg); free(Wu); free(Wd); free(GF);
    free(U); free(F); free(GX); free(GWg); free(GWu); free(GWd);
    return 0;
}
