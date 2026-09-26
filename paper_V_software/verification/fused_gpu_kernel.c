/* C/OpenACC fused kernel for RMSNorm + gated MLP (SwiGLU), Sections 3-8.
 *
 * This extends the GPU kernel artifact from the inference paper (Paper III)
 * to the two new sublayers derived in this paper. It fuses:
 *   RMSNorm forward -> gated MLP forward -> RMSNorm backward -> gated MLP backward
 * on a single (B,N,D,D_ff) problem, with x_hat (RMSNorm) and V,H (MLP) recomputed
 * on-device rather than staged through a second HBM round trip, matching the
 * fused-cost derivations of Sections 5 and 7 and the transpose-elimination
 * lemma of Section 8 (every Wg/Wu/Wd/X/H "transpose" below is a reordered-index
 * read, never a materialized array).
 *
 * NOTE ON THIS ENVIRONMENT: compiled and run here via gcc's OpenACC host
 * fallback (`-fopenacc`), since no CUDA-capable GPU or nvc/pgcc toolchain is
 * available in this sandbox. Every `#pragma acc` below is real, standard
 * OpenACC and will offload to an actual GPU unmodified under nvc/nvc++ or
 * gcc built with an nvptx offload target -- but the timing/bandwidth numbers
 * from a run here are a correctness check only, not a GPU benchmark.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define B 2
#define N 4
#define D 8
#define DFF 16
#define NT (B*N)
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

static double silu(double u) { return u / (1.0 + exp(-u)); }
static double silu_grad(double u) {
    double s = 1.0 / (1.0 + exp(-u));
    return s * (1.0 + u * (1.0 - s));
}

int main() {
    double *X   = load_bin("data/X.bin", NT*D);
    double *gam = load_bin("data/gamma.bin", D);
    double *GZ  = load_bin("data/GZ.bin", NT*D);
    double *Wg  = load_bin("data/Wg.bin", D*DFF);
    double *Wu  = load_bin("data/Wu.bin", D*DFF);
    double *Wd  = load_bin("data/Wd.bin", DFF*D);
    double *GF  = load_bin("data/GF.bin", NT*D);

    double *Z  = malloc(sizeof(double)*NT*D);
    double *U  = malloc(sizeof(double)*NT*DFF);   /* retained, Section 7 tradeoff */
    double *F  = malloc(sizeof(double)*NT*D);
    double *r  = malloc(sizeof(double)*NT);
    double *GX_norm = calloc(NT*D, sizeof(double));
    double *Ggamma  = calloc(D, sizeof(double));
    double *GX_ffn  = calloc(NT*D, sizeof(double));
    double *GWg = calloc(D*DFF, sizeof(double));
    double *GWu = calloc(D*DFF, sizeof(double));
    double *GWd = calloc(DFF*D, sizeof(double));

    /* Single fused data region: X, weights, and gradients live on-device for
     * the whole block pass, matching the additive M_block accounting of
     * Section 9 -- one HBM round trip per array, not one per sublayer. */
    #pragma acc data copyin(X[0:NT*D], gam[0:D], GZ[0:NT*D], \
                             Wg[0:D*DFF], Wu[0:D*DFF], Wd[0:DFF*D], GF[0:NT*D]) \
                      copyout(Z[0:NT*D], F[0:NT*D], GX_norm[0:NT*D], Ggamma[0:D], \
                               GX_ffn[0:NT*D], GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D]) \
                      create(U[0:NT*DFF], r[0:NT])
    {
        /* ---- RMSNorm forward (Section 3) ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) {
                double x = X[t*D+d];
                sumsq += x*x;
            }
            r[t] = sqrt(sumsq / D + EPS);
        }
        #pragma acc parallel loop collapse(2)
        for (int t = 0; t < NT; t++)
            for (int d = 0; d < D; d++)
                Z[t*D+d] = gam[d] * (X[t*D+d] / r[t]);

        /* ---- Gated MLP forward (Section 6), reading Z as its input ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double Vrow[DFF], Hrow[DFF];
            for (int f = 0; f < DFF; f++) {
                double u = 0.0, v = 0.0;
                for (int d = 0; d < D; d++) {
                    u += Z[t*D+d] * Wg[d*DFF+f];
                    v += Z[t*D+d] * Wu[d*DFF+f];
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
        }

        /* ---- Gated MLP backward (Sections 6-7): V,H recomputed from
         * retained U; every W/X/H "transpose" is a reordered-index read
         * (Section 8), never a materialized array. Upstream grad is GF,
         * treating Z as the MLP's input for this fused-kernel slice. ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double Vrow[DFF], Hrow[DFF], GHrow[DFF], GUrow[DFF], GVrow[DFF];
            double GZ_from_ffn[D];
            for (int f = 0; f < DFF; f++) {
                double v = 0.0;
                for (int d = 0; d < D; d++)
                    v += Z[t*D+d] * Wu[d*DFF+f];
                Vrow[f] = v;
                Hrow[f] = silu(U[t*DFF+f]) * v;
            }
            for (int f = 0; f < DFF; f++) {
                for (int d = 0; d < D; d++) {
                    #pragma acc atomic update
                    GWd[f*D+d] += Hrow[f] * GF[t*D+d];
                }
            }
            for (int f = 0; f < DFF; f++) {
                double gh = 0.0;
                for (int d = 0; d < D; d++)
                    gh += GF[t*D+d] * Wd[f*D+d];
                GHrow[f] = gh;
                GVrow[f] = gh * silu(U[t*DFF+f]);
                GUrow[f] = gh * Vrow[f] * silu_grad(U[t*DFF+f]);
            }
            for (int d = 0; d < D; d++) {
                for (int f = 0; f < DFF; f++) {
                    #pragma acc atomic update
                    GWg[d*DFF+f] += Z[t*D+d] * GUrow[f];
                    #pragma acc atomic update
                    GWu[d*DFF+f] += Z[t*D+d] * GVrow[f];
                }
            }
            for (int d = 0; d < D; d++) {
                double gz = 0.0;
                for (int f = 0; f < DFF; f++)
                    gz += GUrow[f] * Wg[d*DFF+f] + GVrow[f] * Wu[d*DFF+f];
                GZ_from_ffn[d] = gz;  /* this becomes the upstream grad into RMSNorm backward */
            }
            /* ---- RMSNorm backward (Section 4), fused in the same kernel launch,
             * x_hat recomputed on-device rather than staged through HBM ---- */
            {
                double correction = 0.0;
                for (int d = 0; d < D; d++) {
                    double x_hat_d = X[t*D+d] / r[t];
                    correction += GZ_from_ffn[d] * gam[d] * x_hat_d;
                }
                for (int e = 0; e < D; e++) {
                    double x_hat_e = X[t*D+e] / r[t];
                    GX_norm[t*D+e] = (GZ_from_ffn[e] * gam[e]
                                       - (x_hat_e / D) * correction) / r[t];
                    #pragma acc atomic update
                    Ggamma[e] += GZ_from_ffn[e] * x_hat_e;
                }
            }
            for (int d = 0; d < D; d++)
                GX_ffn[t*D+d] = GZ_from_ffn[d];
        }
    }

    save_bin("data/g_Z.bin", Z, NT*D);
    save_bin("data/g_F.bin", F, NT*D);

    printf("Fused GPU (OpenACC host-fallback) kernel complete.\n");
    printf("Z[0..3] = %f %f %f %f\n", Z[0], Z[1], Z[2], Z[3]);
    printf("F[0..3] = %f %f %f %f\n", F[0], F[1], F[2], F[3]);

    return 0;
}
