/* full_block_verify.c
 *
 * Complete transformer block, forward + backward, integrating:
 *   - RMSNorm (this paper, Sections 3-5)
 *   - QKV projection (new: Z -> Q,K,V via learned D x D weights,
 *     needed to connect Norm's output to moa_attention.c's DNF, which
 *     takes Q,K,V as already-given arrays)
 *   - Attention (Papers I-III; DNF forward/backward re-derived inline
 *     here in the same style as moa_attention.c's moa_forward_dnf /
 *     moa_backward_dnf, since that file's functions use fixed macros
 *     B=2,N=4,D=8,T=4 rather than being directly callable as a library)
 *   - Gated MLP (this paper, Sections 6-8)
 * with two residual connections, matching Figure 1 / Section 9 of the
 * paper. Verified against a PyTorch double-precision autograd reference
 * (reference_full_block.py) on B=2, N=4, D=8, D_ff=16.
 *
 * Every "transpose" access below (Wq^T, Wk^T, Wv^T, Wg^T, Wu^T, Wd^T,
 * Z1^T, Z2^T, K^T) is a reordered-index read, never a materialized
 * array -- extending the Section 8 transpose-elimination lemma to the
 * QKV projection weights as well as the FFN weights.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define B 2
#define N 4
#define D 8
#define DFF 16
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
    int NT = B * N;
    double *X = load_bin("data/fb_X.bin", NT*D);
    double *gamma1 = load_bin("data/fb_gamma1.bin", D);
    double *gamma2 = load_bin("data/fb_gamma2.bin", D);
    double *Wq = load_bin("data/fb_Wq.bin", D*D);
    double *Wk = load_bin("data/fb_Wk.bin", D*D);
    double *Wv = load_bin("data/fb_Wv.bin", D*D);
    double *Wg = load_bin("data/fb_Wg.bin", D*DFF);
    double *Wu = load_bin("data/fb_Wu.bin", D*DFF);
    double *Wd = load_bin("data/fb_Wd.bin", DFF*D);
    double *GY = load_bin("data/fb_GY.bin", NT*D);

    double *Z1 = malloc(sizeof(double)*NT*D);
    double *r1 = malloc(sizeof(double)*NT);
    double *Q  = malloc(sizeof(double)*NT*D);
    double *K  = malloc(sizeof(double)*NT*D);
    double *V  = malloc(sizeof(double)*NT*D);
    double *A  = malloc(sizeof(double)*B*N*N);
    double *AttnOut = malloc(sizeof(double)*NT*D);
    double *R1 = malloc(sizeof(double)*NT*D);
    double *Z2 = malloc(sizeof(double)*NT*D);
    double *r2 = malloc(sizeof(double)*NT);
    double *U  = malloc(sizeof(double)*NT*DFF);
    double *H  = malloc(sizeof(double)*NT*DFF);
    double *Y  = malloc(sizeof(double)*NT*D);

    const double scale = 1.0 / sqrt((double)D);

    /* ================= FORWARD ================= */

    /* ---- Norm 1: Sections 3 ---- */
    for (int t = 0; t < NT; t++) {
        double sumsq = 0.0;
        for (int d = 0; d < D; d++) { double x = X[t*D+d]; sumsq += x*x; }
        r1[t] = sqrt(sumsq / D + EPS);
        for (int d = 0; d < D; d++)
            Z1[t*D+d] = gamma1[d] * (X[t*D+d] / r1[t]);
    }

    /* ---- QKV projection: Q=Z1@Wq, K=Z1@Wk, V=Z1@Wv ---- */
    for (int t = 0; t < NT; t++) {
        for (int d2 = 0; d2 < D; d2++) {
            double q=0, k=0, v=0;
            for (int d = 0; d < D; d++) {
                q += Z1[t*D+d] * Wq[d*D+d2];
                k += Z1[t*D+d] * Wk[d*D+d2];
                v += Z1[t*D+d] * Wv[d*D+d2];
            }
            Q[t*D+d2] = q; K[t*D+d2] = k; V[t*D+d2] = v;
        }
    }

    /* ---- Attention forward (DNF, Papers I-III eqs 17,20,21), per batch ---- */
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            double arow[N];
            double m = -DBL_MAX;
            for (int ic = 0; ic < N; ic++) {
                double dot = 0.0;
                for (int j = 0; j < D; j++)
                    dot += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                arow[ic] = scale * dot;
                if (arow[ic] > m) m = arow[ic];
            }
            double s = 0.0;
            for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic]-m); s += arow[ic]; }
            for (int ic = 0; ic < N; ic++) { arow[ic] /= s; A[(b*N+ir)*N+ic] = arow[ic]; }
            for (int d = 0; d < D; d++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++)
                    o += arow[ic] * V[(b*N+ic)*D+d];
                AttnOut[(b*N+ir)*D+d] = o;
            }
        }
    }

    /* ---- Residual 1 ---- */
    for (int i = 0; i < NT*D; i++) R1[i] = X[i] + AttnOut[i];

    /* ---- Norm 2: Sections 3 ---- */
    for (int t = 0; t < NT; t++) {
        double sumsq = 0.0;
        for (int d = 0; d < D; d++) { double x = R1[t*D+d]; sumsq += x*x; }
        r2[t] = sqrt(sumsq / D + EPS);
        for (int d = 0; d < D; d++)
            Z2[t*D+d] = gamma2[d] * (R1[t*D+d] / r2[t]);
    }

    /* ---- Gated MLP forward: Sections 6 ---- */
    double *Vg = malloc(sizeof(double)*NT*DFF);
    for (int t = 0; t < NT; t++) {
        for (int f = 0; f < DFF; f++) {
            double u=0, v=0;
            for (int d = 0; d < D; d++) {
                u += Z2[t*D+d] * Wg[d*DFF+f];
                v += Z2[t*D+d] * Wu[d*DFF+f];
            }
            U[t*DFF+f] = u; Vg[t*DFF+f] = v;
            H[t*DFF+f] = silu(u) * v;
        }
    }
    for (int t = 0; t < NT; t++)
        for (int d = 0; d < D; d++) {
            double fo = 0.0;
            for (int f = 0; f < DFF; f++)
                fo += H[t*DFF+f] * Wd[f*D+d];
            Y[t*D+d] = R1[t*D+d] + fo; /* Residual 2 fused in */
        }

    save_bin("data/fb_c_Y.bin", Y, NT*D);

    /* ================= BACKWARD ================= */
    double *GR1 = calloc(NT*D, sizeof(double));
    double *GZ2 = calloc(NT*D, sizeof(double));
    double *GWg = calloc(D*DFF, sizeof(double));
    double *GWu = calloc(D*DFF, sizeof(double));
    double *GWd = calloc(DFF*D, sizeof(double));
    double *Ggamma2 = calloc(D, sizeof(double));

    /* G_F = G_Y (residual 2 passes straight through to R1 too) */
    for (int i = 0; i < NT*D; i++) GR1[i] += GY[i]; /* resid path */

    /* FFN backward: Sections 6-7, feeding G_Z2 */
    for (int t = 0; t < NT; t++) {
        double GHrow[DFF], GUrow[DFF], GVrow[DFF];
        for (int f = 0; f < DFF; f++) {
            double gh = 0.0;
            for (int d = 0; d < D; d++)
                gh += GY[t*D+d] * Wd[f*D+d];      /* Wd^T, reordered-index read */
            GHrow[f] = gh;
            GVrow[f] = gh * silu(U[t*DFF+f]);
            GUrow[f] = gh * Vg[t*DFF+f] * silu_grad(U[t*DFF+f]);
        }
        for (int f = 0; f < DFF; f++)
            for (int d = 0; d < D; d++)
                GWd[f*D+d] += H[t*DFF+f] * GY[t*D+d];   /* H^T, reordered-index read */
        for (int d = 0; d < D; d++)
            for (int f = 0; f < DFF; f++) {
                GWg[d*DFF+f] += Z2[t*D+d] * GUrow[f];   /* Z2^T, reordered-index read */
                GWu[d*DFF+f] += Z2[t*D+d] * GVrow[f];
            }
        for (int d = 0; d < D; d++) {
            double gz = 0.0;
            for (int f = 0; f < DFF; f++)
                gz += GUrow[f]*Wg[d*DFF+f] + GVrow[f]*Wu[d*DFF+f]; /* Wg^T,Wu^T */
            GZ2[t*D+d] = gz;
        }
    }

    /* Norm 2 backward: Sections 4, feeding into G_R1 (added to residual path) */
    for (int t = 0; t < NT; t++) {
        double correction = 0.0;
        for (int d = 0; d < D; d++) {
            double x_hat_d = R1[t*D+d] / r2[t];
            correction += GZ2[t*D+d] * gamma2[d] * x_hat_d;
        }
        for (int e = 0; e < D; e++) {
            double x_hat_e = R1[t*D+e] / r2[t];
            GR1[t*D+e] += (GZ2[t*D+e]*gamma2[e] - (x_hat_e/D)*correction) / r2[t];
            Ggamma2[e] += GZ2[t*D+e] * x_hat_e;
        }
    }

    /* G_AttnOut = G_R1 (residual 1: also flows to G_X directly) */
    double *GX = calloc(NT*D, sizeof(double));
    for (int i = 0; i < NT*D; i++) GX[i] += GR1[i]; /* resid 1 path */

    /* Attention backward (Papers I-III Algorithm 1), feeding G_Q,G_K,G_V */
    double *GQ = calloc(NT*D, sizeof(double));
    double *GK = calloc(NT*D, sizeof(double));
    double *GV = calloc(NT*D, sizeof(double));
    for (int b = 0; b < B; b++) {
        /* GV[ic,d] = sum_ir A[ir,ic] * G_AttnOut[ir,d] */
        for (int ic = 0; ic < N; ic++)
            for (int d = 0; d < D; d++) {
                double gv = 0.0;
                for (int ir = 0; ir < N; ir++)
                    gv += A[(b*N+ir)*N+ic] * GR1[(b*N+ir)*D+d];
                GV[(b*N+ic)*D+d] = gv;
            }
        for (int ir = 0; ir < N; ir++) {
            double ga_row[N];
            for (int ic = 0; ic < N; ic++) {
                double g = 0.0;
                for (int d = 0; d < D; d++)
                    g += GR1[(b*N+ir)*D+d] * V[(b*N+ic)*D+d];
                ga_row[ic] = g;
            }
            double corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += A[(b*N+ir)*N+ic] * ga_row[ic];
            for (int ic = 0; ic < N; ic++) {
                double gs = scale * A[(b*N+ir)*N+ic] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                    GK[(b*N+ic)*D+j] += gs * Q[(b*N+ir)*D+j];
                }
            }
        }
    }

    /* QKV projection backward, feeding G_Z1 */
    double *GZ1 = calloc(NT*D, sizeof(double));
    double *GWq = calloc(D*D, sizeof(double));
    double *GWk = calloc(D*D, sizeof(double));
    double *GWv = calloc(D*D, sizeof(double));
    for (int t = 0; t < NT; t++) {
        for (int d = 0; d < D; d++)
            for (int d2 = 0; d2 < D; d2++) {
                GWq[d*D+d2] += Z1[t*D+d] * GQ[t*D+d2]; /* Z1^T */
                GWk[d*D+d2] += Z1[t*D+d] * GK[t*D+d2];
                GWv[d*D+d2] += Z1[t*D+d] * GV[t*D+d2];
            }
        for (int d = 0; d < D; d++) {
            double g = 0.0;
            for (int d2 = 0; d2 < D; d2++)
                g += GQ[t*D+d2]*Wq[d*D+d2] + GK[t*D+d2]*Wk[d*D+d2] + GV[t*D+d2]*Wv[d*D+d2]; /* Wq^T,Wk^T,Wv^T */
            GZ1[t*D+d] = g;
        }
    }

    /* Norm 1 backward: Sections 4, feeding into G_X (added to residual path) */
    double *Ggamma1 = calloc(D, sizeof(double));
    for (int t = 0; t < NT; t++) {
        double correction = 0.0;
        for (int d = 0; d < D; d++) {
            double x_hat_d = X[t*D+d] / r1[t];
            correction += GZ1[t*D+d] * gamma1[d] * x_hat_d;
        }
        for (int e = 0; e < D; e++) {
            double x_hat_e = X[t*D+e] / r1[t];
            GX[t*D+e] += (GZ1[t*D+e]*gamma1[e] - (x_hat_e/D)*correction) / r1[t];
            Ggamma1[e] += GZ1[t*D+e] * x_hat_e;
        }
    }

    save_bin("data/fb_c_GX.bin", GX, NT*D);
    save_bin("data/fb_c_Ggamma1.bin", Ggamma1, D);
    save_bin("data/fb_c_Ggamma2.bin", Ggamma2, D);
    save_bin("data/fb_c_GWq.bin", GWq, D*D);
    save_bin("data/fb_c_GWk.bin", GWk, D*D);
    save_bin("data/fb_c_GWv.bin", GWv, D*D);
    save_bin("data/fb_c_GWg.bin", GWg, D*DFF);
    save_bin("data/fb_c_GWu.bin", GWu, D*DFF);
    save_bin("data/fb_c_GWd.bin", GWd, DFF*D);

    printf("Full block forward+backward complete. Y[0..3] = %f %f %f %f\n",
           Y[0], Y[1], Y[2], Y[3]);
    printf("GX[0..3] = %f %f %f %f\n", GX[0], GX[1], GX[2], GX[3]);

    return 0;
}
