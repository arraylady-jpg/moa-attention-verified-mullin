/* full_block_gpu_kernel.c
 *
 * Complete transformer block in a single OpenACC data region: Norm ->
 * QKV projection -> Attention -> residual -> Norm -> FFN -> residual,
 * forward and backward. Extends fused_gpu_kernel.c (norm+FFN only) to
 * include attention (Papers I-III) and the QKV projection needed to
 * connect Norm's output to Q,K,V. Mirrors full_block_verify.c exactly,
 * stage by stage, with OpenACC parallel loops in place of plain C loops
 * and all intermediates in one acc data region (single HBM round trip
 * per array across the whole block, per Section 9's M_block accounting).
 *
 * Verified against the PyTorch full-block reference (reference_full_block.py)
 * on B=2, N=4, D=8, D_ff=16.
 *
 * NOTE ON THIS ENVIRONMENT: compiled and run via GCC's OpenACC host
 * fallback (`-fopenacc`), since no CUDA-capable GPU or nvc toolchain is
 * available. Every #pragma acc below is standard OpenACC and should
 * offload to an actual GPU unmodified -- but the run performed here is
 * a correctness check only, not a GPU benchmark.
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
    double *R1 = malloc(sizeof(double)*NT*D);
    double *Z2 = malloc(sizeof(double)*NT*D);
    double *r2 = malloc(sizeof(double)*NT);
    double *U  = malloc(sizeof(double)*NT*DFF);
    double *Vg = malloc(sizeof(double)*NT*DFF);
    double *H  = malloc(sizeof(double)*NT*DFF);
    double *Y  = malloc(sizeof(double)*NT*D);

    double *GX  = calloc(NT*D, sizeof(double));
    double *GR1 = calloc(NT*D, sizeof(double));
    double *GZ1 = calloc(NT*D, sizeof(double));
    double *GZ2 = calloc(NT*D, sizeof(double));
    double *GQ  = calloc(NT*D, sizeof(double));
    double *GK  = calloc(NT*D, sizeof(double));
    double *GV  = calloc(NT*D, sizeof(double));
    double *Ggamma1 = calloc(D, sizeof(double));
    double *Ggamma2 = calloc(D, sizeof(double));
    double *GWq = calloc(D*D, sizeof(double));
    double *GWk = calloc(D*D, sizeof(double));
    double *GWv = calloc(D*D, sizeof(double));
    double *GWg = calloc(D*DFF, sizeof(double));
    double *GWu = calloc(D*DFF, sizeof(double));
    double *GWd = calloc(DFF*D, sizeof(double));

    const double scale = 1.0 / sqrt((double)D);

    #pragma acc data \
      copyin(X[0:NT*D], gamma1[0:D], gamma2[0:D], Wq[0:D*D], Wk[0:D*D], Wv[0:D*D], \
             Wg[0:D*DFF], Wu[0:D*DFF], Wd[0:DFF*D], GY[0:NT*D]) \
      create(Z1[0:NT*D], r1[0:NT], Q[0:NT*D], K[0:NT*D], V[0:NT*D], A[0:B*N*N], \
             R1[0:NT*D], Z2[0:NT*D], r2[0:NT], U[0:NT*DFF], Vg[0:NT*DFF], H[0:NT*DFF], \
             GR1[0:NT*D], GZ1[0:NT*D], GZ2[0:NT*D], GQ[0:NT*D], GK[0:NT*D], GV[0:NT*D]) \
      copyout(Y[0:NT*D], GX[0:NT*D], Ggamma1[0:D], Ggamma2[0:D], \
              GWq[0:D*D], GWk[0:D*D], GWv[0:D*D], GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])
    {
        /* ---- Norm 1 (Section 3) ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) { double x = X[t*D+d]; sumsq += x*x; }
            r1[t] = sqrt(sumsq / D + EPS);
            for (int d = 0; d < D; d++)
                Z1[t*D+d] = gamma1[d] * (X[t*D+d] / r1[t]);
        }

        /* ---- QKV projection ---- */
        #pragma acc parallel loop
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

        /* ---- Attention forward (Papers I-III) + residual 1, per batch/row ---- */
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double arow[N];
                double m = -1e300;
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
                    R1[(b*N+ir)*D+d] = X[(b*N+ir)*D+d] + o;
                }
            }
        }

        /* ---- Norm 2 (Section 3) ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double sumsq = 0.0;
            for (int d = 0; d < D; d++) { double x = R1[t*D+d]; sumsq += x*x; }
            r2[t] = sqrt(sumsq / D + EPS);
            for (int d = 0; d < D; d++)
                Z2[t*D+d] = gamma2[d] * (R1[t*D+d] / r2[t]);
        }

        /* ---- Gated MLP forward (Section 6) + residual 2 ---- */
        #pragma acc parallel loop
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
            for (int d = 0; d < D; d++) {
                double fo = 0.0;
                for (int f = 0; f < DFF; f++)
                    fo += H[t*DFF+f] * Wd[f*D+d];
                Y[t*D+d] = R1[t*D+d] + fo;
            }
        }

        /* ---- FFN backward (Sections 6-7), G_F = G_Y, feeding G_Z2; also
         * seed GR1 with the residual-2 direct path (G_R1 += G_Y) ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double GUrow[DFF], GVrow[DFF];
            for (int f = 0; f < DFF; f++) {
                double gh = 0.0;
                for (int d = 0; d < D; d++)
                    gh += GY[t*D+d] * Wd[f*D+d];        /* Wd^T, reordered read */
                GVrow[f] = gh * silu(U[t*DFF+f]);
                GUrow[f] = gh * Vg[t*DFF+f] * silu_grad(U[t*DFF+f]);
            }
            for (int f = 0; f < DFF; f++)
                for (int d = 0; d < D; d++) {
                    #pragma acc atomic update
                    GWd[f*D+d] += H[t*DFF+f] * GY[t*D+d];  /* H^T, reordered read */
                }
            for (int d = 0; d < D; d++)
                for (int f = 0; f < DFF; f++) {
                    #pragma acc atomic update
                    GWg[d*DFF+f] += Z2[t*D+d] * GUrow[f];  /* Z2^T, reordered read */
                    #pragma acc atomic update
                    GWu[d*DFF+f] += Z2[t*D+d] * GVrow[f];
                }
            double gz2;
            for (int d = 0; d < D; d++) {
                gz2 = 0.0;
                for (int f = 0; f < DFF; f++)
                    gz2 += GUrow[f]*Wg[d*DFF+f] + GVrow[f]*Wu[d*DFF+f]; /* Wg^T,Wu^T */
                GZ2[t*D+d] = gz2;
            }
            for (int d = 0; d < D; d++)
                GR1[t*D+d] = GY[t*D+d];  /* residual-2 direct path, before norm2 backward adds to it */
        }

        /* ---- Norm 2 backward (Section 4), accumulates into GR1 ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = R1[t*D+d] / r2[t];
                correction += GZ2[t*D+d] * gamma2[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = R1[t*D+e] / r2[t];
                GR1[t*D+e] += (GZ2[t*D+e]*gamma2[e] - (x_hat_e/D)*correction) / r2[t];
                #pragma acc atomic update
                Ggamma2[e] += GZ2[t*D+e] * x_hat_e;
            }
        }

        /* ---- Seed GX with residual-1 direct path: GX = GR1 (attention side added below) ---- */
        #pragma acc parallel loop
        for (int i = 0; i < NT*D; i++)
            GX[i] = GR1[i];

        /* ---- Attention backward (Papers I-III), per batch, feeding GQ,GK,GV ---- */
        #pragma acc parallel loop
        for (int b = 0; b < B; b++) {
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

        /* ---- QKV projection backward, feeding GZ1 ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            for (int d = 0; d < D; d++)
                for (int d2 = 0; d2 < D; d2++) {
                    #pragma acc atomic update
                    GWq[d*D+d2] += Z1[t*D+d] * GQ[t*D+d2];  /* Z1^T, reordered read */
                    #pragma acc atomic update
                    GWk[d*D+d2] += Z1[t*D+d] * GK[t*D+d2];
                    #pragma acc atomic update
                    GWv[d*D+d2] += Z1[t*D+d] * GV[t*D+d2];
                }
            for (int d = 0; d < D; d++) {
                double g = 0.0;
                for (int d2 = 0; d2 < D; d2++)
                    g += GQ[t*D+d2]*Wq[d*D+d2] + GK[t*D+d2]*Wk[d*D+d2] + GV[t*D+d2]*Wv[d*D+d2]; /* Wq^T,Wk^T,Wv^T */
                GZ1[t*D+d] = g;
            }
        }

        /* ---- Norm 1 backward (Section 4), accumulates into GX ---- */
        #pragma acc parallel loop
        for (int t = 0; t < NT; t++) {
            double correction = 0.0;
            for (int d = 0; d < D; d++) {
                double x_hat_d = X[t*D+d] / r1[t];
                correction += GZ1[t*D+d] * gamma1[d] * x_hat_d;
            }
            for (int e = 0; e < D; e++) {
                double x_hat_e = X[t*D+e] / r1[t];
                GX[t*D+e] += (GZ1[t*D+e]*gamma1[e] - (x_hat_e/D)*correction) / r1[t];
                #pragma acc atomic update
                Ggamma1[e] += GZ1[t*D+e] * x_hat_e;
            }
        }
    }

    save_bin("data/fb_g_Y.bin", Y, NT*D);
    save_bin("data/fb_g_GX.bin", GX, NT*D);
    save_bin("data/fb_g_Ggamma1.bin", Ggamma1, D);
    save_bin("data/fb_g_Ggamma2.bin", Ggamma2, D);
    save_bin("data/fb_g_GWq.bin", GWq, D*D);
    save_bin("data/fb_g_GWk.bin", GWk, D*D);
    save_bin("data/fb_g_GWv.bin", GWv, D*D);
    save_bin("data/fb_g_GWg.bin", GWg, D*DFF);
    save_bin("data/fb_g_GWu.bin", GWu, D*DFF);
    save_bin("data/fb_g_GWd.bin", GWd, DFF*D);

    printf("Full block GPU (OpenACC host-fallback) kernel complete.\n");
    printf("Y[0..3] = %f %f %f %f\n", Y[0], Y[1], Y[2], Y[3]);
    printf("GX[0..3] = %f %f %f %f\n", GX[0], GX[1], GX[2], GX[3]);

    return 0;
}
