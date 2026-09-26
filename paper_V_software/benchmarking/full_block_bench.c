/* full_block_bench.c
 * Parameterized, timed OpenMP benchmark for the complete transformer block:
 * Norm -> QKV projection -> Attention -> residual -> Norm -> FFN -> residual.
 *
 * REVISION HISTORY:
 * v1: single untimed-nothing, one-shot-per-size measurement -- the
 *     same methodological gap already documented for other CPU
 *     benchmarks in this project (Section 4.1's crash-fix commentary,
 *     Section 6.1's discussion of this file's own real data). Real
 *     Delta data (job 21273806) showed a real N=2048->4096 growth
 *     ratio of 11.7x, far beyond O(n^2)'s predicted 4x -- flagged
 *     honestly as either genuine cache-capacity effects (A is
 *     ~268MB per batch element at N=4096) or single-shot measurement
 *     noise, not yet distinguished (results_paper/moa_experimental_
 *     validation.tex Section 6.1).
 * v2 (this version): repeated-and-averaged timing added, matching
 *     moa_decode_bench.c's already-correct pattern and the same fix
 *     already applied to every GPU benchmark in this project. The
 *     entire computation (previously inline in main()) is now wrapped
 *     in full_block_once(), called once untimed as warmup, then
 *     REPEATS times with the mean reported. If the N=4096 anomaly
 *     persists under averaging, that argues for a genuine effect
 *     (e.g. cache capacity); if it disappears or shrinks toward ~4x,
 *     that confirms it was single-shot noise. Not yet re-run on real
 *     hardware to check which.
 *
 * Usage: ./full_block_bench B N D DFF THREADS [REPEATS]
 * Prints one CSV line: impl,lang,B,N,D,DFF,threads,seconds,repeats
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

/* One full-block call: B,N,D,DFF fixed, allocates fresh each call
 * (matching moa_decode_bench.c's and every GPU bench file's pattern),
 * times just the compute region. All computational logic below is
 * unchanged from v1, line for line -- only the surrounding
 * function/repeat structure is new. */
static double full_block_once(int B, int N, int D, int DFF) {
    const double EPS = 1e-6;
    int NT = B * N;

    double *X = alloc_rand(NT*D, 1);
    double *gamma1 = alloc_rand(D, 2);
    double *gamma2 = alloc_rand(D, 3);
    double *Wq = alloc_rand(D*D, 4);
    double *Wk = alloc_rand(D*D, 5);
    double *Wv = alloc_rand(D*D, 6);
    double *Wg = alloc_rand(D*DFF, 7);
    double *Wu = alloc_rand(D*DFF, 8);
    double *Wd = alloc_rand(DFF*D, 9);
    double *GY = alloc_rand(NT*D, 10);

    double *Z1 = malloc(sizeof(double)*NT*D);
    double *r1 = malloc(sizeof(double)*NT);
    double *Q  = malloc(sizeof(double)*NT*D);
    double *K  = malloc(sizeof(double)*NT*D);
    double *V  = malloc(sizeof(double)*NT*D);
    double *A  = malloc(sizeof(double)*(long)B*N*N);
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

    double t0 = omp_get_wtime();

    /* ---- Norm 1 ---- */
    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double sumsq = 0.0;
        for (int d = 0; d < D; d++) { double x = X[t*D+d]; sumsq += x*x; }
        r1[t] = sqrt(sumsq / D + EPS);
        for (int d = 0; d < D; d++)
            Z1[t*D+d] = gamma1[d] * (X[t*D+d] / r1[t]);
    }

    /* ---- QKV projection ---- */
    #pragma omp parallel for
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

    /* ---- Attention forward + residual 1 ---- */
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            double *arow = malloc(sizeof(double)*N);
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
            for (int ic = 0; ic < N; ic++) { arow[ic] /= s; A[(long)(b*N+ir)*N+ic] = arow[ic]; }
            for (int d = 0; d < D; d++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++)
                    o += arow[ic] * V[(b*N+ic)*D+d];
                R1[(b*N+ir)*D+d] = X[(b*N+ir)*D+d] + o;
            }
            free(arow);
        }
    }

    /* ---- Norm 2 ---- */
    #pragma omp parallel for
    for (int t = 0; t < NT; t++) {
        double sumsq = 0.0;
        for (int d = 0; d < D; d++) { double x = R1[t*D+d]; sumsq += x*x; }
        r2[t] = sqrt(sumsq / D + EPS);
        for (int d = 0; d < D; d++)
            Z2[t*D+d] = gamma2[d] * (R1[t*D+d] / r2[t]);
    }

    /* ---- Gated MLP forward + residual 2 ---- */
    #pragma omp parallel for
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

    /* ---- FFN backward, seeds GR1 with residual-2 path ---- */
    #pragma omp parallel for reduction(+:GWg[0:D*DFF], GWu[0:D*DFF], GWd[0:DFF*D])
    for (int t = 0; t < NT; t++) {
        double *GUrow = malloc(sizeof(double)*DFF);
        double *GVrow = malloc(sizeof(double)*DFF);
        for (int f = 0; f < DFF; f++) {
            double gh = 0.0;
            for (int d = 0; d < D; d++)
                gh += GY[t*D+d] * Wd[f*D+d];
            GVrow[f] = gh * silu(U[t*DFF+f]);
            GUrow[f] = gh * Vg[t*DFF+f] * silu_grad(U[t*DFF+f]);
        }
        for (int f = 0; f < DFF; f++)
            for (int d = 0; d < D; d++)
                GWd[f*D+d] += H[t*DFF+f] * GY[t*D+d];
        for (int d = 0; d < D; d++)
            for (int f = 0; f < DFF; f++) {
                GWg[d*DFF+f] += Z2[t*D+d] * GUrow[f];
                GWu[d*DFF+f] += Z2[t*D+d] * GVrow[f];
            }
        double gz2;
        for (int d = 0; d < D; d++) {
            gz2 = 0.0;
            for (int f = 0; f < DFF; f++)
                gz2 += GUrow[f]*Wg[d*DFF+f] + GVrow[f]*Wu[d*DFF+f];
            GZ2[t*D+d] = gz2;
        }
        for (int d = 0; d < D; d++)
            GR1[t*D+d] = GY[t*D+d];
        free(GUrow); free(GVrow);
    }

    /* ---- Norm 2 backward ---- */
    #pragma omp parallel for reduction(+:Ggamma2[0:D])
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

    #pragma omp parallel for
    for (int i = 0; i < NT*D; i++)
        GX[i] = GR1[i];

    /* ---- Attention backward ---- */
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int ic = 0; ic < N; ic++)
            for (int d = 0; d < D; d++) {
                double gv = 0.0;
                for (int ir = 0; ir < N; ir++)
                    gv += A[(long)(b*N+ir)*N+ic] * GR1[(b*N+ir)*D+d];
                GV[(b*N+ic)*D+d] = gv;
            }
        for (int ir = 0; ir < N; ir++) {
            double *ga_row = malloc(sizeof(double)*N);
            for (int ic = 0; ic < N; ic++) {
                double g = 0.0;
                for (int d = 0; d < D; d++)
                    g += GR1[(b*N+ir)*D+d] * V[(b*N+ic)*D+d];
                ga_row[ic] = g;
            }
            double corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += A[(long)(b*N+ir)*N+ic] * ga_row[ic];
            for (int ic = 0; ic < N; ic++) {
                double gs = scale * A[(long)(b*N+ir)*N+ic] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[(b*N+ir)*D+j] += gs * K[(b*N+ic)*D+j];
                    GK[(b*N+ic)*D+j] += gs * Q[(b*N+ir)*D+j];
                }
            }
            free(ga_row);
        }
    }

    /* ---- QKV projection backward ---- */
    #pragma omp parallel for reduction(+:GWq[0:D*D], GWk[0:D*D], GWv[0:D*D])
    for (int t = 0; t < NT; t++) {
        for (int d = 0; d < D; d++)
            for (int d2 = 0; d2 < D; d2++) {
                GWq[d*D+d2] += Z1[t*D+d] * GQ[t*D+d2];
                GWk[d*D+d2] += Z1[t*D+d] * GK[t*D+d2];
                GWv[d*D+d2] += Z1[t*D+d] * GV[t*D+d2];
            }
        for (int d = 0; d < D; d++) {
            double g = 0.0;
            for (int d2 = 0; d2 < D; d2++)
                g += GQ[t*D+d2]*Wq[d*D+d2] + GK[t*D+d2]*Wk[d*D+d2] + GV[t*D+d2]*Wv[d*D+d2];
            GZ1[t*D+d] = g;
        }
    }

    /* ---- Norm 1 backward ---- */
    #pragma omp parallel for reduction(+:Ggamma1[0:D])
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

    double t1 = omp_get_wtime();

    free(X); free(gamma1); free(gamma2); free(Wq); free(Wk); free(Wv);
    free(Wg); free(Wu); free(Wd); free(GY);
    free(Z1); free(r1); free(Q); free(K); free(V); free(A); free(R1);
    free(Z2); free(r2); free(U); free(Vg); free(H); free(Y);
    free(GX); free(GR1); free(GZ1); free(GZ2); free(GQ); free(GK); free(GV);
    free(Ggamma1); free(Ggamma2); free(GWq); free(GWk); free(GWv);
    free(GWg); free(GWu); free(GWd);

    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s B N D DFF THREADS [REPEATS]\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]), N = atoi(argv[2]), D = atoi(argv[3]);
    int DFF = atoi(argv[4]), THREADS = atoi(argv[5]);
    int repeats = (argc > 6) ? atoi(argv[6]) : 5;
    omp_set_num_threads(THREADS);

    (void)full_block_once(B, N, D, DFF);  /* untimed warmup */

    double total = 0.0;
    for (int r = 0; r < repeats; r++)
        total += full_block_once(B, N, D, DFF);
    double mean = total / repeats;

    printf("fullblock,C,%d,%d,%d,%d,%d,%.6f,%d\n", B, N, D, DFF, THREADS, mean, repeats);

    return 0;
}
