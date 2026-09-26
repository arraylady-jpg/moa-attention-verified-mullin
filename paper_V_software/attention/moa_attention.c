/*
 * moa_attention.c
 *
 * C implementation of MoA DNF and ONF attention algorithms.
 *
 * Source:
 *   Forward:  arXiv:2606.07713v1  eqs. 17, 20, 21
 *   Backward: Mullin & Hains, "Attention at the Theoretical Minimum"
 *             Algorithm 1 (backward, A from DRAM)
 *             Algorithm 2 (fused forward+backward, A never written to DRAM)
 *   ONF:      Dimension-lifted (tiled) versions of the DNF algorithms.
 *
 * Six functions:
 *   DNF:
 *     moa_forward_dnf    -- forward pass (arXiv eqs. 17, 20, 21)
 *     moa_backward_dnf   -- backward pass, A from DRAM (Algorithm 1)
 *     moa_fused_dnf      -- fused fwd+bwd, A recomputed (Algorithm 2)
 *   ONF (tiled, tile size T):
 *     moa_forward_onf    -- tiled forward
 *     moa_backward_onf   -- tiled backward
 *     moa_fused_onf      -- tiled fused
 *
 * Array layout: A[b][i][j]  -->  A[b*N*X + i*X + j]  (row-major, C order)
 *   Q,K,V,dO : [B][N][D]
 *   A         : [B][N][N]
 *   Out,GV,GQ,GK : [B][N][D]
 *
 * Scratch (stack, never DRAM):
 *   DNF forward:  arow[N]
 *   DNF backward: ga_row[N], corr, gs
 *   DNF fused:    arow[N], ga_row[N], corr, gs
 *   ONF adds:     score_tile[T], ga_tile[T], gs_tile[T]
 *
 * Compile:
 *   gcc -O2 -lm -o moa_attention moa_attention.c
 * Run:
 *   ./moa_attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* =========================================================================
 * Parameters
 * ========================================================================= */
#define B      2        /* batch size                                        */
#define N      4        /* sequence length                                   */
#define D      8        /* head dimension                                    */
#define T      4        /* ONF tile size; must divide N exactly              */
#define NT    (N/T)     /* number of tiles                                   */

/* Convenience index macros (row-major) */
#define QKV(b,i,d)   ((b)*N*D + (i)*D + (d))   /* Q,K,V,dO,Out,GV,GQ,GK */
#define AA(b,i,j)    ((b)*N*N + (i)*N + (j))   /* A                      */

/* =========================================================================
 * Utility: fill array with reproducible pseudo-random values in (-1, 1)
 * Simple LCG so no srand() dependency on results.
 * ========================================================================= */
static void fill_random(double *arr, int n, unsigned long *state)
{
    for (int i = 0; i < n; i++) {
        *state = (*state) * 6364136223846793005ULL + 1442695040888963407ULL;
        arr[i] = (double)(*state & 0x7FFFFFFFULL) / (double)0x7FFFFFFF - 1.0;
    }
}

/* =========================================================================
 * Helper: stable softmax over a row of length N in-place
 * ========================================================================= */
static void softmax_row(double *row, int n)
{
    double m = -DBL_MAX;
    for (int i = 0; i < n; i++) if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - m); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

/* =========================================================================
 * DNF FORWARD PASS  (arXiv eqs. 17, 20, 21)
 *
 * For each batch b and query row ir:
 *   arow[ic] = scale * sum_j Q[b,ir,j] * K[b,ic,j]   (eq. 17)
 *   arow     = softmax(arow)                           (eq. 20)
 *   A[b,ir,ic] = arow[ic]
 *   Out[b,ir,d] = sum_ic arow[ic] * V[b,ic,d]         (eq. 21)
 *
 * Scratch: arow[N]  -- one softmax row, O(N)
 * ========================================================================= */
void moa_forward_dnf(
    const double *Q,  const double *K,  const double *V,
    double *A,        double *Out,
    double scale)
{
    double arow[N];

    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {

            /* Step A: pre-softmax scores (arXiv eq. 17) */
            for (int ic = 0; ic < N; ic++) {
                arow[ic] = 0.0;
                for (int j = 0; j < D; j++)
                    arow[ic] += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                arow[ic] *= scale;
            }

            /* Step B: stable softmax (arXiv eq. 20) */
            softmax_row(arow, N);
            for (int ic = 0; ic < N; ic++)
                A[AA(b,ir,ic)] = arow[ic];

            /* Step C: output (arXiv eq. 21) */
            for (int id = 0; id < D; id++) {
                Out[QKV(b,ir,id)] = 0.0;
                for (int ic = 0; ic < N; ic++)
                    Out[QKV(b,ir,id)] += arow[ic] * V[QKV(b,ic,id)];
            }
        }
    }
}

/* =========================================================================
 * DNF BACKWARD PASS  (Algorithm 1 -- A available from DRAM)
 *
 * For each batch b:
 *   GV[b,ic,id] = sum_ir A[b,ir,ic] * dO[b,ir,id]        (A^T @ dO)
 *
 *   For each row ir:
 *     ga_row[ic] = sum_d dO[b,ir,d] * V[b,ic,d]           one row of dA
 *     corr       = sum_ic A[b,ir,ic] * ga_row[ic]          rowsum(A*dA)
 *     gs         = scale*A[b,ir,ic]*(ga_row[ic]-corr)      GScale inline
 *     GQ[b,ir,j] += gs * K[b,ic,j]
 *     GK[b,ic,j] += gs * Q[b,ir,j]
 *
 * Scratch: ga_row[N]  O(N);  corr, gs  scalars
 * No GA[N][N], GS[N][N], GScale[N][N] allocated.
 * ========================================================================= */
void moa_backward_dnf(
    const double *Q,  const double *K,  const double *V,
    const double *A,  const double *dO,
    double *GV,       double *GQ,       double *GK,
    double scale)
{
    double ga_row[N];
    double corr, gs;

    for (int b = 0; b < B; b++) {

        /* GV[b,ic,id] = sum_ir A[b,ir,ic] * dO[b,ir,id] */
        for (int ic = 0; ic < N; ic++)
            for (int id = 0; id < D; id++) {
                GV[QKV(b,ic,id)] = 0.0;
                for (int ir = 0; ir < N; ir++)
                    GV[QKV(b,ic,id)] += A[AA(b,ir,ic)] * dO[QKV(b,ir,id)];
            }

        /* GQ and GK: one row of GA at a time */
        for (int ir = 0; ir < N; ir++)
            for (int j = 0; j < D; j++) {
                GQ[QKV(b,ir,j)] = 0.0;
                GK[QKV(b,ir,j)] = 0.0;
            }

        for (int ir = 0; ir < N; ir++) {

            /* Fill ga_row[ic] = dA[b,ir,ic] = sum_d dO[b,ir,d]*V[b,ic,d] */
            for (int ic = 0; ic < N; ic++) {
                ga_row[ic] = 0.0;
                for (int id = 0; id < D; id++)
                    ga_row[ic] += dO[QKV(b,ir,id)] * V[QKV(b,ic,id)];
            }

            /* corr = rowsum of A[b,ir,:] * ga_row[:] */
            corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += A[AA(b,ir,ic)] * ga_row[ic];

            /* gs = GScale inline scalar; accumulate GQ and GK */
            for (int ic = 0; ic < N; ic++) {
                gs = scale * A[AA(b,ir,ic)] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
                    GK[QKV(b,ic,j)] += gs * Q[QKV(b,ir,j)];
                }
            }
        }
    }
}

/* =========================================================================
 * DNF FUSED FORWARD + BACKWARD  (Algorithm 2 -- A never written to DRAM)
 *
 * Single pass over rows ir per batch b.
 * Step A: arow[ic] = A[b,ir,ic] recomputed from Q,K (arXiv eqs. 17,20)
 * Step B: merged (ic,id) loop reads V and dO once each:
 *           ga_row[ic] += dO[b,ir,id] * V[b,ic,id]
 *           Out[b,ir,id] += arow[ic] * V[b,ic,id]
 *           GV[b,ic,id]  += arow[ic] * dO[b,ir,id]
 * Step C: corr scalar, gs inline, accumulate GQ and GK
 *
 * Scratch: arow[N], ga_row[N]  O(N) each;  corr, gs  scalars
 * No A[B][N][N], GS, GScale allocated.
 * V and dO each read once per (b,ir,ic,id).
 * ========================================================================= */
void moa_fused_dnf(
    const double *Q,  const double *K,  const double *V,  const double *dO,
    double *Out,      double *GV,       double *GQ,       double *GK,
    double scale)
{
    double arow[N], ga_row[N];
    double corr, gs, vv, oo;

    for (int b = 0; b < B; b++) {

        /* zero outputs for this batch */
        for (int ir = 0; ir < N; ir++)
            for (int id = 0; id < D; id++) {
                Out[QKV(b,ir,id)] = 0.0;
                GV[QKV(b,ir,id)]  = 0.0;
                GQ[QKV(b,ir,id)]  = 0.0;
                GK[QKV(b,ir,id)]  = 0.0;
            }

        for (int ir = 0; ir < N; ir++) {

            /* Step A: recompute softmax row ir of A from Q,K only */
            for (int ic = 0; ic < N; ic++) {
                arow[ic] = 0.0;
                for (int j = 0; j < D; j++)
                    arow[ic] += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                arow[ic] *= scale;
            }
            softmax_row(arow, N);

            /* Step B: merged (ic,id) loop -- V and dO each read once */
            for (int ic = 0; ic < N; ic++) {
                ga_row[ic] = 0.0;
                for (int id = 0; id < D; id++) {
                    vv = V[QKV(b,ic,id)];
                    oo = dO[QKV(b,ir,id)];
                    ga_row[ic]       += oo * vv;
                    Out[QKV(b,ir,id)] += arow[ic] * vv;
                    GV[QKV(b,ic,id)]  += arow[ic] * oo;
                }
            }

            /* Step C: corr scalar, then gs inline for GQ and GK */
            corr = 0.0;
            for (int ic = 0; ic < N; ic++)
                corr += arow[ic] * ga_row[ic];

            for (int ic = 0; ic < N; ic++) {
                gs = scale * arow[ic] * (ga_row[ic] - corr);
                for (int j = 0; j < D; j++) {
                    GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
                    GK[QKV(b,ic,j)] += gs * Q[QKV(b,ir,j)];
                }
            }
        }
    }
}

/* =========================================================================
 * ONF FORWARD PASS  (tiled DNF forward, tile size T)
 *
 * Dimension-lifting on ic: ic = it*T + tt,  it=0..NT-1,  tt=0..T-1
 *
 * Tile pass 1: compute score_tile[T] for each tile, track global max
 * Tile pass 2: exp(score - max), accumulate sum, normalise arow[N]
 * Tile pass 3: tiled output accumulation
 *
 * Scratch: arow[N], score_tile[T]  -- fit in registers
 * ========================================================================= */
void moa_forward_onf(
    const double *Q,  const double *K,  const double *V,
    double *A,        double *Out,
    double scale)
{
    double arow[N];
    double score_tile[T];
    double m, s;

    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {

            /* Tile pass 1: compute scores, find global max */
            m = -DBL_MAX;
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    score_tile[tt] = 0.0;
                    for (int j = 0; j < D; j++)
                        score_tile[tt] += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                    score_tile[tt] *= scale;
                    arow[ic] = score_tile[tt];
                }
                for (int tt = 0; tt < T; tt++)
                    if (score_tile[tt] > m) m = score_tile[tt];
            }

            /* Tile pass 2: exp, sum, normalise */
            s = 0.0;
            for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic] - m); s += arow[ic]; }
            for (int ic = 0; ic < N; ic++) { arow[ic] /= s; A[AA(b,ir,ic)] = arow[ic]; }

            /* Tile pass 3: tiled output accumulation */
            for (int id = 0; id < D; id++) Out[QKV(b,ir,id)] = 0.0;
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    for (int id = 0; id < D; id++)
                        Out[QKV(b,ir,id)] += arow[ic] * V[QKV(b,ic,id)];
                }
            }
        }
    }
}

/* =========================================================================
 * ONF BACKWARD PASS  (tiled DNF backward, Algorithm 1)
 *
 * GV tile:    for each tile of ic, accumulate GV over all ir
 * ga_row tile: fill ga_tile[T] from dO and V, store into ga_row
 * corr tile:  accumulate corr from arow and ga_row tile-by-tile
 * gs tile:    compute gs_tile[T], accumulate GQ and GK
 *
 * Scratch: ga_row[N], ga_tile[T], gs_tile[T], corr, gs
 * ========================================================================= */
void moa_backward_onf(
    const double *Q,  const double *K,  const double *V,
    const double *A,  const double *dO,
    double *GV,       double *GQ,       double *GK,
    double scale)
{
    double ga_row[N];
    double ga_tile[T], gs_tile[T];
    double corr, gs;

    for (int b = 0; b < B; b++) {

        /* Tiled GV */
        for (int ic = 0; ic < N; ic++)
            for (int id = 0; id < D; id++) GV[QKV(b,ic,id)] = 0.0;

        for (int it = 0; it < NT; it++) {
            int ic0 = it * T;
            for (int ir = 0; ir < N; ir++)
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    for (int id = 0; id < D; id++)
                        GV[QKV(b,ic,id)] += A[AA(b,ir,ic)] * dO[QKV(b,ir,id)];
                }
        }

        /* Zero GQ, GK */
        for (int ir = 0; ir < N; ir++)
            for (int j = 0; j < D; j++) {
                GQ[QKV(b,ir,j)] = 0.0;
                GK[QKV(b,ir,j)] = 0.0;
            }

        for (int ir = 0; ir < N; ir++) {

            /* Tile pass 1: fill ga_row tile-by-tile */
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++) ga_tile[tt] = 0.0;
                for (int id = 0; id < D; id++)
                    for (int tt = 0; tt < T; tt++)
                        ga_tile[tt] += dO[QKV(b,ir,id)] * V[QKV(b,ic0+tt,id)];
                for (int tt = 0; tt < T; tt++)
                    ga_row[ic0 + tt] = ga_tile[tt];
            }

            /* Tile pass 2: corr accumulated tile-by-tile */
            corr = 0.0;
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++)
                    corr += A[AA(b,ir,ic0+tt)] * ga_row[ic0+tt];
            }

            /* Tile pass 3: gs_tile, accumulate GQ and GK */
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++)
                    gs_tile[tt] = scale * A[AA(b,ir,ic0+tt)]
                                * (ga_row[ic0+tt] - corr);
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    gs = gs_tile[tt];
                    for (int j = 0; j < D; j++) {
                        GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
                        GK[QKV(b,ic,j)] += gs * Q[QKV(b,ir,j)];
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * ONF FUSED FORWARD + BACKWARD  (tiled Algorithm 2)
 *
 * Step A ONF: tiled two-pass softmax into arow[N]
 * Step B ONF: tiled merged (ic,id) loop, ga_tile[T] filled per tile,
 *             V and dO each read once per (b,ir,ic,id)
 * Step C ONF: tiled corr accumulation, tiled gs_tile, GQ and GK scatter
 *
 * Scratch: arow[N], ga_row[N], score_tile[T], ga_tile[T], gs_tile[T]
 * No A[B][N][N] allocated.
 * ========================================================================= */
void moa_fused_onf(
    const double *Q,  const double *K,  const double *V,  const double *dO,
    double *Out,      double *GV,       double *GQ,       double *GK,
    double scale)
{
    double arow[N], ga_row[N];
    double score_tile[T], ga_tile[T], gs_tile[T];
    double corr, gs, vv, oo, m, s;

    for (int b = 0; b < B; b++) {

        for (int ir = 0; ir < N; ir++)
            for (int id = 0; id < D; id++) {
                Out[QKV(b,ir,id)] = 0.0;
                GV[QKV(b,ir,id)]  = 0.0;
                GQ[QKV(b,ir,id)]  = 0.0;
                GK[QKV(b,ir,id)]  = 0.0;
            }

        for (int ir = 0; ir < N; ir++) {

            /* Step A ONF: tiled two-pass softmax */
            m = -DBL_MAX;
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    score_tile[tt] = 0.0;
                    for (int j = 0; j < D; j++)
                        score_tile[tt] += Q[QKV(b,ir,j)] * K[QKV(b,ic,j)];
                    score_tile[tt] *= scale;
                    arow[ic] = score_tile[tt];
                }
                for (int tt = 0; tt < T; tt++)
                    if (score_tile[tt] > m) m = score_tile[tt];
            }
            s = 0.0;
            for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic]-m); s += arow[ic]; }
            for (int ic = 0; ic < N; ic++)   arow[ic] /= s;

            /* Step B ONF: tiled merged V/dO loop */
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++) ga_tile[tt] = 0.0;
                for (int id = 0; id < D; id++) {
                    oo = dO[QKV(b,ir,id)];          /* dO read once per id */
                    for (int tt = 0; tt < T; tt++) {
                        int ic = ic0 + tt;
                        vv = V[QKV(b,ic,id)];        /* V read once per (ic,id) */
                        ga_tile[tt]       += oo * vv;
                        Out[QKV(b,ir,id)] += arow[ic] * vv;
                        GV[QKV(b,ic,id)]  += arow[ic] * oo;
                    }
                }
                for (int tt = 0; tt < T; tt++)
                    ga_row[ic0+tt] = ga_tile[tt];
            }

            /* Step C ONF: tiled corr, then tiled gs scatter */
            corr = 0.0;
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++)
                    corr += arow[ic0+tt] * ga_row[ic0+tt];
            }
            for (int it = 0; it < NT; it++) {
                int ic0 = it * T;
                for (int tt = 0; tt < T; tt++)
                    gs_tile[tt] = scale * arow[ic0+tt] * (ga_row[ic0+tt] - corr);
                for (int tt = 0; tt < T; tt++) {
                    int ic = ic0 + tt;
                    gs = gs_tile[tt];
                    for (int j = 0; j < D; j++) {
                        GQ[QKV(b,ir,j)] += gs * K[QKV(b,ic,j)];
                        GK[QKV(b,ic,j)] += gs * Q[QKV(b,ir,j)];
                    }
                }
            }
        }
    }
}

/* =========================================================================
 * VERIFICATION MAIN
 * ========================================================================= */
static double max_abs_diff(const double *a, const double *b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void print_vec(const char *label, const double *v, int n)
{
    printf("  %s:", label);
    for (int i = 0; i < n; i++) printf(" %8.5f", v[i]);
    printf("\n");
}

int main(void)
{
    const double scale = 1.0 / sqrt((double)D);
    const int    nqkv  = B * N * D;
    const int    naa   = B * N * N;

    /* Allocate input arrays */
    double *Q  = malloc(nqkv * sizeof(double));
    double *K  = malloc(nqkv * sizeof(double));
    double *V  = malloc(nqkv * sizeof(double));
    double *dO = malloc(nqkv * sizeof(double));

    /* Fill with reproducible pseudo-random values in (-1, 1) */
    unsigned long state = 42UL;
    fill_random(Q,  nqkv, &state);
    fill_random(K,  nqkv, &state);
    fill_random(V,  nqkv, &state);
    fill_random(dO, nqkv, &state);

    /* DNF outputs */
    double *A_dnf   = calloc(naa,  sizeof(double));
    double *Out_dnf = calloc(nqkv, sizeof(double));
    double *GV_dnf  = calloc(nqkv, sizeof(double));
    double *GQ_dnf  = calloc(nqkv, sizeof(double));
    double *GK_dnf  = calloc(nqkv, sizeof(double));

    /* Fused DNF outputs */
    double *Out_fdn = calloc(nqkv, sizeof(double));
    double *GV_fdn  = calloc(nqkv, sizeof(double));
    double *GQ_fdn  = calloc(nqkv, sizeof(double));
    double *GK_fdn  = calloc(nqkv, sizeof(double));

    /* ONF outputs */
    double *A_onf   = calloc(naa,  sizeof(double));
    double *Out_onf = calloc(nqkv, sizeof(double));
    double *GV_onf  = calloc(nqkv, sizeof(double));
    double *GQ_onf  = calloc(nqkv, sizeof(double));
    double *GK_onf  = calloc(nqkv, sizeof(double));

    /* Fused ONF outputs */
    double *Out_fon = calloc(nqkv, sizeof(double));
    double *GV_fon  = calloc(nqkv, sizeof(double));
    double *GQ_fon  = calloc(nqkv, sizeof(double));
    double *GK_fon  = calloc(nqkv, sizeof(double));

    /* ---- Run all six functions ---- */
    moa_forward_dnf  (Q, K, V, A_dnf, Out_dnf, scale);
    moa_backward_dnf (Q, K, V, A_dnf, dO, GV_dnf, GQ_dnf, GK_dnf, scale);
    moa_fused_dnf    (Q, K, V, dO, Out_fdn, GV_fdn, GQ_fdn, GK_fdn, scale);

    moa_forward_onf  (Q, K, V, A_onf, Out_onf, scale);
    moa_backward_onf (Q, K, V, A_dnf, dO, GV_onf, GQ_onf, GK_onf, scale);
    moa_fused_onf    (Q, K, V, dO, Out_fon, GV_fon, GQ_fon, GK_fon, scale);

    /* ---- Report ---- */
    printf("============================================================\n");
    printf(" MoA Attention -- C Implementation Verification\n");
    printf(" arXiv:2606.07713v1  eqs. 17,20,21 (forward)\n");
    printf(" Mullin & Hains backward Algorithms 1 & 2\n");
    printf(" B=%d  N=%d  D=%d  T=%d  scale=%.5f\n", B, N, D, T, scale);
    printf("============================================================\n");

    printf("\n DNF fused vs DNF separate -- max |error|:\n");
    printf("   Out : %.4e\n", max_abs_diff(Out_fdn, Out_dnf, nqkv));
    printf("   GV  : %.4e\n", max_abs_diff(GV_fdn,  GV_dnf,  nqkv));
    printf("   GQ  : %.4e\n", max_abs_diff(GQ_fdn,  GQ_dnf,  nqkv));
    printf("   GK  : %.4e\n", max_abs_diff(GK_fdn,  GK_dnf,  nqkv));

    printf("\n ONF vs DNF -- max |error|:\n");
    printf("   A   : %.4e\n", max_abs_diff(A_onf,   A_dnf,   naa ));
    printf("   Out : %.4e\n", max_abs_diff(Out_onf, Out_dnf, nqkv));
    printf("   GV  : %.4e\n", max_abs_diff(GV_onf,  GV_dnf,  nqkv));
    printf("   GQ  : %.4e\n", max_abs_diff(GQ_onf,  GQ_dnf,  nqkv));
    printf("   GK  : %.4e\n", max_abs_diff(GK_onf,  GK_dnf,  nqkv));

    printf("\n ONF fused vs DNF separate -- max |error|:\n");
    printf("   Out : %.4e\n", max_abs_diff(Out_fon, Out_dnf, nqkv));
    printf("   GV  : %.4e\n", max_abs_diff(GV_fon,  GV_dnf,  nqkv));
    printf("   GQ  : %.4e\n", max_abs_diff(GQ_fon,  GQ_dnf,  nqkv));
    printf("   GK  : %.4e\n", max_abs_diff(GK_fon,  GK_dnf,  nqkv));

    double all_errs = max_abs_diff(Out_fdn,Out_dnf,nqkv)
                    + max_abs_diff(GV_fdn, GV_dnf, nqkv)
                    + max_abs_diff(GQ_fdn, GQ_dnf, nqkv)
                    + max_abs_diff(GK_fdn, GK_dnf, nqkv)
                    + max_abs_diff(A_onf,  A_dnf,  naa)
                    + max_abs_diff(Out_onf,Out_dnf,nqkv)
                    + max_abs_diff(GV_onf, GV_dnf, nqkv)
                    + max_abs_diff(GQ_onf, GQ_dnf, nqkv)
                    + max_abs_diff(GK_onf, GK_dnf, nqkv)
                    + max_abs_diff(Out_fon,Out_dnf,nqkv)
                    + max_abs_diff(GV_fon, GV_dnf, nqkv)
                    + max_abs_diff(GQ_fon, GQ_dnf, nqkv)
                    + max_abs_diff(GK_fon, GK_dnf, nqkv);

    printf("\n %s: all six functions agree to machine precision.\n",
           all_errs < 1e-12 ? "PASS" : "FAIL");

    /* Softmax row-sum check */
    printf("\n Softmax row sums of A_dnf (batch 0) -- should all be 1:\n");
    for (int ir = 0; ir < N; ir++) {
        double s = 0.0;
        for (int ic = 0; ic < N; ic++) s += A_dnf[AA(0,ir,ic)];
        printf("   row %d: %.12f\n", ir, s);
    }

    /* Sample output values */
    printf("\n A_dnf[0,0,:]:\n");
    print_vec("", A_dnf, N);

    printf("\n Out_dnf[0,0,:]:\n");
    print_vec("", Out_dnf, D);

    printf("\n GV_dnf[0,0,:]:\n");
    print_vec("", GV_dnf, D);

    printf("\n GQ_dnf[0,0,:]:\n");
    print_vec("", GQ_dnf, D);

    printf("\n GK_dnf[0,0,:]:\n");
    print_vec("", GK_dnf, D);

    printf("\n============================================================\n");
    printf(" Scratch inventory:\n");
    printf("   DNF forward:  arow[%d]                O(N)\n",  N);
    printf("   DNF backward: ga_row[%d], corr, gs    O(N)+2 scalars\n", N);
    printf("   DNF fused:    arow[%d], ga_row[%d]   O(N) each\n", N, N);
    printf("   ONF adds:     score_tile[%d], ga_tile[%d], gs_tile[%d]\n",T,T,T);
    printf("   Eliminated:   A[%d][%d][%d] (fused), GA[N][N], GS[N][N], GScale[N][N]\n",
           B,N,N);
    printf("============================================================\n");

    /* Free */
    free(Q); free(K); free(V); free(dO);
    free(A_dnf); free(Out_dnf); free(GV_dnf); free(GQ_dnf); free(GK_dnf);
    free(Out_fdn); free(GV_fdn); free(GQ_fdn); free(GK_fdn);
    free(A_onf); free(Out_onf); free(GV_onf); free(GQ_onf); free(GK_onf);
    free(Out_fon); free(GV_fon); free(GQ_fon); free(GK_fon);

    return 0;
}
