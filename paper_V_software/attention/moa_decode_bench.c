/*
 * moa_decode_bench.c
 * MoA Decode Attention: CPU OpenMP benchmark
 *
 * Benchmarks the MoA decode kernel across:
 *   - Thread counts: 1, 2, 4, 8, ..., MAX_THREADS (user-specified)
 *   - Array sizes:   n = N_START, 2*N_START, 4*N_START, ..., N_MAX
 *
 * Compile:
 *   gcc  -O3 -fopenmp -o moa_decode_bench moa_decode_bench.c -lm
 *   clang -O3 -fopenmp -o moa_decode_bench moa_decode_bench.c -lm
 *
 * Usage:
 *   ./moa_decode_bench <max_threads> <n_start> <n_max> <dk> <dv> [repeats]
 *
 * Example:
 *   ./moa_decode_bench 8 1024 65536 64 64 5
 *
 * Output: CSV to stdout and timings_cpu.csv
 *   threads, n, dk, dv, time_s, gbps, traffic_mb
 *
 * MoA ONF stride arithmetic (row-major, Definition 2.1 of arXiv:2606.07713v1):
 *   K[l][j] = K[l*dk + j]   gamma(<l,j>,<n,dk>) = l*dk + j
 *   V[l][d] = V[l*dv + d]   gamma(<l,d>,<n,dv>) = l*dv + d
 *   No K^T buffer; K read row-by-row in stored layout.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/* Timing: wall-clock seconds via omp_get_wtime()                     */
static double now(void) { return omp_get_wtime(); }

/* ------------------------------------------------------------------ */
/* Fill inputs with deterministic pattern (same as verification build) */
static void fill_inputs(int n, int dk, int dv,
                        float *q, float *K, float *V)
{
    for (int j = 0; j < dk; j++)
        q[j] = 0.1f * (float)(j + 1);

    for (int l = 0; l < n; l++)
        for (int j = 0; j < dk; j++)
            K[l * dk + j] = 0.01f * (float)((l * dk + j) % 100 + 1);

    for (int l = 0; l < n; l++)
        for (int d = 0; d < dv; d++)
            V[l * dv + d] = 0.01f * (float)((l * dv + d) % 100 + 1);
}

/* ------------------------------------------------------------------ */
/*
 * moa_decode_omp
 *
 * OpenMP parallel MoA decode with nthreads threads.
 * Three passes match the DNF exactly:
 *   Step I   (scores):  s[l] = scale * vq (+.x Omega<1,2>) K  -- K row l
 *   Step II  (softmax): a[l] = exp(s[l]-m) / Z
 *   Step III (wsum):    out[d] = va (+.x Omega<1,2>) V        -- V row l
 *
 * Parallelism via OpenMP:
 *   Pass 1: parallel over l (key rows) -- each thread owns a stripe
 *   Pass 2: parallel over l (exponentiate)
 *   Pass 3: parallel over d (output features) -- coalesced
 *
 * No K^T buffer. K accessed at gamma offset l*dk+j.
 */
void moa_decode_omp(const float * restrict q,
                    const float * restrict K,
                    const float * restrict V,
                    float       * restrict out,
                    int n, int dk, int dv,
                    int nthreads)
{
    const float scale = 1.0f / sqrtf((float)dk);
    float *s = (float *)malloc(n * sizeof(float));
    float *e = (float *)malloc(n * sizeof(float));

    /* Pass 1: scores, parallel over key rows l */
    float m = -1e38f;
#pragma omp parallel for num_threads(nthreads) reduction(max:m) schedule(static)
    for (int l = 0; l < n; l++) {
        float dot = 0.0f;
        /* gamma(<l,j>,<n,dk>) = l*dk + j  -- ONF stride, no K^T */
        for (int j = 0; j < dk; j++)
            dot += q[j] * K[l * dk + j];
        s[l] = scale * dot;
        if (s[l] > m) m = s[l];
    }

    /* Pass 2: exponentiate and accumulate Z */
    float Z = 0.0f;
#pragma omp parallel for num_threads(nthreads) reduction(+:Z) schedule(static)
    for (int l = 0; l < n; l++) {
        e[l] = expf(s[l] - m);
        Z += e[l];
    }
    float inv_Z = 1.0f / Z;

    /* Pass 3: weighted sum, parallel over output features d */
    /* gamma(<l,d>,<n,dv>) = l*dv + d  -- coalesced access */
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (int d = 0; d < dv; d++) {
        float acc = 0.0f;
        for (int l = 0; l < n; l++)
            acc += (e[l] * inv_Z) * V[l * dv + d];
        out[d] = acc;
    }

    free(s); free(e);
}

/* ------------------------------------------------------------------ */
/* Sequential reference (1 thread, for verification)                  */
void moa_decode_seq(const float * restrict q,
                    const float * restrict K,
                    const float * restrict V,
                    float       * restrict out,
                    int n, int dk, int dv)
{
    const float scale = 1.0f / sqrtf((float)dk);
    float *s = (float *)malloc(n * sizeof(float));
    float *e = (float *)malloc(n * sizeof(float));

    float m = -1e38f;
    for (int l = 0; l < n; l++) {
        float dot = 0.0f;
        for (int j = 0; j < dk; j++)
            dot += q[j] * K[l * dk + j];
        s[l] = scale * dot;
        if (s[l] > m) m = s[l];
    }
    float Z = 0.0f;
    for (int l = 0; l < n; l++) { e[l] = expf(s[l] - m); Z += e[l]; }
    float inv_Z = 1.0f / Z;
    for (int d = 0; d < dv; d++) {
        float acc = 0.0f;
        for (int l = 0; l < n; l++)
            acc += (e[l] * inv_Z) * V[l * dv + d];
        out[d] = acc;
    }
    free(s); free(e);
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    /* Parse arguments */
    int  max_threads = (argc > 1) ? atoi(argv[1]) : 8;
    int  n_start     = (argc > 2) ? atoi(argv[2]) : 1024;
    int  n_max       = (argc > 3) ? atoi(argv[3]) : 65536;
    int  dk          = (argc > 4) ? atoi(argv[4]) : 64;
    int  dv          = (argc > 5) ? atoi(argv[5]) : 64;
    int  repeats     = (argc > 6) ? atoi(argv[6]) : 5;

    printf("=== MoA Decode CPU Benchmark ===\n");
    printf("max_threads=%d  n_start=%d  n_max=%d  dk=%d  dv=%d  repeats=%d\n\n",
           max_threads, n_start, n_max, dk, dv, repeats);

    /* Open CSV output */
    FILE *csv = fopen("timings_cpu.csv", "w");
    fprintf(csv, "threads,n,dk,dv,time_s,gbps,traffic_mb,speedup_vs_seq\n");

    /* Header */
    printf("%-8s %-8s %-6s %-6s %-12s %-10s %-12s %-12s\n",
           "threads", "n", "dk", "dv", "time_s", "GB/s",
           "traffic_MB", "speedup");
    printf("%s\n", "------------------------------------------------------------------------");

    /* Baseline: sequential time for speedup computation */
    double seq_time_baseline = -1.0;

    /* Sweep thread counts: 1, 2, 4, 8, ... */
    for (int nt = 1; nt <= max_threads; nt *= 2) {
        /* Sweep array sizes: n_start, 2*n_start, 4*n_start, ... */
        for (int n = n_start; n <= n_max; n *= 2) {

            long traffic_b = ((long)dk + (long)n * dk + (long)n * dv + (long)dv) * 4;
            double traffic_mb = traffic_b / 1e6;

            /* Allocate */
            float *q   = (float *)malloc(dk * sizeof(float));
            float *K   = (float *)malloc((long)n * dk * sizeof(float));
            float *V   = (float *)malloc((long)n * dv * sizeof(float));
            float *out = (float *)malloc(dv * sizeof(float));
            float *ref = (float *)malloc(dv * sizeof(float));
            if (!q || !K || !V || !out || !ref) {
                fprintf(stderr, "malloc failed for n=%d\n", n); break;
            }

            fill_inputs(n, dk, dv, q, K, V);

            /* Verify against sequential */
            moa_decode_seq(q, K, V, ref, n, dk, dv);
            moa_decode_omp(q, K, V, out, n, dk, dv, nt);
            float max_err = 0.0f;
            for (int d = 0; d < dv; d++) {
                float err = fabsf(out[d] - ref[d]);
                if (err > max_err) max_err = err;
            }
            if (max_err > 1e-3f) {
                fprintf(stderr, "WARN: nt=%d n=%d err=%.2e\n", nt, n, max_err);
            }

            /* Warmup */
            for (int r = 0; r < 2; r++)
                moa_decode_omp(q, K, V, out, n, dk, dv, nt);

            /* Timed runs */
            double t0 = now();
            for (int r = 0; r < repeats; r++)
                moa_decode_omp(q, K, V, out, n, dk, dv, nt);
            double elapsed = (now() - t0) / repeats;

            /* Record baseline sequential time for speedup */
            if (nt == 1 && seq_time_baseline < 0.0)
                seq_time_baseline = elapsed;
            double seq_ref = (nt == 1) ? elapsed : seq_time_baseline;
            /* Per-n sequential baseline */
            double t_seq0 = now();
            for (int r = 0; r < repeats; r++)
                moa_decode_seq(q, K, V, out, n, dk, dv);
            double seq_time = (now() - t_seq0) / repeats;
            double speedup = seq_time / elapsed;

            double gbps = (traffic_mb / 1e3) / elapsed;

            printf("%-8d %-8d %-6d %-6d %-12.6f %-10.3f %-12.2f %-12.2f\n",
                   nt, n, dk, dv, elapsed, gbps, traffic_mb, speedup);
            fprintf(csv, "%d,%d,%d,%d,%.8f,%.4f,%.4f,%.4f\n",
                    nt, n, dk, dv, elapsed, gbps, traffic_mb, speedup);

            free(q); free(K); free(V); free(out); free(ref);
        }
        /* Blank line between thread groups for readability */
        printf("\n");
    }

    fclose(csv);
    printf("\nResults saved to timings_cpu.csv\n");
    return 0;
}
