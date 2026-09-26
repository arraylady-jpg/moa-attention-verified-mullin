/* forward_gpu_bench.c
 * Parameterized, timed OpenACC benchmark for Paper I's forward pass.
 *
 * Source: arXiv:2606.07713v1, eqs. 17, 20, 21. Same math as
 * forward_bench.c (the OpenMP CPU version) and moa_forward_dnf in
 * attention/moa_attention.c -- this file targets GPU offload via
 * OpenACC instead of CPU threads via OpenMP.
 *
 * REVISION HISTORY:
 * v1: original version took a single N per process invocation, with
 *     the sweep script launching one process per N. Real hardware data
 *     from Delta (A100, job 21180661) for decode_gpu_bench.c -- the
 *     same one-process-per-size structure this file originally had --
 *     showed wall time essentially flat across a 1024x range of N.
 *     Root cause: CUDA context creation and OpenACC JIT compilation
 *     cost ~300ms PER PROCESS LAUNCH, dominating and masking actual
 *     kernel time. Fixed by sweeping N internally within one process,
 *     paying context-init cost once via an untimed warmup.
 * v2 (this version): v1's real A100 data (job 21181233, Delta
 *     gpuA100x4) showed a local scaling exponent that DECREASED with N
 *     (1.63 -> 1.19 -> 0.89) instead of rising toward 2 as expected --
 *     the opposite of what under-saturated-GPU-converging-to-O(n^2)
 *     would predict. Most likely cause: a single timed measurement per
 *     N has no protection against system jitter at these
 *     sub-millisecond magnitudes, unlike moa_decode_bench.c's CPU
 *     benchmark, which already averages over repeated in-process calls
 *     after warmup. This version adds the same repeated-and-averaged
 *     structure. Also raised the scratch buffer from 512 to 8192
 *     elements so N can match the CPU sweep's full tested range
 *     (N=64..8192) for a direct apples-to-apples comparison -- this
 *     specific change (larger per-thread private array) has NOT been
 *     confirmed on real GPU hardware yet: the array is backed by
 *     global ("local") memory, not the much smaller 48KB shared-memory
 *     pool (confirmed via the A100's own device query in job
 *     21181233's log: 49152 bytes shared / block, 65536 registers /
 *     block), so it should remain correct at the larger size, but may
 *     be slower per-element at large N than a shared-memory or
 *     register-resident version would be -- that is itself data worth
 *     having, not a bug to avoid.
 *
 * Usage: ./forward_gpu_bench B N_START N_END D [REPEATS] [--check]
 * Sweeps N from N_START to N_END, doubling. For each N, runs an
 * untimed warmup call followed by REPEATS (default 5) timed calls,
 * reporting the mean. Prints one CSV line per N:
 * impl,lang,B,N,D,seconds,repeats
 * With --check: also runs the CPU serial reference at each N and
 * reports max diff to stderr (checked once per N, not once per repeat).
 *
 * NOTE ON THIS ENVIRONMENT: compiled and run via GCC's OpenACC host
 * fallback in the sandbox that built this, correctness-only --
 * host-fallback has no comparable per-process context-init cost or
 * measurement-jitter profile to real GPU hardware, so neither the v1
 * nor v2 timing fix can be confirmed here, only correctness and
 * internal consistency. Both fixes require a real GPU re-run to
 * confirm.
 *
 * Fixed-size scratch buffer (arow[8192]) caps usable N at 8192.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_N 8192

static double* alloc_rand(int n, unsigned seed) {
    double* a = malloc(sizeof(double) * n);
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        a[i] = ((double)(s % 20000) / 10000.0 - 1.0);
    }
    return a;
}

static double wall_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void softmax_row(double *row, int n) {
    double m = -1e300;
    for (int i = 0; i < n; i++) if (row[i] > m) m = row[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { row[i] = exp(row[i] - m); s += row[i]; }
    for (int i = 0; i < n; i++) row[i] /= s;
}

/* CPU serial reference, for --check verification (not timed). */
static void forward_dnf_serial(const double *Q, const double *K, const double *V,
                                double *A, double *Out, double scale, int B, int N, int D) {
    double *arow = malloc(sizeof(double) * N);
    for (int b = 0; b < B; b++) {
        for (int ir = 0; ir < N; ir++) {
            for (int ic = 0; ic < N; ic++) {
                double s = 0.0;
                for (int j = 0; j < D; j++)
                    s += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                arow[ic] = s * scale;
            }
            softmax_row(arow, N);
            for (int ic = 0; ic < N; ic++) A[(b*N+ir)*N+ic] = arow[ic];
            for (int id = 0; id < D; id++) {
                double o = 0.0;
                for (int ic = 0; ic < N; ic++) o += arow[ic] * V[(b*N+ic)*D+id];
                Out[(b*N+ir)*D+id] = o;
            }
        }
    }
    free(arow);
}

static double maxdiff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d > m) m = d; }
    return m;
}

/* One forward call: B,N,D fixed, times just this acc data region.
 * do_check runs and reports the CPU-reference comparison for this
 * call; kept separate from the timed region itself. */
static double forward_once(int B, int N, int D, int do_check) {
    const double scale = 1.0 / sqrt((double)D);
    int NT = B * N;

    double *Q = alloc_rand(NT*D, 1);
    double *K = alloc_rand(NT*D, 2);
    double *V = alloc_rand(NT*D, 3);
    double *A = malloc(sizeof(double) * (long)NT*N);
    double *Out = malloc(sizeof(double) * NT*D);

    double t0 = wall_time();

    #pragma acc data copyin(Q[0:NT*D], K[0:NT*D], V[0:NT*D]) \
                      copyout(A[0:(long)NT*N], Out[0:NT*D])
    {
        #pragma acc parallel loop collapse(2)
        for (int b = 0; b < B; b++) {
            for (int ir = 0; ir < N; ir++) {
                double arow[MAX_N];
                double m = -1e300;
                for (int ic = 0; ic < N; ic++) {
                    double s = 0.0;
                    for (int j = 0; j < D; j++)
                        s += Q[(b*N+ir)*D+j] * K[(b*N+ic)*D+j];
                    arow[ic] = s * scale;
                    if (arow[ic] > m) m = arow[ic];
                }
                double sum = 0.0;
                for (int ic = 0; ic < N; ic++) { arow[ic] = exp(arow[ic]-m); sum += arow[ic]; }
                for (int ic = 0; ic < N; ic++) {
                    arow[ic] /= sum;
                    A[(long)(b*N+ir)*N+ic] = arow[ic];
                }
                for (int id = 0; id < D; id++) {
                    double o = 0.0;
                    for (int ic = 0; ic < N; ic++)
                        o += arow[ic] * V[(b*N+ic)*D+id];
                    Out[(b*N+ir)*D+id] = o;
                }
            }
        }
    }

    double t1 = wall_time();

    if (do_check) {
        double *A_ref = malloc(sizeof(double) * (long)NT*N);
        double *Out_ref = malloc(sizeof(double) * NT*D);
        forward_dnf_serial(Q, K, V, A_ref, Out_ref, scale, B, N, D);
        fprintf(stderr, "check: N=%d A   max diff = %.3e\n", N, maxdiff(A, A_ref, NT*N));
        fprintf(stderr, "check: N=%d Out max diff = %.3e\n", N, maxdiff(Out, Out_ref, NT*D));
        free(A_ref); free(Out_ref);
    }

    free(Q); free(K); free(V); free(A); free(Out);
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s B N_START N_END D [REPEATS] [--check]\n", argv[0]);
        return 1;
    }
    int B = atoi(argv[1]);
    int n_start = atoi(argv[2]), n_end = atoi(argv[3]);
    int D = atoi(argv[4]);

    int repeats = 5;
    int do_check = 0;
    if (argc > 5) {
        if (strcmp(argv[5], "--check") == 0) {
            do_check = 1;
        } else {
            repeats = atoi(argv[5]);
            if (argc > 6 && strcmp(argv[6], "--check") == 0) do_check = 1;
        }
    }

    if (n_end > MAX_N) {
        fprintf(stderr, "N_END exceeds fixed scratch buffer size (%d); increase MAX_N in source.\n", MAX_N);
        return 1;
    }

    /* Untimed warmup: pays CUDA context creation / OpenACC JIT cost
     * once, before any measurement begins. */
    fprintf(stderr, "warming up (paying CUDA context init cost once, untimed)...\n");
    (void)forward_once(B, n_start < 64 ? n_start : 64, D, 0);

    for (int N = n_start; N <= n_end; N *= 2) {
        /* One more untimed warmup at this specific N, then REPEATS
         * timed runs, averaged -- matches moa_decode_bench.c's
         * warmup-then-average pattern for the CPU decode kernel. */
        (void)forward_once(B, N, D, 0);

        double total = 0.0;
        for (int r = 0; r < repeats; r++) {
            int check_this_rep = (do_check && r == 0); /* check once, not every repeat */
            total += forward_once(B, N, D, check_this_rep);
        }
        double mean = total / repeats;
        printf("forward,C-OpenACC,%d,%d,%d,%.6f,%d\n", B, N, D, mean, repeats);
        fflush(stdout);
    }

    return 0;
}
