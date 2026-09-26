/* decode_gpu_bench.c
 * Parameterized, timed OpenACC benchmark for Paper III's decode kernel
 * (single query against n cached keys/values -- the inference-time
 * shape, distinct from Papers I/II's full N x N attention).
 *
 * Source: same three-pass structure as moa_decode_omp in
 * attention/moa_decode_bench.c (scores+max, softmax, weighted sum).
 * This file targets GPU offload via OpenACC instead of CPU threads via
 * OpenMP.
 *
 * REVISION HISTORY: the original version of this file took a single N
 * per process invocation, with the sweep script launching one process
 * per N. Real hardware data from Delta (A100, job 21180661) showed wall
 * time essentially flat (~0.30-0.40s) across N=1024 to N=1,048,576 --
 * a 1024x range -- which is wrong for an O(N) kernel. Root cause: CUDA
 * context creation and OpenACC JIT compilation cost ~300ms PER PROCESS
 * LAUNCH, and since each sweep point was a fresh process, that fixed
 * cost completely dominated and masked the actual (much smaller,
 * microsecond-scale) kernel time. This version sweeps N internally
 * within a single process, paying context-init cost once via an
 * explicit untimed warmup region before any measurement begins, so
 * each reported time reflects the kernel itself.
 *
 * DEVIATION FROM SOURCE: moa_decode_bench.c uses `float`; this file
 * uses `double`, for consistency with every other GPU bench file in
 * this project and for exact --check tolerance.
 *
 * Usage: ./decode_gpu_bench N_START N_END dk dv [--check]
 * Sweeps N from N_START to N_END, doubling. Prints one CSV line per N:
 * impl,lang,n,dk,dv,seconds
 * With --check: also runs the CPU serial reference at each N and
 * reports max diff to stderr.
 *
 * NOTE ON THIS ENVIRONMENT: compiled/run via GCC's OpenACC host
 * fallback in the sandbox that built this, correctness-only. The
 * timing-methodology fix above was diagnosed from real Delta A100 data,
 * not from anything visible in host-fallback mode (host-fallback has no
 * comparable per-process context-init cost, so this bug was invisible
 * until real hardware was used -- worth remembering for any other GPU
 * bench file in this project that has the same one-process-per-size
 * structure).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

static void decode_serial(const double *q, const double *K, const double *V,
                           double *out, double scale, int n, int dk, int dv) {
    double *s = malloc(sizeof(double) * n);
    double *e = malloc(sizeof(double) * n);
    double m = -1e300;
    for (int l = 0; l < n; l++) {
        double dot = 0.0;
        for (int j = 0; j < dk; j++) dot += q[j] * K[l*dk+j];
        s[l] = scale * dot;
        if (s[l] > m) m = s[l];
    }
    double Z = 0.0;
    for (int l = 0; l < n; l++) { e[l] = exp(s[l]-m); Z += e[l]; }
    double inv_Z = 1.0 / Z;
    for (int d = 0; d < dv; d++) {
        double acc = 0.0;
        for (int l = 0; l < n; l++) acc += (e[l]*inv_Z) * V[l*dv+d];
        out[d] = acc;
    }
    free(s); free(e);
}

static double maxdiff(const double *a, const double *b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) { double d = fabs(a[i]-b[i]); if (d > m) m = d; }
    return m;
}

/* One decode call: n,dk,dv fixed, times just this acc data region. */
static double decode_once(int n, int dk, int dv, int do_check) {
    const double scale = 1.0 / sqrt((double)dk);
    double *q = alloc_rand(dk, 1);
    double *K = alloc_rand((long)n*dk, 2);
    double *V = alloc_rand((long)n*dv, 3);
    double *out = malloc(sizeof(double) * dv);
    double *s = malloc(sizeof(double) * n);
    double *e = malloc(sizeof(double) * n);

    double t0 = wall_time();

    #pragma acc data copyin(q[0:dk], K[0:(long)n*dk], V[0:(long)n*dv]) \
                      create(s[0:n], e[0:n]) copyout(out[0:dv])
    {
        double m = -1e300;
        #pragma acc parallel loop reduction(max:m)
        for (int l = 0; l < n; l++) {
            double dot = 0.0;
            for (int j = 0; j < dk; j++) dot += q[j] * K[(long)l*dk+j];
            s[l] = scale * dot;
            if (s[l] > m) m = s[l];
        }

        double Z = 0.0;
        #pragma acc parallel loop reduction(+:Z)
        for (int l = 0; l < n; l++) {
            e[l] = exp(s[l] - m);
            Z += e[l];
        }
        double inv_Z = 1.0 / Z;

        #pragma acc parallel loop
        for (int d = 0; d < dv; d++) {
            double acc = 0.0;
            for (int l = 0; l < n; l++)
                acc += (e[l] * inv_Z) * V[(long)l*dv+d];
            out[d] = acc;
        }
    }

    double t1 = wall_time();

    if (do_check) {
        double *out_ref = malloc(sizeof(double) * dv);
        decode_serial(q, K, V, out_ref, scale, n, dk, dv);
        fprintf(stderr, "check: n=%d out max diff = %.3e\n", n, maxdiff(out, out_ref, dv));
        free(out_ref);
    }

    free(q); free(K); free(V); free(out); free(s); free(e);
    return t1 - t0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s N_START N_END dk dv [--check]\n", argv[0]);
        return 1;
    }
    int n_start = atoi(argv[1]), n_end = atoi(argv[2]);
    int dk = atoi(argv[3]), dv = atoi(argv[4]);
    int do_check = (argc > 5 && strcmp(argv[5], "--check") == 0);

    /* Untimed warmup: pays CUDA context creation / OpenACC JIT cost
     * once, before any measurement begins, so it doesn't bleed into the
     * first (or every) reported size. */
    fprintf(stderr, "warming up (paying CUDA context init cost once, untimed)...\n");
    (void)decode_once(64, dk < 64 ? dk : 64, dv < 64 ? dv : 64, 0);

    for (int n = n_start; n <= n_end; n *= 2) {
        double elapsed = decode_once(n, dk, dv, do_check);
        printf("decode,C-OpenACC,%d,%d,%d,%.6f\n", n, dk, dv, elapsed);
        fflush(stdout);
    }

    return 0;
}
