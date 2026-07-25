/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "lock_stats.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

bool lock_stats_active = false;

typedef struct {
    uint64_t count;
    uint64_t wait_ns;
    uint64_t held_ns;
} LockStatCell;

static LockStatCell **stat_rows = NULL; /* [thread][output], one alloc per thread */
static int stat_threads = 0;
static int stat_outputs = 0;
static double stat_generation_seconds = 0.0;

uint64_t lock_stats_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

bool lock_stats_init(int threads, int noutputs) {
    const char *flag = getenv("CCUBES_LOCK_STATS");
    if (!flag || flag[0] == '\0' || flag[0] == '0') return true;
    if (threads <= 0 || noutputs <= 0) return true;

    /*
     * One allocation per thread keeps each thread's counters off every other
     * thread's cache lines; a single [threads * noutputs] block would make
     * the profiler itself a false-sharing bottleneck and inflate the very
     * numbers it is meant to measure.
     */
    stat_rows = (LockStatCell **)calloc((size_t)threads, sizeof(LockStatCell *));
    if (!stat_rows) return false;

    for (int t = 0; t < threads; ++t) {
        stat_rows[t] = (LockStatCell *)calloc((size_t)noutputs, sizeof(LockStatCell));
        if (!stat_rows[t]) {
            for (int u = 0; u < t; ++u) free(stat_rows[u]);
            free(stat_rows);
            stat_rows = NULL;
            return false;
        }
    }

    stat_threads = threads;
    stat_outputs = noutputs;
    lock_stats_active = true;
    return true;
}

void lock_stats_record(
    int tid,
    int output,
    uint64_t wait_ns,
    uint64_t held_ns
) {
    if (!stat_rows || tid < 0 || tid >= stat_threads) return;
    if (output < 0 || output >= stat_outputs) return;

    LockStatCell *cell = &stat_rows[tid][output];
    cell->count++;
    cell->wait_ns += wait_ns;
    cell->held_ns += held_ns;
}

void lock_stats_set_generation_seconds(double seconds) {
    stat_generation_seconds = seconds;
}

void lock_stats_report(FILE *out) {
    if (!lock_stats_active || !stat_rows || !out) return;

    uint64_t grand_count = 0;
    uint64_t grand_wait = 0;
    uint64_t grand_held = 0;

    /* Per-output totals: each output lock serialises its own held time. */
    uint64_t worst_output_held = 0;
    int worst_output = -1;

    fprintf(out, "\n=== CCUBES lock contention (per-output PI insertion) ===\n");
    fprintf(
        out,
        "%-8s %14s %14s %14s %12s\n",
        "output", "acquisitions", "wait_s", "held_s", "held_us/acq"
    );

    for (int o = 0; o < stat_outputs; ++o) {
        uint64_t count = 0, wait = 0, held = 0;
        for (int t = 0; t < stat_threads; ++t) {
            count += stat_rows[t][o].count;
            wait += stat_rows[t][o].wait_ns;
            held += stat_rows[t][o].held_ns;
        }

        grand_count += count;
        grand_wait += wait;
        grand_held += held;

        if (held > worst_output_held) {
            worst_output_held = held;
            worst_output = o;
        }

        fprintf(
            out,
            "%-8d %14llu %14.4f %14.4f %12.3f\n",
            o + 1,
            (unsigned long long)count,
            (double)wait / 1e9,
            (double)held / 1e9,
            count ? ((double)held / 1e3) / (double)count : 0.0
        );
    }

    fprintf(
        out,
        "%-8s %14llu %14.4f %14.4f %12.3f\n",
        "TOTAL",
        (unsigned long long)grand_count,
        (double)grand_wait / 1e9,
        (double)grand_held / 1e9,
        grand_count ? ((double)grand_held / 1e3) / (double)grand_count : 0.0
    );

    const double gen = stat_generation_seconds;
    const double serial_floor = (double)worst_output_held / 1e9;

    fprintf(out, "\nthreads=%d outputs=%d\n", stat_threads, stat_outputs);
    if (gen > 0.0) {
        fprintf(
            out,
            "pi_generation wall           : %.4f s\n"
            "aggregate time holding locks : %.4f s (%.1f%% of wall x threads)\n"
            "aggregate time waiting       : %.4f s (%.1f%% of wall x threads)\n",
            gen,
            (double)grand_held / 1e9,
            100.0 * ((double)grand_held / 1e9) / (gen * (double)stat_threads),
            (double)grand_wait / 1e9,
            100.0 * ((double)grand_wait / 1e9) / (gen * (double)stat_threads)
        );

        /*
         * The output locks run concurrently with one another, but each one
         * serialises every insertion for its own output. So no matter how
         * many cores are added, generation cannot finish faster than the
         * busiest single lock -- that is the Amdahl floor below.
         */
        if (serial_floor > 0.0) {
            fprintf(
                out,
                "\nbusiest lock                 : output %d, %.4f s serialised\n"
                "Amdahl floor (cores -> inf)  : %.4f s\n"
                "max speedup vs this run      : %.1fx\n",
                worst_output + 1,
                serial_floor,
                serial_floor,
                gen / serial_floor
            );
            fprintf(
                out,
                "\nProjected pi_generation wall (unlocked work scaled, lock floor fixed):\n"
            );
            const double unlocked =
                gen - (serial_floor > gen ? gen : serial_floor);
            for (int cores = stat_threads; cores <= 384; cores *= 2) {
                double projected =
                    unlocked * (double)stat_threads / (double)cores + serial_floor;
                fprintf(
                    out,
                    "  %4d cores : %8.3f s   (%5.1fx vs this run, %4.1f%% efficiency)\n",
                    cores,
                    projected,
                    gen / projected,
                    100.0 * (gen / projected) / ((double)cores / (double)stat_threads)
                );
            }
        }
    }
    fprintf(out, "=== end lock contention ===\n\n");
}

void lock_stats_free(void) {
    if (stat_rows) {
        for (int t = 0; t < stat_threads; ++t) free(stat_rows[t]);
        free(stat_rows);
        stat_rows = NULL;
    }
    stat_threads = 0;
    stat_outputs = 0;
    lock_stats_active = false;
}
