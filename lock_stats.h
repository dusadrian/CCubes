/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef LOCK_STATS_H
#define LOCK_STATS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Opt-in contention profiling for the per-output PI-insertion locks.
 *
 * Enabled with CCUBES_LOCK_STATS=1. Counters are per (thread, output) in
 * separately allocated rows so the profiling itself neither contends nor
 * false-shares; totals are combined only in the final report.
 *
 * The report answers one question: how much of PI generation is trapped
 * behind the output locks, and therefore what the maximum achievable
 * speedup is no matter how many cores are added.
 */

/* True when CCUBES_LOCK_STATS is set; read directly in the hot path. */
extern bool lock_stats_active;

/* Reads the env switch and allocates the counters. Safe to call once. */
bool lock_stats_init(int threads, int noutputs);

uint64_t lock_stats_now_ns(void);

void lock_stats_record(
    int tid,
    int output,
    uint64_t wait_ns,
    uint64_t held_ns
);

/* Wall-clock seconds spent generating prime implicants, for the ratios. */
void lock_stats_set_generation_seconds(double seconds);

void lock_stats_report(FILE *out);

void lock_stats_free(void);

#endif
