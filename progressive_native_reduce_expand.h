/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_REDUCE_EXPAND_H
#define PROGRESSIVE_NATIVE_REDUCE_EXPAND_H

#include <stdbool.h>
#include <stdint.h>

#include "bounded_mmcs.h"
#include "progressive_native_master.h"

typedef struct {
    int cubes_examined;
    int cubes_with_private_rows;
    int reduced_cores;
    int variants_attempted;
    int legal_candidates;
    int changed_geometries;
    int inserted_columns;
    int duplicate_columns;
    uint64_t literals_added_by_reduce;
    uint64_t elapsed_nanoseconds;
    bool deadline_reached;
} ProgressiveNativeReduceExpandStats;

/* Stop periodic passes once unique insertions fall below 40% of attempts. */
bool progressive_native_reduce_expand_should_continue(
    int inserted_columns,
    int variants_attempted
);

/*
 * Generate alternate legal cube geometries from a validated incumbent.
 *
 * REDUCE forms the most specific cube that still covers every ON row private
 * to one incumbent cube. EXPAND then drops literals in several deterministic
 * orders while never touching an explicit OFF row. The generated columns are
 * inserted into `master`; the caller subsequently re-solves that master to
 * perform the IRREDUNDANT/re-selection step.
 *
 * `solution` is input-major (`ninputs * cardinality`). `deadline` is an
 * absolute CLOCK_MONOTONIC nanosecond timestamp, or zero for no deadline.
 * Construction uses the configured thread count and bounded batches. Retained
 * candidates are inserted in the original cube/variant order; on interruption
 * only the contiguous completed prefix is inserted.
 */
ProgressiveNativeMasterStatus progressive_native_reduce_expand_generate(
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    int cardinality,
    const int *solution,
    int pass,
    int variants,
    uint64_t deadline,
    ProgressiveNativeReduceExpandStats *stats
);

#endif
