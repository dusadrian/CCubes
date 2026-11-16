/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_EXACT_PRICE_H
#define PROGRESSIVE_NATIVE_EXACT_PRICE_H

#include <stdbool.h>
#include <stdint.h>

#include "bounded_mmcs.h"
#include "progressive_native_master.h"
#include "utils.h"

typedef enum {
    PROGRESSIVE_NATIVE_EXACT_PRICE_OK = 0,
    PROGRESSIVE_NATIVE_EXACT_PRICE_INVALID,
    /* The exact pricer does not handle this chart; leave IPC unchanged. */
    PROGRESSIVE_NATIVE_EXACT_PRICE_UNSUPPORTED,
    PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY,
    PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR
} ProgressiveNativeExactPriceStatus;

/* Call schedule carried across IPC rounds. Zero-initialize before use. */
typedef struct {
    bool stalled;
    int stalled_calls;
    int node_index;
    int calls;
    int node_calls;
    int inserted;
    int node_improvements;
    int node_certificates;
    uint64_t nanoseconds;
} ProgressiveNativeExactPriceState;

typedef struct {
    bool node;
    int returned;
    int inserted;
    double max_found;
    double upper;
    bool complete;
    /* Node calls only. */
    int anchor;
    int released;
    int residual_rows;
    int node_columns;
    int node_cover;
    double node_bound;
    int certified_lower; /* Cubes any cover of the residual rows needs. */
    int improved_cardinality; /* Zero when the incumbent was not improved. */
    double seconds;
} ProgressiveNativeExactPriceStep;

/*
 * One exact pricing call on the full care table, inserting every returned
 * cube with new ON coverage into `master`.
 *
 * Global calls price on the master's current Lagrangian multipliers. Once a
 * global call inserts nothing, later calls price nodes, with a global
 * re-check every third call; a re-check that inserts clears the stall.
 *
 * A node fixes every incumbent cube except a released neighbourhood of
 * max(8, ceil(k/8)) cubes around a rotating anchor (most shared ON rows,
 * then fewest differing literals). The master is projected onto the rows
 * the fixed cubes leave uncovered, seeded with the nonempty, distinct
 * projections of the released incumbent cubes, and solved warm from
 * `multipliers` on those rows. Exact pricing then uses the node multipliers,
 * with zero weight on covered rows. When the fixed cubes plus the node cover
 * are smaller than the incumbent and cover every ON row, `*improved` receives that
 * cover in the transposed incumbent layout (caller frees it).
 *
 * `solution` is the incumbent in the transposed layout of
 * ProgressiveCoverResult (solution[input * cardinality + cube]).
 */
ProgressiveNativeExactPriceStatus progressive_native_exact_price_step(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    const double *multipliers,
    int cardinality,
    const int *solution,
    int requested_threads,
    uint64_t worker_memory_budget_bytes,
    double seconds,
    ProgressiveNativeExactPriceState *state,
    int **improved,
    ProgressiveNativeExactPriceStep *step
);

#endif
