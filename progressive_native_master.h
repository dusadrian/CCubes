/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_MASTER_H
#define PROGRESSIVE_NATIVE_MASTER_H

#include <stdbool.h>
#include <stdint.h>

#include "lagrangian.h"
#include "progressive_column_pool.h"
#include "progressive_seed.h"

typedef enum {
    PROGRESSIVE_NATIVE_MASTER_OK = 0,
    PROGRESSIVE_NATIVE_MASTER_INVALID,
    PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY,
    PROGRESSIVE_NATIVE_MASTER_INFEASIBLE
} ProgressiveNativeMasterStatus;

typedef struct {
    int rows;
    int words;
    int ninputs;
    int max_support;
    int count;
    int capacity;
    uint64_t *coverage;
    int *patterns;
    /* Owned lookup/cache state. Mutate retained columns through insert;
     * a deep copy of the arrays must start with a NULL index. */
    struct ProgressiveNativeMasterIndex *index;
} ProgressiveNativeMaster;

bool progressive_native_master_initialize(
    ProgressiveNativeMaster *master,
    int rows,
    int words,
    int ninputs
);

bool progressive_native_master_set_max_support(
    ProgressiveNativeMaster *master,
    int max_support
);

void progressive_native_master_destroy(
    ProgressiveNativeMaster *master
);

ProgressiveNativeMasterStatus progressive_native_master_insert(
    ProgressiveNativeMaster *master,
    const uint64_t *coverage,
    const int *pattern,
    bool *inserted
);

ProgressiveNativeMasterStatus progressive_native_master_insert_seed(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveSeedResult *seed,
    int *inserted
);

ProgressiveNativeMasterStatus progressive_native_master_insert_pool(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveColumnPool *pool,
    int *inserted
);

/*
 * Run the bundled CFT-style Lagrangian master on all retained columns.
 * `solution` receives retained-column indices and `row_multipliers` receives
 * the best row prices for implicit prime pricing. `initial_multipliers` is
 * an optional warm start: when non-NULL it seeds the subgradient trajectory
 * (typically with the previous solve's returned prices) instead of the
 * dual-ascent initialization; see `solve_scp_lagrangian_with_multipliers`.
 */
ProgressiveNativeMasterStatus progressive_native_master_solve(
    const ProgressiveNativeMaster *master,
    int effort_level,
    const int *initial_solution,
    int initial_cardinality,
    int *solution,
    int *cardinality,
    double *row_multipliers,
    const double *initial_multipliers
);

ProgressiveNativeMasterStatus
progressive_native_master_solve_with_polish_incumbent(
    const ProgressiveNativeMaster *master,
    int effort_level,
    const int *initial_solution,
    int initial_cardinality,
    const int *polish_incumbent,
    int polish_incumbent_cardinality,
    int *solution,
    int *cardinality,
    double *row_multipliers,
    const double *initial_multipliers
);

/* Evaluate the unit-cost Lagrangian bound on this master. */
double progressive_native_master_evaluate_bound(
    const ProgressiveNativeMaster *master,
    const double *multipliers
);

bool progressive_native_master_find_pattern(
    const ProgressiveNativeMaster *master,
    const int *pattern,
    int *column
);

/*
 * Resolve each cube of `solution` to a retained column.  Exact geometry is
 * tried first.  Because the master deduplicates by ON coverage and keeps the
 * lowest-support representative, a saved incumbent can name a geometry that has
 * since been displaced; supplying `prepared` enables a coverage fallback for
 * that case.  Passing NULL restricts the mapping to exact geometry.
 */
bool progressive_native_master_map_solution(
    const ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    int cardinality,
    const int *solution,
    int *columns
);

#endif
