/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_REPAIR_H
#define PROGRESSIVE_NATIVE_REPAIR_H

#include <stdbool.h>
#include <stdint.h>

#include "progressive_native_master.h"

typedef struct {
    int neighborhood;
    int incumbent_cardinality;
    int destroyed_count;
    int uncovered_rows;
    int *incumbent_columns;
    unsigned char *destroyed;
    unsigned char *eligible_columns;
    double *pricing_weights;
    bool allow_strong_retry;
    /* Prepared frozen counts remain valid as the master appends columns.
     * Incumbent IDs and destroyed membership are immutable until destroy.
     * NULL retains support for caller-constructed plans.
     */
    int *frozen_coverage_count;
    int frozen_coverage_rows;
} ProgressiveNativeRepairPlan;

typedef struct {
    int frozen_columns;
    int residual_rows;
    int residual_candidate_columns;
    int lagrangian_cardinality;
    int lagrangian_lower_bound;
    int lagrangian_gap;
    int lagrangian_columns_added;
    int greedy_columns_added;
    int redundant_columns_removed;
    int returned_cardinality;
    int lagrangian_effort_level;
    LagrangianStopReason lagrangian_stop_reason;
    bool lagrangian_attempted;
    bool lagrangian_completed;
    bool lagrangian_proved_optimal;
    bool lagrangian_strong_retry;
    bool greedy_fallback_used;
    bool deadline_reached;
} ProgressiveNativeRepairStats;

/*
 * Destroy a deterministic low-private-contribution neighborhood and augment
 * the Lagrangian row prices for the rows it leaves uncovered.
 *
 * `prepared` may be NULL, which restricts incumbent mapping to exact geometry.
 * Supplying it lets an incumbent whose geometry was displaced by the master's
 * equal-coverage deduplication resolve through coverage instead of failing.
 */
bool progressive_native_repair_prepare(
    const ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    int incumbent_cardinality,
    const int *incumbent_solution,
    const double *multipliers,
    double uncovered_penalty,
    int requested_destroy_count,
    int neighborhood,
    ProgressiveNativeRepairPlan *plan
);

/*
 * Adaptive LNS sizing policy. A nonpositive current count selects the normal
 * starting size. Repeated non-improvements grow the neighborhood gradually,
 * bounded by both an incumbent share and a multiple of the starting size.
 */
int progressive_native_repair_next_destroy_count(
    int incumbent_cardinality,
    int current_destroy_count
);

/*
 * Greedily solve the destroyed residual over every retained column, then
 * remove redundant selections. Returned indices belong to `master`.
 */
bool progressive_native_repair_solve(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    int **selected_columns,
    int *cardinality,
    ProgressiveNativeRepairStats *stats,
    uint64_t deadline_nanoseconds
);

/*
 * Write exactly the residual SCP induced by `plan` as a binary LP model.
 * Frozen incumbent columns are omitted; every variable name retains its
 * master-column index so an oracle solution can be mapped back unambiguously.
 */
bool progressive_native_repair_write_subproblem(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    const char *path,
    int *residual_rows,
    int *residual_candidate_columns
);

void progressive_native_repair_destroy(
    ProgressiveNativeRepairPlan *plan
);

#endif
