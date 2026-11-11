/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-Adrian
*/

#ifndef PROGRESSIVE_GLOBAL_BOUND_H
#define PROGRESSIVE_GLOBAL_BOUND_H

#include <stdbool.h>
#include <stdint.h>

#include "utils.h"

typedef enum {
    PROGRESSIVE_GLOBAL_BOUND_OK = 0,
    PROGRESSIVE_GLOBAL_BOUND_INVALID,
    PROGRESSIVE_GLOBAL_BOUND_UNSUPPORTED,
    PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY,
    PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR
} ProgressiveGlobalBoundStatus;

typedef struct {
    double row_weight_sum;
    double max_cube_weight_found;
    double max_cube_weight_upper_bound;
    double fractional_lower_bound;
    double fixed_weight_fractional_ceiling;
    double elapsed_seconds;
    int cardinality_lower_bound;
    int fixed_weight_cardinality_ceiling;
    int positive_rows;
    int anchors_completed;
    int anchors_total;
    int pricing_threads;
    uint64_t search_nodes;
    bool pricing_complete;
    bool deadline_reached;
} ProgressiveGlobalBoundResult;

/*
 * Compute a global set-cover lower bound from any nonnegative ON-row
 * weights. If W is their sum and U bounds the weight covered by every valid
 * cube, then W / U is a lower bound on the full cover LP, independently of
 * which columns are present in the progressive restricted master.
 *
 * Zero-weight ON rows are removed before pricing. This is safe because every
 * positive-weight cube covers a retained row, while the full OFF set remains
 * in force. Retained rows are ordered by decreasing weight so a timed anchor
 * sweep removes the largest terms from its unsearched suffix first. A
 * positive time limit also interrupts recursive anchor search; the returned
 * U remains certified when that limit is reached.
 */
ProgressiveGlobalBoundStatus progressive_global_bound_compute(
    const PIstorage *pi,
    int ninputs,
    const double *row_weights,
    int requested_threads,
    uint64_t worker_memory_budget_bytes,
    double time_limit_seconds,
    int *best_pattern,
    int witness_capacity,
    int *witness_count,
    int *witness_patterns,
    ProgressiveGlobalBoundResult *result
);

const char *progressive_global_bound_status_name(
    ProgressiveGlobalBoundStatus status
);

#endif
