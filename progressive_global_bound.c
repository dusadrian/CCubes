/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-Adrian
*/

#include "progressive_global_bound.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bounded_mmcs.h"

typedef struct {
    int row;
    double weight;
} PositiveWeightedRow;

static int compare_positive_weighted_rows(
    const void *left,
    const void *right
) {
    const PositiveWeightedRow *a =
        (const PositiveWeightedRow *)left;
    const PositiveWeightedRow *b =
        (const PositiveWeightedRow *)right;
    if (a->weight > b->weight) return -1;
    if (a->weight < b->weight) return 1;
    return (a->row > b->row) - (a->row < b->row);
}

static uint64_t global_bound_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint64_t global_bound_deadline(
    uint64_t started,
    double seconds
) {
    if (seconds <= 0.0) return 0;
    const double span = seconds * 1e9;
    return span >= (double)(UINT64_MAX - started)
        ? UINT64_MAX
        : started + (uint64_t)span;
}

static double global_bound_rounding_margin(double weight_sum, int rows) {
    const double scale = fmax(1.0, fabs(weight_sum));
    return scale * (
        1e-12 + 8.0 * (double)(rows > 0 ? rows : 1) * DBL_EPSILON
    );
}

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
) {
    if (result) {
        memset(result, 0, sizeof(*result));
        result->max_cube_weight_found = -INFINITY;
        result->max_cube_weight_upper_bound = INFINITY;
    }
    if (witness_count) *witness_count = 0;
    if (
        !pi ||
        !row_weights ||
        !result ||
        ninputs <= 0 ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        !pi->ON_set ||
        !pi->OFF_set ||
        requested_threads <= 0 ||
        witness_capacity < 0 ||
        (
            witness_capacity > 0 &&
            (!witness_count || !witness_patterns)
        ) ||
        !isfinite(time_limit_seconds) ||
        time_limit_seconds < 0.0
    ) {
        return PROGRESSIVE_GLOBAL_BOUND_INVALID;
    }

    const uint64_t started = global_bound_monotonic_nanoseconds();
    int positive_rows = 0;
    double weight_sum = 0.0;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        const double weight = row_weights[row];
        if (!isfinite(weight) || weight < 0.0) {
            return PROGRESSIVE_GLOBAL_BOUND_INVALID;
        }
        if (weight > 0.0) {
            positive_rows++;
            weight_sum += weight;
            if (!isfinite(weight_sum)) {
                return PROGRESSIVE_GLOBAL_BOUND_INVALID;
            }
        }
    }
    result->row_weight_sum = weight_sum;
    result->positive_rows = positive_rows;
    if (positive_rows == 0) {
        result->max_cube_weight_found = 0.0;
        result->max_cube_weight_upper_bound = 0.0;
        result->pricing_complete = true;
        result->elapsed_seconds =
            (double)(global_bound_monotonic_nanoseconds() - started) / 1e9;
        return PROGRESSIVE_GLOBAL_BOUND_OK;
    }

    if (
        (size_t)positive_rows >
            SIZE_MAX / (size_t)ninputs / sizeof(int)
    ) {
        return PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY;
    }
    int *compact_on = (int *)malloc(
        (size_t)positive_rows * (size_t)ninputs * sizeof(int)
    );
    double *compact_weights = (double *)malloc(
        (size_t)positive_rows * sizeof(double)
    );
    PositiveWeightedRow *ordered = (PositiveWeightedRow *)malloc(
        (size_t)positive_rows * sizeof(PositiveWeightedRow)
    );
    int *best_vertices = (int *)malloc((size_t)ninputs * sizeof(int));
    if (!compact_on || !compact_weights || !ordered || !best_vertices) {
        free(compact_on);
        free(compact_weights);
        free(ordered);
        free(best_vertices);
        return PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY;
    }

    int compact_row = 0;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        if (row_weights[row] <= 0.0) continue;
        ordered[compact_row] = (PositiveWeightedRow){
            .row = row,
            .weight = row_weights[row]
        };
        compact_row++;
    }
    qsort(
        ordered,
        (size_t)positive_rows,
        sizeof(PositiveWeightedRow),
        compare_positive_weighted_rows
    );
    for (compact_row = 0; compact_row < positive_rows; ++compact_row) {
        const int row = ordered[compact_row].row;
        memcpy(
            compact_on + (size_t)compact_row * (size_t)ninputs,
            pi->ON_set + (size_t)row * (size_t)ninputs,
            (size_t)ninputs * sizeof(int)
        );
        compact_weights[compact_row] = ordered[compact_row].weight;
    }
    PIstorage compact;
    memset(&compact, 0, sizeof(compact));
    compact.inputs = ninputs;
    compact.outputs = 1;
    compact.ON_minterms = positive_rows;
    compact.OFF_minterms = pi->OFF_minterms;
    compact.ON_set = compact_on;
    compact.OFF_set = pi->OFF_set;
    compact.cov_bits = 64;
    compact.pichart_words = (positive_rows + 63) / 64;

    WeightedMMCSPreparedRows prepared;
    memset(&prepared, 0, sizeof(prepared));
    if (!bounded_mmcs_prepare_binary_rows(&compact, ninputs, &prepared)) {
        free(compact_on);
        free(compact_weights);
        free(ordered);
        free(best_vertices);
        return PROGRESSIVE_GLOBAL_BOUND_UNSUPPORTED;
    }

    WeightedMMCSPriceBatch batch;
    memset(&batch, 0, sizeof(batch));
    if (witness_capacity > 0) {
        batch.capacity = witness_capacity;
        batch.weights = (double *)malloc(
            (size_t)witness_capacity * sizeof(double)
        );
        batch.anchors = (int *)malloc(
            (size_t)witness_capacity * sizeof(int)
        );
        batch.support_sizes = (int *)malloc(
            (size_t)witness_capacity * sizeof(int)
        );
        batch.vertices = (int *)malloc(
            (size_t)witness_capacity * (size_t)ninputs * sizeof(int)
        );
        batch.coverage = (uint64_t *)malloc(
            (size_t)witness_capacity *
            (size_t)compact.pichart_words *
            sizeof(uint64_t)
        );
        if (
            !batch.weights || !batch.anchors || !batch.support_sizes ||
            !batch.vertices || !batch.coverage
        ) {
            bounded_mmcs_prepared_rows_destroy(&prepared);
            free(batch.weights); free(batch.anchors);
            free(batch.support_sizes); free(batch.vertices);
            free(batch.coverage);
            free(compact_on); free(compact_weights); free(ordered);
            free(best_vertices);
            return PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY;
        }
    }

    const uint64_t deadline = global_bound_deadline(
        started,
        time_limit_seconds
    );
    WeightedMMCSPriceOptions options = {
        .requested_threads = requested_threads,
        .worker_memory_budget_bytes = worker_memory_budget_bytes,
        .collect_timing = true,
        .enable_root_gate = false,
        .use_materialized_backend = false,
        .cache_edge_words = true,
        .disable_cover_bound = false,
        .anchor_limit = 0,
        .deadline_nanoseconds = deadline
    };
    WeightedMMCSPriceResult price;
    WeightedMMCSPriceStatus price_status =
        bounded_mmcs_price_output_batch_prepared(
            &compact,
            &prepared,
            ninputs,
            compact_weights,
            best_vertices,
            witness_capacity > 0 ? &batch : NULL,
            &options,
            &price
        );

    ProgressiveGlobalBoundStatus status = PROGRESSIVE_GLOBAL_BOUND_OK;
    if (
        price_status != WEIGHTED_MMCS_PRICE_OK &&
        price_status != WEIGHTED_MMCS_PRICE_LIMIT_REACHED
    ) {
        status = price_status == WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY
            ? PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY
            : (
                price_status == WEIGHTED_MMCS_PRICE_UNSUPPORTED
                    ? PROGRESSIVE_GLOBAL_BOUND_UNSUPPORTED
                    : PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR
            );
        goto cleanup;
    }

    result->max_cube_weight_found = price.best_weight;
    result->anchors_completed = price.anchors_completed;
    result->anchors_total = price.anchors_total;
    result->pricing_threads = price.effective_threads;
    result->search_nodes = price.search_nodes;
    result->pricing_complete = price.anchor_sweep_complete;
    result->deadline_reached =
        price_status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED;
    if (best_pattern) {
        memset(best_pattern, 0, (size_t)ninputs * sizeof(int));
        if (
            price.best_anchor >= 0 &&
            price.best_anchor < positive_rows
        ) {
            const int original_anchor =
                ordered[price.best_anchor].row;
            const int *anchor = pi->ON_set +
                (size_t)original_anchor * (size_t)ninputs;
            for (int index = 0; index < price.best_support_size; ++index) {
                const int input = best_vertices[index];
                best_pattern[input] = anchor[input];
            }
        }
    }
    if (witness_count && witness_patterns) {
        *witness_count = batch.count;
        for (int witness = 0; witness < batch.count; ++witness) {
            int *pattern = witness_patterns +
                (size_t)witness * (size_t)ninputs;
            memset(pattern, 0, (size_t)ninputs * sizeof(int));
            const int compact_anchor = batch.anchors[witness];
            if (compact_anchor < 0 || compact_anchor >= positive_rows) {
                status = PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR;
                goto cleanup;
            }
            const int original_anchor = ordered[compact_anchor].row;
            const int *anchor = pi->ON_set +
                (size_t)original_anchor * (size_t)ninputs;
            const int *vertices = batch.vertices +
                (size_t)witness * (size_t)ninputs;
            for (
                int index = 0;
                index < batch.support_sizes[witness];
                ++index
            ) {
                const int input = vertices[index];
                pattern[input] = anchor[input];
            }
        }
    }
    if (isfinite(price.best_weight) && price.best_weight > 0.0) {
        result->fixed_weight_fractional_ceiling =
            weight_sum / price.best_weight;
        const double nearest = round(
            result->fixed_weight_fractional_ceiling
        );
        const double ceiling_value = fabs(
            result->fixed_weight_fractional_ceiling - nearest
        ) <= 1e-12 * fmax(
            1.0,
            result->fixed_weight_fractional_ceiling
        )
            ? nearest
            : result->fixed_weight_fractional_ceiling;
        const double ceiling = ceil(ceiling_value);
        result->fixed_weight_cardinality_ceiling =
            ceiling > (double)INT_MAX ? INT_MAX : (int)ceiling;
    }

    /*
     * Accumulated floating-point sums can round W upward or U downward.
     * Deflate the numerator and inflate the oracle bound before converting
     * the fractional value into an integer lower bound.
     */
    const double margin = global_bound_rounding_margin(
        weight_sum,
        positive_rows
    );
    const double safe_weight_sum = fmax(0.0, weight_sum - margin);
    const double safe_upper = nextafter(
        price.max_weight_upper_bound + margin,
        INFINITY
    );
    if (!isfinite(safe_upper) || safe_upper <= 0.0) {
        status = PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR;
        goto cleanup;
    }
    result->max_cube_weight_upper_bound = safe_upper;
    result->fractional_lower_bound = safe_weight_sum / safe_upper;
    const double below = nextafter(
        result->fractional_lower_bound,
        -INFINITY
    );
    const double integer_bound = ceil(below);
    result->cardinality_lower_bound = integer_bound > (double)INT_MAX
        ? INT_MAX
        : (int)integer_bound;

cleanup:
    result->elapsed_seconds =
        (double)(global_bound_monotonic_nanoseconds() - started) / 1e9;
    bounded_mmcs_prepared_rows_destroy(&prepared);
    free(compact_on);
    free(compact_weights);
    free(ordered);
    free(best_vertices);
    free(batch.weights);
    free(batch.anchors);
    free(batch.support_sizes);
    free(batch.vertices);
    free(batch.coverage);
    return status;
}

const char *progressive_global_bound_status_name(
    ProgressiveGlobalBoundStatus status
) {
    switch (status) {
        case PROGRESSIVE_GLOBAL_BOUND_OK: return "ok";
        case PROGRESSIVE_GLOBAL_BOUND_INVALID: return "invalid";
        case PROGRESSIVE_GLOBAL_BOUND_UNSUPPORTED: return "unsupported";
        case PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY: return "out-of-memory";
        case PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR: return "pricing-error";
    }
    return "unknown";
}
