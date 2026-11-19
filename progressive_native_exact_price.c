/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#define _POSIX_C_SOURCE 200809L
#include "progressive_native_exact_price.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "progressive_global_bound.h"
#include "progressive_validation.h"

#define EXACT_PRICE_BATCH 256
#define EXACT_PRICE_MIN_RELEASE 8
#define EXACT_PRICE_RECHECK_PERIOD 3
#define EXACT_PRICE_ANCHOR_STRIDE 17

typedef ProgressiveNativeExactPriceStatus Status;

static uint64_t now_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec;
}

static bool has_row(const uint64_t *bits, int row) {
    return (bits[row / 64] >> (row % 64)) & UINT64_C(1);
}

/* Price `weights` exactly and insert each returned cube with new coverage. */
static Status price_and_insert(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    const double *weights,
    int threads,
    uint64_t memory,
    double seconds,
    ProgressiveNativeExactPriceStep *step,
    ProgressiveGlobalBoundResult *bound
) {
    const int ninputs = master->ninputs;
    int *patterns = (int *)malloc(
        (size_t)EXACT_PRICE_BATCH * (size_t)ninputs * sizeof(int)
    );
    uint64_t *coverage = (uint64_t *)malloc(
        (size_t)master->words * sizeof(uint64_t)
    );
    Status status = PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY;
    int returned = 0;
    if (!patterns || !coverage) goto done;
    memset(bound, 0, sizeof(*bound));
    const ProgressiveGlobalBoundStatus price = progressive_global_bound_compute_filtered(
        pi, master->coverage, master->count, master->words,
        ninputs, weights, threads, memory, seconds, NULL,
        EXACT_PRICE_BATCH, &returned, patterns, bound
    );
    if (price == PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY) goto done;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_UNSUPPORTED;
    if (price == PROGRESSIVE_GLOBAL_BOUND_UNSUPPORTED) goto done;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
    if (price != PROGRESSIVE_GLOBAL_BOUND_OK) goto done;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_OK;
    for (int i = 0; i < returned; ++i) {
        const int *pattern = patterns + (size_t)i * (size_t)ninputs;
        /* Every master column must be a validated full-care cube. */
        if (!progressive_pattern_coverage_prepared(prepared, pattern, coverage)) {
            status = PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
            break;
        }
        bool inserted = false;
        const ProgressiveNativeMasterStatus insert =
            progressive_native_master_insert(master, coverage, pattern, &inserted);
        if (insert != PROGRESSIVE_NATIVE_MASTER_OK) {
            status = insert == PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY
                ? PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY
                : PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
            break;
        }
        step->inserted += inserted ? 1 : 0;
    }
    step->returned = returned;
    step->max_found = bound->max_cube_weight_found;
    step->upper = bound->max_cube_weight_upper_bound;
    step->complete = bound->pricing_complete;
done:
    free(patterns);
    free(coverage);
    return status;
}

static void incumbent_pattern(const int *solution, int cardinality, int ninputs,
                              int cube, int *pattern) {
    for (int input = 0; input < ninputs; ++input)
        pattern[input] = solution[(size_t)input * (size_t)cardinality + (size_t)cube];
}

/* Release the anchor and the incumbent cubes nearest to it: most shared ON
 * rows, then fewest differing literal positions, then lowest index. */
static void release_neighbourhood(const uint64_t *coverage, const int *solution,
                                  int cardinality, int ninputs, int words,
                                  int anchor, int release, int *order,
                                  int *shared, int *distance, char *released) {
    for (int c = 0; c < cardinality; ++c) {
        order[c] = c;
        shared[c] = 0;
        distance[c] = 0;
        for (int w = 0; w < words; ++w)
            shared[c] += __builtin_popcountll(
                coverage[(size_t)c * words + w] & coverage[(size_t)anchor * words + w]
            );
        for (int input = 0; input < ninputs; ++input)
            distance[c] +=
                solution[(size_t)input * cardinality + c] !=
                solution[(size_t)input * cardinality + anchor];
    }
    for (int i = 1; i < cardinality; ++i) {
        for (int j = i; j > 0; --j) {
            const int a = order[j - 1], b = order[j];
            const bool later = shared[a] < shared[b] || (
                shared[a] == shared[b] && (
                    distance[a] > distance[b] ||
                    (distance[a] == distance[b] && a > b)
                )
            );
            if (!later) break;
            order[j - 1] = b;
            order[j] = a;
        }
    }
    memset(released, 0, (size_t)cardinality);
    for (int i = 0; i < release; ++i) released[order[i]] = 1;
}

static Status price_node(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    const double *multipliers,
    int cardinality,
    const int *solution,
    int threads,
    uint64_t memory,
    double seconds,
    int anchor,
    int **improved,
    ProgressiveNativeExactPriceStep *step
) {
    const int rows = master->rows, words = master->words, ninputs = master->ninputs;
    int release = (cardinality + 7) / 8;
    if (release < EXACT_PRICE_MIN_RELEASE) release = EXACT_PRICE_MIN_RELEASE;
    if (release > cardinality) release = cardinality;
    step->anchor = anchor;
    step->released = release;

    Status status = PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY;
    ProgressiveNativeMaster node;
    memset(&node, 0, sizeof(node));
    uint64_t *coverage = (uint64_t *)calloc((size_t)cardinality * words, sizeof(uint64_t));
    uint64_t *covered = (uint64_t *)calloc((size_t)words, sizeof(uint64_t));
    int *pattern = (int *)malloc((size_t)ninputs * sizeof(int));
    int *order = (int *)malloc((size_t)cardinality * 3 * sizeof(int));
    char *released = (char *)malloc((size_t)cardinality);
    int *residual = (int *)malloc((size_t)rows * sizeof(int));
    double *weights = (double *)calloc((size_t)rows, sizeof(double));
    uint64_t *projected = NULL;
    double *warm = NULL, *node_multipliers = NULL;
    int *node_solution = NULL, *node_initial = NULL;
    if (!coverage || !covered || !pattern || !order || !released || !residual || !weights)
        goto done;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
    for (int c = 0; c < cardinality; ++c) {
        incumbent_pattern(solution, cardinality, ninputs, c, pattern);
        if (!progressive_pattern_coverage_prepared(
            prepared, pattern, coverage + (size_t)c * words)) goto done;
    }
    release_neighbourhood(coverage, solution, cardinality, ninputs, words, anchor,
                          release, order, order + cardinality,
                          order + 2 * cardinality, released);
    int fixed = 0;
    for (int c = 0; c < cardinality; ++c) {
        if (released[c]) continue;
        ++fixed;
        for (int w = 0; w < words; ++w) covered[w] |= coverage[(size_t)c * words + w];
    }
    int count = 0;
    for (int row = 0; row < rows; ++row)
        if (!has_row(covered, row)) residual[count++] = row;
    step->residual_rows = count;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_OK;
    if (count == 0) goto done;

    status = PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY;
    const int node_words = (count + 63) / 64;
    projected = (uint64_t *)malloc((size_t)node_words * sizeof(uint64_t));
    warm = (double *)malloc((size_t)count * sizeof(double));
    node_multipliers = (double *)malloc((size_t)count * sizeof(double));
    node_solution = (int *)malloc((size_t)count * sizeof(int));
    node_initial = (int *)malloc((size_t)release * sizeof(int));
    if (!projected || !warm || !node_multipliers || !node_solution || !node_initial ||
        !progressive_native_master_initialize(&node, count, node_words, ninputs))
        goto done;

    /* The master deduplicates projected coverage and keeps the smallest geometry. */
    for (int column = 0; column < master->count; ++column) {
        const uint64_t *bits = master->coverage + (size_t)column * words;
        memset(projected, 0, (size_t)node_words * sizeof(uint64_t));
        bool any = false;
        for (int i = 0; i < count; ++i) {
            if (!has_row(bits, residual[i])) continue;
            projected[i / 64] |= UINT64_C(1) << (i % 64);
            any = true;
        }
        if (!any) continue;
        bool inserted = false;
        const ProgressiveNativeMasterStatus insert = progressive_native_master_insert(
            &node, projected, master->patterns + (size_t)column * ninputs, &inserted
        );
        if (insert != PROGRESSIVE_NATIVE_MASTER_OK) {
            status = insert == PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY
                ? PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY
                : PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
            goto done;
        }
    }
    /* Seed from the saved incumbent: saved incumbent geometry may have
     * been replaced by another representative of its full ON coverage.
     * Distinct full-table columns can also coincide or vanish on this node.
     * Map by projected coverage, preserving the pool order and omitting empty
     * or duplicate projections from the feasible seed. */
    int initial_count = 0;
    for (int c = 0; c < cardinality; ++c) {
        if (!released[c]) continue;
        memset(projected, 0, (size_t)node_words * sizeof(uint64_t));
        bool any = false;
        for (int i = 0; i < count; ++i) {
            if (!has_row(coverage + (size_t)c * words, residual[i])) continue;
            projected[i / 64] |= UINT64_C(1) << (i % 64);
            any = true;
        }
        if (!any) continue;
        incumbent_pattern(solution, cardinality, ninputs, c, pattern);
        int column = node.count;
        bool inserted = false;
        const ProgressiveNativeMasterStatus insert = progressive_native_master_insert(
            &node, projected, pattern, &inserted
        );
        if (insert != PROGRESSIVE_NATIVE_MASTER_OK) {
            status = insert == PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY
                ? PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY
                : PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
            goto done;
        }
        if (!inserted) {
            for (column = 0; column < node.count; ++column)
                if (memcmp(node.coverage + (size_t)column * node_words, projected,
                           (size_t)node_words * sizeof(uint64_t)) == 0) break;
            if (column == node.count) {
                status = PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
                goto done;
            }
        }
        bool duplicate = false;
        for (int i = 0; i < initial_count; ++i)
            if (node_initial[i] == column) duplicate = true;
        if (!duplicate) node_initial[initial_count++] = column;
    }

    step->node_columns = node.count;

    /* Warm start from the parent multipliers restricted to the node rows. */
    double parent_sum = 0.0;
    for (int i = 0; i < count; ++i) {
        warm[i] = multipliers[residual[i]];
        parent_sum += warm[i];
    }
    int node_cover = 0;
    status = PROGRESSIVE_NATIVE_EXACT_PRICE_ERROR;
    if (progressive_native_master_solve_with_polish_incumbent(
        &node, 0, node_initial, initial_count, node_initial, initial_count,
        node_solution, &node_cover,
        node_multipliers, parent_sum > 0.0 ? warm : NULL
    ) != PROGRESSIVE_NATIVE_MASTER_OK) goto done;
    double bound = 0.0;
    for (int i = 0; i < count; ++i) {
        if (!isfinite(node_multipliers[i]) || node_multipliers[i] < 0.0) goto done;
        weights[residual[i]] = node_multipliers[i];
        bound += node_multipliers[i];
    }
    step->node_bound = bound;
    step->node_cover = node_cover;

    if (fixed + node_cover < cardinality) {
        const int next_cardinality = fixed + node_cover;
        int *candidate = (int *)malloc(
            (size_t)ninputs * (size_t)next_cardinality * sizeof(int)
        );
        status = PROGRESSIVE_NATIVE_EXACT_PRICE_OUT_OF_MEMORY;
        if (!candidate) goto done;
        int next = 0;
        for (int c = 0; c < cardinality; ++c) {
            if (released[c]) continue;
            for (int input = 0; input < ninputs; ++input)
                candidate[(size_t)input * next_cardinality + next] =
                    solution[(size_t)input * cardinality + c];
            ++next;
        }
        for (int i = 0; i < node_cover; ++i, ++next) {
            const int *source = node.patterns + (size_t)node_solution[i] * ninputs;
            for (int input = 0; input < ninputs; ++input)
                candidate[(size_t)input * next_cardinality + next] = source[input];
        }
        if (progressive_solution_is_valid_prepared(prepared, next_cardinality, candidate)) {
            *improved = candidate;
            step->improved_cardinality = next_cardinality;
        } else {
            free(candidate);
        }
    }

    ProgressiveGlobalBoundResult priced;
    status = price_and_insert(pi, prepared, master, weights, threads, memory,
                              seconds, step, &priced);
    if (status == PROGRESSIVE_NATIVE_EXACT_PRICE_OK && priced.pricing_complete)
        step->certified_lower = priced.cardinality_lower_bound;
done:
    progressive_native_master_destroy(&node);
    free(coverage);
    free(covered);
    free(pattern);
    free(order);
    free(released);
    free(residual);
    free(weights);
    free(projected);
    free(warm);
    free(node_multipliers);
    free(node_solution);
    free(node_initial);
    return status;
}

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
) {
    if (step) memset(step, 0, sizeof(*step));
    if (improved) *improved = NULL;
    if (!pi || !prepared || !master || !multipliers || !solution || !state ||
        !improved || !step || cardinality <= 0 || master->rows <= 0 ||
        !(seconds > 0.0))
        return PROGRESSIVE_NATIVE_EXACT_PRICE_INVALID;
    const uint64_t started = now_nanoseconds();
    const bool node = state->stalled &&
        state->stalled_calls % EXACT_PRICE_RECHECK_PERIOD !=
            EXACT_PRICE_RECHECK_PERIOD - 1;
    Status status;
    if (node) {
        const int anchor =
            (int)(((int64_t)EXACT_PRICE_ANCHOR_STRIDE * state->node_index) % cardinality);
        step->node = true;
        status = price_node(pi, prepared, master, multipliers, cardinality, solution,
                            requested_threads, worker_memory_budget_bytes, seconds,
                            anchor, improved, step);
        state->node_index++;
        state->node_calls++;
        state->stalled_calls++;
        if (step->improved_cardinality > 0) state->node_improvements++;
        if (step->certified_lower >= step->released && step->released > 0)
            state->node_certificates++;
    } else {
        ProgressiveGlobalBoundResult bound;
        status = price_and_insert(pi, prepared, master, multipliers,
                                  requested_threads, worker_memory_budget_bytes,
                                  seconds, step, &bound);
        if (state->stalled) state->stalled_calls++;
        state->stalled = status == PROGRESSIVE_NATIVE_EXACT_PRICE_OK && step->inserted == 0;
        if (!state->stalled) state->stalled_calls = 0;
    }
    state->calls++;
    state->inserted += step->inserted;
    const uint64_t elapsed = now_nanoseconds() - started;
    state->nanoseconds += elapsed;
    step->seconds = (double)elapsed / 1e9;
    if (status != PROGRESSIVE_NATIVE_EXACT_PRICE_OK) {
        free(*improved);
        *improved = NULL;
        step->improved_cardinality = 0;
    }
    return status;
}
