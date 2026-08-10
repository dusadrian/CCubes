/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROJECTION_RWLS_H
#define PROJECTION_RWLS_H

#include <stdbool.h>
#include <stdint.h>

#include "pichart_view.h"

typedef enum {
    PROJECTION_RWLS_OK = 0,
    PROJECTION_RWLS_INVALID,
    PROJECTION_RWLS_OUT_OF_MEMORY,
    PROJECTION_RWLS_MEMORY_LIMIT
} ProjectionRWLSStatus;

typedef enum {
    PROJECTION_RWLS_BUDGET_WORK = 0,
    PROJECTION_RWLS_BUDGET_WALL
} ProjectionRWLSBudgetMode;

typedef enum {
    PROJECTION_RWLS_WORKER_FIXED_TARGET = 0,
    PROJECTION_RWLS_WORKER_SOFT_PRESSURE,
    PROJECTION_RWLS_WORKER_MIXED
} ProjectionRWLSWorkerPolicy;

typedef enum {
    PROJECTION_RWLS_SHARED_CACHE_FALLBACK = 0,
    PROJECTION_RWLS_SHARED_CACHE_DARWIN_L3,
    PROJECTION_RWLS_SHARED_CACHE_DARWIN_PERFORMANCE_L2,
    PROJECTION_RWLS_SHARED_CACHE_LINUX_SYSCONF_L3,
    PROJECTION_RWLS_SHARED_CACHE_LINUX_SYSFS,
    PROJECTION_RWLS_SHARED_CACHE_WINDOWS_API
} ProjectionRWLSSharedCacheSource;

typedef struct {
    uint64_t steps_per_family;
    int max_adaptive_rounds;
    int requested_threads;
    int logical_family_count;
    int tabu_tenure;
    int mixed_soft_worker_divisor;
    uint64_t seed;
    uint64_t deadline_nanoseconds;
    uint64_t memory_budget_bytes;
    ProjectionRWLSBudgetMode budget_mode;
    ProjectionRWLSWorkerPolicy worker_policy;
} ProjectionRWLSOptions;

typedef struct {
    int initial_cardinality;
    int final_cardinality;
    int targets_attempted;
    int targets_improved;
    uint64_t work_epochs;
    uint64_t productive_epoch_extensions;
    uint64_t adaptive_rounds;
    uint64_t restart_families;
    int threads_used;
    int logical_family_states;
    uint64_t restarts;
    uint64_t iterations;
    uint64_t add_moves;
    uint64_t remove_moves;
    uint64_t penalty_updates;
    uint64_t weight_reuses;
    uint64_t soft_pressure_updates;
    int maximum_selected_columns;
    int best_uncovered_rows;
    int last_failed_uncovered_rows;
    int best_failed_witness_row;
    uint64_t best_failed_selection_signature;
    int last_failed_witness_row;
    int last_failed_witness_frequency;
    int last_failed_witness_distinct_rows;
    int last_failed_witness_observations;
    uint64_t last_failed_selection_signature;
    uint64_t fixed_restart_families;
    uint64_t soft_restart_families;
    uint64_t fixed_iterations;
    uint64_t soft_iterations;
    int fixed_successful_candidates;
    int soft_successful_candidates;
    int fixed_winning_targets;
    int soft_winning_targets;
    int last_successful_candidates;
    int last_successful_distinct_candidates;
    int last_successful_min_symmetric_difference;
    int last_successful_max_symmetric_difference;
    int last_winning_policy;
    int compulsory_columns;
    int distinct_selected_columns;
    int maximum_worker_distinct_selected_columns;
    uint64_t incidence_entries;
    int column_incidence_index_bytes;
    bool bidirectional_incidence;
    uint64_t shared_cache_bytes;
    uint64_t packed_incidence_threshold_bytes;
    ProjectionRWLSSharedCacheSource shared_cache_source;
    int dense_incidence_invocations;
    int packed_incidence_invocations;
    uint64_t allocated_bytes;
    uint64_t shared_allocated_bytes;
    uint64_t per_family_allocated_bytes;
    int64_t maximum_row_penalty;
    int64_t maximum_uncovered_gain;
    int64_t maximum_private_loss;
    int64_t maximum_timestamp;
    uint64_t elapsed_nanoseconds;
    bool deadline_reached;
    ProjectionRWLSBudgetMode budget_mode;
} ProjectionRWLSStats;

void projection_rwls_options_default(
    ProjectionRWLSOptions *options
);

/*
 * Keep the projection integration policy explicit and independently testable.
 * RWLS is a heuristic fixed-chart finisher: it is useful only at a plateau,
 * only above the one-column global floor, and never after the chart solver has
 * proved its incumbent optimal. Certified mode retains its existing path.
 */
bool projection_rwls_should_run(
    bool enabled,
    bool certified_mode,
    bool boundary_exact,
    bool plateau_due,
    int incumbent_cardinality
);

/*
 * Resolve one mixed-policy family independently of physical worker count.
 * Families [0, N) therefore form the same persistent policy prefix at every
 * thread count and across every target and adaptive round.
 */
ProjectionRWLSWorkerPolicy projection_rwls_mixed_family_policy(
    int soft_divisor,
    uint64_t family_ordinal
);

/*
 * Return the smallest complete persistent policy prefix that can keep the
 * requested physical executors occupied. Growth is in complete policy blocks,
 * so the fixed/soft proportion never depends on a partial final block. Zero
 * reports invalid arguments or an unsupported family count.
 */
int projection_rwls_mixed_logical_family_count(
    int physical_thread_count,
    int soft_divisor
);

/*
 * Return the capacity of one last-level/shared cache domain. Cache instances
 * are deliberately not summed: the RWLS coverage working set is shared, so
 * aggregate cache across independent domains would overstate its residency.
 * The result is detected once and then cached for the life of the process.
 */
uint64_t projection_rwls_shared_cache_bytes(void);

ProjectionRWLSSharedCacheSource
projection_rwls_shared_cache_source(void);

const char *projection_rwls_shared_cache_source_name(
    ProjectionRWLSSharedCacheSource source
);

/* Zero is retained as the compile-time test override that forces packing. */
uint64_t projection_rwls_packed_incidence_threshold_bytes(void);

/*
 * Search a fixed projection PI chart for a cover smaller than `incumbent`.
 * Complete blocks in the default logical-family mix contain three fixed-target
 * RWLS trajectories and one scalar-soft-pressure trajectory; logical growth
 * and memory admission retain complete blocks in deterministic order.
 * Fixed-target families cross between
 * checkpoints through infeasible add/remove pairs. Soft-pressure families may
 * temporarily grow the selection and trade row penalties against one global
 * unicost size pressure. If a mixed target fails, only its persistent
 * fixed-target families retry before stopping; soft-family states are not
 * repurposed. Persistent uncovered rows gain weight in both policies.
 *
 * Returned indices belong to `chart`. The returned cover is always valid;
 * when no improvement is found it is a copy of the supplied incumbent.
 */
ProjectionRWLSStatus
projection_rwls_improve(
    const PIChartView *chart,
    int incumbent_cardinality,
    const int *incumbent_columns,
    const ProjectionRWLSOptions *options,
    int **selected_columns,
    int *cardinality,
    ProjectionRWLSStats *stats
);

#endif
