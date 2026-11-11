/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_RWLS_H
#define PROGRESSIVE_RWLS_H

#include <stdbool.h>
#include <stdint.h>

#include "progressive_native_master.h"

typedef enum {
    PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OK = 0,
    PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_INVALID,
    PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY,
    PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_MEMORY_LIMIT
} ProgressiveNativeFeasibilityJumpStatus;

typedef enum {
    PROGRESSIVE_RWLS_BUDGET_WORK = 0,
    PROGRESSIVE_RWLS_BUDGET_WALL
} ProgressiveRWLSBudgetMode;

typedef enum {
    PROGRESSIVE_RWLS_WORKER_FIXED_TARGET = 0,
    PROGRESSIVE_RWLS_WORKER_SOFT_PRESSURE,
    PROGRESSIVE_RWLS_WORKER_MIXED
} ProgressiveRWLSWorkerPolicy;

typedef enum {
    PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK = 0,
    PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_L3,
    PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_PERFORMANCE_L2,
    PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSCONF_L3,
    PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSFS,
    PROGRESSIVE_RWLS_SHARED_CACHE_WINDOWS_API
} ProgressiveRWLSSharedCacheSource;

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
    ProgressiveRWLSBudgetMode budget_mode;
    ProgressiveRWLSWorkerPolicy worker_policy;
    /* Wall mode only: repeat adaptive rounds on a failed target until the
     * deadline instead of stopping after the first failed round. */
    bool sustained_until_deadline;
    /* Sustained mode only: stop once this many seconds pass without a
     * smaller cover (counted from the start or the last improvement);
     * 0 keeps searching until the deadline. */
    double sustained_patience_seconds;
} ProgressiveNativeFeasibilityJumpOptions;

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
    ProgressiveRWLSSharedCacheSource shared_cache_source;
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
    /* Sustained mode: time of the start or the last improvement, and
     * whether the patience window ended the search. */
    uint64_t sustained_last_progress_nanoseconds;
    uint64_t sustained_last_improvement_offset_nanoseconds;
    bool patience_reached;
    ProgressiveRWLSBudgetMode budget_mode;
} ProgressiveNativeFeasibilityJumpStats;

void progressive_native_feasibility_jump_options_default(
    ProgressiveNativeFeasibilityJumpOptions *options
);

/*
 * Resolve one mixed-policy family independently of physical worker count.
 * Families [0, N) therefore form the same persistent policy prefix at every
 * thread count and across every target and adaptive round.
 */
ProgressiveRWLSWorkerPolicy progressive_rwls_mixed_family_policy(
    int soft_divisor,
    uint64_t family_ordinal
);

/*
 * Return the smallest complete persistent policy prefix that can keep the
 * requested physical executors occupied. Growth is in complete policy blocks,
 * so the fixed/soft proportion never depends on a partial final block. Zero
 * reports invalid arguments or an unsupported family count.
 */
int progressive_rwls_mixed_logical_family_count(
    int physical_thread_count,
    int soft_divisor
);

/*
 * Return the capacity of one last-level/shared cache domain. Cache instances
 * are deliberately not summed: the RWLS coverage working set is shared, so
 * aggregate cache across independent domains would overstate its residency.
 * The result is detected once and then cached for the life of the process.
 */
uint64_t progressive_rwls_shared_cache_bytes(void);

ProgressiveRWLSSharedCacheSource
progressive_rwls_shared_cache_source(void);

const char *progressive_rwls_shared_cache_source_name(
    ProgressiveRWLSSharedCacheSource source
);

/* Zero is retained as the compile-time test override that forces packing. */
uint64_t progressive_rwls_packed_incidence_threshold_bytes(void);

/*
 * Search the retained native master for a cover smaller than `incumbent`.
 * Complete blocks in the default logical-family mix contain three fixed-target
 * RWLS trajectories and one scalar-soft-pressure trajectory; logical growth
 * and memory fallback retain complete blocks in deterministic order.
 * Fixed-target families cross between
 * checkpoints through infeasible add/remove pairs. Soft-pressure families may
 * temporarily grow the selection and trade row penalties against one global
 * unicost size pressure. If a mixed target fails, only its persistent
 * fixed-target families retry before stopping; soft-family states are not
 * repurposed. Persistent uncovered rows gain weight in both policies.
 *
 * Returned indices belong to `master`. The returned cover is always valid;
 * when no improvement is found it is a copy of the supplied incumbent.
 */
ProgressiveNativeFeasibilityJumpStatus
progressive_native_feasibility_jump_improve(
    const ProgressiveNativeMaster *master,
    int incumbent_cardinality,
    const int *incumbent_columns,
    const ProgressiveNativeFeasibilityJumpOptions *options,
    int **selected_columns,
    int *cardinality,
    ProgressiveNativeFeasibilityJumpStats *stats
);

#endif
