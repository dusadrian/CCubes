/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_COVER_H
#define PROGRESSIVE_COVER_H

#include <stdint.h>

#include "bounded_mmcs.h"
#include "progressive_global_bound.h"
#include "utils.h"

typedef enum {
    PROGRESSIVE_COVER_OK = 0,
    PROGRESSIVE_COVER_RESOURCE_LIMIT,
    PROGRESSIVE_COVER_UNAVAILABLE,
    PROGRESSIVE_COVER_INVALID_INPUT,
    PROGRESSIVE_COVER_INFEASIBLE,
    PROGRESSIVE_COVER_SOLVER_ERROR,
    PROGRESSIVE_COVER_PERSISTENCE_ERROR,
    PROGRESSIVE_COVER_OUT_OF_MEMORY
} ProgressiveCoverStatus;

typedef enum {
    PROGRESSIVE_STOP_STABLE = 0,
    PROGRESSIVE_STOP_MAX_HORIZON,
    PROGRESSIVE_STOP_BATCH_COMPLETE,
    PROGRESSIVE_STOP_CARDINALITY_BOUND,
    PROGRESSIVE_STOP_RESOURCE_LIMIT,
    PROGRESSIVE_STOP_CONSTANT_ZERO,
    PROGRESSIVE_STOP_CONSTANT_ONE
} ProgressiveCoverStopReason;

typedef enum {
    PROGRESSIVE_INCUMBENT_NONE = 0,
    PROGRESSIVE_INCUMBENT_SEED,
    PROGRESSIVE_INCUMBENT_COLUMN_POOL,
    PROGRESSIVE_INCUMBENT_PRICED_MASTER,
    PROGRESSIVE_INCUMBENT_CONSTANT
} ProgressiveCoverIncumbentSource;

typedef enum {
    PROGRESSIVE_MASTER_NATIVE = 0,
    PROGRESSIVE_MASTER_GUROBI
} ProgressiveCoverMasterBackend;

typedef bool (*ProgressiveCoverIncumbentCallback)(
    int ninputs,
    int cardinality,
    const int *solution,
    int selected_horizon,
    int generation,
    void *data
);

/* Immutable pool snapshots taken at completed native-search boundaries. */
typedef struct {
    double elapsed_seconds;
    const char *path;
} ProgressiveCoverPoolSnapshot;

typedef struct {
    int max_horizon;
    int stable_horizons;
    int requested_threads;
    uint64_t memory_budget_bytes;
    unsigned int batch_memory_percent;
    unsigned int worker_memory_percent;
    unsigned int seed_memory_percent;
    bool profile_pricing;
    bool greedy_seed;

    ProgressiveCoverMasterBackend master_backend;
    int native_effort_level;

    int native_repair_attempt_limit;
    bool native_repair_pricing;
    double native_repair_penalty;
    double native_repair_time_share;
    double native_feasibility_jump_seconds;
    bool native_feasibility_jump_work_budget;
    int native_feasibility_jump_logical_families;
    bool native_reduce_expand;
    int native_reduce_expand_variants;
    int native_reduce_expand_start_round;
    int native_reduce_expand_interval;
    bool native_private_micro;

    bool unlimited_tail_growth;

    bool enable_root_gate;

    bool disable_mmcs_cover_bound;
    bool cache_mmcs_edge_words;

    unsigned int greedy_anchor_percent;
    int greedy_forced_prefix_depth;
    int greedy_candidate_budget;
    int greedy_max_support;
    int greedy_scout_anchor_rounds;
    bool greedy_multi_start;
    double greedy_multistart_depth2_min_relative_room;
    bool greedy_single_pricing_batch;
    int greedy_mip_interval;
    double greedy_mip_time_limit_seconds;
    double greedy_final_mip_time_limit_seconds;
    int greedy_final_mip_focus;
    double greedy_final_mip_best_obj_stop;
    double time_limit_seconds;
    double global_bound_time_limit_seconds;
    int global_bound_active_rows;
    int global_bound_rounds;
    int literal_polish_mode;
    bool literal_polish_mode_explicit;

    const char *column_pool_input_path;
    const char *column_pool_output_path;

    const ProgressiveCoverPoolSnapshot *column_pool_snapshots;
    int column_pool_snapshot_count;

    ProgressiveCoverIncumbentCallback incumbent_callback;
    void *incumbent_callback_data;
} ProgressiveCoverOptions;

typedef struct {
    int cardinality;
    int selected_horizon;
    int final_horizon;
    int max_horizon;
    int stable_horizons;
    int stability_start_horizon;
    int stability_deferred_horizons;
    int retained_columns;
    uint64_t work_retained_columns;
    int pricing_rounds;
    int oracle_calls;
    uint64_t oracle_search_nodes;
    uint64_t oracle_bound_prunes;
    uint64_t oracle_completed_candidates;
    WeightedMMCSPriceTelemetry pricing_telemetry;
    uint64_t master_lp_nanoseconds;
    uint64_t integer_solve_nanoseconds;
    uint64_t normalization_nanoseconds;
    uint64_t setup_nanoseconds;
    uint64_t seed_call_nanoseconds;
    uint64_t seed_validation_nanoseconds;
    uint64_t seed_publish_nanoseconds;
    uint64_t incumbent_validation_nanoseconds;
    uint64_t incumbent_persistence_nanoseconds;
    uint64_t master_setup_nanoseconds;
    uint64_t memory_budget_bytes;
    uint64_t batch_allocated_bytes;
    uint64_t estimated_bytes_per_worker;
    uint64_t peak_incidence_entries;
    uint64_t seed_elapsed_nanoseconds;
    uint64_t seed_allocated_bytes;
    int batch_capacity;
    int pricing_threads;
    int on_rows_original;
    int off_rows_original;
    int on_rows_unique;
    int off_rows_unique;
    int on_duplicates;
    int off_duplicates;
    int conflicts;
    int seed_cardinality;
    int seed_literal_count;
    int seed_greedy_cubes;
    int seed_fallback_cubes;
    int seed_support_minimum;
    int seed_support_maximum;
    int support_lower_bound;
    int support_bound_anchor_minimum;
    int support_bound_anchor_maximum;
    int support_bound_threads;
    int horizons_skipped;
    int partial_anchor_limit;

    int greedy_anchor_limit;
    int greedy_forced_prefix_depth;
    int greedy_candidate_budget;
    int greedy_multistart_attempts;
    int greedy_multistart_improvements;
    int greedy_secondary_cardinality;
    int greedy_rounds;
    int greedy_intermediate_mip_calls;
    int greedy_final_mip_calls;
    int greedy_mip_rounds_skipped;
    int native_repair_attempts;
    int native_repair_improvements;
    int native_repair_skipped_optimal;
    int native_repair_sideways_moves;
    int native_repair_rejections;
    int native_repair_attempts_since_improvement;
    int native_repair_max_attempts_without_improvement;
    int native_repair_lagrangian_attempts;
    int native_repair_lagrangian_completions;
    int native_repair_greedy_fallbacks;
    int native_repair_max_uncovered_rows;
    int native_repair_max_residual_columns;
    uint64_t native_repair_destroyed_columns;
    uint64_t native_repair_uncovered_rows;
    uint64_t native_repair_residual_columns;
    uint64_t native_repair_lagrangian_columns;
    uint64_t native_repair_greedy_columns;
    uint64_t native_repair_redundant_removed;
    uint64_t native_repair_prepare_nanoseconds;
    uint64_t native_repair_solve_nanoseconds;
    uint64_t native_repair_budget_nanoseconds;
    /* Monotonic timestamp at which the native solve began. Lets budget-relative
     * policies scale against elapsed time when no wall target is configured,
     * instead of degenerating to "unbounded". */
    uint64_t native_started_nanoseconds;
    int native_repair_budget_skips;
    int native_repair_deadline_stops;
    int native_dual_calls;
    uint64_t native_dual_iterations;
    uint64_t native_dual_nanoseconds;
    uint64_t native_dual_last_nanoseconds;
    int native_feasibility_jump_runs;
    int native_feasibility_jump_targets_attempted;
    int native_feasibility_jump_targets_improved;
    uint64_t native_feasibility_jump_work_epochs;
    uint64_t native_feasibility_jump_productive_epoch_extensions;
    uint64_t native_feasibility_jump_adaptive_rounds;
    uint64_t native_feasibility_jump_restart_families;
    int native_feasibility_jump_threads_used;
    int native_feasibility_jump_logical_family_states;
    uint64_t native_feasibility_jump_restarts;
    uint64_t native_feasibility_jump_iterations;
    uint64_t native_feasibility_jump_add_moves;
    uint64_t native_feasibility_jump_remove_moves;
    uint64_t native_feasibility_jump_fixed_iterations;
    uint64_t native_feasibility_jump_soft_iterations;
    uint64_t native_feasibility_jump_weight_reuses;
    uint64_t native_feasibility_jump_soft_pressure_updates;
    int native_feasibility_jump_maximum_selected_columns;
    int native_feasibility_jump_best_uncovered_rows;
    int native_feasibility_jump_last_failed_uncovered_rows;
    uint64_t native_feasibility_jump_incidence_entries;
    uint64_t native_feasibility_jump_allocated_bytes;
    uint64_t native_feasibility_jump_shared_cache_bytes;
    uint64_t native_feasibility_jump_packed_threshold_bytes;
    int native_feasibility_jump_shared_cache_source;
    int native_feasibility_jump_dense_incidence_invocations;
    int native_feasibility_jump_packed_incidence_invocations;
    uint64_t native_feasibility_jump_nanoseconds;
    bool native_feasibility_jump_deadline_reached;
    bool native_feasibility_jump_memory_limited;
    int native_adaptive_phase_checkpoints;
    int native_adaptive_phase_promotions;
    int native_adaptive_phase_grace_promotions;
    int native_adaptive_phase_budget_promotions;
    int native_adaptive_phase_growth_promotions;
    int native_adaptive_phase_pool_checkpoints;
    bool native_adaptive_phase_stopped;
    int native_soft_target_extensions;
    uint64_t native_soft_target_extension_nanoseconds;
    double native_soft_target_grace_seconds;
    int native_soft_target_base_cardinality;
    bool native_soft_target_descending_at_base;
    int native_reduce_expand_passes;
    int native_reduce_expand_cubes;
    int native_reduce_expand_reduced_cores;
    int native_reduce_expand_variants;
    int native_reduce_expand_changed_geometries;
    int native_reduce_expand_inserted_columns;
    int native_reduce_expand_duplicate_columns;
    uint64_t native_reduce_expand_literals_added;
    uint64_t native_reduce_expand_nanoseconds;

    int greedy_support_minimum;
    int greedy_support_maximum;
    uint64_t greedy_anchors_attempted;
    uint64_t greedy_candidates_constructed;
    uint64_t greedy_improving_candidates;
    uint64_t greedy_duplicate_candidates;
    uint64_t greedy_literal_evaluations;
    uint64_t greedy_elapsed_nanoseconds;
    uint64_t greedy_allocated_bytes;
    uint64_t greedy_multistart_elapsed_nanoseconds;
    double greedy_support_mean;
    double greedy_best_weight;
    double greedy_mip_time_limit_seconds;
    double greedy_final_mip_time_limit_seconds;
    double integer_best_bound;
    double integer_mip_gap;
    uint64_t column_pool_load_nanoseconds;
    uint64_t column_pool_save_nanoseconds;
    int column_pool_loaded_columns;
    int column_pool_saved_columns;
    double seed_support_median;
    double support_bound_anchor_median;
    uint64_t support_bound_elapsed_nanoseconds;
    uint64_t support_bound_allocated_bytes;
    double lp_objective;
    double elapsed_seconds;
    ProgressiveCoverStopReason stop_reason;
    ProgressiveCoverIncumbentSource incumbent_source;
    int incumbent_generation;
    int pricing_closed_horizon;
    bool incumbent_valid;
    bool resource_limited;
    bool pricing_closed;
    bool seed_greedy_complete;
    bool seed_memory_exhausted;
    bool seed_master_inserted;
    bool support_bound_skip_applied;

    int restricted_bound_columns; /* Native unit-cost chart, zero if unavailable. */
    bool restricted_master_optimal;
    bool global_bound_requested;
    ProgressiveGlobalBoundStatus global_bound_status;
    ProgressiveGlobalBoundResult global_bound;

    /*
     * Transposed `ninputs x cardinality` matrix in write_pla_file format:
     * zero is a dash, one is input value zero, and two is input value one.
     * Ownership passes to the caller.
     */
    int *solution;
} ProgressiveCoverResult;

/*
 * Build a useful cover without first materialising the complete PI chart.
 *
 * The default unbounded greedy pricer adds negative-reduced-cost prime
 * columns to one persistent restricted master, uses sparse intermediate MIP
 * checkpoints, and reserves a longer final integer solve. Optional exact and
 * partial backends retain the support-horizon workflow.
 *
 * This is a heuristic integer minimizer: root pricing is exact through each
 * explored horizon, but stable integer cardinality is not a global
 * branch-and-price certificate.
 */
ProgressiveCoverStatus progressive_cover_run(
    const PIstorage *pi,
    int ninputs,
    int max_horizon,
    int stable_horizons,
    ProgressiveCoverResult *result
);

void progressive_cover_options_default(
    ProgressiveCoverOptions *options,
    int ninputs,
    int max_horizon,
    int stable_horizons
);

ProgressiveCoverStatus progressive_cover_run_with_options(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result
);

/*
 * Execute every output directly. Constant outputs are classified without
 * pricing; nonconstant outputs use independent normalized progressive
 * masters. The caller owns `results[output].solution`.
 */
ProgressiveCoverStatus progressive_cover_run_multi(
    const PIstorage *outputs,
    int ninputs,
    int noutputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *results
);

void progressive_cover_result_destroy(ProgressiveCoverResult *result);

const char *progressive_cover_status_name(ProgressiveCoverStatus status);
const char *progressive_cover_stop_reason_name(
    ProgressiveCoverStopReason reason
);
const char *progressive_cover_incumbent_source_name(
    ProgressiveCoverIncumbentSource source
);

#endif
