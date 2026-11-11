/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_cover.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bounded_mmcs.h"
#include "ccubes_threads.h"
#include "progressive_column_pool.h"
#include "progressive_greedy_price.h"
#include "progressive_input.h"
#include "progressive_native_cover.h"
#include "progressive_seed.h"
#include "progressive_validation.h"

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <unistd.h>
#endif

static uint64_t detected_memory_budget(void) {
    uint64_t physical = 0;
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        physical = (uint64_t)status.ullTotalPhys;
    }
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        uint64_t upages = (uint64_t)pages;
        uint64_t usize = (uint64_t)page_size;
        if (upages <= UINT64_MAX / usize) {
            physical = upages * usize;
        }
    }
#endif
    /*
     * CCubes is not the only process on the machine. Half of physical memory
     * is the automatic process envelope; callers can provide an explicit
     * lower or higher budget.
     */
    return physical > 0 ? physical / 2u : UINT64_C(1073741824);
}

void progressive_cover_options_default(
    ProgressiveCoverOptions *options,
    int ninputs,
    int max_horizon,
    int stable_horizons
) {
    if (!options) return;
    *options = (ProgressiveCoverOptions){
        .max_horizon = max_horizon > 0 ? max_horizon : ninputs,
        .stable_horizons = stable_horizons > 0 ? stable_horizons : 2,
        .requested_threads = ccubes_thread_count(),
        .memory_budget_bytes = detected_memory_budget(),
        .batch_memory_percent = 25,
        .worker_memory_percent = 50,
        .seed_memory_percent = 10,
        .profile_pricing = false,
        .greedy_seed = true,

        .master_backend = PROGRESSIVE_MASTER_NATIVE,
        .native_effort_level = 0,

        .native_repair_attempt_limit = 16,
        .native_repair_pricing = true,
        .native_repair_penalty = 1.0,
        .native_repair_time_share = 0.05,
        .native_feasibility_jump_seconds = 1.0,
        .native_feasibility_jump_work_budget = true,
        .native_feasibility_jump_logical_families = 0,
        .native_reduce_expand = true,
        .native_reduce_expand_variants = 8,
        .native_reduce_expand_start_round = 8,
        .native_reduce_expand_interval = 4,
        .native_private_micro = true,

        .unlimited_tail_growth = false,

        .enable_root_gate = true,

        .disable_mmcs_cover_bound = false,
        .cache_mmcs_edge_words = true,

        .greedy_anchor_percent = 10,
        .greedy_forced_prefix_depth = -1,
        .greedy_candidate_budget = 16384,
        .greedy_max_support = 0,
        .greedy_scout_anchor_rounds = 0,
        .greedy_multi_start = true,
        .greedy_multistart_depth2_min_relative_room = 0.24,
        .greedy_single_pricing_batch = false,
        .greedy_mip_interval = 8,
        .greedy_mip_time_limit_seconds = 5.0,
        .greedy_final_mip_time_limit_seconds = 60.0,
        .greedy_final_mip_focus = 0,
        .greedy_final_mip_best_obj_stop = -1.0,
        .time_limit_seconds = 0.0,
        .native_terminal_reserve_seconds = 0.0,
        .native_terminal_patience_seconds = CCUBES_TERMINAL_PATIENCE_DEFAULT,
        .global_bound_time_limit_seconds = 0.0,
        .global_bound_active_rows = 512,
        .global_bound_rounds = 5,
        .literal_polish_mode = 1,
        .literal_polish_mode_explicit = false,

        .column_pool_input_path = NULL,
        .column_pool_output_path = NULL,

        .column_pool_snapshots = NULL,
        .column_pool_snapshot_count = 0,

        .incumbent_callback = NULL,
        .incumbent_callback_data = NULL
    };
}

static uint64_t monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static int progressive_solution_literal_count(
    int ninputs,
    int cardinality,
    const int *solution
) {
    if (ninputs <= 0 || cardinality <= 0 || !solution) return INT_MAX;
    int literals = 0;
    for (int input = 0; input < ninputs; ++input) {
        for (int cube = 0; cube < cardinality; ++cube) {
            literals += solution[
                (size_t)input * (size_t)cardinality + (size_t)cube
            ] != 0;
        }
    }
    return literals;
}

#define PROGRESSIVE_VALUE_TOL 1e-7


static ProgressiveCoverStatus progressive_cover_run_core(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result
) {
    if (
        options &&
        options->master_backend == PROGRESSIVE_MASTER_NATIVE
    ) {
        return progressive_native_cover_run(
            pi,
            ninputs,
            options,

            result
        );
    }
    (void)pi;
    (void)ninputs;

    if (result) memset(result, 0, sizeof(*result));
    return PROGRESSIVE_COVER_UNAVAILABLE;
}

/*
 * Independent multistart trajectories are real work even when their cover is
 * not selected.  Keep the published work vector as the sum of every attempted
 * arm while leaving solution geometry and final-pool state with the winner.
 */
static void progressive_cover_work_accumulate(
    ProgressiveCoverResult *total,
    const ProgressiveCoverResult *part
) {
    if (!total || !part) return;
    total->work_retained_columns += (uint64_t)part->retained_columns;
    if (part->pricing_threads > total->pricing_threads) {
        total->pricing_threads = part->pricing_threads;
    }
    total->greedy_rounds += part->greedy_rounds;
    total->pricing_rounds += part->pricing_rounds;
    total->oracle_calls += part->oracle_calls;
    total->oracle_search_nodes += part->oracle_search_nodes;
    total->oracle_bound_prunes += part->oracle_bound_prunes;
    total->oracle_completed_candidates +=
        part->oracle_completed_candidates;
    total->greedy_anchors_attempted += part->greedy_anchors_attempted;
    total->greedy_candidates_constructed +=
        part->greedy_candidates_constructed;
    total->greedy_improving_candidates +=
        part->greedy_improving_candidates;
    total->greedy_duplicate_candidates +=
        part->greedy_duplicate_candidates;
    total->greedy_literal_evaluations +=
        part->greedy_literal_evaluations;
    total->greedy_elapsed_nanoseconds += part->greedy_elapsed_nanoseconds;
    total->greedy_intermediate_mip_calls +=
        part->greedy_intermediate_mip_calls;
    total->greedy_final_mip_calls += part->greedy_final_mip_calls;
    total->greedy_mip_rounds_skipped +=
        part->greedy_mip_rounds_skipped;

    total->native_dual_calls += part->native_dual_calls;
    total->native_dual_iterations += part->native_dual_iterations;
    total->native_dual_nanoseconds += part->native_dual_nanoseconds;

    total->native_adaptive_phase_checkpoints +=
        part->native_adaptive_phase_checkpoints;
    total->native_adaptive_phase_promotions +=
        part->native_adaptive_phase_promotions;
    total->native_adaptive_phase_grace_promotions +=
        part->native_adaptive_phase_grace_promotions;
    total->native_adaptive_phase_budget_promotions +=
        part->native_adaptive_phase_budget_promotions;
    total->native_adaptive_phase_growth_promotions +=
        part->native_adaptive_phase_growth_promotions;
    total->native_adaptive_phase_pool_checkpoints +=
        part->native_adaptive_phase_pool_checkpoints;
    total->native_adaptive_phase_stopped =
        total->native_adaptive_phase_stopped ||
        part->native_adaptive_phase_stopped;

    /* Rolling-target events describe run-level scheduling work, not the
     * geometry of whichever multistart arm supplied the final cover. */
    total->native_soft_target_extensions +=
        part->native_soft_target_extensions;
    total->native_soft_target_extension_nanoseconds +=
        part->native_soft_target_extension_nanoseconds;
    if (
        part->native_soft_target_grace_seconds >
        total->native_soft_target_grace_seconds
    ) {
        total->native_soft_target_grace_seconds =
            part->native_soft_target_grace_seconds;
    }
    if (
        total->native_soft_target_base_cardinality <= 0 &&
        part->native_soft_target_base_cardinality > 0
    ) {
        total->native_soft_target_base_cardinality =
            part->native_soft_target_base_cardinality;
    }
    total->native_soft_target_descending_at_base =
        total->native_soft_target_descending_at_base ||
        part->native_soft_target_descending_at_base;

    total->native_feasibility_jump_runs +=
        part->native_feasibility_jump_runs;
    total->native_feasibility_jump_targets_attempted +=
        part->native_feasibility_jump_targets_attempted;
    total->native_feasibility_jump_targets_improved +=
        part->native_feasibility_jump_targets_improved;
    total->native_feasibility_jump_work_epochs +=
        part->native_feasibility_jump_work_epochs;
    total->native_feasibility_jump_productive_epoch_extensions +=
        part->native_feasibility_jump_productive_epoch_extensions;
    total->native_feasibility_jump_adaptive_rounds +=
        part->native_feasibility_jump_adaptive_rounds;
    total->native_feasibility_jump_restart_families +=
        part->native_feasibility_jump_restart_families;
    if (
        part->native_feasibility_jump_threads_used >
        total->native_feasibility_jump_threads_used
    ) {
        total->native_feasibility_jump_threads_used =
            part->native_feasibility_jump_threads_used;
    }
    if (
        part->native_feasibility_jump_logical_family_states >
        total->native_feasibility_jump_logical_family_states
    ) {
        total->native_feasibility_jump_logical_family_states =
            part->native_feasibility_jump_logical_family_states;
    }
    total->native_feasibility_jump_restarts +=
        part->native_feasibility_jump_restarts;
    total->native_feasibility_jump_iterations +=
        part->native_feasibility_jump_iterations;
    total->native_feasibility_jump_add_moves +=
        part->native_feasibility_jump_add_moves;
    total->native_feasibility_jump_remove_moves +=
        part->native_feasibility_jump_remove_moves;
    total->native_feasibility_jump_fixed_iterations +=
        part->native_feasibility_jump_fixed_iterations;
    total->native_feasibility_jump_soft_iterations +=
        part->native_feasibility_jump_soft_iterations;
    total->native_feasibility_jump_weight_reuses +=
        part->native_feasibility_jump_weight_reuses;
    total->native_feasibility_jump_soft_pressure_updates +=
        part->native_feasibility_jump_soft_pressure_updates;
    total->native_feasibility_jump_dense_incidence_invocations +=
        part->native_feasibility_jump_dense_incidence_invocations;
    total->native_feasibility_jump_packed_incidence_invocations +=
        part->native_feasibility_jump_packed_incidence_invocations;
    total->native_feasibility_jump_nanoseconds +=
        part->native_feasibility_jump_nanoseconds;
    if (
        part->native_feasibility_jump_shared_cache_bytes >
        total->native_feasibility_jump_shared_cache_bytes
    ) {
        total->native_feasibility_jump_shared_cache_bytes =
            part->native_feasibility_jump_shared_cache_bytes;
        total->native_feasibility_jump_shared_cache_source =
            part->native_feasibility_jump_shared_cache_source;
    }
    if (
        part->native_feasibility_jump_packed_threshold_bytes >
        total->native_feasibility_jump_packed_threshold_bytes
    ) {
        total->native_feasibility_jump_packed_threshold_bytes =
            part->native_feasibility_jump_packed_threshold_bytes;
    }

    total->native_reduce_expand_passes +=
        part->native_reduce_expand_passes;
    total->native_reduce_expand_cubes += part->native_reduce_expand_cubes;
    total->native_reduce_expand_reduced_cores +=
        part->native_reduce_expand_reduced_cores;
    total->native_reduce_expand_variants +=
        part->native_reduce_expand_variants;
    total->native_reduce_expand_changed_geometries +=
        part->native_reduce_expand_changed_geometries;
    total->native_reduce_expand_inserted_columns +=
        part->native_reduce_expand_inserted_columns;
    total->native_reduce_expand_duplicate_columns +=
        part->native_reduce_expand_duplicate_columns;
    total->native_reduce_expand_literals_added +=
        part->native_reduce_expand_literals_added;
    total->native_reduce_expand_nanoseconds +=
        part->native_reduce_expand_nanoseconds;

    total->native_repair_attempts += part->native_repair_attempts;
    total->native_repair_improvements += part->native_repair_improvements;
    total->native_repair_skipped_optimal +=
        part->native_repair_skipped_optimal;
    total->native_repair_sideways_moves +=
        part->native_repair_sideways_moves;
    total->native_repair_rejections += part->native_repair_rejections;
    total->native_repair_lagrangian_attempts +=
        part->native_repair_lagrangian_attempts;
    total->native_repair_lagrangian_completions +=
        part->native_repair_lagrangian_completions;
    total->native_repair_greedy_fallbacks +=
        part->native_repair_greedy_fallbacks;
    if (
        part->native_repair_max_uncovered_rows >
        total->native_repair_max_uncovered_rows
    ) {
        total->native_repair_max_uncovered_rows =
            part->native_repair_max_uncovered_rows;
    }
    if (
        part->native_repair_max_residual_columns >
        total->native_repair_max_residual_columns
    ) {
        total->native_repair_max_residual_columns =
            part->native_repair_max_residual_columns;
    }
    total->native_repair_destroyed_columns +=
        part->native_repair_destroyed_columns;
    total->native_repair_uncovered_rows +=
        part->native_repair_uncovered_rows;
    total->native_repair_residual_columns +=
        part->native_repair_residual_columns;
    total->native_repair_lagrangian_columns +=
        part->native_repair_lagrangian_columns;
    total->native_repair_greedy_columns +=
        part->native_repair_greedy_columns;
    total->native_repair_redundant_removed +=
        part->native_repair_redundant_removed;
    total->native_repair_prepare_nanoseconds +=
        part->native_repair_prepare_nanoseconds;
    total->native_repair_solve_nanoseconds +=
        part->native_repair_solve_nanoseconds;
    total->native_repair_budget_nanoseconds +=
        part->native_repair_budget_nanoseconds;
    total->native_repair_budget_skips +=
        part->native_repair_budget_skips;
    total->native_repair_deadline_stops +=
        part->native_repair_deadline_stops;
}

static void progressive_cover_work_apply(
    ProgressiveCoverResult *result,
    const ProgressiveCoverResult *total
) {
    if (!result || !total) return;
    result->work_retained_columns = total->work_retained_columns;
    result->pricing_threads = total->pricing_threads;
    result->greedy_rounds = total->greedy_rounds;
    result->pricing_rounds = total->pricing_rounds;
    result->oracle_calls = total->oracle_calls;
    result->oracle_search_nodes = total->oracle_search_nodes;
    result->oracle_bound_prunes = total->oracle_bound_prunes;
    result->oracle_completed_candidates = total->oracle_completed_candidates;
    result->greedy_anchors_attempted = total->greedy_anchors_attempted;
    result->greedy_candidates_constructed =
        total->greedy_candidates_constructed;
    result->greedy_improving_candidates =
        total->greedy_improving_candidates;
    result->greedy_duplicate_candidates =
        total->greedy_duplicate_candidates;
    result->greedy_literal_evaluations =
        total->greedy_literal_evaluations;
    result->greedy_elapsed_nanoseconds = total->greedy_elapsed_nanoseconds;
    result->greedy_intermediate_mip_calls =
        total->greedy_intermediate_mip_calls;
    result->greedy_final_mip_calls = total->greedy_final_mip_calls;
    result->greedy_mip_rounds_skipped =
        total->greedy_mip_rounds_skipped;

    result->native_dual_calls = total->native_dual_calls;
    result->native_dual_iterations = total->native_dual_iterations;
    result->native_dual_nanoseconds = total->native_dual_nanoseconds;

    result->native_adaptive_phase_checkpoints =
        total->native_adaptive_phase_checkpoints;
    result->native_adaptive_phase_promotions =
        total->native_adaptive_phase_promotions;
    result->native_adaptive_phase_grace_promotions =
        total->native_adaptive_phase_grace_promotions;
    result->native_adaptive_phase_budget_promotions =
        total->native_adaptive_phase_budget_promotions;
    result->native_adaptive_phase_growth_promotions =
        total->native_adaptive_phase_growth_promotions;
    result->native_adaptive_phase_pool_checkpoints =
        total->native_adaptive_phase_pool_checkpoints;
    result->native_adaptive_phase_stopped =
        total->native_adaptive_phase_stopped;

    result->native_soft_target_extensions =
        total->native_soft_target_extensions;
    result->native_soft_target_extension_nanoseconds =
        total->native_soft_target_extension_nanoseconds;
    result->native_soft_target_grace_seconds =
        total->native_soft_target_grace_seconds;
    result->native_soft_target_base_cardinality =
        total->native_soft_target_base_cardinality;
    result->native_soft_target_descending_at_base =
        total->native_soft_target_descending_at_base;

    result->native_feasibility_jump_runs =
        total->native_feasibility_jump_runs;
    result->native_feasibility_jump_targets_attempted =
        total->native_feasibility_jump_targets_attempted;
    result->native_feasibility_jump_targets_improved =
        total->native_feasibility_jump_targets_improved;
    result->native_feasibility_jump_work_epochs =
        total->native_feasibility_jump_work_epochs;
    result->native_feasibility_jump_productive_epoch_extensions =
        total->native_feasibility_jump_productive_epoch_extensions;
    result->native_feasibility_jump_adaptive_rounds =
        total->native_feasibility_jump_adaptive_rounds;
    result->native_feasibility_jump_restart_families =
        total->native_feasibility_jump_restart_families;
    result->native_feasibility_jump_threads_used =
        total->native_feasibility_jump_threads_used;
    result->native_feasibility_jump_logical_family_states =
        total->native_feasibility_jump_logical_family_states;
    result->native_feasibility_jump_restarts =
        total->native_feasibility_jump_restarts;
    result->native_feasibility_jump_iterations =
        total->native_feasibility_jump_iterations;
    result->native_feasibility_jump_add_moves =
        total->native_feasibility_jump_add_moves;
    result->native_feasibility_jump_remove_moves =
        total->native_feasibility_jump_remove_moves;
    result->native_feasibility_jump_fixed_iterations =
        total->native_feasibility_jump_fixed_iterations;
    result->native_feasibility_jump_soft_iterations =
        total->native_feasibility_jump_soft_iterations;
    result->native_feasibility_jump_weight_reuses =
        total->native_feasibility_jump_weight_reuses;
    result->native_feasibility_jump_soft_pressure_updates =
        total->native_feasibility_jump_soft_pressure_updates;
    result->native_feasibility_jump_dense_incidence_invocations =
        total->native_feasibility_jump_dense_incidence_invocations;
    result->native_feasibility_jump_packed_incidence_invocations =
        total->native_feasibility_jump_packed_incidence_invocations;
    result->native_feasibility_jump_nanoseconds =
        total->native_feasibility_jump_nanoseconds;
    result->native_feasibility_jump_shared_cache_bytes =
        total->native_feasibility_jump_shared_cache_bytes;
    result->native_feasibility_jump_shared_cache_source =
        total->native_feasibility_jump_shared_cache_source;
    result->native_feasibility_jump_packed_threshold_bytes =
        total->native_feasibility_jump_packed_threshold_bytes;

    result->native_reduce_expand_passes =
        total->native_reduce_expand_passes;
    result->native_reduce_expand_cubes = total->native_reduce_expand_cubes;
    result->native_reduce_expand_reduced_cores =
        total->native_reduce_expand_reduced_cores;
    result->native_reduce_expand_variants =
        total->native_reduce_expand_variants;
    result->native_reduce_expand_changed_geometries =
        total->native_reduce_expand_changed_geometries;
    result->native_reduce_expand_inserted_columns =
        total->native_reduce_expand_inserted_columns;
    result->native_reduce_expand_duplicate_columns =
        total->native_reduce_expand_duplicate_columns;
    result->native_reduce_expand_literals_added =
        total->native_reduce_expand_literals_added;
    result->native_reduce_expand_nanoseconds =
        total->native_reduce_expand_nanoseconds;

    result->native_repair_attempts = total->native_repair_attempts;
    result->native_repair_improvements = total->native_repair_improvements;
    result->native_repair_skipped_optimal =
        total->native_repair_skipped_optimal;
    result->native_repair_sideways_moves =
        total->native_repair_sideways_moves;
    result->native_repair_rejections = total->native_repair_rejections;
    result->native_repair_lagrangian_attempts =
        total->native_repair_lagrangian_attempts;
    result->native_repair_lagrangian_completions =
        total->native_repair_lagrangian_completions;
    result->native_repair_greedy_fallbacks =
        total->native_repair_greedy_fallbacks;
    result->native_repair_max_uncovered_rows =
        total->native_repair_max_uncovered_rows;
    result->native_repair_max_residual_columns =
        total->native_repair_max_residual_columns;
    result->native_repair_destroyed_columns =
        total->native_repair_destroyed_columns;
    result->native_repair_uncovered_rows =
        total->native_repair_uncovered_rows;
    result->native_repair_residual_columns =
        total->native_repair_residual_columns;
    result->native_repair_lagrangian_columns =
        total->native_repair_lagrangian_columns;
    result->native_repair_greedy_columns =
        total->native_repair_greedy_columns;
    result->native_repair_redundant_removed =
        total->native_repair_redundant_removed;
    result->native_repair_prepare_nanoseconds =
        total->native_repair_prepare_nanoseconds;
    result->native_repair_solve_nanoseconds =
        total->native_repair_solve_nanoseconds;
    result->native_repair_budget_nanoseconds =
        total->native_repair_budget_nanoseconds;
    result->native_repair_budget_skips =
        total->native_repair_budget_skips;
    result->native_repair_deadline_stops =
        total->native_repair_deadline_stops;
}

ProgressiveCoverStatus progressive_cover_run_with_options(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result
) {
    if (result) memset(result, 0, sizeof(*result));
    if (
        !pi ||
        !options ||
        !result ||
        ninputs <= 0 ||
        !isfinite(
            options->greedy_multistart_depth2_min_relative_room
        ) ||
        options->greedy_max_support < 0 ||
        options->greedy_max_support > ninputs ||
        options->greedy_scout_anchor_rounds < 0 ||
        options->greedy_multistart_depth2_min_relative_room < 0.0 ||
        options->greedy_multistart_depth2_min_relative_room > 1.0
    ) {
        return PROGRESSIVE_COVER_INVALID_INPUT;
    }

    uint64_t normalization_started = options->profile_pricing
        ? monotonic_nanoseconds()
        : 0;
    ProgressiveNormalizedInput normalized;
    ProgressiveInputStatus input_status = progressive_input_normalize(
        pi,
        ninputs,
        &normalized
    );
    uint64_t normalization_nanoseconds = normalization_started > 0
        ? monotonic_nanoseconds() - normalization_started
        : 0;
    result->normalization_nanoseconds = normalization_nanoseconds;
    result->on_rows_original = normalized.on_original;
    result->off_rows_original = normalized.off_original;
    result->on_duplicates = normalized.on_duplicates;
    result->off_duplicates = normalized.off_duplicates;
    result->conflicts = normalized.conflicts;
    if (input_status != PROGRESSIVE_INPUT_OK) {
        ProgressiveCoverStatus status =
            input_status == PROGRESSIVE_INPUT_OUT_OF_MEMORY
                ? PROGRESSIVE_COVER_OUT_OF_MEMORY
                : PROGRESSIVE_COVER_INVALID_INPUT;
        progressive_input_destroy(&normalized);
        return status;
    }

    result->on_rows_unique = normalized.pi.ON_minterms;
    result->off_rows_unique = normalized.pi.OFF_minterms;

    if (normalized.pi.ON_minterms == 0) {
        result->stop_reason = PROGRESSIVE_STOP_CONSTANT_ZERO;
        result->max_horizon = options->max_horizon;
        result->stable_horizons = options->stable_horizons;
        result->memory_budget_bytes = options->memory_budget_bytes;
        result->incumbent_valid = true;
        result->incumbent_source = PROGRESSIVE_INCUMBENT_CONSTANT;
        progressive_input_destroy(&normalized);
        return PROGRESSIVE_COVER_OK;
    }
    if (normalized.pi.OFF_minterms == 0) {
        result->solution = (int *)calloc((size_t)ninputs, sizeof(int));
        if (!result->solution) {
            progressive_input_destroy(&normalized);
            return PROGRESSIVE_COVER_OUT_OF_MEMORY;
        }
        result->cardinality = 1;
        result->selected_horizon = 0;
        result->final_horizon = 0;
        result->max_horizon = options->max_horizon;
        result->stable_horizons = options->stable_horizons;
        result->memory_budget_bytes = options->memory_budget_bytes;
        result->stop_reason = PROGRESSIVE_STOP_CONSTANT_ONE;
        result->incumbent_valid = true;
        result->incumbent_source = PROGRESSIVE_INCUMBENT_CONSTANT;
        result->incumbent_generation = 1;
        progressive_input_destroy(&normalized);
        return PROGRESSIVE_COVER_OK;
    }

    ProgressiveCoverResult core_result;
    ProgressiveCoverStatus status = progressive_cover_run_core(
        &normalized.pi,
        ninputs,
        options,

        &core_result
    );
    const bool primary_resource_limited =
        status == PROGRESSIVE_COVER_RESOURCE_LIMIT;
    ProgressiveCoverResult total_work;
    memset(&total_work, 0, sizeof(total_work));
    progressive_cover_work_accumulate(&total_work, &core_result);
    int secondary_capacities[2] = { 0, 0 };
    int secondary_attempt_limit = 0;
    bool primary_usable =
        status == PROGRESSIVE_COVER_OK ||
        (
            status == PROGRESSIVE_COVER_RESOURCE_LIMIT &&
            core_result.incumbent_valid
        );
    bool restricted_bound_leaves_cardinality_room =
        isfinite(core_result.lp_objective) &&
        ceil(
            core_result.lp_objective - PROGRESSIVE_VALUE_TOL
        ) < (double)core_result.cardinality;
    bool depth_one_affordable =
        progressive_greedy_price_candidate_capacity(
            normalized.pi.ON_minterms,
            ninputs,
            1,
            &secondary_capacities[0]
        ) &&
        secondary_capacities[0] <= options->greedy_candidate_budget;
    if (depth_one_affordable) {
        secondary_attempt_limit = 1;
        bool depth_two_fits =
            progressive_greedy_price_candidate_capacity(
                normalized.pi.ON_minterms,
                ninputs,
                2,
                &secondary_capacities[1]
            ) &&
            secondary_capacities[1] <=
                options->greedy_candidate_budget -
                    secondary_capacities[0];
        if (depth_two_fits) secondary_attempt_limit = 2;
    }
    bool run_secondary =
        primary_usable &&
        restricted_bound_leaves_cardinality_room &&
        options->master_backend == PROGRESSIVE_MASTER_NATIVE &&
        options->greedy_multi_start &&
        options->greedy_anchor_percent < 100 &&
        options->greedy_forced_prefix_depth < 0 &&
        !options->column_pool_input_path &&
        !options->column_pool_output_path &&
        secondary_attempt_limit > 0;
    if (run_secondary) {
        int primary_cardinality = core_result.cardinality;
        int best_secondary_cardinality = 0;
        int completed_attempts = 0;
        int improvements = 0;
        bool depth_two_allowed = true;
        uint64_t secondary_nanoseconds_total = 0;
        double combined_elapsed = core_result.elapsed_seconds;

        for (
            int attempt = 0;
            attempt < secondary_attempt_limit;
            ++attempt
        ) {
            if (
                attempt > 0 &&
                (improvements > 0 || !depth_two_allowed)
            ) {
                break;
            }
            int secondary_prefix_depth = attempt + 1;
            ProgressiveCoverOptions secondary_options = *options;
            secondary_options.greedy_anchor_percent = 100;
            secondary_options.greedy_forced_prefix_depth =
                secondary_prefix_depth;
            secondary_options.greedy_multi_start = false;
            secondary_options.greedy_single_pricing_batch = true;
            secondary_options.time_limit_seconds = 0.0;
            secondary_options.native_feasibility_jump_seconds = 0.0;
            secondary_options.native_feasibility_jump_work_budget = false;
            secondary_options.column_pool_input_path = NULL;
            secondary_options.column_pool_output_path = NULL;

            secondary_options.incumbent_callback = NULL;
            secondary_options.incumbent_callback_data = NULL;

            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_NATIVE_MULTISTART phase=start "
                "attempt=%d primary=%d secondary_anchor_percent=100 "
                "secondary_prefix_depth=%d candidate_capacity=%d "
                "pricing_batches=1 "
                "trigger=restricted-bound-room\n",
                attempt + 1,
                primary_cardinality,
                secondary_prefix_depth,
                secondary_capacities[attempt]
            );
            ProgressiveCoverResult secondary_result;
            memset(&secondary_result, 0, sizeof(secondary_result));
            ProgressiveCoverStatus secondary_status =
                progressive_cover_run_core(
                    &normalized.pi,
                    ninputs,
                    &secondary_options,

                    &secondary_result
                );
            progressive_cover_work_accumulate(
                &total_work,
                &secondary_result
            );
            bool secondary_usable =
                secondary_status == PROGRESSIVE_COVER_OK ||
                (
                    secondary_status ==
                        PROGRESSIVE_COVER_RESOURCE_LIMIT &&
                    secondary_result.incumbent_valid
                );
            int secondary_cardinality = secondary_usable
                ? secondary_result.cardinality
                : 0;
            double secondary_relative_room =
                secondary_usable &&
                isfinite(secondary_result.lp_objective) &&
                primary_cardinality > 0
                    ? (
                        (double)primary_cardinality -
                        secondary_result.lp_objective
                    ) / (double)primary_cardinality
                    : NAN;
            if (
                secondary_usable &&
                (
                    best_secondary_cardinality == 0 ||
                    secondary_cardinality < best_secondary_cardinality
                )
            ) {
                best_secondary_cardinality = secondary_cardinality;
            }
            uint64_t secondary_nanoseconds =
                secondary_result.elapsed_seconds >=
                        (double)UINT64_MAX / 1e9
                    ? UINT64_MAX
                    : (uint64_t)(
                        secondary_result.elapsed_seconds * 1e9
                    );
            if (
                UINT64_MAX - secondary_nanoseconds_total <
                    secondary_nanoseconds
            ) {
                secondary_nanoseconds_total = UINT64_MAX;
            } else {
                secondary_nanoseconds_total += secondary_nanoseconds;
            }
            combined_elapsed += secondary_result.elapsed_seconds;
            completed_attempts++;
            bool secondary_cardinality_improved =
                secondary_usable &&
                secondary_result.cardinality < core_result.cardinality;
            bool secondary_literal_improved =
                secondary_usable &&
                secondary_result.cardinality == core_result.cardinality &&
                progressive_solution_literal_count(
                    ninputs,
                    secondary_result.cardinality,
                    secondary_result.solution
                ) < progressive_solution_literal_count(
                    ninputs,
                    core_result.cardinality,
                    core_result.solution
                );
            bool secondary_improved =
                secondary_cardinality_improved ||
                secondary_literal_improved;
            if (secondary_improved) {
                improvements += secondary_cardinality_improved ? 1 : 0;
                progressive_cover_result_destroy(&core_result);
                core_result = secondary_result;
                memset(
                    &secondary_result,
                    0,
                    sizeof(secondary_result)
                );
                if (
                    options->incumbent_callback &&
                    !options->incumbent_callback(
                        ninputs,
                        core_result.cardinality,
                        core_result.solution,
                        core_result.selected_horizon,
                        core_result.incumbent_generation,
                        options->incumbent_callback_data
                    )
                ) {
                    status = PROGRESSIVE_COVER_PERSISTENCE_ERROR;
                } else {
                    status = secondary_status;
                }
            }
            progressive_cover_result_destroy(&secondary_result);
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_NATIVE_MULTISTART phase=finish "
                "attempt=%d primary=%d secondary=%d selected=%d "
                "improved=%s secondary_prefix_depth=%d "
                "literal_improved=%s secondary_status=%s seconds=%.6f "
                "relative_bound_room=%.9f\n",
                attempt + 1,
                primary_cardinality,
                secondary_cardinality,
                core_result.cardinality,
                secondary_improved ? "yes" : "no",
                secondary_prefix_depth,
                secondary_literal_improved ? "yes" : "no",
                progressive_cover_status_name(secondary_status),
                (double)secondary_nanoseconds / 1e9,
                secondary_relative_room
            );
            if (
                attempt == 0 &&
                secondary_attempt_limit > 1 &&
                !secondary_improved
            ) {
                double minimum_room = options->
                    greedy_multistart_depth2_min_relative_room;
                depth_two_allowed =
                    minimum_room <= 0.0 ||
                    (
                        isfinite(secondary_relative_room) &&
                        secondary_relative_room +
                            PROGRESSIVE_VALUE_TOL >= minimum_room
                    );
                fprintf(
                    stderr,
                    "CCUBES_PROGRESSIVE_NATIVE_MULTISTART "
                    "phase=depth2-decision primary=%d "
                    "relative_bound_room=%.9f minimum=%.9f "
                    "escalate=%s\n",
                    primary_cardinality,
                    secondary_relative_room,
                    minimum_room,
                    depth_two_allowed ? "yes" : "no"
                );
            }
            if (status == PROGRESSIVE_COVER_PERSISTENCE_ERROR) break;
        }
        core_result.elapsed_seconds = combined_elapsed;
        core_result.greedy_multistart_attempts = completed_attempts;
        core_result.greedy_multistart_improvements = improvements;
        core_result.greedy_secondary_cardinality =
            best_secondary_cardinality;
        core_result.greedy_multistart_elapsed_nanoseconds =
            secondary_nanoseconds_total;
    }
    progressive_cover_work_apply(&core_result, &total_work);
    if (
        primary_resource_limited &&
        status != PROGRESSIVE_COVER_PERSISTENCE_ERROR
    ) {
        status = PROGRESSIVE_COVER_RESOURCE_LIMIT;
        core_result.resource_limited = true;
        core_result.stop_reason = PROGRESSIVE_STOP_RESOURCE_LIMIT;
    }
    core_result.on_rows_original = normalized.on_original;
    core_result.off_rows_original = normalized.off_original;
    core_result.on_rows_unique = normalized.pi.ON_minterms;
    core_result.off_rows_unique = normalized.pi.OFF_minterms;
    core_result.on_duplicates = normalized.on_duplicates;
    core_result.off_duplicates = normalized.off_duplicates;
    core_result.conflicts = normalized.conflicts;
    core_result.normalization_nanoseconds = normalization_nanoseconds;
    *result = core_result;
    progressive_input_destroy(&normalized);
    return status;
}

ProgressiveCoverStatus progressive_cover_run(
    const PIstorage *pi,
    int ninputs,
    int max_horizon,
    int stable_horizons,
    ProgressiveCoverResult *result
) {
    ProgressiveCoverOptions options;
    progressive_cover_options_default(
        &options,
        ninputs,
        max_horizon,
        stable_horizons
    );
    return progressive_cover_run_with_options(
        pi,
        ninputs,
        &options,
        result
    );
}

ProgressiveCoverStatus progressive_cover_run_multi(
    const PIstorage *outputs,
    int ninputs,
    int noutputs,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *results
) {
    if (
        !outputs ||
        !results ||
        !options ||
        ninputs <= 0 ||
        noutputs <= 0
    ) {
        return PROGRESSIVE_COVER_INVALID_INPUT;
    }
    memset(
        results,
        0,
        (size_t)noutputs * sizeof(ProgressiveCoverResult)
    );
    ProgressiveCoverStatus overall_status = PROGRESSIVE_COVER_OK;
    for (int output = 0; output < noutputs; ++output) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_OUTPUT_START output=%d outputs=%d\n",
            output + 1,
            noutputs
        );
        ProgressiveCoverStatus status = progressive_cover_run_with_options(
            &outputs[output],
            ninputs,
            options,
            &results[output]
        );
        if (options->profile_pricing) {
            const WeightedMMCSPriceTelemetry *profile =
                &results[output].pricing_telemetry;
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_PROFILE output=%d calls=%d "
                "anchors=%llu constructed=%llu would_gate=%llu "
                "edges=%llu would_avoid_edges=%llu xor_words=%llu "
                "incidences=%llu would_avoid_incidences=%llu "
                "pivots=%llu pivot_edge_scans=%llu "
                "literal_incidence_visits=%llu "
                "edge_literal_visits=%llu bound_calls=%llu "
                "bound_rows=%llu edge_worker_seconds=%.6f "
                "incidence_count_worker_seconds=%.6f "
                "incidence_fill_worker_seconds=%.6f "
                "search_worker_seconds=%.6f "
                "candidate_worker_seconds=%.6f merge_seconds=%.6f "
                "oracle_wall_seconds=%.6f "
                "normalization_seconds=%.6f setup_seconds=%.6f "
                "seed_call_seconds=%.6f seed_internal_validation_seconds=%.6f "
                "seed_publish_seconds=%.6f validation_seconds=%.6f "
                "persistence_seconds=%.6f master_setup_seconds=%.6f "
                "master_seconds=%.6f integer_seconds=%.6f\n",
                output + 1,
                results[output].oracle_calls,
                (unsigned long long)profile->anchors_presented,
                (unsigned long long)profile->anchors_constructed,
                (unsigned long long)profile->prospective_gated_anchors,
                (unsigned long long)profile->edges_constructed,
                (unsigned long long)profile->prospective_gated_edges,
                (unsigned long long)profile->xor_words,
                (unsigned long long)profile->incidences_counted,
                (unsigned long long)
                    profile->prospective_gated_incidences,
                (unsigned long long)profile->pivot_selections,
                (unsigned long long)
                    profile->pivot_edge_entries_inspected,
                (unsigned long long)
                    profile->literal_incidence_entries_visited,
                (unsigned long long)
                    profile->edge_literal_entries_visited,
                (unsigned long long)profile->compatible_bound_calls,
                (unsigned long long)
                    profile->compatible_bound_rows_examined,
                (double)profile->edge_build_nanoseconds / 1e9,
                (double)profile->incidence_count_nanoseconds / 1e9,
                (double)profile->incidence_fill_nanoseconds / 1e9,
                (double)profile->search_nanoseconds / 1e9,
                (double)profile->candidate_collection_nanoseconds / 1e9,
                (double)profile->merge_nanoseconds / 1e9,
                (double)profile->pricing_call_nanoseconds / 1e9,
                (double)results[output].normalization_nanoseconds / 1e9,
                (double)results[output].setup_nanoseconds / 1e9,
                (double)results[output].seed_call_nanoseconds / 1e9,
                (double)results[output].seed_validation_nanoseconds / 1e9,
                (double)results[output].seed_publish_nanoseconds / 1e9,
                (double)
                    results[output].incumbent_validation_nanoseconds /
                    1e9,
                (double)
                    results[output].incumbent_persistence_nanoseconds /
                    1e9,
                (double)results[output].master_setup_nanoseconds / 1e9,
                (double)results[output].master_lp_nanoseconds / 1e9,
                (double)results[output].integer_solve_nanoseconds / 1e9
            );
        }
        bool usable_limited =
            status == PROGRESSIVE_COVER_RESOURCE_LIMIT &&
            results[output].incumbent_valid;
        if (status != PROGRESSIVE_COVER_OK && !usable_limited) {
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_OUTPUT output=%d status=%s "
                "rows_on=%d/%d rows_off=%d/%d duplicates_on=%d "
                "duplicates_off=%d conflicts=%d final_horizon=%d "
                "batch_capacity=%d batch_bytes=%llu pricing_threads=%d "
                "worker_bytes=%llu peak_incidence=%llu columns=%d "
                "rounds=%d oracle_calls=%d oracle_nodes=%llu "
                "search_seconds=%.6f\n",
                output + 1,
                progressive_cover_status_name(status),
                results[output].on_rows_unique,
                results[output].on_rows_original,
                results[output].off_rows_unique,
                results[output].off_rows_original,
                results[output].on_duplicates,
                results[output].off_duplicates,
                results[output].conflicts,
                results[output].final_horizon,
                results[output].batch_capacity,
                (unsigned long long)
                    results[output].batch_allocated_bytes,
                results[output].pricing_threads,
                (unsigned long long)
                    results[output].estimated_bytes_per_worker,
                (unsigned long long)
                    results[output].peak_incidence_entries,
                results[output].retained_columns,
                results[output].pricing_rounds,
                results[output].oracle_calls,
                (unsigned long long)
                    results[output].oracle_search_nodes,
                results[output].elapsed_seconds
            );
            for (int done = 0; done <= output; ++done) {
                progressive_cover_result_destroy(&results[done]);
            }
            return status;
        }
        if (usable_limited) {
            overall_status = PROGRESSIVE_COVER_RESOURCE_LIMIT;
        }
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_OUTPUT_DONE output=%d status=%s "
            "cover=%d stop=%s incumbent_generation=%d "
            "pricing_closed=%s closed_horizon=%d search_seconds=%.6f\n",
            output + 1,
            usable_limited ? "resource-limit" : "ok",
            results[output].cardinality,
            progressive_cover_stop_reason_name(
                results[output].stop_reason
            ),
            results[output].incumbent_generation,
            results[output].pricing_closed ? "yes" : "no",
            results[output].pricing_closed_horizon,
            results[output].elapsed_seconds
        );
    }
    return overall_status;
}

void progressive_cover_result_destroy(ProgressiveCoverResult *result) {
    if (!result) return;
    free(result->solution);
    memset(result, 0, sizeof(*result));
}

const char *progressive_cover_status_name(ProgressiveCoverStatus status) {
    switch (status) {
        case PROGRESSIVE_COVER_OK:
            return "ok";
        case PROGRESSIVE_COVER_RESOURCE_LIMIT:
            return "resource-limit";
        case PROGRESSIVE_COVER_UNAVAILABLE:
            return "unavailable";
        case PROGRESSIVE_COVER_INVALID_INPUT:
            return "invalid-input";
        case PROGRESSIVE_COVER_INFEASIBLE:
            return "infeasible";
        case PROGRESSIVE_COVER_SOLVER_ERROR:
            return "solver-error";
        case PROGRESSIVE_COVER_PERSISTENCE_ERROR:
            return "persistence-error";
        case PROGRESSIVE_COVER_OUT_OF_MEMORY:
            return "out-of-memory";
    }
    return "unknown";
}

const char *progressive_cover_stop_reason_name(
    ProgressiveCoverStopReason reason
) {
    switch (reason) {
        case PROGRESSIVE_STOP_STABLE:
            return "stable-horizon";
        case PROGRESSIVE_STOP_MAX_HORIZON:
            return "max-horizon";
        case PROGRESSIVE_STOP_BATCH_COMPLETE:
            return "batch-complete";
        case PROGRESSIVE_STOP_CARDINALITY_BOUND:
            return "cardinality-bound";
        case PROGRESSIVE_STOP_RESOURCE_LIMIT:
            return "resource-limit";
        case PROGRESSIVE_STOP_CONSTANT_ZERO:
            return "constant-zero";
        case PROGRESSIVE_STOP_CONSTANT_ONE:
            return "constant-one";
    }
    return "unknown";
}

const char *progressive_cover_incumbent_source_name(
    ProgressiveCoverIncumbentSource source
) {
    switch (source) {
        case PROGRESSIVE_INCUMBENT_NONE:
            return "none";
        case PROGRESSIVE_INCUMBENT_SEED:
            return "seed";
        case PROGRESSIVE_INCUMBENT_COLUMN_POOL:
            return "column-pool";
        case PROGRESSIVE_INCUMBENT_PRICED_MASTER:
            return "priced-master";
        case PROGRESSIVE_INCUMBENT_CONSTANT:
            return "constant";
    }
    return "unknown";
}
