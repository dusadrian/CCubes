/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_native_cover.h"
#include "runtime_options.h"

#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "progressive_column_pool.h"
#include "progressive_dual_calibration.h"
#include "progressive_global_bound.h"
#include "progressive_greedy_price.h"
#include "progressive_literal_polish.h"
#include "progressive_rwls.h"
#include "progressive_native_master.h"
#include "progressive_native_phase_policy.h"
#include "progressive_native_reduce_expand.h"
#include "progressive_native_repair.h"
#include "progressive_seed.h"
#include "progressive_validation.h"

#define PROGRESSIVE_NATIVE_RC_TOL 1e-8
#define PROGRESSIVE_NATIVE_LITERAL_POLISH_NODES 200000
#define PROGRESSIVE_GLOBAL_DUAL_ITERATIONS 256
#define PROGRESSIVE_GLOBAL_DUAL_WITNESS_BATCH 32
#define PROGRESSIVE_GLOBAL_DUAL_MAX_CONSTRAINTS 256
/*
 * Discovery rounds receive a share of the allocated bound budget rather than a
 * fixed number of seconds. A constant cap does not scale: larger instances need
 * proportionally more discovery time, and when certification does not fit
 * inside the cap the round reports an uncertified anchor and the conservative
 * ceiling collapses the bound. The old constant is retained only as a floor so
 * that very small budgets behave as before.
 */
#define PROGRESSIVE_GLOBAL_DUAL_DISCOVERY_SHARE 0.10
#define PROGRESSIVE_GLOBAL_DUAL_DISCOVERY_MIN_SECONDS 5.0

typedef struct {
    bool has_failed_state;
    int failed_master_columns;
    int failed_cardinality;
    uint64_t failed_incumbent_signature;
    int skipped_optimal;
    int skipped_unchanged;
} NativeFeasibilityJumpGate;

typedef struct {
    double grace_seconds;
    double requested_seconds;
    double recent_round_seconds[3];
    int recent_round_count;
    int recent_round_next;
    int observed_cardinality;
    uint64_t pending_improvement_nanoseconds;
    uint64_t base_deadline;
} NativeSoftTarget;

static uint64_t native_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static bool native_deadline_reached(uint64_t deadline) {
    return deadline > 0 && native_monotonic_nanoseconds() >= deadline;
}

static uint64_t native_deadline_after_seconds(
    uint64_t started,
    double seconds
) {
    if (seconds <= 0.0) return 0;
    const double span = seconds * 1e9;
    return span >= (double)(UINT64_MAX - started)
        ? UINT64_MAX
        : started + (uint64_t)span;
}

static uint64_t native_repair_time_used(
    const ProgressiveCoverResult *result
) {
    const uint64_t prepare = result->native_repair_prepare_nanoseconds;
    const uint64_t solve = result->native_repair_solve_nanoseconds;
    return UINT64_MAX - prepare < solve
        ? UINT64_MAX
        : prepare + solve;
}

static uint64_t native_repair_attempt_deadline(
    const ProgressiveCoverOptions *options,
    const ProgressiveCoverResult *result,
    uint64_t base_deadline
) {
    const uint64_t used = native_repair_time_used(result);
    uint64_t budget = result->native_repair_budget_nanoseconds;
    if (options->time_limit_seconds <= 0.0) {
        /*
         * Without a wall target the 5%-of-budget allowance has no denominator.
         * Returning "no deadline" here made speculative repair unbounded, which
         * is the default configuration a user gets by not setting a time limit:
         * measured on Fiser output 096 it produced the same 60-cube cover in
         * 345s instead of 11s, and on output 058 a worse cover. Scale the same
         * share against elapsed search time instead, so one rule covers both
         * cases and the unbudgeted path stays bounded.
         */
        const uint64_t begun = result->native_started_nanoseconds;
        if (begun == 0) return 0;
        const uint64_t now_elapsed = native_monotonic_nanoseconds();
        const uint64_t elapsed = now_elapsed > begun ? now_elapsed - begun : 0;
        const double share = options->native_repair_time_share > 0.0
            ? options->native_repair_time_share
            : 0.05;
        const double scaled = (double)elapsed * share;
        budget = scaled >= (double)UINT64_MAX
            ? UINT64_MAX
            : (uint64_t)scaled;
    }
    if (used >= budget) return 1;
    const uint64_t now = native_monotonic_nanoseconds();
    const uint64_t remaining = budget - used;
    uint64_t repair_deadline = UINT64_MAX - now < remaining
        ? UINT64_MAX
        : now + remaining;
    if (base_deadline > 0 && base_deadline < repair_deadline) {
        repair_deadline = base_deadline;
    }
    return repair_deadline;
}

static bool native_repair_budget_available(
    const ProgressiveCoverOptions *options,
    const ProgressiveCoverResult *result,
    uint64_t base_deadline
) {
    const uint64_t repair_deadline = native_repair_attempt_deadline(
        options,
        result,
        base_deadline
    );
    return repair_deadline == 0 || !native_deadline_reached(repair_deadline);
}

static double native_soft_target_round_seconds(
    const NativeSoftTarget *target
) {
    if (!target || target->recent_round_count < 3) return 0.0;
    double left = target->recent_round_seconds[0];
    double middle = target->recent_round_seconds[1];
    double right = target->recent_round_seconds[2];
    if (left > middle) {
        const double swap = left;
        left = middle;
        middle = swap;
    }
    if (middle > right) {
        const double swap = middle;
        middle = right;
        right = swap;
    }
    if (left > middle) middle = left;
    return middle;
}

static void observe_native_completed_round(
    NativeSoftTarget *target,
    ProgressiveCoverResult *result,
    uint64_t elapsed_nanoseconds
) {
    if (
        !target ||
        !result ||
        target->requested_seconds <= 0.0 ||
        elapsed_nanoseconds == 0
    ) {
        return;
    }
    target->recent_round_seconds[target->recent_round_next] =
        (double)elapsed_nanoseconds / 1e9;
    target->recent_round_next = (target->recent_round_next + 1) % 3;
    if (target->recent_round_count < 3) target->recent_round_count++;
    const double representative =
        native_soft_target_round_seconds(target);
    if (representative <= 0.0) return;
    target->grace_seconds =
        progressive_native_soft_target_grace_seconds(
            target->requested_seconds,
            representative
        );
    result->native_soft_target_grace_seconds =
        target->grace_seconds;
}

static bool observe_native_soft_target(
    NativeSoftTarget *target,
    ProgressiveCoverResult *result,
    uint64_t *deadline,
    const char *source
) {
    if (
        !target ||
        !result ||
        !deadline ||
        result->cardinality <= 0
    ) {
        return false;
    }
    if (result->cardinality < target->observed_cardinality) {
        target->observed_cardinality = result->cardinality;
        target->pending_improvement_nanoseconds =
            native_monotonic_nanoseconds();
    }
    if (
        target->grace_seconds <= 0.0 ||
        target->pending_improvement_nanoseconds == 0
    ) {
        return false;
    }
    const uint64_t improved_at =
        target->pending_improvement_nanoseconds;
    target->pending_improvement_nanoseconds = 0;
    const uint64_t previous_deadline = *deadline;
    const uint64_t renewed_deadline =
        progressive_native_soft_target_renew_deadline(
            previous_deadline,
            improved_at,
            target->grace_seconds
        );
    if (renewed_deadline <= previous_deadline) return false;

    const uint64_t extension = renewed_deadline - previous_deadline;
    const bool first_base_extension =
        result->native_soft_target_extensions == 0 &&
        target->base_deadline > 0 &&
        previous_deadline == target->base_deadline;
    *deadline = renewed_deadline;
    result->native_soft_target_extensions++;
    if (first_base_extension) {
        result->native_soft_target_descending_at_base = true;
        result->native_soft_target_base_cardinality =
            result->cardinality;
    }
    if (
        UINT64_MAX - result->native_soft_target_extension_nanoseconds <
        extension
    ) {
        result->native_soft_target_extension_nanoseconds = UINT64_MAX;
    } else {
        result->native_soft_target_extension_nanoseconds += extension;
    }
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_TIME_EXTEND "
        "source=%s cover=%d base_target=%s grace_seconds=%.3f "
        "added_seconds=%.6f total_seconds=%.6f extensions=%d\n",
        source ? source : "unknown",
        result->cardinality,
        first_base_extension ? "yes" : "no",
        target->grace_seconds,
        (double)extension / 1e9,
        (double)result->native_soft_target_extension_nanoseconds / 1e9,
        result->native_soft_target_extensions
    );
    return true;
}

static bool native_selected_uses_new_column(
    const int *selected,
    int selected_cardinality,
    int first_new_column
) {
    if (!selected || selected_cardinality <= 0) return false;
    for (int index = 0; index < selected_cardinality; ++index) {
        if (selected[index] >= first_new_column) return true;
    }
    return false;
}

static const char *native_repair_lagrangian_status(
    const ProgressiveNativeRepairStats *stats
) {
    if (!stats || !stats->lagrangian_attempted) return "not-run";
    return stats->lagrangian_completed
        ? "complete"
        : "incomplete";
}

static int compare_ints(const void *left, const void *right) {
    const int a = *(const int *)left;
    const int b = *(const int *)right;
    return (a > b) - (a < b);
}

static int compare_uint64s(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static uint64_t native_solution_signature(
    int ninputs,
    int cardinality,
    const int *solution
) {
    if (ninputs <= 0 || cardinality <= 0 || !solution) return 0;
    uint64_t *cube_hashes = (uint64_t *)malloc(
        (size_t)cardinality * sizeof(uint64_t)
    );
    if (!cube_hashes) return 0;
    for (int cube = 0; cube < cardinality; ++cube) {
        uint64_t hash = UINT64_C(1469598103934665603);
        for (int input = 0; input < ninputs; ++input) {
            hash ^= (uint64_t)(unsigned int)(
                solution[
                    (size_t)input * (size_t)cardinality +
                    (size_t)cube
                ] + 1
            );
            hash *= UINT64_C(1099511628211);
        }
        cube_hashes[cube] = hash;
    }
    qsort(
        cube_hashes,
        (size_t)cardinality,
        sizeof(uint64_t),
        compare_uint64s
    );
    uint64_t signature = UINT64_C(1469598103934665603);
    for (int cube = 0; cube < cardinality; ++cube) {
        signature ^= cube_hashes[cube];
        signature *= UINT64_C(1099511628211);
    }
    free(cube_hashes);
    return signature;
}

static int native_solution_literal_count(
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

static int native_selected_literal_count(
    const ProgressiveNativeMaster *master,
    int cardinality,
    const int *selected
) {
    if (!master || cardinality <= 0 || !selected) return INT_MAX;
    int literals = 0;
    for (int cube = 0; cube < cardinality; ++cube) {
        int column = selected[cube];
        if (column < 0 || column >= master->count) return INT_MAX;
        const int *pattern = &master->patterns[
            (size_t)column * (size_t)master->ninputs
        ];
        for (int input = 0; input < master->ninputs; ++input) {
            literals += pattern[input] != 0;
        }
    }
    return literals;
}

static bool record_seed_supports(
    const ProgressiveSeedResult *seed,
    ProgressiveCoverResult *result
) {
    int *supports = (int *)malloc(
        (size_t)seed->cardinality * sizeof(int)
    );
    if (!supports) return false;
    for (int cube = 0; cube < seed->cardinality; ++cube) {
        int support = 0;
        for (int input = 0; input < seed->ninputs; ++input) {
            support += seed->patterns[
                (size_t)cube * (size_t)seed->ninputs +
                (size_t)input
            ] != 0;
        }
        supports[cube] = support;
    }
    qsort(
        supports,
        (size_t)seed->cardinality,
        sizeof(int),
        compare_ints
    );
    result->seed_support_minimum = supports[0];
    result->seed_support_maximum = supports[seed->cardinality - 1];
    result->seed_support_median =
        (seed->cardinality & 1) != 0
            ? supports[seed->cardinality / 2]
            : (
                (double)supports[seed->cardinality / 2 - 1] +
                (double)supports[seed->cardinality / 2]
            ) / 2.0;
    free(supports);
    return true;
}

static int *transpose_patterns(
    const int *patterns,
    int cardinality,
    int ninputs
) {
    int *solution = (int *)malloc(
        (size_t)cardinality *
        (size_t)ninputs *
        sizeof(int)
    );
    if (!solution) return NULL;
    for (int cube = 0; cube < cardinality; ++cube) {
        for (int input = 0; input < ninputs; ++input) {
            solution[
                (size_t)input * (size_t)cardinality +
                (size_t)cube
            ] = patterns[
                (size_t)cube * (size_t)ninputs +
                (size_t)input
            ];
        }
    }
    return solution;
}

static int *selected_patterns_to_solution(
    const ProgressiveNativeMaster *master,
    const int *selected,
    int cardinality
) {
    int *solution = (int *)malloc(
        (size_t)cardinality *
        (size_t)master->ninputs *
        sizeof(int)
    );
    if (!solution) return NULL;
    for (int cube = 0; cube < cardinality; ++cube) {
        const int column = selected[cube];
        if (column < 0 || column >= master->count) {
            free(solution);
            return NULL;
        }
        const int *pattern = &master->patterns[
            (size_t)column * (size_t)master->ninputs
        ];
        for (int input = 0; input < master->ninputs; ++input) {
            solution[
                (size_t)input * (size_t)cardinality +
                (size_t)cube
            ] = pattern[input];
        }
    }
    return solution;
}

static ProgressiveCoverStatus adopt_incumbent(
    int ninputs,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    int cardinality,
    ProgressiveCoverIncumbentSource source,
    int *candidate
) {
    const uint64_t validation_started = native_monotonic_nanoseconds();
    const bool valid = cardinality > 0 && candidate &&
        progressive_solution_is_valid_prepared(prepared, cardinality, candidate);
    result->incumbent_validation_nanoseconds +=
        native_monotonic_nanoseconds() - validation_started;
    if (
        !valid
    ) {
        free(candidate);
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
    const int generation = result->incumbent_generation + 1;
    const uint64_t persistence_started = native_monotonic_nanoseconds();
    const bool persisted = !options->incumbent_callback ||
        options->incumbent_callback(
            ninputs,
            cardinality,
            candidate,
            0,
            generation,
            options->incumbent_callback_data
        );
    result->incumbent_persistence_nanoseconds +=
        native_monotonic_nanoseconds() - persistence_started;
    if (!persisted) {
        free(candidate);
        return PROGRESSIVE_COVER_PERSISTENCE_ERROR;
    }
    free(result->solution);
    result->solution = candidate;
    result->cardinality = cardinality;
    result->selected_horizon = 0;
    result->incumbent_source = source;
    result->incumbent_generation = generation;
    result->incumbent_valid = true;
    return PROGRESSIVE_COVER_OK;
}

static ProgressiveCoverStatus map_master_status(
    ProgressiveNativeMasterStatus status
) {
    switch (status) {
        case PROGRESSIVE_NATIVE_MASTER_OK:
            return PROGRESSIVE_COVER_OK;
        case PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY:
            return PROGRESSIVE_COVER_OUT_OF_MEMORY;
        case PROGRESSIVE_NATIVE_MASTER_INFEASIBLE:
            return PROGRESSIVE_COVER_INFEASIBLE;
        case PROGRESSIVE_NATIVE_MASTER_INVALID:
        default:
            return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
}

static ProgressiveCoverStatus save_pool(
    const ProgressiveCoverOptions *options,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveNativeMaster *master,
    ProgressiveCoverResult *result
) {
    if (!options->column_pool_output_path) {
        return PROGRESSIVE_COVER_OK;
    }
    const uint64_t started = native_monotonic_nanoseconds();
    ProgressiveColumnPoolStatus status = progressive_column_pool_save(
        options->column_pool_output_path,
        prepared,
        master->words,
        master->count,
        master->patterns,
        master->coverage,
        result->cardinality,
        result->solution
    );
    result->column_pool_save_nanoseconds +=
        native_monotonic_nanoseconds() - started;
    if (status == PROGRESSIVE_COLUMN_POOL_OK) {
        result->column_pool_saved_columns = master->count;
        return PROGRESSIVE_COVER_OK;
    }
    return status == PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY
        ? PROGRESSIVE_COVER_OUT_OF_MEMORY
        : PROGRESSIVE_COVER_PERSISTENCE_ERROR;
}

static ProgressiveCoverStatus save_due_pool_snapshots(
    const ProgressiveCoverOptions *options,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveNativeMaster *master,
    ProgressiveCoverResult *result,
    uint64_t search_started,
    int completed_round,
    int *next_snapshot
) {
    if (
        !options ||
        !prepared ||
        !master ||
        !result ||
        !next_snapshot
    ) {
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
    const uint64_t now = native_monotonic_nanoseconds();
    const double actual_seconds =
        (double)(now - search_started) / 1e9;
    while (
        *next_snapshot < options->column_pool_snapshot_count &&
        actual_seconds >= options->column_pool_snapshots[
            *next_snapshot
        ].elapsed_seconds
    ) {
        const ProgressiveCoverPoolSnapshot *snapshot =
            &options->column_pool_snapshots[*next_snapshot];
        const uint64_t save_started = native_monotonic_nanoseconds();
        ProgressiveColumnPoolStatus pool_status =
            progressive_column_pool_save(
                snapshot->path,
                prepared,
                master->words,
                master->count,
                master->patterns,
                master->coverage,
                result->cardinality,
                result->solution
            );
        const uint64_t save_nanoseconds =
            native_monotonic_nanoseconds() - save_started;
        result->column_pool_save_nanoseconds += save_nanoseconds;
        if (pool_status != PROGRESSIVE_COLUMN_POOL_OK) {
            return pool_status == PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY
                ? PROGRESSIVE_COVER_OUT_OF_MEMORY
                : PROGRESSIVE_COVER_PERSISTENCE_ERROR;
        }
        result->column_pool_saved_columns = master->count;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_POOL_SNAPSHOT index=%d "
            "requested_seconds=%.6f actual_seconds=%.6f path=%s "
            "round=%d columns=%d cover=%d generation=%d "
            "pricing_rounds=%d greedy_rounds=%d anchors=%llu "
            "constructed=%llu improving=%llu duplicates=%llu "
            "literal_evaluations=%llu greedy_seconds=%.6f "
            "dual_calls=%d dual_iterations=%llu dual_seconds=%.6f "
            "repair=%d/%d repair_seconds=%.6f "
            "reduce_expand=%d/%d/%d reduce_expand_seconds=%.6f "
            "rwls_runs=%d rwls_families=%llu rwls_iterations=%llu "
            "rwls_seconds=%.6f phase=%d/%d save_seconds=%.6f\n",
            *next_snapshot + 1,
            snapshot->elapsed_seconds,
            actual_seconds,
            snapshot->path,
            completed_round,
            master->count,
            result->cardinality,
            result->incumbent_generation,
            result->pricing_rounds,
            result->greedy_rounds,
            (unsigned long long)result->greedy_anchors_attempted,
            (unsigned long long)result->greedy_candidates_constructed,
            (unsigned long long)result->greedy_improving_candidates,
            (unsigned long long)result->greedy_duplicate_candidates,
            (unsigned long long)result->greedy_literal_evaluations,
            (double)result->greedy_elapsed_nanoseconds / 1e9,
            result->native_dual_calls,
            (unsigned long long)result->native_dual_iterations,
            (double)result->native_dual_nanoseconds / 1e9,
            result->native_repair_improvements,
            result->native_repair_attempts,
            (double)(
                result->native_repair_prepare_nanoseconds +
                result->native_repair_solve_nanoseconds
            ) / 1e9,
            result->native_reduce_expand_passes,
            result->native_reduce_expand_variants,
            result->native_reduce_expand_inserted_columns,
            (double)result->native_reduce_expand_nanoseconds / 1e9,
            result->native_feasibility_jump_runs,
            (unsigned long long)
                result->native_feasibility_jump_restart_families,
            (unsigned long long)
                result->native_feasibility_jump_iterations,
            (double)result->native_feasibility_jump_nanoseconds / 1e9,
            result->native_adaptive_phase_promotions,
            result->native_adaptive_phase_checkpoints,
            (double)save_nanoseconds / 1e9
        );
        (*next_snapshot)++;
    }
    return PROGRESSIVE_COVER_OK;
}

static bool global_bound_precedes(
    const ProgressiveGlobalBoundResult *candidate,
    const ProgressiveGlobalBoundResult *best
) {
    if (candidate->cardinality_lower_bound != best->cardinality_lower_bound) {
        return candidate->cardinality_lower_bound >
            best->cardinality_lower_bound;
    }
    return candidate->fractional_lower_bound >
        best->fractional_lower_bound;
}

static bool coverage_is_retained_or_extra(
    const ProgressiveNativeMaster *master,
    const uint64_t *extra_coverage,
    int extra_columns,
    const uint64_t *coverage
) {
    const size_t bytes = (size_t)master->words * sizeof(uint64_t);
    for (int column = 0; column < master->count; ++column) {
        if (memcmp(
            master->coverage + (size_t)column * (size_t)master->words,
            coverage,
            bytes
        ) == 0) {
            return true;
        }
    }
    for (int column = 0; column < extra_columns; ++column) {
        if (memcmp(
            extra_coverage + (size_t)column * (size_t)master->words,
            coverage,
            bytes
        ) == 0) {
            return true;
        }
    }
    return false;
}

static void run_native_global_bound(
    const PIstorage *pi,
    int ninputs,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveNativeMaster *master,
    const double *native_multipliers,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result
) {
    result->global_bound_requested = true;
    result->global_bound_status = PROGRESSIVE_GLOBAL_BOUND_PRICING_ERROR;
    const int rows = master->rows;
    const int words = master->words;
    double *weights = (double *)malloc((size_t)rows * sizeof(double));
    double *calibrated = (double *)malloc((size_t)rows * sizeof(double));
    int *best_pattern = (int *)malloc((size_t)ninputs * sizeof(int));
    int *witness_patterns = (int *)malloc(
        (size_t)PROGRESSIVE_GLOBAL_DUAL_WITNESS_BATCH *
        (size_t)ninputs *
        sizeof(int)
    );
    uint64_t *witness_coverage = (uint64_t *)malloc(
        (size_t)words * sizeof(uint64_t)
    );
    uint64_t *extra_coverage = (uint64_t *)calloc(
        (size_t)PROGRESSIVE_GLOBAL_DUAL_MAX_CONSTRAINTS * (size_t)words,
        sizeof(uint64_t)
    );

    if (
        !weights || !calibrated || !best_pattern || !witness_patterns ||
        !witness_coverage || !extra_coverage
    ) {
        result->global_bound_status =
            PROGRESSIVE_GLOBAL_BOUND_OUT_OF_MEMORY;
        goto report;
    }

    const PIChartView chart = {
        .bits = master->coverage,
        .words = words,
        .cov_bits = 64,
        .rows = rows,
        .cols = master->count
    };
    bool native_seed_valid = native_multipliers != NULL;
    if (native_seed_valid) {
        double native_seed_sum = 0.0;
        for (int row = 0; row < rows; ++row) {
            if (
                !isfinite(native_multipliers[row]) ||
                native_multipliers[row] < 0.0
            ) {
                native_seed_valid = false;
                break;
            }
            native_seed_sum += native_multipliers[row];
        }
        native_seed_valid = native_seed_sum > 0.0;
    }
    if (native_seed_valid) {
        memcpy(weights, native_multipliers, (size_t)rows * sizeof(double));
    } else if (!lagrangian_initialize_pricing_multipliers(
        &chart,
        weights,
        rows
    )) {
        goto report;
    }

    const uint64_t started = native_monotonic_nanoseconds();
    const double budget_seconds =
        options->global_bound_time_limit_seconds;

    int extra_columns = 0;
    bool have_bound = false;
    const double discovery_cap = fmax(
        PROGRESSIVE_GLOBAL_DUAL_DISCOVERY_MIN_SECONDS,
        budget_seconds * PROGRESSIVE_GLOBAL_DUAL_DISCOVERY_SHARE
    );
    for (int round = 0; round < options->global_bound_rounds; ++round) {
        const double elapsed =
            (double)(native_monotonic_nanoseconds() - started) / 1e9;
        double remaining = budget_seconds - elapsed;
        if (remaining <= 0.0) break;

        ProgressiveDualCalibrationStats calibration;
        memset(&calibration, 0, sizeof(calibration));
        const bool calibrated_ok = progressive_dual_calibrate(
                &chart,
                weights,
                options->global_bound_active_rows,
                PROGRESSIVE_GLOBAL_DUAL_ITERATIONS,
                extra_coverage,
                extra_columns,
                calibrated,
                &calibration
            );
        if (!calibrated_ok) {
            break;
        }
        memcpy(weights, calibrated, (size_t)rows * sizeof(double));
        for (int row = 0; row < rows; ++row) {
            weights[row] *= calibration.calibrated_restricted_bound;
        }
        remaining = budget_seconds -
            (double)(native_monotonic_nanoseconds() - started) / 1e9;
        if (remaining <= 0.0) break;

        const int constraints_before = extra_columns;
        const int rounds_left = options->global_bound_rounds - round;

        const double equal_slice = remaining / (double)rounds_left;
        const double pricing_slice = rounds_left == 1
            ? remaining
            : fmin(discovery_cap, equal_slice);
        ProgressiveGlobalBoundResult candidate;
        int witness_count = 0;
        ProgressiveGlobalBoundStatus status =
            progressive_global_bound_compute(
                pi,
                ninputs,
                weights,
                options->requested_threads,
                options->memory_budget_bytes *
                    (uint64_t)options->worker_memory_percent / 100u,
                pricing_slice,
                best_pattern,
                PROGRESSIVE_GLOBAL_DUAL_WITNESS_BATCH,
                &witness_count,
                witness_patterns,
                &candidate
            );
        int witnesses_added = 0;
        if (status == PROGRESSIVE_GLOBAL_BOUND_OK) {
            for (
                int witness = 0;
                witness < witness_count &&
                    extra_columns <
                        PROGRESSIVE_GLOBAL_DUAL_MAX_CONSTRAINTS;
                ++witness
            ) {
                const int *pattern = witness_patterns +
                    (size_t)witness * (size_t)ninputs;
                if (!progressive_pattern_coverage_prepared(
                    prepared,
                    pattern,
                    witness_coverage
                )) {
                    continue;
                }
                if (coverage_is_retained_or_extra(
                    master,
                    extra_coverage,
                    extra_columns,
                    witness_coverage
                )) {
                    continue;
                }
                memcpy(
                    extra_coverage +
                        (size_t)extra_columns * (size_t)words,
                    witness_coverage,
                    (size_t)words * sizeof(uint64_t)
                );
                extra_columns++;
                witnesses_added++;
            }
        }
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_GLOBAL_DUAL round=%d/%d "
            "status=%s active_rows=%d constraints=%d+%d "
            "calibration_iterations=%d restricted=%.9f->%.9f "
            "calibration_seconds=%.6f global_fractional=%.9f "
            "global_cardinality=%d max_cube=%.9f upper=%.9f "
            "complete=%s witnesses=%d/%d pricing_seconds=%.6f\n",
            round + 1,
            options->global_bound_rounds,
            progressive_global_bound_status_name(status),
            calibration.active_rows,
            master->count,
            constraints_before,
            calibration.iterations,
            calibration.initial_restricted_bound,
            calibration.calibrated_restricted_bound,
            calibration.elapsed_seconds,
            candidate.fractional_lower_bound,
            candidate.cardinality_lower_bound,
            candidate.max_cube_weight_found,
            candidate.max_cube_weight_upper_bound,
            candidate.pricing_complete ? "yes" : "no",
            witnesses_added,
            witness_count,
            candidate.elapsed_seconds
        );
        if (status != PROGRESSIVE_GLOBAL_BOUND_OK) {
            result->global_bound_status = status;
            break;
        }
        if (!have_bound || global_bound_precedes(
            &candidate,
            &result->global_bound
        )) {
            result->global_bound = candidate;
            result->global_bound_status = status;
            have_bound = true;
        }
        /*
         * No new witnesses means convergence only when the pricing pass
         * completed. A truncated pass finds nothing new because it ran out of
         * time, not because nothing remains; breaking there abandons the run
         * with an uncertified anchor and reports the conservative ceiling
         * instead of a usable bound. Continuing lets a later round -- the last
         * of which receives all remaining budget -- finish certification.
         */
        if (witnesses_added == 0 && candidate.pricing_complete) {
            break;
        }
    }

report:
    {
        const ProgressiveGlobalBoundResult *bound =
            &result->global_bound;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_GLOBAL_BOUND status=%s scope=global "
            "weights=calibrated-sparse-dual "
            "row_weight_sum=%.12g positive_rows=%d "
            "max_cube_found=%.12g max_cube_upper=%.12g "
            "fractional=%.12g cardinality=%d "
            "fixed_weight_ceiling=%.12g/%d "
            "anchors=%d/%d complete=%s deadline=%s "
            "threads=%d nodes=%llu seconds=%.6f\n",
            progressive_global_bound_status_name(
                result->global_bound_status
            ),
            bound->row_weight_sum,
            bound->positive_rows,
            bound->max_cube_weight_found,
            bound->max_cube_weight_upper_bound,
            bound->fractional_lower_bound,
            bound->cardinality_lower_bound,
            bound->fixed_weight_fractional_ceiling,
            bound->fixed_weight_cardinality_ceiling,
            bound->anchors_completed,
            bound->anchors_total,
            bound->pricing_complete ? "yes" : "no",
            bound->deadline_reached ? "yes" : "no",
            bound->pricing_threads,
            (unsigned long long)bound->search_nodes,
            bound->elapsed_seconds
        );
    }
    free(weights);
    free(calibrated);
    free(best_pattern);
    free(witness_patterns);
    free(witness_coverage);
    free(extra_coverage);

}

/*
 * The native master's incidence is append-only and costs are all one.
 * A replacement geometry for an existing coverage changes neither.
 */
static bool native_bound_is_fresh(
    const ProgressiveNativeMaster *master,
    const ProgressiveCoverResult *result
) {
    return result->restricted_bound_columns > 0 &&
        result->restricted_bound_columns == master->count;
}

static void refresh_native_bound(
    const ProgressiveNativeMaster *master,
    const double *multipliers,
    ProgressiveCoverResult *result
) {
    if (native_bound_is_fresh(master, result)) return;

    const int previous_columns = result->restricted_bound_columns;
    const uint64_t started = native_monotonic_nanoseconds();
    result->lp_objective =
        progressive_native_master_evaluate_bound(master, multipliers);
    fprintf(stderr,
        "CCUBES_PROGRESSIVE_NATIVE_BOUND_REFRESH columns=%d previous_columns=%d "
        "restricted_lagrangian=%.9f seconds=%.9f\n",
        master->count, previous_columns, result->lp_objective,
        (double)(native_monotonic_nanoseconds() - started) / 1e9);
    result->restricted_bound_columns =
        isfinite(result->lp_objective) ? master->count : 0;
    result->integer_best_bound = NAN;
    result->integer_mip_gap = NAN;
    result->restricted_master_optimal = false;
}

static ProgressiveCoverStatus solve_native_master_mode(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    int **selected,
    int *selected_cardinality,
    double *multipliers
) {
    int *candidate_indices = (int *)malloc(
        (size_t)master->count * sizeof(int)
    );
    if (!candidate_indices) return PROGRESSIVE_COVER_OUT_OF_MEMORY;

    int cardinality = -1;
    const uint64_t dual_started = native_monotonic_nanoseconds();
    const double *warm_initial_multipliers = NULL;
    if (result->native_dual_calls > 0) {
        /*
         * Warm start is the production default for scaled large-chart
         * masters: the multiplier buffer holds the previous solve's best
         * row prices and the restricted master has only grown since, so
         * seeding the subgradient from them skips the dual-ascent
         * re-derivation and shortens the scaled schedule.
         */

        warm_initial_multipliers = multipliers;
    }
    ProgressiveNativeMasterStatus native_status =
        progressive_native_master_solve_with_polish_incumbent(
            master,
            options->native_effort_level,
            *selected,
            *selected_cardinality,
            NULL,
            0,
            candidate_indices,
            &cardinality,
            multipliers,
            warm_initial_multipliers
        );
    const uint64_t dual_nanoseconds =
        native_monotonic_nanoseconds() - dual_started;
    result->native_dual_calls++;
    result->native_dual_nanoseconds += dual_nanoseconds;
    result->native_dual_last_nanoseconds = dual_nanoseconds;
    if (native_status != PROGRESSIVE_NATIVE_MASTER_OK) {
        free(candidate_indices);
        return map_master_status(native_status);
    }
    const LagrangianStats *stats = lagrangian_last_stats();
    if (stats->iterations > 0) {
        result->native_dual_iterations += (uint64_t)stats->iterations;
    }
    const double nonzero_subgradient_mean =
        stats->subgradient_samples > 0
            ? (double)stats->nonzero_subgradient_rows /
                (double)stats->subgradient_samples
            : 0.0;
    const double nonzero_subgradient_density =
        stats->subgradient_samples > 0 && stats->rows > 0
            ? nonzero_subgradient_mean / (double)stats->rows
            : 0.0;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_DUAL columns=%d "
        "restricted_best=%.9f restricted_last=%.9f "
        "restricted_integer_bound=%d iterations=%d "
        "requested_limit=%d effective_limit=%d limit_reason=%s "
        "warm=%s "
        "schedule=%s convergence=%s reset=%s floor=%s "
        "bound_scope=restricted_master "
        "stop=%s "
        "continuous_improvements=%d last_improvement=%d "
        "requested_step=%.9g/%.9g/%.9g "
        "requested_stabilization=%.9g requested_deflection=%.9g "
        "step=%.9g/%.9g/%.9g contract=%.9g contractions=%d "
        "stabilization=%.9g deflection=%.9g "
        "rc_seconds=%.6f/%d heuristic_seconds=%.6f/%d "
        "primal_ub_cadence=%d "
        "primal_ub_supplied=%d primal_ub_loop_start=%d "
        "primal_ub_loop_final=%d "
        "primal_ub_recurring_calls=%d "
        "primal_ub_improving_calls=%d "
        "primal_ub_improving_below_supplied=%d "
        "primal_ub_first_improvement=%d "
        "primal_ub_last_improvement=%d "
        "primal_ub_longest_gap=%d "
        "bundle_seconds=%.6f/%d subgradient_seconds=%.6f/%d "
        "nonzero_subgradient=%d/%.1f/%d density=%.6f samples=%d "
        "dual_call=%d dual_seconds=%.6f dual_total_seconds=%.6f\n",
        master->count,
        stats->best_zlb,
        stats->last_zlb,
        stats->best_lb,
        stats->iterations,
        stats->requested_iteration_limit,
        stats->iteration_limit,
        lagrangian_iteration_limit_reason_name(
            stats->iteration_limit_reason
        ),
        stats->warm_start ? "yes" : "no",
        lagrangian_multiplier_schedule_name(
            stats->multiplier_schedule
        ),
        lagrangian_multiplier_convergence_name(
            stats->multiplier_convergence
        ),
        stats->reset_on_continuous_improvement
            ? "continuous"
            : "integer",
        stats->stop_at_step_floor ? "stop" : "continue",
        lagrangian_stop_reason_name(stats->stop_reason),
        stats->continuous_improvements,
        stats->last_continuous_improvement_iteration,
        stats->requested_initial_step_coef,
        stats->requested_step_min,
        stats->requested_step_contract,
        stats->requested_stabilization_beta,
        stats->requested_deflection_alpha,
        stats->initial_step_coef,
        stats->step_coef,
        stats->step_min,
        stats->step_contract,
        stats->step_contractions,
        stats->stabilization_beta,
        stats->deflection_alpha,
        (double)stats->reduced_cost_nanoseconds / 1e9,
        stats->reduced_cost_calls,
        (double)stats->heuristic_nanoseconds / 1e9,
        stats->heuristic_calls,
        stats->primal_ub_cadence,
        stats->primal_ub_supplied,
        stats->primal_ub_loop_start,
        stats->primal_ub_loop_final,
        stats->primal_ub_recurring_calls,
        stats->primal_ub_improving_calls,
        stats->primal_ub_improving_calls_below_supplied,
        stats->primal_ub_first_improvement_iteration,
        stats->primal_ub_last_improvement_iteration,
        stats->primal_ub_longest_improvement_gap,
        (double)stats->bundle_nanoseconds / 1e9,
        stats->bundle_calls,
        (double)stats->subgradient_nanoseconds / 1e9,
        stats->subgradient_calls,
        stats->subgradient_samples > 0
            ? stats->nonzero_subgradient_minimum
            : 0,
        nonzero_subgradient_mean,
        stats->nonzero_subgradient_maximum,
        nonzero_subgradient_density,
        stats->subgradient_samples,
        result->native_dual_calls,
        (double)dual_nanoseconds / 1e9,
        (double)result->native_dual_nanoseconds / 1e9
    );
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_POLISH enabled=%s "
        "node_limit=%ld core_cap=%d "
        "initial_ub=%d final_ub=%d initial_target=%d "
        "initial_core=%d search_target=%d search_core=%d "
        "forced=%d skipped_targets=%d searches=%d nodes=%ld "
        "aborted=%d exhausted=%d improvements=%d proved=%s "
        "stop=%s seconds=%.9f\n",
        stats->polish_enabled ? "yes" : "no",
        stats->polish_node_limit,
        stats->polish_core_cap,
        stats->polish_initial_ub,
        stats->polish_final_ub,
        stats->polish_initial_target,
        stats->polish_initial_core_count,
        stats->polish_search_target,
        stats->polish_search_core_count,
        stats->polish_forced_columns,
        stats->polish_skipped_targets,
        stats->polish_searches,
        stats->polish_nodes,
        stats->polish_aborted_searches,
        stats->polish_exhausted_searches,
        stats->polish_improvements,
        stats->polish_proved_optimal ? "yes" : "no",
        lagrangian_polish_stop_reason_name(
            stats->polish_stop_reason
        ),
        stats->polish_seconds
    );
    result->restricted_bound_columns = master->count;
    result->lp_objective = stats->best_zlb;
    result->integer_best_bound =
        stats->best_lb == INT_MIN ? NAN : (double)stats->best_lb;
    result->integer_mip_gap =
        stats->gap >= 0 && cardinality > 0
            ? (double)stats->gap / (double)cardinality
            : NAN;
    result->restricted_master_optimal =
        stats->gap == 0 &&
        stats->stop_reason == LAGR_STOP_OPTIMAL;

    const int candidate_literals = native_selected_literal_count(
        master,
        cardinality,
        candidate_indices
    );
    const int incumbent_literals = native_solution_literal_count(
        master->ninputs,
        result->cardinality,
        result->solution
    );
    const bool literal_improvement =
        cardinality == result->cardinality &&
        candidate_literals < incumbent_literals;
    if (
        cardinality < result->cardinality ||
        literal_improvement
    ) {
        int *candidate = selected_patterns_to_solution(
            master,
            candidate_indices,
            cardinality
        );
        if (!candidate) {
            free(candidate_indices);
            return PROGRESSIVE_COVER_OUT_OF_MEMORY;
        }
        ProgressiveCoverStatus status = adopt_incumbent(
            master->ninputs,
            prepared,
            options,
            result,
            cardinality,
            PROGRESSIVE_INCUMBENT_PRICED_MASTER,
            candidate
        );
        if (status != PROGRESSIVE_COVER_OK) {
            free(candidate_indices);
            return status;
        }
        if (literal_improvement) {
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_NATIVE_LITERAL_IMPROVEMENT "
                "source=master cardinality=%d literals=%d->%d\n",
                cardinality,
                incumbent_literals,
                candidate_literals
            );
        }
    }

    free(*selected);
    *selected = candidate_indices;
    *selected_cardinality = cardinality;
    return PROGRESSIVE_COVER_OK;
}

static ProgressiveCoverStatus solve_native_master(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    int **selected,
    int *selected_cardinality,
    double *multipliers
) {
    return solve_native_master_mode(
        master,
        prepared,
        options,
        result,
        selected,
        selected_cardinality,
        multipliers
    );
}

static ProgressiveCoverStatus generate_native_reduce_expand(
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    int round,
    uint64_t deadline,
    int *inserted_columns
) {
    if (inserted_columns) *inserted_columns = 0;
    if (!inserted_columns) return PROGRESSIVE_COVER_SOLVER_ERROR;

    ProgressiveNativeReduceExpandStats reshape;
    const ProgressiveNativeMasterStatus reshape_status =
        progressive_native_reduce_expand_generate(
            prepared,
            master,
            result->cardinality,
            result->solution,
            result->native_reduce_expand_passes,
            options->native_reduce_expand_variants,
            deadline,
            &reshape
        );
    if (reshape_status != PROGRESSIVE_NATIVE_MASTER_OK) {
        return map_master_status(reshape_status);
    }
    result->native_reduce_expand_passes++;
    result->native_reduce_expand_cubes += reshape.cubes_examined;
    result->native_reduce_expand_reduced_cores += reshape.reduced_cores;
    result->native_reduce_expand_variants += reshape.variants_attempted;
    result->native_reduce_expand_changed_geometries +=
        reshape.changed_geometries;
    result->native_reduce_expand_inserted_columns +=
        reshape.inserted_columns;
    result->native_reduce_expand_duplicate_columns +=
        reshape.duplicate_columns;
    result->native_reduce_expand_literals_added +=
        reshape.literals_added_by_reduce;
    result->native_reduce_expand_nanoseconds +=
        reshape.elapsed_nanoseconds;
    *inserted_columns = reshape.inserted_columns;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_REDUCE_EXPAND "
        "pass=%d round=%d cubes=%d private=%d reduced=%d "
        "variants=%d legal=%d changed=%d inserted=%d "
        "duplicates=%d reduce_literals=%llu columns=%d "
        "deadline=%s seconds=%.6f\n",
        result->native_reduce_expand_passes,
        round,
        reshape.cubes_examined,
        reshape.cubes_with_private_rows,
        reshape.reduced_cores,
        reshape.variants_attempted,
        reshape.legal_candidates,
        reshape.changed_geometries,
        reshape.inserted_columns,
        reshape.duplicate_columns,
        (unsigned long long)reshape.literals_added_by_reduce,
        master->count,
        reshape.deadline_reached ? "yes" : "no",
        (double)reshape.elapsed_nanoseconds / 1e9
    );
    return PROGRESSIVE_COVER_OK;
}

enum {
    NATIVE_PRIVATE_MICRO_PRICE_BATCH = 256,
    NATIVE_PRIVATE_MICRO_PRICE_DEFAULT_INSERTIONS = 32,
    NATIVE_PRIVATE_MICRO_PRICE_INSERTIONS = 64,
    NATIVE_PRIVATE_MICRO_MATURE_ROUND = 32
};

static uint64_t native_private_micro_price_deadline(
    const ProgressiveCoverOptions *options,
    uint64_t global_deadline
) {
    double seconds = options->time_limit_seconds > 0.0
        ? options->time_limit_seconds * 0.01
        : 0.25;
    if (seconds < 0.05) seconds = 0.05;
    if (seconds > 1.0) seconds = 1.0;
    const uint64_t now = native_monotonic_nanoseconds();

    const long double delta = (long double)seconds * 1.0e9L;
    uint64_t deadline = delta >= (long double)(UINT64_MAX - now)
        ? UINT64_MAX
        : now + (uint64_t)delta;
    if (global_deadline > 0 && deadline > global_deadline) {
        deadline = global_deadline;
    }
    return deadline;
}

static ProgressiveCoverStatus generate_native_private_micro_price(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    const ProgressiveCoverOptions *options,
    const ProgressiveCoverResult *result,
    uint64_t global_deadline,
    int remaining_columns,
    WeightedMMCSPriceBatch *workspace,
    int *inserted_columns,
    double *best_score
) {
    if (
        !pi || !prepared || !master || !options || !result || !workspace ||
        !inserted_columns || !best_score || result->cardinality <= 0 ||
        !result->solution
    ) return PROGRESSIVE_COVER_SOLVER_ERROR;
    *inserted_columns = 0;
    *best_score = 0.0;
    const uint64_t started = native_monotonic_nanoseconds();
    const int rows = prepared->on_rows;
    const int words = prepared->on_column_words;
    const int ninputs = prepared->ninputs;
    const int cardinality = result->cardinality;

    int insertion_limit = NATIVE_PRIVATE_MICRO_PRICE_DEFAULT_INSERTIONS;
    if (insertion_limit > remaining_columns) insertion_limit = remaining_columns;
    int *cover_counts = (int *)calloc((size_t)rows, sizeof(int));
    int *owners = (int *)malloc((size_t)rows * sizeof(int));
    int *owner_private = (int *)calloc(
        (size_t)cardinality, sizeof(int)
    );
    int *pattern = (int *)malloc((size_t)ninputs * sizeof(int));
    int *best_vertices = (int *)malloc((size_t)ninputs * sizeof(int));
    uint64_t *coverage = (uint64_t *)malloc(
        (size_t)words * sizeof(uint64_t)
    );
    double *weights = (double *)calloc((size_t)rows, sizeof(double));
    if (
        !cover_counts || !owners || !owner_private || !pattern ||
        !best_vertices || !coverage || !weights
    ) {
        free(cover_counts); free(owners); free(owner_private); free(pattern);
        free(best_vertices); free(coverage); free(weights);
        return PROGRESSIVE_COVER_OUT_OF_MEMORY;
    }
    for (int row = 0; row < rows; ++row) owners[row] = -1;
    for (int cube = 0; cube < cardinality; ++cube) {
        for (int input = 0; input < ninputs; ++input) {
            pattern[input] = result->solution[
                (size_t)input * (size_t)cardinality + (size_t)cube
            ];
        }
        if (!progressive_pattern_coverage_prepared(
            prepared, pattern, coverage
        )) {
            free(cover_counts); free(owners); free(owner_private);
            free(pattern); free(best_vertices); free(coverage); free(weights);
            return PROGRESSIVE_COVER_SOLVER_ERROR;
        }
        for (int row = 0; row < rows; ++row) {
            if (!((coverage[row / 64] >> (row % 64)) & UINT64_C(1))) {
                continue;
            }
            if (cover_counts[row] == 0) owners[row] = cube;
            else owners[row] = -1;
            cover_counts[row]++;
        }
    }
    int private_rows = 0;
    for (int row = 0; row < rows; ++row) {
        if (cover_counts[row] == 1 && owners[row] >= 0) {
            owner_private[owners[row]]++;
            private_rows++;
        }
    }
    for (int row = 0; row < rows; ++row) {
        const int owner = owners[row];
        if (
            cover_counts[row] == 1 && owner >= 0 &&
            owner_private[owner] > 0
        ) {
            weights[row] = 1.0 / (double)owner_private[owner];
        }
    }

    const int saved_capacity = workspace->capacity;
    workspace->capacity = saved_capacity < NATIVE_PRIVATE_MICRO_PRICE_BATCH
        ? saved_capacity
        : NATIVE_PRIVATE_MICRO_PRICE_BATCH;
    workspace->count = 0;
    WeightedMMCSPriceOptions price_options = {
        .requested_threads = options->requested_threads,
        .worker_memory_budget_bytes =
            options->memory_budget_bytes *
            (uint64_t)options->worker_memory_percent / 100u,
        .collect_timing = true,
        .enable_root_gate = false,

        .cache_edge_words = true,
        .disable_cover_bound = false,
        .anchor_limit = 0,
        .deadline_nanoseconds = native_private_micro_price_deadline(
            options, global_deadline
        )
    };
    WeightedMMCSPriceResult price;
    const WeightedMMCSPriceStatus price_status =
        bounded_mmcs_price_output_batch_prepared(
            pi,
            prepared,
            options->greedy_max_support > 0
                ? options->greedy_max_support
                : prepared->ninputs,
            weights,
            best_vertices,
            workspace,
            &price_options,
            &price
        );
    ProgressiveCoverStatus status = PROGRESSIVE_COVER_OK;
    if (
        price_status != WEIGHTED_MMCS_PRICE_OK &&
        price_status != WEIGHTED_MMCS_PRICE_LIMIT_REACHED
    ) {
        status = price_status == WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY
            ? PROGRESSIVE_COVER_OUT_OF_MEMORY
            : PROGRESSIVE_COVER_SOLVER_ERROR;
    } else {
        *best_score = price.best_weight;
        for (
            int candidate = 0;
            candidate < workspace->count &&
                *inserted_columns < insertion_limit;
            ++candidate
        ) {
            memset(pattern, 0, (size_t)ninputs * sizeof(int));
            const int anchor = workspace->anchors[candidate];
            if (anchor < 0 || anchor >= pi->ON_minterms) {
                status = PROGRESSIVE_COVER_SOLVER_ERROR;
                break;
            }
            const int *anchor_row = &pi->ON_set[
                (size_t)anchor * (size_t)ninputs
            ];
            const int *vertices = &workspace->vertices[
                (size_t)candidate * (size_t)ninputs
            ];
            for (
                int literal = 0;
                literal < workspace->support_sizes[candidate];
                ++literal
            ) {
                const int input = vertices[literal];
                pattern[input] = anchor_row[input];
            }
            bool inserted = false;
            const ProgressiveNativeMasterStatus master_status =
                progressive_native_master_insert(
                    master,
                    &workspace->coverage[(size_t)candidate * (size_t)words],
                    pattern,
                    &inserted
                );
            if (master_status != PROGRESSIVE_NATIVE_MASTER_OK) {
                status = map_master_status(master_status);
                break;
            }
            if (inserted) (*inserted_columns)++;
        }
    }
    workspace->capacity = saved_capacity;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_PRIVATE_MICRO_PRICE "
        "status=%s private=%d batch=%d inserted=%d insertion_limit=%d "
        "best=%.9f "
        "anchors=%d/%d nodes=%llu complete=%s deadline=%s seconds=%.6f\n",
        price_status == WEIGHTED_MMCS_PRICE_OK ? "ok" :
            (price_status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED
                ? "limited" : "error"),
        private_rows,
        workspace->count,
        *inserted_columns,
        insertion_limit,
        price.best_weight,
        price.anchors_completed,
        price.anchors_total,
        (unsigned long long)price.search_nodes,
        price.anchor_sweep_complete ? "yes" : "no",
        price_status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED ? "yes" : "no",
        (double)(native_monotonic_nanoseconds() - started) / 1e9
    );
    free(cover_counts); free(owners); free(owner_private); free(pattern);
    free(best_vertices); free(coverage); free(weights);
    return status;
}

static const char *native_feasibility_jump_status_name(
    ProgressiveNativeFeasibilityJumpStatus status
) {
    switch (status) {
        case PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OK:
            return "ok";
        case PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_INVALID:
            return "invalid";
        case PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY:
            return "out-of-memory";
        case PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_MEMORY_LIMIT:
            return "memory-limit";
    }
    return "unknown";
}

static ProgressiveCoverStatus apply_native_feasibility_jump(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    uint64_t deadline,
    NativeFeasibilityJumpGate *gate,
    int **selected,
    int *selected_cardinality
) {
    if (
        options->native_feasibility_jump_seconds <= 0.0 ||
        result->cardinality <= 1
    ) {
        return PROGRESSIVE_COVER_OK;
    }
    const uint64_t now = native_monotonic_nanoseconds();
    if (deadline > 0 && now >= deadline) {
        result->native_feasibility_jump_deadline_reached = true;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_FEASIBILITY_JUMP "
            "status=skipped reason=global-deadline cover=%d\n",
            result->cardinality
        );
        return PROGRESSIVE_COVER_OK;
    }
    if (native_bound_is_fresh(master, result) && result->restricted_master_optimal) {
        if (gate) gate->skipped_optimal++;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_FEASIBILITY_JUMP "
            "status=skipped reason=restricted-master-optimal "
            "columns=%d cover=%d\n",
            master->count,
            result->cardinality
        );
        return PROGRESSIVE_COVER_OK;
    }
    const uint64_t incumbent_signature = native_solution_signature(
        master->ninputs,
        result->cardinality,
        result->solution
    );
    if (
        gate &&
        gate->has_failed_state &&
        incumbent_signature != 0 &&
        gate->failed_master_columns == master->count &&
        gate->failed_cardinality == result->cardinality &&
        gate->failed_incumbent_signature == incumbent_signature
    ) {
        gate->skipped_unchanged++;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_FEASIBILITY_JUMP "
            "status=skipped reason=unchanged-failed-state "
            "columns=%d cover=%d signature=%016" PRIx64 "\n",
            master->count,
            result->cardinality,
            incumbent_signature
        );
        return PROGRESSIVE_COVER_OK;
    }
    int *incumbent_columns = (int *)malloc(
        (size_t)result->cardinality * sizeof(int)
    );
    if (!incumbent_columns) return PROGRESSIVE_COVER_OUT_OF_MEMORY;
    if (!progressive_native_master_map_solution(
        master,
        prepared,
        result->cardinality,
        result->solution,
        incumbent_columns
    )) {
        free(incumbent_columns);
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }

    ProgressiveNativeFeasibilityJumpOptions jump_options;
    progressive_native_feasibility_jump_options_default(&jump_options);
    jump_options.seed = progressive_column_pool_fingerprint(prepared);
    jump_options.requested_threads = options->requested_threads;
    jump_options.logical_family_count =
        options->native_feasibility_jump_logical_families;
    jump_options.memory_budget_bytes =
        options->memory_budget_bytes *
        (uint64_t)options->batch_memory_percent / 100u;
    const bool work_budget =
        options->native_feasibility_jump_work_budget;
    jump_options.budget_mode = work_budget
        ? PROGRESSIVE_RWLS_BUDGET_WORK
        : PROGRESSIVE_RWLS_BUDGET_WALL;
    if (work_budget) {
        /*
         * The orientative target is checked between compound outer work
         * units. Work-mode RWLS itself is deliberately deadline-free, so a
         * target crossing cannot truncate or suppress a logical family round.
         */
        jump_options.deadline_nanoseconds = 0;
    } else {
        const double requested_span =
            options->native_feasibility_jump_seconds * 1e9;
        uint64_t jump_deadline =
            requested_span >= (double)(UINT64_MAX - now)
                ? UINT64_MAX
                : now + (uint64_t)requested_span;
        if (deadline > 0 && deadline < jump_deadline) {
            jump_deadline = deadline;
        }
        jump_options.deadline_nanoseconds = jump_deadline;
    }

    int *jump_columns = NULL;
    int jump_cardinality = -1;
    ProgressiveNativeFeasibilityJumpStats jump_stats;
    const ProgressiveNativeFeasibilityJumpStatus jump_status =
        progressive_native_feasibility_jump_improve(
            master,
            result->cardinality,
            incumbent_columns,
            &jump_options,
            &jump_columns,
            &jump_cardinality,
            &jump_stats
        );
    free(incumbent_columns);
    result->native_feasibility_jump_runs++;
    result->native_feasibility_jump_targets_attempted +=
        jump_stats.targets_attempted;
    result->native_feasibility_jump_targets_improved +=
        jump_stats.targets_improved;
    result->native_feasibility_jump_work_epochs +=
        jump_stats.work_epochs;
    result->native_feasibility_jump_productive_epoch_extensions +=
        jump_stats.productive_epoch_extensions;
    result->native_feasibility_jump_adaptive_rounds +=
        jump_stats.adaptive_rounds;
    result->native_feasibility_jump_restart_families +=
        jump_stats.restart_families;
    if (
        jump_stats.threads_used >
        result->native_feasibility_jump_threads_used
    ) {
        result->native_feasibility_jump_threads_used =
            jump_stats.threads_used;
    }
    if (
        jump_stats.logical_family_states >
        result->native_feasibility_jump_logical_family_states
    ) {
        result->native_feasibility_jump_logical_family_states =
            jump_stats.logical_family_states;
    }
    result->native_feasibility_jump_restarts += jump_stats.restarts;
    result->native_feasibility_jump_iterations += jump_stats.iterations;
    result->native_feasibility_jump_add_moves += jump_stats.add_moves;
    result->native_feasibility_jump_remove_moves += jump_stats.remove_moves;
    result->native_feasibility_jump_fixed_iterations +=
        jump_stats.fixed_iterations;
    result->native_feasibility_jump_soft_iterations +=
        jump_stats.soft_iterations;
    result->native_feasibility_jump_weight_reuses +=
        jump_stats.weight_reuses;
    result->native_feasibility_jump_soft_pressure_updates +=
        jump_stats.soft_pressure_updates;
    if (
        jump_stats.maximum_selected_columns >
        result->native_feasibility_jump_maximum_selected_columns
    ) {
        result->native_feasibility_jump_maximum_selected_columns =
            jump_stats.maximum_selected_columns;
    }
    result->native_feasibility_jump_best_uncovered_rows =
        jump_stats.best_uncovered_rows;
    result->native_feasibility_jump_last_failed_uncovered_rows =
        jump_stats.last_failed_uncovered_rows;
    result->native_feasibility_jump_incidence_entries =
        jump_stats.incidence_entries;
    result->native_feasibility_jump_allocated_bytes =
        jump_stats.allocated_bytes;
    result->native_feasibility_jump_shared_cache_bytes =
        jump_stats.shared_cache_bytes;
    result->native_feasibility_jump_packed_threshold_bytes =
        jump_stats.packed_incidence_threshold_bytes;
    result->native_feasibility_jump_shared_cache_source =
        (int)jump_stats.shared_cache_source;
    result->native_feasibility_jump_dense_incidence_invocations +=
        jump_stats.dense_incidence_invocations;
    result->native_feasibility_jump_packed_incidence_invocations +=
        jump_stats.packed_incidence_invocations;
    result->native_feasibility_jump_nanoseconds +=
        jump_stats.elapsed_nanoseconds;
    result->native_feasibility_jump_deadline_reached =
        result->native_feasibility_jump_deadline_reached ||
        jump_stats.deadline_reached;
    result->native_feasibility_jump_memory_limited =
        result->native_feasibility_jump_memory_limited ||
        jump_status == PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_MEMORY_LIMIT ||
        jump_status == PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_FEASIBILITY_JUMP "
        "status=%s columns=%d cover=%d->%d targets=%d/%d "
        "threads=%d logical_states=%d families=%" PRIu64 " "
        "weight_reuses=%" PRIu64 " "
        "work_epochs=%" PRIu64 "/%" PRIu64 " "
        "adaptive_rounds=%" PRIu64 " restarts=%" PRIu64 " "
        "iterations=%" PRIu64 " moves=%" PRIu64 "/%" PRIu64 " "
        "penalties=%" PRIu64 " soft_pressure=%" PRIu64 " "
        "max_selected=%d "
        "best_uncovered=%d last_failed_uncovered=%d "
        "witness=%d witness_frequency=%d witness_observations=%d "
        "witness_distinct=%d witness_signature=%" PRIu64 " "
        "policy_families=%" PRIu64 "/%" PRIu64 " "
        "policy_iterations=%" PRIu64 "/%" PRIu64 " "
        "policy_successes=%d/%d policy_wins=%d/%d "
        "elite=%d/%d elite_difference=%d/%d winner=%s "
        "compulsory=%d "
        "selected_distinct=%d worker_selected_max=%d "
        "max_penalty=%" PRId64 " max_gain=%" PRId64 " "
        "max_loss=%" PRId64 " max_timestamp=%" PRId64 " "
        "incidence=%llu column_index_bytes=%d bidirectional=%s memory=%llu "
        "shared_memory=%llu per_family_memory=%llu "
        "shared_cache=%llu cache_source=%s packed_threshold=%llu "
        "incidence_paths=%d/%d budget_mode=%s atomic_family_round=%s "
        "deadline=%s seconds=%.6f\n",
        native_feasibility_jump_status_name(jump_status),
        master->count,
        jump_stats.initial_cardinality,
        jump_stats.final_cardinality,
        jump_stats.targets_improved,
        jump_stats.targets_attempted,
        jump_stats.threads_used,
        jump_stats.logical_family_states,
        jump_stats.restart_families,
        jump_stats.weight_reuses,
        jump_stats.work_epochs,
        jump_stats.productive_epoch_extensions,
        jump_stats.adaptive_rounds,
        jump_stats.restarts,
        jump_stats.iterations,
        jump_stats.add_moves,
        jump_stats.remove_moves,
        jump_stats.penalty_updates,
        jump_stats.soft_pressure_updates,
        jump_stats.maximum_selected_columns,
        jump_stats.best_uncovered_rows,
        jump_stats.last_failed_uncovered_rows,
        jump_stats.last_failed_witness_row,
        jump_stats.last_failed_witness_frequency,
        jump_stats.last_failed_witness_observations,
        jump_stats.last_failed_witness_distinct_rows,
        jump_stats.last_failed_selection_signature,
        jump_stats.fixed_restart_families,
        jump_stats.soft_restart_families,
        jump_stats.fixed_iterations,
        jump_stats.soft_iterations,
        jump_stats.fixed_successful_candidates,
        jump_stats.soft_successful_candidates,
        jump_stats.fixed_winning_targets,
        jump_stats.soft_winning_targets,
        jump_stats.last_successful_candidates,
        jump_stats.last_successful_distinct_candidates,
        jump_stats.last_successful_min_symmetric_difference,
        jump_stats.last_successful_max_symmetric_difference,
        jump_stats.last_winning_policy ==
                PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
            ? "fixed"
            : (
                jump_stats.last_winning_policy ==
                    PROGRESSIVE_RWLS_WORKER_SOFT_PRESSURE
                ? "soft"
                : "none"
            ),
        jump_stats.compulsory_columns,
        jump_stats.distinct_selected_columns,
        jump_stats.maximum_worker_distinct_selected_columns,
        jump_stats.maximum_row_penalty,
        jump_stats.maximum_uncovered_gain,
        jump_stats.maximum_private_loss,
        jump_stats.maximum_timestamp,
        (unsigned long long)jump_stats.incidence_entries,
        jump_stats.column_incidence_index_bytes,
        jump_stats.bidirectional_incidence ? "yes" : "no",
        (unsigned long long)jump_stats.allocated_bytes,
        (unsigned long long)jump_stats.shared_allocated_bytes,
        (unsigned long long)jump_stats.per_family_allocated_bytes,
        (unsigned long long)jump_stats.shared_cache_bytes,
        progressive_rwls_shared_cache_source_name(
            jump_stats.shared_cache_source
        ),
        (unsigned long long)
            jump_stats.packed_incidence_threshold_bytes,
        jump_stats.dense_incidence_invocations,
        jump_stats.packed_incidence_invocations,
        jump_stats.budget_mode == PROGRESSIVE_RWLS_BUDGET_WORK
            ? "work"
            : "wall",
        jump_stats.budget_mode == PROGRESSIVE_RWLS_BUDGET_WORK
            ? "yes"
            : "no",
        jump_stats.deadline_reached ? "yes" : "no",
        (double)jump_stats.elapsed_nanoseconds / 1e9
    );
    if (
        jump_status == PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_MEMORY_LIMIT ||
        jump_status == PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY
    ) {
        free(jump_columns);
        return PROGRESSIVE_COVER_OUT_OF_MEMORY;
    }
    if (jump_status != PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OK) {
        free(jump_columns);
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
    if (jump_cardinality >= result->cardinality) {
        if (gate && incumbent_signature != 0) {
            gate->has_failed_state = true;
            gate->failed_master_columns = master->count;
            gate->failed_cardinality = result->cardinality;
            gate->failed_incumbent_signature = incumbent_signature;
        }
        free(jump_columns);
        return PROGRESSIVE_COVER_OK;
    }
    int *candidate = selected_patterns_to_solution(
        master,
        jump_columns,
        jump_cardinality
    );
    if (!candidate) {
        free(jump_columns);
        return PROGRESSIVE_COVER_OUT_OF_MEMORY;
    }
    const ProgressiveCoverStatus status = adopt_incumbent(
        master->ninputs,
        prepared,
        options,
        result,
        jump_cardinality,
        PROGRESSIVE_INCUMBENT_PRICED_MASTER,
        candidate
    );
    if (status != PROGRESSIVE_COVER_OK) {
        free(jump_columns);
        return status;
    }
    free(*selected);
    *selected = jump_columns;
    *selected_cardinality = jump_cardinality;
    if (gate) gate->has_failed_state = false;
    return PROGRESSIVE_COVER_OK;
}

static ProgressiveCoverStatus observe_native_adaptive_phase(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    uint64_t overall_deadline,
    uint64_t wall_deadline,
    uint64_t search_started,
    ProgressiveNativePhasePolicy *policy,
    const char *trigger,
    uint64_t completed_round,
    uint64_t next_milestone_round,
    uint64_t phase_start_round,
    int phase_start_cardinality,
    int phase_start_columns,
    int phase_added_columns,
    bool run_jump,
    NativeFeasibilityJumpGate *jump_gate,
    int **selected,
    int *selected_cardinality,
    ProgressiveNativePhaseDecision *decision
) {
    const int before_jump = result->cardinality;
    if (run_jump) {
        ProgressiveCoverStatus status = apply_native_feasibility_jump(
            master,
            prepared,
            options,
            result,
            overall_deadline,
            jump_gate,
            selected,
            selected_cardinality
        );
        if (status != PROGRESSIVE_COVER_OK) return status;
    }
    if (options->column_pool_output_path) {
        ProgressiveCoverStatus status = save_pool(
            options,
            prepared,
            master,
            result
        );
        if (status != PROGRESSIVE_COVER_OK) return status;
        result->native_adaptive_phase_pool_checkpoints++;
    }

    const ProgressiveNativePhaseSignals signals = {
        .cover_improved = result->cardinality < phase_start_cardinality,
        .selected_new_column = native_selected_uses_new_column(
            *selected,
            *selected_cardinality,
            phase_start_columns
        ),
        .jump_improved = result->cardinality < before_jump,
        .columns_added = phase_added_columns,
        .budget_allows_another_epoch =
            progressive_native_phase_budget_allows_epoch(
                search_started,
                native_monotonic_nanoseconds(),
                wall_deadline
            ),
        .unbudgeted = wall_deadline == 0,
        .allow_unbudgeted_growth = options->unlimited_tail_growth,
        .rounds_in_phase = completed_round > phase_start_round
            ? (int)(completed_round - phase_start_round)
            : 0
    };
    *decision = progressive_native_phase_policy_observe(policy, signals);
    result->native_adaptive_phase_checkpoints++;
    if (progressive_native_phase_decision_promotes(*decision)) {
        result->native_adaptive_phase_promotions++;
        if (*decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_GRACE) {
            result->native_adaptive_phase_grace_promotions++;
        } else if (*decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_BUDGET) {
            result->native_adaptive_phase_budget_promotions++;
        } else if (*decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_GROWTH) {
            result->native_adaptive_phase_growth_promotions++;
        }
    } else {
        result->native_adaptive_phase_stopped = true;
    }
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_PHASE observation=%d trigger=%s "
        "completed_round=%" PRIu64 " next_milestone_round=%" PRIu64 " "
        "columns=%d added=%d selected_new=%s cover=%d->%d "
        "jump_improved=%s stagnant=%d budget_allows_epoch=%s "
        "phase_rounds=%d decision=%s\n",
        result->native_adaptive_phase_checkpoints,
        trigger,
        completed_round,
        next_milestone_round,
        master->count,
        phase_added_columns,
        signals.selected_new_column ? "yes" : "no",
        phase_start_cardinality,
        result->cardinality,
        signals.jump_improved ? "yes" : "no",
        policy->stagnant_epochs,
        signals.budget_allows_another_epoch ? "yes" : "no",
        signals.rounds_in_phase,
        progressive_native_phase_decision_name(*decision)
    );
    return PROGRESSIVE_COVER_OK;
}

static ProgressiveCoverStatus apply_native_repair(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    const ProgressiveNativeRepairPlan *plan,
    uint64_t prepare_nanoseconds,
    uint64_t repair_deadline,
    int **selected,
    int *selected_cardinality
) {
    int *repair_indices = NULL;
    int repair_cardinality = -1;
    ProgressiveNativeRepairStats repair_stats;
    const int incumbent_before = result->cardinality;
    result->native_repair_attempts++;
    result->native_repair_attempts_since_improvement++;

    result->native_repair_destroyed_columns +=
        (uint64_t)plan->destroyed_count;
    result->native_repair_uncovered_rows +=
        (uint64_t)plan->uncovered_rows;
    if (
        plan->uncovered_rows >
        result->native_repair_max_uncovered_rows
    ) {
        result->native_repair_max_uncovered_rows =
            plan->uncovered_rows;
    }
    const uint64_t solve_started =
        native_monotonic_nanoseconds();
    const bool solved = progressive_native_repair_solve(
        master,
        plan,
        &repair_indices,
        &repair_cardinality,
        &repair_stats,
        repair_deadline
    );
    const uint64_t solve_nanoseconds =
        native_monotonic_nanoseconds() - solve_started;
    result->native_repair_solve_nanoseconds +=
        solve_nanoseconds;
    result->native_repair_deadline_stops +=
        repair_stats.deadline_reached ? 1 : 0;
    result->native_repair_residual_columns +=
        (uint64_t)repair_stats.residual_candidate_columns;
    result->native_repair_lagrangian_columns +=
        (uint64_t)repair_stats.lagrangian_columns_added;
    result->native_repair_greedy_columns +=
        (uint64_t)repair_stats.greedy_columns_added;
    result->native_repair_redundant_removed +=
        (uint64_t)repair_stats.redundant_columns_removed;
    result->native_repair_lagrangian_completions +=
        repair_stats.lagrangian_completed ? 1 : 0;
    result->native_repair_lagrangian_attempts +=
        repair_stats.lagrangian_attempted ? 1 : 0;
    result->native_repair_greedy_fallbacks +=
        repair_stats.greedy_fallback_used ? 1 : 0;
    if (
        repair_stats.residual_candidate_columns >
        result->native_repair_max_residual_columns
    ) {
        result->native_repair_max_residual_columns =
            repair_stats.residual_candidate_columns;
    }
    if (!solved) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_REPAIR_ATTEMPT "
            "neighborhood=%d incumbent=%d destroyed=%d "
            "uncovered=%d residual_columns=%d "
            "lagrangian=%s lagrangian_cardinality=%d "
            "lagrangian_bound=%d lagrangian_gap=%d "
            "lagrangian_certified=%s lagrangian_stop=%s "
            "lagrangian_effort=%d lagrangian_retry=%s "
            "lagrangian_columns=%d "
            "fallback=%s fallback_columns=%d pruned=%d "
            "returned=-1 result=error "
            "prepare_seconds=%.6f solve_seconds=%.6f\n",
            plan->neighborhood,
            incumbent_before,
            plan->destroyed_count,
            plan->uncovered_rows,
            repair_stats.residual_candidate_columns,
            native_repair_lagrangian_status(&repair_stats),
            repair_stats.lagrangian_cardinality,
            repair_stats.lagrangian_lower_bound,
            repair_stats.lagrangian_gap,
            repair_stats.lagrangian_proved_optimal ? "yes" : "no",
            lagrangian_stop_reason_name(
                repair_stats.lagrangian_stop_reason
            ),
            repair_stats.lagrangian_effort_level,
            repair_stats.lagrangian_strong_retry ? "yes" : "no",
            repair_stats.lagrangian_columns_added,
            repair_stats.greedy_fallback_used ? "yes" : "no",
            repair_stats.greedy_columns_added,
            repair_stats.redundant_columns_removed,
            (double)prepare_nanoseconds / 1e9,
            (double)solve_nanoseconds / 1e9
        );
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
    int *candidate = NULL;

    uint64_t candidate_signature = 0;
    if (repair_cardinality <= incumbent_before) {
        candidate = selected_patterns_to_solution(
            master,
            repair_indices,
            repair_cardinality
        );
        if (!candidate) {
            free(repair_indices);
            return PROGRESSIVE_COVER_OUT_OF_MEMORY;
        }
        candidate_signature = native_solution_signature(
            master->ninputs,
            repair_cardinality,
            candidate
        );
    }
    const bool sideways =
        false;
    const char *outcome =
        repair_cardinality < incumbent_before
            ? "improved"
            : (
                sideways
                    ? "sideways"
                    : (
                        repair_cardinality == incumbent_before
                            ? "equal"
                            : "worse"
                    )
            );
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_REPAIR_ATTEMPT "
        "neighborhood=%d incumbent=%d destroyed=%d "
        "uncovered=%d residual_columns=%d "
        "lagrangian=%s lagrangian_cardinality=%d "
        "lagrangian_bound=%d lagrangian_gap=%d "
        "lagrangian_certified=%s lagrangian_stop=%s "
        "lagrangian_effort=%d lagrangian_retry=%s "
        "lagrangian_columns=%d "
        "fallback=%s fallback_columns=%d pruned=%d "
        "returned=%d result=%s signature=%016" PRIx64 " "
        "prepare_seconds=%.6f solve_seconds=%.6f\n",
        plan->neighborhood,
        incumbent_before,
        plan->destroyed_count,
        plan->uncovered_rows,
        repair_stats.residual_candidate_columns,
        native_repair_lagrangian_status(&repair_stats),
        repair_stats.lagrangian_cardinality,
        repair_stats.lagrangian_lower_bound,
        repair_stats.lagrangian_gap,
        repair_stats.lagrangian_proved_optimal ? "yes" : "no",
        lagrangian_stop_reason_name(
            repair_stats.lagrangian_stop_reason
        ),
        repair_stats.lagrangian_effort_level,
        repair_stats.lagrangian_strong_retry ? "yes" : "no",
        repair_stats.lagrangian_columns_added,
        repair_stats.greedy_fallback_used ? "yes" : "no",
        repair_stats.greedy_columns_added,
        repair_stats.redundant_columns_removed,
        repair_cardinality,
        outcome,
        candidate_signature,
        (double)prepare_nanoseconds / 1e9,
        (double)solve_nanoseconds / 1e9
    );
    if (
        repair_cardinality > incumbent_before ||
        (
            repair_cardinality == incumbent_before &&
            !sideways
        )
    ) {
        result->native_repair_rejections++;
        if (
            result->native_repair_attempts_since_improvement >
            result->native_repair_max_attempts_without_improvement
        ) {
            result->native_repair_max_attempts_without_improvement =
                result->native_repair_attempts_since_improvement;
        }
        free(candidate);
        free(repair_indices);
        return PROGRESSIVE_COVER_OK;
    }
    ProgressiveCoverStatus status = adopt_incumbent(
        master->ninputs,
        prepared,
        options,
        result,
        repair_cardinality,
        PROGRESSIVE_INCUMBENT_PRICED_MASTER,
        candidate
    );
    if (status != PROGRESSIVE_COVER_OK) {
        free(repair_indices);
        return status;
    }
    if (sideways) {
        result->native_repair_sideways_moves++;
        if (
            result->native_repair_attempts_since_improvement >
            result->native_repair_max_attempts_without_improvement
        ) {
            result->native_repair_max_attempts_without_improvement =
                result->native_repair_attempts_since_improvement;
        }
    } else {
        result->native_repair_improvements++;
        result->native_repair_attempts_since_improvement = 0;
    }
    free(*selected);
    *selected = repair_indices;
    *selected_cardinality = repair_cardinality;
    return PROGRESSIVE_COVER_OK;
}

static ProgressiveCoverStatus price_and_insert_native_candidates(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveCoverOptions *options,
    ProgressiveCoverResult *result,
    ProgressiveNativeMaster *master,
    const double *pricing_weights,
    int anchor_limit,
    int forced_prefix_depth,
    bool scout_anchor_potential,
    int beam_candidate_budget,
    uint64_t deadline,
    WeightedMMCSPriceBatch *batch,
    ProgressiveGreedyPriceWorkspace *price_workspace,
    ProgressiveGreedyPriceStats *price_stats,
    int *added,
    int *duplicates
) {
    if (
        !pi ||
        !prepared ||
        !options ||
        !result ||
        !master ||
        !pricing_weights ||
        !batch ||
        !price_stats ||
        !added ||
        !duplicates
    ) {
        return PROGRESSIVE_COVER_SOLVER_ERROR;
    }
    batch->count = 0;
    ProgressiveGreedyPriceOptions price_options = {
        .workspace = price_workspace,
        .requested_threads = options->requested_threads,
        .anchor_limit = anchor_limit,
        .forced_prefix_depth = forced_prefix_depth,
        .beam_candidate_budget = beam_candidate_budget,
        .max_support = options->greedy_max_support,
        .scout_anchor_potential = scout_anchor_potential,
        .worker_memory_budget_bytes =
            options->memory_budget_bytes *
            (uint64_t)options->worker_memory_percent / 100u,
        .deadline_nanoseconds = deadline,
        .reduced_cost_threshold = 1.0,
        .reduced_cost_tolerance = PROGRESSIVE_NATIVE_RC_TOL
    };
    ProgressiveGreedyPriceStatus price_status =
        progressive_greedy_price_batch(
            pi,
            prepared,
            pricing_weights,
            batch,
            &price_options,
            price_stats
        );
    if (price_status != PROGRESSIVE_GREEDY_PRICE_OK) {
        return price_status == PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY
            ? PROGRESSIVE_COVER_OUT_OF_MEMORY
            : PROGRESSIVE_COVER_SOLVER_ERROR;
    }

    result->greedy_anchors_attempted +=
        price_stats->anchors_attempted +
        price_stats->scout_anchors_attempted;
    result->greedy_candidates_constructed +=
        price_stats->candidates_constructed +
        price_stats->scout_candidates_constructed;
    result->greedy_improving_candidates +=
        price_stats->improving_candidates;
    result->greedy_literal_evaluations +=
        price_stats->literal_evaluations +
        price_stats->scout_literal_evaluations;
    result->greedy_elapsed_nanoseconds +=
        price_stats->elapsed_nanoseconds;
    if (
        price_stats->allocated_bytes >
        result->greedy_allocated_bytes
    ) {
        result->greedy_allocated_bytes =
            price_stats->allocated_bytes;
    }
    result->pricing_threads = price_stats->effective_threads;
    result->greedy_support_minimum =
        price_stats->support_minimum;
    result->greedy_support_mean = price_stats->support_mean;
    result->greedy_support_maximum =
        price_stats->support_maximum;
    result->greedy_best_weight = price_stats->best_weight;

    *added = 0;
    *duplicates = 0;
    int *pattern = (int *)calloc(
        (size_t)master->ninputs,
        sizeof(int)
    );
    if (!pattern) return PROGRESSIVE_COVER_OUT_OF_MEMORY;
    for (int candidate = 0; candidate < batch->count; ++candidate) {
        memset(
            pattern,
            0,
            (size_t)master->ninputs * sizeof(int)
        );
        int anchor = batch->anchors[candidate];
        const int *anchor_row = &pi->ON_set[
            (size_t)anchor * (size_t)master->ninputs
        ];
        for (
            int literal = 0;
            literal < batch->support_sizes[candidate];
            ++literal
        ) {
            int input = batch->vertices[
                (size_t)candidate * (size_t)master->ninputs +
                (size_t)literal
            ];
            pattern[input] = anchor_row[input];
        }
        bool inserted = false;
        ProgressiveNativeMasterStatus insert_status =
            progressive_native_master_insert(
                master,
                &batch->coverage[
                    (size_t)candidate * (size_t)master->words
                ],
                pattern,
                &inserted
            );
        if (insert_status != PROGRESSIVE_NATIVE_MASTER_OK) {
            free(pattern);
            return map_master_status(insert_status);
        }
        if (inserted) {
            (*added)++;
        } else {
            (*duplicates)++;
            result->greedy_duplicate_candidates++;
        }
    }
    free(pattern);
    return PROGRESSIVE_COVER_OK;
}

ProgressiveCoverStatus progressive_native_cover_run(
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
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        options->greedy_forced_prefix_depth < -1 ||
        options->greedy_forced_prefix_depth > 3 ||
        options->greedy_candidate_budget <= 0 ||
        options->greedy_max_support < 0 ||
        options->greedy_max_support > ninputs ||
        options->requested_threads <= 0 ||
        options->native_effort_level < 0 ||
        options->native_effort_level > 2 ||
        options->native_repair_attempt_limit <= 0 ||
        options->native_reduce_expand_variants <= 0 ||
        options->native_reduce_expand_start_round <= 0 ||
        options->native_reduce_expand_interval <= 0 ||
        options->column_pool_snapshot_count < 0 ||
        (
            options->column_pool_snapshot_count > 0 &&
            !options->column_pool_snapshots
        ) ||
        !isfinite(options->native_repair_penalty) ||
        options->native_repair_penalty < 0.0 ||
        !isfinite(options->native_repair_time_share) ||
        options->native_repair_time_share < 0.0 ||
        options->native_repair_time_share > 1.0 ||
        !isfinite(options->native_feasibility_jump_seconds) ||
        options->native_feasibility_jump_seconds < 0.0 ||
        options->native_feasibility_jump_logical_families < 0 ||
        (
            options->native_feasibility_jump_logical_families > 0 &&
            options->native_feasibility_jump_logical_families % 4 != 0
        ) ||
        !isfinite(options->global_bound_time_limit_seconds) ||
        options->global_bound_time_limit_seconds < 0.0 ||
        options->global_bound_active_rows <= 0 ||
        options->global_bound_rounds <= 0
    ) {
        return PROGRESSIVE_COVER_INVALID_INPUT;
    }

    double previous_snapshot_seconds = 0.0;
    for (
        int snapshot = 0;
        snapshot < options->column_pool_snapshot_count;
        ++snapshot
    ) {
        const ProgressiveCoverPoolSnapshot *item =
            &options->column_pool_snapshots[snapshot];
        if (
            !isfinite(item->elapsed_seconds) ||
            item->elapsed_seconds <= previous_snapshot_seconds ||
            !item->path ||
            !*item->path
        ) {
            return PROGRESSIVE_COVER_INVALID_INPUT;
        }
        previous_snapshot_seconds = item->elapsed_seconds;
    }

    const uint64_t started = native_monotonic_nanoseconds();
    const int rows = pi->ON_minterms;
    const int words = (rows + 63) / 64;

    uint64_t deadline = 0;
    result->native_started_nanoseconds = started;
    if (options->time_limit_seconds > 0.0) {
        deadline = native_deadline_after_seconds(
            started,
            options->time_limit_seconds
        );
        const double repair_budget =
            options->time_limit_seconds *
            options->native_repair_time_share * 1e9;
        result->native_repair_budget_nanoseconds =
            repair_budget >= (double)UINT64_MAX
                ? UINT64_MAX
                : (uint64_t)repair_budget;
    }

    result->max_horizon = options->max_horizon;
    result->stable_horizons = options->stable_horizons;
    result->memory_budget_bytes = options->memory_budget_bytes;
    result->greedy_candidate_budget =
        options->greedy_candidate_budget;
    result->greedy_mip_time_limit_seconds =
        options->greedy_mip_time_limit_seconds;
    result->greedy_final_mip_time_limit_seconds =
        options->greedy_final_mip_time_limit_seconds;

    result->final_horizon = 0;
    result->pricing_closed = false;
    result->integer_best_bound = NAN;
    result->integer_mip_gap = NAN;
    result->lp_objective = NAN;

    WeightedMMCSPreparedRows prepared;
    memset(&prepared, 0, sizeof(prepared));
    ProgressiveSeedResult seed;
    memset(&seed, 0, sizeof(seed));
    ProgressiveNativeMaster master;
    memset(&master, 0, sizeof(master));
    ProgressiveGreedyPriceWorkspace *price_workspace =
        progressive_greedy_price_workspace_create();
    int *selected = NULL;
    int selected_cardinality = 0;
    double *multipliers = NULL;
    double *batch_weights = NULL;
    int *batch_anchors = NULL;
    int *batch_supports = NULL;
    int *batch_vertices = NULL;
    uint64_t *batch_coverage = NULL;
    int *automatic_best_solution = NULL;
    ProgressiveCoverStatus status = PROGRESSIVE_COVER_SOLVER_ERROR;

    if (!bounded_mmcs_prepare_binary_rows(pi, ninputs, &prepared)) {
        status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (!progressive_native_master_initialize(
        &master,
        rows,
        words,
        ninputs
    )) {
        status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
        goto cleanup;
    }
    if (!progressive_native_master_set_max_support(
        &master,
        options->greedy_max_support
    )) {
        status = PROGRESSIVE_COVER_INVALID_INPUT;
        goto cleanup;
    }

    {
        ProgressiveSeedStatus seed_status = progressive_seed_build_prepared(
            pi,
            &prepared,
            ninputs,
            options->greedy_seed,
            options->greedy_max_support,
            options->memory_budget_bytes *
                (uint64_t)options->seed_memory_percent / 100u,
            &seed
        );
        if (seed_status != PROGRESSIVE_SEED_OK) {
            status = seed_status == PROGRESSIVE_SEED_OUT_OF_MEMORY
                ? PROGRESSIVE_COVER_OUT_OF_MEMORY
                : PROGRESSIVE_COVER_INVALID_INPUT;
            goto cleanup;
        }
        result->seed_cardinality = seed.cardinality;
        result->seed_literal_count = seed.literal_count;
        result->seed_greedy_cubes = seed.greedy_cubes;
        result->seed_fallback_cubes = seed.fallback_cubes;
        result->seed_elapsed_nanoseconds = seed.elapsed_nanoseconds;
        result->seed_validation_nanoseconds =
            seed.validation_elapsed_nanoseconds;
        result->seed_allocated_bytes = seed.allocated_bytes;
        result->seed_greedy_complete = seed.greedy_complete;
        result->seed_memory_exhausted = seed.memory_exhausted;
        if (!record_seed_supports(&seed, result)) {
            status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
            goto cleanup;
        }
        int *seed_solution = transpose_patterns(
            seed.patterns,
            seed.cardinality,
            ninputs
        );
        if (!seed_solution) {
            status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
            goto cleanup;
        }
        status = adopt_incumbent(
            ninputs,
            &prepared,
            options,
            result,
            seed.cardinality,
            PROGRESSIVE_INCUMBENT_SEED,
            seed_solution
        );
        if (status != PROGRESSIVE_COVER_OK) goto cleanup;
    }

    int seed_inserted = 0;
    {
        status = map_master_status(
            progressive_native_master_insert_seed(
                &master,
                &prepared,
                &seed,
                &seed_inserted
            )
        );
        if (status != PROGRESSIVE_COVER_OK) goto cleanup;
        result->seed_master_inserted = true;
    }

    {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_SEED cover=%d inserted=%d "
            "support=%d/%.1f/%d seconds=%.6f\n",
            seed.cardinality,
            seed_inserted,
            result->seed_support_minimum,
            result->seed_support_median,
            result->seed_support_maximum,
            (double)seed.elapsed_nanoseconds / 1e9
        );
    }

    if (options->column_pool_input_path) {
        ProgressiveColumnPool pool;
        const uint64_t load_started = native_monotonic_nanoseconds();
        ProgressiveColumnPoolStatus pool_status =
            progressive_column_pool_load(
                options->column_pool_input_path,
                &prepared,
                words,
                &pool
            );
        result->column_pool_load_nanoseconds =
            native_monotonic_nanoseconds() - load_started;
        if (pool_status != PROGRESSIVE_COLUMN_POOL_OK) {
            status =
                pool_status == PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY
                    ? PROGRESSIVE_COVER_OUT_OF_MEMORY
                    : PROGRESSIVE_COVER_PERSISTENCE_ERROR;
            goto cleanup;
        }
        int inserted = 0;
        status = map_master_status(
            progressive_native_master_insert_pool(
                &master,
                &prepared,
                &pool,
                &inserted
            )
        );
        if (status == PROGRESSIVE_COVER_OK) {
            result->column_pool_loaded_columns = pool.count;
            if (
                result->cardinality <= 0 ||
                pool.incumbent_cardinality < result->cardinality
            ) {
                const size_t values =
                    (size_t)pool.incumbent_cardinality *
                    (size_t)ninputs;
                int *pool_incumbent = (int *)malloc(
                    values * sizeof(int)
                );
                if (!pool_incumbent) {
                    status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
                } else {
                    memcpy(
                        pool_incumbent,
                        pool.incumbent_solution,
                        values * sizeof(int)
                    );
                    status = adopt_incumbent(
                        ninputs,
                        &prepared,
                        options,
                        result,
                        pool.incumbent_cardinality,
                        PROGRESSIVE_INCUMBENT_COLUMN_POOL,
                        pool_incumbent
                    );
                }
            }
        }
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_POOL path=%s columns=%d "
            "inserted=%d incumbent=%d\n",
            options->column_pool_input_path,
            pool.count,
            inserted,
            pool.incumbent_cardinality
        );
        progressive_column_pool_destroy(&pool);
        if (status != PROGRESSIVE_COVER_OK) goto cleanup;
    }

    int anchor_limit = rows;
    if (options->greedy_anchor_percent < 100) {
        uint64_t scaled =
            (uint64_t)rows *
            (uint64_t)options->greedy_anchor_percent;
        anchor_limit = (int)((scaled + 99u) / 100u);
    }
    if (anchor_limit < 1) anchor_limit = 1;
    int resolved_prefix_depth =
        options->greedy_forced_prefix_depth;
    if (resolved_prefix_depth < 0) {
        int diversified_capacity = 0;
        bool exhaustive_fits =
            progressive_greedy_price_candidate_capacity(
                anchor_limit,
                ninputs,
                2,
                &diversified_capacity
            ) &&
            diversified_capacity <= options->greedy_candidate_budget;
        resolved_prefix_depth = exhaustive_fits ? 2 : 0;
    }
    result->greedy_forced_prefix_depth =
        resolved_prefix_depth;
    const uint64_t bytes_per_candidate =
        2u * sizeof(double) +
        2u * sizeof(int) +
        (uint64_t)ninputs * sizeof(int) +
        (uint64_t)words * sizeof(uint64_t);
    uint64_t batch_budget =
        options->memory_budget_bytes *
        (uint64_t)options->batch_memory_percent / 100u;
    uint64_t batch_capacity64 = bytes_per_candidate > 0
        ? batch_budget / bytes_per_candidate
        : 0;
    int candidate_capacity_int = 0;
    bool candidate_capacity_ok =
        progressive_greedy_price_candidate_capacity(
            anchor_limit,
            ninputs,
            resolved_prefix_depth,
            &candidate_capacity_int
        );
    if (!candidate_capacity_ok) {
        status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
        goto cleanup;
    }
    const uint64_t candidate_capacity = (uint64_t)(
        options->greedy_scout_anchor_rounds > 0 &&
                rows > candidate_capacity_int
            ? rows
            : candidate_capacity_int
    );
    if (batch_capacity64 > candidate_capacity) {
        batch_capacity64 = candidate_capacity;
    }
    if (
        batch_capacity64 < candidate_capacity ||
        batch_capacity64 > INT_MAX
    ) {
        status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
        goto cleanup;
    }
    const int batch_capacity = (int)batch_capacity64;
    result->batch_capacity = batch_capacity;
    result->greedy_anchor_limit = anchor_limit;
    result->batch_allocated_bytes =
        (uint64_t)batch_capacity * bytes_per_candidate;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_PREFIX_POLICY "
        "mode=%s anchors=%d depth=%d beam_width=%d "
        "candidates=%d budget=%d\n",
        resolved_prefix_depth > 0 ? "exhaustive" : "single",
        anchor_limit,
        resolved_prefix_depth,
        0,
        candidate_capacity_int,
        options->greedy_candidate_budget
    );

    multipliers = (double *)malloc((size_t)rows * sizeof(double));
    batch_weights = (double *)malloc(
        (size_t)batch_capacity * sizeof(double)
    );
    batch_anchors = (int *)malloc(
        (size_t)batch_capacity * sizeof(int)
    );
    batch_supports = (int *)malloc(
        (size_t)batch_capacity * sizeof(int)
    );
    batch_vertices = (int *)malloc(
        (size_t)batch_capacity *
        (size_t)ninputs *
        sizeof(int)
    );
    batch_coverage = (uint64_t *)malloc(
        (size_t)batch_capacity *
        (size_t)words *
        sizeof(uint64_t)
    );
    if (
        !multipliers ||
        !batch_weights ||
        !batch_anchors ||
        !batch_supports ||
        !batch_vertices ||
        !batch_coverage
    ) {
        status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
        goto cleanup;
    }

    int repair_destroy_target = 0;
    int solved_master_columns = -1;
    int solved_incumbent_generation = -1;
    bool native_reduce_expand_periodic_enabled =
        options->native_reduce_expand;

    bool native_private_micro_price_attempted = false;

    int previous_round_column_floor = -1;
    int previous_round_added = 0;
    const bool work_phases =
        options->native_feasibility_jump_work_budget &&
        options->native_feasibility_jump_seconds > 0.0;
    ProgressiveNativePhasePolicy phase_policy;
    progressive_native_phase_policy_initialize(&phase_policy);
    int phase_checkpoint_index = 0;
    uint64_t phase_checkpoint_round = work_phases
        ? progressive_native_phase_checkpoint_round(
            phase_checkpoint_index
        )
        : 0;
    int phase_start_cardinality = result->cardinality;
    int phase_start_columns = master.count;
    int phase_added_columns = 0;
    uint64_t phase_start_round = 0;
    NativeFeasibilityJumpGate feasibility_jump_gate = {0};
    NativeSoftTarget soft_target = {
        .grace_seconds = 0.0,
        .requested_seconds = options->time_limit_seconds,
        .observed_cardinality = result->cardinality,
        .base_deadline = deadline
    };

    result->native_soft_target_grace_seconds = 0.0;
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_TIME_TARGET "
        "requested_seconds=%.3f grace_seconds=pending "
        "calibration=median-3-rounds clamp_rounds=1/4 mode=%s\n",
        options->time_limit_seconds,
        soft_target.requested_seconds > 0.0
            ? "work-calibrated-rolling-cardinality"
            : "hard"
    );
    int round = 1;
    int next_pool_snapshot = 0;
resume_native_pricing:
    for (; !false;) {
        status = save_due_pool_snapshots(
            options,
            &prepared,
            &master,
            result,
            started,
            round - 1,
            &next_pool_snapshot
        );
        if (status != PROGRESSIVE_COVER_OK) goto cleanup;
        const uint64_t round_started = native_monotonic_nanoseconds();
        const int round_start_cardinality = result->cardinality;
        const int round_start_columns = master.count;
        bool phase_checkpoint_this_round = false;
        bool phase_promoted_this_round = false;
        if (native_deadline_reached(deadline)) {
            if (
                solved_master_columns != master.count ||
                solved_incumbent_generation !=
                    result->incumbent_generation
            ) {
                status = solve_native_master(
                    &master,
                    &prepared,
                    options,
                    result,
                    &selected,
                    &selected_cardinality,
                    multipliers
                );
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;
                solved_master_columns = master.count;
                solved_incumbent_generation =
                    result->incumbent_generation;
                observe_native_soft_target(
                    &soft_target,
                    result,
                    &deadline,
                    "wall-dirty-master"
                );
            }
            if (
                work_phases &&
                progressive_native_phase_policy_ready(
                    (uint64_t)result->greedy_rounds
                )
            ) {
                ProgressiveNativePhaseDecision phase_decision;
                status = observe_native_adaptive_phase(
                    &master,
                    &prepared,
                    options,
                    result,
                    progressive_native_phase_operator_deadline(
                        work_phases,
                        deadline
                    ),
                    deadline,
                    started,
                    &phase_policy,
                    "wall-guard",
                    (uint64_t)(round - 1),
                    phase_checkpoint_round,
                    phase_start_round,
                    phase_start_cardinality,
                    phase_start_columns,
                    phase_added_columns,
                    true,
                    &feasibility_jump_gate,
                    &selected,
                    &selected_cardinality,
                    &phase_decision
                );
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;
                observe_native_soft_target(
                    &soft_target,
                    result,
                    &deadline,
                    "wall-guard"
                );
                if (!native_deadline_reached(deadline)) {
                    phase_start_cardinality = result->cardinality;
                    phase_start_columns = master.count;
                    phase_added_columns = 0;
                    phase_start_round = (uint64_t)(round - 1);
                    repair_destroy_target = 0;
                    continue;
                }
            }
            if (!native_deadline_reached(deadline)) {
                continue;
            }
            result->resource_limited = true;
            result->stop_reason = PROGRESSIVE_STOP_RESOURCE_LIMIT;
            break;
        }
        if (
            solved_master_columns != master.count ||
            solved_incumbent_generation != result->incumbent_generation
        ) {
            status = solve_native_master(
                &master,
                &prepared,
                options,
                result,
                &selected,
                &selected_cardinality,
                multipliers
            );
            if (status != PROGRESSIVE_COVER_OK) goto cleanup;
            solved_master_columns = master.count;
            solved_incumbent_generation = result->incumbent_generation;
            observe_native_soft_target(
                &soft_target,
                result,
                &deadline,
                "master"
            );
        }
        if (result->cardinality == 1) {
            result->stop_reason =
                PROGRESSIVE_STOP_CARDINALITY_BOUND;
            break;
        }

        int selected_from_previous_round = 0;
        if (previous_round_column_floor >= 0) {
            for (int index = 0; index < selected_cardinality; ++index) {
                selected_from_previous_round +=
                    selected[index] >= previous_round_column_floor;
            }
        }

        ProgressiveNativeRepairPlan repair_plan;
        memset(&repair_plan, 0, sizeof(repair_plan));
        uint64_t repair_prepare_nanoseconds = 0;
        const bool repair_active = options->native_repair_pricing;
        const bool repair_attempt_allowed_before_prepare =
            repair_active && native_repair_budget_available(
                options,
                result,
                soft_target.base_deadline
            );
        const uint64_t repair_prepare_started =
            repair_active
                ? native_monotonic_nanoseconds()
                : 0;
        if (
            repair_active &&
            !progressive_native_repair_prepare(
                &master,
                &prepared,
                result->cardinality,
                result->solution,
                multipliers,
                options->native_repair_penalty,
                repair_destroy_target,
                round - 1,
                &repair_plan
            )
        ) {
            status = PROGRESSIVE_COVER_SOLVER_ERROR;
            goto cleanup;
        }
        if (repair_active) {
            repair_prepare_nanoseconds =
                native_monotonic_nanoseconds() -
                repair_prepare_started;
            if (repair_attempt_allowed_before_prepare) {
                result->native_repair_prepare_nanoseconds +=
                    repair_prepare_nanoseconds;
            }
        }
        const double *pricing_weights =
            repair_active
                ? repair_plan.pricing_weights
                : multipliers;

        WeightedMMCSPriceBatch batch = {
            .capacity = batch_capacity,
            .weights = batch_weights,
            .anchors = batch_anchors,
            .support_sizes = batch_supports,
            .vertices = batch_vertices,
            .coverage = batch_coverage
        };
        ProgressiveGreedyPriceStats price_stats;
        int added = 0;
        int duplicates = 0;
        /*
         * Pricing is one atomic work unit.  The orientative output deadline
         * is checked between rounds instead of truncating a candidate batch
         * at a machine-dependent wall-clock position.
         */
        const uint64_t pricing_deadline = 0;
        const int trace_master_columns = master.count;

        if (status != PROGRESSIVE_COVER_OK) {
            progressive_native_repair_destroy(&repair_plan);
            goto cleanup;
        }
        status = price_and_insert_native_candidates(
            pi,
            &prepared,
            options,
            result,
            &master,
            pricing_weights,
            anchor_limit,
            resolved_prefix_depth,
            round <= options->greedy_scout_anchor_rounds,
            0,
            pricing_deadline,
            &batch,
            price_workspace,
            &price_stats,
            &added,
            &duplicates
        );
        if (status != PROGRESSIVE_COVER_OK) {
            progressive_native_repair_destroy(&repair_plan);
            goto cleanup;
        }
        const int ordinary_pricing_added = added;

        if (status != PROGRESSIVE_COVER_OK) {
            progressive_native_repair_destroy(&repair_plan);
            goto cleanup;
        }
        result->greedy_rounds = round;
        if (repair_active) {
            if (native_bound_is_fresh(&master, result) &&
                result->restricted_master_optimal && added == 0) {
                result->native_repair_skipped_optimal++;
                fprintf(
                    stderr,
                    "CCUBES_PROGRESSIVE_NATIVE_REPAIR_SKIP "
                    "round=%d reason=restricted-master-optimal-no-columns\n",
                    round
                );
                progressive_native_repair_destroy(&repair_plan);
            } else if (
                repair_attempt_allowed_before_prepare &&
                native_repair_budget_available(
                options,
                result,
                soft_target.base_deadline
                )
            ) {
                const int incumbent_before_repair = result->cardinality;
                const uint64_t repair_deadline =
                    native_repair_attempt_deadline(
                        options,
                        result,
                        soft_target.base_deadline
                    );
                status = apply_native_repair(
                    &master,
                    &prepared,
                    options,
                    result,
                    &repair_plan,
                    repair_prepare_nanoseconds,
                    repair_deadline,
                    &selected,
                    &selected_cardinality
                );
                repair_destroy_target =
                    result->cardinality < incumbent_before_repair
                        ? 0
                        : progressive_native_repair_next_destroy_count(
                            result->cardinality,
                            repair_plan.destroyed_count
                        );
                progressive_native_repair_destroy(&repair_plan);
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;
                observe_native_soft_target(
                    &soft_target,
                    result,
                    &deadline,
                    "repair"
                );
            } else {
                result->native_repair_budget_skips++;
                progressive_native_repair_destroy(&repair_plan);
            }
        }
        if (
            native_reduce_expand_periodic_enabled &&
            round >= options->native_reduce_expand_start_round &&
            (
                round - options->native_reduce_expand_start_round
            ) % options->native_reduce_expand_interval == 0 &&
            !native_deadline_reached(pricing_deadline)
        ) {
            const uint64_t variants_before =
                result->native_reduce_expand_variants;
            int reshaped = 0;
            status = generate_native_reduce_expand(
                &prepared,
                &master,
                options,
                result,
                round,
                pricing_deadline,
                &reshaped
            );
            if (status != PROGRESSIVE_COVER_OK) goto cleanup;
            added += reshaped;
            const uint64_t pass_variants =
                result->native_reduce_expand_variants - variants_before;
            if (
                pass_variants > (uint64_t)INT_MAX ||
                !progressive_native_reduce_expand_should_continue(
                    reshaped,
                    (int)pass_variants
                )
            ) {
                native_reduce_expand_periodic_enabled = false;
                fprintf(
                    stderr,
                    "CCUBES_PROGRESSIVE_NATIVE_REDUCE_EXPAND_GATE "
                    "pass=%d round=%d inserted=%d variants=%llu "
                    "yield=%.6f threshold=0.400000 decision=stop\n",
                    result->native_reduce_expand_passes,
                    round,
                    reshaped,
                    (unsigned long long)pass_variants,
                    pass_variants > 0
                        ? (double)reshaped / (double)pass_variants
                        : 0.0
                );
            }
        }

        const bool consider_private_micro =
            options->native_private_micro &&
            !native_private_micro_price_attempted &&
            (added == 0 || round >= NATIVE_PRIVATE_MICRO_MATURE_ROUND) &&
            !native_deadline_reached(deadline);
        if (consider_private_micro) {
            refresh_native_bound(&master, multipliers, result);
        }
        if (
            consider_private_micro &&
            isfinite(result->lp_objective) &&
            ceil(result->lp_objective - 1e-9) < result->cardinality &&
            !native_deadline_reached(deadline)
        ) {
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_NATIVE_PRIVATE_MICRO_TRIGGER "
                "reason=%s round=%d added=%d cover=%d lp=%.9f\n",
                added == 0 ? "pricing-stall" : "mature-round",
                round,
                added,
                result->cardinality,
                result->lp_objective
            );
            native_private_micro_price_attempted = true;

            if (
                options->native_private_micro
            ) {
                int micro_inserted = 0;
                double micro_best_score = 0.0;

                uint64_t micro_deadline = deadline;

                status = generate_native_private_micro_price(
                    pi,
                    &prepared,
                    &master,
                    options,
                    result,
                    micro_deadline,
                    NATIVE_PRIVATE_MICRO_PRICE_INSERTIONS,
                    &batch,
                    &micro_inserted,
                    &micro_best_score
                );
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;

                added += micro_inserted;
            }
        }

        if (added > 0) {
            result->pricing_rounds++;
        }
        if (work_phases) phase_added_columns += added;
        const bool fresh_bound = native_bound_is_fresh(&master, result);
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE round=%d columns=%d added=%d "
            "duplicates=%d "
            "restricted_lagrangian=%.9f "
            "bound_scope=restricted_master bound_columns=%d bound_fresh=%s "
            "integer=%d support=%d/%.1f/%d "
            "best_weight=%.9f repair=%d/%d "
            "repair_no_improvement=%d prefix_mode=%s "
            "dual_seconds=%.6f dual_total_seconds=%.6f\n",
            round,
            master.count,
            added,
            duplicates,
            fresh_bound ? result->lp_objective : NAN,
            result->restricted_bound_columns,
            fresh_bound ? "yes" : "no",
            result->cardinality,
            price_stats.support_minimum,
            price_stats.support_mean,
            price_stats.support_maximum,
            price_stats.best_weight,
            result->native_repair_improvements,
            result->native_repair_attempts,
            result->native_repair_attempts_since_improvement,
            resolved_prefix_depth > 0 ? "exhaustive" : "single",
            (double)result->native_dual_last_nanoseconds / 1e9,
            (double)result->native_dual_nanoseconds / 1e9
        );
        const uint64_t round_finished = native_monotonic_nanoseconds();
        const double restricted_gap =
            fresh_bound && isfinite(result->lp_objective)
                ? result->cardinality - ceil(result->lp_objective - 1e-9)
                : -1.0;
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_PRODUCTIVITY round=%d "
            "round_seconds=%.6f cumulative_seconds=%.6f "
            "cover=%d cover_delta=%d restricted_gap=%.0f "
            "columns=%d master_growth=%d pricing_added=%d "
            "total_added=%d previous_added=%d "
            "selected_from_previous=%d anchors=%llu "
            "constructed=%llu improving=%llu duplicates=%d "
            "best_weight=%.9f\n",
            round,
            (double)(round_finished - round_started) / 1e9,
            (double)(round_finished - started) / 1e9,
            result->cardinality,
            round_start_cardinality - result->cardinality,
            restricted_gap,
            master.count,
            master.count - round_start_columns,
            ordinary_pricing_added,
            added,
            previous_round_added,
            selected_from_previous_round,
            (unsigned long long)price_stats.anchors_attempted,
            (unsigned long long)price_stats.candidates_constructed,
            (unsigned long long)price_stats.improving_candidates,
            duplicates,
            price_stats.best_weight
        );
        previous_round_column_floor = trace_master_columns;
        previous_round_added = added;
        observe_native_completed_round(
            &soft_target,
            result,
            native_monotonic_nanoseconds() - round_started
        );
        observe_native_soft_target(
            &soft_target,
            result,
            &deadline,
            "round"
        );
        /* Advance only after master, pricing, repair, and any scheduled
         * REDUCE--EXPAND work for this logical round have completed.  A
         * wall-guard grace continuation above therefore cannot create a
         * phantom round or shift work-count schedules. */
        const int completed_round = round;
        round++;
        if (
            work_phases &&
            progressive_native_phase_policy_ready(
                (uint64_t)completed_round
            ) &&
            phase_checkpoint_round > 0 &&
            (uint64_t)completed_round >= phase_checkpoint_round
        ) {
            ProgressiveNativePhaseDecision phase_decision;
            if (
                solved_master_columns != master.count ||
                solved_incumbent_generation !=
                    result->incumbent_generation
            ) {
                status = solve_native_master(
                    &master,
                    &prepared,
                    options,
                    result,
                    &selected,
                    &selected_cardinality,
                    multipliers
                );
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;
                solved_master_columns = master.count;
                solved_incumbent_generation =
                    result->incumbent_generation;
                observe_native_soft_target(
                    &soft_target,
                    result,
                    &deadline,
                    "checkpoint-dirty-master"
                );
            }
            status = observe_native_adaptive_phase(
                &master,
                &prepared,
                options,
                result,
                progressive_native_phase_operator_deadline(
                    work_phases,
                    deadline
                ),
                deadline,
                started,
                &phase_policy,
                "work-rounds",
                (uint64_t)completed_round,
                progressive_native_phase_checkpoint_round(
                    phase_checkpoint_index + 1
                ),
                phase_start_round,
                phase_start_cardinality,
                phase_start_columns,
                phase_added_columns,
                true,
                &feasibility_jump_gate,
                &selected,
                &selected_cardinality,
                &phase_decision
            );
            if (status != PROGRESSIVE_COVER_OK) goto cleanup;
            observe_native_soft_target(
                &soft_target,
                result,
                &deadline,
                "phase"
            );
            phase_checkpoint_this_round = true;
            phase_promoted_this_round =
                progressive_native_phase_decision_promotes(
                    phase_decision
                );
            if (!phase_promoted_this_round) {
                result->stop_reason = PROGRESSIVE_STOP_STABLE;
                break;
            }
            phase_start_cardinality = result->cardinality;
            phase_start_columns = master.count;
            phase_added_columns = 0;
            phase_start_round = (uint64_t)completed_round;
            phase_checkpoint_index++;
            phase_checkpoint_round =
                progressive_native_phase_checkpoint_round(
                    phase_checkpoint_index
                );
        }
        if (result->cardinality == 1) {
            result->stop_reason =
                PROGRESSIVE_STOP_CARDINALITY_BOUND;
            break;
        }
        if (added == 0) {
            if (phase_checkpoint_this_round && phase_promoted_this_round) {
                repair_destroy_target = 0;
                continue;
            }
            if (
                work_phases &&
                !phase_checkpoint_this_round &&
                progressive_native_phase_policy_ready(
                    (uint64_t)completed_round
                )
            ) {
                ProgressiveNativePhaseDecision phase_decision;
                status = observe_native_adaptive_phase(
                    &master,
                    &prepared,
                    options,
                    result,
                    progressive_native_phase_operator_deadline(
                        work_phases,
                        deadline
                    ),
                    deadline,
                    started,
                    &phase_policy,
                    "pricing-stable",
                    (uint64_t)completed_round,
                    phase_checkpoint_round,
                    phase_start_round,
                    phase_start_cardinality,
                    phase_start_columns,
                    phase_added_columns,
                    true,
                    &feasibility_jump_gate,
                    &selected,
                    &selected_cardinality,
                    &phase_decision
                );
                if (status != PROGRESSIVE_COVER_OK) goto cleanup;
                observe_native_soft_target(
                    &soft_target,
                    result,
                    &deadline,
                    "phase"
                );
                if (
                    progressive_native_phase_decision_promotes(
                        phase_decision
                    )
                ) {
                    phase_start_cardinality = result->cardinality;
                    phase_start_columns = master.count;
                    phase_added_columns = 0;
                    phase_start_round = (uint64_t)completed_round;
                    repair_destroy_target = 0;
                    continue;
                }
            }
            result->stop_reason = PROGRESSIVE_STOP_STABLE;
            break;
        }
        if (options->greedy_single_pricing_batch) {
            result->stop_reason = PROGRESSIVE_STOP_BATCH_COMPLETE;
            break;
        }
    }

    status = save_due_pool_snapshots(
        options,
        &prepared,
        &master,
        result,
        started,
        round - 1,
        &next_pool_snapshot
    );
    if (status != PROGRESSIVE_COVER_OK) goto cleanup;

    bool final_deadline_renewed = false;
    if (
        (
            solved_master_columns != master.count ||
            solved_incumbent_generation != result->incumbent_generation
        )
    ) {
        status = solve_native_master(
            &master,
            &prepared,
            options,
            result,
            &selected,
            &selected_cardinality,
            multipliers
        );
        if (status != PROGRESSIVE_COVER_OK) goto cleanup;
        solved_master_columns = master.count;
        solved_incumbent_generation = result->incumbent_generation;
        final_deadline_renewed = observe_native_soft_target(
            &soft_target,
            result,
            &deadline,
            "final-master"
        );
    }

    status = apply_native_feasibility_jump(
        &master,
        &prepared,
        options,
        result,
        progressive_native_phase_operator_deadline(
            work_phases,
            deadline
        ),
        &feasibility_jump_gate,
        &selected,
        &selected_cardinality
    );
    if (status != PROGRESSIVE_COVER_OK) goto cleanup;
    if (observe_native_soft_target(
        &soft_target,
        result,
        &deadline,
        "final-rwls"
    )) {
        final_deadline_renewed = true;
    }
    if (
        final_deadline_renewed &&
        !options->greedy_single_pricing_batch &&
        !native_deadline_reached(deadline) &&
        result->cardinality > 1
    ) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_TIME_RESUME "
            "round=%d cover=%d columns=%d deadline_remaining=yes\n",
            round,
            result->cardinality,
            master.count
        );
        result->resource_limited = false;
        result->stop_reason = PROGRESSIVE_STOP_STABLE;
        progressive_native_phase_policy_initialize(&phase_policy);
        phase_start_cardinality = result->cardinality;
        phase_start_columns = master.count;
        phase_added_columns = 0;
        phase_start_round = 0;
        repair_destroy_target = 0;
        goto resume_native_pricing;
    }

    if (native_deadline_reached(deadline)) {
        result->resource_limited = true;
        result->stop_reason = PROGRESSIVE_STOP_RESOURCE_LIMIT;
    }
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_TIME_TARGET_SUMMARY "
        "grace_seconds=%.3f descending_at_base=%s base_cover=%d "
        "extensions=%d extension_seconds=%.6f\n",
        result->native_soft_target_grace_seconds,
        result->native_soft_target_descending_at_base ? "yes" : "no",
        result->native_soft_target_base_cardinality,
        result->native_soft_target_extensions,
        (double)result->native_soft_target_extension_nanoseconds / 1e9
    );
    if (options->native_feasibility_jump_seconds > 0.0) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_SUMMARY "
            "runs=%d targets=%d/%d threads=%d logical_states=%d "
            "families=%llu "
            "weight_reuses=%llu adaptive_rounds=%llu "
            "work_epochs=%llu/%llu "
            "restarts=%llu iterations=%llu soft_pressure=%llu "
            "max_selected=%d "
            "best_uncovered=%d last_failed_uncovered=%d "
            "incidence=%llu memory=%llu "
            "shared_cache=%llu cache_source=%s packed_threshold=%llu "
            "incidence_paths=%d/%d budget_mode=%s "
            "skipped_optimal=%d skipped_unchanged=%d "
            "memory_limited=%s deadline=%s "
            "seconds=%.6f best=%d\n",
            result->native_feasibility_jump_runs,
            result->native_feasibility_jump_targets_improved,
            result->native_feasibility_jump_targets_attempted,
            result->native_feasibility_jump_threads_used,
            result->native_feasibility_jump_logical_family_states,
            (unsigned long long)
                result->native_feasibility_jump_restart_families,
            (unsigned long long)
                result->native_feasibility_jump_weight_reuses,
            (unsigned long long)
                result->native_feasibility_jump_adaptive_rounds,
            (unsigned long long)
                result->native_feasibility_jump_work_epochs,
            (unsigned long long)
                result->
                    native_feasibility_jump_productive_epoch_extensions,
            (unsigned long long)
                result->native_feasibility_jump_restarts,
            (unsigned long long)
                result->native_feasibility_jump_iterations,
            (unsigned long long)
                result->native_feasibility_jump_soft_pressure_updates,
            result->native_feasibility_jump_maximum_selected_columns,
            result->native_feasibility_jump_best_uncovered_rows,
            result->native_feasibility_jump_last_failed_uncovered_rows,
            (unsigned long long)
                result->native_feasibility_jump_incidence_entries,
            (unsigned long long)
                result->native_feasibility_jump_allocated_bytes,
            (unsigned long long)
                result->native_feasibility_jump_shared_cache_bytes,
            progressive_rwls_shared_cache_source_name(
                (ProgressiveRWLSSharedCacheSource)
                    result->native_feasibility_jump_shared_cache_source
            ),
            (unsigned long long)
                result->native_feasibility_jump_packed_threshold_bytes,
            result->native_feasibility_jump_dense_incidence_invocations,
            result->native_feasibility_jump_packed_incidence_invocations,
            options->native_feasibility_jump_work_budget
                ? "work"
                : "wall",
            feasibility_jump_gate.skipped_optimal,
            feasibility_jump_gate.skipped_unchanged,
            result->native_feasibility_jump_memory_limited
                ? "yes"
                : "no",
            result->native_feasibility_jump_deadline_reached
                ? "yes"
                : "no",
            (double)result->native_feasibility_jump_nanoseconds / 1e9,
            result->cardinality
        );
    }

    if (work_phases) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_PHASE_SUMMARY checkpoints=%d "
            "promotions=%d grace=%d budget=%d growth=%d "
            "pool_checkpoints=%d stopped=%s best=%d\n",
            result->native_adaptive_phase_checkpoints,
            result->native_adaptive_phase_promotions,
            result->native_adaptive_phase_grace_promotions,
            result->native_adaptive_phase_budget_promotions,
            result->native_adaptive_phase_growth_promotions,
            result->native_adaptive_phase_pool_checkpoints,
            result->native_adaptive_phase_stopped ? "yes" : "no",
            result->cardinality
        );
    }
    if (options->native_reduce_expand) {
        fprintf(
            stderr,
            "CCUBES_PROGRESSIVE_NATIVE_REDUCE_EXPAND_SUMMARY "
            "passes=%d cubes=%d reduced=%d variants=%d changed=%d "
            "inserted=%d duplicates=%d reduce_literals=%llu "
            "seconds=%.6f best=%d\n",
            result->native_reduce_expand_passes,
            result->native_reduce_expand_cubes,
            result->native_reduce_expand_reduced_cores,
            result->native_reduce_expand_variants,
            result->native_reduce_expand_changed_geometries,
            result->native_reduce_expand_inserted_columns,
            result->native_reduce_expand_duplicate_columns,
            (unsigned long long)
                result->native_reduce_expand_literals_added,
            (double)result->native_reduce_expand_nanoseconds / 1e9,
            result->cardinality
        );
    }

    result->retained_columns = master.count;
    if (!native_bound_is_fresh(&master, result)) {
        result->lp_objective = NAN;
        result->integer_best_bound = NAN;
        result->integer_mip_gap = NAN;
        result->restricted_master_optimal = false;
    }
    fprintf(
        stderr,
        "CCUBES_PROGRESSIVE_NATIVE_TERMINAL columns=%d bound_columns=%d "
        "bound_fresh=%s restricted_lagrangian=%.9f restricted_integer_bound=%.9f "
        "cover=%d\n",
        master.count,
        result->restricted_bound_columns,
        native_bound_is_fresh(&master, result) ? "yes" : "no",
        result->lp_objective,
        result->integer_best_bound,
        result->cardinality
    );
    status = save_pool(options, &prepared, &master, result);
    if (status != PROGRESSIVE_COVER_OK) goto cleanup;
    if (options->global_bound_time_limit_seconds > 0.0) {
        run_native_global_bound(
            pi,
            ninputs,
            &prepared,
            &master,
            multipliers,
            options,
            result
        );
    }
    /*
     * Reduce input literals at fixed cardinality. This runs after the column
     * pool is written, so a replacement cube that is no longer a master column
     * cannot desynchronize a saved pool from its incumbent.
     */
    if (result->solution && result->cardinality > 0) {
        int literal_polish_mode = options->literal_polish_mode;
        if (literal_polish_mode != 0) {
            const size_t values =
                (size_t)ninputs * (size_t)result->cardinality;
            int *restore = (int *)malloc(values * sizeof(int));
            if (!restore) {
                status = PROGRESSIVE_COVER_OUT_OF_MEMORY;
                goto cleanup;
            }
            memcpy(restore, result->solution, values * sizeof(int));
            ProgressiveLiteralPolishStats polish;
            const uint64_t polish_memory_budget =
                options->memory_budget_bytes / 100u *
                    (uint64_t)options->worker_memory_percent +
                options->memory_budget_bytes % 100u *
                    (uint64_t)options->worker_memory_percent / 100u;
            /*
             * The literal polish is the final quality tail after the
             * search loop has already stopped, so it receives its own
             * optional wall allowance rather than the exhausted round
             * deadline. Zero (the default) leaves it unlimited, which
             * preserves the established behavior; a positive value keeps
             * the partial literal reductions of every fully completed
             * pass when the allowance expires.
             */
            uint64_t polish_deadline = 0;
            int polish_pass_cap = literal_polish_mode == 1 ? 1 : 4;
            double polish_yield_gate =
                literal_polish_mode == 3 ? 0.0 : 0.25;
            const char *polish_deadline_setting = ccubes_option(OPTION_PROGRESSIVE_LITERAL_POLISH_DEADLINE_SECONDS);
            if (polish_deadline_setting && *polish_deadline_setting) {
                char *end = NULL;
                double allowance = strtod(
                    polish_deadline_setting,
                    &end
                );
                if (
                    end &&
                    *end == '\0' &&
                    isfinite(allowance) &&
                    allowance > 0.0
                ) {
                    polish_deadline = native_deadline_after_seconds(
                        native_monotonic_nanoseconds(),
                        allowance
                    );
                }
            }
            const bool polished = progressive_literal_polish(
                &prepared,
                result->cardinality,
                result->solution,
                options->requested_threads,
                PROGRESSIVE_NATIVE_LITERAL_POLISH_NODES,
                polish_memory_budget,
                polish_deadline,
                polish_pass_cap,
                polish_yield_gate,
                &polish
            );
            const bool valid =
                polished &&
                progressive_solution_is_valid_prepared(
                    &prepared,
                    result->cardinality,
                    result->solution
                );
            if (!valid) {
                memcpy(result->solution, restore, values * sizeof(int));
                polish.cubes_reduced = 0;
                polish.literals_after = polish.literals_before;
            }
            free(restore);
            fprintf(
                stderr,
                "CCUBES_PROGRESSIVE_LITERAL_POLISH status=%s policy=%d "
                "pass_cap=%d yield_gate=%.6g passes=%d "
                "examined=%d reduced=%d proposals=%d stale=%d "
                "threads=%d literals=%d->%d nodes=%llu "
                "node_limit_reached=%s active_lists=%s deadline=%s "
                "allocated_bytes=%llu active_list_bytes=%llu "
                "seconds=%.6f\n",
                valid ? "applied" : "rejected",
                literal_polish_mode,
                polish_pass_cap,
                polish_yield_gate,
                polish.passes,
                polish.cubes_examined,
                polish.cubes_reduced,
                polish.proposals,
                polish.stale_proposals,
                polish.threads_used,
                polish.literals_before,
                polish.literals_after,
                (unsigned long long)polish.search_nodes,
                polish.node_limit_reached ? "yes" : "no",
                polish.active_lists_used ? "yes" : "no",
                polish.deadline_reached ? "yes" : "no",
                (unsigned long long)polish.allocated_bytes,
                (unsigned long long)polish.active_list_bytes,
                polish.seconds
            );
        }
    }
    status = result->resource_limited
        ? PROGRESSIVE_COVER_RESOURCE_LIMIT
        : PROGRESSIVE_COVER_OK;

cleanup:
    result->elapsed_seconds =
        (double)(native_monotonic_nanoseconds() - started) / 1e9;
    free(selected);
    free(multipliers);
    free(batch_weights);
    free(batch_anchors);
    free(batch_supports);
    free(batch_vertices);
    free(batch_coverage);
    free(automatic_best_solution);
    progressive_native_master_destroy(&master);
    progressive_greedy_price_workspace_destroy(price_workspace);
    progressive_seed_result_destroy(&seed);
    bounded_mmcs_prepared_rows_destroy(&prepared);
    if (
        status != PROGRESSIVE_COVER_OK &&
        status != PROGRESSIVE_COVER_RESOURCE_LIMIT
    ) {
        free(result->solution);
        result->solution = NULL;
        result->incumbent_valid = false;
    }
    return status;
}
