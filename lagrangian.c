/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "lagrangian.h"
#include "ccubes_threads.h"
#include "cover_validation.h"
#include "polish_order.h"
#include "ipc_chart_polish.h"

#include <stdatomic.h>
#include <time.h>
#if defined(HAVE_PTHREAD)
#include <pthread.h>
#elif defined(HAVE_OPENMP) && defined(_WIN32)
#include <windows.h>
#endif

#define EPS 1e-12
#define LAGRANGIAN_REDUCTION_BUCKET_LIMIT 64
#ifndef LAGR_EFFORT_ZERO_PRIMAL_UB_CADENCE
#define LAGR_EFFORT_ZERO_PRIMAL_UB_CADENCE 3
#endif

static LagrangianStats lagr_last_stats = {
    .rows = 0,
    .cols = 0,
    .iterations = 0,
    .best_ub = -1,
    .best_lb = INT_MIN,
    .gap = -1,
    .pool_mode = 0,
    .warm_start_requested = 0,
    .warm_start_accepted = 0,
    .effort_level = 0,
    .certification_requested = 0,
    .requested_iteration_limit = 0,
    .iteration_limit = 0,
    .iteration_limit_reason = LAGR_ITERATION_LIMIT_CONFIGURED,
    .continuous_improvements = 0,
    .last_continuous_improvement_iteration = 0,
    .step_contractions = 0,
    .restarts = 0,
    .multiplier_schedule = 0,
    .multiplier_convergence = 0,
    .reset_on_continuous_improvement = 0,
    .stop_at_step_floor = 0,
    .portfolio_limit = 0,
    .polish_enabled = 0,
    .polish_node_limit = 0,
    .polish_core_cap = 0,
    .polish_initial_ub = -1,
    .polish_final_ub = -1,
    .polish_initial_target = -1,
    .polish_initial_core_count = -1,
    .polish_search_target = -1,
    .polish_search_core_count = -1,
    .polish_forced_columns = 0,
    .polish_skipped_targets = 0,
    .polish_searches = 0,
    .polish_nodes = 0,
    .polish_aborted_searches = 0,
    .polish_exhausted_searches = 0,
    .polish_improvements = 0,
    .polish_proved_optimal = 0,
    .polish_seconds = 0.0,
    .polish_stop_reason = LAGR_POLISH_NOT_RUN,
    .primal_ub_cadence = 0,
    .primal_ub_supplied = -1,
    .primal_ub_loop_start = -1,
    .primal_ub_loop_final = -1,
    .nonzero_subgradient_minimum = INT_MAX,
    .best_zlb = -DBL_MAX,
    .last_zlb = -DBL_MAX,
    .requested_initial_step_coef = 0.0,
    .requested_step_min = 0.0,
    .requested_step_contract = 0.0,
    .requested_stabilization_beta = 0.0,
    .requested_deflection_alpha = 0.0,
    .initial_step_coef = 0.0,
    .step_coef = 0.0,
    .step_min = 0.0,
    .step_contract = 0.0,
    .stabilization_beta = 0.0,
    .deflection_alpha = 0.0,
    .stop_reason = LAGR_STOP_NOT_RUN
};

void lagrangian_reset_stats(void) {
    lagr_last_stats.rows = 0;
    lagr_last_stats.cols = 0;
    lagr_last_stats.iterations = 0;
    lagr_last_stats.best_ub = -1;
    lagr_last_stats.best_lb = INT_MIN;
    lagr_last_stats.gap = -1;
    lagr_last_stats.pool_mode = 0;
    lagr_last_stats.warm_start_requested = 0;
    lagr_last_stats.warm_start_accepted = 0;
    lagr_last_stats.warm_start = 0;
    lagr_last_stats.effort_level = 0;
    lagr_last_stats.certification_requested = 0;
    lagr_last_stats.presolve_cols_removed = 0;
    lagr_last_stats.requested_iteration_limit = 0;
    lagr_last_stats.iteration_limit = 0;
    lagr_last_stats.iteration_limit_reason =
        LAGR_ITERATION_LIMIT_CONFIGURED;
    lagr_last_stats.continuous_improvements = 0;
    lagr_last_stats.last_continuous_improvement_iteration = 0;
    lagr_last_stats.step_contractions = 0;
    lagr_last_stats.restarts = 0;
    lagr_last_stats.multiplier_schedule = 0;
    lagr_last_stats.multiplier_convergence = 0;
    lagr_last_stats.reset_on_continuous_improvement = 0;
    lagr_last_stats.stop_at_step_floor = 0;
    lagr_last_stats.portfolio_limit = 0;
    lagr_last_stats.polish_enabled = 0;
    lagr_last_stats.polish_node_limit = 0;
    lagr_last_stats.polish_core_cap = 0;
    lagr_last_stats.polish_initial_ub = -1;
    lagr_last_stats.polish_final_ub = -1;
    lagr_last_stats.polish_initial_target = -1;
    lagr_last_stats.polish_initial_core_count = -1;
    lagr_last_stats.polish_search_target = -1;
    lagr_last_stats.polish_search_core_count = -1;
    lagr_last_stats.polish_forced_columns = 0;
    lagr_last_stats.polish_skipped_targets = 0;
    lagr_last_stats.polish_searches = 0;
    lagr_last_stats.polish_nodes = 0;
    lagr_last_stats.polish_extra_nodes = 0;
    lagr_last_stats.polish_workers = 0;
    lagr_last_stats.polish_active_workers = 0;
    lagr_last_stats.polish_warm_nodes = 0;
    lagr_last_stats.polish_warm_work = 0;
    lagr_last_stats.polish_warm_peak_bytes = 0;
    lagr_last_stats.polish_warm_workers = 0;
    lagr_last_stats.polish_warm_fallbacks = 0;
    lagr_last_stats.polish_aborted_searches = 0;
    lagr_last_stats.polish_exhausted_searches = 0;
    lagr_last_stats.polish_improvements = 0;
    lagr_last_stats.polish_proved_optimal = 0;
    lagr_last_stats.polish_seconds = 0.0;
    lagr_last_stats.polish_stop_reason = LAGR_POLISH_NOT_RUN;
    lagr_last_stats.reduced_cost_nanoseconds = 0;
    lagr_last_stats.heuristic_nanoseconds = 0;
    lagr_last_stats.heuristic_row_min_nanoseconds = 0;
    lagr_last_stats.heuristic_greedy_nanoseconds = 0;
    lagr_last_stats.heuristic_repair_nanoseconds = 0;
    lagr_last_stats.bundle_nanoseconds = 0;
    lagr_last_stats.subgradient_nanoseconds = 0;
    lagr_last_stats.reduced_cost_teams = (LagrangianTeamStats){0};
    lagr_last_stats.slack_teams = (LagrangianTeamStats){0};
    lagr_last_stats.nonzero_subgradient_rows = 0;
    lagr_last_stats.reduced_cost_calls = 0;
    lagr_last_stats.heuristic_calls = 0;
    lagr_last_stats.primal_ub_cadence = 0;
    lagr_last_stats.primal_ub_supplied = -1;
    lagr_last_stats.primal_ub_loop_start = -1;
    lagr_last_stats.primal_ub_loop_final = -1;
    lagr_last_stats.primal_ub_recurring_calls = 0;
    lagr_last_stats.primal_ub_improving_calls = 0;
    lagr_last_stats.primal_ub_improving_calls_below_supplied = 0;
    lagr_last_stats.primal_ub_first_improvement_iteration = 0;
    lagr_last_stats.primal_ub_last_improvement_iteration = 0;
    lagr_last_stats.primal_ub_longest_improvement_gap = 0;
    lagr_last_stats.bundle_calls = 0;
    lagr_last_stats.subgradient_calls = 0;
    lagr_last_stats.subgradient_samples = 0;
    lagr_last_stats.nonzero_subgradient_minimum = INT_MAX;
    lagr_last_stats.nonzero_subgradient_maximum = 0;
    lagr_last_stats.best_zlb = -DBL_MAX;
    lagr_last_stats.last_zlb = -DBL_MAX;
    lagr_last_stats.requested_initial_step_coef = 0.0;
    lagr_last_stats.requested_step_min = 0.0;
    lagr_last_stats.requested_step_contract = 0.0;
    lagr_last_stats.requested_stabilization_beta = 0.0;
    lagr_last_stats.requested_deflection_alpha = 0.0;
    lagr_last_stats.initial_step_coef = 0.0;
    lagr_last_stats.step_coef = 0.0;
    lagr_last_stats.step_min = 0.0;
    lagr_last_stats.step_contract = 0.0;
    lagr_last_stats.stabilization_beta = 0.0;
    lagr_last_stats.deflection_alpha = 0.0;
    lagr_last_stats.stop_reason = LAGR_STOP_NOT_RUN;
}

const LagrangianStats *lagrangian_last_stats(void) {
    return &lagr_last_stats;
}

bool lagrangian_last_run_proved_optimal(void) {
    return (
        lagr_last_stats.best_lb != INT_MIN &&
        lagr_last_stats.best_ub > 0 &&
        lagr_last_stats.best_lb == lagr_last_stats.best_ub
    );
}

const char *lagrangian_stop_reason_name(LagrangianStopReason reason) {
    switch (reason) {
        case LAGR_STOP_OPTIMAL: return "optimal";
        case LAGR_STOP_MAX_ITER: return "max_iter";
        case LAGR_STOP_NO_UPDATE: return "no_update";
        case LAGR_STOP_STEP_MIN: return "step_min";
        case LAGR_STOP_WARM_STALL: return "warm_stall";
        case LAGR_STOP_DEADLINE: return "deadline";
        case LAGR_STOP_INFEASIBLE: return "infeasible";
        case LAGR_STOP_OOM: return "oom";
        case LAGR_STOP_NO_CANDIDATE: return "no_candidate";
        case LAGR_STOP_NOT_RUN:
        default: return "not_run";
    }
}

const char *lagrangian_multiplier_schedule_name(int schedule) {
    switch ((LagrangianMultiplierSchedule)schedule) {
        case LAGR_MULTIPLIER_SCHEDULE_LEGACY:
            return "legacy";
        case LAGR_MULTIPLIER_SCHEDULE_SCALED:
            return "scaled";
        case LAGR_MULTIPLIER_SCHEDULE_NONE:
        default:
            return "none";
    }
}

const char *lagrangian_multiplier_convergence_name(
    int convergence
) {
    switch ((LagrangianMultiplierConvergence)convergence) {
        case LAGR_MULTIPLIER_CONVERGENCE_LEGACY:
            return "legacy";
        case LAGR_MULTIPLIER_CONVERGENCE_ADAPTIVE:
            return "adaptive";
        case LAGR_MULTIPLIER_CONVERGENCE_NONE:
        default:
            return "none";
    }
}

const char *lagrangian_iteration_limit_reason_name(int reason) {
    switch ((LagrangianIterationLimitReason)reason) {
        case LAGR_ITERATION_LIMIT_LEGACY_FLOOR:
            return "legacy_floor";
        case LAGR_ITERATION_LIMIT_SCALED_BY_ROWS:
            return "scaled_by_rows";
        case LAGR_ITERATION_LIMIT_WARM_SCALED:
            return "warm_scaled";
        case LAGR_ITERATION_LIMIT_CONFIGURED:
        default:
            return "configured";
    }
}

const char *lagrangian_polish_stop_reason_name(
    LagrangianPolishStopReason reason
) {
    switch (reason) {
        case LAGR_POLISH_DISABLED: return "disabled";
        case LAGR_POLISH_NO_TARGET: return "no_target";
        case LAGR_POLISH_CORE_CAP: return "core_cap";
        case LAGR_POLISH_FORCED_INFEASIBLE:
            return "forced_infeasible";
        case LAGR_POLISH_ROOT_BOUND: return "root_bound";
        case LAGR_POLISH_ABORTED: return "aborted";
        case LAGR_POLISH_EXHAUSTED: return "exhausted";
        case LAGR_POLISH_IMPROVED: return "improved";
        case LAGR_POLISH_DEADLINE: return "deadline";
        case LAGR_POLISH_OOM: return "oom";
        case LAGR_POLISH_NOT_RUN:
        default:
            return "not_run";
    }
}

static uint64_t lagr_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static void lagr_stats_begin(int rows, int cols, int pool_mode) {
    lagrangian_reset_stats();
    lagr_last_stats.rows = rows;
    lagr_last_stats.cols = cols;
    lagr_last_stats.pool_mode = pool_mode;
}

static void lagr_stats_finish(
    int best_ub,
    double best_lb,
    double best_zlb,
    double last_zlb,
    double step_coef,
    int iterations,
    LagrangianStopReason stop_reason
) {
    lagr_last_stats.best_ub = best_ub;
    if (best_lb > (double)INT_MIN && best_lb < (double)INT_MAX) {
        lagr_last_stats.best_lb = (int)best_lb;
    } else {
        lagr_last_stats.best_lb = INT_MIN;
    }
    if (lagr_last_stats.best_ub >= 0 && lagr_last_stats.best_lb != INT_MIN) {
        lagr_last_stats.gap = lagr_last_stats.best_ub - lagr_last_stats.best_lb;
        if (lagr_last_stats.gap < 0) lagr_last_stats.gap = 0;
    } else {
        lagr_last_stats.gap = -1;
    }
    lagr_last_stats.best_zlb = best_zlb;
    lagr_last_stats.last_zlb = last_zlb;
    lagr_last_stats.step_coef = step_coef;
    lagr_last_stats.iterations = iterations;
    lagr_last_stats.stop_reason = stop_reason;
}

static int lagr_popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
#endif
}

static int lagr_ctz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & UINT64_C(1)) == 0) { x >>= 1; ++n; }
    return n;
#endif
}

/*
 * Keep floating-point reduction groups independent of the worker count.
 * Threads may execute these buckets in any order, but each bucket visits a
 * fixed contiguous item range and the caller folds bucket totals in index
 * order.  Changing -t therefore changes only who performs the arithmetic,
 * not its grouping or order.
 */
static int lagr_reduction_bucket_count(int items) {
    if (items <= 0) return 0;
    return items < LAGRANGIAN_REDUCTION_BUCKET_LIMIT
        ? items
        : LAGRANGIAN_REDUCTION_BUCKET_LIMIT;
}

static void lagr_reduction_bucket_bounds(
    int items,
    int bucket_count,
    int bucket,
    uint64_t *start,
    uint64_t *end
) {
    const uint64_t count = (uint64_t)items;
    const uint64_t buckets = (uint64_t)bucket_count;
    const uint64_t index = (uint64_t)bucket;
    const uint64_t width = count / buckets;
    const uint64_t remainder = count % buckets;
    const uint64_t prefix = index < remainder ? index : remainder;
    *start = index * width + prefix;
    *end = *start + width + (index < remainder ? 1u : 0u);
}

typedef struct {
    int columns;
    int rows;
} LagrangianLoopThreads;

static _Thread_local uint64_t lagr_incidence_work;

static int lagr_reduction_threads(int items, uint64_t incidences, int requested) {
    (void)incidences;
    /* Arithmetic buckets no longer limit the executor team. This is an
       upper allowance; the measured controller chooses the actual team. */
    return items < requested ? (items > 0 ? items : 1) : (requested > 0 ? requested : 1);
}

static LagrangianLoopThreads lagr_loop_threads(int rows, int cols, const int *rowsCoveredCount) {
    int requested = ccubes_thread_count();
    lagr_incidence_work = 0;
    for (int col = 0; col < cols; ++col) lagr_incidence_work += (uint64_t)rowsCoveredCount[col];
    return (LagrangianLoopThreads){
        .columns = lagr_reduction_threads(cols, lagr_incidence_work, requested),
        .rows = lagr_reduction_threads(rows, lagr_incidence_work, requested)
    };
}

static _Thread_local LagrangianCalibrationStats *lagr_calibration_stats;

static void lagr_record_team(LagrangianTeamStats *stats, int workers) {
    if (!stats->calls || workers < stats->min_workers) stats->min_workers = workers;
    if (workers > stats->max_workers) stats->max_workers = workers;
    stats->calls++;
    stats->serial_calls += workers == 1;
    stats->worker_sum += (uint64_t)workers;
}

static void lagr_merge_teams(LagrangianTeamStats *total, const LagrangianTeamStats *part) {
    if (!part->calls) return;
    if (!total->calls || part->min_workers < total->min_workers) total->min_workers = part->min_workers;
    if (part->max_workers > total->max_workers) total->max_workers = part->max_workers;
    total->calls += part->calls;
    total->serial_calls += part->serial_calls;
    total->worker_sum += part->worker_sum;
}

static int lagr_run_reduction(int items, int threads, ccubes_range_worker worker,
                               void *context, ccubes_adaptive_team *team) {
    threads = ccubes_adaptive_team_choose(team, (uint64_t)items,
        lagr_incidence_work + (uint64_t)items, threads);
    bool measure = !team->settled;
    uint64_t started = measure ? ccubes_monotonic_nanoseconds() : 0;
    int actual_threads = 1;
    if (threads <= 1) {
        worker(0, (uint64_t)items, 1, 0, 1, context);
    } else if (!ccubes_parallel_for_counted(0, (uint64_t)items, threads, false,
                                          worker, context, &actual_threads)) {
        worker(0, (uint64_t)items, 1, 0, 1, context);
        actual_threads = 1;
        team->best = 1;
        team->settled = true;
    }
    if (measure) {
        uint64_t finished = ccubes_monotonic_nanoseconds();
        ccubes_adaptive_team_observe(team, finished > started ? finished - started : 0);
    }
    return actual_threads;
}

typedef struct {
    const PIChartView *chart;
    int rows;
    int cols;
    int *rows_covered_count;
    int *thread_row_count;
    int *thread_row_write;
    int **rows_covered;
    int **cols_covering;
} lagr_adjacency_context;

static void lagr_count_packed_adjacency_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    lagr_adjacency_context *ctx = (lagr_adjacency_context *)data;
    int *row_count = &ctx->thread_row_count[(size_t)worker_id * (size_t)ctx->rows];
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        int count = 0;
        const uint64_t *bits = &ctx->chart->bits[(size_t)c * (size_t)ctx->chart->words];
        for (int w = 0; w < ctx->chart->words; ++w) {
            uint64_t word = bits[w];
            count += lagr_popcount64(word);
            while (word) {
                int row = w * ctx->chart->cov_bits + lagr_ctz64(word);
                if (row < ctx->rows) ++row_count[row];
                word &= word - 1;
            }
        }
        ctx->rows_covered_count[c] = count;
    }
}

static void lagr_fill_packed_adjacency_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    lagr_adjacency_context *ctx = (lagr_adjacency_context *)data;
    int *row_write = &ctx->thread_row_write[(size_t)worker_id * (size_t)ctx->rows];
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        int row_pos = 0;
        const uint64_t *bits = &ctx->chart->bits[(size_t)c * (size_t)ctx->chart->words];
        for (int w = 0; w < ctx->chart->words; ++w) {
            uint64_t word = bits[w];
            while (word) {
                int row = w * ctx->chart->cov_bits + lagr_ctz64(word);
                if (row < ctx->rows) {
                    ctx->rows_covered[c][row_pos++] = row;
                    ctx->cols_covering[row][row_write[row]++] = c;
                }
                word &= word - 1;
            }
        }
    }
}

typedef struct {
    int **rowsCovered;
    int *rowsCoveredCount;
    const double *t;
    double *ls;
    double *partials;
    const double *col_costs;
    unsigned char *x;
    int cols;
    int bucket_count;
} lagr_reduced_context;

static void lagr_reduced_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_reduced_context *ctx = (lagr_reduced_context *)data;
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        double sum = 0.0;
        for (int k = 0; k < ctx->rowsCoveredCount[c]; ++k)
            sum += ctx->t[ctx->rowsCovered[c][k]];
        double cost = ctx->col_costs ? ctx->col_costs[c] : 1.0;
        ctx->ls[c] = cost - sum;
        if (ctx->x) ctx->x[c] = ctx->ls[c] < 0.0;
    }
}

typedef struct {
    int **colsCovering;
    int *colsCoveringCount;
    const unsigned char *x;
    double *slack;
    double *partials;
    uint64_t *nonzero_partials;
    int rows;
    int bucket_count;
} lagr_slack_context;

static void lagr_slack_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_slack_context *ctx = (lagr_slack_context *)data;
    for (uint64_t r0 = start; r0 < end; r0 += stride) {
        int r = (int)r0;
        int covered = 0;
        for (int k = 0; k < ctx->colsCoveringCount[r]; ++k)
            covered += ctx->x[ctx->colsCovering[r][k]];
        ctx->slack[r] = 1.0 - (double)covered;
    }
}

static double lagr_compute_slacks(
    int rows,
    int **colsCovering,
    int *colsCoveringCount,
    const unsigned char *x,
    double *slack,
    int *nonzero_rows_out,
    int threads
) {
    const int bucket_count = lagr_reduction_bucket_count(rows);
    double partials[LAGRANGIAN_REDUCTION_BUCKET_LIMIT];
    uint64_t nonzero_partials[LAGRANGIAN_REDUCTION_BUCKET_LIMIT];
    lagr_slack_context context = {
        .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount,
        .x = x,
        .slack = slack,
        .partials = partials,
        .nonzero_partials = nonzero_partials,
        .rows = rows,
        .bucket_count = bucket_count
    };
    static _Thread_local ccubes_adaptive_team teams[65];
    unsigned key = 0;
    for (uint64_t work = lagr_incidence_work + (uint64_t)rows; work; work >>= 1) ++key;
    int workers = lagr_run_reduction(rows, threads, lagr_slack_worker, &context, &teams[key]);
    lagr_record_team(lagr_calibration_stats ? &lagr_calibration_stats->slack_teams
        : &lagr_last_stats.slack_teams, workers);
    /* Preserve every original addition and bucket boundary, independently
       of which executor computed each row's slack. */
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
        uint64_t first, last;
        lagr_reduction_bucket_bounds(rows, bucket_count, bucket, &first, &last);
        double sum = 0.0;
        uint64_t nonzero = 0;
        for (uint64_t row = first; row < last; ++row) {
            double value = slack[row];
            sum += value * value;
            if (value != 0.0) ++nonzero;
        }
        partials[bucket] = sum;
        nonzero_partials[bucket] = nonzero;
    }

    double sum_s2 = 0.0;
    uint64_t nonzero_rows = 0;
    for (int i = 0; i < bucket_count; ++i) {
        sum_s2 += partials[i];
        nonzero_rows += nonzero_partials[i];
    }
    if (nonzero_rows_out) *nonzero_rows_out = (int)nonzero_rows;
    return sum_s2;
}

/*
Build adjacency:
rowsCovered[c] = list of rows that column c covers
colsCovering[r] = list of columns that cover row r
Returns:
0 on success
-2 if any row has no covering column (infeasible)
-1 on out of memory.
*/

static int build_adjacency_with_threads(
    const PIChartView *chart,
    int rows, int cols,
    int threads,
    int ***rowsCovered,
    int **rowsCoveredCount,
    int ***colsCovering,
    int **colsCoveringCount
) {
    int **rc = NULL, *rcc = NULL;
    int **cr = NULL, *crc = NULL;
    int *thread_row_count = NULL;
    int *thread_row_write = NULL;

    if (!chart || !chart->bits || rows <= 0 || cols <= 0 ||
        chart->rows != rows || chart->cols != cols || chart->words <= 0 ||
        chart->cov_bits <= 0 || chart->cov_bits > 64) {
        return -1;
    }
    if (threads > cols) threads = cols;
    /* Count/write scratch has its own byte allowance, independent of the
       floating-point reduction grouping. Failed allocations still shrink
       the team below. */
    const uint64_t scratch_bytes = UINT64_C(32) * 1024 * 1024;
    uint64_t memory_threads = scratch_bytes / ((uint64_t)rows * 2 * sizeof(int));
    if ((uint64_t)threads > memory_threads) threads = (int)memory_threads;
    if (threads < 1) threads = 1;

    static _Thread_local ccubes_adaptive_team teams[65];
    uint64_t work = (uint64_t)cols * (uint64_t)chart->words + (uint64_t)rows;
    unsigned key = 0;
    for (uint64_t value = work; value; value >>= 1) ++key;
    threads = ccubes_adaptive_team_choose(&teams[key], (uint64_t)cols, work, threads);
    bool measure = !teams[key].settled;
    uint64_t started = measure ? ccubes_monotonic_nanoseconds() : 0;

    rcc = (int*)calloc((size_t)cols, sizeof(int));
    crc = (int*)calloc((size_t)rows, sizeof(int));
    if (!rcc || !crc) goto oom;

    /*
     * A constrained host may still reject the preferred scratch team.  Retry
     * with a smaller physical team; contiguous chunks are concatenated in
     * worker-id order below, so column order remains ascending and independent
     * of the successful team size.
     */
    for (;;) {
        if ((size_t)rows <= SIZE_MAX / (size_t)threads) {
            const size_t entries = (size_t)threads * (size_t)rows;
            thread_row_count = (int*)calloc(entries, sizeof(int));
            thread_row_write = (int*)calloc(entries, sizeof(int));
            if (thread_row_count && thread_row_write) break;
        }
        free(thread_row_count);
        free(thread_row_write);
        thread_row_count = NULL;
        thread_row_write = NULL;
        if (threads <= 1) goto oom;
        threads = (threads + 1) / 2;
    }

    lagr_adjacency_context adjacency = {
        .chart = chart,
        .rows = rows,
        .cols = cols,
        .rows_covered_count = rcc,
        .thread_row_count = thread_row_count,
        .thread_row_write = thread_row_write,
        .rows_covered = NULL,
        .cols_covering = NULL
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, threads, false, lagr_count_packed_adjacency_worker, &adjacency)) {
        goto oom;
    }

    for (int r = 0; r < rows; ++r) {
        int offset = 0;
        for (int t = 0; t < threads; ++t) {
            size_t index = (size_t)t * (size_t)rows + (size_t)r;
            thread_row_write[index] = offset;
            offset += thread_row_count[index];
        }
        crc[r] = offset;
    }

    for (int r = 0; r < rows; ++r) {
        if (crc[r] == 0) {
            free(rcc);
            free(crc);
            free(thread_row_count);
            free(thread_row_write);
            return -2;
        }
    }

    /* Each direction owns one pointer table followed by its incidence arena.
     * Presolve may clear pointers or compact lists, but ownership stays at the
     * table base. Column and row incidence order is unchanged. */
    size_t incidences = 0;
    for (int c = 0; c < cols; ++c) {
        if ((size_t)rcc[c] > SIZE_MAX - incidences) goto oom;
        incidences += (size_t)rcc[c];
    }
    const size_t rc_table_bytes = (size_t)cols * sizeof(int *);
    const size_t cr_table_bytes = (size_t)rows * sizeof(int *);
    if (incidences > (SIZE_MAX - rc_table_bytes) / sizeof(int) ||
        incidences > (SIZE_MAX - cr_table_bytes) / sizeof(int)) goto oom;
    rc = (int **)malloc(rc_table_bytes + incidences * sizeof(int));
    cr = (int **)malloc(cr_table_bytes + incidences * sizeof(int));
    if (!rc || !cr) goto oom;
    int *rc_arena = (int *)((unsigned char *)rc + rc_table_bytes);
    int *cr_arena = (int *)((unsigned char *)cr + cr_table_bytes);
    size_t offset = 0;
    for (int c = 0; c < cols; ++c) {
        rc[c] = rcc[c] > 0 ? rc_arena + offset : NULL;
        offset += (size_t)rcc[c];
    }
    offset = 0;
    for (int r = 0; r < rows; ++r) {
        cr[r] = cr_arena + offset;
        offset += (size_t)crc[r];
    }

    adjacency.rows_covered = rc;
    adjacency.cols_covering = cr;
    if (!ccubes_parallel_for(0, (uint64_t)cols, threads, false, lagr_fill_packed_adjacency_worker, &adjacency)) {
        goto oom;
    }

    free(thread_row_count);
    free(thread_row_write);

    *rowsCovered = rc;
    *rowsCoveredCount = rcc;
    *colsCovering = cr;
    *colsCoveringCount = crc;
    if (measure) {
        uint64_t finished = ccubes_monotonic_nanoseconds();
        ccubes_adaptive_team_observe(&teams[key], finished > started ? finished - started : 0);
    }
    return 0;

oom: // out of memory
    free(thread_row_count);
    free(thread_row_write);
    free(rc);
    free(cr);
    free(rcc);
    free(crc);
    return -1;
}

static int build_adjacency(
    const PIChartView *chart,
    int rows, int cols,
    int ***rowsCovered,
    int **rowsCoveredCount,
    int ***colsCovering,
    int **colsCoveringCount
) {
    return build_adjacency_with_threads(
        chart, rows, cols, ccubes_thread_count(),
        rowsCovered, rowsCoveredCount, colsCovering, colsCoveringCount
    );
}

static void free_adjacency(
    int **rowsCovered,
    int *rowsCoveredCount,
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    int rows
) {
    (void)cols;
    (void)rows;
    free(rowsCovered);
    free(rowsCoveredCount);
    free(colsCovering);
    free(colsCoveringCount);
}

typedef struct {
    int words;
    uint64_t tail_mask;
    const uint64_t *bits;
    int **rows_covered;
    const int *row_counts;
    int **cols_covering;
    const int *col_counts;
    const double *weights;
    int *first_dominator;
} LagrangianDominanceScan;

static int lagr_rarest_row(const LagrangianDominanceScan *scan, int column) {
    int row = -1;
    for (int k = 0; k < scan->row_counts[column]; ++k) {
        const int candidate = scan->rows_covered[column][k];
        if (row < 0 || scan->col_counts[candidate] < scan->col_counts[row]) {
            row = candidate;
        }
    }
    return row;
}

static int lagr_find_dominator(
    const LagrangianDominanceScan *scan,
    int column, int row, int start, const unsigned char *active
) {
    const int count = scan->row_counts[column];
    const double weight = scan->weights ? scan->weights[column] : 0.0;
    const uint64_t *bits = scan->bits + (size_t)column * scan->words;
    for (int k = start; k < scan->col_counts[row]; ++k) {
        const int other = scan->cols_covering[row][k];
        if (other == column || (active && !active[other])) continue;
        if (scan->row_counts[other] < count) continue;

        const double other_weight = scan->weights ? scan->weights[other] : 0.0;
        if (other_weight + EPS < weight) continue;
        if (scan->row_counts[other] == count &&
            fabs(other_weight - weight) <= EPS && other > column) continue;

        const uint64_t *other_bits = scan->bits + (size_t)other * scan->words;
        int word = 0;
        for (; word < scan->words; ++word) {
            const uint64_t mask = word == scan->words - 1 ? scan->tail_mask : UINT64_MAX;
            if (bits[word] & ~other_bits[word] & mask) break;
        }
        if (word == scan->words) return k;
    }
    return -1;
}

static void lagr_scan_dominators(
    uint64_t start, uint64_t end, uint64_t stride,
    int worker_id, int worker_count, void *data
) {
    (void)worker_id;
    (void)worker_count;
    LagrangianDominanceScan *scan = data;
    for (uint64_t c = start; c < end; c += stride) {
        const int row = lagr_rarest_row(scan, (int)c);
        scan->first_dominator[c] = row < 0 ? -1 :
            lagr_find_dominator(scan, (int)c, row, 0, NULL);
    }
}

/*
Column presolve (single-solution path only): deactivate any column whose
coverage is a subset of another column's coverage, provided the dominating
column has at least equal weight. This preserves both the minimum-size
objective and the max-weight tie-break, while shrinking the instance for
the subgradient loop, the heuristics and the bounded finish.
Deactivated columns get rowsCoveredCount[c] = 0 and are removed from the
colsCovering lists; column indices remain stable, so solutions still refer
to the original pichart columns.
Returns the number of deactivated columns (0 on OOM: presolve is skipped).
*/
static int presolve_dominated_columns(
    const PIChartView *chart,
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *weights,
    int threads
) {
    if (rows <= 0 || cols <= 1) return 0;

    static _Thread_local ccubes_adaptive_team teams[65];
    uint64_t work = (uint64_t)cols * ((uint64_t)rows + 63) / 64;
    unsigned key = 0;
    for (uint64_t value = work; value; value >>= 1) ++key;
    threads = ccubes_adaptive_team_choose(&teams[key], (uint64_t)cols, work, threads);
    bool measure = !teams[key].settled;
    uint64_t started = measure ? ccubes_monotonic_nanoseconds() : 0;

    int words = (rows + 63) / 64;
    uint64_t *owned_bits = NULL;
    const uint64_t *bits = NULL;
    if (chart && chart->cov_bits == 64 && chart->words == words) {
        bits = chart->bits;
    } else {
        owned_bits = (uint64_t *)calloc(
            (size_t)cols * (size_t)words, sizeof(uint64_t));
        bits = owned_bits;
    }
    unsigned char *active = (unsigned char*)malloc((size_t)cols);
    if (!bits || !active) {
        free(owned_bits);
        free(active);
        return 0;
    }
    memset(active, 1, (size_t)cols);
    if (owned_bits) {
        for (int c = 0; c < cols; ++c) {
            uint64_t *bc = owned_bits + (size_t)c * words;
            for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                int r = rowsCovered[c][k];
                bc[r >> 6] |= 1ULL << (r & 63);
            }
        }
    }
    LagrangianDominanceScan scan = {
        .words = words,
        .tail_mask = rows % 64 ? (UINT64_C(1) << (rows % 64)) - 1 : UINT64_MAX,
        .bits = bits,
        .rows_covered = rowsCovered,
        .row_counts = rowsCoveredCount,
        .cols_covering = colsCovering,
        .col_counts = colsCoveringCount,
        .weights = weights
    };
    if (threads > 1) {
        scan.first_dominator = malloc((size_t)cols * sizeof(int));
        if (scan.first_dominator &&
            !ccubes_parallel_for(0, (uint64_t)cols, threads, true,
                                 lagr_scan_dominators, &scan)) {
            free(scan.first_dominator);
            scan.first_dominator = NULL;
        }
    }

    int removed = 0;
    for (int c = 0; c < cols; ++c) {
        const int start = scan.first_dominator ? scan.first_dominator[c] : 0;
        if (start < 0) continue;
        const int row = lagr_rarest_row(&scan, c);
        if (row < 0) continue;

        /* Keep the original removal order, including weight ties. A
           precomputed dominator may have been removed by an earlier column. */
        const int dominator = scan.first_dominator && active[colsCovering[row][start]]
            ? start : lagr_find_dominator(&scan, c, row, start, active);
        if (dominator >= 0) {
            active[c] = 0;
            ++removed;
        }
    }
    free(scan.first_dominator);

    if (removed > 0) {
        for (int c = 0; c < cols; ++c) {
            if (!active[c]) {
                rowsCovered[c] = NULL;
                rowsCoveredCount[c] = 0;
            }
        }
        for (int r = 0; r < rows; ++r) {
            int w = 0;
            for (int k = 0; k < colsCoveringCount[r]; ++k) {
                int cc = colsCovering[r][k];
                if (active[cc]) colsCovering[r][w++] = cc;
            }
            colsCoveringCount[r] = w;
        }
    }

    free(owned_bits);
    free(active);
    if (measure) {
        uint64_t finished = ccubes_monotonic_nanoseconds();
        ccubes_adaptive_team_observe(&teams[key], finished > started ? finished - started : 0);
    }
    return removed;
}

/* Scratch belongs to one solve, not to global or worker-local state. The
 * heuristics are sequential; they can reuse these buffers while parallel
 * reductions retain their fixed logical bucket order. */
typedef struct {
    unsigned char *row_covered;
    unsigned char *selected;
    int *cover_count;
    int *order;
    double *selected_scores;
    uint64_t *levelstamp;
    uint64_t *bucket;
    double *slack;
} LagrangianWorkspace;

static void lagr_workspace_destroy(LagrangianWorkspace *workspace) {
    free(workspace->row_covered);
    free(workspace->selected);
    free(workspace->cover_count);
    free(workspace->order);
    free(workspace->selected_scores);
    free(workspace->levelstamp);
    free(workspace->bucket);
    free(workspace->slack);
    memset(workspace, 0, sizeof(*workspace));
}

static bool lagr_workspace_init(LagrangianWorkspace *workspace, int rows, int cols) {
    workspace->row_covered = (unsigned char *)malloc((size_t)rows);
    workspace->selected = (unsigned char *)malloc((size_t)cols);
    workspace->cover_count = (int *)malloc((size_t)rows * sizeof(int));
    const int solution_capacity = rows < cols ? rows : cols;
    workspace->order = (int *)malloc((size_t)solution_capacity * sizeof(int));
    workspace->selected_scores = (double *)malloc((size_t)solution_capacity * sizeof(double));
    workspace->levelstamp = (uint64_t *)malloc((size_t)cols * sizeof(uint64_t));
    workspace->bucket = (uint64_t *)malloc((size_t)cols * sizeof(uint64_t));
    workspace->slack = (double *)malloc((size_t)rows * sizeof(double));
    return workspace->row_covered && workspace->selected &&
        workspace->cover_count && workspace->order && workspace->selected_scores &&
        workspace->levelstamp && workspace->bucket && workspace->slack;
}

/* Redundancy pruning: remove columns that are not needed to keep all rows covered (reverse order) */
static void prune_redundancy_with_counts(
    int rows,
    int **rowsCovered,
    int *rowsCoveredCount,
    int *sol, int *sol_len,
    int *coverCount
) {
    int n = *sol_len;
    if (n <= 0) return;

    memset(coverCount, 0, (size_t)rows * sizeof(int));

    for (int i = 0; i < n; ++i) {
        int c = sol[i];
        for (int k = 0; k < rowsCoveredCount[c]; ++k) {
            int r = rowsCovered[c][k];
            coverCount[r]++;
        }
    }

    int write = n;
    for (int i = n - 1; i >= 0; --i) {
        int c = sol[i];
        bool removable = true;

        for (int k = 0; k < rowsCoveredCount[c]; ++k) {
            int r = rowsCovered[c][k];
            if (coverCount[r] <= 1) {
                removable = false;
                break;
            }
        }

        if (removable) {
            for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                int r = rowsCovered[c][k];
                coverCount[r]--;
            }

        } else {
            sol[--write] = c;
        }
    }

    *sol_len = n - write;
    memmove(sol, sol + write, (size_t)*sol_len * sizeof(int));
}

static void prune_redundancy(
    int rows,
    int **rowsCovered,
    int *rowsCoveredCount,
    int *sol, int *sol_len
) {
    if (*sol_len <= 0) return;
    int *coverCount = (int *)malloc((size_t)rows * sizeof(int));
    if (!coverCount) return;
    prune_redundancy_with_counts(
        rows, rowsCovered, rowsCoveredCount, sol, sol_len, coverCount);
    free(coverCount);
}

/*
Greedy construction guided by Lagrangian scores
Primary: maximize newCover (minimize number of columns)
Tie 1: smaller lagr_score (more negative is better)
Tie 2: higher weight
Tie 3: smaller index

The deferred implementation keeps a packed residual-degree counter and
arrival stamp per column. Covering a row decrements the corresponding column
counters; the current maximum level is rebuilt on demand in deterministic
arrival order.

*/

static void greedy_from_lagr_scores_deferred(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights,
    int *sol,
    int *sol_len,
    LagrangianWorkspace *workspace
);

static void greedy_from_lagr_scores(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights, /* weights can be NULL */
    int *sol,
    int *sol_len,
    LagrangianWorkspace *workspace
) {
    greedy_from_lagr_scores_deferred(
        rows, cols, rowsCovered, rowsCoveredCount,
        colsCovering, colsCoveringCount, lagr_score, weights,
        sol, sol_len,
        workspace
    );
}

static int lagr_compare_stamp_desc(const void *left, const void *right) {
    uint64_t a = *(const uint64_t *)left;
    uint64_t b = *(const uint64_t *)right;
    return (a < b) - (a > b);
}

/*
Deferred bucket-maintenance greedy: the same selection rule as the linked
backend, but the doubly linked bucket chains are not maintained per
incidence. Covering a row only decrements a residual-degree counter and
stamps the touched columns with a global arrival clock. Because levels
never increase and the current maximum level is by definition the largest
live level, no column can arrive at the current top level while it is
scanned. When the top level empties, the next non-empty level is rebuilt
once: its columns are collected and ordered by descending arrival stamp,
which is exactly the linked backend's head-insertion chain order at that
moment. Scans, tie-breaks, and removals are therefore identical, and this
backend returns the same cover as the linked one.
*/
static void greedy_from_lagr_scores_deferred(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights, /* weights can be NULL */
    int *sol,
    int *sol_len,
    LagrangianWorkspace *workspace
) {
    unsigned char *rowCovered = workspace->row_covered;
    memset(rowCovered, 0, (size_t)rows);
    /*
     * One 64-bit word per column packs the residual degree in the high 32
     * bits and the arrival stamp in the low 32 bits, so covering a row
     * costs a single load and a single store per covering column. The
     * stamp clock stays below 2^32 per call: it advances at most once per
     * chart incidence, and the configured memory envelope bounds the total
     * incidence count below that.
     */
    uint64_t *levelstamp = workspace->levelstamp;
    uint64_t *buffer = workspace->bucket;
    int covered = 0, out = 0;

    for (int c = 0; c < cols; ++c) {
        levelstamp[c] =
            ((uint64_t)(uint32_t)rowsCoveredCount[c] << 32) |
            (uint64_t)(uint32_t)c;
    }
    uint64_t clock = (uint64_t)cols;

    int top = 0;
    int buffer_size = 0;

    while (covered < rows) {
        if (buffer_size <= 0) {
            /* rebuild: collect columns at the current maximum level */
            top = 0;
            for (int c = 0; c < cols; ++c) {
                int lvl = (int)(levelstamp[c] >> 32);
                if (lvl > top) top = lvl;
            }
            if (top <= 0) { /* no column can newly cover any row */
                *sol_len = -1;
                return;
            }
            int n = 0;
            for (int c = 0; c < cols; ++c) {
                if ((int)(levelstamp[c] >> 32) == top) {
                    /* sort key: descending arrival stamp, then index */
                    buffer[n++] =
                        ((levelstamp[c] & UINT64_C(0xFFFFFFFF)) << 32) |
                        (uint64_t)(uint32_t)c;
                }
            }
            qsort(
                buffer,
                (size_t)n,
                sizeof(uint64_t),
                lagr_compare_stamp_desc
            );
            buffer_size = n;
        }

        int best = -1;
        double bestLS = DBL_MAX;
        double bestW = -DBL_MAX;
        int live = 0;
        for (int i = 0; i < buffer_size; ++i) {
            int c = (int)(buffer[i] & UINT64_C(0xFFFFFFFF));
            if ((int)(levelstamp[c] >> 32) != top) continue;
            ++live;
            double ls = lagr_score ? lagr_score[c] : 0.0;
            double w = weights ? weights[c] : 0.0;
            bool better = false;
            if (best == -1) {
                better = true;
            } else if (ls < bestLS) {
                better = true;
            } else if (fabs(ls - bestLS) <= EPS) {
                if (w > bestW) {
                    better = true;
                } else if (fabs(w - bestW) <= EPS && c < best) {
                    better = true;
                }
            }
            if (better) {
                best = c;
                bestLS = ls;
                bestW = w;
            }
        }

        if (best < 0 || live <= 0) {
            /* top level exhausted: descend and rebuild */
            buffer_size = 0;
            continue;
        }

        levelstamp[best] = 0;
        sol[out++] = best;

        for (int k = 0; k < rowsCoveredCount[best]; ++k) {
            int r = rowsCovered[best][k];
            if (rowCovered[r]) continue;
            rowCovered[r] = 1;
            ++covered;

            const int *list = colsCovering[r];
            for (int i = 0; i < colsCoveringCount[r]; ++i) {
                int c = list[i];
                uint64_t v = levelstamp[c];
                uint32_t level = (uint32_t)(v >> 32);
                if (level == 0) continue;
                levelstamp[c] =
                    ((uint64_t)(level - 1) << 32) |
                    (uint64_t)(uint32_t)++clock;
            }
        }

        /* compact departed entries while preserving scan order */
        if (buffer_size - live > live && live > 0) {
            int n = 0;
            for (int i = 0; i < buffer_size; ++i) {
                int c = (int)(buffer[i] & UINT64_C(0xFFFFFFFF));
                if ((int)(levelstamp[c] >> 32) == top) {
                    buffer[n++] =
                        ((levelstamp[c] & UINT64_C(0xFFFFFFFF)) << 32) |
                        (uint64_t)(uint32_t)c;
                }
            }
            buffer_size = n;
        }
    }

    *sol_len = out;
    prune_redundancy_with_counts(rows, rowsCovered, rowsCoveredCount, sol, sol_len, workspace->cover_count);
}

/*
Compute Lagrangian scores:
rc_c = 1.0 - sum_{r in rowsCovered_c} t[r]
ZLB = sum_i t_i + sum_c min(0, rc_c)
*/
static double compute_reduced_and_lb(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    const double *t,
    const double *col_costs, // can be NULL
    double *ls, /* out */
    unsigned char *x, /* optional sign flags for subgradient */
    int threads
) {
    double ZLB = 0.0;
    double ZLB_rows = 0.0;
    double ZLB_neg = 0.0;

    const int bucket_count = lagr_reduction_bucket_count(cols);
    double partials[LAGRANGIAN_REDUCTION_BUCKET_LIMIT];

    for (int i = 0; i < rows; ++i) ZLB_rows += t[i];

    lagr_reduced_context reduced_ctx = {
        .rowsCovered = rowsCovered,
        .rowsCoveredCount = rowsCoveredCount,
        .t = t,
        .ls = ls,
        .partials = partials,
        .col_costs = col_costs,
        .x = x,
        .cols = cols,
        .bucket_count = bucket_count
    };
    static _Thread_local ccubes_adaptive_team teams[65];
    unsigned key = 0;
    for (uint64_t work = lagr_incidence_work + (uint64_t)cols; work; work >>= 1) ++key;
    int workers = lagr_run_reduction(cols, threads, lagr_reduced_worker, &reduced_ctx, &teams[key]);
    lagr_record_team(lagr_calibration_stats ? &lagr_calibration_stats->reduced_cost_teams
        : &lagr_last_stats.reduced_cost_teams, workers);
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
        uint64_t first, last;
        lagr_reduction_bucket_bounds(cols, bucket_count, bucket, &first, &last);
        double neg = 0.0;
        for (uint64_t col = first; col < last; ++col)
            if (ls[col] < 0.0) neg += ls[col];
        partials[bucket] = neg;
    }
    for (int i = 0; i < bucket_count; ++i) {
        ZLB_neg += partials[i];
    }

    ZLB = ZLB_rows + ZLB_neg;
    return ZLB;
}

/*
A "slack" measures how far a constraint is from being tight.
Positive slack means the constraint is not yet satisfied, zero slack means tight, negative means surplus.
In this code, a slack is the per-row violation of the covering constraint under the current Lagrangian/primal guess
- Constraint per row i: sum over columns covering i of x_c >= 1
- Slack definition: slack_i = 1 − Σ_{c covers i} x_c
- slack_i > 0: row i is under-covered (violation)
- slack_i = 0: row i is exactly satisfied
- slack_i < 0: row i is over-covered

How it’s used:
- Build a tentative primal x by taking columns with negative lagr_score (x_c = 1 if rc_c < 0, else 0).
- The vector slack is exactly the subgradient of the dual Lagrangian at the current multipliers t.
- Step size uses (UB − ZLB) / Σ slack_i^2, and multipliers update as t_i := max(0, t_i + step · slack_i).
- If all slack_i are zero, there’s no subgradient direction to update (no progress possible).

Subgradient update:
x_c = 1 if rc_c < 0 else 0 (LP solution)
slack_i = 1 - sum_{c covering i} x_c
step = step_coef * (UB - ZLB) / sum(slack^2), t := max(0, t + step*slack)
Returns 1 if step applied, 0 if slacks zero or no progress possible.
*/

static int subgradient_update(
    int rows,
    int **colsCovering,
    int *colsCoveringCount,
    const unsigned char *x,
    double UB, double ZLB,
    double *t,
    double *step_coef,
    double step_min,
    int *stuck_iter,
    int stuck_halve_period,
    double *prev_direction,
    double deflection_alpha,
    double step_contract,
    double stabilization_beta,
    int *nonzero_rows_out,
    LagrangianWorkspace *workspace,
    int threads
) {
    if (!x) return 0;

    double *slack = workspace->slack;
    double sum_s2 = lagr_compute_slacks(
        rows, colsCovering, colsCoveringCount, x, slack, nonzero_rows_out, threads
    );

    if (deflection_alpha > 0.0 && prev_direction) {
        sum_s2 = 0.0;
        for (int i = 0; i < rows; ++i) {
            slack[i] += deflection_alpha * prev_direction[i];
            sum_s2 += slack[i] * slack[i];
        }
    }

    if (sum_s2 <= 1e-15) {
        return 0;
    }

    double gap = UB - ZLB;
    if (gap <= 0.0) {
        return 0;
    }

    double step = (*step_coef) * (gap / sum_s2);
    if (step <= 0.0) {
        return 0;
    }

    for (int i = 0; i < rows; ++i) {
        double trial = t[i] + step * slack[i];
        if (trial < 0.0) trial = 0.0;
        double val = (stabilization_beta >= 1.0)
            ? trial
            : ((1.0 - stabilization_beta) * t[i] + stabilization_beta * trial);
        t[i] = (val > 0.0) ? val : 0.0;
    }

    if (deflection_alpha > 0.0 && prev_direction) {
        memcpy(prev_direction, slack, (size_t)rows * sizeof(double));
    }

    (*stuck_iter)++;
    if (*stuck_iter >= stuck_halve_period) {
        *stuck_iter = 0;
        *step_coef *= step_contract;
        if (*step_coef < step_min) *step_coef = step_min;
    }
    return 1;
}

typedef struct {
    int rows;
    int memory;
    int count;
    int head;
    double prox_mu;
    double min_mu;
    double max_mu;
    double null_expand;
    double serious_shrink;
    double serious_fraction;
    double serious_tol;
    double momentum;
    double center_value;
    double predicted_gain;
    double *center_t;
    double *prev_center_t;
    double *cut_alpha;
    double *cut_slacks;
    int null_streak;
    int serious_streak;
    int has_trial;
} LagrangianBundleState;

static void bundle_project_simplex(double *v, int n) {
    if (!v || n <= 0) return;

    double *u = (double*)malloc((size_t)n * sizeof(double));
    if (!u) {
        double uniform = 1.0 / (double)n;
        for (int i = 0; i < n; ++i) v[i] = uniform;
        return;
    }

    memcpy(u, v, (size_t)n * sizeof(double));
    for (int i = 1; i < n; ++i) {
        double key = u[i];
        int j = i - 1;
        while (j >= 0 && u[j] < key) {
            u[j + 1] = u[j];
            --j;
        }
        u[j + 1] = key;
    }

    double cssv = 0.0;
    int rho = 0;
    for (int i = 0; i < n; ++i) {
        cssv += u[i];
        double theta = (cssv - 1.0) / (double)(i + 1);
        if (u[i] - theta > 0.0) rho = i + 1;
    }

    cssv = 0.0;
    for (int i = 0; i < rho; ++i) cssv += u[i];
    double theta = rho > 0 ? (cssv - 1.0) / (double)rho : 0.0;

    for (int i = 0; i < n; ++i) {
        double projected = v[i] - theta;
        v[i] = projected > 0.0 ? projected : 0.0;
    }

    free(u);
}

static int bundle_update(
    int rows,
    int **colsCovering,
    int *colsCoveringCount,
    const unsigned char *x,
    double UB,
    double ZLB,
    double *t,
    LagrangianBundleState *bs,
    int *nonzero_rows_out,
    int threads
) {
    if (!x || !bs || bs->memory <= 0 || !bs->center_t || !bs->cut_alpha || !bs->cut_slacks) {
        return 0;
    }

    double *slack = (double*)malloc((size_t)rows * sizeof(double));
    if (!slack) return 0;

    double sum_s2 = lagr_compute_slacks(
        rows, colsCovering, colsCoveringCount, x, slack, nonzero_rows_out, threads
    );

    if (sum_s2 <= 1e-15 || UB - ZLB <= 0.0) {
        free(slack);
        return 0;
    }

    if (bs->center_value <= -DBL_MAX / 2.0) {
        memcpy(bs->center_t, t, (size_t)rows * sizeof(double));
        if (bs->prev_center_t) {
            memcpy(bs->prev_center_t, t, (size_t)rows * sizeof(double));
        }
        bs->center_value = ZLB;
        bs->has_trial = 0;
    } else if (bs->has_trial) {
        double actual_gain = ZLB - bs->center_value;
        if (actual_gain > bs->serious_tol) {
            if (bs->prev_center_t) {
                memcpy(bs->prev_center_t, bs->center_t, (size_t)rows * sizeof(double));
            }
            memcpy(bs->center_t, t, (size_t)rows * sizeof(double));
            bs->center_value = ZLB;
            bs->prox_mu = fmax(bs->min_mu, bs->prox_mu * bs->serious_shrink);
            bs->serious_streak++;
            bs->null_streak = 0;
        } else {
            bs->prox_mu = fmin(bs->max_mu, bs->prox_mu * bs->null_expand);
            bs->null_streak++;
            bs->serious_streak = 0;
        }
        bs->has_trial = 0;
    } else if (ZLB > bs->center_value + bs->serious_tol) {
        if (bs->prev_center_t) {
            memcpy(bs->prev_center_t, bs->center_t, (size_t)rows * sizeof(double));
        }
        memcpy(bs->center_t, t, (size_t)rows * sizeof(double));
        bs->center_value = ZLB;
        bs->serious_streak++;
        bs->null_streak = 0;
    } else {
        bs->prox_mu = fmin(bs->max_mu, bs->prox_mu * bs->null_expand);
        bs->null_streak++;
        bs->serious_streak = 0;
    }

    double dot_gt = 0.0;
    for (int r = 0; r < rows; ++r) dot_gt += slack[r] * t[r];
    double alpha = -ZLB + dot_gt;

    if (bs->count < bs->memory) {
        int slot = (bs->head + bs->count) % bs->memory;
        memcpy(bs->cut_slacks + (size_t)slot * (size_t)rows, slack, (size_t)rows * sizeof(double));
        bs->cut_alpha[slot] = alpha;
        bs->count++;
    } else {
        int slot = bs->head;
        memcpy(bs->cut_slacks + (size_t)slot * (size_t)rows, slack, (size_t)rows * sizeof(double));
        bs->cut_alpha[slot] = alpha;
        bs->head = (bs->head + 1) % bs->memory;
    }

    int m = bs->count;
    if (m <= 0 || (bs->prox_mu >= bs->max_mu - 1e-12 && bs->null_streak >= bs->memory * 8)) {
        free(slack);
        return 0;
    }

    double *base_t = (double*)malloc((size_t)rows * sizeof(double));
    double *linear = (double*)malloc((size_t)m * sizeof(double));
    double *gram = (double*)calloc((size_t)m * (size_t)m, sizeof(double));
    double *lambda = (double*)malloc((size_t)m * sizeof(double));
    double *gradient = (double*)malloc((size_t)m * sizeof(double));
    double *gbar = (double*)calloc((size_t)rows, sizeof(double));
    if (!base_t || !linear || !gram || !lambda || !gradient || !gbar) {
        free(base_t);
        free(linear);
        free(gram);
        free(lambda);
        free(gradient);
        free(gbar);
        free(slack);
        return 0;
    }

    for (int r = 0; r < rows; ++r) {
        double base = bs->center_t[r];
        if (bs->momentum > 0.0 && bs->prev_center_t && bs->serious_streak > 0) {
            base += bs->momentum * (bs->center_t[r] - bs->prev_center_t[r]);
        }
        base_t[r] = base > 0.0 ? base : 0.0;
    }

    int start = bs->head;
    for (int i = 0; i < m; ++i) {
        int slot_i = (start + i) % bs->memory;
        const double *g_i = bs->cut_slacks + (size_t)slot_i * (size_t)rows;
        double dot_center = 0.0;
        for (int r = 0; r < rows; ++r) dot_center += g_i[r] * base_t[r];
        linear[i] = bs->cut_alpha[slot_i] - dot_center;

        for (int j = i; j < m; ++j) {
            int slot_j = (start + j) % bs->memory;
            const double *g_j = bs->cut_slacks + (size_t)slot_j * (size_t)rows;
            double dot = 0.0;
            for (int r = 0; r < rows; ++r) dot += g_i[r] * g_j[r];
            gram[i * m + j] = dot;
            gram[j * m + i] = dot;
        }
    }

    double max_row_sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < m; ++j) row_sum += fabs(gram[i * m + j]);
        if (row_sum > max_row_sum) max_row_sum = row_sum;
    }

    double pg_step = 1.0 / ((max_row_sum / bs->prox_mu) + 1e-9);
    for (int i = 0; i < m; ++i) lambda[i] = 1.0 / (double)m;

    for (int iter = 0; iter < 80; ++iter) {
        for (int i = 0; i < m; ++i) {
            double gram_lambda = 0.0;
            for (int j = 0; j < m; ++j) gram_lambda += gram[i * m + j] * lambda[j];
            gradient[i] = linear[i] - gram_lambda / bs->prox_mu;
        }

        for (int i = 0; i < m; ++i) lambda[i] += pg_step * gradient[i];
        bundle_project_simplex(lambda, m);
    }

    for (int i = 0; i < m; ++i) {
        int slot = (start + i) % bs->memory;
        const double *g_i = bs->cut_slacks + (size_t)slot * (size_t)rows;
        for (int r = 0; r < rows; ++r) gbar[r] += lambda[i] * g_i[r];
    }

    for (int r = 0; r < rows; ++r) {
        double next = base_t[r] + gbar[r] / bs->prox_mu;
        t[r] = next > 0.0 ? next : 0.0;
    }

    double model_f = -DBL_MAX;
    for (int i = 0; i < m; ++i) {
        int slot = (start + i) % bs->memory;
        const double *g_i = bs->cut_slacks + (size_t)slot * (size_t)rows;
        double cut = bs->cut_alpha[slot];
        for (int r = 0; r < rows; ++r) cut -= g_i[r] * t[r];
        if (cut > model_f) model_f = cut;
    }

    double predicted_gain = -bs->center_value - model_f;
    if (predicted_gain <= bs->serious_tol) {
        free(base_t);
        free(linear);
        free(gram);
        free(lambda);
        free(gradient);
        free(gbar);
        free(slack);
        return 0;
    }

    bs->predicted_gain = predicted_gain;
    bs->has_trial = 1;

    free(base_t);
    free(linear);
    free(gram);
    free(lambda);
    free(gradient);
    free(gbar);
    free(slack);
    return 1;
}

/* A strict-score order is unique when every selected score is finite and
 * distinct. Use a heap in that case. Exact ties must retain the legacy
 * interchange sort: its nonadjacent swaps can reorder ties indirectly. */
static void lagr_removal_sift_down(
    int *order, const double *scores, int root, int count
) {
    while (root < count / 2) {
        int child = root * 2 + 1;
        if (child + 1 < count && scores[order[child + 1]] < scores[order[child]]) {
            child++;
        }
        if (!(scores[order[child]] < scores[order[root]])) break;
        int tmp = order[root]; order[root] = order[child]; order[child] = tmp;
        root = child;
    }
}

static void lagr_sort_removal_order(int *order, const double *scores, int count) {
    bool strict_order = count >= 32;
    for (int i = 0; strict_order && i < count; ++i) {
        if (!isfinite(scores[i])) strict_order = false;
    }
    if (strict_order) {
        for (int root = count / 2; root > 0; --root) {
            lagr_removal_sift_down(order, scores, root - 1, count);
        }
        for (int end = count - 1; end > 0; --end) {
            int tmp = order[0]; order[0] = order[end]; order[end] = tmp;
            lagr_removal_sift_down(order, scores, 0, end);
        }
        for (int i = 1; i < count; ++i) {
            if (scores[order[i - 1]] == scores[order[i]]) {
                strict_order = false;
                break;
            }
        }
        if (strict_order) return;
        for (int i = 0; i < count; ++i) order[i] = i;
    }
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (scores[order[j]] > scores[order[i]]) {
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }
}

/*
Row-targeted heuristic: for each uncovered row, pick a covering column with minimal Lagrangian score.
Tie order:
(1) more negative Lagrangian score
(2) higher weight
(3) covers more rows
(4) lower index.
*/
static void heuristic_row_min_rc(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights, /* tie-break: higher is better */
    int *sol, int *sol_len,
    LagrangianWorkspace *workspace
) {
    (void)colsCoveringCount;

    unsigned char *rowCovered = workspace->row_covered;
    unsigned char *x = workspace->selected;
    memset(rowCovered, 0, (size_t)rows);
    memset(x, 0, (size_t)cols);
    int out = 0, covered = 0;

    for (int i = 0; i < rows; ++i) {
        if (rowCovered[i]) continue;

        /* choose candidate among columns covering row i */
        int best = -1;
        double bestLS = DBL_MAX;
        int bestOnes = -1;
        double bestWeight = -DBL_MAX;

        for (int k = 0; k < colsCoveringCount[i]; ++k) {
            int c = colsCovering[i][k];
            double ls = lagr_score ? lagr_score[c] : 0.0;
            double w  = weights ? weights[c] : 0.0;
            int ones = rowsCoveredCount[c];

            bool better = false;
            if (ls < bestLS) {
                better = true;
            } else if (fabs(ls - bestLS) <= EPS) {
                if (w > bestWeight) {
                    better = true;
                } else if (fabs(w - bestWeight) <= EPS) {
                    if (ones > bestOnes) {
                        better = true;
                    } else if (ones == bestOnes) {
                        if (best == -1 || c < best) {
                            better = true;
                        }
                    }
                }
            }
            if (better) {
                best = c;
                bestLS = ls;
                bestOnes = ones;
                bestWeight = w;
            }
        }

        if (best == -1) { /* should not happen if adjacency is valid */
            *sol_len = -1;
            return;
        }

        if (!x[best]) {
            x[best] = 1;
            sol[out++] = best;
            for (int c = 0; c < rowsCoveredCount[best]; ++c) {
                int r = rowsCovered[best][c];
                if (!rowCovered[r]) {
                    rowCovered[r] = true;
                    covered++;
                }
            }
        }

        if (covered >= rows) break;
    }

    *sol_len = out;

    /* Removal: try to drop columns in order of worst Lagrangian score first */
    if (out > 0) {
        /* Build cover counts */
        int *coverCount = workspace->cover_count;
        int *order = workspace->order;
        double *ls_sel = workspace->selected_scores;
        memset(coverCount, 0, (size_t)rows * sizeof(int));

        if (coverCount && order && ls_sel) {
            for (int i = 0; i < out; ++i) {
                int s = sol[i];
                order[i] = i; /* index into sol */
                ls_sel[i] = lagr_score ? lagr_score[s] : 0.0;
                for (int c = 0; c < rowsCoveredCount[s]; ++c) {
                    int r = rowsCovered[s][c];
                    coverCount[r]++;
                }
            }
            /* sort order by score descending (worst first) */
            lagr_sort_removal_order(order, ls_sel, out);
            /* attempt removals */
            for (int o = 0; o < out; ++o) {
                int idx = order[o];
                if (idx < 0) continue;
                int s = sol[idx];
                bool can_remove = true;

                for (int c = 0; c < rowsCoveredCount[s]; ++c) {
                    int r = rowsCovered[s][c];
                    if (coverCount[r] <= 1) {
                        can_remove = false;
                        break;
                    }
                }

                if (can_remove) {
                    for (int c = 0; c < rowsCoveredCount[s]; ++c) {
                        int r = rowsCovered[s][c];
                        coverCount[r]--;
                    }

                    /* Keep original positions until all attempts finish.
                     * The shrinking `o < out` bound is intentional: it is
                     * the existing stopping rule, including after removals. */
                    x[s] = 0;
                    out--;
                }
            }
            int write = 0;
            for (int i = 0; i < *sol_len; ++i) {
                if (x[sol[i]]) sol[write++] = sol[i];
            }
            *sol_len = out;
        }
    }

    prune_redundancy_with_counts(rows, rowsCovered, rowsCoveredCount, sol, sol_len, workspace->cover_count);
}

static double solution_total_weight(const int *sol, int sol_len, const double *weights) {
    if (!sol || sol_len <= 0 || !weights) return 0.0;
    double s = 0.0;
    for (int i = 0; i < sol_len; ++i) s += weights[sol[i]];
    return s;
}

static void lagr_initialize_multipliers(
    int rows,
    const int *colsCoveringCount,
    int rarity_init,
    double *t
) {
    if (!t || rows <= 0) return;

    if (!rarity_init || !colsCoveringCount) {
        for (int i = 0; i < rows; ++i) {
            t[i] = 1.0;
        }
        return;
    }

    double mean_support = 0.0;
    for (int i = 0; i < rows; ++i) {
        mean_support += (double)colsCoveringCount[i];
    }
    mean_support /= (double)rows;
    if (mean_support <= 0.0) mean_support = 1.0;

    for (int i = 0; i < rows; ++i) {
        double support = (double)colsCoveringCount[i];
        if (support <= 0.0) support = 1.0;
        t[i] = mean_support / support;
        if (t[i] < 0.1) t[i] = 0.1;
    }
}

/*
 * Build a dual-feasible starting point for restricted-master pricing.
 *
 * For row i, let d_i be the largest cardinality of a retained column that
 * covers i and set t_i = 1 / d_i. For every retained column c and every row
 * i in c, d_i >= |c|, hence sum(i in c) t_i <= 1. Starting pricing from this
 * normalized vector avoids the very large negative Lagrangian values caused
 * by the legacy all-ones primal-heuristic initialization on dense charts.
 */
typedef struct {
    int row;
    int covering_columns;
} LagrangianRowDegree;

static int lagr_compare_row_degree(
    const void *left,
    const void *right
) {
    const LagrangianRowDegree *a =
        (const LagrangianRowDegree *)left;
    const LagrangianRowDegree *b =
        (const LagrangianRowDegree *)right;
    if (a->covering_columns != b->covering_columns) {
        return (a->covering_columns > b->covering_columns) -
            (a->covering_columns < b->covering_columns);
    }
    return (a->row > b->row) - (a->row < b->row);
}

static void lagr_initialize_pricing_multipliers(
    int rows,
    int cols,
    const int *rowsCoveredCount,
    int **colsCovering,
    const int *colsCoveringCount,
    double *t
) {
    if (
        rows <= 0 ||
        cols <= 0 ||
        !rowsCoveredCount ||
        !colsCovering ||
        !colsCoveringCount ||
        !t
    ) {
        return;
    }
    double normalized_value = 0.0;
    for (int row = 0; row < rows; ++row) {
        int maximum_cardinality = 1;
        for (int index = 0; index < colsCoveringCount[row]; ++index) {
            int column = colsCovering[row][index];
            if (rowsCoveredCount[column] > maximum_cardinality) {
                maximum_cardinality = rowsCoveredCount[column];
            }
        }
        t[row] = 1.0 / (double)maximum_cardinality;
        normalized_value += t[row];
    }

    double *ascent = (double *)calloc(
        (size_t)rows,
        sizeof(double)
    );
    double *residual = (double *)malloc(
        (size_t)cols * sizeof(double)
    );
    LagrangianRowDegree *order =
        (LagrangianRowDegree *)malloc(
            (size_t)rows * sizeof(LagrangianRowDegree)
        );
    if (!ascent || !residual || !order) {
        free(ascent);
        free(residual);
        free(order);
        return;
    }
    for (int column = 0; column < cols; ++column) {
        residual[column] = 1.0;
    }
    for (int row = 0; row < rows; ++row) {
        order[row] = (LagrangianRowDegree){
            .row = row,
            .covering_columns = colsCoveringCount[row]
        };
    }
    qsort(
        order,
        (size_t)rows,
        sizeof(LagrangianRowDegree),
        lagr_compare_row_degree
    );

    double ascent_value = 0.0;
    for (int rank = 0; rank < rows; ++rank) {
        const int row = order[rank].row;
        double delta = DBL_MAX;
        for (int index = 0; index < colsCoveringCount[row]; ++index) {
            const int column = colsCovering[row][index];
            if (residual[column] < delta) {
                delta = residual[column];
            }
        }
        if (delta == DBL_MAX || delta <= EPS) continue;
        ascent[row] = delta;
        ascent_value += delta;
        for (int index = 0; index < colsCoveringCount[row]; ++index) {
            const int column = colsCovering[row][index];
            residual[column] -= delta;
            if (residual[column] < EPS) residual[column] = 0.0;
        }
    }
    if (ascent_value > normalized_value + EPS) {
        memcpy(t, ascent, (size_t)rows * sizeof(double));
    }
    free(ascent);
    free(residual);
    free(order);
}

bool lagrangian_initialize_pricing_multipliers(
    const PIChartView *chart,
    double *row_multipliers,
    int multiplier_count
) {
    if (
        !chart ||
        !chart->bits ||
        chart->rows <= 0 ||
        chart->cols <= 0 ||
        !row_multipliers ||
        multiplier_count < chart->rows
    ) {
        return false;
    }
    int **rowsCovered = NULL;
    int *rowsCoveredCount = NULL;
    int **colsCovering = NULL;
    int *colsCoveringCount = NULL;
    if (build_adjacency(
        chart,
        chart->rows,
        chart->cols,
        &rowsCovered,
        &rowsCoveredCount,
        &colsCovering,
        &colsCoveringCount
    ) != 0) {
        return false;
    }
    lagr_initialize_pricing_multipliers(
        chart->rows,
        chart->cols,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        row_multipliers
    );
    bool valid = true;
    for (int row = 0; row < chart->rows; ++row) {
        if (
            !isfinite(row_multipliers[row]) ||
            row_multipliers[row] < 0.0
        ) {
            valid = false;
            break;
        }
    }
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        chart->cols,
        colsCovering,
        colsCoveringCount,
        chart->rows
    );
    return valid;
}

/* Proxy for acceptance: if weights exist, use their sum; else use coverage volume as a cheap heuristic */
static double solution_proxy_value(
    const int *sol,
    int sol_len,
    const double *weights,
    const int *rowsCoveredCount
) {
    if (!sol || sol_len <= 0) return 0.0;
    double v = 0.0;
    if (weights) {
        for (int i = 0; i < sol_len; ++i) v += weights[sol[i]];
    } else if (rowsCoveredCount) {
        for (int i = 0; i < sol_len; ++i) v += (double)rowsCoveredCount[sol[i]];
    }
    return v;
}

/* Local search: try drop-1 + greedy repair + prune; accept only if shorter */
static void local_search_drop1_repair(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights,
    int *sol,
    int *sol_len,
    int max_passes
) {
    if (!sol || !sol_len || *sol_len <= 1) return;
    if (max_passes <= 0) return;

    const double PROXY_EPS = 1e-12;
    double curr_proxy = solution_proxy_value(sol, *sol_len, weights, rowsCoveredCount);

    bool *rowCovered = (bool*)malloc((size_t)rows * sizeof(bool));
    unsigned char *colSelected = (unsigned char*)calloc((size_t)cols, 1);
    int *cand = (int*)malloc((size_t)cols * sizeof(int));
    if (!rowCovered || !colSelected || !cand) {
        free(rowCovered);
        free(colSelected);
        free(cand);
        return;
    }

    for (int pass = 0; pass < max_passes; ++pass) {
        int improved = 0;

        for (int drop_idx = 0; drop_idx < *sol_len; ++drop_idx) {
            int cand_len = 0;
            memset(colSelected, 0, (size_t)cols);

            for (int i = 0; i < *sol_len; ++i) {
                if (i == drop_idx) continue;
                int c = sol[i];
                cand[cand_len++] = c;
                colSelected[c] = 1;
            }

            memset(rowCovered, 0, (size_t)rows * sizeof(bool));
            int covered = 0;
            for (int i = 0; i < cand_len; ++i) {
                int c = cand[i];
                for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                    int r = rowsCovered[c][k];
                    if (!rowCovered[r]) { rowCovered[r] = true; covered++; }
                }
            }

            while (covered < rows) {
                int r0 = -1;
                for (int r = 0; r < rows; ++r) {
                    if (!rowCovered[r]) { r0 = r; break; }
                }
                if (r0 < 0) break;

                int best = -1, bestNew = -1;
                double bestLS = DBL_MAX, bestW = -DBL_MAX;

                for (int kk = 0; kk < colsCoveringCount[r0]; ++kk) {
                    int c = colsCovering[r0][kk];
                    if (colSelected[c]) continue;

                    int newCover = 0;
                    for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                        int rr = rowsCovered[c][k];
                        if (!rowCovered[rr]) newCover++;
                    }
                    if (newCover <= 0) continue;

                    double ls = lagr_score ? lagr_score[c] : 0.0;
                    double w  = weights ? weights[c] : 0.0;

                    bool better = false;
                    if (newCover > bestNew) better = true;
                    else if (newCover == bestNew) {
                        if (ls < bestLS) better = true;
                        else if (fabs(ls - bestLS) <= EPS) {
                            if (w > bestW) better = true;
                            else if (fabs(w - bestW) <= EPS) {
                                if (best == -1 || c < best) better = true;
                            }
                        }
                    }

                    if (better) {
                        best = c; bestNew = newCover; bestLS = ls; bestW = w;
                    }
                }

                if (best < 0) { cand_len = -1; break; }

                colSelected[best] = 1;
                cand[cand_len++] = best;
                for (int k = 0; k < rowsCoveredCount[best]; ++k) {
                    int rr = rowsCovered[best][k];
                    if (!rowCovered[rr]) { rowCovered[rr] = true; covered++; }
                }
            }

            if (cand_len <= 0 || cand_len > *sol_len) continue;

            prune_redundancy(rows, rowsCovered, rowsCoveredCount, cand, &cand_len);

            if (cand_len > 0 && cand_len <= *sol_len) {
                double cand_proxy = solution_proxy_value(cand, cand_len, weights, rowsCoveredCount);
                /* Accept if proxy strictly improves, even for equal length (swap), or for shorter */
                if (cand_proxy > curr_proxy + PROXY_EPS) {
                    memcpy(sol, cand, (size_t)cand_len * sizeof(int));
                    *sol_len = cand_len;
                    curr_proxy = cand_proxy;
                    improved = 1;
                    break; /* restart after improvement */
                }
            }
        }

        if (!improved) break;
    }

    free(rowCovered);
    free(colSelected);
    free(cand);
}

/* A two-column drop only improves if at most one replacement is needed.
   The first two owners suffice to identify rows left uncovered by a pair. */
typedef struct {
    int first;
    int second;
} LagrDropOwners;

typedef struct {
    int rows;
    int size;
    const int *solution;
    const LagrDropOwners *owners;
    int **rowsCovered;
    const int *rowsCoveredCount;
    int **colsCovering;
    const int *colsCoveringCount;
    const double *lagr_score;
    const double *weights;
    uint64_t deadline;
    atomic_uint_fast64_t next;
    atomic_uint_fast64_t winner;
} LagrDrop2;

static bool lagr_drop2_missing(LagrDropOwners owners, int i, int j) {
    return (owners.first == i && (owners.second == -1 || owners.second == j)) ||
        (owners.first == j && owners.second == -1);
}

/* -1: no improvement; -2: no replacement needed; otherwise the column ID. */
static int lagr_drop2_replacement(const LagrDrop2 *search, int i, int j, bool rank) {
    int missing = 0;
    int first_row = search->rows;
    int left = search->solution[i];
    int right = search->solution[j];
    for (int k = 0; k < search->rowsCoveredCount[left]; ++k) {
        const int row = search->rowsCovered[left][k];
        if (!lagr_drop2_missing(search->owners[row], i, j)) continue;
        ++missing;
        if (row < first_row) first_row = row;
    }
    for (int k = 0; k < search->rowsCoveredCount[right]; ++k) {
        const int row = search->rowsCovered[right][k];
        LagrDropOwners owners = search->owners[row];
        if (owners.first != j || owners.second != -1) continue;
        ++missing;
        if (row < first_row) first_row = row;
    }
    if (missing == 0) return -2;

    int best = -1;
    double best_score = DBL_MAX, best_weight = -DBL_MAX;
    for (int k = 0; k < search->colsCoveringCount[first_row]; ++k) {
        int column = search->colsCovering[first_row][k];
        int degree = search->rowsCoveredCount[column];
        if (degree < missing) continue;
        int hits = 0;
        for (int entry = 0; entry < degree; ++entry) {
            int row = search->rowsCovered[column][entry];
            hits += lagr_drop2_missing(search->owners[row], i, j);
        }
        if (hits != missing) continue;
        if (!rank) return column;
        const double score = search->lagr_score ? search->lagr_score[column] : 0.0;
        const double weight = search->weights ? search->weights[column] : 0.0;
        if (best < 0 || score < best_score ||
            (fabs(score - best_score) <= EPS &&
             (weight > best_weight || (fabs(weight - best_weight) <= EPS && column < best)))) {
            best = column;
            best_score = score;
            best_weight = weight;
        }
    }
    return best;
}

static uint64_t lagr_drop2_offset(int size, int i) {
    return (uint64_t)i * (2 * (uint64_t)size - i - 1) / 2;
}

static int lagr_drop2_first(int size, uint64_t pair) {
    int lo = 0, hi = size - 1;
    while (hi - lo > 1) {
        const int mid = lo + (hi - lo) / 2;
        if (lagr_drop2_offset(size, mid) <= pair) lo = mid;
        else hi = mid;
    }
    return lo;
}

static void lagr_drop2_worker(
    uint64_t start, uint64_t end, uint64_t stride, int id, int workers, void *data
) {
    (void)start; (void)end; (void)stride; (void)id; (void)workers;
    LagrDrop2 *search = data;
    const uint64_t chunk = 32;
    for (;;) {
        if (search->deadline && lagr_monotonic_nanoseconds() >= search->deadline) return;
        uint64_t pair = atomic_fetch_add_explicit(&search->next, chunk, memory_order_relaxed);
        if (pair >= atomic_load_explicit(&search->winner, memory_order_relaxed)) return;
        const uint64_t limit = pair + chunk;
        int i = lagr_drop2_first(search->size, pair);
        int j = i + 1 + (int)(pair - lagr_drop2_offset(search->size, i));
        for (; pair < limit; ++pair) {
            uint_fast64_t winner = atomic_load_explicit(&search->winner, memory_order_relaxed);
            if (pair >= winner) return;
            if (lagr_drop2_replacement(search, i, j, false) != -1) {
                /* All earlier chunks have already been claimed. They may
                   still find an earlier move before the team joins. */
                while (pair < winner && !atomic_compare_exchange_weak_explicit(
                    &search->winner, &winner, pair, memory_order_relaxed, memory_order_relaxed
                )) {}
                return;
            }
            if (++j == search->size) j = ++i + 1;
        }
    }
}

static void local_search_drop2_repair(
    int rows,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights,
    int *sol,
    int *sol_len,
    int max_passes,
    uint64_t deadline
) {
    if (!sol || !sol_len || *sol_len <= 2 || max_passes <= 0) return;
    LagrDropOwners *owners = malloc((size_t)rows * sizeof(*owners));
    if (!owners) return;
    LagrDrop2 search = {
        .rows = rows, .solution = sol, .owners = owners, .rowsCovered = rowsCovered,
        .rowsCoveredCount = rowsCoveredCount, .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount, .lagr_score = lagr_score, .weights = weights,
        .deadline = deadline
    };
    atomic_init(&search.next, 0);
    atomic_init(&search.winner, 0);

    for (int pass = 0; pass < max_passes && *sol_len > 1; ++pass) {
        if (deadline && lagr_monotonic_nanoseconds() >= deadline) break;
        int size = *sol_len;
        for (int row = 0; row < rows; ++row) owners[row] = (LagrDropOwners){-1, -1};
        for (int i = 0; i < size; ++i) {
            const int col = sol[i];
            for (int k = 0; k < rowsCoveredCount[col]; ++k) {
                LagrDropOwners *row = &owners[rowsCovered[col][k]];
                if (row->first == -1) row->first = i;
                else if (row->second == -1) row->second = i;
                else row->first = -2; /* at least three owners */
            }
        }

        search.size = size;
        uint64_t pairs = lagr_drop2_offset(size, size - 1);
        atomic_store_explicit(&search.next, 0, memory_order_relaxed);
        atomic_store_explicit(&search.winner, pairs, memory_order_relaxed);
        static _Thread_local ccubes_adaptive_team teams[65];
        unsigned key = 0;
        uint64_t work = pairs;
        for (int i = 0; i < size; ++i) work += (uint64_t)rowsCoveredCount[sol[i]];
        for (uint64_t value = work; value; value >>= 1) ++key;
        int threads = ccubes_adaptive_team_choose(&teams[key], pairs, work, ccubes_thread_count());
        uint64_t started = ccubes_monotonic_nanoseconds();
        if (threads <= 1 || !ccubes_parallel_for(0, (uint64_t)threads, threads, false, lagr_drop2_worker, &search)) {
            lagr_drop2_worker(0, 1, 1, 0, 1, &search);
        }
        uint64_t finished = ccubes_monotonic_nanoseconds();
        /* Deadline-truncated samples cannot justify a faster team. */
        if (!deadline || finished < deadline)
            ccubes_adaptive_team_observe(&teams[key], finished > started ? finished - started : 0);
        uint64_t winner = atomic_load_explicit(&search.winner, memory_order_relaxed);
        if (winner == pairs) break;
        int i = lagr_drop2_first(size, winner);
        int j = i + 1 + (int)(winner - lagr_drop2_offset(size, i));
        int replacement = lagr_drop2_replacement(&search, i, j, true);
        int count = 0;
        for (int k = 0; k < size; ++k) {
            if (k != i && k != j) sol[count++] = sol[k];
        }
        if (replacement >= 0) sol[count++] = replacement;
        prune_redundancy(rows, rowsCovered, rowsCoveredCount, sol, &count);
        *sol_len = count;
    }
    free(owners);
}

/*
 * Fixed-width row-coverage bitset for the exact-search fast path, covering
 * up to POLISH_MASK_WORDS * 64 rows. Passed by value like a single uint64_t
 * used to be, so the recursive search still needs no explicit backtracking:
 * each call computes its own covered mask and hands it down.
 */
#define POLISH_MASK_WORDS 4
#define POLISH_PARALLEL_SCRATCH_BYTES (32u * 1024u * 1024u)
#define POLISH_NODE_BATCH 256
/* Worker stacks can be smaller than the stack of the calling thread. */
#define POLISH_PARALLEL_MAX_DEPTH 512
typedef struct {
    uint64_t w[POLISH_MASK_WORDS];
} PolishMask;

typedef struct {
    atomic_bool stop;             /* a cover was accepted or the deadline expired */
    atomic_int queued;
    atomic_int idle;               /* workers waiting for another subtree */
#if defined(HAVE_PTHREAD)
    pthread_mutex_t mutex;
    pthread_cond_t changed;
#else
    ccubes_mutex mutex;
#endif
    long remaining_nodes;        /* unreserved nodes, under the mutex */
    int reservations;
    int budget_waiters;
    int active;                  /* root assignment plus running subtrees, under the mutex */
    int capacity;
    int *task_depths;
    int *task_paths;
    int path_stride;
    const int *root_candidates;
    int root_count;
    int root_child_row_start;
    int base_depth;
    int base_covered;
    PolishMask base_mask;
    struct PolishSearch *workers;
    struct PolishSearch *ordered;
    bool incomplete, exhausted;
} PolishParallel;

typedef struct PolishSearch {
    int rows;
    int **rowsCovered;
    int *rowsCoveredCount;
    int **colsCovering;
    int *colsCoveringCount;
    const double *lagr_score;
    const double *weights;
    const unsigned char *allowed; /* NULL = every column usable */
    const int *forced;            /* columns pre-selected before the search */
    int forced_len;
    int target;                   /* free slots available to the search (excludes forced) */
    int out_stride;               /* forced_len + target */
    long node_limit;
    long nodes;
    int aborted;                  /* node budget or deadline reached */
    uint64_t deadline_nanoseconds;/* zero means no wall deadline */
    int want;                     /* number of distinct covers to collect */
    int found;                    /* covers collected so far */
    int *coverCount;
    int *chosen;
    int *out;                     /* want * out_stride ints */
    int *out_lens;                /* want ints */
    int *scratch;                 /* out_stride ints, for dedup */
    const LagrangianRowDegree *row_order;  /* shared, by increasing core degree */
    int *cand_buf;                /* ordered column IDs at each open depth */
    PolishCandidate *rank_buf;    /* scores are needed only while sorting one node */
    int max_row_degree;
    int max_cover_static;         /* max rows covered by any usable column */
    /* bitset fast path (rows <= 64 * POLISH_MASK_WORDS) */
    const uint64_t *col_mask;     /* per-column row coverage, mask_words each, or NULL */
    PolishMask full_mask;         /* all rows covered */
    int mask_words;               /* words of col_mask/full_mask actually in use */
    int threads;
    int team_workers;
    int active_workers;
    long extra_limit, extra_nodes;
    atomic_bool *cancel;         /* shared decision for the ordered worker */
    int node_credit;              /* unused nodes in the worker's reservation, or -1 */
    PolishParallel *parallel;    /* shared budget and result for parallel decision search */
} PolishSearch;

static int polish_new_cover(
    const PolishSearch *search,
    int col
) {
    int new_cover = 0;
    for (int i = 0; i < search->rowsCoveredCount[col]; ++i) {
        int row = search->rowsCovered[col][i];
        if (search->coverCount[row] == 0) ++new_cover;
    }
    return new_cover;
}

static int polish_choose_row(
    const PolishSearch *search,
    int uncovered,
    int slots_left,
    int *row_start
) {
    /* bound: every column adds at most max_cover_static newly covered rows */
    if (uncovered > (int64_t)slots_left * search->max_cover_static) return -1;

    /* An uncovered row has no selected covering column, so its viable
       degree stays fixed for the whole search. */
    for (int i = *row_start; i < search->rows; ++i) {
        const LagrangianRowDegree *entry = &search->row_order[i];
        if (search->coverCount[entry->row] == 0) {
            *row_start = i + 1;
            return entry->covering_columns > 0 ? entry->row : -1;
        }
    }
    return -1;
}

static int pool_cmp_int_asc(const void *a, const void *b);
static void pool_clear(int **pool_solutions, int *pool_count);
static int pool_add_if_new(
    int **pool_solutions,
    int *pool_count,
    int max_pool,
    const int *sol_sorted,
    int len
);

static bool polish_record_cover(PolishSearch *search, int depth) {
    /* The first worker gets the output buffer. The caller reads it after join. */
    atomic_bool *stop = search->parallel ? &search->parallel->stop : search->cancel;
    if (stop && atomic_exchange_explicit(stop, true, memory_order_relaxed)) {
        return true;
    }

    int len = search->forced_len + depth;
    int *norm = search->scratch;
    for (int i = 0; i < search->forced_len; ++i) norm[i] = search->forced[i];
    memcpy(norm + search->forced_len, search->chosen, (size_t)depth * sizeof(int));
    qsort(norm, (size_t)len, sizeof(int), pool_cmp_int_asc);

    for (int s = 0; s < search->found; ++s) {
        if (
            search->out_lens[s] == len &&
            memcmp(search->out + (size_t)s * search->out_stride, norm, (size_t)len * sizeof(int)) == 0
        ) {
            return false; /* duplicate cover: keep searching */
        }
    }

    memcpy(search->out + (size_t)search->found * search->out_stride, norm, (size_t)len * sizeof(int));
    search->out_lens[search->found++] = len;
    return search->found >= search->want;
}

static bool polish_deadline_reached(PolishSearch *search) {
    if (!search->deadline_nanoseconds ||
        lagr_monotonic_nanoseconds() < search->deadline_nanoseconds) return false;
    search->aborted = 1;
    if (search->parallel) atomic_store_explicit(&search->parallel->stop, true, memory_order_relaxed);
    return true;
}

static bool polish_claim_node(PolishSearch *search);

static bool polish_visit_node(PolishSearch *search) {
    if (
        ((search->nodes & 255L) == 0 && polish_deadline_reached(search)) ||
        search->nodes >= search->node_limit ||
        (search->parallel && !polish_claim_node(search))
    ) {
        search->aborted = 1;
        return false;
    }
    ++search->nodes;
    return true;
}

static int polish_candidates(
    const PolishSearch *search,
    int slots_left,
    int covered,
    PolishCandidate *candidates,
    int *row_start
) {
    int row = polish_choose_row(search, search->rows - covered, slots_left, row_start);
    if (row < 0) return 0;

    int count = 0;
    for (int i = 0; i < search->colsCoveringCount[row]; ++i) {
        int col = search->colsCovering[row][i];
        if (search->allowed && !search->allowed[col]) continue;

        int new_cover = polish_new_cover(search, col);
        if (new_cover <= 0) continue;

        candidates[count++] = (PolishCandidate){
            .col = col,
            .new_cover = new_cover,
            .cover_count = search->rowsCoveredCount[col],
            .lagr_score = search->lagr_score ? search->lagr_score[col] : 0.0,
            .weight = search->weights ? search->weights[col] : 0.0
        };
    }
    return count;
}

static int polish_candidates_bits(
    const PolishSearch *search,
    int slots_left,
    PolishMask covered_mask,
    PolishCandidate *candidates,
    int *row_start
) {
    const int words = search->mask_words;
    PolishMask uncovered_mask;
    int uncovered = 0;
    for (int w = 0; w < words; ++w) {
        uncovered_mask.w[w] = search->full_mask.w[w] & ~covered_mask.w[w];
        uncovered += lagr_popcount64(uncovered_mask.w[w]);
    }

    if (uncovered > (int64_t)slots_left * search->max_cover_static) return 0;

    int best_row = -1;
    for (int i = *row_start; i < search->rows; ++i) {
        const LagrangianRowDegree *entry = &search->row_order[i];
        const int row = entry->row;
        if (uncovered_mask.w[row / 64] & (UINT64_C(1) << (row % 64))) {
            if (entry->covering_columns == 0) return 0;
            best_row = row;
            *row_start = i + 1;
            break;
        }
    }
    if (best_row < 0) return 0;

    int count = 0;
    for (int i = 0; i < search->colsCoveringCount[best_row]; ++i) {
        int col = search->colsCovering[best_row][i];

        const uint64_t *cmask = &search->col_mask[(size_t)col * (size_t)words];
        int new_cover = 0;
        for (int w = 0; w < words; ++w) {
            new_cover += lagr_popcount64(cmask[w] & uncovered_mask.w[w]);
        }
        if (new_cover <= 0) continue;

        candidates[count++] = (PolishCandidate){
            .col = col,
            .new_cover = new_cover,
            .cover_count = search->rowsCoveredCount[col],
            .lagr_score = search->lagr_score ? search->lagr_score[col] : 0.0,
            .weight = search->weights ? search->weights[col] : 0.0
        };
    }
    return count;
}

static bool polish_search_rec(
    PolishSearch *search, int depth, int covered, PolishMask covered_mask, int row_start
);

static void polish_select_column(
    PolishSearch *search,
    int depth,
    int col,
    int *covered,
    PolishMask *covered_mask
) {
    search->chosen[depth] = col;

    if (search->mask_words) {
        const uint64_t *mask = &search->col_mask[(size_t)col * (size_t)search->mask_words];
        for (int w = 0; w < search->mask_words; ++w) covered_mask->w[w] |= mask[w];
    } else {
        for (int j = 0; j < search->rowsCoveredCount[col]; ++j) {
            int row = search->rowsCovered[col][j];
            if (search->coverCount[row]++ == 0) ++*covered;
        }
    }
}

static void polish_unselect_column(PolishSearch *search, int col) {
    if (!search->mask_words) {
        for (int j = 0; j < search->rowsCoveredCount[col]; ++j) {
            --search->coverCount[search->rowsCovered[col][j]];
        }
    }
}

static bool polish_search_branch(
    PolishSearch *search,
    int depth,
    int col,
    int covered,
    PolishMask covered_mask,
    int row_start
) {
    polish_select_column(search, depth, col, &covered, &covered_mask);
    bool stop = polish_search_rec(search, depth + 1, covered, covered_mask, row_start);
    polish_unselect_column(search, col);
    return stop;
}

static bool polish_alloc_scratch(PolishSearch *search) {
    size_t slots = search->target > 0 ? (size_t)search->target : 1;
    search->coverCount = calloc((size_t)search->rows, sizeof(int));
    search->chosen = malloc(slots * sizeof(int));
    search->scratch = malloc((size_t)search->out_stride * sizeof(int));
    search->cand_buf = slots <= SIZE_MAX / sizeof(int) / (size_t)search->max_row_degree
        ? malloc(slots * (size_t)search->max_row_degree * sizeof(int))
        : NULL;
    search->rank_buf = SIZE_MAX / (size_t)search->max_row_degree >= sizeof(PolishCandidate)
        ? malloc((size_t)search->max_row_degree * sizeof(PolishCandidate))
        : NULL;
    return search->coverCount && search->chosen && search->scratch &&
        search->cand_buf && search->rank_buf;
}

static void polish_free_scratch(PolishSearch *search) {
    free(search->coverCount);
    free(search->chosen);
    free(search->scratch);
    free(search->cand_buf);
    free(search->rank_buf);
}

static bool polish_queue_init(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    if (pthread_mutex_init(&parallel->mutex, NULL) != 0) return false;
    if (pthread_cond_init(&parallel->changed, NULL) != 0) {
        pthread_mutex_destroy(&parallel->mutex);
        return false;
    }
    return true;
#else
    return ccubes_mutex_init(&parallel->mutex);
#endif
}

static void polish_queue_destroy(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    pthread_cond_destroy(&parallel->changed);
    pthread_mutex_destroy(&parallel->mutex);
#else
    ccubes_mutex_destroy(&parallel->mutex);
#endif
}

static void polish_queue_lock(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    pthread_mutex_lock(&parallel->mutex);
#else
    ccubes_mutex_lock(&parallel->mutex);
#endif
}

static void polish_queue_unlock(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    pthread_mutex_unlock(&parallel->mutex);
#else
    ccubes_mutex_unlock(&parallel->mutex);
#endif
}

static void polish_queue_wait(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    pthread_cond_wait(&parallel->changed, &parallel->mutex);
#else
    /* An OpenMP-only build has no condition variable. Release the lock
       while waiting so a busy worker can publish work or finish. */
    polish_queue_unlock(parallel);
#if defined(HAVE_OPENMP) && defined(_WIN32)
    Sleep(1);
#elif defined(HAVE_OPENMP)
    nanosleep(&(struct timespec){.tv_nsec = 100000}, NULL);
#endif
    polish_queue_lock(parallel);
#endif
}

static void polish_queue_wake(PolishParallel *parallel) {
#if defined(HAVE_PTHREAD)
    pthread_cond_broadcast(&parallel->changed);
#else
    (void)parallel;
#endif
}

/* A worker returns unused credit when its task ends. A worker waiting for
   credit has released its own reservation, so another worker can always
   make progress while any reservations remain. */
static void polish_return_node_credit(PolishSearch *search) {
    if (search->node_credit < 0) return;
    PolishParallel *parallel = search->parallel;
    parallel->remaining_nodes += search->node_credit;
    --parallel->reservations;
    search->node_credit = -1;
    if (parallel->budget_waiters && (parallel->remaining_nodes || !parallel->reservations)) {
        polish_queue_wake(parallel);
    }
}

static bool polish_claim_node(PolishSearch *search) {
    if (search->node_credit > 0) {
        --search->node_credit;
        return true;
    }
    PolishParallel *parallel = search->parallel;
    polish_queue_lock(parallel);
    polish_return_node_credit(search);
    while (!parallel->remaining_nodes && parallel->reservations &&
           !atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
        ++parallel->budget_waiters;
        polish_queue_wait(parallel);
        --parallel->budget_waiters;
    }
    bool granted = parallel->remaining_nodes > 0 &&
        !atomic_load_explicit(&parallel->stop, memory_order_relaxed) &&
        !polish_deadline_reached(search);
    if (granted) {
        long grant = parallel->remaining_nodes < POLISH_NODE_BATCH
            ? parallel->remaining_nodes : POLISH_NODE_BATCH;
        parallel->remaining_nodes -= grant;
        ++parallel->reservations;
        search->node_credit = (int)grant - 1;
    }
    polish_queue_unlock(parallel);
    return granted;
}

/* Publish unvisited siblings, keeping the current best candidate local.
   Only the chosen-column prefix travels with each branch. */
static int polish_offer_siblings(
    PolishSearch *search, int depth, const int *siblings, int count
) {
    PolishParallel *parallel = search->parallel;
    if (!parallel || count == 0 || depth + 1 >= search->target) return 0;

    /* Seed the queue at the split. Deeper nodes only donate to workers
       that need work, avoiding a lock and path copy for tiny subtrees
       while the team is busy. */
    const bool initial_split = depth == parallel->base_depth;
    const int demand = initial_split ? parallel->capacity :
        atomic_load_explicit(&parallel->idle, memory_order_relaxed);
    if (atomic_load_explicit(&parallel->queued, memory_order_relaxed) >= demand) return 0;

    int offered = 0;
    polish_queue_lock(parallel);
    if (!atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
        int queued = atomic_load_explicit(&parallel->queued, memory_order_relaxed);
        const int limit = initial_split ? parallel->capacity :
            atomic_load_explicit(&parallel->idle, memory_order_relaxed);
        const int open = limit - queued;
        offered = open <= 0 ? 0 : count < open ? count : open;
        int prefix = depth - parallel->base_depth;
        /* Push in reverse priority so the next available worker gets the
           highest-ranked sibling first. */
        for (int i = offered - 1; i >= 0; --i, ++queued) {
            int *path = parallel->task_paths + (size_t)queued * parallel->path_stride;
            memcpy(path, search->chosen + parallel->base_depth, (size_t)prefix * sizeof(int));
            path[prefix] = siblings[i];
            parallel->task_depths[queued] = depth + 1;
        }
        if (offered) {
            atomic_store_explicit(&parallel->queued, queued, memory_order_relaxed);
            polish_queue_wake(parallel);
        }
    }
    polish_queue_unlock(parallel);
    return offered;
}

static bool polish_search_children(
    PolishSearch *search, int depth, int covered, PolishMask covered_mask,
    const int *candidates, int count, int row_start
) {
    for (int i = 0; i < count;) {
        int offered = polish_offer_siblings(search, depth, candidates + i + 1, count - i - 1);
        if (polish_search_branch(search, depth, candidates[i], covered, covered_mask, row_start)) return true;
        i += 1 + offered;
    }
    return false;
}

static void polish_run_task(PolishSearch *search, int depth) {
    PolishParallel *parallel = search->parallel;
    if (polish_deadline_reached(search)) return;
    int covered = parallel->base_covered;
    PolishMask mask = parallel->base_mask;
    /* Restore the task path without charging its already-expanded ancestors.
       The worker retains the common prefix for its next task. */
    int i = parallel->base_depth;
    for (; i < depth; ++i) {
        if ((i & 63) == 63 && (polish_deadline_reached(search) ||
            atomic_load_explicit(&parallel->stop, memory_order_relaxed))) break;
        polish_select_column(search, i, search->chosen[i], &covered, &mask);
    }
    if (i == depth) polish_search_rec(search, depth, covered, mask, 0);
    while (i > parallel->base_depth) {
        --i;
        polish_unselect_column(search, search->chosen[i]);
    }
}

static void polish_branch_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)start;
    (void)end;
    (void)stride;
    (void)worker_count;
    PolishParallel *parallel = data;
    /* The ordered worker runs the original DFS without donating subtrees.
       Either search can end the other once it decides the target. */
    if (parallel->ordered && worker_id == 0) {
        PolishSearch *ordered = parallel->ordered;
        if (!polish_deadline_reached(ordered)) {
            polish_search_children(ordered, parallel->base_depth, parallel->base_covered,
                                   parallel->base_mask, parallel->root_candidates,
                                   parallel->root_count, parallel->root_child_row_start);
        }
        polish_queue_lock(parallel);
        if (!ordered->aborted && !atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
            parallel->exhausted = true;
            atomic_store_explicit(&parallel->stop, true, memory_order_relaxed);
        }
        polish_queue_wake(parallel);
        polish_queue_unlock(parallel);
        return;
    }
    if (parallel->ordered) --worker_id;
    PolishSearch *search = &parallel->workers[worker_id];

    /* The first sharing worker owns the node that started the team. */
    bool root = worker_id == 0;
    polish_queue_lock(parallel);
    while (!atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
        int queued = atomic_load_explicit(&parallel->queued, memory_order_relaxed);
        if (!root && !queued) {
            if (!parallel->active) break;
            atomic_fetch_add_explicit(&parallel->idle, 1, memory_order_relaxed);
            polish_queue_wait(parallel);
            atomic_fetch_sub_explicit(&parallel->idle, 1, memory_order_relaxed);
            continue;
        }
        int depth = parallel->base_depth;
        if (!root) {
            --queued;
            depth = parallel->task_depths[queued];
            int *path = parallel->task_paths + (size_t)queued * parallel->path_stride;
            memcpy(search->chosen + parallel->base_depth, path,
                   (size_t)(depth - parallel->base_depth) * sizeof(int));
            atomic_store_explicit(&parallel->queued, queued, memory_order_relaxed);
            ++parallel->active;
        }
        polish_queue_unlock(parallel);

        if (root) {
            if (!polish_deadline_reached(search)) {
                polish_search_children(search, depth, parallel->base_covered, parallel->base_mask,
                                       parallel->root_candidates, parallel->root_count,
                                       parallel->root_child_row_start);
            }
            root = false;
        } else {
            polish_run_task(search, depth);
        }

        polish_queue_lock(parallel);
        polish_return_node_credit(search);
        --parallel->active;
        parallel->incomplete |= search->aborted;
        if (parallel->ordered && !parallel->active && !parallel->incomplete &&
            !atomic_load_explicit(&parallel->queued, memory_order_relaxed) &&
            !atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
            parallel->exhausted = true;
            atomic_store_explicit(&parallel->stop, true, memory_order_relaxed);
        }
        if (!parallel->active || atomic_load_explicit(&parallel->stop, memory_order_relaxed)) {
            polish_queue_wake(parallel);
        }
    }
    polish_queue_unlock(parallel);
}

static bool polish_parallel_branches(
    PolishSearch *search,
    int depth,
    int covered,
    PolishMask covered_mask,
    const int *candidates,
    int count,
    int row_start
) {
    const bool preserve_order = search->extra_limit > 0;
    int threads = search->threads;
    int path_stride = search->target - depth;
    const int ordered_workers = preserve_order ? 1 : 0;
    if (threads < 2 + ordered_workers || count < 2 || path_stride <= 1) return false;

    /* Bound worker scratch and allow one queued path per additional worker. */
    uint64_t candidate_entries = (uint64_t)search->target * search->max_row_degree;
    search->threads = 1;
    uint64_t worker_bytes = candidate_entries * sizeof(int)
        + (uint64_t)search->max_row_degree * sizeof(PolishCandidate)
        + ((uint64_t)search->rows + search->target + search->out_stride + path_stride + 1) * sizeof(int)
        + sizeof(PolishSearch);
    uint64_t scratch_threads = POLISH_PARALLEL_SCRATCH_BYTES / worker_bytes;
    if ((uint64_t)threads > scratch_threads) threads = (int)scratch_threads;
    if (threads < 2 + ordered_workers) return false;

    PolishParallel parallel = {
        .remaining_nodes = preserve_order ? search->extra_limit : search->node_limit - search->nodes,
        .active = 1,
        .path_stride = path_stride,
        .root_candidates = candidates,
        .root_count = count,
        .root_child_row_start = row_start,
        .base_depth = depth,
        .base_covered = covered,
        .base_mask = covered_mask,
        .workers = malloc((size_t)(threads - ordered_workers) * sizeof(PolishSearch))
    };
    if (!parallel.workers) return false;
    atomic_init(&parallel.queued, 0);
    atomic_init(&parallel.idle, 0);
    atomic_init(&parallel.stop, false);

    PolishSearch ordered;
    int allocated = 0;
    while (allocated < threads) {
        const bool primary = preserve_order && allocated == 0;
        PolishSearch *worker = primary ? &ordered : &parallel.workers[allocated - ordered_workers];
        *worker = *search;
        worker->parallel = primary ? NULL : &parallel;
        worker->cancel = primary ? &parallel.stop : NULL;
        worker->nodes = 0;
        worker->node_credit = -1;
        if (primary) worker->node_limit -= search->nodes;
        else if (preserve_order) worker->node_limit = search->extra_limit;
        if (!polish_alloc_scratch(worker)) {
            polish_free_scratch(worker);
            break;
        }
        memcpy(worker->coverCount, search->coverCount, (size_t)search->rows * sizeof(int));
        memcpy(worker->chosen, search->chosen, (size_t)depth * sizeof(int));
        if (primary) parallel.ordered = worker;
        ++allocated;
    }

    bool ran = false;
    if (allocated >= 2 + ordered_workers) {
        parallel.capacity = allocated - 1 - ordered_workers;
        parallel.task_depths = malloc((size_t)parallel.capacity * sizeof(int));
        parallel.task_paths = malloc((size_t)parallel.capacity * path_stride * sizeof(int));
        if (parallel.task_depths && parallel.task_paths && polish_queue_init(&parallel)) {
            int actual_workers = 0;
            ran = ccubes_parallel_for_counted(
                0, (uint64_t)allocated, allocated, false, polish_branch_worker, &parallel,
                &actual_workers
            );
            if (ran) search->team_workers = actual_workers;
            polish_queue_destroy(&parallel);
        }
        free(parallel.task_paths);
        free(parallel.task_depths);
    }

    for (int i = 0; i < allocated; ++i) {
        const bool primary = preserve_order && i == 0;
        PolishSearch *worker = primary ? &ordered : &parallel.workers[i - ordered_workers];
        if (preserve_order && !primary) search->extra_nodes += worker->nodes;
        else search->nodes += worker->nodes;
        if (worker->nodes || worker->found) ++search->active_workers;
        search->found += worker->found;
        search->aborted |= worker->aborted;
        polish_free_scratch(worker);
    }
    if (ran && preserve_order) search->aborted = !parallel.exhausted && !search->found;
    else if (search->found) search->aborted = 0;
    free(parallel.workers);
    return ran;
}

static bool polish_search_rec(
    PolishSearch *search,
    int depth,
    int covered,
    PolishMask covered_mask,
    int row_start
) {
    const atomic_bool *stop = search->parallel ? &search->parallel->stop : search->cancel;
    if (stop && atomic_load_explicit(stop, memory_order_relaxed)) return true;

    bool fully_covered = covered >= search->rows;
    if (search->mask_words) {
        fully_covered = true;
        for (int w = 0; w < search->mask_words; ++w) {
            if (covered_mask.w[w] != search->full_mask.w[w]) {
                fully_covered = false;
                break;
            }
        }
    }
    if (fully_covered) {
        return polish_deadline_reached(search) || polish_record_cover(search, depth);
    }
    if (depth >= search->target) return false;
    if (!polish_visit_node(search)) return true;

    int slots_left = search->target - depth;
    /* A child covers the branch row. Earlier rows in the shared order
       will also stay covered for the rest of this path. */
    PolishCandidate *rank = search->rank_buf;
    int count = search->mask_words
        ? polish_candidates_bits(search, slots_left, covered_mask, rank, &row_start)
        : polish_candidates(search, slots_left, covered, rank, &row_start);
    polish_sort_candidates(rank, count);
    int *candidates = search->cand_buf + (size_t)depth * (size_t)search->max_row_degree;
    for (int i = 0; i < count; ++i) candidates[i] = rank[i].col;

    if (search->threads > 1 &&
        polish_parallel_branches(search, depth, covered, covered_mask, candidates, count, row_start)) {
        return search->found || search->aborted;
    }

    return polish_search_children(search, depth, covered, covered_mask, candidates, count, row_start);
}

/*
Bounded search for covers of size <= target_total.
- allowed: optional column mask (NULL = all active columns)
- forced/forced_len: columns fixed into every cover before the search starts
- want: how many distinct covers to collect (enumeration for pool mode)
- out: want * target_total ints; out_lens: length of each collected cover
The single-cover search can share unvisited branches across workers.
Returns the number of covers found; *aborted_out marks an incomplete search
due to a resource limit.
*/
static int lagr_bounded_cover_search(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights,
    const unsigned char *allowed,
    const int *forced,
    int forced_len,
    int target_total,
    long node_limit,
    int want,
    int *out,
    int *out_lens,
    int *aborted_out,
    long *nodes_out,
    uint64_t deadline_nanoseconds,
    int threads,
    long extra_limit
) {
    if (aborted_out) *aborted_out = 0;
    if (nodes_out) *nodes_out = 0;
    if (target_total <= 0 || node_limit <= 0 || want <= 0) return 0;
    if (forced_len > target_total) return 0;

    int free_slots = target_total - forced_len;

    LagrangianRowDegree *row_order = malloc((size_t)rows * sizeof(*row_order));
    if (!row_order) {
        if (aborted_out) *aborted_out = 1;
        return 0;
    }
    int max_row_degree = 1;
    int max_cover_static = 1;
    for (int r = 0; r < rows; ++r) {
        int degree = 0;
        for (int i = 0; i < colsCoveringCount[r]; ++i) {
            int c = colsCovering[r][i];
            if (allowed && !allowed[c]) continue;
            ++degree;
            if (rowsCoveredCount[c] > max_cover_static) max_cover_static = rowsCoveredCount[c];
        }
        row_order[r] = (LagrangianRowDegree){.row = r, .covering_columns = degree};
        if (degree > max_row_degree) max_row_degree = degree;
    }
    qsort(row_order, (size_t)rows, sizeof(*row_order), lagr_compare_row_degree);

    int mask_words = !allowed && rows <= 64 * POLISH_MASK_WORDS ? (rows + 63) / 64 : 0;
    uint64_t *col_mask = mask_words
        ? (uint64_t*)calloc((size_t)cols * (size_t)mask_words, sizeof(uint64_t))
        : NULL;
    unsigned char *mask_done = mask_words
        ? (unsigned char*)calloc((size_t)cols, 1)
        : NULL;

    PolishSearch search = {
        .rows = rows,
        .rowsCovered = rowsCovered,
        .rowsCoveredCount = rowsCoveredCount,
        .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount,
        .lagr_score = lagr_score,
        .weights = weights,
        .allowed = allowed,
        .forced = forced,
        .forced_len = forced_len,
        .target = free_slots,
        .out_stride = target_total,
        .node_limit = node_limit,
        .deadline_nanoseconds = deadline_nanoseconds,
        .want = want,
        .out = out,
        .out_lens = out_lens,
        .row_order = row_order,
        .max_row_degree = max_row_degree,
        .max_cover_static = max_cover_static,
        .col_mask = col_mask,
        .mask_words = mask_words,
        .team_workers = 1,
        .threads = want == 1 && free_slots <= POLISH_PARALLEL_MAX_DEPTH
            ? threads : 1
    };
    if (!polish_alloc_scratch(&search) || (mask_words && (!col_mask || !mask_done))) {
        polish_free_scratch(&search);
        free(col_mask);
        free(mask_done);
        free(row_order);
        if (aborted_out) *aborted_out = 1;
        return 0;
    }

    if (mask_words) {
        /* per-column coverage masks, filled once per core column */
        for (int r = 0; r < rows; ++r) {
            for (int i = 0; i < colsCoveringCount[r]; ++i) {
                int c = colsCovering[r][i];
                if (!mask_done[c]) {
                    mask_done[c] = 1;
                    uint64_t *m = &col_mask[(size_t)c * (size_t)mask_words];
                    for (int j = 0; j < rowsCoveredCount[c]; ++j) {
                        int row = rowsCovered[c][j];
                        m[row / 64] |= UINT64_C(1) << (row % 64);
                    }
                }
            }
        }
        free(mask_done);
    }

    int covered = 0;
    PolishMask covered_mask = {0};
    for (int i = 0; i < forced_len; ++i) {
        int c = forced[i];
        for (int j = 0; j < rowsCoveredCount[c]; ++j) {
            int r = rowsCovered[c][j];
            if (search.coverCount[r]++ == 0) ++covered;
        }
        for (int w = 0; w < mask_words; ++w) {
            covered_mask.w[w] |= col_mask[(size_t)c * (size_t)mask_words + w];
        }
    }

    for (int w = 0; w < mask_words; ++w) {
        int rows_left = rows - w * 64;
        search.full_mask.w[w] = rows_left >= 64
            ? UINT64_MAX
            : (UINT64_C(1) << rows_left) - 1u;
    }
    /* Keep cheap targets serial. If the probe stops early, replay from the
       root with the original allowance protected for one ordered worker.
       The probe and the other workers use a separate shared allowance. */
    const bool probe = extra_limit > 0 && search.threads >= 3 && free_slots > 1 &&
        max_row_degree > 1 && node_limit > POLISH_NODE_BATCH &&
        extra_limit / POLISH_NODE_BATCH >= 3;
    if (extra_limit > 0) {
        search.threads = 1;
        if (probe) search.node_limit = POLISH_NODE_BATCH;
    }
    polish_search_rec(&search, 0, covered, covered_mask, 0);
    if (probe && search.aborted && search.nodes == POLISH_NODE_BATCH &&
        !polish_deadline_reached(&search)) {
        search.aborted = 0;
        search.node_limit = node_limit;
        search.extra_nodes = search.nodes;
        search.extra_limit = extra_limit - search.nodes;
        search.nodes = 0;
        long team_limit = 1 + search.extra_limit / POLISH_NODE_BATCH;
        search.threads = team_limit < threads ? (int)team_limit : threads;
        polish_search_rec(&search, 0, covered, covered_mask, 0);
    }

    if (want == 1) {
        lagr_last_stats.polish_extra_nodes += search.extra_nodes;
        if (!search.active_workers && (search.nodes || search.found)) search.active_workers = 1;
        if (search.team_workers > lagr_last_stats.polish_workers)
            lagr_last_stats.polish_workers = search.team_workers;
        if (search.active_workers > lagr_last_stats.polish_active_workers)
            lagr_last_stats.polish_active_workers = search.active_workers;
    }

    if (aborted_out) *aborted_out = search.aborted;
    if (nodes_out) *nodes_out = search.nodes;

    polish_free_scratch(&search);
    free(col_mask);
    free(row_order);

    return search.found;
}

/*
Lagrangian bounded finish:
1. Recompute reduced costs at the best multipliers found (caller passes ls_best/zlb_best).
2. Reduced-cost variable fixing for a target size T:
   - exclude column c when zlb + max(0, rc_c) > T (cannot be in any cover of cost <= T)
   - force column c when rc_c < 0 and zlb - rc_c > T (must be in every such cover)
3. Search at T = UB-1 within fixed budgets; iterate downward while covers are found.
   An exhaustive (non-aborted) failure at T proves UB optimal, which also lifts
   the integer lower bound to UB.
4. Optional pool enumeration at the final UB.
*/
/*
Build a core-restricted copy of the row -> columns adjacency containing only
columns that can appear in a cover of cost <= target (reduced-cost fixing at
the best multipliers). Also collects columns that MUST be in every such cover.
Returns the core size, or -1 on OOM. *core_covering / *core_counts are
allocated as one block each; caller frees both.
*/
static int lagr_build_core(
    int rows,
    int cols,
    const int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *ls_best,
    double zlb_best,
    int target,
    unsigned char *allowed,      /* cols scratch, filled */
    int *forced,                 /* cols scratch, filled */
    int *forced_len_out,
    int ***core_covering_out,
    int **core_counts_out
) {
    const double FIXTOL = 1e-7;

    int forced_len = 0;
    int core_count = 0;
    for (int c = 0; c < cols; ++c) {
        double rc_c = ls_best[c];
        if (rowsCoveredCount[c] <= 0 || zlb_best + (rc_c > 0.0 ? rc_c : 0.0) > (double)target + FIXTOL) {
            allowed[c] = 0;
        } else {
            allowed[c] = 1;
            ++core_count;
            if (rc_c < 0.0 && zlb_best - rc_c > (double)target + FIXTOL) {
                forced[forced_len++] = c;
            }
        }
    }
    *forced_len_out = forced_len;

    if (!core_covering_out || !core_counts_out) return core_count;

    int **core_covering = (int**)malloc((size_t)rows * sizeof(int*));
    int *core_counts = (int*)malloc((size_t)rows * sizeof(int));
    if (!core_covering || !core_counts) {
        free(core_covering);
        free(core_counts);
        return -1;
    }

    size_t nnz = 0;
    for (int r = 0; r < rows; ++r) {
        int cnt = 0;
        for (int k = 0; k < colsCoveringCount[r]; ++k) {
            if (allowed[colsCovering[r][k]]) ++cnt;
        }
        core_counts[r] = cnt;
        nnz += (size_t)cnt;
    }

    int *pool = (int*)malloc((nnz > 0 ? nnz : 1) * sizeof(int));
    if (!pool) {
        free(core_covering);
        free(core_counts);
        return -1;
    }

    size_t off = 0;
    for (int r = 0; r < rows; ++r) {
        core_covering[r] = pool + off;
        for (int k = 0; k < colsCoveringCount[r]; ++k) {
            int c = colsCovering[r][k];
            if (allowed[c]) pool[off++] = c;
        }
    }

    *core_covering_out = core_covering;   /* core_covering[0] owns the pool */
    *core_counts_out = core_counts;
    return core_count;
}

static void lagr_free_core(int **core_covering, int *core_counts, int rows) {
    if (core_covering) {
        if (rows > 0 && core_covering[0]) free(core_covering[0]); /* the pool */
        free(core_covering);
    }
    free(core_counts);
}

/*
Chvatal-Gomory k-cover bound over a core-restricted chart (all rows
uncovered): a set S of rows such that every core column covers at most k
rows of S forces any cover to use at least ceil(|S| / k) columns. Greedy
separation, rows with the fewest core candidates first. Used as a root
test before launching a bounded search: a bound above the target proves
the target unreachable without expanding a single node.
*/
static int lagr_kcover_bound(
    int rows,
    int cols,
    int **core_covering,
    int *core_counts,
    int k,
    int *hits,       /* cols scratch */
    int *rows_order  /* rows scratch */
) {
    if (k <= 1 || rows <= 0 || rows > 256) return 0;

    for (int r = 0; r < rows; ++r) rows_order[r] = r;

    /* ascending number of core candidates, ties by row id */
    for (int i = 1; i < rows; ++i) {
        int r = rows_order[i];
        int j = i - 1;
        while (j >= 0 && (core_counts[rows_order[j]] > core_counts[r] ||
               (core_counts[rows_order[j]] == core_counts[r] && rows_order[j] > r))) {
            rows_order[j + 1] = rows_order[j];
            --j;
        }
        rows_order[j + 1] = r;
    }

    memset(hits, 0, (size_t)cols * sizeof(int));

    int in_s = 0;
    for (int idx = 0; idx < rows; ++idx) {
        int r = rows_order[idx];
        int fits = 1;

        for (int i = 0; i < core_counts[r]; ++i) {
            if (hits[core_covering[r][i]] >= k) {
                fits = 0;
                break;
            }
        }
        if (!fits) continue;

        for (int i = 0; i < core_counts[r]; ++i) {
            hits[core_covering[r][i]]++;
        }
        ++in_s;
    }

    return (in_s + k - 1) / k;
}

static void lagr_bounded_finish(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *weights,
    const double *ls_best,
    double zlb_best,
    long node_limit,
    int core_cap,
    int *solution,
    int *solmin,
    double *bestLB_io,
    int *proved_optimal_io,
    int max_pool,
    int **pool_solutions,
    int *pool_count,
    int *pool_min_len,
    int *norm_buf,
    uint64_t deadline_nanoseconds,
    const PIChartView *polish_chart,
    size_t warm_memory,
    unsigned warm_workers,
    int bounded_workers,
    bool adaptive_finish
) {
    if (!solution || !solmin || *solmin <= 0) return;

    const uint64_t polish_started = lagr_monotonic_nanoseconds();
    lagr_last_stats.polish_initial_ub = *solmin;
    lagr_last_stats.polish_final_ub = *solmin;
    lagr_last_stats.polish_initial_target =
        *solmin > 1 ? *solmin - 1 : -1;
    lagr_last_stats.polish_stop_reason = LAGR_POLISH_NO_TARGET;

    unsigned char *allowed = (unsigned char*)malloc((size_t)cols);
    int *forced = (int*)malloc((size_t)cols * sizeof(int));
    int *one = (int*)malloc((size_t)(*solmin) * sizeof(int));
    int *found_buf = NULL;
    int *found_lens = NULL;
    if (!allowed || !forced || !one) {
        free(allowed);
        free(forced);
        free(one);
        lagr_last_stats.polish_stop_reason = LAGR_POLISH_OOM;
        const uint64_t finished = lagr_monotonic_nanoseconds();
        if (finished >= polish_started) {
            lagr_last_stats.polish_seconds +=
                (double)(finished - polish_started) / 1e9;
        }
        return;
    }

    IpcChartPolish *warm = NULL;
    bool warm_considered = false;
    int UB = *solmin;
    int proved = proved_optimal_io ? *proved_optimal_io : 0;

    /*
    Improvement loop. Searching size <= target only makes sense while
    target >= LB. When the core at a target is too large (weak fixing),
    step down to a tighter target instead of burning the node budget:
    the mask sharpens as the target approaches ZLB.
    An exhaustive failure at target T proves no cover of size <= T exists,
    lifting the integer LB to T+1; when T == UB-1 that proves UB optimal.
    */
    while (!proved && UB > 1) {
        if (
            deadline_nanoseconds > 0 &&
            lagr_monotonic_nanoseconds() >= deadline_nanoseconds
        ) {
            lagr_last_stats.polish_stop_reason =
                LAGR_POLISH_DEADLINE;
            break;
        }
        int target = UB - 1;
        int improved = 0;

        while (target >= 1 && (!bestLB_io || (double)target >= *bestLB_io - EPS)) {
            int forced_len = 0;
            int **core_covering = NULL;
            int *core_counts = NULL;

            int core_count = lagr_build_core(
                rows, cols,
                rowsCoveredCount,
                colsCovering, colsCoveringCount,
                ls_best, zlb_best,
                target,
                allowed, forced, &forced_len,
                NULL, NULL
            );
            if (lagr_last_stats.polish_initial_core_count < 0) {
                lagr_last_stats.polish_initial_core_count =
                    core_count;
            }

            if (forced_len > target) {
                /* more must-have columns than slots: size <= target impossible */
                if (bestLB_io && (double)(target + 1) > *bestLB_io) *bestLB_io = (double)(target + 1);
                if (target == UB - 1) proved = 1;
                lagr_last_stats.polish_forced_columns =
                    forced_len;
                lagr_last_stats.polish_stop_reason =
                    LAGR_POLISH_FORCED_INFEASIBLE;
                break;
            }

            if (core_count > core_cap) {
                lagr_last_stats.polish_skipped_targets++;
                lagr_last_stats.polish_stop_reason =
                    LAGR_POLISH_CORE_CAP;
                --target; /* tighter target, smaller core */
                continue;
            }

            core_count = lagr_build_core(
                rows, cols,
                rowsCoveredCount,
                colsCovering, colsCoveringCount,
                ls_best, zlb_best,
                target,
                allowed, forced, &forced_len,
                &core_covering, &core_counts
            );
            if (core_count < 0) {
                lagr_last_stats.polish_stop_reason =
                    LAGR_POLISH_OOM;
                break;
            }
            lagr_last_stats.polish_search_target = target;
            lagr_last_stats.polish_search_core_count =
                core_count;
            lagr_last_stats.polish_forced_columns =
                forced_len;

            /* root k-cover cut: may prove target unreachable for free */
            {
                int *hits = (int*)malloc((size_t)cols * sizeof(int));
                int *rows_order = (int*)malloc((size_t)rows * sizeof(int));
                int cut_lb = 0;
                if (hits && rows_order) {
                    for (int k = 2; k <= 3; ++k) {
                        int b = lagr_kcover_bound(
                            rows, cols, core_covering, core_counts, k, hits, rows_order
                        );
                        if (b > cut_lb) cut_lb = b;
                    }
                }
                free(hits);
                free(rows_order);

                if (cut_lb > target) {
                    lagr_free_core(core_covering, core_counts, rows);
                    if (bestLB_io && (double)(target + 1) > *bestLB_io) {
                        *bestLB_io = (double)(target + 1);
                    }
                    if (target == UB - 1) proved = 1;
                    lagr_last_stats.polish_stop_reason =
                        LAGR_POLISH_ROOT_BOUND;
                    break;
                }
            }

            int found_len_one = 0;
            int aborted = 0;
            long visited_nodes = 0;
            long remaining_nodes = node_limit;
            if (polish_chart || adaptive_finish) remaining_nodes -= lagr_last_stats.polish_nodes;
            const long extra_nodes = adaptive_finish ?
                node_limit - lagr_last_stats.polish_extra_nodes : 0;
            if (remaining_nodes <= 0) {
                lagr_last_stats.polish_stop_reason = LAGR_POLISH_ABORTED;
                lagr_free_core(core_covering, core_counts, rows);
                break;
            }

            /* The admission test uses the existing integer bound, before
             * allocating a pricing basis. Retain the unpriced route for
             * chart geometries or work budgets the warm search cannot use. */
            if (polish_chart && !warm_considered) {
                warm_considered = true;
                if (bestLB_io && (double)target - *bestLB_io <= 6.0) {
                    uint64_t stop = polish_started + (node_limit <= 80000
                        ? UINT64_C(4000000000) : UINT64_C(8000000000));
                    if (deadline_nanoseconds && deadline_nanoseconds < stop)
                        stop = deadline_nanoseconds;
                    warm = ipc_chart_polish_open(polish_chart, warm_memory, stop, warm_workers);
                } else {
                    ++lagr_last_stats.polish_warm_fallbacks;
                }
            }

            bool decided = false;
            int n = 0;
            if (warm) {
                int degree = 1;
                for (int row = 0; row < rows; ++row)
                    if (core_counts[row] > degree) degree = core_counts[row];
                const IpcChartPolishQuery query = {
                    .allowed = allowed, .forced = forced, .forced_len = forced_len,
                    .target = target, .max_row_degree = degree,
                    .coverage_counts = rowsCoveredCount, .reduced_costs = ls_best,
                    .tie_weights = weights, .node_limit = (uint64_t)remaining_nodes
                };
                IpcChartPolishResult result = ipc_chart_polish_decide(warm, &query, one);
                lagr_last_stats.polish_warm_work += result.work;
                lagr_last_stats.polish_warm_nodes += result.nodes;
                if (result.peak_bytes > lagr_last_stats.polish_warm_peak_bytes)
                    lagr_last_stats.polish_warm_peak_bytes = result.peak_bytes;
                if (result.workers > lagr_last_stats.polish_warm_workers)
                    lagr_last_stats.polish_warm_workers = result.workers;
                visited_nodes = (long)result.nodes;
                decided = result.status == NSAT_UNSAT;
                if (result.status == NSAT_SAT && result.terms <= target &&
                    cover_is_feasible(polish_chart, one, result.terms)) {
                    decided = true;
                    n = 1;
                    found_len_one = result.terms;
                    qsort(one, (size_t)found_len_one, sizeof(int), pool_cmp_int_asc);
                }
                if (!decided) {
                    ++lagr_last_stats.polish_warm_fallbacks;
                    ipc_chart_polish_close(warm);
                    warm = NULL;
                }
            }

            if (!decided && visited_nodes < remaining_nodes &&
                (!deadline_nanoseconds || lagr_monotonic_nanoseconds() < deadline_nanoseconds)) {
                long ordinary_nodes = 0;
                n = lagr_bounded_cover_search(
                    rows, cols,
                    rowsCovered, rowsCoveredCount,
                    core_covering, core_counts,
                    ls_best, weights,
                    NULL, /* adjacency already core-restricted */
                    forced, forced_len,
                    target,
                    remaining_nodes - visited_nodes,
                    1,
                    one, &found_len_one,
                    &aborted,
                    &ordinary_nodes,
                    deadline_nanoseconds,
                    adaptive_finish && extra_nodes <= 0 ? 1 : bounded_workers,
                    extra_nodes
                );
                visited_nodes += ordinary_nodes;
            } else if (!decided) {
                aborted = 1;
            }
            lagr_last_stats.polish_searches++;
            lagr_last_stats.polish_nodes += visited_nodes;

            lagr_free_core(core_covering, core_counts, rows);

            if (n > 0) {
                memcpy(solution, one, (size_t)found_len_one * sizeof(int));
                *solmin = found_len_one;
                UB = found_len_one;
                improved = 1;
                lagr_last_stats.polish_improvements++;
                lagr_last_stats.polish_stop_reason =
                    LAGR_POLISH_IMPROVED;
                break;
            }

            if (aborted) {
                lagr_last_stats.polish_aborted_searches++;
                lagr_last_stats.polish_stop_reason =
                    deadline_nanoseconds > 0 &&
                    lagr_monotonic_nanoseconds() >=
                        deadline_nanoseconds
                        ? LAGR_POLISH_DEADLINE
                        : LAGR_POLISH_ABORTED;
                break;
            }

            /* exhaustive failure: no cover of size <= target exists */
            lagr_last_stats.polish_exhausted_searches++;
            lagr_last_stats.polish_stop_reason =
                LAGR_POLISH_EXHAUSTED;
            if (bestLB_io && (double)(target + 1) > *bestLB_io) *bestLB_io = (double)(target + 1);
            if (target == UB - 1) proved = 1;
            break;
        }

        if (!improved) break;
    }

    if (proved) {
        if (bestLB_io && (double)UB > *bestLB_io) *bestLB_io = (double)UB;
        if (proved_optimal_io) *proved_optimal_io = 1;
        lagr_last_stats.polish_proved_optimal = 1;
    }

    ipc_chart_polish_close(warm);

    /* ---- pool enumeration at the final size ---- */
    if (max_pool > 1 && pool_solutions && pool_count && pool_min_len && norm_buf) {
        if (UB < *pool_min_len) {
            /* the improvement loop beat every pooled solution: resync the pool */
            pool_clear(pool_solutions, pool_count);
            *pool_min_len = UB;
            memcpy(norm_buf, solution, (size_t)UB * sizeof(int));
            qsort(norm_buf, (size_t)UB, sizeof(int), pool_cmp_int_asc);
            pool_add_if_new(pool_solutions, pool_count, max_pool, norm_buf, UB);
        }

        int restart = 1;
        int rounds = 0;

        while (restart && rounds < 4) {
            if (
                deadline_nanoseconds > 0 &&
                lagr_monotonic_nanoseconds() >= deadline_nanoseconds
            ) {
                lagr_last_stats.polish_stop_reason =
                    LAGR_POLISH_DEADLINE;
                break;
            }
            restart = 0;
            ++rounds;

            int target = UB;
            int forced_len = 0;
            int **core_covering = NULL;
            int *core_counts = NULL;

            int core_count = lagr_build_core(
                rows, cols,
                rowsCoveredCount,
                colsCovering, colsCoveringCount,
                ls_best, zlb_best,
                target,
                allowed, forced, &forced_len,
                NULL, NULL
            );
            if (forced_len > target) break;
            if (core_count > core_cap) break; /* enumeration too expensive here */

            core_count = lagr_build_core(
                rows, cols,
                rowsCoveredCount,
                colsCovering, colsCoveringCount,
                ls_best, zlb_best,
                target,
                allowed, forced, &forced_len,
                &core_covering, &core_counts
            );
            if (core_count < 0) break;

            free(found_buf);
            free(found_lens);
            found_buf = (int*)malloc((size_t)max_pool * (size_t)target * sizeof(int));
            found_lens = (int*)malloc((size_t)max_pool * sizeof(int));
            if (!found_buf || !found_lens) {
                lagr_free_core(core_covering, core_counts, rows);
                break;
            }

            int aborted = 0;
            int n = lagr_bounded_cover_search(
                rows, cols,
                rowsCovered, rowsCoveredCount,
                core_covering, core_counts,
                ls_best, weights,
                NULL,
                forced, forced_len,
                target,
                node_limit,
                max_pool,
                found_buf, found_lens,
                &aborted,
                NULL,
                deadline_nanoseconds,
                1,
                0
            );

            lagr_free_core(core_covering, core_counts, rows);

            for (int s = 0; s < n; ++s) {
                int len = found_lens[s];
                const int *cover = found_buf + (size_t)s * (size_t)target;

                if (len < *pool_min_len || len < UB) {
                    /* better cover surfaced during enumeration */
                    memcpy(solution, cover, (size_t)len * sizeof(int));
                    *solmin = len;
                    UB = len;
                    pool_clear(pool_solutions, pool_count);
                    *pool_min_len = len;
                    memcpy(norm_buf, cover, (size_t)len * sizeof(int));
                    qsort(norm_buf, (size_t)len, sizeof(int), pool_cmp_int_asc);
                    pool_add_if_new(pool_solutions, pool_count, max_pool, norm_buf, len);
                    restart = 1;
                    break;
                }

                if (len == *pool_min_len) {
                    memcpy(norm_buf, cover, (size_t)len * sizeof(int));
                    qsort(norm_buf, (size_t)len, sizeof(int), pool_cmp_int_asc);
                    pool_add_if_new(pool_solutions, pool_count, max_pool, norm_buf, len);
                }
            }
        }
    }

    free(found_buf);
    free(found_lens);
    free(allowed);
    free(forced);
    free(one);
    lagr_last_stats.polish_final_ub = UB;
    const uint64_t polish_finished = lagr_monotonic_nanoseconds();
    if (polish_finished >= polish_started) {
        lagr_last_stats.polish_seconds +=
            (double)(polish_finished - polish_started) / 1e9;
    }
}

/* ---------- Solution pool helpers ---------- */
static int pool_cmp_int_asc(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

static void pool_normalize(const int *sol, int len, int *out_sorted) {
    memcpy(out_sorted, sol, (size_t)len * sizeof(int));
    qsort(out_sorted, (size_t)len, sizeof(int), pool_cmp_int_asc);
}

static int pool_equal_sets(const int *a, int la, const int *b, int lb) {
    if (la != lb) return 0;
    for (int i = 0; i < la; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}

static void pool_clear(int **pool_solutions, int *pool_count) {
    if (!pool_solutions || !pool_count) return;
    for (int i = 0; i < *pool_count; ++i) {
        free(pool_solutions[i]);
        pool_solutions[i] = NULL;
    }
    *pool_count = 0;
}

static int pool_add_if_new(
    int **pool_solutions,
    int *pool_count,
    int max_pool,
    const int *sol_sorted,
    int len
) {
    if (!pool_solutions || !pool_count || max_pool <= 0) return 0;
    for (int i = 0; i < *pool_count; ++i) {
        if (pool_equal_sets(sol_sorted, len, pool_solutions[i], len)) return 0; /* duplicate */
    }

    if (*pool_count >= max_pool) return 0; /* full */

    int *dst = (int*)malloc((size_t)len * sizeof(int));
    if (!dst) return 0;
    memcpy(dst, sol_sorted, (size_t)len * sizeof(int));
    pool_solutions[(*pool_count)++] = dst;
    return 1;
}

static void pool_order_by_weight(
    int **pool_solutions,
    int pool_count,
    int len,
    const double *weights
) {
    if (!pool_solutions || pool_count <= 1 || len <= 0) return;

    for (int i = 1; i < pool_count; ++i) {
        int *candidate = pool_solutions[i];
        double candidate_weight =
            solution_total_weight(candidate, len, weights);
        int position = i;

        while (position > 0) {
            int *previous = pool_solutions[position - 1];
            double previous_weight =
                solution_total_weight(previous, len, weights);
            bool precedes =
                candidate_weight > previous_weight + EPS ||
                (
                    fabs(candidate_weight - previous_weight) <= EPS &&
                    memcmp(
                        candidate,
                        previous,
                        (size_t)len * sizeof(int)
                    ) < 0
                );
            if (!precedes) break;
            pool_solutions[position] = previous;
            position--;
        }
        pool_solutions[position] = candidate;
    }
}

typedef struct {
    int max_iter;
    int heur_every;
    double step_coef;
    double step_min;
    double step_contract;
    double stabilization_beta;
    int halve_period;
    int local_passes;
    int drop2_enabled;
    int small_gap_threshold;
    int small_gap_extra_passes;
    int rarity_init;
    double deflection_alpha;
    int polish_enabled;
    long polish_node_limit;
    int polish_core_cap;
    int portfolio_enabled;
    int portfolio_max_profiles;
    double portfolio_deflection_alpha;
    int hybrid_bundle_portfolio;
    int bundle_enabled;
    int bundle_interval;
    int bundle_memory;
    double bundle_prox_mu;
    double bundle_min_mu;
    double bundle_max_mu;
    double bundle_null_expand;
    double bundle_serious_shrink;
    double bundle_serious_fraction;
    double bundle_serious_tol;
    double bundle_momentum;
    int stop_at_step_floor;
    int reset_on_continuous_improvement;
    const PIChartView *polish_chart;
    int bounded_workers;
    bool adaptive_finish;
} LagrangianConfig;

#define LAGR_PORTFOLIO_PROFILE_COUNT 6
static int lagr_max_int(int a, int b) {
    return a > b ? a : b;
}

static int lagr_normalize_effort_level(int effort_level) {
    if (effort_level < 0) return 0;
    if (effort_level > 2) return 2;
    return effort_level;
}

static void lagr_config_for_effort(LagrangianConfig *cfg, int effort_level) {
    effort_level = lagr_normalize_effort_level(effort_level);

    cfg->max_iter = 200;           /* total iterations of subgradient */
    cfg->heur_every = LAGR_EFFORT_ZERO_PRIMAL_UB_CADENCE;
    cfg->step_coef = 1.5;          /* initial step coefficient */
    cfg->step_min = 0.005;         /* minimal step coefficient */
    cfg->step_contract = 0.5;      /* current CCubes behavior: halve after stagnation */
    cfg->stabilization_beta = 1.0; /* no damping unless explicitly requested */
    cfg->halve_period = 8;         /* halve step if no progress for this many iterations */
    cfg->deflection_alpha = 0.0;   /* deflected subgradient direction, disabled by default */
    cfg->polish_enabled = 1;       /* reduced-cost fixing + bounded strong finish */
    cfg->local_passes = 0;
    cfg->drop2_enabled = 0;
    cfg->small_gap_threshold = 0;
    cfg->small_gap_extra_passes = cfg->local_passes;
    cfg->rarity_init = 0;
    cfg->polish_node_limit = 80000;
    cfg->polish_core_cap = 1500;
    cfg->portfolio_enabled = 0;
    cfg->portfolio_max_profiles = 1;
    cfg->portfolio_deflection_alpha = 0.75;
    cfg->hybrid_bundle_portfolio = 0;
    cfg->bundle_enabled = 0;
    cfg->bundle_interval = 1;
    cfg->bundle_memory = 8;
    cfg->bundle_prox_mu = 8.0;
    cfg->bundle_min_mu = 1.0;
    cfg->bundle_max_mu = 1024.0;
    cfg->bundle_null_expand = 2.0;
    cfg->bundle_serious_shrink = 0.8;
    cfg->bundle_serious_fraction = 0.05;
    cfg->bundle_serious_tol = 1e-9;
    cfg->bundle_momentum = 0.0;
    cfg->stop_at_step_floor = 1;
    cfg->reset_on_continuous_improvement = 0;
    cfg->polish_chart = NULL;
    cfg->bounded_workers = 1;
    cfg->adaptive_finish = effort_level == 2;

    if (effort_level == 1) {
        cfg->max_iter = 1000;
        cfg->heur_every = 5;
        cfg->step_coef = 2.0;
        cfg->local_passes = 10;
        cfg->drop2_enabled = 0;
        cfg->small_gap_extra_passes = cfg->local_passes;
        cfg->step_contract = 0.95;
        cfg->polish_enabled = 1;
        cfg->polish_node_limit = 400000;
        cfg->polish_core_cap = 8000;
        cfg->portfolio_enabled = 1;
        cfg->portfolio_max_profiles = 2;
    }

    if (effort_level >= 2) {
        cfg->max_iter = 5000;
        cfg->heur_every = 10;
        cfg->step_coef = 2.0;
        cfg->local_passes = 10;
        cfg->drop2_enabled = 1;
        cfg->small_gap_extra_passes = cfg->local_passes;
        cfg->step_contract = 0.95;
        cfg->polish_enabled = 1;
        cfg->polish_node_limit = 2000000;
        cfg->polish_core_cap = 30000;
        cfg->portfolio_enabled = 1;
        cfg->portfolio_max_profiles = 6;
        cfg->hybrid_bundle_portfolio = 1;
        cfg->bundle_interval = 8;
        cfg->bundle_momentum = 0.35;
    }

}

static void lagr_apply_certification_policy(
    LagrangianConfig *cfg,
    int effort_level,
    bool certification_requested
) {
    /*
     * Effort 2 is ordinarily a bounded strong heuristic, not an unconditional
     * attempt to reproduce an exact solver. The large portfolio and bounded
     * finish are reserved for a stopping state that has explicitly requested
     * certification.
     */
    if (lagr_normalize_effort_level(effort_level) < 2 ||
        certification_requested) {
        return;
    }

    cfg->max_iter = 1000;
    cfg->portfolio_enabled = 0;
    cfg->portfolio_max_profiles = 1;
    cfg->hybrid_bundle_portfolio = 0;
    cfg->bundle_enabled = 0;
    cfg->polish_node_limit = 400000;
    cfg->polish_core_cap = 8000;
}

static int lagr_build_portfolio_configs(
    const LagrangianConfig *baseline,
    double deflection_alpha,
    LagrangianConfig profiles[LAGR_PORTFOLIO_PROFILE_COUNT]
) {
    profiles[0] = *baseline;

    if (baseline->hybrid_bundle_portfolio) {
        LagrangianConfig bundle = *baseline;
        bundle.bundle_enabled = 1;
        bundle.deflection_alpha = 0.0;

        profiles[1] = *baseline;
        profiles[1].deflection_alpha = deflection_alpha;

        profiles[2] = bundle;
        profiles[2].deflection_alpha = 0.0;

        profiles[3] = bundle;
        profiles[3].deflection_alpha = deflection_alpha;

        profiles[4] = *baseline;
        profiles[4].max_iter = 500;
        profiles[4].step_contract = 0.99;
        profiles[4].deflection_alpha = 0.0;

        profiles[5] = profiles[4];
        profiles[5].deflection_alpha = deflection_alpha;

        return LAGR_PORTFOLIO_PROFILE_COUNT;
    }

    profiles[1] = *baseline;
    profiles[1].deflection_alpha = deflection_alpha;

    profiles[2] = *baseline;
    profiles[2].max_iter = lagr_max_int(baseline->max_iter, 80000);
    profiles[2].halve_period = lagr_max_int(baseline->halve_period, 16);

    profiles[3] = profiles[2];
    profiles[3].deflection_alpha = deflection_alpha;

    return 4;
}

static void lagr_merge_polish_stats(
    LagrangianStats *aggregate,
    const LagrangianStats *additional
) {
    if (!aggregate || !additional) return;
    if (
        aggregate->polish_initial_ub < 0 &&
        additional->polish_initial_ub >= 0
    ) {
        aggregate->polish_initial_ub =
            additional->polish_initial_ub;
        aggregate->polish_initial_target =
            additional->polish_initial_target;
        aggregate->polish_initial_core_count =
            additional->polish_initial_core_count;
    }
    if (additional->polish_search_target >= 0) {
        aggregate->polish_search_target =
            additional->polish_search_target;
        aggregate->polish_search_core_count =
            additional->polish_search_core_count;
        aggregate->polish_forced_columns =
            additional->polish_forced_columns;
    }
    if (additional->polish_final_ub >= 0) {
        if (
            aggregate->polish_final_ub < 0 ||
            additional->polish_final_ub <
                aggregate->polish_final_ub
        ) {
            aggregate->polish_final_ub =
                additional->polish_final_ub;
        }
    }
    aggregate->polish_skipped_targets +=
        additional->polish_skipped_targets;
    aggregate->polish_searches += additional->polish_searches;
    aggregate->polish_nodes += additional->polish_nodes;
    aggregate->polish_extra_nodes += additional->polish_extra_nodes;
    if (additional->polish_workers > aggregate->polish_workers)
        aggregate->polish_workers = additional->polish_workers;
    if (additional->polish_active_workers > aggregate->polish_active_workers)
        aggregate->polish_active_workers = additional->polish_active_workers;
    aggregate->polish_warm_nodes += additional->polish_warm_nodes;
    aggregate->polish_warm_work += additional->polish_warm_work;
    aggregate->polish_warm_fallbacks += additional->polish_warm_fallbacks;
    if (additional->polish_warm_peak_bytes > aggregate->polish_warm_peak_bytes)
        aggregate->polish_warm_peak_bytes = additional->polish_warm_peak_bytes;
    if (additional->polish_warm_workers > aggregate->polish_warm_workers)
        aggregate->polish_warm_workers = additional->polish_warm_workers;
    aggregate->polish_aborted_searches +=
        additional->polish_aborted_searches;
    aggregate->polish_exhausted_searches +=
        additional->polish_exhausted_searches;
    aggregate->polish_improvements +=
        additional->polish_improvements;
    aggregate->polish_proved_optimal =
        aggregate->polish_proved_optimal ||
        additional->polish_proved_optimal;
    aggregate->polish_seconds += additional->polish_seconds;
    if (
        additional->polish_stop_reason !=
        LAGR_POLISH_NOT_RUN
    ) {
        aggregate->polish_stop_reason =
            additional->polish_stop_reason;
    }
}

static int lagr_multiplier_schedule_policy(int rows) {

    /*
     * Small masters converged well under the established stabilized schedule,
     * and forcing the large-chart coefficients can change which columns a
     * heuristic pricer discovers. The frozen Fišer masters are an order of
     * magnitude above this conservative boundary.
     */
    return rows >= 1024
        ? LAGR_MULTIPLIER_SCHEDULE_SCALED
        : LAGR_MULTIPLIER_SCHEDULE_LEGACY;
}

static int lagr_multiplier_convergence_policy(int schedule) {

    /*
     * The adaptive policy was calibrated on large frozen masters. Keep the
     * established small-master trajectory unless it is explicitly selected.
     */
    return schedule == LAGR_MULTIPLIER_SCHEDULE_SCALED
        ? LAGR_MULTIPLIER_CONVERGENCE_ADAPTIVE
        : LAGR_MULTIPLIER_CONVERGENCE_LEGACY;
}

static int lagr_scaled_iteration_limit(
    int rows,
    int requested_limit
) {
    double scale = sqrt((double)rows);
    double factor = 10.0;
    int minimum = 1000;
    int maximum = 2500;
    if (requested_limit > 1000) {
        factor = 100.0;
        minimum = 7500;
        maximum = 20000;
    } else if (requested_limit > 200) {
        factor = 40.0;
        minimum = 2500;
        maximum = 7500;
    }
    double raw = ceil(factor * scale);
    int limit = raw >= (double)INT_MAX
        ? INT_MAX
        : (int)raw;
    if (limit < minimum) limit = minimum;
    if (limit > maximum) limit = maximum;
    return limit;
}

static double lagr_scaled_initial_step(int rows) {
    double coefficient = 0.75 * sqrt((double)rows);
    if (coefficient < 2.0) coefficient = 2.0;
    if (coefficient > 96.0) coefficient = 96.0;
    return coefficient;
}

static int lagr_scaled_stagnation_window(int rows) {
    double raw = ceil(sqrt((double)rows));
    int window = raw >= (double)INT_MAX
        ? INT_MAX
        : (int)raw;
    if (window < 100) window = 100;
    if (window > 250) window = 250;
    return window;
}

/*
 * Warm-started multiplier calls re-optimize a master that has only grown
 * since the previous solve, so the previous best multipliers are already
 * near-optimal. The budget is a third of the cold scaled schedule, with a
 * floor that keeps the trajectory productive on small charts.
 */
static int lagr_scaled_warm_iteration_limit(
    int rows,
    int requested_limit
) {
    int cold = lagr_scaled_iteration_limit(rows, requested_limit);
    int warm = cold / 3;
    if (warm < 150) warm = 150;
    if (warm > cold) warm = cold;
    return warm;
}

void lagrangian_calibration_options_default(
    LagrangianCalibrationOptions *options
) {
    if (!options) return;
    options->maximum_iterations = 1000;
    options->stagnation_window = 25;
    options->initial_step_coefficient = 0.05;
    options->minimum_step_coefficient = 1e-5;
    options->contraction = 0.8;
    options->stabilization_beta = 0.25;
    options->deflection_alpha = 0.0;
    options->stop_at_step_floor = true;
    options->reset_on_continuous_improvement = false;
}

bool lagrangian_calibrate_multipliers(
    const PIChartView *chart,
    int incumbent_upper_bound,
    const LagrangianCalibrationOptions *options,
    double *row_multipliers,
    int multiplier_count,
    LagrangianCalibrationStats *stats
) {
    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->initial_zlb = -DBL_MAX;
        stats->best_zlb = -DBL_MAX;
        stats->last_zlb = -DBL_MAX;
        stats->stop_reason = LAGR_STOP_NOT_RUN;
    }
    if (
        !chart ||
        !chart->bits ||
        chart->rows <= 0 ||
        chart->cols <= 0 ||
        incumbent_upper_bound <= 0 ||
        !options ||
        options->maximum_iterations <= 0 ||
        options->stagnation_window <= 0 ||
        options->initial_step_coefficient <= 0.0 ||
        options->minimum_step_coefficient <= 0.0 ||
        options->minimum_step_coefficient >
            options->initial_step_coefficient ||
        options->contraction <= 0.0 ||
        options->contraction >= 1.0 ||
        options->stabilization_beta <= 0.0 ||
        options->stabilization_beta > 1.0 ||
        options->deflection_alpha < 0.0 ||
        options->deflection_alpha >= 1.0 ||
        !row_multipliers ||
        multiplier_count < chart->rows ||
        !stats
    ) {
        return false;
    }

    int **rowsCovered = NULL;
    int *rowsCoveredCount = NULL;
    int **colsCovering = NULL;
    int *colsCoveringCount = NULL;
    int adjacency_status = build_adjacency(
        chart,
        chart->rows,
        chart->cols,
        &rowsCovered,
        &rowsCoveredCount,
        &colsCovering,
        &colsCoveringCount
    );
    if (adjacency_status != 0) return false;

    const LagrangianLoopThreads loop_threads = lagr_loop_threads(
        chart->rows, chart->cols, rowsCoveredCount
    );
    LagrangianWorkspace workspace = {0};
    /* Calibration only uses the subgradient workspace. */
    workspace.slack = (double *)malloc((size_t)chart->rows * sizeof(double));
    const bool workspace_ready = workspace.slack != NULL;
    double *multipliers = (double *)calloc(
        (size_t)chart->rows,
        sizeof(double)
    );
    double *best_multipliers = (double *)calloc(
        (size_t)chart->rows,
        sizeof(double)
    );
    double *reduced_costs = (double *)malloc(
        (size_t)chart->cols * sizeof(double)
    );
    double *column_costs = (double *)malloc(
        (size_t)chart->cols * sizeof(double)
    );
    unsigned char *dual_selection = (unsigned char *)malloc(
        (size_t)chart->cols
    );
    double *previous_direction =
        options->deflection_alpha > 0.0
            ? (double *)calloc(
                (size_t)chart->rows,
                sizeof(double)
            )
            : NULL;
    if (
        !workspace_ready ||
        !multipliers ||
        !best_multipliers ||
        !reduced_costs ||
        !column_costs ||
        !dual_selection ||
        (
            options->deflection_alpha > 0.0 &&
            !previous_direction
        )
    ) {
        free(multipliers);
        free(best_multipliers);
        free(reduced_costs);
        free(column_costs);
        free(dual_selection);
        lagr_workspace_destroy(&workspace);
        free(previous_direction);
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            chart->cols,
            colsCovering,
            colsCoveringCount,
            chart->rows
        );
        stats->stop_reason = LAGR_STOP_OOM;
        return false;
    }

    for (int column = 0; column < chart->cols; ++column) {
        column_costs[column] = 1.0;
    }
    lagr_initialize_pricing_multipliers(
        chart->rows,
        chart->cols,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        multipliers
    );

    LagrangianCalibrationStats *previous_calibration_stats = lagr_calibration_stats;
    lagr_calibration_stats = stats;
    double initial_zlb = compute_reduced_and_lb(
        chart->rows,
        chart->cols,
        rowsCovered,
        rowsCoveredCount,
        multipliers,
        column_costs,
        reduced_costs,
        dual_selection,
        loop_threads.columns
    );
    double best_zlb = initial_zlb;
    double last_zlb = initial_zlb;
    memcpy(
        best_multipliers,
        multipliers,
        (size_t)chart->rows * sizeof(double)
    );

    int iterations = 0;
    int continuous_improvements = 0;
    int last_continuous_improvement = 0;
    int contractions = 0;
    int stuck_iterations = 0;
    double best_integer_bound = -DBL_MAX;
    double step_coefficient =
        options->initial_step_coefficient;
    LagrangianStopReason stop_reason = LAGR_STOP_MAX_ITER;

    for (
        int iteration = 0;
        iteration < options->maximum_iterations;
        ++iteration
    ) {
        iterations = iteration + 1;
        double zlb = compute_reduced_and_lb(
            chart->rows,
            chart->cols,
            rowsCovered,
            rowsCoveredCount,
            multipliers,
            column_costs,
            reduced_costs,
            dual_selection,
            loop_threads.columns
        );
        last_zlb = zlb;

        bool continuous_improved = zlb > best_zlb + EPS;
        if (continuous_improved) {
            best_zlb = zlb;
            continuous_improvements++;
            last_continuous_improvement = iterations;
            memcpy(
                best_multipliers,
                multipliers,
                (size_t)chart->rows * sizeof(double)
            );
        }

        double integer_bound = ceil(zlb - EPS);
        bool integer_improved =
            integer_bound > best_integer_bound + EPS;
        if (integer_improved) {
            best_integer_bound = integer_bound;
        }
        if (
            (
                options->reset_on_continuous_improvement &&
                continuous_improved
            ) ||
            (
                !options->reset_on_continuous_improvement &&
                integer_improved
            )
        ) {
            stuck_iterations = 0;
        }

        if (
            (double)incumbent_upper_bound <=
            integer_bound + EPS
        ) {
            stop_reason = LAGR_STOP_OPTIMAL;
            break;
        }

        double step_before = step_coefficient;
        int updated = subgradient_update(
            chart->rows,
            colsCovering,
            colsCoveringCount,
            dual_selection,
            (double)incumbent_upper_bound,
            zlb,
            multipliers,
            &step_coefficient,
            options->minimum_step_coefficient,
            &stuck_iterations,
            options->stagnation_window,
            previous_direction,
            options->deflection_alpha,
            options->contraction,
            options->stabilization_beta,
            NULL,
            &workspace,
            loop_threads.rows
        );
        if (step_coefficient < step_before - EPS) {
            contractions++;
        }
        if (!updated) {
            stop_reason = LAGR_STOP_NO_UPDATE;
            break;
        }
        if (
            options->stop_at_step_floor &&
            step_coefficient <=
                options->minimum_step_coefficient + EPS
        ) {
            stop_reason = LAGR_STOP_STEP_MIN;
            break;
        }
    }

    memcpy(
        row_multipliers,
        best_multipliers,
        (size_t)chart->rows * sizeof(double)
    );
    stats->iterations = iterations;
    stats->continuous_improvements = continuous_improvements;
    stats->last_continuous_improvement_iteration =
        last_continuous_improvement;
    stats->step_contractions = contractions;
    stats->initial_zlb = initial_zlb;
    stats->best_zlb = best_zlb;
    stats->last_zlb = last_zlb;
    stats->final_step_coefficient = step_coefficient;
    stats->stop_reason = stop_reason;
    lagr_calibration_stats = previous_calibration_stats;

    free(multipliers);
    free(best_multipliers);
    free(reduced_costs);
    free(column_costs);
    free(dual_selection);
    lagr_workspace_destroy(&workspace);
    free(previous_direction);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        chart->cols,
        colsCovering,
        colsCoveringCount,
        chart->rows
    );
    return true;
}

/* ---------- Main functions ---------- */
static void solve_scp_lagrangian_config(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double weights[],
    const LagrangianConfig *cfg,
    const int *initial_solution,
    int initial_solmin,
    int *solution,
    int *solmin,
    double *ls_out, /* optional: reduced costs at the best multipliers */
    double *multipliers_out, /* optional: best row multipliers */
    const int *polish_incumbent,
    int polish_incumbent_cardinality,
    const double *initial_multipliers, /* optional: warm-start row prices */
    uint64_t deadline_nanoseconds
) {
    if (solmin) *solmin = -1;
    lagr_stats_begin(rows, cols, 0);
    if (
        rows <= 0 ||
        cols <= 0 ||
        !rowsCovered ||
        !rowsCoveredCount ||
        !colsCovering ||
        !colsCoveringCount ||
        !cfg ||
        !solution ||
        !solmin
    ) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }
    lagr_last_stats.polish_enabled = cfg->polish_enabled;
    lagr_last_stats.polish_node_limit =
        cfg->polish_node_limit;
    lagr_last_stats.polish_core_cap = cfg->polish_core_cap;
    if (!cfg->polish_enabled) {
        lagr_last_stats.polish_stop_reason =
            LAGR_POLISH_DISABLED;
    }

    const LagrangianLoopThreads loop_threads = lagr_loop_threads(rows, cols, rowsCoveredCount);
    LagrangianWorkspace workspace = {0};
    const bool workspace_ready = lagr_workspace_init(&workspace, rows, cols);
    double *t = (double*)calloc((size_t)rows, sizeof(double));
    double *t_best = (double*)calloc((size_t)rows, sizeof(double));
    double *ls = (double*)malloc((size_t)cols * sizeof(double));
    int *sol_tmp = (int*)malloc((size_t)cols * sizeof(int));
    int *sol_tmp2 = (int*)malloc((size_t)cols * sizeof(int));
    double *col_costs = (double*)malloc((size_t)cols * sizeof(double));
    unsigned char *dual_x = (unsigned char*)malloc((size_t)cols);
    int best_sol_size = -1;
    int stuck_iter = 0;

    if (!workspace_ready || !t || !t_best || !ls || !sol_tmp || !sol_tmp2 || !col_costs || !dual_x) {
        free(t);
        free(t_best);
        free(ls);
        free(sol_tmp);
        free(sol_tmp2);
        free(col_costs);
        free(dual_x);
        lagr_workspace_destroy(&workspace);

        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
        return;
    }

    double max_w = 0.0;
    if (weights) {
        for (int j = 0; j < cols; ++j) {
            if (weights[j] > max_w) max_w = weights[j];
        }
    }
    double eps = 0.0;
    if (max_w > 0.0) {
        eps = 1e-6 / (max_w + 1.0);
    }
    for (int j = 0; j < cols; ++j) {
        col_costs[j] = 1.0 - eps * (weights ? weights[j] : 0.0);
    }

    int local_passes = cfg->local_passes;
    int max_iter = cfg->max_iter;
    int heur_every = cfg->heur_every;
    lagr_last_stats.primal_ub_cadence = heur_every;
    double step_coef = cfg->step_coef;
    double step_min = cfg->step_min;
    double step_contract = cfg->step_contract;
    double stabilization_beta = cfg->stabilization_beta;
    int halve_period = cfg->halve_period;
    int stop_at_step_floor = cfg->stop_at_step_floor;
    int reset_on_continuous_improvement =
        cfg->reset_on_continuous_improvement;
    double deflection_alpha = cfg->deflection_alpha;
    double *prev_direction = NULL;
    LagrangianBundleState bundle_state = {0};
    int continuous_improvements = 0;
    int last_continuous_improvement_iteration = 0;
    int primal_ub_recurring_calls = 0;
    int primal_ub_improving_calls = 0;
    int primal_ub_improving_calls_below_supplied = 0;
    int primal_ub_first_improvement_iteration = 0;
    int primal_ub_last_improvement_iteration = 0;
    int primal_ub_longest_improvement_gap = 0;
    int step_contractions = 0;
    int multiplier_schedule = LAGR_MULTIPLIER_SCHEDULE_NONE;
    int multiplier_convergence =
        LAGR_MULTIPLIER_CONVERGENCE_NONE;
    int iteration_limit_reason =
        LAGR_ITERATION_LIMIT_CONFIGURED;

    if (multipliers_out) {
        /*
         * Pricing needs a useful continuous dual vector, not merely a fast
         * integer lower-bound ceiling. Frozen-master calibration showed that
         * the former small stabilized steps never improved the dual-ascent
         * start on large charts. Scale the Polyak coefficient and work budget
         * with chart size while preserving the old policy for ablation.
         */
        multiplier_schedule =
            lagr_multiplier_schedule_policy(rows);
        multiplier_convergence =
            lagr_multiplier_convergence_policy(
                multiplier_schedule
            );

        if (
            multiplier_schedule ==
            LAGR_MULTIPLIER_SCHEDULE_LEGACY
        ) {
            if (max_iter < 1000) {
                max_iter = 1000;
                iteration_limit_reason =
                    LAGR_ITERATION_LIMIT_LEGACY_FLOOR;
            }
            step_coef = 0.05;
            step_min = 1e-5;
            step_contract = 0.8;
            stabilization_beta = 0.25;
            if (halve_period < 25) halve_period = 25;
        } else {
            const bool warm_scaled =
                initial_multipliers != NULL;
            max_iter = warm_scaled
                ? lagr_scaled_warm_iteration_limit(
                    rows,
                    cfg->max_iter
                )
                : lagr_scaled_iteration_limit(
                    rows,
                    cfg->max_iter
                );
            iteration_limit_reason = warm_scaled
                ? LAGR_ITERATION_LIMIT_WARM_SCALED
                : LAGR_ITERATION_LIMIT_SCALED_BY_ROWS;
            step_coef = lagr_scaled_initial_step(rows);
            step_min = 1e-8;
            step_contract = 0.99;
            stabilization_beta = 1.0;
            halve_period =
                lagr_scaled_stagnation_window(rows);
            stop_at_step_floor = 0;
        }
        if (
            multiplier_convergence ==
            LAGR_MULTIPLIER_CONVERGENCE_ADAPTIVE
        ) {
            /*
             * Reset stagnation on the continuous quantity being optimized.
             * A stronger contraction lets an oversized row-scaled Polyak
             * coefficient return to a productive range within the bounded
             * iteration budget. The step minimum remains a floor rather than
             * a convergence claim.
             */
            reset_on_continuous_improvement = 1;
            step_contract = 0.8;
            stop_at_step_floor = 0;
        }
    }
    lagr_last_stats.requested_iteration_limit = cfg->max_iter;
    lagr_last_stats.iteration_limit = max_iter;
    lagr_last_stats.iteration_limit_reason =
        iteration_limit_reason;
    lagr_last_stats.multiplier_schedule =
        multiplier_schedule;
    lagr_last_stats.multiplier_convergence =
        multiplier_convergence;
    lagr_last_stats.reset_on_continuous_improvement =
        reset_on_continuous_improvement;
    lagr_last_stats.stop_at_step_floor =
        stop_at_step_floor;
    lagr_last_stats.requested_initial_step_coef =
        cfg->step_coef;
    lagr_last_stats.requested_step_min = cfg->step_min;
    lagr_last_stats.requested_step_contract =
        cfg->step_contract;
    lagr_last_stats.requested_stabilization_beta =
        cfg->stabilization_beta;
    lagr_last_stats.requested_deflection_alpha =
        cfg->deflection_alpha;
    lagr_last_stats.initial_step_coef = step_coef;
    lagr_last_stats.step_min = step_min;
    lagr_last_stats.step_contract = step_contract;
    lagr_last_stats.stabilization_beta = stabilization_beta;
    lagr_last_stats.deflection_alpha = deflection_alpha;
    lagr_last_stats.warm_start =
        (
            multipliers_out != NULL &&
            initial_multipliers != NULL &&
            multiplier_schedule == LAGR_MULTIPLIER_SCHEDULE_SCALED
        ) ? 1 : 0;

    if (deflection_alpha > 0.0) {
        prev_direction = (double*)calloc((size_t)rows, sizeof(double));
        if (!prev_direction) {
            free(t);
            free(t_best);
            free(ls);
            free(sol_tmp);
            free(sol_tmp2);
            free(col_costs);
            free(dual_x);
            lagr_workspace_destroy(&workspace);
            lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
            return;
        }
    }

    if (cfg->bundle_enabled) {
        int memory = cfg->bundle_memory > 0 ? cfg->bundle_memory : 1;
        bundle_state.rows = rows;
        bundle_state.memory = memory;
        bundle_state.prox_mu = cfg->bundle_prox_mu;
        bundle_state.min_mu = cfg->bundle_min_mu;
        bundle_state.max_mu = cfg->bundle_max_mu;
        bundle_state.null_expand = cfg->bundle_null_expand;
        bundle_state.serious_shrink = cfg->bundle_serious_shrink;
        bundle_state.serious_fraction = cfg->bundle_serious_fraction;
        bundle_state.serious_tol = cfg->bundle_serious_tol;
        bundle_state.momentum = cfg->bundle_momentum;
        bundle_state.center_value = -DBL_MAX;
        bundle_state.center_t = (double*)calloc((size_t)rows, sizeof(double));
        bundle_state.prev_center_t = (double*)calloc((size_t)rows, sizeof(double));
        bundle_state.cut_alpha = (double*)calloc((size_t)memory, sizeof(double));
        bundle_state.cut_slacks = (double*)calloc((size_t)memory * (size_t)rows, sizeof(double));
        if (!bundle_state.center_t || !bundle_state.prev_center_t || !bundle_state.cut_alpha || !bundle_state.cut_slacks) {
            free(bundle_state.center_t);
            free(bundle_state.prev_center_t);
            free(bundle_state.cut_alpha);
            free(bundle_state.cut_slacks);
            free(prev_direction);
            free(t);
            free(t_best);
            free(ls);
            free(sol_tmp);
            free(sol_tmp2);
            free(col_costs);
            free(dual_x);
            lagr_workspace_destroy(&workspace);
            lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
            return;
        }
    }

    if (multipliers_out) {
        if (
            initial_multipliers != NULL &&
            multiplier_schedule == LAGR_MULTIPLIER_SCHEDULE_SCALED
        ) {
            /*
             * Warm start: the previous solve of this restricted master
             * (or one of its subsets) already produced near-optimal row
             * prices. Seed the subgradient trajectory from them instead
             * of re-deriving a dual-feasible start from scratch.
             */
            memcpy(
                t,
                initial_multipliers,
                (size_t)rows * sizeof(double)
            );
        } else {
            lagr_initialize_pricing_multipliers(
                rows,
                cols,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                t
            );
        }
    } else {
        lagr_initialize_multipliers(
            rows,
            colsCoveringCount,
            cfg->rarity_init,
            t
        );
    }

    double bestZLB = -DBL_MAX;
    double lastZLB = -DBL_MAX;
    int iterations = 0;
    LagrangianStopReason stop_reason = LAGR_STOP_MAX_ITER;

    /* Initial heuristics with current ls (computed from t) */
    uint64_t component_started = lagr_monotonic_nanoseconds();
    double initialZLB = compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        col_costs,
        ls,
        dual_x,
        loop_threads.columns
    );
    lagr_last_stats.reduced_cost_nanoseconds +=
        lagr_monotonic_nanoseconds() - component_started;
    lagr_last_stats.reduced_cost_calls++;
    bestZLB = initialZLB;
    lastZLB = initialZLB;
    memcpy(t_best, t, (size_t)rows * sizeof(double));

    int sol_size = -1, sol_size2 = -1;
    component_started = lagr_monotonic_nanoseconds();

    heuristic_row_min_rc(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        ls,
        weights,
        sol_tmp,
        &sol_size,
        &workspace
    );

    const uint64_t greedy_started = lagr_monotonic_nanoseconds();
    greedy_from_lagr_scores(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        ls,
        weights,
        sol_tmp2,
        &sol_size2,
        &workspace
    );
    const uint64_t greedy_finished = lagr_monotonic_nanoseconds();
    lagr_last_stats.heuristic_row_min_nanoseconds += greedy_started - component_started;
    lagr_last_stats.heuristic_greedy_nanoseconds += greedy_finished - greedy_started;

    /* Optional deeper search: disabled by default */
    if (local_passes > 0 && (sol_size > 0 || sol_size2 > 0)) {
        if (sol_size > 0) {
            local_search_drop1_repair(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp,
                &sol_size,
                local_passes
            );

            local_search_drop2_repair(
                rows,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp,
                &sol_size,
                local_passes,
                deadline_nanoseconds
            );

        }

        if (sol_size2 > 0) {
            local_search_drop1_repair(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp2,
                &sol_size2,
                local_passes
            );

            local_search_drop2_repair(
                rows,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp2,
                &sol_size2,
                local_passes,
                deadline_nanoseconds
            );
        }
        lagr_last_stats.heuristic_repair_nanoseconds +=
            lagr_monotonic_nanoseconds() - greedy_finished;
    }
    lagr_last_stats.heuristic_nanoseconds +=
        lagr_monotonic_nanoseconds() - component_started;
    lagr_last_stats.heuristic_calls++;

    int have_initial = initial_solution != NULL && initial_solmin > 0 && initial_solmin <= cols;

    if (sol_size == -1 && sol_size2 == -1 && !have_initial) {
        free(t);
        free(t_best);
        free(ls);
        free(sol_tmp);
        free(sol_tmp2);
        free(col_costs);
        free(dual_x);
        lagr_workspace_destroy(&workspace);
        free(prev_direction);
        free(bundle_state.center_t);
        free(bundle_state.prev_center_t);
        free(bundle_state.cut_alpha);
        free(bundle_state.cut_slacks);
        lagr_stats_finish(-1, -DBL_MAX, bestZLB, lastZLB, step_coef, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }

    double w1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
    double w2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

    double bestUB = DBL_MAX; /* UB equals current solution length (unit-cost) */
    int bestTerms = INT_MAX;
    double bestWeight = -DBL_MAX;

    if (sol_size != -1) {
        bestUB = (double)sol_size;
        bestTerms = sol_size;
        bestWeight = w1;
        best_sol_size = sol_size;
        memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
    }

    if (
        sol_size2 != -1 &&
        (
            sol_size2 < bestTerms ||
            (sol_size2 == bestTerms && w2 > bestWeight + EPS)
        )
    ) {
        bestUB = (double)sol_size2;
        bestTerms = sol_size2;
        bestWeight = w2;
        best_sol_size = sol_size2;
        memcpy(solution, sol_tmp2, (size_t)sol_size2 * sizeof(int));
    }

    if (have_initial) {
        double initial_weight = solution_total_weight(initial_solution, initial_solmin, weights);
        if (
            initial_solmin < bestTerms ||
            (initial_solmin == bestTerms && initial_weight > bestWeight + EPS)
        ) {
            bestUB = (double)initial_solmin;
            bestTerms = initial_solmin;
            bestWeight = initial_weight;
            best_sol_size = initial_solmin;
            memcpy(solution, initial_solution, (size_t)initial_solmin * sizeof(int));
        }
    }
    lagr_last_stats.primal_ub_supplied = have_initial
        ? initial_solmin
        : -1;
    lagr_last_stats.primal_ub_loop_start = bestTerms;

    double bestLB = -DBL_MAX;
    for (int it = 0; it < max_iter; ++it) {
        if (
            deadline_nanoseconds > 0 &&
            lagr_monotonic_nanoseconds() >= deadline_nanoseconds
        ) {
            stop_reason = LAGR_STOP_DEADLINE;
            break;
        }
        iterations = it + 1;
        component_started = lagr_monotonic_nanoseconds();
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            col_costs,
            ls,
            dual_x,
            loop_threads.columns
        );
        lagr_last_stats.reduced_cost_nanoseconds +=
            lagr_monotonic_nanoseconds() - component_started;
        lagr_last_stats.reduced_cost_calls++;
        lastZLB = ZLB;
        if (ZLB > bestZLB + EPS) {
            bestZLB = ZLB;
            memcpy(t_best, t, (size_t)rows * sizeof(double));
            continuous_improvements++;
            last_continuous_improvement_iteration = iterations;
            if (reset_on_continuous_improvement) {
                stuck_iter = 0;
            }
        }

        double LBint = ceil(ZLB - 1e-12);

        if (LBint > bestLB + EPS) {
            bestLB = LBint;
            if (!reset_on_continuous_improvement) {
                stuck_iter = 0;
            }
        } /* else no improvement */

        /*
         * Warm-started calls re-optimize a master that has only grown:
         * when the continuous bound has not improved for a full
         * stagnation window, the remaining cold-schedule iterations are
         * re-deriving what the previous solve already knew.
         */
        if (
            lagr_last_stats.warm_start &&
            iterations >= halve_period &&
            iterations - last_continuous_improvement_iteration >=
                halve_period
        ) {
            stop_reason = LAGR_STOP_WARM_STALL;
            break;
        }

        /*
         * The initial heuristic immediately above already used the current
         * multipliers.  At it == 0 they have not changed yet, so repeating
         * the same deterministic pair would only duplicate that work.
         */
        if (it > 0 && it % heur_every == 0) {
            const int primal_ub_before = bestTerms;
            primal_ub_recurring_calls++;
            component_started = lagr_monotonic_nanoseconds();
            heuristic_row_min_rc(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp,
                &sol_size,
                &workspace
            );

            const uint64_t greedy_started = lagr_monotonic_nanoseconds();
            greedy_from_lagr_scores(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp2,
                &sol_size2,
                &workspace
            );
            const uint64_t greedy_finished = lagr_monotonic_nanoseconds();
            lagr_last_stats.heuristic_row_min_nanoseconds += greedy_started - component_started;
            lagr_last_stats.heuristic_greedy_nanoseconds += greedy_finished - greedy_started;

            bool found_new_best = false;
            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

            /* Improve by primary objective: fewer terms; tie-break by higher weight */
            if (
                sol_size != -1 &&
                (
                    sol_size < bestTerms ||
                    (
                        sol_size == bestTerms && candW1 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size;
                bestTerms = sol_size;
                bestWeight = candW1;
                best_sol_size = sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
                found_new_best = true;
            }

            if (
                sol_size2 != -1 &&
                (
                    sol_size2 < bestTerms ||
                    (
                        sol_size2 == bestTerms && candW2 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size2;
                bestTerms = sol_size2;
                bestWeight = candW2;
                best_sol_size = sol_size2;
                memcpy(solution, sol_tmp2, (size_t)sol_size2 * sizeof(int));
                found_new_best = true;
            }

            if (found_new_best && local_passes > 0) {
                const uint64_t repair_started = lagr_monotonic_nanoseconds();
                int current_sol_size = best_sol_size;
                int *current_sol = (int*)malloc((size_t)cols * sizeof(int));
                if (current_sol) {
                    memcpy(current_sol, solution, (size_t)current_sol_size * sizeof(int));

                    local_search_drop1_repair(
                        rows,
                        cols,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        ls,
                        weights,
                        current_sol,
                        &current_sol_size,
                        local_passes
                    );

                    if (cfg->drop2_enabled) {
                        local_search_drop2_repair(
                            rows,
                            rowsCovered,
                            rowsCoveredCount,
                            colsCovering,
                            colsCoveringCount,
                            ls,
                            weights,
                            current_sol,
                            &current_sol_size,
                            local_passes,
                            deadline_nanoseconds
                        );
                    }

                    double polished_weight = solution_total_weight(current_sol, current_sol_size, weights);
                    if (
                        current_sol_size < bestTerms ||
                        (
                            current_sol_size == bestTerms && polished_weight > bestWeight + EPS
                        )
                    ) {
                        bestUB = (double)current_sol_size;
                        bestTerms = current_sol_size;
                        bestWeight = polished_weight;
                        best_sol_size = current_sol_size;
                        memcpy(solution, current_sol, (size_t)current_sol_size * sizeof(int));
                    }
                    free(current_sol);
                }
                lagr_last_stats.heuristic_repair_nanoseconds +=
                    lagr_monotonic_nanoseconds() - repair_started;
            }
            lagr_last_stats.heuristic_nanoseconds +=
                lagr_monotonic_nanoseconds() - component_started;
            lagr_last_stats.heuristic_calls++;
            if (bestTerms < primal_ub_before) {
                const int gap = iterations -
                    primal_ub_last_improvement_iteration;
                if (primal_ub_first_improvement_iteration == 0) {
                    primal_ub_first_improvement_iteration = iterations;
                }
                primal_ub_last_improvement_iteration = iterations;
                if (gap > primal_ub_longest_improvement_gap) {
                    primal_ub_longest_improvement_gap = gap;
                }
                primal_ub_improving_calls++;
                if (have_initial && bestTerms < initial_solmin) {
                    primal_ub_improving_calls_below_supplied++;
                }
            }
        }

        if (bestUB <= LBint + EPS) {
            stop_reason = LAGR_STOP_OPTIMAL;
            break; /* proved optimal (within epsilon) */
        }

        int used_subgradient_update = 0;
        int updated = 0;
        int nonzero_subgradient_rows = -1;

        int use_bundle_update = cfg->bundle_enabled &&
            (cfg->bundle_interval <= 1 || it % cfg->bundle_interval == 0);

        if (use_bundle_update) {
            component_started = lagr_monotonic_nanoseconds();
            updated = bundle_update(
                rows,
                colsCovering,
                colsCoveringCount,
                dual_x,
                bestUB,
                ZLB,
                t,
                &bundle_state,
                &nonzero_subgradient_rows,
                loop_threads.rows
            );
            lagr_last_stats.bundle_nanoseconds +=
                lagr_monotonic_nanoseconds() - component_started;
            lagr_last_stats.bundle_calls++;
        }

        if (!use_bundle_update || !updated) {
            double step_before = step_coef;
            component_started = lagr_monotonic_nanoseconds();
            updated = subgradient_update(
                rows,
                colsCovering,
                colsCoveringCount,
                dual_x,
                bestUB,
                ZLB,
                t,
                &step_coef,
                step_min,
                &stuck_iter,
                halve_period,
                prev_direction,
                deflection_alpha,
                step_contract,
                stabilization_beta,
                &nonzero_subgradient_rows,
                &workspace,
                loop_threads.rows
            );
            lagr_last_stats.subgradient_nanoseconds +=
                lagr_monotonic_nanoseconds() - component_started;
            lagr_last_stats.subgradient_calls++;
            used_subgradient_update = 1;
            if (step_coef < step_before - EPS) {
                step_contractions++;
            }
        }
        if (nonzero_subgradient_rows >= 0) {
            lagr_last_stats.nonzero_subgradient_rows +=
                (uint64_t)nonzero_subgradient_rows;
            lagr_last_stats.subgradient_samples++;
            if (
                nonzero_subgradient_rows <
                lagr_last_stats.nonzero_subgradient_minimum
            ) {
                lagr_last_stats.nonzero_subgradient_minimum =
                    nonzero_subgradient_rows;
            }
            if (
                nonzero_subgradient_rows >
                lagr_last_stats.nonzero_subgradient_maximum
            ) {
                lagr_last_stats.nonzero_subgradient_maximum =
                    nonzero_subgradient_rows;
            }
        }

        if (
            !updated ||
            (
                used_subgradient_update &&
                stop_at_step_floor &&
                step_coef <= step_min + EPS
            )
        ) {
            stop_reason = updated ? LAGR_STOP_STEP_MIN : LAGR_STOP_NO_UPDATE;
            /* final refresh */
            heuristic_row_min_rc(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp,
                &sol_size,
                &workspace
            );

            greedy_from_lagr_scores(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                sol_tmp2,
                &sol_size2,
                &workspace
            );

            bool found_new_best = false;
            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

            if (
                sol_size != -1 &&
                (
                    sol_size < bestTerms ||
                    (
                        sol_size == bestTerms && candW1 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size;
                bestTerms = sol_size;
                bestWeight = candW1;
                best_sol_size = sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
                found_new_best = true;
            }

            if (
                sol_size2 != -1 &&
                (
                    sol_size2 < bestTerms ||
                    (
                        sol_size2 == bestTerms && candW2 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size2;
                bestTerms = sol_size2;
                bestWeight = candW2;
                best_sol_size = sol_size2;
                memcpy(solution, sol_tmp2, (size_t)sol_size2 * sizeof(int));
                found_new_best = true;
            }

            if (found_new_best && local_passes > 0) {
                int current_sol_size = best_sol_size;
                int *current_sol = (int*)malloc((size_t)cols * sizeof(int));
                if (current_sol) {
                    memcpy(current_sol, solution, (size_t)current_sol_size * sizeof(int));

                    local_search_drop1_repair(
                        rows,
                        cols,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        ls,
                        weights,
                        current_sol,
                        &current_sol_size,
                        local_passes
                    );

                    if (cfg->drop2_enabled) {
                        local_search_drop2_repair(
                            rows,
                            rowsCovered,
                            rowsCoveredCount,
                            colsCovering,
                            colsCoveringCount,
                            ls,
                            weights,
                            current_sol,
                            &current_sol_size,
                            local_passes,
                            deadline_nanoseconds
                        );
                    }

                    double polished_weight = solution_total_weight(current_sol, current_sol_size, weights);
                    if (
                        current_sol_size < bestTerms ||
                        (
                            current_sol_size == bestTerms && polished_weight > bestWeight + EPS
                        )
                    ) {
                        bestUB = (double)current_sol_size;
                        bestTerms = current_sol_size;
                        bestWeight = polished_weight;
                        best_sol_size = current_sol_size;
                        memcpy(solution, current_sol, (size_t)current_sol_size * sizeof(int));
                    }
                    free(current_sol);
                }
            }
            break;
        }
    }

    {
        const int trailing_gap = iterations -
            primal_ub_last_improvement_iteration;
        if (trailing_gap > primal_ub_longest_improvement_gap) {
            primal_ub_longest_improvement_gap = trailing_gap;
        }
    }
    lagr_last_stats.primal_ub_loop_final = bestTerms;
    lagr_last_stats.primal_ub_recurring_calls =
        primal_ub_recurring_calls;
    lagr_last_stats.primal_ub_improving_calls =
        primal_ub_improving_calls;
    lagr_last_stats.primal_ub_improving_calls_below_supplied =
        primal_ub_improving_calls_below_supplied;
    lagr_last_stats.primal_ub_first_improvement_iteration =
        primal_ub_first_improvement_iteration;
    lagr_last_stats.primal_ub_last_improvement_iteration =
        primal_ub_last_improvement_iteration;
    lagr_last_stats.primal_ub_longest_improvement_gap =
        primal_ub_longest_improvement_gap;

    if (
        polish_incumbent &&
        polish_incumbent_cardinality > 0 &&
        (
            best_sol_size < 0 ||
            polish_incumbent_cardinality < best_sol_size
        )
    ) {
        memcpy(
            solution,
            polish_incumbent,
            (size_t)polish_incumbent_cardinality *
            sizeof(int)
        );
        best_sol_size = polish_incumbent_cardinality;
        bestTerms = polish_incumbent_cardinality;
        bestUB = (double)polish_incumbent_cardinality;
        bestWeight = solution_total_weight(
            polish_incumbent,
            polish_incumbent_cardinality,
            weights
        );
    }

    if (
        cfg->polish_enabled &&
        best_sol_size > 1 &&
        stop_reason != LAGR_STOP_OPTIMAL &&
        !(
            deadline_nanoseconds > 0 &&
            lagr_monotonic_nanoseconds() >= deadline_nanoseconds
        )
    ) {
        /* reduced costs at the best multipliers found */
        double zlb_best = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t_best,
            col_costs,
            ls,
            dual_x,
            loop_threads.columns
        );

        int proved = 0;
        lagr_bounded_finish(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            colsCovering,
            colsCoveringCount,
            weights,
            ls,
            zlb_best,
            cfg->polish_node_limit,
            cfg->polish_core_cap,
            solution,
            &best_sol_size,
            &bestLB,
            &proved,
            1, NULL, NULL, NULL, NULL,
            deadline_nanoseconds,
            cfg->polish_chart,
            (size_t)(cfg->polish_node_limit <= 80000 ? 32 : 128) * 1024 * 1024,
            (unsigned)ccubes_thread_count(),
            cfg->bounded_workers,
            cfg->adaptive_finish
        );

        if ((double)best_sol_size < bestUB - EPS) {
            bestUB = (double)best_sol_size;
            bestTerms = best_sol_size;
            bestWeight = solution_total_weight(solution, best_sol_size, weights);
        }

        if (proved) {
            stop_reason = LAGR_STOP_OPTIMAL;
        }
    }

    if (ls_out) {
        /* reduced costs at the best multipliers, for reuse by the caller
           (branching priorities of the bounded finish) */
        component_started = lagr_monotonic_nanoseconds();
        compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t_best,
            col_costs,
            ls_out,
            dual_x,
            loop_threads.columns
        );
        lagr_last_stats.reduced_cost_nanoseconds +=
            lagr_monotonic_nanoseconds() - component_started;
        lagr_last_stats.reduced_cost_calls++;
    }
    if (multipliers_out) {
        memcpy(
            multipliers_out,
            t_best,
            (size_t)rows * sizeof(double)
        );
    }

    *solmin = best_sol_size;
    lagr_stats_finish(best_sol_size, bestLB, bestZLB, lastZLB, step_coef, iterations, stop_reason);
    lagr_last_stats.continuous_improvements =
        continuous_improvements;
    lagr_last_stats.last_continuous_improvement_iteration =
        last_continuous_improvement_iteration;
    lagr_last_stats.step_contractions = step_contractions;

    free(t);
    free(t_best);
    free(ls);
    free(sol_tmp);
    free(sol_tmp2);
    free(col_costs);
    free(dual_x);
    lagr_workspace_destroy(&workspace);
    free(prev_direction);
    free(bundle_state.center_t);
    free(bundle_state.prev_center_t);
    free(bundle_state.cut_alpha);
    free(bundle_state.cut_slacks);
}

static void solve_scp_lagrangian_with_deadline(
    const PIChartView *chart,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested,
    double *row_multipliers,
    int multiplier_count,
    const int *polish_incumbent,
    int polish_incumbent_cardinality,
    const double *initial_multipliers,
    uint64_t deadline_nanoseconds,
    bool native_ipc,
    int bounded_workers
) {
    if (solmin) *solmin = -1;
    const int foundPI = chart ? chart->cols : 0;
    const int ON_minterms = chart ? chart->rows : 0;
    lagr_stats_begin(ON_minterms, foundPI, 0);
    if (
        !chart ||
        !chart->bits ||
        foundPI <= 0 ||
        ON_minterms <= 0 ||
        !solution ||
        !solmin ||
        (
            row_multipliers &&
            multiplier_count < ON_minterms
        ) ||
        (
            polish_incumbent &&
            (
                polish_incumbent_cardinality <= 0 ||
                polish_incumbent_cardinality > foundPI
            )
        )
    ) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }
    double *initial_multipliers_copy = NULL;
    if (
        row_multipliers != NULL &&
        initial_multipliers == row_multipliers
    ) {
        /*
         * The progressive master intentionally reuses one buffer for the
         * previous prices and the next output. Preserve that aliased input
         * before restoring the output buffer's failure-state contract.
         */
        initial_multipliers_copy = (double *)malloc(
            (size_t)ON_minterms * sizeof(double)
        );
        if (!initial_multipliers_copy) {
            memset(
                row_multipliers,
                0,
                (size_t)ON_minterms * sizeof(double)
            );
            lagr_stats_finish(
                -1,
                -DBL_MAX,
                -DBL_MAX,
                -DBL_MAX,
                0.0,
                0,
                LAGR_STOP_OOM
            );
            return;
        }
        memcpy(
            initial_multipliers_copy,
            initial_multipliers,
            (size_t)ON_minterms * sizeof(double)
        );
        initial_multipliers = initial_multipliers_copy;
    }
    if (row_multipliers) {
        memset(
            row_multipliers,
            0,
            (size_t)ON_minterms * sizeof(double)
        );
    }
    int **rowsCovered = NULL, *rowsCoveredCount = NULL;
    int **colsCovering = NULL, *colsCoveringCount = NULL;

    /* A compact repair chart cannot amortize launching the full IPC
       team for each of the two adjacency passes. Bounded search still
       receives its own worker allowance. */
    const int repair_team = lagr_reduction_threads(
        foundPI, (uint64_t)foundPI * (unsigned)chart->words, bounded_workers
    );
    int rc_ad = build_adjacency_with_threads(
        chart,
        ON_minterms,
        foundPI,
        bounded_workers > 1 ? repair_team : ccubes_thread_count(),
        &rowsCovered,
        &rowsCoveredCount,
        &colsCovering,
        &colsCoveringCount
    );

    if (rc_ad == -2) { /* infeasible matrix (some row uncovered) */
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_INFEASIBLE);
        free(initial_multipliers_copy);
        return;
    }

    if (rc_ad == -1) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
        free(initial_multipliers_copy);
        return;
    }

    int warm_start_requested =
        initial_solution != NULL && initial_solmin > 0;
    int warm_start_accepted =
        warm_start_requested &&
        cover_is_feasible(
            chart,
            initial_solution,
            initial_solmin
        );
    if (!warm_start_accepted) {
        initial_solution = NULL;
        initial_solmin = 0;
    }
    int polish_incumbent_accepted =
        polish_incumbent != NULL &&
        polish_incumbent_cardinality > 0 &&
        cover_is_feasible(
            chart,
            polish_incumbent,
            polish_incumbent_cardinality
        );
    if (!polish_incumbent_accepted) {
        polish_incumbent = NULL;
        polish_incumbent_cardinality = 0;
    }

    /* shrink the instance once; benefits every portfolio profile */
    int presolve_removed = presolve_dominated_columns(
        chart,
        ON_minterms,
        foundPI,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        weights,
        repair_team
    );

    LagrangianConfig baseline;
    lagr_config_for_effort(&baseline, effort_level);
    baseline.adaptive_finish = baseline.adaptive_finish && !native_ipc;
    baseline.bounded_workers = baseline.adaptive_finish ? ccubes_thread_count() : bounded_workers;
    if (native_ipc) baseline.polish_chart = chart;
    lagr_apply_certification_policy(
        &baseline,
        effort_level,
        certification_requested
    );
    int portfolio_enabled = baseline.portfolio_enabled;
    double portfolio_deflection_alpha = baseline.portfolio_deflection_alpha;

    if (portfolio_enabled) {
        baseline.deflection_alpha = 0.0;
    }

    int *candidate = NULL;
    double *candidate_multipliers = NULL;

    solve_scp_lagrangian_config(
        ON_minterms,
        foundPI,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        weights,
        &baseline,
        initial_solution,
        initial_solmin,
        solution,
        solmin,
        NULL,
        row_multipliers,
        polish_incumbent,
        polish_incumbent_cardinality,
        initial_multipliers,
        deadline_nanoseconds
    );

    LagrangianStats merged_stats = *lagrangian_last_stats();
    int best_solmin = *solmin;
    int total_iterations = merged_stats.iterations;

    if (
        best_solmin < 0 ||
        !portfolio_enabled ||
        (
            merged_stats.gap == 0 &&
            merged_stats.stop_reason == LAGR_STOP_OPTIMAL
        )
    ) {
        goto cleanup;
    }

    {
        LagrangianConfig profiles[LAGR_PORTFOLIO_PROFILE_COUNT];
        int profile_count = lagr_build_portfolio_configs(&baseline, portfolio_deflection_alpha, profiles);
        int max_profiles = baseline.portfolio_max_profiles;
        if (profile_count > max_profiles) profile_count = max_profiles;

        if (profile_count <= 1) {
            goto cleanup;
        }

        candidate = (int*)malloc((size_t)foundPI * sizeof(int));
        if (row_multipliers) {
            candidate_multipliers = (double *)malloc(
                (size_t)ON_minterms * sizeof(double)
            );
        }
        if (!candidate || (row_multipliers && !candidate_multipliers)) {
            goto cleanup;
        }

        for (int i = 1; i < profile_count; ++i) {
            profiles[i].polish_chart = NULL;
            if (
                deadline_nanoseconds > 0 &&
                lagr_monotonic_nanoseconds() >= deadline_nanoseconds
            ) {
                merged_stats.stop_reason = LAGR_STOP_DEADLINE;
                break;
            }
            int candidate_solmin = -1;

            solve_scp_lagrangian_config(
                ON_minterms,
                foundPI,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                weights,
                &profiles[i],
                solution,
                best_solmin,
                candidate,
                &candidate_solmin,
                NULL,
                candidate_multipliers,
                polish_incumbent,
                polish_incumbent_cardinality,
                initial_multipliers,
                deadline_nanoseconds
            );

            const LagrangianStats *candidate_stats = lagrangian_last_stats();
            total_iterations += candidate_stats->iterations;
            merged_stats.reduced_cost_nanoseconds +=
                candidate_stats->reduced_cost_nanoseconds;
            merged_stats.heuristic_nanoseconds +=
                candidate_stats->heuristic_nanoseconds;
            merged_stats.heuristic_row_min_nanoseconds +=
                candidate_stats->heuristic_row_min_nanoseconds;
            merged_stats.heuristic_greedy_nanoseconds +=
                candidate_stats->heuristic_greedy_nanoseconds;
            merged_stats.heuristic_repair_nanoseconds +=
                candidate_stats->heuristic_repair_nanoseconds;
            merged_stats.bundle_nanoseconds +=
                candidate_stats->bundle_nanoseconds;
            merged_stats.subgradient_nanoseconds +=
                candidate_stats->subgradient_nanoseconds;
            lagr_merge_teams(&merged_stats.reduced_cost_teams, &candidate_stats->reduced_cost_teams);
            lagr_merge_teams(&merged_stats.slack_teams, &candidate_stats->slack_teams);
            merged_stats.nonzero_subgradient_rows +=
                candidate_stats->nonzero_subgradient_rows;
            merged_stats.reduced_cost_calls +=
                candidate_stats->reduced_cost_calls;
            merged_stats.heuristic_calls +=
                candidate_stats->heuristic_calls;
            merged_stats.primal_ub_recurring_calls +=
                candidate_stats->primal_ub_recurring_calls;
            merged_stats.primal_ub_improving_calls +=
                candidate_stats->primal_ub_improving_calls;
            merged_stats.primal_ub_improving_calls_below_supplied +=
                candidate_stats->
                    primal_ub_improving_calls_below_supplied;
            if (
                candidate_stats->primal_ub_first_improvement_iteration > 0 &&
                (
                    merged_stats.primal_ub_first_improvement_iteration == 0 ||
                    candidate_stats->primal_ub_first_improvement_iteration <
                        merged_stats.primal_ub_first_improvement_iteration
                )
            ) {
                merged_stats.primal_ub_first_improvement_iteration =
                    candidate_stats->
                        primal_ub_first_improvement_iteration;
            }
            if (
                candidate_stats->primal_ub_last_improvement_iteration >
                    merged_stats.primal_ub_last_improvement_iteration
            ) {
                merged_stats.primal_ub_last_improvement_iteration =
                    candidate_stats->
                        primal_ub_last_improvement_iteration;
            }
            if (
                candidate_stats->primal_ub_longest_improvement_gap >
                    merged_stats.primal_ub_longest_improvement_gap
            ) {
                merged_stats.primal_ub_longest_improvement_gap =
                    candidate_stats->primal_ub_longest_improvement_gap;
            }
            if (
                candidate_stats->primal_ub_loop_final >= 0 &&
                (
                    merged_stats.primal_ub_loop_final < 0 ||
                    candidate_stats->primal_ub_loop_final <
                        merged_stats.primal_ub_loop_final
                )
            ) {
                merged_stats.primal_ub_loop_final =
                    candidate_stats->primal_ub_loop_final;
            }
            merged_stats.bundle_calls += candidate_stats->bundle_calls;
            merged_stats.subgradient_calls +=
                candidate_stats->subgradient_calls;
            merged_stats.subgradient_samples +=
                candidate_stats->subgradient_samples;
            if (
                candidate_stats->subgradient_samples > 0 &&
                candidate_stats->nonzero_subgradient_minimum <
                    merged_stats.nonzero_subgradient_minimum
            ) {
                merged_stats.nonzero_subgradient_minimum =
                    candidate_stats->nonzero_subgradient_minimum;
            }
            if (
                candidate_stats->nonzero_subgradient_maximum >
                merged_stats.nonzero_subgradient_maximum
            ) {
                merged_stats.nonzero_subgradient_maximum =
                    candidate_stats->nonzero_subgradient_maximum;
            }
            lagr_merge_polish_stats(
                &merged_stats,
                candidate_stats
            );

            if (candidate_stats->best_lb > merged_stats.best_lb) {
                merged_stats.best_lb = candidate_stats->best_lb;
                merged_stats.last_zlb = candidate_stats->last_zlb;
                merged_stats.step_coef = candidate_stats->step_coef;
                merged_stats.stop_reason = candidate_stats->stop_reason;
            }
            if (candidate_stats->best_zlb > merged_stats.best_zlb + EPS) {
                merged_stats.best_zlb = candidate_stats->best_zlb;
                merged_stats.last_zlb = candidate_stats->last_zlb;
                merged_stats.requested_iteration_limit =
                    candidate_stats->requested_iteration_limit;
                merged_stats.iteration_limit =
                    candidate_stats->iteration_limit;
                merged_stats.iteration_limit_reason =
                    candidate_stats->iteration_limit_reason;
                merged_stats.continuous_improvements =
                    candidate_stats->continuous_improvements;
                merged_stats.last_continuous_improvement_iteration =
                    candidate_stats->
                        last_continuous_improvement_iteration;
                merged_stats.step_contractions =
                    candidate_stats->step_contractions;
                merged_stats.restarts = candidate_stats->restarts;
                merged_stats.multiplier_schedule =
                    candidate_stats->multiplier_schedule;
                merged_stats.multiplier_convergence =
                    candidate_stats->multiplier_convergence;
                merged_stats.reset_on_continuous_improvement =
                    candidate_stats->
                        reset_on_continuous_improvement;
                merged_stats.stop_at_step_floor =
                    candidate_stats->stop_at_step_floor;
                merged_stats.requested_initial_step_coef =
                    candidate_stats->requested_initial_step_coef;
                merged_stats.requested_step_min =
                    candidate_stats->requested_step_min;
                merged_stats.requested_step_contract =
                    candidate_stats->requested_step_contract;
                merged_stats.requested_stabilization_beta =
                    candidate_stats->
                        requested_stabilization_beta;
                merged_stats.requested_deflection_alpha =
                    candidate_stats->requested_deflection_alpha;
                merged_stats.initial_step_coef =
                    candidate_stats->initial_step_coef;
                merged_stats.step_coef = candidate_stats->step_coef;
                merged_stats.step_min = candidate_stats->step_min;
                merged_stats.step_contract =
                    candidate_stats->step_contract;
                merged_stats.stabilization_beta =
                    candidate_stats->stabilization_beta;
                merged_stats.deflection_alpha =
                    candidate_stats->deflection_alpha;
                if (row_multipliers) {
                    memcpy(
                        row_multipliers,
                        candidate_multipliers,
                        (size_t)ON_minterms * sizeof(double)
                    );
                }
            }
            if (candidate_solmin >= 0 && candidate_solmin < best_solmin) {
                best_solmin = candidate_solmin;
                *solmin = candidate_solmin;
                memcpy(solution, candidate, (size_t)candidate_solmin * sizeof(int));
            }

            if (merged_stats.best_lb != INT_MIN && best_solmin <= merged_stats.best_lb) {
                break;
            }
        }

        LagrangianStopReason final_reason = merged_stats.stop_reason;
        if (merged_stats.best_lb != INT_MIN && best_solmin <= merged_stats.best_lb) {
            final_reason = LAGR_STOP_OPTIMAL;
        }

        lagr_stats_finish(
            best_solmin,
            merged_stats.best_lb == INT_MIN ? -DBL_MAX : (double)merged_stats.best_lb,
            merged_stats.best_zlb,
            merged_stats.last_zlb,
            merged_stats.step_coef,
            total_iterations,
            final_reason
        );
        lagr_last_stats.requested_iteration_limit =
            merged_stats.requested_iteration_limit;
        lagr_last_stats.iteration_limit =
            merged_stats.iteration_limit;
        lagr_last_stats.iteration_limit_reason =
            merged_stats.iteration_limit_reason;
        lagr_last_stats.continuous_improvements =
            merged_stats.continuous_improvements;
        lagr_last_stats.last_continuous_improvement_iteration =
            merged_stats.last_continuous_improvement_iteration;
        lagr_last_stats.step_contractions =
            merged_stats.step_contractions;
        lagr_last_stats.restarts = merged_stats.restarts;
        lagr_last_stats.multiplier_schedule =
            merged_stats.multiplier_schedule;
        lagr_last_stats.multiplier_convergence =
            merged_stats.multiplier_convergence;
        lagr_last_stats.reset_on_continuous_improvement =
            merged_stats.reset_on_continuous_improvement;
        lagr_last_stats.stop_at_step_floor =
            merged_stats.stop_at_step_floor;
        lagr_last_stats.requested_initial_step_coef =
            merged_stats.requested_initial_step_coef;
        lagr_last_stats.requested_step_min =
            merged_stats.requested_step_min;
        lagr_last_stats.requested_step_contract =
            merged_stats.requested_step_contract;
        lagr_last_stats.requested_stabilization_beta =
            merged_stats.requested_stabilization_beta;
        lagr_last_stats.requested_deflection_alpha =
            merged_stats.requested_deflection_alpha;
        lagr_last_stats.initial_step_coef =
            merged_stats.initial_step_coef;
        lagr_last_stats.step_min = merged_stats.step_min;
        lagr_last_stats.step_contract =
            merged_stats.step_contract;
        lagr_last_stats.stabilization_beta =
            merged_stats.stabilization_beta;
        lagr_last_stats.deflection_alpha =
            merged_stats.deflection_alpha;
        lagr_last_stats.polish_initial_ub =
            merged_stats.polish_initial_ub;
        lagr_last_stats.polish_final_ub =
            merged_stats.polish_final_ub;
        lagr_last_stats.polish_initial_target =
            merged_stats.polish_initial_target;
        lagr_last_stats.polish_initial_core_count =
            merged_stats.polish_initial_core_count;
        lagr_last_stats.polish_search_target =
            merged_stats.polish_search_target;
        lagr_last_stats.polish_search_core_count =
            merged_stats.polish_search_core_count;
        lagr_last_stats.polish_forced_columns =
            merged_stats.polish_forced_columns;
        lagr_last_stats.polish_skipped_targets =
            merged_stats.polish_skipped_targets;
        lagr_last_stats.polish_searches =
            merged_stats.polish_searches;
        lagr_last_stats.polish_nodes =
            merged_stats.polish_nodes;
        lagr_last_stats.polish_extra_nodes = merged_stats.polish_extra_nodes;
        lagr_last_stats.polish_workers = merged_stats.polish_workers;
        lagr_last_stats.polish_active_workers = merged_stats.polish_active_workers;
        lagr_last_stats.polish_warm_nodes = merged_stats.polish_warm_nodes;
        lagr_last_stats.polish_warm_work = merged_stats.polish_warm_work;
        lagr_last_stats.polish_warm_peak_bytes = merged_stats.polish_warm_peak_bytes;
        lagr_last_stats.polish_warm_workers = merged_stats.polish_warm_workers;
        lagr_last_stats.polish_warm_fallbacks = merged_stats.polish_warm_fallbacks;
        lagr_last_stats.polish_aborted_searches =
            merged_stats.polish_aborted_searches;
        lagr_last_stats.polish_exhausted_searches =
            merged_stats.polish_exhausted_searches;
        lagr_last_stats.polish_improvements =
            merged_stats.polish_improvements;
        lagr_last_stats.polish_proved_optimal =
            merged_stats.polish_proved_optimal;
        lagr_last_stats.polish_seconds =
            merged_stats.polish_seconds;
        lagr_last_stats.polish_stop_reason =
            merged_stats.polish_stop_reason;
        lagr_last_stats.reduced_cost_nanoseconds =
            merged_stats.reduced_cost_nanoseconds;
        lagr_last_stats.heuristic_nanoseconds =
            merged_stats.heuristic_nanoseconds;
        lagr_last_stats.heuristic_row_min_nanoseconds =
            merged_stats.heuristic_row_min_nanoseconds;
        lagr_last_stats.heuristic_greedy_nanoseconds =
            merged_stats.heuristic_greedy_nanoseconds;
        lagr_last_stats.heuristic_repair_nanoseconds =
            merged_stats.heuristic_repair_nanoseconds;
        lagr_last_stats.bundle_nanoseconds =
            merged_stats.bundle_nanoseconds;
        lagr_last_stats.subgradient_nanoseconds =
            merged_stats.subgradient_nanoseconds;
        lagr_last_stats.reduced_cost_teams = merged_stats.reduced_cost_teams;
        lagr_last_stats.slack_teams = merged_stats.slack_teams;
        lagr_last_stats.nonzero_subgradient_rows =
            merged_stats.nonzero_subgradient_rows;
        lagr_last_stats.reduced_cost_calls =
            merged_stats.reduced_cost_calls;
        lagr_last_stats.heuristic_calls = merged_stats.heuristic_calls;
        lagr_last_stats.primal_ub_cadence =
            merged_stats.primal_ub_cadence;
        lagr_last_stats.primal_ub_supplied =
            merged_stats.primal_ub_supplied;
        lagr_last_stats.primal_ub_loop_start =
            merged_stats.primal_ub_loop_start;
        lagr_last_stats.primal_ub_loop_final =
            merged_stats.primal_ub_loop_final;
        lagr_last_stats.primal_ub_recurring_calls =
            merged_stats.primal_ub_recurring_calls;
        lagr_last_stats.primal_ub_improving_calls =
            merged_stats.primal_ub_improving_calls;
        lagr_last_stats.primal_ub_improving_calls_below_supplied =
            merged_stats.primal_ub_improving_calls_below_supplied;
        lagr_last_stats.primal_ub_first_improvement_iteration =
            merged_stats.primal_ub_first_improvement_iteration;
        lagr_last_stats.primal_ub_last_improvement_iteration =
            merged_stats.primal_ub_last_improvement_iteration;
        lagr_last_stats.primal_ub_longest_improvement_gap =
            merged_stats.primal_ub_longest_improvement_gap;
        lagr_last_stats.bundle_calls = merged_stats.bundle_calls;
        lagr_last_stats.subgradient_calls =
            merged_stats.subgradient_calls;
        lagr_last_stats.subgradient_samples =
            merged_stats.subgradient_samples;
        lagr_last_stats.nonzero_subgradient_minimum =
            merged_stats.nonzero_subgradient_minimum;
        lagr_last_stats.nonzero_subgradient_maximum =
            merged_stats.nonzero_subgradient_maximum;
    }

cleanup:
    lagr_last_stats.warm_start_requested = warm_start_requested;
    lagr_last_stats.warm_start_accepted = warm_start_accepted;
    lagr_last_stats.effort_level =
        lagr_normalize_effort_level(effort_level);
    lagr_last_stats.certification_requested =
        certification_requested ? 1 : 0;
    lagr_last_stats.portfolio_limit = baseline.portfolio_enabled
        ? baseline.portfolio_max_profiles
        : 1;
    lagr_last_stats.polish_enabled = baseline.polish_enabled;
    lagr_last_stats.polish_node_limit = baseline.polish_node_limit;
    lagr_last_stats.polish_core_cap = baseline.polish_core_cap;
    /* assigned here, not at the presolve: solve_scp_lagrangian_config() calls
       lagr_stats_begin(), which resets the block */
    lagr_last_stats.presolve_cols_removed = presolve_removed;
    free(candidate);
    free(candidate_multipliers);
    free(initial_multipliers_copy);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        foundPI,
        colsCovering,
        colsCoveringCount,
        ON_minterms
    );
}

void solve_scp_lagrangian_with_multipliers_and_polish_incumbent(
    const PIChartView *chart,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested,
    double *row_multipliers,
    int multiplier_count,
    const int *polish_incumbent,
    int polish_incumbent_cardinality,
    const double *initial_multipliers
) {
    solve_scp_lagrangian_with_deadline(
        chart,
        weights,
        solution,
        solmin,
        effort_level,
        initial_solution,
        initial_solmin,
        certification_requested,
        row_multipliers,
        multiplier_count,
        polish_incumbent,
        polish_incumbent_cardinality,
        initial_multipliers,
        0,
        true,
        1
    );
}

void solve_scp_lagrangian_with_multipliers(
    const PIChartView *chart,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested,
    double *row_multipliers,
    int multiplier_count,
    const double *initial_multipliers
) {
    solve_scp_lagrangian_with_deadline(
        chart,
        weights,
        solution,
        solmin,
        effort_level,
        initial_solution,
        initial_solmin,
        certification_requested,
        row_multipliers,
        multiplier_count,
        NULL,
        0,
        initial_multipliers,
        0,
        false,
        1
    );
}

void solve_scp_lagrangian(
    const PIChartView *chart,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested
) {
    solve_scp_lagrangian_with_multipliers(
        chart,
        weights,
        solution,
        solmin,
        effort_level,
        initial_solution,
        initial_solmin,
        certification_requested,
        NULL,
        0,
        NULL
    );
}

void solve_scp_lagrangian_until(
    const PIChartView *chart,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested,
    uint64_t deadline_nanoseconds
) {
    solve_scp_lagrangian_with_deadline(
        chart,
        weights,
        solution,
        solmin,
        effort_level,
        initial_solution,
        initial_solmin,
        certification_requested,
        NULL,
        0,
        NULL,
        0,
        NULL,
        deadline_nanoseconds,
        false,
        ccubes_thread_count()
    );
}

void solve_scp_lagrangian_pool(
    const PIChartView *chart,
    const double weights[],
    int max_pool,
    int *out_pool_count,
    int **pool_solutions,
    int *solmin,
    int effort_level,
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested
) {
    if (solmin) *solmin = -1;
    const int foundPI = chart ? chart->cols : 0;
    const int ON_minterms = chart ? chart->rows : 0;
    lagr_stats_begin(ON_minterms, foundPI, 1);
    if (
        !chart ||
        !chart->bits ||
        foundPI <= 0 ||
        ON_minterms <= 0 ||
        !out_pool_count ||
        !pool_solutions ||
        !solmin
    ) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }

    int **rowsCovered = NULL, *rowsCoveredCount = NULL;
    int **colsCovering = NULL, *colsCoveringCount = NULL;

    int rc_ad = build_adjacency(
        chart,
        ON_minterms,
        foundPI,
        &rowsCovered,
        &rowsCoveredCount,
        &colsCovering,
        &colsCoveringCount
    );

    if (rc_ad == -2 || rc_ad == -1) {
        /* infeasible or OOM */
        lagr_stats_finish(
            -1,
            -DBL_MAX,
            -DBL_MAX,
            -DBL_MAX,
            0.0,
            0,
            rc_ad == -2 ? LAGR_STOP_INFEASIBLE : LAGR_STOP_OOM
        );
        return;
    }

    int rows = ON_minterms, cols = foundPI;
    const LagrangianLoopThreads loop_threads = lagr_loop_threads(rows, cols, rowsCoveredCount);
    int warm_start_requested =
        initial_solution != NULL && initial_solmin > 0;
    int warm_start_accepted =
        warm_start_requested &&
        cover_is_feasible(
            chart,
            initial_solution,
            initial_solmin
        );
    if (!warm_start_accepted) {
        initial_solution = NULL;
        initial_solmin = 0;
    }

    LagrangianWorkspace workspace = {0};
    const bool workspace_ready = lagr_workspace_init(&workspace, rows, cols);
    double *t = (double*)calloc((size_t)rows, sizeof(double));
    double *t_best = (double*)calloc((size_t)rows, sizeof(double));
    double *rc = (double*)malloc((size_t)cols * sizeof(double));
    int *sol_tmp = (int*)malloc((size_t)cols * sizeof(int));
    int *sol_tmp2 = (int*)malloc((size_t)cols * sizeof(int));
    int *norm = (int*)malloc((size_t)cols * sizeof(int));
    double *col_costs = (double*)malloc((size_t)cols * sizeof(double));
    unsigned char *dual_x = (unsigned char*)malloc((size_t)cols);

    if (!workspace_ready || !t || !t_best || !rc || !sol_tmp || !sol_tmp2 || !norm || !col_costs || !dual_x) {
        free(t);
        free(t_best);
        free(rc);
        free(sol_tmp);
        free(sol_tmp2);
        free(norm);
        free(col_costs);
        free(dual_x);
        lagr_workspace_destroy(&workspace);
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            cols,
            colsCovering,
            colsCoveringCount,
            rows
        );
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
        return;
    }

    double max_w = 0.0;
    if (weights) {
        for (int j = 0; j < cols; ++j) {
            if (weights[j] > max_w) max_w = weights[j];
        }
    }
    double eps = 0.0;
    if (max_w > 0.0) {
        eps = 1e-6 / (max_w + 1.0);
    }
    for (int j = 0; j < cols; ++j) {
        col_costs[j] = 1.0 - eps * (weights ? weights[j] : 0.0);
    }

    int pool_count = 0;
    int pool_min_len = INT_MAX; /* track minimal len seen, pool only stores solutions with this len */

    LagrangianConfig cfg;
    lagr_config_for_effort(&cfg, effort_level);
    lagr_apply_certification_policy(
        &cfg,
        effort_level,
        certification_requested
    );
    lagr_last_stats.polish_enabled = cfg.polish_enabled;
    lagr_last_stats.polish_node_limit = cfg.polish_node_limit;
    lagr_last_stats.polish_core_cap = cfg.polish_core_cap;
    if (!cfg.polish_enabled) {
        lagr_last_stats.polish_stop_reason =
            LAGR_POLISH_DISABLED;
    }
    if (cfg.hybrid_bundle_portfolio) {
        cfg.bundle_enabled = 1;
    }
    int local_passes = cfg.local_passes;
    LagrangianBundleState bundle_state = {0};

    if (!(max_pool > 1 && out_pool_count && pool_solutions)) {
        max_pool = 1;
    }

    if (cfg.bundle_enabled) {
        int memory = cfg.bundle_memory > 0 ? cfg.bundle_memory : 1;
        bundle_state.rows = rows;
        bundle_state.memory = memory;
        bundle_state.prox_mu = cfg.bundle_prox_mu;
        bundle_state.min_mu = cfg.bundle_min_mu;
        bundle_state.max_mu = cfg.bundle_max_mu;
        bundle_state.null_expand = cfg.bundle_null_expand;
        bundle_state.serious_shrink = cfg.bundle_serious_shrink;
        bundle_state.serious_fraction = cfg.bundle_serious_fraction;
        bundle_state.serious_tol = cfg.bundle_serious_tol;
        bundle_state.momentum = cfg.bundle_momentum;
        bundle_state.center_value = -DBL_MAX;
        bundle_state.center_t = (double*)calloc((size_t)rows, sizeof(double));
        bundle_state.prev_center_t = (double*)calloc((size_t)rows, sizeof(double));
        bundle_state.cut_alpha = (double*)calloc((size_t)memory, sizeof(double));
        bundle_state.cut_slacks = (double*)calloc((size_t)memory * (size_t)rows, sizeof(double));
        if (!bundle_state.center_t || !bundle_state.prev_center_t || !bundle_state.cut_alpha || !bundle_state.cut_slacks) {
            free(bundle_state.center_t);
            free(bundle_state.prev_center_t);
            free(bundle_state.cut_alpha);
            free(bundle_state.cut_slacks);
            free(t);
            free(t_best);
            free(rc);
            free(sol_tmp);
            free(sol_tmp2);
            free(norm);
            free(col_costs);
            free(dual_x);
            lagr_workspace_destroy(&workspace);
            free_adjacency(
                rowsCovered,
                rowsCoveredCount,
                cols,
                colsCovering,
                colsCoveringCount,
                rows
            );
            lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
            return;
        }
    }

    lagr_initialize_multipliers(rows, colsCoveringCount, cfg.rarity_init, t);

    double bestZLB = -DBL_MAX;
    double lastZLB = -DBL_MAX;
    int iterations = 0;
    LagrangianStopReason stop_reason = LAGR_STOP_MAX_ITER;

    /* initial rc */
    double initialZLB = compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        col_costs,
        rc,
        dual_x,
        loop_threads.columns
    );
    bestZLB = initialZLB;
    lastZLB = initialZLB;
    memcpy(t_best, t, (size_t)rows * sizeof(double));

    int sol_size = -1, sol_size2 = -1;

    heuristic_row_min_rc(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        rc,
        weights,
        sol_tmp,
        &sol_size,
        &workspace
    );

    greedy_from_lagr_scores(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        rc,
        weights,
        sol_tmp2,
        &sol_size2,
        &workspace
    );

    if (local_passes > 0) {
        if (sol_size > 0) {
            local_search_drop1_repair(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp,
                &sol_size,
                local_passes
            );

            local_search_drop2_repair(
                rows,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp,
                &sol_size,
                local_passes,
                0
            );
        }

        if (sol_size2 > 0) {
            local_search_drop1_repair(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp2,
                &sol_size2,
                local_passes
            );

            local_search_drop2_repair(
                rows,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp2,
                &sol_size2,
                local_passes,
                0
            );
        }
    }

    if (sol_size == -1 && sol_size2 == -1 && !warm_start_accepted) {
        free(t);
        free(t_best);
        free(rc);
        free(sol_tmp);
        free(sol_tmp2);
        free(norm);
        free(col_costs);
        free(dual_x);
        lagr_workspace_destroy(&workspace);
        free(bundle_state.center_t);
        free(bundle_state.prev_center_t);
        free(bundle_state.cut_alpha);
        free(bundle_state.cut_slacks);
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            cols,
            colsCovering,
            colsCoveringCount,
            rows
        );
        lagr_stats_finish(-1, -DBL_MAX, bestZLB, lastZLB, 0.0, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }

    double w1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
    double w2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

    double bestUB = DBL_MAX;
    int bestTerms = INT_MAX;
    double bestWeight = -DBL_MAX;

    if (sol_size != -1) {
        bestUB = (double)sol_size;
        bestTerms = sol_size;
        bestWeight = w1;
        *solmin = sol_size;
    }
    if (
        sol_size2 != -1 &&
        (
            sol_size2 < bestTerms ||
            (sol_size2 == bestTerms && w2 > bestWeight + EPS)
        )
    ) {
        bestUB = (double)sol_size2;
        bestTerms = sol_size2;
        bestWeight = w2;
        *solmin = sol_size2;
    }
    if (warm_start_accepted) {
        double initial_weight =
            solution_total_weight(initial_solution, initial_solmin, weights);
        if (
            initial_solmin < bestTerms ||
            (
                initial_solmin == bestTerms &&
                initial_weight > bestWeight + EPS
            )
        ) {
            bestUB = (double)initial_solmin;
            bestTerms = initial_solmin;
            bestWeight = initial_weight;
            *solmin = initial_solmin;
        }
    }

    /* seed pool: add only minimal-len solutions */
    if (max_pool > 1) {
        if (sol_size != -1) {
            if (sol_size < pool_min_len) {
                pool_clear(pool_solutions, &pool_count);
                pool_min_len = sol_size;
            }

            if (sol_size == pool_min_len) {
                pool_normalize(sol_tmp, sol_size, norm);
                pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size);
            }
        }

        if (sol_size2 != -1) {
            if (sol_size2 < pool_min_len) {
                pool_clear(pool_solutions, &pool_count);
                pool_min_len = sol_size2;
            }

            if (sol_size2 == pool_min_len) {
                pool_normalize(sol_tmp2, sol_size2, norm);
                pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size2);
            }
        }

        if (warm_start_accepted) {
            if (initial_solmin < pool_min_len) {
                pool_clear(pool_solutions, &pool_count);
                pool_min_len = initial_solmin;
            }
            if (initial_solmin == pool_min_len) {
                pool_normalize(
                    initial_solution,
                    initial_solmin,
                    norm
                );
                pool_add_if_new(
                    pool_solutions,
                    &pool_count,
                    max_pool,
                    norm,
                    initial_solmin
                );
            }
        }
    }

    double bestLB = -DBL_MAX;

    /* parameters */
    const int max_iter = cfg.max_iter;
    const int heur_every = cfg.heur_every;
    lagr_last_stats.primal_ub_cadence = heur_every;
    double step_coef = cfg.step_coef;
    const double step_min = cfg.step_min;
    const int halve_period = cfg.halve_period;
    int stuck_iter = 0;
    int continuous_improvements = 0;
    int last_continuous_improvement_iteration = 0;
    int step_contractions = 0;
    lagr_last_stats.requested_iteration_limit = cfg.max_iter;
    lagr_last_stats.iteration_limit = max_iter;
    lagr_last_stats.requested_initial_step_coef =
        cfg.step_coef;
    lagr_last_stats.requested_step_min = cfg.step_min;
    lagr_last_stats.requested_step_contract =
        cfg.step_contract;
    lagr_last_stats.requested_stabilization_beta =
        cfg.stabilization_beta;
    lagr_last_stats.requested_deflection_alpha =
        cfg.deflection_alpha;
    lagr_last_stats.initial_step_coef = step_coef;
    lagr_last_stats.step_min = step_min;
    lagr_last_stats.step_contract = cfg.step_contract;
    lagr_last_stats.stabilization_beta =
        cfg.stabilization_beta;
    lagr_last_stats.deflection_alpha =
        cfg.deflection_alpha;
    lagr_last_stats.reset_on_continuous_improvement =
        cfg.reset_on_continuous_improvement;
    lagr_last_stats.stop_at_step_floor =
        cfg.stop_at_step_floor;

    for (int it = 0; it < max_iter; ++it) {
        iterations = it + 1;
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            col_costs,
            rc,
            dual_x,
            loop_threads.columns
        );
        lastZLB = ZLB;
        if (ZLB > bestZLB + EPS) {
            bestZLB = ZLB;
            memcpy(t_best, t, (size_t)rows * sizeof(double));
            continuous_improvements++;
            last_continuous_improvement_iteration = iterations;
            if (cfg.reset_on_continuous_improvement) {
                stuck_iter = 0;
            }
        }

        double LBint = ceil(ZLB - EPS);
        if (LBint > bestLB + EPS) {
            bestLB = LBint;
            if (!cfg.reset_on_continuous_improvement) {
                stuck_iter = 0;
            }
        }

        if (it % heur_every == 0) {
            heuristic_row_min_rc(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp,
                &sol_size,
                &workspace
            );

            greedy_from_lagr_scores(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp2,
                &sol_size2,
                &workspace
            );

            if (sol_size != -1 && sol_size <= pool_min_len && local_passes > 0) {
                local_search_drop1_repair(
                    rows,
                    cols,
                    rowsCovered,
                    rowsCoveredCount,
                    colsCovering,
                    colsCoveringCount,
                    rc,
                    weights,
                    sol_tmp,
                    &sol_size,
                    local_passes
                );
                if (cfg.drop2_enabled) {
                    local_search_drop2_repair(
                        rows,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        rc,
                        weights,
                        sol_tmp,
                        &sol_size,
                        local_passes,
                        0
                    );
                }
            }

            if (sol_size2 != -1 && sol_size2 <= pool_min_len && local_passes > 0) {
                local_search_drop1_repair(
                    rows,
                    cols,
                    rowsCovered,
                    rowsCoveredCount,
                    colsCovering,
                    colsCoveringCount,
                    rc,
                    weights,
                    sol_tmp2,
                    &sol_size2,
                    local_passes
                );
                if (cfg.drop2_enabled) {
                    local_search_drop2_repair(
                        rows,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        rc,
                        weights,
                        sol_tmp2,
                        &sol_size2,
                        local_passes,
                        0
                    );
                }
            }

            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

            /* candidate 1 */
            if (
                sol_size != -1 &&
                (
                    sol_size < bestTerms ||
                    (
                        sol_size == bestTerms && candW1 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size;
                bestTerms = sol_size;
                bestWeight = candW1;
                *solmin = sol_size;

                if (max_pool > 1) {
                    if (sol_size < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size;
                    }

                    if (sol_size == pool_min_len) {
                        pool_normalize(sol_tmp, sol_size, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size);
                    }
                }
            } else if (max_pool > 1) {
                if (sol_size != -1) {
                    if (sol_size < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size;
                    }

                    if (sol_size == pool_min_len) {
                        pool_normalize(sol_tmp, sol_size, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size);
                    }
                }

                if (
                    sol_size != -1 &&
                    sol_size == bestTerms &&
                    candW1 > bestWeight + EPS
                ) {
                    bestWeight = candW1;
                    *solmin = sol_size;
                }
            }

            /* candidate 2 */
            if (
                sol_size2 != -1 &&
                (
                    sol_size2 < bestTerms ||
                    (
                        sol_size2 == bestTerms && candW2 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size2;
                bestTerms = sol_size2;
                bestWeight = candW2;
                *solmin = sol_size2;

                if (max_pool > 1) {
                    if (sol_size2 < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size2;
                    }

                    if (sol_size2 == pool_min_len) {
                        pool_normalize(sol_tmp2, sol_size2, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size2);
                    }
                }
            } else if (max_pool > 1) {
                if (sol_size2 != -1) {
                    if (sol_size2 < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size2;
                    }

                    if (sol_size2 == pool_min_len) {
                        pool_normalize(sol_tmp2, sol_size2, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size2);
                    }
                }

                if (
                    sol_size2 != -1 &&
                    sol_size2 == bestTerms &&
                    candW2 > bestWeight + EPS
                ) {
                    bestWeight = candW2;
                    *solmin = sol_size2;
                }
            }
        }

        if (bestUB <= LBint + EPS) {
            stop_reason = LAGR_STOP_OPTIMAL;
            break;
        }

        int used_subgradient_update = 0;
        int updated = 0;

        int use_bundle_update = cfg.bundle_enabled &&
            (cfg.bundle_interval <= 1 || it % cfg.bundle_interval == 0);

        if (use_bundle_update) {
            updated = bundle_update(
                rows,
                colsCovering,
                colsCoveringCount,
                dual_x,
                bestUB,
                ZLB,
                t,
                &bundle_state,
                NULL,
                loop_threads.rows
            );
        }

        if (!use_bundle_update || !updated) {
            double step_before = step_coef;
            updated = subgradient_update(
                rows,
                colsCovering,
                colsCoveringCount,
                dual_x,
                bestUB,
                ZLB,
                t,
                &step_coef,
                step_min,
                &stuck_iter,
                halve_period,
                NULL,
                0.0,
                cfg.step_contract,
                cfg.stabilization_beta,
                NULL,
                &workspace,
                loop_threads.rows
            );
            used_subgradient_update = 1;
            if (step_coef < step_before - EPS) {
                step_contractions++;
            }
        }

        if (
            !updated ||
            (
                used_subgradient_update &&
                cfg.stop_at_step_floor &&
                step_coef <= step_min + EPS
            )
        ) {
            stop_reason = updated ? LAGR_STOP_STEP_MIN : LAGR_STOP_NO_UPDATE;
            /* final refresh */
            heuristic_row_min_rc(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp,
                &sol_size,
                &workspace
            );

            greedy_from_lagr_scores(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                rc,
                weights,
                sol_tmp2,
                &sol_size2,
                &workspace
            );

            if (sol_size != -1 && sol_size <= pool_min_len && local_passes > 0) {
                local_search_drop1_repair(
                    rows,
                    cols,
                    rowsCovered,
                    rowsCoveredCount,
                    colsCovering,
                    colsCoveringCount,
                    rc,
                    weights,
                    sol_tmp,
                    &sol_size,
                    local_passes
                );
                if (cfg.drop2_enabled) {
                    local_search_drop2_repair(
                        rows,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        rc,
                        weights,
                        sol_tmp,
                        &sol_size,
                        local_passes,
                        0
                    );
                }
            }

            if (sol_size2 != -1 && sol_size2 <= pool_min_len && local_passes > 0) {
                local_search_drop1_repair(
                    rows,
                    cols,
                    rowsCovered,
                    rowsCoveredCount,
                    colsCovering,
                    colsCoveringCount,
                    rc,
                    weights,
                    sol_tmp2,
                    &sol_size2,
                    local_passes
                );
                if (cfg.drop2_enabled) {
                    local_search_drop2_repair(
                        rows,
                        rowsCovered,
                        rowsCoveredCount,
                        colsCovering,
                        colsCoveringCount,
                        rc,
                        weights,
                        sol_tmp2,
                        &sol_size2,
                        local_passes,
                        0
                    );
                }
            }

            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

            if (
                sol_size != -1 &&
                (
                    sol_size < bestTerms ||
                    (
                        sol_size == bestTerms && candW1 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size;
                bestTerms = sol_size;
                bestWeight = candW1;
                *solmin = sol_size;

                if (max_pool > 1) {
                    if (sol_size < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size;
                    }

                    if (sol_size == pool_min_len) {
                        pool_normalize(sol_tmp, sol_size, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size);
                    }
                }

            } else if (max_pool > 1) {
                if (sol_size != -1) {
                    if (sol_size < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size;
                    }

                    if (sol_size == pool_min_len) {
                        pool_normalize(sol_tmp, sol_size, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size);
                    }
                }

                if (
                    sol_size != -1 &&
                    sol_size == bestTerms &&
                    candW1 > bestWeight + EPS
                ) {
                    bestWeight = candW1;
                    *solmin = sol_size;
                }
            }

            if (
                sol_size2 != -1 &&
                (
                    sol_size2 < bestTerms ||
                    (
                        sol_size2 == bestTerms && candW2 > bestWeight + EPS
                    )
                )
            ) {
                bestUB = (double)sol_size2;
                bestTerms = sol_size2;
                bestWeight = candW2;
                *solmin = sol_size2;

                if (max_pool > 1) {
                    if (sol_size2 < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size2;
                    }

                    if (sol_size2 == pool_min_len) {
                        pool_normalize(sol_tmp2, sol_size2, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size2);
                    }
                }
            } else if (max_pool > 1) {
                if (sol_size2 != -1) {
                    if (sol_size2 < pool_min_len) {
                        pool_clear(pool_solutions, &pool_count);
                        pool_min_len = sol_size2;
                    }

                    if (sol_size2 == pool_min_len) {
                        pool_normalize(sol_tmp2, sol_size2, norm);
                        pool_add_if_new(pool_solutions, &pool_count, max_pool, norm, sol_size2);
                    }
                }

                if (
                    sol_size2 != -1 &&
                    sol_size2 == bestTerms &&
                    candW2 > bestWeight + EPS
                ) {
                    bestWeight = candW2;
                    *solmin = sol_size2;
                }
            }
            break;
        }
    }

    if (cfg.polish_enabled && *solmin > 0) {
        /* reduced costs at the best multipliers found */
        double zlb_best = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t_best,
            col_costs,
            rc,
            dual_x,
            loop_threads.columns
        );

        int proved = (stop_reason == LAGR_STOP_OPTIMAL);
        int solmin_before = *solmin;

        lagr_bounded_finish(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            colsCovering,
            colsCoveringCount,
            weights,
            rc,
            zlb_best,
            cfg.polish_node_limit,
            cfg.polish_core_cap,
            sol_tmp,
            solmin,
            &bestLB,
            &proved,
            max_pool,
            pool_solutions,
            &pool_count,
            &pool_min_len,
            norm,
            0, NULL, 0, 0,
            cfg.adaptive_finish ? ccubes_thread_count() : 1,
            cfg.adaptive_finish
        );

        if (*solmin < solmin_before) {
            bestTerms = *solmin;
            bestUB = (double)*solmin;

            if (max_pool <= 1) {
                /* no pool to carry the improved cover: keep it in the stats path */
                pool_min_len = *solmin;
            }
        }

        if (proved) {
            stop_reason = LAGR_STOP_OPTIMAL;
        }
    }

    lagr_stats_finish(
        (*solmin > 0) ? *solmin : bestTerms,
        bestLB,
        bestZLB,
        lastZLB,
        step_coef,
        iterations,
        stop_reason
    );
    lagr_last_stats.warm_start_requested = warm_start_requested;
    lagr_last_stats.warm_start_accepted = warm_start_accepted;
    lagr_last_stats.effort_level =
        lagr_normalize_effort_level(effort_level);
    lagr_last_stats.certification_requested =
        certification_requested ? 1 : 0;
    lagr_last_stats.continuous_improvements =
        continuous_improvements;
    lagr_last_stats.last_continuous_improvement_iteration =
        last_continuous_improvement_iteration;
    lagr_last_stats.step_contractions = step_contractions;
    lagr_last_stats.requested_iteration_limit = cfg.max_iter;
    lagr_last_stats.iteration_limit = cfg.max_iter;
    lagr_last_stats.portfolio_limit = 1;
    lagr_last_stats.polish_enabled = cfg.polish_enabled;
    lagr_last_stats.polish_node_limit = cfg.polish_node_limit;
    lagr_last_stats.polish_core_cap = cfg.polish_core_cap;

    /* output */
    if (max_pool > 1 && out_pool_count) {
        /*
         * Candidate zero is the preferred weighted incumbent used as the
         * mandatory seed by cross-output marginal-value retention.
         */
        pool_order_by_weight(
            pool_solutions,
            pool_count,
            pool_min_len,
            weights
        );
        *out_pool_count = pool_count;
    }

    /* cleanup */
    free(t);
    free(t_best);
    free(rc);
    free(sol_tmp);
    free(sol_tmp2);
    free(norm);
    free(col_costs);
    free(dual_x);
    lagr_workspace_destroy(&workspace);
    free(bundle_state.center_t);
    free(bundle_state.prev_center_t);
    free(bundle_state.cut_alpha);
    free(bundle_state.cut_slacks);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        cols,
        colsCovering,
        colsCoveringCount,
        rows
    );
}
