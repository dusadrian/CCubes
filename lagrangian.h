#ifndef LAGRANGIAN_H
#define LAGRANGIAN_H

#include <stdlib.h>
#include <stdint.h>
#include "pichart_view.h"
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

typedef enum {
    LAGR_STOP_NOT_RUN = 0,
    LAGR_STOP_OPTIMAL,
    LAGR_STOP_MAX_ITER,
    LAGR_STOP_NO_UPDATE,
    LAGR_STOP_STEP_MIN,
    LAGR_STOP_WARM_STALL,
    LAGR_STOP_DEADLINE,
    LAGR_STOP_INFEASIBLE,
    LAGR_STOP_OOM,
    LAGR_STOP_NO_CANDIDATE
} LagrangianStopReason;

typedef enum {
    LAGR_MULTIPLIER_SCHEDULE_NONE = 0,
    LAGR_MULTIPLIER_SCHEDULE_LEGACY,
    LAGR_MULTIPLIER_SCHEDULE_SCALED
} LagrangianMultiplierSchedule;

typedef enum {
    LAGR_MULTIPLIER_CONVERGENCE_NONE = 0,
    LAGR_MULTIPLIER_CONVERGENCE_LEGACY,
    LAGR_MULTIPLIER_CONVERGENCE_ADAPTIVE
} LagrangianMultiplierConvergence;

typedef enum {
    LAGR_ITERATION_LIMIT_CONFIGURED = 0,
    LAGR_ITERATION_LIMIT_LEGACY_FLOOR,
    LAGR_ITERATION_LIMIT_SCALED_BY_ROWS,
    LAGR_ITERATION_LIMIT_WARM_SCALED
} LagrangianIterationLimitReason;

typedef enum {
    LAGR_POLISH_NOT_RUN = 0,
    LAGR_POLISH_DISABLED,
    LAGR_POLISH_NO_TARGET,
    LAGR_POLISH_CORE_CAP,
    LAGR_POLISH_FORCED_INFEASIBLE,
    LAGR_POLISH_ROOT_BOUND,
    LAGR_POLISH_ABORTED,
    LAGR_POLISH_EXHAUSTED,
    LAGR_POLISH_IMPROVED,
    LAGR_POLISH_DEADLINE,
    LAGR_POLISH_OOM
} LagrangianPolishStopReason;

typedef struct {
    int rows;
    int cols;
    int iterations;
    int best_ub;
    int best_lb;
    int gap;
    int pool_mode;
    int warm_start_requested;
    int warm_start_accepted;
    int effort_level;
    int certification_requested;
    int presolve_cols_removed;
    int requested_iteration_limit;
    int iteration_limit;
    int iteration_limit_reason;
    int warm_start;
    int continuous_improvements;
    int last_continuous_improvement_iteration;
    int step_contractions;
    int restarts;
    int multiplier_schedule;
    int multiplier_convergence;
    int reset_on_continuous_improvement;
    int stop_at_step_floor;
    int portfolio_limit;
    int polish_enabled;
    long polish_node_limit;
    int polish_core_cap;
    int polish_initial_ub;
    int polish_final_ub;
    int polish_initial_target;
    int polish_initial_core_count;
    int polish_search_target;
    int polish_search_core_count;
    int polish_forced_columns;
    int polish_skipped_targets;
    int polish_searches;
    long polish_nodes;
    int polish_aborted_searches;
    int polish_exhausted_searches;
    int polish_improvements;
    int polish_proved_optimal;
    double polish_seconds;
    LagrangianPolishStopReason polish_stop_reason;
    uint64_t reduced_cost_nanoseconds;
    uint64_t heuristic_nanoseconds;
    uint64_t bundle_nanoseconds;
    uint64_t subgradient_nanoseconds;
    uint64_t nonzero_subgradient_rows;
    int reduced_cost_calls;
    int heuristic_calls;
    int primal_ub_cadence;
    int primal_ub_supplied;
    int primal_ub_loop_start;
    int primal_ub_loop_final;
    int primal_ub_recurring_calls;
    int primal_ub_improving_calls;
    int primal_ub_improving_calls_below_supplied;
    int primal_ub_first_improvement_iteration;
    int primal_ub_last_improvement_iteration;
    int primal_ub_longest_improvement_gap;
    int bundle_calls;
    int subgradient_calls;
    int subgradient_samples;
    int nonzero_subgradient_minimum;
    int nonzero_subgradient_maximum;
    double best_zlb;
    double last_zlb;
    double requested_initial_step_coef;
    double requested_step_min;
    double requested_step_contract;
    double requested_stabilization_beta;
    double requested_deflection_alpha;
    double initial_step_coef;
    double step_coef;
    double step_min;
    double step_contract;
    double stabilization_beta;
    double deflection_alpha;
    LagrangianStopReason stop_reason;
} LagrangianStats;

typedef struct {
    int maximum_iterations;
    int stagnation_window;
    double initial_step_coefficient;
    double minimum_step_coefficient;
    double contraction;
    double stabilization_beta;
    double deflection_alpha;
    bool stop_at_step_floor;
    bool reset_on_continuous_improvement;
} LagrangianCalibrationOptions;

typedef struct {
    int iterations;
    int continuous_improvements;
    int last_continuous_improvement_iteration;
    int step_contractions;
    double initial_zlb;
    double best_zlb;
    double last_zlb;
    double final_step_coefficient;
    LagrangianStopReason stop_reason;
} LagrangianCalibrationStats;

void lagrangian_reset_stats(void);

const LagrangianStats *lagrangian_last_stats(void);

bool lagrangian_last_run_proved_optimal(void);

const char *lagrangian_stop_reason_name(LagrangianStopReason reason);

const char *lagrangian_multiplier_schedule_name(
    int schedule
);

const char *lagrangian_multiplier_convergence_name(
    int convergence
);

const char *lagrangian_iteration_limit_reason_name(
    int reason
);

const char *lagrangian_polish_stop_reason_name(
    LagrangianPolishStopReason reason
);

void solve_scp_lagrangian(
    const PIChartView *chart, // bit-packed column-major coverage
    const double weights[], // can be NULL; higher is better (tie-break inside heuristics)
    int *solution,          // out column indices (0-based)
    int *solmin,            // out size (number of columns); -1 if infeasible
    int effort_level,       // 0 fastest, 1 stronger bound, 2 strong on demand
    const int *initial_solution,
    int initial_solmin,
    bool certification_requested
);

/*
 * Deadline-aware variant used by optional subsolves. The deadline is an
 * absolute CLOCK_MONOTONIC timestamp in nanoseconds, or zero for no limit.
 * A feasible incumbent found before the deadline is returned normally.
 */
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
);

/*
 * Solve the same unit-cost set-covering master and also return the row
 * multipliers that attained the best Lagrangian lower bound.
 *
 * `row_multipliers` is optional. When supplied, `multiplier_count` must be at
 * least chart->rows. These are the CFT-style prices to reuse when the chart is
 * only a restricted master and additional implicit columns can be generated.
 *
 * `initial_multipliers` is an optional warm start: when non-NULL and
 * `row_multipliers` is supplied, the subgradient trajectory is seeded from
 * these row prices instead of the dual-ascent initialization, and the scaled
 * large-chart schedule uses a reduced iteration budget with a stagnation
 * stop. This changes the multiplier trajectory and therefore the priced
 * columns; it is intended for re-solving a restricted master that has only
 * grown since the previous call, and is enabled at runtime through
 * `CCUBES_LAGRANGIAN_WARM_START`. It may alias `row_multipliers`; the input
 * prices are preserved before that output buffer is cleared or overwritten.
 */
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
);

/*
 * Variant used by a restricted master that has a retained incumbent which
 * should bound only the final enumerative finish. Unlike `initial_solution`,
 * `polish_incumbent` does not steer the Lagrangian/subgradient trajectory.
 */
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
);

/*
 * Optimize only the unit-cost Lagrangian relaxation on a frozen chart.
 * No primal heuristic, pricing, repair, presolve, or bounded integer search is
 * run. `incumbent_upper_bound` must be the cardinality of a feasible cover.
 * This entry point exists for convergence calibration and deterministic
 * regression tests; production policy remains in solve_scp_lagrangian().
 */
void lagrangian_calibration_options_default(
    LagrangianCalibrationOptions *options
);

/*
 * Construct the deterministic dual-feasible row-price vector used to start
 * restricted-master pricing. Every retained unit-cost column has total row
 * price at most one. This does not solve the restricted LP, but provides a
 * sound and usually much stronger starting vector for full-universe dual
 * calibration than an unconstrained Lagrangian iterate.
 */
bool lagrangian_initialize_pricing_multipliers(
    const PIChartView *chart,
    double *row_multipliers,
    int multiplier_count
);

bool lagrangian_calibrate_multipliers(
    const PIChartView *chart,
    int incumbent_upper_bound,
    const LagrangianCalibrationOptions *options,
    double *row_multipliers,
    int multiplier_count,
    LagrangianCalibrationStats *stats
);

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
);

#endif
