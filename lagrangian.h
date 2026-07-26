#include <stdlib.h>
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
    LAGR_STOP_INFEASIBLE,
    LAGR_STOP_OOM,
    LAGR_STOP_NO_CANDIDATE
} LagrangianStopReason;

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
    int iteration_limit;
    int portfolio_limit;
    long polish_node_limit;
    double best_zlb;
    double last_zlb;
    double step_coef;
    LagrangianStopReason stop_reason;
} LagrangianStats;

void lagrangian_reset_stats(void);

const LagrangianStats *lagrangian_last_stats(void);

bool lagrangian_last_run_proved_optimal(void);

const char *lagrangian_stop_reason_name(LagrangianStopReason reason);

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
