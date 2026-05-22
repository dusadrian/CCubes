/*
    Copyright (c) 2016–2025, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "lagrangian.h"
#include "ccubes_threads.h"

#define EPS 1e-12

static LagrangianStats lagr_last_stats = {
    .rows = 0,
    .cols = 0,
    .iterations = 0,
    .best_ub = -1,
    .best_lb = INT_MIN,
    .gap = -1,
    .pool_mode = 0,
    .best_zlb = -DBL_MAX,
    .last_zlb = -DBL_MAX,
    .step_coef = 0.0,
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
    lagr_last_stats.best_zlb = -DBL_MAX;
    lagr_last_stats.last_zlb = -DBL_MAX;
    lagr_last_stats.step_coef = 0.0;
    lagr_last_stats.stop_reason = LAGR_STOP_NOT_RUN;
}

const LagrangianStats *lagrangian_last_stats(void) {
    return &lagr_last_stats;
}

const char *lagrangian_stop_reason_name(LagrangianStopReason reason) {
    switch (reason) {
        case LAGR_STOP_OPTIMAL: return "optimal";
        case LAGR_STOP_MAX_ITER: return "max_iter";
        case LAGR_STOP_NO_UPDATE: return "no_update";
        case LAGR_STOP_STEP_MIN: return "step_min";
        case LAGR_STOP_INFEASIBLE: return "infeasible";
        case LAGR_STOP_OOM: return "oom";
        case LAGR_STOP_NO_CANDIDATE: return "no_candidate";
        case LAGR_STOP_NOT_RUN:
        default: return "not_run";
    }
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

typedef struct {
    const int *pichart;
    int rows;
    int cols;
    int *counts;
} lagr_count_context;

static void lagr_count_col_rows_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_count_context *ctx = (lagr_count_context *)data;
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        int cnt = 0;
        for (int r = 0; r < ctx->rows; ++r) {
            cnt += ctx->pichart[c * ctx->rows + r];
        }
        ctx->counts[c] = cnt;
    }
}

static void lagr_count_row_cols_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_count_context *ctx = (lagr_count_context *)data;
    for (uint64_t r0 = start; r0 < end; r0 += stride) {
        int r = (int)r0;
        int cnt = 0;
        for (int c = 0; c < ctx->cols; ++c) {
            cnt += ctx->pichart[c * ctx->rows + r];
        }
        ctx->counts[r] = cnt;
    }
}

typedef struct {
    const int *pichart;
    int rows;
    int cols;
    int **out;
} lagr_fill_context;

static void lagr_fill_rows_covered_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_fill_context *ctx = (lagr_fill_context *)data;
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        int k = 0;
        for (int r = 0; r < ctx->rows; ++r) {
            if (ctx->pichart[c * ctx->rows + r] != 0) {
                ctx->out[c][k++] = r;
            }
        }
    }
}

static void lagr_fill_cols_covering_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_fill_context *ctx = (lagr_fill_context *)data;
    for (uint64_t r0 = start; r0 < end; r0 += stride) {
        int r = (int)r0;
        int k = 0;
        for (int c = 0; c < ctx->cols; ++c) {
            if (ctx->pichart[c * ctx->rows + r] != 0) {
                ctx->out[r][k++] = c;
            }
        }
    }
}

typedef struct {
    const double *t;
    double *partials;
} lagr_sum_t_context;

static void lagr_sum_t_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    lagr_sum_t_context *ctx = (lagr_sum_t_context *)data;
    double sum = 0.0;
    for (uint64_t i = start; i < end; i += stride) {
        sum += ctx->t[i];
    }
    ctx->partials[worker_id] = sum;
}

typedef struct {
    int **rowsCovered;
    int *rowsCoveredCount;
    const double *t;
    double *ls;
    double *partials;
} lagr_reduced_context;

static void lagr_reduced_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    lagr_reduced_context *ctx = (lagr_reduced_context *)data;
    double neg = 0.0;
    for (uint64_t c0 = start; c0 < end; c0 += stride) {
        int c = (int)c0;
        double sum = 0.0;
        for (int k = 0; k < ctx->rowsCoveredCount[c]; ++k) {
            int r = ctx->rowsCovered[c][k];
            sum += ctx->t[r];
        }
        ctx->ls[c] = 1.0 - sum;
        if (ctx->ls[c] < 0.0) neg += ctx->ls[c];
    }
    ctx->partials[worker_id] = neg;
}

typedef struct {
    const double *ls;
    unsigned char *x;
} lagr_x_context;

static void lagr_x_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_x_context *ctx = (lagr_x_context *)data;
    for (uint64_t c = start; c < end; c += stride) {
        ctx->x[c] = (ctx->ls[c] < 0.0) ? 1 : 0;
    }
}

typedef struct {
    int **colsCovering;
    int *colsCoveringCount;
    const unsigned char *x;
    double *slack;
    double *partials;
} lagr_slack_context;

static void lagr_slack_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    lagr_slack_context *ctx = (lagr_slack_context *)data;
    double sum_s2 = 0.0;
    for (uint64_t i0 = start; i0 < end; i0 += stride) {
        int i = (int)i0;
        int covered = 0;
        for (int k = 0; k < ctx->colsCoveringCount[i]; ++k) {
            int c = ctx->colsCovering[i][k];
            covered += ctx->x[c];
        }
        double s = 1.0 - (double)covered;
        ctx->slack[i] = s;
        sum_s2 += s * s;
    }
    ctx->partials[worker_id] = sum_s2;
}

typedef struct {
    double *t;
    const double *slack;
    double step;
    double stabilization_beta;
} lagr_update_t_context;

static void lagr_update_t_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_id;
    (void)worker_count;
    lagr_update_t_context *ctx = (lagr_update_t_context *)data;
    for (uint64_t i = start; i < end; i += stride) {
        double trial = ctx->t[i] + ctx->step * ctx->slack[i];
        if (trial < 0.0) trial = 0.0;
        double beta = ctx->stabilization_beta;
        double val = (beta >= 1.0)
            ? trial
            : ((1.0 - beta) * ctx->t[i] + beta * trial);
        ctx->t[i] = (val > 0.0) ? val : 0.0;
    }
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

static int build_adjacency(
    const int *pichart,
    int rows, int cols,
    int ***rowsCovered,
    int **rowsCoveredCount,
    int ***colsCovering,
    int **colsCoveringCount
) {
    int **rc = NULL, *rcc = NULL;
    int **cr = NULL, *crc = NULL;

    rcc = (int*)calloc((size_t)cols, sizeof(int));
    crc = (int*)calloc((size_t)rows, sizeof(int));
    if (!rcc || !crc) goto oom;

    lagr_count_context count_cols = {
        .pichart = pichart,
        .rows = rows,
        .cols = cols,
        .counts = rcc
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, ccubes_thread_count(), false, lagr_count_col_rows_worker, &count_cols)) {
        goto oom;
    }

    lagr_count_context count_rows = {
        .pichart = pichart,
        .rows = rows,
        .cols = cols,
        .counts = crc
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, ccubes_thread_count(), false, lagr_count_row_cols_worker, &count_rows)) {
        goto oom;
    }

    for (int r = 0; r < rows; ++r) {
        if (crc[r] == 0) {
            free(rcc);
            free(crc);
            return -2;
        }
    }

    rc = (int**)malloc((size_t)cols * sizeof(int*));
    cr = (int**)malloc((size_t)rows * sizeof(int*));
    if (!rc || !cr) goto oom;

    for (int c = 0; c < cols; ++c) {
        rc[c] = (rcc[c] > 0) ? (int*)malloc((size_t)rcc[c] * sizeof(int)) : NULL;
        if (rcc[c] > 0 && !rc[c]) goto oom;
    }
    for (int r = 0; r < rows; ++r) {
        cr[r] = (crc[r] > 0) ? (int*)malloc((size_t)crc[r] * sizeof(int)) : NULL;
        if (crc[r] > 0 && !cr[r]) goto oom;
    }

    lagr_fill_context fill_rows = {
        .pichart = pichart,
        .rows = rows,
        .cols = cols,
        .out = rc
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, ccubes_thread_count(), false, lagr_fill_rows_covered_worker, &fill_rows)) {
        goto oom;
    }

    lagr_fill_context fill_cols = {
        .pichart = pichart,
        .rows = rows,
        .cols = cols,
        .out = cr
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, ccubes_thread_count(), false, lagr_fill_cols_covering_worker, &fill_cols)) {
        goto oom;
    }

    *rowsCovered = rc;
    *rowsCoveredCount = rcc;
    *colsCovering = cr;
    *colsCoveringCount = crc;
    return 0;

oom: // out of memory
    if (rc) {
        for (int c = 0; c < cols; ++c) free(rc[c]);
        free(rc);
    }
    if (cr) {
        for (int r = 0; r < rows; ++r) free(cr[r]);
        free(cr);
    }
    free(rcc);
    free(crc);
    return -1;
}

static void free_adjacency(
    int **rowsCovered,
    int *rowsCoveredCount,
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    int rows
) {
    if (rowsCovered) {
        for (int c = 0; c < cols; ++c) free(rowsCovered[c]);
        free(rowsCovered);
    }

    free(rowsCoveredCount);

    if (colsCovering) {
        for (int r = 0; r < rows; ++r) free(colsCovering[r]);
        free(colsCovering);
    }

    free(colsCoveringCount);
}

/* Redundancy pruning: remove columns that are not needed to keep all rows covered (reverse order) */
static void prune_redundancy(
    int rows,
    int **rowsCovered,
    int *rowsCoveredCount,
    int *sol, int *sol_len
) {
    int n = *sol_len;
    if (n <= 0) return;

    int *coverCount = (int*)calloc((size_t)rows, sizeof(int));
    if (!coverCount) return;

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

            for (int j = i + 1; j < write; ++j) {
                sol[j - 1] = sol[j];
            }
            write--;
        }
    }

    *sol_len = write;
    free(coverCount);
}

/*
Greedy construction guided by Lagrangian scores
Primary: maximize newCover (minimize number of columns)
Tie 1: smaller lagr_score (more negative is better)
Tie 2: higher weight
Tie 3: smaller index
*/
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
    int *sol_len
) {
    (void)colsCovering;
    (void)colsCoveringCount; /* not used here, but kept for symmetry */

    bool *rowCovered = (bool*)calloc((size_t)rows, sizeof(bool));
    bool *colSelected = (bool*)calloc((size_t)cols, sizeof(bool));
    int covered = 0, out = 0;

    if (!rowCovered || !colSelected) {
        free(rowCovered);
        free(colSelected);
        *sol_len = -1;
        return;
    }

    while (covered < rows) {
        int best = -1;
        int bestNew = -1;
        double bestLS = DBL_MAX;
        double bestW = -DBL_MAX;

        for (int c = 0; c < cols; ++c) {
            if (colSelected[c]) continue;

            int newCover = 0;
            for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                int r = rowsCovered[c][k];
                if (!rowCovered[r]) newCover++;
            }
            if (newCover <= 0) continue;

            double ls = lagr_score ? lagr_score[c] : 0.0;
            double w = weights ? weights[c] : 0.0;

            bool better = false;
            if (newCover > bestNew) better = true;
            else if (newCover == bestNew) {
                if (ls < bestLS) {
                    better = true;
                } else if (fabs(ls - bestLS) <= EPS) {
                    if (w > bestW) {
                        better = true;
                    } else if (fabs(w - bestW) <= EPS) {
                        if (best == -1 || c < best) better = true;
                    }
                }
            }
            if (better) {
                best = c;
                bestNew = newCover;
                bestLS = ls;
                bestW = w;
            }
        }

        if (best == -1) { /* cannot cover remaining rows -> infeasible construct */
            free(rowCovered);
            free(colSelected);
            *sol_len = -1;
            return;
        }

        colSelected[best] = true;
        sol[out++] = best;

        for (int k = 0; k < rowsCoveredCount[best]; ++k) {
            int r = rowsCovered[best][k];
            if (!rowCovered[r]) {
                rowCovered[r] = true; covered++;
            }
        }
    }

    *sol_len = out;
    prune_redundancy(
        rows,
        rowsCovered,
        rowsCoveredCount,
        sol,
        sol_len
    );

    free(rowCovered);
    free(colSelected);
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
    double *ls /* out */
) {
    double ZLB = 0.0;
    double ZLB_rows = 0.0;
    double ZLB_neg = 0.0;

    int threads = ccubes_thread_count();
    double *partials = (double*)calloc((size_t)threads, sizeof(double));
    if (!partials) return 0.0;

    lagr_sum_t_context sum_ctx = {
        .t = t,
        .partials = partials
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, threads, false, lagr_sum_t_worker, &sum_ctx)) {
        free(partials);
        return 0.0;
    }
    for (int i = 0; i < threads; ++i) {
        ZLB_rows += partials[i];
        partials[i] = 0.0;
    }

    lagr_reduced_context reduced_ctx = {
        .rowsCovered = rowsCovered,
        .rowsCoveredCount = rowsCoveredCount,
        .t = t,
        .ls = ls,
        .partials = partials
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, threads, false, lagr_reduced_worker, &reduced_ctx)) {
        free(partials);
        return 0.0;
    }
    for (int i = 0; i < threads; ++i) {
        ZLB_neg += partials[i];
    }
    free(partials);

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
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    const double *ls,
    double UB, double ZLB,
    double *t,
    double *step_coef,
    double step_min,
    int *stuck_iter,
    int stuck_halve_period,
    double *prev_direction,
    double deflection_alpha,
    double step_contract,
    double stabilization_beta
) {
    int threads = ccubes_thread_count();
    unsigned char *x = (unsigned char*)malloc((size_t)cols);
    if (!x) return 0;

    lagr_x_context x_ctx = {
        .ls = ls,
        .x = x
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, threads, false, lagr_x_worker, &x_ctx)) {
        free(x);
        return 0;
    }

    double *slack = (double*)malloc((size_t)rows * sizeof(double));
    if (!slack) {
        free(x);
        return 0;
    }

    double sum_s2 = 0.0;

    double *partials = (double*)calloc((size_t)threads, sizeof(double));
    if (!partials) {
        free(x);
        free(slack);
        return 0;
    }

    lagr_slack_context slack_ctx = {
        .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount,
        .x = x,
        .slack = slack,
        .partials = partials
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, threads, false, lagr_slack_worker, &slack_ctx)) {
        free(x);
        free(slack);
        free(partials);
        return 0;
    }
    for (int i = 0; i < threads; ++i) {
        sum_s2 += partials[i];
    }
    free(partials);

    free(x);

    if (deflection_alpha > 0.0 && prev_direction) {
        sum_s2 = 0.0;
        for (int i = 0; i < rows; ++i) {
            slack[i] += deflection_alpha * prev_direction[i];
            sum_s2 += slack[i] * slack[i];
        }
    }

    if (sum_s2 <= 1e-15) {
        free(slack);
        return 0;
    }

    double gap = UB - ZLB;
    if (gap <= 0.0) {
        free(slack);
        return 0;
    }

    double step = (*step_coef) * (gap / sum_s2);
    if (step <= 0.0) {
        free(slack);
        return 0;
    }

    lagr_update_t_context update_ctx = {
        .t = t,
        .slack = slack,
        .step = step,
        .stabilization_beta = stabilization_beta
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, threads, false, lagr_update_t_worker, &update_ctx)) {
        free(slack);
        return 0;
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

    free(slack);
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
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    const double *ls,
    double UB,
    double ZLB,
    double *t,
    LagrangianBundleState *bs
) {
    if (!bs || bs->memory <= 0 || !bs->center_t || !bs->cut_alpha || !bs->cut_slacks) {
        return 0;
    }

    int threads = ccubes_thread_count();
    unsigned char *x = (unsigned char*)malloc((size_t)cols);
    if (!x) return 0;

    lagr_x_context x_ctx = {
        .ls = ls,
        .x = x
    };
    if (!ccubes_parallel_for(0, (uint64_t)cols, threads, false, lagr_x_worker, &x_ctx)) {
        free(x);
        return 0;
    }

    double *slack = (double*)malloc((size_t)rows * sizeof(double));
    if (!slack) {
        free(x);
        return 0;
    }

    double *partials = (double*)calloc((size_t)threads, sizeof(double));
    if (!partials) {
        free(x);
        free(slack);
        return 0;
    }

    lagr_slack_context slack_ctx = {
        .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount,
        .x = x,
        .slack = slack,
        .partials = partials
    };
    if (!ccubes_parallel_for(0, (uint64_t)rows, threads, false, lagr_slack_worker, &slack_ctx)) {
        free(x);
        free(slack);
        free(partials);
        return 0;
    }

    double sum_s2 = 0.0;
    for (int i = 0; i < threads; ++i) sum_s2 += partials[i];
    free(partials);
    free(x);

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
    int *sol, int *sol_len
) {
    (void)colsCoveringCount;

    bool *rowCovered = (bool*)calloc((size_t)rows, sizeof(bool));
    unsigned char *x = (unsigned char*)calloc((size_t)cols, 1);
    int out = 0, covered = 0;

    if (!rowCovered || !x) {
        free(rowCovered);
        free(x);
        *sol_len = -1;
        return;
    }

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
            free(rowCovered);
            free(x);
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
        int *coverCount = (int*)calloc((size_t)rows, sizeof(int));
        int *order = (int*)malloc((size_t)out * sizeof(int));
        double *ls_sel = (double*)malloc((size_t)out * sizeof(double));

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
            for (int i = 0; i < out; ++i) {
                for (int j = i + 1; j < out; ++j) {
                    if (ls_sel[order[j]] > ls_sel[order[i]]) {
                        int tmp = order[i];
                        order[i] = order[j];
                        order[j] = tmp;
                    }
                }
            }
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

                    /* remove from sol: shift left */
                    for (int j = idx + 1; j < out; ++j) {
                        sol[j - 1] = sol[j];
                    }
                    out--;

                    /* adjust later order indices */
                    for (int j = o + 1; j < *sol_len; ++j) {
                        if (order[j] > idx) order[j]--;
                    }
                }
            }
            *sol_len = out;
        }

        free(coverCount);
        free(order);
        free(ls_sel);
    }

    prune_redundancy(rows, rowsCovered, rowsCoveredCount, sol, sol_len);

    free(rowCovered);
    free(x);
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

/* Local search: drop-2 + greedy repair + prune; accept only if shorter */
static void local_search_drop2_repair(
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
    if (!sol || !sol_len || *sol_len <= 2) return;
    if (max_passes <= 0) return;

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

        for (int drop_i = 0; drop_i < *sol_len; ++drop_i) {
            for (int drop_j = drop_i + 1; drop_j < *sol_len; ++drop_j) {
                int cand_len = 0;
                memset(colSelected, 0, (size_t)cols);

                for (int i = 0; i < *sol_len; ++i) {
                    if (i == drop_i || i == drop_j) continue;
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

                if (cand_len <= 0 || cand_len >= *sol_len) continue;

                prune_redundancy(rows, rowsCovered, rowsCoveredCount, cand, &cand_len);

                if (cand_len > 0 && cand_len < *sol_len) {
                    memcpy(sol, cand, (size_t)cand_len * sizeof(int));
                    *sol_len = cand_len;
                    improved = 1;
                    break; /* restart after improvement */
                }
            }

            if (improved) break;
        }

        if (!improved) break;
    }

    free(rowCovered);
    free(colSelected);
    free(cand);
}

typedef struct {
    int col;
    int new_cover;
    int cover_count;
    double lagr_score;
    double weight;
} PolishCandidate;

typedef struct {
    int rows;
    int cols;
    int **rowsCovered;
    int *rowsCoveredCount;
    int **colsCovering;
    int *colsCoveringCount;
    const double *lagr_score;
    const double *weights;
    int target;
    long node_limit;
    long nodes;
    int *coverCount;
    unsigned char *colSelected;
    int *chosen;
    int *out;
} PolishSearch;

static bool polish_candidate_better(const PolishCandidate *a, const PolishCandidate *b) {
    if (a->new_cover != b->new_cover) return a->new_cover > b->new_cover;
    if (fabs(a->lagr_score - b->lagr_score) > EPS) return a->lagr_score < b->lagr_score;
    if (a->cover_count != b->cover_count) return a->cover_count > b->cover_count;
    if (fabs(a->weight - b->weight) > EPS) return a->weight > b->weight;
    return a->col < b->col;
}

static void polish_sort_candidates(PolishCandidate *candidates, int count) {
    for (int i = 1; i < count; ++i) {
        PolishCandidate value = candidates[i];
        int j = i - 1;
        while (j >= 0 && polish_candidate_better(&value, &candidates[j])) {
            candidates[j + 1] = candidates[j];
            --j;
        }
        candidates[j + 1] = value;
    }
}

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
    int slots_left
) {
    int best_row = -1;
    int best_count = INT_MAX;
    int max_new_cover = 0;

    for (int row = 0; row < search->rows; ++row) {
        if (search->coverCount[row] > 0) continue;

        int viable = 0;
        for (int i = 0; i < search->colsCoveringCount[row]; ++i) {
            int col = search->colsCovering[row][i];
            if (search->colSelected[col]) continue;
            int new_cover = polish_new_cover(search, col);
            if (new_cover <= 0) continue;
            ++viable;
            if (new_cover > max_new_cover) max_new_cover = new_cover;
        }

        if (viable == 0) return -1;
        if (viable < best_count) {
            best_count = viable;
            best_row = row;
        }
    }

    if (max_new_cover <= 0) return -1;
    if (uncovered > slots_left * max_new_cover) return -1;
    return best_row;
}

static bool polish_search_rec(
    PolishSearch *search,
    int depth,
    int covered
) {
    if (covered >= search->rows) {
        memcpy(search->out, search->chosen, (size_t)depth * sizeof(int));
        return true;
    }

    if (depth >= search->target) return false;
    if (search->nodes++ >= search->node_limit) return false;

    int slots_left = search->target - depth;
    int uncovered = search->rows - covered;
    int row = polish_choose_row(search, uncovered, slots_left);
    if (row < 0) return false;

    int raw_count = search->colsCoveringCount[row];
    PolishCandidate *candidates = (PolishCandidate*)malloc((size_t)raw_count * sizeof(PolishCandidate));
    if (!candidates) return false;

    int candidate_count = 0;
    for (int i = 0; i < raw_count; ++i) {
        int col = search->colsCovering[row][i];
        if (search->colSelected[col]) continue;

        int new_cover = polish_new_cover(search, col);
        if (new_cover <= 0) continue;

        candidates[candidate_count++] = (PolishCandidate){
            .col = col,
            .new_cover = new_cover,
            .cover_count = search->rowsCoveredCount[col],
            .lagr_score = search->lagr_score ? search->lagr_score[col] : 0.0,
            .weight = search->weights ? search->weights[col] : 0.0
        };
    }

    polish_sort_candidates(candidates, candidate_count);

    for (int i = 0; i < candidate_count; ++i) {
        int col = candidates[i].col;
        int added = 0;

        search->colSelected[col] = 1;
        search->chosen[depth] = col;

        for (int j = 0; j < search->rowsCoveredCount[col]; ++j) {
            int covered_row = search->rowsCovered[col][j];
            if (search->coverCount[covered_row] == 0) ++added;
            ++search->coverCount[covered_row];
        }

        if (polish_search_rec(search, depth + 1, covered + added)) {
            free(candidates);
            return true;
        }

        for (int j = 0; j < search->rowsCoveredCount[col]; ++j) {
            int covered_row = search->rowsCovered[col][j];
            --search->coverCount[covered_row];
        }

        search->colSelected[col] = 0;
    }

    free(candidates);
    return false;
}

static int polish_find_smaller_cover(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *lagr_score,
    const double *weights,
    int target,
    long node_limit,
    int *out
) {
    if (target <= 0 || node_limit <= 0) return 0;

    int *coverCount = (int*)calloc((size_t)rows, sizeof(int));
    unsigned char *colSelected = (unsigned char*)calloc((size_t)cols, 1);
    int *chosen = (int*)malloc((size_t)target * sizeof(int));

    if (!coverCount || !colSelected || !chosen) {
        free(coverCount);
        free(colSelected);
        free(chosen);
        return 0;
    }

    PolishSearch search = {
        .rows = rows,
        .cols = cols,
        .rowsCovered = rowsCovered,
        .rowsCoveredCount = rowsCoveredCount,
        .colsCovering = colsCovering,
        .colsCoveringCount = colsCoveringCount,
        .lagr_score = lagr_score,
        .weights = weights,
        .target = target,
        .node_limit = node_limit,
        .nodes = 0,
        .coverCount = coverCount,
        .colSelected = colSelected,
        .chosen = chosen,
        .out = out
    };

    int found = polish_search_rec(&search, 0, 0) ? 1 : 0;

    free(coverCount);
    free(colSelected);
    free(chosen);

    return found;
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

typedef struct {
    int max_iter;
    int heur_every;
    double step_coef;
    double step_min;
    double step_contract;
    double stabilization_beta;
    int halve_period;
    int local_passes;
    int small_gap_threshold;
    int small_gap_extra_passes;
    int rarity_init;
    double deflection_alpha;
    int polish_enabled;
    long polish_node_limit;
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

    cfg->max_iter = 20000;        /* total iterations of subgradient */
    cfg->heur_every = 1;          /* construct a feasible solution every N iterations */
    cfg->step_coef = 2.0;         /* initial step coefficient */
    cfg->step_min = 0.005;        /* minimal step coefficient */
    cfg->step_contract = 0.5;      /* current CCubes behavior: halve after stagnation */
    cfg->stabilization_beta = 1.0; /* no damping unless explicitly requested */
    cfg->halve_period = 8;        /* halve step if no progress for this many iterations */
    cfg->local_passes = 10;
    cfg->small_gap_threshold = 0;
    cfg->small_gap_extra_passes = cfg->local_passes;
    cfg->rarity_init = 0;
    cfg->deflection_alpha = 0.0;  /* deflected subgradient direction, disabled by default */
    cfg->polish_enabled = 1;      /* bounded one-term exact polish when UB-LB is one */
    cfg->polish_node_limit = 5000;
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

    if (effort_level == 1) {
        cfg->step_contract = 0.95;
        cfg->portfolio_enabled = 1;
        cfg->portfolio_max_profiles = 2;
    } else if (effort_level == 2) {
        cfg->step_contract = 0.95;
        cfg->bundle_interval = 8;
        cfg->bundle_momentum = 0.35;
        cfg->portfolio_enabled = 1;
        cfg->portfolio_max_profiles = 6;
        cfg->hybrid_bundle_portfolio = 1;
    }
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
    int *solution,
    int *solmin
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

    double *t = (double*)calloc((size_t)rows, sizeof(double));
    double *ls = (double*)malloc((size_t)cols * sizeof(double));
    int *sol_tmp = (int*)malloc((size_t)cols * sizeof(int));
    int *sol_tmp2 = (int*)malloc((size_t)cols * sizeof(int));
    int best_sol_size = -1;
    int stuck_iter = 0;

    if (!t || !ls || !sol_tmp || !sol_tmp2) {
        free(t);
        free(ls);
        free(sol_tmp);
        free(sol_tmp2);

        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
        return;
    }

    int local_passes = cfg->local_passes;
    int max_iter = cfg->max_iter;
    int heur_every = cfg->heur_every;
    double step_coef = cfg->step_coef;
    double step_min = cfg->step_min;
    double step_contract = cfg->step_contract;
    double stabilization_beta = cfg->stabilization_beta;
    int halve_period = cfg->halve_period;
    int small_gap_threshold = cfg->small_gap_threshold;
    int small_gap_extra_passes = cfg->small_gap_extra_passes;
    double deflection_alpha = cfg->deflection_alpha;
    double *prev_direction = NULL;
    LagrangianBundleState bundle_state = {0};

    if (deflection_alpha > 0.0) {
        prev_direction = (double*)calloc((size_t)rows, sizeof(double));
        if (!prev_direction) {
            free(t);
            free(ls);
            free(sol_tmp);
            free(sol_tmp2);
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
            free(ls);
            free(sol_tmp);
            free(sol_tmp2);
            lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
            return;
        }
    }

    lagr_initialize_multipliers(rows, colsCoveringCount, cfg->rarity_init, t);

    double bestZLB = -DBL_MAX;
    double lastZLB = -DBL_MAX;
    int iterations = 0;
    LagrangianStopReason stop_reason = LAGR_STOP_MAX_ITER;

    /* Initial heuristics with current ls (computed from t) */
    double initialZLB = compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        ls
    );
    bestZLB = initialZLB;
    lastZLB = initialZLB;

    int sol_size = -1, sol_size2 = -1;

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
        &sol_size
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
        &sol_size2
    );

    /* Optional deeper search: disabled by default */
    if (local_passes > 0) {
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
        }
    }

    if (sol_size == -1 && sol_size2 == -1) {
        free(t);
        free(ls);
        free(sol_tmp);
        free(sol_tmp2);
        free(bundle_state.center_t);
        free(bundle_state.prev_center_t);
        free(bundle_state.cut_alpha);
        free(bundle_state.cut_slacks);
        lagr_stats_finish(-1, -DBL_MAX, bestZLB, lastZLB, step_coef, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }

    double w1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
    double w2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

    double bestUB; /* UB equals current solution length (unit-cost) */
    int bestTerms;
    double bestWeight;

    if (
        sol_size != -1 &&
        (
            sol_size2 == -1 ||
            sol_size < sol_size2 ||
            (
                sol_size == sol_size2 && w1 >= w2
            )
        )
    ) {

        bestUB = (double)sol_size;
        bestTerms = sol_size;
        bestWeight = w1;
        best_sol_size = sol_size;

        memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
    } else {

        bestUB = (double)sol_size2;
        bestTerms = sol_size2;
        bestWeight = w2;
        best_sol_size = sol_size2;

        memcpy(solution, sol_tmp2, (size_t)sol_size2 * sizeof(int));
    }

    double bestLB = -DBL_MAX;
    for (int it = 0; it < max_iter; ++it) {
        iterations = it + 1;
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            ls
        );
        lastZLB = ZLB;
        if (ZLB > bestZLB + EPS) {
            bestZLB = ZLB;
        }

        double LBint = ceil(ZLB - 1e-12);

        if (LBint > bestLB + EPS) {
            bestLB = LBint;
            stuck_iter = 0; /* reset halving on improvement */
        } /* else no improvement */

        if (it % heur_every == 0) {
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
                &sol_size
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
                &sol_size2
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
                        ls,
                        weights,
                        sol_tmp,
                        &sol_size,
                        local_passes
                    );

                    local_search_drop2_repair(
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
                }
            }

            if (
                small_gap_extra_passes > local_passes &&
                small_gap_threshold > 0 &&
                bestTerms < INT_MAX &&
                bestTerms - (int)LBint <= small_gap_threshold
            ) {
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
                        small_gap_extra_passes
                    );

                    local_search_drop2_repair(
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
                        small_gap_extra_passes
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
                        small_gap_extra_passes
                    );

                    local_search_drop2_repair(
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
                        small_gap_extra_passes
                    );
                }
            }

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
            }
        }

        if (bestUB <= LBint + EPS) {
            stop_reason = LAGR_STOP_OPTIMAL;
            break; /* proved optimal (within epsilon) */
        }

        int used_subgradient_update = 0;
        int updated = 0;

        int use_bundle_update = cfg->bundle_enabled &&
            (cfg->bundle_interval <= 1 || it % cfg->bundle_interval == 0);

        if (use_bundle_update) {
            updated = bundle_update(
                rows,
                cols,
                colsCovering,
                colsCoveringCount,
                ls,
                bestUB,
                ZLB,
                t,
                &bundle_state
            );
        }

        if (!use_bundle_update || !updated) {
            updated = subgradient_update(
                rows,
                cols,
                colsCovering,
                colsCoveringCount,
                ls,
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
                stabilization_beta
            );
            used_subgradient_update = 1;
        }

        if (!updated || (used_subgradient_update && step_coef <= step_min + EPS)) {
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
                &sol_size
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
                &sol_size2
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
                        ls,
                        weights,
                        sol_tmp,
                        &sol_size,
                        local_passes
                    );

                    local_search_drop2_repair(
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
                best_sol_size = sol_size;

                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
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
            }
            break;
        }
    }

    if (cfg->polish_enabled && best_sol_size > 1) {
        int polish_target = best_sol_size - 1;
        if (
            bestLB >= (double)polish_target - EPS &&
            bestLB <= (double)polish_target + EPS &&
            polish_find_smaller_cover(
                rows,
                cols,
                rowsCovered,
                rowsCoveredCount,
                colsCovering,
                colsCoveringCount,
                ls,
                weights,
                polish_target,
                cfg->polish_node_limit,
                sol_tmp
            )
        ) {
            best_sol_size = polish_target;
            bestTerms = polish_target;
            bestUB = (double)polish_target;
            bestWeight = solution_total_weight(sol_tmp, polish_target, weights);
            memcpy(solution, sol_tmp, (size_t)polish_target * sizeof(int));

            if (bestLB >= bestUB - EPS) {
                stop_reason = LAGR_STOP_OPTIMAL;
            }
        }
    }

    *solmin = best_sol_size;
    lagr_stats_finish(best_sol_size, bestLB, bestZLB, lastZLB, step_coef, iterations, stop_reason);

    free(t);
    free(ls);
    free(sol_tmp);
    free(sol_tmp2);
    free(prev_direction);
    free(bundle_state.center_t);
    free(bundle_state.prev_center_t);
    free(bundle_state.cut_alpha);
    free(bundle_state.cut_slacks);
}

void solve_scp_lagrangian(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double weights[],
    int *solution,
    int *solmin,
    int effort_level
) {
    if (solmin) *solmin = -1;
    lagr_stats_begin(ON_minterms, foundPI, 0);
    if (!pichart || foundPI <= 0 || ON_minterms <= 0 || !solution || !solmin) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_NO_CANDIDATE);
        return;
    }
    int **rowsCovered = NULL, *rowsCoveredCount = NULL;
    int **colsCovering = NULL, *colsCoveringCount = NULL;

    int rc_ad = build_adjacency(
        pichart,
        ON_minterms,
        foundPI,
        &rowsCovered,
        &rowsCoveredCount,
        &colsCovering,
        &colsCoveringCount
    );

    if (rc_ad == -2) { /* infeasible matrix (some row uncovered) */
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_INFEASIBLE);
        return;
    }

    if (rc_ad == -1) {
        lagr_stats_finish(-1, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0, LAGR_STOP_OOM);
        return;
    }

    LagrangianConfig baseline;
    lagr_config_for_effort(&baseline, effort_level);
    int portfolio_enabled = baseline.portfolio_enabled;
    double portfolio_deflection_alpha = baseline.portfolio_deflection_alpha;

    if (portfolio_enabled) {
        baseline.deflection_alpha = 0.0;
    }

    solve_scp_lagrangian_config(
        ON_minterms,
        foundPI,
        rowsCovered,
        rowsCoveredCount,
        colsCovering,
        colsCoveringCount,
        weights,
        &baseline,
        solution,
        solmin
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
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            foundPI,
            colsCovering,
            colsCoveringCount,
            ON_minterms
        );
        return;
    }

    LagrangianConfig profiles[LAGR_PORTFOLIO_PROFILE_COUNT];
    int profile_count = lagr_build_portfolio_configs(&baseline, portfolio_deflection_alpha, profiles);
    int max_profiles = baseline.portfolio_max_profiles;
    if (profile_count > max_profiles) profile_count = max_profiles;

    if (profile_count <= 1) {
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            foundPI,
            colsCovering,
            colsCoveringCount,
            ON_minterms
        );
        return;
    }

    int *candidate = (int*)malloc((size_t)foundPI * sizeof(int));
    if (!candidate) {
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            foundPI,
            colsCovering,
            colsCoveringCount,
            ON_minterms
        );
        return;
    }

    for (int i = 1; i < profile_count; ++i) {
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
            candidate,
            &candidate_solmin
        );

        const LagrangianStats *candidate_stats = lagrangian_last_stats();
        total_iterations += candidate_stats->iterations;

        if (candidate_stats->best_lb > merged_stats.best_lb) {
            merged_stats.best_lb = candidate_stats->best_lb;
            merged_stats.last_zlb = candidate_stats->last_zlb;
            merged_stats.step_coef = candidate_stats->step_coef;
            merged_stats.stop_reason = candidate_stats->stop_reason;
        }
        if (candidate_stats->best_zlb > merged_stats.best_zlb + EPS) {
            merged_stats.best_zlb = candidate_stats->best_zlb;
            merged_stats.last_zlb = candidate_stats->last_zlb;
            merged_stats.step_coef = candidate_stats->step_coef;
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

    free(candidate);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        foundPI,
        colsCovering,
        colsCoveringCount,
        ON_minterms
    );
}

void solve_scp_lagrangian_pool(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double weights[],
    int max_pool,
    int *out_pool_count,
    int **pool_solutions,
    int *solmin,
    int effort_level
) {
    if (solmin) *solmin = -1;
    lagr_stats_begin(ON_minterms, foundPI, 1);
    if (
        !pichart ||
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
        pichart,
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

    double *t = (double*)calloc((size_t)rows, sizeof(double));
    double *rc = (double*)malloc((size_t)cols * sizeof(double));
    int *sol_tmp = (int*)malloc((size_t)cols * sizeof(int));
    int *sol_tmp2 = (int*)malloc((size_t)cols * sizeof(int));
    int *norm = (int*)malloc((size_t)cols * sizeof(int));

    if (!t || !rc || !sol_tmp || !sol_tmp2 || !norm) {
        free(t);
        free(rc);
        free(sol_tmp);
        free(sol_tmp2);
        free(norm);
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

    int pool_count = 0;
    int pool_min_len = INT_MAX; /* track minimal len seen, pool only stores solutions with this len */

    LagrangianConfig cfg;
    lagr_config_for_effort(&cfg, effort_level);
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
            free(rc);
            free(sol_tmp);
            free(sol_tmp2);
            free(norm);
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
        rc
    );
    bestZLB = initialZLB;
    lastZLB = initialZLB;

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
        &sol_size
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
        &sol_size2
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
        }
    }

    if (sol_size == -1 && sol_size2 == -1) {
        free(t);
        free(rc);
        free(sol_tmp);
        free(sol_tmp2);
        free(norm);
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

    double bestUB;
    int bestTerms;
    double bestWeight;

    if (
        sol_size != -1 &&
        (
            sol_size2 == -1 ||
            sol_size < sol_size2 ||
            (
                sol_size == sol_size2 && w1 >= w2
            )
        )
    ) {
        bestUB = (double)sol_size;
        bestTerms = sol_size;
        bestWeight = w1;
        *solmin = sol_size;
    } else {
        bestUB = (double)sol_size2;
        bestTerms = sol_size2;
        bestWeight = w2;
        *solmin = sol_size2;
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
    }

    double bestLB = -DBL_MAX;

    /* parameters */
    const int max_iter = (cfg.max_iter > 5000) ? cfg.max_iter : 5000;
    const int heur_every = cfg.heur_every;
    double step_coef = cfg.step_coef;
    const double step_min = cfg.step_min;
    const int halve_period = cfg.halve_period;
    int stuck_iter = 0;

    for (int it = 0; it < max_iter; ++it) {
        iterations = it + 1;
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            rc
        );
        lastZLB = ZLB;
        if (ZLB > bestZLB + EPS) {
            bestZLB = ZLB;
        }

        double LBint = ceil(ZLB - EPS);
        if (LBint > bestLB + EPS) {
            bestLB = LBint;
            stuck_iter = 0;
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
                &sol_size
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
                &sol_size2
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
                cols,
                colsCovering,
                colsCoveringCount,
                rc,
                bestUB,
                ZLB,
                t,
                &bundle_state
            );
        }

        if (!use_bundle_update || !updated) {
            updated = subgradient_update(
                rows,
                cols,
                colsCovering,
                colsCoveringCount,
                rc,
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
                cfg.stabilization_beta
            );
            used_subgradient_update = 1;
        }

        if (!updated || (used_subgradient_update && step_coef <= step_min + EPS)) {
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
                &sol_size
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
                &sol_size2
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

    lagr_stats_finish(
        (*solmin > 0) ? *solmin : bestTerms,
        bestLB,
        bestZLB,
        lastZLB,
        step_coef,
        iterations,
        stop_reason
    );

    /* output */
    if (max_pool > 1 && out_pool_count) {
        /* Return the pool */
        *out_pool_count = pool_count;
    }

    /* cleanup */
    free(t);
    free(rc);
    free(sol_tmp);
    free(sol_tmp2);
    free(norm);
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
