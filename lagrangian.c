/*
    Copyright (c) 2016–2025, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "lagrangian.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

#define EPS 1e-12

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

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int c = 0; c < cols; ++c) {
        int cnt = 0;
        for (int r = 0; r < rows; ++r) {
            cnt += pichart[c * rows + r];
        }
        rcc[c] = cnt;
    }

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; ++r) {
        int cnt = 0;
        for (int c = 0; c < cols; ++c) {
            cnt += pichart[c * rows + r];
        }
        crc[r] = cnt;
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

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int c = 0; c < cols; ++c) {
        int k = 0;
        for (int r = 0; r < rows; ++r) {
            if (pichart[c * rows + r] != 0) {
                rc[c][k++] = r;
            }
        }
    }

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; ++r) {
        int k = 0;
        for (int c = 0; c < cols; ++c) {
            if (pichart[c * rows + r] != 0) {
                cr[r][k++] = c;
            }
        }
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

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ZLB_rows) schedule(static)
    #endif
    for (int i = 0; i < rows; ++i) {
        ZLB_rows += t[i];
    }

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ZLB_neg) schedule(static)
    #endif
    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int k = 0; k < rowsCoveredCount[c]; ++k) {
            int r = rowsCovered[c][k];
            sum += t[r];
        }
        double cc = 1.0;
        ls[c] = cc - sum;
        if (ls[c] < 0.0) ZLB_neg += ls[c];
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
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    const double *ls,
    double UB, double ZLB,
    double *t,
    double *step_coef,
    double step_min,
    int *stuck_iter,
    int stuck_halve_period
) {
    unsigned char *x = (unsigned char*)malloc((size_t)cols);
    if (!x) return 0;

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int c = 0; c < cols; ++c) {
        x[c] = (ls[c] < 0.0) ? 1 : 0;
    }

    double *slack = (double*)malloc((size_t)rows * sizeof(double));
    if (!slack) {
        free(x);
        return 0;
    }

    double sum_s2 = 0.0;

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:sum_s2) schedule(static)
    #endif
    for (int i = 0; i < rows; ++i) {
        int covered = 0;
        for (int k = 0; k < colsCoveringCount[i]; ++k) {
            int c = colsCovering[i][k];
            covered += x[c];
        }
        double s = 1.0 - (double)covered;
        slack[i] = s;
        sum_s2 += s * s;
    }

    free(x);

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

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < rows; ++i) {
        double val = t[i] + step * slack[i];
        t[i] = (val > 0.0) ? val : 0.0;
    }

    (*stuck_iter)++;
    if (*stuck_iter >= stuck_halve_period) {
        *stuck_iter = 0;
        *step_coef *= 0.5;
        if (*step_coef < step_min) *step_coef = step_min;
    }

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

/* ---------- Main functions ---------- */
void solve_scp_lagrangian(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double weights[],
    int *solution,
    int *solmin
) {
    *solmin = -1;
    if (!pichart || foundPI <= 0 || ON_minterms <= 0 || !solution || !solmin) return;

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
        *solmin = -1;
        return;
    }

    if (rc_ad == -1) {
        *solmin = -1;
        return;
    }

    int rows = ON_minterms, cols = foundPI;

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

        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            cols,
            colsCovering,
            colsCoveringCount,
            rows
        );

        *solmin = -1;
        return;
    }

    /* Heuristic and subgradient parameters (tunable) */
    int max_iter = 5000;            /* total iterations of subgradient */
    int heur_every = 2;             /* construct a feasible solution every N iterations (tuned default) */
    double step_coef = 1.5;         /* initial step coefficient */
    double step_min = 0.005;        /* minimal step coefficient */
    int halve_period = 8;           /* halve step if no progress for this many iterations */

    /* Environment overrides (for tuning/experiments) */
    const char *s_env = NULL;
    s_env = getenv("CCUBES_LAGR_MAX_ITER");
    if (s_env) {
        long v = strtol(s_env, NULL, 10);
        if (v > 10 && v < 50000000) max_iter = (int)v;
    }
    s_env = getenv("CCUBES_LAGR_HEUR_EVERY");
    if (s_env) {
        long v = strtol(s_env, NULL, 10);
        if (v >= 1 && v < 10000) heur_every = (int)v;
    }
    s_env = getenv("CCUBES_LAGR_STEP_COEF");
    if (s_env) {
        double v = strtod(s_env, NULL);
        if (v > 0.0 && v < 1000.0) step_coef = v;
    }
    s_env = getenv("CCUBES_LAGR_STEP_MIN");
    if (s_env) {
        double v = strtod(s_env, NULL);
        if (v > 0.0 && v < step_coef) step_min = v;
    }
    s_env = getenv("CCUBES_LAGR_HALVE_PERIOD");
    if (s_env) {
        long v = strtol(s_env, NULL, 10);
        if (v >= 1 && v < 10000) halve_period = (int)v;
    }

    /* Initialize multipliers to unit cost */
    for (int i = 0; i < rows; ++i) {
        (void)colsCoveringCount;
        (void)colsCovering;
        t[i] = 1.0;
    }

    /* Initial heuristics with current ls (computed from t) */
    compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        ls
    );

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

    if (sol_size == -1 && sol_size2 == -1) {
        free(t);
        free(ls);
        free(sol_tmp);
        free(sol_tmp2);
        free_adjacency(
            rowsCovered,
            rowsCoveredCount,
            cols,
            colsCovering,
            colsCoveringCount,
            rows
        );
        *solmin = -1;
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
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            ls
        );

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

        if (bestUB <= LBint + EPS) break; /* proved optimal (within epsilon) */

        int updated = subgradient_update(
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
            halve_period
        );

        if (!updated || step_coef <= step_min + EPS) {
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

    *solmin = best_sol_size;

    free(t);
    free(ls);
    free(sol_tmp);
    free(sol_tmp2);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        cols,
        colsCovering,
        colsCoveringCount,
        rows
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
    int *solmin
) {
    if (solmin) *solmin = -1;
    if (
        !pichart ||
        foundPI <= 0 ||
        ON_minterms <= 0 ||
        !out_pool_count ||
        !pool_solutions ||
        !solmin
    ) return;

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
        return;
    }

    int pool_count = 0;
    int pool_min_len = INT_MAX; /* track minimal len seen, pool only stores solutions with this len */

    if (!(max_pool > 1 && out_pool_count && pool_solutions)) {
        max_pool = 1;
    }

    /* init multipliers (unit cost) */
    for (int i = 0; i < rows; ++i) {
        (void)colsCoveringCount;
        (void)colsCovering;
        t[i] = 1.0;
    }

    /* initial rc */
    compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        rc
    );

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

    if (sol_size == -1 && sol_size2 == -1) {
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
    const int max_iter = 5000;
    const int heur_every = 3;
    double step_coef = 1.5;
    const double step_min = 0.005;
    const int halve_period = 8;
    int stuck_iter = 0;

    for (int it = 0; it < max_iter; ++it) {
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            rc
        );

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

        if (bestUB <= LBint + EPS) break;

        int updated = subgradient_update(
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
            halve_period
        );

        if (!updated || step_coef <= step_min + EPS) {
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
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        cols,
        colsCovering,
        colsCoveringCount,
        rows
    );
}
