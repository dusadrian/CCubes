/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "lagrangian.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

#define EPS 1e-12

/* Build adjacency: rowsCovered[c] = list of rows that column c covers
                    colsCovering[r] = list of columns that cover row r
Returns 0 on success, -2 if any row has no covering column (infeasible), -1 on OOM. */
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
                removable = false; break;
            }
        }

        if (removable) {
            for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                int r = rowsCovered[c][k];
                coverCount[r]--;
            }

            for (int j = i + 1; j < write; ++j) sol[j - 1] = sol[j];
            write--;
        }
    }

    *sol_len = write;
    free(coverCount);
}

/* Greedy construction guided by reduced costs:
Primary: maximize newCover (min #columns objective)
Tie 1: smaller reduced_cost (more negative is better)
Tie 2: higher weight
Tie 3: smaller index
*/
static void greedy_from_reduced_costs(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *reduced_cost,
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
        double bestRC = DBL_MAX;
        double bestW = -DBL_MAX;

        for (int c = 0; c < cols; ++c) {
            if (colSelected[c]) continue;

            int newCover = 0;
            for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                int r = rowsCovered[c][k];
                if (!rowCovered[r]) newCover++;
            }
            if (newCover <= 0) continue;

            double rc = reduced_cost ? reduced_cost[c] : 0.0;
            double w = weights ? weights[c] : 0.0;

            bool better = false;
            if (newCover > bestNew) better = true;
            else if (newCover == bestNew) {
                if (rc < bestRC) {
                    better = true;
                } else if (fabs(rc - bestRC) <= EPS) {
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
                bestRC = rc;
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

/* Compute Lagrangian reduced costs with real column costs:
rc[c] = cost[c] - sum_{r in rowsCovered[c]} t[r]
ZLB = sum_i t[i] + sum_c min(0, rc[c]) */
static double compute_reduced_and_lb(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    const double *t,
    const double *col_costs,
    double *rc /* out */
) {
    double ZLB = 0.0;
    double ZLB_rows = 0.0;
    double ZLB_neg = 0.0;

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ZLB_rows) schedule(static)
    #endif
    for (int i = 0; i < rows; ++i) ZLB_rows += t[i];

    #ifdef _OPENMP
        #pragma omp parallel for reduction(+:ZLB_neg) schedule(static)
    #endif
    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int k = 0; k < rowsCoveredCount[c]; ++k) {
            int r = rowsCovered[c][k];
            sum += t[r];
        }
        double cc = col_costs ? col_costs[c] : 1.0;
        rc[c] = cc - sum;
        if (rc[c] < 0.0) ZLB_neg += rc[c];
    }
    ZLB = ZLB_rows + ZLB_neg;
    return ZLB;
}

/* Subgradient update (unchanged logic, but UB is real cost):
x[c] = 1 if rc[c] < 0 else 0 (LP solution)
slack[i] = 1 - sum_{c covering i} x[c]
step = step_coef * (UB - ZLB) / sum(slack^2), t := max(0, t + step*slack)
Returns 1 if step applied, 0 if slacks zero or no progress possible. */
static int subgradient_update(
    int rows,
    int cols,
    int **colsCovering,
    int *colsCoveringCount,
    const double *rc,
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
    for (int c = 0; c < cols; ++c) x[c] = (rc[c] < 0.0) ? 1 : 0;

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

/* Row-targeted heuristic: for each uncovered row, pick a covering column with min reduced cost.
Tie: fewer rc (more negative) preferred; if equal, prefer covering more rows; then lower cost; then lower index. */
static void heuristic_row_min_rc(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    int **colsCovering,
    int *colsCoveringCount,
    const double *reduced_cost,
    const double *weights, /* tie-break: higher is better */
    const double *col_costs,
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
        double bestRC = DBL_MAX;
        int bestOnes = -1;
        double bestCost = DBL_MAX;
        double bestWeight = -DBL_MAX;

        for (int k = 0; k < colsCoveringCount[i]; ++k) {
            int c = colsCovering[i][k];
            double rc = reduced_cost ? reduced_cost[c] : 0.0;
            double cc = col_costs ? col_costs[c] : 1.0;
            double w  = weights ? weights[c] : 0.0;
            int ones = rowsCoveredCount[c];

            bool better = false;
            if (rc < bestRC) better = true;
            else if (fabs(rc - bestRC) <= EPS) {
                if (w > bestWeight) better = true;
                else if (fabs(w - bestWeight) <= EPS) {
                    if (ones > bestOnes) better = true;
                    else if (ones == bestOnes) {
                        if (cc < bestCost) better = true;
                        else if (fabs(cc - bestCost) <= EPS) {
                            if (best == -1 || c < best) better = true;
                        }
                    }
                }
            }
            if (better) {
                best = c;
                bestRC = rc;
                bestOnes = ones;
                bestCost = cc;
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
            for (int kk = 0; kk < rowsCoveredCount[best]; ++kk) {
                int r = rowsCovered[best][kk];
                if (!rowCovered[r]) { rowCovered[r] = true; covered++; }
            }
        }

        if (covered >= rows) break;
    }

    *sol_len = out;

    /* Removal: try to drop columns in order of worst reduced cost first */
    if (out > 0) {
        /* Build cover counts */
        int *coverCount = (int*)calloc((size_t)rows, sizeof(int));
        int *order = (int*)malloc((size_t)out * sizeof(int));
        double *rc_sel = (double*)malloc((size_t)out * sizeof(double));
        if (coverCount && order && rc_sel) {
            for (int i = 0; i < out; ++i) {
                int c = sol[i];
                order[i] = i; /* index into sol */
                rc_sel[i] = reduced_cost ? reduced_cost[c] : 0.0;
                for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                    int r = rowsCovered[c][k];
                    coverCount[r]++;
                }
            }
            /* sort order by rc descending (worst first) */
            for (int i = 0; i < out; ++i) {
                for (int j = i + 1; j < out; ++j) {
                    if (rc_sel[order[j]] > rc_sel[order[i]]) {
                        int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
                    }
                }
            }
            /* attempt removals */
            for (int oi = 0; oi < out; ++oi) {
                int idx = order[oi];
                if (idx < 0) continue;
                int c = sol[idx];
                bool can_remove = true;
                for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                    int r = rowsCovered[c][k];
                    if (coverCount[r] <= 1) {
                        can_remove = false;
                        break;
                    }
                }
                if (can_remove) {
                    for (int k = 0; k < rowsCoveredCount[c]; ++k) {
                        int r = rowsCovered[c][k];
                        coverCount[r]--;
                    }

                    /* remove from sol: shift left */
                    for (int j = idx + 1; j < out; ++j) {
                        sol[j - 1] = sol[j];
                    }
                    out--;

                    /* adjust later order indices */
                    for (int jj = oi + 1; jj < *sol_len; ++jj) {
                        if (order[jj] > idx) order[jj]--;
                    }
                }
            }
            *sol_len = out;
        }
        free(coverCount);
        free(order);
        free(rc_sel);
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

static double solution_total_cost(const int *sol, int sol_len, const double *col_costs) {
    double s = 0.0;

    if (sol && sol_len > 0) {
        for (int i = 0; i < sol_len; ++i) {
            int c = sol[i];
            s += col_costs ? col_costs[c] : 1.0;
        }
    }

    return s;
}

/* Public API (cost-aware): Lagrangian-based SCP heuristic using real column costs.
- pichart: column-major (pichart[c * ON_minterms + r])
- col_costs: per-column costs (NULL => unit costs)
- foundPI: columns
- ON_minterms: rows
- weights: can be NULL; higher is better (tie-break inside heuristics)
- solution: out column indices (0-based), size >= foundPI
- solmin: out size (number of columns); -1 if infeasible
*/

void solve_scp_lagrangian(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    const double col_costs[],
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
    double *rc = (double*)malloc((size_t)cols * sizeof(double));
    int *sol_tmp = (int*)malloc((size_t)cols * sizeof(int));
    int *sol_tmp2 = (int*)malloc((size_t)cols * sizeof(int));
    int best_sol_size = -1;
    int stuck_iter = 0;

    if (!t || !rc || !sol_tmp || !sol_tmp2) {
        free(t);
        free(rc);
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
    const int max_iter = 5000;           /* total iterations of subgradient */
    const int heur_every = 3;            /* construct a feasible solution every N iterations */
    double step_coef = 1.5;              /* initial step coefficient */
    const double step_min = 0.005;       /* minimal step coefficient */
    const int halve_period = 8;          /* halve step if no progress for this many iterations */

    /* Better initialization of multipliers: t[i] = min cost of a covering column */
    for (int i = 0; i < rows; ++i) {
        double m = DBL_MAX;
        for (int k = 0; k < colsCoveringCount[i]; ++k) {
            int c = colsCovering[i][k];
            double cc = col_costs ? col_costs[c] : 1.0;
            if (cc < m) m = cc;
        }
        t[i] = (m == DBL_MAX) ? 1.0 : m;
    }

    /* Initial heuristics with current rc (computed from t) */
    compute_reduced_and_lb(
        rows,
        cols,
        rowsCovered,
        rowsCoveredCount,
        t,
        col_costs,
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
        col_costs,
        sol_tmp,
        &sol_size
    );

    greedy_from_reduced_costs(
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

    double cost1 = (sol_size != -1) ? solution_total_cost(sol_tmp, sol_size, col_costs) : DBL_MAX;
    double cost2 = (sol_size2 != -1) ? solution_total_cost(sol_tmp2, sol_size2, col_costs) : DBL_MAX;
    double w1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
    double w2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

    double bestUBCost;
    int bestTerms;
    double bestWeight;
    if (
        cost1 < cost2 - EPS ||
        (
            fabs(cost1 - cost2) <= EPS &&
            (
                sol_size < sol_size2 || (sol_size == sol_size2 && w1 >= w2)
            )
        )
    ) {
        bestUBCost = cost1;
        bestTerms = sol_size;
        bestWeight = w1;
        best_sol_size = sol_size;
        memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
    } else {
        bestUBCost = cost2;
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
            col_costs,
            rc
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
                rc,
                weights,
                col_costs,
                sol_tmp,
                &sol_size
            );

            greedy_from_reduced_costs(
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

            double candCost1 = (sol_size != -1) ? solution_total_cost(sol_tmp, sol_size, col_costs) : DBL_MAX;
            double candCost2 = (sol_size2 != -1) ? solution_total_cost(sol_tmp2, sol_size2, col_costs) : DBL_MAX;
            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;

            /* Improve by primary objective cost; if equal, fewer terms; if still equal, higher weight */
            if (
                candCost1 < bestUBCost - EPS ||
                (
                    fabs(candCost1 - bestUBCost) <= EPS &&
                    (sol_size < bestTerms || (sol_size == bestTerms && candW1 > bestWeight + EPS))
                )
            ) {
                bestUBCost = candCost1;
                bestTerms = sol_size;
                bestWeight = candW1;
                best_sol_size = sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
            }

            if (
                candCost2 < bestUBCost - EPS ||
                (
                    fabs(candCost2 - bestUBCost) <= EPS &&
                    (sol_size2 < bestTerms || (sol_size2 == bestTerms && candW2 > bestWeight + EPS))
                )
            ) {
                bestUBCost = candCost2;
                bestTerms = sol_size2;
                bestWeight = candW2;
                best_sol_size = sol_size2;
                memcpy(solution, sol_tmp2, (size_t)sol_size2 * sizeof(int));
            }
        }

        if (bestUBCost <= LBint + EPS) break; /* proved optimal (within epsilon) */

        int updated = subgradient_update(
            rows,
            cols,
            colsCovering,
            colsCoveringCount,
            rc,
            bestUBCost,
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
                col_costs,
                sol_tmp,
                &sol_size
            );

            greedy_from_reduced_costs(
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

            double candCost1 = (sol_size != -1) ? solution_total_cost(sol_tmp, sol_size, col_costs) : DBL_MAX;
            double candCost2 = (sol_size2 != -1) ? solution_total_cost(sol_tmp2, sol_size2, col_costs) : DBL_MAX;
            double candW1 = (sol_size != -1) ? solution_total_weight(sol_tmp, sol_size, weights) : -DBL_MAX;
            double candW2 = (sol_size2 != -1) ? solution_total_weight(sol_tmp2, sol_size2, weights) : -DBL_MAX;
            if (
                candCost1 < bestUBCost - EPS ||
                (
                    fabs(candCost1 - bestUBCost) <= EPS &&
                    (
                        sol_size < bestTerms || (sol_size == bestTerms && candW1 > bestWeight + EPS)
                    )
                )
            ) {
                bestUBCost = candCost1;
                bestTerms = sol_size;
                bestWeight = candW1;
                best_sol_size = sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
            }

            if (
                candCost2 < bestUBCost - EPS ||
                (
                    fabs(candCost2 - bestUBCost) <= EPS &&
                    (
                        sol_size2 < bestTerms || (sol_size2 == bestTerms && candW2 > bestWeight + EPS)
                    )
                )
            ) {
                bestUBCost = candCost2;
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
    free(rc);
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
