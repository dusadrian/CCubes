/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "lagrangian.h"


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

    for (int c = 0; c < cols; ++c) {
        int cnt = 0;
        for (int r = 0; r < rows; ++r) {
            cnt += pichart[c * rows + r];
        }
        rcc[c] = cnt;
    }
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

    for (int c = 0; c < cols; ++c) {
        int k = 0;
        for (int r = 0; r < rows; ++r) {
            if (pichart[c * rows + r] != 0) {
                rc[c][k++] = r;
            }
        }
    }

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
    int *sol, int *sol_len
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
                } else if (rc == bestRC) {
                    if (w > bestW) {
                        better = true;
                    } else if (w == bestW) {
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

/* Compute Lagrangian reduced costs (unit cost per column) and the dual lower bound (ZLB).
rc[c] = 1.0 - sum_{r in rowsCovered[c]} t[r]
ZLB = sum_i t[i] + sum_c min(0, rc[c]) */
static double compute_reduced_and_lb(
    int rows,
    int cols,
    int **rowsCovered,
    int *rowsCoveredCount,
    const double *t,
    double *rc /* out */
) {
    double ZLB = 0.0;
    for (int i = 0; i < rows; ++i) ZLB += t[i];

    for (int c = 0; c < cols; ++c) {
        double sum = 0.0;
        for (int k = 0; k < rowsCoveredCount[c]; ++k) {
            int r = rowsCovered[c][k];
            sum += t[r];
        }
        rc[c] = 1.0 - sum; /* unit costs: c_j = 1 */
        if (rc[c] < 0.0) ZLB += rc[c];
    }
    return ZLB;
}

/* Subgradient update:
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
    for (int c = 0; c < cols; ++c) x[c] = (rc[c] < 0.0) ? 1 : 0;

    double *slack = (double*)malloc((size_t)rows * sizeof(double));
    if (!slack) { free(x); return 0; }

    double sum_s2 = 0.0;
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

/* Public API: Lagrangian-based SCP heuristic (unit costs + weight tie-breaks)
- pichart: column-major (pichart[c * ON_minterms + r])
- foundPI: columns
- ON_minterms: rows
- weights: can be NULL; higher is better (tie-break inside heuristic)
- solution: out column indices (0-based), size >= foundPI
- solmin: out size; -1 if infeasible
*/
void solve_scp_lagrangian(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    double weights[],
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
    int best_sol_size = -1;

    if (!t || !rc || !sol_tmp) {
        if (t) free(t);
        if (rc) free(rc);
        if (sol_tmp) free(sol_tmp);
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
    const int max_iter = 2000;           /* total iterations of subgradient */
    const int heur_every = 5;            /* construct a feasible solution every N iterations */
    double step_coef = 2.0;              /* initial step coefficient */
    const double step_min = 0.01;        /* minimal step coefficient */
    const int halve_period = 10;         /* halve step if no progress for this many iterations */

    double bestUB = DBL_MAX;             /* best cardinality found */
    int stuck_iter = 0;

    /* Initial heuristic (t = 0 => rc = 1): pure greedy by newCover, tie by weight */
    for (int c = 0; c < cols; ++c) rc[c] = 1.0;
    int sol_size = -1;
    greedy_from_reduced_costs(
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

    if (sol_size == -1) { /* shouldn't happen if adjacency ok */
        free(t);
        free(rc);
        free(sol_tmp);
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
    bestUB = (double)sol_size;
    best_sol_size = sol_size;
    memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));

    double bestLB = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        double ZLB = compute_reduced_and_lb(
            rows,
            cols,
            rowsCovered,
            rowsCoveredCount,
            t,
            rc
        );

        double LBint = ceil(ZLB - 1e-12);
        if (LBint > bestLB) {
            bestLB = LBint;
            stuck_iter = 0; /* reset halving counter if LB improved */
        }

        if (it % heur_every == 0) {
            greedy_from_reduced_costs(
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

            if (sol_size != -1 && sol_size < best_sol_size) {
                best_sol_size = sol_size;
                bestUB = (double)sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
            }
        }

        if (bestUB <= LBint) break; /* proved optimal */

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

        if (!updated || step_coef <= step_min) {
            /* Try one final heuristic refresh before stop */
            greedy_from_reduced_costs(
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

            if (sol_size != -1 && sol_size < best_sol_size) {
                best_sol_size = sol_size;
                memcpy(solution, sol_tmp, (size_t)sol_size * sizeof(int));
            }
            break;
        }
    }

    *solmin = best_sol_size;

    free(t);
    free(rc);
    free(sol_tmp);
    free_adjacency(
        rowsCovered,
        rowsCoveredCount,
        cols,
        colsCovering,
        colsCoveringCount,
        rows
    );
}
