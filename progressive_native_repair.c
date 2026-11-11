/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_native_repair.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t repair_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

typedef struct {
    int incumbent_index;
    int private_rows;
    uint64_t tie_key;
} RepairDestroyRank;

static int compare_destroy_rank(const void *left, const void *right) {
    const RepairDestroyRank *a = (const RepairDestroyRank *)left;
    const RepairDestroyRank *b = (const RepairDestroyRank *)right;
    if (a->private_rows != b->private_rows) {
        return (a->private_rows > b->private_rows) -
            (a->private_rows < b->private_rows);
    }
    if (a->tie_key != b->tie_key) {
        return (a->tie_key > b->tie_key) -
            (a->tie_key < b->tie_key);
    }
    return (a->incumbent_index > b->incumbent_index) -
        (a->incumbent_index < b->incumbent_index);
}

static uint64_t repair_mix(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) *
        UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) *
        UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static int default_destroy_count(int incumbent_cardinality) {
    if (incumbent_cardinality <= 1) return 0;
    int destroy_count = incumbent_cardinality / 10;
    if (destroy_count < 8) destroy_count = 8;
    if (destroy_count > 32) destroy_count = 32;
    if (destroy_count >= incumbent_cardinality) {
        destroy_count = incumbent_cardinality - 1;
    }
    return destroy_count;
}

int progressive_native_repair_next_destroy_count(
    int incumbent_cardinality,
    int current_destroy_count
) {
    const int initial = default_destroy_count(incumbent_cardinality);
    if (initial <= 0) return 0;
    if (current_destroy_count <= 0) return initial;

    int maximum = incumbent_cardinality / 3;
    if (maximum < initial) maximum = initial;
    if (maximum > initial * 4) maximum = initial * 4;
    if (maximum >= incumbent_cardinality) {
        maximum = incumbent_cardinality - 1;
    }
    if (current_destroy_count >= maximum) return maximum;

    int increment = current_destroy_count / 4;
    if (increment < 4) increment = 4;
    int next = current_destroy_count + increment;
    return next < maximum ? next : maximum;
}

static bool column_covers(
    const ProgressiveNativeMaster *master,
    int column,
    int row
) {
    return (
        master->coverage[
            (size_t)column * (size_t)master->words +
            (size_t)(row / 64)
        ] >> (row % 64)
    ) & UINT64_C(1);
}

static int adjust_column_counts(
    const ProgressiveNativeMaster *master,
    int column,
    int delta,
    int *coverage_count
) {
    const uint64_t *coverage = master->coverage +
        (size_t)column * (size_t)master->words;
    int newly_covered = 0;
    for (int word = 0; word < master->words; ++word) {
        uint64_t bits = coverage[word];
        while (bits) {
            const int row = word * 64 + __builtin_ctzll(bits);
            if (row < master->rows) {
                newly_covered += coverage_count[row] == 0 && delta > 0;
                coverage_count[row] += delta;
            }
            bits &= bits - 1;
        }
    }
    return newly_covered;
}

static bool initialize_frozen_cover(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    unsigned char *selected,
    int *coverage_count,
    int *result,
    int *count,
    int *uncovered
) {
    const bool cached = plan->frozen_coverage_count &&
        plan->frozen_coverage_rows == master->rows;
    if (cached) {
        memcpy(
            coverage_count,
            plan->frozen_coverage_count,
            (size_t)master->rows * sizeof(*coverage_count)
        );
    }
    *count = 0;
    for (
        int cube = 0;
        cube < plan->incumbent_cardinality;
        ++cube
    ) {
        if (plan->destroyed[cube]) continue;
        int column = plan->incumbent_columns[cube];
        if (
            column < 0 ||
            column >= master->count
        ) {
            return false;
        }
        if (selected[column]) continue;
        selected[column] = 1;
        if (result) result[(*count)] = column;
        (*count)++;
        if (!cached) adjust_column_counts(master, column, 1, coverage_count);
    }
    *uncovered = 0;
    for (int row = 0; row < master->rows; ++row) {
        *uncovered += coverage_count[row] == 0 ? 1 : 0;
    }
    return true;
}

bool progressive_native_repair_write_subproblem(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    const char *path,
    int *residual_rows,
    int *residual_candidate_columns
) {
    if (residual_rows) *residual_rows = 0;
    if (residual_candidate_columns) {
        *residual_candidate_columns = 0;
    }
    if (
        !master ||
        master->rows <= 0 ||
        master->count <= 0 ||
        !plan ||
        plan->incumbent_cardinality <= 1 ||
        !plan->incumbent_columns ||
        !plan->destroyed ||
        !path ||
        !*path
    ) {
        return false;
    }

    unsigned char *selected = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    int *coverage_count = (int *)calloc(
        (size_t)master->rows,
        sizeof(int)
    );
    int *row_map = (int *)malloc(
        (size_t)master->rows * sizeof(int)
    );
    unsigned char *candidate = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    if (!selected || !coverage_count || !row_map || !candidate) {
        free(selected);
        free(coverage_count);
        free(row_map);
        free(candidate);
        return false;
    }

    int frozen_count = 0;
    int uncovered = 0;
    bool valid = initialize_frozen_cover(
        master,
        plan,
        selected,
        coverage_count,
        NULL,
        &frozen_count,
        &uncovered
    );
    int mapped_rows = 0;
    for (int row = 0; valid && row < master->rows; ++row) {
        if (coverage_count[row] == 0) {
            row_map[mapped_rows++] = row;
        }
    }

    int candidate_count = 0;
    for (int column = 0; valid && column < master->count; ++column) {
        if (
            selected[column] ||
            (
                plan->eligible_columns &&
                !plan->eligible_columns[column]
            )
        ) {
            continue;
        }
        for (int index = 0; index < mapped_rows; ++index) {
            if (column_covers(master, column, row_map[index])) {
                candidate[column] = 1;
                candidate_count++;
                break;
            }
        }
    }
    for (int index = 0; valid && index < mapped_rows; ++index) {
        bool coverable = false;
        for (int column = 0; column < master->count; ++column) {
            if (
                candidate[column] &&
                column_covers(master, column, row_map[index])
            ) {
                coverable = true;
                break;
            }
        }
        if (!coverable) valid = false;
    }

    FILE *file = valid ? fopen(path, "w") : NULL;
    if (!file) valid = false;
    if (valid) {
        valid =
            fprintf(
                file,
                "\\ CCubes progressive native repair residual\n"
                "\\ neighborhood %d incumbent %d frozen %d "
                "destroyed %d residual_rows %d candidates %d\n"
                "Minimize\n obj:",
                plan->neighborhood,
                plan->incumbent_cardinality,
                frozen_count,
                plan->destroyed_count,
                mapped_rows,
                candidate_count
            ) >= 0;
    }
    bool first = true;
    for (int column = 0; valid && column < master->count; ++column) {
        if (!candidate[column]) continue;
        valid = fprintf(
            file,
            "%s x%d",
            first ? "" : " +",
            column
        ) >= 0;
        first = false;
    }
    if (valid) valid = fprintf(file, "\nSubject To\n") >= 0;
    for (int index = 0; valid && index < mapped_rows; ++index) {
        valid = fprintf(file, " r%d:", index) >= 0;
        first = true;
        for (int column = 0; valid && column < master->count; ++column) {
            if (
                !candidate[column] ||
                !column_covers(master, column, row_map[index])
            ) {
                continue;
            }
            valid = fprintf(
                file,
                "%s x%d",
                first ? "" : " +",
                column
            ) >= 0;
            first = false;
        }
        if (valid) valid = fprintf(file, " >= 1\n") >= 0;
    }
    if (valid) valid = fprintf(file, "Binary\n") >= 0;
    for (int column = 0; valid && column < master->count; ++column) {
        if (candidate[column]) {
            valid = fprintf(file, " x%d\n", column) >= 0;
        }
    }
    if (valid) valid = fprintf(file, "End\n") >= 0;
    if (file && fclose(file) != 0) valid = false;

    if (valid) {
        if (residual_rows) *residual_rows = mapped_rows;
        if (residual_candidate_columns) {
            *residual_candidate_columns = candidate_count;
        }
    }
    free(selected);
    free(coverage_count);
    free(row_map);
    free(candidate);
    return valid;
}

static bool solve_residual_lagrangian(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    unsigned char *selected,
    int *coverage_count,
    int *result,
    int *count,
    int *uncovered,
    ProgressiveNativeRepairStats *stats,
    uint64_t deadline_nanoseconds
) {
    if (*uncovered <= 0) return true;
    if (stats) {
        stats->lagrangian_attempted = true;
        stats->residual_rows = *uncovered;
    }
    int *row_map = (int *)malloc(
        (size_t)(*uncovered) * sizeof(int)
    );
    if (!row_map) return false;
    int residual_row = 0;
    for (int row = 0; row < master->rows; ++row) {
        if (coverage_count[row] == 0) {
            row_map[residual_row++] = row;
        }
    }
    const int residual_words = (*uncovered + 63) / 64;
    int candidate_count = 0;
    for (int column = 0; column < master->count; ++column) {
        if (
            selected[column] ||
            (
                plan->eligible_columns &&
                !plan->eligible_columns[column]
            )
        ) {
            continue;
        }
        for (int index = 0; index < *uncovered; ++index) {
            if (column_covers(master, column, row_map[index])) {
                candidate_count++;
                break;
            }
        }
    }
    if (stats) {
        stats->residual_candidate_columns = candidate_count;
    }
    if (candidate_count <= 0) {
        free(row_map);
        return false;
    }
    uint64_t *coverage = (uint64_t *)calloc(
        (size_t)candidate_count *
        (size_t)residual_words,
        sizeof(uint64_t)
    );
    int *column_map = (int *)malloc(
        (size_t)candidate_count * sizeof(int)
    );
    double *column_weights = (double *)calloc(
        (size_t)candidate_count,
        sizeof(double)
    );
    int *repair_solution = (int *)malloc(
        (size_t)candidate_count * sizeof(int)
    );
    if (
        !coverage ||
        !column_map ||
        !column_weights ||
        !repair_solution
    ) {
        free(row_map);
        free(coverage);
        free(column_map);
        free(column_weights);
        free(repair_solution);
        return false;
    }
    int candidate = 0;
    for (int column = 0; column < master->count; ++column) {
        if (
            selected[column] ||
            (
                plan->eligible_columns &&
                !plan->eligible_columns[column]
            )
        ) {
            continue;
        }
        bool useful = false;
        for (int index = 0; index < *uncovered; ++index) {
            int row = row_map[index];
            if (!column_covers(master, column, row)) continue;
            coverage[
                (size_t)candidate * (size_t)residual_words +
                (size_t)(index / 64)
            ] |= UINT64_C(1) << (index % 64);
            column_weights[candidate] += plan->pricing_weights[row];
            useful = true;
        }
        if (useful) {
            column_map[candidate++] = column;
        }
    }
    const PIChartView chart = {
        .bits = coverage,
        .words = residual_words,
        .cov_bits = 64,
        .rows = *uncovered,
        .cols = candidate_count
    };
    int repair_cardinality = -1;
    solve_scp_lagrangian_until(
        &chart,
        column_weights,
        repair_solution,
        &repair_cardinality,
        0,
        NULL,
        0,
        false,
        deadline_nanoseconds
    );
    LagrangianStats solver_stats = *lagrangian_last_stats();
    if (solver_stats.stop_reason == LAGR_STOP_DEADLINE && stats) {
        stats->deadline_reached = true;
    }
    if (
        repair_cardinality > 0 &&
        plan->allow_strong_retry &&
        !lagrangian_last_run_proved_optimal() &&
        solver_stats.gap > 0 &&
        !(
            deadline_nanoseconds > 0 &&
            repair_monotonic_nanoseconds() >= deadline_nanoseconds
        )
    ) {
        int *strong_solution = (int *)malloc(
            (size_t)candidate_count * sizeof(int)
        );
        if (strong_solution) {
            int strong_cardinality = -1;
            solve_scp_lagrangian_until(
                &chart,
                column_weights,
                strong_solution,
                &strong_cardinality,
                2,
                repair_solution,
                repair_cardinality,
                true,
                deadline_nanoseconds
            );
            const LagrangianStats strong_stats =
                *lagrangian_last_stats();
            if (
                strong_stats.stop_reason == LAGR_STOP_DEADLINE &&
                stats
            ) {
                stats->deadline_reached = true;
            }
            if (
                strong_cardinality > 0 &&
                (
                    strong_cardinality < repair_cardinality ||
                    (
                        strong_cardinality == repair_cardinality &&
                        strong_stats.best_lb >= solver_stats.best_lb
                    )
                )
            ) {
                repair_cardinality = strong_cardinality;
                memcpy(
                    repair_solution,
                    strong_solution,
                    (size_t)strong_cardinality * sizeof(int)
                );
                solver_stats = strong_stats;
            }
            if (stats) stats->lagrangian_strong_retry = true;
            free(strong_solution);
        }
    }
    if (stats) {
        stats->lagrangian_cardinality = repair_cardinality;
        stats->lagrangian_lower_bound = solver_stats.best_lb;
        stats->lagrangian_gap = solver_stats.gap;
        stats->lagrangian_effort_level =
            solver_stats.effort_level;
        stats->lagrangian_stop_reason = solver_stats.stop_reason;
        stats->lagrangian_proved_optimal =
            solver_stats.gap == 0 &&
            solver_stats.stop_reason == LAGR_STOP_OPTIMAL;
    }
    const int count_before = *count;
    bool solved = repair_cardinality > 0;
    if (solved) {
        for (int index = 0; index < repair_cardinality; ++index) {
            int reduced_column = repair_solution[index];
            if (
                reduced_column < 0 ||
                reduced_column >= candidate_count
            ) {
                solved = false;
                break;
            }
            int column = column_map[reduced_column];
            if (selected[column]) continue;
            selected[column] = 1;
            result[(*count)++] = column;
            *uncovered -= adjust_column_counts(master, column, 1, coverage_count);
        }
        solved = solved && *uncovered == 0;
    }
    if (stats) {
        stats->lagrangian_columns_added = *count - count_before;
        stats->lagrangian_completed = solved;
    }
    free(row_map);
    free(coverage);
    free(column_map);
    free(column_weights);
    free(repair_solution);
    return solved;
}

bool progressive_native_repair_prepare(
    const ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    int incumbent_cardinality,
    const int *incumbent_solution,
    const double *multipliers,
    double uncovered_penalty,
    int requested_destroy_count,
    int neighborhood,
    ProgressiveNativeRepairPlan *plan
) {
    if (plan) memset(plan, 0, sizeof(*plan));
    if (
        !master ||
        master->rows <= 0 ||
        master->count <= 0 ||
        incumbent_cardinality <= 1 ||
        !incumbent_solution ||
        !multipliers ||
        uncovered_penalty < 0.0 ||
        neighborhood < 0 ||
        !plan
    ) {
        return false;
    }
    int *columns = (int *)malloc(
        (size_t)incumbent_cardinality * sizeof(int)
    );
    unsigned char *destroyed = (unsigned char *)calloc(
        (size_t)incumbent_cardinality,
        sizeof(unsigned char)
    );
    double *weights = (double *)malloc(
        (size_t)master->rows * sizeof(double)
    );
    int *coverage_count = (int *)calloc(
        (size_t)master->rows,
        sizeof(int)
    );
    RepairDestroyRank *rank = (RepairDestroyRank *)malloc(
        (size_t)incumbent_cardinality *
        sizeof(RepairDestroyRank)
    );
    if (
        !columns ||
        !destroyed ||
        !weights ||
        !coverage_count ||
        !rank ||
        !progressive_native_master_map_solution(
            master,
            prepared,
            incumbent_cardinality,
            incumbent_solution,
            columns
        )
    ) {
        free(columns);
        free(destroyed);
        free(weights);
        free(coverage_count);
        free(rank);
        return false;
    }

    for (int cube = 0; cube < incumbent_cardinality; ++cube) {
        int column = columns[cube];
        adjust_column_counts(master, column, 1, coverage_count);
    }
    for (int cube = 0; cube < incumbent_cardinality; ++cube) {
        int private_rows = 0;
        int column = columns[cube];
        const uint64_t *coverage = master->coverage +
            (size_t)column * (size_t)master->words;
        for (int word = 0; word < master->words; ++word) {
            uint64_t bits = coverage[word];
            while (bits) {
                const int row = word * 64 + __builtin_ctzll(bits);
                if (row < master->rows && coverage_count[row] == 1) {
                    private_rows++;
                }
                bits &= bits - 1;
            }
        }
        rank[cube] = (RepairDestroyRank){
            .incumbent_index = cube,
            .private_rows = private_rows,
            .tie_key = repair_mix(
                (uint64_t)(unsigned int)cube ^
                ((uint64_t)(unsigned int)neighborhood << 32)
            )
        };
    }
    qsort(
        rank,
        (size_t)incumbent_cardinality,
        sizeof(RepairDestroyRank),
        compare_destroy_rank
    );
    int destroy_count = requested_destroy_count > 0
        ? requested_destroy_count
        : default_destroy_count(incumbent_cardinality);
    if (destroy_count >= incumbent_cardinality) {
        destroy_count = incumbent_cardinality - 1;
    }
    /*
     * Vary the membership within the low-private fringe while retaining the
     * structural bias. This supplies deterministic LNS diversity without a
     * random-number generator or solver state.
     */
    int fringe = destroy_count * 3;
    if (fringe > incumbent_cardinality) fringe = incumbent_cardinality;
    for (int pick = 0; pick < destroy_count; ++pick) {
        const uint64_t position_key =
            (uint64_t)(unsigned int)pick * UINT64_C(2) +
            (uint64_t)(unsigned int)neighborhood *
                (uint64_t)(unsigned int)(destroy_count + 1);
        int position = (int)(
            position_key % (uint64_t)(unsigned int)fringe
        );
        while (destroyed[rank[position].incumbent_index]) {
            position = (position + 1) % fringe;
        }
        destroyed[rank[position].incumbent_index] = 1;
    }

    for (int cube = 0; cube < incumbent_cardinality; ++cube) {
        if (destroyed[cube]) {
            adjust_column_counts(master, columns[cube], -1, coverage_count);
            continue;
        }
        /* Equal-coverage incumbent geometries may map to the same ID.
         * Frozen selection retains each ID once, just as the solve does.
         */
        for (int previous = 0; previous < cube; ++previous) {
            if (!destroyed[previous] && columns[previous] == columns[cube]) {
                adjust_column_counts(master, columns[cube], -1, coverage_count);
                break;
            }
        }
    }
    int uncovered = 0;
    for (int row = 0; row < master->rows; ++row) {
        bool missing = coverage_count[row] == 0;
        uncovered += missing ? 1 : 0;
        weights[row] = multipliers[row] +
            (missing ? uncovered_penalty : 0.0);
    }

    plan->neighborhood = neighborhood;
    plan->incumbent_cardinality = incumbent_cardinality;
    plan->destroyed_count = destroy_count;
    plan->uncovered_rows = uncovered;
    plan->incumbent_columns = columns;
    plan->destroyed = destroyed;
    plan->pricing_weights = weights;
    plan->allow_strong_retry = true;
    plan->frozen_coverage_count = coverage_count;
    plan->frozen_coverage_rows = master->rows;
    free(rank);
    return true;
}

bool progressive_native_repair_solve(
    const ProgressiveNativeMaster *master,
    const ProgressiveNativeRepairPlan *plan,
    int **selected_columns,
    int *cardinality,
    ProgressiveNativeRepairStats *stats,
    uint64_t deadline_nanoseconds
) {
    if (selected_columns) *selected_columns = NULL;
    if (cardinality) *cardinality = -1;
    if (stats) memset(stats, 0, sizeof(*stats));
    if (
        !master ||
        !plan ||
        plan->incumbent_cardinality <= 1 ||
        !plan->incumbent_columns ||
        !plan->destroyed ||
        !plan->pricing_weights ||
        !selected_columns ||
        !cardinality
    ) {
        return false;
    }
    unsigned char *selected = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    int *coverage_count = (int *)calloc(
        (size_t)master->rows,
        sizeof(int)
    );
    int *result = (int *)malloc(
        (size_t)master->count * sizeof(int)
    );
    if (!selected || !coverage_count || !result) {
        free(selected);
        free(coverage_count);
        free(result);
        return false;
    }
    int count = 0;
    int uncovered = 0;
    if (!initialize_frozen_cover(
        master,
        plan,
        selected,
        coverage_count,
        result,
        &count,
        &uncovered
    )) {
        free(selected);
        free(coverage_count);
        free(result);
        return false;
    }
    if (stats) {
        stats->frozen_columns = count;
    }

    if (stats) {
        stats->residual_rows = uncovered;
    }
    if (uncovered > 0) {
        /*
         * The destroyed residual is typically only tens or hundreds of rows.
         * Run the bundled bounded Lagrangian solver on that core before the
         * greedy fallback. This is the solver-free analogue of exact
         * subproblem repair: it considers all retained recombinations in the
         * neighborhood without invoking an external MIP solver.
         */
        (void)solve_residual_lagrangian(
            master,
            plan,
            selected,
            coverage_count,
            result,
            &count,
            &uncovered,
            stats,
            deadline_nanoseconds
        );
    }
    const int count_before_greedy = count;
    uint64_t *uncovered_bits = NULL;
    if (uncovered > 0) {
        uncovered_bits = (uint64_t *)calloc((size_t)master->words, sizeof(uint64_t));
        if (!uncovered_bits) {
            free(selected); free(coverage_count); free(result);
            return false;
        }
        for (int row = 0; row < master->rows; ++row) {
            if (coverage_count[row] == 0) {
                uncovered_bits[row / 64] |= UINT64_C(1) << (row % 64);
            }
        }
    }
    if (uncovered > 0 && stats) {
        stats->greedy_fallback_used = true;
    }
    while (uncovered > 0 && count < master->count) {
        if (
            deadline_nanoseconds > 0 &&
            repair_monotonic_nanoseconds() >= deadline_nanoseconds
        ) {
            if (stats) stats->deadline_reached = true;
            memcpy(
                result,
                plan->incumbent_columns,
                (size_t)plan->incumbent_cardinality * sizeof(int)
            );
            free(selected);
            free(coverage_count);
            free(uncovered_bits);
            *selected_columns = result;
            *cardinality = plan->incumbent_cardinality;
            if (stats) {
                stats->returned_cardinality = *cardinality;
            }
            return true;
        }
        int best_column = -1;
        int best_hits = 0;
        double best_weight = -DBL_MAX;
        for (int column = 0; column < master->count; ++column) {
            if (
                selected[column] ||
                (
                    plan->eligible_columns &&
                    !plan->eligible_columns[column]
                )
            ) {
                continue;
            }
            int hits = 0;
            double weight = 0.0;
            const uint64_t *coverage = &master->coverage[
                (size_t)column * (size_t)master->words
            ];
            for (int word = 0; word < master->words; ++word) {
                uint64_t bits = coverage[word] & uncovered_bits[word];
                while (bits != 0) {
                    int bit = __builtin_ctzll(bits);
                    bits &= bits - 1u;
                    int row = word * 64 + bit;
                    hits++;
                    weight += plan->pricing_weights[row];
                }
            }
            if (
                hits > best_hits ||
                (
                    hits == best_hits &&
                    weight > best_weight
                )
            ) {
                best_column = column;
                best_hits = hits;
                best_weight = weight;
            }
        }
        if (best_column < 0 || best_hits <= 0) {
            free(selected);
            free(coverage_count);
            free(uncovered_bits);
            free(result);
            return false;
        }
        selected[best_column] = 1;
        result[count++] = best_column;
        uncovered -= adjust_column_counts(master, best_column, 1, coverage_count);
        const uint64_t *coverage = master->coverage +
            (size_t)best_column * (size_t)master->words;
        for (int word = 0; word < master->words; ++word) {
            uncovered_bits[word] &= ~coverage[word];
        }
    }
    free(uncovered_bits);
    if (stats) {
        stats->greedy_columns_added =
            count - count_before_greedy;
    }

    int redundant_removed = 0;
    for (int index = count - 1; index >= 0; --index) {
        if (
            (index & 7) == 0 &&
            deadline_nanoseconds > 0 &&
            repair_monotonic_nanoseconds() >= deadline_nanoseconds
        ) {
            if (stats) stats->deadline_reached = true;
            break;
        }
        int column = result[index];
        bool removable = true;
        const uint64_t *coverage = master->coverage +
            (size_t)column * (size_t)master->words;
        for (int word = 0; removable && word < master->words; ++word) {
            uint64_t bits = coverage[word];
            while (bits) {
                const int row = word * 64 + __builtin_ctzll(bits);
                if (row < master->rows && coverage_count[row] <= 1) {
                    removable = false;
                    break;
                }
                bits &= bits - 1;
            }
        }
        if (!removable) continue;
        (void)adjust_column_counts(master, column, -1, coverage_count);
        selected[column] = 0;
        memmove(
            &result[index],
            &result[index + 1],
            (size_t)(count - index - 1) * sizeof(int)
        );
        count--;
        redundant_removed++;
    }

    free(selected);
    free(coverage_count);
    *selected_columns = result;
    *cardinality = count;
    if (stats) {
        stats->redundant_columns_removed =
            redundant_removed;
        stats->returned_cardinality = count;
    }
    return true;
}

void progressive_native_repair_destroy(
    ProgressiveNativeRepairPlan *plan
) {
    if (!plan) return;
    free(plan->incumbent_columns);
    free(plan->destroyed);
    free(plan->eligible_columns);
    free(plan->pricing_weights);
    free(plan->frozen_coverage_count);
    memset(plan, 0, sizeof(*plan));
}
