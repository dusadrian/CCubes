/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-Adrian
*/

#include "progressive_dual_calibration.h"

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int row;
    int covering_columns;
    double seed_weight;
} DualRowRank;

static uint64_t dual_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static int compare_dual_rows(const void *left, const void *right) {
    const DualRowRank *a = (const DualRowRank *)left;
    const DualRowRank *b = (const DualRowRank *)right;
    const bool a_seed = a->seed_weight > 0.0;
    const bool b_seed = b->seed_weight > 0.0;
    if (a_seed != b_seed) return a_seed ? -1 : 1;
    if (a_seed && a->seed_weight != b->seed_weight) {
        return a->seed_weight > b->seed_weight ? -1 : 1;
    }
    if (a->covering_columns != b->covering_columns) {
        return (a->covering_columns > b->covering_columns) -
            (a->covering_columns < b->covering_columns);
    }
    return (a->row > b->row) - (a->row < b->row);
}

static bool extra_covers(
    const uint64_t *coverage,
    int words,
    int column,
    int row
) {
    return (
        coverage[(size_t)column * (size_t)words + (size_t)(row / 64)] &
        (UINT64_C(1) << (row % 64))
    ) != 0;
}

static bool constraint_covers(
    const PIChartView *retained,
    const uint64_t *extra_coverage,
    int constraint,
    int row
) {
    if (constraint < retained->cols) {
        return chart_covers(retained, constraint, row);
    }
    return extra_covers(
        extra_coverage,
        retained->words,
        constraint - retained->cols,
        row
    );
}

static double maximum_constraint_weight(
    int constraints,
    const size_t *offsets,
    const int *incidences,
    const double *weights,
    int *maximum_constraint
) {
    double maximum = 0.0;
    int selected = -1;
    for (int constraint = 0; constraint < constraints; ++constraint) {
        double load = 0.0;
        for (
            size_t position = offsets[constraint];
            position < offsets[constraint + 1];
            ++position
        ) {
            load += weights[incidences[position]];
        }
        if (selected < 0 || load > maximum) {
            maximum = load;
            selected = constraint;
        }
    }
    if (maximum_constraint) *maximum_constraint = selected;
    return maximum;
}

static void normalize_weights(double *weights, int count) {
    double sum = 0.0;
    for (int row = 0; row < count; ++row) sum += weights[row];
    if (sum <= 0.0 || !isfinite(sum)) {
        const double uniform = 1.0 / (double)count;
        for (int row = 0; row < count; ++row) weights[row] = uniform;
        return;
    }
    for (int row = 0; row < count; ++row) weights[row] /= sum;
}

static bool progressive_dual_calibrate_internal(
    const PIChartView *retained,
    const double *seed_weights,
    int active_row_limit,
    int iterations,
    const uint64_t *extra_coverage,
    int extra_columns,
    double *row_weights,
    ProgressiveDualCalibrationStats *stats
) {
    if (stats) memset(stats, 0, sizeof(*stats));
    if (
        !retained ||
        !retained->bits ||
        retained->rows <= 0 ||
        retained->cols <= 0 ||
        retained->words != (retained->rows + 63) / 64 ||
        retained->cov_bits != 64 ||
        !seed_weights ||
        active_row_limit <= 0 ||
        iterations <= 0 ||
        extra_columns < 0 ||
        (extra_columns > 0 && !extra_coverage) ||
        !row_weights ||
        !stats
    ) {
        return false;
    }
    const uint64_t started = dual_monotonic_nanoseconds();
    const int active_count = active_row_limit < retained->rows
        ? active_row_limit
        : retained->rows;
    const int constraint_count = retained->cols + extra_columns;

    int *covering_counts = (int *)calloc(
        (size_t)retained->rows,
        sizeof(int)
    );
    DualRowRank *rank = (DualRowRank *)malloc(
        (size_t)retained->rows * sizeof(DualRowRank)
    );
    int *active_rows = (int *)malloc((size_t)active_count * sizeof(int));
    size_t *offsets = (size_t *)calloc(
        (size_t)constraint_count + 1u,
        sizeof(size_t)
    );
    double *current = (double *)calloc(
        (size_t)active_count,
        sizeof(double)
    );
    double *best = (double *)calloc(
        (size_t)active_count,
        sizeof(double)
    );
    double *average = (double *)calloc(
        (size_t)active_count,
        sizeof(double)
    );
    if (
        !covering_counts || !rank || !active_rows || !offsets ||
        !current || !best || !average
    ) {
        free(covering_counts); free(rank); free(active_rows); free(offsets);
        free(current); free(best); free(average);
        return false;
    }

    for (int constraint = 0; constraint < constraint_count; ++constraint) {
        const uint64_t *column = constraint < retained->cols
            ? retained->bits +
                (size_t)constraint * (size_t)retained->words
            : extra_coverage +
                (size_t)(constraint - retained->cols) *
                (size_t)retained->words;
        for (int word = 0; word < retained->words; ++word) {
            uint64_t bits = column[word];
            while (bits != 0) {
                const int bit = __builtin_ctzll(bits);
                const int row = word * 64 + bit;
                if (row < retained->rows) covering_counts[row]++;
                bits &= bits - 1;
            }
        }
    }
    for (int row = 0; row < retained->rows; ++row) {
        if (!isfinite(seed_weights[row]) || seed_weights[row] < 0.0) {
            free(covering_counts); free(rank); free(active_rows); free(offsets);
            free(current); free(best); free(average);
            return false;
        }
        rank[row] = (DualRowRank){
            .row = row,
            .covering_columns = covering_counts[row],
            .seed_weight = seed_weights[row]
        };
    }
    qsort(
        rank,
        (size_t)retained->rows,
        sizeof(DualRowRank),
        compare_dual_rows
    );
    for (int active = 0; active < active_count; ++active) {
        active_rows[active] = rank[active].row;
        current[active] = seed_weights[active_rows[active]];
    }
    normalize_weights(current, active_count);

    size_t incidence_count = 0;
    for (int constraint = 0; constraint < constraint_count; ++constraint) {
        offsets[constraint] = incidence_count;
        for (int active = 0; active < active_count; ++active) {
            if (constraint_covers(
                retained,
                extra_coverage,
                constraint,
                active_rows[active]
            )) {
                incidence_count++;
            }
        }
    }
    offsets[constraint_count] = incidence_count;
    if (incidence_count > SIZE_MAX / sizeof(int)) {
        free(covering_counts); free(rank); free(active_rows); free(offsets);
        free(current); free(best); free(average);
        return false;
    }
    int *incidences = (int *)malloc(incidence_count * sizeof(int));
    if (incidence_count > 0 && !incidences) {
        free(covering_counts); free(rank); free(active_rows); free(offsets);
        free(current); free(best); free(average);
        return false;
    }
    size_t position = 0;
    for (int constraint = 0; constraint < constraint_count; ++constraint) {
        for (int active = 0; active < active_count; ++active) {
            if (constraint_covers(
                retained,
                extra_coverage,
                constraint,
                active_rows[active]
            )) {
                incidences[position++] = active;
            }
        }
    }

    double initial_maximum = maximum_constraint_weight(
        constraint_count,
        offsets,
        incidences,
        current,
        NULL
    );
    double initial_sum = 0.0;
    for (int active = 0; active < active_count; ++active) {
        initial_sum += current[active];
    }
    const double initial_bound = initial_maximum > 0.0
        ? initial_sum / initial_maximum
        : 0.0;
    {
        double best_maximum = initial_maximum;
        memcpy(best, current, (size_t)active_count * sizeof(double));

        int zero_rows = 0;
        for (int active = 0; active < active_count; ++active) {
            if (current[active] == 0.0) zero_rows++;
        }
        if (zero_rows > 0) {
            const double exploration = 0.05;
            for (int active = 0; active < active_count; ++active) {
                current[active] = current[active] > 0.0
                    ? current[active] * (1.0 - exploration)
                    : exploration / (double)zero_rows;
            }
            normalize_weights(current, active_count);
        }

        const double eta = sqrt(
            2.0 * log((double)(active_count > 1 ? active_count : 2)) /
            (double)iterations
        );
        const double covered_factor = exp(-eta);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            int heaviest = -1;
            const double maximum = maximum_constraint_weight(
                constraint_count,
                offsets,
                incidences,
                current,
                &heaviest
            );
            if (maximum < best_maximum) {
                best_maximum = maximum;
                memcpy(best, current, (size_t)active_count * sizeof(double));
            }
            for (int active = 0; active < active_count; ++active) {
                average[active] += current[active];
            }
            if (heaviest < 0) break;
            for (
                size_t item = offsets[heaviest];
                item < offsets[heaviest + 1];
                ++item
            ) {
                current[incidences[item]] *= covered_factor;
            }
            normalize_weights(current, active_count);
        }
        normalize_weights(average, active_count);
        const double average_maximum = maximum_constraint_weight(
            constraint_count,
            offsets,
            incidences,
            average,
            NULL
        );
        if (average_maximum < best_maximum) {
            best_maximum = average_maximum;
            memcpy(best, average, (size_t)active_count * sizeof(double));
        }

        memset(row_weights, 0, (size_t)retained->rows * sizeof(double));
        for (int active = 0; active < active_count; ++active) {
            row_weights[active_rows[active]] = best[active];
        }
        stats->active_rows = active_count;
        stats->retained_columns = retained->cols;
        stats->extra_columns = extra_columns;
        stats->iterations = iterations;
        stats->initial_restricted_bound = initial_bound;
        stats->calibrated_restricted_bound = best_maximum > 0.0
            ? 1.0 / best_maximum
            : 0.0;
        stats->maximum_column_weight = best_maximum;
        stats->elapsed_seconds =
            (double)(dual_monotonic_nanoseconds() - started) / 1e9;

        free(covering_counts); free(rank); free(active_rows); free(offsets);
        free(current); free(best); free(average); free(incidences);

        return true;
    }

}

bool progressive_dual_calibrate(
    const PIChartView *retained,
    const double *seed_weights,
    int active_row_limit,
    int iterations,
    const uint64_t *extra_coverage,
    int extra_columns,
    double *row_weights,
    ProgressiveDualCalibrationStats *stats
) {
    return progressive_dual_calibrate_internal(
        retained,
        seed_weights,
        active_row_limit,
        iterations,
        extra_coverage,
        extra_columns,
        row_weights,
        stats
    );
}
