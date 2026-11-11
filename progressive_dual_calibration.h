/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-Adrian
*/

#ifndef PROGRESSIVE_DUAL_CALIBRATION_H
#define PROGRESSIVE_DUAL_CALIBRATION_H

#include <stdbool.h>

#include "pichart_view.h"

typedef struct {
    int active_rows;
    int retained_columns;
    int extra_columns;
    int iterations;
    double initial_restricted_bound;
    double calibrated_restricted_bound;
    double maximum_column_weight;
    double elapsed_seconds;
} ProgressiveDualCalibrationStats;

/*
 * Improve a sparse row distribution against the retained columns and any
 * globally priced witness columns. The default minimax mirror update is less
 * aggressive than the restricted-LP optimizer below and has produced stronger
 * complete-cube certificates empirically. Every returned vector is
 * nonnegative; exact global pricing still supplies the global certificate.
 */
bool progressive_dual_calibrate(
    const PIChartView *retained,
    const double *seed_weights,
    int active_row_limit,
    int iterations,
    const uint64_t *extra_coverage,
    int extra_columns,
    double *row_weights,
    ProgressiveDualCalibrationStats *stats
);

/*
 * Stronger restricted-master LP optimizer used for controlled experiments.
 * It commonly produces a larger retained-master dual than
 * progressive_dual_calibrate(), but still requires sufficiently strong global
 * separation before its vector is useful as a complete-cube certificate.
 */

#endif
