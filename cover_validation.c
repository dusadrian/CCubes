/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details)
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "cover_validation.h"

#include <stddef.h>

bool cover_is_feasible(
    const PIChartView *chart,
    const int solution[],
    int solution_size
) {
    if (
        !chart ||
        !chart->bits ||
        chart->cols <= 0 ||
        chart->rows <= 0 ||
        chart->cov_bits <= 0 ||
        !solution ||
        solution_size <= 0 ||
        solution_size > chart->cols
    ) {
        return false;
    }

    for (int i = 0; i < solution_size; ++i) {
        int col = solution[i];
        if (col < 0 || col >= chart->cols) return false;

        for (int j = 0; j < i; ++j) {
            if (solution[j] == col) return false;
        }
    }

    for (int row = 0; row < chart->rows; ++row) {
        bool covered = false;
        for (int i = 0; i < solution_size && !covered; ++i) {
            covered = chart_covers(chart, solution[i], row);
        }
        if (!covered) return false;
    }

    return true;
}
