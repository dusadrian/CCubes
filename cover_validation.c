/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details)
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "cover_validation.h"

#include <stddef.h>

bool cover_is_feasible(
    const int pichart[],
    int cols,
    int rows,
    const int solution[],
    int solution_size
) {
    if (
        !pichart ||
        cols <= 0 ||
        rows <= 0 ||
        !solution ||
        solution_size <= 0 ||
        solution_size > cols
    ) {
        return false;
    }

    for (int i = 0; i < solution_size; ++i) {
        int col = solution[i];
        if (col < 0 || col >= cols) return false;

        for (int j = 0; j < i; ++j) {
            if (solution[j] == col) return false;
        }
    }

    for (int row = 0; row < rows; ++row) {
        bool covered = false;
        for (int i = 0; i < solution_size && !covered; ++i) {
            int col = solution[i];
            covered = pichart[(size_t)col * (size_t)rows + (size_t)row] != 0;
        }
        if (!covered) return false;
    }

    return true;
}
