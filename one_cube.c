/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "one_cube.h"

#include <stdlib.h>
#include <string.h>

int find_one_cube(const PIstorage *pi, int ninputs, int *cube) {
    const size_t width = (size_t)ninputs;
    memcpy(cube, pi->ON_set, width * sizeof(*cube));
    int fixed = ninputs;
    for (int r = 1; r < pi->ON_minterms && fixed; ++r) {
        const int *row = pi->ON_set + (size_t)r * width;
        for (int i = 0; i < ninputs; ++i) {
            if (cube[i] && cube[i] != row[i]) {
                cube[i] = 0;
                --fixed;
            }
        }
    }
    if (!fixed) return 0;

    /* Each OFF row needs a mismatch even after expansion. */
    const size_t off_count = (size_t)pi->OFF_minterms;
    if (off_count > SIZE_MAX / sizeof(int)) return -1;
    int *mismatches = malloc(off_count * sizeof(*mismatches));
    if (!mismatches) return -1;
    for (size_t r = 0; r < off_count; ++r) {
        const int *row = pi->OFF_set + r * width;
        int count = 0;
        for (int i = 0; i < ninputs; ++i) {
            count += cube[i] != 0 && cube[i] != row[i];
        }
        if (!count) {
            free(mismatches);
            return 0;
        }
        mismatches[r] = count;
    }

    /* Drop each redundant literal. A retained literal has an OFF witness
       that stays blocked only by that literal, so one pass yields a prime. */
    for (int i = 0; i < ninputs; ++i) {
        if (!cube[i]) continue;
        size_t r = 0;
        for (; r < off_count; ++r) {
            if (mismatches[r] == 1 && pi->OFF_set[r * width + (size_t)i] != cube[i]) break;
        }
        if (r != off_count) continue;
        for (r = 0; r < off_count; ++r) {
            mismatches[r] -= pi->OFF_set[r * width + (size_t)i] != cube[i];
        }
        cube[i] = 0;
        --fixed;
    }
    free(mismatches);
    return fixed;
}
