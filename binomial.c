/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "binomial.h"

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>

static uint64_t *pascal_values = NULL;
static int pascal_max_n = -1;

static size_t pascal_row_start(int n) {
    size_t row = (size_t)n;
    return row * (row + 1u) / 2u;
}

static uint64_t nchoosek_calculate(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;

#if defined(__GNUC__) && defined(__SIZEOF_INT128__)
    __uint128_t result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (__uint128_t)n - (__uint128_t)i;
        result /= (__uint128_t)i + 1u;
        if (result > (__uint128_t)UINT64_MAX) return 0;
    }
    return (uint64_t)result;
#else
    uint64_t result = 1;
    for (int i = 1; i <= k; ++i) {
        uint64_t numerator = (uint64_t)(n - k + i);
        uint64_t denominator = (uint64_t)i;

        /*
         * Reduce before multiplying so platforms without uint128_t do not
         * reject a representable coefficient solely because the temporary
         * product would overflow.
         */
        uint64_t a = numerator;
        uint64_t b = denominator;
        while (b != 0) {
            uint64_t remainder = a % b;
            a = b;
            b = remainder;
        }
        numerator /= a;
        denominator /= a;

        a = result;
        b = denominator;
        while (b != 0) {
            uint64_t remainder = a % b;
            a = b;
            b = remainder;
        }
        result /= a;
        denominator /= a;

        if (denominator != 1 || result > UINT64_MAX / numerator) return 0;
        result *= numerator;
    }
    return result;
#endif
}

bool nchoosek_prepare(int max_n) {
    if (max_n < 0) return false;
    if (pascal_values && max_n <= pascal_max_n) return true;

    size_t rows = (size_t)max_n + 1u;
    if (rows > SIZE_MAX / (rows + 1u)) return false;
    size_t entries = rows * (rows + 1u) / 2u;
    if (entries > SIZE_MAX / sizeof(uint64_t)) return false;

    uint64_t *values = (uint64_t *)calloc(entries, sizeof(uint64_t));
    if (!values) return false;

    for (int n = 0; n <= max_n; ++n) {
        size_t row = pascal_row_start(n);
        values[row] = 1;
        values[row + (size_t)n] = 1;

        if (n < 2) continue;
        size_t previous = pascal_row_start(n - 1);
        for (int k = 1; k < n; ++k) {
            uint64_t left = values[previous + (size_t)(k - 1)];
            uint64_t right = values[previous + (size_t)k];
            if (
                left == 0 ||
                right == 0 ||
                UINT64_MAX - left < right
            ) {
                values[row + (size_t)k] = 0;
            } else {
                values[row + (size_t)k] = left + right;
            }
        }
    }

    free(pascal_values);
    pascal_values = values;
    pascal_max_n = max_n;
    return true;
}

uint64_t nchoosek(int n, int k) {
    if (k < 0 || n < 0 || k > n) return 0;
    if (pascal_values && n <= pascal_max_n) {
        return pascal_values[pascal_row_start(n) + (size_t)k];
    }
    return nchoosek_calculate(n, k);
}
