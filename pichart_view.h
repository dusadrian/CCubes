/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PICHART_VIEW_H
#define PICHART_VIEW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Read-only view over the bit-packed PI chart.
 *
 * The chart is stored column-major as `words` uint64_t per column, with row r
 * at bit (r % cov_bits) of word (r / cov_bits). Storing one bit per (PI, row)
 * instead of one int is what keeps large charts tractable: a 25k-row chart
 * costs ~3 KB per PI packed, against ~100 KB per PI dense.
 *
 * cov_bits is the coverage packing width (always <= 64, see
 * coverage_bits_per_word in utils.h) and travels with the view so consumers
 * never need to know the -b implicant packing width.
 */
typedef struct {
    const uint64_t *bits; /* cols * words */
    int words;            /* uint64_t words per column */
    int cov_bits;         /* coverage bits packed per word (1..64) */
    int rows;
    int cols;
} PIChartView;

static inline bool chart_covers(const PIChartView *chart, int col, int row) {
    return (
        chart->bits[
            (size_t)col * (size_t)chart->words +
            (size_t)(row / chart->cov_bits)
        ] >> (row % chart->cov_bits)
    ) & UINT64_C(1);
}

#endif
