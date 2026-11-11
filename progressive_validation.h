/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_VALIDATION_H
#define PROGRESSIVE_VALIDATION_H

#include <stdbool.h>

#include "bounded_mmcs.h"

/*
 * Validate a transposed ninputs x cardinality cover using prepared ON/OFF
 * columns. Solution values use the write_pla_file convention: zero is a
 * dash, one is input value zero, and two is input value one.
 */
bool progressive_solution_is_valid_prepared(
    const WeightedMMCSPreparedRows *prepared,
    int cardinality,
    const int *solution
);

bool progressive_row_major_cover_is_valid_columns(
    int ninputs,
    int on_rows,
    int off_rows,
    const uint64_t *on_columns,
    const uint64_t *off_columns,
    int cardinality,
    const int *patterns
);

/*
 * Validate one row-major cube and recompute its ON coverage. `coverage`
 * holds `(on_rows + 63) / 64` words.
 */
bool progressive_pattern_coverage_prepared(
    const WeightedMMCSPreparedRows *prepared,
    const int *pattern,
    uint64_t *coverage
);

#endif
