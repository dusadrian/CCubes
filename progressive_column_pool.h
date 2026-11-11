/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_COLUMN_POOL_H
#define PROGRESSIVE_COLUMN_POOL_H

#include <stdint.h>

#include "bounded_mmcs.h"

typedef enum {
    PROGRESSIVE_COLUMN_POOL_OK = 0,
    PROGRESSIVE_COLUMN_POOL_INVALID,
    PROGRESSIVE_COLUMN_POOL_IO_ERROR,
    PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY,
    PROGRESSIVE_COLUMN_POOL_MISMATCH
} ProgressiveColumnPoolStatus;

typedef struct {
    int ninputs;
    int rows;
    int words;
    int count;
    int incumbent_cardinality;
    int *patterns;
    uint64_t *coverage;
    int *incumbent_solution;
} ProgressiveColumnPool;

uint64_t progressive_column_pool_fingerprint(
    const WeightedMMCSPreparedRows *prepared
);

ProgressiveColumnPoolStatus progressive_column_pool_save(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    int words,
    int count,
    const int *patterns,
    const uint64_t *coverage,
    int incumbent_cardinality,
    const int *incumbent_solution
);

/*
 * Convert a complete PI chart to the ordinary progressive pool format.
 * `prepared` defines the normalized ON/OFF row order used by replay, so each
 * column's coverage is recomputed rather than copied from the source chart.
 * The incumbent uses the transposed ninputs x cardinality layout.
 */
ProgressiveColumnPoolStatus progressive_column_pool_save_complete_chart(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    const PIstorage *source,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    int incumbent_cardinality,
    const int *incumbent_solution
);

ProgressiveColumnPoolStatus progressive_column_pool_load(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    int expected_words,
    ProgressiveColumnPool *pool
);

void progressive_column_pool_destroy(ProgressiveColumnPool *pool);

const char *progressive_column_pool_status_name(
    ProgressiveColumnPoolStatus status
);

#endif
