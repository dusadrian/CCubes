/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_input.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t hash;
    int row;
    unsigned char kind;
    unsigned char used;
} RowSlot;

enum {
    ROW_KIND_ON = 1,
    ROW_KIND_OFF = 2
};

static uint64_t row_hash(const int *row, int ninputs) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int input = 0; input < ninputs; ++input) {
        hash ^= (uint64_t)(unsigned int)row[input];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool same_row(
    const int *a,
    const int *b,
    int ninputs
) {
    return memcmp(a, b, (size_t)ninputs * sizeof(int)) == 0;
}

static bool next_table_size(size_t rows, size_t *table_size) {
    if (!table_size || rows > SIZE_MAX / 2u) return false;
    size_t needed = rows > 0 ? rows * 2u : 2u;
    size_t size = 2u;
    while (size < needed) {
        if (size > SIZE_MAX / 2u) return false;
        size *= 2u;
    }
    *table_size = size;
    return true;
}

static const int *slot_row(
    const ProgressiveNormalizedInput *normalized,
    const RowSlot *slot,
    int ninputs
) {
    const int *rows = slot->kind == ROW_KIND_ON
        ? normalized->pi.ON_set
        : normalized->pi.OFF_set;
    return &rows[(size_t)slot->row * (size_t)ninputs];
}

static ProgressiveInputStatus insert_row(
    ProgressiveNormalizedInput *normalized,
    RowSlot *slots,
    size_t table_size,
    const int *row,
    int ninputs,
    unsigned char kind
) {
    uint64_t hash = row_hash(row, ninputs);
    size_t mask = table_size - 1u;
    size_t position = (size_t)hash & mask;

    while (slots[position].used) {
        RowSlot *slot = &slots[position];
        if (
            slot->hash == hash &&
            same_row(
                slot_row(normalized, slot, ninputs),
                row,
                ninputs
            )
        ) {
            if (slot->kind != kind) {
                normalized->conflicts++;
                return PROGRESSIVE_INPUT_INVALID;
            }
            if (kind == ROW_KIND_ON) {
                normalized->on_duplicates++;
            } else {
                normalized->off_duplicates++;
            }
            return PROGRESSIVE_INPUT_OK;
        }
        position = (position + 1u) & mask;
    }

    int target_row;
    int *target;
    if (kind == ROW_KIND_ON) {
        target_row = normalized->pi.ON_minterms++;
        target = &normalized->pi.ON_set[
            (size_t)target_row * (size_t)ninputs
        ];
    } else {
        target_row = normalized->pi.OFF_minterms++;
        target = &normalized->pi.OFF_set[
            (size_t)target_row * (size_t)ninputs
        ];
    }
    memcpy(target, row, (size_t)ninputs * sizeof(int));
    slots[position] = (RowSlot){
        .hash = hash,
        .row = target_row,
        .kind = kind,
        .used = 1
    };
    return PROGRESSIVE_INPUT_OK;
}

ProgressiveInputStatus progressive_input_normalize(
    const PIstorage *source,
    int ninputs,
    ProgressiveNormalizedInput *normalized
) {
    if (normalized) memset(normalized, 0, sizeof(*normalized));
    if (
        !source ||
        !normalized ||
        ninputs <= 0 ||
        source->ON_minterms < 0 ||
        source->OFF_minterms < 0 ||
        (source->ON_minterms > 0 && !source->ON_set) ||
        (source->OFF_minterms > 0 && !source->OFF_set)
    ) {
        return PROGRESSIVE_INPUT_INVALID;
    }

    normalized->on_original = source->ON_minterms;
    normalized->off_original = source->OFF_minterms;

    size_t on_cells =
        (size_t)source->ON_minterms * (size_t)ninputs;
    size_t off_cells =
        (size_t)source->OFF_minterms * (size_t)ninputs;
    if (
        (source->ON_minterms > 0 &&
         on_cells / (size_t)ninputs != (size_t)source->ON_minterms) ||
        (source->OFF_minterms > 0 &&
         off_cells / (size_t)ninputs != (size_t)source->OFF_minterms)
    ) {
        return PROGRESSIVE_INPUT_OUT_OF_MEMORY;
    }

    for (size_t cell = 0; cell < on_cells; ++cell) {
        if (source->ON_set[cell] < 1 || source->ON_set[cell] > 2) {
            return PROGRESSIVE_INPUT_INVALID;
        }
    }
    for (size_t cell = 0; cell < off_cells; ++cell) {
        if (source->OFF_set[cell] < 1 || source->OFF_set[cell] > 2) {
            return PROGRESSIVE_INPUT_INVALID;
        }
    }

    normalized->pi.inputs = ninputs;
    normalized->pi.outputs = 1;
    normalized->pi.cov_bits = 64;

    if (on_cells > 0) {
        normalized->pi.ON_set =
            (int *)malloc(on_cells * sizeof(int));
    }
    if (off_cells > 0) {
        normalized->pi.OFF_set =
            (int *)malloc(off_cells * sizeof(int));
    }
    if (
        (on_cells > 0 && !normalized->pi.ON_set) ||
        (off_cells > 0 && !normalized->pi.OFF_set)
    ) {
        progressive_input_destroy(normalized);
        return PROGRESSIVE_INPUT_OUT_OF_MEMORY;
    }

    size_t total_rows =
        (size_t)source->ON_minterms + (size_t)source->OFF_minterms;
    size_t table_size;
    if (!next_table_size(total_rows, &table_size)) {
        progressive_input_destroy(normalized);
        return PROGRESSIVE_INPUT_OUT_OF_MEMORY;
    }
    RowSlot *slots = (RowSlot *)calloc(table_size, sizeof(RowSlot));
    if (!slots) {
        progressive_input_destroy(normalized);
        return PROGRESSIVE_INPUT_OUT_OF_MEMORY;
    }

    ProgressiveInputStatus status = PROGRESSIVE_INPUT_OK;
    for (
        int row = 0;
        row < source->ON_minterms && status == PROGRESSIVE_INPUT_OK;
        ++row
    ) {
        status = insert_row(
            normalized,
            slots,
            table_size,
            &source->ON_set[(size_t)row * (size_t)ninputs],
            ninputs,
            ROW_KIND_ON
        );
    }
    for (
        int row = 0;
        row < source->OFF_minterms && status == PROGRESSIVE_INPUT_OK;
        ++row
    ) {
        status = insert_row(
            normalized,
            slots,
            table_size,
            &source->OFF_set[(size_t)row * (size_t)ninputs],
            ninputs,
            ROW_KIND_OFF
        );
    }
    free(slots);

    if (status != PROGRESSIVE_INPUT_OK) {
        int conflicts = normalized->conflicts;
        int on_original = normalized->on_original;
        int off_original = normalized->off_original;
        progressive_input_destroy(normalized);
        normalized->conflicts = conflicts;
        normalized->on_original = on_original;
        normalized->off_original = off_original;
        return status;
    }

    normalized->pi.pichart_words =
        (normalized->pi.ON_minterms + 63) / 64;
    return PROGRESSIVE_INPUT_OK;
}

void progressive_input_destroy(ProgressiveNormalizedInput *normalized) {
    if (!normalized) return;
    free(normalized->pi.ON_set);
    free(normalized->pi.OFF_set);
    memset(normalized, 0, sizeof(*normalized));
}
