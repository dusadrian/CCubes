/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_validation.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static uint64_t matching_column_word(
    const uint64_t *columns,
    int column_words,
    int input,
    int word,
    int value,
    int row_count
) {
    uint64_t bits = columns[
        (size_t)input * (size_t)column_words + (size_t)word
    ];
    if (value == 1) bits = ~bits;
    int remaining = row_count - word * 64;
    if (remaining < 64) bits &= low_bits_mask(remaining);
    return bits;
}

static bool cover_is_valid_columns(
    int ninputs,
    int on_rows,
    int off_rows,
    const uint64_t *on_columns,
    const uint64_t *off_columns,
    int cardinality,
    const int *patterns,
    bool transposed
) {
    int on_column_words = (on_rows + 63) / 64;
    int off_column_words = (off_rows + 63) / 64;
    if (
        ninputs <= 0 ||
        on_rows <= 0 ||
        off_rows <= 0 ||
        !on_columns ||
        !off_columns ||
        cardinality <= 0 ||
        !patterns
    ) {
        return false;
    }

    uint64_t *covered_on = (uint64_t *)calloc(
        (size_t)on_column_words,
        sizeof(uint64_t)
    );
    int *selected_inputs = (int *)malloc(
        (size_t)ninputs * sizeof(int)
    );
    unsigned char *selected_values = (unsigned char *)malloc(
        (size_t)ninputs * sizeof(unsigned char)
    );
    if (!covered_on || !selected_inputs || !selected_values) {
        free(covered_on);
        free(selected_inputs);
        free(selected_values);
        return false;
    }

    bool valid = true;
    for (int cube = 0; cube < cardinality && valid; ++cube) {
        int selected_count = 0;
        for (int input = 0; input < ninputs; ++input) {
            int value = transposed
                ? patterns[
                    (size_t)input * (size_t)cardinality +
                    (size_t)cube
                ]
                : patterns[
                    (size_t)cube * (size_t)ninputs + (size_t)input
                ];
            if (value < 0 || value > 2) {
                valid = false;
                break;
            }
            if (value != 0) {
                selected_inputs[selected_count] = input;
                selected_values[selected_count] = (unsigned char)value;
                selected_count++;
            }
        }
        if (!valid) break;

        for (
            int word = 0;
            word < off_column_words;
            ++word
        ) {
            int remaining = off_rows - word * 64;
            uint64_t matching = low_bits_mask(remaining);
            for (int selected = 0; selected < selected_count; ++selected) {
                matching &= matching_column_word(
                    off_columns,
                    off_column_words,
                    selected_inputs[selected],
                    word,
                    selected_values[selected],
                    off_rows
                );
                if (matching == 0) break;
            }
            if (matching != 0) {
                valid = false;
                break;
            }
        }
        if (!valid) break;

        for (
            int word = 0;
            word < on_column_words;
            ++word
        ) {
            int remaining = on_rows - word * 64;
            uint64_t matching = low_bits_mask(remaining);
            for (int selected = 0; selected < selected_count; ++selected) {
                matching &= matching_column_word(
                    on_columns,
                    on_column_words,
                    selected_inputs[selected],
                    word,
                    selected_values[selected],
                    on_rows
                );
                if (matching == 0) break;
            }
            covered_on[word] |= matching;
        }
    }

    if (valid) {
        for (
            int word = 0;
            word < on_column_words;
            ++word
        ) {
            int remaining = on_rows - word * 64;
            if (covered_on[word] != low_bits_mask(remaining)) {
                valid = false;
                break;
            }
        }
    }

    free(covered_on);
    free(selected_inputs);
    free(selected_values);
    return valid;
}

bool progressive_solution_is_valid_prepared(
    const WeightedMMCSPreparedRows *prepared,
    int cardinality,
    const int *solution
) {
    if (
        !prepared ||
        prepared->on_column_words != (prepared->on_rows + 63) / 64 ||
        prepared->off_column_words != (prepared->off_rows + 63) / 64
    ) {
        return false;
    }
    return cover_is_valid_columns(
        prepared->ninputs,
        prepared->on_rows,
        prepared->off_rows,
        prepared->on_columns,
        prepared->off_columns,
        cardinality,
        solution,
        true
    );
}

bool progressive_row_major_cover_is_valid_columns(
    int ninputs,
    int on_rows,
    int off_rows,
    const uint64_t *on_columns,
    const uint64_t *off_columns,
    int cardinality,
    const int *patterns
) {
    return cover_is_valid_columns(
        ninputs,
        on_rows,
        off_rows,
        on_columns,
        off_columns,
        cardinality,
        patterns,
        false
    );
}

bool progressive_pattern_coverage_prepared(
    const WeightedMMCSPreparedRows *prepared,
    const int *pattern,
    uint64_t *coverage
) {
    if (
        !prepared ||
        prepared->ninputs <= 0 ||
        prepared->on_rows <= 0 ||
        prepared->off_rows <= 0 ||
        !prepared->on_columns ||
        !prepared->off_columns ||
        !pattern ||
        !coverage
    ) {
        return false;
    }

    int *selected_inputs = (int *)malloc(
        (size_t)prepared->ninputs * sizeof(int)
    );
    unsigned char *selected_values = (unsigned char *)malloc(
        (size_t)prepared->ninputs * sizeof(unsigned char)
    );
    if (!selected_inputs || !selected_values) {
        free(selected_inputs);
        free(selected_values);
        return false;
    }

    int selected_count = 0;
    for (int input = 0; input < prepared->ninputs; ++input) {
        int value = pattern[input];
        if (value < 0 || value > 2) {
            free(selected_inputs);
            free(selected_values);
            return false;
        }
        if (value != 0) {
            selected_inputs[selected_count] = input;
            selected_values[selected_count] = (unsigned char)value;
            selected_count++;
        }
    }

    for (int word = 0; word < prepared->off_column_words; ++word) {
        int remaining = prepared->off_rows - word * 64;
        uint64_t matching = low_bits_mask(remaining);
        for (int selected = 0; selected < selected_count; ++selected) {
            matching &= matching_column_word(
                prepared->off_columns,
                prepared->off_column_words,
                selected_inputs[selected],
                word,
                selected_values[selected],
                prepared->off_rows
            );
            if (matching == 0) break;
        }
        if (matching != 0) {
            free(selected_inputs);
            free(selected_values);
            return false;
        }
    }

    bool nonempty = false;
    for (int word = 0; word < prepared->on_column_words; ++word) {
        int remaining = prepared->on_rows - word * 64;
        uint64_t matching = low_bits_mask(remaining);
        for (int selected = 0; selected < selected_count; ++selected) {
            matching &= matching_column_word(
                prepared->on_columns,
                prepared->on_column_words,
                selected_inputs[selected],
                word,
                selected_values[selected],
                prepared->on_rows
            );
            if (matching == 0) break;
        }
        coverage[word] = matching;
        nonempty = nonempty || matching != 0;
    }

    free(selected_inputs);
    free(selected_values);
    return nonempty;
}
