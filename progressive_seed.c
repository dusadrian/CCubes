/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_seed.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "progressive_validation.h"

static uint64_t seed_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint64_t seed_low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static bool seed_words_empty(const uint64_t *bits, int words) {
    for (int word = 0; word < words; ++word) {
        if (bits[word] != 0) return false;
    }
    return true;
}

static uint64_t seed_column_word(
    const uint64_t *columns,
    int column_words,
    int vertex,
    int word,
    bool complement,
    int row_count
) {
    uint64_t values = columns[
        (size_t)vertex * (size_t)column_words + (size_t)word
    ];
    if (complement) values = ~values;
    int remaining = row_count - word * 64;
    if (remaining < 64) values &= seed_low_bits_mask(remaining);
    return values;
}

static bool seed_pattern_covers_row(
    const int *pattern,
    const int *row,
    int ninputs
) {
    for (int input = 0; input < ninputs; ++input) {
        if (pattern[input] != 0 && pattern[input] != row[input]) {
            return false;
        }
    }
    return true;
}

bool progressive_seed_pattern_is_valid(
    const PIstorage *pi,
    int ninputs,
    const int *pattern
) {
    if (
        !pi ||
        !pattern ||
        ninputs <= 0 ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        !pi->ON_set ||
        !pi->OFF_set
    ) {
        return false;
    }
    bool covers_on = false;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        if (seed_pattern_covers_row(
            pattern,
            &pi->ON_set[(size_t)row * (size_t)ninputs],
            ninputs
        )) {
            covers_on = true;
            break;
        }
    }
    if (!covers_on) return false;
    for (int row = 0; row < pi->OFF_minterms; ++row) {
        if (seed_pattern_covers_row(
            pattern,
            &pi->OFF_set[(size_t)row * (size_t)ninputs],
            ninputs
        )) {
            return false;
        }
    }
    return true;
}

bool progressive_seed_pattern_is_prime(
    const PIstorage *pi,
    int ninputs,
    const int *pattern
) {
    if (!progressive_seed_pattern_is_valid(pi, ninputs, pattern)) {
        return false;
    }
    int relaxed[ninputs];
    memcpy(relaxed, pattern, (size_t)ninputs * sizeof(int));
    for (int input = 0; input < ninputs; ++input) {
        if (relaxed[input] == 0) continue;
        int value = relaxed[input];
        relaxed[input] = 0;
        bool still_valid = progressive_seed_pattern_is_valid(
            pi,
            ninputs,
            relaxed
        );
        relaxed[input] = value;
        if (still_valid) return false;
    }
    return true;
}

bool progressive_seed_cover_is_valid(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveSeedResult *result
) {
    if (
        !pi ||
        !result ||
        ninputs <= 0 ||
        result->cardinality <= 0 ||
        result->ninputs != ninputs ||
        !result->patterns
    ) {
        return false;
    }
    for (int row = 0; row < pi->ON_minterms; ++row) {
        const int *on = &pi->ON_set[(size_t)row * (size_t)ninputs];
        bool covered = false;
        for (int cube = 0; cube < result->cardinality; ++cube) {
            if (seed_pattern_covers_row(
                &result->patterns[(size_t)cube * (size_t)ninputs],
                on,
                ninputs
            )) {
                covered = true;
                break;
            }
        }
        if (!covered) return false;
    }
    for (int cube = 0; cube < result->cardinality; ++cube) {
        if (!progressive_seed_pattern_is_valid(
            pi,
            ninputs,
            &result->patterns[(size_t)cube * (size_t)ninputs]
        )) {
            return false;
        }
    }
    return true;
}

static bool seed_columns_have_match(
    const uint64_t *columns, int words, int rows,
    const int *pattern, const int *vertices, int support, int omitted
) {
    for (int word = 0; word < words; ++word) {
        uint64_t matching = seed_low_bits_mask(rows - word * 64);
        for (int literal = 0; literal < support; ++literal) {
            if (literal == omitted) continue;
            int vertex = vertices[literal];
            matching &= seed_column_word(
                columns, words, vertex, word, pattern[vertex] == 1, rows
            );
            if (matching == 0) break;
        }
        if (matching != 0) return true;
    }
    return false;
}

/* The builder already owns exact packed ON/OFF columns. Reuse them for the
 * same per-cube validity and every-literal primality checks; the exported
 * scalar validator remains an independent check for callers and tests. */
static bool seed_pattern_is_prime_columns(
    int ninputs, int on_rows, int off_rows, int on_words, int off_words,
    const uint64_t *on_columns, const uint64_t *off_columns,
    const int *pattern, int *vertices
) {
    int support = 0;
    for (int vertex = 0; vertex < ninputs; ++vertex) {
        if (pattern[vertex] != 0) vertices[support++] = vertex;
    }
    if (!seed_columns_have_match(
        on_columns, on_words, on_rows, pattern, vertices, support, -1
    ) || seed_columns_have_match(
        off_columns, off_words, off_rows, pattern, vertices, support, -1
    )) return false;
    for (int literal = 0; literal < support; ++literal) {
        /* Relaxation retains the already verified ON match. It is valid
         * exactly when it still has no OFF match. */
        if (!seed_columns_have_match(
            off_columns, off_words, off_rows, pattern, vertices, support, literal
        )) return false;
    }
    return true;
}

static bool seed_add_size(
    uint64_t *total,
    uint64_t count,
    uint64_t width
) {
    if (width > 0 && count > UINT64_MAX / width) return false;
    uint64_t bytes = count * width;
    if (*total > UINT64_MAX - bytes) return false;
    *total += bytes;
    return true;
}

static bool seed_next_combination(
    int *indices,
    int count,
    int ninputs
) {
    for (int index = count - 1; index >= 0; --index) {
        int limit = ninputs - count + index;
        if (indices[index] >= limit) continue;
        indices[index]++;
        for (int next = index + 1; next < count; ++next) {
            indices[next] = indices[next - 1] + 1;
        }
        return true;
    }
    return false;
}

static bool seed_find_bounded_separator(
    const int *anchor_row,
    int ninputs,
    int max_support,
    int on_rows,
    int off_rows,
    int on_words,
    int off_words,
    const uint64_t *on_columns,
    const uint64_t *off_columns,
    const uint64_t *uncovered,
    uint64_t *blocked,
    uint64_t *candidate_coverage,
    unsigned char *selected,
    int *indices,
    int *selected_count
) {
    if (selected_count) *selected_count = 0;
    if (
        !anchor_row ||
        ninputs <= 0 ||
        max_support <= 0 ||
        max_support > ninputs ||
        on_rows <= 0 ||
        off_rows <= 0 ||
        on_words <= 0 ||
        off_words <= 0 ||
        !on_columns ||
        !off_columns ||
        !uncovered ||
        !blocked ||
        !candidate_coverage ||
        !selected ||
        !indices ||
        !selected_count
    ) {
        return false;
    }

    for (int support = 1; support <= max_support; ++support) {
        for (int index = 0; index < support; ++index) {
            indices[index] = index;
        }
        bool found = false;
        uint64_t best_uncovered = 0;
        do {
            memset(blocked, 0, (size_t)off_words * sizeof(uint64_t));
            for (int literal = 0; literal < support; ++literal) {
                int vertex = indices[literal];
                bool complement = anchor_row[vertex] == 2;
                for (int word = 0; word < off_words; ++word) {
                    blocked[word] |= seed_column_word(
                        off_columns,
                        off_words,
                        vertex,
                        word,
                        complement,
                        off_rows
                    );
                }
            }
            bool valid = true;
            for (int word = 0; word < off_words; ++word) {
                int remaining_rows = off_rows - word * 64;
                uint64_t full = seed_low_bits_mask(remaining_rows);
                if ((blocked[word] & full) != full) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            for (int word = 0; word < on_words; ++word) {
                candidate_coverage[word] = seed_low_bits_mask(
                    on_rows - word * 64
                );
            }
            for (int literal = 0; literal < support; ++literal) {
                int vertex = indices[literal];
                bool complement = anchor_row[vertex] == 1;
                for (int word = 0; word < on_words; ++word) {
                    candidate_coverage[word] &= seed_column_word(
                        on_columns,
                        on_words,
                        vertex,
                        word,
                        complement,
                        on_rows
                    );
                }
            }
            uint64_t newly_covered = 0;
            for (int word = 0; word < on_words; ++word) {
                newly_covered += (uint64_t)__builtin_popcountll(
                    candidate_coverage[word] & uncovered[word]
                );
            }
            if (!found || newly_covered > best_uncovered) {
                memset(selected, 0, (size_t)ninputs);
                for (int literal = 0; literal < support; ++literal) {
                    selected[indices[literal]] = 1;
                }
                best_uncovered = newly_covered;
                found = true;
            }
        } while (seed_next_combination(indices, support, ninputs));
        if (found) {
            *selected_count = support;
            return true;
        }
    }
    return false;
}

static ProgressiveSeedStatus seed_build(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    int ninputs,
    bool enable_greedy,
    int max_support,
    uint64_t memory_budget_bytes,
    ProgressiveSeedResult *result
) {
    if (result) memset(result, 0, sizeof(*result));
    if (
        !pi ||
        !result ||
        ninputs <= 0 ||
        max_support < 0 ||
        max_support > ninputs ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        !pi->ON_set ||
        !pi->OFF_set
    ) {
        return PROGRESSIVE_SEED_INVALID;
    }

    const int on_rows = pi->ON_minterms;
    const int off_rows = pi->OFF_minterms;
    const int on_words = (on_rows + 63) / 64;
    const int off_words = (off_rows + 63) / 64;
    if (prepared && (
        prepared->ninputs != ninputs || prepared->on_rows != on_rows ||
        prepared->off_rows != off_rows || prepared->on_column_words != on_words ||
        prepared->off_column_words != off_words || !prepared->on_columns ||
        !prepared->off_columns
    )) return PROGRESSIVE_SEED_INVALID;
    uint64_t required_bytes = 0;
    if (
        !seed_add_size(
            &required_bytes,
            (uint64_t)on_rows * (uint64_t)ninputs,
            sizeof(int)
        ) ||
        !seed_add_size(
            &required_bytes,
            prepared ? 0 : (uint64_t)ninputs * (uint64_t)on_words,
            sizeof(uint64_t)
        ) ||
        !seed_add_size(
            &required_bytes,
            prepared ? 0 : (uint64_t)ninputs * (uint64_t)off_words,
            sizeof(uint64_t)
        ) ||
        !seed_add_size(&required_bytes, (uint64_t)on_words, sizeof(uint64_t)) ||
        !seed_add_size(&required_bytes, (uint64_t)on_words, sizeof(uint64_t)) ||
        !seed_add_size(
            &required_bytes,
            (uint64_t)off_words * 2u,
            sizeof(uint64_t)
        ) ||
        !seed_add_size(&required_bytes, (uint64_t)ninputs, sizeof(unsigned char)) ||
        !seed_add_size(&required_bytes, (uint64_t)ninputs, sizeof(int))
    ) {
        return PROGRESSIVE_SEED_OUT_OF_MEMORY;
    }
    if (
        memory_budget_bytes > 0 &&
        required_bytes > memory_budget_bytes
    ) {
        return PROGRESSIVE_SEED_OUT_OF_MEMORY;
    }

    int *patterns = (int *)malloc(
        (size_t)on_rows * (size_t)ninputs * sizeof(int)
    );
    uint64_t *owned_on_columns = prepared ? NULL : (uint64_t *)calloc(
        (size_t)ninputs * (size_t)on_words, sizeof(uint64_t)
    );
    uint64_t *owned_off_columns = prepared ? NULL : (uint64_t *)calloc(
        (size_t)ninputs * (size_t)off_words, sizeof(uint64_t)
    );
    const uint64_t *on_columns = prepared ? prepared->on_columns : owned_on_columns;
    const uint64_t *off_columns = prepared ? prepared->off_columns : owned_off_columns;
    uint64_t *uncovered = (uint64_t *)malloc(
        (size_t)on_words * sizeof(uint64_t)
    );
    uint64_t *candidate_coverage = (uint64_t *)malloc(
        (size_t)on_words * sizeof(uint64_t)
    );
    uint64_t *remaining = (uint64_t *)malloc(
        (size_t)off_words * sizeof(uint64_t)
    );
    uint64_t *blocked = (uint64_t *)malloc(
        (size_t)off_words * sizeof(uint64_t)
    );
    unsigned char *selected = (unsigned char *)calloc(
        (size_t)ninputs,
        sizeof(unsigned char)
    );
    int *selection_order = (int *)malloc(
        (size_t)ninputs * sizeof(int)
    );
    uint64_t *greedy_coverage = NULL;
    if (
        !patterns ||
        !on_columns ||
        !off_columns ||
        !uncovered ||
        !candidate_coverage ||
        !remaining ||
        !blocked ||
        !selected ||
        !selection_order
    ) {
        free(patterns);
        free(owned_on_columns);
        free(owned_off_columns);
        free(uncovered);
        free(candidate_coverage);
        free(remaining);
        free(blocked);
        free(selected);
        free(selection_order);
        return PROGRESSIVE_SEED_OUT_OF_MEMORY;
    }

    if (!prepared) {
        for (int row = 0; row < on_rows; ++row) {
            for (int input = 0; input < ninputs; ++input) {
                int value = pi->ON_set[
                    (size_t)row * (size_t)ninputs + (size_t)input
                ];
                if (value < 1 || value > 2) goto INVALID;
                if (value == 2) {
                    owned_on_columns[
                        (size_t)input * (size_t)on_words +
                        (size_t)(row / 64)
                    ] |= UINT64_C(1) << (row % 64);
                }
            }
        }
        for (int row = 0; row < off_rows; ++row) {
            for (int input = 0; input < ninputs; ++input) {
                int value = pi->OFF_set[
                    (size_t)row * (size_t)ninputs + (size_t)input
                ];
                if (value < 1 || value > 2) goto INVALID;
                if (value == 2) {
                    owned_off_columns[
                        (size_t)input * (size_t)off_words +
                        (size_t)(row / 64)
                    ] |= UINT64_C(1) << (row % 64);
                }
            }
        }

    }

    for (int word = 0; word < on_words; ++word) {
        int remaining_rows = on_rows - word * 64;
        uncovered[word] = seed_low_bits_mask(remaining_rows);
    }

    uint64_t started = seed_monotonic_nanoseconds();
    int greedy_capacity = 0;
    int cardinality = 0;
    int greedy_cubes = 0;
    int literal_count = 0;
    bool memory_exhausted = false;

    if (!enable_greedy) goto GREEDY_FINISHED;
    while (!seed_words_empty(uncovered, on_words)) {
        int anchor = -1;
        for (int word = 0; word < on_words && anchor < 0; ++word) {
            if (uncovered[word] != 0) {
                anchor = word * 64 + __builtin_ctzll(uncovered[word]);
            }
        }
        if (anchor < 0 || anchor >= on_rows) goto INVALID;
        const int *anchor_row = &pi->ON_set[
            (size_t)anchor * (size_t)ninputs
        ];

        for (int word = 0; word < off_words; ++word) {
            int remaining_rows = off_rows - word * 64;
            remaining[word] = seed_low_bits_mask(remaining_rows);
        }
        memset(selected, 0, (size_t)ninputs * sizeof(unsigned char));
        int selected_count = 0;

        while (!seed_words_empty(remaining, off_words)) {
            int best_vertex = -1;
            uint64_t best_hits = 0;
            for (int vertex = 0; vertex < ninputs; ++vertex) {
                if (selected[vertex]) continue;
                bool complement = anchor_row[vertex] == 2;
                uint64_t hits = 0;
                if (selected_count == 0 && prepared && prepared->off_one_counts) {
                    uint64_t ones = prepared->off_one_counts[vertex];
                    hits = complement ? (uint64_t)off_rows - ones : ones;
                } else {
                    for (int word = 0; word < off_words; ++word) {
                        hits += (uint64_t)__builtin_popcountll(
                            remaining[word] &
                            seed_column_word(
                                off_columns,
                                off_words,
                                vertex,
                                word,
                                complement,
                                off_rows
                            )
                        );
                    }
                }
                if (
                    hits > best_hits ||
                    (
                        hits == best_hits &&
                        hits > 0 &&
                        (best_vertex < 0 || vertex < best_vertex)
                    )
                ) {
                    best_hits = hits;
                    best_vertex = vertex;
                }
            }
            if (best_vertex < 0 || best_hits == 0) goto INVALID;
            selected[best_vertex] = 1;
            selection_order[selected_count++] = best_vertex;
            bool complement = anchor_row[best_vertex] == 2;
            for (int word = 0; word < off_words; ++word) {
                remaining[word] &= ~seed_column_word(
                    off_columns,
                    off_words,
                    best_vertex,
                    word,
                    complement,
                    off_rows
                );
            }
        }

        for (int index = selected_count - 1; index >= 0; --index) {
            int removed = selection_order[index];
            memset(blocked, 0, (size_t)off_words * sizeof(uint64_t));
            for (int other = 0; other < selected_count; ++other) {
                int vertex = selection_order[other];
                if (vertex == removed || !selected[vertex]) continue;
                bool complement = anchor_row[vertex] == 2;
                for (int word = 0; word < off_words; ++word) {
                    blocked[word] |= seed_column_word(
                        off_columns,
                        off_words,
                        vertex,
                        word,
                        complement,
                        off_rows
                    );
                }
            }
            bool all_blocked = true;
            for (int word = 0; word < off_words; ++word) {
                int remaining_rows = off_rows - word * 64;
                uint64_t full = seed_low_bits_mask(remaining_rows);
                if ((blocked[word] & full) != full) {
                    all_blocked = false;
                    break;
                }
            }
            if (all_blocked) selected[removed] = 0;
        }

        int bounded_support = 0;
        for (int vertex = 0; vertex < ninputs; ++vertex) {
            bounded_support += selected[vertex] != 0;
        }
        if (
            max_support > 0 &&
            bounded_support > max_support &&
            !seed_find_bounded_separator(
                anchor_row,
                ninputs,
                max_support,
                on_rows,
                off_rows,
                on_words,
                off_words,
                on_columns,
                off_columns,
                uncovered,
                blocked,
                candidate_coverage,
                selected,
                selection_order,
                &bounded_support
            )
        ) {
            goto INVALID;
        }

        if (greedy_cubes == greedy_capacity) {
            int next_capacity = greedy_capacity > 0
                ? greedy_capacity * 2
                : 16;
            if (next_capacity > on_rows) next_capacity = on_rows;
            uint64_t next_coverage_bytes =
                (uint64_t)next_capacity *
                (uint64_t)on_words *
                sizeof(uint64_t);
            uint64_t next_total = required_bytes;
            if (
                next_coverage_bytes > UINT64_MAX - next_total ||
                (
                    memory_budget_bytes > 0 &&
                    next_total + next_coverage_bytes >
                        memory_budget_bytes
                )
            ) {
                memory_exhausted = true;
                break;
            }
            uint64_t *grown = (uint64_t *)realloc(
                greedy_coverage,
                (size_t)next_capacity *
                    (size_t)on_words *
                    sizeof(uint64_t)
            );
            if (!grown) goto OUT_OF_MEMORY;
            greedy_coverage = grown;
            greedy_capacity = next_capacity;
        }

        int *pattern = &patterns[
            (size_t)cardinality * (size_t)ninputs
        ];
        memset(pattern, 0, (size_t)ninputs * sizeof(int));
        int cube_literals = 0;
        for (int vertex = 0; vertex < ninputs; ++vertex) {
            if (selected[vertex]) {
                pattern[vertex] = anchor_row[vertex];
                cube_literals++;
            }
        }
        if (!seed_pattern_is_prime_columns(
            ninputs, on_rows, off_rows, on_words, off_words,
            on_columns, off_columns, pattern, selection_order
        )) {
            goto INVALID;
        }
        if (max_support > 0 && cube_literals > max_support) {
            goto INVALID;
        }

        uint64_t *coverage = &greedy_coverage[
            (size_t)greedy_cubes * (size_t)on_words
        ];
        for (int word = 0; word < on_words; ++word) {
            int remaining_rows = on_rows - word * 64;
            coverage[word] = seed_low_bits_mask(remaining_rows);
        }
        for (int vertex = 0; vertex < ninputs; ++vertex) {
            if (!selected[vertex]) continue;
            bool complement = anchor_row[vertex] == 1;
            for (int word = 0; word < on_words; ++word) {
                coverage[word] &= seed_column_word(
                    on_columns,
                    on_words,
                    vertex,
                    word,
                    complement,
                    on_rows
                );
            }
        }
        for (int word = 0; word < on_words; ++word) {
            uncovered[word] &= ~coverage[word];
        }
        cardinality++;
        greedy_cubes++;
        literal_count += cube_literals;
    }

GREEDY_FINISHED:
    for (int word = 0; word < on_words; ++word) {
        uint64_t rows = uncovered[word];
        while (rows) {
            int bit = __builtin_ctzll(rows);
            rows &= rows - 1;
            int row = word * 64 + bit;
            if (row >= on_rows) goto INVALID;
            if (max_support > 0 && ninputs > max_support) goto INVALID;
            memcpy(
                &patterns[(size_t)cardinality * (size_t)ninputs],
                &pi->ON_set[(size_t)row * (size_t)ninputs],
                (size_t)ninputs * sizeof(int)
            );
            cardinality++;
            literal_count += ninputs;
        }
    }

    result->cardinality = cardinality;
    result->greedy_cubes = greedy_cubes;
    result->fallback_cubes = cardinality - greedy_cubes;
    result->literal_count = literal_count;
    result->ninputs = ninputs;
    result->coverage_words = on_words;
    result->greedy_complete = result->fallback_cubes == 0;
    result->memory_exhausted = memory_exhausted;
    result->elapsed_nanoseconds =
        seed_monotonic_nanoseconds() - started;
    result->allocated_bytes =
        required_bytes +
        (uint64_t)greedy_capacity *
            (uint64_t)on_words *
            sizeof(uint64_t);
    result->patterns = patterns;
    result->greedy_coverage = greedy_coverage;
    uint64_t validation_started = seed_monotonic_nanoseconds();
    bool cover_valid = progressive_row_major_cover_is_valid_columns(
        ninputs,
        on_rows,
        off_rows,
        on_columns,
        off_columns,
        result->cardinality,
        result->patterns
    );
    result->validation_elapsed_nanoseconds =
        seed_monotonic_nanoseconds() - validation_started;
    free(owned_on_columns);
    free(owned_off_columns);
    free(uncovered);
    free(candidate_coverage);
    free(remaining);
    free(blocked);
    free(selected);
    free(selection_order);
    if (!cover_valid) {
        progressive_seed_result_destroy(result);
        return PROGRESSIVE_SEED_INVALID;
    }
    return PROGRESSIVE_SEED_OK;

INVALID:
    free(patterns);
    free(owned_on_columns);
    free(owned_off_columns);
    free(uncovered);
    free(candidate_coverage);
    free(remaining);
    free(blocked);
    free(selected);
    free(selection_order);
    free(greedy_coverage);
    return PROGRESSIVE_SEED_INVALID;

OUT_OF_MEMORY:
    free(patterns);
    free(owned_on_columns);
    free(owned_off_columns);
    free(uncovered);
    free(candidate_coverage);
    free(remaining);
    free(blocked);
    free(selected);
    free(selection_order);
    free(greedy_coverage);
    return PROGRESSIVE_SEED_OUT_OF_MEMORY;
}

ProgressiveSeedStatus progressive_seed_build(
    const PIstorage *pi, int ninputs, bool enable_greedy, int max_support,
    uint64_t memory_budget_bytes, ProgressiveSeedResult *result
) {
    return seed_build(pi, NULL, ninputs, enable_greedy, max_support,
                      memory_budget_bytes, result);
}

ProgressiveSeedStatus progressive_seed_build_prepared(
    const PIstorage *pi, const WeightedMMCSPreparedRows *prepared,
    int ninputs, bool enable_greedy, int max_support,
    uint64_t memory_budget_bytes, ProgressiveSeedResult *result
) {
    if (!prepared) {
        if (result) memset(result, 0, sizeof(*result));
        return PROGRESSIVE_SEED_INVALID;
    }
    return seed_build(pi, prepared, ninputs, enable_greedy, max_support,
                      memory_budget_bytes, result);
}

void progressive_seed_result_destroy(ProgressiveSeedResult *result) {
    if (!result) return;
    free(result->patterns);
    free(result->greedy_coverage);
    memset(result, 0, sizeof(*result));
}
