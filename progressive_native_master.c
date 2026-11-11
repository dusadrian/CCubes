/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_native_master.h"

#include <stddef.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "progressive_validation.h"

static int pattern_support(const int *pattern, int ninputs) {
    int support = 0;
    for (int input = 0; input < ninputs; ++input) {
        support += pattern[input] != 0;
    }
    return support;
}

static bool same_coverage(
    const ProgressiveNativeMaster *master,
    int column,
    const uint64_t *coverage
) {
    return memcmp(
        &master->coverage[
            (size_t)column * (size_t)master->words
        ],
        coverage,
        (size_t)master->words * sizeof(uint64_t)
    ) == 0;
}

typedef struct ProgressiveNativeMasterIndex {
    size_t buckets;
    int capacity;
    int count;
    int *coverage_heads;
    int *pattern_heads;
    int *coverage_next;
    int *pattern_next;
    uint64_t *coverage_hashes;
    uint64_t *pattern_hashes;
    double *literal_weights;
} ProgressiveNativeMasterIndex;

static uint64_t hash_coverage(const uint64_t *words, int count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < count; ++i) {
        hash ^= words[i];
        hash *= UINT64_C(1099511628211);
        hash ^= hash >> 32;
    }
    return hash;
}

static uint64_t hash_pattern(const int *pattern, int count) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (int i = 0; i < count; ++i) {
        hash ^= (unsigned int)pattern[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void free_index(ProgressiveNativeMasterIndex *index) {
    if (!index) return;
    free(index->coverage_heads);
    free(index->pattern_heads);
    free(index->coverage_next);
    free(index->pattern_next);
    free(index->coverage_hashes);
    free(index->pattern_hashes);
    free(index->literal_weights);
    free(index);
}

static void index_column(ProgressiveNativeMaster *master, int column) {
    ProgressiveNativeMasterIndex *index = master->index;
    const int *pattern = master->patterns + (size_t)column * master->ninputs;
    const uint64_t ch = hash_coverage(
        master->coverage + (size_t)column * master->words, master->words);
    const uint64_t ph = hash_pattern(pattern, master->ninputs);
    const size_t cb = ch & (index->buckets - 1);
    const size_t pb = ph & (index->buckets - 1);
    index->coverage_hashes[column] = ch;
    index->pattern_hashes[column] = ph;
    index->coverage_next[column] = index->coverage_heads[cb];
    index->pattern_next[column] = index->pattern_heads[pb];
    index->coverage_heads[cb] = column;
    index->pattern_heads[pb] = column;
    index->literal_weights[column] = -(double)pattern_support(pattern, master->ninputs);
    index->count++;
}

/* Indexes are optional accelerators: allocation failure preserves the exact
 * linear implementation rather than rejecting a feasible candidate. */
static bool ensure_index(ProgressiveNativeMaster *master) {
    if (master->index && master->index->count == master->count &&
        master->index->capacity > master->count) return true;
    const int capacity = master->capacity > master->count
        ? master->capacity : (master->count < INT_MAX / 2 ?
            (master->count > 32 ? master->count * 2 : 64) : INT_MAX);
    if (capacity <= master->count || (size_t)capacity > SIZE_MAX / (2 * sizeof(uint64_t))) {
        free_index(master->index); master->index = NULL;
        return false;
    }
    size_t buckets = 128;
    while (buckets < (size_t)capacity * 2) {
        if (buckets > SIZE_MAX / (2 * sizeof(int))) {
            free_index(master->index); master->index = NULL;
            return false;
        }
        buckets *= 2;
    }
    ProgressiveNativeMasterIndex *index = calloc(1, sizeof(*index));
    if (index) {
        index->capacity = capacity;
        index->buckets = buckets;
        index->coverage_heads = malloc(buckets * sizeof(int));
        index->pattern_heads = malloc(buckets * sizeof(int));
        index->coverage_next = malloc((size_t)capacity * sizeof(int));
        index->pattern_next = malloc((size_t)capacity * sizeof(int));
        index->coverage_hashes = malloc((size_t)capacity * sizeof(uint64_t));
        index->pattern_hashes = malloc((size_t)capacity * sizeof(uint64_t));
        index->literal_weights = malloc((size_t)capacity * sizeof(double));
    }
    if (!index || !index->coverage_heads || !index->pattern_heads ||
        !index->coverage_next || !index->pattern_next || !index->coverage_hashes ||
        !index->pattern_hashes || !index->literal_weights) {
        free_index(index);
        free_index(master->index); master->index = NULL;
        return false;
    }
    for (size_t b = 0; b < buckets; ++b) {
        index->coverage_heads[b] = index->pattern_heads[b] = -1;
    }
    free_index(master->index);
    master->index = index;
    for (int column = 0; column < master->count; ++column) index_column(master, column);
    return true;
}

static int find_coverage(const ProgressiveNativeMaster *master, const uint64_t *coverage) {
    const ProgressiveNativeMasterIndex *index = master->index;
    if (index && index->count == master->count) {
        const uint64_t hash = hash_coverage(coverage, master->words);
        for (int c = index->coverage_heads[hash & (index->buckets - 1)];
             c >= 0; c = index->coverage_next[c]) {
            if (index->coverage_hashes[c] == hash && same_coverage(master, c, coverage)) return c;
        }
        return -1;
    }
    for (int c = 0; c < master->count; ++c) {
        if (same_coverage(master, c, coverage)) return c;
    }
    return -1;
}

static void reindex_pattern(ProgressiveNativeMaster *master, int column) {
    ProgressiveNativeMasterIndex *index = master->index;
    if (!index) return;
    int *link = &index->pattern_heads[index->pattern_hashes[column] & (index->buckets - 1)];
    while (*link >= 0 && *link != column) link = &index->pattern_next[*link];
    if (*link == column) *link = index->pattern_next[column];
    const int *pattern = master->patterns + (size_t)column * master->ninputs;
    const uint64_t hash = hash_pattern(pattern, master->ninputs);
    const size_t bucket = hash & (index->buckets - 1);
    index->pattern_hashes[column] = hash;
    index->pattern_next[column] = index->pattern_heads[bucket];
    index->pattern_heads[bucket] = column;
    index->literal_weights[column] = -(double)pattern_support(pattern, master->ninputs);
}

static bool grow_master(ProgressiveNativeMaster *master) {
    if (master->count < master->capacity) return true;
    if (master->capacity > INT_MAX / 2) return false;
    int capacity = master->capacity > 0 ? master->capacity * 2 : 64;
    if (capacity <= master->capacity) return false;

    uint64_t *coverage = (uint64_t *)realloc(
        master->coverage,
        (size_t)capacity *
            (size_t)master->words *
            sizeof(uint64_t)
    );
    if (!coverage) return false;
    master->coverage = coverage;

    int *patterns = (int *)realloc(
        master->patterns,
        (size_t)capacity *
            (size_t)master->ninputs *
            sizeof(int)
    );
    if (!patterns) return false;
    master->patterns = patterns;
    master->capacity = capacity;
    return true;
}

bool progressive_native_master_initialize(
    ProgressiveNativeMaster *master,
    int rows,
    int words,
    int ninputs
) {
    if (!master || rows <= 0 || words <= 0 || ninputs <= 0) {
        return false;
    }
    memset(master, 0, sizeof(*master));
    master->rows = rows;
    master->words = words;
    master->ninputs = ninputs;
    master->max_support = 0;
    return true;
}

bool progressive_native_master_set_max_support(
    ProgressiveNativeMaster *master,
    int max_support
) {
    if (
        !master ||
        max_support < 0 ||
        max_support > master->ninputs ||
        master->count != 0
    ) {
        return false;
    }
    master->max_support = max_support;
    return true;
}

void progressive_native_master_destroy(
    ProgressiveNativeMaster *master
) {
    if (!master) return;
    free(master->coverage);
    free(master->patterns);
    free_index(master->index);
    memset(master, 0, sizeof(*master));
}

ProgressiveNativeMasterStatus progressive_native_master_insert(
    ProgressiveNativeMaster *master,
    const uint64_t *coverage,
    const int *pattern,
    bool *inserted
) {
    if (inserted) *inserted = false;
    if (
        !master ||
        master->rows <= 0 ||
        master->words <= 0 ||
        master->ninputs <= 0 ||
        !coverage ||
        !pattern ||
        !inserted
    ) {
        return PROGRESSIVE_NATIVE_MASTER_INVALID;
    }
    bool nonempty = false;
    for (int word = 0; word < master->words; ++word) {
        nonempty = nonempty || coverage[word] != 0;
    }
    if (!nonempty) return PROGRESSIVE_NATIVE_MASTER_INVALID;
    if (
        master->max_support > 0 &&
        pattern_support(pattern, master->ninputs) > master->max_support
    ) {
        return PROGRESSIVE_NATIVE_MASTER_OK;
    }

    (void)ensure_index(master);
    const int column = find_coverage(master, coverage);
    if (column >= 0) {
        int *retained = &master->patterns[
            (size_t)column * (size_t)master->ninputs
        ];
        if (
            pattern_support(pattern, master->ninputs) <
            pattern_support(retained, master->ninputs)
        ) {
            memcpy(retained, pattern, (size_t)master->ninputs * sizeof(int));
            reindex_pattern(master, column);
        }
        return PROGRESSIVE_NATIVE_MASTER_OK;
    }
    if (!grow_master(master)) {
        return PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;
    }
    memcpy(
        &master->coverage[
            (size_t)master->count * (size_t)master->words
        ],
        coverage,
        (size_t)master->words * sizeof(uint64_t)
    );
    memcpy(
        &master->patterns[
            (size_t)master->count * (size_t)master->ninputs
        ],
        pattern,
        (size_t)master->ninputs * sizeof(int)
    );
    if (master->index) index_column(master, master->count);
    master->count++;
    *inserted = true;
    return PROGRESSIVE_NATIVE_MASTER_OK;
}

ProgressiveNativeMasterStatus progressive_native_master_insert_seed(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveSeedResult *seed,
    int *inserted
) {
    if (inserted) *inserted = 0;
    if (
        !master ||
        !prepared ||
        !seed ||
        seed->cardinality <= 0 ||
        seed->ninputs != master->ninputs ||
        !seed->patterns ||
        !inserted
    ) {
        return PROGRESSIVE_NATIVE_MASTER_INVALID;
    }
    uint64_t *coverage = (uint64_t *)malloc(
        (size_t)master->words * sizeof(uint64_t)
    );
    if (!coverage) return PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;

    for (int cube = 0; cube < seed->cardinality; ++cube) {
        const int *pattern = &seed->patterns[
            (size_t)cube * (size_t)master->ninputs
        ];
        if (!progressive_pattern_coverage_prepared(
            prepared,
            pattern,
            coverage
        )) {
            free(coverage);
            return PROGRESSIVE_NATIVE_MASTER_INVALID;
        }
        bool added = false;
        ProgressiveNativeMasterStatus status =
            progressive_native_master_insert(
                master,
                coverage,
                pattern,
                &added
            );
        if (status != PROGRESSIVE_NATIVE_MASTER_OK) {
            free(coverage);
            return status;
        }
        *inserted += added ? 1 : 0;
    }
    free(coverage);
    return PROGRESSIVE_NATIVE_MASTER_OK;
}

ProgressiveNativeMasterStatus progressive_native_master_insert_pool(
    ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    const ProgressiveColumnPool *pool,
    int *inserted
) {
    if (inserted) *inserted = 0;
    if (
        !master ||
        !prepared ||
        !pool ||
        pool->ninputs != master->ninputs ||
        pool->rows != master->rows ||
        pool->words != master->words ||
        pool->count <= 0 ||
        !pool->patterns ||
        !pool->coverage ||
        !inserted
    ) {
        return PROGRESSIVE_NATIVE_MASTER_INVALID;
    }
    uint64_t *verified = (uint64_t *)malloc(
        (size_t)master->words * sizeof(uint64_t)
    );
    if (!verified) return PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;

    for (int column = 0; column < pool->count; ++column) {
        const int *pattern = &pool->patterns[
            (size_t)column * (size_t)master->ninputs
        ];
        const uint64_t *coverage = &pool->coverage[
            (size_t)column * (size_t)master->words
        ];
        if (
            !progressive_pattern_coverage_prepared(
                prepared,
                pattern,
                verified
            ) ||
            memcmp(
                verified,
                coverage,
                (size_t)master->words * sizeof(uint64_t)
            ) != 0
        ) {
            free(verified);
            return PROGRESSIVE_NATIVE_MASTER_INVALID;
        }
        bool added = false;
        ProgressiveNativeMasterStatus status =
            progressive_native_master_insert(
                master,
                coverage,
                pattern,
                &added
            );
        if (status != PROGRESSIVE_NATIVE_MASTER_OK) {
            free(verified);
            return status;
        }
        *inserted += added ? 1 : 0;
    }
    free(verified);
    return PROGRESSIVE_NATIVE_MASTER_OK;
}

double progressive_native_master_evaluate_bound(
    const ProgressiveNativeMaster *master,
    const double *multipliers
) {
    double bound = 0.0;
    for (int row = 0; row < master->rows; ++row) {
        if (!isfinite(multipliers[row]) || multipliers[row] < 0.0) return NAN;
        bound += multipliers[row];
    }
    for (int column = 0; column < master->count; ++column) {
        const uint64_t *coverage =
            master->coverage + (size_t)column * (size_t)master->words;
        double weight = 0.0;
        for (int word = 0; word < master->words; ++word) {
            for (uint64_t bits = coverage[word]; bits; bits &= bits - 1) {
                weight += multipliers[word * 64 + __builtin_ctzll(bits)];
            }
        }
        if (weight > 1.0) bound += 1.0 - weight;
    }
    return bound;
}

ProgressiveNativeMasterStatus
progressive_native_master_solve_with_polish_incumbent(
    const ProgressiveNativeMaster *master,
    int effort_level,
    const int *initial_solution,
    int initial_cardinality,
    const int *polish_incumbent,
    int polish_incumbent_cardinality,
    int *solution,
    int *cardinality,
    double *row_multipliers,
    const double *initial_multipliers
) {
    if (cardinality) *cardinality = -1;
    if (
        !master ||
        master->rows <= 0 ||
        master->words <= 0 ||
        master->count <= 0 ||
        !master->coverage ||
        !solution ||
        !cardinality ||
        !row_multipliers
    ) {
        return PROGRESSIVE_NATIVE_MASTER_INVALID;
    }
    const PIChartView chart = {
        .bits = master->coverage,
        .words = master->words,
        .cov_bits = 64,
        .rows = master->rows,
        .cols = master->count
    };
    double *allocated_weights = NULL;
    const double *literal_weights = master->index && master->index->count == master->count
        ? master->index->literal_weights : NULL;
    if (!literal_weights) {
        allocated_weights = (double *)malloc(
            (size_t)master->count * sizeof(double)
        );
        if (!allocated_weights) return PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;
        for (int column = 0; column < master->count; ++column) {
            allocated_weights[column] = -(double)pattern_support(
                &master->patterns[(size_t)column * (size_t)master->ninputs],
                master->ninputs
            );
        }
        literal_weights = allocated_weights;
    }
    /*
     * Negative support is a strict secondary preference only.  The
     * Lagrangian implementation leaves unit column costs unchanged when all
     * weights are non-positive, while its primal heuristics prefer the
     * larger weight among equal-cardinality alternatives.  Consequently the
     * primary cube-count objective and its lower bound are untouched.
     */
    solve_scp_lagrangian_with_multipliers_and_polish_incumbent(
        &chart,
        literal_weights,
        solution,
        cardinality,
        effort_level,
        initial_solution,
        initial_cardinality,
        false,
        row_multipliers,
        master->rows,
        polish_incumbent,
        polish_incumbent_cardinality,
        initial_multipliers
    );
    free(allocated_weights);
    return *cardinality >= 0
        ? PROGRESSIVE_NATIVE_MASTER_OK
        : PROGRESSIVE_NATIVE_MASTER_INFEASIBLE;
}

ProgressiveNativeMasterStatus progressive_native_master_solve(
    const ProgressiveNativeMaster *master,
    int effort_level,
    const int *initial_solution,
    int initial_cardinality,
    int *solution,
    int *cardinality,
    double *row_multipliers,
    const double *initial_multipliers
) {
    return progressive_native_master_solve_with_polish_incumbent(
        master,
        effort_level,
        initial_solution,
        initial_cardinality,
        NULL,
        0,
        solution,
        cardinality,
        row_multipliers,
        initial_multipliers
    );
}

bool progressive_native_master_find_pattern(
    const ProgressiveNativeMaster *master,
    const int *pattern,
    int *column
) {
    if (column) *column = -1;
    if (!master || !pattern || !column || !master->patterns) return false;
    const ProgressiveNativeMasterIndex *index = master->index;
    if (index && index->count == master->count) {
        const uint64_t hash = hash_pattern(pattern, master->ninputs);
        int match = -1;
        for (int c = index->pattern_heads[hash & (index->buckets - 1)];
             c >= 0; c = index->pattern_next[c]) {
            if (index->pattern_hashes[c] == hash && (match < 0 || c < match) &&
                !memcmp(master->patterns + (size_t)c * master->ninputs,
                        pattern, (size_t)master->ninputs * sizeof(int))) match = c;
        }
        *column = match;
        return match >= 0;
    }
    for (int candidate = 0; candidate < master->count; ++candidate) {
        if (memcmp(
            &master->patterns[
                (size_t)candidate * (size_t)master->ninputs
            ],
            pattern,
            (size_t)master->ninputs * sizeof(int)
        ) == 0) {
            *column = candidate;
            return true;
        }
    }
    return false;
}

bool progressive_native_master_map_solution(
    const ProgressiveNativeMaster *master,
    const WeightedMMCSPreparedRows *prepared,
    int cardinality,
    const int *solution,
    int *columns
) {
    if (
        !master ||
        cardinality <= 0 ||
        !solution ||
        !columns
    ) {
        return false;
    }
    int *pattern = (int *)malloc(
        (size_t)master->ninputs * sizeof(int)
    );
    uint64_t *coverage = prepared
        ? (uint64_t *)malloc(
            (size_t)master->words * sizeof(uint64_t)
        )
        : NULL;
    if (!pattern || (prepared && !coverage)) {
        free(pattern);
        free(coverage);
        return false;
    }
    for (int cube = 0; cube < cardinality; ++cube) {
        for (int input = 0; input < master->ninputs; ++input) {
            pattern[input] = solution[
                (size_t)input * (size_t)cardinality +
                (size_t)cube
            ];
        }
        if (progressive_native_master_find_pattern(
            master,
            pattern,
            &columns[cube]
        )) {
            continue;
        }
        /*
         * Insertion deduplicates by ON coverage and retains the lowest-support
         * representative, so the exact geometry a saved incumbent names may
         * have been displaced by an equal-coverage column.  Every member of a
         * coverage class is interchangeable inside a cover, so resolve through
         * coverage rather than failing the whole mapping.
         */
        int match = -1;
        if (
            prepared &&
            progressive_pattern_coverage_prepared(
                prepared,
                pattern,
                coverage
            )
        ) {
            match = find_coverage(master, coverage);
        }
        if (match < 0) {
            free(pattern);
            free(coverage);
            return false;
        }
        columns[cube] = match;
    }
    free(pattern);
    free(coverage);
    return true;
}
