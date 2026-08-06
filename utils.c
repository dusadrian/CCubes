/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "utils.h"
#include "checkpoint.h"
#include "lagrangian.h"
#include "prime_check.h"
#include "lock_stats.h"
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <math.h>

static uint64_t point_row_hash(const int *row, int ninputs) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int input = 0; input < ninputs; ++input) {
        hash ^= (uint64_t)(unsigned int)row[input];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/*
 * Certified stopping assumes a Boolean function, so its ON and OFF point sets
 * must be disjoint.  Hash the ON rows once and probe every OFF row; duplicates
 * within either side do not change the function and remain valid.
 */
static bool certified_point_sets_disjoint(
    const PIstorage *pi,
    int ninputs
) {
    size_t required = (size_t)pi->ON_minterms * 2u;
    size_t capacity = 8u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }

    int *slots = malloc(capacity * sizeof(*slots));
    if (!slots) return false;
    for (size_t slot = 0; slot < capacity; ++slot) slots[slot] = -1;

    const size_t mask = capacity - 1u;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        const int *point = &pi->ON_set[(size_t)row * (size_t)ninputs];
        size_t slot = (size_t)point_row_hash(point, ninputs) & mask;
        while (slots[slot] >= 0) {
            const int *existing = &pi->ON_set[
                (size_t)slots[slot] * (size_t)ninputs
            ];
            if (
                memcmp(
                    existing,
                    point,
                    (size_t)ninputs * sizeof(*point)
                ) == 0
            ) {
                break;
            }
            slot = (slot + 1u) & mask;
        }
        if (slots[slot] < 0) slots[slot] = row;
    }

    bool disjoint = true;
    for (int row = 0; row < pi->OFF_minterms && disjoint; ++row) {
        const int *point = &pi->OFF_set[(size_t)row * (size_t)ninputs];
        size_t slot = (size_t)point_row_hash(point, ninputs) & mask;
        while (slots[slot] >= 0) {
            const int *on_point = &pi->ON_set[
                (size_t)slots[slot] * (size_t)ninputs
            ];
            if (
                memcmp(
                    on_point,
                    point,
                    (size_t)ninputs * sizeof(*point)
                ) == 0
            ) {
                disjoint = false;
                break;
            }
            slot = (slot + 1u) & mask;
        }
    }

    free(slots);
    return disjoint;
}

bool certified_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
) {
    if (!PInfo || ninputs <= 0 || noutputs <= 0) return false;

    for (int o = 0; o < noutputs; ++o) {
        size_t on_cells = (size_t)PInfo[o].ON_minterms * (size_t)ninputs;
        size_t off_cells = (size_t)PInfo[o].OFF_minterms * (size_t)ninputs;

        for (size_t j = 0; j < on_cells; ++j) {
            if (PInfo[o].ON_set[j] < 1 || PInfo[o].ON_set[j] > 2) return false;
        }
        for (size_t j = 0; j < off_cells; ++j) {
            if (PInfo[o].OFF_set[j] < 1 || PInfo[o].OFF_set[j] > 2) return false;
        }
        if (!certified_point_sets_disjoint(&PInfo[o], ninputs)) return false;
    }

    return true;
}

bool heuristic_pattern_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
) {
    if (!PInfo || ninputs <= 0 || noutputs <= 0) return false;

    for (int o = 0; o < noutputs; ++o) {
        size_t on_cells = (size_t)PInfo[o].ON_minterms * (size_t)ninputs;
        size_t off_cells = (size_t)PInfo[o].OFF_minterms * (size_t)ninputs;

        for (size_t j = 0; j < on_cells; ++j) {
            if (PInfo[o].ON_set[j] < 0 || PInfo[o].ON_set[j] > 2) return false;
        }
        for (size_t j = 0; j < off_cells; ++j) {
            if (PInfo[o].OFF_set[j] < 0 || PInfo[o].OFF_set[j] > 2) return false;
        }
    }

    return true;
}

void destroy_output_locks(ccubes_mutex *locks, int noutputs) {
    if (!locks) return;
    for (int o = 0; o < noutputs; o++) {
        ccubes_mutex_destroy(&locks[o]);
    }
    free(locks);
}

bool env_flag_enabled(const char *name) {
    const char *value = getenv(name);
    if (!value || !*value) return false;

    return (
        strcmp(value, "0") != 0 &&
        strcmp(value, "false") != 0 &&
        strcmp(value, "FALSE") != 0 &&
        strcmp(value, "no") != 0 &&
        strcmp(value, "NO") != 0
    );
}

bool parse_int_strict(const char *text, int *value) {
    if (!text || !*text || !value) return false;

    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') return false;
    if (parsed < INT_MIN || parsed > INT_MAX) return false;

    *value = (int)parsed;
    return true;
}

bool parse_nonnegative_double(const char *text, double *value) {
    if (!text || !*text || !value) return false;

    char *end = NULL;
    errno = 0;
    double parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0') return false;
    if (!isfinite(parsed) || parsed < 0.0) return false;

    *value = parsed;
    return true;
}

bool parse_hybrid_effort_level(const char *text, int *level) {
    int parsed = 0;
    if (!parse_int_strict(text, &parsed)) return false;
    if (parsed < 0 || parsed > 2) return false;
    *level = parsed;
    return true;
}

void print_hybrid_stats(int output_index) {
    DBG_INFO_BLOCK {
        const LagrangianStats *stats = lagrangian_last_stats();
        if (!stats || stats->stop_reason == LAGR_STOP_NOT_RUN) return;

        fprintf(
            debug_out,
            "Hybrid output %d: rows=%d cols=%d UB=%d ",
            output_index + 1,
            stats->rows,
            stats->cols,
            stats->best_ub
        );

        if (stats->best_lb == INT_MIN) {
            fprintf(debug_out, "LB=- gap=- ");
        } else {
            fprintf(
                debug_out,
                "LB=%d gap=%d ",
                stats->best_lb,
                stats->gap
            );
        }

        fprintf(
            debug_out,
            "bestZLB=%.6f lastZLB=%.6f iterations=%d stop=%s%s "
            "warm_start_requested=%d warm_start_accepted=%d "
            "effort=%d certification_requested=%d iteration_limit=%d "
            "portfolio_limit=%d polish_nodes=%ld presolve_cols_removed=%d\n",
            stats->best_zlb,
            stats->last_zlb,
            stats->iterations,
            lagrangian_stop_reason_name(stats->stop_reason),
            stats->pool_mode ? " pool" : "",
            stats->warm_start_requested,
            stats->warm_start_accepted,
            stats->effort_level,
            stats->certification_requested,
            stats->iteration_limit,
            stats->portfolio_limit,
            stats->polish_node_limit,
            stats->presolve_cols_removed
        );
    }
}

double *build_cover_weights(
    const PIstorage *pi,
    int found_pi,
    int completed_level,
    int weight_mode
) {
    if (!pi || found_pi <= 0 || completed_level <= 0 || weight_mode <= 0) {
        return NULL;
    }

    double *weights = (double *)calloc((size_t)found_pi, sizeof(double));
    if (!weights) return NULL;

    int start = 0;
    for (int level = 1; level <= completed_level && start < found_pi; ++level) {
        int end = pi->nofpi[level - 1];
        if (end < start) end = start;
        if (end > found_pi) end = found_pi;

        double level_weight = scalbn(1.0, completed_level - level);
        if (!isfinite(level_weight)) level_weight = DBL_MAX / 4.0;

        for (int col = start; col < end; ++col) {
            weights[col] = level_weight;
            if (weight_mode == 2 && pi->shared) {
                weights[col] += pi->shared[col];
            }
        }
        start = end;
    }

    /* A malformed or legacy checkpoint must not leave active columns unweighted. */
    for (int col = start; col < found_pi; ++col) {
        weights[col] = 1.0;
        if (weight_mode == 2 && pi->shared) {
            weights[col] += pi->shared[col];
        }
    }

    return weights;
}

double *build_complete_cover_weights(
    const PIstorage *pi,
    int found_pi,
    int ninputs,
    int implicant_words,
    const int *word_index,
    const uint64_t *shifted_mask,
    int weight_mode
) {
    if (
        !pi ||
        !pi->implicants_pos ||
        found_pi <= 0 ||
        ninputs <= 0 ||
        implicant_words <= 0 ||
        !word_index ||
        !shifted_mask ||
        weight_mode <= 0
    ) {
        return NULL;
    }

    double *weights = (double *)calloc((size_t)found_pi, sizeof(double));
    if (!weights) return NULL;

    for (int col = 0; col < found_pi; ++col) {
        const uint64_t *position = &pi->implicants_pos[
            (size_t)col * (size_t)implicant_words
        ];
        int literal_count = 0;
        for (int input = 0; input < ninputs; ++input) {
            if (position[word_index[input]] & shifted_mask[input]) {
                ++literal_count;
            }
        }

        weights[col] = scalbn(1.0, ninputs - literal_count);
        if (!isfinite(weights[col])) weights[col] = DBL_MAX / 4.0;
        if (weight_mode == 2 && pi->shared) {
            weights[col] += pi->shared[col];
        }
    }

    return weights;
}

int automatic_pool_solution_limit(int found_pi) {
    /*
     * Pooling is an optional secondary objective.  A logarithmic discovery
     * budget lets wider charts expose more alternatives without reproducing
     * the square-root growth that made exact pool enumeration dominate both
     * time and memory.  Cross-output marginal-value filtering in
     * pool_selection.c subsequently removes alternatives that add no new
     * sharing opportunity.
     */
    if (found_pi <= 1) return 1;

    int limit = (int)ceil(log2((double)found_pi + 1.0));
    if (limit < CCUBES_POOL_MIN_CANDIDATES) {
        limit = CCUBES_POOL_MIN_CANDIDATES;
    }
    if (limit > CCUBES_POOL_STORAGE_CAPACITY) {
        limit = CCUBES_POOL_STORAGE_CAPACITY;
    }
    return limit;
}

typedef struct {
    PIstorage *pi;
    int implicant_words;
} PICanonicalSortContext;

static PICanonicalSortContext pi_canonical_sort_ctx;

static int cmp_u64_words(
    const uint64_t *a,
    const uint64_t *b,
    int n
) {
    for (int i = 0; i < n; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }

    return 0;
}

static int first_covered_row(
    PIstorage *pi,
    int idx
) {
    int rows = pi->ON_minterms;
    const PIChartView chart = pi_chart_view(pi);
    for (int r = 0; r < rows; ++r) {
        if (chart_covers(&chart, idx, r)) return r;
    }

    return rows;
}

static int cmp_pi_canonical(
    const void *a,
    const void *b
) {
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    PIstorage *pi = pi_canonical_sort_ctx.pi;
    int ipw = pi_canonical_sort_ctx.implicant_words;

    /* Preserve the strongest cross-output representative of equal coverage. */
    if (pi->shared[ia] != pi->shared[ib]) {
        return pi->shared[ib] - pi->shared[ia];
    }

    int cmp = cmp_u64_words(
        &pi->implicants_pos[(size_t)ia * (size_t)ipw],
        &pi->implicants_pos[(size_t)ib * (size_t)ipw],
        ipw
    );
    if (cmp != 0) return cmp;

    int first_a = first_covered_row(pi, ia);
    int first_b = first_covered_row(pi, ib);
    if (first_a != first_b) return first_a - first_b;

    cmp = cmp_u64_words(
        &pi->implicants_val[(size_t)ia * (size_t)ipw],
        &pi->implicants_val[(size_t)ib * (size_t)ipw],
        ipw
    );
    if (cmp != 0) return cmp;

    cmp = cmp_u64_words(
        &pi->pichart_pos[(size_t)ia * (size_t)pi->pichart_words],
        &pi->pichart_pos[(size_t)ib * (size_t)pi->pichart_words],
        pi->pichart_words
    );
    if (cmp != 0) return cmp;

    if (pi->covsum[ia] != pi->covsum[ib]) {
        return pi->covsum[ib] - pi->covsum[ia];
    }

    return 0;
}

static uint64_t coverage_words_hash(
    const uint64_t *coverage,
    int words
) {
    const uint64_t FNV_OFFSET = 1469598103934665603ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;

    for (int w = 0; w < words; ++w) {
        hash ^= coverage[w];
        hash *= FNV_PRIME;
    }

    return hash;
}

static uint64_t pi_coverage_hash(
    const PIstorage *pi,
    int idx
) {
    return coverage_words_hash(
        &pi->pichart_pos[(size_t)idx * (size_t)pi->pichart_words],
        pi->pichart_words
    );
}

static bool pi_coverage_equal(
    PIstorage *pi,
    int a,
    int b
) {
    return cmp_u64_words(
        &pi->pichart_pos[(size_t)a * (size_t)pi->pichart_words],
        &pi->pichart_pos[(size_t)b * (size_t)pi->pichart_words],
        pi->pichart_words
    ) == 0;
}

static int pi_coverage_lookup(
    PIstorage *pi,
    int *slots,
    size_t table_size,
    int idx
) {
    size_t mask = table_size - 1u;
    size_t pos = (size_t)(pi_coverage_hash(pi, idx) & (uint64_t)mask);

    while (slots[pos] >= 0) {
        if (pi_coverage_equal(pi, slots[pos], idx)) return slots[pos];
        pos = (pos + 1u) & mask;
    }

    return -1;
}

static void pi_coverage_insert(
    PIstorage *pi,
    int *slots,
    size_t table_size,
    int idx
) {
    size_t mask = table_size - 1u;
    size_t pos = (size_t)(pi_coverage_hash(pi, idx) & (uint64_t)mask);

    while (slots[pos] >= 0) {
        if (pi_coverage_equal(pi, slots[pos], idx)) return;
        pos = (pos + 1u) & mask;
    }

    slots[pos] = idx;
}

static uint64_t pi_geometry_hash(
    const PIstorage *pi,
    int implicant_words,
    int idx
) {
    const uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t hash = 1469598103934665603ULL;
    const uint64_t *positions = &pi->implicants_pos[
        (size_t)idx * (size_t)implicant_words
    ];
    const uint64_t *values = &pi->implicants_val[
        (size_t)idx * (size_t)implicant_words
    ];

    for (int word = 0; word < implicant_words; ++word) {
        hash ^= positions[word];
        hash *= FNV_PRIME;
        hash ^= values[word];
        hash *= FNV_PRIME;
    }
    return hash;
}

static bool pi_geometry_equal(
    const PIstorage *pi,
    int implicant_words,
    int a,
    int b
) {
    return cmp_u64_words(
        &pi->implicants_pos[(size_t)a * (size_t)implicant_words],
        &pi->implicants_pos[(size_t)b * (size_t)implicant_words],
        implicant_words
    ) == 0 && cmp_u64_words(
        &pi->implicants_val[(size_t)a * (size_t)implicant_words],
        &pi->implicants_val[(size_t)b * (size_t)implicant_words],
        implicant_words
    ) == 0;
}

static int pi_geometry_lookup(
    const PIstorage *pi,
    int implicant_words,
    int *slots,
    size_t table_size,
    int idx
) {
    size_t mask = table_size - 1u;
    size_t pos = (size_t)(
        pi_geometry_hash(pi, implicant_words, idx) & (uint64_t)mask
    );

    while (slots[pos] >= 0) {
        if (pi_geometry_equal(
            pi,
            implicant_words,
            slots[pos],
            idx
        )) {
            return slots[pos];
        }
        pos = (pos + 1u) & mask;
    }
    return -1;
}

static void pi_geometry_insert(
    const PIstorage *pi,
    int implicant_words,
    int *slots,
    size_t table_size,
    int idx
) {
    size_t mask = table_size - 1u;
    size_t pos = (size_t)(
        pi_geometry_hash(pi, implicant_words, idx) & (uint64_t)mask
    );

    while (slots[pos] >= 0) {
        if (pi_geometry_equal(
            pi,
            implicant_words,
            slots[pos],
            idx
        )) {
            return;
        }
        pos = (pos + 1u) & mask;
    }
    slots[pos] = idx;
}

static int coverage_index_lookup_words(
    const PICoverageIndex *index,
    const uint64_t *coverage
) {
    if (!index || !index->slots || index->table_size == 0) return -1;

    size_t mask = index->table_size - 1u;
    size_t pos = (size_t)(
        coverage_words_hash(coverage, index->words) & (uint64_t)mask
    );

    while (index->slots[pos] >= 0) {
        if (
            cmp_u64_words(
                &index->keys[(size_t)pos * (size_t)index->words],
                coverage,
                index->words
            ) == 0
        ) {
            return index->slots[pos];
        }
        pos = (pos + 1u) & mask;
    }

    return -1;
}

static bool coverage_index_resize(
    PICoverageIndex *index,
    size_t requested_size
) {
    size_t table_size = 16u;
    while (table_size < requested_size) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }

    int *slots = (int *)malloc(table_size * sizeof(int));
    if (!slots) return false;
    for (size_t i = 0; i < table_size; ++i) slots[i] = -1;
    uint64_t *keys = index->words > 0
        ? (uint64_t *)calloc(
            table_size * (size_t)index->words,
            sizeof(uint64_t)
        )
        : NULL;
    if (index->words > 0 && !keys) {
        free(slots);
        return false;
    }

    if (index->slots) {
        size_t mask = table_size - 1u;
        for (size_t i = 0; i < index->table_size; ++i) {
            if (index->slots[i] < 0) continue;

            const uint64_t *key = &index->keys[
                i * (size_t)index->words
            ];
            size_t pos = (size_t)(
                coverage_words_hash(key, index->words) & (uint64_t)mask
            );
            while (slots[pos] >= 0) pos = (pos + 1u) & mask;
            slots[pos] = 1;
            memcpy(
                &keys[pos * (size_t)index->words],
                key,
                (size_t)index->words * sizeof(uint64_t)
            );
        }
    }

    free(index->slots);
    free(index->keys);
    index->slots = slots;
    index->keys = keys;
    index->table_size = table_size;
    return true;
}

/*
 * Inserts a coverage key already known to be absent from the index (the
 * caller just performed the lookup itself, e.g. to decide redundancy, and
 * no intervening insert could have changed that under the held output
 * lock). Skips the redundant re-lookup that coverage_index_insert_words
 * would otherwise repeat.
 */
static bool coverage_index_insert_known_new(
    PICoverageIndex *index,
    const uint64_t *coverage
) {
    if (
        index->table_size == 0 ||
        (index->count + 1u) * 10u >= index->table_size * 7u
    ) {
        size_t requested = index->table_size > 0
            ? index->table_size * 2u
            : 16u;
        if (!coverage_index_resize(index, requested)) return false;
    }

    size_t mask = index->table_size - 1u;
    size_t pos = (size_t)(
        coverage_words_hash(coverage, index->words) & (uint64_t)mask
    );

    while (index->slots[pos] >= 0) pos = (pos + 1u) & mask;
    index->slots[pos] = 1;
    memcpy(
        &index->keys[pos * (size_t)index->words],
        coverage,
        (size_t)index->words * sizeof(uint64_t)
    );

    index->count++;
    return true;
}

static bool coverage_index_insert_words(
    PICoverageIndex *index,
    const uint64_t *coverage
) {
    if (coverage_index_lookup_words(index, coverage) >= 0) return true;
    return coverage_index_insert_known_new(index, coverage);
}

bool build_pi_coverage_indices(
    PICoverageIndex **indices,
    PIstorage *PInfo,
    int noutputs,
    const int *level_start,
    int implicant_words
) {
    if (
        !indices ||
        !PInfo ||
        noutputs < 0 ||
        !level_start ||
        implicant_words <= 0
    ) {
        return false;
    }

    *indices = (PICoverageIndex *)calloc(
        (size_t)noutputs,
        sizeof(PICoverageIndex)
    );
    if (!*indices) return false;

    for (int o = 0; o < noutputs; ++o) {
        PICoverageIndex *index = &(*indices)[o];
        index->words = PInfo[o].pichart_words;
        atomic_init(&index->subsumption_rejections, 0);

        int parent_records = level_start[o];
        if (parent_records < 0) parent_records = 0;
        if (parent_records > PInfo[o].foundPI) {
            parent_records = PInfo[o].foundPI;
        }

        if (!subsumption_index_build(
            &index->subsumption_index,
            PInfo[o].implicants_pos,
            PInfo[o].implicants_val,
            parent_records,
            implicant_words
        )) {
            destroy_pi_coverage_indices(*indices, noutputs);
            *indices = NULL;
            return false;
        }

        int records = level_start[o];
        if (records < 0) records = 0;
        if (records > PInfo[o].foundPI) records = PInfo[o].foundPI;

        size_t requested = records > 0 ? (size_t)records * 2u : 16u;
        if (!coverage_index_resize(index, requested)) {
            destroy_pi_coverage_indices(*indices, noutputs);
            *indices = NULL;
            return false;
        }

        for (int record = 0; record < records; ++record) {
            if (!coverage_index_insert_words(
                index,
                &PInfo[o].pichart_pos[
                    (size_t)record * (size_t)PInfo[o].pichart_words
                ]
            )) {
                destroy_pi_coverage_indices(*indices, noutputs);
                *indices = NULL;
                return false;
            }
        }
    }

    return true;
}

void destroy_pi_coverage_indices(
    PICoverageIndex *indices,
    int noutputs
) {
    if (!indices) return;
    for (int o = 0; o < noutputs; ++o) {
        free(indices[o].slots);
        free(indices[o].keys);
        subsumption_index_destroy(&indices[o].subsumption_index);
    }
    free(indices);
}

static void copy_pi_record(
    PIstorage *pi,
    int dst,
    int src,
    int implicant_words
) {
    if (dst == src) return;

    memmove(
        &pi->pichart_pos[(size_t)dst * (size_t)pi->pichart_words],
        &pi->pichart_pos[(size_t)src * (size_t)pi->pichart_words],
        (size_t)pi->pichart_words * sizeof(uint64_t)
    );

    memmove(
        &pi->implicants_pos[(size_t)dst * (size_t)implicant_words],
        &pi->implicants_pos[(size_t)src * (size_t)implicant_words],
        (size_t)implicant_words * sizeof(uint64_t)
    );

    memmove(
        &pi->implicants_val[(size_t)dst * (size_t)implicant_words],
        &pi->implicants_val[(size_t)src * (size_t)implicant_words],
        (size_t)implicant_words * sizeof(uint64_t)
    );

    pi->shared[dst] = pi->shared[src];
    pi->covsum[dst] = pi->covsum[src];
}

static int prune_duplicate_coverage_in_level(
    PIstorage *pi,
    int implicant_words,
    int level_start,
    bool preserve_shared_geometries
) {
    int found = pi->foundPI;
    size_t table_size = 1u;
    while (table_size < (size_t)found * 2u) {
        table_size <<= 1u;
    }

    int *coverage_slots = (int *)malloc(table_size * sizeof(int));
    int *geometry_slots = preserve_shared_geometries
        ? (int *)malloc(table_size * sizeof(int))
        : NULL;
    if (!coverage_slots || (preserve_shared_geometries && !geometry_slots)) {
        free(coverage_slots);
        free(geometry_slots);
        return 0;
    }

    for (size_t i = 0; i < table_size; ++i) {
        coverage_slots[i] = -1;
        if (geometry_slots) geometry_slots[i] = -1;
    }

    for (int i = 0; i < level_start; ++i) {
        pi_coverage_insert(pi, coverage_slots, table_size, i);
        if (geometry_slots) {
            pi_geometry_insert(
                pi,
                implicant_words,
                geometry_slots,
                table_size,
                i
            );
        }
    }

    int write = level_start;
    for (int read = level_start; read < found; ++read) {
        int existing_coverage = pi_coverage_lookup(
            pi,
            coverage_slots,
            table_size,
            read
        );
        int existing_geometry = geometry_slots
            ? pi_geometry_lookup(
                pi,
                implicant_words,
                geometry_slots,
                table_size,
                read
            )
            : -1;

        if (existing_geometry >= 0) continue;
        if (existing_coverage >= 0) {
            /*
             * Equal local coverage makes unshared cubes interchangeable for
             * the per-output covering problem. Distinct shareable geometries
             * are not interchangeable for joint pooling, however: each may
             * match a different output. Retain those alternatives while still
             * collapsing exact geometry duplicates.
             */
            if (
                !preserve_shared_geometries ||
                pi->shared[read] <= 0
            ) {
                continue;
            }
        }

        copy_pi_record(pi, write, read, implicant_words);
        if (existing_coverage < 0) {
            pi_coverage_insert(
                pi,
                coverage_slots,
                table_size,
                write
            );
        }
        if (geometry_slots) {
            pi_geometry_insert(
                pi,
                implicant_words,
                geometry_slots,
                table_size,
                write
            );
        }
        write++;
    }

    pi->foundPI = write;
    free(coverage_slots);
    free(geometry_slots);
    return 1;
}

static int sanitize_covsum(
    int covsum,
    int rows
) {
    if (covsum < 1) return 1;
    if (covsum > rows) return rows;
    return covsum;
}

static int rebuild_pi_buckets(
    PIstorage *pi,
    int level_start
) {
    int rows = pi->ON_minterms;
    int found = pi->foundPI;

    int *prev_counts = (int *)calloc((size_t)rows, sizeof(int));
    int *all_counts = (int *)calloc((size_t)rows, sizeof(int));
    int *next_prev = (int *)calloc((size_t)rows, sizeof(int));
    int *next_curr = (int *)calloc((size_t)rows, sizeof(int));

    if (!prev_counts || !all_counts || !next_prev || !next_curr) {
        free(prev_counts);
        free(all_counts);
        free(next_prev);
        free(next_curr);
        return 0;
    }

    for (int i = 0; i < found; ++i) {
        int bucket = sanitize_covsum(pi->covsum[i], rows) - 1;
        all_counts[bucket]++;
        if (i < level_start) {
            prev_counts[bucket]++;
        }
    }

    int total = 0;
    int prev_total = 0;
    for (int bucket = 0; bucket < rows; ++bucket) {
        next_prev[bucket] = total;
        next_curr[bucket] = total + prev_counts[bucket];

        prev_total += prev_counts[bucket];
        total += all_counts[bucket];

        pi->last_index[bucket] = prev_total;
        pi->k_last_index[bucket] = total;
    }

    for (int i = 0; i < level_start; ++i) {
        int bucket = sanitize_covsum(pi->covsum[i], rows) - 1;
        pi->covered[next_prev[bucket]++] = i;
    }

    for (int i = level_start; i < found; ++i) {
        int bucket = sanitize_covsum(pi->covsum[i], rows) - 1;
        pi->covered[next_curr[bucket]++] = i;
    }

    free(prev_counts);
    free(all_counts);
    free(next_prev);
    free(next_curr);
    return 1;
}

int finalize_pi_level(
    PIstorage *PInfo,
    int implicant_words,
    int level_start,
    bool deterministic_order,
    bool preserve_shared_geometries
) {
    if (deterministic_order) {
        return canonicalize_pi_order(
            PInfo,
            implicant_words,
            level_start,
            preserve_shared_geometries
        );
    }
    if (!prune_duplicate_coverage_in_level(
        PInfo,
        implicant_words,
        level_start,
        preserve_shared_geometries
    )) {
        return 0;
    }
    return rebuild_pi_buckets(PInfo, level_start);
}

int canonicalize_pi_order(
    PIstorage *pi,
    int implicant_words,
    int level_start,
    bool preserve_shared_geometries
) {
    if (!pi || pi->foundPI <= 0 || pi->ON_minterms <= 0) return 1;

    if (level_start < 0) level_start = 0;
    if (level_start > pi->foundPI) level_start = pi->foundPI;

    int n = pi->foundPI - level_start;
    if (n <= 1) {
        if (
            n == 1 &&
            !prune_duplicate_coverage_in_level(
                pi,
                implicant_words,
                level_start,
                preserve_shared_geometries
            )
        ) {
            return 0;
        }
        return rebuild_pi_buckets(pi, level_start);
    }

    int pichart_words = pi->pichart_words;

    int *order = (int *)malloc((size_t)n * sizeof(int));
    uint64_t *tmp_pichart_pos = (uint64_t *)malloc((size_t)n * (size_t)pichart_words * sizeof(uint64_t));
    uint64_t *tmp_implicants_pos = (uint64_t *)malloc((size_t)n * (size_t)implicant_words * sizeof(uint64_t));
    uint64_t *tmp_implicants_val = (uint64_t *)malloc((size_t)n * (size_t)implicant_words * sizeof(uint64_t));
    int *tmp_shared = (int *)malloc((size_t)n * sizeof(int));
    int *tmp_covsum = (int *)malloc((size_t)n * sizeof(int));

    if (
        !order ||
        !tmp_pichart_pos ||
        !tmp_implicants_pos ||
        !tmp_implicants_val ||
        !tmp_shared ||
        !tmp_covsum
    ) {
        free(order);
        free(tmp_pichart_pos);
        free(tmp_implicants_pos);
        free(tmp_implicants_val);
        free(tmp_shared);
        free(tmp_covsum);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        order[i] = level_start + i;
    }

    pi_canonical_sort_ctx.pi = pi;
    pi_canonical_sort_ctx.implicant_words = implicant_words;
    qsort(order, (size_t)n, sizeof(int), cmp_pi_canonical);

    for (int dst = 0; dst < n; ++dst) {
        int src = order[dst];

        memcpy(
            &tmp_pichart_pos[(size_t)dst * (size_t)pichart_words],
            &pi->pichart_pos[(size_t)src * (size_t)pichart_words],
            (size_t)pichart_words * sizeof(uint64_t)
        );

        memcpy(
            &tmp_implicants_pos[(size_t)dst * (size_t)implicant_words],
            &pi->implicants_pos[(size_t)src * (size_t)implicant_words],
            (size_t)implicant_words * sizeof(uint64_t)
        );

        memcpy(
            &tmp_implicants_val[(size_t)dst * (size_t)implicant_words],
            &pi->implicants_val[(size_t)src * (size_t)implicant_words],
            (size_t)implicant_words * sizeof(uint64_t)
        );

        tmp_shared[dst] = pi->shared[src];
        tmp_covsum[dst] = pi->covsum[src];
    }

    memcpy(
        &pi->pichart_pos[(size_t)level_start * (size_t)pichart_words],
        tmp_pichart_pos,
        (size_t)n * (size_t)pichart_words * sizeof(uint64_t)
    );

    memcpy(
        &pi->implicants_pos[(size_t)level_start * (size_t)implicant_words],
        tmp_implicants_pos,
        (size_t)n * (size_t)implicant_words * sizeof(uint64_t)
    );

    memcpy(
        &pi->implicants_val[(size_t)level_start * (size_t)implicant_words],
        tmp_implicants_val,
        (size_t)n * (size_t)implicant_words * sizeof(uint64_t)
    );

    memcpy(
        &pi->shared[level_start],
        tmp_shared,
        (size_t)n * sizeof(int)
    );

    memcpy(
        &pi->covsum[level_start],
        tmp_covsum,
        (size_t)n * sizeof(int)
    );

    free(order);
    free(tmp_pichart_pos);
    free(tmp_implicants_pos);
    free(tmp_implicants_val);
    free(tmp_shared);
    free(tmp_covsum);

    if (!prune_duplicate_coverage_in_level(
        pi,
        implicant_words,
        level_start,
        preserve_shared_geometries
    )) {
        return 0;
    }

    return rebuild_pi_buckets(pi, level_start);
}

void error_message(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

void resize(
    void **array,
    ArrayType type,
    int increase,
    int size,
    int nrows
) {
    // Input validation
    if (!array) {
        error_message("NULL array pointer passed to resize.");
    }

    if (type < TYPE_BOOL || type > TYPE_DOUBLE) {
        error_message("Invalid type for resizing.");
    }

    if (increase <= 0 || size < 0 || nrows <= 0) {
        error_message("Invalid parameters for resizing.");
    }

    // Check for overflow in size calculations
    if ((size_t)size > SIZE_MAX / (size_t)nrows ||
        (size_t)(size + increase) > SIZE_MAX / (size_t)nrows) {
        error_message("Size overflow in resize operation.");
    }

    size_t oldsize = (size_t)size * (size_t)nrows;
    size_t newsize = (size_t)(size + increase) * (size_t)nrows;

    // Additional overflow check for element size multiplication
    size_t element_size = 0;
    switch (type) {
        case TYPE_BOOL:
            element_size = sizeof(bool);
            break;
        case TYPE_INT:
        case TYPE_INT_ONES:
            element_size = sizeof(int);
            break;
        case TYPE_UINT64:
            element_size = sizeof(uint64_t);
            break;
        case TYPE_DOUBLE:
            element_size = sizeof(double);
            break;
    }

    if (newsize > SIZE_MAX / element_size) {
        error_message("Memory requirement too large for resize operation.");
    }

    void *tmp = NULL;

    switch (type) {
        case TYPE_BOOL:
            tmp = calloc(newsize, sizeof(bool));
            break;
        case TYPE_INT:
        case TYPE_INT_ONES:
            tmp = calloc(newsize, sizeof(int));
            break;
        case TYPE_UINT64:
            tmp = calloc(newsize, sizeof(uint64_t));
            break;
        case TYPE_DOUBLE:
            tmp = calloc(newsize, sizeof(double));
            break;
    }

    if (tmp == NULL) {
        error_message("Memory allocation failed during resize.");
    }

    if (type == TYPE_INT_ONES) {
        for (size_t i = 0; i < newsize; i++) {
            ((int *) tmp)[i] = 1; // Initialize all elements to 1
        }
    }

    if (*array != NULL) {
        switch (type) {
            case TYPE_BOOL:
                memcpy(tmp, *array, oldsize * sizeof(bool));
                break;
            case TYPE_INT:
            case TYPE_INT_ONES:
                memcpy(tmp, *array, oldsize * sizeof(int));
                break;
            case TYPE_UINT64:
                memcpy(tmp, *array, oldsize * sizeof(uint64_t));
                break;
            case TYPE_DOUBLE:
                memcpy(tmp, *array, oldsize * sizeof(double));
                break;
        }
        free(*array);
    }

    *array = tmp;
}


void trim_whitespace(char *str) {
    char *end;

    // Trim leading space
    while (isspace((unsigned char)*str)) str++;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end + 1) = '\0';
}

void read_pla_file(
    const char *filename,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int **nofvalues,
    int *max_value
) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        return;
    }

    char line[1024];
    int inputs = 0, outputs = 0;

    bool has_inputs = false;
    bool has_outputs = false;

    *max_value = 0; // Initialize max_value

    bool has_type = false;
    bool correct_type = false;
    int *ON_minterms = NULL;
    int *OFF_minterms = NULL;

    // First pass: Determine dimensions and count rows
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, ".i ", 3) == 0) {
            inputs = atoi(line + 3);
            has_inputs = true;
        } else if (strncmp(line, ".o ", 3) == 0) {
            outputs = atoi(line + 3);
            has_outputs = true;// Allocate ON_minterms and OFF_minterms
            ON_minterms = (int *)calloc(outputs, sizeof(int));
            OFF_minterms = (int *)calloc(outputs, sizeof(int));

            if (!ON_minterms || !OFF_minterms) {
                printf("Error: Memory allocation failed for ON_minterms or OFF_minterms\n");
                fclose(file);
                return;
            }
        } else if (strncmp(line, ".type ", 6) == 0) {
            trim_whitespace(line);
            has_type = true;
            if (strcmp(line + 6, "fr") == 0) {
                correct_type = true;
            } else {
                printf("Error: Only .type fr PLA files are supported (found: %s)\n", line + 6);
                fclose(file);
                return;
            }
            continue;
        } else if (
            line[0] == '#' ||
            strlen(line) <= 1 ||
            strncmp(line, ".p ", 3) == 0 ||
            strncmp(line, ".e", 2) == 0 ||
            strncmp(line, ".ilb ", 5) == 0 ||
            strncmp(line, ".ob ", 4) == 0
        ) {
            continue;
        } else {

            if (!has_inputs || !has_outputs) {
                printf("Error: Missing .i or .o headers in the .pla file\n");
                fclose(file);
                return;
            }

            char *input_part = strtok(line, " |");
            char *output_part = strtok(NULL, " |");

            if (input_part) {
                trim_whitespace(input_part);
            }

            if (output_part) {
                trim_whitespace(output_part);
            };

            if (
                input_part &&
                output_part &&
                (int)strlen(input_part) == inputs &&
                (int)strlen(output_part) == outputs
            ) {
                for (int i = 0; i < outputs; i++) {
                    if (output_part[i] == '1') {
                        ON_minterms[i]++;
                    } else if (output_part[i] == '0') {
                        OFF_minterms[i]++;
                    }
                }
            }
        }
    }

    if (!has_type || !correct_type) {
        printf("Error: Missing or unsupported .type directive (expected .type fr)\n");
        fclose(file);
        return;
    }

    *ninputs = inputs;
    *noutputs = outputs;

    // Allocate and zero-initialize PIstorage array to ensure all fields start as NULL/0
    *PInfo = (PIstorage *)calloc((size_t)outputs, sizeof(PIstorage));
    if (!*PInfo) {
        printf("Error: Memory allocation failed for PInfo\n");
        free(ON_minterms);
        free(OFF_minterms);
        fclose(file);
        return;
    }

    // Pointers are already NULL due to calloc; explicitly set only those we immediately use below
    for (int o = 0; o < outputs; o++) {
        (*PInfo)[o].ON_set = NULL;
        (*PInfo)[o].OFF_set = NULL;
    }

    int temp_poscols[outputs];
    int temp_negcols[outputs];

    for (int o = 0; o < outputs; o++) {
        (*PInfo)[o].inputs = inputs;
        (*PInfo)[o].outputs = outputs;

        // printf("input %d: ON_minterms=%d, OFF_minterms=%d\n", o, ON_minterms[o], OFF_minterms[o]);
        (*PInfo)[o].ON_minterms = ON_minterms[o];
        (*PInfo)[o].ON_set = (int *)calloc(ON_minterms[o] * inputs, sizeof(int));
        if (!(*PInfo)[o].ON_set && ON_minterms[o] > 0) {
            printf("Error: Memory allocation failed for ON_set[%d]\n", o);
            // Free up previously allocated memory
            for (int c = 0; c <= o; c++) {
                free((*PInfo)[c].ON_set);
                free((*PInfo)[c].OFF_set);
            }
            free(*PInfo);
            free(ON_minterms);
            free(OFF_minterms);
            fclose(file);
            return;
        }

        (*PInfo)[o].OFF_minterms = OFF_minterms[o];
        (*PInfo)[o].OFF_set = (int *)calloc(OFF_minterms[o] * inputs, sizeof(int));
        if (!(*PInfo)[o].OFF_set && OFF_minterms[o] > 0) {
            printf("Error: Memory allocation failed for OFF_set[%d]\n", o);
            // Free up previously allocated memory
            for (int c = 0; c <= o; c++) {
                free((*PInfo)[c].ON_set);
                free((*PInfo)[c].OFF_set);
            }
            free(*PInfo);
            free(ON_minterms);
            free(OFF_minterms);
            fclose(file);
            return;
        }

        temp_poscols[o] = ON_minterms[o];
        temp_negcols[o] = OFF_minterms[o];
    }

    *nofvalues = (int *)calloc(inputs, sizeof(int));
    if (!*nofvalues) {
        printf("Error: Memory allocation failed for nofvalues\n");
        for (int o = 0; o < outputs; o++) {
            free((*PInfo)[o].ON_set);
            free((*PInfo)[o].OFF_set);
        }
        free(*PInfo);
        free(ON_minterms);
        free(OFF_minterms);
        fclose(file);
        return;
    }

    // Second pass: Fill ON_set and OFF_set and calculate max_value
    rewind(file);
    while (fgets(line, sizeof(line), file)) {
        if (line[0] == '#' || strlen(line) <= 1 || strncmp(line, ".i", 2) == 0 || strncmp(line, ".o", 2) == 0 || strncmp(line, ".e", 2) == 0 || strncmp(line, ".p", 2) == 0) continue;

        char *input_part = strtok(line, " |");
        char *output_part = strtok(NULL, " |");

        if (input_part) trim_whitespace(input_part);
        if (output_part) trim_whitespace(output_part);

        if (
            input_part &&
            output_part &&
            (int)strlen(input_part) == inputs &&
            (int)strlen(output_part) == outputs
        ) {

            for (int o = 0; o < outputs; o++) {
                int *target_data = NULL;
                int ncols = 0;
                int col_index = 0;

                if (output_part[o] == '1') {
                    target_data = (*PInfo)[o].ON_set;
                    ncols = ON_minterms[o];
                    col_index = ncols - temp_poscols[o];
                    temp_poscols[o]--;
                } else if (output_part[o] == '0') {
                    target_data = (*PInfo)[o].OFF_set;
                    ncols = (*PInfo)[o].OFF_minterms;
                    col_index = ncols - temp_negcols[o];
                    temp_negcols[o]--;
                } else {
                    continue;
                }

                if (col_index < 0 || col_index >= ncols) {
                    error_message("Invalid col_index.");
                }

                for (int j = 0; j < inputs; j++) {
                    int value = (input_part[j] == '1') ? 2
                            : (input_part[j] == '0') ? 1
                            : 0;
                    target_data[col_index * inputs + j] = value;

                    if (value > *max_value) {
                        *max_value = value;
                    }
                    if (value + 1 > (*nofvalues)[j]) {
                        (*nofvalues)[j] = value + 1;
                    }
                }
            }
        }
    }

    free(ON_minterms);
    free(OFF_minterms);

    fclose(file);
}

static uint64_t projection_row_hash(const int *row, int ninputs) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int input = 0; input < ninputs; ++input) {
        hash ^= (uint64_t)(unsigned int)row[input];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void clear_shared_projection_rows(
    PIstorage *PInfo,
    int noutputs
) {
    if (!PInfo || noutputs <= 0) return;
    free(PInfo[0].projection_rows);
    for (int output = 0; output < noutputs; ++output) {
        PInfo[output].projection_rows = NULL;
        PInfo[output].projection_row_count = 0;
        free(PInfo[output].ON_projection_ids);
        PInfo[output].ON_projection_ids = NULL;
    }
}

bool prepare_shared_projection_rows(
    PIstorage *PInfo,
    int ninputs,
    int noutputs
) {
    if (!PInfo || ninputs <= 0 || noutputs <= 0) return false;
    clear_shared_projection_rows(PInfo, noutputs);

    size_t memberships = 0u;
    for (int output = 0; output < noutputs; ++output) {
        size_t rows = (size_t)PInfo[output].ON_minterms;
        /*
         * OFF rows are deliberately absent from this eager layout. Masked
         * outputs validate them by bitset intersection; scalar outputs must
         * project them lazily so their mixed-radix discovery bound can stop
         * the scan without first paying to project the whole OFF set.
         */
        if (rows > SIZE_MAX - memberships) return false;
        memberships += rows;
    }
    if (
        memberships == 0u ||
        memberships > (size_t)INT_MAX ||
        memberships > SIZE_MAX / (size_t)ninputs
    ) {
        return false;
    }

    size_t table_size = 16u;
    while (table_size < memberships * 2u) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }

    int *slots = (int *)malloc(table_size * sizeof(int));
    int *rows = (int *)malloc(
        memberships * (size_t)ninputs * sizeof(int)
    );
    if (!slots || !rows) {
        free(slots);
        free(rows);
        return false;
    }
    for (size_t slot = 0; slot < table_size; ++slot) slots[slot] = -1;

    for (int output = 0; output < noutputs; ++output) {
        if (PInfo[output].ON_minterms > 0) {
            PInfo[output].ON_projection_ids = (int *)malloc(
                (size_t)PInfo[output].ON_minterms * sizeof(int)
            );
        }
        if (
            (PInfo[output].ON_minterms > 0 &&
                !PInfo[output].ON_projection_ids)
        ) {
            free(slots);
            free(rows);
            PInfo[0].projection_rows = NULL;
            clear_shared_projection_rows(PInfo, noutputs);
            return false;
        }
    }

    int row_count = 0;
    for (int output = 0; output < noutputs; ++output) {
        int count = PInfo[output].ON_minterms;
        const int *source = PInfo[output].ON_set;
        int *ids = PInfo[output].ON_projection_ids;

        for (int source_row = 0; source_row < count; ++source_row) {
            const int *row = &source[
                (size_t)source_row * (size_t)ninputs
            ];
            size_t mask = table_size - 1u;
            size_t slot =
                (size_t)(projection_row_hash(row, ninputs) & mask);

            while (slots[slot] >= 0) {
                int candidate = slots[slot];
                if (
                    memcmp(
                        &rows[(size_t)candidate * (size_t)ninputs],
                        row,
                        (size_t)ninputs * sizeof(int)
                    ) == 0
                ) {
                    break;
                }
                slot = (slot + 1u) & mask;
            }

            if (slots[slot] < 0) {
                if ((size_t)row_count >= memberships) {
                    free(slots);
                    free(rows);
                    PInfo[0].projection_rows = NULL;
                    clear_shared_projection_rows(PInfo, noutputs);
                    return false;
                }
                memcpy(
                    &rows[(size_t)row_count * (size_t)ninputs],
                    row,
                    (size_t)ninputs * sizeof(int)
                );
                slots[slot] = row_count++;
            }
            ids[source_row] = slots[slot];
        }
    }

    free(slots);
    PInfo[0].projection_rows = rows;
    for (int output = 0; output < noutputs; ++output) {
        PInfo[output].projection_row_count = row_count;
    }
    return true;
}

static void clear_off_compat_masks(
    PIstorage *PInfo,
    int noutputs
) {
    if (!PInfo || noutputs <= 0) return;
    for (int output = 0; output < noutputs; ++output) {
        free(PInfo[output].off_mask_offsets);
        free(PInfo[output].off_compat_masks);
        PInfo[output].off_mask_offsets = NULL;
        PInfo[output].off_compat_masks = NULL;
        PInfo[output].off_mask_words = 0;
        PInfo[output].off_mask_count = 0;
        PInfo[output].off_has_dc = false;
    }
}

bool prepare_off_compat_masks(
    PIstorage *PInfo,
    int ninputs,
    int noutputs,
    const int *nofvalues
) {
    if (
        !PInfo ||
        !nofvalues ||
        ninputs <= 0 ||
        noutputs <= 0
    ) {
        return false;
    }
    clear_off_compat_masks(PInfo, noutputs);

    size_t mask_count = 0u;
    for (int input = 0; input < ninputs; ++input) {
        if (nofvalues[input] < 1) return false;
        size_t values = (size_t)(nofvalues[input] - 1);
        if (values > (size_t)INT_MAX - mask_count) return false;
        mask_count += values;
    }
    if (mask_count == 0u || mask_count > (size_t)INT_MAX) {
        return true;
    }

    for (int output = 0; output < noutputs; ++output) {
        PIstorage *pi = &PInfo[output];
        if (pi->OFF_minterms <= 0) continue;

        int words = (pi->OFF_minterms + 63) / 64;
        if (
            words <= 0 ||
            mask_count > SIZE_MAX / (size_t)words
        ) {
            clear_off_compat_masks(PInfo, noutputs);
            return false;
        }
        size_t mask_words = mask_count * (size_t)words;
        if (mask_words > SIZE_MAX / sizeof(uint64_t)) {
            clear_off_compat_masks(PInfo, noutputs);
            return false;
        }

        /*
         * Wildcard rows retain the established mask path. For fully
         * specified rows, use the new path only when its persistent bitsets
         * are no larger than the scalar OFF matrix they accelerate. This is
         * comfortably true for binary data (about one sixteenth as large),
         * while avoiding an accidental memory expansion on very high-cardinal
         * multi-valued inputs.
         */
        bool has_dc = false;
        size_t off_cells =
            (size_t)pi->OFF_minterms * (size_t)ninputs;
        for (size_t cell = 0; cell < off_cells; ++cell) {
            if (pi->OFF_set[cell] == 0) {
                has_dc = true;
                break;
            }
        }
        pi->off_has_dc = has_dc;
        if (!has_dc) {
            if (
                off_cells > SIZE_MAX / sizeof(int) ||
                mask_words * sizeof(uint64_t) >
                    off_cells * sizeof(int)
            ) {
                continue;
            }
        }

        pi->off_mask_offsets = (int *)malloc(
            ((size_t)ninputs + 1u) * sizeof(int)
        );
        pi->off_compat_masks = (uint64_t *)calloc(
            mask_words,
            sizeof(uint64_t)
        );
        if (!pi->off_mask_offsets || !pi->off_compat_masks) {
            clear_off_compat_masks(PInfo, noutputs);
            return false;
        }

        int offset = 0;
        for (int input = 0; input < ninputs; ++input) {
            pi->off_mask_offsets[input] = offset;
            offset += nofvalues[input] - 1;
        }
        pi->off_mask_offsets[ninputs] = offset;
        pi->off_mask_words = words;
        pi->off_mask_count = (int)mask_count;

        for (int row = 0; row < pi->OFF_minterms; ++row) {
            int word = row / 64;
            uint64_t bit = UINT64_C(1) << (row % 64);
            for (int input = 0; input < ninputs; ++input) {
                int off_value =
                    pi->OFF_set[(size_t)row * (size_t)ninputs + input];
                int value_begin = off_value == 0 ? 1 : off_value;
                int value_end = off_value == 0
                    ? nofvalues[input] - 1
                    : off_value;
                if (
                    value_begin < 1 ||
                    value_end >= nofvalues[input]
                ) {
                    clear_off_compat_masks(PInfo, noutputs);
                    return false;
                }
                for (int value = value_begin; value <= value_end; ++value) {
                    int mask =
                        pi->off_mask_offsets[input] + value - 1;
                    pi->off_compat_masks[
                        (size_t)mask * (size_t)words + (size_t)word
                    ] |= bit;
                }
            }
        }
    }

    return true;
}

void write_pla_file(
    const char *filename,
    PIstorage *PInfo
) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Unable to open file %s for writing\n", filename);
        return;
    }

    int ninputs = PInfo[0].inputs;
    int noutputs = PInfo[0].outputs;

    // Write the header
    fprintf(file, ".i %d\n", ninputs); // Number of inputs
    fprintf(file, ".o %d\n", noutputs); // Number of outputs

    // --- temporary debugging code ---
    // print each solution matrix
    // for (int o = 7; o < 8; o++) {
    //     printf("Output %d: solmin = %d\n", o + 1, PInfo[o].solmin);
    //     for (int r = 0; r < PInfo[o].solmin; r++) {
    //         printf("Row %d: ", r);
    //         for (int c = 0; c < ninputs; c++) {
    //             printf("%d ", PInfo[o].solution[c * PInfo[o].solmin + r]);
    //         }
    //         printf("\n");
    //     }
    // }
    // --------------------------------

    // Data structure to store unique rows
    typedef struct {
        int *inputs; // Input part of the row
        int *outputs; // Output part of the row
    } UniqueRow;

    UniqueRow *unique_rows = NULL;
    int unique_count = 0;

    // Iterate through all outputs and rows to find unique rows
    for (int o = 0; o < noutputs; o++) {
        for (int r = 0; r < PInfo[o].solmin; r++) {
            // Extract the input part of the row from the "solution" matrix
            int *current_row = (int *)malloc(ninputs * sizeof(int));
            if (!current_row) {
                printf("Error: Memory allocation failed for current_row\n");
                return;
            }

            // printf("Processing output %d, row %d: ", o, r);
            for (int c = 0; c < ninputs; c++) {
                current_row[c] = PInfo[o].solution[c * PInfo[o].solmin + r];
                // printf(" %d", current_row[c]);
            }


            // Check if the row is already in unique_rows
            bool is_unique = true;
            for (int i = 0; i < unique_count; i++) {
                if (memcmp(unique_rows[i].inputs, current_row, ninputs * sizeof(int)) == 0) {
                    // Row already exists, update the output part
                    unique_rows[i].outputs[o] = 1;
                    is_unique = false;
                    break;
                }
            }

            // printf(" (%s)\n", is_unique ? "Unique" : "Not unique");

            if (is_unique) {
                // Add the new unique row
                unique_rows = (UniqueRow *)realloc(unique_rows, (unique_count + 1) * sizeof(UniqueRow));
                if (!unique_rows) {
                    printf("Error: Memory allocation failed for unique_rows\n");
                    free(current_row);
                    return;
                }
                unique_rows[unique_count].inputs = current_row;
                unique_rows[unique_count].outputs = (int *)calloc(noutputs, sizeof(int));
                if (!unique_rows[unique_count].outputs) {
                    printf("Error: Memory allocation failed for outputs\n");
                    free(current_row);
                    return;
                }
                unique_rows[unique_count].outputs[o] = 1;
                unique_count++;
            } else {
                free(current_row); // Free memory if the row is not unique
            }
        }
    }

    // Write the total number of unique rows
    fprintf(file, ".p %d\n", unique_count);

    // Write the body
    for (int i = 0; i < unique_count; i++) {
        // Write the input part
        for (int c = 0; c < ninputs; c++) {
            int value = unique_rows[i].inputs[c];
            if (value == 0) {
                fprintf(file, "-"); // Don't care
            } else if (value == 1) {
                fprintf(file, "0"); // Closed gate
            } else if (value == 2) {
                fprintf(file, "1"); // Open gate
            } else {
                fprintf(file, "%d", value - 1); // Multi-value (adjusted)
            }
        }

        fprintf(file, " "); // Separator between input and output

        // Write the output part
        for (int o = 0; o < noutputs; o++) {
            fprintf(file, "%d", unique_rows[i].outputs[o]);
        }

        fprintf(file, "\n"); // End of row
    }

    // Write the end marker
    fprintf(file, ".e\n");

    // Free allocated memory
    for (int i = 0; i < unique_count; i++) {
        free(unique_rows[i].inputs);
        free(unique_rows[i].outputs);
    }
    free(unique_rows);

    fclose(file);
}

void cleanup(PIstorage *PInfo, ThreadBuffer **buffer) {
    if (!PInfo) {
        free(buffer);
        return;
    }

    int noutputs = PInfo[0].outputs;
    for (int o = 0; o < noutputs; o++) {
        if (o == 0) free(PInfo[o].projection_rows);
        free(PInfo[o].ON_projection_ids);
        free(PInfo[o].off_mask_offsets);
        free(PInfo[o].off_compat_masks);
        free(PInfo[o].ON_set);
        free(PInfo[o].OFF_set);
        free(PInfo[o].covered);
        free(PInfo[o].last_index);
        free(PInfo[o].k_last_index);
        free(PInfo[o].pichart_pos);
        free(PInfo[o].implicants_pos);
        free(PInfo[o].implicants_val);
        free(PInfo[o].shared);
        free(PInfo[o].covsum);
        free(PInfo[o].previndices);
        free(PInfo[o].indices);
        free(PInfo[o].cov_word_index);
        free(PInfo[o].shifted_cov_mask);
        free(PInfo[o].nofpi);

        free(PInfo[o].solution);
        for (int p = 0; p < PInfo[o].pool_count; p++) {
            free(PInfo[o].pool_solutions[p]);
        }
        free(PInfo[o].pool_solutions);
    }

    free(PInfo);

    if (!buffer) return;

    int threads = 1;
    if (buffer[0]) {
        threads = buffer[0]->threads;
    }

    // Free buffer buffers
    for (int t = 0; t < threads; t++) {
        if (!buffer[t]) continue;
        for (int o = 0; o < noutputs; o++) {
            free(buffer[t][o].pichart_values);
            free(buffer[t][o].decpos);
            free(buffer[t][o].covsum);
            free(buffer[t][o].fixed_bits);
            free(buffer[t][o].value_bits);
            free(buffer[t][o].projection_codes);
            free(buffer[t][o].projection_has_dc);
            free(buffer[t][o].task_row_codes);
            free(buffer[t][o].task_seen_stamps);
            free(buffer[t][o].task_config_bits);
        }
        free(buffer[t]);
    }
    if (buffer) free(buffer);
}

/*
 * A worker retains only the candidates for its current position task.  The
 * upper bound is the projected value-space size, not the full ON-set size.
 */
static bool ensure_thread_buffer_capacity(
    ThreadBuffer *buffer,
    int needed,
    int pichart_words,
    int implicant_words
) {
    if (!buffer || needed <= 0 || pichart_words <= 0 || implicant_words <= 0) {
        return false;
    }
    if (buffer->capacity >= needed) return true;

    int capacity = buffer->capacity > 0 ? buffer->capacity : 16;
    while (capacity < needed) {
        if (capacity > INT_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }

    if (
        (size_t)capacity > SIZE_MAX / (size_t)pichart_words ||
        (size_t)capacity > SIZE_MAX / (size_t)implicant_words
    ) {
        return false;
    }

    uint64_t *pichart_values = (uint64_t*)calloc(
        (size_t)capacity * (size_t)pichart_words,
        sizeof(uint64_t)
    );

    int *decpos = (int*)calloc((size_t)capacity, sizeof(int));
    int *covsum = (int*)calloc((size_t)capacity, sizeof(int));
    uint64_t *fixed_bits = (uint64_t*)calloc(
        (size_t)capacity * (size_t)implicant_words,
        sizeof(uint64_t)
    );

    uint64_t *value_bits = (uint64_t*)calloc(
        (size_t)capacity * (size_t)implicant_words,
        sizeof(uint64_t)
    );

    if (!pichart_values || !decpos || !covsum || !fixed_bits || !value_bits) {
        free(pichart_values);
        free(decpos);
        free(covsum);
        free(fixed_bits);
        free(value_bits);
        return false;
    }

    free(buffer->pichart_values);
    free(buffer->decpos);
    free(buffer->covsum);
    free(buffer->fixed_bits);
    free(buffer->value_bits);
    buffer->pichart_values = pichart_values;
    buffer->decpos = decpos;
    buffer->covsum = covsum;
    buffer->fixed_bits = fixed_bits;
    buffer->value_bits = value_bits;
    buffer->capacity = capacity;
    return true;
}

static bool ensure_projection_buffer_capacity(
    ThreadBuffer *buffer,
    int needed
) {
    if (!buffer || needed <= 0) return false;
    if (
        buffer->projection_capacity >= needed &&
        buffer->projection_codes &&
        buffer->projection_has_dc
    ) {
        return true;
    }

    int capacity =
        buffer->projection_capacity > 0
            ? buffer->projection_capacity
            : 16;
    while (capacity < needed) {
        if (capacity > INT_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }

    int *codes = (int *)malloc((size_t)capacity * sizeof(int));
    unsigned char *has_dc = (unsigned char *)malloc(
        (size_t)capacity * sizeof(unsigned char)
    );
    if (!codes || !has_dc) {
        free(codes);
        free(has_dc);
        return false;
    }

    free(buffer->projection_codes);
    free(buffer->projection_has_dc);
    buffer->projection_codes = codes;
    buffer->projection_has_dc = has_dc;
    buffer->projection_capacity = capacity;
    return true;
}

/*
 * Row projections are consumed one output at a time, so one worker-owned
 * array can hold both the current ON and OFF codes. Candidate records remain
 * per output because they must survive until cross-output sharing is merged.
 */
static bool ensure_task_row_capacity(
    ThreadBuffer *workspace,
    size_t needed
) {
    if (!workspace || needed == 0) return false;
    if (
        workspace->task_row_capacity >= needed &&
        workspace->task_row_codes
    ) {
        return true;
    }

    size_t capacity =
        workspace->task_row_capacity > 0
            ? workspace->task_row_capacity
            : 64u;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = needed;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(int)) return false;

    int *codes = (int *)malloc(capacity * sizeof(int));
    if (!codes) return false;

    free(workspace->task_row_codes);
    workspace->task_row_codes = codes;
    workspace->task_row_capacity = capacity;
    return true;
}

/*
 * A generation stamp replaces allocating and clearing projected-space
 * visited arrays for every output of every support. OFF deduplication and ON
 * candidate discovery take successive epochs in the same worker-owned table.
 */
static bool ensure_task_seen_capacity(
    ThreadBuffer *workspace,
    size_t needed
) {
    if (!workspace || needed == 0) return false;
    if (
        workspace->task_seen_capacity >= needed &&
        workspace->task_seen_stamps
    ) {
        return true;
    }

    size_t capacity =
        workspace->task_seen_capacity > 0
            ? workspace->task_seen_capacity
            : 64u;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = needed;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(uint32_t)) return false;

    uint32_t *stamps = (uint32_t *)calloc(
        capacity,
        sizeof(uint32_t)
    );
    if (!stamps) return false;

    free(workspace->task_seen_stamps);
    workspace->task_seen_stamps = stamps;
    workspace->task_seen_capacity = capacity;
    workspace->task_seen_epoch = 0;
    return true;
}

static uint32_t begin_task_seen_epoch(ThreadBuffer *workspace) {
    assert(
        workspace &&
        workspace->task_seen_stamps &&
        workspace->task_seen_capacity > 0
    );

    workspace->task_seen_epoch++;
    if (workspace->task_seen_epoch == 0) {
        memset(
            workspace->task_seen_stamps,
            0,
            workspace->task_seen_capacity * sizeof(uint32_t)
        );
        workspace->task_seen_epoch = 1;
    }
    return workspace->task_seen_epoch;
}

static bool ensure_task_config_word_capacity(
    ThreadBuffer *workspace,
    size_t needed_words
) {
    if (!workspace || needed_words == 0u) return false;
    if (
        workspace->task_config_word_capacity >= needed_words &&
        workspace->task_config_bits
    ) {
        return true;
    }

    size_t capacity = workspace->task_config_word_capacity > 0u
        ? workspace->task_config_word_capacity
        : needed_words;
    while (capacity < needed_words) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = needed_words;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(uint64_t)) return false;

    uint64_t *bits = (uint64_t *)calloc(capacity, sizeof(uint64_t));
    if (!bits) return false;
    free(workspace->task_config_bits);
    workspace->task_config_bits = bits;
    workspace->task_config_word_capacity = capacity;
    return true;
}

static bool compact_projected_configuration(
    const int *row,
    const int *support,
    int support_size,
    const int *nofvalues,
    int *configuration_out
) {
    int configuration = 0;
    int base = 1;
    for (int c = 0; c < support_size; ++c) {
        int input = support[c];
        int value = row[input];
        int levels = nofvalues[input] - 1;
        if (value <= 0 || value > levels) return false;
        configuration += (value - 1) * base;
        base *= levels;
    }
    *configuration_out = configuration;
    return true;
}

static int compare_int_ascending(const void *left, const void *right) {
    int a = *(const int *)left;
    int b = *(const int *)right;
    return (a > b) - (a < b);
}

static bool sorted_int_contains(const int *values, int count, int value) {
    int low = 0;
    int high = count;
    while (low < high) {
        int middle = low + (high - low) / 2;
        if (values[middle] < value) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < count && values[low] == value;
}

char *prefix_basename(const char *filepath, const char *prefix) {
    const char *basename = strrchr(filepath, '/');
    if (basename) {
        basename++; // skip past '/'
    } else {
        basename = filepath; // no '/' found
    }

    size_t prefix_len = strlen(prefix);
    size_t base_len = strlen(basename);

    char *new_name = malloc(prefix_len + base_len + 1);
    if (!new_name) return NULL;

    strcpy(new_name, prefix);
    strcat(new_name, basename);

    return new_name;
}

static int checkpoint_uncovered_rows(const PIstorage *pi) {
    if (!pi || pi->ON_minterms <= 0) return 0;
    if (pi->foundPI <= 0 || !pi->pichart_pos) return pi->ON_minterms;

    PIChartView chart = pi_chart_view(pi);
    int uncovered = 0;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        bool covered = false;
        for (int column = 0; column < pi->foundPI; ++column) {
            if (chart_covers(&chart, column, row)) {
                covered = true;
                break;
            }
        }
        if (!covered) uncovered++;
    }
    return uncovered;
}

void print_info(const char *INFO_PATH, const int info_level) {
    PIstorage *pi_tmp = NULL;
    int ni=0, no=0, bpw=0, vbw=0, ipw=0, ck=0, ml=0, wp=0, st=0, pm=0;
    int *stopc_tmp = NULL; int *nofvals_tmp = NULL;
    char *src_saved=NULL; char *dst_saved=NULL;
    double elapsed_total = 0.0, elapsed_scp = 0.0; uint64_t last_task = 0ull;

    if (load_checkpoint(
            INFO_PATH,
            &pi_tmp,
            &ni,
            &no,
            &bpw,
            &vbw,
            &ipw,
            &ck,
            &stopc_tmp,
            &ml,
            &wp,
            &st,
            &pm,
            &nofvals_tmp,
            &src_saved,
            &dst_saved,
            &elapsed_total,
            &elapsed_scp,
            &last_task
    ) != 0) {
        fprintf(stderr, "Error: failed to load checkpoint from %s\n", INFO_PATH);
        return;
    }

    // printf("Checkpoint: %s\n", INFO_PATH);
    printf("Source: %s\n", src_saved ? src_saved : "-");
    printf("Destination: %s\n", dst_saved ? dst_saved : "-");
    printf("Inputs: %d, Outputs: %d\n", ni, no);
    // printf("Bits per word: %d, value bit width: %d, implicant words: %d\n", bpw, vbw, ipw);
    printf("Level k reached: %d\n", ck);
    const char *stopping_policy = certified_model_supported(pi_tmp, ni, no)
        ? "adaptive heuristic"
        : heuristic_pattern_model_supported(pi_tmp, ni, no)
            ? "heuristic plateau"
            : "unsupported";
    printf("Stopping policy: %s\n", stopping_policy);
    uint64_t maxt = nchoosek(ni, ck);

    if (ck > 0 && maxt > 0) {
        double pct = (double)(last_task + 1) * 100.0 / (double)maxt;
        if (pct > 100.0) pct = 100.0;
        printf("Progress at k: task=%llu / %llu (%.2f%%)\n", (unsigned long long)last_task, (unsigned long long)maxt, pct);
    }

    printf("Total time spent: %.3fs (%.3fs SCP)\n", elapsed_total, elapsed_scp);
    // printf("Stage: ready_for_coverage\n");

    if (info_level > 0) {
        for (int o = 0; o < no; ++o) {
            bool stop_flag = pi_tmp[o].stop_search;
            int uncovered = checkpoint_uncovered_rows(&pi_tmp[o]);

            printf(
                "Output %d: ON=%d OFF=%d foundPI=%d solmin=%d stop=%s "
                "cover_feasible=%s uncovered=%d\n",
                o,
                pi_tmp[o].ON_minterms,
                pi_tmp[o].OFF_minterms,
                pi_tmp[o].foundPI,
                pi_tmp[o].solmin,
                stop_flag ? "yes" : "no",
                uncovered == 0 ? "yes" : "no",
                uncovered
            );
        }
    }

    if (src_saved) free(src_saved);
    if (dst_saved) free(dst_saved);
    if (nofvals_tmp) free(nofvals_tmp);
    if (stopc_tmp) free(stopc_tmp);

    // Use cleanup to free PInfo; provide a dummy buffer holder
    ThreadBuffer **dummy = (ThreadBuffer**)calloc(1, sizeof(ThreadBuffer*));
    cleanup(pi_tmp, dummy);
}

static bool candidate_matches_wildcard_off(
    const PIstorage *pi,
    const int *on_row,
    const int *support,
    int support_size
) {
    assert(pi);
    assert(on_row);
    assert(support);
    assert(support_size > 0);
    assert(pi->off_mask_words > 0);
    assert(pi->off_mask_offsets);
    assert(pi->off_compat_masks);

    int first_input = support[0];
    int first_value = on_row[first_input];
    int first_mask =
        pi->off_mask_offsets[first_input] + first_value - 1;
    assert(first_value > 0);
    assert(first_mask >= 0 && first_mask < pi->off_mask_count);

    const uint64_t *first_words = &pi->off_compat_masks[
        (size_t)first_mask * (size_t)pi->off_mask_words
    ];
    for (int word = 0; word < pi->off_mask_words; ++word) {
        uint64_t matches = first_words[word];
        for (int c = 1; c < support_size && matches != 0; ++c) {
            int input = support[c];
            int value = on_row[input];
            int mask = pi->off_mask_offsets[input] + value - 1;
            assert(value > 0);
            assert(mask >= 0 && mask < pi->off_mask_count);
            matches &= pi->off_compat_masks[
                (size_t)mask * (size_t)pi->off_mask_words +
                (size_t)word
            ];
        }
        if (matches != 0) return true;
    }
    return false;
}


int process_task(
    uint64_t task,
    int k,
    int ninputs,
    int noutputs,
    int *nofvalues,
    int *bit_index,
    int *word_index,
    uint64_t *shifted_mask,
    int implicant_words,
    PIstorage *PInfo,
    ThreadBuffer **buffer,
    int tid,
    ccubes_mutex *output_locks,
    PICoverageIndex *coverage_indices,
    int *max_shared,
    int increase,
    int *multiplier
) {
    int tempk[k];
    uint64_t combination = task;

    DBG_TRACE_BLOCK {
        // if (task % 1000000 == 0 && task / 1000000 > 0) { // every 1M tasks
        //     fprintf(debug_out, "-");
        // }
        // if (task % 50000000 == 0 && task / 50000000 > 0) { // every 50M tasks
        //     fprintf(debug_out, "\n");
        // }
        // if (task % 100000000 == 0 && task / 100000000 > 0) { // every 100M tasks
        //     fprintf(debug_out, " (%lld)", task / 100000000);
        // }
    }

    // fill the combination for the current task / combination number
    int x = 0;
    for (int i = 0; i < k; i++) {
        while (1) {
            uint64_t cval = nchoosek(ninputs - (x + 1), k - (i + 1));
            if (cval == 0 || cval > combination) break; // guard against overflow/invalid
            combination -= cval;
            x++;
        }
        // clamp to valid range [0, ninputs-1]
        if (x < 0) x = 0;
        if (x >= ninputs) x = ninputs - 1;
        tempk[i] = x;
        x++;
    }

    DBG_TRACE_BLOCK {
        // fprintf(debug_out, "tempk: ");
        // for (int i = 0; i < k; i++) {
        //     fprintf(debug_out, "%d ", tempk[i] + 1);
        // }
        // fprintf(debug_out, "\n");
    }

    uint64_t fixed_bits[implicant_words];
    for (int w = 0; w < implicant_words; w++) {
        fixed_bits[w] = 0ULL;
    }

    for (int c = 0; c < k; c++) {
        fixed_bits[word_index[tempk[c]]] |= shifted_mask[tempk[c]]; // for implicants_pos
    }

    /*
     * The support and mixed-radix layout are task-wide. More importantly, the
     * same source input row is usually present in every output-specific ON/OFF
     * partition. Project each distinct row once, then map those codes into the
     * per-output partitions below.
     */
    int mbase[k];
    mbase[0] = 1;
    for (int i = 1; i < k; i++) {
        mbase[i] = mbase[i - 1] * nofvalues[tempk[i - 1]];
    }

    int space_size = 1;
    int configuration_count = 1;
    for (int i = 0; i < k; i++) {
        space_size *= nofvalues[tempk[i]];
        int levels = nofvalues[tempk[i]] - 1;
        if (levels < 1) levels = 1;
        configuration_count *= levels;
    }
    if (space_size < 1) space_size = 1;
    if (configuration_count < 1) configuration_count = 1;

    int mbase_sum = 0;
    for (int i = 0; i < k; i++) {
        mbase_sum += mbase[i];
    }

    int projection_row_count = PInfo[0].projection_row_count;
    const int *projection_rows = PInfo[0].projection_rows;
    bool use_shared_projection =
        projection_row_count > 0 &&
        projection_rows != NULL &&
        ensure_projection_buffer_capacity(
            &buffer[tid][0],
            projection_row_count
        );
    int *projection_codes = use_shared_projection
        ? buffer[tid][0].projection_codes
        : NULL;
    unsigned char *projection_has_dc = use_shared_projection
        ? buffer[tid][0].projection_has_dc
        : NULL;

    if (use_shared_projection) {
        for (int row = 0; row < projection_row_count; ++row) {
            int acc = 0;
            bool has_dc = false;
            const int *values = &projection_rows[
                (size_t)row * (size_t)ninputs
            ];
            for (int c = 0; c < k; ++c) {
                int value = values[tempk[c]];
                if (value == 0) has_dc = true;
                acc += value * mbase[c];
            }
            projection_codes[row] = acc;
            projection_has_dc[row] = has_dc ? 1u : 0u;
        }
    }

    int max_found = 0;
    for (int o = 0; o < noutputs; o++) {
        int ON_minterms = PInfo[o].ON_minterms;

        if (ON_minterms == 0 || PInfo[o].stop_search) {
            continue;
        }

        int OFF_minterms = PInfo[o].OFF_minterms;
        int pichart_words = PInfo[o].pichart_words;
        int *cov_word_index = PInfo[o].cov_word_index;
        uint64_t *shifted_cov_mask = PInfo[o].shifted_cov_mask;
        bool use_off_masks = PInfo[o].off_compat_masks != NULL;
        bool scalar_exact = !use_off_masks && !PInfo[o].off_has_dc;

        ThreadBuffer *ts = &buffer[tid][o];
        ThreadBuffer *workspace = &buffer[tid][0];
        size_t configuration_words =
            ((size_t)configuration_count + 63u) / 64u;
        size_t sparse_capacity = (size_t)OFF_minterms <
                (size_t)configuration_count
            ? (size_t)OFF_minterms
            : (size_t)configuration_count;
        size_t dense_bytes = configuration_words <= SIZE_MAX / 2u
            ? configuration_words * 2u * sizeof(uint64_t)
            : SIZE_MAX;
        size_t sparse_bytes = sparse_capacity <= SIZE_MAX / sizeof(int)
            ? sparse_capacity * sizeof(int)
            : SIZE_MAX;
        bool scalar_dense = scalar_exact &&
            (configuration_count <= 64 || dense_bytes <= sparse_bytes);

        size_t row_codes_needed = (size_t)ON_minterms;
        if (scalar_exact && !scalar_dense) {
            if (
                sparse_capacity >
                SIZE_MAX - row_codes_needed
            ) {
                fprintf(stderr, "Error: sparse OFF-set size overflow\n");
                return 1;
            }
            row_codes_needed += sparse_capacity;
        } else if (!use_off_masks && !scalar_exact) {
            if ((size_t)OFF_minterms > SIZE_MAX - row_codes_needed) {
                fprintf(stderr, "Error: decimal position size overflow\n");
                return 1;
            }
            row_codes_needed += (size_t)OFF_minterms;
        }
        if (!ensure_task_row_capacity(workspace, row_codes_needed)) {
            fprintf(stderr, "Error: Memory allocation failed for decimal position arrays\n");
            return 1;
        }
        int *decpos = workspace->task_row_codes;
        int *sparse_off_configurations = scalar_exact && !scalar_dense
            ? workspace->task_row_codes + ON_minterms
            : NULL;
        int *decneg = !use_off_masks && !scalar_exact
            ? workspace->task_row_codes + ON_minterms
            : NULL;

        bool use_seen_stamps = false;
        if (!scalar_exact) {
            use_seen_stamps = ensure_task_seen_capacity(
                workspace,
                (size_t)space_size
            );
        }

        uint64_t *dense_off_configurations = NULL;
        uint64_t *dense_on_configurations = NULL;
        if (scalar_dense) {
            if (
                configuration_words > SIZE_MAX / 2u ||
                !ensure_task_config_word_capacity(
                    workspace,
                    configuration_words * 2u
                )
            ) {
                fprintf(stderr, "Error: dense OFF-set allocation failed\n");
                return 1;
            }
            dense_off_configurations = workspace->task_config_bits;
            dense_on_configurations =
                workspace->task_config_bits + configuration_words;
            memset(
                dense_off_configurations,
                0,
                configuration_words * 2u * sizeof(uint64_t)
            );
        }
#ifdef CCUBES_TESTING
        ts->scalar_config_scratch_bytes = scalar_exact
            ? (scalar_dense ? dense_bytes : sparse_bytes)
            : 0u;
#endif

        int max_candidates = configuration_count < ON_minterms
            ? configuration_count
            : ON_minterms;
        if (!ensure_thread_buffer_capacity(
            ts,
            max_candidates,
            pichart_words,
            implicant_words
        )) {
            fprintf(stderr, "Error: candidate buffer allocation failed\n");
            return 1;
        }
        uint64_t *task_pichart_values = ts->pichart_values;
        int *task_found = &ts->found;

        // First pass: compute decpos for all ON rows (0 means invalid due to DC on selected inputs)
        // TODO: explore the potential of DC values compared to the OFF-set rows
        for (int r = 0; r < ON_minterms; r++) {
            if (
                use_shared_projection &&
                PInfo[o].ON_projection_ids
            ) {
                int row_id = PInfo[o].ON_projection_ids[r];
                decpos[r] = projection_has_dc[row_id]
                    ? 0
                    : projection_codes[row_id];
                continue;
            }

            int acc = 0;
            bool valid = true;

            for (int c = 0; c < k; c++) {
                int value = PInfo[o].ON_set[r * ninputs + tempk[c]];
                if (value == 0) {
                    valid = false;
                    break;
                }
                acc += value * mbase[c];
            }

            decpos[r] = valid ? acc : 0;
        }

        int sparse_off_count = 0;
        int *unique_off_rows = NULL;
        unsigned char *dc_off_rows = NULL;
        int off_count = 0;
        int exact_off_count = 0;

        if (scalar_exact) {
            for (int r = 0; r < OFF_minterms; ++r) {
#ifdef CCUBES_TESTING
                ts->scalar_off_rows_projected++;
#endif
                int configuration = 0;
                if (!compact_projected_configuration(
                    &PInfo[o].OFF_set[(size_t)r * (size_t)ninputs],
                    tempk,
                    k,
                    nofvalues,
                    &configuration
                )) {
                    fprintf(stderr, "Error: exact OFF row contains a dash\n");
                    return 1;
                }

                if (scalar_dense) {
                    size_t word = (size_t)configuration / 64u;
                    uint64_t bit = UINT64_C(1) << (configuration % 64);
                    if ((dense_off_configurations[word] & bit) == 0u) {
                        dense_off_configurations[word] |= bit;
                        exact_off_count++;
                        if (exact_off_count >= configuration_count) break;
                    }
                } else {
                    assert((size_t)sparse_off_count < sparse_capacity);
                    sparse_off_configurations[sparse_off_count++] =
                        configuration;
                }
            }

            if (!scalar_dense && sparse_off_count > 1) {
                qsort(
                    sparse_off_configurations,
                    (size_t)sparse_off_count,
                    sizeof(int),
                    compare_int_ascending
                );
                int unique = 1;
                for (int index = 1; index < sparse_off_count; ++index) {
                    if (
                        sparse_off_configurations[index] !=
                        sparse_off_configurations[unique - 1]
                    ) {
                        sparse_off_configurations[unique++] =
                            sparse_off_configurations[index];
                    }
                }
                sparse_off_count = unique;
            }

            if (exact_off_count >= configuration_count) {
                continue;
            }
        } else if (!use_off_masks) {
            unique_off_rows = (int *)malloc(
                (size_t)OFF_minterms * sizeof(int)
            );
            dc_off_rows = (unsigned char *)calloc(
                (size_t)OFF_minterms,
                sizeof(unsigned char)
            );
            if (!unique_off_rows || !dc_off_rows) {
                free(unique_off_rows);
                free(dc_off_rows);
                fprintf(stderr, "Error: wildcard OFF workspace allocation failed\n");
                return 1;
            }

            uint32_t off_seen_epoch = use_seen_stamps
                ? begin_task_seen_epoch(workspace)
                : 0;

            for (int r = 0; r < OFF_minterms; r++) {
                if (exact_off_count >= configuration_count) break;

                int acc = 0;
                bool has_dc = false;

#ifdef CCUBES_TESTING
                ts->scalar_off_rows_projected++;
#endif
                for (int c = 0; c < k; c++) {
                    int value =
                        PInfo[o].OFF_set[
                            r * ninputs + tempk[c]
                        ];
                    if (value == 0) has_dc = true;
                    acc += value * mbase[c];
                }

                decneg[r] = acc;
                dc_off_rows[r] = has_dc ? 1u : 0u;

                size_t off_index = (size_t)acc;
                assert(off_index < (size_t)space_size);

                // O(1) uniqueness check; fallback to O(n) if allocation failed
                if (
                    use_seen_stamps &&
                    off_index < (size_t)space_size
                ) {
                    if (
                        workspace->task_seen_stamps[off_index] ==
                        off_seen_epoch
                    ) {
                        continue;
                    }

                    workspace->task_seen_stamps[off_index] =
                        off_seen_epoch;
                    unique_off_rows[off_count++] = r;
                    if (!has_dc) exact_off_count++;
                } else {
                    bool unique = true;
                    for (int prev = 0; prev < off_count; prev++) {
                        if (decneg[unique_off_rows[prev]] == acc) {
                            unique = false;
                            break;
                        }
                    }

                    if (unique) {
                        unique_off_rows[off_count++] = r;
                        if (!has_dc) exact_off_count++;
                    }
                }
            }

            if (exact_off_count >= configuration_count) {
                free(unique_off_rows);
                free(dc_off_rows);
                continue;
            }
        }

        int possible_rows[ON_minterms];
        int found = 0;

        // Use a visited set keyed by normalized decpos (0..space_size - 1) to skip duplicates
        uint32_t pos_seen_epoch = use_seen_stamps
            ? begin_task_seen_epoch(workspace)
            : 0;

        for (int r = 0; r < ON_minterms; r++) {
            if (found >= configuration_count) break;
            if (decpos[r] == 0) continue;   // invalid row (has DC in selected inputs)

            int on_configuration = -1;
            if (scalar_exact) {
                if (!compact_projected_configuration(
                    &PInfo[o].ON_set[(size_t)r * (size_t)ninputs],
                    tempk,
                    k,
                    nofvalues,
                    &on_configuration
                )) {
                    continue;
                }
            }

            size_t on_index = (size_t)(decpos[r] - mbase_sum);
            assert(on_index < (size_t)space_size);

            if (scalar_dense) {
                size_t word = (size_t)on_configuration / 64u;
                uint64_t bit = UINT64_C(1) << (on_configuration % 64);
                if ((dense_on_configurations[word] & bit) != 0u) {
                    continue;
                }
                dense_on_configurations[word] |= bit;
            } else if (
                use_seen_stamps &&
                on_index < (size_t)space_size
            ) {
                if (
                    workspace->task_seen_stamps[on_index] ==
                    pos_seen_epoch
                ) {
                    continue; // accepted or rejected assignment already examined
                }
                /*
                 * Validity depends on the projected assignment, not on which
                 * ON row supplied it. Mark it before OFF validation so a
                 * rejected assignment is cached as well as an accepted one.
                 */
                workspace->task_seen_stamps[on_index] =
                    pos_seen_epoch;
            }
            if (
                !scalar_dense &&
                (
                    !use_seen_stamps ||
                    on_index >= (size_t)space_size
                )
            ) {
                // O(n) fallback: check previously selected rows for duplicate decpos
                bool duplicate = false;

                for (int prev = 0; prev < found; prev++) {
                    if (decpos[possible_rows[prev]] == decpos[r]) {
                        duplicate = true;
                        break;
                    }
                }

                if (duplicate) continue;
            }

#ifdef CCUBES_TESTING
            ts->validation_attempts++;
#endif

            // check if the row is different from any OFF-set row
            bool valid_row = true;
            if (use_off_masks) {
                const int *on_row = &PInfo[o].ON_set[
                    (size_t)r * (size_t)ninputs
                ];
                valid_row = !candidate_matches_wildcard_off(
                    &PInfo[o],
                    on_row,
                    tempk,
                    k
                );
            } else if (scalar_exact) {
                if (scalar_dense) {
                    size_t word = (size_t)on_configuration / 64u;
                    uint64_t bit =
                        UINT64_C(1) << (on_configuration % 64);
                    valid_row =
                        (dense_off_configurations[word] & bit) == 0u;
                } else {
                    valid_row = !sorted_int_contains(
                        sparse_off_configurations,
                        sparse_off_count,
                        on_configuration
                    );
                }
            } else {
                for (int roff = 0; roff < off_count; roff++) {
                    bool different = false;
                    if (dc_off_rows[unique_off_rows[roff]]) {
                        for (int c = 0; c < k; c++) {
                            int v_ON = PInfo[o].ON_set[
                                r * ninputs + tempk[c]
                            ];
                            int v_OFF = PInfo[o].OFF_set[
                                unique_off_rows[roff] * ninputs +
                                tempk[c]
                            ];

                            if (v_OFF != 0 && v_OFF != v_ON) {
                                different = true;
                                break;
                            }
                        }
                    } else {
                        different =
                            decpos[r] != decneg[unique_off_rows[roff]];
                    }
                    if (!different) {
                        valid_row = false;
                        break;
                    }
                }
            }

            if (!valid_row) continue;

            possible_rows[found++] = r;
            max_found++;

            if (found >= configuration_count) {
                break; // also guard after increment
            }
        }

        free(unique_off_rows);
        free(dc_off_rows);

        PICoverageIndex *generation_index = coverage_indices
            ? &coverage_indices[o]
            : NULL;
        int generalizing_removals[k];
        int generalizing_removal_count = generation_index
            ? subsumption_index_find_generalizing_removals(
                &generation_index->subsumption_index,
                fixed_bits,
                tempk,
                k,
                word_index,
                shifted_mask,
                generalizing_removals
            )
            : 0;

        for (int f = 0; f < found; f++) {
            // using bit shifting, store the fixed bits and value bits
            uint64_t value_bits[implicant_words];

            for (int w = 0; w < implicant_words; w++) {
                value_bits[w] = 0ULL;
            }

            for (int c = 0; c < k; c++) {
                int value = PInfo[o].ON_set[possible_rows[f] * ninputs + tempk[c]] - 1;
                // set the relevant bits
                value_bits[word_index[tempk[c]]] |= ((uint64_t)value << bit_index[tempk[c]]);
            }

            if (
                generation_index &&
                generalizing_removal_count > 0 &&
                subsumption_index_has_immediate_generalization(
                    &generation_index->subsumption_index,
                    fixed_bits,
                    value_bits,
                    tempk,
                    k,
                    generalizing_removals,
                    generalizing_removal_count,
                    word_index,
                    shifted_mask
                )
            ) {
                atomic_fetch_add_explicit(
                    &generation_index->subsumption_rejections,
                    1,
                    memory_order_relaxed
                );
                continue;
            }

            uint64_t pichart_values[pichart_words];
            for (int w = 0; w < pichart_words; w++) {
                pichart_values[w] = 0ULL;
            }

            int covsum = 0;
            for (int r = 0; r < ON_minterms; r++) {
                if (decpos[r] == decpos[possible_rows[f]]) {
                    pichart_values[cov_word_index[r]] |= shifted_cov_mask[r];
                    covsum++;
                }
            }

            // add everything to the temporary / task storage objects

            for (int w = 0; w < pichart_words; w++) {
                // the dereference operator in *task_found has precedence
                // over the multiplication operator *
                task_pichart_values[*task_found * pichart_words + w] = pichart_values[w];
            }

            ts->decpos[*task_found] = decpos[possible_rows[f]];
            ts->covsum[*task_found] = covsum;

            for (int w = 0; w < implicant_words; w++) {
                ts->fixed_bits[*task_found * implicant_words + w] = fixed_bits[w];
                ts->value_bits[*task_found * implicant_words + w] = value_bits[w];
            }

            (*task_found)++;

        } // end of found loop
    } // end of outputs loop

    // Identify unique PIs across all outputs, and determine which are shared
    // across multiple outputs. This is done by creating a map of unique PIs
    // and counting how many outputs each unique PI belongs to.

    if (max_found > 0) {
        // matrices with max_found columns and noutputs rows
        int *output_map = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int));
        int *covsum_map = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int));
        int *found_map  = (int *) calloc((size_t)max_found * (size_t)noutputs, sizeof(int)); // the f index within the output vector

        // single vectors
        int *uniquePIs    = (int *) calloc((size_t)max_found, sizeof(int));
        int *shared_count = (int *) calloc((size_t)max_found, sizeof(int));

        int counter = 0; // counter for the unique PIs

        for (int o = 0; o < noutputs; o++) {
            DBG_TRACE_BLOCK {
                // if (buffer[tid][o].found > 0) {
                //     fprintf(debug_out, "Output %d, found PIs:", o + 1);
                // }
            }

            for (int f = 0; f < buffer[tid][o].found; f++) {
                DBG_TRACE_BLOCK {
                    // fprintf(debug_out, " %d", buffer[tid][o].decpos[f]);
                }

                bool unique = true;

                for (int u = 0; u < counter; u++) {
                    if (buffer[tid][o].decpos[f] == uniquePIs[u]) {
                        output_map[u * noutputs + shared_count[u]] = o;
                        found_map[u * noutputs + shared_count[u]] = f;
                        covsum_map[u * noutputs + shared_count[u]] = buffer[tid][o].covsum[f];
                        shared_count[u]++;
                        unique = false;
                        break;
                    }
                }

                if (unique) {
                    uniquePIs[counter] = buffer[tid][o].decpos[f];
                    output_map[counter * noutputs + 0] = o;
                    found_map[counter * noutputs + 0] = f;
                    covsum_map[counter * noutputs + 0] = buffer[tid][o].covsum[f];
                    shared_count[counter]++;
                    counter++;
                }
            }

            DBG_TRACE_BLOCK {
                // if (buffer[tid][o].found > 0) {
                //     fprintf(debug_out, "\n");
                // }
            }
        }

        DBG_TRACE_BLOCK {
            // for (int o = 0; o < noutputs; o++) {
            //     fprintf(debug_out, "Output %d, found PIs: %d\n", o + 1, PInfo[o].task_found);
            //     for (int f = 0; f < PInfo[o].task_found; f++) {
            //         fprintf(debug_out, "  PI %d: decpos = %d, covsum = %d\n",
            //                 f + 1,
            //                 PInfo[o].task_decpos[f],
            //                 PInfo[o].task_covsum[f]);
            //         fprintf(debug_out, "    fixed bits: ");
            //         for (int w = 0; w < implicant_words; w++) {
            //             fprintf(debug_out, "%llu ", PInfo[o].task_fixed_bits[f * implicant_words + w]);
            //         }
            //         fprintf(debug_out, "\n    value bits: ");
            //         for (int w = 0; w < implicant_words; w++) {
            //             fprintf(debug_out, "%llu ", PInfo[o].task_value_bits[f * implicant_words + w]);
            //         }
            //         fprintf(debug_out, "\n");
            //     }
            // }

            // for (int u = 0; u < counter; u++) {
            //     fprintf(debug_out, "Unique PI %d: decpos = %d, shared_count = %d, outputs: ",
            //             u + 1,
            //             uniquePIs[u],
            //             shared_count[u]);
            //     for (int s = 0; s < shared_count[u]; s++) {
            //         fprintf(debug_out, "%d ", output_map[u * max_found + s] + 1);
            //     }
            //     fprintf(debug_out, "\n");
            // }
        }

        for (int u = 0; u < counter; u++) {
            for (int s = 0; s < shared_count[u]; s++) {
                int f = found_map[u * noutputs + s];
                int u_covsum = covsum_map[u * noutputs + s];

                // the output this PI belongs to
                int o = output_map[u * noutputs + s];
                bool shareable = shared_count[u] > 1;
                bool shareable_prime = !shareable || projected_cube_is_prime(
                    &PInfo[o],
                    ninputs,
                    tempk,
                    k,
                    &buffer[tid][o].value_bits[(size_t)f * (size_t)implicant_words],
                    word_index,
                    bit_index
                );

                int ON_minterms = PInfo[o].ON_minterms;
                int pichart_words = PInfo[o].pichart_words;

                uint64_t *task_pichart_values = buffer[tid][o].pichart_values;

                // Sanitize covsum bounds to avoid OOB on k_last_index
                if (u_covsum < 1) u_covsum = 1;
                if (u_covsum > ON_minterms) u_covsum = ON_minterms;

                bool redundant = !shareable_prime;
                PICoverageIndex *coverage_index = coverage_indices
                    ? &coverage_indices[o]
                    : NULL;

                /*
                 * Buckets up to covsum contain only coverages no larger than
                 * this candidate. A stored coverage can dominate the
                 * candidate only when the two bitsets are equal. Use the
                 * level index for that exact lookup instead of scanning every
                 * earlier PI while holding the output lock.
                 *
                 * The index contains only completed levels and is immutable
                 * during the worker pass, so this lookup is contention-free.
                 * Current-level duplicates are removed by finalize_pi_level()
                 * after all workers finish.
                 */
                if (
                    !shareable &&
                    !redundant &&
                    coverage_index
                ) {
                    redundant = coverage_index_lookup_words(
                        coverage_index,
                        &task_pichart_values[
                            (size_t)f * (size_t)pichart_words
                        ]
                    ) >= 0;
                }

                if (redundant) continue;

                uint64_t lock_wait_start = 0, lock_held_start = 0;
                if (lock_stats_active) lock_wait_start = lock_stats_now_ns();
                if (output_locks) ccubes_mutex_lock(&output_locks[o]);
                if (lock_stats_active) {
                    lock_held_start = lock_stats_now_ns();
                }

                    int *covered = PInfo[o].covered;
                    uint64_t *pichart_pos = PInfo[o].pichart_pos;
                    uint64_t *implicants_pos = PInfo[o].implicants_pos;
                    uint64_t *implicants_val = PInfo[o].implicants_val;
                    int *estimPI = &PInfo[o].estimPI;
                    int *foundPI = &PInfo[o].foundPI;
                    int *shared = PInfo[o].shared;
                    int *covsum = PInfo[o].covsum;

                    // Ensure capacity before writing the next PI
                    if ((*foundPI + 1) > *estimPI) {
                        resize((void**)&pichart_pos,    TYPE_UINT64, increase, *estimPI, pichart_words);
                        resize((void**)&implicants_pos, TYPE_UINT64, increase, *estimPI, implicant_words);
                        resize((void**)&implicants_val, TYPE_UINT64, increase, *estimPI, implicant_words);
                        resize((void**)&shared,         TYPE_INT,    increase, *estimPI, 1);
                        resize((void**)&covsum,         TYPE_INT,    increase, *estimPI, 1);
                        resize((void**)&covered,        TYPE_INT,    increase, *estimPI, 1);

                        // Update the PInfo structure pointers after resize
                        PInfo[o].pichart_pos = pichart_pos;
                        PInfo[o].implicants_pos = implicants_pos;
                        PInfo[o].implicants_val = implicants_val;
                        PInfo[o].shared = shared;
                        PInfo[o].covsum = covsum;
                        PInfo[o].covered = covered;

                        *estimPI += increase;

                        DBG_TRACE_BLOCK {
                            (*multiplier)++;
                            printf("%dx", *multiplier);
                        }
                    }

                    // push the PI information to the global arrays

                    for (int w = 0; w < implicant_words; w++) {
                        implicants_pos[(*foundPI) * implicant_words + w] = fixed_bits[w];
                        implicants_val[(*foundPI) * implicant_words + w] = buffer[tid][o].value_bits[f * implicant_words + w];
                    }

                    // populate the coverage matrix (bit-packed; one copy,
                    // not one per row as the dense chart used to require)
                    for (int w = 0; w < pichart_words; w++) {
                        pichart_pos[(*foundPI) * pichart_words + w] =
                            buffer[tid][o].pichart_values[f * pichart_words + w];
                    }

                    shared[*foundPI] = shared_count[u] - 1;
                    if (*max_shared < shared[*foundPI]) {
                        *max_shared = shared[*foundPI];
                    }
                    covsum[*foundPI] = u_covsum;

                    (*foundPI)++;

                if (lock_stats_active) {
                    lock_stats_record(
                        tid, o,
                        lock_held_start - lock_wait_start,
                        lock_stats_now_ns() - lock_held_start
                    );
                }
                if (output_locks) ccubes_mutex_unlock(&output_locks[o]);
            }
        }

        free(output_map);
        free(covsum_map);
        free(found_map);
        free(uniquePIs);
        free(shared_count);
    }

    // reset temporary task objects
    for (int o = 0; o < noutputs; o++) {
        // Only reset the logical count; we overwrite used slots on the next iteration.
        buffer[tid][o].found = 0;
    }

    return 0;
}
