/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "pool_selection.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXACT_COMBINATION_LIMIT 2000000ULL

typedef struct {
    int output;
    int column;
    int output_count;
    int last_output;
} CubeRecord;

typedef struct {
    const PIstorage *pinfo;
    int implicant_words;
    int *slots;
    size_t table_size;
    CubeRecord *records;
    int count;
    int capacity;
} CubeUniverse;

typedef struct {
    int source_index;
    int len;
    int *cube_ids;
    int potential;
} PoolCandidate;

typedef struct {
    int count;
    PoolCandidate *candidate;
} OutputPool;

typedef struct {
    int count;
    int len;
    int pool_count;
    bool incumbent_only;
    bool append_incumbent;
    bool current_fallback;
} CandidateShape;

typedef struct {
    const OutputPool *outputs;
    const int *order;
    int active_count;
    int cube_count;
    int *refcount;
    int *current_choice;
    int *best_choice;
    int best_union;
} ExactSearch;

static uint64_t cube_hash(
    const PIstorage *pinfo,
    int output,
    int column,
    int implicant_words
) {
    const uint64_t offset = 1469598103934665603ULL;
    const uint64_t prime = 1099511628211ULL;
    uint64_t hash = offset;

    const uint64_t *pos = &pinfo[output].implicants_pos[
        (size_t)column * (size_t)implicant_words
    ];
    const uint64_t *val = &pinfo[output].implicants_val[
        (size_t)column * (size_t)implicant_words
    ];

    for (int word = 0; word < implicant_words; ++word) {
        hash ^= pos[word];
        hash *= prime;
        hash ^= val[word];
        hash *= prime;
    }
    return hash;
}

static bool cube_equal(
    const PIstorage *pinfo,
    int output_a,
    int column_a,
    int output_b,
    int column_b,
    int implicant_words
) {
    const uint64_t *pos_a = &pinfo[output_a].implicants_pos[
        (size_t)column_a * (size_t)implicant_words
    ];
    const uint64_t *val_a = &pinfo[output_a].implicants_val[
        (size_t)column_a * (size_t)implicant_words
    ];
    const uint64_t *pos_b = &pinfo[output_b].implicants_pos[
        (size_t)column_b * (size_t)implicant_words
    ];
    const uint64_t *val_b = &pinfo[output_b].implicants_val[
        (size_t)column_b * (size_t)implicant_words
    ];

    return
        memcmp(pos_a, pos_b, (size_t)implicant_words * sizeof(uint64_t)) == 0 &&
        memcmp(val_a, val_b, (size_t)implicant_words * sizeof(uint64_t)) == 0;
}

static bool universe_init(
    CubeUniverse *universe,
    const PIstorage *pinfo,
    int implicant_words,
    size_t expected
) {
    if (!universe || !pinfo || implicant_words <= 0) return false;

    memset(universe, 0, sizeof(*universe));
    universe->pinfo = pinfo;
    universe->implicant_words = implicant_words;

    if (expected < 8u) expected = 8u;
    if (expected > (size_t)INT_MAX) return false;

    size_t table_size = 16u;
    while (table_size < expected * 2u) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }

    universe->slots = (int *)malloc(table_size * sizeof(int));
    universe->records = (CubeRecord *)malloc(expected * sizeof(CubeRecord));
    if (!universe->slots || !universe->records) {
        free(universe->slots);
        free(universe->records);
        memset(universe, 0, sizeof(*universe));
        return false;
    }

    for (size_t i = 0; i < table_size; ++i) universe->slots[i] = -1;
    universe->table_size = table_size;
    universe->capacity = (int)expected;
    return true;
}

static void universe_destroy(CubeUniverse *universe) {
    if (!universe) return;
    free(universe->slots);
    free(universe->records);
    memset(universe, 0, sizeof(*universe));
}

static int universe_add(
    CubeUniverse *universe,
    int output,
    int column
) {
    if (
        !universe || !universe->slots || !universe->records ||
        output < 0 || column < 0 || column >= universe->pinfo[output].foundPI
    ) {
        return -1;
    }

    uint64_t hash = cube_hash(
        universe->pinfo,
        output,
        column,
        universe->implicant_words
    );
    size_t mask = universe->table_size - 1u;
    size_t slot = (size_t)(hash & (uint64_t)mask);

    while (universe->slots[slot] >= 0) {
        int id = universe->slots[slot];
        CubeRecord *record = &universe->records[id];
        if (
            cube_equal(
                universe->pinfo,
                output,
                column,
                record->output,
                record->column,
                universe->implicant_words
            )
        ) {
            if (record->last_output != output) {
                record->last_output = output;
                record->output_count++;
            }
            return id;
        }
        slot = (slot + 1u) & mask;
    }

    if (universe->count >= universe->capacity) return -1;
    int id = universe->count++;
    universe->records[id] = (CubeRecord){
        .output = output,
        .column = column,
        .output_count = 1,
        .last_output = output
    };
    universe->slots[slot] = id;
    return id;
}

static void free_output_pools(OutputPool *outputs, int noutputs) {
    if (!outputs) return;
    for (int output = 0; output < noutputs; ++output) {
        for (int p = 0; p < outputs[output].count; ++p) {
            free(outputs[output].candidate[p].cube_ids);
        }
        free(outputs[output].candidate);
    }
    free(outputs);
}

static CandidateShape candidate_shape(const PIstorage *pi) {
    CandidateShape shape = {0};
    if (!pi || pi->solmin <= 0) return shape;

    bool incumbent_valid =
        pi->prevsolmin > 0 &&
        pi->prevsolmin <= pi->ON_minterms &&
        pi->previndices != NULL;

    if (incumbent_valid && pi->prevsolmin < pi->solmin) {
        shape.count = 1;
        shape.len = pi->prevsolmin;
        shape.incumbent_only = true;
        return shape;
    }

    shape.len = pi->solmin;
    shape.pool_count = pi->pool_count;
    if (shape.pool_count > 0) {
        shape.count = shape.pool_count;
        if (incumbent_valid && pi->prevsolmin == pi->solmin) {
            shape.count++;
            shape.append_incumbent = true;
        }
    } else {
        shape.count = 1;
        if (incumbent_valid && pi->prevsolmin == pi->solmin) {
            shape.incumbent_only = true;
        } else {
            shape.current_fallback = true;
        }
    }
    return shape;
}

static int candidate_added_cubes(
    const PoolCandidate *candidate,
    const int *refcount
) {
    int added = 0;
    for (int i = 0; i < candidate->len; ++i) {
        if (refcount[candidate->cube_ids[i]] == 0) added++;
    }
    return added;
}

static int add_candidate(
    const PoolCandidate *candidate,
    int *refcount
) {
    int added = 0;
    for (int i = 0; i < candidate->len; ++i) {
        int id = candidate->cube_ids[i];
        if (refcount[id]++ == 0) added++;
    }
    return added;
}

static int remove_candidate(
    const PoolCandidate *candidate,
    int *refcount
) {
    int removed = 0;
    for (int i = 0; i < candidate->len; ++i) {
        int id = candidate->cube_ids[i];
        if (--refcount[id] == 0) removed++;
    }
    return removed;
}

static int selection_union(
    const OutputPool *outputs,
    int noutputs,
    const int *choice,
    int cube_count,
    int *refcount
) {
    memset(refcount, 0, (size_t)cube_count * sizeof(int));
    int distinct = 0;
    for (int output = 0; output < noutputs; ++output) {
        if (outputs[output].count <= 0 || choice[output] < 0) continue;
        distinct += add_candidate(
            &outputs[output].candidate[choice[output]],
            refcount
        );
    }
    return distinct;
}

static int coordinate_descent(
    const OutputPool *outputs,
    int noutputs,
    int cube_count,
    int start_mode,
    int *choice,
    int *refcount
) {
    for (int output = 0; output < noutputs; ++output) {
        int count = outputs[output].count;
        if (count <= 0) {
            choice[output] = -1;
            continue;
        }

        if (start_mode == 0) {
            choice[output] = 0;
        } else if (start_mode == 1) {
            int best = 0;
            for (int p = 1; p < count; ++p) {
                if (outputs[output].candidate[p].potential >
                    outputs[output].candidate[best].potential) {
                    best = p;
                }
            }
            choice[output] = best;
        } else {
            choice[output] = output % count;
        }
    }

    int distinct = selection_union(
        outputs,
        noutputs,
        choice,
        cube_count,
        refcount
    );

    for (int pass = 0; pass < 32; ++pass) {
        bool changed = false;
        for (int output = 0; output < noutputs; ++output) {
            int count = outputs[output].count;
            if (count <= 1) continue;

            int old = choice[output];
            distinct -= remove_candidate(
                &outputs[output].candidate[old],
                refcount
            );

            int best = old;
            int best_union = INT_MAX;
            int best_potential = INT_MIN;
            for (int p = 0; p < count; ++p) {
                int candidate_union = distinct + candidate_added_cubes(
                    &outputs[output].candidate[p],
                    refcount
                );
                int potential = outputs[output].candidate[p].potential;
                if (
                    candidate_union < best_union ||
                    (candidate_union == best_union && potential > best_potential) ||
                    (candidate_union == best_union && potential == best_potential && p < best)
                ) {
                    best = p;
                    best_union = candidate_union;
                    best_potential = potential;
                }
            }

            choice[output] = best;
            distinct += add_candidate(
                &outputs[output].candidate[best],
                refcount
            );
            if (best != old) changed = true;
        }
        if (!changed) break;
    }

    return distinct;
}

static void exact_search_recurse(
    ExactSearch *search,
    int depth,
    int current_union
) {
    if (depth == search->active_count) {
        if (current_union < search->best_union) {
            search->best_union = current_union;
            memcpy(
                search->best_choice,
                search->current_choice,
                (size_t)search->active_count * sizeof(int)
            );
        }
        return;
    }

    int output = search->order[depth];
    const OutputPool *pool = &search->outputs[output];
    for (int p = 0; p < pool->count; ++p) {
        const PoolCandidate *candidate = &pool->candidate[p];
        int next_union = current_union + candidate_added_cubes(
            candidate,
            search->refcount
        );
        if (next_union >= search->best_union) continue;

        add_candidate(candidate, search->refcount);
        search->current_choice[depth] = p;
        exact_search_recurse(search, depth + 1, next_union);
        remove_candidate(candidate, search->refcount);
    }
}

static void sort_output_order(
    const OutputPool *outputs,
    int *order,
    int count
) {
    for (int i = 1; i < count; ++i) {
        int value = order[i];
        int j = i - 1;
        while (j >= 0) {
            int left_count = outputs[order[j]].count;
            int value_count = outputs[value].count;
            int left_len = outputs[order[j]].candidate[0].len;
            int value_len = outputs[value].candidate[0].len;
            if (
                left_count < value_count ||
                (left_count == value_count && left_len >= value_len)
            ) {
                break;
            }
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = value;
    }
}

bool select_joint_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats
) {
    if (!pinfo || noutputs <= 0 || implicant_words <= 0 || !chosen_pool) {
        return false;
    }

    if (stats) memset(stats, 0, sizeof(*stats));
    for (int output = 0; output < noutputs; ++output) chosen_pool[output] = -1;

    size_t occurrence_count = 0u;
    int active_outputs = 0;
    int total_pool_solutions = 0;
    for (int output = 0; output < noutputs; ++output) {
        CandidateShape shape = candidate_shape(&pinfo[output]);
        if (shape.count <= 0 || shape.len <= 0) continue;
        if ((size_t)shape.count > (SIZE_MAX - occurrence_count) / (size_t)shape.len) return false;
        occurrence_count += (size_t)shape.count * (size_t)shape.len;
        active_outputs++;
        total_pool_solutions += shape.count;
    }

    if (active_outputs == 0) return true;

    CubeUniverse universe;
    if (!universe_init(&universe, pinfo, implicant_words, occurrence_count)) {
        return false;
    }

    OutputPool *outputs = (OutputPool *)calloc(
        (size_t)noutputs,
        sizeof(OutputPool)
    );
    if (!outputs) {
        universe_destroy(&universe);
        return false;
    }

    bool ok = true;
    for (int output = 0; output < noutputs && ok; ++output) {
        CandidateShape shape = candidate_shape(&pinfo[output]);
        if (shape.count <= 0 || shape.len <= 0) continue;

        outputs[output].count = shape.count;
        outputs[output].candidate = (PoolCandidate *)calloc(
            (size_t)shape.count,
            sizeof(PoolCandidate)
        );
        if (!outputs[output].candidate) {
            ok = false;
            break;
        }

        for (int p = 0; p < shape.count && ok; ++p) {
            PoolCandidate *candidate = &outputs[output].candidate[p];
            bool incumbent =
                shape.incumbent_only ||
                (shape.append_incumbent && p == shape.pool_count);
            bool fallback = shape.current_fallback;

            candidate->source_index = (incumbent || fallback) ? -1 : p;
            candidate->len = shape.len;
            candidate->cube_ids = (int *)malloc(
                (size_t)shape.len * sizeof(int)
            );
            if (!candidate->cube_ids) {
                ok = false;
                break;
            }

            const int *solution = incumbent ?
                pinfo[output].previndices :
                (fallback ? pinfo[output].indices : pinfo[output].pool_solutions[p]);
            if (!solution) {
                ok = false;
                break;
            }

            for (int i = 0; i < shape.len; ++i) {
                int column = solution[i];
                int id = universe_add(&universe, output, column);
                if (id < 0) {
                    ok = false;
                    break;
                }
                candidate->cube_ids[i] = id;
            }
        }
    }

    if (!ok) {
        free_output_pools(outputs, noutputs);
        universe_destroy(&universe);
        return false;
    }

    int pool_shared = 0;
    for (int id = 0; id < universe.count; ++id) {
        if (universe.records[id].output_count > 1) pool_shared++;
    }
    for (int output = 0; output < noutputs; ++output) {
        for (int p = 0; p < outputs[output].count; ++p) {
            PoolCandidate *candidate = &outputs[output].candidate[p];
            for (int i = 0; i < candidate->len; ++i) {
                int id = candidate->cube_ids[i];
                candidate->potential += universe.records[id].output_count - 1;
            }
        }
    }

    int *choice = (int *)malloc((size_t)noutputs * sizeof(int));
    int *trial = (int *)malloc((size_t)noutputs * sizeof(int));
    int *refcount = (int *)calloc((size_t)universe.count, sizeof(int));
    int *order = (int *)malloc((size_t)active_outputs * sizeof(int));
    if (!choice || !trial || !refcount || !order) {
        free(choice);
        free(trial);
        free(refcount);
        free(order);
        free_output_pools(outputs, noutputs);
        universe_destroy(&universe);
        return false;
    }

    int best_union = INT_MAX;
    for (int mode = 0; mode < 3; ++mode) {
        int trial_union = coordinate_descent(
            outputs,
            noutputs,
            universe.count,
            mode,
            trial,
            refcount
        );
        if (trial_union < best_union) {
            best_union = trial_union;
            memcpy(choice, trial, (size_t)noutputs * sizeof(int));
        }
    }

    int order_count = 0;
    uint64_t combinations = 1u;
    bool exact = true;
    for (int output = 0; output < noutputs; ++output) {
        if (outputs[output].count <= 0) continue;
        order[order_count++] = output;
        if (
            combinations > EXACT_COMBINATION_LIMIT /
                (uint64_t)outputs[output].count
        ) {
            exact = false;
        } else if (exact) {
            combinations *= (uint64_t)outputs[output].count;
        }
    }

    if (exact) {
        sort_output_order(outputs, order, order_count);
        int *current_order_choice = (int *)calloc(
            (size_t)order_count,
            sizeof(int)
        );
        int *best_order_choice = (int *)calloc(
            (size_t)order_count,
            sizeof(int)
        );
        if (!current_order_choice || !best_order_choice) {
            free(current_order_choice);
            free(best_order_choice);
            free(choice);
            free(trial);
            free(refcount);
            free(order);
            free_output_pools(outputs, noutputs);
            universe_destroy(&universe);
            return false;
        }

        for (int depth = 0; depth < order_count; ++depth) {
            best_order_choice[depth] = choice[order[depth]];
        }
        memset(refcount, 0, (size_t)universe.count * sizeof(int));
        ExactSearch search = {
            .outputs = outputs,
            .order = order,
            .active_count = order_count,
            .cube_count = universe.count,
            .refcount = refcount,
            .current_choice = current_order_choice,
            .best_choice = best_order_choice,
            .best_union = best_union
        };
        exact_search_recurse(&search, 0, 0);
        best_union = search.best_union;
        for (int depth = 0; depth < order_count; ++depth) {
            choice[order[depth]] = best_order_choice[depth];
        }
        free(current_order_choice);
        free(best_order_choice);
    }

    int selected_distinct = selection_union(
        outputs,
        noutputs,
        choice,
        universe.count,
        refcount
    );
    int selected_shared = 0;
    for (int id = 0; id < universe.count; ++id) {
        if (refcount[id] > 1) selected_shared++;
    }

    int connections = 0;
    for (int output = 0; output < noutputs; ++output) {
        if (outputs[output].count <= 0 || choice[output] < 0) continue;
        PoolCandidate *candidate = &outputs[output].candidate[choice[output]];
        connections += candidate->len;
        chosen_pool[output] = candidate->source_index;
    }

    if (stats) {
        stats->active_outputs = active_outputs;
        stats->total_pool_solutions = total_pool_solutions;
        stats->retained_shared_cubes = -1;
        stats->pool_shared_cubes = pool_shared;
        stats->output_connections = connections;
        stats->selected_distinct_cubes = selected_distinct;
        stats->selected_shared_cubes = selected_shared;
        stats->sharing_savings = connections - selected_distinct;
        stats->selection_exact = exact;
    }

    free(choice);
    free(trial);
    free(refcount);
    free(order);
    free_output_pools(outputs, noutputs);
    universe_destroy(&universe);
    return true;
}

int count_retained_shared_cubes(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words
) {
    if (!pinfo || noutputs <= 0 || implicant_words <= 0) return -1;

    size_t total = 0u;
    for (int output = 0; output < noutputs; ++output) {
        if ((size_t)pinfo[output].foundPI > SIZE_MAX - total) return -1;
        total += (size_t)pinfo[output].foundPI;
    }
    if (total == 0u) return 0;

    CubeUniverse universe;
    if (!universe_init(&universe, pinfo, implicant_words, total)) return -1;

    for (int output = 0; output < noutputs; ++output) {
        for (int column = 0; column < pinfo[output].foundPI; ++column) {
            if (universe_add(&universe, output, column) < 0) {
                universe_destroy(&universe);
                return -1;
            }
        }
    }

    int shared = 0;
    for (int id = 0; id < universe.count; ++id) {
        if (universe.records[id].output_count > 1) shared++;
    }
    universe_destroy(&universe);
    return shared;
}

void clear_output_solution_pool(PIstorage *pi) {
    if (!pi || !pi->pool_solutions) return;
    for (int p = 0; p < pi->pool_count; ++p) {
        free(pi->pool_solutions[p]);
        pi->pool_solutions[p] = NULL;
    }
    pi->pool_count = 0;
}
