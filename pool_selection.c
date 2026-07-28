/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "pool_selection.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EXACT_COMBINATION_LIMIT 2000000ULL
#define POOL_VALUE_EPSILON 1e-12

typedef struct {
    int output;
    int column;
    int literal_count;
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
    int literal_count;
    int potential;
    int value_rank;
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
    int best_literals;
    const CubeRecord *records;
} ExactSearch;

static int cube_literal_count(
    const PIstorage *pinfo,
    int output,
    int column,
    int implicant_words
) {
    const uint64_t pair_low_bits = UINT64_C(0x5555555555555555);
    const uint64_t *position = &pinfo[output].implicants_pos[
        (size_t)column * (size_t)implicant_words
    ];
    int literals = 0;

    for (int word = 0; word < implicant_words; ++word) {
        /*
         * Each input occupies one two-bit lane.  A dash has lane 00; either
         * value of a fixed input has at least one bit set.  Fold each lane
         * onto its low bit, then count occupied lanes.
         */
        uint64_t occupied =
            (position[word] | (position[word] >> 1u)) &
            pair_low_bits;
        literals += __builtin_popcountll(occupied);
    }
    return literals;
}

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
        .literal_count = cube_literal_count(
            universe->pinfo,
            output,
            column,
            universe->implicant_words
        ),
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

static int candidate_overlap(
    const PoolCandidate *a,
    const PoolCandidate *b
) {
    int overlap = 0;
    for (int i = 0; i < a->len; ++i) {
        for (int j = 0; j < b->len; ++j) {
            if (a->cube_ids[i] == b->cube_ids[j]) {
                overlap++;
                break;
            }
        }
    }
    return overlap;
}

/*
 * Retain covers for their marginal compatibility with the candidate
 * landscapes of the other outputs.  For a retained family K_o, the facility
 * value is
 *
 *   F(K_o) = sum_{j != o} (1 / |P_j|)
 *              sum_{D in P_j} max_{C in K_o} |C intersect D|.
 *
 * Greedy marginal gain gives diminishing priority automatically. Candidate
 * zero and a solver-ranked safety seed are always preserved. Beyond the seed,
 * candidates with positive marginal value are ranked first; candidates with
 * an absolute cross-output cube match remain eligible because a higher-order
 * tuple may still use them. Only covers with no sharing opportunity are
 * discarded. Keep decisions for every output are computed before any pool is
 * compacted, so the score is symmetric with respect to the original
 * candidate landscape.
 */
static bool mark_valuable_candidates(
    OutputPool *outputs,
    int noutputs,
    bool ***keep_out,
    int **keep_count_out
) {
    bool **keep = (bool **)calloc((size_t)noutputs, sizeof(*keep));
    int *keep_count = (int *)calloc((size_t)noutputs, sizeof(*keep_count));
    if (!keep || !keep_count) {
        free(keep);
        free(keep_count);
        return false;
    }

    for (int output = 0; output < noutputs; ++output) {
        int count = outputs[output].count;
        if (count <= 0) continue;

        keep[output] = (bool *)calloc((size_t)count, sizeof(bool));
        if (!keep[output]) {
            for (int prior = 0; prior < output; ++prior) free(keep[prior]);
            free(keep);
            free(keep_count);
            return false;
        }

        keep[output][0] = true;
        outputs[output].candidate[0].value_rank = 0;
        keep_count[output] = 1;
        if (count == 1) continue;

        /*
         * Pool compaction is allowed to discard candidates with no sharing
         * opportunity, but the final lexicographic selector still needs the
         * cheapest literal realization of this output's tied cardinality.
         */
        int cheapest = 0;
        for (int p = 1; p < count; ++p) {
            if (
                outputs[output].candidate[p].literal_count <
                    outputs[output].candidate[cheapest].literal_count
            ) {
                cheapest = p;
            }
        }
        if (cheapest != 0) {
            keep[output][cheapest] = true;
            outputs[output].candidate[cheapest].value_rank = 1;
            keep_count[output]++;
        }

        int reference_count = 0;
        for (int other = 0; other < noutputs; ++other) {
            if (other != output) reference_count += outputs[other].count;
        }
        if (reference_count <= 0) continue;

        const PoolCandidate **reference = (const PoolCandidate **)malloc(
            (size_t)reference_count * sizeof(*reference)
        );
        double *reference_weight = (double *)malloc(
            (size_t)reference_count * sizeof(*reference_weight)
        );
        int *best_overlap = (int *)calloc(
            (size_t)reference_count,
            sizeof(*best_overlap)
        );
        if (!reference || !reference_weight || !best_overlap) {
            free(reference);
            free(reference_weight);
            free(best_overlap);
            for (int prior = 0; prior <= output; ++prior) free(keep[prior]);
            free(keep);
            free(keep_count);
            return false;
        }

        int ref = 0;
        for (int other = 0; other < noutputs; ++other) {
            if (other == output || outputs[other].count <= 0) continue;
            double weight = 1.0 / (double)outputs[other].count;
            for (int p = 0; p < outputs[other].count; ++p) {
                reference[ref] = &outputs[other].candidate[p];
                reference_weight[ref] = weight;
                best_overlap[ref] = candidate_overlap(
                    &outputs[output].candidate[0],
                    reference[ref]
                );
                ref++;
            }
        }

        while (
            keep_count[output] < count &&
            keep_count[output] < CCUBES_POOL_STORAGE_CAPACITY
        ) {
            int best_candidate = -1;
            double best_gain = 0.0;
            int best_potential = INT_MIN;

            for (int p = 1; p < count; ++p) {
                if (keep[output][p]) continue;

                double gain = 0.0;
                for (int r = 0; r < reference_count; ++r) {
                    int overlap = candidate_overlap(
                        &outputs[output].candidate[p],
                        reference[r]
                    );
                    if (overlap > best_overlap[r]) {
                        gain += reference_weight[r] *
                            (double)(overlap - best_overlap[r]);
                    }
                }

                int potential = outputs[output].candidate[p].potential;
                if (
                    gain > best_gain + POOL_VALUE_EPSILON ||
                    (
                        fabs(gain - best_gain) <= POOL_VALUE_EPSILON &&
                        gain > POOL_VALUE_EPSILON &&
                        (
                            potential > best_potential ||
                            (
                                potential == best_potential &&
                                (best_candidate < 0 || p < best_candidate)
                            )
                        )
                    )
                ) {
                    best_candidate = p;
                    best_gain = gain;
                    best_potential = potential;
                }
            }

            if (
                best_candidate < 0 ||
                best_gain <= POOL_VALUE_EPSILON
            ) {
                /*
                 * Preserve a small solver-ranked seed before applying the
                 * strict positive-marginal rule.  Whole-cover coordination
                 * can exploit co-occurrence patterns that a pairwise overlap
                 * surrogate does not see; the seed prevents that surrogate
                 * from collapsing every low-overlap output to one cover.
                 */
                if (
                    keep_count[output] <
                    CCUBES_POOL_SEED_CANDIDATES
                ) {
                    best_candidate = -1;
                    for (int p = 1; p < count; ++p) {
                        if (!keep[output][p]) {
                            best_candidate = p;
                            break;
                        }
                    }
                    if (best_candidate < 0) break;
                } else {
                    /*
                     * Marginal facility value is a ranking surrogate, not
                     * the final joint-union objective.  A cover with any
                     * exact cross-output cube match can still matter through
                     * a higher-order combination, so retain it within the
                     * hard 20-cover cap.  Only candidates with no sharing
                     * opportunity at all are safely discarded here.
                     */
                    best_candidate = -1;
                    best_potential = 0;
                    for (int p = 1; p < count; ++p) {
                        if (
                            !keep[output][p] &&
                            outputs[output].candidate[p].potential >
                                best_potential
                        ) {
                            best_candidate = p;
                            best_potential =
                                outputs[output].candidate[p].potential;
                        }
                    }
                    if (best_candidate < 0) break;
                }
            }

            keep[output][best_candidate] = true;
            outputs[output].candidate[best_candidate].value_rank =
                keep_count[output];
            keep_count[output]++;
            for (int r = 0; r < reference_count; ++r) {
                int overlap = candidate_overlap(
                    &outputs[output].candidate[best_candidate],
                    reference[r]
                );
                if (overlap > best_overlap[r]) best_overlap[r] = overlap;
            }
        }

        free(reference);
        free(reference_weight);
        free(best_overlap);
    }

    *keep_out = keep;
    *keep_count_out = keep_count;
    return true;
}

static void compact_valuable_candidates(
    OutputPool *outputs,
    int noutputs,
    bool **keep,
    const int *keep_count
) {
    for (int output = 0; output < noutputs; ++output) {
        int write = 0;
        for (int read = 0; read < outputs[output].count; ++read) {
            if (keep[output] && keep[output][read]) {
                if (write != read) {
                    outputs[output].candidate[write] =
                        outputs[output].candidate[read];
                    memset(
                        &outputs[output].candidate[read],
                        0,
                        sizeof(outputs[output].candidate[read])
                    );
                }
                write++;
            } else {
                free(outputs[output].candidate[read].cube_ids);
                outputs[output].candidate[read].cube_ids = NULL;
            }
        }
        outputs[output].count = keep_count[output];

        for (int i = 1; i < outputs[output].count; ++i) {
            PoolCandidate candidate = outputs[output].candidate[i];
            int position = i;
            while (
                position > 0 &&
                candidate.value_rank <
                    outputs[output].candidate[position - 1].value_rank
            ) {
                outputs[output].candidate[position] =
                    outputs[output].candidate[position - 1];
                position--;
            }
            outputs[output].candidate[position] = candidate;
        }
        free(keep[output]);
    }
    free(keep);
}

static CandidateShape candidate_shape(
    const PIstorage *pi,
    bool include_stopped
) {
    CandidateShape shape = {0};
    /*
    A stopped output has already committed its final indices.  Other outputs
    may continue to later PI levels, but their joint-pool pass must not reopen
    or replace this cover with a pool retained from an earlier boundary.
    */
    if (!pi || (!include_stopped && pi->stop_search) || pi->solmin <= 0) {
        return shape;
    }

    bool incumbent_valid =
        pi->prevsolmin > 0 &&
        pi->prevsolmin <= pi->ON_minterms &&
        pi->previndices != NULL;

    if (include_stopped && pi->stop_search) {
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
            shape.current_fallback = true;
        }
        return shape;
    }

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

static int candidate_added_literals(
    const PoolCandidate *candidate,
    const int *refcount,
    const CubeRecord *records
) {
    int added = 0;
    for (int i = 0; i < candidate->len; ++i) {
        int id = candidate->cube_ids[i];
        if (refcount[id] == 0) added += records[id].literal_count;
    }
    return added;
}

static int candidate_removed_literals(
    const PoolCandidate *candidate,
    const int *refcount,
    const CubeRecord *records
) {
    int removed = 0;
    for (int i = 0; i < candidate->len; ++i) {
        int id = candidate->cube_ids[i];
        if (refcount[id] == 1) removed += records[id].literal_count;
    }
    return removed;
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
    int *refcount,
    const CubeRecord *records,
    int *literal_count
) {
    memset(refcount, 0, (size_t)cube_count * sizeof(int));
    int distinct = 0;
    int literals = 0;
    for (int output = 0; output < noutputs; ++output) {
        if (outputs[output].count <= 0 || choice[output] < 0) continue;
        literals += candidate_added_literals(
            &outputs[output].candidate[choice[output]],
            refcount,
            records
        );
        distinct += add_candidate(
            &outputs[output].candidate[choice[output]],
            refcount
        );
    }
    if (literal_count) *literal_count = literals;
    return distinct;
}

static int coordinate_descent(
    const OutputPool *outputs,
    int noutputs,
    int cube_count,
    const CubeRecord *records,
    int start_mode,
    int *choice,
    int *refcount,
    int *literal_count
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

    int literals = 0;
    int distinct = selection_union(
        outputs,
        noutputs,
        choice,
        cube_count,
        refcount,
        records,
        &literals
    );

    for (int pass = 0; pass < 32; ++pass) {
        bool changed = false;
        for (int output = 0; output < noutputs; ++output) {
            int count = outputs[output].count;
            if (count <= 1) continue;

            int old = choice[output];
            literals -= candidate_removed_literals(
                &outputs[output].candidate[old],
                refcount,
                records
            );
            distinct -= remove_candidate(
                &outputs[output].candidate[old],
                refcount
            );

            int best = old;
            int best_union = INT_MAX;
            int best_literals = INT_MAX;
            int best_potential = INT_MIN;
            for (int p = 0; p < count; ++p) {
                int candidate_union = distinct + candidate_added_cubes(
                    &outputs[output].candidate[p],
                    refcount
                );
                int candidate_literals =
                    literals + candidate_added_literals(
                        &outputs[output].candidate[p],
                        refcount,
                        records
                    );
                int potential = outputs[output].candidate[p].potential;
                if (
                    candidate_union < best_union ||
                    (
                        candidate_union == best_union &&
                        candidate_literals < best_literals
                    ) ||
                    (
                        candidate_union == best_union &&
                        candidate_literals == best_literals &&
                        potential > best_potential
                    ) ||
                    (
                        candidate_union == best_union &&
                        candidate_literals == best_literals &&
                        potential == best_potential &&
                        p < best
                    )
                ) {
                    best = p;
                    best_union = candidate_union;
                    best_literals = candidate_literals;
                    best_potential = potential;
                }
            }

            choice[output] = best;
            literals += candidate_added_literals(
                &outputs[output].candidate[best],
                refcount,
                records
            );
            distinct += add_candidate(
                &outputs[output].candidate[best],
                refcount
            );
            if (best != old) changed = true;
        }
        if (!changed) break;
    }

    if (literal_count) *literal_count = literals;
    return distinct;
}

static void exact_search_recurse(
    ExactSearch *search,
    int depth,
    int current_union,
    int current_literals
) {
    if (depth == search->active_count) {
        if (
            current_union < search->best_union ||
            (
                current_union == search->best_union &&
                current_literals < search->best_literals
            )
        ) {
            search->best_union = current_union;
            search->best_literals = current_literals;
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
        int next_literals =
            current_literals + candidate_added_literals(
                candidate,
                search->refcount,
                search->records
            );
        if (
            next_union > search->best_union ||
            (
                next_union == search->best_union &&
                next_literals >= search->best_literals
            )
        ) {
            continue;
        }

        add_candidate(candidate, search->refcount);
        search->current_choice[depth] = p;
        exact_search_recurse(
            search,
            depth + 1,
            next_union,
            next_literals
        );
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

static bool select_joint_pool_solutions_impl(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats,
    bool include_stopped
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
        CandidateShape shape = candidate_shape(
            &pinfo[output],
            include_stopped
        );
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
        CandidateShape shape = candidate_shape(
            &pinfo[output],
            include_stopped
        );
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
                if (
                    candidate->literal_count >
                    INT_MAX - universe.records[id].literal_count
                ) {
                    ok = false;
                    break;
                }
                candidate->literal_count +=
                    universe.records[id].literal_count;
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

    bool **keep = NULL;
    int *keep_count = NULL;
    if (!mark_valuable_candidates(
        outputs,
        noutputs,
        &keep,
        &keep_count
    )) {
        free_output_pools(outputs, noutputs);
        universe_destroy(&universe);
        return false;
    }
    int valuable_pool_solutions = 0;
    for (int output = 0; output < noutputs; ++output) {
        valuable_pool_solutions += keep_count[output];
    }
    compact_valuable_candidates(
        outputs,
        noutputs,
        keep,
        keep_count
    );
    free(keep_count);

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
    int best_literals = INT_MAX;
    for (int mode = 0; mode < 3; ++mode) {
        int trial_literals = 0;
        int trial_union = coordinate_descent(
            outputs,
            noutputs,
            universe.count,
            universe.records,
            mode,
            trial,
            refcount,
            &trial_literals
        );
        if (
            trial_union < best_union ||
            (
                trial_union == best_union &&
                trial_literals < best_literals
            )
        ) {
            best_union = trial_union;
            best_literals = trial_literals;
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
            .best_union = best_union,
            .best_literals = best_literals,
            .records = universe.records
        };
        exact_search_recurse(&search, 0, 0, 0);
        best_union = search.best_union;
        best_literals = search.best_literals;
        for (int depth = 0; depth < order_count; ++depth) {
            choice[order[depth]] = best_order_choice[depth];
        }
        free(current_order_choice);
        free(best_order_choice);
    }

    int selected_literals = 0;
    int selected_distinct = selection_union(
        outputs,
        noutputs,
        choice,
        universe.count,
        refcount,
        universe.records,
        &selected_literals
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
        stats->generated_pool_solutions = total_pool_solutions;
        stats->valuable_pool_solutions = valuable_pool_solutions;
        stats->discarded_pool_solutions =
            total_pool_solutions - valuable_pool_solutions;
        stats->total_pool_solutions = total_pool_solutions;
        stats->retained_shared_cubes = -1;
        stats->pool_shared_cubes = pool_shared;
        stats->output_connections = connections;
        stats->selected_distinct_cubes = selected_distinct;
        stats->selected_input_literals = selected_literals;
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

bool select_joint_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats
) {
    return select_joint_pool_solutions_impl(
        pinfo,
        noutputs,
        implicant_words,
        chosen_pool,
        stats,
        false
    );
}

bool select_final_joint_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats
) {
    return select_joint_pool_solutions_impl(
        pinfo,
        noutputs,
        implicant_words,
        chosen_pool,
        stats,
        true
    );
}

bool measure_selected_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    PoolSelectionStats *stats
) {
    if (!pinfo || noutputs <= 0 || implicant_words <= 0 || !stats) {
        return false;
    }

    size_t connections = 0u;
    for (int output = 0; output < noutputs; ++output) {
        if (pinfo[output].solmin <= 0 || !pinfo[output].indices) continue;
        if ((size_t)pinfo[output].solmin > SIZE_MAX - connections) return false;
        connections += (size_t)pinfo[output].solmin;
    }
    if (connections == 0u) {
        stats->output_connections = 0;
        stats->selected_distinct_cubes = 0;
        stats->selected_input_literals = 0;
        stats->selected_shared_cubes = 0;
        stats->sharing_savings = 0;
        return true;
    }
    if (connections > (size_t)INT_MAX) return false;

    CubeUniverse universe;
    if (!universe_init(&universe, pinfo, implicant_words, connections)) {
        return false;
    }
    int *refcount = (int *)calloc(connections, sizeof(*refcount));
    if (!refcount) {
        universe_destroy(&universe);
        return false;
    }

    bool ok = true;
    for (int output = 0; output < noutputs && ok; ++output) {
        for (int i = 0; i < pinfo[output].solmin; ++i) {
            int id = universe_add(&universe, output, pinfo[output].indices[i]);
            if (id < 0) {
                ok = false;
                break;
            }
            refcount[id]++;
        }
    }

    if (ok) {
        int selected_shared = 0;
        int selected_literals = 0;
        for (int id = 0; id < universe.count; ++id) {
            if (refcount[id] > 1) selected_shared++;
            selected_literals += universe.records[id].literal_count;
        }
        stats->output_connections = (int)connections;
        stats->selected_distinct_cubes = universe.count;
        stats->selected_input_literals = selected_literals;
        stats->selected_shared_cubes = selected_shared;
        stats->sharing_savings = (int)connections - universe.count;
    }

    free(refcount);
    universe_destroy(&universe);
    return ok;
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
