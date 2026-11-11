/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_native_reduce_expand.h"

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "progressive_validation.h"
#include "ccubes_threads.h"

bool progressive_native_reduce_expand_should_continue(
    int inserted_columns,
    int variants_attempted
) {
    if (inserted_columns < 0 || variants_attempted <= 0) return false;
    return
        (uint64_t)(unsigned int)inserted_columns * UINT64_C(100) >=
        (uint64_t)(unsigned int)variants_attempted * UINT64_C(40);
}

static uint64_t monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static bool deadline_reached(uint64_t deadline) {
    return deadline > 0 && monotonic_nanoseconds() >= deadline;
}

static uint64_t mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) *
        UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) *
        UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static uint64_t low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static bool pattern_touches_off(
    const WeightedMMCSPreparedRows *prepared,
    const int *pattern
) {
    for (int word = 0; word < prepared->off_column_words; ++word) {
        const int remaining = prepared->off_rows - word * 64;
        uint64_t matching = low_bits_mask(remaining);
        for (int input = 0; input < prepared->ninputs; ++input) {
            const int value = pattern[input];
            if (value == 0) continue;
            uint64_t ones = prepared->off_columns[
                (size_t)input * (size_t)prepared->off_column_words +
                (size_t)word
            ];
            matching &= value == 2 ? ones : ~ones;
            if (!matching) break;
        }
        if (matching) return true;
    }
    return false;
}

static int private_rows(
    const uint64_t *coverage,
    int words,
    int cube,
    int rows,
    const int *cover_counts,
    uint64_t *private_bits
) {
    memset(private_bits, 0, (size_t)words * sizeof(uint64_t));
    int count = 0;
    for (int word = 0; word < words; ++word) {
        uint64_t bits = coverage[
            (size_t)cube * (size_t)words + (size_t)word
        ];
        while (bits) {
            const int bit = __builtin_ctzll(bits);
            const int row = word * 64 + bit;
            if (row < rows && cover_counts[row] == 1) {
                private_bits[word] |= UINT64_C(1) << bit;
                count++;
            }
            bits &= bits - 1u;
        }
    }
    return count;
}

static int private_ones(
    const WeightedMMCSPreparedRows *prepared,
    const uint64_t *private_bits,
    int input
) {
    int ones = 0;
    const uint64_t *column = &prepared->on_columns[
        (size_t)input * (size_t)prepared->on_column_words
    ];
    for (int word = 0; word < prepared->on_column_words; ++word) {
        ones += __builtin_popcountll(private_bits[word] & column[word]);
    }
    return ones;
}

static void order_literals(
    int *order,
    int count,
    int pass,
    int cube,
    int variant
) {
    for (int index = 1; index < count; ++index) {
        const int input = order[index];
        const uint64_t key = mix64(
            (uint64_t)(unsigned int)(pass + 1) *
                UINT64_C(0xd6e8feb86659fd93) ^
            (uint64_t)(unsigned int)(cube + 1) *
                UINT64_C(0xa0761d6478bd642f) ^
            (uint64_t)(unsigned int)(variant + 1) *
                UINT64_C(0xe7037ed1a0b428db) ^
            (uint64_t)(unsigned int)(input + 1)
        );
        int position = index;
        while (position > 0) {
            const int prior = order[position - 1];
            const uint64_t prior_key = mix64(
                (uint64_t)(unsigned int)(pass + 1) *
                    UINT64_C(0xd6e8feb86659fd93) ^
                (uint64_t)(unsigned int)(cube + 1) *
                    UINT64_C(0xa0761d6478bd642f) ^
                (uint64_t)(unsigned int)(variant + 1) *
                    UINT64_C(0xe7037ed1a0b428db) ^
                (uint64_t)(unsigned int)(prior + 1)
            );
            if (prior_key < key || (prior_key == key && prior < input)) {
                break;
            }
            order[position] = prior;
            position--;
        }
        order[position] = input;
    }
}

static uint64_t literal_mismatch_bits(
    const WeightedMMCSPreparedRows *prepared,
    int input,
    int value,
    int word
) {
    const uint64_t ones = prepared->off_columns[
        (size_t)input * (size_t)prepared->off_column_words +
        (size_t)word
    ];
    const int remaining = prepared->off_rows - word * 64;
    return (value == 2 ? ~ones : ones) & low_bits_mask(remaining);
}

static void initialize_off_mismatch_counts(
    const WeightedMMCSPreparedRows *prepared,
    const int *pattern,
    uint64_t *planes,
    int plane_count
) {
    const int words = prepared->off_column_words;
    memset(
        planes,
        0,
        (size_t)plane_count * (size_t)words * sizeof(*planes)
    );
    for (int input = 0; input < prepared->ninputs; ++input) {
        const int value = pattern[input];
        if (value == 0) continue;
        for (int word = 0; word < words; ++word) {
            uint64_t carry = literal_mismatch_bits(
                prepared, input, value, word
            );
            for (int plane = 0; plane < plane_count && carry; ++plane) {
                uint64_t *counter = &planes[
                    (size_t)plane * (size_t)words + (size_t)word
                ];
                const uint64_t next_carry = *counter & carry;
                *counter ^= carry;
                carry = next_carry;
            }
        }
    }
}

static bool remove_literal_incremental(
    const WeightedMMCSPreparedRows *prepared,
    int input,
    int value,
    uint64_t *planes,
    int plane_count
) {
    const int words = prepared->off_column_words;
    for (int word = 0; word < words; ++word) {
        const uint64_t bits = literal_mismatch_bits(
            prepared, input, value, word
        );
        uint64_t exactly_one = planes[word];
        for (int plane = 1; plane < plane_count; ++plane) {
            exactly_one &= ~planes[
                (size_t)plane * (size_t)words + (size_t)word
            ];
        }
        if (exactly_one & bits) return false;
    }
    for (int word = 0; word < words; ++word) {
        uint64_t borrow = literal_mismatch_bits(
            prepared, input, value, word
        );
        for (int plane = 0; plane < plane_count && borrow; ++plane) {
            uint64_t *counter = &planes[
                (size_t)plane * (size_t)words + (size_t)word
            ];
            const uint64_t next_borrow = ~*counter & borrow;
            *counter ^= borrow;
            borrow = next_borrow;
        }
    }
    return true;
}

typedef struct {
    uint64_t *private_bits;
    uint64_t *reduced_mismatch_planes;
    uint64_t *off_mismatch_planes;
    int *current;
    int *reduced;
    int *order;
} ReduceExpandScratch;

typedef struct {
    bool started;
    int ready_variants;
    ProgressiveNativeMasterStatus status;
    ProgressiveNativeReduceExpandStats stats;
} ReduceExpandCubeResult;

typedef struct {
    const WeightedMMCSPreparedRows *prepared;
    const uint64_t *incumbent_coverage;
    const int *cover_counts;
    const int *solution;
    int cardinality;
    int first_cube;
    int pass;
    int variants;
    int max_plane_count;
    uint64_t deadline;
    ReduceExpandScratch *scratch;
    ReduceExpandCubeResult *results;
    int *patterns;
    uint64_t *coverage;
    unsigned char *changed;
} ReduceExpandContext;

static void construct_reduce_expand_cube(
    ReduceExpandContext *context,
    int job,
    int worker
) {
    ReduceExpandCubeResult *result = &context->results[job];
    ProgressiveNativeReduceExpandStats *stats = &result->stats;
    if (deadline_reached(context->deadline)) return;
    result->started = true;
    result->status = PROGRESSIVE_NATIVE_MASTER_OK;
    stats->cubes_examined = 1;
    const WeightedMMCSPreparedRows *prepared = context->prepared;
    const int cube = context->first_cube + job;
    const int ninputs = prepared->ninputs;
    const int words = prepared->on_column_words;
    ReduceExpandScratch *scratch = &context->scratch[worker];
    const int private_count = private_rows(
        context->incumbent_coverage, words, cube, prepared->on_rows,
        context->cover_counts, scratch->private_bits
    );
    if (private_count <= 0) return;
    stats->cubes_with_private_rows = 1;
    int current_support = 0;
    int reduced_support = 0;
    for (int input = 0; input < ninputs; ++input) {
        scratch->current[input] = context->solution[
            (size_t)input * (size_t)context->cardinality + (size_t)cube
        ];
        current_support += scratch->current[input] != 0;
        const int ones = private_ones(prepared, scratch->private_bits, input);
        scratch->reduced[input] = ones == private_count
            ? 2 : (ones == 0 ? 1 : 0);
        if (scratch->current[input] != 0 &&
            scratch->reduced[input] != scratch->current[input]) {
            result->status = PROGRESSIVE_NATIVE_MASTER_INVALID;
            return;
        }
        reduced_support += scratch->reduced[input] != 0;
    }
    if (reduced_support > current_support) {
        stats->reduced_cores = 1;
        stats->literals_added_by_reduce =
            (uint64_t)(reduced_support - current_support);
    }
    int plane_count = 1;
    while (
        plane_count < context->max_plane_count &&
        (UINT64_C(1) << plane_count) <=
            (uint64_t)(unsigned int)reduced_support
    ) {
        plane_count++;
    }
    /* Counters for this reduced core are shared by its variant orders.
     * Keep the scalar behavior for a caller-supplied illegal core.
     */
    const bool legal_reduced = !pattern_touches_off(prepared, scratch->reduced);
    const size_t active_mismatch_bytes = (size_t)plane_count *
        (size_t)prepared->off_column_words * sizeof(uint64_t);
    if (legal_reduced) {
        initialize_off_mismatch_counts(
            prepared, scratch->reduced, scratch->reduced_mismatch_planes,
            plane_count
        );
    }
    for (int variant = 0; variant < context->variants; ++variant) {
        if (deadline_reached(context->deadline)) {
            stats->deadline_reached = true;
            break;
        }
        stats->variants_attempted++;
        const size_t slot = (size_t)job * (size_t)context->variants +
            (size_t)variant;
        int *candidate = context->patterns + slot * (size_t)ninputs;
        uint64_t *coverage = context->coverage + slot * (size_t)words;
        memcpy(candidate, scratch->reduced, (size_t)ninputs * sizeof(int));
        int ordered = 0;
        for (int input = 0; input < ninputs; ++input) {
            if (candidate[input] != 0) scratch->order[ordered++] = input;
        }
        order_literals(scratch->order, ordered, context->pass, cube, variant);
        if (legal_reduced) {
            memcpy(
                scratch->off_mismatch_planes, scratch->reduced_mismatch_planes,
                active_mismatch_bytes
            );
        }
        for (int index = 0; index < ordered; ++index) {
            const int input = scratch->order[index];
            const int saved = candidate[input];
            if (legal_reduced) {
                if (remove_literal_incremental(
                    prepared, input, saved, scratch->off_mismatch_planes,
                    plane_count
                )) {
                    candidate[input] = 0;
                }
            } else {
                candidate[input] = 0;
                if (pattern_touches_off(prepared, candidate)) {
                    candidate[input] = saved;
                }
            }
        }
        if (!progressive_pattern_coverage_prepared(prepared, candidate, coverage)) {
            result->status = PROGRESSIVE_NATIVE_MASTER_INVALID;
            return;
        }
        context->changed[slot] = memcmp(
            candidate, scratch->current, (size_t)ninputs * sizeof(int)
        ) != 0;
        result->ready_variants++;
    }
}

static void construct_reduce_expand_range(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker,
    int worker_count,
    void *data
) {
    (void)worker_count;
    ReduceExpandContext *context = (ReduceExpandContext *)data;
    for (uint64_t job = start; job < end; job += stride) {
        construct_reduce_expand_cube(context, (int)job, worker);
    }
}

static void destroy_reduce_expand_scratch(ReduceExpandScratch *scratch, int workers) {
    if (!scratch) return;
    for (int worker = 0; worker < workers; ++worker) {
        free(scratch[worker].private_bits);
        free(scratch[worker].reduced_mismatch_planes);
        free(scratch[worker].off_mismatch_planes);
        free(scratch[worker].current);
        free(scratch[worker].reduced);
        free(scratch[worker].order);
    }
    free(scratch);
}

ProgressiveNativeMasterStatus progressive_native_reduce_expand_generate(
    const WeightedMMCSPreparedRows *prepared,
    ProgressiveNativeMaster *master,
    int cardinality,
    const int *solution,
    int pass,
    int variants,
    uint64_t deadline,
    ProgressiveNativeReduceExpandStats *stats
) {
    ProgressiveNativeReduceExpandStats local;
    memset(&local, 0, sizeof(local));
    const uint64_t started = monotonic_nanoseconds();
    if (stats) *stats = local;
    if (
        !prepared || !master || !solution || cardinality <= 0 ||
        pass < 0 || variants <= 0 || prepared->ninputs <= 0 ||
        prepared->on_rows <= 0 || prepared->on_column_words <= 0 ||
        prepared->off_column_words <= 0 ||
        prepared->ninputs != master->ninputs ||
        prepared->on_rows != master->rows ||
        prepared->on_column_words != master->words ||
        prepared->off_rows <= 0 || !prepared->on_columns ||
        !prepared->off_columns
    ) {
        return PROGRESSIVE_NATIVE_MASTER_INVALID;
    }
    const int ninputs = prepared->ninputs;
    const int rows = prepared->on_rows;
    const int words = prepared->on_column_words;
    int workers = ccubes_thread_count();
    if (workers > cardinality) workers = cardinality;
    if (workers < 1) workers = 1;
    /* Keep scratch and unpublished candidates bounded. Insert each completed
     * batch in original cube/variant order, independently of scheduling.
     */
    const int batch_capacity = cardinality < workers * 2
        ? cardinality : workers * 2;
    if ((size_t)variants > SIZE_MAX / (size_t)batch_capacity ||
        (size_t)batch_capacity * (size_t)variants >
            SIZE_MAX / ((size_t)ninputs * sizeof(int)) ||
        (size_t)batch_capacity * (size_t)variants >
            SIZE_MAX / ((size_t)words * sizeof(uint64_t))) {
        return PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;
    }
    const size_t slots = (size_t)batch_capacity * (size_t)variants;
    int max_plane_count = 1;
    while (
        max_plane_count < 31 &&
        (UINT64_C(1) << max_plane_count) <= (uint64_t)(unsigned int)ninputs
    ) {
        max_plane_count++;
    }
    const size_t mismatch_bytes = (size_t)max_plane_count *
        (size_t)prepared->off_column_words * sizeof(uint64_t);
    uint64_t *incumbent_coverage = (uint64_t *)calloc(
        (size_t)cardinality * (size_t)words, sizeof(uint64_t)
    );
    int *cover_counts = (int *)calloc((size_t)rows, sizeof(int));
    ReduceExpandScratch *scratch = (ReduceExpandScratch *)calloc(
        (size_t)workers, sizeof(*scratch)
    );
    ReduceExpandCubeResult *results = (ReduceExpandCubeResult *)calloc(
        (size_t)batch_capacity, sizeof(*results)
    );
    int *patterns = (int *)malloc(slots * (size_t)ninputs * sizeof(int));
    uint64_t *coverage = (uint64_t *)malloc(
        slots * (size_t)words * sizeof(uint64_t)
    );
    unsigned char *changed = (unsigned char *)malloc(slots);
    ProgressiveNativeMasterStatus status = PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;
    if (!incumbent_coverage || !cover_counts || !scratch || !results ||
        !patterns || !coverage || !changed) goto done;
    for (int worker = 0; worker < workers; ++worker) {
        scratch[worker].private_bits = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));
        scratch[worker].reduced_mismatch_planes = (uint64_t *)malloc(mismatch_bytes);
        scratch[worker].off_mismatch_planes = (uint64_t *)malloc(mismatch_bytes);
        scratch[worker].current = (int *)malloc((size_t)ninputs * sizeof(int));
        scratch[worker].reduced = (int *)malloc((size_t)ninputs * sizeof(int));
        scratch[worker].order = (int *)malloc((size_t)ninputs * sizeof(int));
        if (!scratch[worker].private_bits || !scratch[worker].reduced_mismatch_planes ||
            !scratch[worker].off_mismatch_planes || !scratch[worker].current ||
            !scratch[worker].reduced || !scratch[worker].order) goto done;
    }
    status = PROGRESSIVE_NATIVE_MASTER_OK;
    for (int cube = 0; cube < cardinality; ++cube) {
        for (int input = 0; input < ninputs; ++input) {
            scratch[0].current[input] = solution[
                (size_t)input * (size_t)cardinality + (size_t)cube
            ];
        }
        uint64_t *column_coverage = incumbent_coverage + (size_t)cube * (size_t)words;
        if (!progressive_pattern_coverage_prepared(
            prepared, scratch[0].current, column_coverage
        )) {
            status = PROGRESSIVE_NATIVE_MASTER_INVALID;
            goto done;
        }
        for (int word = 0; word < words; ++word) {
            uint64_t bits = column_coverage[word];
            while (bits) {
                const int row = word * 64 + __builtin_ctzll(bits);
                if (row < rows) cover_counts[row]++;
                bits &= bits - 1;
            }
        }
    }
    ReduceExpandContext context = {
        .prepared = prepared, .incumbent_coverage = incumbent_coverage,
        .cover_counts = cover_counts, .solution = solution,
        .cardinality = cardinality, .pass = pass, .variants = variants,
        .max_plane_count = max_plane_count, .deadline = deadline,
        .scratch = scratch, .results = results, .patterns = patterns,
        .coverage = coverage, .changed = changed
    };
    for (int first = 0; first < cardinality;) {
        if (deadline_reached(deadline)) {
            local.deadline_reached = true;
            break;
        }
        const int remaining = cardinality - first;
        const int jobs = remaining < batch_capacity ? remaining : batch_capacity;
        memset(results, 0, (size_t)jobs * sizeof(*results));
        context.first_cube = first;
        if (!ccubes_parallel_for(
            0, (uint64_t)jobs, workers, false,
            construct_reduce_expand_range, &context
        )) {
            status = PROGRESSIVE_NATIVE_MASTER_OUT_OF_MEMORY;
            goto done;
        }
        for (int job = 0; job < jobs; ++job) {
            const ReduceExpandCubeResult *result = &results[job];
            if (!result->started) {
                local.deadline_reached = true;
                goto done;
            }
            local.cubes_examined++;
            local.cubes_with_private_rows += result->stats.cubes_with_private_rows;
            local.reduced_cores += result->stats.reduced_cores;
            local.literals_added_by_reduce += result->stats.literals_added_by_reduce;
            for (int variant = 0; variant < result->ready_variants; ++variant) {
                const size_t slot = (size_t)job * (size_t)variants + (size_t)variant;
                local.variants_attempted++;
                local.legal_candidates++;
                local.changed_geometries += changed[slot];
                bool inserted = false;
                status = progressive_native_master_insert(
                    master, coverage + slot * (size_t)words,
                    patterns + slot * (size_t)ninputs, &inserted
                );
                if (status != PROGRESSIVE_NATIVE_MASTER_OK) goto done;
                if (inserted) local.inserted_columns++;
                else local.duplicate_columns++;
            }
            /* Only the contiguous completed prefix is publishable. Later
             * jobs may have finished first; their candidates are discarded
             * if an earlier job stopped or failed.
             */
            local.variants_attempted += result->stats.variants_attempted -
                result->ready_variants;
            if (result->status != PROGRESSIVE_NATIVE_MASTER_OK) {
                status = result->status;
                goto done;
            }
            if (result->stats.deadline_reached) {
                local.deadline_reached = true;
                goto done;
            }
        }
        first += jobs;
    }
done:
    local.elapsed_nanoseconds = monotonic_nanoseconds() - started;
    if (stats) *stats = local;
    free(incumbent_coverage); free(cover_counts);
    destroy_reduce_expand_scratch(scratch, workers);
    free(results); free(patterns); free(coverage); free(changed);
    return status;
}
