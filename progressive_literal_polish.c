/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_literal_polish.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ccubes_threads.h"
#include "progressive_validation.h"

#define POLISH_MAX_PASSES 6

static uint64_t low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static int popcount_words(const uint64_t *bits, int words) {
    int total = 0;
    for (int word = 0; word < words; ++word) {
        total += __builtin_popcountll(bits[word]);
    }
    return total;
}

static uint64_t polish_monotonic_nanoseconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static double polish_monotonic_seconds(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static bool polish_checked_mul_size(
    size_t left,
    size_t right,
    size_t *result
) {
    if (!result || (left != 0 && right > SIZE_MAX / left)) return false;
    *result = left * right;
    return true;
}

static bool polish_add_allocation(
    size_t *total,
    size_t count,
    size_t item_size
) {
    size_t bytes = 0;
    if (
        !total ||
        !polish_checked_mul_size(count, item_size, &bytes) ||
        *total > SIZE_MAX - bytes
    ) {
        return false;
    }
    *total += bytes;
    return true;
}

/*
 * Minimum set cover over the OFF rows. Each candidate literal contributes the
 * set of OFF rows it excludes, so a cover of every OFF row is exactly a valid
 * cube support. Branching selects the first still-uncovered OFF row and tries
 * every literal excluding it, which is complete: any solution must contain one
 * of them.
 *
 * `remaining_stack` is private to one worker. Reusing one word slice per
 * recursion depth avoids an allocation at every search node; output 003
 * previously performed more than one hundred million such allocations. When
 * the configured memory envelope permits it, an optional per-depth active list
 * carries the still-relevant candidate indices: a candidate whose mask is
 * disjoint from the current remaining set can never cover a row in any
 * descendant, because remaining only shrinks, so it is dropped from the list
 * once detected. Without that optional scratch, every candidate is scanned in
 * the original order. The prune bound below is the
 * reach of the original search: depth + ceil(outstanding / reach) >= best,
 * where reach is the largest single-candidate coverage. That prunes exactly
 * when reach <= (outstanding - 1) / (best - depth - 1), the divisor being
 * positive by the earlier depth check. Scanning the candidates in order and
 * stopping as soon as one exceeds that threshold decides the same prune
 * without computing every coverage or the exact maximum, and the all-zero
 * case (reach <= 0) is covered by the same comparison. Visited nodes,
 * candidate order, and the returned best are therefore identical to the
 * original full computation.
 */
typedef struct {
    int off_words;
    int candidate_count;
    const uint64_t *masks;      /* candidate_count * off_words */
    int best;
    int *best_choice;
    int *current_choice;
    uint64_t *remaining_stack;  /* (candidate_count + 1) * off_words */
    int *active_stack;          /* (candidate_count + 1) * candidate_count */
    uint64_t nodes;
    uint64_t node_limit;
    bool limit_reached;
} PolishCoverSearch;

static void polish_cover_search(
    PolishCoverSearch *search,
    uint64_t *remaining,
    const int *active,
    int active_count,
    int depth
) {
    if (search->limit_reached) return;
    if (++search->nodes > search->node_limit) {
        search->limit_reached = true;
        return;
    }

    int uncovered = -1;
    for (int word = 0; word < search->off_words; ++word) {
        if (remaining[word]) {
            uncovered = word * 64 + __builtin_ctzll(remaining[word]);
            break;
        }
    }
    if (uncovered < 0) {
        search->best = depth;
        memcpy(
            search->best_choice,
            search->current_choice,
            (size_t)depth * sizeof(int)
        );
        return;
    }
    if (depth + 1 >= search->best) return;

    const int outstanding = popcount_words(remaining, search->off_words);
    const int threshold =
        (outstanding - 1) / (search->best - depth - 1);
    const bool use_active = search->active_stack != NULL;
    int *child_active = use_active
        ? &search->active_stack[
            (size_t)(depth + 1) * (size_t)search->candidate_count
        ]
        : NULL;
    int child_count = 0;
    bool can_improve = false;
    for (int ai = 0; ai < active_count; ++ai) {
        const int candidate = use_active ? active[ai] : ai;
        const uint64_t *mask = &search->masks[
            (size_t)candidate * (size_t)search->off_words
        ];
        int covered = 0;
        for (int word = 0; word < search->off_words; ++word) {
            covered += __builtin_popcountll(mask[word] & remaining[word]);
        }
        if (covered > threshold) {
            can_improve = true;
            /*
             * The exceeding candidate decides the prune; the unscanned
             * remainder keeps its unknown status and stays active.
             */
            while (use_active && ai < active_count) {
                child_active[child_count++] = active[ai++];
            }
            break;
        }
        if (use_active && covered > 0) {
            child_active[child_count++] = candidate;
        }
    }
    if (!can_improve) return;

    uint64_t *next = &search->remaining_stack[
        (size_t)(depth + 1) * (size_t)search->off_words
    ];
    const int word_index = uncovered / 64;
    const uint64_t row_bit = UINT64_C(1) << (uncovered % 64);
    const int branch_count = use_active
        ? child_count
        : search->candidate_count;
    for (int ai = 0; ai < branch_count; ++ai) {
        const int candidate = use_active ? child_active[ai] : ai;
        const uint64_t *mask = &search->masks[
            (size_t)candidate * (size_t)search->off_words
        ];
        if (!(mask[word_index] & row_bit)) continue;
        for (int word = 0; word < search->off_words; ++word) {
            next[word] = remaining[word] & ~mask[word];
        }
        search->current_choice[depth] = candidate;
        polish_cover_search(
            search,
            next,
            use_active ? child_active : NULL,
            branch_count,
            depth + 1
        );
        if (search->limit_reached) break;
    }
}

static int private_rows_from_counts(
    const uint64_t *own,
    const int *cover_counts,
    int on_rows,
    int on_words,
    uint64_t *private_rows
) {
    memset(private_rows, 0, (size_t)on_words * sizeof(uint64_t));
    int count = 0;
    for (int word = 0; word < on_words; ++word) {
        uint64_t bits = own[word];
        while (bits) {
            const int bit = __builtin_ctzll(bits);
            const int row = word * 64 + bit;
            if (row < on_rows && cover_counts[row] == 1) {
                private_rows[word] |= UINT64_C(1) << bit;
                count++;
            }
            bits &= bits - 1u;
        }
    }
    return count;
}

typedef struct {
    const WeightedMMCSPreparedRows *prepared;
    int cardinality;
    int ninputs;
    int on_rows;
    int on_words;
    int off_rows;
    int off_words;
    int off_stride;
    const int *solution;
    const uint64_t *coverage;
    const int *cover_counts;
    uint64_t node_limit;

    int *proposal_patterns;              /* cube-major */
    uint64_t *proposal_coverage;         /* cube-major */
    int *proposal_support;
    unsigned char *proposal_valid;
    unsigned char *proposal_limited;
    uint64_t *proposal_nodes;

    uint64_t *private_rows;              /* worker-major */
    uint64_t *masks;                     /* worker-major */
    uint64_t *remaining_stack;           /* worker-major */
    int *pattern;                        /* worker-major */
    int *agreement;                      /* worker-major */
    int *candidate_input;                /* worker-major */
    int *best_choice;                    /* worker-major */
    int *current_choice;                 /* worker-major */
    int *active_stack;                   /* worker-major */
    atomic_uint_fast64_t next_cube;      /* dynamic work claim */
} PolishProposalContext;

static void propose_cube(
    PolishProposalContext *context,
    int cube,
    int worker_id
) {
    const int ninputs = context->ninputs;
    const int on_words = context->on_words;
    const int off_words = context->off_words;
    const int off_stride = context->off_stride;
    uint64_t *private_rows = &context->private_rows[
        (size_t)worker_id * (size_t)on_words
    ];
    uint64_t *masks = &context->masks[
        (size_t)worker_id * (size_t)ninputs * (size_t)off_stride
    ];
    uint64_t *remaining_stack = &context->remaining_stack[
        (size_t)worker_id * ((size_t)ninputs + 1u) * (size_t)off_stride
    ];
    int *pattern = &context->pattern[(size_t)worker_id * (size_t)ninputs];
    int *agreement = &context->agreement[
        (size_t)worker_id * (size_t)ninputs
    ];
    int *candidate_input = &context->candidate_input[
        (size_t)worker_id * (size_t)ninputs
    ];
    int *best_choice = &context->best_choice[
        (size_t)worker_id * (size_t)ninputs
    ];
    int *current_choice = &context->current_choice[
        (size_t)worker_id * (size_t)ninputs
    ];
    int *active_stack = context->active_stack
        ? &context->active_stack[
            (size_t)worker_id * ((size_t)ninputs + 1u) * (size_t)ninputs
        ]
        : NULL;

    int support = 0;
    for (int input = 0; input < ninputs; ++input) {
        support += context->solution[
            (size_t)input * (size_t)context->cardinality + (size_t)cube
        ] != 0;
    }
    if (support <= 1) return;

    const uint64_t *own = &context->coverage[
        (size_t)cube * (size_t)on_words
    ];
    const int private_count = private_rows_from_counts(
        own,
        context->cover_counts,
        context->on_rows,
        on_words,
        private_rows
    );
    if (private_count <= 0) return;

    int candidates = 0;
    for (int input = 0; input < ninputs; ++input) {
        const uint64_t *column = &context->prepared->on_columns[
            (size_t)input * (size_t)on_words
        ];
        int ones = 0;
        for (int word = 0; word < on_words; ++word) {
            ones += __builtin_popcountll(private_rows[word] & column[word]);
        }
        if (ones == private_count) {
            agreement[input] = 2;
        } else if (ones == 0) {
            agreement[input] = 1;
        } else {
            agreement[input] = 0;
            continue;
        }
        uint64_t *mask = &masks[
            (size_t)candidates * (size_t)off_stride
        ];
        const uint64_t *off_column = &context->prepared->off_columns[
            (size_t)input * (size_t)off_words
        ];
        for (int word = 0; word < off_words; ++word) {
            uint64_t bits = off_column[word];
            if (agreement[input] == 2) bits = ~bits;
            const int rest = context->off_rows - word * 64;
            if (rest < 64) bits &= low_bits_mask(rest);
            mask[word] = bits;
        }
        candidate_input[candidates++] = input;
    }
    if (candidates <= 0) return;

    uint64_t *remaining = remaining_stack;
    for (int word = 0; word < off_words; ++word) {
        const int rest = context->off_rows - word * 64;
        remaining[word] = rest < 64 ? low_bits_mask(rest) : UINT64_MAX;
    }
    if (active_stack) {
        for (int candidate = 0; candidate < candidates; ++candidate) {
            active_stack[candidate] = candidate;
        }
    }

    PolishCoverSearch search = {
        .off_words = off_words,
        .candidate_count = candidates,
        .masks = masks,
        .best = support,
        .best_choice = best_choice,
        .current_choice = current_choice,
        .remaining_stack = remaining_stack,
        .active_stack = active_stack,
        .nodes = 0,
        .node_limit = context->node_limit,
        .limit_reached = false
    };
    polish_cover_search(
        &search,
        remaining,
        active_stack,
        candidates,
        0
    );
    context->proposal_nodes[cube] = search.nodes;
    context->proposal_limited[cube] = search.limit_reached ? 1 : 0;
    if (search.best >= support) return;

    memset(pattern, 0, (size_t)ninputs * sizeof(int));
    for (int pick = 0; pick < search.best; ++pick) {
        const int input = candidate_input[best_choice[pick]];
        pattern[input] = agreement[input];
    }
    uint64_t *candidate_coverage = &context->proposal_coverage[
        (size_t)cube * (size_t)on_words
    ];
    if (!progressive_pattern_coverage_prepared(
        context->prepared,
        pattern,
        candidate_coverage
    )) {
        return;
    }
    memcpy(
        &context->proposal_patterns[(size_t)cube * (size_t)ninputs],
        pattern,
        (size_t)ninputs * sizeof(int)
    );
    context->proposal_support[cube] = search.best;
    context->proposal_valid[cube] = 1;
}

static void polish_proposal_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)start;
    (void)end;
    (void)stride;
    (void)worker_count;
    PolishProposalContext *context = (PolishProposalContext *)data;
    /*
     * Cubes are independent and write disjoint per-cube proposal slots, so
     * the assignment of cubes to workers does not change any result.
     * Dynamic claiming evens out the heavy-tailed per-cube search cost
     * that a fixed strided partition would leave imbalanced.
     */
    for (;;) {
        uint64_t cube = atomic_fetch_add_explicit(
            &context->next_cube,
            1,
            memory_order_relaxed
        );
        if (cube >= (uint64_t)context->cardinality) break;
        propose_cube(context, (int)cube, worker_id);
    }
}

static void update_cover_counts(
    int *cover_counts,
    int on_rows,
    int on_words,
    const uint64_t *old_coverage,
    const uint64_t *new_coverage
) {
    for (int word = 0; word < on_words; ++word) {
        uint64_t removed = old_coverage[word] & ~new_coverage[word];
        while (removed) {
            const int row = word * 64 + __builtin_ctzll(removed);
            if (row < on_rows) cover_counts[row]--;
            removed &= removed - 1u;
        }
        uint64_t added = new_coverage[word] & ~old_coverage[word];
        while (added) {
            const int row = word * 64 + __builtin_ctzll(added);
            if (row < on_rows) cover_counts[row]++;
            added &= added - 1u;
        }
    }
}

bool progressive_literal_polish(
    const WeightedMMCSPreparedRows *prepared,
    int cardinality,
    int *solution,
    int requested_threads,
    uint64_t node_limit,
    uint64_t memory_budget_bytes,
    uint64_t deadline_nanoseconds,
    int pass_cap,
    double yield_gate,
    ProgressiveLiteralPolishStats *stats
) {
    if (stats) memset(stats, 0, sizeof(*stats));
    if (
        !prepared ||
        !solution ||
        cardinality <= 0 ||
        prepared->ninputs <= 0 ||
        prepared->on_rows <= 0 ||
        requested_threads <= 0
    ) {
        return true;
    }

    const double started = polish_monotonic_seconds();
    const int ninputs = prepared->ninputs;
    const int on_rows = prepared->on_rows;
    const int on_words = prepared->on_column_words;
    const int off_rows = prepared->off_rows;
    const int off_words = prepared->off_column_words;
    const int off_stride = off_words > 0 ? off_words : 1;
    int threads = ccubes_effective_thread_count(requested_threads);
    if (threads > cardinality) threads = cardinality;
    if (threads < 1) threads = 1;

    const size_t input_depths = (size_t)ninputs + 1u;
    size_t solution_values = 0;
    size_t coverage_words = 0;
    size_t masks_per_worker = 0;
    size_t remaining_per_worker = 0;
    size_t active_per_worker = 0;
    if (
        !polish_checked_mul_size(
            (size_t)ninputs,
            (size_t)cardinality,
            &solution_values
        ) ||
        !polish_checked_mul_size(
            (size_t)cardinality,
            (size_t)on_words,
            &coverage_words
        ) ||
        !polish_checked_mul_size(
            (size_t)ninputs,
            (size_t)off_stride,
            &masks_per_worker
        ) ||
        !polish_checked_mul_size(
            input_depths,
            (size_t)off_stride,
            &remaining_per_worker
        ) ||
        !polish_checked_mul_size(
            input_depths,
            (size_t)ninputs,
            &active_per_worker
        )
    ) {
        return false;
    }

    size_t fixed_bytes = 0;
    size_t per_worker_bytes = 0;
    size_t active_per_worker_bytes = 0;
    bool sizes_valid =
        polish_add_allocation(&fixed_bytes, solution_values, sizeof(int)) &&
        polish_add_allocation(&fixed_bytes, solution_values, sizeof(int)) &&
        polish_add_allocation(
            &fixed_bytes,
            coverage_words,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            coverage_words,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)on_rows,
            sizeof(int)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)cardinality,
            sizeof(int)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)cardinality,
            sizeof(unsigned char)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)cardinality,
            sizeof(unsigned char)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)cardinality,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &fixed_bytes,
            (size_t)on_words,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &per_worker_bytes,
            (size_t)on_words,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &per_worker_bytes,
            masks_per_worker,
            sizeof(uint64_t)
        ) &&
        polish_add_allocation(
            &per_worker_bytes,
            remaining_per_worker,
            sizeof(uint64_t)
        );
    for (int array = 0; array < 5 && sizes_valid; ++array) {
        sizes_valid = polish_add_allocation(
            &per_worker_bytes,
            (size_t)ninputs,
            sizeof(int)
        );
    }
    sizes_valid = sizes_valid && polish_add_allocation(
        &active_per_worker_bytes,
        active_per_worker,
        sizeof(int)
    );
    if (!sizes_valid) return false;

    if (memory_budget_bytes > 0) {
        const size_t budget = memory_budget_bytes > (uint64_t)SIZE_MAX
            ? SIZE_MAX
            : (size_t)memory_budget_bytes;
        if (fixed_bytes > budget || per_worker_bytes > budget - fixed_bytes) {
            return false;
        }
        size_t budget_threads = (budget - fixed_bytes) / per_worker_bytes;
        if (budget_threads < (size_t)threads) threads = (int)budget_threads;
        if (threads < 1) return false;
    }

    size_t worker_bytes = 0;
    if (!polish_checked_mul_size(
        (size_t)threads,
        per_worker_bytes,
        &worker_bytes
    ) || fixed_bytes > SIZE_MAX - worker_bytes) {
        return false;
    }
    const size_t base_allocated_bytes = fixed_bytes + worker_bytes;
    size_t active_list_bytes = 0;
    bool use_active_lists = memory_budget_bytes > 0 &&
        polish_checked_mul_size(
            (size_t)threads,
            active_per_worker_bytes,
            &active_list_bytes
        );
    if (use_active_lists) {
        const uint64_t base = (uint64_t)base_allocated_bytes;
        use_active_lists =
            base <= memory_budget_bytes &&
            active_list_bytes <= memory_budget_bytes - base;
    }
    if (!use_active_lists) active_list_bytes = 0;

    int *original_solution = (int *)malloc(solution_values * sizeof(int));
    uint64_t *coverage = (uint64_t *)calloc(
        coverage_words,
        sizeof(uint64_t)
    );
    int *cover_counts = (int *)calloc((size_t)on_rows, sizeof(int));
    int *proposal_patterns = (int *)calloc(
        solution_values,
        sizeof(int)
    );
    uint64_t *proposal_coverage = (uint64_t *)calloc(
        coverage_words,
        sizeof(uint64_t)
    );
    int *proposal_support = (int *)calloc(
        (size_t)cardinality,
        sizeof(int)
    );
    unsigned char *proposal_valid = (unsigned char *)calloc(
        (size_t)cardinality,
        sizeof(unsigned char)
    );
    unsigned char *proposal_limited = (unsigned char *)calloc(
        (size_t)cardinality,
        sizeof(unsigned char)
    );
    uint64_t *proposal_nodes = (uint64_t *)calloc(
        (size_t)cardinality,
        sizeof(uint64_t)
    );
    uint64_t *private_rows = (uint64_t *)malloc(
        (size_t)threads * (size_t)on_words * sizeof(uint64_t)
    );
    uint64_t *masks = (uint64_t *)malloc(
        (size_t)threads * (size_t)ninputs * (size_t)off_stride *
        sizeof(uint64_t)
    );
    uint64_t *remaining_stack = (uint64_t *)malloc(
        (size_t)threads * input_depths * (size_t)off_stride *
        sizeof(uint64_t)
    );
    int *pattern = (int *)malloc(
        (size_t)threads * (size_t)ninputs * sizeof(int)
    );
    int *agreement = (int *)malloc(
        (size_t)threads * (size_t)ninputs * sizeof(int)
    );
    int *candidate_input = (int *)malloc(
        (size_t)threads * (size_t)ninputs * sizeof(int)
    );
    int *best_choice = (int *)malloc(
        (size_t)threads * (size_t)ninputs * sizeof(int)
    );
    int *current_choice = (int *)malloc(
        (size_t)threads * (size_t)ninputs * sizeof(int)
    );
    int *active_stack = use_active_lists
        ? (int *)malloc(active_list_bytes)
        : NULL;
    if (use_active_lists && !active_stack) {
        use_active_lists = false;
        active_list_bytes = 0;
    }
    uint64_t *commit_private_rows = (uint64_t *)malloc(
        (size_t)on_words * sizeof(uint64_t)
    );
    if (
        !original_solution || !coverage || !cover_counts ||
        !proposal_patterns || !proposal_coverage || !proposal_support ||
        !proposal_valid || !proposal_limited || !proposal_nodes ||
        !private_rows || !masks || !remaining_stack || !pattern ||
        !agreement || !candidate_input || !best_choice ||
        !current_choice || !commit_private_rows
    ) {
        free(original_solution); free(coverage); free(cover_counts);
        free(proposal_patterns); free(proposal_coverage);
        free(proposal_support); free(proposal_valid);
        free(proposal_limited); free(proposal_nodes);
        free(private_rows); free(masks); free(remaining_stack);
        free(pattern); free(agreement); free(candidate_input);
        free(best_choice); free(current_choice); free(active_stack);
        free(commit_private_rows);
        return false;
    }
    memcpy(original_solution, solution, solution_values * sizeof(int));

    for (int cube = 0; cube < cardinality; ++cube) {
        int *worker_pattern = pattern;
        for (int input = 0; input < ninputs; ++input) {
            worker_pattern[input] = solution[
                (size_t)input * (size_t)cardinality + (size_t)cube
            ];
        }
        if (!progressive_pattern_coverage_prepared(
            prepared,
            worker_pattern,
            &coverage[(size_t)cube * (size_t)on_words]
        )) {
            memcpy(solution, original_solution, solution_values * sizeof(int));
            goto failed;
        }
    }
    for (int cube = 0; cube < cardinality; ++cube) {
        const uint64_t *own = &coverage[(size_t)cube * (size_t)on_words];
        for (int word = 0; word < on_words; ++word) {
            uint64_t bits = own[word];
            while (bits) {
                const int row = word * 64 + __builtin_ctzll(bits);
                if (row < on_rows) cover_counts[row]++;
                bits &= bits - 1u;
            }
        }
    }

    int literals_before = 0;
    for (size_t value = 0; value < solution_values; ++value) {
        literals_before += solution[value] != 0;
    }

    int passes = 0;
    int reduced = 0;
    int examined = 0;
    int proposals = 0;
    int stale = 0;
    uint64_t nodes = 0;
    bool limit_reached = false;
    bool changed = true;
    /*
     * Runtime polish budgets. The pass cap bounds how many complete
     * passes run. The yield gate stops once a pass's accepted reduction
     * falls below the given fraction of the previous pass's reduction,
     * which cuts the long diminishing-return tail.
     */
    if (pass_cap < 1 || pass_cap > POLISH_MAX_PASSES) {
        pass_cap = POLISH_MAX_PASSES;
    }
    if (!isfinite(yield_gate) || yield_gate < 0.0) yield_gate = 0.25;
    int reduced_before_pass = 0;
    int previous_pass_reduced = 0;
    while (changed && passes < pass_cap) {
        reduced_before_pass = reduced;
        /*
         * Inspect the deadline only between complete passes: every
         * committed proposal remains validated, so stopping here keeps
         * the partial literal reductions of the finished passes.
         */
        if (
            deadline_nanoseconds > 0 &&
            polish_monotonic_nanoseconds() >= deadline_nanoseconds
        ) {
            break;
        }
        changed = false;
        passes++;
        memset(proposal_valid, 0, (size_t)cardinality);
        memset(proposal_limited, 0, (size_t)cardinality);
        memset(proposal_nodes, 0, (size_t)cardinality * sizeof(uint64_t));

        PolishProposalContext context = {
            .prepared = prepared,
            .cardinality = cardinality,
            .ninputs = ninputs,
            .on_rows = on_rows,
            .on_words = on_words,
            .off_rows = off_rows,
            .off_words = off_words,
            .off_stride = off_stride,
            .solution = solution,
            .coverage = coverage,
            .cover_counts = cover_counts,
            .node_limit = node_limit,
            .proposal_patterns = proposal_patterns,
            .proposal_coverage = proposal_coverage,
            .proposal_support = proposal_support,
            .proposal_valid = proposal_valid,
            .proposal_limited = proposal_limited,
            .proposal_nodes = proposal_nodes,
            .private_rows = private_rows,
            .masks = masks,
            .remaining_stack = remaining_stack,
            .pattern = pattern,
            .agreement = agreement,
            .candidate_input = candidate_input,
            .best_choice = best_choice,
            .current_choice = current_choice,
            .active_stack = active_stack
        };
        atomic_store_explicit(
            &context.next_cube,
            0,
            memory_order_relaxed
        );
        if (!ccubes_parallel_for(
            0,
            (uint64_t)cardinality,
            threads,
            true,
            polish_proposal_worker,
            &context
        )) {
            memcpy(solution, original_solution, solution_values * sizeof(int));
            goto failed;
        }

        examined += cardinality;
        for (int cube = 0; cube < cardinality; ++cube) {
            nodes += proposal_nodes[cube];
            if (proposal_limited[cube]) limit_reached = true;
            if (proposal_valid[cube]) proposals++;
        }

        for (int cube = 0; cube < cardinality; ++cube) {
            if (!proposal_valid[cube]) continue;
            int current_support = 0;
            for (int input = 0; input < ninputs; ++input) {
                current_support += solution[
                    (size_t)input * (size_t)cardinality + (size_t)cube
                ] != 0;
            }
            if (proposal_support[cube] >= current_support) continue;

            uint64_t *old_coverage = &coverage[
                (size_t)cube * (size_t)on_words
            ];
            const uint64_t *new_coverage = &proposal_coverage[
                (size_t)cube * (size_t)on_words
            ];
            private_rows_from_counts(
                old_coverage,
                cover_counts,
                on_rows,
                on_words,
                commit_private_rows
            );
            bool live = true;
            for (int word = 0; word < on_words; ++word) {
                if (commit_private_rows[word] & ~new_coverage[word]) {
                    live = false;
                    break;
                }
            }
            if (!live) {
                stale++;
                continue;
            }

            update_cover_counts(
                cover_counts,
                on_rows,
                on_words,
                old_coverage,
                new_coverage
            );
            memcpy(
                old_coverage,
                new_coverage,
                (size_t)on_words * sizeof(uint64_t)
            );
            const int *new_pattern = &proposal_patterns[
                (size_t)cube * (size_t)ninputs
            ];
            for (int input = 0; input < ninputs; ++input) {
                solution[
                    (size_t)input * (size_t)cardinality + (size_t)cube
                ] = new_pattern[input];
            }
            reduced++;
            changed = true;
        }

        if (!progressive_solution_is_valid_prepared(
            prepared,
            cardinality,
            solution
        )) {
            memcpy(solution, original_solution, solution_values * sizeof(int));
            goto failed;
        }

        {
            const int pass_reduced = reduced - reduced_before_pass;
            if (
                yield_gate > 0.0 &&
                passes >= 2 &&
                previous_pass_reduced > 0 &&
                (double)pass_reduced <
                    yield_gate * (double)previous_pass_reduced
            ) {
                break;
            }
            previous_pass_reduced = pass_reduced;
        }
    }

    int literals_after = 0;
    for (size_t value = 0; value < solution_values; ++value) {
        literals_after += solution[value] != 0;
    }
    if (stats) {
        stats->passes = passes;
        stats->cubes_examined = examined;
        stats->cubes_reduced = reduced;
        stats->proposals = proposals;
        stats->stale_proposals = stale;
        stats->threads_used = threads;
        stats->literals_before = literals_before;
        stats->literals_after = literals_after;
        stats->search_nodes = nodes;
        stats->allocated_bytes =
            (uint64_t)base_allocated_bytes + (uint64_t)active_list_bytes;
        stats->active_list_bytes = (uint64_t)active_list_bytes;
        stats->node_limit_reached = limit_reached;
        stats->deadline_reached =
            deadline_nanoseconds > 0 &&
            polish_monotonic_nanoseconds() >= deadline_nanoseconds;
        stats->active_lists_used = use_active_lists;
        stats->seconds = polish_monotonic_seconds() - started;
    }

    free(original_solution); free(coverage); free(cover_counts);
    free(proposal_patterns); free(proposal_coverage);
    free(proposal_support); free(proposal_valid);
    free(proposal_limited); free(proposal_nodes);
    free(private_rows); free(masks); free(remaining_stack);
    free(pattern); free(agreement); free(candidate_input);
    free(best_choice); free(current_choice); free(active_stack);
    free(commit_private_rows);
    return true;

failed:
    free(original_solution); free(coverage); free(cover_counts);
    free(proposal_patterns); free(proposal_coverage);
    free(proposal_support); free(proposal_valid);
    free(proposal_limited); free(proposal_nodes);
    free(private_rows); free(masks); free(remaining_stack);
    free(pattern); free(agreement); free(candidate_input);
    free(best_choice); free(current_choice); free(active_stack);
    free(commit_private_rows);
    return false;
}
