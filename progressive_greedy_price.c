/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_greedy_price.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "ccubes_threads.h"

typedef struct {
    double weight;
    int row;
} GreedyAnchor;

typedef struct {
    uint64_t hits;
    double retained;
} GreedyRootScore;

typedef struct {
    bool zero_loss;
    double ratio;
    uint64_t hits;
    double retained;
    int vertex;
} GreedyPrefixScore;

typedef struct {
    const PIstorage *pi;
    const WeightedMMCSPreparedRows *prepared;
    const double *row_weights;
    const GreedyRootScore *root_scores;
    const GreedyAnchor *anchors;
    int anchor_count;
    int variants_per_anchor;
    int prefix_beam_width;
    const int *prefix_beam_vertices;
    int candidate_count;
    int ninputs;
    int on_rows;
    int off_rows;
    int on_words;
    int off_words;
    int max_support;
    double initial_weight;
    double threshold;
    double tolerance;
    uint64_t deadline_nanoseconds;
    WeightedMMCSPriceBatch *batch;
    unsigned char *attempted;
    unsigned char *valid;
    uint64_t *literal_evaluations;
    uint64_t *remaining;
    uint64_t *compatible;
    uint64_t *blocked;
    uint64_t *cleanup_prefix;
    unsigned char *selected;
    int *selection_order;
} GreedyWorkerContext;

struct ProgressiveGreedyPriceWorkspace {
    int ninputs, on_rows, on_words, off_words, threads, work_count;
    size_t beam_entries;
    bool cleanup_prefix_requested;
    GreedyRootScore *root_scores;
    GreedyAnchor *anchors;
    int *prefix_beam_vertices;
    unsigned char *attempted, *valid;
    uint64_t *literal_evaluations;
    uint64_t *remaining, *compatible, *blocked, *cleanup_prefix;
    unsigned char *selected;
    int *selection_order;
};

static void greedy_workspace_release(
    ProgressiveGreedyPriceWorkspace *workspace
) {
    if (!workspace) return;
    free(workspace->root_scores);
    free(workspace->anchors);
    free(workspace->prefix_beam_vertices);
    free(workspace->attempted);
    free(workspace->valid);
    free(workspace->literal_evaluations);
    free(workspace->remaining);
    free(workspace->compatible);
    free(workspace->blocked);
    free(workspace->cleanup_prefix);
    free(workspace->selected);
    free(workspace->selection_order);
    memset(workspace, 0, sizeof(*workspace));
}

ProgressiveGreedyPriceWorkspace *progressive_greedy_price_workspace_create(void) {
    return (ProgressiveGreedyPriceWorkspace *)calloc(
        1, sizeof(ProgressiveGreedyPriceWorkspace)
    );
}

void progressive_greedy_price_workspace_destroy(
    ProgressiveGreedyPriceWorkspace *workspace
) {
    greedy_workspace_release(workspace);
    free(workspace);
}

static bool greedy_workspace_prepare(
    ProgressiveGreedyPriceWorkspace *workspace,
    int ninputs, int on_rows, int on_words, int off_words,
    int threads, int work_count, size_t beam_entries,
    bool have_cleanup_prefix
) {
    if (
        workspace->ninputs != ninputs || workspace->on_rows != on_rows ||
        workspace->on_words != on_words || workspace->off_words != off_words ||
        workspace->threads != threads || workspace->work_count != work_count ||
        workspace->beam_entries != beam_entries ||
        workspace->cleanup_prefix_requested != have_cleanup_prefix
    ) {
        /* Rebuild exact-sized storage on a shape change so retained capacity
         * cannot exceed the current call's checked worker memory allowance. */
        greedy_workspace_release(workspace);
        workspace->root_scores = malloc(
            2u * (size_t)ninputs * sizeof(GreedyRootScore)
        );
        workspace->anchors = malloc((size_t)on_rows * sizeof(GreedyAnchor));
        workspace->prefix_beam_vertices = beam_entries > 0
            ? malloc(beam_entries * sizeof(int)) : NULL;
        workspace->attempted = malloc((size_t)work_count * sizeof(unsigned char));
        workspace->valid = malloc((size_t)work_count * sizeof(unsigned char));
        workspace->literal_evaluations = malloc(
            (size_t)work_count * sizeof(uint64_t)
        );
        workspace->remaining = malloc(
            (size_t)threads * (size_t)off_words * sizeof(uint64_t)
        );
        workspace->compatible = malloc(
            (size_t)threads * (size_t)on_words * sizeof(uint64_t)
        );
        workspace->blocked = malloc(
            (size_t)threads * (size_t)off_words * sizeof(uint64_t)
        );
        workspace->selected = malloc(
            (size_t)threads * (size_t)ninputs * sizeof(unsigned char)
        );
        workspace->selection_order = malloc(
            (size_t)threads * (size_t)ninputs * sizeof(int)
        );
        if (
            !workspace->root_scores || !workspace->anchors ||
            (beam_entries > 0 && !workspace->prefix_beam_vertices) ||
            !workspace->attempted || !workspace->valid ||
            !workspace->literal_evaluations || !workspace->remaining ||
            !workspace->compatible || !workspace->blocked ||
            !workspace->selected || !workspace->selection_order
        ) {
            greedy_workspace_release(workspace);
            return false;
        }
        /* Optional acceleration must not make a usable scalar workspace
         * fail. Allocate it last, and retain the scalar fallback if memory
         * pressure prevents this extra allocation. */
        workspace->cleanup_prefix = have_cleanup_prefix
            ? malloc(
                (size_t)threads * (size_t)(ninputs - 1) *
                (size_t)off_words * sizeof(uint64_t)
            )
            : NULL;
        workspace->ninputs = ninputs;
        workspace->on_rows = on_rows;
        workspace->on_words = on_words;
        workspace->off_words = off_words;
        workspace->threads = threads;
        workspace->work_count = work_count;
        workspace->beam_entries = beam_entries;
        workspace->cleanup_prefix_requested = have_cleanup_prefix;
    }
    memset(workspace->attempted, 0, (size_t)work_count * sizeof(unsigned char));
    memset(workspace->valid, 0, (size_t)work_count * sizeof(unsigned char));
    memset(workspace->literal_evaluations, 0, (size_t)work_count * sizeof(uint64_t));
    return true;
}

static uint64_t greedy_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint64_t greedy_low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static bool greedy_checked_add_u64(
    uint64_t left,
    uint64_t right,
    uint64_t *result
) {
    if (!result || left > UINT64_MAX - right) return false;
    *result = left + right;
    return true;
}

static bool greedy_checked_mul_u64(
    uint64_t left,
    uint64_t right,
    uint64_t *result
) {
    if (!result || (left != 0 && right > UINT64_MAX / left)) return false;
    *result = left * right;
    return true;
}

static bool greedy_words_empty(const uint64_t *bits, int words) {
    for (int word = 0; word < words; ++word) {
        if (bits[word] != 0) return false;
    }
    return true;
}

static bool greedy_deadline_reached(uint64_t deadline_nanoseconds) {
    return deadline_nanoseconds > 0 &&
        greedy_monotonic_nanoseconds() >= deadline_nanoseconds;
}

static uint64_t greedy_column_word(
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
    if (remaining < 64) values &= greedy_low_bits_mask(remaining);
    return values;
}

static double greedy_weighted_intersection(
    const uint64_t *compatible,
    const uint64_t *columns,
    int words,
    int vertex,
    bool complement,
    int rows,
    const double *weights
) {
    double total = 0.0;
    for (int word = 0; word < words; ++word) {
        uint64_t bits = compatible[word] &
            greedy_column_word(
                columns,
                words,
                vertex,
                word,
                complement,
                rows
            );
        while (bits != 0) {
            int bit = __builtin_ctzll(bits);
            bits &= bits - 1u;
            int row = word * 64 + bit;
            if (row < rows) total += weights[row];
        }
    }
    return total;
}

/* Both polarities accumulate their ON weights independently, in ascending
 * row order, exactly as greedy_weighted_intersection does. In particular,
 * deriving either sum by subtraction can change a near-tie. */
static void greedy_prepare_root_scores(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    const double *row_weights,
    GreedyRootScore *scores,
    uint64_t deadline_nanoseconds
) {
    memset(scores, 0, 2u * (size_t)pi->inputs * sizeof(GreedyRootScore));
    for (int vertex = 0; vertex < pi->inputs; ++vertex) {
        if (greedy_deadline_reached(deadline_nanoseconds)) return;
        uint64_t off_ones = 0;
        if (prepared->off_one_counts) {
            off_ones = prepared->off_one_counts[vertex];
        } else {
            for (int word = 0; word < prepared->off_column_words; ++word) {
                off_ones += (uint64_t)__builtin_popcountll(
                    greedy_column_word(
                        prepared->off_columns, prepared->off_column_words,
                        vertex, word, false, pi->OFF_minterms
                    )
                );
            }
        }
        scores[2 * (size_t)vertex].hits = off_ones;
        scores[2 * (size_t)vertex + 1].hits =
            (uint64_t)pi->OFF_minterms - off_ones;
        for (int polarity = 0; polarity < 2; ++polarity) {
            double retained = 0.0;
            for (int word = 0; word < prepared->on_column_words; ++word) {
                uint64_t bits = greedy_column_word(
                    prepared->on_columns, prepared->on_column_words,
                    vertex, word, polarity == 0, pi->ON_minterms
                );
                while (bits != 0) {
                    int row = word * 64 + __builtin_ctzll(bits);
                    bits &= bits - 1u;
                    if (row < pi->ON_minterms) retained += row_weights[row];
                }
            }
            scores[2 * (size_t)vertex + (size_t)polarity].retained = retained;
        }
    }
}

static double greedy_weight_sum(
    const uint64_t *bits_by_word,
    int words,
    int rows,
    const double *weights
) {
    double total = 0.0;
    for (int word = 0; word < words; ++word) {
        uint64_t bits = bits_by_word[word];
        while (bits != 0) {
            int bit = __builtin_ctzll(bits);
            bits &= bits - 1u;
            int row = word * 64 + bit;
            if (row < rows) total += weights[row];
        }
    }
    return total;
}

static int greedy_decode_forced_prefix(
    int variant,
    int ninputs,
    int forced[3]
) {
    if (variant == 0) return 0;
    if (variant <= ninputs) {
        forced[0] = variant - 1;
        return 1;
    }

    int pair_count = ninputs * (ninputs - 1) / 2;
    int pair = variant - ninputs - 1;
    if (pair >= pair_count) {
        int triple = pair - pair_count;
        for (int first = 0; first + 2 < ninputs; ++first) {
            for (
                int second = first + 1;
                second + 1 < ninputs;
                ++second
            ) {
                int triples_with_pair = ninputs - second - 1;
                if (triple < triples_with_pair) {
                    forced[0] = first;
                    forced[1] = second;
                    forced[2] = second + 1 + triple;
                    return 3;
                }
                triple -= triples_with_pair;
            }
        }
        return -1;
    }
    for (int first = 0; first + 1 < ninputs; ++first) {
        int pairs_with_first = ninputs - first - 1;
        if (pair < pairs_with_first) {
            forced[0] = first;
            forced[1] = first + 1 + pair;
            return 2;
        }
        pair -= pairs_with_first;
    }
    return -1;
}

static int greedy_compare_anchors(
    const void *left,
    const void *right
) {
    const GreedyAnchor *a = (const GreedyAnchor *)left;
    const GreedyAnchor *b = (const GreedyAnchor *)right;
    if (a->weight > b->weight) return -1;
    if (a->weight < b->weight) return 1;
    return (a->row > b->row) - (a->row < b->row);
}

/* Keep the worst selected anchor at the root. The weight/row comparator is
 * a total order, so sorting the retained prefix exactly reproduces the
 * first `selected_count` entries of a full qsort, including weight ties. */
static void greedy_anchor_heap_sift(
    GreedyAnchor *anchors, int count, int parent
) {
    for (;;) {
        if (parent >= count / 2) return;
        int child = parent * 2 + 1;
        if (
            child + 1 < count &&
            greedy_compare_anchors(&anchors[child + 1], &anchors[child]) > 0
        ) {
            child++;
        }
        if (greedy_compare_anchors(&anchors[parent], &anchors[child]) >= 0) return;
        GreedyAnchor swap = anchors[parent];
        anchors[parent] = anchors[child];
        anchors[child] = swap;
        parent = child;
    }
}

static void greedy_select_anchors(
    GreedyAnchor *anchors, int count, int selected_count
) {
    if (selected_count >= count / 2) {
        qsort(anchors, (size_t)count, sizeof(GreedyAnchor), greedy_compare_anchors);
        return;
    }
    for (int parent = selected_count / 2; parent > 0; --parent) {
        greedy_anchor_heap_sift(anchors, selected_count, parent - 1);
    }
    for (int next = selected_count; next < count; ++next) {
        if (greedy_compare_anchors(&anchors[next], &anchors[0]) >= 0) continue;
        GreedyAnchor swap = anchors[0];
        anchors[0] = anchors[next];
        anchors[next] = swap;
        greedy_anchor_heap_sift(anchors, selected_count, 0);
    }
    qsort(anchors, (size_t)selected_count, sizeof(GreedyAnchor), greedy_compare_anchors);
}

static bool greedy_choice_precedes(
    bool zero_loss,
    double ratio,
    uint64_t hits,
    double retained,
    int vertex,
    bool best_zero_loss,
    double best_ratio,
    uint64_t best_hits,
    double best_retained,
    int best_vertex
) {
    if (best_vertex < 0) return true;
    if (zero_loss != best_zero_loss) return zero_loss;
    if (zero_loss) {
        if (hits != best_hits) return hits > best_hits;
    } else {
        double scale = fmax(1.0, fmax(fabs(ratio), fabs(best_ratio)));
        if (fabs(ratio - best_ratio) > 1e-12 * scale) {
            return ratio > best_ratio;
        }
    }
    if (retained != best_retained) return retained > best_retained;
    if (hits != best_hits) return hits > best_hits;
    return vertex < best_vertex;
}

static int greedy_compare_prefix_scores(
    const void *left,
    const void *right
) {
    const GreedyPrefixScore *a = (const GreedyPrefixScore *)left;
    const GreedyPrefixScore *b = (const GreedyPrefixScore *)right;
    if (greedy_choice_precedes(
        a->zero_loss,
        a->ratio,
        a->hits,
        a->retained,
        a->vertex,
        b->zero_loss,
        b->ratio,
        b->hits,
        b->retained,
        b->vertex
    )) {
        return -1;
    }
    if (greedy_choice_precedes(
        b->zero_loss,
        b->ratio,
        b->hits,
        b->retained,
        b->vertex,
        a->zero_loss,
        a->ratio,
        a->hits,
        a->retained,
        a->vertex
    )) {
        return 1;
    }
    return 0;
}

static bool greedy_prepare_prefix_beam(
    const PIstorage *pi,
    const GreedyRootScore *root_scores,
    const GreedyAnchor *anchors,
    int anchor_count,
    int beam_width,
    double initial_weight,
    int *beam_vertices
) {
    if (beam_width <= 0) return true;
    GreedyPrefixScore *scores = (GreedyPrefixScore *)malloc(
        (size_t)pi->inputs * sizeof(GreedyPrefixScore)
    );
    if (!scores) return false;

    for (int rank = 0; rank < anchor_count; ++rank) {
        const int *anchor = &pi->ON_set[
            (size_t)anchors[rank].row * (size_t)pi->inputs
        ];
        for (int vertex = 0; vertex < pi->inputs; ++vertex) {
            uint64_t hits = anchor[vertex] == 2
                ? (uint64_t)pi->OFF_minterms -
                    root_scores[2 * (size_t)vertex].hits
                : root_scores[2 * (size_t)vertex].hits;
            double retained = anchor[vertex] == 1
                ? initial_weight - root_scores[2 * (size_t)vertex + 1].retained
                : root_scores[2 * (size_t)vertex + 1].retained;
            double loss = initial_weight - retained;
            double loss_scale = fmax(1.0, fabs(initial_weight));
            if (loss < 0.0 && fabs(loss) <= 1e-12 * loss_scale) {
                loss = 0.0;
            }
            bool zero_loss =
                hits > 0 && loss <= 1e-12 * loss_scale;
            scores[vertex] = (GreedyPrefixScore){
                .zero_loss = zero_loss,
                .ratio = hits == 0
                    ? -DBL_MAX
                    : (
                        zero_loss
                            ? DBL_MAX
                            : (double)hits / loss
                    ),
                .hits = hits,
                .retained = hits == 0 ? -DBL_MAX : retained,
                .vertex = vertex
            };
        }
        qsort(
            scores,
            (size_t)pi->inputs,
            sizeof(GreedyPrefixScore),
            greedy_compare_prefix_scores
        );
        for (int beam = 0; beam < beam_width; ++beam) {
            beam_vertices[
                (size_t)rank * (size_t)beam_width + (size_t)beam
            ] = scores[beam].vertex;
        }
    }

    free(scores);
    return true;
}

static void greedy_construct_range(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    (void)worker_count;
    GreedyWorkerContext *context = (GreedyWorkerContext *)data;
    uint64_t *remaining = &context->remaining[
        (size_t)worker_id * (size_t)context->off_words
    ];
    uint64_t *compatible = &context->compatible[
        (size_t)worker_id * (size_t)context->on_words
    ];
    uint64_t *blocked = &context->blocked[
        (size_t)worker_id * (size_t)context->off_words
    ];
    uint64_t *cleanup_prefix = context->cleanup_prefix
        ? &context->cleanup_prefix[
            (size_t)worker_id * (size_t)(context->ninputs - 1) *
            (size_t)context->off_words
        ]
        : NULL;
    unsigned char *selected = &context->selected[
        (size_t)worker_id * (size_t)context->ninputs
    ];
    int *selection_order = &context->selection_order[
        (size_t)worker_id * (size_t)context->ninputs
    ];

    for (uint64_t rank64 = start; rank64 < end; rank64 += stride) {
        int rank = (int)rank64;
        if (greedy_deadline_reached(context->deadline_nanoseconds)) {
            break;
        }
        context->attempted[rank] = 1;
        context->batch->support_sizes[rank] = 0;
        int anchor_rank = rank / context->variants_per_anchor;
        int variant = rank % context->variants_per_anchor;
        int anchor_index = context->anchors[anchor_rank].row;
        const int *anchor = &context->pi->ON_set[
            (size_t)anchor_index * (size_t)context->ninputs
        ];

        for (int word = 0; word < context->off_words; ++word) {
            remaining[word] = greedy_low_bits_mask(
                context->off_rows - word * 64
            );
        }
        for (int word = 0; word < context->on_words; ++word) {
            compatible[word] = greedy_low_bits_mask(
                context->on_rows - word * 64
            );
        }
        memset(
            selected,
            0,
            (size_t)context->ninputs * sizeof(unsigned char)
        );
        int selected_count = 0;
        double current_weight = context->initial_weight;
        bool failed = false;

        int forced_vertices[3];
        int forced_count = 0;
        if (context->prefix_beam_width > 0) {
            if (variant > 0) {
                forced_vertices[0] = context->prefix_beam_vertices[
                    (size_t)anchor_rank *
                        (size_t)context->prefix_beam_width +
                    (size_t)(variant - 1)
                ];
                forced_count = 1;
            }
        } else {
            forced_count = greedy_decode_forced_prefix(
                variant,
                context->ninputs,
                forced_vertices
            );
        }
        if (forced_count < 0) failed = true;
        for (
            int forced_index = 0;
            forced_index < forced_count && !failed;
            ++forced_index
        ) {
            int forced_vertex = forced_vertices[forced_index];
            context->literal_evaluations[rank]++;
            bool mismatch_complement =
                anchor[forced_vertex] == 2;
            const GreedyRootScore *root = selected_count == 0
                ? &context->root_scores[
                    2 * (size_t)forced_vertex +
                    (size_t)(anchor[forced_vertex] - 1)
                ]
                : NULL;
            uint64_t hits = root ? root->hits : 0;
            if (!root) {
                for (int word = 0; word < context->off_words; ++word) {
                    hits += (uint64_t)__builtin_popcountll(
                        remaining[word] &
                        greedy_column_word(
                            context->prepared->off_columns,
                            context->off_words,
                            forced_vertex,
                            word,
                            mismatch_complement,
                            context->off_rows
                        )
                    );
                }
            }
            if (hits == 0) {
                failed = true;
            } else {
                selected[forced_vertex] = 1;
                selection_order[selected_count++] = forced_vertex;
                bool match_complement =
                    anchor[forced_vertex] == 1;
                for (
                    int word = 0;
                    word < context->off_words;
                    ++word
                ) {
                    remaining[word] &= ~greedy_column_word(
                        context->prepared->off_columns,
                        context->off_words,
                        forced_vertex,
                        word,
                        mismatch_complement,
                        context->off_rows
                    );
                }
                for (
                    int word = 0;
                    word < context->on_words;
                    ++word
                ) {
                    compatible[word] &= greedy_column_word(
                        context->prepared->on_columns,
                        context->on_words,
                        forced_vertex,
                        word,
                        match_complement,
                        context->on_rows
                    );
                }
                current_weight = root ? root->retained : greedy_weight_sum(
                    compatible,
                    context->on_words,
                    context->on_rows,
                    context->row_weights
                );
            }
        }

        while (
            !failed &&
            !greedy_words_empty(remaining, context->off_words)
        ) {
            if (greedy_deadline_reached(context->deadline_nanoseconds)) {
                failed = true;
                break;
            }
            int best_vertex = -1;
            uint64_t best_hits = 0;
            double best_retained = 0.0;
            double best_ratio = -DBL_MAX;
            bool best_zero_loss = false;

            for (int vertex = 0; vertex < context->ninputs; ++vertex) {
                if (selected[vertex]) continue;
                context->literal_evaluations[rank]++;
                bool mismatch_complement = anchor[vertex] == 2;
                const GreedyRootScore *root = selected_count == 0
                    ? &context->root_scores[
                        2 * (size_t)vertex + (size_t)(anchor[vertex] - 1)
                    ]
                    : NULL;
                uint64_t hits = root ? root->hits : 0;
                if (!root) {
                    for (int word = 0; word < context->off_words; ++word) {
                        hits += (uint64_t)__builtin_popcountll(
                            remaining[word] &
                            greedy_column_word(
                                context->prepared->off_columns,
                                context->off_words,
                                vertex,
                                word,
                                mismatch_complement,
                                context->off_rows
                            )
                        );
                    }
                }
                if (hits == 0) continue;

                /*
                 * Exact skip: while the current best is a zero-loss
                 * literal, any candidate with strictly fewer hits cannot
                 * precede it under greedy_choice_precedes, because
                 * zero_loss dominates and equal-zero-loss candidates are
                 * ranked by hits first. Its retained weight and loss are
                 * never consulted, so the expensive weighted
                 * intersection can be skipped without changing the
                 * selected vertex.
                 */
                if (
                    best_vertex >= 0 &&
                    best_zero_loss &&
                    hits < best_hits
                ) {
                    continue;
                }

                bool match_complement = anchor[vertex] == 1;
                double retained = root ? root->retained : greedy_weighted_intersection(
                    compatible,
                    context->prepared->on_columns,
                    context->on_words,
                    vertex,
                    match_complement,
                    context->on_rows,
                    context->row_weights
                );
                double loss = current_weight - retained;
                double loss_scale = fmax(1.0, fabs(current_weight));
                if (loss < 0.0 && fabs(loss) <= 1e-12 * loss_scale) {
                    loss = 0.0;
                }
                bool zero_loss = loss <= 1e-12 * loss_scale;
                double ratio = zero_loss
                    ? DBL_MAX
                    : (double)hits / loss;
                if (greedy_choice_precedes(
                    zero_loss,
                    ratio,
                    hits,
                    retained,
                    vertex,
                    best_zero_loss,
                    best_ratio,
                    best_hits,
                    best_retained,
                    best_vertex
                )) {
                    best_vertex = vertex;
                    best_hits = hits;
                    best_retained = retained;
                    best_ratio = ratio;
                    best_zero_loss = zero_loss;
                }
            }

            if (best_vertex < 0 || best_hits == 0) {
                failed = true;
                break;
            }
            selected[best_vertex] = 1;
            selection_order[selected_count++] = best_vertex;
            bool mismatch_complement = anchor[best_vertex] == 2;
            bool match_complement = anchor[best_vertex] == 1;
            for (int word = 0; word < context->off_words; ++word) {
                remaining[word] &= ~greedy_column_word(
                    context->prepared->off_columns,
                    context->off_words,
                    best_vertex,
                    word,
                    mismatch_complement,
                    context->off_rows
                );
            }
            for (int word = 0; word < context->on_words; ++word) {
                compatible[word] &= greedy_column_word(
                    context->prepared->on_columns,
                    context->on_words,
                    best_vertex,
                    word,
                    match_complement,
                    context->on_rows
                );
            }
            current_weight = best_retained;
        }
        if (failed || selected_count <= 0) continue;

        if (cleanup_prefix && selected_count >= 4) {
            /* Reverse deletion only changes literals after the current one.
             * Keep the unchanged earlier union and the retained later union
             * separately; this is the same deletion order as the scalar loop. */
            for (int index = 0; index + 1 < selected_count; ++index) {
                int vertex = selection_order[index];
                uint64_t *prefix = &cleanup_prefix[
                    (size_t)index * (size_t)context->off_words
                ];
                const uint64_t *previous = index > 0
                    ? prefix - context->off_words : NULL;
                for (int word = 0; word < context->off_words; ++word) {
                    prefix[word] =
                        (previous ? previous[word] : 0) |
                        greedy_column_word(
                            context->prepared->off_columns, context->off_words,
                            vertex, word, anchor[vertex] == 2, context->off_rows
                        );
                }
            }
            memset(blocked, 0, (size_t)context->off_words * sizeof(uint64_t));
            for (int index = selected_count - 1; index >= 0; --index) {
                int removed = selection_order[index];
                const uint64_t *prefix = index > 0
                    ? &cleanup_prefix[(size_t)(index - 1) * (size_t)context->off_words]
                    : NULL;
                bool all_blocked = true;
                for (int word = 0; word < context->off_words; ++word) {
                    uint64_t full = greedy_low_bits_mask(context->off_rows - word * 64);
                    if (((prefix ? prefix[word] : 0) | blocked[word]) != full) {
                        all_blocked = false;
                        break;
                    }
                }
                if (all_blocked) {
                    selected[removed] = 0;
                } else {
                    for (int word = 0; word < context->off_words; ++word) {
                        blocked[word] |= greedy_column_word(
                            context->prepared->off_columns, context->off_words,
                            removed, word, anchor[removed] == 2, context->off_rows
                        );
                    }
                }
            }
        } else {
            for (int index = selected_count - 1; index >= 0; --index) {
                int removed = selection_order[index];
                memset(
                    blocked,
                    0,
                    (size_t)context->off_words * sizeof(uint64_t)
                );
                for (int other = 0; other < selected_count; ++other) {
                    int vertex = selection_order[other];
                    if (vertex == removed || !selected[vertex]) continue;
                    bool mismatch_complement = anchor[vertex] == 2;
                    for (int word = 0; word < context->off_words; ++word) {
                        blocked[word] |= greedy_column_word(
                            context->prepared->off_columns,
                            context->off_words,
                            vertex,
                            word,
                            mismatch_complement,
                            context->off_rows
                        );
                    }
                }
                bool all_blocked = true;
                for (int word = 0; word < context->off_words; ++word) {
                    uint64_t full = greedy_low_bits_mask(
                        context->off_rows - word * 64
                    );
                    if ((blocked[word] & full) != full) {
                        all_blocked = false;
                        break;
                    }
                }
                if (all_blocked) selected[removed] = 0;
            }

        }

        uint64_t *coverage = &context->batch->coverage[
            (size_t)rank * (size_t)context->on_words
        ];
        for (int word = 0; word < context->on_words; ++word) {
            coverage[word] = greedy_low_bits_mask(
                context->on_rows - word * 64
            );
        }
        int support = 0;
        int *vertices = &context->batch->vertices[
            (size_t)rank * (size_t)context->ninputs
        ];
        for (int vertex = 0; vertex < context->ninputs; ++vertex) {
            if (!selected[vertex]) continue;
            vertices[support++] = vertex;
            bool match_complement = anchor[vertex] == 1;
            for (int word = 0; word < context->on_words; ++word) {
                coverage[word] &= greedy_column_word(
                    context->prepared->on_columns,
                    context->on_words,
                    vertex,
                    word,
                    match_complement,
                    context->on_rows
                );
            }
        }
        if (support <= 0) continue;
        if (
            context->max_support > 0 &&
            support > context->max_support
        ) {
            continue;
        }

        double score = 0.0;
        for (int word = 0; word < context->on_words; ++word) {
            uint64_t bits = coverage[word];
            while (bits != 0) {
                int bit = __builtin_ctzll(bits);
                bits &= bits - 1u;
                int row = word * 64 + bit;
                if (row < context->on_rows) {
                    score += context->row_weights[row];
                }
            }
        }
        context->batch->weights[rank] = score;
        context->batch->anchors[rank] = anchor_index;
        context->batch->support_sizes[rank] = support;
        if (
            score >
            context->threshold + context->tolerance
        ) {
            context->valid[rank] = 1;
        }
    }
}

bool progressive_greedy_price_candidate_capacity(
    int anchors,
    int ninputs,
    int forced_prefix_depth,
    int *capacity
) {
    if (
        !capacity ||
        anchors <= 0 ||
        ninputs <= 0 ||
        forced_prefix_depth < 0 ||
        forced_prefix_depth > 3
    ) {
        return false;
    }
    uint64_t variants = 1;
    if (forced_prefix_depth >= 1) {
        variants += (uint64_t)ninputs;
    }
    if (forced_prefix_depth >= 2) {
        variants +=
            (uint64_t)ninputs *
            (uint64_t)(ninputs - 1) /
            2u;
    }
    if (forced_prefix_depth >= 3 && ninputs >= 3) {
        uint64_t first = (uint64_t)ninputs;
        uint64_t second = (uint64_t)(ninputs - 1);
        uint64_t third = (uint64_t)(ninputs - 2);
        if (first % 2u == 0) {
            first /= 2u;
        } else {
            second /= 2u;
        }
        if (first % 3u == 0) {
            first /= 3u;
        } else if (second % 3u == 0) {
            second /= 3u;
        } else {
            third /= 3u;
        }
        if (
            first > UINT64_MAX / second ||
            first * second > UINT64_MAX / third
        ) {
            return false;
        }
        uint64_t triples = first * second * third;
        if (variants > UINT64_MAX - triples) return false;
        variants += triples;
    }
    uint64_t total = (uint64_t)anchors * variants;
    if (total == 0 || total > INT_MAX) return false;
    *capacity = (int)total;
    return true;
}

bool progressive_greedy_price_beam_capacity(
    int anchors,
    int ninputs,
    int candidate_budget,
    int *capacity,
    int *beam_width
) {
    if (
        anchors <= 0 ||
        ninputs <= 0 ||
        candidate_budget < anchors ||
        !capacity ||
        !beam_width
    ) {
        return false;
    }
    int width = candidate_budget / anchors - 1;
    if (width < 0) width = 0;
    if (width > ninputs) width = ninputs;
    uint64_t total =
        (uint64_t)anchors * (uint64_t)(width + 1);
    if (total == 0 || total > INT_MAX) return false;
    *capacity = (int)total;
    *beam_width = width;
    return true;
}

ProgressiveGreedyPriceStatus progressive_greedy_price_batch(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    const double *row_weights,
    WeightedMMCSPriceBatch *batch,
    const ProgressiveGreedyPriceOptions *options,
    ProgressiveGreedyPriceStats *stats
) {
    if (stats) memset(stats, 0, sizeof(*stats));
    if (
        !pi ||
        !prepared ||
        !row_weights ||
        !batch ||
        !options ||
        !stats ||
        pi->inputs <= 0 ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        prepared->on_rows != pi->ON_minterms ||
        prepared->off_rows != pi->OFF_minterms ||
        prepared->ninputs != pi->inputs ||
        !prepared->on_columns ||
        !prepared->off_columns ||
        batch->capacity <= 0 ||
        !batch->weights ||
        !batch->anchors ||
        !batch->support_sizes ||
        !batch->vertices ||
        !batch->coverage ||
        options->requested_threads <= 0 ||
        options->anchor_limit <= 0 ||
        options->forced_prefix_depth < 0 ||
        options->forced_prefix_depth > 3 ||
        options->beam_candidate_budget < 0 ||
        options->max_support < 0 ||
        options->max_support > pi->inputs ||
        !isfinite(options->reduced_cost_threshold) ||
        !isfinite(options->reduced_cost_tolerance) ||
        options->reduced_cost_tolerance < 0.0
    ) {
        return PROGRESSIVE_GREEDY_PRICE_INVALID;
    }
    for (int row = 0; row < pi->ON_minterms; ++row) {
        if (!isfinite(row_weights[row]) || row_weights[row] < 0.0) {
            return PROGRESSIVE_GREEDY_PRICE_INVALID;
        }
    }

    uint64_t started = greedy_monotonic_nanoseconds();
    const int ninputs = pi->inputs;
    const int on_rows = pi->ON_minterms;
    const int off_rows = pi->OFF_minterms;
    const int on_words = prepared->on_column_words;
    const int off_words = prepared->off_column_words;
    int anchor_count = options->anchor_limit;
    if (anchor_count > on_rows) anchor_count = on_rows;
    int candidate_count = 0;
    int prefix_beam_width = 0;
    bool capacity_ok =
        options->forced_prefix_depth == 0 &&
        options->beam_candidate_budget > anchor_count
            ? progressive_greedy_price_beam_capacity(
                anchor_count,
                ninputs,
                options->beam_candidate_budget,
                &candidate_count,
                &prefix_beam_width
            )
            : progressive_greedy_price_candidate_capacity(
                anchor_count,
                ninputs,
                options->forced_prefix_depth,
                &candidate_count
            );
    if (!capacity_ok) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }
    const int variants_per_anchor =
        candidate_count / anchor_count;
    const int work_count =
        options->scout_anchor_potential && on_rows > candidate_count
            ? on_rows
            : candidate_count;
    if (work_count > batch->capacity) {
        return PROGRESSIVE_GREEDY_PRICE_INVALID;
    }
    int threads = ccubes_effective_thread_count(
        options->requested_threads
    );
    if (threads > work_count) threads = work_count;
    if (threads < 1) threads = 1;

    uint64_t fixed_bytes = 0;
    uint64_t term = 0;
    if (
        !greedy_checked_mul_u64(
            (uint64_t)on_rows,
            sizeof(GreedyAnchor),
            &fixed_bytes
        ) ||
        !greedy_checked_mul_u64(
            (uint64_t)anchor_count * (uint64_t)prefix_beam_width,
            sizeof(int),
            &term
        ) ||
        !greedy_checked_add_u64(fixed_bytes, term, &fixed_bytes)
    ) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }
    if (
        !greedy_checked_mul_u64(
            (uint64_t)ninputs, 2u * sizeof(GreedyRootScore), &term
        ) ||
        !greedy_checked_add_u64(fixed_bytes, term, &fixed_bytes)
    ) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }
    if (prefix_beam_width > 0) {
        if (
            !greedy_checked_mul_u64(
                (uint64_t)ninputs,
                sizeof(GreedyPrefixScore),
                &term
            ) ||
            !greedy_checked_add_u64(fixed_bytes, term, &fixed_bytes)
        ) {
            return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
        }
    }
    if (
        !greedy_checked_mul_u64(
            (uint64_t)work_count,
            2u * sizeof(unsigned char) + sizeof(uint64_t),
            &term
        ) ||
        !greedy_checked_add_u64(fixed_bytes, term, &fixed_bytes)
    ) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }

    uint64_t per_thread_bytes = 0;
    if (
        !greedy_checked_mul_u64(
            (uint64_t)on_words + 2u * (uint64_t)off_words,
            sizeof(uint64_t),
            &per_thread_bytes
        ) ||
        !greedy_checked_mul_u64(
            (uint64_t)ninputs,
            sizeof(unsigned char) + sizeof(int),
            &term
        ) ||
        !greedy_checked_add_u64(
            per_thread_bytes,
            term,
            &per_thread_bytes
        )
    ) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }

    if (options->worker_memory_budget_bytes > 0) {
        if (fixed_bytes > options->worker_memory_budget_bytes) {
            return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
        }
        const uint64_t available =
            options->worker_memory_budget_bytes - fixed_bytes;
        const uint64_t memory_threads = per_thread_bytes > 0
            ? available / per_thread_bytes
            : (uint64_t)threads;
        if (memory_threads < 1) {
            return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
        }
        if ((uint64_t)threads > memory_threads) {
            threads = memory_threads > (uint64_t)INT_MAX
                ? INT_MAX
                : (int)memory_threads;
        }
    }

    uint64_t cleanup_bytes_per_thread = 0;
    bool have_cleanup_prefix = ninputs >= 4 &&
        greedy_checked_mul_u64(
            (uint64_t)(ninputs - 1) * (uint64_t)off_words,
            sizeof(uint64_t), &cleanup_bytes_per_thread
        );
    uint64_t cleanup_per_thread_total = 0;
    uint64_t cleanup_worker_total = 0;
    uint64_t cleanup_total = 0;
    have_cleanup_prefix = have_cleanup_prefix &&
        greedy_checked_add_u64(per_thread_bytes, cleanup_bytes_per_thread, &cleanup_per_thread_total) &&
        greedy_checked_mul_u64((uint64_t)threads, cleanup_per_thread_total, &cleanup_worker_total) &&
        greedy_checked_add_u64(fixed_bytes, cleanup_worker_total, &cleanup_total) &&
        (options->worker_memory_budget_bytes == 0 ||
         cleanup_total <= options->worker_memory_budget_bytes);
    if (have_cleanup_prefix) per_thread_bytes = cleanup_per_thread_total;

    uint64_t worker_bytes = 0;
    uint64_t allocated_bytes = 0;
    if (
        !greedy_checked_mul_u64(
            (uint64_t)threads,
            per_thread_bytes,
            &worker_bytes
        ) ||
        !greedy_checked_add_u64(
            fixed_bytes,
            worker_bytes,
            &allocated_bytes
        )
    ) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }

    ProgressiveGreedyPriceWorkspace local_workspace = {0};
    ProgressiveGreedyPriceWorkspace *workspace = options->workspace
        ? options->workspace : &local_workspace;
    if (!greedy_workspace_prepare(
        workspace, ninputs, on_rows, on_words, off_words, threads, work_count,
        (size_t)anchor_count * (size_t)prefix_beam_width, have_cleanup_prefix
    )) {
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }
    if (have_cleanup_prefix && !workspace->cleanup_prefix) {
        allocated_bytes -= (uint64_t)threads * cleanup_bytes_per_thread;
    }
    GreedyRootScore *root_scores = workspace->root_scores;
    GreedyAnchor *anchors = workspace->anchors;
    int *prefix_beam_vertices = workspace->prefix_beam_vertices;
    unsigned char *attempted = workspace->attempted;
    unsigned char *valid = workspace->valid;
    uint64_t *literal_evaluations = workspace->literal_evaluations;
    uint64_t *remaining = workspace->remaining;
    uint64_t *compatible = workspace->compatible;
    uint64_t *blocked = workspace->blocked;
    uint64_t *cleanup_prefix = workspace->cleanup_prefix;
    unsigned char *selected = workspace->selected;
    int *selection_order = workspace->selection_order;

    greedy_prepare_root_scores(
        pi, prepared, row_weights, root_scores, options->deadline_nanoseconds
    );

    double initial_weight = 0.0;
    for (int row = 0; row < on_rows; ++row) {
        anchors[row] = (GreedyAnchor){
            .weight = row_weights[row],
            .row = row
        };
        initial_weight += row_weights[row];
    }
    greedy_select_anchors(
        anchors, on_rows, options->scout_anchor_potential ? on_rows : anchor_count
    );
    if (options->scout_anchor_potential && anchor_count < on_rows) {
        memset(
            batch->support_sizes,
            0,
            (size_t)on_rows * sizeof(int)
        );
        GreedyWorkerContext scout_context = {
            .pi = pi,
            .prepared = prepared,
            .row_weights = row_weights,
            .root_scores = root_scores,
            .anchors = anchors,
            .anchor_count = on_rows,
            .variants_per_anchor = 1,
            .prefix_beam_width = 0,
            .prefix_beam_vertices = NULL,
            .candidate_count = on_rows,
            .ninputs = ninputs,
            .on_rows = on_rows,
            .off_rows = off_rows,
            .on_words = on_words,
            .off_words = off_words,
            .max_support = options->max_support,
            .initial_weight = initial_weight,
            .threshold = options->reduced_cost_threshold,
            .tolerance = options->reduced_cost_tolerance,
            .deadline_nanoseconds = options->deadline_nanoseconds,
            .batch = batch,
            .attempted = attempted,
            .valid = valid,
            .literal_evaluations = literal_evaluations,
            .remaining = remaining,
            .compatible = compatible,
            .blocked = blocked,
            .cleanup_prefix = cleanup_prefix,
            .selected = selected,
            .selection_order = selection_order
        };
        if (!ccubes_parallel_for(
            0,
            (uint64_t)on_rows,
            threads,
            false,
            greedy_construct_range,
            &scout_context
        )) {
            if (!options->workspace) greedy_workspace_release(&local_workspace);
            return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
        }
        for (int rank = 0; rank < on_rows; ++rank) {
            stats->scout_anchors_attempted += attempted[rank] != 0;
            stats->scout_candidates_constructed +=
                attempted[rank] != 0 &&
                batch->support_sizes[rank] > 0;
            stats->scout_literal_evaluations +=
                literal_evaluations[rank];
            anchors[rank].weight =
                batch->support_sizes[rank] > 0
                    ? batch->weights[rank]
                    : -DBL_MAX;
        }
        greedy_select_anchors(anchors, on_rows, anchor_count);
        memset(
            attempted,
            0,
            (size_t)work_count * sizeof(unsigned char)
        );
        memset(valid, 0, (size_t)work_count * sizeof(unsigned char));
        memset(
            literal_evaluations,
            0,
            (size_t)work_count * sizeof(uint64_t)
        );
    }
    if (!greedy_prepare_prefix_beam(
        pi,
        root_scores,
        anchors,
        anchor_count,
        prefix_beam_width,
        initial_weight,
        prefix_beam_vertices
    )) {
        if (!options->workspace) greedy_workspace_release(&local_workspace);
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }

    GreedyWorkerContext context = {
        .pi = pi,
        .prepared = prepared,
        .row_weights = row_weights,
        .root_scores = root_scores,
        .anchors = anchors,
        .anchor_count = anchor_count,
        .variants_per_anchor = variants_per_anchor,
        .prefix_beam_width = prefix_beam_width,
        .prefix_beam_vertices = prefix_beam_vertices,
        .candidate_count = candidate_count,
        .ninputs = ninputs,
        .on_rows = on_rows,
        .off_rows = off_rows,
        .on_words = on_words,
        .off_words = off_words,
        .max_support = options->max_support,
        .initial_weight = initial_weight,
        .threshold = options->reduced_cost_threshold,
        .tolerance = options->reduced_cost_tolerance,
        .deadline_nanoseconds = options->deadline_nanoseconds,
        .batch = batch,
        .attempted = attempted,
        .valid = valid,
        .literal_evaluations = literal_evaluations,
        .remaining = remaining,
        .compatible = compatible,
        .blocked = blocked,
        .cleanup_prefix = cleanup_prefix,
        .selected = selected,
        .selection_order = selection_order
    };
    if (!ccubes_parallel_for(
        0,
        (uint64_t)candidate_count,
        threads,
        false,
        greedy_construct_range,
        &context
    )) {
        if (!options->workspace) greedy_workspace_release(&local_workspace);
        return PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY;
    }

    int output_count = 0;
    uint64_t support_sum = 0;
    stats->support_minimum = ninputs + 1;
    for (int rank = 0; rank < candidate_count; ++rank) {
        stats->anchors_attempted +=
            attempted[rank] != 0 &&
            rank % variants_per_anchor == 0;
        stats->candidates_constructed +=
            attempted[rank] != 0 &&
            batch->support_sizes[rank] > 0;
        stats->literal_evaluations += literal_evaluations[rank];
        if (!valid[rank]) continue;
        stats->improving_candidates++;
        int support = batch->support_sizes[rank];
        if (support < stats->support_minimum) {
            stats->support_minimum = support;
        }
        if (support > stats->support_maximum) {
            stats->support_maximum = support;
        }
        support_sum += (uint64_t)support;
        if (batch->weights[rank] > stats->best_weight) {
            stats->best_weight = batch->weights[rank];
        }
        if (output_count != rank) {
            batch->weights[output_count] = batch->weights[rank];
            batch->anchors[output_count] = batch->anchors[rank];
            batch->support_sizes[output_count] =
                batch->support_sizes[rank];
            memcpy(
                &batch->vertices[
                    (size_t)output_count * (size_t)ninputs
                ],
                &batch->vertices[(size_t)rank * (size_t)ninputs],
                (size_t)ninputs * sizeof(int)
            );
            memcpy(
                &batch->coverage[
                    (size_t)output_count * (size_t)on_words
                ],
                &batch->coverage[(size_t)rank * (size_t)on_words],
                (size_t)on_words * sizeof(uint64_t)
            );
        }
        output_count++;
    }
    batch->count = output_count;
    if (output_count == 0) {
        stats->support_minimum = 0;
    } else {
        stats->support_mean =
            (double)support_sum / (double)output_count;
    }
    stats->effective_threads = threads;
    stats->prefix_beam_width = prefix_beam_width;
    stats->allocated_bytes = allocated_bytes;
    stats->deadline_reached = greedy_deadline_reached(
        options->deadline_nanoseconds
    );
    stats->elapsed_nanoseconds =
        greedy_monotonic_nanoseconds() - started;

    if (!options->workspace) greedy_workspace_release(&local_workspace);
    return PROGRESSIVE_GREEDY_PRICE_OK;
}
