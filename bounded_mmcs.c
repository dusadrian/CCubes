/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "bounded_mmcs.h"
#include "ccubes_threads.h"

#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WEIGHTED_MMCS_REDUCED_COST_TOL 1e-8
#define WEIGHTED_MMCS_TIMEOUT_BOUND_WORK UINT64_C(1000000)

typedef struct {
    int ninputs;
    int edge_count;
    int critical_edge_count;
    int edge_words;
    int target_size;
    bool exact_size;
    const uint64_t *edges;
    const WeightedMMCSPreparedRows *prepared;
    const uint64_t *anchor_bits;
    int anchor_index;
    bool transposed_incidence;

    unsigned char *selected;
    unsigned char *forbidden;
    int *selected_vertices;
    int selected_count;

    int *hit_count;
    int *critical_count;
    int uncovered_count;

    /*
     * Incremental indices.  Both hot loops used to be O(ninputs * edge_count)
     * per search node: choose_uncovered_edge() counted allowed vertices by
     * scanning every vertex of every uncovered edge, and the max_cover bound
     * rescanned every (vertex, edge) pair.  These maintain the same two
     * quantities under add/remove/forbid instead.
     *
     *   inc_edges/inc_offset : for each vertex, the edges containing it
     *   cover[v]             : uncovered edges containing v (exact)
     *   allowed[e]           : non-forbidden vertices in e (exact)
     *
     * choose_uncovered_edge() becomes O(edge_count) and the bound O(ninputs).
     */
    int *inc_offset;    /* ninputs + 1 */
    int *inc_edges;     /* total incidence entries */
    int *cover;         /* ninputs */
    int *allowed;       /* edge_count */
    /*
     * Selected and forbidden vertices are ignored by the max-cover bound.
     * Weighted pricing tracks them as packed bits so edge transitions update
     * cover[] only for vertices whose counts can currently be queried.
     */
    uint64_t *inactive_vertices; /* edge_words; optional */

    bool (*emit)(const int *vertices, int count, void *data);
    void *emit_data;
    bool (*prune)(const int *vertices, int count, void *data);
    double (*branch_priority)(
        const int *vertices,
        int count,
        int candidate,
        void *data
    );
    void *search_data;
    BoundedMMCSStats *stats;
    WeightedMMCSPriceTelemetry *telemetry;
    uint64_t node_limit;
    uint64_t deadline_nanoseconds;
    bool disable_cover_bound;
    bool limit_reached;
    bool failed;
} MMCSState;

typedef struct {
    PIstorage *pi;
    int ninputs;
    const int *anchor;
    const int *word_index;
    const int *bit_index;
    const uint64_t *shifted_mask;
    int implicant_words;

    size_t count;
    size_t capacity;
    size_t table_size;
    int *slots;
    uint64_t *fixed_bits;
    uint64_t *value_bits;
    uint64_t *coverage;
    int *covsum;

    BoundedMMCSStats *stats;
} CubeCollector;

typedef struct {
    const PIstorage *pi;
    int ninputs;
    const int *anchor;
    int anchor_index;
    const double *row_weights;
    int *best_vertices;
    double best_weight;
    int best_anchor;
    int best_support_size;
    uint64_t weight_prunes;
    WeightedMMCSPriceBatch *local_batch;
    WeightedMMCSPriceTelemetry *telemetry;
    bool collect_timing;
} WeightedPriceContext;

typedef struct {
    const PIstorage *pi;
    int ninputs;
    int max_level;
    int edge_words;
    int max_edges;
    const double *row_weights;
    const double *root_upper_bounds;
    const WeightedMMCSPreparedRows *prepared;
    bool collect_timing;
    bool apply_root_gate;
    bool use_materialized_backend;
    bool cache_edge_words;
    bool disable_cover_bound;

    uint64_t *edges;
    unsigned char *selected;
    unsigned char *forbidden;
    int *selected_vertices;
    int *hit_count;
    int *critical_count;
    int *inc_offset;
    int *inc_cursor;
    int *cover;
    int *allowed;
    uint64_t *inactive_vertices;
    int *inc_edges;
    size_t inc_edges_capacity;
    size_t peak_incidence_entries;
    int *best_vertices;

    WeightedMMCSPriceBatch local_batch;

    BoundedMMCSStats search_stats;
    WeightedMMCSPriceTelemetry telemetry;
    WeightedPriceContext price;
    WeightedMMCSPriceStatus status;
    int anchors_completed;
    int first_incomplete_anchor;
} WeightedPriceWorker;

typedef struct {
    WeightedPriceWorker *workers;
    atomic_int next_anchor;
    atomic_bool limit_reached;
    int anchor_count;
    uint64_t deadline_nanoseconds;
} WeightedParallelContext;

static uint64_t monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static void add_price_telemetry(
    WeightedMMCSPriceTelemetry *target,
    const WeightedMMCSPriceTelemetry *source
) {
    target->anchors_presented += source->anchors_presented;
    target->anchors_constructed += source->anchors_constructed;
    target->prospective_gated_anchors +=
        source->prospective_gated_anchors;
    target->edges_constructed += source->edges_constructed;
    target->xor_words += source->xor_words;
    target->incidences_counted += source->incidences_counted;
    target->incidences_written += source->incidences_written;
    target->prospective_gated_edges += source->prospective_gated_edges;
    target->prospective_gated_incidences +=
        source->prospective_gated_incidences;
    target->pivot_selections += source->pivot_selections;
    target->pivot_edge_entries_inspected +=
        source->pivot_edge_entries_inspected;
    target->literal_incidence_entries_visited +=
        source->literal_incidence_entries_visited;
    target->edge_literal_entries_visited +=
        source->edge_literal_entries_visited;
    target->compatible_bound_calls += source->compatible_bound_calls;
    target->compatible_bound_rows_examined +=
        source->compatible_bound_rows_examined;
    target->edge_build_nanoseconds += source->edge_build_nanoseconds;
    target->incidence_count_nanoseconds +=
        source->incidence_count_nanoseconds;
    target->incidence_fill_nanoseconds +=
        source->incidence_fill_nanoseconds;
    target->search_nanoseconds += source->search_nanoseconds;
    target->candidate_collection_nanoseconds +=
        source->candidate_collection_nanoseconds;
    target->merge_nanoseconds += source->merge_nanoseconds;
    target->pricing_call_nanoseconds += source->pricing_call_nanoseconds;
}

static uint64_t low_bits_mask(int count) {
    if (count <= 0) return 0;
    if (count >= 64) return UINT64_MAX;
    return (UINT64_C(1) << count) - 1u;
}

static const uint64_t *mmcs_other_row_bits(
    const MMCSState *state,
    int edge
) {
    if (edge < state->critical_edge_count) {
        return &state->prepared->off_bits[
            (size_t)edge * (size_t)state->edge_words
        ];
    }
    return &state->prepared->on_bits[
        (size_t)(edge - state->critical_edge_count) *
        (size_t)state->edge_words
    ];
}

static uint64_t mmcs_edge_word(
    const MMCSState *state,
    int edge,
    int word
) {
    if (state->edges) {
        return state->edges[
            (size_t)edge * (size_t)state->edge_words + (size_t)word
        ];
    }
    return state->anchor_bits[word] ^
        mmcs_other_row_bits(state, edge)[word];
}

static bool edge_contains(
    const MMCSState *state,
    int edge,
    int vertex
) {
    return (
        mmcs_edge_word(state, edge, vertex / 64) &
        (UINT64_C(1) << (vertex % 64))
    ) != 0;
}

typedef struct {
    const MMCSState *state;
    int vertex;
    int reference_index;
    int reference_end;
    int part;
    int word;
    int word_count;
    int row_limit;
    int edge_base;
    uint64_t values;
} MMCSIncidenceIterator;

static void incidence_iterator_select_part(
    MMCSIncidenceIterator *iterator,
    int part
) {
    const MMCSState *state = iterator->state;
    iterator->part = part;
    iterator->word = 0;
    iterator->values = 0;
    if (part == 0) {
        iterator->row_limit = state->critical_edge_count;
        iterator->word_count = state->prepared->off_column_words;
        iterator->edge_base = 0;
    } else {
        iterator->row_limit = state->anchor_index;
        iterator->word_count = (state->anchor_index + 63) / 64;
        iterator->edge_base = state->critical_edge_count;
    }
}

static void incidence_iterator_init(
    MMCSIncidenceIterator *iterator,
    const MMCSState *state,
    int vertex
) {
    memset(iterator, 0, sizeof(*iterator));
    iterator->state = state;
    iterator->vertex = vertex;
    if (!state->transposed_incidence) {
        iterator->reference_index = state->inc_offset[vertex];
        iterator->reference_end = state->inc_offset[vertex + 1];
        iterator->part = 2;
        return;
    }
    incidence_iterator_select_part(iterator, 0);
}

static bool incidence_iterator_next(
    MMCSIncidenceIterator *iterator,
    int *edge
) {
    const MMCSState *state = iterator->state;
    if (!state->transposed_incidence) {
        if (iterator->reference_index >= iterator->reference_end) {
            return false;
        }
        *edge = state->inc_edges[iterator->reference_index++];
        return true;
    }

    while (iterator->part < 2) {
        if (iterator->values != 0) {
            int bit = __builtin_ctzll(iterator->values);
            iterator->values &= iterator->values - 1;
            *edge = iterator->edge_base +
                (iterator->word - 1) * 64 + bit;
            return true;
        }
        if (iterator->word >= iterator->word_count) {
            incidence_iterator_select_part(
                iterator,
                iterator->part + 1
            );
            continue;
        }

        const uint64_t *columns = iterator->part == 0
            ? state->prepared->off_columns
            : state->prepared->on_columns;
        int column_words = iterator->part == 0
            ? state->prepared->off_column_words
            : state->prepared->on_column_words;
        uint64_t values = columns[
            (size_t)iterator->vertex * (size_t)column_words +
            (size_t)iterator->word
        ];
        bool anchor_is_one = (
            state->anchor_bits[iterator->vertex / 64] &
            (UINT64_C(1) << (iterator->vertex % 64))
        ) != 0;
        if (anchor_is_one) values = ~values;
        int remaining = iterator->row_limit - iterator->word * 64;
        if (remaining < 64) values &= low_bits_mask(remaining);
        iterator->word++;
        iterator->values = values;
    }
    return false;
}

static int unique_selected_on_edge(
    const MMCSState *state,
    int edge,
    int excluded_vertex
) {
    for (int i = 0; i < state->selected_count; ++i) {
        int vertex = state->selected_vertices[i];
        if (vertex != excluded_vertex && edge_contains(state, edge, vertex)) {
            return vertex;
        }
    }
    return -1;
}

/* an edge just became covered: it no longer contributes to any cover[] */
static void edge_now_covered(MMCSState *state, int edge) {
    if (state->disable_cover_bound) return;
    for (int w = 0; w < state->edge_words; ++w) {
        uint64_t word = mmcs_edge_word(state, edge, w);
        if (state->inactive_vertices) {
            word &= ~state->inactive_vertices[w];
        }
        while (word) {
            int v = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            if (state->telemetry) {
                state->telemetry->edge_literal_entries_visited++;
            }
            state->cover[v]--;
        }
    }
}

/* an edge just became uncovered again on backtrack */
static void edge_now_uncovered(MMCSState *state, int edge) {
    if (state->disable_cover_bound) return;
    for (int w = 0; w < state->edge_words; ++w) {
        uint64_t word = mmcs_edge_word(state, edge, w);
        if (state->inactive_vertices) {
            word &= ~state->inactive_vertices[w];
        }
        while (word) {
            int v = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            if (state->telemetry) {
                state->telemetry->edge_literal_entries_visited++;
            }
            state->cover[v]++;
        }
    }
}

static bool add_vertex(
    MMCSState *state,
    int vertex
) {
    bool valid = true;
    state->selected[vertex] = 1;
    if (state->inactive_vertices) {
        state->inactive_vertices[vertex / 64] |=
            UINT64_C(1) << (vertex % 64);
    }
    state->selected_vertices[state->selected_count++] = vertex;

    MMCSIncidenceIterator iterator;
    incidence_iterator_init(&iterator, state, vertex);
    int edge = 0;
    while (incidence_iterator_next(&iterator, &edge)) {
        if (state->telemetry) {
            state->telemetry->literal_incidence_entries_visited++;
        }

        if (state->hit_count[edge] == 0) {
            state->uncovered_count--;
            edge_now_covered(state, edge);
            /*
             * Only OFF edges establish primality.  Canonical earlier-ON
             * edges constrain ownership, but they must never give a selected
             * literal the private edge needed to call the cube prime.
             */
            if (edge < state->critical_edge_count) {
                state->critical_count[vertex]++;
            }
        } else if (state->hit_count[edge] == 1) {
            if (edge < state->critical_edge_count) {
                int previous = unique_selected_on_edge(state, edge, vertex);
                if (previous < 0) {
                    state->failed = true;
                    valid = false;
                } else {
                    state->critical_count[previous]--;
                }
            }
        }
        state->hit_count[edge]++;
    }

    /*
     * MMCS primality pruning: once a selected vertex has no private OFF edge,
     * adding more vertices cannot make it private again.  In particular, a
     * private canonical ownership edge is intentionally insufficient here.
     */
    for (int i = 0; valid && i < state->selected_count; ++i) {
        if (state->critical_count[state->selected_vertices[i]] == 0) {
            valid = false;
        }
    }
    return valid;
}

static void remove_vertex(
    MMCSState *state,
    int vertex
) {
    int restored_cover = 0;
    MMCSIncidenceIterator iterator;
    incidence_iterator_init(&iterator, state, vertex);
    int edge = 0;
    while (incidence_iterator_next(&iterator, &edge)) {
        if (state->telemetry) {
            state->telemetry->literal_incidence_entries_visited++;
        }

        if (state->hit_count[edge] == 1) {
            if (!state->disable_cover_bound) restored_cover++;
            if (edge < state->critical_edge_count) {
                state->critical_count[vertex]--;
            }
            state->uncovered_count++;
            edge_now_uncovered(state, edge);
        } else if (state->hit_count[edge] == 2) {
            if (edge < state->critical_edge_count) {
                int remaining = unique_selected_on_edge(state, edge, vertex);
                if (remaining >= 0) state->critical_count[remaining]++;
            }
        }
        state->hit_count[edge]--;
    }

    state->selected_count--;
    state->selected[vertex] = 0;
    if (state->inactive_vertices) {
        state->cover[vertex] = restored_cover;
        state->inactive_vertices[vertex / 64] &=
            ~(UINT64_C(1) << (vertex % 64));
    }
    state->critical_count[vertex] = 0;
}

/* forbidding a vertex removes it from every edge's allowed count */
static void forbid_vertex(MMCSState *state, int vertex) {
    state->forbidden[vertex] = 1;
    if (state->inactive_vertices) {
        state->inactive_vertices[vertex / 64] |=
            UINT64_C(1) << (vertex % 64);
    }
    MMCSIncidenceIterator iterator;
    incidence_iterator_init(&iterator, state, vertex);
    int edge = 0;
    while (incidence_iterator_next(&iterator, &edge)) {
        if (state->telemetry) {
            state->telemetry->literal_incidence_entries_visited++;
        }
        state->allowed[edge]--;
    }
}

static void unforbid_vertex(MMCSState *state, int vertex) {
    int restored_cover = 0;
    MMCSIncidenceIterator iterator;
    incidence_iterator_init(&iterator, state, vertex);
    int edge = 0;
    while (incidence_iterator_next(&iterator, &edge)) {
        if (state->telemetry) {
            state->telemetry->literal_incidence_entries_visited++;
        }
        state->allowed[edge]++;
        if (!state->disable_cover_bound && state->hit_count[edge] == 0) {
            restored_cover++;
        }
    }
    if (state->inactive_vertices) {
        state->cover[vertex] = restored_cover;
        state->inactive_vertices[vertex / 64] &=
            ~(UINT64_C(1) << (vertex % 64));
    }
    state->forbidden[vertex] = 0;
}

static int choose_uncovered_edge(
    const MMCSState *state
) {
    int best_edge = -1;
    int best_allowed = INT_MAX;
    if (state->telemetry) state->telemetry->pivot_selections++;

    for (int edge = 0; edge < state->edge_count; ++edge) {
        if (state->telemetry) {
            state->telemetry->pivot_edge_entries_inspected++;
        }
        if (state->hit_count[edge] != 0) continue;

        int allowed = state->allowed[edge];
        if (allowed < best_allowed) {
            best_allowed = allowed;
            best_edge = edge;
            if (allowed == 0) break;
        }
    }

    return best_edge;
}

static void mmcs_search(
    MMCSState *state
) {
    if (state->failed || state->limit_reached) return;
    if (
        state->deadline_nanoseconds > 0 &&
        (state->stats->search_nodes & UINT64_C(1023)) == 0 &&
        monotonic_nanoseconds() >= state->deadline_nanoseconds
    ) {
        state->limit_reached = true;
        return;
    }
    if (
        state->node_limit > 0 &&
        state->stats->search_nodes >= state->node_limit
    ) {
        state->limit_reached = true;
        return;
    }
    state->stats->search_nodes++;

    if (
        state->prune &&
        state->prune(
            state->selected_vertices,
            state->selected_count,
            state->search_data
        )
    ) {
        return;
    }

    if (state->uncovered_count == 0) {
        if (!state->exact_size || state->selected_count == state->target_size) {
            state->stats->completed_transversals++;
            if (!state->emit(
                state->selected_vertices,
                state->selected_count,
                state->emit_data
            )) {
                state->failed = true;
            }
        }
        return;
    }

    if (state->selected_count >= state->target_size) return;

    /*
     * A selected vertex can cover at most max_cover currently uncovered
     * edges.  Even under the optimistic assumption that every later vertex
     * achieves that maximum, too few remaining slots cannot finish the
     * transversal.  The one-slot case becomes the useful exact test that one
     * allowed input must occur in every remaining blocker edge.
     */
    if (!state->disable_cover_bound) {
        int max_cover = 0;
        for (int vertex = 0; vertex < state->ninputs; ++vertex) {
            if (state->selected[vertex] || state->forbidden[vertex]) {
                continue;
            }
            if (state->cover[vertex] > max_cover) {
                max_cover = state->cover[vertex];
            }
        }
        int slots = state->target_size - state->selected_count;
        if (
            max_cover == 0 ||
            (state->uncovered_count + max_cover - 1) / max_cover > slots
        ) {
            return;
        }
    }

    int pivot = choose_uncovered_edge(state);
    if (pivot < 0) {
        state->failed = true;
        return;
    }

    int branch_vertices[state->ninputs];

    int branch_count = 0;
    for (int vertex = 0; vertex < state->ninputs; ++vertex) {
        if (
            !state->forbidden[vertex] &&
            edge_contains(state, pivot, vertex)
        ) {
            branch_vertices[branch_count++] = vertex;
        }
    }

    if (state->branch_priority && branch_count > 1) {
        double priorities[state->ninputs];
        for (int i = 0; i < branch_count; ++i) {
            priorities[i] = state->branch_priority(
                state->selected_vertices,
                state->selected_count,
                branch_vertices[i],
                state->search_data
            );
        }
        for (int i = 1; i < branch_count; ++i) {
            int vertex = branch_vertices[i];
            double priority = priorities[i];
            int j = i;
            while (
                j > 0 &&
                (
                    priority > priorities[j - 1] ||
                    (
                        priority == priorities[j - 1] &&
                        vertex < branch_vertices[j - 1]
                    )
                )
            ) {
                branch_vertices[j] = branch_vertices[j - 1];
                priorities[j] = priorities[j - 1];
                --j;
            }
            branch_vertices[j] = vertex;
            priorities[j] = priority;
        }
    }

    for (
        int i = 0;
        i < branch_count && !state->failed && !state->limit_reached;
        ++i
    ) {
        int vertex = branch_vertices[i];
        bool minimal = add_vertex(state, vertex);
        if (minimal) mmcs_search(state);
        remove_vertex(state, vertex);

        /*
         * Canonical branching: later branches exclude earlier vertices from
         * this pivot edge.  Every transversal is therefore emitted through
         * the branch containing its first selected pivot vertex.
         */
        forbid_vertex(state, vertex);
    }

    for (int i = 0; i < branch_count; ++i) {
        unforbid_vertex(state, branch_vertices[i]);
    }
}

static bool row_matches_partial_cube(
    const WeightedPriceContext *context,
    int row,
    const int *vertices,
    int count,
    int candidate
) {
    const int *on_row = &context->pi->ON_set[
        (size_t)row * (size_t)context->ninputs
    ];
    for (int i = 0; i < count; ++i) {
        int vertex = vertices[i];
        if (on_row[vertex] != context->anchor[vertex]) return false;
    }
    return candidate < 0 ||
        on_row[candidate] == context->anchor[candidate];
}

static double compatible_positive_weight(
    const WeightedPriceContext *context,
    const int *vertices,
    int count,
    int candidate
) {
    double bound = 0.0;
    if (context->telemetry) {
        context->telemetry->compatible_bound_calls++;
    }
    /*
     * Canonical ownership forces every completed cube for this anchor to
     * exclude all earlier ON rows. They can therefore never contribute to a
     * descendant's priced weight, even if they still match the partial cube.
     */
    for (
        int row = context->anchor_index;
        row < context->pi->ON_minterms;
        ++row
    ) {
        if (context->telemetry) {
            context->telemetry->compatible_bound_rows_examined++;
        }
        if (
            context->row_weights[row] > 0.0 &&
            row_matches_partial_cube(
                context,
                row,
                vertices,
                count,
                candidate
            )
        ) {
            bound += context->row_weights[row];
        }
    }
    return bound;
}

static int compare_vertex_sets(
    const int *a,
    int a_count,
    const int *b,
    int b_count,
    int ninputs
) {
    for (int vertex = 0; vertex < ninputs; ++vertex) {
        bool in_a = false;
        bool in_b = false;
        for (int i = 0; i < a_count; ++i) {
            if (a[i] == vertex) {
                in_a = true;
                break;
            }
        }
        for (int i = 0; i < b_count; ++i) {
            if (b[i] == vertex) {
                in_b = true;
                break;
            }
        }
        if (in_a != in_b) return in_a ? -1 : 1;
    }
    return 0;
}

static bool weighted_geometry_precedes(
    int anchor,
    int support_size,
    const int *vertices,
    int existing_anchor,
    int existing_support_size,
    const int *existing_vertices,
    int ninputs
) {
    if (anchor != existing_anchor) return anchor < existing_anchor;
    if (support_size != existing_support_size) {
        return support_size < existing_support_size;
    }
    return compare_vertex_sets(
        vertices,
        support_size,
        existing_vertices,
        existing_support_size,
        ninputs
    ) < 0;
}

static bool weighted_batch_candidate_precedes(
    double weight,
    int anchor,
    const uint64_t *coverage,
    double existing_weight,
    int existing_anchor,
    const uint64_t *existing_coverage,
    int coverage_words
) {
    if (weight != existing_weight) return weight > existing_weight;
    if (anchor != existing_anchor) return anchor < existing_anchor;
    for (int word = 0; word < coverage_words; ++word) {
        if (coverage[word] != existing_coverage[word]) {
            return coverage[word] < existing_coverage[word];
        }
    }
    return false;
}

static void weighted_batch_store(
    WeightedMMCSPriceBatch *batch,
    int index,
    int ninputs,
    int coverage_words,
    double weight,
    int anchor,
    int support_size,
    const int *vertices,
    const uint64_t *coverage
) {
    batch->weights[index] = weight;
    batch->anchors[index] = anchor;
    batch->support_sizes[index] = support_size;
    memset(
        &batch->vertices[(size_t)index * (size_t)ninputs],
        0,
        (size_t)ninputs * sizeof(int)
    );
    memcpy(
        &batch->vertices[(size_t)index * (size_t)ninputs],
        vertices,
        (size_t)support_size * sizeof(int)
    );
    memcpy(
        &batch->coverage[(size_t)index * (size_t)coverage_words],
        coverage,
        (size_t)coverage_words * sizeof(uint64_t)
    );
}

static void weighted_batch_offer(
    WeightedMMCSPriceBatch *batch,
    int ninputs,
    int coverage_words,
    double weight,
    int anchor,
    int support_size,
    const int *vertices,
    const uint64_t *coverage
) {
    for (int i = 0; i < batch->count; ++i) {
        uint64_t *existing_coverage = &batch->coverage[
            (size_t)i * (size_t)coverage_words
        ];
        if (memcmp(
            existing_coverage,
            coverage,
            (size_t)coverage_words * sizeof(uint64_t)
        ) != 0) {
            continue;
        }
        const int *existing_vertices = &batch->vertices[
            (size_t)i * (size_t)ninputs
        ];
        if (weighted_geometry_precedes(
            anchor,
            support_size,
            vertices,
            batch->anchors[i],
            batch->support_sizes[i],
            existing_vertices,
            ninputs
        )) {
            weighted_batch_store(
                batch,
                i,
                ninputs,
                coverage_words,
                weight,
                anchor,
                support_size,
                vertices,
                coverage
            );
        }
        return;
    }

    int insert = batch->count;
    while (insert > 0 && weighted_batch_candidate_precedes(
        weight,
        anchor,
        coverage,
        batch->weights[insert - 1],
        batch->anchors[insert - 1],
        &batch->coverage[
            (size_t)(insert - 1) * (size_t)coverage_words
        ],
        coverage_words
    )) {
        insert--;
    }
    if (insert >= batch->capacity) return;

    int old_count = batch->count;
    if (old_count < batch->capacity) batch->count++;
    int move_end = old_count < batch->capacity
        ? old_count
        : batch->capacity - 1;
    for (int i = move_end; i > insert; --i) {
        batch->weights[i] = batch->weights[i - 1];
        batch->anchors[i] = batch->anchors[i - 1];
        batch->support_sizes[i] = batch->support_sizes[i - 1];
        memcpy(
            &batch->vertices[(size_t)i * (size_t)ninputs],
            &batch->vertices[
                (size_t)(i - 1) * (size_t)ninputs
            ],
            (size_t)ninputs * sizeof(int)
        );
        memcpy(
            &batch->coverage[(size_t)i * (size_t)coverage_words],
            &batch->coverage[
                (size_t)(i - 1) * (size_t)coverage_words
            ],
            (size_t)coverage_words * sizeof(uint64_t)
        );
    }
    weighted_batch_store(
        batch,
        insert,
        ninputs,
        coverage_words,
        weight,
        anchor,
        support_size,
        vertices,
        coverage
    );
}

static uint64_t weighted_coverage_hash(
    const uint64_t *coverage,
    int words
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int word = 0; word < words; ++word) {
        hash ^= coverage[word];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static bool weighted_merge_local_batches(
    WeightedPriceWorker *workers,
    int nthreads,
    WeightedMMCSPriceBatch *batch,
    int ninputs,
    int coverage_words
) {
    if (!batch) return true;
    batch->count = 0;

    size_t table_size = 2u;
    size_t needed = (size_t)batch->capacity * 2u;
    while (table_size < needed) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size *= 2u;
    }
    int *slots = (int *)malloc(table_size * sizeof(int));
    int *cursor = (int *)calloc((size_t)nthreads, sizeof(int));
    if (!slots || !cursor) {
        free(slots);
        free(cursor);
        return false;
    }
    for (size_t slot = 0; slot < table_size; ++slot) slots[slot] = -1;

    size_t mask = table_size - 1u;
    while (batch->count < batch->capacity) {
        int best_worker = -1;
        for (int worker = 0; worker < nthreads; ++worker) {
            WeightedMMCSPriceBatch *local = &workers[worker].local_batch;
            if (cursor[worker] >= local->count) continue;
            if (best_worker < 0) {
                best_worker = worker;
                continue;
            }
            WeightedMMCSPriceBatch *best =
                &workers[best_worker].local_batch;
            int candidate = cursor[worker];
            int incumbent = cursor[best_worker];
            if (weighted_batch_candidate_precedes(
                local->weights[candidate],
                local->anchors[candidate],
                &local->coverage[
                    (size_t)candidate * (size_t)coverage_words
                ],
                best->weights[incumbent],
                best->anchors[incumbent],
                &best->coverage[
                    (size_t)incumbent * (size_t)coverage_words
                ],
                coverage_words
            )) {
                best_worker = worker;
            }
        }
        if (best_worker < 0) break;

        WeightedMMCSPriceBatch *local =
            &workers[best_worker].local_batch;
        int candidate = cursor[best_worker]++;
        const uint64_t *coverage = &local->coverage[
            (size_t)candidate * (size_t)coverage_words
        ];
        size_t slot =
            (size_t)weighted_coverage_hash(coverage, coverage_words) &
            mask;
        while (slots[slot] >= 0) {
            int existing = slots[slot];
            uint64_t *existing_coverage = &batch->coverage[
                (size_t)existing * (size_t)coverage_words
            ];
            if (memcmp(
                existing_coverage,
                coverage,
                (size_t)coverage_words * sizeof(uint64_t)
            ) == 0) {
                const int *vertices = &local->vertices[
                    (size_t)candidate * (size_t)ninputs
                ];
                const int *existing_vertices = &batch->vertices[
                    (size_t)existing * (size_t)ninputs
                ];
                if (weighted_geometry_precedes(
                    local->anchors[candidate],
                    local->support_sizes[candidate],
                    vertices,
                    batch->anchors[existing],
                    batch->support_sizes[existing],
                    existing_vertices,
                    ninputs
                )) {
                    weighted_batch_store(
                        batch,
                        existing,
                        ninputs,
                        coverage_words,
                        local->weights[candidate],
                        local->anchors[candidate],
                        local->support_sizes[candidate],
                        vertices,
                        coverage
                    );
                }
                goto NEXT_MERGED_CANDIDATE;
            }
            slot = (slot + 1u) & mask;
        }

        int target = batch->count++;
        weighted_batch_store(
            batch,
            target,
            ninputs,
            coverage_words,
            local->weights[candidate],
            local->anchors[candidate],
            local->support_sizes[candidate],
            &local->vertices[
                (size_t)candidate * (size_t)ninputs
            ],
            coverage
        );
        slots[slot] = target;

NEXT_MERGED_CANDIDATE:
        ;
    }
    free(slots);
    free(cursor);
    return true;
}

static bool weighted_price_prune(
    const int *vertices,
    int count,
    void *data
) {
    WeightedPriceContext *context = (WeightedPriceContext *)data;
    if (!isfinite(context->best_weight)) return false;

    double threshold = context->best_weight;
    bool prune_equal = true;
    WeightedMMCSPriceBatch *local = context->local_batch;
    if (local && local->count > 0) {
        if (local->count < local->capacity) {
            threshold = 1.0;
        } else {
            threshold = local->weights[local->count - 1];
            prune_equal = false;
        }
    }
    double bound = compatible_positive_weight(
        context,
        vertices,
        count,
        -1
    );
    if (
        bound < threshold - 1e-12 ||
        (prune_equal && bound <= threshold + 1e-12)
    ) {
        context->weight_prunes++;
        return true;
    }
    return false;
}

static double weighted_branch_priority(
    const int *vertices,
    int count,
    int candidate,
    void *data
) {
    const WeightedPriceContext *context =
        (const WeightedPriceContext *)data;
    return compatible_positive_weight(
        context,
        vertices,
        count,
        candidate
    );
}

static bool collect_weighted_price(
    const int *vertices,
    int count,
    void *data
) {
    WeightedPriceContext *context = (WeightedPriceContext *)data;
    uint64_t started = context->collect_timing
        ? monotonic_nanoseconds()
        : 0;
    double weight = 0.0;
    int coverage_words = context->pi->pichart_words;
    int coverage_cov_bits = context->pi->cov_bits;
    uint64_t coverage[
        coverage_words > 0 ? coverage_words : 1
    ];
    memset(coverage, 0, sizeof(coverage));
    for (int row = 0; row < context->pi->ON_minterms; ++row) {
        if (row_matches_partial_cube(
            context,
            row,
            vertices,
            count,
            -1
        )) {
            weight += context->row_weights[row];
            if (coverage_words > 0 && coverage_cov_bits > 0) {
                coverage[row / coverage_cov_bits] |=
                    UINT64_C(1) << (row % coverage_cov_bits);
            }
        }
    }

    if (
        weight > context->best_weight ||
        (
            weight == context->best_weight &&
            weighted_geometry_precedes(
                context->anchor_index,
                count,
                vertices,
                context->best_anchor,
                context->best_support_size,
                context->best_vertices,
                context->ninputs
            )
        )
    ) {
        context->best_weight = weight;
        context->best_anchor = context->anchor_index;
        context->best_support_size = count;
        memcpy(
            context->best_vertices,
            vertices,
            (size_t)count * sizeof(int)
        );
    }

    WeightedMMCSPriceBatch *local = context->local_batch;
    if (!local || weight <= 1.0 + WEIGHTED_MMCS_REDUCED_COST_TOL) {
        if (context->collect_timing) {
            context->telemetry->candidate_collection_nanoseconds +=
                monotonic_nanoseconds() - started;
        }
        return true;
    }

    weighted_batch_offer(
        local,
        context->ninputs,
        coverage_words,
        weight,
        context->anchor_index,
        count,
        vertices,
        coverage
    );
    if (context->collect_timing) {
        context->telemetry->candidate_collection_nanoseconds +=
            monotonic_nanoseconds() - started;
    }
    return true;
}

static uint64_t geometry_hash(
    const uint64_t *fixed_bits,
    const uint64_t *value_bits,
    int words
) {
    uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
    for (int w = 0; w < words; ++w) {
        hash ^= fixed_bits[w] + UINT64_C(0x9e3779b97f4a7c15) +
            (hash << 6) + (hash >> 2);
        hash ^= value_bits[w] + UINT64_C(0x9e3779b97f4a7c15) +
            (hash << 6) + (hash >> 2);
    }
    return hash;
}

static bool same_geometry(
    const uint64_t *fixed_a,
    const uint64_t *value_a,
    const uint64_t *fixed_b,
    const uint64_t *value_b,
    int words
) {
    return memcmp(
        fixed_a,
        fixed_b,
        (size_t)words * sizeof(uint64_t)
    ) == 0 && memcmp(
        value_a,
        value_b,
        (size_t)words * sizeof(uint64_t)
    ) == 0;
}

static bool collector_rehash(
    CubeCollector *collector,
    size_t requested
) {
    size_t table_size = 16;
    while (table_size < requested) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }

    int *slots = (int *)malloc(table_size * sizeof(int));
    if (!slots) return false;
    for (size_t i = 0; i < table_size; ++i) slots[i] = -1;

    for (size_t cube = 0; cube < collector->count; ++cube) {
        const uint64_t *fixed_bits = &collector->fixed_bits[
            cube * (size_t)collector->implicant_words
        ];
        const uint64_t *value_bits = &collector->value_bits[
            cube * (size_t)collector->implicant_words
        ];
        size_t pos = (size_t)(
            geometry_hash(
                fixed_bits,
                value_bits,
                collector->implicant_words
            ) & (uint64_t)(table_size - 1u)
        );
        while (slots[pos] >= 0) pos = (pos + 1u) & (table_size - 1u);
        slots[pos] = (int)cube;
    }

    free(collector->slots);
    collector->slots = slots;
    collector->table_size = table_size;
    return true;
}

static bool collector_reserve(
    CubeCollector *collector,
    size_t needed
) {
    if (collector->capacity >= needed) return true;

    size_t capacity = collector->capacity ? collector->capacity * 2u : 16u;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }

    size_t iw = (size_t)collector->implicant_words;
    size_t pw = (size_t)collector->pi->pichart_words;
    if (
        capacity > SIZE_MAX / iw ||
        capacity > SIZE_MAX / pw
    ) {
        return false;
    }

    uint64_t *fixed_bits = (uint64_t *)calloc(
        capacity * iw,
        sizeof(uint64_t)
    );
    uint64_t *value_bits = (uint64_t *)calloc(
        capacity * iw,
        sizeof(uint64_t)
    );
    uint64_t *coverage = (uint64_t *)calloc(
        capacity * pw,
        sizeof(uint64_t)
    );
    int *covsum = (int *)calloc(capacity, sizeof(int));

    if (!fixed_bits || !value_bits || !coverage || !covsum) {
        free(fixed_bits);
        free(value_bits);
        free(coverage);
        free(covsum);
        return false;
    }

    if (collector->count > 0) {
        memcpy(
            fixed_bits,
            collector->fixed_bits,
            collector->count * iw * sizeof(uint64_t)
        );
        memcpy(
            value_bits,
            collector->value_bits,
            collector->count * iw * sizeof(uint64_t)
        );
        memcpy(
            coverage,
            collector->coverage,
            collector->count * pw * sizeof(uint64_t)
        );
        memcpy(
            covsum,
            collector->covsum,
            collector->count * sizeof(int)
        );
    }

    free(collector->fixed_bits);
    free(collector->value_bits);
    free(collector->coverage);
    free(collector->covsum);
    collector->fixed_bits = fixed_bits;
    collector->value_bits = value_bits;
    collector->coverage = coverage;
    collector->covsum = covsum;
    collector->capacity = capacity;
    return true;
}

static bool collect_transversal(
    const int *vertices,
    int count,
    void *data
) {
    CubeCollector *collector = (CubeCollector *)data;
    int words = collector->implicant_words;
    uint64_t fixed_bits[words];
    uint64_t value_bits[words];
    memset(fixed_bits, 0, (size_t)words * sizeof(uint64_t));
    memset(value_bits, 0, (size_t)words * sizeof(uint64_t));

    for (int i = 0; i < count; ++i) {
        int vertex = vertices[i];
        fixed_bits[collector->word_index[vertex]] |=
            collector->shifted_mask[vertex];
        int value = collector->anchor[vertex] - 1;
        value_bits[collector->word_index[vertex]] |=
            (uint64_t)value << collector->bit_index[vertex];
    }

    if (
        collector->table_size == 0 ||
        (collector->count + 1u) * 10u >= collector->table_size * 7u
    ) {
        size_t requested = collector->table_size
            ? collector->table_size * 2u
            : 16u;
        if (!collector_rehash(collector, requested)) return false;
    }

    size_t mask = collector->table_size - 1u;
    size_t pos = (size_t)(
        geometry_hash(fixed_bits, value_bits, words) & (uint64_t)mask
    );

    while (collector->slots[pos] >= 0) {
        size_t existing = (size_t)collector->slots[pos];
        if (same_geometry(
            fixed_bits,
            value_bits,
            &collector->fixed_bits[existing * (size_t)words],
            &collector->value_bits[existing * (size_t)words],
            words
        )) {
            collector->stats->duplicate_cubes++;
            return true;
        }
        pos = (pos + 1u) & mask;
    }

    if (!collector_reserve(collector, collector->count + 1u)) return false;
    size_t cube = collector->count;
    memcpy(
        &collector->fixed_bits[cube * (size_t)words],
        fixed_bits,
        (size_t)words * sizeof(uint64_t)
    );
    memcpy(
        &collector->value_bits[cube * (size_t)words],
        value_bits,
        (size_t)words * sizeof(uint64_t)
    );

    uint64_t *coverage = &collector->coverage[
        cube * (size_t)collector->pi->pichart_words
    ];
    int covsum = 0;
    for (int row = 0; row < collector->pi->ON_minterms; ++row) {
        bool covered = true;
        const int *on_row = &collector->pi->ON_set[
            (size_t)row * (size_t)collector->ninputs
        ];
        for (int i = 0; i < count; ++i) {
            int vertex = vertices[i];
            if (on_row[vertex] != collector->anchor[vertex]) {
                covered = false;
                break;
            }
        }
        if (covered) {
            int cov_bits = collector->pi->cov_bits > 0
                ? collector->pi->cov_bits
                : 64;
            coverage[row / cov_bits] |= UINT64_C(1) << (row % cov_bits);
            covsum++;
        }
    }
    collector->covsum[cube] = covsum;
    collector->slots[pos] = (int)cube;
    collector->count++;
    collector->stats->unique_cubes++;
    return true;
}

static void collector_destroy(
    CubeCollector *collector
) {
    free(collector->slots);
    free(collector->fixed_bits);
    free(collector->value_bits);
    free(collector->coverage);
    free(collector->covsum);
    memset(collector, 0, sizeof(*collector));
}

static bool ensure_pi_capacity(
    PIstorage *pi,
    int implicant_words,
    size_t needed
) {
    if (needed <= (size_t)pi->estimPI) return true;

    size_t capacity = pi->estimPI > 0 ? (size_t)pi->estimPI : 16u;
    while (capacity < needed) {
        if (capacity > (size_t)INT_MAX / 2u) {
            capacity = needed;
            break;
        }
        capacity *= 2u;
    }
    if (capacity > (size_t)INT_MAX) return false;

    size_t iw = (size_t)implicant_words;
    size_t pw = (size_t)pi->pichart_words;
    uint64_t *pichart_pos = (uint64_t *)calloc(
        capacity * pw,
        sizeof(uint64_t)
    );
    uint64_t *implicants_pos = (uint64_t *)calloc(
        capacity * iw,
        sizeof(uint64_t)
    );
    uint64_t *implicants_val = (uint64_t *)calloc(
        capacity * iw,
        sizeof(uint64_t)
    );
    int *shared = (int *)calloc(capacity, sizeof(int));
    int *covsum = (int *)calloc(capacity, sizeof(int));
    int *covered = (int *)calloc(capacity, sizeof(int));

    if (
        !pichart_pos ||
        !implicants_pos ||
        !implicants_val ||
        !shared ||
        !covsum ||
        !covered
    ) {
        free(pichart_pos);
        free(implicants_pos);
        free(implicants_val);
        free(shared);
        free(covsum);
        free(covered);
        return false;
    }

    size_t found = pi->foundPI > 0 ? (size_t)pi->foundPI : 0u;
    if (found > 0) {
        memcpy(
            pichart_pos,
            pi->pichart_pos,
            found * pw * sizeof(uint64_t)
        );
        memcpy(
            implicants_pos,
            pi->implicants_pos,
            found * iw * sizeof(uint64_t)
        );
        memcpy(
            implicants_val,
            pi->implicants_val,
            found * iw * sizeof(uint64_t)
        );
        memcpy(shared, pi->shared, found * sizeof(int));
        memcpy(covsum, pi->covsum, found * sizeof(int));
        memcpy(covered, pi->covered, found * sizeof(int));
    }

    free(pi->pichart_pos);
    free(pi->implicants_pos);
    free(pi->implicants_val);
    free(pi->shared);
    free(pi->covsum);
    free(pi->covered);
    pi->pichart_pos = pichart_pos;
    pi->implicants_pos = implicants_pos;
    pi->implicants_val = implicants_val;
    pi->shared = shared;
    pi->covsum = covsum;
    pi->covered = covered;
    pi->estimPI = (int)capacity;
    return true;
}

static BoundedMMCSResult bounded_mmcs_generate_output_target_limited(
    PIstorage *pi,
    int ninputs,
    int level,
    bool exact_size,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    uint64_t node_limit,
    BoundedMMCSStats *stats
) {
    if (
        !pi ||
        ninputs <= 0 ||
        level <= 0 ||
        level > ninputs ||
        !word_index ||
        !bit_index ||
        !shifted_mask ||
        implicant_words <= 0 ||
        !stats ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        !pi->ON_set ||
        !pi->OFF_set ||
        pi->pichart_words <= 0
    ) {
        return BOUNDED_MMCS_ERROR;
    }

    CubeCollector collector;
    memset(&collector, 0, sizeof(collector));
    collector.pi = pi;
    collector.ninputs = ninputs;
    collector.word_index = word_index;
    collector.bit_index = bit_index;
    collector.shifted_mask = shifted_mask;
    collector.implicant_words = implicant_words;
    collector.stats = stats;

    int edge_words = (ninputs + 63) / 64;
    int max_edges = pi->OFF_minterms + pi->ON_minterms - 1;
    uint64_t *edges = (uint64_t *)calloc(
        (size_t)max_edges * (size_t)edge_words,
        sizeof(uint64_t)
    );
    unsigned char *selected = (unsigned char *)calloc(
        (size_t)ninputs,
        sizeof(unsigned char)
    );
    unsigned char *forbidden = (unsigned char *)calloc(
        (size_t)ninputs,
        sizeof(unsigned char)
    );
    int *selected_vertices = (int *)calloc((size_t)ninputs, sizeof(int));
    int *hit_count = (int *)calloc((size_t)max_edges, sizeof(int));
    int *critical_count = (int *)calloc((size_t)ninputs, sizeof(int));
    /* incidence index: at most one entry per (edge, vertex) incidence */
    int *inc_offset = (int *)calloc((size_t)ninputs + 1, sizeof(int));
    int *inc_cursor = (int *)calloc((size_t)ninputs + 1, sizeof(int));
    int *cover = (int *)calloc((size_t)ninputs, sizeof(int));
    int *allowed = (int *)calloc((size_t)max_edges, sizeof(int));
    int *inc_edges = (int *)calloc(
        (size_t)max_edges * (size_t)ninputs,
        sizeof(int)
    );

    if (
        !edges ||
        !selected ||
        !forbidden ||
        !selected_vertices ||
        !hit_count ||
        !critical_count ||
        !inc_offset ||
        !inc_cursor ||
        !cover ||
        !allowed ||
        !inc_edges
    ) {
        free(edges);
        free(selected);
        free(forbidden);
        free(selected_vertices);
        free(hit_count);
        free(critical_count);
        free(inc_offset);
        free(inc_cursor);
        free(cover);
        free(allowed);
        free(inc_edges);
        collector_destroy(&collector);
        return BOUNDED_MMCS_ERROR;
    }

    bool ok = true;
    bool limit_reached = false;
    for (int on = 0; on < pi->ON_minterms && ok; ++on) {
        const int *anchor = &pi->ON_set[
            (size_t)on * (size_t)ninputs
        ];

        /* A duplicate point has the same canonical owner as its first row. */
        bool duplicate_anchor = false;
        for (int earlier = 0; earlier < on; ++earlier) {
            const int *earlier_row = &pi->ON_set[
                (size_t)earlier * (size_t)ninputs
            ];
            if (
                memcmp(
                    anchor,
                    earlier_row,
                    (size_t)ninputs * sizeof(*anchor)
                ) == 0
            ) {
                duplicate_anchor = true;
                break;
            }
        }
        if (duplicate_anchor) continue;

        memset(
            edges,
            0,
            (size_t)max_edges *
                (size_t)edge_words *
                sizeof(uint64_t)
        );

        for (int off = 0; off < pi->OFF_minterms; ++off) {
            const int *off_row = &pi->OFF_set[
                (size_t)off * (size_t)ninputs
            ];
            bool nonempty = false;
            for (int input = 0; input < ninputs; ++input) {
                if (anchor[input] != off_row[input]) {
                    edges[
                        (size_t)off * (size_t)edge_words +
                        (size_t)(input / 64)
                    ] |= UINT64_C(1) << (input % 64);
                    nonempty = true;
                }
            }
            if (!nonempty) {
                ok = false;
                break;
            }
        }
        if (!ok) break;

        /*
         * Canonical anchor assignment: a cube belongs to the first ON row it
         * covers.  Requiring the current anchor to differ on its support from
         * every earlier ON row prevents the same PI from being enumerated
         * again for later covered rows.  These ordering edges constrain the
         * search but do not count as private OFF edges for primality.
         */
        for (int earlier = 0; earlier < on; ++earlier) {
            const int *earlier_row = &pi->ON_set[
                (size_t)earlier * (size_t)ninputs
            ];
            int edge = pi->OFF_minterms + earlier;
            bool nonempty = false;
            for (int input = 0; input < ninputs; ++input) {
                if (anchor[input] != earlier_row[input]) {
                    edges[
                        (size_t)edge * (size_t)edge_words +
                        (size_t)(input / 64)
                    ] |= UINT64_C(1) << (input % 64);
                    nonempty = true;
                }
            }
            if (!nonempty) {
                ok = false;
                break;
            }
        }
        if (!ok) break;

        int edge_count = pi->OFF_minterms + on;
        memset(selected, 0, (size_t)ninputs * sizeof(unsigned char));
        memset(forbidden, 0, (size_t)ninputs * sizeof(unsigned char));
        memset(selected_vertices, 0, (size_t)ninputs * sizeof(int));
        memset(hit_count, 0, (size_t)edge_count * sizeof(int));
        memset(critical_count, 0, (size_t)ninputs * sizeof(int));
        collector.anchor = anchor;

        /*
         * Build the incidence index and the initial cover[]/allowed[] counts
         * for this anchor.  One pass over the edge bitsets; the search then
         * maintains both incrementally.
         */
        memset(inc_offset, 0, (size_t)(ninputs + 1) * sizeof(int));
        for (int e = 0; e < edge_count; ++e) {
            const uint64_t *bits = &edges[(size_t)e * (size_t)edge_words];
            int size = 0;
            for (int w = 0; w < edge_words; ++w) {
                uint64_t word = bits[w];
                while (word) {
                    int v = w * 64 + __builtin_ctzll(word);
                    word &= word - 1;
                    inc_offset[v + 1]++;
                    size++;
                }
            }
            allowed[e] = size;
        }

        for (int v = 0; v < ninputs; ++v) {
            inc_offset[v + 1] += inc_offset[v];
            cover[v] = 0;
        }

        memcpy(inc_cursor, inc_offset, (size_t)ninputs * sizeof(int));

        for (int e = 0; e < edge_count; ++e) {
            const uint64_t *bits = &edges[(size_t)e * (size_t)edge_words];
            for (int w = 0; w < edge_words; ++w) {
                uint64_t word = bits[w];
                while (word) {
                    int v = w * 64 + __builtin_ctzll(word);
                    word &= word - 1;
                    inc_edges[inc_cursor[v]++] = e;
                    cover[v]++;
                }
            }
        }

        MMCSState state = {
            .ninputs = ninputs,
            .edge_count = edge_count,
            /* Minimality is measured against OFF edges, not ownership edges. */
            .critical_edge_count = pi->OFF_minterms,
            .edge_words = edge_words,
            .target_size = level,
            .exact_size = exact_size,
            .edges = edges,
            .selected = selected,
            .forbidden = forbidden,
            .selected_vertices = selected_vertices,
            .selected_count = 0,
            .hit_count = hit_count,
            .critical_count = critical_count,
            .uncovered_count = edge_count,
            .inc_offset = inc_offset,
            .inc_edges = inc_edges,
            .cover = cover,
            .allowed = allowed,
            .emit = collect_transversal,
            .emit_data = &collector,
            .stats = stats,
            .node_limit = node_limit,
            .limit_reached = false,
            .failed = false
        };
        mmcs_search(&state);
        ok = !state.failed;
        limit_reached = state.limit_reached;
        if (limit_reached) break;
    }

    if (
        ok &&
        !limit_reached &&
        !ensure_pi_capacity(
            pi,
            implicant_words,
            (size_t)pi->foundPI + collector.count
        )
    ) {
        ok = false;
    }

    if (ok && !limit_reached) {
        for (size_t cube = 0; cube < collector.count; ++cube) {
            size_t dst = (size_t)pi->foundPI++;
            memcpy(
                &pi->implicants_pos[dst * (size_t)implicant_words],
                &collector.fixed_bits[cube * (size_t)implicant_words],
                (size_t)implicant_words * sizeof(uint64_t)
            );
            memcpy(
                &pi->implicants_val[dst * (size_t)implicant_words],
                &collector.value_bits[cube * (size_t)implicant_words],
                (size_t)implicant_words * sizeof(uint64_t)
            );
            memcpy(
                &pi->pichart_pos[dst * (size_t)pi->pichart_words],
                &collector.coverage[cube * (size_t)pi->pichart_words],
                (size_t)pi->pichart_words * sizeof(uint64_t)
            );
            pi->covsum[dst] = collector.covsum[cube];
            pi->shared[dst] = 0;
        }
    }

    free(edges);
    free(selected);
    free(forbidden);
    free(selected_vertices);
    free(hit_count);
    free(critical_count);
    free(inc_offset);
    free(inc_cursor);
    free(cover);
    free(allowed);
    free(inc_edges);
    collector_destroy(&collector);
    if (!ok) return BOUNDED_MMCS_ERROR;
    if (limit_reached) return BOUNDED_MMCS_LIMIT_REACHED;
    return BOUNDED_MMCS_COMPLETE;
}

bool bounded_mmcs_prepare_binary_rows(
    const PIstorage *pi,
    int ninputs,
    WeightedMMCSPreparedRows *prepared
) {
    if (prepared) memset(prepared, 0, sizeof(*prepared));
    if (
        !pi ||
        !prepared ||
        ninputs <= 0 ||
        pi->ON_minterms < 0 ||
        pi->OFF_minterms < 0 ||
        (pi->ON_minterms > 0 && !pi->ON_set) ||
        (pi->OFF_minterms > 0 && !pi->OFF_set)
    ) {
        return false;
    }

    size_t on_cells = (size_t)pi->ON_minterms * (size_t)ninputs;
    size_t off_cells = (size_t)pi->OFF_minterms * (size_t)ninputs;
    for (size_t cell = 0; cell < on_cells; ++cell) {
        if (pi->ON_set[cell] < 1 || pi->ON_set[cell] > 2) return false;
    }
    for (size_t cell = 0; cell < off_cells; ++cell) {
        if (pi->OFF_set[cell] < 1 || pi->OFF_set[cell] > 2) return false;
    }

    int row_words = (ninputs + 63) / 64;
    int on_column_words = (pi->ON_minterms + 63) / 64;
    int off_column_words = (pi->OFF_minterms + 63) / 64;
    size_t on_words = (size_t)pi->ON_minterms * (size_t)row_words;
    size_t off_words = (size_t)pi->OFF_minterms * (size_t)row_words;
    size_t on_column_cells =
        (size_t)ninputs * (size_t)on_column_words;
    size_t off_column_cells =
        (size_t)ninputs * (size_t)off_column_words;
    size_t on_prefix_cells =
        (size_t)ninputs * (size_t)(on_column_words + 1);
    if (on_words > 0) {
        prepared->on_bits =
            (uint64_t *)calloc(on_words, sizeof(uint64_t));
    }
    if (off_words > 0) {
        prepared->off_bits =
            (uint64_t *)calloc(off_words, sizeof(uint64_t));
    }
    if (on_column_cells > 0) {
        prepared->on_columns =
            (uint64_t *)calloc(on_column_cells, sizeof(uint64_t));
    }
    if (off_column_cells > 0) {
        prepared->off_columns =
            (uint64_t *)calloc(off_column_cells, sizeof(uint64_t));
    }
    if (on_prefix_cells > 0) {
        prepared->on_column_prefix_ones =
            (uint32_t *)calloc(on_prefix_cells, sizeof(uint32_t));
    }
    prepared->off_one_counts =
        (uint32_t *)calloc((size_t)ninputs, sizeof(uint32_t));
    if (
        (on_words > 0 && !prepared->on_bits) ||
        (off_words > 0 && !prepared->off_bits) ||
        (on_column_cells > 0 && !prepared->on_columns) ||
        (off_column_cells > 0 && !prepared->off_columns) ||
        (on_prefix_cells > 0 && !prepared->on_column_prefix_ones) ||
        !prepared->off_one_counts
    ) {
        bounded_mmcs_prepared_rows_destroy(prepared);
        return false;
    }

    prepared->ninputs = ninputs;
    prepared->row_words = row_words;
    prepared->on_rows = pi->ON_minterms;
    prepared->off_rows = pi->OFF_minterms;
    prepared->on_column_words = on_column_words;
    prepared->off_column_words = off_column_words;
    for (int row = 0; row < pi->ON_minterms; ++row) {
        for (int input = 0; input < ninputs; ++input) {
            if (
                pi->ON_set[(size_t)row * (size_t)ninputs + input] == 2
            ) {
                prepared->on_bits[
                    (size_t)row * (size_t)row_words +
                    (size_t)(input / 64)
                ] |= UINT64_C(1) << (input % 64);
                prepared->on_columns[
                    (size_t)input * (size_t)on_column_words +
                    (size_t)(row / 64)
                ] |= UINT64_C(1) << (row % 64);
            }
        }
    }
    for (int row = 0; row < pi->OFF_minterms; ++row) {
        for (int input = 0; input < ninputs; ++input) {
            if (
                pi->OFF_set[(size_t)row * (size_t)ninputs + input] == 2
            ) {
                prepared->off_bits[
                    (size_t)row * (size_t)row_words +
                    (size_t)(input / 64)
                ] |= UINT64_C(1) << (input % 64);
                prepared->off_columns[
                    (size_t)input * (size_t)off_column_words +
                    (size_t)(row / 64)
                ] |= UINT64_C(1) << (row % 64);
            }
        }
    }
    for (int input = 0; input < ninputs; ++input) {
        uint32_t *prefix = &prepared->on_column_prefix_ones[
            (size_t)input * (size_t)(on_column_words + 1)
        ];
        if (on_column_words > 0) {
            const uint64_t *on_column = &prepared->on_columns[
                (size_t)input * (size_t)on_column_words
            ];
            for (int word = 0; word < on_column_words; ++word) {
                prefix[word + 1] = prefix[word] +
                    (uint32_t)__builtin_popcountll(on_column[word]);
            }
        }

        uint32_t off_ones = 0;
        if (off_column_words > 0) {
            const uint64_t *off_column = &prepared->off_columns[
                (size_t)input * (size_t)off_column_words
            ];
            for (int word = 0; word < off_column_words; ++word) {
                off_ones +=
                    (uint32_t)__builtin_popcountll(off_column[word]);
            }
        }
        prepared->off_one_counts[input] = off_ones;
    }
    return true;
}

void bounded_mmcs_prepared_rows_destroy(
    WeightedMMCSPreparedRows *prepared
) {
    if (!prepared) return;
    free(prepared->on_bits);
    free(prepared->off_bits);
    free(prepared->on_columns);
    free(prepared->off_columns);
    free(prepared->on_column_prefix_ones);
    free(prepared->off_one_counts);
    memset(prepared, 0, sizeof(*prepared));
}

static bool ensure_incidence_capacity(
    WeightedPriceWorker *worker,
    size_t entries
) {
    if (entries <= worker->inc_edges_capacity) return true;
    if (entries > SIZE_MAX / sizeof(int)) return false;
    size_t capacity = worker->inc_edges_capacity > 0
        ? worker->inc_edges_capacity
        : 256u;
    while (capacity < entries) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = entries;
            break;
        }
        capacity *= 2u;
    }
    int *grown = (int *)realloc(
        worker->inc_edges,
        capacity * sizeof(int)
    );
    if (!grown) return false;
    worker->inc_edges = grown;
    worker->inc_edges_capacity = capacity;
    return true;
}

static void weighted_price_worker_destroy(
    WeightedPriceWorker *worker
) {
    if (!worker) return;
    free(worker->edges);
    free(worker->selected);
    free(worker->forbidden);
    free(worker->selected_vertices);
    free(worker->hit_count);
    free(worker->critical_count);
    free(worker->inc_offset);
    free(worker->inc_cursor);
    free(worker->cover);
    free(worker->allowed);
    free(worker->inactive_vertices);
    free(worker->inc_edges);
    free(worker->best_vertices);
    free(worker->local_batch.weights);
    free(worker->local_batch.anchors);
    free(worker->local_batch.support_sizes);
    free(worker->local_batch.vertices);
    free(worker->local_batch.coverage);
    memset(worker, 0, sizeof(*worker));
}

static bool weighted_price_worker_init(
    WeightedPriceWorker *worker,
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    int max_level,
    const double *row_weights,
    const double *root_upper_bounds,
    int local_batch_capacity,
    bool collect_timing,
    bool apply_root_gate,
    bool use_materialized_backend,
    bool cache_edge_words,
    bool disable_cover_bound
) {
    memset(worker, 0, sizeof(*worker));
    worker->pi = pi;
    worker->ninputs = prepared->ninputs;
    worker->max_level = max_level;
    worker->edge_words = prepared->row_words;
    worker->max_edges = pi->OFF_minterms + pi->ON_minterms - 1;
    worker->row_weights = row_weights;
    worker->root_upper_bounds = root_upper_bounds;
    worker->prepared = prepared;
    worker->collect_timing = collect_timing;
    worker->apply_root_gate = apply_root_gate;
    worker->use_materialized_backend = use_materialized_backend;
    worker->cache_edge_words = cache_edge_words && max_level > 1;
    worker->disable_cover_bound = disable_cover_bound;
    worker->status = WEIGHTED_MMCS_PRICE_OK;
    worker->first_incomplete_anchor = -1;
    int ninputs = prepared->ninputs;

    if (use_materialized_backend || worker->cache_edge_words) {
        worker->edges = (uint64_t *)calloc(
            (size_t)worker->max_edges * (size_t)worker->edge_words,
            sizeof(uint64_t)
        );
    }
    worker->selected = (unsigned char *)calloc(
        (size_t)ninputs,
        sizeof(unsigned char)
    );
    worker->forbidden = (unsigned char *)calloc(
        (size_t)ninputs,
        sizeof(unsigned char)
    );
    worker->selected_vertices =
        (int *)calloc((size_t)ninputs, sizeof(int));
    worker->hit_count =
        (int *)calloc((size_t)worker->max_edges, sizeof(int));
    worker->critical_count =
        (int *)calloc((size_t)ninputs, sizeof(int));
    if (use_materialized_backend) {
        worker->inc_offset =
            (int *)calloc((size_t)ninputs + 1, sizeof(int));
        worker->inc_cursor =
            (int *)calloc((size_t)ninputs + 1, sizeof(int));
    }
    worker->cover = (int *)calloc((size_t)ninputs, sizeof(int));
    worker->allowed =
        (int *)calloc((size_t)worker->max_edges, sizeof(int));
    worker->inactive_vertices = (uint64_t *)calloc(
        (size_t)worker->edge_words,
        sizeof(uint64_t)
    );
    worker->best_vertices =
        (int *)calloc((size_t)ninputs, sizeof(int));
    if (local_batch_capacity > 0) {
        WeightedMMCSPriceBatch *local = &worker->local_batch;
        local->capacity = local_batch_capacity;
        local->weights = (double *)malloc(
            (size_t)local_batch_capacity * sizeof(double)
        );
        local->anchors = (int *)malloc(
            (size_t)local_batch_capacity * sizeof(int)
        );
        local->support_sizes = (int *)malloc(
            (size_t)local_batch_capacity * sizeof(int)
        );
        local->vertices = (int *)malloc(
            (size_t)local_batch_capacity *
            (size_t)ninputs *
            sizeof(int)
        );
        local->coverage = (uint64_t *)malloc(
            (size_t)local_batch_capacity *
            (size_t)pi->pichart_words *
            sizeof(uint64_t)
        );
    }
    if (
        (
            (use_materialized_backend || worker->cache_edge_words) &&
            !worker->edges
        ) ||
        !worker->selected ||
        !worker->forbidden ||
        !worker->selected_vertices ||
        !worker->hit_count ||
        !worker->critical_count ||
        (use_materialized_backend && !worker->inc_offset) ||
        (use_materialized_backend && !worker->inc_cursor) ||
        !worker->cover ||
        !worker->allowed ||
        !worker->inactive_vertices ||
        !worker->best_vertices ||
        (
            local_batch_capacity > 0 &&
            (
                !worker->local_batch.weights ||
                !worker->local_batch.anchors ||
                !worker->local_batch.support_sizes ||
                !worker->local_batch.vertices ||
                !worker->local_batch.coverage
            )
        )
    ) {
        weighted_price_worker_destroy(worker);
        return false;
    }

    worker->price = (WeightedPriceContext){
        .pi = pi,
        .ninputs = ninputs,
        .row_weights = row_weights,
        .best_vertices = worker->best_vertices,
        .best_weight = -INFINITY,
        .best_anchor = -1,
        .best_support_size = 0,
        .weight_prunes = 0,
        .local_batch = local_batch_capacity > 0
            ? &worker->local_batch
            : NULL,
        .telemetry = collect_timing ? &worker->telemetry : NULL,
        .collect_timing = collect_timing
    };
    return true;
}

static uint32_t prepared_on_ones_before(
    const WeightedMMCSPreparedRows *prepared,
    int vertex,
    int row_limit
) {
    int full_words = row_limit / 64;
    int remaining = row_limit % 64;
    const uint32_t *prefix = &prepared->on_column_prefix_ones[
        (size_t)vertex * (size_t)(prepared->on_column_words + 1)
    ];
    uint32_t ones = prefix[full_words];
    if (remaining > 0) {
        const uint64_t *column = &prepared->on_columns[
            (size_t)vertex * (size_t)prepared->on_column_words
        ];
        ones += (uint32_t)__builtin_popcountll(
            column[full_words] & low_bits_mask(remaining)
        );
    }
    return ones;
}

static uint64_t prepared_anchor_incidence_count(
    const WeightedMMCSPreparedRows *prepared,
    const uint64_t *anchor_bits,
    int anchor_index
) {
    uint64_t count = 0;
    for (int vertex = 0; vertex < prepared->ninputs; ++vertex) {
        bool anchor_is_one = (
            anchor_bits[vertex / 64] &
            (UINT64_C(1) << (vertex % 64))
        ) != 0;
        uint32_t off_ones = prepared->off_one_counts[vertex];
        uint32_t on_ones = prepared_on_ones_before(
            prepared,
            vertex,
            anchor_index
        );
        count += anchor_is_one
            ? (uint64_t)(prepared->off_rows - (int)off_ones) +
                (uint64_t)(anchor_index - (int)on_ones)
            : (uint64_t)off_ones + (uint64_t)on_ones;
    }
    return count;
}

typedef struct {
    int index;
    int edge_count;
    const int *values;
    const uint64_t *packed_values;
} WeightedAnchorPlan;

static bool weighted_price_anchor_precheck(
    WeightedPriceWorker *worker,
    int on,
    WeightedAnchorPlan *anchor
) {
    const PIstorage *pi = worker->pi;
    *anchor = (WeightedAnchorPlan){
        .index = on,
        .edge_count = pi->OFF_minterms + on,
        .values = &pi->ON_set[
            (size_t)on * (size_t)worker->ninputs
        ],
        .packed_values = &worker->prepared->on_bits[
            (size_t)on * (size_t)worker->edge_words
        ]
    };
    WeightedMMCSPriceTelemetry *telemetry = &worker->telemetry;
    telemetry->anchors_presented++;
    if (
        worker->root_upper_bounds[on] >
        1.0 + WEIGHTED_MMCS_REDUCED_COST_TOL
    ) {
        return true;
    }

    telemetry->prospective_gated_anchors++;
    telemetry->prospective_gated_edges +=
        (uint64_t)anchor->edge_count;
    telemetry->prospective_gated_incidences +=
        prepared_anchor_incidence_count(
            worker->prepared,
            anchor->packed_values,
            on
        );
    return !worker->apply_root_gate;
}

static bool weighted_price_anchor_prepare(
    WeightedPriceWorker *worker,
    const WeightedAnchorPlan *anchor
) {
    const PIstorage *pi = worker->pi;
    int ninputs = worker->ninputs;
    int edge_words = worker->edge_words;
    int on = anchor->index;
    int edge_count = anchor->edge_count;
    const uint64_t *packed_anchor = anchor->packed_values;
    WeightedMMCSPriceTelemetry *telemetry = &worker->telemetry;

    telemetry->anchors_constructed++;
    if (worker->edges) {
        telemetry->edges_constructed += (uint64_t)edge_count;
    }
    telemetry->xor_words +=
        (uint64_t)edge_count * (uint64_t)edge_words;

    uint64_t edge_started = worker->collect_timing
        ? monotonic_nanoseconds()
        : 0;
    for (int edge = 0; edge < edge_count; ++edge) {
        const uint64_t *other = edge < pi->OFF_minterms
            ? &worker->prepared->off_bits[
                (size_t)edge * (size_t)edge_words
            ]
            : &worker->prepared->on_bits[
                (size_t)(edge - pi->OFF_minterms) *
                (size_t)edge_words
            ];
        uint64_t combined = 0;
        int edge_size = 0;
        for (int word = 0; word < edge_words; ++word) {
            uint64_t difference = packed_anchor[word] ^ other[word];
            if (worker->edges) {
                worker->edges[
                    (size_t)edge * (size_t)edge_words + (size_t)word
                ] = difference;
            }
            combined |= difference;
            edge_size += __builtin_popcountll(difference);
        }
        if (combined == 0) {
            worker->status = WEIGHTED_MMCS_PRICE_INVALID;
            return false;
        }
        worker->allowed[edge] = edge_size;
    }
    if (edge_started > 0) {
        telemetry->edge_build_nanoseconds +=
            monotonic_nanoseconds() - edge_started;
    }

    uint64_t count_started = worker->collect_timing
        ? monotonic_nanoseconds()
        : 0;
    memset(worker->selected, 0, (size_t)ninputs * sizeof(unsigned char));
    memset(worker->forbidden, 0, (size_t)ninputs * sizeof(unsigned char));
    memset(worker->selected_vertices, 0, (size_t)ninputs * sizeof(int));
    memset(worker->hit_count, 0, (size_t)edge_count * sizeof(int));
    memset(worker->critical_count, 0, (size_t)ninputs * sizeof(int));
    memset(
        worker->inactive_vertices,
        0,
        (size_t)edge_words * sizeof(uint64_t)
    );
    if (worker->use_materialized_backend) {
        memset(
            worker->inc_offset,
            0,
            (size_t)(ninputs + 1) * sizeof(int)
        );

        size_t incidence_entries = 0;
        for (int edge = 0; edge < edge_count; ++edge) {
            const uint64_t *bits = &worker->edges[
                (size_t)edge * (size_t)edge_words
            ];
            int size = 0;
            for (int word = 0; word < edge_words; ++word) {
                uint64_t values = bits[word];
                while (values) {
                    int vertex = word * 64 + __builtin_ctzll(values);
                    values &= values - 1;
                    worker->inc_offset[vertex + 1]++;
                    size++;
                }
            }
            incidence_entries += (size_t)size;
        }
        telemetry->incidences_counted += (uint64_t)incidence_entries;
        if (count_started > 0) {
            telemetry->incidence_count_nanoseconds +=
                monotonic_nanoseconds() - count_started;
        }
        if (!ensure_incidence_capacity(worker, incidence_entries)) {
            worker->status = WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
            return false;
        }
        if (incidence_entries > worker->peak_incidence_entries) {
            worker->peak_incidence_entries = incidence_entries;
        }
        uint64_t fill_started = worker->collect_timing
            ? monotonic_nanoseconds()
            : 0;
        for (int vertex = 0; vertex < ninputs; ++vertex) {
            worker->inc_offset[vertex + 1] +=
                worker->inc_offset[vertex];
            worker->cover[vertex] = 0;
        }
        memcpy(
            worker->inc_cursor,
            worker->inc_offset,
            (size_t)ninputs * sizeof(int)
        );
        for (int edge = 0; edge < edge_count; ++edge) {
            const uint64_t *bits = &worker->edges[
                (size_t)edge * (size_t)edge_words
            ];
            for (int word = 0; word < edge_words; ++word) {
                uint64_t values = bits[word];
                while (values) {
                    int vertex = word * 64 + __builtin_ctzll(values);
                    values &= values - 1;
                    worker->inc_edges[
                        worker->inc_cursor[vertex]++
                    ] = edge;
                    if (!worker->disable_cover_bound) {
                        worker->cover[vertex]++;
                    }
                }
            }
        }
        telemetry->incidences_written += (uint64_t)incidence_entries;
        if (fill_started > 0) {
            telemetry->incidence_fill_nanoseconds +=
                monotonic_nanoseconds() - fill_started;
        }
    } else {
        if (!worker->disable_cover_bound) {
            for (int vertex = 0; vertex < ninputs; ++vertex) {
                bool anchor_is_one = (
                    packed_anchor[vertex / 64] &
                    (UINT64_C(1) << (vertex % 64))
                ) != 0;
                uint32_t off_ones =
                    worker->prepared->off_one_counts[vertex];
                uint32_t on_ones = prepared_on_ones_before(
                    worker->prepared,
                    vertex,
                    on
                );
                worker->cover[vertex] = anchor_is_one
                    ? pi->OFF_minterms - (int)off_ones +
                        on - (int)on_ones
                    : (int)off_ones + (int)on_ones;
            }
        }
        if (count_started > 0) {
            telemetry->incidence_count_nanoseconds +=
                monotonic_nanoseconds() - count_started;
        }
    }
    return true;
}

static void weighted_price_anchor_search(
    WeightedPriceWorker *worker,
    const WeightedAnchorPlan *anchor,
    uint64_t deadline_nanoseconds
) {
    const PIstorage *pi = worker->pi;
    WeightedMMCSPriceTelemetry *telemetry = &worker->telemetry;
    worker->price.anchor = anchor->values;
    worker->price.anchor_index = anchor->index;
    MMCSState state = {
        .ninputs = worker->ninputs,
        .edge_count = anchor->edge_count,
        .critical_edge_count = pi->OFF_minterms,
        .edge_words = worker->edge_words,
        .target_size = worker->max_level,
        .exact_size = false,
        .edges = worker->edges,
        .prepared = worker->prepared,
        .anchor_bits = anchor->packed_values,
        .anchor_index = anchor->index,
        .transposed_incidence = !worker->use_materialized_backend,
        .selected = worker->selected,
        .forbidden = worker->forbidden,
        .selected_vertices = worker->selected_vertices,
        .selected_count = 0,
        .hit_count = worker->hit_count,
        .critical_count = worker->critical_count,
        .uncovered_count = anchor->edge_count,
        .inc_offset = worker->inc_offset,
        .inc_edges = worker->inc_edges,
        .cover = worker->cover,
        .allowed = worker->allowed,
        .inactive_vertices = worker->inactive_vertices,
        .emit = collect_weighted_price,
        .emit_data = &worker->price,
        .prune = weighted_price_prune,
        .branch_priority = weighted_branch_priority,
        .search_data = &worker->price,
        .stats = &worker->search_stats,
        .telemetry = worker->collect_timing ? telemetry : NULL,
        .node_limit = 0,
        .deadline_nanoseconds = deadline_nanoseconds,
        .disable_cover_bound = worker->disable_cover_bound,
        .limit_reached = false,
        .failed = false
    };
    uint64_t search_started = worker->collect_timing
        ? monotonic_nanoseconds()
        : 0;
    mmcs_search(&state);
    if (search_started > 0) {
        telemetry->search_nanoseconds +=
            monotonic_nanoseconds() - search_started;
    }
    if (state.failed) {
        worker->status = WEIGHTED_MMCS_PRICE_INVALID;
    } else if (state.limit_reached) {
        worker->status = WEIGHTED_MMCS_PRICE_LIMIT_REACHED;
        worker->first_incomplete_anchor = anchor->index;
    } else {
        worker->anchors_completed++;
    }
}

static void weighted_price_search_anchor(
    WeightedPriceWorker *worker,
    int on,
    uint64_t deadline_nanoseconds
) {
    WeightedAnchorPlan anchor;
    if (!weighted_price_anchor_precheck(worker, on, &anchor)) {
        worker->anchors_completed++;
        return;
    }
    if (!weighted_price_anchor_prepare(worker, &anchor)) return;
    weighted_price_anchor_search(worker, &anchor, deadline_nanoseconds);
}

static void weighted_price_worker_range(
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
    WeightedParallelContext *parallel =
        (WeightedParallelContext *)data;
    WeightedPriceWorker *worker = &parallel->workers[worker_id];
    while (worker->status == WEIGHTED_MMCS_PRICE_OK) {
        if (atomic_load_explicit(
            &parallel->limit_reached,
            memory_order_acquire
        )) {
            break;
        }
        if (
            parallel->deadline_nanoseconds > 0 &&
            monotonic_nanoseconds() >= parallel->deadline_nanoseconds &&
            atomic_load_explicit(
                &parallel->next_anchor,
                memory_order_relaxed
            ) < parallel->anchor_count
        ) {
            atomic_store_explicit(
                &parallel->limit_reached,
                true,
                memory_order_release
            );
            break;
        }
        int on = atomic_fetch_add_explicit(
            &parallel->next_anchor,
            1,
            memory_order_relaxed
        );
        if (on >= parallel->anchor_count) break;
        weighted_price_search_anchor(
            worker,
            on,
            parallel->deadline_nanoseconds
        );
        if (worker->status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED) {
            atomic_store_explicit(
                &parallel->limit_reached,
                true,
                memory_order_release
            );
            break;
        }
        if (
            parallel->deadline_nanoseconds > 0 &&
            monotonic_nanoseconds() >= parallel->deadline_nanoseconds &&
            atomic_load_explicit(
                &parallel->next_anchor,
                memory_order_relaxed
            ) < parallel->anchor_count
        ) {
            atomic_store_explicit(
                &parallel->limit_reached,
                true,
                memory_order_release
            );
        }
    }
}

static uint64_t weighted_worker_bytes_estimate(
    const PIstorage *pi,
    int ninputs,
    bool use_materialized_backend,
    bool cache_edge_words
) {
    uint64_t edges =
        (uint64_t)pi->OFF_minterms + (uint64_t)pi->ON_minterms - 1u;
    uint64_t edge_words = ((uint64_t)ninputs + 63u) / 64u;
    uint64_t bytes = 0;

    if (use_materialized_backend || cache_edge_words) {
        if (edges > UINT64_MAX / edge_words / sizeof(uint64_t)) {
            return UINT64_MAX;
        }
        bytes += edges * edge_words * sizeof(uint64_t);
    }

    if (use_materialized_backend) {
        if (edges > UINT64_MAX / (uint64_t)ninputs / sizeof(int)) {
            return UINT64_MAX;
        }
        uint64_t worst_incidence =
            edges * (uint64_t)ninputs * sizeof(int);
        if (bytes > UINT64_MAX - worst_incidence) return UINT64_MAX;
        bytes += worst_incidence;
    }

    uint64_t edge_arrays = edges * 2u * sizeof(int);
    uint64_t input_arrays =
        (
            (uint64_t)ninputs *
            (use_materialized_backend ? 7u : 4u) +
            (use_materialized_backend ? 2u : 0u)
        ) * sizeof(int) +
        (uint64_t)ninputs * 2u * sizeof(unsigned char) +
        edge_words * sizeof(uint64_t);
    if (
        bytes > UINT64_MAX - edge_arrays ||
        bytes + edge_arrays > UINT64_MAX - input_arrays
    ) {
        return UINT64_MAX;
    }
    return bytes + edge_arrays + input_arrays;
}

typedef struct {
    double mass;
    int input;
} WeightedAnchorLiteral;

static int compare_anchor_literal_mass(const void *lhs, const void *rhs) {
    const WeightedAnchorLiteral *a = lhs;
    const WeightedAnchorLiteral *b = rhs;
    if (a->mass != b->mass) return a->mass > b->mass ? -1 : 1;
    return (a->input > b->input) - (a->input < b->input);
}

static double interrupted_anchor_upper_bound(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    const double *weights,
    int anchor,
    double suffix,
    WeightedAnchorLiteral *literals,
    uint64_t *remaining_off
) {
    int ninputs = prepared->ninputs;
    const int *point = &pi->ON_set[(size_t)anchor * (size_t)ninputs];
    for (int input = 0; input < ninputs; ++input) {
        double mass = 0.0;
        for (int row = anchor; row < pi->ON_minterms; ++row) {
            if (
                weights[row] > 0.0 &&
                pi->ON_set[(size_t)row * (size_t)ninputs + (size_t)input]
                    == point[input]
            ) {
                mass = nextafter(mass + weights[row], INFINITY);
            }
        }
        literals[input] = (WeightedAnchorLiteral){ .mass = mass, .input = input };
    }
    qsort(literals, (size_t)ninputs, sizeof(*literals), compare_anchor_literal_mass);

    int words = prepared->off_column_words;
    memset(remaining_off, 0xff, (size_t)words * sizeof(*remaining_off));
    int tail = pi->OFF_minterms % 64;
    if (tail) remaining_off[words - 1] = (UINT64_C(1) << tail) - 1;
    for (int index = 0; index < ninputs; ++index) {
        int input = literals[index].input;
        const uint64_t *column = &prepared->off_columns[(size_t)input * (size_t)words];
        uint64_t any_remaining = 0;
        for (int word = 0; word < words; ++word) {
            remaining_off[word] &= point[input] == 2 ? column[word] : ~column[word];
            any_remaining |= remaining_off[word];
        }
        // The last OFF point blocked by this prefix supplies the smallest
        // maximum literal mass among all OFF difference edges.
        if (!any_remaining) return fmin(suffix, literals[index].mass);
    }
    return suffix;
}

WeightedMMCSPriceStatus bounded_mmcs_price_output_batch_prepared(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceBatch *batch,
    const WeightedMMCSPriceOptions *options,
    WeightedMMCSPriceResult *result
) {
    if (result) {
        memset(result, 0, sizeof(*result));
        result->best_weight = -INFINITY;
        result->max_weight_upper_bound = INFINITY;
        result->best_anchor = -1;
    }
    if (
        !pi ||
        !prepared ||
        prepared->ninputs <= 0 ||
        max_level <= 0 ||
        max_level > prepared->ninputs ||
        !row_weights ||
        !best_vertices ||
        !result ||
        (options && options->anchor_limit < 0) ||
        pi->ON_minterms <= 0 ||
        pi->OFF_minterms <= 0 ||
        !pi->ON_set ||
        !pi->OFF_set
    ) {
        return WEIGHTED_MMCS_PRICE_INVALID;
    }
    int ninputs = prepared->ninputs;
    if (
        prepared->row_words != (ninputs + 63) / 64 ||
        prepared->on_rows != pi->ON_minterms ||
        prepared->off_rows != pi->OFF_minterms ||
        prepared->on_column_words != (pi->ON_minterms + 63) / 64 ||
        prepared->off_column_words != (pi->OFF_minterms + 63) / 64 ||
        !prepared->on_bits ||
        !prepared->off_bits ||
        !prepared->on_columns ||
        !prepared->off_columns ||
        !prepared->on_column_prefix_ones ||
        !prepared->off_one_counts
    ) {
        return WEIGHTED_MMCS_PRICE_INVALID;
    }
    if (batch) {
        batch->count = 0;
        if (
            batch->capacity <= 0 ||
            !batch->weights ||
            !batch->anchors ||
            !batch->support_sizes ||
            !batch->vertices ||
            !batch->coverage ||
            pi->pichart_words <= 0 ||
            pi->cov_bits <= 0
        ) {
            return WEIGHTED_MMCS_PRICE_INVALID;
        }
    }

    for (int row = 0; row < pi->ON_minterms; ++row) {
        if (!isfinite(row_weights[row])) {
            return WEIGHTED_MMCS_PRICE_INVALID;
        }
    }

    int anchor_count = pi->ON_minterms;
    if (
        options &&
        options->anchor_limit > 0 &&
        options->anchor_limit < anchor_count
    ) {
        anchor_count = options->anchor_limit;
    }
    result->anchors_total = pi->ON_minterms;
    int nthreads = options && options->requested_threads > 0
        ? options->requested_threads
        : ccubes_thread_count();
    nthreads = ccubes_effective_thread_count(nthreads);
    if (nthreads > anchor_count) nthreads = anchor_count;
    if (nthreads < 1) nthreads = 1;

    bool apply_root_gate = options && options->enable_root_gate;
    bool use_materialized_backend =
        options && options->use_materialized_backend;
    bool cache_edge_words = options && options->cache_edge_words;
    bool disable_cover_bound =
        options && options->disable_cover_bound;
    uint64_t estimated_worker_bytes = weighted_worker_bytes_estimate(
        pi,
        ninputs,
        use_materialized_backend,
        cache_edge_words && max_level > 1
    );
    uint64_t worker_budget = options
        ? options->worker_memory_budget_bytes
        : 0;
    if (worker_budget > 0) {
        if (estimated_worker_bytes == 0 || estimated_worker_bytes == UINT64_MAX) {
            return WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
        }
        uint64_t memory_threads = worker_budget / estimated_worker_bytes;
        if (memory_threads < 1u) {
            return WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
        }
        if ((uint64_t)nthreads > memory_threads) {
            nthreads = memory_threads > (uint64_t)INT_MAX
                ? INT_MAX
                : (int)memory_threads;
        }
    }
    result->effective_threads = nthreads;
    result->estimated_bytes_per_worker = estimated_worker_bytes;
    bool collect_timing = options && options->collect_timing;
    uint64_t pricing_call_started = collect_timing
        ? monotonic_nanoseconds()
        : 0;

    double *root_upper_bounds = (double *)malloc(
        (size_t)pi->ON_minterms * sizeof(double)
    );
    if (!root_upper_bounds) {
        return WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
    }
    double suffix = 0.0;
    for (int row = pi->ON_minterms - 1; row >= 0; --row) {
        if (row_weights[row] > 0.0) suffix += row_weights[row];
        root_upper_bounds[row] = suffix;
    }

    WeightedPriceWorker *workers = (WeightedPriceWorker *)calloc(
        (size_t)nthreads,
        sizeof(WeightedPriceWorker)
    );
    if (!workers) {
        free(root_upper_bounds);
        return WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
    }

    int initialized = 0;
    for (; initialized < nthreads; ++initialized) {
        int local_batch_capacity = batch ? batch->capacity : 0;
        if (!weighted_price_worker_init(
            &workers[initialized],
            pi,
            prepared,
            max_level,
            row_weights,
            root_upper_bounds,
            local_batch_capacity,
            collect_timing,
            apply_root_gate,
            use_materialized_backend,
            cache_edge_words,
            disable_cover_bound
        )) {
            break;
        }
    }
    if (initialized != nthreads) {
        for (int i = 0; i < initialized; ++i) {
            weighted_price_worker_destroy(&workers[i]);
        }
        free(workers);
        free(root_upper_bounds);
        return WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
    }

    WeightedParallelContext parallel = {
        .workers = workers,
        .anchor_count = anchor_count,
        .deadline_nanoseconds = options
            ? options->deadline_nanoseconds
            : 0
    };
    atomic_init(&parallel.next_anchor, 0);
    atomic_init(&parallel.limit_reached, false);
    WeightedMMCSPriceStatus status = WEIGHTED_MMCS_PRICE_OK;
    if (!ccubes_parallel_for(
        0,
        (uint64_t)nthreads,
        nthreads,
        false,
        weighted_price_worker_range,
        &parallel
    )) {
        status = WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
    }
    if (
        status == WEIGHTED_MMCS_PRICE_OK &&
        atomic_load_explicit(
            &parallel.limit_reached,
            memory_order_acquire
        )
    ) {
        status = WEIGHTED_MMCS_PRICE_LIMIT_REACHED;
        result->limit_reached = true;
    }
    result->anchor_sweep_complete =
        status == WEIGHTED_MMCS_PRICE_OK &&
        anchor_count == pi->ON_minterms;

    int assigned_anchors = atomic_load_explicit(
        &parallel.next_anchor,
        memory_order_acquire
    );
    if (assigned_anchors < 0) assigned_anchors = 0;
    if (assigned_anchors > anchor_count) assigned_anchors = anchor_count;
    int earliest_unfinished = assigned_anchors;

    int best_worker = -1;
    for (int worker = 0; worker < nthreads; ++worker) {
        if (
            workers[worker].status != WEIGHTED_MMCS_PRICE_OK &&
            (
                status == WEIGHTED_MMCS_PRICE_OK ||
                status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED
            )
        ) {
            status = workers[worker].status;
        }
        result->search_nodes +=
            workers[worker].search_stats.search_nodes;
        result->anchors_completed += workers[worker].anchors_completed;
        if (
            workers[worker].first_incomplete_anchor >= 0 &&
            workers[worker].first_incomplete_anchor < earliest_unfinished
        ) {
            earliest_unfinished =
                workers[worker].first_incomplete_anchor;
        }
        result->completed_transversals +=
            workers[worker].search_stats.completed_transversals;
        result->weight_prunes += workers[worker].price.weight_prunes;
        add_price_telemetry(
            &result->telemetry,
            &workers[worker].telemetry
        );
        if (
            workers[worker].peak_incidence_entries >
            result->peak_incidence_entries
        ) {
            result->peak_incidence_entries =
                workers[worker].peak_incidence_entries;
        }

        if (workers[worker].price.best_anchor < 0) continue;
        if (
            best_worker < 0 ||
            workers[worker].price.best_weight >
                workers[best_worker].price.best_weight ||
            (
                workers[worker].price.best_weight ==
                    workers[best_worker].price.best_weight &&
                weighted_geometry_precedes(
                    workers[worker].price.best_anchor,
                    workers[worker].price.best_support_size,
                    workers[worker].best_vertices,
                    workers[best_worker].price.best_anchor,
                    workers[best_worker].price.best_support_size,
                    workers[best_worker].best_vertices,
                    ninputs
                )
            )
        ) {
            best_worker = worker;
        }
    }

    if (
        (
            status == WEIGHTED_MMCS_PRICE_OK ||
            status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED
        ) &&
        best_worker >= 0
    ) {
        result->best_weight = workers[best_worker].price.best_weight;
        result->best_anchor = workers[best_worker].price.best_anchor;
        result->best_support_size =
            workers[best_worker].price.best_support_size;
        memcpy(
            best_vertices,
            workers[best_worker].best_vertices,
            (size_t)result->best_support_size * sizeof(int)
        );
    }
    if (
        status == WEIGHTED_MMCS_PRICE_OK ||
        status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED
    ) {
        double upper = best_worker >= 0
            ? workers[best_worker].price.best_weight
            : 0.0;
        /*
         * Workers claim anchors monotonically. After joining, every claimed
         * anchor either finished or is recorded as interrupted by its worker.
         * Use the original suffix for unassigned anchors and tighten each
         * interrupted anchor separately. Root-gated pricing retains its
         * existing fallback.
         */
        if (
            status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED &&
            !apply_root_gate &&
            earliest_unfinished < assigned_anchors
        ) {
            // Never-assigned anchors remain covered by their original suffix.
            if (assigned_anchors < pi->ON_minterms) {
                upper = fmax(upper, root_upper_bounds[assigned_anchors]);
            }
            WeightedAnchorLiteral *literals = malloc((size_t)ninputs * sizeof(*literals));
            uint64_t *remaining_off = malloc(
                (size_t)prepared->off_column_words * sizeof(*remaining_off)
            );
            uint64_t work_left = WEIGHTED_MMCS_TIMEOUT_BOUND_WORK;
            for (int worker = 0; worker < nthreads; ++worker) {
                int anchor = workers[worker].first_incomplete_anchor;
                if (anchor < 0 || root_upper_bounds[anchor] <= upper) continue;
                double unfinished = root_upper_bounds[anchor];
                uint64_t work = (uint64_t)ninputs * (
                    (uint64_t)(pi->ON_minterms - anchor) +
                    (uint64_t)prepared->off_column_words
                );
                if (literals && remaining_off && work <= work_left) {
                    work_left -= work;
                    unfinished = interrupted_anchor_upper_bound(
                        pi, prepared, row_weights, anchor, unfinished, literals, remaining_off
                    );
                }
                upper = fmax(upper, unfinished);
            }
            free(remaining_off);
            free(literals);
        } else if (earliest_unfinished < pi->ON_minterms) {
            double unfinished = root_upper_bounds[earliest_unfinished];
            if (unfinished > upper) upper = unfinished;
        }
        result->max_weight_upper_bound = upper;
    }
    if (
        (
            status == WEIGHTED_MMCS_PRICE_OK ||
            status == WEIGHTED_MMCS_PRICE_LIMIT_REACHED
        ) &&
        batch
    ) {
        uint64_t merge_started = collect_timing
            ? monotonic_nanoseconds()
            : 0;
        if (!weighted_merge_local_batches(
            workers,
            nthreads,
            batch,
            ninputs,
            pi->pichart_words
        )) {
            status = WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY;
        }
        if (merge_started > 0) {
            result->telemetry.merge_nanoseconds +=
                monotonic_nanoseconds() - merge_started;
        }
    }

    for (int worker = 0; worker < nthreads; ++worker) {
        weighted_price_worker_destroy(&workers[worker]);
    }
    free(workers);
    free(root_upper_bounds);
    if (pricing_call_started > 0) {
        result->telemetry.pricing_call_nanoseconds +=
            monotonic_nanoseconds() - pricing_call_started;
    }
    return status;
}

WeightedMMCSPriceStatus bounded_mmcs_price_output_batch(
    const PIstorage *pi,
    int ninputs,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceBatch *batch,
    WeightedMMCSPriceResult *result
) {
    WeightedMMCSPreparedRows prepared;
    if (!bounded_mmcs_prepare_binary_rows(pi, ninputs, &prepared)) {
        return WEIGHTED_MMCS_PRICE_UNSUPPORTED;
    }
    WeightedMMCSPriceStatus status =
        bounded_mmcs_price_output_batch_prepared(
            pi,
            &prepared,
            max_level,
            row_weights,
            best_vertices,
            batch,
            NULL,
            result
        );
    bounded_mmcs_prepared_rows_destroy(&prepared);
    return status;
}

WeightedMMCSPriceStatus bounded_mmcs_price_output(
    const PIstorage *pi,
    int ninputs,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceResult *result
) {
    return bounded_mmcs_price_output_batch(
        pi,
        ninputs,
        max_level,
        row_weights,
        best_vertices,
        NULL,
        result
    );
}

BoundedMMCSResult bounded_mmcs_generate_output_level_limited(
    PIstorage *pi,
    int ninputs,
    int level,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    uint64_t node_limit,
    BoundedMMCSStats *stats
) {
    return bounded_mmcs_generate_output_target_limited(
        pi,
        ninputs,
        level,
        true,
        word_index,
        bit_index,
        shifted_mask,
        implicant_words,
        node_limit,
        stats
    );
}

bool bounded_mmcs_generate_output_level(
    PIstorage *pi,
    int ninputs,
    int level,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    BoundedMMCSStats *stats
) {
    return bounded_mmcs_generate_output_level_limited(
        pi,
        ninputs,
        level,
        word_index,
        bit_index,
        shifted_mask,
        implicant_words,
        0,
        stats
    ) == BOUNDED_MMCS_COMPLETE;
}

BoundedMMCSResult bounded_mmcs_generate_output_all_limited(
    PIstorage *pi,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    uint64_t node_limit,
    BoundedMMCSStats *stats
) {
    return bounded_mmcs_generate_output_target_limited(
        pi,
        ninputs,
        ninputs,
        false,
        word_index,
        bit_index,
        shifted_mask,
        implicant_words,
        node_limit,
        stats
    );
}

bool bounded_mmcs_generate_output_all(
    PIstorage *pi,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    BoundedMMCSStats *stats
) {
    return bounded_mmcs_generate_output_all_limited(
        pi,
        ninputs,
        word_index,
        bit_index,
        shifted_mask,
        implicant_words,
        0,
        stats
    ) == BOUNDED_MMCS_COMPLETE;
}

typedef struct {
    int output;
    int record;
} SharedGeometryRecord;

bool bounded_mmcs_mark_level_sharing(
    PIstorage *PInfo,
    int noutputs,
    const int *level_start,
    int implicant_words,
    int *max_shared
) {
    if (
        !PInfo ||
        noutputs <= 0 ||
        !level_start ||
        implicant_words <= 0 ||
        !max_shared
    ) {
        return false;
    }

    size_t total = 0;
    for (int output = 0; output < noutputs; ++output) {
        if (
            level_start[output] < 0 ||
            level_start[output] > PInfo[output].foundPI
        ) {
            return false;
        }
        total += (size_t)(PInfo[output].foundPI - level_start[output]);
    }
    if (total == 0) return true;

    /*
     * Single output: sharing is the number of OTHER outputs holding the same
     * geometry, which is necessarily zero, and the generator already
     * deduplicates within an output.  Skip the hash table entirely -- it was
     * the largest single cost on single-output instances.
     */
    if (noutputs == 1) {
        for (
            int record = level_start[0];
            record < PInfo[0].foundPI;
            ++record
        ) {
            PInfo[0].shared[record] = 0;
        }
        return true;
    }

    size_t table_size = 16;
    while (table_size < total * 2u) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }

    int *slots = (int *)malloc(table_size * sizeof(int));
    int *counts = (int *)calloc(total, sizeof(int));
    SharedGeometryRecord *records = (SharedGeometryRecord *)calloc(
        total,
        sizeof(SharedGeometryRecord)
    );

    /*
     * Which unique geometry each scanned record resolved to.  Recording it in
     * the first pass lets the second pass be a direct lookup instead of
     * repeating the hash probe and the memcmp chain.
     */
    int *resolved = (int *)malloc(total * sizeof(int));
    if (!slots || !counts || !records || !resolved) {
        free(slots);
        free(counts);
        free(records);
        free(resolved);
        return false;
    }

    for (size_t i = 0; i < table_size; ++i) slots[i] = -1;

    size_t unique = 0;
    size_t scanned = 0;
    for (int output = 0; output < noutputs; ++output) {
        for (
            int record = level_start[output];
            record < PInfo[output].foundPI;
            ++record
        ) {
            const uint64_t *fixed_bits = &PInfo[output].implicants_pos[
                (size_t)record * (size_t)implicant_words
            ];
            const uint64_t *value_bits = &PInfo[output].implicants_val[
                (size_t)record * (size_t)implicant_words
            ];
            size_t pos = (size_t)(
                geometry_hash(
                    fixed_bits,
                    value_bits,
                    implicant_words
                ) & (uint64_t)(table_size - 1u)
            );

            while (slots[pos] >= 0) {
                int existing = slots[pos];
                int other_output = records[existing].output;
                int other_record = records[existing].record;
                if (same_geometry(
                    fixed_bits,
                    value_bits,
                    &PInfo[other_output].implicants_pos[
                        (size_t)other_record * (size_t)implicant_words
                    ],
                    &PInfo[other_output].implicants_val[
                        (size_t)other_record * (size_t)implicant_words
                    ],
                    implicant_words
                )) {
                    counts[existing]++;
                    resolved[scanned] = existing;
                    break;
                }
                pos = (pos + 1u) & (table_size - 1u);
            }

            if (slots[pos] < 0) {
                slots[pos] = (int)unique;
                records[unique].output = output;
                records[unique].record = record;
                counts[unique] = 1;
                resolved[scanned] = (int)unique;
                unique++;
            }
            scanned++;
        }
    }

    scanned = 0;
    for (int output = 0; output < noutputs; ++output) {
        for (
            int record = level_start[output];
            record < PInfo[output].foundPI;
            ++record
        ) {
            int shared = counts[resolved[scanned]] - 1;
            PInfo[output].shared[record] = shared;
            if (*max_shared < shared) *max_shared = shared;
            scanned++;
        }
    }

    free(slots);
    free(counts);
    free(records);
    free(resolved);
    return true;
}
