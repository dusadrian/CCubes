/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "bounded_mmcs.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int ninputs;
    int edge_count;
    int critical_edge_count;
    int edge_words;
    int target_size;
    const uint64_t *edges;

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

    bool (*emit)(const int *vertices, int count, void *data);
    void *emit_data;
    BoundedMMCSStats *stats;
    uint64_t node_limit;
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

static bool edge_contains(
    const MMCSState *state,
    int edge,
    int vertex
) {
    const uint64_t *bits = &state->edges[
        (size_t)edge * (size_t)state->edge_words
    ];
    return (bits[vertex / 64] & (UINT64_C(1) << (vertex % 64))) != 0;
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
    const uint64_t *bits = &state->edges[
        (size_t)edge * (size_t)state->edge_words
    ];
    for (int w = 0; w < state->edge_words; ++w) {
        uint64_t word = bits[w];
        while (word) {
            int v = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
            state->cover[v]--;
        }
    }
}

/* an edge just became uncovered again on backtrack */
static void edge_now_uncovered(MMCSState *state, int edge) {
    const uint64_t *bits = &state->edges[
        (size_t)edge * (size_t)state->edge_words
    ];
    for (int w = 0; w < state->edge_words; ++w) {
        uint64_t word = bits[w];
        while (word) {
            int v = w * 64 + __builtin_ctzll(word);
            word &= word - 1;
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
    state->selected_vertices[state->selected_count++] = vertex;

    for (int idx = state->inc_offset[vertex];
         idx < state->inc_offset[vertex + 1];
         ++idx) {
        int edge = state->inc_edges[idx];

        if (state->hit_count[edge] == 0) {
            state->uncovered_count--;
            edge_now_covered(state, edge);
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
     * MMCS minimality pruning: once a selected vertex has no private edge,
     * adding more vertices cannot make it private again.
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
    for (int idx = state->inc_offset[vertex];
         idx < state->inc_offset[vertex + 1];
         ++idx) {
        int edge = state->inc_edges[idx];

        if (state->hit_count[edge] == 1) {
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
    state->critical_count[vertex] = 0;
}

/* forbidding a vertex removes it from every edge's allowed count */
static void forbid_vertex(MMCSState *state, int vertex) {
    state->forbidden[vertex] = 1;
    for (int idx = state->inc_offset[vertex];
         idx < state->inc_offset[vertex + 1];
         ++idx) {
        state->allowed[state->inc_edges[idx]]--;
    }
}

static void unforbid_vertex(MMCSState *state, int vertex) {
    state->forbidden[vertex] = 0;
    for (int idx = state->inc_offset[vertex];
         idx < state->inc_offset[vertex + 1];
         ++idx) {
        state->allowed[state->inc_edges[idx]]++;
    }
}

static int choose_uncovered_edge(
    const MMCSState *state
) {
    int best_edge = -1;
    int best_allowed = INT_MAX;

    for (int edge = 0; edge < state->edge_count; ++edge) {
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
        state->node_limit > 0 &&
        state->stats->search_nodes >= state->node_limit
    ) {
        state->limit_reached = true;
        return;
    }
    state->stats->search_nodes++;

    if (state->uncovered_count == 0) {
        if (state->selected_count == state->target_size) {
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
    int max_cover = 0;
    for (int vertex = 0; vertex < state->ninputs; ++vertex) {
        if (state->forbidden[vertex]) continue;
        if (state->cover[vertex] > max_cover) max_cover = state->cover[vertex];
    }
    int slots = state->target_size - state->selected_count;
    if (
        max_cover == 0 ||
        (state->uncovered_count + max_cover - 1) / max_cover > slots
    ) {
        return;
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
            .critical_edge_count = pi->OFF_minterms,
            .edge_words = edge_words,
            .target_size = level,
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
