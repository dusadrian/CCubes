#ifndef BOUNDED_MMCS_H
#define BOUNDED_MMCS_H

#include <stdbool.h>
#include <stdint.h>

#include "utils.h"

typedef struct {
    uint64_t search_nodes;
    uint64_t completed_transversals;
    uint64_t duplicate_cubes;
    uint64_t unique_cubes;
} BoundedMMCSStats;

typedef enum {
    BOUNDED_MMCS_ERROR = -1,
    BOUNDED_MMCS_LIMIT_REACHED = 0,
    BOUNDED_MMCS_COMPLETE = 1
} BoundedMMCSResult;

typedef enum {
    WEIGHTED_MMCS_PRICE_OK = 0,
    WEIGHTED_MMCS_PRICE_LIMIT_REACHED,
    WEIGHTED_MMCS_PRICE_UNSUPPORTED,
    WEIGHTED_MMCS_PRICE_INVALID,
    WEIGHTED_MMCS_PRICE_OUT_OF_MEMORY
} WeightedMMCSPriceStatus;

typedef struct {
    uint64_t anchors_presented;
    uint64_t anchors_constructed;
    uint64_t prospective_gated_anchors;
    uint64_t edges_constructed;
    uint64_t xor_words;
    uint64_t incidences_counted;
    uint64_t incidences_written;
    uint64_t prospective_gated_edges;
    uint64_t prospective_gated_incidences;
    uint64_t pivot_selections;
    uint64_t pivot_edge_entries_inspected;
    uint64_t literal_incidence_entries_visited;
    uint64_t edge_literal_entries_visited;
    uint64_t compatible_bound_calls;
    uint64_t compatible_bound_rows_examined;
    uint64_t edge_build_nanoseconds;
    uint64_t incidence_count_nanoseconds;
    uint64_t incidence_fill_nanoseconds;
    uint64_t search_nanoseconds;
    uint64_t candidate_collection_nanoseconds;
    uint64_t merge_nanoseconds;
    uint64_t pricing_call_nanoseconds;
} WeightedMMCSPriceTelemetry;

typedef struct {
    double best_weight;
    /*
     * Certified upper bound on the weight of every eligible cube through the
     * requested support horizon.  At full completion this equals
     * `best_weight`. A deadline-limited sweep combines the best valid cube
     * found so far with the admissible suffix root bound beginning at the
     * earliest interrupted or unassigned anchor, so callers retain a sound
     * anytime bound even when recursion stops inside an anchor.
     */
    double max_weight_upper_bound;
    int best_anchor;
    int best_support_size;
    int anchors_completed;
    int anchors_total;
    int effective_threads;
    uint64_t estimated_bytes_per_worker;
    uint64_t peak_incidence_entries;
    uint64_t search_nodes;
    uint64_t weight_prunes;
    uint64_t completed_transversals;
    bool limit_reached;
    bool anchor_sweep_complete;
    WeightedMMCSPriceTelemetry telemetry;
} WeightedMMCSPriceResult;

typedef struct {
    int capacity;
    int count;
    double *weights;
    int *anchors;
    int *support_sizes;
    int *vertices;          /* capacity * ninputs */
    uint64_t *coverage;     /* capacity * pi->pichart_words */
} WeightedMMCSPriceBatch;

typedef struct {
    int ninputs;
    int row_words;
    int on_rows;
    int off_rows;
    int on_column_words;
    int off_column_words;
    uint64_t *on_bits;
    uint64_t *off_bits;
    uint64_t *on_columns;
    uint64_t *off_columns;
    uint32_t *on_column_prefix_ones;
    uint32_t *off_one_counts;
} WeightedMMCSPreparedRows;

typedef struct {
    int requested_threads;
    uint64_t worker_memory_budget_bytes;
    bool collect_timing;
    /*
     * Diagnostic switches. The prepared transposed-column backend is the
     * default. Progressive reduced-cost pricing explicitly enables the root
     * gate; exact maximum-weight callers leave it disabled.
     */
    bool enable_root_gate;
    bool use_materialized_backend;
    /*
     * Retain the packed anchor/row XOR words for the current anchor while
     * continuing to use prepared transposed columns for incidence traversal.
     */
    bool cache_edge_words;
    /*
     * Diagnostic ablation of the remaining-slot max-cover bound. Disabling
     * the bound preserves enumeration correctness, but avoids maintaining
     * cover[v] across every edge cover/uncover transition.
     */
    bool disable_cover_bound;
    /*
     * Maximum canonical anchor prefix to inspect. Zero means every anchor.
     * Root upper bounds are non-increasing in this order, so a prefix is the
     * deterministic descending-bound tranche used by partial pricing.
     */
    int anchor_limit;
    /* Absolute CLOCK_MONOTONIC deadline; zero means unlimited. */
    uint64_t deadline_nanoseconds;
} WeightedMMCSPriceOptions;

/*
 * Generate all prime implicants having exactly `level` fixed inputs for one
 * fully specified binary output.  New records are appended to `pi`.
 *
 * This is an independent implementation of the MMCS search principles:
 * branch on an uncovered hyperedge and maintain each selected vertex's
 * critical (private) hyperedges.  No source code from the MMCS distribution
 * is used here.  Algorithmic reference: K. Murakami and T. Uno,
 * "Efficient Algorithms for Dualizing Large-Scale Hypergraphs",
 * arXiv:1102.3813.
 */
bool bounded_mmcs_generate_output_level(
    PIstorage *pi,
    int ninputs,
    int level,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    BoundedMMCSStats *stats
);

/*
 * Transactional bounded form used by automatic generator selection.
 * `node_limit == 0` means unlimited.  If the limit is reached, no record is
 * appended to `pi`; callers may safely complete the level with another
 * generator.
 */
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
);

/*
 * Enumerate the complete prime set for one fully specified binary output in
 * one minimal-transversal traversal per canonical ON anchor. The bounded form
 * is transactional: only BOUNDED_MMCS_COMPLETE may certify a complete chart.
 */
bool bounded_mmcs_generate_output_all(
    PIstorage *pi,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    BoundedMMCSStats *stats
);

BoundedMMCSResult bounded_mmcs_generate_output_all_limited(
    PIstorage *pi,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    uint64_t node_limit,
    BoundedMMCSStats *stats
);

/*
 * Find the maximum-dual-weight prime implicant through `max_level` without
 * appending anything to `pi`. The returned support is written to
 * `best_vertices` (capacity `ninputs`); its values are those of
 * ON_set[best_anchor]. If the horizon contains no complete transversal, the
 * call succeeds with `best_anchor == -1` and an empty batch.
 *
 * The oracle is exact for the same fully specified binary point-row model as
 * bounded MMCS generation. At each search node, the total positive dual
 * weight of ON rows still compatible with the partial cube is an admissible
 * upper bound, because adding literals can only remove compatible rows.
 */
WeightedMMCSPriceStatus bounded_mmcs_price_output(
    const PIstorage *pi,
    int ninputs,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceResult *result
);

WeightedMMCSPriceStatus bounded_mmcs_price_output_batch(
    const PIstorage *pi,
    int ninputs,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceBatch *batch,
    WeightedMMCSPriceResult *result
);

bool bounded_mmcs_prepare_binary_rows(
    const PIstorage *pi,
    int ninputs,
    WeightedMMCSPreparedRows *prepared
);

void bounded_mmcs_prepared_rows_destroy(
    WeightedMMCSPreparedRows *prepared
);

/*
 * Pricing variant used by the persistent progressive master. Packed binary
 * rows are prepared once and reused across pricing rounds and horizons.
 * Worker selection is bounded by the supplied memory envelope.
 */
WeightedMMCSPriceStatus bounded_mmcs_price_output_batch_prepared(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    int max_level,
    const double *row_weights,
    int *best_vertices,
    WeightedMMCSPriceBatch *batch,
    const WeightedMMCSPriceOptions *options,
    WeightedMMCSPriceResult *result
);

/*
 * Geometry is generated independently per output.  Recompute the number of
 * other outputs containing each newly generated cube before level
 * canonicalization performs coverage-equivalence pruning.
 */
bool bounded_mmcs_mark_level_sharing(
    PIstorage *PInfo,
    int noutputs,
    const int *level_start,
    int implicant_words,
    int *max_shared
);

#endif
