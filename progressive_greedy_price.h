#ifndef PROGRESSIVE_GREEDY_PRICE_H
#define PROGRESSIVE_GREEDY_PRICE_H

#include <stdbool.h>
#include <stdint.h>

#include "bounded_mmcs.h"

typedef enum {
    PROGRESSIVE_GREEDY_PRICE_OK = 0,
    PROGRESSIVE_GREEDY_PRICE_INVALID,
    PROGRESSIVE_GREEDY_PRICE_OUT_OF_MEMORY
} ProgressiveGreedyPriceStatus;

/* One workspace may be reused by sequential batches, never concurrent calls. */
typedef struct ProgressiveGreedyPriceWorkspace ProgressiveGreedyPriceWorkspace;

ProgressiveGreedyPriceWorkspace *progressive_greedy_price_workspace_create(void);
void progressive_greedy_price_workspace_destroy(
    ProgressiveGreedyPriceWorkspace *workspace
);

typedef struct {
    ProgressiveGreedyPriceWorkspace *workspace;
    int requested_threads;
    int anchor_limit;
    int forced_prefix_depth;
    int beam_candidate_budget;
    int max_support;
    bool scout_anchor_potential;
    uint64_t worker_memory_budget_bytes;
    uint64_t deadline_nanoseconds;
    double reduced_cost_threshold;
    double reduced_cost_tolerance;
} ProgressiveGreedyPriceOptions;

typedef struct {
    int effective_threads;
    int prefix_beam_width;
    int support_minimum;
    int support_maximum;
    double support_mean;
    double best_weight;
    uint64_t anchors_attempted;
    uint64_t scout_anchors_attempted;
    uint64_t scout_candidates_constructed;
    uint64_t scout_literal_evaluations;
    uint64_t candidates_constructed;
    uint64_t improving_candidates;
    uint64_t literal_evaluations;
    uint64_t elapsed_nanoseconds;
    uint64_t allocated_bytes;
    bool deadline_reached;
} ProgressiveGreedyPriceStats;

/*
 * Construct one ordinary unbounded prime candidate per dual-ranked ON anchor.
 * Optional prefix diversification repeats the construction after forcing
 * every one-literal prefix (depth 1), or every one- and two-literal prefix
 * (depth 2), or every prefix through three literals (depth 3), then applies
 * the same redundancy deletion.
 *
 * Each literal is fixed to the anchor value. The greedy choice first prefers
 * literals that eliminate OFF blockers without losing compatible ON dual
 * weight, then maximizes newly eliminated blockers per unit of dual loss.
 * Redundant literals are deleted before exact coverage and reduced cost are
 * computed. Only candidates above the supplied reduced-cost threshold enter
 * `batch`. `max_support == 0` leaves support unbounded; otherwise candidates
 * wider than the declared support cap are discarded before batching.
 */
ProgressiveGreedyPriceStatus progressive_greedy_price_batch(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    const double *row_weights,
    WeightedMMCSPriceBatch *batch,
    const ProgressiveGreedyPriceOptions *options,
    ProgressiveGreedyPriceStats *stats
);

bool progressive_greedy_price_candidate_capacity(
    int anchors,
    int ninputs,
    int forced_prefix_depth,
    int *capacity
);

bool progressive_greedy_price_beam_capacity(
    int anchors,
    int ninputs,
    int candidate_budget,
    int *capacity,
    int *beam_width
);

#endif
