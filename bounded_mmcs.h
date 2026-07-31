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
 * one minimal-transversal traversal per canonical ON anchor. Unlike repeated
 * level generation, this does not restart the search for every support size.
 *
 * The bounded form is transactional: a node-limited result appends nothing,
 * so only BOUNDED_MMCS_COMPLETE may be used as a complete-chart certificate.
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
