/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
#ifndef CCUBES_IPC_WEIGHTED_COVER_H
#define CCUBES_IPC_WEIGHTED_COVER_H

#include "cover_encoding.h"

typedef struct {
    int lower, terms;
    int lower_target; /* Zero if the lower-target probe did not run. */
    int *cover; /* Transposed. The caller frees it. */
    bool supported;
    NsatStatus status;
    NsatStopReason stop;
    uint64_t compatible, nodes;
    size_t maximal;
    unsigned pivots;
    uint64_t repriced, reprice_pivots, reprice_pruned;
    int first_terms; /* Zero until a smaller cover has been materialized. */
    uint64_t first_cover_work, first_cover_nodes;
    NsatStatus check_status; /* UNSAT if the returned improvement was checked. */
    NsatStats check;
    double check_seconds;
    uint64_t check_auto_base, check_auto_limit; /* Cumulative policy for the last claim. */
    bool check_auto_limit_hit;
} IpcWeightedCoverResult;

typedef struct {
    double search_seconds;
    double verifier_seconds;
    uint64_t verifier_work; /* Zero disables the checker-specific work limit. */
} IpcWeightedCheckBudget;

/* input->target is the size of a validated incumbent. A limited search can
 * still return earlier proved bounds and completed smaller covers. A bound
 * above the supplied lower bound has passed independent verification. */
IpcWeightedCoverResult ipc_weighted_cover(const NsatCoverInput *input, int lower,
                                          NsatLimits limits);

/* Re-price search nodes starting with the first decision. */
IpcWeightedCoverResult ipc_weighted_cover_reprice(const NsatCoverInput *input,
                                                  int lower, NsatLimits limits,
                                                  size_t min_branches, unsigned pivots);

/* Allocate checker work from cumulative search work within the caller's limits. */
IpcWeightedCoverResult ipc_weighted_cover_reprice_with_auto_check(const NsatCoverInput *input,
    int lower, NsatLimits limits, size_t min_branches, unsigned pivots);

/* Use separate cumulative search and checker clocks. limits.work governs
 * search only. Shared limits still govern the combined work and memory. */
IpcWeightedCoverResult ipc_weighted_cover_reprice_with_check_budget(const NsatCoverInput *input,
    int lower, NsatLimits limits, size_t min_branches, unsigned pivots,
    const IpcWeightedCheckBudget *budget);

/* Search with the root weights until a smaller cover has been materialized.
 * Re-price subsequent targets within the same chart and resource limits. */
IpcWeightedCoverResult ipc_weighted_cover_seed_then_reprice(const NsatCoverInput *input,
                                                            int lower, NsatLimits limits,
                                                            size_t min_branches, unsigned pivots);

#endif
