/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_LITERAL_POLISH_H
#define PROGRESSIVE_LITERAL_POLISH_H

#include <stdbool.h>
#include <stdint.h>

#include "bounded_mmcs.h"

typedef struct {
    int passes;
    int cubes_examined;
    int cubes_reduced;
    int proposals;
    int stale_proposals;
    int threads_used;
    int literals_before;
    int literals_after;
    uint64_t search_nodes;
    uint64_t allocated_bytes;
    uint64_t active_list_bytes;
    bool node_limit_reached;
    bool active_lists_used;
    bool deadline_reached;
    double seconds;
} ProgressiveLiteralPolishStats;

/*
 * Reduce the input-literal count of a validated cover without changing its
 * cardinality.
 *
 * Every selected cube is already a prime implicant, so no literal can simply
 * be deleted from it. A cube may nevertheless be replaceable by a different
 * prime with fewer literals: only the ON rows that no other selected cube
 * covers constrain it. For each cube this routine forms the agreement pattern
 * over those private rows, which is the most specific cube covering them, and
 * searches for a minimum-cardinality subset of that pattern's fixed positions
 * still disjoint from every OFF row.
 *
 * Cubes are processed sequentially and each replacement is applied before the
 * next cube's live private rows are checked. Candidate searches run in
 * parallel from one immutable pass snapshot. Proposals are then committed in
 * deterministic cube order and rejected if preceding commits made them stale.
 * This matters because replacing several cubes blindly from one snapshot
 * would be unsound: a row covered by two cubes is private to neither and both
 * replacements could otherwise drop it.
 *
 * `solution` is the input-major `ninputs * cardinality` representation used by
 * the progressive result: zero is a dash, one is input value zero, and two is
 * input value one. It is modified in place and the cover remains valid.
 * `memory_budget_bytes` bounds fixed and worker scratch. The physical team is
 * reduced when necessary; optional active-list scratch falls back to the exact
 * ordered scan when it does not fit. Zero disables active lists and leaves the
 * mandatory scratch uncapped for standalone callers.
 * `deadline_nanoseconds` is an optional monotonic wall deadline. It is
 * inspected between complete passes, never inside one: when it has expired,
 * the already-committed passes are kept, `deadline_reached` is set, and the
 * result remains valid — a bounded run therefore preserves partial literal
 * reductions instead of losing the whole polish tail.
 * Returns false on allocation, worker-launch, or validation failure; in every
 * failure case `solution` is restored to its original value.
 */
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
);

#endif
