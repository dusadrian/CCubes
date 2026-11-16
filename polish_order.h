/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
#ifndef CCUBES_POLISH_ORDER_H
#define CCUBES_POLISH_ORDER_H

#include <math.h>
#include <stdbool.h>

typedef struct {
    int col;
    int new_cover;
    int cover_count;
    double lagr_score;
    double weight;
} PolishCandidate;

static bool polish_candidate_better(const PolishCandidate *a, const PolishCandidate *b) {
    if (a->new_cover != b->new_cover) return a->new_cover > b->new_cover;
    if (fabs(a->lagr_score - b->lagr_score) > 1e-12) return a->lagr_score < b->lagr_score;
    if (a->cover_count != b->cover_count) return a->cover_count > b->cover_count;
    if (fabs(a->weight - b->weight) > 1e-12) return a->weight > b->weight;
    return a->col < b->col;
}

static void polish_sort_candidates(PolishCandidate *candidates, int count) {
    if (count > 32) {
        /*
        Wide nodes: a full sort is wasted work — on failure every branch is
        explored regardless of order, and a successful dive follows the front
        candidates. The previous partial selection sort still compared every
        one of the front 8 slots against the whole remaining range (O(8n)
        comparisons unconditionally); this streams the front 8 once instead:
        seed them with a small insertion sort, then give every remaining
        candidate a single comparison against the current 8th-best, only
        paying the (rare) shift cost when it actually displaces one of the
        front 8. Same top-8 set and order as before -- this is the standard
        streaming top-K argument, not a change in which candidates rank best
        -- just far fewer comparisons (O(n) with a small constant instead of
        O(8n)). The bumped-out front candidate is swapped into the vacated
        tail slot so the full candidate multiset (needed by the caller's
        subsequent scan over all `count` of them) is preserved either way;
        only the tail's incidental leftover order differs from before.
        */
        const int front = 8;
        for (int i = 1; i < front; ++i) {
            PolishCandidate value = candidates[i];
            int j = i - 1;
            while (j >= 0 && polish_candidate_better(&value, &candidates[j])) {
                candidates[j + 1] = candidates[j];
                --j;
            }
            candidates[j + 1] = value;
        }
        for (int i = front; i < count; ++i) {
            if (!polish_candidate_better(&candidates[i], &candidates[front - 1])) continue;

            PolishCandidate incoming = candidates[i];
            PolishCandidate outgoing = candidates[front - 1];
            int j = front - 1;
            while (j > 0 && polish_candidate_better(&incoming, &candidates[j - 1])) {
                candidates[j] = candidates[j - 1];
                --j;
            }
            candidates[j] = incoming;
            candidates[i] = outgoing;
        }
        return;
    }
    for (int i = 1; i < count; ++i) {
        PolishCandidate value = candidates[i];
        int j = i - 1;
        while (j >= 0 && polish_candidate_better(&value, &candidates[j])) {
            candidates[j + 1] = candidates[j];
            --j;
        }
        candidates[j + 1] = value;
    }
}

#endif
