#ifndef PROGRESSIVE_SEED_H
#define PROGRESSIVE_SEED_H

#include <stdbool.h>
#include <stdint.h>

#include "bounded_mmcs.h"

typedef enum {
    PROGRESSIVE_SEED_OK = 0,
    PROGRESSIVE_SEED_INVALID,
    PROGRESSIVE_SEED_OUT_OF_MEMORY
} ProgressiveSeedStatus;

typedef struct {
    int cardinality;
    int greedy_cubes;
    int fallback_cubes;
    int literal_count;
    int ninputs;
    int coverage_words;
    bool greedy_complete;
    bool memory_exhausted;
    uint64_t elapsed_nanoseconds;
    uint64_t validation_elapsed_nanoseconds;
    uint64_t allocated_bytes;

    /* Row-major cardinality x ninputs patterns. */
    int *patterns;

    /*
     * Dense ON coverage for the first greedy_cubes patterns only. Fallback
     * minterms deliberately have no dense coverage allocation.
     */
    uint64_t *greedy_coverage;
} ProgressiveSeedResult;

ProgressiveSeedStatus progressive_seed_build(
    const PIstorage *pi,
    int ninputs,
    bool enable_greedy,
    int max_support,
    uint64_t memory_budget_bytes,
    ProgressiveSeedResult *result
);

/* Reuse immutable columns prepared from this same PIstorage. */
ProgressiveSeedStatus progressive_seed_build_prepared(
    const PIstorage *pi,
    const WeightedMMCSPreparedRows *prepared,
    int ninputs,
    bool enable_greedy,
    int max_support,
    uint64_t memory_budget_bytes,
    ProgressiveSeedResult *result
);

bool progressive_seed_pattern_is_valid(
    const PIstorage *pi,
    int ninputs,
    const int *pattern
);

bool progressive_seed_pattern_is_prime(
    const PIstorage *pi,
    int ninputs,
    const int *pattern
);

bool progressive_seed_cover_is_valid(
    const PIstorage *pi,
    int ninputs,
    const ProgressiveSeedResult *result
);

void progressive_seed_result_destroy(ProgressiveSeedResult *result);

#endif
