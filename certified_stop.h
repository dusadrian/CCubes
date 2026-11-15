#ifndef CERTIFIED_STOP_H
#define CERTIFIED_STOP_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "utils.h"

/* Per-output state for complete-chart certification. */
typedef struct {
    bool complete_prime_chart; /* every prime cube geometry is retained */
} CertifiedStopState;

/* Initialize an empty complete-chart state. */
void certified_stop_state_reset(CertifiedStopState *state);

/* Record transactional completion of complete-prime enumeration. */
void certified_stop_observe_complete_prime_chart(
    CertifiedStopState *state
);

/*
 * A nontrivial stop is certified only after complete transactional prime
 * enumeration and an exact solve of that complete chart. The caller must have
 * established that ON and OFF are nonempty, disjoint, fully specified binary
 * point sets; certified_model_supported() enforces that contract for the CLI.
 */
bool certified_stop_should_stop(
    const CertifiedStopState *state,
    int cover_size,
    bool boundary_exact
);

/* Combine heuristic plateau and complete-chart certified stopping semantics. */
bool certified_stop_policy_decision(
    const CertifiedStopState *certificate,
    bool certified_mode,
    bool plateau_triggered,
    int level,
    int cover_size,
    bool boundary_exact,
    FILE *stream,
    int output_index
);

#endif /* CERTIFIED_STOP_H */
