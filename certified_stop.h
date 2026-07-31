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

/*
 * Structural warning evaluated at a plateau.  A private witness is an ON row
 * covered by exactly one term of the current cover.  A delayed private pair is
 * compatible in the full cube space, but no retained PI through the current
 * level covers both witnesses.  The count is exact for the explored family;
 * it is a warning about possible future mergers, not an optimality certificate.
 */
typedef struct {
    int level;
    int selected_terms;
    int private_witnesses;
    uint64_t private_pairs;
    uint64_t incompatible_private_pairs;
    uint64_t shallow_private_pairs;
    uint64_t delayed_private_pairs;
    double sparse_load;       /* m0 * 2^(-level) */
    double model_union_bound; /* ex-ante all-pair, with-replacement OFF model */
    bool model_union_bound_available;
} BlockingDiagnostic;

/* Per-output state for the observational/adaptive plateau diagnostic. */
typedef struct {
    bool reported;
    bool warning_detected;
    bool pool_warning_avoided;
} BlockingStopState;

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

/* Compute the exact pair-level plateau warning and sparse-model metadata. */
bool certified_blocking_diagnostic(
    const PIstorage *pi,
    int ninputs,
    int level,
    const int *selected_indices,
    int selected_terms,
    bool include_model_metadata,
    BlockingDiagnostic *diagnostic
);

/* Emit one stable, machine-readable diagnostic record. */
void certified_blocking_diagnostic_print(
    FILE *stream,
    int output_index,
    bool boundary_exact,
    const BlockingDiagnostic *diagnostic
);

/* Initialize, then observe at most the first terminating plateau. */
void certified_blocking_state_init(BlockingStopState *state);

bool certified_blocking_observe_plateau(
    BlockingStopState *state,
    bool report_diagnostic,
    bool adaptive_mode,
    bool plateau_triggered,
    FILE *stream,
    int output_index,
    const PIstorage *pi,
    int ninputs,
    int level,
    int *selected_indices,
    int selected_terms,
    bool boundary_exact,
    bool inspect_equal_pool
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
