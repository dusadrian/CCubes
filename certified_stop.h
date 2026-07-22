#ifndef CERTIFIED_STOP_H
#define CERTIFIED_STOP_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "utils.h"

/* Per-output state for the certified stopping theorem. */
typedef struct {
    int agreement_horizon; /* A_max */
    int coverage_horizon;  /* k_cov, observed online */
    int cover_lower_bound; /* pairwise-incompatibility certificate */
    bool static_ready;     /* A_max and lower bound have been computed */
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
    bool certification_required;
} BlockingStopState;

/*
 * The current certificate applies to binary point-set inputs: each observed
 * row is a total binary assignment and every output has nonempty ON/OFF sets.
 */
bool certified_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
);

/* Initialize an empty state; coverage can be recorded before static setup. */
void certified_stop_state_reset(CertifiedStopState *state);

/* Lazily compute the static certificate data while preserving k_cov. */
bool certified_stop_state_prepare(
    CertifiedStopState *state,
    const PIstorage *pi,
    int ninputs
);

/* Reset and compute the complete certificate state for one output. */
bool certified_stop_state_init(
    CertifiedStopState *state,
    const PIstorage *pi,
    int ninputs
);

/* Record the first completed level whose retained pool covers every ON row. */
void certified_stop_observe_coverage(
    CertifiedStopState *state,
    int level,
    bool on_set_covered
);

/* k^dagger = max(A_max, k_cov). */
int certified_stop_horizon(const CertifiedStopState *state);

/*
 * A stop is certified either when a feasible cover meets the global lower
 * bound, or when the certified generation horizon is complete and the
 * boundary problem has been solved exactly.
 */
bool certified_stop_should_stop(
    const CertifiedStopState *state,
    int level,
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
    CertifiedStopState *certificate,
    bool report_diagnostic,
    bool adaptive_mode,
    bool plateau_triggered,
    FILE *stream,
    int output_index,
    const PIstorage *pi,
    int ninputs,
    int level,
    const int *selected_indices,
    int selected_terms,
    bool boundary_exact
);

/* Combine plateau, full-certified, and one-way adaptive stopping semantics. */
bool certified_stop_policy_decision(
    const CertifiedStopState *certificate,
    const BlockingStopState *blocking,
    bool certified_mode,
    bool plateau_triggered,
    int level,
    int cover_size,
    bool boundary_exact
);

#endif /* CERTIFIED_STOP_H */
