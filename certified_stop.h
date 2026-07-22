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
    bool warning_detected;
    bool certification_required;
    bool escalation_suppressed;
    bool pool_warning_avoided;
    bool boundary_horizon_unproved;
    int certification_horizon;
    uint64_t estimated_remaining_tasks;
    bool task_estimate_capped;
} BlockingStopState;

/*
 * Default adaptive stopping is an engineering safeguard, not an implicit
 * request for exhaustive search.  It may continue only when the remaining
 * certified horizon contains at most this many position-subset tasks.  The
 * explicit -c mode is intentionally not subject to this limit.
 */
#define CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT UINT64_C(1000000)

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

/*
 * Estimate sum(C(ninputs, k), k = level + 1 .. horizon), capped at limit + 1.
 * The return value says whether the complete remaining horizon fits the limit.
 */
bool certified_stop_adaptive_work_within_limit(
    int ninputs,
    int level,
    int horizon,
    uint64_t limit,
    uint64_t *estimated_tasks
);

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
    int *selected_indices,
    int selected_terms,
    bool boundary_exact,
    bool inspect_equal_pool
);

/* Combine plateau, full-certified, and one-way adaptive stopping semantics. */
bool certified_stop_policy_decision(
    const CertifiedStopState *certificate,
    BlockingStopState *blocking,
    bool certified_mode,
    bool plateau_triggered,
    int level,
    int cover_size,
    bool boundary_exact,
    FILE *stream,
    int output_index
);

#endif /* CERTIFIED_STOP_H */
