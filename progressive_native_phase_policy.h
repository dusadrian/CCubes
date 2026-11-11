/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_NATIVE_PHASE_POLICY_H
#define PROGRESSIVE_NATIVE_PHASE_POLICY_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int stagnant_epochs;
} ProgressiveNativePhasePolicy;

typedef struct {
    bool cover_improved;
    bool selected_new_column;
    bool jump_improved;
    int columns_added;
    /* Whether the configured wall budget can still pay for another phase.
     * Computed by the caller, which owns the clock. */
    bool budget_allows_another_epoch;
    /* No wall deadline was configured for this search. */
    bool unbudgeted;
    /* Explicit quality-first override: keep promoting a stagnant unbudgeted
     * phase while its column-growth rate remains high.  The production
     * default is false, so two unused-column epochs establish a plateau. */
    bool allow_unbudgeted_growth;
    /* Completed pricing rounds spent in the phase being judged. */
    int rounds_in_phase;
} ProgressiveNativePhaseSignals;

typedef enum {
    PROGRESSIVE_NATIVE_PHASE_PROMOTE_PRODUCTIVE = 0,
    PROGRESSIVE_NATIVE_PHASE_PROMOTE_GRACE,
    PROGRESSIVE_NATIVE_PHASE_PROMOTE_BUDGET,
    PROGRESSIVE_NATIVE_PHASE_PROMOTE_GROWTH,
    PROGRESSIVE_NATIVE_PHASE_STOP_EMPTY,
    PROGRESSIVE_NATIVE_PHASE_STOP_STAGNANT
} ProgressiveNativePhaseDecision;

void progressive_native_phase_policy_initialize(
    ProgressiveNativePhasePolicy *policy
);

/*
 * Return the cumulative completed-pricing-round milestone at which pricing
 * should yield to pool exploitation.  The sequence is 8, 16, 32, ... and is
 * therefore independent of machine speed and worker count.  Zero means that
 * the requested index cannot be represented.
 */
uint64_t progressive_native_phase_checkpoint_round(
    int checkpoint_index
);

/* The work policy starts only after enough complete pricing rounds exist to
 * calibrate the rolling round-cost estimate. */
bool progressive_native_phase_policy_ready(
    uint64_t completed_pricing_rounds
);

/*
 * Automatic runs treat their configured limit as a target.  A real
 * cardinality improvement near that target earns a rolling grace window.
 * Return an estimated wall-clock grace: the requested 5% share is clamped to
 * one through four times the trailing-median observed pricing-round duration.
 * This approximates, rather than guarantees, that many future rounds.
 */
double progressive_native_soft_target_grace_seconds(
    double time_limit_seconds,
    double observed_round_seconds
);

/* Renew only from an improvement completed while the incoming deadline was
 * still live.  An atomic operator that returns late keeps its result but must
 * not open a fresh tranche after the target has already expired. */
uint64_t progressive_native_soft_target_renew_deadline(
    uint64_t current_deadline_nanoseconds,
    uint64_t improvement_nanoseconds,
    double grace_seconds
);

/* Work-mode phase operators are atomic; numeric wall-mode ablations retain
 * their supplied deadline. */
uint64_t progressive_native_phase_operator_deadline(
    bool work_budget,
    uint64_t overall_deadline_nanoseconds
);

/*
 * Whether the wall budget can still pay for another phase.  Phase milestones
 * double (8, 16, 32, ... completed pricing rounds), so the next phase costs
 * about as much as every round so far; elapsed time is therefore the estimate
 * of what continuing costs.  Returns false when no budget was configured --
 * an unbudgeted run has no remaining budget to spend, and its stopping rule
 * stays exactly as before.
 */
bool progressive_native_phase_budget_allows_epoch(
    uint64_t started_nanoseconds,
    uint64_t now_nanoseconds,
    uint64_t deadline_nanoseconds
);

/*
 * Whether pricing is still enlarging the master fast enough to be worth
 * another phase in the explicit quality-first tail mode.  The production
 * unlimited-time default does not use column growth alone to defeat a cover
 * plateau: generated columns must demonstrate value by entering or improving
 * the cover.  The opt-in test is structural: at least one new column per
 * completed pricing round of the phase.
 */
bool progressive_native_phase_growth_sustains(
    int columns_added,
    int rounds_in_phase
);

ProgressiveNativePhaseDecision progressive_native_phase_policy_observe(
    ProgressiveNativePhasePolicy *policy,
    ProgressiveNativePhaseSignals signals
);

bool progressive_native_phase_decision_promotes(
    ProgressiveNativePhaseDecision decision
);

const char *progressive_native_phase_decision_name(
    ProgressiveNativePhaseDecision decision
);

#endif
