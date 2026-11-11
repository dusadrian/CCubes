/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_native_phase_policy.h"

#include <limits.h>
#include <stdint.h>

#define PROGRESSIVE_NATIVE_PHASE_FIRST_CHECKPOINT_ROUND UINT64_C(8)
#define PROGRESSIVE_NATIVE_PHASE_RATE_SAMPLE_ROUNDS UINT64_C(3)
#define PROGRESSIVE_NATIVE_SOFT_TARGET_SHARE 0.05
#define PROGRESSIVE_NATIVE_SOFT_TARGET_MINIMUM_ROUNDS 1.0
#define PROGRESSIVE_NATIVE_SOFT_TARGET_MAXIMUM_ROUNDS 4.0

void progressive_native_phase_policy_initialize(
    ProgressiveNativePhasePolicy *policy
) {
    if (!policy) return;
    policy->stagnant_epochs = 0;
}

uint64_t progressive_native_phase_checkpoint_round(
    int checkpoint_index
) {
    if (checkpoint_index < 0 || checkpoint_index >= 61) return 0;
    const uint64_t first =
        PROGRESSIVE_NATIVE_PHASE_FIRST_CHECKPOINT_ROUND;
    if (first > UINT64_MAX >> checkpoint_index) return 0;
    return first << checkpoint_index;
}

bool progressive_native_phase_policy_ready(
    uint64_t completed_pricing_rounds
) {
    return
        completed_pricing_rounds >=
        PROGRESSIVE_NATIVE_PHASE_RATE_SAMPLE_ROUNDS;
}

double progressive_native_soft_target_grace_seconds(
    double time_limit_seconds,
    double observed_round_seconds
) {
    if (time_limit_seconds <= 0.0 || observed_round_seconds <= 0.0) {
        return 0.0;
    }
    /* The caller supplies its trailing-median pricing-round duration. */
    double grace =
        time_limit_seconds * PROGRESSIVE_NATIVE_SOFT_TARGET_SHARE;
    const double minimum = observed_round_seconds *
        PROGRESSIVE_NATIVE_SOFT_TARGET_MINIMUM_ROUNDS;
    const double maximum = observed_round_seconds *
        PROGRESSIVE_NATIVE_SOFT_TARGET_MAXIMUM_ROUNDS;
    if (grace < minimum) {
        grace = minimum;
    }
    if (grace > maximum) {
        grace = maximum;
    }
    return grace;
}

uint64_t progressive_native_soft_target_renew_deadline(
    uint64_t current_deadline_nanoseconds,
    uint64_t improvement_nanoseconds,
    double grace_seconds
) {
    if (
        current_deadline_nanoseconds == 0 ||
        improvement_nanoseconds == 0 ||
        improvement_nanoseconds >= current_deadline_nanoseconds ||
        grace_seconds <= 0.0
    ) {
        return current_deadline_nanoseconds;
    }
    const double span = grace_seconds * 1e9;
    const uint64_t renewed =
        span >= (double)(UINT64_MAX - improvement_nanoseconds)
            ? UINT64_MAX
            : improvement_nanoseconds + (uint64_t)span;
    return renewed > current_deadline_nanoseconds
        ? renewed
        : current_deadline_nanoseconds;
}

uint64_t progressive_native_phase_operator_deadline(
    bool work_budget,
    uint64_t overall_deadline_nanoseconds
) {
    return work_budget ? 0 : overall_deadline_nanoseconds;
}

bool progressive_native_phase_growth_sustains(
    int columns_added,
    int rounds_in_phase
) {
    if (columns_added <= 0 || rounds_in_phase <= 0) return false;
    return columns_added >= rounds_in_phase;
}

bool progressive_native_phase_budget_allows_epoch(
    uint64_t started_nanoseconds,
    uint64_t now_nanoseconds,
    uint64_t deadline_nanoseconds
) {
    if (deadline_nanoseconds == 0) return false;
    if (now_nanoseconds >= deadline_nanoseconds) return false;
    if (started_nanoseconds == 0 || now_nanoseconds <= started_nanoseconds) {
        return false;
    }
    const uint64_t elapsed = now_nanoseconds - started_nanoseconds;
    const uint64_t remaining = deadline_nanoseconds - now_nanoseconds;
    return remaining > elapsed;
}

ProgressiveNativePhaseDecision progressive_native_phase_policy_observe(
    ProgressiveNativePhasePolicy *policy,
    ProgressiveNativePhaseSignals signals
) {
    if (!policy) return PROGRESSIVE_NATIVE_PHASE_STOP_STAGNANT;
    if (
        signals.cover_improved ||
        signals.selected_new_column ||
        signals.jump_improved
    ) {
        policy->stagnant_epochs = 0;
        return PROGRESSIVE_NATIVE_PHASE_PROMOTE_PRODUCTIVE;
    }
    if (signals.columns_added <= 0) {
        return PROGRESSIVE_NATIVE_PHASE_STOP_EMPTY;
    }
    policy->stagnant_epochs++;
    if (policy->stagnant_epochs < 2) {
        return PROGRESSIVE_NATIVE_PHASE_PROMOTE_GRACE;
    }
    /*
     * Stagnation alone is not exhaustion.  These phases still add thousands of
     * columns each; they simply have not yet produced one the master selects.
     * Stopping there strands whatever wall budget the caller asked for -- on
     * the 180 s family-1 sweep twelve outputs quit with 1,492 s unspent, three
     * of them on a worse cover than the same budget previously reached.  While
     * the budget can still pay for another phase, spend it: the incumbent only
     * ever improves, so the additional phase can cost time but not quality.
     * An exhausted budget, or no budget at all, stops exactly as before.
     */
    if (signals.budget_allows_another_epoch) {
        return PROGRESSIVE_NATIVE_PHASE_PROMOTE_BUDGET;
    }
    /* Column growth by itself is not useful progress.  By default an
     * unbudgeted search stops after two consecutive epochs whose new columns
     * neither improve nor enter the cover.  Users explicitly requesting the
     * quality-first tail may retain the former growth-sustained behavior. */
    if (
        signals.unbudgeted &&
        signals.allow_unbudgeted_growth &&
        progressive_native_phase_growth_sustains(
            signals.columns_added,
            signals.rounds_in_phase
        )
    ) {
        return PROGRESSIVE_NATIVE_PHASE_PROMOTE_GROWTH;
    }
    return PROGRESSIVE_NATIVE_PHASE_STOP_STAGNANT;
}

bool progressive_native_phase_decision_promotes(
    ProgressiveNativePhaseDecision decision
) {
    return
        decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_PRODUCTIVE ||
        decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_GRACE ||
        decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_BUDGET ||
        decision == PROGRESSIVE_NATIVE_PHASE_PROMOTE_GROWTH;
}

const char *progressive_native_phase_decision_name(
    ProgressiveNativePhaseDecision decision
) {
    switch (decision) {
        case PROGRESSIVE_NATIVE_PHASE_PROMOTE_PRODUCTIVE:
            return "productive";
        case PROGRESSIVE_NATIVE_PHASE_PROMOTE_GRACE:
            return "new-columns-grace";
        case PROGRESSIVE_NATIVE_PHASE_PROMOTE_BUDGET:
            return "stagnant-budget-remains";
        case PROGRESSIVE_NATIVE_PHASE_PROMOTE_GROWTH:
            return "stagnant-still-growing";
        case PROGRESSIVE_NATIVE_PHASE_STOP_EMPTY:
            return "no-new-columns";
        case PROGRESSIVE_NATIVE_PHASE_STOP_STAGNANT:
            return "unused-columns-stagnant";
    }
    return "unknown";
}
