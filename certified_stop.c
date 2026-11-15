/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "certified_stop.h"

void certified_stop_state_reset(CertifiedStopState *state) {
    if (state) *state = (CertifiedStopState){0};
}

void certified_stop_observe_complete_prime_chart(
    CertifiedStopState *state
) {
    if (state) state->complete_prime_chart = true;
}

bool certified_stop_should_stop(
    const CertifiedStopState *state,
    int cover_size,
    bool boundary_exact
) {
    return state &&
        state->complete_prime_chart &&
        cover_size > 0 &&
        boundary_exact;
}

bool certified_stop_policy_decision(
    const CertifiedStopState *certificate,
    bool certified_mode,
    bool plateau_triggered,
    int level,
    int cover_size,
    bool boundary_exact,
    FILE *stream,
    int output_index
) {
    FILE *output = stream ? stream : stderr;

    /* Every supported output has a nonempty ON set, so one term is optimal. */
    if (cover_size == 1) {
        if (certified_mode) {
            fprintf(
                output,
                "CCUBES_CERTIFICATE output=%d scope=global "
                "method=cardinality-floor status=certified level=%d "
                "cover=%d\n",
                output_index,
                level,
                cover_size
            );
        }
        return true;
    }

    if (certified_mode) {
        bool certified = certified_stop_should_stop(
            certificate,
            cover_size,
            boundary_exact
        );
        fprintf(
            output,
            "CCUBES_CERTIFICATE output=%d scope=global "
            "method=complete-prime-chart status=%s level=%d "
            "cover=%d boundary_exact=%d\n",
            output_index,
            certified ? "certified" : certificate &&
                certificate->complete_prime_chart
                    ? "boundary-unproved"
                    : "enumeration-incomplete",
            level,
            cover_size,
            boundary_exact ? 1 : 0
        );
        return certified;
    }

    return plateau_triggered;
}
