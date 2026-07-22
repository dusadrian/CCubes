/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "certified_stop.h"

#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

bool certified_model_supported(
    const PIstorage *PInfo,
    int ninputs,
    int noutputs
) {
    if (!PInfo || ninputs <= 0 || noutputs <= 0) return false;

    for (int o = 0; o < noutputs; ++o) {
        if (PInfo[o].ON_minterms <= 0 || PInfo[o].OFF_minterms <= 0) {
            return false;
        }

        size_t on_cells = (size_t)PInfo[o].ON_minterms * (size_t)ninputs;
        size_t off_cells = (size_t)PInfo[o].OFF_minterms * (size_t)ninputs;

        for (size_t j = 0; j < on_cells; ++j) {
            if (PInfo[o].ON_set[j] < 1 || PInfo[o].ON_set[j] > 2) return false;
        }
        for (size_t j = 0; j < off_cells; ++j) {
            if (PInfo[o].OFF_set[j] < 1 || PInfo[o].OFF_set[j] > 2) return false;
        }
    }

    return true;
}

/*
 * Two ON rows are compatible when their agreement supercube covers no OFF
 * row.  If requested, agreement_count receives the supercube's level.
 */
static bool on_pair_compatible(
    const PIstorage *pi,
    int ninputs,
    int p,
    int q,
    int *agreement_count
) {
    const int *row_p = &pi->ON_set[(size_t)p * (size_t)ninputs];
    const int *row_q = &pi->ON_set[(size_t)q * (size_t)ninputs];
    int agreements = 0;

    for (int i = 0; i < ninputs; ++i) {
        if (row_p[i] == row_q[i]) agreements++;
    }
    if (agreement_count) *agreement_count = agreements;
    if (agreements == 0) return false;

    for (int z = 0; z < pi->OFF_minterms; ++z) {
        const int *off = &pi->OFF_set[(size_t)z * (size_t)ninputs];
        bool off_matches_supercube = true;

        for (int i = 0; i < ninputs; ++i) {
            if (row_p[i] == row_q[i] && off[i] != row_p[i]) {
                off_matches_supercube = false;
                break;
            }
        }

        if (off_matches_supercube) return false;
    }

    return true;
}

static int compatible_pair_agreement_horizon(
    const PIstorage *pi,
    int ninputs
) {
    int horizon = 0;

    for (int p = 0; p < pi->ON_minterms; ++p) {
        for (int q = p + 1; q < pi->ON_minterms; ++q) {
            int agreements = 0;
            bool compatible = on_pair_compatible(
                pi,
                ninputs,
                p,
                q,
                &agreements
            );
            if (compatible && agreements > horizon) horizon = agreements;
        }
    }

    return horizon;
}

/*
 * A pairwise-incompatible ON-row set lower-bounds every implicant cover.  A
 * greedy maximal set is sufficient; maximum cardinality is not required.
 */
static bool greedy_incompatibility_lower_bound(
    const PIstorage *pi,
    int ninputs,
    int *lower_bound
) {
    if (!lower_bound || pi->ON_minterms <= 0) return false;

    int *selected = calloc((size_t)pi->ON_minterms, sizeof(int));
    if (!selected) return false;

    int selected_count = 0;
    for (int p = 0; p < pi->ON_minterms; ++p) {
        bool incompatible_with_all = true;
        for (int j = 0; j < selected_count; ++j) {
            if (on_pair_compatible(pi, ninputs, p, selected[j], NULL)) {
                incompatible_with_all = false;
                break;
            }
        }
        if (incompatible_with_all) selected[selected_count++] = p;
    }

    free(selected);
    *lower_bound = selected_count;
    return true;
}

void certified_stop_state_reset(CertifiedStopState *state) {
    if (state) *state = (CertifiedStopState){0};
}

bool certified_stop_state_prepare(
    CertifiedStopState *state,
    const PIstorage *pi,
    int ninputs
) {
    if (!state || !pi || ninputs <= 0) return false;

    state->agreement_horizon = compatible_pair_agreement_horizon(pi, ninputs);
    if (!greedy_incompatibility_lower_bound(
        pi,
        ninputs,
        &state->cover_lower_bound
    )) {
        return false;
    }
    state->static_ready = true;
    return true;
}

bool certified_stop_state_init(
    CertifiedStopState *state,
    const PIstorage *pi,
    int ninputs
) {
    certified_stop_state_reset(state);
    return certified_stop_state_prepare(state, pi, ninputs);
}

void certified_stop_observe_coverage(
    CertifiedStopState *state,
    int level,
    bool on_set_covered
) {
    if (state && on_set_covered && state->coverage_horizon == 0) {
        state->coverage_horizon = level;
    }
}

int certified_stop_horizon(const CertifiedStopState *state) {
    if (!state) return 0;
    return state->agreement_horizon > state->coverage_horizon
        ? state->agreement_horizon
        : state->coverage_horizon;
}

bool certified_stop_should_stop(
    const CertifiedStopState *state,
    int level,
    int cover_size,
    bool boundary_exact
) {
    if (!state || !state->static_ready || cover_size <= 0) return false;

    if (cover_size == state->cover_lower_bound) return true;

    return (
        boundary_exact &&
        state->coverage_horizon > 0 &&
        level >= certified_stop_horizon(state)
    );
}

static bool pair_joint_in_retained_chart(
    const PIstorage *pi,
    int p,
    int q
) {
    for (int c = 0; c < pi->foundPI; ++c) {
        const size_t offset = (size_t)c * (size_t)pi->ON_minterms;
        if (pi->pichart[offset + (size_t)p] &&
            pi->pichart[offset + (size_t)q]) {
            return true;
        }
    }
    return false;
}

static double all_on_pair_model_bound(
    const PIstorage *pi,
    int ninputs,
    int level,
    bool *available
) {
    const uint64_t rows = (uint64_t)pi->ON_minterms;
    const uint64_t pair_count = rows * (rows - 1u) / 2u;
    const uint64_t pair_limit = 1000000u;

    /* Keep an explanatory statistic from dominating a stopping decision. */
    if (pair_count > pair_limit) {
        *available = false;
        return 0.0;
    }

    const double match_probability = ldexp(1.0, -level);
    const double blocked_probability = match_probability == 0.0
        ? 0.0
        : -expm1((double)pi->OFF_minterms * log1p(-match_probability));
    double bound = 0.0;

    *available = true;

    /*
     * This union bound ranges over every ON pair, a set fixed independently
     * of an auxiliary uniform, with-replacement random-OFF draw. Restricting
     * it after observing the selected cover would introduce selection bias
     * and would not justify the bound.
     */
    for (int p = 0; p < pi->ON_minterms; ++p) {
        const int *row_p = &pi->ON_set[(size_t)p * (size_t)ninputs];
        for (int q = p + 1; q < pi->ON_minterms; ++q) {
            const int *row_q = &pi->ON_set[(size_t)q * (size_t)ninputs];
            int agreements = 0;
            for (int i = 0; i < ninputs; ++i) {
                if (row_p[i] == row_q[i]) agreements++;
            }

            /* If a < k, a compatible pair's supercube already has level < k. */
            if (agreements >= level) {
                const int exponent = agreements / level;
                bound += pow(blocked_probability, (double)exponent);
                if (bound >= 1.0) return 1.0;
            }
        }
    }

    return bound;
}

bool certified_blocking_diagnostic(
    const PIstorage *pi,
    int ninputs,
    int level,
    const int *selected_indices,
    int selected_terms,
    bool include_model_metadata,
    BlockingDiagnostic *diagnostic
) {
    if (!pi || !diagnostic || !selected_indices || ninputs <= 0 ||
        level <= 0 || selected_terms <= 0 || pi->ON_minterms <= 0 ||
        pi->foundPI <= 0 || !pi->pichart) {
        return false;
    }

    *diagnostic = (BlockingDiagnostic){0};
    diagnostic->level = level;
    diagnostic->selected_terms = selected_terms;
    diagnostic->sparse_load = ldexp((double)pi->OFF_minterms, -level);
    if (include_model_metadata) {
        diagnostic->model_union_bound = all_on_pair_model_bound(
            pi,
            ninputs,
            level,
            &diagnostic->model_union_bound_available
        );
    }

    int *witness = malloc((size_t)selected_terms * sizeof(int));
    if (!witness) return false;
    for (int j = 0; j < selected_terms; ++j) witness[j] = -1;

    /* Find one private ON-row witness for each irredundant selected term. */
    for (int r = 0; r < pi->ON_minterms; ++r) {
        int owner = -1;
        int covering_terms = 0;
        for (int j = 0; j < selected_terms; ++j) {
            const int c = selected_indices[j];
            if (c < 0 || c >= pi->foundPI) {
                free(witness);
                return false;
            }
            if (pi->pichart[(size_t)c * (size_t)pi->ON_minterms + (size_t)r]) {
                owner = j;
                covering_terms++;
            }
        }
        if (covering_terms == 1 && witness[owner] < 0) witness[owner] = r;
    }

    for (int j = 0; j < selected_terms; ++j) {
        if (witness[j] >= 0) diagnostic->private_witnesses++;
    }

    for (int a = 0; a < selected_terms; ++a) {
        if (witness[a] < 0) continue;
        for (int b = a + 1; b < selected_terms; ++b) {
            if (witness[b] < 0) continue;
            diagnostic->private_pairs++;

            if (!on_pair_compatible(
                pi,
                ninputs,
                witness[a],
                witness[b],
                NULL
            )) {
                diagnostic->incompatible_private_pairs++;
            } else if (pair_joint_in_retained_chart(
                pi,
                witness[a],
                witness[b]
            )) {
                diagnostic->shallow_private_pairs++;
            } else {
                diagnostic->delayed_private_pairs++;
            }
        }
    }

    free(witness);
    return true;
}

void certified_blocking_diagnostic_print(
    FILE *stream,
    int output_index,
    bool boundary_exact,
    const BlockingDiagnostic *diagnostic
) {
    if (!stream || !diagnostic) return;

    fprintf(
        stream,
        "CCUBES_BLOCKING output=%d level=%d terms=%d witnesses=%d "
        "private_pairs=%" PRIu64 " incompatible=%" PRIu64
        " shallow=%" PRIu64 " delayed=%" PRIu64
        " sparse_load=%.17g model_union_bound=",
        output_index,
        diagnostic->level,
        diagnostic->selected_terms,
        diagnostic->private_witnesses,
        diagnostic->private_pairs,
        diagnostic->incompatible_private_pairs,
        diagnostic->shallow_private_pairs,
        diagnostic->delayed_private_pairs,
        diagnostic->sparse_load
    );
    if (diagnostic->model_union_bound_available) {
        fprintf(stream, "%.17g", diagnostic->model_union_bound);
    } else {
        fputs("NA", stream);
    }
    fprintf(stream, " boundary_exact=%d\n", boundary_exact ? 1 : 0);
}

void certified_blocking_state_init(BlockingStopState *state) {
    if (state) *state = (BlockingStopState){0};
}

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
) {
    if (!state) return false;
    if (!plateau_triggered || state->reported) {
        return true;
    }
    if (!adaptive_mode && !report_diagnostic) return true;

    BlockingDiagnostic diagnostic;
    if (!certified_blocking_diagnostic(
        pi,
        ninputs,
        level,
        selected_indices,
        selected_terms,
        report_diagnostic,
        &diagnostic
    )) {
        return false;
    }

    if (report_diagnostic) {
        certified_blocking_diagnostic_print(
            stream,
            output_index,
            boundary_exact,
            &diagnostic
        );
    }
    if (adaptive_mode) {
        state->certification_required =
            diagnostic.delayed_private_pairs > 0;
        if (state->certification_required &&
            !certified_stop_state_prepare(certificate, pi, ninputs)) {
            return false;
        }
        if (report_diagnostic) {
            fprintf(
                stream,
                "CCUBES_ADAPTIVE output=%d level=%d action=%s\n",
                output_index,
                level,
                state->certification_required ? "certify" : "plateau"
            );
        }
    }
    state->reported = true;
    return true;
}

bool certified_stop_policy_decision(
    const CertifiedStopState *certificate,
    const BlockingStopState *blocking,
    bool certified_mode,
    bool plateau_triggered,
    int level,
    int cover_size,
    bool boundary_exact
) {
    const bool escalated = blocking && blocking->certification_required;

    /* Every supported output has a nonempty ON set, so one term is optimal. */
    if (cover_size == 1) return true;

    if (certified_mode || escalated) {
        return certified_stop_should_stop(
            certificate,
            level,
            cover_size,
            boundary_exact
        );
    }

    return plateau_triggered;
}
