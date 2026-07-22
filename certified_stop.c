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
#include <string.h>

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

static uint64_t capped_binomial(int n, int k, uint64_t cap) {
    if (n < 0 || k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k;

    uint64_t value = 1;
    for (int i = 1; i <= k; ++i) {
        const uint64_t numerator = (uint64_t)(n - k + i);
#if defined(__SIZEOF_INT128__)
        __uint128_t next = (__uint128_t)value * numerator / (uint64_t)i;
        if (next > cap) return cap;
        value = (uint64_t)next;
#else
        /* Exact divisibility lets us reduce before multiplying. */
        uint64_t divisor = (uint64_t)i;
        uint64_t a = value;
        uint64_t b = numerator;
        uint64_t x = a;
        uint64_t y = divisor;
        while (y != 0) {
            uint64_t remainder = x % y;
            x = y;
            y = remainder;
        }
        a /= x;
        divisor /= x;
        x = b;
        y = divisor;
        while (y != 0) {
            uint64_t remainder = x % y;
            x = y;
            y = remainder;
        }
        b /= x;
        divisor /= x;
        if (divisor != 1 || (a != 0 && b > cap / a)) return cap;
        value = a * b;
        if (value > cap) return cap;
#endif
    }
    return value;
}

bool certified_stop_adaptive_work_within_limit(
    int ninputs,
    int level,
    int horizon,
    uint64_t limit,
    uint64_t *estimated_tasks
) {
    if (!estimated_tasks || ninputs < 0 || level < 0 || horizon < 0) {
        return false;
    }

    const uint64_t cap = limit == UINT64_MAX ? UINT64_MAX : limit + 1u;
    uint64_t total = 0;
    int final_level = horizon < ninputs ? horizon : ninputs;
    for (int k = level + 1; k <= final_level; ++k) {
        uint64_t remaining_cap = cap - total;
        uint64_t tasks = capped_binomial(ninputs, k, remaining_cap);
        if (tasks >= remaining_cap) {
            total = cap;
            break;
        }
        total += tasks;
    }

    *estimated_tasks = total;
    return total <= limit;
}

static bool same_cover_set(const int *left, const int *right, int terms) {
    if (!left || !right || terms <= 0) return false;
    for (int i = 0; i < terms; ++i) {
        bool found = false;
        for (int j = 0; j < terms; ++j) {
            if (left[i] == right[j]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

/*
 * The blocking warning belongs to a cover, not merely to its cardinality.
 * Hybrid and exact solvers may return different tied covers.  Before an
 * adaptive escalation, inspect every retained equal-cardinality alternative
 * and prefer a warning-free representative when one exists.
 */
static bool prefer_warning_free_pool_cover(
    const PIstorage *pi,
    int ninputs,
    int level,
    int *selected_indices,
    int selected_terms,
    BlockingDiagnostic *selected_diagnostic,
    int *checked,
    int *warning_free,
    bool *replaced
) {
    if (!pi || !selected_indices || !selected_diagnostic || !checked ||
        !warning_free || !replaced) {
        return false;
    }

    *checked = 1;
    *warning_free = selected_diagnostic->delayed_private_pairs == 0 ? 1 : 0;
    *replaced = false;
    const int *best = selected_indices;
    uint64_t best_delayed = selected_diagnostic->delayed_private_pairs;

    for (int p = 0; p < pi->pool_count; ++p) {
        const int *candidate = pi->pool_solutions ? pi->pool_solutions[p] : NULL;
        if (!candidate || same_cover_set(candidate, selected_indices, selected_terms)) {
            continue;
        }

        BlockingDiagnostic diagnostic;
        if (!certified_blocking_diagnostic(
            pi,
            ninputs,
            level,
            candidate,
            selected_terms,
            false,
            &diagnostic
        )) {
            return false;
        }
        (*checked)++;
        if (diagnostic.delayed_private_pairs == 0) (*warning_free)++;
        if (diagnostic.delayed_private_pairs < best_delayed) {
            best = candidate;
            best_delayed = diagnostic.delayed_private_pairs;
            *selected_diagnostic = diagnostic;
        }
    }

    if (pi->prevsolmin == selected_terms && pi->previndices &&
        !same_cover_set(pi->previndices, selected_indices, selected_terms)) {
        BlockingDiagnostic diagnostic;
        if (!certified_blocking_diagnostic(
            pi,
            ninputs,
            level,
            pi->previndices,
            selected_terms,
            false,
            &diagnostic
        )) {
            return false;
        }
        (*checked)++;
        if (diagnostic.delayed_private_pairs == 0) (*warning_free)++;
        if (diagnostic.delayed_private_pairs < best_delayed) {
            best = pi->previndices;
            best_delayed = diagnostic.delayed_private_pairs;
            *selected_diagnostic = diagnostic;
        }
    }

    if (best != selected_indices) {
        memcpy(
            selected_indices,
            best,
            (size_t)selected_terms * sizeof(*selected_indices)
        );
        *replaced = true;
    }
    return true;
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
    int *selected_indices,
    int selected_terms,
    bool boundary_exact,
    bool inspect_equal_pool
) {
    if (!state) return false;
    if (!plateau_triggered || state->reported) {
        return true;
    }
    if (!adaptive_mode && !report_diagnostic) return true;
    FILE *output = stream ? stream : stderr;

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

    const uint64_t delayed_before = diagnostic.delayed_private_pairs;
    int pool_checked = 1;
    int warning_free = delayed_before == 0 ? 1 : 0;
    bool pool_replaced = false;
    if (adaptive_mode && inspect_equal_pool && pi->pool_count > 0 &&
        !prefer_warning_free_pool_cover(
            pi,
            ninputs,
            level,
            selected_indices,
            selected_terms,
            &diagnostic,
            &pool_checked,
            &warning_free,
            &pool_replaced
        )) {
        return false;
    }
    if (pool_replaced && report_diagnostic && !certified_blocking_diagnostic(
        pi,
        ninputs,
        level,
        selected_indices,
        selected_terms,
        true,
        &diagnostic
    )) {
        return false;
    }

    state->warning_detected = delayed_before > 0;
    state->pool_warning_avoided =
        delayed_before > 0 && diagnostic.delayed_private_pairs == 0;

    if (report_diagnostic || pool_replaced || delayed_before > 0) {
        if (inspect_equal_pool && pi->pool_count > 0) {
            fprintf(
                output,
                "CCUBES_ADAPTIVE_POOL output=%d level=%d checked=%d "
                "warning_free=%d replaced=%d delayed_before=%" PRIu64
                " delayed_after=%" PRIu64 "\n",
                output_index,
                level,
                pool_checked,
                warning_free,
                pool_replaced ? 1 : 0,
                delayed_before,
                diagnostic.delayed_private_pairs
            );
        }
    }

    if (report_diagnostic) {
        certified_blocking_diagnostic_print(
            output,
            output_index,
            boundary_exact,
            &diagnostic
        );
    }
    if (adaptive_mode) {
        const bool unresolved_warning = diagnostic.delayed_private_pairs > 0;
        if (unresolved_warning) {
            if (!certified_stop_state_prepare(certificate, pi, ninputs)) {
                return false;
            }
            state->certification_horizon = certified_stop_horizon(certificate);
            state->certification_required = certified_stop_adaptive_work_within_limit(
                ninputs,
                level,
                state->certification_horizon,
                CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT,
                &state->estimated_remaining_tasks
            );
            state->task_estimate_capped =
                !state->certification_required &&
                state->estimated_remaining_tasks >
                    CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT;
            state->escalation_suppressed = !state->certification_required;
        }
        if (report_diagnostic || unresolved_warning) {
            fprintf(
                output,
                "CCUBES_ADAPTIVE output=%d level=%d action=%s horizon=%d "
                "remaining_position_tasks=%" PRIu64 " estimate_capped=%d "
                "limit=%" PRIu64 "\n",
                output_index,
                level,
                state->certification_required
                    ? "certify"
                    : unresolved_warning ? "warn-stop" : "plateau",
                state->certification_horizon,
                state->estimated_remaining_tasks,
                state->task_estimate_capped ? 1 : 0,
                (uint64_t)CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT
            );
        }
    }
    state->reported = true;
    return true;
}

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
) {
    const bool escalated = blocking && blocking->certification_required;

    /* Every supported output has a nonempty ON set, so one term is optimal. */
    if (cover_size == 1) return true;

    if (certified_mode) {
        return certified_stop_should_stop(
            certificate,
            level,
            cover_size,
            boundary_exact
        );
    }

    if (escalated) {
        if (certified_stop_should_stop(
            certificate,
            level,
            cover_size,
            boundary_exact
        )) {
            return true;
        }

        /*
         * Adaptive mode never searches beyond the horizon whose work it
         * budgeted.  If the hybrid boundary has not closed its gap there,
         * preserve the best cover and stop with an explicit warning.  -c is
         * handled above and intentionally remains unbounded by this guard.
         */
        if (blocking->certification_horizon > 0 &&
            level >= blocking->certification_horizon) {
            blocking->certification_required = false;
            blocking->escalation_suppressed = true;
            blocking->boundary_horizon_unproved = true;
            fprintf(
                stream ? stream : stderr,
                "CCUBES_ADAPTIVE output=%d level=%d action=warn-stop "
                "horizon=%d remaining_position_tasks=0 estimate_capped=0 "
                "limit=%" PRIu64 " reason=boundary-not-proved\n",
                output_index,
                level,
                blocking->certification_horizon,
                (uint64_t)CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT
            );
            return true;
        }
        return false;
    }

    return plateau_triggered;
}
