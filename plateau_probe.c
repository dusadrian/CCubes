/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "plateau_probe.h"

#include <limits.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static bool checked_product(size_t a, size_t b, size_t *result) {
    if (!result || (b != 0 && a > SIZE_MAX / b)) return false;
    *result = a * b;
    return true;
}

static bool grow_probe_storage(
    PIstorage *pi,
    int implicant_words,
    int required
) {
    if (required <= pi->estimPI) return true;

    int increase = pi->estimPI > 0 ? pi->estimPI / 2 : 64;
    if (increase < 64) increase = 64;
    if (pi->estimPI > INT_MAX - increase) return false;

    int new_capacity = pi->estimPI + increase;
    if (new_capacity < required) new_capacity = required;

    size_t chart_cells = 0;
    size_t implicant_cells = 0;
    if (
        !checked_product(
            (size_t)new_capacity,
            (size_t)pi->pichart_words,
            &chart_cells
        ) ||
        !checked_product(
            (size_t)new_capacity,
            (size_t)implicant_words,
            &implicant_cells
        )
    ) {
        return false;
    }

    uint64_t *new_chart = calloc(chart_cells, sizeof(uint64_t));
    uint64_t *new_pos = calloc(implicant_cells, sizeof(uint64_t));
    uint64_t *new_val = calloc(implicant_cells, sizeof(uint64_t));
    int *new_shared = calloc((size_t)new_capacity, sizeof(int));
    int *new_covsum = calloc((size_t)new_capacity, sizeof(int));
    int *new_covered = calloc((size_t)new_capacity, sizeof(int));
    if (
        !new_chart || !new_pos || !new_val ||
        !new_shared || !new_covsum || !new_covered
    ) {
        free(new_chart);
        free(new_pos);
        free(new_val);
        free(new_shared);
        free(new_covsum);
        free(new_covered);
        return false;
    }

    if (pi->foundPI > 0) {
        size_t old_chart_cells =
            (size_t)pi->foundPI * (size_t)pi->pichart_words;
        size_t old_implicant_cells =
            (size_t)pi->foundPI * (size_t)implicant_words;
        memcpy(
            new_chart,
            pi->pichart_pos,
            old_chart_cells * sizeof(uint64_t)
        );
        memcpy(
            new_pos,
            pi->implicants_pos,
            old_implicant_cells * sizeof(uint64_t)
        );
        memcpy(
            new_val,
            pi->implicants_val,
            old_implicant_cells * sizeof(uint64_t)
        );
        memcpy(
            new_shared,
            pi->shared,
            (size_t)pi->foundPI * sizeof(int)
        );
        memcpy(
            new_covsum,
            pi->covsum,
            (size_t)pi->foundPI * sizeof(int)
        );
        memcpy(
            new_covered,
            pi->covered,
            (size_t)pi->foundPI * sizeof(int)
        );
    }

    free(pi->pichart_pos);
    free(pi->implicants_pos);
    free(pi->implicants_val);
    free(pi->shared);
    free(pi->covsum);
    free(pi->covered);

    pi->pichart_pos = new_chart;
    pi->implicants_pos = new_pos;
    pi->implicants_val = new_val;
    pi->shared = new_shared;
    pi->covsum = new_covsum;
    pi->covered = new_covered;
    pi->estimPI = new_capacity;
    return true;
}

static bool row_matches_cube(
    const int *row,
    const int *cube,
    int ninputs
) {
    for (int i = 0; i < ninputs; ++i) {
        if (cube[i] >= 0 && row[i] != cube[i]) return false;
    }
    return true;
}

static bool cube_avoids_off(
    const PIstorage *pi,
    const int *cube,
    int ninputs
) {
    for (int z = 0; z < pi->OFF_minterms; ++z) {
        if (row_matches_cube(
            &pi->OFF_set[(size_t)z * (size_t)ninputs],
            cube,
            ninputs
        )) {
            return false;
        }
    }
    return true;
}

/*
 * Expand a repair seed without repeatedly matching every tentative cube
 * against the full OFF set.  For each row, mismatch_count is the number of
 * currently fixed literals that exclude that row.  A literal is removable
 * exactly when it is not the sole remaining blocker of any OFF row.
 *
 * Among safe removals, prefer the literal that immediately admits the most
 * ON rows.  The blocker-pressure tie-break preserves more future choices;
 * opposite coordinate orders provide deterministic diversity when both
 * scores tie.  This is CCubes's compact use of Espresso's general
 * blocking/covering guidance, not a port of Espresso's expansion machinery.
 */
static bool guide_repair_cube(
    const PIstorage *pi,
    int *cube,
    int ninputs,
    bool reverse_ties
) {
    int *off_mismatch = malloc(
        (size_t)pi->OFF_minterms * sizeof(int)
    );
    int *on_mismatch = malloc(
        (size_t)pi->ON_minterms * sizeof(int)
    );
    if (!off_mismatch || !on_mismatch) {
        free(off_mismatch);
        free(on_mismatch);
        return false;
    }

    for (int z = 0; z < pi->OFF_minterms; ++z) {
        const int *row =
            &pi->OFF_set[(size_t)z * (size_t)ninputs];
        int mismatches = 0;
        for (int i = 0; i < ninputs; ++i) {
            if (cube[i] >= 0 && row[i] != cube[i]) mismatches++;
        }
        if (mismatches == 0) {
            free(off_mismatch);
            free(on_mismatch);
            return false;
        }
        off_mismatch[z] = mismatches;
    }

    for (int r = 0; r < pi->ON_minterms; ++r) {
        const int *row =
            &pi->ON_set[(size_t)r * (size_t)ninputs];
        int mismatches = 0;
        for (int i = 0; i < ninputs; ++i) {
            if (cube[i] >= 0 && row[i] != cube[i]) mismatches++;
        }
        on_mismatch[r] = mismatches;
    }

    for (;;) {
        int best = -1;
        int best_gain = -1;
        int best_pressure = INT_MAX;

        for (int step = 0; step < ninputs; ++step) {
            int i = reverse_ties ? ninputs - 1 - step : step;
            if (cube[i] < 0) continue;

            bool removable = true;
            int blocker_pressure = 0;
            for (int z = 0; z < pi->OFF_minterms; ++z) {
                const int *row =
                    &pi->OFF_set[(size_t)z * (size_t)ninputs];
                if (row[i] == cube[i]) continue;
                if (off_mismatch[z] <= 1) {
                    removable = false;
                    break;
                }
                if (off_mismatch[z] == 2) blocker_pressure++;
            }
            if (!removable) continue;

            int coverage_gain = 0;
            for (int r = 0; r < pi->ON_minterms; ++r) {
                const int *row =
                    &pi->ON_set[(size_t)r * (size_t)ninputs];
                if (
                    on_mismatch[r] == 1 &&
                    row[i] != cube[i]
                ) {
                    coverage_gain++;
                }
            }

            if (
                coverage_gain > best_gain ||
                (
                    coverage_gain == best_gain &&
                    blocker_pressure < best_pressure
                )
            ) {
                best = i;
                best_gain = coverage_gain;
                best_pressure = blocker_pressure;
            }
        }

        if (best < 0) break;

        int removed_value = cube[best];
        cube[best] = -1;
        for (int z = 0; z < pi->OFF_minterms; ++z) {
            const int *row =
                &pi->OFF_set[(size_t)z * (size_t)ninputs];
            if (row[best] != removed_value) off_mismatch[z]--;
        }
        for (int r = 0; r < pi->ON_minterms; ++r) {
            const int *row =
                &pi->ON_set[(size_t)r * (size_t)ninputs];
            if (row[best] != removed_value) on_mismatch[r]--;
        }
    }

    free(off_mismatch);
    free(on_mismatch);
    return true;
}

static bool pair_has_retained_joint_cube(
    const PIstorage *pi,
    int p,
    int q,
    int retained_columns
) {
    const PIChartView chart = pi_chart_view(pi);
    for (int c = 0; c < retained_columns; ++c) {
        if (chart_covers(&chart, c, p) && chart_covers(&chart, c, q)) {
            return true;
        }
    }
    return false;
}

static int build_candidate_coverage(
    const PIstorage *pi,
    const int *cube,
    int ninputs,
    uint64_t *coverage
) {
    memset(
        coverage,
        0,
        (size_t)pi->pichart_words * sizeof(uint64_t)
    );

    int covered_rows = 0;
    for (int r = 0; r < pi->ON_minterms; ++r) {
        if (!row_matches_cube(
            &pi->ON_set[(size_t)r * (size_t)ninputs],
            cube,
            ninputs
        )) {
            continue;
        }

        coverage[pi->cov_word_index[r]] |= pi->shifted_cov_mask[r];
        covered_rows++;
    }
    return covered_rows;
}

static bool coverage_is_dominated(
    const PIstorage *pi,
    const uint64_t *candidate
) {
    for (int c = 0; c < pi->foundPI; ++c) {
        const uint64_t *retained = &pi->pichart_pos[
            (size_t)c * (size_t)pi->pichart_words
        ];
        bool contains_candidate = true;
        for (int w = 0; w < pi->pichart_words; ++w) {
            if ((retained[w] & candidate[w]) != candidate[w]) {
                contains_candidate = false;
                break;
            }
        }
        if (contains_candidate) return true;
    }
    return false;
}

static bool cube_geometry_exists(
    const PIstorage *pi,
    const uint64_t *position,
    const uint64_t *value,
    int implicant_words
) {
    for (int c = 0; c < pi->foundPI; ++c) {
        if (
            memcmp(
                &pi->implicants_pos[(size_t)c * (size_t)implicant_words],
                position,
                (size_t)implicant_words * sizeof(uint64_t)
            ) == 0 &&
            memcmp(
                &pi->implicants_val[(size_t)c * (size_t)implicant_words],
                value,
                (size_t)implicant_words * sizeof(uint64_t)
            ) == 0
        ) {
            return true;
        }
    }
    return false;
}

static bool append_cube(
    PIstorage *pi,
    const int *cube,
    int ninputs,
    int implicant_words,
    const int *bit_index,
    const int *word_index,
    const uint64_t *shifted_mask,
    uint64_t *coverage,
    int covered_rows,
    PlateauProbeStats *stats
) {
    uint64_t *position = calloc(
        (size_t)implicant_words,
        sizeof(uint64_t)
    );
    uint64_t *value = calloc(
        (size_t)implicant_words,
        sizeof(uint64_t)
    );
    if (!position || !value) {
        free(position);
        free(value);
        return false;
    }

    for (int i = 0; i < ninputs; ++i) {
        if (cube[i] < 0) continue;
        position[word_index[i]] |= shifted_mask[i];
        value[word_index[i]] |=
            ((uint64_t)(cube[i] - 1) << bit_index[i]);
    }

    stats->candidates_generated++;
    if (
        cube_geometry_exists(pi, position, value, implicant_words) ||
        coverage_is_dominated(pi, coverage)
    ) {
        free(position);
        free(value);
        return true;
    }

    if (pi->foundPI == INT_MAX) {
        free(position);
        free(value);
        return false;
    }
    if (!grow_probe_storage(pi, implicant_words, pi->foundPI + 1)) {
        free(position);
        free(value);
        return false;
    }

    int c = pi->foundPI;
    memcpy(
        &pi->pichart_pos[(size_t)c * (size_t)pi->pichart_words],
        coverage,
        (size_t)pi->pichart_words * sizeof(uint64_t)
    );
    memcpy(
        &pi->implicants_pos[(size_t)c * (size_t)implicant_words],
        position,
        (size_t)implicant_words * sizeof(uint64_t)
    );
    memcpy(
        &pi->implicants_val[(size_t)c * (size_t)implicant_words],
        value,
        (size_t)implicant_words * sizeof(uint64_t)
    );
    pi->shared[c] = 0;
    pi->covsum[c] = covered_rows;
    pi->covered[c] = 0;
    pi->foundPI++;
    stats->candidates_appended++;

    free(position);
    free(value);
    return true;
}

bool plateau_probe_append_candidates(
    PIstorage *pi,
    int ninputs,
    int implicant_words,
    const int *selected_indices,
    int selected_terms,
    const int *bit_index,
    const int *word_index,
    const uint64_t *shifted_mask,
    uint64_t pair_limit,
    int candidate_limit,
    PlateauProbeStats *stats
) {
    if (
        !pi || !selected_indices || !bit_index || !word_index ||
        !shifted_mask || !stats || ninputs <= 0 || implicant_words <= 0 ||
        selected_terms <= 1 || pair_limit == 0 || candidate_limit <= 0 ||
        pi->ON_minterms <= 0 || pi->OFF_minterms <= 0 ||
        pi->pichart_words <= 0 || pi->cov_bits <= 0 ||
        pi->foundPI < 0 || pi->estimPI < pi->foundPI ||
        !pi->ON_set || !pi->OFF_set || !pi->pichart_pos ||
        !pi->implicants_pos || !pi->implicants_val ||
        !pi->shared || !pi->covsum || !pi->covered ||
        !pi->cov_word_index || !pi->shifted_cov_mask
    ) {
        return false;
    }

    *stats = (PlateauProbeStats){0};
    const int retained_columns = pi->foundPI;
    int *witness = malloc((size_t)selected_terms * sizeof(int));
    int *base_cube = malloc((size_t)ninputs * sizeof(int));
    int *cube = malloc((size_t)ninputs * sizeof(int));
    uint64_t *coverage = calloc(
        (size_t)pi->pichart_words,
        sizeof(uint64_t)
    );
    if (!witness || !base_cube || !cube || !coverage) {
        free(witness);
        free(base_cube);
        free(cube);
        free(coverage);
        return false;
    }
    for (int j = 0; j < selected_terms; ++j) witness[j] = -1;

    const PIChartView chart = pi_chart_view(pi);
    for (int r = 0; r < pi->ON_minterms; ++r) {
        int owner = -1;
        int covering_terms = 0;
        for (int j = 0; j < selected_terms; ++j) {
            int c = selected_indices[j];
            if (c < 0 || c >= pi->foundPI) {
                free(witness);
                free(base_cube);
                free(cube);
                free(coverage);
                return false;
            }
            if (chart_covers(&chart, c, r)) {
                owner = j;
                covering_terms++;
            }
        }
        if (covering_terms == 1 && witness[owner] < 0) witness[owner] = r;
    }

    for (int j = 0; j < selected_terms; ++j) {
        if (witness[j] >= 0) stats->private_witnesses++;
    }

    bool ok = true;
    for (
        int a = 0;
        a < selected_terms &&
        stats->pairs_examined < pair_limit &&
        stats->candidates_appended < candidate_limit;
        ++a
    ) {
        if (witness[a] < 0) continue;
        for (
            int b = a + 1;
            b < selected_terms &&
            stats->pairs_examined < pair_limit &&
            stats->candidates_appended < candidate_limit;
            ++b
        ) {
            if (witness[b] < 0) continue;
            stats->pairs_examined++;

            int p = witness[a];
            int q = witness[b];
            if (pair_has_retained_joint_cube(
                pi,
                p,
                q,
                retained_columns
            )) {
                continue;
            }

            const int *row_p =
                &pi->ON_set[(size_t)p * (size_t)ninputs];
            const int *row_q =
                &pi->ON_set[(size_t)q * (size_t)ninputs];
            int fixed = 0;
            for (int i = 0; i < ninputs; ++i) {
                if (row_p[i] == row_q[i]) {
                    base_cube[i] = row_p[i];
                    fixed++;
                } else {
                    base_cube[i] = -1;
                }
            }
            if (fixed == 0 || !cube_avoids_off(pi, base_cube, ninputs)) {
                continue;
            }
            stats->compatible_pairs++;

            for (
                int order = 0;
                order < 2 &&
                stats->candidates_appended < candidate_limit;
                ++order
            ) {
                memcpy(cube, base_cube, (size_t)ninputs * sizeof(int));
                if (!guide_repair_cube(
                    pi,
                    cube,
                    ninputs,
                    order != 0
                )) {
                    ok = false;
                    break;
                }
                /*
                 * Keep an independent exact check at the module boundary.
                 * The mismatch accounting above decides every deletion, but
                 * appended candidates must never rely on heuristic scoring
                 * for semantic safety.
                 */
                if (!cube_avoids_off(pi, cube, ninputs)) {
                    ok = false;
                    break;
                }
                int covered_rows = build_candidate_coverage(
                    pi,
                    cube,
                    ninputs,
                    coverage
                );
                if (covered_rows <= 0) {
                    ok = false;
                    break;
                }
                if (!append_cube(
                    pi,
                    cube,
                    ninputs,
                    implicant_words,
                    bit_index,
                    word_index,
                    shifted_mask,
                    coverage,
                    covered_rows,
                    stats
                )) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
    }

    free(witness);
    free(base_cube);
    free(cube);
    free(coverage);
    return ok;
}
