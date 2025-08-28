/*
 * Copyright (c) 2016–2025, Adrian Dusa
 * All rights reserved.
 *
 * License: Academic Non-Commercial License (see LICENSE file for details).
 * SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
 */

#include "scp.h"

// Simple greedy set cover solver as fallback
void solve_scp_greedy(
    int pichart[],
    const int foundPI,
    const int ON_minterms,
    double weights[],
    int *solution,
    int *solmin
) {
    // Input validation
    if (foundPI <= 0 || ON_minterms <= 0 || !pichart || !solution || !solmin) {
        if (solmin) *solmin = 0;
        return;
    }

    bool *covered = calloc(ON_minterms, sizeof(bool));
    bool *selected = calloc(foundPI, sizeof(bool));
    int solution_count = 0;

    if (!covered || !selected) {
        if (covered) free(covered);
        if (selected) free(selected);
        *solmin = 0;
        return;
    }

    while (solution_count < ON_minterms) { // prevent infinite loop
        // Check if all minterms are covered
        bool all_covered = true;
        for (int i = 0; i < ON_minterms; i++) {
            if (!covered[i]) {
                all_covered = false;
                break;
            }
        }

        if (all_covered) break;

        // Find the best PI to add (most uncovered minterms)
        int best_pi = -1;
        double best_score = -1.0;

        for (int j = 0; j < foundPI; j++) {
            if (selected[j]) continue;

            int new_coverage = 0;
            for (int i = 0; i < ON_minterms; i++) {
                if (!covered[i] && pichart[i + ON_minterms * j] == 1) {
                    new_coverage++;
                }
            }

            if (new_coverage > 0) {
                double score = (double)new_coverage;
                if (weights) {
                    score += weights[j];  // simple additive score
                }

                if (score > best_score) {
                    best_score = score;
                    best_pi = j;
                }
            }
        }

        if (best_pi == -1) break; // No more useful PIs

        // Select this PI
        selected[best_pi] = true;
        if (solution_count < ON_minterms) { // bounds check
            solution[solution_count++] = best_pi;
        }

        // Update coverage
        for (int i = 0; i < ON_minterms; i++) {
            if (pichart[i + ON_minterms * best_pi] == 1) {
                covered[i] = true;
            }
        }
    }

    *solmin = solution_count;
    free(covered);
    free(selected);
}

