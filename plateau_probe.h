/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PLATEAU_PROBE_H
#define PLATEAU_PROBE_H

#include <stdbool.h>
#include <stdint.h>

#include "utils.h"

typedef struct {
    int private_witnesses;
    uint64_t pairs_examined;
    uint64_t compatible_pairs;
    int candidates_generated;
    int candidates_appended;
} PlateauProbeStats;

/*
 * At a terminating cover-size plateau, try a bounded set of deeper prime
 * candidates built from pairs of private incumbent witnesses.
 *
 * This is a CCubes-specific plateau escape probe inspired by Espresso's
 * general lessons that a stalled cover can merit one focused repair and that
 * expansion choices can use separate OFF-blocking and ON-covering evidence.
 * It does not port Espresso's last-gasp implementation or data model.
 * The caller owns the stopping policy: appending no candidate, or finding no
 * smaller cover after appending candidates, is never a proof of optimality.
 */
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
);

#endif
