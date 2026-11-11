/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef PROGRESSIVE_INPUT_H
#define PROGRESSIVE_INPUT_H

#include "utils.h"

typedef enum {
    PROGRESSIVE_INPUT_OK = 0,
    PROGRESSIVE_INPUT_INVALID,
    PROGRESSIVE_INPUT_OUT_OF_MEMORY
} ProgressiveInputStatus;

typedef struct {
    PIstorage pi;
    int on_original;
    int off_original;
    int on_duplicates;
    int off_duplicates;
    int conflicts;
} ProgressiveNormalizedInput;

/*
 * Build an owned, duplicate-free binary point-row view for one output.
 * First occurrence order is retained so canonical anchor ownership remains
 * deterministic. Empty ON or OFF sets are permitted for constant outputs.
 */
ProgressiveInputStatus progressive_input_normalize(
    const PIstorage *source,
    int ninputs,
    ProgressiveNormalizedInput *normalized
);

void progressive_input_destroy(ProgressiveNormalizedInput *normalized);

#endif
