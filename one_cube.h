/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef CCUBES_ONE_CUBE_H
#define CCUBES_ONE_CUBE_H

#include "utils.h"

/* Find a prime cube covering every ON point of a nonconstant binary output.
 * Write 0 for free inputs and 1/2 for fixed values. Return the cube's width,
 * zero if no such cube exists, or -1 on allocation failure. */
int find_one_cube(const PIstorage *pi, int ninputs, int *cube);

#endif
