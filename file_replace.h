/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef CCUBES_FILE_REPLACE_H
#define CCUBES_FILE_REPLACE_H

#include <stdbool.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Publish a closed sibling temporary file over an existing destination. */
static inline bool ccubes_replace_file(const char *temporary, const char *destination) {
#ifdef _WIN32
    return MoveFileExA(temporary, destination,
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(temporary, destination) == 0;
#endif
}

#endif
