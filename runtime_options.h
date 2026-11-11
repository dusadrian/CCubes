/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef CCUBES_RUNTIME_OPTIONS_H
#define CCUBES_RUNTIME_OPTIONS_H
#include <stdbool.h>
#include <stdio.h>

typedef enum {
#define PUBLIC_OPTION(key, ...) OPTION_##key,
#include "runtime_option_defs.h"
#undef PUBLIC_OPTION
    OPTION_COUNT
} RuntimeOption;

/* Read-only after startup parsing; NULL means use the established policy.
 * This interface never consults the process environment. */
const char *ccubes_option(RuntimeOption key);
void ccubes_options_reset(void);
/* 0 unknown, 1 parsed, -1 invalid. May consume the next argv value. */
int ccubes_options_parse(int argc, char *const argv[], int *index, FILE *err);
void ccubes_options_help(FILE *out);
bool ccubes_options_require_ipc(void);
#endif
