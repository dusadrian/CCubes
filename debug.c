/*
    Copyright (c) 2016–2025, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "debug.h"

int debug_enabled = 0;
int debug_level   = DBG_ERROR;
FILE *debug_out   = NULL;

void debug_init(const char *filename, int level) {
    debug_enabled = 1;
    debug_level   = level;

    if (filename && filename[0] != '\0') {
        debug_out = fopen(filename, "w");
        if (!debug_out) {
            debug_out = stderr; // fallback
            fprintf(stderr, "Could not open debug file %s, using stderr\n", filename);
        }
    } else {
        debug_out = stderr;
    }
}

void debug_close(void) {
    if (debug_out && debug_out != stderr) {
        fclose(debug_out);
    }
    debug_out = NULL;
    debug_enabled = 0;
}

