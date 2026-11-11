/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#ifndef CCUBES_CLI_H
#define CCUBES_CLI_H

#include <stdbool.h>
#include <stdio.h>

/* Paths borrow argv storage. Parsing never modifies argv or opens files. */
typedef struct {
    int bits_per_word;
    int threads;
    int weights;
    int solver;
    int effort;
    bool deterministic;
    bool certify;
    bool blocking_diagnostic;
    bool pool;
    bool lock_stats;
    double time_limit;
    double checkpoint_every;
    char *checkpoint;
    char *resume;
    char *inspect;
    int inspect_level;
    bool debug;
    bool debug_level_explicit;
    int debug_level;
    char *debug_file;
    char *log_file;
    char *source;
    char *destination;
    bool legacy_options;
    bool progressive_cover;
    const char *route;
    int literal_polish;
    bool literal_polish_explicit;
} CliOptions;

/* Returns 0 on success, 1 for help, -1 for an error (written to err). */
int cli_parse(int argc, char *const argv[], CliOptions *options, FILE *err);
void cli_help(FILE *out);

#endif
