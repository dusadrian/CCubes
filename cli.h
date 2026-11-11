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

typedef enum {
    PI_GENERATOR_AUTO = 0,
    PI_GENERATOR_PROJECTION,
    PI_GENERATOR_MMCS,
    PI_GENERATOR_COMPLETE_MMCS
} PIGeneratorMode;

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
    int debug_level;
    char *debug_file;
    char *source;
    char *destination;
    bool legacy_options;
#ifdef CCUBES_TESTING
    PIGeneratorMode test_pi_generator;
    bool test_projection_rwls;
#endif
} CliOptions;

/* Returns 0 on success, 1 for help, -1 for an error (written to err). */
int cli_parse(int argc, char *const argv[], CliOptions *options, FILE *err);
void cli_help(FILE *out);

#endif
