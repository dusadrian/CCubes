/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "cli.h"
#include "runtime_options.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    BITS, THREADS, WEIGHTS, SOLVER, EFFORT, DETERMINISTIC, CERTIFY,
    CERTIFY_SYSTEMATIC, CERTIFICATION_TIME_LIMIT,
    /* The contiguous block below stays accepted during CLI migration. */
    SYSTEMATIC_SAT_SECONDS, SYSTEMATIC_SECONDS, SYSTEMATIC_WORK,
    SYSTEMATIC_WEIGHTED_SECONDS, SYSTEMATIC_WEIGHTED_WORK, SYSTEMATIC_NATIVE_SAT_SECONDS,
    SYSTEMATIC_VERIFIER_SECONDS, SYSTEMATIC_VERIFIER_WORK,
    SYSTEMATIC_STRENGTHENING, SYSTEMATIC_DESCENT,
    POOL, LOCK_STATS, TIME_LIMIT, CHECKPOINT_EVERY, CHECKPOINT,
    RESUME, INSPECT, INSPECT_LEVEL, DEBUG, DEBUG_FILE, LOG_FILE, HELP,
    PROGRESSIVE_COVER, ROUTE, LITERAL_POLISH,

} OptionId;

typedef struct {
    const char *name;
    const char *legacy;
    bool takes_value;
    OptionId id;
    const char *description;
} Option;

/* Parsing and help share one registry, with migration options hidden from help. */
static const Option option_table[] = {
    {"bits-per-word", "b", true, BITS, "8, 16, 32, or 64 (default). Legacy 128 uses 64"},
    {"threads", "t", true, THREADS, "CPU worker threads; 0 (default) uses available cores"},
    {"weights", "w", true, WEIGHTS, "none, complexity (default), or sharing"},
    {"solver", "s", true, SOLVER, "hybrid (default) or gurobi"},
    {"effort", "e", true, EFFORT, "hybrid effort: 0 (default), 1, or 2"},
    {"deterministic", "d", false, DETERMINISTIC, "canonical PI ordering"},
    {"certify", "c", false, CERTIFY, "complete-prime enumeration and exact cover (point rows only)"},
    {"certify-systematic", NULL, false, CERTIFY_SYSTEMATIC, "certify ordinary scalar covers with IPC pricing, weighted integer search and checked native SAT"},
    {"certification-time-limit", NULL, true, CERTIFICATION_TIME_LIMIT, "positive seconds per output for systematic certification, after minimization"},
    {"systematic-weighted-seconds", NULL, true, SYSTEMATIC_WEIGHTED_SECONDS, "weighted search and bound checking seconds per output (default 10; 0 skips)"},
    {"systematic-weighted-work", NULL, true, SYSTEMATIC_WEIGHTED_WORK, "weighted search and bound checking work per output (default 0: unlimited)"},
    {"systematic-verifier-seconds", NULL, true, SYSTEMATIC_VERIFIER_SECONDS, "separate weighted bound-checking time per output (positive seconds; default 1 when enabled)"},
    {"systematic-verifier-work", NULL, true, SYSTEMATIC_VERIFIER_WORK, "enable a separate weighted bound-checking work limit per output (0: unlimited)"},
    {"systematic-native-sat-seconds", NULL, true, SYSTEMATIC_NATIVE_SAT_SECONDS, "native SAT and checking seconds per output (default 2; 0 skips)"},
    {"systematic-certification-seconds", NULL, true, SYSTEMATIC_SECONDS, "legacy shared seconds per output, including IPC pricing (0: cardinality floor only)"},
    {"systematic-certification-work", NULL, true, SYSTEMATIC_WORK, "native certification work per output, including proof checking (0: unlimited)"},
    {"systematic-strengthening", NULL, true, SYSTEMATIC_STRENGTHENING, "experimental cover SAT strengthening: none (default), groups, ordering, or combined"},
    {"systematic-descent", NULL, true, SYSTEMATIC_DESCENT, "target descent: fresh (default) or experimental reuse"},
    {"systematic-sat-seconds", NULL, true, SYSTEMATIC_SAT_SECONDS, "legacy seconds per output after initial IPC pricing (0: initial pricing only)"},
    {"pool", "p", false, POOL, "automatic equal-cardinality cover pooling; implies sharing weights"},
    {"lock-stats", NULL, false, LOCK_STATS, "report PI-insertion lock contention"},
    {"time-limit", "l", true, TIME_LIMIT, "nonnegative seconds until checkpoint and exit; 0 disables"},
    {"checkpoint-every", NULL, true, CHECKPOINT_EVERY, "nonnegative seconds between checkpoints; 0 disables"},
    {"checkpoint", NULL, true, CHECKPOINT, "checkpoint save path (otherwise derived from source/resume)"},
    {"resume", "r", true, RESUME, "resume from this checkpoint file"},
    {"inspect", "i", true, INSPECT, "inspect this checkpoint file and exit"},
    {"inspect-level", NULL, true, INSPECT_LEVEL, "0 (default) progress, 1 full metadata"},
    {"debug", "dbg", true, DEBUG, "0 warnings, 1 info, 2 trace; logging is off unless requested"},
    {"debug-file", NULL, true, DEBUG_FILE, "write debug logging to this file (otherwise stderr)"},
    {"log-file", NULL, true, LOG_FILE, "append run diagnostics and solver logs for any backend; defaults to debug level 1"},
    {"help", "h", false, HELP, "show this help message"},
    {"progressive-cover", NULL, false, PROGRESSIVE_COVER, "select IPC (same as --route ipc)"},
    {"route", NULL, true, ROUTE, "ordinary (default, legacy: e0), auto, or ipc"},
    {"literal-polish", NULL, true, LITERAL_POLISH, "IPC finishing: 0 off, 1 one pass (default), 2 yield-gated, 3 four passes"},
};

void cli_help(FILE *out) {
    fprintf(out, "Usage: ccubes [options] source.pla [dest.pla]\n"
                 "       ccubes --resume FILE [options]\n"
                 "       ccubes --inspect FILE [--inspect-level 0|1]\n"
                 "Options (values accept --name VALUE or --name=VALUE):\n");
    for (size_t i = 0; i < sizeof(option_table) / sizeof(option_table[0]); ++i) {
        const Option *option = &option_table[i];
        if (option->id >= SYSTEMATIC_SAT_SECONDS && option->id <= SYSTEMATIC_DESCENT) continue;
        fprintf(out, "  --%s%s\n      %s\n", option->name,
                option->takes_value ? " VALUE" : "", option->description);
    }
    ccubes_options_help(out);
    fprintf(out, "Use -- to end option parsing for filenames beginning with '-'.\n"
                 "Legacy short options are deprecated; use the long names above.\n"
                 "No CCUBES_* environment settings are read.\n");
}

static bool parse_integer(const char *text, int *value) {
    if (!text || !isdigit((unsigned char)*text)) return false;
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || *end || parsed < 0 || parsed > INT_MAX) return false;
    *value = (int)parsed;
    return true;
}

static bool parse_seconds(const char *text, double *value) {
    if (!text || (!isdigit((unsigned char)*text) && *text != '.')) return false;
    char *end = NULL;
    errno = 0;
    double parsed = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(parsed) || parsed < 0) return false;
    *value = parsed;
    return true;
}

static bool apply_option(OptionId id, char *value, CliOptions *o) {
    int number = 0;
    switch (id) {
        case BITS:
            if (!parse_integer(value, &number) ||
                (number != 8 && number != 16 && number != 32 && number != 64 && number != 128)) return false;
            o->bits_per_word = number;
            return true;
        case THREADS: return parse_integer(value, &o->threads);
        case WEIGHTS:
            if (!strcmp(value, "none")) o->weights = 0;
            else if (!strcmp(value, "complexity")) o->weights = 1;
            else if (!strcmp(value, "sharing")) o->weights = 2;
            else if (!parse_integer(value, &o->weights) || o->weights > 2) return false;
            return true;
        case SOLVER:
            if (!strcmp(value, "hybrid")) o->solver = 0;
            else if (!strcmp(value, "gurobi")) o->solver = 1;
            else if (!parse_integer(value, &o->solver) || o->solver > 1) return false;
            return true;
        case EFFORT: return parse_integer(value, &o->effort) && o->effort <= 2;
        case DETERMINISTIC: o->deterministic = true; return true;
        case CERTIFY: o->certify = true; return true;
        case CERTIFY_SYSTEMATIC: o->certify_systematic = true; return true;
        case CERTIFICATION_TIME_LIMIT:
        case SYSTEMATIC_SAT_SECONDS:
        case SYSTEMATIC_SECONDS: {
            SystematicBudgetMode mode = SYSTEMATIC_TOTAL_BUDGET;
            if (id == SYSTEMATIC_SECONDS) mode = SYSTEMATIC_SHARED_BUDGET;
            if (id == SYSTEMATIC_SAT_SECONDS) mode = SYSTEMATIC_POST_PRICING_BUDGET;
            if (o->systematic_budget.mode != SYSTEMATIC_STAGE_BUDGETS &&
                o->systematic_budget.mode != mode) {
                return false;
            }
            o->systematic_budget.mode = mode;
            return parse_seconds(value, &o->systematic_budget.shared_seconds) &&
                (mode != SYSTEMATIC_TOTAL_BUDGET || o->systematic_budget.shared_seconds > 0.0);
        }
        case SYSTEMATIC_WEIGHTED_SECONDS:
            o->systematic_stage_limits_explicit = true;
            return parse_seconds(value, &o->systematic_budget.weighted_seconds);
        case SYSTEMATIC_NATIVE_SAT_SECONDS:
            o->systematic_stage_limits_explicit = true;
            return parse_seconds(value, &o->systematic_budget.sat_seconds);
        case SYSTEMATIC_VERIFIER_SECONDS:
            o->systematic_stage_limits_explicit = true;
            o->systematic_budget.separate_verifier = true;
            return parse_seconds(value, &o->systematic_budget.verifier_seconds) &&
                o->systematic_budget.verifier_seconds > 0.0;
        case SYSTEMATIC_WEIGHTED_WORK:
        case SYSTEMATIC_VERIFIER_WORK:
        case SYSTEMATIC_WORK: {
            if (!value || !isdigit((unsigned char)*value)) return false;
            char *end;
            errno = 0;
            uintmax_t work = strtoumax(value, &end, 10);
            if (errno || *end || work > UINT64_MAX) return false;
            if (id == SYSTEMATIC_WEIGHTED_WORK) {
                o->systematic_budget.weighted_work = (uint64_t)work;
                o->systematic_stage_limits_explicit = true;
            } else if (id == SYSTEMATIC_VERIFIER_WORK) {
                o->systematic_budget.verifier_work = (uint64_t)work;
                o->systematic_budget.separate_verifier = true;
                o->systematic_stage_limits_explicit = true;
            } else {
                o->systematic_budget.work = (uint64_t)work;
                o->systematic_work_explicit = true;
            }
            return true;
        }
        case SYSTEMATIC_STRENGTHENING:
            if (!strcmp(value, "none")) o->systematic_strengthening = NSAT_COVER_BASELINE;
            else if (!strcmp(value, "groups")) o->systematic_strengthening = NSAT_COVER_GROUPS;
            else if (!strcmp(value, "ordering")) o->systematic_strengthening = NSAT_COVER_ORDERING;
            else if (!strcmp(value, "combined")) o->systematic_strengthening = NSAT_COVER_COMBINED;
            else return false;
            o->systematic_strengthening_explicit = true;
            return true;
        case SYSTEMATIC_DESCENT:
            if (!strcmp(value, "fresh")) o->systematic_reuse = false;
            else if (!strcmp(value, "reuse")) o->systematic_reuse = true;
            else return false;
            o->systematic_descent_explicit = true;
            return true;
        case POOL: o->pool = true; return true;
        case LOCK_STATS: o->lock_stats = true; return true;
        case TIME_LIMIT: return parse_seconds(value, &o->time_limit);
        case CHECKPOINT_EVERY: return parse_seconds(value, &o->checkpoint_every);
        case CHECKPOINT: o->checkpoint = value; return true;
        case RESUME: o->resume = value; return true;
        case INSPECT: o->inspect = value; return true;
        case INSPECT_LEVEL: return parse_integer(value, &o->inspect_level) && o->inspect_level <= 1;
        case DEBUG:
            o->debug = true;
            o->debug_level_explicit = true;
            return parse_integer(value, &o->debug_level) && o->debug_level <= 2;
        case DEBUG_FILE: o->debug = true; o->debug_file = value; return true;
        case LOG_FILE: o->log_file = value; return true;
        case HELP: return true;
        case PROGRESSIVE_COVER: o->progressive_cover = true; return true;
        case ROUTE:
            if (strcmp(value, "auto") && strcmp(value, "ordinary") && strcmp(value, "e0") && strcmp(value, "ipc")) return false;
            o->route = !strcmp(value, "ordinary") ? "e0" : value;
            return true;
        case LITERAL_POLISH:
            o->literal_polish_explicit = true;
            return parse_integer(value, &o->literal_polish) && o->literal_polish <= 3;
    }
    return false;
}

int cli_parse(int argc, char *const argv[], CliOptions *o, FILE *err) {
    *o = (CliOptions){.bits_per_word = 64, .weights = 1, .literal_polish = 1,
                     .systematic_budget = {.weighted_seconds = 10.0, .sat_seconds = 2.0,
                                           .verifier_seconds = 1.0}};
    ccubes_options_reset();
    bool positional_only = false;
    bool inspect_level_set = false;
    for (int i = 1; i < argc; ++i) {
        char *arg = argv[i];
        if (!positional_only && !strcmp(arg, "--")) {
            positional_only = true;
            continue;
        }
        if (positional_only || arg[0] != '-') {
            if (!o->source) o->source = arg;
            else if (!o->destination) o->destination = arg;
            else {
                fprintf(err, "Unexpected positional argument: %s\n", arg);
                return -1;
            }
            continue;
        }

        int extra = ccubes_options_parse(argc, argv, &i, err);
        if (extra < 0) return -1;
        if (extra > 0) continue;
        const Option *option = NULL;
        char *value = NULL;
        bool legacy = arg[1] != '-';
        if (!legacy) {
            char *equals = strchr(arg + 2, '=');
            size_t length = equals ? (size_t)(equals - arg - 2) : strlen(arg + 2);
            for (size_t j = 0; j < sizeof(option_table) / sizeof(option_table[0]); ++j) {
                if (strlen(option_table[j].name) == length &&
                    !strncmp(arg + 2, option_table[j].name, length)) {
                    option = &option_table[j];
                    break;
                }
            }
            if (equals) value = equals + 1;
        } else {
            /* Longest legacy prefix first: -debug and -dbg precede -d. */
            size_t matched = 0;
            for (size_t j = 0; j < sizeof(option_table) / sizeof(option_table[0]); ++j) {
                const char *alias = option_table[j].legacy;
                if (!alias) continue;
                size_t length = strlen(alias);
                if (!strncmp(arg + 1, alias, length) && length > matched) {
                    option = &option_table[j];
                    matched = length;
                }
            }
            if (!strncmp(arg, "-debug", 6)) {
                for (size_t j = 0; j < sizeof(option_table) / sizeof(option_table[0]); ++j) {
                    if (option_table[j].id == DEBUG) option = &option_table[j];
                }
                matched = 5;
            }
            if (option && arg[matched + 1]) value = arg + matched + 1;
        }
        if (!option) {
            fprintf(err, "Unknown argument: %s (see --help)\n", arg);
            return -1;
        }
        if (!option->takes_value && value) {
            fprintf(err, "--%s is a Boolean switch and does not accept a value.\n", option->name);
            return -1;
        }
        if (option->id == HELP) return 1;
        if (legacy) o->legacy_options = true;
        /* Decode the old combined -lSEC=FILE, -iLEVEL=FILE, -dbgLEVEL=FILE
         * syntax into the same fields. Never write NULs into argv. */
        char number[64];
        if (legacy && (option->id == TIME_LIMIT || option->id == INSPECT || option->id == DEBUG)) {
            char *equals = value ? strchr(value, '=') : NULL;
            if (option->id == INSPECT && !equals) {
                fprintf(err, "Legacy -i requires LEVEL=FILE; use --inspect FILE --inspect-level LEVEL.\n");
                return -1;
            }
            if (equals) {
                size_t length = (size_t)(equals - value);
                if (!equals[1] || length >= sizeof(number)) {
                    fprintf(err, "Invalid legacy argument: %s\n", arg);
                    return -1;
                }
                memcpy(number, value, length);
                number[length] = '\0';
                if (!length && option->id != TIME_LIMIT) strcpy(number, "0");
                if (option->id == TIME_LIMIT) o->checkpoint = equals + 1;
                else if (option->id == DEBUG) o->debug_file = equals + 1;
                else {
                    o->inspect = equals + 1;
                    if (!apply_option(INSPECT_LEVEL, number, o)) {
                        fprintf(err, "Invalid inspection level: %s\n", number);
                        return -1;
                    }
                    continue;
                }
                value = number;
            } else if (!value && option->id == DEBUG) value = "0";
        }
        if (legacy && option->id == RESUME && value && *value == '=') ++value;
        if (option->takes_value && !value) {
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(err, "--%s requires a value.\n", option->name);
                return -1;
            }
            value = argv[++i];
        }
        if ((option->takes_value && !*value) || !apply_option(option->id, value, o)) {
            fprintf(err, "Invalid value for --%s: %s (%s)\n", option->name,
                    value ? value : "", option->description);
            return -1;
        }
        if (option->id == INSPECT_LEVEL) inspect_level_set = true;
    }
    if (inspect_level_set && !o->inspect) {
        fprintf(err, "--inspect-level requires --inspect FILE.\n");
        return -1;
    }
    if (o->checkpoint && o->time_limit == 0 && o->checkpoint_every == 0) {
        fprintf(err, "--checkpoint requires a positive --time-limit or --checkpoint-every.\n");
        return -1;
    }
    bool shared_budget = o->systematic_budget.mode != SYSTEMATIC_STAGE_BUDGETS;
    if ((shared_budget || o->systematic_stage_limits_explicit ||
         o->systematic_work_explicit || o->systematic_strengthening_explicit ||
         o->systematic_descent_explicit) &&
        !o->certify_systematic) {
        fprintf(err, "Systematic certification options require --certify-systematic.\n");
        return -1;
    }
    if (shared_budget && o->systematic_stage_limits_explicit) {
        if (o->systematic_budget.mode == SYSTEMATIC_TOTAL_BUDGET)
            fprintf(err, "--certification-time-limit cannot be combined with individual stage limits.\n");
        else
            fprintf(err, "Use either a legacy shared certification time or separate stage budgets.\n");
        return -1;
    }
    if (o->pool) o->weights = 2;
    return 0;
}
