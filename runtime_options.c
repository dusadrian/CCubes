/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "runtime_options.h"
#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef enum { INTEGER, NUMBER, PATH, CHOICE } ValueKind;
typedef struct {
    RuntimeOption key;
    const char *name;
    ValueKind kind;
    double minimum, maximum;
    const char *choices;
    const char *help;
    bool ipc;
} OptionDefinition;
static const OptionDefinition definitions[] = {
#define PUBLIC_OPTION(key, name, kind, lo, hi, choices, help) \
    {OPTION_##key, name, kind, lo, hi, choices, help, true},
#include "runtime_option_defs.h"
#undef PUBLIC_OPTION
};
static const char *values[OPTION_COUNT];
static bool requires_ipc;

const char *ccubes_option(RuntimeOption key) {
    return key >= 0 && key < OPTION_COUNT ? values[key] : NULL;
}
void ccubes_options_reset(void) {
    memset(values, 0, sizeof(values));
    requires_ipc = false;
}
bool ccubes_options_require_ipc(void) { return requires_ipc; }

static bool choice_valid(const char *value, const char *choices) {
    size_t n = strlen(value);
    for (const char *p = choices; p && *p;) {
        const char *end = strchr(p, '|');
        size_t length = end ? (size_t)(end - p) : strlen(p);
        if (length == n && !strncmp(p, value, n)) return true;
        p = end ? end + 1 : NULL;
    }
    return false;
}
static bool valid(const OptionDefinition *d, const char *value) {
    if (!value || !*value) return false;
    if (d->kind == PATH) return true;
    if (d->kind == CHOICE) return choice_valid(value, d->choices);
    if (!isdigit((unsigned char)*value) && *value != '.') return false;
    char *end = NULL;
    errno = 0;
    double number;
    if (d->kind == INTEGER) number = (double)strtol(value, &end, 10);
    else number = strtod(value, &end);
    return !errno && end != value && !*end && isfinite(number) &&
        number >= d->minimum && number <= d->maximum;
}
int ccubes_options_parse(int argc, char *const argv[], int *index, FILE *err) {
    const char *arg = argv[*index];
    if (strncmp(arg, "--", 2)) return 0;
    const char *eq = strchr(arg + 2, '=');
    size_t length = eq ? (size_t)(eq - arg - 2) : strlen(arg + 2);
    for (size_t i = 0; i < sizeof(definitions) / sizeof(definitions[0]); ++i) {
        const OptionDefinition *d = &definitions[i];
        if (strlen(d->name) != length || strncmp(arg + 2, d->name, length)) continue;
        const char *value = eq ? eq + 1 : NULL;
        if (!value) {
            if (*index + 1 >= argc || argv[*index + 1][0] == '-') {
                fprintf(err, "--%s requires a value.\n", d->name);
                return -1;
            }
            value = argv[++*index];
        }
        if (!valid(d, value)) {
            fprintf(err, "Invalid value for --%s: %s", d->name, value);
            if (d->choices) fprintf(err, " (expected %s)", d->choices);
            else if (d->kind != PATH) fprintf(err, " (range %.8g to %.8g)", d->minimum, d->maximum);
            fputc('\n', err);
            return -1;
        }
        values[d->key] = value;
        if (d->ipc) requires_ipc = true;
        return 1;
    }
    return 0;
}
void ccubes_options_help(FILE *out) {
    fprintf(out, "IPC resource and diagnostic options:\n");
    for (size_t i = 0; i < sizeof(definitions) / sizeof(definitions[0]); ++i) {
        const OptionDefinition *d = &definitions[i];
        fprintf(out, "  --%s VALUE\n      %s", d->name, d->help);
        if (d->choices) fprintf(out, " (%s)", d->choices);
        fputc('\n', out);
    }
}
