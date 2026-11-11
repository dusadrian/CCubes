/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "debug.h"
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

const char *run_log_path = NULL;

static bool same_file(int fd, const char *path) {
    struct stat log_stat, path_stat;
    return path && fstat(fd, &log_stat) == 0 && stat(path, &path_stat) == 0 &&
        log_stat.st_dev == path_stat.st_dev && log_stat.st_ino == path_stat.st_ino;
}

bool run_log_check_path(const char *path) {
    if (run_log_path && same_file(fileno(stderr), path)) {
        fprintf(stderr, "Error: --log-file must be separate from data and debug files: %s\n", path);
        return false;
    }
    return true;
}

bool run_log_init(const char *filename, const char *const protected_paths[], size_t count) {
    /* Append is also essential when a backend opens its own handle to this log. */
    int fd = open(filename, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0) {
        fprintf(stderr, "Error: cannot open --log-file %s: %s\n", filename, strerror(errno));
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (same_file(fd, protected_paths[i])) {
            fprintf(stderr, "Error: --log-file must be separate from data and debug files: %s\n", protected_paths[i]);
            close(fd);
            return false;
        }
    }
    int original_stderr = dup(fileno(stderr));
    if (original_stderr < 0) {
        fprintf(stderr, "Error: cannot initialize --log-file: %s\n", strerror(errno));
        close(fd);
        return false;
    }
    fflush(stderr);
    if (dup2(fd, fileno(stderr)) < 0) {
        fprintf(stderr, "Error: cannot redirect diagnostics to %s: %s\n", filename, strerror(errno));
        close(fd);
        close(original_stderr);
        return false;
    }
    if (fd != fileno(stderr)) close(fd);
    if (fprintf(stderr, "CCUBES_LOG_BEGIN pid=%ld\n", (long)getpid()) < 0 || fflush(stderr) != 0) {
        int saved_error = errno;
        dup2(original_stderr, fileno(stderr));
        close(original_stderr);
        clearerr(stderr);
        fprintf(stderr, "Error: cannot write --log-file %s: %s\n", filename, strerror(saved_error));
        return false;
    }
    close(original_stderr);
    run_log_path = filename;
    return true;
}

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

