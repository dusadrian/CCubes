/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_rwls.h"

#include "ccubes_threads.h"

#include <assert.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <time.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#ifndef RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES
#define RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES UINT64_C(8388608)
#endif

#define RWLS_MAX_MIXED_DIVISOR 1024
#define RWLS_MAX_LOGICAL_FAMILIES 2048

typedef struct {
    uint64_t bytes;
    ProgressiveRWLSSharedCacheSource source;
} RWLSSharedCacheInfo;

static atomic_int rwls_shared_cache_state = 0;
static RWLSSharedCacheInfo rwls_shared_cache_info;

#if defined(__APPLE__)
static bool rwls_darwin_sysctl_u64(
    const char *name,
    uint64_t *value
) {
    if (!name || !value) return false;
    uint64_t result = 0;
    size_t size = sizeof(result);
    if (
        sysctlbyname(name, &result, &size, NULL, 0) != 0 ||
        size == 0 ||
        result == 0
    ) {
        return false;
    }
    *value = result;
    return true;
}

static RWLSSharedCacheInfo rwls_detect_shared_cache(void) {
    uint64_t bytes = 0;
    if (rwls_darwin_sysctl_u64("hw.l3cachesize", &bytes)) {
        return (RWLSSharedCacheInfo){
            .bytes = bytes,
            .source = PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_L3
        };
    }

    int performance_levels = 0;
    size_t levels_size = sizeof(performance_levels);
    if (
        sysctlbyname(
            "hw.nperflevels",
            &performance_levels,
            &levels_size,
            NULL,
            0
        ) != 0 ||
        performance_levels <= 0
    ) {
        performance_levels = 8;
    }
    if (performance_levels > 64) performance_levels = 64;

    uint64_t largest = 0;
    for (int level = 0; level < performance_levels; ++level) {
        char name[64];
        const int length = snprintf(
            name,
            sizeof(name),
            "hw.perflevel%d.l2cachesize",
            level
        );
        uint64_t candidate = 0;
        if (
            length > 0 &&
            (size_t)length < sizeof(name) &&
            rwls_darwin_sysctl_u64(name, &candidate) &&
            candidate > largest
        ) {
            largest = candidate;
        }
    }
    if (largest > 0) {
        return (RWLSSharedCacheInfo){
            .bytes = largest,
            .source =
                PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_PERFORMANCE_L2
        };
    }

    return (RWLSSharedCacheInfo){
        .bytes = RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES > 0
            ? RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES
            : UINT64_C(8388608),
        .source = PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK
    };
}
#elif defined(__linux__)
static bool rwls_read_text_file(
    const char *path,
    char *buffer,
    size_t capacity
) {
    if (!path || !buffer || capacity < 2) return false;
    FILE *file = fopen(path, "r");
    if (!file) return false;
    const bool ok = fgets(buffer, (int)capacity, file) != NULL;
    fclose(file);
    if (!ok) return false;
    buffer[strcspn(buffer, "\r\n")] = '\0';
    return true;
}

static bool rwls_parse_cache_size(
    const char *text,
    uint64_t *bytes
) {
    if (!text || !bytes) return false;
    char *suffix = NULL;
    unsigned long long value = strtoull(text, &suffix, 10);
    if (suffix == text || value == 0) return false;
    while (*suffix == ' ' || *suffix == '\t') suffix++;
    uint64_t multiplier = 1;
    if (*suffix == 'K' || *suffix == 'k') {
        multiplier = UINT64_C(1024);
    } else if (*suffix == 'M' || *suffix == 'm') {
        multiplier = UINT64_C(1024) * UINT64_C(1024);
    } else if (*suffix == 'G' || *suffix == 'g') {
        multiplier =
            UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);
    }
    if ((uint64_t)value > UINT64_MAX / multiplier) return false;
    *bytes = (uint64_t)value * multiplier;
    return true;
}

static RWLSSharedCacheInfo rwls_detect_shared_cache(void) {
#if defined(_SC_LEVEL3_CACHE_SIZE)
    const long sysconf_bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (sysconf_bytes > 0) {
        return (RWLSSharedCacheInfo){
            .bytes = (uint64_t)sysconf_bytes,
            .source = PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSCONF_L3
        };
    }
#endif

    int highest_level = -1;
    uint64_t largest_at_level = 0;
    for (int index = 0; index < 64; ++index) {
        char path[160];
        char type[32];
        char level_text[32];
        char size_text[32];
        snprintf(
            path,
            sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/type",
            index
        );
        if (!rwls_read_text_file(path, type, sizeof(type))) continue;
        if (strcmp(type, "Unified") != 0 && strcmp(type, "Data") != 0) {
            continue;
        }
        snprintf(
            path,
            sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/level",
            index
        );
        if (!rwls_read_text_file(path, level_text, sizeof(level_text))) {
            continue;
        }
        char *level_end = NULL;
        const long level = strtol(level_text, &level_end, 10);
        if (level_end == level_text || level < 1 || level > INT_MAX) {
            continue;
        }
        snprintf(
            path,
            sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/size",
            index
        );
        uint64_t candidate = 0;
        if (
            !rwls_read_text_file(path, size_text, sizeof(size_text)) ||
            !rwls_parse_cache_size(size_text, &candidate)
        ) {
            continue;
        }
        if ((int)level > highest_level) {
            highest_level = (int)level;
            largest_at_level = candidate;
        } else if (
            (int)level == highest_level &&
            candidate > largest_at_level
        ) {
            largest_at_level = candidate;
        }
    }
    if (largest_at_level > 0) {
        return (RWLSSharedCacheInfo){
            .bytes = largest_at_level,
            .source = PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSFS
        };
    }

    return (RWLSSharedCacheInfo){
        .bytes = RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES > 0
            ? RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES
            : UINT64_C(8388608),
        .source = PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK
    };
}
#elif defined(_WIN32)
static RWLSSharedCacheInfo rwls_detect_shared_cache(void) {
    DWORD bytes_needed = 0;
    GetLogicalProcessorInformationEx(
        RelationCache,
        NULL,
        &bytes_needed
    );
    if (bytes_needed > 0) {
        unsigned char *storage =
            (unsigned char *)malloc((size_t)bytes_needed);
        if (storage) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX entry =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)storage;
            if (GetLogicalProcessorInformationEx(
                RelationCache,
                entry,
                &bytes_needed
            )) {
                DWORD offset = 0;
                BYTE highest_level = 0;
                uint64_t largest_at_level = 0;
                while (offset < bytes_needed) {
                    entry =
                        (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)
                            (storage + offset);
                    if (
                        entry->Relationship == RelationCache &&
                        (
                            entry->Cache.Type == CacheUnified ||
                            entry->Cache.Type == CacheData
                        )
                    ) {
                        const BYTE level = entry->Cache.Level;
                        const uint64_t candidate =
                            (uint64_t)entry->Cache.CacheSize;
                        if (level > highest_level) {
                            highest_level = level;
                            largest_at_level = candidate;
                        } else if (
                            level == highest_level &&
                            candidate > largest_at_level
                        ) {
                            largest_at_level = candidate;
                        }
                    }
                    if (entry->Size == 0) break;
                    offset += entry->Size;
                }
                free(storage);
                if (largest_at_level > 0) {
                    return (RWLSSharedCacheInfo){
                        .bytes = largest_at_level,
                        .source =
                            PROGRESSIVE_RWLS_SHARED_CACHE_WINDOWS_API
                    };
                }
            } else {
                free(storage);
            }
        }
    }
    return (RWLSSharedCacheInfo){
        .bytes = RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES > 0
            ? RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES
            : UINT64_C(8388608),
        .source = PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK
    };
}
#else
static RWLSSharedCacheInfo rwls_detect_shared_cache(void) {
    return (RWLSSharedCacheInfo){
        .bytes = RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES > 0
            ? RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES
            : UINT64_C(8388608),
        .source = PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK
    };
}
#endif

static RWLSSharedCacheInfo rwls_get_shared_cache_info(void) {
    int state = atomic_load_explicit(
        &rwls_shared_cache_state,
        memory_order_acquire
    );
    if (state != 2) {
        int expected = 0;
        if (atomic_compare_exchange_strong_explicit(
            &rwls_shared_cache_state,
            &expected,
            1,
            memory_order_acq_rel,
            memory_order_acquire
        )) {
            rwls_shared_cache_info = rwls_detect_shared_cache();
            atomic_store_explicit(
                &rwls_shared_cache_state,
                2,
                memory_order_release
            );
        } else {
            while (atomic_load_explicit(
                &rwls_shared_cache_state,
                memory_order_acquire
            ) != 2) {
                /* Detection is a one-time, bounded platform query. */
            }
        }
    }
    return rwls_shared_cache_info;
}

uint64_t progressive_rwls_shared_cache_bytes(void) {
    return rwls_get_shared_cache_info().bytes;
}

ProgressiveRWLSSharedCacheSource
progressive_rwls_shared_cache_source(void) {
    return rwls_get_shared_cache_info().source;
}

const char *progressive_rwls_shared_cache_source_name(
    ProgressiveRWLSSharedCacheSource source
) {
    switch (source) {
        case PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_L3:
            return "darwin-l3";
        case PROGRESSIVE_RWLS_SHARED_CACHE_DARWIN_PERFORMANCE_L2:
            return "darwin-performance-l2";
        case PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSCONF_L3:
            return "linux-sysconf-l3";
        case PROGRESSIVE_RWLS_SHARED_CACHE_LINUX_SYSFS:
            return "linux-sysfs";
        case PROGRESSIVE_RWLS_SHARED_CACHE_WINDOWS_API:
            return "windows-api";
        case PROGRESSIVE_RWLS_SHARED_CACHE_FALLBACK:
        default:
            return "fallback";
    }
}

uint64_t progressive_rwls_packed_incidence_threshold_bytes(void) {
    if ((uint64_t)RWLS_PACKED_MIN_DENSE_COVERAGE_BYTES == 0) return 0;
    return progressive_rwls_shared_cache_bytes();
}

typedef struct {
    const ProgressiveNativeMaster *master;
    size_t *row_offsets;
    size_t *column_offsets;
    uint16_t *column_rows16;
    int *row_columns;
    unsigned char *compulsory_columns;
    bool packed_bidirectional;
} FeasibilityJumpIncidence;

typedef struct {
    const ProgressiveNativeMaster *master;
    const FeasibilityJumpIncidence *incidence;
    unsigned char *selected;
    unsigned char *ever_selected;
    int *selected_list;
    int *selected_position;
    int selected_capacity;
    int *coverage_count;
    int *sole_column;
    int *uncovered_list;
    int *uncovered_position;
    uint64_t *uncovered_bits;
    int uncovered_words;
    int64_t *row_penalty;
    int64_t *uncovered_gain;
    int64_t *private_loss;
    int64_t *timestamp;
    int *add_tabu_until;
    int *remove_tabu_until;
    int uncovered_rows;
    int64_t uncovered_weight;
    int selected_columns;
    int distinct_selected_columns;
    int64_t rwls_step;
    int64_t soft_pressure;
    bool rwls_weights_initialized;
    ProgressiveRWLSWorkerPolicy rwls_policy;
    int64_t maximum_row_penalty;
    int64_t maximum_uncovered_gain;
    int64_t maximum_private_loss;
} FeasibilityJumpState;

typedef struct {
    FeasibilityJumpState state;
    int *candidate;
    bool improved;
    bool attempted;
    ProgressiveNativeFeasibilityJumpStats stats;
} FeasibilityJumpWorker;

typedef struct {
    FeasibilityJumpWorker *workers;
    int family_count;
    int physical_threads_used;
    int incumbent_cardinality;
    const int *incumbent;
    ProgressiveNativeFeasibilityJumpOptions options;
    int target_ordinal;
    int adaptive_round;
    bool fixed_only;
} FeasibilityJumpParallelContext;

static uint64_t jump_monotonic_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0;
    return
        (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static bool jump_deadline_reached(uint64_t deadline) {
    return deadline > 0 && jump_monotonic_nanoseconds() >= deadline;
}

static uint64_t jump_mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) *
        UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) *
        UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static bool jump_column_covers(
    const ProgressiveNativeMaster *master,
    int column,
    int row
) {
    return (
        master->coverage[
            (size_t)column * (size_t)master->words +
            (size_t)(row / 64)
        ] >> (row % 64)
    ) & UINT64_C(1);
}

void progressive_native_feasibility_jump_options_default(
    ProgressiveNativeFeasibilityJumpOptions *options
) {
    if (!options) return;
    *options = (ProgressiveNativeFeasibilityJumpOptions){
        .steps_per_family = UINT64_C(8192),
        .max_adaptive_rounds = 8,
        .requested_threads = 0,
        .logical_family_count = 0,
        .tabu_tenure = 3,
        .mixed_soft_worker_divisor = 4,
        .seed = 0,
        .deadline_nanoseconds = 0,
        .memory_budget_bytes = 0,
        .budget_mode = PROGRESSIVE_RWLS_BUDGET_WORK,
        .worker_policy = PROGRESSIVE_RWLS_WORKER_MIXED
    };
}

ProgressiveRWLSWorkerPolicy progressive_rwls_mixed_family_policy(
    int soft_divisor,
    uint64_t family_ordinal
) {
    if (soft_divisor < 2) {
        return PROGRESSIVE_RWLS_WORKER_FIXED_TARGET;
    }
    const uint64_t divisor = (uint64_t)soft_divisor;
    return family_ordinal % divisor == 1
        ? PROGRESSIVE_RWLS_WORKER_SOFT_PRESSURE
        : PROGRESSIVE_RWLS_WORKER_FIXED_TARGET;
}

int progressive_rwls_mixed_logical_family_count(
    int physical_thread_count,
    int soft_divisor
) {
    if (
        physical_thread_count < 1 ||
        soft_divisor < 2 ||
        soft_divisor > RWLS_MAX_MIXED_DIVISOR
    ) {
        return 0;
    }
    if (physical_thread_count > INT_MAX - (soft_divisor - 1)) {
        return 0;
    }
    const int family_count =
        (
            physical_thread_count + soft_divisor - 1
        ) / soft_divisor * soft_divisor;
    return family_count <= RWLS_MAX_LOGICAL_FAMILIES
        ? family_count
        : 0;
}

static bool checked_add_u64(uint64_t *total, uint64_t value) {
    if (!total || *total > UINT64_MAX - value) return false;
    *total += value;
    return true;
}

static bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t *value) {
    if (!value || (left != 0 && right > UINT64_MAX / left)) return false;
    *value = left * right;
    return true;
}

static bool should_use_packed_bidirectional_incidence(
    const ProgressiveNativeMaster *master,
    uint64_t incidence_entries
) {
    if (
        !master ||
        master->rows > (int)UINT16_MAX + 1 ||
        incidence_entries > UINT64_MAX / sizeof(uint16_t)
    ) {
        return false;
    }
    const uint64_t dense_coverage_bytes =
        (uint64_t)master->count * (uint64_t)master->words *
        sizeof(uint64_t);
    const uint64_t sparse_column_bytes =
        incidence_entries * sizeof(uint16_t);
    return
        dense_coverage_bytes >=
            progressive_rwls_packed_incidence_threshold_bytes() &&
        sparse_column_bytes < dense_coverage_bytes;
}

static bool estimate_shared_memory(
    const ProgressiveNativeMaster *master,
    uint64_t incidence_entries,
    uint64_t *bytes
) {
    if (!bytes) return false;
    uint64_t total = 0;
    const bool packed_bidirectional =
        should_use_packed_bidirectional_incidence(
            master,
            incidence_entries
        );
    if (!checked_add_u64(
        &total,
        ((uint64_t)master->rows + 1u) * sizeof(size_t)
    )) {
        return false;
    }
    if (!checked_add_u64(
        &total,
        incidence_entries * sizeof(int)
    )) {
        return false;
    }
    if (packed_bidirectional) {
        if (
            !checked_add_u64(
                &total,
                ((uint64_t)master->count + 1u) * sizeof(size_t)
            ) ||
            !checked_add_u64(
                &total,
                incidence_entries * sizeof(uint16_t)
            )
        ) {
            return false;
        }
    }
    return
        checked_add_u64(
            &total,
            (uint64_t)master->count * sizeof(unsigned char)
        ) &&
        ((*bytes = total), true);
}

static bool estimate_worker_memory(
    const ProgressiveNativeMaster *master,
    int incumbent_cardinality,
    int selected_capacity,
    uint64_t *bytes
) {
    if (!bytes) return false;
    uint64_t total = 0;
    return
        checked_add_u64(
            &total,
            (uint64_t)master->count * 2u * sizeof(unsigned char)
        ) &&
        checked_add_u64(
            &total,
            (uint64_t)master->rows * 4u * sizeof(int)
        ) &&
        checked_add_u64(
            &total,
            ((uint64_t)master->rows + 63u) / 64u * sizeof(uint64_t)
        ) &&
        checked_add_u64(
            &total,
            (uint64_t)master->rows * sizeof(int64_t)
        ) &&
        checked_add_u64(
            &total,
            (uint64_t)master->count * 3u *
                sizeof(int64_t)
        ) &&
        checked_add_u64(
            &total,
            (uint64_t)master->count * 3u * sizeof(int)
        ) &&
        checked_add_u64(
            &total,
            (
                (uint64_t)incumbent_cardinality +
                (uint64_t)selected_capacity
            ) * sizeof(int)
        ) &&
        ((*bytes = total), true);
}

static bool estimate_peak_memory(
    const ProgressiveNativeMaster *master,
    uint64_t shared_bytes,
    uint64_t worker_bytes,
    int families,
    int incumbent_cardinality,
    uint64_t *bytes
) {
    uint64_t parallel_bytes = 0;
    uint64_t worker_total = 0;
    if (
        families <= 0 ||
        !checked_mul_u64(worker_bytes, (uint64_t)families, &worker_total) ||
        !checked_add_u64(&parallel_bytes, worker_total) ||
        !checked_add_u64(
            &parallel_bytes,
            (uint64_t)families * sizeof(FeasibilityJumpWorker)
        ) ||
        !checked_add_u64(
            &parallel_bytes,
            (uint64_t)incumbent_cardinality * sizeof(int)
        )
    ) {
        return false;
    }
    const uint64_t build_scratch =
        (uint64_t)master->rows * sizeof(size_t);
    uint64_t peak = shared_bytes;
    if (!checked_add_u64(
        &peak,
        parallel_bytes > build_scratch ? parallel_bytes : build_scratch
    )) {
        return false;
    }
    *bytes = peak;
    return true;
}

static bool build_incidence(
    FeasibilityJumpIncidence *incidence,
    const ProgressiveNativeMaster *master,
    uint64_t incidence_entries
) {
    memset(incidence, 0, sizeof(*incidence));
    incidence->master = master;
    incidence->packed_bidirectional =
        should_use_packed_bidirectional_incidence(
            master,
            incidence_entries
        );
    incidence->row_offsets = (size_t *)calloc(
        (size_t)master->rows + 1u,
        sizeof(size_t)
    );
    incidence->row_columns = (int *)malloc(
        (size_t)incidence_entries * sizeof(int)
    );
    if (incidence->packed_bidirectional) {
        incidence->column_offsets = (size_t *)calloc(
            (size_t)master->count + 1u,
            sizeof(size_t)
        );
        incidence->column_rows16 = (uint16_t *)malloc(
            (size_t)incidence_entries * sizeof(uint16_t)
        );
    }
    incidence->compulsory_columns = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    size_t *cursor = (size_t *)malloc(
        (size_t)master->rows * sizeof(size_t)
    );
    if (
        !incidence->row_offsets ||
        !incidence->row_columns ||
        (
            incidence->packed_bidirectional &&
            (
                !incidence->column_offsets ||
                !incidence->column_rows16
            )
        ) ||
        !incidence->compulsory_columns ||
        !cursor
    ) {
        free(cursor);
        return false;
    }
    for (int column = 0; column < master->count; ++column) {
        const uint64_t *coverage = master->coverage +
            (size_t)column * (size_t)master->words;
        size_t column_degree = 0;
        for (int word = 0; word < master->words; ++word) {
            uint64_t bits = coverage[word];
            while (bits != 0) {
                const int bit = __builtin_ctzll(bits);
                const int row = word * 64 + bit;
                if (row < master->rows) {
                    incidence->row_offsets[row + 1]++;
                    column_degree++;
                }
                bits &= bits - 1;
            }
        }
        if (incidence->packed_bidirectional) {
            incidence->column_offsets[column + 1] =
                incidence->column_offsets[column] + column_degree;
        }
    }
    for (int row = 0; row < master->rows; ++row) {
        incidence->row_offsets[row + 1] +=
            incidence->row_offsets[row];
        cursor[row] = incidence->row_offsets[row];
    }
    for (int column = 0; column < master->count; ++column) {
        const uint64_t *coverage = master->coverage +
            (size_t)column * (size_t)master->words;
        size_t column_cursor = incidence->packed_bidirectional
            ? incidence->column_offsets[column]
            : 0;
        for (int word = 0; word < master->words; ++word) {
            uint64_t bits = coverage[word];
            while (bits != 0) {
                const int bit = __builtin_ctzll(bits);
                const int row = word * 64 + bit;
                if (row < master->rows) {
                    incidence->row_columns[cursor[row]++] = column;
                    if (incidence->packed_bidirectional) {
                        incidence->column_rows16[column_cursor++] =
                            (uint16_t)row;
                    }
                }
                bits &= bits - 1;
            }
        }
    }
    for (int row = 0; row < master->rows; ++row) {
        const size_t degree =
            incidence->row_offsets[row + 1] -
            incidence->row_offsets[row];
        if (degree == 1) {
            const size_t index = incidence->row_offsets[row];
            const int column = incidence->row_columns[index];
            incidence->compulsory_columns[
                column
            ] = 1;
        }
    }
    free(cursor);
    return true;
}

static void destroy_incidence(FeasibilityJumpIncidence *incidence) {
    if (!incidence) return;
    free(incidence->row_offsets);
    free(incidence->column_offsets);
    free(incidence->column_rows16);
    free(incidence->row_columns);
    free(incidence->compulsory_columns);
    memset(incidence, 0, sizeof(*incidence));
}

static void destroy_state(FeasibilityJumpState *state) {
    if (!state) return;
    free(state->selected);
    free(state->ever_selected);
    free(state->selected_list);
    free(state->selected_position);
    free(state->coverage_count);
    free(state->sole_column);
    free(state->uncovered_list);
    free(state->uncovered_position);
    free(state->uncovered_bits);
    free(state->row_penalty);
    free(state->uncovered_gain);
    free(state->private_loss);
    free(state->timestamp);
    free(state->add_tabu_until);
    free(state->remove_tabu_until);
    memset(state, 0, sizeof(*state));
}

static bool allocate_state(
    FeasibilityJumpState *state,
    const ProgressiveNativeMaster *master,
    const FeasibilityJumpIncidence *incidence,
    int selected_capacity
) {
    memset(state, 0, sizeof(*state));
    state->master = master;
    state->incidence = incidence;
    state->selected = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    state->ever_selected = (unsigned char *)calloc(
        (size_t)master->count,
        sizeof(unsigned char)
    );
    state->selected_list = (int *)malloc(
        (size_t)selected_capacity * sizeof(int)
    );
    state->selected_position = (int *)malloc(
        (size_t)master->count * sizeof(int)
    );
    state->selected_capacity = selected_capacity;
    state->coverage_count = (int *)calloc(
        (size_t)master->rows,
        sizeof(int)
    );
    state->sole_column = (int *)malloc(
        (size_t)master->rows * sizeof(int)
    );
    state->uncovered_list = (int *)malloc(
        (size_t)master->rows * sizeof(int)
    );
    state->uncovered_position = (int *)malloc(
        (size_t)master->rows * sizeof(int)
    );
    state->uncovered_words = (master->rows + 63) / 64;
    state->uncovered_bits = (uint64_t *)calloc(
        (size_t)state->uncovered_words,
        sizeof(uint64_t)
    );
    state->row_penalty = (int64_t *)malloc(
        (size_t)master->rows * sizeof(int64_t)
    );
    state->uncovered_gain = (int64_t *)calloc(
        (size_t)master->count,
        sizeof(int64_t)
    );
    state->private_loss = (int64_t *)calloc(
        (size_t)master->count,
        sizeof(int64_t)
    );
    state->timestamp = (int64_t *)calloc(
        (size_t)master->count,
        sizeof(int64_t)
    );
    state->add_tabu_until = (int *)calloc(
        (size_t)master->count,
        sizeof(int)
    );
    state->remove_tabu_until = (int *)calloc(
        (size_t)master->count,
        sizeof(int)
    );
    if (
        !state->selected ||
        !state->ever_selected ||
        !state->selected_list ||
        !state->selected_position ||
        !state->coverage_count ||
        !state->sole_column ||
        !state->uncovered_list ||
        !state->uncovered_position ||
        !state->uncovered_bits ||
        !state->row_penalty ||
        !state->uncovered_gain ||
        !state->private_loss ||
        !state->timestamp ||
        !state->add_tabu_until ||
        !state->remove_tabu_until
    ) {
        destroy_state(state);
        return false;
    }
    return true;
}

static void select_column(FeasibilityJumpState *state, int column) {
    assert(column >= 0 && column < state->master->count);
    assert(!state->selected[column]);
    assert(state->selected_columns < state->selected_capacity);
    const int position = state->selected_columns++;
    state->selected[column] = 1;
    if (!state->ever_selected[column]) {
        state->ever_selected[column] = 1;
        state->distinct_selected_columns++;
    }
    state->selected_list[position] = column;
    state->selected_position[column] = position;
}

static void deselect_column(FeasibilityJumpState *state, int column) {
    assert(column >= 0 && column < state->master->count);
    assert(state->selected[column]);
    const int position = state->selected_position[column];
    assert(position >= 0 && position < state->selected_columns);
    assert(state->selected_list[position] == column);
    const int last_position = --state->selected_columns;
    const int last_column = state->selected_list[last_position];
    state->selected_list[position] = last_column;
    state->selected_position[last_column] = position;
    state->selected_position[column] = -1;
    state->selected[column] = 0;
}

static void mark_row_uncovered(FeasibilityJumpState *state, int row) {
    assert(row >= 0 && row < state->master->rows);
    assert(state->coverage_count[row] == 0);
    assert(state->uncovered_rows < state->master->rows);
    assert(
        !(state->uncovered_bits[row / 64] &
            (UINT64_C(1) << (row % 64)))
    );
    const int position = state->uncovered_rows++;
    state->uncovered_list[position] = row;
    state->uncovered_position[row] = position;
    state->uncovered_bits[row / 64] |= UINT64_C(1) << (row % 64);
}

static void mark_row_covered(FeasibilityJumpState *state, int row) {
    assert(row >= 0 && row < state->master->rows);
    assert(state->coverage_count[row] == 0);
    assert(
        state->uncovered_bits[row / 64] &
            (UINT64_C(1) << (row % 64))
    );
    const int position = state->uncovered_position[row];
    assert(position >= 0 && position < state->uncovered_rows);
    assert(state->uncovered_list[position] == row);
    const int last_position = --state->uncovered_rows;
    const int last_row = state->uncovered_list[last_position];
    state->uncovered_list[position] = last_row;
    state->uncovered_position[last_row] = position;
    state->uncovered_position[row] = -1;
    state->uncovered_bits[row / 64] &= ~(UINT64_C(1) << (row % 64));
}

static bool selection_is_cover(
    const ProgressiveNativeMaster *master,
    int cardinality,
    const int *columns
) {
    if (cardinality <= 0 || !columns) return false;
    for (int index = 0; index < cardinality; ++index) {
        if (columns[index] < 0 || columns[index] >= master->count) {
            return false;
        }
        for (int previous = 0; previous < index; ++previous) {
            if (columns[previous] == columns[index]) return false;
        }
    }
    for (int row = 0; row < master->rows; ++row) {
        bool covered = false;
        for (int index = 0; index < cardinality; ++index) {
            const int column = columns[index];
            if (!jump_column_covers(master, column, row)) continue;
            covered = true;
            break;
        }
        if (!covered) return false;
    }
    return true;
}

static int find_selected_on_row(
    const FeasibilityJumpState *state,
    int row
) {
    for (
        size_t index = state->incidence->row_offsets[row];
        index < state->incidence->row_offsets[row + 1];
        ++index
    ) {
        const int column = state->incidence->row_columns[index];
        if (state->selected[column]) return column;
    }
    return -1;
}

static inline void add_column_row(
    FeasibilityJumpState *state,
    int column,
    int row
) {
    if (state->coverage_count[row] == 0) {
        mark_row_covered(state, row);
        state->uncovered_weight -= state->row_penalty[row];
        for (
            size_t index = state->incidence->row_offsets[row];
            index < state->incidence->row_offsets[row + 1];
            ++index
        ) {
            const int gain_column = state->incidence->row_columns[index];
            state->uncovered_gain[gain_column] -= state->row_penalty[row];
        }
        state->sole_column[row] = column;
        state->private_loss[column] += state->row_penalty[row];
        if (
            state->private_loss[column] >
            state->maximum_private_loss
        ) {
            state->maximum_private_loss = state->private_loss[column];
        }
    } else if (state->coverage_count[row] == 1) {
        const int old_sole = state->sole_column[row];
        if (old_sole >= 0) {
            state->private_loss[old_sole] -= state->row_penalty[row];
        }
        state->sole_column[row] = -1;
    }
    state->coverage_count[row]++;
}

static inline void remove_column_row(
    FeasibilityJumpState *state,
    int column,
    int row
) {
    if (state->coverage_count[row] == 1) {
        state->private_loss[column] -= state->row_penalty[row];
        state->coverage_count[row] = 0;
        state->sole_column[row] = -1;
        mark_row_uncovered(state, row);
        state->uncovered_weight += state->row_penalty[row];
        for (
            size_t index = state->incidence->row_offsets[row];
            index < state->incidence->row_offsets[row + 1];
            ++index
        ) {
            const int gain_column = state->incidence->row_columns[index];
            state->uncovered_gain[gain_column] += state->row_penalty[row];
            if (
                state->uncovered_gain[gain_column] >
                state->maximum_uncovered_gain
            ) {
                state->maximum_uncovered_gain =
                    state->uncovered_gain[gain_column];
            }
        }
        return;
    }
    state->coverage_count[row]--;
    if (state->coverage_count[row] == 1) {
        const int sole = find_selected_on_row(state, row);
        state->sole_column[row] = sole;
        if (sole >= 0) {
            state->private_loss[sole] += state->row_penalty[row];
            if (
                state->private_loss[sole] >
                state->maximum_private_loss
            ) {
                state->maximum_private_loss = state->private_loss[sole];
            }
        }
    }
}

static void initialize_selection(
    FeasibilityJumpState *state,
    int cardinality,
    const int *columns,
    bool reset_search_memory
) {
    const ProgressiveNativeMaster *master = state->master;
    memset(
        state->selected,
        0,
        (size_t)master->count * sizeof(unsigned char)
    );
    memset(
        state->coverage_count,
        0,
        (size_t)master->rows * sizeof(int)
    );
    memset(
        state->uncovered_gain,
        0,
        (size_t)master->count * sizeof(int64_t)
    );
    memset(
        state->private_loss,
        0,
        (size_t)master->count * sizeof(int64_t)
    );
    memset(
        state->uncovered_bits,
        0,
        (size_t)state->uncovered_words * sizeof(uint64_t)
    );
    if (reset_search_memory) {
        memset(
            state->add_tabu_until,
            0,
            (size_t)master->count * sizeof(int)
        );
        memset(
            state->remove_tabu_until,
            0,
            (size_t)master->count * sizeof(int)
        );
        if (state->timestamp) {
            memset(
                state->timestamp,
                0,
                (size_t)master->count * sizeof(int64_t)
            );
        }
        state->rwls_step = 0;
        state->soft_pressure = 1;
    }
    for (int row = 0; row < master->rows; ++row) {
        state->sole_column[row] = -1;
        if (reset_search_memory || !state->rwls_weights_initialized) {
            state->row_penalty[row] = 1;
            if (state->maximum_row_penalty < 1) {
                state->maximum_row_penalty = 1;
            }
        }
    }
    state->selected_columns = 0;
    for (int index = 0; index < cardinality; ++index) {
        const int column = columns[index];
        select_column(state, column);
        if (state->incidence->packed_bidirectional) {
            for (
                size_t position = state->incidence->column_offsets[column];
                position < state->incidence->column_offsets[column + 1];
                ++position
            ) {
                state->coverage_count[
                    state->incidence->column_rows16[position]
                ]++;
            }
        } else {
            const uint64_t *coverage = master->coverage +
                (size_t)column * (size_t)master->words;
            for (int word = 0; word < master->words; ++word) {
                uint64_t bits = coverage[word];
                while (bits != 0) {
                    const int bit = __builtin_ctzll(bits);
                    const int row = word * 64 + bit;
                    if (row < master->rows) state->coverage_count[row]++;
                    bits &= bits - 1;
                }
            }
        }
    }
    state->uncovered_rows = 0;
    state->uncovered_weight = 0;
    for (int row = 0; row < master->rows; ++row) {
        if (state->coverage_count[row] == 0) {
            mark_row_uncovered(state, row);
            state->uncovered_weight += state->row_penalty[row];
            for (
                size_t index = state->incidence->row_offsets[row];
                index < state->incidence->row_offsets[row + 1];
                ++index
            ) {
                const int column = state->incidence->row_columns[index];
                state->uncovered_gain[column] += state->row_penalty[row];
                if (
                    state->uncovered_gain[column] >
                    state->maximum_uncovered_gain
                ) {
                    state->maximum_uncovered_gain =
                        state->uncovered_gain[column];
                }
            }
        } else if (state->coverage_count[row] == 1) {
            const int sole = find_selected_on_row(state, row);
            state->sole_column[row] = sole;
            if (sole >= 0) {
                state->private_loss[sole] += state->row_penalty[row];
                if (
                    state->private_loss[sole] >
                    state->maximum_private_loss
                ) {
                    state->maximum_private_loss =
                        state->private_loss[sole];
                }
            }
        }
    }
}

static void add_column(FeasibilityJumpState *state, int column) {
    const ProgressiveNativeMaster *master = state->master;
    select_column(state, column);
    if (state->incidence->packed_bidirectional) {
        for (
            size_t position = state->incidence->column_offsets[column];
            position < state->incidence->column_offsets[column + 1];
            ++position
        ) {
            add_column_row(
                state,
                column,
                (int)state->incidence->column_rows16[position]
            );
        }
        return;
    }
    const uint64_t *coverage = master->coverage +
        (size_t)column * (size_t)master->words;
    for (int word = 0; word < master->words; ++word) {
        uint64_t bits = coverage[word];
        while (bits != 0) {
            const int bit = __builtin_ctzll(bits);
            const int row = word * 64 + bit;
            bits &= bits - 1;
            if (row >= master->rows) continue;
            add_column_row(state, column, row);
        }
    }
}

static void remove_column(FeasibilityJumpState *state, int column) {
    const ProgressiveNativeMaster *master = state->master;
    deselect_column(state, column);
    if (state->incidence->packed_bidirectional) {
        for (
            size_t position = state->incidence->column_offsets[column];
            position < state->incidence->column_offsets[column + 1];
            ++position
        ) {
            remove_column_row(
                state,
                column,
                (int)state->incidence->column_rows16[position]
            );
        }
        state->private_loss[column] = 0;
        return;
    }
    const uint64_t *coverage = master->coverage +
        (size_t)column * (size_t)master->words;
    for (int word = 0; word < master->words; ++word) {
        uint64_t bits = coverage[word];
        while (bits != 0) {
            const int bit = __builtin_ctzll(bits);
            const int row = word * 64 + bit;
            bits &= bits - 1;
            if (row >= master->rows) continue;
            remove_column_row(state, column, row);
        }
    }
    state->private_loss[column] = 0;
}

static void raise_uncovered_penalties(
    FeasibilityJumpState *state,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    for (int position = 0; position < state->uncovered_rows; ++position) {
        const int row = state->uncovered_list[position];
        state->row_penalty[row]++;
        if (state->row_penalty[row] > state->maximum_row_penalty) {
            state->maximum_row_penalty = state->row_penalty[row];
        }
        state->uncovered_weight++;
        for (
            size_t index = state->incidence->row_offsets[row];
            index < state->incidence->row_offsets[row + 1];
            ++index
        ) {
            const int column = state->incidence->row_columns[index];
            state->uncovered_gain[column]++;
            if (
                state->uncovered_gain[column] >
                state->maximum_uncovered_gain
            ) {
                state->maximum_uncovered_gain =
                    state->uncovered_gain[column];
            }
        }
        stats->penalty_updates++;
    }
}

static int choose_rwls_uncovered_row(
    const FeasibilityJumpState *state,
    uint64_t seed
) {
    if (state->uncovered_rows <= 0) return -1;
    int ordinal = (int)(
        jump_mix64(seed) % (uint64_t)state->uncovered_rows
    );
    for (int word = 0; word < state->uncovered_words; ++word) {
        uint64_t bits = state->uncovered_bits[word];
        const int count = __builtin_popcountll(bits);
        if (ordinal >= count) {
            ordinal -= count;
            continue;
        }
        while (ordinal-- > 0) bits &= bits - 1;
        const int row = word * 64 + __builtin_ctzll(bits);
        return row < state->master->rows ? row : -1;
    }
    return -1;
}

static int choose_rwls_add_column(
    const FeasibilityJumpState *state,
    int row,
    int iteration
) {
    int best = -1;
    int64_t best_gain = INT64_MIN;
    int64_t best_timestamp = INT64_MAX;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
        for (
            size_t index = state->incidence->row_offsets[row];
            index < state->incidence->row_offsets[row + 1];
            ++index
        ) {
            const int column = state->incidence->row_columns[index];
            if (
                state->selected[column] ||
                (
                    pass == 0 &&
                    state->add_tabu_until[column] > iteration
                )
            ) {
                continue;
            }
            if (
                best < 0 ||
                state->uncovered_gain[column] > best_gain ||
                (
                    state->uncovered_gain[column] == best_gain &&
                    (
                        state->timestamp[column] < best_timestamp ||
                        (
                            state->timestamp[column] == best_timestamp &&
                            column < best
                        )
                    )
                )
            ) {
                best = column;
                best_gain = state->uncovered_gain[column];
                best_timestamp = state->timestamp[column];
            }
        }
    }
    return best;
}

static int choose_rwls_remove_column(
    const FeasibilityJumpState *state,
    int iteration,
    bool honor_tabu
) {
    int best = -1;
    int64_t best_loss = INT64_MAX;
    int64_t best_timestamp = INT64_MAX;
    for (int pass = 0; pass < 2 && best < 0; ++pass) {
        for (
            int position = 0;
            position < state->selected_columns;
            ++position
        ) {
            const int column = state->selected_list[position];
            if (
                state->incidence->compulsory_columns[column]
            ) {
                continue;
            }
            if (
                honor_tabu && pass == 0 &&
                state->remove_tabu_until[column] > iteration
            ) {
                continue;
            }
            if (
                best < 0 ||
                state->private_loss[column] < best_loss ||
                (
                    state->private_loss[column] == best_loss &&
                    (
                        state->timestamp[column] < best_timestamp ||
                        (
                            state->timestamp[column] == best_timestamp &&
                            column < best
                        )
                    )
                )
            ) {
                best = column;
                best_loss = state->private_loss[column];
                best_timestamp = state->timestamp[column];
            }
        }
        if (!honor_tabu) break;
    }
    return best;
}

static int rwls_tabu_until(int iteration, int tenure) {
    return iteration > INT_MAX - tenure
        ? INT_MAX
        : iteration + tenure;
}

static int collect_selection(
    const FeasibilityJumpState *state,
    int *columns
) {
    int count = 0;
    for (int column = 0; column < state->master->count; ++column) {
        if (state->selected[column]) columns[count++] = column;
    }
    return count;
}

static uint64_t selected_state_signature(
    const FeasibilityJumpState *state
) {
    uint64_t mixed_xor = jump_mix64(
        (uint64_t)(unsigned int)state->selected_columns
    );
    uint64_t mixed_sum = 0;
    for (int position = 0; position < state->selected_columns; ++position) {
        const uint64_t mixed = jump_mix64(
            (uint64_t)(unsigned int)(state->selected_list[position] + 1)
        );
        mixed_xor ^= mixed;
        mixed_sum += mixed;
    }
    return jump_mix64(mixed_xor ^ jump_mix64(mixed_sum));
}

static void note_best_failed_state(
    const FeasibilityJumpState *state,
    int target,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    if (
        !state ||
        !stats ||
        state->selected_columns != target ||
        state->uncovered_rows >= stats->best_uncovered_rows
    ) {
        return;
    }
    stats->best_uncovered_rows = state->uncovered_rows;
    stats->best_failed_witness_row = state->uncovered_rows == 1
        ? state->uncovered_list[0]
        : -1;
    stats->best_failed_selection_signature =
        selected_state_signature(state);
}

static bool search_target_rwls(
    FeasibilityJumpState *state,
    int incumbent_cardinality,
    const int *incumbent,
    const ProgressiveNativeFeasibilityJumpOptions *options,
    int *candidate,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    const int target = incumbent_cardinality - 1;
    const bool reuse_weights =
        state->rwls_weights_initialized &&
        state->rwls_policy == PROGRESSIVE_RWLS_WORKER_FIXED_TARGET;
    initialize_selection(
        state,
        incumbent_cardinality,
        incumbent,
        !reuse_weights
    );
    state->rwls_weights_initialized = true;
    state->rwls_policy = PROGRESSIVE_RWLS_WORKER_FIXED_TARGET;
    if (reuse_weights) stats->weight_reuses++;
    if (state->selected_columns > stats->maximum_selected_columns) {
        stats->maximum_selected_columns = state->selected_columns;
    }

    int iteration = state->rwls_step >= INT_MAX
        ? INT_MAX
        : (int)state->rwls_step;
    const int first_drop = choose_rwls_remove_column(
        state,
        iteration,
        false
    );
    if (first_drop < 0) return false;
    remove_column(state, first_drop);
    state->timestamp[first_drop] = ++state->rwls_step;
    stats->remove_moves++;
    stats->restarts++;
    note_best_failed_state(state, target, stats);
    if (state->uncovered_rows == 0) {
        return collect_selection(state, candidate) == target;
    }

    const uint64_t step_limit = options->steps_per_family;
    for (uint64_t step = 0; step < step_limit; ++step) {
        if (
            options->budget_mode == PROGRESSIVE_RWLS_BUDGET_WALL &&
            (step & UINT64_C(31)) == 0 &&
            jump_deadline_reached(options->deadline_nanoseconds)
        ) {
            stats->deadline_reached = true;
            break;
        }
        stats->iterations++;
        state->rwls_step++;
        iteration = state->rwls_step >= INT_MAX
            ? INT_MAX
            : (int)state->rwls_step;

        const int remove = choose_rwls_remove_column(
            state,
            iteration,
            true
        );
        if (remove < 0) break;
        remove_column(state, remove);
        state->timestamp[remove] = state->rwls_step;
        state->add_tabu_until[remove] = rwls_tabu_until(
            iteration,
            options->tabu_tenure
        );
        stats->remove_moves++;

        const int row = choose_rwls_uncovered_row(
            state,
            options->seed ^ (uint64_t)state->rwls_step
        );
        if (row < 0) break;
        const int add = choose_rwls_add_column(state, row, iteration);
        if (add < 0) break;
        add_column(state, add);
        state->timestamp[add] = state->rwls_step;
        state->remove_tabu_until[add] = rwls_tabu_until(
            iteration,
            options->tabu_tenure
        );
        stats->add_moves++;

        if (state->uncovered_rows == 0) {
            return
                state->selected_columns == target &&
                collect_selection(state, candidate) == target;
        }
        raise_uncovered_penalties(state, stats);
        note_best_failed_state(state, target, stats);
    }
    return false;
}

static bool search_target_soft_pressure(
    FeasibilityJumpState *state,
    int incumbent_cardinality,
    const int *incumbent,
    const ProgressiveNativeFeasibilityJumpOptions *options,
    int *candidate,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    const int target = incumbent_cardinality - 1;
    const bool reuse_weights =
        state->rwls_weights_initialized &&
        state->rwls_policy == options->worker_policy;
    initialize_selection(
        state,
        incumbent_cardinality,
        incumbent,
        !reuse_weights
    );
    state->rwls_weights_initialized = true;
    state->rwls_policy = options->worker_policy;
    if (reuse_weights) stats->weight_reuses++;
    stats->restarts++;
    if (state->selected_columns > stats->maximum_selected_columns) {
        stats->maximum_selected_columns = state->selected_columns;
    }

    const uint64_t step_limit = options->steps_per_family;
    for (uint64_t step = 0; step < step_limit; ++step) {
        if (
            options->budget_mode == PROGRESSIVE_RWLS_BUDGET_WALL &&
            (step & UINT64_C(31)) == 0 &&
            jump_deadline_reached(options->deadline_nanoseconds)
        ) {
            stats->deadline_reached = true;
            break;
        }
        stats->iterations++;
        state->rwls_step++;
        const int iteration = state->rwls_step >= INT_MAX
            ? INT_MAX
            : (int)state->rwls_step;

        if (state->uncovered_rows == 0) {
            if (state->selected_columns == target) {
                return collect_selection(state, candidate) == target;
            }
            const int remove = choose_rwls_remove_column(
                state,
                iteration,
                true
            );
            if (remove < 0) break;
            const int64_t loss = state->private_loss[remove];
            if (state->soft_pressure <= loss) {
                state->soft_pressure = loss == INT64_MAX
                    ? INT64_MAX
                    : loss + 1;
                stats->soft_pressure_updates++;
            }
            remove_column(state, remove);
            state->timestamp[remove] = state->rwls_step;
            state->add_tabu_until[remove] = rwls_tabu_until(
                iteration,
                options->tabu_tenure
            );
            stats->remove_moves++;
            if (
                state->selected_columns == target &&
                state->uncovered_rows == 0
            ) {
                return collect_selection(state, candidate) == target;
            }
            note_best_failed_state(state, target, stats);
            continue;
        }

        const int row = choose_rwls_uncovered_row(
            state,
            options->seed ^ (uint64_t)state->rwls_step
        );
        if (row < 0) break;
        int add = -1;
        if (state->selected_columns < state->selected_capacity) {
            add = choose_rwls_add_column(state, row, iteration);
        }
        const int remove = state->selected_columns > target
            ? choose_rwls_remove_column(state, iteration, true)
            : -1;
        const int64_t add_score = add < 0
            ? INT64_MIN
            : state->uncovered_gain[add] - state->soft_pressure;
        const int64_t remove_score = remove < 0
            ? INT64_MIN
            : state->soft_pressure - state->private_loss[remove];

        if (remove_score > 0 && remove_score > add_score) {
            remove_column(state, remove);
            state->timestamp[remove] = state->rwls_step;
            state->add_tabu_until[remove] = rwls_tabu_until(
                iteration,
                options->tabu_tenure
            );
            stats->remove_moves++;
            note_best_failed_state(state, target, stats);
            continue;
        }
        if (add_score <= 0 && remove_score <= 0) {
            raise_uncovered_penalties(state, stats);
        }
        if (add < 0) {
            if (remove < 0) break;
            remove_column(state, remove);
            state->timestamp[remove] = state->rwls_step;
            state->add_tabu_until[remove] = rwls_tabu_until(
                iteration,
                options->tabu_tenure
            );
            stats->remove_moves++;
            note_best_failed_state(state, target, stats);
            continue;
        }
        add_column(state, add);
        state->timestamp[add] = state->rwls_step;
        state->remove_tabu_until[add] = rwls_tabu_until(
            iteration,
            options->tabu_tenure
        );
        stats->add_moves++;
        if (state->selected_columns > stats->maximum_selected_columns) {
            stats->maximum_selected_columns = state->selected_columns;
        }
        note_best_failed_state(state, target, stats);
    }
    return false;
}

static void reset_worker_result(
    FeasibilityJumpWorker *worker,
    int row_count
) {
    worker->improved = false;
    worker->attempted = false;
    memset(&worker->stats, 0, sizeof(worker->stats));
    worker->stats.best_uncovered_rows = row_count;
    worker->stats.best_failed_witness_row = -1;
}

static void search_restart_family_worker(
    uint64_t start,
    uint64_t end,
    uint64_t stride,
    int worker_id,
    int worker_count,
    void *data
) {
    FeasibilityJumpParallelContext *parallel =
        (FeasibilityJumpParallelContext *)data;
    if (worker_id == 0) {
        parallel->physical_threads_used = worker_count;
    }
    for (uint64_t family = start; family < end; family += stride) {
        if (family >= (uint64_t)parallel->family_count) break;
        FeasibilityJumpWorker *worker = &parallel->workers[family];
        ProgressiveNativeFeasibilityJumpOptions family_options =
            parallel->options;
        family_options.seed = jump_mix64(
            parallel->options.seed ^
            ((uint64_t)(unsigned int)(parallel->target_ordinal + 1) << 48) ^
            ((uint64_t)(unsigned int)(parallel->adaptive_round + 1) << 32) ^
            (uint64_t)(unsigned int)(family + 1)
        );
        if (parallel->adaptive_round > 0) {
            const uint64_t multiplier =
                parallel->adaptive_round + 1 < 4
                    ? (uint64_t)(parallel->adaptive_round + 1)
                    : UINT64_C(4);
            family_options.steps_per_family =
                family_options.steps_per_family >
                        UINT64_MAX / multiplier
                    ? UINT64_MAX
                    : family_options.steps_per_family * multiplier;
        }
        if (
            family_options.worker_policy ==
                PROGRESSIVE_RWLS_WORKER_MIXED
        ) {
            family_options.worker_policy =
                progressive_rwls_mixed_family_policy(
                    family_options.mixed_soft_worker_divisor,
                    family
                );
        }
        if (
            parallel->fixed_only &&
            family_options.worker_policy !=
                PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
        ) {
            continue;
        }
        worker->attempted = true;
        if (
            family_options.worker_policy ==
                PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
        ) {
            worker->improved = search_target_rwls(
                &worker->state,
                parallel->incumbent_cardinality,
                parallel->incumbent,
                &family_options,
                worker->candidate,
                &worker->stats
            );
        } else {
            worker->improved = search_target_soft_pressure(
                &worker->state,
                parallel->incumbent_cardinality,
                parallel->incumbent,
                &family_options,
                worker->candidate,
                &worker->stats
            );
        }
        /*
         * A successful family must not suppress later logical families in
         * the same physical shard.  Running the complete logical prefix is
         * what makes equal work deterministic across physical thread counts
         * and lets every persistent family carry its weights forward.
         */
        if (worker->stats.deadline_reached) break;
    }
}

static int candidate_symmetric_difference(
    const int *left,
    const int *right,
    int cardinality
) {
    int left_index = 0;
    int right_index = 0;
    int difference = 0;
    while (left_index < cardinality && right_index < cardinality) {
        if (left[left_index] == right[right_index]) {
            left_index++;
            right_index++;
        } else if (left[left_index] < right[right_index]) {
            difference++;
            left_index++;
        } else {
            difference++;
            right_index++;
        }
    }
    return
        difference +
        (cardinality - left_index) +
        (cardinality - right_index);
}

static void update_success_telemetry(
    const FeasibilityJumpWorker *workers,
    int family_count,
    int cardinality,
    int winner_family,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    int successful = 0;
    int distinct = 0;
    int minimum_difference = INT_MAX;
    int maximum_difference = 0;
    for (int family = 0; family < family_count; ++family) {
        if (!workers[family].attempted || !workers[family].improved) {
            continue;
        }
        successful++;
        if (
            workers[family].state.rwls_policy ==
                PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
        ) {
            stats->fixed_successful_candidates++;
        } else {
            stats->soft_successful_candidates++;
        }
        bool duplicate = false;
        for (int previous = 0; previous < family; ++previous) {
            if (
                !workers[previous].attempted ||
                !workers[previous].improved
            ) {
                continue;
            }
            const int difference = candidate_symmetric_difference(
                workers[family].candidate,
                workers[previous].candidate,
                cardinality
            );
            if (difference == 0) {
                duplicate = true;
            } else {
                if (difference < minimum_difference) {
                    minimum_difference = difference;
                }
                if (difference > maximum_difference) {
                    maximum_difference = difference;
                }
            }
        }
        if (!duplicate) distinct++;
    }
    stats->last_successful_candidates = successful;
    stats->last_successful_distinct_candidates = distinct;
    stats->last_successful_min_symmetric_difference =
        minimum_difference == INT_MAX ? 0 : minimum_difference;
    stats->last_successful_max_symmetric_difference = maximum_difference;
    if (winner_family >= 0 && winner_family < family_count) {
        stats->last_winning_policy =
            (int)workers[winner_family].state.rwls_policy;
        if (
            workers[winner_family].state.rwls_policy ==
                PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
        ) {
            stats->fixed_winning_targets++;
        } else {
            stats->soft_winning_targets++;
        }
    }
}

static void update_failure_witness_telemetry(
    const FeasibilityJumpWorker *workers,
    int family_count,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    stats->last_failed_witness_row = -1;
    stats->last_failed_witness_frequency = 0;
    stats->last_failed_witness_distinct_rows = 0;
    stats->last_failed_witness_observations = 0;
    stats->last_failed_selection_signature = 0;
    for (int worker = 0; worker < family_count; ++worker) {
        if (
            !workers[worker].attempted ||
            workers[worker].improved ||
            workers[worker].stats.best_uncovered_rows != 1 ||
            workers[worker].stats.best_failed_witness_row < 0
        ) {
            continue;
        }
        stats->last_failed_witness_observations++;
        const int row = workers[worker].stats.best_failed_witness_row;
        bool first_row_observation = true;
        int frequency = 0;
        for (int other = 0; other < family_count; ++other) {
            if (
                !workers[other].attempted ||
                workers[other].improved ||
                workers[other].stats.best_uncovered_rows != 1 ||
                workers[other].stats.best_failed_witness_row != row
            ) {
                continue;
            }
            frequency++;
            if (other < worker) first_row_observation = false;
        }
        if (first_row_observation) {
            stats->last_failed_witness_distinct_rows++;
        }
        if (
            frequency > stats->last_failed_witness_frequency ||
            (
                frequency == stats->last_failed_witness_frequency &&
                (
                    stats->last_failed_witness_row < 0 ||
                    row < stats->last_failed_witness_row
                )
            )
        ) {
            stats->last_failed_witness_row = row;
            stats->last_failed_witness_frequency = frequency;
            stats->last_failed_selection_signature =
                workers[worker].stats.best_failed_selection_signature;
        }
    }
}

static void merge_worker_stats(
    ProgressiveNativeFeasibilityJumpStats *total,
    const ProgressiveNativeFeasibilityJumpStats *worker
) {
    total->restarts += worker->restarts;
    total->iterations += worker->iterations;
    total->add_moves += worker->add_moves;
    total->remove_moves += worker->remove_moves;
    total->penalty_updates += worker->penalty_updates;
    total->weight_reuses += worker->weight_reuses;
    total->soft_pressure_updates += worker->soft_pressure_updates;
    if (
        worker->maximum_selected_columns >
        total->maximum_selected_columns
    ) {
        total->maximum_selected_columns =
            worker->maximum_selected_columns;
    }
    if (worker->best_uncovered_rows < total->best_uncovered_rows) {
        total->best_uncovered_rows = worker->best_uncovered_rows;
    }
    total->deadline_reached =
        total->deadline_reached || worker->deadline_reached;
}

static void collect_invocation_telemetry(
    ProgressiveNativeFeasibilityJumpStats *stats,
    const FeasibilityJumpWorker *workers,
    int family_count,
    int column_count
) {
    if (!stats || !workers || family_count <= 0 || column_count <= 0) {
        return;
    }
    for (int worker = 0; worker < family_count; ++worker) {
        const FeasibilityJumpState *state = &workers[worker].state;
        if (
            state->distinct_selected_columns >
            stats->maximum_worker_distinct_selected_columns
        ) {
            stats->maximum_worker_distinct_selected_columns =
                state->distinct_selected_columns;
        }
        if (state->maximum_row_penalty > stats->maximum_row_penalty) {
            stats->maximum_row_penalty = state->maximum_row_penalty;
        }
        if (
            state->maximum_uncovered_gain >
            stats->maximum_uncovered_gain
        ) {
            stats->maximum_uncovered_gain =
                state->maximum_uncovered_gain;
        }
        if (
            state->maximum_private_loss >
            stats->maximum_private_loss
        ) {
            stats->maximum_private_loss = state->maximum_private_loss;
        }
        if (state->rwls_step > stats->maximum_timestamp) {
            stats->maximum_timestamp = state->rwls_step;
        }
    }
    for (int column = 0; column < column_count; ++column) {
        for (int worker = 0; worker < family_count; ++worker) {
            if (workers[worker].state.ever_selected[column]) {
                stats->distinct_selected_columns++;
                break;
            }
        }
    }
}

static bool search_target_adaptive(
    FeasibilityJumpWorker *workers,
    int family_count,
    int physical_thread_count,
    int incumbent_cardinality,
    const int *incumbent,
    const ProgressiveNativeFeasibilityJumpOptions *options,
    int target_ordinal,
    bool recent_improvement,
    int *candidate,
    int *best_failed_uncovered_rows,
    ProgressiveNativeFeasibilityJumpStats *stats,
    bool fixed_only
) {
    const int target = incumbent_cardinality - 1;
    int best_failed_uncovered = workers[0].state.master->rows;
    for (
        int round = 0;
        round < options->max_adaptive_rounds;
        ++round
    ) {
        if (
            options->budget_mode == PROGRESSIVE_RWLS_BUDGET_WALL &&
            jump_deadline_reached(options->deadline_nanoseconds)
        ) {
            stats->deadline_reached = true;
            break;
        }
        for (int family = 0; family < family_count; ++family) {
            reset_worker_result(
                &workers[family],
                workers[family].state.master->rows
            );
        }
        FeasibilityJumpParallelContext parallel = {
            .workers = workers,
            .family_count = family_count,
            .physical_threads_used = 0,
            .incumbent_cardinality = incumbent_cardinality,
            .incumbent = incumbent,
            .options = *options,
            .target_ordinal = target_ordinal,
            .adaptive_round = round,
            .fixed_only = fixed_only
        };
        const bool parallel_completed = ccubes_parallel_for(
            0,
            (uint64_t)family_count,
            physical_thread_count,
            true,
            search_restart_family_worker,
            &parallel
        );
        if (!parallel_completed) {
            for (int family = 0; family < family_count; ++family) {
                reset_worker_result(
                    &workers[family],
                    workers[family].state.master->rows
                );
            }
            search_restart_family_worker(
                0,
                (uint64_t)family_count,
                1,
                0,
                1,
                &parallel
            );
            parallel.physical_threads_used = 1;
        }
        if (parallel.physical_threads_used > stats->threads_used) {
            stats->threads_used = parallel.physical_threads_used;
        }

        int attempted_families = 0;
        for (int family = 0; family < family_count; ++family) {
            if (workers[family].attempted) attempted_families++;
        }
        stats->restart_families += (uint64_t)attempted_families;
        if (round > 0) stats->adaptive_rounds++;
        bool improved = false;
        int winner_family = -1;
        int round_best_uncovered = workers[0].state.master->rows;
        for (int family = 0; family < family_count; ++family) {
            if (!workers[family].attempted) continue;
            merge_worker_stats(stats, &workers[family].stats);
            if (
                workers[family].state.rwls_policy ==
                    PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
            ) {
                stats->fixed_restart_families++;
                stats->fixed_iterations += workers[family].stats.iterations;
            } else {
                stats->soft_restart_families++;
                stats->soft_iterations += workers[family].stats.iterations;
            }
            if (
                workers[family].stats.best_uncovered_rows <
                round_best_uncovered
            ) {
                round_best_uncovered =
                    workers[family].stats.best_uncovered_rows;
            }
            if (!workers[family].improved) continue;
            if (!improved) {
                winner_family = family;
            }
            improved = true;
        }
        if (improved) {
            update_success_telemetry(
                workers,
                family_count,
                target,
                winner_family,
                stats
            );
            if (candidate != workers[winner_family].candidate) {
                memcpy(
                    candidate,
                    workers[winner_family].candidate,
                    (size_t)target * sizeof(int)
                );
            }
            if (best_failed_uncovered_rows) {
                *best_failed_uncovered_rows = 0;
            }
            return true;
        }
        update_failure_witness_telemetry(
            workers,
            family_count,
            stats
        );
        if (round_best_uncovered < best_failed_uncovered) {
            best_failed_uncovered = round_best_uncovered;
        }
        if (stats->deadline_reached) break;
        if (!recent_improvement) {
            break;
        }
    }
    if (best_failed_uncovered_rows) {
        *best_failed_uncovered_rows = best_failed_uncovered;
    }
    return false;
}

ProgressiveNativeFeasibilityJumpStatus
progressive_native_feasibility_jump_improve(
    const ProgressiveNativeMaster *master,
    int incumbent_cardinality,
    const int *incumbent_columns,
    const ProgressiveNativeFeasibilityJumpOptions *options,
    int **selected_columns,
    int *cardinality,
    ProgressiveNativeFeasibilityJumpStats *stats
) {
    if (selected_columns) *selected_columns = NULL;
    if (cardinality) *cardinality = -1;
    if (stats) {
        memset(stats, 0, sizeof(*stats));
        stats->best_failed_witness_row = -1;
        stats->last_failed_witness_row = -1;
        stats->last_winning_policy = -1;
    }
    if (
        !master ||
        master->rows <= 0 ||
        master->words <= 0 ||
        master->count <= 0 ||
        !master->coverage ||
        incumbent_cardinality <= 1 ||
        !incumbent_columns ||
        !options ||
        options->steps_per_family == 0 ||
        options->max_adaptive_rounds <= 0 ||
        options->tabu_tenure < 0 ||
        options->mixed_soft_worker_divisor < 2 ||
        options->mixed_soft_worker_divisor >
            RWLS_MAX_MIXED_DIVISOR ||
        options->worker_policy < PROGRESSIVE_RWLS_WORKER_FIXED_TARGET ||
        options->worker_policy > PROGRESSIVE_RWLS_WORKER_MIXED ||
        (
            options->budget_mode != PROGRESSIVE_RWLS_BUDGET_WORK &&
            options->budget_mode != PROGRESSIVE_RWLS_BUDGET_WALL
        ) ||
        (
            options->budget_mode == PROGRESSIVE_RWLS_BUDGET_WALL &&
            options->deadline_nanoseconds == 0
        ) ||
        !selected_columns ||
        !cardinality ||
        !stats ||
        !selection_is_cover(
            master,
            incumbent_cardinality,
            incumbent_columns
        )
    ) {
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_INVALID;
    }

    const uint64_t started = jump_monotonic_nanoseconds();
    uint64_t incidence_entries = 0;
    for (int column = 0; column < master->count; ++column) {
        const uint64_t *coverage = master->coverage +
            (size_t)column * (size_t)master->words;
        for (int word = 0; word < master->words; ++word) {
            incidence_entries += (uint64_t)__builtin_popcountll(
                coverage[word]
            );
        }
    }
    uint64_t shared_bytes = 0;
    uint64_t worker_bytes = 0;
    const int selected_capacity = options->worker_policy ==
            PROGRESSIVE_RWLS_WORKER_FIXED_TARGET
        ? incumbent_cardinality
        : master->count;
    if (
        !estimate_shared_memory(
            master,
            incidence_entries,
            &shared_bytes
        ) ||
        !estimate_worker_memory(
            master,
            incumbent_cardinality,
            selected_capacity,
            &worker_bytes
        )
    ) {
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    int physical_thread_count = ccubes_effective_thread_count(
        options->requested_threads
    );
    int family_count = options->logical_family_count;
    if (family_count == 0) {
        family_count = progressive_rwls_mixed_logical_family_count(
            physical_thread_count,
            options->mixed_soft_worker_divisor
        );
    } else if (
        family_count < 1 ||
        family_count > RWLS_MAX_LOGICAL_FAMILIES ||
        family_count % options->mixed_soft_worker_divisor != 0
    ) {
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_INVALID;
    }
    if (family_count <= 0) {
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_INVALID;
    }
    uint64_t allocated_bytes = 0;
    if (!estimate_peak_memory(
        master,
        shared_bytes,
        worker_bytes,
        family_count,
        incumbent_cardinality,
        &allocated_bytes
    )) {
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    stats->initial_cardinality = incumbent_cardinality;
    stats->final_cardinality = incumbent_cardinality;
    stats->best_uncovered_rows = master->rows;
    stats->last_failed_uncovered_rows = master->rows;
    stats->incidence_entries = incidence_entries;
    stats->bidirectional_incidence =
        should_use_packed_bidirectional_incidence(
            master,
            incidence_entries
        );
    stats->shared_cache_bytes = progressive_rwls_shared_cache_bytes();
    stats->packed_incidence_threshold_bytes =
        progressive_rwls_packed_incidence_threshold_bytes();
    stats->shared_cache_source = progressive_rwls_shared_cache_source();
    stats->packed_incidence_invocations =
        stats->bidirectional_incidence ? 1 : 0;
    stats->dense_incidence_invocations =
        stats->bidirectional_incidence ? 0 : 1;
    stats->column_incidence_index_bytes =
        stats->bidirectional_incidence ? (int)sizeof(uint16_t) : 0;
    stats->allocated_bytes = allocated_bytes;
    stats->shared_allocated_bytes = shared_bytes;
    stats->per_family_allocated_bytes = worker_bytes;
    if (physical_thread_count > family_count) {
        physical_thread_count = family_count;
    }
    stats->threads_used = 0;
    stats->budget_mode = options->budget_mode;
    if (
        options->memory_budget_bytes > 0 &&
        allocated_bytes > options->memory_budget_bytes
    ) {
        stats->elapsed_nanoseconds =
            jump_monotonic_nanoseconds() - started;
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_MEMORY_LIMIT;
    }

    FeasibilityJumpIncidence incidence;
    if (!build_incidence(&incidence, master, incidence_entries)) {
        destroy_incidence(&incidence);
        stats->elapsed_nanoseconds =
            jump_monotonic_nanoseconds() - started;
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    for (int column = 0; column < master->count; ++column) {
        if (incidence.compulsory_columns[column]) {
            stats->compulsory_columns++;
        }
    }
    FeasibilityJumpWorker *workers =
        (FeasibilityJumpWorker *)calloc(
            (size_t)family_count,
            sizeof(FeasibilityJumpWorker)
        );
    int *best = (int *)malloc(
        (size_t)incumbent_cardinality * sizeof(int)
    );
    if (!workers || !best) {
        free(workers);
        free(best);
        destroy_incidence(&incidence);
        stats->elapsed_nanoseconds =
            jump_monotonic_nanoseconds() - started;
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    int allocated_families = 0;
    for (; allocated_families < family_count; ++allocated_families) {
        FeasibilityJumpWorker *worker = &workers[allocated_families];
        if (!allocate_state(
            &worker->state,
            master,
            &incidence,
            selected_capacity
        )) {
            break;
        }
        worker->candidate = (int *)malloc(
            (size_t)incumbent_cardinality * sizeof(int)
        );
        if (!worker->candidate) {
            destroy_state(&worker->state);
            break;
        }
    }
    if (allocated_families != family_count) {
        for (int family = 0; family < allocated_families; ++family) {
            free(workers[family].candidate);
            destroy_state(&workers[family].state);
        }
        free(workers);
        free(best);
        destroy_incidence(&incidence);
        stats->elapsed_nanoseconds =
            jump_monotonic_nanoseconds() - started;
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    stats->logical_family_states = family_count;
    memcpy(
        best,
        incumbent_columns,
        (size_t)incumbent_cardinality * sizeof(int)
    );
    int best_cardinality = incumbent_cardinality;
    uint64_t work_epoch = 0;
    bool search_finished = false;
    while (best_cardinality > 1 && !search_finished) {
        const int epoch_start_improvements = stats->targets_improved;
        stats->work_epochs++;
        while (best_cardinality > 1) {
            stats->targets_attempted++;
            int best_failed_uncovered = master->rows;
            stats->deadline_reached = false;
            ProgressiveNativeFeasibilityJumpOptions search_options =
                *options;
            if (work_epoch > 0) {
                search_options.seed = jump_mix64(
                    options->seed ^
                    (work_epoch << 32) ^
                    UINT64_C(0x726f756e642d6570)
                );
            }
            bool improved = search_target_adaptive(
                workers,
                family_count,
                physical_thread_count,
                best_cardinality,
                best,
                &search_options,
                stats->targets_attempted - 1,
                stats->targets_improved > 0,
                workers[0].candidate,
                &best_failed_uncovered,
                stats,
                false
            );
            if (
                !improved &&
                options->worker_policy ==
                    PROGRESSIVE_RWLS_WORKER_MIXED &&
                !stats->deadline_reached
            ) {
                int recovery_failed_uncovered = master->rows;
                improved = search_target_adaptive(
                    workers,
                    family_count,
                    physical_thread_count,
                    best_cardinality,
                    best,
                    &search_options,
                    stats->targets_attempted - 1,
                    stats->targets_improved > 0,
                    workers[0].candidate,
                    &recovery_failed_uncovered,
                    stats,
                    true
                );
                if (recovery_failed_uncovered < best_failed_uncovered) {
                    best_failed_uncovered = recovery_failed_uncovered;
                }
            }
            if (!improved) {
                stats->last_failed_uncovered_rows =
                    best_failed_uncovered;
                if (
                    !stats->deadline_reached &&
                    stats->targets_improved > epoch_start_improvements
                ) {
                    stats->productive_epoch_extensions++;
                    work_epoch++;
                } else {
                    search_finished = true;
                }
                break;
            }
            best_cardinality--;
            memcpy(
                best,
                workers[0].candidate,
                (size_t)best_cardinality * sizeof(int)
            );
            stats->targets_improved++;
            stats->final_cardinality = best_cardinality;
        }
    }
    int *returned = (int *)malloc(
        (size_t)best_cardinality * sizeof(int)
    );
    if (!returned) {
        free(best);
        for (int family = 0; family < family_count; ++family) {
            free(workers[family].candidate);
            destroy_state(&workers[family].state);
        }
        free(workers);
        destroy_incidence(&incidence);
        stats->elapsed_nanoseconds =
            jump_monotonic_nanoseconds() - started;
        return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OUT_OF_MEMORY;
    }
    memcpy(
        returned,
        best,
        (size_t)best_cardinality * sizeof(int)
    );
    *selected_columns = returned;
    *cardinality = best_cardinality;
    stats->final_cardinality = best_cardinality;
    collect_invocation_telemetry(
        stats,
        workers,
        family_count,
        master->count
    );
    stats->elapsed_nanoseconds = jump_monotonic_nanoseconds() - started;
    free(best);
    for (int family = 0; family < family_count; ++family) {
        free(workers[family].candidate);
        destroy_state(&workers[family].state);
    }
    free(workers);
    destroy_incidence(&incidence);
    return PROGRESSIVE_NATIVE_FEASIBILITY_JUMP_OK;
}
