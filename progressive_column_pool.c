/*
    Copyright (c) 2016–2026, Adrian Dusa
    All rights reserved.

    License: Academic Non-Commercial License (see LICENSE file for details).
    SPDX-License-Identifier: LicenseRef-ANCL-AdrianDusa
*/

#include "progressive_column_pool.h"
#include "file_replace.h"
#include "progressive_validation.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROGRESSIVE_COLUMN_POOL_VERSION UINT32_C(1)
#define PROGRESSIVE_COLUMN_POOL_ENDIAN UINT32_C(0x01020304)
#define PROGRESSIVE_COLUMN_POOL_HEADER_BYTES UINT64_C(56)

static const unsigned char progressive_column_pool_magic[8] = {
    'C', 'C', 'P', 'C', 'O', 'L', '1', '\0'
};

static bool checked_product(
    size_t left,
    size_t right,
    size_t *product
) {
    if (!product || (left > 0 && right > SIZE_MAX / left)) return false;
    *product = left * right;
    return true;
}

static uint64_t hash_bytes(
    uint64_t hash,
    const void *data,
    size_t bytes
) {
    const unsigned char *values = (const unsigned char *)data;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= (uint64_t)values[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t progressive_column_pool_fingerprint(
    const WeightedMMCSPreparedRows *prepared
) {
    if (
        !prepared ||
        prepared->ninputs <= 0 ||
        prepared->row_words <= 0 ||
        prepared->on_rows <= 0 ||
        prepared->off_rows <= 0 ||
        !prepared->on_bits ||
        !prepared->off_bits
    ) {
        return 0;
    }

    uint64_t hash = UINT64_C(1469598103934665603);
    const uint32_t geometry[] = {
        (uint32_t)prepared->ninputs,
        (uint32_t)prepared->row_words,
        (uint32_t)prepared->on_rows,
        (uint32_t)prepared->off_rows
    };
    hash = hash_bytes(hash, geometry, sizeof(geometry));

    size_t on_words = 0;
    size_t off_words = 0;
    if (
        !checked_product(
            (size_t)prepared->on_rows,
            (size_t)prepared->row_words,
            &on_words
        ) ||
        !checked_product(
            (size_t)prepared->off_rows,
            (size_t)prepared->row_words,
            &off_words
        )
    ) {
        return 0;
    }
    hash = hash_bytes(
        hash,
        prepared->on_bits,
        on_words * sizeof(uint64_t)
    );
    hash = hash_bytes(
        hash,
        prepared->off_bits,
        off_words * sizeof(uint64_t)
    );
    return hash;
}

static bool write_exact(
    FILE *file,
    const void *data,
    size_t bytes
) {
    return bytes == 0 || fwrite(data, 1, bytes, file) == bytes;
}

static bool read_exact(
    FILE *file,
    void *data,
    size_t bytes
) {
    return bytes == 0 || fread(data, 1, bytes, file) == bytes;
}

ProgressiveColumnPoolStatus progressive_column_pool_save(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    int words,
    int count,
    const int *patterns,
    const uint64_t *coverage,
    int incumbent_cardinality,
    const int *incumbent_solution
) {
    if (
        !path ||
        !*path ||
        !prepared ||
        words <= 0 ||
        count <= 0 ||
        !patterns ||
        !coverage ||
        incumbent_cardinality <= 0 ||
        !incumbent_solution
    ) {
        return PROGRESSIVE_COLUMN_POOL_INVALID;
    }
    uint64_t fingerprint =
        progressive_column_pool_fingerprint(prepared);
    if (fingerprint == 0) return PROGRESSIVE_COLUMN_POOL_INVALID;

    size_t pattern_values = 0;
    size_t coverage_values = 0;
    size_t incumbent_values = 0;
    if (
        !checked_product(
            (size_t)count,
            (size_t)prepared->ninputs,
            &pattern_values
        ) ||
        !checked_product(
            (size_t)count,
            (size_t)words,
            &coverage_values
        ) ||
        !checked_product(
            (size_t)incumbent_cardinality,
            (size_t)prepared->ninputs,
            &incumbent_values
        )
    ) {
        return PROGRESSIVE_COLUMN_POOL_INVALID;
    }

    unsigned char *encoded_patterns = (unsigned char *)malloc(pattern_values);
    unsigned char *encoded_incumbent = (unsigned char *)malloc(
        incumbent_values
    );
    if (!encoded_patterns || !encoded_incumbent) {
        free(encoded_patterns);
        free(encoded_incumbent);
        return PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < pattern_values; ++i) {
        if (patterns[i] < 0 || patterns[i] > 2) {
            free(encoded_patterns);
            free(encoded_incumbent);
            return PROGRESSIVE_COLUMN_POOL_INVALID;
        }
        encoded_patterns[i] = (unsigned char)patterns[i];
    }
    for (size_t i = 0; i < incumbent_values; ++i) {
        if (incumbent_solution[i] < 0 || incumbent_solution[i] > 2) {
            free(encoded_patterns);
            free(encoded_incumbent);
            return PROGRESSIVE_COLUMN_POOL_INVALID;
        }
        encoded_incumbent[i] = (unsigned char)incumbent_solution[i];
    }

    size_t path_length = strlen(path);
    char *temporary_path = (char *)malloc(path_length + 10u);
    if (!temporary_path) {
        free(encoded_patterns);
        free(encoded_incumbent);
        return PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY;
    }
    snprintf(
        temporary_path,
        path_length + 10u,
        "%s.tmp",
        path
    );

    FILE *file = fopen(temporary_path, "wb");
    if (!file) {
        free(temporary_path);
        free(encoded_patterns);
        free(encoded_incumbent);
        return PROGRESSIVE_COLUMN_POOL_IO_ERROR;
    }

    uint32_t version = PROGRESSIVE_COLUMN_POOL_VERSION;
    uint32_t endian = PROGRESSIVE_COLUMN_POOL_ENDIAN;
    uint32_t ninputs = (uint32_t)prepared->ninputs;
    uint32_t on_rows = (uint32_t)prepared->on_rows;
    uint32_t off_rows = (uint32_t)prepared->off_rows;
    uint32_t stored_words = (uint32_t)words;
    uint64_t stored_count = (uint64_t)count;
    uint64_t stored_incumbent = (uint64_t)incumbent_cardinality;
    bool ok =
        write_exact(file, progressive_column_pool_magic, 8) &&
        write_exact(file, &version, sizeof(version)) &&
        write_exact(file, &endian, sizeof(endian)) &&
        write_exact(file, &ninputs, sizeof(ninputs)) &&
        write_exact(file, &on_rows, sizeof(on_rows)) &&
        write_exact(file, &off_rows, sizeof(off_rows)) &&
        write_exact(file, &stored_words, sizeof(stored_words)) &&
        write_exact(file, &stored_count, sizeof(stored_count)) &&
        write_exact(file, &stored_incumbent, sizeof(stored_incumbent)) &&
        write_exact(file, &fingerprint, sizeof(fingerprint)) &&
        write_exact(file, encoded_patterns, pattern_values) &&
        write_exact(
            file,
            coverage,
            coverage_values * sizeof(uint64_t)
        ) &&
        write_exact(file, encoded_incumbent, incumbent_values) &&
        fflush(file) == 0;
    if (fclose(file) != 0) ok = false;
    if (ok && !ccubes_replace_file(temporary_path, path)) ok = false;
    if (!ok) remove(temporary_path);

    free(temporary_path);
    free(encoded_patterns);
    free(encoded_incumbent);
    return ok
        ? PROGRESSIVE_COLUMN_POOL_OK
        : PROGRESSIVE_COLUMN_POOL_IO_ERROR;
}

ProgressiveColumnPoolStatus progressive_column_pool_save_complete_chart(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    const PIstorage *source,
    int ninputs,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask,
    int implicant_words,
    int incumbent_cardinality,
    const int *incumbent_solution
) {
    if (
        !path ||
        !*path ||
        !prepared ||
        !source ||
        ninputs <= 0 ||
        prepared->ninputs != ninputs ||
        prepared->on_rows <= 0 ||
        prepared->off_rows <= 0 ||
        source->foundPI <= 0 ||
        !source->implicants_pos ||
        !source->implicants_val ||
        !word_index ||
        !bit_index ||
        !shifted_mask ||
        implicant_words <= 0 ||
        incumbent_cardinality <= 0 ||
        !incumbent_solution
    ) {
        return PROGRESSIVE_COLUMN_POOL_INVALID;
    }

    int words = (prepared->on_rows + 63) / 64;
    size_t pattern_values = 0;
    size_t coverage_values = 0;
    if (
        words <= 0 ||
        !checked_product(
            (size_t)source->foundPI,
            (size_t)ninputs,
            &pattern_values
        ) ||
        !checked_product(
            (size_t)source->foundPI,
            (size_t)words,
            &coverage_values
        )
    ) {
        return PROGRESSIVE_COLUMN_POOL_INVALID;
    }

    int *patterns = (int *)calloc(pattern_values, sizeof(int));
    uint64_t *coverage = (uint64_t *)calloc(
        coverage_values,
        sizeof(uint64_t)
    );
    if (!patterns || !coverage) {
        free(patterns);
        free(coverage);
        return PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY;
    }

    ProgressiveColumnPoolStatus status = PROGRESSIVE_COLUMN_POOL_OK;
    for (int column = 0; column < source->foundPI; ++column) {
        int *pattern = &patterns[(size_t)column * (size_t)ninputs];
        for (int input = 0; input < ninputs; ++input) {
            int word = word_index[input];
            if (word < 0 || word >= implicant_words) {
                status = PROGRESSIVE_COLUMN_POOL_INVALID;
                break;
            }
            if (
                source->implicants_pos[
                    (size_t)column * (size_t)implicant_words +
                    (size_t)word
                ] & shifted_mask[input]
            ) {
                pattern[input] = 1 + (int)(
                    (
                        source->implicants_val[
                            (size_t)column * (size_t)implicant_words +
                            (size_t)word
                        ] >> bit_index[input]
                    ) & UINT64_C(1)
                );
            }
        }
        if (status != PROGRESSIVE_COLUMN_POOL_OK) break;
        if (!progressive_pattern_coverage_prepared(
            prepared,
            pattern,
            &coverage[(size_t)column * (size_t)words]
        )) {
            status = PROGRESSIVE_COLUMN_POOL_MISMATCH;
            break;
        }
    }

    if (
        status == PROGRESSIVE_COLUMN_POOL_OK &&
        !progressive_solution_is_valid_prepared(
            prepared,
            incumbent_cardinality,
            incumbent_solution
        )
    ) {
        status = PROGRESSIVE_COLUMN_POOL_MISMATCH;
    }
    if (status == PROGRESSIVE_COLUMN_POOL_OK) {
        status = progressive_column_pool_save(
            path,
            prepared,
            words,
            source->foundPI,
            patterns,
            coverage,
            incumbent_cardinality,
            incumbent_solution
        );
    }

    free(patterns);
    free(coverage);
    return status;
}

ProgressiveColumnPoolStatus progressive_column_pool_load(
    const char *path,
    const WeightedMMCSPreparedRows *prepared,
    int expected_words,
    ProgressiveColumnPool *pool
) {
    if (
        !path ||
        !*path ||
        !prepared ||
        expected_words <= 0 ||
        !pool
    ) {
        return PROGRESSIVE_COLUMN_POOL_INVALID;
    }
    memset(pool, 0, sizeof(*pool));

    FILE *file = fopen(path, "rb");
    if (!file) return PROGRESSIVE_COLUMN_POOL_IO_ERROR;

    unsigned char magic[8];
    uint32_t version = 0;
    uint32_t endian = 0;
    uint32_t ninputs = 0;
    uint32_t on_rows = 0;
    uint32_t off_rows = 0;
    uint32_t words = 0;
    uint64_t count = 0;
    uint64_t incumbent_cardinality = 0;
    uint64_t fingerprint = 0;
    bool ok =
        read_exact(file, magic, sizeof(magic)) &&
        read_exact(file, &version, sizeof(version)) &&
        read_exact(file, &endian, sizeof(endian)) &&
        read_exact(file, &ninputs, sizeof(ninputs)) &&
        read_exact(file, &on_rows, sizeof(on_rows)) &&
        read_exact(file, &off_rows, sizeof(off_rows)) &&
        read_exact(file, &words, sizeof(words)) &&
        read_exact(file, &count, sizeof(count)) &&
        read_exact(
            file,
            &incumbent_cardinality,
            sizeof(incumbent_cardinality)
        ) &&
        read_exact(file, &fingerprint, sizeof(fingerprint));
    if (!ok) {
        fclose(file);
        return PROGRESSIVE_COLUMN_POOL_IO_ERROR;
    }

    uint64_t expected_fingerprint =
        progressive_column_pool_fingerprint(prepared);
    if (
        memcmp(magic, progressive_column_pool_magic, 8) != 0 ||
        version != PROGRESSIVE_COLUMN_POOL_VERSION ||
        endian != PROGRESSIVE_COLUMN_POOL_ENDIAN ||
        ninputs != (uint32_t)prepared->ninputs ||
        on_rows != (uint32_t)prepared->on_rows ||
        off_rows != (uint32_t)prepared->off_rows ||
        words != (uint32_t)expected_words ||
        count == 0 ||
        count > (uint64_t)INT32_MAX ||
        incumbent_cardinality == 0 ||
        incumbent_cardinality > count ||
        incumbent_cardinality > (uint64_t)INT32_MAX ||
        fingerprint == 0 ||
        fingerprint != expected_fingerprint
    ) {
        fclose(file);
        return PROGRESSIVE_COLUMN_POOL_MISMATCH;
    }

    size_t pattern_values = 0;
    size_t coverage_values = 0;
    size_t incumbent_values = 0;
    if (
        !checked_product(
            (size_t)count,
            (size_t)ninputs,
            &pattern_values
        ) ||
        !checked_product(
            (size_t)count,
            (size_t)words,
            &coverage_values
        ) ||
        !checked_product(
            (size_t)incumbent_cardinality,
            (size_t)ninputs,
            &incumbent_values
        )
    ) {
        fclose(file);
        return PROGRESSIVE_COLUMN_POOL_MISMATCH;
    }
    uint64_t expected_size =
        PROGRESSIVE_COLUMN_POOL_HEADER_BYTES +
        (uint64_t)pattern_values +
        (uint64_t)coverage_values * sizeof(uint64_t) +
        (uint64_t)incumbent_values;
    if (
        fseek(file, 0, SEEK_END) != 0 ||
        ftell(file) < 0 ||
        (uint64_t)ftell(file) != expected_size ||
        fseek(
            file,
            (long)PROGRESSIVE_COLUMN_POOL_HEADER_BYTES,
            SEEK_SET
        ) != 0
    ) {
        fclose(file);
        return PROGRESSIVE_COLUMN_POOL_MISMATCH;
    }

    unsigned char *encoded_patterns = (unsigned char *)malloc(
        pattern_values
    );
    unsigned char *encoded_incumbent = (unsigned char *)malloc(
        incumbent_values
    );
    int *patterns = (int *)malloc(pattern_values * sizeof(int));
    int *incumbent_solution = (int *)malloc(
        incumbent_values * sizeof(int)
    );
    uint64_t *coverage = (uint64_t *)malloc(
        coverage_values * sizeof(uint64_t)
    );
    if (
        !encoded_patterns ||
        !encoded_incumbent ||
        !patterns ||
        !incumbent_solution ||
        !coverage
    ) {
        free(encoded_patterns);
        free(encoded_incumbent);
        free(patterns);
        free(incumbent_solution);
        free(coverage);
        fclose(file);
        return PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY;
    }
    ok =
        read_exact(file, encoded_patterns, pattern_values) &&
        read_exact(
            file,
            coverage,
            coverage_values * sizeof(uint64_t)
        ) &&
        read_exact(file, encoded_incumbent, incumbent_values);
    if (fclose(file) != 0) ok = false;
    if (!ok) {
        free(encoded_patterns);
        free(encoded_incumbent);
        free(patterns);
        free(incumbent_solution);
        free(coverage);
        return PROGRESSIVE_COLUMN_POOL_IO_ERROR;
    }

    for (size_t i = 0; i < pattern_values; ++i) {
        if (encoded_patterns[i] > 2) {
            free(encoded_patterns);
            free(encoded_incumbent);
            free(patterns);
            free(incumbent_solution);
            free(coverage);
            return PROGRESSIVE_COLUMN_POOL_MISMATCH;
        }
        patterns[i] = (int)encoded_patterns[i];
    }
    free(encoded_patterns);
    for (size_t i = 0; i < incumbent_values; ++i) {
        if (encoded_incumbent[i] > 2) {
            free(encoded_incumbent);
            free(patterns);
            free(incumbent_solution);
            free(coverage);
            return PROGRESSIVE_COLUMN_POOL_MISMATCH;
        }
        incumbent_solution[i] = (int)encoded_incumbent[i];
    }
    free(encoded_incumbent);

    int tail_bits = prepared->on_rows % 64;
    uint64_t tail_mask = tail_bits == 0
        ? UINT64_MAX
        : (UINT64_C(1) << tail_bits) - UINT64_C(1);
    for (uint64_t column = 0; column < count; ++column) {
        uint64_t *column_coverage = &coverage[
            (size_t)column * (size_t)words
        ];
        bool nonempty = false;
        for (uint32_t word = 0; word < words; ++word) {
            nonempty = nonempty || column_coverage[word] != 0;
        }
        if (
            !nonempty ||
            (
                tail_bits != 0 &&
                (column_coverage[words - 1] & ~tail_mask) != 0
            )
        ) {
            free(patterns);
            free(incumbent_solution);
            free(coverage);
            return PROGRESSIVE_COLUMN_POOL_MISMATCH;
        }
    }

    pool->ninputs = (int)ninputs;
    pool->rows = (int)on_rows;
    pool->words = (int)words;
    pool->count = (int)count;
    pool->incumbent_cardinality = (int)incumbent_cardinality;
    pool->patterns = patterns;
    pool->coverage = coverage;
    pool->incumbent_solution = incumbent_solution;
    return PROGRESSIVE_COLUMN_POOL_OK;
}

void progressive_column_pool_destroy(ProgressiveColumnPool *pool) {
    if (!pool) return;
    free(pool->patterns);
    free(pool->coverage);
    free(pool->incumbent_solution);
    memset(pool, 0, sizeof(*pool));
}

const char *progressive_column_pool_status_name(
    ProgressiveColumnPoolStatus status
) {
    switch (status) {
        case PROGRESSIVE_COLUMN_POOL_OK:
            return "ok";
        case PROGRESSIVE_COLUMN_POOL_INVALID:
            return "invalid";
        case PROGRESSIVE_COLUMN_POOL_IO_ERROR:
            return "io-error";
        case PROGRESSIVE_COLUMN_POOL_OUT_OF_MEMORY:
            return "out-of-memory";
        case PROGRESSIVE_COLUMN_POOL_MISMATCH:
            return "input-mismatch";
    }
    return "unknown";
}
