#ifndef CCUBES_COVERAGE_EXCLUSION_H
#define CCUBES_COVERAGE_EXCLUSION_H
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint64_t *coverage;
    int words, count;
    size_t capacity;
    int *slots;
} CoverageExclusion;

static inline uint64_t coverage_exclusion_hash(const uint64_t *coverage, int words)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int i = 0; i < words; ++i) {
        hash ^= coverage[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static inline int coverage_exclusion_init(CoverageExclusion *index,
                                          const uint64_t *coverage, int count, int words)
{
    *index = (CoverageExclusion){.coverage = coverage, .count = count, .words = words};
    if (!count) return 1;
    if (!coverage || count < 0 || words <= 0) return 0;
    size_t capacity = 2;
    while (capacity < (size_t)count * 2) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(int)) return 0;
    index->slots = malloc(capacity * sizeof(int));
    if (!index->slots) return 0;
    index->capacity = capacity;
    for (size_t i = 0; i < capacity; ++i) index->slots[i] = -1;
    for (int c = 0; c < count; ++c) {
        const uint64_t *bits = coverage + (size_t)c * words;
        size_t slot = coverage_exclusion_hash(bits, words) & (capacity - 1);
        while (index->slots[slot] >= 0) {
            if (!memcmp(bits, coverage + (size_t)index->slots[slot] * words,
                        (size_t)words * sizeof(uint64_t))) break;
            slot = (slot + 1) & (capacity - 1);
        }
        index->slots[slot] = c;
    }
    return 1;
}

static inline int coverage_exclusion_contains(const CoverageExclusion *index,
                                              const uint64_t *coverage)
{
    if (!index || !index->capacity) return 0;
    size_t slot = coverage_exclusion_hash(coverage, index->words) & (index->capacity - 1);
    while (index->slots[slot] >= 0) {
        if (!memcmp(coverage, index->coverage + (size_t)index->slots[slot] * index->words,
                    (size_t)index->words * sizeof(uint64_t))) return 1;
        slot = (slot + 1) & (index->capacity - 1);
    }
    return 0;
}
#endif
