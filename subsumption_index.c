#include "subsumption_index.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

/*
 * David Stafford's Mix13 64-bit finalizer, published as
 * mix64variant13 in Steele, Lea, and Flood's SplitMix paper:
 * https://doi.org/10.1145/2660193.2660195
 *
 * CCubes uses it only to avalanche the accumulated cube hash before a
 * power-of-two table mask selects the slot. This keeps structured high input
 * bits from clustering in a few low-bit buckets; it is not used as a random
 * number generator or for cryptographic hashing.
 */
static uint64_t hash_finalize(uint64_t hash) {
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash *= UINT64_C(0x94d049bb133111eb);
    hash ^= hash >> 31;
    return hash;
}

static uint64_t cube_hash(
    const uint64_t *fixed_bits,
    const uint64_t *value_bits,
    int words,
    int removed_word,
    uint64_t removed_mask
) {
    const uint64_t FNV_OFFSET = UINT64_C(1469598103934665603);
    const uint64_t FNV_PRIME = UINT64_C(1099511628211);
    uint64_t hash = FNV_OFFSET;

    for (int w = 0; w < words; ++w) {
        uint64_t fixed = fixed_bits[w];
        uint64_t value = value_bits[w];
        if (w == removed_word) {
            fixed &= ~removed_mask;
            value &= ~removed_mask;
        }

        hash ^= fixed;
        hash *= FNV_PRIME;
        hash ^= value;
        hash *= FNV_PRIME;
    }

    return hash_finalize(hash);
}

static uint64_t fixed_hash(
    const uint64_t *fixed_bits,
    int words,
    int removed_word,
    uint64_t removed_mask
) {
    const uint64_t FNV_OFFSET = UINT64_C(1469598103934665603);
    const uint64_t FNV_PRIME = UINT64_C(1099511628211);
    uint64_t hash = FNV_OFFSET;

    for (int w = 0; w < words; ++w) {
        uint64_t fixed = fixed_bits[w];
        if (w == removed_word) fixed &= ~removed_mask;
        hash ^= fixed;
        hash *= FNV_PRIME;
    }

    return hash_finalize(hash);
}

static bool cube_equal(
    const SubsumptionIndex *index,
    size_t key_index,
    const uint64_t *fixed_bits,
    const uint64_t *value_bits,
    int removed_word,
    uint64_t removed_mask
) {
    const uint64_t *stored_fixed =
        &index->fixed_keys[key_index * (size_t)index->words];
    const uint64_t *stored_value =
        &index->value_keys[key_index * (size_t)index->words];

    for (int w = 0; w < index->words; ++w) {
        uint64_t fixed = fixed_bits[w];
        uint64_t value = value_bits[w];
        if (w == removed_word) {
            fixed &= ~removed_mask;
            value &= ~removed_mask;
        }
        if (stored_fixed[w] != fixed || stored_value[w] != value) {
            return false;
        }
    }

    return true;
}

static bool cube_lookup(
    const SubsumptionIndex *index,
    const uint64_t *fixed_bits,
    const uint64_t *value_bits,
    int removed_word,
    uint64_t removed_mask
) {
    if (
        !index ||
        !index->slots ||
        !index->fixed_keys ||
        !index->value_keys ||
        index->table_size == 0 ||
        index->words <= 0
    ) {
        return false;
    }

    size_t mask = index->table_size - 1u;
    size_t slot = (size_t)(
        cube_hash(
            fixed_bits,
            value_bits,
            index->words,
            removed_word,
            removed_mask
        ) &
        (uint64_t)mask
    );

    while (index->slots[slot] >= 0) {
        if (
            cube_equal(
                index,
                (size_t)index->slots[slot],
                fixed_bits,
                value_bits,
                removed_word,
                removed_mask
            )
        ) {
            return true;
        }
        slot = (slot + 1u) & mask;
    }

    return false;
}

static bool fixed_equal(
    const SubsumptionIndex *index,
    size_t key_index,
    const uint64_t *fixed_bits,
    int removed_word,
    uint64_t removed_mask
) {
    const uint64_t *stored_fixed =
        &index->fixed_keys[key_index * (size_t)index->words];

    for (int w = 0; w < index->words; ++w) {
        uint64_t fixed = fixed_bits[w];
        if (w == removed_word) fixed &= ~removed_mask;
        if (stored_fixed[w] != fixed) return false;
    }

    return true;
}

static bool fixed_lookup(
    const SubsumptionIndex *index,
    const uint64_t *fixed_bits,
    int removed_word,
    uint64_t removed_mask
) {
    if (
        !index ||
        !index->support_slots ||
        !index->fixed_keys ||
        index->table_size == 0 ||
        index->words <= 0
    ) {
        return false;
    }

    size_t mask = index->table_size - 1u;
    size_t slot = (size_t)(
        fixed_hash(
            fixed_bits,
            index->words,
            removed_word,
            removed_mask
        ) &
        (uint64_t)mask
    );

    while (index->support_slots[slot] >= 0) {
        if (
            fixed_equal(
                index,
                (size_t)index->support_slots[slot],
                fixed_bits,
                removed_word,
                removed_mask
            )
        ) {
            return true;
        }
        slot = (slot + 1u) & mask;
    }

    return false;
}

bool subsumption_index_build(
    SubsumptionIndex *index,
    const uint64_t *fixed_records,
    const uint64_t *value_records,
    int records,
    int words
) {
    if (!index || records < 0 || words <= 0) return false;
    if (records > 0 && (!fixed_records || !value_records)) return false;

    memset(index, 0, sizeof(*index));
    index->words = words;

    size_t requested = records > 0 ? (size_t)records * 2u : 16u;
    if (records > 0 && requested / 2u != (size_t)records) return false;

    size_t table_size = 16u;
    while (table_size < requested) {
        if (table_size > SIZE_MAX / 2u) return false;
        table_size <<= 1u;
    }
    size_t key_records = records > 0 ? (size_t)records : 1u;
    if (
        key_records > SIZE_MAX / (size_t)words ||
        key_records * (size_t)words > SIZE_MAX / sizeof(uint64_t)
    ) {
        return false;
    }

    index->slots = (int *)malloc(table_size * sizeof(int));
    index->support_slots = (int *)malloc(table_size * sizeof(int));
    index->fixed_keys = (uint64_t *)calloc(
        key_records * (size_t)words,
        sizeof(uint64_t)
    );

    index->value_keys = (uint64_t *)calloc(
        key_records * (size_t)words,
        sizeof(uint64_t)
    );

    if (
        !index->slots ||
        !index->support_slots ||
        !index->fixed_keys ||
        !index->value_keys
    ) {
        subsumption_index_destroy(index);
        return false;
    }

    for (size_t i = 0; i < table_size; ++i) {
        index->slots[i] = -1;
        index->support_slots[i] = -1;
    }

    index->table_size = table_size;

    size_t mask = table_size - 1u;
    for (int record = 0; record < records; ++record) {
        const uint64_t *fixed =
            &fixed_records[(size_t)record * (size_t)words];
        const uint64_t *value =
            &value_records[(size_t)record * (size_t)words];
        size_t slot = (size_t)(
            cube_hash(fixed, value, words, -1, 0) & (uint64_t)mask
        );

        while (index->slots[slot] >= 0) {
            if (
                cube_equal(
                    index,
                    (size_t)index->slots[slot],
                    fixed,
                    value,
                    -1,
                    0
                )
            ) {
                break;
            }
            slot = (slot + 1u) & mask;
        }

        if (index->slots[slot] >= 0) continue;

        size_t key_index = index->count++;
        index->slots[slot] = (int)key_index;
        memcpy(
            &index->fixed_keys[key_index * (size_t)words],
            fixed,
            (size_t)words * sizeof(uint64_t)
        );

        memcpy(
            &index->value_keys[key_index * (size_t)words],
            value,
            (size_t)words * sizeof(uint64_t)
        );

        size_t support_slot = (size_t)(
            fixed_hash(fixed, words, -1, 0) & (uint64_t)mask
        );

        while (index->support_slots[support_slot] >= 0) {
            if (
                fixed_equal(
                    index,
                    (size_t)index->support_slots[support_slot],
                    fixed,
                    -1,
                    0
                )
            ) {
                break;
            }
            support_slot = (support_slot + 1u) & mask;
        }

        if (index->support_slots[support_slot] < 0) {
            index->support_slots[support_slot] = (int)key_index;
        }
    }

    return true;
}

int subsumption_index_find_generalizing_removals(
    const SubsumptionIndex *index,
    const uint64_t *fixed_bits,
    const int *support,
    int support_count,
    const int *word_index,
    const uint64_t *shifted_mask,
    int *removal_offsets
) {
    if (
        !index ||
        !fixed_bits ||
        !support ||
        support_count <= 0 ||
        !word_index ||
        !shifted_mask ||
        !removal_offsets
    ) {
        return 0;
    }

    int found = 0;
    for (int i = 0; i < support_count; ++i) {
        int input = support[i];
        if (input < 0) continue;

        int removed_word = word_index[input];
        uint64_t removed_mask = shifted_mask[input];
        if (
            removed_word < 0 ||
            removed_word >= index->words ||
            removed_mask == 0
        ) {
            continue;
        }

        if (
            fixed_lookup(
                index,
                fixed_bits,
                removed_word,
                removed_mask
            )
        ) {
            removal_offsets[found++] = i;
        }
    }

    return found;
}

bool subsumption_index_has_immediate_generalization(
    const SubsumptionIndex *index,
    const uint64_t *fixed_bits,
    const uint64_t *value_bits,
    const int *support,
    int support_count,
    const int *removal_offsets,
    int removal_count,
    const int *word_index,
    const uint64_t *shifted_mask
) {
    if (
        !index ||
        !fixed_bits ||
        !value_bits ||
        !support ||
        support_count <= 0 ||
        !removal_offsets ||
        removal_count <= 0 ||
        !word_index ||
        !shifted_mask
    ) {
        return false;
    }

    for (int i = 0; i < removal_count; ++i) {
        int offset = removal_offsets[i];
        if (offset < 0 || offset >= support_count) continue;
        int input = support[offset];
        if (input < 0) continue;

        int removed_word = word_index[input];
        uint64_t removed_mask = shifted_mask[input];
        if (
            removed_word < 0 ||
            removed_word >= index->words ||
            removed_mask == 0
        ) {
            continue;
        }

        if (
            cube_lookup(
                index,
                fixed_bits,
                value_bits,
                removed_word,
                removed_mask
            )
        ) {
            return true;
        }
    }

    return false;
}

void subsumption_index_destroy(
    SubsumptionIndex *index
) {
    if (!index) return;
    free(index->slots);
    free(index->support_slots);
    free(index->fixed_keys);
    free(index->value_keys);
    memset(index, 0, sizeof(*index));
}
