#ifndef SUBSUMPTION_INDEX_H
#define SUBSUMPTION_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Immutable hash index over retained cubes from completed complexity levels.
 * Keys are copied so worker lookups remain valid while the production PI
 * arrays grow or move during the next level.
 */
typedef struct {
    int *slots;
    int *support_slots;
    uint64_t *fixed_keys;
    uint64_t *value_keys;
    size_t table_size;
    size_t count;
    int words;
} SubsumptionIndex;

bool subsumption_index_build(
    SubsumptionIndex *index,
    const uint64_t *fixed_records,
    const uint64_t *value_records,
    int records,
    int words
);

/*
 * Resolve which literal deletions produce a retained, more general support.
 * Run this once per support combination, then reuse the returned offsets for
 * every projected candidate value.
 */
int subsumption_index_find_generalizing_removals(
    const SubsumptionIndex *index,
    const uint64_t *fixed_bits,
    const int *support,
    int support_count,
    const int *word_index,
    const uint64_t *shifted_mask,
    int *removal_offsets
);

/*
 * Return true when one of the preselected literal deletions produces an
 * implicant already present in the completed-level index. That retained
 * implicant subsumes the candidate, proving it nonprime before its coverage
 * bitset is built.
 */
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
);

void subsumption_index_destroy(
    SubsumptionIndex *index
);

#endif
