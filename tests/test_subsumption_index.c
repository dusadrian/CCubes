#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "subsumption_index.h"

static void set_literal(
    uint64_t *fixed,
    uint64_t *value,
    int input,
    unsigned int encoded_value,
    const int *word_index,
    const int *bit_index,
    const uint64_t *shifted_mask
) {
    int word = word_index[input];
    fixed[word] |= shifted_mask[input];
    value[word] |= (uint64_t)encoded_value << bit_index[input];
}

static void test_one_word_lookup(void) {
    enum { INPUTS = 4, WORDS = 1 };
    int word_index[INPUTS] = {0, 0, 0, 0};
    int bit_index[INPUTS] = {0, 2, 4, 6};
    uint64_t shifted_mask[INPUTS] = {
        UINT64_C(3) << 0,
        UINT64_C(3) << 2,
        UINT64_C(3) << 4,
        UINT64_C(3) << 6
    };

    uint64_t fixed_records[2 * WORDS];
    uint64_t value_records[2 * WORDS];
    memset(fixed_records, 0, sizeof(fixed_records));
    memset(value_records, 0, sizeof(value_records));

    set_literal(
        &fixed_records[0],
        &value_records[0],
        0,
        0,
        word_index,
        bit_index,
        shifted_mask
    );
    set_literal(
        &fixed_records[0],
        &value_records[0],
        1,
        1,
        word_index,
        bit_index,
        shifted_mask
    );

    set_literal(
        &fixed_records[1],
        &value_records[1],
        0,
        1,
        word_index,
        bit_index,
        shifted_mask
    );
    set_literal(
        &fixed_records[1],
        &value_records[1],
        1,
        0,
        word_index,
        bit_index,
        shifted_mask
    );

    SubsumptionIndex index;
    assert(subsumption_index_build(
        &index,
        fixed_records,
        value_records,
        2,
        WORDS
    ));
    assert(index.count == 2);

    uint64_t candidate_fixed[WORDS] = {0};
    uint64_t candidate_value[WORDS] = {0};
    set_literal(
        candidate_fixed,
        candidate_value,
        0,
        0,
        word_index,
        bit_index,
        shifted_mask
    );
    set_literal(
        candidate_fixed,
        candidate_value,
        1,
        1,
        word_index,
        bit_index,
        shifted_mask
    );
    set_literal(
        candidate_fixed,
        candidate_value,
        2,
        1,
        word_index,
        bit_index,
        shifted_mask
    );

    int support[3] = {0, 1, 2};
    int removals[3];
    int removal_count = subsumption_index_find_generalizing_removals(
        &index,
        candidate_fixed,
        support,
        3,
        word_index,
        shifted_mask,
        removals
    );
    assert(removal_count == 1);
    assert(subsumption_index_has_immediate_generalization(
        &index,
        candidate_fixed,
        candidate_value,
        support,
        3,
        removals,
        removal_count,
        word_index,
        shifted_mask
    ));

    candidate_value[0] &= ~shifted_mask[1];
    assert(!subsumption_index_has_immediate_generalization(
        &index,
        candidate_fixed,
        candidate_value,
        support,
        3,
        removals,
        removal_count,
        word_index,
        shifted_mask
    ));

    subsumption_index_destroy(&index);
}

static void test_multiword_lookup_and_deduplication(void) {
    enum { INPUTS = 40, WORDS = 2 };
    int word_index[INPUTS];
    int bit_index[INPUTS];
    uint64_t shifted_mask[INPUTS];
    for (int input = 0; input < INPUTS; ++input) {
        word_index[input] = input / 32;
        bit_index[input] = (input % 32) * 2;
        shifted_mask[input] = UINT64_C(3) << bit_index[input];
    }

    uint64_t fixed_records[2 * WORDS];
    uint64_t value_records[2 * WORDS];
    memset(fixed_records, 0, sizeof(fixed_records));
    memset(value_records, 0, sizeof(value_records));

    set_literal(
        fixed_records,
        value_records,
        1,
        1,
        word_index,
        bit_index,
        shifted_mask
    );
    set_literal(
        fixed_records,
        value_records,
        35,
        2,
        word_index,
        bit_index,
        shifted_mask
    );
    memcpy(
        &fixed_records[WORDS],
        fixed_records,
        WORDS * sizeof(uint64_t)
    );
    memcpy(
        &value_records[WORDS],
        value_records,
        WORDS * sizeof(uint64_t)
    );

    SubsumptionIndex index;
    assert(subsumption_index_build(
        &index,
        fixed_records,
        value_records,
        2,
        WORDS
    ));
    assert(index.count == 1);

    uint64_t candidate_fixed[WORDS];
    uint64_t candidate_value[WORDS];
    memcpy(candidate_fixed, fixed_records, sizeof(candidate_fixed));
    memcpy(candidate_value, value_records, sizeof(candidate_value));
    set_literal(
        candidate_fixed,
        candidate_value,
        20,
        2,
        word_index,
        bit_index,
        shifted_mask
    );

    int support[3] = {1, 20, 35};
    int removals[3];
    int removal_count = subsumption_index_find_generalizing_removals(
        &index,
        candidate_fixed,
        support,
        3,
        word_index,
        shifted_mask,
        removals
    );
    assert(removal_count == 1);
    assert(subsumption_index_has_immediate_generalization(
        &index,
        candidate_fixed,
        candidate_value,
        support,
        3,
        removals,
        removal_count,
        word_index,
        shifted_mask
    ));

    subsumption_index_destroy(&index);
}

static void test_empty_index(void) {
    SubsumptionIndex index;
    assert(subsumption_index_build(&index, NULL, NULL, 0, 1));

    uint64_t fixed = UINT64_C(3);
    uint64_t value = 0;
    int support = 0;
    int word_index = 0;
    uint64_t shifted_mask = UINT64_C(3);
    int removal = -1;
    assert(subsumption_index_find_generalizing_removals(
        &index,
        &fixed,
        &support,
        1,
        &word_index,
        &shifted_mask,
        &removal
    ) == 0);
    assert(!subsumption_index_has_immediate_generalization(
        &index,
        &fixed,
        &value,
        &support,
        1,
        &removal,
        0,
        &word_index,
        &shifted_mask
    ));

    subsumption_index_destroy(&index);
}

int main(void) {
    test_one_word_lookup();
    test_multiword_lookup_and_deduplication();
    test_empty_index();
    puts("subsumption index tests passed");
    return 0;
}
