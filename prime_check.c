#include "prime_check.h"

#include <assert.h>

/*
 * A consistent cube is prime exactly when deleting any one of its literals
 * makes it intersect at least one OFF pattern. A zero in an OFF row is an
 * input dash and therefore matches either binary literal.
 *
 * DC-bearing outputs already own wildcard-compatible OFF-row masks for PI
 * validation. Reuse them here to test each deletion with word intersections.
 * Fully specified outputs retain the scalar loop and its cheap early exits.
 */
static bool projected_cube_is_prime_with_masks(
    const PIstorage *pi,
    const int *support,
    int level,
    const uint64_t *value_bits,
    const int *word_index,
    const int *bit_index
) {
    /*
     * Removing the only literal leaves the universal cube. It intersects the
     * OFF set iff that set is nonempty, matching the scalar empty-inner-loop
     * semantics.
     */
    if (level == 1) return pi->OFF_minterms > 0;

    int support_masks[level];
    for (int c = 0; c < level; ++c) {
        int input = support[c];
        int cube_value = 1 + (int)(
            (value_bits[word_index[input]] >> bit_index[input]) &
            UINT64_C(1)
        );
        support_masks[c] =
            pi->off_mask_offsets[input] + cube_value - 1;
        assert(
            support_masks[c] >= 0 &&
            support_masks[c] < pi->off_mask_count
        );
    }

    for (int removed = 0; removed < level; ++removed) {
        bool deletion_blocked = false;

        for (int word = 0; word < pi->off_mask_words; ++word) {
            uint64_t matches = 0;
            bool seeded = false;

            for (int c = 0; c < level; ++c) {
                if (c == removed) continue;

                const uint64_t *mask_words = &pi->off_compat_masks[
                    (size_t)support_masks[c] *
                    (size_t)pi->off_mask_words
                ];

                if (!seeded) {
                    matches = mask_words[word];
                    seeded = true;
                } else {
                    matches &= mask_words[word];
                }
                if (matches == 0) break;
            }

            assert(seeded);
            if (matches != 0) {
                deletion_blocked = true;
                break;
            }
        }

        if (!deletion_blocked) return false;
    }

    return true;
}

bool projected_cube_is_prime(
    const PIstorage *pi,
    int ninputs,
    const int *support,
    int level,
    const uint64_t *value_bits,
    const int *word_index,
    const int *bit_index
) {
    if (!pi || !support || !value_bits || !word_index || !bit_index || level <= 0) {
        return false;
    }

    if (
        pi->off_compat_masks &&
        pi->off_mask_offsets &&
        pi->off_mask_words > 0
    ) {
        return projected_cube_is_prime_with_masks(
            pi,
            support,
            level,
            value_bits,
            word_index,
            bit_index
        );
    }

    for (int removed = 0; removed < level; ++removed) {
        bool deletion_blocked = false;

        for (int z = 0; z < pi->OFF_minterms; ++z) {
            bool matches = true;
            for (int c = 0; c < level; ++c) {
                if (c == removed) continue;

                int input = support[c];
                int off_value = pi->OFF_set[(size_t)z * (size_t)ninputs + (size_t)input];
                int cube_value = 1 + (int)(
                    (value_bits[word_index[input]] >> bit_index[input]) & 1ULL
                );

                if (off_value != 0 && off_value != cube_value) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                deletion_blocked = true;
                break;
            }
        }

        if (!deletion_blocked) return false;
    }

    return true;
}
