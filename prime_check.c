#include "prime_check.h"

/*
 * A consistent cube is prime exactly when deleting any one of its literals
 * makes it intersect at least one OFF pattern. A zero in an OFF row is an
 * input dash and therefore matches either binary literal.
 *
 * A single-pass rewrite (recording, per OFF row, a bitmask of which literal
 * positions mismatch the cube) turns the level^2 factor below into level,
 * and wins big when the cube turns out prime -- but it loses the cheap
 * early-out this version gets from returning the moment one literal removal
 * proves unblockable, which is the common case for candidates shared across
 * outputs with dense, unrelated OFF sets (measured ~30-40% slower there on
 * this project's benchmark inputs). Left as the O(level^2 * OFF_minterms)
 * version since that's the actual profile of this codebase's PI generation.
 */
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
