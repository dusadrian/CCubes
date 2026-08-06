#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "prime_check.h"

/* One binary literal per input, packed 64 per word -- the same layout
 * process_task builds for word_index/bit_index in production. */
static void build_layout(int ninputs, int *word_index, int *bit_index) {
    for (int i = 0; i < ninputs; ++i) {
        word_index[i] = i / 64;
        bit_index[i] = i % 64;
    }
}

static void set_cube_bit(uint64_t *value_bits, const int *word_index, const int *bit_index, int input, int value /* 1 or 2 */) {
    if (value == 2) {
        value_bits[word_index[input]] |= (UINT64_C(1) << bit_index[input]);
    } else {
        value_bits[word_index[input]] &= ~(UINT64_C(1) << bit_index[input]);
    }
}

static void prepare_masks(
    PIstorage *pi,
    int ninputs
) {
    int nofvalues[ninputs];
    for (int input = 0; input < ninputs; ++input) {
        nofvalues[input] = 3;
    }
    assert(prepare_off_compat_masks(
        pi,
        ninputs,
        1,
        nofvalues
    ));
    assert(pi->off_compat_masks);
}

static void free_masks(PIstorage *pi) {
    free(pi->off_mask_offsets);
    free(pi->off_compat_masks);
    pi->off_mask_offsets = NULL;
    pi->off_compat_masks = NULL;
}

int main(void) {
    // level=2 cube on inputs {0,1}, both literals fixed to value 1.
    // OFF row A conflicts on input1 only (blocks removing literal 0).
    // OFF row B conflicts on input0 only (blocks removing literal 1).
    // With both rows present, every literal removal is blocked -> prime.
    // With only one row present, the other removal is never blocked -> not prime.
    {
        int ninputs = 2;
        int support[] = {0, 1};
        int word_index[2], bit_index[2];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        set_cube_bit(value_bits, word_index, bit_index, 0, 1);
        set_cube_bit(value_bits, word_index, bit_index, 1, 1);

        int off_both[] = {
            2, 1, // row A: mismatch on input0, matches input1 -> blocks removing literal 0
            1, 2, // row B: matches input0, mismatch on input1 -> blocks removing literal 1
        };
        PIstorage pi_both;
        memset(&pi_both, 0, sizeof(pi_both));
        pi_both.OFF_minterms = 2;
        pi_both.OFF_set = off_both;
        assert(projected_cube_is_prime(&pi_both, ninputs, support, 2, value_bits, word_index, bit_index) == true);

        int off_a_only[] = {2, 1};
        PIstorage pi_a;
        memset(&pi_a, 0, sizeof(pi_a));
        pi_a.OFF_minterms = 1;
        pi_a.OFF_set = off_a_only;
        assert(projected_cube_is_prime(&pi_a, ninputs, support, 2, value_bits, word_index, bit_index) == false);
    }

    /* A fully specified OFF set large enough to select the adaptive mask path. */
    {
        int ninputs = 2;
        int support[] = {0, 1};
        int word_index[2], bit_index[2];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        int off_rows[65 * 2];
        for (int row = 0; row < 64; ++row) {
            off_rows[row * 2] = 2;
            off_rows[row * 2 + 1] = 1;
        }
        off_rows[64 * 2] = 1;
        off_rows[64 * 2 + 1] = 2;

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 65;
        pi.OFF_set = off_rows;
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        prepare_masks(&pi, ninputs);
        assert(pi.off_mask_words == 2);
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);

        pi.OFF_minterms = 64;
        assert(!projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        prepare_masks(&pi, ninputs);
        assert(pi.off_mask_words == 1);
        assert(!projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);
    }

    /*
     * DC-bearing rows exercise the mask path. Each row mismatches the cube on
     * the literal whose removal it blocks, while wildcard positions among the
     * remaining literals must match either cube value.
     */
    {
        int ninputs = 4;
        int support[] = {0, 1, 2};
        int word_index[4], bit_index[4];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        int off_prime[] = {
            2, 0, 1, 0,
            0, 2, 1, 0,
            1, 0, 2, 0
        };

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 3;
        pi.OFF_set = off_prime;
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            3,
            value_bits,
            word_index,
            bit_index
        ));
        prepare_masks(&pi, ninputs);
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            3,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);

        pi.OFF_minterms = 2;
        assert(!projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            3,
            value_bits,
            word_index,
            bit_index
        ));
        prepare_masks(&pi, ninputs);
        assert(!projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            3,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);
    }

    /*
     * Put the two deletion blockers on opposite sides of a 64-row boundary.
     * The scalar and two-word mask paths must agree.
     */
    {
        int ninputs = 3;
        int support[] = {0, 1};
        int word_index[3], bit_index[3];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        int off_rows[65 * 3];
        for (int row = 0; row < 65; ++row) {
            off_rows[row * 3] = 2;
            off_rows[row * 3 + 1] = 2;
            off_rows[row * 3 + 2] = 0;
        }
        off_rows[0] = 2;
        off_rows[1] = 1;
        off_rows[64 * 3] = 1;
        off_rows[64 * 3 + 1] = 2;

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 65;
        pi.OFF_set = off_rows;
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        prepare_masks(&pi, ninputs);
        assert(pi.off_mask_words == 2);
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            2,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);
    }

    // Empty OFF set: no row can ever block a removal, so a cube with any
    // literal (level > 0) can never be prime.
    {
        int ninputs = 4;
        int support[] = {0, 1, 2, 3};
        int word_index[4], bit_index[4];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 0;
        pi.OFF_set = NULL;
        assert(projected_cube_is_prime(&pi, ninputs, support, 4, value_bits, word_index, bit_index) == false);
    }

    // A row that is entirely don't-care (0) on the support blocks every
    // removal at once -> prime regardless of level.
    {
        int ninputs = 3;
        int support[] = {0, 1, 2};
        int word_index[3], bit_index[3];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        int off_row[] = {0, 0, 0};

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 1;
        pi.OFF_set = off_row;
        assert(projected_cube_is_prime(&pi, ninputs, support, 3, value_bits, word_index, bit_index) == true);
    }

    // level=1: a single literal is prime iff at least one OFF row has a
    // value there (any value blocks the only possible removal, since with
    // level=1 the loop body skips the sole column via c==removed).
    {
        int ninputs = 1;
        int support[] = {0};
        int word_index[1], bit_index[1];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        set_cube_bit(value_bits, word_index, bit_index, 0, 1);

        int off_row[] = {2};
        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 1;
        pi.OFF_set = off_row;
        assert(projected_cube_is_prime(&pi, ninputs, support, 1, value_bits, word_index, bit_index) == true);

        pi.OFF_minterms = 0;
        pi.OFF_set = NULL;
        assert(projected_cube_is_prime(&pi, ninputs, support, 1, value_bits, word_index, bit_index) == false);
    }

    // The level-1 empty-intersection rule is identical on a DC-bearing mask.
    {
        int ninputs = 2;
        int support[] = {0};
        int word_index[2], bit_index[2];
        build_layout(ninputs, word_index, bit_index);
        uint64_t value_bits[1] = {0};
        int off_row[] = {2, 0};

        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.OFF_minterms = 1;
        pi.OFF_set = off_row;
        prepare_masks(&pi, ninputs);
        assert(projected_cube_is_prime(
            &pi,
            ninputs,
            support,
            1,
            value_bits,
            word_index,
            bit_index
        ));
        free_masks(&pi);
    }

    // Invalid inputs (null pointers, level<=0) must fail closed.
    {
        int ninputs = 1;
        int support[] = {0};
        int word_index[1] = {0}, bit_index[1] = {0};
        uint64_t value_bits[1] = {0};
        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        assert(projected_cube_is_prime(NULL, ninputs, support, 1, value_bits, word_index, bit_index) == false);
        assert(projected_cube_is_prime(&pi, ninputs, support, 0, value_bits, word_index, bit_index) == false);
    }

    puts("projected cube primality regression: OK");
    return 0;
}
