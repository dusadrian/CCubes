#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bounded_mmcs.h"

static uint32_t rng_state = UINT32_C(0x6d2b79f5);

static uint32_t next_random(void) {
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state;
}

static int popcount_u64(uint64_t value) {
    int count = 0;
    while (value) {
        value &= value - 1u;
        count++;
    }
    return count;
}

static bool cube_covers(
    uint64_t row,
    uint64_t fixed,
    uint64_t value
) {
    return (row & fixed) == value;
}

static bool brute_is_prime(
    const uint64_t *on_rows,
    int on_count,
    const uint64_t *off_rows,
    int off_count,
    uint64_t fixed,
    uint64_t value
) {
    bool covers_on = false;
    for (int row = 0; row < on_count; ++row) {
        if (cube_covers(on_rows[row], fixed, value)) {
            covers_on = true;
            break;
        }
    }
    if (!covers_on) return false;

    for (int row = 0; row < off_count; ++row) {
        if (cube_covers(off_rows[row], fixed, value)) return false;
    }

    uint64_t remaining = fixed;
    while (remaining) {
        uint64_t bit = remaining & (~remaining + 1u);
        uint64_t parent_fixed = fixed & ~bit;
        uint64_t parent_value = value & parent_fixed;
        bool parent_is_implicant = true;
        for (int row = 0; row < off_count; ++row) {
            if (cube_covers(off_rows[row], parent_fixed, parent_value)) {
                parent_is_implicant = false;
                break;
            }
        }
        if (parent_is_implicant) return false;
        remaining &= remaining - 1u;
    }
    return true;
}

static void free_generated_arrays(
    PIstorage *pi
) {
    free(pi->pichart_pos);
    free(pi->implicants_pos);
    free(pi->implicants_val);
    free(pi->shared);
    free(pi->covsum);
    free(pi->covered);
    pi->pichart_pos = NULL;
    pi->implicants_pos = NULL;
    pi->implicants_val = NULL;
    pi->shared = NULL;
    pi->covsum = NULL;
    pi->covered = NULL;
}

static void verify_function(
    int ninputs,
    const unsigned char *is_on
) {
    int rows = 1 << ninputs;
    int on_count = 0;
    for (int row = 0; row < rows; ++row) on_count += is_on[row] != 0;
    int off_count = rows - on_count;
    assert(on_count > 0 && off_count > 0);

    int *on_set = calloc((size_t)on_count * (size_t)ninputs, sizeof(int));
    int *off_set = calloc((size_t)off_count * (size_t)ninputs, sizeof(int));
    uint64_t *on_rows = calloc((size_t)on_count, sizeof(uint64_t));
    uint64_t *off_rows = calloc((size_t)off_count, sizeof(uint64_t));
    assert(on_set && off_set && on_rows && off_rows);

    int on = 0;
    int off = 0;
    for (int row = 0; row < rows; ++row) {
        int *target = is_on[row]
            ? &on_set[(size_t)on * (size_t)ninputs]
            : &off_set[(size_t)off * (size_t)ninputs];
        for (int input = 0; input < ninputs; ++input) {
            target[input] = ((row >> input) & 1) ? 2 : 1;
        }
        if (is_on[row]) {
            on_rows[on++] = (uint64_t)row;
        } else {
            off_rows[off++] = (uint64_t)row;
        }
    }

    int word_index[8];
    int bit_index[8];
    uint64_t shifted_mask[8];
    for (int input = 0; input < ninputs; ++input) {
        word_index[input] = 0;
        bit_index[input] = input * 2;
        shifted_mask[input] = UINT64_C(3) << bit_index[input];
    }

    for (int level = 1; level <= ninputs; ++level) {
        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.ON_minterms = on_count;
        pi.OFF_minterms = off_count;
        pi.ON_set = on_set;
        pi.OFF_set = off_set;
        pi.pichart_words = (on_count + 63) / 64;
        pi.cov_bits = 64;

        BoundedMMCSStats stats;
        memset(&stats, 0, sizeof(stats));
        assert(bounded_mmcs_generate_output_level(
            &pi,
            ninputs,
            level,
            word_index,
            bit_index,
            shifted_mask,
            1,
            &stats
        ));

        int brute_count = 0;
        uint64_t universe = (UINT64_C(1) << ninputs) - 1u;
        for (uint64_t fixed = 1; fixed <= universe; ++fixed) {
            if (popcount_u64(fixed) != level) continue;
            uint64_t value = fixed;
            while (1) {
                if (brute_is_prime(
                    on_rows,
                    on_count,
                    off_rows,
                    off_count,
                    fixed,
                    value
                )) {
                    brute_count++;
                    uint64_t packed_fixed = 0;
                    uint64_t packed_value = 0;
                    for (int input = 0; input < ninputs; ++input) {
                        uint64_t bit = UINT64_C(1) << input;
                        if ((fixed & bit) == 0) continue;
                        packed_fixed |= shifted_mask[input];
                        if (value & bit) {
                            packed_value |= UINT64_C(1) << bit_index[input];
                        }
                    }
                    bool found = false;
                    for (int cube = 0; cube < pi.foundPI; ++cube) {
                        if (
                            pi.implicants_pos[cube] == packed_fixed &&
                            pi.implicants_val[cube] == packed_value
                        ) {
                            found = true;
                            break;
                        }
                    }
                    assert(found);
                }
                if (value == 0) break;
                value = (value - 1u) & fixed;
            }
        }

        assert(pi.foundPI == brute_count);
        assert(stats.unique_cubes == (uint64_t)brute_count);

        for (int cube = 0; cube < pi.foundPI; ++cube) {
            uint64_t fixed = 0;
            uint64_t value = 0;
            for (int input = 0; input < ninputs; ++input) {
                if (pi.implicants_pos[cube] & shifted_mask[input]) {
                    fixed |= UINT64_C(1) << input;
                    if (
                        pi.implicants_val[cube] &
                        (UINT64_C(1) << bit_index[input])
                    ) {
                        value |= UINT64_C(1) << input;
                    }
                }
            }
            assert(popcount_u64(fixed) == level);
            assert(brute_is_prime(
                on_rows,
                on_count,
                off_rows,
                off_count,
                fixed,
                value
            ));

            int covsum = 0;
            for (int row = 0; row < on_count; ++row) {
                bool expected = cube_covers(on_rows[row], fixed, value);
                bool actual = (
                    pi.pichart_pos[
                        (size_t)cube * (size_t)pi.pichart_words +
                        (size_t)(row / 64)
                    ] & (UINT64_C(1) << (row % 64))
                ) != 0;
                assert(actual == expected);
                covsum += expected;
            }
            assert(pi.covsum[cube] == covsum);
        }

        free_generated_arrays(&pi);
    }

    free(on_set);
    free(off_set);
    free(on_rows);
    free(off_rows);
}

static void verify_random_functions(void) {
    for (int trial = 0; trial < 300; ++trial) {
        int ninputs = 2 + (int)(next_random() % 7u);
        int rows = 1 << ninputs;
        unsigned char *is_on = calloc((size_t)rows, sizeof(unsigned char));
        assert(is_on);

        int on_count = 0;
        for (int row = 0; row < rows; ++row) {
            is_on[row] = (unsigned char)((next_random() >> 31) & 1u);
            on_count += is_on[row] != 0;
        }
        if (on_count == 0) is_on[0] = 1;
        if (on_count == rows) is_on[rows - 1] = 0;

        verify_function(ninputs, is_on);
        free(is_on);
    }
}

static void verify_cross_output_sharing(void) {
    PIstorage outputs[3];
    memset(outputs, 0, sizeof(outputs));
    int level_start[3] = {0, 0, 0};

    for (int output = 0; output < 3; ++output) {
        outputs[output].foundPI = 1;
        outputs[output].implicants_pos = calloc(1, sizeof(uint64_t));
        outputs[output].implicants_val = calloc(1, sizeof(uint64_t));
        outputs[output].shared = calloc(1, sizeof(int));
        assert(
            outputs[output].implicants_pos &&
            outputs[output].implicants_val &&
            outputs[output].shared
        );
        outputs[output].implicants_pos[0] = UINT64_C(0x7);
        outputs[output].implicants_val[0] = output == 2
            ? UINT64_C(0x4)
            : UINT64_C(0x3);
    }

    int max_shared = 0;
    assert(bounded_mmcs_mark_level_sharing(
        outputs,
        3,
        level_start,
        1,
        &max_shared
    ));
    assert(outputs[0].shared[0] == 1);
    assert(outputs[1].shared[0] == 1);
    assert(outputs[2].shared[0] == 0);
    assert(max_shared == 1);

    for (int output = 0; output < 3; ++output) {
        free(outputs[output].implicants_pos);
        free(outputs[output].implicants_val);
        free(outputs[output].shared);
    }
}

static void verify_transactional_node_limit(void) {
    int on_set[] = {
        1, 1, 1
    };
    int off_set[] = {
        2, 1, 1,
        1, 2, 1,
        1, 1, 2
    };
    int word_index[] = {0, 0, 0};
    int bit_index[] = {0, 2, 4};
    uint64_t shifted_mask[] = {
        UINT64_C(3),
        UINT64_C(3) << 2,
        UINT64_C(3) << 4
    };

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 1;
    pi.OFF_minterms = 3;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.pichart_words = 1;
    pi.cov_bits = 64;

    BoundedMMCSStats limited_stats;
    memset(&limited_stats, 0, sizeof(limited_stats));
    assert(bounded_mmcs_generate_output_level_limited(
        &pi,
        3,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        1,
        &limited_stats
    ) == BOUNDED_MMCS_LIMIT_REACHED);
    assert(limited_stats.search_nodes == 1);
    assert(pi.foundPI == 0);
    assert(pi.implicants_pos == NULL);
    assert(pi.pichart_pos == NULL);

    BoundedMMCSStats complete_stats;
    memset(&complete_stats, 0, sizeof(complete_stats));
    assert(bounded_mmcs_generate_output_level_limited(
        &pi,
        3,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        0,
        &complete_stats
    ) == BOUNDED_MMCS_COMPLETE);
    assert(pi.foundPI == 1);
    assert(pi.implicants_pos[0] == UINT64_C(0x3f));
    assert(pi.implicants_val[0] == 0);
    free_generated_arrays(&pi);
}

int main(void) {
    verify_random_functions();
    verify_cross_output_sharing();
    verify_transactional_node_limit();
    puts("bounded MMCS regression: OK");
    return 0;
}
