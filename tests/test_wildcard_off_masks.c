#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

typedef struct {
    int found_pi;
    uint64_t pichart;
    uint64_t fixed_bits;
    uint64_t value_bits;
    int covsum;
    uint64_t validations;
} RunResult;

typedef struct {
    int found_pi;
    uint64_t pichart[2];
    uint64_t fixed_bits[2];
    uint64_t value_bits[2];
    int covsum[2];
    int projected_rows;
} ExactRunResult;

static ExactRunResult run_fully_specified_case(bool use_masks) {
    /*
     * ON assignments 01 and 11 are legal; duplicate 01 must contribute a
     * second coverage bit. OFF assignments 00 and 10 are illegal. Both paths
     * eagerly project only the two distinct ON rows; the scalar fallback
     * projects OFF rows lazily inside process_task. They retain identical PIs.
     */
    int on_set[6] = {
        1, 2,
        2, 2,
        1, 2
    };
    int off_set[65 * 2];
    for (int row = 0; row < 65; ++row) {
        off_set[row * 2] = row & 1 ? 2 : 1;
        off_set[row * 2 + 1] = 1;
    }
    int cov_word_index[3] = {0, 0, 0};
    uint64_t shifted_cov_mask[3] = {
        UINT64_C(1),
        UINT64_C(2),
        UINT64_C(4)
    };
    int nofvalues[2] = {3, 3};

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.inputs = 2;
    pi.outputs = 1;
    pi.ON_minterms = 3;
    pi.OFF_minterms = 65;
    pi.pichart_words = 1;
    pi.cov_bits = 64;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.cov_word_index = cov_word_index;
    pi.shifted_cov_mask = shifted_cov_mask;

    if (use_masks) {
        assert(prepare_off_compat_masks(&pi, 2, 1, nofvalues));
    }
    assert(prepare_shared_projection_rows(&pi, 2, 1));

    ThreadBuffer output_buffer;
    memset(&output_buffer, 0, sizeof(output_buffer));
    ThreadBuffer *buffers[1] = {&output_buffer};
    int bit_index[2] = {0, 1};
    int word_index[2] = {0, 0};
    uint64_t shifted_mask[2] = {UINT64_C(1), UINT64_C(2)};
    int max_shared = 0;
    int multiplier = 0;

    assert(process_task(
        0,
        2,
        2,
        1,
        nofvalues,
        bit_index,
        word_index,
        shifted_mask,
        1,
        &pi,
        buffers,
        0,
        NULL,
        NULL,
        &max_shared,
        8,
        &multiplier
    ) == 0);
    assert(pi.foundPI == 2);

    ExactRunResult result;
    memset(&result, 0, sizeof(result));
    result.found_pi = pi.foundPI;
    result.projected_rows = pi.projection_row_count;
    for (int candidate = 0; candidate < pi.foundPI; ++candidate) {
        result.pichart[candidate] = pi.pichart_pos[candidate];
        result.fixed_bits[candidate] = pi.implicants_pos[candidate];
        result.value_bits[candidate] = pi.implicants_val[candidate];
        result.covsum[candidate] = pi.covsum[candidate];
    }

    free(pi.projection_rows);
    free(pi.ON_projection_ids);
    free(pi.off_mask_offsets);
    free(pi.off_compat_masks);
    free(pi.covered);
    free(pi.pichart_pos);
    free(pi.implicants_pos);
    free(pi.implicants_val);
    free(pi.shared);
    free(pi.covsum);
    free(output_buffer.pichart_values);
    free(output_buffer.decpos);
    free(output_buffer.covsum);
    free(output_buffer.fixed_bits);
    free(output_buffer.value_bits);
    free(output_buffer.projection_codes);
    free(output_buffer.projection_has_dc);
    free(output_buffer.task_row_codes);
    free(output_buffer.task_seen_stamps);
    free(output_buffer.task_config_bits);

    return result;
}

static RunResult run_case(bool use_masks) {
    /*
     * OFF patterns:
     *   -0 matches every candidate whose second input is 0
     *   1- matches every candidate whose first input is 1
     *
     * Of the ON rows 01, 11, and 00, only 01 is therefore valid.
     * Values use CCubes' internal encoding: dash=0, input 0=1, input 1=2.
     */
    int on_set[6] = {
        1, 2,
        2, 2,
        1, 1
    };
    int off_set[4] = {
        0, 1,
        2, 0
    };
    int cov_word_index[3] = {0, 0, 0};
    uint64_t shifted_cov_mask[3] = {
        UINT64_C(1),
        UINT64_C(2),
        UINT64_C(4)
    };

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.inputs = 2;
    pi.outputs = 1;
    pi.ON_minterms = 3;
    pi.OFF_minterms = 2;
    pi.pichart_words = 1;
    pi.cov_bits = 64;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.off_has_dc = true;
    pi.cov_word_index = cov_word_index;
    pi.shifted_cov_mask = shifted_cov_mask;

    int nofvalues[2] = {3, 3};
    if (use_masks) {
        assert(prepare_off_compat_masks(&pi, 2, 1, nofvalues));
        assert(pi.off_mask_words == 1);
        assert(pi.off_mask_count == 4);
        assert(pi.off_mask_offsets[0] == 0);
        assert(pi.off_mask_offsets[1] == 2);
        assert(pi.off_mask_offsets[2] == 4);

        /*
         * input 0: value 0 matches row 0; value 1 matches rows 0 and 1
         * input 1: value 0 matches rows 0 and 1; value 1 matches row 1
         */
        assert(pi.off_compat_masks[0] == UINT64_C(1));
        assert(pi.off_compat_masks[1] == UINT64_C(3));
        assert(pi.off_compat_masks[2] == UINT64_C(3));
        assert(pi.off_compat_masks[3] == UINT64_C(2));
    }

    ThreadBuffer output_buffer;
    memset(&output_buffer, 0, sizeof(output_buffer));
    ThreadBuffer *buffers[1] = {&output_buffer};

    int bit_index[2] = {0, 1};
    int word_index[2] = {0, 0};
    uint64_t shifted_mask[2] = {UINT64_C(1), UINT64_C(2)};
    int max_shared = 0;
    int multiplier = 0;

    assert(process_task(
        0,
        2,
        2,
        1,
        nofvalues,
        bit_index,
        word_index,
        shifted_mask,
        1,
        &pi,
        buffers,
        0,
        NULL,
        NULL,
        &max_shared,
        8,
        &multiplier
    ) == 0);

    assert(pi.foundPI == 1);
    RunResult result = {
        .found_pi = pi.foundPI,
        .pichart = pi.pichart_pos[0],
        .fixed_bits = pi.implicants_pos[0],
        .value_bits = pi.implicants_val[0],
        .covsum = pi.covsum[0],
        .validations = output_buffer.validation_attempts
    };

    free(pi.off_mask_offsets);
    free(pi.off_compat_masks);
    free(pi.covered);
    free(pi.pichart_pos);
    free(pi.implicants_pos);
    free(pi.implicants_val);
    free(pi.shared);
    free(pi.covsum);
    free(output_buffer.pichart_values);
    free(output_buffer.decpos);
    free(output_buffer.covsum);
    free(output_buffer.fixed_bits);
    free(output_buffer.value_bits);
    free(output_buffer.task_row_codes);
    free(output_buffer.task_seen_stamps);
    free(output_buffer.task_config_bits);

    return result;
}

int main(void) {
    ExactRunResult exact_fallback = run_fully_specified_case(false);
    ExactRunResult exact_masked = run_fully_specified_case(true);
    assert(exact_fallback.found_pi == exact_masked.found_pi);
    assert(exact_fallback.projected_rows == 2);
    assert(exact_masked.projected_rows == 2);
    for (int candidate = 0; candidate < exact_fallback.found_pi; ++candidate) {
        assert(
            exact_fallback.pichart[candidate] ==
            exact_masked.pichart[candidate]
        );
        assert(
            exact_fallback.fixed_bits[candidate] ==
            exact_masked.fixed_bits[candidate]
        );
        assert(
            exact_fallback.value_bits[candidate] ==
            exact_masked.value_bits[candidate]
        );
        assert(
            exact_fallback.covsum[candidate] ==
            exact_masked.covsum[candidate]
        );
    }

    RunResult fallback = run_case(false);
    RunResult masked = run_case(true);

    assert(fallback.found_pi == masked.found_pi);
    assert(fallback.pichart == masked.pichart);
    assert(fallback.fixed_bits == masked.fixed_bits);
    assert(fallback.value_bits == masked.value_bits);
    assert(fallback.covsum == masked.covsum);
    assert(fallback.validations == masked.validations);
    assert(masked.pichart == UINT64_C(1));
    assert(masked.fixed_bits == UINT64_C(3));
    assert(masked.value_bits == UINT64_C(2));
    assert(masked.covsum == 1);
    assert(masked.validations == 3);

    int fully_specified_on[2] = {2, 1};
    int fully_specified_off[65 * 2];
    for (int row = 0; row < 65; ++row) {
        fully_specified_off[row * 2] = 1;
        fully_specified_off[row * 2 + 1] = 2;
    }
    int nofvalues[2] = {3, 3};
    PIstorage fully_specified;
    memset(&fully_specified, 0, sizeof(fully_specified));
    fully_specified.inputs = 2;
    fully_specified.outputs = 1;
    fully_specified.ON_minterms = 1;
    fully_specified.OFF_minterms = 65;
    fully_specified.ON_set = fully_specified_on;
    fully_specified.OFF_set = fully_specified_off;
    assert(prepare_off_compat_masks(
        &fully_specified,
        2,
        1,
        nofvalues
    ));
    assert(fully_specified.off_mask_words == 2);
    assert(fully_specified.off_mask_count == 4);
    assert(fully_specified.off_mask_offsets != NULL);
    assert(fully_specified.off_compat_masks != NULL);
    assert(fully_specified.off_compat_masks[0] == UINT64_MAX);
    assert(fully_specified.off_compat_masks[1] == UINT64_C(1));
    assert(fully_specified.off_compat_masks[2] == UINT64_C(0));
    assert(fully_specified.off_compat_masks[3] == UINT64_C(0));
    assert(fully_specified.off_compat_masks[4] == UINT64_C(0));
    assert(fully_specified.off_compat_masks[5] == UINT64_C(0));
    assert(fully_specified.off_compat_masks[6] == UINT64_MAX);
    assert(fully_specified.off_compat_masks[7] == UINT64_C(1));

    /* Masked OFF rows are deliberately absent from the projection layout. */
    assert(prepare_shared_projection_rows(&fully_specified, 2, 1));
    assert(fully_specified.projection_row_count == 1);
    assert(fully_specified.ON_projection_ids != NULL);
    assert(memcmp(
        fully_specified.projection_rows,
        fully_specified_on,
        sizeof(fully_specified_on)
    ) == 0);
    free(fully_specified.projection_rows);
    free(fully_specified.ON_projection_ids);
    free(fully_specified.off_mask_offsets);
    free(fully_specified.off_compat_masks);

    /*
     * A high-cardinality exact input can make literal masks larger than the
     * scalar row matrix. The adaptive gate must retain the scalar path there.
     */
    int high_cardinality_off[1] = {1};
    int high_cardinality_values[1] = {100};
    PIstorage high_cardinality;
    memset(&high_cardinality, 0, sizeof(high_cardinality));
    high_cardinality.inputs = 1;
    high_cardinality.outputs = 1;
    high_cardinality.OFF_minterms = 1;
    high_cardinality.OFF_set = high_cardinality_off;
    assert(prepare_off_compat_masks(
        &high_cardinality,
        1,
        1,
        high_cardinality_values
    ));
    assert(high_cardinality.off_mask_words == 0);
    assert(high_cardinality.off_mask_count == 0);
    assert(high_cardinality.off_mask_offsets == NULL);
    assert(high_cardinality.off_compat_masks == NULL);

    /*
     * Exercise the word boundary independently of process_task. Every OFF row
     * is a wildcard, so both value masks must contain all 65 row bits.
     */
    int wildcard_rows[65] = {0};
    int binary_values[1] = {3};
    PIstorage multiword;
    memset(&multiword, 0, sizeof(multiword));
    multiword.inputs = 1;
    multiword.outputs = 1;
    multiword.OFF_minterms = 65;
    multiword.OFF_set = wildcard_rows;
    assert(prepare_off_compat_masks(
        &multiword,
        1,
        1,
        binary_values
    ));
    assert(multiword.off_mask_words == 2);
    assert(multiword.off_mask_count == 2);
    for (int mask = 0; mask < 2; ++mask) {
        assert(multiword.off_compat_masks[mask * 2] == UINT64_MAX);
        assert(multiword.off_compat_masks[mask * 2 + 1] == UINT64_C(1));
    }
    free(multiword.off_mask_offsets);
    free(multiword.off_compat_masks);

    puts("wildcard OFF mask regression: OK");
    return 0;
}
