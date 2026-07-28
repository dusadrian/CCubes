#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

int main(void) {
    /*
     * Both ON rows project to the same assignment, and that assignment is
     * rejected by the OFF row. It must be validated once, not once per ON
     * row. The test-only counter observes the performance contract without
     * adding production instrumentation.
     */
    int on_set[2] = {1, 1};
    int off_set[1] = {1};
    int cov_word_index[2] = {0, 0};
    uint64_t shifted_cov_mask[2] = {1u, 2u};

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.inputs = 1;
    pi.outputs = 1;
    pi.ON_minterms = 2;
    pi.OFF_minterms = 1;
    pi.pichart_words = 1;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.cov_word_index = cov_word_index;
    pi.shifted_cov_mask = shifted_cov_mask;

    ThreadBuffer output_buffer;
    memset(&output_buffer, 0, sizeof(output_buffer));
    ThreadBuffer *buffers[1] = {&output_buffer};

    int nofvalues[1] = {3};
    int bit_index[1] = {0};
    int word_index[1] = {0};
    uint64_t shifted_mask[1] = {1u};
    int max_shared = 0;
    int multiplier = 0;

    assert(prepare_shared_projection_rows(&pi, 1, 1));
    assert(pi.projection_row_count == 1);
    assert(pi.ON_projection_ids[0] == pi.OFF_projection_ids[0]);

    assert(process_task(
        0,
        1,
        1,
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
        16,
        &multiplier
    ) == 0);

    assert(pi.foundPI == 0);
    assert(output_buffer.validation_attempts == 1);

    int *task_row_codes = output_buffer.task_row_codes;
    uint32_t *task_seen_stamps = output_buffer.task_seen_stamps;
    uint32_t first_seen_epoch = output_buffer.task_seen_epoch;
    assert(task_row_codes != NULL);
    assert(task_seen_stamps != NULL);

    /*
     * A second task reuses both worker-owned allocations but takes fresh
     * epochs, so the rejected assignment must be validated once again.
     */
    assert(process_task(
        0,
        1,
        1,
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
        16,
        &multiplier
    ) == 0);
    assert(output_buffer.task_row_codes == task_row_codes);
    assert(output_buffer.task_seen_stamps == task_seen_stamps);
    assert(output_buffer.task_seen_epoch > first_seen_epoch);
    assert(output_buffer.validation_attempts == 2);

    free(output_buffer.pichart_values);
    free(output_buffer.decpos);
    free(output_buffer.covsum);
    free(output_buffer.fixed_bits);
    free(output_buffer.value_bits);
    free(output_buffer.projection_codes);
    free(output_buffer.projection_has_dc);
    free(output_buffer.task_row_codes);
    free(output_buffer.task_seen_stamps);
    free(pi.projection_rows);
    free(pi.ON_projection_ids);
    free(pi.OFF_projection_ids);

    /*
     * The shared layout must identify a physical input row independently of
     * which output-specific partition contains it.
     */
    int row_a[2] = {1, 2};
    int row_b[2] = {2, 1};
    PIstorage cross_output[2];
    memset(cross_output, 0, sizeof(cross_output));
    for (int output = 0; output < 2; ++output) {
        cross_output[output].inputs = 2;
        cross_output[output].outputs = 2;
        cross_output[output].ON_minterms = 1;
        cross_output[output].OFF_minterms = 1;
    }
    cross_output[0].ON_set = row_a;
    cross_output[0].OFF_set = row_b;
    cross_output[1].ON_set = row_b;
    cross_output[1].OFF_set = row_a;

    assert(prepare_shared_projection_rows(cross_output, 2, 2));
    assert(cross_output[0].projection_row_count == 2);
    assert(
        cross_output[0].ON_projection_ids[0] ==
        cross_output[1].OFF_projection_ids[0]
    );
    assert(
        cross_output[0].OFF_projection_ids[0] ==
        cross_output[1].ON_projection_ids[0]
    );
    free(cross_output[0].projection_rows);
    for (int output = 0; output < 2; ++output) {
        free(cross_output[output].ON_projection_ids);
        free(cross_output[output].OFF_projection_ids);
    }

    puts("projection rejection cache regression: OK");
    return 0;
}
