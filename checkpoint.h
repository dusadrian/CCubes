#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include <stdbool.h>
#include "utils.h"

// Save the search state to a binary checkpoint file.
// Returns 0 on success, -1 on error.
int save_checkpoint(
    const char *path,
    PIstorage *PInfo,
    int ninputs,
    int noutputs,
    int bits_per_word,
    int value_bit_width,
    int implicant_words,
    int current_k,
    const int *stop_counter,
    int max_levels,
    int weight_pic,
    int scp_type,
    int pool_max,
    int start_level,
    const char *src_path,
    const char *dst_path,
    double elapsed_total,
    double elapsed_scp,
    uint64_t last_task
);

// Load the search state from a binary checkpoint file.
// On success, allocates and fills PInfo and stop_counter_out. Caller must free PInfo via cleanup()
// and free(*stop_counter_out) manually.
// Returns 0 on success, -1 on error.
int load_checkpoint(
    const char *path,
    PIstorage **PInfo,
    int *ninputs,
    int *noutputs,
    int *bits_per_word,
    int *value_bit_width,
    int *implicant_words,
    int *current_k,
    int **stop_counter_out,
    int *max_levels,
    int *weight_pic,
    int *scp_type,
    int *pool_max,
    int *start_level,
    int **nofvalues_out,
    char **src_path_out,
    char **dst_path_out,
    double *elapsed_total_out,
    double *elapsed_scp_out,
    uint64_t *last_task_out
);

#endif // CHECKPOINT_H

