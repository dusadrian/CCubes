#ifndef POOL_SELECTION_H
#define POOL_SELECTION_H

#include <stdbool.h>
#include "utils.h"

typedef struct {
    int active_outputs;
    int generated_pool_solutions;
    int valuable_pool_solutions;
    int discarded_pool_solutions;
    int total_pool_solutions;
    int retained_shared_cubes;
    int pool_shared_cubes;
    int output_connections;
    int selected_distinct_cubes;
    int selected_input_literals;
    int selected_shared_cubes;
    int sharing_savings;
    bool selection_exact;
} PoolSelectionStats;

/*
Select one candidate cover per output so that their union contains as few
distinct cubes as possible, then as few input literals as possible among
equal-size unions. Exact enumeration is used for bounded pool products;
larger products use deterministic multi-start coordinate descent.
chosen_pool[o] is a pool index, or -1 when the existing indices are the only
available cover for that output.
*/
bool select_joint_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats
);

/*
Repeat the coordination pass after generation has stopped, using the final
pool retained by every output rather than excluding already committed outputs.
*/
bool select_final_joint_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    int *chosen_pool,
    PoolSelectionStats *stats
);

/* Refresh the selected-union fields after a tied cover is replaced in-place. */
bool measure_selected_pool_solutions(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words,
    PoolSelectionStats *stats
);

/* Count exact cube identities retained in the charts of at least two outputs. */
int count_retained_shared_cubes(
    const PIstorage *pinfo,
    int noutputs,
    int implicant_words
);

/* Release the current per-output pool before replacing it at a new level. */
void clear_output_solution_pool(PIstorage *pi);

#endif
