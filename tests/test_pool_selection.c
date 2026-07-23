#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pool_selection.h"

int main(void) {
    PIstorage pinfo[3];
    memset(pinfo, 0, sizeof(pinfo));

    /*
    Candidate order deliberately defeats independent first-on-tie selection:
      output 0: A or B
      output 1: C or A
      output 2: B or C
    Independent first choices use A,C,B (three rows); a joint choice uses two.
    */
    uint64_t positions[3][2] = {
        {1u, 2u}, /* A, B */
        {4u, 1u}, /* C, A */
        {2u, 4u}  /* B, C */
    };
    uint64_t values[3][2] = {{0u, 0u}, {0u, 0u}, {0u, 0u}};
    int solution_index[3][2] = {{0, 1}, {0, 1}, {0, 1}};
    int *pools[3][2];

    for (int output = 0; output < 3; ++output) {
        pools[output][0] = &solution_index[output][0];
        pools[output][1] = &solution_index[output][1];
        pinfo[output].ON_minterms = 1;
        pinfo[output].foundPI = 2;
        pinfo[output].solmin = 1;
        pinfo[output].prevsolmin = 2;
        pinfo[output].implicants_pos = positions[output];
        pinfo[output].implicants_val = values[output];
        pinfo[output].pool_count = 2;
        pinfo[output].pool_solutions = pools[output];
    }

    int chosen[3] = {-1, -1, -1};
    PoolSelectionStats stats;
    assert(select_joint_pool_solutions(pinfo, 3, 1, chosen, &stats));
    assert(stats.selection_exact);
    assert(stats.output_connections == 3);
    assert(stats.selected_distinct_cubes == 2);
    assert(stats.selected_shared_cubes == 1);
    assert(stats.sharing_savings == 1);
    assert(stats.pool_shared_cubes == 3);
    assert(count_retained_shared_cubes(pinfo, 3, 1) == 3);

    int selected[3] = {0, 0, 0};
    for (int output = 0; output < 3; ++output) {
        assert(chosen[output] >= 0);
        selected[output] = pinfo[output].pool_solutions[chosen[output]][0];
        pinfo[output].indices = &selected[output];
    }
    assert(measure_selected_pool_solutions(pinfo, 3, 1, &stats));
    assert(stats.output_connections == 3);
    assert(stats.selected_distinct_cubes == 2);
    assert(stats.selected_shared_cubes == 1);
    assert(stats.sharing_savings == 1);

    /*
    A stopped output may still own a stale pool from the boundary where it
    retained a shorter incumbent.  Later coordination for other outputs must
    leave its committed indices untouched and report no pool choice for it.
    */
    int stale_five_term_cover[5] = {0, 1, 0, 1, 0};
    int *stale_pool[1] = {stale_five_term_cover};
    pinfo[0].stop_search = true;
    pinfo[0].solmin = 1;
    pinfo[0].prevsolmin = 1;
    pinfo[0].pool_count = 1;
    pinfo[0].pool_solutions = stale_pool;
    chosen[0] = 99;
    chosen[1] = 99;
    chosen[2] = 99;

    assert(select_joint_pool_solutions(pinfo, 3, 1, chosen, &stats));
    assert(chosen[0] == -1);
    assert(chosen[1] >= 0);
    assert(chosen[2] >= 0);
    assert(stats.active_outputs == 2);
    assert(stats.output_connections == 2);

    /*
    Pool value is marginal, not a raw candidate count.  Once output 0's
    preferred A cover already matches the only candidate for output 1, its
    unrelated alternatives add no cross-output compatibility.  The first ten
    solver-ranked alternatives are retained as a safety seed, after which
    zero-marginal covers are removed from the coordination search.
    */
    PIstorage value_info[2];
    memset(value_info, 0, sizeof(value_info));
    uint64_t value_positions[2][11] = {
        {
            1u, 2u, 4u, 8u, 16u, 32u,
            64u, 128u, 256u, 512u, 1024u
        },
        {
            1u, 2048u, 4096u, 8192u, 16384u, 32768u,
            65536u, 131072u, 262144u, 524288u, 1048576u
        }
    };
    uint64_t value_values[2][11] = {
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u},
        {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u}
    };
    int value_indices[2][11] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };
    int *value_pools[2][11];
    for (int output = 0; output < 2; ++output) {
        for (int p = 0; p < 11; ++p) {
            value_pools[output][p] = &value_indices[output][p];
        }
        value_info[output].ON_minterms = 1;
        value_info[output].foundPI = 11;
        value_info[output].solmin = 1;
        value_info[output].prevsolmin = 2;
        value_info[output].implicants_pos = value_positions[output];
        value_info[output].implicants_val = value_values[output];
        value_info[output].pool_count = output == 0 ? 11 : 1;
        value_info[output].pool_solutions = value_pools[output];
    }
    int value_chosen[2] = {-1, -1};
    assert(select_joint_pool_solutions(
        value_info,
        2,
        1,
        value_chosen,
        &stats
    ));
    assert(stats.generated_pool_solutions == 12);
    assert(stats.valuable_pool_solutions == 11);
    assert(stats.discarded_pool_solutions == 1);
    assert(value_chosen[0] == 0);
    assert(value_chosen[1] == 0);
    assert(stats.selected_distinct_cubes == 1);

    puts("pool selection regression: OK");
    return 0;
}
