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

    puts("pool selection regression: OK");
    return 0;
}
