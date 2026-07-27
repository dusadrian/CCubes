#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pool_selection.h"
#include "utils.h"

typedef struct {
    PIstorage pi;
    uint64_t coverage[8];
    uint64_t positions[8];
    uint64_t values[8];
    int shared[8];
    int covsum[8];
    int covered[8];
    int last_index[2];
    int k_last_index[2];
} FinalizationFixture;

static void init_fixture(
    FinalizationFixture *fixture,
    int found
) {
    memset(fixture, 0, sizeof(*fixture));
    fixture->pi.ON_minterms = 2;
    fixture->pi.pichart_words = 1;
    fixture->pi.foundPI = found;
    fixture->pi.pichart_pos = fixture->coverage;
    fixture->pi.implicants_pos = fixture->positions;
    fixture->pi.implicants_val = fixture->values;
    fixture->pi.shared = fixture->shared;
    fixture->pi.covsum = fixture->covsum;
    fixture->pi.covered = fixture->covered;
    fixture->pi.last_index = fixture->last_index;
    fixture->pi.k_last_index = fixture->k_last_index;
}

static void verify_geometry_aware_deferred_pruning(void) {
    FinalizationFixture fixture;
    init_fixture(&fixture, 5);

    /*
     * Record zero belongs to a completed level.  The current level contains:
     *
     *   1: same coverage, different but unshared geometry -> discard
     *   2: same coverage, distinct shareable geometry       -> retain
     *   3: exact duplicate of record 2                      -> discard
     *   4: new coverage                                     -> retain
     *
     * Keeping record 2 is essential for cross-output pooling: equal local
     * coverage does not make distinct shared cube geometries interchangeable.
     */
    uint64_t coverage[5] = {3u, 3u, 3u, 3u, 1u};
    uint64_t positions[5] = {1u, 2u, 4u, 4u, 8u};
    int shared[5] = {0, 0, 1, 1, 0};
    int covsum[5] = {2, 2, 2, 2, 1};
    memcpy(fixture.coverage, coverage, sizeof(coverage));
    memcpy(fixture.positions, positions, sizeof(positions));
    memcpy(fixture.shared, shared, sizeof(shared));
    memcpy(fixture.covsum, covsum, sizeof(covsum));

    assert(finalize_pi_level(&fixture.pi, 1, 1, false, true));
    assert(fixture.pi.foundPI == 3);
    assert(fixture.positions[0] == 1u);
    assert(fixture.positions[1] == 4u);
    assert(fixture.shared[1] == 1);
    assert(fixture.positions[2] == 8u);
}

static void verify_d_controls_only_canonical_order(void) {
    FinalizationFixture arrival;
    FinalizationFixture canonical;
    init_fixture(&arrival, 3);
    init_fixture(&canonical, 3);

    uint64_t coverage[3] = {1u, 2u, 3u};
    uint64_t positions[3] = {8u, 2u, 4u};
    int covsum[3] = {1, 1, 2};
    memcpy(arrival.coverage, coverage, sizeof(coverage));
    memcpy(arrival.positions, positions, sizeof(positions));
    memcpy(arrival.covsum, covsum, sizeof(covsum));
    memcpy(canonical.coverage, coverage, sizeof(coverage));
    memcpy(canonical.positions, positions, sizeof(positions));
    memcpy(canonical.covsum, covsum, sizeof(covsum));

    assert(finalize_pi_level(&arrival.pi, 1, 0, false, false));
    assert(arrival.pi.foundPI == 3);
    assert(arrival.positions[0] == 8u);
    assert(arrival.positions[1] == 2u);
    assert(arrival.positions[2] == 4u);

    assert(finalize_pi_level(&canonical.pi, 1, 0, true, false));
    assert(canonical.pi.foundPI == 3);
    assert(canonical.positions[0] == 2u);
    assert(canonical.positions[1] == 4u);
    assert(canonical.positions[2] == 8u);
}

static void verify_pooling_keeps_the_useful_geometry(void) {
    FinalizationFixture output0;
    init_fixture(&output0, 2);
    output0.coverage[0] = 3u;
    output0.coverage[1] = 3u;
    output0.positions[0] = 1u; /* A */
    output0.positions[1] = 2u; /* B */
    output0.shared[0] = 1;
    output0.shared[1] = 1;
    output0.covsum[0] = 2;
    output0.covsum[1] = 2;

    assert(finalize_pi_level(&output0.pi, 1, 0, false, true));
    assert(output0.pi.foundPI == 2);

    PIstorage pinfo[3];
    memset(pinfo, 0, sizeof(pinfo));
    pinfo[0] = output0.pi;

    uint64_t output1_positions[1] = {2u};      /* B */
    uint64_t output1_values[1] = {0u};
    uint64_t output2_positions[2] = {1u, 4u}; /* A, C */
    uint64_t output2_values[2] = {0u, 0u};
    pinfo[1].ON_minterms = 2;
    pinfo[1].foundPI = 1;
    pinfo[1].implicants_pos = output1_positions;
    pinfo[1].implicants_val = output1_values;
    pinfo[2].ON_minterms = 2;
    pinfo[2].foundPI = 2;
    pinfo[2].implicants_pos = output2_positions;
    pinfo[2].implicants_val = output2_values;

    int output0_indices[2] = {0, 1};
    int output1_index = 0;
    int output2_index = 1; /* C; A exists in the chart but not this pool. */
    int *output0_pools[2] = {
        &output0_indices[0],
        &output0_indices[1]
    };
    int *output1_pools[1] = {&output1_index};
    int *output2_pools[1] = {&output2_index};
    pinfo[0].pool_count = 2;
    pinfo[0].pool_solutions = output0_pools;
    pinfo[1].pool_count = 1;
    pinfo[1].pool_solutions = output1_pools;
    pinfo[2].pool_count = 1;
    pinfo[2].pool_solutions = output2_pools;

    for (int output = 0; output < 3; ++output) {
        pinfo[output].solmin = 1;
        pinfo[output].prevsolmin = 2;
    }

    /*
     * Output zero must retain B as well as coverage-equivalent A.  B then
     * matches output one, producing the two-cube union {B,C}; keeping only A
     * would force the inferior three-cube union {A,B,C}.
     */
    int chosen[3] = {-1, -1, -1};
    PoolSelectionStats stats;
    assert(select_joint_pool_solutions(pinfo, 3, 1, chosen, &stats));
    assert(chosen[0] == 1);
    assert(stats.selected_distinct_cubes == 2);
    assert(stats.sharing_savings == 1);
}

int main(void) {
    verify_geometry_aware_deferred_pruning();
    verify_d_controls_only_canonical_order();
    verify_pooling_keeps_the_useful_geometry();
    puts("PI level finalization regression: OK");
    return 0;
}
