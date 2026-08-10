#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cover_validation.h"
#include "projection_rwls.h"

static void exercise_improvement(int cov_bits) {
    const uint64_t coverage[] = {
        UINT64_C(0x1),
        UINT64_C(0x2),
        UINT64_C(0x4),
        UINT64_C(0x8),
        UINT64_C(0x3),
        UINT64_C(0xc)
    };
    const PIChartView chart = {
        .bits = coverage,
        .words = 1,
        .cov_bits = cov_bits,
        .rows = 4,
        .cols = 6
    };
    const int incumbent[] = {0, 1, 2, 3};
    ProjectionRWLSOptions options;
    projection_rwls_options_default(&options);
    options.steps_per_family = 512;
    options.max_adaptive_rounds = 3;
    options.requested_threads = 4;
    options.logical_family_count = 4;
    options.seed = 17;

    int *selected = NULL;
    int cardinality = -1;
    ProjectionRWLSStats stats;
    assert(
        projection_rwls_improve(
            &chart,
            4,
            incumbent,
            &options,
            &selected,
            &cardinality,
            &stats
        ) == PROJECTION_RWLS_OK
    );
    assert(cardinality == 2);
    assert(stats.initial_cardinality == 4);
    assert(stats.final_cardinality == 2);
    assert(stats.targets_improved == 2);
    assert(stats.logical_family_states == 4);
    assert(cover_is_feasible(&chart, selected, cardinality));
    free(selected);

    const int minimum[] = {4, 5};
    selected = NULL;
    cardinality = -1;
    assert(
        projection_rwls_improve(
            &chart,
            2,
            minimum,
            &options,
            &selected,
            &cardinality,
            &stats
        ) == PROJECTION_RWLS_OK
    );
    assert(cardinality == 2);
    assert(memcmp(selected, minimum, sizeof(minimum)) == 0);
    assert(cover_is_feasible(&chart, selected, cardinality));
    free(selected);
}

static void exercise_narrow_coverage_words(void) {
    const uint64_t coverage[] = {
        UINT64_C(0x01), UINT64_C(0x00),
        UINT64_C(0x02), UINT64_C(0x00),
        UINT64_C(0x04), UINT64_C(0x00),
        UINT64_C(0x08), UINT64_C(0x00),
        UINT64_C(0x10), UINT64_C(0x00),
        UINT64_C(0x20), UINT64_C(0x00),
        UINT64_C(0x40), UINT64_C(0x00),
        UINT64_C(0x80), UINT64_C(0x00),
        UINT64_C(0x00), UINT64_C(0x01),
        UINT64_C(0x00), UINT64_C(0x02),
        UINT64_C(0x1f), UINT64_C(0x00),
        UINT64_C(0xe0), UINT64_C(0x03)
    };
    const PIChartView chart = {
        .bits = coverage,
        .words = 2,
        .cov_bits = 8,
        .rows = 10,
        .cols = 12
    };
    const int incumbent[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    ProjectionRWLSOptions options;
    projection_rwls_options_default(&options);
    options.steps_per_family = 2048;
    options.max_adaptive_rounds = 3;
    options.requested_threads = 4;
    options.logical_family_count = 4;
    options.seed = 23;

    int *selected = NULL;
    int cardinality = -1;
    ProjectionRWLSStats stats;
    assert(
        projection_rwls_improve(
            &chart,
            10,
            incumbent,
            &options,
            &selected,
            &cardinality,
            &stats
        ) == PROJECTION_RWLS_OK
    );
    assert(cardinality == 2);
    assert(cover_is_feasible(&chart, selected, cardinality));
    free(selected);
}

int main(void) {
    assert(!projection_rwls_should_run(false, false, false, true, 4));
    assert(!projection_rwls_should_run(true, true, false, true, 4));
    assert(!projection_rwls_should_run(true, false, true, true, 4));
    assert(!projection_rwls_should_run(true, false, false, false, 4));
    assert(!projection_rwls_should_run(true, false, false, true, 1));
    assert(projection_rwls_should_run(true, false, false, true, 4));

    exercise_improvement(64);
    exercise_improvement(8);
    exercise_narrow_coverage_words();

    puts("projection RWLS regression: OK");
    return 0;
}
