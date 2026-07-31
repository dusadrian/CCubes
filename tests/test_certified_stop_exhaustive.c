#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "certified_stop.h"

typedef struct {
    uint16_t coverage;
} BruteCube;

typedef struct {
    const BruteCube *cubes;
    int cube_count;
    uint16_t all_rows;
    signed char *memo;
} CoverSearch;

static bool cube_covers_row(
    uint16_t row,
    uint16_t fixed,
    uint16_t value
) {
    return (row & fixed) == value;
}

static bool cube_hits_any(
    uint16_t fixed,
    uint16_t value,
    uint16_t rows
) {
    while (rows != 0) {
        uint16_t bit = (uint16_t)(rows & (uint16_t)(-rows));
        int row = __builtin_ctz((unsigned int)bit);
        if (cube_covers_row((uint16_t)row, fixed, value)) return true;
        rows &= (uint16_t)(rows - 1u);
    }
    return false;
}

static int enumerate_prime_cubes(
    int ninputs,
    uint16_t on_rows,
    uint16_t off_rows,
    BruteCube *cubes,
    int capacity
) {
    const uint16_t input_mask = (uint16_t)((1u << ninputs) - 1u);
    int count = 0;

    for (uint16_t fixed = 1; fixed <= input_mask; ++fixed) {
        uint16_t value = fixed;
        for (;;) {
            uint16_t coverage = 0;
            for (int row = 0; row < (1 << ninputs); ++row) {
                uint16_t row_bit = (uint16_t)(1u << row);
                if (
                    (on_rows & row_bit) != 0 &&
                    cube_covers_row((uint16_t)row, fixed, value)
                ) {
                    coverage |= row_bit;
                }
            }

            bool prime = coverage != 0 &&
                !cube_hits_any(fixed, value, off_rows);
            uint16_t remaining = fixed;
            while (prime && remaining != 0) {
                uint16_t literal =
                    (uint16_t)(remaining & (uint16_t)(-remaining));
                uint16_t parent_fixed = (uint16_t)(fixed & ~literal);
                uint16_t parent_value = (uint16_t)(value & parent_fixed);
                if (!cube_hits_any(parent_fixed, parent_value, off_rows)) {
                    prime = false;
                }
                remaining &= (uint16_t)(remaining - 1u);
            }

            if (prime) {
                assert(count < capacity);
                cubes[count++].coverage = coverage;
            }

            if (value == 0) break;
            value = (uint16_t)((value - 1u) & fixed);
        }
    }

    return count;
}

static int minimum_cover_rec(CoverSearch *search, uint16_t covered) {
    if (covered == search->all_rows) return 0;
    if (search->memo[covered] >= 0) return search->memo[covered];

    uint16_t uncovered = (uint16_t)(search->all_rows & ~covered);
    uint16_t branch_row =
        (uint16_t)(uncovered & (uint16_t)(-uncovered));
    int best = 127;

    for (int cube = 0; cube < search->cube_count; ++cube) {
        if ((search->cubes[cube].coverage & branch_row) == 0) continue;
        uint16_t next =
            (uint16_t)(covered | search->cubes[cube].coverage);
        int suffix = minimum_cover_rec(search, next);
        if (suffix < 127 && suffix + 1 < best) best = suffix + 1;
    }

    search->memo[covered] = (signed char)best;
    return best;
}

static int minimum_cover(
    const BruteCube *cubes,
    int cube_count,
    uint16_t on_rows
) {
    size_t states = (size_t)UINT16_MAX + 1u;
    signed char *memo = malloc(states * sizeof(*memo));
    assert(memo != NULL);
    memset(memo, -1, states * sizeof(*memo));

    CoverSearch search = {
        .cubes = cubes,
        .cube_count = cube_count,
        .all_rows = on_rows,
        .memo = memo
    };
    int result = minimum_cover_rec(&search, 0);
    free(memo);
    return result;
}

static void verify_function(int ninputs, uint16_t on_rows) {
    uint16_t universe =
        (uint16_t)((UINT32_C(1) << (1 << ninputs)) - 1u);
    uint16_t off_rows = (uint16_t)(universe & ~on_rows);

    BruteCube cubes[81];
    int cube_count = enumerate_prime_cubes(
        ninputs,
        on_rows,
        off_rows,
        cubes,
        (int)(sizeof(cubes) / sizeof(cubes[0]))
    );
    assert(cube_count > 0);

    int global_optimum = minimum_cover(cubes, cube_count, on_rows);
    assert(global_optimum > 0 && global_optimum < 127);

    CertifiedStopState certificate;
    certified_stop_state_reset(&certificate);
    assert(!certified_stop_should_stop(
        &certificate,
        global_optimum,
        true
    ));

    certified_stop_observe_complete_prime_chart(&certificate);
    assert(!certified_stop_should_stop(
        &certificate,
        global_optimum,
        false
    ));
    assert(certified_stop_should_stop(
        &certificate,
        global_optimum,
        true
    ));
}

int main(void) {
    uint64_t functions = 0;
    for (int ninputs = 1; ninputs <= 4; ++ninputs) {
        int row_count = 1 << ninputs;
        uint32_t function_count = UINT32_C(1) << row_count;
        for (
            uint32_t function = 1;
            function + 1u < function_count;
            ++function
        ) {
            verify_function(ninputs, (uint16_t)function);
            functions++;
        }
    }

    assert(functions == UINT64_C(65804));
    printf(
        "complete-chart certification exhaustive regression: "
        "%llu functions OK\n",
        (unsigned long long)functions
    );
    return 0;
}
