#include "cover_validation.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    /*
     * Three rows, four columns, bit-packed column-major (one word per column):
     * c0={r0}, c1={r1}, c2={r2}, c3={r0,r1,r2}.
     */
    const uint64_t bits[] = {
        0x1u, /* c0: row 0      */
        0x2u, /* c1: row 1      */
        0x4u, /* c2: row 2      */
        0x7u  /* c3: rows 0,1,2 */
    };
    const PIChartView chart = {
        .bits = bits,
        .words = 1,
        .cov_bits = 64,
        .rows = 3,
        .cols = 4
    };

    const int singleton[] = {3};
    const int three_columns[] = {0, 1, 2};
    const int incomplete[] = {0, 1};
    const int duplicate[] = {3, 3};
    const int out_of_range[] = {4};

    assert(cover_is_feasible(&chart, singleton, 1));
    assert(cover_is_feasible(&chart, three_columns, 3));
    assert(!cover_is_feasible(&chart, incomplete, 2));
    assert(!cover_is_feasible(&chart, duplicate, 2));
    assert(!cover_is_feasible(&chart, out_of_range, 1));
    assert(!cover_is_feasible(&chart, NULL, 0));
    assert(!cover_is_feasible(NULL, singleton, 1));

    /* multi-word packing: rows past 63 must land in word 1, not alias word 0 */
    {
        uint64_t wide[4];
        wide[0] = 0u;                       /* c0 word 0: no rows 0..63  */
        wide[1] = UINT64_C(1) << 1;         /* c0 word 1: row 65         */
        wide[2] = ~UINT64_C(0);             /* c1 word 0: rows 0..63     */
        wide[3] = UINT64_C(3);              /* c1 word 1: rows 64,65     */
        const PIChartView tall = {
            .bits = wide, .words = 2, .cov_bits = 64, .rows = 66, .cols = 2
        };
        const int only_c1[] = {1};
        const int only_c0[] = {0};
        assert(cover_is_feasible(&tall, only_c1, 1));
        assert(!cover_is_feasible(&tall, only_c0, 1));
    }

    /* narrow packing (-b8 style): 8 coverage bits per 64-bit word */
    {
        uint64_t narrow[4];
        narrow[0] = 0xFFu; /* c0 word 0: rows 0..7 */
        narrow[1] = 0x0u;  /* c0 word 1: none      */
        narrow[2] = 0x0u;  /* c1 word 0: none      */
        narrow[3] = 0x3u;  /* c1 word 1: rows 8,9  */
        const PIChartView packed = {
            .bits = narrow, .words = 2, .cov_bits = 8, .rows = 10, .cols = 2
        };
        const int both[] = {0, 1};
        const int first_only[] = {0};
        assert(cover_is_feasible(&packed, both, 2));
        assert(!cover_is_feasible(&packed, first_only, 1));
    }

    puts("cover incumbent validation regression: OK");
    return 0;
}
