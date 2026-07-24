#include "cover_validation.h"

#include <assert.h>
#include <stdio.h>

int main(void) {
    /*
     * Three rows, four columns, column-major:
     * c0={r0}, c1={r1}, c2={r2}, c3={r0,r1,r2}.
     */
    const int chart[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
        1, 1, 1
    };
    const int singleton[] = {3};
    const int three_columns[] = {0, 1, 2};
    const int incomplete[] = {0, 1};
    const int duplicate[] = {3, 3};
    const int out_of_range[] = {4};

    assert(cover_is_feasible(chart, 4, 3, singleton, 1));
    assert(cover_is_feasible(chart, 4, 3, three_columns, 3));
    assert(!cover_is_feasible(chart, 4, 3, incomplete, 2));
    assert(!cover_is_feasible(chart, 4, 3, duplicate, 2));
    assert(!cover_is_feasible(chart, 4, 3, out_of_range, 1));
    assert(!cover_is_feasible(chart, 4, 3, NULL, 0));

    puts("cover incumbent validation regression: OK");
    return 0;
}
