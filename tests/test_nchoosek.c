#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "binomial.h"

int main(void) {
    /* The calculation fallback remains valid before a table is prepared. */
    assert(nchoosek(5, 2) == UINT64_C(10));
    assert(nchoosek(-1, 0) == 0);
    assert(nchoosek(5, -1) == 0);
    assert(nchoosek(5, 6) == 0);

    assert(nchoosek_prepare(100));
    assert(nchoosek(0, 0) == 1);
    assert(nchoosek(50, 0) == 1);
    assert(nchoosek(50, 50) == 1);
    assert(nchoosek(50, 6) == UINT64_C(15890700));
    assert(nchoosek(67, 33) == UINT64_C(14226520737620288370));
    assert(nchoosek(68, 34) == 0);

    for (int n = 2; n <= 67; ++n) {
        for (int k = 1; k < n; ++k) {
            uint64_t value = nchoosek(n, k);
            uint64_t left = nchoosek(n - 1, k - 1);
            uint64_t right = nchoosek(n - 1, k);
            if (left != 0 && right != 0 && UINT64_MAX - left >= right) {
                assert(value == left + right);
            } else {
                assert(value == 0);
            }
            assert(value == nchoosek(n, n - k));
        }
    }

    /* Preparing a larger table preserves prior values and extends the range. */
    assert(nchoosek_prepare(150));
    assert(nchoosek(50, 6) == UINT64_C(15890700));
    assert(nchoosek(100, 3) == UINT64_C(161700));
    assert(nchoosek(150, 1) == UINT64_C(150));

    puts("binomial lookup regression: OK");
    return 0;
}
