/* Verify the feasible node seed at the solver boundary, independently of
 * heuristic solver quality. The rest of the production path runs unchanged. */
#define progressive_native_master_solve_with_polish_incumbent recording_node_solve
#include "../progressive_native_exact_price.c"
#undef progressive_native_master_solve_with_polish_incumbent
#include <assert.h>
#include <stdio.h>

static int expected_seed, calls;
ProgressiveNativeMasterStatus recording_node_solve(
    const ProgressiveNativeMaster *master, int effort,
    const int *initial, int initial_count, const int *polish, int polish_count,
    int *solution, int *cardinality, double *weights, const double *warm
) {
    (void)effort; (void)warm;
    assert(initial && polish && initial_count == expected_seed);
    assert(polish_count == initial_count);
    uint64_t covered = 0;
    for (int i = 0; i < initial_count; ++i) {
        assert(initial[i] >= 0 && initial[i] < master->count);
        assert(polish[i] == initial[i]);
        for (int j = 0; j < i; ++j) assert(initial[j] != initial[i]);
        covered |= master->coverage[initial[i]];
        solution[i] = initial[i];
    }
    assert(covered == (UINT64_C(1) << master->rows) - 1);
    *cardinality = initial_count;
    memset(weights, 0, (size_t)master->rows * sizeof(double));
    ++calls;
    return PROGRESSIVE_NATIVE_MASTER_OK;
}

static void check_fixture(bool duplicates) {
    int on[32], off[] = {2, 2, 2, 2};
    const int rows = duplicates ? 3 : 8, k = duplicates ? 4 : 9;
    int solution[36] = {0};
    for (int r = 0; r < rows; ++r)
        for (int x = 0; x < 4; ++x) on[r * 4 + x] = 1 + ((r >> x) & 1);
    if (duplicates) {
        /* Two different geometries with equal coverage, plus a duplicate;
         * neither nontrivial coverage class exists in the starting master. */
        int patterns[][4] = {{0, 1, 1, 1}, {1, 0, 1, 1}, {0, 1, 0, 1}, {1, 1, 1, 1}};
        for (int c = 0; c < k; ++c)
            for (int x = 0; x < 4; ++x) solution[x * k + c] = patterns[c][x];
        expected_seed = 3;
    } else {
        for (int c = 0; c < 8; ++c)
            for (int x = 0; x < 4; ++x) solution[x * k + c] = on[c * 4 + x];
        /* The unreleased ninth cube covers rows 6 and 7. Their released
         * minterms project to empty coverage and must not enter the seed. */
        int broad[] = {0, 2, 2, 1};
        for (int x = 0; x < 4; ++x) solution[x * k + 8] = broad[x];
        expected_seed = 6;
    }
    PIstorage pi = {.inputs = 4, .outputs = 1, .ON_minterms = rows, .OFF_minterms = 1,
                    .ON_set = on, .OFF_set = off, .cov_bits = 64, .pichart_words = 1};
    WeightedMMCSPreparedRows prepared = {0};
    ProgressiveNativeMaster master = {0};
    assert(bounded_mmcs_prepare_binary_rows(&pi, 4, &prepared));
    assert(progressive_native_master_initialize(&master, rows, 1, 4));
    for (int r = 0; r < rows; ++r) {
        uint64_t bits = UINT64_C(1) << r;
        bool inserted;
        assert(progressive_native_master_insert(&master, &bits, on + 4 * r, &inserted)
               == PROGRESSIVE_NATIVE_MASTER_OK);
    }
    assert(progressive_solution_is_valid_prepared(&prepared, k, solution));
    double weights[8] = {0};
    int *improved = NULL;
    ProgressiveNativeExactPriceStep step = {0};
    assert(price_node(&pi, &prepared, &master, weights, k, solution, 1, 0, 1.0,
                      0, &improved, &step) == PROGRESSIVE_NATIVE_EXACT_PRICE_OK);
    assert(step.node_cover == expected_seed);
    assert(step.node_cover <= step.released);
    assert(improved && progressive_solution_is_valid_prepared(&prepared,
                                                              step.improved_cardinality, improved));
    free(improved);
    progressive_native_master_destroy(&master);
    bounded_mmcs_prepared_rows_destroy(&prepared);
}
int main(void) {
    check_fixture(true);
    check_fixture(false);
    assert(calls == 2);
    puts("native exact-pricing node seed regression: OK");
}
