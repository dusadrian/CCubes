/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */
#define _POSIX_C_SOURCE 200809L
#include "ipc_chart_polish.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ROWS = 14, COLS = 41, CHARTS = 60 };
static uint32_t seed = UINT32_C(20261116);

static uint32_t next_random(void)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

static uint64_t deadline(void)
{
    struct timespec now;
    assert(clock_gettime(CLOCK_MONOTONIC, &now) == 0);
    return ((uint64_t)now.tv_sec + 4) * UINT64_C(1000000000) + now.tv_nsec;
}

static int oracle(uint64_t rows, const uint64_t *masks, const unsigned char *allowed,
                  int *memo)
{
    if (!rows) return 0;
    if (memo[rows]) return memo[rows];
    uint64_t anchor = rows & (0 - rows);
    int minimum = ROWS + 1;
    for (int c = 0; c < COLS; ++c) {
        if (!allowed[c] || !(anchor & masks[c])) continue;
        int terms = 1 + oracle(rows & ~masks[c], masks, allowed, memo);
        if (terms < minimum) minimum = terms;
    }
    return memo[rows] = minimum;
}

int main(void)
{
    const uint64_t full = (UINT64_C(1) << ROWS) - 1;
    int *memo = calloc((size_t)full + 1, sizeof(int));
    assert(memo);
    for (int test = 0; test < CHARTS; ++test) {
        uint64_t masks[COLS];
        unsigned char allowed[COLS];
        int counts[COLS], solution[ROWS + 1], forced = 0;
        int forced_len = test % 3 == 0;
        for (int c = 0; c < COLS; ++c) {
            uint64_t mask = 0;
            if (c < ROWS) {
                mask = UINT64_C(1) << c;
            } else {
                for (int j = 0; j < 2 + test % 4; ++j)
                    mask |= UINT64_C(1) << (forced_len + next_random() % (ROWS - forced_len));
            }
            masks[c] = mask;
            counts[c] = __builtin_popcountll(mask);
            allowed[c] = c < ROWS || next_random() % 7 != 0;
        }

        memset(memo, 0, ((size_t)full + 1) * sizeof(int));
        int exact = oracle(full, masks, allowed, memo);
        assert(exact - forced_len >= 3);
        const PIChartView chart = {.bits = masks, .words = 1, .cov_bits = 64,
            .rows = ROWS, .cols = COLS};
        for (unsigned workers = 1; workers <= 4; workers *= 2) {
            const size_t memory = 32u * 1024 * 1024;
            IpcChartPolish *polish = ipc_chart_polish_open(&chart, memory, deadline(), workers);
            assert(polish);
            IpcChartPolishQuery query = {.allowed = allowed, .forced = &forced,
                .forced_len = forced_len, .target = exact, .max_row_degree = COLS,
                .coverage_counts = counts, .node_limit = 80000};
            IpcChartPolishResult result = ipc_chart_polish_decide(polish, &query, solution);
            assert(result.status == NSAT_SAT && result.terms == exact);
            assert(result.peak_bytes <= memory && result.nodes <= query.node_limit);
            uint64_t rows = 0, used = 0;
            for (int i = 0; i < result.terms; ++i) {
                int c = solution[i];
                assert(c >= 0 && c < COLS && allowed[c] && !(used & (UINT64_C(1) << c)));
                used |= UINT64_C(1) << c;
                rows |= masks[c];
            }
            assert(rows == full && (!forced_len || (used & 1)));

            --query.target;
            result = ipc_chart_polish_decide(polish, &query, solution);
            assert(result.status == NSAT_UNSAT && result.terms == 0);
            assert(result.peak_bytes <= memory && result.nodes <= query.node_limit);
            ipc_chart_polish_close(polish);

            if (test % 5 == 0) {
                polish = ipc_chart_polish_open(&chart, memory, deadline(), workers);
                assert(polish);
                query.target = exact;
                query.node_limit = 1;
                result = ipc_chart_polish_decide(polish, &query, solution);
                assert(result.status == NSAT_UNKNOWN && result.nodes <= 1);
                ipc_chart_polish_close(polish);
            }
        }
        assert(!ipc_chart_polish_open(&chart, 64, deadline(), 4));
    }
    free(memo);
    puts("Fixed-chart polish matched 180 exact optima and refutations");
    return 0;
}
