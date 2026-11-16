#include "ipc_weighted_cover.h"
#include "ipc_weighted_verify.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static bool matches(const int *cube, size_t stride, const int *point, size_t width)
{
    for (size_t i = 0; i < width; ++i)
        if (cube[i * stride] && cube[i * stride] != point[i]) return false;
    return true;
}

static int brute_force(const NsatCoverInput *q)
{
    unsigned masks[27] = {0};
    for (int code = 0; code < 27; ++code) {
        int cube[3] = {code % 3, code / 3 % 3, code / 9};
        bool legal = true;
        for (size_t r = 0; r < q->negatives; ++r)
            if (matches(cube, 1, q->off + r * 3, 3)) legal = false;
        if (!legal) continue;
        for (size_t r = 0; r < q->positives; ++r)
            if (matches(cube, 1, q->on + r * 3, 3)) masks[code] |= 1u << r;
    }
    unsigned all = (1u << q->positives) - 1;
    int costs[256] = {0};
    for (unsigned set = 1; set <= all; ++set) costs[set] = 8;
    for (unsigned set = 0; set < all; ++set) {
        for (int c = 0; c < 27; ++c) {
            unsigned next = set | masks[c];
            if (costs[set] + 1 < costs[next]) costs[next] = costs[set] + 1;
        }
    }
    return costs[all];
}

static void validate(const NsatCoverInput *q, IpcWeightedCoverResult result, int expected)
{
    assert(result.supported && result.status == NSAT_OK && result.lower == expected);
    assert(result.check_status == (expected > 1 ? NSAT_UNSAT : NSAT_OK));
    if (!result.cover) {
        assert(q->target == (size_t)expected);
        return;
    }
    assert(result.terms == expected && (size_t)expected < q->target);
    for (size_t r = 0; r < q->positives; ++r) {
        bool covered = false;
        for (int t = 0; t < result.terms; ++t)
            covered |= matches(result.cover + t, (size_t)result.terms,
                               q->on + r * q->inputs, q->inputs);
        assert(covered);
    }
    for (size_t r = 0; r < q->negatives; ++r)
        for (int t = 0; t < result.terms; ++t)
            assert(!matches(result.cover + t, (size_t)result.terms,
                            q->off + r * q->inputs, q->inputs));
    free(result.cover);
}

static size_t exhaustive(void)
{
    size_t cases = 0;
    uint64_t repriced = 0;
    for (int table = 0; table < 6561; ++table) {
        int on[24], off[24], values = table;
        size_t positives = 0, negatives = 0;
        for (unsigned point = 0; point < 8; ++point) {
            int value = values % 3;
            values /= 3;
            int *row = value == 1 ? on + 3 * positives++ :
                value == 2 ? off + 3 * negatives++ : NULL;
            if (row)
                for (unsigned i = 0; i < 3; ++i)
                    row[i] = 1 + (int)((point >> i) & 1u);
        }
        if (positives < 2 || !negatives) continue;
        NsatCoverInput q = {.on = on, .off = off, .inputs = 3,
            .positives = positives, .negatives = negatives, .target = positives};
        int expected = brute_force(&q);
        IpcWeightedCoverResult current = ipc_weighted_cover(&q, 1, (NsatLimits){0});
        validate(&q, current, expected);
        IpcWeightedCoverResult warm = ipc_weighted_cover_reprice(&q, 1, (NsatLimits){0}, 1, 128);
        repriced += warm.repriced;
        validate(&q, warm, expected);
        IpcWeightedCoverResult handoff = ipc_weighted_cover_seed_then_reprice(
            &q, 1, (NsatLimits){0}, 1, 128);
        assert(handoff.first_terms == current.first_terms);
        assert(handoff.first_cover_work == current.first_cover_work);
        assert(handoff.first_cover_nodes == current.first_cover_nodes);
        validate(&q, handoff, expected);
        /* A loose maximum forces the checker to decide the target by search. */
        const uint64_t unit_weights[8] = {1,1,1,1,1,1,1,1};
        if (expected > 1)
            assert(ipc_verify_weighted_bound(&q, unit_weights, positives, positives,
                expected, (NsatLimits){0}, NULL) == NSAT_UNSAT);
        if ((size_t)expected < positives)
            assert(ipc_verify_weighted_bound(&q, unit_weights, positives, positives,
                expected + 1, (NsatLimits){0}, NULL) == NSAT_ERROR);
        ++cases;
    }
    assert(repriced > 0);
    return cases;
}

static void lower_target_cover(void)
{
    const unsigned on_points[] = {1, 10, 50, 9, 19, 5, 6, 4, 11, 32, 16, 34};
    const unsigned off_points[] = {23, 36, 62, 59, 26, 58, 43, 31, 15, 45, 40, 20};
    int on[12 * 6], off[12 * 6];
    for (size_t r = 0; r < 12; ++r) {
        for (size_t i = 0; i < 6; ++i) {
            on[r * 6 + i] = 1 + (int)((on_points[r] >> i) & 1u);
            off[r * 6 + i] = 1 + (int)((off_points[r] >> i) & 1u);
        }
    }
    NsatCoverInput q = {.on = on, .off = off, .inputs = 6,
        .positives = 12, .negatives = 12, .target = 8};
    IpcWeightedCoverResult result = ipc_weighted_cover(&q, 1, (NsatLimits){0});
    assert(result.lower_target == 3 && result.first_terms == 3);
    validate(&q, result, 3);

    /* There is room for fixed weights, but not for the projected bases. */
    NsatResources resources = {.memory_limit = 22000};
    result = ipc_weighted_cover_reprice(&q, 1, (NsatLimits){.shared = &resources}, 4, 256);
    assert(result.lower_target == 3 && result.repriced == 0);
    validate(&q, result, 3);
    assert(resources.memory == 0);
}

static void limits_and_width(void)
{
    int on[64][64], off[64][64];
    unsigned p = 0, n = 0;
    for (unsigned point = 0; point < 128; ++point) {
        int *row = (__builtin_popcount(point) & 1) ? on[p++] : off[n++];
        for (size_t i = 0; i < 64; ++i) row[i] = 1;
        for (unsigned i = 0; i < 6; ++i) row[i] += (int)((point >> i) & 1u);
        row[63] += (int)(point >> 6);
    }
    NsatCoverInput q = {.on = on[0], .off = off[0], .inputs = 64,
        .positives = 64, .negatives = 64, .target = 64};
    validate(&q, ipc_weighted_cover(&q, 1, (NsatLimits){0}), 64);
    IpcWeightedCoverResult limited = ipc_weighted_cover(&q, 1, (NsatLimits){.work = 1});
    assert(limited.supported && limited.lower == 1 && !limited.cover);
    assert(limited.status == NSAT_UNKNOWN && limited.stop == NSAT_STOP_WORK);
}

static void many_rows(void)
{
    /* Removing the two specified OFF points permits exactly one pair
     * of ON points to share a cube in this parity table. */
    int on[256][9], off[254][9];
    size_t p = 0, n = 0;
    for (unsigned point = 0; point < 512; ++point) {
        if (point == 1 || point == 2) continue;
        int *row = (__builtin_popcount(point) & 1) ? off[n++] : on[p++];
        for (unsigned i = 0; i < 9; ++i) row[i] = 1 + (int)((point >> i) & 1u);
    }
    NsatCoverInput q = {.on = on[0], .off = off[0], .inputs = 9,
        .positives = p, .negatives = n, .target = p};
    NsatResources resources = {.work_limit = 512000000, .memory_limit = 128u << 20};
    validate(&q, ipc_weighted_cover(&q, 1, (NsatLimits){.shared = &resources}), 255);
    assert(resources.memory == 0);
}

static void wide_covers(void)
{
    const size_t widths[] = {65, 129, 257, 513};
    const int base_on[] = {1,1,1, 2,1,1, 2,2,2}, base_off[] = {1,1,2, 1,2,1};
    int on[3 * 513], off[2 * 513];
    for (size_t w = 0; w < sizeof(widths) / sizeof(*widths); ++w) {
        size_t width = widths[w];
        const size_t active[] = {0, width - 2, width - 1};
        for (size_t i = 0; i < 3 * width; ++i) on[i] = 1;
        for (size_t i = 0; i < 2 * width; ++i) off[i] = 1;
        for (size_t r = 0; r < 3; ++r)
            for (size_t i = 0; i < 3; ++i) on[r * width + active[i]] = base_on[r * 3 + i];
        for (size_t r = 0; r < 2; ++r)
            for (size_t i = 0; i < 3; ++i) off[r * width + active[i]] = base_off[r * 3 + i];
        NsatCoverInput q = {.on = on, .off = off, .inputs = width,
            .positives = 3, .negatives = 2, .target = 3};
        validate(&q, ipc_weighted_cover(&q, 1, (NsatLimits){0}), 2);
        if (width == 513) {
            NsatResources budget = {.memory_limit = 512};
            IpcWeightedCoverResult limited = ipc_weighted_cover(&q, 1, (NsatLimits){.shared = &budget});
            assert(limited.status == NSAT_UNKNOWN && limited.stop == NSAT_STOP_MEMORY);
            assert(limited.lower == 1 && !limited.cover && budget.memory == 0);
        }
    }
}

static void reject_unproved_bounds(void)
{
    const int on[] = {1,1,1, 2,1,1, 2,2,2}, off[] = {1,1,2, 1,2,1};
    const uint64_t weights[] = {1, 1, 1};
    NsatCoverInput q = {.on = on, .off = off, .inputs = 3,
        .positives = 3, .negatives = 2, .target = 3};
    NsatStats stats;
    assert(ipc_verify_weighted_bound(&q, weights, 3, 3, 2, (NsatLimits){0}, &stats)
        == NSAT_UNSAT);
    assert(ipc_verify_weighted_bound(&q, weights, 3, 1, 3, (NsatLimits){0}, &stats)
        == NSAT_ERROR && stats.stop == NSAT_STOP_PROOF);
    assert(ipc_verify_weighted_bound(&q, weights, 3, 2, 3, (NsatLimits){0}, &stats)
        == NSAT_ERROR && stats.stop == NSAT_STOP_PROOF);
    const uint64_t overflow[] = {UINT64_MAX, 1, 2};
    assert(ipc_verify_weighted_bound(&q, overflow, 3, 2, 2, (NsatLimits){0}, &stats)
        == NSAT_ERROR && stats.stop == NSAT_STOP_INPUT);
    assert(ipc_verify_weighted_bound(&q, weights, 3, 2, 2, (NsatLimits){.memory_bytes = 1}, &stats)
        == NSAT_UNKNOWN && stats.stop == NSAT_STOP_MEMORY);

    NsatResources resources = {0};
    q.target = 2;
    IpcWeightedCoverResult result = ipc_weighted_cover(&q, 1, (NsatLimits){.shared = &resources});
    uint64_t pricing_work = resources.work;
    uint64_t search_work = pricing_work - result.check.work;
    assert(result.nodes == 0 && result.check.work > 1);
    validate(&q, result, 2);

    /* The same pricing and check fit, but no search work remains. */
    q.target = 3;
    result = ipc_weighted_cover(&q, 1, (NsatLimits){.work = pricing_work});
    assert(result.check_status == NSAT_UNSAT && result.lower == 2 && !result.cover);
    assert(result.status == NSAT_UNKNOWN && result.stop == NSAT_STOP_WORK);

    result = ipc_weighted_cover(&q, 1, (NsatLimits){.work = pricing_work - 1});
    assert(result.check_status == NSAT_UNKNOWN && result.check.stop == NSAT_STOP_WORK);
    assert(result.status == NSAT_UNKNOWN && result.lower == 1 && !result.cover);

    /* Exactly enough search work for pricing. The independent check can
     * finish only if charged to its own allowance. */
    q.target = 2;
    IpcWeightedCheckBudget budget = {.search_seconds = 10, .verifier_seconds = 10};
    resources = (NsatResources){0};
    NsatLimits limits = {.work = search_work, .shared = &resources};
    result = ipc_weighted_cover_reprice_with_check_budget(&q, 1, limits, 4, 256, &budget);
    validate(&q, result, 2);
    assert(resources.work == pricing_work);

    resources = (NsatResources){.work_limit = pricing_work - 1};
    result = ipc_weighted_cover_reprice_with_check_budget(&q, 1, limits, 4, 256, &budget);
    assert(result.lower == 1 && result.check_status == NSAT_UNKNOWN &&
           result.check.stop == NSAT_STOP_WORK && resources.work <= resources.work_limit);

    resources = (NsatResources){0};
    budget.verifier_work = 1;
    result = ipc_weighted_cover_reprice_with_check_budget(&q, 1, limits, 4, 256, &budget);
    assert(result.lower == 1 && result.check_status == NSAT_UNKNOWN &&
           result.check.stop == NSAT_STOP_WORK && result.check.work <= budget.verifier_work);

    resources = (NsatResources){0};
    budget.verifier_work = 0;
    budget.verifier_seconds = 1e-12;
    result = ipc_weighted_cover_reprice_with_check_budget(&q, 1, limits, 4, 256, &budget);
    assert(result.lower == 1 && result.check_status == NSAT_UNKNOWN &&
           result.check.stop == NSAT_STOP_CANCELLED);

    resources = (NsatResources){0};
    limits.work = 0;
    result = ipc_weighted_cover_reprice_with_auto_check(&q, 1, limits, 4, 256);
    validate(&q, result, 2);
    assert(result.check_auto_base == search_work);
    assert(result.check_auto_limit == search_work + search_work / 8);
    assert(resources.work == pricing_work && !result.check_auto_limit_hit);

    resources = (NsatResources){.work_limit = pricing_work - 1};
    result = ipc_weighted_cover_reprice_with_auto_check(&q, 1, limits, 4, 256);
    assert(result.lower == 1 && result.check_status == NSAT_UNKNOWN);
    assert(result.check.stop == NSAT_STOP_WORK && !result.check_auto_limit_hit);
    assert(resources.work <= resources.work_limit);
}

static void adaptive_root_pricing(void)
{
    /* The ordinary cover has 11 cubes and initial IPC pricing proves 9.
     * The first 1,024 root passes leave the nine-cube question open. */
    FILE *fixture = fopen("tests/fixtures/weighted-30x160.txt", "r");
    assert(fixture);
    int inputs, positives, negatives, slots;
    assert(fscanf(fixture, "%d%d%d%d", &inputs, &positives, &negatives, &slots) == 4);
    assert(inputs == 30 && positives == 80 && negatives == 80 && slots == 80);
    int on[80 * 30], off[80 * 30];
    for (int r = 0; r < 160; ++r) {
        int *point = r < 80 ? on + r * 30 : off + (r - 80) * 30;
        for (int i = 0; i < 30; ++i) {
            int bit;
            assert(fscanf(fixture, " %1d", &bit) == 1 && (bit == 0 || bit == 1));
            point[i] = bit + 1;
        }
    }
    fclose(fixture);

    NsatCoverInput q = {.on = on, .off = off, .inputs = 30,
        .positives = 80, .negatives = 80, .target = 11};
    NsatResources resources = {.work_limit = 700000000, .memory_limit = 128u << 20};
    NsatLimits limits = {.shared = &resources};
    IpcWeightedCoverResult result = ipc_weighted_cover_reprice(&q, 9, limits, 4, 256);
    assert(result.pivots > 1024 && result.lower_target == 9);
    assert(result.lower == 10 && result.check_status == NSAT_UNSAT);
    assert(resources.work <= resources.work_limit && resources.memory == 0);
    free(result.cover);

    resources = (NsatResources){.memory_limit = 128u << 20};
    limits.work = 512000000;
    IpcWeightedCheckBudget budget = {
        .search_seconds = 15, .verifier_seconds = 15, .verifier_work = 300000000};
    result = ipc_weighted_cover_reprice_with_check_budget(&q, 9, limits, 4, 256, &budget);
    assert(result.lower == 10 && result.check_status == NSAT_UNSAT);
    assert(resources.work > limits.work && resources.work - result.check.work <= limits.work);
    assert(result.check.work <= budget.verifier_work && resources.memory == 0);
    free(result.cover);
}

int main(void)
{
    size_t cases = exhaustive();
    limits_and_width();
    wide_covers();
    many_rows();
    lower_target_cover();
    reject_unproved_bounds();
    adaptive_root_pricing();
    printf("IPC weighted integer cover: %zu exhaustive tables, wide covers and checked bounds\n", cases);
    return 0;
}
