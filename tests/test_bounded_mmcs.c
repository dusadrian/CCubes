#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bounded_mmcs.h"
#include "ccubes_threads.h"

static uint32_t rng_state = UINT32_C(0x6d2b79f5);
static bool parallel_pricing_checked = false;
static int weight_profile;

static uint32_t next_random(void) {
    rng_state = rng_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return rng_state;
}

static int popcount_u64(uint64_t value) {
    int count = 0;
    while (value) {
        value &= value - 1u;
        count++;
    }
    return count;
}

static bool cube_covers(
    uint64_t row,
    uint64_t fixed,
    uint64_t value
) {
    return (row & fixed) == value;
}

static bool brute_is_prime(
    const uint64_t *on_rows,
    int on_count,
    const uint64_t *off_rows,
    int off_count,
    uint64_t fixed,
    uint64_t value
) {
    bool covers_on = false;
    for (int row = 0; row < on_count; ++row) {
        if (cube_covers(on_rows[row], fixed, value)) {
            covers_on = true;
            break;
        }
    }
    if (!covers_on) return false;

    for (int row = 0; row < off_count; ++row) {
        if (cube_covers(off_rows[row], fixed, value)) return false;
    }

    uint64_t remaining = fixed;
    while (remaining) {
        uint64_t bit = remaining & (~remaining + 1u);
        uint64_t parent_fixed = fixed & ~bit;
        uint64_t parent_value = value & parent_fixed;
        bool parent_is_implicant = true;
        for (int row = 0; row < off_count; ++row) {
            if (cube_covers(off_rows[row], parent_fixed, parent_value)) {
                parent_is_implicant = false;
                break;
            }
        }
        if (parent_is_implicant) return false;
        remaining &= remaining - 1u;
    }
    return true;
}

static void free_generated_arrays(
    PIstorage *pi
) {
    free(pi->pichart_pos);
    free(pi->implicants_pos);
    free(pi->implicants_val);
    free(pi->shared);
    free(pi->covsum);
    free(pi->covered);
    pi->pichart_pos = NULL;
    pi->implicants_pos = NULL;
    pi->implicants_val = NULL;
    pi->shared = NULL;
    pi->covsum = NULL;
    pi->covered = NULL;
}

static void verify_function(
    int ninputs,
    const unsigned char *is_on,
    bool check_pricing
) {
    int rows = 1 << ninputs;
    int on_count = 0;
    for (int row = 0; row < rows; ++row) on_count += is_on[row] != 0;
    int off_count = rows - on_count;
    assert(on_count > 0 && off_count > 0);

    int *on_set = calloc((size_t)on_count * (size_t)ninputs, sizeof(int));
    int *off_set = calloc((size_t)off_count * (size_t)ninputs, sizeof(int));
    uint64_t *on_rows = calloc((size_t)on_count, sizeof(uint64_t));
    uint64_t *off_rows = calloc((size_t)off_count, sizeof(uint64_t));
    assert(on_set && off_set && on_rows && off_rows);

    int on = 0;
    int off = 0;
    for (int row = 0; row < rows; ++row) {
        int *target = is_on[row]
            ? &on_set[(size_t)on * (size_t)ninputs]
            : &off_set[(size_t)off * (size_t)ninputs];
        for (int input = 0; input < ninputs; ++input) {
            target[input] = ((row >> input) & 1) ? 2 : 1;
        }
        if (is_on[row]) {
            on_rows[on++] = (uint64_t)row;
        } else {
            off_rows[off++] = (uint64_t)row;
        }
    }

    int word_index[8];
    int bit_index[8];
    uint64_t shifted_mask[8];
    for (int input = 0; input < ninputs; ++input) {
        word_index[input] = 0;
        bit_index[input] = input * 2;
        shifted_mask[input] = UINT64_C(3) << bit_index[input];
    }

    for (int level = 1; level <= ninputs; ++level) {
        PIstorage pi;
        memset(&pi, 0, sizeof(pi));
        pi.ON_minterms = on_count;
        pi.OFF_minterms = off_count;
        pi.ON_set = on_set;
        pi.OFF_set = off_set;
        pi.pichart_words = (on_count + 63) / 64;
        pi.cov_bits = 64;

        BoundedMMCSStats stats;
        memset(&stats, 0, sizeof(stats));
        assert(bounded_mmcs_generate_output_level(
            &pi,
            ninputs,
            level,
            word_index,
            bit_index,
            shifted_mask,
            1,
            &stats
        ));

        int brute_count = 0;
        uint64_t universe = (UINT64_C(1) << ninputs) - 1u;
        for (uint64_t fixed = 1; fixed <= universe; ++fixed) {
            if (popcount_u64(fixed) != level) continue;
            uint64_t value = fixed;
            while (1) {
                if (brute_is_prime(
                    on_rows,
                    on_count,
                    off_rows,
                    off_count,
                    fixed,
                    value
                )) {
                    brute_count++;
                    uint64_t packed_fixed = 0;
                    uint64_t packed_value = 0;
                    for (int input = 0; input < ninputs; ++input) {
                        uint64_t bit = UINT64_C(1) << input;
                        if ((fixed & bit) == 0) continue;
                        packed_fixed |= shifted_mask[input];
                        if (value & bit) {
                            packed_value |= UINT64_C(1) << bit_index[input];
                        }
                    }
                    bool found = false;
                    for (int cube = 0; cube < pi.foundPI; ++cube) {
                        if (
                            pi.implicants_pos[cube] == packed_fixed &&
                            pi.implicants_val[cube] == packed_value
                        ) {
                            found = true;
                            break;
                        }
                    }
                    assert(found);
                }
                if (value == 0) break;
                value = (value - 1u) & fixed;
            }
        }

        assert(pi.foundPI == brute_count);
        assert(stats.unique_cubes == (uint64_t)brute_count);

        for (int cube = 0; cube < pi.foundPI; ++cube) {
            uint64_t fixed = 0;
            uint64_t value = 0;
            for (int input = 0; input < ninputs; ++input) {
                if (pi.implicants_pos[cube] & shifted_mask[input]) {
                    fixed |= UINT64_C(1) << input;
                    if (
                        pi.implicants_val[cube] &
                        (UINT64_C(1) << bit_index[input])
                    ) {
                        value |= UINT64_C(1) << input;
                    }
                }
            }
            assert(popcount_u64(fixed) == level);
            assert(brute_is_prime(
                on_rows,
                on_count,
                off_rows,
                off_count,
                fixed,
                value
            ));

            int covsum = 0;
            for (int row = 0; row < on_count; ++row) {
                bool expected = cube_covers(on_rows[row], fixed, value);
                bool actual = (
                    pi.pichart_pos[
                        (size_t)cube * (size_t)pi.pichart_words +
                        (size_t)(row / 64)
                    ] & (UINT64_C(1) << (row % 64))
                ) != 0;
                assert(actual == expected);
                covsum += expected;
            }
            assert(pi.covsum[cube] == covsum);
        }

        free_generated_arrays(&pi);
    }

    PIstorage complete;
    memset(&complete, 0, sizeof(complete));
    complete.ON_minterms = on_count;
    complete.OFF_minterms = off_count;
    complete.ON_set = on_set;
    complete.OFF_set = off_set;
    complete.pichart_words = (on_count + 63) / 64;
    complete.cov_bits = 64;

    BoundedMMCSStats complete_stats;
    memset(&complete_stats, 0, sizeof(complete_stats));
    assert(bounded_mmcs_generate_output_all(
        &complete,
        ninputs,
        word_index,
        bit_index,
        shifted_mask,
        1,
        &complete_stats
    ));

    int complete_brute_count = 0;
    uint64_t universe = (UINT64_C(1) << ninputs) - 1u;
    for (uint64_t fixed = 1; fixed <= universe; ++fixed) {
        uint64_t value = fixed;
        while (1) {
            if (brute_is_prime(
                on_rows,
                on_count,
                off_rows,
                off_count,
                fixed,
                value
            )) {
                complete_brute_count++;
                uint64_t packed_fixed = 0;
                uint64_t packed_value = 0;
                for (int input = 0; input < ninputs; ++input) {
                    uint64_t bit = UINT64_C(1) << input;
                    if ((fixed & bit) == 0) continue;
                    packed_fixed |= shifted_mask[input];
                    if (value & bit) {
                        packed_value |= UINT64_C(1) << bit_index[input];
                    }
                }

                bool found = false;
                for (int cube = 0; cube < complete.foundPI; ++cube) {
                    if (
                        complete.implicants_pos[cube] == packed_fixed &&
                        complete.implicants_val[cube] == packed_value
                    ) {
                        found = true;
                        break;
                    }
                }
                assert(found);
            }
            if (value == 0) break;
            value = (value - 1u) & fixed;
        }
    }

    assert(complete.foundPI == complete_brute_count);
    assert(complete_stats.unique_cubes == (uint64_t)complete_brute_count);
    free_generated_arrays(&complete);

    if (!check_pricing) {
        free(on_set);
        free(off_set);
        free(on_rows);
        free(off_rows);
        return;
    }

    double weights[256];
    for (int row = 0; row < on_count; ++row) {
        weights[row] = weight_profile == 1 ? 1.0 : weight_profile == 2
            ? (row % 7 == 0 ? 0.0 : row % 7 == 1 ? -0.125 : 1.25)
            : 0.25 + (double)((row * 17 + ninputs) % 13) / 7.0;
    }
    int max_level = 1 + (int)(next_random() % (uint32_t)ninputs);
    double brute_best = -1.0;
    for (uint64_t fixed = 1; fixed <= universe; ++fixed) {
        if (popcount_u64(fixed) > max_level) continue;
        uint64_t value = fixed;
        while (1) {
            if (brute_is_prime(
                on_rows,
                on_count,
                off_rows,
                off_count,
                fixed,
                value
            )) {
                double score = 0.0;
                for (int row = 0; row < on_count; ++row) {
                    if (cube_covers(on_rows[row], fixed, value)) {
                        score += weights[row];
                    }
                }
                if (score > brute_best) brute_best = score;
            }
            if (value == 0) break;
            value = (value - 1u) & fixed;
        }
    }

    PIstorage price_pi;
    memset(&price_pi, 0, sizeof(price_pi));
    price_pi.ON_minterms = on_count;
    price_pi.OFF_minterms = off_count;
    price_pi.ON_set = on_set;
    price_pi.OFF_set = off_set;
    price_pi.pichart_words = (on_count + 63) / 64;
    price_pi.cov_bits = 64;
    int best_vertices[8];
    int batch_capacity = on_count < 4 ? on_count : 4;
    double batch_weights[4];
    int batch_anchors[4];
    int batch_support_sizes[4];
    int batch_vertices[4 * 8];
    uint64_t batch_coverage[4 * 4];
    WeightedMMCSPriceBatch batch = {
        .capacity = batch_capacity,
        .weights = batch_weights,
        .anchors = batch_anchors,
        .support_sizes = batch_support_sizes,
        .vertices = batch_vertices,
        .coverage = batch_coverage
    };
    WeightedMMCSPriceResult price_result;
    WeightedMMCSPriceStatus price_status = bounded_mmcs_price_output_batch(
        &price_pi,
        ninputs,
        max_level,
        weights,
        best_vertices,
        &batch,
        &price_result
    );
    /*
     * The max-cover test is only an admissible pruning bound. Verify across
     * every exhaustive/randomized fixture that removing it changes search
     * effort, not the priced columns.
     */
    {
        WeightedMMCSPreparedRows prepared;
        assert(bounded_mmcs_prepare_binary_rows(
            &price_pi,
            ninputs,
            &prepared
        ));
        double ablated_weights[4];
        int ablated_anchors[4];
        int ablated_support_sizes[4];
        int ablated_vertices[4 * 8];
        uint64_t ablated_coverage[4 * 4];
        int ablated_best_vertices[8];
        WeightedMMCSPriceBatch ablated_batch = {
            .capacity = batch_capacity,
            .weights = ablated_weights,
            .anchors = ablated_anchors,
            .support_sizes = ablated_support_sizes,
            .vertices = ablated_vertices,
            .coverage = ablated_coverage
        };
        WeightedMMCSPriceOptions ablated_options = {
            .requested_threads = 1,
            .collect_timing = true,
            .disable_cover_bound = true
        };
        WeightedMMCSPriceResult ablated_result;
        WeightedMMCSPriceStatus ablated_status =
            bounded_mmcs_price_output_batch_prepared(
                &price_pi,
                &prepared,
                max_level,
                weights,
                ablated_best_vertices,
                &ablated_batch,
                &ablated_options,
                &ablated_result
            );
        assert(ablated_status == price_status);
        assert(ablated_result.best_weight == price_result.best_weight);
        assert(ablated_result.best_anchor == price_result.best_anchor);
        assert(
            ablated_result.best_support_size ==
            price_result.best_support_size
        );
        assert(memcmp(
            ablated_best_vertices,
            best_vertices,
            (size_t)price_result.best_support_size * sizeof(int)
        ) == 0);
        assert(ablated_batch.count == batch.count);
        assert(memcmp(
            ablated_batch.weights,
            batch.weights,
            (size_t)batch.count * sizeof(double)
        ) == 0);
        assert(memcmp(
            ablated_batch.anchors,
            batch.anchors,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.support_sizes,
            batch.support_sizes,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.vertices,
            batch.vertices,
            (size_t)batch.count * (size_t)ninputs * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.coverage,
            batch.coverage,
            (size_t)batch.count *
                (size_t)price_pi.pichart_words *
                sizeof(uint64_t)
        ) == 0);
        assert(
            ablated_result.search_nodes >= price_result.search_nodes
        );
        assert(
            ablated_result.telemetry.edge_literal_entries_visited == 0
        );

        WeightedMMCSPriceOptions cached_options = {
            .requested_threads = 1,
            .collect_timing = true,
            .cache_edge_words = true
        };
        WeightedMMCSPriceResult cached_result;
        WeightedMMCSPriceStatus cached_status =
            bounded_mmcs_price_output_batch_prepared(
                &price_pi,
                &prepared,
                max_level,
                weights,
                ablated_best_vertices,
                &ablated_batch,
                &cached_options,
                &cached_result
            );
        assert(cached_status == price_status);
        assert(cached_result.best_weight == price_result.best_weight);
        assert(cached_result.best_anchor == price_result.best_anchor);
        assert(
            cached_result.best_support_size ==
            price_result.best_support_size
        );
        assert(memcmp(
            ablated_best_vertices,
            best_vertices,
            (size_t)price_result.best_support_size * sizeof(int)
        ) == 0);
        assert(ablated_batch.count == batch.count);
        assert(memcmp(
            ablated_batch.weights,
            batch.weights,
            (size_t)batch.count * sizeof(double)
        ) == 0);
        assert(memcmp(
            ablated_batch.anchors,
            batch.anchors,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.support_sizes,
            batch.support_sizes,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.vertices,
            batch.vertices,
            (size_t)batch.count * (size_t)ninputs * sizeof(int)
        ) == 0);
        assert(memcmp(
            ablated_batch.coverage,
            batch.coverage,
            (size_t)batch.count *
                (size_t)price_pi.pichart_words *
                sizeof(uint64_t)
        ) == 0);
        assert(cached_result.search_nodes == price_result.search_nodes);
        bounded_mmcs_prepared_rows_destroy(&prepared);
    }
    uint64_t expected_edges =
        (uint64_t)on_count * (uint64_t)off_count +
        (uint64_t)on_count * (uint64_t)(on_count - 1) / 2u;
    uint64_t expected_gated_anchors = 0;
    uint64_t expected_gated_edges = 0;
    uint64_t expected_gated_incidences = 0;
    double root_suffix = 0.0;
    for (int anchor_index = on_count - 1; anchor_index >= 0; --anchor_index) {
        if (weights[anchor_index] > 0.0) {
            root_suffix += weights[anchor_index];
        }
        if (root_suffix > 1.0 + 1e-8) continue;
        expected_gated_anchors++;
        expected_gated_edges +=
            (uint64_t)off_count + (uint64_t)anchor_index;
        const int *anchor = &on_set[
            (size_t)anchor_index * (size_t)ninputs
        ];
        for (int row = 0; row < off_count; ++row) {
            const int *other = &off_set[(size_t)row * (size_t)ninputs];
            for (int input = 0; input < ninputs; ++input) {
                expected_gated_incidences +=
                    anchor[input] != other[input];
            }
        }
        for (int row = 0; row < anchor_index; ++row) {
            const int *other = &on_set[(size_t)row * (size_t)ninputs];
            for (int input = 0; input < ninputs; ++input) {
                expected_gated_incidences +=
                    anchor[input] != other[input];
            }
        }
    }
    assert(
        price_result.telemetry.anchors_presented == (uint64_t)on_count
    );
    assert(
        price_result.telemetry.anchors_constructed ==
        (uint64_t)on_count
    );
    assert(price_result.telemetry.edges_constructed == 0);
    assert(
        price_result.telemetry.xor_words ==
        expected_edges *
            (uint64_t)((ninputs + 63) / 64)
    );
    assert(price_result.telemetry.incidences_counted == 0);
    assert(price_result.telemetry.incidences_written == 0);
    assert(
        price_result.telemetry.prospective_gated_anchors ==
        expected_gated_anchors
    );
    assert(
        price_result.telemetry.prospective_gated_edges ==
        expected_gated_edges
    );
    assert(
        price_result.telemetry.prospective_gated_incidences ==
        expected_gated_incidences
    );
    assert(
        price_result.telemetry.pivot_edge_entries_inspected >=
        price_result.telemetry.pivot_selections
    );
    if (brute_best >= 0.0) {
        assert(price_status == WEIGHTED_MMCS_PRICE_OK);
        assert(fabs(price_result.best_weight - brute_best) < 1e-10);
        assert(price_result.best_anchor >= 0);
        assert(price_result.best_support_size <= max_level);
        assert(batch.count >= 0 && batch.count <= batch_capacity);
        if (brute_best > 1.0 + 1e-8) {
            assert(batch.count > 0);
            assert(fabs(batch.weights[0] - brute_best) < 1e-10);
        }
        for (int cube = 0; cube < batch.count; ++cube) {
            assert(batch.weights[cube] > 1.0 + 1e-8);
            if (cube > 0) {
                assert(batch.weights[cube - 1] >= batch.weights[cube]);
            }
            assert(
                batch.anchors[cube] >= 0 &&
                batch.anchors[cube] < on_count
            );
            assert(
                batch.support_sizes[cube] > 0 &&
                batch.support_sizes[cube] <= max_level
            );

            uint64_t fixed = 0;
            uint64_t value = 0;
            const int *anchor = &on_set[
                (size_t)batch.anchors[cube] * (size_t)ninputs
            ];
            for (int i = 0; i < batch.support_sizes[cube]; ++i) {
                int vertex = batch.vertices[
                    (size_t)cube * (size_t)ninputs + (size_t)i
                ];
                assert(vertex >= 0 && vertex < ninputs);
                uint64_t bit = UINT64_C(1) << vertex;
                assert((fixed & bit) == 0);
                fixed |= bit;
                if (anchor[vertex] == 2) value |= bit;
            }
            assert(brute_is_prime(
                on_rows,
                on_count,
                off_rows,
                off_count,
                fixed,
                value
            ));

            double score = 0.0;
            for (int row = 0; row < on_count; ++row) {
                bool expected = cube_covers(
                    on_rows[row],
                    fixed,
                    value
                );
                bool actual = (
                    batch.coverage[
                        (size_t)cube *
                            (size_t)price_pi.pichart_words +
                        (size_t)(row / 64)
                    ] & (UINT64_C(1) << (row % 64))
                ) != 0;
                assert(actual == expected);
                if (expected) score += weights[row];
            }
            assert(fabs(score - batch.weights[cube]) < 1e-10);
            for (int earlier = 0; earlier < cube; ++earlier) {
                assert(memcmp(
                    &batch.coverage[
                        (size_t)earlier *
                            (size_t)price_pi.pichart_words
                    ],
                    &batch.coverage[
                        (size_t)cube *
                            (size_t)price_pi.pichart_words
                    ],
                    (size_t)price_pi.pichart_words * sizeof(uint64_t)
                ) != 0);
            }
        }
    } else {
        assert(price_status == WEIGHTED_MMCS_PRICE_OK);
        assert(price_result.best_anchor == -1);
        assert(isinf(price_result.best_weight));
        assert(price_result.best_weight < 0.0);
        assert(batch.count == 0);
    }

    if (
        !parallel_pricing_checked &&
        price_status == WEIGHTED_MMCS_PRICE_OK &&
        price_result.best_anchor >= 0
    ) {
        double parallel_weights[4];
        int parallel_anchors[4];
        int parallel_support_sizes[4];
        int parallel_vertices[4 * 8];
        uint64_t parallel_coverage[4 * 4];
        int parallel_best_vertices[8];
        WeightedMMCSPriceBatch parallel_batch = {
            .capacity = batch_capacity,
            .weights = parallel_weights,
            .anchors = parallel_anchors,
            .support_sizes = parallel_support_sizes,
            .vertices = parallel_vertices,
            .coverage = parallel_coverage
        };
        WeightedMMCSPriceResult parallel_result;

        for (int repeat = 0; repeat < 8; ++repeat) {
            ccubes_set_thread_count(4);
            WeightedMMCSPriceStatus parallel_status =
                bounded_mmcs_price_output_batch(
                &price_pi,
                ninputs,
                max_level,
                weights,
                parallel_best_vertices,
                &parallel_batch,
                &parallel_result
            );
            ccubes_set_thread_count(1);

            assert(parallel_status == price_status);
            assert(
                parallel_result.best_weight == price_result.best_weight
            );
            assert(parallel_result.best_anchor == price_result.best_anchor);
            assert(
                parallel_result.best_support_size ==
                price_result.best_support_size
            );
            assert(
                parallel_result.telemetry.anchors_presented ==
                price_result.telemetry.anchors_presented
            );
            assert(
                parallel_result.telemetry.edges_constructed ==
                price_result.telemetry.edges_constructed
            );
            assert(
                parallel_result.telemetry.xor_words ==
                price_result.telemetry.xor_words
            );
            assert(
                parallel_result.telemetry.incidences_counted ==
                price_result.telemetry.incidences_counted
            );
            assert(
                parallel_result.telemetry.prospective_gated_anchors ==
                price_result.telemetry.prospective_gated_anchors
            );
            assert(
                parallel_result.telemetry.prospective_gated_edges ==
                price_result.telemetry.prospective_gated_edges
            );
            assert(
                parallel_result.telemetry.prospective_gated_incidences ==
                price_result.telemetry.prospective_gated_incidences
            );
            assert(memcmp(
                parallel_best_vertices,
                best_vertices,
                (size_t)price_result.best_support_size * sizeof(int)
            ) == 0);
            assert(parallel_batch.count == batch.count);
            assert(memcmp(
                parallel_batch.weights,
                batch.weights,
                (size_t)batch.count * sizeof(double)
            ) == 0);
            assert(memcmp(
                parallel_batch.anchors,
                batch.anchors,
                (size_t)batch.count * sizeof(int)
            ) == 0);
            assert(memcmp(
                parallel_batch.support_sizes,
                batch.support_sizes,
                (size_t)batch.count * sizeof(int)
            ) == 0);
            assert(memcmp(
                parallel_batch.vertices,
                batch.vertices,
                (size_t)batch.count * (size_t)ninputs * sizeof(int)
            ) == 0);
            assert(memcmp(
                parallel_batch.coverage,
                batch.coverage,
                (size_t)batch.count *
                    (size_t)price_pi.pichart_words *
                    sizeof(uint64_t)
            ) == 0);
        }

        int unbatched_serial_vertices[8];
        int unbatched_parallel_vertices[8];
        WeightedMMCSPriceResult unbatched_serial;
        WeightedMMCSPriceResult unbatched_parallel;
        ccubes_set_thread_count(1);
        assert(bounded_mmcs_price_output(
            &price_pi,
            ninputs,
            max_level,
            weights,
            unbatched_serial_vertices,
            &unbatched_serial
        ) == WEIGHTED_MMCS_PRICE_OK);
        ccubes_set_thread_count(4);
        assert(bounded_mmcs_price_output(
            &price_pi,
            ninputs,
            max_level,
            weights,
            unbatched_parallel_vertices,
            &unbatched_parallel
        ) == WEIGHTED_MMCS_PRICE_OK);
        ccubes_set_thread_count(1);
        assert(
            unbatched_parallel.best_weight ==
            unbatched_serial.best_weight
        );
        assert(
            unbatched_parallel.best_anchor ==
            unbatched_serial.best_anchor
        );
        assert(
            unbatched_parallel.best_support_size ==
            unbatched_serial.best_support_size
        );
        assert(memcmp(
            unbatched_parallel_vertices,
            unbatched_serial_vertices,
            (size_t)unbatched_serial.best_support_size * sizeof(int)
        ) == 0);

        WeightedMMCSPreparedRows prepared;
        assert(bounded_mmcs_prepare_binary_rows(
            &price_pi,
            ninputs,
            &prepared
        ));
        double reference_weights[4];
        int reference_anchors[4];
        int reference_support_sizes[4];
        int reference_vertices[4 * 8];
        uint64_t reference_coverage[4 * 4];
        int reference_best_vertices[8];
        WeightedMMCSPriceBatch reference_batch = {
            .capacity = batch_capacity,
            .weights = reference_weights,
            .anchors = reference_anchors,
            .support_sizes = reference_support_sizes,
            .vertices = reference_vertices,
            .coverage = reference_coverage
        };
        WeightedMMCSPriceOptions reference_options = {
            .requested_threads = 1,
            .use_materialized_backend = true
        };
        WeightedMMCSPriceResult reference_result;
        assert(
            bounded_mmcs_price_output_batch_prepared(
                &price_pi,
                &prepared,
                max_level,
                weights,
                reference_best_vertices,
                &reference_batch,
                &reference_options,
                &reference_result
            ) == price_status
        );
        assert(reference_result.best_weight == price_result.best_weight);
        assert(reference_result.best_anchor == price_result.best_anchor);
        assert(
            reference_result.best_support_size ==
            price_result.best_support_size
        );
        assert(memcmp(
            reference_best_vertices,
            best_vertices,
            (size_t)price_result.best_support_size * sizeof(int)
        ) == 0);
        assert(reference_batch.count == batch.count);
        assert(memcmp(
            reference_batch.weights,
            batch.weights,
            (size_t)batch.count * sizeof(double)
        ) == 0);
        assert(memcmp(
            reference_batch.anchors,
            batch.anchors,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.support_sizes,
            batch.support_sizes,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.vertices,
            batch.vertices,
            (size_t)batch.count * (size_t)ninputs * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.coverage,
            batch.coverage,
            (size_t)batch.count *
                (size_t)price_pi.pichart_words *
                sizeof(uint64_t)
        ) == 0);
        assert(
            reference_result.telemetry.anchors_constructed ==
            (uint64_t)on_count
        );
        assert(reference_result.telemetry.edges_constructed == expected_edges);
        assert(
            reference_result.telemetry.incidences_counted ==
            reference_result.telemetry.incidences_written
        );

        WeightedMMCSPriceOptions gated_options = {
            .requested_threads = 1,
            .enable_root_gate = true
        };
        WeightedMMCSPriceResult gated_result;
        assert(
            bounded_mmcs_price_output_batch_prepared(
                &price_pi,
                &prepared,
                max_level,
                weights,
                reference_best_vertices,
                &reference_batch,
                &gated_options,
                &gated_result
            ) == price_status
        );
        assert(reference_batch.count == batch.count);
        assert(memcmp(
            reference_batch.weights,
            batch.weights,
            (size_t)batch.count * sizeof(double)
        ) == 0);
        assert(memcmp(
            reference_batch.anchors,
            batch.anchors,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.support_sizes,
            batch.support_sizes,
            (size_t)batch.count * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.vertices,
            batch.vertices,
            (size_t)batch.count * (size_t)ninputs * sizeof(int)
        ) == 0);
        assert(memcmp(
            reference_batch.coverage,
            batch.coverage,
            (size_t)batch.count *
                (size_t)price_pi.pichart_words *
                sizeof(uint64_t)
        ) == 0);
        assert(
            gated_result.telemetry.anchors_constructed ==
            (uint64_t)on_count - expected_gated_anchors
        );
        assert(
            gated_result.telemetry.xor_words ==
            (expected_edges - expected_gated_edges) *
                (uint64_t)((ninputs + 63) / 64)
        );
        if (price_result.best_weight > 1.0 + 1e-8) {
            assert(gated_result.best_weight == price_result.best_weight);
            assert(gated_result.best_anchor == price_result.best_anchor);
            assert(
                gated_result.best_support_size ==
                price_result.best_support_size
            );
        }

        WeightedMMCSPriceOptions memory_options = {
            .requested_threads = 4,
            .worker_memory_budget_bytes =
                unbatched_serial.estimated_bytes_per_worker * 2u,
            .collect_timing = true
        };
        WeightedMMCSPriceResult memory_limited;
        int memory_limited_vertices[8];
        assert(
            bounded_mmcs_price_output_batch_prepared(
                &price_pi,
                &prepared,
                max_level,
                weights,
                memory_limited_vertices,
                NULL,
                &memory_options,
                &memory_limited
            ) == WEIGHTED_MMCS_PRICE_OK
        );
#if defined(HAVE_OPENMP) || defined(HAVE_PTHREAD)
        assert(memory_limited.effective_threads == 2);
#else
        assert(memory_limited.effective_threads == 1);
#endif
        assert(memory_limited.best_weight == unbatched_serial.best_weight);
        assert(memory_limited.best_anchor == unbatched_serial.best_anchor);
        assert(
            memory_limited.best_support_size ==
            unbatched_serial.best_support_size
        );
        assert(memcmp(
            memory_limited_vertices,
            unbatched_serial_vertices,
            (size_t)unbatched_serial.best_support_size * sizeof(int)
        ) == 0);
        assert(
            memory_limited.telemetry.edge_build_nanoseconds +
            memory_limited.telemetry.incidence_count_nanoseconds +
            memory_limited.telemetry.incidence_fill_nanoseconds +
            memory_limited.telemetry.search_nanoseconds > 0
        );
        bounded_mmcs_prepared_rows_destroy(&prepared);
        parallel_pricing_checked = true;
    }
    free(on_set);
    free(off_set);
    free(on_rows);
    free(off_rows);
}

static void verify_weighted_oracle_rejects_wildcards(void) {
    int on_set[] = {1, 0};
    int off_set[] = {2, 1};
    double weights[] = {1.0};
    int best_vertices[2];
    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 1;
    pi.OFF_minterms = 1;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;

    WeightedMMCSPriceResult result;
    assert(
        bounded_mmcs_price_output(
            &pi,
            2,
            2,
            weights,
            best_vertices,
            &result
        ) == WEIGHTED_MMCS_PRICE_UNSUPPORTED
    );
}

static void verify_reduced_cost_root_gate(void) {
    int on_set[] = {
        1, 1,
        2, 2
    };
    int off_set[] = {
        1, 2,
        2, 1
    };
    double weights[] = {2.0, 0.25};
    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 2;
    pi.OFF_minterms = 2;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.pichart_words = 1;
    pi.cov_bits = 64;

    WeightedMMCSPreparedRows prepared;
    assert(bounded_mmcs_prepare_binary_rows(&pi, 2, &prepared));

    double ungated_weights[2];
    int ungated_anchors[2];
    int ungated_support_sizes[2];
    int ungated_vertices[4];
    uint64_t ungated_coverage[2];
    WeightedMMCSPriceBatch ungated_batch = {
        .capacity = 2,
        .weights = ungated_weights,
        .anchors = ungated_anchors,
        .support_sizes = ungated_support_sizes,
        .vertices = ungated_vertices,
        .coverage = ungated_coverage
    };
    int ungated_best_vertices[2];
    WeightedMMCSPriceOptions ungated_options = {
        .requested_threads = 1
    };
    WeightedMMCSPriceResult ungated_result;
    assert(bounded_mmcs_price_output_batch_prepared(
        &pi,
        &prepared,
        2,
        weights,
        ungated_best_vertices,
        &ungated_batch,
        &ungated_options,
        &ungated_result
    ) == WEIGHTED_MMCS_PRICE_OK);

    double gated_weights[2];
    int gated_anchors[2];
    int gated_support_sizes[2];
    int gated_vertices[4];
    uint64_t gated_coverage[2];
    WeightedMMCSPriceBatch gated_batch = {
        .capacity = 2,
        .weights = gated_weights,
        .anchors = gated_anchors,
        .support_sizes = gated_support_sizes,
        .vertices = gated_vertices,
        .coverage = gated_coverage
    };
    int gated_best_vertices[2];
    WeightedMMCSPriceOptions gated_options = {
        .requested_threads = 1,
        .enable_root_gate = true
    };
    WeightedMMCSPriceResult gated_result;
    assert(bounded_mmcs_price_output_batch_prepared(
        &pi,
        &prepared,
        2,
        weights,
        gated_best_vertices,
        &gated_batch,
        &gated_options,
        &gated_result
    ) == WEIGHTED_MMCS_PRICE_OK);

    assert(gated_batch.count == ungated_batch.count);
    assert(memcmp(
        gated_batch.weights,
        ungated_batch.weights,
        (size_t)ungated_batch.count * sizeof(double)
    ) == 0);
    assert(memcmp(
        gated_batch.anchors,
        ungated_batch.anchors,
        (size_t)ungated_batch.count * sizeof(int)
    ) == 0);
    assert(memcmp(
        gated_batch.support_sizes,
        ungated_batch.support_sizes,
        (size_t)ungated_batch.count * sizeof(int)
    ) == 0);
    assert(memcmp(
        gated_batch.vertices,
        ungated_batch.vertices,
        (size_t)ungated_batch.count * 2u * sizeof(int)
    ) == 0);
    assert(memcmp(
        gated_batch.coverage,
        ungated_batch.coverage,
        (size_t)ungated_batch.count * sizeof(uint64_t)
    ) == 0);
    assert(gated_result.telemetry.anchors_presented == 2);
    assert(gated_result.telemetry.anchors_constructed == 1);
    assert(gated_result.telemetry.prospective_gated_anchors == 1);
    assert(gated_result.telemetry.prospective_gated_edges == 3);
    assert(gated_result.telemetry.prospective_gated_incidences == 4);
    assert(gated_result.anchor_sweep_complete);
    assert(gated_result.anchors_completed == 2);
    assert(gated_result.anchors_total == 2);
    assert(
        gated_result.max_weight_upper_bound ==
        gated_result.best_weight
    );

    WeightedMMCSPriceOptions partial_options = {
        .requested_threads = 2,
        .anchor_limit = 1
    };
    WeightedMMCSPriceResult partial_result;
    assert(bounded_mmcs_price_output_batch_prepared(
        &pi,
        &prepared,
        2,
        weights,
        gated_best_vertices,
        &gated_batch,
        &partial_options,
        &partial_result
    ) == WEIGHTED_MMCS_PRICE_OK);
    assert(!partial_result.anchor_sweep_complete);
    assert(partial_result.telemetry.anchors_presented == 1);
    assert(partial_result.effective_threads == 1);
    assert(partial_result.anchors_completed == 1);
    assert(partial_result.anchors_total == 2);
    assert(
        partial_result.max_weight_upper_bound >=
        partial_result.best_weight
    );

    WeightedMMCSPriceOptions deadline_options = {
        .requested_threads = 1,
        .deadline_nanoseconds = 1
    };
    WeightedMMCSPriceResult deadline_result;
    assert(bounded_mmcs_price_output_batch_prepared(
        &pi,
        &prepared,
        2,
        weights,
        gated_best_vertices,
        &gated_batch,
        &deadline_options,
        &deadline_result
    ) == WEIGHTED_MMCS_PRICE_LIMIT_REACHED);
    assert(deadline_result.limit_reached);
    assert(deadline_result.telemetry.anchors_constructed == 0);
    assert(gated_batch.count == 0);
    assert(deadline_result.anchors_completed == 0);
    assert(deadline_result.anchors_total == 2);
    assert(deadline_result.max_weight_upper_bound == 2.25);

    bounded_mmcs_prepared_rows_destroy(&prepared);
}

static void verify_random_functions(void) {
    for (int trial = 0; trial < 300; ++trial) {
        int ninputs = 2 + (int)(next_random() % 7u);
        int rows = 1 << ninputs;
        unsigned char *is_on = calloc((size_t)rows, sizeof(unsigned char));
        assert(is_on);

        int on_count = 0;
        for (int row = 0; row < rows; ++row) {
            is_on[row] = (unsigned char)((next_random() >> 31) & 1u);
            on_count += is_on[row] != 0;
        }
        if (on_count == 0) is_on[0] = 1;
        if (on_count == rows) is_on[rows - 1] = 0;

        verify_function(ninputs, is_on, true);
        free(is_on);
    }
}

static void verify_duplicate_on_rows(void) {
    PIstorage pi;
    memset(&pi, 0, sizeof(pi));

    int on_set[4] = {
        1, 1,
        1, 1
    };
    int off_set[2] = {2, 2};
    int word_index[2] = {0, 0};
    int bit_index[2] = {0, 2};
    uint64_t shifted_mask[2] = {
        UINT64_C(3),
        UINT64_C(3) << 2
    };

    pi.ON_minterms = 2;
    pi.OFF_minterms = 1;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.pichart_words = 1;
    pi.cov_bits = 64;

    BoundedMMCSStats stats;
    memset(&stats, 0, sizeof(stats));
    assert(bounded_mmcs_generate_output_all(
        &pi,
        2,
        word_index,
        bit_index,
        shifted_mask,
        1,
        &stats
    ));
    assert(pi.foundPI == 2);
    assert(pi.pichart_pos[0] == UINT64_C(3));
    assert(pi.pichart_pos[1] == UINT64_C(3));
    free_generated_arrays(&pi);
}

static void verify_exhaustive_small_functions(void) {
    unsigned char is_on[16];
    uint64_t functions = 0;

    for (int ninputs = 1; ninputs <= 4; ++ninputs) {
        int rows = 1 << ninputs;
        uint32_t function_count = UINT32_C(1) << rows;
        for (
            uint32_t function = 1;
            function + 1u < function_count;
            ++function
        ) {
            for (int row = 0; row < rows; ++row) {
                is_on[row] = (unsigned char)((function >> row) & 1u);
            }
            verify_function(ninputs, is_on, false);
            functions++;
        }
    }

    assert(functions == UINT64_C(65804));
}

static void verify_cross_output_sharing(void) {
    PIstorage outputs[3];
    memset(outputs, 0, sizeof(outputs));
    int level_start[3] = {0, 0, 0};

    for (int output = 0; output < 3; ++output) {
        outputs[output].foundPI = 1;
        outputs[output].implicants_pos = calloc(1, sizeof(uint64_t));
        outputs[output].implicants_val = calloc(1, sizeof(uint64_t));
        outputs[output].shared = calloc(1, sizeof(int));
        assert(
            outputs[output].implicants_pos &&
            outputs[output].implicants_val &&
            outputs[output].shared
        );
        outputs[output].implicants_pos[0] = UINT64_C(0x7);
        outputs[output].implicants_val[0] = output == 2
            ? UINT64_C(0x4)
            : UINT64_C(0x3);
    }

    int max_shared = 0;
    assert(bounded_mmcs_mark_level_sharing(
        outputs,
        3,
        level_start,
        1,
        &max_shared
    ));
    assert(outputs[0].shared[0] == 1);
    assert(outputs[1].shared[0] == 1);
    assert(outputs[2].shared[0] == 0);
    assert(max_shared == 1);

    for (int output = 0; output < 3; ++output) {
        free(outputs[output].implicants_pos);
        free(outputs[output].implicants_val);
        free(outputs[output].shared);
    }
}

static void verify_transactional_node_limit(void) {
    int on_set[] = {
        1, 1, 1
    };
    int off_set[] = {
        2, 1, 1,
        1, 2, 1,
        1, 1, 2
    };
    int word_index[] = {0, 0, 0};
    int bit_index[] = {0, 2, 4};
    uint64_t shifted_mask[] = {
        UINT64_C(3),
        UINT64_C(3) << 2,
        UINT64_C(3) << 4
    };

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 1;
    pi.OFF_minterms = 3;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.pichart_words = 1;
    pi.cov_bits = 64;

    BoundedMMCSStats limited_stats;
    memset(&limited_stats, 0, sizeof(limited_stats));
    assert(bounded_mmcs_generate_output_level_limited(
        &pi,
        3,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        1,
        &limited_stats
    ) == BOUNDED_MMCS_LIMIT_REACHED);
    assert(limited_stats.search_nodes == 1);
    assert(pi.foundPI == 0);
    assert(pi.implicants_pos == NULL);
    assert(pi.pichart_pos == NULL);

    BoundedMMCSStats complete_stats;
    memset(&complete_stats, 0, sizeof(complete_stats));
    assert(bounded_mmcs_generate_output_level_limited(
        &pi,
        3,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        0,
        &complete_stats
    ) == BOUNDED_MMCS_COMPLETE);
    assert(pi.foundPI == 1);
    assert(pi.implicants_pos[0] == UINT64_C(0x3f));
    assert(pi.implicants_val[0] == 0);
    free_generated_arrays(&pi);
    pi.foundPI = 0;
    pi.estimPI = 0;

    memset(&limited_stats, 0, sizeof(limited_stats));
    assert(bounded_mmcs_generate_output_all_limited(
        &pi,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        1,
        &limited_stats
    ) == BOUNDED_MMCS_LIMIT_REACHED);
    assert(limited_stats.search_nodes == 1);
    assert(pi.foundPI == 0);
    assert(pi.implicants_pos == NULL);
    assert(pi.pichart_pos == NULL);

    memset(&complete_stats, 0, sizeof(complete_stats));
    assert(bounded_mmcs_generate_output_all_limited(
        &pi,
        3,
        word_index,
        bit_index,
        shifted_mask,
        1,
        0,
        &complete_stats
    ) == BOUNDED_MMCS_COMPLETE);
    assert(pi.foundPI == 1);
    assert(pi.implicants_pos[0] == UINT64_C(0x3f));
    assert(pi.implicants_val[0] == 0);
    free_generated_arrays(&pi);
}

/*
 * For the second anchor, the OFF edge is {0} and the earlier-ON ownership
 * edge is {1}.  Their union has the minimal transversal {0,1}, but that cube
 * is not prime against OFF: literal 1 can be deleted.  The generator must
 * use the ownership edge only to select an anchor, never as a private edge
 * establishing primality.
 */
static void verify_ownership_edges_do_not_define_primality(void) {
    int on_set[] = {
        1, 2,
        1, 1
    };
    int off_set[] = {
        2, 1
    };
    uint64_t on_rows[] = {
        UINT64_C(2),
        UINT64_C(0)
    };
    uint64_t off_rows[] = {
        UINT64_C(1)
    };
    int word_index[] = {0, 0};
    int bit_index[] = {0, 2};
    uint64_t shifted_mask[] = {
        UINT64_C(3),
        UINT64_C(3) << 2
    };

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 2;
    pi.OFF_minterms = 1;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.pichart_words = 1;
    pi.cov_bits = 64;

    BoundedMMCSStats stats;
    memset(&stats, 0, sizeof(stats));
    assert(bounded_mmcs_generate_output_all(
        &pi,
        2,
        word_index,
        bit_index,
        shifted_mask,
        1,
        &stats
    ));
    assert(pi.foundPI == 2);

    for (int cube = 0; cube < pi.foundPI; ++cube) {
        uint64_t fixed = 0;
        uint64_t value = 0;
        for (int input = 0; input < 2; ++input) {
            if (pi.implicants_pos[cube] & shifted_mask[input]) {
                fixed |= UINT64_C(1) << input;
                if (
                    pi.implicants_val[cube] &
                    (UINT64_C(1) << bit_index[input])
                ) {
                    value |= UINT64_C(1) << input;
                }
            }
        }
        assert(popcount_u64(fixed) == 1);
        assert(brute_is_prime(
            on_rows,
            2,
            off_rows,
            1,
            fixed,
            value
        ));
    }

    free_generated_arrays(&pi);
}

/* More than 64 ON rows, zero/negative weights in returned coverage, and
 * full batches of tied prices. Compare every returned cube to the scalar
 * brute-force oracle as well as all three incidence backends. */
static void verify_tied_and_signed_weights(void) {
    unsigned char is_on[256];
    for (weight_profile = 1; weight_profile <= 2; ++weight_profile) {
        for (int trial = 0; trial < 16; ++trial) {
            for (int row = 0; row < 256; ++row)
                is_on[row] = (unsigned char)((row + trial * 3) % 5 != 0);
            verify_function(8, is_on, true);
        }
    }
    weight_profile = 0;
}

int main(void) {
#if defined(HAVE_OPENMP) || defined(HAVE_PTHREAD)
    assert(ccubes_effective_thread_count(192) == 192);
#else
    assert(ccubes_effective_thread_count(192) == 1);
#endif
    ccubes_set_thread_count(1);
    verify_random_functions();
    verify_tied_and_signed_weights();
    verify_exhaustive_small_functions();
    verify_duplicate_on_rows();
    assert(parallel_pricing_checked);
    verify_weighted_oracle_rejects_wildcards();
    verify_reduced_cost_root_gate();
    verify_cross_output_sharing();
    verify_transactional_node_limit();
    verify_ownership_edges_do_not_define_primality();
    puts("bounded MMCS regression: OK");
    return 0;
}
