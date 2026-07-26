#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "plateau_probe.h"

static PIstorage make_fixture(bool compatible) {
    PIstorage pi = {0};
    pi.inputs = 4;
    pi.ON_minterms = 2;
    pi.OFF_minterms = compatible ? 2 : 3;
    pi.pichart_words = 1;
    pi.cov_bits = 64;
    pi.estimPI = 2;
    pi.foundPI = 2;

    const int on_rows[8] = {
        1, 1, 1, 1,
        1, 1, 2, 2
    };
    const int compatible_off[8] = {
        2, 1, 1, 1,
        1, 2, 1, 1
    };
    const int incompatible_off[12] = {
        2, 1, 1, 1,
        1, 2, 1, 1,
        1, 1, 1, 2
    };

    pi.ON_set = malloc(sizeof(on_rows));
    pi.OFF_set = malloc(
        (size_t)pi.OFF_minterms * 4u * sizeof(int)
    );
    pi.pichart_pos = calloc(2, sizeof(uint64_t));
    pi.implicants_pos = calloc(2, sizeof(uint64_t));
    pi.implicants_val = calloc(2, sizeof(uint64_t));
    pi.shared = calloc(2, sizeof(int));
    pi.covsum = calloc(2, sizeof(int));
    pi.covered = calloc(2, sizeof(int));
    pi.cov_word_index = calloc(2, sizeof(int));
    pi.shifted_cov_mask = calloc(2, sizeof(uint64_t));
    assert(
        pi.ON_set && pi.OFF_set && pi.pichart_pos &&
        pi.implicants_pos && pi.implicants_val &&
        pi.shared && pi.covsum && pi.covered &&
        pi.cov_word_index && pi.shifted_cov_mask
    );

    for (int i = 0; i < 8; ++i) pi.ON_set[i] = on_rows[i];
    const int *off = compatible ? compatible_off : incompatible_off;
    for (int i = 0; i < pi.OFF_minterms * 4; ++i) pi.OFF_set[i] = off[i];

    pi.pichart_pos[0] = UINT64_C(1);
    pi.pichart_pos[1] = UINT64_C(2);
    pi.implicants_pos[0] = UINT64_C(0xff);
    pi.implicants_pos[1] = UINT64_C(0xff);
    pi.implicants_val[1] = UINT64_C(0xa0);
    pi.covsum[0] = 1;
    pi.covsum[1] = 1;
    pi.shifted_cov_mask[0] = UINT64_C(1);
    pi.shifted_cov_mask[1] = UINT64_C(2);
    return pi;
}

static PIstorage make_guided_fixture(void) {
    PIstorage pi = {0};
    pi.inputs = 5;
    pi.ON_minterms = 4;
    pi.OFF_minterms = 3;
    pi.pichart_words = 1;
    pi.cov_bits = 64;
    pi.estimPI = 2;
    pi.foundPI = 2;

    /*
     * In the internal 1/2 row encoding, the witnesses agree on 1111-
     * (ordinary PLA notation 0000-).  The OFF rows induce blocker pairs
     * {0,1}, {1,2}, and {2,3}.  Plain forward/reverse deletion keeps
     * {1,3}/{0,2}, covering three ON rows.  Coverage guidance removes
     * coordinates 0 and 3 first and reaches -11-- (PLA -00--), covering all
     * four.
     */
    const int on_rows[20] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 2,
        2, 1, 1, 1, 1,
        1, 1, 1, 2, 1
    };
    const int off_rows[15] = {
        2, 2, 1, 1, 1,
        1, 2, 2, 1, 1,
        1, 1, 2, 2, 1
    };

    pi.ON_set = malloc(sizeof(on_rows));
    pi.OFF_set = malloc(sizeof(off_rows));
    pi.pichart_pos = calloc(2, sizeof(uint64_t));
    pi.implicants_pos = calloc(2, sizeof(uint64_t));
    pi.implicants_val = calloc(2, sizeof(uint64_t));
    pi.shared = calloc(2, sizeof(int));
    pi.covsum = calloc(2, sizeof(int));
    pi.covered = calloc(2, sizeof(int));
    pi.cov_word_index = calloc(4, sizeof(int));
    pi.shifted_cov_mask = calloc(4, sizeof(uint64_t));
    assert(
        pi.ON_set && pi.OFF_set && pi.pichart_pos &&
        pi.implicants_pos && pi.implicants_val &&
        pi.shared && pi.covsum && pi.covered &&
        pi.cov_word_index && pi.shifted_cov_mask
    );

    for (int i = 0; i < 20; ++i) pi.ON_set[i] = on_rows[i];
    for (int i = 0; i < 15; ++i) pi.OFF_set[i] = off_rows[i];

    pi.pichart_pos[0] = UINT64_C(0x5);
    pi.pichart_pos[1] = UINT64_C(0xa);
    pi.implicants_pos[0] = UINT64_C(0x3ff);
    pi.implicants_pos[1] = UINT64_C(0x3ff);
    pi.implicants_val[1] = UINT64_C(0x100);
    pi.covsum[0] = 2;
    pi.covsum[1] = 2;
    for (int r = 0; r < 4; ++r) {
        pi.shifted_cov_mask[r] = UINT64_C(1) << r;
    }
    return pi;
}

static void destroy_fixture(PIstorage *pi) {
    free(pi->ON_set);
    free(pi->OFF_set);
    free(pi->pichart_pos);
    free(pi->implicants_pos);
    free(pi->implicants_val);
    free(pi->shared);
    free(pi->covsum);
    free(pi->covered);
    free(pi->cov_word_index);
    free(pi->shifted_cov_mask);
}

static void test_appends_prime_from_private_pair(void) {
    PIstorage pi = make_fixture(true);
    const int selected[2] = {0, 1};
    const int bit_index[4] = {0, 2, 4, 6};
    const int word_index[4] = {0, 0, 0, 0};
    const uint64_t shifted_mask[4] = {
        UINT64_C(0x03),
        UINT64_C(0x0c),
        UINT64_C(0x30),
        UINT64_C(0xc0)
    };
    PlateauProbeStats stats;

    assert(plateau_probe_append_candidates(
        &pi,
        4,
        1,
        selected,
        2,
        bit_index,
        word_index,
        shifted_mask,
        16,
        4,
        &stats
    ));
    assert(stats.private_witnesses == 2);
    assert(stats.pairs_examined == 1);
    assert(stats.compatible_pairs == 1);
    assert(stats.candidates_generated == 2);
    assert(stats.candidates_appended == 1);
    assert(pi.foundPI == 3);

    /* The generated prime is 00-- in ordinary binary notation. */
    assert(pi.implicants_pos[2] == UINT64_C(0x0f));
    assert(pi.implicants_val[2] == UINT64_C(0));
    assert(pi.pichart_pos[2] == UINT64_C(3));
    assert(pi.covsum[2] == 2);

    /* Once retained, the joint cube suppresses the same pair on a later pass. */
    assert(plateau_probe_append_candidates(
        &pi,
        4,
        1,
        selected,
        2,
        bit_index,
        word_index,
        shifted_mask,
        16,
        4,
        &stats
    ));
    assert(stats.candidates_appended == 0);
    assert(pi.foundPI == 3);
    destroy_fixture(&pi);
}

static void test_rejects_off_intersecting_agreement_cube(void) {
    PIstorage pi = make_fixture(false);
    const int selected[2] = {0, 1};
    const int bit_index[4] = {0, 2, 4, 6};
    const int word_index[4] = {0, 0, 0, 0};
    const uint64_t shifted_mask[4] = {
        UINT64_C(0x03),
        UINT64_C(0x0c),
        UINT64_C(0x30),
        UINT64_C(0xc0)
    };
    PlateauProbeStats stats;

    assert(plateau_probe_append_candidates(
        &pi,
        4,
        1,
        selected,
        2,
        bit_index,
        word_index,
        shifted_mask,
        16,
        4,
        &stats
    ));
    assert(stats.compatible_pairs == 0);
    assert(stats.candidates_appended == 0);
    assert(pi.foundPI == 2);
    destroy_fixture(&pi);
}

static void test_guidance_prefers_larger_on_coverage(void) {
    PIstorage pi = make_guided_fixture();
    const uint64_t all_rows = UINT64_C(0x0f);
    const int selected[2] = {0, 1};
    const int bit_index[5] = {0, 2, 4, 6, 8};
    const int word_index[5] = {0, 0, 0, 0, 0};
    const uint64_t shifted_mask[5] = {
        UINT64_C(0x003),
        UINT64_C(0x00c),
        UINT64_C(0x030),
        UINT64_C(0x0c0),
        UINT64_C(0x300)
    };
    PlateauProbeStats stats;

    /*
     * Initially neither column covers the chart, while their union does:
     * the exact cover cardinality is therefore two.
     */
    assert(pi.pichart_pos[0] != all_rows);
    assert(pi.pichart_pos[1] != all_rows);
    assert((pi.pichart_pos[0] | pi.pichart_pos[1]) == all_rows);

    assert(plateau_probe_append_candidates(
        &pi,
        5,
        1,
        selected,
        2,
        bit_index,
        word_index,
        shifted_mask,
        16,
        4,
        &stats
    ));
    assert(stats.private_witnesses == 2);
    assert(stats.pairs_examined == 1);
    assert(stats.compatible_pairs == 1);
    assert(stats.candidates_generated == 2);
    assert(stats.candidates_appended == 1);
    assert(pi.foundPI == 3);

    /* Guided deletion reaches internal -11-- (PLA -00--) and covers all ON. */
    assert(pi.implicants_pos[2] == UINT64_C(0x03c));
    assert(pi.implicants_val[2] == UINT64_C(0));
    assert(pi.pichart_pos[2] == all_rows);
    assert(pi.covsum[2] == 4);
    /* The appended column alone covers the chart, lowering the bound to one. */
    destroy_fixture(&pi);
}

int main(void) {
    test_appends_prime_from_private_pair();
    test_rejects_off_intersecting_agreement_cube();
    test_guidance_prefers_larger_on_coverage();
    puts("plateau_probe tests passed");
    return 0;
}
