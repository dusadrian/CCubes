#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "certified_stop.h"

static FILE *test_stream(void) {
    FILE *stream = tmpfile();
    assert(stream != NULL);
    return stream;
}

static bool stream_contains(FILE *stream, const char *needle) {
    assert(stream != NULL);
    assert(needle != NULL);
    fflush(stream);
    rewind(stream);

    char buffer[4096];
    size_t length = fread(buffer, 1, sizeof(buffer) - 1u, stream);
    buffer[length] = '\0';
    return strstr(buffer, needle) != NULL;
}

static void test_pool_avoids_tied_cover_warning(void) {
    int on_set[12] = {
        0, 0, 0,
        0, 0, 1,
        1, 1, 0,
        1, 1, 1
    };
    int off_set[3] = {0, 1, 1};
    uint64_t pichart_pos[4] = {
        0x3u, /* c0: rows 0,1 */
        0xCu, /* c1: rows 2,3 */
        0x9u, /* c2: rows 0,3 */
        0x6u  /* c3: rows 1,2 */
    };
    int warning_cover[2] = {0, 1};
    int safe_cover[2] = {2, 3};
    int *pool[2] = {warning_cover, safe_cover};
    int selected[2] = {0, 1};

    PIstorage pi;
    memset(&pi, 0, sizeof(pi));
    pi.ON_minterms = 4;
    pi.OFF_minterms = 1;
    pi.ON_set = on_set;
    pi.OFF_set = off_set;
    pi.foundPI = 4;
    pi.pichart_pos = pichart_pos;
    pi.pichart_words = 1;
    pi.cov_bits = 64;
    pi.solmin = 2;
    pi.pool_count = 2;
    pi.pool_solutions = pool;

    BlockingDiagnostic before;
    assert(certified_blocking_diagnostic(
        &pi, 3, 2, selected, 2, false, &before
    ));
    assert(before.delayed_private_pairs == 1);

    BlockingStopState blocking;
    certified_blocking_state_init(&blocking);
    FILE *stream = test_stream();
    assert(certified_blocking_observe_plateau(
        &blocking,
        false,
        true,
        true,
        stream,
        1,
        &pi,
        3,
        2,
        selected,
        2,
        true,
        true
    ));
    fclose(stream);

    assert(blocking.warning_detected);
    assert(blocking.pool_warning_avoided);
    assert(selected[0] == 2 && selected[1] == 3);
}

static void make_delayed_pair(
    PIstorage *pi,
    int ninputs,
    int *on_set,
    int *off_set,
    uint64_t *pichart_pos
) {
    memset(pi, 0, sizeof(*pi));
    memset(on_set, 0, (size_t)(2 * ninputs) * sizeof(*on_set));
    memset(off_set, 0, (size_t)ninputs * sizeof(*off_set));
    on_set[ninputs] = 1;   /* The ON rows disagree only at input 0. */
    off_set[1] = 1;        /* Their agreement supercube excludes this OFF. */
    pichart_pos[0] = 0x1u; /* c0: row 0 */
    pichart_pos[1] = 0x2u; /* c1: row 1 */
    pi->ON_minterms = 2;
    pi->OFF_minterms = 1;
    pi->ON_set = on_set;
    pi->OFF_set = off_set;
    pi->foundPI = 2;
    pi->pichart_pos = pichart_pos;
    pi->pichart_words = 1;
    pi->cov_bits = 64;
    pi->solmin = 2;
}

static void test_adaptive_warning_stops_without_legacy_escalation(void) {
    int on_set[10];
    int off_set[5];
    uint64_t pichart_pos[2];
    int selected[2] = {0, 1};
    PIstorage pi;
    make_delayed_pair(&pi, 5, on_set, off_set, pichart_pos);

    BlockingStopState blocking;
    certified_blocking_state_init(&blocking);
    FILE *stream = test_stream();
    assert(certified_blocking_observe_plateau(
        &blocking,
        false,
        true,
        true,
        stream,
        1,
        &pi,
        5,
        3,
        selected,
        2,
        true,
        false
    ));
    assert(blocking.warning_detected);
    assert(stream_contains(
        stream,
        "action=warn-stop reason=complete-certificate-required"
    ));
    fclose(stream);

    assert(certified_stop_policy_decision(
        NULL,
        false,
        true,
        3,
        2,
        true,
        NULL,
        1
    ));
}

static void test_complete_prime_chart_requires_exact_boundary(void) {
    CertifiedStopState certificate;
    certified_stop_state_reset(&certificate);

    assert(!certified_stop_should_stop(&certificate, 2, true));
    certified_stop_observe_complete_prime_chart(&certificate);
    assert(!certified_stop_should_stop(&certificate, 2, false));
    assert(certified_stop_should_stop(&certificate, 2, true));

    FILE *stream = test_stream();
    assert(!certified_stop_policy_decision(
        &(CertifiedStopState){0},
        true,
        false,
        1,
        2,
        true,
        stream,
        1
    ));
    assert(certified_stop_policy_decision(
        &certificate,
        true,
        false,
        1,
        2,
        true,
        stream,
        1
    ));
    fclose(stream);
}

static void test_one_cube_cardinality_floor_is_global(void) {
    FILE *stream = test_stream();
    assert(certified_stop_policy_decision(
        &(CertifiedStopState){0},
        true,
        false,
        1,
        1,
        false,
        stream,
        1
    ));
    assert(stream_contains(stream, "method=cardinality-floor"));
    fclose(stream);
}

int main(void) {
    test_pool_avoids_tied_cover_warning();
    test_adaptive_warning_stops_without_legacy_escalation();
    test_complete_prime_chart_requires_exact_boundary();
    test_one_cube_cardinality_floor_is_global();
    puts("complete-chart certification regression: OK");
    return 0;
}
