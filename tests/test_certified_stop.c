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

static void test_pool_avoids_tied_cover_warning(void) {
    int on_set[12] = {
        0, 0, 0,
        0, 0, 1,
        1, 1, 0,
        1, 1, 1
    };
    int off_set[3] = {0, 1, 1};
    int pichart[16] = {
        1, 1, 0, 0, /* c0 */
        0, 0, 1, 1, /* c1 */
        1, 0, 0, 1, /* c2 */
        0, 1, 1, 0  /* c3 */
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
    pi.pichart = pichart;
    pi.solmin = 2;
    pi.pool_count = 2;
    pi.pool_solutions = pool;

    BlockingDiagnostic before;
    assert(certified_blocking_diagnostic(
        &pi, 3, 2, selected, 2, false, &before
    ));
    assert(before.delayed_private_pairs == 1);

    CertifiedStopState certificate;
    BlockingStopState blocking;
    certified_stop_state_reset(&certificate);
    certified_blocking_state_init(&blocking);
    FILE *stream = test_stream();
    assert(certified_blocking_observe_plateau(
        &blocking,
        &certificate,
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
    assert(!blocking.certification_required);
    assert(!blocking.escalation_suppressed);
    assert(selected[0] == 2 && selected[1] == 3);
}

static void make_delayed_pair(
    PIstorage *pi,
    int ninputs,
    int *on_set,
    int *off_set,
    int *pichart
) {
    memset(pi, 0, sizeof(*pi));
    memset(on_set, 0, (size_t)(2 * ninputs) * sizeof(*on_set));
    memset(off_set, 0, (size_t)ninputs * sizeof(*off_set));
    on_set[ninputs] = 1; /* The two ON rows disagree only at input 0. */
    off_set[1] = 1;      /* Excludes the OFF row from their supercube. */
    pichart[0] = 1;
    pichart[1] = 0;
    pichart[2] = 0;
    pichart[3] = 1;
    pi->ON_minterms = 2;
    pi->OFF_minterms = 1;
    pi->ON_set = on_set;
    pi->OFF_set = off_set;
    pi->foundPI = 2;
    pi->pichart = pichart;
    pi->solmin = 2;
}

static void test_unaffordable_adaptive_warning_stops(void) {
    int on_set[60];
    int off_set[30];
    int pichart[4];
    int selected[2] = {0, 1};
    PIstorage pi;
    make_delayed_pair(&pi, 30, on_set, off_set, pichart);

    uint64_t estimate = 0;
    assert(!certified_stop_adaptive_work_within_limit(
        30,
        4,
        29,
        CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT,
        &estimate
    ));
    assert(estimate == CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT + 1u);

    CertifiedStopState certificate;
    BlockingStopState blocking;
    certified_stop_state_reset(&certificate);
    certified_stop_observe_coverage(&certificate, 2, true);
    certified_blocking_state_init(&blocking);
    FILE *stream = test_stream();
    assert(certified_blocking_observe_plateau(
        &blocking,
        &certificate,
        false,
        true,
        true,
        stream,
        1,
        &pi,
        30,
        4,
        selected,
        2,
        true,
        false
    ));
    fclose(stream);

    assert(blocking.warning_detected);
    assert(blocking.escalation_suppressed);
    assert(!blocking.certification_required);
    assert(blocking.certification_horizon == 29);
    assert(certified_stop_policy_decision(
        &certificate, &blocking, false, true, 4, 2, true, NULL, 1
    ));

    /* Explicit certified mode ignores the adaptive work guard. */
    assert(!certified_stop_policy_decision(
        &certificate, &blocking, true, true, 4, 2, true, NULL, 1
    ));
    assert(certified_stop_policy_decision(
        &certificate, &blocking, true, true, 29, 2, true, NULL, 1
    ));
}

static void test_affordable_warning_still_certifies(void) {
    int on_set[10];
    int off_set[5];
    int pichart[4];
    int selected[2] = {0, 1};
    PIstorage pi;
    make_delayed_pair(&pi, 5, on_set, off_set, pichart);

    uint64_t estimate = 0;
    assert(certified_stop_adaptive_work_within_limit(
        5, 3, 4, CCUBES_ADAPTIVE_CERTIFICATION_TASK_LIMIT, &estimate
    ));
    assert(estimate == 5);

    CertifiedStopState certificate;
    BlockingStopState blocking;
    certified_stop_state_reset(&certificate);
    certified_stop_observe_coverage(&certificate, 2, true);
    certified_blocking_state_init(&blocking);
    FILE *stream = test_stream();
    assert(certified_blocking_observe_plateau(
        &blocking,
        &certificate,
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
    fclose(stream);

    assert(blocking.certification_required);
    assert(!blocking.escalation_suppressed);
    assert(blocking.certification_horizon == 4);
    assert(!certified_stop_policy_decision(
        &certificate, &blocking, false, true, 3, 2, true, NULL, 1
    ));
    assert(certified_stop_policy_decision(
        &certificate, &blocking, false, true, 4, 2, true, NULL, 1
    ));

    BlockingStopState unproved = blocking;
    unproved.certification_required = true;
    unproved.escalation_suppressed = false;
    unproved.boundary_horizon_unproved = false;
    FILE *fallback_stream = test_stream();
    assert(certified_stop_policy_decision(
        &certificate,
        &unproved,
        false,
        true,
        4,
        2,
        false,
        fallback_stream,
        1
    ));
    fclose(fallback_stream);
    assert(!unproved.certification_required);
    assert(unproved.escalation_suppressed);
    assert(unproved.boundary_horizon_unproved);
}

int main(void) {
    test_pool_avoids_tied_cover_warning();
    test_unaffordable_adaptive_warning_stops();
    test_affordable_warning_still_certifies();
    puts("certified stopping regression: OK");
    return 0;
}
