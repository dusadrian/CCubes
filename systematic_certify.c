/* Copyright (c) 2016–2026, Adrian Dusa. See LICENSE. */

#include "systematic_certify.h"

#include "ccubes_threads.h"
#include "cover_encoding.h"
#include "ipc_weighted_cover.h"
#include "lagrangian.h"
#include "native_sat_support.h"
#include "progressive_dual_calibration.h"
#include "progressive_global_bound.h"

#include <inttypes.h>
#include <stdatomic.h>
#include <time.h>

#define NATIVE_MEMORY_LIMIT (128u * 1024u * 1024u)
#define SYSTEMATIC_MAX_WORKERS 12 /* At most 1.5 GiB of native query allocations. */

static double monotonic_seconds(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static bool expired(void *context)
{
    return monotonic_seconds() >= *(const double *)context;
}

static const char *stop_name(NsatStopReason stop)
{
    switch (stop) {
        case NSAT_STOP_WORK: return "work-limit";
        case NSAT_STOP_MEMORY: return "memory-limit";
        case NSAT_STOP_CANCELLED: return "deadline";
        case NSAT_STOP_INPUT: return "invalid-input";
        case NSAT_STOP_STATE: return "invalid-state";
        case NSAT_STOP_PROOF: return "proof-or-model-rejected";
        case NSAT_STOP_ALLOCATION: return "allocation-failed";
        case NSAT_STOP_CONFLICTS: return "conflict-limit";
        case NSAT_STOP_NONE: return "none";
    }
    return "unknown";
}

static bool cube_covers(const int *cube, const int *point, int inputs, int stride,
                        NsatBudget *budget)
{
    for (int i = 0; i < inputs; ++i) {
        if (!nsat_work(budget, 1)) return false;
        int value = cube[(size_t)i * stride];
        if (value && value != point[i]) return false;
    }
    return true;
}

static bool cover_valid(const PIstorage *pi, int inputs, const int *cover, int terms,
                        NsatBudget *budget)
{
    if (!terms) return pi->ON_minterms == 0;
    if (!cover || terms < 0) return false;
    for (int p = 0; p < pi->ON_minterms; ++p) {
        const int *point = pi->ON_set + (size_t)p * inputs;
        int term = 0;
        for (; term < terms; ++term) {
            if (cube_covers(cover + term, point, inputs, terms, budget)) break;
            if (budget->status != NSAT_OK) return false;
        }
        if (term == terms) return false;
    }
    for (int p = 0; p < pi->OFF_minterms; ++p) {
        const int *point = pi->OFF_set + (size_t)p * inputs;
        for (int term = 0; term < terms; ++term) {
            if (cube_covers(cover + term, point, inputs, terms, budget) ||
                budget->status != NSAT_OK) return false;
        }
    }
    return nsat_poll(budget);
}

typedef struct {
    int lower;
    const char *status;
    const char *deadline;
} IpcBound;

static IpcBound ipc_lower_bound(const PIstorage *pi, int inputs, int threads,
                               const double *deadline)
{
    IpcBound result = {.lower = 1, .status = "deadline", .deadline = "na"};
    if (deadline && monotonic_seconds() >= *deadline) return result;
    PIChartView chart = pi_chart_view(pi);
    double *seed = calloc((size_t)chart.rows, sizeof(*seed));
    double *weights = calloc((size_t)chart.rows, sizeof(*weights));
    result.status = "seed-unavailable";
    if (seed && weights && lagrangian_initialize_pricing_multipliers(&chart, seed, chart.rows)) {
        ProgressiveDualCalibrationStats calibration;
        bool calibrated = (!deadline || monotonic_seconds() < *deadline) && progressive_dual_calibrate(
            &chart, seed, 512, 128, NULL, 0, weights, &calibration
        );
        const double *row_weights = calibrated ? weights : seed;
        double pricing_seconds = deadline ? *deadline - monotonic_seconds() : 0.25;
        if (pricing_seconds > 0.25) pricing_seconds = 0.25;
        if (pricing_seconds > 0.0) {
            ProgressiveGlobalBoundResult bound;
            ProgressiveGlobalBoundStatus status = progressive_global_bound_compute(
                pi, inputs, row_weights, threads, UINT64_C(64) * 1024 * 1024,
                pricing_seconds, NULL, 0, NULL, NULL, &bound
            );
            result.status = progressive_global_bound_status_name(status);
            if (status == PROGRESSIVE_GLOBAL_BOUND_OK) {
                result.deadline = bound.deadline_reached ? "yes" : "no";
                if (bound.cardinality_lower_bound > result.lower)
                    result.lower = bound.cardinality_lower_bound;
            }
        } else {
            result.status = "deadline";
        }
    }
    free(seed);
    free(weights);
    return result;
}

/* Delete a literal only if each OFF point retains at least one mismatch. */
static bool expand_cover(const PIstorage *pi, int inputs, int terms, int *cover,
                         NsatBudget *budget)
{
    size_t off = (size_t)pi->OFF_minterms;
    int *mismatches = nsat_alloc(budget, off, sizeof(*mismatches));
    if (!mismatches) return false;
    bool complete = false;
    for (int t = 0; t < terms; ++t) {
        for (size_t z = 0; z < off; ++z) {
            const int *point = pi->OFF_set + z * inputs;
            int count = 0;
            for (int i = 0; i < inputs; ++i) {
                if (!nsat_work(budget, 1)) goto done;
                int value = cover[(size_t)i * terms + t];
                count += value && value != point[i];
            }
            mismatches[z] = count;
        }
        for (int i = 0; i < inputs; ++i) {
            if (!nsat_work(budget, 1)) goto done;
            int *value = cover + (size_t)i * terms + t;
            if (!*value) continue;
            size_t z = 0;
            for (; z < off; ++z) {
                if (!nsat_work(budget, 1)) goto done;
                if (mismatches[z] == 1 && pi->OFF_set[z * inputs + i] != *value) break;
            }
            if (z != off) continue;
            for (z = 0; z < off; ++z) {
                if (!nsat_work(budget, 1)) goto done;
                mismatches[z] -= pi->OFF_set[z * inputs + i] != *value;
            }
            *value = 0;
        }
    }
    complete = true;
done:
    nsat_free(budget, mismatches, off * sizeof(*mismatches));
    return complete;
}

static int *decode_cover(const NsatSolver *s, const PIstorage *pi, int inputs,
                         int target, NsatBudget *budget)
{
    size_t cells = (size_t)inputs * target; /* The encoder checked this product. */
    int *cover = nsat_alloc(budget, cells, sizeof(*cover));
    if (!cover) return NULL;
    for (int i = 0; i < inputs; ++i) {
        for (int t = 0; t < target; ++t) {
            if (!nsat_work(budget, 1)) goto invalid;
            bool zero = nsat_model_value(s, nsat_cover_selector(inputs, t, i, 1)) > 0;
            bool one = nsat_model_value(s, nsat_cover_selector(inputs, t, i, 2)) > 0;
            if (zero && one) goto invalid;
            cover[(size_t)i * target + t] = zero ? 1 : one ? 2 : 0;
        }
    }
    if (cover_valid(pi, inputs, cover, target, budget) &&
        expand_cover(pi, inputs, target, cover, budget) &&
        cover_valid(pi, inputs, cover, target, budget)) return cover;
invalid:
    nsat_stop(budget, NSAT_ERROR, NSAT_STOP_PROOF);
    nsat_free(budget, cover, cells * sizeof(*cover));
    return NULL;
}

static bool ipc_integer_bound(PIstorage *pi, int inputs, int output, int *lower,
                              const SystematicBudget *budget, double *shared_deadline,
                              NsatResources *resources)
{
    double started = monotonic_seconds();
    double seconds = budget->weighted_seconds;
    uint64_t cap = budget->weighted_work;
    if (shared_deadline) {
        /* Preserve the default 10:2 split and let SAT use any unspent time. */
        seconds = (*shared_deadline - started) * (5.0 / 6.0);
    }
    if (seconds <= 0.0) return false;
    uint64_t shared_cap = 0;
    if (resources->work_limit) {
        shared_cap = resources->work_limit - resources->work;
        if (shared_deadline || budget->sat_seconds > 0.0) shared_cap /= 2;
        if (!shared_cap) return false;
        if (!cap || shared_cap < cap) cap = shared_cap;
    }
    double integer_deadline = started + seconds;
    NsatResources phase = {.work_limit = budget->separate_verifier ? shared_cap : cap,
        .memory_limit = resources->memory_limit};
    NsatLimits limits = {.shared = &phase};
    int incumbent = pi->solmin;
    int initial_lower = *lower;
    NsatCoverInput query = {.on = pi->ON_set, .off = pi->OFF_set, .inputs = (size_t)inputs,
        .positives = (size_t)pi->ON_minterms, .negatives = (size_t)pi->OFF_minterms,
        .target = (size_t)incumbent};
    IpcWeightedCoverResult result;
    if (budget->separate_verifier) {
        const IpcWeightedCheckBudget check = {.search_seconds = seconds,
            .verifier_seconds = budget->verifier_seconds, .verifier_work = budget->verifier_work};
        limits.work = budget->weighted_work;
        result = ipc_weighted_cover_reprice_with_check_budget(&query, initial_lower,
                                                             limits, 4, 256, &check);
    } else {
        limits.poll = expired;
        limits.context = &integer_deadline;
        result = ipc_weighted_cover_reprice_with_auto_check(&query, initial_lower, limits, 4, 256);
    }
    resources->work += phase.work;
    uint64_t before_validation = resources->work;
    if (result.cover) {
        /* Check a cover retained by a limited search before opening the SAT clock. */
        NsatBudget validation = {.limits = {.shared = resources}};
        if (shared_deadline) {
            validation.limits.poll = expired;
            validation.limits.context = shared_deadline;
        }
        if (result.terms < result.lower || result.terms >= incumbent ||
            !cover_valid(pi, inputs, result.cover, result.terms, &validation) ||
            !expand_cover(pi, inputs, result.terms, result.cover, &validation) ||
            !cover_valid(pi, inputs, result.cover, result.terms, &validation)) {
            result.status = validation.status == NSAT_UNKNOWN ? NSAT_UNKNOWN : NSAT_ERROR;
            result.stop = validation.stats.stop ? validation.stats.stop : NSAT_STOP_PROOF;
        } else {
            free(pi->solution);
            pi->solution = result.cover;
            result.cover = NULL;
            pi->solmin = result.terms;
        }
        free(result.cover);
    }
    if (result.check_status == NSAT_UNSAT && result.lower <= pi->solmin)
        *lower = result.lower;
    bool certified = *lower == pi->solmin;
    const char *status = !result.supported ? "unsupported" :
        certified ? "certified" : result.status == NSAT_ERROR ? "error" :
        incumbent > pi->solmin ? "improved" : *lower > initial_lower ? "lower-bound" : "unknown";
    const char *check = result.check_status == NSAT_UNSAT ? "verified" :
        result.check_status == NSAT_OK ? "not-needed" :
        result.check_status == NSAT_UNKNOWN ? "limited" : "rejected";
    fprintf(stderr, "CCUBES_SYSTEMATIC_INTEGER output=%d status=%s stop_reason=%s "
            "incumbent=%d cover=%d lower=%d compatible=%" PRIu64 " maximal=%zu "
            "pivots=%u nodes=%" PRIu64 " work_used=%" PRIu64
            " check_status=%s check_work=%" PRIu64 " lower_target=%d repriced=%" PRIu64
            " elapsed_seconds=%.6f search_work=%" PRIu64 " check_seconds=%.6f"
            " check_auto_base=%" PRIu64 " check_auto_limit=%" PRIu64
            " check_auto_limit_hit=%s\n",
            output, status, stop_name(result.stop), incumbent, pi->solmin, *lower,
            result.compatible, result.maximal, result.pivots, result.nodes,
            phase.work + (resources->work - before_validation), check, result.check.work,
            result.lower_target, result.repriced,
            monotonic_seconds() - started, phase.work - result.check.work, result.check_seconds,
            result.check_auto_base, result.check_auto_limit, result.check_auto_limit_hit ? "yes" : "no");
    return certified;
}

typedef struct {
    NsatSolver *solver;
    NsatVerifier *checker;
} CoverSession;

static void close_session(CoverSession *session)
{
    nsat_destroy(session->solver);
    nsat_verify_destroy(session->checker);
    *session = (CoverSession){0};
}

static NsatStatus sat_target(PIstorage *pi, int inputs, int output, int target,
                              unsigned query, NsatLimits limits, NsatCoverStrengthening mode,
                              CoverSession *session)
{
    NsatCoverInput input = {.on = pi->ON_set, .off = pi->OFF_set,
        .inputs = (size_t)inputs, .positives = (size_t)pi->ON_minterms,
        .negatives = (size_t)pi->OFF_minterms, .target = (size_t)target};
    uint64_t start = limits.shared->work;
    limits.shared->peak_memory = limits.shared->memory;
    double started = monotonic_seconds();
    int incumbent = pi->solmin;
    NsatVerifier *v = session->checker;
    NsatSolver *s = session->solver;
    bool reused = s != NULL;
    NsatStats previous_checker = v ? nsat_verify_stats(v) : (NsatStats){0};
    NsatSearchStats previous_search = s ? nsat_search_stats(s) : (NsatSearchStats){0};
    NsatStopReason stop = NSAT_STOP_NONE;
    NsatCoverDomainStats domain = {0};
    NsatStatus status = reused ? nsat_solve_tighter_cover(s, v, (size_t)target)
                               : nsat_cover_solve(&input, mode, limits, &s, &v, &stop, &domain);
    session->checker = v;
    session->solver = s;
    bool formula_ready = s != NULL;
    if ((status == NSAT_SAT || status == NSAT_UNSAT) &&
        nsat_verify_cover_target(v) != (size_t)target) {
        status = NSAT_ERROR;
        stop = NSAT_STOP_STATE;
    }
    if (status == NSAT_SAT) {
        NsatBudget decode = {.limits = limits};
        int *cover = decode_cover(s, pi, inputs, target, &decode);
        if (cover) {
            free(pi->solution);
            pi->solution = cover;
            pi->solmin = target;
            nsat_release_budget(&decode); /* Ownership passed to the incumbent. */
        } else {
            status = decode.status;
            stop = decode.stats.stop;
        }
    }
    if (!stop) stop = limits.shared->stop;
    if (!stop && status == NSAT_ERROR) stop = NSAT_STOP_INPUT;

    const char *name = status == NSAT_UNSAT ? "unsat-verified" :
        status == NSAT_SAT ? "sat" : status == NSAT_UNKNOWN ? "unknown" : "error";
    NsatSearchStats search = s ? nsat_search_stats(s) : (NsatSearchStats){0};
    NsatStats checker = v ? nsat_verify_stats(v) : (NsatStats){0};
    char vars[32] = "na", clauses[32] = "na", slots[32] = "na";
    char work_limit[32] = "unlimited";
    if (limits.shared->work_limit)
        snprintf(work_limit, sizeof(work_limit), "%" PRIu64, limits.shared->work_limit);
    if (formula_ready) {
        snprintf(vars, sizeof(vars), "%d", nsat_verify_variables(v));
        snprintf(clauses, sizeof(clauses), "%" PRIu64, nsat_verify_original_count(v));
        snprintf(slots, sizeof(slots), "%zu", nsat_verify_cover_slots(v));
    }
    fprintf(stderr, "CCUBES_SYSTEMATIC_SAT schema_ver=1 output=%d query_id=%d.%u "
            "target=%d status=%s stop_reason=%s incumbent=%d encoding_ver=%d work_ver=%d "
            "vars=%s orig_clauses=%s workers=1 conflicts=%" PRIu64
            " lemmas_checked=%" PRIu64 " deleted=%" PRIu64 " compactions=%" PRIu64
            " work_used=%" PRIu64 " output_work_used=%" PRIu64 " work_limit=%s"
            " peak_memory_bytes=%zu memory_limit_bytes=%zu elapsed_seconds=%.6f"
            " strengthening=%d pairs=%zu triples=%zu staircase=%zu anchors=%zu"
            " domain_work=%" PRIu64 " discovery_capped=%s reused=%s formula_slots=%s\n",
            output, output, query, target, name, stop_name(stop), incumbent,
            NSAT_COVER_ENCODING_VERSION, NSAT_WORK_VERSION, vars, clauses,
            search.conflicts - previous_search.conflicts,
            checker.lemmas_checked - previous_checker.lemmas_checked,
            search.deleted - previous_search.deleted, search.compactions - previous_search.compactions,
            limits.shared->work - start, limits.shared->work, work_limit,
            limits.shared->peak_memory, limits.shared->memory_limit,
            monotonic_seconds() - started, mode, domain.pairs, domain.triples,
            domain.staircase, domain.anchors, domain.prep_work,
            domain.discovery_capped ? "yes" : "no", reused ? "yes" : "no", slots);
    return status;
}

static bool systematic_certify_cover(PIstorage *pi, int inputs, int output,
                                     int threads, const SystematicBudget *budget,
                                     NsatCoverStrengthening mode, bool reuse)
{
    const bool shared = budget->shared_seconds > 0.0;
    double deadline = 0.0;
    double *pricing_deadline = NULL;
    if (shared) {
        deadline = monotonic_seconds() + budget->shared_seconds;
        pricing_deadline = &deadline;
    }
    const bool floor = pi->solmin <= 1;
    NsatBudget validation = {0};
    if (shared)
        validation.limits = (NsatLimits){.poll = expired, .context = pricing_deadline};
    if (!cover_valid(pi, inputs, pi->solution, pi->solmin, &validation)) {
        fprintf(stderr, "CCUBES_SYSTEMATIC_CERTIFICATE output=%d status=%s stop_reason=%s\n",
                output, validation.status == NSAT_UNKNOWN ? "unproved" : "invalid-cover",
                validation.status == NSAT_UNKNOWN ? stop_name(validation.stats.stop) : "invalid-cover");
        return false;
    }
    IpcBound bound = floor
        ? (IpcBound){.lower = pi->solmin, .status = "bypassed", .deadline = "na"}
        : ipc_lower_bound(pi, inputs, threads, pricing_deadline);
    fprintf(stderr, "CCUBES_SYSTEMATIC_BOUND output=%d cover=%d lower=%d "
            "ipc_status=%s ipc_deadline_reached=%s\n",
            output, pi->solmin, bound.lower, bound.status, bound.deadline);
    const char *method = floor ? "cardinality-floor" : "ipc-global-bound";
    bool certified = bound.lower == pi->solmin;
    NsatResources resources = {.work_limit = budget->work, .memory_limit = NATIVE_MEMORY_LIMIT};
    unsigned query = 0, decisive = 0;
    int proof_target = 0;
    uint64_t proof_work = 0;
    if (!certified) {
        const int prior_lower = bound.lower;
        certified = ipc_integer_bound(pi, inputs, output, &bound.lower, budget,
                                      shared ? &deadline : NULL, &resources);
        if (bound.lower > prior_lower) method = "ipc-weighted-integer";
    }
    if (!certified && (shared ? !expired(&deadline) : budget->sat_seconds > 0.0)) {
        if (!shared) deadline = monotonic_seconds() + budget->sat_seconds;
        resources.stop = NSAT_STOP_NONE;
        NsatLimits limits = {.poll = expired, .context = &deadline, .shared = &resources};
        CoverSession session = {0};
        while (pi->solmin > bound.lower) {
            int target = pi->solmin - 1;
            uint64_t before = resources.work;
            NsatStatus status = sat_target(pi, inputs, output, target, ++query, limits, mode, &session);
            if (!reuse || status != NSAT_SAT) close_session(&session);
            if (status == NSAT_UNSAT) {
                certified = true;
                method = "native-rup";
                decisive = query;
                proof_target = target;
                proof_work = resources.work - before;
                break;
            }
            if (status != NSAT_SAT) break;
            if (pi->solmin == bound.lower) {
                certified = true;
                if (pi->solmin == 1) method = "cardinality-floor";
                break;
            }
        }
        close_session(&session);
    }
    char target[32] = "na", query_id[32] = "na", work[32] = "na", limit[32] = "unlimited";
    if (decisive) {
        snprintf(target, sizeof(target), "%d", proof_target);
        snprintf(query_id, sizeof(query_id), "%d.%u", output, decisive);
    }
    if (decisive) snprintf(work, sizeof(work), "%" PRIu64, proof_work);
    if (budget->work) snprintf(limit, sizeof(limit), "%" PRIu64, budget->work);
    const char *stop = !certified && shared && expired(&deadline) ? "deadline" : "na";
    fprintf(stderr, "CCUBES_SYSTEMATIC_CERTIFICATE output=%d scope=scalar cover=%d "
            "status=%s method=%s schema_ver=1 target_k=%s query_id=%s work_used=%s work_limit=%s"
            " lower=%d stop_reason=%s\n",
            output, pi->solmin, certified ? "certified" : "unproved",
            certified ? method : "none", target, query_id, work, limit,
            certified ? pi->solmin : bound.lower, stop);
    return certified;
}

typedef struct {
    PIstorage *outputs;
    int inputs, count, pricing_threads, workers;
    const SystematicBudget *budget;
    NsatCoverStrengthening mode;
    bool reuse;
    atomic_uint next, failed;
} CertificationBatch;

static void certify_range(uint64_t start, uint64_t end, uint64_t stride,
                          int worker, int workers, void *data)
{
    /* Pull the next output when a job finishes. Query times vary widely. */
    (void)start;
    (void)end;
    (void)stride;
    CertificationBatch *batch = data;
    if (worker == 0) batch->workers = workers;
    for (;;) {
        unsigned o = atomic_fetch_add_explicit(&batch->next, 1, memory_order_relaxed);
        if (o >= (unsigned)batch->count) return;
        if (!systematic_certify_cover(&batch->outputs[o], batch->inputs, (int)o + 1,
                                     batch->pricing_threads, batch->budget,
                                     batch->mode, batch->reuse))
            atomic_fetch_add_explicit(&batch->failed, 1, memory_order_relaxed);
    }
}

bool systematic_certify_outputs(PIstorage *pi, int inputs, int noutputs,
                                int threads, const SystematicBudget *budget,
                                NsatCoverStrengthening mode, bool reuse)
{
    int requested = ccubes_effective_thread_count(threads);
    int count = requested;
    if (count > SYSTEMATIC_MAX_WORKERS) count = SYSTEMATIC_MAX_WORKERS;
    if (count > noutputs) count = noutputs;
    CertificationBatch batch = {.outputs = pi, .inputs = inputs, .count = noutputs,
        .pricing_threads = noutputs == 1 ? requested : 1,
        .budget = budget, .mode = mode, .reuse = reuse};
    atomic_init(&batch.next, 0);
    atomic_init(&batch.failed, 0);
    double started = monotonic_seconds();
    bool launched = ccubes_parallel_for(0, (uint64_t)count, count, false, certify_range, &batch);
    unsigned failures = launched ? atomic_load_explicit(&batch.failed, memory_order_relaxed)
                                 : (unsigned)noutputs;
    fprintf(stderr, "CCUBES_SYSTEMATIC_BATCH outputs=%d requested_threads=%d workers=%d "
            "pricing_threads=%d failed=%u elapsed_seconds=%.6f\n",
            noutputs, requested, batch.workers, batch.pricing_threads, failures,
            monotonic_seconds() - started);
    return launched && !failures;
}
