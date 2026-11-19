/* The coordinator applies results. The proof thread owns a private copy
 * of the care rows and never accesses MPI or the mutable PI chart. */
typedef struct {
    PIstorage care;
    int inputs, output, terms;
    double deadline;
    atomic_bool done;
    bool collected, proved;
    SystematicSatResult result;
} SystematicOverlapJob;

static struct {
    SystematicOverlapJob *jobs;
    pthread_t thread;
    int count, inputs, running;
    double seconds;
} systematic_overlap;

static double systematic_overlap_now(void) {
    return (double)ccubes_monotonic_nanoseconds() / 1e9;
}

static void systematic_overlap_open(int inputs, int outputs, double seconds) {
    if (seconds <= 0.0 || systematic_rank) return;
    systematic_overlap.jobs = calloc((size_t)outputs, sizeof(*systematic_overlap.jobs));
    systematic_require(systematic_overlap.jobs != NULL, "could not allocate SAT jobs");
    systematic_overlap.inputs = inputs;
    systematic_overlap.count = outputs;
    systematic_overlap.seconds = seconds;
    systematic_overlap.running = -1;
    for (int o = 0; o < outputs; ++o)
        atomic_init(&systematic_overlap.jobs[o].done, false);
}

static void *systematic_overlap_worker(void *arg) {
    SystematicOverlapJob *job = arg;
    job->result = systematic_sat_descent(&job->care, job->inputs, job->output,
                                        job->terms, job->deadline);
    free(job->care.ON_set);
    free(job->care.OFF_set);
    atomic_store_explicit(&job->done, true, memory_order_release);
    return NULL;
}

static int *systematic_overlap_rows(const int *rows, int count, int inputs) {
    if (!count) return NULL;
    systematic_require((size_t)count <= SIZE_MAX / sizeof(int) / (size_t)inputs,
                       "SAT care rows exceed allocation limits");
    size_t bytes = (size_t)count * (size_t)inputs * sizeof(int);
    int *copy = malloc(bytes);
    systematic_require(copy != NULL, "could not copy SAT care rows");
    memcpy(copy, rows, bytes);
    return copy;
}

static void systematic_overlap_poll(const PIstorage *pi, bool wait) {
    if (!systematic_overlap.jobs) return;
    int running = systematic_overlap.running;
    if (running >= 0) {
        SystematicOverlapJob *job = &systematic_overlap.jobs[running];
        if (!wait && !atomic_load_explicit(&job->done, memory_order_acquire)) return;
        systematic_require(pthread_join(systematic_overlap.thread, NULL) == 0,
                           "could not join the SAT worker");
        job->collected = true;
        systematic_overlap.running = -1;
    }
    int next = -1;
    for (int o = 0; o < systematic_overlap.count; ++o) {
        SystematicOverlapJob *job = &systematic_overlap.jobs[o];
        if (job->terms && !job->collected &&
            (next < 0 || job->deadline < systematic_overlap.jobs[next].deadline)) next = o;
    }
    if (next < 0) return;

    SystematicOverlapJob *job = &systematic_overlap.jobs[next];
    if (systematic_overlap_now() >= job->deadline) {
        job->collected = true;
        fprintf(stderr, "CCUBES_SYSTEMATIC_SAT_OVERLAP output=%d status=deadline "
                "start_cover=%d best=%d lower=1 queries=0\n", job->output, job->terms, job->terms);
        return;
    }
    job->care.ON_minterms = pi[next].ON_minterms;
    job->care.OFF_minterms = pi[next].OFF_minterms;
    job->care.ON_set = systematic_overlap_rows(pi[next].ON_set, pi[next].ON_minterms, job->inputs);
    job->care.OFF_set = systematic_overlap_rows(pi[next].OFF_set, pi[next].OFF_minterms, job->inputs);
    systematic_require(pthread_create(&systematic_overlap.thread, NULL,
                                     systematic_overlap_worker, job) == 0,
                       "could not start the SAT worker");
    systematic_overlap.running = next;
}

static bool systematic_overlap_queue(const PIstorage *pi, int output, int level) {
    if (!systematic_overlap.jobs || pi[output].solmin <= 1) return false;
    SystematicOverlapJob *job = &systematic_overlap.jobs[output];
    if (job->terms) return false;
    job->terms = pi[output].solmin;
    job->inputs = systematic_overlap.inputs;
    job->output = output + 1;
    job->deadline = systematic_overlap_now() + systematic_overlap.seconds;
    fprintf(stderr, "CCUBES_SYSTEMATIC_SAT_OVERLAP output=%d status=queued level=%d target=%d seconds=%.9g\n",
            job->output, level, job->terms - 1, systematic_overlap.seconds);
    systematic_overlap_poll(pi, false);
    return level < job->inputs && !job->collected;
}

static bool systematic_overlap_proved(const PIstorage *pi, int output) {
    if (!systematic_overlap.jobs) return false;
    const SystematicOverlapJob *job = &systematic_overlap.jobs[output];
    if (!job->collected) return false;
    const SystematicSatResult *result = &job->result;
    int best = pi->solmin;
    if (result->cover && result->terms < best) best = result->terms;
    return best > 0 && result->lower == best;
}

static bool systematic_overlap_all_active_proved(const PIstorage *pi) {
    if (!systematic_overlap.jobs) return false;
    bool active = false;
    for (int o = 0; o < systematic_overlap.count; ++o) {
        if (pi[o].stop_search) continue;
        if (!systematic_overlap_proved(&pi[o], o)) return false;
        active = true;
    }
    return active;
}

static void systematic_overlap_apply_stops(PIstorage *pi, int level) {
    if (!systematic_overlap.jobs) return;
    systematic_overlap_poll(pi, false);
    for (int o = 0; o < systematic_overlap.count; ++o) {
        if (pi[o].stop_search || !systematic_overlap_proved(&pi[o], o)) continue;
        pi[o].stop_search = true;
        fprintf(stderr, "CCUBES_SYSTEMATIC_SAT_OVERLAP output=%d status=stop-generation level=%d\n",
                o + 1, level);
    }
}

static void systematic_overlap_finish(PIstorage *pi) {
    if (!systematic_overlap.jobs) return;
    for (;;) {
        systematic_overlap_poll(pi, true);
        bool pending = false;
        for (int o = 0; o < systematic_overlap.count; ++o) {
            const SystematicOverlapJob *job = &systematic_overlap.jobs[o];
            pending |= job->terms && !job->collected;
        }
        if (!pending) return;
    }
}

static bool systematic_overlap_publish(PIstorage *pi, int output, int inputs) {
    if (!systematic_overlap.jobs) return true;
    SystematicOverlapJob *job = &systematic_overlap.jobs[output];
    job->proved = systematic_overlap_proved(pi, output);
    if (job->result.cover && job->result.terms < pi->solmin) {
        free(pi->solution);
        pi->solution = job->result.cover;
        job->result.cover = NULL;
        pi->solmin = job->result.terms;
        pi->pi_chart_optimal = false;
    }
    free(job->result.cover);
    job->result.cover = NULL;
    return !job->proved || systematic_validate_cover(pi, inputs);
}
